/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file lower_l2_tensor_collectives_pass.cpp
 * @brief Lower managed ``pld.tensor.*`` collectives written in a CHIP/L2
 *        orchestration body into one local builtin AIV task.
 *
 * The HOST/L3 rail (``LowerHostTensorCollectives``) fans a collective out into
 * one ``builtin.tensor.*`` chip dispatch per device; each dispatch is a whole
 * extra L2 orchestration task whose only job is to submit one AIV kernel. When
 * the collective sits between two compute steps, the sequence costs three
 * separate L3 -> L2 round trips.
 *
 * This pass is the CHIP/L2 rail. It rewrites
 *
 *     target = pld.tensor.all_to_all_v(input, target, signal,
 *                                      send_counts, recv_counts, core_num=L)
 *
 * into a call to a synthesized AIV kernel function backed by the *same*
 * hand-written builtin kernel source the HOST rail uses:
 *
 *     target = __builtin_all_to_all_v__fp32(input, target, signal,
 *                                           send_counts, recv_counts)
 *
 * ``L`` must be a compile-time constant. The pass folds the admitted block count
 * ``B = CalAllToAllVBlocks(P, L)`` (``P`` is the signal's static ``NR``) and, for
 * ``B > 1``, attaches it to the call as the ``core_num`` / ``sync_start`` launch
 * spec — call attrs, not kernel parameters, so the ABI below is unchanged.
 *
 * The result is one ``rt_submit_aiv_task`` inside the caller's own pipeline —
 * no nested L2 -> L2 dispatch — and the collective participates in the normal
 * TensorMap task DAG through its parameter directions, so
 * ``compute -> collective -> consume`` is ordered by real data dependencies.
 *
 * The synthesized function carries no DSL body. Its source is named indirectly
 * by ``kAttrBuiltinTemplateDir`` / ``kAttrBuiltinTemplateVars``; the backend
 * renders the template into the chip sub-build and lists it in the generated
 * ``kernel_config.py``, the same way it handles an ``external_source`` kernel.
 *
 * ABI of the synthesized kernel — identical to the HOST builtin ABI up to the
 * arguments the kernel reads, which is what lets both rails share one source:
 *
 *     args[0..4]  input, target, signal, send_counts, recv_counts (Tensor*)
 *     args[5]     CommContext*
 *     args[6..7]  unread duplicates of args[5] — MaterializeDistTensorCtx
 *                 appends one CommCtx parameter per DistributedTensor
 *                 parameter, and the canonical signature declares exactly
 *                 three of those (target, signal, recv_counts; input and
 *                 send_counts are plain Tensor whatever the call site passes).
 *                 All three resolve to the same device_ctx, since every
 *                 operand of one collective belongs to one comm domain.
 *
 * The rank count comes from ``CommContext::rankNum``, not from a scalar
 * argument: the CHIP orchestration cannot compute it (``pld.system.nranks``
 * has no orchestration codegen), and the context is derived per comm domain,
 * so its ``rankNum`` is exactly the ``domain_size`` the HOST dispatch used to
 * pass. The HOST rail therefore drops that scalar too, and the two argument
 * layouts coincide.
 *
 * Runs immediately before ``DeriveCallDirections`` so the emitted call gets its
 * argument directions and task dependencies derived like any other kernel call.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

using op_predicates::IsManagedTensorCollective;

/// Continuous and indexed all-to-all-v share the managed task lowering.
constexpr const char* kAllToAllV = "pld.tensor.all_to_all_v";

/// A CHIP/L2 orchestration body. HOST orchestrators are excluded by level —
/// they keep their own fan-out rail.
///
/// `Graph` is orchestration-like and also derives `Level::CHIP`, so a
/// collective written in a Graph body reaches this rail as well. Matching only
/// `FunctionType::Orchestration` would leave it unlowered — LowerCompositeOps
/// already defers it, because a Graph body derives `Role::Orchestrator` — and
/// the post-condition check below would then reject a legal program.
[[nodiscard]] bool IsChipOrch(const FunctionPtr& func) {
  return func && IsOrchestrationLike(func->func_type_) && func->level_.has_value() &&
         *func->level_ == Level::CHIP;
}

/// The admitted block count ``B`` for a requested limit ``L`` over ``P`` ranks
/// (RFC #2521 frozen item #3): ``L`` while ``L < P``, otherwise the largest
/// multiple of ``P`` not exceeding ``L``. Payload size never enters.
///
/// This is the compiler-side copy of the formula. The HOST rail evaluates it in
/// its generated entry (``entry.cpp.in``), which is compiled standalone and cannot
/// link against this file, so the two sites are kept in step by tests that run
/// the RFC's worked-example table through both.
[[nodiscard]] int64_t CalAllToAllVBlocks(int64_t p, int64_t l) { return l < p ? l : (l / p) * p; }

/// Variant suffix and C++ element type the builtin kernel template is
/// instantiated with.
///
/// Both rails use DataType's string conversions to render the same template.
/// FP32 and INT8 are the current pair; INT8 is the RFC #2521 A1 canonical payload.
struct BuiltinDType {
  std::string suffix;
  std::string cpp;
};

[[nodiscard]] BuiltinDType ResolveBuiltinDType(const DataType& dtype, const Span& span) {
  CHECK_SPAN(dtype == DataType::FP32 || dtype == DataType::INT8, span)
      << "managed pld.tensor.all_to_all_v currently supports only dtype=FP32 or INT8, got "
      << dtype.ToString();
  return {dtype.ToString(), dtype.ToCTypeString()};
}

/// Encode the template substitutions as a ``k=v`` list for the Function attr.
///
/// A flat string keeps the attr payload to a type the printer, serializer and
/// nanobind attr binding already handle; the backend splits it back apart.
[[nodiscard]] std::string EncodeTemplateVars(const std::vector<std::pair<std::string, std::string>>& vars) {
  std::string encoded;
  for (const auto& [key, value] : vars) {
    if (!encoded.empty()) encoded += ",";
    encoded += key + "=" + value;
  }
  return encoded;
}

/// Per-collective description of the synthesized AIV kernel.
struct BuiltinKernelSpec {
  std::string function_name;
  std::string template_dir;
  std::string template_vars;
  std::vector<ParamDirection> param_directions;
  std::vector<const char*> param_names;
  /// Whether each parameter is declared `DistributedTensor` in the synthesized
  /// signature, independent of what the call site happens to pass.
  std::vector<bool> param_distributed;
};

[[nodiscard]] BuiltinKernelSpec MakeAllToAllVKernelSpec(const DataType& dtype, const Span& span) {
  const auto element = ResolveBuiltinDType(dtype, span);
  // The only substitution either rail makes: both reach the same argument
  // layout, so the rendered kernel source is byte-identical between them.
  auto template_vars = EncodeTemplateVars({{"dtype_cpp", element.cpp}});
  // Read the template package off the builtin op rather than repeating the
  // string, so both rails render the same source by construction.
  const auto& template_dir =
      OpRegistry::GetInstance().GetEntry("builtin.tensor.all_to_all_v").GetTemplateDir();
  INTERNAL_CHECK_SPAN(template_dir.has_value(), span)
      << "LowerL2TensorCollectives: builtin.tensor.all_to_all_v declares no template_dir";
  return BuiltinKernelSpec{
      std::string("__builtin_all_to_all_v__") + element.suffix,
      *template_dir,
      std::move(template_vars),
      {ParamDirection::In, ParamDirection::InOut, ParamDirection::InOut, ParamDirection::In,
       ParamDirection::InOut},
      {"input", "target", "signal", "send_counts", "recv_counts"},
      // `input` stays Tensor-like on the public op (plain Tensor or
      // DistributedTensor — the kernel only reads it locally as the TPUT
      // source). `send_counts` is required to be window-bound at the call site
      // on this rail (peers pull it through CommRemotePtr; see LowerCollective),
      // but BOTH are declared as plain `Tensor*` here: the synthesized
      // signature fixes the ABI and the variant stays keyed on dtype alone. The
      // three genuinely window-bound operands stay distributed, which is also
      // what gives MaterializeDistTensorCtx the CommCtx parameters the kernel
      // needs.
      //
      // The public op's trailing `core_num` arg is deliberately NOT a kernel
      // parameter. This rail's kernel args come from the managed pipeline's own
      // dispatch (see LowerCollective), where every parameter becomes an
      // `add_scalar` slot *ahead* of the CommCtx suffix that
      // MaterializeDistTensorCtx appends, and the shared `kernel.cpp.in` reads
      // its CommContext from `args[5]` — the first ctx. Making `core_num` part
      // of the kernel ABI pushed every ctx one slot to the right, so the kernel
      // reinterpreted the integer `1` as a `CommContext*` and derived a garbage
      // rankNum/rankId on every rank (one AIV block then spins in
      // `ExchangeBarrier` until the scheduler's watchdog fires). Kernel ABI
      // parity with the HOST rail is five operands plus the ctx, nothing more.
      {false, true, true, false, true},
  };
}

/// The parameter type the synthesized signature declares, which is deliberately
/// *not* whatever the call site passed.
///
/// `input` and `send_counts` are both rendered as a flat `Tensor*` — the two
/// kinds are indistinguishable to the kernel — and `send_counts` is additionally
/// pinned to a window-bound DistributedTensor at the call site (LowerCollective),
/// so the canonical declaration cannot depend on the call. Copying the first
/// call's types would make one cached function
/// carry an accidental ABI: a later call passing the other kind would inherit a
/// signature whose CommCtx parameter count no longer matches its arguments,
/// because MaterializeDistTensorCtx appends one per DistributedTensor
/// parameter. Declaring a canonical type instead keeps the variant keyed on
/// dtype alone, which is the only thing that actually changes the kernel.
[[nodiscard]] TypePtr CanonicalParamType(bool distributed, const ExprPtr& arg, const Span& span) {
  auto arg_type = AsTensorTypeLike(arg->GetType());
  INTERNAL_CHECK_SPAN(arg_type, span) << "LowerL2TensorCollectives: collective operand must be Tensor-like";
  if (distributed) {
    auto dist = As<DistributedTensorType>(arg->GetType());
    INTERNAL_CHECK_SPAN(dist, span)
        << "LowerL2TensorCollectives: operand must be a DistributedTensor (deducer-enforced)";
    return dist;
  }
  return std::make_shared<TensorType>(arg_type->shape_, arg_type->dtype_);
}

/// Build the header-only AIV function that stands in for the builtin kernel.
///
/// The signature covers the five kernel operands only. `core_num` is the public
/// op's 6th argument, but it stops at this rail's gate (LowerCollective): every
/// parameter here becomes a dispatch slot ahead of the CommCtx suffix
/// MaterializeDistTensorCtx appends, and the shared `kernel.cpp.in` reads its
/// CommContext from the fixed index `args[5]`.
[[nodiscard]] FunctionPtr MakeBuiltinKernelFunction(const BuiltinKernelSpec& spec, const CallPtr& call) {
  std::vector<VarPtr> params;
  params.reserve(spec.param_names.size());
  for (size_t i = 0; i < spec.param_names.size(); ++i) {
    params.push_back(std::make_shared<Var>(
        spec.param_names[i], CanonicalParamType(spec.param_distributed[i], call->args_[i], call->span_),
        call->span_));
  }
  // The builtin kernel addresses every operand as dense row-major from its view
  // origin. LowerCollective proves that at compile time; this attr asks
  // orchestration codegen for a runtime backstop.
  std::vector<std::pair<std::string, std::any>> attrs = {
      {kAttrBuiltinTemplateDir, spec.template_dir},
      {kAttrBuiltinTemplateVars, spec.template_vars},
      {kAttrRequireContiguousOperands, true},
  };
  // Window-as-result: the kernel returns `target`, matching the public op's
  // result so the call site stays a plain rebind.
  //
  // The body is that one ReturnStmt rather than an empty header. It is never
  // compiled — the backend renders the builtin template instead — but it keeps
  // the function honest for the passes that still read it: ReturnParamsExplicit
  // holds, and MaterializeDistTensorCtx can resolve the returned
  // DistributedTensor back to the parameter it writes.
  std::vector<TypePtr> return_types = {params[1]->GetType()};
  StmtPtr body = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{params[1]}, call->span_);
  return std::make_shared<Function>(spec.function_name, std::move(params), spec.param_directions,
                                    std::move(return_types), body, call->span_, FunctionType::AIV,
                                    std::nullopt, std::nullopt, std::move(attrs));
}

/// Every value that can flow into each Var of an orchestration body: AssignStmt
/// values, a loop-carried value's initial and yielded values, the yielded values
/// bound to If return vars, and a loop return var's carry. Lets an operand be
/// traced back to the definitions that fix its runtime layout. One walk; lookups
/// are hash-map hits.
class ValueSourceIndex : public IRVisitor {
 public:
  std::unordered_map<const Var*, std::vector<ExprPtr>> sources;

  void VisitStmt_(const AssignStmtPtr& op) override {
    sources[op->var_.get()].push_back(op->value_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    AddYields(op->return_vars_, op->then_body_);
    if (op->else_body_.has_value()) AddYields(op->return_vars_, *op->else_body_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    AddLoop(op->iter_args_, op->return_vars_, op->body_, op->span_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    AddLoop(op->iter_args_, op->return_vars_, op->body_, op->span_);
    IRVisitor::VisitStmt_(op);
  }

 private:
  template <typename VarPtrT>
  void AddYields(const std::vector<VarPtrT>& vars, const StmtPtr& body) {
    if (vars.empty()) return;
    auto yield = transform_utils::GetLastYieldStmt(body);
    INTERNAL_CHECK(yield && yield->value_.size() == vars.size())
        << "Internal error: a control-flow body binding " << vars.size()
        << " values must end in a matching yield";
    for (size_t i = 0; i < vars.size(); ++i) sources[vars[i].get()].push_back(yield->value_[i]);
  }

  void AddLoop(const std::vector<IterArgPtr>& iter_args, const std::vector<VarPtr>& return_vars,
               const StmtPtr& body, const Span& span) {
    for (const auto& iter_arg : iter_args) sources[iter_arg.get()].push_back(iter_arg->initValue_);
    AddYields(iter_args, body);
    // A loop that runs zero times returns its initializer, and the trip count
    // can differ per rank, so a result takes both of its carry's sources.
    INTERNAL_CHECK_SPAN(return_vars.size() == iter_args.size(), span)
        << "Internal error: a loop has " << return_vars.size() << " return vars for " << iter_args.size()
        << " iter_args";
    for (size_t i = 0; i < return_vars.size(); ++i) sources[return_vars[i].get()].push_back(iter_args[i]);
  }
};

/// True when `slice` keeps a row-major parent row-major: static shapes, and every
/// dimension after the first taken whole from offset 0. Only the leading
/// dimension may be narrowed, which moves the view origin but keeps the packed
/// strides. A `drop_dims` slice is not proven.
[[nodiscard]] bool SliceKeepsRowMajor(const CallPtr& slice) {
  if (slice->args_.size() < 3 || slice->args_.size() > 4) return false;
  auto parent = AsTensorTypeLike(slice->args_[0]->GetType());
  auto shape = As<MakeTuple>(slice->args_[1]);
  auto offset = As<MakeTuple>(slice->args_[2]);
  if (!parent || !shape || !offset || shape->elements_.size() != parent->shape_.size() ||
      offset->elements_.size() != parent->shape_.size()) {
    return false;
  }
  for (size_t i = 1; i < parent->shape_.size(); ++i) {
    auto parent_dim = As<ConstInt>(parent->shape_[i]);
    auto dim = As<ConstInt>(shape->elements_[i]);
    auto off = As<ConstInt>(offset->elements_[i]);
    if (!parent_dim || !dim || !off || dim->value_ != parent_dim->value_ || off->value_ != 0) return false;
  }
  return true;
}

/// Empty when `type` is provably packed row-major; otherwise an article and
/// adjective naming why not, for the error message. A view without explicit
/// strides is packed for its layout. An explicit stride must match
/// BuildRowMajorStrides entry by entry, which is how MaterializeTensorStrides
/// fills an ND view; a stride that is packed but spelled differently is not
/// proven.
[[nodiscard]] std::string UnpackedViewDescription(const TensorTypePtr& type) {
  if (!type || !type->tensor_view_.has_value()) return "";
  const auto& view = *type->tensor_view_;
  if (view.layout != TensorLayout::ND) return "a " + TensorLayoutToString(view.layout) + "-layout ";
  if (view.stride.empty()) return "";
  auto packed = tensor_view_semantics::BuildRowMajorStrides(type->shape_);
  bool is_packed = view.stride.size() == packed.size();
  for (size_t i = 0; is_packed && i < packed.size(); ++i) {
    auto actual = As<ConstInt>(view.stride[i]);
    auto expected = As<ConstInt>(packed[i]);
    is_packed =
        actual && expected ? actual->value_ == expected->value_ : structural_equal(view.stride[i], packed[i]);
  }
  return is_packed ? "" : "a non-packed-stride ";
}

class LowerL2TensorCollectivesMutator : public IRMutator {
 public:
  LowerL2TensorCollectivesMutator(std::map<std::string, FunctionPtr>* kernels, StmtPtr body,
                                  ProgramPtr program)
      : kernels_(kernels), body_(std::move(body)), program_(std::move(program)) {}

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!IsManagedCollective(call)) return IRMutator::VisitStmt_(op);
    auto visited = As<Call>(VisitExpr(op->value_));
    INTERNAL_CHECK_SPAN(IsManagedCollective(visited), op->span_)
        << "LowerL2TensorCollectives: collective AssignStmt rewrote to a non-collective expression";
    auto kernel_call = LowerCollective(visited, call);
    auto result_var = std::make_shared<Var>(op->var_->name_hint_, kernel_call->GetType(), op->var_->span_);
    var_remap_[op->var_.get()] = result_var;
    return std::make_shared<AssignStmt>(result_var, kernel_call, op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (!IsManagedCollective(call)) return IRMutator::VisitStmt_(op);
    auto visited = As<Call>(VisitExpr(op->expr_));
    INTERNAL_CHECK_SPAN(IsManagedCollective(visited), op->span_)
        << "LowerL2TensorCollectives: collective EvalStmt rewrote to a non-collective expression";
    return std::make_shared<EvalStmt>(LowerCollective(visited, call), op->span_);
  }

  /// ``return pld.tensor.all_to_all_v(...)`` — the collective is embedded in
  /// the ReturnStmt rather than bound to a Var first. Substituting the lowered
  /// call in place is enough: it is another Call of the same type, so no
  /// prelude statement is needed.
  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    std::vector<ExprPtr> values;
    values.reserve(op->value_.size());
    bool changed = false;
    for (const auto& value : op->value_) {
      if (auto call = As<Call>(value); IsManagedCollective(call)) {
        auto visited = As<Call>(VisitExpr(value));
        INTERNAL_CHECK_SPAN(IsManagedCollective(visited), op->span_)
            << "LowerL2TensorCollectives: collective ReturnStmt value rewrote to a non-collective "
               "expression";
        values.push_back(LowerCollective(visited, call));
        changed = true;
        continue;
      }
      values.push_back(VisitExpr(value));
    }
    if (!changed) return IRMutator::VisitStmt_(op);
    return std::make_shared<ReturnStmt>(std::move(values), op->span_, op->leading_comments_);
  }

 private:
  [[nodiscard]] static bool IsManagedCollective(const CallPtr& call) {
    return call && call->op_ && (IsOp(call, kAllToAllV) || IsOp(call, "pld.tensor.indexed_all_to_all_v"));
  }

  /// @p original is the call as written: its operands name Vars of the body the
  /// ValueSourceIndex was built from, whereas @p call's may be rewritten ones.
  [[nodiscard]] CallPtr LowerCollective(const CallPtr& call, const CallPtr& original) {
    const bool indexed = IsOp(call, "pld.tensor.indexed_all_to_all_v");
    INTERNAL_CHECK_SPAN(call->args_.size() == (indexed ? 7 : 6), call->span_)
        << "LowerL2TensorCollectives: " << call->op_->name_
        << " has an unexpected operand count: " << call->args_.size();

    // core_num (args_[5]) is the requested block limit L, carried as a
    // Scalar[INDEX] argument. It is consumed here and then dropped from the
    // kernel call — it never reaches the kernel, whose argument layout must stay
    // identical to the HOST rail's. What it becomes instead is the launch spec on
    // the call (see the end of this function).
    //
    // Only a compile-time L is accepted. A runtime L would need the block count
    // computed in the emitted orchestration, plus a runtime `signal stride >= B`
    // check, and orchestration has no surface for raising that fatal today; the
    // HOST rail takes a runtime L because its entry runs on the host.
    auto core_num_const = As<ConstInt>(call->args_.back());
    CHECK_SPAN(core_num_const != nullptr, call->span_)
        << "CHIP pld.tensor.all_to_all_v: a runtime core_num is not supported on this rail yet; pass a "
           "compile-time core_num (the HOST rail accepts a runtime value)";
    const int64_t requested_blocks = core_num_const->value_;
    // The launch width is a 32-bit count, and the HOST entry bounds `core_num` to
    // the same [1, INT32_MAX] at runtime; a static request outside it can never
    // be honoured, so say so here instead of folding it into an overflowing B.
    CHECK_SPAN(requested_blocks >= 1 && requested_blocks <= std::numeric_limits<int32_t>::max(), call->span_)
        << "CHIP pld.tensor.all_to_all_v: core_num must be in [1, " << std::numeric_limits<int32_t>::max()
        << "], got " << requested_blocks;

    auto target_type = As<DistributedTensorType>(call->args_[1]->GetType());
    INTERNAL_CHECK_SPAN(target_type, call->span_)
        << "LowerL2TensorCollectives: pld.tensor.all_to_all_v target must be DistributedTensorType";

    // The rank count P is a compile-time constant on this rail: the deducer
    // requires `signal` to be [NR, S] with static NR and S, and the program is
    // built once per rank count. That is what lets the compiler fold
    // B = CalAllToAllVBlocks(P, L) here instead of in a generated entry. The
    // kernel itself still reads the rank count from the CommContext, so NR must
    // equal the runtime domain size — an invariant this rail already relies on
    // for correctness (every counts address derives from NR), now also for width.
    auto signal_type = As<DistributedTensorType>(call->args_[2]->GetType());
    INTERNAL_CHECK_SPAN(signal_type && signal_type->shape_.size() == 2, call->span_)
        << "LowerL2TensorCollectives: pld.tensor.all_to_all_v signal must be a 2D DistributedTensor";
    auto nranks_const = As<ConstInt>(signal_type->shape_[0]);
    auto stride_const = As<ConstInt>(signal_type->shape_[1]);
    INTERNAL_CHECK_SPAN(nranks_const && stride_const, call->span_)
        << "LowerL2TensorCollectives: pld.tensor.all_to_all_v signal must have a static [NR, S] shape "
           "(deducer-enforced)";
    const int64_t nranks = nranks_const->value_;
    const int64_t signal_stride = stride_const->value_;
    // Deducer-enforced positive; stated here because CalAllToAllVBlocks divides by P.
    INTERNAL_CHECK_SPAN(nranks >= 1 && signal_stride >= 1, call->span_)
        << "LowerL2TensorCollectives: pld.tensor.all_to_all_v signal [NR, S] must be positive, got ["
        << nranks << ", " << signal_stride << "]";
    const int64_t launched_blocks = indexed ? requested_blocks : CalAllToAllVBlocks(nranks, requested_blocks);

    // A stride narrower than the launched gang is a defect, not something the
    // kernel should clamp its way past (RFC #2521 §13.3). Compare against the
    // folded B, never the raw L: L=10, P=8 launches 8 blocks, so S=8 is enough.
    // Same condition and wording as the HOST rail's entry.
    CHECK_SPAN(signal_stride >= launched_blocks, call->span_)
        << "CHIP pld.tensor.all_to_all_v: signal stride (" << signal_stride
        << ") is smaller than the admitted block count (" << launched_blocks
        << ") for core_num=" << requested_blocks << ", nranks=" << nranks;

    // send_counts is read REMOTELY by every peer (the hand-written kernel
    // resolves each peer's copy through CommRemotePtr for the counts pull), so
    // it must be a DistributedTensor on this rail: a distributed operand is
    // what the managed rail materializes as a window at runtime. The public
    // op's deducer accepts a plain Tensor for the InCore composite path, where
    // counts are consumed locally; reaching this rail with one is user error.
    auto counts_type = As<DistributedTensorType>(call->args_[3]->GetType());
    CHECK_SPAN(counts_type != nullptr, call->span_)
        << "CHIP pld.tensor.all_to_all_v send_counts must be a DistributedTensor: peers read this "
           "rank's send vector through CommRemotePtr, so a plain Tensor (accepted only on the InCore "
           "composite path) is not supported on this rail";

    auto spec = MakeAllToAllVKernelSpec(target_type->dtype_, call->span_);
    if (indexed) {
      spec.function_name =
          "__builtin_indexed_all_to_all_v__" + ResolveBuiltinDType(target_type->dtype_, call->span_).suffix;
      auto template_dir =
          OpRegistry::GetInstance().GetEntry("pld.tensor.indexed_all_to_all_v").GetTemplateDir();
      if (!template_dir.has_value()) {
        INTERNAL_CHECK_SPAN(false, call->span_) << "indexed_all_to_all_v declares no template_dir";
      } else {
        spec.template_dir = *template_dir;
      }
      spec.param_distributed[4] = false;  // Receive counts are written only locally.
      spec.param_names.push_back("index");
      spec.param_directions.push_back(ParamDirection::In);
      spec.param_distributed.push_back(false);
    }
    // Every kernel operand, named as the kernel names it, so a variant's extra
    // operand (the indexed op's `index`) is covered without a second list.
    for (size_t i = 0; i < spec.param_names.size(); ++i) {
      CheckOperandIsRowMajor(original->args_[i], call->op_->name_, spec.param_names[i], call->span_);
    }
    // One kernel per dtype, shared by every call site in the Program. The
    // kernel reads each extent from the runtime `Tensor` descriptor
    // (`target_tensor->shapes[]`, `ndims`), so shape is deliberately *not* part
    // of the variant key: two exchanges of different sizes reusing one kernel
    // is the intent, not an oversight.
    //
    // The cost is that the signature below carries whichever shape the first
    // call site happened to have, and nothing in the pipeline checks a later
    // call's argument shapes against its callee's parameters. That is safe only
    // because everything the signature actually feeds downstream is pinned
    // elsewhere:
    //
    //   dtype  - the variant name IS the target dtype, and the deducer forces
    //            input == target plus INT32 on the three control operands, so
    //            two call sites cannot reach one variant with different dtypes.
    //   kind   - `spec.param_distributed` is a constant, so the Tensor /
    //            DistributedTensor split never follows the call site. The
    //            CommCtx suffix MaterializeDistTensorCtx appends is driven by
    //            the callee's params, so args and params cannot disagree on it.
    //   rank   - the deducer pins every operand but send_counts to 2D, and
    //            send_counts is addressed flatly (`send_counts_base[dest]`
    //            locally; the remote pull word is `send_counts_base[my_rank]`).
    //
    // Relaxing any of those - a second dtype variant, a rank-3 payload - breaks
    // the sharing itself, not just this comment.
    auto inserted = kernels_->find(spec.function_name);
    if (inserted == kernels_->end()) {
      kernels_->emplace(spec.function_name, MakeBuiltinKernelFunction(spec, call));
    }
    // Drop the public op's trailing `core_num` from the kernel call: the kernel
    // sees the five operands (plus the CommCtx suffix MaterializeDistTensorCtx
    // appends from the callee's params), exactly as the HOST rail's entry-built
    // dispatch passes them.
    INTERNAL_CHECK_SPAN(call->args_.size() == spec.param_names.size() + 1, call->span_)
        << "LowerL2TensorCollectives: kernel signature must cover every call argument except the "
           "trailing core_num scalar";
    std::vector<ExprPtr> kernel_args(
        call->args_.begin(), call->args_.begin() + static_cast<std::ptrdiff_t>(spec.param_names.size()));
    auto kernel_callee = std::make_shared<GlobalVar>(spec.function_name);
    if (launched_blocks == 1) {
      // One block needs no gang launch; leave the call exactly as it was.
      return std::make_shared<Call>(kernel_callee, std::move(kernel_args), call->args_[1]->GetType(),
                                    call->span_);
    }
    // The launch spec rides the call's attrs, which are not kernel parameters, so
    // the five-operand ABI above is untouched. `require_sync_start` is not
    // optional: the block-aware barrier needs every admitted block resident
    // together, and the runtime rejects an over-capacity sync-start launch
    // rather than part-starting it.
    auto core_num_dtype = As<ScalarType>(core_num_const->GetType())->dtype_;
    std::vector<std::pair<std::string, std::any>> launch_attrs;
    launch_attrs.emplace_back(
        kAttrCoreNum, ExprPtr(std::make_shared<ConstInt>(launched_blocks, core_num_dtype, call->span_)));
    launch_attrs.emplace_back(kAttrSyncStart, true);
    return std::make_shared<Call>(kernel_callee, std::move(kernel_args),
                                  std::vector<std::pair<std::string, std::any>>{}, std::move(launch_attrs),
                                  call->args_[1]->GetType(), call->span_);
  }

  /// Require an operand to be row-major by construction, decided at compile time.
  ///
  /// The kernel reads every operand as dense row-major from its view origin, and
  /// a runtime verdict would be per rank: a rank that rejects skips only its own
  /// launch while its peers block in ExchangeBarrier. A view built in this body
  /// can differ per rank (a `[R, C]` slice of a `[R, width]` tensor with a
  /// per-rank runtime `width`), so the decision is made here, where it is the
  /// same on every rank.
  ///
  /// The trace is an allow-list (see ProvenLayoutSource): every value that can
  /// reach the operand must be a row-major root or a proven layout-preserving
  /// alias of one; anything else is rejected.
  void CheckOperandIsRowMajor(const ExprPtr& operand, const std::string& op_name, const std::string& role,
                              const Span& span) {
    std::vector<ExprPtr> pending{operand};
    while (!pending.empty()) {
      ExprPtr value = pending.back();
      pending.pop_back();
      if (!traced_.insert(value.get()).second) continue;
      auto var = AsVarLike(value);
      if (var) {
        const auto& sources = Sources().sources;
        auto it = sources.find(var.get());
        if (it != sources.end()) {
          pending.insert(pending.end(), it->second.begin(), it->second.end());
          continue;
        }
      }
      // Each value on the chain must itself be packed row-major, because the steps
      // below assume a packed source and drop its strides: a slice narrowing the
      // leading dimension of a DN or padded-stride tensor is not contiguous, and
      // an ND view of a DN tensor lowers to a runtime Tensor::transpose that the
      // runtime guard rejects. A defined Var is covered by its sources, so this is
      // tested on producers and parameters, where the error can name them.
      std::string unpacked = UnpackedViewDescription(AsTensorTypeLike(value->GetType()));
      std::string producer;
      std::optional<ExprPtr> source;
      if (var) {
        // A Var with no source in this body is a parameter: a window, or a tensor
        // the caller passes in, whose runtime layout the guard checks (see
        // kAttrRequireContiguousOperands). A Graph parameter would come from
        // a CHIP caller instead, but a Graph holding this collective cannot be
        // compiled: its window operands gain CommCtx params, which
        // LegalizeGraphBoundary rejects. If that changes, trace the callers' args.
        if (unpacked.empty()) continue;
        producer = unpacked + "parameter '" + var->name_hint_ + "'";
      } else if (!unpacked.empty()) {
        auto call = As<Call>(value);
        producer = unpacked + (call ? call->op_->name_ : value->TypeName());
      } else {
        source = ProvenLayoutSource(value, &producer);
      }
      CHECK_SPAN(source.has_value(), span)
          << "CHIP " << op_name << ": " << role << " is derived from " << producer
          << ", which is not provably row-major. The kernel reads every operand as dense row-major, and a "
             "runtime check would decide per rank (see the LowerL2TensorCollectives pass docs). Pass a "
             "parameter or a freshly created tensor, or a slice that narrows only the leading dimension of a "
             "statically shaped tensor";
      if (*source) pending.push_back(*source);
    }
  }

  /// Allow-list step of CheckOperandIsRowMajor for a non-Var value. Returns
  /// nullopt (and names the producer) when the layout is not proven; nullptr for
  /// a fresh row-major allocation; otherwise the value whose layout this one
  /// shares (a view, writeback, or returned argument of it).
  [[nodiscard]] std::optional<ExprPtr> ProvenLayoutSource(const ExprPtr& value, std::string* producer) {
    // A tuple element of a kernel or submit result (`x, n = self.kernel(...)`,
    // `x, tid = pl.submit(...)`) is that result at the element's position.
    size_t result_index = 0;
    ExprPtr produced = value;
    if (auto item = As<TupleGetItemExpr>(value)) {
      result_index = static_cast<size_t>(item->index_);
      produced = item->tuple_;
      if (auto var = AsVarLike(produced)) {
        const auto& sources = Sources().sources;
        auto it = sources.find(var.get());
        produced = it != sources.end() && it->second.size() == 1 ? it->second[0] : nullptr;
      }
    }
    if (auto submit = As<Submit>(produced)) {
      *producer = "a submit of " + submit->op_->name_;
      return ReturnedArgument(submit->op_, submit->args_, result_index);
    }
    auto call = As<Call>(produced);
    *producer = call ? call->op_->name_ : value->TypeName();
    if (!call) return std::nullopt;
    if (IsOp(call->op_, "tensor.create")) return ExprPtr(nullptr);
    if (IsOp(call->op_, "tensor.reshape") || IsOp(call->op_, "tensor.view") ||
        IsOp(call->op_, "tensor.set_validshape") || IsOp(call->op_, "tensor.assemble")) {
      return call->args_[0];  // assemble writes back into, and returns, its target
    }
    if (IsOp(call->op_, "tensor.slice")) {
      return SliceKeepsRowMajor(call) ? std::optional<ExprPtr>(call->args_[0]) : std::nullopt;
    }
    if (IsManagedCollective(call)) return call->args_[1];  // returns its target
    return ReturnedArgument(call->op_, call->args_, result_index);
  }

  /// A kernel's result at @p result_index is the argument bound to the parameter
  /// the kernel writes back there, or a fresh allocation when a submit leaves
  /// that parameter to the runtime; anything else (not a kernel, a non-writeback
  /// result, a submit's trailing TaskId) is not proven.
  [[nodiscard]] std::optional<ExprPtr> ReturnedArgument(const OpPtr& op, const std::vector<ExprPtr>& args,
                                                        size_t result_index) {
    auto gvar = As<GlobalVar>(op);
    auto callee = gvar ? program_->GetFunction(gvar->name_) : nullptr;
    if (!callee) return std::nullopt;
    // This pass runs after NormalizeReturnOrder, so the writeback map is read
    // straight off the callee's ReturnStmt.
    auto [it, inserted] = returned_params_.try_emplace(callee.get());
    if (inserted) it->second = return_lineage::ExplicitReturnedParamIndices(callee);
    const auto& returned = it->second;
    if (result_index >= returned.size()) return std::nullopt;
    const auto& param = returned[result_index];
    if (!param) return std::nullopt;
    // args[i] binds params[i]: a Submit has no CommCtx suffix before
    // MaterializeDistTensorCtx. A param past `args` is an Out the runtime
    // allocates from TensorCreateInfo(shapes, ndim, dtype), packed like a
    // tensor.create; the caller's per-value packed check still rejects a
    // non-packed type.
    return *param < args.size() ? args[*param] : ExprPtr(nullptr);
  }

  /// The source index of the body as written, built on the first collective so
  /// a body without one never pays for it.
  const ValueSourceIndex& Sources() {
    if (!sources_) {
      sources_.emplace();
      sources_->VisitStmt(body_);
    }
    return *sources_;
  }

  std::map<std::string, FunctionPtr>* kernels_;
  StmtPtr body_;
  std::optional<ValueSourceIndex> sources_;
  ProgramPtr program_;
  /// Writeback map per callee, read once however many results trace into it.
  std::unordered_map<const Function*, std::vector<std::optional<size_t>>> returned_params_;
  /// Values already traced by CheckOperandIsRowMajor in this function. A trace
  /// that fails throws and aborts the pass, so every value left here is proven,
  /// and sharing the set across collectives makes the tracing O(N) per function
  /// (each value is walked once) rather than once per collective. It also stops
  /// the walk on loop-carried cycles.
  std::unordered_set<const Expr*> traced_;
};

/// Guard the pass' own postcondition: no managed collective survives in an
/// orchestration body this rail owns. A HOST orchestrator is exempt (it keeps
/// its own rail, LowerHostTensorCollectives, five passes later); an InCore body
/// is not checked at all, because the composite rail owns those and already ran
/// 26 passes earlier — re-reporting them here would blame the wrong pass.
///
/// This is the only check that answers "may this collective appear in this
/// body". The orchestration-reference verifier exempts this operator family —
/// a managed collective is an operator, not a function reference, so the
/// Program is not expected to define it — and deliberately answers nothing
/// beyond that. Without this check an unsupported collective would ride that
/// exemption all the way to codegen and surface as an unknown operator.
class ResidualCollectiveChecker : public IRVisitor {
 public:
  explicit ResidualCollectiveChecker(std::string func_name) : func_name_(std::move(func_name)) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    if (op && IsManagedTensorCollective(op->op_)) {
      CHECK_SPAN(false, op->span_)
          << op->op_->name_ << " in function '" << func_name_
          << "' was not lowered. The managed CHIP/L2 rail currently supports only " << kAllToAllV
          << " with a compile-time core_num; write any other collective in a HOST orchestrator (builtin "
             "dispatch rail) or an InCore function (composite rail)";
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  std::string func_name_;
};

/// Orchestration bodies this rail is responsible for: every orchestration-like
/// function except a HOST orchestrator, which defers to
/// LowerHostTensorCollectives and so still holds unlowered collectives here by
/// design.
[[nodiscard]] bool OwnsResidualCheck(const FunctionPtr& func) {
  if (!func || !func->role_.has_value() || *func->role_ != Role::Orchestrator) return false;
  return !func->level_.has_value() || *func->level_ != Level::HOST;
}

void CheckNoResidualCollective(const ProgramPtr& program) {
  for (const auto& [gvar, func] : program->functions_) {
    (void)gvar;
    if (!OwnsResidualCheck(func)) continue;
    ResidualCollectiveChecker checker(func->name_);
    checker.VisitStmt(func->body_);
  }
}

ProgramPtr TransformProgram(const ProgramPtr& program) {
  std::map<std::string, FunctionPtr> kernels;
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
  bool modified = false;
  for (const auto& [gvar, func] : program->functions_) {
    if (!IsChipOrch(func)) {
      new_functions[gvar] = func;
      continue;
    }
    LowerL2TensorCollectivesMutator mutator(&kernels, func->body_, program);
    auto new_func = mutator.VisitFunction(func);
    new_functions[gvar] = new_func;
    if (new_func.get() != func.get()) modified = true;
  }
  if (!modified) {
    CheckNoResidualCollective(program);
    return program;
  }

  for (auto& [name, kernel] : kernels) {
    auto gvar = std::make_shared<GlobalVar>(name);
    CHECK(new_functions.find(gvar) == new_functions.end())
        << "LowerL2TensorCollectives: synthesized builtin kernel name '" << name
        << "' collides with an existing function";
    new_functions[gvar] = kernel;
  }

  auto result = std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  CheckNoResidualCollective(result);
  return result;
}

}  // namespace

namespace pass {

Pass LowerL2TensorCollectives() {
  return CreateProgramPass(TransformProgram, "LowerL2TensorCollectives", kLowerL2TensorCollectivesProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto

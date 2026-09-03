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
 *                                      send_counts, recv_counts, core_num=1)
 *
 * into a call to a synthesized AIV kernel function backed by the *same*
 * hand-written builtin kernel source the HOST rail uses:
 *
 *     target = __builtin_all_to_all_v__fp32(input, target, signal,
 *                                           send_counts, recv_counts)
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
 *     args[6..9]  unread duplicates of args[5] — MaterializeDistTensorCtx
 *                 appends one CommCtx parameter per DistributedTensor
 *                 parameter, and all five resolve to the same device_ctx
 *                 (every operand of one collective belongs to one comm domain)
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
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

using op_predicates::IsManagedTensorCollective;

/// The one managed collective this rail lowers today.
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

/// Variant suffix and C++ element type the builtin kernel template is
/// instantiated with.
///
/// Deliberately the same single-dtype support the HOST rail declares
/// (``Fp32VariantSuffix`` / ``Fp32TypeCpp`` in distributed_ops_codegen.cpp):
/// both rails render the same template, so widening the dtype set is one change
/// for both, not a CHIP-only divergence.
struct BuiltinDType {
  const char* suffix;
  const char* cpp;
};

[[nodiscard]] BuiltinDType ResolveBuiltinDType(const DataType& dtype, const Span& span) {
  CHECK_SPAN(dtype == DataType::FP32, span)
      << "managed pld.tensor.all_to_all_v currently supports only dtype=FP32, got " << dtype.ToString();
  return {"fp32", "float"};
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
      // `input` and `send_counts` are Tensor-like on the public op: either a
      // plain Tensor or a DistributedTensor. The kernel sees a flat `Tensor*`
      // for both, so the synthesized signature fixes them as plain Tensor and
      // the variant stays keyed on dtype alone. The three genuinely
      // window-bound operands stay distributed, which is also what gives
      // MaterializeDistTensorCtx the CommCtx parameters the kernel needs.
      {false, true, true, false, true},
  };
}

/// The parameter type the synthesized signature declares, which is deliberately
/// *not* whatever the call site passed.
///
/// `input` and `send_counts` accept either a Tensor or a DistributedTensor, and
/// the two are indistinguishable to the kernel — both arrive as a flat
/// `Tensor*`. Copying the first call's types would make one cached function
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
[[nodiscard]] FunctionPtr MakeBuiltinKernelFunction(const BuiltinKernelSpec& spec, const CallPtr& call) {
  std::vector<VarPtr> params;
  params.reserve(call->args_.size());
  for (size_t i = 0; i < call->args_.size(); ++i) {
    params.push_back(std::make_shared<Var>(
        spec.param_names[i], CanonicalParamType(spec.param_distributed[i], call->args_[i], call->span_),
        call->span_));
  }
  std::vector<std::pair<std::string, std::any>> attrs = {
      {kAttrBuiltinTemplateDir, spec.template_dir},
      {kAttrBuiltinTemplateVars, spec.template_vars},
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

class LowerL2TensorCollectivesMutator : public IRMutator {
 public:
  explicit LowerL2TensorCollectivesMutator(std::map<std::string, FunctionPtr>* kernels) : kernels_(kernels) {}

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!IsManagedCollective(call)) return IRMutator::VisitStmt_(op);
    auto visited = As<Call>(VisitExpr(op->value_));
    INTERNAL_CHECK_SPAN(IsManagedCollective(visited), op->span_)
        << "LowerL2TensorCollectives: collective AssignStmt rewrote to a non-collective expression";
    auto kernel_call = LowerCollective(visited);
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
    return std::make_shared<EvalStmt>(LowerCollective(visited), op->span_);
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
        values.push_back(LowerCollective(visited));
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
    return call && call->op_ && IsOp(call, kAllToAllV);
  }

  [[nodiscard]] CallPtr LowerCollective(const CallPtr& call) {
    INTERNAL_CHECK_SPAN(call->args_.size() == 5, call->span_)
        << "LowerL2TensorCollectives: " << call->op_->name_ << " must have 5 args, got "
        << call->args_.size();

    // core_num is the requested block limit L. The first version launches a
    // single block, so anything else would silently under-deliver; the dynamic
    // L -> B mapping lands with the multi-AIV entry work.
    const auto core_num = call->GetKwarg<int>("core_num", 1);
    CHECK_SPAN(core_num == 1, call->span_)
        << "CHIP pld.tensor.all_to_all_v currently supports only core_num=1, got core_num=" << core_num
        << "; multi-AIV launch is not implemented yet";

    auto target_type = As<DistributedTensorType>(call->args_[1]->GetType());
    INTERNAL_CHECK_SPAN(target_type, call->span_)
        << "LowerL2TensorCollectives: pld.tensor.all_to_all_v target must be DistributedTensorType";

    auto spec = MakeAllToAllVKernelSpec(target_type->dtype_, call->span_);
    auto inserted = kernels_->find(spec.function_name);
    if (inserted == kernels_->end()) {
      kernels_->emplace(spec.function_name, MakeBuiltinKernelFunction(spec, call));
    }
    return std::make_shared<Call>(std::make_shared<GlobalVar>(spec.function_name), call->args_,
                                  call->args_[1]->GetType(), call->span_);
  }

  std::map<std::string, FunctionPtr>* kernels_;
};

/// Guard the pass' own postcondition: no managed collective survives in an
/// orchestration body this rail owns. A HOST orchestrator is exempt (it keeps
/// its own rail, LowerHostTensorCollectives, five passes later); an InCore body
/// is not checked at all, because the composite rail owns those and already ran
/// 26 passes earlier — re-reporting them here would blame the wrong pass.
///
/// This is the only check that answers "may this collective appear in this
/// body": the orchestration-reference verifier exempts the whole `pld.`
/// operator namespace, so an unsupported collective would otherwise reach
/// codegen as an unknown operator.
class ResidualCollectiveChecker : public IRVisitor {
 public:
  explicit ResidualCollectiveChecker(std::string func_name) : func_name_(std::move(func_name)) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    if (op && IsManagedTensorCollective(op->op_)) {
      CHECK_SPAN(false, op->span_)
          << op->op_->name_ << " in function '" << func_name_
          << "' was not lowered. The managed CHIP/L2 rail currently supports only " << kAllToAllV
          << " with core_num=1; write any other collective in a HOST orchestrator (builtin "
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
    LowerL2TensorCollectivesMutator mutator(&kernels);
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

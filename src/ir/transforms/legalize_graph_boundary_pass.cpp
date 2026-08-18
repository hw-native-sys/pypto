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
 * @file legalize_graph_boundary_pass.cpp
 * @brief Make every FunctionType::Graph function legal to record and replay.
 *
 * The host_build_graph runtime records a Graph function's task topology on the
 * first call and replays it afterwards, patching only buffer addresses and
 * boundary scalars. Two classes of problem follow from that, and this pass
 * exists to catch both at compile time:
 *
 * **Step A — derived boundary scalars (silent wrong answers).** A boundary
 * scalar is tracked by *pointer identity*: the runtime anchors the address of
 * each `args.scalar(k)` slot during recording and re-reads it on replay. A value
 * the body *derives* from a scalar parameter (`base = layer * 5120`) has no such
 * slot, so it is classified as static data and frozen at its first-call value —
 * with no warning on any later replay. Step A hoists those computations to the
 * call sites, where they become ordinary pass-through scalars, and rejects the
 * ones it cannot hoist.
 *
 * **Step D — boundary legality (silent fallback).** Almost every other runtime
 * constraint degrades to a silent non-graph fallback in a release build: the
 * program is correct but the feature does nothing, which no numerical test can
 * detect. Step D checks the statically decidable ones and fails loudly instead.
 *
 * Runs after DeriveCallDirections and AutoDeriveTaskDependencies (so argument
 * directions and cross-task edges are known) and before MaterializeRuntimeScopes
 * (so scopes are not yet materialised around the statements it moves).
 */

#include <algorithm>
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
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/alloc_batching.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Runtime hard limits, from runtime/src/common/host_build_graph/.
/// A violation of either is a compile-time error here, because at runtime it
/// only makes the recording non-cacheable and the graph silently falls back.
constexpr size_t kMaxBoundaryTensors = 128;  ///< GRAPH_MAX_TENSOR_ARGS, host_build_graph/runtime/types.h
constexpr size_t kMaxBoundaryScalars = 64;   ///< GRAPH_MAX_SCALAR_ARGS, host_build_graph/runtime/types.h
constexpr size_t kMaxGraphNodes = 1024;      ///< GRAPH_MAX_NODES, common/host_build_graph/graph_execution.h

/// Directions a Graph parameter may carry.
///
/// `ArgDirection::Output` means "the runtime allocates this buffer", which
/// `rt_graph_args_cacheable` rejects outright: a recorded graph's boundary
/// tensors must already exist so replay can patch their addresses.
[[nodiscard]] bool IsLegalGraphParamDirection(ParamDirection dir) {
  return dir == ParamDirection::In || dir == ParamDirection::InOut;
}

/// True when `type` is a scalar, i.e. a candidate boundary scalar.
[[nodiscard]] bool IsScalarType(const TypePtr& type) { return As<ScalarType>(type) != nullptr; }

/// True when `expr` is built entirely from literals. Defined below, next to the
/// other replay-invariance helpers; declared here for the Step B shape check.
[[nodiscard]] bool IsLiteralScalarExpr(const ExprPtr& expr);

// ---------------------------------------------------------------------------
// Step A — hoisting derived boundary scalars
// ---------------------------------------------------------------------------

/// One value the body computes that the call sites must supply instead.
///
/// The same record serves all three hoisting steps, because they differ only in
/// what is being moved and how the resulting parameter is declared:
///
/// | Step | What moves | Parameter direction |
/// | ---- | ---------- | ------------------- |
/// | A | a scalar derived from scalar params | In |
/// | B | a slice/view of a boundary tensor | In |
/// | C | a `tensor.create` the region allocates | InOut |
struct HoistedValue {
  VarPtr param;              ///< Fresh parameter appended to the Graph signature.
  VarPtr original;           ///< Body variable it replaces.
  ExprPtr value;             ///< Defining expression, in terms of the Graph's own params.
  ParamDirection direction;  ///< How the new parameter is declared.
  bool is_tensor;            ///< Tensor params must precede scalar ones.
};

/// What the hoisting steps decided for one Graph function.
struct GraphPlan {
  std::string name;
  std::vector<HoistedValue> hoisted;
  /// `alias = <var>` bindings, in definition order, alias -> what it names.
  ///
  /// Not hoisted — a bare reference names something that already has a binding —
  /// but a hoisted expression may still be written in terms of one, and the call
  /// site has no such name. Kept in definition order and interleaved with the
  /// scalar hoists at the call site: an alias may name a *derived* value rather
  /// than a parameter (`base = idx * 128; alias = base; end = alias + 1`), and
  /// that value is only bound once its own hoist runs.
  std::vector<std::pair<const Var*, VarPtr>> passthrough;
  /// Position in the body of every hoisted and aliased variable, so the two
  /// lists can be merged back into definition order.
  std::unordered_map<const Var*, size_t> definition_index;
};

/// Collects scalar variables a Graph body derives from its own scalar params.
///
/// A value is *derivable* when its whole expression tree bottoms out in scalar
/// parameters and constants: that is exactly the set a call site can recompute,
/// because it already supplies those parameters. Anything reaching a task output
/// or a tensor read is not, and is reported rather than silently frozen.
class DerivedScalarCollector : public IRVisitor {
 public:
  explicit DerivedScalarCollector(const FunctionPtr& func) : func_(func) {
    for (const auto& param : func->params_) {
      if (IsScalarType(param->GetType())) {
        scalar_params_.insert(param.get());
      } else {
        tensor_params_.insert(param.get());
      }
    }
  }

  /// Body variables bound to a derivable expression, in definition order.
  [[nodiscard]] const std::vector<std::pair<VarPtr, ExprPtr>>& derived() const { return derived_; }

  /// `alias = <var>` bindings in definition order, alias -> what it names.
  [[nodiscard]] const std::vector<std::pair<const Var*, VarPtr>>& passthrough() const {
    return passthrough_order_;
  }

  /// Position in the body of every hoisted and aliased variable.
  [[nodiscard]] const std::unordered_map<const Var*, size_t>& definition_index() const {
    return definition_index_;
  }

  /// Step B: slices/views of a boundary tensor, in definition order.
  [[nodiscard]] const std::vector<std::pair<VarPtr, ExprPtr>>& slices() const { return slices_; }

  /// Boundary tensor parameter each hoisted view ultimately derives from.
  ///
  /// A view can never carry wider access than the tensor it views, so the root's
  /// direction is what the hoisted parameter must be declared with — see
  /// `BuildPlan`.
  [[nodiscard]] const std::unordered_map<const Var*, const Var*>& tensor_root() const { return tensor_root_; }

  /// Step C: tensors the region allocates for itself, in definition order.
  [[nodiscard]] const std::vector<std::pair<VarPtr, ExprPtr>>& creates() const { return creates_; }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    auto var = AsVarLike(op->var_);
    if (!var) return;

    if (IsScalarType(var->GetType())) {
      if (!IsDerivable(op->value_)) return;
      // A bare parameter reference is already a pass-through; only a *computed*
      // value needs hoisting.
      if (auto aliased = AsVarLike(op->value_)) {
        passthrough_.insert(var.get());
        // Record what it names, in definition order. Resolution happens at the
        // call site, where what it names may itself be a hoist not yet bound.
        definition_index_[var.get()] = next_definition_++;
        passthrough_order_.emplace_back(var.get(), aliased);
        return;
      }
      derived_vars_.insert(var.get());
      definition_index_[var.get()] = next_definition_++;
      derived_.emplace_back(var, op->value_);
      return;
    }

    // A bare tensor alias inherits its source's provenance. Without this,
    // `alias = w; wl = slice(alias, ...)` leaves `wl` in the region: its source
    // is neither an original parameter nor a collected view, so Step B skips it
    // and the recording freezes the first call's offset.
    if (auto aliased = AsVarLike(op->value_)) {
      auto root = RootBoundaryTensor(aliased);
      if (root != nullptr) {
        tensor_root_[var.get()] = root;
        // Recorded like a scalar pass-through: the call site has no name for a
        // Graph-local alias, so a hoisted view written in terms of one must be
        // able to resolve it back to whatever the caller passed. Definition order
        // matters because the alias may name an earlier hoisted view.
        definition_index_[var.get()] = next_definition_++;
        passthrough_order_.emplace_back(var.get(), aliased);
      }
      return;
    }

    auto call = As<Call>(op->value_);
    if (!call) return;

    // Step C: an allocation inside the region. Codegen lowers `tensor.create`
    // into a batched `alloc_tensors`, and a bare `alloc_tensors` anywhere in a
    // recorded region poisons the recording outright.
    if (IsOp(call, "tensor.create") || IsOp(call, "tensor.full")) {
      creates_.emplace_back(var, op->value_);
      return;
    }

    // Step B: a slice or view of a boundary tensor. Replay patches a boundary
    // tensor's address, but a view taken *inside* the region is re-derived from
    // whatever the recording froze, so it must be taken at the call site.
    if (!IsOp(call, "tensor.slice") && !IsOp(call, "tensor.view")) return;
    if (call->args_.empty()) return;
    auto source = AsVarLike(call->args_[0]);
    // A view of a hoisted view is a boundary view too. Once `s1 = slice(w, ...)`
    // moves out, `s1` is a boundary parameter, so `s2 = slice(s1, ...)` is in
    // exactly the position `s1` was and has to move out with it. The body is SSA
    // in definition order, so one forward pass reaches the whole chain — a view
    // can only name a source defined before it.
    const Var* root = source ? RootBoundaryTensor(source) : nullptr;
    if (root == nullptr) return;
    // Every non-source operand must be reconstructible at the call site.
    //
    // Rejected rather than skipped: leaving such a view in the region is the
    // silent path. Recording classifies it as a view of its boundary source and
    // freezes the offset it computed on the first call; replay patches the
    // source's address but never re-runs the body, so every later call reads the
    // first call's offset from the right buffer.
    for (size_t i = 1; i < call->args_.size(); ++i) {
      CHECK_SPAN(IsDerivable(call->args_[i]), call->span_)
          << "Graph function '" << func_->name_ << "' derives '" << var->name_hint_
          << "' from boundary tensor '" << source->name_hint_
          << "' using a value the call site cannot recompute. Replay patches the boundary tensor's "
             "address but keeps the offset recorded on the first call, so the view would silently "
             "address the first call's window. Take the view at the call site and pass it in.";
    }
    // The recording stores a view's shape and strides in the node template and
    // patches only the buffer address and start offset (`graph_rebind_tensor`).
    // A shape built from a boundary scalar is therefore frozen at the first
    // call's value and replayed against a later call's buffer.
    if (auto viewed = As<TensorType>(call->GetType())) {
      for (const auto& extent : viewed->shape_) {
        CHECK_SPAN(IsLiteralScalarExpr(extent), call->span_)
            << "Graph function '" << func_->name_ << "' takes a view '" << var->name_hint_
            << "' of boundary tensor '" << source->name_hint_
            << "' whose shape is not a compile-time constant. Recording freezes a view's shape and "
               "strides into the node and replay patches only its address, so a later call with a "
               "different extent would replay the first call's shape. Give the view a fixed shape — "
               "a varying offset is fine — or take it at the call site.";
      }
    }
    slice_vars_.insert(var.get());
    tensor_root_[var.get()] = root;
    definition_index_[var.get()] = next_definition_++;
    slices_.emplace_back(var, op->value_);
  }

 private:
  /// The boundary tensor parameter @p var derives from, or null.
  ///
  /// A parameter is its own root; an alias or a collected view inherits one.
  [[nodiscard]] const Var* RootBoundaryTensor(const VarPtr& var) const {
    if (!var) return nullptr;
    if (tensor_params_.count(var.get()) != 0) return var.get();
    auto it = tensor_root_.find(var.get());
    return it == tensor_root_.end() ? nullptr : it->second;
  }

  /// True when `expr` can be recomputed at a call site.
  ///
  /// That holds when it contains no call — a call could read a task output, a
  /// tensor, or runtime state, none of which the caller can reproduce — and
  /// every variable it names is a scalar parameter or an already-derived
  /// scalar, which the caller does supply.
  ///
  /// Implemented as a generic walk rather than a hand-written recursion over
  /// node kinds: an operand is often a list or tuple (a slice's shape and offset
  /// vectors, for instance), and enumerating every container kind would silently
  /// treat the ones it missed as non-derivable.
  [[nodiscard]] bool IsDerivable(const ExprPtr& expr) const {
    if (!expr) return false;

    class Checker : public IRVisitor {
     public:
      Checker(const std::unordered_set<const Var*>& scalar_params,
              const std::unordered_set<const Var*>& derived, const std::unordered_set<const Var*>& through)
          : scalar_params_(scalar_params), derived_(derived), through_(through) {}
      bool derivable = true;

     protected:
      void VisitExpr_(const CallPtr& op) override { derivable = false; }
      void VisitExpr_(const SubmitPtr& op) override { derivable = false; }
      void VisitVarLike_(const VarPtr& op) override {
        if (scalar_params_.count(op.get()) == 0 && derived_.count(op.get()) == 0 &&
            through_.count(op.get()) == 0) {
          derivable = false;
        }
      }

     private:
      const std::unordered_set<const Var*>& scalar_params_;
      const std::unordered_set<const Var*>& derived_;
      const std::unordered_set<const Var*>& through_;
    };

    Checker checker(scalar_params_, derived_vars_, passthrough_);
    checker.VisitExpr(expr);
    return checker.derivable;
  }

  FunctionPtr func_;
  std::unordered_set<const Var*> scalar_params_;
  std::unordered_set<const Var*> tensor_params_;
  std::unordered_set<const Var*> derived_vars_;
  std::unordered_set<const Var*> passthrough_;
  std::vector<std::pair<const Var*, VarPtr>> passthrough_order_;
  std::unordered_map<const Var*, size_t> definition_index_;
  size_t next_definition_ = 0;
  std::vector<std::pair<VarPtr, ExprPtr>> derived_;
  /// Vars already collected as boundary views, so a view *of* one is collected too.
  std::unordered_set<const Var*> slice_vars_;
  /// Body tensor var -> the boundary parameter it derives from (alias or view).
  std::unordered_map<const Var*, const Var*> tensor_root_;
  std::vector<std::pair<VarPtr, ExprPtr>> slices_;
  std::vector<std::pair<VarPtr, ExprPtr>> creates_;
};

/// True when @p expr is built only from literals.
///
/// Such a value is the same on every call, so freezing it into the recording is
/// harmless. Anything that reads a variable is not.
[[nodiscard]] bool IsLiteralScalarExpr(const ExprPtr& expr) {
  if (!expr) return false;
  if (As<ConstInt>(expr) || As<ConstFloat>(expr) || As<ConstBool>(expr)) return true;
  if (auto bin = std::dynamic_pointer_cast<const BinaryExpr>(expr)) {
    return IsLiteralScalarExpr(bin->left_) && IsLiteralScalarExpr(bin->right_);
  }
  if (auto un = std::dynamic_pointer_cast<const UnaryExpr>(expr)) {
    return IsLiteralScalarExpr(un->operand_);
  }
  return false;
}

/// Rejects scalars a task consumes that Step A could not hoist.
///
/// Reaching a task with a value the runtime cannot anchor is the C4 failure:
/// the value is silently frozen into the recorded Definition and every later
/// replay reuses the first call's number.
class UnhoistableScalarChecker : public IRVisitor {
 public:
  UnhoistableScalarChecker(const FunctionPtr& func, const std::unordered_set<const Var*>& hoistable)
      : func_(func), hoistable_(hoistable) {
    for (const auto& param : func->params_) {
      if (IsScalarType(param->GetType())) scalar_params_.insert(param.get());
    }
  }

 protected:
  /// A bare copy of something that already has a slot keeps that slot.
  ///
  /// Step A classifies `alias = <scalar param>` as a *pass-through* rather than a
  /// hoist — there is nothing to compute at the call site — so the assignment
  /// survives the rewrite. Rejecting it here would contradict that: the value is
  /// trivially reconstructible, it *is* a parameter. `ConvertToSSA` produces this
  /// shape routinely (`layer_idx__ssa_v1 = layer_idx`), so refusing it rejects
  /// ordinary correct programs. Chains are followed, since an alias may name an
  /// earlier alias.
  void VisitStmt_(const AssignStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    auto var = AsVarLike(op->var_);
    if (!var || !IsScalarType(var->GetType())) return;
    auto aliased = AsVarLike(op->value_);
    if (!aliased) return;
    if (scalar_params_.count(aliased.get()) != 0 || hoistable_.count(aliased.get()) != 0 ||
        passthrough_.count(aliased.get()) != 0) {
      passthrough_.insert(var.get());
    }
  }

  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    CheckArgs(op->args_, op->span_);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    IRVisitor::VisitExpr_(op);
    CheckArgs(op->args_, op->span_);
  }

 private:
  void CheckArgs(const std::vector<ExprPtr>& args, const Span& span) const {  // NOLINT(readability-*)
    for (const auto& arg : args) {
      if (!arg || !IsScalarType(arg->GetType())) continue;
      auto var = AsVarLike(arg);
      if (!var) {
        // Not a bare name but an expression written inline at the call, e.g.
        // `self.kernel(a, c, idx * 128)`. Step A only hoists *named* bindings,
        // so nothing rewrites this one and the task receives a computed value
        // with no boundary slot — the same silent freeze as the named case, and
        // previously waved through because the arg is not a Var.
        CHECK_SPAN(IsLiteralScalarExpr(arg), span)
            << "Graph function '" << func_->name_
            << "' computes a scalar inline in a task argument. Under host_build_graph a boundary "
               "scalar is tracked by the address of its argument slot; a value computed inside the "
               "region has no slot, so the runtime would freeze the first call's value into the "
               "recorded graph and silently reuse it on every replay. Bind it to a name first — a "
               "named value derived from this function's scalar parameters and constants is hoisted "
               "to the call site automatically.";
        continue;
      }
      if (scalar_params_.count(var.get()) != 0 || hoistable_.count(var.get()) != 0 ||
          passthrough_.count(var.get()) != 0) {
        continue;
      }
      CHECK_SPAN(false, span)
          << "Graph function '" << func_->name_ << "' passes scalar '" << var->name_hint_
          << "' to a task, but its value cannot be reconstructed at the call site. Under "
             "host_build_graph a boundary scalar is tracked by the address of its argument slot; a "
             "value computed inside the region has no slot, so the runtime would freeze the first "
             "call's value into the recorded graph and silently reuse it on every replay. Compute '"
          << var->name_hint_
          << "' at the call site and pass it in, or derive it only from this function's scalar "
             "parameters and constants.";
    }
  }

  FunctionPtr func_;
  const std::unordered_set<const Var*>& hoistable_;
  std::unordered_set<const Var*> scalar_params_;
  /// Vars that are a bare copy of a parameter (or of another such copy).
  std::unordered_set<const Var*> passthrough_;
};

/// Replaces hoisted body variables with their new parameters and erases the
/// assignments that used to compute them — the value now arrives as an argument.
class HoistedValueRewriter : public IRMutator {
 public:
  explicit HoistedValueRewriter(const GraphPlan& plan) {
    for (const auto& h : plan.hoisted) replacement_[h.original.get()] = h.param;
  }

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = replacement_.find(op.get());
    return it == replacement_.end() ? IRMutator::VisitExpr_(op) : it->second;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto var = AsVarLike(op->var_);
    if (var && replacement_.count(var.get()) != 0) {
      // The value now arrives as a parameter, so its computation is dead.
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
    }
    return IRMutator::VisitStmt_(op);
  }

 private:
  std::unordered_map<const Var*, VarPtr> replacement_;
};

/// Rebuilds one hoisted expression in terms of a call site's actual arguments.
///
/// Uses the generic mutator rather than reconstructing each operator node by
/// hand: the arithmetic family has ~28 concrete kinds, and the mutator already
/// knows how to rebuild every one of them from its reflected fields.
class ExprSubstituter : public IRMutator {
 public:
  explicit ExprSubstituter(const std::unordered_map<const Var*, ExprPtr>& binding) : binding_(binding) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = binding_.find(op.get());
    return it == binding_.end() ? IRMutator::VisitExpr_(op) : it->second;
  }

 private:
  const std::unordered_map<const Var*, ExprPtr>& binding_;
};

[[nodiscard]] ExprPtr SubstituteAtCallSite(const ExprPtr& expr,
                                           const std::unordered_map<const Var*, ExprPtr>& binding) {
  if (!expr) return expr;
  ExprSubstituter substituter(binding);
  return substituter.VisitExpr(expr);
}

/// Appends the hoisted scalars to a Graph function's signature.
[[nodiscard]] FunctionPtr ExtendGraphSignature(const FunctionPtr& func, const GraphPlan& plan,
                                               const StmtPtr& new_body) {
  std::vector<VarPtr> params = func->params_;
  std::vector<ParamDirection> dirs = func->param_directions_;
  // Appended, not prepended, and tensors before scalars within the appended
  // group — `CoreTaskArgs` requires every tensor argument to precede every
  // scalar one. BuildPlan already orders `hoisted` that way.
  for (const auto& h : plan.hoisted) {
    params.push_back(h.param);
    dirs.push_back(h.direction);
  }
  return std::make_shared<Function>(func->name_, std::move(params), std::move(dirs), func->return_types_,
                                    new_body, func->span_, func->func_type_, func->level_, func->role_,
                                    func->attrs_, func->requires_runtime_binding_);
}

// ---------------------------------------------------------------------------
// Step D — boundary legality
// ---------------------------------------------------------------------------

/// Static trip count of @p op, or nullopt when it is not provable here.
///
/// Computed in unsigned arithmetic. `stop - start`, `-step` and the round-up
/// `distance + stride - 1` all overflow int64_t on extreme bounds — `start =
/// INT64_MIN, stop = 0, step = 1` wraps to a negative span and reports a
/// zero-trip loop, which would silently drop every launch inside it from the
/// node count.
[[nodiscard]] std::optional<size_t> StaticTripCount(const ForStmtPtr& op) {
  auto start_c = As<ConstInt>(op->start_);
  auto stop_c = As<ConstInt>(op->stop_);
  auto step_c = As<ConstInt>(op->step_);
  if (!start_c || !stop_c || !step_c || step_c->value_ == 0) return std::nullopt;
  const int64_t start = start_c->value_;
  const int64_t stop = stop_c->value_;
  const int64_t step = step_c->value_;
  if (step > 0 ? stop <= start : stop >= start) return 0;
  const auto u_start = static_cast<uint64_t>(start);
  const auto u_stop = static_cast<uint64_t>(stop);
  const auto u_step = static_cast<uint64_t>(step);
  const uint64_t distance = step > 0 ? u_stop - u_start : u_start - u_stop;
  const uint64_t stride = step > 0 ? u_step : uint64_t{0} - u_step;
  return static_cast<size_t>(distance / stride + (distance % stride != 0 ? 1 : 0));
}

/// True when @p call becomes a submitted task, in the sense codegen means it.
///
/// Matching codegen rather than "is this a call to a function": `system.task_dummy`
/// is an *operator*, and codegen emits `rt_submit_dummy_task` for it — a real
/// node in the recording. `ExpandManualPhaseFence` inserts those automatically,
/// so a Graph body carries nodes its author never wrote, and counting only
/// GlobalVar calls under-reports the topology.
[[nodiscard]] bool IsTaskLaunch(const CallPtr& call) {
  // Matching codegen rather than "is this a call to a function":
  // `system.task_dummy` is an *operator*, and codegen emits
  // `rt_submit_dummy_task` for it — a real node in the recording.
  // `ExpandManualPhaseFence` inserts those automatically, so a Graph body
  // carries nodes its author never wrote.
  return As<GlobalVar>(call->op_) != nullptr || IsOp(call, "system.task_dummy");
}

/// True when @p call is an allocation codegen batches into an `alloc_tensors`.
///
/// Counted separately from a launch because the mapping is not one-to-one: the
/// runtime records each `alloc_tensors` as one kernel-less node, and codegen
/// packs up to `kAllocTensorsArgs` adjacent creates into one call. `tensor.full`
/// is deliberately absent — orchestration codegen has no lowering for it and
/// rejects it as a misplaced tensor op, so calling it an allocation node here
/// would let a Graph past this pass and fail later inside codegen.
[[nodiscard]] bool IsAllocation(const CallPtr& call) { return IsOp(call, "tensor.create"); }

/// Saturating add/multiply, so a nested-loop product cannot wrap into a small
/// number and report a limit violation as a passing count.
[[nodiscard]] size_t SatAdd(size_t a, size_t b) {
  return a > std::numeric_limits<size_t>::max() - b ? std::numeric_limits<size_t>::max() : a + b;
}
[[nodiscard]] size_t SatMul(size_t a, size_t b) {
  if (a == 0 || b == 0) return 0;
  return a > std::numeric_limits<size_t>::max() / b ? std::numeric_limits<size_t>::max() : a * b;
}

/// Counts the tasks a Graph body launches, which become the recorded nodes.
///
/// A launch inside a loop is counted once per iteration, not once per call site:
/// `for i in pl.range(2000): self.kernel(...)` records two thousand nodes, and a
/// lexical count would wave it past the runtime's limit and produce a recording
/// the runtime then refuses to cache.
///
/// That forces the harder question the count alone hides. A recording is made on
/// the first call and replayed unchanged, so the topology must be the same on
/// every call — which it is not when the number of launches depends on a value
/// that changes between calls. A loop whose bounds are not compile-time
/// constants, a `while`, and a runtime `if` around a launch are therefore all
/// rejected rather than counted: the alternative is a graph recorded from call
/// one and silently replayed for call two, which is a wrong answer with no
/// diagnostic anywhere.
///
/// One post-order walk answers all of it. Each control-flow node runs its
/// subtree with a fresh accumulator and reads the result back, so "does this
/// contain a launch" is just "did the subtree count anything" — no separate
/// scan, and every node is visited exactly once. A loop's bounds are counted
/// inside its own scale, which over-counts a launch in a loop bound; that shape
/// does not survive the earlier passes, and over-counting can only make the
/// limit stricter, never miss a violation.
class GraphNodeCounter : public IRVisitor {
 public:
  GraphNodeCounter(FunctionPtr func, ProgramPtr program)
      : func_(std::move(func)), program_(std::move(program)) {}

  [[nodiscard]] size_t count() const { return count_; }

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    if (IsTaskLaunch(op)) count_ = SatAdd(count_, 1);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    IRVisitor::VisitExpr_(op);
    count_ = SatAdd(count_, 1);
  }

  /// Charges the allocations in a statement list the nodes codegen will emit.
  ///
  /// Exact, not an estimate. Codegen collects every eligible `tensor.create` in
  /// the list — an intervening launch does not close the batch — and packs them
  /// `kAllocTensorsArgs` to an `alloc_tensors`, one recorded node each. Two of
  /// its three ineligibility rules cannot fire on a Graph that reaches here: a
  /// shape reading a local is already rejected as non-constant, and an
  /// already-declared var cannot recur under SSA. The third is resolved through
  /// the same helper the emitter uses — an injected GM pipe buffer leaves the
  /// batch when its `core_num` reads a value defined earlier in this list — so
  /// the two cannot disagree about how many nodes the region has.
  void VisitStmt_(const SeqStmtsPtr& op) override {
    size_t batchable = 0;
    std::unordered_set<const Var*> locally_defined;
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
      const auto& stmt = op->stmts_[i];
      auto assign = As<AssignStmt>(stmt);
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (call && IsAllocation(call)) {
        batched_.insert(stmt.get());
        const bool joins_batch =
            !alloc_batching::IsInjectedGMPipeCreateVar(assign->var_) ||
            alloc_batching::GMPipeCreateJoinsBatch(op->stmts_, i, assign->var_, program_, locally_defined);
        if (joins_batch) {
          ++batchable;
        } else {
          count_ = SatAdd(count_, 1);
          locally_defined.insert(assign->var_.get());
        }
      } else if (assign && assign->var_) {
        locally_defined.insert(assign->var_.get());
      }
      VisitStmt(stmt);
    }
    count_ = SatAdd(count_, alloc_batching::BatchedAllocationNodes(batchable));
  }

  /// An allocation outside any statement list — a loop body that is a single
  /// assign, say — is a batch of one.
  void VisitStmt_(const AssignStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    auto call = As<Call>(op->value_);
    if (call && IsAllocation(call) && batched_.count(op.get()) == 0) count_ = SatAdd(count_, 1);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    const size_t per_iteration = CountSubtree([&] { IRVisitor::VisitStmt_(op); });
    if (per_iteration == 0) return;
    auto trips = StaticTripCount(op);
    CHECK_SPAN(trips.has_value(), op->span_)
        << "Graph function '" << func_->name_
        << "' launches tasks inside a loop whose trip count is not a compile-time constant. The "
           "recorded graph fixes the task topology on the first call and replays it unchanged, so a "
           "launch count that can differ between calls would silently replay the first call's "
           "topology. Give the loop constant bounds, or move it outside the Graph function.";
    count_ = SatAdd(count_, SatMul(per_iteration, *trips));
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    const size_t launched = CountSubtree([&] { IRVisitor::VisitStmt_(op); });
    CHECK_SPAN(launched == 0, op->span_)
        << "Graph function '" << func_->name_
        << "' launches tasks inside a while loop. Its iteration count is a runtime value, so the "
           "recorded topology would be whatever the first call happened to produce. Move the loop "
           "outside the Graph function.";
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    const size_t launched = CountSubtree([&] { IRVisitor::VisitStmt_(op); });
    CHECK_SPAN(launched == 0, op->span_)
        << "Graph function '" << func_->name_
        << "' launches tasks inside a conditional. Which branch ran on the first call is baked into "
           "the recording and replayed for every later call, so the condition would stop having any "
           "effect. Hoist the branch to the call site and give each arm its own Graph function.";
  }

 private:
  /// Run @p walk with a fresh accumulator and return what it counted.
  template <typename F>
  size_t CountSubtree(F&& walk) {
    const size_t saved = count_;
    count_ = 0;
    walk();
    const size_t subtree = count_;
    count_ = saved;
    return subtree;
  }

  FunctionPtr func_;
  ProgramPtr program_;
  size_t count_ = 0;
  std::unordered_set<const Stmt*> batched_;
};

/// Rejects allocations a recording cannot reproduce.
///
/// Two separate problems, both silent:
///
/// * `tensor.full` has no orchestration lowering at all — codegen rejects it as
///   a misplaced tensor op. Left unchecked, a Graph carrying one passes every
///   check here and fails inside codegen as an internal error.
/// * `tensor.create` is supported, but recording copies the `TensorCreateInfo`
///   shape into the node and derives the output's address from it; replay never
///   re-runs the body. A shape that reads a boundary scalar is therefore frozen
///   at the first call's value, and a later call with a larger extent gets the
///   first call's buffer — a wrong address layout, not a fallback. Only a shape
///   built from literals is replay-invariant.
class RegionAllocationChecker : public IRVisitor {
 public:
  explicit RegionAllocationChecker(FunctionPtr func) : func_(std::move(func)) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    CHECK_SPAN(!IsOp(op, "tensor.full"), op->span_)
        << "Graph function '" << func_->name_
        << "' calls tensor.full inside the region. Orchestration codegen has no lowering for it and "
           "rejects it as a misplaced tensor op, so the Graph would fail in codegen rather than here. "
           "Allocate in the caller and pass the tensor in as a pl.InOut parameter.";
    if (!IsOp(op, "tensor.create")) return;
    // Checked on the *result* type rather than the call arguments: the shape
    // arrives as one list-shaped operand, and the extents are what end up in the
    // TensorCreateInfo the recording copies.
    auto tensor = As<TensorType>(op->GetType());
    if (!tensor) return;
    for (const auto& extent : tensor->shape_) {
      CHECK_SPAN(IsLiteralScalarExpr(extent), op->span_)
          << "Graph function '" << func_->name_
          << "' allocates a tensor whose shape is not a compile-time constant. Recording copies the "
             "shape into the node and derives the buffer's address from it, and replay never re-runs "
             "the body — so a later call with a different extent would reuse the first call's "
             "buffer. Allocate it in the caller and pass it in as a pl.InOut parameter.";
    }
  }

 private:
  FunctionPtr func_;
};

/// Rejects a launch whose core count replay cannot reproduce.
///
/// Recording stores the launch spec into the node — `logical_block_num =
/// block_num` — and replay copies it straight back (`slot.logical_block_num =
/// source.logical_block_num`). Only boundary tensors and boundary *scalar slots*
/// are refreshed; the block count is not. A `core_num` read from a boundary
/// scalar is therefore frozen at the first call's value, so a later call asking
/// for more blocks silently schedules the first call's number and leaves the
/// rest of the work undone.
///
/// The effective core count is resolved the way codegen resolves it — the launch
/// carries it for a direct `pl.spmd_submit`, the callee for a scope-based
/// `pl.spmd` or Group wrapper — so the two agree on what is being checked.
class LaunchSpecChecker : public IRVisitor {
 public:
  LaunchSpecChecker(FunctionPtr func, ProgramPtr program)
      : func_(std::move(func)), program_(std::move(program)) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    Check(alloc_batching::EffectiveCoreNum(op, Callee(op->op_)), op->span_);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    IRVisitor::VisitExpr_(op);
    // `core_num_` is the Submit's own field; the wrapper attr is the fallback,
    // matching SubmitToCallView's surfacing of kAttrCoreNum.
    ExprPtr core_num = op->core_num_.value_or(nullptr);
    if (!core_num) {
      auto callee = Callee(op->op_);
      if (callee) core_num = callee->GetAttr<ExprPtr>(kAttrCoreNum, nullptr);
    }
    Check(core_num, op->span_);
  }

 private:
  [[nodiscard]] FunctionPtr Callee(const OpPtr& callee_op) const {
    auto gvar = As<GlobalVar>(callee_op);
    if (!gvar || !program_) return nullptr;
    return program_->GetFunction(gvar->name_);
  }

  void Check(const ExprPtr& core_num, const Span& span) const {
    if (!core_num || IsLiteralScalarExpr(core_num)) return;
    CHECK_SPAN(false, span)
        << "Graph function '" << func_->name_
        << "' launches a task whose core_num is not a compile-time constant. Recording copies the "
           "launch spec into the node and replay restores it unchanged — only boundary tensors and "
           "scalars are patched — so the first call's block count would be replayed for every later "
           "call, leaving the rest of the work unexecuted. Use a constant core_num inside the Graph, "
           "or move the launch to the call site.";
  }

  FunctionPtr func_;
  ProgramPtr program_;
};

void CheckRegionAllocations(const FunctionPtr& func) {
  RegionAllocationChecker checker(func);
  checker.VisitStmt(func->body_);
}

/// Rejects a Graph calling another Graph.
class NestedGraphChecker : public IRVisitor {
 public:
  NestedGraphChecker(FunctionPtr func, ProgramPtr program)
      : func_(std::move(func)), program_(std::move(program)) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    Check(op->op_, op->span_);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    IRVisitor::VisitExpr_(op);
    Check(op->op_, op->span_);
  }

 private:
  void Check(const OpPtr& callee_op, const Span& span) const {
    auto gvar = As<GlobalVar>(callee_op);
    if (!gvar || !program_) return;
    auto callee = program_->GetFunction(gvar->name_);
    if (!callee || callee->func_type_ != FunctionType::Graph) return;
    CHECK_SPAN(false, span) << "Graph function '" << func_->name_ << "' calls Graph function '"
                            << callee->name_
                            << "'. Nested graphs are not supported: the runtime cannot record a graph "
                               "from inside one it is already recording, so the inner call would make "
                               "the whole region uncacheable. Inline one of them, or call both from the "
                               "orchestration entry.";
  }

  FunctionPtr func_;
  ProgramPtr program_;
};

/// Validates a Graph function's signature against the runtime's boundary rules.
void CheckGraphSignature(const FunctionPtr& func) {
  size_t tensor_params = 0;
  size_t scalar_params = 0;
  // `param_directions_` is not guaranteed to be as long as `params_` at every
  // point in the pipeline — `window_externalization` guards the same access —
  // and a Graph reaches this check before the pass that would normalise it. A
  // missing entry is not a legality violation to report, so treat it as the
  // default `In` rather than reading out of bounds.
  for (size_t i = 0; i < func->params_.size(); ++i) {
    const auto& param = func->params_[i];
    const auto dir = i < func->param_directions_.size() ? func->param_directions_[i] : ParamDirection::In;
    if (IsScalarType(param->GetType())) {
      CHECK_SPAN(dir == ParamDirection::In, param->span_)
          << "Graph function '" << func->name_ << "' declares scalar parameter '" << param->name_hint_
          << "' as " << (dir == ParamDirection::Out ? "Out" : "InOut")
          << ". A boundary scalar is passed by value and replayed from the call site, so it can only "
             "be an input.";
      ++scalar_params;
      continue;
    }
    ++tensor_params;
    CHECK_SPAN(IsLegalGraphParamDirection(dir), param->span_)
        << "Graph function '" << func->name_ << "' declares tensor parameter '" << param->name_hint_
        << "' as Out, meaning the runtime allocates it. A recorded graph's boundary tensors must "
           "already exist so replay can patch their addresses, so allocate it at the call site and "
           "pass it in as InOut.";
  }

  CHECK_SPAN(tensor_params >= 1, func->span_)
      << "Graph function '" << func->name_
      << "' takes no tensor parameters. A graph with an empty boundary has nothing to patch on "
         "replay and the runtime refuses to cache it.";
  CHECK_SPAN(tensor_params <= kMaxBoundaryTensors, func->span_)
      << "Graph function '" << func->name_ << "' takes " << tensor_params
      << " tensor parameters, over the runtime's boundary limit of " << kMaxBoundaryTensors
      << ". Pack several of them into one larger tensor and slice it inside the region.";
  // The boundary is a fixed-size `GraphTaskArgs = Arg<GRAPH_MAX_TENSOR_ARGS,
  // GRAPH_MAX_SCALAR_ARGS>`, and Step A *adds* scalar parameters, so a signature
  // that fit before hoisting can stop fitting after. Checked after Step A has
  // run for that reason.
  CHECK_SPAN(scalar_params <= kMaxBoundaryScalars, func->span_)
      << "Graph function '" << func->name_ << "' takes " << scalar_params
      << " scalar parameters, over the runtime's boundary limit of " << kMaxBoundaryScalars
      << ". Some of these may have been added by hoisting values the body derived; compute them at "
         "the call site and pass fewer, or split the region.";
}

/// Rejects a Graph that returns a value the caller could consume.
///
/// `return c` where `c` is an InOut parameter is the DSL's spelling for writing
/// in place, and lowers to an alias rather than a value — that is fine. A
/// genuinely new value is not: `rt_submit_graph` yields a valid task id only on
/// a cache *hit*, so nothing downstream can depend on a graph call's result.
///
/// The parameter is matched by *lineage*, not by pointer identity. By the time
/// this pass runs, `OutlineIncoreScopes` has rewritten an in-place body into a
/// call, so the returned value is a rebind of the parameter rather than the
/// parameter node:
///
///     c_1 = layer_incore_0(a, c)
///     return c_1
///
/// That is the shape *every* Graph body with a device scope has here, so an
/// identity match would reject the whole feature while still passing a unit
/// test that stops before the outliner. `ReturnedParamIndices` follows SSA
/// rebinds and recurses through the callee, and yields nullopt for a genuinely
/// computed value — including a scalar, which it resolves only when the return
/// literally is a scalar param.
void CheckGraphReturns(const FunctionPtr& func, const ProgramPtr& program) {
  auto return_stmt = return_lineage::FindFirstReturn(func->body_);
  if (!return_stmt || return_stmt->value_.empty()) return;

  const auto returned_params = return_lineage::ReturnedParamIndices(func, program);
  for (size_t i = 0; i < return_stmt->value_.size(); ++i) {
    const bool writes_back_a_param = i < returned_params.size() && returned_params[i].has_value();
    CHECK_SPAN(writes_back_a_param, return_stmt->span_)
        << "Graph function '" << func->name_ << "' returns a value it computed rather than one of its "
        << "own parameters. A graph call is a task launch whose result is a recording handle — valid "
           "only once the graph is already cached — so nothing can depend on it. Write the result "
           "into an InOut parameter and return that parameter instead.";
  }
}

/// Validates a Graph call site.
class GraphCallSiteChecker : public IRVisitor {
 public:
  GraphCallSiteChecker(FunctionPtr caller, ProgramPtr program)
      : caller_(std::move(caller)), program_(std::move(program)) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    auto callee = LookupGraph(op->op_);
    if (!callee) return;
    CheckArity(op->args_, callee, op->span_);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    IRVisitor::VisitExpr_(op);
    auto callee = LookupGraph(op->op_);
    if (!callee) return;
    CheckArity(op->args_, callee, op->span_);

    CHECK_SPAN(op->deps_.empty(), op->span_)
        << "Graph function '" << callee->name_ << "' is submitted from '" << caller_->name_
        << "' with explicit dependencies. An explicit dependency edge makes the launch uncacheable, "
           "so the region would silently run as ordinary tasks with no graph replay. Order the graph "
           "against its producers through its boundary tensors instead.";
    CHECK_SPAN(op->predicate_ == nullptr, op->span_)
        << "Graph function '" << callee->name_ << "' is submitted from '" << caller_->name_
        << "' with a dispatch predicate. A predicate on a graph launch is neither honoured nor "
           "rejected by the runtime — it is silently zeroed — so the region would run "
           "unconditionally.";
  }

 private:
  [[nodiscard]] FunctionPtr LookupGraph(const OpPtr& callee_op) const {
    auto gvar = As<GlobalVar>(callee_op);
    if (!gvar || !program_) return nullptr;
    auto callee = program_->GetFunction(gvar->name_);
    if (!callee || callee->func_type_ != FunctionType::Graph) return nullptr;
    return callee;
  }

  void CheckArity(const std::vector<ExprPtr>& args, const FunctionPtr& callee, const Span& span) const {
    // A Submit may normally pass a prefix of the callee's parameters, letting
    // the runtime allocate the tail Out params. A Graph has no such tail —
    // CheckGraphSignature already rejected Out params — so every parameter must
    // be supplied here.
    CHECK_SPAN(args.size() == callee->params_.size(), span)
        << "Graph function '" << callee->name_ << "' is called from '" << caller_->name_ << "' with "
        << args.size() << " argument(s) but declares " << callee->params_.size()
        << " parameter(s). A graph cannot leave outputs for the runtime to allocate, so every "
           "parameter must be passed at the call site.";
  }

  FunctionPtr caller_;
  ProgramPtr program_;
};

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Builds the hoisting plan for one Graph function, or an empty plan.
///
/// Tensors are appended before scalars so the resulting signature keeps
/// `CoreTaskArgs`' tensor-before-scalar ordering, which the runtime enforces.
[[nodiscard]] GraphPlan BuildPlan(const FunctionPtr& func) {
  GraphPlan plan;
  plan.name = func->name_;
  if (!func->body_) return plan;

  DerivedScalarCollector collector(func);
  collector.VisitStmt(func->body_);

  auto add = [&plan](const VarPtr& var, const ExprPtr& value, ParamDirection dir, bool is_tensor) {
    auto param = std::make_shared<Var>(var->name_hint_, var->GetType(), var->span_);
    plan.hoisted.push_back(HoistedValue{param, var, value, dir, is_tensor});
  };

  // Step B: a hoisted view takes the direction of the boundary tensor it views.
  //
  // Not unconditionally `In`: a region may store *through* a view back into the
  // parameter it came from. Declared `In`, codegen emits `add_input(view)`, the
  // graph launch is never registered as a writer of that buffer, and a consumer
  // downstream can be scheduled against the pre-write contents. A view cannot
  // carry wider access than the tensor it views, so the root's own direction is
  // both sufficient and never an under-declaration.
  const auto& roots = collector.tensor_root();
  std::unordered_map<const Var*, ParamDirection> param_direction;
  for (size_t i = 0; i < func->params_.size(); ++i) {
    param_direction[func->params_[i].get()] =
        i < func->param_directions_.size() ? func->param_directions_[i] : ParamDirection::In;
  }
  for (const auto& [var, value] : collector.slices()) {
    ParamDirection dir = ParamDirection::In;
    auto root = roots.find(var.get());
    if (root != roots.end()) {
      auto it = param_direction.find(root->second);
      if (it != param_direction.end()) dir = it->second;
    }
    add(var, value, dir, /*is_tensor=*/true);
  }
  // Step A: scalars last, after every tensor.
  for (const auto& [var, value] : collector.derived()) {
    add(var, value, ParamDirection::In, /*is_tensor=*/false);
  }
  plan.passthrough = collector.passthrough();
  plan.definition_index = collector.definition_index();
  return plan;
}

/// Rewrites every call site of a planned Graph to supply the hoisted scalars.
class CallSiteExtender : public IRMutator {
 public:
  CallSiteExtender(ProgramPtr program, const std::unordered_map<std::string, GraphPlan>& plans)
      : program_(std::move(program)), plans_(plans) {}

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto call = As<Call>(base);
    if (!call) return base;
    const auto* plan = LookupPlan(call->op_);
    if (plan == nullptr) return call;

    auto new_args = AppendHoistedArgs(*plan, call->op_, call->args_);
    auto attrs = call->attrs_;
    if (call->HasArgDirections()) {
      attrs =
          WithArgDirectionsAttr(std::move(attrs), AppendHoistedDirections(call->GetArgDirections(), *plan));
    }
    return std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, std::move(attrs),
                                  call->GetType(), call->span_);
  }

  ExprPtr VisitExpr_(const SubmitPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto submit = As<Submit>(base);
    if (!submit) return base;
    const auto* plan = LookupPlan(submit->op_);
    if (plan == nullptr) return submit;

    auto new_args = AppendHoistedArgs(*plan, submit->op_, submit->args_);
    auto attrs = submit->attrs_;
    if (submit->HasArgDirections()) {
      attrs =
          WithArgDirectionsAttr(std::move(attrs), AppendHoistedDirections(submit->GetArgDirections(), *plan));
    }
    return std::make_shared<Submit>(submit->op_, std::move(new_args), submit->deps_, submit->kwargs_,
                                    std::move(attrs), submit->GetType(), submit->span_, submit->core_num_,
                                    submit->sync_start_, submit->allow_early_resolve_, submit->predicate_);
  }

  /// Splice any bindings synthesised while rewriting this statement's
  /// expression in ahead of it.
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    pending_prefix_.clear();
    auto stmt = IRMutator::VisitStmt_(op);
    return WithPrefix(std::move(stmt), op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    pending_prefix_.clear();
    auto stmt = IRMutator::VisitStmt_(op);
    return WithPrefix(std::move(stmt), op->span_);
  }

 private:
  [[nodiscard]] StmtPtr WithPrefix(StmtPtr stmt, const Span& span) {
    if (pending_prefix_.empty()) return stmt;
    std::vector<StmtPtr> stmts = std::move(pending_prefix_);
    pending_prefix_.clear();
    stmts.push_back(std::move(stmt));
    return SeqStmts::Flatten(std::move(stmts), span);
  }

  [[nodiscard]] const GraphPlan* LookupPlan(const OpPtr& op) const {
    auto gvar = As<GlobalVar>(op);
    if (!gvar || !program_) return nullptr;
    auto it = plans_.find(gvar->name_);
    return it == plans_.end() ? nullptr : &it->second;
  }

  /// Rebuild each hoisted expression against this call site's actual arguments.
  ///
  /// Each rebuilt value is bound to a fresh local rather than inlined into the
  /// argument list: orchestration codegen accepts only a Var or a literal as an
  /// argument, so an inline `i * 5120` would fail there. The binding statements
  /// accumulate in `pending_prefix_` and the statement overrides splice them in
  /// immediately before the call.
  [[nodiscard]] std::vector<ExprPtr> AppendHoistedArgs(const GraphPlan& plan, const OpPtr& callee_op,
                                                       const std::vector<ExprPtr>& old_args) {
    auto gvar = As<GlobalVar>(callee_op);
    auto callee = program_->GetFunction(gvar->name_);
    INTERNAL_CHECK(callee) << "Internal error: planned Graph '" << plan.name << "' not found in program";

    std::unordered_map<const Var*, ExprPtr> binding;  // extended per hoist below
    const size_t bound = std::min(old_args.size(), callee->params_.size());
    for (size_t i = 0; i < bound; ++i) binding[callee->params_[i].get()] = old_args[i];

    // Replay every hoist and every alias in **definition order**.
    //
    // The body is SSA, so definition order is dependency order: a slice's offset
    // scalar is defined before the slice, an alias before whatever reads it.
    // Binding in that one order therefore satisfies every cross-reference —
    // scalar naming an earlier alias, tensor alias naming an earlier hoisted
    // view, hoisted view whose offset is an earlier hoisted scalar:
    //
    //     base  = idx * 128        // hoisted scalar
    //     alias = w                // alias of a tensor *parameter*
    //     wl    = slice(alias, base)   // hoisted view, reads both
    //
    // Splitting this into "scalars, then tensors" left `alias` unbound when the
    // tensor phase substituted `wl`, so the caller kept a name only the Graph
    // has and the printer marked it `__FREE_VAR`.
    //
    // Parameter order is unaffected: `plan.hoisted` still lists tensors before
    // scalars, and the argument append below walks that list, not this one.
    std::unordered_map<const Var*, ExprPtr> local_for;
    auto bind = [&](const HoistedValue& h) {
      auto value = SubstituteAtCallSite(h.value, binding);
      // Bound to a local, not spliced in as an expression: codegen emits a
      // boundary scalar as `const uint64_t&` into the argument slot, so the
      // value needs an addressable home at the call site.
      auto local = std::make_shared<Var>(h.param->name_hint_ + "__graph_arg" + std::to_string(next_local_++),
                                         h.param->GetType(), h.param->span_);
      pending_prefix_.push_back(std::make_shared<AssignStmt>(local, value, h.param->span_));
      local_for[h.param.get()] = local;
      // A later hoist may reference this one: the expressions were captured
      // from the original body, where these were locals rather than parameters.
      binding[h.original.get()] = local;
    };
    auto bind_alias = [&](const std::pair<const Var*, VarPtr>& a) {
      auto it = binding.find(a.second.get());
      if (it != binding.end()) binding[a.first] = it->second;
    };

    auto position = [&plan](const Var* var) {
      auto it = plan.definition_index.find(var);
      // A hoist the collector never indexed sorts last; it cannot be named by
      // anything, so any position after the indexed ones is correct.
      return it == plan.definition_index.end() ? std::numeric_limits<size_t>::max() : it->second;
    };
    // `plan.hoisted` is in *parameter* order (tensors before scalars), which is
    // not definition order, so the replay walks its own sorted view of it.
    std::vector<const HoistedValue*> by_definition;
    by_definition.reserve(plan.hoisted.size());
    for (const auto& h : plan.hoisted) by_definition.push_back(&h);
    std::stable_sort(by_definition.begin(), by_definition.end(),
                     [&position](const HoistedValue* a, const HoistedValue* b) {
                       return position(a->original.get()) < position(b->original.get());
                     });

    size_t next_hoist = 0;
    size_t next_alias = 0;
    while (next_hoist < by_definition.size() || next_alias < plan.passthrough.size()) {
      const bool alias_is_next =
          next_alias < plan.passthrough.size() &&
          position(plan.passthrough[next_alias].first) < position(by_definition[next_hoist]->original.get());
      const bool take_alias = next_hoist == by_definition.size() || alias_is_next;
      if (take_alias) {
        bind_alias(plan.passthrough[next_alias++]);
      } else {
        bind(*by_definition[next_hoist++]);
      }
    }

    std::vector<ExprPtr> args = old_args;
    args.reserve(args.size() + plan.hoisted.size());
    for (const auto& h : plan.hoisted) {
      args.push_back(local_for.at(h.param.get()));
    }
    return args;
  }

  /// Mirror the hoisted parameters' directions onto the call's direction attr,
  /// in the same order `AppendHoistedArgs` appends the arguments.
  [[nodiscard]] static std::vector<ArgDirection> AppendHoistedDirections(
      const std::vector<ArgDirection>& old_dirs, const GraphPlan& plan) {
    std::vector<ArgDirection> dirs = old_dirs;
    for (const auto& h : plan.hoisted) {
      if (!h.is_tensor) {
        dirs.push_back(ArgDirection::Scalar);
      } else if (h.direction == ParamDirection::InOut) {
        dirs.push_back(ArgDirection::InOut);
      } else {
        dirs.push_back(ArgDirection::Input);
      }
    }
    return dirs;
  }

  ProgramPtr program_;
  const std::unordered_map<std::string, GraphPlan>& plans_;
  std::vector<StmtPtr> pending_prefix_;
  int next_local_ = 0;
};

[[nodiscard]] ProgramPtr TransformProgram(const ProgramPtr& program) {
  if (!program) return program;

  bool has_graph = false;
  for (const auto& [gvar, func] : program->functions_) {
    if (func && func->func_type_ == FunctionType::Graph) {
      has_graph = true;
      break;
    }
  }
  if (!has_graph) return program;

  // A Graph only exists under host_build_graph. Codegen emits `GraphTaskArgs`
  // and `rt_submit_graph` unconditionally, and the tensormap_and_ringbuffer
  // `orchestration_api.h` declares neither — so compiling a Graph against the
  // default runtime produces orchestration C++ that names undeclared symbols and
  // fails in the C++ compiler, pointing at generated code rather than at the
  // cause. Reported here instead, where the function the user wrote can be named.
  const auto* ctx = PassContext::Current();
  const RuntimeKind runtime = ctx != nullptr ? ctx->GetRuntime() : kDefaultRuntimeKind;
  if (runtime != RuntimeKind::HostBuildGraph) {
    for (const auto& [gvar, func] : program->functions_) {
      if (!func || func->func_type_ != FunctionType::Graph) continue;
      CHECK_SPAN(false, func->span_)
          << "Graph function '" << func->name_ << "' requires the host_build_graph runtime, but this "
          << "compilation targets '" << RuntimeKindToName(runtime)
          << "'. `rt_submit_graph` and `GraphTaskArgs` exist only in that runtime's "
             "orchestration API, so the generated orchestration would not compile. Pass "
             "`runtime=RuntimeKind.HOST_BUILD_GRAPH` to `ir.compile` (or set it on the enclosing "
             "`PassContext`), or drop `type=pl.FunctionType.Graph` from the function.";
    }
  }

  // Step D, part 1: whole-function properties, before anything is rewritten so
  // diagnostics name the user's own signature.
  for (const auto& [gvar, func] : program->functions_) {
    if (!func || func->func_type_ != FunctionType::Graph) continue;
    CheckGraphSignature(func);
    if (!func->body_) continue;

    CheckRegionAllocations(func);

    LaunchSpecChecker launch_spec(func, program);
    launch_spec.VisitStmt(func->body_);

    NestedGraphChecker nested(func, program);
    nested.VisitStmt(func->body_);

    CheckGraphReturns(func, program);

    GraphNodeCounter counter(func, program);
    counter.VisitStmt(func->body_);
    CHECK_SPAN(counter.count() >= 1, func->span_)
        << "Graph function '" << func->name_
        << "' launches no tasks. `graph_execution_storage_layout` refuses a node count of zero, so "
           "the region would never be cached; call it directly instead of marking it a Graph.";
    CHECK_SPAN(counter.count() <= kMaxGraphNodes, func->span_)
        << "Graph function '" << func->name_ << "' launches " << counter.count()
        << " tasks, over the runtime's per-graph limit of " << kMaxGraphNodes
        << ". Split the region into several graphs.";
  }

  // Step A: plan, then rewrite bodies and call sites together.
  std::unordered_map<std::string, GraphPlan> plans;
  for (const auto& [gvar, func] : program->functions_) {
    if (!func || func->func_type_ != FunctionType::Graph) continue;
    auto plan = BuildPlan(func);
    if (!plan.hoisted.empty()) plans.emplace(func->name_, std::move(plan));
  }

  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> rewritten;
  for (const auto& [gvar, func] : program->functions_) {
    auto it = (func == nullptr) ? plans.end() : plans.find(func->name_);
    if (it == plans.end() || func->func_type_ != FunctionType::Graph) {
      rewritten[gvar] = func;
      continue;
    }
    HoistedValueRewriter rewriter(it->second);
    rewritten[gvar] = ExtendGraphSignature(func, it->second, rewriter.VisitStmt(func->body_));
  }
  auto with_signatures = std::make_shared<Program>(std::move(rewritten), program->name_, program->span_);

  CallSiteExtender extender(with_signatures, plans);
  auto result = extender.VisitProgram(with_signatures);

  // Step A, part 2 and Step D, part 2: run on the *rewritten* program, so what
  // is checked is what codegen will see.
  for (const auto& [gvar, func] : result->functions_) {
    if (!func || !func->body_) continue;
    if (func->func_type_ == FunctionType::Graph) {
      // Step A *adds* parameters, so the boundary capacity has to be re-checked
      // against the signature codegen will actually see: a Graph at exactly the
      // scalar limit before hoisting is over it afterwards. Part 1 runs the same
      // check first only so a signature the user wrote is reported against the
      // user's own line.
      CheckGraphSignature(func);
      std::unordered_set<const Var*> hoistable;  // all hoistable values are now params
      UnhoistableScalarChecker checker(func, hoistable);
      checker.VisitStmt(func->body_);
      // Also after the rewrite: Step A turns a derived scalar into a parameter,
      // which does not make it replay-stable as a *launch spec* — only as an
      // argument.
      LaunchSpecChecker launch_spec(func, result);
      launch_spec.VisitStmt(func->body_);
      continue;
    }
    GraphCallSiteChecker call_checker(func, result);
    call_checker.VisitStmt(func->body_);
  }

  return result;
}

}  // namespace

namespace pass {

Pass LegalizeGraphBoundary() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr { return TransformProgram(program); };
  return CreateProgramPass(pass_func, "LegalizeGraphBoundary", kLegalizeGraphBoundaryProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

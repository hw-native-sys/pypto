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
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Runtime hard limits, from runtime/src/common/host_build_graph/.
/// A violation of either is a compile-time error here, because at runtime it
/// only makes the recording non-cacheable and the graph silently falls back.
constexpr size_t kMaxBoundaryTensors = 32;  ///< graph_cache.h
constexpr size_t kMaxGraphNodes = 1024;     ///< graph_execution.h

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

// ---------------------------------------------------------------------------
// Step A — hoisting derived boundary scalars
// ---------------------------------------------------------------------------

/// One scalar value the body derives and the call sites must supply instead.
struct HoistedScalar {
  VarPtr param;     ///< Fresh Scalar parameter appended to the Graph signature.
  VarPtr original;  ///< Body variable it replaces.
  ExprPtr value;    ///< Defining expression, in terms of the Graph's own params.
};

/// What Step A decided for one Graph function.
struct GraphPlan {
  std::string name;
  std::vector<HoistedScalar> hoisted;
  /// Body variables that were hoisted, for erasing their now-dead definitions.
  std::unordered_set<const Var*> hoisted_vars;
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
      if (IsScalarType(param->GetType())) scalar_params_.insert(param.get());
    }
  }

  /// Body variables bound to a derivable expression, in definition order.
  [[nodiscard]] const std::vector<std::pair<VarPtr, ExprPtr>>& derived() const { return derived_; }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    auto var = AsVarLike(op->var_);
    if (!var || !IsScalarType(var->GetType())) return;
    if (!IsDerivable(op->value_)) return;
    // A bare parameter reference is already a pass-through; only a *computed*
    // value needs hoisting.
    if (AsVarLike(op->value_) != nullptr) {
      passthrough_.insert(var.get());
      return;
    }
    derived_vars_.insert(var.get());
    derived_.emplace_back(var, op->value_);
  }

 private:
  /// True when every leaf of `expr` is a scalar parameter, an already-derived
  /// scalar, or a constant, and every interior node is scalar arithmetic.
  ///
  /// Scalar arithmetic is not `Call`: `a * b` is a `Mul` node. All ~28 operator
  /// nodes derive from `BinaryExpr` / `UnaryExpr`, so a `dynamic_pointer_cast`
  /// to those bases covers the whole family at once. (`As<T>` would not: it
  /// matches one exact ObjectKind, and each operator has its own.)
  [[nodiscard]] bool IsDerivable(const ExprPtr& expr) const {
    if (!expr) return false;
    if (As<ConstInt>(expr) || As<ConstFloat>(expr) || As<ConstBool>(expr)) return true;
    if (auto var = AsVarLike(expr)) {
      return scalar_params_.count(var.get()) != 0 || derived_vars_.count(var.get()) != 0 ||
             passthrough_.count(var.get()) != 0;
    }
    if (auto bin = std::dynamic_pointer_cast<const BinaryExpr>(expr)) {
      return IsDerivable(bin->left_) && IsDerivable(bin->right_);
    }
    if (auto un = std::dynamic_pointer_cast<const UnaryExpr>(expr)) {
      return IsDerivable(un->operand_);
    }
    // Anything else — a tensor read, a task output, a runtime query — has no
    // meaning at the call site.
    return false;
  }

  FunctionPtr func_;
  std::unordered_set<const Var*> scalar_params_;
  std::unordered_set<const Var*> derived_vars_;
  std::unordered_set<const Var*> passthrough_;
  std::vector<std::pair<VarPtr, ExprPtr>> derived_;
};

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
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    CheckArgs(op->args_, op->span_);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    IRVisitor::VisitExpr_(op);
    CheckArgs(op->args_, op->span_);
  }

 private:
  void CheckArgs(const std::vector<ExprPtr>& args, const Span& span) const {
    for (const auto& arg : args) {
      if (!arg || !IsScalarType(arg->GetType())) continue;
      auto var = AsVarLike(arg);
      if (!var) continue;
      if (scalar_params_.count(var.get()) != 0 || hoistable_.count(var.get()) != 0) continue;
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
};

/// Replaces hoisted body variables with their new parameters and erases the
/// assignments that used to compute them.
class HoistedScalarRewriter : public IRMutator {
 public:
  explicit HoistedScalarRewriter(const GraphPlan& plan) {
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
  // Appended, not prepended: `CoreTaskArgs` requires every tensor argument to
  // precede every scalar one, and these are scalars.
  for (const auto& h : plan.hoisted) {
    params.push_back(h.param);
    dirs.push_back(ParamDirection::In);
  }
  return std::make_shared<Function>(func->name_, std::move(params), std::move(dirs), func->return_types_,
                                    new_body, func->span_, func->func_type_, func->level_, func->role_,
                                    func->attrs_, func->requires_runtime_binding_);
}

// ---------------------------------------------------------------------------
// Step D — boundary legality
// ---------------------------------------------------------------------------

/// Counts the tasks a Graph body launches, which become the recorded nodes.
class TaskLaunchDetector : public IRVisitor {
 public:
  [[nodiscard]] bool found() const { return found_; }

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    if (As<GlobalVar>(op->op_)) found_ = true;
  }

  void VisitExpr_(const SubmitPtr& op) override {
    IRVisitor::VisitExpr_(op);
    found_ = true;
  }

 private:
  bool found_ = false;
};

[[nodiscard]] bool ContainsTaskLaunch(const StmtPtr& stmt) {
  if (!stmt) return false;
  TaskLaunchDetector detector;
  detector.VisitStmt(stmt);
  return detector.found();
}

/// Static trip count of @p op, or nullopt when it is not provable here.
[[nodiscard]] std::optional<size_t> StaticTripCount(const ForStmtPtr& op) {
  const auto* start = As<ConstInt>(op->start_) ? &As<ConstInt>(op->start_)->value_ : nullptr;
  const auto* stop = As<ConstInt>(op->stop_) ? &As<ConstInt>(op->stop_)->value_ : nullptr;
  const auto* step = As<ConstInt>(op->step_) ? &As<ConstInt>(op->step_)->value_ : nullptr;
  if (!start || !stop || !step || *step == 0) return std::nullopt;
  const int64_t span = *stop - *start;
  if ((*step > 0 && span <= 0) || (*step < 0 && span >= 0)) return 0;
  const int64_t stride = *step > 0 ? *step : -*step;
  const int64_t distance = span > 0 ? span : -span;
  return static_cast<size_t>((distance + stride - 1) / stride);
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
class GraphNodeCounter : public IRVisitor {
 public:
  explicit GraphNodeCounter(FunctionPtr func) : func_(std::move(func)) {}

  [[nodiscard]] size_t count() const { return count_; }

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    if (As<GlobalVar>(op->op_)) count_ += multiplier_;
  }

  void VisitExpr_(const SubmitPtr& op) override {
    IRVisitor::VisitExpr_(op);
    count_ += multiplier_;
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!ContainsTaskLaunch(op->body_)) {
      IRVisitor::VisitStmt_(op);
      return;
    }
    auto trips = StaticTripCount(op);
    CHECK_SPAN(trips.has_value(), op->span_)
        << "Graph function '" << func_->name_
        << "' launches tasks inside a loop whose trip count is not a compile-time constant. The "
           "recorded graph fixes the task topology on the first call and replays it unchanged, so a "
           "launch count that can differ between calls would silently replay the first call's "
           "topology. Give the loop constant bounds, or move it outside the Graph function.";
    // Saturate rather than wrap: a nested-loop product can overflow size_t into a
    // small number, which would report a limit violation as a passing count.
    const size_t outer = multiplier_;
    multiplier_ = (*trips != 0 && outer > std::numeric_limits<size_t>::max() / *trips)
                      ? std::numeric_limits<size_t>::max()
                      : outer * *trips;
    IRVisitor::VisitStmt_(op);
    multiplier_ = outer;
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    CHECK_SPAN(!ContainsTaskLaunch(op->body_), op->span_)
        << "Graph function '" << func_->name_
        << "' launches tasks inside a while loop. Its iteration count is a runtime value, so the "
           "recorded topology would be whatever the first call happened to produce. Move the loop "
           "outside the Graph function.";
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    const bool guarded_launch =
        ContainsTaskLaunch(op->then_body_) || (op->else_body_ && ContainsTaskLaunch(*op->else_body_));
    CHECK_SPAN(!guarded_launch, op->span_)
        << "Graph function '" << func_->name_
        << "' launches tasks inside a conditional. Which branch ran on the first call is baked into "
           "the recording and replayed for every later call, so the condition would stop having any "
           "effect. Hoist the branch to the call site and give each arm its own Graph function.";
    IRVisitor::VisitStmt_(op);
  }

 private:
  FunctionPtr func_;
  size_t count_ = 0;
  size_t multiplier_ = 1;
};

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
  for (size_t i = 0; i < func->params_.size(); ++i) {
    const auto& param = func->params_[i];
    const auto dir = func->param_directions_[i];
    if (IsScalarType(param->GetType())) {
      CHECK_SPAN(dir == ParamDirection::In, param->span_)
          << "Graph function '" << func->name_ << "' declares scalar parameter '" << param->name_hint_
          << "' as " << (dir == ParamDirection::Out ? "Out" : "InOut")
          << ". A boundary scalar is passed by value and replayed from the call site, so it can only "
             "be an input.";
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
}

/// Rejects a Graph returning anything other than one of its own parameters.
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

/// Builds the Step A plan for one Graph function, or an empty plan.
[[nodiscard]] GraphPlan BuildPlan(const FunctionPtr& func) {
  GraphPlan plan;
  plan.name = func->name_;
  if (!func->body_) return plan;

  DerivedScalarCollector collector(func);
  collector.VisitStmt(func->body_);

  for (const auto& [var, value] : collector.derived()) {
    auto param = std::make_shared<Var>(var->name_hint_, var->GetType(), var->span_);
    plan.hoisted.push_back(HoistedScalar{param, var, value});
    plan.hoisted_vars.insert(var.get());
  }
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
      attrs = WithArgDirectionsAttr(std::move(attrs),
                                    AppendScalarDirections(call->GetArgDirections(), plan->hoisted.size()));
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
      attrs = WithArgDirectionsAttr(std::move(attrs),
                                    AppendScalarDirections(submit->GetArgDirections(), plan->hoisted.size()));
    }
    return std::make_shared<Submit>(submit->op_, std::move(new_args), submit->deps_, submit->kwargs_,
                                    std::move(attrs), submit->GetType(), submit->span_, submit->core_num_,
                                    submit->sync_start_, submit->allow_early_resolve_, submit->predicate_);
  }

 private:
  [[nodiscard]] const GraphPlan* LookupPlan(const OpPtr& op) const {
    auto gvar = As<GlobalVar>(op);
    if (!gvar || !program_) return nullptr;
    auto it = plans_.find(gvar->name_);
    return it == plans_.end() ? nullptr : &it->second;
  }

  /// Rebuild each hoisted expression against this call site's actual arguments.
  [[nodiscard]] std::vector<ExprPtr> AppendHoistedArgs(const GraphPlan& plan, const OpPtr& callee_op,
                                                       const std::vector<ExprPtr>& old_args) const {
    auto gvar = As<GlobalVar>(callee_op);
    auto callee = program_->GetFunction(gvar->name_);
    INTERNAL_CHECK(callee) << "Internal error: planned Graph '" << plan.name << "' not found in program";

    std::unordered_map<const Var*, ExprPtr> binding;
    const size_t bound = std::min(old_args.size(), callee->params_.size());
    for (size_t i = 0; i < bound; ++i) binding[callee->params_[i].get()] = old_args[i];

    std::vector<ExprPtr> args = old_args;
    args.reserve(args.size() + plan.hoisted.size());
    for (const auto& h : plan.hoisted) {
      auto value = SubstituteAtCallSite(h.value, binding);
      args.push_back(value);
      // A later hoist may be defined in terms of this one (`base = idx * 128`
      // then `end = base + 128`). The expressions were captured from the body,
      // where `base` is a local whose definition Step A erases — so without
      // binding it here the call site would name a variable that no longer
      // exists anywhere. `plan.hoisted` is in definition order, so one forward
      // pass suffices.
      binding[h.original.get()] = value;
    }
    return args;
  }

  [[nodiscard]] static std::vector<ArgDirection> AppendScalarDirections(
      const std::vector<ArgDirection>& old_dirs, size_t count) {
    std::vector<ArgDirection> dirs = old_dirs;
    dirs.insert(dirs.end(), count, ArgDirection::Scalar);
    return dirs;
  }

  ProgramPtr program_;
  const std::unordered_map<std::string, GraphPlan>& plans_;
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

  // Step D, part 1: whole-function properties, before anything is rewritten so
  // diagnostics name the user's own signature.
  for (const auto& [gvar, func] : program->functions_) {
    if (!func || func->func_type_ != FunctionType::Graph) continue;
    CheckGraphSignature(func);
    if (!func->body_) continue;

    NestedGraphChecker nested(func, program);
    nested.VisitStmt(func->body_);

    CheckGraphReturns(func, program);

    GraphNodeCounter counter(func);
    counter.VisitStmt(func->body_);
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
    HoistedScalarRewriter rewriter(it->second);
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
      std::unordered_set<const Var*> hoistable;  // all hoistable values are now params
      UnhoistableScalarChecker checker(func, hoistable);
      checker.VisitStmt(func->body_);
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

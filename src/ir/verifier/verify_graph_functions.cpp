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
 * @file verify_graph_functions.cpp
 * @brief Verifies the GraphBoundaryLegalized property.
 *
 * `LegalizeGraphBoundary` rejects illegal graphs as it rewrites them. This
 * verifier re-states the resulting invariants over the whole program, so a later
 * pass that reintroduces a violation is caught rather than silently producing a
 * program the runtime declines to record.
 *
 * That matters more here than for a typical property. Almost every
 * host_build_graph constraint degrades to a *silent* non-graph fallback in a
 * release build: the program stays numerically correct and simply loses the
 * speedup, which no correctness test can detect. This verifier is the automated
 * detector.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/alloc_batching.h"
#include "pypto/ir/transforms/utils/graph_replay_invariant.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

constexpr size_t kMaxBoundaryTensors = 128;  ///< GRAPH_MAX_TENSOR_ARGS, host_build_graph/runtime/types.h
constexpr size_t kMaxBoundaryScalars = 64;   ///< GRAPH_MAX_SCALAR_ARGS, host_build_graph/runtime/types.h
constexpr size_t kMaxGraphNodes = 1024;      ///< GRAPH_MAX_NODES, common/host_build_graph/graph_execution.h

/// Local copies, not shared with the pass on purpose — see GraphBodyChecker.
[[nodiscard]] size_t SatAdd(size_t a, size_t b) {
  return a > std::numeric_limits<size_t>::max() - b ? std::numeric_limits<size_t>::max() : a + b;
}

[[nodiscard]] size_t SatMul(size_t a, size_t b) {
  if (a == 0 || b == 0) return 0;
  return a > std::numeric_limits<size_t>::max() / b ? std::numeric_limits<size_t>::max() : a * b;
}

[[nodiscard]] std::optional<size_t> StaticTripCount(const ForStmtPtr& op) {
  auto start_c = As<ConstInt>(op->start_);
  auto stop_c = As<ConstInt>(op->stop_);
  auto step_c = As<ConstInt>(op->step_);
  if (!start_c || !stop_c || !step_c || step_c->value_ == 0) return std::nullopt;
  const int64_t start = start_c->value_;
  const int64_t stop = stop_c->value_;
  const int64_t step = step_c->value_;
  if (step > 0 ? stop <= start : stop >= start) return 0;
  // Unsigned: the signed span, the negation of step and the round-up all
  // overflow int64_t on extreme bounds.
  const auto u_start = static_cast<uint64_t>(start);
  const auto u_stop = static_cast<uint64_t>(stop);
  const auto u_step = static_cast<uint64_t>(step);
  const uint64_t distance = step > 0 ? u_stop - u_start : u_start - u_stop;
  const uint64_t stride = step > 0 ? u_step : uint64_t{0} - u_step;
  return static_cast<size_t>(distance / stride + (distance % stride != 0 ? 1 : 0));
}

/// True when @p call becomes a submitted task, matching what codegen emits.
/// `system.task_dummy` is an operator that lowers to `rt_submit_dummy_task`, and
/// `ExpandManualPhaseFence` inserts them, so counting only GlobalVar calls
/// under-reports the recorded topology.
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

/// Walks one function body and reports every illegal reference to a Graph.
///
/// Covers both call-like kinds. `Submit` is not a subclass of `Call` and
/// `IRVisitor` dispatches them through separate handlers, so a checker that
/// overrides only the `Call` path silently skips every `pl.submit(...)` — and a
/// Graph is launched from a manual scope precisely as a submit.
class GraphReferenceChecker : public IRVisitor {
 public:
  GraphReferenceChecker(ProgramPtr program, FunctionPtr caller, std::vector<Diagnostic>& diagnostics)
      : program_(std::move(program)), caller_(std::move(caller)), diagnostics_(diagnostics) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    auto callee = LookupGraph(op->op_);
    if (!callee) return;
    CheckCaller(callee, op->span_);
    CheckLaunchArity(callee, op->args_.size(), op->span_);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    IRVisitor::VisitExpr_(op);
    auto callee = LookupGraph(op->op_);
    if (!callee) return;
    CheckCaller(callee, op->span_);
    CheckLaunchArity(callee, op->args_.size(), op->span_);

    if (!op->deps_.empty()) {
      Report(callee, op->span_,
             "is submitted with explicit dependencies. An explicit dependency edge makes the launch "
             "uncacheable, so the region would silently run as ordinary tasks with no graph replay. "
             "Order the graph against its producers through its boundary tensors instead.");
    }
    if (op->predicate_ != nullptr) {
      Report(callee, op->span_,
             "is submitted with a dispatch predicate. The runtime neither honours nor rejects a "
             "predicate on a graph launch — it silently zeroes it — so the region would run "
             "unconditionally.");
    }
  }

 private:
  [[nodiscard]] FunctionPtr LookupGraph(const OpPtr& callee_op) const {
    auto gvar = As<GlobalVar>(callee_op);
    if (!gvar || !program_) return nullptr;
    auto callee = program_->GetFunction(gvar->name_);
    if (!callee || callee->func_type_ != FunctionType::Graph) return nullptr;
    return callee;
  }

  /// A Graph launch must supply every parameter, `Submit` or not.
  ///
  /// The usual `Submit` prefix rule lets the runtime allocate the tail `Out`
  /// params, and a Graph has no such tail: `rt_graph_args_cacheable` refuses a
  /// boundary carrying a runtime-allocated output outright. Without this, a
  /// later rewrite that drops an argument leaves the property still verifying.
  void CheckLaunchArity(const FunctionPtr& callee, size_t argc, const Span& span) const {
    if (argc == callee->params_.size()) return;
    Report(callee, span,
           "is launched with " + std::to_string(argc) + " arguments for " +
               std::to_string(callee->params_.size()) +
               " parameters. A graph boundary has no runtime-allocated tail, so every parameter "
               "must be supplied at the call.");
  }

  void CheckCaller(const FunctionPtr& callee, const Span& span) const {
    // Opaque is accepted alongside Orchestration: an entry function is Opaque
    // until OutlineIncoreScopes promotes it, and this verifier also runs on IR
    // that has not reached that point.
    if (caller_->func_type_ == FunctionType::Orchestration || caller_->func_type_ == FunctionType::Opaque) {
      return;
    }
    if (caller_->func_type_ == FunctionType::Graph) {
      Report(callee, span,
             "is called from another Graph function. The runtime cannot record a graph from inside "
             "one it is already recording, so the whole region becomes uncacheable.");
      return;
    }
    Report(callee, span,
           "is called from a '" + FunctionTypeToString(caller_->func_type_) +
               "' function. A graph is a task launch, so only an Orchestration entry may call it.");
  }

  void Report(const FunctionPtr& callee, const Span& span, const std::string& what) const {
    std::ostringstream oss;
    oss << "Graph function '" << callee->name_ << "', referenced from '" << caller_->name_ << "', " << what;
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "GraphBoundaryLegalized", 0, oss.str(), span);
  }

  ProgramPtr program_;
  FunctionPtr caller_;
  std::vector<Diagnostic>& diagnostics_;
};

/// Re-states the signature half of the boundary contract.
void VerifySignature(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) {
  auto report = [&](const std::string& message, const Span& span) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "GraphBoundaryLegalized", 0,
                             "Graph function '" + func->name_ + "' " + message, span);
  };

  size_t tensor_params = 0;
  size_t scalar_params = 0;
  // `param_directions_` may be shorter than `params_` at some points in the
  // pipeline, so a missing entry reads as the default `In` rather than out of
  // bounds — matching the guard in `window_externalization`.
  for (size_t i = 0; i < func->params_.size(); ++i) {
    const auto& param = func->params_[i];
    const auto dir = i < func->param_directions_.size() ? func->param_directions_[i] : ParamDirection::In;
    if (As<ScalarType>(param->GetType()) != nullptr) {
      if (dir != ParamDirection::In) {
        report("declares scalar parameter '" + param->name_hint_ +
                   "' as an output. A boundary scalar is passed by value and replayed from the call "
                   "site, so it can only be an input.",
               param->span_);
      }
      ++scalar_params;
      continue;
    }
    // Only a tensor is a tensor boundary. `GenerateGraphFunctions` binds a
    // parameter through `args.tensor(i).ref()` when `AsTensorTypeLike` holds and
    // through `args.scalar(k)` otherwise, so counting every non-scalar as a
    // tensor would let a Tile, Tuple or pointer-like parameter verify as a
    // boundary tensor and then take the scalar ABI in codegen.
    if (!AsTensorTypeLike(param->GetType())) {
      report("declares parameter '" + param->name_hint_ +
                 "' as neither a tensor nor a scalar. A Graph boundary carries only tensors and "
                 "scalars; codegen would bind this one through the scalar argument slot.",
             param->span_);
      continue;
    }
    ++tensor_params;
    if (dir == ParamDirection::Out) {
      report("declares tensor parameter '" + param->name_hint_ +
                 "' as Out, meaning the runtime allocates it. A recorded graph's boundary tensors "
                 "must already exist so replay can patch their addresses.",
             param->span_);
    }
  }

  if (tensor_params == 0) {
    report(
        "takes no tensor parameters. A graph with an empty boundary has nothing to patch on replay "
        "and the runtime refuses to cache it.",
        func->span_);
  }
  if (tensor_params > kMaxBoundaryTensors) {
    report("takes " + std::to_string(tensor_params) +
               " tensor parameters, over the runtime's boundary "
               "limit of " +
               std::to_string(kMaxBoundaryTensors) + ".",
           func->span_);
  }
  if (scalar_params > kMaxBoundaryScalars) {
    report("takes " + std::to_string(scalar_params) +
               " scalar parameters, over the runtime's boundary limit of " +
               std::to_string(kMaxBoundaryScalars) + ".",
           func->span_);
  }
}

/// Re-derives the body half of the contract: topology, node count, hoisting.
///
/// Deliberately not shared with `LegalizeGraphBoundary`'s own counter. A verifier
/// that calls the pass's code can only report what the pass already decided; the
/// point is to re-read the IR and disagree if a later pass reintroduced a state
/// the pass had ruled out.
///
/// The hoisting post-condition here is *stricter* than the rule the pass applies,
/// and easier to check: once Step A has run, every scalar a task consumes must
/// have somewhere for replay to get its value — a boundary parameter's slot, or
/// a value replay reproduces identically. Whether it was originally derivable is
/// the pass's problem; that nothing call-varying survives is the property.
///
/// `ReplayInvariantSet` is the one piece shared with the pass rather than
/// re-derived. It is not a decision the pass made and this file rubber-stamps —
/// it is a reading of the runtime's own contract (a boundary shape is pinned by
/// `graph_boundary_matches`, a constant-trip loop replays the same literals). Two
/// hand-written copies of that reading could disagree, and a verifier that
/// disagreed with the pass would reject IR the pass just produced.
class GraphBodyChecker : public IRVisitor {
 public:
  GraphBodyChecker(ProgramPtr program, FunctionPtr func, std::vector<Diagnostic>& diagnostics)
      : program_(std::move(program)), func_(std::move(func)), diagnostics_(diagnostics), invariant_(func_) {
    for (const auto& param : func_->params_) {
      if (As<ScalarType>(param->GetType()) != nullptr) {
        scalar_params_.insert(param.get());
      } else {
        tensor_params_.insert(param.get());
        tensor_root_.insert(param.get());  // a parameter is its own boundary root
      }
    }
    invariant_.Collect(func_->body_);
  }

  [[nodiscard]] size_t count() const { return count_; }

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    CheckAllocation(op);
    CheckLaunchSpec(alloc_batching::EffectiveCoreNum(op, Callee(op->op_)), op->span_);
    CheckBoundaryView(op);
    if (!IsTaskLaunch(op)) return;
    count_ = SatAdd(count_, 1);
    CheckArity(op->op_, op->args_.size(), /*is_submit=*/false, op->span_);
    CheckScalarArgs(op->args_, op->span_);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    IRVisitor::VisitExpr_(op);
    ExprPtr submit_core_num = op->core_num_.value_or(nullptr);
    if (!submit_core_num) {
      auto callee = Callee(op->op_);
      if (callee) submit_core_num = callee->GetAttr<ExprPtr>(kAttrCoreNum, nullptr);
    }
    CheckLaunchSpec(submit_core_num, op->span_);
    count_ = SatAdd(count_, 1);
    CheckArity(op->op_, op->args_.size(), /*is_submit=*/true, op->span_);
    CheckScalarArgs(op->args_, op->span_);
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
  /// Boundary provenance: a parameter is its own root, an alias inherits one.
  void TrackTensorAlias(const AssignStmtPtr& op) {
    auto var = AsVarLike(op->var_);
    if (!var || As<ScalarType>(var->GetType()) != nullptr) return;
    auto aliased = AsVarLike(op->value_);
    if (aliased && tensor_root_.count(aliased.get()) != 0) tensor_root_.insert(var.get());
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    TrackTensorAlias(op);
    auto call = As<Call>(op->value_);
    if (call && IsAllocation(call) && batched_.count(op.get()) == 0) count_ = SatAdd(count_, 1);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    const size_t per_iteration = CountSubtree([&] { IRVisitor::VisitStmt_(op); });
    if (per_iteration == 0) return;
    auto trips = StaticTripCount(op);
    if (!trips.has_value()) {
      Report("launches tasks inside a loop whose trip count is not a compile-time constant.", op->span_);
      return;
    }
    count_ = SatAdd(count_, SatMul(per_iteration, *trips));
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (CountSubtree([&] { IRVisitor::VisitStmt_(op); }) != 0) {
      Report("launches tasks inside a while loop, whose iteration count is a runtime value.", op->span_);
    }
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (CountSubtree([&] { IRVisitor::VisitStmt_(op); }) != 0) {
      Report(
          "launches tasks inside a conditional, so the recorded topology depends on the branch "
          "the first call happened to take.",
          op->span_);
    }
  }

 private:
  /// Re-derives what the region may allocate.
  ///
  /// `tensor.full` has no orchestration lowering at all, and a `tensor.create`
  /// whose shape reads a boundary scalar is frozen: recording copies the shape
  /// into the node and derives the buffer's address from it, and replay never
  /// re-runs the body. Both are re-checked here rather than trusted from the
  /// pass, so a later rewrite that reintroduces either does not verify clean.
  void CheckAllocation(const CallPtr& op) {
    if (IsOp(op, "tensor.full")) {
      Report("calls tensor.full inside the region; orchestration codegen has no lowering for it.", op->span_);
      return;
    }
    if (!IsOp(op, "tensor.create")) return;
    // `AsTensorTypeLike` for the same reason the pass uses it, and for the same
    // reason the boundary-parameter check above does: it is the test codegen
    // applies when deciding what a tensor is.
    auto tensor = AsTensorTypeLike(op->GetType());
    if (!tensor) return;
    for (const auto& extent : tensor->shape_) {
      if (IsLiteralScalarExpr(extent)) continue;
      Report(
          "allocates a tensor whose shape is not a compile-time constant; recording freezes the "
          "shape and replay never re-runs the body.",
          op->span_);
      return;
    }
  }

  [[nodiscard]] FunctionPtr Callee(const OpPtr& callee_op) const {
    auto gvar = As<GlobalVar>(callee_op);
    if (!gvar || !program_) return nullptr;
    return program_->GetFunction(gvar->name_);
  }

  /// A launch spec is frozen into the recorded node, not patched on replay.
  ///
  /// Replay restores `slot.logical_block_num = source.logical_block_num` from the
  /// Definition and refreshes only boundary tensors and scalar slots, so a
  /// `core_num` reading a boundary scalar replays the first call's block count
  /// and silently leaves the rest of the work unscheduled.
  void CheckLaunchSpec(const ExprPtr& core_num, const Span& span) {
    if (!core_num || IsLiteralScalarExpr(core_num)) return;
    Report(
        "launches a task whose core_num is not a compile-time constant; recording freezes the launch "
        "spec into the node and replay restores it unchanged.",
        span);
  }

  /// Step B's post-condition: no boundary view survives whose window can move.
  ///
  /// Recording classifies such a view as a `BOUNDARY_VIEW`; replay restores
  /// `start_offset = boundary.start_offset + packed_offset` with `packed_offset`
  /// and the shape/strides taken from the *first* call's node, patching only the
  /// address. A view whose offset or extent can differ on a later call therefore
  /// replays call one's window against call two's buffer — silently.
  ///
  /// Stated on the window rather than on the pass's collection rule, so it holds
  /// however that rule changes: a view left in place with a replay-invariant
  /// window is not a leak, it is the only correct outcome for one the call site
  /// cannot name.
  void CheckBoundaryView(const CallPtr& op) {
    if (!IsOp(op, "tensor.slice") && !IsOp(op, "tensor.view")) return;
    if (op->args_.empty()) return;
    auto source = AsVarLike(op->args_[0]);
    // Through aliases, not just direct parameters: `alias = w; slice(alias, ...)`
    // is a boundary view too, and checking only the immediate source lets it
    // verify clean while the recording freezes the first call's offset.
    if (!source || tensor_root_.count(source.get()) == 0) return;
    if (HasReplayInvariantWindow(op)) return;
    Report("takes a view of boundary tensor '" + source->name_hint_ +
               "' inside the region whose window can differ between calls; replay patches the "
               "boundary address but keeps the offset and shape recorded on the first call, so the "
               "view must be taken at the call site.",
           op->span_);
  }

  /// True when both halves of @p op's window are frozen at the right values.
  [[nodiscard]] bool HasReplayInvariantWindow(const CallPtr& op) const {
    for (size_t i = 1; i < op->args_.size(); ++i) {
      if (!invariant_.IsInvariant(op->args_[i])) return false;
    }
    auto viewed = As<TensorType>(op->GetType());
    if (!viewed) return false;
    for (const auto& extent : viewed->shape_) {
      if (!invariant_.IsInvariant(extent)) return false;
    }
    return true;
  }

  void Report(const std::string& message, const Span& span) {
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "GraphBoundaryLegalized", 0,
                              "Graph function '" + func_->name_ + "' " + message, span);
  }

  /// A `Submit`'s args are a positional *prefix* of the callee's params — the
  /// tail is runtime-allocated `Out` params the runtime fills in. Requiring
  /// equality there would reject legal IR; only a plain `Call` covers every
  /// param. (The stricter "supply every parameter" rule belongs on launches *of*
  /// a Graph, which `GraphReferenceChecker` checks, not on the ordinary tasks a
  /// Graph body launches.)
  void CheckArity(const OpPtr& callee_op, size_t argc, bool is_submit, const Span& span) {
    auto gvar = As<GlobalVar>(callee_op);
    if (!gvar || !program_) return;
    auto callee = program_->GetFunction(gvar->name_);
    if (!callee) return;
    const size_t params = callee->params_.size();
    if (argc > params || (!is_submit && argc != params)) {
      Report("launches '" + callee->name_ + "' with " + std::to_string(argc) + " arguments for " +
                 std::to_string(params) + " parameters.",
             span);
      return;
    }
    // A Submit may stop short only where the tail it omits is entirely `Out` —
    // those are the runtime-allocated outputs it lets the runtime fill in. An
    // omitted `In` or `InOut` is a missing argument, and checking only the count
    // lets it through here and fails much later, as an INTERNAL_CHECK in
    // codegen's direction handling.
    for (size_t i = argc; i < params; ++i) {
      if (i < callee->param_directions_.size() && callee->param_directions_[i] == ParamDirection::Out) {
        continue;
      }
      Report("launches '" + callee->name_ + "' without supplying '" + callee->params_[i]->name_hint_ +
                 "', which is not an Out parameter. A Submit may omit only a runtime-allocated Out "
                 "tail.",
             span);
      return;
    }
  }

  void CheckScalarArgs(const std::vector<ExprPtr>& args, const Span& span) {
    for (const auto& arg : args) {
      if (!arg || As<ScalarType>(arg->GetType()) == nullptr) continue;
      // A TaskId is topology, not a boundary scalar — see graph_replay_invariant.h.
      if (graph_replay::IsTaskIdScalar(arg->GetType())) continue;
      if (IsLiteralScalarExpr(arg)) continue;
      auto var = AsVarLike(arg);
      // Only a parameter itself, never a rename of one: codegen emits a
      // surviving alias as a value copy, whose address is not the argument slot
      // the recording anchors, so the copy is frozen at the first call's value.
      // Step A deletes those, and one reaching here is a real leak.
      if (var && scalar_params_.count(var.get()) != 0) continue;
      // No slot, but nothing to patch either: replay recomputes this value
      // identically, so the copy frozen into the recording is the right one.
      if (invariant_.IsInvariant(arg)) continue;
      Report(
          "passes a scalar to a task that has no argument slot and can differ between calls, so the "
          "runtime would freeze the first call's value into the recording.",
          span);
    }
  }

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

  ProgramPtr program_;
  FunctionPtr func_;
  std::vector<Diagnostic>& diagnostics_;
  graph_replay::ReplayInvariantSet invariant_;
  std::unordered_set<const Var*> scalar_params_;
  std::unordered_set<const Var*> tensor_params_;
  /// Tensor vars that derive from a boundary parameter, directly or by alias.
  std::unordered_set<const Var*> tensor_root_;
  size_t count_ = 0;
  std::unordered_set<const Stmt*> batched_;
};

/// Re-states that a Graph hands nothing computed back to its caller.
void VerifyReturns(const FunctionPtr& func, const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) {
  auto return_stmt = return_lineage::FindFirstReturn(func->body_);
  if (!return_stmt || return_stmt->value_.empty()) return;
  const auto returned_params = return_lineage::ReturnedParamIndices(func, program);
  for (size_t i = 0; i < return_stmt->value_.size(); ++i) {
    if (i < returned_params.size() && returned_params[i].has_value()) continue;
    diagnostics.emplace_back(
        DiagnosticSeverity::Error, "GraphBoundaryLegalized", 0,
        "Graph function '" + func->name_ +
            "' returns a value it computed rather than one of its own parameters. A graph call's "
            "result is a recording handle, valid only on a cache hit, so nothing can depend on it.",
        return_stmt->span_);
  }
}

class GraphBoundaryLegalizedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "GraphBoundaryLegalized"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func) continue;
      if (func->func_type_ == FunctionType::Graph) {
        VerifySignature(func, diagnostics);
        if (func->body_) {
          VerifyReturns(func, program, diagnostics);
          GraphBodyChecker body_checker(program, func, diagnostics);
          body_checker.VisitStmt(func->body_);
          if (body_checker.count() == 0) {
            diagnostics.emplace_back(DiagnosticSeverity::Error, "GraphBoundaryLegalized", 0,
                                     "Graph function '" + func->name_ +
                                         "' launches no tasks; the runtime refuses a node count of zero.",
                                     func->span_);
          }
          if (body_checker.count() > kMaxGraphNodes) {
            diagnostics.emplace_back(
                DiagnosticSeverity::Error, "GraphBoundaryLegalized", 0,
                "Graph function '" + func->name_ + "' launches " + std::to_string(body_checker.count()) +
                    " tasks, over the runtime's per-graph limit of " + std::to_string(kMaxGraphNodes) + ".",
                func->span_);
          }
        }
      }
      if (!func->body_) continue;
      GraphReferenceChecker checker(program, func, diagnostics);
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateGraphBoundaryLegalizedPropertyVerifier() {
  return std::make_shared<GraphBoundaryLegalizedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

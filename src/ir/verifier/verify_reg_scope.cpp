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
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/hardware_loop_analysis.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/reg_scope.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto::ir {
namespace {
/// Two linear walks separate ownership from uses, including uses preceding an
/// illegal definition. Maps use Var identity, never spelling: rebinding a name
/// to a Tile result must not accidentally authorize an earlier Reg escape.
class RegionChecker : public IRVisitor {
  const Call* statement_barrier_ = nullptr;

 public:
  RegionChecker(FunctionPtr function, std::vector<Diagnostic>& diagnostics, bool final)
      : function_(std::move(function)), diagnostics_(diagnostics), final_(final) {}

  void Run() {
    for (const auto& p : function_->params_) {
      owners_[p.get()] = nullptr;
      if (ContainsRegionValue(p->GetType(), false)) Error("Reg cannot cross a function boundary", p->span_);
    }
    for (const auto& t : function_->return_types_) {
      if (ContainsRegionValue(t, false)) {
        Error("Reg cannot cross a function return boundary", function_->span_);
      }
    }
    collecting_ = true;
    VisitStmt(function_->body_);
    collecting_ = false;
    VisitStmt(function_->body_);
  }

 protected:
  void VisitStmt_(const RegScopeStmtPtr& op) override {
    if (collecting_) {
      if (region_) Error("RegScope cannot nest", op->span_);
      if (!incore_ || blocked_ || function_->func_type_ == FunctionType::AIC) {
        Error("RegScope requires an InCore/AIV context", op->span_);
      }
      if (final_ && function_->func_type_ != FunctionType::AIV && function_->func_type_ != FunctionType::VF) {
        Error("RegScope final placement requires AIV after ExpandMixedKernel", op->span_);
      }
      if (backend::BackendConfig::IsConfigured() &&
          backend::BackendConfig::GetBackendType() != backend::BackendType::Ascend950) {
        Error("RegScope requires an A5 (Ascend950) target", op->span_);
      }
    }
    auto previous = region_;
    region_ = op.get();
    IRVisitor::VisitStmt_(op);
    region_ = previous;
  }
  void VisitStmt_(const InCoreScopeStmtPtr& op) override {
    bool prev = incore_;
    bool previous_blocked = blocked_;
    // Wrapper Group/Spmd functions cannot contain RegScope directly. An
    // explicit InCore scope still establishes a legal pre-outlining context.
    // Match the existing outliner's orchestration-like owners: this is a
    // device boundary even while its containing function still orchestrates.
    incore_ = IsOrchestrationLike(function_->func_type_) || function_->func_type_ == FunctionType::Opaque ||
              function_->func_type_ == FunctionType::Inline ||
              function_->func_type_ == FunctionType::InCore || function_->func_type_ == FunctionType::AIV;
    // An orchestration ancestor is outside this device region. Scopes opened
    // inside it still set blocked_ through Block(); restore the outer state
    // afterward so sibling orchestration code cannot inherit device access.
    blocked_ = false;
    IRVisitor::VisitStmt_(op);
    blocked_ = previous_blocked;
    incore_ = prev;
  }
  void VisitStmt_(const RuntimeScopeStmtPtr& op) override { Block(op); }
  void VisitStmt_(const GraphScopeStmtPtr& op) override { Block(op); }
  void VisitStmt_(const CommDomainScopeStmtPtr& op) override { Block(op); }
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto call = As<Call>(op->value_); call && IsOp(call, "reg.mem_bar")) {
      Error("reg.mem_bar is statement-only", op->span_);
    }
    if (collecting_) {
      auto type = op->var_->GetType();
      if (ContainsRegionValue(type, true)) {
        auto [it, inserted] = owners_.emplace(op->var_.get(), region_);
        if (!inserted && it->second != region_) {
          Error("Reg/local Scalar assignment escapes RegScope ownership", op->span_);
        }
      }
    }
    if (!collecting_ &&
        (ContainsRegionValue(op->var_->GetType(), false) ||
         ContainsRegionValue(op->value_->GetType(), false)) &&
        !structural_equal(op->var_->GetType(), op->value_->GetType())) {
      Error("Register assignment dtype/kind/lane mismatch", op->span_);
    }
    if (collecting_ && region_ && As<TileType>(op->var_->GetType())) {
      auto call = As<Call>(op->value_);
      if (!IsOp(call, "reg.store") && !IsOp(call, "reg.scatter") && !IsOp(call, "reg.squeeze_store_flush")) {
        Error(
            "RegScope Tile inputs must originate outside the region; only reg.store/reg.scatter return Tile",
            op->span_);
      }
    }
    // LHS is a definition, not a use. Only inspect the RHS here.
    VisitExpr(op->value_);
  }
  void VisitVarLike_(const VarPtr& op) override {
    if (collecting_) return;
    auto found = owners_.find(op.get());
    if (found != owners_.end() && found->second && found->second != region_) {
      Error("Reg/local Scalar cannot escape its RegScope", op->span_);
    }
    if (storing_container_ && found != owners_.end() && found->second && found->second == region_) {
      Error("Reg/local Scalar cannot escape through an external container", op->span_);
    }
    if (ContainsRegionValue(op->GetType(), false) &&
        (found == owners_.end() || !region_ || found->second != region_)) {
      Error("Reg must be defined and used within the same RegScope", op->span_);
    }
  }
  void VisitExpr_(const TupleGetItemExprPtr& op) override {
    // Codegen resolves projections from the SSA results of an assigned pair.
    // A nested call has no assignment to emit or bind those results first.
    if (collecting_ && As<RegType>(op->GetType()) && !AsVarLike(op->tuple_)) {
      Error("Register tuple projections require a bound tuple variable; bind the paired call before indexing",
            op->span_);
    }
    IRVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const CallPtr& op) override {
    if (!collecting_ && IsOp(op, "reg.mem_bar") && op.get() != statement_barrier_) {
      Error("reg.mem_bar is statement-only", op->span_);
    }
    if (!collecting_ && op->op_->name_.rfind("reg.", 0) == 0) {
      if (!region_) Error("reg operation requires RegScope", op->span_);
      // Raw Call construction can bypass type deduction. Revalidate the actual
      // operands and compare the supplied result rather than trusting the type.
      try {
        auto inferred =
            OpRegistry::GetInstance().Create(op->op_->name_, op->args_, op->kwargs_, op->span_)->GetType();
        if (!structural_equal(inferred, op->GetType())) {
          Error("Register call result type mismatch", op->span_);
        }
      } catch (const pypto::Error& error) {
        Error(error.what(), op->span_);
      }
    }
    bool previous = storing_container_;
    if (region_ && IsOp(op, "array.update_element")) storing_container_ = true;
    IRVisitor::VisitExpr_(op);
    storing_container_ = previous;
  }
  void VisitStmt_(const EvalStmtPtr& op) override {
    if (auto call = As<Call>(op->expr_);
        call && call->op_->name_.rfind("reg.", 0) == 0 && !IsOp(call, "reg.mem_bar")) {
      Error("Register calls require AssignStmt; bare store is not allowed", op->span_);
    }
    const auto* previous = statement_barrier_;
    if (auto call = As<Call>(op->expr_); call && IsOp(call, "reg.mem_bar")) {
      statement_barrier_ = call.get();
    }
    IRVisitor::VisitStmt_(op);
    statement_barrier_ = previous;
  }
  void VisitStmt_(const IfStmtPtr& op) override {
    const auto previous = yield_region_;
    yield_region_ = region_;
    if (collecting_) {
      for (const auto& var : op->return_vars_) RecordControlFlowValue(var);
    }
    IRVisitor::VisitStmt_(op);
    yield_region_ = previous;
  }
  void VisitStmt_(const ForStmtPtr& op) override {
    // Require is VF-local intent, including for empty loops. Check containment
    // before visiting the body: a RegScope inside a loop cannot authorize it.
    if (collecting_ && !region_ && op->hardware_loop_policy_ == HardwareLoopPolicy::Require) {
      Error("HardwareLoopPolicy.Require requires RegScope", op->span_);
    }
    if (collecting_ && op->hardware_loop_policy_ == HardwareLoopPolicy::Require &&
        !HasCanonicalHardwareLoopBounds(op)) {
      Error("HardwareLoopPolicy.Require requires constant start=0 and step=1 with INDEX dtype", op->span_);
    }
    if (collecting_ && op->hardware_loop_policy_ == HardwareLoopPolicy::Require && !op->attrs_.empty()) {
      Error("HardwareLoopPolicy.Require does not support loop attrs", op->span_);
    }
    const auto induction_type = As<ScalarType>(op->loop_var_->GetType());
    if (collecting_ && op->hardware_loop_policy_ == HardwareLoopPolicy::Require &&
        (!induction_type || induction_type->dtype_ != DataType::INDEX)) {
      Error("Hardware loop induction variable must have INDEX dtype", op->span_);
    }
    // Enforce target-independent hardware-loop limits during the existing IR
    // walk. No target profile or repeated per-loop body analysis is needed.
    const bool hardware_loop = op->hardware_loop_policy_ == HardwareLoopPolicy::Require;
    if (hardware_loop) ++hardware_loop_depth_;
    const bool previous_empty = empty_hardware_loop_;
    const auto exact_stop =
        collecting_ && !previous_empty ? arith::TryEvaluateIntConstant(op->stop_) : std::nullopt;
    const auto start = exact_stop ? arith::TryEvaluateIntConstant(op->start_) : std::nullopt;
    const auto step = start ? arith::TryEvaluateIntConstant(op->step_) : std::nullopt;
    // Empty sequential ancestors suppress hardware realization requirements
    // even when the ancestor uses Auto policy rather than Require.
    if (exact_stop && start && step &&
        ((*step > 0 && *start >= *exact_stop) || (*step < 0 && *start <= *exact_stop))) {
      empty_hardware_loop_ = true;
    }
    if (collecting_ && hardware_loop && !previous_empty) {
      if (!empty_hardware_loop_) {
        if (hardware_loop_depth_ > 4) {
          Error("HWL_DEPTH: hardware loops can nest at most four levels", op->span_);
        }
        if (exact_stop && *exact_stop > 65535) {
          Error("Hardware loop stop must not exceed 65535", op->span_);
        }
      }
    }
    const auto previous = yield_region_;
    yield_region_ = region_;
    if (collecting_) {
      owners_.emplace(op->loop_var_.get(), region_);
      for (const auto& arg : op->iter_args_) RecordControlFlowValue(arg);
      for (const auto& var : op->return_vars_) RecordControlFlowValue(var);
    }
    IRVisitor::VisitStmt_(op);
    if (hardware_loop) --hardware_loop_depth_;
    empty_hardware_loop_ = previous_empty;
    yield_region_ = previous;
  }
  void VisitStmt_(const WhileStmtPtr& op) override {
    // Check lexical containment, including while nested under if/for. The
    // two ownership walks must report this unsupported statement only once.
    if (collecting_ && region_) Error("WhileStmt is not supported inside RegScope", op->span_);
    const auto previous = yield_region_;
    yield_region_ = region_;
    if (collecting_) {
      for (const auto& arg : op->iter_args_) RecordControlFlowValue(arg);
      for (const auto& var : op->return_vars_) RecordControlFlowValue(var);
    }
    IRVisitor::VisitStmt_(op);
    yield_region_ = previous;
  }
  void VisitStmt_(const YieldStmtPtr& op) override {
    auto previous = region_;
    region_ = yield_region_;
    IRVisitor::VisitStmt_(op);
    region_ = previous;
  }
  void VisitStmt_(const BreakStmtPtr& op) override {
    if (collecting_ && hardware_loop_depth_ > 0) {
      Error("BreakStmt is not supported inside hardware_loop", op->span_);
    }
  }
  void VisitStmt_(const ContinueStmtPtr& op) override {
    if (collecting_ && hardware_loop_depth_ > 0) {
      Error("ContinueStmt is not supported inside hardware_loop", op->span_);
    }
  }
  void VisitStmt_(const ReturnStmtPtr& op) override {
    if (collecting_ && region_) {
      Error("ReturnStmt is not supported inside RegScope", op->span_);
    }
    auto previous = region_;
    region_ = nullptr;
    IRVisitor::VisitStmt_(op);
    region_ = previous;
  }

 private:
  bool ContainsRegionValue(const TypePtr& type, bool include_scalar) {
    const auto key = std::make_pair(type.get(), include_scalar);
    if (auto found = region_types_.find(key); found != region_types_.end()) {
      return found->second;
    }
    bool result = As<RegType>(type) || As<SqueezeStoreType>(type) || (include_scalar && As<ScalarType>(type));
    if (auto tuple = As<TupleType>(type); tuple && !result) {
      for (const auto& element : tuple->types_) {
        if (ContainsRegionValue(element, include_scalar)) {
          result = true;
          break;
        }
      }
    }
    // Shared tuple types must not be rescanned at every use in a large body.
    region_types_.emplace(key, result);
    return result;
  }

  void RecordControlFlowValue(const VarPtr& var) {
    // Tuple aliases are only lowered within a straight-line block. SCF results
    // and carries must expose individual registers, including nested tuples.
    if (region_ && As<TupleType>(var->GetType()) && ContainsRegionValue(var->GetType(), false)) {
      Error("Register tuples cannot cross control-flow joins or loop carries; unpack the tuple first",
            var->span_);
    }
    if (ContainsRegionValue(var->GetType(), true)) owners_.emplace(var.get(), region_);
  }
  template <typename T>
  void Block(const std::shared_ptr<const T>& op) {
    bool prev = blocked_;
    if (collecting_ && region_) Error("Orchestration scopes cannot appear inside RegScope", op->span_);
    blocked_ = true;
    IRVisitor::VisitStmt_(op);
    blocked_ = prev;
  }
  void Error(const std::string& message, const Span& span) {
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "RegScopeValid", 0, message, span);
  }
  FunctionPtr function_;
  std::vector<Diagnostic>& diagnostics_;
  bool final_;
  int hardware_loop_depth_ = 0;
  bool empty_hardware_loop_ = false;
  // Inline helpers inherit execution placement from their eventual caller.
  // The final verifier still requires AIV after all helpers are expanded.
  bool incore_ = (IsInCoreType(function_->func_type_) && function_->func_type_ != FunctionType::AIC) ||
                 function_->func_type_ == FunctionType::Inline || function_->func_type_ == FunctionType::VF;
  bool collecting_ = true;
  bool blocked_ = false;
  bool storing_container_ = false;
  const RegScopeStmt* region_ = nullptr;
  const RegScopeStmt* yield_region_ = nullptr;
  std::map<const Var*, const RegScopeStmt*> owners_;
  std::map<std::pair<const Type*, bool>, bool> region_types_;
};

// Track equivalent pure register producers without expanding their expression
// DAGs. In particular, independently synthesized full/prefix masks can become
// one operand under PTOAS CSE. Loads stay distinct because they observe memory.
class StreamValueCanonicalizer : public IRMutator {
 public:
  void Clear() {
    var_remap_.clear();
    producers_.clear();
  }

  void Record(const VarPtr& var, const ExprPtr& value) {
    if (!As<RegType>(var->GetType())) return;
    if (AsVarLike(value)) {
      var_remap_[var.get()] = VisitExpr(value);
      return;
    }
    auto call = As<Call>(value);
    if (!call || IsOp(call, "reg.load") || IsOp(call, "reg.gather")) return;
    auto normalized = VisitExpr(value);
    auto& bucket = producers_[structural_hash(normalized)];
    for (const auto& [producer, representative] : bucket) {
      // Hashes only index candidates; exact equality resolves collisions.
      if (structural_equal(producer, normalized)) {
        var_remap_[var.get()] = representative;
        return;
      }
    }
    bucket.emplace_back(normalized, var);
  }

 private:
  std::map<uint64_t, std::vector<std::pair<ExprPtr, VarPtr>>> producers_;
};

// A stream token is linear state, not a reusable register value. Check its
// structured lifetime before optimizations and again at the codegen boundary.
// This walk follows SSA loop carries without enumerating loop iterations.
class SqueezeStreamChecker : public IRVisitor {
 public:
  explicit SqueezeStreamChecker(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

 protected:
  void VisitStmt_(const RegScopeStmtPtr& scope) override {
    class StreamFinder : public IRVisitor {
     public:
      bool found = false;
      void VisitExpr_(const CallPtr& call) override {
        found |= IsOp(call, "reg.squeeze_store_init") || IsOp(call, "reg.squeeze_store_step") ||
                 IsOp(call, "reg.squeeze_store_flush");
        IRVisitor::VisitExpr_(call);
      }
    } finder;
    finder.VisitStmt(scope->body_);
    if (!finder.found) return;
    inside_region_ = true;
    auto outer_loops = std::move(loops_);
    const auto outer_depth = depth_;
    loops_.clear();
    depth_ = 0;
    write_inputs_.clear();
    canonical_values_.Clear();
    IRVisitor::VisitStmt_(scope);
    CheckStepDependencies();
    dependencies_.clear();
    dependents_.clear();
    loop_steps_.clear();
    if (state_) Error("Squeeze-store state must be flushed before leaving RegScope", scope->span_);
    state_.reset();
    destination_.reset();
    loops_ = std::move(outer_loops);
    depth_ = outer_depth;
    inside_region_ = false;
  }
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    if (!inside_region_) return;
    auto call = As<Call>(stmt->value_);
    if (IsOp(call, "reg.squeeze_store_init")) {
      // Hardware has several store-unalign registers, but this lowering keeps
      // one cursor and shared AR count. That is an implementation restriction,
      // not a claim that the hardware has only one UnalignRegForStore.
      if (state_) {
        Error("Squeeze streams cannot nest or overlap in the current AR-based lowering; flush first",
              stmt->span_);
      }
      state_ = stmt->var_;
      destination_ = call->args_[0];
      init_depth_ = depth_;
      return;
    }
    if (IsOp(call, "reg.squeeze_store_step") || IsOp(call, "reg.squeeze_store_flush")) {
      if (!state_ || call->args_[0] != state_) {
        Error("Squeeze-store state must be consumed exactly once, in order", stmt->span_);
      }
      if (!destination_ || call->args_[1] != destination_) {
        Error("Squeeze-store state must use its original destination", stmt->span_);
      }
      if (IsOp(call, "reg.squeeze_store_flush")) {
        if (depth_ != init_depth_) {
          Error("A squeeze stream must close at its opening loop depth", stmt->span_);
        }
        state_.reset();
        destination_.reset();
      } else {
        if (!loops_.empty()) {
          RecordDependencies(stmt->var_.get(), call->args_[2]);
          RecordDependencies(stmt->var_.get(), call->args_[3]);
          loop_steps_.emplace_back(stmt.get(), loops_.back());
        }
        if (!write_inputs_
                 .emplace(canonical_values_.VisitExpr(call->args_[2]),
                          canonical_values_.VisitExpr(call->args_[3]))
                 .second) {
          Error(
              "PTOAS 0.74 may common stored SQZ producers; do not reuse equivalent "
              "value/mask pairs at multiple append sites",
              stmt->span_);
        }
        state_ = stmt->var_;
      }
      return;
    }
    if (As<SqueezeStoreType>(stmt->var_->GetType())) {
      // Eliminating a zero/single-trip loop can forward its state result by
      // assignment. This is a linear rename: consume the old SSA name so any
      // subsequent use of it still fails. Public state copies are rejected by
      // the frontend, but canonical IR and its printer must preserve these
      // compiler-generated transfers.
      if (AsVarLike(stmt->value_) && stmt->value_ == state_) {
        state_ = stmt->var_;
        return;
      }
      Error("Squeeze-store state cannot be copied or constructed by another operation", stmt->span_);
    }
    if (state_ && call &&
        (IsOp(call, "reg.load") || IsOp(call, "reg.store") || IsOp(call, "reg.gather") ||
         IsOp(call, "reg.scatter"))) {
      auto target = GetRegMemoryEffect(call).target;
      auto source_type = As<TileType>(target->GetType());
      auto dst_type = As<TileType>(destination_->GetType());
      bool aliases = target == destination_;
      // Views of one allocation share a root even when their variable names
      // differ. Pending partial blocks must not be observed through an alias.
      if (source_type->memref_ && dst_type->memref_) {
        aliases |= (*source_type->memref_)->base_ == (*dst_type->memref_)->base_;
      }
      if (aliases) {
        Error("Flush the squeeze stream before accessing its destination or an alias", stmt->span_);
      }
    }
    RecordDependencies(stmt->var_.get(), stmt->value_);
    canonical_values_.Record(stmt->var_, stmt->value_);
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    if (!inside_region_) {
      VisitStmt(loop->body_);
      return;
    }
    if (loops_.size() == 4) {
      Error("Squeeze-store loops can nest at most four levels", loop->span_);
      return;
    }
    auto outer = state_;
    size_t carry = loop->iter_args_.size();
    const auto yield = transform_utils::GetLastYieldStmt(loop->body_);
    const bool empty = transform_utils::EvalConstTripCount(loop) == 0;
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& arg = loop->iter_args_[i];
      if (!As<SqueezeStoreType>(arg->GetType())) {
        RecordDependencies(arg.get(), arg->initValue_);
        if (i < loop->return_vars_.size()) {
          const auto* result = loop->return_vars_[i].get();
          // A result has the carry's incoming and yielded dependencies. An
          // empty loop forwards only the initializer.
          if (empty) {
            RecordDependencies(result, arg->initValue_);
          } else {
            dependents_[arg.get()].push_back({result, loop.get()});
          }
        }
        continue;
      }
      if (carry != loop->iter_args_.size() || !outer || arg->initValue_ != outer) {
        Error("Squeeze-store loop carry must continue the active stream", loop->span_);
      }
      carry = i;
      state_ = arg;
    }
    if (state_ && loop->kind_ != ForKind::Sequential) {
      Error("Squeeze-store writes require sequential loops", loop->span_);
    }
    loops_.push_back(loop.get());
    dependencies_[loop->loop_var_.get()] = {loop.get()};
    // A carry receives its initializer on entry and its yielded value on the
    // backedge. Record both edges even when the yield refers to a later definition.
    if (yield) {
      for (size_t i = 0; i < loop->iter_args_.size() && i < yield->value_.size(); ++i) {
        const auto& arg = loop->iter_args_[i];
        if (As<SqueezeStoreType>(arg->GetType())) continue;
        RecordDependencies(arg.get(), yield->value_[i]);
      }
    }
    ++depth_;
    VisitStmt(loop->body_);
    --depth_;
    loops_.pop_back();
    if (carry != loop->iter_args_.size()) {
      if (!yield || carry >= yield->value_.size() || yield->value_[carry] != state_) {
        Error("Squeeze-store loop must yield its final state", loop->span_);
      }
      state_ = loop->return_vars_[carry];
    } else if (!outer && state_) {
      Error("Squeeze-store state cannot escape its opening loop", loop->span_);
    }
  }
  void VisitStmt_(const IfStmtPtr& stmt) override {
    if (state_) {
      Error("Use lane masks instead of conditionals while squeeze-store state is active", stmt->span_);
    }
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitStmt_(const WhileStmtPtr& stmt) override {
    if (state_) Error("Squeeze-store state currently supports sequential for loops", stmt->span_);
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitStmt_(const BreakStmtPtr& stmt) override {
    if (state_) {
      Error("Flush squeeze-store state before break; early loop exits are unsupported", stmt->span_);
    }
  }
  void VisitStmt_(const ContinueStmtPtr& stmt) override {
    if (state_) {
      Error("Flush squeeze-store state before continue; early loop exits are unsupported", stmt->span_);
    }
  }
  void VisitVarLike_(const VarPtr& var) override {
    if (dependent_) dependents_[var.get()].push_back({dependent_});
  }
  // The initializer is already recorded where it is evaluated, outside the loop.
  void VisitExpr_(const IterArgPtr& arg) override { VisitVarLike_(arg); }

  void VisitExpr_(const CallPtr& call) override {
    if (!inside_region_) return;
    if (dependent_ && (IsOp(call, "reg.load") || IsOp(call, "reg.gather"))) {
      dependencies_[dependent_].insert(loops_.begin(), loops_.end());
    }
    for (const auto& arg : call->args_) {
      if (As<SqueezeStoreType>(arg->GetType())) {
        Error("Squeeze-store state is only valid in its stream lifecycle operations", call->span_);
      }
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  struct DependencyUse {
    const Var* var;
    // Leaving a loop drops its label and preserves enclosing-loop dependencies.
    const ForStmt* exited_loop = nullptr;
  };

  void RecordDependencies(const Var* dependent, const ExprPtr& expr) {
    dependent_ = dependent;
    VisitExpr(expr);
    dependent_ = nullptr;
  }

  void CheckStepDependencies() {
    if (loop_steps_.empty()) return;
    // Propagate loop-local loads and induction variables through SSA uses and
    // carry backedges. Each (value, loop) enters the worklist once. Every value
    // has at most four enclosing VF loops, keeping the walk O(N log N).
    std::vector<std::pair<const Var*, const ForStmt*>> pending;
    for (const auto& [var, loops] : dependencies_) {
      for (const auto* loop : loops) pending.emplace_back(var, loop);
    }
    while (!pending.empty()) {
      const auto [var, loop] = pending.back();
      pending.pop_back();
      auto users = dependents_.find(var);
      if (users == dependents_.end()) continue;
      for (const auto& user : users->second) {
        if (user.exited_loop == loop) continue;
        if (dependencies_[user.var].insert(loop).second) pending.emplace_back(user.var, loop);
      }
    }
    // PTOAS 0.74 treats stored VSQZ as pure even though it pushes SQZN.
    // Hoisting it can leave VSTUR without one count per iteration.
    for (const auto& [step, loop] : loop_steps_) {
      if (!dependencies_[step->var_.get()].count(loop)) {
        Error(
            "PTOAS 0.74 cannot safely stream loop-invariant value and mask; "
            "load or derive an input inside the current loop",
            step->span_);
      }
    }
  }
  void Error(const std::string& message, const Span& span) {
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "RegScopeValid", 0, message, span);
  }
  std::vector<Diagnostic>& diagnostics_;
  std::vector<const ForStmt*> loops_;
  const Var* dependent_ = nullptr;
  std::map<const Var*, std::set<const ForStmt*>> dependencies_;
  std::map<const Var*, std::vector<DependencyUse>> dependents_;
  std::vector<std::pair<const AssignStmt*, const ForStmt*>> loop_steps_;
  std::set<std::pair<ExprPtr, ExprPtr>> write_inputs_;
  StreamValueCanonicalizer canonical_values_;
  bool inside_region_ = false;
  ExprPtr state_;
  ExprPtr destination_;
  int depth_ = 0;
  int init_depth_ = 0;
};
}  // namespace

void VerifyRegScopes(const FunctionPtr& function, std::vector<Diagnostic>& diagnostics,
                     bool final_placement) {
  if (function && function->body_) {
    RegionChecker(function, diagnostics, final_placement).Run();
    // Type/arity diagnostics above take precedence over lifecycle validation.
    if (diagnostics.empty()) SqueezeStreamChecker(diagnostics).VisitStmt(function->body_);
  }
}
void CheckRegScopes(const FunctionPtr& function) {
  std::vector<Diagnostic> diagnostics;
  VerifyRegScopes(function, diagnostics);
  if (!diagnostics.empty()) {
    CHECK_SPAN(false, diagnostics.front().span) << diagnostics.front().message;
  }
}
class RegScopePropertyVerifier : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "RegScopeValid"; }
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, function] : program->functions_) VerifyRegScopes(function, diagnostics);
  }
};
PropertyVerifierPtr CreateRegScopePropertyVerifier() { return std::make_shared<RegScopePropertyVerifier>(); }
}  // namespace pypto::ir

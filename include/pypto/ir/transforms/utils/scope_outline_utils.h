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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_SCOPE_OUTLINE_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_SCOPE_OUTLINE_UTILS_H_

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
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
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

using transform_utils::Substitute;
namespace outline_utils {

// Re-export from var_collectors for backward compatibility
using var_collectors::VarDefUseCollector;

/**
 * @brief Visitor to collect target tensors of tile.store calls (by pointer identity).
 *
 * These tensors are modified via side-effect inside scopes but are not
 * captured by VarDefUseCollector since they are defined externally.  The third
 * argument of store is the output tensor.
 */
class StoreTargetCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> store_targets;

 protected:
  void VisitExpr_(const CallPtr& op) override {
    auto opnode = std::dynamic_pointer_cast<const Op>(op->op_);
    if (opnode && opnode->name_ == "tile.store" && op->args_.size() >= 3) {
      if (auto var = As<Var>(op->args_[2])) {
        store_targets.insert(var.get());
      }
    }
    IRVisitor::VisitExpr_(op);
  }
};

/**
 * @brief Collect SSA post-store aliases: variables bound via
 *        ``AssignStmt(v, Call(tile.store, ..., target))``.
 *
 * Maps ``v`` (the SSA-assignee) to ``target`` (the store's last argument).
 * Both identify the same tensor state (post-store) with different pointer
 * identities — ``v`` lives in ``body_collector.var_defs`` while ``target``
 * shows up as a ``store_targets`` entry.  ScopeOutliner uses this to avoid
 * exporting the same tensor twice when both sets would otherwise contribute
 * outputs.
 */
class PostStoreAliasCollector : public IRVisitor {
 public:
  /// alias var (the SSA assignee of a tile.store call) → original store target
  std::unordered_map<const Var*, const Var*> alias_to_target;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    auto call = std::dynamic_pointer_cast<const Call>(op->value_);
    auto opnode = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
    if (opnode && opnode->name_ == "tile.store" && call->args_.size() >= 3) {
      if (auto target = As<Var>(call->args_[2])) {
        alias_to_target.emplace(op->var_.get(), target.get());
      }
    }
    IRVisitor::VisitStmt_(op);
  }
};

/**
 * @brief Mutator that converts EvalStmt(Call(tile.store, ...)) into
 *        AssignStmt(target_var, Call(tile.store, ...)) for specified
 *        store targets.
 *
 * tile.store returns the output tensor (same type as the 3rd argument).  When
 * the original IR uses EvalStmt (discarding the return value), this mutator
 * re-writes it as an AssignStmt so the return value is captured and can be
 * referenced in a subsequent ReturnStmt.
 */
class StoreEvalToAssignMutator : public IRMutator {
 public:
  explicit StoreEvalToAssignMutator(const std::unordered_map<const Var*, VarPtr>& target_vars)
      : target_vars_(target_vars) {}

 protected:
  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = std::dynamic_pointer_cast<const Call>(op->expr_);
    if (!call) return op;
    auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
    if (!opnode || opnode->name_ != "tile.store") {
      return op;
    }
    if (call->args_.size() < 3) return op;
    auto var = As<Var>(call->args_[2]);
    if (!var) return op;
    auto it = target_vars_.find(var.get());
    if (it == target_vars_.end()) return op;
    return std::make_shared<AssignStmt>(it->second, call, op->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Handle SSA-converted stores: AssignStmt(buf_1, Call(tile.store, ..., buf_0))
    // The _store_ret var needs to be assigned the store result too.
    auto call = std::dynamic_pointer_cast<const Call>(op->value_);
    if (!call) return IRMutator::VisitStmt_(op);
    auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
    if (!opnode || opnode->name_ != "tile.store") return IRMutator::VisitStmt_(op);
    if (call->args_.size() < 3) return IRMutator::VisitStmt_(op);
    auto var = As<Var>(call->args_[2]);
    if (!var) return IRMutator::VisitStmt_(op);
    auto it = target_vars_.find(var.get());
    if (it == target_vars_.end()) return IRMutator::VisitStmt_(op);
    // Keep original assignment (buf_1 = store(...)) and add _store_ret = buf_1
    auto store_ret_assign = std::make_shared<AssignStmt>(it->second, op->var_, op->span_);
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{op, store_ret_assign}, op->span_);
  }

 private:
  std::unordered_map<const Var*, VarPtr> target_vars_;
};

/** @brief Visitor to build a symbol table mapping variable pointers to their types and Var objects. */
class VarCollector : public IRVisitor {
 public:
  std::unordered_map<const Var*, TypePtr> var_types;
  std::unordered_map<const Var*, VarPtr> var_objects;
  std::unordered_set<std::string> known_names;

 protected:
  // Use VisitVarLike_ to collect both Var and IterArg references.
  // VisitExpr_(IterArgPtr) calls VisitVarLike_ then visits initValue_,
  // so IterArgs from outer loops are included in the symbol table.
  void VisitVarLike_(const VarPtr& op) override {
    var_types.try_emplace(op.get(), op->GetType());
    var_objects.try_emplace(op.get(), op);
    known_names.insert(op->name_hint_);
    IRVisitor::VisitVarLike_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    var_types[op->var_.get()] = op->var_->GetType();
    var_objects[op->var_.get()] = op->var_;
    known_names.insert(op->var_->name_hint_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    var_types[op->loop_var_.get()] = op->loop_var_->GetType();
    var_objects[op->loop_var_.get()] = op->loop_var_;
    known_names.insert(op->loop_var_->name_hint_);
    for (const auto& iter_arg : op->iter_args_) {
      var_types[iter_arg.get()] = iter_arg->GetType();
      var_objects[iter_arg.get()] = iter_arg;
      known_names.insert(iter_arg->name_hint_);
    }
    for (const auto& return_var : op->return_vars_) {
      var_types[return_var.get()] = return_var->GetType();
      var_objects[return_var.get()] = return_var;
      known_names.insert(return_var->name_hint_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& iter_arg : op->iter_args_) {
      var_types[iter_arg.get()] = iter_arg->GetType();
      var_objects[iter_arg.get()] = iter_arg;
      known_names.insert(iter_arg->name_hint_);
    }
    for (const auto& return_var : op->return_vars_) {
      var_types[return_var.get()] = return_var->GetType();
      var_objects[return_var.get()] = return_var;
      known_names.insert(return_var->name_hint_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& return_var : op->return_vars_) {
      var_types[return_var.get()] = return_var->GetType();
      var_objects[return_var.get()] = return_var;
      known_names.insert(return_var->name_hint_);
    }
    IRVisitor::VisitStmt_(op);
  }
};

// ============================================================================
// Parameterized scope outliner
// ============================================================================

/**
 * @brief Mutator to outline scopes of a given ScopeKind into separate functions.
 *
 * Parameterized by the target ScopeKind, the FunctionType for outlined functions,
 * and a naming suffix. Handles SeqStmts specially to determine which scope-defined
 * variables are actually used after each scope (output filtering), and recursively
 * transforms scope bodies to handle nested scopes.
 */
class ScopeOutliner : public IRMutator {
 public:
  ScopeOutliner(std::string func_name, const std::unordered_map<const Var*, TypePtr>& var_types,
                const std::unordered_map<const Var*, VarPtr>& var_objects,
                const std::unordered_set<std::string>& known_names, ScopeKind target_scope_kind,
                FunctionType outlined_func_type, std::string name_suffix, ProgramPtr program = nullptr)
      : func_name_(std::move(func_name)),
        var_types_(var_types),
        var_objects_(var_objects),
        known_names_(known_names),
        target_scope_kind_(target_scope_kind),
        outlined_func_type_(outlined_func_type),
        name_suffix_(std::move(name_suffix)),
        program_(std::move(program)) {}

  [[nodiscard]] const std::vector<FunctionPtr>& GetOutlinedFunctions() const { return outlined_functions_; }

 protected:
  /**
   * @brief Substitute store-target variables that were renamed for SSA compliance.
   *
   * When a store-target output is assigned a fresh SSA name at the call site
   * (e.g., buf_0 -> buf_1), subsequent references must use the new variable.
   */
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = store_target_renames_.find(op.get());
    if (it != store_target_renames_.end()) {
      return it->second;
    }
    return IRMutator::VisitExpr_(op);
  }

  /**
   * @brief Compute used_after when no explicit context is available.
   *
   * Two regimes:
   *   1. Scope is nested inside another (non-target) ScopeStmt — only store
   *      targets escape, because scope boundaries confine locally-defined
   *      variables.
   *   2. Scope is at the top level or inside a control-flow body — retain the
   *      original defensive fallback: all defined vars + store targets are
   *      treated as outputs so the caller retains access.
   */
  std::unordered_set<const Var*> ComputeFallbackUsedAfter(const ScopeStmtPtr& scope) const {
    StoreTargetCollector store_collector;
    store_collector.VisitStmt(scope->body_);
    std::unordered_set<const Var*> used_after;
    if (!inside_nested_scope_body_) {
      VarDefUseCollector def_collector;
      def_collector.VisitStmt(scope->body_);
      used_after = def_collector.var_defs;
    }
    used_after.insert(store_collector.store_targets.begin(), store_collector.store_targets.end());
    return used_after;
  }

  /**
   * @brief Process SeqStmts to analyze scope outputs using subsequent statements.
   *
   * For each target scope, collects variables referenced in all subsequent statements
   * plus any variables required by a parent scope (propagated via required_outputs_).
   */
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    bool changed = false;

    for (size_t i = 0; i < op->stmts_.size(); ++i) {
      auto scope = std::dynamic_pointer_cast<const ScopeStmt>(op->stmts_[i]);
      // Always compute what's used in the tail of this SeqStmts; this set is
      // the "used_after" for a target scope at position i, and doubles as the
      // required_outputs_ propagated into a non-target statement so any
      // target-kind scope nested inside knows what needs to leak out.
      VarDefUseCollector after_ref_collector;
      for (size_t j = i + 1; j < op->stmts_.size(); ++j) {
        after_ref_collector.VisitStmt(op->stmts_[j]);
      }
      auto after_refs = after_ref_collector.GetAllVarRefs();

      if (scope && scope->GetScopeKind() == target_scope_kind_) {
        // Also include variables required by parent scope
        auto used_after = after_refs;
        used_after.insert(required_outputs_.begin(), required_outputs_.end());

        // When no context is available (no subsequent statements and no parent
        // requirements), fall back to scope-nesting-aware defaults.  This
        // happens when a single ScopeStmt is wrapped in SeqStmts inside a
        // control-flow body (if/for/while) where the outer context hasn't
        // propagated required_outputs_, or when the scope sits directly
        // inside another scope (e.g. an InCoreScopeStmt at the top of a
        // SpmdScopeStmt body produced by the ``for i in pl.spmd(...)`` form).
        if (used_after.empty()) {
          used_after = ComputeFallbackUsedAfter(scope);
        }

        // Outline this scope with context about what's used after
        auto outlined_stmt = OutlineScope(scope, used_after);
        new_stmts.push_back(outlined_stmt);
        changed = true;
      } else {
        // Recursively visit non-target statements.  Temporarily extend
        // required_outputs_ with the tail-use set so any target-kind scope
        // nested inside this statement can compute a correct used_after.
        auto saved_required_outputs = required_outputs_;
        required_outputs_.insert(after_refs.begin(), after_refs.end());
        auto visited = VisitStmt(op->stmts_[i]);
        required_outputs_ = saved_required_outputs;
        new_stmts.push_back(visited);
        if (visited != op->stmts_[i]) {
          changed = true;
        }
      }
    }

    if (!changed) {
      return op;
    }
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }

  /**
   * @brief Handle ScopeStmts that are direct children of another node (not
   * inside a SeqStmts).
   *
   * The fallback honours scope nesting via ``inside_nested_scope_body_``: an
   * inner scope whose only "context" is an enclosing non-target scope has no
   * way for its locally-defined variables to escape, so we only expose store
   * targets. Scopes at the true top level (outside any parent scope body)
   * retain the original defensive "all defs are outputs" behaviour.
   */
  // Shared per-kind logic: outline if kind matches, else descend with the
  // nested-scope flag set so any target-kind scope we find inside can make a
  // correct used_after decision.
  template <typename ScopeT>
  StmtPtr VisitScopeKind(const std::shared_ptr<const ScopeT>& op) {
    if (op->GetScopeKind() != target_scope_kind_) {
      bool prev = std::exchange(inside_nested_scope_body_, true);
      auto result = IRMutator::VisitStmt_(op);
      inside_nested_scope_body_ = prev;
      return result;
    }
    // Prefer the enclosing SeqStmts' propagated requirements over the
    // "no context" fallback.  This matters when a target scope is a direct
    // child of another scope (SpmdScopeStmt{InCoreScopeStmt{...}}) — the
    // enclosing SeqStmts visitor populates required_outputs_ with variables
    // the post-scope statements still reference.
    if (required_outputs_.empty()) {
      return OutlineScope(op, ComputeFallbackUsedAfter(op));
    }
    std::unordered_set<const Var*> used_after = required_outputs_;
    StoreTargetCollector store_collector;
    store_collector.VisitStmt(op->body_);
    used_after.insert(store_collector.store_targets.begin(), store_collector.store_targets.end());
    return OutlineScope(op, used_after);
  }

  StmtPtr VisitStmt_(const InCoreScopeStmtPtr& op) override { return VisitScopeKind(op); }
  StmtPtr VisitStmt_(const AutoInCoreScopeStmtPtr& op) override { return VisitScopeKind(op); }
  StmtPtr VisitStmt_(const ClusterScopeStmtPtr& op) override { return VisitScopeKind(op); }
  StmtPtr VisitStmt_(const HierarchyScopeStmtPtr& op) override { return VisitScopeKind(op); }
  StmtPtr VisitStmt_(const SpmdScopeStmtPtr& op) override { return VisitScopeKind(op); }

 private:
  /**
   * @brief Outline a single scope into a separate function.
   *
   * @param op The scope statement to outline
   * @param used_after Variables (by pointer) used in subsequent statements (determines outputs)
   */
  StmtPtr OutlineScope(const ScopeStmtPtr& op, const std::unordered_set<const Var*>& used_after) {
    // Generate function name: use user-provided hint when available, otherwise auto-generate.
    // On collision, auto-deduplicate with a numeric suffix (name_hint semantics).
    std::string outlined_func_name;
    if (!op->name_hint_.empty()) {
      outlined_func_name = op->name_hint_;
      scope_counter_++;  // Keep counter stable for unnamed scopes
    } else {
      std::string suffix = name_suffix_;
      if (auto hier = As<HierarchyScopeStmt>(op)) {
        suffix = GenerateHierarchySuffix(hier->level_, hier->role_);
      }
      std::ostringstream name_stream;
      name_stream << func_name_ << suffix << scope_counter_++;
      outlined_func_name = name_stream.str();
    }
    // Deduplicate: append _0, _1, ... if name already taken
    if (known_names_.count(outlined_func_name)) {
      std::string base = outlined_func_name;
      int dedup_counter = 0;
      do {
        outlined_func_name = base + "_" + std::to_string(dedup_counter++);
      } while (known_names_.count(outlined_func_name));
    }
    known_names_.insert(outlined_func_name);

    // Analyze the scope body for inputs and outputs (before recursing).
    // A single VarDefUseCollector replaces the old VarRefCollector + VarDefCollector pair.
    VarDefUseCollector body_collector;
    body_collector.VisitStmt(op->body_);

    // Inputs: variables used but not defined in the scope.
    // Iterate in first-encounter order to preserve the callee's parameter ordering.
    std::vector<VarPtr> input_vars;
    for (const Var* var_ptr : body_collector.var_uses_ordered) {
      if (!body_collector.var_defs.count(var_ptr)) {
        auto obj_it = var_objects_.find(var_ptr);
        CHECK(obj_it != var_objects_.end())
            << "Variable " << var_ptr->name_hint_ << " not found in var_objects";
        input_vars.push_back(obj_it->second);
      }
    }

    // Outputs: variables defined in the scope AND used after it
    std::vector<VarPtr> output_vars;
    std::unordered_set<const Var*> store_output_set;

    // Collect type info from scope body for output variables
    VarCollector scope_var_collector;
    scope_var_collector.VisitStmt(op->body_);

    // Map any SSA post-store alias (var_def bound to a tile.store call) back
    // to its store target so we don't export the same tensor twice.
    PostStoreAliasCollector post_store_collector;
    post_store_collector.VisitStmt(op->body_);

    // Store targets present in the scope body — used to decide whether a
    // post-store alias's original target will already appear in output_vars
    // via the store-target-output pass below.
    StoreTargetCollector store_collector;
    store_collector.VisitStmt(op->body_);

    // Aliases deferred to the call-site emission: each pair maps a
    // scope-local SSA post-store alias (pointer identity in the scope body)
    // to the external store target's var_objects_ pointer.  After the fresh
    // store-target Var is created at the call site we rename the alias to
    // it, so subsequent parent-function references resolve correctly.
    std::vector<std::pair<const Var*, const Var*>> deferred_post_store_aliases;
    for (const Var* var_ptr : body_collector.var_defs_ordered) {
      if (!used_after.count(var_ptr)) continue;

      // Skip if this var_def is a post-store alias of an external store
      // target: that same tensor will be exported by the store-target pass
      // below, and we don't want a duplicated output entry.
      auto alias_it = post_store_collector.alias_to_target.find(var_ptr);
      const Var* target_ptr =
          (alias_it != post_store_collector.alias_to_target.end()) ? alias_it->second : nullptr;
      if (target_ptr && store_collector.store_targets.count(target_ptr) &&
          !body_collector.var_defs.count(target_ptr)) {
        auto ext_it = var_objects_.find(target_ptr);
        CHECK(ext_it != var_objects_.end())
            << "Store target " << target_ptr->name_hint_ << " not found in var_objects";
        deferred_post_store_aliases.emplace_back(var_ptr, ext_it->second.get());
        continue;
      }

      auto scope_it = scope_var_collector.var_objects.find(var_ptr);
      CHECK(scope_it != scope_var_collector.var_objects.end())
          << "Variable " << var_ptr->name_hint_ << " not found in scope body";
      output_vars.push_back(scope_it->second);
    }

    // Also treat store targets as outputs: external tensors modified via
    // tile.store.  These represent side-effect outputs that must be
    // returned regardless of whether they appear in used_after, because the
    // store mutates an externally-visible buffer (e.g. loop-carried state).
    //
    // Skip for Hierarchy scopes: the outlined function receives the buffer
    // as an InOut parameter, so the store side-effect is already visible
    // to the caller without an explicit return.
    //
    // Track two pointer identities per store target:
    //   - var_objects_ pointer (ext_it->second.get()) — goes into output_vars
    //     and store_output_set for consistent classification
    //   - body pointer (var_ptr) — kept in store_body_ptrs for the
    //     StoreEvalToAssignMutator, which matches against the un-substituted
    //     scope body where store targets retain their original pointers
    std::unordered_map<const Var*, const Var*> store_body_ptrs;
    if (op->GetScopeKind() != ScopeKind::Hierarchy) {
      for (const Var* var_ptr : store_collector.store_targets) {
        if (!body_collector.var_defs.count(var_ptr)) {
          auto ext_it = var_objects_.find(var_ptr);
          CHECK(ext_it != var_objects_.end())
              << "Variable " << var_ptr->name_hint_ << " not found in var_objects";
          output_vars.push_back(ext_it->second);
          store_output_set.insert(ext_it->second.get());
          store_body_ptrs[ext_it->second.get()] = var_ptr;
        }
      }
    }

    // Recursively transform the scope body (handles nested scopes).
    // Save/restore state so nested scopes get their own hierarchical names and counters.
    // Also overlay the current scope's symbol table while recursing so nested
    // outlining resolves names to the lexically-nearest Var, not to an unrelated
    // same-named Var elsewhere in the function.
    // store_target_renames_ must be cleared so parent renames don't leak into the scope
    // body — the scope's own parameter substitution handles variable mapping instead.
    std::string saved_func_name = func_name_;
    int saved_scope_counter = scope_counter_;
    auto saved_var_types = var_types_;
    auto saved_var_objects = var_objects_;
    auto saved_known_names = known_names_;
    auto saved_required_outputs = required_outputs_;
    auto saved_renames = store_target_renames_;
    func_name_ = outlined_func_name;
    scope_counter_ = 0;
    for (const auto& [ptr, type] : scope_var_collector.var_types) {
      var_types_[ptr] = type;
    }
    for (const auto& [ptr, var] : scope_var_collector.var_objects) {
      var_objects_[ptr] = var;
    }
    known_names_.insert(scope_var_collector.known_names.begin(), scope_var_collector.known_names.end());
    store_target_renames_.clear();
    // Propagate output requirements so nested scopes know what's needed
    required_outputs_.clear();
    for (const auto& var : output_vars) {
      required_outputs_.insert(var.get());
    }
    auto recursed_body = VisitStmt(op->body_);
    func_name_ = saved_func_name;
    scope_counter_ = saved_scope_counter;
    var_types_ = saved_var_types;
    var_objects_ = saved_var_objects;
    known_names_ = saved_known_names;
    required_outputs_ = saved_required_outputs;
    store_target_renames_ = saved_renames;

    // Create fresh parameters for the outlined function.
    // Infer param directions from the inner callee when possible (requires program_).
    std::vector<ParamDirection> inferred_directions =
        InferParamDirections(input_vars, op->body_, store_output_set);
    std::vector<VarPtr> input_params;
    std::vector<ParamDirection> input_param_directions;
    std::unordered_map<const Var*, VarPtr> var_substitution_map;
    for (size_t i = 0; i < input_vars.size(); ++i) {
      const auto& input_var = input_vars[i];
      auto param_var = std::make_shared<Var>(input_var->name_hint_, input_var->GetType(), op->span_);
      input_params.push_back(param_var);
      input_param_directions.push_back(inferred_directions[i]);
      var_substitution_map[input_var.get()] = param_var;
    }

    // Build the set of names already used in the outlined function (inputs + scope-body locals)
    // to ensure generated output names don't collide.
    std::unordered_set<std::string> outlined_used_names;
    for (const auto& input_var : input_vars) {
      outlined_used_names.insert(input_var->name_hint_);
    }
    outlined_used_names.insert(scope_var_collector.known_names.begin(),
                               scope_var_collector.known_names.end());

    // Create fresh output variables for the outlined function
    std::vector<VarPtr> outlined_output_vars;
    std::vector<TypePtr> return_types;
    for (const auto& out_var : output_vars) {
      bool is_store = store_output_set.count(out_var.get()) > 0;
      TypePtr var_type;
      if (is_store) {
        // Store target: external variable, look up from outer symbol table
        auto type_it = var_types_.find(out_var.get());
        CHECK(type_it != var_types_.end())
            << "Variable " << out_var->name_hint_ << " not found in symbol table";
        var_type = type_it->second;
      } else {
        // Regular output: defined in scope body
        var_type = out_var->GetType();
      }
      // For store targets, create a fresh variable with a unique "_store_ret" suffix
      // to avoid redefining the input parameter in SSA form.
      std::string out_var_name;
      if (is_store) {
        out_var_name = auto_name::BuildName(auto_name::GetBaseName(out_var->name_hint_), "", "store");
        if (outlined_used_names.count(out_var_name)) {
          out_var_name = auto_name::GenerateFreshNameLike(out_var_name, outlined_used_names);
        }
      } else {
        out_var_name = out_var->name_hint_;
      }
      outlined_used_names.insert(out_var_name);
      auto outlined_var = std::make_shared<Var>(out_var_name, var_type, op->span_);
      outlined_output_vars.push_back(outlined_var);
      return_types.push_back(var_type);
      if (!is_store) {
        var_substitution_map[out_var.get()] = outlined_var;
      }
    }

    // Convert EvalStmt/AssignStmt(tile.store) to assign _store_ret vars BEFORE
    // Substitute, since store_body_ptrs uses the original body Var pointers.
    auto pre_sub_body = recursed_body;
    if (!store_output_set.empty()) {
      std::unordered_map<const Var*, VarPtr> store_target_vars;
      for (size_t idx = 0; idx < output_vars.size(); ++idx) {
        auto body_it = store_body_ptrs.find(output_vars[idx].get());
        if (body_it != store_body_ptrs.end()) {
          store_target_vars[body_it->second] = outlined_output_vars[idx];
        }
      }
      StoreEvalToAssignMutator store_mutator(store_target_vars);
      pre_sub_body = store_mutator.VisitStmt(pre_sub_body);
    }

    // Apply pointer-based substitution after store results are materialized.
    //
    // We can't reuse `Substitute` here because IRMutator::VisitExpr_(VarPtr)
    // mints a *fresh* Var when an old Var's type embeds a remapped shape Var
    // (see mutator.cpp:225-239). For a tensor input whose shape references
    // another input scalar, this means the body ends up referencing a Var
    // that's NOT the one we just pushed into `input_params`; the codegen's
    // param-binding loop then can't find a tensor view for it. We need
    // visibility into the post-substitution remap state to pull out the
    // freshened param Vars and update `input_params` accordingly.
    class TrackingSubstituteMutator : public IRMutator {
     public:
      explicit TrackingSubstituteMutator(const std::unordered_map<const Var*, VarPtr>& var_map) {
        for (const auto& [k, v] : var_map) {
          var_remap_[k] = v;
        }
      }
      const std::unordered_map<const Expr*, ExprPtr>& GetVarRemap() const { return var_remap_; }
    };
    TrackingSubstituteMutator subst_mutator(var_substitution_map);
    auto transformed_body = subst_mutator.VisitStmt(pre_sub_body);

    // Reconcile param/output Vars with any freshened versions created during
    // substitution. ResolveVarRemapHit memoizes the resolved (final) Var back
    // into var_remap_ keyed by the original Var, so the chain
    //   old → seed (initial param/outlined) → freshened (after type remap)
    // collapses to old → freshened. Pick that out and replace the stale
    // entry in input_params / outlined_output_vars / return_types.
    const auto& post_remap = subst_mutator.GetVarRemap();
    auto resolve_to_freshened = [&](const VarPtr& original, const VarPtr& seeded) -> VarPtr {
      auto it = post_remap.find(original.get());
      if (it == post_remap.end()) return seeded;
      auto freshened = AsVarLike(it->second);
      if (!freshened) return seeded;
      return freshened;
    };
    for (size_t i = 0; i < input_vars.size(); ++i) {
      input_params[i] = resolve_to_freshened(input_vars[i], input_params[i]);
    }
    for (size_t i = 0; i < output_vars.size(); ++i) {
      bool is_store = store_output_set.count(output_vars[i].get()) > 0;
      if (is_store) {
        // Store targets aren't seeded into var_substitution_map, but their types
        // may still embed remapped shape vars that trigger freshening during
        // substitution. Check only the outlined var key; the original key may
        // alias an input param (InOut parameters) and produce a false match.
        auto it = post_remap.find(outlined_output_vars[i].get());
        if (it != post_remap.end()) {
          if (auto freshened = AsVarLike(it->second)) {
            outlined_output_vars[i] = freshened;
            return_types[i] = freshened->GetType();
          }
        }
        continue;
      }
      auto freshened = resolve_to_freshened(output_vars[i], outlined_output_vars[i]);
      outlined_output_vars[i] = freshened;
      return_types[i] = freshened->GetType();
    }

    // Build outlined function body (transformed body + return statement)
    StmtPtr outlined_body;
    if (outlined_output_vars.empty()) {
      outlined_body = transformed_body;
    } else {
      std::vector<ExprPtr> return_exprs(outlined_output_vars.begin(), outlined_output_vars.end());
      auto return_stmt = std::make_shared<ReturnStmt>(return_exprs, op->span_);

      std::vector<StmtPtr> body_stmts;
      if (auto seq_stmts = std::dynamic_pointer_cast<const SeqStmts>(transformed_body)) {
        body_stmts = seq_stmts->stmts_;
      } else {
        body_stmts.push_back(transformed_body);
      }
      body_stmts.push_back(return_stmt);
      outlined_body = std::make_shared<SeqStmts>(body_stmts, op->span_);
    }

    // Register the outlined function (propagate level/role from ScopeStmt, convert split/core_num to attrs)
    std::vector<std::pair<std::string, std::any>> outlined_attrs;
    auto append_split_attr = [&](std::optional<SplitMode> split) {
      if (split.has_value() && split.value() != SplitMode::None) {
        outlined_attrs.emplace_back("split", static_cast<int>(split.value()));
      }
    };
    if (auto incore = As<InCoreScopeStmt>(op)) {
      append_split_attr(incore->split_);
    } else if (auto auto_incore = As<AutoInCoreScopeStmt>(op)) {
      append_split_attr(auto_incore->split_);
    } else if (auto spmd = As<SpmdScopeStmt>(op)) {
      outlined_attrs.emplace_back("core_num", spmd->core_num_);
      if (spmd->sync_start_) {
        outlined_attrs.emplace_back("sync_start", true);
      }
    }
    std::optional<Level> outlined_level;
    std::optional<Role> outlined_role;
    if (auto hier = As<HierarchyScopeStmt>(op)) {
      outlined_level = hier->level_;
      outlined_role = hier->role_;
    }
    auto outlined_func = std::make_shared<Function>(
        outlined_func_name, input_params, input_param_directions, return_types, outlined_body, op->span_,
        outlined_func_type_, outlined_level, outlined_role, std::move(outlined_attrs));
    outlined_functions_.push_back(outlined_func);

    // Build the call site in the parent function
    auto global_var = std::make_shared<GlobalVar>(outlined_func_name);
    std::vector<ExprPtr> call_args;
    for (const auto& input_var : input_vars) {
      auto var_it = var_objects_.find(input_var.get());
      CHECK(var_it != var_objects_.end())
          << "Variable " << input_var->name_hint_ << " not found in var_objects";
      call_args.push_back(var_it->second);
    }

    // Determine call return type
    TypePtr call_return_type;
    if (return_types.empty()) {
      call_return_type = nullptr;
    } else if (return_types.size() == 1) {
      call_return_type = return_types[0];
    } else {
      call_return_type = std::make_shared<TupleType>(return_types);
    }

    std::shared_ptr<Call> call_expr;
    if (call_return_type) {
      call_expr = std::make_shared<Call>(global_var, call_args, call_return_type, op->span_);
    } else {
      call_expr = std::make_shared<Call>(global_var, call_args, op->span_);
    }

    // Resolve the call-site Var for an output variable. Scope-defined vars come from
    // scope_var_collector; store targets (external tensors) fall back to the outer symbol table.
    // Store targets get a fresh SSA name to avoid re-assigning the input variable.
    auto resolve_call_site_var = [&](const VarPtr& out_var) -> VarPtr {
      bool is_store = store_output_set.count(out_var.get()) > 0;
      if (!is_store) {
        auto var_it = scope_var_collector.var_objects.find(out_var.get());
        if (var_it != scope_var_collector.var_objects.end()) {
          return var_it->second;
        }
        auto ext_it = var_objects_.find(out_var.get());
        CHECK(ext_it != var_objects_.end())
            << "Variable " << out_var->name_hint_ << " not found in var_objects";
        return ext_it->second;
      }
      auto ext_it = var_objects_.find(out_var.get());
      CHECK(ext_it != var_objects_.end())
          << "Variable " << out_var->name_hint_ << " not found in var_objects";
      return CreateFreshStoreTargetVar(ext_it->second, op->span_);
    };

    // Create assignments for output variables in the parent function.
    // We assemble the result first, then register deferred post-store alias
    // renames once — each alias maps to a store target whose fresh var is
    // populated by resolve_call_site_var (via CreateFreshStoreTargetVar) into
    // store_target_renames_, so the registration must happen after all
    // resolve_call_site_var calls.
    StmtPtr result;
    if (output_vars.empty()) {
      result = std::make_shared<EvalStmt>(call_expr, op->span_);
    } else if (output_vars.size() == 1) {
      auto output_var = resolve_call_site_var(output_vars[0]);
      result = std::make_shared<AssignStmt>(output_var, call_expr, op->span_);
    } else {
      // Assign call result to a temporary variable, then unpack with TupleGetItem
      auto ret_var =
          std::make_shared<Var>(auto_name::BuildName("ret", "", "tmp", 0), call_return_type, op->span_);
      std::vector<StmtPtr> stmts;
      stmts.push_back(std::make_shared<AssignStmt>(ret_var, call_expr, op->span_));
      for (size_t i = 0; i < output_vars.size(); ++i) {
        auto tuple_get = std::make_shared<TupleGetItemExpr>(ret_var, static_cast<int>(i), op->span_);
        auto output_var = resolve_call_site_var(output_vars[i]);
        stmts.push_back(std::make_shared<AssignStmt>(output_var, tuple_get, op->span_));
      }
      result = std::make_shared<SeqStmts>(stmts, op->span_);
    }

    // For each scope-local SSA post-store alias we elided from output_vars,
    // look up the already-renamed store target and map the alias body
    // pointer to that fresh var so later parent-function references resolve
    // correctly.
    for (const auto& [alias_ptr, target_ptr] : deferred_post_store_aliases) {
      auto rename_it = store_target_renames_.find(target_ptr);
      if (rename_it != store_target_renames_.end()) {
        store_target_renames_[alias_ptr] = rename_it->second;
      }
    }
    return result;
  }

  /**
   * @brief Generate a fresh SSA name by incrementing the numeric suffix.
   *
   * E.g. "buf_0" -> "buf_1", "x_2" -> "x_3".  Falls back to appending "_1".
   */
  [[nodiscard]] std::string GenerateFreshSSAName(const std::string& original_name) const {
    std::unordered_set<std::string> used_names;
    for (const auto& [var, _] : var_types_) {
      used_names.insert(var->name_hint_);
    }
    return auto_name::GenerateFreshNameLike(original_name, used_names);
  }

  /**
   * @brief Create a fresh Var for a store-target output and register the rename.
   *
   * Updates var_types_, var_objects_, and store_target_renames_ so that subsequent
   * statements visited by the mutator will use the new variable.
   */
  VarPtr CreateFreshStoreTargetVar(const VarPtr& original_var, const Span& span) {
    std::string fresh_name = GenerateFreshSSAName(original_var->name_hint_);
    auto type = original_var->GetType();
    auto fresh_var = std::make_shared<Var>(fresh_name, type, span);
    store_target_renames_[original_var.get()] = fresh_var;
    var_types_[fresh_var.get()] = type;
    var_objects_[fresh_var.get()] = fresh_var;
    known_names_.insert(fresh_name);
    // Also update the original pointer so subsequent scopes pass the renamed var as call args
    var_objects_[original_var.get()] = fresh_var;
    return fresh_var;
  }

  /**
   * @brief Generate a naming suffix from hierarchy level and optional role.
   *
   * Produces lowercase suffixes like "_host_sub_worker_", "_global_orch_", "_chip_".
   */
  static std::string GenerateHierarchySuffix(Level level, const std::optional<Role>& role) {
    std::string name = "_";
    switch (level) {
      case Level::UNDEFINED:
        name += "undefined";
        break;
      case Level::AIV:
        name += "aiv";
        break;
      case Level::AIC:
        name += "aic";
        break;
      case Level::CORE_GROUP:
        name += "core_group";
        break;
      case Level::CHIP_DIE:
        name += "chip_die";
        break;
      case Level::CHIP:
        name += "chip";
        break;
      case Level::HOST:
        name += "host";
        break;
      case Level::CLUSTER_0:
        name += "cluster0";
        break;
      case Level::CLUSTER_1:
        name += "cluster1";
        break;
      case Level::CLUSTER_2:
        name += "cluster2";
        break;
      case Level::GLOBAL:
        name += "global";
        break;
    }
    if (role.has_value()) {
      name += (role.value() == Role::Orchestrator) ? "_orch" : "_sub_worker";
    }
    return name + "_";
  }

  /// Infer parameter directions for the outlined function by examining the scope body.
  ///
  /// Strategy:
  ///   1. Mark tile.store targets (from ``store_output_set``) as InOut
  ///   2. Mark tensor.assemble destinations as InOut (SSA-pure but
  ///      semantically a destination update; without this the spmd wrapper for
  ///      ``for n0 in pl.spmd(...): out = pl.assemble(out, slice, [...])``
  ///      keeps direction In on the shared output and the orchestration
  ///      codegen drops the SSA-result alias for the inout call)
  ///   3. Merge ``Out``/``InOut`` directions from inner GlobalVar calls
  std::vector<ParamDirection> InferParamDirections(
      const std::vector<VarPtr>& input_vars, const StmtPtr& body,
      const std::unordered_set<const Var*>& store_output_set) const {
    std::vector<ParamDirection> directions(input_vars.size(), ParamDirection::In);

    // Build input_var pointer → index map (shared by every inference step below).
    std::unordered_map<const Var*, size_t> var_to_idx;
    for (size_t i = 0; i < input_vars.size(); ++i) {
      var_to_idx[input_vars[i].get()] = i;
    }

    // Step 1: mark tile.store targets as InOut
    for (size_t i = 0; i < input_vars.size(); ++i) {
      if (store_output_set.count(input_vars[i].get())) {
        directions[i] = ParamDirection::InOut;
      }
    }

    // Step 2: mark tensor.assemble destinations as InOut. ``tensor.assemble``
    // is SSA-pure (returns a fresh Tensor) but the first arg is a destination
    // that the result aliases — when the destination is a parameter the
    // function effectively reads and writes the same backing buffer.
    class AssembleDestUpgrader : public IRVisitor {
     public:
      AssembleDestUpgrader(const std::unordered_map<const Var*, size_t>& var_to_idx,
                           std::vector<ParamDirection>& directions)
          : var_to_idx_(var_to_idx), directions_(directions) {}

     protected:
      void VisitExpr_(const CallPtr& call) override {
        auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
        if (opnode && opnode->name_ == "tensor.assemble" && !call->args_.empty()) {
          if (auto var = As<Var>(call->args_[0])) {
            auto it = var_to_idx_.find(var.get());
            if (it != var_to_idx_.end() && directions_[it->second] == ParamDirection::In) {
              directions_[it->second] = ParamDirection::InOut;
            }
          }
        }
        IRVisitor::VisitExpr_(call);
      }

     private:
      const std::unordered_map<const Var*, size_t>& var_to_idx_;
      std::vector<ParamDirection>& directions_;
    };
    AssembleDestUpgrader(var_to_idx, directions).VisitStmt(body);

    if (!program_) return directions;

    // Step 3: collect all GlobalVar function calls in the body and merge
    // ``Out``/``InOut`` directions from their callees onto our parameters.
    class CallFinder : public IRVisitor {
     public:
      std::vector<CallPtr> found_calls;
      void VisitExpr_(const CallPtr& call) override {
        if (std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
          found_calls.push_back(call);
        }
        IRVisitor::VisitExpr_(call);
      }
    };

    CallFinder finder;
    finder.VisitStmt(body);
    if (finder.found_calls.empty()) return directions;

    // Merge directions from all calls, preferring Out/InOut over In
    for (const auto& call : finder.found_calls) {
      auto gv = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
      if (!gv) continue;
      auto callee = program_->GetFunction(gv->name_);
      if (!callee) continue;
      const auto& call_args = call->args_;
      const auto& callee_dirs = callee->param_directions_;
      for (size_t arg_idx = 0; arg_idx < call_args.size() && arg_idx < callee_dirs.size(); ++arg_idx) {
        auto arg_var = As<Var>(call_args[arg_idx]);
        if (!arg_var) continue;
        auto it = var_to_idx.find(arg_var.get());
        if (it == var_to_idx.end()) continue;
        // Prefer Out/InOut over In
        ParamDirection d = callee_dirs[arg_idx];
        if (d == ParamDirection::Out || d == ParamDirection::InOut) {
          directions[it->second] = d;
        }
      }
    }

    return directions;
  }

  std::string func_name_;
  std::unordered_map<const Var*, TypePtr> var_types_;
  std::unordered_map<const Var*, VarPtr> var_objects_;
  std::unordered_set<std::string> known_names_;
  std::unordered_set<const Var*> required_outputs_;
  /// Accumulates across scopes intentionally (not saved/restored like func_name_
  /// etc.) so that subsequent scopes and statements see the renamed variables.
  std::unordered_map<const Var*, VarPtr> store_target_renames_;
  ScopeKind target_scope_kind_;
  FunctionType outlined_func_type_;
  std::string name_suffix_;
  ProgramPtr program_;
  int scope_counter_ = 0;
  /// True while we are visiting the body of a non-target ScopeStmt — a
  /// target-kind scope encountered here cannot leak locally-defined vars to
  /// any surrounding context (scope boundaries confine them), so the
  /// used_after fallback exposes only store targets.
  bool inside_nested_scope_body_ = false;
  std::vector<FunctionPtr> outlined_functions_;
};

// ============================================================================
// Property verifier: ScopeKind absence
// ============================================================================

/// Verifies that a given ScopeKind does not appear in an IR subtree.
///
/// Used by outline passes to confirm that all scopes of a particular kind
/// have been successfully outlined into separate functions.
///
/// Usage:
///   ScopeKindAbsenceVerifier<ScopeKind::InCore> verifier(diagnostics, "PassName", "error message");
///   verifier.VisitStmt(func->body_);
template <ScopeKind Kind>
class ScopeKindAbsenceVerifier : public IRVisitor {
 public:
  ScopeKindAbsenceVerifier(std::vector<Diagnostic>& diagnostics, std::string pass_name, std::string message)
      : diagnostics_(diagnostics), pass_name_(std::move(pass_name)), message_(std::move(message)) {}

  template <typename ScopeT>
  void CheckKind(const std::shared_ptr<const ScopeT>& op) {
    if (!op) return;
    if (op->GetScopeKind() == Kind) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, pass_name_, 0, message_, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const InCoreScopeStmtPtr& op) override { CheckKind(op); }
  void VisitStmt_(const AutoInCoreScopeStmtPtr& op) override { CheckKind(op); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { CheckKind(op); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { CheckKind(op); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override { CheckKind(op); }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string pass_name_;
  std::string message_;
};

}  // namespace outline_utils
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SCOPE_OUTLINE_UTILS_H_

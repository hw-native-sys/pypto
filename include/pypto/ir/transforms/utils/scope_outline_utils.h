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

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/substitute_vars.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace outline_utils {

// ============================================================================
// Helper visitors/mutators shared by scope-outlining passes
// ============================================================================

/** @brief Visitor to collect all variable references in an IR subtree (by pointer identity). */
class VarRefCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> var_refs;

 protected:
  void VisitExpr_(const VarPtr& op) override { var_refs.insert(op.get()); }

  void VisitExpr_(const IterArgPtr& op) override {
    var_refs.insert(op.get());
    // Use dynamic_pointer_cast (not As<Var>) to match both Var and IterArg,
    // avoiding recursive initValue_ traversal that would pull in outer-scope vars.
    if (op->initValue_) {
      auto as_var = std::dynamic_pointer_cast<const Var>(op->initValue_);
      if (as_var) {
        var_refs.insert(as_var.get());
      } else {
        VisitExpr(op->initValue_);
      }
    }
  }
};

/** @brief Visitor to collect all variable definitions in an IR subtree (by pointer identity). */
class VarDefCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> var_defs;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    var_defs.insert(op->var_.get());
    // Don't visit the RHS - we only care about definitions
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    var_defs.insert(op->loop_var_.get());
    for (const auto& iter_arg : op->iter_args_) {
      var_defs.insert(iter_arg.get());
    }
    for (const auto& return_var : op->return_vars_) {
      var_defs.insert(return_var.get());
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& iter_arg : op->iter_args_) {
      var_defs.insert(iter_arg.get());
    }
    for (const auto& return_var : op->return_vars_) {
      var_defs.insert(return_var.get());
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& return_var : op->return_vars_) {
      var_defs.insert(return_var.get());
    }
    IRVisitor::VisitStmt_(op);
  }
};

/**
 * @brief Visitor to collect target tensors of tile.store calls (by pointer identity).
 *
 * These tensors are modified via side-effect inside scopes but are not
 * captured by VarDefCollector since they are defined externally.  The third
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
                FunctionType outlined_func_type, std::string name_suffix)
      : func_name_(std::move(func_name)),
        var_types_(var_types),
        var_objects_(var_objects),
        known_names_(known_names),
        target_scope_kind_(target_scope_kind),
        outlined_func_type_(outlined_func_type),
        name_suffix_(std::move(name_suffix)) {}

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
      if (scope && scope->scope_kind_ == target_scope_kind_) {
        // Collect variables referenced in all subsequent statements
        VarRefCollector after_ref_collector;
        for (size_t j = i + 1; j < op->stmts_.size(); ++j) {
          after_ref_collector.VisitStmt(op->stmts_[j]);
        }
        // Also include variables required by parent scope
        auto used_after = after_ref_collector.var_refs;
        used_after.insert(required_outputs_.begin(), required_outputs_.end());

        // When no context is available (no subsequent statements and no parent
        // requirements), fall back to standalone behaviour: treat all
        // scope-defined vars + store targets as outputs.  This happens when
        // a single ScopeStmt is wrapped in SeqStmts inside a control-flow
        // body (if/for/while) where the outer context hasn't propagated
        // required_outputs_.
        if (used_after.empty()) {
          VarDefCollector fallback_def;
          fallback_def.VisitStmt(scope->body_);
          StoreTargetCollector fallback_store;
          fallback_store.VisitStmt(scope->body_);
          used_after = fallback_def.var_defs;
          used_after.insert(fallback_store.store_targets.begin(), fallback_store.store_targets.end());
        }

        // Outline this scope with context about what's used after
        auto outlined_stmt = OutlineScope(scope, used_after);
        new_stmts.push_back(outlined_stmt);
        changed = true;
      } else {
        // Recursively visit non-scope statements
        auto visited = VisitStmt(op->stmts_[i]);
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
   * @brief Handle standalone ScopeStmts (not inside SeqStmts).
   *
   * When a scope appears outside a SeqStmts, all defined variables are outputs.
   */
  StmtPtr VisitStmt_(const ScopeStmtPtr& op) override {
    if (op->scope_kind_ != target_scope_kind_) {
      return IRMutator::VisitStmt_(op);
    }

    // Without context, treat all defined variables + store targets as outputs
    VarDefCollector def_collector;
    def_collector.VisitStmt(op->body_);

    StoreTargetCollector store_collector;
    store_collector.VisitStmt(op->body_);
    def_collector.var_defs.insert(store_collector.store_targets.begin(), store_collector.store_targets.end());

    return OutlineScope(op, def_collector.var_defs);
  }

 private:
  /**
   * @brief Outline a single scope into a separate function.
   *
   * @param op The scope statement to outline
   * @param used_after Variables (by pointer) used in subsequent statements (determines outputs)
   */
  StmtPtr OutlineScope(const ScopeStmtPtr& op, const std::unordered_set<const Var*>& used_after) {
    // Generate unique function name (use level/role-aware suffix for Hierarchy scopes)
    std::string suffix = (op->scope_kind_ == ScopeKind::Hierarchy && op->level_.has_value())
                             ? GenerateHierarchySuffix(op->level_.value(), op->role_)
                             : name_suffix_;
    std::ostringstream name_stream;
    name_stream << func_name_ << suffix << scope_counter_++;
    std::string outlined_func_name = name_stream.str();

    // Analyze the scope body for inputs and outputs (before recursing)
    VarRefCollector ref_collector;
    ref_collector.VisitStmt(op->body_);

    VarDefCollector def_collector;
    def_collector.VisitStmt(op->body_);

    // Inputs: variables referenced but not defined in the scope.
    // Look up the VarPtr from var_objects_ (outer symbol table) for each input.
    std::vector<VarPtr> input_vars;
    for (const Var* var_ptr : ref_collector.var_refs) {
      if (!def_collector.var_defs.count(var_ptr)) {
        auto obj_it = var_objects_.find(var_ptr);
        CHECK(obj_it != var_objects_.end())
            << "Variable " << var_ptr->name_hint_ << " not found in var_objects";
        input_vars.push_back(obj_it->second);
      }
    }
    std::sort(input_vars.begin(), input_vars.end(),
              [](const VarPtr& a, const VarPtr& b) { return a->name_hint_ < b->name_hint_; });

    // Outputs: variables defined in the scope AND used after it
    std::vector<VarPtr> output_vars;
    std::unordered_set<const Var*> store_output_set;

    // Collect type info from scope body for output variables
    VarCollector scope_var_collector;
    scope_var_collector.VisitStmt(op->body_);

    for (const Var* var_ptr : def_collector.var_defs) {
      if (used_after.count(var_ptr)) {
        auto scope_it = scope_var_collector.var_objects.find(var_ptr);
        CHECK(scope_it != scope_var_collector.var_objects.end())
            << "Variable " << var_ptr->name_hint_ << " not found in scope body";
        output_vars.push_back(scope_it->second);
      }
    }

    // Also treat store targets as outputs: external tensors modified via
    // tile.store.  These represent side-effect outputs that must be
    // returned regardless of whether they appear in used_after, because the
    // store mutates an externally-visible buffer (e.g. loop-carried state).
    //
    // Track two pointer identities per store target:
    //   - var_objects_ pointer (ext_it->second.get()) — goes into output_vars
    //     and store_output_set for consistent classification
    //   - body pointer (var_ptr) — kept in store_body_ptrs for the
    //     StoreEvalToAssignMutator, which matches against the un-substituted
    //     scope body where store targets retain their original pointers
    StoreTargetCollector store_collector;
    store_collector.VisitStmt(op->body_);
    std::unordered_map<const Var*, const Var*> store_body_ptrs;
    for (const Var* var_ptr : store_collector.store_targets) {
      if (!def_collector.var_defs.count(var_ptr)) {
        auto ext_it = var_objects_.find(var_ptr);
        CHECK(ext_it != var_objects_.end())
            << "Variable " << var_ptr->name_hint_ << " not found in var_objects";
        output_vars.push_back(ext_it->second);
        store_output_set.insert(ext_it->second.get());
        store_body_ptrs[ext_it->second.get()] = var_ptr;
      }
    }

    std::sort(output_vars.begin(), output_vars.end(),
              [](const VarPtr& a, const VarPtr& b) { return a->name_hint_ < b->name_hint_; });

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

    // Create fresh parameters for the outlined function
    std::vector<VarPtr> input_params;
    std::vector<ParamDirection> input_param_directions;
    std::unordered_map<const Var*, VarPtr> var_substitution_map;
    for (const auto& input_var : input_vars) {
      auto param_var = std::make_shared<Var>(input_var->name_hint_, input_var->GetType(), op->span_);
      input_params.push_back(param_var);
      input_param_directions.push_back(ParamDirection::In);
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
    // SubstituteVars, since store_body_ptrs uses the original body Var pointers.
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
    auto transformed_body = SubstituteVars(pre_sub_body, var_substitution_map);

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

    // Register the outlined function (propagate level/role from ScopeStmt for Hierarchy scopes)
    auto outlined_func =
        std::make_shared<Function>(outlined_func_name, input_params, input_param_directions, return_types,
                                   outlined_body, op->span_, outlined_func_type_, op->level_, op->role_);
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

    // Create assignments for output variables in the parent function
    if (output_vars.empty()) {
      return std::make_shared<EvalStmt>(call_expr, op->span_);
    } else if (output_vars.size() == 1) {
      auto output_var = resolve_call_site_var(output_vars[0]);
      return std::make_shared<AssignStmt>(output_var, call_expr, op->span_);
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
      return std::make_shared<SeqStmts>(stmts, op->span_);
    }
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
   * Produces lowercase suffixes like "_host_worker_", "_global_orch_", "_chip_".
   */
  static std::string GenerateHierarchySuffix(Level level, const std::optional<Role>& role) {
    std::string name = "_";
    switch (level) {
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
      name += (role.value() == Role::Orchestrator) ? "_orch" : "_worker";
    }
    return name + "_";
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
  int scope_counter_ = 0;
  std::vector<FunctionPtr> outlined_functions_;
};

}  // namespace outline_utils
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SCOPE_OUTLINE_UTILS_H_

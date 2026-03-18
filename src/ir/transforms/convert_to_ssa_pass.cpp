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

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
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
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// SSA naming strategy: each variable name (e.g. "tmp_0", "tmp_1") is treated
// as a unique identity — no suffix stripping or base name normalization.

/**
 * @brief Collects all assigned variable names in a statement.
 *
 * Used to pre-analyze loop bodies to find which outer variables are modified,
 * allowing us to create iter_args before visiting the body.
 */
class AssignmentCollector : public IRVisitor {
 public:
  std::set<std::string> assigned_vars;

  void Collect(const StmtPtr& stmt) { VisitStmt(stmt); }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    assigned_vars.insert(op->var_->name_hint_);
    // Also visit the value in case of nested assignments
    VisitExpr(op->value_);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    // Don't recurse into nested for loops - they handle their own iter_args
    // But we do need to record the loop variable
    assigned_vars.insert(op->loop_var_->name_hint_);
    // Visit the body to collect assignments
    VisitStmt(op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    // Don't recurse into nested while loops - they handle their own iter_args
    // Visit condition to collect any assignments (though unusual)
    VisitExpr(op->condition_);
    // Visit the body to collect assignments
    VisitStmt(op->body_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    // Visit both branches
    VisitStmt(op->then_body_);
    if (op->else_body_.has_value()) {
      VisitStmt(*op->else_body_);
    }
  }

  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& s : op->stmts_) {
      VisitStmt(s);
    }
  }
};

/**
 * @brief Collects all variable names *used* (referenced) in a statement subtree.
 *
 * Used to determine which variables from a loop body are referenced in
 * subsequent statements, enabling precise escaping-variable detection.
 */
class UseCollector : public IRVisitor {
 public:
  std::set<std::string> used_vars;

  void Collect(const StmtPtr& stmt) {
    if (stmt) VisitStmt(stmt);
  }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) used_vars.insert(op->name_hint_);
    IRVisitor::VisitVarLike_(op);
  }
};

/**
 * @brief SSA Converter - Transforms non-SSA IR to SSA form
 *
 * This mutator converts IR with multiple assignments per variable to SSA form by:
 * 1. Renaming variables with version suffixes (x -> x_0, x_1, x_2)
 * 2. Adding phi nodes (return_vars + YieldStmt) for IfStmt control flow
 * 3. Converting loop-modified variables to iter_args + return_vars pattern
 * 4. Promoting variables first defined inside loops to iter_args + return_vars
 *    so they are accessible after the loop (escaping variables)
 */
class SSAConverter : public IRMutator {
 public:
  SSAConverter() = default;

  /**
   * @brief Convert a function to SSA form
   */
  FunctionPtr Convert(const FunctionPtr& func) {
    // Store original function parameter info for Out-parameter matching
    orig_params_ = func->params_;
    orig_param_directions_ = func->param_directions_;

    // Initialize version counters for parameters
    for (size_t i = 0; i < func->params_.size(); ++i) {
      const auto& var = func->params_[i];
      std::string base_name = var->name_hint_;
      int version = NextVersion(base_name);
      auto versioned_param = CreateVersionedVar(var, base_name, version);
      current_version_[base_name] = versioned_param;
      new_params_.push_back(versioned_param);
      new_param_directions_.push_back(func->param_directions_[i]);
    }

    // Transform the function body
    StmtPtr new_body = nullptr;
    if (func->body_) {
      new_body = VisitStmt(func->body_);
    }

    // Create the new function with versioned parameters
    return std::make_shared<Function>(func->name_, new_params_, new_param_directions_, func->return_types_,
                                      new_body, func->span_, func->func_type_);
  }

 protected:
  // Override expression visitation to replace Var with current version
  ExprPtr VisitExpr_(const VarPtr& op) override {
    std::string base_name = op->name_hint_;
    auto it = current_version_.find(base_name);
    if (it != current_version_.end()) {
      return it->second;
    }
    // Variable not found in current scope - return as-is
    // This can happen for variables that are only defined once
    return op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    // Visit the initValue first
    auto new_init = VisitExpr(op->initValue_);

    // IterArgs are handled specially - they become the current version within loop
    std::string base_name = op->name_hint_;
    auto it = current_version_.find(base_name);
    if (it != current_version_.end()) {
      // Return the current version (which should be the iter_arg itself)
      return it->second;
    }

    // If no current version, create one
    if (new_init != op->initValue_) {
      return std::make_shared<IterArg>(op->name_hint_, op->GetType(), new_init, op->span_);
    }
    return op;
  }

  // Override assignment statement to create versioned variables
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // First, visit the RHS expression (uses current versions)
    auto new_value = VisitExpr(op->value_);

    // Create a new versioned variable for LHS
    std::string base_name = op->var_->name_hint_;
    int version = NextVersion(base_name);
    auto new_var = CreateVersionedVar(op->var_, base_name, version);

    // Update current version mapping
    current_version_[base_name] = new_var;

    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

  // Override IfStmt to handle phi nodes
  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    // Visit condition first (in current scope)
    auto new_condition = VisitExpr(op->condition_);

    // Save current versions before branches
    auto versions_before = current_version_;

    // Visit then branch
    EnterScope();
    auto new_then = VisitStmt(op->then_body_);
    auto versions_after_then = current_version_;
    ExitScope();

    // Restore and visit else branch
    current_version_ = versions_before;
    std::optional<StmtPtr> new_else;
    std::unordered_map<std::string, VarPtr> versions_after_else;

    if (op->else_body_.has_value()) {
      EnterScope();
      new_else = VisitStmt(*op->else_body_);
      versions_after_else = current_version_;
      ExitScope();
    } else {
      versions_after_else = versions_before;
    }

    // Find variables that diverged between branches (need phi nodes).
    // Consider variables that existed before the if (re-assigned in a branch),
    // AND variables that are new but defined in BOTH branches (both branches
    // created the same base name independently).
    std::vector<std::string> phi_vars;
    std::set<std::string> checked_vars;

    // Check variables from then branch
    for (const auto& [base_name, var] : versions_after_then) {
      checked_vars.insert(base_name);
      auto before_it = versions_before.find(base_name);

      if (before_it != versions_before.end()) {
        // Variable existed before: check if either branch changed it
        bool changed_in_then = (before_it->second != var);
        auto else_it = versions_after_else.find(base_name);
        bool changed_in_else = (else_it != versions_after_else.end() && before_it->second != else_it->second);

        if (changed_in_then || changed_in_else) {
          phi_vars.push_back(base_name);
        }
      } else {
        // Variable is new (not in versions_before). If it exists in BOTH branches,
        // it needs a phi node to merge the two independent definitions.
        auto else_it = versions_after_else.find(base_name);
        if (else_it != versions_after_else.end()) {
          phi_vars.push_back(base_name);
        }
      }
    }

    // Check variables from else branch that weren't in then
    for (const auto& [base_name, var] : versions_after_else) {
      if (checked_vars.count(base_name)) continue;

      auto before_it = versions_before.find(base_name);
      // Skip variables not defined before the if (branch-local only in else)
      if (before_it == versions_before.end()) continue;

      if (before_it->second != var) {
        phi_vars.push_back(base_name);
      }
    }

    // Sort phi_vars for deterministic ordering across platforms
    std::sort(phi_vars.begin(), phi_vars.end());

    // If no variables diverged, just return the updated if statement
    if (phi_vars.empty() && op->return_vars_.empty()) {
      current_version_ = versions_before;
      return std::make_shared<IfStmt>(new_condition, new_then, new_else, std::vector<VarPtr>{}, op->span_);
    }

    // Restore to pre-branch state before creating phi outputs
    current_version_ = versions_before;

    // Create return_vars and yields for phi nodes
    std::vector<VarPtr> return_vars;
    std::vector<ExprPtr> then_yields;
    std::vector<ExprPtr> else_yields;

    for (const auto& base_name : phi_vars) {
      // Get versions from each branch
      VarPtr then_var = versions_after_then.count(base_name) ? versions_after_then.at(base_name)
                                                             : versions_before.at(base_name);
      VarPtr else_var = versions_after_else.count(base_name) ? versions_after_else.at(base_name)
                                                             : versions_before.at(base_name);

      // Create phi output variable with new version
      int phi_version = NextVersion(base_name);
      auto phi_var = std::make_shared<Var>(base_name + "_" + std::to_string(phi_version), then_var->GetType(),
                                           op->span_);

      return_vars.push_back(phi_var);
      then_yields.push_back(then_var);
      else_yields.push_back(else_var);

      // Update current version to phi output
      current_version_[base_name] = phi_var;
    }

    // Preserve any existing return_vars (version them)
    for (const auto& existing_rv : op->return_vars_) {
      std::string base_name = existing_rv->name_hint_;
      // Only add if not already in phi_vars
      bool already_handled = false;
      for (const auto& pv : phi_vars) {
        if (pv == base_name) {
          already_handled = true;
          break;
        }
      }
      if (!already_handled) {
        int rv_version = NextVersion(base_name);
        auto versioned_rv = std::make_shared<Var>(base_name + "_" + std::to_string(rv_version),
                                                  existing_rv->GetType(), existing_rv->span_);
        return_vars.push_back(versioned_rv);
        current_version_[base_name] = versioned_rv;
      }
    }

    // Append YieldStmt to branches
    auto then_with_yield = AppendYield(new_then, then_yields, op->span_);
    StmtPtr else_with_yield;
    if (new_else.has_value()) {
      else_with_yield = AppendYield(*new_else, else_yields, op->span_);
    } else {
      // Create an else branch with just the yield
      else_with_yield = std::make_shared<YieldStmt>(else_yields, op->span_);
    }

    return std::make_shared<IfStmt>(new_condition, then_with_yield, std::make_optional(else_with_yield),
                                    return_vars, op->span_);
  }

  // Override ForStmt to handle loop-carried and escaping variables
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Save future_uses_ before visiting children — nested SeqStmts will overwrite it
    auto saved_future_uses = future_uses_;

    // Visit range expressions in outer scope
    auto new_start = VisitExpr(op->start_);
    auto new_stop = VisitExpr(op->stop_);
    auto new_step = VisitExpr(op->step_);

    // Save outer scope versions
    auto versions_before = current_version_;

    // Process existing iter_args (visit their init values in outer scope)
    std::vector<IterArgPtr> new_iter_args;
    for (const auto& iter_arg : op->iter_args_) {
      auto new_init = VisitExpr(iter_arg->initValue_);
      auto new_ia =
          std::make_shared<IterArg>(iter_arg->name_hint_, iter_arg->GetType(), new_init, iter_arg->span_);
      new_iter_args.push_back(new_ia);
    }

    // PRE-ANALYSIS: Find which outer variables are assigned in the loop body.
    // This allows us to create iter_args BEFORE visiting the body.
    AssignmentCollector collector;
    collector.Collect(op->body_);

    // Identify loop-carried variables: assigned in body AND existed before the loop
    std::string loop_var_base = op->loop_var_->name_hint_;
    std::vector<std::string> loop_carried_vars;
    for (const auto& assigned_name : collector.assigned_vars) {
      // Skip loop variable
      if (assigned_name == loop_var_base) continue;

      // Skip existing iter_args
      bool is_existing_iter_arg = false;
      for (const auto& ia : op->iter_args_) {
        if (ia->name_hint_ == assigned_name) {
          is_existing_iter_arg = true;
          break;
        }
      }
      if (is_existing_iter_arg) continue;

      // Check if variable existed before the loop
      auto before_it = versions_before.find(assigned_name);
      if (before_it != versions_before.end()) {
        loop_carried_vars.push_back(assigned_name);
      }
    }

    // Create iter_args for loop-carried variables BEFORE visiting the body
    std::vector<VarPtr> return_vars;
    for (const auto& base_name : loop_carried_vars) {
      auto init_var = versions_before.at(base_name);
      int ia_version = NextVersion(base_name);
      auto iter_arg = std::make_shared<IterArg>(base_name + "_iter_" + std::to_string(ia_version),
                                                init_var->GetType(), init_var, op->span_);
      new_iter_args.push_back(iter_arg);

      // Create return var for post-loop access
      int rv_version = NextVersion(base_name);
      auto return_var =
          std::make_shared<Var>(base_name + "_" + std::to_string(rv_version), init_var->GetType(), op->span_);
      return_vars.push_back(return_var);
    }

    // Enter loop scope
    EnterScope();

    // Create versioned loop variable
    int loop_var_version = NextVersion(loop_var_base);
    auto new_loop_var = std::make_shared<Var>(loop_var_base + "_" + std::to_string(loop_var_version),
                                              op->loop_var_->GetType(), op->loop_var_->span_);
    current_version_[loop_var_base] = new_loop_var;

    RegisterIterArgs(new_iter_args);

    // Visit loop body - now it will correctly reference iter_args
    auto new_body = VisitStmt(op->body_);
    auto versions_after_body = current_version_;

    // Exit loop scope
    ExitScope();

    // Restore current_version_ to pre-loop state to prevent scope leakage
    // of loop-local variables (e.g. x_chunk defined inside loop should not
    // be visible after the loop exits)
    current_version_ = versions_before;

    // Update outer scope to use return_vars for loop-carried variables
    for (size_t i = 0; i < loop_carried_vars.size(); ++i) {
      current_version_[loop_carried_vars[i]] = return_vars[i];
    }

    RegisterExistingReturnVars(op->iter_args_, op->return_vars_);

    // Handle escaping variables: variables first defined inside the loop body
    // that may be used after the loop (e.g., out = tile.store(...) inside a loop
    // followed by "return out" after the loop). These need iter_args + return_vars
    // to properly escape the loop scope in SSA form.
    auto escaping_vars = FindEscapingVars(versions_after_body, versions_before, collector.assigned_vars,
                                          loop_var_base, new_iter_args, saved_future_uses);

    for (const auto& base_name : escaping_vars) {
      const auto& final_var = versions_after_body.at(base_name);
      auto type = final_var->GetType();

      // Find initial value: prefer Out parameter with matching type, fall back to any match
      auto init_value = FindInitValueForEscapingVar(type, versions_before);
      if (!init_value) {
        // Last resort: use the final_var itself (may cause issues for zero-trip loops,
        // but preserves correctness for non-zero-trip loops which is the common case)
        init_value = final_var;
      }

      int ia_version = NextVersion(base_name);
      auto iter_arg = std::make_shared<IterArg>(base_name + "_iter_" + std::to_string(ia_version), type,
                                                init_value, op->span_);
      new_iter_args.push_back(iter_arg);

      int rv_version = NextVersion(base_name);
      auto rv = std::make_shared<Var>(base_name + "_" + std::to_string(rv_version), type, op->span_);
      return_vars.push_back(rv);

      // Register return_var in outer scope so post-loop references resolve correctly
      current_version_[base_name] = rv;
    }

    // Collect yield values: first existing iter_args, then loop-carried, then escaping
    std::vector<ExprPtr> yield_values;

    // First, collect yields for existing (original) iter_args
    if (auto yield_stmt = GetLastYieldStmt(new_body)) {
      yield_values = yield_stmt->value_;
    }

    // Then add yield values for new loop-carried variables
    for (const auto& base_name : loop_carried_vars) {
      // Get the final version from within the loop
      const auto& final_var = versions_after_body.at(base_name);
      yield_values.push_back(final_var);
    }

    // Then add yield values for escaping variables
    for (const auto& base_name : escaping_vars) {
      const auto& final_var = versions_after_body.at(base_name);
      yield_values.push_back(final_var);
    }

    // Copy existing return_vars (from explicit iter_args in original code)
    for (const auto& rv : op->return_vars_) {
      return_vars.push_back(rv);
    }

    // Update body with new yield
    StmtPtr final_body = new_body;
    if (!yield_values.empty()) {
      final_body = ReplaceOrAppendYield(new_body, yield_values, op->span_);
    }

    return std::make_shared<ForStmt>(new_loop_var, new_start, new_stop, new_step, new_iter_args, final_body,
                                     return_vars, op->span_, op->kind_, op->chunk_size_, op->chunk_policy_,
                                     op->loop_origin_);
  }

  // Override WhileStmt to handle loop-carried and escaping variables
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    // Save future_uses_ before visiting children — nested SeqStmts will overwrite it
    auto saved_future_uses = future_uses_;

    // Save outer scope versions
    auto versions_before = current_version_;

    // Process existing iter_args (visit their init values in outer scope)
    std::vector<IterArgPtr> new_iter_args;
    for (const auto& iter_arg : op->iter_args_) {
      auto new_init = VisitExpr(iter_arg->initValue_);
      auto new_ia =
          std::make_shared<IterArg>(iter_arg->name_hint_, iter_arg->GetType(), new_init, iter_arg->span_);
      new_iter_args.push_back(new_ia);
    }

    // PRE-ANALYSIS: Find which outer variables are assigned in the loop body
    AssignmentCollector collector;
    collector.Collect(op->body_);
    // Also collect from condition (though unusual, it's possible)
    collector.Collect(std::make_shared<EvalStmt>(op->condition_, op->span_));

    // Identify loop-carried variables: assigned in body AND existed before the loop
    std::vector<std::string> loop_carried_vars;
    for (const auto& assigned_name : collector.assigned_vars) {
      // Skip existing iter_args
      bool is_existing_iter_arg = false;
      for (const auto& ia : op->iter_args_) {
        if (ia->name_hint_ == assigned_name) {
          is_existing_iter_arg = true;
          break;
        }
      }
      if (is_existing_iter_arg) continue;

      // Check if variable existed before the loop
      auto before_it = versions_before.find(assigned_name);
      if (before_it != versions_before.end()) {
        loop_carried_vars.push_back(assigned_name);
      }
    }

    // Create iter_args for loop-carried variables BEFORE visiting the body
    std::vector<VarPtr> new_loop_carried_return_vars;
    for (const auto& base_name : loop_carried_vars) {
      auto init_var = versions_before.at(base_name);
      int ia_version = NextVersion(base_name);
      auto iter_arg = std::make_shared<IterArg>(base_name + "_iter_" + std::to_string(ia_version),
                                                init_var->GetType(), init_var, op->span_);
      new_iter_args.push_back(iter_arg);

      // Create return var for post-loop access
      int rv_version = NextVersion(base_name);
      auto return_var =
          std::make_shared<Var>(base_name + "_" + std::to_string(rv_version), init_var->GetType(), op->span_);
      new_loop_carried_return_vars.push_back(return_var);
    }

    // Enter loop scope
    EnterScope();

    RegisterIterArgs(new_iter_args);

    // Visit condition - it will reference iter_args
    auto new_condition = VisitExpr(op->condition_);

    // Visit loop body - now it will correctly reference iter_args
    auto new_body = VisitStmt(op->body_);
    auto versions_after_body = current_version_;

    // Exit loop scope
    ExitScope();

    // Restore current_version_ to pre-loop state to prevent scope leakage
    current_version_ = versions_before;

    // Build return_vars in same order as new_iter_args and yield_values:
    // First existing return_vars, then new loop-carried return_vars
    std::vector<VarPtr> return_vars;
    for (const auto& rv : op->return_vars_) {
      return_vars.push_back(rv);
    }
    for (const auto& rv : new_loop_carried_return_vars) {
      return_vars.push_back(rv);
    }

    // Update outer scope to use return_vars for loop-carried variables
    for (size_t i = 0; i < loop_carried_vars.size(); ++i) {
      current_version_[loop_carried_vars[i]] = new_loop_carried_return_vars[i];
    }

    RegisterExistingReturnVars(op->iter_args_, op->return_vars_);

    // Handle escaping variables (same as ForStmt — see comment there)
    // WhileStmt has no loop_var, so pass empty string
    auto escaping_vars = FindEscapingVars(versions_after_body, versions_before, collector.assigned_vars, "",
                                          new_iter_args, saved_future_uses);

    for (const auto& base_name : escaping_vars) {
      const auto& final_var = versions_after_body.at(base_name);
      auto type = final_var->GetType();

      auto init_value = FindInitValueForEscapingVar(type, versions_before);
      if (!init_value) {
        init_value = final_var;
      }

      int ia_version = NextVersion(base_name);
      auto iter_arg = std::make_shared<IterArg>(base_name + "_iter_" + std::to_string(ia_version), type,
                                                init_value, op->span_);
      new_iter_args.push_back(iter_arg);

      int rv_version = NextVersion(base_name);
      auto rv = std::make_shared<Var>(base_name + "_" + std::to_string(rv_version), type, op->span_);
      return_vars.push_back(rv);

      current_version_[base_name] = rv;
    }

    // Collect yield values: first existing iter_args, then loop-carried, then escaping
    std::vector<ExprPtr> yield_values;

    // First, collect yields for existing (original) iter_args
    if (auto yield_stmt = GetLastYieldStmt(new_body)) {
      yield_values = yield_stmt->value_;
    }

    // Then add yield values for new loop-carried variables
    for (const auto& base_name : loop_carried_vars) {
      // Get the final version from within the loop
      const auto& final_var = versions_after_body.at(base_name);
      yield_values.push_back(final_var);
    }

    // Then add yield values for escaping variables
    for (const auto& base_name : escaping_vars) {
      const auto& final_var = versions_after_body.at(base_name);
      yield_values.push_back(final_var);
    }

    // Update body with new yield
    StmtPtr final_body = new_body;
    if (!yield_values.empty()) {
      final_body = ReplaceOrAppendYield(new_body, yield_values, op->span_);
    }

    return std::make_shared<WhileStmt>(new_condition, new_iter_args, final_body, return_vars, op->span_);
  }

  // Override SeqStmts to compute future variable uses before processing each statement.
  // This enables precise detection of escaping variables in loop handlers.
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
      // Before processing a loop statement, compute which variables are
      // referenced in all subsequent statements (future uses).
      if (i + 1 < op->stmts_.size()) {
        future_uses_.clear();
        UseCollector use_collector;
        for (size_t j = i + 1; j < op->stmts_.size(); ++j) {
          use_collector.Collect(op->stmts_[j]);
        }
        future_uses_ = use_collector.used_vars;
      } else {
        future_uses_.clear();
      }
      new_stmts.push_back(VisitStmt(op->stmts_[i]));
    }
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }

 private:
  // Version counter per base variable name
  std::unordered_map<std::string, int> version_counter_;

  // Current version of each variable (base_name -> versioned VarPtr)
  std::unordered_map<std::string, VarPtr> current_version_;

  // Scope stack for nested control flow
  std::vector<std::unordered_map<std::string, VarPtr>> scope_stack_;

  // New versioned parameters
  std::vector<VarPtr> new_params_;
  std::vector<ParamDirection> new_param_directions_;

  // Original function parameter info (for Out-parameter matching)
  std::vector<VarPtr> orig_params_;
  std::vector<ParamDirection> orig_param_directions_;

  // Variable names used in statements after the current one (computed by SeqStmts override)
  std::set<std::string> future_uses_;

  /**
   * @brief Strip a generated "_iter_<digits>" suffix from an iter_arg name to recover
   * the base variable name (e.g., "acc_iter_1" → "acc").
   *
   * Only strips if the name ends with "_iter_" followed by one or more digits.
   * Names that merely contain "iter" as part of a word (e.g., "filter_data",
   * "writer_iter_count") are left unchanged.
   */
  static std::string StripIterSuffix(const std::string& name) {
    // Look for "_iter_" followed by digits at the end
    size_t pos = name.rfind("_iter_");
    if (pos == std::string::npos) return name;
    // Verify everything after "_iter_" is digits
    size_t after = pos + 6;  // length of "_iter_"
    if (after >= name.size()) return name;
    for (size_t i = after; i < name.size(); ++i) {
      if (!std::isdigit(static_cast<unsigned char>(name[i]))) return name;
    }
    return name.substr(0, pos);
  }

  /**
   * @brief Get next version number for a base name
   */
  int NextVersion(const std::string& base_name) {
    int version = version_counter_[base_name];
    version_counter_[base_name] = version + 1;
    return version;
  }

  /**
   * @brief Find an initial value for an escaping variable's iter_arg.
   *
   * Searches the pre-loop variable versions for an Out/InOut parameter whose
   * type matches the escaping variable's type. Falls back to any variable with
   * a matching type.
   */
  VarPtr FindInitValueForEscapingVar(const TypePtr& target_type,
                                     const std::unordered_map<std::string, VarPtr>& versions_before) {
    // First pass: look for an Out/InOut parameter with matching type
    for (size_t i = 0; i < orig_params_.size(); ++i) {
      if (orig_param_directions_[i] == ParamDirection::Out ||
          orig_param_directions_[i] == ParamDirection::InOut) {
        auto it = versions_before.find(orig_params_[i]->name_hint_);
        if (it != versions_before.end() && it->second->GetType() == target_type) {
          return it->second;
        }
      }
    }
    // Second pass: look for any pre-loop variable with matching type.
    // Collect and sort by name for deterministic results across runs.
    std::vector<std::pair<std::string, VarPtr>> candidates;
    for (const auto& [name, var] : versions_before) {
      if (var->GetType() == target_type) {
        candidates.emplace_back(name, var);
      }
    }
    if (!candidates.empty()) {
      std::sort(candidates.begin(), candidates.end());
      return candidates.front().second;
    }
    return nullptr;
  }

  /**
   * @brief Identify variables first defined inside a loop body that escape to outer scope.
   *
   * A variable escapes if it is:
   * 1. Present in versions_after_body but not in versions_before
   * 2. Assigned inside the body (not inherited from nested constructs)
   * 3. Actually referenced in subsequent statements (future_uses)
   * 4. Not the loop variable or an already-handled iter_arg
   */
  std::vector<std::string> FindEscapingVars(
      const std::unordered_map<std::string, VarPtr>& versions_after_body,
      const std::unordered_map<std::string, VarPtr>& versions_before,
      const std::set<std::string>& assigned_in_body, const std::string& loop_var_base,
      const std::vector<IterArgPtr>& iter_args, const std::set<std::string>& future_uses) {
    std::vector<std::string> escaping_vars;
    for (const auto& [name, var] : versions_after_body) {
      if (versions_before.count(name)) continue;
      if (name == loop_var_base) continue;
      if (!assigned_in_body.count(name)) continue;
      // Only escape if actually used after the loop
      if (!future_uses.count(name)) continue;
      // Skip variables already handled as iter_args
      bool is_iter_arg = false;
      for (const auto& ia : iter_args) {
        if (StripIterSuffix(ia->name_hint_) == name) {
          is_iter_arg = true;
          break;
        }
      }
      if (is_iter_arg) continue;
      escaping_vars.push_back(name);
    }
    std::sort(escaping_vars.begin(), escaping_vars.end());
    return escaping_vars;
  }

  /**
   * @brief Substitute Var references in a type using current_version_.
   *
   * Handles:
   * - TensorType::shape_ (e.g., Tensor[[M, N], FP32] → Tensor[[M_0, N_0], FP32])
   * - TileType::tile_view.valid_shape
   *
   * This ensures all type-embedded Var references stay consistent after SSA renaming.
   */
  TypePtr SubstituteVarsInType(const TypePtr& type) {
    if (!type) return type;

    // Handle TensorType shapes
    if (auto tensor_type = As<TensorType>(type)) {
      std::vector<ExprPtr> new_shape;
      bool changed = false;
      for (const auto& dim : tensor_type->shape_) {
        auto new_dim = VisitExpr(dim);
        if (new_dim != dim) changed = true;
        new_shape.push_back(new_dim);
      }
      if (changed) {
        return std::make_shared<TensorType>(std::move(new_shape), tensor_type->dtype_, tensor_type->memref_,
                                            tensor_type->tensor_view_);
      }
      return type;
    }

    // Handle TileType tile_view.valid_shape
    auto tile_type = As<TileType>(type);
    if (!tile_type || !tile_type->tile_view_.has_value()) return type;

    const auto& tv = tile_type->tile_view_.value();
    if (tv.valid_shape.empty()) return type;

    std::vector<ExprPtr> new_valid_shape;
    bool changed = false;
    for (const auto& vs : tv.valid_shape) {
      auto new_vs = VisitExpr(vs);
      if (new_vs != vs) changed = true;
      new_valid_shape.push_back(new_vs);
    }
    if (!changed) return type;

    TileView new_tile_view = tv;
    new_tile_view.valid_shape = std::move(new_valid_shape);
    return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, tile_type->memref_,
                                      std::make_optional(std::move(new_tile_view)), tile_type->memory_space_);
  }

  /**
   * @brief Create a versioned variable from an original variable
   *
   * Also substitutes any Var references embedded in the variable's type
   * (e.g., TileType::tile_view.valid_shape) using the current version map.
   */
  VarPtr CreateVersionedVar(const VarPtr& original, const std::string& base_name, int version) {
    std::string versioned_name = base_name + "_" + std::to_string(version);
    auto type = SubstituteVarsInType(original->GetType());
    return std::make_shared<Var>(versioned_name, type, original->span_);
  }

  /**
   * @brief Register iter_args in current_version_ under both their full name
   * and stripped base name (e.g., "acc_iter_1" registers as both "acc_iter_1" and "acc").
   */
  void RegisterIterArgs(const std::vector<IterArgPtr>& iter_args) {
    for (const auto& iter_arg : iter_args) {
      std::string full_name = iter_arg->name_hint_;
      std::string stripped_name = StripIterSuffix(full_name);
      current_version_[stripped_name] = iter_arg;
      if (full_name != stripped_name) {
        current_version_[full_name] = iter_arg;
      }
    }
  }

  /**
   * @brief Map existing iter_args' return_vars back into current_version_
   * using the stripped base name.
   */
  void RegisterExistingReturnVars(const std::vector<IterArgPtr>& iter_args,
                                  const std::vector<VarPtr>& return_vars) {
    for (size_t i = 0; i < iter_args.size() && i < return_vars.size(); ++i) {
      std::string base_name = StripIterSuffix(iter_args[i]->name_hint_);
      current_version_[base_name] = return_vars[i];
    }
  }

  /**
   * @brief Enter a new scope
   */
  void EnterScope() { scope_stack_.push_back(current_version_); }

  /**
   * @brief Exit current scope
   */
  void ExitScope() {
    if (!scope_stack_.empty()) {
      scope_stack_.pop_back();
    }
  }

  /**
   * @brief Append a YieldStmt to a statement
   */
  StmtPtr AppendYield(const StmtPtr& stmt, const std::vector<ExprPtr>& values, const Span& span) {
    if (values.empty()) return stmt;
    return ReplaceOrAppendYield(stmt, values, span);
  }

  /**
   * @brief Get the last YieldStmt from a statement (if any)
   */
  YieldStmtPtr GetLastYieldStmt(const StmtPtr& stmt) {
    if (auto yield = As<YieldStmt>(stmt)) {
      return yield;
    }
    if (auto seq = As<SeqStmts>(stmt)) {
      if (!seq->stmts_.empty()) {
        return As<YieldStmt>(seq->stmts_.back());
      }
    }
    return nullptr;
  }

  /**
   * @brief Replace or append yield statement
   */
  StmtPtr ReplaceOrAppendYield(const StmtPtr& stmt, const std::vector<ExprPtr>& values, const Span& span) {
    auto new_yield = std::make_shared<YieldStmt>(values, span);

    if (auto seq = As<SeqStmts>(stmt)) {
      if (!seq->stmts_.empty() && As<YieldStmt>(seq->stmts_.back())) {
        // Replace last yield
        std::vector<StmtPtr> new_stmts(seq->stmts_.begin(), seq->stmts_.end() - 1);
        new_stmts.push_back(new_yield);
        return SeqStmts::Flatten(std::move(new_stmts), seq->span_);
      }
      // Append yield
      std::vector<StmtPtr> new_stmts = seq->stmts_;
      new_stmts.push_back(new_yield);
      return SeqStmts::Flatten(std::move(new_stmts), seq->span_);
    }

    if (As<YieldStmt>(stmt)) {
      return new_yield;
    }

    // Wrap single statement and yield in SeqStmts
    return SeqStmts::Flatten(std::vector<StmtPtr>{stmt, new_yield}, span);
  }
};

/**
 * @brief Transform function: Convert a function to SSA form
 */
FunctionPtr TransformConvertToSSA(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "ConvertToSSA cannot run on null function";
  SSAConverter converter;
  return converter.Convert(func);
}

}  // namespace

// Factory function
namespace pass {
Pass ConvertToSSA() {
  return CreateFunctionPass(TransformConvertToSSA, "ConvertToSSA", kConvertToSSAProperties);
}
}  // namespace pass

}  // namespace ir
}  // namespace pypto

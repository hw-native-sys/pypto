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

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/**
 * @brief Collects all assigned variable base names in a statement
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
    // Extract base name from assignment target
    assigned_vars.insert(GetBaseName(op->var_->name_));
    // Also visit the value in case of nested assignments
    VisitExpr(op->value_);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    // Don't recurse into nested for loops - they handle their own iter_args
    // But we do need to record the loop variable
    assigned_vars.insert(GetBaseName(op->loop_var_->name_));
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

 private:
  static std::string GetBaseName(const std::string& name) {
    size_t last_underscore = name.rfind('_');
    if (last_underscore != std::string::npos && last_underscore > 0) {
      bool all_digits = true;
      for (size_t i = last_underscore + 1; i < name.size(); ++i) {
        if (!std::isdigit(name[i])) {
          all_digits = false;
          break;
        }
      }
      if (all_digits && last_underscore + 1 < name.size()) {
        return name.substr(0, last_underscore);
      }
    }
    return name;
  }
};

/**
 * @brief SSA Converter - Transforms non-SSA IR to SSA form
 *
 * This mutator converts IR with multiple assignments per variable to SSA form by:
 * 1. Renaming variables with version suffixes (x -> x_0, x_1, x_2)
 * 2. Adding phi nodes (return_vars + YieldStmt) for IfStmt control flow
 * 3. Converting loop-modified variables to iter_args + return_vars pattern
 */
class SSAConverter : public IRMutator {
 public:
  SSAConverter() = default;

  /**
   * @brief Convert a function to SSA form
   */
  FunctionPtr Convert(const FunctionPtr& func) {
    // Initialize version counters for parameters
    for (const auto& param : func->params_) {
      std::string base_name = GetBaseName(param->name_);
      int version = NextVersion(base_name);
      auto versioned_param = CreateVersionedVar(param, base_name, version);
      current_version_[base_name] = versioned_param;
      new_params_.push_back(versioned_param);
    }

    // Transform the function body
    StmtPtr new_body = nullptr;
    if (func->body_) {
      new_body = VisitStmt(func->body_);
    }

    // Create the new function with versioned parameters
    return std::make_shared<Function>(func->name_, new_params_, func->return_types_, new_body, func->span_,
                                      func->func_type_);
  }

 protected:
  // Override expression visitation to replace Var with current version
  ExprPtr VisitExpr_(const VarPtr& op) override {
    std::string base_name = GetBaseName(op->name_);
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
    std::string base_name = GetBaseName(op->name_);
    auto it = current_version_.find(base_name);
    if (it != current_version_.end()) {
      // Return the current version (which should be the iter_arg itself)
      return it->second;
    }

    // If no current version, create one
    if (new_init != op->initValue_) {
      return std::make_shared<IterArg>(op->name_, op->GetType(), new_init, op->span_);
    }
    return op;
  }

  // Override assignment statement to create versioned variables
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // First, visit the RHS expression (uses current versions)
    auto new_value = VisitExpr(op->value_);

    // Create a new versioned variable for LHS
    std::string base_name = GetBaseName(op->var_->name_);
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
    // Only consider variables that existed before the if statement -
    // variables created only inside a branch are branch-local and don't need phi nodes.
    std::vector<std::string> phi_vars;
    std::set<std::string> checked_vars;

    // Check variables from then branch
    for (const auto& [base_name, var] : versions_after_then) {
      checked_vars.insert(base_name);
      auto before_it = versions_before.find(base_name);

      // Skip variables not defined before the if (branch-local)
      if (before_it == versions_before.end()) continue;

      bool changed_in_then = (before_it->second != var);
      auto else_it = versions_after_else.find(base_name);
      bool changed_in_else = (else_it != versions_after_else.end() && before_it->second != else_it->second);

      if (changed_in_then || changed_in_else) {
        phi_vars.push_back(base_name);
      }
    }

    // Check variables from else branch that weren't in then
    for (const auto& [base_name, var] : versions_after_else) {
      if (checked_vars.count(base_name)) continue;

      auto before_it = versions_before.find(base_name);
      // Skip variables not defined before the if (branch-local)
      if (before_it == versions_before.end()) continue;

      if (before_it->second != var) {
        phi_vars.push_back(base_name);
      }
    }

    // If no variables diverged, just return the updated if statement
    if (phi_vars.empty() && op->return_vars_.empty()) {
      current_version_ = versions_after_then;  // Use then branch versions as default
      return std::make_shared<IfStmt>(new_condition, new_then, new_else, std::vector<VarPtr>{}, op->span_);
    }

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
      std::string base_name = GetBaseName(existing_rv->name_);
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

  // Override ForStmt to handle loop-carried variables
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
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
          std::make_shared<IterArg>(iter_arg->name_, iter_arg->GetType(), new_init, iter_arg->span_);
      new_iter_args.push_back(new_ia);
    }

    // PRE-ANALYSIS: Find which outer variables are assigned in the loop body.
    // This allows us to create iter_args BEFORE visiting the body.
    AssignmentCollector collector;
    collector.Collect(op->body_);

    // Identify loop-carried variables: assigned in body AND existed before the loop
    std::string loop_var_base = GetBaseName(op->loop_var_->name_);
    std::vector<std::string> loop_carried_vars;
    for (const auto& assigned_name : collector.assigned_vars) {
      // Skip loop variable
      if (assigned_name == loop_var_base) continue;

      // Skip existing iter_args
      bool is_existing_iter_arg = false;
      for (const auto& ia : op->iter_args_) {
        if (GetBaseName(ia->name_) == assigned_name) {
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

    // Register ALL iter_args (existing + new loop-carried) in loop scope
    for (const auto& iter_arg : new_iter_args) {
      std::string base_name = GetBaseName(iter_arg->name_);
      // For iter_args like "acc_iter_1", extract "acc" as base
      size_t iter_pos = base_name.find("_iter");
      if (iter_pos != std::string::npos) {
        base_name = base_name.substr(0, iter_pos);
      }
      current_version_[base_name] = iter_arg;
    }

    // Visit loop body - now it will correctly reference iter_args
    auto new_body = VisitStmt(op->body_);
    auto versions_after_body = current_version_;

    // Exit loop scope
    ExitScope();

    // Update outer scope to use return_vars for loop-carried variables
    for (size_t i = 0; i < loop_carried_vars.size(); ++i) {
      current_version_[loop_carried_vars[i]] = return_vars[i];
    }

    // Collect yield values: first existing iter_args, then new loop-carried
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
                                     return_vars, op->span_);
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

  /**
   * @brief Get the base name from a potentially versioned name
   */
  std::string GetBaseName(const std::string& name) {
    // If name already has version suffix like "x_0", extract base
    size_t last_underscore = name.rfind('_');
    if (last_underscore != std::string::npos && last_underscore > 0) {
      // Check if everything after underscore is numeric
      bool all_digits = true;
      for (size_t i = last_underscore + 1; i < name.size(); ++i) {
        if (!std::isdigit(name[i])) {
          all_digits = false;
          break;
        }
      }
      if (all_digits && last_underscore + 1 < name.size()) {
        return name.substr(0, last_underscore);
      }
    }
    return name;
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
   * @brief Create a versioned variable from an original variable
   */
  VarPtr CreateVersionedVar(const VarPtr& original, const std::string& base_name, int version) {
    std::string versioned_name = base_name + "_" + std::to_string(version);
    return std::make_shared<Var>(versioned_name, original->GetType(), original->span_);
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

    auto yield = std::make_shared<YieldStmt>(values, span);

    if (auto seq = As<SeqStmts>(stmt)) {
      // Check if last statement is already a yield
      if (!seq->stmts_.empty() && As<YieldStmt>(seq->stmts_.back())) {
        // Replace last yield
        std::vector<StmtPtr> new_stmts(seq->stmts_.begin(), seq->stmts_.end() - 1);
        new_stmts.push_back(yield);
        return std::make_shared<SeqStmts>(new_stmts, seq->span_);
      }
      // Append yield
      std::vector<StmtPtr> new_stmts = seq->stmts_;
      new_stmts.push_back(yield);
      return std::make_shared<SeqStmts>(new_stmts, seq->span_);
    }

    // Wrap single statement and yield in SeqStmts
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{stmt, yield}, span);
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
        return std::make_shared<SeqStmts>(new_stmts, seq->span_);
      }
      // Append yield
      std::vector<StmtPtr> new_stmts = seq->stmts_;
      new_stmts.push_back(new_yield);
      return std::make_shared<SeqStmts>(new_stmts, seq->span_);
    }

    if (As<YieldStmt>(stmt)) {
      return new_yield;
    }

    // Wrap single statement and yield in SeqStmts
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{stmt, new_yield}, span);
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
Pass ConvertToSSA() { return CreateFunctionPass(TransformConvertToSSA, "ConvertToSSA"); }
}  // namespace pass

}  // namespace ir
}  // namespace pypto

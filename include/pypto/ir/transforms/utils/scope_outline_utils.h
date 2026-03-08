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
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace outline_utils {

// ============================================================================
// Helper visitors/mutators shared by scope-outlining passes
// ============================================================================

/** @brief Mutator to substitute variables in an IR subtree. */
class VarSubstitutor : public IRMutator {
 public:
  explicit VarSubstitutor(const std::unordered_map<std::string, VarPtr>& var_map) : var_map_(var_map) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_map_.find(op->name_);
    if (it != var_map_.end()) {
      return it->second;
    }
    return op;
  }

 private:
  std::unordered_map<std::string, VarPtr> var_map_;
};

/** @brief Visitor to collect all variable references in an IR subtree. */
class VarRefCollector : public IRVisitor {
 public:
  std::unordered_set<std::string> var_refs;

 protected:
  void VisitExpr_(const VarPtr& op) override { var_refs.insert(op->name_); }

  void VisitExpr_(const IterArgPtr& op) override {
    var_refs.insert(op->name_);
    // Use dynamic_pointer_cast (not As<Var>) to match both Var and IterArg,
    // avoiding recursive initValue_ traversal that would pull in outer-scope vars.
    if (op->initValue_) {
      auto as_var = std::dynamic_pointer_cast<const Var>(op->initValue_);
      if (as_var) {
        var_refs.insert(as_var->name_);
      } else {
        VisitExpr(op->initValue_);
      }
    }
  }
};

/** @brief Visitor to collect all variable definitions in an IR subtree. */
class VarDefCollector : public IRVisitor {
 public:
  std::unordered_set<std::string> var_defs;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    var_defs.insert(op->var_->name_);
    // Don't visit the RHS - we only care about definitions
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    var_defs.insert(op->loop_var_->name_);
    for (const auto& iter_arg : op->iter_args_) {
      var_defs.insert(iter_arg->name_);
    }
    for (const auto& return_var : op->return_vars_) {
      var_defs.insert(return_var->name_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& iter_arg : op->iter_args_) {
      var_defs.insert(iter_arg->name_);
    }
    for (const auto& return_var : op->return_vars_) {
      var_defs.insert(return_var->name_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& return_var : op->return_vars_) {
      var_defs.insert(return_var->name_);
    }
    IRVisitor::VisitStmt_(op);
  }
};

/**
 * @brief Visitor to collect target tensors of tile.store calls.
 *
 * These tensors are modified via side-effect inside scopes but are not
 * captured by VarDefCollector since they are defined externally.  The third
 * argument of store is the output tensor.
 */
class StoreTargetCollector : public IRVisitor {
 public:
  std::unordered_set<std::string> store_targets;

 protected:
  void VisitExpr_(const CallPtr& op) override {
    auto opnode = std::dynamic_pointer_cast<const Op>(op->op_);
    if (opnode && opnode->name_ == "tile.store" && op->args_.size() >= 3) {
      if (auto var = As<Var>(op->args_[2])) {
        store_targets.insert(var->name_);
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
  explicit StoreEvalToAssignMutator(const std::unordered_map<std::string, VarPtr>& target_vars)
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
    auto it = target_vars_.find(var->name_);
    if (it == target_vars_.end()) return op;
    return std::make_shared<AssignStmt>(it->second, call, op->span_);
  }

 private:
  std::unordered_map<std::string, VarPtr> target_vars_;
};

/** @brief Visitor to build a symbol table mapping variable names to their types and Var objects. */
class VarCollector : public IRVisitor {
 public:
  std::unordered_map<std::string, TypePtr> var_types;
  std::unordered_map<std::string, VarPtr> var_objects;

 protected:
  void VisitExpr_(const VarPtr& op) override {
    var_types.try_emplace(op->name_, op->GetType());
    var_objects.try_emplace(op->name_, op);
    IRVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    var_types[op->var_->name_] = op->var_->GetType();
    var_objects[op->var_->name_] = op->var_;
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    var_types[op->loop_var_->name_] = op->loop_var_->GetType();
    var_objects[op->loop_var_->name_] = op->loop_var_;
    for (const auto& iter_arg : op->iter_args_) {
      var_types[iter_arg->name_] = iter_arg->GetType();
      var_objects[iter_arg->name_] = iter_arg;
    }
    for (const auto& return_var : op->return_vars_) {
      var_types[return_var->name_] = return_var->GetType();
      var_objects[return_var->name_] = return_var;
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& iter_arg : op->iter_args_) {
      var_types[iter_arg->name_] = iter_arg->GetType();
      var_objects[iter_arg->name_] = iter_arg;
    }
    for (const auto& return_var : op->return_vars_) {
      var_types[return_var->name_] = return_var->GetType();
      var_objects[return_var->name_] = return_var;
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& return_var : op->return_vars_) {
      var_types[return_var->name_] = return_var->GetType();
      var_objects[return_var->name_] = return_var;
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
  ScopeOutliner(std::string func_name, const std::unordered_map<std::string, TypePtr>& var_types,
                const std::unordered_map<std::string, VarPtr>& var_objects, ScopeKind target_scope_kind,
                FunctionType outlined_func_type, std::string name_suffix)
      : func_name_(std::move(func_name)),
        var_types_(var_types),
        var_objects_(var_objects),
        target_scope_kind_(target_scope_kind),
        outlined_func_type_(outlined_func_type),
        name_suffix_(std::move(name_suffix)) {}

  [[nodiscard]] const std::vector<FunctionPtr>& GetOutlinedFunctions() const { return outlined_functions_; }

 protected:
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
        // Outline this scope with context about what's used after
        auto outlined_stmt = OutlineScope(scope, used_after);
        // Flatten nested SeqStmts into the parent
        if (auto nested_seq = std::dynamic_pointer_cast<const SeqStmts>(outlined_stmt)) {
          for (const auto& s : nested_seq->stmts_) {
            new_stmts.push_back(s);
          }
        } else {
          new_stmts.push_back(outlined_stmt);
        }
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
    return std::make_shared<SeqStmts>(new_stmts, op->span_);
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
   * @param used_after Variables used in subsequent statements (determines outputs)
   */
  StmtPtr OutlineScope(const ScopeStmtPtr& op, const std::unordered_set<std::string>& used_after) {
    // Generate unique function name
    std::ostringstream name_stream;
    name_stream << func_name_ << name_suffix_ << scope_counter_++;
    std::string outlined_func_name = name_stream.str();

    // Analyze the scope body for inputs and outputs (before recursing)
    VarRefCollector ref_collector;
    ref_collector.VisitStmt(op->body_);

    VarDefCollector def_collector;
    def_collector.VisitStmt(op->body_);

    // Inputs: variables referenced but not defined in the scope
    std::vector<std::string> sorted_inputs;
    for (const auto& var_name : ref_collector.var_refs) {
      if (!def_collector.var_defs.count(var_name)) {
        sorted_inputs.push_back(var_name);
      }
    }
    std::sort(sorted_inputs.begin(), sorted_inputs.end());

    // Outputs: variables defined in the scope AND used after it
    std::vector<std::string> sorted_outputs;
    for (const auto& var_name : def_collector.var_defs) {
      if (used_after.count(var_name)) {
        sorted_outputs.push_back(var_name);
      }
    }

    // Also treat store targets as outputs: external tensors modified via
    // tile.store.  These represent side-effect outputs that must be
    // returned regardless of whether they appear in used_after, because the
    // store mutates an externally-visible buffer (e.g. loop-carried state).
    StoreTargetCollector store_collector;
    store_collector.VisitStmt(op->body_);
    std::unordered_set<std::string> store_output_set;
    for (const auto& var_name : store_collector.store_targets) {
      if (!def_collector.var_defs.count(var_name)) {
        sorted_outputs.push_back(var_name);
        store_output_set.insert(var_name);
      }
    }

    std::sort(sorted_outputs.begin(), sorted_outputs.end());

    // Recursively transform the scope body (handles nested scopes)
    // Save/restore state so nested scopes get their own hierarchical names and counters
    std::string saved_func_name = func_name_;
    int saved_scope_counter = scope_counter_;
    auto saved_required_outputs = required_outputs_;
    func_name_ = outlined_func_name;
    scope_counter_ = 0;
    // Propagate output requirements so nested scopes know what's needed
    required_outputs_ = std::unordered_set<std::string>(sorted_outputs.begin(), sorted_outputs.end());
    auto recursed_body = VisitStmt(op->body_);
    func_name_ = saved_func_name;
    scope_counter_ = saved_scope_counter;
    required_outputs_ = saved_required_outputs;

    // Create fresh parameters for the outlined function
    std::vector<VarPtr> input_params;
    std::vector<ParamDirection> input_param_directions;
    std::unordered_map<std::string, VarPtr> var_substitution_map;
    for (const auto& var_name : sorted_inputs) {
      auto type_it = var_types_.find(var_name);
      CHECK(type_it != var_types_.end()) << "Variable " << var_name << " not found in symbol table";
      auto param_var = std::make_shared<Var>(var_name, type_it->second, op->span_);
      input_params.push_back(param_var);
      input_param_directions.push_back(ParamDirection::In);
      var_substitution_map[var_name] = param_var;
    }

    // Collect type info from scope body for output variables
    VarCollector scope_var_collector;
    scope_var_collector.VisitStmt(op->body_);

    // Create fresh output variables for the outlined function
    std::vector<VarPtr> outlined_output_vars;
    std::vector<TypePtr> return_types;
    for (const auto& var_name : sorted_outputs) {
      TypePtr var_type;
      if (store_output_set.count(var_name)) {
        // Store target: external variable, look up from outer symbol table
        auto type_it = var_types_.find(var_name);
        CHECK(type_it != var_types_.end()) << "Variable " << var_name << " not found in symbol table";
        var_type = type_it->second;
      } else {
        // Regular output: defined in scope body
        auto var_it = scope_var_collector.var_objects.find(var_name);
        CHECK(var_it != scope_var_collector.var_objects.end())
            << "Variable " << var_name << " not found in scope body";
        var_type = var_it->second->GetType();
      }
      // For store targets, create a fresh variable with "_store_ret" suffix
      // to avoid redefining the input parameter in SSA form.
      std::string out_var_name = store_output_set.count(var_name) ? var_name + "_store_ret" : var_name;
      auto outlined_var = std::make_shared<Var>(out_var_name, var_type, op->span_);
      outlined_output_vars.push_back(outlined_var);
      return_types.push_back(var_type);
      if (!store_output_set.count(var_name)) {
        var_substitution_map[var_name] = outlined_var;
      }
    }

    // Apply variable substitution to the (already recursively transformed) body
    VarSubstitutor substitutor(var_substitution_map);
    auto transformed_body = substitutor.VisitStmt(recursed_body);

    // Convert EvalStmt(tile.store) to AssignStmt for store targets
    // so the return value is captured with a fresh SSA name (e.g. oi_0_store_ret).
    if (!store_output_set.empty()) {
      // Map: original target name -> new _store_ret Var
      std::unordered_map<std::string, VarPtr> store_target_vars;
      for (size_t idx = 0; idx < sorted_outputs.size(); ++idx) {
        if (store_output_set.count(sorted_outputs[idx])) {
          store_target_vars[sorted_outputs[idx]] = outlined_output_vars[idx];
        }
      }
      StoreEvalToAssignMutator store_mutator(store_target_vars);
      transformed_body = store_mutator.VisitStmt(transformed_body);
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

    // Register the outlined function
    auto outlined_func =
        std::make_shared<Function>(outlined_func_name, input_params, input_param_directions, return_types,
                                   outlined_body, op->span_, outlined_func_type_);
    outlined_functions_.push_back(outlined_func);

    // Build the call site in the parent function
    auto global_var = std::make_shared<GlobalVar>(outlined_func_name);
    std::vector<ExprPtr> call_args;
    for (const auto& var_name : sorted_inputs) {
      auto var_it = var_objects_.find(var_name);
      CHECK(var_it != var_objects_.end()) << "Variable " << var_name << " not found in var_objects";
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

    // Resolve the original Var for an output variable. Scope-defined vars come from
    // scope_var_collector; store targets (external tensors) fall back to the outer symbol table.
    auto resolve_output_var = [&](const std::string& name) -> VarPtr {
      auto var_it = scope_var_collector.var_objects.find(name);
      if (var_it != scope_var_collector.var_objects.end() && !store_output_set.count(name)) {
        return var_it->second;
      }
      auto ext_it = var_objects_.find(name);
      CHECK(ext_it != var_objects_.end()) << "Variable " << name << " not found in var_objects";
      return ext_it->second;
    };

    // Create assignments for output variables in the parent function
    if (sorted_outputs.empty()) {
      return std::make_shared<EvalStmt>(call_expr, op->span_);
    } else if (sorted_outputs.size() == 1) {
      return std::make_shared<AssignStmt>(resolve_output_var(sorted_outputs[0]), call_expr, op->span_);
    } else {
      // Assign call result to a temporary variable, then unpack with TupleGetItem
      auto ret_var = std::make_shared<Var>("ret", call_return_type, op->span_);
      std::vector<StmtPtr> stmts;
      stmts.push_back(std::make_shared<AssignStmt>(ret_var, call_expr, op->span_));
      for (size_t i = 0; i < sorted_outputs.size(); ++i) {
        auto tuple_get = std::make_shared<TupleGetItemExpr>(ret_var, static_cast<int>(i), op->span_);
        stmts.push_back(
            std::make_shared<AssignStmt>(resolve_output_var(sorted_outputs[i]), tuple_get, op->span_));
      }
      return std::make_shared<SeqStmts>(stmts, op->span_);
    }
  }

  std::string func_name_;
  std::unordered_map<std::string, TypePtr> var_types_;
  std::unordered_map<std::string, VarPtr> var_objects_;
  std::unordered_set<std::string> required_outputs_;
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

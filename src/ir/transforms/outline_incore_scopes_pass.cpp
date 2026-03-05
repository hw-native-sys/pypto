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
#include <cstddef>
#include <memory>
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
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/**
 * @brief Mutator to substitute variables in an IR subtree
 */
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

/**
 * @brief Visitor to collect all variable references in an IR subtree
 */
class VarRefCollector : public IRVisitor {
 public:
  std::unordered_set<std::string> var_refs;

 protected:
  void VisitExpr_(const VarPtr& op) override { var_refs.insert(op->name_); }

  void VisitExpr_(const IterArgPtr& op) override { var_refs.insert(op->name_); }
};

/**
 * @brief Visitor to collect all variable definitions in an IR subtree
 */
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
 * @brief Visitor to collect target tensors of block.store / block.l0c_store calls.
 *
 * These tensors are modified via side-effect inside InCore scopes but are not
 * captured by VarDefCollector since they are defined externally.  The fourth
 * argument of store/l0c_store is the output tensor.
 */
class StoreTargetCollector : public IRVisitor {
 public:
  std::unordered_set<std::string> store_targets;

 protected:
  void VisitExpr_(const CallPtr& op) override {
    auto opnode = std::dynamic_pointer_cast<const Op>(op->op_);
    if (opnode && (opnode->name_ == "block.store" || opnode->name_ == "block.l0c_store")) {
      if (op->args_.size() >= 4) {
        if (auto var = As<Var>(op->args_[3])) {
          store_targets.insert(var->name_);
        }
      }
    }
    IRVisitor::VisitExpr_(op);
  }
};

/**
 * @brief Mutator that converts EvalStmt(Call(block.store/l0c_store, ...)) into
 *        AssignStmt(target_var, Call(block.store/l0c_store, ...)) for specified
 *        store targets.
 *
 * block.store returns the output tensor (same type as the 4th argument).  When
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
    if (!opnode || (opnode->name_ != "block.store" && opnode->name_ != "block.l0c_store")) {
      return op;
    }
    if (call->args_.size() < 4) return op;
    auto var = As<Var>(call->args_[3]);
    if (!var) return op;
    auto it = target_vars_.find(var->name_);
    if (it == target_vars_.end()) return op;
    return std::make_shared<AssignStmt>(it->second, call, op->span_);
  }

 private:
  std::unordered_map<std::string, VarPtr> target_vars_;
};

/**
 * @brief Visitor to build a symbol table mapping variable names to their types and Var objects
 */
class VarCollector : public IRVisitor {
 public:
  std::unordered_map<std::string, TypePtr> var_types;
  std::unordered_map<std::string, VarPtr> var_objects;

 protected:
  void VisitExpr_(const VarPtr& op) override {
    if (var_types.find(op->name_) == var_types.end()) {
      var_types[op->name_] = op->GetType();
      var_objects[op->name_] = op;
    }
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

/**
 * @brief Mutator to outline InCore scopes into separate functions
 *
 * Handles SeqStmts specially to determine which scope-defined variables
 * are actually used after each scope (output filtering), and recursively
 * transforms scope bodies to handle nested InCore scopes.
 */
class IncoreScopeOutliner : public IRMutator {
 public:
  explicit IncoreScopeOutliner(std::string func_name,
                               const std::unordered_map<std::string, TypePtr>& var_types,
                               const std::unordered_map<std::string, VarPtr>& var_objects)
      : func_name_(std::move(func_name)), var_types_(var_types), var_objects_(var_objects) {}

  [[nodiscard]] const std::vector<FunctionPtr>& GetOutlinedFunctions() const { return outlined_functions_; }

 protected:
  /**
   * @brief Process SeqStmts to analyze scope outputs using subsequent statements
   *
   * For each InCore scope, collects variables referenced in all subsequent statements
   * plus any variables required by a parent scope (propagated via required_outputs_).
   */
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    bool changed = false;

    for (size_t i = 0; i < op->stmts_.size(); ++i) {
      auto scope = std::dynamic_pointer_cast<const ScopeStmt>(op->stmts_[i]);
      if (scope && scope->scope_kind_ == ScopeKind::InCore) {
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
   * @brief Handle standalone ScopeStmts (not inside SeqStmts)
   *
   * When a scope appears outside a SeqStmts, all defined variables are outputs.
   */
  StmtPtr VisitStmt_(const ScopeStmtPtr& op) override {
    if (op->scope_kind_ != ScopeKind::InCore) {
      return IRMutator::VisitStmt_(op);
    }

    // Without context, treat all defined variables + store targets as outputs
    VarDefCollector def_collector;
    def_collector.VisitStmt(op->body_);
    std::unordered_set<std::string> all_defs(def_collector.var_defs.begin(), def_collector.var_defs.end());

    StoreTargetCollector store_collector;
    store_collector.VisitStmt(op->body_);
    for (const auto& t : store_collector.store_targets) {
      all_defs.insert(t);
    }

    return OutlineScope(op, all_defs);
  }

 private:
  /**
   * @brief Outline a single InCore scope into a separate function
   *
   * @param op The scope statement to outline
   * @param used_after Variables used in subsequent statements (determines outputs)
   */
  StmtPtr OutlineScope(const ScopeStmtPtr& op, const std::unordered_set<std::string>& used_after) {
    // Generate unique function name
    std::ostringstream name_stream;
    name_stream << func_name_ << "_incore_" << scope_counter_++;
    std::string outlined_func_name = name_stream.str();

    // Analyze the scope body for inputs and outputs (before recursing)
    VarRefCollector ref_collector;
    ref_collector.VisitStmt(op->body_);

    VarDefCollector def_collector;
    def_collector.VisitStmt(op->body_);

    // Inputs: variables referenced but not defined in the scope
    std::vector<std::string> sorted_inputs;
    for (const auto& var_name : ref_collector.var_refs) {
      if (def_collector.var_defs.find(var_name) == def_collector.var_defs.end()) {
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
    // block.store/l0c_store.  These represent side-effect outputs that must be
    // returned regardless of whether they appear in used_after, because the
    // store mutates an externally-visible buffer (e.g. loop-carried state).
    StoreTargetCollector store_collector;
    store_collector.VisitStmt(op->body_);
    std::unordered_set<std::string> store_output_set;
    for (const auto& var_name : store_collector.store_targets) {
      if (def_collector.var_defs.find(var_name) == def_collector.var_defs.end()) {
        sorted_outputs.push_back(var_name);
        store_output_set.insert(var_name);
      }
    }

    std::sort(sorted_outputs.begin(), sorted_outputs.end());

    // Recursively transform the scope body (handles nested InCore scopes)
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

    // Convert EvalStmt(block.store/l0c_store) to AssignStmt for store targets
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
                                   outlined_body, op->span_, FunctionType::InCore);
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

    // Create assignments for output variables in the parent function
    // Use the original Var objects from the scope body so they match later references.
    // For store outputs (external tensors), fall back to the outer symbol table.
    if (sorted_outputs.empty()) {
      return std::make_shared<EvalStmt>(call_expr, op->span_);
    } else if (sorted_outputs.size() == 1) {
      auto var_it = scope_var_collector.var_objects.find(sorted_outputs[0]);
      VarPtr out_var;
      if (var_it != scope_var_collector.var_objects.end() && !store_output_set.count(sorted_outputs[0])) {
        out_var = var_it->second;
      } else {
        auto ext_it = var_objects_.find(sorted_outputs[0]);
        CHECK(ext_it != var_objects_.end())
            << "Variable " << sorted_outputs[0] << " not found in var_objects";
        out_var = ext_it->second;
      }
      return std::make_shared<AssignStmt>(out_var, call_expr, op->span_);
    } else {
      // Assign call result to a temporary variable, then unpack with TupleGetItem
      auto ret_var = std::make_shared<Var>("ret", call_return_type, op->span_);
      std::vector<StmtPtr> stmts;
      stmts.push_back(std::make_shared<AssignStmt>(ret_var, call_expr, op->span_));
      for (size_t i = 0; i < sorted_outputs.size(); ++i) {
        VarPtr out_var;
        auto var_it = scope_var_collector.var_objects.find(sorted_outputs[i]);
        if (var_it != scope_var_collector.var_objects.end() && !store_output_set.count(sorted_outputs[i])) {
          out_var = var_it->second;
        } else {
          auto ext_it = var_objects_.find(sorted_outputs[i]);
          CHECK(ext_it != var_objects_.end())
              << "Variable " << sorted_outputs[i] << " not found in var_objects";
          out_var = ext_it->second;
        }
        auto tuple_get = std::make_shared<TupleGetItemExpr>(ret_var, static_cast<int>(i), op->span_);
        stmts.push_back(std::make_shared<AssignStmt>(out_var, tuple_get, op->span_));
      }
      return std::make_shared<SeqStmts>(stmts, op->span_);
    }
  }

  std::string func_name_;
  std::unordered_map<std::string, TypePtr> var_types_;
  std::unordered_map<std::string, VarPtr> var_objects_;
  std::unordered_set<std::string> required_outputs_;
  int scope_counter_ = 0;
  std::vector<FunctionPtr> outlined_functions_;
};

}  // namespace

namespace pass {

/**
 * @brief Pass to outline InCore scopes into separate functions
 *
 * This pass transforms ScopeStmt(InCore) nodes into separate Function(InCore) definitions
 * and replaces the scope with a Call to the outlined function.
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Only processes Opaque functions (InCore functions are left unchanged)
 *
 * Transformation:
 * 1. For each ScopeStmt(InCore) in an Opaque function:
 *    - Analyze body to determine external variable references (inputs)
 *    - Analyze subsequent statements to determine which definitions are outputs
 *    - Extract body into new Function(InCore) with appropriate params/returns
 *    - Replace scope with Call to the outlined function + output assignments
 *    - EvalStmt(store) calls on output tensors are converted to AssignStmt
 * 2. Recursively handles nested InCore scopes
 * 3. Add outlined functions to the program
 * 4. Promote the parent function from Opaque to Orchestration
 */
Pass OutlineIncoreScopes() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> new_functions;
    std::vector<FunctionPtr> all_outlined_functions;

    // Process each function in the program
    for (const auto& [gvar, func] : program->functions_) {
      // Only process Opaque functions (InCore functions are already outlined)
      if (func->func_type_ != FunctionType::Opaque) {
        new_functions.push_back(func);
        continue;
      }

      // Build symbol table for this function
      VarCollector type_collector;
      for (const auto& var : func->params_) {
        type_collector.var_types[var->name_] = var->GetType();
        type_collector.var_objects[var->name_] = var;
      }
      type_collector.VisitStmt(func->body_);

      // Outline InCore scopes in this function
      IncoreScopeOutliner outliner(func->name_, type_collector.var_types, type_collector.var_objects);
      auto new_body = outliner.VisitStmt(func->body_);

      // Create new function with transformed body.
      // If any InCore scopes were outlined, promote Opaque -> Orchestration.
      const auto& outlined = outliner.GetOutlinedFunctions();
      FunctionType new_func_type = outlined.empty() ? func->func_type_ : FunctionType::Orchestration;
      auto new_func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                                 func->return_types_, new_body, func->span_, new_func_type);
      new_functions.push_back(new_func);

      // Collect outlined functions (prepend before parent so inner functions come first)
      all_outlined_functions.insert(all_outlined_functions.end(), outlined.begin(), outlined.end());
    }

    // Add all outlined functions before the originals
    all_outlined_functions.insert(all_outlined_functions.end(), new_functions.begin(), new_functions.end());

    // Create new program with all functions
    return std::make_shared<Program>(all_outlined_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "OutlineIncoreScopes", kOutlineIncoreScopesProperties);
}

}  // namespace pass

// ============================================================================
// SplitIncoreOrch property verifier
// ============================================================================

namespace {

/**
 * @brief Checks no InCore ScopeStmts remain in Opaque or Orchestration functions.
 */
class SplitIncoreOrchVerifier : public IRVisitor {
 public:
  explicit SplitIncoreOrchVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const ScopeStmtPtr& op) override {
    if (!op) return;
    if (op->scope_kind_ == ScopeKind::InCore) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "SplitIncoreOrch", 0,
                                "InCore ScopeStmt found in non-InCore function (should have been outlined)",
                                op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class SplitIncoreOrchPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "SplitIncoreOrch"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // Check Opaque and Orchestration functions — InCore functions are expected to have InCore content
      if (func->func_type_ == FunctionType::InCore) continue;
      SplitIncoreOrchVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateSplitIncoreOrchPropertyVerifier() {
  return std::make_shared<SplitIncoreOrchPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

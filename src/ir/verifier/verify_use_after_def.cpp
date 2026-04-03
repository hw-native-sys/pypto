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

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verification_error.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace use_after_def {
std::string ErrorTypeToString(ErrorType type) {
  switch (type) {
    case ErrorType::USE_BEFORE_DEF:
      return "USE_BEFORE_DEF";
    default:
      return "UNKNOWN";
  }
}
}  // namespace use_after_def

namespace {

/**
 * @brief Collects all Var pointers referenced in expression trees.
 *
 * Used to find type-dynamic variables embedded in parameter type annotations
 * (shape, valid_shape, stride, etc.) that are not defined by any statement.
 */
class VarCollector : public IRVisitor {
 public:
  std::vector<const Var*> vars_;

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) vars_.push_back(op.get());
    // Don't recurse into the var's type — prevents infinite traversal.
  }
};

/**
 * @brief Visit all expression fields in a type using the given visitor.
 *
 * Covers: TensorType shape_, tensor_view_.{valid_shape, stride};
 *         TileType shape_, tile_view_.{valid_shape, stride, start_offset};
 *         TupleType elements (recursively).
 */
static void VisitTypeExprFields(IRVisitor& visitor, const TypePtr& type) {
  if (!type) return;

  auto visit_exprs = [&visitor](const std::vector<ExprPtr>& exprs) {
    for (const auto& e : exprs) {
      if (e) visitor.VisitExpr(e);
    }
  };

  if (auto tensor_type = As<TensorType>(type)) {
    visit_exprs(tensor_type->shape_);
    if (tensor_type->tensor_view_.has_value()) {
      const auto& tv = tensor_type->tensor_view_.value();
      visit_exprs(tv.valid_shape);
      visit_exprs(tv.stride);
    }
  } else if (auto tile_type = As<TileType>(type)) {
    visit_exprs(tile_type->shape_);
    if (tile_type->tile_view_.has_value()) {
      const auto& tv = tile_type->tile_view_.value();
      visit_exprs(tv.valid_shape);
      visit_exprs(tv.stride);
      if (tv.start_offset) visitor.VisitExpr(tv.start_offset);
    }
  } else if (auto tuple_type = As<TupleType>(type)) {
    for (const auto& elem : tuple_type->types_) {
      VisitTypeExprFields(visitor, elem);
    }
  }
}

/**
 * @brief Collect all Var pointers from a type's expression fields.
 *
 * Recursively walks expression trees in shape, valid_shape, stride, and
 * start_offset to find all referenced Var nodes and registers them via callback.
 */
static void CollectTypeVars(const TypePtr& type, const std::function<void(const Var*)>& register_var) {
  VarCollector collector;
  VisitTypeExprFields(collector, type);
  for (const auto* var : collector.vars_) {
    register_var(var);
  }
}

/**
 * @brief Visitor that checks every Var use is preceded by a definition.
 *
 * Scoping rules:
 * - Function params: in scope for entire function body
 * - AssignStmt::var_: defined after the RHS is evaluated
 * - ForStmt::loop_var_ and iter_args_: in scope only inside the loop body
 * - ForStmt::return_vars_: defined in the enclosing scope after the loop
 * - WhileStmt::iter_args_: in scope inside the loop body (including condition)
 * - WhileStmt::return_vars_: defined in the enclosing scope after the loop
 * - IfStmt::return_vars_: defined in the enclosing scope after the if
 * - IfStmt without return_vars_ ("leak" mode): definitions inside then/else branches are
 *   merged (unioned) back into the outer scope; otherwise, branch-local definitions
 *   do NOT propagate to the outer scope
 */
class UseAfterDefChecker : public IRVisitor {
 public:
  UseAfterDefChecker(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

  void AddDefinition(const Var* var) {
    if (var) in_scope_.insert(var);
  }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (!op) return;
    if (!in_scope_.count(op.get())) {
      std::ostringstream msg;
      msg << "Variable '" << op->name_hint_ << "' used before definition"
          << " in function '" << func_name_ << "'";
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "UseAfterDefCheck",
                                static_cast<int>(use_after_def::ErrorType::USE_BEFORE_DEF), msg.str(),
                                op->span_);
    }
    // Visit variable references in type expressions (valid_shape, stride, etc.)
    // to detect undefined vars in type metadata.  Guard against recursion:
    // vars found inside type expressions are typically scalars whose types
    // don't contain further view expressions, but the flag ensures safety.
    if (!visiting_type_) {
      visiting_type_ = true;
      VisitTypeExprFields(*this, op->GetType());
      visiting_type_ = false;
    }
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    // Evaluate RHS first (use site), then define LHS.
    if (op->value_) VisitExpr(op->value_);
    if (op->var_) in_scope_.insert(op->var_.get());
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;

    // Bounds and chunk_size are evaluated in the outer scope.
    if (op->start_) VisitExpr(op->start_);
    if (op->stop_) VisitExpr(op->stop_);
    if (op->step_) VisitExpr(op->step_);
    if (op->chunk_size_.has_value() && *op->chunk_size_) {
      VisitExpr(*op->chunk_size_);
    }

    // IterArg initial values are evaluated in the outer scope.
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) {
        VisitExpr(iter_arg->initValue_);
      }
    }

    auto saved_scope = in_scope_;

    // loop_var and iter_args are in scope only inside the body.
    if (op->loop_var_) in_scope_.insert(op->loop_var_.get());
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg) in_scope_.insert(iter_arg.get());
    }

    if (op->body_) VisitStmt(op->body_);

    // Remove loop-scoped variables.
    if (op->loop_var_) in_scope_.erase(op->loop_var_.get());
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg) in_scope_.erase(iter_arg.get());
    }

    if (!op->return_vars_.empty()) {
      // SSA mode: only return_vars are visible after the loop.
      in_scope_ = saved_scope;
      for (const auto& rv : op->return_vars_) {
        if (rv) in_scope_.insert(rv.get());
      }
    } else {
      // Leak mode: body-local definitions (excluding loop_var and iter_args)
      // are visible after the loop. loop_var and iter_args were already removed
      // above. in_scope_ now contains saved_scope + body-leaked vars.
      // UseAfterDef does not check leak validity — that is SSAVerify's job.
    }
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op) return;

    auto saved_scope = in_scope_;

    // IterArg initial values are evaluated in the outer scope.
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) {
        VisitExpr(iter_arg->initValue_);
      }
    }

    // iter_args are in scope for condition and body.
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg) in_scope_.insert(iter_arg.get());
    }

    if (op->condition_) VisitExpr(op->condition_);
    if (op->body_) VisitStmt(op->body_);

    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg) in_scope_.erase(iter_arg.get());
    }

    if (!op->return_vars_.empty()) {
      // SSA mode: only return_vars are visible after the loop.
      in_scope_ = saved_scope;
      for (const auto& rv : op->return_vars_) {
        if (rv) in_scope_.insert(rv.get());
      }
    }
    // Leak mode: body-local definitions remain visible after the loop.
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (!op) return;

    if (op->condition_) VisitExpr(op->condition_);

    auto saved_scope = in_scope_;

    if (op->then_body_) VisitStmt(op->then_body_);
    auto then_scope = in_scope_;

    in_scope_ = saved_scope;
    if (op->else_body_.has_value() && *op->else_body_) {
      VisitStmt(*op->else_body_);
    }
    auto else_scope = in_scope_;

    if (!op->return_vars_.empty()) {
      // SSA phi-node mode: only return_vars are visible after the if.
      in_scope_ = saved_scope;
      for (const auto& rv : op->return_vars_) {
        if (rv) in_scope_.insert(rv.get());
      }
    } else {
      // Leak mode: branch-local definitions are visible after the if.
      // UseAfterDef does not check leak validity — that is SSAVerify's job.
      in_scope_ = saved_scope;
      in_scope_.insert(then_scope.begin(), then_scope.end());
      in_scope_.insert(else_scope.begin(), else_scope.end());
    }
  }

 private:
  std::unordered_set<const Var*> in_scope_;
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  bool visiting_type_ = false;
};

class UseAfterDefPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "UseAfterDefCheck"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;

    for (const auto& [global_var, func] : program->functions_) {
      if (!func) continue;

      UseAfterDefChecker checker(diagnostics, func->name_);

      // Function parameters are definitions visible throughout the body.
      for (const auto& param : func->params_) {
        if (param) checker.AddDefinition(param.get());
      }

      // Type-dynamic vars in this function's parameter and return types are
      // implicitly in scope.  E.g., Tensor[[N, M], FP32] where N, M are
      // dynamic shape vars that exist only in the function signature.
      auto add_def = [&checker](const Var* v) { checker.AddDefinition(v); };
      for (const auto& param : func->params_) {
        if (param) CollectTypeVars(param->GetType(), add_def);
      }
      for (const auto& ret_type : func->return_types_) {
        CollectTypeVars(ret_type, add_def);
      }

      if (func->body_) checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateUseAfterDefPropertyVerifier() {
  return std::make_shared<UseAfterDefPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

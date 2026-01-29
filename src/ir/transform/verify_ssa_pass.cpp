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

#include "pypto/ir/transform/verify_ssa_pass.h"

#include <memory>
#include <sstream>
#include <string>

#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {

// Error type to string conversion
static const char* ErrorTypeToString(SSAErrorType type) {
  switch (type) {
    case SSAErrorType::MULTIPLE_ASSIGNMENT:
      return "MULTIPLE_ASSIGNMENT";
    case SSAErrorType::NAME_SHADOWING:
      return "NAME_SHADOWING";
    case SSAErrorType::MISSING_YIELD:
      return "MISSING_YIELD";
    case SSAErrorType::CONTROL_FLOW_TYPE_MISMATCH:
      return "CONTROL_FLOW_TYPE_MISMATCH";
    default:
      return "UNKNOWN";
  }
}

FunctionPtr VerifySSAPass::Run(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "VerifySSAPass cannot run on null function";

  // Clear previous errors
  errors_.clear();

  // Create verifier and run verification
  SSAVerifier verifier(errors_);

  // Enter top-level scope and declare function parameters
  verifier.EnterScope();
  for (const auto& param : func->params_) {
    verifier.DeclareVariable(param);
  }

  // Visit function body
  if (func->body_) {
    verifier.VisitStmt(func->body_);
  }

  // Exit top-level scope
  verifier.ExitScope();

  // Return the same function (verification doesn't modify IR)
  return func;
}

std::string VerifySSAPass::GetReport() const {
  std::ostringstream oss;
  oss << "SSA Verification Report\n";
  oss << "=======================\n";
  oss << "Total errors found: " << errors_.size() << "\n\n";

  if (errors_.empty()) {
    oss << "Status: PASSED\n";
  } else {
    for (size_t i = 0; i < errors_.size(); ++i) {
      const auto& error = errors_[i];
      oss << "[Error " << (i + 1) << "] " << ErrorTypeToString(error.type) << "\n";
      oss << "  " << error.message << "\n";
      const auto& span = error.span;
      oss << "  Location: " << span.filename_ << ":" << span.begin_line_ << ":" << span.begin_column_ << "\n";
      oss << "\n";
    }
    oss << "Status: FAILED (" << errors_.size() << " errors)\n";
  }

  return oss.str();
}

// SSAVerifier implementation

void SSAVerifier::CheckVariableAssignment(const VarPtr& var) {
  if (!var) return;

  const std::string& var_name = var->name_;
  var_assignment_count_[var_name]++;

  if (var_assignment_count_[var_name] > 1) {
    std::ostringstream msg;
    msg << "Variable '" << var_name << "' is assigned more than once (" << var_assignment_count_[var_name]
        << " times), violating SSA form";
    RecordError(SSAErrorType::MULTIPLE_ASSIGNMENT, msg.str(), var->span_);
  }
}

void SSAVerifier::CheckNameShadowing(const VarPtr& var) {
  if (!var) return;

  const std::string& var_name = var->name_;

  // Check all scopes except the current one (outermost to innermost)
  for (size_t i = 0; i + 1 < scope_stack_.size(); ++i) {
    if (scope_stack_[i].count(var_name) > 0) {
      std::ostringstream msg;
      msg << "Variable '" << var_name << "' shadows outer scope variable with the same name";
      RecordError(SSAErrorType::NAME_SHADOWING, msg.str(), var->span_);
      return;  // Only report once
    }
  }
}

void SSAVerifier::EnterScope() {
  scope_stack_.emplace_back();  // Push new empty scope
}

void SSAVerifier::ExitScope() {
  if (!scope_stack_.empty()) {
    scope_stack_.pop_back();
  }
}

void SSAVerifier::DeclareVariable(const VarPtr& var) {
  if (!var || scope_stack_.empty()) return;

  // Add variable to current scope
  scope_stack_.back().insert(var->name_);
}

void SSAVerifier::RecordError(SSAErrorType type, const std::string& message, const Span& span) {
  errors_.push_back(SSAError{type, message, span});
}

void SSAVerifier::CheckTypeEquality(const TypePtr& type1, const TypePtr& type2, const std::string& context,
                                    const std::string& desc1, const std::string& desc2, const Span& span) {
  if (!type1 || !type2) return;

  // Use unified control flow type mismatch error
  SSAErrorType error_type = SSAErrorType::CONTROL_FLOW_TYPE_MISMATCH;

  // Check ObjectKind first
  if (type1->GetKind() != type2->GetKind()) {
    std::ostringstream msg;
    msg << "Type kind mismatch in " << context << ": " << desc1 << " type '" << type1->TypeName()
        << "' != " << desc2 << " type '" << type2->TypeName() << "'";
    RecordError(error_type, msg.str(), span);
    return;
  }

  // For ScalarType, check dtype
  if (type1->GetKind() == ObjectKind::ScalarType) {
    auto scalar1 = std::dynamic_pointer_cast<const ScalarType>(type1);
    auto scalar2 = std::dynamic_pointer_cast<const ScalarType>(type2);
    if (scalar1 && scalar2 && scalar1->dtype_ != scalar2->dtype_) {
      std::ostringstream msg;
      msg << "Dtype mismatch in " << context << ": " << desc1 << " dtype != " << desc2 << " dtype";
      RecordError(error_type, msg.str(), span);
    }
    return;
  }

  // For TensorType and TileType, check dtype and shape
  if (type1->GetKind() == ObjectKind::TensorType || type1->GetKind() == ObjectKind::TileType) {
    auto shaped1 = std::dynamic_pointer_cast<const ShapedType>(type1);
    auto shaped2 = std::dynamic_pointer_cast<const ShapedType>(type2);

    if (!shaped1 || !shaped2) return;

    // Check dtype
    if (shaped1->dtype_ != shaped2->dtype_) {
      std::ostringstream msg;
      msg << "Dtype mismatch in " << context << ": " << desc1 << " dtype != " << desc2 << " dtype";
      RecordError(error_type, msg.str(), span);
    }

    // Check shape dimensions count
    if (shaped1->shape_.size() != shaped2->shape_.size()) {
      std::ostringstream msg;
      msg << "Shape dimension count mismatch in " << context << ": " << desc1 << " has "
          << shaped1->shape_.size() << " dimensions, but " << desc2 << " has " << shaped2->shape_.size()
          << " dimensions";
      RecordError(error_type, msg.str(), span);
      return;
    }

    // Check each shape dimension
    for (size_t i = 0; i < shaped1->shape_.size(); ++i) {
      const auto& dim1 = shaped1->shape_[i];
      const auto& dim2 = shaped2->shape_[i];

      if (!dim1 || !dim2) continue;

      // Try to compare as constants
      if (!IsSameConstant(dim1, dim2)) {
        // Check if both are ConstInt but different values
        auto const_int1 = As<ConstInt>(dim1);
        auto const_int2 = As<ConstInt>(dim2);
        if (const_int1 && const_int2) {
          std::ostringstream msg;
          msg << "Shape dimension mismatch in " << context << ": " << desc1 << " dimension[" << i
              << "] = " << const_int1->value_ << ", but " << desc2 << " dimension[" << i
              << "] = " << const_int2->value_;
          RecordError(error_type, msg.str(), span);
        }
        // For symbolic dimensions, we skip detailed checking
        // A more sophisticated analysis would be needed for symbolic shape verification
      }
    }
  }
}

bool SSAVerifier::IsSameConstant(const ExprPtr& expr1, const ExprPtr& expr2) const {
  if (!expr1 || !expr2) return false;

  // Check if both are ConstInt
  auto const_int1 = As<ConstInt>(expr1);
  auto const_int2 = As<ConstInt>(expr2);
  if (const_int1 && const_int2) {
    return const_int1->value_ == const_int2->value_;
  }

  // For symbolic expressions, we consider them potentially equal if they have the same structure
  // A more sophisticated check would require symbolic comparison, but for SSA verification
  // we primarily care about constant dimensions
  return false;
}

StmtPtr SSAVerifier::GetLastStmt(const StmtPtr& stmt) {
  if (!stmt) return nullptr;

  // If it's a SeqStmts, recursively get the last statement
  if (auto seq = As<SeqStmts>(stmt)) {
    if (!seq->stmts_.empty()) {
      return GetLastStmt(seq->stmts_.back());
    }
  }

  return stmt;
}

void SSAVerifier::VerifyForStmt(const ForStmtPtr& for_stmt) {
  if (!for_stmt) return;

  // Check 1: If iter_args is not empty, body must end with YieldStmt
  if (!for_stmt->iter_args_.empty()) {
    StmtPtr last_stmt = GetLastStmt(for_stmt->body_);
    if (!last_stmt || !As<YieldStmt>(last_stmt)) {
      RecordError(SSAErrorType::MISSING_YIELD,
                  "ForStmt with iter_args must have YieldStmt as last statement in body", for_stmt->span_);
      return;  // Skip type checking if yield is missing
    }

    // Check 2: Type consistency between iter_args initValue, yield values, and return_vars
    auto yield_stmt = As<YieldStmt>(last_stmt);

    // Check that all three vectors have the same size
    size_t num_iter_args = for_stmt->iter_args_.size();
    size_t num_yield_values = yield_stmt->value_.size();
    size_t num_return_vars = for_stmt->return_vars_.size();

    if (num_iter_args != num_yield_values || num_iter_args != num_return_vars) {
      std::ostringstream msg;
      msg << "ForStmt size mismatch: iter_args=" << num_iter_args << ", yield values=" << num_yield_values
          << ", return_vars=" << num_return_vars;
      RecordError(SSAErrorType::CONTROL_FLOW_TYPE_MISMATCH, msg.str(), for_stmt->span_);
      return;
    }

    // Check type consistency for each index
    for (size_t i = 0; i < num_iter_args; ++i) {
      const auto& iter_arg = for_stmt->iter_args_[i];
      const auto& yield_value = yield_stmt->value_[i];
      const auto& return_var = for_stmt->return_vars_[i];

      if (!iter_arg || !iter_arg->initValue_ || !yield_value || !return_var) continue;

      auto init_type = iter_arg->initValue_->GetType();
      auto yield_type = yield_value->GetType();
      auto return_type = return_var->GetType();

      if (!init_type || !yield_type || !return_type) continue;

      // Check initValue type == yield type
      CheckTypeEquality(init_type, yield_type, "ForStmt", "iter_arg[" + std::to_string(i) + "] initValue",
                        "yield value[" + std::to_string(i) + "]", for_stmt->span_);

      // Check yield type == return_var type
      CheckTypeEquality(yield_type, return_type, "ForStmt", "yield value[" + std::to_string(i) + "]",
                        "return_var[" + std::to_string(i) + "]", for_stmt->span_);

      // Check initValue type == return_var type (for completeness)
      CheckTypeEquality(init_type, return_type, "ForStmt", "iter_arg[" + std::to_string(i) + "] initValue",
                        "return_var[" + std::to_string(i) + "]", for_stmt->span_);
    }
  }
}

void SSAVerifier::VerifyIfStmt(const IfStmtPtr& if_stmt) {
  if (!if_stmt) return;

  // Check only if return_vars is not empty
  if (if_stmt->return_vars_.empty()) {
    return;
  }

  // Check 1: else_body must exist
  if (!if_stmt->else_body_.has_value()) {
    RecordError(SSAErrorType::MISSING_YIELD, "IfStmt with return_vars must have else branch", if_stmt->span_);
    return;
  }

  // Check 2: Both then_body and else_body must end with YieldStmt
  StmtPtr then_last = GetLastStmt(if_stmt->then_body_);
  StmtPtr else_last = GetLastStmt(if_stmt->else_body_.value());

  auto then_yield = As<YieldStmt>(then_last);
  auto else_yield = As<YieldStmt>(else_last);

  if (!then_yield) {
    RecordError(SSAErrorType::MISSING_YIELD,
                "IfStmt then branch must end with YieldStmt when return_vars exist", if_stmt->span_);
  }

  if (!else_yield) {
    RecordError(SSAErrorType::MISSING_YIELD,
                "IfStmt else branch must end with YieldStmt when return_vars exist", if_stmt->span_);
  }

  if (!then_yield || !else_yield) {
    return;  // Skip type checking if yields are missing
  }

  // Check 3: Type consistency between then yield and else yield
  size_t num_then_values = then_yield->value_.size();
  size_t num_else_values = else_yield->value_.size();
  size_t num_return_vars = if_stmt->return_vars_.size();

  if (num_then_values != num_else_values || num_then_values != num_return_vars) {
    std::ostringstream msg;
    msg << "IfStmt size mismatch: then yield=" << num_then_values << ", else yield=" << num_else_values
        << ", return_vars=" << num_return_vars;
    RecordError(SSAErrorType::CONTROL_FLOW_TYPE_MISMATCH, msg.str(), if_stmt->span_);
    return;
  }

  // Check type consistency for each index
  for (size_t i = 0; i < num_then_values; ++i) {
    const auto& then_value = then_yield->value_[i];
    const auto& else_value = else_yield->value_[i];

    if (!then_value || !else_value) continue;

    auto then_type = then_value->GetType();
    auto else_type = else_value->GetType();

    if (!then_type || !else_type) continue;

    CheckTypeEquality(then_type, else_type, "IfStmt", "then yield value[" + std::to_string(i) + "]",
                      "else yield value[" + std::to_string(i) + "]", if_stmt->span_);
  }
}

void SSAVerifier::VisitStmt_(const AssignStmtPtr& op) {
  if (!op || !op->var_) return;

  // Check for name shadowing
  CheckNameShadowing(op->var_);

  // Declare the variable in current scope
  DeclareVariable(op->var_);

  // Check for multiple assignments
  CheckVariableAssignment(op->var_);

  // Continue with default traversal
  IRVisitor::VisitStmt_(op);
}

void SSAVerifier::VisitStmt_(const ForStmtPtr& op) {
  if (!op) return;

  // First, check and declare return_vars in the current (outer) scope
  for (const auto& return_var : op->return_vars_) {
    if (return_var) {
      CheckNameShadowing(return_var);
      DeclareVariable(return_var);
      CheckVariableAssignment(return_var);
    }
  }

  // Visit start, stop, step, and iter_args' initValue in current scope
  // These are all evaluated in the outer scope before the loop begins
  if (op->start_) VisitExpr(op->start_);
  if (op->stop_) VisitExpr(op->stop_);
  if (op->step_) VisitExpr(op->step_);

  for (const auto& iter_arg : op->iter_args_) {
    if (iter_arg && iter_arg->initValue_) {
      VisitExpr(iter_arg->initValue_);
    }
  }

  // Enter new scope for loop body
  EnterScope();

  // Declare loop_var in the loop scope
  if (op->loop_var_) {
    CheckNameShadowing(op->loop_var_);
    DeclareVariable(op->loop_var_);
  }

  // Declare iter_args in the loop scope
  for (const auto& iter_arg : op->iter_args_) {
    if (iter_arg) {
      CheckNameShadowing(iter_arg);
      DeclareVariable(iter_arg);
    }
  }

  // Visit loop body
  if (op->body_) {
    VisitStmt(op->body_);
  }

  // Exit loop scope
  ExitScope();

  // Verify ForStmt specific constraints
  VerifyForStmt(op);
}

void SSAVerifier::VisitStmt_(const IfStmtPtr& op) {
  if (!op) return;

  // Check and declare return_vars in current scope (before entering branches)
  for (const auto& return_var : op->return_vars_) {
    if (return_var) {
      CheckNameShadowing(return_var);
      DeclareVariable(return_var);
      CheckVariableAssignment(return_var);
    }
  }

  // Visit condition in current scope
  if (op->condition_) {
    VisitExpr(op->condition_);
  }

  // Visit then branch in its own scope
  EnterScope();
  if (op->then_body_) {
    VisitStmt(op->then_body_);
  }
  ExitScope();

  // Visit else branch in its own scope (if exists)
  if (op->else_body_.has_value() && op->else_body_.value()) {
    EnterScope();
    VisitStmt(op->else_body_.value());
    ExitScope();
  }

  // Verify IfStmt specific constraints
  VerifyIfStmt(op);
}

}  // namespace ir
}  // namespace pypto

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
#include <sstream>
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/verification_error.h"
#include "pypto/ir/transforms/verifier.h"

namespace pypto {
namespace ir {

// Implement type check error type to string conversion
namespace typecheck {
std::string ErrorTypeToString(ErrorType type) {
  switch (type) {
    case ErrorType::TYPE_KIND_MISMATCH:
      return "TYPE_KIND_MISMATCH";
    case ErrorType::DTYPE_MISMATCH:
      return "DTYPE_MISMATCH";
    case ErrorType::SHAPE_DIMENSION_MISMATCH:
      return "SHAPE_DIMENSION_MISMATCH";
    case ErrorType::SHAPE_VALUE_MISMATCH:
      return "SHAPE_VALUE_MISMATCH";
    case ErrorType::SIZE_MISMATCH:
      return "SIZE_MISMATCH";
    case ErrorType::IF_CONDITION_MUST_BE_SCALAR:
      return "IF_CONDITION_MUST_BE_SCALAR";
    case ErrorType::FOR_RANGE_MUST_BE_SCALAR:
      return "FOR_RANGE_MUST_BE_SCALAR";
    default:
      return "UNKNOWN";
  }
}
}  // namespace typecheck

namespace {
/**
 * @brief Helper visitor class for type checking
 *
 * Traverses the IR tree and checks type consistency in control flow constructs
 */
class TypeChecker : public IRVisitor {
 public:
  explicit TypeChecker(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const ForStmtPtr& op) override;
  void VisitStmt_(const WhileStmtPtr& op) override;
  void VisitStmt_(const IfStmtPtr& op) override;

  [[nodiscard]] const std::vector<Diagnostic>& GetDiagnostics() const { return diagnostics_; }

 private:
  std::vector<Diagnostic>& diagnostics_;

  /**
   * @brief Record an error
   */
  void RecordError(typecheck::ErrorType type, const std::string& message, const Span& span);

  /**
   * @brief Get the last statement in a statement block (recursive for SeqStmts)
   */
  StmtPtr GetLastStmt(const StmtPtr& stmt);

  /**
   * @brief Check type equality including shape for TensorType and TileType
   */
  void CheckTypeEquality(const TypePtr& type1, const TypePtr& type2, const std::string& context,
                         const std::string& desc1, const std::string& desc2, const Span& span);

  /**
   * @brief Check if two ExprPtr represent the same constant value
   */
  [[nodiscard]] bool IsSameConstant(const ExprPtr& expr1, const ExprPtr& expr2) const;

  /**
   * @brief Check if expression type is ScalarType
   */
  void CheckIsScalarType(const ExprPtr& expr, const std::string& context, const Span& span);
};

// TypeChecker implementation

void TypeChecker::RecordError(typecheck::ErrorType type, const std::string& message, const Span& span) {
  diagnostics_.emplace_back(DiagnosticSeverity::Error, "TypeCheck", static_cast<int>(type), message, span);
}

StmtPtr TypeChecker::GetLastStmt(const StmtPtr& stmt) {
  if (!stmt) return nullptr;

  // If it's a SeqStmts, recursively get the last statement
  if (auto seq = As<SeqStmts>(stmt)) {
    if (!seq->stmts_.empty()) {
      return GetLastStmt(seq->stmts_.back());
    }
  }

  return stmt;
}

void TypeChecker::CheckTypeEquality(const TypePtr& type1, const TypePtr& type2, const std::string& context,
                                    const std::string& desc1, const std::string& desc2, const Span& span) {
  if (!type1 || !type2) return;

  // Check ObjectKind first
  if (type1->GetKind() != type2->GetKind()) {
    std::ostringstream msg;
    msg << "Type kind mismatch in " << context << ": " << desc1 << " type '" << type1->TypeName()
        << "' != " << desc2 << " type '" << type2->TypeName() << "'";
    RecordError(typecheck::ErrorType::TYPE_KIND_MISMATCH, msg.str(), span);
    return;
  }

  // For ScalarType, check dtype
  if (type1->GetKind() == ObjectKind::ScalarType) {
    auto scalar1 = std::dynamic_pointer_cast<const ScalarType>(type1);
    auto scalar2 = std::dynamic_pointer_cast<const ScalarType>(type2);
    if (scalar1 && scalar2 && scalar1->dtype_ != scalar2->dtype_) {
      std::ostringstream msg;
      msg << "Dtype mismatch in " << context << ": " << desc1 << " dtype != " << desc2 << " dtype";
      RecordError(typecheck::ErrorType::DTYPE_MISMATCH, msg.str(), span);
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
      RecordError(typecheck::ErrorType::DTYPE_MISMATCH, msg.str(), span);
    }

    // Check shape dimensions count
    if (shaped1->shape_.size() != shaped2->shape_.size()) {
      std::ostringstream msg;
      msg << "Shape dimension count mismatch in " << context << ": " << desc1 << " has "
          << shaped1->shape_.size() << " dimensions, but " << desc2 << " has " << shaped2->shape_.size()
          << " dimensions";
      RecordError(typecheck::ErrorType::SHAPE_DIMENSION_MISMATCH, msg.str(), span);
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
          RecordError(typecheck::ErrorType::SHAPE_VALUE_MISMATCH, msg.str(), span);
        }
        // For symbolic dimensions, we skip detailed checking
        // A more sophisticated analysis would be needed for symbolic shape verification
      }
    }
  }
}

bool TypeChecker::IsSameConstant(const ExprPtr& expr1, const ExprPtr& expr2) const {
  if (!expr1 || !expr2) return false;

  // Check if both are ConstInt
  auto const_int1 = As<ConstInt>(expr1);
  auto const_int2 = As<ConstInt>(expr2);
  if (const_int1 && const_int2) {
    return const_int1->value_ == const_int2->value_;
  }

  // For symbolic expressions, we consider them potentially equal if they have the same structure
  // A more sophisticated check would require symbolic comparison, but for type checking
  // we primarily care about constant dimensions
  return false;
}

void TypeChecker::CheckIsScalarType(const ExprPtr& expr, const std::string& context, const Span& span) {
  if (!expr || !expr->GetType()) return;

  if (!As<ScalarType>(expr->GetType())) {
    std::ostringstream msg;
    msg << context << " must be ScalarType, but got " << expr->GetType()->TypeName();

    // Determine error type based on context
    auto error_type = (context.find("condition") != std::string::npos)
                          ? typecheck::ErrorType::IF_CONDITION_MUST_BE_SCALAR
                          : typecheck::ErrorType::FOR_RANGE_MUST_BE_SCALAR;

    RecordError(error_type, msg.str(), span);
  }
}

void TypeChecker::VisitStmt_(const ForStmtPtr& op) {
  if (!op) return;

  // Check start, stop, step must be ScalarType
  if (op->start_ && op->start_->GetType()) {
    CheckIsScalarType(op->start_, "ForStmt start", op->span_);
  }
  if (op->stop_ && op->stop_->GetType()) {
    CheckIsScalarType(op->stop_, "ForStmt stop", op->span_);
  }
  if (op->step_ && op->step_->GetType()) {
    CheckIsScalarType(op->step_, "ForStmt step", op->span_);
  }

  // Check type consistency between iter_args initValue, yield values, and return_vars
  if (!op->iter_args_.empty()) {
    StmtPtr last_stmt = GetLastStmt(op->body_);
    auto yield_stmt = As<YieldStmt>(last_stmt);

    if (yield_stmt) {
      // Check that all three vectors have the same size
      size_t num_iter_args = op->iter_args_.size();
      size_t num_yield_values = yield_stmt->value_.size();
      size_t num_return_vars = op->return_vars_.size();

      if (num_iter_args != num_yield_values || num_iter_args != num_return_vars) {
        std::ostringstream msg;
        msg << "ForStmt size mismatch: iter_args=" << num_iter_args << ", yield values=" << num_yield_values
            << ", return_vars=" << num_return_vars;
        RecordError(typecheck::ErrorType::SIZE_MISMATCH, msg.str(), op->span_);
      } else {
        // Check type consistency for each index
        for (size_t i = 0; i < num_iter_args; ++i) {
          const auto& iter_arg = op->iter_args_[i];
          const auto& yield_value = yield_stmt->value_[i];
          const auto& return_var = op->return_vars_[i];

          if (!iter_arg || !iter_arg->initValue_ || !yield_value || !return_var) continue;

          auto init_type = iter_arg->initValue_->GetType();
          auto yield_type = yield_value->GetType();
          auto return_type = return_var->GetType();

          if (!init_type || !yield_type || !return_type) continue;

          // Check initValue type == yield type
          CheckTypeEquality(init_type, yield_type, "ForStmt", "iter_arg[" + std::to_string(i) + "] initValue",
                            "yield value[" + std::to_string(i) + "]", op->span_);

          // Check yield type == return_var type
          CheckTypeEquality(yield_type, return_type, "ForStmt", "yield value[" + std::to_string(i) + "]",
                            "return_var[" + std::to_string(i) + "]", op->span_);

          // Check initValue type == return_var type (for completeness)
          CheckTypeEquality(init_type, return_type, "ForStmt",
                            "iter_arg[" + std::to_string(i) + "] initValue",
                            "return_var[" + std::to_string(i) + "]", op->span_);
        }
      }
    }
  }

  // Continue with default traversal
  IRVisitor::VisitStmt_(op);
}

void TypeChecker::VisitStmt_(const WhileStmtPtr& op) {
  if (!op) return;

  // Check condition must be ScalarType (bool)
  if (op->condition_ && op->condition_->GetType()) {
    CheckIsScalarType(op->condition_, "WhileStmt condition", op->span_);
  }

  // Check type consistency between iter_args initValue, yield values, and return_vars
  if (!op->iter_args_.empty()) {
    StmtPtr last_stmt = GetLastStmt(op->body_);
    auto yield_stmt = As<YieldStmt>(last_stmt);

    if (yield_stmt) {
      // Check that all three vectors have the same size
      size_t num_iter_args = op->iter_args_.size();
      size_t num_yield_values = yield_stmt->value_.size();
      size_t num_return_vars = op->return_vars_.size();

      if (num_iter_args != num_yield_values || num_iter_args != num_return_vars) {
        std::ostringstream msg;
        msg << "WhileStmt size mismatch: iter_args=" << num_iter_args << ", yield values=" << num_yield_values
            << ", return_vars=" << num_return_vars;
        RecordError(typecheck::ErrorType::SIZE_MISMATCH, msg.str(), op->span_);
      } else {
        // Check type consistency for each index
        for (size_t i = 0; i < num_iter_args; ++i) {
          const auto& iter_arg = op->iter_args_[i];
          const auto& yield_value = yield_stmt->value_[i];
          const auto& return_var = op->return_vars_[i];

          if (!iter_arg || !iter_arg->initValue_ || !yield_value || !return_var) continue;

          auto init_type = iter_arg->initValue_->GetType();
          auto yield_type = yield_value->GetType();
          auto return_type = return_var->GetType();

          if (!init_type || !yield_type || !return_type) continue;

          // Check initValue type == yield type
          CheckTypeEquality(init_type, yield_type, "WhileStmt",
                            "iter_arg[" + std::to_string(i) + "] initValue",
                            "yield value[" + std::to_string(i) + "]", op->span_);

          // Check yield type == return_var type
          CheckTypeEquality(yield_type, return_type, "WhileStmt", "yield value[" + std::to_string(i) + "]",
                            "return_var[" + std::to_string(i) + "]", op->span_);

          // Check initValue type == return_var type (for completeness)
          CheckTypeEquality(init_type, return_type, "WhileStmt",
                            "iter_arg[" + std::to_string(i) + "] initValue",
                            "return_var[" + std::to_string(i) + "]", op->span_);
        }
      }
    }
  }

  // Continue with default traversal
  IRVisitor::VisitStmt_(op);
}

void TypeChecker::VisitStmt_(const IfStmtPtr& op) {
  if (!op) return;

  // Check condition must be ScalarType
  if (op->condition_ && op->condition_->GetType()) {
    CheckIsScalarType(op->condition_, "IfStmt condition", op->span_);
  }

  // Check type consistency only if return_vars is not empty
  if (!op->return_vars_.empty() && op->else_body_.has_value()) {
    StmtPtr then_last = GetLastStmt(op->then_body_);
    StmtPtr else_last = GetLastStmt(op->else_body_.value());

    auto then_yield = As<YieldStmt>(then_last);
    auto else_yield = As<YieldStmt>(else_last);

    if (then_yield && else_yield) {
      // Check type consistency between then yield and else yield
      size_t num_then_values = then_yield->value_.size();
      size_t num_else_values = else_yield->value_.size();
      size_t num_return_vars = op->return_vars_.size();

      if (num_then_values != num_else_values || num_then_values != num_return_vars) {
        std::ostringstream msg;
        msg << "IfStmt size mismatch: then yield=" << num_then_values << ", else yield=" << num_else_values
            << ", return_vars=" << num_return_vars;
        RecordError(typecheck::ErrorType::SIZE_MISMATCH, msg.str(), op->span_);
      } else {
        // Check type consistency for each index
        for (size_t i = 0; i < num_then_values; ++i) {
          const auto& then_value = then_yield->value_[i];
          const auto& else_value = else_yield->value_[i];

          if (!then_value || !else_value) continue;

          auto then_type = then_value->GetType();
          auto else_type = else_value->GetType();

          if (!then_type || !else_type) continue;

          CheckTypeEquality(then_type, else_type, "IfStmt", "then yield value[" + std::to_string(i) + "]",
                            "else yield value[" + std::to_string(i) + "]", op->span_);
        }
      }
    }
  }

  // Continue with default traversal
  IRVisitor::VisitStmt_(op);
}

/**
 * @brief Transform a function by checking type consistency
 *
 * This transformation checks type consistency in control flow constructs and logs any violations.
 * The function is returned unchanged (type checking is read-only).
 */
FunctionPtr TransformTypeCheck(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "TypeCheck cannot run on null function";

  // Collect diagnostics during type checking
  std::vector<Diagnostic> diagnostics;
  TypeChecker checker(diagnostics);

  // Visit function body
  if (func->body_) {
    checker.VisitStmt(func->body_);
  }

  // If errors found, log the report
  if (!diagnostics.empty()) {
    std::string report = IRVerifier::GenerateReport(diagnostics);
    LOG_ERROR << "Type checking failed for function '" << func->name_ << "':\n" << report;
  }

  // Return the same function (type checking doesn't modify IR)
  return func;
}

}  // namespace

/**
 * @brief Type checking rule for use with IRVerifier
 */
class TypeCheckRule : public VerifyRule {
 public:
  [[nodiscard]] std::string GetName() const override { return "TypeCheck"; }

  void Verify(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) override {
    if (!func) {
      return;
    }

    // Create type checker and run checking
    TypeChecker checker(diagnostics);

    // Visit function body
    if (func->body_) {
      checker.VisitStmt(func->body_);
    }
  }
};

// Factory function for creating TypeCheckRule (for use with IRVerifier)
VerifyRulePtr CreateTypeCheckRule() { return std::make_shared<TypeCheckRule>(); }

// Factory function
namespace pass {
Pass TypeCheck() { return CreateFunctionPass(TransformTypeCheck, "TypeCheck"); }
}  // namespace pass

}  // namespace ir
}  // namespace pypto

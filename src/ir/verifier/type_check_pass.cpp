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
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "pypto/ir/verifier/verification_error.h"
#include "pypto/ir/verifier/verifier.h"

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
    case ErrorType::CONDITION_MUST_BE_BOOL:
      return "CONDITION_MUST_BE_BOOL";
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

  void VisitFunction(const FunctionPtr& func) override;
  void VisitExpr(const ExprPtr& expr) override;
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
   * @brief Check strict type equality at a call/control-flow boundary
   */
  void CheckTypeEquality(const TypePtr& type1, const TypePtr& type2, const std::string& context,
                         const std::string& desc1, const std::string& desc2, const Span& span);

  /**
   * @brief Check if expression type is ScalarType
   */
  void CheckIsScalarType(const ExprPtr& expr, const std::string& context, const Span& span);

  /**
   * @brief Check if expression is a ScalarType with BOOL dtype (for if/while conditions)
   */
  void CheckIsBoolCondition(const ExprPtr& expr, const std::string& context, const Span& span);

  /**
   * @brief Recursively validate explicit valid shapes carried by a type
   */
  void ValidateTypeValidShape(const TypePtr& type, const std::string& context, const Span& span);
};

// TypeChecker implementation

void TypeChecker::RecordError(typecheck::ErrorType type, const std::string& message, const Span& span) {
  diagnostics_.emplace_back(DiagnosticSeverity::Error, "TypeCheck", static_cast<int>(type), message, span);
}

void TypeChecker::VisitFunction(const FunctionPtr& func) {
  if (!func) return;

  for (size_t i = 0; i < func->params_.size(); ++i) {
    const auto& param = func->params_[i];
    if (!param) continue;
    ValidateTypeValidShape(
        param->GetType(),
        "Function '" + func->name_ + "' parameter[" + std::to_string(i) + "] '" + param->name_hint_ + "'",
        param->span_);
  }
  for (size_t i = 0; i < func->return_types_.size(); ++i) {
    ValidateTypeValidShape(func->return_types_[i],
                           "Function '" + func->name_ + "' return type[" + std::to_string(i) + "]",
                           func->span_);
  }
  if (func->body_) {
    VisitStmt(func->body_);
  }
}

void TypeChecker::VisitExpr(const ExprPtr& expr) {
  if (!expr) return;

  std::string context = expr->TypeName() + " expression";
  if (auto var = AsVarLike(expr)) {
    context = var->TypeName() + " '" + var->name_hint_ + "'";
  } else if (As<Call>(expr)) {
    context = "Call result";
  } else if (As<Submit>(expr)) {
    context = "Submit result";
  }
  ValidateTypeValidShape(expr->GetType(), context, expr->span_);
  IRVisitor::VisitExpr(expr);
}

void TypeChecker::ValidateTypeValidShape(const TypePtr& type, const std::string& context, const Span& span) {
  if (!type) return;

  if (auto tuple_type = As<TupleType>(type)) {
    for (size_t i = 0; i < tuple_type->types_.size(); ++i) {
      ValidateTypeValidShape(tuple_type->types_[i], context + " TupleType element[" + std::to_string(i) + "]",
                             span);
    }
    return;
  }

  const std::vector<ExprPtr>* valid_shape = nullptr;
  const std::vector<ExprPtr>* physical_shape = nullptr;
  if (auto distributed_type = As<DistributedTensorType>(type)) {
    if (distributed_type->tensor_view_) {
      valid_shape = &distributed_type->tensor_view_->valid_shape;
    }
    physical_shape = &distributed_type->shape_;
  } else if (auto tensor_type = As<TensorType>(type)) {
    if (tensor_type->tensor_view_) {
      valid_shape = &tensor_type->tensor_view_->valid_shape;
    }
    physical_shape = &tensor_type->shape_;
  } else if (auto tile_type = As<TileType>(type)) {
    if (tile_type->tile_view_) {
      valid_shape = &tile_type->tile_view_->valid_shape;
    }
    physical_shape = &tile_type->shape_;
  }

  if (!valid_shape || !physical_shape || valid_shape->empty()) return;
  for (const auto& error : ValidateValidShapeBounds(*valid_shape, *physical_shape, type->TypeName())) {
    auto error_type = error.violation == ValidShapeBoundsViolation::kRankMismatch
                          ? typecheck::ErrorType::SHAPE_DIMENSION_MISMATCH
                          : typecheck::ErrorType::SHAPE_VALUE_MISMATCH;
    RecordError(error_type, context + " has invalid " + error.message, span);
  }
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

  if (type1->GetKind() != type2->GetKind()) {
    std::ostringstream msg;
    msg << "Type kind mismatch in " << context << ": " << desc1 << " type '" << type1->TypeName()
        << "' != " << desc2 << " type '" << type2->TypeName() << "'";
    RecordError(typecheck::ErrorType::TYPE_KIND_MISMATCH, msg.str(), span);
    return;
  }

  if (auto tuple1 = As<TupleType>(type1)) {
    auto tuple2 = As<TupleType>(type2);
    if (tuple1->types_.size() != tuple2->types_.size()) {
      std::ostringstream msg;
      msg << "Tuple size mismatch in " << context << ": " << desc1 << " has " << tuple1->types_.size()
          << " elements, but " << desc2 << " has " << tuple2->types_.size() << " elements";
      RecordError(typecheck::ErrorType::SIZE_MISMATCH, msg.str(), span);
      return;
    }
    for (size_t i = 0; i < tuple1->types_.size(); ++i) {
      CheckTypeEquality(tuple1->types_[i], tuple2->types_[i], context,
                        desc1 + " tuple element[" + std::to_string(i) + "]",
                        desc2 + " tuple element[" + std::to_string(i) + "]", span);
    }
    return;
  }

  auto scalar1 = As<ScalarType>(type1);
  if (scalar1) {
    auto scalar2 = As<ScalarType>(type2);
    if (scalar1->dtype_ != scalar2->dtype_) {
      std::ostringstream msg;
      msg << "Dtype mismatch in " << context << ": " << desc1 << " dtype != " << desc2 << " dtype";
      RecordError(typecheck::ErrorType::DTYPE_MISMATCH, msg.str(), span);
    }
    return;
  }

  auto shaped1 = std::dynamic_pointer_cast<const ShapedType>(type1);
  auto shaped2 = std::dynamic_pointer_cast<const ShapedType>(type2);
  if (!shaped1 || !shaped2) return;

  if (shaped1->dtype_ != shaped2->dtype_) {
    std::ostringstream msg;
    msg << "Dtype mismatch in " << context << ": " << desc1 << " dtype != " << desc2 << " dtype";
    RecordError(typecheck::ErrorType::DTYPE_MISMATCH, msg.str(), span);
  }

  auto check_shape = [&](const std::vector<ExprPtr>& shape1, const std::vector<ExprPtr>& shape2,
                         const std::string& shape_name) {
    if (shape1.size() != shape2.size()) {
      std::ostringstream msg;
      msg << shape_name << " dimension count mismatch in " << context << ": " << desc1 << " has "
          << shape1.size() << " dimensions, but " << desc2 << " has " << shape2.size() << " dimensions";
      RecordError(typecheck::ErrorType::SHAPE_DIMENSION_MISMATCH, msg.str(), span);
      return false;
    }
    for (size_t i = 0; i < shape1.size(); ++i) {
      const auto& dim1 = shape1[i];
      const auto& dim2 = shape2[i];
      if (!dim1 || !dim2) continue;
      auto proof = ProveValidExtentEqual(dim1, dim2);
      if (proof != ProofResult::kTrue) {
        std::ostringstream msg;
        msg << shape_name << " dimension mismatch in " << context << ": " << desc1 << " dimension[" << i
            << "] = " << PythonPrint(dim1) << ", but " << desc2 << " dimension[" << i
            << "] = " << PythonPrint(dim2);
        if (proof == ProofResult::kUnknown) {
          msg << "; equality is not provable and this boundary has no runtime guard";
        }
        RecordError(typecheck::ErrorType::SHAPE_VALUE_MISMATCH, msg.str(), span);
      }
    }
    return true;
  };

  if (!check_shape(shaped1->shape_, shaped2->shape_, "Shape")) return;

  if (auto tensor1 = AsTensorTypeLike(type1)) {
    auto tensor2 = AsTensorTypeLike(type2);
    const auto& valid1 = tensor1->tensor_view_ && !tensor1->tensor_view_->valid_shape.empty()
                             ? tensor1->tensor_view_->valid_shape
                             : tensor1->shape_;
    const auto& valid2 = tensor2->tensor_view_ && !tensor2->tensor_view_->valid_shape.empty()
                             ? tensor2->tensor_view_->valid_shape
                             : tensor2->shape_;
    check_shape(valid1, valid2, "Valid shape");

    const auto pad1 = tensor1->tensor_view_ ? tensor1->tensor_view_->pad : PadValue::null;
    const auto pad2 = tensor2->tensor_view_ ? tensor2->tensor_view_->pad : PadValue::null;
    if (pad1 != pad2) {
      std::ostringstream msg;
      msg << "Tensor padding mismatch in " << context << ": " << desc1 << " pad != " << desc2 << " pad";
      RecordError(typecheck::ErrorType::TYPE_KIND_MISMATCH, msg.str(), span);
    }

    if (auto dist1 = As<DistributedTensorType>(type1)) {
      auto dist2 = As<DistributedTensorType>(type2);
      const bool same_window =
          dist1->window_buffer_.has_value() == dist2->window_buffer_.has_value() &&
          (!dist1->window_buffer_ || dist1->window_buffer_->get() == dist2->window_buffer_->get());
      if (!same_window) {
        std::ostringstream msg;
        msg << "Distributed window-buffer identity mismatch in " << context << ": " << desc1 << " and "
            << desc2 << " refer to different window buffers";
        RecordError(typecheck::ErrorType::TYPE_KIND_MISMATCH, msg.str(), span);
      }
    }
    return;
  }

  auto tile1 = As<TileType>(type1);
  auto tile2 = As<TileType>(type2);
  if (!tile1 || !tile2) return;
  const auto view1 = tile_view_semantics::GetEffectiveTileView(*tile1);
  const auto view2 = tile_view_semantics::GetEffectiveTileView(*tile2);
  const auto& valid1 = view1.valid_shape.empty() ? tile1->shape_ : view1.valid_shape;
  const auto& valid2 = view2.valid_shape.empty() ? tile2->shape_ : view2.valid_shape;
  check_shape(valid1, valid2, "Valid shape");
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

void TypeChecker::CheckIsBoolCondition(const ExprPtr& expr, const std::string& context, const Span& span) {
  if (!expr || !expr->GetType()) return;

  auto scalar = As<ScalarType>(expr->GetType());
  if (!scalar) return;  // Already reported by CheckIsScalarType

  if (scalar->dtype_ != DataType::BOOL) {
    std::ostringstream msg;
    msg << context << " dtype must be BOOL, but got " << scalar->dtype_.ToString();
    RecordError(typecheck::ErrorType::CONDITION_MUST_BE_BOOL, msg.str(), span);
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
          auto declared_type = iter_arg->GetType();
          auto yield_type = yield_value->GetType();
          auto return_type = return_var->GetType();

          if (!init_type || !declared_type || !yield_type || !return_type) continue;

          // Check every edge of the loop-carried type contract.
          CheckTypeEquality(init_type, declared_type, "ForStmt",
                            "iter_arg[" + std::to_string(i) + "] initValue",
                            "declared iter_arg[" + std::to_string(i) + "]", op->span_);

          CheckTypeEquality(declared_type, yield_type, "ForStmt",
                            "declared iter_arg[" + std::to_string(i) + "]",
                            "yield value[" + std::to_string(i) + "]", op->span_);

          CheckTypeEquality(yield_type, return_type, "ForStmt", "yield value[" + std::to_string(i) + "]",
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

  // Check condition must be ScalarType with BOOL dtype
  if (op->condition_ && op->condition_->GetType()) {
    CheckIsScalarType(op->condition_, "WhileStmt condition", op->span_);
    CheckIsBoolCondition(op->condition_, "WhileStmt condition", op->span_);
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
          auto declared_type = iter_arg->GetType();
          auto yield_type = yield_value->GetType();
          auto return_type = return_var->GetType();

          if (!init_type || !declared_type || !yield_type || !return_type) continue;

          // Check every edge of the loop-carried type contract.
          CheckTypeEquality(init_type, declared_type, "WhileStmt",
                            "iter_arg[" + std::to_string(i) + "] initValue",
                            "declared iter_arg[" + std::to_string(i) + "]", op->span_);

          CheckTypeEquality(declared_type, yield_type, "WhileStmt",
                            "declared iter_arg[" + std::to_string(i) + "]",
                            "yield value[" + std::to_string(i) + "]", op->span_);

          CheckTypeEquality(yield_type, return_type, "WhileStmt", "yield value[" + std::to_string(i) + "]",
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

  // Check condition must be ScalarType with BOOL dtype
  if (op->condition_ && op->condition_->GetType()) {
    CheckIsScalarType(op->condition_, "IfStmt condition", op->span_);
    CheckIsBoolCondition(op->condition_, "IfStmt condition", op->span_);
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
          const auto& return_var = op->return_vars_[i];

          if (!then_value || !else_value || !return_var) continue;

          auto then_type = then_value->GetType();
          auto else_type = else_value->GetType();
          auto return_type = return_var->GetType();

          if (!then_type || !else_type || !return_type) continue;

          CheckTypeEquality(then_type, else_type, "IfStmt", "then yield value[" + std::to_string(i) + "]",
                            "else yield value[" + std::to_string(i) + "]", op->span_);
          CheckTypeEquality(then_type, return_type, "IfStmt", "then yield value[" + std::to_string(i) + "]",
                            "return_var[" + std::to_string(i) + "]", op->span_);
        }
      }
    }
  }

  // Continue with default traversal
  IRVisitor::VisitStmt_(op);
}

}  // namespace

/**
 * @brief Type check property verifier for use with PropertyVerifierRegistry
 */
class TypeCheckPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "TypeCheck"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) {
      return;
    }

    for (const auto& [global_var, func] : program->functions_) {
      if (!func) {
        continue;
      }

      // Create type checker and run checking
      TypeChecker checker(diagnostics);
      checker.VisitFunction(func);
    }
  }
};

// Factory function for creating TypeCheck property verifier
PropertyVerifierPtr CreateTypeCheckPropertyVerifier() {
  return std::make_shared<TypeCheckPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

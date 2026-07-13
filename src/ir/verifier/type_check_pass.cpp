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
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
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

  void VisitStmt_(const ForStmtPtr& op) override;
  void VisitStmt_(const WhileStmtPtr& op) override;
  void VisitStmt_(const IfStmtPtr& op) override;
  void VisitVarLike_(const VarPtr& op) override;
  void VisitExpr_(const CallPtr& op) override;

  [[nodiscard]] const std::vector<Diagnostic>& GetDiagnostics() const { return diagnostics_; }

  /**
   * @brief Enforce the standing valid_shape well-formedness invariant on a type.
   *
   * For a TileType/TensorType carrying an explicit (non-empty) ``valid_shape``:
   * ``rank(valid_shape) == rank(shape)`` and, for every dim where both the valid
   * extent and the physical extent are compile-time constants,
   * ``0 <= valid_shape[i] <= shape[i]``. Symbolic (dynamic) valid extents are a
   * supported feature and are skipped. Recurses into ``TupleType`` elements.
   * Public so param/return types can be checked directly from ``Verify``.
   */
  void CheckValidShape(const TypePtr& type, const Span& span);

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
   * @brief Require effective valid extents to be provably equal at a control-flow join.
   */
  void CheckEffectiveValidShapeEquality(const std::vector<ExprPtr>& valid1,
                                        const std::vector<ExprPtr>& valid2, const std::string& context,
                                        const std::string& desc1, const std::string& desc2, const Span& span);

  /**
   * @brief Check if two ExprPtr represent the same constant value
   */
  [[nodiscard]] bool IsSameConstant(const ExprPtr& expr1, const ExprPtr& expr2) const;

  /**
   * @brief Check if expression type is ScalarType
   */
  void CheckIsScalarType(const ExprPtr& expr, const std::string& context, const Span& span);

  /**
   * @brief Check if expression is a ScalarType with BOOL dtype (for if/while conditions)
   */
  void CheckIsBoolCondition(const ExprPtr& expr, const std::string& context, const Span& span);
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

  // Tuple values cross a control-flow boundary as a single carrier, but every
  // element still participates in the join's type contract. Recurse so shaped
  // element semantics (including effective valid_shape) cannot be hidden
  // inside a tuple.
  if (const auto tuple1 = As<TupleType>(type1)) {
    const auto tuple2 = As<TupleType>(type2);
    if (tuple1->types_.size() != tuple2->types_.size()) {
      std::ostringstream msg;
      msg << "Tuple arity mismatch in " << context << ": " << desc1 << " has " << tuple1->types_.size()
          << " elements, but " << desc2 << " has " << tuple2->types_.size() << " elements";
      RecordError(typecheck::ErrorType::SIZE_MISMATCH, msg.str(), span);
      return;
    }

    for (size_t i = 0; i < tuple1->types_.size(); ++i) {
      CheckTypeEquality(tuple1->types_[i], tuple2->types_[i], context,
                        desc1 + " element[" + std::to_string(i) + "]",
                        desc2 + " element[" + std::to_string(i) + "]", span);
    }
    return;
  }

  // For tensor-like types (including DistributedTensorType) and TileType, check
  // dtype, physical shape, and the effective validity carried across the join.
  const auto tensor1 = AsTensorTypeLike(type1);
  const auto tensor2 = AsTensorTypeLike(type2);
  const auto tile1 = As<TileType>(type1);
  const auto tile2 = As<TileType>(type2);
  if ((tensor1 && tensor2) || (tile1 && tile2)) {
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

    if (tile1 && tile2) {
      CheckEffectiveValidShapeEquality(GetValidShape(tile1), GetValidShape(tile2), context, desc1, desc2,
                                       span);
    } else {
      CheckEffectiveValidShapeEquality(GetValidShape(tensor1), GetValidShape(tensor2), context, desc1, desc2,
                                       span);

      // Padding determines the value observed outside a partial valid box, so
      // two tensor values cannot share a control-flow result type when their
      // padding policies differ. An absent TensorView has the default null pad.
      const PadValue pad1 = tensor1->tensor_view_.has_value() ? tensor1->tensor_view_->pad : PadValue::null;
      const PadValue pad2 = tensor2->tensor_view_.has_value() ? tensor2->tensor_view_->pad : PadValue::null;
      if (pad1 != pad2) {
        std::ostringstream msg;
        msg << "TensorView pad mismatch in " << context << ": " << desc1 << " pad != " << desc2 << " pad";
        RecordError(typecheck::ErrorType::TYPE_KIND_MISMATCH, msg.str(), span);
      }

      // A distributed tensor is tied to a concrete WindowBuffer allocation.
      // Match structural type equality's default (non-auto-mapped) semantics:
      // presence must agree and a materialized back-reference must identify the
      // very same WindowBuffer Var.
      const auto distributed1 = As<DistributedTensorType>(type1);
      const auto distributed2 = As<DistributedTensorType>(type2);
      if (distributed1 && distributed2) {
        if (distributed1->window_buffer_.has_value() != distributed2->window_buffer_.has_value()) {
          std::ostringstream msg;
          msg << "DistributedTensorType window_buffer presence mismatch in " << context << ": " << desc1
              << (distributed1->window_buffer_.has_value() ? " has" : " doesn't have")
              << " a window_buffer, but " << desc2
              << (distributed2->window_buffer_.has_value() ? " has" : " doesn't have") << " a window_buffer";
          RecordError(typecheck::ErrorType::TYPE_KIND_MISMATCH, msg.str(), span);
        } else if (distributed1->window_buffer_.has_value() &&
                   *distributed1->window_buffer_ != *distributed2->window_buffer_) {
          std::ostringstream msg;
          msg << "DistributedTensorType window_buffer identity mismatch in " << context << ": " << desc1
              << " and " << desc2 << " refer to different window buffers";
          RecordError(typecheck::ErrorType::TYPE_KIND_MISMATCH, msg.str(), span);
        }
      }
    }
  }

  // For TileType, also check tile_view
  if (type1->GetKind() == ObjectKind::TileType) {
    // Check if both have tile_view or both don't
    if (tile1->tile_view_.has_value() != tile2->tile_view_.has_value()) {
      std::ostringstream msg;
      msg << "TileView presence mismatch in " << context << ": " << desc1
          << (tile1->tile_view_.has_value() ? " has" : " doesn't have") << " tile_view, but " << desc2
          << (tile2->tile_view_.has_value() ? " has" : " doesn't have") << " tile_view";
      RecordError(typecheck::ErrorType::TYPE_KIND_MISMATCH, msg.str(), span);
      return;
    }

    // If both have tile_view, compare the fields
    if (tile1->tile_view_.has_value() && tile2->tile_view_.has_value()) {
      const auto& view1 = tile1->tile_view_.value();
      const auto& view2 = tile2->tile_view_.value();

      // valid_shape was compared above in its effective form (unset == shape).

      // Check stride dimensions count
      if (view1.stride.size() != view2.stride.size()) {
        std::ostringstream msg;
        msg << "TileView stride dimension count mismatch in " << context << ": " << desc1 << " has "
            << view1.stride.size() << " dimensions, but " << desc2 << " has " << view2.stride.size()
            << " dimensions";
        RecordError(typecheck::ErrorType::SHAPE_DIMENSION_MISMATCH, msg.str(), span);
        return;
      }

      // Check each stride dimension
      for (size_t i = 0; i < view1.stride.size(); ++i) {
        const auto& stride1 = view1.stride[i];
        const auto& stride2 = view2.stride[i];

        if (!stride1 || !stride2) continue;

        if (!IsSameConstant(stride1, stride2)) {
          auto const_int1 = As<ConstInt>(stride1);
          auto const_int2 = As<ConstInt>(stride2);
          if (const_int1 && const_int2) {
            std::ostringstream msg;
            msg << "TileView stride dimension mismatch in " << context << ": " << desc1 << " stride[" << i
                << "] = " << const_int1->value_ << ", but " << desc2 << " stride[" << i
                << "] = " << const_int2->value_;
            RecordError(typecheck::ErrorType::SHAPE_VALUE_MISMATCH, msg.str(), span);
          }
        }
      }

      // Check start_offset presence
      if (static_cast<bool>(view1.start_offset) != static_cast<bool>(view2.start_offset)) {
        std::ostringstream msg;
        msg << "TileView start_offset presence mismatch in " << context << ": " << desc1
            << (view1.start_offset ? " has" : " doesn't have") << " start_offset, but " << desc2
            << (view2.start_offset ? " has" : " doesn't have") << " start_offset";
        RecordError(typecheck::ErrorType::TYPE_KIND_MISMATCH, msg.str(), span);
      } else if (view1.start_offset && view2.start_offset) {
        if (!IsSameConstant(view1.start_offset, view2.start_offset)) {
          auto const_int1 = As<ConstInt>(view1.start_offset);
          auto const_int2 = As<ConstInt>(view2.start_offset);
          if (const_int1 && const_int2) {
            std::ostringstream msg;
            msg << "TileView start_offset mismatch in " << context << ": " << desc1
                << " start_offset = " << const_int1->value_ << ", but " << desc2
                << " start_offset = " << const_int2->value_;
            RecordError(typecheck::ErrorType::SHAPE_VALUE_MISMATCH, msg.str(), span);
          }
        }
      }

      // Check blayout
      if (view1.blayout != view2.blayout) {
        std::ostringstream msg;
        msg << "TileView blayout mismatch in " << context << ": " << desc1 << " blayout != " << desc2
            << " blayout";
        RecordError(typecheck::ErrorType::TYPE_KIND_MISMATCH, msg.str(), span);
      }

      // Check slayout
      if (view1.slayout != view2.slayout) {
        std::ostringstream msg;
        msg << "TileView slayout mismatch in " << context << ": " << desc1 << " slayout != " << desc2
            << " slayout";
        RecordError(typecheck::ErrorType::TYPE_KIND_MISMATCH, msg.str(), span);
      }

      // Check fractal
      if (view1.fractal != view2.fractal) {
        std::ostringstream msg;
        msg << "TileView fractal mismatch in " << context << ": " << desc1 << " fractal = " << view1.fractal
            << ", but " << desc2 << " fractal = " << view2.fractal;
        RecordError(typecheck::ErrorType::SHAPE_VALUE_MISMATCH, msg.str(), span);
      }

      // Check pad
      if (view1.pad != view2.pad) {
        std::ostringstream msg;
        msg << "TileView pad mismatch in " << context << ": " << desc1 << " pad != " << desc2 << " pad";
        RecordError(typecheck::ErrorType::TYPE_KIND_MISMATCH, msg.str(), span);
      }
    }
  }
}

void TypeChecker::CheckEffectiveValidShapeEquality(const std::vector<ExprPtr>& valid1,
                                                   const std::vector<ExprPtr>& valid2,
                                                   const std::string& context, const std::string& desc1,
                                                   const std::string& desc2, const Span& span) {
  if (valid1.size() != valid2.size()) {
    std::ostringstream msg;
    msg << "Effective valid_shape rank mismatch in " << context << ": " << desc1 << " has " << valid1.size()
        << " dimensions, but " << desc2 << " has " << valid2.size() << " dimensions";
    RecordError(typecheck::ErrorType::SHAPE_DIMENSION_MISMATCH, msg.str(), span);
    return;
  }

  for (size_t i = 0; i < valid1.size(); ++i) {
    const ProofResult proof = ProveValidExtentEqual(valid1[i], valid2[i]);
    if (proof == ProofResult::kTrue) continue;

    std::ostringstream msg;
    msg << "Effective valid_shape mismatch in " << context << " at dimension " << i << ": " << desc1
        << " and " << desc2 << " must carry provably equal valid extents. "
        << (proof == ProofResult::kFalse ? "The extents are provably different."
                                         : "Their symbolic equality cannot be proven.");
    RecordError(typecheck::ErrorType::SHAPE_VALUE_MISMATCH, msg.str(), span);
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
          auto iter_type = iter_arg->GetType();
          auto yield_type = yield_value->GetType();
          auto return_type = return_var->GetType();

          if (!init_type || !iter_type || !yield_type || !return_type) continue;

          // The IterArg's declared type is the type visible to every use in the
          // loop body. It must therefore agree with every boundary carrier, not
          // merely rely on init/yield/return agreeing with one another.
          CheckTypeEquality(iter_type, init_type, "ForStmt",
                            "iter_arg[" + std::to_string(i) + "] declared type",
                            "iter_arg[" + std::to_string(i) + "] initValue", op->span_);
          CheckTypeEquality(iter_type, yield_type, "ForStmt",
                            "iter_arg[" + std::to_string(i) + "] declared type",
                            "yield value[" + std::to_string(i) + "]", op->span_);
          CheckTypeEquality(iter_type, return_type, "ForStmt",
                            "iter_arg[" + std::to_string(i) + "] declared type",
                            "return_var[" + std::to_string(i) + "]", op->span_);

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
          auto iter_type = iter_arg->GetType();
          auto yield_type = yield_value->GetType();
          auto return_type = return_var->GetType();

          if (!init_type || !iter_type || !yield_type || !return_type) continue;

          // The IterArg's declared type is the type visible to every use in the
          // loop body. It must therefore agree with every boundary carrier, not
          // merely rely on init/yield/return agreeing with one another.
          CheckTypeEquality(iter_type, init_type, "WhileStmt",
                            "iter_arg[" + std::to_string(i) + "] declared type",
                            "iter_arg[" + std::to_string(i) + "] initValue", op->span_);
          CheckTypeEquality(iter_type, yield_type, "WhileStmt",
                            "iter_arg[" + std::to_string(i) + "] declared type",
                            "yield value[" + std::to_string(i) + "]", op->span_);
          CheckTypeEquality(iter_type, return_type, "WhileStmt",
                            "iter_arg[" + std::to_string(i) + "] declared type",
                            "return_var[" + std::to_string(i) + "]", op->span_);

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
          CheckTypeEquality(else_type, return_type, "IfStmt", "else yield value[" + std::to_string(i) + "]",
                            "return_var[" + std::to_string(i) + "]", op->span_);
        }
      }
    }
  }

  // Continue with default traversal
  IRVisitor::VisitStmt_(op);
}

void TypeChecker::VisitVarLike_(const VarPtr& op) {
  // Every Var/IterArg carries a declared type — including the LHS bindings of
  // Submit results — so checking Var-like nodes covers all typed values without a
  // separate Submit override.
  if (op) CheckValidShape(op->GetType(), op->span_);
  IRVisitor::VisitVarLike_(op);
}

void TypeChecker::VisitExpr_(const CallPtr& op) {
  if (op) CheckValidShape(op->GetType(), op->span_);
  IRVisitor::VisitExpr_(op);
}

void TypeChecker::CheckValidShape(const TypePtr& type, const Span& span) {
  if (!type) return;

  // Recurse into tuple element types (multi-value binds, Submit returns).
  if (auto tuple_type = As<TupleType>(type)) {
    for (const auto& sub : tuple_type->types_) {
      CheckValidShape(sub, span);
    }
    return;
  }

  // Extract (valid_shape, shape) for whichever shaped view carries an explicit,
  // non-empty valid_shape. An empty/absent valid_shape means "fully valid"
  // (== shape) and needs no check.
  const std::vector<ExprPtr>* valid = nullptr;
  const std::vector<ExprPtr>* shape = nullptr;
  const char* kind = nullptr;
  if (auto tile_type = As<TileType>(type)) {
    if (tile_type->tile_view_.has_value() && !tile_type->tile_view_->valid_shape.empty()) {
      valid = &tile_type->tile_view_->valid_shape;
      shape = &tile_type->shape_;
      kind = "TileType";
    }
  } else if (auto tensor_type = AsTensorTypeLike(type)) {
    if (tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->valid_shape.empty()) {
      valid = &tensor_type->tensor_view_->valid_shape;
      shape = &tensor_type->shape_;
      kind = type->GetKind() == ObjectKind::DistributedTensorType ? "DistributedTensorType" : "TensorType";
    }
  }
  if (valid == nullptr) return;

  // rank(valid_shape) must equal rank(shape).
  if (valid->size() != shape->size()) {
    std::ostringstream msg;
    msg << kind << " valid_shape rank (" << valid->size() << ") does not match shape rank (" << shape->size()
        << ")";
    RecordError(typecheck::ErrorType::SHAPE_DIMENSION_MISMATCH, msg.str(), span);
    return;
  }

  // Per-dim bound: 0 <= valid_shape[i] <= shape[i]. Reject every relation the
  // shared arithmetic analyzer can disprove; genuinely unknown symbolic bounds
  // remain supported and defer to runtime.
  auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  for (size_t i = 0; i < valid->size(); ++i) {
    const ProofResult non_negative = ProveValidExtentLessEqual(zero, (*valid)[i]);
    const ProofResult within_shape = ProveValidExtentLessEqual((*valid)[i], (*shape)[i]);
    if (non_negative == ProofResult::kFalse || within_shape == ProofResult::kFalse) {
      std::ostringstream msg;
      msg << kind << " valid_shape[" << i
          << "] is provably out of bounds: it must satisfy 0 <= " << PythonPrint((*valid)[i])
          << " <= " << PythonPrint((*shape)[i]);
      RecordError(typecheck::ErrorType::SHAPE_VALUE_MISMATCH, msg.str(), span);
    }
  }
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

      // Enforce the standing valid_shape invariant on parameter and return
      // types too — the body walk below only reaches Var/Call types.
      for (const auto& param : func->params_) {
        if (param) checker.CheckValidShape(param->GetType(), param->span_);
      }
      for (const auto& rt : func->return_types_) {
        checker.CheckValidShape(rt, func->span_);
      }

      // Visit function body
      if (func->body_) {
        checker.VisitStmt(func->body_);
      }
    }
  }
};

// Factory function for creating TypeCheck property verifier
PropertyVerifierPtr CreateTypeCheckPropertyVerifier() {
  return std::make_shared<TypeCheckPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

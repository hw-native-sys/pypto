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

#include "pypto/ir/type_inference.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

BroadcastResult BroadcastShapes(const std::vector<ExprPtr>& shape1, const std::vector<ExprPtr>& shape2) {
  // Handle empty shapes
  if (shape1.empty() && shape2.empty()) {
    return BroadcastResult::Success({});
  }
  if (shape1.empty()) {
    return BroadcastResult::Success(shape2);
  }
  if (shape2.empty()) {
    return BroadcastResult::Success(shape1);
  }

  // Broadcast from right to left
  size_t max_ndim = std::max(shape1.size(), shape2.size());
  std::vector<ExprPtr> result_shape;
  result_shape.reserve(max_ndim);

  for (size_t i = 0; i < max_ndim; ++i) {
    // Get dimensions from right to left
    int64_t idx1 = static_cast<int64_t>(shape1.size()) - 1 - i;  // NOLINT
    int64_t idx2 = static_cast<int64_t>(shape2.size()) - 1 - i;  // NOLINT

    ExprPtr dim1 = (idx1 >= 0) ? shape1[idx1] : nullptr;
    ExprPtr dim2 = (idx2 >= 0) ? shape2[idx2] : nullptr;

    // If one dimension is missing, use the other
    if (!dim1) {
      result_shape.push_back(dim2);
      continue;
    }
    if (!dim2) {
      result_shape.push_back(dim1);
      continue;
    }

    // Check if dimensions are equal
    if (DimensionsEqual(dim1, dim2)) {
      result_shape.push_back(dim1);
      continue;
    }

    // Check if either dimension is 1 (broadcastable)
    auto const_dim1 = GetConstantDimension(dim1);
    auto const_dim2 = GetConstantDimension(dim2);

    if (const_dim1 && *const_dim1 == 1) {
      result_shape.push_back(dim2);
      continue;
    }
    if (const_dim2 && *const_dim2 == 1) {
      result_shape.push_back(dim1);
      continue;
    }

    // Dimensions are incompatible for broadcasting
    std::ostringstream oss;
    oss << "Cannot broadcast shapes: dimension " << i << " mismatch";
    return BroadcastResult::Failure(oss.str());
  }

  // Reverse result since we built it from right to left
  std::reverse(result_shape.begin(), result_shape.end());
  return BroadcastResult::Success(std::move(result_shape));
}

std::optional<DataType> PromoteDataTypes(DataType dtype1, DataType dtype2) {
  // If types are the same, return that type
  if (dtype1 == dtype2) {
    return dtype1;
  }

  // Float types take precedence
  bool is_float1 = dtype1.IsFloat();
  bool is_float2 = dtype2.IsFloat();

  if (is_float1 && !is_float2) {
    return dtype1;
  }
  if (is_float2 && !is_float1) {
    return dtype2;
  }

  // Both are floats or both are integers
  // Return the larger type
  size_t bits1 = dtype1.GetBit();
  size_t bits2 = dtype2.GetBit();

  if (bits1 > bits2) {
    return dtype1;
  }
  if (bits2 > bits1) {
    return dtype2;
  }

  // Same size - prefer signed over unsigned for integers
  if (!is_float1 && dtype1.IsSignedInt()) {
    return dtype1;
  }
  if (!is_float2 && dtype2.IsSignedInt()) {
    return dtype2;
  }

  // Default to first type
  return dtype1;
}

bool CheckTypeCompatibility(const TypePtr& type1, const TypePtr& type2) {
  // Check if both are scalar types
  auto scalar1 = As<ScalarType>(type1);
  auto scalar2 = As<ScalarType>(type2);
  if (scalar1 && scalar2) {
    return true;
  }

  // Check if both are tensor types
  auto tensor1 = As<TensorType>(type1);
  auto tensor2 = As<TensorType>(type2);
  if (tensor1 && tensor2) {
    return true;
  }

  // Check if both are tile types
  auto tile1 = As<TileType>(type1);
  auto tile2 = As<TileType>(type2);
  if (tile1 && tile2) {
    return true;
  }

  // Types are not compatible
  return false;
}

std::optional<DataType> ExtractDataType(const TypePtr& type) {
  // Try ScalarType
  if (auto scalar = As<ScalarType>(type)) {
    return scalar->dtype_;
  }

  // Try TensorType
  if (auto tensor = As<TensorType>(type)) {
    return tensor->dtype_;
  }

  // Try TileType
  if (auto tile = As<TileType>(type)) {
    return tile->dtype_;
  }

  return std::nullopt;
}

std::vector<ExprPtr> ExtractShape(const TypePtr& type) {
  // Try TensorType
  if (auto tensor = As<TensorType>(type)) {
    return tensor->shape_;
  }

  // Try TileType
  if (auto tile = As<TileType>(type)) {
    return tile->shape_;
  }

  // Not a shaped type
  return {};
}

std::optional<int64_t> GetConstantDimension(const ExprPtr& dim) {
  // Try to cast to ConstInt
  if (auto const_int = As<ConstInt>(dim)) {
    return const_int->value_;
  }

  // Not a constant
  return std::nullopt;
}

bool DimensionsEqual(const ExprPtr& dim1, const ExprPtr& dim2) {
  // Pointer equality (same object)
  if (dim1 == dim2) {
    return true;
  }

  // Try constant comparison
  auto const1 = GetConstantDimension(dim1);
  auto const2 = GetConstantDimension(dim2);

  if (const1 && const2) {
    return *const1 == *const2;
  }

  // For symbolic dimensions, prove equality via expression simplification.
  // Handles cases like `(x + 64) - x` vs `(x + 128) - (x + 64)` which both
  // reduce to 64 but are not structurally identical.
  //
  // Uses a thread_local analyzer so repeated calls on the slow path (e.g.
  // per-dim inside BroadcastShapes) reuse sub-analyzer state instead of
  // paying full setup per call.
  thread_local arith::Analyzer analyzer;
  return analyzer.CanProveEqual(dim1, dim2);
}

namespace {
bool AreComparableIntegerScalarExprs(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (!lhs || !rhs) {
    return false;
  }
  auto lhs_type = As<ScalarType>(lhs->GetType());
  auto rhs_type = As<ScalarType>(rhs->GetType());
  if (!lhs_type || !rhs_type || !lhs_type->dtype_.IsInt() || !rhs_type->dtype_.IsInt()) {
    return false;
  }
  return lhs_type->dtype_.IsSignedInt() == rhs_type->dtype_.IsSignedInt();
}
}  // namespace

ProofResult ProveValidExtentEqual(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (!AreComparableIntegerScalarExprs(lhs, rhs)) {
    return ProofResult::kUnknown;
  }
  if (AreExprsEqual(lhs, rhs)) {
    return ProofResult::kTrue;
  }

  thread_local arith::Analyzer analyzer;
  if (analyzer.CanProveEqual(lhs, rhs)) {
    return ProofResult::kTrue;
  }
  if (analyzer.CanProve(MakeNe(lhs, rhs))) {
    return ProofResult::kFalse;
  }
  return ProofResult::kUnknown;
}

ProofResult ProveValidExtentLessEqual(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (!AreComparableIntegerScalarExprs(lhs, rhs)) {
    return ProofResult::kUnknown;
  }
  if (AreExprsEqual(lhs, rhs)) {
    return ProofResult::kTrue;
  }

  thread_local arith::Analyzer analyzer;
  if (analyzer.CanProve(MakeLe(lhs, rhs))) {
    return ProofResult::kTrue;
  }
  if (analyzer.CanProve(MakeGt(lhs, rhs))) {
    return ProofResult::kFalse;
  }
  return ProofResult::kUnknown;
}

bool IsBroadcastable(const ExprPtr& source_dim, const ExprPtr& target_dim) {
  // If dimensions are equal, they're broadcastable
  if (DimensionsEqual(source_dim, target_dim)) {
    return true;
  }

  // Check if source is constant 1
  auto const_source = GetConstantDimension(source_dim);
  if (const_source && *const_source == 1) {
    return true;
  }

  // Check if target is constant 1
  auto const_target = GetConstantDimension(target_dim);
  if (const_target && *const_target == 1) {
    return true;
  }

  return false;
}

std::string FormatShape(const std::vector<ExprPtr>& shape) {
  if (shape.empty()) {
    return "[]";
  }

  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << PythonPrint(shape[i]);
  }
  oss << "]";
  return oss.str();
}

std::vector<ValidShapeBoundsError> ValidateValidShapeBounds(const std::vector<ExprPtr>& valid,
                                                            const std::vector<ExprPtr>& physical,
                                                            const std::string& type_kind) {
  if (valid.empty()) {
    return {};
  }

  if (valid.size() != physical.size()) {
    std::ostringstream msg;
    msg << type_kind << " valid_shape rank mismatch: got rank " << valid.size() << " " << FormatShape(valid)
        << ", but physical shape has rank " << physical.size() << " " << FormatShape(physical);
    return {{ValidShapeBoundsViolation::kRankMismatch, std::nullopt, msg.str()}};
  }

  std::vector<ValidShapeBoundsError> errors;
  static const auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, Span::unknown());
  for (size_t i = 0; i < valid.size(); ++i) {
    if (ProveValidExtentLessEqual(zero, valid[i]) == ProofResult::kFalse) {
      std::ostringstream msg;
      msg << type_kind << " valid_shape dimension " << i << " has provably negative extent "
          << PythonPrint(valid[i]) << "; expected 0 <= valid_shape[" << i << "] <= shape[" << i << "] ("
          << PythonPrint(physical[i]) << ")";
      errors.push_back({ValidShapeBoundsViolation::kNegativeExtent, i, msg.str()});
    }
    if (ProveValidExtentLessEqual(valid[i], physical[i]) == ProofResult::kFalse) {
      std::ostringstream msg;
      msg << type_kind << " valid_shape dimension " << i << " extent " << PythonPrint(valid[i])
          << " provably exceeds physical shape extent " << PythonPrint(physical[i]);
      errors.push_back({ValidShapeBoundsViolation::kExceedsPhysicalExtent, i, msg.str()});
    }
  }
  return errors;
}

// ============================================================================
// Slice rank-reduction (drop_dims) helpers
// ============================================================================

std::vector<int64_t> ParseSliceDropDims(const ExprPtr& drop_dims_arg, const std::vector<ExprPtr>& full_shape,
                                        const std::string& op_name) {
  if (!drop_dims_arg) {
    return {};
  }
  auto tuple = As<MakeTuple>(drop_dims_arg);
  CHECK(tuple) << op_name << " drop_dims must be a MakeTuple of compile-time int constants";

  std::vector<int64_t> axes;
  axes.reserve(tuple->elements_.size());
  std::vector<bool> seen(full_shape.size(), false);
  for (size_t i = 0; i < tuple->elements_.size(); ++i) {
    auto const_int = As<ConstInt>(tuple->elements_[i]);
    CHECK(const_int) << op_name << " drop_dims element " << i << " must be a compile-time int constant";
    int64_t axis = const_int->value_;
    CHECK(axis >= 0 && axis < static_cast<int64_t>(full_shape.size()))
        << op_name << " drop_dims index " << axis << " out of range for rank " << full_shape.size();
    CHECK(!seen[static_cast<size_t>(axis)]) << op_name << " drop_dims index " << axis << " is repeated";
    seen[static_cast<size_t>(axis)] = true;
    auto dim = GetConstantDimension(full_shape[static_cast<size_t>(axis)]);
    CHECK(dim.has_value() && *dim == 1)
        << op_name << " drop_dims index " << axis
        << " must select a static unit dimension (rank reduction only erases size-1 dims), but dim " << axis
        << " is " << (dim.has_value() ? std::to_string(*dim) : std::string("dynamic"));
    axes.push_back(axis);
  }
  std::sort(axes.begin(), axes.end());
  return axes;
}

std::vector<ExprPtr> ApplyDropDims(const std::vector<ExprPtr>& shape, const std::vector<int64_t>& drop_dims) {
  if (drop_dims.empty()) {
    return shape;
  }
  std::vector<bool> drop(shape.size(), false);
  for (int64_t d : drop_dims) {
    if (d >= 0 && d < static_cast<int64_t>(shape.size())) {
      drop[static_cast<size_t>(d)] = true;
    }
  }
  std::vector<ExprPtr> result;
  result.reserve(shape.size() - drop_dims.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    if (!drop[i]) {
      result.push_back(shape[i]);
    }
  }
  return result;
}

// ============================================================================
// Cross-function call return type deduction
// ============================================================================

namespace {

using TypeVarMap = std::unordered_map<const Var*, ExprPtr>;

struct CallTypeBindingConstraint {
  VarPtr var;
  ExprPtr existing;
  ExprPtr candidate;
  std::string context;
};

void BindCallTypeVar(const VarPtr& var, const ExprPtr& value, const std::string& context, TypeVarMap& var_map,
                     std::vector<CallTypeBindingConstraint>& constraints) {
  // A callee placeholder can also appear verbatim in the caller's annotation.
  // Treat that as an uninformative unification constraint so a later concrete
  // actual can refine the placeholder.
  if (var.get() == value.get()) return;

  auto [it, inserted] = var_map.emplace(var.get(), value);
  if (inserted) return;

  constraints.push_back({var, it->second, value, context});
}

void CollectCallExprBindings(const ExprPtr& pattern, const ExprPtr& value, const std::string& context,
                             TypeVarMap& var_map, std::vector<CallTypeBindingConstraint>& constraints) {
  if (!pattern || !value) return;
  if (auto var = As<Var>(pattern)) {
    BindCallTypeVar(var, value, context, var_map, constraints);
    return;
  }

  // Composite parameter metadata can bind variables when the actual metadata
  // has the same expression structure. A direct binding discovered elsewhere
  // still substitutes through differently-shaped composite return metadata.
  if (pattern->GetKind() != value->GetKind()) return;
  auto pattern_binary = std::dynamic_pointer_cast<const BinaryExpr>(pattern);
  auto value_binary = std::dynamic_pointer_cast<const BinaryExpr>(value);
  if (pattern_binary && value_binary) {
    CollectCallExprBindings(pattern_binary->left_, value_binary->left_, context + " left operand", var_map,
                            constraints);
    CollectCallExprBindings(pattern_binary->right_, value_binary->right_, context + " right operand", var_map,
                            constraints);
    return;
  }
  auto pattern_unary = std::dynamic_pointer_cast<const UnaryExpr>(pattern);
  auto value_unary = std::dynamic_pointer_cast<const UnaryExpr>(value);
  if (pattern_unary && value_unary) {
    CollectCallExprBindings(pattern_unary->operand_, value_unary->operand_, context + " operand", var_map,
                            constraints);
  }
}

void CollectCallExprVectorBindings(const std::vector<ExprPtr>& patterns, const std::vector<ExprPtr>& values,
                                   const std::string& context, TypeVarMap& var_map,
                                   std::vector<CallTypeBindingConstraint>& constraints) {
  const size_t count = std::min(patterns.size(), values.size());
  for (size_t i = 0; i < count; ++i) {
    CollectCallExprBindings(patterns[i], values[i], context + "[" + std::to_string(i) + "]", var_map,
                            constraints);
  }
}

const std::vector<ExprPtr>& GetEffectiveTensorValidShape(const TensorType& type) {
  if (type.tensor_view_ && !type.tensor_view_->valid_shape.empty()) {
    return type.tensor_view_->valid_shape;
  }
  return type.shape_;
}

const std::vector<ExprPtr>& GetEffectiveTileValidShape(const TileType& type) {
  if (type.tile_view_ && !type.tile_view_->valid_shape.empty()) {
    return type.tile_view_->valid_shape;
  }
  return type.shape_;
}

void CollectCallTypeBindings(const TypePtr& pattern, const TypePtr& value, const std::string& context,
                             TypeVarMap& var_map, std::vector<CallTypeBindingConstraint>& constraints) {
  if (!pattern || !value) return;

  if (auto pattern_tuple = As<TupleType>(pattern)) {
    auto value_tuple = As<TupleType>(value);
    if (!value_tuple) return;
    const size_t count = std::min(pattern_tuple->types_.size(), value_tuple->types_.size());
    for (size_t i = 0; i < count; ++i) {
      CollectCallTypeBindings(pattern_tuple->types_[i], value_tuple->types_[i],
                              context + " tuple element[" + std::to_string(i) + "]", var_map, constraints);
    }
    return;
  }

  if (auto pattern_tensor = AsTensorTypeLike(pattern)) {
    auto value_tensor = AsTensorTypeLike(value);
    if (!value_tensor) return;
    CollectCallExprVectorBindings(pattern_tensor->shape_, value_tensor->shape_, context + " physical shape",
                                  var_map, constraints);
    CollectCallExprVectorBindings(GetEffectiveTensorValidShape(*pattern_tensor),
                                  GetEffectiveTensorValidShape(*value_tensor), context + " valid shape",
                                  var_map, constraints);
    if (pattern_tensor->tensor_view_ && value_tensor->tensor_view_) {
      CollectCallExprVectorBindings(pattern_tensor->tensor_view_->stride, value_tensor->tensor_view_->stride,
                                    context + " tensor stride", var_map, constraints);
    }
    return;
  }

  auto pattern_tile = As<TileType>(pattern);
  auto value_tile = As<TileType>(value);
  if (!pattern_tile || !value_tile) return;
  CollectCallExprVectorBindings(pattern_tile->shape_, value_tile->shape_, context + " physical shape",
                                var_map, constraints);
  CollectCallExprVectorBindings(GetEffectiveTileValidShape(*pattern_tile),
                                GetEffectiveTileValidShape(*value_tile), context + " valid shape", var_map,
                                constraints);
  if (pattern_tile->tile_view_ && value_tile->tile_view_) {
    CollectCallExprVectorBindings(pattern_tile->tile_view_->stride, value_tile->tile_view_->stride,
                                  context + " tile stride", var_map, constraints);
    CollectCallExprBindings(pattern_tile->tile_view_->start_offset, value_tile->tile_view_->start_offset,
                            context + " tile start_offset", var_map, constraints);
  }
}

TypePtr SubstituteCallReturnType(const TypePtr& type, const TypeVarMap& var_map) {
  if (!type) return type;
  if (auto tuple = As<TupleType>(type)) {
    std::vector<TypePtr> elements;
    elements.reserve(tuple->types_.size());
    bool changed = false;
    for (const auto& element : tuple->types_) {
      auto new_element = SubstituteCallReturnType(element, var_map);
      if (new_element.get() != element.get()) changed = true;
      elements.push_back(std::move(new_element));
    }
    if (!changed) return type;
    return std::make_shared<TupleType>(std::move(elements));
  }

  const auto memref = GetTypeMemRef(type);
  return CloneTypeWithMemRefAndRemapExprs(
      type, memref, [&var_map](const ExprPtr& expr) { return transform_utils::Substitute(expr, var_map); });
}

}  // namespace

std::vector<TypePtr> DeduceCallReturnType(const std::vector<VarPtr>& callee_params,
                                          const std::vector<ExprPtr>& args,
                                          const std::vector<TypePtr>& return_types) {
  if (return_types.empty()) return return_types;
  CHECK(callee_params.size() == args.size())
      << "DeduceCallReturnType: callee_params size (" << callee_params.size() << ") must match args size ("
      << args.size() << ")";

  TypeVarMap var_map;
  std::vector<CallTypeBindingConstraint> constraints;
  for (size_t i = 0; i < callee_params.size(); ++i) {
    if (!callee_params[i] || !args[i]) continue;
    CollectCallTypeBindings(callee_params[i]->GetType(), args[i]->GetType(), "argument " + std::to_string(i),
                            var_map, constraints);
  }
  if (var_map.empty()) return return_types;

  // Validate repeated bindings only after all arguments have contributed.
  // A constraint may mention another callee placeholder that is bound by a
  // later argument (for example STAGED = NR * 64, then NR = world_size()).
  for (const auto& constraint : constraints) {
    if (structural_equal(constraint.existing, constraint.candidate)) continue;
    auto existing = transform_utils::Substitute(constraint.existing, var_map);
    auto candidate = transform_utils::Substitute(constraint.candidate, var_map);
    if (structural_equal(existing, candidate)) continue;
    CHECK(ProveValidExtentEqual(existing, candidate) == ProofResult::kTrue)
        << "Dynamic type variable '" << constraint.var->name_hint_ << "' has conflicting bindings "
        << PythonPrint(existing) << " and " << PythonPrint(candidate) << " that are not provably equal at "
        << constraint.context << "; cross-function calls do not emit a runtime shape guard";
  }

  std::vector<TypePtr> result;
  result.reserve(return_types.size());
  for (const auto& rt : return_types) {
    result.push_back(SubstituteCallReturnType(rt, var_map));
  }
  return result;
}

}  // namespace ir
}  // namespace pypto

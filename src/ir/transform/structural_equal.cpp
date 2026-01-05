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
#include <string>
#include <tuple>
#include <unordered_map>

#include "pypto/core/logging.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tensor_expr.h"
#include "pypto/ir/transform/transformers.h"

namespace pypto {
namespace ir {

// Forward declaration
class StructuralEqual;

/**
 * @brief Internal implementation: Structural equality checker for expressions
 *
 * Compares expression tree structure, ignoring Span (source location).
 * This class is not part of the public API - use structural_equal() function instead.
 */
class StructuralEqual {
 public:
  explicit StructuralEqual(bool enable_auto_mapping) : enable_auto_mapping_(enable_auto_mapping) {}
  bool operator()(const ExprPtr& lhs, const ExprPtr& rhs);

 private:
  bool Equal(const ExprPtr& lhs, const ExprPtr& rhs);
  bool EqualVar(const VarPtr& lhs, const VarPtr& rhs);
  bool EqualTensorVar(const TensorVarPtr& lhs, const TensorVarPtr& rhs);

  /**
   * @brief Generic field-based equality check for IR nodes
   *
   * Uses field descriptors to compare all fields of two nodes generically.
   *
   * @tparam NodePtr Shared pointer type to the node
   * @param lhs_op Left-hand side node
   * @param rhs_op Right-hand side node
   * @return true if all fields are equal
   */
  template <typename NodePtr>
  bool EqualWithFields(const NodePtr& lhs_op, const NodePtr& rhs_op) {
    using NodeType = typename NodePtr::element_type;
    auto descriptors = NodeType::GetFieldDescriptors();

    // Visit all fields using custom iteration over both nodes
    return EqualWithFieldsImpl(lhs_op, rhs_op, descriptors);
  }

  /**
   * @brief Implementation of field-based equality check
   *
   * Iterates over field descriptors and compares corresponding fields from LHS and RHS.
   *
   * @tparam NodePtr Shared pointer type to the node
   * @tparam Descriptors Tuple of field descriptors
   */
  template <typename NodePtr, typename Descriptors>
  bool EqualWithFieldsImpl(const NodePtr& lhs_op, const NodePtr& rhs_op, const Descriptors& descriptors) {
    return EqualWithFieldsTuple(lhs_op, rhs_op, descriptors,
                                std::make_index_sequence<std::tuple_size_v<Descriptors>>{});
  }

  /**
   * @brief Helper to iterate over tuple of descriptors using index sequence
   */
  template <typename NodePtr, typename Descriptors, std::size_t... Is>
  bool EqualWithFieldsTuple(const NodePtr& lhs_op, const NodePtr& rhs_op, const Descriptors& descriptors,
                            std::index_sequence<Is...>) {
    // Use fold expression to check all fields (short-circuit on first false)
    return (EqualField(lhs_op, rhs_op, std::get<Is>(descriptors)) && ...);
  }

  /**
   * @brief Compare a single field from two nodes using a descriptor
   */
  template <typename NodePtr, typename Descriptor>
  bool EqualField(const NodePtr& lhs_op, const NodePtr& rhs_op, const Descriptor& descriptor) {
    using FieldType = typename Descriptor::field_type;
    using KindTag = typename Descriptor::kind_tag;

    const auto& lhs_field = descriptor.Get(*lhs_op);
    const auto& rhs_field = descriptor.Get(*rhs_op);

    // Dispatch based on field type
    if constexpr (std::is_same_v<KindTag, reflection::IgnoreFieldTag>) {
      return true;
    } else if constexpr (reflection::IsExprField<FieldType>::value) {
      // Single ExprPtr field
      INTERNAL_CHECK(lhs_field) << "structural_equal encountered null lhs expression field";
      INTERNAL_CHECK(rhs_field) << "structural_equal encountered null rhs expression field";
      return Equal(lhs_field, rhs_field);
    } else if constexpr (reflection::IsExprVectorField<FieldType>::value) {
      // Vector of ExprPtr
      if (lhs_field.size() != rhs_field.size()) return false;
      for (size_t i = 0; i < lhs_field.size(); ++i) {
        INTERNAL_CHECK(lhs_field[i]) << "structural_equal encountered null lhs expression in vector at index "
                                     << i;
        INTERNAL_CHECK(rhs_field[i]) << "structural_equal encountered null rhs expression in vector at index "
                                     << i;
        if (!Equal(lhs_field[i], rhs_field[i])) return false;
      }
      return true;
    } else {
      // Scalar field - direct comparison
      return EqualScalar(lhs_field, rhs_field);
    }
  }

  /**
   * @brief Compare scalar fields
   */
  bool EqualScalar(const int& lhs, const int& rhs) const { return lhs == rhs; }

  bool EqualScalar(const std::string& lhs, const std::string& rhs) const { return lhs == rhs; }

  bool EqualScalar(const OpPtr& lhs, const OpPtr& rhs) const { return lhs->name_ == rhs->name_; }

  bool EqualScalar(const DataType& lhs, const DataType& rhs) const { return lhs == rhs; }

  bool enable_auto_mapping_;
  // Variable mapping: lhs variable pointer -> rhs variable pointer
  std::unordered_map<const Var*, const Var*> var_map_;
  // Tensor variable mapping: lhs tensor variable pointer -> rhs tensor variable pointer
  std::unordered_map<const TensorVar*, const TensorVar*> tensor_var_map_;
};

bool StructuralEqual::operator()(const ExprPtr& lhs, const ExprPtr& rhs) { return Equal(lhs, rhs); }

bool StructuralEqual::Equal(const ExprPtr& lhs, const ExprPtr& rhs) {
  // Fast path: reference equality
  if (lhs.get() == rhs.get()) return true;
  if (!lhs || !rhs) return false;

  // Type check: must be same concrete type
  if (std::string(lhs->type_name()) != std::string(rhs->type_name())) return false;

  // Dispatch to type-specific handlers using dynamic_cast
  // Var requires special handling for auto-mapping
  if (auto lhs_var = std::dynamic_pointer_cast<const Var>(lhs)) {
    return EqualVar(lhs_var, std::static_pointer_cast<const Var>(rhs));
  }

  // ConstInt: use field-based comparison
  if (auto lhs_const = std::dynamic_pointer_cast<const ConstInt>(lhs)) {
    return EqualWithFields(lhs_const, std::static_pointer_cast<const ConstInt>(rhs));
  }

  // Call: use field-based comparison
  if (auto lhs_call = std::dynamic_pointer_cast<const Call>(lhs)) {
    return EqualWithFields(lhs_call, std::static_pointer_cast<const Call>(rhs));
  }

  // Binary operations: use field-based comparison
  if (auto lhs_bin = std::dynamic_pointer_cast<const BinaryExpr>(lhs)) {
    return EqualWithFields(lhs_bin, std::static_pointer_cast<const BinaryExpr>(rhs));
  }

  // Unary operations: use field-based comparison
  if (auto lhs_un = std::dynamic_pointer_cast<const UnaryExpr>(lhs)) {
    return EqualWithFields(lhs_un, std::static_pointer_cast<const UnaryExpr>(rhs));
  }

  // Tensor expressions
  // TensorVar requires special handling for auto-mapping
  if (auto lhs_tvar = std::dynamic_pointer_cast<const TensorVar>(lhs)) {
    return EqualTensorVar(lhs_tvar, std::static_pointer_cast<const TensorVar>(rhs));
  }

  // Unknown type
  throw pypto::TypeError("Unknown expression type in StructuralEqual::Equal");
}

bool StructuralEqual::EqualVar(const VarPtr& lhs, const VarPtr& rhs) {
  if (!enable_auto_mapping_) {
    // Without auto mapping, require exact pointer match (strict identity)
    return lhs.get() == rhs.get();
  }

  // With auto mapping: maintain consistent variable mapping using pointers
  // This allows x+1 to equal y+1 by mapping x->y
  auto it = var_map_.find(lhs.get());
  if (it != var_map_.end()) {
    // Variable already mapped, verify consistency (same pointer)
    return it->second == rhs.get();
  }

  // New variable, add to mapping
  var_map_[lhs.get()] = rhs.get();
  return true;
}

bool StructuralEqual::EqualTensorVar(const TensorVarPtr& lhs, const TensorVarPtr& rhs) {
  if (!enable_auto_mapping_) {
    // Without auto mapping, require exact pointer match (strict identity)
    return lhs.get() == rhs.get();
  }

  // With auto mapping: maintain consistent tensor variable mapping using pointers
  auto it = tensor_var_map_.find(lhs.get());
  if (it != tensor_var_map_.end()) {
    // Variable already mapped, verify consistency (same pointer)
    return it->second == rhs.get();
  }

  // New variable, add to mapping
  tensor_var_map_[lhs.get()] = rhs.get();

  // Check dtype and shape equality (but not name, since we're auto-mapping)
  if (lhs->dtype_ != rhs->dtype_) return false;
  if (lhs->shape_.size() != rhs->shape_.size()) return false;
  for (size_t i = 0; i < lhs->shape_.size(); ++i) {
    if (!Equal(lhs->shape_[i], rhs->shape_[i])) return false;
  }
  return true;
}

// Public API implementation
bool structural_equal(const ExprPtr& lhs, const ExprPtr& rhs, bool enable_auto_mapping) {
  StructuralEqual checker(enable_auto_mapping);
  return checker(lhs, rhs);
}

}  // namespace ir
}  // namespace pypto

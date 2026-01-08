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
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transform/transformers.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// Forward declaration
class StructuralEqual;

/**
 * @brief Internal implementation: Structural equality checker for IR nodes
 *
 * Compares IR node tree structure, ignoring Span (source location).
 * This class is not part of the public API - use structural_equal() function instead.
 */
class StructuralEqual {
 public:
  explicit StructuralEqual(bool enable_auto_mapping) : enable_auto_mapping_(enable_auto_mapping) {}
  bool operator()(const IRNodePtr& lhs, const IRNodePtr& rhs);

 private:
  bool Equal(const IRNodePtr& lhs, const IRNodePtr& rhs);
  bool EqualVar(const VarPtr& lhs, const VarPtr& rhs);
  bool EqualType(const TypePtr& lhs, const TypePtr& rhs);

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
    } else if constexpr (reflection::IsIRNodeField<FieldType>::value) {
      // Single IRNodePtr field (includes ExprPtr, StmtPtr, etc.)
      INTERNAL_CHECK(lhs_field) << "structural_equal encountered null lhs IR node field";
      INTERNAL_CHECK(rhs_field) << "structural_equal encountered null rhs IR node field";
      return Equal(lhs_field, rhs_field);
    } else if constexpr (reflection::IsIRNodeVectorField<FieldType>::value) {
      // Vector of IRNodePtr (includes ExprPtr, StmtPtr, etc.)
      if (lhs_field.size() != rhs_field.size()) return false;
      for (size_t i = 0; i < lhs_field.size(); ++i) {
        INTERNAL_CHECK(lhs_field[i]) << "structural_equal encountered null lhs IR node in vector at index "
                                     << i;
        INTERNAL_CHECK(rhs_field[i]) << "structural_equal encountered null rhs IR node in vector at index "
                                     << i;
        if (!Equal(lhs_field[i], rhs_field[i])) return false;
      }
      return true;
    } else if constexpr (std::is_same_v<FieldType, TypePtr>) {
      // TypePtr - need special handling as it may contain ExprPtr
      INTERNAL_CHECK(lhs_field) << "structural_equal encountered null lhs TypePtr field";
      INTERNAL_CHECK(rhs_field) << "structural_equal encountered null rhs TypePtr field";
      return EqualType(lhs_field, rhs_field);
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
  std::unordered_map<VarPtr, VarPtr> lhs_to_rhs_var_map_;
  std::unordered_map<VarPtr, VarPtr> rhs_to_lhs_var_map_;
};

bool StructuralEqual::operator()(const IRNodePtr& lhs, const IRNodePtr& rhs) { return Equal(lhs, rhs); }

// Type dispatch macro for generic field-based comparison
#define EQUAL_DISPATCH(Type)                                                       \
  if (auto lhs_##Type = std::dynamic_pointer_cast<const Type>(lhs)) {              \
    return EqualWithFields(lhs_##Type, std::static_pointer_cast<const Type>(rhs)); \
  }

bool StructuralEqual::Equal(const IRNodePtr& lhs, const IRNodePtr& rhs) {
  // Fast path: reference equality
  if (lhs.get() == rhs.get()) return true;
  if (!lhs || !rhs) return false;

  // Type check: must be same concrete type
  if (lhs->TypeName() != rhs->TypeName()) return false;

  // Dispatch to type-specific handlers using dynamic_cast
  // Check types that require special handling first
  if (auto lhs_var = std::dynamic_pointer_cast<const Var>(lhs)) {
    return EqualVar(lhs_var, std::static_pointer_cast<const Var>(rhs));
  }

  // All other types use generic field-based comparison
  EQUAL_DISPATCH(ConstInt)
  EQUAL_DISPATCH(Call)
  EQUAL_DISPATCH(BinaryExpr)
  EQUAL_DISPATCH(UnaryExpr)
  EQUAL_DISPATCH(AssignStmt)
  EQUAL_DISPATCH(Stmt)

  // Unknown IR node type
  throw pypto::TypeError("Unknown IR node type in StructuralEqual::Equal");
}

#undef EQUAL_DISPATCH

bool StructuralEqual::EqualType(const TypePtr& lhs, const TypePtr& rhs) {
  if (lhs->TypeName() != rhs->TypeName()) return false;

  if (auto lhs_scalar = std::dynamic_pointer_cast<const ScalarType>(lhs)) {
    auto rhs_scalar = std::dynamic_pointer_cast<const ScalarType>(rhs);
    if (!rhs_scalar) return false;
    return lhs_scalar->dtype_ == rhs_scalar->dtype_;
  } else if (auto lhs_tensor = std::dynamic_pointer_cast<const TensorType>(lhs)) {
    auto rhs_tensor = std::dynamic_pointer_cast<const TensorType>(rhs);
    if (!rhs_tensor) return false;
    if (lhs_tensor->dtype_ != rhs_tensor->dtype_) return false;
    if (lhs_tensor->shape_.size() != rhs_tensor->shape_.size()) return false;
    for (size_t i = 0; i < lhs_tensor->shape_.size(); ++i) {
      if (!Equal(lhs_tensor->shape_[i], rhs_tensor->shape_[i])) return false;
    }
    return true;
  } else if (std::dynamic_pointer_cast<const UnknownType>(lhs)) {
    // UnknownType has no fields, so if TypeName() matches, they are equal
    return true;
  }
  // If TypeName() matches but none of the known types, this is an error
  INTERNAL_CHECK(false) << "EqualType encountered unhandled Type: " << lhs->TypeName();
  return false;
}

bool StructuralEqual::EqualVar(const VarPtr& lhs, const VarPtr& rhs) {
  if (!enable_auto_mapping_) {
    // Without auto mapping, require exact pointer match (strict identity)
    return lhs.get() == rhs.get();
  }

  // Check type equality first - only add to mapping if types match
  if (!EqualType(lhs->GetType(), rhs->GetType())) {
    return false;
  }

  // With auto mapping: maintain consistent variable mapping using pointers
  // This allows x+1 to equal y+1 by mapping x->y
  auto it = lhs_to_rhs_var_map_.find(lhs);
  if (it != lhs_to_rhs_var_map_.end()) {
    // Variable already mapped, verify consistency (same pointer)
    return it->second == rhs;
  }

  // If rhs is already mapped to a different lhs, return false
  auto rhs_it = rhs_to_lhs_var_map_.find(rhs);
  if (rhs_it != rhs_to_lhs_var_map_.end() && rhs_it->second != lhs) {
    return false;
  }

  // New variable, add to mapping
  lhs_to_rhs_var_map_[lhs] = rhs;
  rhs_to_lhs_var_map_[rhs] = lhs;
  return true;
}

// Public API implementation
bool structural_equal(const IRNodePtr& lhs, const IRNodePtr& rhs, bool enable_auto_mapping) {
  StructuralEqual checker(enable_auto_mapping);
  return checker(lhs, rhs);
}

}  // namespace ir
}  // namespace pypto

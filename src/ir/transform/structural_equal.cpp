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

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
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
 *
 * Implements the FieldIterator visitor interface for generic field-based comparison.
 * Uses the dual-node Visit overload which calls visitor methods with two field arguments.
 */
class StructuralEqual {
 public:
  using result_type = bool;

  explicit StructuralEqual(bool enable_auto_mapping) : enable_auto_mapping_(enable_auto_mapping) {}
  bool operator()(const IRNodePtr& lhs, const IRNodePtr& rhs);

  // FieldIterator visitor interface (dual-node version - methods receive two fields)
  [[nodiscard]] result_type InitResult() const { return true; }

  template <typename IRNodePtrType>
  result_type VisitIRNodeField(const IRNodePtrType& lhs, const IRNodePtrType& rhs) {
    INTERNAL_CHECK(lhs) << "structural_equal encountered null lhs IR node field";
    INTERNAL_CHECK(rhs) << "structural_equal encountered null rhs IR node field";
    return Equal(lhs, rhs);
  }

  template <typename IRNodePtrType>
  result_type VisitIRNodeVectorField(const std::vector<IRNodePtrType>& lhs,
                                     const std::vector<IRNodePtrType>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      INTERNAL_CHECK(lhs[i]) << "structural_equal encountered null lhs IR node in vector at index " << i;
      INTERNAL_CHECK(rhs[i]) << "structural_equal encountered null rhs IR node in vector at index " << i;
      if (!Equal(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  template <typename KeyType, typename ValueType, typename Compare>
  result_type VisitIRNodeMapField(const std::map<KeyType, ValueType, Compare>& lhs,
                                  const std::map<KeyType, ValueType, Compare>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    auto lhs_it = lhs.begin();
    auto rhs_it = rhs.begin();
    while (lhs_it != lhs.end()) {
      INTERNAL_CHECK(lhs_it->first) << "structural_equal encountered null lhs key in map";
      INTERNAL_CHECK(lhs_it->second) << "structural_equal encountered null lhs value in map";
      INTERNAL_CHECK(rhs_it->first) << "structural_equal encountered null rhs key in map";
      INTERNAL_CHECK(rhs_it->second) << "structural_equal encountered null rhs value in map";
      // Compare keys by name (keys are Op types, not IRNode)
      if (lhs_it->first->name_ != rhs_it->first->name_) return false;
      // Compare values (values are IRNode types)
      if (!Equal(lhs_it->second, rhs_it->second)) return false;
      ++lhs_it;
      ++rhs_it;
    }
    return true;
  }

  // Leaf field comparisons (dual-node version)
  result_type VisitLeafField(const int& lhs, const int& rhs) const { return lhs == rhs; }

  result_type VisitLeafField(const std::string& lhs, const std::string& rhs) const { return lhs == rhs; }

  result_type VisitLeafField(const OpPtr& lhs, const OpPtr& rhs) const { return lhs->name_ == rhs->name_; }

  result_type VisitLeafField(const DataType& lhs, const DataType& rhs) const { return lhs == rhs; }

  result_type VisitLeafField(const TypePtr& lhs, const TypePtr& rhs) { return EqualType(lhs, rhs); }

  result_type VisitLeafField(const std::vector<TypePtr>& lhs, const std::vector<TypePtr>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      INTERNAL_CHECK(lhs[i]) << "structural_equal encountered null lhs TypePtr in vector at index " << i;
      INTERNAL_CHECK(rhs[i]) << "structural_equal encountered null rhs TypePtr in vector at index " << i;
      if (!EqualType(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  result_type VisitLeafField(const Span& lhs, const Span& rhs) const {
    INTERNAL_UNREACHABLE << "structural_equal should not visit Span field";
  }

  // Field kind hooks
  template <typename FVisitOp>
  void VisitIgnoreField([[maybe_unused]] FVisitOp&& visit_op) {
    // Ignored fields are always considered equal
  }

  template <typename FVisitOp>
  void VisitDefField(FVisitOp&& visit_op) {
    bool enable_auto_mapping = true;
    std::swap(enable_auto_mapping, enable_auto_mapping_);
    visit_op();
    std::swap(enable_auto_mapping, enable_auto_mapping_);
  }

  template <typename FVisitOp>
  void VisitUsualField(FVisitOp&& visit_op) {
    visit_op();
  }

  // Combine results (AND logic)
  template <typename Desc>
  void CombineResult(result_type& accumulator, result_type field_result, [[maybe_unused]] const Desc& desc) {
    accumulator = accumulator && field_result;
  }

 private:
  bool Equal(const IRNodePtr& lhs, const IRNodePtr& rhs);
  bool EqualVar(const VarPtr& lhs, const VarPtr& rhs);
  bool EqualType(const TypePtr& lhs, const TypePtr& rhs);

  /**
   * @brief Generic field-based equality check for IR nodes using FieldIterator
   *
   * Uses the dual-node Visit overload which passes two fields to each visitor method.
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

    return std::apply(
        [&](auto&&... descs) {
          return reflection::FieldIterator<NodeType, StructuralEqual, decltype(descs)...>::Visit(
              *lhs_op, *rhs_op, *this, descs...);
        },
        descriptors);
  }

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
  EQUAL_DISPATCH(IfStmt)
  EQUAL_DISPATCH(YieldStmt)
  EQUAL_DISPATCH(ForStmt)
  EQUAL_DISPATCH(SeqStmts)
  EQUAL_DISPATCH(OpStmts)
  EQUAL_DISPATCH(Function)
  EQUAL_DISPATCH(Program)

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
  INTERNAL_UNREACHABLE << "EqualType encountered unhandled Type: " << lhs->TypeName();
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

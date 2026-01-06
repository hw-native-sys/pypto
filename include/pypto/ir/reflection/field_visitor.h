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

#ifndef PYPTO_IR_REFLECTION_FIELD_VISITOR_H_
#define PYPTO_IR_REFLECTION_FIELD_VISITOR_H_

#include <memory>
#include <type_traits>
#include <vector>

#include "pypto/ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

// Forward declarations
class IRNode;
using IRNodePtr = std::shared_ptr<const IRNode>;

namespace reflection {

/**
 * @brief Type trait to check if a type is a shared_ptr to an IRNode-derived type
 *
 * Used to dispatch field visiting logic based on field type.
 * This is the general trait that supports all IRNode types (Expr, Stmt, etc.).
 */
template <typename T, typename = void>
struct IsIRNodeField : std::false_type {};

// Generic specialization for any shared_ptr<const T> where T derives from IRNode
template <typename IRNodeType>
struct IsIRNodeField<std::shared_ptr<const IRNodeType>,
                     std::enable_if_t<std::is_base_of_v<IRNode, IRNodeType>>> : std::true_type {};

/**
 * @brief Type trait to check if a type is std::vector of IRNode pointers
 *
 * Used to handle collections of IR nodes specially.
 * Matches any vector<shared_ptr<const T>> where T derives from IRNode.
 */
template <typename T>
struct IsIRNodeVectorField : std::false_type {};

// Generic specialization for any vector<shared_ptr<const T>> where T derives from IRNode
template <typename IRNodeType>
struct IsIRNodeVectorField<std::vector<std::shared_ptr<const IRNodeType>>>
    : std::integral_constant<bool, std::is_base_of_v<IRNode, IRNodeType>> {};

/**
 * @brief Generic field iterator for compile-time field visitation
 *
 * Iterates over all fields in an IR node using field descriptors,
 * calling appropriate visitor methods for each field type.
 *
 * Uses C++17 fold expressions for compile-time iteration.
 *
 * @tparam NodeType The IR node type being visited
 * @tparam Visitor The visitor type (must have result_type and visit methods)
 * @tparam Descriptors Parameter pack of field descriptors
 */
template <typename NodeType, typename Visitor, typename... Descriptors>
class FieldIterator {
 public:
  using result_type = typename Visitor::result_type;

  /**
   * @brief Visit all fields of a node
   *
   * @param node The node instance to visit
   * @param visitor The visitor instance
   * @param descriptors Field descriptor instances
   * @return Accumulated result from visiting all fields
   */
  static result_type Visit(const NodeType& node, Visitor& visitor, const Descriptors&... descriptors) {
    result_type result = visitor.InitResult();
    // C++17 fold expression: expands to (VisitSingleField(...), VisitSingleField(...), ...)
    (VisitSingleField(node, visitor, descriptors, result), ...);
    return result;
  }

 private:
  /**
   * @brief Visit a single field using its descriptor
   *
   * Dispatches based on field kind (IGNORE/DEF/USUAL) and field type (Expr/vector/scalar).
   *
   * @tparam Desc The field descriptor type
   * @param node The node instance
   * @param visitor The visitor instance
   * @param desc The field descriptor
   * @param result Accumulated result (modified in place)
   */
  template <typename Desc>
  static void VisitSingleField(const NodeType& node, Visitor& visitor, const Desc& desc,
                               result_type& result) {
    using KindTag = typename Desc::kind_tag;
    using FieldType = typename Desc::field_type;

    const auto& field = desc.Get(node);

    // Dispatch based on field type
    if constexpr (std::is_same_v<KindTag, IgnoreFieldTag>) {
      return;
    } else if constexpr (IsIRNodeField<FieldType>::value) {
      // Single IRNodePtr field (includes ExprPtr, StmtPtr, etc.)
      auto field_result = visitor.VisitIRNodeField(field);
      visitor.AccumulateResult(result, field_result, desc);
    } else if constexpr (IsIRNodeVectorField<FieldType>::value) {
      // Vector of IRNodePtr (includes ExprPtr, StmtPtr, etc.)
      auto field_result = visitor.VisitIRNodeVectorField(field);
      visitor.AccumulateResult(result, field_result, desc);
    } else {
      // Scalar field (int, string, OpPtr, etc.)
      auto field_result = visitor.VisitScalarField(field);
      visitor.AccumulateResult(result, field_result, desc);
    }
  }
};

}  // namespace reflection
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_REFLECTION_FIELD_VISITOR_H_

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

#include "pypto/core/logging.h"
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
 * Iterates over all fields in one or more IR nodes using field descriptors,
 * calling appropriate visitor methods for each field type.
 *
 * Supports single-node visitation (e.g., for hashing) and multi-node visitation
 * (e.g., for equality comparison). The visitor methods receive as many field
 * arguments as there are nodes being visited.
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
   * @brief Visit all fields of a single node
   *
   * Visitor methods are called with single field arguments:
   *   - VisitIRNodeField(field)
   *   - VisitIRNodeVectorField(field)
   *   - VisitLeafField(field)
   *
   * @param node The node instance to visit
   * @param visitor The visitor instance
   * @param descriptors Field descriptor instances
   * @return Accumulated result from visiting all fields
   */
  static result_type Visit(const NodeType& node, Visitor& visitor, const Descriptors&... descriptors) {
    result_type result = visitor.InitResult();
    (VisitField(visitor, descriptors, result, node), ...);
    return result;
  }

  /**
   * @brief Visit all fields of two nodes pairwise
   *
   * Visitor methods are called with two field arguments:
   *   - VisitIRNodeField(lhs_field, rhs_field)
   *   - VisitIRNodeVectorField(lhs_field, rhs_field)
   *   - VisitLeafField(lhs_field, rhs_field)
   *
   * @param lhs Left-hand side node
   * @param rhs Right-hand side node
   * @param visitor The visitor instance
   * @param descriptors Field descriptor instances
   * @return Accumulated result from visiting all field pairs
   */
  static result_type Visit(const NodeType& lhs, const NodeType& rhs, Visitor& visitor,
                           const Descriptors&... descriptors) {
    result_type result = visitor.InitResult();
    (VisitField(visitor, descriptors, result, lhs, rhs), ...);
    return result;
  }

 private:
  /**
   * @brief Visit a single field from N nodes using its descriptor
   *
   * Dispatches based on field kind (IGNORE/DEF/USUAL).
   *
   * @tparam Desc The field descriptor type
   * @tparam Nodes Parameter pack of node types (all must be NodeType)
   */
  template <typename Desc, typename... Nodes>
  static void VisitField(Visitor& visitor, const Desc& desc, result_type& result, const Nodes&... nodes) {
    using KindTag = typename Desc::kind_tag;

    if constexpr (std::is_same_v<KindTag, IgnoreFieldTag>) {
      visitor.VisitIgnoreField([&]() { VisitFieldImpl(visitor, desc, result, nodes...); });
    } else if constexpr (std::is_same_v<KindTag, DefFieldTag>) {
      visitor.VisitDefField([&]() { VisitFieldImpl(visitor, desc, result, nodes...); });
    } else if constexpr (std::is_same_v<KindTag, UsualFieldTag>) {
      visitor.VisitUsualField([&]() { VisitFieldImpl(visitor, desc, result, nodes...); });
    } else {
      INTERNAL_UNREACHABLE << "Invalid field kind tag: " << typeid(KindTag).name() << " for field "
                           << desc.name;
    }
  }

  /**
   * @brief Implementation of field visitation
   *
   * Dispatches based on field type (IRNode/vector/scalar) and calls
   * the appropriate visitor method with fields from all nodes.
   */
  template <typename Desc, typename... Nodes>
  static void VisitFieldImpl(Visitor& visitor, const Desc& desc, result_type& result, const Nodes&... nodes) {
    using FieldType = typename Desc::field_type;

    if constexpr (IsIRNodeField<FieldType>::value) {
      // Single IRNodePtr field - expand to visitor.VisitIRNodeField(desc.Get(node1), desc.Get(node2), ...)
      auto field_result = visitor.VisitIRNodeField(desc.Get(nodes)...);
      visitor.CombineResult(result, field_result, desc);
    } else if constexpr (IsIRNodeVectorField<FieldType>::value) {
      // Vector of IRNodePtr
      auto field_result = visitor.VisitIRNodeVectorField(desc.Get(nodes)...);
      visitor.CombineResult(result, field_result, desc);
    } else {
      // Scalar field (int, string, OpPtr, etc.)
      auto field_result = visitor.VisitLeafField(desc.Get(nodes)...);
      visitor.CombineResult(result, field_result, desc);
    }
  }
};

}  // namespace reflection
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_REFLECTION_FIELD_VISITOR_H_

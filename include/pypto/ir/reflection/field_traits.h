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

#ifndef PYPTO_IR_REFLECTION_FIELD_TRAITS_H_
#define PYPTO_IR_REFLECTION_FIELD_TRAITS_H_

namespace pypto {
namespace ir {
namespace reflection {

/**
 * @brief Field kind tags for compile-time dispatch
 *
 * These zero-sized tag types are used to classify fields into three categories:
 * - DEF: Binding/definition fields (e.g., loop variables, function parameters)
 * - USUAL: Normal fields to visit and compare
 * - IGNORE: Fields to skip in structural comparisons (e.g., source locations)
 */
struct DefFieldTag {};     // For binding variables - enables auto-mapping
struct UsualFieldTag {};   // Normal fields to visit/compare
struct IgnoreFieldTag {};  // Skip in comparisons (e.g., span)

/**
 * @brief Field descriptor template
 *
 * Captures metadata about a field in an IR node using pointer-to-member.
 * Field descriptors are stored at compile-time and enable generic field visitation.
 *
 * @tparam NodeType The IR node class containing this field
 * @tparam FieldType The type of the field
 * @tparam KindTag The field kind tag (DefFieldTag, UsualFieldTag, or IgnoreFieldTag)
 */
template <typename NodeType, typename FieldType, typename KindTag>
struct FieldDescriptor {
  using node_type = NodeType;
  using field_type = FieldType;
  using kind_tag = KindTag;

  FieldType NodeType::* field_ptr;  // Pointer-to-member for type-safe field access
  const char* name;                 // Field name for debugging

  /**
   * @brief Construct a field descriptor
   *
   * @param ptr Pointer-to-member for the field
   * @param n Field name (string literal)
   */
  constexpr FieldDescriptor(FieldType NodeType::* ptr, const char* n) : field_ptr(ptr), name(n) {}

  /**
   * @brief Access field value from a node instance
   *
   * @param node The node instance to access the field from
   * @return const reference to the field value
   */
  const FieldType& Get(const NodeType& node) const { return node.*field_ptr; }
};

/**
 * @brief Factory function for DEF fields
 *
 * DEF fields are binding variables (e.g., loop variables) that can participate
 * in auto-mapping during structural comparison.
 *
 * @tparam NodeType The IR node class
 * @tparam FieldType The field type
 * @param ptr Pointer-to-member for the field
 * @param name Field name for debugging
 * @return Field descriptor with DefFieldTag
 */
template <typename NodeType, typename FieldType>
constexpr auto DefField(FieldType NodeType::* ptr, const char* name) {
  return FieldDescriptor<NodeType, FieldType, DefFieldTag>{ptr, name};
}

/**
 * @brief Factory function for USUAL fields
 *
 * USUAL fields are normal fields that should be visited and compared
 * during structural hash/equality operations.
 *
 * @tparam NodeType The IR node class
 * @tparam FieldType The field type
 * @param ptr Pointer-to-member for the field
 * @param name Field name for debugging
 * @return Field descriptor with UsualFieldTag
 */
template <typename NodeType, typename FieldType>
constexpr auto UsualField(FieldType NodeType::* ptr, const char* name) {
  return FieldDescriptor<NodeType, FieldType, UsualFieldTag>{ptr, name};
}

/**
 * @brief Factory function for IGNORE fields
 *
 * IGNORE fields are skipped during structural comparisons.
 * Typically used for source locations (Span) and other metadata.
 *
 * @tparam NodeType The IR node class
 * @tparam FieldType The field type
 * @param ptr Pointer-to-member for the field
 * @param name Field name for debugging
 * @return Field descriptor with IgnoreFieldTag
 */
template <typename NodeType, typename FieldType>
constexpr auto IgnoreField(FieldType NodeType::* ptr, const char* name) {
  return FieldDescriptor<NodeType, FieldType, IgnoreFieldTag>{ptr, name};
}

}  // namespace reflection
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_REFLECTION_FIELD_TRAITS_H_

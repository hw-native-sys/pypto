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

#ifndef PYPTO_IR_CORE_H_
#define PYPTO_IR_CORE_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "pypto/ir/pipe.h"
#include "pypto/ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

/**
 * @brief Source location information for IR nodes
 *
 * Tracks the exact position in source code where an IR node originated.
 * Immutable value type - all fields are const.
 */
class Span {
 public:
  const std::string filename_;  ///< Source filename
  const int begin_line_;        ///< Beginning line number (1-indexed)
  const int begin_column_;      ///< Beginning column number (1-indexed)
  const int end_line_;          ///< Ending line number (1-indexed), -1 means unknown
  const int end_column_;        ///< Ending column number (1-indexed), -1 means unknown

  /**
   * @brief Construct a source span
   *
   * @param file Source filename
   * @param bl Begin line (1-indexed)
   * @param bc Begin column (1-indexed)
   * @param el End line (1-indexed)
   * @param ec End column (1-indexed)
   */
  Span(std::string file, int begin_line, int begin_column, int end_line = -1, int end_column = -1);

  /**
   * @brief Convert span to string representation
   *
   * @return String in format "filename:begin_line:begin_column"
   */
  [[nodiscard]] std::string to_string() const;

  /**
   * @brief Check if the span is valid (has valid line/column numbers)
   *
   * @return true if all line/column numbers are positive
   */
  [[nodiscard]] bool is_valid() const;

  /**
   * @brief Create an unknown/invalid span
   *
   * @return Span with empty filename and invalid coordinates
   */
  static Span unknown();
};

/**
 * @brief Base class for all IR nodes
 *
 * Abstract base providing common functionality for all IR nodes.
 * All IR nodes are immutable - once constructed, they cannot be modified.
 */
class IRNode {
 public:
  explicit IRNode(Span s) : span_(std::move(s)) {}
  virtual ~IRNode() = default;

  // Disable copying and moving to enforce immutability
  IRNode(IRNode&&) = delete;
  IRNode& operator=(IRNode&&) = delete;

  /**
   * @brief Get the type name of this IR node
   *
   * @return Human-readable type name (e.g., "Expr", "Stmt", "Var")
   */
  [[nodiscard]] virtual std::string TypeName() const { return "IRNode"; }

  Span span_;  // Source location

  static constexpr auto GetFieldDescriptors() {
    return std::make_tuple(reflection::IgnoreField(&IRNode::span_, "span"));
  }
};
using IRNodePtr = std::shared_ptr<const IRNode>;

/**
 * @brief Reference equality operator for IRNodePtr
 *
 * Compares two expression pointers by their address (reference equality).
 * Two IRNodePtr are equal only if they point to the same object.
 *
 * @param lhs Left-hand side expression pointer
 * @param rhs Right-hand side expression pointer
 * @return true if pointers reference the same object
 */
inline bool operator==(const IRNodePtr& lhs, const IRNodePtr& rhs) { return lhs.get() == rhs.get(); }

/**
 * @brief Reference inequality operator for IRNodePtr
 *
 * @param lhs Left-hand side expression pointer
 * @param rhs Right-hand side expression pointer
 * @return true if pointers reference different objects
 */
inline bool operator!=(const IRNodePtr& lhs, const IRNodePtr& rhs) { return !(lhs == rhs); }

}  // namespace ir
}  // namespace pypto

// std::hash specialization for IRNodePtr (reference-based hash)
namespace std {
/**
 * @brief Hash specialization for IRNodePtr
 *
 * Computes hash based on pointer address (reference hash).
 * Enables use of IRNodePtr in std::unordered_map and std::unordered_set
 * with reference equality semantics.
 *
 * Usage:
 * @code
 * std::unordered_map<pypto::ir::IRNodePtr, int> my_map;
 * @endcode
 */
template <>
struct hash<pypto::ir::IRNodePtr> {
  size_t operator()(const pypto::ir::IRNodePtr& ptr) const noexcept {
    return std::hash<const pypto::ir::IRNode*>{}(ptr.get());
  }
};

}  // namespace std

#endif  // PYPTO_IR_CORE_H_

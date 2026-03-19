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

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {

/**
 * @brief Kind enumeration for all IR node types
 *
 * Used for efficient type checking and casting without RTTI overhead.
 * Each concrete IR node class has a unique Kind value.
 */
enum class ObjectKind {
  // Base kinds (abstract base classes)
  IRNode,
  Expr,
  Stmt,
  Type,

  // Expression kinds
  Var,
  IterArg,
  MemRef,
  Call,
  MakeTuple,
  TupleGetItemExpr,
  ConstInt,
  ConstFloat,
  ConstBool,

  // Binary expression kinds
  Add,
  Sub,
  Mul,
  FloorDiv,
  FloorMod,
  FloatDiv,
  Min,
  Max,
  Pow,
  Eq,
  Ne,
  Lt,
  Le,
  Gt,
  Ge,
  And,
  Or,
  Xor,
  BitAnd,
  BitOr,
  BitXor,
  BitShiftLeft,
  BitShiftRight,

  // Unary expression kinds
  Abs,
  Neg,
  Not,
  BitNot,
  Cast,

  // Statement kinds
  AssignStmt,
  IfStmt,
  YieldStmt,
  ReturnStmt,
  ForStmt,
  WhileStmt,
  ScopeStmt,
  SeqStmts,
  OpStmts,
  EvalStmt,
  BreakStmt,
  ContinueStmt,

  // Type kinds
  UnknownType,
  MemRefType,
  ScalarType,
  ShapedType,
  TensorType,
  TileType,
  TupleType,

  // Other IR node kinds
  Function,
  Program,

  // Op kinds
  Op,
  GlobalVar
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
   * @brief Get the Kind of this IR node
   *
   * @return The ObjectKind enum value identifying the concrete type
   */
  [[nodiscard]] virtual ObjectKind GetKind() const = 0;

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

// Forward declarations for KindTrait specializations
// (Actual specializations will be added after the concrete types are defined)
template <typename T>
struct KindTrait;

// SFINAE helpers to detect trait type
namespace detail {
// Detect if KindTrait<T> has a 'kind' member (concrete type)
template <typename T, typename = void>
struct HasSingleKind : std::false_type {};

template <typename T>
struct HasSingleKind<T, std::void_t<decltype(KindTrait<T>::kind)>> : std::true_type {};

// Detect if KindTrait<T> has a 'kinds' member (base class)
template <typename T, typename = void>
struct HasKindArray : std::false_type {};

template <typename T>
struct HasKindArray<T, std::void_t<decltype(KindTrait<T>::kinds)>> : std::true_type {};

// Check if kind is in array (compile-time)
template <typename T>
constexpr bool IsKindInArray(ObjectKind kind) {
  for (size_t i = 0; i < KindTrait<T>::count; ++i) {
    if (KindTrait<T>::kinds[i] == kind) {
      return true;
    }
  }
  return false;
}
}  // namespace detail

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

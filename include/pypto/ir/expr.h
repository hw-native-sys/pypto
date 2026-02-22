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

#ifndef PYPTO_IR_EXPR_H_
#define PYPTO_IR_EXPR_H_

#include <any>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/core.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for all expressions in the IR
 *
 * This is the root base class for all expression types (scalar, tensor, etc).
 * Expressions represent computations that produce values.
 * All expressions are immutable.
 */
class Expr : public IRNode {
 protected:
  TypePtr type_;  // Type of the expression result

 public:
  /**
   * @brief Create an expression
   *
   * @param span Source location
   * @param type Type of the expression result (defaults to UnknownType)
   */
  explicit Expr(Span s, TypePtr type = GetUnknownType()) : IRNode(std::move(s)), type_(std::move(type)) {}
  ~Expr() override = default;

  /**
   * @brief Get the type name of this expression
   *
   * @return Human-readable type name (e.g., "ScalarExpr", "Var", "Call")
   */
  [[nodiscard]] std::string TypeName() const override { return "Expr"; }

  /**
   * @brief Get the type of this expression
   *
   * @return Type pointer of the expression result
   */
  [[nodiscard]] const TypePtr& GetType() const { return type_; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(IRNode::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&Expr::type_, "type")));
  }
};

using ExprPtr = std::shared_ptr<const Expr>;

// Forward declaration for MemorySpace enum (defined in memref.h)
enum class MemorySpace;

/**
 * @brief Base class for operations/functions
 *
 * Represents callable operations in the IR.
 * Stores the schema of allowed kwargs (key -> expected type mapping).
 * Actual kwarg values are stored per-Call instance in Call::kwargs_.
 */
class Op {
 public:
  std::string name_;

  explicit Op(std::string name) : name_(std::move(name)) {}
  virtual ~Op() = default;

  /**
   * @brief Register an allowed kwarg with its expected type
   *
   * Defines that this operator accepts a kwarg with the given key and type.
   * This is used for validation when creating Call expressions.
   *
   * Only specific types are allowed: bool, int, std::string, double, DataType, MemorySpace
   * This is enforced at compile-time via static_assert.
   *
   * @tparam T Expected type of the kwarg value (must be one of the allowed types)
   * @param key Kwarg key (string identifier)
   */
  template <typename T>
  void SetAttrType(const std::string& key) const {
    // Compile-time check: only allow specific types
    static_assert(std::is_same_v<T, bool> || std::is_same_v<T, int> || std::is_same_v<T, std::string> ||
                      std::is_same_v<T, double> || std::is_same_v<T, DataType> ||
                      std::is_same_v<T, MemorySpace>,
                  "SetAttrType only accepts: bool, int, std::string, double, DataType, MemorySpace");

    attrs_.emplace(key, std::type_index(typeid(T)));
  }

  /**
   * @brief Get the expected type for a kwarg
   *
   * @param key Kwarg key
   * @return type_index of the expected type
   * @throws pypto::ValueError if kwarg is not registered
   */
  [[nodiscard]] std::type_index GetAttrType(const std::string& key) const {
    auto it = attrs_.find(key);
    if (it == attrs_.end()) {
      throw pypto::ValueError("Attribute '" + key + "' not found in operator '" + name_ + "'");
    }
    return it->second;
  }

  /**
   * @brief Check if a kwarg is registered
   *
   * @param key Kwarg key
   * @return true if the kwarg is registered
   */
  [[nodiscard]] bool HasAttr(const std::string& key) const { return attrs_.find(key) != attrs_.end(); }

  /**
   * @brief Get all registered kwarg keys
   *
   * @return Vector of all kwarg keys
   */
  [[nodiscard]] std::vector<std::string> GetAttrKeys() const {
    std::vector<std::string> keys;
    keys.reserve(attrs_.size());
    for (const auto& pair : attrs_) {
      keys.push_back(pair.first);
    }
    return keys;
  }

  /**
   * @brief Get all registered kwargs as a map
   *
   * @return Map of kwarg keys to expected types
   */
  [[nodiscard]] const std::unordered_map<std::string, std::type_index>& GetAttrs() const { return attrs_; }

  /**
   * @brief Set the pipeline type for this operator
   *
   * @param pipe Pipeline type (e.g., MTE2, V)
   */
  void SetPipe(PipeType pipe) const { pipe_ = pipe; }

  /**
   * @brief Get the pipeline type for this operator
   *
   * @return Optional pipeline type (nullopt if not set)
   */
  [[nodiscard]] std::optional<PipeType> GetPipe() const { return pipe_; }

  [[nodiscard]] virtual ObjectKind GetKind() const { return ObjectKind::Op; }
  [[nodiscard]] virtual std::string TypeName() const { return "Op"; }

 private:
  mutable std::unordered_map<std::string, std::type_index> attrs_;  ///< Kwarg schema (key -> type)
  mutable std::optional<PipeType> pipe_;                            ///< Pipeline type
};

using OpPtr = std::shared_ptr<const Op>;

/**
 * @brief Global variable reference for functions in a program
 *
 * Represents a reference to a function in the program's global scope.
 * Can be used as an operation in Call expressions to call functions within the same program.
 * The name of the GlobalVar should match the name of the function it references.
 */
class GlobalVar : public Op {
 public:
  explicit GlobalVar(std::string name) : Op(std::move(name)) {}
  ~GlobalVar() override = default;
  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::GlobalVar; }
  [[nodiscard]] std::string TypeName() const override { return "GlobalVar"; }
};

using GlobalVarPtr = std::shared_ptr<const GlobalVar>;

/**
 * @brief Custom comparator for ordering GlobalVarPtr by name
 *
 * Used in std::map to maintain deterministic ordering of functions in a Program.
 * Ensures consistent structural equality and hashing.
 */
struct GlobalVarPtrLess {
  bool operator()(const GlobalVarPtr& lhs, const GlobalVarPtr& rhs) const { return lhs->name_ < rhs->name_; }
};

/**
 * @brief Variable reference expression
 *
 * Represents a reference to a named variable.
 * Can represent both scalar and tensor variables based on its type.
 */
class Var : public Expr {
 public:
  std::string name_;

  /**
   * @brief Create a variable reference
   *
   * @param name Variable name
   * @param type Type of the variable (ScalarType, TensorType, or TileType)
   *             Memory reference information is stored in ShapedType for Tensor/Tile types
   * @param span Source location
   * @return Shared pointer to const Var expression
   */
  Var(std::string name, TypePtr type, Span span)
      : Expr(std::move(span), std::move(type)), name_(std::move(name)), unique_id_(next_unique_id_++) {}

  /**
   * @brief Get the unique identity of this variable
   *
   * Monotonically increasing ID assigned at construction, providing
   * deterministic identity that is stable across runs within a process.
   *
   * @return Process-unique identifier for this variable instance
   */
  [[nodiscard]] uint64_t UniqueId() const { return unique_id_; }

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::Var; }
  [[nodiscard]] std::string TypeName() const override { return "Var"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (name_ as IGNORE field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::IgnoreField(&Var::name_, "name")));
  }

 private:
  static inline std::atomic<uint64_t> next_unique_id_{0};
  uint64_t unique_id_;
};

using VarPtr = std::shared_ptr<const Var>;

/**
 * @brief Iteration argument variable
 *
 * Represents an iteration argument (loop-carried value) in for loops.
 * IterArgs implement SSA-style loop-carried dependencies where values are
 * carried from one iteration to the next via yield statements.
 *
 * **Scoping Rules:**
 * - IterArg variables are scoped to the loop body only
 * - Cannot be directly accessed outside the loop
 * - Must use return_vars to expose final values after the loop
 *
 * **Usage Pattern:**
 * 1. Create IterArg with initial value
 * 2. Use in ForStmt's iter_args list
 * 3. Update via YieldStmt in loop body
 * 4. Capture final value in ForStmt's return_vars
 *
 * @example
 * // for i, (sum,) in pl.range(0, n, 1, init_values=(0,)):
 * //     sum = pl.yield_(sum + i)
 * // sum_final = sum
 * auto sum_iter = std::make_shared<IterArg>("sum", type, init_val, span);
 * auto sum_final = std::make_shared<Var>("sum_final", type, span);
 * auto for_stmt = std::make_shared<ForStmt>(
 *     i, start, stop, step,
 *     std::vector{sum_iter},  // iter_args (loop-scoped)
 *     body,
 *     std::vector{sum_final}, // return_vars (accessible after loop)
 *     span
 * );
 */
class IterArg : public Var {
 public:
  ExprPtr initValue_;  // Initial value expression for first iteration

  /**
   * @brief Create an iteration argument
   *
   * @param name Variable name (scoped to loop body)
   * @param type Type of the variable (ScalarType, TensorType, or TileType)
   *             Memory reference information is stored in ShapedType for Tensor/Tile types
   * @param initValue Initial value expression for first iteration
   * @param span Source location
   */
  IterArg(std::string name, TypePtr type, ExprPtr initValue, Span span)
      : Var(std::move(name), std::move(type), std::move(span)), initValue_(std::move(initValue)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::IterArg; }
  [[nodiscard]] std::string TypeName() const override { return "IterArg"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (initValue_ as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Var::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&IterArg::initValue_, "initValue")));
  }
};

using IterArgPtr = std::shared_ptr<const IterArg>;

/**
 * @brief Function call expression
 *
 * Represents a function call with an operation and arguments.
 * Can accept any Expr as arguments, not just scalar expressions.
 * Supports keyword arguments (kwargs) for operator metadata.
 */
class Call : public Expr {
 public:
  OpPtr op_;                                              // Operation/function
  std::vector<ExprPtr> args_;                             // Positional arguments
  std::vector<std::pair<std::string, std::any>> kwargs_;  // Keyword arguments (metadata, ordered)

  /**
   * @brief Create a function call expression
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param span Source location
   */
  Call(OpPtr op, std::vector<ExprPtr> args, Span span)
      : Expr(std::move(span)), op_(std::move(op)), args_(std::move(args)), kwargs_() {}

  /**
   * @brief Create a function call expression with explicit type
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param type Result type of the call
   * @param span Source location
   */
  Call(OpPtr op, std::vector<ExprPtr> args, TypePtr type, Span span)
      : Expr(std::move(span), std::move(type)), op_(std::move(op)), args_(std::move(args)), kwargs_() {}

  /**
   * @brief Create a function call expression with kwargs
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param kwargs Keyword arguments (metadata)
   * @param span Source location
   */
  Call(OpPtr op, std::vector<ExprPtr> args, std::vector<std::pair<std::string, std::any>> kwargs, Span span)
      : Expr(std::move(span)), op_(std::move(op)), args_(std::move(args)), kwargs_(std::move(kwargs)) {}

  /**
   * @brief Create a function call expression with kwargs and explicit type
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param kwargs Keyword arguments (metadata)
   * @param type Result type of the call
   * @param span Source location
   */
  Call(OpPtr op, std::vector<ExprPtr> args, std::vector<std::pair<std::string, std::any>> kwargs,
       TypePtr type, Span span)
      : Expr(std::move(span), std::move(type)),
        op_(std::move(op)),
        args_(std::move(args)),
        kwargs_(std::move(kwargs)) {}

  /**
   * @brief Get a kwarg value with type checking
   *
   * @tparam T Type of the kwarg value
   * @param key Kwarg key
   * @param default_value Default value if key doesn't exist
   * @return The kwarg value or default
   */
  template <typename T>
  T GetKwarg(const std::string& key, const T& default_value = T{}) const {
    for (const auto& [k, v] : kwargs_) {
      if (k == key) {
        return AnyCast<T>(v, "kwarg key: " + key);
      }
    }
    return default_value;
  }

  /**
   * @brief Check if a kwarg exists
   *
   * @param key Kwarg key
   * @return true if the kwarg exists
   */
  [[nodiscard]] bool HasKwarg(const std::string& key) const {
    for (const auto& [k, v] : kwargs_) {
      if (k == key) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::Call; }
  [[nodiscard]] std::string TypeName() const override { return "Call"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (op, args, and kwargs as USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&Call::op_, "op"),
                                          reflection::UsualField(&Call::args_, "args"),
                                          reflection::UsualField(&Call::kwargs_, "kwargs")));
  }
};

using CallPtr = std::shared_ptr<const Call>;

/**
 * @brief Expression to create a tuple from multiple expressions
 *
 * Takes a list of expressions and creates a tuple value.
 * The result type is TupleType containing the types of all input expressions.
 */
class MakeTuple : public Expr {
 public:
  std::vector<ExprPtr> elements_;  // Elements of the tuple

  /**
   * @brief Create a tuple construction expression
   *
   * @param elements Expressions to be tuple elements
   * @param span Source location
   */
  MakeTuple(std::vector<ExprPtr> elements, Span span);

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::MakeTuple; }
  [[nodiscard]] std::string TypeName() const override { return "MakeTuple"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&MakeTuple::elements_, "elements")));
  }
};

using MakeTuplePtr = std::shared_ptr<const MakeTuple>;

/**
 * @brief Tuple element access expression
 *
 * Represents accessing an element from a tuple by index.
 * The tuple must have TupleType and index must be a compile-time constant.
 */
class TupleGetItemExpr : public Expr {
 public:
  ExprPtr tuple_;  // Tuple expression (must have TupleType)
  int index_;      // Index of the element to access (0-based)

  /**
   * @brief Create a tuple element access expression
   *
   * @param tuple Tuple expression (must have TupleType)
   * @param index Index of the element (0-based, must be within bounds)
   * @param span Source location
   */
  TupleGetItemExpr(ExprPtr tuple, int index, Span span);

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TupleGetItemExpr; }
  [[nodiscard]] std::string TypeName() const override { return "TupleGetItemExpr"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&TupleGetItemExpr::tuple_, "tuple"),
                                          reflection::UsualField(&TupleGetItemExpr::index_, "index")));
  }
};

using TupleGetItemExprPtr = std::shared_ptr<const TupleGetItemExpr>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_EXPR_H_

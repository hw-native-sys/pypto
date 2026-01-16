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
#include <memory>
#include <string>
#include <tuple>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/core.h"
#include "pypto/ir/reflection/field_traits.h"
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

/**
 * @brief Base class for operations/functions
 *
 * Represents callable operations in the IR.
 * Supports storing arbitrary typed attributes for operator metadata.
 */
class Op {
 public:
  std::string name_;

  explicit Op(std::string name) : name_(std::move(name)) {}
  virtual ~Op() = default;

  /**
   * @brief Set an attribute with a typed value
   *
   * Stores an attribute with the given key and value. The value is stored
   * as std::any and can be retrieved later with the same type.
   *
   * @tparam T Type of the attribute value
   * @param key Attribute key (string identifier)
   * @param value Attribute value
   */
  template <typename T>
  void SetAttr(const std::string& key, T value) const {
    attrs_[key] = std::any(std::move(value));
  }

  /**
   * @brief Get an attribute value
   *
   * Retrieves an attribute value with type checking. Throws std::bad_any_cast
   * if the type doesn't match the stored type.
   *
   * @tparam T Expected type of the attribute value
   * @param key Attribute key
   * @return The attribute value
   * @throws pypto::ValueError if attribute doesn't exist
   * @throws pypto::TypeError if type doesn't match
   */
  template <typename T>
  T GetAttr(const std::string& key) const {
    auto it = attrs_.find(key);
    if (it == attrs_.end()) {
      throw pypto::ValueError("Attribute '" + key + "' not found in operator '" + name_ + "'");
    }
    if (it->second.type() != typeid(T)) {
      throw pypto::TypeError("Attribute '" + key + "' in operator '" + name_ + "' has incompatible type");
    }
    return std::any_cast<T>(it->second);
  }

  /**
   * @brief Get raw attribute storage
   *
   * Retrieves the stored std::any for an attribute. Throws pypto::ValueError
   * if the attribute doesn't exist.
   *
   * @param key Attribute key
   * @return const std::any& stored attribute value
   */
  const std::any& GetAttrAny(const std::string& key) const {
    auto it = attrs_.find(key);
    if (it == attrs_.end()) {
      throw pypto::ValueError("Attribute '" + key + "' not found in operator '" + name_ + "'");
    }
    return it->second;
  }

  /**
   * @brief Check if an attribute exists
   *
   * @param key Attribute key
   * @return true if the attribute exists
   */
  bool HasAttr(const std::string& key) const { return attrs_.find(key) != attrs_.end(); }

  /**
   * @brief Get all attribute keys
   *
   * @return Vector of all attribute keys
   */
  std::vector<std::string> GetAttrKeys() const {
    std::vector<std::string> keys;
    keys.reserve(attrs_.size());
    for (const auto& pair : attrs_) {
      keys.push_back(pair.first);
    }
    return keys;
  }

 private:
  mutable std::unordered_map<std::string, std::any> attrs_;  ///< Attribute storage (mutable for metadata)
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
   * @param type Type of the variable (ScalarType or TensorType)
   * @param span Source location
   * @return Shared pointer to const Var expression
   */
  Var(std::string name, TypePtr type, Span span)
      : Expr(std::move(span), std::move(type)), name_(std::move(name)) {}

  [[nodiscard]] std::string TypeName() const override { return "Var"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (name_ as USUAL field, type_ is in Expr)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::IgnoreField(&Var::name_, "name")));
  }
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
 * // for i, (sum,) in pi.range(0, n, 1, init_values=[0]):
 * //     sum = pi.yield(sum + i)
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
   * @param type Type of the variable (ScalarType or TensorType)
   * @param initValue Initial value expression for first iteration
   * @param span Source location
   */
  IterArg(std::string name, TypePtr type, ExprPtr initValue, Span span)
      : Var(std::move(name), std::move(type), std::move(span)), initValue_(std::move(initValue)) {}

  [[nodiscard]] std::string TypeName() const override { return "IterArg"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (initValue_ and value_ as USUAL fields)
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
 */
class Call : public Expr {
 public:
  OpPtr op_;                   // Operation/function
  std::vector<ExprPtr> args_;  // Arguments

  /**
   * @brief Create a function call expression
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param span Source location
   */
  Call(OpPtr op, std::vector<ExprPtr> args, Span span)
      : Expr(std::move(span)), op_(std::move(op)), args_(std::move(args)) {}

  /**
   * @brief Create a function call expression with explicit type
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param type Result type of the call
   * @param span Source location
   */
  Call(OpPtr op, std::vector<ExprPtr> args, TypePtr type, Span span)
      : Expr(std::move(span), std::move(type)), op_(std::move(op)), args_(std::move(args)) {}

  [[nodiscard]] std::string TypeName() const override { return "Call"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (op and args as USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&Call::op_, "op"),
                                          reflection::UsualField(&Call::args_, "args")));
  }
};

using CallPtr = std::shared_ptr<const Call>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_EXPR_H_

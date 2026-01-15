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
#include <unordered_map>
#include <utility>
#include <vector>

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
    attrs_[key] = std::make_any<T>(std::move(value));
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
   * @throws std::runtime_error if attribute doesn't exist
   * @throws std::bad_any_cast if type doesn't match
   */
  template <typename T>
  T GetAttr(const std::string& key) const {
    auto it = attrs_.find(key);
    if (it == attrs_.end()) {
      throw std::runtime_error("Attribute '" + key + "' not found in operator '" + name_ + "'");
    }
    return std::any_cast<T>(it->second);
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

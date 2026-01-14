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

/**
 * @file op_registry.h
 * @brief Operator registration system for PyPTO IR
 *
 * This file provides a modern C++ template-based operator registration system
 * that enables compile-time type checking and automatic type deduction for
 * tensor, tile, and scalar operations.
 */

#ifndef PYPTO_IR_OP_REGISTRY_H_
#define PYPTO_IR_OP_REGISTRY_H_

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/common.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Type-erased operator registration entry
 *
 * This class represents a registered operator in the registry system. It stores
 * metadata about the operator including its name, description, expected arguments,
 * validation logic, and type deduction function. The entry provides a fluent
 * interface for configuring operator properties during registration.
 *
 * Example usage:
 * @code
 * OpRegistryEntry entry;
 * entry.set_name("tensor.add")
 *      .set_description("Element-wise addition of two tensors")
 *      .add_argument("lhs", "Left-hand side tensor")
 *      .add_argument("rhs", "Right-hand side tensor")
 *      .f_deduce_type([](const std::vector<ExprPtr>& args) {
 *          return args[0]->GetType();
 *      });
 * @endcode
 */
class OpRegistryEntry {
 public:
  /**
   * @brief Get the operator instance
   *
   * Validates that the operator is properly configured with all required fields
   * before returning the operator instance. This ensures that operators cannot
   * be used until they are fully defined.
   *
   * Required fields:
   * - name: Set automatically during registration
   * - description: Must be set via set_description()
   * - op_category: Must be set via set_op_category()
   * - arguments: Must be set via add_argument() or no_argument()
   * - deduce_type: Must be set via f_deduce_type()
   *
   * @return Const reference to the operator pointer
   * @throws ValueError if any required field is not set
   */
  [[nodiscard]] inline const OpPtr& GetOp() const {
    // Check operator instance
    CHECK(op_) << "Operator '" + name_ + "' has no operator instance";

    // Check description is set
    CHECK(description_.has_value()) << "Operator '" + name_ +
                                           "' has no description. Use .set_description() to provide one.";

    // Check op_category is set
    CHECK(op_category_.has_value()) << "Operator '" + name_ +
                                           "' has no category. Use .set_op_category() to provide one.";

    // Check arguments are defined (either with arguments or marked as no_argument)
    CHECK(arguments_.has_value())
        << "Operator '" + name_ +
               "' has no argument definition. Use .add_argument() or .no_argument() to define arguments.";

    // Check deduce_type is set
    CHECK(deduce_type_.has_value())
        << "Operator '" + name_ + "' has no type deduction function. Use .f_deduce_type() to provide one.";

    return op_;
  }

  /**
   * @brief Get the operator name
   *
   * @return Const reference to the operator name
   */
  [[nodiscard]] inline const std::string& GetName() const { return name_; }

  /**
   * @brief Get the operator description
   *
   * @return Const reference to the operator description
   * @throws ValueError if description is not set
   */
  [[nodiscard]] inline const std::string& GetDescription() const {
    CHECK(description_.has_value()) << "Operator '" + name_ + "' has no description";
    return *description_;
  }

  /**
   * @brief Get the operator category
   *
   * @return Const reference to the operator category (e.g., "TensorOp", "TileOp", "ScalarOp")
   * @throws ValueError if category is not set
   */
  [[nodiscard]] inline const std::string& GetOpCategory() const {
    CHECK(op_category_.has_value()) << "Operator '" + name_ + "' has no category";
    return *op_category_;
  }

  /**
   * @brief Get the type deduction function
   *
   * Validates that the type deduction function is properly registered.
   *
   * @return Const reference to the type deduction function
   * @throws ValueError if the type deduction function is not set
   */
  [[nodiscard]] inline const std::function<TypePtr(const std::vector<ExprPtr>&)>& GetDeduceType() const {
    CHECK(deduce_type_.has_value()) << "Operator '" + name_ + "' has no type deduction function";
    return *deduce_type_;
  }

  /**
   * @brief Set the operator description
   *
   * Provides human-readable documentation for the operator. Should describe
   * what the operator does, its semantics, and any important constraints.
   *
   * @param description Human-readable description of the operator
   * @return Reference to this entry for method chaining
   */
  inline OpRegistryEntry& set_description(std::string description) {
    CHECK(!description_.has_value()) << "Operator '" + name_ + "' description is already set";
    description_ = std::move(description);
    return *this;
  }

  /**
   * @brief Set the operator category
   *
   * Specifies the category of the operator (e.g., "TensorOp", "TileOp", "ScalarOp").
   * This is used for categorization and type checking without requiring specific type details.
   *
   * @param category Operator category (e.g., "TensorOp", "TileOp", "ScalarOp")
   * @return Reference to this entry for method chaining
   */
  inline OpRegistryEntry& set_op_category(std::string category) {
    CHECK(!op_category_.has_value()) << "Operator '" + name_ + "' category is already set";
    op_category_ = std::move(category);
    return *this;
  }

  /**
   * @brief Add an argument specification
   *
   * Documents an expected argument with its name, type, and description.
   * Arguments should be added in the order they appear in the operator's
   * argument list.
   *
   * @param name Argument name (for documentation)
   * @param type Expected type of the argument (nullptr for any type)
   * @param description Description of the argument's purpose
   * @return Reference to this entry for method chaining
   */
  inline OpRegistryEntry& add_argument(std::string name, std::string description) {
    // Initialize the vector if not already initialized
    if (!arguments_.has_value()) {
      arguments_ = std::vector<std::pair<std::string, std::string>>();
    }
    arguments_->emplace_back(std::move(name), std::move(description));
    return *this;
  }

  /**
   * @brief Mark the operator as having no arguments
   *
   * This method must be called explicitly for operators that take no arguments
   * to distinguish from operators where arguments were simply not defined.
   *
   * @return Reference to this entry for method chaining
   */
  inline OpRegistryEntry& no_argument() {
    CHECK(!arguments_.has_value()) << "Operator '" + name_ +
                                          "' already has arguments defined. Cannot call no_argument() after "
                                          "add_argument().";
    arguments_ = std::vector<std::pair<std::string, std::string>>();
    return *this;
  }

  /**
   * @brief Set the type deduction function
   *
   * Provides a function that computes the result type of the operator given
   * its arguments. This is called during operator creation to determine the
   * type of the resulting Call expression.
   *
   * The function should:
   * - Validate that argument types are compatible
   * - Compute and return the result type
   * - Throw std::invalid_argument if types are incompatible
   *
   * @param dt Function that takes arguments and returns the deduced result type
   * @return Reference to this entry for method chaining
   */
  inline OpRegistryEntry& f_deduce_type(std::function<TypePtr(const std::vector<ExprPtr>&)> dt) {
    CHECK(!deduce_type_.has_value()) << "Operator '" + name_ + "' type deduction function is already set";
    deduce_type_ = std::move(dt);
    return *this;
  }

 private:
  /**
   * @brief Set the operator name
   *
   * The name is used as the unique identifier for the operator in the registry.
   * Convention: use dotted notation like "tensor.add" or "tile.matmul".
   *
   * @param name The operator name (e.g., "tensor.add", "tile.conv2d")
   * @return Reference to this entry for method chaining
   */
  inline OpRegistryEntry& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }
  friend class OpRegistry;

  OpPtr op_;                                ///< Operator instance
  std::string name_;                        ///< Operator name (unique identifier)
  std::optional<std::string> description_;  ///< Human-readable description
  std::optional<std::string> op_category_;  ///< Operator category (e.g., "TensorOp", "TileOp", "ScalarOp")
  std::optional<std::vector<std::pair<std::string, std::string>>>
      arguments_;  ///< Argument specifications (name, description)
  std::optional<std::function<TypePtr(const std::vector<ExprPtr>&)>>
      deduce_type_;  ///< Type deduction function
};

/**
 * @brief Global operator registry (singleton)
 *
 * Manages registration and creation of operators with automatic type deduction.
 * Uses template metaprogramming to provide compile-time type safety while
 * supporting runtime operator lookup by name.
 *
 * Thread-safety: The registry is not thread-safe during registration.
 * Register all operators during initialization before concurrent access.
 */
class OpRegistry {
 public:
  // Disable copy and move
  OpRegistry(const OpRegistry&) = delete;
  OpRegistry& operator=(const OpRegistry&) = delete;
  OpRegistry(OpRegistry&&) = delete;
  OpRegistry& operator=(OpRegistry&&) = delete;

  /**
   * @brief Get the singleton instance
   *
   * @return Reference to the global operator registry
   */
  static OpRegistry& GetInstance();

  /**
   * @brief Register an operator by name
   *
   * Creates a new operator registry entry that can be configured using
   * the fluent API (set_description, add_argument, f_deduce_type, etc.).
   *
   * @param op_name Name of the operator (e.g., "tensor.add", "tile.mul")
   * @throws std::runtime_error if operator is already registered
   */
  OpRegistryEntry& Register(const std::string& op_name);

  /**
   * @brief Create a Call expression for a registered operator
   *
   * Looks up the operator by name, validates arguments, deduces the result type,
   * and creates a Call expression with proper typing.
   *
   * @param op_name Name of the operator to call
   * @param args Arguments to pass to the operator
   * @param span Source location information
   * @return Shared pointer to Call expression with deduced type
   * @throws std::runtime_error if operator not found or argument count invalid
   */
  CallPtr Create(const std::string& op_name, const std::vector<ExprPtr>& args, Span span) const;

  /**
   * @brief Check if an operator is registered
   *
   * @param op_name Name of the operator
   * @return true if the operator is registered
   */
  bool IsRegistered(const std::string& op_name) const { return registry_.find(op_name) != registry_.end(); }

  /**
   * @brief Get the operator instance by name
   *
   * @param op_name Name of the operator
   * @return Shared pointer to the operator instance
   * @throws std::runtime_error if operator not found
   */
  OpPtr GetOp(const std::string& op_name) const;

 private:
  OpRegistry() = default;
  ~OpRegistry() = default;

  std::unordered_map<std::string, OpRegistryEntry> registry_;
};

/**
 * @brief Helper macro for operator registration
 *
 * Use this macro to register operators in initialization code:
 * @code
 * REGISTER_OP("TensorAdd");
 * REGISTER_OP("TileAdd");
 * @endcode
 */
#define REGISTER_OP(OpName)                                                                           \
  static PYPTO_STR_CONCAT(PYPTO_UNUSED ::pypto::ir::OpRegistryEntry& OpRegistryEntry_, __COUNTER__) = \
      ::pypto::ir::OpRegistry::GetInstance().Register(OpName)

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_OP_REGISTRY_H_

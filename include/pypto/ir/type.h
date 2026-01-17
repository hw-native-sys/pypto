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

#ifndef PYPTO_IR_TYPE_H_
#define PYPTO_IR_TYPE_H_

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

// Forward declaration
class Expr;
using ExprPtr = std::shared_ptr<const Expr>;

/**
 * @brief Base class for type representations in the IR
 *
 * Types represent the structure and properties of values in the IR.
 * All types are immutable.
 */
class Type {
 public:
  virtual ~Type() = default;

  /**
   * @brief Get the type name of this type
   *
   * @return Human-readable type name (e.g., "ScalarType", "TensorType")
   */
  [[nodiscard]] virtual std::string TypeName() const { return "Type"; }

  static constexpr auto GetFieldDescriptors() { return std::make_tuple(); }
};

using TypePtr = std::shared_ptr<const Type>;

/**
 * @brief Unknown type representation
 *
 * Represents an unknown or unspecified type.
 * Used as the default type for expressions when type information is not available.
 */
class UnknownType : public Type {
 public:
  /**
   * @brief Create an unknown type
   */
  UnknownType() = default;

  [[nodiscard]] std::string TypeName() const override { return "UnknownType"; }

  static constexpr auto GetFieldDescriptors() { return Type::GetFieldDescriptors(); }
};

using UnknownTypePtr = std::shared_ptr<const UnknownType>;

/**
 * @brief Get a shared pointer to the singleton UnknownType instance
 *
 * @return Shared pointer to UnknownType
 */
inline UnknownTypePtr GetUnknownType() {
  static const auto unknown_type = std::make_shared<UnknownType>();
  return unknown_type;
}

/**
 * @brief Scalar type representation
 *
 * Represents a scalar value type with a data type.
 */
class ScalarType : public Type {
 public:
  DataType dtype_;

  /**
   * @brief Create a scalar type
   *
   * @param dtype Data type
   */
  explicit ScalarType(DataType dtype) : dtype_(dtype) {}

  [[nodiscard]] std::string TypeName() const override { return "ScalarType"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Type::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&ScalarType::dtype_, "dtype")));
  }
};

using ScalarTypePtr = std::shared_ptr<const ScalarType>;

/**
 * @brief Tensor type representation
 *
 * Represents a tensor type with a data type and shape dimensions.
 */
class TensorType : public Type {
 public:
  DataType dtype_;              // Element data type
  std::vector<ExprPtr> shape_;  // Shape dimensions (symbolic or constant)

  /**
   * @brief Create a tensor type
   *
   * @param shape Shape dimensions
   * @param dtype Element data type
   */
  TensorType(std::vector<ExprPtr> shape, DataType dtype) : dtype_(dtype), shape_(std::move(shape)) {}

  [[nodiscard]] std::string TypeName() const override { return "TensorType"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Type::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&TensorType::dtype_, "dtype"),
                                          reflection::UsualField(&TensorType::shape_, "shape")));
  }
};

using TensorTypePtr = std::shared_ptr<const TensorType>;

/**
 * @brief Tile type representation
 *
 * Represents a tile type (2D tensor with at most 2 dimensions).
 * Tiles are used for hardware-optimized operations on 2D data structures.
 */
class TileType : public Type {
 public:
  DataType dtype_;              // Element data type
  std::vector<ExprPtr> shape_;  // Shape dimensions (at most 2 dimensions)

  /**
   * @brief Create a tile type
   *
   * @param shape Shape dimensions (must have at most 2 dimensions)
   * @param dtype Element data type
   * @throws std::invalid_argument if shape has more than 2 dimensions
   */
  TileType(std::vector<ExprPtr> shape, DataType dtype) : dtype_(dtype), shape_(std::move(shape)) {
    CHECK(shape_.size() <= 2) << "TileType can have at most 2 dimensions, got " << shape_.size();
  }

  [[nodiscard]] std::string TypeName() const override { return "TileType"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Type::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&TileType::dtype_, "dtype"),
                                          reflection::UsualField(&TileType::shape_, "shape")));
  }
};

using TileTypePtr = std::shared_ptr<const TileType>;

/**
 * @brief Tuple type representation
 *
 * Represents a tuple type containing multiple types.
 * Tuples are used for multiple return values and structured data.
 */
class TupleType : public Type {
 public:
  std::vector<TypePtr> types_;  // Types in the tuple

  /**
   * @brief Create a tuple type
   *
   * @param types List of types in the tuple
   */
  explicit TupleType(std::vector<TypePtr> types) : types_(std::move(types)) {}

  [[nodiscard]] std::string TypeName() const override { return "TupleType"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Type::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&TupleType::types_, "types")));
  }
};

using TupleTypePtr = std::shared_ptr<const TupleType>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TYPE_H_

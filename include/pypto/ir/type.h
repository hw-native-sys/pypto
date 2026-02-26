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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/core.h"
#include "pypto/ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

// Forward declarations
class Expr;
using ExprPtr = std::shared_ptr<const Expr>;

class MemRef;
using MemRefPtr = std::shared_ptr<const MemRef>;

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
   * @brief Get the Kind of this type
   *
   * @return The ObjectKind enum value identifying the concrete type
   */
  [[nodiscard]] virtual ObjectKind GetKind() const = 0;

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

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::UnknownType; }
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

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ScalarType; }
  [[nodiscard]] std::string TypeName() const override { return "ScalarType"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Type::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&ScalarType::dtype_, "dtype")));
  }
};

using ScalarTypePtr = std::shared_ptr<const ScalarType>;

/**
 * @brief Tensor layout enumeration
 *
 * Defines the available tensor layout types:
 * - ND: ND layout
 * - DN: DN layout
 * - NZ: NZ layout
 */
enum class TensorLayout {
  ND,  ///< ND layout
  DN,  ///< DN layout
  NZ   ///< NZ layout
};

/**
 * @brief Tensor view representation
 *
 * Represents the view information for a tensor, including stride and layout.
 * The shape is stored in TensorType itself, so TensorView only needs
 * stride and layout information.
 */
struct TensorView {
  std::vector<ExprPtr> stride;  ///< Stride for each dimension
  TensorLayout layout;          ///< Tensor layout type

  /**
   * @brief Default constructor for aggregate initialization
   */
  TensorView() : layout(TensorLayout::ND) {}

  /**
   * @brief Constructor with all parameters
   *
   * @param stride Stride for each dimension
   * @param layout Tensor layout type
   */
  TensorView(std::vector<ExprPtr> stride, TensorLayout layout) : stride(std::move(stride)), layout(layout) {}

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors
   */
  static constexpr auto GetFieldDescriptors() {
    return std::make_tuple(reflection::UsualField(&TensorView::stride, "stride"),
                           reflection::UsualField(&TensorView::layout, "layout"));
  }
};

/**
 * @brief Tile layout enumeration
 *
 * Shared by blayout and slayout fields in TileView:
 * - none_box: No layout constraint
 * - row_major: Row-major layout
 * - col_major: Column-major layout
 */
enum class TileLayout {
  none_box,   ///< No layout constraint
  row_major,  ///< Row-major layout
  col_major   ///< Column-major layout
};

/**
 * @brief Tile pad enumeration
 *
 * Defines the padding mode for out-of-bound tile accesses:
 * - null: No padding
 * - zero: Pad with zero
 * - max: Pad with maximum value of the element type
 * - min: Pad with minimum value of the element type
 */
enum class TilePad {
  null,  ///< No padding
  zero,  ///< Zero padding
  max,   ///< Max value padding
  min    ///< Min value padding
};

/**
 * @brief Tile view representation
 *
 * Represents the view information for a tile, including valid shape,
 * stride, start offset, block layout, scatter layout, fractal size,
 * and pad mode. This is used by TileType to track how a tile views
 * its underlying memory.
 */
struct TileView {
  std::vector<ExprPtr> valid_shape;            ///< Valid shape dimensions
  std::vector<ExprPtr> stride;                 ///< Stride for each dimension
  ExprPtr start_offset;                        ///< Starting offset
  TileLayout blayout = TileLayout::row_major;  ///< Block layout
  TileLayout slayout = TileLayout::none_box;   ///< Scatter layout
  uint64_t fractal = 512;                      ///< Fractal size
  TilePad pad = TilePad::null;                 ///< Pad mode

  /**
   * @brief Default constructor for aggregate initialization
   */
  TileView() = default;

  /**
   * @brief Constructor with all parameters
   *
   * @param valid_shape Valid shape dimensions
   * @param stride Stride for each dimension
   * @param start_offset Starting offset
   * @param blayout Block layout
   * @param slayout Scatter layout
   * @param fractal Fractal size
   * @param pad Pad mode
   */
  TileView(std::vector<ExprPtr> valid_shape, std::vector<ExprPtr> stride, ExprPtr start_offset,
           TileLayout blayout = TileLayout::row_major, TileLayout slayout = TileLayout::none_box,
           uint64_t fractal = 512, TilePad pad = TilePad::null)
      : valid_shape(std::move(valid_shape)),
        stride(std::move(stride)),
        start_offset(std::move(start_offset)),
        blayout(blayout),
        slayout(slayout),
        fractal(fractal),
        pad(pad) {}

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors
   */
  static constexpr auto GetFieldDescriptors() {
    return std::make_tuple(reflection::UsualField(&TileView::valid_shape, "valid_shape"),
                           reflection::UsualField(&TileView::stride, "stride"),
                           reflection::UsualField(&TileView::start_offset, "start_offset"),
                           reflection::UsualField(&TileView::blayout, "blayout"),
                           reflection::UsualField(&TileView::slayout, "slayout"),
                           reflection::UsualField(&TileView::fractal, "fractal"),
                           reflection::UsualField(&TileView::pad, "pad"));
  }
};

/**
 * @brief Base class for shaped types (tensors and tiles)
 *
 * Represents types that have shape dimensions and optional memory references.
 * Both TensorType and TileType inherit from this class.
 */
class ShapedType : public Type {
 public:
  DataType dtype_;                   ///< Element data type
  std::vector<ExprPtr> shape_;       ///< Shape dimensions (symbolic or constant)
  std::optional<MemRefPtr> memref_;  ///< Optional memory reference (shared pointer)

  /**
   * @brief Create a shaped type without memory reference
   *
   * @param dtype Element data type
   * @param shape Shape dimensions
   */
  ShapedType(DataType dtype, std::vector<ExprPtr> shape)
      : dtype_(dtype), shape_(std::move(shape)), memref_(std::nullopt) {}

  /**
   * @brief Create a shaped type with constant shape
   *
   * @param dtype Element data type
   * @param shape Shape dimensions
   */
  ShapedType(DataType dtype, const std::vector<int64_t>& shape, std::optional<MemRefPtr> memref);

  /**
   * @brief Create a shaped type with memory reference (shared_ptr)
   *
   * @param dtype Element data type
   * @param shape Shape dimensions
   * @param memref Memory reference (shared pointer)
   */
  ShapedType(DataType dtype, std::vector<ExprPtr> shape, MemRefPtr memref)
      : dtype_(dtype), shape_(std::move(shape)), memref_(std::move(memref)) {}

  /**
   * @brief Create a shaped type with optional memory reference (shared_ptr)
   *
   * @param dtype Element data type
   * @param shape Shape dimensions
   * @param memref Optional memory reference (shared pointer)
   */
  ShapedType(DataType dtype, std::vector<ExprPtr> shape, std::optional<MemRefPtr> memref)
      : dtype_(dtype), shape_(std::move(shape)), memref_(std::move(memref)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ShapedType; }
  [[nodiscard]] std::string TypeName() const override { return "ShapedType"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Type::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&ShapedType::dtype_, "dtype"),
                                          reflection::UsualField(&ShapedType::shape_, "shape"),
                                          reflection::UsualField(&ShapedType::memref_, "memref")));
  }
};

using ShapedTypePtr = std::shared_ptr<const ShapedType>;

/**
 * @brief Tensor type representation
 *
 * Represents a tensor type with a data type and shape dimensions.
 */
class TensorType : public ShapedType {
 public:
  std::optional<TensorView> tensor_view_;  ///< Optional tensor view information

  /**
   * @brief Create a tensor type without memory reference or tensor view
   *
   * @param shape Shape dimensions
   * @param dtype Element data type
   */
  TensorType(std::vector<ExprPtr> shape, DataType dtype)
      : ShapedType(dtype, std::move(shape)), tensor_view_(std::nullopt) {}

  /**
   * @brief Create a tensor type with memory reference (shared_ptr)
   *
   * @param shape Shape dimensions
   * @param dtype Element data type
   * @param memref Memory reference (shared pointer)
   */
  TensorType(std::vector<ExprPtr> shape, DataType dtype, MemRefPtr memref)
      : ShapedType(dtype, std::move(shape), std::move(memref)), tensor_view_(std::nullopt) {}

  /**
   * @brief Create a tensor type with constant shape
   *
   * @param shape Shape dimensions
   * @param dtype Element data type
   * @param memref Optional memory reference (shared pointer)
   */
  TensorType(const std::vector<int64_t>& shape, DataType dtype, std::optional<MemRefPtr> memref)
      : ShapedType(dtype, shape, std::move(memref)), tensor_view_(std::nullopt) {}

  /**
   * @brief Create a tensor type with optional memory reference (shared_ptr)
   *
   * @param shape Shape dimensions
   * @param dtype Element data type
   * @param memref Optional memory reference (shared pointer)
   */
  TensorType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref)
      : ShapedType(dtype, std::move(shape), std::move(memref)), tensor_view_(std::nullopt) {}

  /**
   * @brief Create a tensor type with optional memory reference and tensor view (shared_ptr)
   *
   * @param shape Shape dimensions
   * @param dtype Element data type
   * @param memref Optional memory reference (shared pointer)
   * @param tensor_view Tensor view information
   */
  TensorType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref,
             std::optional<TensorView> tensor_view)
      : ShapedType(dtype, std::move(shape), std::move(memref)), tensor_view_(std::move(tensor_view)) {}

  /**
   * @brief Create a tensor type with constant shape and tensor view
   *
   * @param shape Shape dimensions
   * @param dtype Element data type
   * @param memref Optional memory reference (shared pointer)
   * @param tensor_view Optional tensor view information
   */
  TensorType(const std::vector<int64_t>& shape, DataType dtype, std::optional<MemRefPtr> memref,
             std::optional<TensorView> tensor_view)
      : ShapedType(dtype, shape, std::move(memref)), tensor_view_(std::move(tensor_view)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TensorType; }
  [[nodiscard]] std::string TypeName() const override { return "TensorType"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(ShapedType::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&TensorType::tensor_view_, "tensor_view")));
  }
};

using TensorTypePtr = std::shared_ptr<const TensorType>;

/**
 * @brief Tile type representation
 *
 * Represents a tile type (multi-dimensional tensor).
 * Tiles are used for hardware-optimized operations on multi-dimensional data structures.
 * Note: Code generation currently only supports up to 2D tiles.
 */
class TileType : public ShapedType {
 public:
  std::optional<TileView> tile_view_;  ///< Optional tile view information

  /**
   * @brief Create a tile type without memory reference or tile view
   *
   * @param shape Shape dimensions (supports multi-dimensional tensors)
   * @param dtype Element data type
   */
  TileType(std::vector<ExprPtr> shape, DataType dtype)
      : ShapedType(dtype, std::move(shape)), tile_view_(std::nullopt) {}

  /**
   * @brief Create a tile type with memory reference (shared_ptr)
   *
   * @param shape Shape dimensions (supports multi-dimensional tensors)
   * @param dtype Element data type
   * @param memref Memory reference (shared pointer)
   */
  TileType(std::vector<ExprPtr> shape, DataType dtype, MemRefPtr memref)
      : ShapedType(dtype, std::move(shape), std::move(memref)), tile_view_(std::nullopt) {}

  /**
   * @brief Create a tile type with optional memory reference (shared_ptr)
   *
   * @param shape Shape dimensions (supports multi-dimensional tensors)
   * @param dtype Element data type
   * @param memref Optional memory reference (shared pointer)
   */
  TileType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref)
      : ShapedType(dtype, std::move(shape), std::move(memref)), tile_view_(std::nullopt) {
    // No dimension limit at type level; code generation may have constraints
  }

  /**
   * @brief Create a tile type with constant shape
   *
   * @param shape Shape dimensions (supports multi-dimensional tensors)
   * @param dtype Element data type
   * @param memref Optional memory reference (shared pointer)
   * @param tile_view Optional tile view information
   */
  TileType(const std::vector<int64_t>& shape, DataType dtype, std::optional<MemRefPtr> memref,
           std::optional<TileView> tile_view)
      : ShapedType(dtype, shape, std::move(memref)), tile_view_(std::move(tile_view)) {}

  /**
   * @brief Create a tile type with memory reference and tile view (shared_ptr)
   *
   * @param shape Shape dimensions (supports multi-dimensional tensors)
   * @param dtype Element data type
   * @param memref Memory reference (shared pointer)
   * @param tile_view Tile view information
   */
  TileType(std::vector<ExprPtr> shape, DataType dtype, MemRefPtr memref, std::optional<TileView> tile_view)
      : ShapedType(dtype, std::move(shape), std::move(memref)), tile_view_(std::move(tile_view)) {}

  /**
   * @brief Create a tile type with optional memory reference and tile view (shared_ptr)
   *
   * @param shape Shape dimensions (supports multi-dimensional tensors)
   * @param dtype Element data type
   * @param memref Optional memory reference (shared pointer)
   * @param tile_view Tile view information
   */
  TileType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref,
           std::optional<TileView> tile_view)
      : ShapedType(dtype, std::move(shape), std::move(memref)), tile_view_(std::move(tile_view)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TileType; }
  [[nodiscard]] std::string TypeName() const override { return "TileType"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(ShapedType::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&TileType::tile_view_, "tile_view")));
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

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TupleType; }
  [[nodiscard]] std::string TypeName() const override { return "TupleType"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Type::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&TupleType::types_, "types")));
  }
};

using TupleTypePtr = std::shared_ptr<const TupleType>;

/**
 * @brief Memory reference type representation
 *
 * Represents a memory reference type for shaped data (tensors and tiles).
 * Used as the type for MemRef variables.
 */
class MemRefType : public Type {
 public:
  /**
   * @brief Create a memory reference type
   */
  MemRefType() = default;

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::MemRefType; }
  [[nodiscard]] std::string TypeName() const override { return "MemRefType"; }

  static constexpr auto GetFieldDescriptors() { return Type::GetFieldDescriptors(); }
};

using MemRefTypePtr = std::shared_ptr<const MemRefType>;

/**
 * @brief Get a shared pointer to the singleton MemRefType instance
 *
 * @return Shared pointer to MemRefType
 */
inline MemRefTypePtr GetMemRefType() {
  static const auto memref_type = std::make_shared<MemRefType>();
  return memref_type;
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TYPE_H_

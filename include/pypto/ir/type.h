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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/core.h"
#include "pypto/ir/memory_space.h"
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
 * @brief Convert TensorLayout enum to string
 *
 * @param layout Tensor layout enum value
 * @return String representation
 */
std::string TensorLayoutToString(TensorLayout layout);

/**
 * @brief Convert string to TensorLayout enum
 *
 * @param str String representation of tensor layout
 * @return Corresponding TensorLayout enum value
 * @throws TypeError if the string does not match any known layout
 */
TensorLayout StringToTensorLayout(const std::string& str);

/**
 * @brief Pad mode enumeration (shared by TileView and TensorView)
 *
 * Defines the padding mode applied when a tile/tensor view access falls
 * outside `valid_shape` but still within the physical shape:
 * - null: No padding
 * - zero: Pad with zero
 * - max: Pad with maximum value of the element type
 * - min: Pad with minimum value of the element type
 */
enum class PadValue {
  null,  ///< No padding
  zero,  ///< Zero padding
  max,   ///< Max value padding
  min    ///< Min value padding
};

/**
 * @brief Tensor view representation
 *
 * Represents the view information for a tensor, including stride, layout,
 * valid_shape, and pad mode. The shape is stored in TensorType itself.
 */
struct TensorView {
  std::vector<ExprPtr> stride;  ///< Stride for each dimension
  TensorLayout layout;          ///< Tensor layout type
  std::vector<ExprPtr>
      valid_shape;                ///< Valid shape for each dimension (optional, empty means use full shape)
  PadValue pad = PadValue::null;  ///< Pad mode for accesses outside valid_shape but within shape

  /**
   * @brief Default constructor with ND layout and empty stride/valid_shape
   */
  TensorView() : layout(TensorLayout::ND) {}

  /**
   * @brief Constructor with all parameters
   *
   * @param stride Stride for each dimension
   * @param layout Tensor layout type
   * @param valid_shape Valid shape for each dimension (optional, defaults to empty)
   * @param pad Pad mode (optional, defaults to PadValue::null)
   */
  TensorView(std::vector<ExprPtr> stride, TensorLayout layout, std::vector<ExprPtr> valid_shape = {},
             PadValue pad = PadValue::null)
      : stride(std::move(stride)), layout(layout), valid_shape(std::move(valid_shape)), pad(pad) {}

  /**
   * @brief Constructor with integer stride and valid_shape (auto-converted to ConstInt)
   *
   * @param stride Stride for each dimension (int64, converted to ConstInt with INDEX dtype)
   * @param layout Tensor layout type
   * @param valid_shape Valid shape for each dimension (int64, defaults to empty)
   * @param pad Pad mode (optional, defaults to PadValue::null)
   */
  TensorView(const std::vector<int64_t>& stride, TensorLayout layout,
             const std::vector<int64_t>& valid_shape = {}, PadValue pad = PadValue::null);

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors
   */
  static constexpr auto GetFieldDescriptors() {
    return std::make_tuple(reflection::UsualField(&TensorView::stride, "stride"),
                           reflection::UsualField(&TensorView::layout, "layout"),
                           reflection::UsualField(&TensorView::valid_shape, "valid_shape"),
                           reflection::UsualField(&TensorView::pad, "pad"));
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
 * @brief Convert TileLayout enum to string
 */
std::string TileLayoutToString(TileLayout layout);

/**
 * @brief Convert string to TileLayout enum
 */
TileLayout StringToTileLayout(const std::string& str);

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
  PadValue pad = PadValue::null;               ///< Pad mode

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
           uint64_t fractal = 512, PadValue pad = PadValue::null)
      : valid_shape(std::move(valid_shape)),
        stride(std::move(stride)),
        start_offset(std::move(start_offset)),
        blayout(blayout),
        slayout(slayout),
        fractal(fractal),
        pad(pad) {}

  /**
   * @brief Constructor with integer valid_shape and stride (auto-converted to ConstInt)
   *
   * @param valid_shape Valid shape dimensions (int64, converted to ConstInt with INDEX dtype)
   * @param stride Stride for each dimension (int64, converted to ConstInt with INDEX dtype)
   * @param start_offset Starting offset
   * @param blayout Block layout
   * @param slayout Scatter layout
   * @param fractal Fractal size
   * @param pad Pad mode
   */
  TileView(const std::vector<int64_t>& valid_shape, const std::vector<int64_t>& stride, ExprPtr start_offset,
           TileLayout blayout = TileLayout::row_major, TileLayout slayout = TileLayout::none_box,
           uint64_t fractal = 512, PadValue pad = PadValue::null);

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

bool operator==(const TileView& lhs, const TileView& rhs);
bool operator!=(const TileView& lhs, const TileView& rhs);

/**
 * @brief Hash a TileView consistently with operator==
 *
 * For each ExprPtr field, mirrors AreExprsEqual's identity rule: ConstInt nodes
 * hash by integer value (matching the value carve-out), all other expressions
 * hash by pointer identity (matching shared_ptr equality). Other fields hash by
 * value. The contract is: lhs == rhs implies Hash(lhs) == Hash(rhs).
 */
size_t Hash(const TileView& tv);

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
  ShapedType(DataType dtype, std::vector<ExprPtr> shape);

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
  ShapedType(DataType dtype, std::vector<ExprPtr> shape, MemRefPtr memref);

  /**
   * @brief Create a shaped type with optional memory reference (shared_ptr)
   *
   * @param dtype Element data type
   * @param shape Shape dimensions
   * @param memref Optional memory reference (shared pointer)
   */
  ShapedType(DataType dtype, std::vector<ExprPtr> shape, std::optional<MemRefPtr> memref);

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ShapedType; }
  [[nodiscard]] std::string TypeName() const override { return "ShapedType"; }
  [[nodiscard]] virtual std::optional<MemorySpace> GetMemorySpace() const { return std::nullopt; }

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
  [[nodiscard]] std::optional<MemorySpace> GetMemorySpace() const override { return MemorySpace::DDR; }

  /// Returns true when this tensor uses DN (data-normal / transposed) layout.
  [[nodiscard]] bool IsDNLayout() const {
    return tensor_view_.has_value() && tensor_view_->layout == TensorLayout::DN;
  }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(ShapedType::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&TensorType::tensor_view_, "tensor_view")));
  }
};

using TensorTypePtr = std::shared_ptr<const TensorType>;

/**
 * @brief Distributed tensor type — a per-rank slice of a CommGroup HCCL window buffer.
 *
 * Subclass of :class:`TensorType` distinguished only by ``ObjectKind`` so that
 * verifiers can reject plain ``TensorType`` arguments to cross-rank ops
 * (``pld.tile.remote_load`` / ``pld.system.notify`` / ``pld.system.wait``).
 *
 * Note ``As<TensorType>`` does NOT match ``DistributedTensorType`` (precise
 * ObjectKind match). This is intentional — the cross-rank ops use
 * ``As<DistributedTensorType>`` to enforce that only window-bound tensors flow
 * through them.
 */
class DistributedTensorType : public TensorType {
 public:
  DistributedTensorType(std::vector<ExprPtr> shape, DataType dtype) : TensorType(std::move(shape), dtype) {}

  DistributedTensorType(std::vector<ExprPtr> shape, DataType dtype, MemRefPtr memref)
      : TensorType(std::move(shape), dtype, std::move(memref)) {}

  DistributedTensorType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref)
      : TensorType(std::move(shape), dtype, std::move(memref)) {}

  DistributedTensorType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref,
                        std::optional<TensorView> tensor_view)
      : TensorType(std::move(shape), dtype, std::move(memref), std::move(tensor_view)) {}

  DistributedTensorType(const std::vector<int64_t>& shape, DataType dtype)
      : TensorType(shape, dtype, std::nullopt) {}

  DistributedTensorType(const std::vector<int64_t>& shape, DataType dtype, std::optional<MemRefPtr> memref)
      : TensorType(shape, dtype, std::move(memref)) {}

  DistributedTensorType(const std::vector<int64_t>& shape, DataType dtype, std::optional<MemRefPtr> memref,
                        std::optional<TensorView> tensor_view)
      : TensorType(shape, dtype, std::move(memref), std::move(tensor_view)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::DistributedTensorType; }
  [[nodiscard]] std::string TypeName() const override { return "DistributedTensorType"; }

  static constexpr auto GetFieldDescriptors() { return TensorType::GetFieldDescriptors(); }
};

using DistributedTensorTypePtr = std::shared_ptr<const DistributedTensorType>;

/**
 * @brief Tile type representation
 *
 * Represents a tile type (multi-dimensional tensor).
 * Tiles are used for hardware-optimized operations on multi-dimensional data structures.
 * Note: Code generation currently only supports up to 2D tiles.
 */
class TileType : public ShapedType {
 public:
  std::optional<TileView> tile_view_;        ///< Optional tile view information
  std::optional<MemorySpace> memory_space_;  ///< Optional tile memory space; required when memref_ is present

  TileType(const std::vector<int64_t>& shape, DataType dtype, std::optional<MemRefPtr> memref = std::nullopt,
           std::optional<TileView> tile_view = std::nullopt,
           std::optional<MemorySpace> memory_space = std::nullopt);

  TileType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref = std::nullopt,
           std::optional<TileView> tile_view = std::nullopt,
           std::optional<MemorySpace> memory_space = std::nullopt);

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TileType; }
  [[nodiscard]] std::string TypeName() const override { return "TileType"; }
  [[nodiscard]] std::optional<MemorySpace> GetMemorySpace() const override;

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(ShapedType::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&TileType::tile_view_, "tile_view"),
                                          reflection::UsualField(&TileType::memory_space_, "memory_space")));
  }

 private:
  static std::optional<MemorySpace> ValidateMemorySpace(const std::optional<MemRefPtr>& memref,
                                                        std::optional<MemorySpace> memory_space);
};

using TileTypePtr = std::shared_ptr<const TileType>;

/**
 * @brief Array type representation
 *
 * Represents a small fixed-size homogeneous 1-D array that lives on the
 * on-core scalar register file / C stack (memory space ScalarLocal). All
 * elements share the same scalar dtype. Distinct from TensorType (which
 * lowers to GM/DDR pointers) and TileType (which is vector or cube hardware
 * state).
 *
 * Writes are SSA-functional: `array.update_element(arr, i, v)` returns a new
 * SSA value of ArrayType representing "arr with element i set to v" — no
 * in-place mutation in the IR. Codegen lowers a chain of update_element
 * Calls back to in-place C-stack mutation when the dataflow allows.
 *
 * v1 constraints:
 *  - Element dtype must be an integer type (signed or unsigned, or BOOL).
 *  - Shape is exactly rank-1; the single extent must be a compile-time ConstInt.
 *  - Never carries a MemRef (memref_ is always nullopt) — codegen lowers
 *    create/get_element/update_element directly to a C-stack array.
 *
 * An ArrayType value may be created and used inside any function/region
 * (orchestration or incore), but cannot cross function boundaries — see the
 * ArrayNotEscapedVerifier for the rule.
 */
class ArrayType : public ShapedType {
 public:
  /**
   * @brief Create an array type with a ConstInt extent.
   *
   * @param dtype Element data type (must be integer).
   * @param extent Number of elements (must be a ConstInt).
   */
  ArrayType(DataType dtype, ExprPtr extent);

  /**
   * @brief Create an array type with a literal int extent.
   *
   * @param dtype Element data type (must be integer).
   * @param extent Number of elements.
   */
  ArrayType(DataType dtype, int64_t extent);

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ArrayType; }
  [[nodiscard]] std::string TypeName() const override { return "ArrayType"; }
  [[nodiscard]] std::optional<MemorySpace> GetMemorySpace() const override {
    return MemorySpace::ScalarLocal;
  }

  /// Single-axis extent. Always a ConstInt — validated in constructor.
  [[nodiscard]] ExprPtr extent() const { return shape_.at(0); }

  static constexpr auto GetFieldDescriptors() { return ShapedType::GetFieldDescriptors(); }
};

using ArrayTypePtr = std::shared_ptr<const ArrayType>;

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

/**
 * @brief Pointer type for base allocation identity tokens
 *
 * Represents the type of variables returned by tile.alloc / tensor.alloc.
 * A Ptr variable is the allocation identity token that MemRefs reference
 * via their base_ field.
 */
class PtrType : public Type {
 public:
  PtrType() = default;

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::PtrType; }
  [[nodiscard]] std::string TypeName() const override { return "Ptr"; }

  static constexpr auto GetFieldDescriptors() { return Type::GetFieldDescriptors(); }
};

using PtrTypePtr = std::shared_ptr<const PtrType>;

/**
 * @brief Get a shared pointer to the singleton PtrType instance
 *
 * @return Shared pointer to PtrType
 */
inline PtrTypePtr GetPtrType() {
  static const auto ptr_type = std::make_shared<PtrType>();
  return ptr_type;
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TYPE_H_

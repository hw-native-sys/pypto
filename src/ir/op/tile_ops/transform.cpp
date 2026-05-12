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
 * @file transform.cpp
 * @brief Shape transformation tile operations (slice, reshape, transpose)
 *
 * This file implements shape transformation operations for tiles including
 * slice, reshape and transpose operations.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {
// ============================================================================
// Helper Functions (file-local)
// ============================================================================

/**
 * @brief Normalize axis index to handle negative indexing
 *
 * @param axis The axis index (can be negative)
 * @param ndim The number of dimensions
 * @return The normalized axis index
 */
int NormalizeAxis(int axis, size_t ndim) {
  if (axis < 0) {
    axis += static_cast<int>(ndim);
  }
  CHECK(axis >= 0 && axis < static_cast<int>(ndim))
      << "Axis " << axis << " is out of range for " << ndim << "D tile";
  return axis;
}

/**
 * @brief Compute the product of shape dimensions (for static shapes)
 *
 * @param shape The shape dimensions
 * @return The product if all dimensions are ConstInt, -1 otherwise
 */
int64_t ComputeShapeProduct(const std::vector<ExprPtr>& shape) {
  int64_t product = 1;
  for (const auto& dim : shape) {
    auto const_dim = As<ConstInt>(dim);
    if (!const_dim) {
      return -1;  // Dynamic shape, cannot compute product
    }
    product *= const_dim->value_;
  }
  return product;
}

TileLayout InferTileLayoutFromShape(const std::vector<ExprPtr>& shape) {
  if (shape.size() != 2) {
    return TileLayout::row_major;
  }

  auto rows_const = As<ConstInt>(shape[0]);
  auto cols_const = As<ConstInt>(shape[1]);
  if (!rows_const || !cols_const) {
    return TileLayout::row_major;
  }
  return (cols_const->value_ == 1 && rows_const->value_ > 1) ? TileLayout::col_major : TileLayout::row_major;
}

/**
 * @brief Validate that all elements of a TupleType are ScalarType with an index-like dtype
 *
 * @param tuple_type The tuple type whose elements to validate
 * @param op_name Name of the operation (for error messages)
 * @param arg_name Name of the argument (for error messages), e.g. "shape" or "offset"
 */
void ValidateIndexTupleElements(const TupleTypePtr& tuple_type, const std::string& op_name,
                                const std::string& arg_name) {
  for (size_t i = 0; i < tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(tuple_type->types_[i]);
    CHECK(scalar_type) << op_name << " " << arg_name << " tuple element " << i
                       << " must be ScalarType, but got " << tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsIndexLike())
        << op_name << " " << arg_name << " tuple element " << i
        << " must have dtype INT64, UINT64, or INDEX, but got " << scalar_type->dtype_.ToString();
  }
}

}  // anonymous namespace

// ============================================================================
// Type Inference Functions
// ============================================================================

TypePtr DeduceTileSliceType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tile.slice: (input, shape, offset[, valid_shape[, drop_dims]]).
  //   - valid_shape (4th arg): an empty MakeTuple means "no valid_shape".
  //   - drop_dims (5th arg): a MakeTuple of ConstInt listing axes to erase from
  //     the result type (numpy-style rank reduction); each listed axis must be a
  //     static unit dim of `shape`. An empty / absent operand drops nothing.
  // Tile floor: tiles are physically 2D, so if rank reduction would take the
  // result below 2D it is clamped back to 2D by prepending unit axes.
  CHECK(args.size() >= 3 && args.size() <= 5)
      << "tile.slice requires 3-5 arguments (input, shape, offset[, valid_shape[, drop_dims]]), but got "
      << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.slice requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (shape)
  auto shape_tuple_type = As<TupleType>(args[1]->GetType());
  CHECK(shape_tuple_type) << "tile.slice requires shape to be TupleType, but got "
                          << args[1]->GetType()->TypeName();

  // Validate all shape elements are ScalarType(INT64, UINT64, or INDEX)
  ValidateIndexTupleElements(shape_tuple_type, "tile.slice", "shape");

  auto shape_tuple = As<MakeTuple>(args[1]);
  CHECK(shape_tuple) << "tile.slice shape must be a MakeTuple with static compile-time dimensions";

  // Third argument must be TupleType (offset)
  auto offset_tuple_type = As<TupleType>(args[2]->GetType());
  CHECK(offset_tuple_type) << "tile.slice requires offset to be TupleType, but got "
                           << args[2]->GetType()->TypeName();

  // Validate all offset elements are ScalarType(INT64, UINT64, or INDEX)
  ValidateIndexTupleElements(offset_tuple_type, "tile.slice", "offset");
  CHECK(offset_tuple_type->types_.size() == shape_tuple_type->types_.size())
      << "tile.slice requires offset and shape to have the same rank, but got offset rank "
      << offset_tuple_type->types_.size() << " and shape rank " << shape_tuple_type->types_.size();

  std::vector<ExprPtr> new_shape;
  new_shape.reserve(shape_tuple->elements_.size());
  for (size_t i = 0; i < shape_tuple->elements_.size(); ++i) {
    auto static_dim = As<ConstInt>(shape_tuple->elements_[i]);
    CHECK(static_dim) << "tile.slice shape element " << i
                      << " must be a compile-time constant so InitMemRef can allocate storage";
    CHECK(static_dim->value_ > 0) << "tile.slice shape element " << i << " must be positive, got "
                                  << static_dim->value_;
    new_shape.push_back(shape_tuple->elements_[i]);
  }

  std::vector<ExprPtr> valid_shape = new_shape;
  if (args.size() >= 4) {
    auto valid_shape_tuple_type = As<TupleType>(args[3]->GetType());
    CHECK(valid_shape_tuple_type) << "tile.slice requires valid_shape to be TupleType, but got "
                                  << args[3]->GetType()->TypeName();
    // An empty tuple is the explicit "no valid_shape" form (so callers can pass
    // drop_dims as the 5th arg without supplying a custom valid_shape).
    if (!valid_shape_tuple_type->types_.empty()) {
      ValidateIndexTupleElements(valid_shape_tuple_type, "tile.slice", "valid_shape");
      CHECK(valid_shape_tuple_type->types_.size() == shape_tuple_type->types_.size())
          << "tile.slice requires valid_shape and shape to have the same rank, but got valid_shape rank "
          << valid_shape_tuple_type->types_.size() << " and shape rank " << shape_tuple_type->types_.size();

      valid_shape.clear();
      valid_shape.reserve(valid_shape_tuple_type->types_.size());
      if (auto valid_shape_tuple = As<MakeTuple>(args[3])) {
        valid_shape = valid_shape_tuple->elements_;
      } else {
        for (size_t i = 0; i < valid_shape_tuple_type->types_.size(); ++i) {
          valid_shape.emplace_back(
              std::make_shared<TupleGetItemExpr>(args[3], static_cast<int>(i), args[3]->span_));
        }
      }
    }
  }

  // Optional drop_dims (5th arg): axes erased from the result type, validated
  // against the full pre-reduction shape. Apply to both the static shape and the
  // valid_shape, then clamp back up to 2D (tiles are physically 2D) by prepending
  // unit axes if the natural result would be < 2D.
  const ExprPtr drop_dims_arg = args.size() == 5 ? args[4] : nullptr;
  const std::vector<int64_t> drop_dims = ParseSliceDropDims(drop_dims_arg, new_shape, "tile.slice");
  if (!drop_dims.empty()) {
    new_shape = ApplyDropDims(new_shape, drop_dims);
    valid_shape = ApplyDropDims(valid_shape, drop_dims);
    if (new_shape.size() < 2) {
      // Reuse the dtype of an existing static shape element for the synthetic unit axes.
      auto first_dim = As<ConstInt>(shape_tuple->elements_[0]);
      const DataType dim_dtype = first_dim->dtype();
      const size_t pad = 2 - new_shape.size();
      std::vector<ExprPtr> unit_axes(pad);
      for (size_t i = 0; i < pad; ++i) {
        unit_axes[i] = std::make_shared<ConstInt>(1, dim_dtype, args[1]->span_);
      }
      new_shape.insert(new_shape.begin(), unit_axes.begin(), unit_axes.end());
      valid_shape.insert(valid_shape.begin(), unit_axes.begin(), unit_axes.end());
    }
  }

  // Slice produces a window over the source tile.  PTO's pto.subview semantics
  // require the result tile_buf to share dtype, memory space, and the four
  // tile-config fields (blayout, slayout, fractal, pad) with the source.
  // Inherit only those fields from the source TileView; recompute the
  // window-specific fields (valid_shape and physical stride/start_offset are
  // inherent to the slice itself, not the parent buffer).  Tiles without an
  // explicit TileView fall back to inferring layout from shape (covers
  // intermediate IR built before backend-level allocation).
  TileView tile_view;
  if (tile_type->tile_view_.has_value()) {
    const auto& src_v = *tile_type->tile_view_;
    tile_view.blayout = src_v.blayout;
    tile_view.slayout = src_v.slayout;
    tile_view.fractal = src_v.fractal;
    tile_view.pad = src_v.pad;
  } else {
    tile_view.blayout = InferTileLayoutFromShape(new_shape);
  }
  tile_view.valid_shape = valid_shape;

  // Optional pad_value kwarg overrides the inherited pad mode.  When the user
  // explicitly requests padding on a slice, codegen will reject it via
  // CheckSubviewTileCompat (pto.subview is a pure view and cannot pad) and
  // direct callers to use tile.fillpad on the slice result.
  bool pad_value_specified = false;
  PadValue pad_value = PadValue::null;
  for (const auto& [k, v] : kwargs) {
    if (k != "pad_value") continue;
    CHECK(v.type() == typeid(PadValue))
        << "tile.slice pad_value must be a PadValue enum, got " << v.type().name();
    pad_value = std::any_cast<PadValue>(v);
    CHECK(pad_value == PadValue::null || pad_value == PadValue::zero || pad_value == PadValue::max ||
          pad_value == PadValue::min)
        << "tile.slice pad_value has invalid enum value: " << static_cast<int>(pad_value);
    pad_value_specified = true;
    break;
  }
  if (pad_value_specified) {
    tile_view.pad = pad_value;
  }

  return std::make_shared<TileType>(new_shape, tile_type->dtype_, std::nullopt, tile_view);
}

TypePtr DeduceTileReshapeType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tile.reshape requires exactly 2 arguments: input tile and shape tuple
  CHECK(args.size() == 2) << "tile.reshape requires exactly 2 arguments (input, shape), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.reshape requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (shape)
  auto shape_tuple_type = As<TupleType>(args[1]->GetType());
  CHECK(shape_tuple_type) << "tile.reshape requires shape to be TupleType, but got "
                          << args[1]->GetType()->TypeName();

  // Validate all shape elements are ScalarType(INT64, UINT64, or INDEX)
  ValidateIndexTupleElements(shape_tuple_type, "tile.reshape", "shape");

  // Extract new shape dimensions
  // If args[1] is MakeTuple, extract elements directly to preserve constants
  // Otherwise use TupleGetItemExpr for runtime tuples
  std::vector<ExprPtr> new_shape;
  new_shape.reserve(shape_tuple_type->types_.size());

  if (auto make_tuple = As<MakeTuple>(args[1])) {
    // MakeTuple: extract elements directly to preserve ConstInt
    new_shape = make_tuple->elements_;
  } else {
    // Runtime tuple: use TupleGetItemExpr
    for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
      new_shape.emplace_back(
          std::make_shared<TupleGetItemExpr>(args[1], static_cast<int>(i), args[1]->span_));
    }
  }

  // For static shapes, verify that the total number of elements matches
  int64_t old_product = ComputeShapeProduct(tile_type->shape_);
  int64_t new_product = ComputeShapeProduct(new_shape);

  if (old_product > 0 && new_product > 0) {
    CHECK(old_product == new_product) << "tile.reshape: cannot reshape tile of size " << old_product
                                      << " into shape with size " << new_product;
  }

  // Return new TileType with reshaped dimensions and same dtype
  TileView tile_view;
  tile_view.valid_shape = new_shape;

  tile_view.blayout = InferTileLayoutFromShape(new_shape);

  return std::make_shared<TileType>(new_shape, tile_type->dtype_, std::nullopt, tile_view);
}

TypePtr DeduceTileTransposeType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tile.transpose requires exactly 3 arguments: input tile, axis1, axis2
  CHECK(args.size() == 3) << "tile.transpose requires exactly 3 arguments (input, axis1, axis2), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.transpose requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  const auto& input_shape = tile_type->shape_;
  size_t ndim = input_shape.size();

  CHECK(ndim >= 2) << "tile.transpose requires at least 2 dimensions, but got " << ndim;

  // Second argument is axis1 (ConstInt)
  auto axis1_const = As<ConstInt>(args[1]);
  CHECK(axis1_const) << "tile.transpose requires second argument (axis1) to be a ConstInt";

  // Third argument is axis2 (ConstInt)
  auto axis2_const = As<ConstInt>(args[2]);
  CHECK(axis2_const) << "tile.transpose requires third argument (axis2) to be a ConstInt";

  int axis1 = NormalizeAxis(static_cast<int>(axis1_const->value_), ndim);
  int axis2 = NormalizeAxis(static_cast<int>(axis2_const->value_), ndim);

  CHECK(axis1 != axis2) << "tile.transpose: axis1 and axis2 must be different, but got axis1=" << axis1
                        << ", axis2=" << axis2;

  // Create new shape by swapping the specified dimensions
  std::vector<ExprPtr> new_shape = input_shape;
  std::swap(new_shape[axis1], new_shape[axis2]);

  // Return new TileType with transposed shape and same dtype
  TileView tile_view;
  tile_view.valid_shape = new_shape;
  return std::make_shared<TileType>(new_shape, tile_type->dtype_, std::nullopt, tile_view);
}

// ============================================================================
// Registration Function for Tile Transform Operations
// ============================================================================

REGISTER_OP("tile.slice")
    .set_op_category("TileOp")
    .set_description("Create a slice of a tile with static shape and optional dynamic valid_shape")
    .add_argument("input", "Input tile (TileType)")
    .add_argument("shape", "Static shape dimensions (TupleType of ScalarType(INT64/UINT64/INDEX))")
    .add_argument("offset", "Offset dimensions (TupleType of ScalarType(INT64/UINT64/INDEX))")
    .add_argument("valid_shape",
                  "Optional logical valid shape (TupleType of ScalarType(INT64/UINT64/INDEX)); "
                  "an empty tuple means none")
    .add_argument("drop_dims",
                  "Optional axes (MakeTuple of ConstInt) erased from the result type; the result is "
                  "clamped to 2D if reduction would take it below 2D")
    .set_output_memory_inherit_input()
    .set_attr<PadValue>("pad_value")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileSliceType(args, kwargs);
    });

REGISTER_OP("tile.reshape")
    .set_op_category("TileOp")
    .set_description("Reshape tile to new shape")
    .add_argument("input", "Input tile (TileType)")
    .add_argument("shape", "New shape dimensions (TupleType of ScalarType(INT64/UINT64/INDEX))")
    .set_output_memory_inherit_input()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileReshapeType(args, kwargs);
    });

REGISTER_OP("tile.transpose")
    .set_op_category("TileOp")
    .set_description("Transpose tile by swapping two axes")
    .add_argument("input", "Input tile (TileType)")
    .add_argument("axis1", "First axis to swap (ConstInt)")
    .add_argument("axis2", "Second axis to swap (ConstInt)")
    .set_output_memory_inherit_input()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTransposeType(args, kwargs);
    });

TypePtr DeduceTileAssembleType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3) << "tile.assemble requires exactly 3 arguments (target, source, offset), but got "
                          << args.size();

  auto target_type = As<TileType>(args[0]->GetType());
  CHECK(target_type) << "tile.assemble requires first argument (target) to be a TileType, but got "
                     << args[0]->GetType()->TypeName();

  auto source_type = As<TileType>(args[1]->GetType());
  CHECK(source_type) << "tile.assemble requires second argument (source) to be a TileType, but got "
                     << args[1]->GetType()->TypeName();

  auto offset_tuple_type = As<TupleType>(args[2]->GetType());
  CHECK(offset_tuple_type) << "tile.assemble requires offset to be TupleType, but got "
                           << args[2]->GetType()->TypeName();

  CHECK(As<MakeTuple>(args[2])) << "tile.assemble offset must be a literal tuple (e.g., (row, col)), "
                                << "not a variable or computed expression";

  ValidateIndexTupleElements(offset_tuple_type, "tile.assemble", "offset");

  CHECK(target_type->dtype_ == source_type->dtype_)
      << "tile.assemble requires target and source to have the same dtype, but got "
      << target_type->dtype_.ToString() << " and " << source_type->dtype_.ToString();

  // Inherit layout metadata (blayout, slayout, fractal, pad) from the target so that
  // the result type carries the correct tile_buf type annotation for codegen.
  TileView tile_view;
  if (target_type->tile_view_.has_value()) {
    tile_view = *target_type->tile_view_;
  }
  tile_view.valid_shape = target_type->shape_;
  return std::make_shared<TileType>(target_type->shape_, target_type->dtype_, std::nullopt, tile_view,
                                    target_type->memory_space_);
}

REGISTER_OP("tile.assemble")
    .set_op_category("TileOp")
    .set_description("Write source tile data into target tile at specified offset")
    .add_argument("target", "Target tile (TileType)")
    .add_argument("source", "Source tile to write (TileType)")
    .add_argument("offset", "Offset dimensions (TupleType of ScalarType(INT64/UINT64/INDEX))")
    .set_output_memory_inherit_input()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileAssembleType(args, kwargs);
    });

TypePtr DeduceTileExtractType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tile.extract(src, index_row, index_col, shape) — ISA TEXTRACT Variant 1.
  // shape carries the static destination shape (2D MakeTuple of ConstInt).
  // target_memory kwarg drives the destination memory space.
  CHECK(args.size() == 4) << "tile.extract requires exactly 4 arguments "
                          << "(src, index_row, index_col, shape), but got " << args.size();

  auto src_type = As<TileType>(args[0]->GetType());
  CHECK(src_type) << "tile.extract requires src to be a TileType, but got " << args[0]->GetType()->TypeName();
  CHECK(src_type->shape_.size() == 2)
      << "tile.extract requires a 2D source tile, but got rank " << src_type->shape_.size();

  for (size_t i = 1; i <= 2; ++i) {
    auto idx_type = As<ScalarType>(args[i]->GetType());
    const char* name = (i == 1) ? "index_row" : "index_col";
    CHECK(idx_type) << "tile.extract " << name << " must be ScalarType, but got "
                    << args[i]->GetType()->TypeName();
    CHECK(idx_type->dtype_.IsIndexLike())
        << "tile.extract " << name << " must have INT64/UINT64/INDEX dtype, but got "
        << idx_type->dtype_.ToString();
  }

  auto shape_tuple_type = As<TupleType>(args[3]->GetType());
  CHECK(shape_tuple_type) << "tile.extract shape must be TupleType, but got "
                          << args[3]->GetType()->TypeName();
  ValidateIndexTupleElements(shape_tuple_type, "tile.extract", "shape");

  auto shape_tuple = As<MakeTuple>(args[3]);
  CHECK(shape_tuple) << "tile.extract shape must be a literal MakeTuple of ConstInt";
  CHECK(shape_tuple->elements_.size() == 2)
      << "tile.extract shape must be 2D, got rank " << shape_tuple->elements_.size();

  std::vector<ExprPtr> dst_shape;
  dst_shape.reserve(2);
  for (size_t i = 0; i < 2; ++i) {
    auto c = As<ConstInt>(shape_tuple->elements_[i]);
    CHECK(c) << "tile.extract shape[" << i << "] must be a compile-time ConstInt";
    CHECK(c->value_ > 0) << "tile.extract shape[" << i << "] must be positive, got " << c->value_;
    dst_shape.push_back(shape_tuple->elements_[i]);
  }

  // Static-bounds check: when src dim, dst dim, and offset are all constants,
  // verify offset + dst_shape <= src_shape per ISA TEXTRACT bounds rule.
  auto check_axis = [&](size_t axis, const char* axis_name, const ExprPtr& offset_arg) {
    auto src_dim = As<ConstInt>(src_type->shape_[axis]);
    auto dst_dim = As<ConstInt>(dst_shape[axis]);
    if (!src_dim || !dst_dim) return;
    CHECK(dst_dim->value_ <= src_dim->value_) << "tile.extract shape[" << axis << "]=" << dst_dim->value_
                                              << " exceeds src " << axis_name << " " << src_dim->value_;
    auto off = As<ConstInt>(offset_arg);
    if (!off) return;
    CHECK(off->value_ >= 0) << "tile.extract index_" << axis_name << " must be >= 0, got " << off->value_;
    CHECK(off->value_ + dst_dim->value_ <= src_dim->value_)
        << "tile.extract index_" << axis_name << "=" << off->value_ << " + shape[" << axis
        << "]=" << dst_dim->value_ << " exceeds src " << axis_name << " " << src_dim->value_;
  };
  check_axis(0, "row", args[1]);
  check_axis(1, "col", args[2]);

  TileView tile_view;
  tile_view.valid_shape = dst_shape;
  tile_view.blayout = InferTileLayoutFromShape(dst_shape);

  // Override blayout/slayout for L0-resident destinations to match the
  // architectural layouts the A2/A3 `pto.textract` codegen verifier
  // expects.  Mat/other targets keep the inferred default.  These
  // hardcoded layouts mirror the L0A/L0B formats the hardware imposes on
  // TEXTRACT outputs (and differ from `tile.move`'s TMOV-side L0 formats).
  for (const auto& [k, v] : kwargs) {
    if (k != "target_memory") continue;
    auto target = AnyCast<MemorySpace>(v, "kwarg key: target_memory");
    if (target == MemorySpace::Left) {
      tile_view.blayout = TileLayout::row_major;
      tile_view.slayout = TileLayout::row_major;
    } else if (target == MemorySpace::Right) {
      tile_view.blayout = TileLayout::row_major;
      tile_view.slayout = TileLayout::col_major;
    }
    break;
  }

  return std::make_shared<TileType>(dst_shape, src_type->dtype_, std::nullopt, tile_view);
}

REGISTER_OP("tile.extract")
    .set_op_category("TileOp")
    .set_description("Extract a sub-tile from src at (index_row, index_col) — ISA TEXTRACT Variant 1")
    .add_argument("src", "Source tile (TileType, 2D; typically Mat or Acc memory)")
    .add_argument("index_row", "Starting row offset (ScalarType INT64/UINT64/INDEX)")
    .add_argument("index_col", "Starting col offset (ScalarType INT64/UINT64/INDEX)")
    .add_argument("shape", "Static destination shape (TupleType, 2D MakeTuple of ConstInt)")
    .set_attr<MemorySpace>("target_memory")
    .set_output_memory_from_kwarg("target_memory", MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileExtractType(args, kwargs);
    });

TypePtr DeduceTileScatterUpdateType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tile.scatter_update(input, index, src, scratch) -> TileType same as input
  // input:   TileType 2D [rows, d] or 4D [blockNum, blockSize, 1, d]
  // index:   TileType 2D [b, s] of integer dtype
  // src:     TileType 2D [b*s, d] or 4D [b, s, 1, d] (same rank as input)
  // scratch: TileType 2D [1, d] with same dtype as input (caller-allocated row buffer)
  CHECK(args.size() == 4)
      << "tile.scatter_update requires exactly 4 arguments (input, index, src, scratch), got " << args.size();

  auto input_type = As<TileType>(args[0]->GetType());
  CHECK(input_type) << "tile.scatter_update: input must be TileType, got " << args[0]->GetType()->TypeName();
  CHECK(input_type->shape_.size() == 2 || input_type->shape_.size() == 4)
      << "tile.scatter_update: input must be 2D or 4D, got rank " << input_type->shape_.size();

  auto index_type = As<TileType>(args[1]->GetType());
  CHECK(index_type) << "tile.scatter_update: index must be TileType, got " << args[1]->GetType()->TypeName();
  CHECK(index_type->shape_.size() == 2)
      << "tile.scatter_update: index must be 2D [b, s], got rank " << index_type->shape_.size();
  CHECK(index_type->dtype_.IsInt()) << "tile.scatter_update: index dtype must be integer, got "
                                    << index_type->dtype_.ToString();

  auto src_type = As<TileType>(args[2]->GetType());
  CHECK(src_type) << "tile.scatter_update: src must be TileType, got " << args[2]->GetType()->TypeName();
  CHECK(src_type->shape_.size() == input_type->shape_.size())
      << "tile.scatter_update: src rank (" << src_type->shape_.size() << ") must match input rank ("
      << input_type->shape_.size() << ")";
  CHECK(src_type->dtype_ == input_type->dtype_)
      << "tile.scatter_update: src dtype (" << src_type->dtype_.ToString() << ") must match input dtype ("
      << input_type->dtype_.ToString() << ")";

  auto scratch_type = As<TileType>(args[3]->GetType());
  CHECK(scratch_type) << "tile.scatter_update: scratch must be TileType, got "
                      << args[3]->GetType()->TypeName();
  CHECK(scratch_type->shape_.size() == 2)
      << "tile.scatter_update: scratch must be 2D [1, d], got rank " << scratch_type->shape_.size();
  auto scratch_rows = As<ConstInt>(scratch_type->shape_[0]);
  CHECK(scratch_rows && scratch_rows->value_ == 1) << "tile.scatter_update: scratch rows must be 1";
  CHECK(AreExprsEqual(scratch_type->shape_[1], input_type->shape_.back()))
      << "tile.scatter_update: scratch cols must match input feature dimension";
  CHECK(scratch_type->dtype_ == input_type->dtype_)
      << "tile.scatter_update: scratch dtype (" << scratch_type->dtype_.ToString()
      << ") must match input dtype (" << input_type->dtype_.ToString() << ")";

  for (const auto& [key, val] : kwargs) {
    if (key == "dim") {
      int dim_val = AnyCast<int>(val, "kwarg key: dim");
      CHECK(dim_val == -2) << "tile.scatter_update: only dim=-2 is currently supported, got " << dim_val;
    }
  }

  // Inherit tile_view (with valid_shape = input shape) and memory_space from input,
  // same pattern as tile.assemble — ensures tile.store can read valid_shape downstream.
  TileView tile_view;
  if (input_type->tile_view_.has_value()) {
    tile_view = *input_type->tile_view_;
  }
  if (tile_view.valid_shape.empty()) {
    tile_view.valid_shape = input_type->shape_;
  }
  return std::make_shared<TileType>(input_type->shape_, input_type->dtype_, std::nullopt, tile_view,
                                    input_type->memory_space_);
}

REGISTER_OP("tile.scatter_update")
    .set_op_category("TileOp")
    .set_description(
        "Update input tile rows at positions given by 2D index tile with values from src. "
        "Supports 2D input [rows, d] with 2D src [b*s, d], and 4D input [blockNum, blockSize, 1, d] "
        "with 4D src [b, s, 1, d]. Index is always 2D [b, s] of integer dtype. Caller must pass a "
        "[1, d] scratch tile for the per-row staging buffer.")
    .add_argument("input", "Destination tile (2D [rows, d] or 4D [blockNum, blockSize, 1, d])")
    .add_argument("index", "2D index tile [b, s] of integer dtype")
    .add_argument("src", "Source tile (2D [b*s, d] or 4D [b, s, 1, d])")
    .add_argument("scratch", "Scratch row tile [1, d] with same dtype as input")
    .set_attr<int>("dim")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_input_memory(3, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .set_output_reuses_input(0)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileScatterUpdateType(args, kwargs);
    });

TypePtr DeduceTileConcatType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 2) << "tile.concat requires 2 arguments (src0, src1), got " << args.size();

  auto t0 = As<TileType>(args[0]->GetType());
  auto t1 = As<TileType>(args[1]->GetType());
  CHECK(t0) << "tile.concat: src0 must be TileType, got " << args[0]->GetType()->TypeName();
  CHECK(t1) << "tile.concat: src1 must be TileType, got " << args[1]->GetType()->TypeName();
  CHECK(t0->dtype_ == t1->dtype_) << "tile.concat: src0 and src1 must have same dtype, got "
                                  << t0->dtype_.ToString() << " and " << t1->dtype_.ToString();
  CHECK(t0->shape_.size() == 2 && t1->shape_.size() == 2) << "tile.concat requires 2D tiles";

  auto r0 = As<ConstInt>(t0->shape_[0]);
  auto r1 = As<ConstInt>(t1->shape_[0]);
  if (r0 && r1) {
    CHECK(r0->value_ == r1->value_) << "tile.concat: row count must match, got " << r0->value_ << " vs "
                                    << r1->value_;
  }

  std::vector<ExprPtr> out_shape = {t0->shape_[0]};
  auto c0 = As<ConstInt>(t0->shape_[1]);
  auto c1 = As<ConstInt>(t1->shape_[1]);
  if (c0 && c1) {
    out_shape.push_back(std::make_shared<ConstInt>(c0->value_ + c1->value_, c0->dtype(), args[0]->span_));
  } else {
    out_shape.push_back(std::make_shared<Add>(t0->shape_[1], t1->shape_[1], DataType::INDEX, args[0]->span_));
  }

  TileView tile_view;
  tile_view.valid_shape = out_shape;
  return std::make_shared<TileType>(out_shape, t0->dtype_, std::nullopt, tile_view);
}

REGISTER_OP("tile.concat")
    .set_op_category("TileOp")
    .set_description("Concatenate two tiles along column dimension")
    .add_argument("src0", "First source tile (TileType)")
    .add_argument("src1", "Second source tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileConcatType(args, kwargs);
    });

TypePtr DeduceTileSetValidShapeType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3)
      << "tile.set_validshape requires exactly 3 arguments (tile, valid_rows, valid_cols), but got "
      << args.size();

  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.set_validshape requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();
  CHECK(tile_type->shape_.size() == 2)
      << "tile.set_validshape requires a 2D tile, but got rank " << tile_type->shape_.size();

  auto vr_type = As<ScalarType>(args[1]->GetType());
  CHECK(vr_type) << "tile.set_validshape valid_rows must be ScalarType, but got "
                 << args[1]->GetType()->TypeName();
  CHECK(vr_type->dtype_.IsIndexLike())
      << "tile.set_validshape valid_rows must have dtype INT64, UINT64, or INDEX, but got "
      << vr_type->dtype_.ToString();

  auto vc_type = As<ScalarType>(args[2]->GetType());
  CHECK(vc_type) << "tile.set_validshape valid_cols must be ScalarType, but got "
                 << args[2]->GetType()->TypeName();
  CHECK(vc_type->dtype_.IsIndexLike())
      << "tile.set_validshape valid_cols must have dtype INT64, UINT64, or INDEX, but got "
      << vc_type->dtype_.ToString();

  auto check_const_bound = [&](const char* name, const ExprPtr& valid, const ExprPtr& bound) {
    if (auto c = As<ConstInt>(valid)) {
      CHECK(c->value_ >= 0) << "tile.set_validshape " << name << " must be >= 0, got " << c->value_;
      if (auto b = As<ConstInt>(bound)) {
        CHECK(c->value_ <= b->value_)
            << "tile.set_validshape " << name << " (" << c->value_ << ") exceeds tile bound " << b->value_;
      }
    }
  };
  check_const_bound("valid_rows", args[1], tile_type->shape_[0]);
  check_const_bound("valid_cols", args[2], tile_type->shape_[1]);

  TileView tile_view;
  if (tile_type->tile_view_.has_value()) {
    tile_view = *tile_type->tile_view_;
  }
  tile_view.valid_shape = {args[1], args[2]};

  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
}

// NOTE: Internal op for compiler-generated code only; should not be exposed to end users in future releases.
REGISTER_OP("tile.set_validshape")
    .set_op_category("TileOp")
    .set_description("Update valid-shape metadata of a tile without data movement (internal)")
    .add_argument("tile", "Input tile (TileType, 2D)")
    .add_argument("valid_rows", "Number of valid rows (ScalarType INDEX/INT64/UINT64)")
    .add_argument("valid_cols", "Number of valid columns (ScalarType INDEX/INT64/UINT64)")
    .set_output_memory_inherit_input()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileSetValidShapeType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto

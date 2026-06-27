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
 * @file reduction.cpp
 * @brief Reduction tile operations (Sum, Max, Min)
 *
 * This file implements reduction operations for tile-level programming.
 * Reduction operations can reduce a TileType along specified axes.
 */

#include <any>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

// Helper to get kwargs value with default (uses vector to preserve order)
template <typename T>
T GetKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
           const std::optional<T>& default_value = std::nullopt) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<T>(v, "kwarg key: " + key);
    }
  }
  if (default_value) {
    return *default_value;
  }
  throw ValueError("Missing kwarg: " + key);
}

TypePtr DeduceTileReductionType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name) {
  // tile.sum and tile.max require 1 argument (tile) and 2 attributes (axis, keepdim)
  CHECK(args.size() == 1) << "The operator " << op_name << " requires 1 argument, but got " << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Get the input shape
  const auto& input_shape = tile_type->shape_;
  int64_t input_ndim = static_cast<int64_t>(input_shape.size());

  // Determine which axes to reduce
  std::set<int64_t> reduce_axes;

  // Extract axis from kwargs (required)
  int axis_value = GetKwarg<int>(kwargs, "axis");
  if (axis_value < 0) {
    // Negative axis: convert to positive
    axis_value = static_cast<int>(input_ndim) + axis_value;
  }
  CHECK(axis_value >= 0 && static_cast<int64_t>(axis_value) < input_ndim)
      << "The operator " << op_name << " axis " << axis_value << " is out of range for shape with "
      << input_ndim << " dimensions";
  reduce_axes.insert(static_cast<int64_t>(axis_value));

  // Extract keepdim from kwargs (optional, default to false)
  bool keepdim = GetKwarg<bool>(kwargs, "keepdim", false);

  // If all axes are reduced and keepdim is false, return ScalarType
  if (static_cast<int64_t>(reduce_axes.size()) == input_ndim && !keepdim) {
    return std::make_shared<ScalarType>(tile_type->dtype_);
  }

  // Source of truth for valid_shape — falls back to physical shape when the
  // input has no TileView populated (issue #1401).
  const auto input_valid = GetValidShape(tile_type);

  // Build output shape and valid_shape together: the reduction rule applies
  // identically to both — physical dims drive the static shape, the input's
  // valid extents drive the output's valid extents.
  std::vector<ExprPtr> output_shape;
  std::vector<ExprPtr> output_valid;
  output_shape.reserve(input_ndim);
  output_valid.reserve(input_ndim);
  for (int64_t i = 0; i < input_ndim; ++i) {
    const bool is_reduced = reduce_axes.find(i) != reduce_axes.end();
    if (is_reduced) {
      if (keepdim) {
        output_shape.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown()));
        output_valid.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown()));
      }
    } else {
      output_shape.push_back(input_shape[i]);
      output_valid.push_back(input_valid[i]);
    }
  }

  // If output shape is empty, return ScalarType
  if (output_shape.empty()) {
    return std::make_shared<ScalarType>(tile_type->dtype_);
  }

  // Return TileType with reduced shape
  TileView tile_view;
  tile_view.valid_shape = std::move(output_valid);
  return std::make_shared<TileType>(std::move(output_shape), tile_type->dtype_, std::nullopt,
                                    std::move(tile_view));
}

// Type deduction for row reduction operations (reduces along last axis with keepdim=True).
// `out_dtype` overrides the output element type — used by argmax/argmin, whose index output
// is int32 rather than the source dtype.
TypePtr DeduceTileRowReductionType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name,
                                   std::optional<DataType> out_dtype = std::nullopt) {
  // tile.row_max and tile.row_sum require 2 arguments (tile and tmp_tile)
  CHECK(args.size() == 2) << "The operator " << op_name << " requires 2 arguments, but got " << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Get the input shape
  const auto& input_shape = tile_type->shape_;
  int64_t input_ndim = static_cast<int64_t>(input_shape.size());

  // Row reduction requires at least 2D tile
  CHECK(input_ndim >= 2) << "The operator " << op_name << " requires at least a 2D tile, but got "
                         << input_ndim << " dimensions";

  // Output shape is [...batch_dims, rows, 1] - reduce along last axis with keepdim=True
  std::vector<ExprPtr> output_shape(input_shape.begin(), input_shape.end() - 1);
  output_shape.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown()));

  // Inherit valid_shape from the input along the non-reduced dims so that
  // downstream codegen emits trowsum with the correct valid_row (issue #1401).
  // The reduced (last) dim collapses to 1 in the output.
  const auto input_valid = GetValidShape(tile_type);
  std::vector<ExprPtr> output_valid(input_valid.begin(), input_valid.end() - 1);
  output_valid.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown()));

  TileView tile_view;
  tile_view.blayout = TileLayout::col_major;
  tile_view.valid_shape = std::move(output_valid);
  return std::make_shared<TileType>(std::move(output_shape), out_dtype.value_or(tile_type->dtype_),
                                    std::nullopt, std::move(tile_view));
}

// Type deduction for column reduction operations (reduces along first axis with keepdim=True).
// `out_dtype` overrides the output element type — used by argmax/argmin index output (int32).
// col_sum accepts 1 arg (sequential) or 2 args (tile + tmp_tile for binary-tree reduction).
// col_max and col_min require 1 argument.
TypePtr DeduceTileColReductionType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name,
                                   std::optional<DataType> out_dtype = std::nullopt) {
  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  const auto& input_shape = tile_type->shape_;
  int64_t input_ndim = static_cast<int64_t>(input_shape.size());
  CHECK(input_ndim >= 2) << "The operator " << op_name << " requires at least a 2D tile, but got "
                         << input_ndim << " dimensions";

  // Output shape: [1, ...remaining] — reduce along first axis with keepdim=True
  std::vector<ExprPtr> output_shape;
  output_shape.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown()));
  output_shape.insert(output_shape.end(), input_shape.begin() + 1, input_shape.end());

  // Inherit valid_shape from the input along the non-reduced dims so that
  // downstream codegen emits tcolsum with the correct valid_col (issue #1401).
  // The reduced (first) dim is always 1 in the output.
  const auto input_valid = GetValidShape(tile_type);
  std::vector<ExprPtr> output_valid;
  output_valid.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown()));
  output_valid.insert(output_valid.end(), input_valid.begin() + 1, input_valid.end());

  TileView tile_view;
  tile_view.blayout = TileLayout::row_major;
  tile_view.valid_shape = std::move(output_valid);
  return std::make_shared<TileType>(std::move(output_shape), out_dtype.value_or(tile_type->dtype_),
                                    std::nullopt, std::move(tile_view));
}

// ============================================================================
// Registration Function for Block Reduction Operations
// ============================================================================

REGISTER_OP("tile.sum")
    .set_op_category("TileOp")
    .set_description("Sum reduction of a tile along specified axis")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp_tile", "Temporary tile (TileType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keepdim")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileReductionType(args, kwargs, "tile.sum");
    });

REGISTER_OP("tile.max")
    .set_op_category("TileOp")
    .set_description("Max reduction of a tile along specified axis")
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keepdim")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileReductionType(args, kwargs, "tile.max");
    });
REGISTER_OP("tile.min")
    .set_op_category("TileOp")
    .set_description("Min reduction of a tile along specified axis")
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keepdim")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileReductionType(args, kwargs, "tile.min");
    });

// ============================================================================
// Row Reduction Operations (TROWSUM, TROWMAX, TROWMIN)
// ============================================================================

REGISTER_OP("tile.row_sum")
    .set_op_category("TileOp")
    .set_description("Row-wise sum reduction (reduces along axis=1, maps to TROWSUM)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp_tile", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // TROW* reads the full input row + tmp scratch while writing the reduced
    // output, so the output must not share a buffer with either input.
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileRowReductionType(args, kwargs, "tile.row_sum");
    });

REGISTER_OP("tile.row_max")
    .set_op_category("TileOp")
    .set_description("Row-wise max reduction (reduces along axis=1, maps to TROWMAX)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp_tile", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // TROW* reads the full input row + tmp scratch while writing the reduced
    // output, so the output must not share a buffer with either input.
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileRowReductionType(args, kwargs, "tile.row_max");
    });

REGISTER_OP("tile.row_min")
    .set_op_category("TileOp")
    .set_description("Row-wise min reduction (reduces along axis=1, maps to TROWMIN)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp_tile", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // TROW* reads the full input row + tmp scratch while writing the reduced
    // output, so the output must not share a buffer with either input.
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileRowReductionType(args, kwargs, "tile.row_min");
    });

REGISTER_OP("tile.row_prod")
    .set_op_category("TileOp")
    .set_description("Row-wise product reduction (reduces along axis=1, maps to TROWPROD)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp_tile", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // TROW* reads the full input row + tmp scratch while writing the reduced
    // output, so the output must not share a buffer with either input.
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileRowReductionType(args, kwargs, "tile.row_prod");
    });

// ============================================================================
// Column Reduction Operations (TCOLSUM, TCOLMAX, TCOLMIN)
// ============================================================================

REGISTER_OP("tile.col_sum")
    .set_op_category("TileOp")
    .set_description(
        "Column-wise sum reduction (reduces along axis=0, maps to TCOLSUM). "
        "Passing an optional second tmp_tile activates the binary-tree reduction path.")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp_tile", "Optional scratch tile for binary-tree reduction (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 1 || args.size() == 2)
          << "The operator tile.col_sum requires 1 or 2 arguments, but got " << args.size();
      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(tile_type) << "The operator tile.col_sum requires first argument to be a TileType, but got "
                       << args[0]->GetType()->TypeName();
      if (args.size() == 2) {
        auto tmp_type = As<TileType>(args[1]->GetType());
        CHECK(tmp_type) << "The operator tile.col_sum requires tmp_tile to be a TileType, but got "
                        << args[1]->GetType()->TypeName();
        CHECK(tmp_type->dtype_ == tile_type->dtype_)
            << "The operator tile.col_sum requires tmp_tile dtype to match input dtype";
      }
      return DeduceTileColReductionType(args, kwargs, "tile.col_sum");
    });

REGISTER_OP("tile.col_max")
    .set_op_category("TileOp")
    .set_description("Column-wise max reduction (reduces along axis=0, maps to TCOLMAX)")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 1) << "The operator tile.col_max requires 1 argument, but got " << args.size();
      return DeduceTileColReductionType(args, kwargs, "tile.col_max");
    });

REGISTER_OP("tile.col_min")
    .set_op_category("TileOp")
    .set_description("Column-wise min reduction (reduces along axis=0, maps to TCOLMIN)")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 1) << "The operator tile.col_min requires 1 argument, but got " << args.size();
      return DeduceTileColReductionType(args, kwargs, "tile.col_min");
    });

REGISTER_OP("tile.col_prod")
    .set_op_category("TileOp")
    .set_description("Column-wise product reduction (reduces along axis=0, maps to TCOLPROD)")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 1) << "The operator tile.col_prod requires 1 argument, but got " << args.size();
      return DeduceTileColReductionType(args, kwargs, "tile.col_prod");
    });

// ============================================================================
// Argmax / argmin reductions (index-typed int32 output, requires a tmp tile)
// ============================================================================

REGISTER_OP("tile.row_argmax")
    .set_op_category("TileOp")
    .set_description("Row-wise argmax: column index of the per-row maximum (maps to TROWARGMAX)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp_tile", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // The TROWARGMAX path reads the full source/tmp while producing a smaller
    // index output, so the output must not alias an input (mirrors row_max).
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileRowReductionType(args, kwargs, "tile.row_argmax", DataType(DataType::INT32));
    });

REGISTER_OP("tile.row_argmin")
    .set_op_category("TileOp")
    .set_description("Row-wise argmin: column index of the per-row minimum (maps to TROWARGMIN)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp_tile", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileRowReductionType(args, kwargs, "tile.row_argmin", DataType(DataType::INT32));
    });

REGISTER_OP("tile.col_argmax")
    .set_op_category("TileOp")
    .set_description("Column-wise argmax: row index of the per-column maximum (maps to TCOLARGMAX)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp_tile", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // TCOLARGMAX reads the full source/tmp into a smaller index output — the
    // output must not alias an input (the tmp-bearing column variants share the
    // row-reduction in-place hazard, unlike the 1-arg col_max/col_min).
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2) << "The operator tile.col_argmax requires 2 arguments, but got " << args.size();
      return DeduceTileColReductionType(args, kwargs, "tile.col_argmax", DataType(DataType::INT32));
    });

REGISTER_OP("tile.col_argmin")
    .set_op_category("TileOp")
    .set_description("Column-wise argmin: row index of the per-column minimum (maps to TCOLARGMIN)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp_tile", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2) << "The operator tile.col_argmin requires 2 arguments, but got " << args.size();
      return DeduceTileColReductionType(args, kwargs, "tile.col_argmin", DataType(DataType::INT32));
    });

}  // namespace ir
}  // namespace pypto

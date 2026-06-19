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
 * @file interleave.cpp
 * @brief Interleave tile operations
 *
 * This file implements interleave operators (issue #1325):
 * - tile.interleave: interleave elements of two tiles, two outputs
 *   (hardware vintlv; returns TupleType{low, high})
 * - tile.deinterleave: split even/odd elements of two tiles, two outputs
 *   (hardware vdintlv; returns TupleType{even, odd})
 *
 * Both ops take two tiles with identical dtype, shape, and valid_shape and
 * produce two tiles of the same dtype/shape. Supported element widths are
 * 8/16/32-bit (B8/B16/B32).
 */

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

// Shared deducer for tile.interleave / tile.deinterleave: 2 same-typed tile
// inputs -> TupleType{out0, out1} where each output copies the lhs tile type.
static TypePtr DeduceTileInterleaveType(const std::vector<ExprPtr>& args,
                                        const std::vector<std::pair<std::string, std::any>>& kwargs,
                                        const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires 2 arguments (lhs, rhs), but got "
                          << args.size();

  auto lhs_type = As<TileType>(args[0]->GetType());
  CHECK(lhs_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  auto rhs_type = As<TileType>(args[1]->GetType());
  CHECK(rhs_type) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                  << args[1]->GetType()->TypeName();

  CHECK(lhs_type->dtype_ == rhs_type->dtype_)
      << "The operator " << op_name << " requires lhs and rhs dtype to match, but got "
      << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();

  const auto bits = lhs_type->dtype_.GetBit();
  CHECK(bits == 8 || bits == 16 || bits == 32)
      << "The operator " << op_name << " requires 8/16/32-bit element dtype, but got "
      << lhs_type->dtype_.ToString() << " (" << bits << " bits)";

  CHECK(lhs_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 2D tiles, but lhs has rank " << lhs_type->shape_.size();
  CHECK(lhs_type->shape_.size() == rhs_type->shape_.size())
      << "The operator " << op_name << " requires lhs and rhs rank to match, but got "
      << lhs_type->shape_.size() << " and " << rhs_type->shape_.size();
  for (size_t i = 0; i < lhs_type->shape_.size(); ++i) {
    CHECK(DimensionsEqual(lhs_type->shape_[i], rhs_type->shape_[i]))
        << "The operator " << op_name << " requires lhs and rhs shapes to match, but got "
        << FormatShape(lhs_type->shape_) << " and " << FormatShape(rhs_type->shape_);
  }

  auto lhs_valid = GetValidShape(lhs_type);
  auto rhs_valid = GetValidShape(rhs_type);
  CHECK(lhs_valid.size() == rhs_valid.size())
      << "The operator " << op_name << " requires lhs and rhs valid_shape rank to match, but got "
      << lhs_valid.size() << " and " << rhs_valid.size();
  for (size_t i = 0; i < lhs_valid.size(); ++i) {
    CHECK(DimensionsEqual(lhs_valid[i], rhs_valid[i]))
        << "The operator " << op_name << " requires lhs and rhs valid_shape to match, but got "
        << FormatShape(lhs_valid) << " and " << FormatShape(rhs_valid);
  }

  // Both outputs copy lhs shape / dtype / valid_shape and inherit lhs layout.
  std::vector<TypePtr> elements;
  elements.reserve(2);
  for (int i = 0; i < 2; ++i) {
    TileView tile_view;
    tile_view.valid_shape = lhs_valid;
    InheritTileViewLayout(tile_view, lhs_type);
    elements.push_back(
        std::make_shared<TileType>(lhs_type->shape_, lhs_type->dtype_, std::nullopt, tile_view));
  }
  return std::make_shared<TupleType>(std::move(elements));
}

// ============================================================================
// Registration for Interleave Operations
// ============================================================================

REGISTER_OP("tile.interleave")
    .set_op_category("TileOp")
    .set_description(
        "Interleave elements of two same-typed tiles. Returns TupleType{low, high}: "
        "low = lhs0,rhs0,lhs1,rhs1,... over the lower halves, high = same over the upper halves.")
    .add_argument("lhs", "First source tile (8/16/32-bit dtype)")
    .add_argument("rhs", "Second source tile (same dtype/shape/valid_shape as lhs)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    // Output is a TupleType{low, high}; set_output_memory applies Vec to every
    // TileType element inside the TupleType.
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileInterleaveType(args, kwargs, "tile.interleave");
    });

REGISTER_OP("tile.deinterleave")
    .set_op_category("TileOp")
    .set_description(
        "De-interleave elements of two same-typed tiles. Returns TupleType{even, odd}: "
        "even = even-indexed elements of lhs|rhs concat, odd = odd-indexed elements.")
    .add_argument("lhs", "First source tile (8/16/32-bit dtype)")
    .add_argument("rhs", "Second source tile (same dtype/shape/valid_shape as lhs)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileInterleaveType(args, kwargs, "tile.deinterleave");
    });

}  // namespace ir
}  // namespace pypto

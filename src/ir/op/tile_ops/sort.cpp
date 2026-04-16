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
 * @file sort.cpp
 * @brief Sorting tile operations (Sort32, MrgSort)
 *
 * This file implements sort operations for tile-level programming.
 * Sort32 sorts fixed-size 32-element blocks and maps to pto.tsort32.
 * MrgSort merges 4 pre-sorted lists and maps to pto.tmrgsort (format2).
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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

// Helper to get kwargs value with optional default
template <typename T>
static T GetKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
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

TypePtr DeduceTileSort32Type(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires 2 arguments (src, idx), but got "
                          << args.size();

  // First arg: src tile (f16 or f32)
  auto src_type = As<TileType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::FP32)
      << "The operator " << op_name << " requires src dtype to be FP16 or FP32, but got "
      << src_type->dtype_.ToString();

  // Second arg: idx tile
  auto idx_type = As<TileType>(args[1]->GetType());
  CHECK(idx_type) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                  << args[1]->GetType()->TypeName();

  // Build output shape: double the last dimension for value-index pairs
  const auto& input_shape = src_type->shape_;
  CHECK(!input_shape.empty()) << "The operator " << op_name << " requires non-empty input shape";

  std::vector<ExprPtr> output_shape(input_shape.begin(), input_shape.end() - 1);
  auto last_dim = input_shape.back();
  // Try constant evaluation for the common case (sort32 always uses cols=32 -> 64)
  if (auto const_dim = As<ConstInt>(last_dim)) {
    int64_t doubled = const_dim->value_ * 2;
    output_shape.push_back(std::make_shared<ConstInt>(doubled, DataType::INDEX, Span::unknown()));
  } else {
    auto two = std::make_shared<ConstInt>(2, DataType::INDEX, Span::unknown());
    output_shape.push_back(std::make_shared<Mul>(last_dim, two, DataType::INDEX, Span::unknown()));
  }

  TileView tile_view;
  tile_view.valid_shape = output_shape;
  InheritTileViewLayout(tile_view, src_type);
  return std::make_shared<TileType>(output_shape, src_type->dtype_, std::nullopt, tile_view);
}

// ============================================================================
// Registration for Sort Operations
// ============================================================================

REGISTER_OP("tile.sort32")
    .set_op_category("TileOp")
    .set_description("Sort fixed 32-element blocks (maps to pto.tsort32)")
    .add_argument("src", "Input value tile (TileType, f16 or f32)")
    .add_argument("idx", "Input index tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileSort32Type(args, kwargs, "tile.sort32");
    });

// ============================================================================
// MrgSort: 2-4 way merge sort (format2) — maps to pto.tmrgsort
// ============================================================================

TypePtr DeduceTileMrgSortType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs,
                              const std::string& op_name) {
  // Arg layout: (src0, ..., srcN-1, tmp, executed)
  //   2-way: 4 args  (src0, src1, tmp, executed)
  //   3-way: 5 args  (src0, src1, src2, tmp, executed)
  //   4-way: 6 args  (src0, src1, src2, src3, tmp, executed)
  CHECK(args.size() >= 4 && args.size() <= 6)
      << "The operator " << op_name
      << " requires 4-6 arguments (2-4 srcs + tmp + executed), but got " << args.size();

  size_t n_srcs = args.size() - 2;

  // src0: sorted input tile (f16 or f32)
  auto src0_type = As<TileType>(args[0]->GetType());
  CHECK(src0_type) << "The operator " << op_name << " requires argument 0 to be a TileType, but got "
                   << args[0]->GetType()->TypeName();
  CHECK(src0_type->dtype_ == DataType::FP16 || src0_type->dtype_ == DataType::FP32)
      << "The operator " << op_name << " requires src dtype to be FP16 or FP32, but got "
      << src0_type->dtype_.ToString();

  // src1..srcN-1: remaining source tiles — must match src0 dtype
  for (size_t i = 1; i < n_srcs; ++i) {
    auto src_type = As<TileType>(args[i]->GetType());
    CHECK(src_type) << "The operator " << op_name << " requires argument " << i
                    << " to be a TileType, but got " << args[i]->GetType()->TypeName();
    CHECK(src_type->dtype_ == src0_type->dtype_)
        << "The operator " << op_name << " requires all src tiles to have matching dtype, but argument " << i
        << " has " << src_type->dtype_.ToString() << " (expected " << src0_type->dtype_.ToString() << ")";
  }

  // tmp workspace tile (second-to-last arg)
  auto tmp_type = As<TileType>(args[n_srcs]->GetType());
  CHECK(tmp_type) << "The operator " << op_name << " requires argument " << n_srcs
                  << " (tmp) to be a TileType, but got " << args[n_srcs]->GetType()->TypeName();

  // executed status tile (last arg)
  auto exc_type = As<TileType>(args[n_srcs + 1]->GetType());
  CHECK(exc_type) << "The operator " << op_name << " requires argument " << (n_srcs + 1)
                  << " (executed) to be a TileType, but got " << args[n_srcs + 1]->GetType()->TypeName();

  // kwarg: exhausted (bool, default false)
  [[maybe_unused]] bool exhausted = GetKwarg<bool>(kwargs, "exhausted", false);

  // Output shape matches tmp tile (the merge destination buffer)
  TileView tile_view;
  tile_view.valid_shape = tmp_type->shape_;
  InheritTileViewLayout(tile_view, src0_type);
  return std::make_shared<TileType>(tmp_type->shape_, src0_type->dtype_, std::nullopt, tile_view);
}

REGISTER_OP("tile.mrgsort_format2")
    .set_op_category("TileOp")
    .set_description("Merge sort 2-4 sorted lists, format2 (maps to pto.tmrgsort). "
                     "Args: (src0, src1[, src2[, src3]], tmp, executed). "
                     "2-way: 4 args, 3-way: 5 args, 4-way: 6 args.")
    .add_argument("src0", "First sorted input tile (FP16 or FP32)")
    .add_argument("src1", "Second sorted input tile")
    .add_argument("tmp_or_src2", "Third sorted input tile (3/4-way) or tmp workspace (2-way)")
    .add_argument("executed_or_src3", "Fourth sorted input tile (4-way), tmp (3-way), or executed (2-way)")
    .add_argument("tmp", "(3/4-way only) Temporary workspace tile")
    .add_argument("executed", "(4-way only) Exhaustion status output tile")
    .set_attr<bool>("exhausted")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMrgSortType(args, kwargs, "tile.mrgsort_format2");
    });

// ============================================================================
// MrgSort1: single-list merge sort (format1) — maps to pto.tmrgsort
// ============================================================================

TypePtr DeduceTileMrgSort1Type(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires 2 arguments (src, block_len), but got "
                          << args.size();

  // arg0: src tile (f16 or f32)
  auto src_type = As<TileType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires argument 0 to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::FP32)
      << "The operator " << op_name << " requires src dtype to be FP16 or FP32, but got "
      << src_type->dtype_.ToString();

  // arg1: block_len (integer scalar)
  auto block_len_type = As<ScalarType>(args[1]->GetType());
  CHECK(block_len_type) << "The operator " << op_name
                        << " requires argument 1 (block_len) to be a ScalarType, but got "
                        << args[1]->GetType()->TypeName();
  CHECK(block_len_type->dtype_.IsInt())
      << "The operator " << op_name << " requires block_len to be an integer type, but got "
      << block_len_type->dtype_.ToString();

  // Validate constant block_len: must be a positive multiple of 64
  if (auto const_val = As<ConstInt>(args[1])) {
    CHECK(const_val->value_ > 0 && const_val->value_ % 64 == 0)
        << "The operator " << op_name << " requires block_len to be a positive multiple of 64, but got "
        << const_val->value_;
  }

  // Output shape matches src (merge destination has same dimensions as input)
  TileView tile_view;
  tile_view.valid_shape = src_type->shape_;
  InheritTileViewLayout(tile_view, src_type);
  return std::make_shared<TileType>(src_type->shape_, src_type->dtype_, std::nullopt, tile_view);
}

REGISTER_OP("tile.mrgsort_format1")
    .set_op_category("TileOp")
    .set_description("Single-list merge sort, format1 (maps to pto.tmrgsort format1)")
    .add_argument("src", "Input tile containing pre-sorted runs (FP16 or FP32)")
    .add_argument("block_len", "Run length for merge sort (integer scalar, multiple of 64)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMrgSort1Type(args, kwargs, "tile.mrgsort_format1");
    });

}  // namespace ir
}  // namespace pypto

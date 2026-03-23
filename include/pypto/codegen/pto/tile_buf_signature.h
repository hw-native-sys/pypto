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

#ifndef PYPTO_CODEGEN_PTO_TILE_BUF_SIGNATURE_H_
#define PYPTO_CODEGEN_PTO_TILE_BUF_SIGNATURE_H_

#include <cstdint>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

/**
 * @brief PTO-visible typed buffer signature
 *
 * Captures the full set of attributes that PTO alloc_tile uses to declare a
 * buffer: memory space, element type, physical shape, layout, fractal, pad,
 * and valid-shape dimensions.  Two tiles that share the same MemRef (storage
 * slot) are PTO-compatible only when they have either the same signature, or
 * their differences can be materialised via existing PTO view ops
 * (treshape, textract, tfillpad).
 */
struct TileBufSignature {
  ir::MemorySpace memory_space = ir::MemorySpace::Vec;
  DataType dtype = DataType::FP32;
  int64_t rows = 32;
  int64_t cols = 32;
  ir::TileLayout blayout = ir::TileLayout::row_major;
  ir::TileLayout slayout = ir::TileLayout::none_box;
  uint64_t fractal = 512;
  ir::PadValue pad = ir::PadValue::null;
  int64_t v_row = 32;
  int64_t v_col = 32;
  bool v_row_dynamic = false;
  bool v_col_dynamic = false;

  bool operator==(const TileBufSignature& o) const {
    return memory_space == o.memory_space && dtype == o.dtype && rows == o.rows && cols == o.cols &&
           blayout == o.blayout && slayout == o.slayout && fractal == o.fractal && pad == o.pad &&
           v_row == o.v_row && v_col == o.v_col && v_row_dynamic == o.v_row_dynamic &&
           v_col_dynamic == o.v_col_dynamic;
  }
  bool operator!=(const TileBufSignature& o) const { return !(*this == o); }

  /**
   * @brief Check whether two signatures are storage-compatible
   *
   * Storage compatibility means the two can share one MemRef byte-slot.
   * This only checks memory_space equality and that the physical size
   * (rows * cols * dtype_bytes) are equal.
   */
  [[nodiscard]] bool IsStorageCompatible(const TileBufSignature& other) const {
    return memory_space == other.memory_space && dtype == other.dtype && rows == other.rows &&
           cols == other.cols;
  }

  /**
   * @brief Build a signature from a TileType (extracts the same info
   * that PTOCodegen::GetTileBufTypeStringFromTileType uses).
   */
  static TileBufSignature FromTileType(const ir::TileType& tile_type) {
    TileBufSignature sig;
    auto ms = tile_type.GetMemorySpace();
    if (ms.has_value()) sig.memory_space = *ms;
    sig.dtype = tile_type.dtype_;

    if (tile_type.shape_.size() >= 2) {
      if (auto c0 = ir::As<ir::ConstInt>(tile_type.shape_[0])) sig.rows = c0->value_;
      if (auto c1 = ir::As<ir::ConstInt>(tile_type.shape_[1])) sig.cols = c1->value_;
    } else if (tile_type.shape_.size() == 1) {
      if (auto c0 = ir::As<ir::ConstInt>(tile_type.shape_[0])) {
        sig.rows = 1;
        sig.cols = c0->value_;
      }
    }
    sig.v_row = sig.rows;
    sig.v_col = sig.cols;

    if (tile_type.tile_view_.has_value()) {
      const auto& tv = *tile_type.tile_view_;
      sig.blayout = tv.blayout;
      sig.slayout = tv.slayout;
      sig.fractal = tv.fractal;
      sig.pad = tv.pad;
      if (tv.valid_shape.size() >= 1) {
        if (auto c0 = ir::As<ir::ConstInt>(tv.valid_shape[0])) {
          sig.v_row = c0->value_;
        } else if (ir::As<ir::Var>(tv.valid_shape[0])) {
          sig.v_row_dynamic = true;
        }
      }
      if (tv.valid_shape.size() >= 2) {
        if (auto c1 = ir::As<ir::ConstInt>(tv.valid_shape[1])) {
          sig.v_col = c1->value_;
        } else if (ir::As<ir::Var>(tv.valid_shape[1])) {
          sig.v_col_dynamic = true;
        }
      }
    } else if (sig.cols == 1 && sig.rows > 1) {
      sig.blayout = ir::TileLayout::col_major;
    }
    return sig;
  }

  /**
   * @brief The "root" portion of a signature, ignoring view-level fields
   *
   * Two signatures that differ only in fields that PTO view ops can
   * materialise (pad, valid_shape, reshape shape) share the same root.
   * The root keeps: memory_space, dtype, physical rows/cols, blayout,
   * slayout, fractal.
   */
  [[nodiscard]] TileBufSignature RootSignature() const {
    TileBufSignature root = *this;
    root.pad = ir::PadValue::null;
    root.v_row = root.rows;
    root.v_col = root.cols;
    root.v_row_dynamic = false;
    root.v_col_dynamic = false;
    return root;
  }

  /**
   * @brief Check whether differences between this and another signature
   *        can be materialised with existing PTO view ops.
   *
   * Currently recognised materialisable differences:
   *  - pad only (fillpad)
   *  - valid_shape only (load with padding / dynamic valid shape)
   *  - shape only (reshape) — memory_space & dtype must match
   */
  [[nodiscard]] bool IsPTOMaterializable(const TileBufSignature& other) const {
    if (memory_space != other.memory_space || dtype != other.dtype) return false;

    // Same root: differences in pad / valid_shape are view-level
    if (RootSignature() == other.RootSignature()) return true;

    // Different physical shape but same memory_space + dtype → reshape
    // Element count must match to ensure the physical buffer capacity is compatible
    if (blayout == other.blayout && slayout == other.slayout && fractal == other.fractal) {
      const __int128 lhs_elems = static_cast<__int128>(rows) * static_cast<__int128>(cols);
      const __int128 rhs_elems = static_cast<__int128>(other.rows) * static_cast<__int128>(other.cols);
      return lhs_elems == rhs_elems;
    }

    return false;
  }
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_PTO_TILE_BUF_SIGNATURE_H_

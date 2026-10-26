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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_NZ_STORE_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_NZ_STORE_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/phase.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto::ir::nz_store_utils {

// TStoreAccNz2nz encodes the logical row/column extents in 16/12-bit fields.
// The tiler and final emitter must enforce the same per-instruction limits.
inline constexpr int64_t kMaxStoreRows = 65535;
inline constexpr int64_t kMaxStoreCols = 4095;

/// Validate the deliberately narrow NZ writeback contract both before blocking
/// and after later tiling/layout passes. The latter check is essential: a pass
/// can split a previously full-row store into unsupported partial-row stores.
/// Memory-space inference cannot turn an unknown source into evidence of Acc;
/// supported matmul producers already stamp their result with MemorySpace::Acc.
inline std::vector<ExprPtr> CheckSourceAndAttrs(const CallPtr& op, const backend::BackendHandler* handler) {
  INTERNAL_CHECK_SPAN(op->args_.size() == 3 || op->args_.size() == 4, op->span_)
      << "Internal error: tile.store expects (tile, offsets, tensor[, shapes])";
  auto tile = As<TileType>(op->args_[0]->GetType());
  auto tensor = AsTensorTypeLike(op->args_[2]->GetType());
  INTERNAL_CHECK_SPAN(tile && tensor, op->span_)
      << "Internal error: NZ tile.store requires a tile source and tensor destination";
  CHECK_SPAN(tile->dtype_ == DataType::FP32 && tensor->dtype_ == DataType::FP32 &&
                 tile->GetMemorySpace() == MemorySpace::Acc,
             op->span_)
      << "NZ tile.store supports only an FP32 Acc tile into an FP32 NZ tensor. "
      << "Use an FP32 matmul accumulator, or store to pl.ND.";
  CHECK_SPAN(handler && handler->SupportsAccToNzGmStore(), op->span_)
      << "NZ tile.store is supported only on A2/A3. Store to pl.ND on this backend.";

  const TileView view = tile_view_semantics::GetEffectiveTileView(*tile);
  CHECK_SPAN(tile->shape_.size() == 2 && view.valid_shape.size() == 2, op->span_)
      << "NZ tile.store requires a two-dimensional Acc tile.";
  CHECK_SPAN(view.blayout == TileLayout::col_major && view.slayout == TileLayout::row_major &&
                 view.fractal == tile_view_semantics::kAccFractal,
             op->span_)
      << "NZ tile.store requires the ordinary NZ-boxed Acc layout.";
  // With full valid extents and 16-row alignment, CompactMode::normal and
  // CompactMode::null use the same source stride (ceil(validRows/16)*16).
  for (size_t axis = 0; axis < 2; ++axis) {
    auto physical = As<ConstInt>(tile->shape_[axis]);
    auto valid = As<ConstInt>(view.valid_shape[axis]);
    // C0=8 describes the GM bytes, but TStoreAccNz2nz still requires validCol
    // divisible by 16. Every write must retain the full M extent, so its
    // 16-bit row field applies here. The 12-bit column field and L0C capacity
    // apply only to the physical tiles emitted after automatic N tiling.
    CHECK_SPAN(physical && valid && physical->value_ == valid->value_ && valid->value_ > 0 &&
                   valid->value_ % 16 == 0 && (axis != 0 || valid->value_ <= kMaxStoreRows),
               op->span_)
        << "NZ tile.store requires a static, fully valid Acc tile with rows and columns divisible by 16, "
        << "and rows <= " << kMaxStoreRows << ". Partial or dynamic Acc tiles are not supported.";
  }
  CHECK_SPAN(op->GetKwarg<int>("atomic", 0) == static_cast<int>(AtomicType::kNone) &&
                 !GetOptionalDoubleKwarg(op->kwargs_, "pre_quant").has_value() &&
                 !op->GetKwarg<bool>("pre_relu", false),
             op->span_)
      << "NZ tile.store does not support atomic, pre_quant, or pre_relu. Use a plain FP32 writeback.";
  CHECK_SPAN(IsValidSTPhase(op->GetKwarg<int>("st_phase", static_cast<int>(STPhase::kUnspecified))),
             op->span_)
      << "NZ tile.store st_phase must be STPhase.Unspecified or STPhase.Final.";
  return view.valid_shape;
}

/// Validate one concrete write after automatic tiling. Initial NZ blocking
/// must allow an oversized logical matmul result: AutoTileMatmulL0 can replace
/// its sole store with full-row column windows, each backed by a smaller Acc.
/// Recheck at emission so unsupported consumers or later transformations can
/// never bypass the actual instruction and allocation limits.
inline std::vector<ExprPtr> CheckPhysicalSource(const CallPtr& op, const backend::BackendHandler* handler) {
  auto valid = CheckSourceAndAttrs(op, handler);
  auto tile = As<TileType>(op->args_[0]->GetType());
  auto cols = As<ConstInt>(valid[1]);
  INTERNAL_CHECK_SPAN(tile && cols, op->span_)
      << "Internal error: validated NZ tile.store source must have a static TileType";
  CHECK_SPAN(cols->value_ <= kMaxStoreCols, op->span_)
      << "NZ tile.store requires each emitted Acc tile to have columns <= " << kMaxStoreCols << ". "
      << "Use a matmul with a single direct store for automatic N tiling, tile the output columns "
      << "explicitly, or store to pl.ND.";
  const auto bytes = utils::StaticPhysicalAllocationBytes(tile, MemorySpace::Acc, handler);
  INTERNAL_CHECK_SPAN(bytes.has_value(), op->span_)
      << "Internal error: the physical footprint of a validated static NZ store source must be known";
  const uint64_t capacity = handler->GetCapacityBytes(MemorySpace::Acc);
  CHECK_SPAN(*bytes <= capacity, op->span_)
      << "NZ tile.store requires one Acc tile to fit L0C: the source needs " << *bytes
      << " bytes, but this backend's L0C capacity is " << capacity
      << " bytes. NZ writes must cover all destination rows; use a matmul with a single direct store "
      << "for automatic N tiling, tile the output columns explicitly, or store to pl.ND.";
  return valid;
}

/// Recheck the physical partition at the final emission boundary. The ISA
/// computes its NZ destination stride from the window height, ignoring the
/// parent's stride, and folds batch into columns rather than looping batches.
/// Thus a single store must cover all parent rows and exactly one batch panel.
inline void CheckBlockedPartition(const CallPtr& op, const std::vector<ExprPtr>& valid_shape) {
  auto tensor = AsTensorTypeLike(op->args_[2]->GetType());
  auto offsets = As<MakeTuple>(op->args_[1]);
  auto sizes = op->args_.size() == 4 ? As<MakeTuple>(op->args_[3]) : nullptr;
  INTERNAL_CHECK_SPAN(tensor && tensor_view_semantics::IsBlockedNzShape(tensor->shape_, tensor->dtype_) &&
                          offsets && offsets->elements_.size() == tensor_view_semantics::kNzBlockedRank &&
                          sizes && sizes->elements_.size() == tensor_view_semantics::kNzBlockedRank,
                      op->span_)
      << "Internal error: NZ tile.store needs blocked rank-5 offsets and partition sizes";

  auto rows = As<ConstInt>(valid_shape[0]);
  auto cols = As<ConstInt>(valid_shape[1]);
  INTERNAL_CHECK_SPAN(rows && cols, op->span_) << "Internal error: NZ Acc valid shape must be static";
  const std::vector<int64_t> expected = {1, cols->value_ / 8, rows->value_ / 16, 16, 8};
  for (size_t axis = 0; axis < expected.size(); ++axis) {
    auto size = As<ConstInt>(sizes->elements_[axis]);
    CHECK_SPAN(size && size->value_ == expected[axis], op->span_)
        << "NZ tile.store partition must describe exactly one full Acc tile in a single batch panel.";
  }
  auto parent_rows = As<ConstInt>(tensor->shape_[2]);
  auto row_offset = As<ConstInt>(offsets->elements_[2]);
  CHECK_SPAN(parent_rows && parent_rows->value_ == rows->value_ / 16 && row_offset && row_offset->value_ == 0,
             op->span_)
      << "NZ tile.store must cover all rows of its destination at row offset 0: PTO-ISA's Acc NZ store "
      << "uses the partition height as its column-block stride. Stack independent panels on a leading "
      << "batch axis, or store to pl.ND.";
  for (size_t axis = 3; axis < tensor_view_semantics::kNzBlockedRank; ++axis) {
    auto offset = As<ConstInt>(offsets->elements_[axis]);
    INTERNAL_CHECK_SPAN(offset && offset->value_ == 0, op->span_)
        << "Internal error: NZ tile.store offsets must begin at a complete fractal";
  }
}

}  // namespace pypto::ir::nz_store_utils

#endif  // PYPTO_IR_TRANSFORMS_UTILS_NZ_STORE_UTILS_H_

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

#include "pypto/backend/950/backend_950_handler.h"

#include <initializer_list>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

const Ascend950Handler& Ascend950Handler::Instance() {
  static const Ascend950Handler instance;
  return instance;
}

namespace {

bool IsMxScaleStagingView(const ir::TileView& view) {
  // tquant_mx / MX Mat staging use fractal-32 with matching blayout==slayout
  // (row/row for LeftScale, col/col for RightScale). Ordinary Mat NZ is
  // col_major/row_major and must not match this predicate.
  return view.fractal == ir::tile_view_semantics::kMXScaleFractal && view.blayout == view.slayout &&
         (view.blayout == ir::TileLayout::row_major || view.blayout == ir::TileLayout::col_major);
}

}  // namespace

ir::TileView Ascend950Handler::BuildCrossCoreTransferView(ir::MemorySpace dest_ms,
                                                          const ir::TileView& original_view) const {
  // Ascend950 (a5): hardware cross-core pipe carries data in fractal layout.
  //   Left -> NZ (col_major blayout, row_major slayout)
  //   Right -> NZ (A5 V2C inserts Vec tiles into the Mat FIFO via
  //                TINSERT_IMPL<TInsertMode::NZ>, so the bridge tile must
  //                stay NZ rather than ZN)
  //   Mat -> NZ
  //   Vec -> preserve the caller-requested final layout
  //
  // Exception: MX E8M0 scale staging tiles keep their row/row/32 or
  // col/col/32 view. Forcing Mat NZ here makes TMov_mx into LeftScale /
  // RightScale fail (`TMov_mx: SrcTile Invalid Fractal`).
  if (IsMxScaleStagingView(original_view)) {
    return original_view;
  }
  ir::TileView result = original_view;
  switch (dest_ms) {
    case ir::MemorySpace::Left:
    case ir::MemorySpace::Right:
    case ir::MemorySpace::Mat:
      result.blayout = ir::TileLayout::col_major;
      result.slayout = ir::TileLayout::row_major;
      return result;
    case ir::MemorySpace::Vec:
      return original_view;
    default:
      INTERNAL_UNREACHABLE << "cross-core move destination must be Vec, Mat, Left, or Right, got "
                           << static_cast<int>(dest_ms);
  }
}

// ISA Supported Conversions (pto-isa tcvt docs, a5 column). a5 adds the FP8 /
// HF8 / FP4 formats and the unsigned integer widths, but drops a2a3's
// INT32 -> FP16 deq and every INT4 pair -- which is why pl.cast(x_i32, pl.FP16)
// has to be legalized here. Grouped by source, mirroring the ISA table.
const TcvtAdjacency& Ascend950Handler::GetTcvtAdjacency() const {
  static const TcvtAdjacency kTable = [] {
    TcvtAdjacency t;
    auto add = [&t](DataType from, std::initializer_list<DataType> tos) {
      for (DataType to : tos) {
        t.edges.emplace_back(from, to);
      }
    };
    add(DataType::FP32, {DataType::FP16, DataType::BF16, DataType::INT16, DataType::INT32, DataType::INT64});
    add(DataType::FP32, {DataType::FP8E4M3FN, DataType::FP8E5M2, DataType::HF8});
    add(DataType::FP16, {DataType::FP32, DataType::INT32, DataType::INT16, DataType::INT8, DataType::UINT8});
    add(DataType::FP16, {DataType::HF8});
    add(DataType::BF16, {DataType::FP32, DataType::INT32, DataType::FP16, DataType::FP4});
    add(DataType::INT16, {DataType::FP16, DataType::FP32, DataType::UINT8, DataType::UINT32});
    add(DataType::INT16, {DataType::INT32});
    add(DataType::INT32, {DataType::FP32, DataType::INT16, DataType::INT64, DataType::UINT16});
    add(DataType::INT32, {DataType::UINT8});
    add(DataType::INT64, {DataType::FP32, DataType::INT32});
    add(DataType::UINT8, {DataType::FP16, DataType::UINT16});
    add(DataType::INT8, {DataType::FP16, DataType::INT16, DataType::INT32});
    add(DataType::UINT32, {DataType::UINT8, DataType::UINT16, DataType::INT16});
    add(DataType::FP8E4M3FN, {DataType::FP32});
    add(DataType::FP8E5M2, {DataType::FP32});
    add(DataType::HF8, {DataType::FP32});
    add(DataType::FP4, {DataType::BF16});
    return t;
  }();
  return kTable;
}

}  // namespace backend
}  // namespace pypto

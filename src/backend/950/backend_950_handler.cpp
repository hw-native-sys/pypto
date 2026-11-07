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

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <utility>

#include "pypto/backend/950/backend_950.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/cast_rules.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"
#include "pypto/sync/resource_catalog.h"
#include "src/backend/common/sync_resources.h"

namespace pypto {
namespace backend {

Ascend950Handler::Ascend950Handler() : BackendHandler(Backend950::Instance()) {}

bool Ascend950Handler::SupportsRegType(const ir::RegType& type) const {
  if (type.shape_ <= 0) return false;
  if (type.kind_ == ir::RegKind::MaskReg) return type.dtype_ == DataType::BOOL;
  const auto dtype = type.dtype_;
  return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32 ||
         dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
         dtype == DataType::UINT16 || dtype == DataType::INT32 || dtype == DataType::UINT32;
}

const Ascend950Handler& Ascend950Handler::Instance() {
  static const Ascend950Handler instance;
  return instance;
}

/// Own the A5 resource model independently of the process-global target.
const sync::ResourceCatalog& Ascend950Handler::GetSyncResourceCatalog() const {
  // PTOAS UnifiedSyncModel::ResourceModel exposes 32 AIV buffer credits on A5.
  static const auto catalog = MakeAscendSyncResourceCatalog(GetPtoTargetArch(), 32);
  return catalog;
}

ir::TileView Ascend950Handler::BuildCrossCoreTransferView(ir::MemorySpace dest_ms,
                                                          const ir::TileView& original_view) const {
  // Ascend950 (a5): hardware cross-core pipe carries data in fractal layout.
  //   Left -> NZ (col_major blayout, row_major slayout)
  //   Right -> NZ (A5 V2C inserts Vec tiles into the Mat FIFO via
  //                TINSERT_IMPL<TInsertMode::NZ>, so the bridge tile must
  //                stay NZ rather than ZN)
  //   Mat -> NZ
  //   Vec -> preserve the caller-requested final layout
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
        TcvtRule rule{from, to, {}};
        if (from.IsPackedFp4() || to.IsPackedFp4()) {
          TcvtConditions nd;
          nd.source.blayout = ir::TileLayout::row_major;
          nd.source.slayout = ir::TileLayout::none_box;
          nd.result = nd.source;
          nd.matching_valid_shape = true;
          TcvtConditions dn = nd;
          dn.source.blayout = ir::TileLayout::col_major;
          dn.source.static_shape = true;
          dn.source.full_valid = true;
          dn.source.dense_stride = true;
          dn.source.zero_start_offset = true;
          dn.result = dn.source;
          dn.preserve_input_layout = true;
          dn.preserve_output_layout = true;
          dn.descriptor = CastDescriptor::StaticPlannedAddress;
          rule.alternatives = {nd, dn};
        }
        if (from == DataType::FP4E2M1X2 && to == DataType::FP8E4M3FN) {
          // PTOAS v0.72 direct packed decode (PTO_IR_manual.md "f4E2M1x2 ->
          // f8E4M3FN"). All values/strides count byte carriers on the packed
          // side. The source is always ND; the destination is either ND or the
          // standard 512-byte NZ box grid, and PTOAS accepts no other boxing.
          TcvtConditions direct;
          direct.source.blayout = ir::TileLayout::row_major;
          direct.source.slayout = ir::TileLayout::none_box;
          direct.source.null_compact = true;
          direct.source.static_shape = true;
          direct.source.static_valid = true;
          direct.source.dense_stride = true;
          direct.source.zero_start_offset = true;
          direct.source.vec_only = true;
          direct.source.max_rows = std::numeric_limits<uint16_t>::max();
          direct.source.max_columns = std::numeric_limits<int32_t>::max();
          direct.source.max_byte_span = std::numeric_limits<int32_t>::max();
          direct.result = direct.source;
          direct.matching_valid_shape = true;
          direct.allow_empty_replay = true;
          direct.source_col_alignment = 32;
          direct.source_register_bytes = 256;
          direct.max_source_registers = std::numeric_limits<uint16_t>::max();
          direct.descriptor = CastDescriptor::StaticValidShape;

          // NZ destination: same decode, boxed result. `dense_stride` and
          // `zero_start_offset` do not apply -- HasCanonicalDenseStrides is
          // defined for none_box views only, and a boxed tile's addressing is
          // the fractal grid rather than a 2D pitch. The 16-row granularity is
          // the ISA's own requirement on the *physical* rows, which also keeps
          // the tile a whole number of boxes so its byte footprint stays the
          // plain row*col product the allocator already computes.
          TcvtConditions direct_nz = direct;
          direct_nz.result.blayout = ir::TileLayout::col_major;
          direct_nz.result.slayout = ir::TileLayout::row_major;
          direct_nz.result.fractal = ir::tile_view_semantics::kStandardNzFractalBytes;
          direct_nz.result.dense_stride = false;
          direct_nz.result.zero_start_offset = false;
          direct_nz.result.row_alignment = ir::tile_view_semantics::kNzFractalRows;
          // The native write already lands in NZ, so the backend's row-major
          // result repair must not wrap it in a relayout tile.move. The ND
          // *source* requirement stands, so input repair is left enabled.
          direct_nz.preserve_output_layout = true;
          rule.alternatives = {direct, direct_nz};
        }
        t.edges.push_back(std::move(rule));
      }
    };
    add(DataType::FP32, {DataType::FP16, DataType::BF16, DataType::INT16, DataType::INT32, DataType::INT64});
    add(DataType::FP32, {DataType::FP8E4M3FN, DataType::FP8E5M2, DataType::HF8});
    add(DataType::FP16, {DataType::FP32, DataType::INT32, DataType::INT16, DataType::INT8, DataType::UINT8});
    add(DataType::FP16, {DataType::HF8});
    add(DataType::BF16,
        {DataType::FP32, DataType::INT32, DataType::FP16, DataType::FP4E2M1X2, DataType::FP4E1M2X2});
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
    add(DataType::FP4E2M1X2, {DataType::BF16, DataType::FP8E4M3FN});
    add(DataType::FP4E1M2X2, {DataType::BF16});
    return t;
  }();
  return kTable;
}

}  // namespace backend
}  // namespace pypto

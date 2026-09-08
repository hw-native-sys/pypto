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

#ifndef PYPTO_CODEGEN_PTO_PTO_TYPE_UTILS_H_
#define PYPTO_CODEGEN_PTO_PTO_TYPE_UTILS_H_

#include <cstdint>
#include <string>

#include "pypto/core/dtype.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

/// Convert DataType to MLIR type string (e.g., FP32 -> "f32", INT32 -> "i32")
std::string DataTypeToMLIR(DataType dtype);

/// Convert MemorySpace to PTO address space string (e.g., Vec -> "vec", DDR -> "gm")
std::string MemorySpaceToMLIR(ir::MemorySpace space);

/// Format a `!pto.local_array<NxT>` type string for an on-core ArrayType.
///
/// Mirrors PTOAS's `!pto.local_array<shape x elementType>` (the
/// `LocalArrayType` def in the PTOAS `PTOTypeDefs.td`). The extent must be a compile-time
/// `ConstInt` and the element dtype must be a scalar integer/float — the same
/// `ArrayType` v1 constraints enforced by its constructor.
std::string FormatLocalArrayTypeString(const ir::ArrayType& array_type);

/// Convert TileLayout to its string name (e.g., row_major -> "row_major")
const char* TileLayoutToStr(ir::TileLayout layout);

/// Format a complete !pto.tile_buf<...> type string from individual components.
/// v_row/v_col are the valid shape dimensions (may differ from rows/cols).
std::string FormatTileBufTypeString(const std::string& loc, const std::string& dtype_str, int64_t rows,
                                    int64_t cols, ir::TileLayout blayout, ir::TileLayout slayout,
                                    uint64_t fractal, ir::PadValue pad, ir::CompactMode compact,
                                    int64_t v_row, int64_t v_col, bool v_row_dynamic = false,
                                    bool v_col_dynamic = false);

/// The slot-count bounds ptoas's `!pto.multi_tile_buf` verifier enforces
/// (`MAX_MULTI_BUFFER_NUM = 16`). Under the PTOAS memory planner a declaration
/// outside them is rejected — falling back to one alloc per slot would let ptoas
/// plan the slots on top of each other. The ordinary one-alloc-per-slot lowering
/// is the PyPTO planner's path, where the baked addresses keep them apart.
inline constexpr uint64_t kMinMultiTileBufSlots = 2;
inline constexpr uint64_t kMaxMultiTileBufSlots = 16;

/// Wrap a single-slot `!pto.tile_buf<...>` into ptoas's N-slot container type,
/// `!pto.multi_tile_buf<<slot>, count=N>` — the result of `pto.alloc_multi_tile`
/// and the operand of `pto.multi_tile_get`. `count` must be within ptoas's
/// `[2, 16]`; callers gate on that before reaching here.
std::string FormatMultiTileBufTypeString(const std::string& slot_type_str, uint64_t count);

/// Intermediate result holder for ExtractTileTypeInfo.
struct TileTypeComponents {
  std::string dtype_str = "f32";
  int64_t rows = 32;
  int64_t cols = 32;
  ir::TileLayout blayout = ir::TileLayout::row_major;
  ir::TileLayout slayout = ir::TileLayout::none_box;
  uint64_t fractal = 512;
  ir::PadValue pad = ir::PadValue::null;
  ir::CompactMode compact = ir::CompactMode::null;
  int64_t v_row = 32;
  int64_t v_col = 32;
  bool v_row_dynamic = false;
  bool v_col_dynamic = false;
};

/// Extract dtype, shape, and layout from a TileType into a TileTypeComponents struct.
///
/// `v_row_dynamic` / `v_col_dynamic` are always set when the corresponding rank is
/// present, so the resulting `!pto.tile_buf<...>` type string always reads
/// `v_row=?, v_col=?`. The actual extents are conveyed via the `valid_row` /
/// `valid_col` operands on `pto.alloc_tile` (see ComputeAllocTileFields).
///
/// @param dtype_str_override Optional override for the dtype string (e.g.,
///                           PTOCodegen::GetTypeString); empty falls back to
///                           DataTypeToMLIR(tile_type.dtype_).
/// Reject a boxed tile whose physical extent is not a whole number of fractal
/// boxes, before the `pto.alloc_tile` that declares it reaches PTOAS.
///
/// PTO addresses a boxed tile one box at a time, so a partial box has no
/// address at all. PTOAS enforces that on the op it receives, but its message
/// (``'pto.alloc_tile' op expects result boxed tile rows to be a multiple of
/// innerRows (16)``) names PTOAS internals, points at whichever line the
/// location happened to carry, and offers no remedy. Checking the same rule
/// where PyPTO emits the allocation reports the tile, the axis, the extent to
/// reach, and how to reach it.
///
/// The rule mirrors PTOAS' ``verifyBoxedTileLayout`` exactly: the box is 16x16
/// for the fractal-1024 accumulator, and ``16 x (32 / sizeof(dtype))`` for a
/// fractal-512 Mat/Left/Right tile (transposed on a ``col_major`` scatter
/// layout). A non-boxed (``none_box``) layout, the MX-scale fractal, a
/// sub-byte carrier, a single-row tile, and Vec's row axis are all exempt —
/// the same exemptions PTOAS applies.
///
/// A boxed tile's physical extents must be static by the time it is emitted —
/// ``InitMemRef`` refuses a dynamic ``TileType::shape_`` outright, telling the
/// author to put the runtime extent in ``TileView`` instead. That matters here
/// because ``ExtractTileTypeInfo`` substitutes its struct default for a
/// non-``ConstInt`` dimension, so a dynamic extent would be checked (and
/// rendered) as a placeholder rather than as itself. The assumption is asserted
/// rather than assumed silently.
///
/// @param tile_type  Tile being allocated; supplies the dtype, the memory space
///                   (Vec relaxes the row rule) and the physical shape.
/// @param components Rendered tile geometry, as it will appear in the emitted
///                   ``!pto.tile_buf<...>`` type string.
/// @param span       IR location reported on failure; may be null when the
///                   emitter has no statement in scope.
void CheckBoxedTileExtents(const ir::TileType& tile_type, const TileTypeComponents& components,
                           const ir::Span* span);

/// Reject an unboxed (``none_box``) tile whose contiguous axis is not a whole
/// number of 32-byte units, which PTO cannot address.
///
/// The complement of ``CheckBoxedTileExtents``: a boxed tile is addressed one
/// fractal box at a time, an unboxed one as a flat run of bytes whose stride
/// along the contiguous axis PTO takes in 32-byte units. ``blayout`` picks that
/// axis -- ``row_major`` makes it the columns, ``col_major`` the rows -- so an
/// FP32 tile needs 8 elements there, and ``[1, 1]``, ``[2, 2]`` and ``[4, 4]``
/// are all equally unallocatable.
///
/// PTOAS does diagnose this, but names its own type internals and points at
/// whichever line the location happened to carry; reporting it here gives the
/// shape the author wrote, the axis at fault, and a remedy that compiles.
///
/// @param tile_type  Tile being allocated; supplies the dtype and memory space.
/// @param components Rendered tile geometry, as it will appear in the emitted
///                   ``!pto.tile_buf<...>`` type string.
/// @param span       IR location reported on failure; may be null when the
///                   emitter has no statement in scope.
void CheckFlatTileExtents(const ir::TileType& tile_type, const TileTypeComponents& components,
                          const ir::Span* span);

TileTypeComponents ExtractTileTypeInfo(const ir::TileType& tile_type,
                                       const std::string& dtype_str_override = "");

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_PTO_PTO_TYPE_UTILS_H_

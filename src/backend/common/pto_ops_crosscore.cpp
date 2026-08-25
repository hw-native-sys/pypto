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
 * @file pto_ops_crosscore.cpp
 * @brief PTO codegen registration for cross-core (TPUSH/TPOP/TFREE/pipe) ops.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "src/backend/common/pto_ops_internal.h"

namespace pypto {
namespace backend {

using ir::As;
using ir::AsTensorTypeLike;
using ir::AsVarLike;
using ir::CallPtr;
using ir::ExprPtr;
using ir::ScalarType;
using ir::TensorType;
using ir::Var;

using pto_ops_detail::AsPto;
using pto_ops_detail::CheckSafeIdentifier;
using pto_ops_detail::EmitIndexOperand;

static bool IsSameDimExpr(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (lhs == rhs) {
    return true;
  }
  auto lhs_const = As<ir::ConstInt>(lhs);
  auto rhs_const = As<ir::ConstInt>(rhs);
  return lhs_const && rhs_const && lhs_const->value_ == rhs_const->value_;
}

static std::shared_ptr<const ir::TileType> GetTpushTileType(const ExprPtr& tile_expr) {
  if (auto tile_type = ir::GetTileTypeWithMemRef(tile_expr->GetType())) {
    return tile_type;
  }
  return As<ir::TileType>(tile_expr->GetType());
}

static bool EmitTpushTransportValidShape(const char* target, const CallPtr& op, codegen::PTOCodegen& codegen,
                                         const std::string& tile_buf, const std::string& tile_type,
                                         int split) {
  // split == 0 normally means no cross-core split. Two physical FIFO cases
  // still require a box-normalized transport: partial Acc->Vec payloads and
  // the 910B no-split dual-AIV dispatch path. The latter
  // (function attr `dual_aiv_dispatch`) runs the producer on TWO AIV subblocks
  // that share one FIFO slot while the single cube consumer pops the FULL
  // slot. If the producer narrowed its valid_shape (e.g. set_validshape on a
  // partial attention block), the un-narrowed rows/cols of the slot stay stale
  // and feed garbage into the consumer's matmul. So for that mode we must
  // still transport the full box, exactly as for split==1/2 — this extends
  // PR #1454's fix to the split==0 dual-dispatch case.
  auto source_tile_type = GetTpushTileType(op->args_[0]);
  if (!source_tile_type || source_tile_type->shape_.size() < 2) {
    return false;
  }
  const bool dual_aiv_no_split = (split == 0) && codegen.IsDualAivDispatchFunction();
  const bool acc_to_vec_no_split = (split == 0) && std::string_view(target) == "aiv" &&
                                   source_tile_type->GetMemorySpace() == ir::MemorySpace::Acc;
  if ((split == 0 && !dual_aiv_no_split && !acc_to_vec_no_split) || tile_buf.empty() || tile_type.empty()) {
    return false;
  }

  const auto tile_view = ir::tile_view_semantics::GetEffectiveTileView(*source_tile_type);
  if (tile_view.valid_shape.size() < 2) {
    return false;
  }

  // The transport must carry the full box for both subblocks to receive
  // complete data: each subblock reads its half of the slot regardless of the
  // user-declared valid_shape. Narrowing valid_shape on the producer side
  // before tpush would leave the non-split axis under-written and (on LR
  // splits in particular) make even subblock 0 see zeros for the cells the
  // producer skipped. Localization back to the user's logical valid happens
  // on the consumer side via LocalizeValidDimForSplit.
  const auto& shape = source_tile_type->shape_;
  const auto& valid_shape = tile_view.valid_shape;
  ExprPtr transport_row = shape[0];
  ExprPtr transport_col = shape[1];

  // A no-split Acc->Vec FIFO slot is laid out using the source's physical box,
  // so a partially written COLUMN range leaves stale bytes *inside* the valid
  // rows -- those columns must be transported at the full box width. An empty
  // tile remains an empty protocol operation and must not be widened at all.
  //
  // The ROW extent is the opposite: it must stay exactly as the producer wrote
  // it. Every L0C reader derives its source pitch from validRow --
  // `ceil(validRow/16)*16` for a compact tile, `TileData::Rows` otherwise
  // (pto-isa `tstore_common.hpp`, `TStoreAccNz2nd`) -- and `mad` laid the
  // result out at the pitch implied by the L0A operand's *valid* rows
  // (`TMatmul.hpp`: `uint16_t m = aMatrix.GetValidRow()`). Widening the rows
  // here re-derives that pitch from the physical box, so TPUSH walks L0C at a
  // stride `mad` never wrote at: with a 64-row box valid to 16 the push picks
  // up N-fractal 4j for every fractal j, silently corrupting the valid rows
  // (issue #2510). Rows past validRow stay stale in the slot, which is exactly
  // what a narrowed valid_shape already promises about its invalid region --
  // and the transport moves validRow rows instead of the whole box.
  if (acc_to_vec_no_split) {
    for (const auto& valid_dim : valid_shape) {
      if (auto dim_const = As<ir::ConstInt>(valid_dim); dim_const && dim_const->value_ == 0) {
        return false;
      }
    }
    transport_row = valid_shape[0];
  }

  // For the 910B no-split dual-AIV path there is NO genuine cross-core row
  // split: subblock 0 runs the full computation while subblock 1 is a
  // pipe-balancing replay whose tile carries valid_shape (0, 0). So here we
  // widen the COLUMNS only -- carrying the producer's fillpad'd cols >=
  // valid_col, which fixes the stale-col feed into the consumer matmul --
  // while PRESERVING the producer's row valid_shape. Widening the rows to the
  // full box would push subblock-1's garbage rows into the shared FIFO slot
  // and race/overwrite subblock-0's real data. Genuine split==1/2 paths keep
  // widening both axes because the row split is real there.
  if (dual_aiv_no_split) {
    transport_row = valid_shape[0];
    // A statically-zero-row producer IS the subblock-1 pipe-balancing replay:
    // it moves no data regardless of the column box, so a col-widening
    // transport is pure overhead AND (on 910B) perturbs the shared-slot
    // dual-AIV merge -- emitting it regressed the cross_core_v2c_nosplit
    // golden. Only the real subblock-0 push (non-zero rows, possibly narrowed
    // by set_validshape) needs the full-column transport.
    if (auto row_const = As<ir::ConstInt>(transport_row); row_const && row_const->value_ == 0) {
      return false;
    }
  }

  // A genuine row split needs the full box in the slot -- lane 1 reads the band
  // starting at the box half, which only exists if the producer wrote it -- but
  // writing it means reading L0C at the physical pitch, which is not the pitch
  // `mad` used for a row-narrowed operand. The two requirements are mutually
  // exclusive, so the shape is refused here instead of being lowered into
  // silently skewed data (measured: a 64-row box valid to 16 across
  // `pl.split(UP_DOWN)` returns 1808 of 8192 elements wrong). Gated on the
  // pitches actually differing, so a single-fractal-block accumulator -- where
  // `ceil(validRow/16)*16 == Rows` -- keeps crossing as before.
  CHECK_SPAN(!(split != 0 && tile_view.compact == ir::CompactMode::normal) ||
                 ir::AccPitchesCoincide(valid_shape[0], shape[0]),
             op->span_)
      << "a row-narrowed matmul accumulator cannot cross a split Cube-to-Vector boundary: mad wrote "
      << "L0C at the pitch implied by the matmul's valid rows, while a split transport must carry "
      << "the full physical box so both vector lanes receive their band. Either drop the row "
      << "narrowing on the matmul's left operand (narrow the result with pl.set_validshape "
      << "instead), or route the accumulator through GM and dequantize it in a second scope.";

  if (IsSameDimExpr(transport_row, valid_shape[0]) && IsSameDimExpr(transport_col, valid_shape[1])) {
    return false;
  }

  std::string row = EmitIndexOperand(codegen, transport_row, "tpush transport valid_row");
  std::string col = EmitIndexOperand(codegen, transport_col, "tpush transport valid_col");
  codegen.Emit("pto.set_validshape " + tile_buf + ", " + row + ", " + col + " : " + tile_type);
  return true;
}

static void EmitLogicalTpushValidShapeRestore(const CallPtr& op, codegen::PTOCodegen& codegen,
                                              const std::string& tile_buf, const std::string& tile_type) {
  auto source_tile_type = GetTpushTileType(op->args_[0]);
  INTERNAL_CHECK(source_tile_type) << "Internal error: tpush validShape restore requires a TileType source";
  const auto tile_view = ir::tile_view_semantics::GetEffectiveTileView(*source_tile_type);
  INTERNAL_CHECK(tile_view.valid_shape.size() >= 2)
      << "Internal error: tpush validShape restore requires rank-2 validShape";
  const auto& valid_shape = tile_view.valid_shape;
  std::string row = EmitIndexOperand(codegen, valid_shape[0], "tpush logical valid_row");
  std::string col = EmitIndexOperand(codegen, valid_shape[1], "tpush logical valid_col");
  codegen.Emit("pto.set_validshape " + tile_buf + ", " + row + ", " + col + " : " + tile_type);
}

static std::string FormatFrontendPipeAttrs(const CallPtr& op, int split) {
  std::ostringstream oss;
  oss << "{";
  if (op->HasKwarg("id")) {
    const int id = op->GetKwarg<int>("id", 0);
    CHECK(id >= 0) << "Frontend pipe 'id' attribute must be non-negative, got " << id;
    oss << "id = " << id << ", ";
  }
  oss << "split = " << split << "}";
  return oss.str();
}

static std::string FormatInitializePipeAttrs(const CallPtr& op, int dir_mask, int slot_size) {
  std::ostringstream oss;
  oss << "{";
  if (op->HasKwarg("id")) {
    const int id = op->GetKwarg<int>("id", 0);
    CHECK(id >= 0) << "Frontend initialize_pipe 'id' attribute must be non-negative, got " << id;
    oss << "id = " << id << ", ";
  }
  oss << "dir_mask = " << dir_mask << ", slot_size = " << slot_size;
  if (op->HasKwarg("slot_num")) {
    const int slot_num = op->GetKwarg<int>("slot_num", 0);
    CHECK(slot_num > 0) << "Frontend initialize_pipe 'slot_num' attribute must be positive, got " << slot_num;
    oss << ", slot_num = " << slot_num;
  }
  if (op->HasKwarg("local_slot_num")) {
    const int local_slot_num = op->GetKwarg<int>("local_slot_num", 0);
    CHECK(local_slot_num > 0) << "Frontend initialize_pipe 'local_slot_num' attribute must be positive, got "
                              << local_slot_num;
    oss << ", local_slot_num = " << local_slot_num;
  }
  oss << "}";
  return oss.str();
}

// tile.tpush_to_{aic,aiv}: Push a tile across cores (Cube<->Vector). `target`
// is "aic" or "aiv" and selects both the emitted pto op name and diagnostics.
static std::string MakeTpushCodegenPTO(const char* target, const CallPtr& op,
                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  const std::string op_name = std::string("tpush_to_") + target;

  INTERNAL_CHECK_SPAN(op->args_.size() == 1, op->span_)
      << op_name << " requires 1 argument (tile), got " << op->args_.size();
  auto tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK_SPAN(tile, op->span_) << op_name << " first argument must be a Var or IterArg";

  const int split = op->GetKwarg<int>("split", -1);
  CHECK(ir::IsValidSplitCode(split))
      << op_name
      << " requires 'split' attribute (0=none, 1=up-down, 2=left-right, 3=up-down/odd, "
         "4=left-right/odd), got "
      << split;
  CHECK(!(ir::IsOddSplitCode(split) && std::string_view(target) == "aic"))
      << op_name << ": pto-isa has no odd split for the Vector -> Cube direction (split = " << split
      << "). Only the Cube -> Vector shard splits an odd axis.";

  std::string tile_buf = codegen.GetExprAsCode(op->args_[0]);
  std::string tile_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  const bool restore_valid_shape =
      EmitTpushTransportValidShape(target, op, codegen, tile_buf, tile_type, split);

  std::ostringstream oss;
  oss << "pto.tpush_to_" << target << "(" << tile_buf;
  if (!tile_type.empty()) {
    oss << " : " << tile_type;
  }
  oss << ") " << FormatFrontendPipeAttrs(op, split);
  codegen.Emit(oss.str());
  if (restore_valid_shape) {
    EmitLogicalTpushValidShapeRestore(op, codegen, tile_buf, tile_type);
  }

  return "";
}

// tile.tpop_from_{aic,aiv}: Pop a tile another core pushed. `target` is "aic"
// (from Cube into Vector) or "aiv" (from Vector into Cube).
static std::string MakeTpopCodegenPTO(const char* target, const CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  const std::string op_name = std::string("tpop_from_") + target;

  INTERNAL_CHECK_SPAN(op->args_.size() == 0, op->span_)
      << op_name << " takes no arguments, got " << op->args_.size();

  const int split = op->GetKwarg<int>("split", 0);
  CHECK(ir::IsValidSplitCode(split))
      << op_name
      << " requires 'split' attribute (0=none, 1=up-down, 2=left-right, 3=up-down/odd, "
         "4=left-right/odd), got "
      << split;
  CHECK(!(ir::IsOddSplitCode(split) && std::string_view(target) == "aiv"))
      << op_name << ": pto-isa has no odd split for the Vector -> Cube direction (split = " << split
      << "). Only the Cube -> Vector shard splits an odd axis.";

  std::string result_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK_SPAN(!result_buf.empty(), op->span_) << op_name << " requires assignment target (tile_buf)";
  std::string result_type = codegen.GetCurrentResultTileBufTypeString();
  auto [logical_row, logical_col] = codegen.GetCurrentResultTpopValidShapeOperands();

  // PTO's Cube->Vector FIFO copies the physical consumer box, at EVERY split.
  // pto-isa builds the GM slot view from the popped tile's compile-time
  // rows/cols and the producer's box row pitch (a2a3 TPush.hpp
  // popVecTileFromGMFiFo: gmValidR/gmValidC = ConsM/ConsN, gmStrideR = ProdN),
  // then strides that view with the tile's RUNTIME validCol (TLoadGm2ubNd2nd:
  // lenBurst = validCol, but gmGap = gStride3 - gShape4). A narrowed validCol
  // therefore collapses the GM gap to zero and the pop reads one contiguous
  // run instead of one box row per burst -- silently, since the matching
  // PTO_ASSERT(validCol == gShape4) is compiled out in release builds. Give
  // TPOP the full-box extent, mirroring the producer-side widening in
  // EmitTpushTransportValidShape, then restore the logical result shape before
  // any vector consumer sees it. Keep a statically empty pop empty: the
  // dual-AIV no-split path uses such a replay only to balance the pipe, and a
  // split lane whose localized valid extent is statically 0 must likewise
  // move nothing.
  auto result_tile_type = codegen.GetCurrentResultTileType();
  bool statically_empty = false;
  bool has_static_logical_shape = false;
  if (result_tile_type) {
    const auto result_view = ir::tile_view_semantics::GetEffectiveTileView(*result_tile_type);
    has_static_logical_shape = result_view.valid_shape.size() >= 2;
    for (const auto& valid_dim : result_view.valid_shape) {
      auto dim_const = As<ir::ConstInt>(valid_dim);
      has_static_logical_shape = has_static_logical_shape && static_cast<bool>(dim_const);
      if (dim_const && dim_const->value_ == 0) {
        statically_empty = true;
      }
    }
  }
  // The physical box must be static too, not just the logical extent: the
  // treshape below rebuilds the logical type from ConstInt dims, and a split
  // tile CAN carry a dynamic physical extent (ReshapeSplitAxis lowers a dynamic
  // split axis to floordiv(dim, 2)). Without this the pop would reach that
  // INTERNAL_CHECK instead of falling back to the direct-TPOP path.
  // An ODD split is the one case that must NOT take the full-box path: the two
  // lanes pop different extents (ceil / floor), and pto-isa derives lane 1's
  // slot offset from the popped tile's own RUNTIME valid extents
  // (TILE_UP_DOWN_ODD: subAIVOffset = subBlockId * (validRow + subBlockId) *
  // validCol). Widening either lane to the box would place lane 1 one cell past
  // the producer's band. Such a tile carries a per-lane (non-static) valid
  // extent anyway, so this only makes the requirement explicit.
  const bool use_full_box =
      std::string_view(target) == "aic" && !ir::IsOddSplitCode(split) && !logical_row.empty() &&
      !logical_col.empty() && has_static_logical_shape && !statically_empty && result_tile_type &&
      result_tile_type->GetMemorySpace() == ir::MemorySpace::Vec && result_tile_type->shape_.size() >= 2 &&
      As<ir::ConstInt>(result_tile_type->shape_[0]) && As<ir::ConstInt>(result_tile_type->shape_[1]);
  // PTOAS enforces the same contract from the other side ("expects odd C2V
  // split tpop to provide per-sub-core valid_row and valid_col operands"), so
  // fail here with the op's own span rather than in the assembler.
  INTERNAL_CHECK_SPAN(!ir::IsOddSplitCode(split) || (!logical_row.empty() && !logical_col.empty()), op->span_)
      << "Internal error: " << op_name
      << " with an odd split must carry per-lane valid_row / valid_col operands; the split-axis "
         "localization in LowerAutoVectorSplit produces them";
  std::string transport_row = logical_row;
  std::string transport_col = logical_col;
  if (use_full_box) {
    transport_row = EmitIndexOperand(codegen, result_tile_type->shape_[0], "tpop transport row");
    transport_col = EmitIndexOperand(codegen, result_tile_type->shape_[1], "tpop transport col");
  }

  // A frontend tpop result is not itself a locally bound tile in PTOAS, so
  // pto.set_validshape cannot mutate it directly. When transport uses the
  // full physical box, pop into a temporary and expose the logical result via
  // pto.treshape. PTOAS has a dedicated zero-copy treshape-over-tpop lowering
  // that preserves the FIFO tile handle while rebuilding its static valid
  // metadata. A full-box pto.subview is not equivalent here: PTOAS lowers that
  // path through a disconnected tile handle on A2/A3.
  //
  // Treshape carries no valid-row/valid-col operands, so it can restore only
  // static logical extents. Keep the existing direct TPOP behavior for dynamic
  // valid shapes until PTO exposes a dynamic metadata rebind for pipe entries.
  std::string transport_buf = use_full_box ? codegen.NewNamedTemp("tpop_transport") : result_buf;
  std::ostringstream oss;
  oss << transport_buf << " = pto.tpop_from_" << target;
  if (!transport_row.empty() || !transport_col.empty()) {
    INTERNAL_CHECK_SPAN(!transport_row.empty() && !transport_col.empty(), op->span_)
        << "Internal error: " << op_name << " dynamic valid_shape requires both valid_row and valid_col";
    oss << "(" << transport_row << ", " << transport_col << ")";
  }
  oss << " " << FormatFrontendPipeAttrs(op, split);
  if (!result_type.empty()) {
    oss << " -> " << result_type;
  }
  codegen.Emit(oss.str());
  if (use_full_box) {
    const auto& shape = result_tile_type->shape_;
    INTERNAL_CHECK_SPAN(As<ir::ConstInt>(shape[0]) && As<ir::ConstInt>(shape[1]), op->span_)
        << "Internal error: full-box tpop localization requires a compile-time constant physical shape";
    std::string logical_type = codegen.GetViewTileBufTypeStringFromTileType(result_tile_type);
    INTERNAL_CHECK_SPAN(!logical_type.empty(), op->span_)
        << "Internal error: full-box tpop localization requires a logical tile type";
    std::string logical_buf = codegen.NewNamedTemp("tpop_logical");
    codegen.Emit(logical_buf + " = pto.treshape " + transport_buf + " : " + result_type + " -> " +
                 logical_type);
    codegen.RegisterTileBufType(logical_buf, logical_type);
    codegen.SetCurrentResultBuf(logical_buf);
  }

  return "";
}

/// tfree codegen for system.tfree_to_{aic,aiv}: emits pto.tfree_from_{aic,aiv} {split = N}
static std::string MakeTfreeCodegenPTO(const char* target, const CallPtr& op,
                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  const std::string op_name = std::string("tfree_to_") + target;

  INTERNAL_CHECK_SPAN(op->args_.size() == 1, op->span_)
      << op_name << " requires 1 argument (tile from tpop), got " << op->args_.size();
  INTERNAL_CHECK_SPAN(op->HasKwarg("split"), op->span_)
      << "Internal error: system." << op_name
      << " is missing its 'split' kwarg; StampTfreeSplit must "
         "run before codegen to copy it from the originating tpop";
  const int split = op->GetKwarg<int>("split", 0);

  std::ostringstream oss;
  oss << "pto.tfree_from_" << target << " " << FormatFrontendPipeAttrs(op, split);
  codegen.Emit(oss.str());

  return "";
}

static bool ExprIsI32Scalar(const ir::ExprPtr& expr) {
  if (auto st = As<ScalarType>(expr->GetType())) {
    return st->dtype_ == DataType::INT32;
  }
  return false;
}

// Pipe buffer operands are i32 SSA. GetExprAsCode(ConstInt) uses index constants; use i32 here.
static std::string GetPipeBufOperandI32SSA(codegen::PTOCodegen& codegen, const ir::ExprPtr& expr) {
  if (auto c = As<ir::ConstInt>(expr)) {
    return codegen.GetOrEmitConstant(static_cast<int64_t>(static_cast<int32_t>(c->value_)), DataType::INT32);
  }
  INTERNAL_CHECK_SPAN(ExprIsI32Scalar(expr), expr->span_)
      << "Initialize-pipe buffer operand must be INT32 scalar SSA or integral ConstInt placeholder";
  return codegen.GetExprAsCode(expr);
}

// Helper to format initialize_pipe operand list
static void EmitInitializePipeOperands(std::ostringstream& oss, const std::string& gm_ssa,
                                       const std::string& c2v_ssa, const std::string& v2c_ssa) {
  if (!gm_ssa.empty()) {
    oss << "\n      (gm_slot_buffer = " << gm_ssa << " : !pto.ptr<f32>"
        << ", c2v_consumer_buf = " << c2v_ssa << " : i32"
        << ", v2c_consumer_buf = " << v2c_ssa << " : i32)";
  } else {
    oss << " (c2v_consumer_buf = " << c2v_ssa << " : i32"
        << ", v2c_consumer_buf = " << v2c_ssa << " : i32)";
  }
}

// system.{aic,aiv}_initialize_pipe: Initialize a cross-core pipe. `target` is
// "aic" (Cube side) or "aiv" (Vector side); operands are explicit i32 SSAs
// (validated by the MixedKernelExpanded verifier).
static std::string MakeInitializePipeCodegenPTO(const char* target, const CallPtr& op,
                                                codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  const std::string op_name = std::string(target) + "_initialize_pipe";

  INTERNAL_CHECK_SPAN(op->args_.size() == 2, op->span_)
      << op_name << " requires 2 arguments (c2v_consumer_buf, v2c_consumer_buf), got " << op->args_.size();
  const int dir_mask = op->GetKwarg<int>("dir_mask", -1);
  const int slot_size = op->GetKwarg<int>("slot_size", -1);
  CHECK(dir_mask >= 0) << op_name << " requires 'dir_mask' attribute";
  CHECK(slot_size > 0) << op_name << " requires 'slot_size' attribute";

  std::string c2v_ssa = GetPipeBufOperandI32SSA(codegen, op->args_[0]);
  std::string v2c_ssa = GetPipeBufOperandI32SSA(codegen, op->args_[1]);
  CHECK(!c2v_ssa.empty() && !v2c_ssa.empty()) << op_name << ": failed to lower buffer operands to SSA names";

  std::ostringstream oss;
  oss << "pto." << target << "_initialize_pipe " << FormatInitializePipeAttrs(op, dir_mask, slot_size);
  const int pipe_id = op->GetKwarg<int>("id", 0);
  EmitInitializePipeOperands(oss, codegen.GetGMSlotBufferSSAForPipe(pipe_id, dir_mask), c2v_ssa, v2c_ssa);
  codegen.Emit(oss.str());

  return "";
}

static std::string MakeCrossCoreSyncCodegenPTO(const char* action, const CallPtr& op,
                                               codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  const std::string op_name = std::string("system.sync_") + action;
  INTERNAL_CHECK_SPAN(op->args_.size() <= 1, op->span_)
      << op_name << " accepts at most one dynamic event-id operand, got " << op->args_.size();

  const int pipe_value = op->GetKwarg<int>("pipe", -1);
  INTERNAL_CHECK_SPAN(
      pipe_value >= static_cast<int>(ir::PipeType::MTE1) && pipe_value <= static_cast<int>(ir::PipeType::ALL),
      op->span_)
      << op_name << " requires a valid pipe attribute, got " << pipe_value;

  const bool has_static_event_id = op->HasKwarg("event_id");
  const bool has_dynamic_event_id = op->args_.size() == 1;
  INTERNAL_CHECK_SPAN(has_static_event_id != has_dynamic_event_id, op->span_)
      << op_name << " requires exactly one static event_id attribute or dynamic event-id operand";

  std::string event_code;
  if (has_static_event_id) {
    const int event_id = op->GetKwarg<int>("event_id", -1);
    INTERNAL_CHECK_SPAN(event_id >= 0 && event_id <= 13, op->span_)
        << op_name << " event_id must be in the user-available range [0, 13], got " << event_id;
    event_code = std::to_string(event_id);
  } else {
    auto event_type = ir::As<ScalarType>(op->args_[0]->GetType());
    INTERNAL_CHECK_SPAN(event_type && event_type->dtype_ == DataType::INDEX, op->span_)
        << op_name << " dynamic event id must have ScalarType(INDEX)";
    event_code = codegen.GetExprAsCode(op->args_[0]);
  }

  std::ostringstream oss;
  oss << "pto.sync." << action << " <PIPE_" << ir::PipeTypeToString(static_cast<ir::PipeType>(pipe_value))
      << ">, " << event_code;
  if (op->HasKwarg("ffts_mode")) {
    INTERNAL_CHECK_SPAN(std::string_view(action) == "set", op->span_)
        << op_name << " does not support ffts_mode";
    const int ffts_mode = op->GetKwarg<int>("ffts_mode", -1);
    INTERNAL_CHECK_SPAN(ffts_mode >= 0 && ffts_mode <= 2, op->span_)
        << op_name << " ffts_mode must be in [0, 2], got " << ffts_mode;
    oss << " {ffts_mode = " << ffts_mode << " : i32}";
  }
  codegen.Emit(oss.str());
  return "";
}

static std::string MakeSetFFTSCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  INTERNAL_CHECK_SPAN(op->args_.size() == 1, op->span_)
      << "system.set_ffts requires one workspace tensor, got " << op->args_.size();
  auto workspace = As<ir::Var>(op->args_[0]);
  INTERNAL_CHECK_SPAN(workspace, op->span_) << "system.set_ffts workspace must be a tensor variable";
  auto tensor_type = ir::AsTensorTypeLike(workspace->GetType());
  INTERNAL_CHECK_SPAN(
      tensor_type && tensor_type->dtype_ == DataType::INT64 && tensor_type->shape_.size() == 1, op->span_)
      << "system.set_ffts workspace must be a one-dimensional INT64 tensor";
  auto extent = As<ir::ConstInt>(tensor_type->shape_[0]);
  INTERNAL_CHECK_SPAN(extent && extent->value_ >= 256, op->span_)
      << "system.set_ffts workspace must have a static length of at least 256 INT64 elements";
  codegen.Emit("pto.set_ffts " + codegen.GetVarName(workspace) + " : !pto.ptr<i64>");
  return "";
}

void RegisterCrossCoreOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops) {
  // Register ops with custom codegen logic
  auto reg = [&](const char* op_name, BackendCodegenFunc fn) {
    if (exclude_ops.count(op_name) > 0) return;
    backend.RegisterOp(op_name).f_codegen(std::move(fn));
  };

  reg("tile.tpush_to_aiv", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTpushCodegenPTO("aiv", op, codegen);
  });
  reg("tile.tpop_from_aiv", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTpopCodegenPTO("aiv", op, codegen);
  });
  reg("tile.tpush_to_aic", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTpushCodegenPTO("aic", op, codegen);
  });
  reg("tile.tpop_from_aic", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTpopCodegenPTO("aic", op, codegen);
  });
  reg("system.tfree_to_aic", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTfreeCodegenPTO("aic", op, codegen);
  });
  reg("system.tfree_to_aiv", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTfreeCodegenPTO("aiv", op, codegen);
  });
  reg("system.aic_initialize_pipe", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeInitializePipeCodegenPTO("aic", op, codegen);
  });
  reg("system.aiv_initialize_pipe", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeInitializePipeCodegenPTO("aiv", op, codegen);
  });
  reg("system.sync_set", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeCrossCoreSyncCodegenPTO("set", op, codegen);
  });
  reg("system.sync_wait", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeCrossCoreSyncCodegenPTO("wait", op, codegen);
  });
  reg("system.set_ffts", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeSetFFTSCodegenPTO(op, codegen);
  });

  reg("system.reserve_buffer", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    INTERNAL_CHECK_SPAN(op->args_.size() == 0, op->span_)
        << "reserve_buffer takes no arguments, got " << op->args_.size();

    const auto name = op->GetKwarg<std::string>("name");
    const int size = op->GetKwarg<int>("size", -1);
    const int base = op->GetKwarg<int>("base", -1);
    // Under memory_planner=PtoAS, AllocateMemoryAddr is skipped and ptoas PlanMemory places the
    // reserved region itself (`auto = true`, base absent — ptoas rejects both being present).
    const bool auto_alloc = !codegen.EmitTileAddr();
    CHECK(!name.empty()) << "reserve_buffer requires 'name' attribute";
    CHECK(size > 0) << "reserve_buffer requires positive 'size' attribute, got " << size;
    INTERNAL_CHECK_SPAN(auto_alloc || base >= 0, op->span_)
        << "reserve_buffer requires AllocateMemoryAddr to resolve 'base' before PTO emission, got " << base;
    CheckSafeIdentifier(name, "reserve_buffer 'name'");

    std::string ssa_name = codegen.GetCurrentResultTarget();
    if (ssa_name.empty()) {
      // EvalStmt context — derive SSA name from buffer name hint
      ssa_name = codegen.NewNamedTemp(name);
    }

    std::string location;
    if (codegen.IsAICFunction()) {
      location = "mat";
    } else if (codegen.IsAIVFunction()) {
      location = "vec";
    } else {
      location = "undefined";
    }

    std::ostringstream oss;
    oss << ssa_name << " = pto.reserve_buffer {name = \"" << name << "\", size = " << size
        << ", location = #pto.address_space<" << location << ">, auto = " << (auto_alloc ? "true" : "false");
    if (!auto_alloc) {
      oss << ", base = " << base;
    }
    oss << "} -> i32";
    codegen.Emit(oss.str());

    return std::string("");
  });

  reg("system.import_peer_buffer", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    INTERNAL_CHECK_SPAN(op->args_.size() == 0, op->span_)
        << "import_peer_buffer takes no arguments, got " << op->args_.size();

    const auto name = op->GetKwarg<std::string>("name");
    const auto peer_func = op->GetKwarg<std::string>("peer_func");
    CHECK(!name.empty()) << "import_peer_buffer requires 'name' attribute";
    CHECK(!peer_func.empty()) << "import_peer_buffer requires 'peer_func' attribute";
    CheckSafeIdentifier(name, "import_peer_buffer 'name'");
    CheckSafeIdentifier(peer_func, "import_peer_buffer 'peer_func'");

    std::string ssa_name = codegen.GetCurrentResultTarget();
    if (ssa_name.empty()) {
      ssa_name = codegen.NewNamedTemp(name + "_import");
    }

    std::ostringstream oss;
    oss << ssa_name << " = pto.import_reserved_buffer {name = \"" << name << "\", peer_func = @" << peer_func
        << "} -> i32";
    codegen.Emit(oss.str());

    return std::string("");
  });

  reg("system.syncall", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    // Cross-core all-participant barrier (pto::SYNCALL). `mode` selects the
    // hard/FFTS form (no operands) or the soft/GM-polling form (operands).
    const auto core_type = op->GetKwarg<std::string>("core_type", "mix");
    const auto mode = op->GetKwarg<std::string>("mode", "hard");
    CHECK(core_type == "aiv_only" || core_type == "aic_only" || core_type == "mix")
        << "system.syncall: core_type must be aiv_only|aic_only|mix, got " << core_type;

    if (mode == "hard") {
      INTERNAL_CHECK_SPAN(op->args_.empty(), op->span_)
          << "system.syncall (hard form) takes no arguments, got " << op->args_.size();
      codegen.Emit("pto.syncall() mode = #pto.sync_all_mode<hard>, core_type = #pto.sync_core_type<" +
                   core_type + ">");
      return std::string("");
    }

    CHECK(mode == "soft") << "system.syncall: mode must be hard|soft, got " << mode;
    // The current PTO-ISA uses one soft operand ABI for every participant set:
    // [gm_workspace] or [gm_workspace, used_cores]. Omitting used_cores asks
    // PTO-ISA to derive the participant count from the launch configuration.
    INTERNAL_CHECK_SPAN(op->args_.size() == 1 || op->args_.size() == 2, op->span_)
        << "system.syncall (soft " << core_type << ") requires gm_workspace and optional used_cores, got "
        << op->args_.size() << " operands";

    // gm_workspace: shared GM int32 tensor. PTOAS v0.60's tile-native soft
    // SYNCALL ABI takes the raw GM pointer and constructs its fixed 16-element
    // GlobalTensor internally. PTO-ISA uses one exclusive 64-byte cache line,
    // so a statically shaped workspace must contain at least 16 int32 elements.
    auto gm_var = AsVarLike(op->args_[0]);
    CHECK_SPAN(gm_var, op->span_) << "system.syncall soft: gm_workspace must be a tensor variable";
    auto gm_tt = As<ir::TensorType>(gm_var->GetType());
    CHECK_SPAN(gm_tt && !gm_tt->shape_.empty(), op->span_)
        << "system.syncall soft: gm_workspace must be a tensor with rank >= 1";
    const std::string dtype_str = codegen.GetTypeString(gm_tt->dtype_);
    CHECK_SPAN(dtype_str == "i32", op->span_)
        << "system.syncall soft: gm_workspace must be an INT32 tensor, got " << dtype_str;

    constexpr int64_t kSyncAllSoftWorkspaceInt32 = 16;
    int64_t static_capacity = 1;
    bool has_dynamic_dim = false;
    for (const auto& dim_expr : gm_tt->shape_) {
      auto dim = As<ir::ConstInt>(dim_expr);
      if (!dim) {
        has_dynamic_dim = true;
        continue;
      }
      CHECK_SPAN(dim->value_ > 0, op->span_)
          << "system.syncall soft: gm_workspace dimensions must be positive, got " << dim->value_;
      static_capacity = std::min(kSyncAllSoftWorkspaceInt32,
                                 static_capacity * std::min(kSyncAllSoftWorkspaceInt32, dim->value_));
    }
    if (!has_dynamic_dim) {
      CHECK_SPAN(static_capacity >= kSyncAllSoftWorkspaceInt32, op->span_)
          << "system.syncall soft: gm_workspace must contain at least " << kSyncAllSoftWorkspaceInt32
          << " INT32 elements (64 bytes), got " << static_capacity;
    }

    // Assemble the operand and type lists: gm_ptr[, used_cores].
    std::vector<std::string> operands = {codegen.GetTensorBasePtr(gm_var)};
    std::vector<std::string> types = {"!pto.ptr<i32>"};
    if (op->args_.size() == 2) {
      CHECK_SPAN(ExprIsI32Scalar(op->args_[1]), op->span_)
          << "system.syncall soft: used_cores must be an INT32 scalar";
      operands.push_back(codegen.GetExprAsCode(op->args_[1]));
      types.emplace_back("i32");
    }

    std::ostringstream oss;
    oss << "pto.syncall(";
    for (size_t i = 0; i < operands.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << operands[i];
    }
    oss << " : ";
    for (size_t i = 0; i < types.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << types[i];
    }
    oss << ") mode = #pto.sync_all_mode<soft>, core_type = #pto.sync_core_type<" << core_type << ">";
    codegen.Emit(oss.str());
    return std::string("");
  });
}
}  // namespace backend
}  // namespace pypto

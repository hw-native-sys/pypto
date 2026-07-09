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
 * @file pto_ops_datamove.cpp
 * @brief PTO codegen registration for data-movement / tile-view / shuffle ops.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/pto_ops_common.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/distributed/comm_layout.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/codegen/pto/pto_type_utils.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"
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
using pto_ops_detail::CheckSubviewTileCompat;
using pto_ops_detail::EmitPartitionViewPTO;
using pto_ops_detail::GetDimStrings;
using pto_ops_detail::GetIndexOffsetCodes;
using pto_ops_detail::GetSizeCodes;
using pto_ops_detail::InferSubviewTileTypeComponents;
using pto_ops_detail::MakePartitionTensorViewType;
using pto_ops_detail::mask_patterns;
using pto_ops_detail::MaterializeSubviewOperandIfNeeded;

// Helper function for tile.assemble → pto.subview + pto.tmov
// Writes source tile into target tile at a given row/col offset.  Lowering:
//   1. (optional) pto.tmov target → dst when buffer reuse did not merge them
//      (preserves any data outside the insertion window).
//   2. %dst_view = pto.subview %dst[row, col] sizes [src.rows, src.cols] : ... -> ...
//   3. pto.tmov ins(%src) outs(%dst_view)
// Arguments: args[0] = target (destination base), args[1] = source, args[2] = offset MakeTuple
static std::string MakeTileAssembleCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 3) << "tile.assemble requires 3 arguments (target, source, offset), got "
                               << op->args_.size();

  auto target_tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
  auto source_tile_type = ir::As<ir::TileType>(op->args_[1]->GetType());
  INTERNAL_CHECK_SPAN(target_tile_type && source_tile_type, op->span_)
      << "tile.assemble target and source must both be TileType";
  // The result tile is the actual base that pto.subview views (the dst %r); its
  // tile config — not the target arg's — is what the subview must match.
  auto result_tile_type = ir::As<ir::TileType>(op->GetType());
  INTERNAL_CHECK_SPAN(result_tile_type, op->span_) << "tile.assemble result must be a TileType";

  // An Acc->Mat assemble is a *converting* move, not a pure same-config view: the
  // Mat destination uses a different tile config than the Acc source (Mat fractal
  // 512 vs Acc fractal 1024). PTOAS supports this as an MTE1 `pto.tmov` (its TMovOp
  // verifier handles isAccToMat and only requires the Mat *result* fractal to be
  // 512 — no element-type constraint). The subview is typed entirely from the Mat
  // result below, so we skip CheckSubviewTileCompat (which enforces identical
  // source/result config and must stay strict for every other, pure subview).
  // Gate on the *result* space — that is the tile pto.subview is actually OF, and
  // what every downstream decision (view config + space) reads.
  const bool cross_space_acc_to_mat = source_tile_type->memory_space_ == ir::MemorySpace::Acc &&
                                      result_tile_type->memory_space_ == ir::MemorySpace::Mat;
  if (!cross_space_acc_to_mat) {
    CheckSubviewTileCompat(*target_tile_type, *source_tile_type, "tile.assemble");
  } else {
    // Element-type, address-space, and full-config equality with the base are
    // guaranteed by construction (the view is built from the result tile below, and
    // the IR already requires source.dtype == result.dtype — see
    // DeduceTileAssembleType). The one invariant the bypass must still enforce is
    // CheckSubviewTileCompat's pad rule: pto.subview cannot carry a pad_value.
    const auto src_v = ir::tile_view_semantics::GetEffectiveTileView(*source_tile_type);
    const auto res_v = ir::tile_view_semantics::GetEffectiveTileView(*result_tile_type);
    CHECK_SPAN(src_v.pad == ir::PadValue::null && res_v.pad == ir::PadValue::null, op->span_)
        << "tile.assemble: Acc->Mat pto.subview does not support pad_value; apply "
           "tile.fillpad on the result tile instead of carrying a pad on the window";
  }

  std::string target = codegen.GetExprAsCode(op->args_[0]);
  std::string src = codegen.GetExprAsCode(op->args_[1]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  auto offset_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK_SPAN(offset_tuple, op->span_) << "tile.assemble third argument must be a tuple (offset)";
  INTERNAL_CHECK_SPAN(offset_tuple->elements_.size() >= 2, op->span_)
      << "tile.assemble offset tuple must have at least 2 elements (row, col), got "
      << offset_tuple->elements_.size();
  std::string row_off = codegen.GetExprAsCode(offset_tuple->elements_[0]);
  std::string col_off = codegen.GetExprAsCode(offset_tuple->elements_[1]);

  // pto.subview is a view, so writing into the dst_view only affects the
  // [row, col]+sizes window.  Data outside that window must already be present
  // in dst — when target and dst are different buffers (memory reuse did not
  // merge them), copy target → dst first to preserve target's outer data.
  //
  // For a full-window cross-space Acc->Mat insert (offset 0 and source physical
  // shape == result physical shape) the source tmov below overwrites the entire
  // result tile, so this preservation copy is a dead write that is immediately
  // overwritten — and it would otherwise be an unsupported Mat->Mat tmov. Skip it.
  // Scoped to the Acc->Mat path so same-config assemble codegen is untouched.
  // Physical shapes are the right test: a full-window tmov rewrites the whole
  // window, so source valid_shape < physical does not retain target data (plain
  // replace — DeduceTileAssembleType sets the result valid to the target's full
  // shape). Dynamic dims fall through to the guard below.
  auto off_row_c = ir::As<ir::ConstInt>(offset_tuple->elements_[0]);
  auto off_col_c = ir::As<ir::ConstInt>(offset_tuple->elements_[1]);
  bool full_window_acc_to_mat = cross_space_acc_to_mat && off_row_c && off_col_c && off_row_c->value_ == 0 &&
                                off_col_c->value_ == 0 &&
                                source_tile_type->shape_.size() == result_tile_type->shape_.size();
  for (size_t i = 0; full_window_acc_to_mat && i < source_tile_type->shape_.size(); ++i) {
    auto s = ir::As<ir::ConstInt>(source_tile_type->shape_[i]);
    auto r = ir::As<ir::ConstInt>(result_tile_type->shape_[i]);
    full_window_acc_to_mat = s && r && s->value_ == r->value_;
  }
  if (target != dst && !full_window_acc_to_mat) {
    // A partial, non-merged insert needs target's out-of-window data copied into
    // dst. For a cross-space Acc->Mat assemble that copy is an unsupported Mat->Mat
    // tmov; it is only reachable when the Mat target is not reused in-place as the
    // result (PR3's iter-arg merging makes target == dst, dissolving this path).
    // Fail loud rather than emit invalid IR.
    CHECK_SPAN(!cross_space_acc_to_mat, op->span_)
        << "tile.assemble: a partial Acc->Mat insert whose Mat target is not reused "
           "in-place as the result is not yet supported — the out-of-window "
           "preservation copy would be an unsupported Mat->Mat move; assemble into a "
           "result tile that reuses the Mat target in place";
    std::string target_type = codegen.GetExprTypeAnnotation(op->args_[0]);
    std::ostringstream mov;
    mov << "pto.tmov ins(" << target;
    if (!target_type.empty()) mov << " : " << target_type;
    mov << ") outs(" << dst;
    if (!dst_type.empty()) mov << " : " << dst_type;
    mov << ")";
    codegen.Emit(mov.str());
  }

  // An Acc(L0C, f32) -> Mat(L1, bf16/f16) write at an (row, col) offset is the
  // cube's FIXPIPE writeback (pto-isa `mte_l0c_l1`): the only offset Acc->Mat
  // path on A2/A3, and it intrinsically downcasts the f32 accumulator to the Mat
  // tile's low-precision dtype. Lower it to `pto.tinsert` (the documented offset
  // Acc->Mat op) rather than `pto.subview` + a converting `pto.tmov`: ptoas
  // rejects the latter for a partial window (its tmov shape check reads the
  // subview's *base* [M, N], not the [m, n] window — verified against ptoas
  // v0.45), and the f32->bf16 cast has no MTE1 `tmov` form. A same-dtype (f32)
  // full-window cross-space assemble keeps the subview/tmov path below.
  const bool fixpipe_insert = cross_space_acc_to_mat && (result_tile_type->dtype_ == DataType::BF16 ||
                                                         result_tile_type->dtype_ == DataType::FP16);
  if (fixpipe_insert) {
    std::ostringstream tins;
    tins << "pto.tinsert ins(" << src << ", " << row_off << ", " << col_off;
    if (!src_type.empty()) tins << " : " << src_type << ", index, index";
    tins << ") outs(" << dst;
    if (!dst_type.empty()) tins << " : " << dst_type;
    tins << ")";
    codegen.Emit(tins.str());
    return "";
  }

  // Build %dst_view = pto.subview %dst[%row, %col] sizes [R, C] valid [Vr, Vc] : <dst_type> -> <view_type>
  // The subview "sizes" attribute is the source tile's physical shape, while
  // the explicit `valid [...]` operands must match the source tile's logical
  // valid_shape. PTOAS v0.32 validates that the result tile_buf type's
  // v_row/v_col agree with those explicit valid operands, so the result type
  // must be static when source valid_shape is static, and dynamic only when the
  // source valid_shape itself is dynamic.
  const auto& src_shape = source_tile_type->shape_;
  INTERNAL_CHECK_SPAN(src_shape.size() >= 2, op->span_)
      << "tile.assemble source must have at least 2 dimensions for pto.subview";
  auto rows_const = ir::As<ir::ConstInt>(src_shape[0]);
  auto cols_const = ir::As<ir::ConstInt>(src_shape[1]);
  INTERNAL_CHECK_SPAN(rows_const && cols_const, op->span_)
      << "tile.assemble source shape must be compile-time constant for pto.subview sizes attribute";

  ir::ExprPtr valid_row_expr = src_shape[0];
  ir::ExprPtr valid_col_expr = src_shape[1];
  const auto src_valid = ir::tile_view_semantics::GetEffectiveTileView(*source_tile_type).valid_shape;
  if (src_valid.size() >= 1 && src_valid[0]) valid_row_expr = src_valid[0];
  if (src_valid.size() >= 2 && src_valid[1]) valid_col_expr = src_valid[1];

  auto valid_row_const = ir::As<ir::ConstInt>(valid_row_expr);
  auto valid_col_const = ir::As<ir::ConstInt>(valid_col_expr);
  std::string valid_rows = valid_row_const
                               ? codegen.GetOrEmitConstant(valid_row_const->value_, DataType::INDEX)
                               : codegen.GetExprAsCode(valid_row_expr);
  std::string valid_cols = valid_col_const
                               ? codegen.GetOrEmitConstant(valid_col_const->value_, DataType::INDEX)
                               : codegen.GetExprAsCode(valid_col_expr);

  INTERNAL_CHECK_SPAN(source_tile_type->memory_space_.has_value(), op->span_)
      << "tile.assemble source must carry a memory space for pto.subview result typing";
  // The pto.subview is a window OF the assemble's RESULT tile (the dst %r), and
  // SubViewOp::verify requires the view to match its base in element type, address
  // space, AND the whole tile config (it compares TileBufConfigAttr as one attr).
  // For a same-config assemble the source equals the result (guaranteed by
  // CheckSubviewTileCompat above), so seeding from the source is unchanged behavior.
  // For a cross-space Acc->Mat assemble the Acc source and Mat result differ, so
  // build the view from the *result* and keep only the source's physical window
  // size (rows/cols; valid extents are applied below). Every dtype/layout/fractal/
  // pad/space field then comes from the base by construction — no field-by-field
  // patch to keep in sync, and it stays correct if a future IR change permits a
  // requantising Acc->Mat (TMovOp::isAccToMat imposes no element-type constraint).
  auto view_type_info =
      codegen::ExtractTileTypeInfo(*source_tile_type, codegen.GetTypeString(source_tile_type->dtype_));
  auto view_memory_space = source_tile_type->memory_space_.value();
  if (cross_space_acc_to_mat) {
    const int64_t window_rows = view_type_info.rows;
    const int64_t window_cols = view_type_info.cols;
    view_type_info =
        codegen::ExtractTileTypeInfo(*result_tile_type, codegen.GetTypeString(result_tile_type->dtype_));
    view_type_info.rows = window_rows;
    view_type_info.cols = window_cols;
    view_memory_space = result_tile_type->memory_space_.value();
  }
  if (valid_row_const) {
    view_type_info.v_row = valid_row_const->value_;
    view_type_info.v_row_dynamic = false;
  }
  if (valid_col_const) {
    view_type_info.v_col = valid_col_const->value_;
    view_type_info.v_col_dynamic = false;
  }
  std::string view_type = codegen::FormatTileBufTypeString(
      codegen::MemorySpaceToMLIR(view_memory_space), view_type_info.dtype_str, view_type_info.rows,
      view_type_info.cols, view_type_info.blayout, view_type_info.slayout, view_type_info.fractal,
      view_type_info.pad, view_type_info.v_row, view_type_info.v_col, view_type_info.v_row_dynamic,
      view_type_info.v_col_dynamic);

  std::string dst_view = codegen.NewNamedTemp("assemble_view");
  std::ostringstream sv;
  sv << dst_view << " = pto.subview " << dst << "[" << row_off << ", " << col_off << "] sizes ["
     << rows_const->value_ << ", " << cols_const->value_ << "]";
  sv << " valid [" << valid_rows << ", " << valid_cols << "]";
  if (!dst_type.empty() && !view_type.empty()) {
    sv << " : " << dst_type << " -> " << view_type;
  }
  codegen.Emit(sv.str());
  if (!view_type.empty()) {
    codegen.RegisterTileBufType(dst_view, view_type);
  }

  // Emit pto.tmov ins(%src) outs(%dst_view) — the actual data transfer.
  std::ostringstream tmov;
  tmov << "pto.tmov ins(" << src;
  if (!src_type.empty()) tmov << " : " << src_type;
  tmov << ") outs(" << dst_view;
  if (!view_type.empty()) tmov << " : " << view_type;
  tmov << ")";
  codegen.Emit(tmov.str());
  return "";
}

// tile.gather_row: load one GM row directly into a sub-region of the destination
// (Mat/Vec) accumulator. Lowering (no pto.tmov):
//   1. %dst_view = pto.subview %dst[row, col] sizes [R, C] valid [R, C] : ... -> ...
//   2. %src_pview = pto.partition_view %src_view, offsets = [...], sizes = [r, c] : ... -> ...
//   3. pto.tload ins(%src_pview) outs(%dst_view)
// Filling an L1 (Mat) tile is only valid via GM->Mat tload (MAT->MAT tmov is
// unsupported on a2a3), so the row is written straight into the accumulator
// sub-region. DPS: %dst is the in-place result target. ``transpose`` swaps the
// destination subview dims (GM row [r, c] -> L1 column [c, r]) for the matmul
// B-operand layout.
static std::string MakeGatherRowCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 5) << "tile.gather_row requires 5 arguments "
                                  "(dst, src, dst_offset, src_offset, shapes), got "
                               << op->args_.size();

  auto dst_tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(dst_tile_type, op->span_) << "tile.gather_row dst must be a TileType";
  auto src = ir::AsVarLike(op->args_[1]);
  INTERNAL_CHECK_SPAN(src, op->span_) << "tile.gather_row src must be a Var or IterArg";
  auto src_tensor_type = ir::AsTensorTypeLike(src->GetType());
  INTERNAL_CHECK_SPAN(src_tensor_type, op->span_) << "tile.gather_row src must have TensorType";

  auto dst_off = ir::As<ir::MakeTuple>(op->args_[2]);
  auto src_off = ir::As<ir::MakeTuple>(op->args_[3]);
  auto shapes = ir::As<ir::MakeTuple>(op->args_[4]);
  INTERNAL_CHECK_SPAN(dst_off && src_off && shapes, op->span_)
      << "tile.gather_row offsets and shapes must be literal tuples";
  INTERNAL_CHECK_SPAN(
      dst_off->elements_.size() >= 2 && src_off->elements_.size() >= 2 && shapes->elements_.size() >= 2,
      op->span_)
      << "tile.gather_row offsets and shapes must have at least 2 elements";

  bool transpose = false;
  for (const auto& [k, v] : op->kwargs_) {
    if (k == "transpose") transpose = AnyCast<bool>(v, "transpose");
  }

  auto r_const = ir::As<ir::ConstInt>(shapes->elements_[0]);
  auto c_const = ir::As<ir::ConstInt>(shapes->elements_[1]);
  INTERNAL_CHECK_SPAN(r_const && c_const, op->span_)
      << "tile.gather_row shapes must be compile-time constants for pto.subview sizes";
  // Destination subview shape: transpose maps a GM row [r, c] to an L1 column [c, r].
  const int64_t sv_rows = transpose ? c_const->value_ : r_const->value_;
  const int64_t sv_cols = transpose ? r_const->value_ : c_const->value_;

  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();
  std::string row_off = codegen.GetExprAsCode(dst_off->elements_[0]);
  std::string col_off = codegen.GetExprAsCode(dst_off->elements_[1]);

  // Build the destination subview type from the accumulator's tile type, with the
  // per-row shape/valid.
  const auto dst_space = dst_tile_type->memory_space_.value_or(ir::MemorySpace::Mat);
  auto view_info = codegen::ExtractTileTypeInfo(*dst_tile_type, codegen.GetTypeString(dst_tile_type->dtype_));

  // In a boxed (NZ/fractal) layout — i.e. an L1/Mat matmul operand — the inner
  // box has a fixed granularity, and pto.subview requires the *physical* size to
  // be a whole number of boxes per dim (ptoas: "boxed layout subview sizes must
  // be multiples of inner shape"). A per-row gather writes a single row, so we
  // carve a box-aligned physical sub-region (size = phys_rows x phys_cols) but
  // mark only the real extent valid (valid = sv_rows x sv_cols); the tload then
  // fills just that row. ND tiles (Vec, slayout=none_box) have no inner box and
  // use the exact per-row size.
  const bool boxed = view_info.slayout != ir::TileLayout::none_box;
  auto round_up = [](int64_t n, int64_t mult) { return ((n + mult - 1) / mult) * mult; };
  // NZ fractal granularity: M0 = 16 rows; the C0 lane count along columns is
  // fractal_bytes / dtype_bytes / M0 (both collapse to 16 for fp16/bf16).
  constexpr int64_t kNZFractalRows = 16;
  const int64_t dtype_bytes = std::max<int64_t>(1, static_cast<int64_t>(dst_tile_type->dtype_.GetBit()) / 8);
  const int64_t box_cols =
      view_info.fractal > 0
          ? std::max<int64_t>(1, static_cast<int64_t>(view_info.fractal) / dtype_bytes / kNZFractalRows)
          : kNZFractalRows;
  const int64_t phys_rows = boxed ? round_up(sv_rows, kNZFractalRows) : sv_rows;
  const int64_t phys_cols = boxed ? round_up(sv_cols, box_cols) : sv_cols;

  view_info.rows = phys_rows;
  view_info.cols = phys_cols;
  view_info.v_row = sv_rows;
  view_info.v_row_dynamic = false;
  view_info.v_col = sv_cols;
  view_info.v_col_dynamic = false;
  std::string view_type = codegen::FormatTileBufTypeString(
      codegen::MemorySpaceToMLIR(dst_space), view_info.dtype_str, view_info.rows, view_info.cols,
      view_info.blayout, view_info.slayout, view_info.fractal, view_info.pad, view_info.v_row,
      view_info.v_col, view_info.v_row_dynamic, view_info.v_col_dynamic);

  std::string valid_rows = codegen.GetOrEmitConstant(sv_rows, DataType::INDEX);
  std::string valid_cols = codegen.GetOrEmitConstant(sv_cols, DataType::INDEX);
  std::string dst_view = codegen.NewNamedTemp("gather_row_view");
  std::ostringstream sv;
  sv << dst_view << " = pto.subview " << dst << "[" << row_off << ", " << col_off << "] sizes [" << phys_rows
     << ", " << phys_cols << "] valid [" << valid_rows << ", " << valid_cols << "]";
  if (!dst_type.empty() && !view_type.empty()) {
    sv << " : " << dst_type << " -> " << view_type;
  }
  codegen.Emit(sv.str());
  if (!view_type.empty()) codegen.RegisterTileBufType(dst_view, view_type);

  // GM source window [r, c] -> partition_view, then tload into the subview.
  std::string dtype_str = codegen.GetTypeString(src_tensor_type->dtype_);
  std::string src_view_type = codegen.GetTensorViewTypeString(src_tensor_type.get());
  const auto& shape_elems = shapes->elements_;
  const auto& soff_elems = src_off->elements_;

  std::string src_pview;
  std::string partition_type;
  if (transpose) {
    // Transposing per-row gather: the GM row [r=1, c] must land as the L1 column
    // [c, 1]. pto.tload itself does NOT transpose, so we feed it a DN-strided
    // source (a [c, 1] DN partition) -> pto.tload runs DN2NZ, which IS the
    // transpose. Build a DN make_tensor_view of the GM source (shape/strides
    // swapped vs the canonical ND view, same base ptr) and partition the row as
    // a column, mirroring the matmul b_trans load (which loads a DN view into an
    // NZ tile). Without this the straight ND2NZ tload scrambles in the fractal
    // layout (wrong results / AICore 507018 at scale).
    INTERNAL_CHECK_SPAN(src_tensor_type->shape_.size() == 2, op->span_)
        << "tile.gather_row transpose requires a 2D src";
    std::string src_ptr = codegen.GetVarName(src);
    std::string rows_code = codegen.GetExprAsCode(src_tensor_type->shape_[0]);
    std::string cols_code = codegen.GetExprAsCode(src_tensor_type->shape_[1]);
    std::string one_code = codegen.GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
    std::string dn_view = codegen.NewNamedTemp(src->name_hint_ + "_dn_view");
    std::ostringstream mv;
    // DN view: shape [C, R], strides [1, C] -> DN[i, j] aliases src[j, i].
    mv << dn_view << " = pto.make_tensor_view " << src_ptr << ", shape = [" << cols_code << ", " << rows_code
       << "], strides = [" << one_code << ", " << cols_code
       << "] {layout = #pto.layout<dn>}: " << src_view_type;
    codegen.Emit(mv.str());
    // Read src[phys, col_off : col_off + c] presented as the DN column [c, 1]:
    // offsets [col_off, phys] (swapped), sizes [c, r] (swapped).
    std::vector<ExprPtr> tr_off = {soff_elems[1], soff_elems[0]};
    std::vector<ExprPtr> tr_shape = {shape_elems[1], shape_elems[0]};
    partition_type = MakePartitionTensorViewType(GetDimStrings(tr_shape), dtype_str);
    src_pview =
        EmitPartitionViewPTO(src->name_hint_, dn_view, src_view_type, partition_type,
                             GetIndexOffsetCodes(tr_off, codegen), GetSizeCodes(tr_shape, codegen), codegen);
  } else {
    std::string src_view = codegen.GetOrCreateTensorView(src);
    partition_type = MakePartitionTensorViewType(GetDimStrings(shape_elems), dtype_str);
    src_pview = EmitPartitionViewPTO(src->name_hint_, src_view, src_view_type, partition_type,
                                     GetIndexOffsetCodes(soff_elems, codegen),
                                     GetSizeCodes(shape_elems, codegen), codegen);
  }

  std::ostringstream tload_line;
  tload_line << "pto.tload ins(" << src_pview << " : " << partition_type << ") outs(" << dst_view;
  if (!view_type.empty()) tload_line << " : " << view_type;
  tload_line << ")";
  codegen.Emit(tload_line.str());
  return "";
}

// Helper function for Sort32: emits pto.tsort32
// PTOAS expects: ins(src, idx : src_type, idx_type) outs(dst : dst_type)
static std::string MakeSort32CodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                        codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name
                               << "] requires 2 arguments (src, idx), but got " << op->args_.size();

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string idx = codegen.GetExprAsCode(op->args_[1]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string idx_type = codegen.GetExprTypeAnnotation(op->args_[1]);

  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  std::ostringstream oss;
  oss << pto_op_name;
  // ins clause: src, idx
  oss << " ins(" << src << ", " << idx;
  if (!src_type.empty() || !idx_type.empty()) {
    oss << " : " << src_type << ", " << idx_type;
  }
  // outs clause: dst only (idx is modified in-place by hardware)
  oss << ") outs(" << dst;
  if (!dst_type.empty()) {
    oss << " : " << dst_type;
  }
  oss << ")";

  codegen.Emit(oss.str());
  return "";
}

// Helper function for GatherMask: emits pto.tgather with maskPattern attribute
// PTOAS expects: ins(src, {maskPattern = #pto.mask_pattern<Pxxxx>} : src_type) outs(dst : dst_type)
static std::string MakeGatherMaskCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 1) << "tile.gather_mask requires 1 argument (src), but got " << op->args_.size();

  int pattern = op->GetKwarg<int>("mask_pattern");
  CHECK(pattern >= 1 && pattern < static_cast<int>(mask_patterns.size()))
      << "mask_pattern out of range: " << pattern;

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  std::ostringstream oss;
  oss << "pto.tgather ins(" << src << ", {maskPattern = #pto.mask_pattern<" << mask_patterns.at(pattern)
      << ">}";
  if (!src_type.empty()) {
    oss << " : " << src_type;
  }
  oss << ") outs(" << dst;
  if (!dst_type.empty()) {
    oss << " : " << dst_type;
  }
  oss << ")";

  codegen.Emit(oss.str());
  return "";
}

// Helper for tile.gather_compare (TGATHER compare-form, two outputs):
//   pto.tgather ins(src, kvalue, tmp {cmpMode = #pto<cmp eq>, offset = N : i32}
//                   : src_ty, kv_ty, tmp_ty)
//               outs(dst, cdst : dst_ty, cdst_ty)
//
// Op surface: 3 inputs / TupleType{dst_TileType, cdst_TileType} output. DPS
// dst/cdst buffers are bound by downstream `<element> = tuple_var[i]`
// AssignStmts (parser desugaring of `dst, cdst = ...`). Because the framework
// only pre-binds `fs_.current_result_*` for TileType LHS, multi-output ops
// must resolve their own DPS targets — done via ResolveTupleResultElements.
static std::string MakeGatherCompareCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 3) << "tile.gather_compare requires 3 arguments (src, kvalue, tmp), but got "
                               << op->args_.size();

  ir::VarPtr tuple_var = codegen.GetCurrentResultVar();
  INTERNAL_CHECK_SPAN(tuple_var, op->span_)
      << "Internal error: tile.gather_compare codegen requires current_result_var";

  auto element_vars = codegen.ResolveTupleResultElements(tuple_var, /*arity=*/2);
  INTERNAL_CHECK_SPAN(element_vars[0] && element_vars[1], op->span_)
      << "Internal error: tile.gather_compare expects two TupleGetItemExpr consumers (dst, cdst), got "
      << (element_vars[0] ? "dst-yes" : "dst-no") << "/" << (element_vars[1] ? "cdst-yes" : "cdst-no");

  // Eagerly emit alloc_tile for dst/cdst; the later `dst = tuple_var[i]`
  // AssignStmts skip re-emission via fs_.emitted_tile_alloc_vars.
  std::array<std::shared_ptr<const ir::TileType>, 2> elem_types;
  for (size_t i = 0; i < 2; ++i) {
    elem_types[i] = ir::GetTileTypeWithMemRef(element_vars[i]->GetType());
    INTERNAL_CHECK_SPAN(elem_types[i], element_vars[i]->span_)
        << "Internal error: tile.gather_compare element var " << i
        << " must have TileType with MemRef set by InitMemRef";
    codegen.EmitAllocTileForVar(element_vars[i], elem_types[i]);
  }

  int cmp_mode = op->GetKwarg<int>("cmp_mode");
  CHECK(cmp_mode >= 0 && cmp_mode < 6) << "tile.gather_compare cmp_mode out of range: " << cmp_mode;
  static constexpr const char* kCmpNames[] = {"eq", "ne", "lt", "le", "gt", "ge"};
  int offset = op->GetKwarg<int>("offset", 0);

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string kvalue = codegen.GetExprAsCode(op->args_[1]);
  std::string tmp = codegen.GetExprAsCode(op->args_[2]);
  std::string src_ty = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string kv_ty = codegen.GetExprTypeAnnotation(op->args_[1]);
  std::string tmp_ty = codegen.GetExprTypeAnnotation(op->args_[2]);
  std::string dst = codegen.GetVarName(element_vars[0]);
  std::string cdst = codegen.GetVarName(element_vars[1]);
  std::string dst_ty = codegen.GetTileBufTypeStringFromTileType(elem_types[0]);
  std::string cdst_ty = codegen.GetTileBufTypeStringFromTileType(elem_types[1]);

  std::ostringstream oss;
  oss << "pto.tgather ins(" << src << ", " << kvalue << ", " << tmp;
  if (!src_ty.empty() || !kv_ty.empty() || !tmp_ty.empty()) {
    oss << " : " << src_ty << ", " << kv_ty << ", " << tmp_ty;
  }
  oss << ") outs(" << dst << ", " << cdst;
  if (!dst_ty.empty() || !cdst_ty.empty()) {
    oss << " : " << dst_ty << ", " << cdst_ty;
  }
  oss << ") {cmpMode = #pto<cmp " << kCmpNames[cmp_mode] << ">, offset = " << offset << " : i32}";

  codegen.Emit(oss.str());
  return "";
}

// Helper for tile.scatter (TSCATTER index form, DPS):
//   pto.tscatter ins(%src, %indexes : src_ty, idx_ty) outs(%dst : dst_ty)
//
// IR surface: 3-input op (dst, src, indexes) marked
// set_output_reuses_input(0) — the AssignStmt LHS aliases `dst` so
// GetCurrentResultTarget() returns the same SSA as args_[0].
static std::string MakeScatterCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 3) << "tile.scatter requires 3 arguments (dst, src, indexes), but got "
                               << op->args_.size();

  std::string src = codegen.GetExprAsCode(op->args_[1]);
  std::string idx = codegen.GetExprAsCode(op->args_[2]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[1]);
  std::string idx_type = codegen.GetExprTypeAnnotation(op->args_[2]);

  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  // DPS in-place contract: tile.scatter is set_output_reuses_input(0), so the
  // result buffer must alias the `dst` input tile (args_[0]). Otherwise the
  // tscatter writes a freshly-allocated tile and the rows it does not touch are
  // never initialized with `dst`'s values. PTOCodegen guarantees this by
  // binding the result var to the input SSA (ShouldAliasScatterResultToInput).
  std::string input_ssa = codegen.GetExprAsCode(op->args_[0]);
  INTERNAL_CHECK(!dst.empty() && dst == input_ssa)
      << "Internal error: tile.scatter result SSA must alias the dst input tile SSA, got dst=" << dst
      << ", input=" << input_ssa;

  std::ostringstream oss;
  oss << "pto.tscatter ins(" << src << ", " << idx;
  // Emit the type clause only when both annotations are present; printing one
  // alone would produce malformed PTOAS (": , idx" or ": src, "). The two
  // operands are typed tiles produced by the same lowering, so they should
  // either both carry an annotation or (in untyped contexts) both lack one — a
  // one-sided annotation signals a real codegen bug, not a valid input.
  INTERNAL_CHECK_SPAN(src_type.empty() == idx_type.empty(), op->span_)
      << "Internal error: tile.scatter src/indexes type annotations must both be present or both "
         "absent, got src_type='"
      << src_type << "', idx_type='" << idx_type << "'";
  if (!src_type.empty() && !idx_type.empty()) {
    oss << " : " << src_type << ", " << idx_type;
  }
  oss << ") outs(" << dst;
  if (!dst_type.empty()) {
    oss << " : " << dst_type;
  }
  oss << ")";

  codegen.Emit(oss.str());
  return "";
}

// Helper for tile.scatter_mask (DPS; PyPTO codegen mask form, not a real ISA op):
//   pto.tscatter ins(%src, {maskPattern = #pto.mask_pattern<Pxxxx>} : src_ty)
//                outs(%dst : dst_ty)
//
// The maskPattern rides *inside* ins() right after the src operand, exactly
// like pto.tgather's mask form — PTOAS parses ins() as "src, attr-dict :
// type" and rejects a bare ins(%src ...) ("expected ',' after src operand").
// The type annotation follows the attr dict, still inside ins().
//
// IR surface: 2-input op (dst, src) + mask_pattern attr; dst aliased via
// set_output_reuses_input(0). NOTE: pto-isa/PTOAS expose a maskPattern form
// only for tgather, not tscatter — this tscatter mask emission is a PyPTO
// codegen construct, not a distinct ISA instruction. Emitted for A2/A3 /
// CPU-sim style lowering paths.
static std::string MakeScatterMaskCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 2) << "tile.scatter_mask requires 2 arguments (dst, src), but got "
                               << op->args_.size();

  int pattern = op->GetKwarg<int>("mask_pattern");
  CHECK(pattern >= 1 && pattern < static_cast<int>(mask_patterns.size()))
      << "tile.scatter_mask mask_pattern out of range: " << pattern;

  std::string src = codegen.GetExprAsCode(op->args_[1]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  // DPS in-place contract (mirror of tile.scatter): the result must alias the
  // `dst` input tile (args_[0]). NOTE: the mask-form emission zero-fills dst and
  // writes only the mask-marked columns, so it does NOT itself preserve dst's
  // unselected columns — DPS preserve is reconstructed upstream by the
  // tensor.scatter_mask lowering (sel-blend); see op_conversion_registry.cpp.
  std::string input_ssa = codegen.GetExprAsCode(op->args_[0]);
  INTERNAL_CHECK(!dst.empty() && dst == input_ssa)
      << "Internal error: tile.scatter_mask result SSA must alias the dst input tile SSA, got dst=" << dst
      << ", input=" << input_ssa;

  std::ostringstream oss;
  // maskPattern rides inside ins() after src, then the type annotation:
  //   pto.tscatter ins(%src, {maskPattern = #pto.mask_pattern<Pxxxx>} : src_ty) outs(%dst : dst_ty)
  oss << "pto.tscatter ins(" << src << ", {maskPattern = #pto.mask_pattern<" << mask_patterns.at(pattern)
      << ">}";
  if (!src_type.empty()) {
    oss << " : " << src_type;
  }
  oss << ") outs(" << dst;
  if (!dst_type.empty()) {
    oss << " : " << dst_type;
  }
  oss << ")";

  codegen.Emit(oss.str());
  return "";
}

// Helper function for MrgSort format2: emits pto.tmrgsort
// Supports 2-4 way merge. tmp is the last ins operand and carries the
// {exhausted} attribute; outs holds dst plus a synthesized executed vector
// (vector<4xi16>) — the executed status is not an IR-level tile operand:
//   2-way: ins(src0, src1, tmp {exhausted} : src_types..., tmp_type)
//          outs(dst, executed : dst_type, vector<4xi16>)
//   3-way: ins(src0, src1, src2, tmp {exhausted} : ...) outs(dst, executed : ...)
//   4-way: ins(src0, src1, src2, src3, tmp {exhausted} : ...) outs(dst, executed : ...)
static std::string MakeMrgSortCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() >= 3 && op->args_.size() <= 5)
      << "Operation:[" << pto_op_name << "] requires 3-5 arguments (2-4 srcs + tmp), but got "
      << op->args_.size();

  size_t n_srcs = op->args_.size() - 1;

  std::vector<std::string> srcs, src_types;
  for (size_t i = 0; i < n_srcs; ++i) {
    srcs.push_back(codegen.GetExprAsCode(op->args_[i]));
    src_types.push_back(codegen.GetExprTypeAnnotation(op->args_[i]));
  }

  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();
  std::string tmp = codegen.GetExprAsCode(op->args_[n_srcs]);
  std::string tmp_type = codegen.GetExprTypeAnnotation(op->args_[n_srcs]);
  std::string executed_vec = codegen.NewNamedTemp("executed_vec");
  codegen.Emit(executed_vec + " = arith.constant dense<0> : vector<4xi16>");

  bool exhausted = op->GetKwarg<bool>("exhausted", false);
  std::string exhausted_attr = exhausted ? "{exhausted = true}" : "{exhausted = false}";

  std::ostringstream oss;
  oss << pto_op_name << " ins(";
  for (size_t i = 0; i < n_srcs; ++i) {
    oss << srcs[i] << ", ";
  }
  oss << tmp << " " << exhausted_attr;

  bool has_types = !tmp_type.empty();
  for (const auto& t : src_types) {
    if (!t.empty()) {
      has_types = true;
      break;
    }
  }
  if (has_types) {
    oss << " : ";
    for (size_t i = 0; i < n_srcs; ++i) {
      oss << src_types[i] << ", ";
    }
    oss << tmp_type;
  }

  oss << ") outs(" << dst << ", " << executed_vec;
  if (!dst_type.empty()) {
    oss << " : " << dst_type << ", vector<4xi16>";
  }
  oss << ")";

  codegen.Emit(oss.str());
  return "";
}

// Helper function for MrgSort1 format1: emits pto.tmrgsort
// format1: ins(src, blockLen : src_type, i32) outs(dst : dst_type)
static std::string MakeMrgSort1CodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                          codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name
                               << "] requires 2 arguments (src, block_len), but got " << op->args_.size();

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);

  // blockLen must be i32 per PTO ISA. Constants use the optimized dedup path;
  // runtime variables (e.g., loop-carried block_len) go through GetExprAsCode + cast.
  std::string block_len;
  if (auto const_int = ir::As<ir::ConstInt>(op->args_[1])) {
    block_len = codegen.GetOrEmitConstant(static_cast<int64_t>(static_cast<int32_t>(const_int->value_)),
                                          DataType::INT32);
  } else {
    block_len = codegen.GetExprAsCode(op->args_[1]);
    block_len = codegen.EmitCastToI32(op->args_[1], block_len);
  }

  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  std::ostringstream oss;
  oss << pto_op_name << " ins(" << src << ", " << block_len;
  if (!src_type.empty()) {
    oss << " : " << src_type << ", i32";
  }
  oss << ") outs(" << dst;
  if (!dst_type.empty()) {
    oss << " : " << dst_type;
  }
  oss << ")";

  codegen.Emit(oss.str());
  return "";
}

// Emit a metadata-only `pto.treshape` reinterpret of `src_arg` into the current
// result. Shared by the tile.reshape and tile.transpose_view codegen lambdas:
// both lower a MemRef-less view (e.g. over a cross-core tpop slot) to a
// pto.treshape that READS the source SSA — no data movement. `result_type` is the
// result var's TileType buf-type (empty if none); when present a fresh temp
// buffer is bound so the view gets its own SSA name and `: src -> dst` annotation
// (the MemRef-less source's type comes from the TileType, not a MemRef).
static void EmitTreshapeView(codegen::PTOCodegen& codegen, const ir::ExprPtr& src_arg,
                             std::string result_target, const std::string& result_type,
                             const std::string& temp_prefix) {
  std::string src = codegen.GetExprAsCode(src_arg);
  // Annotate the operand with the type its SSA value was DEFINED with, which
  // GetExprTypeAnnotation resolves through the SSA → tile_buf-type map. Deriving
  // it from the IR TileType instead breaks whenever the def carries static valid
  // dims that `ExtractTileTypeInfo` renders as `v_row=?, v_col=?`: a `pto.subview`
  // def infers its valid from the slice `sizes`, so a reshape of a slice would
  // print `valid=?x?` at the use and MLIR rejects the def/use type mismatch.
  std::string src_type = codegen.GetExprTypeAnnotation(src_arg);
  if (src_type.empty()) {
    // MemRef-less source (a view over a cross-core tpop slot): no SSA type was
    // registered and GetExprTypeAnnotation's TileType arm requires a MemRef.
    if (auto src_var = AsVarLike(src_arg)) {
      if (auto tile_type = As<ir::TileType>(src_var->GetType())) {
        src_type = codegen.GetTileBufTypeStringFromTileType(tile_type);
      }
    }
  }
  if (!result_type.empty()) {
    result_target = codegen.NewNamedTemp(temp_prefix);
    codegen.SetCurrentResultBuf(result_target);
    codegen.RegisterTileBufType(result_target, result_type);
  }
  std::ostringstream oss;
  oss << result_target << " = pto.treshape " << src;
  if (!src_type.empty() && !result_type.empty()) {
    oss << " : " << src_type << " -> " << result_type;
  }
  codegen.Emit(oss.str());
}

void RegisterDataMoveOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops) {
  // Register ops with custom codegen logic
  auto reg = [&](const char* op_name, BackendCodegenFunc fn) {
    if (exclude_ops.count(op_name) > 0) return;
    backend.RegisterOp(op_name).f_codegen(std::move(fn));
  };

  // tile.sort32 (TSORT32): all inputs and output must be row_major per ISA
  if (exclude_ops.count("tile.sort32") == 0) {
    backend.RegisterOp("tile.sort32")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeSort32CodegenPTO("pto.tsort32", op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_input_layout(1, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }

  // tile.gather_mask (TGATHER mask form): only src operand + maskPattern attribute
  reg("tile.gather_mask", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeGatherMaskCodegenPTO(op, codegen);
  });
  // tile.gather_compare (TGATHER compare form, two outputs):
  // 3-input op returning TupleType{dst, cdst}; outs() bound to downstream
  // TupleGetItemExpr consumers (parser desugars `dst, cdst = ...`).
  reg("tile.gather_compare", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeGatherCompareCodegenPTO(op, codegen);
  });
  // tile.scatter (TSCATTER index form, DPS): 3-input op (dst, src, indexes).
  reg("tile.scatter", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeScatterCodegenPTO(op, codegen);
  });
  // tile.scatter_mask (DPS): 2-input op (dst, src) + maskPattern attr. PyPTO
  // codegen form lowered to a pto.tscatter mask emission — not a distinct ISA
  // instruction (see MakeScatterMaskCodegenPTO).
  reg("tile.scatter_mask", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeScatterMaskCodegenPTO(op, codegen);
  });

  // tile.mrgsort_format2 (TMRGSORT format2): all inputs and output must be row_major per ISA
  if (exclude_ops.count("tile.mrgsort_format2") == 0) {
    backend.RegisterOp("tile.mrgsort_format2")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeMrgSortCodegenPTO("pto.tmrgsort", op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_input_layout(1, ir::TileLayout::row_major)
        .set_input_layout(2, ir::TileLayout::row_major)
        .set_input_layout(3, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }
  // tile.mrgsort_format1 (TMRGSORT format1): src and output must be row_major per ISA
  if (exclude_ops.count("tile.mrgsort_format1") == 0) {
    backend.RegisterOp("tile.mrgsort_format1")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeMrgSort1CodegenPTO("pto.tmrgsort", op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }

  reg("tile.slice", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    // 3-5 args: (tile, shape, offset[, valid_shape[, drop_dims]]). The optional
    // 5th `drop_dims` operand only affects the result type's rank (already
    // reflected in the result tile-buf type) — the pto.subview sizes/offset come
    // from the full-rank shape/offset tuples, so codegen ignores it. An empty
    // 4th MakeTuple is the "no valid_shape" sentinel that pairs with drop_dims.
    CHECK(op->args_.size() >= 3 && op->args_.size() <= 5)
        << "Operation:[tile.slice] requires 3-5 arguments (tile, shape, offset[, valid_shape[, "
           "drop_dims]]), but got "
        << op->args_.size();

    auto source_tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    INTERNAL_CHECK_SPAN(source_tile_type, op->span_) << "tile.slice source must be TileType";

    std::string src = codegen.GetExprAsCode(op->args_[0]);
    std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);

    auto offset_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
    INTERNAL_CHECK_SPAN(offset_tuple, op->span_) << "tile.slice third argument must be a tuple (offset)";
    INTERNAL_CHECK_SPAN(offset_tuple->elements_.size() >= 2, op->span_)
        << "tile.slice offset tuple must have at least 2 elements (row, col), got "
        << offset_tuple->elements_.size();
    std::string row_off = codegen.GetExprAsCode(offset_tuple->elements_[0]);
    std::string col_off = codegen.GetExprAsCode(offset_tuple->elements_[1]);

    auto shape_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
    INTERNAL_CHECK_SPAN(shape_tuple, op->span_) << "tile.slice shape must be a literal tuple";
    INTERNAL_CHECK_SPAN(shape_tuple->elements_.size() >= 2, op->span_)
        << "tile.slice shape must have at least 2 elements (rows, cols)";
    auto rows_const = ir::As<ir::ConstInt>(shape_tuple->elements_[0]);
    auto cols_const = ir::As<ir::ConstInt>(shape_tuple->elements_[1]);
    INTERNAL_CHECK_SPAN(rows_const && cols_const, op->span_)
        << "tile.slice shape must be compile-time constant for pto.subview sizes attribute";

    std::string valid_row;
    std::string valid_col;
    // valid_shape is the optional 4th operand; an empty MakeTuple means "none"
    // (the form used when only drop_dims is supplied).
    auto valid_tuple = op->args_.size() >= 4 ? ir::As<ir::MakeTuple>(op->args_[3]) : nullptr;
    bool has_explicit_valid_shape = valid_tuple != nullptr && !valid_tuple->elements_.empty();
    if (has_explicit_valid_shape) {
      INTERNAL_CHECK_SPAN(valid_tuple, op->span_) << "tile.slice valid_shape must be a literal tuple";
      INTERNAL_CHECK_SPAN(valid_tuple->elements_.size() >= 2, op->span_)
          << "tile.slice valid_shape must have at least 2 elements";
      valid_row = codegen.GetExprAsCode(valid_tuple->elements_[0]);
      valid_col = codegen.GetExprAsCode(valid_tuple->elements_[1]);
    }

    std::string result_target = codegen.GetCurrentResultTarget();
    std::string result_type = codegen.GetCurrentResultTileBufTypeString();
    INTERNAL_CHECK_SPAN(!result_target.empty(), op->span_) << "tile.slice requires assignment target";

    // tile.slice is always a pure pto.subview view of its source. The 4-arg
    // form (explicit valid_shape) encodes the result valid into pto.subview's
    // `valid [...]` clause; no pto.tmov / pto.set_validshape is emitted. This
    // keeps tile.slice consistent with its set_output_memory_inherit_input
    // contract (the result MemRef shares the source base by design) and
    // avoids the alias corruption seen in #1622 where the previous
    // materializing path's pto.tmov destination overlapped the source
    // allocation when the slice offset was dynamic.
    auto view_type_info = InferSubviewTileTypeComponents(*source_tile_type, *shape_tuple, *offset_tuple,
                                                         codegen.GetTypeString(source_tile_type->dtype_));
    if (has_explicit_valid_shape) {
      // User-supplied valid_shape takes precedence over inference. Each dim is
      // honored independently: a ConstInt valid operand yields a static result
      // dim, a dynamic operand yields a dynamic one. Do NOT promote both dims
      // to dynamic when one is — for the explicit `valid [...]` form PTOAS
      // reads each valid operand directly and requires the result tile_buf
      // type's v_row/v_col to match per-dim (a static `valid_col` operand with
      // a `v_col=?` result type is rejected: 'pto.subview' op expects result
      // valid_shape[1] to match inferred/explicit valid_col). This mirrors
      // tile.assemble's per-dim handling.
      auto valid_row_const = ir::As<ir::ConstInt>(valid_tuple->elements_[0]);
      auto valid_col_const = ir::As<ir::ConstInt>(valid_tuple->elements_[1]);
      view_type_info.v_row_dynamic = valid_row_const == nullptr;
      view_type_info.v_col_dynamic = valid_col_const == nullptr;
      if (valid_row_const) view_type_info.v_row = valid_row_const->value_;
      if (valid_col_const) view_type_info.v_col = valid_col_const->value_;
    }

    INTERNAL_CHECK_SPAN(source_tile_type->memory_space_.has_value(), op->span_)
        << "tile.slice source must carry a memory space for pto.subview result typing";
    std::string view_type = codegen::FormatTileBufTypeString(
        codegen::MemorySpaceToMLIR(*source_tile_type->memory_space_), view_type_info.dtype_str,
        view_type_info.rows, view_type_info.cols, view_type_info.blayout, view_type_info.slayout,
        view_type_info.fractal, view_type_info.pad, view_type_info.v_row, view_type_info.v_col,
        view_type_info.v_row_dynamic, view_type_info.v_col_dynamic);

    std::string view_ssa = codegen.NewNamedTemp("slice_view");
    std::ostringstream oss;
    oss << view_ssa << " = pto.subview " << src << "[" << row_off << ", " << col_off << "] sizes ["
        << rows_const->value_ << ", " << cols_const->value_ << "]";
    if (has_explicit_valid_shape) {
      oss << " valid [" << valid_row << ", " << valid_col << "]";
    }
    if (!src_type.empty() && !view_type.empty()) {
      oss << " : " << src_type << " -> " << view_type;
    }
    codegen.Emit(oss.str());
    codegen.RegisterTileBufType(view_ssa, view_type);

    // Lazy materialization fallback: a few downstream ops (e.g. pto.tcolexpandmul)
    // cannot consume a subview SSA directly because their hardware lowering
    // reads physical tile dims from the operand type. MaterializeSubviewOperandIfNeeded
    // will emit pto.textract on demand into result_target if such a consumer appears.
    codegen::PTOCodegen::SubviewMaterializationInfo mat_info;
    mat_info.source_ssa = src;
    mat_info.source_type = src_type;
    mat_info.row_off_ssa = row_off;
    mat_info.col_off_ssa = col_off;
    mat_info.materialize_target_ssa = result_target;
    mat_info.materialize_target_type = result_type;
    mat_info.source_memory_space = source_tile_type->memory_space_;
    codegen.RegisterSubviewMaterialization(view_ssa, mat_info);

    // Bind the slice's result variable to the subview SSA; the pre-emitted
    // alloc_tile for the result becomes dead and is eliminated by downstream
    // PTOAS passes.
    codegen.SetCurrentResultBuf(view_ssa);
    return std::string("");
  });

  reg("tile.assemble", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileAssembleCodegenPTO(op, codegen);
  });

  reg("tile.gather_row", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeGatherRowCodegenPTO(op, codegen);
  });

  reg("tile.extract", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    CHECK(op->args_.size() == 4)
        << "tile.extract requires 4 arguments (src, index_row, index_col, shape), but got "
        << op->args_.size();

    std::string src = codegen.GetExprAsCode(op->args_[0]);
    std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
    std::string row_off = codegen.GetExprAsCode(op->args_[1]);
    std::string col_off = codegen.GetExprAsCode(op->args_[2]);
    // Use the actual offset SSA dtype (`index` / `i64` / `i32` ...) — the IR
    // type-check accepts any IndexLike scalar, so don't hardcode `index`.
    std::string row_type = codegen.GetExprTypeAnnotation(op->args_[1]);
    std::string col_type = codegen.GetExprTypeAnnotation(op->args_[2]);
    if (row_type.empty()) row_type = "index";
    if (col_type.empty()) col_type = "index";
    // args_[3] is the shape tuple: type-deduction only, no PTO operand.

    std::string result_target = codegen.GetCurrentResultTarget();
    std::string result_type = codegen.GetCurrentResultTileBufTypeStringFromTileType();

    auto existing_type = codegen.GetSSATileBufType(result_target);
    if (!result_type.empty() && existing_type != result_type) {
      result_target = codegen.AllocNewTileBuf(result_type, "extract_buf");
      codegen.SetCurrentResultBuf(result_target);
    } else if (!result_type.empty()) {
      codegen.RegisterTileBufType(result_target, result_type);
    }

    std::ostringstream oss;
    oss << "pto.textract ins(" << src << ", " << row_off << ", " << col_off;
    if (!src_type.empty()) {
      oss << " : " << src_type << ", " << row_type << ", " << col_type;
    }
    oss << ") outs(" << result_target;
    if (!result_type.empty()) {
      oss << " : " << result_type;
    }
    oss << ")";
    codegen.Emit(oss.str());
    return std::string("");
  });

  reg("tile.reshape", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    CHECK(op->args_.size() == 2) << "Operation:[tile.reshape] requires 2 arguments (tile, shape), but got "
                                 << op->args_.size();
    std::string result_target = codegen.GetCurrentResultTarget();

    // Derive the result type from the result var's TileType so a MemRef-less
    // result (a zero-copy view over a tpop slot) still gets a type — the shared
    // helper only returns one for alloc-backed results. Also note whether the
    // result is alloc-backed: the reshape is a PTO-level no-op only then (the
    // per-var alloc model pre-declared it with the reshaped type at a shared
    // addr); a MemRef-less result has no alloc, so it must emit pto.treshape.
    std::string result_type;
    bool result_has_memref = false;
    if (auto result_var = codegen.GetCurrentResultVar()) {
      if (auto result_tile = ir::As<ir::TileType>(result_var->GetType())) {
        result_type = codegen.GetTileBufTypeStringFromTileType(result_tile);
        result_has_memref = result_tile->memref_.has_value();
      }
    }
    auto existing_type = codegen.GetSSATileBufType(result_target);
    if (result_has_memref && !existing_type.empty() && existing_type == result_type) {
      return std::string("");
    }

    // Fallback: emit pto.treshape reading the source SSA.
    EmitTreshapeView(codegen, op->args_[0], result_target, result_type, "reshape_buf");
    return std::string("");
  });

  reg("tile.transpose_view", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    // Zero-copy fractal-layout reinterpretation (NZ<->ZN).
    //  - The result var owns a pre-declared pto.alloc_tile already carrying the
    //    transposed (ZN) type, aliased to the source buffer's address through the
    //    shared MemRef (the #1776 case, PyPTO planner). The two alloc_tile decls
    //    at the same addr ARE the whole mechanism — emit nothing.
    //  - Otherwise there is no declaration carrying the transposed type, so
    //    reinterpret the source in place with pto.treshape reading its SSA (a
    //    metadata-only op — no data movement), exactly like tile.reshape. This
    //    covers both a MemRef-less result (a view over a cross-core tpop slot,
    //    which owns no buffer) and the PTOAS planner, where addr-less aliased
    //    vars collapse onto ONE tile_buf handle: the second alloc_tile that
    //    would have carried the transposed layout is never emitted.
    CHECK(op->args_.size() == 1) << "Operation:[tile.transpose_view] requires 1 argument (tile), but got "
                                 << op->args_.size();
    auto& codegen = AsPto(codegen_base);
    std::string result_target = codegen.GetCurrentResultTarget();

    std::string result_type;
    bool result_has_memref = false;
    if (auto result_var = codegen.GetCurrentResultVar()) {
      if (auto result_tile = ir::As<ir::TileType>(result_var->GetType())) {
        result_type = codegen.GetTileBufTypeStringFromTileType(result_tile);
        result_has_memref = result_tile->memref_.has_value();
      }
    }
    // The result's own alloc_tile already declares the transposed type: it IS
    // the view, so emit nothing. Mirrors tile.reshape's no-op check.
    auto existing_type = codegen.GetSSATileBufType(result_target);
    if (result_has_memref && !existing_type.empty() && existing_type == result_type) {
      return std::string("");
    }

    // No declaration carries the transposed type — reinterpret the source in
    // place via pto.treshape reading its SSA, exactly like tile.reshape.
    EmitTreshapeView(codegen, op->args_[0], result_target, result_type, "transpose_view_buf");
    return std::string("");
  });

  reg("tile.set_validshape", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    CHECK(op->args_.size() == 3)
        << "tile.set_validshape requires 3 arguments (tile, valid_rows, valid_cols), but got "
        << op->args_.size();

    std::string tile_buf = codegen.GetExprAsCode(op->args_[0]);
    std::string tile_buf_type = codegen.GetExprTypeAnnotation(op->args_[0]);
    if (tile_buf.empty()) {
      tile_buf = codegen.GetCurrentResultTarget();
    }
    if (tile_buf_type.empty()) {
      tile_buf_type = codegen.GetCurrentResultTileBufTypeStringFromTileType();
    }

    auto emit_index_arg = [&](const ir::ExprPtr& arg) -> std::string {
      if (auto var = ir::As<ir::Var>(arg)) {
        std::string mlir_name = codegen.GetVarName(var);
        return codegen.EmitCastToIndex(var, mlir_name);
      }
      if (auto c = ir::As<ir::ConstInt>(arg)) {
        return codegen.GetOrEmitConstant(c->value_, DataType::INDEX);
      }
      std::string ssa = codegen.GetExprAsCode(arg);
      if (auto st = ir::As<ir::ScalarType>(arg->GetType())) {
        if (st->dtype_ != DataType::INDEX) {
          std::string src_type = codegen.GetTypeString(st->dtype_);
          std::string idx = codegen.NewTemp();
          codegen.Emit(idx + " = arith.index_cast " + ssa + " : " + src_type + " to index");
          return idx;
        }
      }
      return ssa;
    };

    std::string vr = emit_index_arg(op->args_[1]);
    std::string vc = emit_index_arg(op->args_[2]);

    codegen.RegisterTileBufType(tile_buf, tile_buf_type);
    codegen.SetCurrentResultBuf(tile_buf);
    codegen.Emit("pto.set_validshape " + tile_buf + ", " + vr + ", " + vc + " : " + tile_buf_type);
    return std::string("");
  });
}
}  // namespace backend
}  // namespace pypto

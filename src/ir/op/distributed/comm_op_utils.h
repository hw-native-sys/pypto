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

#ifndef SRC_IR_OP_DISTRIBUTED_COMM_OP_UTILS_H_
#define SRC_IR_OP_DISTRIBUTED_COMM_OP_UTILS_H_

/**
 * @file comm_op_utils.h
 * @brief Shared deducer helpers for the mirror-image cross-rank transfer ops
 *        ``pld.tensor.put`` / ``pld.tile.put`` (put.cpp) and ``pld.tensor.get``
 *        / ``pld.tile.get`` (get.cpp).
 *
 * put and get are structural mirrors: the only semantic differences are which
 * operand must be window-bound (put: dst; get: src) and put's extra ``atomic``
 * attr. Everything else — chunk/pipeline attr parsing, dynamic-extent chunk
 * requirements, staging-tile validation, region-arg validation, and the
 * shape positivity/match loop — is identical, so it lives here instead of being
 * duplicated across the two translation units.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace comm_op {

using Kwargs = std::vector<std::pair<std::string, std::any>>;

// Reject negative chunk sizes regardless of pipeline / dynamic state. The DSL
// `_validate_chunk` enforces this at the user surface, but a direct IR-op call
// (e.g. dist_tensor_ops.put(chunk_rows=-1)) bypasses it and would otherwise feed
// a negative extent into stage-tile creation. 0 = full is still valid here.
inline void ValidateChunkNonNegative(const Kwargs& kwargs, const std::string& op_name) {
  const int chunk_rows = GetKwargOr<int>(kwargs, "chunk_rows", 0);
  const int chunk_cols = GetKwargOr<int>(kwargs, "chunk_cols", 0);
  CHECK(chunk_rows >= 0) << op_name << ": chunk_rows must be non-negative (0 = full), got " << chunk_rows;
  CHECK(chunk_cols >= 0) << op_name << ": chunk_cols must be non-negative (0 = full), got " << chunk_cols;
}

// Double-buffering (ping-pong) only helps a chunked transfer — pto-isa slides
// the transfer through two staging tiles with overlapped TLOAD/TSTORE. Require
// both chunk dims so the staging tile is fully bounded before the second tile
// is allocated. User-facing (driven by the `pipeline` kwarg on put/get).
inline void ValidatePipelineHasChunk(const Kwargs& kwargs, const std::string& op_name) {
  if (!GetKwargOr<bool>(kwargs, "pipeline", false)) return;
  const int chunk_rows = GetKwargOr<int>(kwargs, "chunk_rows", 0);
  const int chunk_cols = GetKwargOr<int>(kwargs, "chunk_cols", 0);
  CHECK(chunk_rows > 0 && chunk_cols > 0)
      << op_name
      << ": pipeline=True requires both chunk_rows>0 and chunk_cols>0 (double-buffering needs a "
         "chunked transfer to ping-pong through two staging tiles)";
}

// A dynamic transfer extent (a runtime sub-extent of the fixed window) needs a
// static chunk to bound the VEC staging tile (UB allocation is static). The
// flattened transfer is [rows = prod(leading dims), cols = innermost dim]: a
// dynamic innermost requires chunk_cols, a dynamic leading dim requires
// chunk_rows. Fully-static transfers need no chunk. User-facing.
inline void ValidateDynamicTransferHasChunk(const std::vector<ExprPtr>& transfer_shape, const Kwargs& kwargs,
                                            const std::string& op_name) {
  if (transfer_shape.empty()) return;
  const bool innermost_dynamic = !As<ConstInt>(transfer_shape.back());
  bool leading_dynamic = false;
  for (size_t i = 0; i + 1 < transfer_shape.size(); ++i) {
    if (!As<ConstInt>(transfer_shape[i])) leading_dynamic = true;
  }
  if (!innermost_dynamic && !leading_dynamic) return;
  const int chunk_rows = GetKwargOr<int>(kwargs, "chunk_rows", 0);
  const int chunk_cols = GetKwargOr<int>(kwargs, "chunk_cols", 0);
  // Span sources are non-null transfer-shape elements: the leading branch only
  // fails when a leading dim exists (size >= 2, front() valid); back() is valid
  // whenever the vector is non-empty (guaranteed by the early return above).
  CHECK_SPAN(!leading_dynamic || chunk_rows > 0, transfer_shape.front()->span_)
      << op_name
      << ": a dynamic leading transfer dim needs a static chunk_rows to bound the VEC staging tile";
  CHECK_SPAN(!innermost_dynamic || chunk_cols > 0, transfer_shape.back()->span_)
      << op_name
      << ": a dynamic innermost transfer dim needs a static chunk_cols to bound the VEC staging tile";
}

// Validate one VEC staging tile operand: must be a 2D TileType whose dtype
// matches the local dst. Returns the TileType for downstream extent checks.
// Shared by the single stage and the optional ping-pong second stage of
// pld.tile.put / pld.tile.get.
inline TileTypePtr ValidateStageTile(const ExprPtr& stage, const DataType& dst_dtype,
                                     const std::string& what) {
  auto stage_type = As<TileType>(stage->GetType());
  CHECK(stage_type) << what << " must be a TileType, got " << stage->GetType()->TypeName();
  CHECK(stage_type->dtype_ == dst_dtype)
      << what << " dtype must match dst dtype, got stage=" << stage_type->dtype_.ToString()
      << " dst=" << dst_dtype.ToString();
  CHECK(stage_type->shape_.size() == 2)
      << what << " must be a 2D VEC staging tile, got rank " << stage_type->shape_.size();
  return stage_type;
}

// Flatten an N-D transfer shape to its 2-D [rows, cols] extent
// (rows = prod(leading dims), cols = innermost dim) and confirm the 2-D VEC
// stage tile fits within it. pto-isa TPUT/TGET auto-chunks the transfer through
// a smaller stage, so the stage is allowed to be smaller than the transfer; it
// must only not exceed it. Dynamic transfer dims can't be compared statically
// (the chunk bounds them at runtime), so they are skipped here.
inline void ValidateStageFitsTransfer(const std::vector<ExprPtr>& stage_shape,
                                      const std::vector<ExprPtr>& transfer_shape, const Span& stage_span,
                                      const std::string& op_name) {
  int64_t transfer_cols = 1;
  bool cols_static = false;
  int64_t transfer_rows = 1;
  bool rows_static = true;
  for (size_t i = 0; i < transfer_shape.size(); ++i) {
    auto d = As<ConstInt>(transfer_shape[i]);
    if (i + 1 == transfer_shape.size()) {
      cols_static = static_cast<bool>(d);
      if (d) transfer_cols = d->value_;
    } else if (d) {
      transfer_rows *= d->value_;
    } else {
      rows_static = false;
    }
  }
  auto stage_rows_c = As<ConstInt>(stage_shape[0]);
  auto stage_cols_c = As<ConstInt>(stage_shape[1]);
  INTERNAL_CHECK_SPAN(stage_rows_c && stage_cols_c, stage_span)
      << "Internal error: " << op_name << " stage dims must be static ConstInt";
  const int64_t stage_rows = stage_rows_c->value_;
  const int64_t stage_cols = stage_cols_c->value_;
  INTERNAL_CHECK_SPAN(stage_rows > 0 && stage_cols > 0, stage_span)
      << "Internal error: " << op_name << " stage dims must be positive, got [" << stage_rows << ", "
      << stage_cols << "]";
  INTERNAL_CHECK_SPAN(!rows_static || stage_rows <= transfer_rows, stage_span)
      << "Internal error: " << op_name << " stage rows " << stage_rows
      << " must fit within flattened transfer rows " << transfer_rows
      << " (pto-isa auto-chunks a smaller stage)";
  INTERNAL_CHECK_SPAN(!cols_static || stage_cols <= transfer_cols, stage_span)
      << "Internal error: " << op_name << " stage cols " << stage_cols << " must fit within transfer cols "
      << transfer_cols << " (pto-isa auto-chunks a smaller stage)";
}

// The element-type / rank / per-dim positivity (and, for full-slice transfers,
// shape-match) loop shared by put's and get's contract validators. Callers
// resolve dst/src to their respective window-bound vs plain-tensor roles first;
// this checks only the shape vectors, which are validated identically for both.
inline void ValidateTransferShapeContract(const std::vector<ExprPtr>& dst_shape,
                                          const std::vector<ExprPtr>& src_shape, const std::string& op_name,
                                          bool require_same_shape) {
  CHECK(!dst_shape.empty()) << op_name << " requires at least one dimension on dst/src";
  CHECK(dst_shape.size() == src_shape.size())
      << op_name << " dst rank (" << dst_shape.size() << ") must match src rank (" << src_shape.size() << ")";
  for (size_t i = 0; i < dst_shape.size(); ++i) {
    // Window dims may be dynamic (a runtime-sized window). Static positivity
    // applies only to ConstInt dims; a dynamic transfer extent needs a static
    // chunk to bound the VEC staging tile (enforced by
    // ValidateDynamicTransferHasChunk in the deducer). For full-slice the
    // dst/src dims must still match — by value when static, structurally when
    // dynamic (the codegen drives both partition views from one extent).
    auto d = As<ConstInt>(dst_shape[i]);
    auto s = As<ConstInt>(src_shape[i]);
    if (d) {
      CHECK_SPAN(d->value_ > 0, dst_shape[i]->span_)
          << op_name << " shape dimension " << i << " must be positive, got " << d->value_;
    }
    if (s) {
      CHECK_SPAN(s->value_ > 0, src_shape[i]->span_)
          << op_name << " src shape dimension " << i << " must be positive, got " << s->value_;
    }
    if (require_same_shape) {
      if (d && s) {
        CHECK_SPAN(d->value_ == s->value_, dst_shape[i]->span_)
            << op_name << " dst and src must have the same static shape; dimension " << i
            << " differs (dst=" << d->value_ << ", src=" << s->value_ << ")";
      } else {
        CHECK_SPAN(AreExprsEqual(dst_shape[i], src_shape[i]), dst_shape[i]->span_)
            << op_name << " full-slice dst and src must have the same shape; dimension " << i
            << " differs (dynamic dims must match structurally)";
      }
    }
  }
}

// Validate the subregion args (dst_offsets, src_offsets, shape) at
// ``region_arg_base`` and return the per-dim transfer shape (the ``shape``
// tuple's elements). Each offset/shape tuple must be rank-matched; static dims
// are bounds-checked against the window, dynamic dims are bounded by the chunk
// at runtime. Shared by put and get (both narrow their partition views with the
// same offset+shape contract).
inline std::vector<ExprPtr> ValidateRegionArgs(const std::vector<ExprPtr>& args, size_t region_arg_base,
                                               const std::vector<ExprPtr>& dst_shape,
                                               const std::vector<ExprPtr>& src_shape,
                                               const std::string& op_name) {
  auto dst_offsets = As<MakeTuple>(args[region_arg_base]);
  auto src_offsets = As<MakeTuple>(args[region_arg_base + 1]);
  auto transfer_shape = As<MakeTuple>(args[region_arg_base + 2]);
  CHECK(dst_offsets) << op_name << " dst_offsets must be a tuple";
  CHECK(src_offsets) << op_name << " src_offsets must be a tuple";
  CHECK(transfer_shape) << op_name << " shape must be a tuple";
  CHECK(dst_offsets->elements_.size() == dst_shape.size())
      << op_name << " dst_offsets rank must match dst rank";
  CHECK(src_offsets->elements_.size() == src_shape.size())
      << op_name << " src_offsets rank must match src rank";
  CHECK(transfer_shape->elements_.size() == dst_shape.size())
      << op_name << " shape rank must match tensor rank";

  for (size_t i = 0; i < transfer_shape->elements_.size(); ++i) {
    // The transfer extent may be dynamic (a runtime sub-extent of the fixed
    // window). A dynamic dim requires a static chunk to bound the staging tile
    // (enforced by ValidateDynamicTransferHasChunk in the deducer). The static
    // positivity / bounds checks below only apply when the dim is a ConstInt.
    auto dim = As<ConstInt>(transfer_shape->elements_[i]);
    if (dim) {
      CHECK_SPAN(dim->value_ > 0, transfer_shape->elements_[i]->span_)
          << op_name << " shape dimension " << i << " must be positive, got " << dim->value_;
    }
    // The window dims may also be dynamic; the subregion bounds check only runs
    // when both the transfer dim and the window dim are static ConstInts.
    auto dst_dim = As<ConstInt>(dst_shape[i]);
    auto src_dim = As<ConstInt>(src_shape[i]);
    if (auto dst_offset = As<ConstInt>(dst_offsets->elements_[i])) {
      CHECK_SPAN(dst_offset->value_ >= 0, dst_offsets->elements_[i]->span_)
          << op_name << " dst_offsets dimension " << i << " must be non-negative, got " << dst_offset->value_;
      if (dim && dst_dim) {
        CHECK_SPAN(dst_offset->value_ + dim->value_ <= dst_dim->value_, dst_offsets->elements_[i]->span_)
            << op_name << " dst subregion dimension " << i
            << " exceeds dst shape (offset=" << dst_offset->value_ << ", shape=" << dim->value_
            << ", dst_dim=" << dst_dim->value_ << ")";
      }
    }
    if (auto src_offset = As<ConstInt>(src_offsets->elements_[i])) {
      CHECK_SPAN(src_offset->value_ >= 0, src_offsets->elements_[i]->span_)
          << op_name << " src_offsets dimension " << i << " must be non-negative, got " << src_offset->value_;
      if (dim && src_dim) {
        CHECK_SPAN(src_offset->value_ + dim->value_ <= src_dim->value_, src_offsets->elements_[i]->span_)
            << op_name << " src subregion dimension " << i
            << " exceeds src shape (offset=" << src_offset->value_ << ", shape=" << dim->value_
            << ", src_dim=" << src_dim->value_ << ")";
      }
    }
  }
  return transfer_shape->elements_;
}

}  // namespace comm_op
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_OP_DISTRIBUTED_COMM_OP_UTILS_H_

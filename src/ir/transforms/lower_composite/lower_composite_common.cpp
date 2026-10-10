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

#include "src/ir/transforms/lower_composite/lower_composite_common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/storage_size.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace lower_composite {

std::vector<ExprPtr> CollapseShapeTo2D(const std::vector<ExprPtr>& shape, const Span& span) {
  INTERNAL_CHECK_SPAN(!shape.empty(), span) << "Cannot flatten a rank-0 tensor shape";
  if (shape.size() == 1) {
    return {std::make_shared<ConstInt>(1, DataType::INDEX, span), shape[0]};
  }
  if (shape.size() == 2) return shape;

  ExprPtr rows = shape[0];
  for (size_t i = 1; i + 1 < shape.size(); ++i) {
    rows = tile_conversion_utils::MakeCanonicalIndexMul(rows, shape[i], span, "LowerCompositeOps");
  }
  return {rows, shape.back()};
}

std::vector<ExprPtr> CollapseShapeToLinear2D(const std::vector<ExprPtr>& shape, const Span& span) {
  INTERNAL_CHECK_SPAN(!shape.empty(), span) << "Cannot flatten a rank-0 tensor shape";
  ExprPtr elements = shape[0];
  for (size_t i = 1; i < shape.size(); ++i) {
    elements = tile_conversion_utils::MakeCanonicalIndexMul(elements, shape[i], span, "LowerCompositeOps");
  }
  return {std::make_shared<ConstInt>(1, DataType::INDEX, span), elements};
}

void CheckAllReduceTargetIsPackedNd(const DistributedTensorTypePtr& target_type, const Span& span) {
  if (!target_type->tensor_view_.has_value()) return;

  const auto& view = target_type->tensor_view_.value();
  CHECK_SPAN(view.layout == TensorLayout::ND, span)
      << "pld.tensor.allreduce target view only supports ND layout";
  if (view.stride.empty()) return;

  const auto packed_strides =
      tensor_view_semantics::BuildLogicalStridesFromLayout(target_type->shape_, TensorLayout::ND);
  CHECK_SPAN(view.stride.size() == packed_strides.size(), span)
      << "pld.tensor.allreduce target shape reinterpret requires a packed source";
  for (size_t i = 0; i < view.stride.size(); ++i) {
    CHECK_SPAN(AreExprsEqual(view.stride[i], packed_strides[i]), span)
        << "pld.tensor.allreduce target shape reinterpret requires a packed source";
  }
}

bool IsRowMajorLinearPrefix(const std::vector<ExprPtr>& valid, const std::vector<ExprPtr>& physical) {
  if (valid.size() != physical.size()) return false;
  bool past_boundary = false;
  for (size_t i = 0; i < valid.size(); ++i) {
    const bool is_full = AreExprsEqual(valid[i], physical[i]);
    if (past_boundary) {
      if (!is_full) return false;
      continue;
    }
    auto valid_const = As<ConstInt>(valid[i]);
    if (!(valid_const && valid_const->value_ == 1)) past_boundary = true;
  }
  return true;
}

const std::vector<ExprPtr>* GetPartialValidShape(const DistributedTensorTypePtr& target_type,
                                                 const Span& span) {
  if (!target_type->tensor_view_.has_value() || target_type->tensor_view_->valid_shape.empty()) {
    return nullptr;
  }

  const auto& valid_shape = target_type->tensor_view_->valid_shape;
  CHECK_SPAN(valid_shape.size() == target_type->shape_.size(), span)
      << "pld.tensor.allreduce target valid_shape rank must match target rank";
  // TensorType canonicalization removes an explicit valid_shape that exactly
  // equals shape. Any remaining valid_shape is therefore a genuine partial
  // region and must stay on the rectangular path below.
  return &valid_shape;
}

CallPtr CreateAllReduceTargetView(const ExprPtr& target, const std::vector<ExprPtr>& flat_shape,
                                  const std::vector<ExprPtr>& flat_valid_shape,
                                  const std::vector<ExprPtr>* partial_valid_shape, const Span& span) {
  // Allreduce owns this alias and reduces only flat_valid_shape. Public
  // tensor.view cannot infer a shape reinterpretation for partial validity.
  auto shape_tuple = tile_conversion_utils::MakeShapeTuple(flat_shape, span);
  std::vector<ExprPtr> view_args{target, shape_tuple};
  if (partial_valid_shape != nullptr) {
    view_args.push_back(tile_conversion_utils::MakeShapeTuple(flat_valid_shape, span));
  }
  return OpRegistry::GetInstance().Create("tensor.view", view_args, {}, span);
}

/// Validates that ``signal_type`` matches the mesh barrier convention (one
/// cell per rank: ``[NR, 1]``). Ring allreduce's signal is ``[2*(NR-1), NR]``
/// instead — one row per round, addressed ``[row, rank]``. Sharing one buffer
/// between the two conventions no longer trips a generation-table state error
/// (the self-clearing protocol is call-local, so there is no cross-call state
/// to mismatch); this shape check is the sole remaining guard against a mesh
/// op silently targeting the wrong cell of a ring-shaped signal. Skip the
/// second-dimension check when it is symbolic, matching the ring rule's own
/// existing shape checks.
void ValidateMeshSignalShape(const DistributedTensorTypePtr& signal_type, const std::string& op_name,
                             const Span& span) {
  CHECK_SPAN(signal_type, span) << op_name << " signal must be a DistributedTensor";
  CHECK_SPAN(signal_type->shape_.size() == 2, span)
      << op_name << " signal must be 2D [NR, 1], got rank " << signal_type->shape_.size();
  if (auto col_dim = As<ConstInt>(signal_type->shape_[1])) {
    CHECK_SPAN(col_dim->value_ == 1, span)
        << op_name << " signal shape[1] must be 1 (one cell per rank), got " << col_dim->value_
        << " — a signal shaped for mode=\"ring\" allreduce ([2*(NR-1), NR]) cannot be shared with "
           "this collective; give it its own [NR, 1] signal window";
  }
}

/// Derives the chunk geometry for ``dtype``. ``op_name`` is woven into the
/// diagnostics so a caller-specific message survives the extraction.
CollectiveChunkGeometry MakeChunkGeometry(const DataType& dtype, const Span& span, const char* op_name) {
  CollectiveChunkGeometry geo;
  geo.storage_bits = static_cast<int64_t>(storage_size::GetStorageBitWidth(dtype));
  INTERNAL_CHECK_SPAN(geo.storage_bits > 0, span)
      << op_name << " target dtype has no storage width: " << dtype.ToString();
  constexpr int64_t kBitsPerByte = 8;
  const int64_t chunk_bits = kAllReduceChunkBytes * kBitsPerByte;
  const int64_t alignment_bits = kPTOTileAlignmentBytes * kBitsPerByte;
  INTERNAL_CHECK_SPAN(chunk_bits % geo.storage_bits == 0, span)
      << op_name << " dtype storage width must divide the chunk bit budget";
  geo.chunk_elements = chunk_bits / geo.storage_bits;
  INTERNAL_CHECK_SPAN(geo.chunk_elements > 0, span)
      << op_name << " dtype is wider than the chunk byte budget";
  INTERNAL_CHECK_SPAN(alignment_bits % geo.storage_bits == 0, span)
      << op_name << " dtype storage width must divide the tile-alignment bit budget";
  geo.alignment_elements = alignment_bits / geo.storage_bits;
  geo.alignment_elements_idx = std::make_shared<ConstInt>(geo.alignment_elements, DataType::INDEX, span);
  geo.alignment_minus_one_idx = std::make_shared<ConstInt>(geo.alignment_elements - 1, DataType::INDEX, span);
  geo.max_chunk_cols = std::make_shared<ConstInt>(geo.chunk_elements, DataType::INDEX, span);
  return geo;
}

/// Picks the physical chunk width for a chunked collective. A statically known
/// ``logical_extent`` smaller than one full chunk is rounded up to the nearest
/// 32-byte-aligned element count, so a short collective does not reserve a full
/// ``kAllReduceChunkBytes`` tile and the caller keeps its remaining VEC UB
/// budget. Anything else — including a symbolic extent — uses the full chunk.
ExprPtr SelectStaticChunkCols(const CollectiveChunkGeometry& geo, const ExprPtr& logical_extent,
                              const Span& span) {
  if (auto extent = As<ConstInt>(logical_extent);
      extent && extent->value_ > 0 && extent->value_ < geo.chunk_elements) {
    const int64_t aligned_extent =
        ((extent->value_ - 1) / geo.alignment_elements + 1) * geo.alignment_elements;
    return std::make_shared<ConstInt>(aligned_extent, DataType::INDEX, span);
  }
  return geo.max_chunk_cols;
}

/// Sizes the 2D VEC staging tile that a ``pld.tile.put`` / ``pld.tile.get``
/// transfer slides through.
///
/// pto-isa TPUT/TGET read the full extent from the partition views and 2D-slide
/// the transfer through the stage, so the stage only has to *fit within* the
/// flattened transfer rather than equal it (see comm_op::ValidateStageFitsTransfer).
/// Sizing the stage from the whole transfer therefore buys nothing and costs
/// everything: a [1, 65537] FP32 transfer would reserve 256 KiB of VEC and fail
/// in AllocateMemoryAddr. Cap it to one ``kAllReduceChunkBytes`` tile instead.
///
/// ``transfer_shape`` is flattened to [rows = prod(leading dims), cols = innermost].
/// A dynamic dim contributes the chunk bound directly, matching MakeTputStageShape's
/// contract for pld.tensor.put / pld.tensor.get: the stage is a static UB
/// allocation, and pto-isa reads the runtime extent from the partition views.
/// ``ValidateStageFitsTransfer`` therefore compares only statically known dims
/// (``!cols_static || stage_cols <= transfer_cols``). A symbolic SIZE of runtime
/// 17 still gets a 4096-element FP32 stage — that is the same bound a user-level
/// ``chunk_cols`` would supply, not a stage-fits-transfer violation. The result
/// never exceeds a *statically known* transfer dim.
ExprPtr MakeCollectiveStageShape(const std::vector<ExprPtr>& transfer_shape,
                                 const CollectiveChunkGeometry& geo, const Span& span, const char* op_name) {
  INTERNAL_CHECK_SPAN(!transfer_shape.empty(), span) << op_name << " transfer shape must have rank >= 1";

  int64_t cols_val = geo.chunk_elements;
  if (auto cols_c = As<ConstInt>(transfer_shape.back()); cols_c && cols_c->value_ > 0) {
    cols_val = std::min(cols_c->value_, geo.chunk_elements);
    // Unlike SelectStaticChunkCols (allreduce's own accumulator scratch,
    // decoupled from the transfer shape), this stage is the literal bounce
    // buffer pld.tile.put/get slides the transfer through, so it can never
    // exceed the transfer (comm_op::ValidateStageFitsTransfer,
    // "!cols_static || stage_cols <= transfer_cols"). A short, non-tile-aligned
    // length (e.g. 17 elements of FP32) must therefore round its column width
    // DOWN to the nearest kPTOTileAlignmentBytes-aligned width rather than up
    // — pto.alloc_tile requires the row byte size aligned, and the remainder
    // (17 - 16 = 1 element here) rides the same auto-chunked partial last
    // slide that already handles a transfer wider than one full chunk. Sizes
    // below one alignment unit (< 8 elements of FP32) have no valid aligned
    // width that still fits within the transfer; leave those unrounded — a
    // pre-existing, narrower gap this cap fix does not claim to close.
    if (cols_val >= geo.alignment_elements && cols_val < geo.chunk_elements) {
      cols_val = (cols_val / geo.alignment_elements) * geo.alignment_elements;
    }
  }

  // Keep a 2D stage (e.g. all_to_all_v's [MAX_RECV, SIZE]) inside the same
  // byte budget by trading rows against the chosen column width.
  const int64_t rows_budget = std::max<int64_t>(1, geo.chunk_elements / cols_val);
  int64_t rows_prod = 1;
  bool rows_static = true;
  for (size_t d = 0; d + 1 < transfer_shape.size(); ++d) {
    auto dim_c = As<ConstInt>(transfer_shape[d]);
    if (!dim_c || dim_c->value_ <= 0) {
      rows_static = false;
      break;
    }
    // Only min(product, rows_budget) is observable. Saturate before
    // multiplication so an otherwise-valid very large static shape cannot
    // overflow int64_t while computing the bounded stage geometry.
    if (rows_prod >= rows_budget || dim_c->value_ > rows_budget / rows_prod) {
      rows_prod = rows_budget;
    } else {
      rows_prod *= dim_c->value_;
    }
  }
  const int64_t rows_val = rows_static ? std::min(rows_prod, rows_budget) : 1;

  return std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(rows_val, DataType::INDEX, span),
                           std::make_shared<ConstInt>(cols_val, DataType::INDEX, span)},
      span);
}

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto

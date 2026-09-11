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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/storage_size.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/flatten_tile_nd_to_2d/rewrite_internal.h"

namespace pypto {
namespace ir {

using transform_utils::FlattenToStmts;

namespace flatten_tile_nd_to_2d {
namespace rewrite_internal {

// ============================================================================

/**
 * @brief Check if a TileType has >2 dimensions.
 */
bool IsNdTile(const TileTypePtr& tile_type) { return tile_type && tile_type->shape_.size() > 2; }

/**
 * @brief Extract a static int64_t from a ConstInt expression.
 *
 * Raises CHECK if the expression is not a ConstInt (dynamic shape).
 */
int64_t GetStaticDim(const ExprPtr& expr, const std::string& context) {
  auto ci = As<ConstInt>(expr);
  CHECK(ci) << "FlattenTileNdTo2D: found a dynamic (non-constant) dimension in " << context
            << ", but flattening >2D tiles to 2D (and unrolling batched matmul) requires every "
               "tile dimension to be a compile-time constant. A pl.dynamic dimension has no static "
               "bound and cannot back a tile dimension directly. Tile/iterate the dynamic dimension "
               "with pl.range/pl.parallel, or reshape to 2D before the InCore (pl.at) scope so the "
               "dynamic extent lands on the pl.parallel loop bound instead of inside the tile shape.";
  return ci->value_;
}

/**
 * @brief Compute the merged 2D shape from an ND shape.
 *
 * [A, B, C, D] -> {A*B*C, D}
 */
std::pair<int64_t, int64_t> ComputeMergedShape(const std::vector<ExprPtr>& shape,
                                               const std::string& context) {
  int64_t merged = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    int64_t dim = GetStaticDim(shape[i], context);
    CHECK(dim > 0) << "FlattenTileNdTo2D: tile dimension " << i << " must be positive in " << context
                   << ", got " << dim;
    // Overflow check: merged * dim must fit in int64_t
    CHECK(merged <= INT64_MAX / dim) << "FlattenTileNdTo2D: integer overflow when computing merged dimension "
                                     << "in " << context << " (merged=" << merged << ", dim=" << dim << ")";
    merged *= dim;
  }
  int64_t last = GetStaticDim(shape.back(), context);
  return {merged, last};
}

/**
 * @brief Build a MakeTuple from int64_t values.
 */
ExprPtr MakeShapeTupleFromInts(const std::vector<int64_t>& dims, const Span& span) {
  std::vector<ExprPtr> elems;
  elems.reserve(dims.size());
  for (auto d : dims) {
    elems.push_back(std::make_shared<ConstInt>(d, DataType::INDEX, span));
  }
  return std::make_shared<MakeTuple>(elems, span);
}

/// Build the ``tile.store`` partition window over a rank>2 output tensor.
///
/// Aligning the tile's dims against the tensor's trailing dims and padding the
/// front with 1s is right whenever each tile dim IS the tensor dim it lands on
/// — tensor ``[B, M, N]`` written from an ``[M, N]`` tile gives ``[1, M, N]``.
/// That window is returned unchanged.
///
/// It stops being right once the tile's leading extent is a COLLAPSE of several
/// tensor dims, which is what ``tensor.gather`` lowering produces: a ``[2, 3,
/// 8]`` result becomes a ``[6, 8]`` tile, and the aligned rule would emit
/// ``[1, 6, 8]`` — 6 of a dim whose extent is 3. PTOAS >= 0.61 rejects that
/// outright, and it only ever reached the intended bytes because the outer
/// stride happened to be contiguous.
///
/// **The collapsed form is not free to pick any in-bounds box.** A flattened
/// store writes ``rows`` CONSECUTIVE positions in row-major order over the
/// leading axes, starting at ``offsets``. ``pto.partition_view`` can only
/// describe a box, and the two coincide exactly when every axis the row count
/// consumes whole is covered whole *and* starts at 0. Where they do not, no
/// window writes the right elements — a ``[12, 8]`` tile over ``[2, 2, 4, 8]``
/// has ``[2, 2, 3, 8]`` available as an in-bounds box, but that box covers
/// flat positions {0,1,2, 4,5,6, 8,9,10, 12,13,14} while the store means
/// {0..11}. Such a store is rejected rather than silently retargeted.
///
/// The innermost axis is exempt from the consecutiveness rule: it carries the
/// tile's columns, so a partial column range is still rectangular. It only has
/// to fit, and its extent may be symbolic — it takes no part in distributing
/// the rows, so a dynamic destination width does not prevent a window.
///
/// @param tile_shape The tile's VALID dims, not its physical ones: the window
///        describes the region the store transfers, which is what codegen's
///        2D path also sizes the partition from. Callers pass
///        `GetEffectiveTileView(tile_type).valid_shape`, which falls back to
///        the physical shape when no valid_shape is set. Static, except where
///        the aligned window is returned untouched (see ComputeMergedShape).
/// @param tensor_shape Output tensor dims, possibly dynamic
/// @param offsets Store offsets, one per tensor dim, possibly dynamic
/// @param span Source location for the emitted ConstInts
/// @return One size per tensor dim
std::vector<ExprPtr> ComputeStorePartitionShape(const std::vector<ExprPtr>& tile_shape,
                                                const std::vector<ExprPtr>& tensor_shape,
                                                const std::vector<ExprPtr>& offsets, const Span& span) {
  const size_t tensor_rank = tensor_shape.size();
  const size_t tile_rank = tile_shape.size();
  INTERNAL_CHECK_SPAN(tile_rank > 0 && tile_rank <= tensor_rank, span)
      << "Internal error: tile.store tile rank " << tile_rank << " must be in [1, " << tensor_rank
      << "] (the output tensor's rank)";
  INTERNAL_CHECK_SPAN(offsets.size() == tensor_rank, span)
      << "Internal error: tile.store has " << offsets.size() << " offsets for a rank-" << tensor_rank
      << " output tensor";

  // The aligned window: 1s for the leading tensor dims the tile does not
  // reach, then the tile's own dims. Each 1 is its own node rather than a
  // shared one — IR consumers match by value, but a node reused across
  // positions is a needless aliasing hazard.
  std::vector<ExprPtr> aligned;
  aligned.reserve(tensor_rank);
  for (size_t i = tile_rank; i < tensor_rank; ++i) {
    aligned.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, span));
  }
  aligned.insert(aligned.end(), tile_shape.begin(), tile_shape.end());

  // Keep it whenever it is already a sub-box. A dim is only rejected when both
  // sides are static and the size provably overflows, so dynamic tensor dims
  // take this path unchanged.
  auto overflows = [](const ExprPtr& size, const ExprPtr& extent) {
    auto size_ci = As<ConstInt>(size);
    auto extent_ci = As<ConstInt>(extent);
    return size_ci && extent_ci && size_ci->value_ > extent_ci->value_;
  };
  bool is_collapsed = false;
  for (size_t i = 0; i < tensor_rank; ++i) {
    if (overflows(aligned[i], tensor_shape[i])) {
      is_collapsed = true;
      break;
    }
  }
  if (!is_collapsed) return aligned;

  const std::string context = "tile.store partition window";
  const auto [rows, cols] = ComputeMergedShape(tile_shape, context);
  auto is_const_zero = [](const ExprPtr& e) {
    auto ci = As<ConstInt>(e);
    return ci && ci->value_ == 0;
  };

  // The innermost axis carries the tile's columns and only has to fit. It takes
  // no part in the row decomposition below, so a symbolic extent there is fine
  // -- demanding a constant would reject a store whose window is fully
  // determined anyway ([6, 8] into [2, 3, D] still gives [2, 3, 8]). Check the
  // bound only where both sides are static enough to prove it violated.
  if (auto last_extent = As<ConstInt>(tensor_shape.back())) {
    if (auto last_offset = As<ConstInt>(offsets.back())) {
      CHECK_SPAN(last_offset->value_ + cols <= last_extent->value_, span)
          << "tile.store writes " << cols << " columns at offset " << last_offset->value_
          << " of an axis whose extent is " << last_extent->value_ << "; the write runs past the tensor";
    }
  }

  // Walk the leading axes outward from the innermost. While the row count is
  // larger than an axis it must consume that axis whole, at offset 0, or the
  // consecutive run it means is not the box this would describe.
  std::vector<int64_t> window(tensor_rank, 1);
  window.back() = cols;
  int64_t remaining = rows;
  for (size_t axis = tensor_rank - 1; axis > 0; --axis) {
    const size_t i = axis - 1;
    const int64_t extent = GetStaticDim(tensor_shape[i], context);
    if (remaining <= extent) {
      // This axis takes what is left; outer axes stay 1 and their offsets pick
      // a single slice, which is rectangular whatever they are.
      if (auto offset = As<ConstInt>(offsets[i])) {
        CHECK_SPAN(offset->value_ + remaining <= extent, span)
            << "tile.store writes " << remaining << " rows at offset " << offset->value_ << " of axis " << i
            << ", whose extent is " << extent << "; the write runs past the tensor";
      }
      window[i] = remaining;
      remaining = 1;
      break;
    }
    CHECK_SPAN(remaining % extent == 0, span)
        << "tile.store cannot lower a " << rows << "x" << cols
        << " tile into this tensor: its rows span several axes, so they must fill axis " << i << " (extent "
        << extent << ") a whole number of times, but " << remaining
        << " rows remain. A flattened store writes consecutive row-major positions, and no "
           "partition window describes that here. Store one whole axis at a time (loop the outer "
           "axis with pl.range/pl.parallel), or reshape the destination so the tile's rows are a "
           "single axis";
    CHECK_SPAN(is_const_zero(offsets[i]), span)
        << "tile.store fills axis " << i << " (extent " << extent
        << ") completely, so it must start at offset 0; a non-zero offset there would make the "
           "written region a non-contiguous run that no partition window describes";
    window[i] = extent;
    remaining /= extent;
  }
  // Falling out of the loop with rows left over means the tile has more rows
  // than the destination's leading axes hold at all.
  CHECK_SPAN(remaining == 1, span) << "tile.store writes a " << rows << "x" << cols
                                   << " tile into a tensor whose leading axes hold only "
                                   << (rows / remaining) << " rows; the write runs past the tensor";

  std::vector<ExprPtr> result;
  result.reserve(tensor_rank);
  for (auto dim : window) result.push_back(std::make_shared<ConstInt>(dim, DataType::INDEX, span));
  return result;
}

/**
 * @brief Build a 2D shape vector from merged dimensions.
 */
std::vector<ExprPtr> Make2DShapeExprs(int64_t merged, int64_t last, const Span& span) {
  return {std::make_shared<ConstInt>(merged, DataType::INDEX, span),
          std::make_shared<ConstInt>(last, DataType::INDEX, span)};
}

/// Merge an ND ``valid_shape`` into its 2D form ``[product(leading), last]``,
/// allowing dynamic (non-ConstInt) entries — unlike ComputeMergedShape, which
/// requires static dims. Static factors are folded into a single ConstInt; the
/// identity factor 1 is dropped. This lets a dynamic ``valid_shape`` (e.g. the
/// ``min(CHUNK, D - c)`` tail from the dynamic-tile strip-mine below) survive the
/// flatten of the physical tile shape rather than being reset to the full static
/// shape.
std::vector<ExprPtr> ComputeMergedValidShape(const std::vector<ExprPtr>& valid, const Span& span) {
  int64_t const_prod = 1;
  ExprPtr dyn = nullptr;
  for (size_t i = 0; i + 1 < valid.size(); ++i) {
    if (auto ci = As<ConstInt>(valid[i])) {
      const_prod *= ci->value_;
    } else {
      dyn = dyn ? MakeMul(dyn, valid[i], span) : valid[i];
    }
  }
  ExprPtr merged;
  if (!dyn) {
    merged = std::make_shared<ConstInt>(const_prod, DataType::INDEX, span);
  } else if (const_prod == 1) {
    merged = dyn;
  } else {
    merged = MakeMul(std::make_shared<ConstInt>(const_prod, DataType::INDEX, span), dyn, span);
  }
  return {merged, valid.back()};
}

/// Build a canonical index add, folding simple ConstInt cases to avoid
/// unstable roundtrip forms such as `0 + 1`.
ExprPtr MakeCanonicalIndexAdd(const ExprPtr& lhs, const ExprPtr& rhs, const Span& span) {
  auto lhs_const = As<ConstInt>(lhs);
  auto rhs_const = As<ConstInt>(rhs);
  if (lhs_const && rhs_const) {
    CHECK((rhs_const->value_ >= 0 && lhs_const->value_ <= INT64_MAX - rhs_const->value_) ||
          (rhs_const->value_ < 0 && lhs_const->value_ >= INT64_MIN - rhs_const->value_))
        << "FlattenTileNdTo2D: integer overflow while canonicalizing index add";
    return std::make_shared<ConstInt>(lhs_const->value_ + rhs_const->value_, DataType::INDEX, span);
  }
  if (lhs_const && lhs_const->value_ == 0) {
    return rhs;
  }
  if (rhs_const && rhs_const->value_ == 0) {
    return lhs;
  }
  return MakeAdd(lhs, rhs, span);
}

std::vector<ExprPtr> CollapseLeadingDimsTo2D(const std::vector<ExprPtr>& dims, const Span& span) {
  INTERNAL_CHECK(dims.size() > 2) << "FlattenTileNdTo2D: collapse to 2D requires rank > 2";
  ExprPtr rows = dims[0];
  for (size_t i = 1; i + 1 < dims.size(); ++i) {
    rows = tile_conversion_utils::MakeCanonicalIndexMul(rows, dims[i], span, "FlattenTileNdTo2D");
  }
  return {rows, dims.back()};
}

CallPtr CreateCollapsedTensorView(const ExprPtr& tensor, const TensorTypePtr& tensor_type, const Span& span) {
  // This compiler-owned alias carries a collapsed partial valid_shape. Public
  // tensor.view intentionally rejects partial-valid shape reinterpretation.
  auto flat_shape = CollapseLeadingDimsTo2D(tensor_type->shape_, span);
  const TensorLayout layout =
      tensor_type->tensor_view_.has_value() ? tensor_type->tensor_view_->layout : TensorLayout::ND;
  // Collapsing a blocked NZ tensor would multiply its fractal dims into a flat
  // row count and silently destroy the fractal addressing. NZ loads are routed
  // around this helper by IsNzSourceLoad; reaching here means that guard was
  // bypassed.
  INTERNAL_CHECK_SPAN(layout != TensorLayout::NZ, span)
      << "Internal error: FlattenTileNdTo2D tried to collapse an NZ tensor view to 2D; "
      << "a blocked NZ tensor must be passed through unchanged";
  TensorView flat_view = tensor_view_semantics::CanonicalizeView(flat_shape, layout);

  if (tensor_type->tensor_view_.has_value()) {
    const auto& source_view = *tensor_type->tensor_view_;
    if (!source_view.valid_shape.empty()) {
      INTERNAL_CHECK_SPAN(source_view.valid_shape.size() == tensor_type->shape_.size(), span)
          << "FlattenTileNdTo2D: tensor valid_shape rank must match tensor rank";
      INTERNAL_CHECK_SPAN(
          tile_conversion_utils::IsRowMajorCollapseContiguous(source_view.valid_shape, tensor_type->shape_),
          span)
          << "FlattenTileNdTo2D: tensor valid_shape cannot be represented by a single 2D view";
      flat_view.valid_shape = CollapseLeadingDimsTo2D(source_view.valid_shape, span);
    }
    flat_view.pad = source_view.pad;
  }

  auto shape_tuple = std::make_shared<MakeTuple>(flat_shape, span);
  std::vector<ExprPtr> view_args{tensor, shape_tuple};
  if (!flat_view.valid_shape.empty() || flat_view.pad != PadValue::null) {
    view_args.push_back(std::make_shared<MakeTuple>(flat_view.valid_shape, span));
  }
  return OpRegistry::GetInstance().Create("tensor.view", view_args, {}, span);
}

ExprPtr CollapseLeadingOffsetsToRow(const std::vector<ExprPtr>& offsets,
                                    const std::vector<ExprPtr>& tensor_shape, const Span& span) {
  INTERNAL_CHECK(offsets.size() == tensor_shape.size())
      << "FlattenTileNdTo2D: offsets and tensor shape rank must match";
  INTERNAL_CHECK(offsets.size() > 2) << "FlattenTileNdTo2D: offset collapse requires rank > 2";
  ExprPtr row_offset = offsets[0];
  for (size_t i = 1; i + 1 < offsets.size(); ++i) {
    row_offset = MakeCanonicalIndexAdd(
        tile_conversion_utils::MakeCanonicalIndexMul(row_offset, tensor_shape[i], span, "FlattenTileNdTo2D"),
        offsets[i], span);
  }
  return row_offset;
}

/// Mat (L1) byte budget for the whole-tile batch_matmul slicing path. Returns the
/// backend's Mat size when a backend is configured (codegen / ST); otherwise
/// SIZE_MAX so passes run without a backend (most unit tests) always take the fit
/// path and keep the whole-load + slice behaviour.
uint64_t GetMatBudgetBytes() {
  if (!backend::BackendConfig::IsConfigured()) return std::numeric_limits<uint64_t>::max();
  return backend::GetBackend()->GetMemSize(ir::MemorySpace::Mat);
}

/// Whole (un-sliced) byte size of an operand from its original ND type. nullopt
/// when any dim is dynamic (size unknown — treated as "fits").
std::optional<uint64_t> OperandWholeBytes(const TileTypePtr& original_type) {
  if (!original_type) return std::nullopt;
  uint64_t elems = 1;
  for (const auto& d : original_type->shape_) {
    auto ci = As<ConstInt>(d);
    if (!ci || ci->value_ < 0) return std::nullopt;
    const uint64_t extent = static_cast<uint64_t>(ci->value_);
    if (extent != 0 && elems > std::numeric_limits<uint64_t>::max() / extent) return std::nullopt;
    elems *= extent;
  }
  return storage_size::StaticStorageBytes(elems, original_type->dtype_);
}

/// Whether both operands' whole tiles fit Mat together, so each can be brought
/// whole into L1 and per-batch sliced. When false (large shapes), a load-sourced
/// (GM) operand is loaded per batch instead (ExtractBatchPage !fit path). Dynamic
/// dims / no backend -> fit (keep the simpler whole+slice path).
///
/// TODO(V2C !fit): a move-sourced operand (Vec compute result moved to Mat, mixed
/// kernel) has no underlying tile.load, so when !fit it still takes the whole-slice
/// path — correct only while the whole moved tile fits the fixed cross-core ring.
/// A per-batch V2C move (slice in Vec → move per batch) is the deferred fallback.
bool BatchOperandsWholeFit(const TileTypePtr& lhs_type, const TileTypePtr& rhs_type) {
  auto lhs_bytes = OperandWholeBytes(lhs_type);
  auto rhs_bytes = OperandWholeBytes(rhs_type);
  if (!lhs_bytes || !rhs_bytes) return true;
  return *lhs_bytes + *rhs_bytes <= GetMatBudgetBytes();
}

/// Convert a vector of ExprPtr shape dimensions into static int64 values.
std::vector<int64_t> ToStaticDims(const std::vector<ExprPtr>& shape, const std::string& context) {
  std::vector<int64_t> dims;
  dims.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    dims.push_back(GetStaticDim(shape[i], context + " dim " + std::to_string(i)));
  }
  return dims;
}

/// Multiply all static dimensions together, with overflow checking.
int64_t MultiplyStaticDims(const std::vector<int64_t>& dims, const std::string& context) {
  int64_t product = 1;
  for (size_t i = 0; i < dims.size(); ++i) {
    CHECK(dims[i] > 0) << "FlattenTileNdTo2D: dimension " << i << " must be positive in " << context
                       << ", got " << dims[i];
    CHECK(product <= INT64_MAX / dims[i]) << "FlattenTileNdTo2D: integer overflow when computing " << context;
    product *= dims[i];
  }
  return product;
}

/// Decompose a flat batch index into per-dimension indices for the given batch shape.
/// e.g. flat_index=5 with batch_shape=[2,3] → indices=[1,2].
std::vector<int64_t> BuildBatchIndices(int64_t flat_index, const std::vector<int64_t>& batch_shape) {
  std::vector<int64_t> indices;
  if (batch_shape.empty()) return indices;

  indices.reserve(batch_shape.size());
  for (size_t dim = 0; dim < batch_shape.size(); ++dim) {
    int64_t stride = 1;
    for (size_t suffix = dim + 1; suffix < batch_shape.size(); ++suffix) {
      CHECK(stride <= INT64_MAX / batch_shape[suffix])
          << "FlattenTileNdTo2D: integer overflow while computing batch stride";
      stride *= batch_shape[suffix];
    }
    int64_t linear_index = (dim + 1 < batch_shape.size()) ? flat_index / stride : flat_index;
    indices.push_back(linear_index % batch_shape[dim]);
  }
  return indices;
}

/// Compute the flat batch index for an operand whose batch shape may be smaller
/// than the output batch shape (NumPy-style broadcast: size-1 dims map to index 0).
int64_t BuildOperandFlatBatchIndex(const std::vector<int64_t>& operand_batch_shape,
                                   const std::vector<int64_t>& output_batch_shape,
                                   const std::vector<int64_t>& output_batch_indices) {
  if (operand_batch_shape.empty()) return 0;

  CHECK(output_batch_shape.size() >= operand_batch_shape.size())
      << "FlattenTileNdTo2D: output batch rank must cover operand batch rank";
  CHECK(output_batch_indices.size() == output_batch_shape.size())
      << "FlattenTileNdTo2D: output batch indices must match output batch rank";

  int64_t flat_index = 0;
  const size_t lead_dims = output_batch_shape.size() - operand_batch_shape.size();
  for (size_t i = 0; i < operand_batch_shape.size(); ++i) {
    int64_t operand_dim = operand_batch_shape[i];
    int64_t batch_index = operand_dim == 1 ? 0 : output_batch_indices[lead_dims + i];
    CHECK(flat_index <= INT64_MAX / operand_dim)
        << "FlattenTileNdTo2D: integer overflow while flattening broadcasted batch index";
    flat_index = flat_index * operand_dim + batch_index;
  }
  return flat_index;
}

/// Normalize a potentially negative axis index (Python-style) to a valid range.
int64_t NormalizeAxisIndex(int64_t axis, size_t ndim, const std::string& context) {
  int64_t normalized = axis;
  if (normalized < 0) {
    normalized += static_cast<int64_t>(ndim);
  }
  CHECK(normalized >= 0 && normalized < static_cast<int64_t>(ndim))
      << "FlattenTileNdTo2D: axis " << axis << " is out of range for rank " << ndim << " in " << context;
  return normalized;
}

/// Check whether (axis1, axis2) is a swap of the last two dimensions.
bool IsTrailingMatrixAxisSwap(int64_t axis1, int64_t axis2, size_t ndim) {
  int64_t trailing_axis0 = static_cast<int64_t>(ndim) - 2;
  int64_t trailing_axis1 = static_cast<int64_t>(ndim) - 1;
  return (axis1 == trailing_axis0 && axis2 == trailing_axis1) ||
         (axis1 == trailing_axis1 && axis2 == trailing_axis0);
}

/**
 * @brief Extract yield value types from the first YieldStmt found in a statement list.
 *
 * Recurses into SeqStmts and ScopeStmt to find yields in nested containers.
 */
std::vector<TypePtr> FindYieldTypes(const std::vector<StmtPtr>& stmts) {
  for (const auto& stmt : stmts) {
    if (auto yield = As<YieldStmt>(stmt)) {
      std::vector<TypePtr> types;
      types.reserve(yield->value_.size());
      for (const auto& val : yield->value_) {
        types.push_back(val->GetType());
      }
      return types;
    }
    if (auto seq = As<SeqStmts>(stmt)) {
      auto found = FindYieldTypes(seq->stmts_);
      if (!found.empty()) return found;
    }
    if (auto scope = As<ScopeStmt>(stmt)) {
      auto body_stmts = FlattenToStmts(scope->body_);
      auto found = FindYieldTypes(body_stmts);
      if (!found.empty()) return found;
    }
  }
  return {};
}

}  // namespace rewrite_internal
}  // namespace flatten_tile_nd_to_2d
}  // namespace ir
}  // namespace pypto

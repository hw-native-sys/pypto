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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_TILE_CONVERSION_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_TILE_CONVERSION_UTILS_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"

namespace pypto::ir::tile_conversion_utils {

/// Whether the valid sub-box over the leading "row" dims ``[0, ndim-1)`` of an ND
/// load window flattens to a contiguous row axis in row-major order — the
/// precondition for collapsing an ND ``tile.load`` into a 2D ND2NZ GlobalTensor.
///
/// Scanning the row dims from the outermost in: any number of leading singleton
/// (``valid == 1``) dims, then at most one partial "boundary" dim, after which
/// every dim must span its full tensor extent. (A whole load, a batch sub-range,
/// and batch-1 matrix-row tiling all satisfy this; only a partial dim *under a
/// non-singleton outer dim* — a multi-batch slice that also cuts the matrix-row
/// dim — makes the rows non-contiguous.) Offsets do not affect contiguity (they
/// fold into the partition's row offset), so only valid vs tensor extents matter.
///
/// This is the single source of truth shared by the ``FlattenTileNdTo2D``
/// whole-vs-per-batch routing decision and the ``tile.load`` codegen contiguity
/// guard, so the two stay in lockstep. ``valid`` and ``tensor_dims`` must be the
/// same length (the load window rank); a mismatch returns true (the caller owns
/// that guard).
inline bool IsRowMajorCollapseContiguous(const std::vector<ExprPtr>& valid,
                                         const std::vector<ExprPtr>& tensor_dims) {
  if (valid.size() != tensor_dims.size()) return true;
  const size_t ndim = valid.size();
  bool past_boundary = false;
  for (size_t i = 0; i + 1 < ndim; ++i) {
    const bool is_full = AreExprsEqual(valid[i], tensor_dims[i]);
    if (past_boundary) {
      if (!is_full) return false;
      continue;
    }
    auto vi = As<ConstInt>(valid[i]);
    if (!(vi && vi->value_ == 1)) past_boundary = true;
  }
  return true;
}

/// Build a MakeTuple of zero INDEX constants for load/store offsets.
inline ExprPtr MakeZeroOffsets(size_t ndim, const Span& span) {
  std::vector<ExprPtr> zeros;
  zeros.reserve(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    zeros.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, span));
  }
  return std::make_shared<MakeTuple>(zeros, span);
}

/// Build a MakeTuple from a shape vector.
inline ExprPtr MakeShapeTuple(const std::vector<ExprPtr>& shape, const Span& span) {
  return std::make_shared<MakeTuple>(shape, span);
}

/// Build a signal-slot offset tuple [rank_expr, 0] for notify/wait ops.
/// The signal matrix is shape [nranks, 1], so two INDEX elements suffice.
inline ExprPtr MakeSignalOffsets(const ExprPtr& rank_expr, const Span& span) {
  std::vector<ExprPtr> elements = {rank_expr, std::make_shared<ConstInt>(0, DataType::INDEX, span)};
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

/// Build a 2D signal-slot offset tuple [row_expr, rank_expr] for 2D signal
/// matrices (e.g. ring allreduce signals of shape [2*(NR-1), NR]).
inline ExprPtr MakeSignalOffsets(const ExprPtr& rank_expr, const ExprPtr& row_expr, const Span& span) {
  std::vector<ExprPtr> elements = {row_expr, rank_expr};
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

}  // namespace pypto::ir::tile_conversion_utils

#endif  // PYPTO_IR_TRANSFORMS_UTILS_TILE_CONVERSION_UTILS_H_

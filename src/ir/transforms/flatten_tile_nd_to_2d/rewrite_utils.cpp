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

#include <algorithm>
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
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
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
    elems *= static_cast<uint64_t>(ci->value_);
  }
  const uint64_t bytes_per = std::max<uint64_t>(1, original_type->dtype_.GetBit() / 8);
  return elems * bytes_per;
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

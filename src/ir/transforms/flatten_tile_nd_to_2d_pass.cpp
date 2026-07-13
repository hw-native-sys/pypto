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
#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

using transform_utils::FlattenToStmts;
using transform_utils::Substitute;

namespace {

// ============================================================================
// Helpers
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

/**
 * @brief Replace a rebuilt call's result type without changing its operation.
 *
 * OpRegistry re-deduction is useful for validating substituted operands, but a
 * contextual Tensor-to-Tile conversion may intentionally have assigned a
 * narrower valid_shape (or a pad contract) to the call result.  Re-running the
 * ordinary deducer must not erase that authoritative contract.
 */
CallPtr RetypeCall(const CallPtr& rebuilt, const TypePtr& result_type) {
  return std::make_shared<Call>(rebuilt->op_, rebuilt->args_, rebuilt->kwargs_, rebuilt->attrs_, result_type,
                                rebuilt->span_);
}

/**
 * @brief Reapply an authoritative <=2D TileType after validating its physical contract.
 */
CallPtr RetypeTileCall(const CallPtr& rebuilt, const TileTypePtr& authoritative, const std::string& context,
                       const Span& span) {
  auto deduced = As<TileType>(rebuilt->GetType());
  INTERNAL_CHECK_SPAN(deduced && authoritative, span)
      << "FlattenTileNdTo2D: " << context << " must produce a TileType";
  INTERNAL_CHECK_SPAN(
      deduced->dtype_ == authoritative->dtype_ && AreExprVectorsEqual(deduced->shape_, authoritative->shape_),
      span)
      << "FlattenTileNdTo2D: " << context
      << " re-deduction changed the physical shape or dtype while preserving its contextual result contract";
  return RetypeCall(rebuilt, authoritative);
}

/// Merge an ND ``valid_shape`` into its 2D form ``[product(leading), last]``,
/// allowing dynamic (non-ConstInt) entries — unlike ComputeMergedShape, which
/// requires static dims. Static factors are folded into a single ConstInt; the
/// identity factor 1 is dropped. This lets a dynamic ``valid_shape`` (e.g. the
/// ``min(CHUNK, D - c)`` tail from the dynamic-tile strip-mine below) survive the
/// flatten of the physical tile shape rather than being reset to the full static
/// shape.
///
/// Soundness precondition (valid_shape B2): the ND valid region must map to a
/// *contiguous* prefix of the flattened rows — the only thing
/// ``(valid_row, valid_col)`` can express. For row dims ``valid[0 .. n-2]`` (the
/// innermost dim ``n-1`` becomes the free column) read most-significant first,
/// this holds iff there is a single "free" row dim: every row dim before it is
/// unit (its index is pinned to 0, so it contributes no stride) and every row dim
/// after it is fully valid (``valid == shape``). A partially-valid row dim that
/// sits below a non-unit more-significant dim yields a *strided* region and is
/// rejected here rather than approximated by the product fold (which would
/// silently mark batch-0 padding valid and real batch-1 data invalid).
///
/// ``CHECK_SPAN`` (user error, not INTERNAL): the offending ND ``valid_shape`` is
/// user-authored via ``pl.load(valid_shapes=...)``, so this is a documented
/// user-facing limitation (see ``.claude/rules/error-checking.md`` carve-out).
std::vector<ExprPtr> ComputeMergedValidShape(const std::vector<ExprPtr>& valid,
                                             const std::vector<ExprPtr>& shape, const Span& span) {
  INTERNAL_CHECK_SPAN(valid.size() == shape.size(), span)
      << "FlattenTileNdTo2D: valid_shape rank (" << valid.size() << ") must match tile shape rank ("
      << shape.size() << ") before merging";
  // Row dims are ``valid[0 .. n-2]``; the innermost dim (``n-1``) is the free
  // column. The most-significant row dim that is not pinned to a unit extent is
  // the free row dim (it, together with any folded leading unit dims, becomes
  // valid_row). Every row dim after it must be fully valid or the flattened valid
  // rows are strided and cannot be expressed as a single valid_row extent.
  //
  // An EMPTY valid region (any dim provably 0 — e.g. the ``create_tile(valid_shape=
  // [0, 0, 0])`` empty accumulator) denotes the empty set, which is trivially a
  // contiguous prefix of length 0. The precondition below reasons about which dims
  // are *pinned* (valid == 1) and would otherwise mistake a 0 dim for a partially
  // valid middle dim and reject with a misleading message. The product fold handles
  // it correctly on its own: a zero row dim folds to valid_row = 0, and a zero
  // column dim yields valid_col = 0 — either way, zero elements.
  const bool region_is_empty = std::any_of(valid.begin(), valid.end(), [](const ExprPtr& e) {
    auto ci = As<ConstInt>(e);
    return ci && ci->value_ == 0;
  });

  // A rank-0/1 valid_shape has no row dims: the fold below degenerates to the
  // identity and there is nothing to check. Guard explicitly — ``valid.size() - 1``
  // is unsigned and would wrap to SIZE_MAX on an empty vector.
  if (!region_is_empty && valid.size() >= 2) {
    const size_t num_row_dims = valid.size() - 1;
    size_t free_dim = 0;
    while (free_dim + 1 < num_row_dims) {
      auto valid_ci = As<ConstInt>(valid[free_dim]);
      auto shape_ci = As<ConstInt>(shape[free_dim]);
      // A physical unit axis cannot introduce a gap in flattened storage even
      // when its validity is a symbolic 0/1 gate.  Folding that gate into a
      // later partial row extent remains an exact contiguous prefix.
      if ((valid_ci && valid_ci->value_ == 1) || (shape_ci && shape_ci->value_ == 1)) {
        ++free_dim;
      } else {
        break;
      }
    }
    for (size_t i = free_dim + 1; i < num_row_dims; ++i) {
      CHECK_SPAN(AreExprsEqual(valid[i], shape[i]), span)
          << "tile dim " << i << " is partially valid (valid_shape[" << i << "] != shape[" << i
          << "]) below a non-unit outer dim; an ND valid region with such a partial middle dim is not a "
          << "contiguous row range and cannot be folded into (valid_row, valid_col). Only the outermost "
          << "non-unit dim and the innermost dim may be partially valid.";
    }
  }

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

/**
 * @brief Flatten an authoritative ND tile result contract to its PTO 2D form.
 *
 * This is a representation reshape, not a new allocation.  The physical shape
 * and valid prefix are folded together, while production layout/fractal/pad and
 * the memory-space intent remain authoritative.  Addressing metadata is rebuilt
 * later for the flattened SSA value and therefore is not copied across ranks.
 */
TileTypePtr FlattenTileTypeContract(const TileTypePtr& original, const std::string& context,
                                    const Span& span) {
  INTERNAL_CHECK_SPAN(original && original->shape_.size() > 2, span)
      << "FlattenTileNdTo2D: " << context << " requires an ND TileType";
  auto [merged, last] = ComputeMergedShape(original->shape_, context);
  auto flat_shape = Make2DShapeExprs(merged, last, span);
  TileView view = tile_view_semantics::GetEffectiveTileView(*original);
  view.valid_shape = ComputeMergedValidShape(GetValidShape(original), original->shape_, span);
  view.stride.clear();
  view.start_offset = nullptr;
  return std::make_shared<TileType>(flat_shape, original->dtype_, std::nullopt, view,
                                    original->memory_space_);
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

// ============================================================================
// Precondition validation
// ============================================================================

/**
 * @brief Visitor that validates preconditions for the FlattenTileNdTo2D pass.
 *
 * Checks:
 * 1. All tile shapes are static (ConstInt)
 * 2. All tile reduce ops (tile.sum/max/min) on >2D tiles reduce the last axis
 * 3. No tile.read/tile.write/tile.slice on >2D tiles
 */
class PreconditionChecker : public IRVisitor {
 public:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->value_)) {
      CheckCall(call);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->expr_)) {
      CheckCall(call);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  static void CheckStaticShape(const TileTypePtr& tile_type, const std::string& op_name) {
    if (!tile_type) return;
    for (size_t i = 0; i < tile_type->shape_.size(); ++i) {
      CHECK(As<ConstInt>(tile_type->shape_[i]))
          << "FlattenTileNdTo2D: tile op '" << op_name << "' has a dynamic (non-constant) "
          << "physical dimension " << i << ", which cannot be flattened to 2D or lowered to PTO. "
          << "Hardware tiles map to fixed-size on-chip buffers, so every tile dimension must "
          << "be a compile-time constant; a pl.dynamic dimension has no static bound and cannot "
          << "back a tile dimension directly. Fix by either: (1) iterating/tiling the dynamic "
          << "dimension with pl.range/pl.parallel so each per-iteration tile slice is static, or "
          << "(2) reshaping the tensor to 2D before the InCore (pl.at) scope so the dynamic extent "
          << "lands on the pl.parallel loop bound instead of inside the tile shape.";
    }
  }

  void CheckCall(const CallPtr& call) {
    if (!call || !call->op_) return;
    auto gv = As<GlobalVar>(call->op_);
    if (gv) return;  // Skip function calls

    const auto& name = call->op_->name_;
    if (name.substr(0, 5) != "tile.") return;

    // Check static shapes on any tile-typed argument and result
    for (const auto& arg : call->args_) {
      CheckStaticShape(As<TileType>(arg->GetType()), name);
    }
    CheckStaticShape(As<TileType>(call->GetType()), name);

    // Disallow tile.read/tile.write/tile.slice on >2D tiles
    if (name == "tile.read" || name == "tile.write" || name == "tile.slice") {
      if (!call->args_.empty()) {
        auto input_tile = As<TileType>(call->args_[0]->GetType());
        CHECK(!IsNdTile(input_tile)) << "FlattenTileNdTo2D: " << name << " is not supported on >2D tiles";
      }
    }

    // Check reduce ops reduce the last axis
    if (name == "tile.sum" || name == "tile.max" || name == "tile.min") {
      if (!call->args_.empty()) {
        auto input_tile = As<TileType>(call->args_[0]->GetType());
        if (IsNdTile(input_tile)) {
          int axis = call->GetKwarg<int>("axis", -1);
          int last_axis = static_cast<int>(input_tile->shape_.size()) - 1;
          CHECK(axis == last_axis) << "FlattenTileNdTo2D: tile reduce op '" << name
                                   << "' must reduce along the last axis "
                                   << "(axis=" << last_axis << "), but got axis=" << axis;
          // keepdim must be True so the output stays 2D after flatten
          bool keepdim = call->GetKwarg<bool>("keepdim", false);
          CHECK(keepdim) << "FlattenTileNdTo2D: tile reduce op '" << name
                         << "' on >2D tile must use keepdim=True to maintain 2D output shape";
        }
      }
    }
  }
};

// ============================================================================
// Main transformation
// ============================================================================

struct FlattenContext {
  std::unordered_map<const Var*, VarPtr> var_map;  // old Var* -> new 2D var

  void Insert(const VarPtr& old_var, const VarPtr& new_var) { var_map[old_var.get()] = new_var; }

  void Erase(const VarPtr& var) { var_map.erase(var.get()); }
};

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

// ============================================================================
// Batch matmul lowering
// ============================================================================
//
// tile.batch_matmul performs batched matrix multiplication on rank>2 tiles:
//   lhs [..., M, K] x rhs [..., K, N] -> result [..., M, N]
// where "..." are broadcast-compatible batch dimensions.
//
// The 2D backend only supports tile.matmul on rank-2 tiles. This lowering
// eliminates tile.batch_matmul by unrolling the batch dimensions at compile
// time (all shapes are static) into a flat sequence of 2D tile.matmul calls.
//
// Overall flow:
//
//   1. Normalize operands — peel safe batch-only tile.reshape wrappers and flag a
//      tile.transpose_view operand (the b_trans / a_trans form: the view already
//      presents [.., K, N], so batch_matmul itself carries no transpose semantic).
//
//   2. Broadcast batch dimensions — compute the output batch shape via
//      NumPy-style broadcasting (e.g. [2,1] x [1,3] -> [2,3]).
//
//   3. Detect direct-store fusion — if the very next statement is a tile.store
//      consuming this result, fuse per-batch stores directly instead of
//      assembling into a temporary tile. This avoids an intermediate buffer.
//
//   4. Capacity gate — decide once whether both operands' whole tiles fit Mat
//      together (BatchOperandsWholeFit).
//
//   5. Unroll — for each flat batch index 0..batch_count-1:
//      a. Decompose the flat index into per-dim indices for lhs and rhs,
//         respecting broadcast (size-1 dims always map to index 0).
//      b. Extract the 2D [M,K] / [K,N] page (ExtractBatchPage). When the whole
//         tiles fit, every operand is sliced from its kept whole Mat tile (row
//         slice for plain operands, column slice for tile.transpose_view); when
//         they do not, each operand is loaded per batch instead. Transposed
//         operands are realised by a zero-copy tile.transpose_view — never a copy.
//      c. Emit tile.matmul(lhs_2d, rhs_2d).
//      d. Cast dtype if matmul output (FP32) differs from expected result dtype.
//      e. Either tile.store (fused path) or tile.assemble into output tile.
//
// The result is a flat 2D tile [batch_count*M, N] (non-fused) or a chain
// of per-batch tile.store calls (fused), with no tile.batch_matmul remaining.
//

/// Map from Var raw pointer to its defining AssignStmt, for O(1) def lookup.
using AssignDefMap = std::unordered_map<const Var*, AssignStmtPtr>;

AssignDefMap BuildAssignDefMap(const std::vector<StmtPtr>& stmts) {
  AssignDefMap map;
  for (const auto& stmt : stmts) {
    if (auto assign = As<AssignStmt>(stmt)) {
      map[assign->var_.get()] = assign;
    }
  }
  return map;
}

/// Parsed information about a batch_matmul operand.
struct BatchOperandInfo {
  ExprPtr operand;                   ///< After var_map substitution
  ExprPtr original_operand;          ///< Before substitution (for def lookup)
  TileTypePtr operand_type;          ///< Type after substitution
  TileTypePtr original_type;         ///< Type before substitution
  bool from_transpose_view = false;  ///< True if the operand is a tile.transpose_view result.
                                     ///< Its trailing two dims are already swapped to the matmul
                                     ///< orientation, but when flattened the batch is concatenated
                                     ///< on the COLUMN axis ([K, B*N]), so per-batch extraction is a
                                     ///< column slice (offset {0, b*N}) rather than a row slice.
  bool whole_fits = true;            ///< Whether both operands' whole tiles fit Mat together. When
                                     ///< false, ExtractBatchPage loads this operand per batch (from
                                     ///< base_load) instead of slicing a kept whole tile.
  CallPtr base_load;                 ///< Underlying natural tile.load for this operand (traced through
                                     ///< the tile.transpose_view when from_transpose_view). Used by the
                                     ///< !fit per-batch path to re-emit a per-batch load.
};

/// Resolve an inline or single-definition `op_name` wrapper around a batch_matmul operand.
CallPtr ResolveBatchOperandCall(const ExprPtr& operand_expr, const AssignDefMap& def_map,
                                const std::string& op_name) {
  if (auto call = As<Call>(operand_expr)) {
    if (call->op_ && call->op_->name_ == op_name) return call;
  }
  if (auto operand_var = As<Var>(operand_expr)) {
    auto def_it = def_map.find(operand_var.get());
    if (def_it != def_map.end()) {
      if (auto call = As<Call>(def_it->second->value_)) {
        if (call->op_ && call->op_->name_ == op_name) return call;
      }
    }
  }
  return nullptr;
}

/// Check if a `tile.reshape` call is safe to peel when feeding `tile.batch_matmul`.
///
/// A reshape is "safe to peel" when it only reinterprets the batch portion of
/// the shape and leaves the trailing (M, N) matrix dims untouched:
///   * input and output ranks are both >= 2,
///   * the last two dims (the matmul page) are identical static values,
///   * the product of the leading batch dims is the same on both sides.
bool IsSafePeelableBatchMatmulReshape(const CallPtr& reshape_call) {
  if (!reshape_call || !reshape_call->op_ || !IsOp(reshape_call, "tile.reshape")) {
    return false;
  }
  if (reshape_call->args_.size() != 2) return false;

  auto out_type = As<TileType>(reshape_call->GetType());
  auto in_type = As<TileType>(reshape_call->args_[0]->GetType());
  if (!out_type || !in_type) return false;
  if (out_type->shape_.size() < 2 || in_type->shape_.size() < 2) return false;

  // Trailing matmul page must be preserved.
  auto in_rows = As<ConstInt>(in_type->shape_[in_type->shape_.size() - 2]);
  auto in_cols = As<ConstInt>(in_type->shape_.back());
  auto out_rows = As<ConstInt>(out_type->shape_[out_type->shape_.size() - 2]);
  auto out_cols = As<ConstInt>(out_type->shape_.back());
  if (!in_rows || !in_cols || !out_rows || !out_cols) return false;
  if (in_rows->value_ != out_rows->value_ || in_cols->value_ != out_cols->value_) return false;

  // Batch element count must be preserved (so the reshape is a pure batch reinterpretation).
  auto static_batch_product = [](const std::vector<ExprPtr>& shape) -> std::optional<int64_t> {
    int64_t product = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
      auto ci = As<ConstInt>(shape[i]);
      if (!ci) return std::nullopt;
      product *= ci->value_;
    }
    return product;
  };
  auto in_batch = static_batch_product(in_type->shape_);
  auto out_batch = static_batch_product(out_type->shape_);
  if (!in_batch || !out_batch || *in_batch != *out_batch) return false;
  return true;
}

/// Peel safe tile.reshape wrappers around a batch_matmul operand.
///
/// Peeling lets `LowerBatchMatmul` look through e.g. `tile.reshape([1, M, N],
/// [1, 1, M, N])` and reuse the upstream `tile.load` operand directly. The
/// alternative (the rank>2 fallback in `ExtractBatchPage`) would otherwise emit a
/// redundant ND `tile.slice` + `tile.reshape` chain per batch element, which
/// can lower to invalid degenerate tiles for zero-valid sub-blocks.
///
/// Iterates so nested reshapes (e.g. two consecutive safe reshapes) all peel.
/// Returns the deepest safe operand, or the input unchanged when no reshape
/// is found / the reshape fails the safety conditions.
ExprPtr PeelSafeBatchReshape(const ExprPtr& operand_expr, const AssignDefMap& def_map) {
  ExprPtr current = operand_expr;
  while (true) {
    CallPtr reshape_call;
    if (auto call = As<Call>(current)) {
      if (IsOp(call, "tile.reshape")) {
        reshape_call = call;
      }
    }
    if (!reshape_call) {
      if (auto var = As<Var>(current)) {
        auto def_it = def_map.find(var.get());
        if (def_it != def_map.end()) {
          if (auto call = As<Call>(def_it->second->value_)) {
            if (IsOp(call, "tile.reshape")) {
              reshape_call = call;
            }
          }
        }
      }
    }
    if (!reshape_call) return current;
    if (!IsSafePeelableBatchMatmulReshape(reshape_call)) return current;
    current = reshape_call->args_[0];
  }
}

/// Whether a natural `tile.load`'s whole source window collapses to a contiguous
/// 2D row axis (the precondition the codegen ND2NZ collapse enforces). A
/// non-contiguous whole load (a partial middle dim under a non-singleton outer
/// dim, e.g. a multi-batch slice that also cuts the matrix-row dim) cannot be
/// legalized as one 2D ND2NZ load, so the operand must instead be re-emitted per
/// batch (ExtractBatchPage !fit path). Returns true (keep whole) when the load is
/// absent / 2D / dynamic-shaped. The contiguity rule itself is shared with the
/// codegen guard via `IsRowMajorCollapseContiguous`, so routing and guard agree.
bool WholeLoadContiguous(const CallPtr& base_load) {
  if (!base_load || base_load->args_.size() < 4) return true;
  auto tensor_type = AsTensorTypeLike(base_load->args_[0]->GetType());
  auto valid = As<MakeTuple>(base_load->args_[3]);
  if (!tensor_type || !valid) return true;
  const size_t ndim = valid->elements_.size();
  if (ndim <= 2 || tensor_type->shape_.size() != ndim) return true;
  return tile_conversion_utils::IsRowMajorCollapseContiguous(valid->elements_, tensor_type->shape_);
}

/// The per-operand whole-vs-per-batch routing decision, shared by LowerBatchMatmul,
/// LowerBatchMatmulAcc, and the dead-load drop pre-scan so all three stay in sync:
/// keep this operand whole only when both operands' whole tiles fit Mat together
/// (the joint `capacity_fits` gate) AND its whole load collapses contiguously;
/// otherwise it is re-emitted per batch.
bool KeepOperandWhole(bool capacity_fits, const CallPtr& base_load) {
  return capacity_fits && WholeLoadContiguous(base_load);
}

/// Trace a batch_matmul operand var to its underlying natural `tile.load` (through
/// safe batch reshape wrappers and a tile.transpose_view), mirroring
/// NormalizeBatchMatmulOperand's base_load resolution. Used by the drop pre-scan
/// to apply the same whole-vs-per-batch routing decision as LowerBatchMatmul.
CallPtr TraceOperandBaseLoad(const ExprPtr& operand_expr, const AssignDefMap& def_map) {
  ExprPtr base = PeelSafeBatchReshape(operand_expr, def_map);
  if (auto tv = ResolveBatchOperandCall(base, def_map, "tile.transpose_view")) {
    if (!tv->args_.empty()) base = tv->args_[0];
  }
  return ResolveBatchOperandCall(base, def_map, "tile.load");
}

/// Normalize one batch_matmul operand:
///  - peel safe batch-only tile.reshape wrappers that only reinterpret batch dims
///  - recognize a tile.transpose_view operand (the canonical b_trans/a_trans form):
///    its trailing dims are already swapped to the matmul orientation, so no
///    per-batch transpose is needed; we only record that per-batch extraction must
///    column-slice (the flattened view concatenates batch on the column axis).
///  - return the base operand plus type information.
BatchOperandInfo NormalizeBatchMatmulOperand(const ExprPtr& operand_expr, const std::string& operand_name,
                                             const AssignDefMap& def_map, const FlattenContext& ctx) {
  BatchOperandInfo info;
  // Peel safe batch-only tile.reshape wrappers first so the transpose_view check
  // below sees the underlying operand directly.
  ExprPtr base_operand = PeelSafeBatchReshape(operand_expr, def_map);

  // A b_trans/a_trans operand arrives as a tile.transpose_view (issues #1776 / ND
  // extension): the view already presents the operand in [.., K, N] orientation, so
  // batch_matmul carries no transpose semantic. Keep the view as the operand (it is
  // a whole Mat tile we slice per batch); just flag column-slicing.
  if (ResolveBatchOperandCall(base_operand, def_map, "tile.transpose_view") != nullptr) {
    info.from_transpose_view = true;
  }

  info.original_operand = base_operand;
  info.original_type = As<TileType>(base_operand->GetType());
  CHECK(info.original_type) << "FlattenTileNdTo2D: tile.batch_matmul " << operand_name
                            << " expects TileType operand, but got " << base_operand->GetType()->TypeName();

  // Trace the underlying NATURAL tile.load (through the tile.transpose_view when
  // transposed). The !fit per-batch path re-emits a per-batch load from it.
  ExprPtr load_src = base_operand;
  if (info.from_transpose_view) {
    auto tv = ResolveBatchOperandCall(base_operand, def_map, "tile.transpose_view");
    if (tv && !tv->args_.empty()) load_src = tv->args_[0];
  }
  info.base_load = ResolveBatchOperandCall(load_src, def_map, "tile.load");

  info.operand = Substitute(base_operand, ctx.var_map);
  info.operand_type = As<TileType>(info.operand->GetType());
  CHECK(info.operand_type) << "FlattenTileNdTo2D: tile.batch_matmul substituted " << operand_name
                           << " expects TileType operand, but got " << info.operand->GetType()->TypeName();
  return info;
}

/// Build batch-adjusted offset elements: add batch indices to the batch dimensions
/// of base offsets, then append the trailing matrix-dimension offsets unchanged.
std::vector<ExprPtr> BuildBatchAdjustedOffsets(const std::vector<ExprPtr>& base_offset_elems,
                                               const std::vector<int64_t>& batch_indices, size_t batch_rank,
                                               const Span& span) {
  std::vector<ExprPtr> adjusted;
  adjusted.reserve(base_offset_elems.size());
  for (size_t dim = 0; dim < batch_rank; ++dim) {
    if (batch_indices[dim] == 0) {
      adjusted.push_back(base_offset_elems[dim]);
    } else {
      auto offset = std::make_shared<ConstInt>(batch_indices[dim], DataType::INDEX, span);
      adjusted.push_back(MakeCanonicalIndexAdd(base_offset_elems[dim], offset, span));
    }
  }
  for (size_t dim = batch_rank; dim < base_offset_elems.size(); ++dim) {
    adjusted.push_back(base_offset_elems[dim]);
  }
  return adjusted;
}

/// Result of extracting a 2D batch page from a rank>2 operand.
struct BatchPageResult {
  VarPtr var;                  ///< The 2D variable (possibly transposed)
  std::vector<StmtPtr> stmts;  ///< Statements emitted to produce it
};

/// Extract the 2D matrix page for one batch index from a batch_matmul operand.
///
/// Every operand — lhs or rhs, transposed or not, load- or move-sourced — is
/// handled identically:
///  * whole_fits (default): the operand's whole tile is already in Mat; take its
///    [source_rows, source_cols] page with a single tile.slice — a ROW slice for a
///    plain (row-batched) operand, a COLUMN slice for a tile.transpose_view
///    (column-batched) operand. A broadcast operand reuses its single page.
///  * !whole_fits (large operands): load THIS operand per batch from its
///    underlying natural tile.load, adding a per-batch tile.transpose_view when the
///    operand is transposed. The dead whole load/view is dropped during rewriting.
///
/// The operand is always 2D by the time it reaches here (loads flatten to 2D,
/// transpose_views are 2D, safe batch-only reshapes are peeled to the 2D load).
BatchPageResult ExtractBatchPage(const BatchOperandInfo& info, const std::vector<int64_t>& operand_dims,
                                 const std::vector<int64_t>& operand_batch_shape, int64_t batch_index,
                                 const std::string& base_name, const FlattenContext& ctx,
                                 const OpRegistry& op_registry, const Span& span) {
  BatchPageResult page;
  const auto& operand = info.operand;
  const auto& operand_type = info.operand_type;

  int64_t source_rows = operand_dims[operand_dims.size() - 2];
  int64_t source_cols = operand_dims.back();
  std::string suffix = std::to_string(batch_index);

  VarPtr current;

  if (!info.whole_fits && info.base_load) {
    // !fit + load (GM): the operands' whole tiles do not fit Mat together, so load
    // THIS operand PER BATCH from its underlying natural tile.load (a per-batch
    // [1,..,X,Y] window → 2D [X,Y], which the hardware ND2NZ path accepts as the
    // leading dims are 1). A transposed operand then gets a per-batch
    // tile.transpose_view — same as the whole-tile path, just one batch at a time.
    // The whole load/view is dropped upstream (no longer referenced) so it does not
    // occupy L1.
    //
    // NOTE: this only covers load-sourced (GM) operands. A move-sourced operand
    // (V2C mixed kernel: a Vec compute result moved to Mat) has base_load == null,
    // so a !fit V2C operand falls through to the whole-slice path below — correct
    // only while the whole moved tile fits the fixed cross-core ring. A per-batch
    // V2C move for large shapes is a deferred follow-up (see BatchOperandsWholeFit).
    auto load_tensor = info.base_load->args_[0];
    auto load_tensor_type = AsTensorTypeLike(load_tensor->GetType());
    auto base_offsets = As<MakeTuple>(info.base_load->args_[1]);
    auto base_shapes = As<MakeTuple>(info.base_load->args_[2]);
    INTERNAL_CHECK_SPAN(load_tensor_type && base_offsets && base_shapes &&
                            load_tensor_type->shape_.size() >= 2 &&
                            base_shapes->elements_.size() == load_tensor_type->shape_.size(),
                        span)
        << "FlattenTileNdTo2D: !fit per-batch load expects a tensor-backed tile.load with rank >= 2";
    // Use the load's WINDOW matrix dims (the actual sliced tile), not the source
    // tensor's full trailing dims — they differ when the operand is a partial
    // sub-tile of a larger tensor (e.g. a multi-batch slice that also cuts the
    // matrix-row dim, the non-contiguous case routed here).
    const size_t win_rank = base_shapes->elements_.size();
    auto x_dim = As<ConstInt>(base_shapes->elements_[win_rank - 2]);
    auto y_dim = As<ConstInt>(base_shapes->elements_.back());
    INTERNAL_CHECK_SPAN(x_dim && y_dim, span)
        << "FlattenTileNdTo2D: !fit per-batch load needs static trailing dims";

    auto batch_indices = BuildBatchIndices(batch_index, operand_batch_shape);
    auto load_offset_elems =
        BuildBatchAdjustedOffsets(base_offsets->elements_, batch_indices, operand_batch_shape.size(), span);
    std::vector<int64_t> load_shape_values(operand_batch_shape.size(), 1);
    load_shape_values.push_back(x_dim->value_);
    load_shape_values.push_back(y_dim->value_);
    auto load_offsets = std::make_shared<MakeTuple>(load_offset_elems, span);
    auto load_shape = MakeShapeTupleFromInts(load_shape_values, span);

    // Preserve the original load window's validity while selecting one batch.
    // Reusing `load_shape` as the valid tuple makes every page appear full,
    // including pages beyond a partial batch tail and partial M/K/N extents.
    auto base_load_type = As<TileType>(info.base_load->GetType());
    INTERNAL_CHECK_SPAN(base_load_type, span)
        << "FlattenTileNdTo2D: !fit per-batch load expects a TileType base load";
    const std::vector<ExprPtr> base_valid = GetValidShape(base_load_type);
    INTERNAL_CHECK_SPAN(base_valid.size() == load_shape_values.size(), span)
        << "FlattenTileNdTo2D: !fit per-batch load valid_shape rank must match its window rank";
    std::vector<ExprPtr> page_valid;
    page_valid.reserve(base_valid.size());
    const auto one = std::make_shared<ConstInt>(1, DataType::INDEX, span);
    for (size_t dim = 0; dim < operand_batch_shape.size(); ++dim) {
      auto local_batch_index = std::make_shared<ConstInt>(batch_indices[dim], DataType::INDEX, span);
      page_valid.push_back(IntersectWindowValidDim(base_valid[dim], local_batch_index, one, one, span));
    }
    page_valid.insert(page_valid.end(), base_valid.end() - 2, base_valid.end());
    auto load_valid = std::make_shared<MakeTuple>(page_valid, span);
    std::vector<ExprPtr> load_args = {load_tensor, load_offsets, load_shape, load_valid};
    auto per_batch_load = op_registry.Create("tile.load", load_args, info.base_load->kwargs_, span);

    // The [1,..,X,Y] window deduces a rank>2 TileType; hardware tiles are 2D —
    // override the result to a 2D [X,Y] tile with the implicit Mat view.
    auto load_2d_shape = Make2DShapeExprs(x_dim->value_, y_dim->value_, span);
    auto pbl_type = As<TileType>(per_batch_load->GetType());
    INTERNAL_CHECK_SPAN(pbl_type, span) << "FlattenTileNdTo2D: !fit per-batch load must deduce a TileType";
    auto load_2d_view = tile_view_semantics::GetImplicitTileView(load_2d_shape, MemorySpace::Mat);
    load_2d_view.valid_shape = ComputeMergedValidShape(GetValidShape(pbl_type), pbl_type->shape_, span);
    load_2d_view.pad = tile_view_semantics::GetEffectiveTileView(*pbl_type).pad;
    auto load_2d_type = std::make_shared<TileType>(load_2d_shape, pbl_type->dtype_, std::nullopt,
                                                   load_2d_view, MemorySpace::Mat);
    auto load_2d =
        std::make_shared<Call>(per_batch_load->op_, load_args, per_batch_load->kwargs_, load_2d_type, span);
    current = std::make_shared<Var>(base_name + "_pbload_" + suffix, load_2d_type, span);
    page.stmts.push_back(std::make_shared<AssignStmt>(current, load_2d, span));

    if (info.from_transpose_view) {
      auto view = op_registry.Create("tile.transpose_view", {current}, {}, span);
      auto view_var = std::make_shared<Var>(base_name + "_pbview_" + suffix, view->GetType(), span);
      page.stmts.push_back(std::make_shared<AssignStmt>(view_var, view, span));
      current = view_var;
    }
    page.var = current;
    return page;

  } else {
    // Whole-fit: slice the [source_rows, source_cols] page from the kept whole 2D
    // tile. The operand is ALWAYS 2D here — a load flattens to a 2D result, a
    // tile.transpose_view is 2D, and a safe batch-only reshape is peeled to its 2D
    // load — so no rank>2 fallback is needed (verified: this assert fires on no
    // test across the full UT + ST suites). A plain operand is row-batched
    // ([B*rows, cols]) so the page is a row slice at row b*source_rows; a
    // tile.transpose_view operand is COLUMN-batched ([source_rows, B*source_cols] =
    // [K, B*N]: the whole-tile transpose concatenates the batches along the column
    // axis), so its page is a column slice at col b*source_cols. Either way the
    // page is [source_rows, source_cols].
    INTERNAL_CHECK_SPAN(operand_type->shape_.size() == 2, span)
        << "FlattenTileNdTo2D: batch_matmul operand must be flattened to 2D before "
           "ExtractBatchPage, got rank "
        << operand_type->shape_.size();
    std::vector<int64_t> offset_values = info.from_transpose_view
                                             ? std::vector<int64_t>{0, batch_index * source_cols}
                                             : std::vector<int64_t>{batch_index * source_rows, 0};
    auto offset = MakeShapeTupleFromInts(offset_values, span);
    auto shape = MakeShapeTupleFromInts({source_rows, source_cols}, span);
    auto slice = op_registry.Create("tile.slice", {operand, shape, offset}, span);
    current = std::make_shared<Var>(base_name + "_slice_" + suffix, slice->GetType(), span);
    page.stmts.push_back(std::make_shared<AssignStmt>(current, slice, span));
  }

  // No per-batch transpose: a transposed (b_trans/a_trans) operand arrives as a
  // tile.transpose_view whose page is already in the matmul orientation (the
  // column-slice above extracts batch_b^T directly).
  page.var = current;
  return page;
}

/// Detect whether the next statement is a tile.store consuming the batch_matmul result.
struct DirectStoreInfo {
  bool detected = false;
  AssignStmtPtr store_assign;
  CallPtr store_call;
};

DirectStoreInfo DetectDirectStore(const std::vector<StmtPtr>& stmts, size_t stmt_index,
                                  const VarPtr& result_var) {
  DirectStoreInfo info;
  if (stmt_index + 1 >= stmts.size()) return info;

  auto store_assign = As<AssignStmt>(stmts[stmt_index + 1]);
  auto store_call = store_assign ? As<Call>(store_assign->value_) : nullptr;
  if (!store_call || !IsOp(store_call, "tile.store")) return info;

  auto store_input = !store_call->args_.empty() ? As<Var>(store_call->args_[0]) : nullptr;
  if (!store_input || store_input.get() != result_var.get()) return info;

  info.detected = true;
  info.store_assign = store_assign;
  info.store_call = store_call;
  return info;
}

/// Result of lowering a tile.batch_matmul operation.
struct BatchMatmulResult {
  std::vector<StmtPtr> stmts;  ///< Emitted statements
  VarPtr output_var;           ///< Result variable (non-fused path)
  bool fused_store = false;    ///< True if direct-store fusion was applied
  VarPtr store_result_var;     ///< Final store var (fused path)
  VarPtr store_orig_var;       ///< Original store var being replaced (fused path)
};

/// Lower tile.batch_matmul into unrolled 2D tile.matmul calls.
///
/// Enumerates every batch index combination, extracts the 2D matrix page from each
/// operand, emits a tile.matmul per batch element, and either assembles results into
/// a flat 2D output tile or fuses directly into per-batch tile.store when possible.
BatchMatmulResult LowerBatchMatmul(const AssignStmtPtr& assign, const CallPtr& call,
                                   const std::vector<StmtPtr>& stmts, size_t stmt_index,
                                   const FlattenContext& ctx, const OpRegistry& op_registry,
                                   const Span& span) {
  BatchMatmulResult out;
  auto def_map = BuildAssignDefMap(stmts);

  // Normalize operands.
  auto lhs_info = NormalizeBatchMatmulOperand(call->args_[0], "lhs", def_map, ctx);
  auto rhs_info = NormalizeBatchMatmulOperand(call->args_[1], "rhs", def_map, ctx);
  // Route each operand to whole-slice vs per-batch independently: keep whole only
  // when the operands' whole tiles fit Mat together (capacity) AND this operand's
  // whole load collapses contiguously. A non-contiguous whole load (multi-batch +
  // partial matrix-row dim) is re-emitted per batch — each per-batch page is a
  // [1, X, Y] window that collapses cleanly — instead of erroring in codegen.
  const bool capacity_fits = BatchOperandsWholeFit(lhs_info.original_type, rhs_info.original_type);
  lhs_info.whole_fits = KeepOperandWhole(capacity_fits, lhs_info.base_load);
  rhs_info.whole_fits = KeepOperandWhole(capacity_fits, rhs_info.base_load);
  auto orig_result_type = As<TileType>(call->GetType());
  CHECK(orig_result_type) << "FlattenTileNdTo2D: tile.batch_matmul expects TileType result";

  // Detect direct-store fusion before asking whether the logical ND result can
  // be represented as one flattened valid box.  A result [B, m<M, N] is
  // non-contiguous in [B*M, N], but the fused path emits B independent tensor
  // partitions [1, m, N] and therefore represents it exactly.
  auto direct_store = DetectDirectStore(stmts, stmt_index, assign->var_);
  std::vector<ExprPtr> expected_flat_valid;
  if (!direct_store.detected) {
    expected_flat_valid =
        ComputeMergedValidShape(GetValidShape(orig_result_type), orig_result_type->shape_, span);
  }

  // Extract static dimensions.
  auto lhs_dims = ToStaticDims(lhs_info.original_type->shape_, "tile.batch_matmul lhs");
  auto rhs_dims = ToStaticDims(rhs_info.original_type->shape_, "tile.batch_matmul rhs");
  CHECK(lhs_dims.size() >= 2) << "FlattenTileNdTo2D: tile.batch_matmul lhs must be at least 2D";
  CHECK(rhs_dims.size() >= 2) << "FlattenTileNdTo2D: tile.batch_matmul rhs must be at least 2D";

  // Compute broadcast batch dimensions.
  std::vector<ExprPtr> lhs_batch_exprs(lhs_info.original_type->shape_.begin(),
                                       lhs_info.original_type->shape_.end() - 2);
  std::vector<ExprPtr> rhs_batch_exprs(rhs_info.original_type->shape_.begin(),
                                       rhs_info.original_type->shape_.end() - 2);
  auto broadcast_result = BroadcastShapes(lhs_batch_exprs, rhs_batch_exprs);
  CHECK(broadcast_result.success) << "FlattenTileNdTo2D: tile.batch_matmul batch dimensions must broadcast";

  auto output_batch_dims = ToStaticDims(broadcast_result.shape, "tile.batch_matmul output batch");
  int64_t batch_count = MultiplyStaticDims(output_batch_dims, "tile.batch_matmul output batch size");

  std::vector<int64_t> lhs_batch_dims(lhs_dims.begin(), lhs_dims.end() - 2);
  std::vector<int64_t> rhs_batch_dims(rhs_dims.begin(), rhs_dims.end() - 2);

  // Matrix dimensions. A transposed operand already arrives in matmul orientation
  // via its tile.transpose_view (original_type is the post-transpose [.., K, N]),
  // so the trailing two dims are used directly — no swap.
  int64_t lhs_rows = lhs_dims[lhs_dims.size() - 2];
  int64_t lhs_cols = lhs_dims.back();
  int64_t rhs_rows = rhs_dims[rhs_dims.size() - 2];
  int64_t rhs_cols = rhs_dims.back();

  // K-match is validated user-facing at op construction (DeduceTileBatchMatMulType);
  // a mismatch here would be a compiler bug in an earlier pass.
  INTERNAL_CHECK_SPAN(lhs_cols == rhs_rows, span)
      << "Internal error: tile.batch_matmul inner dimensions must match, but got " << lhs_cols << " and "
      << rhs_rows;

  // Fast path: batch_count == 1, non-fused, and no dtype cast required. The
  // result tile is exactly what a single tile.matmul produces (2D, Acc). Skip
  // the create + per-batch move-to-Vec + tile.assemble dance and let the Acc
  // tile flow directly to the consumer. This is essential when the consumer is
  // tile.matmul_acc / tile.batch_matmul_acc — those need an Acc accumulator,
  // and a Vec-staged tile would force an illegal cross-core Vec→Acc move at
  // codegen time. Any downstream Vec consumer can still insert its own Acc→Vec
  // move.
  //
  // Skip the fast path when the deduced tile.matmul accumulator dtype differs
  // from the requested orig_result_type dtype: returning the raw Acc tile
  // would leak the wider accumulator dtype (e.g. fp32/int32) instead of the
  // expected output dtype, and the cast must be inserted in Vec memory by the
  // general path below.
  if (batch_count == 1 && !direct_store.detected) {
    auto output_batch_indices = BuildBatchIndices(0, output_batch_dims);
    int64_t lhs_batch_idx =
        BuildOperandFlatBatchIndex(lhs_batch_dims, output_batch_dims, output_batch_indices);
    int64_t rhs_batch_idx =
        BuildOperandFlatBatchIndex(rhs_batch_dims, output_batch_dims, output_batch_indices);

    auto lhs_page =
        ExtractBatchPage(lhs_info, lhs_dims, lhs_batch_dims, lhs_batch_idx, "lhs", ctx, op_registry, span);
    auto rhs_page =
        ExtractBatchPage(rhs_info, rhs_dims, rhs_batch_dims, rhs_batch_idx, "rhs", ctx, op_registry, span);
    auto matmul = op_registry.Create("tile.matmul", {lhs_page.var, rhs_page.var}, span);
    auto matmul_type = As<TileType>(matmul->GetType());
    bool needs_cast = matmul_type && matmul_type->dtype_ != orig_result_type->dtype_;
    if (!needs_cast) {
      out.stmts.insert(out.stmts.end(), lhs_page.stmts.begin(), lhs_page.stmts.end());
      out.stmts.insert(out.stmts.end(), rhs_page.stmts.begin(), rhs_page.stmts.end());
      auto matmul_var = std::make_shared<Var>(assign->var_->name_hint_, matmul->GetType(), span);
      out.stmts.push_back(std::make_shared<AssignStmt>(matmul_var, matmul, span));
      // The logical batch_matmul result remains authoritative.  Even the one-page
      // fast path must not return a raw tile.matmul type that widened or otherwise
      // changed the ND result's effective valid region during flattening.
      if (!AreExprVectorsEqual(GetValidShape(As<TileType>(matmul_var->GetType())), expected_flat_valid)) {
        INTERNAL_CHECK_SPAN(expected_flat_valid.size() == 2, span)
            << "FlattenTileNdTo2D: flattened batch_matmul valid_shape must be 2D";
        auto set_valid = op_registry.Create(
            "tile.set_validshape", {matmul_var, expected_flat_valid[0], expected_flat_valid[1]}, span);
        auto set_valid_var =
            std::make_shared<Var>(assign->var_->name_hint_, set_valid->GetType(), assign->var_->span_);
        out.stmts.push_back(std::make_shared<AssignStmt>(set_valid_var, set_valid, span));
        out.output_var = set_valid_var;
      } else {
        out.output_var = matmul_var;
      }
      return out;
    }
    // Discard the speculative pages and matmul (no out.stmts modification yet);
    // fall through to the general path which inserts the required tile.cast.
  }

  // Allocate output tile (non-fused path only).
  VarPtr out_var;
  if (!direct_store.detected) {
    auto out_shape =
        std::make_shared<MakeTuple>(Make2DShapeExprs(batch_count * lhs_rows, rhs_cols, span), span);
    // Per-batch matmul results are moved to Vec via tile.move(target_memory=Vec)
    // before being assembled into this tile, so allocate the staging tile in Vec
    // up-front. This keeps the printed/parsed IR consistent (parser otherwise
    // backfills target_memory=Vec from the assemble consumer chain).
    std::vector<std::pair<std::string, std::any>> create_kw = {
        {"dtype", orig_result_type->dtype_},
        {"target_memory", MemorySpace::Vec},
    };
    auto create_out = op_registry.Create("tile.create", {out_shape}, create_kw, span);
    out_var = std::make_shared<Var>(assign->var_->name_hint_, create_out->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(out_var, create_out, span));
  }

  // Prepare direct-store state.
  ExprPtr current_store_tensor;
  MakeTuplePtr direct_store_offsets;
  std::vector<ExprPtr> direct_store_shape;
  if (direct_store.detected) {
    current_store_tensor = Substitute(direct_store.store_call->args_[2], ctx.var_map);
    direct_store_offsets = As<MakeTuple>(Substitute(direct_store.store_call->args_[1], ctx.var_map));
    auto store_tensor_type = AsTensorTypeLike(current_store_tensor->GetType());
    CHECK(store_tensor_type)
        << "FlattenTileNdTo2D: tile.batch_matmul direct store target must be tensor-like";
    CHECK(direct_store_offsets) << "FlattenTileNdTo2D: tile.store offsets must be a MakeTuple";
    CHECK(direct_store_offsets->elements_.size() == output_batch_dims.size() + 2)
        << "FlattenTileNdTo2D: tile.store offsets rank must match batch_matmul result rank";
    if (store_tensor_type->shape_.size() > 2) {
      // Build the original tensor-rank partition prefix.  The trailing two
      // extents come from each batch result's *effective* valid shape below,
      // never from the physical M/N tile capacity.
      const size_t tensor_rank = store_tensor_type->shape_.size();
      const size_t tile_rank = 2;  // matmul result is always 2D
      direct_store_shape.reserve(tensor_rank);
      for (size_t i = tile_rank; i < tensor_rank; ++i) {
        direct_store_shape.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, span));
      }
    }
  }

  // Unroll batch dimensions.
  for (int64_t i = 0; i < batch_count; ++i) {
    auto output_batch_indices = BuildBatchIndices(i, output_batch_dims);
    int64_t lhs_batch_idx =
        BuildOperandFlatBatchIndex(lhs_batch_dims, output_batch_dims, output_batch_indices);
    int64_t rhs_batch_idx =
        BuildOperandFlatBatchIndex(rhs_batch_dims, output_batch_dims, output_batch_indices);

    // Extract 2D pages.
    auto lhs_page =
        ExtractBatchPage(lhs_info, lhs_dims, lhs_batch_dims, lhs_batch_idx, "lhs", ctx, op_registry, span);
    auto rhs_page =
        ExtractBatchPage(rhs_info, rhs_dims, rhs_batch_dims, rhs_batch_idx, "rhs", ctx, op_registry, span);
    out.stmts.insert(out.stmts.end(), lhs_page.stmts.begin(), lhs_page.stmts.end());
    out.stmts.insert(out.stmts.end(), rhs_page.stmts.begin(), rhs_page.stmts.end());

    // Emit tile.matmul.
    auto matmul = op_registry.Create("tile.matmul", {lhs_page.var, rhs_page.var}, span);
    auto matmul_var = std::make_shared<Var>("matmul_" + std::to_string(i), matmul->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(matmul_var, matmul, span));

    // Move matmul result from Acc to Vec, then cast dtype if needed.
    // The explicit tile.move is always required for the non-fused (assemble) path so
    // that ExpandMixedKernel sees a clear AIC→AIV boundary. For the fused (direct
    // store) path, the tile.store codegen handles the Acc→DDR transfer directly.
    ExprPtr batch_result = matmul_var;
    auto batch_result_type = As<TileType>(matmul_var->GetType());
    bool needs_cast = batch_result_type && batch_result_type->dtype_ != orig_result_type->dtype_;
    if (!direct_store.detected || needs_cast) {
      std::vector<std::pair<std::string, std::any>> move_kw = {
          {"target_memory", MemorySpace::Vec},
      };
      auto move = op_registry.Create("tile.move", {matmul_var}, move_kw, span);
      auto move_var = std::make_shared<Var>("matmul_vec_" + std::to_string(i), move->GetType(), span);
      out.stmts.push_back(std::make_shared<AssignStmt>(move_var, move, span));
      batch_result = move_var;
    }
    if (needs_cast) {
      std::vector<std::pair<std::string, std::any>> cast_kw = {
          {"target_type", orig_result_type->dtype_},
          {"mode", 2},
      };
      auto cast = op_registry.Create("tile.cast", {batch_result}, cast_kw, span);
      auto cast_var = std::make_shared<Var>("matmul_cast_" + std::to_string(i), cast->GetType(), span);
      out.stmts.push_back(std::make_shared<AssignStmt>(cast_var, cast, span));
      batch_result = cast_var;
    }

    if (direct_store.detected) {
      // Fused path: emit per-batch tile.store.
      // Keep the original tensor-rank offsets — codegen reconstructs the
      // corresponding partition_view from that window description.
      auto store_offset_elems = BuildBatchAdjustedOffsets(
          direct_store_offsets->elements_, output_batch_indices, output_batch_dims.size(), span);
      auto store_offset = std::make_shared<MakeTuple>(store_offset_elems, span);

      std::vector<ExprPtr> store_args = {batch_result, store_offset, current_store_tensor};
      if (!direct_store_shape.empty()) {
        std::vector<ExprPtr> partition_valid = direct_store_shape;
        auto result_tile_type = As<TileType>(batch_result->GetType());
        INTERNAL_CHECK_SPAN(result_tile_type, span)
            << "FlattenTileNdTo2D: batch_matmul direct-store value must be TileType";
        const auto result_valid = GetValidShape(result_tile_type);
        INTERNAL_CHECK_SPAN(result_valid.size() == 2, span)
            << "FlattenTileNdTo2D: batch_matmul direct-store value must have rank-2 valid_shape";
        partition_valid.insert(partition_valid.end(), result_valid.begin(), result_valid.end());
        store_args.push_back(std::make_shared<MakeTuple>(partition_valid, span));
      }
      auto batch_store = op_registry.Create("tile.store", store_args, span);
      auto batch_store_var =
          std::make_shared<Var>(direct_store.store_assign->var_->name_hint_ + "_" + std::to_string(i),
                                batch_store->GetType(), span);
      out.stmts.push_back(std::make_shared<AssignStmt>(batch_store_var, batch_store, span));
      current_store_tensor = batch_store_var;
    } else {
      // Non-fused path: assemble into output tile.
      auto out_offset = MakeShapeTupleFromInts({i * lhs_rows, 0}, span);
      auto assemble = op_registry.Create("tile.assemble", {out_var, batch_result, out_offset}, span);
      out_var = std::make_shared<Var>(out_var->name_hint_, assemble->GetType(), span);
      out.stmts.push_back(std::make_shared<AssignStmt>(out_var, assemble, span));
    }
  }

  if (direct_store.detected) {
    auto final_store_var = As<Var>(current_store_tensor);
    CHECK(final_store_var) << "FlattenTileNdTo2D: expected final direct store result to be a Var";
    out.fused_store = true;
    out.store_result_var = final_store_var;
    out.store_orig_var = direct_store.store_assign->var_;
  } else {
    // The staging tile is intentionally physical-capacity-sized while pages are
    // assembled.  Narrow it back to the authoritative flattened logical region
    // before exposing it to any consumer.
    if (!AreExprVectorsEqual(GetValidShape(As<TileType>(out_var->GetType())), expected_flat_valid)) {
      INTERNAL_CHECK_SPAN(expected_flat_valid.size() == 2, span)
          << "FlattenTileNdTo2D: flattened batch_matmul valid_shape must be 2D";
      auto set_valid = op_registry.Create("tile.set_validshape",
                                          {out_var, expected_flat_valid[0], expected_flat_valid[1]}, span);
      auto set_valid_var = std::make_shared<Var>(out_var->name_hint_, set_valid->GetType(), out_var->span_);
      out.stmts.push_back(std::make_shared<AssignStmt>(set_valid_var, set_valid, span));
      out_var = set_valid_var;
    }
    out.output_var = out_var;
  }

  return out;
}

// ============================================================================
// Batch matmul_acc lowering
// ============================================================================
//
// tile.batch_matmul_acc semantics:
//   acc[..., M, N] += lhs[..., M, K] @ rhs[..., K, N]   (with batch broadcast)
//
// The 2D backend only supports tile.matmul_acc on rank-2 tiles. After earlier
// flattening (which has already turned the original ND acc into its flat 2D form
// [batch_count*M, N]), this lowering unrolls the batch dim into a sequence of
// per-batch tile.matmul_acc calls writing into the corresponding row-band of acc.
//
// Direct-store fusion is intentionally not applied here — the canonical use is
// "y_acc = matmul; for k: y_acc = matmul_acc(y_acc, ...); store(y_acc)" where
// the store consumes the loop-carried accumulator after the loop, not the
// individual matmul_acc results. The acc operand itself is the in-place target.
//

/// Result of lowering a tile.batch_matmul_acc operation.
struct BatchMatmulAccResult {
  std::vector<StmtPtr> stmts;  ///< Emitted statements
  VarPtr output_var;           ///< Updated acc tile (always 2D after flatten)
};

/// Lower tile.batch_matmul_acc into unrolled 2D tile.matmul_acc calls.
BatchMatmulAccResult LowerBatchMatmulAcc(const AssignStmtPtr& assign, const CallPtr& call,
                                         const std::vector<StmtPtr>& stmts, const FlattenContext& ctx,
                                         const OpRegistry& op_registry, const Span& span) {
  BatchMatmulAccResult out;
  auto def_map = BuildAssignDefMap(stmts);
  auto original_result_type = As<TileType>(call->GetType());
  INTERNAL_CHECK_SPAN(original_result_type, span)
      << "FlattenTileNdTo2D: tile.batch_matmul_acc result must be TileType";
  const std::vector<ExprPtr> expected_flat_valid =
      ComputeMergedValidShape(GetValidShape(original_result_type), original_result_type->shape_, span);
  INTERNAL_CHECK_SPAN(expected_flat_valid.size() == 2, span)
      << "FlattenTileNdTo2D: flattened batch_matmul_acc valid_shape must be 2D";

  // The acc operand has already been flattened (or is naturally 2D) by earlier
  // statement processing; substitute via var_map to pick up any rewrites.
  // Accept both Var and IterArg (loop-carried accumulator) — both are Var-like
  // in the IR and downstream code only needs name_hint_ + a stable Expr.
  auto acc_operand = Substitute(call->args_[0], ctx.var_map);
  auto acc_var = AsVarLike(acc_operand);
  CHECK(acc_var) << "FlattenTileNdTo2D: tile.batch_matmul_acc acc must be a Var/IterArg after "
                    "substitution, got "
                 << acc_operand->TypeName();
  auto acc_type = As<TileType>(acc_operand->GetType());
  CHECK(acc_type) << "FlattenTileNdTo2D: tile.batch_matmul_acc acc must be TileType";
  CHECK(acc_type->shape_.size() == 2)
      << "FlattenTileNdTo2D: tile.batch_matmul_acc expects acc to be 2D after flatten, got rank "
      << acc_type->shape_.size();

  // Normalize lhs/rhs operands (peel safe reshape, flag tile.transpose_view).
  auto lhs_info = NormalizeBatchMatmulOperand(call->args_[1], "lhs", def_map, ctx);
  auto rhs_info = NormalizeBatchMatmulOperand(call->args_[2], "rhs", def_map, ctx);
  // Route each operand to whole-slice vs per-batch independently: keep whole only
  // when the operands' whole tiles fit Mat together (capacity) AND this operand's
  // whole load collapses contiguously. A non-contiguous whole load (multi-batch +
  // partial matrix-row dim) is re-emitted per batch — each per-batch page is a
  // [1, X, Y] window that collapses cleanly — instead of erroring in codegen.
  const bool capacity_fits = BatchOperandsWholeFit(lhs_info.original_type, rhs_info.original_type);
  lhs_info.whole_fits = KeepOperandWhole(capacity_fits, lhs_info.base_load);
  rhs_info.whole_fits = KeepOperandWhole(capacity_fits, rhs_info.base_load);

  // Extract original (pre-flatten) static dimensions for batch + matrix axes.
  auto lhs_dims = ToStaticDims(lhs_info.original_type->shape_, "tile.batch_matmul_acc lhs");
  auto rhs_dims = ToStaticDims(rhs_info.original_type->shape_, "tile.batch_matmul_acc rhs");
  CHECK(lhs_dims.size() >= 2) << "FlattenTileNdTo2D: tile.batch_matmul_acc lhs must be at least 2D";
  CHECK(rhs_dims.size() >= 2) << "FlattenTileNdTo2D: tile.batch_matmul_acc rhs must be at least 2D";

  // Compute broadcast batch dims (must equal acc's batch by op contract).
  std::vector<ExprPtr> lhs_batch_exprs(lhs_info.original_type->shape_.begin(),
                                       lhs_info.original_type->shape_.end() - 2);
  std::vector<ExprPtr> rhs_batch_exprs(rhs_info.original_type->shape_.begin(),
                                       rhs_info.original_type->shape_.end() - 2);
  auto broadcast_result = BroadcastShapes(lhs_batch_exprs, rhs_batch_exprs);
  CHECK(broadcast_result.success)
      << "FlattenTileNdTo2D: tile.batch_matmul_acc batch dimensions must broadcast";

  auto output_batch_dims = ToStaticDims(broadcast_result.shape, "tile.batch_matmul_acc output batch");
  int64_t batch_count = MultiplyStaticDims(output_batch_dims, "tile.batch_matmul_acc output batch size");

  std::vector<int64_t> lhs_batch_dims(lhs_dims.begin(), lhs_dims.end() - 2);
  std::vector<int64_t> rhs_batch_dims(rhs_dims.begin(), rhs_dims.end() - 2);

  // Transposed operands arrive in matmul orientation via tile.transpose_view, so
  // the trailing two dims are used directly (no swap).
  int64_t lhs_rows = lhs_dims[lhs_dims.size() - 2];
  int64_t lhs_cols = lhs_dims.back();
  int64_t rhs_rows = rhs_dims[rhs_dims.size() - 2];
  int64_t rhs_cols = rhs_dims.back();

  // K-match is validated user-facing at op construction (DeduceTileBatchMatMulAccType);
  // a mismatch here would be a compiler bug in an earlier pass.
  INTERNAL_CHECK_SPAN(lhs_cols == rhs_rows, span)
      << "Internal error: tile.batch_matmul_acc inner dimensions must match, got " << lhs_cols << " and "
      << rhs_rows;

  // Sanity check on flat acc shape: should be [batch_count*M, N].
  auto acc_rows_const = As<ConstInt>(acc_type->shape_[0]);
  auto acc_cols_const = As<ConstInt>(acc_type->shape_[1]);
  CHECK(acc_rows_const && acc_cols_const)
      << "FlattenTileNdTo2D: tile.batch_matmul_acc expects static acc dims after flatten";
  CHECK(acc_rows_const->value_ == batch_count * lhs_rows)
      << "FlattenTileNdTo2D: tile.batch_matmul_acc acc rows " << acc_rows_const->value_ << " != batch_count("
      << batch_count << ") * M(" << lhs_rows << ")";
  CHECK(acc_cols_const->value_ == rhs_cols) << "FlattenTileNdTo2D: tile.batch_matmul_acc acc cols "
                                            << acc_cols_const->value_ << " != N(" << rhs_cols << ")";

  // Memory-space concerns (Vec/Acc round-trips on the acc operand, retargetable
  // producer promotion of the loop-carried tile.create, and matching TileView
  // layout refresh) belong to InferTileMemorySpace (pass 17, runs immediately
  // after this pass). See:
  //   * DemandCollector — propagates the matmul_acc Acc input_constraint back
  //     through inherit-input ops.
  //   * TileMemorySpaceAnalyzer::VisitStmt_(ForStmtPtr) — explicitly
  //     back-propagates the yield's memory space to the iter_arg AND its
  //     initValue (covering the dummy tile.create dummy-acc-init pattern in
  //     issue #1235).
  //   * TileMemorySpaceMutator::VisitStmt_(AssignStmtPtr) — rewrites the
  //     retargetable producer's target_memory kwarg and refreshes its
  //     TileView via GetImplicitTileView.
  // FlattenTileNdTo2D's job here is purely the shape lowering: pass the acc
  // operand through and emit 2D tile.matmul_acc (and per-batch
  // tile.slice/tile.assemble in the general path). Any required tile.move
  // calls are inserted by InferTileMemorySpace's MoveCollector. This avoids
  // the cross-core Vec→Acc move that previously failed verification in mixed
  // CUBE/VECTOR kernels (issue #1235).
  VarPtr current_acc = acc_var;

  // Fast path: batch_count == 1. The acc is already [M, N] and per-batch slicing
  // would be identity. Emit a single tile.matmul_acc directly. This also avoids
  // tile.slice/tile.assemble in Acc memory which is the standard codegen path
  // covered by the existing 2D tile.matmul_acc handling.
  if (batch_count == 1) {
    auto output_batch_indices = BuildBatchIndices(0, output_batch_dims);
    int64_t lhs_batch_idx =
        BuildOperandFlatBatchIndex(lhs_batch_dims, output_batch_dims, output_batch_indices);
    int64_t rhs_batch_idx =
        BuildOperandFlatBatchIndex(rhs_batch_dims, output_batch_dims, output_batch_indices);

    auto lhs_page =
        ExtractBatchPage(lhs_info, lhs_dims, lhs_batch_dims, lhs_batch_idx, "lhs", ctx, op_registry, span);
    auto rhs_page =
        ExtractBatchPage(rhs_info, rhs_dims, rhs_batch_dims, rhs_batch_idx, "rhs", ctx, op_registry, span);
    out.stmts.insert(out.stmts.end(), lhs_page.stmts.begin(), lhs_page.stmts.end());
    out.stmts.insert(out.stmts.end(), rhs_page.stmts.begin(), rhs_page.stmts.end());

    auto matmul_acc = op_registry.Create("tile.matmul_acc", {current_acc, lhs_page.var, rhs_page.var}, span);
    auto new_acc = std::make_shared<Var>(current_acc->name_hint_, matmul_acc->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(new_acc, matmul_acc, span));
    if (!AreExprVectorsEqual(GetValidShape(As<TileType>(new_acc->GetType())), expected_flat_valid)) {
      auto set_valid = op_registry.Create("tile.set_validshape",
                                          {new_acc, expected_flat_valid[0], expected_flat_valid[1]}, span);
      auto set_valid_var =
          std::make_shared<Var>(assign->var_->name_hint_, set_valid->GetType(), assign->var_->span_);
      out.stmts.push_back(std::make_shared<AssignStmt>(set_valid_var, set_valid, span));
      new_acc = set_valid_var;
    }
    out.output_var = new_acc;
    return out;
  }

  // General path: unroll batch dims using slice + tile.matmul_acc + assemble on acc.
  for (int64_t i = 0; i < batch_count; ++i) {
    auto output_batch_indices = BuildBatchIndices(i, output_batch_dims);
    int64_t lhs_batch_idx =
        BuildOperandFlatBatchIndex(lhs_batch_dims, output_batch_dims, output_batch_indices);
    int64_t rhs_batch_idx =
        BuildOperandFlatBatchIndex(rhs_batch_dims, output_batch_dims, output_batch_indices);

    auto lhs_page =
        ExtractBatchPage(lhs_info, lhs_dims, lhs_batch_dims, lhs_batch_idx, "lhs", ctx, op_registry, span);
    auto rhs_page =
        ExtractBatchPage(rhs_info, rhs_dims, rhs_batch_dims, rhs_batch_idx, "rhs", ctx, op_registry, span);
    out.stmts.insert(out.stmts.end(), lhs_page.stmts.begin(), lhs_page.stmts.end());
    out.stmts.insert(out.stmts.end(), rhs_page.stmts.begin(), rhs_page.stmts.end());

    auto suffix = std::to_string(i);
    auto acc_offset = MakeShapeTupleFromInts({i * lhs_rows, 0}, span);
    auto acc_shape = MakeShapeTupleFromInts({lhs_rows, rhs_cols}, span);
    auto acc_slice = op_registry.Create("tile.slice", {current_acc, acc_shape, acc_offset}, span);
    auto acc_page_var = std::make_shared<Var>("acc_page_" + suffix, acc_slice->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(acc_page_var, acc_slice, span));

    auto matmul_acc = op_registry.Create("tile.matmul_acc", {acc_page_var, lhs_page.var, rhs_page.var}, span);
    auto matmul_var = std::make_shared<Var>("matmul_acc_" + suffix, matmul_acc->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(matmul_var, matmul_acc, span));

    auto assemble = op_registry.Create("tile.assemble", {current_acc, matmul_var, acc_offset}, span);
    current_acc = std::make_shared<Var>(current_acc->name_hint_, assemble->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(current_acc, assemble, span));
  }

  if (!AreExprVectorsEqual(GetValidShape(As<TileType>(current_acc->GetType())), expected_flat_valid)) {
    auto set_valid = op_registry.Create("tile.set_validshape",
                                        {current_acc, expected_flat_valid[0], expected_flat_valid[1]}, span);
    auto set_valid_var =
        std::make_shared<Var>(assign->var_->name_hint_, set_valid->GetType(), assign->var_->span_);
    out.stmts.push_back(std::make_shared<AssignStmt>(set_valid_var, set_valid, span));
    current_acc = set_valid_var;
  }

  out.output_var = current_acc;
  return out;
}

// ============================================================================
// Standalone N-D transpose lowering
// ============================================================================
//
// A standalone >2D tile.transpose (not consumed by a tile.batch_matmul) has no
// dedicated 2D hardware op: the backend pto.ttrans is strictly 2D. The frontend
// auto-allocates the transpose scratch (tmp) at INPUT rank (see
// python/pypto/ir/op/tile_ops.py), so a 3D input yields a 3D tmp. The generic
// re-create path would substitute a flattened 2D tmp while leaving the input
// rank at 3, tripping the input-rank == tmp-rank CHECK in
// DeduceTileTransposeType.
//
// This lowering mirrors LowerBatchMatmul's non-fused path: it handles only the
// last-two-axes swap (axes {ndim-2, ndim-1}) with leading batch dims. The
// already-flattened 2D input [batch_count*A, B] is sliced per batch into an
// [A, B] page, transposed via a genuine 2D tile.transpose into [B, A], and
// assembled into a flat [batch_count*B, A] output tile. Every emitted transpose
// is 2D, so no codegen change is needed. Transposes that move a batch axis
// cannot be expressed as a per-batch 2D ttrans and are rejected with a clear
// user-facing error.

/// Result of lowering a standalone N-D tile.transpose operation.
struct NdTransposeResult {
  std::vector<StmtPtr> stmts;  ///< Emitted statements
  VarPtr output_var;           ///< Flat 2D result tile [batch_count*B, A]
};

/// Lower a standalone >2D last-two-axes tile.transpose into per-batch 2D
/// tile.transpose calls assembled into a flat 2D output.
NdTransposeResult LowerNdTranspose(const AssignStmtPtr& assign, const CallPtr& call,
                                   const FlattenContext& ctx, const OpRegistry& op_registry,
                                   const Span& span) {
  NdTransposeResult out;

  // Read the ORIGINAL (pre-substitution) input type: its >2D dims are intact.
  auto input_type = As<TileType>(call->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(input_type, span)
      << "Internal error: tile.transpose input must be TileType in FlattenTileNdTo2D";
  auto input_dims = ToStaticDims(input_type->shape_, "tile.transpose");
  size_t ndim = input_dims.size();
  INTERNAL_CHECK_SPAN(ndim > 2, span)
      << "Internal error: LowerNdTranspose called on a tile of rank " << ndim << " (expected >2)";
  auto original_result_type = As<TileType>(call->GetType());
  INTERNAL_CHECK_SPAN(original_result_type, span)
      << "Internal error: tile.transpose result must be TileType in FlattenTileNdTo2D";
  const std::vector<ExprPtr> expected_flat_valid =
      ComputeMergedValidShape(GetValidShape(original_result_type), original_result_type->shape_, span);

  // Normalize axes and require a last-two-axes swap.
  auto axis1_const = As<ConstInt>(call->args_[1]);
  auto axis2_const = As<ConstInt>(call->args_[2]);
  INTERNAL_CHECK_SPAN(axis1_const && axis2_const, span)
      << "Internal error: tile.transpose axes must be ConstInt in FlattenTileNdTo2D";
  int64_t axis1 = NormalizeAxisIndex(axis1_const->value_, ndim, "tile.transpose");
  int64_t axis2 = NormalizeAxisIndex(axis2_const->value_, ndim, "tile.transpose");
  CHECK_SPAN(IsTrailingMatrixAxisSwap(axis1, axis2, ndim), span)
      << "FlattenTileNdTo2D: only last-two-axes tile.transpose on >2D tiles is supported; "
      << "transpose involving a batch axis (axes " << axis1 << ", " << axis2 << " of rank " << ndim
      << ") is not yet lowered. Reshape the tile so the transposed axes are the last two.";

  std::vector<int64_t> batch_dims(input_dims.begin(), input_dims.end() - 2);
  int64_t a = input_dims[ndim - 2];
  int64_t b = input_dims[ndim - 1];
  int64_t batch_count = MultiplyStaticDims(batch_dims, "tile.transpose batch size");

  // Resolve the input operand. After var_map substitution it is usually the
  // already-flattened 2D tile [batch_count*A, B]. But a producer that this pass
  // leaves at rank>2 (e.g. a tile.reshape to a 3D shape, which the generic path
  // re-creates without flattening) yields a still-ND operand. In that case emit
  // a tile.reshape down to the merged 2D [batch_count*A, B] so the per-batch
  // tile.slice below has a 2D parent.
  auto operand = Substitute(call->args_[0], ctx.var_map);
  auto operand_type = As<TileType>(operand->GetType());
  INTERNAL_CHECK_SPAN(operand_type, span)
      << "Internal error: tile.transpose input must be TileType in FlattenTileNdTo2D";
  MemorySpace target_mem =
      operand_type->memory_space_.has_value() ? *operand_type->memory_space_ : MemorySpace::Vec;

  if (operand_type->shape_.size() > 2) {
    auto [merged, last] = ComputeMergedShape(operand_type->shape_, "tile.transpose input");
    INTERNAL_CHECK_SPAN(merged == batch_count * a && last == b, span)
        << "Internal error: tile.transpose flattened input shape [" << merged << ", " << last
        << "] does not match expected [" << (batch_count * a) << ", " << b << "]";
    auto reshape_shape = std::make_shared<MakeTuple>(Make2DShapeExprs(merged, last, span), span);
    auto reshape = op_registry.Create("tile.reshape", {operand, reshape_shape}, span);
    auto reshape_var = std::make_shared<Var>("trans_in_2d", reshape->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(reshape_var, reshape, span));
    operand = reshape_var;
    operand_type = As<TileType>(operand->GetType());
  }

  // Pre-create the flat output tile [batch_count*B, A].
  auto out_shape = std::make_shared<MakeTuple>(Make2DShapeExprs(batch_count * b, a, span), span);
  std::vector<std::pair<std::string, std::any>> create_kw = {
      {"dtype", operand_type->dtype_},
      {"target_memory", target_mem},
  };
  auto create_out = op_registry.Create("tile.create", {out_shape}, create_kw, span);
  VarPtr out_var = std::make_shared<Var>(assign->var_->name_hint_, create_out->GetType(), span);
  out.stmts.push_back(std::make_shared<AssignStmt>(out_var, create_out, span));

  // Pre-create one flat scratch pool [batch_count*A, B] sliced per batch.
  // pto.ttrans requires a scratch operand whose type matches the source page's;
  // its codegen reuses the SOURCE's type for BOTH ins operands
  // (MakeTileTransposeCodegenPTO emits "src_type, src_type"). The source page is
  // a tile.slice -> pto.subview, which produces a NEW SSA value with STATIC valid
  // [A, B]. The scratch must be the same kind of value, so it is sliced from this
  // pool per batch (a partial tile.slice -> pto.subview), NOT
  // created+set_validshape: set_validshape mutates the alloc in place (dynamic
  // valid ?x?) without renaming it, so ttrans would see the same SSA value typed
  // both dynamic (at its def/set_validshape) and static (at the ttrans use) ->
  // ptoas type conflict. The pool lives across the loop; being a single
  // allocation it is cheap relative to per-batch scratch churn.
  auto tmp_pool_shape = std::make_shared<MakeTuple>(Make2DShapeExprs(batch_count * a, b, span), span);
  std::vector<std::pair<std::string, std::any>> tmp_pool_kw = {
      {"dtype", operand_type->dtype_},
      {"target_memory", target_mem},
  };
  auto tmp_pool_create = op_registry.Create("tile.create", {tmp_pool_shape}, tmp_pool_kw, span);
  VarPtr tmp_pool_var = std::make_shared<Var>("trans_tmp_pool", tmp_pool_create->GetType(), span);
  out.stmts.push_back(std::make_shared<AssignStmt>(tmp_pool_var, tmp_pool_create, span));

  auto axis0_expr = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto axis1_expr = std::make_shared<ConstInt>(1, DataType::INDEX, span);

  for (int64_t i = 0; i < batch_count; ++i) {
    auto suffix = std::to_string(i);

    // Extract the i-th dense 2D input page [A, B] from the flat operand.
    auto in_offset = MakeShapeTupleFromInts({i * a, 0}, span);
    auto in_shape = MakeShapeTupleFromInts({a, b}, span);
    auto slice = op_registry.Create("tile.slice", {operand, in_shape, in_offset}, span);
    ExprPtr src_page = std::make_shared<Var>("trans_page_" + suffix, slice->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(As<Var>(src_page), slice, span));

    // Slice the i-th 2D scratch page [A, B] from the flat tmp pool (subview with
    // the same effective validity as the source page). A partial ND input can
    // produce a zero/short tail page; leaving scratch fully valid would give
    // pto.ttrans two incompatible tile_buf types.
    auto tmp_offset = MakeShapeTupleFromInts({i * a, 0}, span);
    auto tmp_shape = MakeShapeTupleFromInts({a, b}, span);
    auto src_page_type = As<TileType>(src_page->GetType());
    INTERNAL_CHECK_SPAN(src_page_type, span) << "FlattenTileNdTo2D: transpose source page must be TileType";
    auto tmp_valid = std::make_shared<MakeTuple>(GetValidShape(src_page_type), span);
    auto tmp_slice = op_registry.Create("tile.slice", {tmp_pool_var, tmp_shape, tmp_offset, tmp_valid}, span);
    ExprPtr scratch_page = std::make_shared<Var>("trans_tmp_" + suffix, tmp_slice->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(As<Var>(scratch_page), tmp_slice, span));

    // Transpose the page [A, B] -> [B, A]. Ranks match, CHECK passes.
    auto transpose =
        op_registry.Create("tile.transpose", {src_page, axis0_expr, axis1_expr, scratch_page}, span);
    auto transpose_var = std::make_shared<Var>("trans_" + suffix, transpose->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(transpose_var, transpose, span));

    // Assemble the [B, A] result into the flat output at row offset i*B.
    auto out_offset = MakeShapeTupleFromInts({i * b, 0}, span);
    auto assemble = op_registry.Create("tile.assemble", {out_var, transpose_var, out_offset}, span);
    out_var = std::make_shared<Var>(out_var->name_hint_, assemble->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(out_var, assemble, span));
  }

  // The page assembly uses a capacity-sized staging tile.  Restore the exact
  // validity inferred by the logical ND transpose before publishing the result.
  if (!AreExprVectorsEqual(GetValidShape(As<TileType>(out_var->GetType())), expected_flat_valid)) {
    INTERNAL_CHECK_SPAN(expected_flat_valid.size() == 2, span)
        << "FlattenTileNdTo2D: flattened transpose valid_shape must be 2D";
    auto set_valid = op_registry.Create("tile.set_validshape",
                                        {out_var, expected_flat_valid[0], expected_flat_valid[1]}, span);
    auto set_valid_var = std::make_shared<Var>(out_var->name_hint_, set_valid->GetType(), out_var->span_);
    out.stmts.push_back(std::make_shared<AssignStmt>(set_valid_var, set_valid, span));
    out_var = set_valid_var;
  }

  out.output_var = out_var;
  return out;
}

/**
 * @brief Recursively transform statements, flattening >2D tile ops to 2D.
 */
std::vector<StmtPtr> TransformBody(const std::vector<StmtPtr>& stmts, FlattenContext& ctx,
                                   const OpRegistry& op_registry, const Span& span) {
  std::vector<StmtPtr> result;

  // Pre-scan: identify operand chains (tile.load -> tile.transpose_view/reshape)
  // consumed EXCLUSIVELY by tile.batch_matmul. Used for two things: (a) the
  // tile.transpose / tile.reshape rewrite-loop skips below (peeled by the lowering),
  // and (b) the !fit capacity drop — when a matmul's whole operands do not fit Mat,
  // ExtractBatchPage loads them per batch and the dead whole chain is dropped.
  //
  // Safety: we count ALL Var references across every statement type (Return, Yield,
  // If conditions, For/While bounds, etc.), not just Call arguments. A Var used
  // anywhere outside a tile.batch_matmul Call prevents it from being treated as a
  // batch_matmul-only chain.
  std::unordered_set<const Var*> batch_matmul_only_vars;
  std::vector<const Var*> batch_matmul_only_vars_in_order;
  auto record_batch_matmul_only_var = [&](const Var* var) {
    if (batch_matmul_only_vars.insert(var).second) batch_matmul_only_vars_in_order.push_back(var);
  };
  // Operand-chain vars (whole load + transpose_view) feeding EXCLUSIVELY !fit
  // batch_matmuls — their whole tile would overflow Mat, so ExtractBatchPage loads
  // them per batch and the dead whole chain is dropped during rewriting below.
  std::unordered_set<const Var*> not_fit_drop_vars;
  {
    // Count every use and every direct batch-matmul operand use recursively.
    // Nested loops/branches are part of the same lexical SSA scope: an ND
    // transpose_view used only by a nested batch_matmul is still eligible for
    // the column-batched representation, while any non-matmul use disqualifies
    // it. Keeping both counts makes that distinction exact.
    std::unordered_map<const Var*, int> total_counts;
    std::unordered_map<const Var*, int> batch_matmul_operand_uses;
    std::vector<const Var*> batch_matmul_operands;  // deterministic first-seen order
    std::function<void(const ExprPtr&)> CountTotalExprRefs = [&](const ExprPtr& expr) {
      if (!expr) return;
      if (auto v = As<Var>(expr)) {
        total_counts[v.get()]++;
        return;
      }
      if (auto tup = As<MakeTuple>(expr)) {
        for (const auto& e : tup->elements_) CountTotalExprRefs(e);
        return;
      }
      if (auto call = As<Call>(expr)) {
        for (const auto& a : call->args_) CountTotalExprRefs(a);
        return;
      }
    };
    std::function<void(const StmtPtr&)> CountTotalStmtRefs = [&](const StmtPtr& s) {
      if (!s) return;
      if (auto seq = As<SeqStmts>(s)) {
        for (const auto& inner : seq->stmts_) CountTotalStmtRefs(inner);
      } else if (auto scope = As<ScopeStmt>(s)) {
        CountTotalStmtRefs(scope->body_);
      } else if (auto if_stmt = As<IfStmt>(s)) {
        CountTotalExprRefs(if_stmt->condition_);
        CountTotalStmtRefs(if_stmt->then_body_);
        if (if_stmt->else_body_.has_value()) CountTotalStmtRefs(*if_stmt->else_body_);
      } else if (auto for_stmt = As<ForStmt>(s)) {
        CountTotalExprRefs(for_stmt->start_);
        CountTotalExprRefs(for_stmt->stop_);
        CountTotalExprRefs(for_stmt->step_);
        for (const auto& ia : for_stmt->iter_args_) CountTotalExprRefs(ia->initValue_);
        CountTotalStmtRefs(for_stmt->body_);
      } else if (auto while_stmt = As<WhileStmt>(s)) {
        CountTotalExprRefs(while_stmt->condition_);
        for (const auto& ia : while_stmt->iter_args_) CountTotalExprRefs(ia->initValue_);
        CountTotalStmtRefs(while_stmt->body_);
      } else if (auto a = As<AssignStmt>(s)) {
        if (auto c = As<Call>(a->value_)) {
          const bool is_batch_mm = IsOp(c, "tile.batch_matmul");
          const bool is_batch_mm_acc = IsOp(c, "tile.batch_matmul_acc");
          for (size_t arg_i = 0; arg_i < c->args_.size(); ++arg_i) {
            const bool eligible = is_batch_mm || (is_batch_mm_acc && (arg_i == 1 || arg_i == 2));
            if (!eligible) continue;
            if (auto v = As<Var>(c->args_[arg_i])) {
              batch_matmul_operand_uses[v.get()]++;
              batch_matmul_operands.push_back(v.get());
            }
          }
        }
        CountTotalExprRefs(a->value_);
      } else if (auto ret = As<ReturnStmt>(s)) {
        for (const auto& v : ret->value_) CountTotalExprRefs(v);
      } else if (auto yield = As<YieldStmt>(s)) {
        for (const auto& v : yield->value_) CountTotalExprRefs(v);
      } else if (auto eval = As<EvalStmt>(s)) {
        CountTotalExprRefs(eval->expr_);
      }
    };
    for (const auto& s : stmts) CountTotalStmtRefs(s);

    // Collect operands whose EVERY use, at any nesting depth, is a
    // batch_matmul operand. Comparing against the total use count — rather than
    // requiring one use — covers the shared-operand case:
    // e.g. a SwiGLU FFN where the activation X is the common LHS of both the
    // gate (X@W1) and up (X@W3) matmuls (use_count > 1). Treating a shared
    // operand as batch_matmul-only is safe: in the fit path it is sliced (not
    // dropped); in the !fit path it is dropped only when EVERY consuming matmul
    // is !fit (see the capacity gate below).
    std::unordered_set<const Var*> seen;
    for (const auto* v : batch_matmul_operands) {
      if (seen.insert(v).second && batch_matmul_operand_uses[v] == total_counts[v]) {
        record_batch_matmul_only_var(v);
      }
    }

    // Walk back through safe peelable `tile.reshape` chains so the upstream
    // dead `tile.reshape` (and any further-upstream `tile.load`) are also
    // skipped during rewriting. Without this, the orphan reshape would emit a
    // rank>2 tile that violates the post-pass `TileOps2D` property. Require the
    // input to have exactly one use across the WHOLE block (total_counts) so a
    // load also consumed inside a nested body is never peeled away.
    // Also walk back through a single-use tile.transpose_view so its underlying
    // whole tile.load joins the chain (needed so the !fit path can drop the dead
    // whole load, not just the view).
    auto stmt_def_map = BuildAssignDefMap(stmts);
    std::vector<const Var*> reshape_worklist = batch_matmul_only_vars_in_order;
    while (!reshape_worklist.empty()) {
      const Var* current = reshape_worklist.back();
      reshape_worklist.pop_back();
      auto def_it = stmt_def_map.find(current);
      if (def_it == stmt_def_map.end()) continue;
      auto chain_call = As<Call>(def_it->second->value_);
      const bool is_view = IsOp(chain_call, "tile.transpose_view");
      if (!is_view && !IsSafePeelableBatchMatmulReshape(chain_call)) continue;
      auto input_var = As<Var>(chain_call->args_[0]);
      if (!input_var) continue;
      if (total_counts[input_var.get()] != 1) continue;
      const size_t old_size = batch_matmul_only_vars_in_order.size();
      record_batch_matmul_only_var(input_var.get());
      if (batch_matmul_only_vars_in_order.size() != old_size) {
        reshape_worklist.push_back(input_var.get());
      }
    }

    // Per-batch_matmul capacity gate: when the operands' whole tiles do not fit
    // Mat together, their batch_matmul_only operand chains are loaded per batch
    // (ExtractBatchPage !fit path) — drop the dead whole chain here. A chain
    // shared with any FIT matmul stays whole (drop only chains feeding
    // exclusively !fit matmuls).
    std::unordered_set<const Var*> any_fit, any_notfit;
    std::function<void(const Var*, bool)> MarkChain = [&](const Var* v, bool fits) {
      if (!v) return;
      (fits ? any_fit : any_notfit).insert(v);
      auto it = stmt_def_map.find(v);
      if (it == stmt_def_map.end()) return;
      auto c = As<Call>(it->second->value_);
      if (!c || !c->op_ || c->args_.empty()) return;
      const std::string& n = c->op_->name_;
      if (n == "tile.transpose_view" || n == "tile.reshape") {
        if (auto in = As<Var>(c->args_[0])) MarkChain(in.get(), fits);
      }
    };
    for (const auto& s : stmts) {
      auto a = As<AssignStmt>(s);
      auto c = a ? As<Call>(a->value_) : nullptr;
      if (!c || !c->op_) continue;
      const std::string& n = c->op_->name_;
      const bool is_bmm = (n == "tile.batch_matmul");
      const bool is_bmm_acc = (n == "tile.batch_matmul_acc");
      if (!is_bmm && !is_bmm_acc) continue;
      const size_t lhs_i = is_bmm ? 0 : 1;
      const size_t rhs_i = is_bmm ? 1 : 2;
      if (rhs_i >= c->args_.size()) continue;
      // Mirror LowerBatchMatmul's per-operand routing so a non-contiguous whole
      // load (which goes per-batch) is also recognized as !fit here and its dead
      // whole chain is dropped (otherwise it would survive to codegen and trip the
      // ND2NZ contiguity guard).
      const bool capacity_fits = BatchOperandsWholeFit(As<TileType>(c->args_[lhs_i]->GetType()),
                                                       As<TileType>(c->args_[rhs_i]->GetType()));
      const bool lhs_fits =
          KeepOperandWhole(capacity_fits, TraceOperandBaseLoad(c->args_[lhs_i], stmt_def_map));
      const bool rhs_fits =
          KeepOperandWhole(capacity_fits, TraceOperandBaseLoad(c->args_[rhs_i], stmt_def_map));
      if (auto lv = As<Var>(c->args_[lhs_i])) MarkChain(lv.get(), lhs_fits);
      if (auto rv = As<Var>(c->args_[rhs_i])) MarkChain(rv.get(), rhs_fits);
    }
    for (const auto* v : batch_matmul_only_vars_in_order) {
      if (any_notfit.count(v) != 0 && any_fit.count(v) == 0) not_fit_drop_vars.insert(v);
    }
  }

  for (size_t stmt_index = 0; stmt_index < stmts.size(); ++stmt_index) {
    const auto& stmt = stmts[stmt_index];
    // ReturnStmt: substitute return values
    if (auto ret = As<ReturnStmt>(stmt)) {
      std::vector<ExprPtr> new_values;
      new_values.reserve(ret->value_.size());
      for (const auto& v : ret->value_) {
        new_values.push_back(Substitute(v, ctx.var_map));
      }
      result.push_back(std::make_shared<ReturnStmt>(new_values, ret->span_));
      continue;
    }

    // YieldStmt: substitute variables
    if (auto yield = As<YieldStmt>(stmt)) {
      std::vector<ExprPtr> new_values;
      new_values.reserve(yield->value_.size());
      for (const auto& v : yield->value_) {
        new_values.push_back(Substitute(v, ctx.var_map));
      }
      result.push_back(std::make_shared<YieldStmt>(new_values, yield->span_));
      continue;
    }

    // SeqStmts: recurse
    if (auto seq = As<SeqStmts>(stmt)) {
      auto inner = TransformBody(seq->stmts_, ctx, op_registry, span);
      result.insert(result.end(), inner.begin(), inner.end());
      continue;
    }

    // ScopeStmt: recurse into body — dispatch on the concrete derived class
    // since ScopeStmt is abstract and MutableCopy needs a concrete type.
    if (auto scope = As<ScopeStmt>(stmt)) {
      auto body_stmts = FlattenToStmts(scope->body_);
      auto inner = TransformBody(body_stmts, ctx, op_registry, span);
      auto new_body = SeqStmts::Flatten(std::move(inner), scope->body_->span_);
      auto rewrite = [&](auto&& concrete) -> StmtPtr {
        auto new_scope = MutableCopy(concrete);
        new_scope->body_ = new_body;
        return new_scope;
      };
      if (auto in_core = As<InCoreScopeStmt>(stmt)) {
        result.push_back(rewrite(in_core));
      } else if (auto cluster = As<ClusterScopeStmt>(stmt)) {
        result.push_back(rewrite(cluster));
      } else if (auto hier = As<HierarchyScopeStmt>(stmt)) {
        result.push_back(rewrite(hier));
      } else if (auto spmd = As<SpmdScopeStmt>(stmt)) {
        result.push_back(rewrite(spmd));
      } else if (auto split_aiv = As<SplitAivScopeStmt>(stmt)) {
        result.push_back(rewrite(split_aiv));
      } else if (auto runtime_scope = As<RuntimeScopeStmt>(stmt)) {
        result.push_back(rewrite(runtime_scope));
      } else {
        INTERNAL_UNREACHABLE_SPAN(scope->span_) << "Unknown ScopeStmt subclass: " << scope->TypeName();
      }
      continue;
    }

    // IfStmt: recurse into branches, substitute return_vars
    if (auto if_stmt = As<IfStmt>(stmt)) {
      auto new_cond = Substitute(if_stmt->condition_, ctx.var_map);

      auto then_ctx = ctx;
      auto then_stmts = FlattenToStmts(if_stmt->then_body_);
      auto new_then = TransformBody(then_stmts, then_ctx, op_registry, span);
      // Extract yield types before moving the vector
      auto yield_types = FindYieldTypes(new_then);
      auto new_then_body = SeqStmts::Flatten(std::move(new_then), if_stmt->then_body_->span_);

      FlattenContext else_ctx = ctx;
      std::optional<StmtPtr> new_else_body;
      if (if_stmt->else_body_.has_value()) {
        auto else_stmts = FlattenToStmts(*if_stmt->else_body_);
        auto new_else = TransformBody(else_stmts, else_ctx, op_registry, span);
        new_else_body = SeqStmts::Flatten(std::move(new_else), (*if_stmt->else_body_)->span_);
      }

      // Update return_vars types based on yield types (positional matching)
      if (yield_types.empty() && new_else_body.has_value()) {
        yield_types = FindYieldTypes(FlattenToStmts(*new_else_body));
      }
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(if_stmt->return_vars_.size());
      for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
        const auto& rv = if_stmt->return_vars_[i];
        if (i < yield_types.size() && yield_types[i] != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_hint_, yield_types[i], rv->span_);
          new_return_vars.push_back(new_rv);
          ctx.Insert(rv, new_rv);
        } else {
          new_return_vars.push_back(rv);
        }
      }

      auto new_if = MutableCopy(if_stmt);
      new_if->condition_ = new_cond;
      new_if->then_body_ = new_then_body;
      new_if->else_body_ = new_else_body;
      new_if->return_vars_ = new_return_vars;
      result.push_back(new_if);
      continue;
    }

    // ForStmt: recurse into body, substitute return_vars
    if (auto for_stmt = As<ForStmt>(stmt)) {
      auto new_start = Substitute(for_stmt->start_, ctx.var_map);
      auto new_stop = Substitute(for_stmt->stop_, ctx.var_map);
      auto new_step = Substitute(for_stmt->step_, ctx.var_map);

      auto body_ctx = ctx;
      std::vector<IterArgPtr> new_iter_args;
      new_iter_args.reserve(for_stmt->iter_args_.size());
      for (const auto& ia : for_stmt->iter_args_) {
        auto new_init = Substitute(ia->initValue_, ctx.var_map);
        auto new_ia = ia;
        if (new_init != ia->initValue_) {
          new_ia = std::make_shared<IterArg>(ia->name_hint_, new_init->GetType(), new_init, ia->span_);
          body_ctx.Insert(ia, new_ia);
        } else {
          body_ctx.Erase(ia);
        }
        new_iter_args.push_back(new_ia);
      }

      auto body_stmts = FlattenToStmts(for_stmt->body_);
      auto new_body_stmts = TransformBody(body_stmts, body_ctx, op_registry, span);
      auto new_body = SeqStmts::Flatten(std::move(new_body_stmts), for_stmt->body_->span_);

      // Update return_vars types to match iter_arg types (positional matching)
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(for_stmt->return_vars_.size());
      for (size_t i = 0; i < for_stmt->return_vars_.size(); ++i) {
        const auto& rv = for_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          ctx.Insert(rv, new_rv);
        } else {
          new_return_vars.push_back(rv);
        }
      }

      auto new_for = MutableCopy(for_stmt);
      new_for->start_ = new_start;
      new_for->stop_ = new_stop;
      new_for->step_ = new_step;
      new_for->iter_args_ = new_iter_args;
      new_for->body_ = new_body;
      new_for->return_vars_ = new_return_vars;
      result.push_back(new_for);
      continue;
    }

    // WhileStmt: recurse into body, substitute return_vars
    if (auto while_stmt = As<WhileStmt>(stmt)) {
      auto body_ctx = ctx;
      std::vector<IterArgPtr> new_iter_args;
      new_iter_args.reserve(while_stmt->iter_args_.size());
      for (const auto& ia : while_stmt->iter_args_) {
        auto new_init = Substitute(ia->initValue_, ctx.var_map);
        auto new_ia = ia;
        if (new_init != ia->initValue_) {
          new_ia = std::make_shared<IterArg>(ia->name_hint_, new_init->GetType(), new_init, ia->span_);
          body_ctx.Insert(ia, new_ia);
        } else {
          body_ctx.Erase(ia);
        }
        new_iter_args.push_back(new_ia);
      }

      auto new_cond = Substitute(while_stmt->condition_, body_ctx.var_map);
      auto body_stmts = FlattenToStmts(while_stmt->body_);
      auto new_body_stmts = TransformBody(body_stmts, body_ctx, op_registry, span);
      auto new_body = SeqStmts::Flatten(std::move(new_body_stmts), while_stmt->body_->span_);

      // Update return_vars types to match iter_arg types (positional matching)
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(while_stmt->return_vars_.size());
      for (size_t i = 0; i < while_stmt->return_vars_.size(); ++i) {
        const auto& rv = while_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          ctx.Insert(rv, new_rv);
        } else {
          new_return_vars.push_back(rv);
        }
      }

      auto new_while = MutableCopy(while_stmt);
      new_while->condition_ = new_cond;
      new_while->iter_args_ = new_iter_args;
      new_while->body_ = new_body;
      new_while->return_vars_ = new_return_vars;
      result.push_back(new_while);
      continue;
    }

    // EvalStmt: substitute variables in the expression
    if (auto eval = As<EvalStmt>(stmt)) {
      auto new_expr = Substitute(eval->expr_, ctx.var_map);
      if (new_expr != eval->expr_) {
        // Re-create tile ops via OpRegistry for proper type deduction
        if (auto call = As<Call>(new_expr)) {
          if (call->op_ && call->op_->name_.substr(0, 5) == "tile.") {
            auto new_call = op_registry.Create(call->op_->name_, call->args_, call->kwargs_, span);
            result.push_back(std::make_shared<EvalStmt>(new_call, eval->span_));
            continue;
          }
        }
        result.push_back(std::make_shared<EvalStmt>(new_expr, eval->span_));
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    // AssignStmt: the main transformation logic
    auto assign = As<AssignStmt>(stmt);
    if (!assign) {
      result.push_back(stmt);
      continue;
    }

    // Drop the dead whole load/transpose_view chain of a !fit batch_matmul: it is
    // loaded per batch at the matmul (ExtractBatchPage !fit path), so the whole
    // tile is never referenced and must not occupy L1.
    if (not_fit_drop_vars.count(assign->var_.get()) != 0) {
      ctx.Insert(assign->var_, assign->var_);  // identity mapping so any lookup resolves
      continue;
    }

    auto call = As<Call>(assign->value_);
    auto global_var = call ? As<GlobalVar>(call->op_) : nullptr;

    // Non-call assignment or function call (GlobalVar): substitute and pass through
    if (!call || global_var) {
      auto new_value = Substitute(assign->value_, ctx.var_map);
      if (new_value != assign->value_) {
        auto new_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_value, assign->span_));
        ctx.Insert(assign->var_, new_var);
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    const auto& op_name = call->op_->name_;

    // ---- tile.load on >2D tile: flatten the result tile to 2D (hardware tiles
    //      are always 2D), keeping the tensor-rank source window for codegen —
    //      except a natural Mat load with a real batch (>1) collapses its window
    //      to 2D as well (the ND2NZ path rejects rank>2 GlobalTensors). ----
    if (IsOp(call, "tile.load")) {
      // A batch_matmul operand load is KEPT here and sliced per batch by
      // ExtractBatchPage (the fit path) — both operands, transposed or not, are
      // handled identically (whole tile in L1 + per-batch slice). Only a !fit
      // operand load is dropped (above), to be re-emitted per batch.

      // Substitute args via ctx.var_map so all operand Vars reference the latest SSA values.
      std::vector<ExprPtr> sub_args;
      sub_args.reserve(call->args_.size());
      for (const auto& arg : call->args_) {
        sub_args.push_back(Substitute(arg, ctx.var_map));
      }

      auto result_tile = As<TileType>(call->GetType());
      if (result_tile && result_tile->shape_.size() > 2) {
        // Rank>2 tile.load: keep the original tensor-rank offsets/shapes, but
        // construct a 2D TileType for the result. DeduceTileLoadType produces a
        // rank>2 TileType from those shapes, but hardware tiles are always 2D.
        // The pass manually overrides the result type to 2D.
        auto [merged, last] = ComputeMergedShape(result_tile->shape_, "tile.load result");

        auto flat_shape_exprs = Make2DShapeExprs(merged, last, span);
        // Preserve any TileView (blayout/slayout/fractal/pad) the source tile
        // already carried — e.g. LowerCompositeOps tags a transposed-load Mat
        // rhs with TileView(blayout=row_major, slayout=col_major) so the
        // downstream TLOAD matches the DN2ZN pattern. The implicit Mat default
        // (col/row = ND) would otherwise emit an unsupported DN2ND TLOAD (#1540).
        std::optional<TileView> flat_tile_view;
        if (result_tile->tile_view_.has_value()) {
          const auto& orig_tv = *result_tile->tile_view_;
          // Carry the original valid_shape through the flatten. When it is a
          // proper per-dim valid_shape (e.g. a dynamic min(CHUNK, D-c) tail from
          // the dynamic-tile strip-mine), merge it the same way as the physical
          // shape so the runtime tail extent survives; otherwise the flattened
          // tile is fully valid (valid_shape == physical 2D shape).
          std::vector<ExprPtr> flat_valid = flat_shape_exprs;
          if (orig_tv.valid_shape.size() == result_tile->shape_.size()) {
            flat_valid = ComputeMergedValidShape(orig_tv.valid_shape, result_tile->shape_, span);
          }
          flat_tile_view = TileView(flat_valid, /*stride=*/{}, /*start_offset=*/nullptr, orig_tv.blayout,
                                    orig_tv.slayout, orig_tv.fractal, orig_tv.pad);
        } else {
          flat_tile_view =
              tile_view_semantics::GetImplicitTileView(flat_shape_exprs, result_tile->memory_space_);
        }
        auto flat_tile_type = std::make_shared<TileType>(flat_shape_exprs, result_tile->dtype_, std::nullopt,
                                                         flat_tile_view, result_tile->memory_space_);

        // The rank>2 source window is preserved as-is for codegen. A natural Mat
        // load lowers to ND2NZ, which requires a 2-dim GlobalTensor — that
        // collapse is owned by the tile.load codegen (it triggers on the NZ result
        // tile), so flatten only needs to flatten the RESULT tile to 2D here.
        auto flat_call =
            std::make_shared<Call>(call->op_, sub_args, call->kwargs_, flat_tile_type, call->span_);
        auto flat_var = std::make_shared<Var>(assign->var_->name_hint_, flat_tile_type, assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(flat_var, flat_call, assign->span_));
        ctx.Insert(assign->var_, flat_var);
        continue;
      }
      // ≤2D tile.load: honor any pending var_map substitutions
      auto rebuilt = op_registry.Create(op_name, sub_args, call->kwargs_, span);
      // Contextual tensor.slice lowering may deliberately narrow valid_shape or
      // attach a pad contract beyond what a standalone tile.load deducer can
      // recover from its operands.  Validate the rebuilt load, then keep that
      // authoritative result type.
      auto new_call = RetypeTileCall(rebuilt, result_tile, "tile.load", call->span_);
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
      result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
      ctx.Insert(assign->var_, new_var);
      continue;
    }

    // ---- tile.slice: preserve contextual validity; lower drop_dims explicitly ----
    // PTO subview operates in the source rank and has no rank-reduction operand.
    // Represent a rank-dropping slice as a full-rank subview followed by an
    // ordinary tile.reshape.  This also handles trailing-axis drops correctly
    // ([M, 1] -> [1, M]) instead of letting codegen bind the pre-drop shape to
    // the post-drop result buffer.
    if (IsOp(call, "tile.slice")) {
      std::vector<ExprPtr> sub_args;
      sub_args.reserve(call->args_.size());
      for (const auto& arg : call->args_) {
        sub_args.push_back(Substitute(arg, ctx.var_map));
      }

      auto authoritative = As<TileType>(call->GetType());
      INTERNAL_CHECK_SPAN(authoritative && authoritative->shape_.size() <= 2, call->span_)
          << "FlattenTileNdTo2D: tile.slice result must be at most 2D after flattening";

      bool has_drop_dims = false;
      if (sub_args.size() == 5) {
        auto drop_tuple = As<MakeTuple>(sub_args[4]);
        INTERNAL_CHECK_SPAN(drop_tuple, call->args_[4]->span_)
            << "FlattenTileNdTo2D: tile.slice drop_dims must be a compile-time MakeTuple";
        has_drop_dims = !drop_tuple->elements_.empty();
      }

      if (has_drop_dims) {
        std::vector<ExprPtr> full_rank_args(sub_args.begin(), sub_args.begin() + 4);
        auto full_slice = op_registry.Create("tile.slice", full_rank_args, call->kwargs_, span);
        auto full_var = std::make_shared<Var>(assign->var_->name_hint_ + "_full_rank", full_slice->GetType(),
                                              assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(full_var, full_slice, assign->span_));

        auto result_shape = std::make_shared<MakeTuple>(authoritative->shape_, call->span_);
        auto rebuilt = op_registry.Create("tile.reshape", {full_var, result_shape}, {}, span);
        auto new_call = RetypeTileCall(rebuilt, authoritative, "rank-dropping tile.slice", call->span_);
        auto new_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
        ctx.Insert(assign->var_, new_var);
        continue;
      }

      auto rebuilt = op_registry.Create("tile.slice", sub_args, call->kwargs_, span);
      auto new_call = RetypeTileCall(rebuilt, authoritative, "tile.slice", call->span_);
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
      result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
      ctx.Insert(assign->var_, new_var);
      continue;
    }

    // ---- tile.store: inject original tensor-rank partition shape for rank>2 tensors ----
    // tile.store semantics: (2D) tile -> rank>2 tensor. Original tensor-rank
    // offsets are preserved; codegen uses the tensor view plus a partition_view
    // over the original tensor-rank window to produce the 2D result.
    // Signature: (tile, offsets, output_tensor[, shapes])
    if (IsOp(call, "tile.store")) {
      auto orig_tile_type = As<TileType>(call->args_[0]->GetType());

      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size() + 1);
      // Push all original args (tile, offsets, output_tensor) with substitution
      for (const auto& arg : call->args_) {
        new_args.push_back(Substitute(arg, ctx.var_map));
      }

      // If the (substituted) tile operand is still >2D — e.g. a user-written
      // ``pl.reshape(tile_2d, [B, 1, D])`` to feed ``pl.assemble`` into a
      // rank>2 tensor view — insert a ``tile.reshape`` to flatten it to 2D.
      // Codegen for ``tile.store`` requires a 2D tile; the original N-rank
      // shape still flows through as the ``shapes`` partition operand built
      // below from ``orig_tile_type``.
      auto tile_arg_type = As<TileType>(new_args[0]->GetType());
      if (tile_arg_type && tile_arg_type->shape_.size() > 2) {
        auto [merged, last] = ComputeMergedShape(tile_arg_type->shape_, "tile.store tile operand");
        auto reshape_shape = MakeShapeTupleFromInts({merged, last}, span);
        auto reshape_call = op_registry.Create("tile.reshape", {new_args[0], reshape_shape}, span);
        auto flat_var = std::make_shared<Var>("flat_tile", reshape_call->GetType(), span);
        result.push_back(std::make_shared<AssignStmt>(flat_var, reshape_call, span));
        new_args[0] = flat_var;
      }

      auto out_tensor_type = AsTensorTypeLike(new_args[2]->GetType());
      if (orig_tile_type && out_tensor_type && out_tensor_type->shape_.size() > 2) {
        // Inject the original tensor-rank *valid partition* tuple as the 4th
        // argument.  The partition has the same rank as the tensor, with 1s for
        // leading batch dims not covered by the tile, followed by the tile's
        // effective valid extents (not its physical capacity).
        const size_t tensor_rank = out_tensor_type->shape_.size();
        const size_t tile_rank = orig_tile_type->shape_.size();
        std::vector<ExprPtr> partition_shape;
        partition_shape.reserve(tensor_rank);
        for (size_t i = tile_rank; i < tensor_rank; ++i) {
          partition_shape.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, span));
        }
        for (const auto& dim : GetValidShape(orig_tile_type)) {
          partition_shape.push_back(dim);
        }
        new_args.push_back(std::make_shared<MakeTuple>(partition_shape, span));
      }

      // The store result carries the post-write semantic contract established by
      // ConvertTensorToTileOps (valid_shape, pad, and DistributedTensor window
      // identity).  The destination operand still describes the pre-write value,
      // so rebuilding from args[2] would silently roll that metadata back.  Remap
      // embedded Exprs to their flattened SSA versions while taking only the
      // substituted destination's MemRef.
      auto out_type = CloneTypeWithMemRefAndRemapExprs(
          call->GetType(), GetTypeMemRef(new_args[2]->GetType()),
          [&](const ExprPtr& expr) { return Substitute(expr, ctx.var_map); });
      auto new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, out_type, call->span_);
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
      result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
      ctx.Insert(assign->var_, new_var);
      continue;
    }

    // ---- tile.create / tile.full with >2D shape: flatten shape directly ----
    if (IsOp(call, "tile.create") || IsOp(call, "tile.full")) {
      auto result_tile = As<TileType>(call->GetType());
      if (result_tile && result_tile->shape_.size() > 2) {
        auto [merged, last] = ComputeMergedShape(result_tile->shape_, op_name);

        // Rebuild the call with 2D shape
        auto new_shape_tuple = MakeShapeTupleFromInts({merged, last}, span);
        std::vector<ExprPtr> new_args;
        // First arg is the shape tuple
        new_args.push_back(new_shape_tuple);
        // Remaining args:
        //  - tile.create's optional 2nd arg is a valid_shape tuple ranked like the
        //    ND physical shape; merge it to 2D the same way (contiguous-prefix
        //    precondition enforced by ComputeMergedValidShape) so the flattened
        //    create keeps a rank-2 valid region instead of a rank-mismatched one.
        //  - tile.full's 2nd arg is a scalar fill value — pass it through.
        const bool is_create = IsOp(call, "tile.create");
        for (size_t i = 1; i < call->args_.size(); ++i) {
          if (is_create) {
            auto valid_tuple = As<MakeTuple>(call->args_[i]);
            if (valid_tuple && valid_tuple->elements_.size() == result_tile->shape_.size()) {
              auto merged_valid = ComputeMergedValidShape(valid_tuple->elements_, result_tile->shape_, span);
              new_args.push_back(std::make_shared<MakeTuple>(merged_valid, span));
              continue;
            }
          }
          new_args.push_back(Substitute(call->args_[i], ctx.var_map));
        }

        auto new_call = op_registry.Create(op_name, new_args, call->kwargs_, span);
        auto flat_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(flat_var, new_call, assign->span_));
        ctx.Insert(assign->var_, flat_var);
        continue;
      }
      // ≤2D: pass through
      result.push_back(stmt);
      continue;
    }

    // ---- tile.sum/tile.max/tile.min: remap axis to 1 (last axis of 2D) ----
    if (IsOp(call, "tile.sum") || IsOp(call, "tile.max") || IsOp(call, "tile.min")) {
      if (!call->args_.empty()) {
        auto input_tile = As<TileType>(call->args_[0]->GetType());
        if (IsNdTile(input_tile)) {
          // Substitute args
          std::vector<ExprPtr> new_args;
          new_args.reserve(call->args_.size());
          for (const auto& arg : call->args_) {
            new_args.push_back(Substitute(arg, ctx.var_map));
          }

          // Update axis kwarg to 1 (last axis of 2D tile)
          std::vector<std::pair<std::string, std::any>> new_kwargs;
          for (const auto& [key, val] : call->kwargs_) {
            if (key == "axis") {
              new_kwargs.emplace_back("axis", 1);
            } else {
              new_kwargs.emplace_back(key, val);
            }
          }

          auto new_call = op_registry.Create(op_name, new_args, new_kwargs, span);
          auto new_var =
              std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
          result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
          ctx.Insert(assign->var_, new_var);
          continue;
        }
      }
    }

    // ---- tile.batch_matmul: delegate to LowerBatchMatmul ----
    if (IsOp(call, "tile.batch_matmul")) {
      auto lowering = LowerBatchMatmul(assign, call, stmts, stmt_index, ctx, op_registry, span);
      result.insert(result.end(), lowering.stmts.begin(), lowering.stmts.end());
      if (lowering.fused_store) {
        ctx.Insert(lowering.store_orig_var, lowering.store_result_var);
        ++stmt_index;  // Skip the next tile.store; it has been fused above.
      } else {
        ctx.Insert(assign->var_, lowering.output_var);
      }
      continue;
    }

    // ---- tile.batch_matmul_acc: delegate to LowerBatchMatmulAcc ----
    if (IsOp(call, "tile.batch_matmul_acc")) {
      auto lowering = LowerBatchMatmulAcc(assign, call, stmts, ctx, op_registry, span);
      result.insert(result.end(), lowering.stmts.begin(), lowering.stmts.end());
      ctx.Insert(assign->var_, lowering.output_var);
      continue;
    }

    // An ND transpose_view cannot be represented by transposing the merged 2D
    // view: [B,M,N] -> [B,N,M] must flatten to [B*N,M], whereas a 2D view swap
    // would produce [N,B*M].  The latter encoding is intentionally used only
    // inside the batch-matmul page extractor, which understands its
    // column-batched representation.  Reject every standalone consumer rather
    // than silently changing the logical transpose.
    if (IsOp(call, "tile.transpose_view") && IsNdTile(As<TileType>(call->args_[0]->GetType())) &&
        batch_matmul_only_vars.count(assign->var_.get()) == 0) {
      CHECK_SPAN(false, call->span_)
          << "FlattenTileNdTo2D: standalone tile.transpose_view on an ND tile is not representable as a "
             "single 2D zero-copy view; use tile.transpose for a materialized trailing-axis transpose, "
             "or consume the view exclusively with tile.batch_matmul";
    }

    // ---- tile.transpose feeding only tile.batch_matmul[_acc]: skip and let lowering peel it ----
    if (IsOp(call, "tile.transpose") && batch_matmul_only_vars.count(assign->var_.get()) != 0) {
      ctx.Insert(assign->var_, assign->var_);  // identity mapping for safety
      continue;
    }

    // ---- standalone tile.transpose: this pass solely owns scratch materialization ----
    // High-level transposes arrive in the 3-arg form (input, axis1, axis2); the
    // pto.ttrans scratch is emitted here as the codegen-ready 4-arg form.
    //   >2D  → LowerNdTranspose: per-page 2D transposes, each with sliced scratch.
    //   2D   → emit one scratch tile.create + a 4-arg tile.transpose.
    // An already-4-arg 2D transpose (e.g. hand-built IR) is revalidated here as
    // well, while retaining its authoritative result validity.
    if (IsOp(call, "tile.transpose") && batch_matmul_only_vars.count(assign->var_.get()) == 0) {
      if (IsNdTile(As<TileType>(call->args_[0]->GetType()))) {
        auto lowering = LowerNdTranspose(assign, call, ctx, op_registry, span);
        result.insert(result.end(), lowering.stmts.begin(), lowering.stmts.end());
        ctx.Insert(assign->var_, lowering.output_var);
        continue;
      }
      auto in = Substitute(call->args_[0], ctx.var_map);
      auto in_type = As<TileType>(in->GetType());
      INTERNAL_CHECK_SPAN(in_type, span)
          << "Internal error: tile.transpose input must be TileType in FlattenTileNdTo2D";

      ExprPtr scratch;
      if (call->args_.size() == 3) {
        // pto.ttrans reuses the SOURCE type for both ins operands, so scratch shape ==
        // input shape (NOT the transposed output shape), in the input's memory space.
        MemorySpace scratch_mem =
            in_type->memory_space_.has_value() ? *in_type->memory_space_ : MemorySpace::Vec;
        auto scratch_shape = std::make_shared<MakeTuple>(in_type->shape_, span);
        auto scratch_valid = std::make_shared<MakeTuple>(GetValidShape(in_type), span);
        std::vector<std::pair<std::string, std::any>> scratch_kw = {
            {"dtype", in_type->dtype_},
            {"target_memory", scratch_mem},
        };
        auto scratch_create =
            op_registry.Create("tile.create", {scratch_shape, scratch_valid}, scratch_kw, span);
        // Scratch is a fresh allocation but must expose the same PTO tile-buffer
        // layout and validity as the source operand consumed by pto.ttrans.
        TileView scratch_view = tile_view_semantics::GetEffectiveTileView(*in_type);
        scratch_view.valid_shape = GetValidShape(in_type);
        scratch_view.stride.clear();
        scratch_view.start_offset = nullptr;
        auto scratch_type = std::make_shared<TileType>(in_type->shape_, in_type->dtype_, std::nullopt,
                                                       scratch_view, scratch_mem);
        scratch_create = RetypeCall(scratch_create, scratch_type);
        auto scratch_var = std::make_shared<Var>("transpose_tmp", scratch_create->GetType(), span);
        result.push_back(std::make_shared<AssignStmt>(scratch_var, scratch_create, span));
        scratch = scratch_var;
      } else {
        scratch = Substitute(call->args_[3], ctx.var_map);
      }

      auto rebuilt = op_registry.Create(
          "tile.transpose",
          {in, Substitute(call->args_[1], ctx.var_map), Substitute(call->args_[2], ctx.var_map), scratch},
          call->kwargs_, span);
      auto authoritative = As<TileType>(call->GetType());
      auto t_call = RetypeTileCall(rebuilt, authoritative, "tile.transpose", call->span_);
      auto t_var = std::make_shared<Var>(assign->var_->name_hint_, t_call->GetType(), assign->var_->span_);
      result.push_back(std::make_shared<AssignStmt>(t_var, t_call, assign->span_));
      ctx.Insert(assign->var_, t_var);
      continue;
    }

    // ---- tile.reshape feeding only tile.batch_matmul: skip (identity) when it is
    //      a safe batch-only reshape that `NormalizeBatchMatmulOperand` peels, so
    //      no orphan rank>2 reshape survives. The underlying tile.load is reused by
    //      the lowering (fit path) or re-emitted per batch (!fit path). ----
    if (IsOp(call, "tile.reshape") && batch_matmul_only_vars.count(assign->var_.get()) != 0 &&
        IsSafePeelableBatchMatmulReshape(call)) {
      ctx.Insert(assign->var_, assign->var_);  // identity mapping for safety
      continue;
    }

    // ---- tile.reshape: flatten rank-raising results and keep contextual validity ----
    // A user-visible ND reshape is only a logical view; PTO must see its merged
    // 2D representation.  Re-deducing a contextual <=2D reshape from a newly
    // substituted operand can likewise widen valid_shape, so both cases retain
    // the original call's authoritative result contract.
    if (IsOp(call, "tile.reshape")) {
      auto input = Substitute(call->args_[0], ctx.var_map);
      auto authoritative = As<TileType>(call->GetType());
      INTERNAL_CHECK_SPAN(authoritative, call->span_)
          << "FlattenTileNdTo2D: tile.reshape result must be TileType";

      TileTypePtr result_type = authoritative;
      ExprPtr result_shape = Substitute(call->args_[1], ctx.var_map);
      if (authoritative->shape_.size() > 2) {
        result_type = FlattenTileTypeContract(authoritative, "tile.reshape result", call->span_);
        result_shape = std::make_shared<MakeTuple>(result_type->shape_, call->span_);
      }

      auto rebuilt = op_registry.Create("tile.reshape", {input, result_shape}, call->kwargs_, span);
      auto new_call = RetypeTileCall(rebuilt, result_type, "tile.reshape", call->span_);
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
      result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
      ctx.Insert(assign->var_, new_var);
      continue;
    }

    // ---- All other tile ops and non-tile ops: substitute args ----
    {
      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size());
      bool changed = false;
      for (const auto& arg : call->args_) {
        auto new_arg = Substitute(arg, ctx.var_map);
        new_args.push_back(new_arg);
        if (new_arg != arg) changed = true;
      }

      if (!changed) {
        result.push_back(stmt);
      } else {
        // Re-create tile ops via OpRegistry for proper type deduction with 2D args;
        // non-tile ops keep the original type.
        auto new_call =
            (op_name.substr(0, 5) == "tile.")
                ? op_registry.Create(op_name, new_args, call->kwargs_, span)
                : std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->GetType(), call->span_);

        auto new_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
        ctx.Insert(assign->var_, new_var);
      }
    }
  }

  return result;
}

/**
 * @brief Transform a single InCore function: flatten >2D tiles to 2D.
 *
 * This includes:
 * 1. Flattening >2D tile ops in the function body to 2D
 * 2. Preserving original tensor-rank offsets/shapes in tile.load/store for
 *    codegen to use with tensor_view + partition_view
 */
FunctionPtr TransformFunction(const FunctionPtr& func) {
  if (!IsInCoreType(func->func_type_)) {
    return func;
  }

  const auto& span = func->span_;
  auto& op_registry = OpRegistry::GetInstance();

  // Validate preconditions
  PreconditionChecker checker;
  checker.VisitStmt(func->body_);

  FlattenContext ctx;

  // Transform body
  auto body_stmts = FlattenToStmts(func->body_);
  auto new_stmts = TransformBody(body_stmts, ctx, op_registry, span);
  auto new_body = SeqStmts::Flatten(std::move(new_stmts), span);

  // return_types_ are unchanged: InCore functions return tensors (not tiles),
  // and this pass only flattens tile ops. Tensor types are never modified.
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

// ============================================================================
// Property Verifier
// ============================================================================

/**
 * @brief Visitor that checks all tile ops in InCore functions use static, at-most-2D tiles.
 */
class TileOps2DVerifier : public IRVisitor {
 public:
  explicit TileOps2DVerifier(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->value_)) {
      CheckCall(call, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->expr_)) {
      CheckCall(call, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  void CheckCall(const CallPtr& call, const Span& stmt_span) {
    if (!call || !call->op_) return;
    auto gv = As<GlobalVar>(call->op_);
    if (gv) return;

    const auto& name = call->op_->name_;
    if (name.substr(0, 5) != "tile.") return;

    auto check_tile_type = [&](const TileTypePtr& tile, const std::string& role) {
      if (!tile) return false;
      if (tile->shape_.size() > 2) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileOps2D", 0,
                                  "Tile op '" + name + "' in InCore function '" + func_name_ + "' " + role +
                                      " >2D tile (should have been flattened to at most 2D)",
                                  stmt_span);
        return true;
      }
      for (size_t i = 0; i < tile->shape_.size(); ++i) {
        if (As<ConstInt>(tile->shape_[i])) continue;
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileOps2D", 0,
                                  "Tile op '" + name + "' in InCore function '" + func_name_ + "' " + role +
                                      " tile with dynamic physical dimension " + std::to_string(i) +
                                      " (PTO tile buffers require static capacity)",
                                  stmt_span);
        return true;
      }
      return false;
    };

    // Rank-1 logical tiles are allowed here; PTO type conversion maps them to
    // a physical [1, N] tile_buf. Rank>2 and dynamic capacity are forbidden.
    check_tile_type(As<TileType>(call->GetType()), "produces");

    // Post-pass, every tile.transpose must be the codegen-ready 4-arg form: this pass
    // materializes the pto.ttrans scratch for both 2D and per-page >2D transposes, so a
    // surviving 3-arg form means scratch was never allocated.
    if (name == "tile.transpose" && call->args_.size() != 4) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileOps2D", 0,
                                "tile.transpose in InCore function '" + func_name_ + "' has " +
                                    std::to_string(call->args_.size()) +
                                    " arguments (expected 4: input, axis1, axis2, scratch after "
                                    "FlattenTileNdTo2D)",
                                stmt_span);
    }

    // Check argument types
    for (const auto& arg : call->args_) {
      auto arg_tile = As<TileType>(arg->GetType());
      if (check_tile_type(arg_tile, "has")) break;
    }
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
};

}  // namespace

// ============================================================================
// Property Verifier Impl (public)
// ============================================================================

class TileOps2DPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "TileOps2D"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (!IsInCoreType(func->func_type_)) continue;
      TileOps2DVerifier verifier(diagnostics, func->name_);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateTileOps2DPropertyVerifier() {
  return std::make_shared<TileOps2DPropertyVerifierImpl>();
}

// ============================================================================
// Pass Factory
// ============================================================================

namespace pass {

Pass FlattenTileNdTo2D() {
  return CreateFunctionPass(TransformFunction, "FlattenTileNdTo2D", kFlattenTileNdTo2DProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

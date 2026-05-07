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
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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
#include "pypto/ir/transforms/utils/mutable_copy.h"
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
  CHECK(ci) << "FlattenTileNdTo2D: all tile dimensions must be static (ConstInt), "
            << "but found dynamic dimension in " << context;
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
    if (!tile_type || tile_type->shape_.size() <= 2) return;
    for (size_t i = 0; i < tile_type->shape_.size(); ++i) {
      CHECK(As<ConstInt>(tile_type->shape_[i]))
          << "FlattenTileNdTo2D: tile dimension " << i << " must be static (ConstInt) "
          << "for tile op '" << op_name << "'";
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
//   1. Normalize operand transpose semantics — peel off inline tile.transpose
//      wrappers and recognize tile.load(transpose=True) as the same logical
//      operand-transpose contract (batch_matmul uses structural transpose, not kwargs).
//
//   2. Broadcast batch dimensions — compute the output batch shape via
//      NumPy-style broadcasting (e.g. [2,1] x [1,3] -> [2,3]).
//
//   3. Detect direct-store fusion — if the very next statement is a tile.store
//      consuming this result, fuse per-batch stores directly instead of
//      assembling into a temporary tile. This avoids an intermediate buffer.
//
//   4. Unroll — for each flat batch index 0..batch_count-1:
//      a. Decompose the flat index into per-dim indices for lhs and rhs,
//         respecting broadcast (size-1 dims always map to index 0).
//      b. Extract the 2D [M,K] / [K,N] page via one of three strategies:
//         - Re-emit tile.load with batch-adjusted offsets (when the original
//           operand was a Mat-memory tile.load — avoids a slice+reshape).
//         - tile.slice on already-flattened 2D tile (row-offset slicing).
//         - rank>2 tile.slice + tile.reshape to 2D (general fallback).
//      c. Optionally append tile.transpose(0,1) for transposed operands.
//      d. Emit tile.matmul(lhs_2d, rhs_2d).
//      e. Cast dtype if matmul output (FP32) differs from expected result dtype.
//      f. Either tile.store (fused path) or tile.assemble into output tile.
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
  ExprPtr operand;            ///< After var_map substitution
  ExprPtr original_operand;   ///< Before substitution (for def lookup)
  TileTypePtr operand_type;   ///< Type after substitution
  TileTypePtr original_type;  ///< Type before substitution
  bool transpose = false;     ///< True if wrapped in trailing-axis tile.transpose
};

/// Resolve an inline or single-definition tile.transpose wrapper around a batch_matmul operand.
CallPtr ResolveBatchOperandTranspose(const ExprPtr& operand_expr, const AssignDefMap& def_map) {
  if (auto transpose_call = As<Call>(operand_expr)) {
    if (transpose_call->op_ && transpose_call->op_->name_ == "tile.transpose") {
      return transpose_call;
    }
  }

  if (auto operand_var = As<Var>(operand_expr)) {
    auto def_it = def_map.find(operand_var.get());
    if (def_it != def_map.end()) {
      if (auto transpose_call = As<Call>(def_it->second->value_)) {
        if (transpose_call->op_ && transpose_call->op_->name_ == "tile.transpose") {
          return transpose_call;
        }
      }
    }
  }

  return nullptr;
}

/// Normalize one batch_matmul operand by:
///  - peeling off a direct tile.transpose wrapper
///  - recognizing tile.load(transpose=True) as the same operand-transpose semantic
///  - returning a base operand plus unified transpose/type information
BatchOperandInfo NormalizeBatchMatmulOperand(const ExprPtr& operand_expr, const std::string& operand_name,
                                             const AssignDefMap& def_map, const FlattenContext& ctx) {
  BatchOperandInfo info;
  ExprPtr base_operand = operand_expr;

  if (auto transpose_call = ResolveBatchOperandTranspose(operand_expr, def_map)) {
    if (transpose_call->op_ && transpose_call->op_->name_ == "tile.transpose") {
      CHECK(transpose_call->args_.size() == 3)
          << "FlattenTileNdTo2D: tile.transpose inside tile.batch_matmul must have 3 arguments";

      auto input_type = As<TileType>(transpose_call->args_[0]->GetType());
      CHECK(input_type) << "FlattenTileNdTo2D: tile.batch_matmul " << operand_name
                        << " transpose operand must wrap a TileType, but got "
                        << transpose_call->args_[0]->GetType()->TypeName();

      auto axis1_const = As<ConstInt>(transpose_call->args_[1]);
      auto axis2_const = As<ConstInt>(transpose_call->args_[2]);
      CHECK(axis1_const && axis2_const)
          << "FlattenTileNdTo2D: tile.batch_matmul " << operand_name << " transpose axes must be ConstInt";

      int64_t axis1 = NormalizeAxisIndex(axis1_const->value_, input_type->shape_.size(),
                                         "tile.batch_matmul " + operand_name + " transpose axis1");
      int64_t axis2 = NormalizeAxisIndex(axis2_const->value_, input_type->shape_.size(),
                                         "tile.batch_matmul " + operand_name + " transpose axis2");
      CHECK(IsTrailingMatrixAxisSwap(axis1, axis2, input_type->shape_.size()))
          << "FlattenTileNdTo2D: tile.batch_matmul only supports operand transpose on the trailing "
             "matrix axes, but got axes "
          << axis1 << " and " << axis2;

      base_operand = transpose_call->args_[0];
      info.transpose = true;
    }
  }

  // If no tile.transpose wrapper found, check if the operand is a tile.load(transpose=True).
  // In this case the load result already has swapped trailing dims (done by DeduceTileLoadType),
  // so we record transpose=true but must also un-swap original_type so that LowerBatchMatmul
  // sees the pre-transpose "source" shape (matching the tile.transpose-wrapper convention).
  bool transpose_from_load = false;
  if (!info.transpose) {
    ExprPtr check_expr = base_operand;
    // Resolve through Var to its definition if needed.
    if (auto operand_var = As<Var>(check_expr)) {
      auto def_it = def_map.find(operand_var.get());
      if (def_it != def_map.end()) {
        check_expr = def_it->second->value_;
      }
    }
    if (auto load_call = As<Call>(check_expr)) {
      if (load_call->op_ && load_call->op_->name_ == "tile.load" &&
          load_call->GetKwarg<bool>("transpose", false)) {
        info.transpose = true;
        transpose_from_load = true;
      }
    }
  }

  info.original_operand = base_operand;
  info.original_type = As<TileType>(base_operand->GetType());
  CHECK(info.original_type) << "FlattenTileNdTo2D: tile.batch_matmul " << operand_name
                            << " expects TileType operand, but got " << base_operand->GetType()->TypeName();

  // When transpose came from tile.load(transpose=True), the result shape is already
  // post-transpose. Un-swap the trailing two dims so original_type matches the convention
  // expected by LowerBatchMatmul (pre-transpose "source" shape).
  if (transpose_from_load && info.original_type->shape_.size() >= 2) {
    auto unswapped_shape = info.original_type->shape_;
    std::iter_swap(unswapped_shape.end() - 2, unswapped_shape.end() - 1);
    info.original_type =
        std::make_shared<TileType>(unswapped_shape, info.original_type->dtype_, info.original_type->memref_,
                                   info.original_type->tile_view_, info.original_type->memory_space_);
  }

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

/// Extract the 2D matrix page for a given batch index from an operand.
///
/// Three strategies:
///  (1) Mat-load: re-emit tile.load with batch-adjusted offsets (avoids intermediate tile)
///  (2) 2D-flat: tile.slice at the right row offset (operand already flattened)
///  (3) Rank>2 fallback: tile.slice + tile.reshape to 2D
///
/// Appends tile.transpose if the operand should be transposed.
BatchPageResult ExtractBatchPage(const BatchOperandInfo& info, const std::vector<int64_t>& operand_dims,
                                 const std::vector<int64_t>& operand_batch_shape, int64_t batch_index,
                                 const std::string& base_name, const AssignDefMap& def_map,
                                 const FlattenContext& ctx, const OpRegistry& op_registry, const Span& span) {
  BatchPageResult page;
  const auto& operand = info.operand;
  const auto& operand_type = info.operand_type;

  int64_t source_rows = operand_dims[operand_dims.size() - 2];
  int64_t source_cols = operand_dims.back();
  std::string suffix = std::to_string(batch_index);

  // Check if original operand was produced by a tile.load with Mat target_memory.
  auto original_var = As<Var>(info.original_operand);
  auto def_it = original_var ? def_map.find(original_var.get()) : def_map.end();
  auto original_assign = (def_it != def_map.end()) ? def_it->second : nullptr;
  auto original_load_call = original_assign ? As<Call>(original_assign->value_) : nullptr;
  bool is_mat_load = original_load_call && original_load_call->op_->name_ == "tile.load" &&
                     original_load_call->args_.size() >= 4;
  auto original_load_offsets = is_mat_load ? As<MakeTuple>(original_load_call->args_[1]) : nullptr;
  auto original_load_input = is_mat_load ? Substitute(original_load_call->args_[0], ctx.var_map) : nullptr;
  auto original_load_input_type =
      original_load_input ? As<TensorType>(original_load_input->GetType()) : nullptr;
  auto original_target_memory =
      is_mat_load ? original_load_call->GetKwarg<MemorySpace>("target_memory") : MemorySpace::DDR;
  bool original_load_transpose = is_mat_load ? original_load_call->GetKwarg<bool>("transpose", false) : false;

  VarPtr current;

  // Track whether Strategy 1 is used so we can skip redundant tile.transpose.
  bool used_strategy1 = false;

  if (is_mat_load && original_load_offsets && original_load_input_type &&
      original_target_memory == MemorySpace::Mat) {
    // Strategy 1: Re-emit tile.load with batch-adjusted offsets.
    // transpose=True is preserved in the kwargs, so per-batch loads inherit it.
    auto batch_indices = BuildBatchIndices(batch_index, operand_batch_shape);
    auto load_offset_elems = BuildBatchAdjustedOffsets(original_load_offsets->elements_, batch_indices,
                                                       operand_batch_shape.size(), span);

    std::vector<int64_t> load_shape_values(operand_batch_shape.size(), 1);
    load_shape_values.push_back(source_rows);
    load_shape_values.push_back(source_cols);

    // Keep the original tensor-rank offsets and shapes on the load call so
    // codegen still sees the original source window. Type inference will still
    // produce a rank>2 TileType from these shapes; this pass immediately
    // overrides the call result type to 2D below because hardware tiles are
    // always 2D.
    // For transposed loads (transpose=True), the original tensor-rank offsets
    // are required so that codegen routes the load through the standard DN
    // path. A flat 2D view cannot represent within-batch column-major
    // addressing.

    auto batch_load_offsets = std::make_shared<MakeTuple>(load_offset_elems, span);
    auto batch_load_shape = MakeShapeTupleFromInts(load_shape_values, span);
    std::vector<ExprPtr> batch_load_args = {original_load_input, batch_load_offsets, batch_load_shape,
                                            batch_load_shape};
    auto batch_load = op_registry.Create("tile.load", batch_load_args, original_load_call->kwargs_, span);

    // DeduceTileLoadType produces a rank>2 TileType from these shapes, but
    // hardware tiles are always 2D. Manually override the result type to 2D.
    int64_t flat_rows = original_load_transpose ? source_cols : source_rows;
    int64_t flat_cols = original_load_transpose ? source_rows : source_cols;
    auto flat_shape_exprs = Make2DShapeExprs(flat_rows, flat_cols, span);
    auto batch_load_type = As<TileType>(batch_load->GetType());
    std::optional<TileView> flat_tile_view;
    if (batch_load_type && batch_load_type->tile_view_.has_value()) {
      const auto& orig_tv = *batch_load_type->tile_view_;
      flat_tile_view = TileView(flat_shape_exprs, /*stride=*/{}, /*start_offset=*/nullptr, orig_tv.blayout,
                                orig_tv.slayout, orig_tv.fractal, orig_tv.pad);
    }
    auto flat_type = std::make_shared<TileType>(
        flat_shape_exprs, batch_load_type ? batch_load_type->dtype_ : operand_type->dtype_, std::nullopt,
        flat_tile_view, batch_load_type ? batch_load_type->memory_space_ : operand_type->memory_space_);
    auto flat_load = std::make_shared<Call>(batch_load->op_, batch_load_args, batch_load->kwargs_, flat_type,
                                            batch_load->span_);
    current = std::make_shared<Var>(base_name + "_load_" + suffix, flat_type, span);
    page.stmts.push_back(std::make_shared<AssignStmt>(current, flat_load, span));
    used_strategy1 = true;

  } else if (operand_type->shape_.size() == 2) {
    // Strategy 2: Slice from already-flattened 2D tile.
    auto offset = MakeShapeTupleFromInts({batch_index * source_rows, 0}, span);
    auto shape = MakeShapeTupleFromInts({source_rows, source_cols}, span);
    auto slice = op_registry.Create("tile.slice", {operand, shape, offset}, span);
    current = std::make_shared<Var>(base_name + "_slice_" + suffix, slice->GetType(), span);
    page.stmts.push_back(std::make_shared<AssignStmt>(current, slice, span));

  } else {
    // Strategy 3: rank>2 tile.slice + tile.reshape to 2D.
    auto batch_indices = BuildBatchIndices(batch_index, operand_batch_shape);
    std::vector<int64_t> offset_values = batch_indices;
    offset_values.push_back(0);
    offset_values.push_back(0);

    std::vector<int64_t> slice_shape_values(operand_batch_shape.size(), 1);
    slice_shape_values.push_back(source_rows);
    slice_shape_values.push_back(source_cols);

    auto offset = MakeShapeTupleFromInts(offset_values, span);
    auto slice_shape = MakeShapeTupleFromInts(slice_shape_values, span);
    auto slice = op_registry.Create("tile.slice", {operand, slice_shape, offset}, span);
    auto slice_var = std::make_shared<Var>(base_name + "_nd_slice_" + suffix, slice->GetType(), span);
    page.stmts.push_back(std::make_shared<AssignStmt>(slice_var, slice, span));

    auto reshape_shape = std::make_shared<MakeTuple>(Make2DShapeExprs(source_rows, source_cols, span), span);
    auto reshape = op_registry.Create("tile.reshape", {slice_var, reshape_shape}, span);
    current = std::make_shared<Var>(base_name + "_2d_" + suffix, reshape->GetType(), span);
    page.stmts.push_back(std::make_shared<AssignStmt>(current, reshape, span));
  }

  // Optionally transpose the 2D page.
  // Skip if Strategy 1 was used AND the original load has transpose=True: the re-emitted
  // tile.load already carries transpose=True in its kwargs, so an extra tile.transpose
  // would double-transpose. When Strategy 1 is used but the load doesn't have transpose,
  // we still need the explicit tile.transpose (old tile.transpose-wrapper pattern).
  if (info.transpose && !(used_strategy1 && original_load_transpose)) {
    auto axis0 = std::make_shared<ConstInt>(0, DataType::INDEX, span);
    auto axis1 = std::make_shared<ConstInt>(1, DataType::INDEX, span);
    auto transpose_call = op_registry.Create("tile.transpose", {current, axis0, axis1}, span);
    auto transpose_var = std::make_shared<Var>(base_name + "_t_" + suffix, transpose_call->GetType(), span);
    page.stmts.push_back(std::make_shared<AssignStmt>(transpose_var, transpose_call, span));
    current = transpose_var;
  }

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
  if (!store_call || store_call->op_->name_ != "tile.store") return info;

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
  auto orig_result_type = As<TileType>(call->GetType());
  CHECK(orig_result_type) << "FlattenTileNdTo2D: tile.batch_matmul expects TileType result";

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

  // Compute effective matrix dimensions (after transpose).
  int64_t lhs_source_rows = lhs_dims[lhs_dims.size() - 2];
  int64_t lhs_source_cols = lhs_dims.back();
  int64_t rhs_source_rows = rhs_dims[rhs_dims.size() - 2];
  int64_t rhs_source_cols = rhs_dims.back();

  int64_t lhs_rows = lhs_info.transpose ? lhs_source_cols : lhs_source_rows;
  int64_t lhs_cols = lhs_info.transpose ? lhs_source_rows : lhs_source_cols;
  int64_t rhs_rows = rhs_info.transpose ? rhs_source_cols : rhs_source_rows;
  int64_t rhs_cols = rhs_info.transpose ? rhs_source_rows : rhs_source_cols;

  CHECK(lhs_cols == rhs_rows)
      << "FlattenTileNdTo2D: tile.batch_matmul requires matching inner dimensions after "
         "transpose, but got "
      << lhs_cols << " and " << rhs_rows;

  // Detect direct-store fusion opportunity.
  auto direct_store = DetectDirectStore(stmts, stmt_index, assign->var_);

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

    auto lhs_page = ExtractBatchPage(lhs_info, lhs_dims, lhs_batch_dims, lhs_batch_idx, "lhs", def_map, ctx,
                                     op_registry, span);
    auto rhs_page = ExtractBatchPage(rhs_info, rhs_dims, rhs_batch_dims, rhs_batch_idx, "rhs", def_map, ctx,
                                     op_registry, span);
    auto matmul = op_registry.Create("tile.matmul", {lhs_page.var, rhs_page.var}, span);
    auto matmul_type = As<TileType>(matmul->GetType());
    bool needs_cast = matmul_type && matmul_type->dtype_ != orig_result_type->dtype_;
    if (!needs_cast) {
      out.stmts.insert(out.stmts.end(), lhs_page.stmts.begin(), lhs_page.stmts.end());
      out.stmts.insert(out.stmts.end(), rhs_page.stmts.begin(), rhs_page.stmts.end());
      auto matmul_var = std::make_shared<Var>(assign->var_->name_hint_, matmul->GetType(), span);
      out.stmts.push_back(std::make_shared<AssignStmt>(matmul_var, matmul, span));
      out.output_var = matmul_var;
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
    auto store_tensor_type = As<TensorType>(current_store_tensor->GetType());
    CHECK(store_tensor_type) << "FlattenTileNdTo2D: tile.batch_matmul direct store target must be TensorType";
    CHECK(direct_store_offsets) << "FlattenTileNdTo2D: tile.store offsets must be a MakeTuple";
    CHECK(direct_store_offsets->elements_.size() == output_batch_dims.size() + 2)
        << "FlattenTileNdTo2D: tile.store offsets rank must match batch_matmul result rank";
    if (store_tensor_type->shape_.size() > 2) {
      // Build the original tensor-rank partition shape:
      // [1, ..., 1, M, N] (left-padded with 1s for batch dims)
      const size_t tensor_rank = store_tensor_type->shape_.size();
      const size_t tile_rank = 2;  // matmul result is always 2D
      direct_store_shape.reserve(tensor_rank);
      for (size_t i = tile_rank; i < tensor_rank; ++i) {
        direct_store_shape.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, span));
      }
      direct_store_shape.push_back(std::make_shared<ConstInt>(lhs_rows, DataType::INDEX, span));
      direct_store_shape.push_back(std::make_shared<ConstInt>(rhs_cols, DataType::INDEX, span));
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
    auto lhs_page = ExtractBatchPage(lhs_info, lhs_dims, lhs_batch_dims, lhs_batch_idx, "lhs", def_map, ctx,
                                     op_registry, span);
    auto rhs_page = ExtractBatchPage(rhs_info, rhs_dims, rhs_batch_dims, rhs_batch_idx, "rhs", def_map, ctx,
                                     op_registry, span);
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
        store_args.push_back(std::make_shared<MakeTuple>(direct_store_shape, span));
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
  (void)assign;
  BatchMatmulAccResult out;
  auto def_map = BuildAssignDefMap(stmts);

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

  // Normalize lhs/rhs operands (peel transpose, recognize tile.load(transpose=True)).
  auto lhs_info = NormalizeBatchMatmulOperand(call->args_[1], "lhs", def_map, ctx);
  auto rhs_info = NormalizeBatchMatmulOperand(call->args_[2], "rhs", def_map, ctx);

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

  int64_t lhs_source_rows = lhs_dims[lhs_dims.size() - 2];
  int64_t lhs_source_cols = lhs_dims.back();
  int64_t rhs_source_rows = rhs_dims[rhs_dims.size() - 2];
  int64_t rhs_source_cols = rhs_dims.back();

  int64_t lhs_rows = lhs_info.transpose ? lhs_source_cols : lhs_source_rows;
  int64_t lhs_cols = lhs_info.transpose ? lhs_source_rows : lhs_source_cols;
  int64_t rhs_rows = rhs_info.transpose ? rhs_source_cols : rhs_source_rows;
  int64_t rhs_cols = rhs_info.transpose ? rhs_source_rows : rhs_source_cols;

  CHECK(lhs_cols == rhs_rows)
      << "FlattenTileNdTo2D: tile.batch_matmul_acc requires matching K after transpose, got " << lhs_cols
      << " and " << rhs_rows;

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

  // tile.matmul_acc requires its acc input to be in MemorySpace::Acc. The
  // upstream tile.batch_matmul lowering moves its result to Vec for the
  // tile.assemble path, so the acc operand here may arrive as Vec. Insert an
  // explicit tile.move(acc, target_memory=Acc) when the memory space is
  // resolved and not already Acc; remember the original space so we can move
  // the matmul_acc result back to it at the end (preserves the iter_arg /
  // consumer memory-space contract — e.g. when the accumulator is a loop
  // iter_arg initialised from a Vec tile, the next iteration must see Vec
  // again).
  VarPtr current_acc = acc_var;
  std::optional<MemorySpace> original_acc_memory;
  if (acc_type->memory_space_.has_value() && *acc_type->memory_space_ != MemorySpace::Acc) {
    original_acc_memory = acc_type->memory_space_;
    std::vector<std::pair<std::string, std::any>> move_kw = {{"target_memory", MemorySpace::Acc}};
    auto move = op_registry.Create("tile.move", {current_acc}, move_kw, span);
    auto moved = std::make_shared<Var>(current_acc->name_hint_ + "_acc", move->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(moved, move, span));
    current_acc = moved;
  }

  // Move the final accumulator back to its original memory space (typically
  // Vec when the acc came from a tile.assemble produced by LowerBatchMatmul).
  // No-op when original_acc_memory is unset (acc was already in Acc) or when
  // the value already lives in the right space.
  auto move_back_if_needed = [&](VarPtr value) -> VarPtr {
    if (!original_acc_memory.has_value()) return value;
    auto value_type = As<TileType>(value->GetType());
    if (value_type && value_type->memory_space_.has_value() &&
        *value_type->memory_space_ == *original_acc_memory) {
      return value;
    }
    std::vector<std::pair<std::string, std::any>> move_kw = {
        {"target_memory", *original_acc_memory},
    };
    auto move = op_registry.Create("tile.move", {value}, move_kw, span);
    auto restored = std::make_shared<Var>(value->name_hint_ + "_back", move->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(restored, move, span));
    return restored;
  };

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

    auto lhs_page = ExtractBatchPage(lhs_info, lhs_dims, lhs_batch_dims, lhs_batch_idx, "lhs", def_map, ctx,
                                     op_registry, span);
    auto rhs_page = ExtractBatchPage(rhs_info, rhs_dims, rhs_batch_dims, rhs_batch_idx, "rhs", def_map, ctx,
                                     op_registry, span);
    out.stmts.insert(out.stmts.end(), lhs_page.stmts.begin(), lhs_page.stmts.end());
    out.stmts.insert(out.stmts.end(), rhs_page.stmts.begin(), rhs_page.stmts.end());

    auto matmul_acc = op_registry.Create("tile.matmul_acc", {current_acc, lhs_page.var, rhs_page.var}, span);
    auto new_acc = std::make_shared<Var>(current_acc->name_hint_, matmul_acc->GetType(), span);
    out.stmts.push_back(std::make_shared<AssignStmt>(new_acc, matmul_acc, span));
    out.output_var = move_back_if_needed(new_acc);
    return out;
  }

  // General path: unroll batch dims using slice + tile.matmul_acc + assemble on acc.
  for (int64_t i = 0; i < batch_count; ++i) {
    auto output_batch_indices = BuildBatchIndices(i, output_batch_dims);
    int64_t lhs_batch_idx =
        BuildOperandFlatBatchIndex(lhs_batch_dims, output_batch_dims, output_batch_indices);
    int64_t rhs_batch_idx =
        BuildOperandFlatBatchIndex(rhs_batch_dims, output_batch_dims, output_batch_indices);

    auto lhs_page = ExtractBatchPage(lhs_info, lhs_dims, lhs_batch_dims, lhs_batch_idx, "lhs", def_map, ctx,
                                     op_registry, span);
    auto rhs_page = ExtractBatchPage(rhs_info, rhs_dims, rhs_batch_dims, rhs_batch_idx, "rhs", def_map, ctx,
                                     op_registry, span);
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

  out.output_var = move_back_if_needed(current_acc);
  return out;
}

/**
 * @brief Recursively transform statements, flattening >2D tile ops to 2D.
 */
std::vector<StmtPtr> TransformBody(const std::vector<StmtPtr>& stmts, FlattenContext& ctx,
                                   const OpRegistry& op_registry, const Span& span) {
  std::vector<StmtPtr> result;

  // Pre-scan: identify tile.load/tile.transpose results consumed exclusively by
  // tile.batch_matmul. When ExtractBatchPage Strategy 1 re-emits per-batch loads
  // from the original tensor, the full-batch load becomes dead code. Skip emitting
  // it to avoid wasted memory and potential hardware pipeline interference.
  //
  // Safety: we count ALL Var references across every statement type (Return, Yield,
  // If conditions, For/While bounds, etc.), not just Call arguments. A Var used
  // anywhere outside a tile.batch_matmul Call prevents it from being skipped.
  std::unordered_set<const Var*> batch_matmul_only_vars;
  {
    std::unordered_map<const Var*, int> use_count;
    std::vector<const Var*> batch_matmul_operands;  // ordered to avoid nondeterministic iteration

    // Helper: recursively count all Var references within an expression.
    std::function<void(const ExprPtr&)> CountVarRefs = [&](const ExprPtr& expr) {
      if (!expr) return;
      if (auto v = As<Var>(expr)) {
        use_count[v.get()]++;
        return;
      }
      if (auto tup = As<MakeTuple>(expr)) {
        for (const auto& e : tup->elements_) CountVarRefs(e);
        return;
      }
      if (auto call = As<Call>(expr)) {
        for (const auto& a : call->args_) CountVarRefs(a);
        return;
      }
    };

    for (const auto& s : stmts) {
      // AssignStmt: count call args; mark batch_matmul[_acc] operands separately.
      // For tile.batch_matmul_acc, only lhs (arg 1) and rhs (arg 2) are eligible
      // for Strategy 1 skip-load — the acc operand (arg 0) is in Acc memory and
      // never goes through tile.load(target_memory=Mat).
      if (auto a = As<AssignStmt>(s)) {
        if (auto c = As<Call>(a->value_)) {
          const std::string& cname = c->op_->name_;
          bool is_batch_mm = (cname == "tile.batch_matmul");
          bool is_batch_mm_acc = (cname == "tile.batch_matmul_acc");
          for (size_t arg_i = 0; arg_i < c->args_.size(); ++arg_i) {
            const auto& arg = c->args_[arg_i];
            if (auto v = As<Var>(arg)) {
              use_count[v.get()]++;
              const bool eligible = is_batch_mm || (is_batch_mm_acc && (arg_i == 1 || arg_i == 2));
              if (eligible) {
                batch_matmul_operands.push_back(v.get());
              }
            }
          }
        } else {
          // Non-call assignment (e.g. plain Var alias): count all Var refs.
          CountVarRefs(a->value_);
        }
        continue;
      }
      // ReturnStmt / YieldStmt: count all returned/yielded Var refs.
      if (auto ret = As<ReturnStmt>(s)) {
        for (const auto& v : ret->value_) CountVarRefs(v);
        continue;
      }
      if (auto yield = As<YieldStmt>(s)) {
        for (const auto& v : yield->value_) CountVarRefs(v);
        continue;
      }
      // EvalStmt: count Var refs in the expression.
      if (auto eval = As<EvalStmt>(s)) {
        CountVarRefs(eval->expr_);
        continue;
      }
      // IfStmt: count condition Var refs.
      if (auto if_stmt = As<IfStmt>(s)) {
        CountVarRefs(if_stmt->condition_);
        continue;
      }
      // ForStmt: count start/stop/step and iter_arg init Var refs.
      if (auto for_stmt = As<ForStmt>(s)) {
        CountVarRefs(for_stmt->start_);
        CountVarRefs(for_stmt->stop_);
        CountVarRefs(for_stmt->step_);
        for (const auto& ia : for_stmt->iter_args_) CountVarRefs(ia->initValue_);
        continue;
      }
      // WhileStmt: count condition and iter_arg init Var refs.
      if (auto while_stmt = As<WhileStmt>(s)) {
        CountVarRefs(while_stmt->condition_);
        for (const auto& ia : while_stmt->iter_args_) CountVarRefs(ia->initValue_);
        continue;
      }
    }
    // De-duplicate batch_matmul_operands before checking counts.
    std::unordered_set<const Var*> seen;
    for (const auto* v : batch_matmul_operands) {
      if (seen.insert(v).second && use_count[v] == 1) {
        batch_matmul_only_vars.insert(v);
      }
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
      } else if (auto auto_in_core = As<AutoInCoreScopeStmt>(stmt)) {
        result.push_back(rewrite(auto_in_core));
      } else if (auto cluster = As<ClusterScopeStmt>(stmt)) {
        result.push_back(rewrite(cluster));
      } else if (auto hier = As<HierarchyScopeStmt>(stmt)) {
        result.push_back(rewrite(hier));
      } else if (auto spmd = As<SpmdScopeStmt>(stmt)) {
        result.push_back(rewrite(spmd));
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

    // ---- tile.load on >2D tile: preserve the original tensor-rank source window,
    //      but flatten the result tile ----
    // Keep the original tensor-rank offsets/shapes on the call for codegen, then
    // manually replace the inferred rank>2 TileType with a 2D TileType because
    // hardware tiles are always 2D.
    if (op_name == "tile.load") {
      // Skip tile.load whose result is consumed exclusively by tile.batch_matmul,
      // but ONLY when ExtractBatchPage Strategy 1 can reconstruct per-batch loads
      // (i.e. target_memory == Mat). Strategy 1 now supports transpose=True loads
      // by propagating the kwarg to per-batch loads. Other strategies (2/3) need
      // the original load result as their input operand.
      if (batch_matmul_only_vars.count(assign->var_.get())) {
        auto target_mem = call->GetKwarg<MemorySpace>("target_memory", MemorySpace::DDR);
        if (target_mem == MemorySpace::Mat) {
          ctx.Insert(assign->var_, assign->var_);  // identity mapping so lookups still work
          continue;
        }
      }

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
        // Assign the implicit TileView for the flattened 2D shape+memory_space.
        auto flat_tile_view = std::make_optional(
            tile_view_semantics::GetImplicitTileView(flat_shape_exprs, result_tile->memory_space_));
        auto flat_tile_type = std::make_shared<TileType>(flat_shape_exprs, result_tile->dtype_, std::nullopt,
                                                         flat_tile_view, result_tile->memory_space_);
        auto flat_call =
            std::make_shared<Call>(call->op_, sub_args, call->kwargs_, flat_tile_type, call->span_);
        auto flat_var = std::make_shared<Var>(assign->var_->name_hint_, flat_tile_type, assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(flat_var, flat_call, assign->span_));
        ctx.Insert(assign->var_, flat_var);
        continue;
      }
      // ≤2D tile.load: honor any pending var_map substitutions
      auto new_call = op_registry.Create(op_name, sub_args, call->kwargs_, span);
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
    if (op_name == "tile.store") {
      auto orig_tile_type = As<TileType>(call->args_[0]->GetType());

      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size() + 1);
      // Push all original args (tile, offsets, output_tensor) with substitution
      for (const auto& arg : call->args_) {
        new_args.push_back(Substitute(arg, ctx.var_map));
      }

      auto out_tensor_type = As<TensorType>(new_args[2]->GetType());
      if (orig_tile_type && out_tensor_type && out_tensor_type->shape_.size() > 2) {
        // Inject the original tensor-rank partition shape tuple as the 4th argument.
        // The partition shape has the same rank as the tensor, with 1s for
        // batch dims that are not covered by the tile, followed by the tile dims.
        const size_t tensor_rank = out_tensor_type->shape_.size();
        const size_t tile_rank = orig_tile_type->shape_.size();
        std::vector<ExprPtr> partition_shape;
        partition_shape.reserve(tensor_rank);
        for (size_t i = tile_rank; i < tensor_rank; ++i) {
          partition_shape.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, span));
        }
        for (const auto& dim : orig_tile_type->shape_) {
          partition_shape.push_back(dim);
        }
        new_args.push_back(std::make_shared<MakeTuple>(partition_shape, span));
      }

      // Construct call directly: store result type = output tensor type (args[2])
      auto out_type = new_args[2]->GetType();
      auto new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, out_type, call->span_);
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
      result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
      ctx.Insert(assign->var_, new_var);
      continue;
    }

    // ---- tile.create / tile.full with >2D shape: flatten shape directly ----
    if (op_name == "tile.create" || op_name == "tile.full") {
      auto result_tile = As<TileType>(call->GetType());
      if (result_tile && result_tile->shape_.size() > 2) {
        auto [merged, last] = ComputeMergedShape(result_tile->shape_, op_name);

        // Rebuild the call with 2D shape
        auto new_shape_tuple = MakeShapeTupleFromInts({merged, last}, span);
        std::vector<ExprPtr> new_args;
        // First arg is the shape tuple
        new_args.push_back(new_shape_tuple);
        // Remaining args (e.g., fill value for tile.full)
        for (size_t i = 1; i < call->args_.size(); ++i) {
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
    if (op_name == "tile.sum" || op_name == "tile.max" || op_name == "tile.min") {
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
    if (op_name == "tile.batch_matmul") {
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
    if (op_name == "tile.batch_matmul_acc") {
      auto lowering = LowerBatchMatmulAcc(assign, call, stmts, ctx, op_registry, span);
      result.insert(result.end(), lowering.stmts.begin(), lowering.stmts.end());
      ctx.Insert(assign->var_, lowering.output_var);
      continue;
    }

    // ---- tile.transpose feeding only tile.batch_matmul[_acc]: skip and let lowering peel it ----
    if (op_name == "tile.transpose" && batch_matmul_only_vars.count(assign->var_.get()) != 0) {
      ctx.Insert(assign->var_, assign->var_);  // identity mapping for safety
      continue;
    }

    // ---- All other tile ops (including tile.reshape) and non-tile ops: substitute args ----
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
 * @brief Visitor that checks all tile ops in InCore functions use ≤2D tiles.
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

    // tile.load/tile.store are permitted to have any tile rank:
    // load produces 2D tiles from rank>2 tensors; store accepts 2D tiles and
    // writes them back to rank>2 tensors.
    if (name == "tile.load" || name == "tile.store" || name == "tile.reshape") return;

    // Check result type
    auto result_tile = As<TileType>(call->GetType());
    if (result_tile && result_tile->shape_.size() > 2) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileOps2D", 0,
                                "Tile op '" + name + "' in InCore function '" + func_name_ +
                                    "' produces >2D tile (should have been flattened to 2D)",
                                stmt_span);
    }

    // Check argument types
    for (const auto& arg : call->args_) {
      auto arg_tile = As<TileType>(arg->GetType());
      if (arg_tile && arg_tile->shape_.size() > 2) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileOps2D", 0,
                                  "Tile op '" + name + "' in InCore function '" + func_name_ +
                                      "' has >2D tile argument (should have been flattened to 2D)",
                                  stmt_span);
        break;
      }
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

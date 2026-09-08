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

#ifndef SRC_IR_TRANSFORMS_FLATTEN_TILE_ND_TO_2D_REWRITE_INTERNAL_H_
#define SRC_IR_TRANSFORMS_FLATTEN_TILE_ND_TO_2D_REWRITE_INTERNAL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace flatten_tile_nd_to_2d {
namespace rewrite_internal {

/// Geometry of one batched accumulator that this pass packs along COLUMNS.
///
/// The pages of a batched `Acc` (L0C) accumulator cannot be stacked along rows:
/// L0C is NZ-boxed, so box `(r_b, c_b)` of an `[M, N]` tile begins at
/// `(c_b * M/16 + r_b) * 1024` bytes, and a row window of a multi-block-column
/// tile is therefore *strided*. pto-isa's MAD writes its destination compactly
/// from a bare pointer and carries no destination stride
/// (hw-native-sys/pto-isa#253), so such a window silently miscomputes -- which is
/// why `CanonicalizeTileSlice` rejects it outright.
///
/// A COLUMN window spans the parent's full row extent, so the window's compact
/// geometry and the parent's coincide and the discarded stride does not matter.
/// The whole accumulator therefore becomes one `[rows, batch_count * cols]` Acc
/// tile with page `b` at `[0, b * cols]`.
struct AccPackingPlan {
  int64_t batch_count = 1;          ///< B -- number of pages packed side by side.
  int64_t rows = 0;                 ///< M -- page (and packed-tile) row extent.
  int64_t cols = 0;                 ///< N -- page column extent.
  DataType dtype = DataType::FP32;  ///< 4-byte accumulator element type.
  std::vector<int64_t> batch_dims;  ///< The ND batch dims, for per-page drain offsets.
  std::vector<int64_t> nd_shape;    ///< The full pre-flatten ND accumulator shape.
  bool row_windows = false;         ///< Logical 2-D row windows, drained back along rows.
};

/// Which (pre-rewrite) Vars name a column-packed accumulator, and with what
/// geometry. Immutable once built; `FlattenContext` holds it behind a
/// `shared_ptr` because the context is copied once per nested block.
class AccPackingMap {
 public:
  size_t AddPlan(AccPackingPlan plan) {
    plans_.push_back(std::move(plan));
    return plans_.size() - 1;
  }
  void Bind(const Var* var, size_t plan_index) { members_.emplace(var, plan_index); }

  [[nodiscard]] const AccPackingPlan* Lookup(const Var* var) const {
    if (var == nullptr) return nullptr;
    auto it = members_.find(var);
    return it == members_.end() ? nullptr : &plans_[it->second];
  }
  [[nodiscard]] bool Empty() const { return members_.empty(); }

 private:
  std::vector<AccPackingPlan> plans_;
  std::unordered_map<const Var*, size_t> members_;
};

using AccPackingMapPtr = std::shared_ptr<const AccPackingMap>;

/// Whole-function analysis: decide which batched accumulator chains are packed
/// along columns, and reject the ones that can neither be column-packed nor left
/// on the legacy row-packed path. Runs once, before any rewriting, because a
/// chain routinely spans blocks (create outside the K loop, `batch_matmul_acc`
/// inside it, store after it) and the rewrite loop's own pre-scan is per-block.
AccPackingMapPtr BuildAccPackingMap(const FunctionPtr& func);

struct FlattenContext {
  std::unordered_map<const Var*, VarPtr> var_map;
  /// Shared, immutable column-packing decision (see `BuildAccPackingMap`). Held
  /// by pointer so the per-block context copies stay O(1).
  AccPackingMapPtr acc_packing;

  void Insert(const VarPtr& old_var, const VarPtr& new_var) { var_map[old_var.get()] = new_var; }
  void Erase(const VarPtr& var) { var_map.erase(var.get()); }

  /// The packing plan for a PRE-substitution operand, or null when the operand is
  /// not a column-packed accumulator. Keyed on the original Var because that is
  /// what `BuildAccPackingMap` walked.
  [[nodiscard]] const AccPackingPlan* AccPackingFor(const ExprPtr& original_operand) const {
    if (!acc_packing || !original_operand) return nullptr;
    auto var = AsVarLike(original_operand);
    return var ? acc_packing->Lookup(var.get()) : nullptr;
  }
  [[nodiscard]] const AccPackingPlan* AccPackingForVar(const VarPtr& original_var) const {
    return acc_packing ? acc_packing->Lookup(original_var.get()) : nullptr;
  }
};

bool IsNdTile(const TileTypePtr& tile_type);
int64_t GetStaticDim(const ExprPtr& expr, const std::string& context);
std::pair<int64_t, int64_t> ComputeMergedShape(const std::vector<ExprPtr>& shape, const std::string& context);
ExprPtr MakeShapeTupleFromInts(const std::vector<int64_t>& dims, const Span& span);
std::vector<ExprPtr> ComputeStorePartitionShape(const std::vector<ExprPtr>& tile_shape,
                                                const std::vector<ExprPtr>& tensor_shape,
                                                const std::vector<ExprPtr>& offsets, const Span& span);
std::vector<ExprPtr> Make2DShapeExprs(int64_t merged, int64_t last, const Span& span);
std::vector<ExprPtr> ComputeMergedValidShape(const std::vector<ExprPtr>& valid, const Span& span);
ExprPtr MakeCanonicalIndexAdd(const ExprPtr& lhs, const ExprPtr& rhs, const Span& span);
std::vector<ExprPtr> CollapseLeadingDimsTo2D(const std::vector<ExprPtr>& dims, const Span& span);
CallPtr CreateCollapsedTensorView(const ExprPtr& tensor, const TensorTypePtr& tensor_type, const Span& span);
ExprPtr CollapseLeadingOffsetsToRow(const std::vector<ExprPtr>& offsets,
                                    const std::vector<ExprPtr>& tensor_shape, const Span& span);
bool BatchOperandsWholeFit(const TileTypePtr& lhs_type, const TileTypePtr& rhs_type);
std::vector<int64_t> ToStaticDims(const std::vector<ExprPtr>& shape, const std::string& context);
int64_t MultiplyStaticDims(const std::vector<int64_t>& dims, const std::string& context);
std::vector<int64_t> BuildBatchIndices(int64_t flat_index, const std::vector<int64_t>& batch_shape);
std::vector<ExprPtr> BuildBatchAdjustedOffsets(const std::vector<ExprPtr>& base_offset_elems,
                                               const std::vector<int64_t>& batch_indices, size_t batch_rank,
                                               const Span& span);
int64_t BuildOperandFlatBatchIndex(const std::vector<int64_t>& operand_batch_shape,
                                   const std::vector<int64_t>& output_batch_shape,
                                   const std::vector<int64_t>& output_batch_indices);
int64_t NormalizeAxisIndex(int64_t axis, size_t ndim, const std::string& context);
bool IsTrailingMatrixAxisSwap(int64_t axis1, int64_t axis2, size_t ndim);
std::vector<TypePtr> FindYieldTypes(const std::vector<StmtPtr>& stmts);

using AssignDefMap = std::unordered_map<const Var*, AssignStmtPtr>;

AssignDefMap BuildAssignDefMap(const std::vector<StmtPtr>& stmts);
bool IsSafePeelableBatchMatmulReshape(const CallPtr& reshape_call);
bool KeepOperandWhole(bool capacity_fits, const CallPtr& base_load);
CallPtr TraceOperandBaseLoad(const ExprPtr& operand_expr, const AssignDefMap& def_map);

struct BatchMatmulResult {
  std::vector<StmtPtr> stmts;
  VarPtr output_var;
  bool fused_store = false;
  VarPtr store_result_var;
  VarPtr store_orig_var;
};

BatchMatmulResult LowerBatchMatmul(const AssignStmtPtr& assign, const CallPtr& call,
                                   const std::vector<StmtPtr>& stmts, size_t stmt_index,
                                   const FlattenContext& ctx, const OpRegistry& op_registry,
                                   const Span& span);

struct BatchMatmulAccResult {
  std::vector<StmtPtr> stmts;
  VarPtr output_var;
};

BatchMatmulAccResult LowerBatchMatmulAcc(const AssignStmtPtr& assign, const CallPtr& call,
                                         const std::vector<StmtPtr>& stmts, const FlattenContext& ctx,
                                         const OpRegistry& op_registry, const Span& span);

struct NdTransposeResult {
  std::vector<StmtPtr> stmts;
  VarPtr output_var;
};

NdTransposeResult LowerNdTranspose(const AssignStmtPtr& assign, const CallPtr& call,
                                   const FlattenContext& ctx, const OpRegistry& op_registry,
                                   const Span& span);

}  // namespace rewrite_internal
}  // namespace flatten_tile_nd_to_2d
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_FLATTEN_TILE_ND_TO_2D_REWRITE_INTERNAL_H_

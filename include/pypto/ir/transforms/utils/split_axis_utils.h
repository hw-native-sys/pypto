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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_SPLIT_AXIS_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_SPLIT_AXIS_UTILS_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {
namespace split_axis {

/**
 * @brief Map a SplitMode to the tile dimension it partitions.
 *
 * ``SplitMode::UpDown`` halves the height (dimension 0); any other mode
 * (``LeftRight``) halves the width (dimension 1). 2D-only is already enforced
 * upstream (deducer + cross_core.cpp DeduceSplitReshape), so the binary 0/1
 * answer is sufficient.
 *
 * @param mode The split mode of the AIV/AIC function.
 * @return The partitioned tile dimension (0 for UpDown, 1 otherwise).
 */
int SplitDimension(SplitMode mode);

/**
 * @brief Detect a vector reduction that collapses the split axis.
 *
 * When an AIV lane holds only half of a tile (after the split), a reduction
 * over the split axis produces a partial result on each lane — a miscompile.
 * Recognizes the tile reduce ops:
 *   - ``tile.row_*`` (sum/max/min/prod) → reduces the last axis;
 *   - ``tile.col_*`` (sum/max/min/prod) → reduces axis 0.
 *
 * Returns ``true`` iff the reduced axis equals ``split_dim``. Non-reduce calls
 * (and Submits, which carry a GlobalVar callee and no ``op_``) return ``false``.
 *
 * @param call The call expression to inspect.
 * @param split_dim The dimension partitioned by the split (see SplitDimension).
 * @return ``true`` when the reduction collapses the split axis.
 */
bool IsReduceOnSplitAxis(const CallPtr& call, int split_dim);

/**
 * @brief Per-split-dim metadata tracked for a halved tile-producing var.
 *
 * Once a tile var has been partitioned along the split axis, downstream ops
 * (e.g. ``tile.store``, loop ``iter_args``/``return_vars``) need its halved
 * extent to re-localize their split-dim offsets. ``half_dim_size`` is that
 * extent (a ``ConstInt`` for static dims, a ``floordiv`` expression otherwise).
 */
struct TileInfo {
  ExprPtr half_dim_size;
  // The dimension this tile is currently split along. Usually the global split
  // dim, but a reshape can migrate the split axis to another dimension (e.g. the
  // rms_norm [N,1]<->[1,N] column reshape), so each tracked tile carries its own.
  int split_dim = 0;
};

/**
 * @brief Result of injecting the per-subblock index at the top of a body.
 *
 * For AIV functions, ``InjectSubblockIdx`` prepends an assignment binding a
 * fresh ``subblock_idx`` var to ``tile.get_subblock_idx()``; ``subblock_idx_expr``
 * references that var. For non-AIV functions it is null and no statement is
 * prepended. ``used_names`` is the seeded name set (params + def vars, plus the
 * freshly reserved subblock name) so callers can keep generating collision-free
 * names.
 */
struct SubblockInjectionResult {
  ExprPtr subblock_idx_expr;
  std::vector<StmtPtr> body_stmts;
  std::unordered_set<std::string> used_names;
};

/**
 * @brief Inject the per-subblock index binding at the top of a function body.
 *
 * @param func The AIV/AIC function whose body is being split.
 * @param is_aiv Whether the function is an AIV lane (only AIV gets the index).
 * @return The (possibly prepended) body statements plus the subblock-idx expr.
 */
SubblockInjectionResult InjectSubblockIdx(const FunctionPtr& func, bool is_aiv);

/**
 * @brief Inject the per-subblock index binding at the head of a region body.
 *
 * Region-scoped analogue of ``InjectSubblockIdx`` for the explicit
 * ``SplitAivScopeStmt`` consumer in LowerAutoVectorSplit (pass 21). Prepends a
 * fresh ``subblock_idx = tile.get_subblock_idx()`` binding to ``region_stmts``
 * (a region is always an AIV lane, so the index is always injected) and returns
 * the rewritten body plus the index expr. ``used_names`` seeds the collision-free
 * name set (caller-supplied names plus the region's own def vars are added) so the
 * injected name never clashes with an existing binding.
 *
 * @param region_stmts The flattened statements of the region body.
 * @param used_names Externally-reserved names to avoid colliding with.
 * @return The prepended body statements plus the subblock-idx expr.
 */
SubblockInjectionResult InjectSubblockIdxIntoStmts(const std::vector<StmtPtr>& region_stmts,
                                                   const std::unordered_set<std::string>& used_names);

/**
 * @brief A tile.transpose that swaps the split axis on a non-singleton source.
 *
 * ``call`` is the first offending transpose found (null when none). ``result_name``
 * is the name_hint of the assigned result (empty for an EvalStmt or anonymous LHS).
 */
struct TransposeSplitHazard {
  CallPtr call;
  std::string result_name;
};

/**
 * @brief Find the first split-axis-swapping tile.transpose within a body.
 *
 * Splitting halves the ``split_dim`` axis, but ``tile.transpose`` swaps axes — so
 * a transpose that moves the split axis migrates the per-lane data to the other
 * dimension and cannot be split correctly. A source that is statically singleton
 * on the split axis carries no split data and is safe. Shared by ExpandMixedKernel
 * (AUTO whole-function check) and LowerAutoVectorSplit (explicit per-region check);
 * each caller builds its own actionable diagnostic from the result.
 *
 * @param body The statement tree to scan.
 * @param split_dim The partitioned tile dimension (see SplitDimension).
 * @return The first offending transpose (``call == nullptr`` when none).
 */
TransposeSplitHazard FindTransposeSplitHazard(const StmtPtr& body, int split_dim);

/**
 * @brief Halve every split-axis tile along a statement list (recursive driver).
 *
 * Rewrites cross-core push/pop sync, halves AIV ``tile.load``/``tile.store``/
 * compute/``tile.slice``/``tile.reshape`` results along ``split_dim``, and
 * threads the per-var ``tile_vars`` tracking and ``var_replacements`` rebind map
 * through nested control flow. The maps are mutated in place so the caller can
 * apply the final ``Substitute`` over the rebuilt body.
 *
 * @param stmts The statements to process.
 * @param mode The split mode (UpDown / LeftRight).
 * @param split_int The integer split attribute stamped on cross-core ops.
 * @param split_dim The partitioned tile dimension (see SplitDimension).
 * @param tile_vars In/out map of split-tracked tile vars to their halved extent.
 * @param is_aiv Whether this is an AIV lane (gates per-op halving).
 * @param subblock_idx The per-subblock index expr (null for non-AIV).
 * @param var_replacements In/out map of original vars to their rebuilt versions.
 * @return The rewritten statement list.
 */
std::vector<StmtPtr> ProcessStmts(const std::vector<StmtPtr>& stmts, SplitMode mode, int split_int,
                                  int split_dim, std::unordered_map<const Var*, TileInfo>& tile_vars,
                                  bool is_aiv, const ExprPtr& subblock_idx,
                                  std::unordered_map<const Var*, VarPtr>& var_replacements);

}  // namespace split_axis
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SPLIT_AXIS_UTILS_H_

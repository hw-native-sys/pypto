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

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

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
 * @brief Half of a split-axis extent: the CEIL half.
 *
 * Both lanes get the same physical box, so an odd extent ``2k + 1`` gives each
 * of them ``k + 1`` cells and the per-lane valid extent carries the raggedness
 * (pto-isa's ``TILE_UP_DOWN_ODD``: "AIV0 = rows/2 + 1, AIV1 = rows/2"). An even
 * extent is unaffected. A dynamic extent has no compile-time parity and keeps
 * ``floordiv(dim, 2)``.
 *
 * @param dim_size The pre-split extent on the split axis.
 * @return The per-lane physical extent.
 */
ExprPtr ComputeHalfDimSize(const ExprPtr& dim_size);

/**
 * @brief The per-lane partition stride for an AUTO-split body, or null.
 *
 * The split partitions the split axis between the two AIV lanes: lane L owns
 * ``[L * S, L * S + S)`` of it, where ``S`` is this stride. Two partitions are
 * possible, and they differ only when the boundary tile is ragged:
 *
 * | Partition | ``S`` | Lane extents for a `[16, ...]` box valid to 13 |
 * | --------- | ----- | ---------------------------------------------- |
 * | box (default) | ``ceil(box / 2)`` = 8 | 8 and 5 |
 * | valid (this)  | ``ceil(V / 2)`` = 7   | 7 and 6 |
 *
 * The box partition is universal — it works for every tile in the region
 * whatever its valid extent — but pto-isa can only place lane 1's FIFO band at
 * its own extent (even codes) or one past it (odd codes), so lanes 8 and 5 have
 * no expressible transport. Balancing the VALID region instead makes the two
 * lanes differ by at most one by construction, which is exactly what the codes
 * express — and it splits the real work evenly rather than leaving lane 1 the
 * remainder.
 *
 * It is only sound when the whole body agrees on one logical row space, so this
 * returns null (keep the box partition) unless:
 *
 * - the body has at least one Cube -> Vector boundary and they all agree on the
 *   same static ``(box, valid)`` on the split axis;
 * - no Vector -> Cube boundary is present (the gather re-joins the lanes
 *   positionally at their own extents, which only abut when they are equal);
 * - every other split-axis tile in the body is derived from that boundary — a
 *   tile with an independent origin (a `tile.load`, a generator) spans the FULL
 *   box, which the balanced partition would not cover.
 *
 * @param stmts The pre-lowering body statements.
 * @param split_dim The partitioned tile dimension (see SplitDimension).
 * @return The balanced stride as a ConstInt, or null to keep the box partition.
 */
ExprPtr ResolveLaneStride(const std::vector<StmtPtr>& stmts, int split_dim);

/**
 * @brief The pto-isa split code a Cube -> Vector boundary op must carry.
 *
 * pto-isa derives lane 1's band inside the FIFO slot from the popped tile's own
 * RUNTIME valid extents, so the code has to state how the two lanes' extents
 * relate (a2a3/a5 ``TPush.hpp popVecTileFromGMFiFo``):
 *
 * | Code                    | lane 1's band starts at | requires   |
 * | ----------------------- | ----------------------- | ---------- |
 * | ``kSplitUpDown`` / LR    | ``e1 * pitch``          | ``e0 == e1`` |
 * | ``kSplitUpDownOdd`` / LR | ``(e1 + 1) * pitch``    | ``e0 == e1 + 1`` |
 *
 * where ``eL = clamp(V - L * half, 0, half)`` is lane L's extent, ``V`` the
 * boundary tile's valid split-axis extent and ``half = ceil(box / 2)`` — the
 * same clamp the halving materializes into the popped tile's valid_shape.
 *
 * That covers both odd shapes: an odd physical box (``V == box == 2k + 1`` gives
 * ``k + 1`` / ``k``) and an odd valid extent inside an even box (``V == box - 1``
 * gives ``half`` / ``half - 1``). An empty lane 1 (``e1 == 0``, i.e.
 * ``V <= half``) reads nothing wherever it is pointed, so it keeps the even
 * code. Any other ragged extent has no expressible band pair and is rejected
 * with an actionable message rather than silently popping the wrong rows.
 *
 * A dynamic (non-ConstInt) box or valid extent has no compile-time lane extents;
 * it keeps the even code, the only one whose lowering does not depend on the
 * parity.
 *
 * @param mode The split mode (``None`` yields ``kSplitNone``).
 * @param full_type The PRE-split (full-width) boundary tile type.
 * @param split_dim The partitioned tile dimension (see SplitDimension).
 * @param lane_stride The body's partition stride (see ResolveLaneStride); null
 *        for the default box partition, where ``S = ceil(box / 2)``.
 * @param op_name Op name for diagnostics.
 * @param span Span for diagnostics.
 * @return The split code to stamp on the boundary / tpush / tpop op.
 */
int ShardSplitCode(SplitMode mode, const TypePtr& full_type, int split_dim, const ExprPtr& lane_stride,
                   const std::string& op_name, const Span& span);

/**
 * @brief The pto-isa split code a Vector -> Cube boundary op must carry.
 *
 * The gather has no odd form — pto-isa's vector-side producer
 * (``pushVec2GMFiFo``) offsets lane 1 by lane 1's OWN extent, so the two bands
 * abut only when the lanes are equal. This returns the even code and rejects a
 * boundary whose lanes would differ.
 *
 * @param mode The split mode (``None`` yields ``kSplitNone``).
 * @param full_type The FULL (gathered) boundary tile type.
 * @param split_dim The partitioned tile dimension (see SplitDimension).
 * @param op_name Op name for diagnostics.
 * @param span Span for diagnostics.
 * @return The even split code for @p mode.
 */
int GatherSplitCode(SplitMode mode, const TypePtr& full_type, int split_dim, const std::string& op_name,
                    const Span& span);

/**
 * @brief The lane extents ``eL = clamp(V - L * S, 0, S)`` of a boundary tile.
 *
 * Exposed for the passes that need the pair itself (diagnostics, the transport
 * code choice); returns nullopt when either the box or the valid extent on the
 * split axis is not a compile-time constant.
 *
 * @param full_type The PRE-split (full-width) boundary tile type.
 * @param split_dim The partitioned tile dimension (see SplitDimension).
 * @param lane_stride The body's partition stride; null for the box partition.
 * @return ``{lane0, lane1}`` extents, or nullopt when not static.
 */
std::optional<std::pair<int64_t, int64_t>> StaticLaneExtents(const TypePtr& full_type, int split_dim,
                                                             const ExprPtr& lane_stride);

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
 * extent (a ``ConstInt`` for static dims, a ``floordiv`` expression otherwise)
 * — the tile's PHYSICAL half. How far apart the two lanes' data sits is the
 * partition stride instead (see ResolveLaneStride), which equals this half
 * unless the body was rebalanced onto a ragged boundary's valid region.
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
 * ``SplitAivScopeStmt`` consumer in LowerAutoVectorSplit (pass 20). Prepends a
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
 * @param mode The split mode (UpDown / LeftRight). The split attribute stamped
 *        on each cross-core op is derived from it and the parity of that op's
 *        own split-axis extent (see the ``kSplitUpDownOdd`` codes in stmt.h).
 * @param split_dim The partitioned tile dimension (see SplitDimension).
 * @param tile_vars In/out map of split-tracked tile vars to their halved extent.
 * @param is_aiv Whether this is an AIV lane (gates per-op halving).
 * @param subblock_idx The per-subblock index expr (null for non-AIV).
 * @param var_replacements In/out map of original vars to their rebuilt versions.
 * @param lane_stride The body's partition stride (see ResolveLaneStride); null
 *        keeps the default box partition.
 * @return The rewritten statement list.
 */
std::vector<StmtPtr> ProcessStmts(const std::vector<StmtPtr>& stmts, SplitMode mode, int split_dim,
                                  std::unordered_map<const Var*, TileInfo>& tile_vars, bool is_aiv,
                                  const ExprPtr& subblock_idx,
                                  std::unordered_map<const Var*, VarPtr>& var_replacements,
                                  const ExprPtr& lane_stride = nullptr);

/**
 * @brief Give each explicit-boundary ``tile.aiv_shard`` its TRUE per-lane valid
 *        extent on the split axis, and carry that extent to its consumers.
 *
 * ``ReshapeSplitAxis`` (the op deducer) can only halve the split-axis valid
 * extent with a ceil-div, because the lane index is not part of an op's type
 * function. When the operand's split-axis extent does not cover its physical
 * box, that guess is wrong for BOTH lanes: lane L holds
 * ``clamp(V - L * half, 0, half)``, so a ``[16, 256]`` accumulator valid to 5
 * rows must give lane 0 five rows and lane 1 none, not ``ceil(5 / 2) = 3`` each.
 *
 * A ``pl.split_aiv`` region body is the one place that guess can be repaired:
 * the region's own ``aiv_id = tile.get_subblock_idx()`` binding is in scope, so
 * the per-lane extent is materializable as an expression. The extent is then
 * propagated to every consumer that passes it through unchanged, because the
 * store is what finally reads it — and PTOAS offers no way to re-narrow a popped
 * tile after the fact (``pto.set_validshape`` needs a locally bound source,
 * which a frontend tpop result is not).
 *
 * Only the split axis is touched, and only when the operand is genuinely
 * partial there; a fully-valid operand keeps the deducer's exact ``half``.
 * A consumer that TRANSFORMS the extent (a reduction, a slice) rather than
 * passing it through is reported as a user-facing limitation with its span.
 *
 * The walk descends into nested control flow, so a shard / consumer / store
 * inside a loop or a branch is repaired exactly like a top-level one.
 *
 * @param stmts The region body statements (already half-width).
 * @param split_dim The partitioned tile dimension (see SplitDimension).
 * @param region_span Span of the enclosing region, for diagnostics.
 * @return The rewritten statement list (input order preserved).
 */
std::vector<StmtPtr> LocalizeExplicitBoundaryValid(const std::vector<StmtPtr>& stmts, int split_dim,
                                                   const Span& region_span);

/**
 * @brief Replace a shard result type's split-axis valid extent with the lane's own.
 *
 * ``ReshapeSplitAxis`` halves that extent with a ceil-div because an op's type
 * function does not know the lane; wherever a subblock index IS in scope, this
 * restores the truth ``clamp(V - idx * half, 0, half)``. Returns @p shard_type
 * unchanged when the operand's split axis is fully valid (both lanes then hold
 * exactly ``half``, which the deducer already produced) or when either type is
 * not a rank-covering TileType.
 *
 * @param shard_type The deduced per-lane (halved) tile type.
 * @param operand_type The pre-split tile type the shard reads.
 * @param split_dim The partitioned tile dimension (see SplitDimension).
 * @param subblock_idx The per-lane index expr; null leaves the type unchanged.
 * @param lane_stride The body's partition stride; null for the box partition.
 * @return The retyped shard type, or @p shard_type when no repair applies.
 */
TypePtr LocalizeShardValidForLane(const TypePtr& shard_type, const TypePtr& operand_type, int split_dim,
                                  const ExprPtr& subblock_idx, const ExprPtr& lane_stride = nullptr);

}  // namespace split_axis
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SPLIT_AXIS_UTILS_H_

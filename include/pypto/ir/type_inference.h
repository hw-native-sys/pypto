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

/**
 * @file type_inference.h
 * @brief Type inference utilities for operator type deduction
 *
 * This file provides utilities for automatic type deduction in operator
 * registration, including broadcasting shape inference, data type promotion,
 * and type compatibility checking.
 */

#ifndef PYPTO_IR_TYPE_INFERENCE_H_
#define PYPTO_IR_TYPE_INFERENCE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/printer.h"  // NOLINT(misc-include-cleaner) -- needed for operator<< on ExprPtr
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Result of shape broadcasting
 *
 * Contains the broadcast result shape or an error message if broadcasting fails.
 */
struct BroadcastResult {
  bool success;                // Whether broadcasting succeeded
  std::vector<ExprPtr> shape;  // Resulting broadcast shape (empty if failed)
  std::string error_message;   // Error message if broadcasting failed

  /**
   * @brief Create a successful broadcast result
   */
  static BroadcastResult Success(std::vector<ExprPtr> result_shape) {
    return BroadcastResult{true, std::move(result_shape), ""};
  }

  /**
   * @brief Create a failed broadcast result with error message
   */
  static BroadcastResult Failure(std::string message) {
    return BroadcastResult{false, {}, std::move(message)};
  }
};

/**
 * @brief Broadcast two shapes following NumPy-style broadcasting rules
 *
 * Broadcasting rules:
 * - Dimensions are aligned from right to left
 * - Size 1 dimensions are broadcast to match the other operand
 * - Missing dimensions are treated as size 1
 * - If dimensions don't match and neither is 1, broadcasting fails
 *
 * Examples:
 * - [4, 8] + [4, 8] -> [4, 8]
 * - [4, 8] + [8] -> [4, 8]
 * - [4, 1] + [8] -> [4, 8]
 * - [4, 8] + [5] -> Error (8 != 5)
 *
 * @param shape1 First shape
 * @param shape2 Second shape
 * @return BroadcastResult with the resulting shape or error
 */
BroadcastResult BroadcastShapes(const std::vector<ExprPtr>& shape1, const std::vector<ExprPtr>& shape2);

/**
 * @brief Promote two data types to a common type
 *
 * Type promotion rules follow standard numeric promotion:
 * - If types are the same, return that type
 * - Float types take precedence over integer types
 * - Larger types take precedence over smaller types
 * - Signed types take precedence over unsigned types of the same size
 *
 * Examples:
 * - INT32 + INT32 -> INT32
 * - INT32 + FP32 -> FP32
 * - INT32 + INT64 -> INT64
 * - UINT32 + INT32 -> INT32
 *
 * @param dtype1 First data type
 * @param dtype2 Second data type
 * @return Promoted data type, or std::nullopt if types are incompatible
 */
std::optional<DataType> PromoteDataTypes(DataType dtype1, DataType dtype2);

/**
 * @brief Check if two types are compatible for binary operations
 *
 * Types are compatible if:
 * - Both are scalar types
 * - Both are tensor types (shapes may differ for broadcasting)
 * - Both are tile types (shapes may differ for broadcasting)
 *
 * @param type1 First type
 * @param type2 Second type
 * @return true if types are compatible
 */
bool CheckTypeCompatibility(const TypePtr& type1, const TypePtr& type2);

/**
 * @brief Extract data type from a type pointer
 *
 * Works for ScalarType, TensorType, and TileType.
 *
 * @param type Type pointer
 * @return Data type, or std::nullopt if type is not a scalar/tensor/tile type
 */
std::optional<DataType> ExtractDataType(const TypePtr& type);

/**
 * @brief Extract shape from a tensor or tile type
 *
 * @param type Type pointer
 * @return Shape vector, or empty vector if type is not a tensor/tile type
 */
std::vector<ExprPtr> ExtractShape(const TypePtr& type);

/**
 * @brief Check if a dimension expression represents a constant value
 *
 * @param dim Dimension expression
 * @return std::optional with the constant value, or std::nullopt if not constant
 */
std::optional<int64_t> GetConstantDimension(const ExprPtr& dim);

/**
 * @brief Check if two dimension expressions are equal
 *
 * Handles both constant and symbolic dimensions.
 * For constant dimensions, compares values.
 * For symbolic dimensions, applies expression simplification and proves
 * equality via the arithmetic analyzer (e.g. (x + 64) - x and
 * (x + 128) - (x + 64) are both recognised as 64).
 *
 * @param dim1 First dimension
 * @param dim2 Second dimension
 * @return true if dimensions are equal
 */
bool DimensionsEqual(const ExprPtr& dim1, const ExprPtr& dim2);

/**
 * @brief Tri-state result for symbolic valid-extent proofs.
 *
 * ``kUnknown`` is intentionally distinct from ``kFalse``: callers that cannot
 * emit a runtime guard must reject both, but diagnostics should explain whether
 * the requested relation was disproved or merely could not be established.
 */
enum class ProofResult { kTrue, kFalse, kUnknown };

/** @brief Prove that two valid extents are equal for every runtime value. */
ProofResult ProveValidExtentEqual(const ExprPtr& lhs, const ExprPtr& rhs);

/** @brief Prove that ``lhs <= rhs`` for every runtime value. */
ProofResult ProveValidExtentLessEqual(const ExprPtr& lhs, const ExprPtr& rhs);

/**
 * @brief Validate the standing valid-shape rank and static-bounds invariant.
 *
 * Rejects rank mismatches and every statically provable violation of
 * ``0 <= valid[i] <= shape[i]``. Symbolic relations that cannot be proved are
 * deferred to the verifier/runtime contract rather than silently clamped.
 */
void ValidateValidShapeBounds(const std::vector<ExprPtr>& valid_shape,
                              const std::vector<ExprPtr>& physical_shape, const Span& span,
                              const std::string& op_name);

/**
 * @brief Validate that a physical window lies within its target allocation.
 *
 * Rejects rank mismatches, provably-negative offsets, and every provable
 * ``offset[i] + window_shape[i] > target_shape[i]`` violation. Unknown symbolic
 * relations defer to later verification/runtime guards. This checks the physical
 * window, not merely its valid sub-region, which is required for subview safety.
 */
void ValidatePhysicalWindowBounds(const std::vector<ExprPtr>& window_shape,
                                  const std::vector<ExprPtr>& offsets,
                                  const std::vector<ExprPtr>& target_shape, const Span& span,
                                  const std::string& op_name);

/**
 * @brief Lift an operand's origin-anchored valid box into broadcast-output coordinates.
 *
 * Shapes are right-aligned. A missing leading axis is an implicit valid singleton
 * and therefore covers the output axis. An explicit physical singleton contributes
 * ``valid * out_extent``: valid 1 covers the output axis, while valid 0 remains
 * empty. Non-broadcast axes keep their original valid extent.
 */
std::vector<ExprPtr> LiftValidShapeForBroadcast(const std::vector<ExprPtr>& operand_shape,
                                                const std::vector<ExprPtr>& operand_valid,
                                                const std::vector<ExprPtr>& out_shape, const Span& span,
                                                const std::string& op_name);

/**
 * @brief Map an origin-anchored valid box through a reshape without widening.
 *
 * Fully-valid inputs map to ``new_shape``. A partial input is accepted only when
 * it denotes a contiguous flat prefix and that prefix is a rectangular box under
 * ``new_shape``. Two exact exceptions do not repartition data: an empty input
 * stays empty, and inserting/removing provably-full physical unit axes preserves
 * any origin-anchored rectangle. Otherwise the relation is rejected because
 * ``valid_shape`` cannot represent the reshaped region. Tensor and tile reshape
 * share this rule.
 */
std::vector<ExprPtr> ComputeReshapeValidShape(const std::vector<ExprPtr>& src_valid,
                                              const std::vector<ExprPtr>& in_shape,
                                              const std::vector<ExprPtr>& new_shape, const Span& span,
                                              const std::string& op_name);

/**
 * @brief Compute a 2-D column-concat valid box shared by tensor and tile ops.
 *
 * Valid rows must agree, and the first operand must be fully valid in columns so
 * the second operand begins immediately after real data rather than after a gap.
 * The final operand may have a trailing invalid column suffix.
 */
std::vector<ExprPtr> ComputeConcatValidShape(const std::vector<ExprPtr>& lhs_shape,
                                             const std::vector<ExprPtr>& lhs_valid,
                                             const std::vector<ExprPtr>& rhs_shape,
                                             const std::vector<ExprPtr>& rhs_valid, const Span& span,
                                             const std::string& op_name);

/**
 * @brief Check if a dimension is broadcastable to another
 *
 * A dimension is broadcastable if:
 * - It's equal to the target dimension
 * - It's a constant 1
 * - The target dimension is a constant 1
 *
 * @param source_dim Source dimension
 * @param target_dim Target dimension
 * @return true if source can be broadcast to target
 */
bool IsBroadcastable(const ExprPtr& source_dim, const ExprPtr& target_dim);

/**
 * @brief Format a shape vector as a string for error messages
 *
 * Converts a shape (vector of ExprPtr) to a human-readable string.
 * Each dimension is printed using PythonPrint via operator<<.
 *
 * Examples:
 * - [ConstInt(64), ConstInt(128)] -> "[64, 128]"
 * - [ConstInt(64), Var("N")] -> "[64, N]"
 * - [BinaryOp(Var("M"), *, ConstInt(2))] -> "[M * 2]"
 * - [] -> "[]"
 *
 * @param shape Shape vector to format
 * @return String representation of the shape
 */
std::string FormatShape(const std::vector<ExprPtr>& shape);

/**
 * @brief Propagate blayout and pad from a source TileType's tile_view into a new TileView
 *
 * Many tile ops preserve the layout properties of their primary input. This helper copies
 * blayout and pad when the source has a tile_view, avoiding repeated inline checks.
 *
 * @param dst Destination TileView (valid_shape should already be set)
 * @param src Source TileType whose tile_view properties are inherited
 */
inline void InheritTileViewLayout(TileView& dst, const std::shared_ptr<const TileType>& src) {
  // Use the effective view: under canonicalization an implicit view is stored
  // as nullopt, but the inheritance still needs to see the resolved layout.
  const TileView eff = tile_view_semantics::GetEffectiveTileView(*src);
  dst.blayout = eff.blayout;
  dst.slayout = eff.slayout;
  dst.pad = eff.pad;
}

/**
 * @brief Preserve only layout requirements on a freshly computed tile result.
 *
 * Block/scatter layout and fractal size constrain how the new result is produced.
 * Source stride/start-offset are alias metadata, and source padding describes reads
 * from the old allocation, so those fields intentionally remain at fresh defaults.
 */
inline void InheritFreshTileComputeLayout(TileView& dst, const std::shared_ptr<const TileType>& src) {
  const TileView eff = tile_view_semantics::GetEffectiveTileView(*src);
  dst.blayout = eff.blayout;
  dst.slayout = eff.slayout;
  dst.fractal = eff.fractal;
}

/**
 * @brief Return the source tile's effective valid_shape, falling back to its static shape.
 *
 * Same-shape elementwise tile ops (tile.neg, tile.muls, tile.cast, ...) must propagate
 * the input's runtime valid_shape onto their result so that downstream codegen emits
 * matching validRow/validCol for src and dst. Without this propagation, a result built
 * from `tile_type->shape_` re-expands to the full allocation shape and the lowered
 * intrinsic receives mismatched valid extents (see issue #1370).
 *
 * @param tile_type Source TileType
 * @return The TileView::valid_shape if set, otherwise the static shape
 */
inline std::vector<ExprPtr> GetValidShape(const std::shared_ptr<const TileType>& tile_type) {
  if (tile_type->tile_view_ && !tile_type->tile_view_->valid_shape.empty()) {
    return tile_type->tile_view_->valid_shape;
  }
  return tile_type->shape_;
}

/**
 * @brief Return the source tensor's effective valid_shape, falling back to its static shape.
 *
 * The tensor-side dual of the TileType overload above. Tensor compute ops must propagate the
 * input's runtime valid_shape onto their result so the valid sub-region survives the tensor↔tile
 * boundary rather than re-expanding to the full allocation shape. An unset (empty) valid_shape
 * means "fully valid" — the physical shape — matching TileView and the canonical encoding enforced
 * by CanonicalizeTensorViewInPlace (src/ir/type.cpp): an explicit valid_shape equal to shape_ is
 * collapsed to empty, so the two in-memory forms are indistinguishable here by construction.
 *
 * @param t Source TensorType
 * @return The TensorView::valid_shape if set, otherwise the static shape
 */
inline std::vector<ExprPtr> GetValidShape(const std::shared_ptr<const TensorType>& t) {
  if (t->tensor_view_ && !t->tensor_view_->valid_shape.empty()) {
    return t->tensor_view_->valid_shape;
  }
  return t->shape_;  // unset => fully valid
}

/**
 * @brief Build the canonical view metadata for a freshly allocated tensor result.
 *
 * A compute result preserves the semantic valid box but does not alias its input
 * allocation. Input strides/layout and padding policy therefore must not leak onto
 * it: they describe how to address or read the source allocation, not the newly
 * produced values. Fresh tensor compute results use the default ND layout, empty
 * strides, and null padding. ``TensorType`` canonicalization removes this view when
 * ``effective_valid_shape`` equals the physical result shape.
 *
 * View-producing operations (slice, reshape, and similar) intentionally do not
 * use this helper; each derives whatever view metadata its own contract requires.
 */
inline std::optional<TensorView> MakeFreshTensorResultView(std::vector<ExprPtr> effective_valid_shape) {
  TensorView result;
  result.valid_shape = std::move(effective_valid_shape);
  return result;
}

/**
 * @brief Clip one dimension of a windowed view's valid region to the source's own.
 *
 * A window of ``window`` elements starting at ``offset`` into a source whose valid
 * extent along this dim is ``src_valid`` sees
 * ``clamp(src_valid - offset, 0, window)`` valid elements; the result is that
 * clipped by an explicit ``valid_arg`` (pass ``window`` itself when the op carries
 * no explicit valid extent for the dim):
 *
 * ```text
 * result = min( valid_arg, clamp(src_valid - offset, 0, window) )
 * ```
 *
 * Never widens past the source's real valid region (``valid_shape`` North Star):
 * it intersects. Constants fold to a single ``ConstInt`` and redundant
 * nodes are dropped: a zero offset skips the subtract/floor, and a ``valid_arg`` (or
 * available extent) equal to the full window carries no extra narrowing so the
 * ``min`` collapses. Shared by ``tile.load`` (tensor source), ``tile.slice`` and
 * ``tile.extract`` (tile source).
 *
 * A provably-negative constant ``offset`` is rejected via ``CHECK_SPAN``: the
 * window would start before the source origin, so its valid cells begin at local
 * index ``-offset`` and the region is no longer origin-anchored — a per-dim
 * ``valid_shape`` cannot represent it, and widening it to a full/anchored extent
 * is exactly the failure mode the North Star forbids. A symbolic offset defers to
 * runtime.
 *
 * @param src_valid Source's valid extent along this dim (``GetValidShape(src)[i]``).
 * @param offset    Window start offset along this dim (index-typed scalar).
 * @param window    Result's physical extent along this dim (the window size).
 * @param valid_arg Explicit valid extent requested for this dim, or ``window``.
 * @param span      Source location for folded-constant provenance.
 * @return The clipped valid extent, ``<= window``.
 */
ExprPtr IntersectWindowValidDim(const ExprPtr& src_valid, const ExprPtr& offset, const ExprPtr& window,
                                const ExprPtr& valid_arg, const Span& span);

/**
 * @brief Compute the per-dim bounding-box ``valid_shape`` for an ``assemble`` result.
 *
 * ``assemble(target, source, offset)`` writes ``source``'s valid region into
 * ``target`` starting at ``offset``. The written region is
 * ``[offset, offset + valid(source))``; the result's valid region is the UNION of
 * that written rectangle and the target's existing valid rectangle
 * ``[0, target_valid)``. A per-dim ``valid_shape`` describes one origin-anchored
 * rectangle, so it can express that union only when the union is itself such a
 * rectangle, in which case it equals the per-dim bounding box:
 *
 * ```text
 * out_valid[i] = min( shape[i], max( target_valid[i], offset[i] + source_valid[i] ) )
 * ```
 *
 * The box is accepted ONLY when the union is *provably* an origin-anchored
 * rectangle (target fully valid; or source within target; or target covered by an
 * origin-anchored source; or a single contiguously-growing free dim with every
 * other dim coinciding); otherwise it rejects via ``CHECK_SPAN`` — a non-rectangular
 * union (a gap, or an L-shape) OR one whose representability cannot be *proven* at
 * compile time would let the bounding box mark cells written by neither operand as
 * valid, and per the ``valid_shape`` North Star that is never widened away. The
 * proof is per-dim: a symbolic passenger dim does NOT disable the rejection of a
 * provably non-rectangular union in the remaining static dims. A provably-empty
 * source (some ``source_valid`` dim == 0) writes nothing and returns
 * ``target_valid`` unchanged.
 *
 * The outer ``min`` clamps to the physical shape so the standing
 * ``valid_shape[i] <= shape[i]`` invariant holds even for a symbolic write
 * (operators enforce provable violations directly; ``TypeCheck`` also enforces
 * it when property verification is enabled). When
 * ``offset[i] + source_valid[i]`` and ``shape[i]`` are all
 * compile-time constants, an out-of-bounds write (the box would exceed the
 * physical shape) is a user error and is rejected via ``CHECK_SPAN``. When both
 * operands of a fold are ``ConstInt`` the result is folded to a single ``ConstInt``
 * (no ``Max``/``Min``/``Add`` node — even when other dims are symbolic).
 *
 * All three of ``target_valid`` / ``source_valid`` / ``offset`` must share the
 * physical ``shape``'s rank: ``assemble`` writes ``source`` at ``offset`` into
 * ``target``, so a differing rank yields a region that a per-dim ``valid_shape``
 * cannot represent and is rejected rather than widened.
 *
 * @param target_valid Target's effective valid_shape (``GetValidShape(target)``).
 * @param source_valid Source's effective valid_shape (``GetValidShape(source)``).
 * @param offset       Per-dim write offsets (index-typed scalar expressions).
 * @param shape        Target's physical shape.
 * @param span         Source location for CHECK_SPAN diagnostics.
 * @param op_name      Operator name for error messages (e.g. "tile.assemble").
 * @return The bounding-box valid_shape, rank == ``shape.size()``.
 */
std::vector<ExprPtr> ComputeAssembleUnionValidShape(const std::vector<ExprPtr>& target_valid,
                                                    const std::vector<ExprPtr>& source_valid,
                                                    const std::vector<ExprPtr>& offset,
                                                    const std::vector<ExprPtr>& shape, const Span& span,
                                                    const std::string& op_name);

/**
 * @brief Require enough valid RHS rows for the LHS-selected contraction length.
 *
 * PTO TMATMUL takes its contraction length from the LHS tile's valid column
 * extent. The RHS must therefore provide at least that many valid rows:
 * ``valid(lhs)[K] <= valid(rhs)[K]``. A provably-false relation is rejected as
 * unsafe; an unprovable relation is also rejected because this type-inference
 * path emits no runtime guard. Equal symbolic expressions and other relations
 * established by the arithmetic analyzer remain supported.
 *
 * This validates the *valid* K extents, which may be narrower than the physical K the
 * caller already shape-checks; e.g. two ``128``-column operands where one carries an
 * explicit ``valid_shape`` K of ``64``.
 *
 * ``CHECK_SPAN`` (user error), not ``INTERNAL``: the mismatch is authored via
 * ``pl.matmul`` / ``pl.tile.matmul`` and friends.
 *
 * @param k_lhs_valid lhs valid extent along its K axis (``GetValidShape(lhs)[K]``).
 * @param k_rhs_valid rhs valid extent along its K axis (``GetValidShape(rhs)[K]``).
 * @param span        Source location for CHECK_SPAN diagnostics.
 * @param op_name     Operator name for error messages (e.g. "tile.matmul").
 */
void CheckMatMulValidKCompat(const ExprPtr& k_lhs_valid, const ExprPtr& k_rhs_valid, const Span& span,
                             const std::string& op_name);

/**
 * @brief Compute the output ``valid_shape`` of a matmul-with-bias (``C = A @ B + bias``).
 *
 * ``C[i,j] = (A@B)[i,j] + bias[0,j]``. The matmul term is valid over rows
 * ``valid(A)[M]`` and columns ``valid(B)[N]``. The bias term is real only where
 * ``bias[0,j]`` is real — column ``j < valid(bias)[N]`` AND the single ``[1, N]`` bias
 * row itself valid. A column whose bias is padding yields ``C[:,j] = A@B + garbage``,
 * which is NOT the intended ``A@B + bias`` and is therefore invalid.
 *
 * The output N extent is the INTERSECTION ``min(valid(B)[N], valid(bias)[N])`` — the
 * bias narrows the reported region but never widens it past the real bias columns (the
 * ``valid_shape`` North Star forbids marking padding as valid). The singleton bias row
 * must be provably valid (extent 1); zero or unknown row validity rejects because this
 * path emits no runtime guard. The M extent is ``valid(A)[M]``, unaffected by the bias.
 * When both N extents are ``ConstInt`` the ``min`` folds to a single ``ConstInt``.
 *
 * Shared by ``tile.matmul_bias`` and ``tile.gemv_bias`` (M = lhs axis 0, N = rhs axis 1).
 *
 * @param lhs_valid  lhs effective valid_shape (``GetValidShape(lhs)``); [M, K].
 * @param rhs_valid  rhs effective valid_shape (``GetValidShape(rhs)``); [K, N].
 * @param bias_valid bias effective valid_shape (``GetValidShape(bias)``); [1, N].
 * @param span       Source location for folded-constant provenance.
 * @return The 2-D output valid_shape [valid(A)[M], min(valid(B)[N], valid(bias)[N])].
 */
std::vector<ExprPtr> ComputeMatMulBiasValidShape(const std::vector<ExprPtr>& lhs_valid,
                                                 const std::vector<ExprPtr>& rhs_valid,
                                                 const std::vector<ExprPtr>& bias_valid, const Span& span);

/**
 * @brief Compute the output ``valid_shape`` of a broadcasting elementwise op.
 *
 * Per-dim agreement: ``valid_shape`` never silently widens or narrows.
 *
 * The type-agnostic core of the valid-region agreement rule shared by
 * ``DeduceBroadcastElementwiseTileView`` (tile side) and the tensor-level
 * elementwise-binary deducer. For each output dim ``i``, every non-broadcast
 * operand contributes its valid extent and all contributors must be provably
 * equal. An explicit physical singleton that broadcasts to a larger output dim
 * is exempt only after its sole element is proved valid (``valid == 1``); an
 * empty or symbolically-unknown singleton is rejected because every output cell
 * would otherwise read padding. Missing leading axes are implicit valid
 * singleton axes. Broadcasting is right-aligned.
 *
 * ``operand_shapes[k]`` / ``operand_valids[k]`` are operand ``k``'s physical shape and
 * effective valid_shape (``GetValidShape``). The returned vector has rank
 * ``out_shape.size()``.
 *
 * @param out_shape       The broadcast output (result) physical shape.
 * @param operand_shapes  Per-operand physical shapes (non-empty; rank <= out rank).
 * @param operand_valids  Per-operand effective valid_shapes (parallel to shapes).
 * @param span            Source location for CHECK_SPAN diagnostics.
 * @param op_name         Operator name for error messages (e.g. "tensor.add").
 * @return The agreed output valid_shape, rank == ``out_shape.size()``.
 */
std::vector<ExprPtr> ComputeBroadcastElementwiseValidShape(
    const std::vector<ExprPtr>& out_shape, const std::vector<std::vector<ExprPtr>>& operand_shapes,
    const std::vector<std::vector<ExprPtr>>& operand_valids, const Span& span, const std::string& op_name);

/**
 * @brief Reject a partially-valid tensor input for an op that cannot derive its
 *        output valid region from one.
 *
 * The tensor-side dual of ``tile_ops/sort.cpp``'s ``CheckSortInputFullyValid``.
 * Some ops have no way to prove the output valid region when an input is only
 * partially valid: a sort mixes padding into the valid region under comparison,
 * while indirect writes such as scatter land at runtime indices. Per the
 * ``valid_shape`` North Star such a case is
 * rejected rather than silently widened to the full shape. A canonicalized fully
 * valid input has ``GetValidShape == shape`` (its redundant view collapses to
 * empty), so this passes exactly the truly-full inputs.
 *
 * ``CHECK_SPAN`` (user error): the tensor is authored via ``pl.slice(valid_shape=)``
 * / ``pl.load(valid_shapes=)`` and friends.
 *
 * @param t        Candidate tensor input.
 * @param op_name  Operator name for the error message (e.g. "tensor.sort32").
 * @param arg_desc Human-readable argument name (e.g. "src").
 * @param span     Source location for CHECK_SPAN diagnostics.
 */
void CheckTensorInputFullyValid(const std::shared_ptr<const TensorType>& t, const std::string& op_name,
                                const std::string& arg_desc, const Span& span);

/**
 * @brief Tile-side counterpart of ``CheckTensorInputFullyValid``.
 *
 * Use for tile operations whose semantics cannot safely consume or propagate a
 * partial valid region (for example, indirect gather/scatter families). The
 * diagnostic is a user error and suggests materializing padding first.
 */
void CheckTileInputFullyValid(const std::shared_ptr<const TileType>& t, const std::string& op_name,
                              const std::string& arg_desc, const Span& span);

/**
 * @brief Require every explicitly-present batch axis to be provably full.
 *
 * Batch matmul lowering enumerates physical pages. Until it can guard a partial
 * batch tail, accepting a partial or symbolically-unproved batch extent would
 * address padding as a real matrix page. Matrix axes are checked separately.
 */
void CheckBatchAxesFullyValid(const std::vector<ExprPtr>& batch_shape,
                              const std::vector<ExprPtr>& batch_valid, const Span& span,
                              const std::string& op_name, const std::string& operand_name);

/**
 * @brief Pick the "shaping" tile operand for result-layout inheritance.
 *
 * A broadcasting multi-operand tile op (``tile.add``, ``tile.sel``, ...) must take
 * its result ``blayout`` / ``slayout`` / ``fractal`` from an operand whose PHYSICAL
 * shape equals the broadcast output shape — never from a shape-1 broadcast operand,
 * whose layout (e.g. the ``col_major`` a ``[N,1]`` tile infers) would otherwise leak
 * into a full-shaped result. Returns the first operand whose ``shape_`` equals
 * ``out_shape``. Returns null when none is full-shaped (for example mutually
 * broadcasting ``[R,1]`` and ``[1,C]`` operands); callers must then infer the
 * fresh result layout from ``out_shape`` itself so commutative operand order
 * cannot change the result type.
 *
 * @param out_shape The broadcast output (result) physical shape.
 * @param operands  The value tile operands participating in the op.
 * @return The layout-source tile, or null when no operand owns the output shape.
 */
std::shared_ptr<const TileType> PickElementwiseLayoutSource(
    const std::vector<ExprPtr>& out_shape, const std::vector<std::shared_ptr<const TileType>>& operands);

/**
 * @brief Deduce the result ``TileView`` for a broadcasting multi-operand tile op.
 *
 * Encapsulates the two 4e fixes for every true-elementwise deducer (binary, shift,
 * ternary, tri-tile, tile-scalar-tile, sel, sel-scalar, and the cmp/cmps mask):
 *
 * 1. **valid_shape agreement.** Every non-broadcast contributor must be
 *    provably equal. Unknown equality rejects because no runtime guard is
 *    emitted. A physical singleton broadcasting to a larger dim is exempt only
 *    when its valid extent is provably 1; a valid-zero or unknown singleton is
 *    rejected rather than treated as real data.
 *
 * 2. **layout from the shaping operand** — via ``PickElementwiseLayoutSource``, never
 *    from a shape-1 broadcast operand. Only layout/fractal requirements survive;
 *    fresh results intentionally do not inherit source pad or alias metadata.
 *
 * ``operands`` lists ONLY the value tiles whose valid regions participate — exclude
 * scalar operands, ``tmp`` scratch tiles, and the ``sel`` predicate mask (whose
 * ceil-div'd valid columns cannot agree with a value tile). The returned view's
 * ``valid_shape`` always has rank ``out_shape.size()`` (a fully-valid result carries
 * ``valid_shape == out_shape``; ``TileType`` construction canonicalizes it away).
 *
 * @param out_shape The broadcast output (result) physical shape.
 * @param operands  The value tile operands (non-empty).
 * @param span      Source location for CHECK_SPAN diagnostics.
 * @param op_name   Operator name for error messages (e.g. "tile.add").
 * @return The result TileView (agreed valid_shape + inherited layout/fractal).
 */
TileView DeduceBroadcastElementwiseTileView(const std::vector<ExprPtr>& out_shape,
                                            const std::vector<std::shared_ptr<const TileType>>& operands,
                                            const Span& span, const std::string& op_name);

/**
 * @brief Deduce return types for a cross-function call by substituting dynamic
 *        shape variables in the callee's return types with concrete values from
 *        the actual call arguments.
 *
 * Builds a mapping from Var dimensions in callee param types to the
 * corresponding dimensions in actual arg types, then substitutes those
 * Vars in each return type. Handles TensorType and DistributedTensorType
 * (shape + tensor_view, preserving distributed window identity), TileType
 * (shape + tile_view), and TupleType (recursive). Substitution recursively
 * rebuilds composite metadata expressions such as ``M + 1``.
 *
 * @param callee_params  Callee function parameter variables
 * @param args           Actual call argument expressions
 * @param return_types   Callee's declared return types
 * @return Substituted return types (unchanged if no dynamic vars found)
 */
std::vector<TypePtr> DeduceCallReturnType(const std::vector<VarPtr>& callee_params,
                                          const std::vector<ExprPtr>& args,
                                          const std::vector<TypePtr>& return_types);

/**
 * @brief Parse and validate the optional ``drop_dims`` operand of a slice op.
 *
 * ``tensor.slice`` / ``tile.slice`` accept an optional trailing positional
 * argument listing axes to remove from the result type (numpy-style rank
 * reduction). The operand is a ``MakeTuple`` of ``ConstInt``; an empty tuple,
 * or a null operand, means "drop nothing". Every listed axis must be in
 * ``[0, full_shape.size())``, appear at most once, and select a statically
 * unit-sized dimension of ``full_shape`` — rank reduction only erases unit dims.
 *
 * @param drop_dims_arg The drop_dims operand, or nullptr if the op has no such argument.
 * @param full_shape The full (pre-reduction) slice shape.
 * @param op_name Operator name for error messages (e.g. "tensor.slice").
 * @return The validated axes in ascending order; empty when nothing is dropped.
 */
std::vector<int64_t> ParseSliceDropDims(const ExprPtr& drop_dims_arg, const std::vector<ExprPtr>& full_shape,
                                        const std::string& op_name);

/**
 * @brief Remove the axes in ``drop_dims`` (ascending, validated) from ``shape``.
 *
 * Returns ``shape`` unchanged when ``drop_dims`` is empty.
 */
std::vector<ExprPtr> ApplyDropDims(const std::vector<ExprPtr>& shape, const std::vector<int64_t>& drop_dims);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TYPE_INFERENCE_H_

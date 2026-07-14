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

#include <cstddef>
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
 * @brief Tri-state result for symbolic valid-extent proof obligations
 *
 * A relation is true or false only when the arithmetic analyzer can prove that
 * result. Symbolic relations that cannot be decided remain unknown.
 */
enum class ProofResult {
  kTrue,
  kFalse,
  kUnknown,
};

/**
 * @brief Prove whether two valid-extent expressions are equal
 *
 * Recognizes structural identity, equal constants, and relations established
 * by the arithmetic analyzer.
 */
ProofResult ProveValidExtentEqual(const ExprPtr& lhs, const ExprPtr& rhs);

/**
 * @brief Prove whether one valid-extent expression is less than or equal to another
 *
 * @return kTrue when lhs <= rhs is proven, kFalse when lhs > rhs is proven,
 *         and kUnknown otherwise
 */
ProofResult ProveValidExtentLessEqual(const ExprPtr& lhs, const ExprPtr& rhs);

/**
 * @brief Kinds of malformed explicit valid shapes
 */
enum class ValidShapeBoundsViolation {
  kRankMismatch,
  kNegativeExtent,
  kExceedsPhysicalExtent,
};

/**
 * @brief A structured valid-shape validation failure
 */
struct ValidShapeBoundsError {
  ValidShapeBoundsViolation violation;
  std::optional<size_t> dimension;
  std::string message;
};

/**
 * @brief Validate the standing bounds invariant for an explicit valid shape
 *
 * Checks rank(valid) == rank(physical) and every provable violation of
 * 0 <= valid[i] <= physical[i]. Unknown symbolic relations are accepted.
 * An empty valid shape represents the full physical shape and is valid.
 *
 * @param valid Explicit valid shape, or empty for implicit full validity
 * @param physical Physical shape
 * @param type_kind Shaped type name used in diagnostics
 * @return All provable violations
 */
std::vector<ValidShapeBoundsError> ValidateValidShapeBounds(const std::vector<ExprPtr>& valid,
                                                            const std::vector<ExprPtr>& physical,
                                                            const std::string& type_kind);

/**
 * @brief Read the elements of a tuple-typed operand
 *
 * A ``MakeTuple`` operand yields its elements directly, which preserves the
 * ``ConstInt``s the arithmetic analyzer needs to fold. Any other tuple
 * expression is only reachable element-wise, through a ``TupleGetItemExpr``
 * projection.
 *
 * @param tuple_expr A tuple-typed operand
 * @param rank Arity of the tuple. Used only for a runtime tuple, whose elements are
 *             projected one by one; a ``MakeTuple`` already carries its own elements.
 * @return One expression per tuple element
 */
std::vector<ExprPtr> ExtractTupleElements(const ExprPtr& tuple_expr, size_t rank);

/**
 * @brief Whether the substrate under a window read trims an over-extent window
 *
 * This decides which extent has to lie inside the source, and therefore what a
 * non-clamping read is allowed to promise. It is a property of the machinery
 * beneath the operator, not of aliasing: what matters is whether an over-extent
 * window is trimmed for us, or reaches the hardware as written.
 */
enum class WindowReadKind {
  /// The substrate trims the window, so it may deliberately overhang the source
  /// and only the extent actually read has to fit.
  ///
  /// ``tensor.slice``: PTO codegen emits the view shape already clamped to
  /// ``min(shape, parent - offset)``, because the strided-Tensor runtime enforces
  /// ``offset + shape <= parent`` in ``Tensor::view``. A padded fixed-width window
  /// with an explicit ``valid_shape`` naming the real extent is the standard idiom.
  ///
  /// ``tile.load``: the DMA fetches only the valid extent, so the destination tile
  /// is free to be larger than the region that exists.
  kClampedWindow,
  /// Nothing trims the window, so all of it must lie inside the source.
  ///
  /// ``tile.slice`` lowers to ``pto.subview``, a pure view that does no bounds work,
  /// and ``tile.extract`` lowers to ISA TEXTRACT, whose bounds are hard. An on-chip
  /// window that overhangs is simply unrepresentable.
  kExactWindow,
};

/**
 * @brief Inputs to the shared window-read valid-region rule
 *
 * All shape-like vectors are in source coordinates and must share one rank,
 * except ``requested_valid`` which may be empty to mean "no explicit request".
 */
struct WindowReadValidShapeParams {
  std::vector<ExprPtr> source_physical;  ///< Physical shape of the source
  /// Source valid shape, already resolved to the source rank by ``GetValidShape``
  /// / ``GetEffectiveTensorValidShape`` — never empty.
  std::vector<ExprPtr> source_valid;
  std::vector<ExprPtr> offsets;          ///< Window origin, in source coordinates
  std::vector<ExprPtr> window;           ///< Physical shape of the result window
  std::vector<ExprPtr> requested_valid;  ///< Explicit valid request; empty means "none"
  WindowReadKind kind = WindowReadKind::kExactWindow;
  bool clamp = false;   ///< Sanction a ragged window that crosses the source edge
  std::string op_name;  ///< Operator name, used in diagnostics
  /// Way out, appended to a physical-bounds rejection. Reads that can clamp point
  /// the caller at ``clamp=True``; an on-chip tile window, which nothing can
  /// clamp, has to say so instead of naming an option it does not have.
  std::string bounds_remedy;
  Span span = Span::unknown();
};

/**
 * @brief Derive the valid region of a window read
 *
 * Implements the one rule shared by every window read, per dimension:
 *
 * ```text
 * available    = clamp(source_valid - offset, 0, window)
 * result_valid = min(requested_valid, available)
 * ```
 *
 * so a read can never widen beyond the source valid region, the requested valid
 * region, or the result window.
 *
 * **The non-clamping contract.** A read with ``clamp == false`` asserts that its
 * window lies inside the source: ``offset[i] + extent[i] <= source_physical[i]``,
 * where ``extent`` is the whole window for ``kExactWindow``, and the extent actually
 * read (the explicit valid request, when given) for ``kClampedWindow``.
 * Provable violations are rejected here; relations that stay symbolic are taken
 * on trust, because that inequality *is* the operator's precondition. Under it a
 * fully-valid source yields a fully-valid window, so the clamp collapses to the
 * window and no guard expression is built — an in-bounds read of an unpadded
 * source keeps the shape it had before this rule existed. Pass ``clamp = true``
 * to drop the assertion and clamp the valid region to the source edge instead,
 * which is how a sanctioned ragged tail is expressed.
 *
 * Expressions are built proof-first: a term is emitted only when the arithmetic
 * analyzer cannot already settle the comparison, and every term is simplified, so
 * constant arithmetic folds and no redundant ``min`` / ``max`` nesting survives.
 *
 * @param params Window-read description; see WindowReadValidShapeParams
 * @return The result valid shape, one extent per window dimension
 * @throws pypto::ValueError on rank mismatch, provably negative offset, or a
 *         provable physical-bounds violation of a non-clamping read
 */
std::vector<ExprPtr> InferWindowReadValidShape(const WindowReadValidShapeParams& params);

/**
 * @brief Return the effective valid shape of a tensor type
 *
 * Falls back to the physical shape when no explicit valid shape is set, matching
 * ``GetValidShape`` for tiles.
 */
const std::vector<ExprPtr>& GetEffectiveTensorValidShape(const TensorType& type);

/**
 * @brief Reject ``drop_dims`` axes that do not carry provably unit validity
 *
 * Rank reduction erases an axis, so the axis must have nothing left to say: its
 * post-intersection valid extent must be provably one. ``ParseSliceDropDims``
 * already requires a static unit *physical* extent; this is the validity-side
 * obligation, which only bites when a partial source or a clamp narrows the axis
 * below its physical extent.
 *
 * @param drop_dims Validated axes, ascending, indexing into ``valid_shape``
 * @param valid_shape Post-intersection valid shape, at full pre-reduction rank
 * @param op_name Operator name, used in diagnostics
 * @param span IR source location, reported when a dropped axis is rejected
 * @throws pypto::ValueError when a dropped axis is not provably one
 */
void ValidateDropDimsValidExtents(const std::vector<int64_t>& drop_dims,
                                  const std::vector<ExprPtr>& valid_shape, const std::string& op_name,
                                  const Span& span);

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
 * @brief Deduce return types for a cross-function call by substituting dynamic
 *        shape variables in the callee's return types with concrete values from
 *        the actual call arguments.
 *
 * Builds a mapping from Var dimensions in callee param types to the
 * corresponding metadata expressions in actual arg types, then substitutes
 * those Vars in each return type. Handles TensorType, DistributedTensorType,
 * TileType, and TupleType recursively, including expressions nested in shapes
 * and view metadata.
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

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

#include "pypto/ir/type_inference.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

BroadcastResult BroadcastShapes(const std::vector<ExprPtr>& shape1, const std::vector<ExprPtr>& shape2) {
  // Handle empty shapes
  if (shape1.empty() && shape2.empty()) {
    return BroadcastResult::Success({});
  }
  if (shape1.empty()) {
    return BroadcastResult::Success(shape2);
  }
  if (shape2.empty()) {
    return BroadcastResult::Success(shape1);
  }

  // Broadcast from right to left
  size_t max_ndim = std::max(shape1.size(), shape2.size());
  std::vector<ExprPtr> result_shape;
  result_shape.reserve(max_ndim);

  for (size_t i = 0; i < max_ndim; ++i) {
    // Get dimensions from right to left
    int64_t idx1 = static_cast<int64_t>(shape1.size()) - 1 - i;  // NOLINT
    int64_t idx2 = static_cast<int64_t>(shape2.size()) - 1 - i;  // NOLINT

    ExprPtr dim1 = (idx1 >= 0) ? shape1[idx1] : nullptr;
    ExprPtr dim2 = (idx2 >= 0) ? shape2[idx2] : nullptr;

    // If one dimension is missing, use the other
    if (!dim1) {
      result_shape.push_back(dim2);
      continue;
    }
    if (!dim2) {
      result_shape.push_back(dim1);
      continue;
    }

    // Check if dimensions are equal
    if (DimensionsEqual(dim1, dim2)) {
      result_shape.push_back(dim1);
      continue;
    }

    // Check if either dimension is 1 (broadcastable)
    auto const_dim1 = GetConstantDimension(dim1);
    auto const_dim2 = GetConstantDimension(dim2);

    if (const_dim1 && *const_dim1 == 1) {
      result_shape.push_back(dim2);
      continue;
    }
    if (const_dim2 && *const_dim2 == 1) {
      result_shape.push_back(dim1);
      continue;
    }

    // Dimensions are incompatible for broadcasting
    std::ostringstream oss;
    oss << "Cannot broadcast shapes: dimension " << i << " mismatch";
    return BroadcastResult::Failure(oss.str());
  }

  // Reverse result since we built it from right to left
  std::reverse(result_shape.begin(), result_shape.end());
  return BroadcastResult::Success(std::move(result_shape));
}

std::optional<DataType> PromoteDataTypes(DataType dtype1, DataType dtype2) {
  // If types are the same, return that type
  if (dtype1 == dtype2) {
    return dtype1;
  }

  // Float types take precedence
  bool is_float1 = dtype1.IsFloat();
  bool is_float2 = dtype2.IsFloat();

  if (is_float1 && !is_float2) {
    return dtype1;
  }
  if (is_float2 && !is_float1) {
    return dtype2;
  }

  // Both are floats or both are integers
  // Return the larger type
  size_t bits1 = dtype1.GetBit();
  size_t bits2 = dtype2.GetBit();

  if (bits1 > bits2) {
    return dtype1;
  }
  if (bits2 > bits1) {
    return dtype2;
  }

  // Same size - prefer signed over unsigned for integers
  if (!is_float1 && dtype1.IsSignedInt()) {
    return dtype1;
  }
  if (!is_float2 && dtype2.IsSignedInt()) {
    return dtype2;
  }

  // Default to first type
  return dtype1;
}

bool CheckTypeCompatibility(const TypePtr& type1, const TypePtr& type2) {
  // Check if both are scalar types
  auto scalar1 = As<ScalarType>(type1);
  auto scalar2 = As<ScalarType>(type2);
  if (scalar1 && scalar2) {
    return true;
  }

  // Check if both are tensor types
  auto tensor1 = As<TensorType>(type1);
  auto tensor2 = As<TensorType>(type2);
  if (tensor1 && tensor2) {
    return true;
  }

  // Check if both are tile types
  auto tile1 = As<TileType>(type1);
  auto tile2 = As<TileType>(type2);
  if (tile1 && tile2) {
    return true;
  }

  // Types are not compatible
  return false;
}

std::optional<DataType> ExtractDataType(const TypePtr& type) {
  // Try ScalarType
  if (auto scalar = As<ScalarType>(type)) {
    return scalar->dtype_;
  }

  // Try TensorType
  if (auto tensor = As<TensorType>(type)) {
    return tensor->dtype_;
  }

  // Try TileType
  if (auto tile = As<TileType>(type)) {
    return tile->dtype_;
  }

  return std::nullopt;
}

std::vector<ExprPtr> ExtractShape(const TypePtr& type) {
  // Try TensorType
  if (auto tensor = As<TensorType>(type)) {
    return tensor->shape_;
  }

  // Try TileType
  if (auto tile = As<TileType>(type)) {
    return tile->shape_;
  }

  // Not a shaped type
  return {};
}

std::optional<int64_t> GetConstantDimension(const ExprPtr& dim) {
  // Try to cast to ConstInt
  if (auto const_int = As<ConstInt>(dim)) {
    return const_int->value_;
  }

  // Not a constant
  return std::nullopt;
}

bool DimensionsEqual(const ExprPtr& dim1, const ExprPtr& dim2) {
  // Pointer equality (same object)
  if (dim1 == dim2) {
    return true;
  }

  // Try constant comparison
  auto const1 = GetConstantDimension(dim1);
  auto const2 = GetConstantDimension(dim2);

  if (const1 && const2) {
    return *const1 == *const2;
  }

  // For symbolic dimensions, prove equality via expression simplification.
  // Handles cases like `(x + 64) - x` vs `(x + 128) - (x + 64)` which both
  // reduce to 64 but are not structurally identical.
  //
  // Uses a thread_local analyzer so repeated calls on the slow path (e.g.
  // per-dim inside BroadcastShapes) reuse sub-analyzer state instead of
  // paying full setup per call.
  thread_local arith::Analyzer analyzer;
  return analyzer.CanProveEqual(dim1, dim2);
}

namespace {
bool AreComparableIntegerScalarExprs(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (!lhs || !rhs) {
    return false;
  }
  auto lhs_type = As<ScalarType>(lhs->GetType());
  auto rhs_type = As<ScalarType>(rhs->GetType());
  if (!lhs_type || !rhs_type || !lhs_type->dtype_.IsInt() || !rhs_type->dtype_.IsInt()) {
    return false;
  }
  return lhs_type->dtype_.IsSignedInt() == rhs_type->dtype_.IsSignedInt();
}

// The zero extent every valid-shape bound is compared against. Cached because it is otherwise
// rebuilt per dimension on a hot construction path; ConstInt is immutable, so sharing it is safe.
const ExprPtr& ZeroExtent() {
  static const ExprPtr zero = std::make_shared<ConstInt>(0, DataType::INDEX, Span::unknown());
  return zero;
}

// True when `extent` is provably zero, i.e. the region it bounds is empty.
//
// A constant is compared by value rather than routed through ProveValidExtentEqual. That helper
// only decides extents of matching signedness, so an *unsigned* zero -- e.g. a UINT64 valid_rows
// from set_validshape -- against the signed INDEX zero comes back kUnknown, which would let exactly
// the empty region this predicate exists to catch through. Symbolic extents are compared against a
// zero of their own dtype so the analyzer can decide them at all.
bool IsProvablyEmptyExtent(const ExprPtr& extent) {
  if (!extent) {
    return false;
  }
  if (const auto constant = GetConstantDimension(extent)) {
    return *constant == 0;
  }
  auto scalar_type = As<ScalarType>(extent->GetType());
  if (!scalar_type || !scalar_type->dtype_.IsInt()) {
    return false;
  }
  const auto zero = std::make_shared<ConstInt>(0, scalar_type->dtype_, Span::unknown());
  return ProveValidExtentEqual(extent, zero) == ProofResult::kTrue;
}
}  // namespace

ProofResult ProveValidExtentEqual(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (!AreComparableIntegerScalarExprs(lhs, rhs)) {
    return ProofResult::kUnknown;
  }
  if (AreExprsEqual(lhs, rhs)) {
    return ProofResult::kTrue;
  }

  thread_local arith::Analyzer analyzer;
  if (analyzer.CanProveEqual(lhs, rhs)) {
    return ProofResult::kTrue;
  }
  if (analyzer.CanProve(MakeNe(lhs, rhs))) {
    return ProofResult::kFalse;
  }
  return ProofResult::kUnknown;
}

ProofResult ProveValidExtentLessEqual(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (!AreComparableIntegerScalarExprs(lhs, rhs)) {
    return ProofResult::kUnknown;
  }
  if (AreExprsEqual(lhs, rhs)) {
    return ProofResult::kTrue;
  }

  thread_local arith::Analyzer analyzer;
  if (analyzer.CanProve(MakeLe(lhs, rhs))) {
    return ProofResult::kTrue;
  }
  if (analyzer.CanProve(MakeGt(lhs, rhs))) {
    return ProofResult::kFalse;
  }
  return ProofResult::kUnknown;
}

bool IsBroadcastable(const ExprPtr& source_dim, const ExprPtr& target_dim) {
  // If dimensions are equal, they're broadcastable
  if (DimensionsEqual(source_dim, target_dim)) {
    return true;
  }

  // Check if source is constant 1
  auto const_source = GetConstantDimension(source_dim);
  if (const_source && *const_source == 1) {
    return true;
  }

  // Check if target is constant 1
  auto const_target = GetConstantDimension(target_dim);
  if (const_target && *const_target == 1) {
    return true;
  }

  return false;
}

std::string FormatShape(const std::vector<ExprPtr>& shape) {
  if (shape.empty()) {
    return "[]";
  }

  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << PythonPrint(shape[i]);
  }
  oss << "]";
  return oss.str();
}

std::vector<ValidShapeBoundsError> ValidateValidShapeBounds(const std::vector<ExprPtr>& valid,
                                                            const std::vector<ExprPtr>& physical,
                                                            const std::string& type_kind) {
  if (valid.empty()) {
    return {};
  }

  if (valid.size() != physical.size()) {
    std::ostringstream msg;
    msg << type_kind << " valid_shape rank mismatch: got rank " << valid.size() << " " << FormatShape(valid)
        << ", but physical shape has rank " << physical.size() << " " << FormatShape(physical);
    return {{ValidShapeBoundsViolation::kRankMismatch, std::nullopt, msg.str()}};
  }

  std::vector<ValidShapeBoundsError> errors;
  const ExprPtr& zero = ZeroExtent();
  for (size_t i = 0; i < valid.size(); ++i) {
    if (ProveValidExtentLessEqual(zero, valid[i]) == ProofResult::kFalse) {
      std::ostringstream msg;
      msg << type_kind << " valid_shape dimension " << i << " has provably negative extent "
          << PythonPrint(valid[i]) << "; expected 0 <= valid_shape[" << i << "] <= shape[" << i << "] ("
          << PythonPrint(physical[i]) << ")";
      errors.push_back({ValidShapeBoundsViolation::kNegativeExtent, i, msg.str()});
    }
    if (ProveValidExtentLessEqual(valid[i], physical[i]) == ProofResult::kFalse) {
      std::ostringstream msg;
      msg << type_kind << " valid_shape dimension " << i << " extent " << PythonPrint(valid[i])
          << " provably exceeds physical shape extent " << PythonPrint(physical[i]);
      errors.push_back({ValidShapeBoundsViolation::kExceedsPhysicalExtent, i, msg.str()});
    }
  }
  return errors;
}

void CheckGatherRowOperands(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  auto shapes = As<MakeTuple>(args[4]);
  CHECK(shapes) << "The operator " << op_name << " requires shapes to be a literal tuple, but got "
                << args[4]->TypeName();
  for (size_t i = 0; i < shapes->elements_.size(); ++i) {
    CHECK(As<ConstInt>(shapes->elements_[i]))
        << "The operator " << op_name << " requires shapes[" << i
        << "] to be a compile-time constant (it sizes pto.subview, whose sizes is a static attribute), "
           "but got a runtime value. Pass a dynamic row count through valid_shape instead — it keeps "
           "the window (and so the tile allocation and box alignment) static while varying only the "
           "transfer length.";
  }
  if (args.size() < 6) return;

  auto valid = As<MakeTuple>(args[5]);
  CHECK(valid) << "The operator " << op_name << " requires valid_shape to be a literal tuple, but got "
               << args[5]->TypeName();
  // Rank must be checked here, not left to ValidateValidShapeBounds: that helper
  // reads an *empty* valid shape as "implicitly fully valid" and accepts it, which
  // is right for a type's valid_shape but wrong for an explicit operand — an empty
  // one would sail past deduction and trip an INTERNAL_CHECK in the backend.
  CHECK(valid->elements_.size() == shapes->elements_.size())
      << "The operator " << op_name << " requires valid_shape to have the same rank as shapes ("
      << shapes->elements_.size() << "), but got rank " << valid->elements_.size();
  // The bounds proofs below return "unknown" for a non-integer scalar rather than
  // rejecting it, so a literal like [1.5, 128] would otherwise reach lowering as a
  // fractional transfer extent.
  for (size_t i = 0; i < valid->elements_.size(); ++i) {
    auto dtype = ExtractDataType(valid->elements_[i]->GetType());
    CHECK(dtype.has_value() && !dtype->IsFloat())
        << "The operator " << op_name << " requires valid_shape[" << i
        << "] to be an integer extent, but got "
        << (dtype.has_value() ? dtype->ToString() : valid->elements_[i]->GetType()->TypeName());
  }

  bool transpose = false;
  for (const auto& [k, v] : kwargs) {
    if (k == "transpose") transpose = AnyCast<bool>(v, "transpose");
  }
  if (transpose) {
    for (const auto& elem : valid->elements_) {
      CHECK(As<ConstInt>(elem))
          << "The operator " << op_name
          << " does not support a dynamic valid_shape together with transpose=True (the DN2NZ per-row "
             "path would need a runtime column extent on a boxed NZ tile). Use a static valid_shape "
             "with transpose=True, or gather without transpose and read the operand with "
             "matmul(b_trans=True).";
    }
  }

  // Report every violation at once rather than only the first — the messages
  // already carry the op name and dimension index.
  const auto errors = ValidateValidShapeBounds(valid->elements_, shapes->elements_, op_name);
  if (errors.empty()) return;
  std::ostringstream msg;
  for (size_t i = 0; i < errors.size(); ++i) {
    if (i > 0) msg << "; ";
    msg << errors[i].message;
  }
  throw ValueError(msg.str());
}

void CheckMatmulInitCond(const std::vector<ExprPtr>& args, size_t index, const std::string& op_name) {
  if (args.size() <= index) return;
  const auto& cond = args[index];
  auto scalar_type = As<ScalarType>(cond->GetType());
  CHECK(scalar_type) << "The operator " << op_name << " requires init_cond to be a boolean scalar, but got "
                     << cond->GetType()->TypeName();
  CHECK(scalar_type->dtype_ == DataType::BOOL)
      << "The operator " << op_name << " requires init_cond to have dtype BOOL, but got "
      << scalar_type->dtype_.ToString()
      << ". Write a comparison such as `k == 0` rather than passing the index itself.";
}

void CheckReductionInputNonEmpty(const std::vector<ExprPtr>& valid, const std::string& op_name,
                                 const Span& span) {
  for (size_t i = 0; i < valid.size(); ++i) {
    // Only a *provable* zero rejects; an unproved symbolic extent is accepted.
    CHECK_SPAN(!IsProvablyEmptyExtent(valid[i]), span)
        << op_name << ": input valid extent on axis " << i << " is 0 (valid_shape " << FormatShape(valid)
        << "), so the reduction has no real data to consume. The backend reduction kernels require a "
           "non-empty valid region on every axis and assert on an empty one, and an empty region also "
           "leaves max/min with no value to return. Widen the valid region, or guard the reduction so "
           "it does not run when the axis can be empty.";
  }
}

// ============================================================================
// Tuple operand decoding
// ============================================================================

std::vector<ExprPtr> ExtractTupleElements(const ExprPtr& tuple_expr, size_t rank) {
  if (auto make_tuple = As<MakeTuple>(tuple_expr)) {
    return make_tuple->elements_;
  }
  std::vector<ExprPtr> elements;
  elements.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    elements.emplace_back(
        std::make_shared<TupleGetItemExpr>(tuple_expr, static_cast<int>(i), tuple_expr->span_));
  }
  return elements;
}

// ============================================================================
// Window-read valid-region intersection
// ============================================================================

const std::vector<ExprPtr>& GetEffectiveTensorValidShape(const TensorType& type) {
  if (type.tensor_view_ && !type.tensor_view_->valid_shape.empty()) {
    return type.tensor_view_->valid_shape;
  }
  return type.shape_;
}

namespace {

/// Zero, in the dtype the analyzer compares extents in.
ExprPtr IndexZero() {
  static const ExprPtr zero = std::make_shared<ConstInt>(0, DataType::INDEX, Span::unknown());
  return zero;
}

/// Fold an expression through the arithmetic analyzer so constants collapse.
ExprPtr FoldExtent(const ExprPtr& expr) {
  thread_local arith::Analyzer analyzer;
  return analyzer.Simplify(expr);
}

/// Return `lhs` when it is provably the smaller of the two, `rhs` when it is
/// provably the smaller, and a folded `min` only when neither is settled.
ExprPtr MinExtent(const ExprPtr& lhs, const ExprPtr& rhs, const Span& span) {
  if (ProveValidExtentLessEqual(lhs, rhs) == ProofResult::kTrue) {
    return lhs;
  }
  if (ProveValidExtentLessEqual(rhs, lhs) == ProofResult::kTrue) {
    return rhs;
  }
  return FoldExtent(MakeMin(lhs, rhs, span));
}

/// The extent of dimension `i` that a read must keep inside its source.
///
/// Under kExactWindow nothing trims the window, so all of it has to fit,
/// however small an explicit valid_shape may be.
///
/// Under kClampedWindow the window is trimmed for us — codegen clamps a
/// tensor.slice view to the parent, and a tile.load DMA fetches only the valid
/// extent — so the window may deliberately overhang and what has to fit is the
/// extent actually read: the explicit request when the caller made one (a padded
/// fixed-width window with a declared valid_shape is the standard idiom), and the
/// window itself when they did not, since then the read implicitly claims all of it.
const ExprPtr& BoundsReach(const WindowReadValidShapeParams& p, size_t i) {
  if (p.kind == WindowReadKind::kExactWindow || p.requested_valid.empty()) {
    return p.window[i];
  }
  return p.requested_valid[i];
}

/// Whether the analyzer can do integer arithmetic on this expression at all.
/// Extents reaching an operator are normally integer scalars, but an operand may
/// be an arbitrary expression (a tuple, say) that no proof obligation is defined
/// over; such a dimension is simply undecidable rather than a bounds violation.
bool IsIntegerScalarExpr(const ExprPtr& expr) {
  if (!expr) {
    return false;
  }
  auto scalar_type = As<ScalarType>(expr->GetType());
  return scalar_type && scalar_type->dtype_.IsInt();
}

void CheckWindowReadRanks(const WindowReadValidShapeParams& p) {
  const size_t rank = p.window.size();
  CHECK_SPAN(p.source_physical.size() == rank, p.span)
      << p.op_name << " requires the window and the source to have the same rank, but got window rank "
      << rank << " " << FormatShape(p.window) << " and source rank " << p.source_physical.size() << " "
      << FormatShape(p.source_physical);
  CHECK_SPAN(p.offsets.size() == rank, p.span)
      << p.op_name << " requires one offset per window dimension, but got " << p.offsets.size()
      << " offsets for window rank " << rank;
  CHECK_SPAN(p.source_valid.size() == rank, p.span)
      << p.op_name << " source valid_shape rank " << p.source_valid.size()
      << " does not match its shape rank " << rank;
  CHECK_SPAN(p.requested_valid.empty() || p.requested_valid.size() == rank, p.span)
      << p.op_name << " requires valid_shape to have the same rank as the window, but got valid_shape rank "
      << p.requested_valid.size() << " " << FormatShape(p.requested_valid) << " and window rank " << rank;
}

/// Enforce what a window read promises about dimension `i` before its valid
/// region is derived: the window starts inside the source, the request fits in
/// the window that holds it, and — unless the read clamps — the extent it touches
/// also ends inside the source.
void CheckWindowReadDimBounds(const WindowReadValidShapeParams& p, size_t i) {
  const ExprPtr& offset = p.offsets[i];
  const ExprPtr& source = p.source_physical[i];

  CHECK_SPAN(ProveValidExtentLessEqual(IndexZero(), offset) != ProofResult::kFalse, p.span)
      << p.op_name << " offset " << i << " is provably negative (" << PythonPrint(offset)
      << "); a window must start inside its source";

  // An explicit request also has to fit the window that holds it: `valid <= shape`
  // is the standing bounds invariant of the type this read produces. The request
  // is returned as the result whenever the source cannot be proven narrower, so
  // an oversized one would otherwise walk straight into the result type. Reject
  // what we can disprove and trust the rest, as everywhere else here. Operators
  // that require stricter scalar-kind validation enforce it in their deducers.
  if (!p.requested_valid.empty()) {
    const ExprPtr& requested = p.requested_valid[i];
    CHECK_SPAN(ProveValidExtentLessEqual(requested, p.window[i]) != ProofResult::kFalse, p.span)
        << p.op_name << " valid_shape " << i << " is " << PythonPrint(requested)
        << ", which exceeds the window extent " << PythonPrint(p.window[i])
        << "; a valid region cannot be larger than the shape that holds it";
  }

  // A non-clamping read asserts that the extent it touches stays inside the
  // source. Reject what we can disprove; trust what stays symbolic, because
  // that inequality is the operator's precondition, not a guess.
  const ExprPtr& reach = BoundsReach(p, i);
  if (!p.clamp && IsIntegerScalarExpr(offset) && IsIntegerScalarExpr(reach) && IsIntegerScalarExpr(source)) {
    const ExprPtr end = FoldExtent(MakeAdd(offset, reach, p.span));
    CHECK_SPAN(ProveValidExtentLessEqual(end, source) != ProofResult::kFalse, p.span)
        << p.op_name << " reads past the end of dimension " << i << ": offset " << PythonPrint(offset)
        << " + extent " << PythonPrint(reach) << " exceeds the source extent " << PythonPrint(source) << ". "
        << p.bounds_remedy;
  }
}

}  // namespace

std::vector<ExprPtr> InferWindowReadValidShape(const WindowReadValidShapeParams& params) {
  CheckWindowReadRanks(params);

  const size_t rank = params.window.size();
  std::vector<ExprPtr> result;
  result.reserve(rank);

  for (size_t i = 0; i < rank; ++i) {
    CheckWindowReadDimBounds(params, i);

    const ExprPtr& offset = params.offsets[i];
    const ExprPtr& window = params.window[i];
    const ExprPtr& src_valid = params.source_valid[i];
    const bool source_fully_valid = AreExprsEqual(src_valid, params.source_physical[i]);

    // available = clamp(source_valid - offset, 0, window).
    //
    // When the source is fully valid and the read is non-clamping, proof-only
    // callers may trust the precondition checked above and avoid building a
    // guard expression. Runtime-intersection callers must retain a symbolic
    // source bound: an unknown `window <= source_valid - offset` relation is
    // accepted by the verifier, but the emitted read still has to stay in-bounds.
    // Static source bounds keep the historical proof-only path so bounded
    // dynamic offsets produced by collective schedules remain codegen-friendly.
    ExprPtr available;
    const bool source_bound_is_static = As<ConstInt>(src_valid) != nullptr;
    if (source_fully_valid && !params.clamp &&
        (!params.materialize_symbolic_intersection || source_bound_is_static)) {
      available = window;
    } else {
      CHECK_SPAN(IsIntegerScalarExpr(offset), params.span)
          << params.op_name << " offset " << i << " must be an integer scalar to narrow dimension " << i
          << " against a partial source, but got " << offset->GetType()->TypeName();
      // max(src_valid, offset) - offset is equivalent to
      // max(src_valid - offset, 0), but cannot underflow when src_valid is a
      // UINT64 symbolic extent smaller than the offset.
      const ExprPtr remaining =
          ProveValidExtentEqual(offset, IndexZero()) == ProofResult::kTrue
              ? src_valid
              : FoldExtent(MakeSub(MakeMax(src_valid, offset, params.span), offset, params.span));
      available = MinExtent(remaining, window, params.span);
    }

    // result = min(requested, available).
    //
    // With no explicit request, the source's extent under the window *is* the
    // answer, guard expression and all.
    //
    // With one, proof-only callers narrow to the source's extent only when that
    // is provably the smaller of the two. An undecidable relation is otherwise
    // taken on trust because a source-valid type expression can mention a symbol
    // that is not bound in the reading function. Callers that know those symbols
    // are available at runtime may opt into materializing the exact min instead.
    if (params.requested_valid.empty()) {
      result.push_back(available);
      continue;
    }
    const ExprPtr& requested = params.requested_valid[i];
    // Most window readers preserve the historical proof-only narrowing below:
    // when ordering is unknown, keep the caller's requested extent. Remote
    // loads opt into an exact runtime min because their partition must never
    // include invalid peer-buffer elements.
    if (params.materialize_symbolic_intersection) {
      result.push_back(MinExtent(requested, available, params.span));
      continue;
    }
    const bool source_is_narrower = !AreExprsEqual(available, window) &&
                                    ProveValidExtentLessEqual(available, requested) == ProofResult::kTrue;
    result.push_back(source_is_narrower ? available : requested);
  }

  return result;
}

std::vector<ExprPtr> InferTensorSliceFullValidShape(const TensorType& source_type,
                                                    const std::vector<ExprPtr>& full_shape,
                                                    const std::vector<ExprPtr>& offsets,
                                                    const std::vector<ExprPtr>& requested_valid, bool clamp,
                                                    const Span& span) {
  if (full_shape.size() == source_type.shape_.size() && offsets.size() == full_shape.size()) {
    return InferWindowReadValidShape({
        /*source_physical=*/source_type.shape_,
        /*source_valid=*/GetEffectiveTensorValidShape(source_type),
        /*offsets=*/offsets,
        /*window=*/full_shape,
        /*requested_valid=*/requested_valid,
        /*kind=*/WindowReadKind::kClampedWindow,
        /*clamp=*/clamp,
        /*op_name=*/"tensor.slice",
        /*bounds_remedy=*/
        "Pass clamp=True -- pl.slice(x, shape, offset, clamp=True) -- to narrow the valid region to "
        "the source edge instead",
        /*span=*/span,
    });
  }

  CHECK_SPAN(requested_valid.empty() || requested_valid.size() == full_shape.size(), span)
      << "tensor.slice requires valid_shape to have the same rank as shape, but got valid_shape rank "
      << requested_valid.size() << " and shape rank " << full_shape.size();
  return requested_valid.empty() ? full_shape : requested_valid;
}

void ValidateDropDimsValidExtents(const std::vector<int64_t>& drop_dims,
                                  const std::vector<ExprPtr>& valid_shape, const std::string& op_name,
                                  const Span& span) {
  static const auto one = std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown());
  for (int64_t axis : drop_dims) {
    const auto index = static_cast<size_t>(axis);
    INTERNAL_CHECK_SPAN(index < valid_shape.size(), span)
        << "Internal error: " << op_name << " drop_dims axis " << axis
        << " is out of range for valid_shape rank " << valid_shape.size();
    const ExprPtr& extent = valid_shape[index];
    CHECK_SPAN(ProveValidExtentEqual(extent, one) == ProofResult::kTrue, span)
        << op_name << " cannot drop dimension " << axis << ": its valid extent is " << PythonPrint(extent)
        << ", which is not provably 1. Rank reduction erases an axis, so the axis must be fully valid; "
           "keep the dimension instead of dropping it";
  }
}

namespace {

// Whether two extents are provably the same. Constants decide by value first:
// ProveValidExtentEqual only compares extents of matching signedness, so a
// UINT64 valid extent (what tile.set_validshape emits) against an INDEX
// physical extent comes back kUnknown even when both are the same literal.
// Reading a full axis as partial that way rejects reshapes that map exactly --
// the same signedness caveat IsProvablyEmptyExtent documents above.
bool ExtentsProvablyEqual(const ExprPtr& lhs, const ExprPtr& rhs) {
  const auto lhs_const = GetConstantDimension(lhs);
  const auto rhs_const = GetConstantDimension(rhs);
  if (lhs_const.has_value() && rhs_const.has_value()) {
    return *lhs_const == *rhs_const;
  }
  return ProveValidExtentEqual(lhs, rhs) == ProofResult::kTrue;
}

// Align the input and target axes of a reshape that only inserts or erases
// provably-full physical unit axes, returning the mapped valid shape.
//
// Such a reshape is a coordinate-only rank change -- rows stay rows, columns
// stay columns -- so it preserves an arbitrary origin-anchored rectangle
// exactly, which the flat-prefix rule below could not. Ambiguous runs of unit
// axes are resolved by a small sequence alignment: matching equal axes is tried
// first so a partial or empty unit axis is preserved rather than erased and
// recreated as fully valid. An input unit axis may be erased only when its sole
// coordinate is provably valid; an output unit axis is inserted fully valid.
//
// `failed` memoizes the states already proven unmappable, indexed
// `input_dim * (out_rank + 1) + output_dim`. Without it the three moves make
// this a backtracking search over monotone lattice paths -- Delannoy-many,
// ~5.83^rank -- and the miss path is not rare: it is the ordinary fall-through
// to the flat-prefix rule. Since the answer at a state depends only on that
// state, caching failures bounds the walk at one visit per state.
std::optional<std::vector<ExprPtr>> MapUnitAxisRankChange(const std::vector<ExprPtr>& src_valid,
                                                          const std::vector<ExprPtr>& in_shape,
                                                          const std::vector<ExprPtr>& new_shape,
                                                          size_t input_dim, size_t output_dim,
                                                          std::vector<char>* failed) {
  const size_t state = input_dim * (new_shape.size() + 1) + output_dim;
  INTERNAL_CHECK(state < failed->size())
      << "Internal error: reshape unit-axis memo table is sized " << failed->size() << ", need > " << state;
  if ((*failed)[state]) return std::nullopt;

  if (input_dim == in_shape.size() && output_dim == new_shape.size()) {
    return std::vector<ExprPtr>{};
  }

  // Match two axes carrying the same extent. Tried first so an axis that is
  // only partially valid keeps its own extent instead of being erased and
  // recreated as fully valid.
  if (input_dim < in_shape.size() && output_dim < new_shape.size() &&
      ProveValidExtentEqual(in_shape[input_dim], new_shape[output_dim]) == ProofResult::kTrue) {
    if (auto tail =
            MapUnitAxisRankChange(src_valid, in_shape, new_shape, input_dim + 1, output_dim + 1, failed)) {
      tail->insert(tail->begin(), src_valid[input_dim]);
      return tail;
    }
  }
  // Erase an input unit axis -- lossless only when its sole coordinate is valid.
  if (input_dim < in_shape.size() && IsConstValue(in_shape[input_dim], 1) &&
      ExtentsProvablyEqual(src_valid[input_dim], in_shape[input_dim])) {
    if (auto tail =
            MapUnitAxisRankChange(src_valid, in_shape, new_shape, input_dim + 1, output_dim, failed)) {
      return tail;
    }
  }
  // Insert a target unit axis -- one coordinate, and it holds real data.
  if (output_dim < new_shape.size() && IsConstValue(new_shape[output_dim], 1)) {
    if (auto tail =
            MapUnitAxisRankChange(src_valid, in_shape, new_shape, input_dim, output_dim + 1, failed)) {
      tail->insert(tail->begin(), new_shape[output_dim]);
      return tail;
    }
  }
  (*failed)[state] = 1;
  return std::nullopt;
}

}  // namespace

std::vector<ExprPtr> ComputeReshapeValidShape(const std::vector<ExprPtr>& src_valid,
                                              const std::vector<ExprPtr>& in_shape,
                                              const std::vector<ExprPtr>& new_shape,
                                              bool row_major_contiguous, const Span& span,
                                              const std::string& op_name) {
  INTERNAL_CHECK_SPAN(src_valid.size() == in_shape.size(), span)
      << "Internal error: " << op_name << " source valid_shape rank (" << src_valid.size()
      << ") must match the source shape rank (" << in_shape.size()
      << "); callers resolve the valid shape through GetValidShape";
  CHECK_SPAN(!src_valid.empty() && !new_shape.empty(), span)
      << op_name << ": reshape validity mapping requires non-empty input and output ranks";

  // (1) A fully valid source stays fully valid. Returning the target shape
  // verbatim keeps an unpadded program byte-identical to what it deduced before
  // this rule existed -- the type constructor canonicalizes the redundant full
  // valid_shape away, so no view survives.
  bool fully_valid = true;
  for (size_t i = 0; i < src_valid.size(); ++i) {
    if (!ExtentsProvablyEqual(src_valid[i], in_shape[i])) {
      fully_valid = false;
      break;
    }
  }
  if (fully_valid) {
    return new_shape;
  }

  // (2) The empty set stays empty under every reshape. This is settled before
  // the prefix proof below because a box such as [1, 0, N] is not a flat prefix
  // by that syntactic form, yet it denotes no cells and so has an exact
  // representation in every target shape.
  if (std::any_of(src_valid.begin(), src_valid.end(), IsProvablyEmptyExtent)) {
    return std::vector<ExprPtr>(new_shape.size(), IndexZero());
  }

  // (3) A pure rank change over provably-full unit axes preserves an arbitrary
  // rectangle, which the flat-prefix rule cannot see.
  std::vector<char> unit_axis_failed((in_shape.size() + 1) * (new_shape.size() + 1), 0);
  if (auto unit_mapped = MapUnitAxisRankChange(src_valid, in_shape, new_shape, 0, 0, &unit_axis_failed)) {
    return *unit_mapped;
  }

  // (4) Otherwise the region has to occupy a contiguous flat prefix of the
  // buffer, so that some rectangle of the target shape spans exactly the same
  // cells. Everything below walks flat positions in row-major order, so a
  // source stored any other way would be measured against the wrong offsets:
  // a col_major [2, 3] valid [1, 3] really occupies flat {0, 2, 4}, and the
  // row-major reading would hand back a box covering {0, 1, 2} -- marking two
  // padding elements as real. Reject instead of guessing.
  CHECK_SPAN(row_major_contiguous, span)
      << op_name << ": cannot reshape " << FormatShape(in_shape) << " to " << FormatShape(new_shape)
      << " because only part of it holds real data (valid_shape " << FormatShape(src_valid)
      << ") and its elements are not stored row-major, so the real data does not occupy a contiguous "
         "run that the new shape can describe. Reshape the full extent and narrow afterwards, or copy "
         "the real data out first (pl.slice / pl.store).";

  // Leading axes pinned to a single valid coordinate contribute nothing to the
  // extent; the first remaining axis carries the prefix's one free extent, and
  // every axis below it must be full.
  const size_t input_rank = src_valid.size();
  size_t free_dim = 0;
  while (free_dim + 1 < input_rank && IsConstValue(src_valid[free_dim], 1)) {
    ++free_dim;
  }

  for (size_t i = free_dim + 1; i < input_rank; ++i) {
    const bool full_axis = ExtentsProvablyEqual(src_valid[i], in_shape[i]);
    const ProofResult full = ProveValidExtentEqual(src_valid[i], in_shape[i]);
    CHECK_SPAN(full_axis, span)
        << op_name << ": cannot reshape " << FormatShape(in_shape) << " to " << FormatShape(new_shape)
        << " because only part of it holds real data (valid_shape " << FormatShape(src_valid)
        << "). Dimension " << i << " is valid for " << PythonPrint(src_valid[i]) << " of "
        << PythonPrint(in_shape[i])
        << (full == ProofResult::kUnknown ? " (a runtime extent that cannot be proven equal)" : "")
        << ", so the real data is scattered across the buffer rather than filling it from the start, and "
           "no region of the new shape describes the same cells. Reshape the full extent and narrow "
           "afterwards, or copy the real data out first (pl.slice / pl.store).";
  }

  // The prefix is measured in elements, so every extent it spans has to be a
  // compile-time constant.
  int64_t trailing_volume = 1;
  for (size_t i = free_dim + 1; i < input_rank; ++i) {
    const auto extent = GetConstantDimension(in_shape[i]);
    CHECK_SPAN(extent.has_value(), span)
        << op_name << ": cannot reshape a partially-valid " << FormatShape(in_shape) << " because dimension "
        << i << " has the runtime extent " << PythonPrint(in_shape[i]) << ". Mapping the real data into "
        << FormatShape(new_shape)
        << " needs its size at compile time; use a static shape, or reshape before narrowing.";
    trailing_volume *= *extent;
  }
  std::vector<int64_t> target(new_shape.size());
  for (size_t i = 0; i < new_shape.size(); ++i) {
    const auto extent = GetConstantDimension(new_shape[i]);
    CHECK_SPAN(extent.has_value(), span)
        << op_name << ": cannot reshape a partially-valid " << FormatShape(in_shape) << " into "
        << FormatShape(new_shape) << " because target dimension " << i << " has the runtime extent "
        << PythonPrint(new_shape[i])
        << ". Mapping the real data needs the target size at compile time; use a static shape, or "
           "reshape before narrowing.";
    target[i] = *extent;
  }

  // Row-major volume below each target axis: the number of elements one step
  // along that axis advances by.
  std::vector<int64_t> suffix(target.size(), 1);
  for (size_t i = target.size(); i-- > 0;) {
    suffix[i] = i + 1 < target.size() ? suffix[i + 1] * target[i + 1] : 1;
  }

  // The result box is full below its own free axis and pinned to one coordinate
  // above it -- the target-shape spelling of "a flat prefix".
  const ExprPtr pinned = std::make_shared<ConstInt>(1, DataType::INDEX, span);
  auto build_box = [&](size_t output_free_dim, const ExprPtr& free_extent) {
    std::vector<ExprPtr> output(new_shape.size());
    for (size_t i = 0; i < new_shape.size(); ++i) {
      if (i < output_free_dim) {
        output[i] = pinned;
      } else if (i == output_free_dim) {
        output[i] = free_extent;
      } else {
        output[i] = new_shape[i];
      }
    }
    return output;
  };

  const ExprPtr& free_valid = src_valid[free_dim];
  if (const auto extent = GetConstantDimension(free_valid)) {
    // A static prefix maps onto the outermost target axis whose suffix volume
    // divides it -- the prefix is then a whole number of that axis's steps.
    const int64_t prefix_elements = *extent * trailing_volume;
    for (size_t i = 0; i < new_shape.size(); ++i) {
      if (suffix[i] == 0 || prefix_elements % suffix[i] != 0) continue;
      const int64_t output_extent = prefix_elements / suffix[i];
      if (output_extent <= target[i]) {
        return build_box(i, std::make_shared<ConstInt>(output_extent, DataType::INDEX, span));
      }
    }
    CHECK_SPAN(false, span)
        << op_name << ": cannot reshape " << FormatShape(in_shape) << " to " << FormatShape(new_shape)
        << " because only " << prefix_elements << " of its "
        << (prefix_elements == 1 ? "element" : "elements") << " hold real data (valid_shape "
        << FormatShape(src_valid) << "), and no region of " << FormatShape(new_shape)
        << " covers exactly those " << prefix_elements
        << " elements -- they do not fill a whole number of rows there. Pick a target shape whose "
           "trailing dimensions divide it, or copy the real data out first (pl.slice / pl.store).";
  }

  // A dynamic prefix cannot be divided, so it survives only on a target axis
  // whose step is exactly the input's trailing volume: the free extent then
  // carries over unchanged. That axis has to have room for the whole free
  // dimension, which is knowable only if the free dimension is itself static --
  // a requirement of this branch alone, not of the static one above.
  const auto free_physical = GetConstantDimension(in_shape[free_dim]);
  if (free_physical.has_value()) {
    for (size_t i = 0; i < new_shape.size(); ++i) {
      if (suffix[i] == trailing_volume && *free_physical <= target[i]) {
        return build_box(i, free_valid);
      }
    }
  }
  CHECK_SPAN(false, span)
      << op_name << ": cannot reshape " << FormatShape(in_shape) << " to " << FormatShape(new_shape)
      << " because its real data extends a runtime number of rows (" << PythonPrint(free_valid)
      << ") and no dimension of " << FormatShape(new_shape) << " has the matching row size of "
      << trailing_volume
      << " elements. Keep that dimension intact in the target shape, or copy the real data out first "
         "(pl.slice / pl.store).";
  return {};
}

// ============================================================================
// Write valid-region unions (assemble / store)
// ============================================================================

namespace {

/// Dual of `MinExtent`: the provably larger operand when the ordering is settled,
/// and a folded `max` only when it is not.
ExprPtr MaxExtent(const ExprPtr& lhs, const ExprPtr& rhs, const Span& span) {
  if (ProveValidExtentLessEqual(lhs, rhs) == ProofResult::kTrue) {
    return rhs;
  }
  if (ProveValidExtentLessEqual(rhs, lhs) == ProofResult::kTrue) {
    return lhs;
  }
  return FoldExtent(MakeMax(lhs, rhs, span));
}

/// Whether `valid` names the whole of `physical`, dimension by dimension.
///
/// Canonicalization stores redundant full validity as an absent view, so this is
/// usually the identity `GetValidShape` returned; an explicit spelling that the
/// analyzer can still settle is accepted too.
bool IsFullyValid(const std::vector<ExprPtr>& valid, const std::vector<ExprPtr>& physical) {
  for (size_t i = 0; i < valid.size(); ++i) {
    if (!AreExprsEqual(valid[i], physical[i]) &&
        ProveValidExtentEqual(valid[i], physical[i]) != ProofResult::kTrue) {
      return false;
    }
  }
  return true;
}

/// Whether every offset is provably zero, i.e. the written region is itself
/// origin-anchored and can stand alone as a valid shape.
bool IsOriginAnchored(const std::vector<ExprPtr>& offsets) {
  for (const auto& offset : offsets) {
    if (!IsProvablyEmptyExtent(offset)) {
      return false;
    }
  }
  return true;
}

std::string FormatDimensionList(const std::vector<size_t>& dims) {
  std::string out;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) {
      out += i + 1 == dims.size() ? " and " : ", ";
    }
    out += std::to_string(dims[i]);
  }
  return out;
}

void CheckWriteUnionRanks(const WriteValidShapeUnionParams& p) {
  const size_t rank = p.target_physical.size();
  CHECK_SPAN(p.source_physical.size() == rank, p.span)
      << p.op_name << " requires the source and the target to have the same rank, but got source rank "
      << p.source_physical.size() << " " << FormatShape(p.source_physical) << " and target rank " << rank
      << " " << FormatShape(p.target_physical);
  CHECK_SPAN(p.offsets.size() == rank, p.span)
      << p.op_name << " requires one offset per target dimension, but got " << p.offsets.size()
      << " offsets for target rank " << rank;
  CHECK_SPAN(p.target_valid.size() == rank, p.span)
      << p.op_name << " target valid_shape rank " << p.target_valid.size()
      << " does not match its shape rank " << rank;
  CHECK_SPAN(p.source_valid.size() == rank, p.span)
      << p.op_name << " source valid_shape rank " << p.source_valid.size()
      << " does not match its shape rank " << rank;
}

/// The extent of dimension `i` the write must keep inside the target.
///
/// Under `kExactSubview` the whole physical source lands, so all of it has to fit,
/// however small its valid region. Under `kValidRegionTransfer` only the valid
/// region moves, so a larger physical source allocation is free.
const ExprPtr& WriteReach(const WriteValidShapeUnionParams& p, size_t i) {
  return p.kind == WriteBoundsKind::kExactSubview ? p.source_physical[i] : p.source_valid[i];
}

/// Enforce what a write promises about dimension `i` before its union is derived:
/// it starts inside the target, and the extent it touches also ends inside it.
///
/// Provable violations reject; relations that stay symbolic are taken on trust,
/// exactly as for a non-clamping window read, because that inequality is the
/// operator's precondition rather than a guess.
void CheckWriteDimBounds(const WriteValidShapeUnionParams& p, size_t i) {
  const ExprPtr& offset = p.offsets[i];
  const ExprPtr& target = p.target_physical[i];

  CHECK_SPAN(ProveValidExtentLessEqual(IndexZero(), offset) != ProofResult::kFalse, p.span)
      << p.op_name << " offset " << i << " is provably negative (" << PythonPrint(offset)
      << "); a write must start inside its target";

  const ExprPtr& reach = WriteReach(p, i);
  if (IsIntegerScalarExpr(offset) && IsIntegerScalarExpr(reach) && IsIntegerScalarExpr(target)) {
    const ExprPtr end = FoldExtent(MakeAdd(offset, reach, p.span));
    CHECK_SPAN(ProveValidExtentLessEqual(end, target) != ProofResult::kFalse, p.span)
        << p.op_name << " writes past the end of dimension " << i << ": offset " << PythonPrint(offset)
        << " + extent " << PythonPrint(reach) << " exceeds the target extent " << PythonPrint(target)
        << (p.bounds_remedy.empty() ? "" : ". ") << p.bounds_remedy;
  }
}

/// The far edge `offset[i] + source_valid[i]` of the written region.
ExprPtr WriteFarEdge(const WriteValidShapeUnionParams& p, size_t i) {
  const ExprPtr& offset = p.offsets[i];
  const ExprPtr& extent = p.source_valid[i];
  if (IsProvablyEmptyExtent(offset)) {
    return extent;
  }
  CHECK_SPAN(IsIntegerScalarExpr(offset) && IsIntegerScalarExpr(extent), p.span)
      << p.op_name << " needs an integer scalar offset and valid extent to place dimension " << i
      << " against a partially valid target, but got offset " << offset->GetType()->TypeName()
      << " and extent " << extent->GetType()->TypeName();
  return FoldExtent(MakeAdd(offset, extent, p.span));
}

}  // namespace

std::vector<ExprPtr> InferWriteValidShapeUnion(const WriteValidShapeUnionParams& params) {
  CheckWriteUnionRanks(params);
  const size_t rank = params.target_physical.size();
  for (size_t i = 0; i < rank; ++i) {
    CheckWriteDimBounds(params, i);
  }

  const std::vector<ExprPtr>& target_valid = params.target_valid;
  const std::vector<ExprPtr>& source_valid = params.source_valid;

  // An empty written region leaves the target exactly as it was.
  for (size_t i = 0; i < rank; ++i) {
    if (IsProvablyEmptyExtent(source_valid[i])) {
      return target_valid;
    }
  }

  // An empty target holds nothing to union with, so the result is the written
  // region alone — which is a valid shape only if it starts at the origin.
  for (size_t i = 0; i < rank; ++i) {
    if (IsProvablyEmptyExtent(target_valid[i])) {
      CHECK_SPAN(IsOriginAnchored(params.offsets), params.span)
          << params.op_name << " writes at offset " << FormatShape(params.offsets)
          << " into a target whose valid region is empty (dimension " << i
          << " has extent 0), which would leave a region that does not start at the origin. A valid "
             "shape names one origin-anchored rectangle, so initialize an empty target at offset 0";
      return source_valid;
    }
  }

  // A fully valid target stays fully valid: the physical bounds asserted above
  // are exactly the containment proof, including for the symbolic offsets that
  // dominate real code, so this must not be re-derived from the weaker
  // per-dimension proofs below.
  if (IsFullyValid(target_valid, params.target_physical)) {
    return target_valid;
  }

  // Dimensions the write may push past the target's valid region. Everything it
  // cannot reach is already covered, so an empty set means the write lands inside.
  std::vector<ExprPtr> far_edge;
  std::vector<size_t> growing;
  far_edge.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    far_edge.push_back(WriteFarEdge(params, i));
    if (ProveValidExtentLessEqual(far_edge[i], target_valid[i]) != ProofResult::kTrue) {
      growing.push_back(i);
    }
  }
  if (growing.empty()) {
    return target_valid;
  }

  // The written region swallows the target whole, and starts at the origin, so it
  // stands alone. This is the multi-dimensional overwrite the single-axis rule
  // below cannot express.
  if (IsOriginAnchored(params.offsets)) {
    bool covers_target = true;
    for (size_t i = 0; i < rank && covers_target; ++i) {
      covers_target = ProveValidExtentLessEqual(target_valid[i], source_valid[i]) == ProofResult::kTrue;
    }
    if (covers_target) {
      return source_valid;
    }
  }

  // Otherwise the only representable growth is along a single axis: the new slab
  // must abut what is already there, and must span every other axis exactly, or
  // the union is an L-shape that no valid shape can name.
  CHECK_SPAN(growing.size() == 1, params.span)
      << params.op_name << " grows the valid region along dimensions " << FormatDimensionList(growing)
      << " at once, whose union with the target is an L-shape rather than one origin-anchored "
         "rectangle. Target valid "
      << FormatShape(target_valid) << ", writing " << FormatShape(source_valid) << " at offset "
      << FormatShape(params.offsets);

  const size_t axis = growing.front();
  CHECK_SPAN(ProveValidExtentLessEqual(params.offsets[axis], target_valid[axis]) == ProofResult::kTrue,
             params.span)
      << params.op_name << " leaves a gap in dimension " << axis << ": the write starts at "
      << PythonPrint(params.offsets[axis]) << ", which is not provably at or before the target valid extent "
      << PythonPrint(target_valid[axis])
      << ". A valid shape names one contiguous origin-anchored rectangle, so a write that grows it must "
         "abut the region already there";

  for (size_t i = 0; i < rank; ++i) {
    if (i == axis) {
      continue;
    }
    CHECK_SPAN(IsProvablyEmptyExtent(params.offsets[i]), params.span)
        << params.op_name << " grows dimension " << axis << ", so it must span dimension " << i
        << " from the origin, but it starts at " << PythonPrint(params.offsets[i])
        << ". The added region would be narrower than the region it extends, making the union an L-shape";
    CHECK_SPAN(ProveValidExtentEqual(source_valid[i], target_valid[i]) == ProofResult::kTrue, params.span)
        << params.op_name << " grows dimension " << axis << ", so its extent in dimension " << i << " ("
        << PythonPrint(source_valid[i]) << ") must provably equal the target valid extent ("
        << PythonPrint(target_valid[i])
        << "). The added region would otherwise not line up with the region it extends, making the union "
           "an L-shape";
  }

  std::vector<ExprPtr> result = target_valid;
  result[axis] = MinExtent(MaxExtent(target_valid[axis], far_edge[axis], params.span),
                           params.target_physical[axis], params.span);
  return result;
}

// ============================================================================
// Slice rank-reduction (drop_dims) helpers
// ============================================================================

std::vector<int64_t> ParseSliceDropDims(const ExprPtr& drop_dims_arg, const std::vector<ExprPtr>& full_shape,
                                        const std::string& op_name) {
  if (!drop_dims_arg) {
    return {};
  }
  auto tuple = As<MakeTuple>(drop_dims_arg);
  CHECK(tuple) << op_name << " drop_dims must be a MakeTuple of compile-time int constants";

  std::vector<int64_t> axes;
  axes.reserve(tuple->elements_.size());
  std::vector<bool> seen(full_shape.size(), false);
  for (size_t i = 0; i < tuple->elements_.size(); ++i) {
    auto const_int = As<ConstInt>(tuple->elements_[i]);
    CHECK(const_int) << op_name << " drop_dims element " << i << " must be a compile-time int constant";
    int64_t axis = const_int->value_;
    CHECK(axis >= 0 && axis < static_cast<int64_t>(full_shape.size()))
        << op_name << " drop_dims index " << axis << " out of range for rank " << full_shape.size();
    CHECK(!seen[static_cast<size_t>(axis)]) << op_name << " drop_dims index " << axis << " is repeated";
    seen[static_cast<size_t>(axis)] = true;
    auto dim = GetConstantDimension(full_shape[static_cast<size_t>(axis)]);
    CHECK(dim.has_value() && *dim == 1)
        << op_name << " drop_dims index " << axis
        << " must select a static unit dimension (rank reduction only erases size-1 dims), but dim " << axis
        << " is " << (dim.has_value() ? std::to_string(*dim) : std::string("dynamic"));
    axes.push_back(axis);
  }
  std::sort(axes.begin(), axes.end());
  return axes;
}

std::vector<ExprPtr> ApplyDropDims(const std::vector<ExprPtr>& shape, const std::vector<int64_t>& drop_dims) {
  if (drop_dims.empty()) {
    return shape;
  }
  std::vector<bool> drop(shape.size(), false);
  for (int64_t d : drop_dims) {
    if (d >= 0 && d < static_cast<int64_t>(shape.size())) {
      drop[static_cast<size_t>(d)] = true;
    }
  }
  std::vector<ExprPtr> result;
  result.reserve(shape.size() - drop_dims.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    if (!drop[i]) {
      result.push_back(shape[i]);
    }
  }
  return result;
}

// ============================================================================
// Cross-function call return type deduction
// ============================================================================

namespace {

using TypeVarMap = std::unordered_map<const Var*, ExprPtr>;

struct CallTypeBindingConstraint {
  VarPtr var;
  ExprPtr existing;
  ExprPtr candidate;
  std::string context;
};

void BindCallTypeVar(const VarPtr& var, const ExprPtr& value, const std::string& context, TypeVarMap& var_map,
                     std::vector<CallTypeBindingConstraint>& constraints) {
  // A callee placeholder can also appear verbatim in the caller's annotation.
  // Treat that as an uninformative unification constraint so a later concrete
  // actual can refine the placeholder.
  if (var.get() == value.get()) return;

  auto [it, inserted] = var_map.emplace(var.get(), value);
  if (inserted) return;

  constraints.push_back({var, it->second, value, context});
}

bool CanDecomposeCallExprPattern(const ExprPtr& pattern, const ExprPtr& value) {
  if (!pattern || !value) return false;
  if (As<Var>(pattern)) return true;
  if (structural_equal(pattern, value) || ProveValidExtentEqual(pattern, value) == ProofResult::kTrue) {
    return true;
  }
  if (pattern->GetKind() != value->GetKind()) return false;

  auto pattern_binary = std::dynamic_pointer_cast<const BinaryExpr>(pattern);
  auto value_binary = std::dynamic_pointer_cast<const BinaryExpr>(value);
  if (pattern_binary && value_binary) {
    return CanDecomposeCallExprPattern(pattern_binary->left_, value_binary->left_) &&
           CanDecomposeCallExprPattern(pattern_binary->right_, value_binary->right_);
  }

  auto pattern_unary = std::dynamic_pointer_cast<const UnaryExpr>(pattern);
  auto value_unary = std::dynamic_pointer_cast<const UnaryExpr>(value);
  return pattern_unary && value_unary &&
         CanDecomposeCallExprPattern(pattern_unary->operand_, value_unary->operand_);
}

void CollectCallExprBindings(const ExprPtr& pattern, const ExprPtr& value, const std::string& context,
                             TypeVarMap& var_map, std::vector<CallTypeBindingConstraint>& constraints) {
  if (!pattern || !value) return;
  if (auto var = As<Var>(pattern)) {
    BindCallTypeVar(var, value, context, var_map, constraints);
    return;
  }

  // Composite parameter metadata can bind variables when the actual metadata
  // has a compatible expression structure. Check the whole pattern before
  // recording any bindings so a mismatch in a non-variable operand cannot
  // leave behind a partial, incorrect binding. A direct binding discovered
  // elsewhere still substitutes through differently-shaped return metadata.
  if (!CanDecomposeCallExprPattern(pattern, value)) return;
  auto pattern_binary = std::dynamic_pointer_cast<const BinaryExpr>(pattern);
  auto value_binary = std::dynamic_pointer_cast<const BinaryExpr>(value);
  if (pattern_binary && value_binary) {
    CollectCallExprBindings(pattern_binary->left_, value_binary->left_, context + " left operand", var_map,
                            constraints);
    CollectCallExprBindings(pattern_binary->right_, value_binary->right_, context + " right operand", var_map,
                            constraints);
    return;
  }
  auto pattern_unary = std::dynamic_pointer_cast<const UnaryExpr>(pattern);
  auto value_unary = std::dynamic_pointer_cast<const UnaryExpr>(value);
  if (pattern_unary && value_unary) {
    CollectCallExprBindings(pattern_unary->operand_, value_unary->operand_, context + " operand", var_map,
                            constraints);
  }
}

void CollectCallExprVectorBindings(const std::vector<ExprPtr>& patterns, const std::vector<ExprPtr>& values,
                                   const std::string& context, TypeVarMap& var_map,
                                   std::vector<CallTypeBindingConstraint>& constraints) {
  const size_t count = std::min(patterns.size(), values.size());
  for (size_t i = 0; i < count; ++i) {
    CollectCallExprBindings(patterns[i], values[i], context + "[" + std::to_string(i) + "]", var_map,
                            constraints);
  }
}

const std::vector<ExprPtr>& GetEffectiveTileValidShape(const TileType& type) {
  if (type.tile_view_ && !type.tile_view_->valid_shape.empty()) {
    return type.tile_view_->valid_shape;
  }
  return type.shape_;
}

void CollectCallTypeBindings(const TypePtr& pattern, const TypePtr& value, const std::string& context,
                             TypeVarMap& var_map, std::vector<CallTypeBindingConstraint>& constraints) {
  if (!pattern || !value) return;

  if (auto pattern_tuple = As<TupleType>(pattern)) {
    auto value_tuple = As<TupleType>(value);
    if (!value_tuple) return;
    const size_t count = std::min(pattern_tuple->types_.size(), value_tuple->types_.size());
    for (size_t i = 0; i < count; ++i) {
      CollectCallTypeBindings(pattern_tuple->types_[i], value_tuple->types_[i],
                              context + " tuple element[" + std::to_string(i) + "]", var_map, constraints);
    }
    return;
  }

  if (auto pattern_tensor = AsTensorTypeLike(pattern)) {
    auto value_tensor = AsTensorTypeLike(value);
    if (!value_tensor) return;
    CollectCallExprVectorBindings(pattern_tensor->shape_, value_tensor->shape_, context + " physical shape",
                                  var_map, constraints);
    CollectCallExprVectorBindings(GetEffectiveTensorValidShape(*pattern_tensor),
                                  GetEffectiveTensorValidShape(*value_tensor), context + " valid shape",
                                  var_map, constraints);
    if (pattern_tensor->tensor_view_ && value_tensor->tensor_view_) {
      CollectCallExprVectorBindings(pattern_tensor->tensor_view_->stride, value_tensor->tensor_view_->stride,
                                    context + " tensor stride", var_map, constraints);
    }
    return;
  }

  auto pattern_tile = As<TileType>(pattern);
  auto value_tile = As<TileType>(value);
  if (!pattern_tile || !value_tile) return;
  CollectCallExprVectorBindings(pattern_tile->shape_, value_tile->shape_, context + " physical shape",
                                var_map, constraints);
  CollectCallExprVectorBindings(GetEffectiveTileValidShape(*pattern_tile),
                                GetEffectiveTileValidShape(*value_tile), context + " valid shape", var_map,
                                constraints);
  if (pattern_tile->tile_view_ && value_tile->tile_view_) {
    CollectCallExprVectorBindings(pattern_tile->tile_view_->stride, value_tile->tile_view_->stride,
                                  context + " tile stride", var_map, constraints);
    CollectCallExprBindings(pattern_tile->tile_view_->start_offset, value_tile->tile_view_->start_offset,
                            context + " tile start_offset", var_map, constraints);
  }
}

TypePtr SubstituteCallReturnType(const TypePtr& type, const TypeVarMap& var_map) {
  if (!type) return type;
  if (auto tuple = As<TupleType>(type)) {
    std::vector<TypePtr> elements;
    elements.reserve(tuple->types_.size());
    bool changed = false;
    for (const auto& element : tuple->types_) {
      auto new_element = SubstituteCallReturnType(element, var_map);
      if (new_element.get() != element.get()) changed = true;
      elements.push_back(std::move(new_element));
    }
    if (!changed) return type;
    return std::make_shared<TupleType>(std::move(elements));
  }

  const auto memref = GetTypeMemRef(type);
  return CloneTypeWithMemRefAndRemapExprs(
      type, memref, [&var_map](const ExprPtr& expr) { return transform_utils::Substitute(expr, var_map); });
}

}  // namespace

std::vector<TypePtr> DeduceCallReturnType(const std::vector<VarPtr>& callee_params,
                                          const std::vector<ExprPtr>& args,
                                          const std::vector<TypePtr>& return_types) {
  if (return_types.empty()) return return_types;
  CHECK(callee_params.size() == args.size())
      << "DeduceCallReturnType: callee_params size (" << callee_params.size() << ") must match args size ("
      << args.size() << ")";

  TypeVarMap var_map;
  std::vector<CallTypeBindingConstraint> constraints;
  for (size_t i = 0; i < callee_params.size(); ++i) {
    if (!callee_params[i] || !args[i]) continue;
    CollectCallTypeBindings(callee_params[i]->GetType(), args[i]->GetType(), "argument " + std::to_string(i),
                            var_map, constraints);
  }
  if (var_map.empty()) return return_types;

  // Validate repeated bindings only after all arguments have contributed.
  // A constraint may mention another callee placeholder that is bound by a
  // later argument (for example STAGED = NR * 64, then NR = world_size()).
  for (const auto& constraint : constraints) {
    if (structural_equal(constraint.existing, constraint.candidate)) continue;
    auto existing = transform_utils::Substitute(constraint.existing, var_map);
    auto candidate = transform_utils::Substitute(constraint.candidate, var_map);
    if (structural_equal(existing, candidate)) continue;
    CHECK(ProveValidExtentEqual(existing, candidate) == ProofResult::kTrue)
        << "Dynamic type variable '" << constraint.var->name_hint_ << "' has conflicting bindings "
        << PythonPrint(existing) << " and " << PythonPrint(candidate) << " that are not provably equal at "
        << constraint.context << "; cross-function calls do not emit a runtime shape guard";
  }

  std::vector<TypePtr> result;
  result.reserve(return_types.size());
  for (const auto& rt : return_types) {
    result.push_back(SubstituteCallReturnType(rt, var_map));
  }
  return result;
}

}  // namespace ir
}  // namespace pypto

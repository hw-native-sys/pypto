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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_TENSOR_VIEW_SEMANTICS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_TENSOR_VIEW_SEMANTICS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"  // CHECK
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto::ir::tensor_view_semantics {

/// Compute the product of static shape dimensions; returns -1 if any dim is dynamic.
inline int64_t ComputeShapeProduct(const std::vector<ExprPtr>& shape) {
  int64_t product = 1;
  for (const auto& dim : shape) {
    auto const_dim = As<ConstInt>(dim);
    if (!const_dim) {
      return -1;
    }
    product *= const_dim->value_;
  }
  return product;
}

/// Build an INDEX-typed multiply, folding ConstInt * ConstInt and the
/// multiplicative identity (×1) so that downstream codegen sees the same
/// strides whether the source shape is static or dynamic.
///
/// Uses ``__builtin_mul_overflow`` to detect signed overflow in the constant
/// fold path; on overflow, falls back to a symbolic ``Mul`` rather than
/// silently wrapping (which would yield an incorrect stride that the
/// canonical-view verifier cannot detect).
inline ExprPtr MakeIndexMul(const ExprPtr& lhs, const ExprPtr& rhs, const Span& span = Span::unknown()) {
  auto const_lhs = As<ConstInt>(lhs);
  auto const_rhs = As<ConstInt>(rhs);
  if (const_lhs && const_rhs) {
    int64_t folded = 0;
    if (!__builtin_mul_overflow(const_lhs->value_, const_rhs->value_, &folded)) {
      return std::make_shared<ConstInt>(folded, DataType::INDEX, span);
    }
    // Overflow — drop to symbolic so callers / verifiers see a non-folded form.
  }
  if (const_rhs && const_rhs->value_ == 1) return lhs;
  if (const_lhs && const_lhs->value_ == 1) return rhs;
  return std::make_shared<Mul>(lhs, rhs, DataType::INDEX, span);
}

/// Build row-major (ND-packed) strides for the given shape:
///   strides[ndim-1] = 1; strides[i] = strides[i+1] * shape[i+1].
/// Works for both static and dynamic dims; ConstInt chains collapse via MakeIndexMul.
inline std::vector<ExprPtr> BuildRowMajorStrides(const std::vector<ExprPtr>& shape) {
  size_t ndim = shape.size();
  if (ndim == 0) return {};
  std::vector<ExprPtr> strides(ndim);
  strides[ndim - 1] = std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown());
  for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
    strides[i] = MakeIndexMul(strides[i + 1], shape[i + 1]);
  }
  return strides;
}

/// NZ fractal geometry, mirroring pto-isa's ``Layout::NZ``.
///
/// Ground truth (pto-isa ``include/pto/common/``):
///   * ``constants.hpp``  — ``FRACTAL_NZ_ROW = 16``, ``C0_SIZE_BYTE = 32``
///   * ``pto_tile.hpp``   — ``TileShape2D<T, R, C, Layout::NZ>``
///                          = ``Shape<1, C/c0, R/16, 16, c0>``
///   * ``pto_tile.hpp``   — ``BaseShape2D<T, R, C, Layout::NZ>``
///                          = ``Stride<C*R, R*c0, 16*c0, c0, 1>``
///
/// pto-isa spells the C0 line as ``C0_SIZE_BYTE / sizeof(T)``. That is the same
/// quantity as ``NzC0Elems`` below computes, but only for whole-byte dtypes —
/// ``sizeof`` has no sub-byte value, which is why sub-byte dtypes have no NZ
/// form here at all. See ``NzC0Elems``.
constexpr int64_t kNzFractalRow = 16;
constexpr int64_t kNzC0SizeByte = 32;
constexpr int64_t kNzC0SizeBit = kNzC0SizeByte * 8;

/// Number of elements in one NZ C0 line (32 bytes) for ``dtype``.
///
/// Derived from the *bit* width, not ``GetByte()``. ``GetByte()`` is
/// ``ceil(bits/8)``, so every sub-byte dtype (INT4 / UINT4 / FP4 / HF4 / BOOL)
/// reports 1 and would yield ``c0 = 32`` instead of the 64 elements that
/// actually fit in a 32-byte C0 line — silently mis-blocking the tensor and
/// accepting misaligned extents (e.g. FP4 ``C = 544`` passes ``% 32`` but is
/// not a multiple of 64).
///
/// Sub-byte dtypes are rejected for now. This is a **PyPTO milestone-1 scope
/// limit, not a hardware or pto-isa one**: pto-isa's NZ machinery does handle
/// FP4 (``tload_common.hpp`` carries explicit ``caps::IsFP4`` branches through
/// the NZ paths and asserts ``staticShape[4] == C0_SIZE_BYTE / sizeof(DType)``).
/// Supporting it here means validating the packed-nibble addressing end to end,
/// which milestone 1 does not attempt. The bit-based formula above is already
/// the correct one to build on when it does.
inline int64_t NzC0Elems(DataType dtype) {
  const auto bits = static_cast<int64_t>(dtype.GetBit());
  CHECK(bits >= 8) << "NZ layout does not support the sub-byte dtype '" << dtype.ToString() << "' (" << bits
                   << " bits per element) yet. This is a current PyPTO limitation, not a hardware one. "
                   << "Use a whole-byte dtype, or annotate the tensor as pl.ND.";
  CHECK(kNzC0SizeBit % bits == 0) << "NZ layout: dtype '" << dtype.ToString() << "' (" << bits
                                  << " bits per element) does not "
                                  << "evenly divide the " << kNzC0SizeByte << "-byte C0 line";
  return kNzC0SizeBit / bits;
}

/// True when ``shape`` is already in blocked NZ form: rank >= 4 with trailing
/// dims ``[16, c0]``.
///
/// This is the post-``BlockNzTensorViews`` invariant. It is a *structural*
/// test, not a proof of provenance — an ordinary ND tensor that happens to end
/// in ``[16, c0]`` also satisfies it. Callers use it to assert that a tensor
/// *tagged* NZ has been blocked, never to infer that a tensor *is* NZ.
inline bool IsBlockedNzShape(const std::vector<ExprPtr>& shape, DataType dtype) {
  if (shape.size() < 4) return false;
  // A predicate must answer, not throw: a dtype with no NZ C0 line simply has
  // no blocked form. ``NzC0Elems`` raises for those, so screen them here.
  const auto bits = static_cast<int64_t>(dtype.GetBit());
  if (bits < 8 || kNzC0SizeBit % bits != 0) return false;
  auto fractal = As<ConstInt>(shape[shape.size() - 2]);
  auto line = As<ConstInt>(shape.back());
  return fractal && line && fractal->value_ == kNzFractalRow && line->value_ == NzC0Elems(dtype);
}

/// Rewrite a logical shape ``[..., R, C]`` into the blocked NZ shape
/// ``[..., C/c0, R/16, 16, c0]`` that pto-isa's ``Layout::NZ`` GlobalTensor
/// requires. Rank grows by 2 (the trailing logical pair becomes four dims).
///
/// The blocked shape's *row-major* strides are exactly pto-isa's NZ strides:
///
///   row-major over ``[..., C/c0, R/16, 16, c0]``
///     = ``[..., (C/c0)*R*c0, (R/16)*16*c0, 16*c0, c0, 1]``
///     = ``[..., C*R,          R*c0,         16*c0, c0, 1]``
///     = ``BaseShape2D<T, R, C, Layout::NZ>``
///
/// so NZ needs no dedicated stride rule — ``BuildLogicalStridesFromLayout``
/// routes it through ``BuildRowMajorStrides`` once the shape is blocked.
///
/// Alignment is a *user* contract (the annotation asserts how the bytes were
/// written), so violations raise ``pypto::ValueError`` naming the authoring fix.
/// Milestone 1 requires static trailing dims: a dynamic extent cannot be proven
/// divisible, and silently mis-addressing GM is worse than refusing to compile.
inline std::vector<ExprPtr> BlockNzShape(const std::vector<ExprPtr>& shape, DataType dtype,
                                         const Span& span = Span::unknown()) {
  CHECK_SPAN(shape.size() >= 2, span)
      << "NZ layout requires a tensor of rank >= 2 (the trailing pair is the fractal plane), got rank "
      << shape.size();
  const int64_t c0 = NzC0Elems(dtype);

  auto rows = As<ConstInt>(shape[shape.size() - 2]);
  auto cols = As<ConstInt>(shape.back());
  CHECK_SPAN(rows, span) << "NZ layout requires a static shape[-2], got a dynamic extent. "
                         << "Dynamic NZ tensors are not supported yet.";
  CHECK_SPAN(cols, span) << "NZ layout requires a static shape[-1], got a dynamic extent. "
                         << "Dynamic NZ tensors are not supported yet.";
  CHECK_SPAN(rows->value_ > 0 && rows->value_ % kNzFractalRow == 0, span)
      << "NZ layout requires shape[-2] to be a positive multiple of " << kNzFractalRow << ", got "
      << rows->value_ << ". The bytes of an NZ tensor are grouped into " << kNzFractalRow
      << "-row fractals, so a partial fractal has no representation.";
  CHECK_SPAN(cols->value_ > 0 && cols->value_ % c0 == 0, span)
      << "NZ layout requires shape[-1] to be a positive multiple of c0 = " << c0 << " (" << kNzC0SizeBit
      << " bits / " << dtype.GetBit() << "-bit '" << dtype.ToString() << "'), got " << cols->value_ << ".";

  std::vector<ExprPtr> blocked;
  blocked.reserve(shape.size() + 2);
  for (size_t i = 0; i + 2 < shape.size(); ++i) blocked.push_back(shape[i]);
  auto make_index = [&span](int64_t v) { return std::make_shared<ConstInt>(v, DataType::INDEX, span); };
  blocked.push_back(make_index(cols->value_ / c0));             // C/c0  — column blocks (outermost)
  blocked.push_back(make_index(rows->value_ / kNzFractalRow));  // R/16 — row fractals
  blocked.push_back(make_index(kNzFractalRow));                 // 16    — rows within a fractal
  blocked.push_back(make_index(c0));                            // c0    — contiguous C0 line
  return blocked;
}

/// Map logical offsets ``[..., r0, c0off]`` into the blocked NZ coordinate
/// system ``[..., c0off/c0, r0/16, 0, 0]`` produced by ``BlockNzShape``.
///
/// A slice must start on a fractal boundary; an unaligned offset has no blocked
/// representation and is rejected rather than silently truncated.
inline std::vector<ExprPtr> BlockNzOffsets(const std::vector<ExprPtr>& offsets, DataType dtype,
                                           const Span& span = Span::unknown()) {
  CHECK_SPAN(offsets.size() >= 2, span) << "NZ layout requires rank >= 2 offsets, got " << offsets.size();
  const int64_t c0 = NzC0Elems(dtype);

  // Milestone 1 maps only *constant* trailing offsets. A symbolic offset would
  // need a divisibility proof plus an algebraic rewrite (``nb*256`` -> ``nb*16``
  // for the 16-row axis); that is not implemented, so even a provably aligned
  // expression is refused rather than guessed at. This is the limit that keeps
  // a loop-derived slice (``n0 = nb * N_TILE``) out of NZ for now.
  auto row_off = As<ConstInt>(offsets[offsets.size() - 2]);
  auto col_off = As<ConstInt>(offsets.back());
  CHECK_SPAN(row_off, span) << "NZ layout does not support a dynamic offset on shape[-2] yet: only a "
                            << "constant offset (a multiple of " << kNzFractalRow << ") can be mapped to "
                            << "blocked coordinates. A loop-derived slice offset is not supported yet.";
  CHECK_SPAN(col_off, span) << "NZ layout does not support a dynamic offset on shape[-1] yet: only a "
                            << "constant offset (a multiple of c0 = " << c0 << ") can be mapped to "
                            << "blocked coordinates. A loop-derived slice offset is not supported yet.";
  CHECK_SPAN(row_off->value_ >= 0 && row_off->value_ % kNzFractalRow == 0, span)
      << "NZ slice offset on shape[-2] must be a non-negative multiple of " << kNzFractalRow << ", got "
      << row_off->value_ << ".";
  CHECK_SPAN(col_off->value_ >= 0 && col_off->value_ % c0 == 0, span)
      << "NZ slice offset on shape[-1] must be a non-negative multiple of c0 = " << c0 << ", got "
      << col_off->value_ << ".";

  std::vector<ExprPtr> blocked;
  blocked.reserve(offsets.size() + 2);
  for (size_t i = 0; i + 2 < offsets.size(); ++i) blocked.push_back(offsets[i]);
  auto make_index = [&span](int64_t v) { return std::make_shared<ConstInt>(v, DataType::INDEX, span); };
  blocked.push_back(make_index(col_off->value_ / c0));
  blocked.push_back(make_index(row_off->value_ / kNzFractalRow));
  blocked.push_back(make_index(0));  // start of the fractal's rows
  blocked.push_back(make_index(0));  // start of the C0 line
  return blocked;
}

/// Build packed canonical strides for the given (shape, layout).
///
/// Definitions (per RFC #1300 §2.3, amended for NZ):
///   ND : strides[n-1] = 1; strides[k] = strides[k+1] * shape[k+1]
///   DN : strides[n-2] = 1; strides[n-1] = shape[n-2];
///        strides[n-3] = shape[n-2] * shape[n-1];
///        strides[k]   = strides[k+1] * shape[k+1]   (k = n-4 .. 0)
///   NZ : row-major over the *blocked* shape (see ``BlockNzShape``). RFC #1300
///        originally declared NZ unrepresentable; that holds for a logical 2-D
///        shape but not for the blocked rank-(r+2) form, whose strides are
///        ordinary row-major and match pto-isa's ``BaseShape2D<..., NZ>``
///        exactly. Callers must block the shape first — ``CheckNzViewIsBlocked``
///        enforces that invariant downstream.
///
/// Throws ``pypto::ValueError`` for DN layout with rank < 2.
inline std::vector<ExprPtr> BuildLogicalStridesFromLayout(const std::vector<ExprPtr>& shape,
                                                          TensorLayout layout) {
  size_t ndim = shape.size();
  if (ndim == 0) return {};

  // NZ joins the row-major family once its shape is blocked — see the NZ note
  // above and ``BlockNzShape``.
  if (layout == TensorLayout::ND || layout == TensorLayout::NZ || IsMxTensorLayout(layout)) {
    return BuildRowMajorStrides(shape);
  }

  if (layout == TensorLayout::DN) {
    CHECK(ndim >= 2) << "BuildLogicalStridesFromLayout: DN layout requires rank >= 2, got " << ndim;
    std::vector<ExprPtr> strides(ndim);
    auto one = std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown());
    // Innermost two dims: stride[n-2]=1, stride[n-1]=shape[n-2].
    strides[ndim - 2] = one;
    strides[ndim - 1] = shape[ndim - 2];
    if (ndim >= 3) {
      // The dim immediately preceding the trailing pair gets the product of
      // the trailing two shape dims (one full DN-block worth of elements).
      strides[ndim - 3] = MakeIndexMul(shape[ndim - 2], shape[ndim - 1]);
      // Outer dims: row-major over the DN-block volume.
      for (int i = static_cast<int>(ndim) - 4; i >= 0; --i) {
        strides[i] = MakeIndexMul(strides[i + 1], shape[i + 1]);
      }
    }
    return strides;
  }

  // Every TensorLayout is handled above; a new enum value must pick a family.
  INTERNAL_CHECK(false) << "Internal error: BuildLogicalStridesFromLayout has no stride rule for layout '"
                        << TensorLayoutToString(layout) << "'";
  return {};
}

/// Static structural pattern detection from (shape, stride).
///
/// Returns:
///   - ``TensorLayout::ND`` if ``stride[-1]`` is the static constant 1
///     (covers ND-packed and ND-strided families)
///   - ``TensorLayout::DN`` if ``stride[-2]`` is the static constant 1 and
///     the trailing-stride structural condition holds
///     (covers DN-packed and DN-strided families)
///   - ``std::nullopt`` for symbolic / ambiguous / non-canonical cases
///
/// This is purely structural — it does not enforce the strided-family
/// inequality (``stride[-2] >= shape[-1]`` for ND, ``stride[-1] >= shape[-2]``
/// for DN); the verifier handles that with optional symbolic relaxation.
inline std::optional<TensorLayout> DeriveLayoutFromStrides(const std::vector<ExprPtr>& shape,
                                                           const std::vector<ExprPtr>& stride) {
  if (shape.size() != stride.size() || shape.empty()) {
    return std::nullopt;
  }
  size_t n = stride.size();

  auto trailing = As<ConstInt>(stride[n - 1]);
  if (trailing && trailing->value_ == 1) {
    return TensorLayout::ND;
  }

  if (n >= 2) {
    auto second_last = As<ConstInt>(stride[n - 2]);
    if (second_last && second_last->value_ == 1) {
      return TensorLayout::DN;
    }
  }

  return std::nullopt;
}

/// Result of a canonical-view check: ``ok`` plus a human-readable reason on
/// failure (empty when ``ok``).
struct CanonicalCheckResult {
  bool ok;
  std::string reason;
};

namespace detail {

/// Return true iff two index expressions are structurally equal as static
/// constants. Symbolic exprs are not compared (``relaxed_symbolic`` controls
/// whether the caller treats that as a pass or fail).
inline bool StaticEqual(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (lhs == rhs) return true;
  auto lc = As<ConstInt>(lhs);
  auto rc = As<ConstInt>(rhs);
  return lc && rc && lc->value_ == rc->value_;
}

inline bool IsConstOne(const ExprPtr& e) {
  auto c = As<ConstInt>(e);
  return c != nullptr && c->value_ == 1;
}

/// Check ``lhs >= rhs`` when both are static ConstInt. Returns std::nullopt
/// when either operand is symbolic.
inline std::optional<bool> StaticGreaterEqual(const ExprPtr& lhs, const ExprPtr& rhs) {
  auto lc = As<ConstInt>(lhs);
  auto rc = As<ConstInt>(rhs);
  if (!lc || !rc) return std::nullopt;
  return lc->value_ >= rc->value_;
}

}  // namespace detail

/// Verify (shape, stride, layout) is canonical per RFC #1300 §2.2:
///   - rank consistency
///   - innermost-stride constant 1 at the layout-specific axis
///   - strided-family inequality (when statically decidable)
///
/// ``relaxed_symbolic`` (default true): when an inequality cannot be statically
/// decided due to symbolic dims, accept the relaxed form (only the innermost
/// stride structural equality is enforced). When false, symbolic cases that
/// cannot prove the inequality are flagged.
inline CanonicalCheckResult CheckCanonicalView(const std::vector<ExprPtr>& shape,
                                               const std::vector<ExprPtr>& stride, TensorLayout layout,
                                               bool relaxed_symbolic = true) {
  // 0-rank tensors (scalar tensors) are canonical iff stride is also empty.
  // Check this before the generic stride.empty() rejection so a scalar tensor
  // doesn't trip the "must be materialized" error.
  if (shape.empty() && stride.empty()) {
    return {true, ""};
  }
  if (stride.empty()) {
    return {false, "stride is empty (must be materialized via MaterializeTensorStrides)"};
  }
  if (shape.size() != stride.size()) {
    std::ostringstream oss;
    oss << "stride rank " << stride.size() << " does not match shape rank " << shape.size();
    return {false, oss.str()};
  }

  size_t n = shape.size();

  // Blocked NZ is row-major over its rank-(r+2) shape, so it shares the ND
  // canonical form. ``CheckNzViewIsBlocked`` separately enforces that an NZ
  // view has actually been blocked; this only checks the stride structure.
  if (layout == TensorLayout::ND || layout == TensorLayout::NZ || IsMxTensorLayout(layout)) {
    if (!detail::IsConstOne(stride[n - 1])) {
      return {false, TensorLayoutToString(layout) + " layout requires innermost stride to be ConstInt(1)"};
    }
    // Outer-dim strided family: stride[k] >= stride[k+1] * shape[k+1].
    // Statically decidable cases enforce; symbolic cases pass under relaxed_symbolic.
    for (int k = static_cast<int>(n) - 2; k >= 0; --k) {
      auto packed = MakeIndexMul(stride[k + 1], shape[k + 1]);
      auto cmp = detail::StaticGreaterEqual(stride[k], packed);
      if (cmp.has_value() && !*cmp) {
        std::ostringstream oss;
        oss << TensorLayoutToString(layout) << " stride[" << k << "] is smaller than packed stride["
            << (k + 1) << "] * shape[" << (k + 1) << "]";
        return {false, oss.str()};
      }
      if (!cmp.has_value() && !relaxed_symbolic) {
        return {false, TensorLayoutToString(layout) +
                           " outer-dim stride relation is symbolic and cannot be statically verified"};
      }
    }
    return {true, ""};
  }

  // layout == DN
  if (n < 2) {
    return {false, "DN layout requires rank >= 2"};
  }
  if (!detail::IsConstOne(stride[n - 2])) {
    return {false, "DN layout requires stride[-2] to be ConstInt(1)"};
  }
  // Trailing stride: stride[-1] >= shape[-2].
  auto trailing_cmp = detail::StaticGreaterEqual(stride[n - 1], shape[n - 2]);
  if (trailing_cmp.has_value() && !*trailing_cmp) {
    return {false, "DN stride[-1] is smaller than shape[-2]"};
  }
  if (!trailing_cmp.has_value() && !relaxed_symbolic) {
    return {false, "DN trailing-stride relation is symbolic and cannot be statically verified"};
  }
  // Outer-dim relation: stride[k] >= stride[k+1] * shape[k+1] for k <= n-3.
  for (int k = static_cast<int>(n) - 3; k >= 0; --k) {
    auto packed = MakeIndexMul(stride[k + 1], shape[k + 1]);
    auto cmp = detail::StaticGreaterEqual(stride[k], packed);
    if (cmp.has_value() && !*cmp) {
      std::ostringstream oss;
      oss << "DN stride[" << k << "] is smaller than packed stride[" << (k + 1) << "] * shape[" << (k + 1)
          << "]";
      return {false, oss.str()};
    }
    if (!cmp.has_value() && !relaxed_symbolic) {
      return {false, "DN outer-dim stride relation is symbolic and cannot be statically verified"};
    }
  }
  return {true, ""};
}

/// Convenience wrapper around CheckCanonicalView returning only the ok flag.
inline bool IsCanonicalView(const std::vector<ExprPtr>& shape, const std::vector<ExprPtr>& stride,
                            TensorLayout layout, bool relaxed_symbolic = true) {
  return CheckCanonicalView(shape, stride, layout, relaxed_symbolic).ok;
}

/// Build a packed canonical TensorView for (shape, layout). Used by the
/// MaterializeTensorStrides pass to fill stride.empty() slots.
inline TensorView CanonicalizeView(const std::vector<ExprPtr>& shape, TensorLayout layout) {
  return TensorView(BuildLogicalStridesFromLayout(shape, layout), layout, /*valid_shape=*/{});
}

}  // namespace pypto::ir::tensor_view_semantics

#endif  // PYPTO_IR_TRANSFORMS_UTILS_TENSOR_VIEW_SEMANTICS_H_

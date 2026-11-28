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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"  // CHECK
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/scalar_offset_facts.h"
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

/// Arity of pto-isa's NZ ``GlobalTensor`` — ``<B, C/c0, R/16, 16, c0>``.
///
/// Fixed, not derived from the logical rank: the leading batch slot is present
/// whether or not the logical tensor has a leading axis, which is why a logical
/// rank-2 NZ tensor blocks to rank 5 (batch ``1``) rather than rank 4. PTOAS
/// enforces this exactly — ``user-specified layout=nz requires a rank-5 view``.
constexpr size_t kNzBlockedRank = 5;

/// Gate a logical NZ rank to the window that has a canonical blocked form.
///
/// pto-isa declares NZ at a fixed rank-5 arity ``<B, C/c0, R/16, 16, c0>`` with
/// a *single* batch slot, so everything before the trailing ``[R, C]`` pair folds
/// into that one slot: a logical ``[RANKS, E, R, C]`` blocks with
/// ``B = RANKS*E``. The fold is exact because those leading axes are dense and
/// row-major — collapsing them multiplies the extents and leaves every stride
/// below untouched — and an offset into them folds the same way
/// (``FoldNzLeadingOffsets``). Only the trailing pair has no fold: it *is* the
/// fractal plane.
///
/// Both ``BlockNzShape`` and ``BlockNzOffsets`` must agree on the accepted
/// window — they produce the two halves of one ``tile.load`` and a rank they
/// disagree on would emit a view whose shape and coordinates have different
/// arity. ``what`` names which half is being blocked so the message points at
/// the offending annotation rather than at whichever half ran first.
///
/// A ``CHECK`` rather than an ``INTERNAL_CHECK``: the rank comes from a user's
/// ``pl.NZ`` annotation, so rank 1 is an authoring mistake, not a broken
/// invariant.
inline void CheckNzLogicalRank(size_t rank, const char* what, const Span& span = Span::unknown()) {
  CHECK_SPAN(rank >= 2, span) << "NZ layout requires a tensor of rank >= 2 (the trailing pair is the "
                              << "fractal plane), got " << what << " of rank " << rank << ".";
}

/// Number of elements in one NZ C0 line (32 bytes) for ``dtype``.
///
/// Derived from the bit width; sub-byte INT4, UINT4, HF4, and BOOL have no
/// supported NZ addressing contract here. Both packed FP4 x2 formats use
/// eight-bit byte carriers, so one C0 line contains 32 carrier elements.
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

/// True when ``shape`` is in canonical blocked NZ form: exactly
/// ``kNzBlockedRank`` dims, with trailing dims ``[16, c0]``.
///
/// The rank test is an equality, not a lower bound. Any other rank is a shape
/// PTOAS refuses to assemble, so accepting one here would let it pass every
/// downstream invariant check and fail only in the backend, naming SSA the user
/// never wrote.
///
/// This is the post-``BlockNzTensorViews`` invariant. It is a *structural*
/// test, not a proof of provenance — an ordinary ND tensor that happens to end
/// in ``[16, c0]`` also satisfies it. Callers use it to assert that a tensor
/// *tagged* NZ has been blocked, never to infer that a tensor *is* NZ.
inline bool IsBlockedNzShape(const std::vector<ExprPtr>& shape, DataType dtype) {
  if (shape.size() != kNzBlockedRank) return false;
  // A predicate must answer, not throw: a dtype with no NZ C0 line simply has
  // no blocked form. ``NzC0Elems`` raises for those, so screen them here.
  const auto bits = static_cast<int64_t>(dtype.GetBit());
  if (bits < 8 || kNzC0SizeBit % bits != 0) return false;
  auto fractal = As<ConstInt>(shape[shape.size() - 2]);
  auto line = As<ConstInt>(shape.back());
  return fractal && line && fractal->value_ == kNzFractalRow && line->value_ == NzC0Elems(dtype);
}

/// Rewrite a logical shape ``[B, R, C]`` (or ``[R, C]``) into the canonical
/// blocked NZ shape ``[B, C/c0, R/16, 16, c0]`` that pto-isa's ``Layout::NZ``
/// GlobalTensor requires.
///
/// The result is always ``kNzBlockedRank`` dims — the trailing logical ``[R, C]``
/// pair expands into four, and the leading batch slot is *materialised as ``1``*
/// when the logical tensor has no leading axis. That slot is what makes the form
/// canonical: pto-isa declares NZ at a fixed arity, so a logical rank-2 tensor
/// blocked to rank 4 is not a smaller NZ tensor, it is a shape PTOAS rejects.
///
/// The blocked shape's *row-major* strides are exactly pto-isa's NZ strides:
///
///   row-major over ``[B, C/c0, R/16, 16, c0]``
///     = ``[(C/c0)*R*c0, (R/16)*16*c0, 16*c0, c0, 1]``
///     = ``[C*R,          R*c0,         16*c0, c0, 1]``
///     = ``BaseShape2D<T, R, C, Layout::NZ>``
///
/// so NZ needs no dedicated stride rule — ``BuildLogicalStridesFromLayout``
/// routes it through ``BuildRowMajorStrides`` once the shape is blocked. The
/// synthesised ``B = 1`` costs nothing there: a leading extent of 1 contributes
/// a stride the addressing never multiplies by anything but zero.
///
/// Alignment is a *user* contract (the annotation asserts how the bytes were
/// written), so violations raise ``pypto::ValueError`` naming the authoring fix.
/// Milestone 1 requires static trailing dims: a dynamic extent cannot be proven
/// divisible, and silently mis-addressing GM is worse than refusing to compile.
/// Product of every axis before the trailing ``[R, C]`` pair — the extent of the
/// blocked batch slot.
///
/// Dense row-major leading axes collapse exactly: ``[RANKS, E, R, C]`` walks the
/// same bytes as ``[RANKS*E, R, C]``, because the only strides the fold removes
/// are the ones it multiplies back in. A tensor with no leading axis yields the
/// materialised ``1`` that makes the rank-5 form canonical.
inline ExprPtr FoldNzLeadingExtents(const std::vector<ExprPtr>& shape, const Span& span = Span::unknown()) {
  const size_t lead = shape.size() - 2;
  if (lead == 0) return std::make_shared<ConstInt>(1, DataType::INDEX, span);
  ExprPtr folded = shape[0];
  CHECK_SPAN(lead == 1 || As<ConstInt>(folded), span)
      << "NZ layout requires every leading extent to be static so the batch axes can be folded into one, "
      << "but axis 0 is dynamic. Reshape to [B, R, C] before the NZ annotation, or annotate the tensor as "
      << "pl.ND.";
  for (size_t i = 1; i < lead; ++i) {
    auto extent = As<ConstInt>(shape[i]);
    auto known = As<ConstInt>(folded);
    CHECK_SPAN(extent, span) << "NZ layout requires every leading extent to be static so the batch axes can "
                             << "be folded into one, but axis " << i << " is dynamic. Reshape to [B, R, C] "
                             << "before the NZ annotation, or annotate the tensor as pl.ND.";
    if (known) {
      folded = std::make_shared<ConstInt>(known->value_ * extent->value_, DataType::INDEX, span);
      continue;
    }
    folded = MakeMul(folded, shape[i], span);
  }
  return folded;
}

/// Fold an offset into the leading axes the same way ``FoldNzLeadingExtents``
/// folds their extents: row-major, against the *parent's* extents.
///
/// ``[r, e, 0, 0]`` into a ``[RANKS, E, R, C]`` tensor addresses batch
/// ``r*E + e``. Unlike the trailing pair this is a plain re-association of the
/// address arithmetic the ND path performs anyway — the offsets are multiplied
/// by the strides the fold removed — so no divisibility has to be proven and
/// nothing is discarded: the fold is exact for every value, not only aligned
/// ones.
inline ExprPtr FoldNzLeadingOffsets(const std::vector<ExprPtr>& offsets,
                                    const std::vector<ExprPtr>& parent_shape,
                                    const Span& span = Span::unknown()) {
  const size_t lead = offsets.size() - 2;
  if (lead == 0) return std::make_shared<ConstInt>(0, DataType::INDEX, span);
  CHECK_SPAN(parent_shape.size() == offsets.size(), span)
      << "NZ layout needs the parent's own rank to fold a leading offset, but the offsets are rank "
      << offsets.size() << " against a rank-" << parent_shape.size() << " tensor.";
  ExprPtr folded = offsets[0];
  for (size_t i = 1; i < lead; ++i) {
    auto extent = As<ConstInt>(parent_shape[i]);
    CHECK_SPAN(extent, span) << "NZ layout requires every leading extent to be static so an offset into the "
                             << "folded batch axis can be built, but axis " << i << " is dynamic. Reshape to "
                             << "[B, R, C] before the NZ annotation, or annotate the tensor as pl.ND.";
    // Fold constants as they go: a layer index written inline stays a literal
    // coordinate instead of arriving at codegen as arithmetic on two constants.
    auto known = As<ConstInt>(folded);
    auto step = As<ConstInt>(offsets[i]);
    if (known && step) {
      folded =
          std::make_shared<ConstInt>(known->value_ * extent->value_ + step->value_, DataType::INDEX, span);
      continue;
    }
    folded = MakeAdd(MakeMul(folded, parent_shape[i], span), offsets[i], span);
  }
  return folded;
}

inline std::vector<ExprPtr> BlockNzShape(const std::vector<ExprPtr>& shape, DataType dtype,
                                         const Span& span = Span::unknown()) {
  CheckNzLogicalRank(shape.size(), "shape", span);
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
  blocked.reserve(kNzBlockedRank);
  auto make_index = [&span](int64_t v) { return std::make_shared<ConstInt>(v, DataType::INDEX, span); };
  // Batch: every leading axis folded into the one slot, or a materialised 1
  // when the logical tensor has no leading axis at all.
  blocked.push_back(FoldNzLeadingExtents(shape, span));
  blocked.push_back(make_index(cols->value_ / c0));             // C/c0  — column blocks
  blocked.push_back(make_index(rows->value_ / kNzFractalRow));  // R/16 — row fractals
  blocked.push_back(make_index(kNzFractalRow));                 // 16    — rows within a fractal
  blocked.push_back(make_index(c0));                            // c0    — contiguous C0 line
  INTERNAL_CHECK_SPAN(blocked.size() == kNzBlockedRank, span)
      << "Internal error: blocked NZ shape has rank " << blocked.size() << ", expected " << kNzBlockedRank;
  return blocked;
}

using NzOffsetFacts = scalar_offset_facts::ScalarOffsetFacts;
constexpr int kNzDivideStepBudget = scalar_offset_facts::kScalarOffsetProofStepBudget;
using scalar_offset_facts::IsProvableMultipleOf;
using scalar_offset_facts::IsProvableNonNegative;

/// Block a ``tile.load``'s ``valid_shape`` into NZ coordinates.
///
/// Unlike ``BlockNzShape`` the row extent may be dynamic: a ragged last tile
/// names its width at run time (``min(extent - o0, TILE)``), and a valid region
/// that covers whole fractals has a blocked form -- fewer row fractals. The row
/// extent must therefore be *proven* a multiple of 16, and it is
/// divided as a whole (``FloorDiv(rows, 16)``), for the same wraparound reason
/// ``BlockNzOffsets`` divides an offset whole rather than re-associating it. The
/// column extent is the contiguous C0 line and stays static.
inline std::vector<ExprPtr> BlockNzValidShape(const std::vector<ExprPtr>& valid, DataType dtype,
                                              const Span& span, const NzOffsetFacts& facts) {
  CheckNzLogicalRank(valid.size(), "valid_shape", span);
  const ExprPtr& rows = valid[valid.size() - 2];
  if (As<ConstInt>(rows)) return BlockNzShape(valid, dtype, span);

  const int64_t c0 = NzC0Elems(dtype);
  auto cols = As<ConstInt>(valid.back());
  CHECK_SPAN(cols && cols->value_ > 0 && cols->value_ % c0 == 0, span)
      << "NZ layout requires a static valid_shape[-1] that is a positive multiple of c0 = " << c0
      << "; a partial C0 line has no blocked form.";
  int budget = kNzDivideStepBudget;
  CHECK_SPAN(IsProvableMultipleOf(rows, kNzFractalRow, facts, &budget), span)
      << "NZ layout requires a dynamic valid_shape[-2] to be a provable multiple of " << kNzFractalRow
      << ": a valid region that ends inside a fractal has no blocked form. Provable forms are a "
      << "constant, a loop variable whose start and step are multiples, and any sum, difference, "
      << "product, min or max built from those.";
  // The sign is deliberately not proven. It matters for an *offset* because
  // a negative one is clamped to 0 at the partition view and reads the wrong
  // fractal; a negative valid extent instead yields an empty partition, exactly
  // as it does for an ND load, so it carries no NZ-specific hazard. Divisibility
  // is the NZ-specific part -- a partial fractal has no blocked form.

  std::vector<ExprPtr> blocked;
  blocked.reserve(kNzBlockedRank);
  auto make_index = [&span](int64_t v) { return std::make_shared<ConstInt>(v, DataType::INDEX, span); };
  blocked.push_back(FoldNzLeadingExtents(valid, span));
  blocked.push_back(make_index(cols->value_ / c0));
  blocked.push_back(MakeFloorDiv(rows, make_index(kNzFractalRow), span));
  blocked.push_back(make_index(kNzFractalRow));
  blocked.push_back(make_index(c0));
  return blocked;
}

/// Map logical offsets ``[b, r0, c0off]`` (or ``[r0, c0off]``) into the blocked
/// NZ coordinate system ``[b, c0off/c0, r0/16, 0, 0]`` produced by
/// ``BlockNzShape``.
///
/// Mirrors ``BlockNzShape`` slot for slot, including the batch: a logical
/// rank-2 slice gets a materialised ``0`` against that shape's materialised
/// ``1``. The two must stay in lockstep — a ``tile.load`` whose shapes and
/// offsets disagree on arity is malformed well before PTOAS sees it.
///
/// A slice must start on a fractal boundary. A constant offset is folded
/// directly; a symbolic one is accepted only when ``IsProvableMultipleOf``
/// proves it a multiple of the axis factor, and is then divided as a whole
/// rather than re-associated. Anything else is rejected rather than silently
/// truncated.
inline std::vector<ExprPtr> BlockNzOffsets(const std::vector<ExprPtr>& offsets,
                                           const std::vector<ExprPtr>& parent_shape, DataType dtype,
                                           const Span& span = Span::unknown(),
                                           const NzOffsetFacts& facts = {}) {
  CheckNzLogicalRank(offsets.size(), "offsets", span);
  const int64_t c0 = NzC0Elems(dtype);

  // A constant keeps its own diagnostic: the offending value is in hand, so the
  // message can name it. Only a symbolic offset needs the proof machinery, and
  // its message has to describe the shapes that *are* provable instead.
  auto block_axis = [&](const ExprPtr& offset, int64_t divisor, const char* axis,
                        const std::string& unit) -> ExprPtr {
    if (auto const_offset = As<ConstInt>(offset)) {
      CHECK_SPAN(const_offset->value_ >= 0 && const_offset->value_ % divisor == 0, span)
          << "NZ slice offset on shape[" << axis << "] must be a non-negative multiple of " << unit
          << ", got " << const_offset->value_ << ".";
      return std::make_shared<ConstInt>(const_offset->value_ / divisor, DataType::INDEX, span);
    }
    // Both halves of the constant path's "non-negative multiple" contract have
    // to be re-proven for a symbolic offset, or the same value would be
    // rejected written inline and accepted once bound to a name.
    int budget = kNzDivideStepBudget;
    CHECK_SPAN(IsProvableMultipleOf(offset, divisor, facts, &budget), span)
        << "NZ layout requires the slice offset on shape[" << axis << "] to be a multiple of " << unit
        << ", and this one cannot be proven to be. Provable forms are a constant, a loop variable whose "
        << "start and step are both multiples of " << unit
        << ", and any sum, difference or constant multiple built from those. Slice on a " << unit
        << "-aligned boundary, or annotate the tensor as pl.ND.";
    int sign_budget = kNzDivideStepBudget;
    CHECK_SPAN(IsProvableNonNegative(offset, facts, &sign_budget), span)
        << "NZ layout requires the slice offset on shape[" << axis
        << "] to be non-negative, and this one cannot be proven to be. A negative offset is clamped to 0 "
        << "at the partition view rather than rejected, so it would read the wrong fractal silently. "
        << "Provable forms are a non-negative constant, the SPMD block index, a loop variable whose "
        << "start and step are both non-negative, and any sum or product built from those — note that a "
        << "difference never qualifies. Slice from a non-negative offset, or annotate the tensor as "
        << "pl.ND.";
    // Divide the offset's *result*, never its arithmetic: the expression keeps
    // its own dtype and wraparound behaviour, and only the value it produces is
    // divided. See ``IsProvableMultipleOf`` for why re-association is unsound.
    return MakeFloorDiv(offset, std::make_shared<ConstInt>(divisor, DataType::INDEX, span), span);
  };

  // Evaluate the row axis first so its diagnostic wins when both are malformed.
  auto row_blocked =
      block_axis(offsets[offsets.size() - 2], kNzFractalRow, "-2", std::to_string(kNzFractalRow));
  auto col_blocked = block_axis(offsets.back(), c0, "-1", "c0 = " + std::to_string(c0));

  std::vector<ExprPtr> blocked;
  blocked.reserve(kNzBlockedRank);
  auto make_index = [&span](int64_t v) { return std::make_shared<ConstInt>(v, DataType::INDEX, span); };
  // Batch: the leading offsets folded against the parent's extents, or the only
  // in-range coordinate for the shape's materialised extent of 1.
  blocked.push_back(FoldNzLeadingOffsets(offsets, parent_shape, span));
  blocked.push_back(std::move(col_blocked));
  blocked.push_back(std::move(row_blocked));
  blocked.push_back(make_index(0));  // start of the fractal's rows
  blocked.push_back(make_index(0));  // start of the C0 line
  INTERNAL_CHECK_SPAN(blocked.size() == kNzBlockedRank, span)
      << "Internal error: blocked NZ offsets have rank " << blocked.size() << ", expected " << kNzBlockedRank;
  return blocked;
}

/// True when ``shape`` is in canonical blocked MX form:
/// ``[1, positive block count, positive group count, 16, 2]``.
///
/// Structural only — callers assert that a tensor *tagged* MX has been
/// blocked, never infer that a tensor *is* MX from shape alone.
inline bool IsBlockedMxShape(const std::vector<ExprPtr>& shape) {
  if (shape.size() != 5) return false;
  auto batch = As<ConstInt>(shape[0]);
  auto block_count = As<ConstInt>(shape[1]);
  auto group_count = As<ConstInt>(shape[2]);
  auto fractal_rows = As<ConstInt>(shape[3]);
  auto fractal_cols = As<ConstInt>(shape[4]);
  return batch && block_count && group_count && fractal_rows && fractal_cols && batch->value_ == 1 &&
         block_count->value_ > 0 && group_count->value_ > 0 &&
         fractal_rows->value_ == tile_view_semantics::kMXSFractalRows &&
         fractal_cols->value_ == tile_view_semantics::kMXSFractalCols;
}

/// PTOAS v0.74 accepts logical MX views; their layout tag owns physical packing.
/// ZZ/NN retain static fractal alignment. ND activation scales are already
/// row-major GM data: rows may be dynamic and need no 16-row alignment, while
/// the contiguous group axis still consists of complete scale pairs.
inline bool IsLogicalMxShape(const std::vector<ExprPtr>& shape, TensorLayout layout) {
  if (shape.size() != 2 || !IsMxTensorLayout(layout)) return false;
  if (layout == TensorLayout::MX_A_ND) {
    auto rows = As<ConstInt>(shape[0]);
    auto groups = As<ConstInt>(shape[1]);
    return (!rows || rows->value_ > 0) && groups && groups->value_ > 0 &&
           groups->value_ % tile_view_semantics::kMXSFractalCols == 0;
  }
  const bool is_a = layout == TensorLayout::MX_A_ZZ;
  auto block = As<ConstInt>(shape[is_a ? 0 : 1]);
  auto group = As<ConstInt>(shape[is_a ? 1 : 0]);
  return block && group && block->value_ > 0 && group->value_ > 0 &&
         block->value_ % tile_view_semantics::kMXSFractalRows == 0 &&
         group->value_ % tile_view_semantics::kMXSFractalCols == 0;
}

/// Accept the default logical form or legacy blocked ZZ/NN views. MX_A_ND
/// always retains its rank-2 row-major ABI, including in the legacy pass.
inline bool IsCanonicalMxShape(const std::vector<ExprPtr>& shape, TensorLayout layout) {
  return IsLogicalMxShape(shape, layout) || (layout != TensorLayout::MX_A_ND && IsBlockedMxShape(shape));
}

/// Build packed canonical strides for the given (shape, layout).
///
/// Definitions (per RFC #1300 §2.3, amended for NZ / MX):
///   ND : strides[n-1] = 1; strides[k] = strides[k+1] * shape[k+1]
///   DN : strides[n-2] = 1; strides[n-1] = shape[n-2];
///        strides[n-3] = shape[n-2] * shape[n-1];
///        strides[k]   = strides[k+1] * shape[k+1]   (k = n-4 .. 0)
///   NZ : row-major over the *blocked* shape (see ``BlockNzShape``). RFC #1300
///        originally declared NZ unrepresentable; that holds for a logical 2-D
///        shape but not for the blocked rank-5 form, whose strides are
///        ordinary row-major and match pto-isa's ``BaseShape2D<..., NZ>``
///        exactly. Callers must block the shape first — ``CheckNzViewIsBlocked``
///        enforces that invariant downstream.
///   MX : logical rank-2 row-major strides for PTOAS-owned packing, or physical
///        row-major strides for the explicit legacy rank-5 blocking pass.
///        MX_A_ND already matches row-major GM data and always remains rank 2.
///
/// Throws ``pypto::ValueError`` for DN layout with rank < 2.
inline std::vector<ExprPtr> BuildLogicalStridesFromLayout(const std::vector<ExprPtr>& shape,
                                                          TensorLayout layout) {
  size_t ndim = shape.size();
  if (ndim == 0) return {};

  // NZ / MX join the row-major family once their shapes are blocked.
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
///   - non-overlapping storage axes for ND/DN, including sampled inner axes
///   - packed innermost stride for blocked layouts
///   - strided-family inequality (when statically decidable)
///
/// ``relaxed_symbolic`` (default true): when an inequality cannot be statically
/// decided due to symbolic dims, accept the relaxed form. When false, symbolic cases that
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

  if (layout == TensorLayout::ND || layout == TensorLayout::DN) {
    if (layout == TensorLayout::DN && n < 2) return {false, "DN layout requires rank >= 2"};
    // The distance between adjacent outer slices must cover the span of the
    // inner axes, including holes from sampling. For example a 5-wide row
    // sampled at columns 0, 2, 4 spans 5 elements, not 3 * 2.
    const bool empty = std::any_of(shape.begin(), shape.end(), [](const ExprPtr& dim) {
      auto value = As<ConstInt>(dim);
      return value && value->value_ == 0;
    });
    std::optional<int64_t> span = 1;
    for (size_t i = n; i-- > 0;) {
      const size_t axis = layout == TensorLayout::DN && i >= n - 2 ? (i == n - 1 ? n - 2 : n - 1) : i;
      auto pitch = As<ConstInt>(stride[axis]);
      auto extent = As<ConstInt>(shape[axis]);
      if ((!pitch || !span) && !relaxed_symbolic) {
        return {false, TensorLayoutToString(layout) + " stride relation is symbolic"};
      }
      if (pitch && (pitch->value_ < 0 || (pitch->value_ == 0 && !empty))) {
        return {false, "ordinary tensor view strides must be positive for a nonempty shape"};
      }
      if (pitch && span && pitch->value_ < *span) {
        return {false, TensorLayoutToString(layout) + " stride is smaller than the inner storage span"};
      }
      if (!extent || !pitch || !span) {
        span.reset();
      } else if (extent->value_ == 0) {
        span = 0;
      } else {
        int64_t delta;
        int64_t next_span;
        if (extent->value_ < 0 || __builtin_mul_overflow(extent->value_ - 1, pitch->value_, &delta) ||
            __builtin_add_overflow(*span, delta, &next_span)) {
          return {false, "tensor view storage span overflows index range"};
        }
        span = next_span;
      }
    }
    return {true, ""};
  }

  // Blocked NZ is row-major over its rank-5 shape, so it shares the ND
  // canonical form. ``CheckNzViewIsBlocked`` separately enforces that an NZ
  // view has actually been blocked; this only checks the stride structure.
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
      oss << TensorLayoutToString(layout) << " stride[" << k << "] is smaller than packed stride[" << (k + 1)
          << "] * shape[" << (k + 1) << "]";
      return {false, oss.str()};
    }
    if (!cmp.has_value() && !relaxed_symbolic) {
      return {false, TensorLayoutToString(layout) +
                         " outer-dim stride relation is symbolic and cannot be statically verified"};
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

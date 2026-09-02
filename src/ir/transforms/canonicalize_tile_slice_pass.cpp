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

/// CanonicalizeTileSlice
/// ---------------------
/// Lowers a ``tile.slice`` into the canonical ``tile.extract`` form so that
/// movement is unified on ``pto.textract`` — Mat-resident slices (folded into
/// matmul / ``tile.extract`` consumers), Vec slices that need materialization
/// for ``tile.col_expand_*`` (#1640, #2010), and Vec slices whose byte address
/// is not provably 32-byte aligned (#1789).
///
/// A ``tile.slice`` whose result tile is ``Mem.Mat`` is a legal high-level
/// "sub-window of a Mat tile" construct — ``FlattenTileNdTo2D`` emits one per
/// batch page when it unrolls a ``tile.batch_matmul`` (the page offset is
/// ``batch_index * page_rows``; for a leading-dim-1 batch the offset is 0 and
/// the window covers the whole tile, but it is still a ``tile.slice``).
/// PTO ISA supports ``pto.subview`` on Mat as a zero-copy alias (no data
/// movement), but a standalone Mat slice followed by a consumer that triggers
/// lazy materialization would attempt a ``loc=mat -> loc=mat``
/// ``pto.textract`` — an unsupported L1→L1 DMA path.
///
/// This pass eliminates Mat-resident ``tile.slice`` nodes whose consumers it
/// can canonicalize by folding the offset into each consumer:
///
///   * Consumed by ``tile.extract(s, ir, ic, shape)`` — the extract reads the
///     slice's source directly and the slice offset is added into ``ir`` /
///     ``ic``:
///         extract(slice(src, _, [or, oc]), ir, ic, shape)
///       == extract(src, ir + or, ic + oc, shape)
///
///   * Consumed by a ``tile.matmul`` / ``tile.matmul_acc`` / ``tile.matmul_bias``
///     operand — the operand is replaced by a fresh
///     ``tile.extract(src, or, oc, shape, target_memory=Left|Right)`` (Left for
///     the lhs operand, Right for the rhs).  This is the same Mat->Left/Right
///     extract that ``AutoTileMatmulL0`` emits for tiled matmuls.
///
/// It also canonicalizes a **Vec** ``tile.slice`` consumed by the
/// ``tile.col_expand_*`` family (issues #1640, #2010).  Those ops cannot read a
/// ``pto.subview`` operand, so codegen lazily materializes the slice via
/// ``pto.textract`` into the slice's own result buffer — and because
/// ``tile.slice`` inherits its source's memory, that buffer sits *inside the
/// still-live source*.  The extract is therefore performed in place over its own
/// input, which is safe only when it is an identity copy.  Two things must hold:
///
///   * the destination **address** must be right — a const offset is folded into
///     ``base + off``, but a dynamic one cannot be encoded as a ``ConstInt``
///     address and falls back to the bare source base, so the extracted window
///     lands on the source's row 0 (#1640);
///   * the destination **layout** must match — the slice buffer is dense (row
///     pitch = slice cols) while the source window is strided (row pitch = base
///     cols).  These coincide only for a *contiguous* window: a single row, or a
///     window spanning every column.  A column slice of a multi-row tile
///     (``t[:, a:b]``) repacks strided -> dense on top of its own source and
///     destroys it — only row 0 survives, since its dense destination happens to
///     equal its source address (#2010).
///
/// Whenever either condition fails, the operand is replaced by a fresh
/// ``tile.extract(src, or, oc, shape, target_memory=Vec)`` — whose result gets
/// its own non-inherited allocation — which removes the aliasing.  An
/// identity-copy slice is left untouched so it keeps sharing the source buffer.
///
/// Finally, PTO vector instructions require their tile operands' base addresses
/// to be 32-byte aligned.  A zero-copy Vec subview inherits
/// ``base + (row * base_cols + col) * storage_bits``; a column slice such as
/// ``fp32_tile[:, 1:2]`` therefore starts four bytes past an aligned allocation.
/// For every ordinary consumer of such a slice, this pass inserts a fresh Vec
/// ``tile.extract``.  The extract result has its own aligned allocation, while
/// slices whose offset is provably aligned remain zero-copy. Dynamic offsets
/// are handled conservatively, but scalar SSA arithmetic can prove alignment:
/// a dynamic row is safe when its known multiple times the base row stride is
/// aligned, and a dynamic column is safe when its known multiple times the
/// element storage width is aligned.
/// Last, the pass **rejects** the col-major (``Acc`` / L0C) dual of the #2010
/// contiguity condition when the slice is a matmul *accumulator*.  Here there is
/// no repair to apply — an ``Acc`` window cannot be copied out and back, because
/// nothing in the memory graph points into ``Acc`` — so the only correct
/// response is a diagnostic.  In L0C's NZ layout block ``(r_b, c_b)`` of an
/// ``[M, N]`` tile sits at ``(c_b * M/16 + r_b) * fractal``, so a window is
/// contiguous only when it spans the parent's full row extent or occupies a
/// single 16-column block.  A row slice of a multi-block-column accumulator is
/// therefore strided, and the MAD cannot express a destination stride: pto-isa's
/// ``TMATMUL_ACC_IMPL`` forwards the destination as a bare ``.data()`` pointer
/// and derives ``m`` from the *left operand*, so ``TileRes::Rows`` — the only
/// carrier of the parent stride — is discarded at the intrinsic boundary.  ptoas
/// preserves it faithfully up to that point (``getSubviewPhysicalType`` keeps the
/// parent shape and narrows via ``valid``); the information is lost in the last
/// call.  Without this guard the kernel silently computes wrong results, with
/// only the first 16 columns of each row tile correct.
///
/// Tracked upstream as hw-native-sys/pto-isa#253.  **This guard is scoped to
/// that defect, not to a property of the DSL** — a row window of an ``Acc`` tile
/// is a legitimate thing to write, and the IR expresses it correctly all the way
/// down.  If pto-isa gains a destination stride (or otherwise passes
/// ``TileRes::Rows`` into ``mad``), the shape becomes representable and this
/// rejection must be relaxed or deleted, not kept as a permanent DSL rule.
///
/// The window an accumulator operand names is rarely the ``tile.slice`` result
/// Var itself.  Hoisting the slice out of the K loop — the natural spelling,
/// since the window does not change per K step — turns the in-body operand into
/// a loop ``IterArg``; a shape-preserving ``tile.reshape`` of the window is
/// another SSA name for the same bytes.  ``SliceCollector`` therefore records a
/// window not only for the slice result but for every Var reached from it along
/// an *identity-preserving* edge (see ``SliceCollector::BindWindowAlias``), so
/// the guard's lookup answers "which window does this operand name", not "is
/// this operand a slice".  Widening that reach must never widen what the guard
/// *rejects*: an edge is recorded only when both ends provably describe the same
/// window extent, and anything unproven is left unrecorded — which reproduces
/// the pre-existing silence rather than guessing.
///
/// A loop's *back* edge is the one place a single Var names two different
/// windows — the initializer's on iteration 0, the yielded one afterwards — so
/// it is checked rather than recorded, and only when the loop provably runs more
/// than once (``SliceCollector::ResolveLoopEdges``).
///
/// **This guard can reject a shape the compiler itself emitted**, so the
/// diagnostic below must explain a window the user may never have written.
/// ``FlattenTileNdTo2D`` used to be the source of exactly that: it stacked the
/// pages of a batched accumulator along rows, so a ``tile.batch_matmul_acc``
/// wider than 16 columns produced the rejected window with no ``tile.slice`` in
/// the kernel source.  It now packs those pages along *columns* — the accepted
/// full-row-extent shape — and reports the batch shapes it cannot pack with its
/// own diagnostic, so that lowering no longer reaches here.  The rejection stays
/// as the backstop for any future pass that re-introduces a strided accumulator
/// destination.
///
/// After all consumers are rewritten the now-dead ``tile.slice`` is dropped.
/// Chained slices (a slice of a slice) are peeled, accumulating the offset.
///
/// Pipeline position: right after ``AutoTileMatmulL0`` (so the per-iter
/// ``tile.extract``s that read the batch-page slices already exist) and before
/// ``InferTileMemorySpace``.

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/storage_size.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

constexpr const char* kPassName = "CanonicalizeTileSlice";

/// Build a canonical index add, folding ConstInt cases so a zero offset leaves
/// the original index untouched (avoids spurious ``ko + 0`` forms).
ExprPtr MakeCanonicalIndexAdd(const ExprPtr& lhs, const ExprPtr& rhs, const Span& span) {
  auto lhs_const = As<ConstInt>(lhs);
  auto rhs_const = As<ConstInt>(rhs);
  if (lhs_const && rhs_const) {
    return std::make_shared<ConstInt>(lhs_const->value_ + rhs_const->value_, DataType::INDEX, span);
  }
  if (lhs_const && lhs_const->value_ == 0) return rhs;
  if (rhs_const && rhs_const->value_ == 0) return lhs;
  return MakeAdd(lhs, rhs, span);
}

/// True if `type` is a TileType resident in `Mem.Mat`.
bool IsMatTile(const TypePtr& type) {
  auto tile = As<TileType>(type);
  if (!tile) return false;
  auto mem = tile->GetMemorySpace();
  return mem.has_value() && *mem == MemorySpace::Mat;
}

/// A canonical `tile.slice` peeled to its (non-slice) base tile plus the
/// accumulated row/column offset.  Covers both Mem.Mat slices (folded into
/// matmul / `tile.extract` consumers) and Vec slices (materialized for a
/// `tile.col_expand_mul` consumer, see #1640).
struct SliceInfo {
  VarPtr base;      ///< Tile the consumer's `tile.extract` should read from.
  ExprPtr off_row;  ///< Row offset to fold into the consumer index.
  ExprPtr off_col;  ///< Column offset to fold into the consumer index.
  std::optional<MemorySpace>
      memory_space;  ///< Result tile's space (nullopt until InferTileMemorySpace runs).
  bool is_mat;       ///< memory_space == Mem.Mat (drives the matmul/extract rewrite).
};

/// Resolve a scalar Var that is known to be a direct ConstInt SSA definition.
ExprPtr ResolveKnownConstInt(const ExprPtr& expr,
                             const std::unordered_map<const Var*, ExprPtr>& known_consts) {
  auto var = AsVarLike(expr);
  if (!var) return expr;
  auto it = known_consts.find(var.get());
  return it == known_consts.end() ? expr : it->second;
}

/// Return a divisor of every possible value of `expr`, capped to the 256-bit
/// Vec alignment modulus. Unknown expressions are conservatively divisible by
/// one. Following scalar SSA definitions lets expressions such as
/// `block_idx * 32` prove an aligned column offset without pretending the
/// runtime value itself is constant.
int64_t KnownMultipleModuloAlignment(const ExprPtr& expr,
                                     const std::unordered_map<const Var*, ExprPtr>& scalar_defs,
                                     std::unordered_set<const Var*>& visiting) {
  constexpr int64_t modulus = 256;
  if (auto value = As<ConstInt>(expr)) {
    int64_t residue = value->value_ % modulus;
    if (residue < 0) residue += modulus;
    return std::gcd(residue, modulus);
  }
  if (auto var = AsVarLike(expr)) {
    auto it = scalar_defs.find(var.get());
    if (it == scalar_defs.end() || !visiting.insert(var.get()).second) return 1;
    int64_t multiple = KnownMultipleModuloAlignment(it->second, scalar_defs, visiting);
    visiting.erase(var.get());
    return multiple;
  }
  auto binary_multiple = [&](const BinaryExprPtr& binary, bool multiply) {
    int64_t lhs = KnownMultipleModuloAlignment(binary->left_, scalar_defs, visiting);
    int64_t rhs = KnownMultipleModuloAlignment(binary->right_, scalar_defs, visiting);
    return multiply ? std::gcd(lhs * rhs, modulus) : std::gcd(lhs, rhs);
  };
  if (auto add = As<Add>(expr)) return binary_multiple(add, false);
  if (auto sub = As<Sub>(expr)) return binary_multiple(sub, false);
  if (auto mul = As<Mul>(expr)) return binary_multiple(mul, true);
  return 1;
}

/// If `assign` is `var = tile.slice(src, shape, [off_row, off_col])`, return the
/// peeled base/offset.  `known` holds slices collected so far; a slice whose
/// source is itself a recorded slice is peeled through it (offsets summed), so
/// `base` is always a non-slice tile.
std::optional<SliceInfo> ParseSliceWindow(const AssignStmtPtr& assign,
                                          const std::unordered_map<const Var*, SliceInfo>& known,
                                          const std::unordered_map<const Var*, ExprPtr>& known_consts,
                                          bool require_canonical) {
  if (!assign || !assign->var_) return std::nullopt;
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_ || !IsOp(call, "tile.slice")) return std::nullopt;
  // Rewriting is restricted to canonical 3-arg slices (input, shape, offset): a
  // slice carrying valid_shape / drop_dims is not a plain window, so the
  // canonicalization rules below do not apply to it.  The Acc safety check has
  // no such restriction — it only needs the physical base and offset, and a
  // non-canonical slice miscompiles exactly the same way — so it parses with
  // `require_canonical = false`.
  if (require_canonical ? call->args_.size() != 3 : call->args_.size() < 3) return std::nullopt;

  auto src = AsVarLike(call->args_[0]);
  if (!src) return std::nullopt;
  auto offset = As<MakeTuple>(call->args_[2]);
  if (!offset || offset->elements_.size() != 2) return std::nullopt;

  // Record the slice result's actual memory space.  This pass runs before
  // InferTileMemorySpace, so it may be unset (nullopt); that is treated as
  // "Vec-or-unassigned" by the col-expand rewrite (see TryRewriteColExpand).
  auto slice_tile = As<TileType>(assign->var_->GetType());
  std::optional<MemorySpace> memory_space = slice_tile ? slice_tile->GetMemorySpace() : std::nullopt;
  bool is_mat = IsMatTile(assign->var_->GetType());
  ExprPtr off_row = ResolveKnownConstInt(offset->elements_[0], known_consts);
  ExprPtr off_col = ResolveKnownConstInt(offset->elements_[1], known_consts);
  VarPtr base = src;
  // Peel a chained slice: src itself may be a slice we already recorded.
  auto it = known.find(src.get());
  if (it != known.end()) {
    base = it->second.base;
    off_row = MakeCanonicalIndexAdd(it->second.off_row, off_row, assign->span_);
    off_col = MakeCanonicalIndexAdd(it->second.off_col, off_col, assign->span_);
  }
  return SliceInfo{base, off_row, off_col, memory_space, is_mat};
}

/// Rewrite-eligible slices only — the input to every canonicalization below.
std::optional<SliceInfo> ParseCanonicalSlice(const AssignStmtPtr& assign,
                                             const std::unordered_map<const Var*, SliceInfo>& known,
                                             const std::unordered_map<const Var*, ExprPtr>& known_consts) {
  return ParseSliceWindow(assign, known, known_consts, /*require_canonical=*/true);
}

/// Number of columns in one L0C fractal box (1024 B / 4 B per INT32|FP32 = 16x16).
constexpr int64_t kAccBlockCols = 16;

/// True when two tile types provably describe the *same window extent*: both
/// `TileType`, same rank, same element dtype, and every dimension a `ConstInt`
/// of the same value.
///
/// This is the side condition that makes operand resolution sound.  Landing in
/// a source's storage — which the registry predicates below establish — says
/// the two Vars name the same *bytes*; this says the extent the guard reads
/// off the operand's own `TileType` is still the extent of the window those
/// bytes form.  Together they are a proof, not a heuristic, and everything
/// unproven simply goes unrecorded (i.e. stays silent, as before):
///
///   * a shape-changing `tile.reshape` fails it, so the guard does not compare
///     a reshaped extent against the parent's rows;
///   * `tile.reinterpret_view` fails it on dtype — and the `kAccBlockCols` /
///     `kAccFractal` arithmetic downstream assumes a 4-byte accumulator element,
///     so a re-typed view must not inherit the window;
///   * `tile.transpose_view` swaps the trailing two extents, so it fails it for
///     every non-square window.  A *square* window does pass — and is harmless:
///     the transposed view spans the identical bytes with identical row and
///     column extents, so both the full-row-extent rule and the single-block
///     rule reach exactly the verdict they reach for the untransposed window;
///   * a symbolic extent fails it too (a `ConstInt` is required on both sides),
///     matching the guard's own refusal to reason about symbolic shapes.
///
/// Memory space is deliberately *not* compared: the verdict is derived from the
/// *parent* tile's space and the operand's extents, never from the operand's own
/// space — and the tile view ops legitimately deduce a result type that leaves
/// it unset for `InferTileMemorySpace` to fill in later.
bool NamesSameWindowExtent(const TileTypePtr& lhs, const TileTypePtr& rhs) {
  if (!lhs || !rhs) return false;
  if (lhs->dtype_ != rhs->dtype_) return false;
  if (lhs->shape_.size() != rhs->shape_.size()) return false;
  for (size_t i = 0; i < lhs->shape_.size(); ++i) {
    auto lhs_dim = As<ConstInt>(lhs->shape_[i]);
    auto rhs_dim = As<ConstInt>(rhs->shape_[i]);
    if (!lhs_dim || !rhs_dim || lhs_dim->value_ != rhs_dim->value_) return false;
  }
  return true;
}

/// The accumulator operand of `call` — the argument the op registry declares as
/// the in-place destination via `set_output_reuses_input`.  Null when the call
/// is not an accumulator op (or carries no Var there).
///
/// Reading the relation from the registry — rather than naming
/// `tile.matmul_acc` / `tile.gemv_acc` / `tile.matmul_mx_acc` — is what lets the
/// guard below cover any future accumulator op for free.
VarPtr AccumulatorOperand(const CallPtr& call) {
  if (!call || !call->op_) return nullptr;
  auto& reg = OpRegistry::GetInstance();
  if (!reg.IsRegistered(call->op_->name_)) return nullptr;
  auto declared = reg.GetEntry(call->op_->name_).GetOutputReusesInputArg();
  if (!declared.has_value() || *declared >= call->args_.size()) return nullptr;
  return AsVarLike(call->args_[*declared]);
}

/// The verdict itself: reject when a `view`-shaped destination inside the window
/// `window` is a *strided* window of a col-major (`Acc`) tile.  See the file
/// header for the full derivation; the short form is that the MAD writes its
/// `[m, n]` destination compactly from a bare pointer, so a window is only
/// representable when it spans the parent's full row extent or occupies a single
/// 16-column block.
///
/// The `col_major` + `kAccFractal` gate excludes the Vec-resident in-place ops
/// (`tile.scatter`, `tile.fillpad_inplace`), whose row-major counterpart of this
/// rule is handled by the rewrite paths above.
///
/// Split out of `CheckAccumulatorSliceContiguous` so the identical rule can be
/// applied to a destination reached by a second route: the loop back edge, where
/// a carried accumulator holds — from iteration 1 onward — the window the body's
/// trailing `pl.yield_` names rather than the initializer's.  `view` is passed
/// in rather than read off a Var because on that route the destination's extent
/// is the `IterArg`'s, not the yielded value's.
///
/// Silent on anything it cannot prove: a symbolic extent or a non-`Acc` layout
/// falls through untouched.  The guard exists to convert a known-wrong lowering
/// into a diagnostic, not to second-guess shapes it cannot evaluate — a false
/// rejection would break the *accepted* full-row-extent shape, which is
/// load-bearing.
///
/// Provisional: this rejects a shape the IR models correctly, purely because
/// pto-isa's MAD cannot write it (hw-native-sys/pto-isa#253).  Revisit when that
/// issue closes — if the intrinsic learns the destination stride, drop the
/// `view_rows != parent_rows` rejection below and keep only whatever the fixed
/// hardware still cannot express.
void CheckAccWindowContiguous(const CallPtr& call, const TileTypePtr& view, const SliceInfo& window) {
  if (!call || !call->op_ || !window.base) return;
  auto parent = As<TileType>(window.base->GetType());
  if (!view || !parent || view->shape_.size() != 2 || parent->shape_.size() != 2) return;

  // Only the col-major L0C orientation loses its stride at the MAD boundary.
  // Accept either statement of it: the memory space (set explicitly in tile
  // programming) or the block layout (which `tile.slice` inherits from source).
  const auto parent_view = tile_view_semantics::GetEffectiveTileView(*parent);
  const bool is_acc = parent->memory_space_.has_value() && *parent->memory_space_ == MemorySpace::Acc;
  if (!is_acc && parent_view.blayout != TileLayout::col_major) return;
  if (parent_view.fractal != tile_view_semantics::kAccFractal) return;

  auto view_rows = As<ConstInt>(view->shape_[0]);
  auto view_cols = As<ConstInt>(view->shape_[1]);
  auto parent_rows = As<ConstInt>(parent->shape_[0]);
  if (!view_rows || !view_cols || !parent_rows) return;

  if (view_rows->value_ == parent_rows->value_) return;  // spans the full row extent

  // A narrow window is safe only when it lies *inside* one 16-column block:
  // there is then no second block column for the MAD's compact write to
  // mis-stride. Width alone is not enough — a [16, 16] window at column offset
  // 8 of a [32, 32] accumulator straddles two blocks and corrupts the parent
  // exactly like a wider one. A dynamic offset cannot be proven and is
  // rejected rather than assumed.
  if (view_cols->value_ <= kAccBlockCols) {
    auto off_col = As<ConstInt>(window.off_col);
    if (off_col && off_col->value_ >= 0 &&
        off_col->value_ / kAccBlockCols == (off_col->value_ + view_cols->value_ - 1) / kAccBlockCols) {
      return;
    }
  }

  // The parent's column extent is only needed to describe the tile; keep it out
  // of the decision so a symbolic N is still rejected rather than dereferenced.
  auto parent_cols = As<ConstInt>(parent->shape_[1]);
  const std::string parent_cols_text = parent_cols ? std::to_string(parent_cols->value_) : std::string("?");

  // The column-packed replacement, spelled out only when the row split is a
  // whole number of windows — an unevenly divided or degenerate parent has no
  // single obvious rewrite, and a bogus suggestion is worse than none.
  std::string remedy;
  if (view_rows->value_ > 0 && parent_rows->value_ % view_rows->value_ == 0) {
    const int64_t packed_cols = (parent_rows->value_ / view_rows->value_) * view_cols->value_;
    const std::string rows_text = std::to_string(view_rows->value_);
    const std::string cols_text = std::to_string(view_cols->value_);
    remedy = " Pack the accumulator along columns rather than rows: allocate " + rows_text + "x" +
             std::to_string(packed_cols) + " and take tile.slice(acc, [" + rows_text + ", " + cols_text +
             "], [0, i * " + cols_text + "]) — the same L0C memory, addressed in the order the hardware " +
             "writes it.";
  }

  CHECK_SPAN(false, call->span_)
      << call->op_->name_ << ": the accumulator is a " << view_rows->value_ << "x" << view_cols->value_
      << " row window of a " << parent_rows->value_ << "x" << parent_cols_text
      << " Acc (L0C) tile, which is not contiguous in L0C's block layout and cannot be a matmul "
         "destination — the hardware MAD writes its result compactly and has no destination stride, "
         "so only the first "
      << kAccBlockCols << " columns of each row tile would be correct.\n"
      << "An accumulator window must either span every row of its parent tile, or be at most "
      << kAccBlockCols << " columns wide inside a single " << kAccBlockCols << "-column block." << remedy
      << "\n"
      // The window is not always spelled in the kernel, so the sentence above
      // must not be the only advice on offer — see
      // `docs/en/dev/passes/18-canonicalize_tile_slice.md`.
      << "Note: this window may not appear in the kernel source — a compiler pass can produce it. "
         "FlattenTileNdTo2D packs the pages of a batched accumulator along columns for exactly this "
         "reason, and rejects the batch shapes it cannot pack with its own diagnostic, so a "
         "batch_matmul_acc — or an ND matmul_acc, which lowers to one — no longer reaches this "
         "limit. If you did not write the tile.slice named above, the pass that produced it is "
         "handing the MAD a destination it cannot write; report it rather than reshaping your "
         "kernel around it.";
}

/// Reject a matmul accumulator that is a *strided* window of a col-major (`Acc`)
/// tile: resolve the call's declared in-place destination back to the window it
/// names, then apply `CheckAccWindowContiguous`.
///
/// `windows` maps a Var to the window it *names*, not just the windows produced
/// by a `tile.slice` — see `SliceCollector::BindWindowAlias` — so an accumulator
/// reached through a loop carry or a shape-preserving view op is checked exactly
/// like the slice result itself.  Because every recorded edge is geometry-
/// preserving, the destination's extent is still read from the operand's own
/// `TileType`.  An accumulator whose Var resolves to no recorded window falls
/// through untouched, i.e. stays silent.
///
/// `value` is the statement's call expression — an `AssignStmt`'s RHS or an
/// `EvalStmt`'s expression.  `As<Call>` is exact-kind, so a `Submit` is not
/// covered; `Submit` cannot occur here because the pass is gated to InCore
/// functions, whose bodies contain no task launches (see
/// `.claude/rules/pass-submit-awareness.md`).
void CheckAccumulatorSliceContiguous(const ExprPtr& value,
                                     const std::unordered_map<const Var*, SliceInfo>& windows) {
  auto call = As<Call>(value);
  auto acc = AccumulatorOperand(call);
  if (!acc) return;
  auto window = windows.find(acc.get());
  if (window == windows.end()) return;
  CheckAccWindowContiguous(call, As<TileType>(acc->GetType()), window->second);
}

/// Phase 1 — collect every canonical `tile.slice` definition in the function,
/// keyed by its result Var.  AssignStmts are visited in program order, so a
/// chained slice's source is always already recorded.
class SliceCollector : public IRVisitor {
 public:
  std::unordered_map<const Var*, SliceInfo> slices;
  std::unordered_map<const Var*, ExprPtr> scalar_defs;
  /// Which window each Var *names*, whether or not that Var is itself a
  /// `tile.slice` result.  Seeded from every `tile.slice`, canonical or not —
  /// `slices` drives the rewrites and is therefore restricted to the canonical
  /// 3-arg form, while the Acc safety check needs the physical base and offset
  /// of *any* window, since a slice carrying an explicit valid_shape reaches
  /// the MAD with exactly the same broken stride — and then extended along
  /// identity-preserving edges by `BindWindowAlias`.
  std::unordered_map<const Var*, SliceInfo> windows;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_) {
      if (As<ScalarType>(op->var_->GetType())) scalar_defs.emplace(op->var_.get(), op->value_);
      if (As<ConstInt>(op->value_)) {
        known_consts_.emplace(op->var_.get(), op->value_);
      } else if (auto source = AsVarLike(op->value_)) {
        auto it = known_consts_.find(source.get());
        if (it != known_consts_.end()) known_consts_.emplace(op->var_.get(), it->second);
      }
    }
    if (auto window = ParseSliceWindow(op, windows, known_consts_, /*require_canonical=*/false)) {
      windows.emplace(op->var_.get(), *window);
    } else {
      RecordAliasedWindow(op);
    }
    if (auto info = ParseCanonicalSlice(op, slices, known_consts_)) {
      slices.emplace(op->var_.get(), *info);
      return;
    }
    RecordCarriedAccumulatorUse(op->value_);
    CheckAccumulatorSliceContiguous(op->value_, windows);
  }

  /// An accumulator op written as a bare statement (`pl.tile.matmul_acc(win, a,
  /// b)`) parses to an `EvalStmt`, not an `AssignStmt`.  It reaches the MAD with
  /// the same destination, so it must reach the same guard.
  void VisitStmt_(const EvalStmtPtr& op) override {
    if (!op) return;
    IRVisitor::VisitStmt_(op);
    RecordCarriedAccumulatorUse(op->expr_);
    CheckAccumulatorSliceContiguous(op->expr_, windows);
  }

  /// Loop-carried windows, forward edge: bind an `IterArg` to the window its
  /// *initializer* names.  `IRVisitor::VisitStmt_(ForStmtPtr/WhileStmtPtr)`
  /// visits `iter_args_` before `body_`, so this fires before any in-body use of
  /// the carried Var — which is exactly the confirmed defect (a window hoisted
  /// out of the K loop).
  void VisitExpr_(const IterArgPtr& op) override {
    if (!op) return;
    BindWindowAlias(op, op->initValue_);
    IRVisitor::VisitExpr_(op);
  }

  /// The loop's two remaining edges, both resolvable only once the body has been
  /// visited and every in-body definition is in `windows`.
  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;
    IRVisitor::VisitStmt_(op);
    ResolveLoopEdges(op->iter_args_, op->body_, op->return_vars_, RunsAtLeastTwice(op));
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op) return;
    IRVisitor::VisitStmt_(op);
    // A `while` trip count is not statically known, so the back edge cannot be
    // proven to execute; only the exit edge is resolved.
    ResolveLoopEdges(op->iter_args_, op->body_, op->return_vars_, /*back_edge_taken=*/false);
  }

  /// An if-merge names a single window only when *both* arms yield that same
  /// window; anything else stays unbound, i.e. silent.
  void VisitStmt_(const IfStmtPtr& op) override {
    if (!op) return;
    IRVisitor::VisitStmt_(op);
    if (op->return_vars_.empty() || !op->else_body_.has_value()) return;
    auto then_yield = transform_utils::GetLastYieldStmt(op->then_body_);
    auto else_yield = transform_utils::GetLastYieldStmt(*op->else_body_);
    if (!then_yield || !else_yield) return;
    if (then_yield->value_.size() != op->return_vars_.size()) return;
    if (else_yield->value_.size() != op->return_vars_.size()) return;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const SliceInfo* then_window = LookupWindow(then_yield->value_[i]);
      const SliceInfo* else_window = LookupWindow(else_yield->value_[i]);
      if (!then_window || !else_window || !SameWindow(*then_window, *else_window)) continue;
      BindWindowAlias(op->return_vars_[i], then_yield->value_[i]);
    }
  }

 private:
  /// The two loop edges that only become resolvable after the body is visited.
  ///
  /// **Back edge (a correctness check, not a binding).**  A carried accumulator
  /// holds the initializer's window on iteration 0 — covered by
  /// `VisitExpr_(IterArgPtr)` — but from iteration 1 onward it holds whatever
  /// the trailing `pl.yield_` names, and the MAD writes *that* window on every
  /// remaining step.  So the yielded window is checked against the same rule,
  /// with the destination extent taken from the `IterArg` (the type the loop
  /// gives the carry).  It is checked, not recorded: a Var cannot name two
  /// windows, and recording either one would make the *other* iteration's
  /// destination invisible.
  ///
  /// The check is scoped to carries that are actually accumulator destinations
  /// (`carried_acc_uses_`), so a carry the body only reads is never rejected for
  /// a shape no MAD ever writes.  The canonical carry —
  /// `yield matmul_acc(iter_arg, ...)` — resolves straight back to the
  /// initializer's window and therefore re-reaches the verdict the body already
  /// passed, so this costs nothing on the shapes the pipeline generates.
  ///
  /// `back_edge_taken` gates it: a loop that provably runs at most once never
  /// feeds the yielded value back, so checking it there would reject a shape no
  /// MAD ever writes.  Unprovable trip counts (symbolic bounds, `while`) count
  /// as not taken — silence, as everywhere else in this guard.
  ///
  /// **Exit edge.**  `return_vars_[i]` takes the final value yielded for
  /// `iter_args_[i]` — or, on a zero-trip loop, the initializer.  It is bound
  /// only when those two agree on the window, which is the canonical carry
  /// (`yield matmul_acc(iter_arg, ...)`, and the plain `yield iter_arg`); the
  /// binding is then exact whatever the trip count. That keeps an accumulator
  /// consumed *after* the loop (`settled = pl.yield_(...)` then
  /// `matmul_acc(settled, ...)`) resolvable, without ever recording a window one
  /// path does not name — which matters because `windows` also feeds
  /// `ParseSliceWindow`'s chained-slice offset arithmetic.
  ///
  /// Both are single hash lookups over the map already built — no fixpoint, and
  /// the back edge is never traversed as a *binding* edge, so `windows` stays
  /// acyclic and the sweep stays O(N).
  void ResolveLoopEdges(const std::vector<IterArgPtr>& iter_args, const StmtPtr& body,
                        const std::vector<VarPtr>& return_vars, bool back_edge_taken) {
    auto yield = transform_utils::GetLastYieldStmt(body);
    if (!yield || yield->value_.size() != iter_args.size()) return;
    for (size_t i = 0; i < iter_args.size(); ++i) {
      const auto& iter_arg = iter_args[i];
      if (!iter_arg) continue;
      const SliceInfo* carried = LookupWindow(yield->value_[i]);
      if (back_edge_taken && carried) {
        auto used = carried_acc_uses_.find(iter_arg.get());
        if (used != carried_acc_uses_.end()) {
          CheckAccWindowContiguous(used->second, As<TileType>(iter_arg->GetType()), *carried);
        }
      }
      const SliceInfo* initial = LookupWindow(iter_arg);
      if (i < return_vars.size() && carried && initial && SameWindow(*carried, *initial)) {
        BindWindowAlias(return_vars[i], yield->value_[i]);
      }
    }
  }

  /// True when the loop provably runs at least twice, i.e. the value its body
  /// yields really does become a later iteration's carry.  Anything unprovable
  /// — a symbolic bound, a non-positive step — answers false.
  static bool RunsAtLeastTwice(const ForStmtPtr& op) {
    auto start = As<ConstInt>(op->start_);
    auto stop = As<ConstInt>(op->stop_);
    auto step = As<ConstInt>(op->step_);
    if (!start || !stop || !step || step->value_ <= 0) return false;
    return start->value_ + step->value_ < stop->value_;
  }

  /// The window `expr` names, or null when it names none.
  const SliceInfo* LookupWindow(const ExprPtr& expr) const {
    auto var = AsVarLike(expr);
    if (!var) return nullptr;
    auto it = windows.find(var.get());
    return it == windows.end() ? nullptr : &it->second;
  }

  /// True when two windows provably describe the same bytes: same parent Var and
  /// the same offsets, either as identical expressions or as equal `ConstInt`s.
  static bool SameWindow(const SliceInfo& lhs, const SliceInfo& rhs) {
    if (lhs.base.get() != rhs.base.get()) return false;
    return SameOffset(lhs.off_row, rhs.off_row) && SameOffset(lhs.off_col, rhs.off_col);
  }

  static bool SameOffset(const ExprPtr& lhs, const ExprPtr& rhs) {
    if (lhs.get() == rhs.get()) return true;
    auto lhs_const = As<ConstInt>(lhs);
    auto rhs_const = As<ConstInt>(rhs);
    return lhs_const && rhs_const && lhs_const->value_ == rhs_const->value_;
  }

  /// Remember the first accumulator op whose destination is a loop carry, so
  /// `ResolveLoopEdges` knows which carries the MAD actually writes.  Only
  /// `IterArg` destinations are recorded — every other destination is already
  /// checked where it is used.
  void RecordCarriedAccumulatorUse(const ExprPtr& value) {
    auto call = As<Call>(value);
    auto acc = AccumulatorOperand(call);
    if (!acc || !As<IterArg>(acc)) return;
    carried_acc_uses_.emplace(acc.get(), call);
  }

  /// Record `target` as naming the same window as `source`, when both provably
  /// describe the same window extent.  Unproven edges are simply not recorded.
  void BindWindowAlias(const VarPtr& target, const ExprPtr& source) {
    if (!target) return;
    auto source_var = AsVarLike(source);  // matches Var AND IterArg
    if (!source_var) return;
    auto it = windows.find(source_var.get());
    if (it == windows.end()) return;
    if (!NamesSameWindowExtent(As<TileType>(target->GetType()), As<TileType>(source_var->GetType()))) {
      return;
    }
    // Copy before inserting: the insert may rehash `windows`, and taking the
    // value by copy keeps this independent of that.
    const SliceInfo source_window = it->second;
    windows.emplace(target.get(), source_window);
  }

  /// A definition that is not itself a `tile.slice` but still binds a fresh SSA
  /// name to a window's storage:
  ///
  ///   * a plain SSA alias `v = w` (ConvertToSSA and Simplify both leave these
  ///     in place);
  ///   * a call whose output lands in a source operand's storage rather than a
  ///     buffer of its own — a buffer-aliasing view op (`tile.reshape`,
  ///     `tile.set_validshape`, ...) or an in-place / accumulate op
  ///     (`tile.matmul_acc`, so a chained accumulation stays checked).
  ///
  /// The relation is read from the op registry via
  /// `op_predicates::OutputInheritsSourceBuffer`, so a newly added in-place or
  /// view op is covered without editing this pass — and `tile.transpose` /
  /// `tile.extract`, which permute or copy into a *fresh* buffer, are excluded
  /// for the right reason rather than by name.  An in-place op names its source
  /// through the registry's declared index; a view op declares no index, and
  /// every inherit-input view in the registry takes its source as argument 0.
  void RecordAliasedWindow(const AssignStmtPtr& op) {
    if (!op || !op->var_) return;
    if (AsVarLike(op->value_)) {
      BindWindowAlias(op->var_, op->value_);
      return;
    }
    auto call = As<Call>(op->value_);
    if (!call || !call->op_) return;
    // A `tile.slice` that ParseSliceWindow could not read (no 2-element offset
    // tuple, or fewer than 3 args) carries an offset this pass cannot prove.
    // It is an inherit-input view op, so it would otherwise fall through below
    // and inherit its *source's* window — silently dropping that offset.
    if (IsOp(call, "tile.slice")) return;
    if (!op_predicates::OutputInheritsSourceBuffer(call->op_->name_)) return;
    size_t source_index = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size()).value_or(0);
    if (source_index >= call->args_.size()) return;
    BindWindowAlias(op->var_, call->args_[source_index]);
  }

  // Convert direct ConstInt SSA definitions (and their plain aliases) back to
  // constants before alignment analysis. ConvertToSSA commonly introduces
  // such Vars for literal slice offsets; treating them as dynamic causes
  // unnecessary Vec-to-Vec extracts even when the address is provably aligned.
  std::unordered_map<const Var*, ExprPtr> known_consts_;

  /// Loop carries the body uses as an accumulator destination, mapped to the
  /// first such call — the site `ResolveLoopEdges` blames when the loop's back
  /// edge feeds that carry a window the MAD cannot write.
  std::unordered_map<const Var*, CallPtr> carried_acc_uses_;
};

/// Phase 2 — rewrite canonicalizable `tile.slice` consumers: Mat slices are
/// folded into `tile.extract` / matmul, hazardous col-expand operands get fresh
/// storage, and unaligned Vec operands are materialized before ordinary ops.
class CanonicalizeMutator : public IRMutator {
 public:
  CanonicalizeMutator(const std::unordered_map<const Var*, SliceInfo>& slices,
                      const std::unordered_map<const Var*, ExprPtr>& scalar_defs)
      : slices_(slices), scalar_defs_(scalar_defs) {}

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto assign = As<AssignStmt>(base);
    if (!assign) return base;
    auto call = As<Call>(assign->value_);
    if (call && call->op_ && IsOp(call, "tile.extract") && call->args_.size() == 4) {
      auto src = AsVarLike(call->args_[0]);
      auto it = src ? slices_.find(src.get()) : slices_.end();
      if (it != slices_.end()) {
        // extract(slice(base, _, [or, oc]), ir, ic, shape)
        //   -> extract(base, ir + or, ic + oc, shape)
        const auto& info = it->second;
        const Span& sp = call->span_;
        std::vector<ExprPtr> args = {info.base, MakeCanonicalIndexAdd(call->args_[1], info.off_row, sp),
                                     MakeCanonicalIndexAdd(call->args_[2], info.off_col, sp), call->args_[3]};
        auto& reg = OpRegistry::GetInstance();
        auto new_call = reg.Create("tile.extract", args, call->kwargs_, sp);
        auto new_assign = MutableCopy(assign);
        new_assign->value_ = new_call;
        return new_assign;
      }
    }

    if (auto rewrite = TryRewriteMatmul(assign)) return SeqStmts::Flatten(std::move(*rewrite), assign->span_);
    if (auto rewrite = TryRewriteColExpand(assign)) {
      return SeqStmts::Flatten(std::move(*rewrite), assign->span_);
    }
    if (auto rewrite = TryMaterializeUnalignedVecCall(call)) {
      auto new_assign = MutableCopy(assign);
      new_assign->value_ = rewrite->call;
      rewrite->extracts.push_back(new_assign);
      return SeqStmts::Flatten(std::move(rewrite->extracts), assign->span_);
    }
    if (auto rewrite = TryMaterializeUnalignedVecAlias(assign)) return *rewrite;
    return base;
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto eval = As<EvalStmt>(base);
    if (!eval) return base;
    auto call = As<Call>(eval->expr_);
    auto rewrite = TryMaterializeUnalignedVecCall(call);
    if (!rewrite) return base;

    auto new_eval = MutableCopy(eval);
    new_eval->expr_ = rewrite->call;
    rewrite->extracts.push_back(new_eval);
    return SeqStmts::Flatten(std::move(rewrite->extracts), eval->span_);
  }

  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto yield = As<YieldStmt>(base);
    if (!yield) return base;

    std::vector<StmtPtr> extracts;
    std::vector<ExprPtr> new_values = yield->value_;
    for (size_t i = 0; i < yield->value_.size(); ++i) {
      auto value = AsVarLike(yield->value_[i]);
      auto it = value ? slices_.find(value.get()) : slices_.end();
      if (it == slices_.end() || !NeedsAlignedVecMaterialization(it->second)) continue;
      auto extract = BuildOperandExtract(value, it->second, MemorySpace::Vec, yield->span_);
      extracts.push_back(extract);
      new_values[i] = extract->var_;
    }
    if (extracts.empty()) return base;

    auto new_yield = MutableCopy(yield);
    new_yield->value_ = std::move(new_values);
    extracts.push_back(new_yield);
    return SeqStmts::Flatten(std::move(extracts), yield->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return MaterializeLoopInitsAndVisit(op); }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return MaterializeLoopInitsAndVisit(op); }

 private:
  struct MaterializedCall {
    std::vector<StmtPtr> extracts;
    CallPtr call;
  };

  /// Operand layout of the matmul family: (lhs index, rhs index) or nullopt.
  static std::optional<std::pair<size_t, size_t>> MatmulOperandIndices(const CallPtr& call) {
    if (!call || !call->op_) return std::nullopt;
    if (IsOp(call, "tile.matmul") || IsOp(call, "tile.matmul_bias")) {
      return call->args_.size() >= 2 ? std::optional<std::pair<size_t, size_t>>({0, 1}) : std::nullopt;
    }
    if (IsOp(call, "tile.matmul_acc")) {
      return call->args_.size() >= 3 ? std::optional<std::pair<size_t, size_t>>({1, 2}) : std::nullopt;
    }
    return std::nullopt;
  }

  /// Build `var = tile.extract(base, off_row, off_col, slice_shape,
  /// target_memory=target)` for a slice operand that needs materialization. The
  /// slice's result tile shape is forwarded as the extract shape — passing the
  /// existing shape expressions through (rather than extracting int64 values
  /// and rebuilding ConstInts) keeps the path safe under future symbolic dims.
  AssignStmtPtr BuildOperandExtract(const VarPtr& slice_var, const SliceInfo& info, MemorySpace target,
                                    const Span& span) {
    auto slice_tile = As<TileType>(slice_var->GetType());
    INTERNAL_CHECK(slice_tile && slice_tile->shape_.size() == 2)
        << "CanonicalizeTileSlice: materialized slice must have a 2-D TileType result";
    auto shape_tuple = std::make_shared<MakeTuple>(slice_tile->shape_, span);
    std::vector<ExprPtr> args = {info.base, info.off_row, info.off_col, shape_tuple};
    std::vector<std::pair<std::string, std::any>> kwargs = {{"target_memory", target}};
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("tile.extract", args, kwargs, span);
    auto var = std::make_shared<Var>(slice_var->name_hint_ + "_textract", call->GetType(), span);
    return std::make_shared<AssignStmt>(var, call, span);
  }

  /// If `assign` is a matmul-family op with a Mat-slice lhs/rhs operand, return
  /// the per-operand `tile.extract` statement(s) followed by the rebuilt
  /// matmul.  Returns nullopt when no operand is a Mat slice.
  std::optional<std::vector<StmtPtr>> TryRewriteMatmul(const AssignStmtPtr& assign) {
    auto call = As<Call>(assign->value_);
    if (!call) return std::nullopt;
    auto indices = MatmulOperandIndices(call);
    if (!indices) return std::nullopt;

    const Span& sp = call->span_;
    std::vector<StmtPtr> extracts;
    std::vector<ExprPtr> new_args = call->args_;
    bool rewrote = false;

    auto rewrite_operand = [&](size_t arg_idx, MemorySpace target) {
      auto operand = AsVarLike(call->args_[arg_idx]);
      if (!operand) return;
      auto it = slices_.find(operand.get());
      if (it == slices_.end() || !it->second.is_mat) return;
      auto extract = BuildOperandExtract(operand, it->second, target, sp);
      extracts.push_back(extract);
      new_args[arg_idx] = extract->var_;
      rewrote = true;
    };
    rewrite_operand(indices->first, MemorySpace::Left);
    rewrite_operand(indices->second, MemorySpace::Right);
    if (!rewrote) return std::nullopt;

    auto& reg = OpRegistry::GetInstance();
    auto new_call = reg.Create(call->op_->name_, new_args, call->kwargs_, sp);
    auto new_assign = MutableCopy(assign);
    new_assign->value_ = new_call;
    std::vector<StmtPtr> out = std::move(extracts);
    out.push_back(new_assign);
    return out;
  }

  /// True for the col-expand ops whose `pto.*` lowering materializes a subview
  /// operand via the lazy `pto.textract` path (pto_ops_common.cpp).  Must mirror
  /// the materializing set in `MakeNaryCodegenPTO` exactly (#1640).
  static bool IsColExpandMaterializingOp(const OpPtr& op) {
    return IsOp(op, "tile.col_expand_mul") || IsOp(op, "tile.col_expand_add") ||
           IsOp(op, "tile.col_expand_div") || IsOp(op, "tile.col_expand_sub") ||
           IsOp(op, "tile.col_expand_max") || IsOp(op, "tile.col_expand_min") ||
           IsOp(op, "tile.col_expand_expdif");
  }

  /// True when a slice offset is dynamic (either component is not a `ConstInt`).
  /// A dynamic offset cannot be encoded as a `ConstInt` address, so the slice
  /// buffer falls back to the bare source base and the lazy `pto.textract`
  /// materialization writes the extracted window into the source's row 0
  /// (#1640).  A const offset is folded into `base + off` by
  /// `AllocateMemoryAddr`, so the destination address is at least correct — but
  /// see `IsContiguousWindow` for why that alone is not enough.
  static bool IsDynamicSliceOffset(const SliceInfo& info) {
    return !As<ConstInt>(info.off_row) || !As<ConstInt>(info.off_col);
  }

  /// True when the slice's window occupies one unbroken run of bytes in the base
  /// tile's buffer — i.e. it is a single row, or it spans every column of the
  /// base.  Only such a window has a row pitch equal to the *dense* pitch of the
  /// slice's own (source-inherited) buffer, which is what makes the lazy
  /// `pto.textract` materialization an identity copy.
  ///
  /// A column slice of a multi-row tile is NOT contiguous: `pto.textract` has to
  /// repack strided (base cols) -> dense (slice cols), and its destination —
  /// `base + off`, inside the still-live source — overlaps its own input.  Row 0
  /// happens to land on its source address, every later row is written over data
  /// the extract has not read yet, and the source is destroyed (#2010).  Returns
  /// false conservatively when the shapes are not 2-D `ConstInt` (rewriting is
  /// always safe; skipping the rewrite is not).
  static bool IsContiguousWindow(const VarPtr& slice_var, const SliceInfo& info) {
    auto slice_tile = As<TileType>(slice_var->GetType());
    auto base_tile = info.base ? As<TileType>(info.base->GetType()) : nullptr;
    if (!slice_tile || !base_tile) return false;
    if (slice_tile->shape_.size() != 2 || base_tile->shape_.size() != 2) return false;
    auto slice_rows = As<ConstInt>(slice_tile->shape_[0]);
    auto slice_cols = As<ConstInt>(slice_tile->shape_[1]);
    auto base_cols = As<ConstInt>(base_tile->shape_[1]);
    if (!slice_rows || !slice_cols || !base_cols) return false;
    return slice_rows->value_ == 1 || slice_cols->value_ == base_cols->value_;
  }

  /// True when codegen's lazy `pto.textract` materialization of this slice into
  /// its own source-inherited buffer would NOT be an identity copy — i.e. it
  /// would corrupt the source.  Either the destination address is wrong (dynamic
  /// offset, #1640) or the destination layout is wrong (non-contiguous window,
  /// #2010).  Such a slice must be materialized through a `tile.extract` into a
  /// fresh, non-aliasing buffer instead.
  static bool MaterializationCorruptsSource(const VarPtr& slice_var, const SliceInfo& info) {
    return IsDynamicSliceOffset(info) || !IsContiguousWindow(slice_var, info);
  }

  /// If `assign` is a col-expand op with a Vec `tile.slice` operand whose lazy
  /// materialization would corrupt its source, return a fresh
  /// `tile.extract(src, off_row, off_col, shape, target_memory=Vec)` for each
  /// such operand followed by the rebuilt col-expand op (issues #1640, #2010).
  ///
  /// Codegen materializes a subview operand of the `pto.tcolexpand*` family via
  /// `pto.textract` into the slice's own result buffer (pto_ops_shared.cpp).
  /// That buffer inherits — and aliases — the source's allocation, so the
  /// extract writes into its own still-live input.  This is harmless only when
  /// the write is an identity copy: the destination address must be right (const
  /// offset) *and* its dense layout must match the source window's (a contiguous
  /// window).  Otherwise the repack destroys the source — see
  /// `MaterializationCorruptsSource`.  Materializing through `tile.extract`
  /// (which gets its own fresh non-inherited allocation) removes the aliasing.
  /// Identity-copy slices are left untouched so they keep sharing the source
  /// buffer rather than paying for a duplicate allocation.
  /// Returns nullopt when no operand is such a Vec slice.
  std::optional<std::vector<StmtPtr>> TryRewriteColExpand(const AssignStmtPtr& assign) {
    auto call = As<Call>(assign->value_);
    if (!call || !call->op_ || !IsColExpandMaterializingOp(call->op_) || call->args_.size() != 2) {
      return std::nullopt;
    }

    const Span& sp = call->span_;
    std::vector<StmtPtr> extracts;
    std::vector<ExprPtr> new_args = call->args_;
    bool rewrote = false;

    // Both operands are materialized by the codegen lazy path, so both can be
    // a hazardous Vec slice.
    for (size_t i = 0; i < call->args_.size(); ++i) {
      auto operand = AsVarLike(call->args_[i]);
      if (!operand) continue;
      auto it = slices_.find(operand.get());
      if (it == slices_.end()) continue;
      // Only Vec-or-unassigned slices: an explicit non-Vec slice (Left/Right/Acc)
      // feeding a col-expand op keeps the later InferTileMemorySpace implicit
      // move(..., Vec) path; rewriting it here would synthesize a tile.extract
      // from the wrong source memory class.  (memory_space is unset before that
      // pass — treat nullopt as Vec.)
      const auto& ms = it->second.memory_space;
      if (ms.has_value() && *ms != MemorySpace::Vec) continue;
      if (!MaterializationCorruptsSource(operand, it->second)) continue;  // identity textract, safe
      auto extract = BuildOperandExtract(operand, it->second, MemorySpace::Vec, sp);
      extracts.push_back(extract);
      new_args[i] = extract->var_;
      rewrote = true;
    }
    if (!rewrote) return std::nullopt;

    auto& reg = OpRegistry::GetInstance();
    auto new_call = reg.Create(call->op_->name_, new_args, call->kwargs_, sp);
    auto new_assign = MutableCopy(assign);
    new_assign->value_ = new_call;
    std::vector<StmtPtr> out = std::move(extracts);
    out.push_back(new_assign);
    return out;
  }

  static constexpr int64_t kVecOperandAlignmentBytes = 32;
  static constexpr int64_t kBitsPerByte = 8;
  static constexpr int64_t kVecOperandAlignmentBits = kVecOperandAlignmentBytes * kBitsPerByte;

  /// Normalize an integer into [0, modulus), including negative offsets.
  static int64_t PositiveModulo(int64_t value, int64_t modulus) {
    int64_t result = value % modulus;
    return result < 0 ? result + modulus : result;
  }

  /// Return the slice base-address offset modulo the 32-byte Vec operand
  /// alignment, in bits, when it can be proved statically.  The root tile's
  /// allocation is aligned; the slice adds
  /// `(row * base_cols + col) * storage_bits`.
  ///
  /// Dynamic offsets remain safe when their scalar SSA expressions have a
  /// known multiple that makes the corresponding row/column bit offset a
  /// multiple of the alignment. Calculating entirely modulo 256 avoids
  /// overflow for large static shapes and also handles packed sub-byte dtypes
  /// correctly.
  std::optional<int64_t> VecSliceAddressModulo(const SliceInfo& info) const {
    auto base_tile = info.base ? As<TileType>(info.base->GetType()) : nullptr;
    if (!base_tile || base_tile->shape_.size() != 2) return std::nullopt;

    const int64_t storage_bits = static_cast<int64_t>(storage_size::GetStorageBitWidth(base_tile->dtype_));
    if (storage_bits <= 0) return std::nullopt;

    // Root allocations use the -1 planning sentinel and are aligned. Preserve
    // a concrete MemRef byte offset when one is already known; a symbolic base
    // offset makes the inherited address unprovable.
    int64_t base_bits = 0;
    if (base_tile->memref_.has_value()) {
      auto byte_offset = As<ConstInt>((*base_tile->memref_)->byte_offset_);
      if (!byte_offset) return std::nullopt;
      if (byte_offset->value_ >= 0) {
        base_bits = PositiveModulo(byte_offset->value_, kVecOperandAlignmentBytes) * kBitsPerByte;
      }
    }

    auto col = As<ConstInt>(info.off_col);
    int64_t col_bits = 0;
    if (col) {
      col_bits =
          PositiveModulo(col->value_, kVecOperandAlignmentBits) * storage_bits % kVecOperandAlignmentBits;
    } else {
      std::unordered_set<const Var*> visiting;
      const int64_t col_multiple = KnownMultipleModuloAlignment(info.off_col, scalar_defs_, visiting);
      if (col_multiple * storage_bits % kVecOperandAlignmentBits != 0) return std::nullopt;
    }

    auto row = As<ConstInt>(info.off_row);
    if (row && row->value_ == 0) return (base_bits + col_bits) % kVecOperandAlignmentBits;

    auto base_cols = As<ConstInt>(base_tile->shape_[1]);
    if (!base_cols) return std::nullopt;
    const int64_t row_stride_bits =
        PositiveModulo(base_cols->value_, kVecOperandAlignmentBits) * storage_bits % kVecOperandAlignmentBits;
    if (!row) {
      std::unordered_set<const Var*> visiting;
      const int64_t row_multiple = KnownMultipleModuloAlignment(info.off_row, scalar_defs_, visiting);
      return row_multiple * row_stride_bits % kVecOperandAlignmentBits == 0
                 ? std::optional<int64_t>((base_bits + col_bits) % kVecOperandAlignmentBits)
                 : std::nullopt;
    }

    const int64_t row_bits =
        PositiveModulo(row->value_, kVecOperandAlignmentBits) * row_stride_bits % kVecOperandAlignmentBits;
    return (base_bits + row_bits + col_bits) % kVecOperandAlignmentBits;
  }

  /// True when the slice is explicitly Vec-resident or still awaits the
  /// default Vec assignment from InferTileMemorySpace.
  static bool IsVecOrUnassigned(const SliceInfo& info) {
    return !info.memory_space.has_value() || *info.memory_space == MemorySpace::Vec;
  }

  /// True unless the Vec slice's inherited base address can be proved 32-byte
  /// aligned.  Unknown shape/offset cases are materialized conservatively.
  bool NeedsAlignedVecMaterialization(const SliceInfo& info) const {
    if (!IsVecOrUnassigned(info)) return false;
    auto address_modulo = VecSliceAddressModulo(info);
    return !address_modulo.has_value() || *address_modulo != 0;
  }

  /// Materialize every unaligned Vec slice operand of an ordinary call through
  /// a fresh Vec `tile.extract`.  This is deliberately consumer-independent:
  /// the alignment contract belongs to PTO vector operands, not to one
  /// elementwise opcode. `tile.slice` is only another view and remains peeled;
  /// `tile.extract` has its own direct slice-folding rewrite above.
  std::optional<MaterializedCall> TryMaterializeUnalignedVecCall(const CallPtr& call) {
    if (!call || !call->op_ || IsOp(call, "tile.slice") || IsOp(call, "tile.extract")) {
      return std::nullopt;
    }

    const Span& sp = call->span_;
    std::vector<StmtPtr> extracts;
    std::vector<ExprPtr> new_args = call->args_;
    bool rewrote = false;
    for (size_t i = 0; i < call->args_.size(); ++i) {
      auto operand = AsVarLike(call->args_[i]);
      if (!operand) continue;
      auto it = slices_.find(operand.get());
      if (it == slices_.end() || !NeedsAlignedVecMaterialization(it->second)) continue;
      auto extract = BuildOperandExtract(operand, it->second, MemorySpace::Vec, sp);
      extracts.push_back(extract);
      new_args[i] = extract->var_;
      rewrote = true;
    }
    if (!rewrote) return std::nullopt;

    auto& reg = OpRegistry::GetInstance();
    auto new_call = reg.Create(call->op_->name_, new_args, call->kwargs_, sp);
    return MaterializedCall{std::move(extracts), std::move(new_call)};
  }

  /// Replace a plain SSA alias of an unaligned slice with an extract assigned
  /// directly to the alias Var.  The slice cannot then escape the consumer
  /// lookup merely by changing SSA identity.
  std::optional<StmtPtr> TryMaterializeUnalignedVecAlias(const AssignStmtPtr& assign) {
    auto source = AsVarLike(assign->value_);
    auto it = source ? slices_.find(source.get()) : slices_.end();
    if (it == slices_.end() || !NeedsAlignedVecMaterialization(it->second)) return std::nullopt;

    auto extract = BuildOperandExtract(source, it->second, MemorySpace::Vec, assign->span_);
    auto new_assign = MutableCopy(assign);
    new_assign->value_ = extract->value_;
    return new_assign;
  }

  /// Materialize unaligned slice initializers before a loop and substitute the
  /// fresh aligned buffers through that loop's IterArgs.  YieldStmt handling
  /// above performs the same conversion for values carried to later iterations.
  template <typename LoopStmtPtr>
  StmtPtr MaterializeLoopInitsAndVisit(const LoopStmtPtr& op) {
    std::vector<StmtPtr> extracts;
    std::vector<const Expr*> remapped_sources;
    for (const auto& iter_arg : op->iter_args_) {
      auto init = AsVarLike(VisitExpr(iter_arg->initValue_));
      auto it = init ? slices_.find(init.get()) : slices_.end();
      if (it == slices_.end() || !NeedsAlignedVecMaterialization(it->second)) continue;
      auto extract = BuildOperandExtract(init, it->second, MemorySpace::Vec, op->span_);
      extracts.push_back(extract);
      var_remap_[init.get()] = extract->var_;
      remapped_sources.push_back(init.get());
    }

    auto new_loop = IRMutator::VisitStmt_(op);
    for (const auto* source : remapped_sources) var_remap_.erase(source);
    if (extracts.empty()) return new_loop;
    extracts.push_back(new_loop);
    return SeqStmts::Flatten(std::move(extracts), op->span_);
  }

  const std::unordered_map<const Var*, SliceInfo>& slices_;
  const std::unordered_map<const Var*, ExprPtr>& scalar_defs_;
};

/// Phase 3a — collect every Var *used* (referenced on a statement's RHS).  An
/// AssignStmt's LHS is a definition, not a use, so it is deliberately skipped.
class VarUseCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> used;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override { VisitExpr(op->value_); }
  void VisitVarLike_(const VarPtr& op) override {
    used.insert(op.get());
    IRVisitor::VisitVarLike_(op);
  }
};

/// Phase 3b — drop the AssignStmts whose result Var is in the `dead` set.
class DropDeadSliceMutator : public IRMutator {
 public:
  explicit DropDeadSliceMutator(const std::unordered_set<const Var*>& dead) : dead_(dead) {}

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> out;
    out.reserve(op->stmts_.size());
    bool changed = false;
    for (const auto& child : op->stmts_) {
      auto assign = As<AssignStmt>(child);
      if (assign && assign->var_ && dead_.count(assign->var_.get())) {
        changed = true;  // dead Mat-slice definition — drop it
        continue;
      }
      auto visited = VisitStmt(child);
      if (visited.get() != child.get()) changed = true;
      out.push_back(visited);
    }
    if (!changed) return op;
    return SeqStmts::Flatten(std::move(out), op->span_);
  }

 private:
  const std::unordered_set<const Var*>& dead_;
};

}  // namespace

Pass CanonicalizeTileSlice() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_) return func;
    if (!IsInCoreType(func->func_type_)) return func;

    // Phase 1 — index every canonical tile.slice.
    SliceCollector collector;
    collector.VisitStmt(func->body_);
    if (collector.slices.empty()) return func;

    // Phase 2 — fold or materialize canonical slice consumers.
    CanonicalizeMutator mutator(collector.slices, collector.scalar_defs);
    auto new_body = mutator.VisitStmt(func->body_);

    // Phase 3 — drop the slice defs that no longer have any use.  A chained
    // slice (a slice of a slice) only becomes dead once the slice that consumes
    // it is dropped, so iterate to a fixpoint — bounded by the slice count,
    // since every non-terminating iteration drops at least one statement.  A
    // slice still used at the end had a consumer this pass does not
    // canonicalize; it is left intact (no regression versus the pre-pass IR).
    for (size_t round = 0; round <= collector.slices.size(); ++round) {
      VarUseCollector uses;
      uses.VisitStmt(new_body);
      std::unordered_set<const Var*> dead;
      for (const auto& [slice_var, info] : collector.slices) {
        if (uses.used.find(slice_var) == uses.used.end()) dead.insert(slice_var);
      }
      if (dead.empty()) break;
      DropDeadSliceMutator dropper(dead);
      auto dropped = dropper.VisitStmt(new_body);
      if (dropped.get() == new_body.get()) break;  // nothing left to remove
      new_body = dropped;
    }

    if (new_body.get() == func->body_.get()) return func;
    auto new_func = MutableCopy(func);
    new_func->body_ = new_body;
    return new_func;
  };
  return CreateFunctionPass(pass_func, kPassName, kCanonicalizeTileSliceProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

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

/// Reject a matmul accumulator that is a *strided* window of a col-major (`Acc`)
/// tile.  See the file header for the full derivation; the short form is that
/// the MAD writes its `[m, n]` destination compactly from a bare pointer, so a
/// window is only representable when it spans the parent's full row extent or
/// occupies a single 16-column block.
///
/// Scoped by the op registry's `set_output_reuses_input` — the declared,
/// op-level statement of "argument `i` is the in-place destination" — so it
/// covers `tile.matmul_acc` / `tile.gemv_acc` / `tile.matmul_mx_acc` without
/// naming them, and any future accumulator op for free.  The `col_major` +
/// `kAccFractal` gate excludes the Vec-resident in-place ops (`tile.scatter`,
/// `tile.fillpad_inplace`), whose row-major counterpart of this rule is handled
/// by the rewrite paths above.
///
/// Silent on anything it cannot prove: a symbolic extent, a non-`Acc` layout, or
/// an accumulator that is not a recorded slice all fall through untouched.  The
/// guard exists to convert a known-wrong lowering into a diagnostic, not to
/// second-guess shapes it cannot evaluate.
///
/// Provisional: this rejects a shape the IR models correctly, purely because
/// pto-isa's MAD cannot write it (hw-native-sys/pto-isa#253).  Revisit when that
/// issue closes — if the intrinsic learns the destination stride, drop the
/// `view_rows != parent_rows` rejection below and keep only whatever the fixed
/// hardware still cannot express.
void CheckAccumulatorSliceContiguous(const AssignStmtPtr& assign,
                                     const std::unordered_map<const Var*, SliceInfo>& slices) {
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_) return;
  auto& reg = OpRegistry::GetInstance();
  if (!reg.IsRegistered(call->op_->name_)) return;
  auto declared = reg.GetEntry(call->op_->name_).GetOutputReusesInputArg();
  if (!declared.has_value() || *declared >= call->args_.size()) return;

  auto acc = AsVarLike(call->args_[*declared]);
  if (!acc) return;
  auto slice = slices.find(acc.get());
  if (slice == slices.end()) return;

  auto view = As<TileType>(acc->GetType());
  auto parent = As<TileType>(slice->second.base->GetType());
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
    auto off_col = As<ConstInt>(slice->second.off_col);
    if (off_col && off_col->value_ >= 0 &&
        off_col->value_ / kAccBlockCols == (off_col->value_ + view_cols->value_ - 1) / kAccBlockCols) {
      return;
    }
  }

  // The parent's column extent is only needed to describe the tile; keep it out
  // of the decision so a symbolic N is still rejected rather than dereferenced.
  auto parent_cols = As<ConstInt>(parent->shape_[1]);
  const std::string parent_cols_text = parent_cols ? std::to_string(parent_cols->value_) : std::string("?");

  CHECK_SPAN(false, call->span_)
      << call->op_->name_ << ": the accumulator is a " << view_rows->value_ << "x" << view_cols->value_
      << " row window of a " << parent_rows->value_ << "x" << parent_cols_text
      << " Acc (L0C) tile, which is not contiguous in L0C's block layout and cannot be a matmul "
         "destination — the hardware MAD writes its result compactly and has no destination stride, "
         "so only the first "
      << kAccBlockCols << " columns of each row tile would be correct.\n"
      << "Slice the accumulator along columns instead, so each window spans every row: allocate "
      << view_rows->value_ << "x" << (parent_rows->value_ / view_rows->value_) * view_cols->value_
      << " and use tile.slice(acc, [" << view_rows->value_ << ", " << view_cols->value_ << "], [0, i * "
      << view_cols->value_
      << "]). That is the same L0C memory, addressed in the order the hardware writes it.";
}

/// Phase 1 — collect every canonical `tile.slice` definition in the function,
/// keyed by its result Var.  AssignStmts are visited in program order, so a
/// chained slice's source is always already recorded.
class SliceCollector : public IRVisitor {
 public:
  std::unordered_map<const Var*, SliceInfo> slices;
  std::unordered_map<const Var*, ExprPtr> scalar_defs;
  /// Every `tile.slice` window, canonical or not.  `slices` drives the
  /// rewrites and is therefore restricted to the canonical 3-arg form; the Acc
  /// safety check needs the physical base and offset of *any* window, since a
  /// slice carrying an explicit valid_shape reaches the MAD with exactly the
  /// same broken stride.
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
    }
    if (auto info = ParseCanonicalSlice(op, slices, known_consts_)) {
      slices.emplace(op->var_.get(), *info);
      return;
    }
    CheckAccumulatorSliceContiguous(op, windows);
  }

 private:
  // Convert direct ConstInt SSA definitions (and their plain aliases) back to
  // constants before alignment analysis. ConvertToSSA commonly introduces
  // such Vars for literal slice offsets; treating them as dynamic causes
  // unnecessary Vec-to-Vec extracts even when the address is provably aligned.
  std::unordered_map<const Var*, ExprPtr> known_consts_;
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

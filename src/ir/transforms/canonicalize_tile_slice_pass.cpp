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
/// movement is unified on ``pto.textract`` — both Mat-resident slices (folded
/// into matmul / ``tile.extract`` consumers) and dynamic-offset Vec slices
/// (materialized for ``tile.col_expand_mul`` / ``tile.col_expand_add``, #1640).
///
/// A ``tile.slice`` whose result tile is ``Mem.Mat`` is a legal high-level
/// "sub-window of a Mat tile" construct — ``FlattenTileNdTo2D`` emits one per
/// batch page when it unrolls a ``tile.batch_matmul`` (the page offset is
/// ``batch_index * page_rows``; for a leading-dim-1 batch the offset is 0 and
/// the window covers the whole tile, but it is still a ``tile.slice``).  Such
/// a slice has no standalone hardware lowering: codegen would materialize it
/// as an unsupported ``loc=mat -> loc=mat`` ``pto.tmov``.
///
/// This pass eliminates every Mat-resident ``tile.slice`` by folding its
/// offset into each consumer:
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
/// It also canonicalizes a **dynamic-offset Vec** ``tile.slice`` consumed by
/// ``tile.col_expand_mul`` / ``tile.col_expand_add`` (issue #1640).
/// ``pto.tcolexpandmul`` / ``pto.tcolexpandadd`` cannot read a ``pto.subview``
/// operand, so codegen lazily materializes the slice via ``pto.textract`` into
/// the slice's own result buffer.  Because ``tile.slice`` inherits its source's
/// memory, and ``AllocateMemoryAddr`` cannot encode a dynamic offset as a
/// ``ConstInt`` address, that buffer falls back to the bare source base — the
/// materialization then writes the extracted row into the source's row 0.
/// Replacing the operand with a fresh
/// ``tile.extract(src, or, oc, shape, target_memory=Vec)`` — whose result gets
/// its own non-inherited allocation — removes the aliasing.  Only **dynamic**
/// offsets are the hazard: ``AllocateMemoryAddr`` folds a const offset into
/// ``base + off``, so the lazy ``pto.textract`` is an identity copy and a
/// static-offset slice is left untouched.
///
/// After all consumers are rewritten the now-dead ``tile.slice`` is dropped.
/// Chained slices (a slice of a slice) are peeled, accumulating the offset.
///
/// Pipeline position: right after ``AutoTileMatmulL0`` (so the per-iter
/// ``tile.extract``s that read the batch-page slices already exist) and before
/// ``InferTileMemorySpace``.

#include <any>
#include <cstddef>
#include <memory>
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

/// If `assign` is `var = tile.slice(src, shape, [off_row, off_col])`, return the
/// peeled base/offset.  `known` holds slices collected so far; a slice whose
/// source is itself a recorded slice is peeled through it (offsets summed), so
/// `base` is always a non-slice tile.
std::optional<SliceInfo> ParseCanonicalSlice(const AssignStmtPtr& assign,
                                             const std::unordered_map<const Var*, SliceInfo>& known) {
  if (!assign || !assign->var_) return std::nullopt;
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_ || !IsOp(call, "tile.slice")) return std::nullopt;
  // Only canonical 3-arg slices (input, shape, offset).  A slice carrying
  // valid_shape / drop_dims is not a plain window and is left untouched.
  if (call->args_.size() != 3) return std::nullopt;

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
  ExprPtr off_row = offset->elements_[0];
  ExprPtr off_col = offset->elements_[1];
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

/// Phase 1 — collect every canonical `tile.slice` definition in the function,
/// keyed by its result Var.  AssignStmts are visited in program order, so a
/// chained slice's source is always already recorded.
class SliceCollector : public IRVisitor {
 public:
  std::unordered_map<const Var*, SliceInfo> slices;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto info = ParseCanonicalSlice(op, slices)) {
      slices.emplace(op->var_.get(), *info);
    }
  }
};

/// Phase 2 — rewrite `tile.extract` / matmul (Mat slices) and
/// `tile.col_expand_mul` / `tile.col_expand_add` (Vec slices) consumers so they
/// no longer reference a canonicalizable `tile.slice`.
class CanonicalizeMutator : public IRMutator {
 public:
  explicit CanonicalizeMutator(const std::unordered_map<const Var*, SliceInfo>& slices) : slices_(slices) {}

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> out;
    out.reserve(op->stmts_.size());
    bool changed = false;
    for (const auto& child : op->stmts_) {
      // matmul rewrites splice in `tile.extract` statements, so they are
      // handled here at SeqStmts level rather than in VisitStmt_(AssignStmt).
      if (auto assign = As<AssignStmt>(child)) {
        if (auto rewrite = TryRewriteMatmul(assign)) {
          for (auto& s : *rewrite) out.push_back(std::move(s));
          changed = true;
          continue;
        }
        if (auto rewrite = TryRewriteColExpand(assign)) {
          for (auto& s : *rewrite) out.push_back(std::move(s));
          changed = true;
          continue;
        }
      }
      auto visited = VisitStmt(child);
      if (visited.get() != child.get()) changed = true;
      out.push_back(visited);
    }
    if (!changed) return op;
    return SeqStmts::Flatten(std::move(out), op->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto assign = As<AssignStmt>(base);
    if (!assign) return base;
    auto call = As<Call>(assign->value_);
    if (!call || !call->op_ || !IsOp(call, "tile.extract") || call->args_.size() != 4) {
      return base;
    }
    auto src = AsVarLike(call->args_[0]);
    if (!src) return base;
    auto it = slices_.find(src.get());
    if (it == slices_.end() || !it->second.is_mat) return base;

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

 private:
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
  /// target_memory=target)` for a matmul operand that was a Mat slice.  The
  /// slice's result tile shape is forwarded as the extract shape — passing the
  /// existing shape expressions through (rather than extracting int64 values
  /// and rebuilding ConstInts) keeps the path safe under future symbolic dims.
  AssignStmtPtr BuildOperandExtract(const VarPtr& slice_var, const SliceInfo& info, MemorySpace target,
                                    const Span& span) {
    auto slice_tile = As<TileType>(slice_var->GetType());
    INTERNAL_CHECK(slice_tile && slice_tile->shape_.size() == 2)
        << "CanonicalizeTileSlice: matmul-operand slice must have a 2-D TileType result";
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
  /// operand via the lazy `pto.textract` path (pto_ops_common.cpp): both
  /// `tile.col_expand_mul` and `tile.col_expand_add` (#1640).
  static bool IsColExpandMaterializingOp(const OpPtr& op) {
    return IsOp(op, "tile.col_expand_mul") || IsOp(op, "tile.col_expand_add");
  }

  /// True when a slice offset is dynamic (either component is not a `ConstInt`).
  /// Only a dynamic offset is the #1640 hazard: `AllocateMemoryAddr` folds a
  /// const offset into `base + off`, so the lazy `pto.textract` materializes the
  /// row into its own (offset-correct) address — an identity copy that leaves
  /// the source intact.  A dynamic offset cannot be encoded as a `ConstInt`
  /// address, so the slice buffer falls back to the bare source base and the
  /// materialization writes the extracted row into the source's row 0.
  static bool IsDynamicSliceOffset(const SliceInfo& info) {
    return !As<ConstInt>(info.off_row) || !As<ConstInt>(info.off_col);
  }

  /// If `assign` is `tile.col_expand_mul(a, b)` / `tile.col_expand_add(a, b)`
  /// with a dynamic-offset Vec `tile.slice` operand, return a fresh
  /// `tile.extract(src, off_row, off_col, shape, target_memory=Vec)` for each
  /// sliced operand followed by the rebuilt col-expand op (issue #1640).
  ///
  /// Codegen materializes a subview operand of `pto.tcolexpandmul` /
  /// `pto.tcolexpandadd` via `pto.textract` into the slice's own result buffer
  /// (pto_ops_common.cpp).  For a dynamic-offset slice of a local tile that
  /// buffer inherits — and aliases — the bare source allocation base, so the
  /// materialization corrupts the source.  Materializing through `tile.extract`
  /// (which gets its own fresh non-inherited allocation) instead removes the
  /// aliasing.  Static-offset slices are skipped (`AllocateMemoryAddr` folds the
  /// offset into `base + off`, so their lazy textract is a safe identity copy).
  /// Returns nullopt when no operand is a rewritable dynamic Vec slice.
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
      if (!IsDynamicSliceOffset(it->second)) continue;  // static offset → identity textract, safe
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

  const std::unordered_map<const Var*, SliceInfo>& slices_;
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

    // Phase 2 — fold each slice into its tile.extract / matmul / col_expand_mul
    // consumers.
    CanonicalizeMutator mutator(collector.slices);
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

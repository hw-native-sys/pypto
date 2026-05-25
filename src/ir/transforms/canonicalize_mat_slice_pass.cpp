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

/// CanonicalizeMatSlice
/// --------------------
/// Lowers Mat-resident ``tile.slice`` into the canonical ``tile.extract``
/// form so that Mat->Left/Right movement is unified on ``pto.textract``.
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
/// After all consumers are rewritten the now-dead ``tile.slice`` is dropped.
/// Chained slices (a slice of a slice) are peeled, accumulating the offset.
///
/// Pipeline position: right after ``AutoTileMatmulL0`` (so the per-iter
/// ``tile.extract``s that read the batch-page slices already exist) and before
/// ``InferTileMemorySpace``.

#include <any>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
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
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

constexpr const char* kPassName = "CanonicalizeMatSlice";

ExprPtr MakeIndex(int64_t v, const Span& span) {
  return std::make_shared<ConstInt>(v, DataType::INDEX, span);
}

ExprPtr MakeIndexTuple(int64_t d0, int64_t d1, const Span& span) {
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{MakeIndex(d0, span), MakeIndex(d1, span)}, span);
}

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

/// A Mat-resident `tile.slice` peeled to its (non-slice) base tile plus the
/// accumulated row/column offset.
struct MatSliceInfo {
  VarPtr base;      ///< Tile the consumer's `tile.extract` should read from.
  ExprPtr off_row;  ///< Row offset to fold into the consumer index.
  ExprPtr off_col;  ///< Column offset to fold into the consumer index.
};

/// If `assign` is `var = tile.slice(src, shape, [off_row, off_col])` with a
/// Mat-resident result, return the peeled base/offset.  `known` holds slices
/// collected so far; a slice whose source is itself a Mat slice is peeled
/// through it (offsets summed), so `base` is always a non-slice tile.
std::optional<MatSliceInfo> ParseMatSlice(const AssignStmtPtr& assign,
                                          const std::unordered_map<const Var*, MatSliceInfo>& known) {
  if (!assign || !assign->var_) return std::nullopt;
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_ || call->op_->name_ != "tile.slice") return std::nullopt;
  // Only canonical 3-arg slices (input, shape, offset).  A slice carrying
  // valid_shape / drop_dims is not a plain window and is left untouched.
  if (call->args_.size() != 3) return std::nullopt;
  if (!IsMatTile(assign->var_->GetType())) return std::nullopt;

  auto src = AsVarLike(call->args_[0]);
  if (!src) return std::nullopt;
  auto offset = As<MakeTuple>(call->args_[2]);
  if (!offset || offset->elements_.size() != 2) return std::nullopt;

  ExprPtr off_row = offset->elements_[0];
  ExprPtr off_col = offset->elements_[1];
  VarPtr base = src;
  // Peel a chained Mat slice: src itself may be a slice we already recorded.
  auto it = known.find(src.get());
  if (it != known.end()) {
    base = it->second.base;
    off_row = MakeCanonicalIndexAdd(it->second.off_row, off_row, assign->span_);
    off_col = MakeCanonicalIndexAdd(it->second.off_col, off_col, assign->span_);
  }
  return MatSliceInfo{base, off_row, off_col};
}

/// Phase 1 — collect every Mat-resident `tile.slice` definition in the
/// function, keyed by its result Var.  AssignStmts are visited in program
/// order, so a chained slice's source is always already recorded.
class MatSliceCollector : public IRVisitor {
 public:
  std::unordered_map<const Var*, MatSliceInfo> slices;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto info = ParseMatSlice(op, slices)) {
      slices.emplace(op->var_.get(), *info);
    }
  }
};

/// Phase 2 — rewrite `tile.extract` / matmul consumers so they no longer
/// reference any Mat-resident `tile.slice`.
class CanonicalizeMutator : public IRMutator {
 public:
  explicit CanonicalizeMutator(const std::unordered_map<const Var*, MatSliceInfo>& slices)
      : slices_(slices) {}

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
    if (!call || !call->op_ || call->op_->name_ != "tile.extract" || call->args_.size() != 4) {
      return base;
    }
    auto src = AsVarLike(call->args_[0]);
    if (!src) return base;
    auto it = slices_.find(src.get());
    if (it == slices_.end()) return base;

    // extract(slice(base, _, [or, oc]), ir, ic, shape)
    //   -> extract(base, ir + or, ic + oc, shape)
    const auto& info = it->second;
    const Span& sp = call->span_;
    std::vector<ExprPtr> args = {info.base, MakeCanonicalIndexAdd(call->args_[1], info.off_row, sp),
                                 MakeCanonicalIndexAdd(call->args_[2], info.off_col, sp), call->args_[3]};
    auto& reg = OpRegistry::GetInstance();
    auto new_call = reg.Create("tile.extract", args, call->kwargs_, sp);
    return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
  }

 private:
  /// Operand layout of the matmul family: (lhs index, rhs index) or nullopt.
  static std::optional<std::pair<size_t, size_t>> MatmulOperandIndices(const CallPtr& call) {
    if (!call || !call->op_) return std::nullopt;
    const std::string& name = call->op_->name_;
    if (name == "tile.matmul" || name == "tile.matmul_bias") {
      return call->args_.size() >= 2 ? std::optional<std::pair<size_t, size_t>>({0, 1}) : std::nullopt;
    }
    if (name == "tile.matmul_acc") {
      return call->args_.size() >= 3 ? std::optional<std::pair<size_t, size_t>>({1, 2}) : std::nullopt;
    }
    return std::nullopt;
  }

  /// Build `var = tile.extract(base, off_row, off_col, [d0, d1],
  /// target_memory=target)` for a matmul operand that was a Mat slice.
  AssignStmtPtr BuildOperandExtract(const VarPtr& slice_var, const MatSliceInfo& info, MemorySpace target,
                                    const Span& span) {
    auto slice_tile = As<TileType>(slice_var->GetType());
    auto d0 = As<ConstInt>(slice_tile->shape_[0]);
    auto d1 = As<ConstInt>(slice_tile->shape_[1]);
    std::vector<ExprPtr> args = {info.base, info.off_row, info.off_col,
                                 MakeIndexTuple(d0->value_, d1->value_, span)};
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
      if (it == slices_.end()) return;
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
    std::vector<StmtPtr> out = std::move(extracts);
    out.push_back(std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_));
    return out;
  }

  const std::unordered_map<const Var*, MatSliceInfo>& slices_;
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

Pass CanonicalizeMatSlice() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_) return func;
    if (!IsInCoreType(func->func_type_)) return func;

    // Phase 1 — index every Mat-resident tile.slice.
    MatSliceCollector collector;
    collector.VisitStmt(func->body_);
    if (collector.slices.empty()) return func;

    // Phase 2 — fold each slice into its tile.extract / matmul consumers.
    CanonicalizeMutator mutator(collector.slices);
    auto new_body = mutator.VisitStmt(func->body_);

    // Phase 3 — drop the Mat-slice defs that no longer have any use.  A chained
    // slice (a slice of a slice) only becomes dead once the slice that consumes
    // it is dropped, so iterate to a fixpoint — bounded by the Mat-slice count,
    // since every non-terminating iteration drops at least one statement.  A
    // Mat slice still used at the end had a consumer this pass does not
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
    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, new_body, func->span_, func->func_type_,
                                      func->level_, func->role_, func->attrs_);
  };
  return CreateFunctionPass(pass_func, kPassName, kCanonicalizeMatSliceProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

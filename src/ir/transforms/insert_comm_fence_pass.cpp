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
 * @file insert_comm_fence_pass.cpp
 * @brief Insert `system.cacheinvalid` after each publishing write and a GM
 *        `system.fence` before each releasing `pld.system.notify`
 *        (data-before-signal).
 *
 * The latest PTOAS requires the upper layer to explicitly order a cross-rank
 * write against the notify that signals its completion: the written data must be
 * visible to the peer before the signal arrives. Two ops implement this, at two
 * different granularities and — because they share nothing — via two independent
 * traversals:
 *
 *   - `system.cacheinvalid` is **per address**: each publishing write invalidates
 *     the cache lines of exactly the region it wrote, so one cacheinvalid is
 *     emitted **immediately after every publishing write** (`CacheInvalidInserter`,
 *     a single structural traversal — no control-flow analysis). Placing it at the
 *     write site keeps its target trivially in scope, so there is no cross-scope
 *     tracking and nothing is ever silently dropped.
 *   - `system.fence` is **per notify**: a single GM barrier before a notify
 *     orders *all* prior writes, so at most one fence is emitted before a notify
 *     that has an unflushed publishing write (`FenceInserter`, carrying a
 *     `pending` bool over a memoized subtree summary).
 *
 * So the shapes are, e.g.:
 *
 *   remote_store(a); remote_store(b); notify
 *     -> remote_store(a); cacheinvalid(a); remote_store(b); cacheinvalid(b); fence; notify
 *
 * The cacheinvalid currently covers the **whole target tensor** (region = full
 * shape at zero offsets), reusing the tensor type's dim exprs. Narrowing to the
 * precise written sub-region (available from the write's own args at the write
 * site) is a planned follow-up.
 *
 * Fence placement (`FenceInserter`), two passes, O(N):
 *
 * - Pass 1 (`SummaryCache`, bottom-up / post-order, memoized): per statement,
 *   `opens_with_notify` (reaches a notify before any write/fence),
 *   `may_end_with_write` (may exit with an uncovered publishing write),
 *   `transparent` (may fall through touching nothing). Loops are always
 *   `transparent` (they may iterate zero times).
 *
 * - Pass 2 (forward): carry a `pending` bool. At each `SeqStmts`, before a child
 *   that `opens_with_notify`, emit a fence if `pending` — except before an `if`,
 *   which is recursed into so each branch fences at its own real notify (a
 *   conditional notify's barrier belongs in the taken branch). Loops still hoist
 *   (the body runs every iteration). A loop body is entered with
 *   `pending || may_end_with_write(body)` (the ring back-edge). An existing fence
 *   clears `pending`, so it is idempotent.
 *
 * Runs last in the Default pipeline (after all statement-reordering passes) so
 * the inserted ops stay adjacent through codegen. See
 * `op_predicates::IsPublishingWrite` / `IsNotify` for the write/notify sets.
 */

#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

// The effect a leaf statement has on the running `pending` state.
enum class Effect { kWrite, kNotify, kFence, kNone };

// The call carried by a leaf statement, if any.
CallPtr LeafCall(const StmtPtr& stmt) {
  ExprPtr value;
  if (auto eval = As<EvalStmt>(stmt)) {
    value = eval->expr_;
  } else if (auto assign = As<AssignStmt>(stmt)) {
    value = assign->value_;
  }
  return value ? As<Call>(value) : nullptr;
}

// Classify a call-like statement (EvalStmt / AssignStmt). A submit is treated
// conservatively as a publishing write for fence ordering; its launched task
// body is not analysed here (and it has no single region to cacheinvalidate).
Effect StmtEffect(const StmtPtr& stmt) {
  ExprPtr value;
  if (auto eval = As<EvalStmt>(stmt)) {
    value = eval->expr_;
  } else if (auto assign = As<AssignStmt>(stmt)) {
    value = assign->value_;
  }
  if (!value) return Effect::kNone;
  if (auto call = As<Call>(value)) {
    if (op_predicates::IsPublishingWrite(call)) return Effect::kWrite;
    if (op_predicates::IsNotify(call)) return Effect::kNotify;
    if (IsOp(call, "system.fence")) return Effect::kFence;
    return Effect::kNone;
  }
  if (As<Submit>(value)) return Effect::kWrite;
  return Effect::kNone;
}

// The destination tensor of a publishing write, whose cache lines must be
// invalidated after it. Null when there is no single addressable target (a
// `Submit`, or an op not in the publishing set). Callers guard on
// `IsPublishingWrite` first; the arg positions mirror each op's registration.
ExprPtr PublishingWriteTarget(const CallPtr& call) {
  if (!call || !call->op_) return nullptr;
  if (IsOp(call, "pld.tile.remote_store")) {
    return call->args_.size() > 1 ? call->args_[1] : nullptr;  // (src_tile, target, ...)
  }
  if (IsOp(call, "pld.tile.put") || IsOp(call, "pld.tensor.put") || IsOp(call, "pld.tile.get") ||
      IsOp(call, "pld.tensor.get")) {
    return call->args_.empty() ? nullptr : call->args_[0];  // (dst, ...)
  }
  if (IsOp(call, "tile.store")) {
    return call->args_.size() > 2 ? call->args_[2] : nullptr;  // (tile, indices, dst)
  }
  return nullptr;
}

// A target tensor usable for cacheinvalid: a `Var`-like with a `TensorType`.
VarPtr AsInvalidatableTarget(const ExprPtr& target) {
  if (!target) return nullptr;
  auto var = AsVarLike(target);
  if (!var || !AsTensorTypeLike(var->GetType())) return nullptr;
  return var;
}

// The invalidatable target of `stmt` if it is a publishing write, else null.
ExprPtr WriteTargetToInvalidate(const StmtPtr& stmt) {
  if (StmtEffect(stmt) != Effect::kWrite) return nullptr;
  auto target = PublishingWriteTarget(LeafCall(stmt));
  return AsInvalidatableTarget(target) ? target : nullptr;
}

// True if `stmt` is a `system.cacheinvalid` whose target Var is `target`.
bool IsCacheInvalidFor(const StmtPtr& stmt, const ExprPtr& target) {
  auto call = LeafCall(stmt);
  if (!call || !IsOp(call, "system.cacheinvalid") || call->args_.empty()) return false;
  auto have = AsVarLike(call->args_[0]);
  auto want = AsVarLike(target);
  return have && want && have.get() == want.get();
}

// Whole-tensor cacheinvalid for `target`: region = the target's full shape at
// all-zero offsets. Reuses the tensor type's dim exprs (in scope — the target
// was just written), so no per-write offset SSA is needed.
StmtPtr MakeCacheInvalid(const ExprPtr& target, const Span& span) {
  auto var = AsInvalidatableTarget(target);
  INTERNAL_CHECK_SPAN(var, span)
      << "Internal error: cacheinvalid target must be a tensor Var (checked before insert)";
  auto tensor_type = AsTensorTypeLike(var->GetType());
  std::vector<ExprPtr> shape_elems = tensor_type->shape_;
  std::vector<ExprPtr> zero_offsets;
  zero_offsets.reserve(shape_elems.size());
  for (size_t i = 0; i < shape_elems.size(); ++i) {
    zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, span));
  }
  auto shapes_tuple = std::make_shared<MakeTuple>(std::move(shape_elems), span);
  auto offsets_tuple = std::make_shared<MakeTuple>(std::move(zero_offsets), span);
  auto call =
      OpRegistry::GetInstance().Create("system.cacheinvalid", {target, shapes_tuple, offsets_tuple}, span);
  return std::make_shared<EvalStmt>(call, span);
}

// ---------------------------------------------------------------------------
// Traversal 1: insert a whole-tensor `system.cacheinvalid` right after every
// publishing write. Pure structural rewrite — no control-flow analysis.
// ---------------------------------------------------------------------------
class CacheInvalidInserter : public IRMutator {
 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> out;
    out.reserve(op->stmts_.size());
    bool changed = false;
    const auto& stmts = op->stmts_;
    for (size_t i = 0; i < stmts.size(); ++i) {
      auto new_child = VisitStmt(stmts[i]);
      if (new_child.get() != stmts[i].get()) changed = true;
      out.push_back(std::move(new_child));
      if (auto target = WriteTargetToInvalidate(stmts[i])) {
        // Idempotent: skip if the next original sibling already is that cacheinvalid.
        const bool already = (i + 1 < stmts.size()) && IsCacheInvalidFor(stmts[i + 1], target);
        if (!already) {
          out.push_back(MakeCacheInvalid(target, stmts[i]->span_));
          changed = true;
        }
      }
    }
    if (!changed) return op;
    return SeqStmts::Flatten(std::move(out), op->span_);
  }

  // Bare single-statement bodies (a publishing write not wrapped in a SeqStmts,
  // e.g. `if c: remote_store(...)`) have no SeqStmts handler to append to, so
  // wrap them here. After the first run the body becomes a SeqStmts and takes the
  // handler above, which keeps it idempotent.
  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto new_then = WrapBareWrite(op->then_body_);
    std::optional<StmtPtr> new_else = op->else_body_;
    if (op->else_body_.has_value()) new_else = WrapBareWrite(op->else_body_.value());
    const bool then_changed = new_then.get() != op->then_body_.get();
    const bool else_changed = op->else_body_.has_value() && new_else->get() != op->else_body_->get();
    if (!then_changed && !else_changed) return op;
    auto result = MutableCopy(op);
    result->then_body_ = std::move(new_then);
    result->else_body_ = std::move(new_else);
    return result;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return VisitLoopBody(op, op->body_); }
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return VisitLoopBody(op, op->body_); }

 private:
  StmtPtr WrapBareWrite(const StmtPtr& body) {
    auto visited = VisitStmt(body);
    auto target = WriteTargetToInvalidate(body);
    if (!target) return visited;  // SeqStmts / non-write body: handled elsewhere
    return SeqStmts::Flatten({visited, MakeCacheInvalid(target, body->span_)}, body->span_);
  }

  template <typename LoopPtr>
  StmtPtr VisitLoopBody(const LoopPtr& op, const StmtPtr& body) {
    // WrapBareWrite already fully visited `body`; if it is unchanged the loop is
    // unchanged — return `op` directly (re-invoking IRMutator::VisitStmt_ would
    // walk the body a second time, O(2^depth) over nested loops).
    auto new_body = WrapBareWrite(body);
    if (new_body.get() == body.get()) return op;
    auto result = MutableCopy(op);
    result->body_ = std::move(new_body);
    return result;
  }
};

// ---------------------------------------------------------------------------
// Traversal 2: insert a GM `system.fence` before each releasing notify.
// ---------------------------------------------------------------------------
struct Summary {
  bool opens_with_notify = false;   // reaches a notify before any write/fence
  bool may_end_with_write = false;  // may exit with an uncovered publishing write
  bool transparent = false;         // may fall through touching no write/fence/notify
};

// Pass 1: memoized bottom-up subtree summaries (drives fence placement).
class SummaryCache {
 public:
  const Summary& Get(const StmtPtr& stmt) {
    auto it = cache_.find(stmt.get());
    if (it != cache_.end()) return it->second;
    Summary summary = Compute(stmt);
    return cache_.emplace(stmt.get(), summary).first->second;
  }

 private:
  Summary Compute(const StmtPtr& stmt) {
    if (auto seq = As<SeqStmts>(stmt)) {
      bool opens = false;
      bool may_end = false;
      bool transparent = true;
      bool clear = true;  // some path reaches here without a write/fence/notify
      for (const auto& child : seq->stmts_) {
        const Summary& cs = Get(child);
        if (clear && cs.opens_with_notify) opens = true;
        clear = clear && cs.transparent;
        may_end = cs.may_end_with_write || (cs.transparent && may_end);
        transparent = transparent && cs.transparent;
      }
      return {opens, may_end, transparent};
    }
    if (auto iff = As<IfStmt>(stmt)) {
      const Summary& then_s = Get(iff->then_body_);
      Summary else_s =
          iff->else_body_.has_value() ? Get(iff->else_body_.value()) : Summary{false, false, true};
      return {then_s.opens_with_notify || else_s.opens_with_notify,
              then_s.may_end_with_write || else_s.may_end_with_write,
              then_s.transparent || else_s.transparent};
    }
    if (auto loop = As<ForStmt>(stmt)) {
      const Summary& body = Get(loop->body_);
      return {body.opens_with_notify, body.may_end_with_write, true};
    }
    if (auto loop = As<WhileStmt>(stmt)) {
      const Summary& body = Get(loop->body_);
      return {body.opens_with_notify, body.may_end_with_write, true};
    }
    if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      return Get(scope->body_);
    }
    switch (StmtEffect(stmt)) {
      case Effect::kWrite:
        return {false, true, false};
      case Effect::kNotify:
        return {true, false, false};
      case Effect::kFence:
        return {false, false, false};
      case Effect::kNone:
        return {false, false, true};
    }
    return {false, false, true};
  }

  std::unordered_map<const Stmt*, Summary> cache_;
};

// Pass 2: forward insertion carrying a single `pending` bool.
class FenceInserter : public IRMutator {
 public:
  explicit FenceInserter(SummaryCache* summaries) : summaries_(summaries) {}

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> out;
    out.reserve(op->stmts_.size());
    bool changed = false;
    for (const auto& child : op->stmts_) {
      if (MaybeFenceBefore(&out, child)) changed = true;
      auto new_child = VisitStmt(child);
      if (new_child.get() != child.get()) changed = true;
      out.push_back(std::move(new_child));
    }
    if (!changed) return op;
    return SeqStmts::Flatten(std::move(out), op->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return VisitLoop(op, op->body_); }
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return VisitLoop(op, op->body_); }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    const bool pending_in = pending_;
    auto new_then = VisitBranch(op->then_body_);
    const bool pending_then = pending_;
    bool pending_else = pending_in;
    std::optional<StmtPtr> new_else = op->else_body_;
    if (op->else_body_.has_value()) {
      pending_ = pending_in;
      new_else = VisitBranch(op->else_body_.value());
      pending_else = pending_;
    }
    pending_ = pending_then || pending_else;
    const bool then_changed = new_then.get() != op->then_body_.get();
    const bool else_changed = op->else_body_.has_value() && new_else->get() != op->else_body_->get();
    if (!then_changed && !else_changed) return op;
    auto result = MutableCopy(op);
    result->then_body_ = std::move(new_then);
    result->else_body_ = std::move(new_else);
    return result;
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    pending_ = NextPending(pending_, StmtEffect(op));
    return op;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    pending_ = NextPending(pending_, StmtEffect(op));
    return op;
  }

 private:
  static bool NextPending(bool pending, Effect effect) {
    switch (effect) {
      case Effect::kWrite:
        return true;
      case Effect::kNotify:
      case Effect::kFence:
        return false;
      case Effect::kNone:
        return pending;
    }
    return pending;
  }

  // Emit a fence into `out` before `next` when a write is pending and `next`
  // opens with a notify — with two exceptions that are recursed into instead:
  //   - an `if`, so each branch fences at its own real notify (a conditional
  //     notify's barrier belongs in the taken branch);
  //   - a loop whose body may end with a write, which already fences at its own
  //     head once the incoming pending write flows in (the ring back-edge seed) —
  //     a hoisted fence here would be a redundant second barrier before the same
  //     first-iteration notify. A loop that does *not* end with a write (e.g. a
  //     `for p: notify` barrier) still hoists, so one fence covers all iterations
  //     instead of one per iteration.
  // Returns true if a fence was inserted.
  bool MaybeFenceBefore(std::vector<StmtPtr>* out, const StmtPtr& next) {
    const Summary& s = summaries_->Get(next);
    if (!pending_ || As<IfStmt>(next) || !s.opens_with_notify) return false;
    if ((As<ForStmt>(next) || As<WhileStmt>(next)) && s.may_end_with_write) return false;
    out->push_back(MakeFence(next->span_));
    pending_ = false;
    return true;
  }

  // Visit an if-branch body. A `SeqStmts` body inserts internally; a bare
  // single-statement notify body has no `SeqStmts` handler, so fence around it.
  StmtPtr VisitBranch(const StmtPtr& body) {
    if (As<SeqStmts>(body)) return VisitStmt(body);
    std::vector<StmtPtr> out;
    if (!MaybeFenceBefore(&out, body)) return VisitStmt(body);
    out.push_back(VisitStmt(body));
    return SeqStmts::Flatten(std::move(out), body->span_);
  }

  template <typename LoopPtr>
  StmtPtr VisitLoop(const LoopPtr& op, const StmtPtr& body) {
    // Back-edge: a write at the tail of one iteration reaches a notify at the
    // head of the next, so enter the body already pending if the body may write.
    const bool pending_in = pending_ || summaries_->Get(body).may_end_with_write;
    pending_ = pending_in;
    auto result = IRMutator::VisitStmt_(op);  // reconstructs the loop, visiting `body`
    pending_ = pending_in;                    // loop exit state (may run zero times)
    return result;
  }

  StmtPtr MakeFence(const Span& span) {
    auto fence_call = OpRegistry::GetInstance().Create("system.fence", /*args=*/{}, span);
    return std::make_shared<EvalStmt>(fence_call, span);
  }

  SummaryCache* summaries_;
  bool pending_ = false;
};

}  // namespace

Pass InsertCommFence() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_) return func;
    // Independent traversals: per-address cacheinvalid after each write, then the
    // per-notify fence. Order is irrelevant — a cacheinvalid is inert to the fence
    // summary — so cacheinvalid runs first for a simpler mental model.
    auto with_ci = CacheInvalidInserter().VisitStmt(func->body_);
    SummaryCache summaries;
    auto new_body = FenceInserter(&summaries).VisitStmt(with_ci);
    if (new_body.get() == func->body_.get()) return func;
    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, new_body, func->span_, func->func_type_,
                                      func->level_, func->role_, func->attrs_);
  };
  return CreateFunctionPass(pass_func, "InsertCommFence", kInsertCommFenceProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

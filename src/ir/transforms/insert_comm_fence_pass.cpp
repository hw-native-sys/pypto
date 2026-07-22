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
 * @brief Insert the ptoas data-before-signal memory markers around cross-rank
 *        publish (`pld.system.notify`) and consume (`pld.system.wait`) points.
 *
 * The latest PTOAS enforces a two-sided contract in its `pto-memory-consistency`
 * pass and pushes the markers onto the compiler:
 *
 *   - Publish side: a `pto.comm.tnotify` requires an explicit
 *     `pto.fence.barrier_all #pto.fence_scope<gm>` after the matching
 *     `pto.cmo.cacheinvalid` release marker and before the signal.
 *   - Consume side: a cacheable GM load after `pto.comm.twait` / a successful
 *     `pto.comm.ttest` requires an explicit `pto.cmo.cacheinvalid all
 *     #pto.address_space<gm>` first (so the reader sees the peer's fresh write).
 *
 * Both markers are the same `system.cacheinvalid` op: with a (tensor, shapes,
 * offsets) region it invalidates that sub-region; with no argument it
 * invalidates the whole GM address space (`... cacheinvalid all ...`).
 *
 * A single forward structural traversal inserts, per op:
 *
 *   - after each **publishing write** (remote_store / put / get / window-bound
 *     tile.store): a whole-tensor region `system.cacheinvalid` of the written
 *     region followed **immediately** by a GM `system.fence`. ptoas requires the
 *     fence to directly follow the release marker, so both land at the write site
 *     (per address) — not deferred to the notify.
 *   - before each **bare barrier notify** (a notify with no fenced publishing
 *     write pending — e.g. a pure barrier signal, or a notify whose write lives in
 *     a prior loop so `pending` was reset): a no-arg (whole-GM)
 *     `system.cacheinvalid` + `system.fence` as the release marker (there is no
 *     data region to point at). A notify that *does* have a pending fenced write
 *     needs nothing — the write's own `cacheinvalid; fence` is the marker.
 *   - after each **wait**: a no-arg (whole-GM) `system.cacheinvalid` (the
 *     consume-side invalidate before the next cacheable read).
 *
 * Example shapes:
 *
 *   remote_store; notify         -> remote_store; cacheinvalid(dst); fence; notify
 *   for p: (if p!=me: notify)     -> for p: (if p!=me: cacheinvalid; fence; notify)
 *   for c: store; for p: notify   -> for c: (store; cacheinvalid; fence);
 *                                     for p: (cacheinvalid; fence; notify)
 *   wait; read                   -> wait; cacheinvalid; read
 *
 * The region cacheinvalid covers the whole target tensor (full shape at zero
 * offsets), reusing the tensor type's dim exprs; narrowing to the precise written
 * sub-region is a planned follow-up.
 *
 * A single forward `pending` bool decides the notify marker: it is set by a
 * publishing write (which also emits its own fence) and cleared by a notify/fence.
 * It is conservative across control flow (reset after a loop; `and`-merged across
 * `if` branches), so an uncertain path emits the safe whole-GM cacheinvalid +
 * fence rather than relying on a region marker that may not precede.
 *
 * Idempotent: a notify already preceded by a fence, a write already followed by
 * its region cacheinvalid, and a wait already followed by a whole-GM cacheinvalid
 * are left alone. Runs last in the Default pipeline (after all statement-reordering
 * passes) so the inserted ops stay adjacent through codegen.
 */

#include <cstddef>
#include <memory>
#include <optional>
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

// The effect a leaf statement has on the running `pending` state / markers.
enum class Effect { kWrite, kNotify, kWait, kFence, kNone };

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
// conservatively as a publishing write for ordering; its launched task body is
// not analysed here (and it has no single region to cacheinvalidate).
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
    if (IsOp(call, "pld.system.wait")) return Effect::kWait;
    if (IsOp(call, "system.fence")) return Effect::kFence;
    return Effect::kNone;
  }
  if (As<Submit>(value)) return Effect::kWrite;
  return Effect::kNone;
}

// `pending` after a leaf with the given effect. A wait / cacheinvalid does not
// touch the publish-side pending state.
bool NextPending(bool pending, Effect effect) {
  switch (effect) {
    case Effect::kWrite:
      return true;
    case Effect::kNotify:
    case Effect::kFence:
      return false;
    case Effect::kWait:
    case Effect::kNone:
      return pending;
  }
  return pending;
}

// The destination tensor of a publishing write, whose cache lines must be
// invalidated after it. Null when there is no single addressable target (a
// `Submit`, or an op not in the publishing set). The arg positions mirror each
// op's registration.
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

bool IsLeafOp(const StmtPtr& stmt, const char* op_name) {
  auto call = LeafCall(stmt);
  return call && IsOp(call, op_name);
}

// True if `stmt` is a region `system.cacheinvalid` whose target Var is `target`.
bool IsCacheInvalidFor(const StmtPtr& stmt, const ExprPtr& target) {
  auto call = LeafCall(stmt);
  if (!call || !IsOp(call, "system.cacheinvalid") || call->args_.empty()) return false;
  auto have = AsVarLike(call->args_[0]);
  auto want = AsVarLike(target);
  return have && want && have.get() == want.get();
}

// True if `stmt` is a whole-GM `system.cacheinvalid` (the no-argument form).
bool IsCacheInvalidAll(const StmtPtr& stmt) {
  auto call = LeafCall(stmt);
  return call && IsOp(call, "system.cacheinvalid") && call->args_.empty();
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

StmtPtr MakeNoArgOp(const char* op_name, const Span& span) {
  return std::make_shared<EvalStmt>(OpRegistry::GetInstance().Create(op_name, /*args=*/{}, span), span);
}

// Whole-GM cacheinvalid: the no-argument form of `system.cacheinvalid`.
StmtPtr MakeCacheInvalidAll(const Span& span) { return MakeNoArgOp("system.cacheinvalid", span); }

// Forward traversal: emit the ptoas data-before-signal markers per op.
class InsertCommMarkers : public IRMutator {
 public:
  // Entry point: process a function body. A SeqStmts is walked statement by
  // statement; a bare single statement (a degenerate one-op body) is wrapped
  // with its own markers, matching the treatment of if/for bodies.
  StmtPtr MarkTopLevel(const StmtPtr& body) { return MarkBody(body); }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> out;
    out.reserve(op->stmts_.size());
    bool changed = false;
    const auto& stmts = op->stmts_;
    for (size_t i = 0; i < stmts.size(); ++i) {
      const StmtPtr& child = stmts[i];
      const Effect eff = StmtEffect(child);
      // Publish side, bare barrier notify: when no fenced publishing write is
      // pending, the notify has no data region to release, so emit a whole-GM
      // cacheinvalid + fence before it (unless already fenced). A pending write
      // already emitted its own `cacheinvalid; fence` — ptoas accepts that as the
      // release marker — so a pending notify needs nothing here.
      if (eff == Effect::kNotify && !pending_ && !(i > 0 && IsLeafOp(stmts[i - 1], "system.fence"))) {
        out.push_back(MakeCacheInvalidAll(child->span_));
        out.push_back(MakeNoArgOp("system.fence", child->span_));
        changed = true;
      }
      auto new_child = VisitStmt(child);
      if (new_child.get() != child.get()) changed = true;
      out.push_back(std::move(new_child));
      // Per-address: a region cacheinvalid + fence after each publishing write.
      // ptoas requires the fence to immediately follow the release marker.
      if (auto target = WriteTargetToInvalidate(child)) {
        const bool already = i + 2 < stmts.size() && IsCacheInvalidFor(stmts[i + 1], target) &&
                             IsLeafOp(stmts[i + 2], "system.fence");
        if (!already) {
          out.push_back(MakeCacheInvalid(target, child->span_));
          out.push_back(MakeNoArgOp("system.fence", child->span_));
          changed = true;
        }
      }
      // Consume side: a whole-GM cacheinvalid after each wait.
      if (eff == Effect::kWait && !(i + 1 < stmts.size() && IsCacheInvalidAll(stmts[i + 1]))) {
        out.push_back(MakeCacheInvalidAll(child->span_));
        changed = true;
      }
      pending_ = NextPending(pending_, eff);
    }
    if (!changed) return op;
    return SeqStmts::Flatten(std::move(out), op->span_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    const bool pending_in = pending_;
    auto new_then = MarkBody(op->then_body_);
    const bool then_pending = pending_;
    bool else_pending = pending_in;
    std::optional<StmtPtr> new_else = op->else_body_;
    if (op->else_body_.has_value()) {
      pending_ = pending_in;
      new_else = MarkBody(op->else_body_.value());
      else_pending = pending_;
    }
    // Conservative: only stay `pending` if both paths definitely leave a write
    // pending; otherwise the next notify emits the safe cacheinvalid_all.
    pending_ = then_pending && else_pending;
    const bool then_changed = new_then.get() != op->then_body_.get();
    const bool else_changed = op->else_body_.has_value() && new_else->get() != op->else_body_->get();
    if (!then_changed && !else_changed) return op;
    auto result = MutableCopy(op);
    result->then_body_ = std::move(new_then);
    result->else_body_ = std::move(new_else);
    return result;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return VisitLoop(op, op->body_); }
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return VisitLoop(op, op->body_); }

 private:
  // Visit a body that may be a bare single statement (an `if`/`for` body without
  // an enclosing SeqStmts, e.g. `if p != me: notify`). A SeqStmts body is handled
  // by its own visitor; a bare leaf gets its markers wrapped around it here. After
  // the first run a wrapped body is a SeqStmts, so the pass stays idempotent.
  StmtPtr MarkBody(const StmtPtr& body) {
    if (As<SeqStmts>(body)) return VisitStmt(body);
    const Effect eff = StmtEffect(body);
    std::vector<StmtPtr> out;
    if (eff == Effect::kNotify && !pending_) {
      out.push_back(MakeCacheInvalidAll(body->span_));
      out.push_back(MakeNoArgOp("system.fence", body->span_));
    }
    auto visited = VisitStmt(body);
    out.push_back(visited);
    if (auto target = WriteTargetToInvalidate(body)) {
      out.push_back(MakeCacheInvalid(target, body->span_));
      out.push_back(MakeNoArgOp("system.fence", body->span_));
    }
    if (eff == Effect::kWait) out.push_back(MakeCacheInvalidAll(body->span_));
    pending_ = NextPending(pending_, eff);
    if (out.size() == 1) return visited;
    return SeqStmts::Flatten(std::move(out), body->span_);
  }

  template <typename LoopPtr>
  StmtPtr VisitLoop(const LoopPtr& op, const StmtPtr& body) {
    // Enter the loop body with `pending_ = false`: ptoas checks the release marker
    // lexically, so a loop-head notify cannot rely on a fence carried in from
    // before the loop (iteration 0) or from the previous iteration's tail write
    // (the back-edge) — neither precedes it lexically in the body. Marking with a
    // cleared `pending` forces every loop-body notify to get its own
    // cacheinvalid + fence. After the loop, stay conservative for the same reason.
    pending_ = false;
    auto new_body = MarkBody(body);
    pending_ = false;
    if (new_body.get() == body.get()) return op;
    auto result = MutableCopy(op);
    result->body_ = std::move(new_body);
    return result;
  }

  bool pending_ = false;
};

}  // namespace

Pass InsertCommFence() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_) return func;
    InsertCommMarkers mutator;
    auto new_body = mutator.MarkTopLevel(func->body_);
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

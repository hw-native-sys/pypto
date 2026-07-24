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
 * pass and pushes the markers onto the compiler. Verified empirically on ptoas
 * 0.50, the contract reduces to exactly two purely-local rules — the *notify*
 * itself needs nothing:
 *
 *   - Publish side: each publishing GM write requires a `pto.cmo.cacheinvalid`
 *     of the written region **immediately followed by** a
 *     `pto.fence.barrier_all #pto.fence_scope<gm>`. Any later `pto.comm.tnotify`
 *     that releases that data is satisfied by this fence — including a notify in a
 *     different loop; the fence does *not* need to sit next to the notify. A pure
 *     barrier notify (no data at all) needs nothing.
 *   - Consume side: a cacheable GM load after `pto.comm.twait` / a successful
 *     `pto.comm.ttest` requires a `pto.cmo.cacheinvalid all #pto.address_space<gm>`
 *     first (so the reader sees the peer's fresh write).
 *
 * Both markers are the same `system.cacheinvalid` op: with a (tensor, shapes,
 * offsets) region it invalidates that sub-region; with no argument it
 * invalidates the whole GM address space (`... cacheinvalid all ...`).
 *
 * So a single structural traversal inserts, per op, with no control-flow state:
 *
 *   - after each **local publishing write** (window-bound `tile.store`, or `get`
 *     into a local destination): only a GM `system.fence`. The publish-side region
 *     cacheinvalid this rule calls for is currently NOT emitted — see the
 *     "region cacheinvalid suspended" note below.
 *   - after each **remote publishing write** (`remote_store` / `put`): only a GM
 *     `system.fence`. The data lands at a peer-offset GM address that a
 *     local-target cacheinvalid cannot address; the peer offset is not yet
 *     expressible in the IR, so the peer-region cacheinvalid is emitted by the
 *     op's codegen as a WORKAROUND (see pto_ops_distributed.cpp). The GM release
 *     fence, however, is always an explicit `system.fence` op inserted here — the
 *     codegen must not embed it. (TODO: a first-class IR representation of the
 *     peer-region cacheinvalid would let the pass own the whole marker.)
 *   - after each **opaque publishing write** (`Submit`, or a write with no single
 *     addressable region): a whole-GM `system.cacheinvalid` + a GM `system.fence`.
 *   - after each **wait**: a no-arg (whole-GM) `system.cacheinvalid`.
 *   - **notify**: nothing.
 *
 * Example shapes (`cacheinvalid(peer)` shown in brackets is codegen-only, not IR):
 *
 *   remote_store; notify         -> remote_store; [cacheinvalid(peer)]; fence; notify
 *   store(win); notify           -> store(win); fence; notify
 *   wait; read                   -> wait; cacheinvalid(); read
 *
 * ## Region cacheinvalid suspended (publish side, local writes)
 *
 * The publish-side marker should be a whole-tensor region `system.cacheinvalid` of
 * the written target, which lowers to
 * `pto.cmo.cacheinvalid %pview single_cache_line : !pto.partition_tensor_view<...>`.
 * No shipped ptoas turns that into working code:
 *
 *   - ptoas 0.50 parses it and emits **nothing** — the marker never reached the
 *     device, so this pass's publish-side invalidate has always been a no-op.
 *   - ptoas 0.51 emits `PTOAS__DCCI_SINGLE_CACHE_LINE(<GlobalTensor>)`, but that
 *     helper is `dcci((__gm__ void*)ptr, ...)` and a `GlobalTensor` has no
 *     conversion to `__gm__ void*` in any pto-isa revision, so the kernel fails to
 *     compile.
 *
 * Reported as hw-native-sys/PTOAS#995.
 *
 * Emitting only the fence therefore reproduces exactly what runs today, and is the
 * only form that compiles under 0.51. The whole-GM form is unaffected (identical,
 * working lowering on both) and is still emitted after waits and opaque writes.
 * Restore `MakeCacheInvalid` (see git history) once ptoas lowers the region form.
 *
 * Idempotent: a write already followed by its fence, and a wait already followed by
 * a whole-GM cacheinvalid, are left alone. Runs last in the Default pipeline (after
 * all statement-reordering passes) so the inserted ops stay adjacent through
 * codegen.
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

// The effect a leaf statement has on the inserted markers.
enum class Effect { kWrite, kWait, kNone };

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
// not analysed here (and it has no single region to cacheinvalidate). Notifies
// are intentionally *not* classified: ptoas ties the release fence to the write's
// cacheinvalid, not to the notify, so the notify needs no marker of its own.
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
    if (IsOp(call, "pld.system.wait")) return Effect::kWait;
    return Effect::kNone;
  }
  if (As<Submit>(value)) return Effect::kWrite;
  return Effect::kNone;
}

// The local destination tensor of a publishing write, whose cache lines this pass
// invalidates after it (a region `system.cacheinvalid(target)` addresses `target`'s
// local base). Correct for a local-window store (`tile.store`) or a peer-read into
// a local destination (`get`). The **remote** writes `remote_store` / `put` land at
// a peer-offset address (`local_ptr + delems(peer)`) that a local-target
// cacheinvalid cannot address, so they return null here — see `IsRemoteWrite`.
ExprPtr PublishingWriteTarget(const CallPtr& call) {
  if (!call || !call->op_) return nullptr;
  if (IsOp(call, "pld.tile.get") || IsOp(call, "pld.tensor.get")) {
    return call->args_.empty() ? nullptr : call->args_[0];  // (dst, ...) — local destination
  }
  if (IsOp(call, "tile.store")) {
    return call->args_.size() > 2 ? call->args_[2] : nullptr;  // (tile, indices, dst)
  }
  return nullptr;  // remote_store / put -> remote write, see IsRemoteWrite
}

// True if `call` is a remote publishing write (`remote_store` / `put`). Its data
// lands at a peer-offset GM address; the peer offset is only known during codegen
// (`EmitCommRemoteView`) and is not yet expressible in the IR, so its codegen
// emits the peer-region `pto.cmo.cacheinvalid` itself — a WORKAROUND pending a
// first-class IR representation of the peer address. The GM release **fence**,
// however, IS inserted here as an explicit `system.fence`, uniformly with the
// local writes (codegen must not embed the fence).
bool IsRemoteWrite(const CallPtr& call) {
  if (!call || !call->op_) return false;
  return IsOp(call, "pld.tile.remote_store") || IsOp(call, "pld.tile.put") || IsOp(call, "pld.tensor.put");
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

// True if `stmt` is a whole-GM `system.cacheinvalid` (the no-argument form).
bool IsCacheInvalidAll(const StmtPtr& stmt) {
  auto call = LeafCall(stmt);
  return call && IsOp(call, "system.cacheinvalid") && call->args_.empty();
}

StmtPtr MakeNoArgOp(const char* op_name, const Span& span) {
  return std::make_shared<EvalStmt>(OpRegistry::GetInstance().Create(op_name, /*args=*/{}, span), span);
}

// Whole-GM cacheinvalid: the no-argument form of `system.cacheinvalid`.
StmtPtr MakeCacheInvalidAll(const Span& span) { return MakeNoArgOp("system.cacheinvalid", span); }

// Structural traversal: emit `cacheinvalid; fence` after every publishing write
// and `cacheinvalid()` after every wait. No control-flow state is needed — both
// rules are purely local — so if/for/while bodies are visited normally; the only
// special handling is wrapping a bare single-statement body (a write/wait that is
// the sole body of an if/for without an enclosing SeqStmts).
class InsertCommMarkers : public IRMutator {
 public:
  // Entry point: process a function body (delegates to the same bare-body-aware
  // wrapping used for if/for/while bodies).
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
      auto new_child = VisitStmt(child);
      if (new_child.get() != child.get()) changed = true;
      out.push_back(std::move(new_child));
      // Publish side, addressable write: only the GM release fence.
      //   - local write: its region cacheinvalid is suspended (see the file header);
      //   - remote write: codegen emits the peer-region cacheinvalid, since the peer
      //     offset is not IR-expressible yet.
      // Either way the pass owns the fence and nothing else.
      if (WriteTargetToInvalidate(child) || IsRemoteWrite(LeafCall(child))) {
        if (!(i + 1 < stmts.size() && IsLeafOp(stmts[i + 1], "system.fence"))) {
          out.push_back(MakeNoArgOp("system.fence", child->span_));
          changed = true;
        }
      } else if (eff == Effect::kWrite) {
        // Opaque publishing write with no single addressable region — a `Submit`
        // (async task launch) or a call to an unregistered/un-analysed op. Be
        // conservative: a whole-GM cacheinvalid + GM fence covers whatever it wrote.
        const bool already =
            i + 2 < stmts.size() && IsCacheInvalidAll(stmts[i + 1]) && IsLeafOp(stmts[i + 2], "system.fence");
        if (!already) {
          out.push_back(MakeCacheInvalidAll(child->span_));
          out.push_back(MakeNoArgOp("system.fence", child->span_));
          changed = true;
        }
      }
      // Consume side: a whole-GM cacheinvalid after each wait.
      if (eff == Effect::kWait && !(i + 1 < stmts.size() && IsCacheInvalidAll(stmts[i + 1]))) {
        out.push_back(MakeCacheInvalidAll(child->span_));
        changed = true;
      }
    }
    if (!changed) return op;
    return SeqStmts::Flatten(std::move(out), op->span_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto new_then = MarkBody(op->then_body_);
    std::optional<StmtPtr> new_else = op->else_body_;
    if (op->else_body_.has_value()) new_else = MarkBody(op->else_body_.value());
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
  // an enclosing SeqStmts, e.g. `if p != me: remote_store(...)`). A SeqStmts body
  // is handled by its own visitor; a bare leaf gets its markers wrapped here.
  // After the first run a wrapped body is a SeqStmts, so the pass stays idempotent.
  StmtPtr MarkBody(const StmtPtr& body) {
    if (As<SeqStmts>(body)) return VisitStmt(body);
    const Effect eff = StmtEffect(body);
    auto visited = VisitStmt(body);
    std::vector<StmtPtr> out{visited};
    if (WriteTargetToInvalidate(body) || IsRemoteWrite(LeafCall(body))) {
      // Fence only: local region cacheinvalid is suspended, the peer-region one is
      // emitted by codegen. See the file header.
      out.push_back(MakeNoArgOp("system.fence", body->span_));
    } else if (eff == Effect::kWrite) {
      // Opaque write (Submit / unregistered op): conservative whole-GM ci + fence.
      out.push_back(MakeCacheInvalidAll(body->span_));
      out.push_back(MakeNoArgOp("system.fence", body->span_));
    }
    if (eff == Effect::kWait) out.push_back(MakeCacheInvalidAll(body->span_));
    if (out.size() == 1) return visited;
    return SeqStmts::Flatten(std::move(out), body->span_);
  }

  template <typename LoopPtr>
  StmtPtr VisitLoop(const LoopPtr& op, const StmtPtr& body) {
    auto new_body = MarkBody(body);
    if (new_body.get() == body.get()) return op;
    auto result = MutableCopy(op);
    result->body_ = std::move(new_body);
    return result;
  }
};

}  // namespace

Pass InsertCommFence() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_) return func;
    // The data-before-signal contract is an InCore-only concern: the publishing
    // writes, waits, and the system.cacheinvalid / system.fence markers are
    // InCore GM builtins. Orchestration / HOST functions only dispatch tasks —
    // their cross-function calls are not GM publishing writes, and inserting an
    // InCore builtin there is rejected by orchestration codegen.
    if (!IsInCoreType(func->func_type_)) return func;
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

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

// MaterializeAllocTiles (issue #1956).
//
// Promotes the PTO tile *handle* from a codegen-synthesized artifact to an
// explicit IR op so PTO codegen becomes a strict 1:1 emitter. For each distinct
// tile buffer used by a function (grouped by MemRef identity — base + byte_offset
// + size, the same key PTO codegen uses to decide same-buffer aliasing), this
// pass emits exactly one `alloc_tile(base, byte_offset, shape)` op at the
// function head — a scope that dominates every use of the buffer. Codegen then
// emits `pto.alloc_tile` 1:1 from these ops instead of synthesizing a handle at
// each tile var's (possibly in-branch) definition site, which is what let an
// if/else-yield phi handle be declared inside a branch and used outside it — an
// undeclared-SSA scoping violation under memory_planner=PTOAS.
//
// The physical address, when known (memory_planner=PYPTO, after
// AllocateMemoryAddr), rides on the MemRef's byte_offset that the emitted op
// carries; codegen surfaces it as the `addr` operand exactly as before. Under
// PTOAS no address is baked and codegen omits the operand.
//
// Must-alias unification (loop-carry, if/else-yield phi, in-place) already ran in
// MaterializeSemanticAliases, so members of a must-alias group share one MemRef
// identity here and therefore collapse to a single alloc_tile op.

#include <any>
#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/codegen/pto/tile_buf_signature.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

// Handle-dedup key, kept in lock-step with PTO codegen's BufferHandleKey
// (pto_codegen.cpp). The MemRef identity (base + byte_offset + size) names the
// physical byte-slot. Under memory_planner=PyPTO the slot alone is too coarse:
// two tile vars can share one byte-slot yet need distinct pto.alloc_tile handles
// because they differ in pad / physical shape / layout / valid-shape (e.g. a
// `tile.fillpad` result at pad=min over the load's pad=0 tile, or a row-major
// [1,N] reshape-tmp aliasing an [N,1] col-major phi value). PyPTO handles carry
// an explicit `addr`, so multiple typed handles at one address correctly alias
// the same bytes — matching the pre-#1956 per-var model. The TileBufSignature
// suffix keeps them apart; true aliases (loop-carry, if/else-yield phi, in-place
// results) share one signature and therefore one handle. Under PTOAS the handle
// has no addr and ptoas allocates one buffer per handle, so the slot must map to
// exactly one handle — memref identity only (pad is collapsed, a pre-existing
// PTOAS trait).
std::string BufferKey(const MemRefPtr& memref, const std::shared_ptr<const TileType>& tile_type,
                      bool split_by_signature) {
  // The byte-slot key is the shared MemRefIdentityKey — codegen's BufferHandleKey
  // and the AllocTileDominatesUses verifier must resolve/check against the exact
  // same string, so all three derive it from one helper (see memref_utils.h).
  std::string key = MemRefIdentityKey(memref);
  if (split_by_signature) {
    key += "|sig=" + codegen::TileBufSignature::FromTileType(*tile_type).Key();
  }
  return key;
}

struct BufferInfo {
  MemRefPtr memref;
  std::shared_ptr<const TileType> tile_type;
  std::string name_hint;  // representative (first-seen) member's name, for readable handle names
};

// Collect one representative (MemRef, TileType, name) per distinct tile buffer,
// in first-seen program order. Covers every buffer codegen must emit a handle
// for: AssignStmt-defined tiles, loop iter_args, and control-flow return_vars.
// Function params are excluded — codegen binds them to %argN, not to a handle.
// MemRef-less tiles (e.g. a cross-core tpop result living in the reserved C2V
// slot, and views over it) are skipped: they borrow a reserved slot and get no
// alloc_tile handle.
class BufferCollector : public IRVisitor {
 public:
  std::vector<std::string> order;
  std::map<std::string, BufferInfo> buffers;

  explicit BufferCollector(bool split_by_signature) : split_by_signature_(split_by_signature) {}

  void Record(const VarPtr& var) {
    auto tile_type = GetTileTypeWithMemRef(var->GetType());
    if (!tile_type) return;
    auto memref = GetDefinedMemRef(tile_type);
    if (!memref) return;
    auto key = BufferKey(memref, tile_type, split_by_signature_);
    if (buffers.emplace(key, BufferInfo{memref, tile_type, var->name_hint_}).second) {
      order.push_back(key);
    }
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    Record(op->var_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    for (const auto& ia : op->iter_args_) Record(std::static_pointer_cast<const Var>(ia));
    for (const auto& rv : op->return_vars_) Record(rv);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& ia : op->iter_args_) Record(std::static_pointer_cast<const Var>(ia));
    for (const auto& rv : op->return_vars_) Record(rv);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& rv : op->return_vars_) Record(rv);
    IRVisitor::VisitStmt_(op);
  }

 private:
  bool split_by_signature_;
};

// h = alloc_tile(base, byte_offset, shape) : tile_type
StmtPtr MakeAllocTileStmt(const BufferInfo& info, int idx) {
  const auto& memref = info.memref;
  const auto& tile_type = info.tile_type;

  std::vector<ExprPtr> shape_elems(tile_type->shape_.begin(), tile_type->shape_.end());
  auto shape_tuple = std::make_shared<MakeTuple>(std::move(shape_elems), Span::unknown());

  auto memory_space = tile_type->GetMemorySpace();
  INTERNAL_CHECK(memory_space.has_value())
      << "Internal error: tile buffer with a MemRef must have a resolved memory_space";

  std::vector<std::pair<std::string, std::any>> kwargs;
  kwargs.emplace_back("dtype", std::any(tile_type->dtype_));
  kwargs.emplace_back("memory_space", std::any(*memory_space));

  std::vector<ExprPtr> args = {memref->base_, memref->byte_offset_, shape_tuple};
  auto op = OpRegistry::GetInstance().GetOp("alloc_tile");
  // Reuse the buffer's exact TileType (it already carries this MemRef) as the
  // handle's type, so the handle shares the group's MemRef identity and codegen
  // resolves every member to it.
  auto call = std::make_shared<Call>(op, std::move(args), std::move(kwargs),
                                     std::static_pointer_cast<const Type>(tile_type), Span::unknown());
  // Name the handle after its representative member (+ "__buf") so the emitted
  // pto.alloc_tile keeps a readable, buffer-descriptive SSA name and stays
  // unique against the members it backs. `idx` guarantees uniqueness when two
  // buffers share a member name_hint (or it is empty).
  std::string name = (info.name_hint.empty() ? "alloc_tile" : info.name_hint) + "__buf" + std::to_string(idx);
  auto handle = std::make_shared<Var>(name, tile_type, Span::unknown());
  return std::make_shared<AssignStmt>(handle, call, Span::unknown());
}

// The buffer key of a var, or "" if it holds no MemRef-backed tile.
std::string BufferKeyOfVar(const VarPtr& var, bool split_by_signature) {
  auto tile_type = GetTileTypeWithMemRef(var->GetType());
  if (!tile_type) return "";
  auto memref = GetDefinedMemRef(tile_type);
  if (!memref) return "";
  return BufferKey(memref, tile_type, split_by_signature);
}

// Which of `keys` does `stmt`'s subtree use? Returned in `keys` order.
std::vector<std::string> KeysUsedBy(const StmtPtr& stmt, const std::vector<std::string>& keys,
                                    bool split_by_signature) {
  BufferCollector bc(split_by_signature);
  bc.VisitStmt(stmt);
  std::set<std::string> used(bc.order.begin(), bc.order.end());
  std::vector<std::string> out;
  for (const auto& k : keys)
    if (used.count(k)) out.push_back(k);
  return out;
}

std::vector<StmtPtr> BodyToVec(const StmtPtr& body) {
  if (auto seq = As<SeqStmts>(body)) return seq->stmts_;
  return {body};
}

// Forward decl: recursive placement of handles into a statement list.
std::vector<StmtPtr> PlaceHandlesInSeq(const std::vector<StmtPtr>& stmts,
                                       const std::vector<std::string>& keys,
                                       const std::map<std::string, BufferInfo>& buffers, int* idx,
                                       bool split_by_signature);

// A buffer used by exactly one child statement `stmt` may descend into a nested
// scope of that child — but only when the descent still dominates every use and
// keeps the handle after its TileView operand deps:
//   - ForStmt / WhileStmt: descend into the body iff the buffer is NOT a loop
//     carry (iter_arg / return_var), which would live across the loop boundary.
//   - IfStmt: descend into a branch iff the buffer is used in exactly one branch
//     and is not a return_var (a phi that must be declared before the if).
// Otherwise the handle is placed at the current level, before the child.
bool CanDescend(const StmtPtr& stmt, const std::string& key, bool split_by_signature) {
  auto is_carry = [&](const auto& iter_args, const std::vector<VarPtr>& return_vars) {
    for (const auto& ia : iter_args)
      if (BufferKeyOfVar(std::static_pointer_cast<const Var>(ia), split_by_signature) == key) return true;
    for (const auto& rv : return_vars)
      if (BufferKeyOfVar(rv, split_by_signature) == key) return true;
    return false;
  };
  if (auto f = As<ForStmt>(stmt)) return !is_carry(f->iter_args_, f->return_vars_);
  if (auto w = As<WhileStmt>(stmt)) return !is_carry(w->iter_args_, w->return_vars_);
  if (auto ic = As<IfStmt>(stmt)) {
    for (const auto& rv : ic->return_vars_)
      if (BufferKeyOfVar(rv, split_by_signature) == key) return false;
    bool in_then = ic->then_body_ && !KeysUsedBy(ic->then_body_, {key}, split_by_signature).empty();
    bool in_else = ic->else_body_.has_value() && *ic->else_body_ &&
                   !KeysUsedBy(*ic->else_body_, {key}, split_by_signature).empty();
    return in_then != in_else;  // exactly one branch
  }
  return false;
}

// Rebuild `stmt` with `keys` placed inside its (single) nested scope. Only called
// for stmts where CanDescend returned true for every key.
StmtPtr RewriteStmtWithPushdown(const StmtPtr& stmt, const std::vector<std::string>& keys,
                                const std::map<std::string, BufferInfo>& buffers, int* idx,
                                bool split_by_signature) {
  if (auto f = As<ForStmt>(stmt)) {
    auto new_body = SeqStmts::Flatten(
        PlaceHandlesInSeq(BodyToVec(f->body_), keys, buffers, idx, split_by_signature), f->body_->span_);
    return std::make_shared<ForStmt>(f->loop_var_, f->start_, f->stop_, f->step_, f->iter_args_, new_body,
                                     f->return_vars_, f->span_, f->kind_, f->attrs_, f->leading_comments_);
  }
  if (auto w = As<WhileStmt>(stmt)) {
    auto new_body = SeqStmts::Flatten(
        PlaceHandlesInSeq(BodyToVec(w->body_), keys, buffers, idx, split_by_signature), w->body_->span_);
    return std::make_shared<WhileStmt>(w->condition_, w->iter_args_, new_body, w->return_vars_, w->span_,
                                       w->leading_comments_);
  }
  auto ic = As<IfStmt>(stmt);
  INTERNAL_CHECK(ic) << "Internal error: RewriteStmtWithPushdown expects For/While/If";
  std::vector<std::string> then_keys, else_keys;
  for (const auto& k : keys) {
    if (ic->then_body_ && !KeysUsedBy(ic->then_body_, {k}, split_by_signature).empty()) {
      then_keys.push_back(k);
    } else {
      else_keys.push_back(k);
    }
  }
  StmtPtr new_then = ic->then_body_;
  if (!then_keys.empty() && ic->then_body_) {
    new_then = SeqStmts::Flatten(
        PlaceHandlesInSeq(BodyToVec(ic->then_body_), then_keys, buffers, idx, split_by_signature),
        ic->then_body_->span_);
  }
  std::optional<StmtPtr> new_else = ic->else_body_;
  if (!else_keys.empty() && ic->else_body_.has_value() && *ic->else_body_) {
    new_else = SeqStmts::Flatten(
        PlaceHandlesInSeq(BodyToVec(*ic->else_body_), else_keys, buffers, idx, split_by_signature),
        (*ic->else_body_)->span_);
  }
  return std::make_shared<IfStmt>(ic->condition_, new_then, new_else, ic->return_vars_, ic->span_,
                                  ic->leading_comments_);
}

// Deps-aware, scope-recursive placement: put each buffer's alloc_tile op at the
// smallest scope that dominates all its uses AND follows its TileView operand
// deps. A buffer used across several statements (or across if/else branches, or
// as a loop carry) is placed at this level before its first use; a buffer used
// entirely within one nested loop/branch descends into that scope (so a handle
// whose valid_shape references a loop-body scalar is not hoisted above it). O(N)
// per level; each stmt is rescanned once per enclosing level (nesting depth is
// bounded), so overall O(N * depth).
std::vector<StmtPtr> PlaceHandlesInSeq(const std::vector<StmtPtr>& stmts,
                                       const std::vector<std::string>& keys,
                                       const std::map<std::string, BufferInfo>& buffers, int* idx,
                                       bool split_by_signature) {
  std::map<std::string, std::vector<size_t>> users;
  for (size_t i = 0; i < stmts.size(); ++i)
    for (const auto& k : KeysUsedBy(stmts[i], keys, split_by_signature)) users[k].push_back(i);

  std::map<size_t, std::vector<std::string>> place_before, push_into;
  for (const auto& k : keys) {  // iterate in `keys` order for deterministic handle naming
    auto it = users.find(k);
    if (it == users.end() || it->second.empty()) continue;
    size_t first = it->second.front();
    if (it->second.size() == 1 && CanDescend(stmts[first], k, split_by_signature)) {
      push_into[first].push_back(k);
    } else {
      place_before[first].push_back(k);
    }
  }

  std::vector<StmtPtr> out;
  out.reserve(stmts.size() + keys.size());
  for (size_t i = 0; i < stmts.size(); ++i) {
    auto pb = place_before.find(i);
    if (pb != place_before.end())
      for (const auto& k : pb->second) out.push_back(MakeAllocTileStmt(buffers.at(k), (*idx)++));
    auto pi = push_into.find(i);
    out.push_back(pi != push_into.end()
                      ? RewriteStmtWithPushdown(stmts[i], pi->second, buffers, idx, split_by_signature)
                      : stmts[i]);
  }
  return out;
}

FunctionPtr TransformMaterializeAllocTiles(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "MaterializeAllocTiles cannot run on null function";
  // Orchestration functions submit tasks and never hold TileType variables.
  if (func->func_type_ == FunctionType::Orchestration) return func;
  if (!func->body_) return func;

  // Handle granularity depends on who plans memory (see BufferKey). PyPTO (the
  // default when no context is active) splits one byte-slot into per-signature
  // handles; PTOAS keeps one handle per byte-slot. This must stay consistent
  // with codegen's emit_tile_addr_ gate in PTOCodegen::BufferHandleKey.
  const auto* ctx = PassContext::Current();
  const bool split_by_signature = !ctx || ctx->GetMemoryPlanner() == MemoryPlanner::PyPTO;

  BufferCollector collector(split_by_signature);
  collector.VisitStmt(func->body_);
  if (collector.order.empty()) return func;

  int idx = 0;
  StmtPtr new_body = SeqStmts::Flatten(
      PlaceHandlesInSeq(BodyToVec(func->body_), collector.order, collector.buffers, &idx, split_by_signature),
      func->body_->span_);
  return std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                          func->return_types_, new_body, func->span_, func->func_type_,
                                          func->level_, func->role_, func->attrs_);
}

}  // namespace

Pass MaterializeAllocTiles() {
  return CreateFunctionPass(TransformMaterializeAllocTiles, "MaterializeAllocTiles",
                            kMaterializeAllocTilesProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

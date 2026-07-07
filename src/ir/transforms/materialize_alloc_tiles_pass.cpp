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

  auto seq = As<SeqStmts>(func->body_);
  if (!seq) {
    // Single-statement body: the buffer's sole use is that statement, so place
    // every handle before it (its base Ptr and any type deps are in scope).
    std::vector<StmtPtr> new_stmts;
    new_stmts.reserve(collector.order.size() + 1);
    int idx = 0;
    for (const auto& key : collector.order) {
      new_stmts.push_back(MakeAllocTileStmt(collector.buffers.at(key), idx++));
    }
    new_stmts.push_back(func->body_);
    StmtPtr new_body = SeqStmts::Flatten(std::move(new_stmts), func->body_->span_);
    return std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                            func->return_types_, new_body, func->span_, func->func_type_,
                                            func->level_, func->role_, func->attrs_);
  }

  // Deps-aware placement: insert each buffer's alloc_tile op immediately before
  // the first top-level statement whose subtree uses that buffer. That point
  // (a) dominates all uses — a buffer written across if/else branches is first
  // seen at the enclosing IfStmt, so the handle lands before it — and (b) follows
  // any body-defined value the handle's TileView references (e.g. a runtime valid
  // length), which a blind hoist to the function head would precede. O(N): each
  // top-level subtree is scanned once.
  std::map<std::string, size_t> first_use;
  for (size_t i = 0; i < seq->stmts_.size(); ++i) {
    BufferCollector stmt_bufs(split_by_signature);
    stmt_bufs.VisitStmt(seq->stmts_[i]);
    for (const auto& key : stmt_bufs.order) first_use.emplace(key, i);
  }

  std::map<size_t, std::vector<StmtPtr>> inserts;  // stmt index -> handles to prepend
  int idx = 0;
  for (const auto& key : collector.order) {
    auto stmt = MakeAllocTileStmt(collector.buffers.at(key), idx++);
    auto it = first_use.find(key);
    inserts[it != first_use.end() ? it->second : 0].push_back(std::move(stmt));
  }

  std::vector<StmtPtr> new_stmts;
  new_stmts.reserve(seq->stmts_.size() + collector.order.size());
  for (size_t i = 0; i < seq->stmts_.size(); ++i) {
    auto ins = inserts.find(i);
    if (ins != inserts.end()) {
      for (auto& s : ins->second) new_stmts.push_back(std::move(s));
    }
    new_stmts.push_back(seq->stmts_[i]);
  }

  StmtPtr new_body = SeqStmts::Flatten(std::move(new_stmts), func->body_->span_);
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

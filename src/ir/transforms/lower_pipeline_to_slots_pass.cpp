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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// ptoas' slot-count bounds for a `!pto.multi_tile_buf`, mirrored from
/// `include/pypto/codegen/pto/pto_type_utils.h` (`kMinMultiTileBufSlots` /
/// `kMaxMultiTileBufSlots`). Duplicated rather than included because `ir/` must
/// not depend on `codegen/`; a region outside these bounds is a hard codegen
/// error, so this pass must never synthesize one.
constexpr int64_t kMinSlots = 2;
constexpr int64_t kMaxSlots = 16;

/// Memory spaces ptoas accepts for a multi-buffer slot. Mirrors
/// `IsMultiBufferMemorySpace` in `src/codegen/pto/pto_codegen.cpp`.
bool IsMultiBufferSpace(const std::optional<MemorySpace>& space) {
  return space.has_value() &&
         (*space == MemorySpace::Vec || *space == MemorySpace::Mat || *space == MemorySpace::Acc);
}

/// Does the tile state a compile-time valid extent?
///
/// Same source of truth as `StaticValidExtents` in `src/codegen/pto/pto_codegen.cpp`:
/// the author's `valid_shape` when there is one, the physical shape otherwise, and
/// only the leading two dimensions matter. A region declares ONE static extent for
/// all its slots, so a runtime extent is a codegen blocker — only the *existence* of
/// the extent is needed here, never its value.
bool HasStaticValidExtents(const std::shared_ptr<const TileType>& tile_type) {
  const std::vector<ExprPtr>* dims = nullptr;
  if (const auto& tile_view = tile_type->tile_view_;
      tile_view.has_value() && !tile_view->valid_shape.empty()) {
    dims = &tile_view->valid_shape;
  } else if (!tile_type->shape_.empty()) {
    dims = &tile_type->shape_;
  }
  if (dims == nullptr || dims->empty()) return false;
  for (size_t i = 0; i < dims->size() && i < 2; ++i) {
    if (!As<ConstInt>((*dims)[i])) return false;
  }
  return true;
}

/// Every Var an expression reads.
class VarReadCollector : public IRVisitor {
 public:
  std::set<const Var*> vars;

  void VisitVarLike_(const VarPtr& op) override {
    if (op) vars.insert(op.get());
    IRVisitor::VisitVarLike_(op);
  }
};

/// The two ways a tile inside a loop body disqualifies itself from becoming a
/// multi-buffer slot, collected in one walk over that body.
///
/// Both mirror a `PlanMultiBufferRegions` blocker rather than a rule of this pass:
///   - `phi_carried`: a tile carried out of an `if` / loop makes the enclosing
///     statement's return var share its MemRef, which codegen rejects as "one of
///     its slots is carried out of an if or a loop as a phi".
///   - `inherit_consumed`: a view / in-place op's result physically IS its source's
///     buffer, so `InitMemRef` hands it the same MemRef — with a different tile_buf
///     type, which codegen rejects as "its slots hold differently shaped tiles".
class DisqualifyingUseCollector : public IRVisitor {
 public:
  std::set<const Var*> phi_carried;
  std::set<const Var*> inherit_consumed;

  void VisitStmt_(const YieldStmtPtr& op) override {
    for (const auto& value : op->value_) {
      RecordPhiCarry(value);
    }
    IRVisitor::VisitStmt_(op);
  }

  /// A nested loop's `init_values` reach the phi through `IterArg::initValue_`
  /// rather than through a `YieldStmt`, but bind the same way: the IterArg is a
  /// phi that shares its initializer's MemRef. Without this, a slotted tile used
  /// as an inner loop's init survives into the phi and codegen rejects the region.
  void VisitExpr_(const IterArgPtr& op) override {
    RecordPhiCarry(op->initValue_);
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    RecordCallLike(op->op_, op->args_);
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    RecordCallLike(op->op_, op->args_);
    IRVisitor::VisitExpr_(op);
  }

  /// SSA guarantees a definition precedes its uses, so recording the alias edge
  /// here — in the same pre-order walk — always beats the yield that reads it.
  void VisitStmt_(const AssignStmtPtr& op) override {
    RecordAlias(op);
    IRVisitor::VisitStmt_(op);
  }

 private:
  /// `InitMemRef` shares one MemRef along two alias edges, and neither is visible
  /// as a use of the tile that owns the buffer:
  ///   - a bare `a = b` tile copy (`init_memref.cpp`'s "Tile alias" case);
  ///   - a view / in-place result, which physically *is* its source's buffer.
  /// Without the edge, `alias = t; yield alias` records only `alias`, the candidate
  /// `t` looks unyielded, and its slot still reaches the phi that codegen rejects.
  /// Resolving every recorded use to its alias root attributes the carry to `t`.
  void RecordAlias(const AssignStmtPtr& op) {
    if (!op->var_ || !op->value_) return;
    if (!As<TileType>(op->var_->GetType())) return;
    if (auto src = AsVarLike(op->value_)) {
      alias_root_[op->var_.get()] = Root(src.get());
      return;
    }
    auto call = As<Call>(op->value_);
    if (!call || !call->op_ || call->args_.empty()) return;
    if (!op_predicates::OutputInheritsSourceBuffer(call->op_->name_)) return;
    if (auto src = AsVarLike(call->args_[0])) alias_root_[op->var_.get()] = Root(src.get());
  }

  /// The tile that owns the buffer `var` resolves to. SSA rules out a cycle; the
  /// self-edge guard keeps a malformed chain from spinning anyway.
  [[nodiscard]] const Var* Root(const Var* var) const {
    for (auto it = alias_root_.find(var); it != alias_root_.end(); it = alias_root_.find(var)) {
      if (it->second == var) break;
      var = it->second;
    }
    return var;
  }

  void RecordPhiCarry(const ExprPtr& value) {
    if (!value) return;
    VarReadCollector collector;
    collector.VisitExpr(value);
    for (const Var* var : collector.vars) phi_carried.insert(Root(var));
  }

  void RecordCallLike(const OpPtr& op_node, const std::vector<ExprPtr>& args) {
    if (!op_node || !op_predicates::OutputInheritsSourceBuffer(op_node->name_)) return;
    for (const auto& arg : args) {
      VarReadCollector collector;
      collector.VisitExpr(arg);
      for (const Var* var : collector.vars) inherit_consumed.insert(Root(var));
    }
  }

  /// Alias target -> the tile whose buffer it shares.
  std::map<const Var*, const Var*> alias_root_;
};

/// Every allocation the author already declared, keyed by its base Ptr.
///
/// A declared allocation is `is_pinned_`, so ptoas may not reuse a byte of it — the
/// same property that makes this pass' own regions count against the budget. Ignoring
/// them lets the gate admit a synthesized region that fits *on its own* while the pair
/// overflows, which ptoas answers with a hard error instead of the degradable
/// replication path. Grouping by base Ptr matches `InitMemRef`, which sizes one
/// allocation from the largest tile bound to any of its slots.
class PinnedAllocCollector : public IRVisitor {
 public:
  explicit PinnedAllocCollector(const backend::BackendHandler* handler) : handler_(handler) {}

  /// Pinned bytes already spoken for, per memory space.
  [[nodiscard]] std::map<MemorySpace, uint64_t> BytesPerSpace() const {
    std::map<MemorySpace, uint64_t> bytes;
    for (const auto& [base, alloc] : allocs_) {
      (void)base;
      if (alloc.slot_size == 0 || alloc.slot_count == 0) continue;
      if (alloc.slot_size > std::numeric_limits<uint64_t>::max() / alloc.slot_count) continue;
      uint64_t& total = bytes[alloc.space];
      const uint64_t region = alloc.slot_size * alloc.slot_count;
      if (total > std::numeric_limits<uint64_t>::max() - region) continue;
      total += region;
    }
    return bytes;
  }

  void VisitVarLike_(const VarPtr& op) override {
    Record(op);
    IRVisitor::VisitVarLike_(op);
  }

 private:
  struct Alloc {
    uint64_t slot_size = 0;
    uint64_t slot_count = 0;
    MemorySpace space = MemorySpace::DDR;
  };

  void Record(const VarPtr& var) {
    if (!var) return;
    auto tile_type = As<TileType>(var->GetType());
    if (!tile_type || !tile_type->memref_.has_value()) return;
    const auto& memref = *tile_type->memref_;
    if (!memref || !memref->base_ || !memref->is_pinned_) return;
    const auto& space_opt = tile_type->memory_space_;
    if (!space_opt.has_value()) return;
    auto size = utils::StaticPhysicalAllocationBytes(tile_type, *space_opt, handler_);
    if (!size.has_value()) return;

    auto& alloc = allocs_[memref->base_.get()];
    alloc.space = *space_opt;
    // The slots are uniform, so one slot must hold the largest tile bound to any.
    alloc.slot_size = std::max(alloc.slot_size, *size);
    alloc.slot_count = std::max(alloc.slot_count, static_cast<uint64_t>(memref->slot_count_));
  }

  const backend::BackendHandler* handler_;
  std::map<const Var*, Alloc> allocs_;
};

/// One tile that will be rebound onto a freshly declared N-slot allocation.
struct Candidate {
  VarPtr var;
  std::shared_ptr<const TileType> tile_type;
};

/// Rewrites eligible `pl.pipeline` loops to rotate through the slots of one
/// declared allocation instead of being replicated by `LowerPipelineLoops`.
class SlotBindingMutator : public IRMutator {
 public:
  bool changed = false;

  /// Charge the budget for allocations the author pinned before this pass ran.
  void SeedCommittedBytes(std::map<MemorySpace, uint64_t> bytes) { committed_slot_bytes_ = std::move(bytes); }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (op->kind_ != ForKind::Pipeline || !op->HasAttr(kPipelineStagesAttr)) {
      return IRMutator::VisitStmt_(op);
    }
    const auto factor = static_cast<int64_t>(op->GetAttr<int>(kPipelineStagesAttr, 0));
    INTERNAL_CHECK_SPAN(factor >= 1, op->span_)
        << "Internal error: pipeline_stages must be >= 1, got " << factor;

    // `factor == 1` is a user-written `pl.pipeline(stage=1)` or the marker a
    // previous `LowerPipelineLoops` run left behind. Nothing to multi-buffer:
    // leave the (kind, attr) pair whole for `CanonicalizeIOOrder` to scope on.
    std::vector<Candidate> candidates;
    if (factor > 1 && !unslotted_pipeline_ancestor_ && LoopShapeAllowsSlots(op, factor)) {
      // `nullopt` means a load this pass would have slotted hit a codegen blocker,
      // so the whole loop goes back to `LowerPipelineLoops`.
      if (auto collected = CollectCandidates(op, factor)) candidates = std::move(*collected);
    }

    if (candidates.empty()) {
      // Left intact for `LowerPipelineLoops` to replicate. Everything nested
      // below is then replicated with it, so no descendant may take a slot —
      // its `factor` clones would each select one slot of the same allocation
      // inside one loop body, which codegen rejects as co-live.
      return VisitBodyWithAncestorFlag(op, factor > 1);
    }

    for (const auto& candidate : candidates) BindSlot(op, candidate, factor);
    changed = true;

    auto visited = VisitBodyWithAncestorFlag(op, false);
    auto for_stmt = As<ForStmt>(visited);
    INTERNAL_CHECK_SPAN(for_stmt, op->span_) << "Internal error: pipeline loop did not survive as a ForStmt";

    // The slots carry the ping-pong now, so the loop is an ordinary sequential
    // one. Kind and attr are dropped together — `PipelineLoopValid` verifies the
    // bidirectional invariant `kind == Pipeline <=> pipeline_stages attr present`.
    auto cleaned = MutableCopy(for_stmt);
    cleaned->kind_ = ForKind::Sequential;
    cleaned->attrs_ = StripAttr(for_stmt->attrs_, kPipelineStagesAttr);
    return cleaned;
  }

  ExprPtr VisitExpr_(const VarPtr& op) override { return Substitute(op); }

  /// An IterArg not being rebound still has to be *visited*, not returned as-is:
  /// `IterArg::initValue_` is a child expression that may read a rebound tile, and
  /// only `IRMutator`'s implementation rewrites it. Returning `op` unchanged on a
  /// miss leaves a nested loop's `init_values` pointing at the pre-substitution
  /// Var, which fails SSA verification the moment this pass rebinds anything an
  /// inner loop carries.
  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto it = var_map_.find(std::static_pointer_cast<const Var>(op));
    if (it != var_map_.end()) return std::static_pointer_cast<const Expr>(it->second);
    return IRMutator::VisitExpr_(op);
  }

 private:
  ExprPtr Substitute(const VarPtr& op) {
    auto it = var_map_.find(op);
    if (it == var_map_.end()) return op;
    return std::static_pointer_cast<const Expr>(it->second);
  }

  StmtPtr VisitBodyWithAncestorFlag(const ForStmtPtr& op, bool ancestor) {
    const bool saved = unslotted_pipeline_ancestor_;
    unslotted_pipeline_ancestor_ = ancestor || saved;
    auto result = IRMutator::VisitStmt_(op);
    unslotted_pipeline_ancestor_ = saved;
    return result;
  }

  /// Can the loop's induction variable index the slots directly as `iv % factor`?
  ///
  /// ptoas matches the *affine form* of the slot index to decide which accesses
  /// share a slot, and that match is what earns the rotation its per-slot dynamic
  /// event ids. A general `((iv - start) / step) % factor` would have to be
  /// materialized as an intermediate SSA value, which risks losing exactly the
  /// analysis this transform exists to trigger — so any loop whose slot index is
  /// not literally `iv % factor` is left to `LowerPipelineLoops`.
  static bool LoopShapeAllowsSlots(const ForStmtPtr& op, int64_t factor) {
    if (factor < kMinSlots || factor > kMaxSlots) return false;
    auto step = As<ConstInt>(op->step_);
    if (!step || step->value_ != 1) return false;
    auto start = As<ConstInt>(op->start_);
    if (!start || start->value_ % factor != 0) return false;
    return true;
  }

  /// The loop body's top-level loads that each want a private per-stage buffer,
  /// or `nullopt` when one of them cannot become a slot.
  ///
  /// Loads only: `MemoryReuse` already treats a load buffer as the thing that must
  /// stay private for ping-pong (iteration i+1's prefetch overlaps iteration i's
  /// compute) while letting compute intermediates coalesce, and giving every tile
  /// its own `factor` copies overflows the on-chip budget on real kernels.
  ///
  /// **Every** unbound top-level load is a candidate — there is no loop-invariance
  /// filter. Deciding invariance from "does an argument read the loop variable"
  /// is unsound: a load addressed through a loop-carried `IterArg` reads different
  /// data each iteration without ever naming the induction variable, and skipping
  /// it would strand it with neither a slot nor a replicated copy. Slotting an
  /// actually-invariant load costs nothing over the fallback either, since
  /// `LowerPipelineLoops` replicates its buffer `factor` times all the same.
  ///
  /// The decline is all-or-nothing per loop. A load that *wants* a per-stage buffer
  /// but trips a codegen blocker cannot simply be dropped from the candidate list:
  /// any surviving candidate would still demote the loop to `Sequential`, and the
  /// blocked load would then reach neither this pass' slots nor `LowerPipelineLoops`'
  /// replication, losing its per-stage privacy outright. Sending the whole loop down
  /// the replication path keeps that fallback intact for every load in the body.
  std::optional<std::vector<Candidate>> CollectCandidates(const ForStmtPtr& op, int64_t factor) {
    std::vector<Candidate> candidates;
    auto seq = As<SeqStmts>(op->body_);
    if (!seq) return candidates;

    DisqualifyingUseCollector uses;
    uses.VisitStmt(op->body_);

    for (const auto& stmt : seq->stmts_) {
      auto assign = As<AssignStmt>(stmt);
      if (!assign || !assign->var_) continue;
      auto call = As<Call>(assign->value_);
      if (!call || !call->op_) continue;
      // `tile.read` is deliberately absent: it returns a ScalarType element, not a
      // tile, so it allocates no buffer to rotate.
      if (!IsOp(call, "tile.load")) continue;

      auto tile_type = As<TileType>(assign->var_->GetType());
      if (!tile_type) continue;
      // An allocation the author declared stays the author's — and declining the
      // loop over it would push that declaration onto the replication path, which
      // rejects an author-declared slot inside a `pl.pipeline(stage=F)` body.
      if (tile_type->memref_.has_value()) continue;

      // Past here the load wants a slot, so every remaining gate is a blocker.
      const Var* raw = assign->var_.get();
      if (!IsMultiBufferSpace(tile_type->memory_space_) || !HasStaticValidExtents(tile_type) ||
          uses.phi_carried.count(raw) != 0 || uses.inherit_consumed.count(raw) != 0) {
        LOG_DEBUG << "LowerPipelineToSlots: declining the loop; '" << assign->var_->name_hint_
                  << "' cannot become a slot, so the body must stay replicable";
        return std::nullopt;
      }

      candidates.push_back({assign->var_, tile_type});
    }
    if (!candidates.empty() && !SlotsFitOnChip(candidates, factor)) return std::nullopt;
    return candidates;
  }

  /// Would binding `candidates` push a memory space past its on-chip capacity?
  ///
  /// The slots this pass declares are **pinned**: `InitMemRef` sizes the allocation
  /// at `factor * slot_size` and ptoas may not reuse any of it, so the pass is
  /// directly accountable for those bytes. Without this gate a loop with many
  /// eligible loads silently multiplies its footprint by `factor` and ptoas fails
  /// the whole compile with a hard `mat overflow` rather than degrading — unlike the
  /// replication path, where `MemoryReuse`'s capacity gate lowers the effective
  /// double-buffering depth until it fits. Declining sends the loop back to that
  /// gracefully-degrading path.
  ///
  /// The budget is per space and accumulated across the whole function
  /// (`committed_slot_bytes_`), because a slotted inner loop's region is co-live
  /// with its slotted ancestor's.
  ///
  /// Scope: this bounds only what *this pass* pins. Tiles it does not slot are still
  /// planned by ptoas with lifetime reuse, which the pass cannot model, so a space
  /// can still overflow on their account — that is ptoas' budget to enforce, not a
  /// footprint this pass introduced.
  bool SlotsFitOnChip(const std::vector<Candidate>& candidates, int64_t factor) {
    const auto* handler = PassContext::Current()->GetBackendHandler();
    std::map<MemorySpace, uint64_t> added;
    for (const auto& candidate : candidates) {
      const auto& space_opt = candidate.tile_type->memory_space_;
      // `IsMultiBufferSpace` already required a resolved space for every candidate;
      // re-checking keeps the access sound on its own terms rather than on that
      // caller-side invariant.
      if (!space_opt.has_value()) return false;
      const MemorySpace space = *space_opt;
      auto slot_bytes = utils::StaticPhysicalAllocationBytes(candidate.tile_type, space, handler);
      if (!slot_bytes.has_value()) {
        // No compile-time size means no way to prove the region fits.
        LOG_DEBUG << "LowerPipelineToSlots: declining the loop; '" << candidate.var->name_hint_
                  << "' has no static allocation size to budget";
        return false;
      }
      const auto factor_u = static_cast<uint64_t>(factor);
      if (*slot_bytes != 0 && factor_u > std::numeric_limits<uint64_t>::max() / *slot_bytes) return false;
      uint64_t& total = added[space];
      const uint64_t region = *slot_bytes * factor_u;
      if (total > std::numeric_limits<uint64_t>::max() - region) return false;
      total += region;
    }

    // An unconfigured backend has no capacity to check against. Mirrors MemoryReuse,
    // which leaves a space whose capacity is unknown ungated rather than guessing.
    const backend::Backend* be = backend::BackendConfig::IsConfigured() ? backend::GetBackend() : nullptr;
    if (be == nullptr) return true;

    for (const auto& [space, bytes] : added) {
      const uint64_t capacity = be->GetMemSize(space);
      if (capacity == 0) continue;  // unknown capacity for this space -> ungated
      const uint64_t committed = committed_slot_bytes_[space];
      if (committed >= capacity || bytes > capacity - committed) {
        LOG_DEBUG << "LowerPipelineToSlots: declining the loop; " << bytes << " bytes of " << factor
                  << "-slot regions plus " << committed << " already committed exceed the " << capacity
                  << "-byte capacity of memory space " << static_cast<int>(space);
        return false;
      }
    }
    for (const auto& [space, bytes] : added) committed_slot_bytes_[space] += bytes;
    return true;
  }

  /// Rebind one tile onto slot `iv % factor` of a freshly declared allocation.
  ///
  /// The MemRef is built exactly as the parser builds `pl.MemRef(name, slots=N)[k]`
  /// — an interned base Ptr, `is_pinned_` set, zero offset and size — so `InitMemRef`
  /// resolves it through the same path as an author's declaration: one slot sized to
  /// the tile, the allocation sized to `factor` slots, the index folded into the byte
  /// offset with the slot geometry kept for codegen.
  void BindSlot(const ForStmtPtr& op, const Candidate& candidate, int64_t factor) {
    const Span& span = candidate.var->span_;
    // One base Ptr per region. `InitMemRef` groups declarations by `base_` pointer
    // identity, so a fresh Var per candidate keeps the regions apart without a
    // name-uniqueness scheme.
    auto base = std::make_shared<Var>("pipe_" + candidate.var->name_hint_, GetPtrType(), span);
    ExprPtr slot_index =
        MakeFloorMod(op->loop_var_, std::make_shared<ConstInt>(factor, DataType::INDEX, span), span);
    auto memref = std::make_shared<MemRef>(std::static_pointer_cast<const Var>(base), int64_t{0}, uint64_t{0},
                                           span, /*is_pinned=*/true, static_cast<uint64_t>(factor),
                                           std::make_optional(slot_index));

    TypePtr new_type = CloneTypeWithMemRef(candidate.var->GetType(), std::optional<MemRefPtr>(memref));
    var_map_[candidate.var] = std::make_shared<Var>(candidate.var->name_hint_, new_type, span);

    LOG_DEBUG << "LowerPipelineToSlots: '" << candidate.var->name_hint_ << "' -> slot ("
              << op->loop_var_->name_hint_ << " % " << factor << ") of a " << factor << "-slot allocation";
  }

  /// Old tile Var -> the same tile bound to a slot. Registered before the body is
  /// visited, so the definition and every use are rewritten in one walk (the IR is
  /// in SSA form, so no use precedes its definition).
  std::map<VarPtr, VarPtr> var_map_;

  /// Pinned slot bytes this pass has already committed, per memory space. A slotted
  /// inner loop's region is co-live with its slotted ancestor's, so the capacity gate
  /// budgets against the running function-wide total rather than one loop at a time.
  std::map<MemorySpace, uint64_t> committed_slot_bytes_;

  /// Set while visiting a subtree under a pipeline loop this pass declined — that
  /// loop will be replicated, and a slot taken below it would be replicated with it.
  bool unslotted_pipeline_ancestor_ = false;
};

FunctionPtr TransformLowerPipelineToSlots(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "LowerPipelineToSlots cannot run on null function";
  if (!func->body_) return func;

  // Only ptoas' memory planner emits a multi-buffer region today: PTO codegen's
  // `PlanMultiBufferRegions` bails under the PyPTO planner, so a rotation
  // synthesized here would resolve to a runtime address on an ordinary
  // `alloc_tile` and buy nothing. Every loop then stays untouched and
  // `LowerPipelineLoops` replicates it, keeping the default pipeline
  // byte-identical.
  //
  // This is a property of the current codegen gate, NOT of ptoas: given
  // `alloc_multi_tile addr = <constant base>`, ptoas 0.55 derives the same
  // per-slot dynamic-event synchronization at `--pto-level=level3` as it does at
  // level2 (measured: identical sync-op sequence). Widening the gate needs the
  // PyPTO address allocator to reserve `slot_count * slot_size` for the region
  // base and codegen to emit that address, so it is left to a follow-up.
  auto* ctx = PassContext::Current();
  if (ctx == nullptr || ctx->GetMemoryPlanner() != MemoryPlanner::PtoAS) return func;

  SlotBindingMutator mutator;
  PinnedAllocCollector pinned(ctx->GetBackendHandler());
  pinned.VisitStmt(func->body_);
  mutator.SeedCommittedBytes(pinned.BytesPerSpace());
  auto new_body = mutator.VisitStmt(func->body_);
  if (!mutator.changed) return func;
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass LowerPipelineToSlots() {
  return CreateFunctionPass(TransformLowerPipelineToSlots, "LowerPipelineToSlots",
                            kLowerPipelineToSlotsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

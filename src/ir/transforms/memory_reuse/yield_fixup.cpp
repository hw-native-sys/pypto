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
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/memref_collectors.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/memory_reuse/internal.h"
#include "src/ir/transforms/memory_reuse/storage.h"

namespace pypto {
namespace ir {
namespace memory_reuse {

namespace {

/**
 * @brief Align loop-carried MemRefs to their initValue, top-down (fixes #1352).
 *
 * Greedy reuse (ApplyMemRefSharing) retypes an accumulator's *producer* and
 * *init* AssignStmt vars onto a reused buffer, but the loop-carried iter_arg /
 * return_var nodes never enter the lifetime/reuse maps (AnalyzeAllocationLifetimes
 * deliberately excludes them — see RegisterReturnVars), so they keep their
 * original buffer.  For *nested* pipelined accumulators this splits the chain:
 * an inner K-loop's initValue is the *outer* loop's iter_arg, and
 * YieldFixupMutator runs bottom-up — so it patches the inner carry against the
 * still-stale outer iter_arg and then inserts acc->acc tile.move ops to
 * reconcile the two buffers.  Ascend 910B has no hardware path between disjoint
 * accumulator addresses, so ptoas rejects those moves.
 *
 * This mutator restores the invariant codegen already assumes — "iter_arg /
 * return_var share initValue's buffer" — *before* YieldFixupMutator, and does
 * it top-down: it retypes each ForStmt's iter_args/return_vars to their
 * initValue's MemRef and seeds var_remap_ before recursing, so a nested loop
 * observes the corrected outer iter_arg as its init.  Once producers and
 * carries agree on one buffer, YieldFixupMutator inserts no spurious move.
 *
 * It only ever aligns a carry to its own init (a no-op when reuse left them
 * consistent), so a loop that genuinely yields a different buffer than its init
 * is untouched here and still gets its legitimate move from YieldFixupMutator.
 */
class AlignLoopCarriesToInitMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (op->iter_args_.empty()) return IRMutator::VisitStmt_(op);

    std::vector<IterArgPtr> new_iter_args = op->iter_args_;
    std::vector<VarPtr> new_return_vars = op->return_vars_;
    bool changed = false;

    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      // Resolve initValue through var_remap_ so an enclosing loop's alignment
      // (which rebuilt the init var this iter_arg carries) is observed here.
      auto init_expr = VisitExpr(op->iter_args_[i]->initValue_);
      bool init_changed = init_expr.get() != op->iter_args_[i]->initValue_.get();

      // MemRef alignment only applies when the init resolves to a tile-typed
      // var/iter_arg with a defined MemRef.  A non-tile or non-var carry (e.g.
      // a scalar, or a non-Var init expression) has no MemRef to align to, but
      // any remapped init_expr must still be propagated to the rebuilt IterArg
      // below — otherwise an enclosing loop's alignment is silently dropped.
      auto init_var = AsVarLike(init_expr);
      auto init_tile = init_var ? GetTileTypeWithMemRef(init_var->GetType()) : nullptr;
      MemRefPtr init_memref = nullptr;
      std::optional<MemorySpace> init_memory = std::nullopt;
      if (init_tile) {
        init_memref = GetDefinedMemRef(init_tile);
        init_memory = init_tile->GetMemorySpace();
      }

      // Re-type a carry's TileType onto init's MemRef, remapping any embedded
      // exprs (shape/strides) through var_remap_.  Only invoked when init_tile
      // is non-null, so init_memref is always defined at the call sites.
      auto retype_to_init = [&](const TileTypePtr& tile) {
        return CloneTypeWithMemRefAndRemapExprs(
            tile, init_memref, [this](const ExprPtr& e) { return VisitExpr(e); }, init_memory);
      };

      // Align the iter_arg's TileType to init's MemRef when both have a defined
      // MemRef, and always rebuild when the carried init reference changed so a
      // remapped init_expr is preserved regardless of the carry's type.
      auto ia_tile = As<TileType>(op->iter_args_[i]->GetType());
      // Addresses, not allocations: an iter_arg left on a sibling slot of one
      // multi-slot allocation names different bytes and still has to be moved to
      // init's. Anything not provably init's address is realigned -- this only
      // retypes a node, so realigning an unprovable pair costs nothing while
      // leaving it keeps a possibly-wrong slot in the metadata.
      bool ia_memref_differs =
          init_tile && ia_tile && ia_tile->memref_.has_value() &&
          CompareBaseAddress(GetDefinedMemRef(ia_tile), init_memref) != AddressRelation::kSame;
      if (ia_memref_differs || init_changed) {
        TypePtr new_ia_type = ia_memref_differs ? retype_to_init(ia_tile) : op->iter_args_[i]->GetType();
        new_iter_args[i] = std::make_shared<IterArg>(op->iter_args_[i]->name_hint_, new_ia_type, init_expr,
                                                     op->iter_args_[i]->span_);
        var_remap_[op->iter_args_[i].get()] = new_iter_args[i];
        changed = true;
      }

      // Align the matching return_var's TileType to init's MemRef (only when
      // init has a tile MemRef to align to).
      if (init_tile && i < new_return_vars.size()) {
        auto rv_tile = As<TileType>(op->return_vars_[i]->GetType());
        if (rv_tile && rv_tile->memref_.has_value() &&
            CompareBaseAddress(GetDefinedMemRef(rv_tile), init_memref) != AddressRelation::kSame) {
          new_return_vars[i] = std::make_shared<Var>(op->return_vars_[i]->name_hint_, retype_to_init(rv_tile),
                                                     op->return_vars_[i]->span_);
          var_remap_[op->return_vars_[i].get()] = new_return_vars[i];
          changed = true;
        }
      }
    }

    // Recurse into the body AFTER seeding var_remap_ so nested loops observe
    // the corrected outer iter_args as their init.
    auto new_body = VisitStmt(op->body_);

    // iter_arg references are confined to the loop body; drop them so a sibling
    // loop cannot pick up a stale remap.  return_var remaps must persist for
    // post-loop uses, so they are intentionally left in place.
    for (const auto& old_iter_arg : op->iter_args_) {
      var_remap_.erase(old_iter_arg.get());
    }

    if (!changed && new_body == op->body_) return op;
    auto new_for = MutableCopy(op);
    new_for->iter_args_ = std::move(new_iter_args);
    new_for->return_vars_ = std::move(new_return_vars);
    new_for->body_ = new_body;
    return new_for;
  }
};

/**
 * @brief Fix yield/return_var MemRef mismatch after MemoryReuse.
 *
 * Handles both ForStmt and IfStmt:
 * - ForStmt: MemoryReuse may assign different MemRefs to iter_arg and yield value.
 *   Since yield value becomes the next iteration's iter_arg, they must use the
 *   same buffer. When they differ, insert a tile.move before yield, except that
 *   divergent Acc buffers are rejected because Acc->Acc moves are unsupported.
 * - IfStmt: MemoryReuse may change MemRefs of variables inside branches.
 *   Patch return_vars to match the yield value's MemRef, using the same Acc guard.
 */
class YieldFixupMutator : public StorageBoundaryMutator {
 public:
  YieldFixupMutator() = default;

  StmtPtr Run(const StmtPtr& body, const std::vector<VarPtr>& parameters) {
    if (canonical_branch_storage_) names_.Collect(body, parameters);
    if (selections_) {
      for (const auto& [old_var, replacement] : selections_->tiles) {
        var_remap_[old_var] = replacement;
      }
    }
    return VisitStmt(body);
  }

  explicit YieldFixupMutator(StorageSelectionPlan selections) : selections_(std::move(selections)) {}

  /// `fixup_if_stmts=false` reconciles ForStmt carries and preserves already
  /// shared tile windows across if results, without inserting branch copies. Used under
  /// memory_planner=PtoAS, where PTO codegen already re-points a branch-local
  /// producer at the phi handle (#1956/#1985) and copies in anything it declines
  /// to re-point — a copy-free path that an IR-level `tile.move` would displace
  /// with an extra buffer plus a `pto.tmov`. Loop carries have no such codegen
  /// path, so they still need the move.
  explicit YieldFixupMutator(bool fixup_if_stmts, bool canonical_branch_storage = false)
      : fixup_if_stmts_(fixup_if_stmts), canonical_branch_storage_(canonical_branch_storage) {}

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto result = IRMutator::VisitStmt_(op);
    if (selections_ || fixup_if_stmts_) return result;

    // Follow semantic aliases of a repaired phi. Registered views re-derive
    // their relative window; writeback results keep their input's whole window.
    // A new SSA input alone does not refresh either result's stored MemRef.
    VarPtr input = AsVarLike(op->value_);
    bool is_aliasing_view = false;
    if (auto call = As<Call>(op->value_); call && call->op_) {
      is_aliasing_view = op_predicates::IsBufferAliasingViewOp(call->op_->name_);
      auto index = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
      if (!index && !call->args_.empty() && is_aliasing_view) {
        index = 0;
      }
      if (index) input = AsVarLike(call->args_[*index]);
    }
    if (!input) return result;
    auto new_input = AsVarLike(VisitExpr(input));
    if (!new_input || repaired_tile_aliases_.count(new_input.get()) == 0) return result;
    auto output_tile = GetTileTypeWithMemRef(op->var_->GetType());
    auto input_tile = GetTileTypeWithMemRef(input->GetType());
    auto new_input_tile = GetTileTypeWithMemRef(new_input->GetType());
    if (!SameKnownTileMemorySpace(input_tile, output_tile) ||
        !SameKnownTileMemorySpace(input_tile, new_input_tile) ||
        SamePhysicalWindow(GetDefinedMemRef(input_tile), GetDefinedMemRef(new_input_tile))) {
      return result;
    }
    auto assignment = As<AssignStmt>(result);
    auto target = GetDefinedMemRef(new_input_tile);
    if (is_aliasing_view) {
      // The view's old absolute offset belongs to the abandoned phi allocation.
      // Recompute only its relative offset from the rewritten call, retaining its
      // own extent and inheriting the repaired source's allocation/slot identity.
      auto view_call = As<Call>(assignment->value_);
      auto relative_offset = ComputeViewByteOffset(view_call, new_input_tile);
      target = MakeAliasingMemRef(target, relative_offset, GetDefinedMemRef(output_tile)->size_, op->span_);
    } else if (!SamePhysicalWindow(GetDefinedMemRef(output_tile), GetDefinedMemRef(input_tile))) {
      return result;
    }
    auto new_type = CloneTypeWithMemRef(assignment->var_->GetType(), target);
    auto new_var = std::make_shared<Var>(op->var_->name_hint_, new_type, op->var_->span_);
    var_remap_[op->var_.get()] = new_var;
    repaired_tile_aliases_.insert(new_var.get());
    return std::make_shared<AssignStmt>(new_var, assignment->value_, assignment->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (selections_) return op;  // Loop-touched storage is excluded from the plan.
    if (canonical_branch_storage_ && !LoopMayExecute(op, false)) {
      return IRMutator::VisitStmt_(As<ForStmt>(FixupCanonicalLoop(op)));
    }
    // First recurse into nested control flow
    auto result = IRMutator::VisitStmt_(op);
    auto for_stmt = As<ForStmt>(result);
    if (!for_stmt || for_stmt->iter_args_.empty()) return result;
    if (canonical_branch_storage_) return FixupCanonicalLoop(for_stmt);

    auto yield_stmt = FindYieldStmt(for_stmt->body_);
    if (!yield_stmt) return result;

    // Check each (iter_arg, yield_value) pair for MemRef mismatch.
    // Use initValue's MemRef as the target because codegen maps iter_arg to initValue's buffer,
    // and MemoryReuse may have updated initValue's MemRef without updating iter_arg's.
    // One index over the carry buffers serves both the overlap rejection below
    // and the conflict graph in OrderCarryCopies.
    const std::vector<MemRefPtr> carry_ranges = CollectCarryRanges(for_stmt);
    const CarryRangeIndex carry_index(carry_ranges);
    RejectOverlappingCarryBuffers(for_stmt, carry_index);

    std::vector<CarryCopy> copies;
    for (size_t i = 0; i < yield_stmt->value_.size() && i < for_stmt->iter_args_.size(); ++i) {
      // AsVarLike, not As<Var>: a nested loop both yields and seeds carries with
      // the enclosing loop's IterArg, whose own ObjectKind As<Var> does not match.
      // Skipping those left the nested carry with no writeback at all.
      auto yield_var = AsVarLike(yield_stmt->value_[i]);
      if (!yield_var) continue;

      // Yielding this position's own iter_arg leaves the carry untouched, so
      // there is nothing to write back. Short-circuit before comparing buffers:
      // it is the one case where the two sides are the same storage by
      // construction, whatever their (possibly still stale) types say.
      if (yield_var.get() == static_cast<const Var*>(for_stmt->iter_args_[i].get())) continue;

      auto yield_tile = GetTileTypeWithMemRef(yield_var->GetType());
      if (!yield_tile) continue;
      auto yield_memref = GetDefinedMemRef(yield_tile);

      // Get the initValue's MemRef (the actual buffer used at runtime)
      auto init_var = AsVarLike(for_stmt->iter_args_[i]->initValue_);
      if (!init_var) continue;
      auto init_tile = GetTileTypeWithMemRef(init_var->GetType());
      if (!init_tile) continue;
      auto init_memref = GetDefinedMemRef(init_tile);

      // Compare addresses, not allocations: two slots of one pl.MemRef(slots=N)
      // share a base, so "same allocation" would skip the copy that moves the
      // value from one slot to the other and silently carry the wrong slot.
      const AddressRelation relation = CompareBaseAddress(yield_memref, init_memref);
      if (relation == AddressRelation::kSame) continue;
      // Two symbolic offsets into one allocation may or may not name the same
      // bytes. Copying is wrong if they coincide (a tile.move onto itself, which
      // Acc cannot lower at all) and skipping is wrong if they do not, so say so
      // rather than pick one and hope.
      CHECK_SPAN(relation == AddressRelation::kDifferent, for_stmt->span_)
          << "Loop-carried value '" << for_stmt->iter_args_[i]->name_hint_
          << "' is yielded at a slot of its own allocation that cannot be shown equal to, or "
             "distinct from, the slot it is carried in, so whether the carry needs a copy is "
             "undecidable here. Subscript the declared allocation with the same expression at both "
             "ends, or give the carry its own allocation.";

      // MemRef mismatch — create tile.move to copy yield value into initValue's buffer
      auto target_memory = init_tile->GetMemorySpace();
      auto [moved_var, move_stmt] = CreateTileMove(yield_var, init_memref, target_memory);

      copies.push_back(CarryCopy{i, yield_var, std::move(moved_var), std::move(move_stmt), yield_memref,
                                 init_memref, target_memory});
    }

    if (copies.empty()) {
      // Even without tile.move insertion, iter_arg and return_var may have stale MemRefs
      // (MemoryReuse updates initValue/yield but not iter_arg/return_var types).
      // Ensure all 4 loop-carry variables share the same MemRef (= initValue's).
      return PatchIterArgsAndReturnVars(for_stmt, yield_stmt);
    }

    // `pl.yield_` rebinds every carry simultaneously, so the copies that realize
    // it must not observe each other's writes. Order them first.
    std::vector<StmtPtr> spill_stmts;
    const std::vector<size_t> order = OrderCarryCopies(&copies, carry_index, &spill_stmts);

    // Build new yield values with moved vars substituted
    std::vector<ExprPtr> new_yield_values = yield_stmt->value_;
    for (const auto& copy : copies) {
      new_yield_values[copy.index] = copy.moved_var;
    }
    auto new_yield = MutableCopy(yield_stmt);
    new_yield->value_ = std::move(new_yield_values);

    // A spill reads a carry buffer before any copy writes one, so every spill
    // precedes every copy.
    std::vector<StmtPtr> move_stmts = std::move(spill_stmts);
    move_stmts.reserve(move_stmts.size() + order.size());
    for (size_t idx : order) move_stmts.push_back(copies[idx].move_stmt);

    // Insert tile.move stmts before yield and replace yield in body
    auto new_body = InsertMovesAndReplaceYield(for_stmt->body_, new_yield, move_stmts);

    // Build intermediate ForStmt with new body, then patch iter_args/return_vars
    auto intermediate_for = MutableCopy(for_stmt);
    intermediate_for->body_ = new_body;

    return PatchIterArgsAndReturnVars(intermediate_for, new_yield);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    if (selections_) return op;
    if (canonical_branch_storage_ && !LoopMayExecute(op, false)) {
      return IRMutator::VisitStmt_(As<WhileStmt>(FixupCanonicalLoop(op)));
    }
    auto result = IRMutator::VisitStmt_(op);
    if (!canonical_branch_storage_) return result;
    return FixupCanonicalLoop(As<WhileStmt>(result));
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    // First recurse into nested control flow
    auto result = IRMutator::VisitStmt_(op);
    auto if_stmt = As<IfStmt>(result);
    if (!if_stmt || if_stmt->return_vars_.empty()) return result;
    if (selections_) {
      // A logical if result selects a source window; it does not require either
      // arm to copy its payload onto the other arm. Rewrite the signature and
      // both yields together, retaining the original condition and body order.
      const auto& plan = *selections_;
      auto selected = MutableCopy(if_stmt);
      selected->then_body_ = RewriteSelectionYield(if_stmt->then_body_, op->return_vars_, plan);
      if (if_stmt->else_body_) {
        selected->else_body_ = RewriteSelectionYield(*if_stmt->else_body_, op->return_vars_, plan);
      }
      return FinishSelection(op->return_vars_, selected, plan);
    }
    if (canonical_branch_storage_) return FixupCanonicalIf(if_stmt);

    // Find yield statements in each branch
    auto then_yield = FindYieldStmt(if_stmt->then_body_);
    auto else_yield = if_stmt->else_body_.has_value() ? FindYieldStmt(if_stmt->else_body_.value()) : nullptr;
    if (!then_yield && !else_yield) return result;

    // For each return_var position, check if the two branches yield tiles
    // with different MemRefs.  When they differ, insert tile.move in the
    // branch whose MemRef ≠ the canonical target (then-branch is canonical).
    bool body_changed = false;
    std::vector<std::pair<size_t, VarPtr>> else_moves;
    std::vector<StmtPtr> else_move_stmts;
    std::vector<VarPtr> new_return_vars = if_stmt->return_vars_;

    for (size_t i = 0; i < new_return_vars.size(); ++i) {
      // AsVarLike, not As<Var>: a branch may yield the enclosing loop's IterArg
      // unchanged (`if cond: acc = f(acc)` leaves `yield acc_iter` on the other
      // arm).  IterArg has its own ObjectKind, so As<Var> returns null there and
      // the arm would silently skip its reconciling move while the phi's
      // return_var is still patched onto the sibling arm's buffer -- leaving the
      // phi buffer unwritten on that path and feeding garbage into the carry.
      VarPtr then_var =
          (then_yield && i < then_yield->value_.size()) ? AsVarLike(then_yield->value_[i]) : nullptr;
      VarPtr else_var =
          (else_yield && i < else_yield->value_.size()) ? AsVarLike(else_yield->value_[i]) : nullptr;

      auto then_tile = then_var ? GetTileTypeWithMemRef(then_var->GetType()) : nullptr;
      auto else_tile = else_var ? GetTileTypeWithMemRef(else_var->GetType()) : nullptr;
      auto rv_tile = As<TileType>(new_return_vars[i]->GetType());
      if (!then_tile && !else_tile) continue;

      // PTOAS owns reconciliation of distinct branch buffers. Only propagate
      // an existing alias here: both arms must denote the exact same byte window,
      // including its extent and pipeline slot, in the result's known memory
      // space. Keep the result's view metadata; no producer is retargeted.
      if (!fixup_if_stmts_ &&
          (!SameKnownTileMemorySpace(then_tile, else_tile) || !SameKnownTileMemorySpace(then_tile, rv_tile) ||
           !SamePhysicalWindow(GetDefinedMemRef(then_tile), GetDefinedMemRef(else_tile)))) {
        continue;
      }

      // Use then-branch MemRef as canonical target (fall back to else if then absent)
      auto target_tile = then_tile ? then_tile : else_tile;
      auto target_memref = GetDefinedMemRef(target_tile);
      auto target_memory = target_tile->GetMemorySpace();

      // Check else branch: if MemRef differs from target, insert tile.move
      if (fixup_if_stmts_ && then_tile && else_tile) {
        auto else_memref = GetDefinedMemRef(else_tile);
        // Addresses, not allocations: an arm sitting on a sibling slot of the same
        // allocation still has to be copied into the phi's slot.
        const AddressRelation arm = CompareBaseAddress(else_memref, target_memref);
        CHECK_SPAN(arm != AddressRelation::kUnknown, if_stmt->span_)
            << "Branch result '" << if_stmt->return_vars_[i]->name_hint_
            << "' is produced at a slot of its own allocation that cannot be shown equal to, or "
               "distinct from, the slot the other branch produces, so whether this arm needs a copy "
               "is undecidable here. Subscript the declared allocation with the same expression in "
               "both branches, or give the branches separate allocations.";
        if (arm == AddressRelation::kDifferent) {
          auto [moved_var, move_stmt] = CreateTileMove(else_var, target_memref, target_memory);
          else_move_stmts.emplace_back(std::move(move_stmt));
          else_moves.emplace_back(i, moved_var);
          body_changed = true;
        }
      }

      // Patch return_var to share target MemRef
      if (rv_tile && rv_tile->memref_.has_value()) {
        auto rv_memref = GetDefinedMemRef(rv_tile);
        // Metadata only, so realign anything not provably already the target.
        const bool shares_target =
            fixup_if_stmts_ ? CompareBaseAddress(rv_memref, target_memref) == AddressRelation::kSame
                            : SamePhysicalWindow(rv_memref, target_memref);
        if (!shares_target) {
          auto new_rv_type = CloneTypeWithMemRefAndRemapExprs(
              rv_tile, target_memref, [this](const ExprPtr& e) { return VisitExpr(e); }, target_memory);
          new_return_vars[i] =
              std::make_shared<Var>(new_return_vars[i]->name_hint_, new_rv_type, new_return_vars[i]->span_);
          var_remap_[if_stmt->return_vars_[i].get()] = new_return_vars[i];
          if (!fixup_if_stmts_) repaired_tile_aliases_.insert(new_return_vars[i].get());
          body_changed = true;
        }
      }
    }

    if (!body_changed) return result;

    // Rebuild else body with tile.move stmts inserted before yield
    auto new_else_body = if_stmt->else_body_;
    if (!else_moves.empty() && else_yield && if_stmt->else_body_.has_value()) {
      std::vector<ExprPtr> new_else_yield_values = else_yield->value_;
      for (const auto& [idx, moved_var] : else_moves) {
        new_else_yield_values[idx] = moved_var;
      }
      auto new_yield = MutableCopy(else_yield);
      new_yield->value_ = std::move(new_else_yield_values);
      new_else_body = InsertMovesAndReplaceYield(if_stmt->else_body_.value(), new_yield, else_move_stmts);
    }

    auto new_if = MutableCopy(if_stmt);
    new_if->else_body_ = new_else_body;
    new_if->return_vars_ = std::move(new_return_vars);
    return new_if;
  }

 public:
  /// Allocations this fixup had to create (cycle spills), for the caller to
  /// hoist onto the function-body head. Empty for every loop whose carries do
  /// not form a copy cycle, which is the overwhelmingly common case.
  std::vector<StmtPtr> TakePendingAllocs() { return std::move(pending_allocs_); }

 private:
  std::optional<StorageSelectionPlan> selections_;

  static StmtPtr SelectionBarrier(const Span& span) {
    return std::make_shared<EvalStmt>(OpRegistry::GetInstance().Create("system.bar_all", {}, {}, span), span);
  }

  StmtPtr SelectionDescriptor(const VarPtr& original, const StorageSelectionPlan& plan) {
    const auto& variable = plan.tiles.at(original.get());
    const auto tile = As<TileType>(variable->GetType());
    const auto memory_space = tile->GetMemorySpace();
    INTERNAL_CHECK_SPAN(memory_space.has_value(), original->span_)
        << "Storage selection descriptor must have a memory space";
    const auto create = OpRegistry::GetInstance().Create(
        "tile.create", {std::make_shared<MakeTuple>(tile->shape_, original->span_)},
        {{"dtype", tile->dtype_}, {"target_memory", memory_space.value()}}, original->span_);
    return std::make_shared<AssignStmt>(variable, create, original->span_);
  }

  StmtPtr RewriteSelectionYield(const StmtPtr& body, const std::vector<VarPtr>& results,
                                const StorageSelectionPlan& plan) {
    return transform_utils::RewriteLastYieldStmt(body, [&](const YieldStmtPtr& yield) -> StmtPtr {
      auto values = yield->value_;
      bool changed = false;
      for (size_t i = 0; i < values.size() && i < results.size(); ++i) {
        if (!plan.tiles.count(results[i].get())) continue;
        const auto source = AsVarLike(values[i]);
        INTERNAL_CHECK_SPAN(source, yield->span_) << "Storage selection must yield a storage variable";
        values[i] = plan.indices.at(source.get());
        changed = true;
      }
      return changed ? std::make_shared<YieldStmt>(values, yield->span_) : yield;
    });
  }

  StmtPtr FinishSelection(const std::vector<VarPtr>& original, const std::shared_ptr<IfStmt>& rewritten,
                          const StorageSelectionPlan& plan) {
    std::vector<StmtPtr> suffix;
    for (size_t i = 0; i < original.size(); ++i) {
      if (!plan.tiles.count(original[i].get())) continue;
      rewritten->return_vars_[i] = AsVarLike(plan.indices.at(original[i].get()));
      suffix.push_back(SelectionDescriptor(original[i], plan));
    }
    if (suffix.empty()) return rewritten;
    // Wait after the region so branch-local asynchronous producers are also
    // complete before a selected descriptor is consumed.
    std::vector<StmtPtr> statements{rewritten, SelectionBarrier(rewritten->span_)};
    statements.insert(statements.end(), suffix.begin(), suffix.end());
    return SeqStmts::Flatten(statements, rewritten->span_);
  }

  bool fixup_if_stmts_ = true;
  bool canonical_branch_storage_ = false;
  std::set<const Var*> repaired_tile_aliases_;
  StorageNameSupply names_;

  // The enabled pipeline establishes each phi's target before planning. Never
  // choose a branch input as a new target here: it may remain independently live
  // after the IfStmt. Both arms must explicitly write the declared destination.
  StmtPtr FixupCanonicalIf(const IfStmtPtr& branch) {
    auto then_body = FixupCanonicalBoundary(branch->then_body_, branch->return_vars_, branch->span_);
    auto else_body = branch->else_body_;
    if (else_body) else_body = FixupCanonicalBoundary(*else_body, branch->return_vars_, branch->span_);
    if (then_body == branch->then_body_ && else_body == branch->else_body_) return branch;
    auto result = MutableCopy(branch);
    result->then_body_ = std::move(then_body);
    result->else_body_ = std::move(else_body);
    return result;
  }

  template <typename LoopPtr>
  StmtPtr FixupCanonicalLoop(const LoopPtr& loop) {
    // Tensor descriptors are reanchored by their dedicated boundary logic.
    // This fixup only reconciles Tile storage; preserve Tensor-only bodies,
    // including their writeback statements and sequence structure at zero trips.
    if (std::none_of(loop->iter_args_.begin(), loop->iter_args_.end(),
                     [](const IterArgPtr& argument) { return As<TileType>(argument->GetType()); })) {
      return loop;
    }
    if (!LoopMayExecute(loop, false)) {
      // The body cannot run, so no transfer into a carry is needed. Give the
      // unreachable boundary canonical identity yields rather than requesting
      // an unsupported copy solely to reconcile its dead source storage.
      auto result = MutableCopy(loop);
      std::vector<ExprPtr> values(loop->iter_args_.begin(), loop->iter_args_.end());
      result->body_ = std::make_shared<YieldStmt>(std::move(values), loop->span_);
      return result;
    }
    std::vector<VarPtr> targets(loop->iter_args_.begin(), loop->iter_args_.end());
    auto body = FixupCanonicalBoundary(loop->body_, targets, loop->span_);
    if (body == loop->body_) return loop;
    auto result = MutableCopy(loop);
    result->body_ = std::move(body);
    return result;
  }

  StmtPtr FixupCanonicalBoundary(const StmtPtr& body, const std::vector<VarPtr>& targets, const Span& span) {
    auto yield = transform_utils::GetLastYieldStmt(body);
    std::vector<ExprPtr> values = yield ? yield->value_ : std::vector<ExprPtr>{};
    std::vector<MemRefPtr> destinations(targets.size());
    std::vector<CarryCopy> copies;
    for (size_t i = 0; i < targets.size(); ++i) {
      auto target_tile = GetTileTypeWithMemRef(targets[i]->GetType());
      if (!target_tile) continue;
      CHECK_SPAN(yield && i < values.size(), span)
          << "Buffer IR storage legalization requires every region to yield every tile result";
      auto source = AsVarLike(values[i]);
      auto source_tile = source ? GetTileTypeWithMemRef(source->GetType()) : nullptr;
      CHECK_SPAN(source_tile, span)
          << "Buffer IR storage legalization requires tile yields to be storage variables";
      auto target = GetDefinedMemRef(target_tile);
      destinations[i] = target;
      auto source_memref = GetDefinedMemRef(source_tile);
      if (SamePhysicalWindow(source_memref, target)) continue;
      CHECK_SPAN(CompareBaseAddress(source_memref, target) == AddressRelation::kDifferent, span)
          << "Buffer IR storage legalization cannot reconcile ambiguous views of one destination";
      const auto source_memory = source_tile->GetMemorySpace();
      const auto target_memory = target_tile->GetMemorySpace();
      CHECK_SPAN(source_memory && target_memory && IsTileMoveEverSupported(*source_memory, *target_memory),
                 span)
          << "Buffer IR cannot reconcile region storage with an unsupported tile.move memory-space pair";
      auto [moved, move] = CreateTileMove(source, target, target_tile->GetMemorySpace());
      copies.push_back({i, source, moved, move, source_memref, target, target_tile->GetMemorySpace()});
    }
    if (copies.empty()) return body;
    const CarryRangeIndex index(destinations);
    CHECK_SPAN(index.OverlappingPairs().empty(), span)
        << "Buffer IR region transfers require independent destination windows";
    // Snapshot each source that another transfer may overwrite. This deliberately
    // uses overlap existence queries, not an O(k^2) graph for broad/fanout views.
    // Every snapshot precedes every destination write: swaps, cycles, partial
    // overlap and fanout therefore share one O(k log k) algorithm.
    std::vector<StmtPtr> moves;
    for (auto& copy : copies) {
      if (index.AnyOverlap(copy.src_memref)) SpillCarrySource(&copy, &moves);
    }
    for (const auto& copy : copies) {
      moves.push_back(copy.move_stmt);
      values[copy.index] = copy.moved_var;
    }
    auto new_yield = MutableCopy(yield);
    new_yield->value_ = std::move(values);
    return InsertMovesAndReplaceYield(body, new_yield, moves);
  }

  /// One reconciling copy for one loop carry: `dst_memref <- source`.
  struct CarryCopy {
    size_t index;                           ///< iter_arg / yield position
    VarPtr source;                          ///< what the copy reads
    VarPtr moved_var;                       ///< the copy's LHS, replacing the yield value
    StmtPtr move_stmt;                      ///< the `tile.move` itself
    MemRefPtr src_memref;                   ///< byte range the copy reads
    MemRefPtr dst_memref;                   ///< the carry buffer the copy writes
    std::optional<MemorySpace> dst_memory;  ///< memory space of `dst_memref`
  };

  std::vector<StmtPtr> pending_allocs_;
  uint64_t spill_counter_ = 0;

  /// Byte ranges grouped by allocation, so overlap questions are index lookups
  /// rather than all-pairs scans.
  ///
  /// Within an allocation the constant-offset ranges are sorted by start, which
  /// makes both users linear-after-sort: the pairwise-overlap check only has to
  /// compare neighbours (sorted by start, if any two ranges overlap then some
  /// adjacent pair does), and a query binary-searches to its first candidate and
  /// stops at the first range past its end. A range whose offset is not constant
  /// has no position on that axis, so it is kept aside and treated as touching
  /// everything in its allocation — the same conservative answer
  /// `MemRef::MayAlias` gives.
  class CarryRangeIndex {
   public:
    /// `ranges[i]` is the byte range owned by carry `i`; a null entry is skipped.
    explicit CarryRangeIndex(const std::vector<MemRefPtr>& ranges) {
      for (size_t i = 0; i < ranges.size(); ++i) {
        if (!ranges[i]) continue;
        const Var* base = ranges[i]->base_.get();
        auto offset = As<ConstInt>(ranges[i]->byte_offset_);
        if (!offset) {
          unpositioned_[base].push_back(i);
          continue;
        }
        const int64_t begin = offset->value_;
        positioned_[base].push_back(Entry{begin, RangeEnd(begin, ranges[i]->size_, ranges[i]->span_), i});
      }
      for (auto& [base, entries] : positioned_) {
        static_cast<void>(base);
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b) { return a.begin < b.begin; });
        auto& maxima = prefix_end_[base];
        for (const auto& entry : entries) {
          maxima.push_back(maxima.empty() ? entry.end : std::max(maxima.back(), entry.end));
        }
      }
    }

    /// Bounded existence query for general overlapping source windows.
    [[nodiscard]] bool AnyOverlap(const MemRefPtr& query) const {
      const auto* base = query->base_.get();
      if (unpositioned_.count(base) != 0) return true;
      auto found = positioned_.find(base);
      if (found == positioned_.end() || found->second.empty()) return false;
      auto offset = As<ConstInt>(query->byte_offset_);
      if (!offset) return true;
      const auto& entries = found->second;
      const auto end = RangeEnd(offset->value_, query->size_, query->span_);
      auto limit = std::lower_bound(entries.begin(), entries.end(), end,
                                    [](const Entry& entry, int64_t value) { return entry.begin < value; });
      if (limit == entries.begin()) return false;
      return prefix_end_.at(base)[static_cast<size_t>(limit - entries.begin() - 1)] > offset->value_;
    }

    /// Call `fn(i)` for every indexed range that may overlap `query`.
    template <typename Fn>
    void ForEachOverlapping(const MemRefPtr& query, const Fn& fn) const {
      const Var* base = query->base_.get();
      if (auto loose = unpositioned_.find(base); loose != unpositioned_.end()) {
        for (size_t i : loose->second) fn(i);
      }
      auto bucket = positioned_.find(base);
      if (bucket == positioned_.end()) return;
      const auto& entries = bucket->second;

      auto offset = As<ConstInt>(query->byte_offset_);
      if (!offset) {  // unknown position — conservatively touches the whole allocation
        for (const Entry& entry : entries) fn(entry.index);
        return;
      }
      const int64_t begin = offset->value_;
      const int64_t end = RangeEnd(begin, query->size_, query->span_);
      // Entries are sorted by start, so the first candidate is the first one
      // starting at or after `begin`, less the single earlier one that may still
      // cover `begin`.
      auto it = std::lower_bound(entries.begin(), entries.end(), begin,
                                 [](const Entry& entry, int64_t value) { return entry.begin < value; });
      if (it != entries.begin() && std::prev(it)->end > begin) --it;
      for (; it != entries.end() && it->begin < end; ++it) {
        if (it->end > begin) fn(it->index);
      }
    }

    /// Every pair of indexed ranges that overlaps, as (lower index, higher index).
    [[nodiscard]] std::vector<std::pair<size_t, size_t>> OverlappingPairs() const {
      std::vector<std::pair<size_t, size_t>> pairs;
      for (const auto& [base, entries] : positioned_) {
        for (size_t n = 1; n < entries.size(); ++n) {
          if (entries[n].begin < entries[n - 1].end) {
            pairs.emplace_back(std::minmax(entries[n - 1].index, entries[n].index));
          }
        }
        // An unpositioned range conservatively covers its whole allocation, so it
        // conflicts with every other range there. One witness is enough to report.
        auto loose = unpositioned_.find(base);
        if (loose != unpositioned_.end() && !entries.empty()) {
          pairs.emplace_back(std::minmax(loose->second.front(), entries.front().index));
        }
      }
      for (const auto& [base, loose] : unpositioned_) {
        static_cast<void>(base);
        if (loose.size() > 1) pairs.emplace_back(std::minmax(loose[0], loose[1]));
      }
      return pairs;
    }

   private:
    struct Entry {
      int64_t begin;
      int64_t end;
      size_t index;
    };

    /// One past the last byte of `[begin, begin + size)`, as a signed offset.
    ///
    /// `MemRef::size_` is unsigned and the interval arithmetic below is signed,
    /// so a size past `INT64_MAX` or a sum that wraps would silently corrupt both
    /// the sort order and every overlap answer derived from it. No on-chip buffer
    /// comes anywhere near that, so a violation is a corrupt MemRef rather than
    /// something to accommodate.
    static int64_t RangeEnd(int64_t begin, uint64_t size, const Span& span) {
      constexpr auto kMax = std::numeric_limits<int64_t>::max();
      INTERNAL_CHECK_SPAN(size <= static_cast<uint64_t>(kMax), span)
          << "Internal error: MemRef size " << size << " does not fit a signed byte offset";
      const auto signed_size = static_cast<int64_t>(size);
      INTERNAL_CHECK_SPAN(begin <= kMax - signed_size, span)
          << "Internal error: MemRef byte range [" << begin << ", +" << size << ") overflows";
      return begin + signed_size;
    }
    std::map<const Var*, std::vector<Entry>> positioned_;
    std::map<const Var*, std::vector<int64_t>> prefix_end_;
    std::map<const Var*, std::vector<size_t>> unpositioned_;
  };

  /// The byte range each carry owns, indexed by iter_arg position (null when the
  /// carry is not a tile with a MemRef).
  static std::vector<MemRefPtr> CollectCarryRanges(const ForStmtPtr& for_stmt) {
    std::vector<MemRefPtr> ranges(for_stmt->iter_args_.size());
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      // AsVarLike: a nested loop seeds its carries from the enclosing loop's
      // IterArg, which As<Var> would skip -- and skipping it is exactly how two
      // nested carries seeded from one outer carry would slip past the check below.
      auto init_var = AsVarLike(for_stmt->iter_args_[i]->initValue_);
      if (!init_var) continue;
      auto init_tile = GetTileTypeWithMemRef(init_var->GetType());
      if (!init_tile) continue;
      ranges[i] = GetDefinedMemRef(init_tile);
    }
    return ranges;
  }

  /// Reject a loop whose carries do not each own their buffer.
  ///
  /// Two carries whose buffers overlap cannot both survive an iteration: whatever
  /// order the writebacks run in, the second one destroys the first. The usual
  /// source is seeding two `pl.range` carries from one tile (`lag1 = one;
  /// lag2 = one`), which makes both carries that tile's buffer. That is a real
  /// limitation of the carry lowering rather than a compiler invariant, so it is
  /// reported to the author -- and reported here, where the carries are still
  /// named, rather than downstream as a degenerate `tile.move` onto itself.
  ///
  /// Byte ranges, not allocations: disjoint slices of one declared multi-slot
  /// buffer are independent carries and stay legal.
  static void RejectOverlappingCarryBuffers(const ForStmtPtr& for_stmt, const CarryRangeIndex& index) {
    for (const auto& [lhs, rhs] : index.OverlappingPairs()) {
      CHECK_SPAN(false, for_stmt->span_)
          << "Loop-carried values '" << for_stmt->iter_args_[lhs]->name_hint_ << "' and '"
          << for_stmt->iter_args_[rhs]->name_hint_
          << "' share the same on-chip buffer, so one iteration cannot preserve both. Give each "
             "loop-carried tile its own initial value -- build them with separate ops instead of "
             "seeding both from the same tile.";
    }
  }

  /// Order the carry copies so none overwrites a buffer another still reads.
  ///
  /// `pl.yield_` rebinds every carry at once, but the copies realizing it run in
  /// sequence, so copy A must precede copy B whenever A reads the buffer B
  /// writes. Emitting them in iter_arg order instead collapses a shift register:
  /// for `lag2 = lag1; lag1 = v`, writing lag1's buffer first makes lag2 read the
  /// new lag1 and the loop silently carries `lag2 == lag1` ([#2481]).
  ///
  /// Returns indices into `*copies` in emission order. A copy cycle — the swap
  /// `a, b = b, a`, where each carry's value lives in the other's buffer — has no
  /// valid order, so its members are spilled into scratch buffers first (appended
  /// to `*spills`, which the caller emits ahead of every copy) and their copies
  /// read the spills instead. A spill destination is a fresh allocation nothing
  /// else writes, so spilling removes every outgoing edge of the spilled copy.
  ///
  /// The graph is built once and consumed incrementally: a spill drops only the
  /// spilled copy's own outgoing edges, so the traversal resumes where it stalled
  /// instead of rebuilding anything. Victims come from the residual graph's
  /// strongly connected components, so a node merely downstream of a cycle is
  /// never spilled, and one victim set is enough to leave the residual acyclic —
  /// the traversal stalls at most once.
  ///
  /// Cost is O(k log k + E) in one loop's carry count k: `carry_index` answers each
  /// source with a binary search plus one step per range it actually overlaps, so
  /// k carries on disjoint slots of one allocation cost k lookups rather than k^2
  /// pairwise tests, and every edge is relaxed at most once.
  std::vector<size_t> OrderCarryCopies(std::vector<CarryCopy>* copies, const CarryRangeIndex& carry_index,
                                       std::vector<StmtPtr>* spills) {
    const size_t count = copies->size();

    // A copy writes its carry's byte range, so `carry_index` — built over exactly
    // those ranges — already indexes every destination. Map the carry positions it
    // reports back onto copy positions; a carry with no copy has nothing to order.
    std::map<size_t, size_t> copy_of_carry;
    for (size_t j = 0; j < count; ++j) copy_of_carry.emplace((*copies)[j].index, j);

    // A copy reads its source's byte range and writes its destination's, so two
    // copies conflict exactly when one's source overlaps the other's destination.
    std::vector<std::vector<size_t>> successors(count);
    std::vector<size_t> in_degree(count, 0);
    for (size_t i = 0; i < count; ++i) {
      carry_index.ForEachOverlapping((*copies)[i].src_memref, [&](size_t carry) {
        auto hit = copy_of_carry.find(carry);
        if (hit == copy_of_carry.end() || hit->second == i) return;
        successors[i].push_back(hit->second);  // i reads what hit->second overwrites
        ++in_degree[hit->second];
      });
    }

    std::vector<size_t> order;
    order.reserve(count);
    std::vector<bool> emitted(count, false);
    std::vector<size_t> ready;
    for (size_t i = 0; i < count; ++i) {
      if (in_degree[i] == 0) ready.push_back(i);
    }

    auto release = [&](size_t node) {
      for (size_t next : successors[node]) {
        if (--in_degree[next] == 0 && !emitted[next]) ready.push_back(next);
      }
      successors[node].clear();
    };

    while (order.size() < count) {
      if (!ready.empty()) {
        const size_t node = ready.back();
        ready.pop_back();
        if (emitted[node]) continue;
        // Kahn: every predecessor is already emitted, and an edge means "must run
        // before", so appending here builds the emission order directly.
        emitted[node] = true;
        order.push_back(node);
        release(node);
        continue;
      }

      // Nothing is runnable, so every remaining copy sits on or behind a cycle.
      // Take victims from the strongly connected components rather than from the
      // first node that still has an outgoing edge: such a node need not be on a
      // cycle at all, and spilling one that merely sits downstream of one costs a
      // scratch buffer without unblocking anything. One victim per component, all
      // of them in this round, so k/2 independent two-cycles cost a single pass.
      const std::vector<size_t> victims = CycleVictims(successors, emitted);
      INTERNAL_CHECK_SPAN(!victims.empty(), (*copies)[0].source->span_)
          << "Internal error: loop-carry copy ordering stalled with no cycle to break";
      // CycleVictims returns a set that leaves the residual acyclic, so Kahn
      // drains the rest without stalling again.
      for (size_t victim : victims) {
        SpillCarrySource(&(*copies)[victim], spills);
        release(victim);
        if (in_degree[victim] == 0 && !emitted[victim]) ready.push_back(victim);
      }
    }
    return order;
  }

  /// The strongly connected components of `nodes`, using only edges between
  /// nodes still marked active.
  ///
  /// Tarjan, iteratively so a deep chain cannot overflow the stack. Restricting
  /// to a node subset is what lets the caller re-decompose one component without
  /// walking the whole graph again.
  static std::vector<std::vector<size_t>> StronglyConnectedComponents(
      const std::vector<std::vector<size_t>>& successors, const std::vector<char>& active,
      const std::vector<size_t>& nodes) {
    constexpr size_t kUnvisited = std::numeric_limits<size_t>::max();
    const size_t count = successors.size();
    std::vector<size_t> index(count, kUnvisited);
    std::vector<size_t> lowlink(count, 0);
    std::vector<char> in_subset(count, 0);
    std::vector<char> on_stack(count, 0);
    for (size_t node : nodes) in_subset[node] = 1;

    std::vector<size_t> component_stack;
    std::vector<std::vector<size_t>> components;
    size_t next_index = 0;

    for (size_t root : nodes) {
      if (!active[root] || index[root] != kUnvisited) continue;
      index[root] = lowlink[root] = next_index++;
      component_stack.push_back(root);
      on_stack[root] = 1;
      std::vector<std::pair<size_t, size_t>> frames{{root, size_t{0}}};

      while (!frames.empty()) {
        // Re-read the frame each step: descending can reallocate the vector.
        const size_t node = frames.back().first;
        if (frames.back().second < successors[node].size()) {
          const size_t next = successors[node][frames.back().second++];
          if (!in_subset[next] || !active[next]) continue;
          if (index[next] == kUnvisited) {
            index[next] = lowlink[next] = next_index++;
            component_stack.push_back(next);
            on_stack[next] = 1;
            frames.emplace_back(next, size_t{0});
          } else if (on_stack[next]) {
            lowlink[node] = std::min(lowlink[node], index[next]);
          }
          continue;
        }

        if (lowlink[node] == index[node]) {
          std::vector<size_t> component;
          while (true) {
            const size_t member = component_stack.back();
            component_stack.pop_back();
            on_stack[member] = 0;
            component.push_back(member);
            if (member == node) break;
          }
          components.push_back(std::move(component));
        }

        frames.pop_back();
        if (!frames.empty()) {
          const size_t parent = frames.back().first;
          lowlink[parent] = std::min(lowlink[parent], lowlink[node]);
        }
      }
    }
    return components;
  }

  /// A set of copies whose spilling leaves the residual graph acyclic.
  ///
  /// Every cycle lies inside one strongly connected component, so it is enough
  /// to make each component acyclic. Spilling a node drops all of its outgoing
  /// edges, which breaks every cycle through it -- but a component holding
  /// several cycles that do not all pass through one node stays cyclic after a
  /// single spill, so what is left of that component is decomposed again. The
  /// re-decomposition is confined to the component's own nodes: the caller gets
  /// one acyclic residual from one call and never re-walks the whole graph.
  ///
  /// Victims are always on a cycle at the moment they are chosen, so no scratch
  /// buffer is spent on a copy that merely sits downstream of one. The set is
  /// not guaranteed minimal -- a minimum feedback vertex set is NP-hard, and the
  /// lowest-numbered member is picked instead.
  ///
  /// Cost is O(k + E) for the first decomposition plus, per component that needs
  /// more than one spill, another pass over that component alone.
  static std::vector<size_t> CycleVictims(const std::vector<std::vector<size_t>>& successors,
                                          const std::vector<bool>& emitted) {
    const size_t count = successors.size();
    std::vector<char> active(count, 0);
    std::vector<size_t> all_nodes;
    for (size_t i = 0; i < count; ++i) {
      if (emitted[i]) continue;
      active[i] = 1;
      all_nodes.push_back(i);
    }

    std::vector<size_t> victims;
    std::vector<std::vector<size_t>> pending{std::move(all_nodes)};
    while (!pending.empty()) {
      const std::vector<size_t> subset = std::move(pending.back());
      pending.pop_back();
      for (auto& component : StronglyConnectedComponents(successors, active, subset)) {
        const bool has_self_edge = component.size() == 1 &&
                                   std::find(successors[component[0]].begin(), successors[component[0]].end(),
                                             component[0]) != successors[component[0]].end();
        if (component.size() < 2 && !has_self_edge) continue;

        const size_t victim = *std::min_element(component.begin(), component.end());
        victims.push_back(victim);
        active[victim] = 0;  // its outgoing edges are gone once it reads a spill

        // The rest of this component may still hold a cycle the victim was not on.
        std::vector<size_t> remainder;
        for (size_t member : component) {
          if (member != victim) remainder.push_back(member);
        }
        if (remainder.size() > 1) pending.push_back(std::move(remainder));
      }
    }
    return victims;
  }

  /// Redirect one carry copy through a fresh scratch buffer, so the copy no
  /// longer reads a buffer its siblings overwrite. Emits the scratch allocation
  /// into `pending_allocs_` for the caller to hoist.
  void SpillCarrySource(CarryCopy* copy, std::vector<StmtPtr>* spills) {
    auto src_tile = GetTileTypeWithMemRef(copy->source->GetType());
    INTERNAL_CHECK_SPAN(src_tile, copy->source->span_)
        << "Internal error: a loop-carry copy source must be a TileType with a MemRef";
    auto src_memref = GetDefinedMemRef(src_tile);
    auto src_memory = src_tile->GetMemorySpace();
    if (canonical_branch_storage_) {
      CHECK_SPAN(src_memory && IsTileMoveEverSupported(*src_memory, *src_memory), copy->source->span_)
          << "Buffer IR cannot snapshot a transfer source without a supported same-space tile.move";
    }
    INTERNAL_CHECK_SPAN(src_memory.has_value(), copy->source->span_)
        << "Internal error: a loop-carry copy source must carry a memory space to spill";

    // `mem_<space>_carry_spill_N` cannot collide with InitMemRef's
    // `mem_<space>_<N>`, which never has a non-numeric segment before its
    // counter, and the counter is per-function so two spills cannot collide
    // either. The name is cosmetic, but it has to stay unique for the
    // print -> parse roundtrip to rebind the same allocation.
    std::string space_str = MemorySpaceToString(*src_memory);
    std::transform(space_str.begin(), space_str.end(), space_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    const std::string base_name =
        canonical_branch_storage_ ? names_.Fresh("mem_" + space_str + "_carry_spill_")
                                  : "mem_" + space_str + "_carry_spill_" + std::to_string(spill_counter_++);
    auto scratch_base = std::make_shared<Var>(base_name, GetPtrType(), copy->source->span_);
    auto scratch_memref = std::make_shared<MemRef>(scratch_base, static_cast<int64_t>(0), src_memref->size_,
                                                   copy->source->span_);
    pending_allocs_.push_back(CreateAllocStatement(scratch_memref, *src_memory));

    auto [spilled_var, spill_stmt] = CreateTileMove(copy->source, scratch_memref, src_memory);
    spills->push_back(std::move(spill_stmt));

    auto [moved_var, move_stmt] = CreateTileMove(spilled_var, copy->dst_memref, copy->dst_memory);
    copy->source = spilled_var;
    copy->moved_var = std::move(moved_var);
    copy->move_stmt = std::move(move_stmt);
    copy->src_memref = scratch_memref;
  }

  // Create a tile.move operation that copies source into target_memref's buffer.
  // Returns (moved_var, move_assign_stmt).
  std::pair<VarPtr, StmtPtr> CreateTileMove(const VarPtr& source, const MemRefPtr& target_memref,
                                            std::optional<MemorySpace> target_memory) {
    return CreateStorageMove(
        source, target_memref, target_memory, [this](const ExprPtr& expr) { return VisitExpr(expr); },
        canonical_branch_storage_ ? names_.Fresh("storage_transfer_") : source->name_hint_ + "_mv");
  }

  // Patch iter_args and return_vars to share initValue's MemRef.
  // Returns the original ForStmt if no patching is needed.
  StmtPtr PatchIterArgsAndReturnVars(const ForStmtPtr& for_stmt, const YieldStmtPtr& yield_stmt) {
    bool changed = false;
    std::vector<IterArgPtr> new_iter_args = for_stmt->iter_args_;
    std::vector<VarPtr> new_return_vars = for_stmt->return_vars_;

    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      auto init_var = AsVarLike(for_stmt->iter_args_[i]->initValue_);
      if (!init_var) continue;
      auto init_tile = GetTileTypeWithMemRef(init_var->GetType());
      if (!init_tile) continue;
      auto init_memref = GetDefinedMemRef(init_tile);

      // Patch iter_arg if its MemRef differs from initValue's. Addresses, not
      // allocations: an iter_arg left on a sibling slot of one multi-slot
      // allocation is a different buffer and still has to be re-pointed. These
      // sites only retype a node, they never move data, so anything not provably
      // init's address is repointed: an unprovable pair costs nothing to realign
      // and would otherwise leave a possibly-wrong slot in the metadata.
      auto ia_tile = As<TileType>(for_stmt->iter_args_[i]->GetType());
      if (ia_tile && ia_tile->memref_.has_value()) {
        auto ia_memref = GetDefinedMemRef(ia_tile);
        if (CompareBaseAddress(ia_memref, init_memref) != AddressRelation::kSame) {
          auto new_ia_type = CloneTypeWithMemRefAndRemapExprs(
              ia_tile, init_memref, [this](const ExprPtr& expr) { return VisitExpr(expr); },
              init_tile->GetMemorySpace());
          new_iter_args[i] =
              std::make_shared<IterArg>(for_stmt->iter_args_[i]->name_hint_, new_ia_type,
                                        for_stmt->iter_args_[i]->initValue_, for_stmt->iter_args_[i]->span_);
          var_remap_[for_stmt->iter_args_[i].get()] = new_iter_args[i];
          changed = true;
        }
      }

      // Patch return_var to share yield value's MemRef (which should == initValue's after fixup)
      if (i >= new_return_vars.size() || !yield_stmt || i >= yield_stmt->value_.size()) continue;
      auto yield_var = AsVarLike(yield_stmt->value_[i]);
      if (!yield_var) continue;
      auto yield_tile = GetTileTypeWithMemRef(yield_var->GetType());
      if (!yield_tile) continue;
      auto yield_memref = GetDefinedMemRef(yield_tile);
      auto rv_tile = As<TileType>(new_return_vars[i]->GetType());
      if (!rv_tile || !rv_tile->memref_.has_value()) continue;
      auto rv_memref = GetDefinedMemRef(rv_tile);
      if (CompareBaseAddress(rv_memref, yield_memref) != AddressRelation::kSame) {
        auto new_rv_type = CloneTypeWithMemRefAndRemapExprs(
            rv_tile, yield_memref, [this](const ExprPtr& expr) { return VisitExpr(expr); },
            yield_tile->GetMemorySpace());
        new_return_vars[i] =
            std::make_shared<Var>(new_return_vars[i]->name_hint_, new_rv_type, new_return_vars[i]->span_);
        // Register old→new so downstream references (e.g., ReturnStmt) are updated
        var_remap_[for_stmt->return_vars_[i].get()] = new_return_vars[i];
        changed = true;
      }
    }

    if (!changed) return for_stmt;

    auto patched_body = VisitStmt(for_stmt->body_);

    for (const auto& old_iter_arg : for_stmt->iter_args_) {
      var_remap_.erase(old_iter_arg.get());
    }

    auto new_for = MutableCopy(for_stmt);
    new_for->iter_args_ = new_iter_args;
    new_for->body_ = patched_body;
    new_for->return_vars_ = std::move(new_return_vars);
    return new_for;
  }

  // Follow the same trailing path as GetLastYieldStmt. Keep transfers inside
  // transparent scopes and leave yields of nested control-flow regions alone.
  static StmtPtr InsertMovesAndReplaceYield(const StmtPtr& body, const YieldStmtPtr& new_yield,
                                            const std::vector<StmtPtr>& move_stmts) {
    if (As<YieldStmt>(body)) {
      // Body is just a yield — wrap moves + yield in SeqStmts
      std::vector<StmtPtr> stmts;
      stmts.insert(stmts.end(), move_stmts.begin(), move_stmts.end());
      stmts.push_back(new_yield);
      return SeqStmts::Flatten(std::move(stmts), body->span_);
    }
    if (auto seq = As<SeqStmts>(body)) {
      if (seq->stmts_.empty()) return body;
      std::vector<StmtPtr> new_children = seq->stmts_;
      new_children.back() = InsertMovesAndReplaceYield(new_children.back(), new_yield, move_stmts);
      return SeqStmts::Flatten(std::move(new_children), body->span_);
    }
    if (auto scope = As<SplitAivScopeStmt>(body)) {
      auto result = MutableCopy(scope);
      result->body_ = InsertMovesAndReplaceYield(scope->body_, new_yield, move_stmts);
      return result;
    }
    if (auto scope = As<RuntimeScopeStmt>(body)) {
      auto result = MutableCopy(scope);
      result->body_ = InsertMovesAndReplaceYield(scope->body_, new_yield, move_stmts);
      return result;
    }
    return body;
  }
};

// Check if a statement is a tile.alloc AssignStmt for an unused MemRef
bool IsUnusedAllocStmt(const StmtPtr& stmt, const std::set<const Var*>& used_bases) {
  auto assign = As<AssignStmt>(stmt);
  if (!assign) return false;
  auto call = As<Call>(assign->value_);
  if (!call) return false;
  if (!IsOp(call, "tile.alloc") && !IsOp(call, "tensor.alloc")) return false;
  // Alloc LHS is a Ptr Var — check if any MemRef's base_ still references it
  return used_bases.find(assign->var_.get()) == used_bases.end();
}

/// Strip the transient ``pipeline_membership`` attr from every Call once
/// MemoryReuse has consumed it (in AnalyzeAllocationLifetimes). The attr exists only to
/// carry pipeline-stage identity from LowerPipelineLoops to here; leaving it on
/// the IR would ride downstream into later passes and codegen. Stripping at the
/// end of MemoryReuse keeps the post-reuse IR clean.
class StripPipelineMembershipMutator : public IRMutator {
 public:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto visited = IRMutator::VisitExpr_(op);
    auto call = As<Call>(visited);
    if (!call || !call->HasAttr(kPipelineMembershipAttr)) return visited;
    auto new_attrs = StripAttr(call->attrs_, kPipelineMembershipAttr);
    return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, std::move(new_attrs),
                                  call->GetType(), call->span_);
  }
};

/// Reconcile required storage aliases after producer/control-flow retyping.
///
/// An AssignStmt whose value is a bare Var (`x = y`, not a Call) is a pure SSA
/// rename — `x` must alias `y`'s buffer.  CoalesceAccumulatorIfPhis retargets an
/// if-phi's producers + return_var onto the accumulator buffer, which can leave a
/// downstream copy of the (now-moved) return_var still typed on the old buffer:
///   c: Tile[..., mem_acc_17] = c_phi_2   // c_phi_2 was retargeted to mem_acc_5
/// Codegen would then store from the stale buffer.  This single forward pass
/// retypes every such copy's LHS to its RHS's MemRef and substitutes the LHS's
/// downstream uses, so the whole rename chain follows.  A no-op when no identity
/// copy has a buffer mismatch (the common case). Registered writeback results
/// and first-input buffer aliases obey the same forward-propagation rule.
class NormalizeIdentityCopyBuffersMutator : public StorageBoundaryMutator {
 public:
  NormalizeIdentityCopyBuffersMutator() = default;

  /// `unsettled` seeds the IfStmt return_vars whose buffer YieldFixup has not
  /// chosen yet; the traversal grows it with every required alias. See
  /// `ReanchorAliasingOutput` for why anchoring onto one is wrong.
  explicit NormalizeIdentityCopyBuffersMutator(std::set<const Var*> unsettled)
      : unsettled_(std::move(unsettled)) {}

  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = subst_.find(op);
    return it != subst_.end() ? it->second : op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto found = var_remap_.find(op.get());
    if (found != var_remap_.end()) return found->second;
    auto visited = As<IterArg>(StorageBoundaryMutator::VisitExpr_(op));
    INTERNAL_CHECK_SPAN(visited, op->span_) << "Internal error: IterArg visit must yield an IterArg";
    if (!AsTensorTypeLike(visited->GetType())) return visited;
    // GM loop carries are mutable references: codegen binds their handles to
    // the initializer, including on zero-trip loops. A renamed initializer may
    // have moved even when this carry's own type has not changed.
    if (auto type = TensorBoundaryType(visited->GetType(), visited->initValue_)) {
      visited = std::make_shared<IterArg>(visited->name_hint_, type, visited->initValue_, visited->span_);
    }
    if (IsUnsettled(AsVarLike(op->initValue_)) || IsUnsettled(AsVarLike(visited->initValue_))) {
      unsettled_.insert(op.get());
      unsettled_.insert(visited.get());
    }
    var_remap_[op.get()] = visited;
    return visited;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return NormalizeTensorLoop(op); }
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return NormalizeTensorLoop(op); }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Only bare-Var identity copies are pure renames; Call producers are not.
    auto src_var = AsVarLike(op->value_);
    if (src_var) {
      auto new_src = AsVarLike(VisitExpr(op->value_));  // follow prior substitutions
      // A rename of an unsettled value denotes that same storage, so it is just as
      // unsettled: `tmp = phi` then `t = <in-place>(tmp, ...)` would otherwise walk
      // straight past the guard in ReanchorAliasingOutput and anchor onto the phi's
      // stale buffer through the alias.
      if (IsUnsettled(src_var) || IsUnsettled(new_src)) {
        unsettled_.insert(op->var_.get());
      }
      auto lhs = std::dynamic_pointer_cast<const ShapedType>(op->var_->GetType());
      auto rhs = new_src ? std::dynamic_pointer_cast<const ShapedType>(new_src->GetType()) : nullptr;
      if (lhs && rhs && lhs->memref_ && rhs->memref_ &&
          (!SamePhysicalWindow(*lhs->memref_, *rhs->memref_) ||
           lhs->GetMemorySpace() != rhs->GetMemorySpace())) {
        auto new_lhs = std::make_shared<Var>(op->var_->name_hint_, new_src->GetType(), op->var_->span_);
        if (IsUnsettled(op->var_)) unsettled_.insert(new_lhs.get());
        subst_[op->var_] = new_lhs;
        return std::make_shared<AssignStmt>(new_lhs, new_src, op->span_);
      }
    }
    // Required aliases must follow the current input even when another mutator
    // changed it earlier. Normalize views in this same walk so a writeback
    // repair cannot leave its downstream views on the previous buffer.
    if (auto reanchored = ReanchorAliasingOutput(op)) return reanchored;
    return IRMutator::VisitStmt_(op);
  }

 private:
  TypePtr TensorBoundaryType(const TypePtr& type, const ExprPtr& source) {
    auto target = AsTensorTypeLike(type);
    auto input = source ? AsTensorTypeLike(source->GetType()) : nullptr;
    if (!target || !input || !target->memref_ || !input->memref_ ||
        (SamePhysicalWindow(*target->memref_, *input->memref_) &&
         target->GetMemorySpace() == input->GetMemorySpace())) {
      return nullptr;
    }
    return CloneTypeWithMemRef(type, input->memref_, input->GetMemorySpace());
  }

  template <typename LoopPtr>
  StmtPtr NormalizeTensorLoop(const LoopPtr& op) {
    auto visited = std::static_pointer_cast<const typename LoopPtr::element_type>(IRMutator::VisitStmt_(op));
    auto result = MutableCopy(visited);
    bool changed = false;
    for (size_t i = 0; i < visited->iter_args_.size() && i < visited->return_vars_.size(); ++i) {
      const auto& argument = visited->iter_args_[i];
      const auto& return_var = visited->return_vars_[i];
      if (!AsTensorTypeLike(return_var->GetType())) continue;
      auto replacement = return_var;
      if (auto type = TensorBoundaryType(return_var->GetType(), argument)) {
        replacement = std::make_shared<Var>(return_var->name_hint_, type, return_var->span_);
        subst_[op->return_vars_[i]] = replacement;
        result->return_vars_[i] = replacement;
        changed = true;
      }
      if (IsUnsettled(argument)) {
        unsettled_.insert(op->return_vars_[i].get());
        unsettled_.insert(replacement.get());
      }
    }
    // Only the loop boundary aliases its initializer. Independently produced
    // yielded tensors keep their own storage and are never retargeted here.
    return changed ? result : visited;
  }

  /// True when `var` denotes storage whose buffer YieldFixup has not chosen yet —
  /// an IfStmt return_var seeded into `unsettled_`, or any required alias of one.
  [[nodiscard]] bool IsUnsettled(const VarPtr& var) const {
    return var != nullptr && unsettled_.count(var.get()) != 0;
  }

  /// Skips a reused input listed in `unsettled_` — an IfStmt return_var, or a
  /// alias of one, on the pre-YieldFixup run. A phi is not a tracked def in the
  /// reuse analysis (see
  /// LifetimeAnalyzer's IfStmt handler), so it still carries its pre-reuse buffer
  /// here and YieldFixup chooses the real one in Step 4.  Anchoring onto the
  /// stale buffer pins this producer to storage that is about to be abandoned:
  /// YieldFixup then reads it as the canonical target for the *next* phi in the
  /// chain and reconciles the other arm with a same-space `tile.move` no target
  /// can lower, while the post-fixup run silently re-anchors the producer back —
  /// leaving the phi buffer unwritten on the taken arm.  Two conditionals over
  /// one in-place tile (what `pl.unroll(2)` around an `if` produces) are enough.
  /// Rebuild a registered writeback result or first-input alias from its current
  /// source. Preserve result shape/dtype/view metadata, including TensorType and
  /// DistributedTensorType writebacks. These are required aliases, not reuse
  /// opportunities; allocating transpose/extract operations are excluded.
  StmtPtr ReanchorAliasingOutput(const AssignStmtPtr& op) {
    auto call = As<Call>(op->value_);
    if (!call || !call->op_) return nullptr;
    auto reuse_idx = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
    const bool is_view = op_predicates::IsBufferAliasingViewOp(call->op_->name_);
    if (!reuse_idx && (!is_view || call->args_.empty())) return nullptr;
    // An explicit writeback contract wins over memory-space inheritance.
    auto in_var = AsVarLike(call->args_[reuse_idx ? *reuse_idx : 0]);
    if (!in_var) return nullptr;
    auto new_in = AsVarLike(VisitExpr(in_var));  // follow prior subst_ renames
    if (!new_in) return nullptr;
    if (IsUnsettled(in_var) || IsUnsettled(new_in)) {
      // A view or another writeback cannot hide an unresolved phi from the
      // next writer in the chain. YieldFixup must choose that storage first.
      unsettled_.insert(op->var_.get());
      return nullptr;
    }
    auto lhs = std::dynamic_pointer_cast<const ShapedType>(op->var_->GetType());
    auto input = std::dynamic_pointer_cast<const ShapedType>(new_in->GetType());
    if (!lhs || !input || !lhs->memref_ || !input->memref_) return nullptr;
    auto target = *input->memref_;
    if (!reuse_idx) {
      // Derive only the view-relative offset. A writeback's transfer offset
      // selects what it writes; its result still denotes the whole destination.
      auto relative = ComputeViewByteOffset(call, new_in->GetType());
      target = MakeAliasingMemRef(target, relative, (*lhs->memref_)->size_, (*lhs->memref_)->span_);
    }
    if (SamePhysicalWindow(*lhs->memref_, target) && lhs->GetMemorySpace() == input->GetMemorySpace()) {
      return nullptr;
    }
    auto new_type = CloneTypeWithMemRef(op->var_->GetType(), target, input->GetMemorySpace());
    auto new_lhs = std::make_shared<Var>(op->var_->name_hint_, new_type, op->var_->span_);
    subst_[op->var_] = new_lhs;
    auto recursed = As<AssignStmt>(IRMutator::VisitStmt_(op));
    INTERNAL_CHECK_SPAN(recursed, op->span_) << "Internal error: AssignStmt visit must yield an AssignStmt";
    return std::make_shared<AssignStmt>(new_lhs, recursed->value_, recursed->span_);
  }

  std::map<VarPtr, ExprPtr> subst_;
  std::set<const Var*> unsettled_;
};

/// Every IfStmt return_var (phi) reachable in a body.  These are the vars whose
/// buffer YieldFixup, not the reuse analysis, decides.
class IfPhiReturnVarCollector : public IRVisitor {
 public:
  std::set<const Var*> phis;

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& rv : op->return_vars_) phis.insert(rv.get());
    IRVisitor::VisitStmt_(op);
  }
};

}  // namespace

// Remove unused alloc statements from a SeqStmts body
StmtPtr RemoveUnusedAllocStatements(const StmtPtr& body, const std::set<const Var*>& used_bases) {
  auto seq = As<SeqStmts>(body);
  if (!seq) return body;

  std::vector<StmtPtr> new_seq_stmts;
  bool changed = false;

  for (const auto& child : seq->stmts_) {
    if (IsUnusedAllocStmt(child, used_bases)) {
      changed = true;
      continue;
    }
    new_seq_stmts.push_back(child);
  }

  if (!changed) return body;
  return SeqStmts::Flatten(std::move(new_seq_stmts), body->span_);
}

/// The pre-YieldFixup half of the identity-copy normalization bracket: identical
/// to the plain run except that an in-place producer is never anchored onto a phi
/// whose buffer YieldFixup has yet to choose (`ReanchorAliasingOutput`).
StmtPtr NormalizeIdentityCopyBuffersBeforeYieldFixup(const StmtPtr& body) {
  IfPhiReturnVarCollector collector;
  collector.VisitStmt(body);
  return NormalizeIdentityCopyBuffersMutator(std::move(collector.phis)).VisitStmt(body);
}

StmtPtr AlignLoopCarriesToInit(const StmtPtr& body) {
  return AlignLoopCarriesToInitMutator().VisitStmt(body);
}

StmtPtr NormalizeIdentityCopyBuffers(const StmtPtr& body) {
  return NormalizeIdentityCopyBuffersMutator().VisitStmt(body);
}

StmtPtr FixupYields(const StmtPtr& body, const std::vector<VarPtr>& parameters, bool fixup_if_stmts,
                    bool canonical_branch_storage) {
  YieldFixupMutator fixup(fixup_if_stmts, canonical_branch_storage);
  auto result = fixup.Run(body, parameters);
  return InsertAllocsIntoBody(result, fixup.TakePendingAllocs());
}

StmtPtr MaterializeStorageSelections(const StmtPtr& body, const std::vector<VarPtr>& parameters,
                                     StorageSelectionPlan selections) {
  if (selections.tiles.empty()) return body;
  auto allocations = std::move(selections.allocations);
  auto rewritten = YieldFixupMutator(std::move(selections)).Run(body, parameters);
  rewritten = InsertAllocsIntoBody(rewritten, allocations);
  return RemoveUnusedAllocStatements(rewritten, memref_collectors::CollectUsedBasePtrs(rewritten));
}

StmtPtr StripPipelineMembership(const StmtPtr& body) {
  return StripPipelineMembershipMutator().VisitStmt(body);
}

}  // namespace memory_reuse
}  // namespace ir
}  // namespace pypto

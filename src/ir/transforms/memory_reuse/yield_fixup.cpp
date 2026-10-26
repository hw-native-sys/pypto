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
#include <climits>
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
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/buffer_view_analysis.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"
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

// A Vec view retains its parent's physical pitch. TMOV requires equal physical
// shapes after subview resolution, whereas TEXTRACT explicitly repacks the
// requested window into the already planned destination. Index the final SSA
// body so retargeting and carry repair cannot leave stale view provenance.
class StorageTransferLegalizer : public IRMutator {
 public:
  StorageTransferLegalizer(const StmtPtr& body, const std::vector<VarPtr>& parameters)
      : views_(body), geometry_(views_, names_) {
    names_.Collect(body, parameters);
  }

 protected:
  ExprPtr VisitExpr_(const IterArgPtr& argument) override { return argument; }

  StmtPtr VisitStmt_(const AssignStmtPtr& assign) override {
    const auto call = As<Call>(assign->value_);
    if (call && IsOp(call, "tile.slice")) return geometry_.Index(assign, call);
    if (!call || !IsOp(call, "tile.move") || call->args_.size() != 1) return assign;
    const auto source = GetTileTypeWithMemRef(call->args_[0]->GetType());
    const auto target = GetTileTypeWithMemRef(assign->var_->GetType());
    const auto view = views_.Get(call->args_[0]);
    const auto root = view.root ? GetTileTypeWithMemRef(view.root->GetType()) : nullptr;
    if (!source || !target || source->GetMemorySpace() != MemorySpace::Vec ||
        target->GetMemorySpace() != MemorySpace::Vec || !view.window || !root || source->shape_.size() != 2 ||
        root->shape_.size() != 2 || SamePhysicalWindow(GetDefinedMemRef(source), GetDefinedMemRef(target))) {
      return assign;
    }
    // A post-placement repair cannot introduce a wider read that aliases the
    // destination. Preplanning transfers expose this root to no-alias analysis.
    if (MemRef::MayAlias(GetDefinedMemRef(root), GetDefinedMemRef(target))) return assign;
    const auto offsets = geometry_.RootOffsets(view, call->span_);
    auto& registry = OpRegistry::GetInstance();
    auto extract = MutableCopy(registry.Create(
        "tile.extract",
        {view.root, offsets[0], offsets[1], std::make_shared<MakeTuple>(source->shape_, call->span_)},
        {{"target_memory", MemorySpace::Vec}}, call->span_));
    // Keep scheduling metadata; argument directions depend on the new arity.
    for (const auto& attribute : call->attrs_) {
      if (attribute.first != kAttrArgDirections) extract->attrs_.push_back(attribute);
    }
    const auto expected = CloneTypeWithoutMemRef(target);
    if (structural_equal(CloneTypeWithoutMemRef(extract->GetType()), expected)) {
      return std::make_shared<AssignStmt>(assign->var_, extract, assign->span_);
    }

    // A slice/set_validshape may narrow validity beyond the root's window.
    // Preserve that logical requirement without changing the extraction pitch
    // or inventing a differently laid-out transfer.
    const auto extracted = std::make_shared<Var>(
        names_.Fresh("storage_window_"), CloneTypeWithMemRef(extract->GetType(), GetDefinedMemRef(target)),
        assign->span_);
    const auto valid = tile_view_semantics::GetEffectiveTileView(*target).valid_shape;
    const auto refine =
        registry.Create("tile.set_validshape", {extracted, valid[0], valid[1]}, {}, assign->span_);
    if (!structural_equal(CloneTypeWithoutMemRef(refine->GetType()), expected)) return assign;
    return std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::make_shared<AssignStmt>(extracted, extract, assign->span_),
                             std::make_shared<AssignStmt>(assign->var_, refine, assign->span_)},
        assign->span_);
  }

 private:
  BufferViewAnalysis views_;
  StorageNameSupply names_;
  BufferWindowGeometry geometry_;
};

/**
 * @brief Fix yield/return_var MemRef mismatch after MemoryReuse.
 *
 * Handles ForStmt, WhileStmt and IfStmt:
 * - Loops: MemoryReuse may assign different MemRefs to iter_arg and yield value.
 *   Since yield value becomes the next iteration's iter_arg, they must use the
 *   same buffer. When they differ, insert a tile.move before yield, except that
 *   divergent Acc buffers are rejected because Acc->Acc moves are unsupported.
 * - IfStmt: each result already has its canonical destination. Reconcile both
 *   incoming edges with that window; reject an Acc-to-Acc copy only if a
 *   producer could not be retargeted and an actual transfer remains necessary.
 *   Scope::IfResults traverses loops while leaving their carry scheduling to
 *   the later planner-specific reconciliation stage.
 */
class YieldFixupMutator : public StorageBoundaryMutator {
 public:
  enum class Scope { AllControlFlow, IfResults };

  YieldFixupMutator() = default;

  StmtPtr Run(const StmtPtr& body, const std::vector<VarPtr>& parameters) {
    StmtPtr aligned = body;
    if (scope_ == Scope::AllControlFlow) {
      aligned = RefreshStorageTypes(aligned);
    }
    // Carry alignment can move a result; propagate its required aliases before
    // reconciling transfers while keeping unresolved if results provisional.
    // An if-only run must leave loop storage to its later legalization stage.
    aligned = NormalizeIdentityCopyBuffersBeforeYieldFixup(aligned);
    class TileRegions : public IRVisitor {
     public:
      std::set<const Stmt*> regions;

      void VisitStmt(const StmtPtr& statement) override {
        const bool outer = contains_tile_;
        contains_tile_ = false;
        IRVisitor::VisitStmt(statement);
        if (contains_tile_) regions.insert(statement.get());
        contains_tile_ = contains_tile_ || outer;
      }

      void VisitExpr(const ExprPtr& expression) override {
        if (As<TileType>(expression->GetType())) contains_tile_ = true;
        // Types suffice for variable occurrences. Do not repeatedly expand
        // shared IterArg initializer/type chains while indexing statements.
        if (!AsVarLike(expression)) IRVisitor::VisitExpr(expression);
      }

     private:
      bool contains_tile_ = false;
    } tile_regions;
    tile_regions.VisitStmt(aligned);
    tile_regions_ = std::move(tile_regions.regions);
    names_.Collect(aligned, parameters);
    const auto rewritten = VisitStmt(aligned);
    return StorageTransferLegalizer(rewritten, parameters).VisitStmt(rewritten);
  }

  /// Region targets are already planned. This stage only realizes edge
  /// transfers; legacy PYPTO retains its established loop-copy scheduler.
  explicit YieldFixupMutator(bool canonical_loop_copies, Scope scope = Scope::AllControlFlow)
      : canonical_loop_copies_(canonical_loop_copies), scope_(scope) {}

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Dead Tile bodies require no storage repair, including a nested Acc phi
    // inside a scalar-only loop during the if-only phase. Tensor-only bodies
    // retain their dedicated boundary representation.
    if (!LoopMayExecute(op, false) && tile_regions_.count(op.get())) {
      return IRMutator::VisitStmt_(As<ForStmt>(PruneDeadTileBody(op)));
    }
    // First recurse into nested control flow
    auto result = IRMutator::VisitStmt_(op);
    auto for_stmt = As<ForStmt>(result);
    if (scope_ == Scope::IfResults) return result;
    if (!for_stmt || for_stmt->iter_args_.empty()) return result;
    if (canonical_loop_copies_) return FixupCanonicalLoop(for_stmt);

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
      // A reference carry passes its handle on the backedge. Its initializer
      // owns storage, but is not a fixed destination for later iterations.
      const auto argument_type = As<TileType>(for_stmt->iter_args_[i]->GetType());
      if (argument_type && !argument_type->memref_) continue;
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
    if (!LoopMayExecute(op, false) && tile_regions_.count(op.get())) {
      return IRMutator::VisitStmt_(As<WhileStmt>(PruneDeadTileBody(op)));
    }
    auto result = IRMutator::VisitStmt_(op);
    if (scope_ == Scope::IfResults) return result;
    return FixupCanonicalLoop(As<WhileStmt>(result));
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto result = As<IfStmt>(IRMutator::VisitStmt_(op));
    return result && !result->return_vars_.empty() ? FixupCanonicalIf(result) : result;
  }

 public:
  /// Allocations this fixup had to create (cycle spills), for the caller to
  /// hoist onto the function-body head. Empty for every loop whose carries do
  /// not form a copy cycle, which is the overwhelmingly common case.
  std::vector<StmtPtr> TakePendingAllocs() { return std::move(pending_allocs_); }

 private:
  std::set<const Stmt*> tile_regions_;

  template <typename LoopPtr>
  static StmtPtr PruneDeadTileBody(const LoopPtr& loop) {
    auto result = MutableCopy(loop);
    std::vector<ExprPtr> values(loop->iter_args_.begin(), loop->iter_args_.end());
    result->body_ = std::make_shared<YieldStmt>(std::move(values), loop->span_);
    return result;
  }
  bool canonical_loop_copies_ = false;
  Scope scope_ = Scope::AllControlFlow;
  StorageNameSupply names_;

  enum class BoundaryKind { IfResult, LoopCarry };

  // Each phi's target is established before planning. Never
  // choose a branch input as a new target here: it may remain independently live
  // after the IfStmt. Both arms must explicitly write the declared destination.
  StmtPtr FixupCanonicalIf(const IfStmtPtr& branch) {
    auto then_body = FixupCanonicalBoundary(branch->then_body_, branch->return_vars_, branch->span_,
                                            BoundaryKind::IfResult);
    auto else_body = branch->else_body_;
    if (else_body) {
      else_body =
          FixupCanonicalBoundary(*else_body, branch->return_vars_, branch->span_, BoundaryKind::IfResult);
    }
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
    auto body = FixupCanonicalBoundary(loop->body_, targets, loop->span_, BoundaryKind::LoopCarry);
    if (body == loop->body_) return loop;
    auto result = MutableCopy(loop);
    result->body_ = std::move(body);
    return result;
  }

  StmtPtr FixupCanonicalBoundary(const StmtPtr& body, const std::vector<VarPtr>& targets, const Span& span,
                                 BoundaryKind kind) {
    auto yield = transform_utils::GetLastYieldStmt(body);
    std::vector<ExprPtr> values = yield ? yield->value_ : std::vector<ExprPtr>{};
    std::vector<MemRefPtr> destinations(targets.size());
    std::vector<VarPtr> sources(targets.size());
    std::vector<size_t> owners(targets.size());
    std::vector<CarryCopy> copies;
    for (size_t i = 0; i < targets.size(); ++i) {
      if (const auto tile = GetTileTypeWithMemRef(targets[i]->GetType())) {
        destinations[i] = GetDefinedMemRef(tile);
      }
    }
    const CarryRangeIndex index(destinations);
    for (size_t i = 0; i < targets.size(); ++i) {
      owners[i] = i;
      auto target_tile = GetTileTypeWithMemRef(targets[i]->GetType());
      if (!target_tile) continue;
      CHECK_SPAN(yield && i < values.size(), span)
          << "Buffer IR storage legalization requires every region to yield every tile result";
      auto source = AsVarLike(values[i]);
      auto source_tile = source ? GetTileTypeWithMemRef(source->GetType()) : nullptr;
      sources[i] = source;
      CHECK_SPAN(source && As<TileType>(source->GetType()), span)
          << "Buffer IR storage legalization requires tile yields to be storage variables";
      if (!source_tile) {
        // Legacy tile handles can be supplied without a physical MemRef (for
        // example cross-core inputs). Their edge still targets the declared r;
        // codegen realizes that transfer once it resolves the opaque handle.
        // Selected Buffer IR requires explicit source storage and cannot defer.
        const auto* context = PassContext::Current();
        CHECK_SPAN(context == nullptr || !context->GetEnableBufferIR(), span)
            << "Buffer IR storage legalization requires known storage for tile yield inputs";
        continue;
      }
      sources[i] = source;
      auto target = GetDefinedMemRef(target_tile);
      destinations[i] = target;
      auto source_memref = GetDefinedMemRef(source_tile);
      if (SamePhysicalWindow(source_memref, target)) continue;
      CHECK_SPAN(CompareBaseAddress(source_memref, target) == AddressRelation::kDifferent, span)
          << "Buffer IR storage legalization cannot reconcile ambiguous views of one destination";
      const auto source_memory = source_tile->GetMemorySpace();
      const auto target_memory = target_tile->GetMemorySpace();
      const bool supported =
          source_memory && target_memory && IsTileMoveEverSupported(*source_memory, *target_memory);
      CHECK_SPAN(kind != BoundaryKind::LoopCarry || supported || !index.AnyOverlap(source_memref), span)
          << "Buffer IR cannot reconcile region storage with an unsupported tile.move memory-space pair";
      // Constructing an actual copy preserves the accumulator invariant's
      // existing InternalError; fresh Acc producers were allowed to retarget
      // earlier without requiring a transfer at all.
      auto [moved, move] = CreateTileMove(source, target, target_memory);
      CHECK_SPAN(supported, span)
          << "Buffer IR cannot reconcile region storage with an unsupported tile.move memory-space pair";
      copies.push_back({i, source, moved, move, source_memref, target, target_tile->GetMemorySpace()});
    }
    if (copies.empty()) return body;
    std::vector<bool> written_slots(targets.size(), false);
    for (const auto& copy : copies) written_slots[copy.index] = true;
    // Multiple SSA results may return the same value through one window. Merge
    // only equivalent transfers, including their descriptors; equal destinations
    // alone would silently overwrite distinct results. Mutable loop carries
    // retain their independent-slot contract.
    const auto overlaps =
        kind == BoundaryKind::IfResult ? index.OverlappingPairs(written_slots) : index.OverlappingPairs();
    for (const auto& [lhs, rhs] : overlaps) {
      auto lhs_source = GetTileTypeWithMemRef(sources[lhs]->GetType());
      auto rhs_source = GetTileTypeWithMemRef(sources[rhs]->GetType());
      CHECK_SPAN(lhs_source && rhs_source && kind == BoundaryKind::IfResult &&
                     SamePhysicalWindow(destinations[lhs], destinations[rhs]) &&
                     SamePhysicalWindow(GetDefinedMemRef(lhs_source), GetDefinedMemRef(rhs_source)) &&
                     SameTransferDescriptor(lhs_source, rhs_source) &&
                     SameTransferDescriptor(GetTileTypeWithMemRef(targets[lhs]->GetType()),
                                            GetTileTypeWithMemRef(targets[rhs]->GetType())),
                 span)
          << "Buffer IR region transfers require independent destination windows";
      UnionSets(owners, lhs, rhs);
    }
    for (size_t i = 0; i < owners.size(); ++i) owners[i] = FindSet(owners, i);
    copies.erase(
        std::remove_if(copies.begin(), copies.end(),
                       [&owners](const CarryCopy& copy) { return owners[copy.index] != copy.index; }),
        copies.end());
    std::vector<MemRefPtr> written_destinations(targets.size());
    for (const auto& copy : copies) written_destinations[copy.index] = copy.dst_memref;
    const CarryRangeIndex writes(written_destinations);
    // Snapshot each source that another transfer may overwrite. This deliberately
    // uses overlap existence queries, not an O(k^2) graph for broad/fanout views.
    // Every snapshot precedes every destination write: swaps, cycles, partial
    // overlap and fanout therefore share one O(k log k) algorithm.
    std::vector<StmtPtr> moves;
    for (auto& copy : copies) {
      if (writes.AnyOverlap(copy.src_memref)) SpillCarrySource(&copy, &moves);
    }
    std::vector<VarPtr> moved_values(targets.size());
    for (const auto& copy : copies) {
      moves.push_back(copy.move_stmt);
      moved_values[copy.index] = copy.moved_var;
    }
    // Spilling may replace the representative's moved var. Publish aliases only
    // after the complete copy plan has settled, so every slot sees that final var.
    for (size_t i = 0; i < owners.size(); ++i) {
      if (const auto& moved = moved_values[owners[i]]) values[i] = moved;
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
  /// Within an allocation the constant-offset ranges are sorted by start.
  /// Prefix maxima provide linear overlap witnesses even when only a subset
  /// of the destinations will be written. Existence queries use binary search.
  /// A range whose offset is not constant
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
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
          return std::tie(a.begin, a.end, a.index) < std::tie(b.begin, b.end, b.index);
        });
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

    /// Linear overlap witnesses, as (lower index, higher index). An empty mask
    /// marks every slot active; otherwise each witness touches an active slot.
    /// Adjacent equal windows stay connected, and prefix maxima catch an active
    /// window hidden behind smaller inactive windows inside an enclosing range.
    [[nodiscard]] std::vector<std::pair<size_t, size_t>> OverlappingPairs(
        const std::vector<bool>& active = {}) const {
      std::vector<std::pair<size_t, size_t>> pairs;
      const auto is_active = [&active](size_t index) { return active.empty() || active[index]; };
      const auto append = [&pairs, &is_active](size_t lhs, size_t rhs) {
        if (lhs == rhs || (!is_active(lhs) && !is_active(rhs))) return;
        const std::pair<size_t, size_t> pair = std::minmax(lhs, rhs);
        if (pairs.empty() || pairs.back() != pair) pairs.push_back(pair);
      };
      for (const auto& [base, entries] : positioned_) {
        std::optional<size_t> widest;
        std::optional<size_t> widest_active;
        std::optional<size_t> first_active;
        for (size_t n = 0; n < entries.size(); ++n) {
          const auto& entry = entries[n];
          const bool current_active = is_active(entry.index);
          if (n > 0 && entry.begin < entries[n - 1].end) append(entries[n - 1].index, entry.index);
          const auto previous = current_active ? widest : widest_active;
          if (previous && entry.begin < entries[*previous].end) append(entries[*previous].index, entry.index);
          if (!widest || entry.end > entries[*widest].end) widest = n;
          if (current_active) {
            if (!widest_active || entry.end > entries[*widest_active].end) widest_active = n;
            if (!first_active) first_active = entry.index;
          }
        }
        // Symbolic offsets may touch every positioned window. One active
        // cross-group witness suffices to reject their unproven separation.
        const auto loose = unpositioned_.find(base);
        if (loose != unpositioned_.end() && !entries.empty()) {
          append(loose->second.front(), entries.front().index);
          if (first_active) append(loose->second.front(), *first_active);
          const auto active_loose = std::find_if(loose->second.begin(), loose->second.end(), is_active);
          if (active_loose != loose->second.end()) append(*active_loose, entries.front().index);
        }
      }
      for (const auto& [base, loose] : unpositioned_) {
        static_cast<void>(base);
        const auto representative = std::find_if(loose.begin(), loose.end(), is_active);
        for (size_t i = 1; i < loose.size(); ++i) {
          append(loose[i - 1], loose[i]);
        }
        if (representative != loose.end()) {
          for (size_t index : loose) append(*representative, index);
        }
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
      const auto argument_type = As<TileType>(for_stmt->iter_args_[i]->GetType());
      if (argument_type && !argument_type->memref_) continue;
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
    if (canonical_loop_copies_) {
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
        canonical_loop_copies_ ? names_.Fresh("mem_" + space_str + "_carry_spill_")
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
        canonical_loop_copies_ ? names_.Fresh("storage_transfer_") : source->name_hint_ + "_mv");
  }

  // Patch iter_args and return_vars to share initValue's MemRef.
  // Returns the original ForStmt if no patching is needed.
  StmtPtr PatchIterArgsAndReturnVars(const ForStmtPtr& for_stmt, const YieldStmtPtr& yield_stmt) {
    bool changed = false;
    std::vector<IterArgPtr> new_iter_args = for_stmt->iter_args_;
    std::vector<VarPtr> new_return_vars = for_stmt->return_vars_;

    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      const auto argument_type = As<TileType>(for_stmt->iter_args_[i]->GetType());
      if (argument_type && !argument_type->memref_) continue;
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

// Remove unused alloc statements from a SeqStmts body
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
/// rename — `x` must alias `y`'s buffer.  Accumulator target planning retargets an
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
    if (!AsTensorTypeLike(visited->GetType())) {
      var_remap_[op.get()] = visited;
      return visited;
    }
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

  /// Before branch destinations are established, aliases of a provisional
  /// phi must stay unsettled too. Otherwise a downstream writeback could pin
  /// itself to the temporary buffer inherited from one arm. Once destinations
  /// are established, rebuild required aliases from their current input while
  /// preserving shape/dtype/view metadata. Allocating view operations are excluded.
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

/// Every IfStmt result reachable in a body. During pre-boundary normalization,
/// these can still carry a provisional input buffer.
class IfPhiReturnVarCollector : public IRVisitor {
 public:
  std::set<const Var*> phis;
  // Only statement definitions can introduce a join. Expanding each
  // IterArg's initializer here repeats shared carry chains.
  void VisitExpr(const ExprPtr&) override {}

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& rv : op->return_vars_) phis.insert(rv.get());
    IRVisitor::VisitStmt_(op);
  }
};

/// Copy coalescing removes the operation instead of executing an in-place TMOV.
/// Exact descriptor equality makes source substitution preserve the type/view.
class RemoveIdentityTransfers : public IRMutator {
 public:
  explicit RemoveIdentityTransfers(const std::set<const Var*>& fixed_bases) : fixed_bases_(fixed_bases) {}

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto result = As<AssignStmt>(IRMutator::VisitStmt_(op));
    auto call = As<Call>(result->value_);
    if (!IsOp(call, "tile.move") || call->args_.size() != 1) return result;
    auto source = GetTileTypeWithMemRef(call->args_[0]->GetType());
    auto target = GetTileTypeWithMemRef(result->var_->GetType());
    if (!SameTransferDescriptor(source, target) ||
        !SamePhysicalWindow(GetDefinedMemRef(source), GetDefinedMemRef(target))) {
      return result;
    }
    // Fixed storage never participates in optional copy coalescing. Preserve
    // its declared operations so their address restrictions remain validated.
    if (fixed_bases_.count(GetDefinedMemRef(source)->base_.get())) return result;
    if (!AsVarLike(call->args_[0])) return result;
    var_remap_[op->var_.get()] = call->args_[0];
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
  }

 private:
  const std::set<const Var*>& fixed_bases_;
};

}  // namespace

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

/// Strip the transient ``pipeline_membership`` attr from every Call once
/// MemoryReuse has consumed it (in ComputeLifetimes). The attr exists only to
/// carry pipeline-stage identity from LowerPipelineLoops to here; leaving it on
/// the IR would ride downstream into later passes and codegen. Stripping at the
/// end of MemoryReuse keeps the post-reuse IR clean.
/// The pre-YieldFixup half of the identity-copy normalization bracket: identical
/// to the plain run except that an in-place producer is never anchored onto a phi
/// whose buffer YieldFixup has yet to choose (`ReanchorAliasingOutput`).
StmtPtr NormalizeIdentityCopyBuffersBeforeYieldFixup(const StmtPtr& body) {
  IfPhiReturnVarCollector collector;
  collector.VisitStmt(body);
  return NormalizeIdentityCopyBuffersMutator(std::move(collector.phis)).VisitStmt(body);
}

/// Whole-root, same-descriptor copies can become SSA identities when placement
/// coalesces their storage. Remaining copies must keep disjoint byte windows.
TransferCoalescing CollectCoalescibleTransfers(const StmtPtr& body, const LifetimeAnalysisResult& analysis,
                                               const std::vector<VarPtr>& parameters) {
  class Collector : public IRVisitor {
   public:
    Collector(const LifetimeAnalysisResult& analysis, const std::vector<VarPtr>& parameters) {
      // Carries and region results are not necessarily sharing-group members.
      // Resolve every handle through its allocation identity instead of
      // treating a missing variable entry as proof that it is the whole root.
      for (const auto& interval : analysis.lifetimes) {
        if (const auto memref = GetTypeMemRef(interval.variable->GetType()); memref && *memref) {
          representatives_.emplace((*memref)->base_.get(), interval.variable);
        }
      }
      // This planner tracks body definitions. Preserve the existing no-alias
      // protection for incoming storage, including aliases of Tile parameters.
      for (const auto& parameter : parameters) {
        if (auto tile = GetTileTypeWithMemRef(parameter->GetType())) {
          parameter_bases_.insert(GetDefinedMemRef(tile)->base_.get());
        }
      }
    }
    TransferCoalescing transfers;

    // Expressions cannot contain assignments; do not expand carry initializers.
    void VisitExpr(const ExprPtr&) override {}

    void VisitStmt_(const AssignStmtPtr& assign) override {
      auto call = As<Call>(assign->value_);
      if (!IsOp(call, "tile.move") || call->args_.size() != 1) return;
      auto input = AsVarLike(call->args_[0]);
      auto source = input ? GetTileTypeWithMemRef(input->GetType()) : nullptr;
      auto output = GetTileTypeWithMemRef(assign->var_->GetType());
      if (!SameTransferDescriptor(source, output)) return;
      auto source_memref = GetDefinedMemRef(source);
      auto output_memref = GetDefinedMemRef(output);
      if (parameter_bases_.count(source_memref->base_.get()) ||
          parameter_bases_.count(output_memref->base_.get())) {
        return;
      }
      const bool same_slot =
          source_memref->slot_count_ == output_memref->slot_count_ &&
          source_memref->slot_index_.has_value() == output_memref->slot_index_.has_value() &&
          (!source_memref->slot_index_ ||
           structural_equal(*source_memref->slot_index_, *output_memref->slot_index_));
      auto source_root = Representative(input);
      auto output_root = Representative(assign->var_);
      if (!source_root || !output_root) return;
      if (same_slot && source_memref->size_ == output_memref->size_ &&
          SamePhysicalWindow(GetTypeMemRef(source_root->GetType()).value_or(nullptr), source_memref) &&
          SamePhysicalWindow(GetTypeMemRef(output_root->GetType()).value_or(nullptr), output_memref)) {
        transfers.definitions.insert(assign->var_.get());
        transfers.subrange_separation[output_root.get()].push_back(input);
      }
    }

   private:
    std::map<const Var*, VarPtr> representatives_;
    std::set<const Var*> parameter_bases_;
    [[nodiscard]] VarPtr Representative(const VarPtr& value) const {
      const auto memref = GetTypeMemRef(value->GetType());
      if (!memref || !*memref) return nullptr;
      const auto found = representatives_.find((*memref)->base_.get());
      return found == representatives_.end() ? nullptr : found->second;
    }
  } collector(analysis, parameters);
  collector.VisitStmt(body);
  return std::move(collector.transfers);
}

StmtPtr NormalizeIdentityCopyBuffers(const StmtPtr& body) {
  return NormalizeIdentityCopyBuffersMutator().VisitStmt(body);
}

StmtPtr FixupYields(const StmtPtr& body, const std::vector<VarPtr>& parameters, bool canonical_loop_copies,
                    bool if_results_only) {
  YieldFixupMutator fixup(canonical_loop_copies, if_results_only ? YieldFixupMutator::Scope::IfResults
                                                                 : YieldFixupMutator::Scope::AllControlFlow);
  auto result = fixup.Run(body, parameters);
  return InsertAllocsIntoBody(result, fixup.TakePendingAllocs());
}

StmtPtr RemoveIdentityStorageTransfers(const StmtPtr& body, const std::set<const Var*>& fixed_bases) {
  return RemoveIdentityTransfers(fixed_bases).VisitStmt(body);
}

StmtPtr StripPipelineMembership(const StmtPtr& body) {
  return StripPipelineMembershipMutator().VisitStmt(body);
}

}  // namespace memory_reuse
}  // namespace ir
}  // namespace pypto

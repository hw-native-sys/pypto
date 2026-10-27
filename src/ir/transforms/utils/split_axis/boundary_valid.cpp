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

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/split_axis_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/utils/split_axis/internal.h"

namespace pypto {
namespace ir {
namespace split_axis {

using detail::IsOddSplitExtent;
using detail::WithValidDim;

namespace {

// Whether @p call fills a tile's PAD region, i.e. reads where the operand's valid
// data ends in order to define everything past it. Such an op is the one kind of
// consumer that cannot receive a deferred extent: its result is fully valid by
// construction, so the extent has nowhere to land, and it would fill nothing.
// `tile.set_validshape` to the full box looks the same in the type system but is
// a relabel, not a fill — the author is declaring the padding to be data, which
// is exactly what a deferred boundary hands them.
bool FillsPadRegion(const CallPtr& call) {
  return IsOp(call, "tile.fillpad") || IsOp(call, "tile.fillpad_inplace") ||
         IsOp(call, "tile.fillpad_expand");
}

// A store must not read a boundary tile whose split-axis extent was left at the
// transport's (see WithFullSplitAxisValid): that extent is the truth about the
// FIFO band and a lie about how much of the lane's box is data.
void RejectDeferredValidStore(bool deferred, const CallPtr& call) {
  CHECK_SPAN(!deferred, call->span_)
      << "tile.store: this store consumes a Cube -> Vector boundary tile directly, and that "
         "boundary's split-axis valid extent is a RUNTIME value, so the two AIV lanes' extents "
         "are not known at compile time. The popped tile therefore has to declare the transport's "
         "full box — that is what places lane 1's band where the producer wrote it — rather than "
         "the lane's own extent, and storing it would write the lane's padding as data.\n"
         "Author one of these instead:\n"
         "  * narrow after the crossing: run a vector op whose result carries the extent you want "
         "stored (e.g. pl.fillpad(...) to define the padding, then pl.set_validshape(...)), and "
         "store that\n"
         "  * make the split-axis valid extent a compile-time constant, so the lanes can be "
         "balanced and the extent can ride on the transport itself\n"
         "  * keep the extent at or below the box half, so the second lane is empty";
}

// A pad fill of a boundary tile whose split-axis extent was left at the
// transport's box (see WithFullSplitAxisValid) would fill up to the box, not to
// the lane's data, and declare the transport's padding valid. `crossing` names
// the author's crossing in the advice: `pl.aiv_shard(acc)` for an explicit
// region, the first vector op for the AUTO path.
void RejectDeferredPadFill(bool deferred, const CallPtr& call, const char* crossing) {
  CHECK_SPAN(!(deferred && FillsPadRegion(call)), call->span_)
      << call->op_->name_
      << ": this fills the padding of a Cube -> Vector boundary tile whose split-axis valid extent "
         "is a RUNTIME value. The boundary has to declare the transport's full box (that is what "
         "places lane 1's band where the producer wrote it), so by the time the fill runs there is "
         "no per-lane boundary left to fill up to, and it would define nothing.\n"
      << "Author one of these instead:\n"
      << "  * fill the padding on the CUBE side, before the crossing: pl.fillpad(acc) ahead of " << crossing
      << ", so the transported box carries defined padding\n"
      << "  * put the lane's own compute first — any op that passes valid_shape through (a cast, an "
         "elementwise) materializes the lane extent, and a fill after it works normally\n"
      << "  * make the split-axis valid extent a compile-time constant, so it rides the boundary "
         "itself";
}

}  // namespace

namespace {

CallPtr AsCall(const ExprPtr& expr) { return std::dynamic_pointer_cast<const Call>(expr); }

// A consumer PASSES THROUGH the split-axis extent when its result is the same
// physical box as the operand and carries the operand's own (pre-localization)
// valid extents — an elementwise op, a cast, a move. A consumer that reshapes
// the logical rectangle (a reduction, a slice, an extract) reports a different
// valid_shape, and rewriting its split-axis extent to the lane's would be a lie
// about what it computes; those are rejected by the caller instead.
bool PassesThroughValidShape(const std::shared_ptr<const TileType>& operand_before,
                             const std::shared_ptr<const TileType>& result) {
  if (!operand_before || !result) return false;
  if (!tile_view_semantics::ShapeExprListsEquivalent(operand_before->shape_, result->shape_)) {
    return false;
  }
  return tile_view_semantics::ShapeExprListsEquivalent(
      tile_view_semantics::GetEffectiveTileView(*operand_before).valid_shape,
      tile_view_semantics::GetEffectiveTileView(*result).valid_shape);
}

bool IsProvablyPositive(const ExprPtr& extent) {
  auto c = std::dynamic_pointer_cast<const ConstInt>(extent);
  return c != nullptr && c->value_ > 0;
}

// What a tracked (localized) var carries: the per-lane extent now on its split
// axis, plus the type it had BEFORE localization, which is what a consumer's
// deduced valid_shape is compared against to decide pass-through.
struct LocalizedTile {
  ExprPtr lane_extent;                     ///< clamp(V - idx*half, 0, half)
  ExprPtr joined_extent;                   ///< V, the pre-shard extent a gather restores
  std::shared_ptr<const TileType> before;  ///< the type before localization
  /// The extent could NOT ride on the boundary op itself (see
  /// WithFullSplitAxisValid): the shard keeps the full box so the transport
  /// places lane 1's band correctly, and the extent lands on this consumer chain
  /// instead. An op that READS the extent without producing a tile — a store —
  /// has nowhere to put it and is rejected.
  bool deferred = false;
};

}  // namespace

namespace {

/// Mutable state threaded through the (recursive) localization walk.
struct LocalizeState {
  std::unordered_map<const Var*, LocalizedTile> tracked;
  std::unordered_map<const Var*, VarPtr> replacements;
  ExprPtr lane_index;                                  ///< the region's aiv_id, resolved lazily
  std::unordered_set<const Var*> chained_store_dests;  ///< region-wide, indexed once
};

std::vector<StmtPtr> LocalizeStmts(const std::vector<StmtPtr>& stmts, int split_dim, const Span& span,
                                   LocalizeState& state);

/// The tracked (localized or deferred) boundary tile this call reads, or null.
const LocalizedTile* FindTrackedSource(const CallPtr& call, const LocalizeState& state) {
  for (const auto& arg : call->args_) {
    auto arg_var = AsVarLike(arg);
    if (!arg_var) continue;
    auto replaced = state.replacements.find(arg_var.get());
    const Var* key = (replaced != state.replacements.end()) ? replaced->second.get() : arg_var.get();
    auto it = state.tracked.find(key);
    if (it != state.tracked.end()) return &it->second;
  }
  return nullptr;
}

/// `extent > 0`, typed after the extent.
ExprPtr MakeNonEmpty(const ExprPtr& extent, const Span& span) {
  return MakeGt(extent, std::make_shared<ConstInt>(0, GetScalarDtype(extent), span), span);
}

/// Run `store` only on a lane whose split-axis extent is non-zero. A zero-row
/// store is outside pto-isa's contract (see the AssignStmt store arm).
StmtPtr GuardNonEmptyLane(const StmtPtr& store, const ExprPtr& lane_extent, const Span& span) {
  return std::make_shared<IfStmt>(MakeNonEmpty(lane_extent, span), store, std::nullopt, std::vector<VarPtr>{},
                                  span);
}

/// Recurse into a nested body, preserving its statement kind.
StmtPtr LocalizeNestedBody(const StmtPtr& body, int split_dim, const Span& span, LocalizeState& state) {
  if (!body) return body;
  auto inner = LocalizeStmts(transform_utils::FlattenToStmts(body), split_dim, span, state);
  if (inner.size() == 1) return inner[0];
  return std::make_shared<SeqStmts>(inner, body->span_);
}

/// Whether another ``tile.store`` in this body writes THROUGH `var`, i.e. takes
/// it as its destination tensor (arg 2) — a chained store.
///
/// This is the one read that makes the empty-lane guard unsafe, because it
/// orders two stores against each other: skipping the first would leave the
/// second writing through a tensor version that was never produced. Every other
/// read of a store's SSA result is benign. It is a destination-passing alias of
/// a tensor the region does not own, so a phi that carries it out of a branch,
/// or the enclosing function threading it to a return, still denotes the SAME
/// buffer — an empty lane that stored nothing leaves exactly the value any
/// reader must see, and ExpandMixedKernel drops the alias from the AIV half.
/// Indexed ONCE over the whole region, recursively: a chained store nested in a
/// branch or a loop orders against an outer store just as a sibling one does, so
/// a sibling-only scan would miss it and wrongly allow the empty-lane guard.
/// Building the index up front also keeps the walk linear in the region size
/// instead of quadratic in its store count (see `.claude/rules/pass-complexity.md`).
///
/// A store's own destination is an operand, never its SSA result, so a store can
/// never place its own result in this set; no self-exclusion is needed.
void CollectChainedStoreDestinations(const std::vector<StmtPtr>& stmts, std::unordered_set<const Var*>* out) {
  for (const auto& s : stmts) {
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(s)) {
      CollectChainedStoreDestinations(transform_utils::FlattenToStmts(for_stmt->body_), out);
      continue;
    }
    if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(s)) {
      CollectChainedStoreDestinations(transform_utils::FlattenToStmts(while_stmt->body_), out);
      continue;
    }
    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(s)) {
      CollectChainedStoreDestinations(transform_utils::FlattenToStmts(if_stmt->then_body_), out);
      if (if_stmt->else_body_.has_value()) {
        CollectChainedStoreDestinations(transform_utils::FlattenToStmts(*if_stmt->else_body_), out);
      }
      continue;
    }
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(s)) {
      CollectChainedStoreDestinations(seq->stmts_, out);
      continue;
    }
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(s);
    auto call = assign ? AsCall(assign->value_) : nullptr;
    if (!call || !IsOp(call, "tile.store") || call->args_.size() < 3) continue;
    if (auto dest = AsVarLike(call->args_[2])) out->insert(dest.get());
  }
}

/// A loop carry seeded by a localized (per-lane) tile would need its `IterArg`
/// and the paired return var retyped to the lane's extent. Those are separate
/// SSA definitions that this walk does not rebuild, so the carry would keep the
/// deducer's lane-agnostic type while its init value carries the lane's — the
/// silent-wrong-data shape this pass exists to remove. Refuse it explicitly
/// instead, naming the authoring that avoids it.
void RejectLocalizedCarry(const std::vector<IterArgPtr>& iter_args, const char* loop_form, const Span& span,
                          const LocalizeState& state) {
  for (const auto& iter_arg : iter_args) {
    if (!iter_arg) continue;
    auto init = AsVarLike(iter_arg->initValue_);
    if (!init) continue;
    auto replaced = state.replacements.find(init.get());
    const Var* key = (replaced != state.replacements.end()) ? replaced->second.get() : init.get();
    auto it = state.tracked.find(key);
    if (it == state.tracked.end()) continue;
    CHECK_SPAN(false, span)
        << "pl.split_aiv: carrying a per-lane cross-core value across a " << loop_form
        << " boundary is not supported. '" << iter_arg->name_hint_
        << "' is seeded by a value whose split-axis extent differs per lane ("
        << PythonPrint(it->second.lane_extent)
        << "), but a loop carry is a separate binding that would keep the lane-agnostic extent.\n"
        << "Author one of these instead:\n"
        << "  * keep the shard and everything that consumes it inside the loop body\n"
        << "  * make the split axis fully valid before the crossing (pl.set_validshape to the full "
           "extent) and treat the padding as don't-care\n"
        << "  * store the per-lane shard before the loop and reload it inside";
  }
}

std::vector<StmtPtr> LocalizeStmts(const std::vector<StmtPtr>& stmts, int split_dim, const Span& span,
                                   LocalizeState& state) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());

  for (const auto& stmt : stmts) {
    // --- Control flow: the repair must reach as deep as the region does. ------
    // A shard, a consumer or a store nested in a loop or a branch is ordinary
    // authoring (a per-lane tail stored under `if`, a shard inside `pl.range`),
    // and a flat walk would leave it on the deducer's lane-agnostic ceil(V/2).
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto new_body = LocalizeNestedBody(for_stmt->body_, split_dim, span, state);
      RejectLocalizedCarry(for_stmt->iter_args_, "pl.range", for_stmt->span_, state);
      result.push_back(new_body == for_stmt->body_
                           ? stmt
                           : loop_repair::RebuildForStmt(for_stmt, for_stmt->iter_args_, new_body,
                                                         for_stmt->return_vars_));
      continue;
    }
    if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto new_body = LocalizeNestedBody(while_stmt->body_, split_dim, span, state);
      RejectLocalizedCarry(while_stmt->iter_args_, "pl.while_", while_stmt->span_, state);
      if (new_body == while_stmt->body_) {
        result.push_back(stmt);
      } else {
        auto new_while = MutableCopy(while_stmt);
        new_while->body_ = new_body;
        result.push_back(new_while);
      }
      continue;
    }
    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto new_then = LocalizeNestedBody(if_stmt->then_body_, split_dim, span, state);
      std::optional<StmtPtr> new_else = if_stmt->else_body_;
      if (if_stmt->else_body_.has_value()) {
        new_else = LocalizeNestedBody(*if_stmt->else_body_, split_dim, span, state);
      }
      result.push_back(std::make_shared<IfStmt>(if_stmt->condition_, new_then, new_else,
                                                if_stmt->return_vars_, if_stmt->span_));
      continue;
    }
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
      for (auto& s : LocalizeStmts(seq->stmts_, split_dim, span, state)) result.push_back(s);
      continue;
    }

    // A store whose result is ignored is an EvalStmt, so the AssignStmt walk below
    // never sees it — but it reads the boundary tile's extent just the same, and
    // a deferred one would put the transport's padding in the output. Its lane
    // may be empty too, and with no SSA result it can never be a chained store,
    // so it always takes the empty-lane guard (see the AssignStmt arm).
    if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      if (auto eval_call = AsCall(eval->expr_);
          eval_call && eval_call->op_ && IsOp(eval_call, "tile.store")) {
        if (const LocalizedTile* eval_source = FindTrackedSource(eval_call, state)) {
          RejectDeferredValidStore(eval_source->deferred, eval_call);
          if (!IsProvablyPositive(eval_source->lane_extent)) {
            result.push_back(GuardNonEmptyLane(stmt, eval_source->lane_extent, eval_call->span_));
            continue;
          }
        }
      }
      result.push_back(stmt);
      continue;
    }

    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    auto call = assign ? AsCall(assign->value_) : nullptr;
    if (!assign || !call || !call->op_) {
      result.push_back(stmt);
      continue;
    }

    if (IsOp(call, "tile.get_subblock_idx")) {
      state.lane_index = assign->var_;
      result.push_back(stmt);
      continue;
    }

    // --- The boundary itself: repair the deducer's ceil-div guess. -----------
    if (IsOp(call, "tile.aiv_shard") && !call->args_.empty()) {
      auto operand = std::dynamic_pointer_cast<const TileType>(call->args_[0]->GetType());
      auto shard = std::dynamic_pointer_cast<const TileType>(call->GetType());
      if (!operand || !shard || split_dim < 0 || split_dim >= static_cast<int>(shard->shape_.size())) {
        result.push_back(stmt);
        continue;
      }
      const auto operand_valid = tile_view_semantics::GetEffectiveTileView(*operand).valid_shape;
      // A fully-valid EVEN split axis needs no repair: both lanes hold `half`,
      // which is exactly what ReshapeSplitAxis already produced. A fully-valid
      // ODD one does: its lanes hold ceil and floor, the transport picks the odd
      // code for exactly that reason, and pto-isa then reads the per-lane extents
      // off the popped tile. Leaving both lanes at the ceil guess would place
      // lane 1's band one cell too far.
      if (static_cast<int>(operand_valid.size()) <= split_dim ||
          (AreExprsEqual(operand_valid[split_dim], operand->shape_[split_dim]) &&
           !IsOddSplitExtent(operand->shape_[split_dim]))) {
        result.push_back(stmt);
        continue;
      }
      INTERNAL_CHECK_SPAN(state.lane_index, call->span_)
          << "Internal error: a pl.split_aiv region with a partially-valid split axis has no "
             "tile.get_subblock_idx binding, so the per-lane extent cannot be materialized";
      auto localized = LocalizeShardValidForLane(shard, operand, split_dim, state.lane_index);
      auto lane_extent =
          tile_view_semantics::GetEffectiveTileView(*std::dynamic_pointer_cast<const TileType>(localized))
              .valid_shape[split_dim];
      // Where the extent may LAND. It belongs on the shard itself only when the
      // transport was verified to place both lanes' bands, which needs
      // compile-time lane extents: pto-isa reads lane 1's band offset off this
      // very tile, and the producer laid its rows at their box positions. A runtime extent
      // leaves the shard full-width and hands the lane's extent to the consumers
      // below (see WithFullSplitAxisValid).
      const bool deferred = !HasStaticLaneExtents(operand, split_dim, /*lane_stride=*/nullptr);
      auto new_type = deferred ? WithFullSplitAxisValid(shard, split_dim) : localized;
      auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_type, assign->var_->span_);
      auto new_call = std::make_shared<Call>(call->op_, call->args_, call->kwargs_, new_type, call->span_);
      state.replacements[assign->var_.get()] = new_var;
      state.tracked[new_var.get()] = LocalizedTile{lane_extent, operand_valid[split_dim], shard, deferred};
      result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
      continue;
    }

    // --- Consumers: carry the per-lane extent, or say why we cannot. ---------
    const LocalizedTile* source = FindTrackedSource(call, state);
    // tile.aic_gather is the region's EXIT: it re-joins the two lanes' bands and
    // hands a FULL tile back to the cube. This is the one place the join can be
    // typed correctly, because it is the only place the lanes' TRUE extents are
    // known.
    //
    // Under the per-lane clamp the bands always abut, so the joined extent is
    // exactly the pre-shard V:
    //
    //   V >  half : lane 0 saturates to half, lane 1 holds V - half
    //               -> [0, half) U [half, V) = [0, V)
    //   V <= half : lane 0 holds V, lane 1 is empty
    //               -> [0, V) U {} = [0, V)
    //
    // The deducer cannot compute that. It runs before the lanes exist and can
    // only double its own lane-agnostic guess (2 * ceil(V / 2)), which claims
    // the hole between the bands as real data. So restore V here.
    if (IsOp(call, "tile.aic_gather")) {
      auto gathered = std::dynamic_pointer_cast<const TileType>(call->GetType());
      if (source != nullptr && gathered && split_dim < static_cast<int>(gathered->shape_.size())) {
        auto joined_type = WithValidDim(gathered, split_dim, source->joined_extent);
        auto new_var = std::make_shared<Var>(assign->var_->name_hint_, joined_type, assign->var_->span_);
        auto new_call =
            std::make_shared<Call>(call->op_, call->args_, call->kwargs_, joined_type, call->span_);
        state.replacements[assign->var_.get()] = new_var;
        // The result is a FULL tile again, so it leaves the per-lane dataflow.
        result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
        continue;
      }
      // Nothing localized this operand, so a provable partial is one the author
      // gave BOTH lanes: the bands do not abut and no valid_shape describes the
      // union. This is the check the deducer used to make on a value it could
      // not see the truth of.
      auto operand = call->args_.empty()
                         ? nullptr
                         : std::dynamic_pointer_cast<const TileType>(call->args_[0]->GetType());
      if (operand && split_dim < static_cast<int>(operand->shape_.size())) {
        const auto operand_valid = tile_view_semantics::GetEffectiveTileView(*operand).valid_shape;
        auto valid_const = split_dim < static_cast<int>(operand_valid.size())
                               ? std::dynamic_pointer_cast<const ConstInt>(operand_valid[split_dim])
                               : nullptr;
        auto half_const = std::dynamic_pointer_cast<const ConstInt>(operand->shape_[split_dim]);
        const bool provably_partial = valid_const && half_const && valid_const->value_ < half_const->value_;
        const char* axis_name = (split_dim == 0) ? "row" : "column";
        CHECK_SPAN(!provably_partial, call->span_)
            << "pl.split_aiv: tile.aic_gather re-joins the two lanes positionally, but each lane's "
               "valid "
            << axis_name << " extent (" << valid_const->value_ << " of " << half_const->value_
            << ") covers only part of its half, so the lanes contribute the disjoint bands [0, "
            << valid_const->value_ << ") and [" << half_const->value_ << ", "
            << (half_const->value_ + valid_const->value_)
            << ") of the re-joined tile. No valid_shape describes that union.\n"
            << "Author one of these instead:\n"
            << "  * make each lane's half fully valid before the gather -- pl.fillpad(v) to fill the "
               "padding with data, or pl.set_validshape(v, <full half extent>, ...) to declare it "
               "data -- so the bands abut\n"
            << "  * let the shard's own per-lane extent flow into the gather instead of overriding "
               "it: a shard-then-gather round trip re-joins exactly the pre-shard extent";
      }
      result.push_back(stmt);
      continue;
    }
    if (source == nullptr) {
      result.push_back(stmt);
      continue;
    }

    auto consumer_result = std::dynamic_pointer_cast<const TileType>(call->GetType());
    if (!consumer_result) {
      // Not a tile result (tile.store returns a Tensor, system.tfree returns
      // nothing): the op reads the extent but does not carry it onward, which is
      // exactly where the per-lane extent is meant to land.
      //
      // The store is the one consumer that must not see an EMPTY lane. When the
      // ragged extent does not reach the second lane (V <= half) that lane's
      // extent is 0, and a zero-row store is outside pto-isa's contract:
      // TSTORE_IMPL asserts ``src.GetValidRow() > 0 && src.GetValidCol() > 0``
      // (npu/a2a3/TStore.hpp) and PTOAS's PartitionViewOp rejects a statically
      // non-positive size. A release build compiles the assert out and the DMA
      // moves nothing, but a debug / CPU-sim / CA-model build traps. So guard
      // the store on a non-empty lane. The tpop and the tfree stay
      // UNCONDITIONAL: both lanes must still pop and free the slot or the pipe
      // desynchronizes.
      // A store of a DEFERRED boundary tile would write the transport's box, not
      // the lane's data: the extent never got a consumer to land on.
      if (IsOp(call, "tile.store")) {
        RejectDeferredValidStore(source->deferred, call);
      }
      if (IsOp(call, "tile.store") && !IsProvablyPositive(source->lane_extent)) {
        // The guard can be a plain IfStmt because a store's SSA result is a
        // destination-passing alias (see CollectChainedStoreDestinations). Only a
        // CHAINED store makes that unsafe — skipping the first store would leave
        // the second writing through a tensor version that was never produced —
        // and guarding it would need a return-carrying if (a phi over stored /
        // not-stored). Report that shape instead of emitting an unguarded
        // zero-row store.
        CHECK_SPAN(state.chained_store_dests.count(assign->var_.get()) == 0, call->span_)
            << "pl.split_aiv: a store whose result feeds another store cannot be guarded against an "
               "empty lane. The split axis is only partially valid, so one lane's extent can be 0 at "
               "runtime ("
            << PythonPrint(source->lane_extent) << "), and a zero-row store is outside the ISA contract.\n"
            << "Author one of these instead:\n"
            << "  * write the per-lane shard to its destination with a single subscript assignment "
               "inside the region\n"
            << "  * make the split axis fully valid before the crossing (pl.set_validshape to the "
               "full extent) so every lane is non-empty";
        result.push_back(GuardNonEmptyLane(stmt, source->lane_extent, call->span_));
        continue;
      }
      result.push_back(stmt);
      continue;
    }

    // A pad FILL is the one consumer that reads where the lane's data ends. Its
    // result is fully valid by construction, so it hits the widen shortcut below
    // and the deferred extent would be dropped — leaving the fill with nothing to
    // do and the transport's padding declared as data.
    RejectDeferredPadFill(source->deferred, call, "pl.aiv_shard(acc)");

    // A consumer that WIDENS the logical region back to the whole physical box
    // (a set_validshape to the full extent) deliberately drops the per-lane
    // extent — the author is declaring the padding to be data. Nothing to carry.
    if (tile_view_semantics::ShapeExprListsEquivalent(
            tile_view_semantics::GetEffectiveTileView(*consumer_result).valid_shape,
            consumer_result->shape_)) {
      result.push_back(stmt);
      continue;
    }
    CHECK_SPAN(PassesThroughValidShape(source->before, consumer_result), call->span_)
        << "pl.split_aiv: '" << call->op_->name_
        << "' reshapes the logical valid region of a per-lane cross-core value, which is not "
           "supported. The shard's split-axis extent differs per lane ("
        << PythonPrint(source->lane_extent)
        << "), and that extent can only be carried by ops that pass the valid_shape through "
           "unchanged — the boundary offers no way to re-narrow a popped tile afterwards.\n"
        << "Author one of these instead:\n"
        << "  * do the reshaping compute on the CUBE side, before pl.aiv_shard\n"
        << "  * make the split axis fully valid before the crossing "
           "(pl.set_validshape to the full extent) and treat the padding as don't-care\n"
        << "  * store the per-lane shard first, and reshape in a later kernel";
    auto new_type = WithValidDim(consumer_result, split_dim, source->lane_extent);
    auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_type, assign->var_->span_);
    auto new_call = std::make_shared<Call>(call->op_, call->args_, call->kwargs_, new_type, call->span_);
    state.replacements[assign->var_.get()] = new_var;
    state.tracked[new_var.get()] = LocalizedTile{source->lane_extent, source->joined_extent, consumer_result};
    result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
  }

  return result;
}

}  // namespace

std::vector<StmtPtr> LocalizeExplicitBoundaryValid(const std::vector<StmtPtr>& stmts, int split_dim,
                                                   const Span& region_span) {
  LocalizeState state;
  CollectChainedStoreDestinations(stmts, &state.chained_store_dests);
  auto result = LocalizeStmts(stmts, split_dim, region_span, state);

  if (state.replacements.empty()) {
    return result;
  }
  // One final substitution re-points every downstream use (including the ones
  // inside nested control flow) at the retyped vars, mirroring how the AUTO
  // halving path finishes in ProcessStmts.
  StmtPtr body = (result.size() == 1) ? result[0] : std::make_shared<SeqStmts>(result, region_span);
  return transform_utils::FlattenToStmts(transform_utils::Substitute(body, state.replacements));
}

namespace {

/// Mutable state threaded through DeferAutoRuntimeShardExtents' walk.
struct AutoDeferState {
  int split_dim = 0;
  VarPtr lane_index;                                    ///< the injected subblock_idx binding
  std::unordered_map<const Var*, VarPtr> replacements;  ///< retyped shard: old var -> new var
  std::unordered_set<const Var*> deferred;              ///< the retyped shards (new vars)
  std::unordered_set<const Var*> read_vars;             ///< every var some expression reads
};

/// Whether `call` is a Cube -> Vector shard whose lane extents are only known
/// at runtime: not rebalanced (no `lane_stride`), partially valid on the split
/// axis (or on an odd one), and not statically resolvable into lane extents.
bool IsRuntimeExtentShard(const CallPtr& call, int split_dim) {
  if (!IsOp(call, "tile.aiv_shard") || call->args_.empty() || call->HasKwarg("lane_stride")) return false;
  auto operand = std::dynamic_pointer_cast<const TileType>(call->args_[0]->GetType());
  if (!operand || split_dim >= static_cast<int>(operand->shape_.size())) return false;
  const auto operand_valid = tile_view_semantics::GetEffectiveTileView(*operand).valid_shape;
  const bool fully_valid_even = static_cast<int>(operand_valid.size()) <= split_dim ||
                                (AreExprsEqual(operand_valid[split_dim], operand->shape_[split_dim]) &&
                                 !IsOddSplitExtent(operand->shape_[split_dim]));
  return !fully_valid_even && !HasStaticLaneExtents(operand, split_dim, /*lane_stride=*/nullptr);
}

/// Whether `expr` names a shard this walk retyped to the full box.
bool IsDeferredShard(const ExprPtr& expr, const AutoDeferState& state) {
  auto var = AsVarLike(expr);
  if (!var) return false;
  auto replaced = state.replacements.find(var.get());
  const Var* key = (replaced != state.replacements.end()) ? replaced->second.get() : var.get();
  return state.deferred.count(key) != 0;
}

/// A deferred shard declares the transport's box rather than the lane's data, so
/// it may only feed an op whose own result the halving typed with the lane's
/// extent. Binding it to a carry, a branch result or a return would hand the
/// box to a binding the halving typed with the lane's extent.
void RejectDeferredShardBinding(const std::vector<ExprPtr>& values, const char* binding, const Span& span,
                                const AutoDeferState& state) {
  for (const auto& value : values) {
    CHECK_SPAN(!IsDeferredShard(value, state), span)
        << "pl.split: a Cube -> Vector boundary tile whose split-axis valid extent is a RUNTIME value "
           "is bound directly to a "
        << binding
        << ". The tile has to declare the transport's full box (that is what places lane 1's band where "
           "the producer wrote it), so it cannot carry the lane's own extent across that binding.\n"
        << "Author one of these instead:\n"
        << "  * apply the lane's own compute first (a cast, an elementwise op) and bind its result\n"
        << "  * make the split-axis valid extent a compile-time constant";
  }
}

/// The condition under which a store of `stored` has data on this lane: every
/// valid extent that depends on the lane index and is not provably positive
/// must be non-zero. Null when the store cannot be empty on either lane.
ExprPtr StoreNonEmptyCondition(const ExprPtr& stored, const AutoDeferState& state, const Span& span) {
  if (!state.lane_index) return nullptr;
  auto tile = std::dynamic_pointer_cast<const TileType>(stored->GetType());
  if (!tile) return nullptr;
  ExprPtr condition;
  for (const auto& extent : tile_view_semantics::GetEffectiveTileView(*tile).valid_shape) {
    if (!extent || IsProvablyPositive(extent)) continue;
    var_collectors::VarDefUseCollector uses;
    uses.VisitExpr(extent);
    if (uses.var_uses.count(state.lane_index.get()) == 0) continue;
    auto non_empty = MakeNonEmpty(extent, span);
    condition = condition ? MakeAnd(condition, non_empty, span) : non_empty;
  }
  return condition;
}

/// Guard `stmt` (a store of `store_call`) against an empty lane when needed.
/// Only a store whose result nothing reads is guarded: a plain `if` would leave
/// a read result (a later store's destination, a loop yield, a return)
/// conditionally defined, and guarding it soundly needs a phi over the stored
/// / not-stored tensor versions. Such a store is left unguarded, as every AUTO
/// store was before -- ordinary authoring (slice writes into one tensor, a
/// store inside a loop), where a zero-row store moves nothing in a release
/// build.
StmtPtr GuardLaneStore(const StmtPtr& stmt, const CallPtr& store_call, const VarPtr& result,
                       const AutoDeferState& state) {
  if (result && state.read_vars.count(result.get()) != 0) return stmt;
  auto condition = StoreNonEmptyCondition(store_call->args_[0], state, store_call->span_);
  if (!condition) return stmt;
  return std::make_shared<IfStmt>(condition, stmt, std::nullopt, std::vector<VarPtr>{}, store_call->span_);
}

std::vector<StmtPtr> DeferAutoStmts(const std::vector<StmtPtr>& stmts, AutoDeferState& state);

StmtPtr DeferAutoNestedBody(const StmtPtr& body, AutoDeferState& state) {
  if (!body) return body;
  auto inner = DeferAutoStmts(transform_utils::FlattenToStmts(body), state);
  if (inner.size() == 1) return inner[0];
  return std::make_shared<SeqStmts>(inner, body->span_);
}

std::vector<ExprPtr> IterArgInits(const std::vector<IterArgPtr>& iter_args) {
  std::vector<ExprPtr> inits;
  for (const auto& iter_arg : iter_args) {
    if (iter_arg) inits.push_back(iter_arg->initValue_);
  }
  return inits;
}

std::vector<StmtPtr> DeferAutoStmts(const std::vector<StmtPtr>& stmts, AutoDeferState& state) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      RejectDeferredShardBinding(IterArgInits(for_stmt->iter_args_), "loop carry", for_stmt->span_, state);
      auto new_body = DeferAutoNestedBody(for_stmt->body_, state);
      result.push_back(new_body == for_stmt->body_
                           ? stmt
                           : loop_repair::RebuildForStmt(for_stmt, for_stmt->iter_args_, new_body,
                                                         for_stmt->return_vars_));
      continue;
    }
    if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      RejectDeferredShardBinding(IterArgInits(while_stmt->iter_args_), "loop carry", while_stmt->span_,
                                 state);
      auto new_body = DeferAutoNestedBody(while_stmt->body_, state);
      if (new_body == while_stmt->body_) {
        result.push_back(stmt);
      } else {
        auto new_while = MutableCopy(while_stmt);
        new_while->body_ = new_body;
        result.push_back(new_while);
      }
      continue;
    }
    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto new_if = MutableCopy(if_stmt);
      new_if->then_body_ = DeferAutoNestedBody(if_stmt->then_body_, state);
      if (if_stmt->else_body_.has_value()) {
        new_if->else_body_ = DeferAutoNestedBody(*if_stmt->else_body_, state);
      }
      result.push_back(new_if);
      continue;
    }
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
      for (auto& s : DeferAutoStmts(seq->stmts_, state)) result.push_back(s);
      continue;
    }
    if (auto yield = std::dynamic_pointer_cast<const YieldStmt>(stmt)) {
      RejectDeferredShardBinding(yield->value_, "loop or branch result", yield->span_, state);
      result.push_back(stmt);
      continue;
    }
    if (auto ret = std::dynamic_pointer_cast<const ReturnStmt>(stmt)) {
      RejectDeferredShardBinding(ret->value_, "function return", ret->span_, state);
      result.push_back(stmt);
      continue;
    }

    CallPtr call;
    VarPtr bound;
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      call = AsCall(assign->value_);
      bound = assign->var_;
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      call = AsCall(eval->expr_);
    }
    if (!call || !call->op_) {
      result.push_back(stmt);
      continue;
    }

    if (bound && IsOp(call, "tile.get_subblock_idx")) {
      state.lane_index = bound;
      result.push_back(stmt);
      continue;
    }

    // --- The boundary: a runtime lane extent must not reach the pop. --------
    if (bound && IsRuntimeExtentShard(call, state.split_dim)) {
      auto full_type = WithFullSplitAxisValid(bound->GetType(), state.split_dim);
      auto new_var = std::make_shared<Var>(bound->name_hint_, full_type, bound->span_);
      auto new_call = std::make_shared<Call>(call->op_, call->args_, call->kwargs_, full_type, call->span_);
      state.replacements[bound.get()] = new_var;
      state.deferred.insert(new_var.get());
      result.push_back(std::make_shared<AssignStmt>(new_var, new_call, stmt->span_));
      continue;
    }

    // --- The two consumers that read the deferred box itself. ---------------
    const bool reads_deferred = !call->args_.empty() && IsDeferredShard(call->args_[0], state);
    RejectDeferredPadFill(reads_deferred, call, "the first vector op that reads acc");
    if (IsOp(call, "tile.store") && !call->args_.empty()) {
      RejectDeferredValidStore(reads_deferred, call);
      result.push_back(GuardLaneStore(stmt, call, bound, state));
      continue;
    }
    result.push_back(stmt);
  }
  return result;
}

/// Whether any Cube -> Vector shard in `body` has lane extents only known at runtime.
class RuntimeLaneExtentShardFinder : public IRVisitor {
 public:
  explicit RuntimeLaneExtentShardFinder(int split_dim) : split_dim_(split_dim) {}
  [[nodiscard]] bool Found() const { return found_; }

 protected:
  void VisitExpr_(const CallPtr& op) override {
    if (!found_) found_ = IsRuntimeExtentShard(op, split_dim_);
    IRVisitor::VisitExpr_(op);
  }

 private:
  int split_dim_;
  bool found_ = false;
};

}  // namespace

std::vector<StmtPtr> DeferAutoRuntimeShardExtents(const std::vector<StmtPtr>& stmts, int split_dim,
                                                  const Span& span) {
  RuntimeLaneExtentShardFinder finder(split_dim);
  for (const auto& stmt : stmts) finder.VisitStmt(stmt);
  if (!finder.Found()) return stmts;

  AutoDeferState state;
  state.split_dim = split_dim;
  var_collectors::VarDefUseCollector reads;
  for (const auto& stmt : stmts) reads.VisitStmt(stmt);
  state.read_vars = std::move(reads.var_uses);
  auto result = DeferAutoStmts(stmts, state);
  if (state.replacements.empty()) return result;
  StmtPtr body = (result.size() == 1) ? result[0] : std::make_shared<SeqStmts>(result, span);
  return transform_utils::FlattenToStmts(transform_utils::Substitute(body, state.replacements));
}

}  // namespace split_axis
}  // namespace ir
}  // namespace pypto

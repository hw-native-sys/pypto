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

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

using Attrs = std::vector<std::pair<std::string, std::any>>;

namespace {

/// Extract a compile-time integer from a ConstInt or Neg(ConstInt) expression.
int64_t GetConstIntValue(const ExprPtr& expr, const std::string& what) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(expr)) {
    return ci->value_;
  }
  if (auto neg = std::dynamic_pointer_cast<const Neg>(expr)) {
    if (auto inner = std::dynamic_pointer_cast<const ConstInt>(neg->operand_)) {
      return -inner->value_;
    }
  }
  throw pypto::ValueError("LowerPipelineLoops: " + what + " must be a compile-time integer constant, got " +
                          expr->TypeName());
}

/// Non-throwing variant — returns nullopt if `expr` is not a compile-time integer.
std::optional<int64_t> TryGetConstInt(const ExprPtr& expr) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(expr)) {
    return ci->value_;
  }
  if (auto neg = std::dynamic_pointer_cast<const Neg>(expr)) {
    if (auto inner = std::dynamic_pointer_cast<const ConstInt>(neg->operand_)) {
      return -inner->value_;
    }
  }
  return std::nullopt;
}

/// Trip count for a static for-loop range.
int64_t ComputeStaticTripCount(int64_t start, int64_t stop, int64_t step) {
  if (step > 0 && start < stop) return (stop - start + step - 1) / step;
  if (step < 0 && start > stop) return (start - stop + (-step) - 1) / (-step);
  return 0;
}

ExprPtr MakeConstIndex(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

/// `base + offset_val`, with constant-folding when `base` is a ConstInt.
/// Emitting the unfolded form trips the round-trip verifier because the
/// reparser folds `8 + 1` back to `9`.
ExprPtr OffsetIndex(const ExprPtr& base, int64_t offset_val, const Span& span) {
  if (offset_val == 0) return base;
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(base)) {
    return MakeConstIndex(ci->value_ + offset_val, span);
  }
  return MakeAdd(base, MakeConstIndex(offset_val, span), span);
}

/// Build a fresh outer loop variable mirroring `original` (same name, same type, same span).
VarPtr CloneLoopVar(const VarPtr& original) {
  return std::make_shared<Var>(original->name_hint_, original->GetType(), original->span_);
}

/// Fresh IterArg mirroring `original`, with `init_value` as the initial value.
IterArgPtr MakeFreshIterArg(const IterArgPtr& original, const ExprPtr& init_value) {
  return std::make_shared<IterArg>(original->name_hint_, original->GetType(), init_value, original->span_);
}

/// Fresh Var mirroring `original` with a suffixed name (for intermediate return_vars).
VarPtr MakeFreshVar(const VarPtr& original, const std::string& suffix) {
  return std::make_shared<Var>(original->name_hint_ + suffix, original->GetType(), original->span_);
}

/// Split a body into (stmts_before_yield, yield_values). If the body ends with a
/// terminal `YieldStmt` (either standalone or as the final stmt of a top-level
/// `SeqStmts`), strip it and return its values. Otherwise return the body unchanged
/// and an empty value list. Always pass through — callers that have no iter_args
/// simply see an empty yield vector and treat `stmts` as the whole body.
/// Tag every tile-producing ``Call`` in a cloned pipeline-stage body with one
/// ``(group, stage)`` pipeline-membership pair. The pair is *appended* to any
/// membership already present, so a tile inside an already-lowered inner
/// pipeline accumulates both its inner and this (enclosing) membership — which
/// keeps nested same-core pipelines separated at every level. MemoryReuse later
/// uses this with role-aware granularity: it blocks a cross-stage buffer share
/// only when at least one side is a load buffer (compute intermediates of
/// different stages may still coalesce) — see kPipelineMembershipAttr.
///
/// Only the LHS-defining ``Call`` of a TileType ``AssignStmt`` is tagged — that
/// is exactly the set of tile *definitions* MemoryReuse keys its reuse decision
/// on. Non-tile assigns and non-Call tile defs (e.g. a bare Var alias) carry no
/// membership and simply fall through unconstrained.
///
/// Cube accumulators are the one exception: they are *not* tagged. A pipeline
/// loop double-buffers the operands it *loads* — their loads overlap the previous
/// stage's compute, so two stages' operand buffers are genuinely co-live and must
/// stay apart. An accumulator is not loaded; it is written by the single
/// serialized cube (a `tile.matmul*` MAD), which retires one tile's MAD before
/// starting the next regardless of how many stages the scheduler overlaps. So an
/// accumulator nested in a pipeline loop is never co-live with the next stage's
/// accumulator, and tagging it only makes MemoryReuse's capacity-gate request one
/// L0C buffer per stage and then shed it back (a redundant separation that emits a
/// spurious PH-MR-001, and for an accumulator nested N loops deep balloons to 2^N
/// requested buffers). Left untagged, the drain-before-next accumulator coalesces
/// by lifetime alone onto the single buffer it actually needs. The exception keys
/// on the *producer* op, not on `Mem.Acc`: a data-movement op that also targets
/// Acc (e.g. `tile.extract(..., target_memory=Acc)`) is a real per-stage buffer
/// that overlaps across stages, so it stays tagged like any other loaded operand.
///
/// One loop *does* tag its cube accumulator: the moving loop of a dbC=2
/// double-buffered-L0C emit (kPipelineDoubleBufferCAttr), which genuinely co-lives
/// two accumulators. See `loop_double_buffers_c_` in the tagger.
class PipelineMembershipTagger : public IRMutator {
 public:
  PipelineMembershipTagger(int32_t group, int32_t stage, bool loop_double_buffers_c)
      : group_(group), stage_(stage), loop_double_buffers_c_(loop_double_buffers_c) {}

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Recurse first so nested control flow (e.g. an inner lowered pipeline) is
    // visited; AssignStmt itself carries no child statements, but keeping the
    // default traversal makes the tagger robust to future stmt nesting.
    auto visited = IRMutator::VisitStmt_(op);
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(visited);
    if (!assign) return visited;
    auto tile_type = std::dynamic_pointer_cast<const TileType>(assign->var_->GetType());
    if (!tile_type) return visited;
    auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
    if (!call) return visited;

    // Cube accumulators are written by the serialized cube, never co-live across a
    // pipeline stage — skip them so MemoryReuse coalesces them onto the single L0C
    // buffer they need (see the class comment above). Gate on the *producer* being
    // a cube MAD op (`tile.matmul` and its `_acc` / `_bias` / `_mx*` variants), not
    // on `Mem.Acc` alone: a data-movement op that also targets Acc (e.g.
    // `tile.extract(..., target_memory=Acc)`) is a genuine per-stage buffer that
    // overlaps across stages and must keep its membership.
    //
    // The dbC=2 exception (loop_double_buffers_c_): the moving loop that carries
    // kPipelineDoubleBufferCAttr *does* co-live two accumulators — tile i's FIXPIPE
    // drain overlaps tile i+1's MAD — so its accumulator DOES need this loop's
    // (group, stage) to double-buffer. Only that loop tags it; every enclosing
    // pipeline loop still skips it (the outer loop double-buffers operands, but the
    // MADs writing successive outer-stage accumulators still serialize on the one
    // cube). The result is a flat depth-2 membership from the dbC loop alone, so the
    // PyPTO capacity-gate (#1475) allocates exactly the two co-live L0C buffers
    // rather than the per-loop cross-product it would shed back to one.
    const bool is_cube_matmul = call->op_ && call->op_->name_.rfind("tile.matmul", 0) == 0;
    const bool is_cube_accumulator = is_cube_matmul && tile_type->GetMemorySpace() == MemorySpace::Acc;
    if (is_cube_accumulator && !loop_double_buffers_c_) return visited;

    auto packed = call->GetAttr<std::string>(kPipelineMembershipAttr, std::string());
    packed = AppendPipelineMembership(packed, group_, stage_);
    auto new_attrs = StripAttr(call->attrs_, kPipelineMembershipAttr);
    new_attrs.emplace_back(kPipelineMembershipAttr, std::move(packed));
    auto new_call = std::make_shared<Call>(call->op_, call->args_, call->kwargs_, std::move(new_attrs),
                                           call->GetType(), call->span_);
    return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
  }

 private:
  int32_t group_;
  int32_t stage_;
  bool loop_double_buffers_c_;
};

std::pair<StmtPtr, std::vector<ExprPtr>> SplitBodyYield(const StmtPtr& body) {
  if (auto yield = std::dynamic_pointer_cast<const YieldStmt>(body)) {
    return {std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, body->span_), yield->value_};
  }
  auto seq = std::dynamic_pointer_cast<const SeqStmts>(body);
  if (!seq || seq->stmts_.empty()) {
    return {body, {}};
  }
  auto yield = std::dynamic_pointer_cast<const YieldStmt>(seq->stmts_.back());
  if (!yield) {
    return {body, {}};
  }
  std::vector<StmtPtr> without(seq->stmts_.begin(), seq->stmts_.end() - 1);
  return {std::make_shared<SeqStmts>(std::move(without), seq->span_), yield->value_};
}

/**
 * @brief Mutator that lowers user-written `pl.pipeline(N, stage=F)` loops
 *        (`ForKind::Pipeline` + `attrs_["pipeline_stages"] == F` with `F > 1`)
 *        into a replicated main loop plus a modulo-dispatch remainder.
 *
 * The produced outer loop **keeps `ForKind::Pipeline` and downgrades
 * `pipeline_stages` to `1`** as the post-lowering marker for the downstream
 * `CanonicalizeIOOrder` pass (which scopes its IO reorder to pipeline bodies
 * and demotes the kind/strips the attr on exit). Keeping the (kind, attr) pair
 * together at every observable state preserves the bidirectional structural
 * invariant `kind == Pipeline ⇔ pipeline_stages attr present` (verified by
 * `PipelineLoopValid`), so the IR survives print/parse round-trip throughout.
 *
 * Idempotency: re-running this pass on its own output sees `factor == 1` and
 * leaves the loop untouched (trigger requires `factor > 1`). User-written
 * `pl.pipeline(stage=1)` is treated identically — no replication happens, the
 * loop is left intact for `CanonicalizeIOOrder`.
 *
 * Static bounds → bare `SeqStmts` tail with exactly rem_iters clones flattened
 *   into the outer scope (plus trailing `AssignStmt`s to bind the outer loop's
 *   `return_vars` when iter_args exist).
 * Dynamic bounds (start and/or stop are runtime Exprs) → a cascaded
 *   `if rem == k` dispatch for k in [1, factor); each branch body is a bare
 *   `SeqStmts` of k cloned bodies (followed by a `YieldStmt` when iter_args
 *   exist). Step must always be a compile-time constant.
 *
 * `iter_args` are supported: loop-carried state threads sequentially through the
 * F replicated clones in the main loop (each clone consumes the previous clone's
 * yielded expressions), and through the tail clones starting from the main
 * loop's return_vars. In the dynamic case, each IfStmt in the cascade carries
 * `return_vars` matching the iter_args types; the innermost else yields the
 * main-loop return_vars so the `rem == 0` fall-through is a no-op.
 */
class LowerPipelineMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (op->kind_ != ForKind::Pipeline || !op->HasAttr(kPipelineStagesAttr)) {
      return IRMutator::VisitStmt_(op);
    }
    int64_t factor = static_cast<int64_t>(op->GetAttr<int>(kPipelineStagesAttr, 0));
    INTERNAL_CHECK_SPAN(factor >= 1, op->span_)
        << "LowerPipelineLoops: pipeline_stages must be >= 1, got " << factor;

    // factor == 1 is either a user-written `pl.pipeline(stage=1)` or the
    // post-lowering marker emitted by a previous run. Either way nothing needs
    // replicating — leave the (kind, attr) pair intact for CanonicalizeIOOrder
    // to scope on, and just recurse into the body to lower nested pipelines.
    // This also makes re-running the pass a natural no-op (idempotency).
    if (factor == 1) {
      return IRMutator::VisitStmt_(op);
    }

    // Recurse into the body first so nested pipeline-marked loops are lowered too.
    auto inner_body = VisitStmt(op->body_);

    // Step must always be static — the main loop's stride and per-clone offsets
    // both depend on `factor * step` being a compile-time integer.
    int64_t step = GetConstIntValue(op->step_, "step");
    INTERNAL_CHECK_SPAN(step != 0, op->span_) << "LowerPipelineLoops: step cannot be zero";

    auto start_const = TryGetConstInt(op->start_);
    auto stop_const = TryGetConstInt(op->stop_);

    // Under the PTOAS memory planner, a supported tile loop-carry is lowered to
    // an intra-core multi-buffer (ptoas multi_tile_buf): the loop stays rolled
    // and the carry rotates through `factor` physical slots, letting ptoas
    // overlap iterations (the counterpart of the cross-core slot_num ring
    // buffer). Unsupported shapes fall through to the unroll path below, so
    // nothing regresses.
    const auto* ctx = PassContext::Current();
    if (ctx != nullptr && ctx->GetMemoryPlanner() == MemoryPlanner::PtoAS && stop_const.has_value() &&
        CanLowerToMultiTile(op, factor, start_const, stop_const, step)) {
      // CanLowerToMultiTile pins start==0 / step==1, so the loop var equals the
      // iteration number and the trip count is just the (static) stop bound.
      return LowerToMultiTile(op, inner_body, factor, *stop_const);
    }

    // Cross-core loops are skewed/demoted to Sequential by the earlier
    // SkewCrossCorePipeline pass, so by here only same-core pipeline loops
    // (GM->L1, L1->L0, nested matmul stage loops) remain Pipeline-marked. They
    // are replicated for ping-pong via the uniform unroll path below.
    if (start_const.has_value() && stop_const.has_value()) {
      return LowerStatic(op, inner_body, factor, *start_const, *stop_const, step);
    }
    return LowerDynamic(op, inner_body, factor, step);
  }

 private:
  /// Monotonic id assigned per replicated region (one per ReplicateBody call).
  /// Distinct regions get distinct ids so MemoryReuse only forbids cross-stage
  /// reuse *within* a region, never between unrelated pipelines.
  int32_t next_pipeline_group_ = 0;

  /// Empty static trip-count path — no replication needed and nothing for
  /// CanonicalizeIOOrder to cluster (the loop never executes). Demote kind to
  /// Sequential and strip `pipeline_stages` together so the bidirectional
  /// invariant `kind == Pipeline ⇔ pipeline_stages attr present` stays whole.
  StmtPtr DemoteToSequential(const ForStmtPtr& op, const StmtPtr& inner_body) {
    auto cleaned = MutableCopy(op);
    cleaned->body_ = inner_body;
    cleaned->kind_ = ForKind::Sequential;
    cleaned->attrs_ = StripAttr(op->attrs_, kPipelineStagesAttr);
    return cleaned;
  }

  // ====================================================================
  // PTOAS-planner intra-core multi-buffer lowering (ptoas multi_tile_buf)
  // ====================================================================

  /// Monotonic id for uniquely naming multi-buffer temporaries in a function.
  int32_t next_multi_tile_id_ = 0;

  /// Find the top-level `AssignStmt` in `body` that defines `var` (nullptr if
  /// the definer is nested / absent — such bodies are not multi-buffer lowerable).
  static const AssignStmt* TopLevelDefiner(const StmtPtr& body, const Var* var) {
    if (auto as = std::dynamic_pointer_cast<const AssignStmt>(body)) {
      return as->var_.get() == var ? as.get() : nullptr;
    }
    auto seq = std::dynamic_pointer_cast<const SeqStmts>(body);
    if (!seq) return nullptr;
    for (const auto& s : seq->stmts_) {
      if (auto as = std::dynamic_pointer_cast<const AssignStmt>(s)) {
        if (as->var_.get() == var) return as.get();
      }
    }
    return nullptr;
  }

  /// Whether `op` is a single-tile-carry pipeline loop this pass can rewrite to a
  /// multi_tile_buf. v1 supports: factor in [2,16] (ptoas count bound), one tile
  /// iter_arg with a resolved memory space, one return_var, static bounds with
  /// start==0 / step==1, and a carry produced by a top-level yield producer.
  /// Everything else falls back to the unroll path (no regression).
  bool CanLowerToMultiTile(const ForStmtPtr& op, int64_t factor, const std::optional<int64_t>& start_const,
                           const std::optional<int64_t>& stop_const, int64_t step) {
    if (factor < 2 || factor > 16) return false;
    if (!start_const.has_value() || !stop_const.has_value()) return false;
    if (*start_const != 0 || step != 1 || *stop_const <= 0) return false;
    if (op->iter_args_.size() != 1 || op->return_vars_.size() != 1) return false;
    auto carry_type = std::dynamic_pointer_cast<const TileType>(op->iter_args_[0]->GetType());
    if (!carry_type || !carry_type->memory_space_.has_value()) return false;
    auto [body_no_yield, yields] = SplitBodyYield(op->body_);
    if (yields.size() != 1) return false;
    auto yield_var = std::dynamic_pointer_cast<const Var>(yields[0]);
    if (!yield_var) return false;
    return TopLevelDefiner(op->body_, yield_var.get()) != nullptr;
  }

  /// Wrap a synthesized op Call in a fresh named Var + AssignStmt.
  std::pair<VarPtr, StmtPtr> BindCall(const CallPtr& call, const std::string& name, const Span& sp) {
    auto var = std::make_shared<Var>(name, call->GetType(), sp);
    return {var, std::make_shared<AssignStmt>(var, call, sp)};
  }

  /// `%var = tile.alloc_multi(shape) {dtype, target_memory, count}` -> (var, stmt).
  std::pair<VarPtr, StmtPtr> MakeAllocMulti(const std::shared_ptr<const TileType>& slot_type, int64_t factor,
                                            const std::string& name, const Span& sp) {
    Attrs kwargs;
    kwargs.emplace_back("dtype", slot_type->dtype_);
    // Guaranteed present by CanLowerToMultiTile; guarded to satisfy the
    // unchecked-optional-access lint (matches the tile.create build idiom).
    if (const auto& ms = slot_type->memory_space_) {
      kwargs.emplace_back("target_memory", *ms);
    }
    kwargs.emplace_back("count", static_cast<int>(factor));
    auto call = OpRegistry::GetInstance().Create(
        "tile.alloc_multi", {tile_conversion_utils::MakeShapeTuple(slot_type->shape_, sp)}, kwargs, sp);
    return BindCall(call, name, sp);
  }

  /// `%var = tile.multi_get(mtb, index) {count}` -> (var, stmt).
  std::pair<VarPtr, StmtPtr> MakeMultiGet(const VarPtr& mtb, const ExprPtr& index, int64_t factor,
                                          const std::string& name, const Span& sp) {
    Attrs kwargs;
    kwargs.emplace_back("count", static_cast<int>(factor));
    auto call = OpRegistry::GetInstance().Create("tile.multi_get", {mtb, index}, kwargs, sp);
    return BindCall(call, name, sp);
  }

  /// Rebuild `body`, stamping the producer of `yield_var` with the slot-alias
  /// attr so InitMemRef retargets its buffer onto `slot_name`.
  StmtPtr StampProducerSlot(const StmtPtr& body, const Var* yield_var, const std::string& slot_name) {
    auto rebuild = [&](const AssignStmtPtr& as) -> StmtPtr {
      auto call = std::dynamic_pointer_cast<const Call>(as->value_);
      INTERNAL_CHECK_SPAN(call, as->span_) << "Internal error: multi-buffer carry producer must be a Call";
      Attrs attrs = call->attrs_;
      attrs.emplace_back(kMultiBufferAliasSlotAttr, slot_name);
      auto stamped = std::make_shared<Call>(call->op_, call->args_, call->kwargs_, std::move(attrs),
                                            call->GetType(), call->span_);
      return std::make_shared<AssignStmt>(as->var_, stamped, as->span_);
    };
    if (auto as = std::dynamic_pointer_cast<const AssignStmt>(body)) {
      return as->var_.get() == yield_var ? rebuild(as) : body;
    }
    auto seq = std::dynamic_pointer_cast<const SeqStmts>(body);
    INTERNAL_CHECK(seq) << "Internal error: multi-buffer body must be a SeqStmts or AssignStmt";
    std::vector<StmtPtr> out;
    out.reserve(seq->stmts_.size());
    for (const auto& s : seq->stmts_) {
      auto as = std::dynamic_pointer_cast<const AssignStmt>(s);
      out.push_back(as && as->var_.get() == yield_var ? rebuild(as) : s);
    }
    return std::make_shared<SeqStmts>(std::move(out), seq->span_);
  }

  /// Clone `inner_body` with (loop_var -> loop_var_val, carry -> carry_read),
  /// strip its yield, and stamp the yield producer to alias `slot_name`.
  StmtPtr CloneIterBody(const ForStmtPtr& op, const StmtPtr& inner_body, const ExprPtr& loop_var_val,
                        const ExprPtr& carry_read, const std::string& slot_name, const Span& sp) {
    std::unordered_map<const Var*, ExprPtr> sub;
    sub[op->loop_var_.get()] = loop_var_val;
    sub[op->iter_args_[0].get()] = carry_read;
    auto cloned = DeepClone(inner_body, sub, /*clone_def_vars=*/true);
    auto [body_no_yield, yields] = SplitBodyYield(cloned.cloned_body);
    INTERNAL_CHECK_SPAN(yields.size() == 1, sp)
        << "Internal error: multi-buffer body must yield exactly 1 value";
    auto yield_var = std::dynamic_pointer_cast<const Var>(yields[0]);
    INTERNAL_CHECK_SPAN(yield_var, sp) << "Internal error: multi-buffer yield must be a Var";
    return StampProducerSlot(body_no_yield, yield_var.get(), slot_name);
  }

  /// Lower a single-tile-carry pipeline loop (start==0, step==1, `trip`
  /// iterations) to a rolled loop over a `factor`-slot multi_tile_buf. Iteration
  /// k reads slot `(k-1) mod factor` (or the init for k==0, peeled out) and
  /// writes slot `k mod factor`; the loop result is the final slot, aliased to
  /// the original return_var.
  StmtPtr LowerToMultiTile(const ForStmtPtr& op, const StmtPtr& inner_body, int64_t factor, int64_t trip) {
    Span sp = op->span_;
    const std::string pfx = "__mtb" + std::to_string(next_multi_tile_id_++);
    auto carry_iter = op->iter_args_[0];
    auto carry_type = std::dynamic_pointer_cast<const TileType>(carry_iter->GetType());
    const ExprPtr init_val = carry_iter->initValue_;
    const VarPtr return_var = op->return_vars_[0];

    std::vector<StmtPtr> result;

    // 1. Hoisted multi-buffer allocation.
    auto [mtb_var, mtb_stmt] = MakeAllocMulti(carry_type, factor, pfx, sp);
    result.push_back(mtb_stmt);

    // 2. Peeled first iteration (k == 0): reads the init directly, writes slot 0.
    {
      auto [sc0, sc0_stmt] = MakeMultiGet(mtb_var, MakeConstIndex(0, sp), factor, pfx + "_s0", sp);
      auto body0 = CloneIterBody(op, inner_body, MakeConstIndex(0, sp), init_val, sc0->name_hint_, sp);
      result.push_back(sc0_stmt);
      result.push_back(body0);
    }

    // 3. Rolled remainder loop for k in [1, trip): reads slot (k-1)%F, writes k%F.
    if (trip > 1) {
      VarPtr new_loop_var = CloneLoopVar(op->loop_var_);
      ExprPtr cur_idx = MakeFloorMod(new_loop_var, MakeConstIndex(factor, sp), sp);
      ExprPtr prev_idx = MakeFloorMod(MakeAdd(new_loop_var, MakeConstIndex(factor - 1, sp), sp),
                                      MakeConstIndex(factor, sp), sp);
      auto [sp_var, sp_stmt] = MakeMultiGet(mtb_var, prev_idx, factor, pfx + "_sp", sp);
      auto [sc_var, sc_stmt] = MakeMultiGet(mtb_var, cur_idx, factor, pfx + "_sc", sp);
      auto bodyL = CloneIterBody(op, inner_body, new_loop_var, sp_var, sc_var->name_hint_, sp);
      auto loop_body = SeqStmts::Flatten(std::vector<StmtPtr>{sp_stmt, sc_stmt, bodyL}, sp);
      auto loop = std::make_shared<ForStmt>(new_loop_var, MakeConstIndex(1, sp), MakeConstIndex(trip, sp),
                                            MakeConstIndex(1, sp), std::vector<IterArgPtr>{}, loop_body,
                                            std::vector<VarPtr>{}, sp, ForKind::Sequential, Attrs{},
                                            op->leading_comments_);
      result.push_back(loop);
    }

    // 4. Bind the final slot ((trip-1) mod F) to the original return_var so
    //    downstream consumers read the last write (trip >= 1 is guaranteed).
    auto [res_var, res_stmt] =
        MakeMultiGet(mtb_var, MakeConstIndex((trip - 1) % factor, sp), factor, pfx + "_res", sp);
    result.push_back(res_stmt);
    result.push_back(
        std::make_shared<AssignStmt>(return_var, std::static_pointer_cast<const Expr>(res_var), sp));

    return SeqStmts::Flatten(std::move(result), sp);
  }

  /**
   * @brief Clone `body` `n` times with loop-var / iter-arg substitutions,
   *        threading loop-carried state through the clones.
   *
   * Each clone k:
   *  - substitutes `loop_var → base + k * step` (via OffsetIndex)
   *  - substitutes original iter_args with `initial_iter_substitutes` (when k == 0)
   *    or with the previous clone's yielded expressions (when k > 0)
   *  - is DeepCloned with `clone_def_vars=true` so nested definitions get fresh SSA vars
   *  - has its trailing `YieldStmt` (if any) stripped into the next clone's substitution
   *
   * Returns the concatenated body (a `SeqStmts` of the stripped clones) paired with
   * the last clone's yielded expressions. For loops without iter_args, the yield
   * vector is empty and each cloned body is appended verbatim.
   */
  struct ReplicatedRegion {
    StmtPtr body;                       // SeqStmts of cloned bodies (yields stripped)
    std::vector<ExprPtr> final_yields;  // last clone's yielded expressions
  };

  ReplicatedRegion ReplicateBody(const ForStmtPtr& op, const StmtPtr& body, int64_t n_clones, int64_t step,
                                 const ExprPtr& base, const std::vector<ExprPtr>& initial_iter_substitutes) {
    Span sp = op->span_;
    INTERNAL_CHECK_SPAN(initial_iter_substitutes.size() == op->iter_args_.size(), sp)
        << "Internal error: iter substitute count mismatch";

    // One replicated region == one pipeline group: its clones run concurrently
    // under the event scheduler, so their tiles must occupy distinct buffers.
    // The main loop and each (static / dynamic-cascade) tail are separate
    // ReplicateBody calls and thus separate groups — main-loop and tail tiles
    // are temporally disjoint and may freely reuse across the group boundary.
    const int32_t group = next_pipeline_group_++;

    std::vector<StmtPtr> clones;
    clones.reserve(static_cast<size_t>(n_clones));
    std::vector<ExprPtr> prev_yields;
    // Loop-invariant across clones: a dbC=2 emit's moving loop tags its cube accumulator
    // (co-live drain ping-pong); every other pipeline loop leaves cube accumulators
    // untagged (see the tagger). Read the attr once, not per clone.
    const bool loop_double_buffers_c = op->GetAttr<bool>(kPipelineDoubleBufferCAttr, false);
    for (int64_t k = 0; k < n_clones; ++k) {
      std::unordered_map<const Var*, ExprPtr> sub_map;
      sub_map[op->loop_var_.get()] = OffsetIndex(base, k * step, sp);
      for (size_t j = 0; j < op->iter_args_.size(); ++j) {
        sub_map[op->iter_args_[j].get()] = (k == 0) ? initial_iter_substitutes[j] : prev_yields[j];
      }
      auto cloned = DeepClone(body, sub_map, /*clone_def_vars=*/true);
      auto [cloned_stmts, cloned_yields] = SplitBodyYield(cloned.cloned_body);
      INTERNAL_CHECK_SPAN(cloned_yields.size() == op->iter_args_.size(), sp)
          << "Internal error: loop body must yield " << op->iter_args_.size() << " values for iter_args, got "
          << cloned_yields.size();
      // Tag this clone's tile definitions with (group, stage=k) so MemoryReuse
      // keeps the F clones' buffers apart (explicit ping-pong constraint).
      PipelineMembershipTagger tagger(group, static_cast<int32_t>(k), loop_double_buffers_c);
      cloned_stmts = tagger.VisitStmt(cloned_stmts);
      clones.push_back(cloned_stmts);
      prev_yields = std::move(cloned_yields);
    }
    return {SeqStmts::Flatten(std::move(clones), sp), std::move(prev_yields)};
  }

  std::vector<ExprPtr> ReturnVarsAsExprs(const std::vector<VarPtr>& vars) {
    std::vector<ExprPtr> result;
    result.reserve(vars.size());
    for (const auto& v : vars) result.push_back(v);
    return result;
  }

  /// Collect `initValue_` expressions from a vector of IterArgs — used when the
  /// tail runs without a preceding main loop, so its iter_args seed directly
  /// from the source loop's init values rather than a main-loop return_var.
  std::vector<ExprPtr> InitValueExprs(const std::vector<IterArgPtr>& iter_args) {
    std::vector<ExprPtr> result;
    result.reserve(iter_args.size());
    for (const auto& ia : iter_args) result.push_back(ia->initValue_);
    return result;
  }

  /// Fresh return_vars matching the originals' types, with a suffix applied to names.
  std::vector<VarPtr> MakeFreshReturnVars(const std::vector<VarPtr>& originals, const std::string& suffix) {
    std::vector<VarPtr> result;
    result.reserve(originals.size());
    for (const auto& v : originals) result.push_back(MakeFreshVar(v, suffix));
    return result;
  }

  /**
   * @brief Build the replicated main loop.
   *
   * Body is a SeqStmts of `factor` clones threading iter_args sequentially. When
   * the original loop has iter_args, the main loop gets fresh iter_args seeded
   * from the originals' init values; each clone consumes the previous clone's
   * yield, and the body ends with a YieldStmt of the last clone's yields to feed
   * the next outer iteration. `main_return_vars` controls the ForStmt's
   * return_vars (may be the original return_vars or fresh ones, depending on
   * whether a tail follows).
   */
  StmtPtr BuildMainLoop(const ForStmtPtr& op, const StmtPtr& body, int64_t factor, int64_t step,
                        const ExprPtr& main_start, const ExprPtr& main_stop,
                        const std::vector<VarPtr>& main_return_vars) {
    Span sp = op->span_;
    VarPtr new_loop_var = CloneLoopVar(op->loop_var_);

    // Fresh iter_args mirroring the originals (same init values as the source loop).
    std::vector<IterArgPtr> new_iter_args;
    new_iter_args.reserve(op->iter_args_.size());
    std::vector<ExprPtr> initial_substitutes;
    initial_substitutes.reserve(op->iter_args_.size());
    for (const auto& orig : op->iter_args_) {
      auto fresh = MakeFreshIterArg(orig, orig->initValue_);
      new_iter_args.push_back(fresh);
      initial_substitutes.push_back(fresh);
    }

    auto region = ReplicateBody(op, body, factor, step, new_loop_var, initial_substitutes);

    // Body = replicated clones, followed by YieldStmt(last_yields) when iter_args exist.
    std::vector<StmtPtr> body_parts = {region.body};
    if (!op->iter_args_.empty()) {
      body_parts.push_back(std::make_shared<YieldStmt>(region.final_yields, sp));
    }
    auto new_body = SeqStmts::Flatten(std::move(body_parts), sp);

    ExprPtr new_step = MakeConstIndex(factor * step, sp);
    // Post-lowering marker: kind stays Pipeline, attr is downgraded to 1 (the
    // body is already replicated, so no further stage expansion is needed).
    // Keeping the (kind, attr) pair together preserves PipelineLoopValid and
    // makes the IR print/parse round-trip safe (renders as
    // `pl.pipeline(..., stage=1)`). CanonicalizeIOOrder consumes the marker;
    // re-running LowerPipelineLoops sees factor=1 and skips (idempotent).
    Attrs new_attrs = StripAttr(op->attrs_, kPipelineStagesAttr);
    new_attrs.emplace_back(kPipelineStagesAttr, 1);
    return std::make_shared<ForStmt>(new_loop_var, main_start, main_stop, new_step, new_iter_args, new_body,
                                     main_return_vars, sp, op->kind_, new_attrs, op->leading_comments_);
  }

  /**
   * @brief Build the tail as a bare `SeqStmts` of `k_clones` cloned bodies at
   *        offsets `base_index + j*step` (j in [0, k_clones)).
   *
   * Iter-args of the source loop are substituted directly with `iter_init_values`
   * for the first clone and with the previous clone's yields for subsequent
   * clones. Callers thread loop-carried state explicitly — either by wiring
   * `final_yields` into the enclosing IfStmt's `YieldStmt` (dynamic cascade) or
   * by emitting `AssignStmt`s that bind the outer loop's `return_vars` to the
   * final yields (static tail after a main loop).
   */
  ReplicatedRegion BuildTailSeq(const ForStmtPtr& op, const StmtPtr& body, int64_t k_clones, int64_t step,
                                const ExprPtr& base_index, const std::vector<ExprPtr>& iter_init_values) {
    return ReplicateBody(op, body, k_clones, step, base_index, iter_init_values);
  }

  /**
   * @brief Static lowering: compile-time trip count → main loop + (optional)
   *        bare-SeqStmts tail with exactly rem_iters clones, flattened into the
   *        outer scope. No dispatch needed because the remainder count is known.
   *
   * When iter_args are present, the main loop's return_vars forward loop-carried
   * state to the tail clones as their iter_arg substitutes; the tail's final
   * yields bind to the outer loop's `return_vars` via trailing `AssignStmt`s so
   * downstream references to those vars stay valid.
   */
  StmtPtr LowerStatic(const ForStmtPtr& op, const StmtPtr& body, int64_t factor, int64_t start, int64_t stop,
                      int64_t step) {
    int64_t trip = ComputeStaticTripCount(start, stop, step);
    if (trip == 0) {
      return DemoteToSequential(op, body);
    }
    int64_t main_iters = trip / factor;
    int64_t rem_iters = trip % factor;
    bool has_tail = rem_iters > 0;

    // Main loop's return_vars forward to the tail via fresh names when a tail
    // follows; otherwise they terminate at the original names. When there are
    // no return_vars, fresh-rename is a no-op (both empty), so guard that.
    std::vector<VarPtr> main_return_vars = op->return_vars_;
    if (has_tail && !main_return_vars.empty()) {
      main_return_vars = MakeFreshReturnVars(op->return_vars_, "_main");
    }

    std::vector<StmtPtr> result;
    if (main_iters > 0) {
      ExprPtr main_start = op->start_;
      ExprPtr main_stop = MakeConstIndex(start + main_iters * factor * step, op->span_);
      result.push_back(BuildMainLoop(op, body, factor, step, main_start, main_stop, main_return_vars));
    }
    if (has_tail) {
      int64_t tail_base = start + main_iters * factor * step;
      ExprPtr base_index = MakeConstIndex(tail_base, op->span_);
      // Tail iter_args seed from main_return_vars when a main loop precedes
      // the tail; otherwise (trip < factor) they seed from the original loop's
      // init_values.
      std::vector<ExprPtr> tail_init_values =
          (main_iters > 0) ? ReturnVarsAsExprs(main_return_vars) : InitValueExprs(op->iter_args_);
      auto region = BuildTailSeq(op, body, rem_iters, step, base_index, tail_init_values);
      // Push the bare SeqStmts of clones — SeqStmts::Flatten will splice them
      // directly into the outer result sequence.
      result.push_back(region.body);
      // Bind the original loop's return_vars to the tail's final yields so
      // downstream references to op->return_vars_ remain valid.
      for (size_t j = 0; j < op->return_vars_.size(); ++j) {
        result.push_back(
            std::make_shared<AssignStmt>(op->return_vars_[j], region.final_yields[j], op->span_));
      }
    }
    return SeqStmts::Flatten(std::move(result), op->span_);
  }

  /**
   * @brief Dynamic lowering: start and/or stop are runtime Exprs. Emits:
   *
   *   trip_iters    = ceil_div(stop - start, step)
   *   main_iters    = trip_iters / factor                       (compile-time: `/ factor`)
   *   main_end      = start + main_iters * (factor * step)      (SSA-bound to `unroll_main_end`)
   *   for i in range(start, main_end, F*step): <F clones>
   *   rem_iters     = trip_iters - main_iters * factor          (SSA-bound to `unroll_rem`)
   *   if rem_iters == 1: <1 clone>      # outermost
   *   else if rem_iters == 2: <2 clones>
   *   else ...
   *   else if rem_iters == F-1: <F-1 clones>
   *   # rem_iters == 0 matches no branch → tail is skipped.
   *
   * Dynamic bounds require step > 0; negative-step dynamic ranges are not in
   * the first-cut scope (static bounds handle negative step via
   * ComputeStaticTripCount).
   */
  StmtPtr LowerDynamic(const ForStmtPtr& op, const StmtPtr& body, int64_t factor, int64_t step) {
    Span sp = op->span_;
    INTERNAL_CHECK_SPAN(step > 0, sp) << "LowerPipelineLoops: dynamic bounds require a positive step, got "
                                      << step << ". Use static bounds for negative-step loops.";

    // trip_iters = ceil_div(stop - start, step). For step == 1 the ceil_div
    // collapses to (stop - start), so skip the `+ (step-1)` / `// step` wrapping
    // to keep the emitted IR minimal.
    ExprPtr span_expr = MakeSub(op->stop_, op->start_, sp);
    ExprPtr trip_expr;
    if (step == 1) {
      trip_expr = span_expr;
    } else {
      ExprPtr step_expr = MakeConstIndex(step, sp);
      ExprPtr adjusted = MakeAdd(span_expr, MakeConstIndex(step - 1, sp), sp);
      trip_expr = MakeFloorDiv(adjusted, step_expr, sp);
    }

    ExprPtr factor_expr = MakeConstIndex(factor, sp);
    ExprPtr main_iters_expr = MakeFloorDiv(trip_expr, factor_expr, sp);

    ExprPtr chunk = MakeConstIndex(factor * step, sp);
    ExprPtr scaled = MakeMul(main_iters_expr, chunk, sp);
    ExprPtr main_end_value = MakeAdd(op->start_, scaled, sp);

    VarPtr main_end_var =
        std::make_shared<Var>("unroll_main_end", std::make_shared<ScalarType>(DataType::INDEX), sp);
    auto main_end_assign = std::make_shared<AssignStmt>(main_end_var, main_end_value, sp);

    // The cascade always lives after the main loop, so the main loop's
    // return_vars forward state to the IfStmt cascade and need fresh names.
    std::vector<VarPtr> main_return_vars =
        op->return_vars_.empty() ? op->return_vars_ : MakeFreshReturnVars(op->return_vars_, "_main");

    // Main loop — stop is the fresh SSA var `main_end_var`.
    StmtPtr main_loop = BuildMainLoop(op, body, factor, step, /*main_start=*/op->start_,
                                      /*main_stop=*/main_end_var, main_return_vars);

    // rem_iters = trip_iters - main_iters * factor. For step == 1 this equals
    // stop - main_end (since trip == stop - start and main_iters*factor*step ==
    // main_end - start collapse), which keeps the emitted IR simple for the
    // common case.
    VarPtr rem_var = std::make_shared<Var>("unroll_rem", std::make_shared<ScalarType>(DataType::INDEX), sp);
    ExprPtr rem_value = (step == 1) ? MakeSub(op->stop_, main_end_var, sp)
                                    : MakeSub(trip_expr, MakeMul(main_iters_expr, factor_expr, sp), sp);
    auto rem_assign = std::make_shared<AssignStmt>(rem_var, rem_value, sp);

    // Fall-through (rem == 0) state expressions — the main loop's return_vars
    // passed through unchanged. Used as the innermost else's YieldStmt and as
    // the seed for each branch's tail iter_args.
    std::vector<ExprPtr> main_return_exprs = ReturnVarsAsExprs(main_return_vars);
    bool has_iter_args = !op->iter_args_.empty();

    // Build the cascade from innermost (k = factor-1) outward so each outer
    // IfStmt's else points at the previously-built IfStmt. With iter_args,
    // every IfStmt carries return_vars (fresh at inner levels, the original
    // outer return_vars at the outermost level) and every branch ends with a
    // YieldStmt — including the innermost else, which yields main_return_exprs
    // for the rem == 0 case. Each branch body is a bare SeqStmts of k cloned
    // bodies (the IfStmt provides the enclosing scope that declares its
    // return_vars; no inner ForStmt wrapper is required).
    std::optional<StmtPtr> inner;
    std::vector<VarPtr> inner_return_vars;
    for (int64_t k = factor - 1; k >= 1; --k) {
      // Each branch's tail clones seed iter-arg uses with the main-loop's
      // return_vars directly: the cascade is a dispatch on `rem`, so every
      // live branch starts from the same post-main state.
      auto region = BuildTailSeq(op, body, k, step, main_end_var, main_return_exprs);

      std::vector<StmtPtr> then_parts = {region.body};
      if (has_iter_args) {
        then_parts.push_back(std::make_shared<YieldStmt>(region.final_yields, sp));
      }
      auto then_body = SeqStmts::Flatten(std::move(then_parts), sp);

      std::optional<StmtPtr> else_body;
      if (k == factor - 1) {
        // Innermost: rem == 0 fall-through yields the main-loop state.
        if (has_iter_args) else_body = std::make_shared<YieldStmt>(main_return_exprs, sp);
      } else {
        INTERNAL_CHECK_SPAN(inner.has_value(), sp)
            << "Internal error: inner IfStmt must be built by the previous iteration";
        std::vector<StmtPtr> else_parts = {*inner};
        if (has_iter_args) {
          else_parts.push_back(std::make_shared<YieldStmt>(ReturnVarsAsExprs(inner_return_vars), sp));
        }
        else_body = SeqStmts::Flatten(std::move(else_parts), sp);
      }

      // return_vars: original names at the outermost level (k == 1); fresh at inner levels.
      std::vector<VarPtr> my_return_vars;
      if (has_iter_args) {
        my_return_vars =
            (k == 1) ? op->return_vars_ : MakeFreshReturnVars(op->return_vars_, "_rem" + std::to_string(k));
      }

      ExprPtr cond = MakeEq(rem_var, MakeConstIndex(k, sp), sp);
      auto if_stmt = std::make_shared<IfStmt>(cond, then_body, else_body, my_return_vars, sp);
      inner = StmtPtr(if_stmt);
      inner_return_vars = std::move(my_return_vars);
    }

    std::vector<StmtPtr> result;
    result.push_back(main_end_assign);
    result.push_back(main_loop);
    if (inner.has_value()) {
      result.push_back(rem_assign);
      result.push_back(*inner);
    }
    return SeqStmts::Flatten(std::move(result), sp);
  }
};

FunctionPtr TransformLowerPipelineLoops(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "LowerPipelineLoops cannot run on null function";
  LowerPipelineMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) return func;
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass LowerPipelineLoops() {
  return CreateFunctionPass(TransformLowerPipelineLoops, "LowerPipelineLoops", kLowerPipelineLoopsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

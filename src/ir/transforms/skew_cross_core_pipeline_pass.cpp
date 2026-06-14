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
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

using Attrs = std::vector<std::pair<std::string, std::any>>;

namespace {

using transform_utils::FlattenToStmts;
using transform_utils::GetCallFromStmt;

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
  throw pypto::ValueError("SkewCrossCorePipeline: " + what +
                          " must be a compile-time integer constant, got " + expr->TypeName());
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

// ---------------------------------------------------------------------------
// Cross-core software-pipeline (prologue / steady / epilogue skew).
//
// The sole lowering for a mixed cube/vector pipeline loop (the legacy
// unroll+IO-cluster cross-core path has been removed). EVERY bidirectional
// cross-core loop (body produces a tile to the peer core via tile.tpush_* and
// consumes the peer's reply via tile.tpop_*) leaves this pass as
// ForKind::Sequential, so no cross-core loop ever reaches LowerPipelineLoops or
// CanonicalizeIOOrder as a Pipeline loop:
//  - SINGLE round-trip, producer role (exactly one tpush + one tpop, the tpush's
//    backward slice does not feed the body via SSA): the two halves are linked
//    only by the in-order cross-core FIFO, so run the producer one iteration
//    AHEAD — produce(start) prologue, a KEPT Sequential steady ForStmt whose loop
//    var k indexes the produce and pairs produce(k) with the trailing consume(k-step)
//    over k in [start+step, start+trip*step), and a consume(last) epilogue. This lets
//    the cube issue iteration k's QK while the vector runs iteration k-step's softmax.
//  - CONSUMER role / MULTI round-trip / not statically skewable: demote to a plain
//    Sequential loop — order-preserving; overlap comes from the peer core's
//    producer skew.
// ---------------------------------------------------------------------------

bool IsTpushStmt(const StmtPtr& s) {
  auto call = GetCallFromStmt(s);
  if (!call) return false;
  const std::string& n = call->op_->name_;
  return n == "tile.tpush_to_aiv" || n == "tile.tpush_to_aic";
}

bool IsTpopStmt(const StmtPtr& s) {
  auto call = GetCallFromStmt(s);
  if (!call) return false;
  const std::string& n = call->op_->name_;
  return n == "tile.tpop_from_aiv" || n == "tile.tpop_from_aic";
}

/// Collect every Var *used* (RHS references) in a statement — LHS def of an
/// AssignStmt is deliberately skipped (mirrors CanonicalizeTileSlice's collector).
class VarUseCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> used;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override { VisitExpr(op->value_); }
  void VisitVarLike_(const VarPtr& op) override {
    used.insert(op.get());
    IRVisitor::VisitVarLike_(op);
  }
};

/// True when @p body (after stripping any trailing yield) contains BOTH a
/// cross-core tpush and a tpop — i.e. a bidirectional cross-core loop body.
bool BodyHasCrossCorePair(const StmtPtr& body) {
  bool has_push = false, has_pop = false;
  for (const auto& s : FlattenToStmts(SplitBodyYield(body).first)) {
    has_push |= IsTpushStmt(s);
    has_pop |= IsTpopStmt(s);
  }
  return has_push && has_pop;
}

/**
 * @brief Mutator that software-pipelines (skews) mixed cube/vector cross-core
 *        `pl.pipeline(N, stage=F)` loops (`ForKind::Pipeline` + `pipeline_stages
 *        == F`, `F > 1`); runs immediately before `LowerPipelineLoops`.
 *
 * For a loop whose body has BOTH a cross-core `tile.tpush_*` and a `tile.tpop_*`:
 * a statically-skewable single-round-trip producer loop is rewritten to a prologue
 * + Sequential steady ForStmt + epilogue; any other cross-core loop (consumer role,
 * multi-round-trip, dynamic bounds, trip < 2) is demoted to a plain
 * `ForKind::Sequential` loop. Either way the result is `ForKind::Sequential` with
 * NO `pipeline_stages` marker, so EVERY cross-core loop leaves this pass as
 * Sequential and never reaches `LowerPipelineLoops` (trigger `kind == Pipeline`) or
 * `CanonicalizeIOOrder` (scoped to Pipeline bodies) — neither of which carries any
 * cross-core handling anymore.
 *
 * NON-cross-core pipeline loops (same-core GM->L1 / L1->L0 / nested matmul stage
 * loops) are left intact as `ForKind::Pipeline` for `LowerPipelineLoops` to
 * replicate.
 *
 * Idempotency: the steady loop and the demoted loop are both `Sequential` with no
 * `pipeline_stages` attr, so re-running this pass finds no trigger.
 */
class SkewCrossCoreMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (op->kind_ != ForKind::Pipeline || !op->HasAttr(kPipelineStagesAttr)) {
      return IRMutator::VisitStmt_(op);
    }
    int64_t factor = static_cast<int64_t>(op->GetAttr<int>(kPipelineStagesAttr, 0));
    INTERNAL_CHECK_SPAN(factor >= 1, op->span_)
        << "SkewCrossCorePipeline: pipeline_stages must be >= 1, got " << factor;

    // Recurse into the loop bounds and body first so nested cross-core pipelines
    // are skewed and any expr-level mutations are preserved before this pass
    // rewrites the loop kind.
    auto inner_start = VisitExpr(op->start_);
    auto inner_stop = VisitExpr(op->stop_);
    auto inner_step = VisitExpr(op->step_);
    auto inner_body = VisitStmt(op->body_);

    // factor == 1 (user `pl.pipeline(stage=1)` or a prior run's marker): nothing to
    // skew. A cross-core body must STILL be demoted to Sequential here so it never
    // reaches LowerPipelineLoops / CanonicalizeIOOrder as a Pipeline body; a
    // same-core stage=1 body is left as Pipeline (rebuilt only if a child changed).
    if (factor == 1) {
      if (BodyHasCrossCorePair(inner_body)) {
        return DemoteToSequential(op, inner_start, inner_stop, inner_step, inner_body);
      }
      return RebuildIfChanged(op, inner_start, inner_stop, inner_step, inner_body);
    }

    // Non-cross-core (no tpush/tpop pair) — leave the Pipeline loop intact for
    // LowerPipelineLoops to replicate (same-core GM->L1 / L1->L0 / matmul stages).
    if (!BodyHasCrossCorePair(inner_body)) {
      return RebuildIfChanged(op, inner_start, inner_stop, inner_step, inner_body);
    }

    // Cross-core: skew if statically skewable, otherwise demote to Sequential. A
    // cross-core loop must NEVER leave this pass as ForKind::Pipeline — the unroll
    // pass and CanonicalizeIOOrder no longer handle cross-core ops.
    int64_t step = GetConstIntValue(inner_step, "step");
    INTERNAL_CHECK_SPAN(step != 0, op->span_) << "SkewCrossCorePipeline: step cannot be zero";
    auto start_const = TryGetConstInt(inner_start);
    auto stop_const = TryGetConstInt(inner_stop);
    if (start_const.has_value() && stop_const.has_value()) {
      if (auto skewed = LowerSkewed(op, inner_body, *start_const, *stop_const, step)) {
        return skewed;
      }
    }
    return DemoteToSequential(op, inner_start, inner_stop, inner_step, inner_body);
  }

 private:
  /// Rebuild the loop with the recursed bounds/body, preserving kind and attrs.
  /// Returns `op` unchanged (identity fast path) when nothing changed.
  StmtPtr RebuildIfChanged(const ForStmtPtr& op, const ExprPtr& start, const ExprPtr& stop,
                           const ExprPtr& step, const StmtPtr& body) {
    if (start.get() == op->start_.get() && stop.get() == op->stop_.get() && step.get() == op->step_.get() &&
        body.get() == op->body_.get()) {
      return op;
    }
    auto rebuilt = MutableCopy(op);
    rebuilt->start_ = start;
    rebuilt->stop_ = stop;
    rebuilt->step_ = step;
    rebuilt->body_ = body;
    return rebuilt;
  }

  /// Cross-core SWP (prototype): migrate a mixed cube/vector pipeline loop off the
  /// unroll+IO-cluster style. One analysis (lead = backward slice of the FIRST
  /// cross-core op), two emissions keyed on whether the lead feeds the body via an
  /// SSA edge (`carried`):
  ///  - Producer role (AIC, lead = tpush, `carried` empty): the lead and body are
  ///    linked only by the in-order FIFO, so run the producer one iteration AHEAD
  ///    (prologue + steady ForStmt + epilogue), overlapping QK[k+1] with the
  ///    peer's softmax[k].
  ///  - Consumer role (AIV, lead = tpop whose tile the body consumes, `carried`
  ///    non-empty): demote to a plain Sequential loop. The peer's producer skew
  ///    already puts each tile in the FIFO a step early, so the in-order tpop does
  ///    not block — and this drops the unroll's back-to-back tpop. (A real
  ///    iter-arg prefetch is rejected: it breaks codegen's tpop->tfree slot
  ///    tracking and a blocking tpop issued early would just stall.)
  /// Returns nullptr when not skewable: not a bidirectional cross-core loop (needs
  /// both a tpush and a tpop), trip < 2, a degenerate lead/body split, or the lead
  /// consuming an iter_arg or a body-defined value (non-hoistable).
  StmtPtr LowerSkewed(const ForStmtPtr& op, const StmtPtr& body, int64_t start, int64_t stop, int64_t step) {
    Span sp = op->span_;
    int64_t trip = ComputeStaticTripCount(start, stop, step);
    if (trip < 2) return nullptr;

    // Strip the loop body's trailing YieldStmt (loop-carried iter_args). The yield
    // lives in the consumer half (after the stores), so split the remaining body.
    // Use a named pair (not a structured binding) so `body_yields` can be captured
    // by the clone_half lambda below without a C++20 extension.
    auto body_split = SplitBodyYield(body);
    const std::vector<ExprPtr>& body_yields = body_split.second;
    std::vector<StmtPtr> stmts = FlattenToStmts(body_split.first);
    // Structural invariants (guaranteed by earlier verified passes): one
    // return_var and one yielded value per iter_arg. Fail fast here rather than
    // index out of bounds when seeding the epilogue from the body's yields.
    INTERNAL_CHECK_SPAN(op->return_vars_.size() == op->iter_args_.size(), op->span_)
        << "SkewCrossCorePipeline: ForStmt return_vars and iter_args size mismatch";
    INTERNAL_CHECK_SPAN(op->iter_args_.empty() || body_yields.size() == op->iter_args_.size(), op->span_)
        << "SkewCrossCorePipeline: loop body must yield one value per iter_arg";
    // Lead = backward slice of the FIRST cross-core op in program order (tpush
    // OR tpop). For the producer-role core (AIC) that is the tpush (the QK chain
    // feeding the peer); for the consumer-role core (AIV) that is the tpop (the
    // prefetch of the peer's tile). Picking the lead by program order — not
    // "every tpush" — is what lets one algorithm skew both cores.
    int lead_idx = -1;
    int num_tpush = 0, num_tpop = 0;
    for (int i = 0; i < static_cast<int>(stmts.size()); ++i) {
      bool push = IsTpushStmt(stmts[i]);
      bool pop = IsTpopStmt(stmts[i]);
      if ((push || pop) && lead_idx < 0) {
        lead_idx = i;
      }
      num_tpush += push;
      num_tpop += pop;
    }
    // Conservative scope: a genuine bidirectional cross-core loop (the qk_pv
    // head loop). One-directional pipes fall back to uniform replication.
    if (lead_idx < 0 || num_tpush == 0 || num_tpop == 0) {
      return nullptr;
    }

    // Producer half = backward slice of every tpush (the QK chain feeding the
    // peer). tpop carries no SSA args, so the consumer half (tpop -> SV -> store
    // and its own index scalars) is everything outside the slice. The iter_args
    // (the mi/li/oi output tensors) are read+updated only in the consumer half,
    // so they thread sequentially through the consume clones — the producer
    // clones, run one iteration ahead, never touch them.
    // Map EVERY defined var to its top-level stmt index. Critically this must
    // include vars defined by nested ForStmt/IfStmt return_vars (e.g. the QK
    // matmul's L0 K-loop yields qk_raw) — indexing only AssignStmt defs would
    // miss them and the backward slice would drop the matmul, leaving a dangling
    // tpush(qk_raw) free var.
    std::unordered_map<const Var*, int> def_idx;
    for (int i = 0; i < static_cast<int>(stmts.size()); ++i) {
      for (const auto& v : transform_utils::CollectDefVars(stmts[i])) {
        def_idx[v.get()] = i;
      }
      // CollectDefVars recurses into a ForStmt/IfStmt body but does NOT record
      // its own return_vars — yet at this (head-loop body) level those ARE the
      // defs (e.g. the QK matmul's K-loop yields qk_raw via its return_var).
      if (auto f = As<ForStmt>(stmts[i])) {
        for (const auto& rv : f->return_vars_) {
          def_idx[rv.get()] = i;
        }
      }
      if (auto iff = As<IfStmt>(stmts[i])) {
        for (const auto& rv : iff->return_vars_) {
          def_idx[rv.get()] = i;
        }
      }
    }
    std::set<int> produce_set;
    std::vector<int> work = {lead_idx};
    while (!work.empty()) {
      int i = work.back();
      work.pop_back();
      if (!produce_set.insert(i).second) {
        continue;
      }
      VarUseCollector c;
      c.VisitStmt(stmts[i]);
      for (const Var* v : c.used) {
        auto it = def_idx.find(v);
        if (it != def_idx.end()) {
          work.push_back(it->second);
        }
      }
    }
    if (produce_set.empty() || produce_set.size() == stmts.size()) {
      return nullptr;
    }
    // Guard: a producer stmt must not reference an iter_arg (would break the
    // "producer is iter_arg-transparent" assumption the skew relies on).
    {
      std::unordered_set<const Var*> ia_set;
      for (const auto& ia : op->iter_args_) {
        ia_set.insert(ia.get());
      }
      for (int i : produce_set) {
        VarUseCollector c;
        c.VisitStmt(stmts[i]);
        for (const Var* v : c.used) {
          if (ia_set.count(v)) {
            return nullptr;
          }
        }
      }
    }

    std::set<int> consume_set;
    for (int i = 0; i < static_cast<int>(stmts.size()); ++i) {
      if (!produce_set.count(i)) {
        consume_set.insert(i);
      }
    }

    // Gather body-defined and body-used vars (incl. nested ForStmt/IfStmt
    // return_vars, which CollectDefVars does not record at this level).
    std::unordered_set<const Var*> body_defs, body_used;
    for (int i : consume_set) {
      for (const auto& v : transform_utils::CollectDefVars(stmts[i])) {
        body_defs.insert(v.get());
      }
      if (auto f = As<ForStmt>(stmts[i])) {
        for (const auto& rv : f->return_vars_) {
          body_defs.insert(rv.get());
        }
      }
      if (auto iff = As<IfStmt>(stmts[i])) {
        for (const auto& rv : iff->return_vars_) {
          body_defs.insert(rv.get());
        }
      }
      VarUseCollector c;
      c.VisitStmt(stmts[i]);
      body_used.insert(c.used.begin(), c.used.end());
    }

    // Reverse-direction guard: the lead runs one iteration AHEAD, so it must be
    // hoistable — it may not consume any body-defined value (that would be
    // circular). Lead-defs consumed BY the body are fine: they become the
    // `carried` set and thread through as extra iter_args below.
    for (int i : produce_set) {
      VarUseCollector c;
      c.VisitStmt(stmts[i]);
      for (const Var* v : c.used) {
        if (body_defs.count(v)) {
          return nullptr;
        }
      }
    }

    // Carried vars = lead-defined vars consumed by the body (the AIV's prefetched
    // scores tile). EMPTY for the AIC, whose lead feeds the peer only through the
    // FIFO — that case degenerates to the original FIFO-decoupled skew. Stable
    // order: ascending lead-stmt index (produce_set is a std::set), then def
    // order within a stmt.
    std::vector<VarPtr> carried;
    {
      std::unordered_set<const Var*> seen;
      auto add = [&](const VarPtr& v) {
        if (body_used.count(v.get()) && seen.insert(v.get()).second) {
          carried.push_back(v);
        }
      };
      for (int i : produce_set) {
        for (const auto& v : transform_utils::CollectDefVars(stmts[i])) {
          add(v);
        }
        if (auto f = As<ForStmt>(stmts[i])) {
          for (const auto& rv : f->return_vars_) {
            add(rv);
          }
        }
        if (auto iff = As<IfStmt>(stmts[i])) {
          for (const auto& rv : iff->return_vars_) {
            add(rv);
          }
        }
      }
    }

    // Fall back to a plain Sequential demotion (no skew, no unroll) for the two
    // shapes the producer-ahead skew cannot safely handle:
    //
    //  - MULTI-ROUND-TRIP (num_tpush != 1 || num_tpop != 1): more than one message
    //    per iteration on a cross-core FIFO direction. The skew advances ONLY the
    //    lead's message one iteration ahead; any other same-direction message stays
    //    put, which REORDERS the in-order cross-core FIFO (e.g. push p0[k+1] before
    //    p1[k]) — the peer then pops the wrong tile. This is a SILENT wrong-data bug
    //    because the property verifiers do not model FIFO order.
    // TODO(crosscore-skew): skew multi-round-trip loops (e.g. C->V->C->V) by
    // advancing every same-direction message one round-trip together so the FIFO
    // order is preserved; until that exists, demote to Sequential.
    //  - CONSUMER ROLE (`carried` non-empty, e.g. the AIV whose lead = tpop feeds
    //    the body via SSA): not a producer-ahead shape, and a true iter-arg-threaded
    //    prefetch is wrong anyway — it breaks codegen's tpop->tfree FIFO-slot
    //    tracking (keys on SSA var identity, cannot cross an iter_arg) and a blocking
    //    tpop issued a full iteration early just stalls.
    //
    // Sequential is order-preserving and off the unroll style; cross-core overlap
    // then comes from the PEER core's producer skew putting each tile in the FIFO a
    // step early, so the in-order tpop never blocks.
    if (num_tpush != 1 || num_tpop != 1 || !carried.empty()) {
      return DemoteToSequential(op, op->start_, op->stop_, op->step_, body);
    }

    // Producer-role single-round-trip cross-core loop (the AIC: exactly one tpush
    // + one tpop, lead = tpush, FIFO-decoupled from the body -> `carried` empty).
    // Clone the producer / consumer halves with
    // loop_var -> `lv_sub` and iter_args -> `iter_subs`. The two halves are cloned
    // with DIFFERENT loop_var substitutes (k vs k-step) — they are SSA-independent
    // (linked only by the in-order cross-core FIFO), so this is safe. A steady
    // ForStmt is KEPT (not fully unrolled) so the matmul Acc double-buffering
    // (running-acc / ping-pong addresses assigned by AllocateMemoryAddr) still has
    // a loop to alternate over.
    auto clone_half = [&](const std::set<int>& which, const ExprPtr& lv_sub,
                          const std::vector<ExprPtr>& iter_subs,
                          bool with_yield) -> std::pair<StmtPtr, std::vector<ExprPtr>> {
      std::vector<StmtPtr> sel;
      for (int i = 0; i < static_cast<int>(stmts.size()); ++i) {
        if (which.count(i)) {
          sel.push_back(stmts[i]);
        }
      }
      if (with_yield && !op->iter_args_.empty()) {
        sel.push_back(std::make_shared<YieldStmt>(body_yields, sp));
      }
      auto seq = std::make_shared<SeqStmts>(std::move(sel), sp);
      std::unordered_map<const Var*, ExprPtr> sub;
      sub[op->loop_var_.get()] = lv_sub;
      for (size_t j = 0; j < op->iter_args_.size(); ++j) {
        sub[op->iter_args_[j].get()] = iter_subs[j];
      }
      auto cloned = DeepClone(seq, sub, /*clone_def_vars=*/true);
      return SplitBodyYield(cloned.cloned_body);
    };

    std::vector<ExprPtr> init = InitValueExprs(op->iter_args_);

    // Prologue: produce(start) — primes the peer with iteration 0's tile so it
    // can start consuming while the steady loop computes iteration 1's produce.
    auto [prologue, _pl] = clone_half(produce_set, MakeConstIndex(start, sp), init, /*with_yield=*/false);

    // Steady loop: trip-1 iterations whose loop var k = start+step ..
    // start+(trip-1)*step indexes the PRODUCE (running one step ahead); the
    // CONSUME trails one step behind at k-step. Body = produce(k) ;
    // consume(k-step) ; yield(updated iter_args), overlapping the cube (QK of k)
    // with the vector (softmax of k-step). Equivalent to a produce-ahead skew
    // with the +step offset on the consume side, so the loop ranges over the
    // natural produce indices `[start+step, start+trip*step)` and `produce(k)`
    // uses the bare loop var.
    VarPtr new_lv = CloneLoopVar(op->loop_var_);
    std::vector<IterArgPtr> new_iter_args;
    std::vector<ExprPtr> steady_init_subs;
    for (const auto& ia : op->iter_args_) {
      auto fresh = MakeFreshIterArg(ia, ia->initValue_);
      new_iter_args.push_back(fresh);
      steady_init_subs.push_back(fresh);
    }
    auto [steady_prod, _sp] = clone_half(produce_set, new_lv, steady_init_subs, /*with_yield=*/false);
    auto [steady_cons, steady_yields] = clone_half(consume_set, MakeSub(new_lv, MakeConstIndex(step, sp), sp),
                                                   steady_init_subs, /*with_yield=*/true);
    std::vector<StmtPtr> steady_body_parts = {steady_prod, steady_cons};
    if (!op->iter_args_.empty()) {
      steady_body_parts.push_back(std::make_shared<YieldStmt>(steady_yields, sp));
    }
    auto steady_body = SeqStmts::Flatten(std::move(steady_body_parts), sp);

    std::vector<VarPtr> steady_rv =
        op->return_vars_.empty() ? op->return_vars_ : MakeFreshReturnVars(op->return_vars_, "_swp");
    auto steady_loop = std::make_shared<ForStmt>(
        new_lv, MakeConstIndex(start + step, sp), MakeConstIndex(start + trip * step, sp),
        MakeConstIndex(step, sp), new_iter_args, steady_body, steady_rv, sp, ForKind::Sequential);
    // Preserve loop metadata, stripping the pipeline marker (the steady loop is
    // Sequential): any non-pipeline attrs and leading comments carry through.
    steady_loop->attrs_ = StripAttr(op->attrs_, kPipelineStagesAttr);
    steady_loop->leading_comments_ = op->leading_comments_;

    // Epilogue: consume(start+(trip-1)*step), seeded from the steady loop's final
    // iter_args (or the loop's init values when there are no iter_args).
    std::vector<ExprPtr> epi_iter_subs = steady_rv.empty() ? init : ReturnVarsAsExprs(steady_rv);
    auto [epilogue, epi_yields] = clone_half(consume_set, MakeConstIndex(start + (trip - 1) * step, sp),
                                             epi_iter_subs, /*with_yield=*/true);

    std::vector<StmtPtr> result = {prologue, steady_loop, epilogue};
    // Bind the original loop's return_vars to the epilogue's final yields so the
    // enclosing block loop's iter_arg threading still sees the result.
    for (size_t j = 0; j < op->return_vars_.size(); ++j) {
      result.push_back(std::make_shared<AssignStmt>(op->return_vars_[j], epi_yields[j], sp));
    }
    return SeqStmts::Flatten(std::move(result), sp);
  }

  /// Run the cross-core loop SEQUENTIALLY (consumer-role / multi-round-trip case):
  /// keep the body as-is, demote kind to Sequential and strip `pipeline_stages`
  /// together so the bidirectional invariant `kind == Pipeline ⇔ pipeline_stages
  /// attr present` stays whole and the loop is not re-sorted by CanonicalizeIOOrder.
  StmtPtr DemoteToSequential(const ForStmtPtr& op, const ExprPtr& start, const ExprPtr& stop,
                             const ExprPtr& step, const StmtPtr& inner_body) {
    auto cleaned = MutableCopy(op);
    cleaned->start_ = start;
    cleaned->stop_ = stop;
    cleaned->step_ = step;
    cleaned->body_ = inner_body;
    cleaned->kind_ = ForKind::Sequential;
    cleaned->attrs_ = StripAttr(op->attrs_, kPipelineStagesAttr);
    return cleaned;
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
};

FunctionPtr TransformSkewCrossCorePipeline(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "SkewCrossCorePipeline cannot run on null function";
  SkewCrossCoreMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) return func;
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass SkewCrossCorePipeline() {
  return CreateFunctionPass(TransformSkewCrossCorePipeline, "SkewCrossCorePipeline",
                            kSkewCrossCorePipelineProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

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

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_allocator_policy.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/auto_tile_matmul_l0/internal.h"

namespace pypto {
namespace ir {
namespace auto_tile_matmul_l0 {

namespace {

// Keep the PyPTO-only recognizer disabled independently of PTOAS's chooser
// policy. Distinct Acc slots do not establish profitable compute/drain overlap;
// caller-owned explicit buffering remains unchanged.
constexpr bool kAutomaticPipelineDbcEnabled = false;

/// Static physical footprint of a tile, rounded exactly as the active memory
/// allocator rounds one buffer in @p space.  Unknown/dynamic shapes and
/// arithmetic overflow return nullopt: an automatic capacity decision must
/// never guess low.
std::optional<uint64_t> StaticAlignedTileBytes(const TileTypePtr& tile, MemorySpace space,
                                               const MemoryAllocatorPolicy& policy,
                                               const backend::BackendHandler* handler) {
  if (!tile || tile->GetMemorySpace() != space) return std::nullopt;
  auto bytes = utils::StaticPhysicalAllocationBytes(tile, space, handler);
  if (!bytes) return std::nullopt;
  const uint64_t aligned = policy.AlignAddress(*bytes, space);
  if (aligned < *bytes) return std::nullopt;  // alignment arithmetic overflow
  return aligned;
}

/// Whole-function conservative L0C inventory after accounting for pipeline
/// replication.  LowerPipelineLoops gives every non-cube Acc producer one
/// physical-membership request per source stage (and the product of the stage
/// depths under nested pipelines).  Counting only the pre-lowering SSA value
/// would therefore underestimate the placement that MemoryReuse may preserve.
///
/// Cube matmul accumulators are normally serialized and left untagged, so they
/// need one slot.  An already-marked dbC pipeline is conservatively charged at
/// its full source depth; a newly selected candidate is charged separately by
/// BuildPipelineDbCPlan as one existing slot plus one extra ping-pong slot.
///
/// This remains an intentional upper bound: sequential values and independent
/// pipeline groups may later coalesce, but the automatic dbC plan must not
/// force an existing pipeline to shed buffering depth merely because its
/// post-lowering multiplicity was omitted here.
class AccFootprintCollector : public IRVisitor {
 public:
  AccFootprintCollector(const MemoryAllocatorPolicy& policy, const backend::BackendHandler* handler)
      : policy_(policy), handler_(handler) {}

  bool valid = true;
  uint64_t total_bytes = 0;

 protected:
  void VisitVarLike_(const VarPtr& op) override { Record(op, /*copies=*/1); }

  void VisitStmt_(const AssignStmtPtr& op) override {
    uint64_t copies = 1;
    auto tile = As<TileType>(op->var_->GetType());
    if (tile && tile->GetMemorySpace() == MemorySpace::Acc) {
      auto call = As<Call>(op->value_);
      const bool is_cube_matmul = call && call->op_ && call->op_->name_.rfind("tile.matmul", 0) == 0;
      // The lowering tagger skips ordinary cube accumulators because the cube
      // serializes them. Every other Acc producer is replicated across all
      // enclosing source pipeline stages. An explicit dbC marker makes the
      // cube accumulator replicated too; charging the full depth is safe even
      // when CanonicalizeIOOrder later rotates it over only two slots.
      if (!is_cube_matmul || explicit_dbc_depth_ != 0) copies = pipeline_depth_;
    }
    Record(op->var_, copies);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    const uint64_t saved_pipeline_depth = pipeline_depth_;
    const int saved_explicit_dbc_depth = explicit_dbc_depth_;
    if (op && op->kind_ == ForKind::Pipeline) {
      const int stages = op->GetAttr<int>(kPipelineStagesAttr, 0);
      if (stages <= 0 ||
          pipeline_depth_ > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(stages)) {
        valid = false;
      } else {
        pipeline_depth_ *= static_cast<uint64_t>(stages);
      }
      if (op->GetAttr<bool>(kPipelineDoubleBufferCAttr, false)) ++explicit_dbc_depth_;
    }
    IRVisitor::VisitStmt_(op);
    pipeline_depth_ = saved_pipeline_depth;
    explicit_dbc_depth_ = saved_explicit_dbc_depth;
  }

 private:
  struct Entry {
    uint64_t bytes = 0;
    uint64_t copies = 0;
  };

  void Record(const VarPtr& var, uint64_t copies) {
    if (!valid || !var || copies == 0) return;
    auto tile = As<TileType>(var->GetType());
    if (!tile || tile->GetMemorySpace() != MemorySpace::Acc) return;
    auto bytes = StaticAlignedTileBytes(tile, MemorySpace::Acc, policy_, handler_);
    if (!bytes || copies > std::numeric_limits<uint64_t>::max() / *bytes) {
      valid = false;
      return;
    }
    auto [it, inserted] = entries_.try_emplace(var.get(), Entry{*bytes, 0});
    if (!inserted && it->second.bytes != *bytes) {
      valid = false;
      return;
    }
    if (copies <= it->second.copies) return;
    const uint64_t added_copies = copies - it->second.copies;
    if (added_copies > std::numeric_limits<uint64_t>::max() / *bytes) {
      valid = false;
      return;
    }
    const uint64_t added_bytes = added_copies * *bytes;
    if (total_bytes > std::numeric_limits<uint64_t>::max() - added_bytes) {
      valid = false;
      return;
    }
    it->second.copies = copies;
    total_bytes += added_bytes;
  }

  const MemoryAllocatorPolicy& policy_;
  const backend::BackendHandler* handler_ = nullptr;
  uint64_t pipeline_depth_ = 1;
  int explicit_dbc_depth_ = 0;
  std::unordered_map<const Var*, Entry> entries_;
};

/// True for any call whose result occupies L0C. This future-proofs the
/// recognizer against new MAD-family operations: a candidate body may have one
/// Acc producer in total, and that producer must be the plain matmul selected
/// below.
bool ProducesAcc(const CallPtr& call) {
  auto tile = call ? As<TileType>(call->GetType()) : nullptr;
  return tile && tile->GetMemorySpace() == MemorySpace::Acc;
}

/// A per-iteration Mat->L0 transfer that the pipeline already replicates.  This
/// proves the "moving" side of the stationary-panel pattern; an arbitrary local
/// assignment is not sufficient evidence that the operand is a prefetchable
/// pipeline input.
bool IsMovingMatmulOperandProducer(const CallPtr& call, MemorySpace target_space) {
  if (!call || call->args_.empty()) return false;
  auto out = As<TileType>(call->GetType());
  auto source = As<TileType>(call->args_[0]->GetType());
  if (!out || out->GetMemorySpace() != target_space || !source ||
      source->GetMemorySpace() != MemorySpace::Mat) {
    return false;
  }
  return IsOp(call, "tile.extract") || IsOp(call, "tile.move");
}

enum class PipelineAccumulatorDrainPath {
  DirectGm,
  MatScratch,
};

struct PipelineAccumulatorCandidate {
  uint64_t aligned_acc_bytes = 0;
  int64_t trip_count = 0;
  PipelineAccumulatorDrainPath drain_path = PipelineAccumulatorDrainPath::DirectGm;
};

/// Two L0C slots need at least two complete compute/drain pairs before their
/// fill/drain bubble is reliably amortized.  The device sweep for #2131 found
/// the two-iteration (one-pair) form tied, while every direct-store case with
/// four or more iterations won across M/N/K, operand side, and Acc size.
constexpr int64_t kMinAutoDirectGmDbCTripCount = 4;

/// Acc->Mat has a cheaper drain and therefore needs more work and a larger
/// accumulator before the overlap repays the two-slot schedule.  Device
/// calibration found 8/16 KiB Mat-scratch accumulators regressed or tied while
/// independent 32/40 KiB cases won at eight iterations.  Keep this path's
/// first automatic region deliberately conservative and backend-relative:
/// one Acc tile must occupy at least one quarter of L0C, and the loop must
/// provide at least four complete compute/drain pairs.
constexpr int64_t kMinAutoMatScratchDbCTripCount = 8;
constexpr uint64_t kMinAutoMatScratchAccL0cDenominator = 4;

bool IsPipelineAccumulatorProfitable(const PipelineAccumulatorCandidate& candidate, uint64_t l0c_bytes) {
  if (candidate.drain_path == PipelineAccumulatorDrainPath::DirectGm) {
    return candidate.trip_count >= kMinAutoDirectGmDbCTripCount;
  }
  const uint64_t min_acc_bytes = l0c_bytes / kMinAutoMatScratchAccL0cDenominator +
                                 (l0c_bytes % kMinAutoMatScratchAccL0cDenominator != 0);
  return candidate.trip_count >= kMinAutoMatScratchDbCTripCount &&
         candidate.aligned_acc_bytes >= min_acc_bytes;
}

/// Recognize an already-L0, directly-drained matmul in a software pipeline and
/// return the extra accumulator slot it would need.
///
/// This is the user-authored-pipeline counterpart of the M/N tiler's dbC mode:
///
///   for n in pl.pipeline(..., stage=2):
///     b_l0 = ... Right
///     c_l0 = tile.matmul(a_l0, b_l0)  # Acc
///     out  = tile.store(c_l0, ..., out)
///
/// Success proves the local schedule and dataflow.  Whole-function L0C capacity
/// is checked separately by ``BuildPipelineDbCPlan`` before any loop is marked.
/// ``LowerPipelineLoops`` gives adjacent Acc clones two rotating memberships,
/// and ``CanonicalizeIOOrder`` emits depth-two chunks
/// ``matmul, matmul, drain, drain``.  A user-selected pipeline depth greater
/// than two still controls operand prefetch depth; L0C remains a two-slot
/// ping-pong rather than growing to one accumulator per stage.
///
/// Keep the recognition deliberately conservative:
///   * static pipeline with stage >= 2 and a trip count divisible by the stage
///     count (no separately lowered tail group);
///   * no nested control flow in the candidate body;
///   * exactly one cube MAD, and it is plain ``tile.matmul`` over Left/Right;
///   * the selected moving operand is a recognized per-iteration Mat->L0
///     transfer; the stationary operand is defined outside the loop and is not
///     an IterArg;
///   * no other Acc definition/read or store-like operation in the body;
///   * one canonical loop-carried ``tile.store`` or ``tile.assemble`` chain:
///     the drain targets IterArg i and its result is yielded at index i.
///
/// The returned path, trip count, and Acc footprint are filtered by
/// ``IsPipelineAccumulatorProfitable`` after the backend L0C capacity is known.
///
/// The direct-body scans are disjoint across nested loops, so this remains
/// linear in program size.
std::optional<PipelineAccumulatorCandidate> AnalyzePipelineAccumulator(
    const ForStmtPtr& loop, const MemoryAllocatorPolicy& policy, const backend::BackendHandler* handler) {
  const int stages = loop ? loop->GetAttr<int>(kPipelineStagesAttr, 0) : 0;
  const int64_t trip_count = loop ? transform_utils::EvalConstTripCount(loop).value_or(0) : -1;
  if (!loop || loop->kind_ != ForKind::Pipeline || loop->HasAttr(kPipelineOverlapStoresAttr) ||
      loop->HasAttr(kPipelineDoubleBufferCAttr) || stages < 2 || trip_count < stages ||
      trip_count % stages != 0) {
    return std::nullopt;
  }

  std::vector<StmtPtr> body_stmts;
  if (auto seq = As<SeqStmts>(loop->body_)) {
    body_stmts = seq->stmts_;
  } else {
    body_stmts.push_back(loop->body_);
  }

  AssignStmtPtr candidate;
  CallPtr candidate_call;
  std::unordered_set<const Var*> direct_defs;
  std::unordered_map<const Var*, CallPtr> direct_call_defs;
  int acc_producers = 0;
  for (const auto& stmt : body_stmts) {
    // A nested region could contain another use or cube operation that needs a
    // joint schedule. Defer instead of trying to summarize it locally.
    if (As<ForStmt>(stmt) || As<IfStmt>(stmt) || As<WhileStmt>(stmt) || As<ScopeStmt>(stmt)) {
      return std::nullopt;
    }
    auto assign = As<AssignStmt>(stmt);
    if (assign) direct_defs.insert(assign->var_.get());
    auto call = transform_utils::GetCallFromStmt(stmt);
    if (assign && call) direct_call_defs.emplace(assign->var_.get(), call);
    if (!ProducesAcc(call)) continue;
    ++acc_producers;
    if (assign && IsOp(call, "tile.matmul")) {
      candidate = assign;
      candidate_call = call;
    }
  }
  if (acc_producers != 1 || !candidate || !candidate_call || candidate_call->args_.size() != 2) {
    return std::nullopt;
  }

  auto lhs = AsVarLike(candidate_call->args_[0]);
  auto rhs = AsVarLike(candidate_call->args_[1]);
  auto lhs_ty = lhs ? As<TileType>(lhs->GetType()) : nullptr;
  auto rhs_ty = rhs ? As<TileType>(rhs->GetType()) : nullptr;
  int64_t M = 0;
  int64_t K_lhs = 0;
  int64_t K_rhs = 0;
  int64_t N = 0;
  if (!IsStatic2DInSpaces(lhs_ty, {MemorySpace::Left}, M, K_lhs) ||
      !IsStatic2DInSpaces(rhs_ty, {MemorySpace::Right}, K_rhs, N) || K_lhs != K_rhs) {
    return std::nullopt;
  }

  std::unordered_set<const Var*> iter_args;
  for (const auto& iter_arg : loop->iter_args_) iter_args.insert(iter_arg.get());
  // Match the stationary-panel construction from #2131.  A direct definition
  // proves movement only when it is the result of a Mat->Left/Right transfer;
  // the other operand must be a true outside-loop value, not loop-carried state.
  const bool lhs_moves = direct_defs.count(lhs.get()) != 0;
  const bool rhs_moves = direct_defs.count(rhs.get()) != 0;
  if (lhs_moves == rhs_moves) return std::nullopt;
  const VarPtr& moving = lhs_moves ? lhs : rhs;
  const VarPtr& stationary = lhs_moves ? rhs : lhs;
  const MemorySpace moving_space = lhs_moves ? MemorySpace::Left : MemorySpace::Right;
  auto producer = direct_call_defs.find(moving.get());
  if (producer == direct_call_defs.end() || !IsMovingMatmulOperandProducer(producer->second, moving_space) ||
      direct_defs.count(stationary.get()) != 0 || iter_args.count(stationary.get()) != 0) {
    return std::nullopt;
  }

  auto acc_ty = As<TileType>(candidate->var_->GetType());
  int64_t acc_m = 0;
  int64_t acc_n = 0;
  if (!IsStatic2DInSpaces(acc_ty, {MemorySpace::Acc}, acc_m, acc_n) || acc_m != M || acc_n != N) {
    return std::nullopt;
  }
  auto acc_bytes = StaticAlignedTileBytes(acc_ty, MemorySpace::Acc, policy, handler);
  if (!acc_bytes) return std::nullopt;

  // Count all uses in the direct body, excluding assignment LHS definitions.
  // Compound statements were rejected above, so this scan cannot descend into
  // a child region and re-count work owned by another loop.
  SiblingUseCounter uses;
  for (const auto& stmt : body_stmts) uses.VisitStmt(stmt);
  auto use_it = uses.counts.find(candidate->var_.get());
  if (use_it == uses.counts.end() || use_it->second != 1) return std::nullopt;
  // Any other Acc read is loop-carried/external L0C state whose lifetime and
  // aliasing would be changed by the loop-wide drain reorder.
  for (const auto& [raw, var] : uses.vars) {
    if (raw == candidate->var_.get()) continue;
    auto tile = As<TileType>(var->GetType());
    if (tile && tile->GetMemorySpace() == MemorySpace::Acc) return std::nullopt;
  }
  // Likewise reject every other Acc definition, including non-MAD data
  // movement such as tile.extract(..., target_memory=Acc).
  for (const auto& stmt : body_stmts) {
    auto assign = As<AssignStmt>(stmt);
    if (!assign || assign.get() == candidate.get()) continue;
    auto tile = As<TileType>(assign->var_->GetType());
    if (tile && tile->GetMemorySpace() == MemorySpace::Acc) return std::nullopt;
  }

  AssignStmtPtr drain_assign;
  CallPtr drain_call;
  int store_like_calls = 0;
  for (const auto& stmt : body_stmts) {
    auto assign = As<AssignStmt>(stmt);
    auto call = transform_utils::GetCallFromStmt(stmt);
    if (!call) continue;
    const bool is_store = IsOp(call, "tile.store");
    const bool is_assemble = IsOp(call, "tile.assemble");
    if (!is_store && !is_assemble && !IsOp(call, "tile.write")) continue;
    ++store_like_calls;
    if (!assign || call->args_.size() != 3) return std::nullopt;
    const size_t source_index = is_assemble ? 1 : 0;
    auto source = AsVarLike(call->args_[source_index]);
    if (!source || source.get() != candidate->var_.get()) return std::nullopt;
    if (drain_assign) return std::nullopt;
    drain_assign = assign;
    drain_call = call;
  }
  if (store_like_calls != 1 || !drain_assign || !drain_call) return std::nullopt;

  // Canonical loop-carried drain: target IterArg i -> drain result -> yield i.
  auto yield = transform_utils::GetLastYieldStmt(loop->body_);
  if (!yield || yield->value_.size() != loop->iter_args_.size()) return std::nullopt;
  std::optional<size_t> yield_index;
  for (size_t i = 0; i < yield->value_.size(); ++i) {
    auto value = AsVarLike(yield->value_[i]);
    if (value && value.get() == drain_assign->var_.get()) {
      if (yield_index) return std::nullopt;
      yield_index = i;
    }
  }
  if (!yield_index || *yield_index >= loop->iter_args_.size()) return std::nullopt;
  const bool is_assemble = IsOp(drain_call, "tile.assemble");
  const size_t target_index = is_assemble ? 0 : 2;
  auto target = AsVarLike(drain_call->args_[target_index]);
  if (!target || target.get() != loop->iter_args_[*yield_index].get()) return std::nullopt;
  if (uses.counts[drain_assign->var_.get()] != 1 || uses.counts[target.get()] != 1) {
    return std::nullopt;
  }
  if (is_assemble) {
    auto target_ty = As<TileType>(target->GetType());
    auto result_ty = As<TileType>(drain_assign->var_->GetType());
    if (!target_ty || target_ty->GetMemorySpace() != MemorySpace::Mat || !result_ty ||
        result_ty->GetMemorySpace() != MemorySpace::Mat) {
      return std::nullopt;
    }
  }
  return PipelineAccumulatorCandidate{
      *acc_bytes, trip_count,
      is_assemble ? PipelineAccumulatorDrainPath::MatScratch : PipelineAccumulatorDrainPath::DirectGm};
}

}  // namespace

/// Select all structurally eligible loops only when their worst-case combined
/// L0C footprint fits.  The base footprint counts every Acc SSA value once; the
/// extra term adds one slot per loop this pass will mark.  This intentionally
/// over-approximates liveness but guarantees that the loop-wide schedule change
/// cannot turn a previously fitting function into an L0C overflow.
std::unordered_set<const ForStmt*> BuildPipelineDbCPlan(const FunctionPtr& func) {
  // This recognizer historically targets PyPTO only. Restoring PTOAS chooser
  // enumeration does not enable automatic annotation of existing L0 pipelines.
  if (!kAutomaticPipelineDbcEnabled) return {};
  const auto* ctx = PassContext::Current();
  const MemoryPlanner planner = ctx ? ctx->GetMemoryPlanner() : MemoryPlanner::PyPTO;
  // #2131 explicitly targets the PyPTO planner. PTOAS already gives the
  // reproduced loop four distinct Acc placements and showed no measurable
  // benefit from this source-level marker.
  if (planner != MemoryPlanner::PyPTO) return {};

  // Profitability and capacity are backend-specific. Direct pass invocation
  // without a configured backend must leave an already-L0 pipeline unchanged,
  // just as it did before this recognizer existed.
  if (!pypto::backend::BackendConfig::IsConfigured()) return {};

  auto policy = pypto::backend::GetBackend()->CreateMemoryAllocatorPolicy();
  if (!policy) return {};

  const auto* handler = ctx ? ctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
  const uint64_t l0c_bytes = handler ? handler->GetCapacityBytes(MemorySpace::Acc) : 0;
  if (!handler || l0c_bytes == 0) return {};

  AccFootprintCollector footprint(*policy, handler);
  footprint.VisitFunction(func);
  if (!footprint.valid) return {};

  class CandidateCollector : public IRVisitor {
   public:
    CandidateCollector(const MemoryAllocatorPolicy& policy, const backend::BackendHandler* handler)
        : policy_(policy), handler_(handler) {}
    std::unordered_map<const ForStmt*, PipelineAccumulatorCandidate> candidates;

   protected:
    void VisitStmt_(const ForStmtPtr& op) override {
      if (auto candidate = AnalyzePipelineAccumulator(op, policy_, handler_)) {
        candidates.emplace(op.get(), *candidate);
      }
      IRVisitor::VisitStmt_(op);
    }

   private:
    const MemoryAllocatorPolicy& policy_;
    const backend::BackendHandler* handler_ = nullptr;
  } candidates(*policy, handler);
  candidates.VisitStmt(func->body_);
  if (candidates.candidates.empty()) return {};

  for (auto it = candidates.candidates.begin(); it != candidates.candidates.end();) {
    if (!IsPipelineAccumulatorProfitable(it->second, l0c_bytes)) {
      it = candidates.candidates.erase(it);
    } else {
      ++it;
    }
  }
  if (candidates.candidates.empty()) return {};

  uint64_t worst_case = footprint.total_bytes;
  if (ctx && ctx->GetEnablePyptoL0cDoubleBuffer()) {
    // The chooser may also double-buffer any other Acc result in this function;
    // reserve a second slot for the full inventory rather than trying to
    // duplicate its profitability decision here.
    if (worst_case > std::numeric_limits<uint64_t>::max() - footprint.total_bytes) return {};
    worst_case += footprint.total_bytes;
  } else {
    for (const auto& entry : candidates.candidates) {
      const auto& candidate = entry.second;
      if (worst_case > std::numeric_limits<uint64_t>::max() - candidate.aligned_acc_bytes) return {};
      worst_case += candidate.aligned_acc_bytes;
    }
  }
  if (worst_case > l0c_bytes) return {};

  std::unordered_set<const ForStmt*> plan;
  for (const auto& entry : candidates.candidates) plan.insert(entry.first);
  return plan;
}

}  // namespace auto_tile_matmul_l0
}  // namespace ir
}  // namespace pypto

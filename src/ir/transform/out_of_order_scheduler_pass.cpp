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

#include "pypto/ir/transform/out_of_order_scheduler_pass.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transform/base/mutator.h"
#include "pypto/ir/transform/base/visitor.h"

namespace pypto {
namespace ir {

namespace {

constexpr int kMaxEventIds = 8;

using PipePair = std::pair<PipeType, PipeType>;  // (SRC=set_pipe, DST=wait_pipe)

struct CandidateScore {
  int pred_max = 0;
  int pred_sum = 0;
  int idx = -1;
  bool releases_event = false;  // true if scheduling this candidate will free at least one event_id slot
};

/**
 * @brief Conservative side-effect summary for any statement.
 *
 * Used to analyze control flow nodes (IfStmt/ForStmt) as black boxes in the dependency graph,
 * enabling cross-CF reordering while preserving semantic correctness.
 */
struct StmtEffect {
  std::set<MemRefPtr> reads;           // MemRefs read by this statement (or its children)
  std::set<MemRefPtr> writes;          // MemRefs written by this statement (or its children)
  bool has_unknown_side_effect = false;  // Unanalyzable/external side effects (I/O, terminators, etc.)
};

/**
 * @brief Collector for all MemRefs in an expression
 */
class MemRefCollector : public IRVisitor {
 public:
  std::set<MemRefPtr> memrefs;

  void VisitExpr_(const VarPtr& var) override {
    if (auto shaped_type = std::dynamic_pointer_cast<const ShapedType>(var->GetType())) {
      if (shaped_type->memref_.has_value()) {
        memrefs.insert(*shaped_type->memref_);
      }
    }
    IRVisitor::VisitExpr_(var);
  }
};

bool IsSameMem(const MemRefPtr& a, const MemRefPtr& b) { return a.get() == b.get(); }

PipeType PipeTypeFromString(const std::string& s) {
  if (s == "CUBE" || s == "M") return PipeType::M;
  if (s == "VECTOR" || s == "V") return PipeType::V;
  if (s == "MTE1") return PipeType::MTE1;
  if (s == "MTE2") return PipeType::MTE2;
  if (s == "MTE3") return PipeType::MTE3;
  if (s == "SCALAR" || s == "S") return PipeType::S;
  if (s == "FIX") return PipeType::FIX;
  if (s == "ALL") return PipeType::ALL;
  return PipeType::S;
}

/**
 * @brief Extract pipe type from a statement
 *
 * Prefer Op::GetPipe(); fallback to call.kwargs["pipe_type"] (string) if present.
 */
PipeType GetStmtPipe(const StmtPtr& stmt) {
  auto get_call_pipe = [](const CallPtr& call) -> PipeType {
    if (!call || !call->op_) return PipeType::S;
    if (auto pipe_opt = call->op_->GetPipe(); pipe_opt.has_value()) {
      return *pipe_opt;
    }
    if (call->HasKwarg("pipe_type")) {
      try {
        std::string pipe_str = call->GetKwarg<std::string>("pipe_type", "S");
        return PipeTypeFromString(pipe_str);
      } catch (...) {
        return PipeType::S;
      }
    }
    return PipeType::S;
  };

  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    if (auto call = std::dynamic_pointer_cast<const Call>(assign->value_)) {
      return get_call_pipe(call);
    }
  } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    if (auto call = std::dynamic_pointer_cast<const Call>(eval->expr_)) {
      return get_call_pipe(call);
    }
  }
  return PipeType::S;
}

struct DepEdge {
  int producer_idx;
  int consumer_idx;
  bool cross_pipe;
};

/**
 * @brief Check if statement is a control flow or terminator node.
 *
 * Control flow nodes act as barriers in the dependency graph and must preserve relative order.
 */
bool IsControlFlowLike(const StmtPtr& stmt) {
  return std::dynamic_pointer_cast<const IfStmt>(stmt) || std::dynamic_pointer_cast<const ForStmt>(stmt) ||
         std::dynamic_pointer_cast<const ReturnStmt>(stmt) || std::dynamic_pointer_cast<const YieldStmt>(stmt);
}

/**
 * @brief Resource accounting for live cross-pipe events during scheduling.
 *
 * Models hardware event_id slot lifetime based on actual sync instruction insertion:
 * - One event allocated per (producer, dst_pipe) pair (broadcast semantics)
 * - Event_id slot freed when FIRST consumer is scheduled (matches InsertSyncPass)
 * - Subsequent consumers tracked for dependency correctness but don't use event_id
 *
 * Key design principles:
 * 1. Event_id resource tracking: first consumer frees the slot (can be reused)
 * 2. Data dependency tracking: all consumers tracked via pending_successors_
 * 3. Scheduling correctness: DAG constraints ensure proper consumer ordering
 *
 * This accurately models InsertSyncPass's instruction insertion pattern where
 * sync_dst only appears before the first consumer (pipe-level barrier).
 */
class LiveCrossPipeEvents {
 public:
  LiveCrossPipeEvents(int n, int max_events)
      : max_events_(max_events), pending_successors_(n), incoming_producers_(n) {}

  // Predict whether scheduling `cand` next would keep every (SRC,DST) bucket <= kMaxEventIds.
  // Returns {ok, pred_max, pred_sum}. This mirrors the old delta-based logic, but is encapsulated.
  [[nodiscard]] std::tuple<bool, int, int> PredictAfterScheduling(int cand, const std::vector<PipeType>& pipes,
                                                                  const std::vector<std::vector<int>>& succ_cross,
                                                                  const std::vector<bool>& scheduled,
                                                                  bool enforce_limit) const {
    std::map<PipePair, int> delta = ComputeDelta(cand, pipes, succ_cross, scheduled);
    return PredictWithDelta(delta, enforce_limit);
  }

  // Apply the "wait/release" side for `consumer` BEFORE instruction execution.
  void ReleaseIncomingBeforeExecute(int consumer) {
    // For each incoming producer dependency:
    // - Decrement the producer's pending successor count for this pair
    // - Dynamically check if this is the FIRST consumer to be scheduled: free the event_id slot (live_by_pair_)
    // - If count reaches 0: cleanup bookkeeping
    for (const auto& [producer, pair] : incoming_producers_[consumer]) {
      // Verify producer has pending successors for this pair
      auto prod_it = pending_successors_[producer].find(pair);
      INTERNAL_CHECK(prod_it != pending_successors_[producer].end())
          << "OutOfOrderSchedulerPass: attempted to release event for producer_idx=" << producer
          << " on pipe pair " << static_cast<int>(pair.first) << "->" << static_cast<int>(pair.second)
          << " but no pending successors recorded (consumer_idx=" << consumer << "). "
          << "This indicates state inconsistency between pending_successors_ and incoming_producers_.";

      INTERNAL_CHECK(prod_it->second.remaining > 0)
          << "OutOfOrderSchedulerPass: pending successor count for producer_idx=" << producer << " on pipe pair "
          << static_cast<int>(pair.first) << "->" << static_cast<int>(pair.second) << " is already "
          << prod_it->second.remaining << " (consumer_idx=" << consumer << "). "
          << "Cannot decrement below 1. This indicates double-release or state corruption.";

      // Decrement pending successor count
      prod_it->second.remaining -= 1;

      // Dynamic first-consumer detection (per-producer, per-pair):
      // Only the first scheduled consumer for a given (producer, pair) frees the event_id slot.
      // (Matches InsertSyncPass: only the first consumer gets sync_dst, so event_id can be reused after it)
      if (prod_it->second.event_live) {
        prod_it->second.event_live = false;
        auto live_it = live_by_pair_.find(pair);
        INTERNAL_CHECK(live_it != live_by_pair_.end() && live_it->second > 0)
            << "OutOfOrderSchedulerPass: attempted to free a live event_id slot for pipe pair "
            << static_cast<int>(pair.first) << "->" << static_cast<int>(pair.second)
            << " but live_by_pair_ has no positive count. State corruption detected.";
        live_it->second -= 1;
        if (live_it->second == 0) {
          live_by_pair_.erase(live_it);
        }
      }

      // If this was the LAST consumer for this producer-pair, cleanup bookkeeping
      if (prod_it->second.remaining == 0) {
        // Remove from pending_successors_ (cleanup)
        pending_successors_[producer].erase(prod_it);
      }
      // Otherwise: other consumers still pending
    }

    // Clear incoming producer list for this consumer (it's now scheduled)
    incoming_producers_[consumer].clear();
  }

  // Apply the "set/allocate" side for `producer` AFTER instruction execution.
  void AllocateOutgoingAfterExecute(int producer, const std::vector<PipeType>& pipes,
                                    const std::vector<std::vector<int>>& succ_cross,
                                    const std::vector<bool>& scheduled) {
    // Group unscheduled cross-pipe successors by PipePair
    std::map<PipePair, std::vector<int>> successors_by_pair;

    for (int consumer : succ_cross[producer]) {
      if (!scheduled[consumer]) {
        PipePair pair{pipes[producer], pipes[consumer]};
        successors_by_pair[pair].push_back(consumer);
      }
    }

    // For each unique PipePair with at least one successor:
    // - Allocate exactly ONE event (increment live_by_pair_ by 1, not by edge count)
    // - Record pending successor count for this producer-pair combination
    // - Register this producer as an incoming dependency for each consumer
    for (const auto& [pair, consumers] : successors_by_pair) {
      int successor_count = static_cast<int>(consumers.size());

      // Allocate ONE event_id slot for this producer on this PipePair.
      // live_by_pair_ counts how many (producer, pair) events are currently still occupying an event_id slot
      // (i.e., have not yet been freed by the first scheduled consumer).
      live_by_pair_[pair] += 1;

      // Record: producer has `successor_count` pending consumers on this pair, and the event_id slot is live.
      pending_successors_[producer][pair] = PendingSuccessorInfo{successor_count, /*event_live=*/true};

      // Design note: The FIRST consumer (dynamically determined during scheduling)
      // will release the event_id slot when it is scheduled.
      // We track ALL consumers in incoming_producers_ to maintain dependency
      // relationships and update pending_successors_ correctly.
      for (int consumer : consumers) {
        incoming_producers_[consumer].emplace_back(producer, pair);
      }
    }
  }

  void UpdatePeak() {
    for (const auto& [pair, cur] : live_by_pair_) {
      peak_by_pair_[pair] = std::max(peak_by_pair_[pair], cur);
    }
  }

  [[nodiscard]] int WorstPeakBucket() const {
    int worst = 0;
    for (const auto& [_, pk] : peak_by_pair_) worst = std::max(worst, pk);
    return worst;
  }

  // Check if a candidate will release any event_id when scheduled
  // Returns true if the candidate has at least one incoming producer dependency
  // where the event_id is still live (dynamically determined)
  [[nodiscard]] bool IsFirstConsumer(int consumer) const {
    for (const auto& [producer, pair] : incoming_producers_[consumer]) {
      // Check if this specific (producer, pair) still has a live event_id slot.
      auto prod_it = pending_successors_[producer].find(pair);
      INTERNAL_CHECK(prod_it != pending_successors_[producer].end())
          << "OutOfOrderSchedulerPass: IsFirstConsumer state inconsistency: producer_idx=" << producer
          << " has no pending successors recorded for pipe pair " << static_cast<int>(pair.first) << "->"
          << static_cast<int>(pair.second) << " (consumer_idx=" << consumer << ").";
      if (prod_it->second.event_live) return true;
    }
    return false;
  }

 private:
  [[nodiscard]] std::map<PipePair, int> ComputeDelta(int cand, const std::vector<PipeType>& pipes,
                                                     const std::vector<std::vector<int>>& succ_cross,
                                                     const std::vector<bool>& scheduled) const {
    std::map<PipePair, int> delta;

    // === RELEASE SIDE: Incoming events consumed BEFORE candidate execution ===
    // For each incoming producer dependency of the candidate:
    // - If event_id is still live for this pair: delta -= 1 (event_id slot freed by first scheduled consumer)
    // - Otherwise: no delta for event_id (already freed by another consumer)
    for (const auto& [producer, pair] : incoming_producers_[cand]) {
      auto prod_it = pending_successors_[producer].find(pair);
      INTERNAL_CHECK(prod_it != pending_successors_[producer].end())
          << "OutOfOrderSchedulerPass: prediction for candidate=" << cand << " failed: producer_idx=" << producer
          << " has no pending successors for pair " << static_cast<int>(pair.first) << "->"
          << static_cast<int>(pair.second) << ". State inconsistency detected.";

      int pending_count = prod_it->second.remaining;
      INTERNAL_CHECK(pending_count >= 1)
          << "OutOfOrderSchedulerPass: prediction for candidate=" << cand << " failed: producer_idx=" << producer
          << " has pending_count=" << pending_count << " for pair " << static_cast<int>(pair.first) << "->"
          << static_cast<int>(pair.second) << ". Count must be >= 1 since candidate is unscheduled.";

      // Dynamic first-consumer check (per-producer, per-pair): if event_id is still live for this producer,
      // this candidate will free it (Matches InsertSyncPass: only first scheduled consumer has sync_dst).
      if (prod_it->second.event_live) {
        delta[pair] -= 1;
      }
      // Otherwise: event_id already freed by another consumer that was scheduled earlier for this producer-pair
    }

    // === ALLOCATE SIDE: Outgoing events issued AFTER candidate execution ===
    // Group unscheduled cross-pipe successors by unique PipePair
    // Each unique pair contributes +1 to delta (broadcast semantics)
    std::set<PipePair> unique_pairs;
    for (int consumer : succ_cross[cand]) {
      if (!scheduled[consumer]) {
        PipePair pair{pipes[cand], pipes[consumer]};
        unique_pairs.insert(pair);
      }
    }

    // Allocate one event per unique PipePair
    for (const PipePair& pair : unique_pairs) {
      delta[pair] += 1;
    }

    return delta;
  }

  [[nodiscard]] std::tuple<bool, int, int> PredictWithDelta(const std::map<PipePair, int>& delta,
                                                            bool enforce_limit) const {
    int pred_max = 0;
    int pred_sum = 0;

    // Existing pairs
    for (const auto& [pair, cur] : live_by_pair_) {
      int d = 0;
      auto it = delta.find(pair);
      if (it != delta.end()) d = it->second;
      int pred = cur + d;
      INTERNAL_CHECK(pred >= 0) << "OutOfOrderSchedulerPass: negative prediction detected for pipe pair "
                                << static_cast<int>(pair.first) << "->" << static_cast<int>(pair.second)
                                << " (cur=" << cur << ", delta=" << d
                                << "). This indicates state inconsistency between live_by_pair_ and "
                                << "live_incoming_cross_by_pair_.";
      if (enforce_limit && pred > max_events_) return {false, 0, 0};
      pred_max = std::max(pred_max, pred);
      pred_sum += pred;
    }

    // New pairs introduced by delta
    for (const auto& [pair, d] : delta) {
      if (live_by_pair_.count(pair)) continue;
      INTERNAL_CHECK(d >= 0)
          << "OutOfOrderSchedulerPass: negative delta for a non-live pipe pair "
          << static_cast<int>(pair.first) << "->" << static_cast<int>(pair.second) << " (delta=" << d
          << "). This indicates state inconsistency between live_by_pair_ and live_incoming_cross_by_pair_.";
      int pred = d;
      if (enforce_limit && pred > max_events_) return {false, 0, 0};
      pred_max = std::max(pred_max, pred);
      pred_sum += pred;
    }

    return {true, pred_max, pred_sum};
  }

  int max_events_;
  struct PendingSuccessorInfo {
    int remaining = 0;        // Count of unscheduled consumers for this producer-pair
    bool event_live = false;  // Whether this producer-pair still occupies an event_id slot (freed by first consumer)
  };
  // Per-producer tracking: map from producer_idx to (PipePair → PendingSuccessorInfo)
  // - remaining: how many consumers are still unscheduled
  // - event_live: whether the producer's event_id slot for this pair is still live (not yet freed)
  std::vector<std::map<PipePair, PendingSuccessorInfo>> pending_successors_;
  // Per-consumer tracking: list of (producer_idx, PipePair) incoming dependencies.
  // All consumers decrement pending_successors_ when scheduled.
  // The first consumer to be actually scheduled (dynamically determined) will release the event_id.
  std::vector<std::vector<std::tuple<int, PipePair>>> incoming_producers_;
  // Global live event_id slot count per pipeline pair.
  // Counts how many (producer, pair) events still occupy an event_id slot (not edges).
  std::map<PipePair, int> live_by_pair_;
  // Peak pressure statistics
  std::map<PipePair, int> peak_by_pair_;
};

/**
 * @brief Helper function to extract memory references from an expression.
 */
std::set<MemRefPtr> GetMemRefs(const ExprPtr& expr) {
  MemRefCollector collector;
  collector.VisitExpr(expr);
  return collector.memrefs;
}

/**
 * @brief Visitor to analyze statement side effects for dependency analysis.
 *
 * Accumulates a conservative summary of MemRef reads/writes and unknown side effects.
 * Utilizes IRVisitor infrastructure for automatic traversal of nested statements.
 */
class StmtEffectAnalyzer : public IRVisitor {
 public:
  StmtEffect result;

 private:
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    auto writes = GetMemRefs(stmt->var_);
    result.writes.insert(writes.begin(), writes.end());
    auto reads = GetMemRefs(stmt->value_);
    result.reads.insert(reads.begin(), reads.end());
  }

  void VisitStmt_(const EvalStmtPtr& stmt) override {
    auto reads = GetMemRefs(stmt->expr_);
    result.reads.insert(reads.begin(), reads.end());
    result.has_unknown_side_effect = true;  // Conservative: may have side effects
  }

  void VisitStmt_(const IfStmtPtr& stmt) override {
    auto cond_refs = GetMemRefs(stmt->condition_);
    result.reads.insert(cond_refs.begin(), cond_refs.end());

    // Visit then branch
    VisitStmt(stmt->then_body_);

    // Visit else branch if exists
    if (stmt->else_body_.has_value()) {
      VisitStmt(*stmt->else_body_);
    }

    // Important: return_vars are def-points that can be used after the If statement.
    // We must add them to writes so that subsequent statements using these vars
    // are correctly identified as depending on this If statement.
    for (const auto& ret_var : stmt->return_vars_) {
      auto ret_writes = GetMemRefs(ret_var);
      result.writes.insert(ret_writes.begin(), ret_writes.end());
    }
  }

  void VisitStmt_(const ForStmtPtr& stmt) override {
    // Loop bounds
    auto start_refs = GetMemRefs(stmt->start_);
    result.reads.insert(start_refs.begin(), start_refs.end());
    auto stop_refs = GetMemRefs(stmt->stop_);
    result.reads.insert(stop_refs.begin(), stop_refs.end());
    auto step_refs = GetMemRefs(stmt->step_);
    result.reads.insert(step_refs.begin(), step_refs.end());

    // Iter args initialization
    for (const auto& iter_arg : stmt->iter_args_) {
      auto init_refs = GetMemRefs(iter_arg->initValue_);
      result.reads.insert(init_refs.begin(), init_refs.end());
    }

    // Visit body (automatically accumulates reads/writes)
    VisitStmt(stmt->body_);

    // Important: return_vars capture final iteration values and can be used after the loop.
    // We must add them to writes so that subsequent statements using these vars
    // are correctly identified as depending on this For statement.
    for (const auto& ret_var : stmt->return_vars_) {
      auto ret_writes = GetMemRefs(ret_var);
      result.writes.insert(ret_writes.begin(), ret_writes.end());
    }
  }

  void VisitStmt_(const SeqStmtsPtr& stmt) override {
    for (const auto& s : stmt->stmts_) {
      VisitStmt(s);
    }
  }

  void VisitStmt_(const OpStmtsPtr& stmt) override {
    for (const auto& s : stmt->stmts_) {
      VisitStmt(s);
    }
  }

  void VisitStmt_(const ReturnStmtPtr& stmt) override { result.has_unknown_side_effect = true; }

  void VisitStmt_(const YieldStmtPtr& stmt) override { result.has_unknown_side_effect = true; }
};

/**
 * @brief Analyze statement side effects for dependency analysis.
 *
 * Computes a conservative summary of MemRef reads/writes and unknown side effects.
 * Used to enable control flow nodes to participate in parent-level dependency graphs.
 *
 * @param stmt Statement to analyze
 * @return StmtEffect with reads, writes, and side effect flags
 */
StmtEffect AnalyzeStmtEffect(const StmtPtr& stmt) {
  StmtEffectAnalyzer analyzer;
  analyzer.VisitStmt(stmt);
  return analyzer.result;
}

/**
 * @brief Build dependency graph based on MemRef-based hazard detection (RAW/WAW/WAR).
 *
 * @param stmts Input statements in original order
 * @param pipes Pipeline type for each statement
 * @return Vector of dependency edges
 */
std::vector<DepEdge> BuildDependencyGraph(const std::vector<StmtPtr>& stmts,
                                          const std::vector<PipeType>& pipes,
                                          bool analyze_cf = false) {
  const int n = static_cast<int>(stmts.size());
  std::vector<DepEdge> edges;
  std::set<std::pair<int, int>> existing_deps;
  std::map<MemRefPtr, int> last_writer;
  std::map<MemRefPtr, std::vector<int>> last_readers;

  auto add_dep = [&](int prod, int cons) {
    if (prod < 0) return;
    if (existing_deps.count({prod, cons})) return;
    existing_deps.insert({prod, cons});
    // Cross-pipe edges are only meaningful for the event_id resource model when both endpoints
    // are "compute-like" statements that can participate in InsertSyncPass set/wait insertion.
    // Control-flow/terminator nodes (If/For/Return/Yield) should not contribute to event_id pressure,
    // otherwise we can vastly overestimate peak events (e.g., everything depends on ReturnStmt(S)).
    bool cross_pipe =
        (pipes[prod] != pipes[cons]) && !IsControlFlowLike(stmts[prod]) && !IsControlFlowLike(stmts[cons]);
    edges.push_back(DepEdge{prod, cons, cross_pipe});
  };

  for (int i = 0; i < n; ++i) {
    const auto& stmt = stmts[i];
    std::set<MemRefPtr> reads;
    std::set<MemRefPtr> writes;
    bool has_unknown_side_effect = false;

    if (analyze_cf) {
      // CF-aware: use StmtEffect for all statements
      auto effect = AnalyzeStmtEffect(stmt);
      reads = effect.reads;
      writes = effect.writes;
      has_unknown_side_effect = effect.has_unknown_side_effect;
    } else {
      // Original behavior: only analyze AssignStmt/EvalStmt
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
        writes = GetMemRefs(assign->var_);
        reads = GetMemRefs(assign->value_);
      } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
        reads = GetMemRefs(eval->expr_);
      }
    }

    // RAW: Read-After-Write dependencies
    for (const auto& r : reads) {
      for (auto const& [m, idx] : last_writer) {
        if (IsSameMem(r, m)) add_dep(idx, i);
      }
    }

    // WAW and WAR: Write-After-Write and Write-After-Read dependencies
    for (const auto& w : writes) {
      // WAW: Write-After-Write
      for (auto const& [m, idx] : last_writer) {
        if (IsSameMem(w, m)) add_dep(idx, i);
      }
      // WAR: Write-After-Read
      for (auto const& [m, indices] : last_readers) {
        if (IsSameMem(w, m)) {
          for (int r_idx : indices) add_dep(r_idx, i);
        }
      }
    }

    // Unknown side effects create barriers: prevent reordering both before and after
    if (analyze_cf && has_unknown_side_effect) {
      // Barrier prevents statements before from moving after (all predecessors -> barrier)
      for (int j = 0; j < i; ++j) {
        add_dep(j, i);
      }
      // Barrier prevents statements after from moving before (barrier -> all successors)
      for (int k = i + 1; k < n; ++k) {
        add_dep(i, k);
      }
    }

    // Update last write/read sets
    for (const auto& w : writes) {
      last_writer[w] = i;
      last_readers[w].clear();
    }
    for (const auto& r : reads) {
      last_readers[r].push_back(i);
    }
  }

  return edges;
}

/**
 * @brief Add ordering constraints to preserve control flow node relative order.
 *
 * This function adds edges between consecutive control flow nodes to preserve their relative order,
 * while still allowing compute statements to cross them. This is critical for Phase 1 correctness.
 *
 * @param edges Dependency edges (modified in-place)
 * @param stmts Input statements in original order
 */
void AddControlFlowOrderingConstraints(std::vector<DepEdge>& edges, const std::vector<StmtPtr>& stmts) {
  const int n = static_cast<int>(stmts.size());
  std::vector<int> cf_nodes;  // Indices of control flow nodes in original order

  // Identify all CF-like nodes
  for (int i = 0; i < n; ++i) {
    if (IsControlFlowLike(stmts[i])) {
      cf_nodes.push_back(i);
    }
  }

  // Add edges: cf_0 -> cf_1 -> ... -> cf_k
  // This preserves CF relative order while allowing compute to cross them
  std::set<std::pair<int, int>> existing_deps;
  for (const auto& e : edges) {
    existing_deps.insert({e.producer_idx, e.consumer_idx});
  }

  for (size_t i = 0; i + 1 < cf_nodes.size(); ++i) {
    int producer = cf_nodes[i];
    int consumer = cf_nodes[i + 1];
    if (!existing_deps.count({producer, consumer})) {
      existing_deps.insert({producer, consumer});
      edges.push_back(DepEdge{producer, consumer, false});  // CF ordering edge
    }
  }
}

struct AdjacencyInfo {
  std::vector<std::vector<int>> succ;         // All successors
  std::vector<std::vector<int>> succ_cross;   // Cross-pipe successors only
  std::vector<int> indegree;                  // Indegree for each node
};

/**
 * @brief Candidate selection strategies for Kahn scheduling.
 */
enum class PickStrategy {
  // Existing behavior: minimize max bucket, then minimize sum buckets, then original order.
  kMinMaxThenSumThenIndex,
  // Alternative: minimize sum buckets first; sometimes avoids greedy dead-ends.
  kMinSumThenMaxThenIndex,
  // Alternative: ignore sum; strictly minimize max bucket; keeps tie-breaking simple.
  kMinMaxThenIndex,
};

/**
 * @brief Get human-readable name for a strategy.
 */
const char* StrategyName(PickStrategy s) {
  switch (s) {
    case PickStrategy::kMinMaxThenSumThenIndex:
      return "min_max_then_sum_then_index";
    case PickStrategy::kMinSumThenMaxThenIndex:
      return "min_sum_then_max_then_index";
    case PickStrategy::kMinMaxThenIndex:
      return "min_max_then_index";
    default:
      return "unknown";
  }
}

/**
 * @brief Compare two candidate scores based on the given strategy.
 * @return true if a is better than b
 */
bool IsBetterCandidate(const CandidateScore& a, const CandidateScore& b, PickStrategy strategy) {
  // Primary objective: schedule first-consumers as early as possible to free event_id slots.
  // This is a hard priority to match the requirement that event_id should be released ASAP.
  if (a.releases_event != b.releases_event) return a.releases_event > b.releases_event;

  switch (strategy) {
    case PickStrategy::kMinMaxThenSumThenIndex: {
      if (a.pred_max != b.pred_max) return a.pred_max < b.pred_max;
      if (a.pred_sum != b.pred_sum) return a.pred_sum < b.pred_sum;
      return a.idx < b.idx;
    }
    case PickStrategy::kMinSumThenMaxThenIndex: {
      if (a.pred_sum != b.pred_sum) return a.pred_sum < b.pred_sum;
      if (a.pred_max != b.pred_max) return a.pred_max < b.pred_max;
      return a.idx < b.idx;
    }
    case PickStrategy::kMinMaxThenIndex: {
      if (a.pred_max != b.pred_max) return a.pred_max < b.pred_max;
      return a.idx < b.idx;
    }
    default:
      return false;
  }
}

struct ScheduleResult {
  std::vector<int> order;
  int worst_peak = 0;
  bool satisfied_limit = false;
};

/**
 * @brief Run Kahn's topological scheduling with resource constraints.
 *
 * @param n Number of statements
 * @param pipes Pipeline type for each statement
 * @param succ All successors for each statement
 * @param succ_cross Cross-pipe successors for each statement
 * @param indegree Initial indegree for each statement
 * @param max_events Maximum events per pipeline pair
 * @param enforce_limit If true, fail if constraint violated; if false, continue heuristically
 * @param strategy Candidate selection strategy
 * @param cf_nodes Optional: set of indices that are control flow nodes (for Strategy A prioritization)
 * @return Schedule result if successful, nullopt if enforcement fails
 */
std::optional<ScheduleResult> RunKahnScheduling(int n, const std::vector<PipeType>& pipes,
                                                const std::vector<std::vector<int>>& succ,
                                                const std::vector<std::vector<int>>& succ_cross,
                                                const std::vector<int>& indegree, int max_events,
                                                bool enforce_limit, PickStrategy strategy,
                                                const std::set<int>& cf_nodes = {}) {
  std::set<int> ready;
  std::vector<int> indeg = indegree;  // copy for mutation
  for (int i = 0; i < n; ++i) {
    if (indeg[i] == 0) ready.insert(i);
  }

  std::vector<bool> scheduled(n, false);
  LiveCrossPipeEvents live_events(n, max_events);

  std::vector<int> order;
  order.reserve(n);

  while (static_cast<int>(order.size()) < n) {
    // Strategy A: Prioritize schedulable compute statements over control flow nodes.
    // This prevents CF nodes from blocking compute reordering.

    CandidateScore best;
    best.pred_max = 1 << 30;
    best.pred_sum = 1 << 30;
    best.idx = -1;

    auto consider_candidates = [&](bool skip_cf) {
      for (int cand : ready) {
        if (skip_cf && cf_nodes.count(cand)) continue;

        auto [ok, pred_max, pred_sum] =
            live_events.PredictAfterScheduling(cand, pipes, succ_cross, scheduled, enforce_limit);
        if (!ok) continue;

        CandidateScore cur;
        cur.pred_max = pred_max;
        cur.pred_sum = pred_sum;
        cur.idx = cand;
        cur.releases_event = live_events.IsFirstConsumer(cand);

        if (best.idx == -1 || IsBetterCandidate(cur, best, strategy)) best = cur;
      }
    };

    // First pass: try to find a non-CF compute statement (if cf_nodes is specified)
    if (!cf_nodes.empty()) consider_candidates(/*skip_cf=*/true);

    // Second pass: if no compute candidate was found, consider all ready nodes
    if (best.idx == -1) consider_candidates(/*skip_cf=*/false);

    if (best.idx == -1) {
      if (enforce_limit) {
        return std::nullopt;  // No feasible candidate found
      }
      // Fallback: pick smallest ready index to guarantee progress.
      best.idx = *ready.begin();
    }

    // Schedule the selected candidate.
    ready.erase(best.idx);

    live_events.ReleaseIncomingBeforeExecute(best.idx);
    scheduled[best.idx] = true;
    order.push_back(best.idx);

    live_events.AllocateOutgoingAfterExecute(best.idx, pipes, succ_cross, scheduled);
    live_events.UpdatePeak();

    // Update ready set.
    for (int v : succ[best.idx]) {
      indeg[v]--;
      if (indeg[v] == 0) ready.insert(v);
    }
  }

  ScheduleResult res;
  res.order = std::move(order);
  res.worst_peak = live_events.WorstPeakBucket();
  res.satisfied_limit = (res.worst_peak <= max_events);
  return res;
}

/**
 * @brief Try multiple strategies to find a feasible schedule.
 *
 * @param cf_nodes Optional: set of control flow node indices (for Strategy A prioritization)
 * @return Schedule result with order and peak pressure information
 */
ScheduleResult FindFeasibleSchedule(int n, const std::vector<PipeType>& pipes,
                                    const std::vector<std::vector<int>>& succ,
                                    const std::vector<std::vector<int>>& succ_cross,
                                    const std::vector<int>& indegree,
                                    const std::set<int>& cf_nodes = {}) {
  // Try strict scheduling with several deterministic strategies to avoid greedy dead-ends.
  const std::vector<PickStrategy> strategies = {
      PickStrategy::kMinMaxThenSumThenIndex,
      PickStrategy::kMinSumThenMaxThenIndex,
      PickStrategy::kMinMaxThenIndex,
  };

  std::optional<ScheduleResult> best_strict;
  for (auto s : strategies) {
    auto r = RunKahnScheduling(n, pipes, succ, succ_cross, indegree, kMaxEventIds,
                               /*enforce_limit=*/true, s, cf_nodes);
    if (r.has_value()) {
      best_strict = std::move(r);
      if (s != PickStrategy::kMinMaxThenSumThenIndex) {
        LOG_WARN << "OutOfOrderSchedulerPass: recovered a feasible schedule with strategy=" << StrategyName(s)
                 << " for a SeqStmts segment of size " << n << ".";
      }
      break;
    }
  }

  if (best_strict.has_value()) {
    return *best_strict;
  }

  // Last resort: produce a dependency-preserving order that minimizes peak pressure heuristically,
  // even if the limit cannot be satisfied.
  auto relaxed = RunKahnScheduling(n, pipes, succ, succ_cross, indegree, kMaxEventIds,
                                   /*enforce_limit=*/false, PickStrategy::kMinMaxThenSumThenIndex, cf_nodes);
  INTERNAL_CHECK(relaxed.has_value()) << "OutOfOrderSchedulerPass: relaxed scheduling unexpectedly failed";

  LOG_WARN << "OutOfOrderSchedulerPass: cannot find a schedule satisfying per-(SRC,DST) event limit <= "
           << kMaxEventIds << " for a SeqStmts segment of size " << n
           << ". Falling back to a best-effort topological order (worst_peak_live_events_per_pair="
           << relaxed->worst_peak << ").";

  return *relaxed;
}

AdjacencyInfo BuildAdjacencyLists(int n, const std::vector<DepEdge>& edges) {
  AdjacencyInfo info;
  info.succ.resize(n);
  info.succ_cross.resize(n);
  info.indegree.resize(n, 0);

  for (const auto& e : edges) {
    info.succ[e.producer_idx].push_back(e.consumer_idx);
    info.indegree[e.consumer_idx]++;
    if (e.cross_pipe) {
      info.succ_cross[e.producer_idx].push_back(e.consumer_idx);
    }
  }

  return info;
}

/**
 * @brief Schedule a contiguous segment of reorderable statements.
 *
 * The dependency graph is built based on the *original order* to preserve sequential semantics.
 * The scheduler then finds a dependency-preserving permutation whose peak number of live
 * cross-pipe edges is <= 8 when possible.
 *
 * @param stmts Input statements (can include control flow if analyze_cf=true)
 * @param changed Output: true if any reordering occurred
 * @param analyze_cf If true, treat CF nodes as black boxes in dependency analysis;
 *                   if false, only reorder contiguous compute segments (legacy behavior)
 */
std::vector<StmtPtr> ScheduleSegment(const std::vector<StmtPtr>& stmts, bool* changed,
                                     bool analyze_cf = true) {
  if (changed) *changed = false;
  const int n = static_cast<int>(stmts.size());
  if (n <= 1) return stmts;

  // Precompute per-stmt pipe type for cross-pipe edge identification.
  std::vector<PipeType> pipes;
  pipes.reserve(n);

  // DEBUG: Log pipe types for large segments
  bool log_pipes = (false);
  if (log_pipes) {
    LOG_WARN << "OutOfOrderSchedulerPass: ScheduleSegment analyzing " << n << " statements";
  }

  for (int i = 0; i < n; ++i) {
    PipeType pipe = GetStmtPipe(stmts[i]);
    pipes.push_back(pipe);

    if (log_pipes) {
      LOG_WARN << "  stmt[" << i << "]: pipe=" << static_cast<int>(pipe)
                << " (" << (pipe == PipeType::M ? "M" :
                           pipe == PipeType::V ? "V" :
                           pipe == PipeType::MTE1 ? "MTE1" :
                           pipe == PipeType::MTE2 ? "MTE2" :
                           pipe == PipeType::MTE3 ? "MTE3" :
                           pipe == PipeType::S ? "S" :
                           pipe == PipeType::FIX ? "FIX" :
                           pipe == PipeType::ALL ? "ALL" : "UNKNOWN") << ")";
    }
  }

  // 1) Build dependencies using StmtEffect-based analysis (CF-aware) or MemRef-only (legacy)
  std::vector<DepEdge> edges = BuildDependencyGraph(stmts, pipes, analyze_cf);

  // 2) For CF-aware analysis, add ordering constraints to preserve control flow relative order
  if (analyze_cf) {
    AddControlFlowOrderingConstraints(edges, stmts);
  }

  // 3) Build adjacency for topological scheduling.
  AdjacencyInfo adj_info = BuildAdjacencyLists(n, edges);
  const auto& succ = adj_info.succ;
  const auto& succ_cross = adj_info.succ_cross;

  // Identify CF nodes for Strategy A prioritization (only when analyze_cf=true)
  std::set<int> cf_nodes;
  if (analyze_cf) {
    for (int i = 0; i < n; ++i) {
      if (IsControlFlowLike(stmts[i])) {
        cf_nodes.insert(i);
      }
    }
  }

  // 4) Find a feasible schedule using Kahn algorithm with resource constraints.
  ScheduleResult result = FindFeasibleSchedule(n, pipes, succ, succ_cross, adj_info.indegree, cf_nodes);

  // Build reordered statement list and detect changes.
  std::vector<StmtPtr> out;
  out.reserve(n);
  for (int idx : result.order) out.push_back(stmts[idx]);

  // Detect changes.
  if (changed) {
    *changed = false;
    for (int i = 0; i < n; ++i) {
      if (result.order[i] != i) {
        *changed = true;
        break;
      }
    }
  }

  LOG_DEBUG << "OutOfOrderSchedulerPass: scheduled segment size=" << n
            << ", CF-aware=" << (analyze_cf ? "yes" : "no")
            << ", worst_peak_live_events_per_pair=" << result.worst_peak;
  return out;
}

class OutOfOrderSchedulerMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> visited;
    visited.reserve(op->stmts_.size());
    for (const auto& s : op->stmts_) visited.push_back(VisitStmt(s));

    // Phase 1: Use CF-aware scheduling directly on all direct children.
    // This allows compute statements to cross over CF nodes while preserving CF relative order.
    bool changed = false;
    auto scheduled = ScheduleSegment(visited, &changed, /*analyze_cf=*/true);

    // If nothing changed, preserve original node when possible (copy-on-write optimization).
    if (!changed) {
      return op;
    }

    return std::make_shared<const SeqStmts>(scheduled, op->span_);
  }
};

}  // namespace

FunctionPtr OutOfOrderSchedulerPass::Run(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "OutOfOrderSchedulerPass cannot run on null function";
  OutOfOrderSchedulerMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);
  return std::make_shared<const Function>(func->name_, func->params_, func->return_types_, new_body,
                                          func->span_);
}

}  // namespace ir
}  // namespace pypto

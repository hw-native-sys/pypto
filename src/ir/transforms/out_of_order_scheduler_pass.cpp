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

bool IsReorderableStmt(const StmtPtr& stmt) {
  // Conservative: only reorder straight-line "compute" statements.
  // Control-flow and terminators (Return/Yield/For/If/While, etc.) are barriers.
  return std::dynamic_pointer_cast<const AssignStmt>(stmt) || std::dynamic_pointer_cast<const EvalStmt>(stmt);
}

class LiveCrossPipeEvents {
 public:
  LiveCrossPipeEvents(int n, int max_events) : max_events_(max_events), live_incoming_cross_by_pair_(n) {}

  // Predict whether scheduling `cand` next would keep every (SRC,DST) bucket <= kMaxEventIds.
  // Returns {ok, pred_max, pred_sum}. This mirrors the old delta-based logic, but is encapsulated.
  std::tuple<bool, int, int> PredictAfterScheduling(int cand, const std::vector<PipeType>& pipes,
                                                    const std::vector<std::vector<int>>& succ_cross,
                                                    const std::vector<bool>& scheduled,
                                                    bool enforce_limit) const {
    std::map<PipePair, int> delta = ComputeDelta(cand, pipes, succ_cross, scheduled);
    return PredictWithDelta(delta, enforce_limit);
  }

  // Apply the "wait/release" side for `consumer` BEFORE instruction execution.
  void ReleaseIncomingBeforeExecute(int consumer) {
    for (const auto& [pair, cnt] : live_incoming_cross_by_pair_[consumer]) {
      auto it = live_by_pair_.find(pair);
      INTERNAL_CHECK(it != live_by_pair_.end())
          << "OutOfOrderSchedulerPass: attempted to release events for pipe pair "
          << static_cast<int>(pair.first) << "->" << static_cast<int>(pair.second)
          << " but it does not exist in live_by_pair_ (consumer_idx=" << consumer << ", cnt=" << cnt << ")";
      INTERNAL_CHECK(it->second >= cnt)
          << "OutOfOrderSchedulerPass: attempted to release more events than live for pipe pair "
          << static_cast<int>(pair.first) << "->" << static_cast<int>(pair.second)
          << " (consumer_idx=" << consumer << ", live=" << it->second << ", release=" << cnt << ")";
      it->second -= cnt;
      if (it->second <= 0) {
        live_by_pair_.erase(it);
      }
    }
    live_incoming_cross_by_pair_[consumer].clear();
  }

  // Apply the "set/allocate" side for `producer` AFTER instruction execution.
  void AllocateOutgoingAfterExecute(int producer, const std::vector<PipeType>& pipes,
                                    const std::vector<std::vector<int>>& succ_cross,
                                    const std::vector<bool>& scheduled) {
    for (int v : succ_cross[producer]) {
      if (!scheduled[v]) {
        PipePair pair{pipes[producer], pipes[v]};
        live_by_pair_[pair] += 1;
        live_incoming_cross_by_pair_[v][pair] += 1;
      }
    }
  }

  void UpdatePeak() {
    for (const auto& [pair, cur] : live_by_pair_) {
      peak_by_pair_[pair] = std::max(peak_by_pair_[pair], cur);
    }
  }

  int WorstCurrentBucket() const {
    int worst = 0;
    for (const auto& [_, cur] : live_by_pair_) worst = std::max(worst, cur);
    return worst;
  }

  int WorstPeakBucket() const {
    int worst = 0;
    for (const auto& [_, pk] : peak_by_pair_) worst = std::max(worst, pk);
    return worst;
  }

 private:
  std::map<PipePair, int> ComputeDelta(int cand, const std::vector<PipeType>& pipes,
                                       const std::vector<std::vector<int>>& succ_cross,
                                       const std::vector<bool>& scheduled) const {
    std::map<PipePair, int> delta;

    // Release incoming events for this consumer BEFORE instruction execution.
    for (const auto& [pair, cnt] : live_incoming_cross_by_pair_[cand]) {
      if (cnt != 0) delta[pair] -= cnt;
    }

    // Allocate outgoing events for this producer AFTER instruction execution.
    for (int v : succ_cross[cand]) {
      if (!scheduled[v]) {
        PipePair pair{pipes[cand], pipes[v]};
        delta[pair] += 1;
      }
    }

    return delta;
  }

  std::tuple<bool, int, int> PredictWithDelta(const std::map<PipePair, int>& delta,
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
  std::vector<std::map<PipePair, int>> live_incoming_cross_by_pair_;
  std::map<PipePair, int> live_by_pair_;
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
 * @brief Build dependency graph based on MemRef-based hazard detection (RAW/WAW/WAR).
 *
 * @param stmts Input statements in original order
 * @param pipes Pipeline type for each statement
 * @return Vector of dependency edges
 */
std::vector<DepEdge> BuildDependencyGraph(const std::vector<StmtPtr>& stmts,
                                          const std::vector<PipeType>& pipes) {
  const int n = static_cast<int>(stmts.size());
  std::vector<DepEdge> edges;
  std::set<std::pair<int, int>> existing_deps;
  std::map<MemRefPtr, int> last_writer;
  std::map<MemRefPtr, std::vector<int>> last_readers;

  auto add_dep = [&](int prod, int cons) {
    if (prod < 0) return;
    if (existing_deps.count({prod, cons})) return;
    existing_deps.insert({prod, cons});
    edges.push_back(DepEdge{prod, cons, pipes[prod] != pipes[cons]});
  };

  for (int i = 0; i < n; ++i) {
    const auto& stmt = stmts[i];
    std::set<MemRefPtr> reads;
    std::set<MemRefPtr> writes;

    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      writes = GetMemRefs(assign->var_);
      reads = GetMemRefs(assign->value_);
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      reads = GetMemRefs(eval->expr_);
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

struct AdjacencyInfo {
  std::vector<std::vector<int>> succ;         // All successors
  std::vector<std::vector<int>> succ_cross;   // Cross-pipe successors only
  std::vector<int> indegree;                  // Indegree for each node
};

/**
 * @brief Build adjacency lists from dependency edges.
 *
 * @param n Number of statements
 * @param edges Dependency edges
 * @return Adjacency information (successor lists and indegrees)
 */
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
 * @return Schedule result if successful, nullopt if enforcement fails
 */
std::optional<ScheduleResult> RunKahnScheduling(int n, const std::vector<PipeType>& pipes,
                                                const std::vector<std::vector<int>>& succ,
                                                const std::vector<std::vector<int>>& succ_cross,
                                                const std::vector<int>& indegree, int max_events,
                                                bool enforce_limit, PickStrategy strategy) {
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
    // Find best candidate among ready statements.
    CandidateScore best;
    best.pred_max = 1 << 30;
    best.pred_sum = 1 << 30;
    best.idx = -1;

    for (int cand : ready) {
      auto [ok, pred_max, pred_sum] =
          live_events.PredictAfterScheduling(cand, pipes, succ_cross, scheduled, enforce_limit);
      if (!ok) continue;

      CandidateScore cur;
      cur.pred_max = pred_max;
      cur.pred_sum = pred_sum;
      cur.idx = cand;
      if (best.idx == -1 || IsBetterCandidate(cur, best, strategy)) {
        best = cur;
      }
    }

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
 * @return Schedule result with order and peak pressure information
 */
ScheduleResult FindFeasibleSchedule(int n, const std::vector<PipeType>& pipes,
                                    const std::vector<std::vector<int>>& succ,
                                    const std::vector<std::vector<int>>& succ_cross,
                                    const std::vector<int>& indegree) {
  // Try strict scheduling with several deterministic strategies to avoid greedy dead-ends.
  const std::vector<PickStrategy> strategies = {
      PickStrategy::kMinMaxThenSumThenIndex,
      PickStrategy::kMinSumThenMaxThenIndex,
      PickStrategy::kMinMaxThenIndex,
  };

  std::optional<ScheduleResult> best_strict;
  for (auto s : strategies) {
    auto r = RunKahnScheduling(n, pipes, succ, succ_cross, indegree, kMaxEventIds,
                               /*enforce_limit=*/true, s);
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
                                   /*enforce_limit=*/false, PickStrategy::kMinMaxThenSumThenIndex);
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
 */
std::vector<StmtPtr> ScheduleSegment(const std::vector<StmtPtr>& stmts, bool* changed) {
  if (changed) *changed = false;
  const int n = static_cast<int>(stmts.size());
  if (n <= 1) return stmts;

  // Precompute per-stmt pipe type for cross-pipe edge identification.
  std::vector<PipeType> pipes;
  pipes.reserve(n);
  for (const auto& s : stmts) pipes.push_back(GetStmtPipe(s));

  // 1) Build dependencies using MemRef-based hazard detection.
  std::vector<DepEdge> edges = BuildDependencyGraph(stmts, pipes);

  // 2) Build adjacency for topological scheduling.
  AdjacencyInfo adj_info = BuildAdjacencyLists(n, edges);
  const auto& succ = adj_info.succ;
  const auto& succ_cross = adj_info.succ_cross;
  auto indegree = adj_info.indegree;  // Copy for mutation during scheduling

  // 3) Find a feasible schedule using Kahn algorithm with resource constraints.
  ScheduleResult result = FindFeasibleSchedule(n, pipes, succ, succ_cross, adj_info.indegree);

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
            << ", worst_peak_live_events_per_pair=" << result.worst_peak;
  return out;
}

class OutOfOrderSchedulerMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> visited;
    visited.reserve(op->stmts_.size());
    for (const auto& s : op->stmts_) visited.push_back(VisitStmt(s));

    std::vector<StmtPtr> out;
    out.reserve(visited.size());

    std::vector<StmtPtr> segment;
    auto flush_segment = [&]() {
      if (segment.empty()) return;
      bool changed = false;
      auto scheduled = ScheduleSegment(segment, &changed);
      out.insert(out.end(), scheduled.begin(), scheduled.end());
      segment.clear();
    };

    for (const auto& s : visited) {
      if (IsReorderableStmt(s)) {
        segment.push_back(s);
      } else {
        flush_segment();
        out.push_back(s);  // barrier stays in place
      }
    }
    flush_segment();

    // If nothing changed, preserve original node when possible.
    if (out.size() == visited.size()) {
      bool identical = true;
      for (size_t i = 0; i < out.size(); ++i) {
        if (out[i].get() != visited[i].get()) {
          identical = false;
          break;
        }
      }
      if (identical) return op;
    }

    return std::make_shared<const SeqStmts>(out, op->span_);
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

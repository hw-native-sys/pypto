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
#include <any>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Path element representing a position in the IR tree
struct PathElement {
  enum class Kind { SeqIndex, OpIndex, IfThen, IfElse, ForBody };
  Kind kind;
  int index;  // Index within SeqStmts/OpStmts, or -1 for branch/body markers

  bool operator==(const PathElement& other) const { return kind == other.kind && index == other.index; }

  bool operator<(const PathElement& other) const {
    if (kind != other.kind) return static_cast<int>(kind) < static_cast<int>(other.kind);
    return index < other.index;
  }
};

// Position in the IR tree, represented as a path from root
struct Position {
  std::vector<PathElement> path;

  bool operator<(const Position& other) const { return path < other.path; }
  bool operator==(const Position& other) const { return path == other.path; }

  // Determines if this position and another are within the same control flow scope
  [[nodiscard]] bool IsInSameScope(const Position& other) const {
    // Helper: Find the innermost SeqIndex by searching backwards.
    auto get_scope_anchor_idx = [](const std::vector<PathElement>& p) -> int {
      for (int i = static_cast<int>(p.size()) - 1; i >= 0; --i) {
        if (p[i].kind == PathElement::Kind::SeqIndex) return i;
      }
      return -1;
    };
    int idx_this = get_scope_anchor_idx(path);
    int idx_other = get_scope_anchor_idx(other.path);
    // If anchor depths differ or no SeqIndex is found, the scopes are different.
    if (idx_this < 0 || idx_this != idx_other) return false;

    // Compare all path elements before the anchor (scope prefix).
    for (int i = 0; i < idx_this; ++i) {
      if (!(path[i] == other.path[i])) return false;
    }
    return true;
  }

  // Determines if this position is before another, based on path ordering rules.
  [[nodiscard]] bool IsBefore(const Position& other) const {
    size_t min_len = std::min(path.size(), other.path.size());
    for (size_t i = 0; i < min_len; ++i) {
      if (!(path[i] == other.path[i])) {
        if (path[i].kind == other.path[i].kind &&
            (path[i].kind == PathElement::Kind::SeqIndex || path[i].kind == PathElement::Kind::OpIndex)) {
          return path[i].index < other.path[i].index;
        }
        return false;
      }
    }
    return false;
  }
};

class MemRefCollector : public IRVisitor {
 public:
  std::set<MemRefPtr> memrefs;
  void VisitExpr_(const VarPtr& var) override {
    if (auto shaped_type = As<ShapedType>(var->GetType())) {
      if (shaped_type->memref_.has_value()) memrefs.insert(*shaped_type->memref_);
    }
    IRVisitor::VisitExpr_(var);
  }
};

std::set<MemRefPtr> GetExprMemRefs(const ExprPtr& expr) {
  MemRefCollector collector;
  collector.VisitExpr(expr);
  return collector.memrefs;
}

PipeType GetPipeForCall(const Call* call) {
  if (call->op_->GetPipe().has_value()) return *call->op_->GetPipe();
  const pypto::backend::Backend* backend = pypto::backend::GetBackend();
  if (!backend) {
    throw ValueError("InsertSync requires a configured backend to determine op pipelines.");
  }
  const auto* info = backend->GetOpInfo(call->op_->name_);
  if (info) return info->pipe;
  return PipeType::S;
}

PipeType GetStmtPipe(const StmtPtr& stmt) {
  if (auto assign = As<AssignStmt>(stmt)) {
    if (auto call = As<Call>(assign->value_)) return GetPipeForCall(call.get());
  } else if (auto eval = As<EvalStmt>(stmt)) {
    if (auto call = As<Call>(eval->expr_)) return GetPipeForCall(call.get());
  }
  return PipeType::S;
}

struct MemRefSummary {
  std::map<MemRefPtr, std::vector<Position>> last_writers;
  std::map<MemRefPtr, std::vector<Position>> last_readers;

  static MemRefSummary Merge(const MemRefSummary& a, const MemRefSummary& b) {
    MemRefSummary merged;
    MergeMap(a.last_writers, merged.last_writers);
    MergeMap(b.last_writers, merged.last_writers);
    MergeMap(a.last_readers, merged.last_readers);
    MergeMap(b.last_readers, merged.last_readers);
    return merged;
  }

 private:
  static void MergeMap(const std::map<MemRefPtr, std::vector<Position>>& src,
                       std::map<MemRefPtr, std::vector<Position>>& dst) {
    for (const auto& [memref, positions] : src) {
      auto& d = dst[memref];
      d.insert(d.end(), positions.begin(), positions.end());
    }
  }
};

std::pair<std::set<MemRefPtr>, std::set<MemRefPtr>> GetLeafMemRefs(const StmtPtr& stmt) {
  std::set<MemRefPtr> reads, writes;
  if (!stmt) return {reads, writes};
  if (auto assign = As<AssignStmt>(stmt)) {
    // block.store is special: the AssignStmt LHS (assign->var_) holds the output tensor memref
    // (write), but the RHS Call's args[0] is the source tile being stored (read), not a write.
    // For all other ops, the entire RHS expression is treated as reads.
    if (auto call = As<Call>(assign->value_); call && call->op_ && call->op_->name_ == "block.store") {
      reads = GetExprMemRefs(call->args_[0]);
    } else {
      reads = GetExprMemRefs(assign->value_);
    }
    writes = GetExprMemRefs(assign->var_);
  } else if (auto eval = As<EvalStmt>(stmt)) {
    reads = GetExprMemRefs(eval->expr_);
  }
  return {reads, writes};
}

StmtPtr CreateSyncCall(const std::string& op_name, PipeType p, PipeType tp, int event_id) {
  auto& registry = OpRegistry::GetInstance();
  std::vector<std::pair<std::string, std::any>> kwargs = {
      {"set_pipe", static_cast<int>(p)}, {"wait_pipe", static_cast<int>(tp)}, {"event_id", event_id}};
  auto call = registry.Create(op_name, {}, kwargs, Span::unknown());
  return std::make_shared<const EvalStmt>(call, Span::unknown());
}

StmtPtr CreateBarCall(const std::string& op_name) {
  auto& registry = OpRegistry::GetInstance();
  auto call = registry.Create(op_name, {}, {}, Span::unknown());
  return std::make_shared<const EvalStmt>(call, Span::unknown());
}

class EventIdManager {
 public:
  static constexpr int kMaxEvents = 8;
  EventIdManager() = default;

  int Alloc(PipeType src_pipe, PipeType dst_pipe, const Position& set_position) {
    SetKey set_key = std::make_tuple(src_pipe, dst_pipe, set_position);
    auto it = set_to_id_.find(set_key);
    if (it != set_to_id_.end()) return it->second;

    for (int id = 0; id < kMaxEvents; ++id) {
      IdKey id_key = std::make_tuple(src_pipe, dst_pipe, id);
      auto pos_it = id_to_free_pos_.find(id_key);
      if (pos_it == id_to_free_pos_.end() || pos_it->second.IsBefore(set_position)) {
        set_to_id_[set_key] = id;
        return id;
      }
    }
    std::stringstream ss;
    ss << "Out of hardware event IDs (max " << kMaxEvents << ") for pipe pair " << static_cast<int>(src_pipe)
       << "->" << static_cast<int>(dst_pipe);
    throw ValueError(ss.str());
  }

  void Free(PipeType src_pipe, PipeType dst_pipe, const Position& wait_position, int event_id) {
    IdKey id_key = std::make_tuple(src_pipe, dst_pipe, event_id);
    auto it = id_to_free_pos_.find(id_key);
    if (it == id_to_free_pos_.end() || it->second.IsBefore(wait_position)) {
      id_to_free_pos_[id_key] = wait_position;
    }
  }

 private:
  using SetKey = std::tuple<PipeType, PipeType, Position>;
  std::map<SetKey, int> set_to_id_;
  using IdKey = std::tuple<PipeType, PipeType, int>;
  std::map<IdKey, Position> id_to_free_pos_;
};

struct SyncPair {
  PipeType producer_pipe;
  PipeType consumer_pipe;
  int event_id = -1;
  Position set_position;
  Position wait_position;
  bool set_before = false;  // When true: sync_src uses insert_before (not insert_after)
  bool wait_after = false;  // When true: sync_dst/bar uses insert_after (not insert_before)

  [[nodiscard]] bool IsSamePipe() const { return producer_pipe == consumer_pipe; }
};

struct InsertionPlan {
  // Key: (seq_index, op_index). op_index = -1 for positions at the SeqStmts child level.
  using PosKey = std::pair<int, int>;
  std::map<PosKey, std::vector<StmtPtr>> insert_before;
  std::map<PosKey, std::vector<StmtPtr>> insert_after;
};

class AnalysisContext {
 public:
  std::vector<SyncPair> sync_pairs;
  std::vector<PathElement> current_path;
  std::map<Position, StmtPtr> pos_to_stmt;
  std::map<std::vector<PathElement>, int> op_counts;

  [[nodiscard]] Position CurrentPosition() const {
    Position pos;
    pos.path = current_path;
    return pos;
  }

  void EnterSeq(int index) { current_path.push_back({PathElement::Kind::SeqIndex, index}); }
  void EnterOp(int index) { current_path.push_back({PathElement::Kind::OpIndex, index}); }
  void EnterIfThen() { current_path.push_back({PathElement::Kind::IfThen, -1}); }
  void EnterIfElse() { current_path.push_back({PathElement::Kind::IfElse, -1}); }
  void EnterForBody() { current_path.push_back({PathElement::Kind::ForBody, -1}); }
  void Leave() { current_path.pop_back(); }
};

// --------------------------------------------------------------------------
// Main Inserter Pass
// --------------------------------------------------------------------------

class SyncInserter {
 public:
  SyncInserter() = default;

  FunctionPtr Run(const FunctionPtr& func) {
    ctx_ = AnalysisContext();

    // Phase 1: Collect raw sync pairs
    CollectSyncPairs(func->body_);

    // Phase 2: Adjust scope crossings
    AdjustScopeCrossings();

    // Phase 3: Assign Hardware Event IDs
    AssignEventIds();

    // Phase 4: Build Insertion Plans and mutate AST
    BuildInsertionPlans();
    std::vector<PathElement> path;
    auto new_body = ApplyInsertions(func->body_, path);

    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, new_body, func->span_, func->func_type_);
  }

 private:
  AnalysisContext ctx_;
  std::map<std::vector<PathElement>, InsertionPlan> insertion_plans_;

  // --------------------------------------------------------------------------
  // Phase 1: Collect Sync Pairs
  // --------------------------------------------------------------------------

  void CollectSyncPairs(const StmtPtr& stmt) {
    MemRefSummary state;
    CollectSyncPairsImpl(stmt, state);
  }

  void CollectSyncPairsImpl(const StmtPtr& stmt, MemRefSummary& state) {
    if (auto seq = As<SeqStmts>(stmt)) {
      CollectFromSeqStmts(seq, state);
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      CollectFromIfStmt(if_stmt, state);
    } else if (auto for_stmt = As<ForStmt>(stmt)) {
      CollectFromForStmt(for_stmt, state);
    }
  }

  void ProcessLeafStmt(const StmtPtr& leaf_stmt, MemRefSummary& state) {
    Position current_pos = ctx_.CurrentPosition();
    auto [reads, writes] = GetLeafMemRefs(leaf_stmt);
    if (reads.empty() && writes.empty()) return;

    ctx_.pos_to_stmt[current_pos] = leaf_stmt;

    // RAW + WAW
    auto sync_against_writers = [&](const std::set<MemRefPtr>& memrefs) {
      for (const auto& mr : memrefs) {
        auto it = state.last_writers.find(mr);
        if (it != state.last_writers.end()) {
          for (const auto& writer_pos : it->second) CreateSyncPair(writer_pos, current_pos);
        }
      }
    };
    sync_against_writers(reads);
    sync_against_writers(writes);

    // WAR
    for (const auto& w : writes) {
      auto it = state.last_readers.find(w);
      if (it != state.last_readers.end()) {
        for (const auto& reader_pos : it->second) CreateSyncPair(reader_pos, current_pos);
      }
    }

    for (const auto& w : writes) {
      state.last_writers[w] = {current_pos};
      state.last_readers[w].clear();
    }
    for (const auto& r : reads) {
      state.last_readers[r].push_back(current_pos);
    }
  }

  void CollectFromSeqStmts(const SeqStmtsPtr& seq, MemRefSummary& state) {
    size_t pairs_start_idx = ctx_.sync_pairs.size();

    for (int i = 0; i < static_cast<int>(seq->stmts_.size()); ++i) {
      const auto& stmt = seq->stmts_[i];
      ctx_.EnterSeq(i);

      if (auto op_stmts = As<OpStmts>(stmt)) {
        ctx_.op_counts[ctx_.current_path] = static_cast<int>(op_stmts->stmts_.size());
        for (int j = 0; j < static_cast<int>(op_stmts->stmts_.size()); ++j) {
          ctx_.EnterOp(j);
          ProcessLeafStmt(op_stmts->stmts_[j], state);
          ctx_.Leave();
        }
      } else if (auto if_stmt = As<IfStmt>(stmt)) {
        CollectFromIfStmt(if_stmt, state);
      } else if (auto for_stmt = As<ForStmt>(stmt)) {
        CollectFromForStmt(for_stmt, state);
      } else {
        ProcessLeafStmt(stmt, state);
      }
      ctx_.Leave();
    }
    DeduplicateSyncPairs(pairs_start_idx);
    RemoveTransitiveRedundantPairs(pairs_start_idx);
    RemoveLinearRedundantPairs(pairs_start_idx);
  }

  void CollectFromIfStmt(const IfStmtPtr& if_stmt, MemRefSummary& state) {
    MemRefSummary state_before = state;
    ctx_.EnterIfThen();
    CollectSyncPairsImpl(if_stmt->then_body_, state);
    ctx_.Leave();
    MemRefSummary state_after_then = state;

    MemRefSummary state_after_else = state_before;
    if (if_stmt->else_body_) {
      ctx_.EnterIfElse();
      state = state_before;
      CollectSyncPairsImpl(*if_stmt->else_body_, state);
      ctx_.Leave();
      state_after_else = state;
    }
    state = MemRefSummary::Merge(state_after_then, state_after_else);
  }

  void CollectFromForStmt(const ForStmtPtr& for_stmt, MemRefSummary& state) {
    ctx_.EnterForBody();
    auto seq = As<SeqStmts>(for_stmt->body_);
    if (!seq || seq->stmts_.empty()) {
      if (seq) CollectSyncPairsImpl(for_stmt->body_, state);
      ctx_.Leave();
      return;
    }

    int body_size = static_cast<int>(seq->stmts_.size());
    size_t pairs_start_idx = ctx_.sync_pairs.size();

    // unroll the loop body once to expose cross-iteration
    std::vector<StmtPtr> unrolled_stmts;
    unrolled_stmts.reserve(static_cast<size_t>(body_size) * 2);
    unrolled_stmts.insert(unrolled_stmts.end(), seq->stmts_.begin(), seq->stmts_.end());
    unrolled_stmts.insert(unrolled_stmts.end(), seq->stmts_.begin(), seq->stmts_.end());
    auto unrolled_seq = std::make_shared<SeqStmts>(unrolled_stmts, seq->span_);
    CollectFromSeqStmts(unrolled_seq, state);

    // restore unrolled positions to original and deduplicate pairs
    AdjustUnrolledPositions(pairs_start_idx, body_size, state);
    DeduplicateSyncPairs(pairs_start_idx);

    ctx_.Leave();
  }

  void AdjustUnrolledPositions(size_t pairs_start_idx, int body_size, MemRefSummary& state) {
    auto adjust_path = [body_size](std::vector<PathElement>& path) {
      for (size_t i = 0; i < path.size(); ++i) {
        if (path[i].kind == PathElement::Kind::ForBody && i + 1 < path.size() &&
            path[i + 1].kind == PathElement::Kind::SeqIndex && path[i + 1].index >= body_size) {
          path[i + 1].index -= body_size;
        }
      }
    };

    for (size_t i = pairs_start_idx; i < ctx_.sync_pairs.size(); ++i) {
      adjust_path(ctx_.sync_pairs[i].set_position.path);
      adjust_path(ctx_.sync_pairs[i].wait_position.path);
    }

    auto adjust_and_dedup = [&adjust_path](std::map<MemRefPtr, std::vector<Position>>& pos_map) {
      for (auto& [memref, positions] : pos_map) {
        for (auto& pos : positions) adjust_path(pos.path);
        std::set<Position> seen;
        std::vector<Position> unique;
        for (auto& pos : positions) {
          if (seen.insert(pos).second) unique.push_back(std::move(pos));
        }
        positions = std::move(unique);
      }
    };
    adjust_and_dedup(state.last_writers);
    adjust_and_dedup(state.last_readers);
  }

  void CreateSyncPair(const Position& producer_pos, const Position& consumer_pos) {
    auto producer_it = ctx_.pos_to_stmt.find(producer_pos);
    auto consumer_it = ctx_.pos_to_stmt.find(consumer_pos);
    if (producer_it == ctx_.pos_to_stmt.end() || consumer_it == ctx_.pos_to_stmt.end()) return;

    PipeType p_pipe = GetStmtPipe(producer_it->second);
    PipeType c_pipe = GetStmtPipe(consumer_it->second);

    if (p_pipe == PipeType::S || c_pipe == PipeType::S) return;

    SyncPair pair;
    pair.producer_pipe = p_pipe;
    pair.consumer_pipe = c_pipe;
    pair.set_position = producer_pos;
    pair.wait_position = consumer_pos;
    ctx_.sync_pairs.push_back(std::move(pair));
  }

  void DeduplicateSyncPairs(size_t start_idx) {
    using Key = std::tuple<Position, Position, PipeType, PipeType>;
    std::map<Key, size_t> best_pair;
    for (size_t i = start_idx; i < ctx_.sync_pairs.size(); ++i) {
      const auto& p = ctx_.sync_pairs[i];
      Key key = std::make_tuple(p.set_position, p.wait_position, p.producer_pipe, p.consumer_pipe);
      if (best_pair.find(key) == best_pair.end()) best_pair[key] = i;
    }

    std::vector<SyncPair> deduped;
    deduped.reserve(ctx_.sync_pairs.size());
    for (size_t i = 0; i < start_idx; ++i) deduped.push_back(ctx_.sync_pairs[i]);

    std::set<size_t> kept;
    for (const auto& [_, idx] : best_pair) kept.insert(idx);
    for (size_t i = start_idx; i < ctx_.sync_pairs.size(); ++i) {
      if (kept.count(i)) deduped.push_back(ctx_.sync_pairs[i]);
    }
    ctx_.sync_pairs = std::move(deduped);
  }

  // If A→C && A→B && B→C, while A, B, C are ordered as A < B < C, then A→C is redundant.
  void RemoveTransitiveRedundantPairs(size_t start_idx) {
    size_t count = ctx_.sync_pairs.size();
    std::vector<bool> is_redundant(count, false);

    for (size_t i = start_idx; i < count; ++i) {
      const auto& pair_ac = ctx_.sync_pairs[i];
      for (size_t j = 0; j < count; ++j) {
        if (i == j) continue;
        const auto& pair_ab = ctx_.sync_pairs[j];
        if (!(pair_ab.set_position == pair_ac.set_position)) continue;
        if (!(pair_ac.set_position.IsBefore(pair_ab.wait_position) &&
              pair_ab.wait_position.IsBefore(pair_ac.wait_position))) {
          continue;
        }

        for (size_t k = 0; k < count; ++k) {
          if (k == i || k == j) continue;
          const auto& pair_bc = ctx_.sync_pairs[k];
          if (!(pair_bc.set_position == pair_ab.wait_position)) continue;
          if (!(pair_bc.wait_position == pair_ac.wait_position)) continue;
          is_redundant[i] = true;
          break;
        }
        if (is_redundant[i]) break;
      }
    }

    std::vector<SyncPair> filtered;
    for (size_t i = 0; i < count; ++i) {
      if (!is_redundant[i]) filtered.push_back(ctx_.sync_pairs[i]);
    }
    ctx_.sync_pairs = std::move(filtered);
  }

  // For same pipe pair: same wait → keep latest set; same set → keep earliest wait.
  void RemoveLinearRedundantPairs(size_t start_idx) {
    size_t count = ctx_.sync_pairs.size();
    std::vector<bool> is_redundant(count, false);

    for (size_t i = start_idx; i < count; ++i) {
      if (is_redundant[i]) continue;
      const auto& p1 = ctx_.sync_pairs[i];

      for (size_t j = i + 1; j < count; ++j) {
        if (is_redundant[j]) continue;
        const auto& p2 = ctx_.sync_pairs[j];

        if (p1.producer_pipe != p2.producer_pipe || p1.consumer_pipe != p2.consumer_pipe) {
          continue;
        }

        // Same wait_position -> keep the LATEST set_position
        if (p1.wait_position == p2.wait_position) {
          if (p1.set_position.IsBefore(p2.set_position)) {
            is_redundant[i] = true;
            break;
          } else if (p2.set_position.IsBefore(p1.set_position)) {
            is_redundant[j] = true;
          }
        } else if (p1.set_position == p2.set_position) {
          // Same set_position -> keep the EARLIEST wait_position
          if (p1.wait_position.IsBefore(p2.wait_position)) {
            is_redundant[j] = true;
          } else if (p2.wait_position.IsBefore(p1.wait_position)) {
            is_redundant[i] = true;
            break;
          }
        }
      }
    }

    std::vector<SyncPair> filtered;
    filtered.reserve(count);
    for (size_t i = 0; i < start_idx; ++i) filtered.push_back(ctx_.sync_pairs[i]);
    for (size_t i = start_idx; i < count; ++i) {
      if (!is_redundant[i]) filtered.push_back(ctx_.sync_pairs[i]);
    }
    ctx_.sync_pairs = std::move(filtered);
  }

  // --------------------------------------------------------------------------
  // Phase 2: Adjust Scope Crossings
  // --------------------------------------------------------------------------

  [[nodiscard]] static int GetScopeDepth(const Position& pos) {
    for (int i = static_cast<int>(pos.path.size()) - 1; i >= 0; --i) {
      if (pos.path[i].kind == PathElement::Kind::SeqIndex) return i;
    }
    return -1;
  }

  [[nodiscard]] Position EndOfOpScope(const Position& pos) const {
    if (pos.path.size() >= 2 && pos.path.back().kind == PathElement::Kind::OpIndex) {
      std::vector<PathElement> key(pos.path.begin(), pos.path.end() - 1);
      auto it = ctx_.op_counts.find(key);
      if (it != ctx_.op_counts.end()) {
        Position result;
        result.path = std::move(key);
        result.path.push_back(PathElement{PathElement::Kind::OpIndex, it->second - 1});
        return result;
      }
    }
    return pos;
  }

  [[nodiscard]] static Position BeginOfOpScope(const Position& pos) {
    if (pos.path.size() >= 2 && pos.path.back().kind == PathElement::Kind::OpIndex) {
      Position result;
      result.path.assign(pos.path.begin(), pos.path.end() - 1);
      result.path.push_back({PathElement::Kind::OpIndex, 0});
      return result;
    }
    return pos;
  }

  void AdjustScopeCrossings() {
    for (auto& pair : ctx_.sync_pairs) {
      // Cross-iteration(same scope, wait <= set), Move wait to end of iteration.
      if (pair.set_position.IsInSameScope(pair.wait_position) &&
          (pair.wait_position.IsBefore(pair.set_position) || pair.wait_position == pair.set_position)) {
        pair.wait_position = EndOfOpScope(pair.set_position);
        pair.wait_after = true;
        continue;
      }

      if (pair.IsSamePipe()) continue;
      if (pair.set_position.IsInSameScope(pair.wait_position)) continue;

      // Different scopes: adjust based on relative depth
      int set_depth = GetScopeDepth(pair.set_position);
      int wait_depth = GetScopeDepth(pair.wait_position);

      if (wait_depth > set_depth) {
        pair.wait_position = EndOfOpScope(pair.set_position);
        pair.wait_after = true;
      } else {
        pair.set_position = BeginOfOpScope(pair.wait_position);
        pair.set_before = true;
      }
    }

    DeduplicateSyncPairs(0);
  }

  // --------------------------------------------------------------------------
  // Phase 3: Event ID Allocation
  // --------------------------------------------------------------------------

  void AssignEventIds() {
    EventIdManager event_manager;

    // Collect indices of cross-pipe pairs and sort by set_position
    std::vector<size_t> indices;
    for (size_t i = 0; i < ctx_.sync_pairs.size(); ++i) {
      if (!ctx_.sync_pairs[i].IsSamePipe()) indices.push_back(i);
    }
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
      return ctx_.sync_pairs[a].set_position < ctx_.sync_pairs[b].set_position;
    });

    for (size_t idx : indices) {
      auto& pair = ctx_.sync_pairs[idx];
      pair.event_id = event_manager.Alloc(pair.producer_pipe, pair.consumer_pipe, pair.set_position);
      event_manager.Free(pair.producer_pipe, pair.consumer_pipe, pair.wait_position, pair.event_id);
    }
  }

  // --------------------------------------------------------------------------
  // Phase 4: AST Construction
  // --------------------------------------------------------------------------

  static std::vector<PathElement> GetParentPath(const Position& pos) {
    std::vector<PathElement> parent;
    if (pos.path.empty()) return parent;
    // Strip trailing OpIndex if present, then strip SeqIndex
    size_t end = pos.path.size();
    if (pos.path.back().kind == PathElement::Kind::OpIndex && end >= 2) {
      end -= 2;  // Strip both OpIndex and SeqIndex
    } else if (end >= 1) {
      end -= 1;  // Strip just SeqIndex
    }
    parent.assign(pos.path.begin(), pos.path.begin() + static_cast<std::ptrdiff_t>(end));
    return parent;
  }

  static InsertionPlan::PosKey GetPlanIndex(const Position& pos) {
    if (pos.path.empty()) return {-1, -1};
    if (pos.path.back().kind == PathElement::Kind::OpIndex && pos.path.size() >= 2) {
      return {pos.path[pos.path.size() - 2].index, pos.path.back().index};
    }
    if (pos.path.back().kind == PathElement::Kind::SeqIndex) {
      return {pos.path.back().index, -1};
    }
    return {-1, -1};
  }

  void BuildInsertionPlans() {
    using SyncInsKey = std::tuple<InsertionPlan::PosKey, PipeType, PipeType, int>;
    std::map<std::vector<PathElement>, std::set<SyncInsKey>> scope_sync_src;
    std::map<std::vector<PathElement>, std::set<SyncInsKey>> scope_sync_dst;
    std::map<std::vector<PathElement>, std::set<std::pair<InsertionPlan::PosKey, PipeType>>> scope_bars;

    for (const auto& pair : ctx_.sync_pairs) {
      if (!pair.IsSamePipe()) {
        auto set_parent = GetParentPath(pair.set_position);
        auto set_key = GetPlanIndex(pair.set_position);
        if (set_key.first >= 0) {
          auto src_key = std::make_tuple(set_key, pair.producer_pipe, pair.consumer_pipe, pair.event_id);
          if (!scope_sync_src[set_parent].count(src_key)) {
            auto& plan = insertion_plans_[set_parent];
            auto& target = pair.set_before ? plan.insert_before : plan.insert_after;
            target[set_key].push_back(
                CreateSyncCall("system.sync_src", pair.producer_pipe, pair.consumer_pipe, pair.event_id));
            scope_sync_src[set_parent].insert(src_key);
          }
        }
      }

      auto wait_parent = GetParentPath(pair.wait_position);
      auto wait_key = GetPlanIndex(pair.wait_position);
      if (wait_key.first >= 0) {
        if (pair.IsSamePipe()) {
          auto bar_key = std::make_pair(wait_key, pair.producer_pipe);
          if (!scope_bars[wait_parent].count(bar_key)) {
            auto& plan = insertion_plans_[wait_parent];
            auto& target = pair.wait_after ? plan.insert_after : plan.insert_before;
            if (pair.producer_pipe == PipeType::V) {
              target[wait_key].push_back(CreateBarCall("system.bar_v"));
            } else if (pair.producer_pipe == PipeType::M) {
              target[wait_key].push_back(CreateBarCall("system.bar_m"));
            }
            scope_bars[wait_parent].insert(bar_key);
          }
        } else {
          auto dst_key = std::make_tuple(wait_key, pair.producer_pipe, pair.consumer_pipe, pair.event_id);
          if (!scope_sync_dst[wait_parent].count(dst_key)) {
            auto& plan = insertion_plans_[wait_parent];
            auto& target = pair.wait_after ? plan.insert_after : plan.insert_before;
            target[wait_key].push_back(
                CreateSyncCall("system.sync_dst", pair.producer_pipe, pair.consumer_pipe, pair.event_id));
            scope_sync_dst[wait_parent].insert(dst_key);
          }
        }
      }
    }
  }

  StmtPtr ApplyInsertions(const StmtPtr& stmt, std::vector<PathElement>& path) {
    if (auto seq = As<SeqStmts>(stmt)) {
      return ApplyToSeqStmts(seq, path);
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      return ApplyToIfStmt(if_stmt, path);
    } else if (auto for_stmt = As<ForStmt>(stmt)) {
      return ApplyToForStmt(for_stmt, path);
    }
    return stmt;
  }

  StmtPtr ApplyToSeqStmts(const SeqStmtsPtr& seq, std::vector<PathElement>& path) {
    auto plan_it = insertion_plans_.find(path);
    bool has_plan = (plan_it != insertion_plans_.end());
    std::vector<StmtPtr> result;
    // Buffer for pending op-compatible stmts to merge into adjacent OpStmts
    std::vector<StmtPtr> pending_ops;

    // Flush pending_ops: merge into previous OpStmts if possible, otherwise create new one
    auto flush_pending = [&]() {
      if (pending_ops.empty()) return;
      if (!result.empty()) {
        if (auto prev_ops = As<OpStmts>(result.back())) {
          // Merge into previous OpStmts
          std::vector<StmtPtr> merged(prev_ops->stmts_.begin(), prev_ops->stmts_.end());
          merged.insert(merged.end(), pending_ops.begin(), pending_ops.end());
          result.back() = std::make_shared<const OpStmts>(merged, prev_ops->span_);
          pending_ops.clear();
          return;
        }
      }
      result.push_back(std::make_shared<const OpStmts>(pending_ops, pending_ops[0]->span_));
      pending_ops.clear();
    };

    for (int i = 0; i < static_cast<int>(seq->stmts_.size()); ++i) {
      // Collect insert_before at SeqStmts level into pending_ops
      if (has_plan) CollectInsertions(plan_it->second.insert_before, {i, -1}, pending_ops, result);

      if (auto op_stmts = As<OpStmts>(seq->stmts_[i])) {
        // Merge pending_ops as prefix into this OpStmts
        BuildOpStmtsWithInsertions(op_stmts, i, has_plan ? &plan_it->second : nullptr, pending_ops, result);
        // pending_ops is now clear (consumed by BuildOpStmtsWithInsertions)
      } else if (auto if_stmt = As<IfStmt>(seq->stmts_[i])) {
        flush_pending();
        path.push_back({PathElement::Kind::SeqIndex, i});
        result.push_back(ApplyToIfStmt(if_stmt, path));
        path.pop_back();
      } else if (auto for_stmt = As<ForStmt>(seq->stmts_[i])) {
        flush_pending();
        path.push_back({PathElement::Kind::SeqIndex, i});
        result.push_back(ApplyToForStmt(for_stmt, path));
        path.pop_back();
      } else {
        flush_pending();
        result.push_back(seq->stmts_[i]);
      }

      // Collect insert_after at SeqStmts level into pending_ops
      if (has_plan) CollectInsertions(plan_it->second.insert_after, {i, -1}, pending_ops, result);
    }

    // Handle catch-all positions targeting past the last child
    if (has_plan) {
      CollectInsertions(plan_it->second.insert_before, {static_cast<int>(seq->stmts_.size()), -1},
                        pending_ops, result);
    }
    flush_pending();

    return std::make_shared<const SeqStmts>(result, seq->span_);
  }

  static void CollectInsertions(const std::map<InsertionPlan::PosKey, std::vector<StmtPtr>>& plan_map,
                                InsertionPlan::PosKey key, std::vector<StmtPtr>& pending_ops,
                                std::vector<StmtPtr>& result) {
    auto it = plan_map.find(key);
    if (it == plan_map.end()) return;
    for (const auto& stmt : it->second) {
      if (As<AssignStmt>(stmt) || As<EvalStmt>(stmt)) {
        pending_ops.push_back(stmt);
      } else {
        if (!pending_ops.empty()) {
          result.push_back(std::make_shared<const OpStmts>(pending_ops, pending_ops[0]->span_));
          pending_ops.clear();
        }
        result.push_back(stmt);
      }
    }
  }

  static void BuildOpStmtsWithInsertions(const OpStmtsPtr& op_stmts, int seq_idx, const InsertionPlan* plan,
                                         std::vector<StmtPtr>& pending_ops, std::vector<StmtPtr>& result) {
    // Start with pending_ops as prefix
    std::vector<StmtPtr> current_ops = std::move(pending_ops);
    pending_ops.clear();

    auto flush_ops = [&]() {
      if (!current_ops.empty()) {
        result.push_back(std::make_shared<const OpStmts>(current_ops, current_ops[0]->span_));
        current_ops.clear();
      }
    };

    for (int j = 0; j < static_cast<int>(op_stmts->stmts_.size()); ++j) {
      if (plan) {
        auto before_it = plan->insert_before.find({seq_idx, j});
        if (before_it != plan->insert_before.end()) {
          for (const auto& stmt : before_it->second) {
            if (As<AssignStmt>(stmt) || As<EvalStmt>(stmt)) {
              current_ops.push_back(stmt);
            } else {
              flush_ops();
              result.push_back(stmt);
            }
          }
        }
      }

      current_ops.push_back(op_stmts->stmts_[j]);

      if (plan) {
        auto after_it = plan->insert_after.find({seq_idx, j});
        if (after_it != plan->insert_after.end()) {
          for (const auto& stmt : after_it->second) {
            if (As<AssignStmt>(stmt) || As<EvalStmt>(stmt)) {
              current_ops.push_back(stmt);
            } else {
              flush_ops();
              result.push_back(stmt);
            }
          }
        }
      }
    }
    flush_ops();
  }

  StmtPtr ApplyToIfStmt(const IfStmtPtr& if_stmt, std::vector<PathElement>& path) {
    path.push_back({PathElement::Kind::IfThen, -1});
    auto new_then = ApplyInsertions(if_stmt->then_body_, path);
    path.pop_back();

    std::optional<StmtPtr> new_else = std::nullopt;
    if (if_stmt->else_body_) {
      path.push_back({PathElement::Kind::IfElse, -1});
      new_else = ApplyInsertions(*if_stmt->else_body_, path);
      path.pop_back();
    }
    return std::make_shared<const IfStmt>(if_stmt->condition_, new_then, new_else, if_stmt->return_vars_,
                                          if_stmt->span_);
  }

  StmtPtr ApplyToForStmt(const ForStmtPtr& for_stmt, std::vector<PathElement>& path) {
    path.push_back({PathElement::Kind::ForBody, -1});
    auto new_body = ApplyInsertions(for_stmt->body_, path);
    path.pop_back();
    return std::make_shared<const ForStmt>(for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_,
                                           for_stmt->step_, for_stmt->iter_args_, new_body,
                                           for_stmt->return_vars_, for_stmt->span_, for_stmt->kind_);
  }
};

}  // namespace

namespace pass {
Pass InsertSync() {
  return CreateFunctionPass(
      [](const FunctionPtr& func) {
        SyncInserter inserter;
        return inserter.Run(func);
      },
      "InsertSync", kInsertSyncProperties);
}
}  // namespace pass
}  // namespace ir
}  // namespace pypto

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
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_config.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/normalize_stmt_structure.h"
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

  [[nodiscard]] bool IsInForBody() const {
    for (const auto& elem : path) {
      if (elem.kind == PathElement::Kind::ForBody) return true;
    }
    return false;
  }

  // Determines if this position and another are within the same control flow scope (e.g., same block or
  // branches)
  [[nodiscard]] bool IsInSameScope(const Position& other) const {
    // Helper: Find the innermost SeqIndex by searching backwards.
    auto get_scope_anchor_idx = [](const std::vector<PathElement>& p) -> int {
      for (int i = static_cast<int>(p.size()) - 1; i >= 0; --i) {
        if (p[i].kind == PathElement::Kind::SeqIndex) return i;
      }
      return -1;  // Empty or invalid path
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

bool IsSameMem(const MemRefPtr& a, const MemRefPtr& b) { return a.get() == b.get(); }

std::set<MemRefPtr> GetExprMemRefs(const ExprPtr& expr) {
  MemRefCollector collector;
  collector.VisitExpr(expr);
  return collector.memrefs;
}

PipeType GetPipeForCall(const Call* call) {
  if (call->op_->GetPipe().has_value()) return *call->op_->GetPipe();
  const pypto::backend::Backend* be = pypto::backend::GetBackend();
  const auto* info = be->GetOpInfo(call->op_->name_);
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
    // block.store: args[0] (tile) is read, args.back() (output_tensor) is write
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
  int id;
  PipeType producer_pipe;
  PipeType consumer_pipe;
  int event_id = -1;
  Position set_position;
  Position wait_position;
  bool set_emits_sync_id = false;
  bool wait_needs_if = false;
  bool wait_clears_sync_id = false;
  int sync_id_index = -1;

  [[nodiscard]] bool IsSamePipe() const { return producer_pipe == consumer_pipe; }
};

struct SyncGroup {
  PipeType producer_pipe;
  PipeType consumer_pipe;
  std::vector<size_t> pair_indices;
  std::set<Position> set_positions;
  std::map<Position, std::vector<size_t>> wait_to_pair_indices;
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
  int next_pair_id = 0;
  int next_sync_id_index = 0;
  std::map<int, VarPtr> sync_id_vars;
  std::map<Position, StmtPtr> pos_to_stmt;

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

  int AllocatePairId() { return next_pair_id++; }

  int AllocateSyncIdIndex() {
    int idx = next_sync_id_index++;
    std::string name = "sync_id_" + std::to_string(idx);
    auto bool_type = std::make_shared<ScalarType>(DataType::BOOL);
    sync_id_vars[idx] = std::make_shared<Var>(name, bool_type, Span::unknown());
    return idx;
  }

  [[nodiscard]] VarPtr GetSyncIdVar(int idx) const {
    auto it = sync_id_vars.find(idx);
    if (it != sync_id_vars.end()) return it->second;
    return nullptr;
  }
};

class CoverageAnalyzer {
  const std::set<Position>& set_positions_;

 public:
  explicit CoverageAnalyzer(const std::set<Position>& sp) : set_positions_(sp) {}

  // Checks if wait_pos is fully covered by upstream set_positions
  [[nodiscard]] bool IsCovered(const Position& wait_pos) const {
    std::vector<PathElement> current_prefix;

    // Walk down the path of the wait position
    for (const auto& next_elem : wait_pos.path) {
      if (next_elem.kind == PathElement::Kind::SeqIndex || next_elem.kind == PathElement::Kind::OpIndex) {
        // Check if any upstream statement within the same block guarantees a set
        for (int i = 0; i < next_elem.index; ++i) {
          std::vector<PathElement> check_prefix = current_prefix;
          check_prefix.push_back({next_elem.kind, i});

          if (IsPrefixGuaranteed(check_prefix)) {
            return true;
          }
        }
      }
      current_prefix.push_back(next_elem);
    }
    return false;
  }

  // Evaluates whether reaching prefix path P guarantees execution of a set operation
  [[nodiscard]] bool IsPrefixGuaranteed(const std::vector<PathElement>& P) const {
    // 1. Leaf hit: The prefix itself is exactly a known set position
    Position pos{P};
    if (set_positions_.count(pos)) return true;

    // 2. Collect all possible branches beneath this prefix
    std::set<int> seq_indices;
    std::set<int> op_indices;
    bool has_ifthen = false;
    bool has_ifelse = false;
    bool has_forbody = false;

    for (const auto& s : set_positions_) {
      if (s.path.size() > P.size()) {
        bool match = true;
        for (size_t i = 0; i < P.size(); ++i) {
          if (!(s.path[i] == P[i])) {
            match = false;
            break;
          }
        }
        if (match) {
          const auto& next_elem = s.path[P.size()];
          if (next_elem.kind == PathElement::Kind::SeqIndex) {
            seq_indices.insert(next_elem.index);
          } else if (next_elem.kind == PathElement::Kind::OpIndex) {
            op_indices.insert(next_elem.index);
          } else if (next_elem.kind == PathElement::Kind::IfThen) {
            has_ifthen = true;
          } else if (next_elem.kind == PathElement::Kind::IfElse) {
            has_ifelse = true;
          } else if (next_elem.kind == PathElement::Kind::ForBody) {
            has_forbody = true;
          }
        }
      }
    }

    // 3. Sequence Rule: Any fully covered statement in a sequence block covers the whole block
    for (int idx : seq_indices) {
      std::vector<PathElement> next_P = P;
      next_P.push_back({PathElement::Kind::SeqIndex, idx});
      if (IsPrefixGuaranteed(next_P)) return true;
    }

    // 3b. OpIndex Rule: Any fully covered op within an OpStmts covers the whole OpStmts
    for (int idx : op_indices) {
      std::vector<PathElement> next_P = P;
      next_P.push_back({PathElement::Kind::OpIndex, idx});
      if (IsPrefixGuaranteed(next_P)) return true;
    }

    // 4. Branch Reduction Rule: Both Then and Else paths exist and guarantee a Set
    if (has_ifthen && has_ifelse) {
      std::vector<PathElement> then_P = P;
      then_P.push_back({PathElement::Kind::IfThen, -1});
      std::vector<PathElement> else_P = P;
      else_P.push_back({PathElement::Kind::IfElse, -1});

      if (IsPrefixGuaranteed(then_P) && IsPrefixGuaranteed(else_P)) {
        return true;
      }
    }

    // 5. Loop Rule: Assume loop executes at least once; if for body guarantees coverage, then covered
    if (has_forbody) {
      std::vector<PathElement> for_P = P;
      for_P.push_back({PathElement::Kind::ForBody, -1});
      if (IsPrefixGuaranteed(for_P)) return true;
    }

    return false;
  }
};

struct UnionFind {
  std::vector<int> parent;
  explicit UnionFind(int n) : parent(n) {
    for (int i = 0; i < n; ++i) parent[i] = i;
  }
  int Find(int i) {
    if (parent[i] == i) return i;
    return parent[i] = Find(parent[i]);
  }
  void Union(int i, int j) {
    int root_i = Find(i);
    int root_j = Find(j);
    if (root_i != root_j) parent[root_i] = root_j;
  }
};

// --------------------------------------------------------------------------
// Main Inserter Pass
// --------------------------------------------------------------------------

class SyncInserter {
 public:
  SyncInserter() = default;

  FunctionPtr Run(const FunctionPtr& func) {
    // Normalize input: ensure all leaf stmts are wrapped in OpStmts
    auto normalized = NormalizeStmtStructure(func);

    ctx_ = AnalysisContext();

    // Phase 1: Collect raw sync pairs
    CollectSyncPairs(normalized->body_);

    // Phase 2: Hierarchical Control Flow Analysis & Grouping
    AnalyzeConditionalExecution();

    // Phase 3: Assign Hardware Event IDs
    AssignEventIds();

    // Phase 4: Build Insertion Plans and mutate AST
    BuildInsertionPlans();
    std::vector<PathElement> path;
    auto new_body = ApplyInsertions(normalized->body_, path);
    new_body = AddSyncIdInitializations(new_body);

    return std::make_shared<Function>(normalized->name_, normalized->params_, normalized->return_types_,
                                      new_body, normalized->span_, normalized->func_type_);
  }

 private:
  AnalysisContext ctx_;
  std::vector<SyncGroup> final_groups_;
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

  void CollectFromSeqStmts(const SeqStmtsPtr& seq, MemRefSummary& state) {
    size_t pairs_start_idx = ctx_.sync_pairs.size();

    for (int i = 0; i < static_cast<int>(seq->stmts_.size()); ++i) {
      const auto& stmt = seq->stmts_[i];
      ctx_.EnterSeq(i);

      if (auto op_stmts = As<OpStmts>(stmt)) {
        // Traverse into OpStmts, processing each leaf statement
        for (int j = 0; j < static_cast<int>(op_stmts->stmts_.size()); ++j) {
          ctx_.EnterOp(j);
          Position current_pos = ctx_.CurrentPosition();
          const auto& leaf_stmt = op_stmts->stmts_[j];

          auto [reads, writes] = GetLeafMemRefs(leaf_stmt);
          ctx_.pos_to_stmt[current_pos] = leaf_stmt;

          for (const auto& r : reads) {
            for (const auto& [m, writers] : state.last_writers) {
              if (IsSameMem(r, m)) {
                for (const auto& writer_pos : writers) CreateSyncPair(writer_pos, current_pos);
              }
            }
          }
          for (const auto& w : writes) {
            for (const auto& [m, writers] : state.last_writers) {
              if (IsSameMem(w, m)) {
                for (const auto& writer_pos : writers) CreateSyncPair(writer_pos, current_pos);
              }
            }
          }
          for (const auto& w : writes) {
            for (const auto& [m, readers] : state.last_readers) {
              if (IsSameMem(w, m)) {
                for (const auto& reader_pos : readers) CreateSyncPair(reader_pos, current_pos);
              }
            }
          }

          for (const auto& w : writes) {
            state.last_writers[w] = {current_pos};
            state.last_readers[w].clear();
          }
          for (const auto& r : reads) {
            state.last_readers[r].push_back(current_pos);
          }
          ctx_.Leave();  // Leave OpIndex
        }
      } else if (auto if_stmt = As<IfStmt>(stmt)) {
        CollectFromIfStmt(if_stmt, state);
      } else if (auto for_stmt = As<ForStmt>(stmt)) {
        CollectFromForStmt(for_stmt, state);
      } else {
        // Bare leaf stmt (ReturnStmt, YieldStmt, or unnormalized AssignStmt/EvalStmt)
        Position current_pos = ctx_.CurrentPosition();
        auto [reads, writes] = GetLeafMemRefs(stmt);
        if (!reads.empty() || !writes.empty()) {
          ctx_.pos_to_stmt[current_pos] = stmt;

          for (const auto& r : reads) {
            for (const auto& [m, writers] : state.last_writers) {
              if (IsSameMem(r, m)) {
                for (const auto& writer_pos : writers) CreateSyncPair(writer_pos, current_pos);
              }
            }
          }
          for (const auto& w : writes) {
            for (const auto& [m, writers] : state.last_writers) {
              if (IsSameMem(w, m)) {
                for (const auto& writer_pos : writers) CreateSyncPair(writer_pos, current_pos);
              }
            }
          }
          for (const auto& w : writes) {
            for (const auto& [m, readers] : state.last_readers) {
              if (IsSameMem(w, m)) {
                for (const auto& reader_pos : readers) CreateSyncPair(reader_pos, current_pos);
              }
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
      }
      ctx_.Leave();  // Leave SeqIndex
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

    std::vector<StmtPtr> unrolled_stmts;
    unrolled_stmts.reserve(static_cast<size_t>(body_size) * 2);
    unrolled_stmts.insert(unrolled_stmts.end(), seq->stmts_.begin(), seq->stmts_.end());
    unrolled_stmts.insert(unrolled_stmts.end(), seq->stmts_.begin(), seq->stmts_.end());
    auto unrolled_seq = std::make_shared<SeqStmts>(unrolled_stmts, seq->span_);

    CollectFromSeqStmts(unrolled_seq, state);
    RemoveTransitiveRedundantPairs(pairs_start_idx);
    RemoveLinearRedundantPairs(pairs_start_idx);
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
    pair.id = ctx_.AllocatePairId();
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

  void RemoveLinearRedundantPairs(size_t start_idx) {
    size_t count = ctx_.sync_pairs.size();
    std::vector<bool> is_redundant(count, false);

    for (size_t i = start_idx; i < count; ++i) {
      if (is_redundant[i]) continue;
      const auto& p1 = ctx_.sync_pairs[i];

      for (size_t j = i + 1; j < count; ++j) {
        if (is_redundant[j]) continue;
        const auto& p2 = ctx_.sync_pairs[j];

        // Must share the exact producer and consumer pipes
        if (p1.producer_pipe != p2.producer_pipe || p1.consumer_pipe != p2.consumer_pipe) {
          continue;
        }

        // Rule A: Same wait_position -> keep the LATEST set_position
        if (p1.wait_position == p2.wait_position) {
          if (p1.set_position.IsBefore(p2.set_position)) {
            is_redundant[i] = true;  // p1 is earlier, covered by p2
            break;
          } else if (p2.set_position.IsBefore(p1.set_position)) {
            is_redundant[j] = true;  // p2 is earlier, covered by p1
          }
        } else if (p1.set_position == p2.set_position) {
          // Rule B: Same set_position -> keep the EARLIEST wait_position
          if (p1.wait_position.IsBefore(p2.wait_position)) {
            is_redundant[j] = true;  // p2 is later, sheltered by p1
          } else if (p2.wait_position.IsBefore(p1.wait_position)) {
            is_redundant[i] = true;  // p1 is later, sheltered by p2
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
  // Phase 2: Hierarchical Control Flow Analysis
  // --------------------------------------------------------------------------

  std::vector<SyncGroup> BuildSyncGroups() {
    auto n = static_cast<int>(ctx_.sync_pairs.size());
    UnionFind uf(n);

    for (int i = 0; i < n; ++i) {
      if (ctx_.sync_pairs[i].IsSamePipe()) continue;
      for (int j = i + 1; j < n; ++j) {
        if (ctx_.sync_pairs[j].IsSamePipe()) continue;
        auto& a = ctx_.sync_pairs[i];
        auto& b = ctx_.sync_pairs[j];
        if (a.producer_pipe == b.producer_pipe && a.consumer_pipe == b.consumer_pipe) {
          if (a.set_position == b.set_position || a.wait_position == b.wait_position) {
            uf.Union(i, j);
          }
        }
      }
    }

    std::map<int, SyncGroup> group_map;
    for (int i = 0; i < n; ++i) {
      if (ctx_.sync_pairs[i].IsSamePipe()) continue;
      int root = uf.Find(i);
      auto& pair = ctx_.sync_pairs[i];
      auto& grp = group_map[root];
      grp.producer_pipe = pair.producer_pipe;
      grp.consumer_pipe = pair.consumer_pipe;
      grp.pair_indices.push_back(i);
      grp.set_positions.insert(pair.set_position);
      grp.wait_to_pair_indices[pair.wait_position].push_back(i);
    }

    std::vector<SyncGroup> result;
    result.reserve(group_map.size());
    for (auto& [_, v] : group_map) result.push_back(std::move(v));
    return result;
  }

  void CheckGroupOverlaps(const SyncGroup& grp) {
    std::vector<Position> waits;
    waits.reserve(grp.wait_to_pair_indices.size());
    for (const auto& [w, _] : grp.wait_to_pair_indices) waits.push_back(w);
    for (size_t i = 0; i < waits.size(); ++i) {
      for (size_t j = i + 1; j < waits.size(); ++j) {
        if (waits[i].IsBefore(waits[j]) || waits[j].IsBefore(waits[i])) {
          throw ValueError("Invalid sync siblings: Overlapping wait paths detected for the same set.");
        }
      }
    }
  }

  void AnalyzeConditionalExecution() {
    auto initial_groups = BuildSyncGroups();
    std::vector<SyncPair> catch_all_pairs;

    for (auto& grp : initial_groups) {
      CheckGroupOverlaps(grp);
      AnalyzeGroupAndProcess(grp, catch_all_pairs);
    }

    AddSyncAfterCrossIteration(catch_all_pairs);

    for (auto& cp : catch_all_pairs) {
      ctx_.sync_pairs.push_back(std::move(cp));
    }

    final_groups_ = BuildSyncGroups();
  }

  void AnalyzeGroupAndProcess(const SyncGroup& grp, std::vector<SyncPair>& catch_all_pairs) {
    CoverageAnalyzer analyzer(grp.set_positions);
    bool group_needs_sync_id = false;
    int max_branch_seq_index = -1;
    Position base_set_pos;

    // Check if inner is inside a loop body that outer is not in
    auto is_loop_boundary_crossed = [](const Position& inner, const Position& outer) {
      for (size_t i = 0; i < inner.path.size(); ++i) {
        if (inner.path[i].kind == PathElement::Kind::ForBody) {
          // If outer path is shorter, or diverged before entering this loop
          if (i >= outer.path.size() || !(outer.path[i] == inner.path[i])) return true;
        }
      }
      return false;
    };

    // Collect all wait positions to build backward analyzer for token escape detection
    std::set<Position> wait_positions;
    for (const auto& [w, _] : grp.wait_to_pair_indices) {
      wait_positions.insert(w);
    }
    CoverageAnalyzer wait_analyzer(wait_positions);

    // 1. Check coverage of each wait independently
    std::map<Position, bool> wait_covered;
    for (const auto& [wait_pos, indices] : grp.wait_to_pair_indices) {
      bool covered = analyzer.IsCovered(wait_pos);
      wait_covered[wait_pos] = covered;
      if (!covered) group_needs_sync_id = true;

      for (size_t idx : indices) {
        if (wait_pos.IsBefore(ctx_.sync_pairs[idx].set_position)) {
          group_needs_sync_id = true;
        }
        // If loop boundary is crossed (in or out), force sync_id
        if (is_loop_boundary_crossed(wait_pos, ctx_.sync_pairs[idx].set_position) ||
            is_loop_boundary_crossed(ctx_.sync_pairs[idx].set_position, wait_pos)) {
          group_needs_sync_id = true;
        }
      }
    }

    // 2. Check Token Leakage (Set Escape)
    for (size_t idx : grp.pair_indices) {
      auto& pair = ctx_.sync_pairs[idx];
      if (!pair.set_position.IsInSameScope(pair.wait_position) &&
          !pair.wait_position.IsBefore(pair.set_position)) {
        // Compute effective scope depth (strip trailing OpIndex from set position)
        size_t set_effective_len = pair.set_position.path.size();
        if (set_effective_len > 0 && pair.set_position.path.back().kind == PathElement::Kind::OpIndex) {
          set_effective_len--;
        }
        size_t scope_depth = set_effective_len - 1;
        if (scope_depth < pair.wait_position.path.size()) {
          int branch_idx = pair.wait_position.path[scope_depth].index;

          std::vector<PathElement> branch_prefix;
          for (size_t i = 0; i <= scope_depth; ++i) {
            if (i == scope_depth) {
              branch_prefix.push_back({PathElement::Kind::SeqIndex, branch_idx});
            } else {
              branch_prefix.push_back(pair.set_position.path[i]);
            }
          }

          // If this diverging branch (e.g. IfStmt) guarantees hitting a wait 100%,
          // then token is perfectly consumed and won't escape, so no sync_id needed!
          if (!wait_analyzer.IsPrefixGuaranteed(branch_prefix)) {
            group_needs_sync_id = true;
            if (branch_idx > max_branch_seq_index) {
              max_branch_seq_index = branch_idx;
              base_set_pos = pair.set_position;
            }
          }
        }
      }
    }

    int shared_sync_id = -1;
    if (group_needs_sync_id) shared_sync_id = ctx_.AllocateSyncIdIndex();

    // 2. Precise assignment strategy
    for (size_t idx : grp.pair_indices) {
      auto& pair = ctx_.sync_pairs[idx];
      pair.sync_id_index = shared_sync_id;
      if (group_needs_sync_id) pair.set_emits_sync_id = true;

      bool covered = wait_covered[pair.wait_position];
      // Force Loop Boundary Crossing to use Conditional
      bool is_crossing = is_loop_boundary_crossed(pair.wait_position, pair.set_position) ||
                         is_loop_boundary_crossed(pair.set_position, pair.wait_position);
      if (is_crossing || pair.wait_position.IsBefore(pair.set_position) || !covered) {
        pair.wait_needs_if = true;
      } else {
        // If wait is 100% covered, no conditional check needed!
        pair.wait_needs_if = false;
        // But if group tracking is enabled, clear flag after consumption
        if (group_needs_sync_id) pair.wait_clears_sync_id = true;
      }
    }

    if (group_needs_sync_id && max_branch_seq_index != -1) {
      // Strip trailing OpIndex from base_set_pos to get the SeqStmts-level prefix
      size_t base_effective_len = base_set_pos.path.size();
      if (base_effective_len > 0 && base_set_pos.path.back().kind == PathElement::Kind::OpIndex) {
        base_effective_len--;
      }
      Position catch_pos;
      for (size_t i = 0; i + 1 < base_effective_len; ++i) {
        catch_pos.path.push_back(base_set_pos.path[i]);
      }
      catch_pos.path.push_back({PathElement::Kind::SeqIndex, max_branch_seq_index + 1});

      SyncPair cp;
      cp.id = ctx_.AllocatePairId();
      cp.producer_pipe = grp.producer_pipe;
      cp.consumer_pipe = grp.consumer_pipe;
      cp.set_position = base_set_pos;
      cp.wait_position = catch_pos;
      cp.set_emits_sync_id = false;
      cp.wait_needs_if = true;
      cp.sync_id_index = shared_sync_id;
      catch_all_pairs.push_back(std::move(cp));
    }
  }

  void AddSyncAfterCrossIteration(std::vector<SyncPair>& catch_all_pairs) {
    for (const auto& pair : ctx_.sync_pairs) {
      if (pair.IsSamePipe()) continue;
      if (pair.sync_id_index == -1) continue;
      if (!pair.set_position.IsInForBody() || !pair.wait_position.IsInForBody()) continue;
      if (!pair.set_position.IsInSameScope(pair.wait_position)) continue;
      if (!pair.wait_position.IsBefore(pair.set_position)) continue;

      Position after_for_pos;
      bool found_for = false;
      for (size_t i = 0; i < pair.wait_position.path.size(); ++i) {
        const auto& elem = pair.wait_position.path[i];
        if (elem.kind == PathElement::Kind::ForBody && !found_for) {
          found_for = true;
          if (i > 0 && after_for_pos.path.back().kind == PathElement::Kind::SeqIndex) {
            after_for_pos.path.back().index += 1;
          }
          break;
        }
        after_for_pos.path.push_back(elem);
      }
      if (!found_for) continue;

      SyncPair comp_pair;
      comp_pair.id = ctx_.AllocatePairId();
      comp_pair.producer_pipe = pair.producer_pipe;
      comp_pair.consumer_pipe = pair.consumer_pipe;
      comp_pair.set_position = pair.set_position;
      comp_pair.wait_position = after_for_pos;
      comp_pair.set_emits_sync_id = false;
      comp_pair.wait_needs_if = true;
      comp_pair.sync_id_index = pair.sync_id_index;
      catch_all_pairs.push_back(std::move(comp_pair));
    }
  }

  // --------------------------------------------------------------------------
  // Phase 3: Event ID Allocation
  // --------------------------------------------------------------------------

  void AssignEventIds() {
    EventIdManager event_manager;
    std::vector<SyncGroup*> sorted_groups;
    sorted_groups.reserve(final_groups_.size());
    for (auto& grp : final_groups_) sorted_groups.push_back(&grp);

    std::sort(sorted_groups.begin(), sorted_groups.end(), [](SyncGroup* a, SyncGroup* b) {
      return *(a->set_positions.begin()) < *(b->set_positions.begin());
    });

    for (auto* grp : sorted_groups) {
      Position earliest_set = *(grp->set_positions.begin());
      int event_id = event_manager.Alloc(grp->producer_pipe, grp->consumer_pipe, earliest_set);

      for (size_t idx : grp->pair_indices) {
        ctx_.sync_pairs[idx].event_id = event_id;
        event_manager.Free(grp->producer_pipe, grp->consumer_pipe, ctx_.sync_pairs[idx].wait_position,
                           event_id);
      }
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
    std::map<std::vector<PathElement>, std::set<std::tuple<PipeType, PipeType, int>>> scope_sync_src;
    std::map<std::vector<PathElement>, std::set<std::tuple<PipeType, PipeType, int>>> scope_sync_dst;
    std::map<std::vector<PathElement>, std::set<std::pair<InsertionPlan::PosKey, PipeType>>> scope_bars;

    for (const auto& pair : ctx_.sync_pairs) {
      if (!pair.IsSamePipe()) {
        auto set_parent = GetParentPath(pair.set_position);
        auto set_key = GetPlanIndex(pair.set_position);
        if (set_key.first >= 0) {
          auto src_key = std::make_tuple(pair.producer_pipe, pair.consumer_pipe, pair.event_id);
          if (!scope_sync_src[set_parent].count(src_key)) {
            auto& plan = insertion_plans_[set_parent];
            plan.insert_after[set_key].push_back(
                CreateSyncCall("system.sync_src", pair.producer_pipe, pair.consumer_pipe, pair.event_id));
            if (pair.set_emits_sync_id) {
              plan.insert_after[set_key].push_back(CreateSyncIdAssign(pair.sync_id_index, true));
            }
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
            if (pair.producer_pipe == PipeType::V) {
              plan.insert_before[wait_key].push_back(CreateBarCall("system.bar_v"));
            } else if (pair.producer_pipe == PipeType::M) {
              plan.insert_before[wait_key].push_back(CreateBarCall("system.bar_m"));
            }
            scope_bars[wait_parent].insert(bar_key);
          }
        } else {
          auto dst_key = std::make_tuple(pair.producer_pipe, pair.consumer_pipe, pair.event_id);
          if (!scope_sync_dst[wait_parent].count(dst_key)) {
            auto& plan = insertion_plans_[wait_parent];
            if (pair.wait_needs_if) {
              plan.insert_before[wait_key].push_back(CreateConditionalWait(pair));
            } else {
              plan.insert_before[wait_key].push_back(
                  CreateSyncCall("system.sync_dst", pair.producer_pipe, pair.consumer_pipe, pair.event_id));
              if (pair.wait_clears_sync_id) {
                plan.insert_before[wait_key].push_back(CreateSyncIdAssign(pair.sync_id_index, false));
              }
            }
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

    for (int i = 0; i < static_cast<int>(seq->stmts_.size()); ++i) {
      if (has_plan) EmitInsertions(plan_it->second.insert_before, {i, -1}, result);

      if (auto op_stmts = As<OpStmts>(seq->stmts_[i])) {
        BuildOpStmtsWithInsertions(op_stmts, i, has_plan ? &plan_it->second : nullptr, result);
      } else if (auto if_stmt = As<IfStmt>(seq->stmts_[i])) {
        path.push_back({PathElement::Kind::SeqIndex, i});
        result.push_back(ApplyToIfStmt(if_stmt, path));
        path.pop_back();
      } else if (auto for_stmt = As<ForStmt>(seq->stmts_[i])) {
        path.push_back({PathElement::Kind::SeqIndex, i});
        result.push_back(ApplyToForStmt(for_stmt, path));
        path.pop_back();
      } else {
        result.push_back(seq->stmts_[i]);
      }

      if (has_plan) EmitInsertions(plan_it->second.insert_after, {i, -1}, result);
    }

    // Handle catch-all positions targeting past the last child
    if (has_plan) {
      EmitInsertions(plan_it->second.insert_before, {static_cast<int>(seq->stmts_.size()), -1}, result);
    }

    return std::make_shared<const SeqStmts>(result, seq->span_);
  }

  // Emit insertions for a child-level key, wrapping AssignStmt/EvalStmt in OpStmts
  static void EmitInsertions(const std::map<InsertionPlan::PosKey, std::vector<StmtPtr>>& plan_map,
                             InsertionPlan::PosKey key, std::vector<StmtPtr>& result) {
    auto it = plan_map.find(key);
    if (it == plan_map.end()) return;
    std::vector<StmtPtr> op_buf;
    auto flush_op_buf = [&]() {
      if (!op_buf.empty()) {
        result.push_back(std::make_shared<const OpStmts>(op_buf, op_buf[0]->span_));
        op_buf.clear();
      }
    };
    for (const auto& stmt : it->second) {
      if (As<AssignStmt>(stmt) || As<EvalStmt>(stmt)) {
        op_buf.push_back(stmt);
      } else {
        flush_op_buf();
        result.push_back(stmt);
      }
    }
    flush_op_buf();
  }

  // Build OpStmts children directly with insertions, splitting on non-op-compatible stmts
  static void BuildOpStmtsWithInsertions(const OpStmtsPtr& op_stmts, int seq_idx, const InsertionPlan* plan,
                                         std::vector<StmtPtr>& result) {
    std::vector<StmtPtr> current_ops;
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
                                           for_stmt->return_vars_, for_stmt->span_);
  }

  StmtPtr CreateSyncIdAssign(int sync_id_index, bool value) {
    auto var = ctx_.GetSyncIdVar(sync_id_index);
    auto bool_const = std::make_shared<ConstInt>(value ? 1 : 0, DataType::BOOL, Span::unknown());
    return std::make_shared<const AssignStmt>(var, bool_const, Span::unknown());
  }

  StmtPtr CreateConditionalWait(const SyncPair& pair) {
    auto var = ctx_.GetSyncIdVar(pair.sync_id_index);
    ExprPtr condition = var;
    std::vector<StmtPtr> then_stmts;
    then_stmts.push_back(
        CreateSyncCall("system.sync_dst", pair.producer_pipe, pair.consumer_pipe, pair.event_id));
    auto false_const = std::make_shared<ConstInt>(0, DataType::BOOL, Span::unknown());
    then_stmts.push_back(std::make_shared<const AssignStmt>(var, false_const, Span::unknown()));
    // Wrap in OpStmts inside SeqStmts for normalized structure
    auto op_stmts = std::make_shared<const OpStmts>(then_stmts, Span::unknown());
    auto then_body = std::make_shared<const SeqStmts>(std::vector<StmtPtr>{op_stmts}, Span::unknown());
    return std::make_shared<const IfStmt>(condition, then_body, std::nullopt, std::vector<VarPtr>{},
                                          Span::unknown());
  }

  StmtPtr AddSyncIdInitializations(const StmtPtr& body) {
    if (ctx_.sync_id_vars.empty()) return body;

    std::vector<StmtPtr> init_stmts;
    for (const auto& [idx, var] : ctx_.sync_id_vars) {
      auto false_const = std::make_shared<ConstInt>(0, DataType::BOOL, Span::unknown());
      init_stmts.push_back(std::make_shared<const AssignStmt>(var, false_const, Span::unknown()));
    }

    if (auto seq = As<SeqStmts>(body)) {
      std::vector<StmtPtr> new_children;
      if (!seq->stmts_.empty()) {
        if (auto first_op = As<OpStmts>(seq->stmts_[0])) {
          // Merge init stmts into the beginning of the first OpStmts
          std::vector<StmtPtr> merged;
          merged.insert(merged.end(), init_stmts.begin(), init_stmts.end());
          merged.insert(merged.end(), first_op->stmts_.begin(), first_op->stmts_.end());
          new_children.push_back(std::make_shared<const OpStmts>(merged, first_op->span_));
          for (size_t i = 1; i < seq->stmts_.size(); ++i) {
            new_children.push_back(seq->stmts_[i]);
          }
        } else {
          // First child is not OpStmts — create a new OpStmts for init stmts
          new_children.push_back(std::make_shared<const OpStmts>(init_stmts, Span::unknown()));
          new_children.insert(new_children.end(), seq->stmts_.begin(), seq->stmts_.end());
        }
      } else {
        new_children.push_back(std::make_shared<const OpStmts>(init_stmts, Span::unknown()));
      }
      return std::make_shared<const SeqStmts>(new_children, seq->span_);
    }
    // Fallback: wrap in SeqStmts
    std::vector<StmtPtr> children;
    children.push_back(std::make_shared<const OpStmts>(init_stmts, Span::unknown()));
    children.push_back(body);
    return std::make_shared<const SeqStmts>(children, body->span_);
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
      "InsertSync");
}
}  // namespace pass
}  // namespace ir
}  // namespace pypto

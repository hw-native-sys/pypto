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
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/**
 * @brief Collector for all MemRefs in an expression
 */
class MemRefCollector : public IRVisitor {
 public:
  std::set<MemRefPtr> memrefs;

  void VisitExpr_(const VarPtr& var) override {
    if (auto shaped_type = As<ShapedType>(var->GetType())) {
      if (shaped_type->memref_.has_value()) {
        memrefs.insert(*shaped_type->memref_);
      }
    }
    IRVisitor::VisitExpr_(var);
  }
};

/**
 * @brief Helper to check if two MemRefs refer to the same memory
 */
bool IsSameMem(const MemRefPtr& a, const MemRefPtr& b) { return a.get() == b.get(); }

/**
 * @brief Extract MemRefs from an expression
 */
std::set<MemRefPtr> GetExprMemRefs(const ExprPtr& expr) {
  MemRefCollector collector;
  collector.VisitExpr(expr);
  return collector.memrefs;
}

/**
 * @brief Structure to represent a dependency edge
 */
struct DepEdge {
  int producer_idx;
  int consumer_idx;
  PipeType producer_pipe;
  PipeType consumer_pipe;
  int event_id = -1;  // Assigned later
};

/**
 * @brief Create a sync call statement (sync_src or sync_dst)
 */
StmtPtr CreateSyncCall(const std::string& op_name, PipeType p, PipeType tp, int event_id) {
  auto& registry = OpRegistry::GetInstance();
  std::vector<std::pair<std::string, std::any>> kwargs;
  kwargs.emplace_back("set_pipe", static_cast<int>(p));
  kwargs.emplace_back("wait_pipe", static_cast<int>(tp));
  kwargs.emplace_back("event_id", event_id);
  auto call = registry.Create(op_name, {}, kwargs, Span::unknown());
  return std::make_shared<const EvalStmt>(call, Span::unknown());
}

/**
 * @brief Create a barrier call statement (bar_v or bar_m)
 */
StmtPtr CreateBarCall(const std::string& op_name) {
  auto& registry = OpRegistry::GetInstance();
  auto call = registry.Create(op_name, {}, {}, Span::unknown());
  return std::make_shared<const EvalStmt>(call, Span::unknown());
}

/**
 * @brief Manager for hardware event IDs (0-7) per SRC-DST pipe pair
 */
class EventIdManager {
 public:
  static constexpr int kMaxEvents = 8;
  using PipePair = std::pair<PipeType, PipeType>;
  std::map<PipePair, std::vector<bool>> busy_per_pipe_;

  EventIdManager() = default;

  int Allocate(PipeType src_pipe, PipeType dst_pipe) {
    const PipePair pair = {src_pipe, dst_pipe};
    auto& busy = busy_per_pipe_[pair];
    if (busy.empty()) {
      busy.resize(kMaxEvents, false);
    }

    for (int i = 0; i < kMaxEvents; ++i) {
      if (!busy[i]) {
        busy[i] = true;
        return i;
      }
    }

    std::stringstream ss;
    ss << "Out of hardware event IDs (max 8) for pipe pair " << static_cast<int>(src_pipe) << "->"
       << static_cast<int>(dst_pipe) << ". Deadlock or resource exhaustion.";
    throw ValueError(ss.str());
  }

  void Release(PipeType src_pipe, PipeType dst_pipe, int id) {
    if (id < 0 || id >= kMaxEvents) return;
    const PipePair pair = {src_pipe, dst_pipe};
    auto it = busy_per_pipe_.find(pair);
    if (it != busy_per_pipe_.end()) {
      it->second[id] = false;
    }
  }
};

/**
 * @brief Mutator that inserts sync operations into SeqStmts
 */
class SyncInserter : public IRMutator {
 public:
  SyncInserter() = default;

  FunctionPtr Run(const FunctionPtr& func) {
    auto new_body = VisitStmt(func->body_);
    return std::make_shared<Function>(func->name_, func->params_, func->return_types_, new_body, func->span_,
                                      func->func_type_);
  }

 private:
  /** @brief Get pipe type for a call: from IR op if set, else from backend (backend is required). */
  PipeType GetPipeForCall(const Call* call) {
    if (call->op_->GetPipe().has_value()) {
      return *call->op_->GetPipe();
    }
    const pypto::backend::Backend* backend = pypto::backend::GetBackend();
    const auto* info = backend->GetOpInfo(call->op_->name_);
    if (info) return info->pipe;
    return PipeType::S;
  }

  /** @brief Extract pipe type from a statement. */
  PipeType GetStmtPipe(const StmtPtr& stmt) {
    if (auto assign = As<AssignStmt>(stmt)) {
      if (auto call = As<Call>(assign->value_)) {
        return GetPipeForCall(call.get());
      }
    } else if (auto eval = As<EvalStmt>(stmt)) {
      if (auto call = As<Call>(eval->expr_)) {
        return GetPipeForCall(call.get());
      }
    }
    return PipeType::S;
  }

  /** @brief Extract pipe types from operations within a statement that access given memrefs. */
  std::set<PipeType> ExtractPipesForMemRefs(const StmtPtr& stmt, const std::set<MemRefPtr>& target_memrefs,
                                            bool for_reads) {
    std::set<PipeType> pipes;
    if (!stmt || target_memrefs.empty()) return pipes;

    auto has_target = [&](const std::set<MemRefPtr>& memrefs) {
      for (const auto& m : memrefs) {
        for (const auto& t : target_memrefs) {
          if (IsSameMem(m, t)) return true;
        }
      }
      return false;
    };

    if (auto assign = As<AssignStmt>(stmt)) {
      PipeType pipe = GetStmtPipe(stmt);
      if (for_reads && has_target(GetExprMemRefs(assign->value_))) {
        pipes.insert(pipe);
      }
      if (!for_reads && has_target(GetExprMemRefs(assign->var_))) {
        pipes.insert(pipe);
      }
    } else if (auto eval = As<EvalStmt>(stmt)) {
      if (for_reads && has_target(GetExprMemRefs(eval->expr_))) {
        pipes.insert(GetStmtPipe(stmt));
      }
    } else if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& s : seq->stmts_) {
        auto sub_pipes = ExtractPipesForMemRefs(s, target_memrefs, for_reads);
        pipes.insert(sub_pipes.begin(), sub_pipes.end());
      }
    } else if (auto op_stmts = As<OpStmts>(stmt)) {
      for (const auto& s : op_stmts->stmts_) {
        auto sub_pipes = ExtractPipesForMemRefs(s, target_memrefs, for_reads);
        pipes.insert(sub_pipes.begin(), sub_pipes.end());
      }
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      auto then_pipes = ExtractPipesForMemRefs(if_stmt->then_body_, target_memrefs, for_reads);
      pipes.insert(then_pipes.begin(), then_pipes.end());
      if (if_stmt->else_body_) {
        auto else_pipes = ExtractPipesForMemRefs(*if_stmt->else_body_, target_memrefs, for_reads);
        pipes.insert(else_pipes.begin(), else_pipes.end());
      }
    } else if (auto for_stmt = As<ForStmt>(stmt)) {
      auto body_pipes = ExtractPipesForMemRefs(for_stmt->body_, target_memrefs, for_reads);
      pipes.insert(body_pipes.begin(), body_pipes.end());
    } else if (auto while_stmt = As<WhileStmt>(stmt)) {
      auto body_pipes = ExtractPipesForMemRefs(while_stmt->body_, target_memrefs, for_reads);
      pipes.insert(body_pipes.begin(), body_pipes.end());
    }
    return pipes;
  }

  /**
   * @brief Helper struct to summarize memrefs in a statement
   */
  struct MemRefSummary {
    std::set<MemRefPtr> reads;
    std::set<MemRefPtr> writes;
  };

  /**
   * @brief Extract read and write memrefs from a statement
   * Recursively traverses the statement tree to collect all memrefs
   */
  MemRefSummary ExtractMemRefs(const StmtPtr& stmt) {
    MemRefSummary summary;

    if (!stmt) return summary;

    if (auto assign = As<AssignStmt>(stmt)) {
      // AssignStmt: var_ is written, value_ is read
      summary.writes = GetExprMemRefs(assign->var_);
      summary.reads = GetExprMemRefs(assign->value_);
    } else if (auto eval = As<EvalStmt>(stmt)) {
      // EvalStmt: expr_ is read
      summary.reads = GetExprMemRefs(eval->expr_);
    } else if (auto seq = As<SeqStmts>(stmt)) {
      // SeqStmts: merge all reads and writes from sub-statements
      for (const auto& s : seq->stmts_) {
        auto sub_summary = ExtractMemRefs(s);
        summary.reads.insert(sub_summary.reads.begin(), sub_summary.reads.end());
        summary.writes.insert(sub_summary.writes.begin(), sub_summary.writes.end());
      }
    } else if (auto op_stmts = As<OpStmts>(stmt)) {
      // OpStmts: merge all reads and writes from sub-statements (like SeqStmts)
      for (const auto& s : op_stmts->stmts_) {
        auto sub_summary = ExtractMemRefs(s);
        summary.reads.insert(sub_summary.reads.begin(), sub_summary.reads.end());
        summary.writes.insert(sub_summary.writes.begin(), sub_summary.writes.end());
      }
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      // IfStmt: merge reads/writes from both branches (union of all memrefs).
      // Note: condition_ is a scalar boolean expression and does not involve
      // Tile/Tensor memrefs requiring pipe synchronization.
      auto then_summary = ExtractMemRefs(if_stmt->then_body_);
      summary.reads.insert(then_summary.reads.begin(), then_summary.reads.end());
      summary.writes.insert(then_summary.writes.begin(), then_summary.writes.end());

      if (if_stmt->else_body_) {
        auto else_summary = ExtractMemRefs(*if_stmt->else_body_);
        summary.reads.insert(else_summary.reads.begin(), else_summary.reads.end());
        summary.writes.insert(else_summary.writes.begin(), else_summary.writes.end());
      }
    } else if (auto for_stmt = As<ForStmt>(stmt)) {
      // ForStmt: extract reads/writes from body only.
      // Note: start_, stop_, step_ are scalar integer expressions and do not involve
      // Tile/Tensor memrefs requiring pipe synchronization. iter_args_ initValues are
      // SSA constructs whose memrefs are already covered by body analysis.
      auto body_summary = ExtractMemRefs(for_stmt->body_);
      summary.reads.insert(body_summary.reads.begin(), body_summary.reads.end());
      summary.writes.insert(body_summary.writes.begin(), body_summary.writes.end());
    } else if (auto while_stmt = As<WhileStmt>(stmt)) {
      // WhileStmt: extract reads/writes from condition and body.
      // condition_ may reference memrefs, and body_ contains the loop operations.
      auto cond_memrefs = GetExprMemRefs(while_stmt->condition_);
      summary.reads.insert(cond_memrefs.begin(), cond_memrefs.end());
      auto body_summary = ExtractMemRefs(while_stmt->body_);
      summary.reads.insert(body_summary.reads.begin(), body_summary.reads.end());
      summary.writes.insert(body_summary.writes.begin(), body_summary.writes.end());
    } else if (auto yield_stmt = As<YieldStmt>(stmt)) {
      // YieldStmt: value_ expressions are reads
      for (const auto& expr : yield_stmt->value_) {
        auto memrefs = GetExprMemRefs(expr);
        summary.reads.insert(memrefs.begin(), memrefs.end());
      }
    } else if (auto return_stmt = As<ReturnStmt>(stmt)) {
      // ReturnStmt: value_ expressions are reads
      for (const auto& expr : return_stmt->value_) {
        auto memrefs = GetExprMemRefs(expr);
        summary.reads.insert(memrefs.begin(), memrefs.end());
      }
    }
    // For other statement types, no memrefs

    return summary;
  }

  /**
   * @brief Analyze dependencies between statements
   */
  std::vector<std::shared_ptr<DepEdge>> AnalyzeDependencies(const std::vector<StmtPtr>& stmts) {
    std::vector<std::shared_ptr<DepEdge>> deps;
    std::map<MemRefPtr, int> last_writer;
    std::map<MemRefPtr, std::vector<int>> last_readers;

    auto add_dep = [&](int prod, int cons, const MemRefPtr& memref, bool consumer_reads) {
      if (prod < 0) return;

      // Extract actual pipe types for IfStmt/ForStmt/WhileStmt
      auto get_pipes_for_stmt = [&](int idx, bool for_reads) -> std::set<PipeType> {
        if (auto if_stmt = As<IfStmt>(stmts[idx])) {
          return ExtractPipesForMemRefs(if_stmt, {memref}, for_reads);
        } else if (auto for_stmt = As<ForStmt>(stmts[idx])) {
          return ExtractPipesForMemRefs(for_stmt, {memref}, for_reads);
        } else if (auto while_stmt = As<WhileStmt>(stmts[idx])) {
          return ExtractPipesForMemRefs(while_stmt, {memref}, for_reads);
        } else {
          return {GetStmtPipe(stmts[idx])};
        }
      };

      // For RAW: producer writes, consumer reads
      // For WAW: producer writes, consumer writes
      // For WAR: producer reads, consumer writes
      auto producer_pipes =
          get_pipes_for_stmt(prod, !consumer_reads);  // Producer: writes if consumer reads, reads otherwise
      auto consumer_pipes = get_pipes_for_stmt(cons, consumer_reads);

      // Create edges for all pipe combinations
      for (auto p_pipe : producer_pipes) {
        for (auto c_pipe : consumer_pipes) {
          // Skip S pipe edges
          if (p_pipe == PipeType::S || c_pipe == PipeType::S) continue;

          deps.push_back(std::make_shared<DepEdge>(DepEdge{prod, cons, p_pipe, c_pipe}));
        }
      }
    };

    for (int i = 0; i < static_cast<int>(stmts.size()); ++i) {
      const auto& stmt = stmts[i];
      std::set<MemRefPtr> reads;
      std::set<MemRefPtr> writes;

      if (auto assign = As<AssignStmt>(stmt)) {
        writes = GetExprMemRefs(assign->var_);
        reads = GetExprMemRefs(assign->value_);
      } else if (auto eval = As<EvalStmt>(stmt)) {
        reads = GetExprMemRefs(eval->expr_);
      } else if (auto if_stmt = As<IfStmt>(stmt)) {
        // IfStmt: extract ALL reads and writes from body (union of branches)
        auto memref_summary = ExtractMemRefs(if_stmt);
        reads = memref_summary.reads;
        writes = memref_summary.writes;
      } else if (auto for_stmt = As<ForStmt>(stmt)) {
        // ForStmt: treated as opaque statement with body's reads/writes
        auto memref_summary = ExtractMemRefs(for_stmt);
        reads = memref_summary.reads;
        writes = memref_summary.writes;
      } else if (auto while_stmt = As<WhileStmt>(stmt)) {
        // WhileStmt: treated as opaque statement with condition and body's reads/writes
        auto memref_summary = ExtractMemRefs(while_stmt);
        reads = memref_summary.reads;
        writes = memref_summary.writes;
      }

      // Check RAW
      for (const auto& r : reads) {
        for (auto const& [m, idx] : last_writer) {
          if (IsSameMem(r, m)) {
            add_dep(idx, i, r, true);  // Consumer reads
          }
        }
      }

      // Check WAW and WAR
      for (const auto& w : writes) {
        // WAW
        for (auto const& [m, idx] : last_writer) {
          if (IsSameMem(w, m)) {
            add_dep(idx, i, w, false);  // Consumer writes
          }
        }
        // WAR
        for (auto const& [m, indices] : last_readers) {
          if (IsSameMem(w, m)) {
            for (int r_idx : indices) {
              add_dep(r_idx, i, w, false);  // Consumer writes
            }
          }
        }
      }

      // Update last write/read
      for (const auto& w : writes) {
        last_writer[w] = i;
        last_readers[w].clear();  // Reset readers on write
      }
      for (const auto& r : reads) {
        last_readers[r].push_back(i);
      }
    }
    return deps;
  }

  /**
   * @brief Simulate execution to assign hardware event IDs
   */
  void AssignEventIds(size_t stmt_count, const std::vector<std::shared_ptr<DepEdge>>& deps) {
    EventIdManager event_manager;
    // Organize edges by producer and consumer for sequential processing
    std::map<int, std::vector<std::shared_ptr<DepEdge>>> prod_edges;  // Outgoing (Set)
    std::map<int, std::vector<std::shared_ptr<DepEdge>>> cons_edges;  // Incoming (Wait)

    for (const auto& edge : deps) {
      if (edge->producer_pipe != edge->consumer_pipe) {
        prod_edges[edge->producer_idx].push_back(edge);
        cons_edges[edge->consumer_idx].push_back(edge);
      }
    }

    // Track allocated event IDs for each (producer_idx, src_pipe, dst_pipe) to enable sharing
    using PipePairKey = std::tuple<int, PipeType, PipeType>;
    std::map<PipePairKey, int> allocated_event_ids;

    // Simulate execution to assign IDs
    for (int i = 0; i < static_cast<int>(stmt_count); ++i) {
      // Process Waits (release IDs) BEFORE instruction execution
      // NOTE: Wait instruction is inserted BEFORE consumer instruction.
      if (cons_edges.count(i)) {
        for (const auto& edge : cons_edges[i]) {
          // Release ID
          if (edge->event_id != -1) {
            event_manager.Release(edge->producer_pipe, edge->consumer_pipe, edge->event_id);
          }
        }
      }

      // Process Sets (allocate IDs) AFTER instruction execution
      // NOTE: Set instruction is inserted AFTER producer instruction.
      if (prod_edges.count(i)) {
        for (const auto& edge : prod_edges[i]) {
          // Check if we already allocated an event ID for this pipe pair
          PipePairKey key = std::make_tuple(edge->producer_idx, edge->producer_pipe, edge->consumer_pipe);
          auto it = allocated_event_ids.find(key);
          if (it != allocated_event_ids.end()) {
            // Reuse existing event ID
            edge->event_id = it->second;
          } else {
            // Allocate new ID and store it
            int new_id = event_manager.Allocate(edge->producer_pipe, edge->consumer_pipe);
            edge->event_id = new_id;
            allocated_event_ids[key] = new_id;
          }
        }
      }
    }
  }

  /**
   * @brief Generate synchronization instructions
   */
  void GenerateInsertions(const std::vector<std::shared_ptr<DepEdge>>& deps,
                          std::map<int, std::vector<StmtPtr>>& insert_before,
                          std::map<int, std::vector<StmtPtr>>& insert_after) {
    std::set<std::pair<int, PipeType>> inserted_bars;  // Track inserted barriers (position, pipe)
    // Track inserted sync instructions by (src_pipe, dst_pipe, event_id) only
    // This ensures each event ID is only set/waited once
    std::set<std::tuple<PipeType, PipeType, int>> inserted_sync_src;
    std::set<std::tuple<PipeType, PipeType, int>> inserted_sync_dst;

    for (const auto& edge : deps) {
      if (edge->producer_pipe != edge->consumer_pipe) {
        // Cross-pipe
        // Skip syncs involving S pipe (scalar pipe doesn't need sync)
        if (edge->producer_pipe == PipeType::S || edge->consumer_pipe == PipeType::S) {
          continue;
        }
        if (edge->event_id == -1) continue;  // Should have been assigned

        // Insert sync_src only if not already inserted for this (pipe_pair, event_id)
        auto src_key = std::make_tuple(edge->producer_pipe, edge->consumer_pipe, edge->event_id);
        if (!inserted_sync_src.count(src_key)) {
          insert_after[edge->producer_idx].push_back(
              CreateSyncCall("system.sync_src", edge->producer_pipe, edge->consumer_pipe, edge->event_id));
          inserted_sync_src.insert(src_key);
        }

        // Insert sync_dst only if not already inserted for this (pipe_pair, event_id)
        auto dst_key = std::make_tuple(edge->producer_pipe, edge->consumer_pipe, edge->event_id);
        if (!inserted_sync_dst.count(dst_key)) {
          insert_before[edge->consumer_idx].push_back(
              CreateSyncCall("system.sync_dst", edge->producer_pipe, edge->consumer_pipe, edge->event_id));
          inserted_sync_dst.insert(dst_key);
        }
      } else {
        // Same pipe - only insert one barrier per position and pipe type
        auto bar_key = std::make_pair(edge->consumer_idx, edge->producer_pipe);
        if (inserted_bars.count(bar_key)) {
          continue;  // Already inserted a barrier at this position for this pipe
        }
        inserted_bars.insert(bar_key);

        if (edge->producer_pipe == PipeType::V) {
          insert_before[edge->consumer_idx].push_back(CreateBarCall("system.bar_v"));
        } else if (edge->producer_pipe == PipeType::M) {
          insert_before[edge->consumer_idx].push_back(CreateBarCall("system.bar_m"));
        }
      }
    }
  }

  /**
   * @brief Analyze cross-iteration dependencies within a loop body
   * Detects RAW dependencies between consecutive iterations:
   * - RAW: iteration i writes M, iteration i+1 reads M
   * Only considers memrefs where READ occurs BEFORE WRITE within the loop body,
   * because in that case, the read in iteration i+1 depends on the write in iteration i.
   * If WRITE occurs before READ, the read gets the value from the same iteration.
   */
  std::vector<std::shared_ptr<DepEdge>> AnalyzeCrossIterationDeps(const StmtPtr& body) {
    std::vector<std::shared_ptr<DepEdge>> deps;
    MemRefSummary summary = ExtractMemRefs(body);

    // Find memrefs that are both read and written
    std::set<MemRefPtr> read_write_memrefs;
    for (const auto& w : summary.writes) {
      for (const auto& r : summary.reads) {
        if (IsSameMem(w, r)) {
          read_write_memrefs.insert(w);
        }
      }
    }

    if (read_write_memrefs.empty()) {
      return deps;
    }

    // For each memref, check if first read comes before first write
    // If so, there's a cross-iteration dependency
    for (const auto& memref : read_write_memrefs) {
      int first_read_idx = FindFirstAccessIndex(body, memref, true);
      int first_write_idx = FindFirstAccessIndex(body, memref, false);

      // Cross-iteration dependency exists if read comes before write
      if (first_read_idx >= 0 && first_write_idx >= 0 && first_read_idx < first_write_idx) {
        // Get pipes that write this memref (producer in iter i)
        auto write_pipes = ExtractPipesForMemRefs(body, {memref}, false);
        // Get pipes that read this memref (consumer in iter i+1)
        auto read_pipes = ExtractPipesForMemRefs(body, {memref}, true);

        for (auto p_pipe : write_pipes) {
          for (auto c_pipe : read_pipes) {
            if (p_pipe == PipeType::S || c_pipe == PipeType::S) continue;
            deps.push_back(std::make_shared<DepEdge>(DepEdge{0, 1, p_pipe, c_pipe}));
          }
        }
      }
    }

    return deps;
  }

  /**
   * @brief Find the index of first read or write access to a memref in a statement
   * Returns -1 if not found
   */
  int FindFirstAccessIndex(const StmtPtr& stmt, const MemRefPtr& target, bool for_reads) {
    if (!stmt) return -1;

    if (auto seq = As<SeqStmts>(stmt)) {
      for (int i = 0; i < static_cast<int>(seq->stmts_.size()); ++i) {
        if (HasMemRefAccess(seq->stmts_[i], target, for_reads)) {
          return i;
        }
      }
      return -1;
    }

    if (auto op_stmts = As<OpStmts>(stmt)) {
      for (int i = 0; i < static_cast<int>(op_stmts->stmts_.size()); ++i) {
        if (HasMemRefAccess(op_stmts->stmts_[i], target, for_reads)) {
          return i;
        }
      }
      return -1;
    }

    // For non-SeqStmts/OpStmts, check if it accesses the memref
    return HasMemRefAccess(stmt, target, for_reads) ? 0 : -1;
  }

  /**
   * @brief Check if a statement accesses a specific memref (read or write)
   */
  bool HasMemRefAccess(const StmtPtr& stmt, const MemRefPtr& target, bool for_reads) {
    if (!stmt) return false;

    if (auto assign = As<AssignStmt>(stmt)) {
      if (for_reads) {
        auto memrefs = GetExprMemRefs(assign->value_);
        for (const auto& m : memrefs) {
          if (IsSameMem(m, target)) return true;
        }
      } else {
        auto memrefs = GetExprMemRefs(assign->var_);
        for (const auto& m : memrefs) {
          if (IsSameMem(m, target)) return true;
        }
      }
    } else if (auto eval = As<EvalStmt>(stmt)) {
      if (for_reads) {
        auto memrefs = GetExprMemRefs(eval->expr_);
        for (const auto& m : memrefs) {
          if (IsSameMem(m, target)) return true;
        }
      }
    } else if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& s : seq->stmts_) {
        if (HasMemRefAccess(s, target, for_reads)) return true;
      }
    } else if (auto op_stmts = As<OpStmts>(stmt)) {
      for (const auto& s : op_stmts->stmts_) {
        if (HasMemRefAccess(s, target, for_reads)) return true;
      }
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      // Note: condition_ is a scalar boolean expression and does not involve
      // Tile/Tensor memrefs requiring pipe synchronization.
      if (HasMemRefAccess(if_stmt->then_body_, target, for_reads)) return true;
      if (if_stmt->else_body_ && HasMemRefAccess(*if_stmt->else_body_, target, for_reads)) return true;
    } else if (auto for_stmt = As<ForStmt>(stmt)) {
      // Note: start_, stop_, step_ are scalar integer expressions and do not involve
      // Tile/Tensor memrefs requiring pipe synchronization. iter_args_ initValues are
      // SSA constructs whose memrefs are already covered by body analysis.
      if (HasMemRefAccess(for_stmt->body_, target, for_reads)) return true;
    } else if (auto while_stmt = As<WhileStmt>(stmt)) {
      // Note: condition_ may reference memrefs, and body_ contains the loop operations.
      // iter_args_ initValues are SSA constructs whose memrefs are covered by body analysis.
      if (for_reads) {
        auto cond_memrefs = GetExprMemRefs(while_stmt->condition_);
        for (const auto& m : cond_memrefs) {
          if (IsSameMem(m, target)) return true;
        }
      }
      if (HasMemRefAccess(while_stmt->body_, target, for_reads)) return true;
    }

    return false;
  }

  /**
   * @brief Generate sync statements for cross-iteration dependencies
   */
  std::vector<StmtPtr> GenerateCrossIterSyncs(const std::vector<std::shared_ptr<DepEdge>>& deps) {
    std::vector<StmtPtr> sync_stmts;
    std::set<std::pair<PipeType, PipeType>> inserted_cross_pipe;
    std::set<PipeType> inserted_bars;

    // Allocate event IDs for cross-pipe dependencies
    EventIdManager event_manager;
    for (auto& edge : deps) {
      if (edge->producer_pipe != edge->consumer_pipe) {
        auto key = std::make_pair(edge->producer_pipe, edge->consumer_pipe);
        if (!inserted_cross_pipe.count(key)) {
          edge->event_id = event_manager.Allocate(edge->producer_pipe, edge->consumer_pipe);
          inserted_cross_pipe.insert(key);
        }
      }
    }

    // Reset for insertion phase
    inserted_cross_pipe.clear();

    for (const auto& edge : deps) {
      if (edge->producer_pipe != edge->consumer_pipe) {
        auto key = std::make_pair(edge->producer_pipe, edge->consumer_pipe);
        if (!inserted_cross_pipe.count(key)) {
          sync_stmts.push_back(
              CreateSyncCall("system.sync_src", edge->producer_pipe, edge->consumer_pipe, edge->event_id));
          sync_stmts.push_back(
              CreateSyncCall("system.sync_dst", edge->producer_pipe, edge->consumer_pipe, edge->event_id));
          inserted_cross_pipe.insert(key);
        }
      } else {
        if (!inserted_bars.count(edge->producer_pipe)) {
          if (edge->producer_pipe == PipeType::V) {
            sync_stmts.push_back(CreateBarCall("system.bar_v"));
          } else if (edge->producer_pipe == PipeType::M) {
            sync_stmts.push_back(CreateBarCall("system.bar_m"));
          }
          inserted_bars.insert(edge->producer_pipe);
        }
      }
    }
    return sync_stmts;
  }

  /**
   * @brief Append sync statements to the end of loop body (before YieldStmt if present)
   */
  StmtPtr AppendSyncsToBody(const StmtPtr& body, const std::vector<StmtPtr>& sync_stmts) {
    if (sync_stmts.empty()) return body;

    if (auto seq = As<SeqStmts>(body)) {
      std::vector<StmtPtr> new_stmts;
      bool yield_found = false;

      for (const auto& stmt : seq->stmts_) {
        if (As<YieldStmt>(stmt)) {
          // Insert syncs BEFORE yield
          for (const auto& sync : sync_stmts) new_stmts.push_back(sync);
          new_stmts.push_back(stmt);
          yield_found = true;
        } else {
          new_stmts.push_back(stmt);
        }
      }

      if (!yield_found) {
        for (const auto& sync : sync_stmts) new_stmts.push_back(sync);
      }
      return std::make_shared<const SeqStmts>(new_stmts, seq->span_);
    }

    // Single statement body
    std::vector<StmtPtr> new_stmts;
    if (As<YieldStmt>(body)) {
      for (const auto& sync : sync_stmts) new_stmts.push_back(sync);
      new_stmts.push_back(body);
    } else {
      new_stmts.push_back(body);
      for (const auto& sync : sync_stmts) new_stmts.push_back(sync);
    }
    return std::make_shared<const SeqStmts>(new_stmts, body->span_);
  }

 public:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    // Visit each child and flatten any OpStmts into individual statements.
    // NormalizeStmtStructure (called by FlattenCallExpr) wraps consecutive
    // AssignStmt/EvalStmt into OpStmts blocks, which must be flattened here
    // for dependency analysis to see individual statements.
    std::vector<StmtPtr> original_stmts;
    for (const auto& s : op->stmts_) {
      auto visited = VisitStmt(s);
      if (auto op_stmts = As<OpStmts>(visited)) {
        // Flatten OpStmts into individual statements
        for (const auto& inner : op_stmts->stmts_) {
          original_stmts.push_back(inner);
        }
      } else {
        original_stmts.push_back(visited);
      }
    }

    // 1. Analyze dependencies in this sequence
    auto deps = AnalyzeDependencies(original_stmts);

    // 2. Assign Event IDs (Simulation)
    AssignEventIds(original_stmts.size(), deps);

    // 3. Generate Insertions
    std::map<int, std::vector<StmtPtr>> insert_before;
    std::map<int, std::vector<StmtPtr>> insert_after;
    GenerateInsertions(deps, insert_before, insert_after);

    // 4. Build new statement list
    std::vector<StmtPtr> final_stmts;
    for (int i = 0; i < static_cast<int>(original_stmts.size()); ++i) {
      if (insert_before.count(i)) {
        for (const auto& s : insert_before[i]) final_stmts.push_back(s);
      }
      final_stmts.push_back(original_stmts[i]);
      if (insert_after.count(i)) {
        for (const auto& s : insert_after[i]) final_stmts.push_back(s);
      }
    }

    return std::make_shared<const SeqStmts>(final_stmts, op->span_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    // Recursively process then_body and else_body
    // Internal dependencies are handled automatically through recursive calls
    auto new_then_body = VisitStmt(op->then_body_);
    std::optional<StmtPtr> new_else_body =
        op->else_body_ ? std::make_optional(VisitStmt(*op->else_body_)) : std::nullopt;

    return std::make_shared<const IfStmt>(op->condition_, new_then_body, new_else_body,
                                          op->return_vars_,  // Keep return_vars unchanged
                                          op->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Recursively process body
    // Internal dependencies are handled automatically through recursive calls
    auto new_body = VisitStmt(op->body_);

    // Analyze cross-iteration dependencies (RAW, WAW, WAR)
    auto cross_iter_deps = AnalyzeCrossIterationDeps(new_body);

    // Generate and insert cross-iteration sync instructions at body end
    auto sync_stmts = GenerateCrossIterSyncs(cross_iter_deps);
    new_body = AppendSyncsToBody(new_body, sync_stmts);

    return std::make_shared<const ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, op->iter_args_,
                                           new_body, op->return_vars_, op->span_, op->kind_);
  }
};

}  // namespace

// Factory function
namespace pass {
/**
 * @brief Create an InsertSync pass
 *
 * This pass analyzes data dependencies between operations based on MemRef
 * and inserts synchronization operations (sync_src, sync_dst, bar_v, bar_m)
 * to ensure correct execution order across different hardware pipes.
 */
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

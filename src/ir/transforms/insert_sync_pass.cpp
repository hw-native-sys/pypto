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
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/passes.h"

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
 * @brief Extract pipe type from a statement
 */
PipeType GetStmtPipe(const StmtPtr& stmt) {
  if (auto assign = As<AssignStmt>(stmt)) {
    if (auto call = As<Call>(assign->value_)) {
      return call->op_->GetPipe().value_or(PipeType::S);
    }
  } else if (auto eval = As<EvalStmt>(stmt)) {
    if (auto call = As<Call>(eval->expr_)) {
      return call->op_->GetPipe().value_or(PipeType::S);
    }
  }
  return PipeType::S;
}

/**
 * @brief Extract pipe types from operations within a statement that access given memrefs
 */
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
      PipeType pipe = GetStmtPipe(stmt);
      pipes.insert(pipe);
    }
  } else if (auto seq = As<SeqStmts>(stmt)) {
    for (const auto& s : seq->stmts_) {
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
  }

  return pipes;
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
  FunctionPtr Run(const FunctionPtr& func) {
    auto new_body = VisitStmt(func->body_);
    return std::make_shared<Function>(func->name_, func->params_, func->return_types_, new_body, func->span_);
  }

 private:
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
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      // IfStmt: merge then and else branches
      auto then_summary = ExtractMemRefs(if_stmt->then_body_);
      summary.reads.insert(then_summary.reads.begin(), then_summary.reads.end());
      summary.writes.insert(then_summary.writes.begin(), then_summary.writes.end());

      if (if_stmt->else_body_) {
        auto else_summary = ExtractMemRefs(*if_stmt->else_body_);
        summary.reads.insert(else_summary.reads.begin(), else_summary.reads.end());
        summary.writes.insert(else_summary.writes.begin(), else_summary.writes.end());
      }
    }
    // For other statement types (YieldStmt, ReturnStmt, etc.), no memrefs

    return summary;
  }

  /**
   * @brief Extract memrefs from a list of variables
   */
  std::set<MemRefPtr> ExtractMemRefsFromVars(const std::vector<VarPtr>& vars) {
    std::set<MemRefPtr> memrefs;
    for (const auto& var : vars) {
      if (auto shaped_type = std::dynamic_pointer_cast<const ShapedType>(var->GetType())) {
        if (shaped_type->memref_.has_value()) {
          memrefs.insert(*shaped_type->memref_);
        }
      }
    }
    return memrefs;
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

      // Extract actual pipe types for IfStmt
      auto get_pipes_for_stmt = [&](int idx, bool for_reads) -> std::set<PipeType> {
        if (auto if_stmt = As<IfStmt>(stmts[idx])) {
          return ExtractPipesForMemRefs(if_stmt, {memref}, for_reads);
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
        // Extract reads from then/else branches for dependencies before IfStmt
        auto memref_summary = ExtractMemRefs(if_stmt);
        reads = memref_summary.reads;

        // Extract writes from return_vars only (IfStmt's external output)
        // Only return_vars' memrefs are visible to subsequent statements
        writes = ExtractMemRefsFromVars(if_stmt->return_vars_);
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

    auto create_sync_call = [](const std::string& op_name, PipeType p, PipeType tp, int event_id) {
      auto& registry = OpRegistry::GetInstance();
      std::vector<std::pair<std::string, std::any>> kwargs;
      kwargs.emplace_back("set_pipe", static_cast<int>(p));
      kwargs.emplace_back("wait_pipe", static_cast<int>(tp));
      kwargs.emplace_back("event_id", event_id);
      auto call = registry.Create(op_name, {}, kwargs, Span::unknown());
      return std::make_shared<const EvalStmt>(call, Span::unknown());
    };

    auto create_bar_call = [](const std::string& op_name) {
      auto& registry = OpRegistry::GetInstance();
      auto call = registry.Create(op_name, {}, {}, Span::unknown());
      return std::make_shared<const EvalStmt>(call, Span::unknown());
    };

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
              create_sync_call("system.sync_src", edge->producer_pipe, edge->consumer_pipe, edge->event_id));
          inserted_sync_src.insert(src_key);
        }

        // Insert sync_dst only if not already inserted for this (pipe_pair, event_id)
        auto dst_key = std::make_tuple(edge->producer_pipe, edge->consumer_pipe, edge->event_id);
        if (!inserted_sync_dst.count(dst_key)) {
          insert_before[edge->consumer_idx].push_back(
              create_sync_call("system.sync_dst", edge->producer_pipe, edge->consumer_pipe, edge->event_id));
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
          insert_before[edge->consumer_idx].push_back(create_bar_call("system.bar_v"));
        } else if (edge->producer_pipe == PipeType::M) {
          insert_before[edge->consumer_idx].push_back(create_bar_call("system.bar_m"));
        }
      }
    }
  }

 public:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> original_stmts;
    for (const auto& s : op->stmts_) {
      original_stmts.push_back(VisitStmt(s));
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
    auto new_else_body = op->else_body_ ? VisitStmt(*op->else_body_) : nullptr;

    return std::make_shared<const IfStmt>(op->condition_, new_then_body, new_else_body,
                                          op->return_vars_,  // Keep return_vars unchanged
                                          op->span_);
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
      "InsertSync");
}
}  // namespace pass
}  // namespace ir
}  // namespace pypto

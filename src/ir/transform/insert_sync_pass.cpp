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

#include "pypto/ir/transform/insert_sync_pass.h"

#include <algorithm>
#include <any>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transform/base/mutator.h"
#include "pypto/ir/transform/base/visitor.h"
#include "pypto/ir/transform/transformers.h"

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
    if (auto shaped_type = std::dynamic_pointer_cast<const ShapedType>(var->GetType())) {
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

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> original_stmts;
    for (const auto& s : op->stmts_) {
      original_stmts.push_back(VisitStmt(s));
    }

    // 1. Analyze dependencies in this sequence
    std::vector<std::shared_ptr<DepEdge>> deps;
    std::set<std::pair<int, int>> existing_deps;  // Keep track of existing edges to avoid duplicates
    std::map<MemRefPtr, int> last_writer;
    std::map<MemRefPtr, std::vector<int>> last_readers;

    auto get_memrefs = [](const ExprPtr& expr) {
      MemRefCollector collector;
      collector.VisitExpr(expr);
      return collector.memrefs;
    };

    auto add_dep = [&](int prod, int cons, const std::vector<StmtPtr>& stmts) {
      if (prod < 0) return;
      if (existing_deps.count({prod, cons})) return;  // Skip if edge already exists

      existing_deps.insert({prod, cons});
      deps.push_back(
          std::make_shared<DepEdge>(DepEdge{prod, cons, GetStmtPipe(stmts[prod]), GetStmtPipe(stmts[cons])}));
    };

    for (int i = 0; i < static_cast<int>(original_stmts.size()); ++i) {
      const auto& stmt = original_stmts[i];
      std::set<MemRefPtr> reads;
      std::set<MemRefPtr> writes;

      if (auto assign = As<AssignStmt>(stmt)) {
        writes = get_memrefs(assign->var_);
        reads = get_memrefs(assign->value_);
      } else if (auto eval = As<EvalStmt>(stmt)) {
        reads = get_memrefs(eval->expr_);
      }

      // Check RAW
      for (const auto& r : reads) {
        for (auto const& [m, idx] : last_writer) {
          if (IsSameMem(r, m)) {
            add_dep(idx, i, original_stmts);
          }
        }
      }

      // Check WAW and WAR
      for (const auto& w : writes) {
        // WAW
        for (auto const& [m, idx] : last_writer) {
          if (IsSameMem(w, m)) {
            add_dep(idx, i, original_stmts);
          }
        }
        // WAR
        for (auto const& [m, indices] : last_readers) {
          if (IsSameMem(w, m)) {
            for (int r_idx : indices) {
              add_dep(r_idx, i, original_stmts);
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

    // 2. Assign Event IDs (Simulation)
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

    // Simulate execution to assign IDs
    for (int i = 0; i < static_cast<int>(original_stmts.size()); ++i) {
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
          // Allocate ID
          edge->event_id = event_manager.Allocate(edge->producer_pipe, edge->consumer_pipe);
        }
      }
    }

    // 3. Generate Insertions
    std::map<int, std::vector<StmtPtr>> insert_before;
    std::map<int, std::vector<StmtPtr>> insert_after;

    auto create_sync_call = [](const std::string& op_name, PipeType p, PipeType tp, int event_id) {
      auto& registry = OpRegistry::GetInstance();
      std::vector<std::pair<std::string, std::any>> kwargs;
      kwargs.push_back({"set_pipe", static_cast<int>(p)});
      kwargs.push_back({"wait_pipe", static_cast<int>(tp)});
      kwargs.push_back({"event_id", event_id});
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
        if (edge->event_id == -1) continue;  // Should have been assigned
        insert_after[edge->producer_idx].push_back(
            create_sync_call("system.sync_src", edge->producer_pipe, edge->consumer_pipe, edge->event_id));
        insert_before[edge->consumer_idx].push_back(
            create_sync_call("system.sync_dst", edge->producer_pipe, edge->consumer_pipe, edge->event_id));
      } else {
        // Same pipe
        if (edge->producer_pipe == PipeType::V) {
          insert_before[edge->consumer_idx].push_back(create_bar_call("system.bar_v"));
        } else if (edge->producer_pipe == PipeType::M) {
          insert_before[edge->consumer_idx].push_back(create_bar_call("system.bar_m"));
        }
      }
    }

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
};

}  // namespace

FunctionPtr InsertSyncPass::Run(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "InsertSyncPass cannot run on null function";
  SyncInserter inserter;
  return inserter.Run(func);
}

}  // namespace ir
}  // namespace pypto

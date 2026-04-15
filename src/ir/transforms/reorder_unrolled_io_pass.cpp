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
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/stmt_dependency_analysis.h"

namespace pypto {
namespace ir {
namespace {

constexpr const char* kUnrollReplicatedAttr = "unroll_replicated";
constexpr const char* kTileLoadOpName = "tile.load";
constexpr const char* kTileStoreOpName = "tile.store";

/// IO category used for priority during the topological sort. Lower is emitted first.
enum class IOCategory : int { Load = 0, Compute = 1, Store = 2 };

bool IsCallTo(const ExprPtr& expr, const std::string& op_name) {
  auto call = std::dynamic_pointer_cast<const Call>(expr);
  return call && call->op_ && call->op_->name_ == op_name;
}

IOCategory CategorizeStmt(const StmtPtr& stmt) {
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    if (IsCallTo(assign->value_, kTileLoadOpName)) return IOCategory::Load;
    if (IsCallTo(assign->value_, kTileStoreOpName)) return IOCategory::Store;
  }
  if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    if (IsCallTo(eval->expr_, kTileStoreOpName)) return IOCategory::Store;
  }
  return IOCategory::Compute;
}

/**
 * @brief Mutator that reorders statements inside ``unroll_replicated`` SeqStmts:
 *        loads pulled to the top, stores pushed to the bottom, compute in the middle —
 *        all subject to the dependency graph.
 */
class ReorderUnrolledIOMutator : public IRMutator {
 public:
  explicit ReorderUnrolledIOMutator(ProgramPtr program) : program_(std::move(program)) {}

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Recurse first so any inner unroll-replicated regions are reordered too.
    auto visited = IRMutator::VisitStmt_(op);
    auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(visited);
    if (!for_stmt || !for_stmt->HasAttr(kUnrollReplicatedAttr)) {
      return visited;
    }
    auto seq = std::dynamic_pointer_cast<const SeqStmts>(for_stmt->body_);
    if (!seq || seq->stmts_.size() < 2) {
      return visited;  // single stmt — nothing to reorder
    }
    auto reordered = ReorderRegion(seq);
    if (reordered.get() == seq.get()) {
      return visited;
    }
    auto new_for = MutableCopy(for_stmt);
    new_for->body_ = reordered;
    return new_for;
  }

 private:
  /// Stable, priority-aware topological sort.
  StmtPtr ReorderRegion(const SeqStmtsPtr& seq) {
    // Discipline check ensures dataflow soundness — see RFC #1026 / PR #1029.
    stmt_dep::CheckInOutUseDiscipline(seq, program_);
    auto graph = stmt_dep::BuildStmtDependencyGraph(seq, program_);

    const auto& stmts = seq->stmts_;
    const size_t N = stmts.size();

    std::vector<IOCategory> cats(N);
    for (size_t i = 0; i < N; ++i) cats[i] = CategorizeStmt(stmts[i]);

    // Per-stmt count of unsatisfied predecessors. Only count predecessors that
    // are themselves nodes in this region (the graph may include external preds
    // for region-leaving SSA edges).
    std::unordered_map<const Stmt*, size_t> stmt_index;
    stmt_index.reserve(N);
    for (size_t i = 0; i < N; ++i) stmt_index[stmts[i].get()] = i;

    std::vector<size_t> remaining(N, 0);
    for (size_t i = 0; i < N; ++i) {
      auto it = graph.predecessors.find(stmts[i].get());
      if (it == graph.predecessors.end()) continue;
      for (const auto* pred : it->second) {
        if (stmt_index.count(pred)) ++remaining[i];
      }
    }

    std::vector<bool> emitted(N, false);
    std::vector<StmtPtr> out;
    out.reserve(N);

    auto pick_next = [&]() -> std::optional<size_t> {
      // Prefer the smallest (cat, idx) among non-store ready stmts; only fall
      // back to a store when nothing else is ready, so stores defer to the end.
      std::optional<std::pair<int, size_t>> best_non_store;
      std::optional<size_t> best_store;
      for (size_t i = 0; i < N; ++i) {
        if (emitted[i] || remaining[i] != 0) continue;
        if (cats[i] == IOCategory::Store) {
          if (!best_store) best_store = i;
        } else {
          std::pair<int, size_t> key{static_cast<int>(cats[i]), i};
          if (!best_non_store || key < *best_non_store) best_non_store = key;
        }
      }
      if (best_non_store) return best_non_store->second;
      return best_store;
    };

    while (out.size() < N) {
      auto pick = pick_next();
      INTERNAL_CHECK(pick.has_value())
          << "ReorderUnrolledIO: dependency graph appears cyclic — should be impossible "
             "for an SSA region under the InOut-use discipline";
      size_t i = *pick;
      emitted[i] = true;
      out.push_back(stmts[i]);
      // Decrement successors' unsatisfied-pred counts.
      for (size_t j = 0; j < N; ++j) {
        if (emitted[j] || remaining[j] == 0) continue;
        auto it = graph.predecessors.find(stmts[j].get());
        if (it == graph.predecessors.end()) continue;
        if (it->second.count(stmts[i].get())) {
          --remaining[j];
        }
      }
    }

    // No-op detection.
    bool changed = false;
    for (size_t i = 0; i < N; ++i) {
      if (out[i].get() != stmts[i].get()) {
        changed = true;
        break;
      }
    }
    if (!changed) return seq;
    return std::make_shared<SeqStmts>(std::move(out), seq->span_);
  }

  ProgramPtr program_;
};

}  // namespace

namespace pass {

Pass ReorderUnrolledIO() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    INTERNAL_CHECK(program) << "ReorderUnrolledIO cannot run on null program";

    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    bool any_change = false;
    for (const auto& [gvar, func] : program->functions_) {
      ReorderUnrolledIOMutator mutator(program);
      auto new_body = mutator.VisitStmt(func->body_);
      if (new_body.get() == func->body_.get()) {
        new_functions.emplace(gvar, func);
      } else {
        auto new_func = MutableCopy(func);
        new_func->body_ = new_body;
        new_functions.emplace(gvar, new_func);
        any_change = true;
      }
    }
    if (!any_change) return program;

    auto new_program = MutableCopy(program);
    new_program->functions_ = std::move(new_functions);
    return new_program;
  };

  return CreateProgramPass(pass_func, "ReorderUnrolledIO", kReorderUnrolledIOProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

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
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/stmt_dependency_analysis.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

/// IO category used for priority during the topological sort. Lower is emitted first.
///
/// ``ScalarCompute`` sits above ``Load`` so that address-arithmetic assigns
/// (e.g. ``k = i * 512``) — the typical predecessors of a tile.load offset —
/// are emitted first, allowing all sibling clones' loads to become ready and
/// cluster at the top of the region. Without this split, a per-clone scalar
/// gate would interleave between clones and prevent the load-cluster layout
/// that ping-pong buffering depends on.
enum class IOCategory : int { ScalarCompute = 0, Load = 1, TileCompute = 2, Store = 3 };

/// Singletons for the ops the pass cares about — resolved once from the registry
/// and compared by identity in ``CategorizeStmt``. Using pointer identity instead
/// of name strings avoids string comparisons in the hot path and makes the set
/// of recognized ops explicit at pass construction.
struct IOCategoryOps {
  OpPtr tile_load;   ///< Read: tensor → tile data movement
  OpPtr tile_read;   ///< Read: extract scalar from a tile
  OpPtr tile_store;  ///< Write: tile → tensor data movement
  OpPtr tile_write;  ///< Write: put scalar into a tile

  static IOCategoryOps Build() {
    const auto& registry = OpRegistry::GetInstance();
    return {
        registry.GetOp("tile.load"),
        registry.GetOp("tile.read"),
        registry.GetOp("tile.store"),
        registry.GetOp("tile.write"),
    };
  }

  [[nodiscard]] bool IsLoadLike(const OpPtr& op) const { return op == tile_load || op == tile_read; }
  [[nodiscard]] bool IsStoreLike(const OpPtr& op) const { return op == tile_store || op == tile_write; }
};

OpPtr CalledOp(const ExprPtr& expr) {
  auto call = std::dynamic_pointer_cast<const Call>(expr);
  return call ? call->op_ : OpPtr{};
}

IOCategory CategorizeStmt(const StmtPtr& stmt, const IOCategoryOps& ops) {
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    auto op = CalledOp(assign->value_);
    if (op) {
      // tile.read keeps Load even though its LHS is scalar — it's I/O against
      // a tile and belongs in the load tier alongside tile.load.
      if (ops.IsLoadLike(op)) return IOCategory::Load;
      if (ops.IsStoreLike(op)) return IOCategory::Store;
    }
    // Scalar-producing compute lifts to the top so it unblocks downstream
    // loads; tile/tensor-producing compute stays in the middle.
    if (std::dynamic_pointer_cast<const ScalarType>(assign->var_->GetType())) {
      return IOCategory::ScalarCompute;
    }
    return IOCategory::TileCompute;
  }
  if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    auto op = CalledOp(eval->expr_);
    if (op && ops.IsStoreLike(op)) return IOCategory::Store;
  }
  return IOCategory::TileCompute;
}

/// Terminators (`YieldStmt`, `ReturnStmt`, `BreakStmt`, `ContinueStmt`) must
/// stay last in their scope: moving them ahead of a side-effecting `tile.store`
/// would make the store unreachable. Valid SSA always places a terminator at
/// the end of the enclosing `SeqStmts`.
bool IsTerminator(const StmtPtr& stmt) {
  return std::dynamic_pointer_cast<const YieldStmt>(stmt) ||
         std::dynamic_pointer_cast<const ReturnStmt>(stmt) ||
         std::dynamic_pointer_cast<const BreakStmt>(stmt) ||
         std::dynamic_pointer_cast<const ContinueStmt>(stmt);
}

/**
 * @brief Mutator that reorders every multi-stmt ``SeqStmts`` in the program.
 *
 * Layered priority (top → bottom): scalar compute, loads, tile compute, stores —
 * all subject to the dependency graph. Lifting scalar compute (typically address
 * arithmetic) above loads ensures sibling clones' loads become ready together
 * and cluster at the top, the layout ``MemoryReuse`` needs for ping-pong.
 */
class ReorderUnrolledIOMutator : public IRMutator {
 public:
  explicit ReorderUnrolledIOMutator(ProgramPtr program)
      : program_(std::move(program)), io_ops_(IOCategoryOps::Build()) {}

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    // Recurse first so any nested SeqStmts are reordered bottom-up.
    auto visited = IRMutator::VisitStmt_(op);
    auto seq = std::dynamic_pointer_cast<const SeqStmts>(visited);
    if (!seq || seq->stmts_.size() < 2) {
      return visited;  // single stmt — nothing to reorder
    }
    // Regions that violate the InOut-use discipline are left alone: under
    // strict verification they would be caught earlier, but with
    // VerificationLevel.NONE a pre-existing violation can reach us and we
    // shouldn't reorder potentially-unsound dataflow. We do the discipline
    // check ourselves so that the subsequent `BuildStmtDependencyGraph` call
    // (which would repeat the check when given a non-null program) can skip
    // it by passing `nullptr` — avoiding an O(N) double traversal.
    if (!stmt_dep::CollectInOutUseDisciplineDiagnostics(seq, program_).empty()) {
      return visited;
    }
    return ReorderRegion(seq);
  }

 private:
  /// Stable, priority-aware topological sort. Caller is responsible for
  /// verifying the InOut-use discipline before invoking.
  ///
  /// Complexity: O(N + E + N log N) per region — adjacency list built once,
  /// the ready set maintained as a min-heap keyed by (category, index).
  /// N = number of top-level stmts, E = number of def-use edges (bounded by N).
  ///
  /// A trailing terminator (`YieldStmt` / `ReturnStmt` / `BreakStmt` /
  /// `ContinueStmt`) is peeled off before sorting and re-appended at the end
  /// so stores can never be emitted after it (which would make them
  /// unreachable / semantically dropped).
  StmtPtr ReorderRegion(const SeqStmtsPtr& seq) {
    // Pass `nullptr` for the program — we've already validated the discipline,
    // and re-running it inside BuildStmtDependencyGraph would be redundant work.
    auto graph = stmt_dep::BuildStmtDependencyGraph(seq, /*program=*/nullptr);

    const auto& stmts = seq->stmts_;
    const size_t N = stmts.size();

    // Peel off a trailing terminator — it stays last regardless of category.
    const bool has_terminator = IsTerminator(stmts.back());
    const size_t sort_count = has_terminator ? N - 1 : N;
    if (sort_count < 2) return seq;  // nothing to reorder among non-terminators

    std::vector<IOCategory> cats(sort_count);
    std::unordered_map<const Stmt*, size_t> idx_of;
    idx_of.reserve(sort_count);
    for (size_t i = 0; i < sort_count; ++i) {
      cats[i] = CategorizeStmt(stmts[i], io_ops_);
      idx_of.emplace(stmts[i].get(), i);
    }

    // Build successors adjacency lists + in-degree counts in one pass over
    // the region's predecessor map. Predecessor entries for the terminator
    // (if any) are ignored so it cannot decrement any non-terminator's
    // remaining count and end up "ready" early.
    std::vector<std::vector<size_t>> successors(sort_count);
    std::vector<size_t> remaining(sort_count, 0);
    for (size_t j = 0; j < sort_count; ++j) {
      auto it = graph.predecessors.find(stmts[j].get());
      if (it == graph.predecessors.end()) continue;
      for (const Stmt* pred : it->second) {
        auto pit = idx_of.find(pred);
        if (pit == idx_of.end()) continue;  // predecessor is the terminator — ignore
        successors[pit->second].push_back(j);
        ++remaining[j];
      }
    }

    // Ready-set as a min-heap keyed by (category_bias, original_index). For
    // non-store stmts the key is (category, i); stores get a bias of +1 above
    // the max non-store category so they only surface when nothing else is
    // ready — preserving the original load-top / compute-middle / store-bottom
    // behavior. Using index as the tiebreaker keeps the sort stable.
    constexpr int kStoreDeferBias = static_cast<int>(IOCategory::Store) + 100;
    using HeapKey = std::pair<int, size_t>;
    std::priority_queue<HeapKey, std::vector<HeapKey>, std::greater<>> ready;
    auto key_for = [&](size_t i) -> HeapKey {
      int bias = (cats[i] == IOCategory::Store) ? kStoreDeferBias : static_cast<int>(cats[i]);
      return {bias, i};
    };
    for (size_t i = 0; i < sort_count; ++i) {
      if (remaining[i] == 0) ready.push(key_for(i));
    }

    std::vector<StmtPtr> out;
    out.reserve(N);
    while (!ready.empty()) {
      size_t i = ready.top().second;
      ready.pop();
      out.push_back(stmts[i]);
      for (size_t j : successors[i]) {
        if (--remaining[j] == 0) ready.push(key_for(j));
      }
    }
    INTERNAL_CHECK(out.size() == sort_count)
        << "ReorderUnrolledIO: dependency graph appears cyclic — should be impossible "
           "for an SSA region under the InOut-use discipline";
    if (has_terminator) out.push_back(stmts.back());

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
  IOCategoryOps io_ops_;
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

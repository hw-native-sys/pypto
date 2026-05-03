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

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
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
  OpPtr tile_load;     ///< Read: tensor → tile data movement
  OpPtr tile_read;     ///< Read: extract scalar from a tile
  OpPtr tile_store;    ///< Write: tile → tensor data movement
  OpPtr tile_write;    ///< Write: put scalar into a tile
  OpPtr tile_extract;  ///< Sub-tile extract — load-like only when L1→L0 (see IsL1ToL0ExtractCall)

  static IOCategoryOps Build() {
    const auto& registry = OpRegistry::GetInstance();
    return {
        registry.GetOp("tile.load"),  registry.GetOp("tile.read"),    registry.GetOp("tile.store"),
        registry.GetOp("tile.write"), registry.GetOp("tile.extract"),
    };
  }

  [[nodiscard]] bool IsLoadLike(const OpPtr& op) const { return op == tile_load || op == tile_read; }
  [[nodiscard]] bool IsStoreLike(const OpPtr& op) const { return op == tile_store || op == tile_write; }

  /// True when @p call is a `tile.extract` whose source lives in L1 (Mat) and
  /// whose destination lives in L0a/L0b (Left/Right) — i.e. the ISA TEXTRACT
  /// L1→L0 data-movement pattern emitted by AutoTileMatmulL0. Such extracts
  /// are load-like for scheduling purposes: clustering them ahead of the
  /// matmul/matmul_acc consumers in the iteration body lets the codegen
  /// ping-pong on Left/Right buffers (analogous to how tile.load clustering
  /// enables DDR→Mat ping-pong).
  ///
  /// Other tile.extract patterns — non-Mat source, non-{Left,Right} target,
  /// or unknown memory space — keep the default TileCompute tier so we don't
  /// disturb compute orderings the dependency graph already constrains.
  [[nodiscard]] bool IsL1ToL0ExtractCall(const Call& call) const {
    if (call.op_ != tile_extract) return false;
    if (call.args_.empty()) return false;
    auto src_tile = std::dynamic_pointer_cast<const TileType>(call.args_[0]->GetType());
    if (!src_tile) return false;
    auto src_ms = src_tile->GetMemorySpace();
    if (!src_ms.has_value() || *src_ms != MemorySpace::Mat) return false;
    for (const auto& [k, v] : call.kwargs_) {
      if (k != "target_memory") continue;
      auto target = AnyCast<MemorySpace>(v, "kwarg key: target_memory");
      return target == MemorySpace::Left || target == MemorySpace::Right;
    }
    return false;
  }
};

OpPtr CalledOp(const ExprPtr& expr) {
  auto call = std::dynamic_pointer_cast<const Call>(expr);
  return call ? call->op_ : OpPtr{};
}

IOCategory CategorizeStmt(const StmtPtr& stmt, const IOCategoryOps& ops) {
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    if (auto call = std::dynamic_pointer_cast<const Call>(assign->value_)) {
      // tile.read keeps Load even though its LHS is scalar — it's I/O against
      // a tile and belongs in the load tier alongside tile.load.
      if (ops.IsLoadLike(call->op_)) return IOCategory::Load;
      if (ops.IsStoreLike(call->op_)) return IOCategory::Store;
      // tile.extract is load-like only when it represents an L1→L0 transfer
      // (Mat source, Left/Right target). Other extract shapes stay in
      // TileCompute — see IsL1ToL0ExtractCall doc for rationale.
      if (ops.IsL1ToL0ExtractCall(*call)) return IOCategory::Load;
    }
    INTERNAL_CHECK_SPAN(assign->var_, assign->span_) << "Internal error: AssignStmt has null var_";
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
 *
 * Soundness precondition (InOut-use discipline) is validated once per function
 * by the driver before the mutator runs, so per-region checks are unnecessary
 * here. Keeping the check out of the visitor avoids O(function-size) work for
 * every nested ``SeqStmts`` we visit.
 */
class CanonicalizeIOOrderMutator : public IRMutator {
 public:
  CanonicalizeIOOrderMutator() : io_ops_(IOCategoryOps::Build()) {}

  /// Scope the IO reorder to bodies of `ForKind::Pipeline` loops. Non-pipelined
  /// code is visited recursively but its SeqStmts are left as-is.
  ///
  /// On exit from a pipeline scope, demote `kind_` to `Sequential` and strip
  /// the `pipeline_stages` attr together — the marker has served its purpose
  /// (gated this reorder) and must not survive past this pass. The bidirectional
  /// invariant `kind == Pipeline ⇔ pipeline_stages attr present` (PipelineLoopValid)
  /// guarantees the attr is present on entry, so we strip unconditionally.
  /// The PipelineResolved verifier checks the post-condition: no
  /// ForKind::Pipeline loops downstream of CanonicalizeIOOrder.
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    const bool is_pipeline = (op->kind_ == ForKind::Pipeline);
    if (is_pipeline) inside_pipeline_depth_++;
    auto visited = IRMutator::VisitStmt_(op);
    if (is_pipeline) inside_pipeline_depth_--;

    if (!is_pipeline) return visited;
    auto visited_for = std::dynamic_pointer_cast<const ForStmt>(visited);
    if (!visited_for) return visited;
    auto demoted = MutableCopy(visited_for);
    demoted->kind_ = ForKind::Sequential;
    demoted->attrs_ = StripAttr(demoted->attrs_, kPipelineStagesAttr);
    return demoted;
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    // Recurse first so any nested SeqStmts are reordered bottom-up.
    auto visited = IRMutator::VisitStmt_(op);
    if (inside_pipeline_depth_ == 0) {
      return visited;  // outside any pipeline scope — do not reorder
    }
    auto seq = std::dynamic_pointer_cast<const SeqStmts>(visited);
    if (!seq || seq->stmts_.size() < 2) {
      return visited;  // single stmt — nothing to reorder
    }
    return ReorderRegion(seq);
  }

 private:
  /// Depth counter: increments on entry to a `ForKind::Pipeline`, decrements
  /// on exit. Non-zero when a pipeline loop is an ancestor of the currently
  /// visited SeqStmts. Supports nested pipelines (each level increments).
  int inside_pipeline_depth_ = 0;

  /// Stable, priority-aware topological sort.
  ///
  /// Complexity: O(N log N + E) per region — the dependency graph is built
  /// once, successors/in-degrees are filled in a single linear pass, and the
  /// ready set is maintained as a min-heap keyed by (category, index).
  /// N is the number of top-level stmts in the region; E is the number of
  /// def-use edges produced by ``BuildStmtDependencyGraph`` (equal to the
  /// region's total variable uses, and so O(N²) in the pathological worst
  /// case even though it is linear-with-a-small-constant in practice).
  ///
  /// A trailing terminator (`YieldStmt` / `ReturnStmt` / `BreakStmt` /
  /// `ContinueStmt`) is peeled off before sorting and re-appended at the end
  /// so stores can never be emitted after it (which would make them
  /// unreachable / semantically dropped).
  StmtPtr ReorderRegion(const SeqStmtsPtr& seq) {
    // The driver already validated the InOut-use discipline at function scope,
    // so passing `nullptr` here skips a redundant check inside the builder.
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

    // Ready-set as a min-heap keyed by (category, original_index). Emitting
    // the smallest category first gives the desired top-to-bottom layout —
    // ``ScalarCompute`` (0) first, then ``Load`` (1), ``TileCompute`` (2), and
    // finally ``Store`` (3). Using the original index as the tiebreaker keeps
    // the sort stable within each tier.
    using HeapKey = std::pair<int, size_t>;
    std::priority_queue<HeapKey, std::vector<HeapKey>, std::greater<>> ready;
    auto key_for = [&](size_t i) -> HeapKey { return {static_cast<int>(cats[i]), i}; };
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
    INTERNAL_CHECK_SPAN(out.size() == sort_count, seq->span_)
        << "CanonicalizeIOOrder: dependency graph appears cyclic — should be impossible "
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

  IOCategoryOps io_ops_;
};

}  // namespace

namespace pass {

Pass CanonicalizeIOOrder() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    INTERNAL_CHECK(program) << "CanonicalizeIOOrder cannot run on null program";

    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    bool any_change = false;
    for (const auto& [gvar, func] : program->functions_) {
      // Validate the InOut-use discipline once per function: variable scopes
      // don't cross function boundaries, so a single walk over the function
      // body catches every violation that could affect any nested SeqStmts.
      // Under strict verification such violations are rejected earlier, but
      // with VerificationLevel.NONE a non-conforming function can reach us,
      // and we must not reorder potentially-unsound dataflow.
      if (!stmt_dep::CollectInOutUseDisciplineDiagnostics(func->body_, program).empty()) {
        new_functions.emplace(gvar, func);
        continue;
      }
      CanonicalizeIOOrderMutator mutator;
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

  return CreateProgramPass(pass_func, "CanonicalizeIOOrder", kCanonicalizeIOOrderProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

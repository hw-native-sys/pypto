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

/// @file acc_packing.cpp
/// Whole-function decision: which batched `Acc` accumulators are packed along
/// COLUMNS instead of rows.
///
/// ## Why the decision has to be whole-function
///
/// The rewrite loop in `rewrite.cpp` is per-block, and so is its
/// `batch_matmul_only_vars` pre-scan. A batched accumulator routinely spans
/// blocks:
///
/// ```text
/// acc0 = tile.create([B, M, N])            # outer block
/// for k, (acc,) in pl.range(...):          # acc0 is the iter_arg's init
///     acc1 = tile.batch_matmul_acc(acc, l, r, k == 0)   # inner block
///     accf = pl.yield_(acc1)
/// out = tile.store(accf, [0, 0, 0], out)   # outer block again
/// ```
///
/// The `tile.create` is flattened by the generic rule long before the
/// `tile.batch_matmul_acc` inside the loop is ever seen, so nothing block-local
/// can tell it to allocate `[M, B*N]` rather than `[B*M, N]`. The analysis below
/// therefore runs once over the whole function body before any rewriting.
///
/// ## What it computes
///
/// A *chain* is a connected component of the same-buffer alias graph:
///
/// | Edge | Source |
/// | ---- | ------ |
/// | `v` -- `a` | `v = tile.batch_matmul_acc(a, l, r[, p])` (in-place on `a`) |
/// | `v` -- `w` | plain SSA alias `v = w` |
/// | `iter_arg` -- `init` / `iter_arg` -- `yield[i]` / `return_var[i]` -- `iter_arg[i]` | loop carry |
/// | `return_var[i]` -- `then_yield[i]` / `else_yield[i]` | `IfStmt` merge |
///
/// A chain is column-packed only when EVERY member is produced and consumed by a
/// form this pass rewrites page-wise, and the geometry is one L0C can address:
/// `M % 16 == 0`, `N % 16 == 0`, a 4-byte accumulator element, and the whole
/// `[M, B*N]` tile fits L0C. Anything else is left on the legacy row-packed
/// path — which is correct only for `N <= 16` (a single L0C block column), so a
/// wider one is rejected here with an actionable diagnostic rather than emitted
/// and mis-addressed downstream.
///
/// Complexity: one walk over the body, a union-find over the alias edges, and
/// one pass over the components — O(N log N) including the hash lookups.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/storage_size.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/flatten_tile_nd_to_2d/rewrite_internal.h"

namespace pypto {
namespace ir {

using transform_utils::FlattenToStmts;

namespace flatten_tile_nd_to_2d {
namespace rewrite_internal {
namespace {

/// L0C's NZ box for a 4-byte accumulator element is 16x16 — the granularity both
/// `GetSliceAccumulatorGeometry` (which needs a box-aligned column origin and
/// whole-box parent extents) and PTO's `alloc_tile` verifier work in.
constexpr int64_t kAccBoxDim = 16;
/// The element width `kAccFractal` implies for a 16x16 box: 1024 / 256 bytes.
constexpr uint64_t kAccElementBits = 32;

/// L0C byte budget. Mirrors `GetMatBudgetBytes` in rewrite_utils.cpp: without a
/// configured backend (most unit tests) every shape "fits", so the decision is
/// driven purely by geometry and stays reproducible off-device. A backend that
/// reports 0 means "no limit", matching `AllocateMemoryAddr`.
uint64_t GetAccBudgetBytes() {
  if (!backend::BackendConfig::IsConfigured()) return std::numeric_limits<uint64_t>::max();
  const uint64_t size = backend::GetBackend()->GetMemSize(MemorySpace::Acc);
  return size == 0 ? std::numeric_limits<uint64_t>::max() : size;
}

/// How a member Var is consumed. Only the first three are rewritable page-wise;
/// `kOther` disqualifies the whole chain.
enum class UseKind {
  kAccumulate,  ///< argument 0 of `tile.batch_matmul_acc` — the in-place destination.
  kDrainStore,  ///< argument 0 of `tile.store` — drained one page per batch index.
  kDrainMove,   ///< argument 0 of `tile.move` — drained one page per batch index.
  kCarry,       ///< a loop/branch carry edge or a plain SSA alias — buffer-preserving.
  kOther,       ///< anything else: the chain cannot be column-packed.
};

struct UseRecord {
  UseKind kind = UseKind::kOther;
  CallPtr call;  ///< The consuming call for kAccumulate / kDrainStore; null otherwise.
};

/// One `tile.batch_matmul_acc` that demands a batched accumulator.
struct AccSeed {
  CallPtr call;
  AssignStmtPtr assign;
  const Var* acc = nullptr;  ///< null when the operand is not a named value.
  int64_t batch_count = 1;
  int64_t rows = 0;
  int64_t cols = 0;
  DataType dtype = DataType::FP32;
  std::vector<int64_t> batch_dims;
  std::vector<int64_t> nd_shape;
};

/// Static shape, or nullopt when any dimension is symbolic.
std::optional<std::vector<int64_t>> StaticShape(const std::vector<ExprPtr>& shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.size());
  for (const auto& dim : shape) {
    auto ci = As<ConstInt>(dim);
    if (!ci || ci->value_ <= 0) return std::nullopt;
    dims.push_back(ci->value_);
  }
  return dims;
}

/// Whether the tile carries a partial `valid_shape`. A partially-valid batched
/// accumulator has no single parent valid rectangle once the pages sit side by
/// side (page `b`'s valid region starts at `b*N` but is only `N_valid` wide), so
/// such a chain is left alone rather than described wrongly.
bool HasPartialValidShape(const TileTypePtr& tile) {
  if (!tile || !tile->tile_view_.has_value()) return false;
  const auto& valid = tile->tile_view_->valid_shape;
  if (valid.empty()) return false;
  return !tile_view_semantics::ShapeExprListsEquivalent(valid, tile->shape_);
}

/// `pl.yield_` statements belonging to THIS block. The search descends through
/// `SeqStmts` / `ScopeStmt` nesting only: a yield inside a nested If / For /
/// While terminates *that* construct, so attributing it here would wire the
/// wrong carry edge.
void CollectOwnYields(const StmtPtr& body, std::vector<YieldStmtPtr>* out) {
  if (!body) return;
  if (auto yield = As<YieldStmt>(body)) {
    out->push_back(yield);
    return;
  }
  if (auto seq = As<SeqStmts>(body)) {
    for (const auto& child : seq->stmts_) CollectOwnYields(child, out);
    return;
  }
  if (auto scope = As<ScopeStmt>(body)) {
    CollectOwnYields(scope->body_, out);
  }
}

/// Phase A + B: collect the alias graph, the use classification and the
/// accumulator seeds, then close the graph with union-find.
class ChainCollector {
 public:
  void Run(const FunctionPtr& func) {
    // A tile-typed parameter can never be retyped (Rewrite never touches
    // params_), so a chain that reaches one is disqualified up front.
    for (const auto& param : func->params_) {
      if (!param) continue;
      NoteVar(param);
      Poison(param.get());
    }
    Walk(FlattenToStmts(func->body_));
  }

  // --- Results ---------------------------------------------------------------

  const std::vector<AccSeed>& seeds() const { return seeds_; }

  /// Union-find with iterative path compression (a chain can be arbitrarily
  /// long, so recursion here would be a stack depth proportional to the IR).
  const Var* Find(const Var* var) {
    auto it = parent_.find(var);
    if (it == parent_.end()) return var;
    const Var* root = var;
    while (true) {
      auto step = parent_.find(root);
      if (step == parent_.end() || step->second == root) break;
      root = step->second;
    }
    while (var != root) {
      const Var* next = parent_[var];
      parent_[var] = root;
      var = next;
    }
    return root;
  }

  /// Members grouped by component root, in first-seen order so the diagnostics
  /// and the emitted IR do not depend on hash iteration order.
  std::unordered_map<const Var*, std::vector<const Var*>> Components() {
    std::unordered_map<const Var*, std::vector<const Var*>> out;
    for (const auto* var : var_order_) out[Find(var)].push_back(var);
    return out;
  }

  bool IsPoisoned(const Var* var) const { return poisoned_.count(var) != 0; }
  const std::vector<UseRecord>* UsesOf(const Var* var) const {
    auto it = uses_.find(var);
    return it == uses_.end() ? nullptr : &it->second;
  }
  AssignStmtPtr DefOf(const Var* var) const {
    auto it = defs_.find(var);
    return it == defs_.end() ? nullptr : it->second;
  }
  VarPtr VarNode(const Var* var) const {
    auto it = var_nodes_.find(var);
    return it == var_nodes_.end() ? nullptr : it->second;
  }
  bool IsDeclaredReturnVar(const Var* var) const { return declared_return_vars_.count(var) != 0; }

 private:
  // --- Graph bookkeeping -----------------------------------------------------

  void NoteVar(const VarPtr& var) {
    if (!var) return;
    if (parent_.emplace(var.get(), var.get()).second) {
      var_order_.push_back(var.get());
      var_nodes_.emplace(var.get(), var);
    }
  }

  void Union(const Var* lhs, const Var* rhs) {
    if (!lhs || !rhs) return;
    const Var* a = Find(lhs);
    const Var* b = Find(rhs);
    if (a != b) parent_[a] = b;
  }

  void Poison(const Var* var) {
    if (var) poisoned_.insert(var);
  }

  void AddUse(const Var* var, UseKind kind, const CallPtr& call) {
    if (var) uses_[var].push_back(UseRecord{kind, call});
  }

  /// Every Var reachable inside @p expr is an unclassified use.
  ///
  /// `Submit` is covered alongside `Call` here and in `HandleAssign`. It cannot
  /// occur today -- `FlattenTileNdTo2D` is gated to InCore functions, whose
  /// bodies hold no task launches, and a `pl.manual_scope` orchestration body
  /// carries no `tile.batch_matmul_acc` to seed a chain from -- but unlike the
  /// forwarding sites elsewhere in this pass, a missed use here is a soundness
  /// input, not a dropped rewrite: an unrecorded operand would let a chain look
  /// unconsumed, be marked packable, and then leave that operand reading an
  /// `[M, B*N]` tile as if it were `[B*M, N]`. Recording it as `kOther` costs
  /// nothing and fails safe (see `.claude/rules/pass-submit-awareness.md`).
  void RecordOtherUses(const ExprPtr& expr) {
    if (!expr) return;
    if (auto var = AsVarLike(expr)) {
      NoteVar(var);
      AddUse(var.get(), UseKind::kOther, nullptr);
      return;
    }
    if (auto tup = As<MakeTuple>(expr)) {
      for (const auto& element : tup->elements_) RecordOtherUses(element);
      return;
    }
    if (auto call = As<Call>(expr)) {
      for (const auto& arg : call->args_) RecordOtherUses(arg);
      return;
    }
    if (auto submit = As<Submit>(expr)) {
      // `deps_` are part of a Submit's use-def chain, not metadata.
      for (const auto& arg : submit->args_) RecordOtherUses(arg);
      for (const auto& dep : submit->deps_) RecordOtherUses(dep);
      return;
    }
    if (auto get_item = As<TupleGetItemExpr>(expr)) {
      RecordOtherUses(get_item->tuple_);
    }
  }

  // --- Walk ------------------------------------------------------------------

  void Walk(const std::vector<StmtPtr>& stmts) {
    for (const auto& stmt : stmts) {
      if (!stmt) continue;
      if (auto seq = As<SeqStmts>(stmt)) {
        Walk(seq->stmts_);
      } else if (auto scope = As<ScopeStmt>(stmt)) {
        Walk(FlattenToStmts(scope->body_));
      } else if (auto if_stmt = As<IfStmt>(stmt)) {
        HandleIf(if_stmt);
      } else if (auto for_stmt = As<ForStmt>(stmt)) {
        RecordOtherUses(for_stmt->start_);
        RecordOtherUses(for_stmt->stop_);
        RecordOtherUses(for_stmt->step_);
        HandleLoop(for_stmt);
      } else if (auto while_stmt = As<WhileStmt>(stmt)) {
        RecordOtherUses(while_stmt->condition_);
        HandleLoop(while_stmt);
      } else if (auto yield = As<YieldStmt>(stmt)) {
        if (handled_yields_.count(yield.get()) == 0) {
          for (const auto& value : yield->value_) RecordOtherUses(value);
        }
      } else if (auto ret = As<ReturnStmt>(stmt)) {
        for (const auto& value : ret->value_) RecordOtherUses(value);
      } else if (auto eval = As<EvalStmt>(stmt)) {
        RecordOtherUses(eval->expr_);
      } else if (auto assign = As<AssignStmt>(stmt)) {
        HandleAssign(assign);
      }
    }
  }

  void HandleAssign(const AssignStmtPtr& assign) {
    NoteVar(assign->var_);
    defs_[assign->var_.get()] = assign;

    // `As<Call>` is exact-kind and so does not match a `Submit`; `RecordOtherUses`
    // handles that sibling kind, marking every arg and dep an unclassified use so
    // the chain is rejected rather than silently repacked underneath a task
    // launch. See the note on `RecordOtherUses`.
    auto call = As<Call>(assign->value_);
    if (!call) {
      if (auto alias = AsVarLike(assign->value_)) {
        NoteVar(alias);
        Union(assign->var_.get(), alias.get());
        AddUse(alias.get(), UseKind::kCarry, nullptr);
      } else {
        RecordOtherUses(assign->value_);
      }
      return;
    }
    if (As<GlobalVar>(call->op_)) {
      for (const auto& arg : call->args_) RecordOtherUses(arg);
      return;
    }

    const bool is_acc = IsOp(call, "tile.batch_matmul_acc");
    const bool is_store = IsOp(call, "tile.store");
    const bool is_move = IsOp(call, "tile.move");
    for (size_t i = 0; i < call->args_.size(); ++i) {
      auto operand = AsVarLike(call->args_[i]);
      if (!operand) {
        RecordOtherUses(call->args_[i]);
        continue;
      }
      NoteVar(operand);
      if (is_acc && i == 0) {
        AddUse(operand.get(), UseKind::kAccumulate, call);
        Union(assign->var_.get(), operand.get());
      } else if (is_store && i == 0) {
        AddUse(operand.get(), UseKind::kDrainStore, call);
      } else if (is_move && i == 0) {
        // A move READS the accumulator and writes a fresh tile elsewhere, so it
        // is a drain, not a carry -- it must not join the chain's component.
        AddUse(operand.get(), UseKind::kDrainMove, call);
      } else {
        AddUse(operand.get(), UseKind::kOther, call);
      }
    }
    if (is_acc) RecordSeed(assign, call);
  }

  void RecordSeed(const AssignStmtPtr& assign, const CallPtr& call) {
    auto acc_type = As<TileType>(call->args_[0]->GetType());
    if (!acc_type || acc_type->shape_.size() <= 2) return;  // 2-D acc: nothing batched to pack.
    auto dims = StaticShape(acc_type->shape_);
    if (!dims) return;  // A symbolic accumulator never reaches the batch unroll anyway.

    AccSeed seed;
    seed.call = call;
    seed.assign = assign;
    auto acc_var = AsVarLike(call->args_[0]);
    seed.acc = acc_var ? acc_var.get() : nullptr;
    seed.nd_shape = *dims;
    seed.rows = (*dims)[dims->size() - 2];
    seed.cols = dims->back();
    seed.batch_dims.assign(dims->begin(), dims->end() - 2);
    seed.dtype = acc_type->dtype_;
    seed.batch_count = 1;
    for (int64_t d : seed.batch_dims) seed.batch_count *= d;
    if (seed.batch_count <= 1) return;  // The batch_count == 1 fast path is unchanged.
    seeds_.push_back(std::move(seed));
  }

  template <typename LoopPtr>
  void HandleLoop(const LoopPtr& loop) {
    std::vector<YieldStmtPtr> yields;
    CollectOwnYields(loop->body_, &yields);
    const bool matched = yields.size() == 1 && loop->return_vars_.size() == loop->iter_args_.size() &&
                         yields[0]->value_.size() == loop->iter_args_.size();

    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& iter_arg = loop->iter_args_[i];
      NoteVar(iter_arg);
      if (auto init = AsVarLike(iter_arg->initValue_)) {
        NoteVar(init);
        Union(iter_arg.get(), init.get());
        AddUse(init.get(), UseKind::kCarry, nullptr);
      } else {
        RecordOtherUses(iter_arg->initValue_);
        Poison(iter_arg.get());
      }
      if (i < loop->return_vars_.size()) {
        const auto& return_var = loop->return_vars_[i];
        NoteVar(return_var);
        declared_return_vars_.insert(return_var.get());
        Union(return_var.get(), iter_arg.get());
        if (!matched) Poison(return_var.get());
      }
      if (!matched) {
        Poison(iter_arg.get());
        continue;
      }
      if (auto yielded = AsVarLike(yields[0]->value_[i])) {
        NoteVar(yielded);
        Union(iter_arg.get(), yielded.get());
        AddUse(yielded.get(), UseKind::kCarry, nullptr);
      } else {
        RecordOtherUses(yields[0]->value_[i]);
        Poison(iter_arg.get());
      }
    }

    for (const auto& yield : yields) {
      if (!matched) {
        for (const auto& value : yield->value_) RecordOtherUses(value);
      }
      handled_yields_.insert(yield.get());
    }
    Walk(FlattenToStmts(loop->body_));
  }

  void HandleIf(const IfStmtPtr& if_stmt) {
    RecordOtherUses(if_stmt->condition_);
    std::vector<YieldStmtPtr> then_yields;
    std::vector<YieldStmtPtr> else_yields;
    CollectOwnYields(if_stmt->then_body_, &then_yields);
    if (if_stmt->else_body_.has_value()) CollectOwnYields(*if_stmt->else_body_, &else_yields);

    const size_t arity = if_stmt->return_vars_.size();
    const bool matched = arity == 0 || (then_yields.size() == 1 && then_yields[0]->value_.size() == arity &&
                                        if_stmt->else_body_.has_value() && else_yields.size() == 1 &&
                                        else_yields[0]->value_.size() == arity);

    for (size_t i = 0; i < arity; ++i) {
      const auto& return_var = if_stmt->return_vars_[i];
      NoteVar(return_var);
      declared_return_vars_.insert(return_var.get());
      if (!matched) {
        Poison(return_var.get());
        continue;
      }
      for (const auto* arm : {&then_yields, &else_yields}) {
        auto value = (*arm)[0]->value_[i];
        if (auto yielded = AsVarLike(value)) {
          NoteVar(yielded);
          Union(return_var.get(), yielded.get());
          AddUse(yielded.get(), UseKind::kCarry, nullptr);
        } else {
          RecordOtherUses(value);
          Poison(return_var.get());
        }
      }
    }

    for (auto* arm : {&then_yields, &else_yields}) {
      for (const auto& yield : *arm) {
        if (!matched) {
          for (const auto& value : yield->value_) RecordOtherUses(value);
        }
        handled_yields_.insert(yield.get());
      }
    }

    Walk(FlattenToStmts(if_stmt->then_body_));
    if (if_stmt->else_body_.has_value()) Walk(FlattenToStmts(*if_stmt->else_body_));
  }

  std::unordered_map<const Var*, const Var*> parent_;
  std::vector<const Var*> var_order_;
  std::unordered_map<const Var*, VarPtr> var_nodes_;
  std::unordered_map<const Var*, AssignStmtPtr> defs_;
  std::unordered_map<const Var*, std::vector<UseRecord>> uses_;
  std::unordered_set<const Var*> poisoned_;
  std::unordered_set<const Var*> declared_return_vars_;
  std::unordered_set<const YieldStmt*> handled_yields_;
  std::vector<AccSeed> seeds_;
};

/// Verdict for one chain: either a plan, or the reason it cannot be packed.
struct ChainVerdict {
  // Heap-allocated so returning ChainVerdict never move-constructs AccPackingPlan
  // through std::optional (clang-analyzer-core.uninitialized.Assign false-positive
  // on AccPackingPlan::batch_count when the plan was filled from a seed pointer
  // the analyzer cannot see was published by RecordSeed).
  std::unique_ptr<AccPackingPlan> plan;
  std::string reason;
  bool has_batch_producer = false;
  /// The chain holds more than one allocating definition. Neither packing can
  /// express it, so it is rejected regardless of how narrow the pages are.
  bool multi_root = false;
  /// The row-packed fallback cannot rescue this chain either, so reject at any
  /// page width instead of falling through to it.
  bool force_reject = false;
  /// The reason has nothing to do with page geometry (it fails at batch 1, or at
  /// any width), so `RejectChain` omits the column-packing rationale — printing
  /// it would point the reader at the wrong thing.
  bool geometry_irrelevant = false;
  /// Reason-specific remedy for `RejectChain`. Empty means the default one.
  std::string workaround;
};

/// Whether `v = tile.batch_matmul(...)` produces the ND accumulator directly.
bool IsBatchMatmulProducer(const AssignStmtPtr& def, const std::vector<int64_t>& nd_shape) {
  auto call = def ? As<Call>(def->value_) : nullptr;
  if (!IsOp(call, "tile.batch_matmul")) return false;
  auto result = As<TileType>(call->GetType());
  auto dims = result ? StaticShape(result->shape_) : std::nullopt;
  return dims.has_value() && *dims == nd_shape;
}

/// Validate one chain against the column-packing contract.
ChainVerdict JudgeChain(ChainCollector& graph, const std::vector<const Var*>& members,
                        const std::vector<const AccSeed*>& chain_seeds) {
  ChainVerdict verdict;
  const AccSeed& head = *chain_seeds.front();

  for (const auto* seed : chain_seeds) {
    if (seed->acc == nullptr) {
      verdict.reason = "the accumulator operand is not a named value, so its chain cannot be tracked";
      return verdict;
    }
    if (seed->nd_shape != head.nd_shape || seed->dtype != head.dtype) {
      verdict.reason =
          "the same accumulator is written by two tile.batch_matmul_acc calls with different "
          "batch geometries, so there is no single packed shape for it";
      return verdict;
    }
  }

  const int64_t batch_count = head.batch_count;
  const int64_t rows = head.rows;
  const int64_t cols = head.cols;

  // First pass over the members: everything that decides *whether* the chain can
  // be rewritten page-wise, plus the producer flag the caller needs even on the
  // reject path (a Vec-staged producer has no working fallback either).
  int64_t allocating_defs = 0;
  for (const auto* member : members) {
    auto def = graph.DefOf(member);
    if (IsBatchMatmulProducer(def, head.nd_shape)) {
      verdict.has_batch_producer = true;
      ++allocating_defs;
      continue;
    }
    auto def_call = def ? As<Call>(def->value_) : nullptr;
    if (IsOp(def_call, "tile.create")) ++allocating_defs;
  }

  // One chain is one buffer: the whole packing rests on every member naming the
  // same allocation. Two allocating definitions joined by a control-flow merge
  // (`if k == 0: acc = matmul(...) else: acc = matmul_acc(acc, ...)`, or a loop
  // body that re-creates the accumulator) would have to be reconciled with an
  // L0C-to-L0C copy, which the ISA does not have. The row-packed path cannot
  // express it either, so this is fatal at any page width -- reported here so
  // the user gets an actionable message instead of MemoryReuse's YieldFixup
  // internal error twenty passes later.
  if (allocating_defs > 1) {
    verdict.multi_root = true;
    verdict.force_reject = true;
    verdict.geometry_irrelevant = true;
    verdict.reason = "the accumulator is allocated by " + std::to_string(allocating_defs) +
                     " separate definitions that control flow then merges into one value, and "
                     "reconciling them would need an L0C-to-L0C copy the hardware does not have";
    verdict.workaround =
        "Workaround: give the accumulator a single allocation that every branch and iteration "
        "writes in place -- hoist the tile.create above the branch (or the loop) and make the "
        "first-iteration case a tile.batch_matmul_acc with init_cond=True, which overwrites the "
        "pages instead of allocating a second buffer.";
    return verdict;
  }

  if (storage_size::GetStorageBitWidth(head.dtype) != kAccElementBits) {
    verdict.reason = "the accumulator element is not 4 bytes wide (" + head.dtype.ToString() +
                     "), so its L0C tile has no 16x16 box grid";
    return verdict;
  }
  if (rows % kAccBoxDim != 0) {
    verdict.reason = "the page row extent M=" + std::to_string(rows) + " is not a multiple of " +
                     std::to_string(kAccBoxDim);
    return verdict;
  }
  if (cols % kAccBoxDim != 0) {
    verdict.reason = "the page column extent N=" + std::to_string(cols) + " is not a multiple of " +
                     std::to_string(kAccBoxDim);
    return verdict;
  }

  const uint64_t packed_bytes = static_cast<uint64_t>(batch_count) * static_cast<uint64_t>(rows) *
                                static_cast<uint64_t>(cols) * (kAccElementBits / 8);
  const uint64_t budget = GetAccBudgetBytes();
  if (packed_bytes > budget) {
    verdict.reason = "the packed accumulator needs " + std::to_string(packed_bytes) +
                     " bytes of Acc (L0C), which exceeds the " + std::to_string(budget) +
                     "-byte budget this target has";
    return verdict;
  }

  for (const auto* member : members) {
    if (graph.IsPoisoned(member)) {
      verdict.reason =
          "the accumulator reaches a function parameter or a control-flow carry this pass cannot retype";
      return verdict;
    }

    auto member_node = graph.VarNode(member);
    auto member_type = member_node ? As<TileType>(member_node->GetType()) : nullptr;
    auto member_dims = member_type ? StaticShape(member_type->shape_) : std::nullopt;
    if (!member_dims || *member_dims != head.nd_shape || member_type->dtype_ != head.dtype) {
      verdict.reason = "an SSA name for the accumulator does not carry its " +
                       std::to_string(head.nd_shape.size()) + "-D shape, so the chain is not one buffer";
      return verdict;
    }
    if (HasPartialValidShape(member_type)) {
      verdict.reason =
          "the accumulator carries a partial valid_shape, which has no equivalent once the pages sit "
          "side by side in one tile";
      return verdict;
    }
    // Packing commits the buffer to Acc. An explicit annotation to any other
    // space is a stated fact this pass must not silently override -- it is also
    // illegal for a matmul accumulator, so leave it to be reported as such.
    if (member_type->memory_space_.has_value() && *member_type->memory_space_ != MemorySpace::Acc) {
      verdict.reason = "the accumulator is annotated to live in " +
                       MemorySpaceToString(*member_type->memory_space_) +
                       " memory, but only the matrix unit writes Acc (L0C)";
      return verdict;
    }

    // Definition: a create root, the producing batch_matmul, another
    // batch_matmul_acc in the chain, or an implicit binding (iter_arg / a loop or
    // branch return_var).
    auto def = graph.DefOf(member);
    if (!def) {
      const bool implicit = As<IterArg>(member_node) != nullptr || graph.IsDeclaredReturnVar(member);
      if (!implicit) {
        verdict.reason = "the accumulator has no definition this pass can rewrite";
        return verdict;
      }
    } else {
      auto def_call = As<Call>(def->value_);
      const bool ok = AsVarLike(def->value_) != nullptr ||  // plain SSA alias
                      IsOp(def_call, "tile.batch_matmul_acc") || IsBatchMatmulProducer(def, head.nd_shape) ||
                      (IsOp(def_call, "tile.create") && !def_call->args_.empty());
      if (!ok) {
        // Not a packing problem, and not a batch problem: no op outside the
        // matmul family can write Acc at all, so the same accumulator fails the
        // same way at batch 1. The rejection still has to happen here -- pass 13
        // is where a batched accumulator stops, and CanonicalizeTileSlice's
        // guard tells users that a batch_matmul_acc never reaches it -- but the
        // message must name the real cause and must not offer either
        // batch-packing remedy, since neither can work.
        const std::string produced =
            def_call && def_call->op_ ? def_call->op_->name_ : std::string("a non-call value");
        verdict.reason = "the accumulator is produced by " + produced +
                         ", which cannot write Acc (L0C) at all -- only the matrix unit writes that "
                         "memory, and no target has a data path into it";
        verdict.workaround =
            "Workaround: an accumulator has to come from a matmul, or from an allocation "
            "(pl.tile.create) the compiler is free to place in Acc memory. This is not about the "
            "batch dimension -- the same accumulator fails the same way at batch 1.";
        verdict.force_reject = true;
        verdict.geometry_irrelevant = true;
        return verdict;
      }
    }

    // Uses: accumulate, drain, or a buffer-preserving carry.
    const auto* member_uses = graph.UsesOf(member);
    if (member_uses == nullptr) continue;
    for (const auto& use : *member_uses) {
      if (use.kind == UseKind::kCarry || use.kind == UseKind::kAccumulate) continue;
      if (use.kind == UseKind::kDrainStore) {
        auto store = use.call;
        auto offsets = store->args_.size() >= 2 ? As<MakeTuple>(store->args_[1]) : nullptr;
        auto out_type = store->args_.size() >= 3 ? AsTensorTypeLike(store->args_[2]->GetType()) : nullptr;
        if (store->args_.size() != 3 || !offsets || !out_type ||
            offsets->elements_.size() != head.nd_shape.size() ||
            out_type->shape_.size() != head.nd_shape.size()) {
          verdict.reason =
              "the accumulator is stored through a tile.store whose window this pass cannot split "
              "into one store per batch page";
          return verdict;
        }
        continue;
      }
      if (use.kind == UseKind::kDrainMove) {
        // The cube->vector epilogue. Splitting the move page-wise is easy -- one
        // move per column window -- but the pages then have to be GATHERED into
        // one row-packed vector tile, and codegen cannot express that: the moved
        // page keeps L0C's col_major/1024 block layout, while the row-packed
        // destination is row_major, and tile.assemble rejects the mismatch
        // ("blayout mismatch between source and result; pto.subview requires
        // identical block layout", src/backend/common/pto_ops_shared.cpp).
        //
        // This limit is NOT specific to accumulators: a plain batch>1
        // tile.batch_matmul followed by any vector op fails in exactly the same
        // place, at every page width including 16. So there is nothing to pack
        // here, and emitting the page-wise form would only move the failure from
        // an actionable message to a codegen assertion.
        verdict.reason =
            "the accumulator is drained by tile.move into a vector epilogue, and gathering the "
            "pages of a multi-batch Acc (L0C) result into one vector tile is not expressible -- the "
            "moved pages keep L0C's block layout, which tile.assemble cannot write into a "
            "row-packed vector tile";
        verdict.workaround =
            "Workaround: store each batch page to global memory and reload it for the vector work, "
            "or write the batch loop out in the kernel. This limit is not specific to accumulators "
            "-- a plain batch>1 pl.matmul followed by a vector op fails the same way, at any page "
            "width.";
        // No force_reject: the N <= 16 fallback keeps whatever behaviour it had.
        verdict.geometry_irrelevant = true;
        return verdict;
      }
      const std::string consumer =
          use.call && use.call->op_ ? use.call->op_->name_ : std::string("a non-call consumer");
      verdict.reason = "the accumulator is consumed by " + consumer +
                       ", which reads it as one flat tile rather than one page at a time";
      return verdict;
    }
  }

  // Publish via unique_ptr so returning ChainVerdict only moves the pointer —
  // never AccPackingPlan itself through std::optional's move constructor, which
  // triggered a clang-analyzer-core.uninitialized.Assign false positive on
  // batch_count (the analyzer cannot see that `head` was published by RecordSeed).
  auto plan = std::make_unique<AccPackingPlan>();
  plan->batch_count = batch_count;
  plan->rows = rows;
  plan->cols = cols;
  plan->dtype = head.dtype;
  plan->batch_dims = head.batch_dims;
  plan->nd_shape = head.nd_shape;
  verdict.plan = std::move(plan);
  return verdict;
}

/// Surface a chain that can neither be column-packed nor left on the row-packed
/// path. Row packing survives only for `N <= 16` (one L0C block column, which
/// `CanonicalizeTileSlice` explicitly whitelists) and only when nothing stages
/// the accumulator through Vec on the way in.
///
/// The middle paragraph explains the packing *geometry*, so it is printed only
/// for the reasons that are about geometry. A chain rejected for holding two
/// allocations, or for a definition that cannot write Acc at all, fails at any
/// page width and at batch 1 too; both get their own remedy instead, because
/// telling those users about 16-column pages would be wrong.
void RejectChain(const AccSeed& seed, const ChainVerdict& verdict) {
  const int64_t packed_cols = seed.batch_count * seed.cols;
  const uint64_t packed_bytes = static_cast<uint64_t>(seed.batch_count) * static_cast<uint64_t>(seed.rows) *
                                static_cast<uint64_t>(seed.cols) * (kAccElementBits / 8);

  std::string geometry;
  if (!verdict.geometry_irrelevant) {
    geometry = "The pages of a batched accumulator have to be packed along COLUMNS — one " +
               std::to_string(seed.rows) + "x" + std::to_string(packed_cols) +
               " Acc (L0C) tile with page b at tile.slice(acc, [" + std::to_string(seed.rows) + ", " +
               std::to_string(seed.cols) + "], [0, b * " + std::to_string(seed.cols) +
               "]) — because the hardware MAD writes its destination compactly and has no destination "
               "stride, so a row window of a multi-block-column accumulator is not addressable. That "
               "packing needs M and N to be whole multiples of " +
               std::to_string(kAccBoxDim) + ", a 4-byte (FP32/INT32) accumulator, one buffer per chain, " +
               "and the whole " + std::to_string(packed_bytes) + "-byte tile to fit L0C.\n";
  }

  const std::string workaround =
      verdict.workaround.empty()
          ? "Workarounds: write the batch loop out in the kernel and accumulate each page into its "
            "own 2-D tile (pl.matmul / pl.matmul_acc on 2-D operands); or keep the accumulator at "
            "most " +
                std::to_string(kAccBoxDim) +
                " columns wide, which fits a single L0C block column and needs no packing."
          : verdict.workaround;

  CHECK_SPAN(false, seed.call->span_)
      << "tile.batch_matmul_acc: cannot lower a batch-" << seed.batch_count << " accumulator of " << seed.rows
      << "x" << seed.cols << " " << seed.dtype.ToString() << " pages, because " << verdict.reason << ".\n"
      << geometry << workaround;
}

}  // namespace

AccPackingMapPtr BuildAccPackingMap(const FunctionPtr& func) {
  auto map = std::make_shared<AccPackingMap>();
  if (!func || !func->body_) return map;

  ChainCollector graph;
  graph.Run(func);
  if (graph.seeds().empty()) return map;

  // Group the seeds by the chain (component) their accumulator belongs to.
  std::vector<const Var*> chain_order;
  std::unordered_map<const Var*, std::vector<const AccSeed*>> chain_seeds;
  for (const auto& seed : graph.seeds()) {
    // An unnamed accumulator operand belongs to no component; it buckets under
    // the null root so the reject path below still reports it.
    const Var* root = seed.acc != nullptr ? graph.Find(seed.acc) : nullptr;
    auto& bucket = chain_seeds[root];
    if (bucket.empty()) chain_order.push_back(root);
    bucket.push_back(&seed);
  }

  auto components = graph.Components();
  for (const auto* root : chain_order) {
    const auto& these_seeds = chain_seeds[root];
    static const std::vector<const Var*> kNoMembers;
    auto members_it = root != nullptr ? components.find(root) : components.end();
    const auto& members = members_it == components.end() ? kNoMembers : members_it->second;

    auto verdict = JudgeChain(graph, members, these_seeds);
    if (verdict.plan) {
      const size_t plan_index = map->AddPlan(*verdict.plan);
      for (const auto* member : members) map->Bind(member, plan_index);
      continue;
    }

    // Not column-packable. Row packing is still correct when every page window
    // fits one 16-column L0C block AND nothing forces the accumulator through
    // Vec on the way in; otherwise the shape has no working lowering at all and
    // must be reported here rather than mis-addressed downstream.
    //
    // Two reasons defeat row packing as well, so they reject at any page width:
    // a chain with two allocations (MemoryReuse cannot coalesce the buffers at
    // the merge) and a definition that cannot write Acc at all. Rejecting them
    // here rather than downstream also keeps CanonicalizeTileSlice's guard note
    // honest -- it tells users a batch_matmul_acc never reaches that limit.
    const bool row_packing_still_works =
        !verdict.force_reject && these_seeds.front()->cols <= kAccBoxDim && !verdict.has_batch_producer;
    if (!row_packing_still_works) RejectChain(*these_seeds.front(), verdict);
  }

  return map;
}

}  // namespace rewrite_internal
}  // namespace flatten_tile_nd_to_2d
}  // namespace ir
}  // namespace pypto

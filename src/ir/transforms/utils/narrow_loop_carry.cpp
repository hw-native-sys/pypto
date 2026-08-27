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

/// Loop-carry valid-shape repair. See ``narrow_loop_carry.h`` for the contract and the
/// reasoning behind the scope limits; this file is the mechanism.
///
/// Three linear sweeps, each visiting every node once:
///
///   1. ``ScopeIndex``    -- where every Var is defined and which clock range each loop
///                           spans, so "is this extent visible before the loop?" is an
///                           interval test on the extent's own (tiny) expression rather
///                           than a scan of the loop body.
///   2. ``CarryAnalyzer`` -- decides, per carry, the valid shape to declare. Reads types
///                           only; rewrites nothing.
///   3. ``CarryRewriter`` -- applies the decisions top-down. A loop's seed is re-declared
///                           and its ``IterArg`` re-minted *before* its body is visited,
///                           so one visit types the whole body against the narrowed carry.
///
/// Total cost is O(N log N) in the size of the function: three O(N) traversals over
/// ordered-map lookups. The ordering in (3) is what keeps it there -- deciding after the
/// body was already visited would mean re-typing it a second time, and a nested carry
/// would compound that per level.

#include "pypto/ir/transforms/utils/narrow_loop_carry.h"

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/acc_init_builder.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {
namespace narrow_loop_carry {

namespace {

/// Values of the statement list's terminating ``YieldStmt``, if it has one.
std::vector<ExprPtr> TrailingYieldValues(const StmtPtr& body) {
  auto yield = transform_utils::GetLastYieldStmt(body);
  return yield ? yield->value_ : std::vector<ExprPtr>{};
}

/// The valid extents a carry should be declared at: per axis, the yield's extent when it
/// is adoptable (see the comment on the loop below), else the init's. ``std::nullopt``
/// when no axis narrows.
std::optional<std::vector<ExprPtr>> NarrowedValidShape(const TileTypePtr& init_type,
                                                       const TileTypePtr& yield_type) {
  if (!init_type || !yield_type) return std::nullopt;
  if (init_type->shape_.size() != yield_type->shape_.size()) return std::nullopt;

  const auto init_valid = GetValidShape(init_type);
  const auto yield_valid = GetValidShape(yield_type);
  if (init_valid.size() != yield_valid.size()) return std::nullopt;

  std::vector<ExprPtr> narrowed = init_valid;
  bool any = false;
  for (size_t i = 0; i < init_valid.size(); ++i) {
    if (ProveValidExtentEqual(yield_valid[i], init_valid[i]) == ProofResult::kTrue) continue;
    // A yield extent is adoptable when it is provably no wider than the one declared --
    // or when the declared one is the whole box, because every valid_shape is bounded by
    // its physical shape (`ValidateValidShapeBounds`), so a dynamic yield extent is
    // already trusted to fit. Without the second case an unconstrained runtime row count
    // (`v` rather than `min(v, rows)`) would never be adoptable, which is precisely the
    // shape that reaches a matmul from `pl.slice(..., valid_shape=[v, ...])`.
    const bool init_fills_the_box =
        ProveValidExtentEqual(init_valid[i], init_type->shape_[i]) == ProofResult::kTrue;
    if (!init_fills_the_box &&
        ProveValidExtentLessEqual(yield_valid[i], init_valid[i]) != ProofResult::kTrue) {
      continue;
    }
    narrowed[i] = yield_valid[i];
    any = true;
  }
  return any ? std::optional<std::vector<ExprPtr>>{std::move(narrowed)} : std::nullopt;
}

/// Where each Var is defined, and which half-open clock range each loop spans.
///
/// One pre-order sweep. It answers the only structural question this repair asks -- "is
/// this extent visible at the point the seed is declared?" -- without re-walking the
/// body per carry, which is what would make the pass quadratic on a function with many
/// carries or deeply nested loops.
class ScopeIndex : public IRVisitor {
 public:
  void Build(const StmtPtr& body) { VisitStmt(body); }

  /// The call that defines @p var, or null when it is not an operator result.
  [[nodiscard]] CallPtr DefiningCall(const Var* var) const {
    auto it = defining_call_.find(var);
    return it == defining_call_.end() ? nullptr : it->second;
  }

  /// Whether @p var is defined inside @p loop -- an interval test, O(log N).
  ///
  /// A var this index never saw (a function parameter, or one defined in an outer
  /// function) is by construction not inside the loop.
  [[nodiscard]] bool IsDefinedInside(const Var* var, const Stmt* loop) const {
    auto def = def_clock_.find(var);
    auto span = loop_span_.find(loop);
    if (def == def_clock_.end() || span == loop_span_.end()) return false;
    return def->second >= span->second.first && def->second < span->second.second;
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    Define(op->var_.get());
    if (auto call = As<Call>(op->value_)) defining_call_[op->var_.get()] = call;
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& rv : op->return_vars_) Define(rv.get());
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    for (const auto& rv : op->return_vars_) Define(rv.get());
    EnterLoop(op, [&] {
      Define(op->loop_var_.get());
      IRVisitor::VisitStmt_(op);
    });
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& rv : op->return_vars_) Define(rv.get());
    EnterLoop(op, [&] { IRVisitor::VisitStmt_(op); });
  }

 private:
  void Define(const Var* var) { def_clock_.emplace(var, ++clock_); }

  /// Record the clock range @p loop spans. The loop variable and the carries are
  /// defined *inside* it, so an extent naming one of them is correctly refused.
  template <typename LoopPtr, typename Fn>
  void EnterLoop(const LoopPtr& loop, Fn&& visit_children) {
    const size_t enter = ++clock_;
    for (const auto& iter_arg : loop->iter_args_) Define(static_cast<const Var*>(iter_arg.get()));
    visit_children();
    loop_span_[loop.get()] = {enter, ++clock_};
  }

  size_t clock_ = 0;
  std::map<const Var*, size_t> def_clock_;
  std::map<const Stmt*, std::pair<size_t, size_t>> loop_span_;
  std::map<const Var*, CallPtr> defining_call_;
};

/// The valid shape to declare for a carry, keyed by the ``IterArg`` that carries it.
using CarryDecisions = std::map<const Var*, std::vector<ExprPtr>>;

/// Decides which carries to re-declare. Reads types; rewrites nothing.
class CarryAnalyzer : public IRVisitor {
 public:
  explicit CarryAnalyzer(const ScopeIndex& index) : index_(index) {}

  [[nodiscard]] CarryDecisions Take() { return std::move(decisions_); }

 protected:
  void VisitStmt_(const ForStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    Decide(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    Decide(op);
  }

 private:
  template <typename LoopPtr>
  void Decide(const LoopPtr& loop) {
    const auto& iter_args = loop->iter_args_;
    if (iter_args.empty()) return;
    const auto yields = TrailingYieldValues(loop->body_);
    if (yields.size() != iter_args.size()) return;

    for (size_t i = 0; i < iter_args.size(); ++i) {
      auto init_tile = As<TileType>(iter_args[i]->GetType());
      auto yield_tile = As<TileType>(yields[i]->GetType());
      auto narrowed = NarrowedValidShape(init_tile, yield_tile);
      if (!narrowed) continue;
      // Only an L0C carry is re-declared, and `tile.set_validshape` is 2D.
      if (narrowed->size() != 2) continue;
      if (yield_tile->GetMemorySpace() != MemorySpace::Acc) continue;
      // Nothing to reconcile when both readings of the buffer land on the same pitch --
      // notably a single-fractal-block box, where `ceil(validRow/16)*16` is the physical
      // row count whatever the valid rows are. The same predicate `AccCompactValid` uses,
      // so a carry this declines is also a carry the verifier does not ask about, and a
      // `[16, N]` accumulator keeps the exact form it has today.
      if (AccPitchesCoincide(narrowed->at(0), init_tile->shape_[0])) continue;
      if (!ExtentsAreVisibleBefore(*narrowed, loop)) continue;
      // Only a `tile.create` seed can be re-declared as a narrowed box.
      auto seed = AsVarLike(iter_args[i]->initValue_);
      if (!seed || !IsOp(index_.DefiningCall(seed.get()), "tile.create")) continue;

      decisions_[static_cast<const Var*>(iter_args[i].get())] = std::move(*narrowed);
    }
  }

  /// Whether every var the extents name is defined outside @p loop.
  ///
  /// The re-declared seed sits *before* the loop, so an extent computed inside the body
  /// cannot be named there. `pl.min(M_TILE, t_dim - t0)` written next to the slice it
  /// bounds is the common spelling of exactly that, and hoisting it would leave codegen
  /// with a symbol it cannot bind to a dimension, a scalar parameter, or a loop variable.
  /// The narrowing is declined instead of moving the computation, which would need the
  /// extent to be loop-invariant and is a larger change than this repair.
  template <typename LoopPtr>
  [[nodiscard]] bool ExtentsAreVisibleBefore(const std::vector<ExprPtr>& extents, const LoopPtr& loop) const {
    for (const auto& extent : extents) {
      if (!extent) continue;
      ExtentVarCollector used;
      used.VisitExpr(extent);
      for (const auto* var : used.vars) {
        if (index_.IsDefinedInside(var, loop.get())) return false;
      }
    }
    return true;
  }

  /// The vars an extent expression names. Extents are small arithmetic on scalars, so
  /// this walks a handful of nodes -- unlike a sweep of the loop body it replaces.
  class ExtentVarCollector : public IRVisitor {
   public:
    std::vector<const Var*> vars;

   protected:
    void VisitExpr_(const VarPtr& op) override { vars.push_back(op.get()); }
    void VisitExpr_(const IterArgPtr& op) override { vars.push_back(static_cast<const Var*>(op.get())); }
  };

  const ScopeIndex& index_;
  CarryDecisions decisions_;
};

/// Applies the decisions, and re-types everything downstream of a carry it moved.
///
/// Top-down: a loop's seed is re-declared and its ``IterArg`` re-minted before the body
/// is visited, so a single visit types the body against the narrowed carry. Every rebuilt
/// ``Call`` goes back through ``OpRegistry::Create``, so the operator's own deducer
/// supplies the new result type -- this repair never invents one -- with the original
/// call's attributes carried across. A result whose re-deduced type is unchanged stops the
/// propagation there.
///
/// A ``Submit`` value is substituted like any other expression but never re-deduced: it
/// launches a user function rather than an operator, and it lives in a ``manual_scope`` at
/// orchestration level, where no Acc tile carry can reach it.
class CarryRewriter : public IRMutator {
 public:
  CarryRewriter(const ScopeIndex& index, CarryDecisions decisions)
      : index_(index), decisions_(std::move(decisions)) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = replaced_.find(op.get());
    return it == replaced_.end() ? IRMutator::VisitExpr_(op) : it->second;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto it = replaced_.find(static_cast<const Var*>(op.get()));
    return it == replaced_.end() ? IRMutator::VisitExpr_(op) : it->second;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto rebuilt = IRMutator::VisitStmt_(op);
    auto assign = As<AssignStmt>(rebuilt);
    if (!assign) return rebuilt;
    return BindResult(op, assign, RededuceIfOperandsMoved(op, assign));
  }

  /// The value to bind, re-deduced when this visit moved one of a call's operands.
  ///
  /// Re-deduce ONLY then. A freshly deduced type is not interchangeable with the stored
  /// one -- it carries no MemRef and no resolved memory space -- so re-deducing calls this
  /// pass did not disturb would silently strip both from every tile in the program.
  ExprPtr RededuceIfOperandsMoved(const AssignStmtPtr& original, const AssignStmtPtr& rebuilt) {
    auto call = As<Call>(rebuilt->value_);
    if (!call || !call->op_) return rebuilt->value_;
    if (!OperandsMoved(original->value_, rebuilt->value_)) return rebuilt->value_;
    // A call to a user function carries a GlobalVar, not a registered operator --
    // `OpRegistry::Create` rejects those by name, so screen them out first.
    auto& registry = OpRegistry::GetInstance();
    if (!registry.IsRegistered(call->op_->name_)) return rebuilt->value_;
    auto deduced = registry.Create(call->op_->name_, call->args_, call->kwargs_, call->span_);
    if (!deduced) return rebuilt->value_;
    // Attributes come from the *rebuilt* call, whose contents this visit already remapped.
    return transform_utils::PreserveCallAttrs(call, deduced);
  }

  /// Bind an assignment's result at the type its value now carries.
  ///
  /// Every value shape reaches here, not only an operator call: ``alias = acc`` is a legal
  /// SSA copy, and a mutator that rewrote only its right-hand side would leave the bound
  /// Var at the width the carry used to have -- an asymmetry ``AssignTypeSymmetry`` and the
  /// ``TypeCheck`` diagnostic both reject, and a dead end for the propagation, since every
  /// later use still reads the old type through the alias.
  ///
  /// An assignment this visit did not touch keeps its Var, so a program the repair has no
  /// business in comes through byte-identical.
  StmtPtr BindResult(const AssignStmtPtr& original, const AssignStmtPtr& rebuilt, const ExprPtr& value) {
    if (value.get() == original->value_.get()) return rebuilt;

    const auto& old_type = rebuilt->var_->GetType();
    const auto& new_type = value->GetType();
    const bool retype = new_type && (!old_type || !structural_equal(old_type, new_type));
    if (!retype && value.get() == rebuilt->value_.get()) return rebuilt;

    auto result = MutableCopy(rebuilt);  // MutableCopy so leading comments survive
    result->value_ = value;
    if (retype) {
      auto new_var = std::make_shared<Var>(rebuilt->var_->name_hint_, new_type, rebuilt->var_->span_);
      replaced_[original->var_.get()] = new_var;
      result->var_ = new_var;
    }
    return result;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto rebuilt = IRMutator::VisitStmt_(op);
    auto if_stmt = As<IfStmt>(rebuilt);
    if (!if_stmt || if_stmt->return_vars_.empty()) return rebuilt;

    // A phi is typed from the then branch (ConvertTensorToTileOps and the DSL parser
    // agree on that), so follow it there rather than inventing a merge.
    const auto then_yield = TrailingYieldValues(if_stmt->then_body_);
    if (then_yield.size() != if_stmt->return_vars_.size()) return rebuilt;

    std::vector<VarPtr> new_return_vars;
    new_return_vars.reserve(if_stmt->return_vars_.size());
    bool changed = false;
    for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
      const auto& rv = if_stmt->return_vars_[i];
      const auto& yield_type = then_yield[i]->GetType();
      if (!yield_type || (rv->GetType() && structural_equal(rv->GetType(), yield_type))) {
        new_return_vars.push_back(rv);
        continue;
      }
      auto new_rv = std::make_shared<Var>(rv->name_hint_, yield_type, rv->span_);
      replaced_[op->return_vars_[i].get()] = new_rv;
      new_return_vars.push_back(new_rv);
      changed = true;
    }
    if (!changed) return rebuilt;
    return std::make_shared<IfStmt>(if_stmt->condition_, if_stmt->then_body_, if_stmt->else_body_,
                                    new_return_vars, if_stmt->span_);
  }

  /// Splice a rewritten loop's prologue into the enclosing statement list, so the
  /// re-declared seed is a sibling of the loop rather than a nested ``SeqStmts``.
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> out;
    out.reserve(op->stmts_.size());
    bool changed = false;
    for (const auto& stmt : op->stmts_) {
      auto visited = VisitStmt(stmt);
      if (!visited) {
        changed = true;
        continue;
      }
      auto expanded = As<SeqStmts>(visited);
      if (expanded && !As<SeqStmts>(stmt)) {
        out.insert(out.end(), expanded->stmts_.begin(), expanded->stmts_.end());
        changed = true;
        continue;
      }
      if (visited.get() != stmt.get()) changed = true;
      out.push_back(visited);
    }
    if (!changed) return op;
    return std::make_shared<SeqStmts>(out, op->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return VisitLoop(op); }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return VisitLoop(op); }

 private:
  /// Whether visiting rewrote any operand of this call -- i.e. whether the value is
  /// downstream of a var this pass re-typed.
  static bool OperandsMoved(const ExprPtr& before, const ExprPtr& after) {
    auto old_call = As<Call>(before);
    auto new_call = As<Call>(after);
    if (!old_call || !new_call) return false;
    if (old_call->args_.size() != new_call->args_.size()) return true;
    for (size_t i = 0; i < old_call->args_.size(); ++i) {
      if (old_call->args_[i].get() != new_call->args_[i].get()) return true;
    }
    return false;
  }

  static StmtPtr RebuildLoopLike(const ForStmtPtr& loop, const std::vector<IterArgPtr>& iter_args,
                                 const StmtPtr& body, const std::vector<VarPtr>& return_vars) {
    return loop_repair::RebuildForStmt(loop, iter_args, body, return_vars);
  }
  static StmtPtr RebuildLoopLike(const WhileStmtPtr& loop, const std::vector<IterArgPtr>& iter_args,
                                 const StmtPtr& body, const std::vector<VarPtr>& return_vars) {
    return loop_repair::RebuildWhileStmt(loop, iter_args, body, return_vars);
  }

  /// Settle a loop's carries, then visit its body exactly once.
  ///
  /// Two things can move a carry: this loop has a decision of its own, or its init value
  /// was re-typed by an enclosing rewrite. Both are resolved here, before the body is
  /// visited, so the body's deducers see the final carry type on their first and only
  /// pass. Substituting the init alone would leave the ``IterArg`` -- and everything the
  /// body deduced from it -- at the old type.
  template <typename LoopPtr>
  StmtPtr VisitLoop(const LoopPtr& op) {
    std::vector<StmtPtr> prologue;
    std::vector<IterArgPtr> new_iter_args = op->iter_args_;
    bool carries_moved = false;

    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      const auto& iter_arg = op->iter_args_[i];
      if (!iter_arg->initValue_) continue;
      ExprPtr init = VisitExpr(iter_arg->initValue_);

      auto decision = decisions_.find(static_cast<const Var*>(iter_arg.get()));
      if (decision != decisions_.end()) {
        if (auto seed = BuildNarrowedInit(iter_arg, decision->second, &prologue)) init = seed;
      }
      if (!init || !init->GetType()) continue;
      if (iter_arg->GetType() && structural_equal(iter_arg->GetType(), init->GetType())) continue;

      auto new_iter_arg =
          std::make_shared<IterArg>(iter_arg->name_hint_, init->GetType(), init, iter_arg->span_);
      new_iter_args[i] = new_iter_arg;
      replaced_[static_cast<const Var*>(iter_arg.get())] = new_iter_arg;
      carries_moved = true;
    }

    auto new_body = VisitStmt(op->body_);
    auto new_return_vars = RetypeReturnVars(new_body, op->return_vars_);
    if (!carries_moved && new_body.get() == op->body_.get() && new_return_vars == op->return_vars_ &&
        prologue.empty()) {
      return op;
    }

    auto new_loop = RebuildLoopLike(op, new_iter_args, new_body, new_return_vars);
    if (prologue.empty()) return new_loop;
    prologue.push_back(new_loop);
    return std::make_shared<SeqStmts>(prologue, new_loop->span_);
  }

  /// Re-type a loop's ``return_vars`` from its (re-typed) yields, and record each
  /// replacement so later statements re-deduce through the new type rather than merely
  /// substituting the var.
  std::vector<VarPtr> RetypeReturnVars(const StmtPtr& new_body, const std::vector<VarPtr>& return_vars) {
    const auto new_yields = TrailingYieldValues(new_body);
    std::vector<VarPtr> new_return_vars = return_vars;
    if (new_yields.size() != return_vars.size()) return new_return_vars;
    for (size_t i = 0; i < return_vars.size(); ++i) {
      const auto& rv = return_vars[i];
      const auto& yield_type = new_yields[i]->GetType();
      if (!yield_type || (rv->GetType() && structural_equal(rv->GetType(), yield_type))) continue;
      auto new_rv = std::make_shared<Var>(rv->name_hint_, yield_type, rv->span_);
      new_return_vars[i] = new_rv;
      replaced_[rv.get()] = new_rv;
    }
    return new_return_vars;
  }

  /// Re-declare a ``tile.create`` seed with the narrowed valid shape.
  ///
  /// The seed is rebuilt at its *declaration* rather than aliased through
  /// ``tile.set_validshape`` alone: that op inherits its source's compact mode, which is
  /// right for a tile whose bytes may already be written but would leave a fresh
  /// accumulator advertising the physical row pitch. Declaring the box lets
  /// ``tile.create``'s own deducer derive the Acc layout for the box it declares.
  /// Returns null when the seed is not one this repair can re-declare.
  ExprPtr BuildNarrowedInit(const IterArgPtr& iter_arg, const std::vector<ExprPtr>& valid,
                            std::vector<StmtPtr>* prologue) {
    auto init_tile = As<TileType>(iter_arg->GetType());
    auto seed = AsVarLike(iter_arg->initValue_);
    if (!init_tile || !seed || !IsOp(index_.DefiningCall(seed.get()), "tile.create")) return nullptr;

    // The extents were chosen against the types as the analyzer found them; an enclosing
    // rewrite may since have replaced the vars they name.
    std::vector<ExprPtr> visited_valid;
    visited_valid.reserve(valid.size());
    for (const auto& extent : valid) visited_valid.push_back(VisitExpr(extent));

    auto narrowed = acc_init::BuildNarrowedAccInit(init_tile->shape_, visited_valid, init_tile->dtype_,
                                                   seed->name_hint_ + "_narrowed", iter_arg->span_);
    for (auto& stmt : narrowed.stmts) prologue->push_back(std::move(stmt));
    return narrowed.value;
  }

  const ScopeIndex& index_;
  CarryDecisions decisions_;
  std::map<const Var*, VarPtr> replaced_;
};

}  // namespace

FunctionPtr NarrowAccCarries(const FunctionPtr& func) {
  if (!func || !func->body_) return func;

  ScopeIndex index;
  index.Build(func->body_);

  CarryAnalyzer analyzer(index);
  analyzer.VisitStmt(func->body_);
  auto decisions = analyzer.Take();
  if (decisions.empty()) return func;

  CarryRewriter rewriter(index, std::move(decisions));
  auto new_body = rewriter.VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) return func;
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace narrow_loop_carry
}  // namespace ir
}  // namespace pypto

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

#include "pypto/ir/transforms/utils/narrow_loop_carry.h"

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/acc_init_builder.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {
namespace narrow_loop_carry {

namespace {

/// Values of the statement list's terminating ``YieldStmt``, if it has one.
std::vector<ExprPtr> TrailingYieldValues(const StmtPtr& body) {
  if (auto seq = As<SeqStmts>(body)) {
    if (seq->stmts_.empty()) return {};
    if (auto yield = As<YieldStmt>(seq->stmts_.back())) return yield->value_;
    return {};
  }
  if (auto direct = As<YieldStmt>(body)) return direct->value_;
  return {};
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

/// Whether every var the extents name is defined outside @p body.
///
/// The re-declared seed sits *before* the loop, so an extent computed inside the body
/// cannot be named there. `pl.min(M_TILE, t_dim - t0)` written next to the slice it
/// bounds is the common spelling of exactly that, and hoisting it would leave codegen
/// with a symbol it cannot bind to a dimension, a scalar parameter, or a loop variable.
/// The narrowing is declined instead of moving the computation, which would need the
/// extent to be loop-invariant and is a larger change than this repair.
bool ExtentsAreVisibleBeforeLoop(const std::vector<ExprPtr>& extents, const StmtPtr& body) {
  var_collectors::VarDefUseCollector body_defs;
  body_defs.VisitStmt(body);
  for (const auto& extent : extents) {
    if (!extent) continue;
    var_collectors::VarDefUseCollector extent_vars;
    extent_vars.VisitExpr(extent);
    for (const auto* used : extent_vars.var_uses) {
      if (body_defs.var_defs.count(used) > 0) return false;
    }
  }
  return true;
}

/// Re-type the def-use closure of a set of re-typed vars.
///
/// Every rebuilt ``Call`` goes back through ``OpRegistry::Create``, so the operator's own
/// deducer supplies the new result type — this repair never invents one. A result whose
/// re-deduced type is unchanged stops the propagation there.
///
/// A ``Submit`` value is substituted like any other expression but never re-deduced: it
/// launches a user function rather than an operator, and it lives in a ``manual_scope`` at
/// orchestration level, where no Acc tile carry can reach it.
class RetypeClosureMutator : public IRMutator {
 public:
  RetypeClosureMutator() = default;
  explicit RetypeClosureMutator(std::map<const Var*, VarPtr> seed) : replaced_(std::move(seed)) {}

  /// Register a var whose type moved, so later statements re-deduce through it.
  void RecordReplacement(const Var* old_var, const VarPtr& new_var) { replaced_[old_var] = new_var; }

 protected:
  /// Whether visiting rewrote any operand of this call — i.e. whether the value is
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

    // Re-deduce ONLY when an operand actually moved. A freshly deduced type is not
    // interchangeable with the stored one — it carries no MemRef and no resolved
    // memory space — so re-deducing calls this pass did not disturb would silently
    // strip both from every tile in the program.
    auto call = As<Call>(assign->value_);
    if (!call || !call->op_) return rebuilt;
    if (!OperandsMoved(op->value_, assign->value_)) return rebuilt;
    // A call to a user function carries a GlobalVar, not a registered operator —
    // `OpRegistry::Create` rejects those by name, so screen them out first.
    auto& registry = OpRegistry::GetInstance();
    if (!registry.IsRegistered(call->op_->name_)) return rebuilt;
    auto fresh = registry.Create(call->op_->name_, call->args_, call->kwargs_, call->span_);
    if (!fresh) return rebuilt;

    const auto& old_type = assign->var_->GetType();
    const auto& new_type = fresh->GetType();
    if (old_type && new_type && structural_equal(old_type, new_type)) return rebuilt;

    auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_type, assign->var_->span_);
    replaced_[op->var_.get()] = new_var;
    return std::make_shared<AssignStmt>(new_var, fresh, assign->span_);
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

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return VisitLoop(op); }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return VisitLoop(op); }

  /// Visit a loop's children, then carry any re-typed value across its boundary.
  template <typename LoopPtr>
  StmtPtr VisitLoop(const LoopPtr& op) {
    auto rebuilt = RebuildWithVisitedChildren(op);
    auto loop = As<std::remove_const_t<typename LoopPtr::element_type>>(rebuilt);
    if (!loop) return rebuilt;
    return PropagateIntoLoop(loop, op->iter_args_, op->return_vars_);
  }

  /// Rebuild a loop of either kind with new carry state.
  static StmtPtr RebuildLoopLike(const ForStmtPtr& loop, const std::vector<IterArgPtr>& iter_args,
                                 const StmtPtr& body, const std::vector<VarPtr>& return_vars) {
    return loop_repair::RebuildForStmt(loop, iter_args, body, return_vars);
  }
  static StmtPtr RebuildLoopLike(const WhileStmtPtr& loop, const std::vector<IterArgPtr>& iter_args,
                                 const StmtPtr& body, const std::vector<VarPtr>& return_vars) {
    return loop_repair::RebuildWhileStmt(loop, iter_args, body, return_vars);
  }

  /// Visit a loop's children and rebuild it, without either level's carry handling -- the
  /// caller applies that itself once the body is in its final shape. A named helper rather
  /// than a qualified ``IRMutator::VisitStmt_`` call, which would read as skipping this
  /// class's own override.
  StmtPtr RebuildWithVisitedChildren(const ForStmtPtr& op) { return IRMutator::VisitStmt_(op); }
  StmtPtr RebuildWithVisitedChildren(const WhileStmtPtr& op) { return IRMutator::VisitStmt_(op); }

  /// Carry a re-typed value across a nested loop boundary.
  ///
  /// Substituting the init value alone leaves the nested ``IterArg`` — and therefore
  /// everything the nested body deduced from it — at the old type. Re-mint the iter_arg
  /// from its new init and re-run the closure through the nested body, exactly as the
  /// outer narrowing does; the loop's own results then follow its re-typed yields.
  template <typename LoopPtr>
  StmtPtr PropagateIntoLoop(const LoopPtr& loop, const std::vector<IterArgPtr>& original_iter_args,
                            const std::vector<VarPtr>& original_return_vars) {
    const auto& iter_args = loop->iter_args_;
    const auto& body = loop->body_;
    if (iter_args.size() != original_iter_args.size()) {
      return loop;
    }

    std::map<const Var*, VarPtr> carry_seed;
    std::vector<IterArgPtr> new_iter_args = iter_args;
    for (size_t i = 0; i < iter_args.size(); ++i) {
      const auto& iter_arg = iter_args[i];
      const auto& init = iter_arg->initValue_;
      if (!init || !init->GetType()) continue;
      if (iter_arg->GetType() && structural_equal(iter_arg->GetType(), init->GetType())) continue;
      auto new_iter_arg =
          std::make_shared<IterArg>(iter_arg->name_hint_, init->GetType(), init, iter_arg->span_);
      new_iter_args[i] = new_iter_arg;
      carry_seed[static_cast<const Var*>(original_iter_args[i].get())] = new_iter_arg;
    }
    if (carry_seed.empty()) return loop;

    RetypeClosureMutator retyper(std::move(carry_seed));
    auto new_body = retyper.VisitStmt(body);
    auto new_return_vars = RetypeReturnVars(new_body, loop->return_vars_, original_return_vars, this);
    return RebuildLoopLike(loop, new_iter_args, new_body, new_return_vars);
  }

  /// Re-type a loop's ``return_vars`` from its (re-typed) yields, and publish each
  /// replacement to @p publish_to so later statements re-deduce through the new type
  /// rather than merely substituting the var.
  static std::vector<VarPtr> RetypeReturnVars(const StmtPtr& new_body, const std::vector<VarPtr>& return_vars,
                                              const std::vector<VarPtr>& original_return_vars,
                                              RetypeClosureMutator* publish_to) {
    const auto new_yields = TrailingYieldValues(new_body);
    std::vector<VarPtr> new_return_vars = return_vars;
    if (new_yields.size() != return_vars.size() || original_return_vars.size() != return_vars.size()) {
      return new_return_vars;
    }
    for (size_t i = 0; i < return_vars.size(); ++i) {
      const auto& rv = return_vars[i];
      const auto& yield_type = new_yields[i]->GetType();
      if (!yield_type || (rv->GetType() && structural_equal(rv->GetType(), yield_type))) continue;
      auto new_rv = std::make_shared<Var>(rv->name_hint_, yield_type, rv->span_);
      new_return_vars[i] = new_rv;
      publish_to->RecordReplacement(original_return_vars[i].get(), new_rv);
    }
    return new_return_vars;
  }

 private:
  std::map<const Var*, VarPtr> replaced_;
};

class NarrowLoopCarryMutator : public RetypeClosureMutator {
 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // A Var does not carry its defining expression, so remember it: the seed of a carry
    // is a Var whose definition is the `tile.create` this pass re-declares.
    if (auto call = As<Call>(op->value_)) defs_[op->var_.get()] = call;
    return RetypeClosureMutator::VisitStmt_(op);
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

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return VisitLoopAndNarrow(op); }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return VisitLoopAndNarrow(op); }

 private:
  /// Visit a loop's children, then re-declare every Acc carry the body proves narrower.
  template <typename LoopPtr>
  StmtPtr VisitLoopAndNarrow(const LoopPtr& op) {
    // Inner loops first: a nested carry may itself narrow this loop's yields.
    auto rebuilt = RebuildWithVisitedChildren(op);
    auto loop = As<std::remove_const_t<typename LoopPtr::element_type>>(rebuilt);
    if (!loop) return rebuilt;
    return NarrowCarries(loop, op->iter_args_, op->return_vars_);
  }

  template <typename LoopPtr>
  StmtPtr NarrowCarries(const LoopPtr& loop, const std::vector<IterArgPtr>& original_iter_args,
                        const std::vector<VarPtr>& original_return_vars) {
    const auto& iter_args = loop->iter_args_;
    const auto& body = loop->body_;
    const auto& return_vars = loop->return_vars_;
    if (iter_args.empty() || iter_args.size() != original_iter_args.size()) {
      return PropagateIntoLoop(loop, original_iter_args, original_return_vars);
    }

    const auto yields = TrailingYieldValues(body);
    if (yields.size() != iter_args.size()) {
      return PropagateIntoLoop(loop, original_iter_args, original_return_vars);
    }

    std::vector<StmtPtr> prologue;
    std::map<const Var*, VarPtr> carry_seed;
    std::vector<IterArgPtr> new_iter_args = iter_args;
    for (size_t i = 0; i < iter_args.size(); ++i) {
      const auto& iter_arg = iter_args[i];
      auto init_tile = As<TileType>(iter_arg->GetType());
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
      if (!ExtentsAreVisibleBeforeLoop(*narrowed, body)) continue;

      auto narrowed_init = BuildNarrowedInit(iter_arg->initValue_, init_tile, *narrowed, &prologue);
      if (!narrowed_init) continue;

      auto new_iter_arg = std::make_shared<IterArg>(iter_arg->name_hint_, narrowed_init->GetType(),
                                                    narrowed_init, iter_arg->span_);
      new_iter_args[i] = new_iter_arg;
      carry_seed[static_cast<const Var*>(original_iter_args[i].get())] = new_iter_arg;
    }
    if (carry_seed.empty()) {
      return PropagateIntoLoop(loop, original_iter_args, original_return_vars);
    }

    RetypeClosureMutator retyper(std::move(carry_seed));
    auto new_body = retyper.VisitStmt(body);
    auto new_return_vars = RetypeReturnVars(new_body, return_vars, original_return_vars, this);
    auto new_loop = RebuildLoopLike(loop, new_iter_args, new_body, new_return_vars);
    if (prologue.empty()) return new_loop;
    prologue.push_back(new_loop);
    return std::make_shared<SeqStmts>(prologue, new_loop->span_);
  }

  /// Re-declare a ``tile.create`` seed with the narrowed valid shape.
  ///
  /// The seed is rebuilt at its *declaration* rather than aliased through
  /// ``tile.set_validshape`` alone: that op inherits its source's compact mode, which is
  /// right for a tile whose bytes may already be written but would leave a fresh
  /// accumulator advertising the physical row pitch. Declaring the box lets
  /// ``tile.create``'s own deducer derive the Acc layout for the box it declares.
  /// Returns null when the init is not a ``tile.create`` this pass can re-declare.
  ExprPtr BuildNarrowedInit(const ExprPtr& init, const TileTypePtr& init_tile,
                            const std::vector<ExprPtr>& valid, std::vector<StmtPtr>* prologue) const {
    auto init_var = AsVarLike(init);
    if (!init_var) return nullptr;
    auto def = defs_.find(init_var.get());
    if (def == defs_.end()) return nullptr;
    if (!IsOp(def->second, "tile.create")) return nullptr;

    auto narrowed = acc_init::BuildNarrowedAccInit(init_tile->shape_, valid, init_tile->dtype_,
                                                   init_var->name_hint_ + "_narrowed", init->span_);
    for (auto& stmt : narrowed.stmts) prologue->push_back(std::move(stmt));
    return narrowed.value;
  }

  /// Defining call of each assigned Var, so a carry's seed can be traced back to the
  /// ``tile.create`` that declared it.
  std::map<const Var*, CallPtr> defs_;
};

}  // namespace

FunctionPtr NarrowAccCarries(const FunctionPtr& func) {
  if (!func || !func->body_) return func;
  NarrowLoopCarryMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) return func;
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace narrow_loop_carry
}  // namespace ir
}  // namespace pypto

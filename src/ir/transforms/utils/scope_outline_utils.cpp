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

#include "pypto/ir/transforms/utils/scope_outline_utils.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/deferred_wait_contract.h"
#include "pypto/ir/transforms/utils/result_alias_utils.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace outline_utils {

std::vector<CallWriteTarget> CallWriteTargets(const CallPtr& call) {
  std::vector<CallWriteTarget> targets;
  if (!call) return targets;
  const auto* entry = LookupOpEntry(call->op_);
  if (!entry || !entry->WritesAnyArg()) return targets;
  for (size_t i = 0; i < call->args_.size(); ++i) {
    auto effect = entry->GetArgEffect(i, call->kwargs_);
    if (!ArgEffectWrites(effect)) continue;
    if (auto var = AsVarLike(call->args_[i])) {
      targets.push_back(CallWriteTarget{var, i, effect});
    }
  }
  return targets;
}

namespace {

/**
 * @brief Visitor to collect target tensors of tile.store calls (by pointer identity).
 *
 * These tensors are modified via side-effect inside scopes but are not
 * captured by VarDefUseCollector since they are defined externally.  The third
 * argument of store is the output tensor.
 *
 * Deliberately narrower than `CallWriteTargets`, which answers "what does this
 * call write" for the direction analysis. This set drives the *export*
 * machinery — a store target becomes an extra outlined-function output, and
 * `StoreEvalToAssignMutator` binds a result Var for it — and that mechanism
 * exists for a write whose result the body does not already thread. An SSA-pure
 * writer such as `tensor.assemble` returns the updated tensor and the caller
 * binds it, so exporting it too would add a redundant output.
 */
class StoreTargetCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> store_targets;

 protected:
  void VisitExpr_(const CallPtr& op) override {
    auto opnode = std::dynamic_pointer_cast<const Op>(op->op_);
    if (opnode && IsOp(opnode, "tile.store") && op->args_.size() >= 3) {
      if (auto var = As<Var>(op->args_[2])) {
        store_targets.insert(var.get());
      }
    }
    IRVisitor::VisitExpr_(op);
  }
};

/// Combine two direction observations about one parameter, keeping every
/// access either one witnessed: ``In < Out < InOut``.
///
/// Evidence about a parameter arrives from several independent places
/// (``ParamReadCollector``, the store-target set, the assemble scan), and each
/// one is a *lower* bound — none of them can prove the absence of an access the
/// others saw. Merging rather than assigning is what keeps a later observation
/// from erasing an earlier one.
///
/// The inner callees' evidence does **not** arrive through this ordering.
/// ``In`` is the seeded *no evidence yet* floor, so it cannot also stand for
/// "somebody read this" — folding a callee's ``In`` slot against another's
/// ``Out`` slot along the ranks above would yield ``Out`` and lose the read.
/// ``InferParamDirections`` accumulates the callee slots as two independent
/// flags and calls this once, with the verdict already formed.
[[nodiscard]] inline ParamDirection MergeParamDirection(ParamDirection lhs, ParamDirection rhs) {
  auto rank = [](ParamDirection d) {
    switch (d) {
      case ParamDirection::In:
        return 0;
      case ParamDirection::Out:
        return 1;
      case ParamDirection::InOut:
        return 2;
    }
    return 0;
  };
  if (lhs == rhs) return lhs;
  // Out and InOut both write; a read on either side makes the result InOut.
  if ((lhs == ParamDirection::Out && rhs == ParamDirection::InOut) ||
      (lhs == ParamDirection::InOut && rhs == ParamDirection::Out)) {
    return ParamDirection::InOut;
  }
  return rank(lhs) >= rank(rhs) ? lhs : rhs;
}

/**
 * @brief Marks which captured variables a scope body *reads*.
 *
 * A "read" is any use of the variable other than an operand the callee purely
 * overwrites. Two declarations say so, one per kind of callee: a builtin's
 * ``ArgEffect::Write`` slot (``DestinationSlots``, read straight off the
 * argument-effect registry rather than from a hand-kept list of op names) and a
 * user function's ``ParamDirection::Out`` parameter (``CalleeWriteOnlySlots``). Such
 * an operand replaces a sub-region of the destination *in place*: the untouched
 * region is neither loaded nor re-stored, so passing the variable there moves
 * no data into the scope and is not a read. ``tile.store(tile, offsets,
 * target)`` and ``tensor.assemble(dst, src, offsets)`` are the familiar cases,
 * but nothing here names them.
 *
 * The registry already draws the distinction this analysis needs: an *atomic*
 * store or assemble (``atomic=pl.AtomicType.Add``) declares ``ReadWrite``
 * rather than ``Write``, because it accumulates into the destination, so that
 * operand stays on the read path without being special-cased. Every other use
 * is a read, including a use inside the same call's remaining operands
 * (``pl.assemble(dst, pl.slice(dst, ...))`` really does read ``dst``).
 *
 * Definition sites are not reads either, so the stmt hooks below skip the
 * binding fields (an ``AssignStmt`` LHS, a loop's ``loop_var_`` / ``iter_args_``
 * / ``return_vars_``) before recursion reaches ``VisitVarLike_``. This matters
 * for pre-SSA input, where ``c = pl.store(t, off, c)`` binds the *same* Var it
 * stores into: counting that LHS as a read would classify ``c`` ``InOut`` here
 * and ``Out`` on the identical program after ``ConvertToSSA`` renames the LHS.
 *
 * Under SSA the destination's post-write state is bound to a *fresh* Var, and
 * reading that alias reads the same backing buffer — ``v1 = tile.store(t, off,
 * dst); x = tile.load(v1, <a region the scope never wrote>)`` does need ``dst``'s
 * incoming contents. So an ``AssignStmt`` whose value writes a tracked
 * destination registers its LHS as another name for that destination (the same
 * aliasing ``PostStoreAliasCollector`` records for export bookkeeping), and a
 * later read of the alias marks the destination read. Chained writes stay
 * write-only: the alias then appears only in the next write's destination slot,
 * which is skipped.
 *
 * Unrecognised uses count as reads, so ``InferParamDirections`` can only err
 * towards ``InOut``.
 */
class ParamReadCollector : public IRVisitor {
 public:
  /// @param program Resolves a ``GlobalVar`` callee so an argument the callee
  ///                 declares ``Out`` can be skipped, exactly as a builtin's
  ///                 declared ``Write`` slot is. May be null, in which case
  ///                 every call argument stays on the read path — the
  ///                 conservative answer, since an unresolvable callee could
  ///                 read anything.
  ParamReadCollector(const std::unordered_map<const Var*, size_t>& var_to_idx, std::vector<bool>& has_read,
                     ProgramPtr program)
      : aliases_(var_to_idx), has_read_(has_read), program_(std::move(program)) {}

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    auto it = aliases_.find(op.get());
    if (it != aliases_.end()) has_read_[it->second] = true;
    IRVisitor::VisitVarLike_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // ``var_`` is the binding, not a use — but when the value writes a tracked
    // destination in place, that binding is another name for the destination,
    // so register it before descending. Registering first is safe: the value's
    // own destination slot is skipped below, so the statement cannot read
    // itself into ``InOut``.
    if (op->var_) {
      if (auto dst = WrittenDestination(op->value_)) {
        auto it = aliases_.find(dst.get());
        if (it != aliases_.end()) aliases_.emplace(op->var_.get(), it->second);
      }
    }
    VisitExpr(op->value_);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    // ``loop_var_``, ``iter_args_`` and ``return_vars_`` are definitions at the
    // loop header; only the bounds and the iter-arg inits are reads.
    VisitExpr(op->start_);
    VisitExpr(op->stop_);
    VisitExpr(op->step_);
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) VisitExpr(iter_arg->initValue_);
    }
    VisitStmt(op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    VisitExpr(op->condition_);
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) VisitExpr(iter_arg->initValue_);
    }
    VisitStmt(op->body_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    VisitExpr(op->condition_);
    VisitStmt(op->then_body_);
    // Bound to a local so the optional's engagement is provable at the use:
    // through `op->`, the analysis cannot tie the guard to the access.
    const auto& else_body = op->else_body_;
    if (else_body.has_value()) VisitStmt(else_body.value());
  }

  /// The operand indices ``op`` overwrites in place. Such an operand replaces a
  /// sub-region of the destination without moving data into the scope, which is
  /// what makes the slot a non-read.
  ///
  /// Read from the operator's registry declaration, which already draws the
  /// distinction this analysis needs: an *atomic* store or assemble declares
  /// ``ReadWrite`` rather than ``Write``, because ``out += x`` reads the value
  /// already there. Such an operand stays on the normal read path, which keeps
  /// an accumulator ``InOut`` — and therefore staged, so it starts from the
  /// caller's zeros rather than allocator garbage. The same rule now covers
  /// every declared writer, so an ``AtomicAdd`` notify keeps its signal
  /// ``InOut`` while a ``Set`` notify does not, without either being named here.
  [[nodiscard]] static std::set<size_t> DestinationSlots(const CallPtr& op) {
    std::set<size_t> slots;
    for (const auto& target : CallWriteTargets(op)) {
      if (target.effect == ArgEffect::Write) slots.insert(target.slot);
    }
    return slots;
  }

  /// The argument indices a call to a *user function* purely overwrites.
  ///
  /// A callee parameter declared ``Out`` is the user-function counterpart of a
  /// builtin's ``ArgEffect::Write`` slot: the callee replaces the buffer's
  /// contents without consulting them, so handing a capture to that slot moves
  /// no data into this scope and is not a read. Without this, *every* argument
  /// of *every* inner call counted as a read, and the later merge had to ignore
  /// the resulting ``has_read`` to keep write-only captures ``Out`` — which is
  /// how a genuine body read next to a write-only call slot got lost.
  ///
  /// Both call-like kinds are covered. ``Call`` maps ``args_[i]`` to
  /// ``params_[i]`` with full coverage; ``Submit`` maps the same way over a
  /// *prefix*, with ``args_.size() <= params_.size()`` — the omitted tail is
  /// runtime-allocated and never appears as an argument here. The trailing
  /// ``CommCtx`` params that would break that identity are materialised by
  /// pass 43, long after any outliner runs, so the prefix mapping is exact at
  /// this point (`.claude/rules/pass-submit-awareness.md`).
  ///
  /// Anything that fails those constraints — no program to resolve the callee,
  /// a non-``GlobalVar`` callee, or a size that violates the coverage bound —
  /// yields no skips, leaving every argument on the read path. That
  /// over-approximates reads, which is the safe direction.
  [[nodiscard]] std::set<size_t> CalleeWriteOnlySlots(const OpPtr& callee_op, size_t arg_count,
                                                      bool is_submit) const {
    std::set<size_t> slots;
    if (!program_) return slots;
    auto gv = std::dynamic_pointer_cast<const GlobalVar>(callee_op);
    if (!gv) return slots;
    auto callee = program_->GetFunction(gv->name_);
    if (!callee) return slots;
    const auto& dirs = callee->param_directions_;
    const bool covered = is_submit ? (arg_count <= dirs.size()) : (arg_count == dirs.size());
    if (!covered) return slots;
    for (size_t i = 0; i < arg_count; ++i) {
      if (dirs[i] == ParamDirection::Out) slots.insert(i);
    }
    return slots;
  }

  /// The tensor ``value`` names on the way out, or null when it names none.
  ///
  /// Only the operator's declared result-alias contract answers, which is the
  /// same one ``ConvertTensorToTileOps`` reads. Writing an argument does not
  /// make the result name it: ``tile.mgather`` stages Mat *elem* gathers
  /// through a GM ``scratch`` operand and returns a **fresh** tile, so treating
  /// its lone write slot as the alias would register the gathered tile as
  /// another name for ``scratch``. Reading the tile would then mark a
  /// write-only ``scratch`` read and promote it to ``InOut`` — the false read
  /// that turns disjoint per-rank slices into a cross-rank dependency
  /// (issue #2415). An operator whose result really does name a destination
  /// says so in the contract.
  [[nodiscard]] static VarPtr WrittenDestination(const ExprPtr& value) {
    auto call = As<Call>(value);
    if (!call) return nullptr;
    auto index = ResultAliasedArgIndex(call);
    if (!index) return nullptr;
    return AsVarLike(call->args_[*index]);
  }

  void VisitExpr_(const CallPtr& op) override {
    // The operand indices this call purely overwrites; every other operand is
    // walked as a read.
    const auto dst_slots = DestinationSlots(op);
    const auto callee_out_slots = CalleeWriteOnlySlots(op->op_, op->args_.size(), /*is_submit=*/false);
    for (size_t i = 0; i < op->args_.size(); ++i) {
      // Skipping the whole operand, not just a bare Var in it, is deliberate:
      // the destination slot only ever holds the tensor being written, so
      // anything nested there is address computation, not a content read.
      if (dst_slots.count(i) > 0 || callee_out_slots.count(i) > 0) continue;
      INTERNAL_CHECK_SPAN(op->args_[i], op->span_) << "Call has null argument at index " << i;
      VisitExpr(op->args_[i]);
    }
    // Reference-typed attrs name Vars used elsewhere; mirror the base visitor so
    // a var reachable only through an attr is still counted as read — except
    // for the bookkeeping keys, which name a tensor without accessing it.
    for (const auto& [key, value] : op->attrs_) {
      if (!ShouldVisitScopeAttr(key)) continue;
      ForEachAttrExpr(value, [this](const ExprPtr& e) { VisitExpr(e); });
    }
  }

  /// A task launch reads its capture exactly as a plain call does, and the base
  /// visitor's ``Submit`` handler does not forward to the ``Call`` one (see
  /// `.claude/rules/pass-submit-awareness.md`). Without this override every
  /// ``pl.submit`` argument counted as a read, so a capture handed only to a
  /// callee's ``Out`` slot came back ``InOut`` — the false cross-rank
  /// dependency of issue #2415, for exactly the ``manual_scope`` programs that
  /// launch work asynchronously.
  void VisitExpr_(const SubmitPtr& op) override {
    const auto callee_out_slots = CalleeWriteOnlySlots(op->op_, op->args_.size(), /*is_submit=*/true);
    for (size_t i = 0; i < op->args_.size(); ++i) {
      if (callee_out_slots.count(i) > 0) continue;
      INTERNAL_CHECK_SPAN(op->args_[i], op->span_) << "Submit has null argument at index " << i;
      VisitExpr(op->args_[i]);
    }
    // ``deps_`` are TaskId values this launch consumes — real SSA uses, never
    // a write destination, so they are always read.
    for (const auto& dep : op->deps_) {
      if (dep) VisitExpr(dep);
    }
    for (const auto& [key, value] : op->attrs_) {
      if (!ShouldVisitScopeAttr(key)) continue;
      ForEachAttrExpr(value, [this](const ExprPtr& e) { VisitExpr(e); });
    }
  }

  /// ``dump_vars`` marks a tensor for post-hoc dumping and
  /// ``arg_direction_overrides_vars`` marks a slot as ``NoDep``; both name a
  /// tensor as *bookkeeping* rather than accessing its contents, so neither is
  /// a read. Counting them would promote a write-only capture to ``InOut`` and
  /// re-create the false cross-rank dependency this analysis exists to avoid
  /// (issue #2415) — for the very programs that opted their slots out of
  /// dependency tracking. Every other attr stays a read: this walk dispatches
  /// on the stored type, so an attr nobody thought to enumerate is still
  /// treated conservatively.
  ///
  /// Overriding the shared hook covers both surfaces at once — the base
  /// visitor's ``VisitScopeAttrs`` consults it for a ``ScopeStmt``, and the
  /// ``Call`` walk above consults it for a call.
  [[nodiscard]] bool ShouldVisitScopeAttr(const std::string& key) const override {
    return key != kAttrDumpVars && key != kAttrArgDirOverrideVars;
  }

 private:
  /// Every name for a tracked destination: the captured Vars the caller seeded,
  /// plus each SSA binding of a post-write state discovered along the way.
  std::unordered_map<const Var*, size_t> aliases_;
  std::vector<bool>& has_read_;
  ProgramPtr program_;
};

/**
 * @brief Collect SSA post-store aliases: variables bound via
 *        ``AssignStmt(v, Call(tile.store, ..., target))``.
 *
 * Maps ``v`` (the SSA-assignee) to ``target`` (the store's last argument).
 * Both identify the same tensor state (post-store) with different pointer
 * identities — ``v`` lives in ``body_collector.var_defs`` while ``target``
 * shows up as a ``store_targets`` entry.  ScopeOutliner uses this to avoid
 * exporting the same tensor twice when both sets would otherwise contribute
 * outputs.
 */
class PostStoreAliasCollector : public IRVisitor {
 public:
  /// alias var (the SSA assignee of a tile.store call) → original store target
  std::unordered_map<const Var*, const Var*> alias_to_target;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    auto call = std::dynamic_pointer_cast<const Call>(op->value_);
    auto opnode = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
    if (opnode && IsOp(opnode, "tile.store") && call->args_.size() >= 3) {
      if (auto target = As<Var>(call->args_[2])) {
        alias_to_target.emplace(op->var_.get(), target.get());
      }
    }
    IRVisitor::VisitStmt_(op);
  }
};

/**
 * @brief Summarize the SplitMode of the nested SplitAivScopeStmt regions.
 *
 * The explicit ``pl.split_aiv`` form is a first-class node in the InCore body,
 * not a scope attr (the old ``MarkCurrentScopeSplitAiv`` marker was removed).
 * OutlineIncoreScopes uses this to bridge the node(s) into the function-level
 * ``split_aiv`` marker the downstream contract (passes 11-24) expects.
 *
 * ``found`` is true when the body carries at least one region. ``uniform_mode``
 * is set ONLY when every region shares ONE mode — that single mode is a valid
 * function-level representative ``split``. When sibling regions carry DIFFERING
 * modes there is no representative; ``uniform_mode`` is reset to ``nullopt`` and
 * the outliner stamps ``split_aiv=true`` WITHOUT a function-level ``split`` mode.
 * The authoritative per-region mode is always ``node->split_``, consumed at
 * LowerAutoVectorSplit (20); downstream readers of the function-level mode
 * (ExpandMixedKernel, SplitVectorKernel, MemoryReuse) tolerate the unset mode by
 * keying on the ``split_aiv`` marker / per-op split.
 */
class SplitAivModeSummaryFinder : public IRVisitor {
 public:
  bool found = false;                     ///< at least one SplitAivScopeStmt region
  std::optional<SplitMode> uniform_mode;  ///< set iff ALL regions share one mode

 protected:
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override {
    if (!found) {
      found = true;
      uniform_mode = op->split_;
    } else if (uniform_mode.has_value() && uniform_mode.value() != op->split_) {
      uniform_mode.reset();  // differing sibling modes -> no representative mode
    }
    // No need to descend into the region body for the mode summary.
  }
};

/**
 * @brief Mutator that converts EvalStmt(Call(tile.store, ...)) into
 *        AssignStmt(target_var, Call(tile.store, ...)) for specified
 *        store targets.
 *
 * tile.store returns the output tensor (same type as the 3rd argument).  When
 * the original IR uses EvalStmt (discarding the return value), this mutator
 * re-writes it as an AssignStmt so the return value is captured and can be
 * referenced in a subsequent ReturnStmt.
 *
 * Three input shapes are handled, all producing a single-assignment result:
 *
 * | Input                                    | Output                                     |
 * | ---------------------------------------- | ------------------------------------------ |
 * | ``EvalStmt(store(.., tgt))``             | ``AssignStmt(ret, store(.., tgt))``        |
 * | ``AssignStmt(v, store(.., tgt))``, v≠tgt | keep, plus ``AssignStmt(ret, v)``          |
 * | ``AssignStmt(tgt, store(.., tgt))``      | ``AssignStmt(ret, store(.., tgt))``        |
 *
 * The third row is the read-modify-write shape the parser emits for a captured
 * ``pl.Out`` tensor before ConvertToSSA splits it (``c = pl.store(t, off, c)``).
 * Keeping the original assignment there would rebind the outlined function's
 * own parameter; binding the store result to ``ret`` instead leaves the body
 * single-assignment. Later reads of the pre-store name still resolve to the
 * parameter, which is correct: ``tile.store`` writes through its target in
 * place, so both names denote the same buffer (the same reason OutlineScope
 * returns the *param* rather than the store result — see #1702).
 *
 * When one target is stored to more than once, only the first store binds the
 * declared ``ret`` Var; each subsequent one gets a fresh Var so the result
 * never contains a duplicate definition.
 */
class StoreEvalToAssignMutator : public IRMutator {
 public:
  /// @param target_vars  store target (body pointer) -> declared result Var
  /// @param used_names   names already claimed in the outlined function, used
  ///                     to mint collision-free Vars for repeated stores
  StoreEvalToAssignMutator(const std::unordered_map<const Var*, VarPtr>& target_vars,
                           std::unordered_set<std::string> used_names)
      : target_vars_(target_vars), used_names_(std::move(used_names)) {}

 protected:
  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto store = MatchTrackedStore(op->expr_);
    if (!store) return op;
    return std::make_shared<AssignStmt>(NextResultVar(*store), store->call, op->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Handle SSA-converted stores: AssignStmt(buf_1, Call(tile.store, ..., buf_0))
    // The _store_ret var needs to be assigned the store result too.
    auto store = MatchTrackedStore(op->value_);
    if (!store) return IRMutator::VisitStmt_(op);
    auto result_var = NextResultVar(*store);
    if (op->var_.get() == store->target) {
      // Read-modify-write under the target's own name: replace the assignment
      // rather than appending to it, so the outlined body never rebinds the
      // parameter this target becomes.
      return std::make_shared<AssignStmt>(result_var, store->call, op->span_);
    }
    // Keep original assignment (buf_1 = store(...)) and add _store_ret = buf_1
    auto store_ret_assign = std::make_shared<AssignStmt>(result_var, op->var_, op->span_);
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{op, store_ret_assign}, op->span_);
  }

 private:
  /// One ``tile.store`` whose target this mutator was asked to capture.
  struct TrackedStore {
    CallPtr call;            ///< the store call itself
    const Var* target;       ///< store target (un-substituted body pointer)
    VarPtr declared_result;  ///< the result Var declared for `target`
  };

  /// Match `value` against ``Call(tile.store, ..., target)`` for a tracked
  /// `target`; nullopt for anything else. Both statement forms funnel through
  /// here so they cannot drift apart on what counts as a store.
  [[nodiscard]] std::optional<TrackedStore> MatchTrackedStore(const ExprPtr& value) const {
    auto call = std::dynamic_pointer_cast<const Call>(value);
    if (!IsOp(call, "tile.store") || call->args_.size() < 3) return std::nullopt;
    auto target = As<Var>(call->args_[2]);
    if (!target) return std::nullopt;
    auto it = target_vars_.find(target.get());
    if (it == target_vars_.end()) return std::nullopt;
    return TrackedStore{call, target.get(), it->second};
  }

  /// Result Var for one store: the declared Var for the target's first store,
  /// a fresh same-typed Var for every later one.
  VarPtr NextResultVar(const TrackedStore& store) {
    const VarPtr& declared = store.declared_result;
    if (claimed_.insert(store.target).second) {
      used_names_.insert(declared->name_hint_);
      return declared;
    }
    std::string name = auto_name::GenerateFreshNameLike(declared->name_hint_, used_names_);
    used_names_.insert(name);
    return std::make_shared<Var>(name, declared->GetType(), declared->span_);
  }

  std::unordered_map<const Var*, VarPtr> target_vars_;
  std::unordered_set<std::string> used_names_;
  std::unordered_set<const Var*> claimed_;
};

/**
 * @brief Collect the *upward-exposed* uses of a subtree — variables read
 *        before the subtree (re)defines them.
 *
 * This is the scope's live-in set: every such variable's incoming value is
 * produced outside the scope, so it must become a parameter of the outlined
 * function.
 *
 * It is strictly more precise than ``var_uses \ var_defs``, which is
 * flow-insensitive: a tensor that is read *and then rebound under the same
 * name* — the read-modify-write ``c = pl.store(t, off, c)`` the parser emits
 * for a captured ``pl.Out`` param before ConvertToSSA splits it — lands in
 * both sets, so the difference silently drops it from the parameter list and
 * leaves the read dangling in the outlined body.
 *
 * On SSA input the two agree exactly: no Var is defined twice, so a variable
 * defined inside the body can never also be read ahead of that definition.
 * The traversal order deliberately mirrors ``VarDefUseCollector`` so the
 * derived parameter order is unchanged; only *when* a definition is recorded
 * differs (after the assigned value is visited, matching ``SSAVerifier``).
 */
class UpwardExposedUseCollector : public IRVisitor {
 public:
  /// Live-in variables in first-read order. Deliberately the only iterable
  /// view: the membership set behind ``IsExternal`` is unordered, and the
  /// derived parameter list must stay deterministic.
  std::vector<const Var*> ordered;

  /// True when `var`'s value on entry to the visited subtree comes from
  /// outside it — i.e. the subtree captures rather than defines it.
  [[nodiscard]] bool IsExternal(const Var* var) const { return live_in_.count(var) > 0; }

 protected:
  // Var and IterArg are handled by separate overrides rather than a shared
  // ``VisitVarLike_``, mirroring VarDefUseCollector: neither may descend to the
  // base visitor, because ``IRVisitor::VisitExpr_(IterArgPtr)`` walks the
  // IterArg's ``initValue_``. That init value belongs to the *enclosing* loop
  // header, not to whatever reads the IterArg — descending would capture an
  // outer loop's seed tensor as an extra parameter of the outlined function.
  void VisitExpr_(const VarPtr& op) override { RecordUse(op.get()); }
  void VisitExpr_(const IterArgPtr& op) override { RecordUse(op.get()); }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // The assigned value is evaluated before the target is bound.
    if (op->value_) VisitExpr(op->value_);
    Define(op->var_);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    // iter_arg init values are evaluated in the enclosing scope.
    for (const auto& ia : op->iter_args_) {
      if (ia->initValue_) VisitExpr(ia->initValue_);
    }
    Define(op->loop_var_);
    for (const auto& rv : op->return_vars_) Define(rv);
    for (const auto& ia : op->iter_args_) Define(ia);
    VisitExpr(op->start_);
    VisitExpr(op->stop_);
    VisitExpr(op->step_);
    VisitStmt(op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& ia : op->iter_args_) {
      if (ia->initValue_) VisitExpr(ia->initValue_);
    }
    for (const auto& rv : op->return_vars_) Define(rv);
    for (const auto& ia : op->iter_args_) Define(ia);
    VisitExpr(op->condition_);
    VisitStmt(op->body_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    VisitExpr(op->condition_);
    for (const auto& rv : op->return_vars_) Define(rv);
    // Walk both arms from the same incoming state so a read on one arm is never
    // masked by a write on the other, then join on *must*-definitions: a
    // variable written by only one arm may still carry its incoming value.
    const std::vector<const Var*> then_new = VisitBranchIsolated(op->then_body_);
    // Bound to a local so the optional's engagement is provable at the use:
    // through `op->`, the analysis cannot tie the guard to the access.
    const auto& else_body = op->else_body_;
    if (!else_body.has_value()) return;
    const std::vector<const Var*> else_new = VisitBranchIsolated(else_body.value());
    const std::unordered_set<const Var*> then_set(then_new.begin(), then_new.end());
    for (const auto* var : else_new) {
      if (then_set.count(var) > 0) Define(var);
    }
  }

 private:
  void RecordUse(const Var* var) {
    if (var && defined_.count(var) == 0 && live_in_.insert(var).second) {
      ordered.push_back(var);
    }
  }

  void Define(const VarPtr& var) { Define(var.get()); }

  void Define(const Var* var) {
    if (var && defined_.insert(var).second) journal_.push_back(var);
  }

  /// Visit `body`, then roll back every definition it introduced and return
  /// them. Undoing via the insertion journal (rather than copying ``defined_``
  /// per arm) keeps the walk linear in the branch's own size; total work is
  /// O(N * if-nesting depth), i.e. O(N) for the straight-line bodies that
  /// dominate InCore scopes.
  std::vector<const Var*> VisitBranchIsolated(const StmtPtr& body) {
    const size_t mark = journal_.size();
    VisitStmt(body);
    std::vector<const Var*> introduced(journal_.begin() + static_cast<std::ptrdiff_t>(mark), journal_.end());
    for (const auto* var : introduced) defined_.erase(var);
    journal_.resize(mark);
    return introduced;
  }

  std::unordered_set<const Var*> live_in_;  ///< membership view of `ordered`
  std::unordered_set<const Var*> defined_;
  std::vector<const Var*> journal_;  ///< insertion order, for branch rollback
};

}  // namespace

ScopeOutliner::ScopeOutliner(std::string func_name, const std::unordered_map<const Var*, TypePtr>& var_types,
                             const std::unordered_map<const Var*, VarPtr>& var_objects,
                             const std::unordered_set<std::string>& known_names, ScopeKind target_scope_kind,
                             FunctionType outlined_func_type, std::string name_suffix, ProgramPtr program,
                             std::shared_ptr<std::unordered_set<std::string>> reserved_func_names)
    : func_name_(std::move(func_name)),
      var_types_(var_types),
      var_objects_(var_objects),
      known_names_(known_names),
      target_scope_kind_(target_scope_kind),
      outlined_func_type_(outlined_func_type),
      name_suffix_(std::move(name_suffix)),
      program_(std::move(program)),
      reserved_func_names_(std::move(reserved_func_names)) {}

/**
 * @brief Substitute store-target variables that were renamed for SSA compliance.
 *
 * When a store-target output is assigned a fresh SSA name at the call site
 * (e.g., buf_0 -> buf_1), subsequent references must use the new variable.
 *
 * ``store_target_renames_`` is kept flat: every entry maps an original
 * store-target Var directly to its *latest* renamed Var. When N sibling
 * scopes write the same target, each scope's CreateFreshStoreTargetVar
 * overwrites the single entry rather than appending a chain link (see
 * CreateFreshStoreTargetVar). A single lookup therefore always yields the
 * current value — a reference after the last scope (the function's
 * ReturnStmt) resolves to the latest, never a stale intermediate.
 */
ExprPtr ScopeOutliner::VisitExpr_(const VarPtr& op) {
  auto it = store_target_renames_.find(op.get());
  if (it != store_target_renames_.end()) {
    return it->second;
  }
  return IRMutator::VisitExpr_(op);
}

/**
 * @brief Compute used_after when no explicit context is available.
 *
 * Two regimes:
 *   1. Scope is nested inside another (non-target) ScopeStmt — only store
 *      targets escape, because scope boundaries confine locally-defined
 *      variables.
 *   2. Scope is at the top level or inside a control-flow body — retain the
 *      original defensive fallback: all defined vars + store targets are
 *      treated as outputs so the caller retains access.
 */
std::unordered_set<const Var*> ScopeOutliner::ComputeFallbackUsedAfter(const ScopeStmtPtr& scope) const {
  // A deferred-completion waiter is registration-only: its scalar
  // bookkeeping and loop induction variables are local implementation
  // details, never semantic outputs.  In particular, exporting all local
  // definitions from a terminal waiter would try to return a loop variable
  // outside the ForStmt that defines it.  OutlineScope validates the full
  // waiter contract immediately after this fallback is computed, so no
  // legitimate store/output can be hidden by returning an empty set here.
  if (ContainsDeferredWait(scope->body_)) return {};

  StoreTargetCollector store_collector;
  store_collector.VisitStmt(scope->body_);
  std::unordered_set<const Var*> used_after;
  if (!inside_nested_scope_body_) {
    var_collectors::VarDefUseCollector def_collector;
    def_collector.VisitStmt(scope->body_);
    used_after = def_collector.var_defs;
  }
  used_after.insert(store_collector.store_targets.begin(), store_collector.store_targets.end());
  return used_after;
}

/**
 * @brief Process SeqStmts to analyze scope outputs using subsequent statements.
 *
 * For each target scope, collects variables referenced in all subsequent statements
 * plus any variables required by a parent scope (propagated via required_outputs_).
 */
StmtPtr ScopeOutliner::VisitStmt_(const SeqStmtsPtr& op) {
  std::vector<StmtPtr> new_stmts;
  bool changed = false;

  // ``with pl.at(...) as tid:`` emits a placeholder
  // ``AssignStmt(tid, system.task_invalid())`` right *before* the scope so
  // ConvertToSSA has a def to rename consistently with the scope's
  // ``task_id_var`` attr reference and subsequent ``deps=[tid]`` uses. The
  // real binding is synthesised by ``OutlineScope`` below (as a
  // TupleGetItem on the outlined Call's tail). When we see such a
  // placeholder, drop it so the synthesised binding is the sole definition.
  auto is_task_id_placeholder = [](const StmtPtr& s, const Var* target) -> bool {
    if (!s || !target) return false;
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(s);
    if (!assign || assign->var_.get() != target) return false;
    auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
    if (!call || !call->op_) return false;
    return IsOp(call, "system.task_invalid");
  };

  // Indices of any preceding-scope placeholders we plan to drop.
  std::unordered_set<size_t> dropped_indices;
  for (size_t i = 1; i < op->stmts_.size(); ++i) {
    auto scope = std::dynamic_pointer_cast<const ScopeStmt>(op->stmts_[i]);
    if (!scope || scope->GetScopeKind() != target_scope_kind_) continue;
    auto tid_var = scope->GetAttr<VarPtr>(kAttrTaskIdVar);
    if (!tid_var) continue;
    if (is_task_id_placeholder(op->stmts_[i - 1], tid_var.get())) {
      dropped_indices.insert(i - 1);
    }
  }

  for (size_t i = 0; i < op->stmts_.size(); ++i) {
    if (dropped_indices.count(i)) {
      // Skip the ``with pl.at(...) as tid:`` placeholder — OutlineScope
      // emits the real ``AssignStmt(tid, TupleGetItem(...))`` in its
      // returned SeqStmts.
      changed = true;
      continue;
    }
    auto scope = std::dynamic_pointer_cast<const ScopeStmt>(op->stmts_[i]);
    // Always compute what's used in the tail of this SeqStmts; this set is
    // the "used_after" for a target scope at position i, and doubles as the
    // required_outputs_ propagated into a non-target statement so any
    // target-kind scope nested inside knows what needs to leak out.
    var_collectors::VarDefUseCollector after_ref_collector;
    for (size_t j = i + 1; j < op->stmts_.size(); ++j) {
      if (dropped_indices.count(j)) continue;
      after_ref_collector.VisitStmt(op->stmts_[j]);
    }
    auto after_refs = after_ref_collector.GetAllVarRefs();

    if (scope && scope->GetScopeKind() == target_scope_kind_) {
      // Also include variables required by parent scope
      auto used_after = after_refs;
      used_after.insert(required_outputs_.begin(), required_outputs_.end());

      // When no context is available (no subsequent statements and no parent
      // requirements), fall back to scope-nesting-aware defaults.  This
      // happens when a single ScopeStmt is wrapped in SeqStmts inside a
      // control-flow body (if/for/while) where the outer context hasn't
      // propagated required_outputs_, or when the scope sits directly
      // inside another scope (e.g. an InCoreScopeStmt at the top of a
      // SpmdScopeStmt body produced by the ``for i in pl.spmd(...)`` form).
      if (used_after.empty()) {
        used_after = ComputeFallbackUsedAfter(scope);
      }

      // Outline this scope with context about what's used after
      auto outlined_stmt = OutlineScope(scope, used_after);
      new_stmts.push_back(outlined_stmt);
      changed = true;
    } else {
      // Recursively visit non-target statements.  Temporarily extend
      // required_outputs_ with the tail-use set so any target-kind scope
      // nested inside this statement can compute a correct used_after.
      auto saved_required_outputs = required_outputs_;
      required_outputs_.insert(after_refs.begin(), after_refs.end());
      auto visited = VisitStmt(op->stmts_[i]);
      required_outputs_ = saved_required_outputs;
      new_stmts.push_back(visited);
      if (visited != op->stmts_[i]) {
        changed = true;
      }
    }
  }

  if (!changed) {
    return op;
  }
  return SeqStmts::Flatten(std::move(new_stmts), op->span_);
}

/**
 * @brief Handle ScopeStmts that are direct children of another node (not
 * inside a SeqStmts).
 *
 * The fallback honours scope nesting via ``inside_nested_scope_body_``: an
 * inner scope whose only "context" is an enclosing non-target scope has no
 * way for its locally-defined variables to escape, so we only expose store
 * targets. Scopes at the true top level (outside any parent scope body)
 * retain the original defensive "all defs are outputs" behaviour.
 */
// Shared per-kind logic: outline if kind matches, else descend with the
// nested-scope flag set so any target-kind scope we find inside can make a
// correct used_after decision.
template <typename ScopeT>
StmtPtr ScopeOutliner::VisitScopeKind(const std::shared_ptr<const ScopeT>& op) {
  if (op->GetScopeKind() != target_scope_kind_) {
    bool prev = std::exchange(inside_nested_scope_body_, true);
    auto result = IRMutator::VisitStmt_(op);
    inside_nested_scope_body_ = prev;
    return result;
  }
  // Prefer the enclosing SeqStmts' propagated requirements over the
  // "no context" fallback.  This matters when a target scope is a direct
  // child of another scope (SpmdScopeStmt{InCoreScopeStmt{...}}) — the
  // enclosing SeqStmts visitor populates required_outputs_ with variables
  // the post-scope statements still reference.
  if (required_outputs_.empty()) {
    return OutlineScope(op, ComputeFallbackUsedAfter(op));
  }
  std::unordered_set<const Var*> used_after = required_outputs_;
  StoreTargetCollector store_collector;
  store_collector.VisitStmt(op->body_);
  used_after.insert(store_collector.store_targets.begin(), store_collector.store_targets.end());
  return OutlineScope(op, used_after);
}

StmtPtr ScopeOutliner::VisitStmt_(const InCoreScopeStmtPtr& op) { return VisitScopeKind(op); }

StmtPtr ScopeOutliner::VisitStmt_(const ClusterScopeStmtPtr& op) { return VisitScopeKind(op); }

StmtPtr ScopeOutliner::VisitStmt_(const HierarchyScopeStmtPtr& op) { return VisitScopeKind(op); }

StmtPtr ScopeOutliner::VisitStmt_(const SpmdScopeStmtPtr& op) { return VisitScopeKind(op); }

// SplitAiv is never an outline target (target is always InCore), so this
// descends into the body via VisitScopeKind's non-target branch, preserving
// the nested SplitAivScopeStmt inside the outlined InCore function body.
StmtPtr ScopeOutliner::VisitStmt_(const SplitAivScopeStmtPtr& op) { return VisitScopeKind(op); }

/// True when `name` is already claimed by this function (`known_names_`) or,
/// when the pass opts in, by any earlier function in the program
/// (`reserved_func_names_`).
bool ScopeOutliner::IsNameTaken(const std::string& name) const {
  return known_names_.count(name) > 0 ||
         (reserved_func_names_ != nullptr && reserved_func_names_->count(name) > 0);
}

/// Append the smallest `_<n>` suffix that makes `base` unique program-wide.
std::string ScopeOutliner::NumericSuffix(const std::string& base) const {
  int dedup_counter = 0;
  std::string disambiguated;
  do {
    disambiguated = base + "_" + std::to_string(dedup_counter++);
  } while (IsNameTaken(disambiguated));
  return disambiguated;
}

/**
 * @brief Resolve an outlined-function name collision.
 *
 * `known_names_` is function-local; `reserved_func_names_` (when the pass
 * provides it) is the program-wide set of outlined names already emitted by
 * earlier functions. Two collision shapes are handled differently:
 *
 * - **Cross-function** (name free locally but already taken by another
 *   function): almost always a reused `@pl.jit.inline` helper outlined from
 *   two sibling child kernels (issue #1711). Namespace under the originating
 *   function for a debuggable, source-derived name (`single_b_dup_scope`)
 *   rather than an opaque numeric suffix.
 * - **In-function** (name taken within this same function, e.g. two scopes
 *   sharing a `name_hint`): preserve the historical numeric-suffix behavior
 *   (`my_kernel` -> `my_kernel_0`).
 */
std::string ScopeOutliner::DisambiguateOutlinedName(const std::string& candidate) const {
  const bool taken_local = known_names_.count(candidate) > 0;
  if (!IsNameTaken(candidate)) {
    return candidate;
  }
  if (!taken_local) {
    // Cross-function collision: namespace under the source function.
    std::string namespaced = func_name_ + "_" + candidate;
    return IsNameTaken(namespaced) ? NumericSuffix(namespaced) : namespaced;
  }
  // In-function collision: numeric suffix, matching long-standing behavior.
  return NumericSuffix(candidate);
}

/**
 * @brief Outline a single scope into a separate function.
 *
 * @param op The scope statement to outline
 * @param used_after Variables (by pointer) used in subsequent statements (determines outputs)
 */
StmtPtr ScopeOutliner::OutlineScope(const ScopeStmtPtr& op,
                                    const std::unordered_set<const Var*>& used_after) {
  // Generate function name: use user-provided hint when available, otherwise auto-generate.
  std::string outlined_func_name;
  if (!op->name_hint_.empty()) {
    outlined_func_name = op->name_hint_;
    scope_counter_++;  // Keep counter stable for unnamed scopes
  } else {
    std::string suffix = name_suffix_;
    if (auto hier = As<HierarchyScopeStmt>(op)) {
      suffix = GenerateHierarchySuffix(hier->level_, hier->role_);
    }
    std::ostringstream name_stream;
    name_stream << func_name_ << suffix << scope_counter_++;
    outlined_func_name = name_stream.str();
  }
  outlined_func_name = DisambiguateOutlinedName(outlined_func_name);
  known_names_.insert(outlined_func_name);
  if (reserved_func_names_ != nullptr) {
    reserved_func_names_->insert(outlined_func_name);
  }

  auto deferred_wait = DeferredWaitContractValidator::Validate(op->body_, op->span_);
  if (deferred_wait.has_deferred_wait) {
    CHECK_SPAN(op->GetScopeKind() == ScopeKind::InCore, op->span_)
        << "pld.system.defer_wait must be the body of a CORE_GROUP / InCore task";
    CHECK_SPAN(!inside_nested_scope_body_, op->span_)
        << "pld.system.defer_wait must be in a task-level pl.at(CORE_GROUP) scope; nesting the "
           "waiter under pl.spmd or another task-launch scope is unsupported because the outer "
           "launch owns dispatch predicate and early-resolve semantics";
    CHECK_SPAN(!op->GetAttr<ExprPtr>(kAttrPredicate, nullptr), op->span_)
        << "pld.system.defer_wait task cannot use a dispatch predicate; every submitted waiter "
           "must register its completion condition";
  }

  // Definitions made by the scope body (before recursing) — the basis for the
  // output set below, and for the rebind check that follows the input set.
  var_collectors::VarDefUseCollector body_collector;
  body_collector.VisitStmt(op->body_);

  // Store targets present in the scope body. Needed both for the captured
  // read-modify-write check below and, further down, to decide whether a
  // post-store alias's original target already appears in output_vars.
  StoreTargetCollector store_collector;
  store_collector.VisitStmt(op->body_);

  // Inputs: the scope's live-in set — variables the body reads before it
  // (re)defines them, so their incoming value comes from the caller.
  // Deliberately NOT ``var_uses \ var_defs``; see UpwardExposedUseCollector
  // for why that flow-insensitive difference drops a rebound capture.
  //
  // Iterate in first-read order to preserve the callee's parameter ordering.
  // var_objects_ is a pure identity symbol table (never rewritten with
  // renames), so obj_it->second is the same Var that appears in the body —
  // input_vars[i].get() is the body pointer, the key used for both the body
  // substitution and the call-site lookup below.
  UpwardExposedUseCollector live_in;
  live_in.VisitStmt(op->body_);

  std::vector<VarPtr> input_vars;
  for (const Var* var_ptr : live_in.ordered) {
    auto obj_it = var_objects_.find(var_ptr);
    CHECK(obj_it != var_objects_.end()) << "Variable " << var_ptr->name_hint_ << " not found in var_objects";
    input_vars.push_back(obj_it->second);
  }

  // A captured variable the body also rebinds is only representable in the
  // outlined function when the rebind is a ``tile.store`` — that shape gets
  // an InOut param plus a distinct result Var below (for Hierarchy scopes,
  // only the InOut param: they skip the store-target export, so the body
  // keeps the original rebind. Still correct — the capture is a parameter
  // either way, which is what was broken). Anything else (e.g. a pre-SSA
  // ``out = pl.assemble(out, ...)``) would need real SSA construction, so
  // reject it here rather than emit a function that reassigns a variable it
  // never declared. Unreachable on SSA input, where no live-in variable is
  // ever assigned inside the body.
  for (const Var* var_ptr : live_in.ordered) {
    if (!body_collector.var_defs.count(var_ptr)) continue;
    INTERNAL_CHECK_SPAN(store_collector.store_targets.count(var_ptr) > 0, op->span_)
        << "Internal error: scope '" << outlined_func_name << "' captures '" << var_ptr->name_hint_
        << "' and rebinds it under the same name without a tile.store. The outlining passes "
           "require SSA form (IRProperty::SSAForm) — run ConvertToSSA first.";
  }

  // Outputs: variables defined in the scope AND used after it
  std::vector<VarPtr> output_vars;
  std::unordered_set<const Var*> store_output_set;

  // Collect type info from scope body for output variables
  VarCollector scope_var_collector;
  scope_var_collector.VisitStmt(op->body_);

  // Map any SSA post-store alias (var_def bound to a tile.store call) back
  // to its store target so we don't export the same tensor twice.
  PostStoreAliasCollector post_store_collector;
  post_store_collector.VisitStmt(op->body_);

  // Aliases deferred to the call-site emission: each pair maps a
  // scope-local SSA post-store alias (pointer identity in the scope body)
  // to the external store target's var_objects_ pointer.  After the fresh
  // store-target Var is created at the call site we rename the alias to
  // it, so subsequent parent-function references resolve correctly.
  std::vector<std::pair<const Var*, const Var*>> deferred_post_store_aliases;
  for (const Var* var_ptr : body_collector.var_defs_ordered) {
    if (!used_after.count(var_ptr)) continue;

    // Skip if this var_def is a post-store alias of an external store
    // target: that same tensor will be exported by the store-target pass
    // below, and we don't want a duplicated output entry.
    auto alias_it = post_store_collector.alias_to_target.find(var_ptr);
    const Var* target_ptr =
        (alias_it != post_store_collector.alias_to_target.end()) ? alias_it->second : nullptr;
    if (target_ptr && store_collector.store_targets.count(target_ptr) && live_in.IsExternal(target_ptr)) {
      auto ext_it = var_objects_.find(target_ptr);
      CHECK(ext_it != var_objects_.end())
          << "Store target " << target_ptr->name_hint_ << " not found in var_objects";
      deferred_post_store_aliases.emplace_back(var_ptr, ext_it->second.get());
      continue;
    }

    auto scope_it = scope_var_collector.var_objects.find(var_ptr);
    CHECK(scope_it != scope_var_collector.var_objects.end())
        << "Variable " << var_ptr->name_hint_ << " not found in scope body";
    output_vars.push_back(scope_it->second);
  }

  // Also treat store targets as outputs: external tensors modified via
  // tile.store.  These represent side-effect outputs that must be
  // returned regardless of whether they appear in used_after, because the
  // store mutates an externally-visible buffer (e.g. loop-carried state).
  //
  // Skip for Hierarchy scopes: the outlined function receives the buffer
  // as an InOut parameter, so the store side-effect is already visible
  // to the caller without an explicit return.
  //
  // Track two pointer identities per store target:
  //   - var_objects_ pointer (ext_it->second.get()) — goes into output_vars
  //     and store_output_set for consistent classification
  //   - body pointer (var_ptr) — kept in store_body_ptrs for the
  //     StoreEvalToAssignMutator, which matches against the un-substituted
  //     scope body where store targets retain their original pointers
  //
  // "External" is the live-in test, not ``!var_defs.count(...)``: a target
  // the body rebinds under its own name (``c = pl.store(t, off, c)``) is
  // still caller-owned, and dropping it here is what left the read dangling.
  std::unordered_map<const Var*, const Var*> store_body_ptrs;
  if (op->GetScopeKind() != ScopeKind::Hierarchy) {
    for (const Var* var_ptr : store_collector.store_targets) {
      if (live_in.IsExternal(var_ptr)) {
        auto ext_it = var_objects_.find(var_ptr);
        CHECK(ext_it != var_objects_.end())
            << "Variable " << var_ptr->name_hint_ << " not found in var_objects";
        output_vars.push_back(ext_it->second);
        store_output_set.insert(ext_it->second.get());
        store_body_ptrs[ext_it->second.get()] = var_ptr;
      }
    }
  }

  // Recursively transform the scope body (handles nested scopes).
  // Save/restore state so nested scopes get their own hierarchical names and counters.
  // Also overlay the current scope's symbol table while recursing so nested
  // outlining resolves names to the lexically-nearest Var, not to an unrelated
  // same-named Var elsewhere in the function.
  // store_target_renames_ must be cleared so parent renames don't leak into the scope
  // body — the scope's own parameter substitution handles variable mapping instead.
  std::string saved_func_name = func_name_;
  int saved_scope_counter = scope_counter_;
  auto saved_var_types = var_types_;
  auto saved_var_objects = var_objects_;
  auto saved_known_names = known_names_;
  auto saved_required_outputs = required_outputs_;
  auto saved_renames = store_target_renames_;
  func_name_ = outlined_func_name;
  scope_counter_ = 0;
  for (const auto& [ptr, type] : scope_var_collector.var_types) {
    var_types_[ptr] = type;
  }
  for (const auto& [ptr, var] : scope_var_collector.var_objects) {
    var_objects_[ptr] = var;
  }
  known_names_.insert(scope_var_collector.known_names.begin(), scope_var_collector.known_names.end());
  store_target_renames_.clear();
  // Propagate output requirements so nested scopes know what's needed
  required_outputs_.clear();
  for (const auto& var : output_vars) {
    required_outputs_.insert(var.get());
  }
  auto recursed_body = VisitStmt(op->body_);
  func_name_ = saved_func_name;
  scope_counter_ = saved_scope_counter;
  var_types_ = saved_var_types;
  var_objects_ = saved_var_objects;
  known_names_ = saved_known_names;
  required_outputs_ = saved_required_outputs;
  store_target_renames_ = saved_renames;

  // Create fresh parameters for the outlined function.
  // Infer param directions from the inner callee when possible (requires program_).
  std::vector<ParamDirection> inferred_directions =
      InferParamDirections(input_vars, op->body_, store_output_set);
  std::vector<VarPtr> input_params;
  std::vector<ParamDirection> input_param_directions;
  std::unordered_map<const Var*, VarPtr> var_substitution_map;
  for (size_t i = 0; i < input_vars.size(); ++i) {
    const auto& input_var = input_vars[i];
    auto param_var = std::make_shared<Var>(input_var->name_hint_, input_var->GetType(), op->span_);
    input_params.push_back(param_var);
    input_param_directions.push_back(inferred_directions[i]);
    // input_var.get() is the body pointer (var_objects_ is identity), so the
    // substitution reaches the actual use-site in the scope body.
    var_substitution_map[input_var.get()] = param_var;
  }

  // Build the set of names already used in the outlined function (inputs + scope-body locals)
  // to ensure generated output names don't collide.
  std::unordered_set<std::string> outlined_used_names;
  for (const auto& input_var : input_vars) {
    outlined_used_names.insert(input_var->name_hint_);
  }
  outlined_used_names.insert(scope_var_collector.known_names.begin(), scope_var_collector.known_names.end());

  // Create fresh output variables for the outlined function
  std::vector<VarPtr> outlined_output_vars;
  std::vector<TypePtr> return_types;
  for (const auto& out_var : output_vars) {
    bool is_store = store_output_set.count(out_var.get()) > 0;
    TypePtr var_type;
    if (is_store) {
      // Store target: external variable, look up from outer symbol table
      auto type_it = var_types_.find(out_var.get());
      CHECK(type_it != var_types_.end())
          << "Variable " << out_var->name_hint_ << " not found in symbol table";
      var_type = type_it->second;
    } else {
      // Regular output: defined in scope body
      var_type = out_var->GetType();
    }
    // For store targets, create a fresh variable with a unique "_store_ret" suffix
    // to avoid redefining the input parameter in SSA form.
    std::string out_var_name;
    if (is_store) {
      out_var_name = auto_name::BuildName(auto_name::GetBaseName(out_var->name_hint_), "", "store");
      if (outlined_used_names.count(out_var_name)) {
        out_var_name = auto_name::GenerateFreshNameLike(out_var_name, outlined_used_names);
      }
    } else {
      out_var_name = out_var->name_hint_;
    }
    outlined_used_names.insert(out_var_name);
    auto outlined_var = std::make_shared<Var>(out_var_name, var_type, op->span_);
    outlined_output_vars.push_back(outlined_var);
    return_types.push_back(var_type);
    if (!is_store) {
      var_substitution_map[out_var.get()] = outlined_var;
    }
  }

  // Convert EvalStmt/AssignStmt(tile.store) to assign _store_ret vars BEFORE
  // Substitute, since store_body_ptrs uses the original body Var pointers.
  auto pre_sub_body = recursed_body;
  if (!store_output_set.empty()) {
    std::unordered_map<const Var*, VarPtr> store_target_vars;
    for (size_t idx = 0; idx < output_vars.size(); ++idx) {
      auto body_it = store_body_ptrs.find(output_vars[idx].get());
      if (body_it != store_body_ptrs.end()) {
        store_target_vars[body_it->second] = outlined_output_vars[idx];
      }
    }
    StoreEvalToAssignMutator store_mutator(store_target_vars, outlined_used_names);
    pre_sub_body = store_mutator.VisitStmt(pre_sub_body);
  }

  // Apply pointer-based substitution after store results are materialized.
  //
  // We can't reuse `Substitute` here because IRMutator::VisitExpr_(VarPtr)
  // mints a *fresh* Var when an old Var's type embeds a remapped shape Var
  // (see mutator.cpp:225-239). For a tensor input whose shape references
  // another input scalar, this means the body ends up referencing a Var
  // that's NOT the one we just pushed into `input_params`; the codegen's
  // param-binding loop then can't find a tensor view for it. We need
  // visibility into the post-substitution remap state to pull out the
  // freshened param Vars and update `input_params` accordingly.
  class TrackingSubstituteMutator : public IRMutator {
   public:
    explicit TrackingSubstituteMutator(const std::unordered_map<const Var*, VarPtr>& var_map) {
      for (const auto& [k, v] : var_map) {
        var_remap_[k] = v;
      }
    }
    [[nodiscard]] const std::unordered_map<const Expr*, ExprPtr>& GetVarRemap() const { return var_remap_; }
  };
  TrackingSubstituteMutator subst_mutator(var_substitution_map);
  auto transformed_body = subst_mutator.VisitStmt(pre_sub_body);

  // Reconcile param/output Vars with any freshened versions created during
  // substitution. ResolveVarRemapHit memoizes the resolved (final) Var back
  // into var_remap_ keyed by the original Var, so the chain
  //   old → seed (initial param/outlined) → freshened (after type remap)
  // collapses to old → freshened. Pick that out and replace the stale
  // entry in input_params / outlined_output_vars / return_types.
  const auto& post_remap = subst_mutator.GetVarRemap();
  auto resolve_to_freshened = [&](const VarPtr& original, const VarPtr& seeded) -> VarPtr {
    auto it = post_remap.find(original.get());
    if (it == post_remap.end()) return seeded;
    auto freshened = AsVarLike(it->second);
    if (!freshened) return seeded;
    return freshened;
  };
  for (size_t i = 0; i < input_vars.size(); ++i) {
    input_params[i] = resolve_to_freshened(input_vars[i], input_params[i]);
  }
  for (size_t i = 0; i < output_vars.size(); ++i) {
    bool is_store = store_output_set.count(output_vars[i].get()) > 0;
    if (is_store) {
      // Store targets aren't seeded into var_substitution_map, but their types
      // may still embed remapped shape vars that trigger freshening during
      // substitution. Check only the outlined var key; the original key may
      // alias an input param (InOut parameters) and produce a false match.
      auto it = post_remap.find(outlined_output_vars[i].get());
      if (it != post_remap.end()) {
        if (auto freshened = AsVarLike(it->second)) {
          outlined_output_vars[i] = freshened;
          return_types[i] = freshened->GetType();
        }
      }
      continue;
    }
    auto freshened = resolve_to_freshened(output_vars[i], outlined_output_vars[i]);
    outlined_output_vars[i] = freshened;
    return_types[i] = freshened->GetType();
  }

  // Build outlined function body (transformed body + return statement).
  //
  // Return params, not SSA result vars: every tensor output the scope
  // produces is physically one of the function's params (store targets are
  // InOut inputs; call results write through Out/InOut args). Returning
  // the param makes the return->param mapping explicit by pointer identity
  // so orchestration codegen never re-derives it heuristically (#1702).
  StmtPtr outlined_body;
  if (outlined_output_vars.empty()) {
    outlined_body = transformed_body;
  } else {
    std::unordered_map<const Var*, VarPtr> input_to_param;
    for (size_t i = 0; i < input_vars.size(); ++i) {
      input_to_param[input_vars[i].get()] = input_params[i];
    }
    std::vector<ExprPtr> return_exprs;
    return_exprs.reserve(outlined_output_vars.size());
    for (size_t i = 0; i < output_vars.size(); ++i) {
      VarPtr ret = outlined_output_vars[i];
      if (store_output_set.count(output_vars[i].get())) {
        // Store target: also an input, so the param is known directly.
        auto param_it = input_to_param.find(output_vars[i].get());
        if (param_it != input_to_param.end()) ret = param_it->second;
      } else if (AsTensorTypeLike(ret->GetType())) {
        if (auto param = return_lineage::TraceToParam(ret, transformed_body, input_params, program_)) {
          ret = param;
        }
      }
      return_exprs.push_back(ret);
    }
    auto return_stmt = std::make_shared<ReturnStmt>(return_exprs, op->span_);

    std::vector<StmtPtr> body_stmts;
    if (auto seq_stmts = std::dynamic_pointer_cast<const SeqStmts>(transformed_body)) {
      body_stmts = seq_stmts->stmts_;
    } else {
      body_stmts.push_back(transformed_body);
    }
    body_stmts.push_back(return_stmt);
    outlined_body = std::make_shared<SeqStmts>(body_stmts, op->span_);
  }

  // Map each captured input Var to its positional index. The index is exact for
  // BOTH surfaces the translations below need: ``input_params`` is built
  // index-parallel to ``input_vars`` and is what the outlined ``Function`` is
  // constructed from, and ``call_args`` is built from ``input_vars`` in the same
  // order. Built once here, ahead of the attr resolution that follows, and
  // reused by the no_dep / dump translations further down.
  std::unordered_map<const Var*, int32_t> input_var_to_idx;
  input_var_to_idx.reserve(input_vars.size());
  for (size_t i = 0; i < input_vars.size(); ++i) {
    input_var_to_idx[input_vars[i].get()] = static_cast<int32_t>(i);
  }

  // Register the outlined function (propagate level/role from ScopeStmt, convert split/core_num to attrs)
  std::vector<std::pair<std::string, std::any>> outlined_attrs;
  auto append_split_attr = [&](SplitMode split) {
    if (split != SplitMode::None) {
      outlined_attrs.emplace_back("split", static_cast<int>(split));
    }
  };
  // Propagate the optional cross-core ring depth (pl.split(mode, slot_num=N))
  // from the scope's attrs onto the outlined function, where ExpandMixedKernel
  // reads it to size the automatic cube->vector pipe.
  auto append_slot_num_attr = [&]() {
    if (op->HasAttr("slot_num")) {
      outlined_attrs.emplace_back("slot_num", op->GetAttr<int>("slot_num", 0));
    }
  };
  auto append_windowize_attr = [&]() {
    if (op->GetAttr<bool>("windowize", false)) {
      outlined_attrs.emplace_back("windowize", true);
    }
  };
  auto append_deferred_completion_waiter_attr = [&]() {
    if (deferred_wait.has_deferred_wait) {
      outlined_attrs.emplace_back(kAttrDeferredCompletionWaiter, true);
    }
  };
  // Resolve pl.set_cache_policy declarations onto the outlined function's params.
  // The scope attr is consumed here and never propagated: downstream the function
  // attr (param indices) is the single carrier until ConvertTensorToTileOps
  // converts it to per-load kwargs at pass 10.
  auto append_cache_policy_attr = [&]() {
    auto scope_cache_policies = op->GetAttr<std::vector<std::pair<VarPtr, int>>>(kAttrCachePolicyVars);
    if (scope_cache_policies.empty()) return;
    std::vector<std::pair<int32_t, int>> cache_policy_indices;
    cache_policy_indices.reserve(scope_cache_policies.size());
    for (const auto& [v, policy] : scope_cache_policies) {
      INTERNAL_CHECK_SPAN(v, op->span_)
          << "Internal error: null Var in cache_policy_vars on outlined scope '" << outlined_func_name << "'";
      auto it = input_var_to_idx.find(v.get());
      CHECK_SPAN(it != input_var_to_idx.end(), op->span_)
          << "pl.set_cache_policy(...) references tensor '" << v->name_hint_
          << "', which is not captured by the scope body. Only tensors actually read inside the "
             "scope can be declared.";
      // A bypassing read only makes sense on a tensor this kernel does not
      // write: the policy is a promise about the bytes, and the direction
      // inference above already knows whether the scope writes them.
      const ParamDirection dir = input_param_directions[static_cast<size_t>(it->second)];
      CHECK_SPAN(static_cast<CachePolicy>(policy) != CachePolicy::kBypass || dir == ParamDirection::In,
                 op->span_)
          << "pl.set_cache_policy(" << v->name_hint_
          << ", CachePolicy.BYPASS) is not allowed on a tensor this scope writes ("
          << ParamDirectionToString(dir)
          << "). A bypassing read of bytes the same kernel writes is a coherency bug.";
      cache_policy_indices.emplace_back(it->second, policy);
    }
    // Sorted by param index for the same reason ``arg_dir_override_indices`` is:
    // the declaration set is order-independent, so two programs that differ only
    // in the order the user wrote the declarations (or in capture order) must
    // produce structurally equal IR, and dumps must stay deterministic.
    std::sort(cache_policy_indices.begin(), cache_policy_indices.end());
    outlined_attrs.emplace_back(kAttrCachePolicyParams, std::move(cache_policy_indices));
  };
  // Bridge the first-class SplitAivScopeStmt region into the function-level
  // AIV-split markers the downstream contract (passes 11-24) expects. The
  // explicit ``pl.split_aiv`` form is a node in the body, not a scope attr
  // (the old MarkCurrentScopeSplitAiv marker was deleted). When the InCore
  // body contains a region, stamp the mode-agnostic ``split_aiv=true`` bool
  // (ExpandMixedKernel copies it to both lanes; SplitVectorKernel reads it to
  // bypass its automatic per-op halving). Also stamp a coarse representative
  // ``split`` mode from the region node — but only when the scope itself
  // carries no AUTO cross-core transfer split (``incore->split_``), which has
  // a separate meaning. The authoritative per-region mode is ``node->split_``
  // (consumed at pass 21).
  auto append_split_aiv_attr = [&](SplitMode incore_split) {
    SplitAivModeSummaryFinder finder;
    finder.VisitStmt(op->body_);
    if (!finder.found) return;
    // A function-level AUTO split (optimizations=[pl.split(mode)], carried as the
    // scope's own split_) and explicit pl.split_aiv region(s) are mutually
    // exclusive AIV-split mechanisms. Downstream lowering takes the per-region
    // path and would silently drop the function-level split, so reject the
    // combination HERE — the scope's user split (incore_split) and the regions
    // are both visible only at outline time; post-outline they merge
    // indistinguishably into the function's split / split_aiv attrs (a single
    // pl.split_aiv region legitimately yields a derived function-level split).
    //
    // RFC #1820 additionally rejects a literal `pl.split(pl.SplitMode.NONE)`
    // here: it carries no split of its own, but writing it still reads as "auto
    // and manual split mixed on one scope". That spelling is invisible to this
    // pass since #2205 collapsed InCoreScopeStmt's two encodings of "no split"
    // into `SplitMode::None`, so the parser rejects it instead — where the
    // literal is still visible (see `_reject_user_split_with_split_aiv_region`).
    // This check remains the backstop for IR that never went through the parser
    // (deserialized `.pto`, programmatically built scopes).
    CHECK_SPAN(incore_split == SplitMode::None, op->span_)
        << "scope combines a function-level pl.split(...) (optimizations=[pl.split(...)]) with "
           "pl.split_aiv region(s); these are mutually exclusive AIV-split mechanisms. Remove "
           "optimizations=[pl.split(...)] or the pl.split_aiv region(s) — the function-level "
           "split would otherwise be silently dropped (the per-region split governs the lanes). "
           "To pin a custom cross-core slot count, use "
           "optimizations=[pl.cross_core_slot(slot_num=N)], which is orthogonal to splitting.";
    outlined_attrs.emplace_back(kAttrSplitAiv, true);
    // Stamp a function-level representative ``split`` mode ONLY when all regions
    // share one mode (``uniform_mode``) AND that mode is a real split. Differing
    // sibling modes have no single representative: leave the function-level mode
    // unset — the authoritative per-region mode rides ``node->split_`` (consumed
    // at pass 20). No need to re-check incore_split here: the CHECK above
    // guarantees it is None.
    //
    // ``SplitMode::None`` is excluded for the same reason the sibling
    // ``append_split_attr`` excludes it: "no split" has ONE canonical encoding at
    // the function-attr level — an absent key. ``Function::GetSplitMode`` maps a
    // stored 0 to ``nullopt`` exactly as it does an absent key, so the entry is
    // invisible to every consumer, while the parser drops it on the way back in —
    // which made print -> parse lossy (``Kwargs size mismatch``).
    if (finder.uniform_mode.has_value() && finder.uniform_mode.value() != SplitMode::None) {
      outlined_attrs.emplace_back("split", static_cast<int>(finder.uniform_mode.value()));
    }
  };
  // Captured here, attached to the synthesised dispatch below — the launch
  // spec belongs to the launch site, never to the outlined callee (see
  // ``kAttrCoreNum`` for why).
  auto spmd_scope = As<SpmdScopeStmt>(op);
  if (auto incore = As<InCoreScopeStmt>(op)) {
    append_split_attr(incore->split_);
    append_slot_num_attr();
    append_windowize_attr();
    append_deferred_completion_waiter_attr();
    append_split_aiv_attr(incore->split_);
  }
  // Scope-kind agnostic: a cache-policy declaration reads the same on an InCore
  // task and on the Hierarchy scope that encloses one, and both outline through
  // this helper.
  append_cache_policy_attr();
  std::optional<Level> outlined_level;
  std::optional<Role> outlined_role;
  if (auto hier = As<HierarchyScopeStmt>(op)) {
    outlined_level = hier->level_;
    outlined_role = hier->role_;
  }
  auto outlined_func = std::make_shared<Function>(outlined_func_name, input_params, input_param_directions,
                                                  return_types, outlined_body, op->span_, outlined_func_type_,
                                                  outlined_level, outlined_role, std::move(outlined_attrs));
  outlined_functions_.push_back(outlined_func);

  // Build the call site in the parent function
  auto global_var = std::make_shared<GlobalVar>(outlined_func_name);
  std::vector<ExprPtr> call_args;
  for (const auto& input_var : input_vars) {
    // The argument must be the value current as of this scope. A store
    // target written by an earlier sibling scope has been renamed; its
    // latest Var lives in store_target_renames_ keyed on the original Var
    // (the map is flat — see CreateFreshStoreTargetVar — so one lookup
    // yields the current value regardless of how many scopes wrote it).
    // Everything else passes through var_objects_ unchanged.
    auto rename_it = store_target_renames_.find(input_var.get());
    if (rename_it != store_target_renames_.end()) {
      call_args.push_back(rename_it->second);
      continue;
    }
    auto var_it = var_objects_.find(input_var.get());
    CHECK(var_it != var_objects_.end())
        << "Variable " << input_var->name_hint_ << " not found in var_objects";
    call_args.push_back(var_it->second);
  }

  // ``with pl.at(..., deps=[...]) as tid:`` attaches metadata to the
  // ScopeStmt via ``attrs_``. The outliner propagates it onto the
  // synthesised call-like expression:
  //   * ``task_id_var`` (a ``Scalar[TASK_ID]`` Var) → emit an ``ir.Submit``
  //     instead of a plain ``ir.Call``. The Submit's return type is the
  //     augmented ``Tuple{*<scope outputs>, Scalar[TASK_ID]}`` so the
  //     trailing TupleGetItem binds the producer TaskId. Mid-pipeline
  //     dumps print the synthesised call as ``pl.submit(self.<outlined>,
  //     ..., deps=[...])`` — visually distinct from a plain function call,
  //     matching the explicit ``pl.submit(...)`` surface.
  //   * ``manual_dep_edges`` → fold into ``Submit::deps_``. A scope with
  //     deps but no ``as tid`` binding gets a synthetic TaskId Var so the
  //     dispatch is still a Submit — a plain GlobalVar Call must never
  //     carry ``manual_dep_edges`` (ManualDepsOnSubmitOnly invariant).
  //   * ``arg_direction_overrides_vars`` (from ``pl.at(no_dep_args=[...])``) →
  //     translate the captured-Var list into positional indices into
  //     ``call_args`` (using the ``input_vars`` order, which call_args
  //     mirrors 1:1) and attach as ``attrs[arg_direction_overrides]``
  //     on whichever node we emit. ``DeriveCallDirections`` then
  //     overwrites those slots to ``ArgDirection::NoDep`` — the same path
  //     as ``pl.no_dep(t)`` wrappers at explicit kernel call sites.
  VarPtr scope_task_id_var = op->GetAttr<VarPtr>(kAttrTaskIdVar);
  std::vector<VarPtr> scope_dep_edges = op->GetAttr<std::vector<VarPtr>>(kAttrManualDepEdges);
  std::vector<VarPtr> scope_no_dep_vars = op->GetAttr<std::vector<VarPtr>>(kAttrArgDirOverrideVars);
  // Scope-level selective-dump carrier (from ``pl.dump_tag`` at parse, an
  // explicit / round-trip ``dumps=`` list, or the inline-call ``dump_vars``
  // transfer). Each entry is an outer-scope tensor Var that should be dumped on
  // this dispatch — translated below into the synthesised Call/Submit's
  // ``kAttrDumpVars`` by Var identity, exactly as ``scope_no_dep_vars`` is
  // translated into ``kAttrArgDirectionOverrides``.
  std::vector<VarPtr> scope_dump_vars = op->GetAttr<std::vector<VarPtr>>(kAttrDumpVars);
  // Speculative early-dispatch opt-in (``pl.at(..., allow_early_resolve=True)``).
  // Threaded onto the synthesised Submit below — same hint as
  // ``pl.submit(..., allow_early_resolve=True)``.
  bool scope_allow_early_resolve = op->GetAttr<bool>("allow_early_resolve", false);
  CHECK_SPAN(!deferred_wait.has_deferred_wait || !scope_allow_early_resolve, op->span_)
      << "pl.at(...) containing pld.system.defer_wait cannot use "
         "allow_early_resolve=True; the waiter's TaskId must remain unresolved until its "
         "registered signal condition is satisfied";
  // Dispatch predicate (``with pl.spmd(..., predicate=(t[i] > 0)):``). Rides
  // on the scope from parse through SSA (which versions the Vars inside it);
  // moved onto ``Submit::predicate_`` below, after which the attr is gone —
  // the field is the single source of truth, so it is deliberately NOT copied
  // into ``submit_attrs``.
  ExprPtr scope_predicate = op->GetAttr<ExprPtr>(kAttrPredicate, nullptr);

  // Dependency edges (or an early-resolve flag, or a dispatch predicate)
  // force the Submit shape: deps live in the typed ``Submit::deps_`` field,
  // the flag in ``Submit::allow_early_resolve_`` and the predicate in
  // ``Submit::predicate_`` — none has a plain-Call carrier. A scope written
  // without ``as tid`` gets a synthetic (unused) TaskId Var; DCE keeps the
  // Submit itself alive (task launches are effectful) and codegen skips the
  // unconsumed trailing tuple element.
  if ((!scope_dep_edges.empty() || scope_allow_early_resolve || scope_predicate) && !scope_task_id_var) {
    scope_task_id_var = std::make_shared<Var>(GenerateFreshSSAName("tid"),
                                              std::make_shared<ScalarType>(DataType::TASK_ID), op->span_);
    var_types_[scope_task_id_var.get()] = scope_task_id_var->GetType();
    var_objects_[scope_task_id_var.get()] = scope_task_id_var;
    known_names_.insert(scope_task_id_var->name_hint_);
  }

  std::vector<int32_t> arg_dir_override_indices;
  if (!scope_no_dep_vars.empty()) {
    arg_dir_override_indices.reserve(scope_no_dep_vars.size());
    for (const auto& v : scope_no_dep_vars) {
      INTERNAL_CHECK_SPAN(v, op->span_)
          << "Internal error: null Var in arg_direction_overrides_vars on outlined scope '"
          << outlined_func_name << "'";
      auto it = input_var_to_idx.find(v.get());
      CHECK(it != input_var_to_idx.end())
          << "pl.at(no_dep_args=[...]) references tensor '" << v->name_hint_
          << "', which is not captured by the scope body. Only tensors actually "
          << "read or written inside `with pl.at(...):` can appear in no_dep_args=.";
      arg_dir_override_indices.push_back(it->second);
    }
    // Sort so the attr is order-independent: ``DeriveCallDirections``
    // applies overrides as a set (writes to ``dirs[idx]`` are idempotent),
    // so two programs that differ only in user-provided / capture-order
    // ordering describe the same NoDep slot set. Sorted indices make
    // ``structural_equal``'s order-sensitive vector comparison report
    // semantic equality naturally and keep IR dumps deterministic.
    std::sort(arg_dir_override_indices.begin(), arg_dir_override_indices.end());
  }

  // Translate scope dump vars into the dispatch's selective-dump arg list.
  // The dispatch's ``args_`` are exactly ``call_args`` (the captured input
  // Vars), so the entry we record IS the Var object orchestration codegen
  // matches ``args_[i]`` against by identity. A tagged tensor the scope does
  // not capture is skipped (not an error) — it is simply a forward-sticky tag
  // that this particular scope never consumes as a kernel arg. Dedup by
  // identity; preserve arg order via ``input_var_to_idx`` ascending.
  std::vector<VarPtr> dump_call_args;
  if (!scope_dump_vars.empty()) {
    std::vector<int32_t> dump_indices;
    dump_indices.reserve(scope_dump_vars.size());
    std::unordered_set<int32_t> seen_idx;
    for (const auto& v : scope_dump_vars) {
      if (!v) continue;
      auto it = input_var_to_idx.find(v.get());
      if (it == input_var_to_idx.end()) continue;  // tensor not captured here
      if (seen_idx.insert(it->second).second) dump_indices.push_back(it->second);
    }
    std::sort(dump_indices.begin(), dump_indices.end());
    dump_call_args.reserve(dump_indices.size());
    for (int32_t idx : dump_indices) {
      if (auto cav = AsVarLike(call_args[static_cast<size_t>(idx)])) dump_call_args.push_back(cav);
    }
  }

  // Determine call return type. When ``task_id_var`` is set, append the
  // TaskId element at the tail of the (already flat) return-type tuple so
  // the augmentation matches the ``pl.submit(...)`` shape.
  std::vector<TypePtr> effective_return_types = return_types;
  if (scope_task_id_var) {
    effective_return_types.push_back(std::make_shared<ScalarType>(DataType::TASK_ID));
  }
  // Determine call return type. Single non-task-id return is unwrapped; the
  // task_id_var path always uses TupleType (even for the lone TASK_ID
  // element) so codegen's ``IsSubmitCall`` detection — which keys on a
  // trailing ``Scalar[TASK_ID]`` element of a ``TupleType`` — fires
  // uniformly.
  TypePtr call_return_type;
  if (effective_return_types.empty()) {
    call_return_type = nullptr;
  } else if (!scope_task_id_var && effective_return_types.size() == 1) {
    call_return_type = effective_return_types[0];
  } else {
    call_return_type = std::make_shared<TupleType>(effective_return_types);
  }

  // Build the synthesised call expression. When ``scope_task_id_var`` is
  // set (user-written ``as tid`` OR synthesized above for a deps-only
  // scope) we emit an ``ir.Submit`` matching the explicit ``pl.submit(...)``
  // surface: deps_ comes from the scope's ``kAttrManualDepEdges`` attr, and
  // only the non-deps attrs (dump_vars / arg_direction_overrides) land in
  // ``attrs_``. The dep-free path keeps the plain Call shape.
  ExprPtr synthesised_call_expr;
  if (scope_task_id_var) {
    std::vector<ExprPtr> submit_deps;
    submit_deps.reserve(scope_dep_edges.size());
    for (const auto& v : scope_dep_edges) {
      submit_deps.push_back(v);
    }
    std::vector<std::pair<std::string, std::any>> submit_attrs;
    // ``dump_vars`` first to mirror the parser's canonical Call attr order
    // (_make_call_with_return_type writes dump_vars before arg_directions),
    // so a print -> reparse of the synthesised dispatch round-trips.
    if (!dump_call_args.empty()) {
      submit_attrs.emplace_back(kAttrDumpVars, std::move(dump_call_args));
    }
    if (!arg_dir_override_indices.empty()) {
      submit_attrs.emplace_back(kAttrArgDirectionOverrides, std::move(arg_dir_override_indices));
    }
    // Launch spec in the first-class Submit fields — the same shape
    // ``pl.spmd_submit(..., core_num=N)`` produces, so printing / codegen /
    // verification take one path.
    synthesised_call_expr = std::make_shared<Submit>(
        global_var, call_args, std::move(submit_deps), std::vector<std::pair<std::string, std::any>>{},
        std::move(submit_attrs), call_return_type ? call_return_type : std::make_shared<UnknownType>(),
        op->span_,
        /*core_num=*/spmd_scope ? std::optional<ExprPtr>(spmd_scope->core_num_) : std::nullopt,
        /*sync_start=*/spmd_scope && spmd_scope->sync_start_,
        /*allow_early_resolve=*/scope_allow_early_resolve,
        /*predicate=*/scope_predicate ? std::optional<ExprPtr>(scope_predicate) : std::nullopt);
  } else {
    std::vector<std::pair<std::string, std::any>> call_attrs;
    // ``dump_vars`` first — see the Submit branch above for the ordering
    // rationale (matches _parse_kernel_call's reparse order).
    if (!dump_call_args.empty()) {
      call_attrs.emplace_back(kAttrDumpVars, std::move(dump_call_args));
    }
    if (!arg_dir_override_indices.empty()) {
      call_attrs.emplace_back(kAttrArgDirectionOverrides, std::move(arg_dir_override_indices));
    }
    // Launch spec last: the printer emits bespoke keys first and the rest in
    // ``attrs_`` order, and a reparse rebuilds them in printed order — so
    // appending keeps the dispatch structurally equal across print -> parse.
    if (spmd_scope) {
      call_attrs.emplace_back(kAttrCoreNum, ExprPtr(spmd_scope->core_num_));
      if (spmd_scope->sync_start_) {
        call_attrs.emplace_back(kAttrSyncStart, true);
      }
    }
    if (!call_attrs.empty()) {
      synthesised_call_expr = std::make_shared<Call>(
          global_var, call_args, std::vector<std::pair<std::string, std::any>>{}, std::move(call_attrs),
          call_return_type ? call_return_type : std::make_shared<UnknownType>(), op->span_);
    } else if (call_return_type) {
      synthesised_call_expr = std::make_shared<Call>(global_var, call_args, call_return_type, op->span_);
    } else {
      synthesised_call_expr = std::make_shared<Call>(global_var, call_args, op->span_);
    }
  }
  // Keep the original name (``call_expr``) for the rest of the function —
  // AssignStmt / EvalStmt accept any ExprPtr, so the type widening from
  // ``shared_ptr<Call>`` to ``ExprPtr`` is transparent at the use sites.
  const ExprPtr& call_expr = synthesised_call_expr;

  // Resolve the call-site Var for an output variable. Scope-defined vars come from
  // scope_var_collector; store targets (external tensors) fall back to the outer symbol table.
  // Store targets get a fresh SSA name to avoid re-assigning the input variable.
  auto resolve_call_site_var = [&](const VarPtr& out_var) -> VarPtr {
    bool is_store = store_output_set.count(out_var.get()) > 0;
    if (!is_store) {
      auto var_it = scope_var_collector.var_objects.find(out_var.get());
      if (var_it != scope_var_collector.var_objects.end()) {
        return var_it->second;
      }
      auto ext_it = var_objects_.find(out_var.get());
      CHECK(ext_it != var_objects_.end())
          << "Variable " << out_var->name_hint_ << " not found in var_objects";
      return ext_it->second;
    }
    auto ext_it = var_objects_.find(out_var.get());
    CHECK(ext_it != var_objects_.end()) << "Variable " << out_var->name_hint_ << " not found in var_objects";
    return CreateFreshStoreTargetVar(ext_it->second, op->span_);
  };

  // Create assignments for output variables in the parent function.
  // We assemble the result first, then register deferred post-store alias
  // renames once — each alias maps to a store target whose fresh var is
  // populated by resolve_call_site_var (via CreateFreshStoreTargetVar) into
  // store_target_renames_, so the registration must happen after all
  // resolve_call_site_var calls.
  StmtPtr result;
  if (scope_task_id_var) {
    // ``with pl.at(...) as tid:`` always goes through the temp+unpack path
    // even for ≤1 output: the producer TaskId is an extra trailing element
    // that needs its own ``TupleGetItem``, and we cannot bind it inline.
    auto ret_var =
        std::make_shared<Var>(auto_name::BuildName("ret", "", "tmp", 0), call_return_type, op->span_);
    std::vector<StmtPtr> stmts;
    stmts.push_back(std::make_shared<AssignStmt>(ret_var, call_expr, op->span_));
    for (size_t i = 0; i < output_vars.size(); ++i) {
      auto tuple_get = std::make_shared<TupleGetItemExpr>(ret_var, static_cast<int>(i), op->span_);
      auto output_var = resolve_call_site_var(output_vars[i]);
      stmts.push_back(std::make_shared<AssignStmt>(output_var, tuple_get, op->span_));
    }
    // The TaskId element sits at the flat tuple's trailing position.
    auto tid_get =
        std::make_shared<TupleGetItemExpr>(ret_var, static_cast<int>(output_vars.size()), op->span_);
    stmts.push_back(std::make_shared<AssignStmt>(scope_task_id_var, tid_get, op->span_));
    result = std::make_shared<SeqStmts>(stmts, op->span_);
  } else if (output_vars.empty()) {
    result = std::make_shared<EvalStmt>(call_expr, op->span_);
  } else if (output_vars.size() == 1) {
    auto output_var = resolve_call_site_var(output_vars[0]);
    result = std::make_shared<AssignStmt>(output_var, call_expr, op->span_);
  } else {
    // Assign call result to a temporary variable, then unpack with TupleGetItem
    auto ret_var =
        std::make_shared<Var>(auto_name::BuildName("ret", "", "tmp", 0), call_return_type, op->span_);
    std::vector<StmtPtr> stmts;
    stmts.push_back(std::make_shared<AssignStmt>(ret_var, call_expr, op->span_));
    for (size_t i = 0; i < output_vars.size(); ++i) {
      auto tuple_get = std::make_shared<TupleGetItemExpr>(ret_var, static_cast<int>(i), op->span_);
      auto output_var = resolve_call_site_var(output_vars[i]);
      stmts.push_back(std::make_shared<AssignStmt>(output_var, tuple_get, op->span_));
    }
    result = std::make_shared<SeqStmts>(stmts, op->span_);
  }

  // For each scope-local SSA post-store alias we elided from output_vars,
  // look up the already-renamed store target and map the alias body
  // pointer to that fresh var so later parent-function references resolve
  // correctly.
  for (const auto& [alias_ptr, target_ptr] : deferred_post_store_aliases) {
    auto rename_it = store_target_renames_.find(target_ptr);
    if (rename_it != store_target_renames_.end()) {
      store_target_renames_[alias_ptr] = rename_it->second;
    }
  }
  return result;
}

/**
 * @brief Generate a fresh SSA name by incrementing the numeric suffix.
 *
 * E.g. "buf_0" -> "buf_1", "x_2" -> "x_3".  Falls back to appending "_1".
 */
std::string ScopeOutliner::GenerateFreshSSAName(const std::string& original_name) const {
  std::unordered_set<std::string> used_names;
  for (const auto& [var, _] : var_types_) {
    used_names.insert(var->name_hint_);
  }
  return auto_name::GenerateFreshNameLike(original_name, used_names);
}

/**
 * @brief Create a fresh Var for a store-target output and register the rename.
 *
 * Registers the fresh Var in var_types_/var_objects_ and records the rename
 * in store_target_renames_ so subsequent statements (and the ReturnStmt)
 * resolve the store target to its new value.
 *
 * ``original_var`` is the *original* store-target Var, not a prior rename:
 * var_objects_ is kept as a pure identity symbol table (never rewritten with
 * call-site renames), so when N sibling scopes write the same target every
 * scope resolves it back to the same original and this overwrites the single
 * store_target_renames_ entry. The map therefore stays flat — one key, the
 * latest value — and call-site / ReturnStmt lookups need no chain chasing.
 */
VarPtr ScopeOutliner::CreateFreshStoreTargetVar(const VarPtr& original_var, const Span& span) {
  std::string fresh_name = GenerateFreshSSAName(original_var->name_hint_);
  auto type = original_var->GetType();
  auto fresh_var = std::make_shared<Var>(fresh_name, type, span);
  store_target_renames_[original_var.get()] = fresh_var;
  var_types_[fresh_var.get()] = type;
  var_objects_[fresh_var.get()] = fresh_var;
  known_names_.insert(fresh_name);
  return fresh_var;
}

/**
 * @brief Generate a naming suffix from hierarchy level and optional role.
 *
 * Produces lowercase suffixes like "_host_sub_worker_", "_global_orch_", "_chip_".
 */
std::string ScopeOutliner::GenerateHierarchySuffix(Level level, const std::optional<Role>& role) {
  std::string name = "_";
  switch (level) {
    case Level::UNDEFINED:
      name += "undefined";
      break;
    case Level::AIV:
      name += "aiv";
      break;
    case Level::AIC:
      name += "aic";
      break;
    case Level::CORE_GROUP:
      name += "core_group";
      break;
    case Level::CHIP_DIE:
      name += "chip_die";
      break;
    case Level::CHIP:
      name += "chip";
      break;
    case Level::HOST:
      name += "host";
      break;
    case Level::CLUSTER_0:
      name += "cluster0";
      break;
    case Level::CLUSTER_1:
      name += "cluster1";
      break;
    case Level::CLUSTER_2:
      name += "cluster2";
      break;
    case Level::GLOBAL:
      name += "global";
      break;
  }
  if (role.has_value()) {
    name += (role.value() == Role::Orchestrator) ? "_orch" : "_sub_worker";
  }
  return name + "_";
}

/// Infer parameter directions for the outlined function by examining the scope body.
///
/// Strategy:
///   0. Collect which captured vars the body *reads* — every use except an
///      operand the operator declares it purely overwrites. Conservative by
///      construction: an unrecognised use counts as a read, so the
///      classification can only err towards ``InOut``.
///   1. Mark every captured var the body writes. Which argument an operator
///      writes comes from its registry declaration (`set_arg_effect`), so a
///      scope writing through ``tile.mscatter``, ``tensor.write``,
///      ``pld.tile.put`` or any other declared writer is classified the same
///      way as one writing through ``tile.store``. Two sources feed this: the
///      exported ``store_output_set`` (targets that also become outputs) and a
///      scan of the body, which catches an SSA-pure writer such as
///      ``tensor.assemble`` whose result the caller rebinds — without it the
///      spmd wrapper for ``for n0 in pl.spmd(...): out = pl.assemble(out,
///      slice, [...])`` keeps direction In on the shared output and the
///      orchestration codegen drops the SSA-result alias for the call.
///   2. Merge ``Out``/``InOut`` directions from inner GlobalVar calls
///
/// A written param is ``InOut`` only when Step 0 also saw a read; a
/// write-only param is ``Out``. Claiming ``InOut`` for a param the body never
/// reads is not a conservative approximation — it is a false read that
/// survives all the way into ``DistributedCodegen::EmitCallToWorker``, which
/// tags each per-rank chip dispatch from the callee direction and so turns
/// disjoint per-rank slices of one ``pl.Out`` tensor into a cross-rank
/// dependency (issue #2415). Ordering that a write-only param genuinely needs
/// is not lost: ``DeriveCallDirections`` re-derives the *call-site* direction
/// and promotes a callee ``Out`` back to ``InOut`` under a sequential
/// ancestor, behind a prior writer of the same root, or when the root is an
/// enclosing ``InOut`` param.
std::vector<ParamDirection> ScopeOutliner::InferParamDirections(
    const std::vector<VarPtr>& input_vars, const StmtPtr& body,
    const std::unordered_set<const Var*>& store_output_set) const {
  std::vector<ParamDirection> directions(input_vars.size(), ParamDirection::In);

  // Build input_var pointer → index map (shared by every inference step below).
  std::unordered_map<const Var*, size_t> var_to_idx;
  for (size_t i = 0; i < input_vars.size(); ++i) {
    var_to_idx[input_vars[i].get()] = i;
  }

  // Step 0: which captured vars does the body read? A written param earns
  // ``InOut`` only when it is also read; write-only earns ``Out``.
  std::vector<bool> has_read(input_vars.size(), false);
  ParamReadCollector(var_to_idx, has_read, program_).VisitStmt(body);
  std::vector<ParamDirection> written_direction(input_vars.size());
  for (size_t i = 0; i < input_vars.size(); ++i) {
    written_direction[i] = has_read[i] ? ParamDirection::InOut : ParamDirection::Out;
  }

  // Step 1a: mark the exported write targets
  for (size_t i = 0; i < input_vars.size(); ++i) {
    if (store_output_set.count(input_vars[i].get())) {
      directions[i] = written_direction[i];
    }
  }

  // Step 1b: scan the body for writes the exported set does not carry. An
  // SSA-pure writer such as ``tensor.assemble`` returns a fresh Tensor, so it
  // never enters ``store_output_set``, yet its destination operand is the
  // caller's backing buffer and the result aliases it in place.
  class WrittenParamUpgrader : public IRVisitor {
   public:
    WrittenParamUpgrader(const std::unordered_map<const Var*, size_t>& var_to_idx,
                         std::vector<ParamDirection>& directions,
                         const std::vector<ParamDirection>& written_direction)
        : var_to_idx_(var_to_idx), directions_(directions), written_direction_(written_direction) {}

   protected:
    void VisitExpr_(const CallPtr& call) override {
      for (const auto& target : CallWriteTargets(call)) {
        auto it = var_to_idx_.find(target.var.get());
        if (it == var_to_idx_.end()) continue;
        directions_[it->second] =
            MergeParamDirection(directions_[it->second], written_direction_[it->second]);
      }
      IRVisitor::VisitExpr_(call);
    }

   private:
    const std::unordered_map<const Var*, size_t>& var_to_idx_;
    std::vector<ParamDirection>& directions_;
    const std::vector<ParamDirection>& written_direction_;
  };
  WrittenParamUpgrader(var_to_idx, directions, written_direction).VisitStmt(body);

  if (!program_) return directions;

  // Step 2: collect all GlobalVar function calls in the body and merge
  // ``Out``/``InOut`` directions from their callees onto our parameters.
  class CallFinder : public IRVisitor {
   public:
    std::vector<CallPtr> found_calls;
    void VisitExpr_(const CallPtr& call) override {
      Record(call);
      IRVisitor::VisitExpr_(call);
    }

    /// A task launch calls its callee just as a plain call does, and the base
    /// visitor's Submit handler does not forward here (see
    /// `.claude/rules/pass-submit-awareness.md`), so a `pl.submit` inside an
    /// outlined scope would otherwise contribute no callee direction at all.
    /// The view is transient — the merge below only reads its `op_` and
    /// `args_`, and it is never stored in the IR. Its arguments are a
    /// positional *prefix* of the callee's parameters; the merge is already
    /// bounded by both sizes.
    void VisitExpr_(const SubmitPtr& submit) override {
      Record(SubmitToCallView(submit));
      IRVisitor::VisitExpr_(submit);
    }

   private:
    void Record(const CallPtr& call) {
      if (std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
        found_calls.push_back(call);
      }
    }
  };

  CallFinder finder;
  finder.VisitStmt(body);
  if (finder.found_calls.empty()) return directions;

  // Accumulate what the callees prove about each capture as two independent
  // observations rather than folding one direction at a time.
  //
  // ``ParamDirection`` is not a lattice this evidence can be merged along.
  // ``In`` is the *no evidence yet* floor — ``directions`` is seeded with it —
  // so ``MergeParamDirection`` cannot read an ``In`` operand as "somebody read
  // this" without also promoting every write-only capture to ``InOut``, which
  // is the false read issue #2415 exists to prevent. Folding per call therefore
  // dropped a real read: a capture handed to one callee's ``In`` slot and
  // another's ``Out`` slot merged to ``In``, then to ``Out``, losing the first
  // observation. Kept apart, the two combine into the ``InOut`` such a capture
  // actually is, and a capture only ever written still comes out ``Out``.
  std::vector<bool> callee_reads(input_vars.size(), false);
  std::vector<bool> callee_writes(input_vars.size(), false);
  for (const auto& call : finder.found_calls) {
    auto gv = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (!gv) continue;
    auto callee = program_->GetFunction(gv->name_);
    if (!callee) continue;
    const auto& call_args = call->args_;
    const auto& callee_dirs = callee->param_directions_;
    for (size_t arg_idx = 0; arg_idx < call_args.size() && arg_idx < callee_dirs.size(); ++arg_idx) {
      // ``AsVarLike``: a loop-carried capture arrives as an ``IterArg``, which
      // has its own ``ObjectKind`` and so never matches ``As<Var>``
      // (`.claude/rules/ir-kind-traits.md`). Missing it pinned the wrapper
      // parameter to the seeded ``In`` no matter what the callee declared.
      auto arg_var = AsVarLike(call_args[arg_idx]);
      if (!arg_var) continue;
      auto it = var_to_idx.find(arg_var.get());
      if (it == var_to_idx.end()) continue;
      const ParamDirection callee_dir = callee_dirs[arg_idx];
      if (callee_dir == ParamDirection::In || callee_dir == ParamDirection::InOut) {
        callee_reads[it->second] = true;
      }
      if (callee_dir == ParamDirection::Out || callee_dir == ParamDirection::InOut) {
        callee_writes[it->second] = true;
      }
    }
  }

  // Only a write is new information here; a callee that merely reads its slot
  // leaves the capture where the body put it. The read half of the verdict
  // combines *both* sources: a slot some callee reads, and a read Step 0 saw in
  // this body. Step 0's answer is now trustworthy for this — it skips the
  // arguments a callee declares ``Out``, so ``has_read`` no longer counts the
  // argument pass itself and means what it says. Consulting only the callees
  // would drop the read in
  //
  //     value = pl.load(shared, ...)   # this body reads it
  //     self.overwrite(shared)         # and a callee overwrites it
  //
  // leaving ``shared`` ``Out`` and telling the wrapper it need not stage the
  // very contents ``pl.load`` consumes.
  for (size_t i = 0; i < input_vars.size(); ++i) {
    if (!callee_writes[i]) continue;
    const bool is_read = callee_reads[i] || has_read[i];
    directions[i] = MergeParamDirection(directions[i], is_read ? ParamDirection::InOut : ParamDirection::Out);
  }

  return directions;
}

}  // namespace outline_utils
}  // namespace ir
}  // namespace pypto

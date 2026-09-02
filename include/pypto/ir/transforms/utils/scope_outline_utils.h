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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_SCOPE_OUTLINE_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_SCOPE_OUTLINE_UTILS_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace outline_utils {

/** @brief Visitor to build a symbol table mapping variable pointers to their types and Var objects. */
/// One variable a call writes, and how.
struct CallWriteTarget {
  VarPtr var;        ///< the written variable (`Var` or `IterArg`)
  size_t slot;       ///< the argument index carrying it
  ArgEffect effect;  ///< `Write` for a pure overwrite, `ReadWrite` for an update
};

/// Every variable @p call writes through one of its arguments.
///
/// Which argument an operator writes is declared once on the registry
/// (`set_arg_effect`), so a new write operator reaches the outliner's read
/// analysis and its direction inference together instead of having to be added
/// to each of them. Before, both matched `tile.store` and `tensor.assemble` by
/// name, and a scope whose only write to a captured tensor was, say, a
/// `pld.tile.put` left that tensor looking untouched.
///
/// `AsVarLike` rather than `As<Var>`: a loop-carried destination is an `IterArg`
/// (`for ... : c = pl.store(t, off, c)`), and `As<Var>` does not match it (see
/// `.claude/rules/ir-kind-traits.md`).
[[nodiscard]] std::vector<CallWriteTarget> CallWriteTargets(const CallPtr& call);

class VarCollector : public IRVisitor {
 public:
  std::unordered_map<const Var*, TypePtr> var_types;
  std::unordered_map<const Var*, VarPtr> var_objects;
  std::unordered_set<std::string> known_names;

 protected:
  // Use VisitVarLike_ to collect both Var and IterArg references.
  // VisitExpr_(IterArgPtr) calls VisitVarLike_ then visits initValue_,
  // so IterArgs from outer loops are included in the symbol table.
  void VisitVarLike_(const VarPtr& op) override {
    var_types.try_emplace(op.get(), op->GetType());
    var_objects.try_emplace(op.get(), op);
    known_names.insert(op->name_hint_);
    IRVisitor::VisitVarLike_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    var_types[op->var_.get()] = op->var_->GetType();
    var_objects[op->var_.get()] = op->var_;
    known_names.insert(op->var_->name_hint_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    var_types[op->loop_var_.get()] = op->loop_var_->GetType();
    var_objects[op->loop_var_.get()] = op->loop_var_;
    known_names.insert(op->loop_var_->name_hint_);
    for (const auto& iter_arg : op->iter_args_) {
      var_types[iter_arg.get()] = iter_arg->GetType();
      var_objects[iter_arg.get()] = iter_arg;
      known_names.insert(iter_arg->name_hint_);
    }
    for (const auto& return_var : op->return_vars_) {
      var_types[return_var.get()] = return_var->GetType();
      var_objects[return_var.get()] = return_var;
      known_names.insert(return_var->name_hint_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& iter_arg : op->iter_args_) {
      var_types[iter_arg.get()] = iter_arg->GetType();
      var_objects[iter_arg.get()] = iter_arg;
      known_names.insert(iter_arg->name_hint_);
    }
    for (const auto& return_var : op->return_vars_) {
      var_types[return_var.get()] = return_var->GetType();
      var_objects[return_var.get()] = return_var;
      known_names.insert(return_var->name_hint_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& return_var : op->return_vars_) {
      var_types[return_var.get()] = return_var->GetType();
      var_objects[return_var.get()] = return_var;
      known_names.insert(return_var->name_hint_);
    }
    IRVisitor::VisitStmt_(op);
  }
};

// ============================================================================
// Parameterized scope outliner
// ============================================================================

/**
 * @brief Mutator to outline scopes of a given ScopeKind into separate functions.
 *
 * Parameterized by the target ScopeKind, the FunctionType for outlined functions,
 * and a naming suffix. Handles SeqStmts specially to determine which scope-defined
 * variables are actually used after each scope (output filtering), and recursively
 * transforms scope bodies to handle nested scopes.
 */
class ScopeOutliner : public IRMutator {
 public:
  ScopeOutliner(std::string func_name, const std::unordered_map<const Var*, TypePtr>& var_types,
                const std::unordered_map<const Var*, VarPtr>& var_objects,
                const std::unordered_set<std::string>& known_names, ScopeKind target_scope_kind,
                FunctionType outlined_func_type, std::string name_suffix, ProgramPtr program = nullptr,
                std::shared_ptr<std::unordered_set<std::string>> reserved_func_names = nullptr);

  [[nodiscard]] const std::vector<FunctionPtr>& GetOutlinedFunctions() const { return outlined_functions_; }

 protected:
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
  ExprPtr VisitExpr_(const VarPtr& op) override;

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
  [[nodiscard]] std::unordered_set<const Var*> ComputeFallbackUsedAfter(const ScopeStmtPtr& scope) const;

  /**
   * @brief Process SeqStmts to analyze scope outputs using subsequent statements.
   *
   * For each target scope, collects variables referenced in all subsequent statements
   * plus any variables required by a parent scope (propagated via required_outputs_).
   */
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override;

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
  StmtPtr VisitScopeKind(const std::shared_ptr<const ScopeT>& op);

  StmtPtr VisitStmt_(const InCoreScopeStmtPtr& op) override;
  StmtPtr VisitStmt_(const ClusterScopeStmtPtr& op) override;
  StmtPtr VisitStmt_(const HierarchyScopeStmtPtr& op) override;
  StmtPtr VisitStmt_(const SpmdScopeStmtPtr& op) override;
  // SplitAiv is never an outline target (target is always InCore), so this
  // descends into the body via VisitScopeKind's non-target branch, preserving
  // the nested SplitAivScopeStmt inside the outlined InCore function body.
  StmtPtr VisitStmt_(const SplitAivScopeStmtPtr& op) override;
  // Graph is an outline target of its own pass (OutlineGraphScopes); for every
  // other outliner this descends via VisitScopeKind's non-target branch, so a
  // Graph region nested around, say, an InCore scope keeps its wrapper.
  StmtPtr VisitStmt_(const GraphScopeStmtPtr& op) override;

  /**
   * @brief Thread store-target renames made inside a control-flow body out of it.
   *
   * A scope that writes a captured tensor is replaced by a call whose result is
   * bound to a *fresh* SSA name (see CreateFreshStoreTargetVar). When that scope
   * sits inside a loop or an ``if``, the fresh Var is bound inside the body and
   * is therefore out of scope afterwards — but ``store_target_renames_`` is flat
   * and function-wide, so without these overrides every later reference to the
   * store target still resolves to that body-local Var. The result reads as
   * ``for ...: t__ssa_v1 = self.k_incore_0(..., t__ssa_v0)`` followed by a use of
   * ``t__ssa_v1`` after the loop: a dangling reference that ``SSAVerify`` rejects
   * ("used outside its defining scope").
   *
   * Nothing miscompiles today only because every SSA version of an orchestration
   * tensor denotes the same GM buffer. The def-use edge is still wrong, so each
   * override rebuilds the statement with a real carry: the value on entry seeds a
   * new ``IterArg``, the body yields the fresh Var, and a new ``return_var``
   * becomes the value visible afterwards. ``ClassifyIterArgCarry`` (pass 47) sees
   * the yield in the iter_arg's alias class (the Out-call and TupleGetItem rules)
   * and marks the carry *trivial*, so codegen is unchanged.
   */
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override;
  StmtPtr VisitStmt_(const ForStmtPtr& op) override;
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override;
  StmtPtr VisitStmt_(const IfStmtPtr& op) override;

 private:
  /// One store target that a control-flow body renamed, and the values that
  /// bracket the body.
  struct BodyStoreRename {
    VarPtr original;                  ///< the ``store_target_renames_`` key (always the *original* Var)
    VarPtr seed;                      ///< the value current on entry to the body — the carry's init
    std::vector<VarPtr> body_values;  ///< values bound inside the body, in order; the last is yielded
  };

  /// Store-target renames and definitions made directly in one control-flow body.
  struct BodyRenameFrame {
    std::vector<BodyStoreRename> renames;
    std::unordered_set<const Var*> local_defs;
  };

  /// Record vars that become visible in the enclosing body after a child control-flow statement.
  void NoteLocalDefinitions(const std::vector<VarPtr>& vars);

  /// Record @p fresh as the current value of store target @p original.
  ///
  /// When a control-flow body is open, the rename is also noted on the innermost
  /// frame so that body can thread it out as a carry, unless @p original was
  /// defined in that same body and is therefore local to each execution of it.
  /// @p seed is the value current *before* the rename; a target renamed by N
  /// sibling scopes in one body keeps the first seed and appends each value,
  /// because the intermediate ones may still be held by a post-store alias entry
  /// that the publish sweep has to retarget too.
  void NoteStoreTargetRename(const VarPtr& original, const VarPtr& seed, const VarPtr& fresh);

  /// Visit @p body with a fresh control-flow frame open, collecting into @p renames
  /// every store target the body renamed.
  StmtPtr VisitControlFlowBody(const StmtPtr& body, std::vector<BodyStoreRename>* renames);

  /// One fresh ``IterArg`` per rename, seeded with that rename's entry value.
  [[nodiscard]] std::vector<IterArgPtr> MakeCarryIterArgs(const std::vector<BodyStoreRename>& renames,
                                                          const Span& span);

  /// One fresh return ``Var`` per rename — the value visible after the statement.
  [[nodiscard]] std::vector<VarPtr> MakeCarryReturnVars(const std::vector<BodyStoreRename>& renames,
                                                        const Span& span);

  /// Rebind @p body onto @p carry_iter_args and append each rename's last value
  /// to its trailing yield (adding one when the loop carried nothing before).
  ///
  /// The seed is defined *outside* the loop, so every reference to it inside the
  /// body means "this buffer" — which is exactly what the carry now names.
  /// @p total_carries is the expected post-append yield arity, checked here.
  [[nodiscard]] StmtPtr BuildCarriedLoopBody(const StmtPtr& body, const std::vector<BodyStoreRename>& renames,
                                             const std::vector<IterArgPtr>& carry_iter_args,
                                             size_t total_carries, const Span& span) const;

  /// Record @p fresh as the current value of store target @p original, keeping
  /// ``renamed_by_value_`` in step. Every write to ``store_target_renames_``
  /// goes through here.
  void SetStoreTargetRename(const VarPtr& original, const VarPtr& fresh);

  /// Undo @p renames, putting each store target back to the value its body was
  /// entered with. Used between the two arms of an ``if``, which are
  /// alternatives rather than a sequence.
  void RewindRenames(const std::vector<BodyStoreRename>& renames);

  /// Point every ``store_target_renames_`` entry holding one of @p renames'
  /// body-local values at the matching entry of @p visible_values.
  ///
  /// Keyed by *value*, not by store target: OutlineScope also registers
  /// scope-local post-store aliases that name the same store target, and those
  /// entries hold a body-local Var that is equally out of scope afterwards.
  /// Reached through ``renamed_by_value_`` rather than by scanning the map, so
  /// the cost is proportional to the entries that actually move.
  void RetargetBodyValues(const std::vector<BodyStoreRename>& renames,
                          const std::vector<VarPtr>& visible_values);

  /// RetargetBodyValues, plus pointing each store target itself at its
  /// @p visible_values entry.
  void RetargetCarries(const std::vector<BodyStoreRename>& renames,
                       const std::vector<VarPtr>& visible_values);

  /// RetargetCarries, plus telling the enclosing control-flow frame (if any)
  /// about each carry, so a nested loop's carry is threaded on out of the outer
  /// loop too.
  void PublishCarries(const std::vector<BodyStoreRename>& renames, const std::vector<VarPtr>& return_vars);

  /// Split @p renames into the ones this loop must newly carry (@p fresh) and the
  /// ones it already carries (@p carried, with their post-loop @p carried_values).
  ///
  /// A store target that *is* one of the loop's own iter_args is already threaded
  /// through the loop; giving it a second carry seeded with itself would make the
  /// loop carry its own carry, and the seed would not even be in scope at the
  /// loop header. Its post-loop value is simply the matching return_var.
  void SplitAlreadyCarried(const std::vector<BodyStoreRename>& renames,
                           const std::vector<IterArgPtr>& iter_args,
                           const std::vector<IterArgPtr>& visited_iter_args,
                           const std::vector<VarPtr>& return_vars, const Span& span,
                           std::vector<BodyStoreRename>* fresh, std::vector<BodyStoreRename>* carried,
                           std::vector<VarPtr>* carried_values) const;

  /// True when `name` is already claimed by this function (`known_names_`) or,
  /// when the pass opts in, by any earlier function in the program
  /// (`reserved_func_names_`).
  [[nodiscard]] bool IsNameTaken(const std::string& name) const;

  /// Append the smallest `_<n>` suffix that makes `base` unique program-wide.
  [[nodiscard]] std::string NumericSuffix(const std::string& base) const;

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
  [[nodiscard]] std::string DisambiguateOutlinedName(const std::string& candidate) const;

  /**
   * @brief Outline a single scope into a separate function.
   *
   * @param op The scope statement to outline
   * @param used_after Variables (by pointer) used in subsequent statements (determines outputs)
   */
  StmtPtr OutlineScope(const ScopeStmtPtr& op, const std::unordered_set<const Var*>& used_after);

  /**
   * @brief Generate a fresh SSA name by incrementing the numeric suffix.
   *
   * E.g. "buf_0" -> "buf_1", "x_2" -> "x_3".  Falls back to appending "_1".
   *
   * @param role Optional SSA role to stamp instead of the name's current one —
   *             ``"iter"`` for a loop carry, ``"rv"`` for a return var, matching
   *             what ConvertToSSA emits. Empty keeps the existing role.
   */
  [[nodiscard]] std::string GenerateFreshSSAName(const std::string& original_name,
                                                 const std::string& role = "") const;

  /// Register @p var in the symbol table so later fresh-name generation avoids it.
  void RegisterVar(const VarPtr& var);

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
  VarPtr CreateFreshStoreTargetVar(const VarPtr& original_var, const Span& span);

  /**
   * @brief Generate a naming suffix from hierarchy level and optional role.
   *
   * Produces lowercase suffixes like "_host_sub_worker_", "_global_orch_", "_chip_".
   */
  static std::string GenerateHierarchySuffix(Level level, const std::optional<Role>& role);

  /// Infer parameter directions for the outlined function by examining the scope body.
  ///
  /// Strategy:
  ///   0. Collect which captured vars the body *reads* — every use except the
  ///      two write-destination operand slots (``tile.store``'s target and
  ///      ``tensor.assemble``'s destination). Conservative by construction: an
  ///      unrecognised use counts as a read, so the classification can only err
  ///      towards ``InOut``.
  ///   1. Mark tile.store targets (from ``store_output_set``) as written
  ///   2. Mark tensor.assemble destinations as written (``tensor.assemble`` is
  ///      SSA-pure but its first arg is a destination the result aliases in
  ///      place; without this the spmd wrapper for
  ///      ``for n0 in pl.spmd(...): out = pl.assemble(out, slice, [...])``
  ///      keeps direction In on the shared output and the orchestration
  ///      codegen drops the SSA-result alias for the call)
  ///   3. Merge ``Out``/``InOut`` directions from inner GlobalVar calls
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
  [[nodiscard]] std::vector<ParamDirection> InferParamDirections(
      const std::vector<VarPtr>& input_vars, const StmtPtr& body,
      const std::unordered_set<const Var*>& store_output_set) const;

  std::string func_name_;
  std::unordered_map<const Var*, TypePtr> var_types_;
  std::unordered_map<const Var*, VarPtr> var_objects_;
  std::unordered_set<std::string> known_names_;
  std::unordered_set<const Var*> required_outputs_;
  /// Accumulates across scopes intentionally (not saved/restored like func_name_
  /// etc.) so that subsequent scopes and statements see the renamed variables.
  ///
  /// Flat and function-wide, so an entry outlives the control-flow body that
  /// created it. The ForStmt / WhileStmt / IfStmt overrides are what keep that
  /// safe: they retarget each entry onto a value the following statements can
  /// actually see. Do not write this map directly from a path that can run
  /// inside a control-flow body — go through NoteStoreTargetRename so the
  /// enclosing frame learns the rename and can thread it out.
  std::unordered_map<const Var*, VarPtr> store_target_renames_;
  /// Reverse index over ``store_target_renames_``: which keys currently resolve
  /// to a given Var. Lets a control-flow node retarget the entries that name a
  /// value going out of scope without scanning the whole (ever-growing) map,
  /// which would make the pass quadratic in the number of control-flow nodes
  /// (.claude/rules/pass-complexity.md). Buckets may hold stale keys — a key
  /// retargeted since is skipped on the next visit — so every read re-checks
  /// the forward map.
  std::unordered_map<const Var*, std::vector<const Var*>> renamed_by_value_;
  /// Open control-flow frames, innermost last. Empty at the function's top level.
  std::vector<BodyRenameFrame> body_rename_stack_;
  ScopeKind target_scope_kind_;
  FunctionType outlined_func_type_;
  std::string name_suffix_;
  ProgramPtr program_;
  /// Program-wide set of outlined function names already emitted by earlier
  /// functions in the same pass run. Shared (non-owning of the pass) so that
  /// duplicate `name_hint` values across functions auto-disambiguate instead of
  /// colliding at Program construction (issue #1711). Null when the pass does
  /// not opt in to cross-function name reservation.
  std::shared_ptr<std::unordered_set<std::string>> reserved_func_names_;
  int scope_counter_ = 0;
  /// True while we are visiting the body of a non-target ScopeStmt — a
  /// target-kind scope encountered here cannot leak locally-defined vars to
  /// any surrounding context (scope boundaries confine them), so the
  /// used_after fallback exposes only store targets.
  bool inside_nested_scope_body_ = false;
  std::vector<FunctionPtr> outlined_functions_;
};

// ============================================================================
// Property verifier: ScopeKind absence
// ============================================================================

/// Verifies that a given ScopeKind does not appear in an IR subtree.
///
/// Used by outline passes to confirm that all scopes of a particular kind
/// have been successfully outlined into separate functions.
///
/// Usage:
///   ScopeKindAbsenceVerifier<ScopeKind::InCore> verifier(diagnostics, "PassName", "error message");
///   verifier.VisitStmt(func->body_);
template <ScopeKind Kind>
class ScopeKindAbsenceVerifier : public IRVisitor {
 public:
  ScopeKindAbsenceVerifier(std::vector<Diagnostic>& diagnostics, std::string pass_name, std::string message)
      : diagnostics_(diagnostics), pass_name_(std::move(pass_name)), message_(std::move(message)) {}

  template <typename ScopeT>
  void CheckKind(const std::shared_ptr<const ScopeT>& op) {
    if (!op) return;
    if (op->GetScopeKind() == Kind) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, pass_name_, 0, message_, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const InCoreScopeStmtPtr& op) override { CheckKind(op); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { CheckKind(op); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { CheckKind(op); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override { CheckKind(op); }
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override { CheckKind(op); }
  void VisitStmt_(const GraphScopeStmtPtr& op) override { CheckKind(op); }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string pass_name_;
  std::string message_;
};

}  // namespace outline_utils
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SCOPE_OUTLINE_UTILS_H_

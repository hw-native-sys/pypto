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

 private:
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
   */
  [[nodiscard]] std::string GenerateFreshSSAName(const std::string& original_name) const;

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
  std::unordered_map<const Var*, VarPtr> store_target_renames_;
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

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string pass_name_;
  std::string message_;
};

}  // namespace outline_utils
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SCOPE_OUTLINE_UTILS_H_

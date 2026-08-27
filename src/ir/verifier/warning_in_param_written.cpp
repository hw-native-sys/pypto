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

/**
 * @file warning_in_param_written.cpp
 * @brief Warn when a parameter declared `In` is written by its own function body.
 *
 * Direction inference builds a write set from two declarations: an operator's
 * registry effects and a callee's own `param_directions_`. This check reads the
 * same two and reports where they contradict a parameter's `In`. It is a
 * consistency check over the *declared* write semantics, not an independent
 * discovery of them — an operator that declared no effect contributes no write
 * here, so the undeclared case that motivated the work (`pld.system.notify`
 * before #2391, `tile.mscatter` before it was classified) is invisible to it.
 *
 * It runs `PostPipeline` (`DiagnosticCheck::InParamWritten`) rather than after
 * any one pass: a Group/Spmd wrapper's signature legitimately reads `In` for a
 * parameter its inner kernel writes, until `DeriveCallDirections` (pass 37)
 * materialises the effective directions back into the IR.
 *
 * **Best-effort, and deliberately not an `IRProperty`.** `InitMemRef` (pass 31)
 * declares `.invalidated = {IRProperty::SSAForm}` and nothing re-establishes it,
 * so the IR here is not in SSA form — and since pass 37 is the earliest this can
 * run, no pipeline position satisfies both. The buffer lineage below is a single
 * environment with no merging at a join, which is exact only when each name has
 * one definition. Without that it is wrong in both directions:
 *
 *   - a branch that re-points a name leaks its lineage past the join, so a write
 *     after the branch can be attributed to a buffer it reaches on no path; and
 *   - `BufferRootCollector` scans the whole body up front, so a rebound name has
 *     one final mapping that is applied to earlier writes too.
 *
 * Reports are therefore a signal to go and look, not a proof of a defect, and
 * silence proves nothing at all. Naming this a *sound* property would have
 * promised what its placement cannot deliver.
 *
 * What it does not invent is a write: both sources are declarations already in
 * the IR, and a variable whose owning buffer control flow leaves ambiguous is
 * skipped rather than blamed.
 *
 * A write reaching its parameter through a zero-copy view — `tile.slice`,
 * `tensor.slice`, ... — is resolved here, from the `ResultAliasedArgIndex`
 * contract and `op_predicates::IsBufferAliasingViewOp`. `BufferRootCollector`
 * maps a slice to a fresh root and three other passes share it, so this resolves
 * the chain rather than widening that analysis. Lineage is not carried across a
 * phi (`return_vars_` / `iter_args_`).
 */

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/buffer_root_collector.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/result_alias_utils.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

using ::pypto::ir::buffer_root::AmbiguousRootPolicy;
using ::pypto::ir::buffer_root::BufferRootCollector;
using ::pypto::ir::outline_utils::CallWriteTargets;

/// Report every write that reaches a parameter its function declares `In`.
class InParamWriteFinder : public IRVisitor {
 public:
  InParamWriteFinder(const ProgramPtr& program, const FunctionPtr& func,
                     const std::unordered_map<const Var*, const Var*>& buffer_roots,
                     const std::unordered_set<const Var*>& ambiguous, std::vector<Diagnostic>& diagnostics)
      : program_(program),
        func_(func),
        buffer_roots_(buffer_roots),
        ambiguous_(ambiguous),
        diagnostics_(diagnostics) {
    for (size_t i = 0; i < func->params_.size() && i < func->param_directions_.size(); ++i) {
      if (func->param_directions_[i] != ParamDirection::In) continue;
      // Only a buffer can be written through. A scalar parameter is passed by
      // value and its direction carries no aliasing claim.
      //
      // ``As<ShapedType>`` rather than ``AsTensorTypeLike``: the latter matches
      // ``TensorType`` and ``DistributedTensorType`` only, and says so — "Use
      // As<ShapedType>() when you want the wider union". A ``Tile`` parameter
      // may be declared ``Out``/``InOut`` and is written by ops that declare a
      // ``ReadWrite`` effect on it, so filtering it out here made a whole
      // parameter kind invisible to a check whose entire job is to notice
      // exactly that. ``ArrayType`` is in the union for the same reason.
      if (!As<ShapedType>(func->params_[i]->GetType())) continue;
      in_params_.emplace(func->params_[i].get(), func->params_[i]);
    }
  }

  [[nodiscard]] bool HasInParams() const { return !in_params_.empty(); }

 protected:
  void VisitStmt_(const AssignStmtPtr& assign) override {
    // Record `lhs = <view>(src, ...)` before walking the value, so a write in
    // the same statement already resolves through it.
    //
    // The source is resolved to its own root *here*, not at lookup time. Two
    // things follow. Lookup becomes a single map read, so the walk stays linear
    // in the body rather than O(views x writes) — the `.claude/rules/
    // pass-complexity.md` bound. And a chain collapses as it is built, so no
    // cycle can form for a later traversal to spin on.
    //
    // Every binding updates its own lineage, including by erasing it. These
    // verifier tests run on pre-SSA IR where a rebind reuses the same `Var`, so
    // a stale entry survives an intervening non-view assignment and blames a
    // buffer the variable no longer names:
    //
    //     t = pl.tile.slice(src1, ...)        # t names src1's buffer
    //     t = pl.tile.transpose(src2, 0, 1)   # ... and now it does not
    //     t = pl.tile.assemble(t, patch, ...) # writes t's own buffer, not src1
    // Read the lineage the value inherits *before* walking it, and apply the
    // rebinding *after*. The right-hand side is evaluated in the environment
    // that precedes the assignment, so a statement that both writes through a
    // view and rebinds the same name — `view = tensor.assemble(view, ...)` —
    // must still see the lineage the old `view` carried.
    const Var* resolved = assign->var_ ? SourceBufferOf(assign->value_) : nullptr;

    IRVisitor::VisitStmt_(assign);

    if (assign->var_) {
      // A self-view (`t = tile.assemble(t, ...)` on a `t` that names nothing
      // else) adds no lineage; recording it would only make the map cyclic.
      if (resolved != nullptr && resolved != assign->var_.get()) {
        view_sources_[assign->var_.get()] = resolved;
      } else {
        view_sources_.erase(assign->var_.get());
      }
    }
  }

  /// The buffer @p value names on the way out, already resolved, or null when it
  /// names a buffer of its own.
  ///
  /// Two declarations answer, and they are the same two the rest of the tree
  /// reads. `ResultAliasedArgIndex` is the result-alias contract — the operator
  /// returns the argument it updated (`tensor.assemble`, `tensor.write`, the
  /// collectives). `IsBufferAliasingViewOp` covers the zero-copy views, which
  /// declare no reuse contract because they update nothing: they reinterpret
  /// argument 0 and hand it back.
  [[nodiscard]] const Var* SourceBufferOf(const ExprPtr& value) const {
    auto call = As<Call>(value);
    if (!call || !call->op_ || call->args_.empty()) return nullptr;
    size_t index = 0;
    if (auto aliased = ResultAliasedArgIndex(call)) {
      index = *aliased;
    } else if (!op_predicates::IsBufferAliasingViewOp(call->op_->name_)) {
      return nullptr;
    }
    if (index >= call->args_.size()) return nullptr;
    auto source = AsVarLike(call->args_[index]);
    if (!source) return nullptr;
    auto it = view_sources_.find(source.get());
    return it == view_sources_.end() ? source.get() : it->second;
  }

  void VisitExpr_(const CallPtr& call) override {
    CheckCallLike(call, call->span_);
    IRVisitor::VisitExpr_(call);
  }

  /// A task launch writes its callee's Out/InOut parameters just as a call
  /// does; the base visitor does not forward Submit to the Call handler.
  void VisitExpr_(const SubmitPtr& submit) override {
    CheckCallLike(SubmitToCallView(submit), submit->span_);
    IRVisitor::VisitExpr_(submit);
  }

 private:
  void CheckCallLike(const CallPtr& call, const Span& span) {
    // Builtin operator: the registry says which arguments it writes.
    for (const auto& target : CallWriteTargets(call)) {
      Report(target.var, call->op_->name_, /*writer_is_operator=*/true, span);
    }

    // Cross-function call: the callee's signature says which of its parameters
    // it writes, and an argument in such a slot is written by this body too.
    auto gvar = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (!gvar || !program_) return;
    auto callee = program_->GetFunction(gvar->name_);
    if (!callee) return;
    // Submit carries a positional prefix of the callee's parameters; both kinds
    // map args_[i] to params_[i] identically over the args they do carry.
    for (size_t i = 0; i < call->args_.size() && i < callee->param_directions_.size(); ++i) {
      if (callee->param_directions_[i] == ParamDirection::In) continue;
      if (auto var = AsVarLike(call->args_[i])) {
        Report(var, gvar->name_, /*writer_is_operator=*/false, span);
      }
    }
  }

  /// The name the user wrote, without the SSA version suffix.
  ///
  /// This check must run PostPipeline (a wrapper's signature legitimately lags
  /// its kernel's until `DeriveCallDirections`), so unlike the PrePipeline
  /// warnings it only ever sees `out__ssa_v0`. Naming the parameter back is the
  /// whole point of the message: the reader has to find it in their own source.
  [[nodiscard]] static std::string SourceName(const std::string& name_hint) {
    auto pos = name_hint.rfind("__ssa_v");
    return pos == std::string::npos ? name_hint : name_hint.substr(0, pos);
  }

  /// @param writer_is_operator  true when the write came from a builtin's
  ///        registry effects, false when it came from a callee's own
  ///        `param_directions_`. Only the wording differs — the fix is the same
  ///        one in both cases, and it is *not* "declare the effect": a builtin
  ///        reaches here precisely *because* its effect is declared (that is how
  ///        `CallWriteTargets` found the write), and a callee is a user function
  ///        with no `REGISTER_OP` block to edit at all.
  /// Follow zero-copy view bindings back to the buffer they name.
  ///
  /// ``BufferRootCollector`` records nothing for a builtin view — it maps
  /// ``tensor.slice`` to a *fresh* root and skips the tile views entirely — so a
  /// write through one reached no parameter at all:
  ///
  ///     view = pl.tile.slice(acc, [8, 128], [0, 0])   # acc declared In
  ///     view = pl.tile.assemble(view, src, [0, 0])    # writes acc's buffer
  ///
  /// Resolving it here rather than in the collector is deliberate: three other
  /// passes share that analysis, and widening what counts as an alias for all of
  /// them is a separate change with its own blast radius.
  ///
  /// The view set is decided by ``op_predicates::IsBufferAliasingViewOp`` —
  /// ``OutputMemoryInheritsInput() && IsInplaceSafe()``, the shared registry
  /// read that ``InitMemRef`` and the tpop lifetime analysis already use. It
  /// excludes ``tile.transpose``, which inherits the memory *space* but permutes
  /// into a fresh buffer, and it will exclude any future inherit-input op
  /// registered ``not_inplace_safe()`` without an edit here.
  /// One map read: entries are recorded already resolved (see `VisitStmt_`).
  [[nodiscard]] const Var* ResolveThroughViews(const Var* var) const {
    auto it = view_sources_.find(var);
    return it == view_sources_.end() ? var : it->second;
  }

  void Report(const VarPtr& written, const std::string& writer, bool writer_is_operator, const Span& span) {
    // Resolve the written variable to the buffer it owns, so a write through a
    // slice or a loop-carried alias is attributed to the parameter behind it.
    if (ambiguous_.count(written.get()) > 0) return;
    const Var* viewed = ResolveThroughViews(written.get());
    if (ambiguous_.count(viewed) > 0) return;
    auto root_it = buffer_roots_.find(viewed);
    const Var* root = root_it == buffer_roots_.end() ? viewed : root_it->second;

    auto param_it = in_params_.find(root);
    if (param_it == in_params_.end()) return;
    if (!reported_.insert(root).second) return;

    diagnostics_.emplace_back(
        DiagnosticSeverity::Error, "InParamWritten", 0,
        "parameter '" + SourceName(param_it->second->name_hint_) + "' of function '" + func_->name_ +
            "' is declared In but is written by " +
            (writer_is_operator ? "operator '" + writer +
                                      "', which declares that write in its "
                                      "registry effects"
                                : "'" + writer + "', which declares that parameter Out or InOut") +
            ". A written parameter read as an input drops the dependency edge its writer needs, so "
            "the program races or deadlocks on device instead of failing here. Declare the "
            "parameter pl.Out (written, never read) or pl.InOut (read and written)",
        span);
  }

  const ProgramPtr& program_;
  const FunctionPtr& func_;
  const std::unordered_map<const Var*, const Var*>& buffer_roots_;
  const std::unordered_set<const Var*>& ambiguous_;
  std::vector<Diagnostic>& diagnostics_;
  std::unordered_map<const Var*, VarPtr> in_params_;
  /// `lhs` of a zero-copy view binding -> the variable it views.
  std::unordered_map<const Var*, const Var*> view_sources_;
  /// One diagnostic per parameter: a loop writing it every iteration is one bug.
  std::unordered_set<const Var*> reported_;
};

}  // namespace

class InParamWrittenWarningVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "InParamWritten"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gvar, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // Only signatures the compiler derived. An Orchestration function is the
      // program's entry: its directions are the user's declaration and its
      // parameters are the host ABI, so flipping one is a migration the user
      // makes, not an inference the compiler completes. Under-declaring there
      // is worth a warning (a written buffer the host is never told to copy
      // back), but it is a different diagnostic with a different audience — and
      // making it an error here would reject working programs.
      if (func->func_type_ == FunctionType::Orchestration) continue;

      BufferRootCollector roots(program, AmbiguousRootPolicy::kSkip);
      roots.Initialize(func->params_);
      roots.VisitStmt(func->body_);

      InParamWriteFinder finder(program, func, roots.buffer_roots, roots.ambiguous_buffer_vars, diagnostics);
      if (!finder.HasInParams()) continue;
      finder.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateInParamWrittenWarningVerifier() {
  return std::make_shared<InParamWrittenWarningVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

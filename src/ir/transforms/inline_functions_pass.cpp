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
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/result_alias_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

// =============================================================================
// Cycle detection in the Inline → Inline call graph
// =============================================================================

class CalledInlineCollector : public IRVisitor {
 public:
  explicit CalledInlineCollector(const std::unordered_set<std::string>& inline_names)
      : inline_names_(inline_names) {}

  void VisitExpr_(const CallPtr& op) override {
    if (op) {
      if (auto gv = As<GlobalVar>(op->op_)) {
        if (inline_names_.count(gv->name_) > 0) {
          called_.insert(gv->name_);
        }
      }
    }
    IRVisitor::VisitExpr_(op);
  }

  std::unordered_set<std::string> called_;

 private:
  const std::unordered_set<std::string>& inline_names_;
};

void DetectInlineCycles(const std::unordered_map<std::string, FunctionPtr>& inline_fns) {
  std::unordered_set<std::string> inline_names;
  for (const auto& [n, _] : inline_fns) inline_names.insert(n);

  std::unordered_map<std::string, std::unordered_set<std::string>> graph;
  for (const auto& [name, fn] : inline_fns) {
    CalledInlineCollector collector(inline_names);
    collector.VisitStmt(fn->body_);
    graph[name] = std::move(collector.called_);
  }

  enum class Color { White, Gray, Black };
  std::unordered_map<std::string, Color> color;
  for (const auto& [n, _] : inline_fns) color[n] = Color::White;
  std::vector<std::string> stack;

  std::function<void(const std::string&)> dfs = [&](const std::string& u) {
    color[u] = Color::Gray;
    stack.push_back(u);
    for (const auto& v : graph[u]) {
      if (color[v] == Color::Gray) {
        std::string cycle;
        bool started = false;
        for (const auto& s : stack) {
          if (s == v) started = true;
          if (started) cycle += s + " -> ";
        }
        cycle += v;
        throw pypto::ValueError("Cycle detected in FunctionType::Inline call graph: " + cycle);
      }
      if (color[v] == Color::White) dfs(v);
    }
    stack.pop_back();
    color[u] = Color::Black;
  };

  for (const auto& [n, _] : inline_fns) {
    if (color.at(n) == Color::White) dfs(n);
  }
}

// =============================================================================
// Splice an inline call site
// =============================================================================

// Process-wide counter ensuring distinct fresh names across multiple call
// sites of the same inline function. Pass execution is sequential, so a
// plain static int suffices.
//
// `__` is reserved by the IR auto-naming utility (see
// auto_name_utils.h::ValidateBaseName) for its own `name__role__version`
// scheme; re-using `__` here trips the validator when downstream passes rename
// the inlined Vars again.
//
// A single-underscore *suffix* is not enough on its own: `orig` may already end
// in one — `_` is Python's throwaway name and the documented loop variable of
// `for _ in pl.split_aiv(...)` — in which case plain concatenation *creates* the
// reserved delimiter out of two individually-legal halves. JoinNameSuffix trims
// that tail, so `_` inlines to `_inline7` rather than the rejected `__inline7`,
// while leaving an already-invalid `a__b` invalid for ValidateBaseName to report.
std::string FreshName(const std::string& orig) {
  static int counter = 0;
  return auto_name::JoinNameSuffix(orig, "inline" + std::to_string(counter++));
}

// Counts ReturnStmts anywhere inside a body. Splicing only handles a trailing
// return at top level; an early return nested inside an If/For/While/Scope
// body would leak into the caller and trigger the OUTER function to return,
// silently miscompiling. The pl-DSL doesn't expose nested returns, but
// hand-built IR could; reject it explicitly.
class NestedReturnCounter : public IRVisitor {
 public:
  int count = 0;
  void VisitStmt_(const ReturnStmtPtr& op) override {
    ++count;
    IRVisitor::VisitStmt_(op);
  }
};

// Counts call-like nodes whose *evaluation* must survive even when the value
// they produce is thrown away — which, deliberately, is every Call and every
// Submit.
//
// Nothing in the IR answers "is this call safe to delete". The nearest
// registry data, `OpRegistryEntry::WritesAnyArg`, answers a different question:
// whether the operator writes *through an argument*. Deleting on that basis is
// wrong in both directions. Most operators are simply unclassified — 263 of 315
// at the time of writing, among them `tile.tpush_to_aiv` and
// `system.aic_initialize_pipe`, which the shared DCE lists as side-effecting
// (dead_code_elimination.cpp::IsSideEffectOp). And a *positive*
// `no_arg_writes()` verdict does not mean deletable either: `pld.system.wait`
// blocks until a signal slot satisfies a threshold, `pld.system.defer_wait`
// registers a completion condition, and `system.set_ffts` hands the FFTS unit
// its workspace pointer — all three declare `no_arg_writes()` while carrying
// synchronization or hardware-setup semantics that deleting would break.
//
// So the pass keeps every call. The cost is that a discarded genuinely pure call
// survives as a dead EvalStmt, which the pipeline carries harmlessly; the
// alternative costs correctness. Narrowing this needs a real "safely deletable"
// operator property, declared per operator, not an inference from writes.
class EffectfulCallCounter : public IRVisitor {
 public:
  int count = 0;
  void VisitExpr_(const CallPtr& op) override {
    if (op) ++count;
    IRVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const SubmitPtr& op) override {
    if (op) ++count;
    IRVisitor::VisitExpr_(op);
  }
};

// Is `value` ITSELF a call-like node, as opposed to merely wrapping one? Only
// such a value can be re-emitted verbatim as an EvalStmt.
bool IsEffectfulCallLike(const ExprPtr& value) {
  return As<Call>(value) != nullptr || As<Submit>(value) != nullptr;
}

// Result of splicing an inline call's body without yet wiring up its return
// values into a specific caller statement. The caller picks the wiring form
// (assign / drop / return / ...) based on its own statement kind.
struct SplicedInlineBody {
  std::vector<StmtPtr> stmts;          // Pre-return statements, in order
  std::vector<ExprPtr> return_values;  // Trailing-return values (empty if has_return is false)
  bool has_return;                     // Whether the body ended with a ReturnStmt
};

using InlineFunctionMap = std::unordered_map<std::string, FunctionPtr>;

// Direct substitution at def-sites is safe only while a shaped parameter keeps
// naming its own storage. If any definition can change that binding, use one
// callee-local handle for the entire call, initialized before its control flow.
// Writes through that handle still alias the argument's storage; value-producing
// assignments and loop/branch results cannot retarget the caller's variable.
// Collect this alongside the existing def/use walk, without per-param scans.
class InlineParamDefCollector : public var_collectors::VarDefUseCollector {
 public:
  explicit InlineParamDefCollector(const std::vector<VarPtr>& params) {
    for (const auto& param : params) {
      params_.insert(param.get());
      aliases_.try_emplace(param.get());
    }
  }

  // Alias edges are deliberately undirected: a mutable alias may change on a
  // later loop iteration or in another branch. Only a component whose every
  // definition preserves one parameter's storage permits direct substitution.
  // One graph walk handles cycles and backedges in O(nodes + alias edges).
  std::unordered_set<const Var*> LocalBindings() const {
    std::unordered_set<const Var*> visited;
    std::unordered_set<const Var*> result;
    for (const auto& [start, edges] : aliases_) {
      if (!visited.insert(start).second) continue;
      std::vector<const Var*> pending{start};
      std::vector<const Var*> params;
      bool changes_value = false;
      while (!pending.empty()) {
        const Var* var = pending.back();
        pending.pop_back();
        if (params_.count(var)) params.push_back(var);
        changes_value |= value_defs_.count(var) > 0 || (params_.count(var) == 0 && var_defs.count(var) == 0);
        for (const Var* source : aliases_.at(var)) {
          if (visited.insert(source).second) pending.push_back(source);
        }
      }
      if (changes_value || params.size() > 1) result.insert(params.begin(), params.end());
    }
    return result;
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    ExprPtr source = op->value_;
    if (auto call = As<Call>(source)) {
      auto index = ResultAliasedArgIndex(call);
      source = index ? call->args_[*index] : nullptr;
    }
    auto source_var = AsVarLike(source);
    aliases_.try_emplace(op->var_.get());
    if (source_var) {
      aliases_[op->var_.get()].push_back(source_var.get());
      aliases_[source_var.get()].push_back(op->var_.get());
    } else {
      // Submit results (tuple projections) and unknown calls conservatively
      // require a local handle; they are not converted into ordinary Calls.
      value_defs_.insert(op->var_.get());
    }
    VarDefUseCollector::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    RecordResults(op->return_vars_);
    RecordResults(op->iter_args_);
    VarDefUseCollector::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    RecordResults(op->return_vars_);
    RecordResults(op->iter_args_);
    VarDefUseCollector::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    RecordResults(op->return_vars_);
    VarDefUseCollector::VisitStmt_(op);
  }

 private:
  template <typename VarPtrT>
  void RecordResults(const std::vector<VarPtrT>& vars) {
    for (const auto& var : vars) {
      aliases_.try_emplace(var.get());
      value_defs_.insert(var.get());
    }
  }

  std::unordered_set<const Var*> params_;
  std::unordered_set<const Var*> value_defs_;
  std::unordered_map<const Var*, std::vector<const Var*>> aliases_;
};

// Conflicting argument extents may leave a callee dimension unbound. Reject it
// only if it survives cloning; a helper using just the actual arguments remains
// valid. Reuse the shared type-field walk for tensor, tile, tuple and view types.
class UnboundInlineDimensionChecker : public IRVisitor {
 public:
  UnboundInlineDimensionChecker(const FunctionPtr& callee, const std::unordered_set<const Var*>& unbound)
      : callee_(callee), unbound_(unbound) {}

  void VisitExpr(const ExprPtr& expr) override {
    if (!expr) return;
    if (auto var = As<Var>(expr)) CheckDimension(var.get(), expr->span_);
    if (auto type = expr->GetType(); type && checked_types_.insert(type.get()).second) {
      for (const auto* var : var_collectors::GetSortedVarRefs(var_collectors::CollectTypeVars(type))) {
        CheckDimension(var, expr->span_);
      }
    }
    IRVisitor::VisitExpr(expr);
  }

 private:
  void CheckDimension(const Var* var, const Span& span) const {
    CHECK_SPAN(unbound_.count(var) == 0, span)
        << "Cannot inline '" << callee_->name_ << "': type dimension '" << var->name_hint_
        << "' remains unresolved from the argument shapes. Use distinct dynamic symbols for parameters "
           "with different extents.";
  }

  const FunctionPtr& callee_;
  const std::unordered_set<const Var*>& unbound_;
  std::unordered_set<const Type*> checked_types_;
};

// Splice an EvalStmt-shaped call (no LHS) — the callee's trailing return VALUE
// has no destination, but *evaluating* it can still be observable, so the value
// may not simply be discarded along with the ReturnStmt that carried it.
//
// A discarded Call or Submit is re-emitted as an EvalStmt, in return order —
// every one of them, for the reasons on EffectfulCallCounter. Nested inline
// calls have already been expanded while visiting the cloned body. A non-Inline
// callee stays an ordinary dispatch, exactly as if the author had written
// `self.inner(...)` at the call site. Without this, an ignored wrapper whose body is
// `return self.inner(x, out)` silently lost `inner`'s write to `out` (#2705),
// one whose body is `return pl.tile.store(t, [0, 0], out)` lost the store, and
// one whose body is `return pl.system.set_ffts(ws)` lost the hardware setup —
// the assign and return call-site forms never had the hole, because they re-emit
// the value into an AssignStmt / ReturnStmt.
//
// Any other value is dropped: a Var or a constant hides nothing. But such a
// value may *wrap* a call (`return self.bump(n) + 1` reaching here as one `Add`,
// or a MakeTuple / TupleGetItemExpr over a call), and that cannot become an
// EvalStmt the way a call-like value can — reject it loudly instead of deleting
// the nested call with it.
std::vector<StmtPtr> SpliceInlineCallAsEval(const FunctionPtr& callee, SplicedInlineBody body) {
  for (const auto& value : body.return_values) {
    if (!value) continue;
    if (IsEffectfulCallLike(value)) {
      body.stmts.push_back(std::make_shared<const EvalStmt>(value, value->span_));
      continue;
    }
    EffectfulCallCounter counter;
    counter.VisitExpr(value);
    CHECK_SPAN(counter.count == 0, value->span_)
        << "Inline function '" << callee->name_
        << "' is called for its side effects only (its result is discarded), but its return "
           "expression wraps a call whose evaluation cannot be preserved once the value is "
           "dropped. Either return that call directly ('return self.inner(...)'), or bind the "
           "result at the call site ('result = self."
        << callee->name_ << "(...)').";
  }
  return std::move(body.stmts);
}

// Splice a single-return call into `LHS = inlined_return_value`. CHECK-fails
// if the callee returns multiple values — multi-return goes through
// SpliceInlineCallAsTupleSub, which avoids the dead `LHS = MakeTuple(...)`.
std::vector<StmtPtr> SpliceInlineCallAsAssign(const FunctionPtr& callee, SplicedInlineBody body,
                                              const VarPtr& lhs, const Span& call_site_span) {
  INTERNAL_CHECK_SPAN(body.has_return, call_site_span)
      << "Internal error: inline function '" << callee->name_
      << "' is called for its value but has no return statement (parser should reject "
         "value-use of a void inline function before InlineFunctions runs)";
  INTERNAL_CHECK_SPAN(body.return_values.size() == 1, call_site_span)
      << "Internal error: SpliceInlineCallAsAssign requires single-return callee; got "
      << body.return_values.size() << " return values for '" << callee->name_
      << "' (caller dispatches multi-return through SpliceInlineCallAsTupleSub)";

  ExprPtr final_value = body.return_values[0];
  // Skip the no-op `lhs = lhs` that arises when an arg is also returned —
  // it would otherwise survive into SSA and break structural equality.
  if (auto var_expr = As<Var>(final_value); var_expr && var_expr.get() == lhs.get()) {
    return std::move(body.stmts);
  }
  body.stmts.push_back(std::make_shared<const AssignStmt>(lhs, final_value, call_site_span));
  return std::move(body.stmts);
}

// Splice a tuple-return call without emitting `LHS = MakeTuple(values)`.
// Instead, hands back the cloned return values via `out_substitution` so the
// caller (the InlineCallsMutator) can substitute downstream
// `TupleGetItemExpr(LHS, i)` uses with `values[i]` directly.
//
// The parser-generated tuple-unpack pattern (`_tuple_tmp = call(); y_i =
// _tuple_tmp[i]`) needs no aggregate receiver: substitute its element uses.
// A callee-local tuple may still need its definition to preserve body uses
// and the values captured before later reassignments.
std::vector<StmtPtr> SpliceInlineCallAsTupleSub(const FunctionPtr& callee, SplicedInlineBody body,
                                                std::vector<ExprPtr>& out_substitution) {
  INTERNAL_CHECK_SPAN(body.has_return, callee->span_)
      << "Internal error: inline function '" << callee->name_
      << "' is called for its value but has no return statement (parser should reject "
         "value-use of a void inline function before InlineFunctions runs)";
  if (body.return_values.size() > 1) {
    out_substitution = std::move(body.return_values);
    return std::move(body.stmts);
  }

  // Python may return a tuple through a temporary (`tmp = (a, b); return
  // tmp`) rather than directly as `return a, b`. Inline it as individual
  // elements too: leaving `lhs = tmp` would make the later TupleGetItemExpr
  // uses invisible to tuple_subs_ and leave a MakeTuple for codegen.
  INTERNAL_CHECK_SPAN(body.return_values.size() == 1, callee->span_)
      << "Internal error: tuple-return inline function '" << callee->name_ << "' has no return value";
  if (auto tuple = As<MakeTuple>(body.return_values[0])) {
    out_substitution = tuple->elements_;
    return std::move(body.stmts);
  }
  if (auto returned_var = As<Var>(body.return_values[0])) {
    for (size_t i = body.stmts.size(); i-- > 0;) {
      auto assign = As<AssignStmt>(body.stmts[i]);
      if (assign && assign->var_.get() == returned_var.get()) {
        if (auto tuple = As<MakeTuple>(assign->value_)) {
          if (i + 1 == body.stmts.size()) {
            out_substitution = tuple->elements_;
            body.stmts.pop_back();
          } else {
            // Intervening statements may read this tuple or rebind its elements.
            // Keep the definition and return its captured values, not later Vars.
            for (size_t index = 0; index < tuple->elements_.size(); ++index) {
              out_substitution.push_back(
                  std::make_shared<TupleGetItemExpr>(returned_var, static_cast<int>(index), assign->span_));
            }
          }
          return std::move(body.stmts);
        }
      }
    }
  }
  INTERNAL_CHECK_SPAN(false, callee->span_) << "Inline function '" << callee->name_
                                            << "' returns a tuple through an unsupported expression; return "
                                               "its elements directly or via a tuple literal";
  return std::move(body.stmts);
}

// Splice a `return inline_call(args...)` statement: emit the cloned pre-return
// body followed by a fresh ReturnStmt that returns the callee's trailing return
// values directly. Single-return → ReturnStmt({v}); multi-return →
// ReturnStmt({v0, v1, ...}). No MakeTuple, no temporary.
std::vector<StmtPtr> SpliceInlineCallAsReturn(const FunctionPtr& callee, SplicedInlineBody body,
                                              const Span& call_site_span) {
  INTERNAL_CHECK_SPAN(body.has_return, call_site_span)
      << "Internal error: inline function '" << callee->name_
      << "' is used as a return value but has no return statement (parser should reject "
         "value-use of a void inline function before InlineFunctions runs)";
  body.stmts.push_back(std::make_shared<const ReturnStmt>(std::move(body.return_values), call_site_span));
  return std::move(body.stmts);
}

bool InlineReturnsTuple(const FunctionPtr& callee) {
  StmtPtr body = callee->body_;
  if (auto seq = As<SeqStmts>(body)) {
    if (seq->stmts_.empty()) return false;
    body = seq->stmts_.back();
  }
  auto ret = As<ReturnStmt>(body);
  if (!ret) return false;
  return ret->value_.size() > 1 ||
         (ret->value_.size() == 1 && ret->value_[0] && As<TupleType>(ret->value_[0]->GetType()));
}

// =============================================================================
// Selective-dump carry-through across inlining (simpler#844)
// =============================================================================

// Collect every Var pointer referenced in a statement (use- and def-sites;
// VisitVarLike_ covers both Var and IterArg per ir-kind-traits).
class VarUseCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> uses;
  void VisitVarLike_(const VarPtr& op) override {
    if (op) uses.insert(op.get());
    IRVisitor::VisitVarLike_(op);
  }
};

// Transfer an inline call-site's ``kAttrDumpVars`` onto the spliced callee body.
//
// The dump entries include caller arg Vars and any local handles introduced
// for them by ``CloneInlineBody``. Transfer runs before recursively expanding
// nested inline calls, so each call can remap its own bindings. Two carriers are
// stamped (both round-trip and are tracked by Var identity downstream):
//
//   * Dispatch scopes (``pl.at`` / ``pl.spmd`` / ``pl.cluster`` / ``pl.graph``)
//     whose body uses a tagged arg get it merged into their ``kAttrDumpVars`` —
//     the same scope-level carrier ``pl.dump_tag`` seeds at parse, printed back
//     as each construct's ``dumps=``. The outliner later maps it onto the
//     synthesised dispatch.
//   * Nested cross-function (``GlobalVar``) Calls that take a tagged arg get it
//     merged into the Call's ``kAttrDumpVars``. This is what makes a tag survive
//     *multi-level* inlining: when the callee itself just forwards the arg into
//     a deeper ``self.foo(...)`` (no scope of its own consumes it), the tag
//     rides that Call so the next inline iteration (or the final dispatch, if
//     ``foo`` is a real kernel) carries it.
//
// Builtin tile/tensor op Calls (``OpExpr`` callee) are intentionally NOT
// stamped: their ``dump_vars`` would not round-trip (the printer only emits the
// marker for ``GlobalVar`` calls and scopes), and codegen never reads them.
//
// A tag not consumed by any scope or dispatch in the spliced body is dropped:
// there is no kernel launch to dump it on.
class InlineDumpVarTransfer : public IRMutator {
 public:
  explicit InlineDumpVarTransfer(std::vector<VarPtr> dump_vars) : dump_vars_(std::move(dump_vars)) {}

  StmtPtr VisitStmt_(const InCoreScopeStmtPtr& op) override { return Attach<InCoreScopeStmt>(op); }
  StmtPtr VisitStmt_(const HierarchyScopeStmtPtr& op) override { return Attach<HierarchyScopeStmt>(op); }
  StmtPtr VisitStmt_(const ClusterScopeStmtPtr& op) override { return Attach<ClusterScopeStmt>(op); }
  StmtPtr VisitStmt_(const SpmdScopeStmtPtr& op) override { return Attach<SpmdScopeStmt>(op); }
  StmtPtr VisitStmt_(const GraphScopeStmtPtr& op) override { return Attach<GraphScopeStmt>(op); }
  // ``SplitAivScopeStmt`` is deliberately not stamped: a ``pl.split_aiv`` region
  // lives inside a kernel and is never outlined into a dispatch, so no pass reads
  // a dump mark off it (and ``pl.split_aiv`` has no ``dumps=`` to print one as).
  // The enclosing InCore scope carries the mark instead.

  ExprPtr VisitExpr_(const CallPtr& op) override {
    // Recurse first so nested args (this pass runs pre-flatten, so a call arg
    // may itself be a dispatch Call) are mutated before we stamp the attr.
    auto mutated_expr = IRMutator::VisitExpr_(op);
    auto mutated_call = As<Call>(mutated_expr);
    // Only cross-function dispatches carry a round-trippable dump attr; skip
    // builtin tile/tensor ops (OpExpr callee).
    if (!mutated_call || !As<GlobalVar>(mutated_call->op_)) return mutated_expr;
    auto existing = mutated_call->GetAttr<std::vector<VarPtr>>(kAttrDumpVars);
    auto merged = Merge(existing, ArgVarSet(mutated_call->args_));
    if (!Changed(existing, merged)) return mutated_expr;
    auto result = MutableCopy(mutated_call);
    result->attrs_ = WithDumpVarsAttr(mutated_call->attrs_, std::move(merged));
    return result;
  }

 private:
  template <typename ScopeT>
  StmtPtr Attach(const std::shared_ptr<const ScopeT>& op) {
    // Recurse first so nested scopes / dispatch calls also receive their tags.
    auto recursed_stmt = IRMutator::VisitStmt_(op);
    auto recursed = std::dynamic_pointer_cast<const ScopeT>(recursed_stmt);
    if (!recursed) return recursed_stmt;

    VarUseCollector uc;
    uc.VisitStmt(recursed->body_);

    auto existing = recursed->template GetAttr<std::vector<VarPtr>>(kAttrDumpVars);
    auto merged = Merge(existing, uc.uses);
    if (!Changed(existing, merged)) return recursed_stmt;

    auto result = MutableCopy(recursed);
    result->attrs_ = WithDumpVarsAttr(recursed->attrs_, std::move(merged));
    return result;
  }

  std::unordered_set<const Var*> ArgVarSet(const std::vector<ExprPtr>& args) const {
    std::unordered_set<const Var*> s;
    for (const auto& a : args) {
      if (auto v = AsVarLike(a)) s.insert(v.get());
    }
    return s;
  }

  // Append the dump vars present in ``candidates`` to ``existing`` (dedup,
  // preserving existing entries first then dump_vars_ order).
  std::vector<VarPtr> Merge(std::vector<VarPtr> existing,
                            const std::unordered_set<const Var*>& candidates) const {
    std::unordered_set<const Var*> present;
    for (const auto& v : existing) {
      if (v) present.insert(v.get());
    }
    for (const auto& dv : dump_vars_) {
      if (!dv || candidates.count(dv.get()) == 0) continue;
      if (!present.insert(dv.get()).second) continue;
      existing.push_back(dv);
    }
    return existing;
  }

  static bool Changed(const std::vector<VarPtr>& before, const std::vector<VarPtr>& after) {
    return before.size() != after.size();
  }

  std::vector<VarPtr> dump_vars_;
};

// =============================================================================
// NestedInlineCallHoister — moves a Call to an inline callee out of a nested
// expression position into an AssignStmt of its own.
// =============================================================================

// `InlineCallsMutator` recognises a call site only when the Call *is* the whole
// statement value (`LHS = f(...)`, `EvalStmt(f(...))`, `return f(...)`), yet the
// pass drops every Inline function unconditionally when it finishes. A Call in
// any other position therefore survives as a reference to a deleted function,
// and the failure only surfaces 40 passes later as
// "references undefined function" from the orchestration codegen precondition.
//
// Such calls come straight from ordinary DSL. The parser desugars
// `arr[i] = f(x)` into `arr = array.update_element(arr, i, f(x))`, so the call
// is nested even though the user never wrote a nested call; `k = f(n) + 1`,
// `for i in pl.range(f(n))` and `k = f(f(n))` reach the same place.
//
// FlattenCallExpr (pass 06) already performs exactly this hoist for *all*
// calls, but it declares `.required = {SSAForm, NormalizedStmtStructure}`, both
// established after this pass, so it cannot simply run first. Hoist the inline
// callees here instead; the general job stays with pass 06.
class NestedInlineCallHoister : public IRMutator {
 public:
  NestedInlineCallHoister(const std::unordered_map<std::string, FunctionPtr>& inline_fns,
                          std::vector<StmtPtr>* pending)
      : inline_fns_(inline_fns), pending_(pending) {}

  /// Hoist every nested inline call inside @p expr, returning @p expr itself
  /// when nothing moved.
  ExprPtr Hoist(const ExprPtr& expr) { return VisitExpr(expr); }

  /// Hoist inside @p call's *arguments* while leaving the call itself in place.
  /// Used when the call already sits where `HandleTopLevelInlineCall` splices
  /// it, so hoisting it too would only insert a redundant copy (and churn the
  /// output of every existing call site).
  ExprPtr HoistInArgs(const CallPtr& call) {
    std::vector<ExprPtr> new_args;
    new_args.reserve(call->args_.size());
    bool changed = false;
    for (const auto& arg : call->args_) {
      auto new_arg = VisitExpr(arg);
      if (new_arg.get() != arg.get()) changed = true;
      new_args.push_back(std::move(new_arg));
    }
    if (!changed) return call;
    return std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, call->attrs_,
                                  call->GetType(), call->span_);
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    // Recurse first, so the innermost call hoists first and `f(g(x))` yields
    // `t0 = g(x); t1 = f(t0)` in evaluation order.
    auto visited = IRMutator::VisitExpr_(op);
    auto call = As<Call>(visited);
    if (!call) return visited;
    auto gvar = As<GlobalVar>(call->op_);
    if (!gvar) return visited;
    auto it = inline_fns_.find(gvar->name_);
    if (it == inline_fns_.end()) return visited;

    // A tuple-returning callee must not be hoisted. `SpliceInlineCallAsTupleSub`
    // deliberately emits no `tmp = ...` binding — it records the cloned return
    // values against the LHS Var and rewrites downstream
    // `TupleGetItemExpr(tmp, i)` uses instead. A nested consumer holds `tmp`
    // itself rather than a TupleGetItemExpr, so hoisting would leave the temp
    // undefined (`return self.pair(x), y` printed `t__inline_arg_v0__FREE_VAR`).
    // Leaving the Call in place preserves the pre-hoist behaviour: the
    // InlineFunctionsEliminated verifier reports it at its own source line
    // right after this pass.
    if (InlineReturnsTuple(it->second)) return visited;

    VarPtr tmp = std::make_shared<Var>(HoistTempName(), call->GetType(), call->span_);
    pending_->push_back(std::make_shared<const AssignStmt>(tmp, call, call->span_));
    return tmp;
  }

 private:
  // Distinct role from FlattenCallExpr's `t__tmp_vN`: both passes mint temps
  // into the same pre-SSA function, and a name collision would be merged by
  // ConvertToSSA into one versioned variable.
  static std::string HoistTempName() {
    static int counter = 0;
    return auto_name::BuildName("t", "", "inline_arg", counter++);
  }

  const std::unordered_map<std::string, FunctionPtr>& inline_fns_;
  std::vector<StmtPtr>* pending_;
};

// =============================================================================
// InlineCallsMutator — expands inline calls and propagates the resulting types
// through the same traversal and variable map. A shared shape placeholder may
// describe different buffers, so operator results follow their actual operands.
// =============================================================================

class InlineCallsMutator : public IRMutator {
 public:
  InlineCallsMutator(const std::unordered_map<std::string, FunctionPtr>& inline_fns,
                     const FunctionPtr& caller, const InlineFunctionMap& functions)
      : inline_fns_(inline_fns), functions_(functions) {
    // Signature dimensions are in scope throughout the caller, including when
    // a particular inline call receives only locally allocated buffers.
    for (const auto& param : caller->params_) {
      preserved_vars_.insert(param.get());
      auto vars = var_collectors::CollectTypeVars(param->GetType());
      caller_dimensions_.insert(vars.begin(), vars.end());
    }
    for (const auto& type : caller->return_types_) {
      auto vars = var_collectors::CollectTypeVars(type);
      caller_dimensions_.insert(vars.begin(), vars.end());
    }
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto call = As<Call>(IRMutator::VisitExpr_(op));
    TypePtr type;
    if (As<GlobalVar>(call->op_)) {
      type = DeduceDispatchType(call->op_, call->args_, false);
    } else {
      auto& registry = OpRegistry::GetInstance();
      const auto& entry = registry.GetEntry(call->op_->name_);
      if (entry.RequiresExplicitType()) return call;
      // Static operations untouched by substitution retain their descriptors.
      // Writebacks and dynamic results need operand-specific specialization.
      if (call == op && !entry.GetOutputReusesInputArg() &&
          var_collectors::CollectTypeVars(call->GetType()).empty()) {
        return call;
      }
      type = registry.Create(call->op_->name_, call->args_, call->kwargs_, call->span_)->GetType();
    }
    // Some ops infer UnknownType and rely on an explicit result annotation.
    if (!type || As<UnknownType>(type)) return call;
    type = WithCarriedMemRef(type, call->GetType());
    if (structural_equal(type, call->GetType())) return call;
    return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, call->attrs_, type, call->span_);
  }

  ExprPtr VisitExpr_(const SubmitPtr& op) override {
    auto submit = As<Submit>(IRMutator::VisitExpr_(op));
    auto type = DeduceDispatchType(submit->op_, submit->args_, true);
    if (!type || structural_equal(type, submit->GetType())) return submit;
    return std::make_shared<Submit>(submit->op_, submit->args_, submit->deps_, submit->kwargs_,
                                    submit->attrs_, type, submit->span_, submit->core_num_,
                                    submit->sync_start_, submit->allow_early_resolve_, submit->predicate_);
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto arg = As<IterArg>(IRMutator::VisitExpr_(op));
    auto type = arg->initValue_->GetType();
    if (structural_equal(arg->GetType(), type)) return arg;
    auto fresh = std::make_shared<IterArg>(arg->name_hint_, type, arg->initValue_, arg->span_);
    retained_.push_back(arg);
    var_remap_[op.get()] = fresh;
    return fresh;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return RetypeLoop(op); }
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return RetypeLoop(op); }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto result = As<IfStmt>(IRMutator::VisitStmt_(op));
    auto yield = transform_utils::GetLastYieldStmt(result->then_body_);
    if (!yield && result->else_body_) yield = transform_utils::GetLastYieldStmt(*result->else_body_);
    if (!yield) return result;
    auto vars = result->return_vars_;
    bool changed = false;
    for (size_t i = 0; i < vars.size() && i < yield->value_.size(); ++i) {
      auto type = WithCarriedMemRef(yield->value_[i]->GetType(), vars[i]->GetType());
      auto var = Retype(op->return_vars_[i], vars[i], type);
      changed |= var.get() != vars[i].get();
      vars[i] = std::move(var);
    }
    if (!changed) return result;
    auto copy = MutableCopy(result);
    copy->return_vars_ = std::move(vars);
    return copy;
  }
  bool Changed() const { return changed_; }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    bool any_changed = false;
    for (const auto& stmt : op->stmts_) {
      // Pull nested inline calls onto statements of their own first; SpliceHoisted
      // then treats each as an ordinary call site, so a hoist and the splice it
      // enables land in the same fixpoint iteration.
      if (auto hoisted = HoistNestedInlineCalls(stmt)) {
        new_stmts.push_back(SpliceHoisted(*hoisted, stmt->span_));
        any_changed = true;
        continue;
      }
      auto recursed = VisitStmt(stmt);
      if (recursed.get() != stmt.get()) any_changed = true;
      new_stmts.push_back(recursed);
    }
    if (!any_changed) return op;
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }

  // Bare AssignStmt body — e.g. `if c: x = inline_f(...)` where the IfStmt's
  // then_body is a single AssignStmt, not a SeqStmts. InlineFunctions runs
  // before NormalizeStmtStructure, so non-SeqStmts bodies are possible. Wrap
  // the splice in a SeqStmts so the parent body remains a single Stmt;
  // SeqStmts::Flatten collapses any redundant nesting later.
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    if (auto hoisted = HoistNestedInlineCalls(op)) return SpliceHoisted(*hoisted, op->span_);
    if (auto handled = HandleTopLevelInlineCall(op)) {
      changed_ = true;
      return SeqStmts::Flatten(std::move(*handled), op->span_);
    }
    auto value = VisitExpr(op->value_);
    auto var = As<Var>(VisitExpr(op->var_));
    if (!var) return IRMutator::VisitStmt_(op);  // MemRef definitions retain their descriptors.
    // Before SSA, one Var can be assigned repeatedly, including in different
    // branches. Keep its binding identity once defined: a branch-local valid
    // shape refinement must not redirect the other branch or the loop seed to
    // a different Var. Fresh definitions use the same type selection as SSA.
    auto [definition, inserted] = definitions_.emplace(op->var_.get(), var);
    if (inserted) {
      definition->second = Retype(op->var_, var, GetAuthoritativeAssignmentType(var->GetType(), value));
    }
    var = definition->second;
    if (var.get() == op->var_.get() && value.get() == op->value_.get()) return op;
    return std::make_shared<AssignStmt>(var, value, op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    if (auto hoisted = HoistNestedInlineCalls(op)) return SpliceHoisted(*hoisted, op->span_);
    auto handled = HandleTopLevelInlineCall(op);
    if (!handled.has_value()) return IRMutator::VisitStmt_(op);
    changed_ = true;
    return SeqStmts::Flatten(std::move(*handled), op->span_);
  }

  // `return inline_call(...)`. Same SeqStmts caveat as AssignStmt /
  // EvalStmt: a function body that is a bare ReturnStmt (no enclosing
  // SeqStmts) reaches this override directly.
  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    if (auto hoisted = HoistNestedInlineCalls(op)) return SpliceHoisted(*hoisted, op->span_);
    auto handled = HandleTopLevelInlineCall(op);
    if (!handled.has_value()) return IRMutator::VisitStmt_(op);
    changed_ = true;
    return SeqStmts::Flatten(std::move(*handled), op->span_);
  }

  // Apply tuple substitutions registered by multi-return inline splices:
  // `TupleGetItemExpr(LHS_var, i)` → `return_values[i]`. The substituted
  // expression is then visited again (via VisitExpr) to fold any nested
  // TupleGetItemExpr's the same way.
  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
    // Fast-path the common case: programs without multi-return inline calls
    // never populate tuple_subs_, so every TupleGetItemExpr in the IR would
    // otherwise pay an unnecessary VisitExpr+find on the tuple operand.
    if (tuple_subs_.empty()) return IRMutator::VisitExpr_(op);

    auto recursed_tuple = VisitExpr(op->tuple_);
    if (auto var = As<Var>(recursed_tuple)) {
      auto it = tuple_subs_.find(var.get());
      if (it != tuple_subs_.end() && op->index_ >= 0 && static_cast<size_t>(op->index_) < it->second.size()) {
        // Visit the replacement so nested substitutions also apply.
        return VisitExpr(it->second[op->index_]);
      }
    }
    if (recursed_tuple.get() == op->tuple_.get()) return op;
    return std::make_shared<TupleGetItemExpr>(recursed_tuple, op->index_, op->span_);
  }

 private:
  TypePtr DeduceDispatchType(const OpPtr& op, const std::vector<ExprPtr>& args, bool submit) {
    auto found = functions_.find(op->name_);
    if (found == functions_.end() || found->second->func_type_ == FunctionType::Inline) return nullptr;
    const auto& callee = found->second;
    auto returns = callee->return_types_;
    if (returns.empty() && !submit) {
      auto body = callee->body_;
      if (auto seq = As<SeqStmts>(body); seq && !seq->stmts_.empty()) {
        body = seq->stmts_.back();
      }
      if (auto ret = As<ReturnStmt>(body)) {
        for (const auto& value : ret->value_) returns.push_back(value->GetType());
      }
    }
    auto params = callee->params_;
    if (submit && args.size() < params.size()) {
      // Submit may omit trailing Out buffers before its CommCtx suffix.
      const size_t contexts = transform_utils::TrailingCommCtxParamCount(callee);
      INTERNAL_CHECK_SPAN(args.size() >= contexts, callee->span_) << "Invalid Submit context arguments";
      params.erase(params.begin() + static_cast<std::ptrdiff_t>(args.size() - contexts),
                   params.end() - static_cast<std::ptrdiff_t>(contexts));
    }
    returns = DeduceCallReturnType(params, args, returns);
    if (submit) returns.push_back(std::make_shared<ScalarType>(DataType::TASK_ID));
    if (returns.empty()) return nullptr;
    if (!submit && returns.size() == 1) return returns.front();
    return std::make_shared<TupleType>(std::move(returns));
  }

  VarPtr Retype(const VarPtr& original, const VarPtr& current, const TypePtr& type) {
    if (preserved_vars_.count(original.get())) return current;
    if (structural_equal(current->GetType(), type) &&
        GetTypeMemRef(current->GetType()) == GetTypeMemRef(type)) {
      return current;
    }
    auto fresh = std::make_shared<Var>(current->name_hint_, type, current->span_);
    // Keep intermediate Vars alive while their raw pointers key var_remap_.
    retained_.push_back(current);
    var_remap_[original.get()] = fresh;
    if (current.get() != original.get()) var_remap_[current.get()] = fresh;
    return fresh;
  }

  template <typename LoopPtr>
  StmtPtr RetypeLoop(const LoopPtr& op) {
    auto result = std::static_pointer_cast<typename LoopPtr::element_type>(IRMutator::VisitStmt_(op));
    auto vars = result->return_vars_;
    bool changed = false;
    for (size_t i = 0; i < vars.size() && i < result->iter_args_.size(); ++i) {
      auto type = WithCarriedMemRef(result->iter_args_[i]->GetType(), vars[i]->GetType());
      auto var = Retype(op->return_vars_[i], vars[i], type);
      changed |= var.get() != vars[i].get();
      vars[i] = std::move(var);
    }
    if (!changed) return result;
    auto copy = MutableCopy(result);
    copy->return_vars_ = std::move(vars);
    return copy;
  }

  // Clone and visit the inline body before the SpliceInlineCall* helpers wire
  // its trailing return values into the caller.
  //
  //   1. Build the substitution seed: type variables → actual dimensions and
  //      params → actual args. An actual arg that cannot be substituted verbatim
  //      is first bound to a fresh Var at the call site.
  //   2. DeepClone the callee body, alpha-renaming locals and remapping their
  //      types with that seed. The clone uses the same
  //      substitution at use-sites and def-sites, so a rebinding of a param
  //      (`out = pl.assemble(out, ...)`) collapses to `actual = pl.assemble(
  //      actual, ...)` whenever the actual arg is a Var.
  //   3. Visit the clone with this mutator: expand nested inline calls and
  //      propagate types using the same mappings as subsequent caller uses.
  //   4. Split it into pre-return statements and trailing return values;
  //      reject any non-trailing return.
  SplicedInlineBody CloneInlineBody(const FunctionPtr& callee, const std::vector<ExprPtr>& args,
                                    std::vector<VarPtr> dump_vars) {
    INTERNAL_CHECK_SPAN(callee->params_.size() == args.size(), callee->span_)
        << "Internal error: inline call to '" << callee->name_ << "' has " << args.size()
        << " argument(s) but callee expects " << callee->params_.size()
        << " (parser/type-checker should have caught arity mismatch before InlineFunctions)";

    InlineParamDefCollector def_collector(callee->params_);
    def_collector.VisitStmt(callee->body_);
    const auto local_bindings = def_collector.LocalBindings();

    // 1. Build the seed substitution map for DeepClone:
    //    - Each param Var → its actual-arg Expr. The same substitution is
    //      consulted at both use-sites and def-sites of the param, so a
    //      rebinding `out = pl.assemble(out, ...)` where `out` is a param
    //      becomes `q_out = pl.assemble(q_out, ...)` when the actual arg is
    //      the Var `q_out` and the shaped binding is proven to keep that
    //      storage. Array and explicit scalar output conventions are preserved.
    //    - Some actual args are instead bound to a fresh Var ahead of the body,
    //      and that Var is substituted — exactly the IR the parser emits when
    //      the caller names the argument itself (`cr = c[r]; f(x, cr)`):
    //        * A rebound param whose arg is not an assignable Var (a slice
    //          `c[r]`, an IterArg, a computed scalar). Substituting it would put
    //          that Expr on the LHS of the rebinding.
    //        * A rebound plain scalar, or a tensor / tile whose binding may
    //          change. The local handle preserves storage writes without
    //          splicing `t = add(t, t)` onto the caller's own Var. It is bound
    //          before the body so SSA carries it through branches and loops.
    //        * A computed tensor / tile arg (a `Call`, e.g. `a[r]`). Python
    //          evaluates it once at the call; substituting it would re-evaluate
    //          it at every use, inside the callee's `pl.spmd` / `pl.pipeline` /
    //          `pl.at` bodies, moving the view from the call site into the
    //          outlined kernel.
    //      Read-only scalar and constant args stay substituted, so shape
    //      expressions that read a param keep folding to the caller's value.
    //    - Dynamic dimensions in parameter types → actual argument dimensions,
    //      using the same binding rules as cross-function return-type deduction.
    auto seed = DeduceCallTypeBindings(callee->params_, args);
    std::vector<StmtPtr> arg_bindings;
    std::unordered_set<const Var*> tagged_args;
    for (const auto& var : dump_vars) {
      if (var) tagged_args.insert(var.get());
    }
    for (size_t i = 0; i < callee->params_.size(); ++i) {
      const VarPtr& param = callee->params_[i];
      ExprPtr actual = args[i];
      const TypePtr actual_type = actual->GetType();
      // The same assignment targets IRMutator::VisitStmt_(AssignStmtPtr) accepts.
      const bool assignable = As<Var>(actual) || As<MemRef>(actual);
      const bool rebound = def_collector.var_defs.count(param.get()) > 0;
      const bool computed_shaped =
          As<Call>(actual) && actual_type && (AsTensorTypeLike(actual_type) || As<TileType>(actual_type));
      // Scalar value rebinding and shaped-handle rebinding stay callee-local.
      // Direction alone cannot classify the latter: @pl.jit.inline strips
      // Out/InOut, and even an explicitly annotated output can be rebound to
      // a new value without writing its old storage. Array's existing update
      // convention is separate: a bare array alias is not codegen-declarable.
      const ParamDirection direction = callee->param_directions_[i];
      const TypePtr param_type = param->GetType();
      const bool scalar_value_rebind = As<ScalarType>(param_type) && direction != ParamDirection::Out &&
                                       direction != ParamDirection::InOut;
      const bool shaped_value_rebind =
          (AsTensorTypeLike(param_type) || As<TileType>(param_type)) && local_bindings.count(param.get()) > 0;
      const bool pass_by_value_rebind = scalar_value_rebind || shaped_value_rebind;
      if ((rebound && (!assignable || pass_by_value_rebind)) || computed_shaped) {
        INTERNAL_CHECK_SPAN(actual_type, actual->span_)
            << "Internal error: argument bound at the call site for inline parameter '" << param->name_hint_
            << "' of '" << callee->name_ << "' has no type";
        auto bound = std::make_shared<Var>(FreshName(param->name_hint_), actual_type, actual->span_);
        arg_bindings.push_back(std::make_shared<const AssignStmt>(bound, actual, actual->span_));
        if (auto var = AsVarLike(actual); var && tagged_args.count(var.get())) {
          dump_vars.push_back(bound);
        }
        actual = bound;
      }
      seed[param.get()] = actual;
    }

    // 2. Let DeepClone create fresh locals so its existing type remapping also
    //    substitutes dynamic dimensions and references to other cloned locals.
    //    Pre-seeding fresh Vars would bypass that remapping: seeded replacements
    //    are intentionally used verbatim to preserve the caller's arguments.
    auto renamed_body = DeepClone(callee->body_, seed, /*clone_def_vars=*/true, FreshName).cloned_body;
    // Transfer before nested inlining: a nested helper may introduce another
    // local handle and needs the tag on its call to map that handle in turn.
    if (!dump_vars.empty()) {
      InlineDumpVarTransfer attacher(std::move(dump_vars));
      renamed_body = attacher.VisitStmt(renamed_body);
    }
    // Preserve seeded caller bindings even when a writeback refines
    // the RHS view; subsequent caller uses must still observe the rebinding.
    for (const auto& param : callee->params_) {
      if (auto var = AsVarLike(seed.at(param.get()))) preserved_vars_.insert(var.get());
    }
    // Keep cloned definitions alive while this mutator's pointer-keyed maps
    // serve subsequent call sites. Process the body before wiring its returns.
    retained_.push_back(renamed_body);
    for (auto& binding : arg_bindings) binding = VisitStmt(binding);
    renamed_body = VisitStmt(renamed_body);

    std::unordered_set<const Var*> unbound_dimensions;
    for (const auto& param : callee->params_) {
      for (const auto* var : var_collectors::CollectTypeVars(param->GetType())) {
        if (seed.count(var) == 0 && caller_dimensions_.count(var) == 0) unbound_dimensions.insert(var);
      }
    }
    // Caller-owned symbols in actual arguments are valid even when the callee's
    // other arguments cannot establish a single substitution for them.
    for (const auto& arg : args) {
      for (const auto* var : var_collectors::CollectTypeVars(arg->GetType())) {
        unbound_dimensions.erase(var);
      }
      var_collectors::VarDefUseCollector uses;
      uses.VisitExpr(arg);
      for (const auto* var : uses.var_uses) unbound_dimensions.erase(var);
    }
    if (!unbound_dimensions.empty()) {
      UnboundInlineDimensionChecker(callee, unbound_dimensions).VisitStmt(renamed_body);
    }

    // 4. Walk renamed_body and separate trailing ReturnStmt from the rest.
    std::vector<StmtPtr> spliced = std::move(arg_bindings);  // must precede the body
    std::vector<ExprPtr> return_values;
    bool has_return = false;

    auto extract_from_stmt = [&](const StmtPtr& s) {
      auto seq = std::dynamic_pointer_cast<const SeqStmts>(s);
      if (!seq) {
        // Single statement — could be the ReturnStmt itself or some other stmt
        auto ret = std::dynamic_pointer_cast<const ReturnStmt>(s);
        if (ret) {
          return_values = ret->value_;
          has_return = true;
        } else {
          spliced.push_back(s);
        }
        return;
      }
      for (const auto& sub : seq->stmts_) {
        auto ret = std::dynamic_pointer_cast<const ReturnStmt>(sub);
        if (ret) {
          return_values = ret->value_;
          has_return = true;
          // Anything after a return is dead; stop here.
          break;
        }
        spliced.push_back(sub);
      }
    };
    extract_from_stmt(renamed_body);

    // 4a. Reject any ReturnStmt that survived extraction (nested inside an
    //     If/For/While branch, or non-trailing). Such returns would otherwise
    //     splice straight into the caller and trigger the OUTER function to
    //     return prematurely. The pre-splice total-count alone isn't enough:
    //     a single ReturnStmt nested in `if c: return x` passes a `count <= 1`
    //     check yet still miscompiles, especially for EvalStmt call sites
    //     where there's no LHS-driven `has_return` guard downstream.
    NestedReturnCounter post_extract;
    for (const auto& s : spliced) post_extract.VisitStmt(s);
    INTERNAL_CHECK_SPAN(post_extract.count == 0, callee->span_)
        << "Inline function '" << callee->name_
        << "' contains a non-trailing ReturnStmt; only a single trailing return is "
           "supported (early-return inside an If/For/While branch is rejected)";

    return SplicedInlineBody{std::move(spliced), std::move(return_values), has_return};
  }

  // Rewrite @p stmt's OWN expressions, hoisting every nested Call to an inline
  // callee onto a fresh AssignStmt placed before it. Bodies are deliberately
  // left alone — the caller still recurses into them, so a hoist inside a loop
  // or branch body lands inside that body rather than escaping to here.
  //
  // Returns the replacement sequence (hoisted assignments, then the rewritten
  // statement) or std::nullopt when nothing moved.
  //
  // Out of scope on purpose, each reported by
  // IRProperty::InlineFunctionsEliminated right after this pass rather than
  // silently mis-lowered:
  //  - WhileStmt condition: re-evaluated every iteration, so hoisting it would
  //    evaluate the spliced body exactly once. The other positions handled here
  //    are all evaluated exactly once at the point the hoisted statement lands,
  //    so hoisting them preserves semantics.
  //  - IterArg init values, and a bare (non-SeqStmts) ForStmt / IfStmt body:
  //    not intercepted, matching the pass's existing bare-body coverage.
  std::optional<std::vector<StmtPtr>> HoistNestedInlineCalls(const StmtPtr& stmt) {
    std::vector<StmtPtr> pending;
    NestedInlineCallHoister hoister(inline_fns_, &pending);

    // A Call already sitting where HandleTopLevelInlineCall splices it stays
    // put; only its arguments are hoisted.
    auto rewrite = [&](const ExprPtr& value) -> ExprPtr {
      auto call = As<Call>(value);
      if (call && LookupInlineCallee(call)) {
        return hoister.HoistInArgs(call);
      }
      return hoister.Hoist(value);
    };
    auto rewrite_all = [&](const std::vector<ExprPtr>& values, std::vector<ExprPtr>* out) {
      bool changed = false;
      out->reserve(values.size());
      for (const auto& v : values) {
        auto nv = hoister.Hoist(v);
        if (nv.get() != v.get()) changed = true;
        out->push_back(std::move(nv));
      }
      return changed;
    };

    StmtPtr rewritten = stmt;
    if (auto assign = As<AssignStmt>(stmt)) {
      auto new_value = rewrite(assign->value_);
      if (new_value.get() != assign->value_.get()) {
        auto copy = MutableCopy(assign);
        copy->value_ = std::move(new_value);
        rewritten = copy;
      }
    } else if (auto eval = As<EvalStmt>(stmt)) {
      auto new_expr = rewrite(eval->expr_);
      if (new_expr.get() != eval->expr_.get()) {
        auto copy = MutableCopy(eval);
        copy->expr_ = std::move(new_expr);
        rewritten = copy;
      }
    } else if (auto ret = As<ReturnStmt>(stmt)) {
      std::vector<ExprPtr> new_values;
      bool changed = false;
      if (ret->value_.size() == 1) {
        // `return inline_call(...)` is a splice site; anything else is hoisted.
        new_values.push_back(rewrite(ret->value_[0]));
        changed = new_values[0].get() != ret->value_[0].get();
      } else {
        changed = rewrite_all(ret->value_, &new_values);
      }
      if (changed) {
        auto copy = MutableCopy(ret);
        copy->value_ = std::move(new_values);
        rewritten = copy;
      }
    } else if (auto loop = As<ForStmt>(stmt)) {
      auto new_start = hoister.Hoist(loop->start_);
      auto new_stop = hoister.Hoist(loop->stop_);
      auto new_step = loop->step_ ? hoister.Hoist(loop->step_) : loop->step_;
      if (new_start.get() != loop->start_.get() || new_stop.get() != loop->stop_.get() ||
          new_step.get() != loop->step_.get()) {
        auto copy = MutableCopy(loop);
        copy->start_ = std::move(new_start);
        copy->stop_ = std::move(new_stop);
        copy->step_ = std::move(new_step);
        rewritten = copy;
      }
    } else if (auto branch = As<IfStmt>(stmt)) {
      auto new_condition = hoister.Hoist(branch->condition_);
      if (new_condition.get() != branch->condition_.get()) {
        auto copy = MutableCopy(branch);
        copy->condition_ = std::move(new_condition);
        rewritten = copy;
      }
    } else if (auto yield = As<YieldStmt>(stmt)) {
      std::vector<ExprPtr> new_values;
      if (rewrite_all(yield->value_, &new_values)) {
        auto copy = MutableCopy(yield);
        copy->value_ = std::move(new_values);
        rewritten = copy;
      }
    }

    if (pending.empty()) return std::nullopt;
    pending.push_back(std::move(rewritten));
    return pending;
  }

  // Splice each statement of a hoisted sequence as an ordinary call site. Doing
  // it here rather than deferring to the next fixpoint iteration keeps the
  // pass within its `inline_fns.size() + 1` iteration bound.
  StmtPtr SpliceHoisted(const std::vector<StmtPtr>& work, const Span& span) {
    changed_ = true;
    std::vector<StmtPtr> out;
    out.reserve(work.size());
    for (const auto& s : work) {
      out.push_back(VisitStmt(s));
    }
    return SeqStmts::Flatten(std::move(out), span);
  }

  // Recognise `LHS = inline_call(args...)`, `EvalStmt(inline_call(args...))`,
  // or `ReturnStmt({inline_call(args...)})` and return the spliced sequence;
  // otherwise return std::nullopt.
  std::optional<std::vector<StmtPtr>> HandleTopLevelInlineCall(const StmtPtr& stmt) {
    std::optional<std::vector<StmtPtr>> spliced;
    std::vector<VarPtr> call_dump_vars;
    if (auto call = transform_utils::GetCallFromStmt(stmt)) {
      if (auto callee = LookupInlineCallee(call)) {
        call = As<Call>(VisitExpr(call));
        call_dump_vars = call->GetAttr<std::vector<VarPtr>>(kAttrDumpVars);
        if (auto assign = As<AssignStmt>(stmt)) {
          spliced = SpliceAssignCallSite(callee, call->args_, assign->var_, assign->span_, call_dump_vars);
        } else if (auto eval = As<EvalStmt>(stmt)) {
          spliced = SpliceInlineCallAsEval(callee, CloneInlineBody(callee, call->args_, call_dump_vars));
        }
      }
    }
    // ReturnStmt's value list isn't covered by GetCallFromStmt — handle it
    // here. The form is exactly `return inline_call(args...)`: a single Call
    // expression as the only return value.
    if (!spliced.has_value()) {
      if (auto ret = As<ReturnStmt>(stmt); ret && ret->value_.size() == 1) {
        if (auto call = As<Call>(ret->value_[0])) {
          if (auto callee = LookupInlineCallee(call)) {
            call = As<Call>(VisitExpr(call));
            call_dump_vars = call->GetAttr<std::vector<VarPtr>>(kAttrDumpVars);
            spliced = SpliceInlineCallAsReturn(callee, CloneInlineBody(callee, call->args_, call_dump_vars),
                                               ret->span_);
          }
        }
      }
    }
    return spliced;
  }

  // Dispatch on the trailing ReturnStmt: single-return → `LHS = value`
  // AssignStmt; tuple-return → record `LHS → values` for downstream
  // TupleGetItemExpr substitution and emit no LHS assignment.  Do not use
  // Function::return_types_ here: an annotation such as ``tuple[T, Scalar]``
  // is one TupleType entry even though the IR ReturnStmt has two values.
  std::vector<StmtPtr> SpliceAssignCallSite(const FunctionPtr& callee, const std::vector<ExprPtr>& args,
                                            const VarPtr& lhs, const Span& span,
                                            const std::vector<VarPtr>& dump_vars) {
    auto body = CloneInlineBody(callee, args, dump_vars);
    if (InlineReturnsTuple(callee)) {
      std::vector<ExprPtr> sub;
      auto stmts = SpliceInlineCallAsTupleSub(callee, std::move(body), sub);
      tuple_subs_[lhs.get()] = std::move(sub);
      return stmts;
    }
    auto stmts = SpliceInlineCallAsAssign(callee, std::move(body), lhs, span);
    // The cloned body is already visited; only the new receiver assignment
    // still needs to propagate its returned type to subsequent caller uses.
    if (!stmts.empty()) {
      if (auto assign = As<AssignStmt>(stmts.back()); assign && assign->var_ == lhs) {
        stmts.back() = VisitStmt(assign);
      }
    }
    return stmts;
  }

  FunctionPtr LookupInlineCallee(const CallPtr& call) const {
    auto gv = As<GlobalVar>(call->op_);
    if (!gv) return nullptr;
    auto it = inline_fns_.find(gv->name_);
    if (it == inline_fns_.end()) return nullptr;
    return it->second;
  }

 private:
  std::vector<IRNodePtr> retained_;
  std::unordered_map<const Var*, VarPtr> definitions_;
  std::unordered_set<const Var*> preserved_vars_;
  const std::unordered_map<std::string, FunctionPtr>& inline_fns_;
  const InlineFunctionMap& functions_;
  std::unordered_set<const Var*> caller_dimensions_;
  bool changed_ = false;
  // LHS Var → return values, populated by SpliceAssignCallSite for multi-return
  // call sites. Subsequent TupleGetItemExpr uses of the Var are substituted
  // with the corresponding value, so the LHS Var ends up with no references
  // and we never emit a `LHS = MakeTuple(...)` assignment.
  std::unordered_map<const Var*, std::vector<ExprPtr>> tuple_subs_;
};

}  // namespace

namespace pass {

/**
 * @brief Pass that eliminates FunctionType::Inline functions by splicing their
 *        bodies at every call site.
 *
 * Runs as the first pipeline pass. Subsequent passes never observe Inline
 * functions or Calls to them.
 *
 * Algorithm:
 *  1. Collect all FunctionType::Inline functions in the program.
 *  2. Detect cycles in the Inline → Inline call graph (raise on cycle).
 *  3. Iterate all non-Inline AND Inline functions, splicing top-level
 *     `LHS = inline_call(...)` or `EvalStmt(inline_call(...))` statements
 *     with the inlined body (alpha-rename + param substitution).
 *  4. Repeat (3) to fixpoint so that Inline-calls-Inline is fully expanded.
 *  5. Drop all Inline functions from the program.
 *
 * Edge cases:
 *  - Multi-return inline: does NOT emit `LHS = MakeTuple([rets...])` —
 *    orchestration codegen can't lower `MakeTuple`. Instead, the cloned
 *    return values are recorded in `tuple_subs_` keyed by the LHS Var, and
 *    downstream `TupleGetItemExpr(LHS, i)` uses are rewritten to the
 *    corresponding cloned value (see `SpliceInlineCallAsTupleSub`). The LHS
 *    binding ends up unreferenced and is elided.
 *  - `return inline_call(...)`: spliced via `SpliceInlineCallAsReturn` to the
 *    cloned pre-return body followed by a fresh ReturnStmt over the cloned
 *    trailing values (single or multi).
 *  - `EvalStmt(inline_call(...))` discards the callee's trailing return value,
 *    but not its evaluation: every discarded Call and Submit is re-emitted as
 *    an EvalStmt (see `SpliceInlineCallAsEval`) so an ignored wrapper ending in
 *    `return self.inner(...)`, `return pl.tile.store(...)` or
 *    `return pl.system.set_ffts(...)` keeps its write, store or hardware setup.
 *  - Nested Call to inline (e.g. inside a binary expression) is left alone in
 *    v1; the verifier flags any surviving Calls to Inline functions.
 *  - Inline function with no callers is silently dropped in step (5) — that
 *    naturally covers the "Inline function as program entry" case too: with
 *    no Call sites it just disappears in the cleanup phase.
 *  - Inline body containing a non-trailing ReturnStmt is rejected at splice
 *    time with a CHECK (only a single trailing return is supported).
 */
Pass InlineFunctions() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Collect inline functions
    std::unordered_map<std::string, FunctionPtr> inline_fns;
    for (const auto& [gvar, fn] : program->functions_) {
      if (fn->func_type_ == FunctionType::Inline) {
        INTERNAL_CHECK_SPAN(inline_fns.count(fn->name_) == 0, fn->span_)
            << "Duplicate FunctionType::Inline function name '" << fn->name_ << "' in program";
        inline_fns[fn->name_] = fn;
      }
    }

    // Fast path: nothing to do
    if (inline_fns.empty()) return program;

    // Cycle detection
    DetectInlineCycles(inline_fns);

    // Iterate to fixpoint. Each iteration mutates every function (incl. Inline
    // ones, so that Inline-calls-Inline expands too). The loop terminates after
    // at most (inline_fns.size() + 1) iterations because each iteration either
    // makes progress or hits the fixpoint.
    std::unordered_map<std::string, FunctionPtr> current;
    for (const auto& [gvar, fn] : program->functions_) {
      current[fn->name_] = fn;
    }

    const size_t max_iters = inline_fns.size() + 1;
    for (size_t iter = 0; iter < max_iters; ++iter) {
      bool any_changed = false;

      // Refresh inline_fns view to point at the *latest* bodies — important
      // because a previous iteration may have inlined Inline-calls-Inline.
      std::unordered_map<std::string, FunctionPtr> latest_inline;
      for (const auto& [name, fn] : inline_fns) {
        latest_inline[name] = current[name];
      }

      for (auto& [name, fn] : current) {
        InlineCallsMutator mutator(latest_inline, fn, current);
        auto new_body = mutator.VisitStmt(fn->body_);
        if (mutator.Changed()) {
          auto updated = MutableCopy(fn);
          updated->body_ = new_body;
          fn = updated;
          any_changed = true;
        }
      }

      if (!any_changed) break;

      INTERNAL_CHECK(iter + 1 < max_iters) << "InlineFunctions did not reach a fixpoint within " << max_iters
                                           << " iterations; this indicates a bug or an undetected cycle";
    }

    // Drop inline functions and rebuild the program
    std::vector<FunctionPtr> kept_functions;
    for (const auto& [gvar, fn] : program->functions_) {
      auto it = current.find(fn->name_);
      INTERNAL_CHECK(it != current.end()) << "Internal error: function '" << fn->name_ << "' missing";
      const auto& latest = it->second;
      if (latest->func_type_ == FunctionType::Inline) continue;
      kept_functions.push_back(latest);
    }

    return std::make_shared<Program>(kept_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "InlineFunctions", kInlineFunctionsProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto

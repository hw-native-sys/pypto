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

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

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
// Collect all defining Vars in a function body (excludes the function's params)
// =============================================================================

// Collects Vars whose binding sites must be alpha-renamed at each splice.
//
// We deliberately omit `iter_args_` of For/While loops: the base IRMutator
// already mints fresh IterArg instances per visit (see mutator.cpp:581 / 664).
// Including them here would seed `rename_map_` with entries the base mutator
// later overwrites/erases, leading to inconsistent def-use after the splice.
class DefVarCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> defs;

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op->var_) defs.insert(op->var_.get());
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (op->loop_var_) defs.insert(op->loop_var_.get());
    for (const auto& v : op->return_vars_) {
      if (v) defs.insert(v.get());
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& v : op->return_vars_) {
      if (v) defs.insert(v.get());
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& v : op->return_vars_) {
      if (v) defs.insert(v.get());
    }
    IRVisitor::VisitStmt_(op);
  }
};

// =============================================================================
// Splice an inline call site
// =============================================================================

// Process-wide counter ensuring distinct fresh names across multiple call
// sites of the same inline function. Pass execution is sequential, so a
// plain static int suffices.
//
// Single underscore is intentional — `__` is reserved by the IR auto-naming
// utility (see auto_name_utils.h::ValidateBaseName) for its own
// `name__role__version` scheme; re-using `__` here would trip the validator
// when downstream passes rename the inlined Vars again.
std::string FreshName(const std::string& orig) {
  static int counter = 0;
  return orig + "_inline" + std::to_string(counter++);
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

// Result of splicing an inline call's body without yet wiring up its return
// values into a specific caller statement. The caller picks the wiring form
// (assign / drop / return / ...) based on its own statement kind.
struct SplicedInlineBody {
  std::vector<StmtPtr> stmts;          // Pre-return statements, in order
  std::vector<ExprPtr> return_values;  // Trailing-return values (empty if has_return is false)
  bool has_return;                     // Whether the body ended with a ReturnStmt
};

// Steps 1-3 of an inline splice — call-site-form-agnostic. Used by every
// SpliceInlineCall* helper.
//
//   1. Build the substitution seed: params → actual args, locally-defined
//      Vars → fresh `_inlineN` Vars.
//   2. DeepClone the callee body with that seed. The clone uses the same
//      substitution at use-sites and def-sites, so a rebinding of a param
//      (`out = pl.assemble(out, ...)`) collapses to `actual = pl.assemble(
//      actual, ...)` whenever the actual arg is a Var.
//   3. Split the cloned body into pre-return statements and the trailing
//      return value(s); reject any non-trailing return.
SplicedInlineBody CloneInlineBody(const FunctionPtr& callee, const std::vector<ExprPtr>& args) {
  INTERNAL_CHECK_SPAN(callee->params_.size() == args.size(), callee->span_)
      << "Internal error: inline call to '" << callee->name_ << "' has " << args.size()
      << " argument(s) but callee expects " << callee->params_.size()
      << " (parser/type-checker should have caught arity mismatch before InlineFunctions)";

  // 1. Build the seed substitution map for DeepClone:
  //    - Each param Var → its actual-arg Expr. The same substitution is
  //      consulted at both use-sites and def-sites of the param, so a
  //      rebinding `out = pl.assemble(out, ...)` where `out` is a param
  //      becomes `q_out = pl.assemble(q_out, ...)` when the actual arg is
  //      the Var `q_out` (the natural pre-SSA in-place semantics for
  //      pl.Out parameters). If the actual arg is not a Var and the param
  //      is rebound, IRMutator::VisitStmt_(AssignStmtPtr)'s Var-cast check
  //      fires with a clear span-tagged error.
  //    - Each locally-defined Var → a fresh Var with a `_inlineN` name so
  //      multi-call-site expansions of the same callee remain
  //      distinguishable in IR dumps. DeepClone uses the seeded fresh Var
  //      verbatim; only Vars NOT in the seed receive an auto-cloned copy
  //      with their original name_hint as a safety fallback.
  std::unordered_map<const Var*, ExprPtr> seed;
  for (size_t i = 0; i < callee->params_.size(); ++i) {
    seed[callee->params_[i].get()] = args[i];
  }
  DefVarCollector def_collector;
  def_collector.VisitStmt(callee->body_);
  for (const Var* v : def_collector.defs) {
    if (seed.count(v) > 0) continue;  // param — already seeded with actual arg
    auto fresh = std::make_shared<Var>(FreshName(v->name_hint_), v->GetType(), v->span_);
    seed[v] = fresh;
  }

  // 2. Deep-clone the body with substitutions applied. DeepClone visits
  //    every DefField Var via the same VisitExpr_(VarPtr) pathway as
  //    use-sites, so the seed map covers both — no separate def-site map
  //    is needed. clone_def_vars=true is a safety net: if DefVarCollector
  //    misses a binding kind, DeepClone still produces a fresh Var rather
  //    than leaving the callee's original Var leaking into the caller.
  auto [renamed_body, _unused] = DeepClone(callee->body_, seed, /*clone_def_vars=*/true);

  // 3. Walk renamed_body and separate trailing ReturnStmt from the rest.
  std::vector<StmtPtr> spliced;
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

  // 3a. Reject any ReturnStmt that survived extraction (nested inside an
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

// Splice an EvalStmt-shaped call (no LHS) — drop the return, return only
// the pre-return statements.
std::vector<StmtPtr> SpliceInlineCallAsEval(const FunctionPtr& callee, const std::vector<ExprPtr>& args) {
  auto body = CloneInlineBody(callee, args);
  return std::move(body.stmts);
}

// Splice a single-return call into `LHS = inlined_return_value`. CHECK-fails
// if the callee returns multiple values — multi-return goes through
// SpliceInlineCallAsTupleSub, which avoids the dead `LHS = MakeTuple(...)`.
std::vector<StmtPtr> SpliceInlineCallAsAssign(const FunctionPtr& callee, const std::vector<ExprPtr>& args,
                                              const VarPtr& lhs, const Span& call_site_span) {
  auto body = CloneInlineBody(callee, args);
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
// Why no MakeTuple: orchestration codegen can't lower a MakeTuple expression,
// and the parser-generated tuple-unpack pattern (`_tuple_tmp = call(); y_i =
// _tuple_tmp[i]`) means every LHS use is a TupleGetItemExpr — substituting
// makes the LHS unreferenced and the binding effectively dead.
std::vector<StmtPtr> SpliceInlineCallAsTupleSub(const FunctionPtr& callee, const std::vector<ExprPtr>& args,
                                                std::vector<ExprPtr>& out_substitution) {
  auto body = CloneInlineBody(callee, args);
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
          out_substitution = tuple->elements_;
          body.stmts.erase(body.stmts.begin() + i);
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
std::vector<StmtPtr> SpliceInlineCallAsReturn(const FunctionPtr& callee, const std::vector<ExprPtr>& args,
                                              const Span& call_site_span) {
  auto body = CloneInlineBody(callee, args);
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
// The dump entries are caller arg Vars; ``CloneInlineBody`` has already
// substituted each in for its matching callee param, so a tagged arg consumed
// inside the callee now appears verbatim in the spliced body. Two carriers are
// stamped (both round-trip and are tracked by Var identity downstream):
//
//   * ``with pl.at(...)`` scopes whose body uses a tagged arg get it merged into
//     their ``kAttrDumpVars`` — the same scope-level carrier ``pl.dump_tag``
//     seeds at parse. The outliner later maps it onto the synthesised dispatch.
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
  StmtPtr VisitStmt_(const SplitAivScopeStmtPtr& op) override { return Attach<SplitAivScopeStmt>(op); }

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
// InlineCallsMutator — walks a function body and replaces top-level inline-call
// statements with the spliced inline body.
// =============================================================================

class InlineCallsMutator : public IRMutator {
 public:
  explicit InlineCallsMutator(const std::unordered_map<std::string, FunctionPtr>& inline_fns)
      : inline_fns_(inline_fns) {}

  bool Changed() const { return changed_; }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    bool any_changed = false;
    for (const auto& stmt : op->stmts_) {
      auto handled = HandleTopLevelInlineCall(stmt);
      if (handled.has_value()) {
        for (auto& s : *handled) new_stmts.push_back(std::move(s));
        any_changed = true;
        changed_ = true;
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
    auto handled = HandleTopLevelInlineCall(op);
    if (!handled.has_value()) return IRMutator::VisitStmt_(op);
    changed_ = true;
    return SeqStmts::Flatten(std::move(*handled), op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto handled = HandleTopLevelInlineCall(op);
    if (!handled.has_value()) return IRMutator::VisitStmt_(op);
    changed_ = true;
    return SeqStmts::Flatten(std::move(*handled), op->span_);
  }

  // `return inline_call(...)`. Same SeqStmts caveat as AssignStmt /
  // EvalStmt: a function body that is a bare ReturnStmt (no enclosing
  // SeqStmts) reaches this override directly.
  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
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
  // Recognise `LHS = inline_call(args...)`, `EvalStmt(inline_call(args...))`,
  // or `ReturnStmt({inline_call(args...)})` and return the spliced sequence;
  // otherwise return std::nullopt.
  std::optional<std::vector<StmtPtr>> HandleTopLevelInlineCall(const StmtPtr& stmt) {
    std::optional<std::vector<StmtPtr>> spliced;
    std::vector<VarPtr> call_dump_vars;
    if (auto call = transform_utils::GetCallFromStmt(stmt)) {
      if (auto callee = LookupInlineCallee(call)) {
        call_dump_vars = call->GetAttr<std::vector<VarPtr>>(kAttrDumpVars);
        if (auto assign = As<AssignStmt>(stmt)) {
          spliced = SpliceAssignCallSite(callee, call->args_, assign->var_, assign->span_);
        } else if (auto eval = As<EvalStmt>(stmt)) {
          spliced = SpliceInlineCallAsEval(callee, call->args_);
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
            call_dump_vars = call->GetAttr<std::vector<VarPtr>>(kAttrDumpVars);
            spliced = SpliceInlineCallAsReturn(callee, call->args_, ret->span_);
          }
        }
      }
    }
    // Carry the call-site selective-dump tags onto the spliced scopes — the
    // inline Call node (which held ``kAttrDumpVars``) is about to be destroyed,
    // so the dump intent must move onto the surviving scope bodies (see
    // InlineDumpVarTransfer) to reach the outliner by Var identity.
    if (spliced.has_value() && !call_dump_vars.empty()) {
      InlineDumpVarTransfer attacher(std::move(call_dump_vars));
      for (auto& s : *spliced) s = attacher.VisitStmt(s);
    }
    return spliced;
  }

  // Dispatch on the trailing ReturnStmt: single-return → `LHS = value`
  // AssignStmt; tuple-return → record `LHS → values` for downstream
  // TupleGetItemExpr substitution and emit no LHS assignment.  Do not use
  // Function::return_types_ here: an annotation such as ``tuple[T, Scalar]``
  // is one TupleType entry even though the IR ReturnStmt has two values.
  std::vector<StmtPtr> SpliceAssignCallSite(const FunctionPtr& callee, const std::vector<ExprPtr>& args,
                                            const VarPtr& lhs, const Span& span) {
    if (InlineReturnsTuple(callee)) {
      std::vector<ExprPtr> sub;
      auto stmts = SpliceInlineCallAsTupleSub(callee, args, sub);
      tuple_subs_[lhs.get()] = std::move(sub);
      return stmts;
    }
    return SpliceInlineCallAsAssign(callee, args, lhs, span);
  }

  FunctionPtr LookupInlineCallee(const CallPtr& call) const {
    auto gv = As<GlobalVar>(call->op_);
    if (!gv) return nullptr;
    auto it = inline_fns_.find(gv->name_);
    if (it == inline_fns_.end()) return nullptr;
    return it->second;
  }

 private:
  const std::unordered_map<std::string, FunctionPtr>& inline_fns_;
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
        InlineCallsMutator mutator(latest_inline);
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

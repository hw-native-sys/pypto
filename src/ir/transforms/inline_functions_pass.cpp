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
// VarSubstituteMutator — substitutes param Vars with actual-arg Exprs and
// alpha-renames local Vars per the rename map.
// =============================================================================

class VarSubstituteMutator : public IRMutator {
 public:
  VarSubstituteMutator(std::unordered_map<const Var*, ExprPtr> param_subst,
                       std::unordered_map<const Var*, VarPtr> rename_map)
      : param_subst_(std::move(param_subst)), rename_map_(std::move(rename_map)) {}

  // Use-site Var refs: substitute params, then rename locals.
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto pit = param_subst_.find(op.get());
    if (pit != param_subst_.end()) return pit->second;
    auto rit = rename_map_.find(op.get());
    if (rit != rename_map_.end()) return rit->second;
    return op;
  }

  // Defining var in AssignStmt — rename if in map.
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_value = VisitExpr(op->value_);
    auto rit = rename_map_.find(op->var_.get());
    VarPtr new_var = (rit != rename_map_.end()) ? rit->second : op->var_;
    if (new_var.get() == op->var_.get() && new_value.get() == op->value_.get()) {
      return op;
    }
    return std::make_shared<const AssignStmt>(new_var, new_value, op->span_, op->leading_comments_);
  }

  // Defining vars in ForStmt — rename loop_var_ + return_vars_ if in map.
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto fp = std::dynamic_pointer_cast<const ForStmt>(base);
    INTERNAL_CHECK(fp) << "Internal error: VisitStmt_(ForStmtPtr) must return a ForStmt";

    auto loop_var_renamed = rename_map_.find(fp->loop_var_.get());
    auto new_return_vars = RenameVarVec(fp->return_vars_);
    bool any_renamed = (loop_var_renamed != rename_map_.end()) || new_return_vars.has_value();
    if (!any_renamed) return fp;

    auto result = MutableCopy(fp);
    if (loop_var_renamed != rename_map_.end()) result->loop_var_ = loop_var_renamed->second;
    if (new_return_vars.has_value()) result->return_vars_ = std::move(*new_return_vars);
    return result;
  }

  // Defining vars in WhileStmt.return_vars_ — rename each if in map.
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto wp = std::dynamic_pointer_cast<const WhileStmt>(base);
    INTERNAL_CHECK(wp) << "Internal error: VisitStmt_(WhileStmtPtr) must return a WhileStmt";

    auto new_return_vars = RenameVarVec(wp->return_vars_);
    if (!new_return_vars.has_value()) return wp;
    auto result = MutableCopy(wp);
    result->return_vars_ = std::move(*new_return_vars);
    return result;
  }

  // Defining vars in IfStmt.return_vars_ — rename each if in map.
  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto ip = std::dynamic_pointer_cast<const IfStmt>(base);
    INTERNAL_CHECK(ip) << "Internal error: VisitStmt_(IfStmtPtr) must return an IfStmt";

    auto new_return_vars = RenameVarVec(ip->return_vars_);
    if (!new_return_vars.has_value()) return ip;
    auto result = MutableCopy(ip);
    result->return_vars_ = std::move(*new_return_vars);
    return result;
  }

 private:
  // Rebuild a vector of VarPtr with renames applied. Returns nullopt if no
  // entry needed renaming (so the caller can keep the original instance).
  std::optional<std::vector<VarPtr>> RenameVarVec(const std::vector<VarPtr>& vars) const {
    bool any_renamed = false;
    std::vector<VarPtr> result;
    result.reserve(vars.size());
    for (const auto& v : vars) {
      auto rit = rename_map_.find(v.get());
      if (rit != rename_map_.end()) {
        result.push_back(rit->second);
        any_renamed = true;
      } else {
        result.push_back(v);
      }
    }
    if (!any_renamed) return std::nullopt;
    return result;
  }

  std::unordered_map<const Var*, ExprPtr> param_subst_;
  std::unordered_map<const Var*, VarPtr> rename_map_;
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

// Splice a call site into a SeqStmts. Returns the splice's statements, in order.
//
// For `LHS = inline_call(arg1, ...)`:
//   1. Reject nested ReturnStmts (only a single trailing return is supported).
//   2. Build param_subst: callee.params_[i] → args[i]
//   3. Build rename_map: every local def in callee.body → fresh Var
//   4. Walk callee.body with both maps applied, accumulating statements,
//      EXCEPT the trailing ReturnStmt
//   5. For the ReturnStmt:
//        - If lhs is set and return has 1 value: emit `LHS = renamed_ret[0]`
//        - If lhs is set and return has N values: emit `LHS = MakeTuple(...)`
//        - If lhs is null (EvalStmt): drop the return
std::vector<StmtPtr> SpliceInlineCall(const FunctionPtr& callee, const std::vector<ExprPtr>& args,
                                      const VarPtr& lhs, const Span& call_site_span) {
  CHECK(callee->params_.size() == args.size())
      << "Inline call to '" << callee->name_ << "' has " << args.size() << " argument(s) but callee expects "
      << callee->params_.size();

  // 1. Param substitution map
  std::unordered_map<const Var*, ExprPtr> param_subst;
  for (size_t i = 0; i < callee->params_.size(); ++i) {
    param_subst[callee->params_[i].get()] = args[i];
  }

  // 2. Rename map for locally-defined Vars
  DefVarCollector def_collector;
  def_collector.VisitStmt(callee->body_);
  std::unordered_map<const Var*, VarPtr> rename_map;
  for (const Var* v : def_collector.defs) {
    // Skip any Var that is also a param (params are substituted, not renamed)
    if (param_subst.count(v) > 0) continue;
    auto fresh = std::make_shared<Var>(FreshName(v->name_hint_), v->GetType(), v->span_);
    rename_map[v] = fresh;
  }

  // 3. Apply substitution to body
  VarSubstituteMutator mutator(param_subst, rename_map);
  StmtPtr renamed_body = mutator.VisitStmt(callee->body_);

  // 4. Walk renamed_body and separate trailing ReturnStmt from the rest.
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

  // 4a. Reject any ReturnStmt that survived extraction (nested inside an
  //     If/For/While branch, or non-trailing). Such returns would otherwise
  //     splice straight into the caller and trigger the OUTER function to
  //     return prematurely. The pre-splice total-count alone isn't enough:
  //     a single ReturnStmt nested in `if c: return x` passes a `count <= 1`
  //     check yet still miscompiles, especially for EvalStmt call sites
  //     where there's no LHS-driven `has_return` guard downstream.
  NestedReturnCounter post_extract;
  for (const auto& s : spliced) post_extract.VisitStmt(s);
  CHECK(post_extract.count == 0) << "Inline function '" << callee->name_
                                 << "' contains a non-trailing ReturnStmt; only a single trailing return is "
                                    "supported (early-return inside an If/For/While branch is rejected)";

  // 5. Wire up the return value(s) at the call site
  if (lhs) {
    CHECK(has_return) << "Inline function '" << callee->name_
                      << "' is called for its value but has no return statement";
    ExprPtr final_value;
    if (return_values.size() == 1) {
      final_value = return_values[0];
    } else {
      // Multi-return: pack into MakeTuple. The LHS is expected to have TupleType.
      final_value = std::make_shared<MakeTuple>(return_values, call_site_span);
    }
    // Skip the no-op `lhs = lhs` that arises when an arg is also returned —
    // it would otherwise survive into SSA and break structural equality.
    if (auto var_expr = As<Var>(final_value); var_expr && var_expr.get() == lhs.get()) {
      return spliced;
    }
    spliced.push_back(std::make_shared<const AssignStmt>(lhs, final_value, call_site_span));
  }

  return spliced;
}

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

 private:
  // Recognise `LHS = inline_call(args...)` or `EvalStmt(inline_call(args...))`
  // and return the spliced sequence; otherwise return std::nullopt.
  std::optional<std::vector<StmtPtr>> HandleTopLevelInlineCall(const StmtPtr& stmt) {
    auto call = transform_utils::GetCallFromStmt(stmt);
    if (!call) return std::nullopt;
    auto callee = LookupInlineCallee(call);
    if (!callee) return std::nullopt;

    if (auto assign = As<AssignStmt>(stmt)) {
      return SpliceInlineCall(callee, call->args_, assign->var_, assign->span_);
    }
    if (auto eval = As<EvalStmt>(stmt)) {
      return SpliceInlineCall(callee, call->args_, /*lhs=*/nullptr, eval->span_);
    }
    return std::nullopt;
  }

  FunctionPtr LookupInlineCallee(const CallPtr& call) const {
    auto gv = As<GlobalVar>(call->op_);
    if (!gv) return nullptr;
    auto it = inline_fns_.find(gv->name_);
    return (it == inline_fns_.end()) ? nullptr : it->second;
  }

  const std::unordered_map<std::string, FunctionPtr>& inline_fns_;
  bool changed_ = false;
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
 *  - Multi-return inline: emits `LHS = MakeTuple([rets...])`. Subsequent
 *    Simplify can fold `TupleGetItemExpr(MakeTuple(...), i)` if needed.
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
        CHECK(inline_fns.count(fn->name_) == 0)
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

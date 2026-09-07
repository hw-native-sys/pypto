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

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Rejects a Graph scope nested inside another one, and reports whether the
/// visited body contains a Graph scope at all.
///
/// The runtime cannot record a graph inside a graph (`pto_orchestrator.cpp`
/// treats a nested `graph_begin` as unsupported and falls the whole region back
/// to ordinary submits), and a fallback is silent — the program still computes
/// the right answer, it just loses every bit of the speedup the user asked for.
/// Catching it here turns that into a compile error naming both regions, which
/// is the only place the source-level nesting is still visible: after outlining
/// the inner region is a plain call and the nesting is gone.
///
/// The first region found rides along on the same traversal because every caller
/// needs both answers: this pass runs on every default compilation, and a
/// program with no Graph region at all must not pay for the outliner (see
/// `First()`).
class NestedGraphScopeChecker : public IRVisitor {
 public:
  explicit NestedGraphScopeChecker(std::string func_name) : func_name_(std::move(func_name)) {}

  void VisitStmt_(const GraphScopeStmtPtr& op) override {
    if (!op) return;
    CHECK_SPAN(outer_.empty(), op->span_)
        << "pl.graph(\"" << op->name_hint_ << "\") is nested inside pl.graph(\"" << outer_
        << "\") in function '" << func_name_
        << "'. The host_build_graph runtime cannot record a graph inside a graph, so keep the regions "
           "disjoint — mark either the outer region or the inner one, not both.";
    if (!first_) first_ = op;
    outer_ = op->name_hint_;
    IRVisitor::VisitStmt_(op);
    outer_.clear();
  }

  /// The first Graph scope seen, or null when the body contains none. Doubles as
  /// the presence flag and as the span source for diagnostics.
  [[nodiscard]] const GraphScopeStmtPtr& First() const { return first_; }

 private:
  std::string func_name_;
  std::string outer_;  ///< name of the enclosing Graph region, empty when at top level
  GraphScopeStmtPtr first_;
};

}  // namespace

namespace pass {

/**
 * @brief Pass to outline Graph scopes into separate Graph functions
 *
 * Transforms `GraphScopeStmt` (`with pl.graph("name"):`) into a
 * `FunctionType::Graph` definition plus a `Call` to it — the same shape a
 * hand-written `@pl.jit.graph` function reaches this point in. The scope form is
 * therefore sugar: no downstream pass, verifier or codegen path needs to know
 * which surface the user wrote. Parameter *order* is the one thing that differs
 * (the outliner appends in capture order, the decorator form uses the declared
 * signature); nothing downstream reads a boundary parameter by position.
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - InlineFunctions must have run (InlineFunctionsEliminated): the parser permits
 *   pl.graph inside an Inline body precisely because that body is spliced into
 *   its orchestration caller before this pass sees it
 * - Processes Opaque and orchestration-like (Orchestration / Graph) functions
 * - Runs immediately before OutlineIncoreScopes, so the InCore scopes inside the
 *   freshly outlined Graph body are outlined by that pass on the same terms as
 *   the ones inside a hand-written Graph function
 *
 * Transformation:
 * 1. Scan each function once, rejecting a Graph scope nested inside another
 *    (runtime C7), and leave a Graph-free function untouched — this pass runs on
 *    every compilation and must not make a program without one pay for it.
 * 2. Reject a surviving Graph scope in a function this pass does not outline,
 *    rather than skipping it and still advertising GraphOutlined.
 * 3. For each `GraphScopeStmt`, extract the body into a `Function(Graph)` named
 *    after the region, and replace the scope with a Call plus output bindings.
 * 4. Stamp the orchestration level/role on the outlined function, matching what
 *    the parser gives a `@pl.jit.graph` function.
 * 5. The parent function type is preserved — a Graph region does not change what
 *    the function containing it is.
 */
Pass OutlineGraphScopes() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> new_functions;
    std::vector<FunctionPtr> all_outlined_functions;

    // Program-wide set of outlined function names, seeded with the existing
    // function names and shared across every ScopeOutliner so that two regions
    // named alike in different functions get suffix-disambiguated instead of
    // colliding at Program construction (#1711).
    //
    // Disambiguation is safe for a Graph specifically: the emitted symbol is
    // derived from the final function name, so two distinct regions that both
    // asked to be called "layer" end up with distinct graph keys rather than
    // sharing one Definition.
    auto reserved_func_names = std::make_shared<std::unordered_set<std::string>>();
    for (const auto& [gvar, func] : program->functions_) {
      reserved_func_names->insert(func->name_);
    }

    for (const auto& [gvar, func] : program->functions_) {
      // One linear traversal answers both questions this pass asks of a body:
      // does it nest Graph regions (rejected), and does it contain one at all.
      NestedGraphScopeChecker nesting_checker(func->name_);
      nesting_checker.VisitStmt(func->body_);

      // Whether this pass would outline a Graph region out of this body at all.
      const bool outlinable =
          func->func_type_ == FunctionType::Opaque || IsOrchestrationLike(func->func_type_);

      // Fast path: no Graph region, nothing to outline. The outliner's own
      // block walk is linear in the absence of target scopes, so this is not
      // what keeps the pass inside the O(N log N) bound
      // (`.claude/rules/pass-complexity.md`); it simply skips building the
      // symbol table and walking the body a second time for the overwhelming
      // majority of programs, which contain no `pl.graph` at all.
      //
      // Skipping the outliner must not also skip the pass's copy-on-write
      // discipline: a body this pass would have rewritten still goes downstream
      // as its own Function, so the returned Program never shares a mutable node
      // with the one handed in.
      if (!nesting_checker.First()) {
        new_functions.push_back(outlinable ? MutableCopy(func) : func);
        continue;
      }

      // Outline Opaque and orchestration-like (Orchestration / Graph) bodies.
      // A body of any other type still holding a Graph region is a
      // pipeline-composition error, not something to skip: skipping would leave
      // the GraphScopeStmt in place while the pass still advertises
      // GraphOutlined, and `.required` is checked only by VerificationInstrument
      // — with verification off, that false property would reach codegen.
      //
      // Inline is the reachable case. The parser deliberately permits `pl.graph`
      // in an Inline body because InlineFunctions splices it into its caller
      // before this pass runs, which is what IRProperty::InlineFunctionsEliminated
      // in this pass's required set states. The device kernel types are rejected
      // at the parser, so reaching here with one is a compiler bug.
      if (!outlinable) {
        UNREACHABLE_SPAN(nesting_checker.First()->span_)
            << "pl.graph(...) survives in function '" << func->name_
            << "' (func_type=" << FunctionTypeToString(func->func_type_)
            << "), which OutlineGraphScopes does not outline. Run InlineFunctions before "
               "OutlineGraphScopes so that an Inline body carrying a Graph region is spliced into "
               "its orchestration caller first.";
      }

      // Build symbol table for this function
      outline_utils::VarCollector type_collector;
      for (const auto& var : func->params_) {
        type_collector.var_types[var.get()] = var->GetType();
        type_collector.var_objects[var.get()] = var;
        type_collector.known_names.insert(var->name_hint_);
      }
      type_collector.VisitStmt(func->body_);

      // `program` resolves a GlobalVar callee, which is what lets a capture the
      // region only ever hands to an inner kernel's `Out`/`InOut` slot come out
      // as anything other than the seeded `In`. Null is *not* the conservative
      // choice on the write side: `CallWriteTargets` answers from the operator
      // registry, so a call to a *function* contributes no write evidence at
      // all, and the capture is under-declared `In`. Under-declaring is the
      // direction that fails silently — an `In` boundary tensor loses its RAW
      // edge, where an over-declared one only over-orders.
      //
      // A Graph region is where this bites: its body is a topology of kernel
      // *calls*, whereas the InCore/hierarchy outliners extract bodies of
      // registry-backed tile ops that Step 1 already sees. Resolution is
      // against the pass's input program, which is what the region's calls name
      // — this pass only ever appends outlined functions.
      outline_utils::ScopeOutliner outliner(func->name_, type_collector.var_types, type_collector.var_objects,
                                            type_collector.known_names, ScopeKind::Graph, FunctionType::Graph,
                                            "_graph_", program, reserved_func_names);
      auto new_body = outliner.VisitStmt(func->body_);

      // Preserve the parent function type — containing a Graph region says
      // nothing about what the enclosing function is.
      auto new_func = MutableCopy(func);
      new_func->body_ = new_body;
      new_functions.push_back(new_func);

      // No level/role stamping here: the outliner passes nullopt for both, and
      // `Function`'s constructor already derives `Level::CHIP` +
      // `Role::Orchestrator` for any orchestration-like type, Graph included. An
      // explicit copy would write exactly what is already there.
      for (const auto& outlined : outliner.GetOutlinedFunctions()) {
        INTERNAL_CHECK(outlined) << "Internal error: OutlineGraphScopes produced a null function";
        all_outlined_functions.push_back(outlined);
      }
    }

    // Add all outlined functions before the originals
    all_outlined_functions.insert(all_outlined_functions.end(), new_functions.begin(), new_functions.end());

    return std::make_shared<Program>(all_outlined_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "OutlineGraphScopes", kOutlineGraphScopesProperties);
}

}  // namespace pass

// ============================================================================
// GraphOutlined property verifier
// ============================================================================

namespace {

using GraphOutlinedVerifier = outline_utils::ScopeKindAbsenceVerifier<ScopeKind::Graph>;

}  // namespace

class GraphOutlinedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "GraphOutlined"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // No function type is allowed to keep a Graph scope: the region is always
      // outlined into a function of its own, including out of a Graph body (the
      // pass rejects that nesting outright).
      GraphOutlinedVerifier verifier(diagnostics, "GraphOutlined",
                                     "Graph ScopeStmt found in function (should have been outlined)");
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateGraphOutlinedPropertyVerifier() {
  return std::make_shared<GraphOutlinedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

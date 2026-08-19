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

#include <algorithm>
#include <cstddef>
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
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace pass {

namespace {

/// SPMD launch spec lifted out of a Group function's nested ``pl.spmd`` scope,
/// together with the Group it came from (needed to translate the spec back into
/// caller scope). ``core_num`` is null when the Group had no nested Spmd scope.
struct SpmdLaunchSpec {
  ExprPtr core_num;
  bool sync_start = false;
  FunctionPtr group;
  /// Every Var the Group binds (params + body definitions). ``core_num`` must
  /// reference none of them once translated into caller scope — see
  /// ``LaunchSpecStamper::ToCallerScope``.
  std::unordered_set<const Var*> callee_bound;
};

/// Unwrap a nested Spmd scope in a Group function body: replace the
/// ScopeStmt(Spmd) with its body and hand the launch spec back to the caller,
/// which attaches it to the Group's DISPATCH.
///
/// The spec must not stay on the Group (see ``kAttrCoreNum``); what does stay is
/// the self-contained ``kAttrSpmdUnwrapped`` marker. core_num is propagated as
/// an ExprPtr — codegen is responsible for evaluating it.
FunctionPtr UnwrapNestedSpmd(const FunctionPtr& group_func, SpmdLaunchSpec* spec_out) {
  class SpmdUnwrapper : public IRMutator {
   public:
    ExprPtr core_num;
    std::optional<bool> sync_start;

   protected:
    StmtPtr VisitStmt_(const SpmdScopeStmtPtr& op) override {
      INTERNAL_CHECK_SPAN(core_num == nullptr, op->span_)  // NOLINT(misc-include-cleaner)
          << "Only one pl.spmd() block is allowed per cluster scope";
      // A cluster-nested pl.spmd is unwrapped into the Group function and never
      // outlined to a Submit, so a captured producer TaskId (kAttrTaskIdVar), an
      // explicit dependency fence (kAttrManualDepEdges), a speculative
      // early-dispatch hint (allow_early_resolve), OR a dispatch predicate
      // (kAttrPredicate) would be silently dropped. The parser rejects
      // `with pl.spmd(...) as tid:` / `deps=` / `allow_early_resolve=True` /
      // `predicate=` inside pl.cluster() (see
      // ASTParser._parse_spmd_scope_with_tid and
      // ASTParser._reject_spmd_submit_only_kwargs_in_cluster); guard here for
      // hand-built / deserialized IR so the invalid case fails loudly instead of
      // miscompiling.
      INTERNAL_CHECK_SPAN(
          op->GetAttr<VarPtr>(kAttrTaskIdVar) == nullptr && !op->HasAttr(kAttrManualDepEdges) &&
              !op->GetAttr<bool>("allow_early_resolve", false) && !op->HasAttr(kAttrPredicate),
          op->span_)
          << "Internal error: a pl.spmd() nested inside pl.cluster() cannot carry a producer "
             "TASK_ID (kAttrTaskIdVar), dependency edges (kAttrManualDepEdges), an "
             "allow_early_resolve hint, or a dispatch predicate (kAttrPredicate); it is unwrapped "
             "into the Group function and never outlined to a Submit. The parser must reject this "
             "at parse time.";
      core_num = op->core_num_;
      sync_start = op->sync_start_;
      return VisitStmt(op->body_);
    }
  };

  SpmdUnwrapper unwrapper;
  auto new_body = unwrapper.VisitStmt(group_func->body_);
  if (unwrapper.core_num == nullptr) {
    return group_func;
  }

  auto mutable_func = MutableCopy(group_func);
  mutable_func->body_ = new_body;
  // Mark the Group as a launch wrapper: dispatching it launches the kernels its
  // body calls, not the Group itself as a mixed kernel. Self-contained bool —
  // unlike core_num it references nothing outside this function.
  mutable_func->attrs_.emplace_back(kAttrSpmdUnwrapped, true);
  spec_out->core_num = unwrapper.core_num;
  spec_out->sync_start = unwrapper.sync_start.value_or(false);
  return mutable_func;
}

/// Attach each unwrapped Group's launch spec to the dispatch that calls it.
///
/// The spec is lifted from *inside* the Group, so its Vars are the Group's
/// params — the cluster outliner already captured the caller's scalars as
/// params. Translating them back through the dispatch's args (identity mapping,
/// ``params_[i] <-> args_[i]``) is what makes the attr reference Vars that are
/// actually live at the call site; without it the printer would emit a
/// ``__FREE_VAR``-marked name that no scope binds.
///
/// Handles ``Submit`` as well as ``Call`` (a ``with pl.cluster() as tid`` scope
/// outlines to a Submit) — the Submit carries the spec in its first-class
/// fields, per pass-submit-awareness.
class LaunchSpecStamper : public IRMutator {
 public:
  explicit LaunchSpecStamper(const std::unordered_map<std::string, SpmdLaunchSpec>& specs) : specs_(specs) {}

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto visited = IRMutator::VisitExpr_(op);
    auto call = As<Call>(visited);
    auto resolved = Resolve(call, call ? call->args_ : std::vector<ExprPtr>{});
    if (!resolved) return visited;
    auto attrs = call->attrs_;
    attrs.emplace_back(kAttrCoreNum, resolved->first);
    if (resolved->second) attrs.emplace_back(kAttrSyncStart, true);
    return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, std::move(attrs),
                                        call->GetType(), call->span_);
  }

  ExprPtr VisitExpr_(const SubmitPtr& op) override {
    auto visited = IRMutator::VisitExpr_(op);
    auto submit = As<Submit>(visited);
    auto resolved = Resolve(submit, submit ? submit->args_ : std::vector<ExprPtr>{});
    if (!resolved) return visited;
    return std::make_shared<const Submit>(submit->op_, submit->args_, submit->deps_, submit->kwargs_,
                                          submit->attrs_, submit->GetType(), submit->span_,
                                          std::optional<ExprPtr>(resolved->first), resolved->second,
                                          submit->allow_early_resolve_, submit->predicate_);
  }

 private:
  /// Shared Call/Submit prologue: does this dispatch target an unwrapped Group,
  /// and if so what is its launch spec in THIS caller's Var space?
  template <typename NodePtr>
  [[nodiscard]] std::optional<std::pair<ExprPtr, bool>> Resolve(const NodePtr& node,
                                                                const std::vector<ExprPtr>& args) const {
    if (!node || !node->op_) return std::nullopt;
    auto it = specs_.find(node->op_->name_);
    if (it == specs_.end()) return std::nullopt;
    return std::make_pair(ToCallerScope(it->second, args), it->second.sync_start);
  }

  /// Rewrite the callee-scoped ``core_num`` into the caller's Var space via the
  /// dispatch's positional args. ``Submit::args_`` may be a prefix of
  /// ``params_`` (pass-submit-awareness rule 5), so bound by the arg count.
  [[nodiscard]] static ExprPtr ToCallerScope(const SpmdLaunchSpec& spec, const std::vector<ExprPtr>& args) {
    // Every entry in specs_ is built with its Group attached. Falling back to
    // the un-translated expression here would silently re-introduce the very
    // defect this carrier move fixes — a dispatch attr naming a Var bound only
    // in the callee — so treat a missing Group as the compiler bug it is.
    INTERNAL_CHECK(spec.group) << "Internal error: launch spec for an unwrapped Group has no Group "
                                  "attached; core_num cannot be translated into caller scope";
    const auto& params = spec.group->params_;
    std::unordered_map<const Var*, ExprPtr> param_to_arg;
    const size_t n = std::min(params.size(), args.size());
    for (size_t i = 0; i < n; ++i) {
      if (params[i] && args[i]) param_to_arg.emplace(params[i].get(), args[i]);
    }
    // No mapping to apply — a param-less Group, or a constant core_num that
    // references nothing. Substitution would be a no-op either way.
    auto translated =
        param_to_arg.empty() ? spec.core_num : transform_utils::Substitute(spec.core_num, param_to_arg);
    RejectCalleeBoundCoreNum(spec, translated);
    return translated;
  }

  /// The dispatch's args can only recover a count the cluster *captured* from
  /// the caller. A count bound by a statement INSIDE ``pl.cluster()`` has no
  /// corresponding arg, so it would be stamped onto the dispatch still naming a
  /// Var only the callee binds — the closed-scope violation this carrier move
  /// exists to remove, reappearing at the call site. Reject it with an
  /// actionable message instead of emitting an unbound name.
  static void RejectCalleeBoundCoreNum(const SpmdLaunchSpec& spec, const ExprPtr& translated) {
    if (spec.callee_bound.empty() || !translated) return;
    var_collectors::VarDefUseCollector uses;
    uses.VisitExpr(translated);
    for (const auto* used : uses.var_uses) {
      CHECK_SPAN(spec.callee_bound.count(used) == 0, spec.group->span_)
          << "pl.spmd() block count references '" << used->name_hint_
          << "', which is defined inside the enclosing pl.cluster(). The block count is evaluated at "
             "the cluster's launch site, so it must be computed before `with pl.cluster():` and "
             "captured, e.g.\n"
             "    n = pl.system.available_cluster_count()\n"
             "    with pl.cluster():\n"
             "        with pl.spmd(n):";
    }
  }

  const std::unordered_map<std::string, SpmdLaunchSpec>& specs_;
};

}  // namespace

/**
 * @brief Pass to outline Cluster and standalone Spmd scopes into separate functions
 *
 * This pass transforms ScopeStmt(Cluster) and ScopeStmt(Spmd) nodes into separate
 * Function(Group/Spmd) definitions and replaces the scope with a Call to the
 * outlined function.
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Processes both Opaque and Orchestration functions
 *
 * Transformation:
 * 1. Outline ScopeStmt(Cluster) into Function(Group) (first pass)
 * 2. Outline standalone ScopeStmt(Spmd) into Function(Spmd) (second pass)
 * 3. For nested Spmd inside Cluster: unwrap the Spmd scope and propagate
 *    core_num/sync_start onto the Group's dispatch (only the self-contained
 *    spmd_unwrapped marker stays on the Group)
 * 4. Parent function type is preserved (not promoted)
 */
Pass OutlineClusterScopes() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> new_functions;
    std::vector<FunctionPtr> all_outlined_functions;

    // Program-wide set of outlined function names, seeded with the existing
    // function names and shared across every ScopeOutliner (both the Cluster and
    // Spmd passes, across all functions) so duplicate `name_hint` values produced
    // from reused helpers auto-disambiguate instead of colliding at Program
    // construction (#1711).
    auto reserved_func_names = std::make_shared<std::unordered_set<std::string>>();
    for (const auto& [gvar, func] : program->functions_) {
      reserved_func_names->insert(func->name_);
    }

    for (const auto& [gvar, func] : program->functions_) {
      // Only process Opaque and Orchestration functions (Group functions are already outlined)
      if (func->func_type_ != FunctionType::Opaque && func->func_type_ != FunctionType::Orchestration) {
        new_functions.push_back(func);
        continue;
      }

      // First pass: outline Cluster scopes
      outline_utils::VarCollector type_collector;
      for (const auto& var : func->params_) {
        type_collector.var_types[var.get()] = var->GetType();
        type_collector.var_objects[var.get()] = var;
        type_collector.known_names.insert(var->name_hint_);
      }
      type_collector.VisitStmt(func->body_);

      outline_utils::ScopeOutliner cluster_outliner(
          func->name_, type_collector.var_types, type_collector.var_objects, type_collector.known_names,
          ScopeKind::Cluster, FunctionType::Group, "_cluster_", program, reserved_func_names);
      auto body_after_cluster = cluster_outliner.VisitStmt(func->body_);

      // Unwrap a ``pl.spmd`` nested inside each freshly outlined Group and move
      // its launch spec onto the dispatch the outliner just synthesised in THIS
      // body. Done here rather than in a trailing program-wide sweep because
      // that is the only point where the dispatch is still reachable.
      auto cluster_outlined = cluster_outliner.GetOutlinedFunctions();
      std::unordered_map<std::string, SpmdLaunchSpec> group_launch_specs;
      for (auto& outlined : cluster_outlined) {
        if (!outlined || outlined->func_type_ != FunctionType::Group) continue;
        SpmdLaunchSpec spec;
        outlined = UnwrapNestedSpmd(outlined, &spec);
        if (!spec.core_num) continue;
        spec.group = outlined;
        // Snapshot what the Group binds once, so the per-dispatch scope check
        // below is a hash lookup rather than a re-walk of the body.
        for (const auto& param : outlined->params_) {
          if (param) spec.callee_bound.insert(param.get());
        }
        var_collectors::VarDefUseCollector group_defs;
        if (outlined->body_) group_defs.VisitStmt(outlined->body_);
        spec.callee_bound.insert(group_defs.var_defs.begin(), group_defs.var_defs.end());
        group_launch_specs.emplace(outlined->name_, std::move(spec));
      }
      if (!group_launch_specs.empty()) {
        LaunchSpecStamper stamper(group_launch_specs);
        body_after_cluster = stamper.VisitStmt(body_after_cluster);
      }

      all_outlined_functions.insert(all_outlined_functions.end(), cluster_outlined.begin(),
                                    cluster_outlined.end());

      // Second pass: outline standalone Spmd scopes (those not inside a Cluster)
      outline_utils::VarCollector refreshed_collector;
      for (const auto& var : func->params_) {
        refreshed_collector.var_types[var.get()] = var->GetType();
        refreshed_collector.var_objects[var.get()] = var;
        refreshed_collector.known_names.insert(var->name_hint_);
      }
      refreshed_collector.VisitStmt(body_after_cluster);

      // Build a lookup program that includes both the original functions and the
      // newly outlined cluster (Group) functions, so that spmd_outliner can resolve
      // callees created during the cluster pass and infer correct param directions.
      std::vector<FunctionPtr> lookup_functions;
      lookup_functions.reserve(program->functions_.size() + cluster_outlined.size());
      for (const auto& [_, existing] : program->functions_) {
        lookup_functions.push_back(existing);
      }
      lookup_functions.insert(lookup_functions.end(), cluster_outlined.begin(), cluster_outlined.end());
      auto lookup_program = std::make_shared<Program>(lookup_functions, program->name_, program->span_);

      outline_utils::ScopeOutliner spmd_outliner(
          func->name_, refreshed_collector.var_types, refreshed_collector.var_objects,
          refreshed_collector.known_names, ScopeKind::Spmd, FunctionType::Spmd, "_spmd_", lookup_program,
          reserved_func_names);
      auto body_after_spmd = spmd_outliner.VisitStmt(body_after_cluster);

      const auto& spmd_outlined = spmd_outliner.GetOutlinedFunctions();
      all_outlined_functions.insert(all_outlined_functions.end(), spmd_outlined.begin(), spmd_outlined.end());

      auto new_func = MutableCopy(func);
      new_func->body_ = body_after_spmd;
      new_functions.push_back(new_func);
    }

    // Add all outlined functions before the originals
    all_outlined_functions.insert(all_outlined_functions.end(), new_functions.begin(), new_functions.end());

    return std::make_shared<Program>(all_outlined_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "OutlineClusterScopes", kOutlineClusterScopesProperties);
}

}  // namespace pass

// ============================================================================
// ClusterOutlined property verifier
// ============================================================================

namespace {

using ClusterOutlinedVerifier = outline_utils::ScopeKindAbsenceVerifier<ScopeKind::Cluster>;
using SpmdOutlinedVerifier = outline_utils::ScopeKindAbsenceVerifier<ScopeKind::Spmd>;

}  // namespace

class ClusterOutlinedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "ClusterOutlined"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // Group and Spmd functions are expected to contain cluster/spmd content
      if (IsWrapperType(func->func_type_)) continue;
      ClusterOutlinedVerifier cluster_verifier(
          diagnostics, "ClusterOutlined",
          "Cluster ScopeStmt found in non-Group function (should have been outlined)");
      cluster_verifier.VisitStmt(func->body_);
      SpmdOutlinedVerifier spmd_verifier(
          diagnostics, "ClusterOutlined",
          "Spmd ScopeStmt found in non-Group function (should have been outlined)");
      spmd_verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateClusterOutlinedPropertyVerifier() {
  return std::make_shared<ClusterOutlinedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

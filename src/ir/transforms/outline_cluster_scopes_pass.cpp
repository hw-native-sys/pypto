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
#include <optional>
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace pass {

namespace {

/// Unwrap nested Spmd scopes in a Group function body:
/// Extract core_num/sync_start from the Spmd scope and add as function attrs,
/// then replace the ScopeStmt(Spmd) with its body.
///
/// The Spmd scope's `core_num_` is an ExprPtr but must have folded to a
/// positive ConstInt by this pass (guaranteed by the CoreNumResolved property
/// verifier, produced by Simplify and required here via pass_properties).
FunctionPtr UnwrapNestedSpmd(const FunctionPtr& group_func) {
  class SpmdUnwrapper : public IRMutator {
   public:
    std::optional<int> core_num;
    std::optional<bool> sync_start;

   protected:
    StmtPtr VisitStmt_(const SpmdScopeStmtPtr& op) override {
      CHECK(!core_num.has_value())  // NOLINT(misc-include-cleaner)
          << "Only one pl.spmd() block is allowed per cluster scope";
      auto ci = As<ConstInt>(op->core_num_);
      INTERNAL_CHECK(ci != nullptr)
          << "SpmdScopeStmt core_num did not fold to ConstInt before OutlineClusterScopes; "
             "the CoreNumResolved verifier should have caught this";
      core_num = static_cast<int>(ci->value_);
      sync_start = op->sync_start_;
      return VisitStmt(op->body_);
    }
  };

  SpmdUnwrapper unwrapper;
  auto new_body = unwrapper.VisitStmt(group_func->body_);
  if (!unwrapper.core_num.has_value()) {
    return group_func;
  }

  auto mutable_func = MutableCopy(group_func);
  mutable_func->body_ = new_body;
  mutable_func->attrs_.emplace_back("core_num", *unwrapper.core_num);
  if (unwrapper.sync_start.has_value() && *unwrapper.sync_start) {
    mutable_func->attrs_.emplace_back("sync_start", true);
  }
  return mutable_func;
}

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
 *    core_num/sync_start as function attrs on the Group
 * 4. Parent function type is preserved (not promoted)
 */
Pass OutlineClusterScopes() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> new_functions;
    std::vector<FunctionPtr> all_outlined_functions;

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
          ScopeKind::Cluster, FunctionType::Group, "_cluster_", program);
      auto body_after_cluster = cluster_outliner.VisitStmt(func->body_);

      const auto& cluster_outlined = cluster_outliner.GetOutlinedFunctions();
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
          refreshed_collector.known_names, ScopeKind::Spmd, FunctionType::Spmd, "_spmd_", lookup_program);
      auto body_after_spmd = spmd_outliner.VisitStmt(body_after_cluster);

      const auto& spmd_outlined = spmd_outliner.GetOutlinedFunctions();
      all_outlined_functions.insert(all_outlined_functions.end(), spmd_outlined.begin(), spmd_outlined.end());

      auto new_func = MutableCopy(func);
      new_func->body_ = body_after_spmd;
      new_functions.push_back(new_func);
    }

    // Unwrap nested Spmd scopes in Group functions (Spmd inside Cluster case)
    for (auto& func : all_outlined_functions) {
      if (func->func_type_ == FunctionType::Group) {
        func = UnwrapNestedSpmd(func);
      }
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
      if (func->func_type_ == FunctionType::Group || func->func_type_ == FunctionType::Spmd) continue;
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

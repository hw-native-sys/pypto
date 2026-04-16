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
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace pass {

/**
 * @brief Pass to outline non-CORE_GROUP Hierarchy scopes into separate functions.
 *
 * This pass transforms HierarchyScopeStmt nodes whose `level_` is anything other
 * than `Level::CORE_GROUP` into separate Function definitions that carry the
 * scope's Level/Role metadata, and replaces the scope with a Call to the outlined
 * function. CORE_GROUP scopes are intentionally left intact for the subsequent
 * `OutlineIncoreScopes` pass, which emits `Function(InCore)` and promotes the
 * parent function from `Opaque` to `Orchestration`.
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Only processes Opaque functions
 * - Should run before OutlineIncoreScopes and OutlineClusterScopes
 *
 * Transformation:
 * 1. For each HierarchyScopeStmt at level != CORE_GROUP in an Opaque function:
 *    - Analyze body for inputs/outputs
 *    - Extract body into a new Opaque Function carrying the scope's level/role
 *    - Replace the scope with a Call to the outlined function + output assignments
 * 2. Recursively descends into other scopes; nested non-CORE_GROUP Hierarchy
 *    scopes are outlined together with their parent.
 * 3. CORE_GROUP scopes (and their bodies) are preserved verbatim.
 */
Pass OutlineHierarchyScopes() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> new_functions;
    std::vector<FunctionPtr> all_outlined_functions;

    for (const auto& [gvar, func] : program->functions_) {
      // Only process Opaque functions (hierarchy scopes appear in user-written programs)
      if (func->func_type_ != FunctionType::Opaque) {
        new_functions.push_back(func);
        continue;
      }

      // Build symbol table for this function
      outline_utils::VarCollector type_collector;
      for (const auto& var : func->params_) {
        type_collector.var_types[var.get()] = var->GetType();
        type_collector.var_objects[var.get()] = var;
        type_collector.known_names.insert(var->name_hint_);
      }
      type_collector.VisitStmt(func->body_);

      // Outline non-CORE_GROUP Hierarchy scopes; CORE_GROUP scopes are skipped
      // and handled by OutlineIncoreScopes downstream.
      outline_utils::ScopeOutliner::HierarchyLevelFilter filter{
          Level::CORE_GROUP, outline_utils::ScopeOutliner::HierarchyLevelFilter::Mode::Exclude};
      outline_utils::ScopeOutliner outliner(func->name_, type_collector.var_types, type_collector.var_objects,
                                            type_collector.known_names, ScopeKind::Hierarchy,
                                            /*outlined_func_type=*/FunctionType::Opaque, "_hierarchy_",
                                            /*program=*/nullptr, filter);
      auto new_body = outliner.VisitStmt(func->body_);

      auto new_func = MutableCopy(func);
      new_func->body_ = new_body;
      // Parent type unchanged; CORE_GROUP-driven promotion to Orchestration
      // happens in OutlineIncoreScopes.
      new_functions.push_back(new_func);

      const auto& outlined = outliner.GetOutlinedFunctions();
      all_outlined_functions.insert(all_outlined_functions.end(), outlined.begin(), outlined.end());
    }

    // Add all outlined functions before the originals
    all_outlined_functions.insert(all_outlined_functions.end(), new_functions.begin(), new_functions.end());

    // Create new program with all functions
    return std::make_shared<Program>(all_outlined_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "OutlineHierarchyScopes", kOutlineHierarchyScopesProperties);
}

}  // namespace pass

// ============================================================================
// HierarchyOutlined property verifier
// ============================================================================
//
// This verifier is shared between OutlineHierarchyScopes and OutlineIncoreScopes.
// The HierarchyOutlined property is produced by OutlineIncoreScopes (which runs
// after OutlineHierarchyScopes), since CORE_GROUP scopes survive the first pass.

namespace {

using HierarchyOutlinedVerifier = outline_utils::ScopeKindAbsenceVerifier<ScopeKind::Hierarchy>;

}  // namespace

class HierarchyOutlinedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "HierarchyOutlined"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // After both outline passes have run, no Hierarchy scopes should remain in
      // Opaque/Orchestration functions. Inside InCore/Group/Spmd outlined
      // functions, Hierarchy scopes are disallowed by construction (the outliner
      // only produces leaf scope bodies).
      if (func->func_type_ != FunctionType::Opaque && func->func_type_ != FunctionType::Orchestration) {
        continue;
      }
      HierarchyOutlinedVerifier verifier(diagnostics, "HierarchyOutlined",
                                         "Hierarchy ScopeStmt found in function (should have been outlined)");
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateHierarchyOutlinedPropertyVerifier() {
  return std::make_shared<HierarchyOutlinedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

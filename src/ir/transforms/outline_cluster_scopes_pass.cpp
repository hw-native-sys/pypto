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
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace pass {

/**
 * @brief Pass to outline Cluster scopes into separate Group functions
 *
 * This pass transforms ScopeStmt(Cluster) nodes into separate Function(Group) definitions
 * and replaces the scope with a Call to the outlined function.
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Processes both Opaque and Orchestration functions
 *
 * Transformation:
 * 1. For each ScopeStmt(Cluster) in an Opaque or Orchestration function:
 *    - Analyze body to determine external variable references (inputs)
 *    - Analyze subsequent statements to determine which definitions are outputs
 *    - Extract body into new Function(Group) with appropriate params/returns
 *    - Replace scope with Call to the outlined function + output assignments
 * 2. Recursively handles nested Cluster scopes
 * 3. Add outlined functions to the program
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

      outline_utils::VarCollector type_collector;
      for (const auto& var : func->params_) {
        type_collector.var_types[var.get()] = var->GetType();
        type_collector.var_objects[var.get()] = var;
        type_collector.known_names.insert(var->name_hint_);
      }
      type_collector.VisitStmt(func->body_);

      outline_utils::ScopeOutliner outliner(func->name_, type_collector.var_types, type_collector.var_objects,
                                            type_collector.known_names, ScopeKind::Cluster,
                                            FunctionType::Group, "_cluster_");
      auto new_body = outliner.VisitStmt(func->body_);

      auto new_func =
          std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                     new_body, func->span_, func->func_type_, func->level_, func->role_);
      new_functions.push_back(new_func);

      const auto& outlined = outliner.GetOutlinedFunctions();
      all_outlined_functions.insert(all_outlined_functions.end(), outlined.begin(), outlined.end());
    }

    // Add all outlined functions before the originals
    all_outlined_functions.insert(all_outlined_functions.end(), new_functions.begin(), new_functions.end());

    // Create new program with all functions
    return std::make_shared<Program>(all_outlined_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "OutlineClusterScopes", kOutlineClusterScopesProperties);
}

}  // namespace pass

// ============================================================================
// ClusterOutlined property verifier
// ============================================================================

namespace {

/**
 * @brief Checks no Cluster ScopeStmts remain in non-Group functions.
 */
class ClusterOutlinedVerifier : public IRVisitor {
 public:
  explicit ClusterOutlinedVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const ScopeStmtPtr& op) override {
    if (!op) return;
    if (op->scope_kind_ == ScopeKind::Cluster) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "ClusterOutlined", 0,
                                "Cluster ScopeStmt found in non-Group function (should have been outlined)",
                                op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class ClusterOutlinedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "ClusterOutlined"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // Group functions are expected to contain cluster content
      if (func->func_type_ == FunctionType::Group) continue;
      ClusterOutlinedVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateClusterOutlinedPropertyVerifier() {
  return std::make_shared<ClusterOutlinedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

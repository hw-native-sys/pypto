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
 * @brief Pass to outline Hierarchy scopes into separate functions with level/role
 *
 * This pass transforms ScopeStmt(Hierarchy) nodes into separate Function definitions
 * that carry the scope's Level and Role metadata, and replaces the scope with a Call
 * to the outlined function.
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Only processes Opaque functions
 * - Should run before OutlineIncoreScopes and OutlineClusterScopes
 *
 * Transformation:
 * 1. For each ScopeStmt(Hierarchy) in an Opaque function:
 *    - Analyze body to determine external variable references (inputs)
 *    - Analyze subsequent statements to determine which definitions are outputs
 *    - Extract body into new Function(Opaque, level, role) with appropriate params/returns
 *    - Replace scope with Call to the outlined function + output assignments
 * 2. Recursively handles nested Hierarchy scopes
 * 3. Add outlined functions to the program
 * 4. Parent function type is preserved (not promoted)
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

      // Outline Hierarchy scopes in this function
      outline_utils::ScopeOutliner outliner(func->name_, type_collector.var_types, type_collector.var_objects,
                                            type_collector.known_names, ScopeKind::Hierarchy,
                                            FunctionType::Opaque, "_hierarchy_");
      auto new_body = outliner.VisitStmt(func->body_);

      // Preserve parent function type (don't promote — hierarchy is orthogonal to FunctionType)
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

  return CreateProgramPass(pass_func, "OutlineHierarchyScopes", kOutlineHierarchyScopesProperties);
}

}  // namespace pass

// ============================================================================
// HierarchyOutlined property verifier
// ============================================================================

namespace {

/**
 * @brief Checks no Hierarchy ScopeStmts remain in any function.
 */
class HierarchyOutlinedVerifier : public IRVisitor {
 public:
  explicit HierarchyOutlinedVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const ScopeStmtPtr& op) override {
    if (!op) return;
    if (op->scope_kind_ == ScopeKind::Hierarchy) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "HierarchyOutlined", 0,
                                "Hierarchy ScopeStmt found in function (should have been outlined)",
                                op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class HierarchyOutlinedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "HierarchyOutlined"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // Only check Opaque functions — the pass only processes Opaque functions,
      // so Hierarchy scopes in other function types are not expected to be outlined.
      if (func->func_type_ != FunctionType::Opaque) continue;
      HierarchyOutlinedVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateHierarchyOutlinedPropertyVerifier() {
  return std::make_shared<HierarchyOutlinedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

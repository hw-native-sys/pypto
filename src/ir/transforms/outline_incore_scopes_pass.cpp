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
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
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
 * @brief Pass to outline InCore scopes into separate functions
 *
 * This pass transforms ScopeStmt(InCore) nodes into separate Function(InCore) definitions
 * and replaces the scope with a Call to the outlined function.
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Only processes Opaque functions (InCore functions are left unchanged)
 *
 * Transformation:
 * 1. For each ScopeStmt(InCore) in an Opaque function:
 *    - Analyze body to determine external variable references (inputs)
 *    - Analyze subsequent statements to determine which definitions are outputs
 *    - Extract body into new Function(InCore) with appropriate params/returns
 *    - Replace scope with Call to the outlined function + output assignments
 *    - EvalStmt(store) calls on output tensors are converted to AssignStmt
 * 2. Recursively handles nested InCore scopes
 * 3. Add outlined functions to the program
 * 4. Promote the parent function from Opaque to Orchestration
 */
Pass OutlineIncoreScopes() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> new_functions;
    std::vector<FunctionPtr> all_outlined_functions;

    for (const auto& [gvar, func] : program->functions_) {
      // Only process Opaque functions (InCore functions are already outlined)
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

      // Outline InCore scopes in this function
      outline_utils::ScopeOutliner outliner(func->name_, type_collector.var_types, type_collector.var_objects,
                                            type_collector.known_names, ScopeKind::InCore,
                                            FunctionType::InCore, "_incore_");
      auto new_body = outliner.VisitStmt(func->body_);

      // Create new function with transformed body.
      // If any InCore scopes were outlined, promote Opaque -> Orchestration.
      const auto& outlined = outliner.GetOutlinedFunctions();
      FunctionType new_func_type = outlined.empty() ? func->func_type_ : FunctionType::Orchestration;
      auto new_func =
          std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                     new_body, func->span_, new_func_type, func->level_, func->role_);
      new_functions.push_back(new_func);

      // Collect outlined functions (prepend before parent so inner functions come first)
      all_outlined_functions.insert(all_outlined_functions.end(), outlined.begin(), outlined.end());
    }

    // Add all outlined functions before the originals
    all_outlined_functions.insert(all_outlined_functions.end(), new_functions.begin(), new_functions.end());

    // Create new program with all functions
    return std::make_shared<Program>(all_outlined_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "OutlineIncoreScopes", kOutlineIncoreScopesProperties);
}

}  // namespace pass

// ============================================================================
// SplitIncoreOrch property verifier
// ============================================================================

namespace {

/**
 * @brief Checks no InCore ScopeStmts remain in Opaque or Orchestration functions.
 */
class SplitIncoreOrchVerifier : public IRVisitor {
 public:
  explicit SplitIncoreOrchVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const ScopeStmtPtr& op) override {
    if (!op) return;
    if (op->scope_kind_ == ScopeKind::InCore) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "SplitIncoreOrch", 0,
                                "InCore ScopeStmt found in non-InCore function (should have been outlined)",
                                op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

/// Returns true if op_name is a compute tensor op (not host-side memory/transfer/metadata).
/// Host-side ops are memory allocation/transfer (create, read, write, slice, assemble, dim)
/// and metadata-only transforms (reshape, transpose at tensor level — actual data ops use tile.*).
static bool IsComputeTensorOp(const std::string& op_name) {
  if (op_name.compare(0, 7, "tensor.") != 0) return false;
  static const std::unordered_set<std::string> kHostSideOps = {
      "tensor.create",   "tensor.read", "tensor.write",   "tensor.slice",
      "tensor.assemble", "tensor.dim",  "tensor.reshape", "tensor.transpose",
  };
  return kHostSideOps.count(op_name) == 0;
}

/// Checks Orchestration functions for compute tensor ops that should be in InCore.
class OrchComputeTensorOpVerifier : public IRVisitor {
 public:
  explicit OrchComputeTensorOpVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitExpr_(const CallPtr& op) override {
    if (op && op->op_ && IsComputeTensorOp(op->op_->name_)) {
      diagnostics_.emplace_back(DiagnosticSeverity::Warning, "SplitIncoreOrch", 0,
                                "Compute tensor op '" + op->op_->name_ +
                                    "' found in Orchestration function (should be inside InCore)",
                                op->span_);
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class SplitIncoreOrchPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "SplitIncoreOrch"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // Check Opaque and Orchestration functions — InCore functions are expected to have InCore content
      if (func->func_type_ == FunctionType::InCore) continue;
      SplitIncoreOrchVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
      // Also check Orchestration functions for leaked compute tensor ops
      if (func->func_type_ == FunctionType::Orchestration) {
        OrchComputeTensorOpVerifier compute_verifier(diagnostics);
        compute_verifier.VisitStmt(func->body_);
      }
    }
  }
};

PropertyVerifierPtr CreateSplitIncoreOrchPropertyVerifier() {
  return std::make_shared<SplitIncoreOrchPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

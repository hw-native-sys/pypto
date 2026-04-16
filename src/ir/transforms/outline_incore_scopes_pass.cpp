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
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"

namespace pypto {
namespace ir {

namespace pass {

/**
 * @brief Pass to outline CORE_GROUP Hierarchy scopes into InCore functions.
 *
 * This pass picks up where OutlineHierarchyScopes leaves off: it transforms
 * every `HierarchyScopeStmt(level=CORE_GROUP)` that survived the previous pass
 * into a separate `Function(InCore)` definition and replaces the scope with a
 * `Call` to that function. When any CORE_GROUP scope is outlined out of an
 * `Opaque` function, the parent function is promoted from `Opaque` to
 * `Orchestration` so downstream tile-level passes see the canonical
 * Orchestration → InCore call shape.
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Should run after OutlineHierarchyScopes and before OutlineClusterScopes
 * - Only processes Opaque functions
 *
 * Together with OutlineHierarchyScopes this pass establishes the
 * `HierarchyOutlined` property: after both have run, no `HierarchyScopeStmt`
 * remains in any Opaque/Orchestration function body.
 */
namespace {

/// Returns true iff any HierarchyScopeStmt at Level::CORE_GROUP appears under
/// the given statement. Used to decide whether to promote the parent function
/// from Opaque to Orchestration.
class CoreGroupHierarchyFinder : public IRVisitor {
 public:
  bool found = false;

 protected:
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override {
    if (op->level_ == Level::CORE_GROUP) {
      found = true;
    }
    IRVisitor::VisitStmt_(op);
  }
};

}  // namespace

Pass OutlineIncoreScopes() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> new_functions;
    std::vector<FunctionPtr> all_outlined_functions;

    for (const auto& [gvar, func] : program->functions_) {
      // Only Opaque functions can carry CORE_GROUP HierarchyScopeStmts at this
      // point in the pipeline.
      if (func->func_type_ != FunctionType::Opaque) {
        new_functions.push_back(func);
        continue;
      }

      // Detect CORE_GROUP scopes before outlining; outliner.GetOutlinedFunctions()
      // tells us *what* was outlined, but we need the parent-promotion decision
      // up front so it is symmetric with future filters.
      CoreGroupHierarchyFinder finder;
      finder.VisitStmt(func->body_);

      // Build symbol table for this function
      outline_utils::VarCollector type_collector;
      for (const auto& var : func->params_) {
        type_collector.var_types[var.get()] = var->GetType();
        type_collector.var_objects[var.get()] = var;
        type_collector.known_names.insert(var->name_hint_);
      }
      type_collector.VisitStmt(func->body_);

      // Outline only HierarchyScopeStmts at CORE_GROUP into InCore functions.
      outline_utils::ScopeOutliner::HierarchyLevelFilter filter{
          Level::CORE_GROUP, outline_utils::ScopeOutliner::HierarchyLevelFilter::Mode::Only};
      outline_utils::ScopeOutliner outliner(func->name_, type_collector.var_types, type_collector.var_objects,
                                            type_collector.known_names, ScopeKind::Hierarchy,
                                            /*outlined_func_type=*/FunctionType::InCore, "_incore_",
                                            /*program=*/nullptr, filter);
      auto new_body = outliner.VisitStmt(func->body_);

      auto new_func = MutableCopy(func);
      new_func->body_ = new_body;
      if (finder.found) {
        // Promote parent Opaque → Orchestration whenever any CORE_GROUP scope
        // was outlined, matching the contract the former OutlineIncoreScopes
        // (driven by InCoreScopeStmt) used to satisfy.
        new_func->func_type_ = FunctionType::Orchestration;
      }
      new_functions.push_back(new_func);

      const auto& outlined = outliner.GetOutlinedFunctions();
      all_outlined_functions.insert(all_outlined_functions.end(), outlined.begin(), outlined.end());
    }

    // Outlined functions go before the originals so call sites can reference them.
    all_outlined_functions.insert(all_outlined_functions.end(), new_functions.begin(), new_functions.end());
    return std::make_shared<Program>(all_outlined_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "OutlineIncoreScopes", kOutlineIncoreScopesProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

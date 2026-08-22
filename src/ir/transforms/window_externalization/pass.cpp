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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/utils/window_externalization.h"
#include "src/ir/transforms/window_externalization/internal.h"

namespace pypto {
namespace ir {
namespace window_externalization {
namespace {

ProgramPtr Run(const ProgramPtr& program) {
  WindowRewriteContext rewrite_context;
  auto analyses = AnalyzeProgram(program);
  if (analyses.empty()) return program;

  auto function_lookup = BuildFunctionLookup(program);

  std::unordered_map<std::string, FunctionPtr> cloned_funcs;
  for (const auto& [func_name, analysis] : analyses) {
    auto callee_it = function_lookup.find(func_name);
    if (callee_it == function_lookup.end()) continue;
    auto callee = callee_it->second;
    auto cloned = RewriteCallee(program, callee, analysis, "__windowed", rewrite_context);
    if (!cloned) {
      continue;
    }
    cloned_funcs.emplace(func_name, cloned);
    cloned_funcs.emplace(cloned->name_, cloned);
  }
  if (cloned_funcs.empty()) return program;

  // RewriteCallee() also records dynamic extent parameters for rewritten
  // outputs. Keep all callee rewrites complete before rewriting orchestration
  // calls so each callsite sees the full ABI side table.
  std::unordered_map<const Function*, FunctionPtr> rewritten_orch_funcs;
  std::unordered_set<std::string> used_clone_names;
  bool changed = false;
  for (const auto& [_, func] : program->functions_) {
    // The clone loop above is not type-gated, so restricting the call-site
    // rewrite to plain Orchestration would leave a Graph body calling the
    // original signature while the clone carries the windowed ABI.
    if (!IsOrchestrationLike(func)) continue;
    auto rewritten = RewriteOrchestrationBody(program, analyses, cloned_funcs, function_lookup,
                                              rewrite_context, func->body_);
    if (rewritten.body.get() == func->body_.get()) continue;
    changed = true;
    for (const auto& clone_name : rewritten.used_clone_names) used_clone_names.insert(clone_name);
    rewritten_orch_funcs.emplace(
        func.get(), std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                               func->return_types_, rewritten.body, func->span_,
                                               func->func_type_, func->level_, func->role_, func->attrs_));
  }

  if (!changed) return program;
  std::vector<FunctionPtr> new_functions;
  new_functions.reserve(program->functions_.size() + used_clone_names.size());
  for (const auto& [_, func] : program->functions_) {
    auto rewritten_it = rewritten_orch_funcs.find(func.get());
    new_functions.push_back(rewritten_it == rewritten_orch_funcs.end() ? func : rewritten_it->second);
    auto clone_it = cloned_funcs.find(func->name_);
    if (clone_it != cloned_funcs.end() && used_clone_names.count(func->name_) != 0) {
      new_functions.push_back(clone_it->second);
    }
  }
  return std::make_shared<Program>(new_functions, program->name_, program->span_);
}

}  // namespace

bool HasWindowizeEnabledFunction(const ProgramPtr& program) {
  if (!program) return false;
  for (const auto& [_, func] : program->functions_) {
    if (func && IsInCoreType(func->func_type_) && IsWindowizeEnabled(func)) {
      return true;
    }
  }
  return false;
}

ProgramPtr ApplyWindowExternalization(const ProgramPtr& program) {
  if (!HasWindowizeEnabledFunction(program)) return program;
  return Run(program);
}

}  // namespace window_externalization
}  // namespace ir
}  // namespace pypto

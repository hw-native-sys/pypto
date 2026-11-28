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
#include <optional>
#include <string>
#include <utility>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/auto_layout/extractor.h"
#include "pypto/ir/transforms/auto_layout/model.h"
#include "pypto/ir/transforms/auto_layout/solver.h"
#include "pypto/ir/transforms/auto_layout/writer.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

// HOST tensor types describe transport geometry. Publish the device ABI as a
// separate, versioned slot map only after the solver's writeback succeeds.
ProgramPtr PublishHostInputLayouts(const ProgramPtr& program, const std::string& entry,
                                   const std::string& layouts) {
  if (entry.empty()) return program;
  const auto key = program->GetGlobalVar(entry);
  INTERNAL_CHECK(key) << "Missing HOST entry during AutoLayout writeback";
  auto function = MutableCopy(program->functions_.at(key));
  auto& attrs = function->attrs_;
  attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                             [](const auto& attr) { return attr.first == kAttrAutoLayoutHostInputs; }),
              attrs.end());
  attrs.emplace_back(kAttrAutoLayoutHostInputs, layouts);
  auto updated = MutableCopy(program);
  updated->functions_.at(key) = std::move(function);
  return updated;
}

std::string HostInputLayouts(const auto_layout::LayoutWritebackBindings& bindings,
                             const auto_layout::LayoutProblemInstance& instance,
                             const auto_layout::LayoutAssignment* assignment) {
  using auto_layout::LayoutSpec;
  std::string descriptor = "v1";
  for (const auto& [slot, name] : bindings.host_input_roots) {
    std::string layout = "UNSUPPORTED";
    if (const auto* value = instance.FindValue(name); assignment && value) {
      const auto& choices = value->admissible_layouts;
      const auto contains = [&](const LayoutSpec& candidate) {
        return std::find(choices.begin(), choices.end(), candidate) != choices.end();
      };
      // Neither a pinned baseline nor the HOST annotation proves that the
      // loader can deliver the only representation the device expects.
      const auto selected = assignment->value_layouts.find(name);
      if (contains(LayoutSpec::ND()) && contains(LayoutSpec::NZ()) &&
          selected != assignment->value_layouts.end()) {
        if (selected->second == LayoutSpec::ND()) layout = "ND";
        if (selected->second == LayoutSpec::NZ()) layout = "NZ";
      }
    }
    descriptor += ";" + std::to_string(slot) + "=" + layout;
  }
  return descriptor;
}

}  // namespace

Pass AutoLayout(AutoLayoutOptions options) {
  options.cost_model.ValidateUseHorizon();
  if (options.solver != "auto" && options.solver != "greedy" && options.solver != "treewidth_dp") {
    throw ValueError("Unknown AutoLayout solver '" + options.solver +
                     "'; expected 'auto', 'greedy', or 'treewidth_dp'");
  }
  const std::string pass_name = options.name.empty() ? std::string("AutoLayout") : options.name;
  return CreateProgramPass(
      [opts = std::move(options)](const ProgramPtr& program) -> ProgramPtr {
        auto_layout::LayoutWritebackBindings bindings;
        auto_layout::LayoutProblemInstance instance = auto_layout::ExtractLayoutInstance(
            program, opts.cost_model, opts.reusable_tensors, opts.nz_inputs, &bindings);
        instance.custom_conversion_costs = opts.custom_conversion_costs;
        auto_layout::LayoutSolverConfig solver_cfg;
        solver_cfg.max_treewidth = opts.max_treewidth;
        solver_cfg.collect_treewidth_stats = false;

        auto_layout::LayoutSolverResult result;
        if (opts.solver == "greedy") {
          result = auto_layout::SolveGreedy(instance, std::nullopt, solver_cfg);
        } else if (opts.solver == "treewidth_dp") {
          result = auto_layout::SolveTreewidthDP(instance, solver_cfg);
        } else {
          result = auto_layout::SolveAutoLayout(instance, std::nullopt, solver_cfg);
        }
        if (!opts.fail_on_writeback_issues && !result.writeback_issues.empty()) {
          return PublishHostInputLayouts(program, bindings.host_input_entry,
                                         HostInputLayouts(bindings, instance, nullptr));
        }
        const auto& assignment = result.WritebackAssignment();
        const auto entry = bindings.host_input_entry;
        const auto layouts = HostInputLayouts(bindings, instance, &assignment);
        auto rewritten = auto_layout::detail::ApplySolvedLayout(instance, std::move(bindings), assignment);
        return PublishHostInputLayouts(rewritten, entry, layouts);
      },
      pass_name, kAutoLayoutProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

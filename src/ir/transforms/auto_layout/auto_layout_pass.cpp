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

#include <optional>
#include <string>
#include <utility>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/auto_layout/extractor.h"
#include "pypto/ir/transforms/auto_layout/model.h"
#include "pypto/ir/transforms/auto_layout/solver.h"
#include "pypto/ir/transforms/auto_layout/writer.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"

namespace pypto {
namespace ir {
namespace pass {

Pass AutoLayout(AutoLayoutOptions options) {
  if (options.solver != "auto" && options.solver != "greedy" && options.solver != "treewidth_dp") {
    throw ValueError("Unknown AutoLayout solver '" + options.solver +
                     "'; expected 'auto', 'greedy', or 'treewidth_dp'");
  }
  const std::string pass_name = options.name.empty() ? std::string("AutoLayout") : options.name;
  return CreateProgramPass(
      [opts = std::move(options)](const ProgramPtr& program) -> ProgramPtr {
        auto_layout::LayoutProblemInstance instance =
            auto_layout::ExtractLayoutInstance(program, opts.cost_model, opts.reusable_tensors);
        instance.custom_conversion_costs = opts.custom_conversion_costs;
        auto_layout::LayoutSolverConfig solver_cfg;
        solver_cfg.max_treewidth = opts.max_treewidth;

        auto_layout::LayoutSolverResult result;
        if (opts.solver == "greedy") {
          result = auto_layout::SolveGreedy(instance, std::nullopt, solver_cfg);
        } else if (opts.solver == "treewidth_dp") {
          result = auto_layout::SolveTreewidthDP(instance, solver_cfg);
        } else {
          result = auto_layout::SolveAutoLayout(instance, std::nullopt, solver_cfg);
        }
        if (!opts.fail_on_writeback_issues && !result.writeback_issues.empty()) {
          return program;
        }
        return auto_layout::ApplyLayoutAssignment(program, result.WritebackAssignment());
      },
      pass_name, kAutoLayoutProperties);
}

Pass AutoLayoutFromAssignment(const auto_layout::LayoutAssignment& assignment, const std::string& name) {
  const std::string pass_name = name.empty() ? std::string("AutoLayout") : name;
  return CreateProgramPass(
      [assignment](const ProgramPtr& program) -> ProgramPtr {
        return auto_layout::ApplyLayoutAssignment(program, assignment);
      },
      pass_name, kAutoLayoutProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

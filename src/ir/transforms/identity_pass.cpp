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

#include "./pass_impl.h"
#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/passes.h"

namespace pypto {
namespace ir {

namespace {

/**
 * @brief Identity pass implementation that appends a suffix to function names
 *
 * This pass appends "_identity" to each function name for testing purposes.
 * This allows tests to verify that the pass was actually executed.
 */
class Identity : public PassImpl {
 public:
  Identity() = default;
  ~Identity() override = default;

  ProgramPtr operator()(const ProgramPtr& program) override {
    INTERNAL_CHECK(program) << "Identity pass cannot run on null program";

    // Apply transformation to each function in the program
    std::vector<FunctionPtr> transformed_functions;
    transformed_functions.reserve(program->functions_.size());

    for (const auto& [global_var, func] : program->functions_) {
      // Append "_identity" suffix to the function name
      std::string new_name = func->name_ + "_identity";

      // Create a new function with the modified name
      auto transformed_func = std::make_shared<const Function>(new_name, func->params_, func->return_types_,
                                                               func->body_, func->span_);

      transformed_functions.push_back(transformed_func);
    }

    // Create a new program with the transformed functions
    return std::make_shared<const Program>(transformed_functions, program->name_, program->span_);
  }

  [[nodiscard]] std::string GetName() const override { return "Identity"; }
};

}  // namespace

namespace pass {
// Factory function
Pass Identity() { return Pass(std::make_shared<pypto::ir::Identity>()); }
}  // namespace pass
}  // namespace ir
}  // namespace pypto

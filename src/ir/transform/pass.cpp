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

#include "pypto/ir/transform/base/pass.h"

#include <memory>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace ir {

ProgramPtr Pass::Run(const ProgramPtr& program) {
  INTERNAL_CHECK(program) << "Pass cannot run on null program";

  // Apply the pass to each function in the program independently
  std::vector<FunctionPtr> transformed_functions;
  transformed_functions.reserve(program->functions_.size());

  for (const auto& [global_var, func] : program->functions_) {
    FunctionPtr transformed_func = Run(func);
    transformed_functions.push_back(transformed_func);
  }

  // Create a new program with the transformed functions
  // The Program constructor will create new GlobalVars and maintain the sorted map
  return std::make_shared<const Program>(transformed_functions, program->name_, program->span_);
}

}  // namespace ir
}  // namespace pypto

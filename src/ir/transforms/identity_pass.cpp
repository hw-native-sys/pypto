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

#include "pypto/ir/function.h"
#include "pypto/ir/transforms/passes.h"

namespace pypto {
namespace ir {
namespace pass {

/**
 * @brief Create an identity pass for testing
 *
 * This pass appends "_identity" to each function name for testing purposes.
 * This allows tests to verify that the pass was actually executed.
 */
Pass Identity() {
  return CreateFunctionPass(
      [](const FunctionPtr& func) {
        // Append "_identity" suffix to the function name
        std::string new_name = func->name_ + "_identity";

        // Create a new function with the modified name
        return std::make_shared<const Function>(new_name, func->params_, func->return_types_, func->body_,
                                                func->span_);
      },
      "Identity");
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

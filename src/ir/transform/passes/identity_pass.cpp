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

#include "pypto/ir/transform/passes/identity_pass.h"

#include "pypto/core/logging.h"

namespace pypto {
namespace ir {

FunctionPtr IdentityPass::Run(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "IdentityPass cannot run on null function";
  
  // Append "_identity" suffix to the function name to mark that this pass was applied
  std::string new_name = func->name_ + "_identity";
  
  // Create a new function with the modified name
  return std::make_shared<const Function>(
      new_name, func->params_, func->return_types_, func->body_, func->span_);
}

}  // namespace ir
}  // namespace pypto

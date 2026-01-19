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

#ifndef PYPTO_IR_TRANSFORM_PASSES_IDENTITY_PASS_H_
#define PYPTO_IR_TRANSFORM_PASSES_IDENTITY_PASS_H_

#include <string>

#include "pypto/ir/function.h"
#include "pypto/ir/transform/base/pass.h"

namespace pypto {
namespace ir {

/**
 * @brief Identity pass that appends a suffix to function name
 *
 * This pass appends "_identity" to the function name for testing purposes.
 * This allows tests to verify that the pass was actually executed.
 */
class IdentityPass : public Pass {
 public:
  /**
   * @brief Execute the identity pass
   *
   * @param func Input function
   * @return New function with modified name
   */
  FunctionPtr Run(const FunctionPtr& func) override;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_PASSES_IDENTITY_PASS_H_

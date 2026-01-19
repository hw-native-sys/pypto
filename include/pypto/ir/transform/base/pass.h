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

#ifndef PYPTO_IR_TRANSFORM_BASE_PASS_H_
#define PYPTO_IR_TRANSFORM_BASE_PASS_H_

#include <memory>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/transform/base/mutator.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for IR transformation passes
 *
 * Pass is an abstract base class that extends IRMutator to provide function-level transformations.
 * Each pass operates on a Function and returns a transformed Function.
 * Passes maintain immutability - they return new FunctionPtr instances rather than modifying in place.
 */
class Pass : public IRMutator {
 public:
  ~Pass() override = default;

  /**
   * @brief Execute the pass on a function
   *
   * This is the main entry point for pass execution. Subclasses must implement this method
   * to define their transformation logic.
   *
   * @param func Input function to transform
   * @return Transformed function (may be the same pointer if no changes were made)
   */
  virtual FunctionPtr Run(const FunctionPtr& func) = 0;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_BASE_PASS_H_

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
#include "pypto/ir/program.h"
#include "pypto/ir/transform/base/mutator.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for IR transformation passes
 *
 * Pass is an abstract base class that extends IRMutator to provide transformations
 * on both Function and Program levels. Each pass can operate on individual Functions
 * or entire Programs, returning transformed IR nodes.
 * Passes maintain immutability - they return new IR instances rather than modifying in place.
 */
class Pass : public IRMutator {
 public:
  ~Pass() override = default;

  /**
   * @brief Execute the pass on a function
   *
   * This is the main entry point for function-level pass execution.
   * Subclasses must implement this method to define their transformation logic.
   *
   * @param func Input function to transform
   * @return Transformed function (may be the same pointer if no changes were made)
   */
  virtual FunctionPtr Run(const FunctionPtr& func) = 0;

  /**
   * @brief Execute the pass on a program
   *
   * This method provides program-level transformation capability.
   * The default implementation applies the pass to each function in the program
   * independently. Subclasses can override this method to implement program-wide
   * transformations (e.g., inter-procedural optimizations).
   *
   * @param program Input program to transform
   * @return Transformed program with all functions processed
   */
  virtual ProgramPtr Run(const ProgramPtr& program);
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_BASE_PASS_H_

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

#ifndef SRC_IR_TRANSFORMS_PASS_IMPL_H_
#define SRC_IR_TRANSFORMS_PASS_IMPL_H_

#include <string>

#include "pypto/ir/program.h"

namespace pypto {
namespace ir {

/**
 * @brief Internal base class for pass implementations
 *
 * This is an internal class used for implementing passes via pimpl pattern.
 * Concrete passes should inherit from this class in their source files.
 */
class PassImpl {
 public:
  virtual ~PassImpl() = default;

  /**
   * @brief Execute the pass on a program
   *
   * @param program Input program to transform
   * @return Transformed program
   */
  virtual ProgramPtr operator()(const ProgramPtr& program) = 0;

  /**
   * @brief Get the name of the pass (for debugging)
   */
  [[nodiscard]] virtual std::string GetName() const { return "UnnamedPass"; }
};

}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_PASS_IMPL_H_

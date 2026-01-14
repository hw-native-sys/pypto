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

#ifndef PYPTO_IR_PROGRAM_H_
#define PYPTO_IR_PROGRAM_H_

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/ir/core.h"
#include "pypto/ir/function.h"
#include "pypto/ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

/**
 * @brief Program definition
 *
 * Represents a complete program with a list of functions and optional program name.
 * Programs are immutable IR nodes.
 */
class Program : public IRNode {
 public:
  /**
   * @brief Create a program
   *
   * @param functions List of functions
   * @param name Program name
   * @param span Source location
   */
  Program(std::vector<FunctionPtr> functions, std::string name, Span span)
      : IRNode(std::move(span)), functions_(std::move(functions)), name_(std::move(name)) {}

  [[nodiscard]] std::string TypeName() const override { return "Program"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (name as IGNORE field, functions as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(IRNode::GetFieldDescriptors(),
                          std::make_tuple(reflection::IgnoreField(&Program::name_, "name"),
                                          reflection::UsualField(&Program::functions_, "functions")));
  }

 public:
  std::string name_;                    // Program name
  std::vector<FunctionPtr> functions_;  // List of functions
};

using ProgramPtr = std::shared_ptr<const Program>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_PROGRAM_H_

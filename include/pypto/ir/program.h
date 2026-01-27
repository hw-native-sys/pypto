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

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

/**
 * @brief Program definition
 *
 * Represents a complete program with functions mapped by GlobalVar references.
 * Programs are immutable IR nodes.
 *
 * Functions are stored in a sorted map (by GlobalVar name) to ensure deterministic
 * ordering for structural equality and hashing.
 *
 * @note The GlobalVar name must match the function name and be unique within the program.
 *       Validation of this constraint may be added in future passes.
 */
class Program : public IRNode {
 public:
  /**
   * @brief Create a program from a map of GlobalVars to Functions
   *
   * @param functions Map of GlobalVar references to their corresponding functions
   * @param name Program name (optional)
   * @param span Source location
   */
  Program(std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> functions, std::string name, Span span)
      : IRNode(std::move(span)), functions_(std::move(functions)), name_(std::move(name)) {}

  /**
   * @brief Create a program from a list of functions
   *
   * Convenience constructor that creates GlobalVar references for each function
   * using the function's name. Functions are automatically sorted by name in the map.
   *
   * @param functions List of functions
   * @param name Program name (optional)
   * @param span Source location
   */
  Program(const std::vector<FunctionPtr>& functions, std::string name, Span span);

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::Program; }
  [[nodiscard]] std::string TypeName() const override { return "Program"; }

  /**
   * @brief Get a function by name
   *
   * @param name Function name to look up
   * @return Shared pointer to the function, or nullptr if not found
   */
  [[nodiscard]] FunctionPtr GetFunction(const std::string& name) const;

  /**
   * @brief Get a GlobalVar by name
   *
   * @param name GlobalVar name to look up
   * @return Shared pointer to the GlobalVar, or nullptr if not found
   */
  [[nodiscard]] GlobalVarPtr GetGlobalVar(const std::string& name) const;

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
  std::string name_;                                                 // Program name
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> functions_;  // Map of GlobalVars to Functions
};

using ProgramPtr = std::shared_ptr<const Program>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_PROGRAM_H_

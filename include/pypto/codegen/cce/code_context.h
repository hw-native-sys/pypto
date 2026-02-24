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

#ifndef PYPTO_CODEGEN_CCE_CODE_CONTEXT_H_
#define PYPTO_CODEGEN_CCE_CODE_CONTEXT_H_

#include <string>
#include <unordered_map>

#include "pypto/ir/expr.h"

namespace pypto {
namespace codegen {

/**
 * @brief Context for tracking IR variable to C++ variable name mapping
 *
 * CodeContext provides name mapping from IR variables to valid C++ identifiers,
 * handling name sanitization and caching for consistent code generation.
 */
class CodeContext {
 public:
  CodeContext() = default;

  /**
   * @brief Sanitize an IR variable name for use in C++
   *
   * @param var The IR variable
   * @return A valid C++ identifier
   */
  [[nodiscard]] std::string SanitizeName(const ir::VarPtr& var) const;

  /**
   * @brief Get the C++ variable name for an IR variable
   *
   * If the variable has been seen before, returns the previously assigned name.
   * Otherwise, generates a new name based on the IR variable's name.
   *
   * @param var The IR variable
   * @return The C++ variable name to use
   */
  std::string GetVarName(const ir::VarPtr& var);

  /**
   * @brief Register a variable with a specific C++ name
   *
   * Overrides any previously registered or generated name for this variable.
   * Used to establish naming conventions (e.g., "Global" suffix for GlobalTensor instances).
   *
   * @param var The IR variable
   * @param cpp_name The C++ name to associate with this variable
   */
  void RegisterVar(const ir::VarPtr& var, const std::string& cpp_name);

  /**
   * @brief Register a tensor variable's underlying pointer
   *
   * Associates a tensor variable (e.g., "outputGlobal" or "output_iter") with its
   * underlying raw pointer name (e.g., "output"). This is needed for TASSIGN address
   * computation in block.load/store operations.
   *
   * @param tensor_var_name The tensor variable name (GlobalTensor or iter_arg)
   * @param ptr_name The raw pointer name from function parameters
   */
  void RegisterPointer(const std::string& tensor_var_name, const std::string& ptr_name);

  /**
   * @brief Get the raw pointer name for a tensor variable
   *
   * Returns the raw pointer name that should be used for address computation.
   * If no mapping exists, returns the input tensor_var_name itself (for compatibility).
   *
   * @param tensor_var_name The tensor variable name
   * @return The raw pointer name
   */
  [[nodiscard]] std::string GetPointer(const std::string& tensor_var_name) const;

  /**
   * @brief Clear all state
   */
  void Clear();

 private:
  std::unordered_map<std::string, std::string> name_to_cpp_;  ///< Mapping from IR var name to C++ name
  std::unordered_map<std::string, std::string>
      tensor_to_pointer_;  ///< Mapping from tensor var to raw pointer
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_CCE_CODE_CONTEXT_H_

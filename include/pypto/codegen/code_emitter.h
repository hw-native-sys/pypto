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

#ifndef PYPTO_CODEGEN_CODE_EMITTER_H_
#define PYPTO_CODEGEN_CODE_EMITTER_H_

#include <sstream>
#include <string>

namespace pypto {
namespace codegen {

/**
 * @brief Utility class for emitting structured C++ code with indentation
 *
 * CodeEmitter manages code generation with proper indentation and line buffering.
 * It provides methods to emit lines, blocks, and manage indentation levels.
 */
class CodeEmitter {
 public:
  CodeEmitter() = default;

  /**
   * @brief Emit a single line of code with current indentation
   *
   * @param line The line of code to emit (without indentation or newline)
   */
  void EmitLine(const std::string& line);

  /**
   * @brief Emit a block of code as-is (no indentation added)
   *
   * @param code The code block to emit
   */
  void EmitBlock(const std::string& code);

  /**
   * @brief Increase indentation level by one
   */
  void IncreaseIndent();

  /**
   * @brief Decrease indentation level by one
   */
  void DecreaseIndent();

  /**
   * @brief Get the currently accumulated code
   *
   * @return The generated code as a string
   */
  std::string GetCode() const;

  /**
   * @brief Clear all accumulated code
   */
  void Clear();

 private:
  /**
   * @brief Get the indentation string for the current level
   *
   * @return String containing spaces for current indentation
   */
  std::string GetIndent() const;

  std::stringstream buffer_;  ///< Buffer for accumulated code
  int indent_level_{0};       ///< Current indentation level (0-based)

  static constexpr int kIndentSpaces = 4;  ///< Number of spaces per indent level
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_CODE_EMITTER_H_

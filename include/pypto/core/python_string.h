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

/**
 * @file error.h
 * @brief Core error handling framework with stack trace support
 *
 * This header provides a comprehensive error handling system that captures
 * stack traces at the point of error creation. It includes a base Error class
 * and several specialized error types that mirror Python's exception hierarchy.
 *
 * Key features:
 * - Automatic stack trace capture using libbacktrace
 * - Multiple error types for different error categories
 * - Formatted stack trace output for debugging
 * - Integration with standard C++ exception mechanisms
 */

#ifndef PYPTO_CORE_PYTHON_STRING_H_
#define PYPTO_CORE_PYTHON_STRING_H_

#include <string>

namespace pypto {

/// Quote a UTF-8 string as a Python literal, including embedded control bytes.
inline std::string QuotePythonString(const std::string& value) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string out = "\"";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (c < 32 || c == 127) {
      out += "\\x";
      out += kHex[c >> 4];
      out += kHex[c & 0xf];
    } else {
      out += c;
    }
  }
  out += '"';
  return out;
}

}  // namespace pypto

#endif  // PYPTO_CORE_PYTHON_STRING_H_

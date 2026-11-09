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
 * @file python_string.h
 * @brief Python string literal quoting shared by IR printing and code generation.
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

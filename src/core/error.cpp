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
#include "pypto/core/error.h"

#include <sstream>
#include <string>

namespace pypto {

std::string Error::GetFormattedStackTrace() const { return Backtrace::FormatStackTrace(stack_trace_); }

std::string Error::GetFullMessage() const {
  std::ostringstream oss;

  oss << what();

  // Append C++ stack trace
  std::string stack_trace = GetFormattedStackTrace();
  if (!stack_trace.empty()) {
    oss << "\n\nC++ Traceback (most recent call last):\n";
    oss << stack_trace;
  } else {
    oss << "\n\nNo stack trace available. \n"
           "(Tip: Build with CMake in Debug or RelWithDebInfo mode to enable stack trace support.)";
  }

  return oss.str();
}

}  // namespace pypto

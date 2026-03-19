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

#include "pypto/codegen/cce/code_emitter.h"

#include <cstddef>
#include <string>

#include "pypto/core/logging.h"

namespace pypto {

namespace codegen {

void CodeEmitter::EmitLine(const std::string& line) {
  if (!line.empty()) {
    buffer_ << GetIndent() << line;
  }
  buffer_ << "\n";
}

void CodeEmitter::IncreaseIndent() { indent_level_++; }

void CodeEmitter::DecreaseIndent() {
  INTERNAL_CHECK(indent_level_ > 0) << "Internal error: cannot decrease indent level below 0";
  indent_level_--;
}

std::string CodeEmitter::GetCode() const { return buffer_.str(); }

void CodeEmitter::Clear() {
  buffer_.str("");
  buffer_.clear();
  indent_level_ = 0;
}

std::string CodeEmitter::GetIndent() const {
  return std::string(static_cast<size_t>(indent_level_ * kIndentSpaces), ' ');
}

}  // namespace codegen

}  // namespace pypto

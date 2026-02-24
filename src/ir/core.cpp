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

#include "pypto/ir/core.h"

#include <sstream>
#include <string>
#include <utility>

#include "pypto/ir/span.h"

namespace pypto {
namespace ir {

Span::Span(std::string filename, int begin_line, int begin_column, int end_line, int end_column)
    : filename_(std::move(filename)),
      begin_line_(begin_line),
      begin_column_(begin_column),
      end_line_(end_line),
      end_column_(end_column) {}

std::string Span::to_string() const {
  std::ostringstream oss;
  oss << filename_ << ":" << begin_line_ << ":" << begin_column_;
  return oss.str();
}

bool Span::is_valid() const {
  if (begin_line_ <= 0 || (begin_column_ <= 0 && begin_column_ != -1)) {
    return false;
  }
  if (end_line_ == -1 || end_column_ == -1) {
    return true;
  }
  if (end_line_ <= 0 || (end_column_ <= 0 && end_column_ != -1)) {
    return false;
  }
  if (begin_column_ == -1 || end_column_ == -1) {
    return end_line_ >= begin_line_;
  }
  return end_line_ >= begin_line_ && (end_line_ > begin_line_ || end_column_ >= begin_column_);
}

Span Span::unknown() { return Span("", -1, -1, -1, -1); }

}  // namespace ir
}  // namespace pypto

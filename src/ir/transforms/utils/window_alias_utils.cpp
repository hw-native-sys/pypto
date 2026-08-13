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

#include "pypto/ir/transforms/utils/window_alias_utils.h"

#include <cstddef>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {

void CheckPairwiseDistinctWindows(const std::vector<NamedWindowBuffer>& buffers, const Span& span,
                                  const char* op_name) {
  for (size_t i = 0; i < buffers.size(); ++i) {
    for (size_t j = i + 1; j < buffers.size(); ++j) {
      CHECK_SPAN(buffers[i].buffer.get() != buffers[j].buffer.get(), span)
          << op_name << " " << buffers[i].role << " and " << buffers[j].role
          << " must be different window allocations (two pld.window views over the same "
             "alloc_window_buffer are a cross-process data race under in-kernel TPUT/notify)";
    }
  }
}

}  // namespace ir
}  // namespace pypto

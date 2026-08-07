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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_WINDOW_ALIAS_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_WINDOW_ALIAS_UTILS_H_

#include <vector>

#include "pypto/ir/program.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {

/// A window operand paired with a human-readable role name (e.g. "allreduce
/// signal"), used to produce clear pairwise-aliasing diagnostics.
struct NamedWindowBuffer {
  WindowBufferPtr buffer;
  const char* role;
};

/// Rejects any pair of the given named window buffers that alias the same
/// allocation. A HOST collective's window operands are independently
/// read/written across ranks inside one AIV kernel; any pairwise aliasing is
/// a cross-process data race — data-vs-data is a TPUT/reduce overwrite,
/// data-vs-control is a notify/count write racing a kernel read, and
/// control-vs-control is a notify racing a count publish.
void CheckPairwiseDistinctWindows(const std::vector<NamedWindowBuffer>& buffers, const Span& span,
                                  const char* op_name);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_WINDOW_ALIAS_UTILS_H_

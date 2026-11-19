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

#ifndef PYPTO_IR_TRANSFORMS_AUTO_LAYOUT_WRITER_H_
#define PYPTO_IR_TRANSFORMS_AUTO_LAYOUT_WRITER_H_

#include <vector>

#include "pypto/ir/program.h"
#include "pypto/ir/transforms/auto_layout/model.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace auto_layout {

/// Write a `LayoutAssignment` (both GM `TensorLayout` and on-chip `LayoutSpec`) back to IR.
ProgramPtr ApplyLayoutAssignment(const ProgramPtr& program, const LayoutAssignment& assignment);

/// Synchronize a `Call` or `Submit` return type with updated callee return types.
TypePtr SyncCallReturnType(const TypePtr& call_type, const std::vector<TypePtr>& ret_types);

}  // namespace auto_layout
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_AUTO_LAYOUT_WRITER_H_

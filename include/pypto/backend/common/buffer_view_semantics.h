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

#ifndef PYPTO_BACKEND_COMMON_BUFFER_VIEW_SEMANTICS_H_
#define PYPTO_BACKEND_COMMON_BUFFER_VIEW_SEMANTICS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/type.h"

namespace pypto::backend {

/// Dense physical window size. Unsupported layouts and overflow are unproven.
std::optional<uint64_t> DenseBufferBytes(const ir::BufferTypePtr& type);

/// Physical window size of any byte-addressable rank-1/2 descriptor, including
/// fractal matrix layouts, whose physical extents are already whole boxes.
std::optional<uint64_t> PhysicalBufferBytes(const ir::BufferTypePtr& type);

/// Static native view forms, with every intermediate descriptor already in IR.
/// Subview slices full-width UINT8[N,32] rows; reshape preserves physical bytes.
/// A matrix-space reshape relabels one fractal window within its memory space.
void ValidateBufferSubview(const std::vector<ir::ExprPtr>& args, const ir::TypePtr& result);
void ValidateBufferReshape(const std::vector<ir::ExprPtr>& args, const ir::TypePtr& result);

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_BUFFER_VIEW_SEMANTICS_H_

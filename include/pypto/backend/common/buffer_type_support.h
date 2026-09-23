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

#ifndef PYPTO_BACKEND_COMMON_BUFFER_TYPE_SUPPORT_H_
#define PYPTO_BACKEND_COMMON_BUFFER_TYPE_SUPPORT_H_

#include "pypto/core/dtype.h"
#include "pypto/ir/memory_space.h"

namespace pypto::backend {

/// Element types supported by the dense Vec descriptor and ordinary GM transfer
/// recipes. Arithmetic instructions have separate operand-type contracts.
inline bool IsDenseBufferTransferDtype(DataType dtype) {
  return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32 ||
         dtype == DataType::INT32;
}

/// On-chip spaces whose descriptors carry a cube fractal layout. Their bytes
/// have no row-major byte-view form, so each storage window keeps its own
/// typed descriptor.
inline bool IsMatrixBufferSpace(ir::MemorySpace space) {
  return space == ir::MemorySpace::Mat || space == ir::MemorySpace::Left || space == ir::MemorySpace::Right ||
         space == ir::MemorySpace::Acc;
}

/// Element types a Mat/Left/Right cube operand may hold on every supported target.
inline bool IsCubeOperandDtype(DataType dtype) {
  return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32 ||
         dtype == DataType::INT8;
}

/// Element types a matrix-space descriptor may hold: cube operands in
/// Mat/Left/Right and the FP32/INT32 accumulators in Acc.
inline bool IsMatrixBufferDtype(ir::MemorySpace space, DataType dtype) {
  if (space == ir::MemorySpace::Acc) return dtype == DataType::FP32 || dtype == DataType::INT32;
  return IsMatrixBufferSpace(space) && IsCubeOperandDtype(dtype);
}

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_BUFFER_TYPE_SUPPORT_H_

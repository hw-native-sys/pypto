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

namespace pypto::backend {

/// Element types supported by the dense Vec descriptor and ordinary GM transfer
/// recipes. Arithmetic instructions have separate operand-type contracts.
inline bool IsDenseBufferTransferDtype(DataType dtype) {
  return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32 ||
         dtype == DataType::INT32;
}

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_BUFFER_TYPE_SUPPORT_H_

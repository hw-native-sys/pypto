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
 * @file backend_910b_ops.cpp
 * @brief Backend op registration for Backend910B
 *
 * Registers all standard PTO ops to the 910B backend by delegating
 * to the shared RegisterPTOOps() function. To override specific ops for
 * this backend, register them before calling RegisterPTOOps() and pass
 * the op names in the exclude_ops set.
 */

#include "pypto/backend/910B/backend_910b.h"
#include "pypto/backend/common/pto_ops_common.h"

namespace pypto {
namespace backend {

static const bool kOpsRegistered = [] {
  RegisterPTOOps(Backend910B::Instance());
  return true;
}();

}  // namespace backend
}  // namespace pypto

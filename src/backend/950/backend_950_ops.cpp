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
 * @file backend_950_ops.cpp
 * @brief Backend op registration for Backend950
 *
 * Registers ops to the 950 backend. 950-specific op overrides should be
 * registered here before calling RegisterPTOOps(), and the overridden op
 * names must be passed in the exclude_ops set to avoid duplicate registration.
 *
 * Example of overriding an op:
 * @code
 * REGISTER_BACKEND_OP(Backend950, "tile.matmul")
 *     .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
 *       // 950-specific implementation
 *     });
 *
 * static const bool kOpsRegistered = [] {
 *   RegisterPTOOps(Backend950::Instance(), {"tile.matmul"});
 *   return true;
 * }();
 * @endcode
 */

#include "pypto/backend/950/backend_950.h"
#include "pypto/backend/common/pto_ops_common.h"

namespace pypto {
namespace backend {

static const bool kOpsRegistered = [] {
  RegisterPTOOps(Backend950::Instance());
  return true;
}();

}  // namespace backend
}  // namespace pypto

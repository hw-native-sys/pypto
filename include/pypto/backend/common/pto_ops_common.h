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

#ifndef PYPTO_BACKEND_COMMON_PTO_OPS_COMMON_H_
#define PYPTO_BACKEND_COMMON_PTO_OPS_COMMON_H_

#include <string>
#include <unordered_set>

#include "pypto/backend/common/backend.h"

namespace pypto {
namespace backend {

/**
 * @brief Register all standard PTO ops to the given backend
 *
 * Registers the full set of PTO operator codegen functions to the specified
 * backend. Ops listed in exclude_ops are skipped, allowing a backend to
 * override specific ops with its own REGISTER_BACKEND_OP calls before
 * invoking this function.
 *
 * Typical usage in a backend's ops file:
 * @code
 * // Override a specific op first
 * REGISTER_BACKEND_OP(MyBackend, "tile.matmul")
 *     .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
 *       // custom implementation
 *     });
 *
 * // Then register all remaining standard ops
 * static bool kOpsRegistered = [] {
 *   RegisterPTOOps(MyBackend::Instance(), {"tile.matmul"});
 *   return true;
 * }();
 * @endcode
 *
 * @param backend Backend instance to register ops to
 * @param exclude_ops Set of op names to skip (already registered by the backend)
 */
void RegisterPTOOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops = {});

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_COMMON_PTO_OPS_COMMON_H_

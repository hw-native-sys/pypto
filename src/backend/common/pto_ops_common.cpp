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
 * @file pto_ops_common.cpp
 * @brief Shared PTO op registration for all PTO-based backends.
 *
 * Provides RegisterPTOOps() which registers the full set of standard PTO
 * operator codegen functions to any backend instance by delegating to the
 * per-category RegisterXxxOps entry points (declared in pto_ops_internal.h).
 * Backends that need to override specific ops can pass those op names in the
 * exclude_ops set and register their own implementations before calling this
 * function.
 */

#include "pypto/backend/common/pto_ops_common.h"

#include <string>
#include <unordered_set>

#include "pypto/backend/common/backend.h"
#include "src/backend/common/pto_ops_internal.h"

namespace pypto {
namespace backend {

void RegisterPTOOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops) {
  // Elementwise first to preserve the historical kSimpleOps-first registration
  // order. No op name is registered by more than one category, so the relative
  // order across categories is otherwise immaterial.
  RegisterElementwiseOps(backend, exclude_ops);
  RegisterMemoryOps(backend, exclude_ops);
  RegisterDataMoveOps(backend, exclude_ops);
  RegisterCrossCoreOps(backend, exclude_ops);
  RegisterDistributedOps(backend, exclude_ops);
}

}  // namespace backend
}  // namespace pypto

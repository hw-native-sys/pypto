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
 * @file sync_resources.h
 * @brief Shared Ascend synchronization model construction for backend handlers.
 *
 * This private helper shares the event and signal rules of the supported Ascend
 * backends. Each concrete handler supplies its target and buffer-credit capacity
 * and owns the resulting catalog. Other architectures define their own models.
 */

#ifndef SRC_BACKEND_COMMON_SYNC_RESOURCES_H_
#define SRC_BACKEND_COMMON_SYNC_RESOURCES_H_

#include <cstdint>
#include <string>

#include "pypto/sync/resource_catalog.h"

namespace pypto::backend {

/// Construct validated Ascend resource facts; this helper retains no allocation state.
sync::ResourceCatalog MakeAscendSyncResourceCatalog(std::string target, uint32_t buffer_credit_count);

}  // namespace pypto::backend

#endif  // SRC_BACKEND_COMMON_SYNC_RESOURCES_H_

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
 * @file isa_macro_resources.h
 * @brief Private builders for the backend's ISA signal-resource declarations.
 */
#ifndef SRC_BACKEND_COMMON_ISA_MACRO_RESOURCES_H_
#define SRC_BACKEND_COMMON_ISA_MACRO_RESOURCES_H_

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "pypto/ir/pipe.h"
#include "pypto/sync/isa_macro_resources.h"
#include "pypto/sync/resource_catalog.h"

namespace pypto::backend {

/// Append variant rule sets, not resolved calls; duplicate qualified names are rejected at assembly.
void AppendNpuIsaMacroResources(const std::string& target, std::vector<sync::IsaMacroResources>& entries);
void AppendCommIsaMacroResources(const std::string& target, std::vector<sync::IsaMacroResources>& entries);
void AppendProtocolIsaMacroResources(const std::string& target,
                                     std::vector<sync::IsaMacroResources>& entries);

/// Reject malformed backend declarations before publication; inputs and the catalog remain unchanged.
/// This internal boundary raises InternalError and does not resolve caller-owned parameter values.
void ValidateIsaMacroResources(const sync::IsaMacroResources& entry, const sync::ResourceCatalog& catalog);

/// Start a target-specific declaration; callers must add at least one signal claim.
sync::IsaMacroResources IsaMacro(const std::string& name, const std::string& target);

/// Construct a fixed pipeline claim; hardware pool validation runs when the table is assembled.
sync::IsaResourceClaim IsaFixedEvent(ir::PipeType src, ir::PipeType dst,
                                     std::initializer_list<uint32_t> ids = {0},
                                     sync::CoreScope core = sync::CoreScope::AIV);

/// PtoSetWaitFlag is fixed in Manual, compiler allocated in FlagTest, and empty in Auto mode.
/// Candidate IDs come from the target's hardware catalog; no hidden token is bound by this helper.
void AddIsaHelperEvent(sync::IsaMacroResources& entry, ir::PipeType src, ir::PipeType dst,
                       sync::CoreScope core = sync::CoreScope::AIV);

}  // namespace pypto::backend
#endif  // SRC_BACKEND_COMMON_ISA_MACRO_RESOURCES_H_

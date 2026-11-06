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
 * @brief Backend-owned declarations of signal resources occupied by ISA macros.
 *
 * A declaration collects signal rules across implementation variants for one target. A backend
 * adapter must select the variant, bind parameters and owner identities, and select the build mode
 * before allocation. Mutually exclusive variants must not all be bound to one invocation. Only
 * macros with signal claims are registered; this table does not describe ISA availability or
 * complete memory effects.
 */

#ifndef PYPTO_SYNC_ISA_MACRO_RESOURCES_H_
#define PYPTO_SYNC_ISA_MACRO_RESOURCES_H_

#include <cstdint>
#include <string>
#include <vector>

#include "pypto/sync/resource_catalog.h"

namespace pypto::sync {

/// Manual is neither __PTO_AUTO__ nor PTO_FLAG_TEST; Auto takes precedence when both are defined.
enum class IsaSyncMode { Manual, Auto, FlagTest };

/// CompilerAllocated IDs are candidates, not simultaneous reservations or control over a hidden token.
enum class IsaIdSource { Fixed, Parameter, CompilerAllocated };

/// Owner reserves a shared binding across calls, such as a pipe's construction through its drain.
enum class IsaResourceLifetime { Invocation, Owner };

/// One symbolic signal requirement. Backend rule comments document the originating variant; these
/// records carry no executable variant selector and are not independently resolved invocations.
struct IsaResourceClaim {
  PoolKey pool;
  IsaIdSource source{IsaIdSource::Fixed};
  // Hidden compiler tokens may use hardware-legal IDs that PyPTO reserves for its own allocation policy.
  std::vector<uint32_t> ids;      ///< Fixed IDs, or hardware-legal candidate IDs for a compiler token.
  std::string parameter;          ///< Stable operand/template role for Parameter; empty otherwise.
  std::vector<uint32_t> offsets;  ///< Parameter + each offset; empty for non-parameter claims.
  IsaResourceLifetime lifetime{IsaResourceLifetime::Invocation};
  std::string owner;  ///< Role identifying a shared owner; the caller supplies its actual identity.
  std::vector<IsaSyncMode> modes{IsaSyncMode::Manual, IsaSyncMode::Auto, IsaSyncMode::FlagTest};
};

/// Nonempty signal rule set for one public ISA macro on one target. Missing lookup means no
/// registered declaration; it does not certify arbitrary external implementations as signal-free.
struct IsaMacroResources {
  std::string name;  ///< Qualified ISA name, e.g. pto::TGATHER versus pto::comm::TGATHER.
  std::string target;
  std::vector<IsaResourceClaim> claims;
};

}  // namespace pypto::sync

#endif  // PYPTO_SYNC_ISA_MACRO_RESOURCES_H_

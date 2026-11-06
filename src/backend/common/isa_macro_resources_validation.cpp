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
 * @file isa_macro_resources_validation.cpp
 * @brief Validate symbolic ISA declarations independently of backend registration and allocation.
 *
 * Invalid enum values and impossible ID shapes are backend rule errors.
 * Parameter values remain unresolved until a concrete invocation is available. Validation never
 * rewrites declarations or applies compiler reservation policy. Every published entry must contain
 * at least one signal claim; barriers and macros without signal occupancy do not belong in this table.
 */

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>

#include "pypto/core/logging.h"
#include "pypto/sync/isa_macro_resources.h"
#include "pypto/sync/resource_catalog.h"
#include "src/backend/common/isa_macro_resources.h"

namespace pypto::backend {
namespace {

/// A nonempty owner cannot make an unknown lifetime valid.
void ValidateLifetime(sync::IsaResourceLifetime lifetime, const std::string& owner) {
  INTERNAL_CHECK(lifetime == sync::IsaResourceLifetime::Invocation ||
                 lifetime == sync::IsaResourceLifetime::Owner)
      << "Internal error: ISA declaration has an invalid resource lifetime";
  INTERNAL_CHECK(lifetime == sync::IsaResourceLifetime::Invocation || !owner.empty())
      << "Internal error: cross-call ISA resources require a shared owner role";
}

/// Validate one claim without resolving caller-owned parameter values or allocating candidate IDs.
void ValidateClaim(const sync::IsaResourceClaim& claim, const sync::ResourceCatalog& catalog) {
  INTERNAL_CHECK(claim.pool.kind == sync::ResourceKind::PipelineEvent ||
                 claim.pool.kind == sync::ResourceKind::IntraBlockSignal)
      << "Internal error: ISA macro claims must name a synchronization signal pool";
  const auto pool = catalog.FindPool(claim.pool);
  INTERNAL_CHECK(pool.has_value()) << "Internal error: ISA declaration names an absent or invalid pool";
  INTERNAL_CHECK(!claim.modes.empty() &&
                 std::set<sync::IsaSyncMode>(claim.modes.begin(), claim.modes.end()).size() ==
                     claim.modes.size())
      << "Internal error: ISA declaration has missing or duplicate synchronization modes";
  for (const auto mode : claim.modes) {
    INTERNAL_CHECK(mode == sync::IsaSyncMode::Manual || mode == sync::IsaSyncMode::Auto ||
                   mode == sync::IsaSyncMode::FlagTest)
        << "Internal error: ISA declaration has an invalid synchronization mode";
  }
  ValidateLifetime(claim.lifetime, claim.owner);

  // Offsets describe a shared unresolved base, not several independently allocatable constants.
  if (claim.source == sync::IsaIdSource::Parameter) {
    INTERNAL_CHECK(!claim.parameter.empty() && !claim.offsets.empty() && claim.ids.empty())
        << "Internal error: parameterized ISA claims require a parameter and offsets, not fixed IDs";
    INTERNAL_CHECK(std::set<uint32_t>(claim.offsets.begin(), claim.offsets.end()).size() ==
                   claim.offsets.size())
        << "Internal error: parameterized ISA claim has duplicate offsets";
    for (const auto offset : claim.offsets) {
      INTERNAL_CHECK(offset < pool->capacity)
          << "Internal error: ISA parameter offset cannot fit in the resource pool";
    }
    return;
  }

  // Fixed and candidate IDs obey hardware legality even when compiler policy reserves those IDs.
  INTERNAL_CHECK(claim.source == sync::IsaIdSource::Fixed ||
                 claim.source == sync::IsaIdSource::CompilerAllocated)
      << "Internal error: ISA declaration has an invalid ID source";
  INTERNAL_CHECK(!claim.ids.empty() && claim.parameter.empty() && claim.offsets.empty())
      << "Internal error: fixed or allocated ISA claims require IDs and no parameter expression";
  std::set<uint32_t> seen;
  for (const auto id : claim.ids) {
    INTERNAL_CHECK(id < pool->capacity && seen.insert(id).second &&
                   (pool->legal_ids.empty() ||
                    std::find(pool->legal_ids.begin(), pool->legal_ids.end(), id) != pool->legal_ids.end()))
        << "Internal error: ISA declaration has an illegal or duplicate ID";
  }
}

}  // namespace

/// Validate the same rule boundary for table publication and isolated synthetic native tests.
void ValidateIsaMacroResources(const sync::IsaMacroResources& entry, const sync::ResourceCatalog& catalog) {
  INTERNAL_CHECK(!entry.name.empty() && entry.target == catalog.GetTarget())
      << "Internal error: ISA declaration requires a name and matching target";
  INTERNAL_CHECK(!entry.claims.empty())
      << "Internal error: ISA macro declarations must contain synchronization signal claims";
  for (const auto& claim : entry.claims) ValidateClaim(claim, catalog);
}

}  // namespace pypto::backend

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
 * @file sync_isa_macro_resources_test.cpp
 * @brief Validate ISA signal declarations without a backend registry or device runtime.
 *
 * Synthetic dense and sparse hardware pools exercise invalid declarations and preserve explicit
 * reservations. Target-specific signal footprints are tested through the backend query separately.
 */

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/pipe.h"
#include "pypto/sync/isa_macro_resources.h"
#include "pypto/sync/resource_catalog.h"
#include "src/backend/common/isa_macro_resources.h"

namespace {

namespace sync = pypto::sync;
using pypto::ir::PipeType;

/// Use independent directed banks so duplicate numeric IDs across directions remain legal.
sync::PoolKey EventPool(PipeType src = PipeType::MTE2, PipeType dst = PipeType::MTE3) {
  return {sync::ResourceKind::PipelineEvent, sync::CoreScope::AIV, src, dst};
}

/// Deliberately combine dense, sparse and shared pools without relying on any target facts.
sync::ResourceCatalog MakeCatalog(const std::string& target = "synthetic") {
  return {target,
          {{EventPool(), 8, {}, {7}, false},
           {EventPool(PipeType::MTE3, PipeType::MTE2), 8, {7, 1, 3}, {7}, true},
           {{sync::ResourceKind::IntraBlockSignal, sync::CoreScope::CoreGroup}, 4, {}, {}, false}}};
}

/// The native table uses the same validator, without linking registrations or a concrete backend.
sync::IsaMacroResources MakeIsaDeclaration(const std::string& target = "synthetic") {
  sync::IsaMacroResources entry;
  entry.name = "pto::SYNTHETIC";
  entry.target = target;
  sync::IsaResourceClaim claim;
  claim.pool = EventPool();
  claim.ids = {7, 1};
  entry.claims = {claim};
  return entry;
}

/// Invalid symbolic rules must fail before the backend publishes their signal claims.
void ExpectInvalid(const sync::IsaMacroResources& entry, const sync::ResourceCatalog& catalog) {
  bool caught = false;
  try {
    pypto::backend::ValidateIsaMacroResources(entry, catalog);
  } catch (const pypto::InternalError&) {
    caught = true;
  }
  assert(caught);
}

/// Sparse declarations require actual signal claims; reserved IDs remain hardware-legal.
void TestIsaDeclarationBoundary() {
  const auto catalog = MakeCatalog();
  auto entry = MakeIsaDeclaration();
  pypto::backend::ValidateIsaMacroResources(entry, catalog);
  assert((entry.claims[0].ids == std::vector<uint32_t>{7, 1}));
  assert(entry.claims[0].pool == EventPool());

  auto malformed = entry;
  malformed.claims.clear();
  ExpectInvalid(malformed, catalog);
  malformed = entry;
  malformed.target = "different-target";
  ExpectInvalid(malformed, catalog);
  malformed = entry;
  malformed.name.clear();
  ExpectInvalid(malformed, catalog);
}

/// Owner strings cannot hide invalid enum tags, and every nonlocal lifetime needs an actual owner role.
void TestIsaClaimEnumsAndOwners() {
  const auto catalog = MakeCatalog();
  auto entry = MakeIsaDeclaration();
  entry.claims[0].lifetime = sync::IsaResourceLifetime::Owner;
  ExpectInvalid(entry, catalog);
  entry.claims[0].owner = "pipe";
  pypto::backend::ValidateIsaMacroResources(entry, catalog);

  entry.claims[0].owner = "present-but-not-a-valid-lifetime";
  entry.claims[0].lifetime = static_cast<sync::IsaResourceLifetime>(255);
  ExpectInvalid(entry, catalog);
  entry = MakeIsaDeclaration();
  entry.claims[0].source = static_cast<sync::IsaIdSource>(255);
  ExpectInvalid(entry, catalog);
  for (const auto& modes :
       {std::vector<sync::IsaSyncMode>{}, std::vector{sync::IsaSyncMode::Manual, sync::IsaSyncMode::Manual},
        std::vector{static_cast<sync::IsaSyncMode>(255)}}) {
    entry = MakeIsaDeclaration();
    entry.claims[0].modes = modes;
    ExpectInvalid(entry, catalog);
  }
}

/// Parameter shape and candidate IDs must be checked before any concrete invocation binds a value.
void TestIsaIdSources() {
  const auto catalog = MakeCatalog();
  for (const auto source : {sync::IsaIdSource::Fixed, sync::IsaIdSource::CompilerAllocated}) {
    auto entry = MakeIsaDeclaration();
    entry.claims[0].source = source;
    // ID 7 is compiler-reserved, but fixed ISA uses and downstream allocation can still occupy it.
    pypto::backend::ValidateIsaMacroResources(entry, catalog);
    for (const auto& ids : {std::vector<uint32_t>{}, std::vector<uint32_t>{0, 0}, std::vector<uint32_t>{8}}) {
      entry.claims[0].ids = ids;
      ExpectInvalid(entry, catalog);
    }
    entry.claims[0].ids = {0};
    entry.claims[0].pool = EventPool(PipeType::MTE3, PipeType::MTE2);
    ExpectInvalid(entry, catalog);  // Sparse bank only admits 1, 3 and 7.
  }

  auto entry = MakeIsaDeclaration();
  auto& claim = entry.claims[0];
  claim.source = sync::IsaIdSource::Parameter;
  claim.ids.clear();
  claim.parameter = "session.syncId";
  claim.offsets = {0};
  claim.owner = "session";
  pypto::backend::ValidateIsaMacroResources(entry, catalog);
  for (const auto& offsets :
       {std::vector<uint32_t>{}, std::vector<uint32_t>{0, 0}, std::vector<uint32_t>{8}}) {
    claim.offsets = offsets;
    ExpectInvalid(entry, catalog);
  }
  claim.offsets = {0};
  claim.ids = {0};
  ExpectInvalid(entry, catalog);
  claim.ids.clear();
  claim.parameter.clear();
  ExpectInvalid(entry, catalog);
}

/// Absent, invalid and zero-capacity event domains cannot accept even a numerically small ID.
void TestIsaEventPoolDomains() {
  const auto catalog = MakeCatalog();
  for (const auto pool : {EventPool(PipeType::MTE2, PipeType::V), EventPool(PipeType::ALL, PipeType::MTE3),
                          sync::PoolKey{sync::ResourceKind::PipelineEvent, static_cast<sync::CoreScope>(255),
                                        PipeType::MTE2, PipeType::MTE3}}) {
    auto entry = MakeIsaDeclaration();
    entry.claims[0].pool = pool;
    ExpectInvalid(entry, catalog);
  }

  const sync::ResourceCatalog unavailable("synthetic", {{EventPool(), 0, {}, {}, false}});
  auto entry = MakeIsaDeclaration();
  entry.claims[0].ids = {0};
  ExpectInvalid(entry, unavailable);
}

/// Validating claims must preserve alias spelling, replica identity, ID order and catalog policy.
void TestIsaValidationPreservesInputs() {
  const auto catalog = MakeCatalog();
  auto entry = MakeIsaDeclaration();
  const sync::PoolKey alias{sync::ResourceKind::IntraBlockSignal, sync::CoreScope::AIC, PipeType::M,
                            PipeType::FIX, 19};
  entry.claims[0].pool = alias;
  entry.claims[0].ids = {3, 0};
  const auto before = catalog.FindPool(alias);
  assert(before);
  pypto::backend::ValidateIsaMacroResources(entry, catalog);
  assert(entry.claims[0].pool == alias);
  assert((entry.claims[0].ids == std::vector<uint32_t>{3, 0}));
  const auto after = catalog.FindPool(alias);
  assert(after && after->key == before->key && after->capacity == before->capacity);
  assert(after->legal_ids == before->legal_ids && after->reserved_ids == before->reserved_ids);
}

/// Other hardware capacities are not signal IDs even when their pools permit the numeric claim.
void TestIsaSignalPoolKinds() {
  const sync::PoolKey credit{sync::ResourceKind::BufferCredit, sync::CoreScope::AIV};
  const sync::ResourceCatalog catalog("synthetic", {{credit, 8, {}, {}, false}});
  auto entry = MakeIsaDeclaration();
  entry.claims[0].pool = credit;
  ExpectInvalid(entry, catalog);
  entry.claims[0].pool.kind = static_cast<sync::ResourceKind>(255);
  ExpectInvalid(entry, catalog);
}

}  // namespace

/// The CMake target keeps these assertions active for release configurations.
int main() {
  TestIsaDeclarationBoundary();
  TestIsaClaimEnumsAndOwners();
  TestIsaIdSources();
  TestIsaEventPoolDomains();
  TestIsaValidationPreservesInputs();
  TestIsaSignalPoolKinds();
}

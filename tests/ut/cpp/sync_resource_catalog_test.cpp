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
 * @file sync_resource_catalog_test.cpp
 * @brief Native hardware-catalog contracts, independent of Python and device runtime.
 *
 * Synthetic catalogs exercise the same validation as native target rules.
 * Backend-provided facts are tested through the real backend in Python tests.
 */

#include <cassert>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/pipe.h"
#include "pypto/sync/resource_catalog.h"

namespace {

namespace sync = pypto::sync;
using pypto::ir::PipeType;

/// Identify a conventional event bank while retaining explicit executor/direction isolation.
sync::PoolKey EventPool(PipeType src = PipeType::MTE2, PipeType dst = PipeType::V,
                        sync::CoreScope core = sync::CoreScope::AIV, uint32_t replica = 0) {
  return {sync::ResourceKind::PipelineEvent, core, src, dst, replica};
}

/// Require a user-facing validation error rather than a crash or accepted invalid native construction.
template <typename Callback>
void ExpectValueError(Callback callback) {
  bool caught = false;
  try {
    callback();
  } catch (const pypto::ValueError&) {
    caught = true;
  }
  assert(caught);
}

/// Build a fresh catalog so every malformed row exercises constructor validation independently.
void ExpectInvalidPool(sync::PoolSpec spec) {
  ExpectValueError([&] { sync::ResourceCatalog("synthetic", {std::move(spec)}); });
}

/// Hardware legality, compiler policy and unavailable families remain distinguishable in query results.
void TestDenseSparseAndUnavailablePools() {
  const auto dense_key = EventPool();
  const auto sparse_key = EventPool(PipeType::V, PipeType::MTE3);
  const sync::PoolKey unavailable_key{sync::ResourceKind::BufferCredit, sync::CoreScope::AIV};
  const sync::ResourceCatalog catalog("synthetic", {{dense_key, 8, {}, {7}, false},
                                                    {sparse_key, 8, {6, 0, 3}, {3}, true},
                                                    {unavailable_key, 0, {}, {}, false}});
  assert(catalog.GetTarget() == "synthetic");
  const auto dense = catalog.FindPool(dense_key);
  assert(dense && dense->capacity == 8 && dense->legal_ids.empty());
  // Reserving ID7 for compiler policy does not remove it from the dense hardware range [0,8).
  assert(dense->reserved_ids == std::vector<uint32_t>{7});
  const auto sparse = catalog.FindPool(sparse_key);
  assert(sparse && sparse->capacity == 8 && sparse->requires_contiguous_ids);
  assert((sparse->legal_ids == std::vector<uint32_t>{6, 0, 3}));
  assert(sparse->reserved_ids == std::vector<uint32_t>{3});
  const auto unavailable = catalog.FindPool(unavailable_key);
  assert(unavailable && unavailable->capacity == 0);
  assert(!catalog.FindPool(EventPool(PipeType::MTE3, PipeType::MTE2)));
}

/// Direction/core select capabilities, while arbitrary replicas preserve independent conflict identities.
void TestEventDomainsAndReplicas() {
  const auto forward = EventPool();
  const auto reverse = EventPool(PipeType::V, PipeType::MTE2);
  const auto aic = EventPool(PipeType::MTE2, PipeType::V, sync::CoreScope::AIC);
  const sync::ResourceCatalog catalog(
      "synthetic", {{forward, 8, {}, {}, false}, {reverse, 4, {}, {}, false}, {aic, 2, {}, {}, false}});
  assert(catalog.FindPool(forward)->capacity == 8);
  assert(catalog.FindPool(reverse)->capacity == 4);
  assert(catalog.FindPool(aic)->capacity == 2);
  auto replica_key = forward;
  replica_key.replica = 27;
  const auto replica = catalog.FindPool(replica_key);
  assert(replica && replica->capacity == 8 && replica->key == replica_key);
  // A replica lookup must not overwrite the prototype that later replica-zero callers receive.
  assert(catalog.FindPool(forward)->key.replica == 0);
}

/// Signal spellings alias one group domain; buffer credits keep core identity and normalize only pipes.
void TestCanonicalAliases() {
  const sync::PoolKey signal{sync::ResourceKind::IntraBlockSignal, sync::CoreScope::AIC, PipeType::M,
                             PipeType::FIX};
  const sync::PoolKey credits{sync::ResourceKind::BufferCredit, sync::CoreScope::AIV, PipeType::MTE2,
                              PipeType::V};
  const sync::ResourceCatalog catalog("synthetic",
                                      {{signal, 16, {}, {}, false}, {credits, 32, {}, {}, false}});
  auto alternate = signal;
  alternate.core = sync::CoreScope::AIV;
  alternate.src = PipeType::V;
  alternate.dst = PipeType::S;
  alternate.replica = 12;
  const auto canonical = sync::CanonicalizePoolKey(alternate);
  assert(canonical.core == sync::CoreScope::CoreGroup);
  assert(canonical.src == PipeType::ALL && canonical.dst == PipeType::ALL && canonical.replica == 12);
  assert(alternate.core == sync::CoreScope::AIV && alternate.src == PipeType::V);
  const auto shared = catalog.FindPool(alternate);
  assert(shared && shared->key == canonical && shared->capacity == 16);
  assert(sync::DescribePoolKey(alternate) == "intra-block:group:ALL->ALL:12");
  // Query normalization never rewrites numeric IDs: the catalog exposes sixteen logical bases, not
  // an extra independently allocatable A5 bank for physical base+16 companion signaling.
  assert(shared->legal_ids.empty());
  const auto buffer = catalog.FindPool(credits);
  assert(buffer && buffer->key.core == sync::CoreScope::AIV && buffer->key.src == PipeType::ALL);
  auto other_core = credits;
  other_core.core = sync::CoreScope::AIC;
  assert(!catalog.FindPool(other_core));
}

/// Raw equality/order must agree; physical alias equivalence remains an explicit boundary operation.
void TestRawKeyIdentity() {
  const sync::PoolKey aic{sync::ResourceKind::IntraBlockSignal, sync::CoreScope::AIC};
  const sync::PoolKey aiv{sync::ResourceKind::IntraBlockSignal, sync::CoreScope::AIV};
  assert(!(aic == aiv));
  assert((std::set<sync::PoolKey>{aic, aiv}.size() == 2));
  assert(sync::CanonicalizePoolKey(aic) == sync::CanonicalizePoolKey(aiv));
  assert(sync::DescribePoolKey(EventPool()) == "pipeline-event:aiv:MTE2->V:0");
}

/// Catalog ownership must isolate both caller-supplied vectors and every returned value snapshot.
void TestCopiedInputAndQuerySnapshots() {
  const auto key = EventPool();
  std::vector<sync::PoolSpec> input{{key, 8, {0, 3, 7}, {7}, true}};
  const sync::ResourceCatalog catalog("synthetic", input);
  input[0].capacity = 1;
  input[0].legal_ids.clear();
  input[0].reserved_ids.clear();
  auto changed = catalog.FindPool(key);
  assert(changed);
  changed->key.src = PipeType::S;
  changed->capacity = 0;
  changed->legal_ids.clear();
  changed->reserved_ids.clear();
  changed->requires_contiguous_ids = false;
  const auto original = catalog.FindPool(key);
  assert(original && original->key == key && original->capacity == 8 && original->requires_contiguous_ids);
  assert((original->legal_ids == std::vector<uint32_t>{0, 3, 7}));
  assert(original->reserved_ids == std::vector<uint32_t>{7});
}

/// Native callers cannot bypass dense/sparse legality or policy-subset validation.
void TestInvalidIdSetsAndTarget() {
  const auto key = EventPool();
  ExpectInvalidPool({key, 8, {8}, {}, false});
  ExpectInvalidPool({key, 8, {0, 0}, {}, false});
  ExpectInvalidPool({key, 8, {}, {8}, false});
  ExpectInvalidPool({key, 8, {}, {7, 7}, false});
  ExpectInvalidPool({key, 8, {0, 3}, {1}, false});
  ExpectInvalidPool({key, 0, {0}, {}, false});
  ExpectInvalidPool({key, 0, {}, {0}, false});
  auto replica = key;
  replica.replica = 1;
  ExpectInvalidPool({replica, 8, {}, {}, false});
  ExpectValueError([] { sync::ResourceCatalog("", {}); });
}

/// Validation precedes normalization: malformed signal aliases must not become valid ALL/ALL keys.
void TestInvalidKeysAreRejectedAtBothBoundaries() {
  const sync::ResourceCatalog catalog(
      "synthetic", {{EventPool(), 8, {}, {}, false},
                    {{sync::ResourceKind::IntraBlockSignal, sync::CoreScope::CoreGroup}, 16, {}, {}, false},
                    {{sync::ResourceKind::BufferCredit, sync::CoreScope::AIV}, 32, {}, {}, false}});
  std::vector<sync::PoolKey> invalid;
  auto key = EventPool();
  key.kind = static_cast<sync::ResourceKind>(255);
  invalid.push_back(key);
  key = EventPool();
  key.core = static_cast<sync::CoreScope>(255);
  invalid.push_back(key);
  key = EventPool();
  key.src = static_cast<PipeType>(255);
  invalid.push_back(key);
  key = EventPool();
  key.dst = static_cast<PipeType>(255);
  invalid.push_back(key);
  invalid.push_back(EventPool(PipeType::V, PipeType::V));
  invalid.push_back(EventPool(PipeType::ALL, PipeType::V));
  invalid.push_back(EventPool(PipeType::V, PipeType::ALL));
  invalid.push_back(EventPool(PipeType::MTE2, PipeType::V, sync::CoreScope::CoreGroup));
  key = {sync::ResourceKind::IntraBlockSignal, sync::CoreScope::AIC, static_cast<PipeType>(255), PipeType::V};
  invalid.push_back(key);
  key = {sync::ResourceKind::IntraBlockSignal, static_cast<sync::CoreScope>(255)};
  invalid.push_back(key);
  key = {sync::ResourceKind::IntraBlockSignal, sync::CoreScope::AIV, PipeType::V, static_cast<PipeType>(255)};
  invalid.push_back(key);
  key = {sync::ResourceKind::BufferCredit, static_cast<sync::CoreScope>(255)};
  invalid.push_back(key);
  for (const auto& bad : invalid) {
    ExpectInvalidPool({bad, 8, {}, {}, false});
    assert(!catalog.FindPool(bad));
  }
}

/// Duplicate physical domains fail even if their raw API spellings differ or their facts agree.
void TestCanonicalAliasCollisionsAreRejected() {
  const auto key = EventPool();
  ExpectValueError(
      [&] { sync::ResourceCatalog("synthetic", {{key, 8, {}, {}, false}, {key, 8, {}, {}, false}}); });
  const sync::PoolKey aic{sync::ResourceKind::IntraBlockSignal, sync::CoreScope::AIC, PipeType::M,
                          PipeType::FIX};
  const sync::PoolKey aiv{sync::ResourceKind::IntraBlockSignal, sync::CoreScope::AIV, PipeType::V,
                          PipeType::MTE2};
  ExpectValueError(
      [&] { sync::ResourceCatalog("synthetic", {{aic, 16, {}, {}, false}, {aiv, 8, {}, {}, false}}); });
  const sync::PoolKey credits{sync::ResourceKind::BufferCredit, sync::CoreScope::AIV};
  auto alias = credits;
  alias.src = PipeType::S;
  alias.dst = PipeType::MTE3;
  ExpectValueError([&] {
    sync::ResourceCatalog("synthetic", {{credits, 32, {}, {}, false}, {alias, 32, {}, {}, false}});
  });
}

}  // namespace

/// Keep assertion checks active through the native target's release-build compile options.
int main() {
  TestDenseSparseAndUnavailablePools();
  TestEventDomainsAndReplicas();
  TestCanonicalAliases();
  TestRawKeyIdentity();
  TestCopiedInputAndQuerySnapshots();
  TestInvalidIdSetsAndTarget();
  TestInvalidKeysAreRejectedAtBothBoundaries();
  TestCanonicalAliasCollisionsAreRejected();
}

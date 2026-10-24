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
 * @file resource_catalog.h
 * @brief Immutable resource domains and capacity queries for synchronization.
 *
 * Backends provide catalogs of target facts; callers own per-compilation state. Queries
 * require neither an installed ISA nor runtime discovery and return copied pool
 * specifications. No allocation, library footprints or synchronization placement
 * is performed by this interface. A modeled bank identifies resource capacity;
 * the backend still determines whether an instruction is legal on its executor.
 */

#ifndef PYPTO_SYNC_RESOURCE_CATALOG_H_
#define PYPTO_SYNC_RESOURCE_CATALOG_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "pypto/ir/pipe.h"

namespace pypto::sync {

/// Equal numeric IDs in different resource families do not imply shared ownership.
enum class ResourceKind { PipelineEvent, IntraBlockSignal, BufferCredit };
/// Pipeline events use AIC/AIV executors; CoreGroup names a shared signal domain.
enum class CoreScope { AIC, AIV, CoreGroup };

/// One resource conflict domain. Pipeline events retain their ordered pipe pair
/// and executor. Intra-block signal aliases canonicalize to CoreGroup and ALL/ALL.
/// replica identifies independent instances of the same hardware capabilities;
/// it is not an event ID, subblock lane offset or capacity limit.
struct PoolKey {
  ResourceKind kind{ResourceKind::PipelineEvent};
  CoreScope core{CoreScope::AIV};
  ir::PipeType src{ir::PipeType::ALL};
  ir::PipeType dst{ir::PipeType::ALL};
  uint32_t replica{0};

  /// Compare raw fields; normalize aliases explicitly before comparing physical domains.
  bool operator==(const PoolKey& other) const;
  /// Order raw fields consistently with equality, including the replica identity.
  bool operator<(const PoolKey& other) const;
};

/// Capacity bounds hardware IDs in [0, capacity); zero denotes an unavailable
/// resource family. Empty legal_ids means that entire dense range, while a
/// nonempty vector names a sparse legal subset. reserved_ids are compiler policy
/// within that legal set and do not make an explicitly authored hardware ID illegal.
struct PoolSpec {
  PoolKey key;
  uint32_t capacity{0};
  std::vector<uint32_t> legal_ids;
  std::vector<uint32_t> reserved_ids;
  /// Multi-ID bindings must be consecutive when true; individual ID legality is unchanged.
  bool requires_contiguous_ids{false};
};

/// Owned, read-only target facts. Construction validates custom and built-in
/// input through the same boundary; const queries can serve concurrent callers.
class ResourceCatalog {
 public:
  /// Own target facts. Invalid enums, event directions, nonzero
  /// prototype replicas, duplicate alias domains or illegal IDs raise pypto::ValueError.
  /// target must be nonempty. Construction performs no toolchain discovery.
  ResourceCatalog(std::string target, std::vector<PoolSpec> pools);

  /// Return a copied pool for a validated, canonicalized query, or nullopt for an
  /// invalid key/absent domain. Stored replica-zero prototypes describe all replicas;
  /// the returned key retains the caller's replica. No query changes catalog facts.
  [[nodiscard]] std::optional<PoolSpec> FindPool(const PoolKey& key) const;

  /// Borrow the target name for this catalog's lifetime.
  [[nodiscard]] const std::string& GetTarget() const { return target_; }

 private:
  std::string target_;
  std::map<PoolKey, PoolSpec> pools_;
};

/// Return a canonical alias domain while preserving replica and leaving enum
/// validation to catalog boundaries. Numeric IDs are never transformed: A5's
/// base+16 companion signaling belongs to lowering, not an independent ID bank.
[[nodiscard]] PoolKey CanonicalizePoolKey(PoolKey key);
/// Format a valid key's canonical identity for diagnostics; no catalog lookup occurs.
[[nodiscard]] std::string DescribePoolKey(const PoolKey& key);

/// Borrow a backend's process-lifetime immutable catalog, or null for an unknown
/// target. Backend dispatch does not consult or change the active compilation
/// target, read files or discover an ISA installation.
[[nodiscard]] const ResourceCatalog* FindResourceCatalog(const std::string& target);

}  // namespace pypto::sync

#endif  // PYPTO_SYNC_RESOURCE_CATALOG_H_

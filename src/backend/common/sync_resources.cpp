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
 * @file sync_resources.cpp
 * @brief Ascend resource definitions and backend dispatch for catalog queries.
 *
 * Hardware constants and reservation rules live with their backend providers.
 * The public sync lookup borrows those providers' catalogs; the generic catalog
 * implementation remains independent of backend construction and selection.
 */

#include "src/backend/common/sync_resources.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/pipe.h"
#include "pypto/sync/resource_catalog.h"

namespace pypto::backend {
namespace {

// PTOAS include/PTO/Transforms/InsertSync/SyncEventIdAllocation.h defines
// kTotalEventIdNum=8 and kBlockSyncSetWaitEventIdNum=16. Signals count logical
// bases, not A5's base+16 physical companion: pto-isa include/pto/npu/a5/TPush.hpp
// uses VEC_CORE_ID_OFFSET for that subblock mapping during lowering.
constexpr uint32_t kEventIdCount = 8;
constexpr uint32_t kIntraBlockSignalBaseCount = 16;

// PTOAS lib/PTO/Transforms/InsertSync/SyncEventIdAllocation.cpp uses
// reservedEventIdNum to withhold the last event in each V<->S pool from automatic
// allocation. This is compiler policy; ID7 remains in the hardware range for
// explicitly authored synchronization.
constexpr uint32_t kScalarVectorReservedEventId = kEventIdCount - 1;

}  // namespace

using sync::CoreScope;
using sync::PoolSpec;
using sync::ResourceKind;

/// Build one target's resource model without inferring executor instruction legality.
sync::ResourceCatalog MakeAscendSyncResourceCatalog(std::string target, uint32_t buffer_credit_count) {
  constexpr std::array cores{CoreScope::AIC, CoreScope::AIV};
  constexpr std::array pipes{ir::PipeType::MTE1, ir::PipeType::MTE2, ir::PipeType::MTE3, ir::PipeType::M,
                             ir::PipeType::V,    ir::PipeType::S,    ir::PipeType::FIX};
  std::vector<PoolSpec> pools;
  pools.reserve(cores.size() * pipes.size() * (pipes.size() - 1) + 2);

  // PTOAS SyncEventIdAllocation::ScopePair keys events by ordered pipe pair.
  // Both executors receive the same capability prototype; backend validation,
  // not the existence of this entry, decides which instructions each can run.
  for (const auto core : cores) {
    for (const auto src : pipes) {
      for (const auto dst : pipes) {
        if (src == dst) continue;
        std::vector<uint32_t> reserved;
        if ((src == ir::PipeType::V && dst == ir::PipeType::S) ||
            (src == ir::PipeType::S && dst == ir::PipeType::V)) {
          reserved.push_back(kScalarVectorReservedEventId);
        }
        pools.push_back(
            {{ResourceKind::PipelineEvent, core, src, dst}, kEventIdCount, {}, std::move(reserved), false});
      }
    }
  }

  // Block set/wait uses one shared ScopePair in PTOAS, independent of pipe.
  // Block-all's conditional use of IDs14/15 belongs to per-compilation policy,
  // so the catalog reports the full sixteen-base domain without reserving them.
  pools.push_back(
      {{ResourceKind::IntraBlockSignal, CoreScope::CoreGroup}, kIntraBlockSignalBaseCount, {}, {}, false});
  // Buffer credits use the existing AIV-owned model, shared across directions.
  // Preserve a registered zero-capacity pool to distinguish unavailable from unknown.
  pools.push_back({{ResourceKind::BufferCredit, CoreScope::AIV}, buffer_credit_count, {}, {}, false});
  return sync::ResourceCatalog(std::move(target), std::move(pools));
}

}  // namespace pypto::backend

namespace pypto::sync {

/// Borrow the matching backend's catalog without consulting or changing BackendConfig.
const ResourceCatalog* FindResourceCatalog(const std::string& target) {
  // Target names come from their existing handlers, not a parallel synchronization
  // registry. Only a matching backend constructs its resource catalog.
  constexpr std::array types{backend::BackendType::Ascend910, backend::BackendType::Ascend950};
  for (const auto type : types) {
    const auto* handler = backend::GetBackendInstance(type)->GetHandler();
    if (handler->GetPtoTargetArch() == target) return &handler->GetSyncResourceCatalog();
  }
  return nullptr;
}

}  // namespace pypto::sync

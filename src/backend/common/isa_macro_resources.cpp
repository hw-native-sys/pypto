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
 * @file isa_macro_resources.cpp
 * @brief Assemble and validate the backend-owned ISA synchronization declarations.
 */

#include "src/backend/common/isa_macro_resources.h"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/logging.h"
#include "pypto/ir/pipe.h"
#include "pypto/sync/isa_macro_resources.h"
#include "pypto/sync/resource_catalog.h"

namespace pypto::backend {

sync::IsaMacroResources IsaMacro(const std::string& name, const std::string& target) {
  sync::IsaMacroResources entry;
  entry.name = name;
  entry.target = target;
  return entry;
}

sync::IsaResourceClaim IsaFixedEvent(ir::PipeType src, ir::PipeType dst, std::initializer_list<uint32_t> ids,
                                     sync::CoreScope core) {
  sync::IsaResourceClaim claim;
  claim.pool = {sync::ResourceKind::PipelineEvent, core, src, dst, 0};
  claim.ids = ids;
  return claim;
}

void AddIsaHelperEvent(sync::IsaMacroResources& entry, ir::PipeType src, ir::PipeType dst,
                       sync::CoreScope core) {
  auto claim = IsaFixedEvent(src, dst, {0}, core);
  claim.modes = {sync::IsaSyncMode::Manual};
  // PtoSetWaitFlag emits no call in Auto mode and uses its default ID in Manual mode.
  entry.claims.push_back(claim);
  claim.source = sync::IsaIdSource::CompilerAllocated;
  // Reuse hardware legality instead of maintaining a second event-capacity definition. Hidden
  // downstream tokens need not follow PyPTO's reservation policy, so include reserved legal IDs.
  // Deriving this possible footprint does not bind or control the hidden token.
  const auto* catalog = sync::FindResourceCatalog(entry.target);
  INTERNAL_CHECK(catalog != nullptr) << "Internal error: ISA helper requires a target resource catalog";
  const auto pool = catalog->FindPool(claim.pool);
  INTERNAL_CHECK(pool.has_value()) << "Internal error: ISA helper names an absent resource pool";
  claim.ids = pool->legal_ids;
  if (claim.ids.empty()) {
    claim.ids.resize(pool->capacity);
    std::iota(claim.ids.begin(), claim.ids.end(), uint32_t{0});
  }
  claim.modes = {sync::IsaSyncMode::FlagTest};
  entry.claims.push_back(std::move(claim));
}

namespace {

using Table = std::map<std::string, sync::IsaMacroResources>;

/// Check native rules once when publishing each target table; never silently discard malformed rows.
Table BuildTable(const std::string& target) {
  std::vector<sync::IsaMacroResources> entries;
  AppendNpuIsaMacroResources(target, entries);
  AppendCommIsaMacroResources(target, entries);
  AppendProtocolIsaMacroResources(target, entries);
  const auto* catalog = sync::FindResourceCatalog(target);
  INTERNAL_CHECK(catalog != nullptr) << "ISA declarations require a hardware resource catalog";
  Table table;
  for (auto& entry : entries) {
    ValidateIsaMacroResources(entry, *catalog);
    auto name = entry.name;
    INTERNAL_CHECK(table.emplace(name, std::move(entry)).second)
        << "Duplicate ISA macro resource declaration: " << name;
  }
  return table;
}

/// Initialization and lookups never inspect an installed ISA tree or global backend configuration.
const Table& GetTable(const std::string& target) {
  if (target == "a2a3") {
    static const auto table = BuildTable("a2a3");
    return table;
  }
  INTERNAL_CHECK(target == "a5") << "No ISA macro resource table for target " << target;
  static const auto table = BuildTable("a5");
  return table;
}

}  // namespace

std::optional<sync::IsaMacroResources> Backend::FindIsaMacroResources(const std::string& name) const {
  const auto& table = GetTable(GetHandler()->GetPtoTargetArch());
  const auto it = table.find(name);
  if (it == table.end()) return std::nullopt;
  return it->second;
}

std::vector<std::string> Backend::GetIsaMacroNames() const {
  const auto& table = GetTable(GetHandler()->GetPtoTargetArch());
  std::vector<std::string> names;
  names.reserve(table.size());
  for (const auto& item : table) names.push_back(item.first);
  return names;
}

}  // namespace pypto::backend

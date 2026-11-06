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
 * @file isa_macro_resources_protocol.cpp
 * @brief Hidden signal reservations of cross-core, collective and prefetch macros.
 *
 * Ready/free signals are owned by a pipe across macro calls. They occupy IntraBlockSignal IDs,
 * not the separate BufferCredit pool. Prefetch control events instead complete within one call.
 * Rules include multiple variants; template arguments and session ownership must select the
 * applicable claims before binding. They are not all requirements of each concrete invocation.
 */

#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/pipe.h"
#include "pypto/sync/isa_macro_resources.h"
#include "pypto/sync/resource_catalog.h"
#include "src/backend/common/isa_macro_resources.h"

namespace pypto::backend {
namespace {

using ir::PipeType;
using sync::CoreScope;
using sync::IsaIdSource;
using sync::IsaMacroResources;
using sync::IsaResourceClaim;
using sync::IsaResourceLifetime;
using sync::IsaSyncMode;

/// Reserve logical signal bases. A5's +16 companions belong to lowering, not extra pool capacity.
IsaResourceClaim ParameterSignal(const std::string& parameter, std::initializer_list<uint32_t> offsets,
                                 const std::string& owner) {
  IsaResourceClaim claim;
  claim.pool = {sync::ResourceKind::IntraBlockSignal, CoreScope::CoreGroup};
  claim.source = IsaIdSource::Parameter;
  claim.parameter = parameter;
  claim.offsets = offsets;
  claim.lifetime = IsaResourceLifetime::Owner;
  claim.owner = owner;
  return claim;
}

/// Reserve fixed collective IDs independently of their pipe-specific instruction spelling.
IsaResourceClaim FixedSignal(std::initializer_list<uint32_t> ids) {
  IsaResourceClaim claim;
  claim.pool = {sync::ResourceKind::IntraBlockSignal, CoreScope::CoreGroup};
  claim.ids = ids;
  claim.modes = {IsaSyncMode::Manual, IsaSyncMode::FlagTest};
  return claim;
}

/// Ready/free flags share the pipe's identity from initialization through final drain/destruction.
void AddPipeOwner(IsaMacroResources& entry) {
  // Source anomalies do not become valid protocols merely because their headers are available:
  // a5/TPush.hpp:197 records DIR_BOTH_GM C2V ready at FlagID, but :521 waits at FlagID+1.
  // a2a3/TPush.hpp:219 has a CTRL payload path and :349 a ready wait, while producer
  // allocate/record and consumer free omit CTRL. Its CTRL_SLOT_BUFFER is absent from RingFIFO.
  // Retain their conservative resource union; binding must reject unsupported specializations.
  // Ownership crosses iterations: returning from a push/pop does not release the shared flags.
  // TPipe and TMPipe both have a ready/free pair. Split TPipe and TMPipe paths on A5 may
  // also signal base+16; lowering maps that companion to the same logical allocation.
  auto claim = ParameterSignal("pipe.FlagID", {0, 1}, "pipe");
  entry.claims.push_back(std::move(claim));
  // Only bidirectional TPipe uses this second pair. Binding every rule for a single-direction
  // pipe would incorrectly demand four IDs and reject a legal base at the end of the pool.
  claim = ParameterSignal("pipe.FlagID", {2, 3}, "pipe");
  entry.claims.push_back(std::move(claim));
}

/// Public FIFO macros reuse owner bindings; TPipe/TMPipe themselves need no separate table entry.
void AppendFifo(const std::string& target, std::vector<IsaMacroResources>& entries) {
  for (const auto& name : {"TPUSH", "TPOP", "TFREE", "TALLOC"}) {
    auto entry = IsaMacro("pto::" + std::string(name), target);
    const std::string short_name = name;
    AddPipeOwner(entry);

    // A2/A3 control payload extraction uses literal flags even in Auto and FlagTest builds.
    if (short_name == "TPUSH" && target == "a2a3") {
      // V2C control payload extraction in the TPipe/TMPipe pushVec2Ctrl path.
      auto control = IsaFixedEvent(PipeType::V, PipeType::S);
      entry.claims.push_back(std::move(control));
    }
    entries.push_back(std::move(entry));
  }
}

/// Hard SYNCALL reserves shared IDs; its soft GM counter path reserves no numbered signal.
void AppendSyncAll(const std::string& target, std::vector<IsaMacroResources>& entries) {
  auto entry = IsaMacro("pto::SYNCALL", target);
  // Hard AICOnly and Mix use global FFTS mode-0 AIC synchronization.
  entry.claims.push_back(FixedSignal({11}));
  // Hard AIVOnly uses the separate global AIV synchronization signal.
  entry.claims.push_back(FixedSignal({14}));
  // Hard Mix additionally uses cross-core arrival/release. A5 lowering also emits +16
  // companions for these two logical bases; they never enlarge the allocation domain.
  entry.claims.push_back(FixedSignal({12, 13}));
  entries.push_back(std::move(entry));
}

/// Prefetch's locally paired control flag does not remain occupied until the SDMA completes.
void AppendPrefetch(const std::string& target, std::vector<IsaMacroResources>& entries) {
  auto entry = IsaMacro("pto::TPREFETCH_ASYNC", target);
  // The internally initialized session uses syncId=0; SDMA SetValue publishes its control data.
  // This variant and the external-session variant below must not both be bound unconditionally.
  auto owned = IsaFixedEvent(PipeType::MTE3, PipeType::S);
  owned.modes = {IsaSyncMode::Manual, IsaSyncMode::FlagTest};
  entry.claims.push_back(std::move(owned));
  // A valid external SDMA session supplies its own syncId. SetValue's set/wait completes
  // within this invocation even though that session's ID binding persists across calls.
  auto external = IsaFixedEvent(PipeType::MTE3, PipeType::S);
  external.source = IsaIdSource::Parameter;
  external.ids.clear();
  external.parameter = "session.syncId";
  external.offsets = {0};
  external.owner = "session";
  external.modes = {IsaSyncMode::Manual, IsaSyncMode::FlagTest};
  entry.claims.push_back(std::move(external));
  entries.push_back(std::move(entry));
}

}  // namespace

/// Register only public macros that occupy pipeline events or cross-core signal IDs.
void AppendProtocolIsaMacroResources(const std::string& target, std::vector<IsaMacroResources>& entries) {
  AppendFifo(target, entries);
  AppendSyncAll(target, entries);
  AppendPrefetch(target, entries);
}

}  // namespace pypto::backend

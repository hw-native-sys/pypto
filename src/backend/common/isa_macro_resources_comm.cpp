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
 * @file isa_macro_resources_comm.cpp
 * @brief Hidden event reservations of communication macros.
 *
 * Rules collect signal use across engine variants; a backend adapter must select the actual engine
 * before binding them. Session parameters bind event IDs, but their set/wait pairs complete within
 * each invocation. DMA completion and scalar notification protocols reserve no event IDs here.
 */

#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/pipe.h"
#include "pypto/sync/isa_macro_resources.h"
#include "src/backend/common/isa_macro_resources.h"

namespace pypto::backend {
namespace {

using ir::PipeType;
using sync::IsaMacroResources;

/// Preserve the communication namespace because pto::TGATHER is a distinct tile instruction.
IsaMacroResources CommMacro(const std::string& name, const std::string& target) {
  return IsaMacro("pto::comm::" + name, target);
}

/// Session syncId is one caller-bound value shared by every direction that uses it, never fresh per call.
void AddSessionEvent(IsaMacroResources& entry, PipeType src, PipeType dst) {
  auto claim = IsaFixedEvent(src, dst);
  claim.source = sync::IsaIdSource::Parameter;
  claim.ids.clear();
  claim.parameter = "session.syncId";
  claim.offsets = {0};
  claim.owner = "session";
  // All corresponding set/wait pairs complete in this invocation. Persisting the session binding
  // does not mean a pipeline event remains pending until the asynchronous DMA completes.
  entry.claims.push_back(std::move(claim));
}

/// RDMA with PTO_RDMA_SUPPORTED and the HNS_1825 backend may progress its CQ while posting.
/// Include both read and write control paths for that engine variant.
void AddRdmaEvents(IsaMacroResources& entry) {
  AddSessionEvent(entry, PipeType::S, PipeType::MTE3);
  AddSessionEvent(entry, PipeType::MTE3, PipeType::S);
  AddSessionEvent(entry, PipeType::MTE2, PipeType::S);
}

/// Declare both ping-pong banks; one explicit TPUT/TGET staging tile can also split into two lanes.
void AppendTransfers(const std::string& target, std::vector<IsaMacroResources>& entries) {
  for (const auto& name : {std::string("TPUT"), std::string("TGET")}) {
    auto entry = CommMacro(name, target);
    entry.claims = {IsaFixedEvent(PipeType::MTE2, PipeType::MTE3, {0, 1}),
                    IsaFixedEvent(PipeType::MTE3, PipeType::MTE2, {0, 1})};
    entries.push_back(std::move(entry));
  }
}

/// A5's AIV-stored CCU input path uses one local event before triggering the collective.
void AddCcuCollective(IsaMacroResources& entry) {
  // CollEngine::CCU with CcuInputSource::AivStored; scatter/broadcast use this only on the root.
  auto claim = IsaFixedEvent(PipeType::V, PipeType::MTE3);
  entry.claims.push_back(std::move(claim));
}

/// Collectives use a shared AIV implementation; A5 additionally permits the external CCU engine.
void AppendCollectives(const std::string& target, std::vector<IsaMacroResources>& entries) {
  for (const auto& name :
       {std::string("TGATHER"), std::string("TSCATTER"), std::string("TBROADCAST"), std::string("TREDUCE")}) {
    auto entry = CommMacro(name, target);
    const bool reduce = name == "TREDUCE";
    // CollEngine::AIV includes single-buffer and ping-pong overloads. The CCU variant below
    // contributes different claims; these engine variants must be selected before binding.
    if (reduce) {
      // The one-rank copy path still needs both MTE directions. Reduction ping-pong uses 0 for
      // the root accumulator and 1/2 for remote inputs on MTE2->V, never 1/2 on the MTE3 bank.
      entry.claims = {IsaFixedEvent(PipeType::MTE2, PipeType::MTE3),
                      IsaFixedEvent(PipeType::MTE3, PipeType::MTE2),
                      IsaFixedEvent(PipeType::MTE2, PipeType::V, {0, 1, 2}),
                      IsaFixedEvent(PipeType::V, PipeType::MTE2), IsaFixedEvent(PipeType::V, PipeType::MTE3)};
    } else {
      entry.claims = {IsaFixedEvent(PipeType::MTE2, PipeType::MTE3, {0, 1}),
                      IsaFixedEvent(PipeType::MTE3, PipeType::MTE2, {0, 1})};
    }
    if (reduce || name == "TBROADCAST") {
      // Non-chunked broadcast and rank-one reduce also use PtoSetWaitFlag. Its FlagTest tokens
      // are compiler allocated and cannot be represented by the direct fixed-event union above.
      AddIsaHelperEvent(entry, PipeType::MTE2, PipeType::MTE3);
      AddIsaHelperEvent(entry, PipeType::MTE3, PipeType::MTE2);
    }
    if (target == "a5") {
      AddCcuCollective(entry);
    }
    entries.push_back(std::move(entry));
  }
}

/// Preserve the target-dependent engine union, including batching paths that can implicitly flush.
void AppendAsyncTransfers(const std::string& target, std::vector<IsaMacroResources>& entries) {
  for (const auto& name :
       {std::string("TPUT_ASYNC"), std::string("TGET_ASYNC"), std::string("TPUT_ASYNC_NOTIFY")}) {
    auto entry = CommMacro(name, target);
    const bool get = name == "TGET_ASYNC";
    if (target == "a2a3" || get) {
      // DmaEngine::SDMA posting, persistence and batch flush all use this session event.
      AddSessionEvent(entry, PipeType::MTE3, PipeType::S);
    }
    if (target == "a5") {
      if (!get) {
        // The A5 SDMA PUT variant falls back to a synchronous transfer through MTE2/MTE3.
        AddSessionEvent(entry, PipeType::MTE2, PipeType::MTE3);
        AddSessionEvent(entry, PipeType::MTE3, PipeType::MTE2);
      }
      AddRdmaEvents(entry);
    }
    entries.push_back(std::move(entry));
  }
}

}  // namespace

/// Only communication macros with hidden numbered events belong in the allocation table.
void AppendCommIsaMacroResources(const std::string& target, std::vector<sync::IsaMacroResources>& entries) {
  AppendTransfers(target, entries);
  AppendCollectives(target, entries);
  AppendAsyncTransfers(target, entries);
}

}  // namespace pypto::backend

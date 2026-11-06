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
 * @file isa_macro_resources_npu.cpp
 * @brief Audited synchronization declarations for ordinary A2/A3 and A5 ISA macros.
 *
 * Only macros with signal claims are registered. Rules collect resource use across implementation
 * variants; including a helper header alone does not imply resource use. Manual
 * PtoSetWaitFlag, Auto-mode replacement flags and FlagTest tokens remain separate obligations.
 */

#include <algorithm>
#include <initializer_list>
#include <string>
#include <string_view>
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
using sync::IsaMacroResources;
using sync::IsaSyncMode;

// These macros use signals on both targets. TRECIP intentionally shares TDIVS's implementation.
constexpr const char* kCommonNpuMacros[] = {
    "MGATHER", "MSCATTER", "TDIVS", "TEXTRACT", "TINSERT", "TMRGSORT", "TRECIP",
};

// Their A5 implementations do not use numbered signals and require no resource-table entries.
constexpr const char* kA2A3NpuMacros[] = {
    "TCMPS",   "TCONCAT", "TDEQUANT", "TFILLPAD",   "TFILLPAD_EXPAND", "TFILLPAD_INPLACE",
    "TGATHER", "TPOW",    "TPOWS",    "TROWARGMAX", "TROWARGMIN",      "TROWEXPAND",
    "TROWMAX", "TROWMIN", "TROWPROD", "TROWSUM",    "TSCATTER",        "TSELS",
    "TSHL",    "TSHR",    "TTRANS",
};

/// Keep readable rule groups without exposing a parallel registry or duplicating entry objects.
bool IsOneOf(std::string_view name, std::initializer_list<std::string_view> names) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

/// Scalar, tail, integer and layout branches with literal EVENT_ID0 retain it in the supplied modes.
void AddFixedPair(IsaMacroResources& entry, PipeType src, PipeType dst,
                  std::vector<IsaSyncMode> modes = {IsaSyncMode::Manual, IsaSyncMode::Auto,
                                                    IsaSyncMode::FlagTest},
                  CoreScope core = CoreScope::AIV) {
  auto claim = IsaFixedEvent(src, dst, {0}, core);
  claim.modes = std::move(modes);
  entry.claims.push_back(std::move(claim));
}

/// Scalar accesses to vector storage require both ordered domains, which have independent ID banks.
void AddFixedVectorScalar(IsaMacroResources& entry,
                          std::vector<IsaSyncMode> modes = {IsaSyncMode::Manual, IsaSyncMode::Auto,
                                                            IsaSyncMode::FlagTest}) {
  AddFixedPair(entry, PipeType::V, PipeType::S, modes);
  AddFixedPair(entry, PipeType::S, PipeType::V, std::move(modes));
}

/// Helper-based scalar transitions disappear in Auto and use compiler tokens in FlagTest.
void AddHelperVectorScalar(IsaMacroResources& entry) {
  AddIsaHelperEvent(entry, PipeType::V, PipeType::S);
  AddIsaHelperEvent(entry, PipeType::S, PipeType::V);
}

/// A2/A3 GM gather/scatter use scalar indices plus DMA and scalar fallbacks, each with ID0 helpers.
void AddA2MemoryIndexEvents(IsaMacroResources& entry, std::string_view name) {
  AddIsaHelperEvent(entry, PipeType::V, PipeType::S);
  AddIsaHelperEvent(entry, PipeType::MTE3, PipeType::S);
  AddIsaHelperEvent(entry, PipeType::MTE2, PipeType::S);
  AddIsaHelperEvent(entry, PipeType::S, PipeType::V);
  AddIsaHelperEvent(entry, PipeType::S, PipeType::MTE2);
  AddIsaHelperEvent(entry, PipeType::S, PipeType::MTE3);
  if (name == "MGATHER") {
    AddIsaHelperEvent(entry, PipeType::MTE2, PipeType::V);
    AddIsaHelperEvent(entry, PipeType::MTE2, PipeType::MTE3);
    // The GM-to-L1 overload executes its scalar-to-load transition on AIC instead of AIV.
    AddIsaHelperEvent(entry, PipeType::S, PipeType::MTE2, CoreScope::AIC);
  } else {
    AddIsaHelperEvent(entry, PipeType::MTE3, PipeType::V);
    AddIsaHelperEvent(entry, PipeType::MTE3, PipeType::MTE2);
  }
}

/// A5's SIMT GM-to-L1 gather additionally pairs AIV scratch production with AIC consumption.
void AddA5GatherEvents(IsaMacroResources& entry) {
  AddFixedVectorScalar(entry);
  AddFixedPair(entry, PipeType::S, PipeType::MTE2,
               {IsaSyncMode::Manual, IsaSyncMode::Auto, IsaSyncMode::FlagTest}, CoreScope::AIC);
  sync::IsaResourceClaim signal;
  signal.pool = {sync::ResourceKind::IntraBlockSignal, CoreScope::CoreGroup};
  signal.ids = {2};
  // GatherExec::Simt / Coalesce::Elem / GM-to-L1 uses the internal kSimtSyncId=2.
  // AIV produces scratch and AIC consumes it; both halves share this invocation's signal lease.
  entry.claims.push_back(std::move(signal));
}

/// Register only flags reachable from the macro's implementation, excluding unrelated included overloads.
void AddNpuEvents(IsaMacroResources& entry, std::string_view name, bool a5) {
  // FP aliases and TMOV do not call the scalar Vec-to-Vec helpers in their included TEXTRACT headers.
  if (IsOneOf(name, {"TEXTRACT", "TINSERT"})) {
    AddFixedVectorScalar(entry);
  }
  if (name == "MGATHER") {
    if (a5) {
      AddA5GatherEvents(entry);
    } else {
      AddA2MemoryIndexEvents(entry, name);
    }
    return;
  }
  if (name == "MSCATTER") {
    if (a5) {
      AddFixedVectorScalar(entry);
    } else {
      AddA2MemoryIndexEvents(entry, name);
    }
    return;
  }
  if (name == "TMRGSORT") {
    AddIsaHelperEvent(entry, PipeType::V, PipeType::S);
  }
  if (IsOneOf(name, {"TDIVS", "TRECIP"})) {
    AddIsaHelperEvent(entry, PipeType::V, PipeType::S);
    if (!a5) {
      AddIsaHelperEvent(entry, PipeType::S, PipeType::V);
    }
  }
  if (a5) {
    return;
  }

  // These A2/A3 paths retain explicit ID0 even when automatic synchronization is enabled.
  if (IsOneOf(name,
              {"TDEQUANT", "TROWEXPAND", "TROWMAX", "TROWMIN", "TROWSUM", "TROWARGMAX", "TROWARGMIN"})) {
    AddFixedVectorScalar(entry);
  }
  if (name == "TCONCAT") {
    AddFixedVectorScalar(entry);
    AddFixedPair(entry, PipeType::MTE2, PipeType::S);
    AddFixedPair(entry, PipeType::S, PipeType::MTE3);
  }
  if (name == "TCMPS") {
    AddIsaHelperEvent(entry, PipeType::V, PipeType::S);
    AddIsaHelperEvent(entry, PipeType::MTE1, PipeType::S);
    AddIsaHelperEvent(entry, PipeType::MTE2, PipeType::S);
  }
  if (IsOneOf(name, {"TGATHER", "TPOW", "TPOWS", "TROWARGMAX", "TROWARGMIN", "TROWPROD", "TSCATTER", "TSHL",
                     "TSHR", "TTRANS"})) {
    AddHelperVectorScalar(entry);
  }
  if (name == "TSELS") {
    AddIsaHelperEvent(entry, PipeType::S, PipeType::V);
  }

  // TFILLPAD and TTRANS deliberately substitute explicit ID0 for a helper suppressed in Auto mode.
  if (IsOneOf(name, {"TFILLPAD", "TFILLPAD_INPLACE", "TFILLPAD_EXPAND"})) {
    AddIsaHelperEvent(entry, PipeType::V, PipeType::S);
    AddFixedPair(entry, PipeType::V, PipeType::S, {IsaSyncMode::Auto});
  }
  if (name == "TTRANS") {
    AddFixedVectorScalar(entry, {IsaSyncMode::Auto});
  }
}

}  // namespace

/// Append only ordinary macros whose implementations occupy synchronization signal resources.
void AppendNpuIsaMacroResources(const std::string& target, std::vector<IsaMacroResources>& entries) {
  const bool a5 = target == "a5";
  const auto append = [&](const char* name) {
    auto entry = IsaMacro(std::string("pto::") + name, target);
    AddNpuEvents(entry, name, a5);
    entries.push_back(std::move(entry));
  };
  for (const auto* name : kCommonNpuMacros) {
    append(name);
  }
  if (!a5) {
    for (const auto* name : kA2A3NpuMacros) {
      append(name);
    }
  }
}

}  // namespace pypto::backend

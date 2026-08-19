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

#include "pypto/ir/verifier/diagnostic_check_registry.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

std::string DiagnosticCheckToString(DiagnosticCheck check) {
  switch (check) {
    case DiagnosticCheck::UnusedVariable:
      return "UnusedVariable";
    case DiagnosticCheck::UnusedControlFlowResult:
      return "UnusedControlFlowResult";
    case DiagnosticCheck::TileInnermostDimGranularity:
      return "TileInnermostDimGranularity";
    case DiagnosticCheck::OutParamWriteDropped:
      return "OutParamWriteDropped";
    case DiagnosticCheck::ScalarWriteLineShared:
      return "ScalarWriteLineShared";
    case DiagnosticCheck::InParamWritten:
      return "InParamWritten";
    default:
      return "Unknown";
  }
}

std::vector<DiagnosticCheck> DiagnosticCheckSet::ToVector() const {
  std::vector<DiagnosticCheck> result;
  for (uint32_t i = 0; i < static_cast<uint32_t>(DiagnosticCheck::kCount); ++i) {
    auto check = static_cast<DiagnosticCheck>(i);
    if (Contains(check)) {
      result.push_back(check);
    }
  }
  return result;
}

std::string DiagnosticCheckSet::ToString() const {
  auto checks = ToVector();
  if (checks.empty()) {
    return "{}";
  }

  std::ostringstream oss;
  oss << "{";
  for (size_t i = 0; i < checks.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << DiagnosticCheckToString(checks[i]);
  }
  oss << "}";
  return oss.str();
}

DiagnosticCheckRegistry& DiagnosticCheckRegistry::GetInstance() {
  static DiagnosticCheckRegistry instance;
  return instance;
}

DiagnosticCheckRegistry::DiagnosticCheckRegistry() {
  // Warnings — run before the first pass by default.
  Register(DiagnosticCheck::UnusedVariable, DiagnosticSeverity::Warning, DiagnosticPhase::PrePipeline,
           /*hint_code=*/"", CreateUnusedVariableWarningVerifier);
  Register(DiagnosticCheck::UnusedControlFlowResult, DiagnosticSeverity::Warning,
           DiagnosticPhase::PrePipeline,
           /*hint_code=*/"", CreateUnusedControlFlowResultWarningVerifier);
  // Issue #2352 — runs on the pipeline input so the message names the
  // parameter the user wrote, before ConvertToSSA versions it to `out__ssa_v2`.
  Register(DiagnosticCheck::OutParamWriteDropped, DiagnosticSeverity::Warning, DiagnosticPhase::PrePipeline,
           /*hint_code=*/"", CreateOutParamWriteDroppedWarningVerifier);
  // Runs on the pipeline input: the instance-multiplying scopes are still the
  // user's own, and the index expressions are the ones they wrote, before SSA
  // versioning and before ConvertTensorToTileOps rewrites the surrounding ops.
  Register(DiagnosticCheck::ScalarWriteLineShared, DiagnosticSeverity::Warning, DiagnosticPhase::PrePipeline,
           /*hint_code=*/"", CreateScalarWriteLineSharedWarningVerifier);
  // Runs on the finished program, not after any one pass: a wrapper's signature
  // legitimately lags its inner kernel's until DeriveCallDirections materialises
  // the effective directions, so the invariant only holds once the pipeline is
  // done. A warning, and deliberately not an `IRProperty`: by the time it runs,
  // `InitMemRef` has invalidated `SSAForm` and nothing re-establishes it, so its
  // buffer lineage is best-effort across control flow. There is no pipeline
  // position that is both after `DeriveCallDirections` (pass 37) and still in
  // SSA (invalidated at pass 31), so this is a property of the placement, not a
  // gap to close later.
  Register(DiagnosticCheck::InParamWritten, DiagnosticSeverity::Warning, DiagnosticPhase::PostPipeline,
           /*hint_code=*/"", CreateInParamWrittenWarningVerifier);

  // Performance hints (issue #1180) — run once at the end of the pipeline,
  // after tile shapes and memory layout are fully resolved.
  Register(DiagnosticCheck::TileInnermostDimGranularity, DiagnosticSeverity::PerfHint,
           DiagnosticPhase::PostPipeline, "PH001", CreateTileInnermostDimGranularityVerifier);
}

void DiagnosticCheckRegistry::Register(DiagnosticCheck check, DiagnosticSeverity severity,
                                       DiagnosticPhase phase, std::string hint_code,
                                       std::function<PropertyVerifierPtr()> factory) {
  entries_[static_cast<uint32_t>(check)] = Entry{severity, phase, std::move(hint_code), std::move(factory)};
}

PropertyVerifierPtr DiagnosticCheckRegistry::GetVerifier(DiagnosticCheck check) const {
  auto it = entries_.find(static_cast<uint32_t>(check));
  if (it == entries_.end()) {
    return nullptr;
  }
  return it->second.factory();
}

const DiagnosticCheckRegistry::Entry* DiagnosticCheckRegistry::GetEntry(DiagnosticCheck check) const {
  auto it = entries_.find(static_cast<uint32_t>(check));
  if (it == entries_.end()) {
    return nullptr;
  }
  return &it->second;
}

std::vector<Diagnostic> DiagnosticCheckRegistry::RunChecks(const DiagnosticCheckSet& checks,
                                                           DiagnosticPhase phase,
                                                           const ProgramPtr& program) const {
  std::vector<Diagnostic> all_diagnostics;
  if (!program) {
    return all_diagnostics;
  }

  for (auto check : checks.ToVector()) {
    auto it = entries_.find(static_cast<uint32_t>(check));
    if (it == entries_.end()) continue;
    if (it->second.phase != phase) continue;

    auto verifier = it->second.factory();
    if (!verifier) continue;

    std::size_t before = all_diagnostics.size();
    verifier->Verify(program, all_diagnostics);

    // Stamp severity / hint_code from the registration unconditionally — the
    // registry is the single source of truth, so a verifier that emits a
    // mismatching severity or hint code (e.g. a Warning verifier accidentally
    // setting "PH001") is overwritten with the registered values.
    for (std::size_t i = before; i < all_diagnostics.size(); ++i) {
      all_diagnostics[i].severity = it->second.severity;
      all_diagnostics[i].hint_code = it->second.hint_code;
    }
  }
  return all_diagnostics;
}

DiagnosticCheckSet DiagnosticCheckRegistry::GetAllChecks() {
  DiagnosticCheckSet all;
  const auto& entries = GetInstance().entries_;
  for (uint32_t i = 0; i < static_cast<uint32_t>(DiagnosticCheck::kCount); ++i) {
    if (entries.find(i) != entries.end()) {
      all.Insert(static_cast<DiagnosticCheck>(i));
    }
  }
  return all;
}

DiagnosticCheckSet DiagnosticCheckRegistry::GetWarningChecks() {
  DiagnosticCheckSet result;
  for (const auto& [bit, entry] : GetInstance().entries_) {
    if (entry.severity == DiagnosticSeverity::Warning) {
      result.Insert(static_cast<DiagnosticCheck>(bit));
    }
  }
  return result;
}

DiagnosticCheckSet DiagnosticCheckRegistry::GetPerfHintChecks() {
  DiagnosticCheckSet result;
  for (const auto& [bit, entry] : GetInstance().entries_) {
    if (entry.severity == DiagnosticSeverity::PerfHint) {
      result.Insert(static_cast<DiagnosticCheck>(bit));
    }
  }
  return result;
}

}  // namespace ir
}  // namespace pypto

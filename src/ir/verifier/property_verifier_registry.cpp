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

#include "pypto/ir/verifier/property_verifier_registry.h"

#include <algorithm>
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

// ============================================================================
// PropertyVerifierRegistry implementation
// ============================================================================

PropertyVerifierRegistry& PropertyVerifierRegistry::GetInstance() {
  static PropertyVerifierRegistry instance;
  return instance;
}

PropertyVerifierRegistry::PropertyVerifierRegistry() {
  // Register all built-in property verifiers
  Register(IRProperty::SSAForm, CreateSSAPropertyVerifier);
  Register(IRProperty::TypeChecked, CreateTypeCheckPropertyVerifier);
  Register(IRProperty::NoNestedCalls, CreateNoNestedCallPropertyVerifier);
  Register(IRProperty::NormalizedStmtStructure, CreateNormalizedStmtPropertyVerifier);
  Register(IRProperty::NoRedundantBlocks, CreateNoRedundantBlocksPropertyVerifier);
  Register(IRProperty::SplitIncoreOrch, CreateSplitIncoreOrchPropertyVerifier);
  Register(IRProperty::ClusterOutlined, CreateClusterOutlinedPropertyVerifier);
  Register(IRProperty::HierarchyOutlined, CreateHierarchyOutlinedPropertyVerifier);
  Register(IRProperty::HasMemRefs, CreateHasMemRefsPropertyVerifier);
  Register(IRProperty::IncoreTileOps, CreateIncoreTileOpsPropertyVerifier);
  Register(IRProperty::MixedKernelExpanded, CreateMixedKernelExpandedPropertyVerifier);
  Register(IRProperty::AllocatedMemoryAddr, CreateAllocatedMemoryAddrPropertyVerifier);
  Register(IRProperty::TileOps2D, CreateTileOps2DPropertyVerifier);
  Register(IRProperty::TileMemoryInferred, CreateTileMemoryInferredPropertyVerifier);
  Register(IRProperty::BreakContinueValid, CreateBreakContinuePropertyVerifier);
  Register(IRProperty::UseAfterDef, CreateUseAfterDefPropertyVerifier);
  Register(IRProperty::StructuredCtrlFlow, CreateStructuredCtrlFlowPropertyVerifier);
  Register(IRProperty::OutParamNotShadowed, CreateOutParamNotShadowedPropertyVerifier);
  Register(IRProperty::NoNestedInCore, CreateNoNestedIncorePropertyVerifier);
  Register(IRProperty::InOutUseValid, CreateInOutUseValidPropertyVerifier);
  Register(IRProperty::PipelineLoopValid, CreatePipelineLoopValidPropertyVerifier);
  Register(IRProperty::PipelineResolved, CreatePipelineResolvedPropertyVerifier);
  Register(IRProperty::CallDirectionsResolved, CreateCallDirectionsResolvedPropertyVerifier);
  Register(IRProperty::TileTypeCoherence, CreateTileTypeCoherencePropertyVerifier);
  Register(IRProperty::InlineFunctionsEliminated, CreateInlineFunctionsEliminatedPropertyVerifier);
  Register(IRProperty::OrchestrationReferencesResolved,
           CreateOrchestrationReferencesResolvedPropertyVerifier);
}

void PropertyVerifierRegistry::Register(IRProperty prop, std::function<PropertyVerifierPtr()> factory) {
  factories_[static_cast<uint32_t>(prop)] = std::move(factory);
}

PropertyVerifierPtr PropertyVerifierRegistry::GetVerifier(IRProperty prop) const {
  auto it = factories_.find(static_cast<uint32_t>(prop));
  if (it == factories_.end()) {
    return nullptr;
  }
  return it->second();
}

bool PropertyVerifierRegistry::HasVerifier(IRProperty prop) const {
  return factories_.count(static_cast<uint32_t>(prop)) > 0;
}

std::vector<Diagnostic> PropertyVerifierRegistry::VerifyProperties(const IRPropertySet& properties,
                                                                   const ProgramPtr& program) const {
  std::vector<Diagnostic> all_diagnostics;
  if (!program) {
    return all_diagnostics;
  }

  for (auto prop : properties.ToVector()) {
    auto verifier = GetVerifier(prop);
    if (verifier) {
      verifier->Verify(program, all_diagnostics);
    }
  }
  return all_diagnostics;
}

void PropertyVerifierRegistry::VerifyOrThrow(const IRPropertySet& properties,
                                             const ProgramPtr& program) const {
  auto diagnostics = VerifyProperties(properties, program);
  bool has_errors = std::any_of(diagnostics.begin(), diagnostics.end(),
                                [](const Diagnostic& d) { return d.severity == DiagnosticSeverity::Error; });
  if (has_errors) {
    std::string report = GenerateReport(diagnostics);
    throw VerificationError(report, std::move(diagnostics));
  }
}

std::string PropertyVerifierRegistry::GenerateReport(const std::vector<Diagnostic>& diagnostics) {
  std::ostringstream oss;

  size_t error_count = 0;
  size_t warning_count = 0;
  for (const auto& d : diagnostics) {
    if (d.severity == DiagnosticSeverity::Error) {
      error_count++;
    } else {
      warning_count++;
    }
  }

  oss << "IR Verification Report\n";
  oss << "======================\n";
  oss << "Total diagnostics: " << diagnostics.size() << " (";
  oss << error_count << " errors, " << warning_count << " warnings)\n\n";

  if (diagnostics.empty()) {
    oss << "Status: PASSED\n";
    return oss.str();
  }

  for (size_t i = 0; i < diagnostics.size(); ++i) {
    const auto& d = diagnostics[i];
    std::string severity_str = (d.severity == DiagnosticSeverity::Error) ? "ERROR" : "WARNING";
    oss << "[" << (i + 1) << "] " << severity_str << " - " << d.rule_name << "\n";
    oss << "  Message: " << d.message << "\n";
    oss << "  Location: " << d.span.filename_ << ":" << d.span.begin_line_ << ":" << d.span.begin_column_
        << "\n";
    oss << "  Error Code: " << d.error_code << "\n";
    oss << "\n";
  }

  if (error_count > 0) {
    oss << "Status: FAILED (" << error_count << " error(s) found)\n";
  } else {
    oss << "Status: PASSED with " << warning_count << " warning(s)\n";
  }

  return oss.str();
}

}  // namespace ir
}  // namespace pypto

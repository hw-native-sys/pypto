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

#include "pypto/ir/transforms/ir_property.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace pypto {
namespace ir {

std::string IRPropertyToString(IRProperty prop) {
  switch (prop) {
    case IRProperty::SSAForm:
      return "SSAForm";
    case IRProperty::TypeChecked:
      return "TypeChecked";
    case IRProperty::NoNestedCalls:
      return "NoNestedCalls";
    case IRProperty::NormalizedStmtStructure:
      return "NormalizedStmtStructure";
    case IRProperty::NoRedundantBlocks:
      return "NoRedundantBlocks";
    case IRProperty::SplitIncoreOrch:
      return "SplitIncoreOrch";
    case IRProperty::HasMemRefs:
      return "HasMemRefs";
    case IRProperty::IncoreTileOps:
      return "IncoreTileOps";
    case IRProperty::AllocatedMemoryAddr:
      return "AllocatedMemoryAddr";
    case IRProperty::MixedKernelExpanded:
      return "MixedKernelExpanded";
    case IRProperty::ClusterOutlined:
      return "ClusterOutlined";
    case IRProperty::TileOps2D:
      return "TileOps2D";
    case IRProperty::TileMemoryInferred:
      return "TileMemoryInferred";
    case IRProperty::BreakContinueValid:
      return "BreakContinueValid";
    case IRProperty::UseAfterDef:
      return "UseAfterDef";
    case IRProperty::HierarchyOutlined:
      return "HierarchyOutlined";
    case IRProperty::StructuredCtrlFlow:
      return "StructuredCtrlFlow";
    case IRProperty::VectorKernelSplit:
      return "VectorKernelSplit";
    case IRProperty::OutParamNotShadowed:
      return "OutParamNotShadowed";
    case IRProperty::NoNestedInCore:
      return "NoNestedInCore";
    case IRProperty::InOutUseValid:
      return "InOutUseValid";
    case IRProperty::PipelineLoopValid:
      return "PipelineLoopValid";
    case IRProperty::PipelineResolved:
      return "PipelineResolved";
    case IRProperty::CallDirectionsResolved:
      return "CallDirectionsResolved";
    case IRProperty::TileTypeCoherence:
      return "TileTypeCoherence";
    case IRProperty::InlineFunctionsEliminated:
      return "InlineFunctionsEliminated";
    default:
      return "Unknown";
  }
}

std::vector<IRProperty> IRPropertySet::ToVector() const {
  std::vector<IRProperty> result;
  for (uint32_t i = 0; i < static_cast<uint32_t>(IRProperty::kCount); ++i) {
    auto prop = static_cast<IRProperty>(i);
    if (Contains(prop)) {
      result.push_back(prop);
    }
  }
  return result;
}

std::string IRPropertySet::ToString() const {
  auto props = ToVector();
  if (props.empty()) {
    return "{}";
  }

  std::ostringstream oss;
  oss << "{";
  for (size_t i = 0; i < props.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << IRPropertyToString(props[i]);
  }
  oss << "}";
  return oss.str();
}

const IRPropertySet& GetVerifiedProperties() {
  static const IRPropertySet props{IRProperty::SSAForm,
                                   IRProperty::TypeChecked,
                                   IRProperty::MixedKernelExpanded,
                                   IRProperty::AllocatedMemoryAddr,
                                   IRProperty::BreakContinueValid,
                                   IRProperty::NoRedundantBlocks,
                                   IRProperty::InOutUseValid,
                                   IRProperty::CallDirectionsResolved};
  return props;
}

VerificationLevel GetDefaultVerificationLevel() {
  // C++17: static local initialization is thread-safe and runs exactly once
  static const VerificationLevel level = [] {
    const char* env = std::getenv("PYPTO_VERIFY_LEVEL");
    if (env == nullptr) {
      return VerificationLevel::Basic;
    }
    std::string val(env);
    if (val == "none") {
      return VerificationLevel::None;
    }
    if (val == "roundtrip") {
      return VerificationLevel::Roundtrip;
    }
    return VerificationLevel::Basic;
  }();
  return level;
}

const IRPropertySet& GetStructuralProperties() {
  static const IRPropertySet props{IRProperty::TypeChecked,         IRProperty::BreakContinueValid,
                                   IRProperty::NoRedundantBlocks,   IRProperty::UseAfterDef,
                                   IRProperty::OutParamNotShadowed, IRProperty::NoNestedInCore,
                                   IRProperty::InOutUseValid,       IRProperty::PipelineLoopValid};
  return props;
}

const IRPropertySet& GetDefaultVerifyProperties() {
  static const IRPropertySet props{IRProperty::SSAForm,
                                   IRProperty::TypeChecked,
                                   IRProperty::NoNestedCalls,
                                   IRProperty::BreakContinueValid,
                                   IRProperty::NoRedundantBlocks,
                                   IRProperty::UseAfterDef,
                                   IRProperty::OutParamNotShadowed,
                                   IRProperty::NoNestedInCore,
                                   IRProperty::TileTypeCoherence};
  return props;
}

DiagnosticPhase GetDefaultDiagnosticPhase() {
  static const DiagnosticPhase phase = [] {
    const char* env = std::getenv("PYPTO_WARNING_LEVEL");
    if (env == nullptr) {
      return DiagnosticPhase::PrePipeline;
    }
    std::string val(env);
    if (val == "none") {
      return DiagnosticPhase::None;
    }
    if (val == "post_pass") {
      return DiagnosticPhase::PostPass;
    }
    if (val == "post_pipeline") {
      return DiagnosticPhase::PostPipeline;
    }
    return DiagnosticPhase::PrePipeline;
  }();
  return phase;
}

}  // namespace ir
}  // namespace pypto

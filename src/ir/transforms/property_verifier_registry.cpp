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

#include "pypto/ir/transforms/property_verifier_registry.h"

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/verifier.h"

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
  Register(IRProperty::FlattenedSingleStmt, CreateFlattenedSingleStmtPropertyVerifier);
  Register(IRProperty::SplitIncoreOrch, CreateSplitIncoreOrchPropertyVerifier);
  Register(IRProperty::HasMemRefs, CreateHasMemRefsPropertyVerifier);
  Register(IRProperty::IncoreBlockOps, CreateIncoreBlockOpsPropertyVerifier);
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

}  // namespace ir
}  // namespace pypto

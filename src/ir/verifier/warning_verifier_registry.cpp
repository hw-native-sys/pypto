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

#include "pypto/ir/verifier/warning_verifier_registry.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

std::string WarningCheckToString(WarningCheck check) {
  switch (check) {
    case WarningCheck::UnusedVariable:
      return "UnusedVariable";
    default:
      return "Unknown";
  }
}

std::vector<WarningCheck> WarningCheckSet::ToVector() const {
  std::vector<WarningCheck> result;
  for (uint32_t i = 0; i < static_cast<uint32_t>(WarningCheck::kCount); ++i) {
    auto check = static_cast<WarningCheck>(i);
    if (Contains(check)) {
      result.push_back(check);
    }
  }
  return result;
}

std::string WarningCheckSet::ToString() const {
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
    oss << WarningCheckToString(checks[i]);
  }
  oss << "}";
  return oss.str();
}

WarningVerifierRegistry& WarningVerifierRegistry::GetInstance() {
  static WarningVerifierRegistry instance;
  return instance;
}

WarningVerifierRegistry::WarningVerifierRegistry() {
  // Register all built-in warning verifiers
  Register(WarningCheck::UnusedVariable, CreateUnusedVariableWarningVerifier);
}

void WarningVerifierRegistry::Register(WarningCheck check, std::function<PropertyVerifierPtr()> factory) {
  factories_[static_cast<uint32_t>(check)] = std::move(factory);
}

PropertyVerifierPtr WarningVerifierRegistry::GetVerifier(WarningCheck check) const {
  auto it = factories_.find(static_cast<uint32_t>(check));
  if (it == factories_.end()) {
    return nullptr;
  }
  return it->second();
}

std::vector<Diagnostic> WarningVerifierRegistry::RunChecks(const WarningCheckSet& checks,
                                                           const ProgramPtr& program) const {
  std::vector<Diagnostic> all_diagnostics;
  if (!program) {
    return all_diagnostics;
  }

  for (auto check : checks.ToVector()) {
    auto verifier = GetVerifier(check);
    if (verifier) {
      verifier->Verify(program, all_diagnostics);
    }
  }
  return all_diagnostics;
}

WarningCheckSet WarningVerifierRegistry::GetAllChecks() {
  WarningCheckSet all;
  for (uint32_t i = 0; i < static_cast<uint32_t>(WarningCheck::kCount); ++i) {
    all.Insert(static_cast<WarningCheck>(i));
  }
  return all;
}

}  // namespace ir
}  // namespace pypto

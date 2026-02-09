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

#include "pypto/ir/transforms/verifier.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"

namespace pypto {
namespace ir {

// Forward declarations of built-in rules (implemented in their respective files)
class SSAVerifyRule;
class TypeCheckRule;

// External rule instances - these are defined in verify_ssa_pass.cpp and type_check_pass.cpp
extern VerifyRulePtr CreateSSAVerifyRule();
extern VerifyRulePtr CreateTypeCheckRule();
extern VerifyRulePtr CreateNoNestedCallVerifyRule();

IRVerifier::IRVerifier() = default;

void IRVerifier::AddRule(VerifyRulePtr rule) {
  if (!rule) {
    return;
  }

  // Check if rule with same name already exists
  auto it = std::find_if(rules_.begin(), rules_.end(),
                         [&rule](const VerifyRulePtr& r) { return r->GetName() == rule->GetName(); });

  if (it == rules_.end()) {
    rules_.push_back(rule);
  }
}

void IRVerifier::EnableRule(const std::string& name) { disabled_rules_.erase(name); }

void IRVerifier::DisableRule(const std::string& name) { disabled_rules_.insert(name); }

bool IRVerifier::IsRuleEnabled(const std::string& name) const { return disabled_rules_.count(name) == 0; }

std::vector<Diagnostic> IRVerifier::Verify(const ProgramPtr& program) const {
  if (!program) {
    return {};
  }

  std::vector<Diagnostic> all_diagnostics;

  // Run all enabled rules on all functions
  // program->functions_ is a map from GlobalVar to Function
  for (const auto& [global_var, func] : program->functions_) {
    if (!func) {
      continue;
    }

    for (const auto& rule : rules_) {
      if (!rule) {
        continue;
      }

      // Skip disabled rules
      if (!IsRuleEnabled(rule->GetName())) {
        continue;
      }

      // Run the rule
      rule->Verify(func, all_diagnostics);
    }
  }

  return all_diagnostics;
}

void IRVerifier::VerifyOrThrow(const ProgramPtr& program) const {
  auto diagnostics = Verify(program);

  // Check if there are any errors (not just warnings)
  bool has_errors = std::any_of(diagnostics.begin(), diagnostics.end(),
                                [](const Diagnostic& d) { return d.severity == DiagnosticSeverity::Error; });

  if (has_errors) {
    std::string report = GenerateReport(diagnostics);
    throw VerificationError(report, std::move(diagnostics));
  }
}

std::string IRVerifier::GenerateReport(const std::vector<Diagnostic>& diagnostics) {
  std::ostringstream oss;

  // Count errors and warnings
  size_t error_count = 0;
  size_t warning_count = 0;
  for (const auto& d : diagnostics) {
    if (d.severity == DiagnosticSeverity::Error) {
      error_count++;
    } else {
      warning_count++;
    }
  }

  // Header
  oss << "IR Verification Report\n";
  oss << "======================\n";
  oss << "Total diagnostics: " << diagnostics.size() << " (";
  oss << error_count << " errors, " << warning_count << " warnings)\n\n";

  if (diagnostics.empty()) {
    oss << "Status: PASSED\n";
    return oss.str();
  }

  // List all diagnostics
  for (size_t i = 0; i < diagnostics.size(); ++i) {
    const auto& d = diagnostics[i];

    // Severity label
    std::string severity_str = (d.severity == DiagnosticSeverity::Error) ? "ERROR" : "WARNING";

    oss << "[" << (i + 1) << "] " << severity_str << " - " << d.rule_name << "\n";
    oss << "  Message: " << d.message << "\n";
    oss << "  Location: " << d.span.filename_ << ":" << d.span.begin_line_ << ":" << d.span.begin_column_
        << "\n";
    oss << "  Error Code: " << d.error_code << "\n";
    oss << "\n";
  }

  // Summary
  if (error_count > 0) {
    oss << "Status: FAILED (" << error_count << " error(s) found)\n";
  } else {
    oss << "Status: PASSED with " << warning_count << " warning(s)\n";
  }

  return oss.str();
}

IRVerifier IRVerifier::CreateDefault() {
  IRVerifier verifier;

  // Add built-in verification rules
  verifier.AddRule(CreateSSAVerifyRule());
  verifier.AddRule(CreateTypeCheckRule());
  verifier.AddRule(CreateNoNestedCallVerifyRule());

  return verifier;
}

}  // namespace ir
}  // namespace pypto

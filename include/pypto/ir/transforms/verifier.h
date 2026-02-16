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

#ifndef PYPTO_IR_TRANSFORMS_VERIFIER_H_
#define PYPTO_IR_TRANSFORMS_VERIFIER_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for verification rules
 *
 * Each verification rule implements a specific check on IR functions.
 * Rules can detect errors or warnings and add them to a diagnostics vector.
 *
 * To create a new verification rule:
 * 1. Inherit from VerifyRule
 * 2. Implement GetName() to return a unique rule name
 * 3. Implement Verify() to perform the verification logic
 *
 * Example:
 * @code
 *   class MyCustomRule : public VerifyRule {
 *    public:
 *     std::string GetName() const override { return "MyCustomRule"; }
 *     void Verify(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) override {
 *       // Verification logic
 *     }
 *   };
 * @endcode
 */
class VerifyRule {
 public:
  virtual ~VerifyRule() = default;

  /**
   * @brief Get the name of this verification rule
   * @return Unique name for this rule (e.g., "SSAVerify", "TypeCheck")
   */
  [[nodiscard]] virtual std::string GetName() const = 0;

  /**
   * @brief Verify a function and collect diagnostics
   * @param func Function to verify
   * @param diagnostics Vector to append diagnostics to
   *
   * This method should examine the function and add any detected issues
   * to the diagnostics vector. It should not throw exceptions - all issues
   * should be reported through diagnostics.
   */
  virtual void Verify(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) = 0;
};

/// Shared pointer to a verification rule
using VerifyRulePtr = std::shared_ptr<VerifyRule>;

/**
 * @brief Factory function for creating SSA verification rule
 * @return Shared pointer to SSAVerifyRule
 */
VerifyRulePtr CreateSSAVerifyRule();

/**
 * @brief Factory function for creating type check verification rule
 * @return Shared pointer to TypeCheckRule
 */
VerifyRulePtr CreateTypeCheckRule();

/**
 * @brief Factory function for creating no nested call verification rule
 * @return Shared pointer to NoNestedCallVerifyRule
 */
VerifyRulePtr CreateNoNestedCallVerifyRule();

/**
 * @brief IR verification system
 *
 * IRVerifier manages a collection of verification rules and applies them to programs.
 * Rules can be enabled/disabled individually, and the verifier can operate in two modes:
 * - Verify(): Collects all diagnostics without throwing
 * - VerifyOrThrow(): Collects diagnostics and throws if errors are found
 *
 * Usage:
 * @code
 *   // Create default verifier with all built-in rules
 *   auto verifier = IRVerifier::CreateDefault();
 *
 *   // Disable specific rules
 *   verifier.DisableRule("TypeCheck");
 *
 *   // Run verification
 *   auto diagnostics = verifier.Verify(program);
 *   for (const auto& d : diagnostics) {
 *     if (d.severity == DiagnosticSeverity::Error) {
 *       LOG_ERROR << d.message;
 *     }
 *   }
 *
 *   // Or throw on errors
 *   verifier.VerifyOrThrow(program);
 * @endcode
 */
class IRVerifier {
 public:
  /**
   * @brief Construct an empty verifier with no rules
   */
  IRVerifier();

  /**
   * @brief Add a verification rule to this verifier
   * @param rule Shared pointer to the rule to add
   *
   * Rules are executed in the order they are added.
   * If a rule with the same name already exists, it will not be added again.
   */
  void AddRule(VerifyRulePtr rule);

  /**
   * @brief Enable a previously disabled rule
   * @param name Name of the rule to enable
   *
   * If the rule is not found or is already enabled, this is a no-op.
   */
  void EnableRule(const std::string& name);

  /**
   * @brief Disable a rule
   * @param name Name of the rule to disable
   *
   * Disabled rules will be skipped during verification.
   */
  void DisableRule(const std::string& name);

  /**
   * @brief Check if a rule is currently enabled
   * @param name Name of the rule to check
   * @return true if the rule is enabled, false if disabled or not found
   */
  [[nodiscard]] bool IsRuleEnabled(const std::string& name) const;

  /**
   * @brief Verify a program and collect diagnostics
   * @param program Program to verify
   * @return Vector of all diagnostics (errors and warnings)
   *
   * This method runs all enabled rules on all functions in the program
   * and collects diagnostics. It does not throw exceptions even if errors
   * are found - use VerifyOrThrow() if you want exception-based error handling.
   */
  [[nodiscard]] std::vector<Diagnostic> Verify(const ProgramPtr& program) const;

  /**
   * @brief Verify a program and throw on errors
   * @param program Program to verify
   * @throws VerificationError if any errors are found
   *
   * This method runs verification and throws a VerificationError if any
   * diagnostics with severity Error are found. Warnings do not cause an exception.
   */
  void VerifyOrThrow(const ProgramPtr& program) const;

  /**
   * @brief Generate a formatted report from diagnostics
   * @param diagnostics Vector of diagnostics to format
   * @return Formatted report string
   *
   * The report includes:
   * - Total count of errors and warnings
   * - Details for each diagnostic (severity, rule name, message, location)
   * - Overall verification status
   */
  static std::string GenerateReport(const std::vector<Diagnostic>& diagnostics);

  /**
   * @brief Create a verifier with default built-in rules
   * @return IRVerifier with SSAVerify and TypeCheck rules
   *
   * This factory method creates a verifier pre-configured with all
   * standard verification rules enabled.
   */
  static IRVerifier CreateDefault();

 private:
  std::vector<VerifyRulePtr> rules_;                ///< All registered verification rules
  std::unordered_set<std::string> disabled_rules_;  ///< Names of disabled rules
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_VERIFIER_H_

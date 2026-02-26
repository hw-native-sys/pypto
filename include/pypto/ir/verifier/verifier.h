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

#ifndef PYPTO_IR_VERIFIER_VERIFIER_H_
#define PYPTO_IR_VERIFIER_VERIFIER_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for IR property verifiers
 *
 * Each verifier implements a specific check on IR programs.
 * Verifiers can detect errors or warnings and add them to a diagnostics vector.
 * Each verifier receives a ProgramPtr and internally decides whether to iterate
 * over functions or check program-level properties.
 *
 * To create a new property verifier:
 * 1. Inherit from PropertyVerifier
 * 2. Implement GetName() to return a unique name
 * 3. Implement Verify() to perform the verification logic
 *
 * Example:
 * @code
 *   class MyVerifier : public PropertyVerifier {
 *    public:
 *     std::string GetName() const override { return "MyVerifier"; }
 *     void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
 *       for (const auto& [gv, func] : program->functions_) {
 *         // Verification logic per function
 *       }
 *     }
 *   };
 * @endcode
 */
class PropertyVerifier {
 public:
  virtual ~PropertyVerifier() = default;

  /**
   * @brief Get the name of this verifier
   * @return Unique name (e.g., "SSAVerify", "TypeCheck")
   */
  [[nodiscard]] virtual std::string GetName() const = 0;

  /**
   * @brief Verify a program and collect diagnostics
   * @param program Program to verify
   * @param diagnostics Vector to append diagnostics to
   *
   * This method should examine the program and add any detected issues
   * to the diagnostics vector. It should not throw exceptions - all issues
   * should be reported through diagnostics.
   */
  virtual void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) = 0;
};

/// Shared pointer to a property verifier
using PropertyVerifierPtr = std::shared_ptr<PropertyVerifier>;

// Backward compatibility aliases
using VerifyRule = PropertyVerifier;
using VerifyRulePtr = PropertyVerifierPtr;

/**
 * @brief Factory function for creating SSA property verifier
 * @return Shared pointer to SSA PropertyVerifier
 */
PropertyVerifierPtr CreateSSAPropertyVerifier();

/**
 * @brief Factory function for creating type check property verifier
 * @return Shared pointer to TypeCheck PropertyVerifier
 */
PropertyVerifierPtr CreateTypeCheckPropertyVerifier();

/**
 * @brief Factory function for creating no nested call property verifier
 * @return Shared pointer to NoNestedCall PropertyVerifier
 */
PropertyVerifierPtr CreateNoNestedCallPropertyVerifier();

/**
 * @brief Factory function for creating NormalizedStmtStructure property verifier
 * @return Shared pointer to NormalizedStmtStructure PropertyVerifier
 */
PropertyVerifierPtr CreateNormalizedStmtPropertyVerifier();

/**
 * @brief Factory function for creating FlattenedSingleStmt property verifier
 * @return Shared pointer to FlattenedSingleStmt PropertyVerifier
 */
PropertyVerifierPtr CreateFlattenedSingleStmtPropertyVerifier();

/**
 * @brief Factory function for creating SplitIncoreOrch property verifier
 * @return Shared pointer to SplitIncoreOrch PropertyVerifier
 */
PropertyVerifierPtr CreateSplitIncoreOrchPropertyVerifier();

/**
 * @brief Factory function for creating HasMemRefs property verifier
 * @return Shared pointer to HasMemRefs PropertyVerifier
 */
PropertyVerifierPtr CreateHasMemRefsPropertyVerifier();

/**
 * @brief Factory function for creating IncoreBlockOps property verifier
 * @return Shared pointer to IncoreBlockOps PropertyVerifier
 */
PropertyVerifierPtr CreateIncoreBlockOpsPropertyVerifier();

// Backward compatibility aliases for factory functions
inline VerifyRulePtr CreateSSAVerifyRule() { return CreateSSAPropertyVerifier(); }
inline VerifyRulePtr CreateTypeCheckRule() { return CreateTypeCheckPropertyVerifier(); }
inline VerifyRulePtr CreateNoNestedCallVerifyRule() { return CreateNoNestedCallPropertyVerifier(); }

/**
 * @brief IR verification system
 *
 * IRVerifier manages a collection of property verifiers and applies them to programs.
 * Verifiers can be enabled/disabled individually, and the verifier can operate in two modes:
 * - Verify(): Collects all diagnostics without throwing
 * - VerifyOrThrow(): Collects diagnostics and throws if errors are found
 *
 * Usage:
 * @code
 *   auto verifier = IRVerifier::CreateDefault();
 *   verifier.DisableRule("TypeCheck");
 *   auto diagnostics = verifier.Verify(program);
 *   verifier.VerifyOrThrow(program);
 * @endcode
 */
class IRVerifier {
 public:
  IRVerifier();

  /**
   * @brief Add a property verifier
   * @param rule Shared pointer to the verifier to add
   *
   * Verifiers are executed in the order they are added.
   * If a verifier with the same name already exists, it will not be added again.
   */
  void AddRule(PropertyVerifierPtr rule);

  void EnableRule(const std::string& name);
  void DisableRule(const std::string& name);
  [[nodiscard]] bool IsRuleEnabled(const std::string& name) const;

  [[nodiscard]] std::vector<Diagnostic> Verify(const ProgramPtr& program) const;
  void VerifyOrThrow(const ProgramPtr& program) const;

  static std::string GenerateReport(const std::vector<Diagnostic>& diagnostics);
  static IRVerifier CreateDefault();

 private:
  std::vector<PropertyVerifierPtr> rules_;
  std::unordered_set<std::string> disabled_rules_;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_VERIFIER_VERIFIER_H_

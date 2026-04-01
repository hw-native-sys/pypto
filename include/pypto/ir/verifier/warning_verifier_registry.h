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

#ifndef PYPTO_IR_VERIFIER_WARNING_VERIFIER_REGISTRY_H_
#define PYPTO_IR_VERIFIER_WARNING_VERIFIER_REGISTRY_H_

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

/// Identifies a specific warning check (independent of IRProperty)
enum class WarningCheck : uint32_t {
  UnusedVariable = 0,
  // future: SuspiciousInOutAnnotation, UnreachableCode, RedundantAssignment, ...
  kCount
};

/// Convert a WarningCheck to its string name
std::string WarningCheckToString(WarningCheck check);

/// Bitset for selecting which warning checks to run (mirrors IRPropertySet pattern)
class WarningCheckSet {
 public:
  WarningCheckSet() : bits_(0) {}

  WarningCheckSet(std::initializer_list<WarningCheck> checks) : bits_(0) {
    for (auto c : checks) {
      Insert(c);
    }
  }

  void Insert(WarningCheck check) { bits_ |= Bit(check); }
  void Remove(WarningCheck check) { bits_ &= ~Bit(check); }
  [[nodiscard]] bool Contains(WarningCheck check) const { return (bits_ & Bit(check)) != 0; }
  [[nodiscard]] bool Empty() const { return bits_ == 0; }

  [[nodiscard]] WarningCheckSet Difference(const WarningCheckSet& other) const {
    WarningCheckSet result;
    result.bits_ = bits_ & ~other.bits_;
    return result;
  }

  [[nodiscard]] std::vector<WarningCheck> ToVector() const;
  [[nodiscard]] std::string ToString() const;

  bool operator==(const WarningCheckSet& other) const { return bits_ == other.bits_; }
  bool operator!=(const WarningCheckSet& other) const { return bits_ != other.bits_; }

 private:
  uint32_t bits_;

  static uint32_t Bit(WarningCheck c) { return uint32_t{1} << static_cast<uint32_t>(c); }
};

/**
 * @brief Registry mapping WarningCheck values to their PropertyVerifier factories
 *
 * Parallel to PropertyVerifierRegistry, but keyed by WarningCheck instead of
 * IRProperty. Warning verifiers reuse the PropertyVerifier interface — they
 * push Diagnostic with severity = Warning.
 */
class WarningVerifierRegistry {
 public:
  static WarningVerifierRegistry& GetInstance();

  void Register(WarningCheck check, std::function<PropertyVerifierPtr()> factory);

  [[nodiscard]] PropertyVerifierPtr GetVerifier(WarningCheck check) const;

  /// Run selected checks, return Warning-severity diagnostics
  [[nodiscard]] std::vector<Diagnostic> RunChecks(const WarningCheckSet& checks,
                                                  const ProgramPtr& program) const;

  /// All registered checks
  static WarningCheckSet GetAllChecks();

 private:
  WarningVerifierRegistry();

  std::unordered_map<uint32_t, std::function<PropertyVerifierPtr()>> factories_;
};

/// Factory function for creating UnusedVariable warning verifier
PropertyVerifierPtr CreateUnusedVariableWarningVerifier();

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_VERIFIER_WARNING_VERIFIER_REGISTRY_H_

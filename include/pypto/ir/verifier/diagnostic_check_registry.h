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

#ifndef PYPTO_IR_VERIFIER_DIAGNOSTIC_CHECK_REGISTRY_H_
#define PYPTO_IR_VERIFIER_DIAGNOSTIC_CHECK_REGISTRY_H_

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

/// Identifies a specific advisory check.
///
/// Each value is registered with a severity (Warning or PerfHint), a phase
/// (PrePipeline / PostPass / PostPipeline) and, for PerfHint checks, a stable
/// hint code such as "PH001". Severity and phase are independent: a Warning
/// may run at PrePipeline, a PerfHint may run at PostPipeline — declared per
/// check, not per severity.
enum class DiagnosticCheck : uint32_t {
  // --- Warnings ----------------------------------------------------------
  UnusedVariable = 0,
  UnusedControlFlowResult = 1,
  // --- Performance hints (issue #1180) -----------------------------------
  TileInnermostDimGranularity = 2,
  // future: TileShapeBlocksDmaVectorization, PartialPipelineFill, ...
  kCount
};

/// Convert a DiagnosticCheck to its string name
std::string DiagnosticCheckToString(DiagnosticCheck check);

/// Bitset for selecting which checks to run.
class DiagnosticCheckSet {
 public:
  DiagnosticCheckSet() : bits_(0) {}

  DiagnosticCheckSet(std::initializer_list<DiagnosticCheck> checks) : bits_(0) {
    for (auto c : checks) {
      Insert(c);
    }
  }

  void Insert(DiagnosticCheck check) { bits_ |= Bit(check); }
  void Remove(DiagnosticCheck check) { bits_ &= ~Bit(check); }
  [[nodiscard]] bool Contains(DiagnosticCheck check) const { return (bits_ & Bit(check)) != 0; }
  [[nodiscard]] bool Empty() const { return bits_ == 0; }

  [[nodiscard]] DiagnosticCheckSet Difference(const DiagnosticCheckSet& other) const {
    DiagnosticCheckSet result;
    result.bits_ = bits_ & ~other.bits_;
    return result;
  }

  [[nodiscard]] DiagnosticCheckSet Union(const DiagnosticCheckSet& other) const {
    DiagnosticCheckSet result;
    result.bits_ = bits_ | other.bits_;
    return result;
  }

  [[nodiscard]] std::vector<DiagnosticCheck> ToVector() const;
  [[nodiscard]] std::string ToString() const;

  bool operator==(const DiagnosticCheckSet& other) const { return bits_ == other.bits_; }
  bool operator!=(const DiagnosticCheckSet& other) const { return bits_ != other.bits_; }

  /**
   * @brief Hash by the underlying bitset (consistent with operator==)
   */
  [[nodiscard]] uint64_t Hash() const { return std::hash<uint32_t>{}(bits_); }

 private:
  uint32_t bits_;

  static uint32_t Bit(DiagnosticCheck c) { return uint32_t{1} << static_cast<uint32_t>(c); }

  static_assert(static_cast<uint32_t>(DiagnosticCheck::kCount) <= 32,
                "DiagnosticCheck count exceeds 32, which is the maximum supported by "
                "DiagnosticCheckSet's uint32_t bitset");
};

/**
 * @brief Registry mapping DiagnosticCheck values to their PropertyVerifier factories.
 *
 * Each registration carries a severity, a phase, and an optional hint code.
 * Verifiers reuse the PropertyVerifier interface — they push Diagnostic
 * objects with the appropriate severity. The registry stamps the registered
 * severity and hint_code onto every diagnostic the verifier produces.
 */
class DiagnosticCheckRegistry {
 public:
  struct Entry {
    DiagnosticSeverity severity;
    DiagnosticPhase phase;
    std::string hint_code;  ///< Empty for warnings; e.g. "PH001" for perf hints
    std::function<PropertyVerifierPtr()> factory;
  };

  static DiagnosticCheckRegistry& GetInstance();

  void Register(DiagnosticCheck check, DiagnosticSeverity severity, DiagnosticPhase phase,
                std::string hint_code, std::function<PropertyVerifierPtr()> factory);

  [[nodiscard]] PropertyVerifierPtr GetVerifier(DiagnosticCheck check) const;
  [[nodiscard]] const Entry* GetEntry(DiagnosticCheck check) const;

  /**
   * @brief Run the subset of `checks` whose registered phase equals `phase`.
   *
   * The returned diagnostics carry the registered severity and hint code,
   * regardless of what the verifier itself wrote into those fields.
   */
  [[nodiscard]] std::vector<Diagnostic> RunChecks(const DiagnosticCheckSet& checks, DiagnosticPhase phase,
                                                  const ProgramPtr& program) const;

  /// All registered checks, regardless of severity or phase.
  static DiagnosticCheckSet GetAllChecks();

  /// All registered Warning-severity checks.
  static DiagnosticCheckSet GetWarningChecks();

  /// All registered PerfHint-severity checks.
  static DiagnosticCheckSet GetPerfHintChecks();

 private:
  DiagnosticCheckRegistry();

  std::unordered_map<uint32_t, Entry> entries_;
};

/// Factory function for creating UnusedVariable warning verifier
PropertyVerifierPtr CreateUnusedVariableWarningVerifier();

/// Factory function for creating UnusedControlFlowResult warning verifier
PropertyVerifierPtr CreateUnusedControlFlowResultWarningVerifier();

/// Factory function for creating TileInnermostDimGranularity perf-hint verifier (issue #1180)
PropertyVerifierPtr CreateTileInnermostDimGranularityVerifier();

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_VERIFIER_DIAGNOSTIC_CHECK_REGISTRY_H_

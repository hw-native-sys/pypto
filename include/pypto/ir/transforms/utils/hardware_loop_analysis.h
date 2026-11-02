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
#ifndef PYPTO_IR_TRANSFORMS_UTILS_HARDWARE_LOOP_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_HARDWARE_LOOP_ANALYSIS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"

namespace pypto::ir {

/// Capabilities are immutable inputs to an analysis invocation, not IR fields.
/// An op allowlist certifies that preserving its sequential execution preserves
/// dependencies and fits the target's resource model. Unlisted calls stay Unknown.
struct HardwareLoopProfile {
  bool supported = false;
  /// Unsigned counter width; both the initial and terminating values must fit.
  int counter_bits = 16;
  bool allow_negative_step = false;
  int64_t max_trip_count = 65535;
  int max_depth = 4;
  int64_t max_body_nodes = 256;
  bool allow_scalar_carry = true;
  bool allow_reg_carry = false;
  bool allow_tile_carry = false;
  std::vector<std::string> allowed_ops;
};

enum class HardwareLoopState { Eligible, Ineligible, Unknown };

struct HardwareLoopQualification {
  HardwareLoopState state = HardwareLoopState::Unknown;
  std::string reason;
  std::optional<int64_t> trip_count;
  std::optional<int64_t> post_iv;
  std::string loop_id;
  std::optional<int64_t> lower;
  std::optional<int64_t> upper;
  std::optional<int64_t> step;
};

/// Require's fixed bounds are mandatory even for statically empty loops.
bool HasCanonicalHardwareLoopBounds(const ForStmtPtr& loop);
/// Recomputed from current IR/profile on every call; summaries never enter hashes.
HardwareLoopQualification AnalyzeHardwareLoop(const ForStmtPtr& loop, const HardwareLoopProfile& profile);
std::vector<HardwareLoopQualification> AnalyzeHardwareLoops(const ProgramPtr& program,
                                                            const HardwareLoopProfile& profile);

/// A standalone qualification is never final hardware evidence.
void VerifyHardwareLoopRequirement(const ForStmtPtr& loop, const HardwareLoopProfile& profile);
/// Source generation may defer evidence to the production compiler, which must
/// carry each requirement into its helper artifact and validate the linked ELF.
void VerifyHardwareLoopRequirements(const ProgramPtr& program, const HardwareLoopProfile& profile,
                                    bool defer_evidence = false);

}  // namespace pypto::ir
#endif  // PYPTO_IR_TRANSFORMS_UTILS_HARDWARE_LOOP_ANALYSIS_H_

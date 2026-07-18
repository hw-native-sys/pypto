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

#ifndef PYPTO_IR_TRANSFORMS_DSA_REUSE_PENALTY_RECOGNIZER_H_
#define PYPTO_IR_TRANSFORMS_DSA_REUSE_PENALTY_RECOGNIZER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"

namespace pypto {
namespace ir {
namespace dsa_adapter {

enum class RecognizedReuseHazard : uint8_t {
  SamePipe,
  CrossPipe,
};

enum class RecognizedReuseDependence : uint8_t {
  WriteAfterRead,
  WriteAfterWrite,
};

struct RecognizedReuseCandidate {
  size_t first_interval;
  size_t second_interval;
  RecognizedReuseHazard hazard;
  RecognizedReuseDependence dependence;
  bool nested_control = false;
};

struct RecognizedReusePenalty {
  size_t first_interval;
  size_t second_interval;
  uint64_t cost;
  RecognizedReuseHazard hazard;
};

struct ReusePenaltyRecognition {
  std::vector<RecognizedReuseCandidate> candidates;
  std::vector<RecognizedReusePenalty> penalties;
  size_t supported_allocations = 0;
  size_t candidate_pairs = 0;
  size_t already_ordered_pairs = 0;
  size_t same_pipe_candidates = 0;
  size_t cross_pipe_candidates = 0;
  size_t write_after_read_candidates = 0;
  size_t write_after_write_candidates = 0;
  size_t nested_control_candidates = 0;
};

/**
 * @brief Recognize candidate physical-reuse hazards from PyPTO access order.
 *
 * Linear mode considers only adjacent statement handoffs and runs in
 * O(N log N). Quadratic mode is an explicitly approved research reference that
 * scans all lifetime-compatible allocation pairs; its cached reachability
 * queries cost O(B*(N+E) + B^2) in the worst case.
 */
[[nodiscard]] ReusePenaltyRecognition RecognizeReusePenaltyCandidates(const FunctionPtr& func,
                                                                      const AllocationPlan& allocation_plan,
                                                                      DsaReusePenaltyRecognizer recognizer);

/**
 * @brief Promote the experimental v1 subset to unit-weight pair penalties.
 *
 * Candidate recognition records mechanism evidence. This separate policy
 * promotes flat cross-pipe candidates only; same-pipe and nested candidates
 * remain observable but unpriced until their behavior is better characterized.
 */
void ApplyExperimentalUnitPenaltyPolicy(ReusePenaltyRecognition* recognition);

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_DSA_REUSE_PENALTY_RECOGNIZER_H_

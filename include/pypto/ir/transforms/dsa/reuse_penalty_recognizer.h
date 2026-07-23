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
#include <limits>
#include <string>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"

namespace pypto {
namespace ir {
namespace dsa_adapter {

enum class RecognizedReuseHazard : uint8_t {
  SameResource,
  CrossResource,
};

enum class RecognizedMemoryClass : uint8_t {
  External,
  Ub,
  L1,
  L0,
  Scalar,
};

enum class RecognizedAccessResource : uint8_t {
  InboundDma,
  OutboundDma,
  L0ToExternal,
  L1ToL0,
  L0ToL1,
  UbToL1,
  L1ToUb,
  UbToL0,
  L0ToUb,
  VectorCompute,
  MatrixCompute,
  ScalarAccess,
};

struct RecognizedAccessRoute {
  RecognizedMemoryClass source = RecognizedMemoryClass::Scalar;
  RecognizedMemoryClass destination = RecognizedMemoryClass::Scalar;
  RecognizedAccessResource resource = RecognizedAccessResource::ScalarAccess;
};

[[nodiscard]] std::string RecognizedMemoryClassToString(RecognizedMemoryClass memory_class);
[[nodiscard]] std::string RecognizedAccessResourceToString(RecognizedAccessResource resource);
[[nodiscard]] std::string RecognizedAccessRouteToString(const RecognizedAccessRoute& route);

enum class RecognizedReuseDependence : uint8_t {
  WriteAfterRead,
  WriteAfterWrite,
};

struct RecognizedReuseCandidate {
  size_t first_interval;
  size_t second_interval;
  size_t prior_interval;
  size_t next_interval;
  RecognizedReuseHazard hazard;
  RecognizedReuseDependence dependence;
  RecognizedAccessRoute prior_route;
  RecognizedAccessRoute next_route;
  MemorySpace prior_memory_space = MemorySpace::ScalarLocal;
  MemorySpace next_memory_space = MemorySpace::ScalarLocal;
  size_t prior_access_order = 0;
  size_t next_access_order = 0;
  uint64_t prior_byte_offset = 0;
  uint64_t prior_byte_size = 0;
  uint64_t next_byte_offset = 0;
  uint64_t next_byte_size = 0;
  size_t loop_id = std::numeric_limits<size_t>::max();
  bool ordered_by_logical_dag = false;
  bool requires_alias_contract = false;
  bool partial_access = false;
  bool incomplete_access_set = false;
  bool conservative_initial_anchor = false;
  bool nested_control = false;
  bool in_loop = false;
  bool loop_carried = false;
};

struct RecognizedReuseEdge {
  size_t first_interval;
  size_t second_interval;
  RecognizedReuseHazard hazard;
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
  std::vector<RecognizedReuseEdge> edges;
  std::vector<RecognizedReusePenalty> penalties;
  std::vector<RecognizedAccessRoute> observed_routes;
  size_t supported_allocations = 0;
  size_t candidate_pairs = 0;
  size_t already_ordered_pairs = 0;
  size_t partially_supported_allocations = 0;
  size_t same_resource_candidates = 0;
  size_t cross_resource_candidates = 0;
  size_t write_after_read_candidates = 0;
  size_t write_after_write_candidates = 0;
  size_t ordered_evidence_candidates = 0;
  size_t alias_contract_candidates = 0;
  size_t partial_access_candidates = 0;
  size_t incomplete_access_candidates = 0;
  size_t conservative_initial_anchor_candidates = 0;
  size_t nested_control_candidates = 0;
  size_t in_loop_candidates = 0;
  size_t loop_carried_candidates = 0;
};

/**
 * @brief Recognize candidate physical-reuse hazards from PyPTO access order.
 *
 * Quadratic mode is the coverage-first research reference: it scans all
 * lifetime-compatible allocation pairs and compares per-resource access
 * frontiers, including nested and loop-carried handoffs. It is intentionally
 * super-linear and must not become a default production pass.
 */
[[nodiscard]] ReusePenaltyRecognition RecognizeReusePenaltyCandidates(const FunctionPtr& func,
                                                                      const AllocationPlan& allocation_plan,
                                                                      DsaReusePenaltyRecognizer recognizer);

/**
 * @brief Construct the mechanically justified experimental pair-edge subset.
 *
 * Candidate recognition records mechanism evidence. This separate policy
 * constructs one edge per qualifying cross-resource buffer pair when the
 * recognizer has a complete, full-allocation handoff, a verified minimal-write
 * antichain, no same-operation alias question, and no pre-existing completion
 * relation. Each abstract resource is modeled as one completion-ordered issue
 * chain; SSA def-use is also a completion dependency, but bare lexical
 * statement order is not. Distance-zero handoffs in structured control are
 * included; loop-carried, same-resource, partial-view, and uncertain
 * candidates remain report-only.
 */
void ConstructExperimentalPairEdges(ReusePenaltyRecognition* recognition);

/**
 * @brief Apply the experimental unit weight model to constructed pair edges.
 *
 * Edge construction and weight assignment are deliberately separate: changing
 * a compiler- or device-specific cost model must not change which hazards were
 * mechanically recognized.
 */
void ApplyExperimentalUnitPenaltyWeights(ReusePenaltyRecognition* recognition);

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_DSA_REUSE_PENALTY_RECOGNIZER_H_

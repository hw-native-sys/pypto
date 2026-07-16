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

#ifndef PYPTO_IR_TRANSFORMS_DSA_MEMREF_DSA_ADAPTER_H_
#define PYPTO_IR_TRANSFORMS_DSA_MEMREF_DSA_ADAPTER_H_

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dsa/model.h"
#include "dsa/solver.h"
#include "dsa/structured_problem.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"

namespace pypto {
namespace ir {

class MemoryAllocatorPolicy;

namespace dsa_adapter {

using MemRefWithSpace = std::pair<MemRefPtr, MemorySpace>;

/**
 * @brief A standalone structured problem plus its transient IR writeback map.
 *
 * ``document`` is IR-free and safe to serialize into the benchmark corpus.
 * ``buffer_id_by_base`` remains inside PyPTO: it maps each allocation identity
 * to the standalone buffer whose placement must be written back.
 */
struct ExportedProblem {
  ::dsa::StructuredProblemDocument document;
  std::unordered_map<const Var*, ::dsa::BufferId> buffer_id_by_base;
};

/**
 * @brief Result of capability matching, solving, and independent validation.
 */
struct SolverRun {
  ::dsa::SolverCompatibility compatibility;
  ::dsa::DsaResult result;
  std::vector<std::string> problem_errors;
  std::vector<std::string> solution_errors;
};

/**
 * @brief Convert unmerged PyPTO allocation identities to schema-v1 structured DSA.
 *
 * Each MemRef ``base_`` becomes one fixed-pool buffer. Semantics-required aliases
 * remain one identity with a conservative allocation-level lifetime hull;
 * per-member SSA gaps are not physical dead-time proofs. Opportunistic reuse
 * remains entirely for the standalone solver. PyPTO statement points are
 * expanded into read/write sub-points so an input's last read may share an
 * address with an output written by the same statement.
 */
[[nodiscard]] ExportedProblem BuildStructuredProblem(
    const FunctionPtr& func, const AllocationPlan& allocation_plan, const MemoryAllocatorPolicy& policy,
    const std::unordered_map<MemorySpace, uint64_t>& reserved_end_by_space,
    const std::unordered_map<MemorySpace, uint64_t>& pool_caps);

/**
 * @brief Write a deterministic ``<function>.dsa.json`` corpus artifact.
 *
 * External-library and filesystem exceptions are translated to PyPTO errors at
 * this boundary. The returned path is the exact file written.
 */
[[nodiscard]] std::string WriteProblemJson(const ExportedProblem& exported, const std::string& directory);

/**
 * @brief Write the validated placement selected for an exported problem.
 *
 * The solution artifact carries a fingerprint of the complete problem
 * document. A later compilation can therefore reject stale or mismatched
 * placements before any address is written back to PyPTO IR.
 */
[[nodiscard]] std::string WriteSolutionJson(const ExportedProblem& exported,
                                            const ::dsa::DsaSolution& solution, const std::string& directory,
                                            std::map<std::string, std::string> metadata = {});

/**
 * @brief Read the deterministic solution artifact for one function.
 */
[[nodiscard]] ::dsa::StructuredSolutionDocument ReadSolutionJson(const std::string& instance,
                                                                 const std::string& directory);

/**
 * @brief Run one standalone solver and independently validate its result.
 */
[[nodiscard]] SolverRun Solve(const ExportedProblem& exported, const ::dsa::DsaSolver& solver);

/**
 * @brief Run the standalone deterministic baseline and independently validate it.
 */
[[nodiscard]] SolverRun SolveWithFirstFit(const ExportedProblem& exported);

/**
 * @brief Convert validated standalone placements to fresh PyPTO MemRefs.
 *
 * Every view preserves its relative ``byte_offset_`` within the newly placed
 * allocation base, matching AllocateMemoryAddr's legacy writeback semantics.
 */
[[nodiscard]] std::vector<std::pair<const MemRef*, MemRefPtr>> BuildMemRefReplacements(
    const ExportedProblem& exported, const ::dsa::DsaSolution& solution,
    const std::vector<MemRefWithSpace>& memrefs, const MemoryAllocatorPolicy& policy);

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_DSA_MEMREF_DSA_ADAPTER_H_

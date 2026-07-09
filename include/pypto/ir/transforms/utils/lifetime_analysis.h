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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_LIFETIME_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_LIFETIME_ANALYSIS_H_

#include <cstdint>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {

/**
 * @brief Lifetime interval for one allocation (a base-group of TileType vars).
 *
 * One interval per physical allocation: views and semantic must-aliases that
 * share a ``base_`` Ptr are collapsed into a single interval whose [def, last_use]
 * is the union over the group's members (topological order).  This is the unit
 * the reuse packer — and the DSA adapter — treats as one buffer.
 */
struct LifetimeInterval {
  VarPtr variable;           ///< Representative variable of the sharing group.
  int def_point;             ///< Group's earliest definition point (topological order).
  int last_use_point;        ///< Group's latest last-use point (topological order).
  MemorySpace memory_space;  ///< Memory space (== DSA pool).
  uint64_t size;             ///< Slot size in bytes (largest member).
};

/**
 * @brief Per-allocation lifetime intervals for a function body.
 *
 * Thin, IR-facing entry point over the reuse pass's lifetime analysis, exposed so
 * the DSA adapter can build a DsaProblem without duplicating the (phi/loop-aware)
 * liveness computation.  Intervals reflect must-aliases + views already collapsed
 * (via ``base_`` identity), but NOT opportunistic lifetime reuse — that is exactly
 * what a DSA solver decides from these intervals.
 *
 * @param func_body The function body to analyze.
 * @return One LifetimeInterval per allocation; empty if the body holds no tiles.
 */
[[nodiscard]] std::vector<LifetimeInterval> ComputeAllocationLifetimes(const StmtPtr& func_body);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_LIFETIME_ANALYSIS_H_

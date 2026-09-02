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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_SCALAR_OUTPUT_HOIST_H_
#define PYPTO_IR_TRANSFORMS_UTILS_SCALAR_OUTPUT_HOIST_H_

#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {
namespace outline_utils {

/**
 * @brief Plan for removing Scalar values from a device kernel's return set.
 *
 * The runtime has no scalar output channel: ``Arg::add_scalar`` passes a
 * scalar *in* by value, and ``TaskOutputTensors`` returns only tensors. A
 * ScalarType in an outlined function's ``return_types_`` is therefore
 * unrepresentable — orchestration codegen has no runtime carrier to bind it to,
 * which is why such a return used to reach the generated C++ as an undefined
 * identifier and later as a silently wrong ``= 0`` (issue #631).
 *
 * When the value is a pure scalar function of the scope's live-in set, the
 * *caller* can compute it instead. This plan moves those defining statements
 * out of the scope body, ahead of the call. Whatever the body still reads then
 * becomes an ordinary scalar input parameter — the direction the runtime does
 * support.
 */
struct ScalarHoistPlan {
  /// Statements to emit in the caller immediately before the call, in their
  /// original body order.
  std::vector<StmtPtr> hoisted;

  /// The scope body with @ref hoisted removed. Equal to the input body when
  /// nothing is hoisted.
  StmtPtr new_body;

  /// First Scalar live-out that could *not* be hoisted; null when the plan is
  /// complete. The caller turns this into a user-facing error — the value has
  /// no way out of the kernel, so compilation cannot continue.
  const Var* blocker = nullptr;
};

/**
 * @brief Plan the hoist for one scope body.
 *
 * Only top-level statements of a ``SeqStmts`` body are candidates: a definition
 * nested in a ForStmt / IfStmt is not loop-invariant by construction, so lifting
 * it out of its control flow would be unsound. Such a live-out becomes a
 * @ref ScalarHoistPlan::blocker instead.
 *
 * Complexity is O(N log N) in the size of @p body: one forward scan to build the
 * candidate table, then a worklist closure bounded by the candidate count.
 *
 * @param body       The scope body to analyse.
 * @param live_in    Variables whose incoming value comes from the caller (the
 *                   scope's upward-exposed uses). These are the leaves the
 *                   caller-side computation may reference.
 * @param used_after Variables referenced after the scope — the output candidate
 *                   set the outliner is about to turn into return values.
 */
ScalarHoistPlan PlanScalarOutputHoist(const StmtPtr& body, const std::vector<const Var*>& live_in,
                                      const std::unordered_set<const Var*>& used_after);

}  // namespace outline_utils
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SCALAR_OUTPUT_HOIST_H_

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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_DEAD_CODE_ELIMINATION_H_
#define PYPTO_IR_TRANSFORMS_UTILS_DEAD_CODE_ELIMINATION_H_

#include <memory>
#include <string>
#include <vector>

#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {
namespace dce {

/// Extract the Op name from an AssignStmt or EvalStmt containing a Call.
/// Returns empty string if the statement doesn't match this pattern.
std::string GetStmtOpName(const StmtPtr& stmt);

/// Check if a statement is a side-effect op that must be preserved.
/// The predicate is customizable; this is a default implementation for
/// cross-core and tile ops.
bool IsSideEffectOp(const StmtPtr& stmt);

/// Collect all AssignStmts recursively from nested statements.
void CollectAllAssignStmts(const std::vector<StmtPtr>& stmts,
                           std::vector<std::shared_ptr<const AssignStmt>>& assigns);

/// Eliminate dead code from a statement list.
/// Dead statements are those whose defined variable is not transitively
/// used by any return, yield, or side-effect statement.
std::vector<StmtPtr> EliminateDeadCode(const std::vector<StmtPtr>& stmts);

}  // namespace dce
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_DEAD_CODE_ELIMINATION_H_

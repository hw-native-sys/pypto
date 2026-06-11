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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_RETURN_LINEAGE_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_RETURN_LINEAGE_UTILS_H_

#include <cstddef>
#include <optional>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {
namespace return_lineage {

/// Trace the first value of @p func's topmost ReturnStmt back to a Param.
///
/// Follows SSA rebinds (var-to-var assigns, For/While iter args and
/// return_vars), builtin writeback ops (``tensor.assemble``, ``tile.store``,
/// ``tensor.set_validshape``), TupleGetItem of a user-call result, and
/// user calls (single- and tuple-result) — recursing into callees with
/// memoization and a cycle guard. Group/Spmd wrappers with no top-level
/// ReturnStmt resolve through their unique returning inner call.
///
/// @return index into ``func->params_``, or nullopt when not a param writeback.
std::optional<size_t> ReturnedParamIndex(const FunctionPtr& func, const ProgramPtr& program);

/// Per-position generalization of ReturnedParamIndex over all return values.
///
/// @return empty vector when @p func has no traceable top-level ReturnStmt
///         and is not a wrapper resolvable through its inner call; otherwise
///         one entry per return position (nullopt = not a param writeback).
std::vector<std::optional<size_t>> ReturnedParamIndices(const FunctionPtr& func, const ProgramPtr& program);

/// Trace @p var defined in @p body back to one of @p params (pointer identity).
///
/// Same lineage rules as ReturnedParamIndex but scoped to an arbitrary
/// body/param-set (used by the scope outliner before the Function exists).
///
/// @return the matching param, or nullptr when untraceable.
VarPtr TraceToParam(const VarPtr& var, const StmtPtr& body, const std::vector<VarPtr>& params,
                    const ProgramPtr& program);

}  // namespace return_lineage
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_RETURN_LINEAGE_UTILS_H_

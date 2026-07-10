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
/// ReturnStmt resolve through their unique returning inner call; a wrapper that
/// *does* return, but returns the forwarded tuple of a multi-result inner call
/// (``result = self.inner(...); return result``), is expanded position-by-
/// position through that call — only when the wrapper declares that many flat
/// return positions, so a single ``pl.Tuple[...]`` return (one TupleType in
/// ``return_types_``) keeps its 1-arity map.
///
/// @return index into ``func->params_``, or nullopt when not a param writeback.
std::optional<size_t> ReturnedParamIndex(const FunctionPtr& func, const ProgramPtr& program);

/// Per-position generalization of ReturnedParamIndex over all return values.
///
/// @return empty vector when @p func has no traceable top-level ReturnStmt
///         and is not a wrapper resolvable through its inner call; otherwise
///         one entry per return position (nullopt = not a param writeback).
std::vector<std::optional<size_t>> ReturnedParamIndices(const FunctionPtr& func, const ProgramPtr& program);

/// Read the return->param map straight off @p func's ReturnStmt.
///
/// `NormalizeReturnOrder` rewrites every tensor param-writeback return value to
/// reference its param directly, and `IRProperty::ReturnParamsExplicit` verifies
/// it. Once that property holds the map is a *local structural fact*: return
/// position `j` writes back param `i` exactly when `ReturnStmt->value_[j]` is
/// `func->params_[i]` by pointer identity. No SSA walk, no callee recursion, no
/// `Program`.
///
/// Every consumer at or after `NormalizeReturnOrder` — orchestration codegen,
/// `ClassifyIterArgCarry` — must use this rather than `ReturnedParamIndices`:
/// it is a 1-to-1 read of the IR, so it cannot silently disagree with what the
/// IR actually says. Reserve `ReturnedParamIndices` for callers that run
/// *before* the property is established (`ExpandMixedKernel`, the scope
/// outliner), for `NormalizeReturnOrder` itself, and for the property verifier,
/// which must re-derive independently to have anything to check.
///
/// Scalar return positions are never canonicalized, so they resolve here only
/// when the value literally *is* a scalar param. That is deliberate: propagating
/// param lineage onto a scalar return once made codegen emit `const Tensor&` for
/// an `int64_t` (issue #1580).
///
/// @return one entry per return position (nullopt = not a param writeback);
///         empty when @p func has no ReturnStmt.
std::vector<std::optional<size_t>> ExplicitReturnedParamIndices(const FunctionPtr& func);

/// First-position convenience wrapper over ExplicitReturnedParamIndices.
std::optional<size_t> ExplicitReturnedParamIndex(const FunctionPtr& func);

/// Locate the topmost ReturnStmt of @p body, or nullptr when there is none.
///
/// Fast-paths the common shape (a `SeqStmts` whose last statement is the
/// return) and falls back to a pre-order walk for bodies that nest it — split
/// AIV kernels keep theirs inside the split body.
///
/// The two paths agree because a function body carries at most one ReturnStmt
/// by the time this runs: `CtrlFlowTransform` rewrites early returns into
/// structured control flow, so returns are function-terminal. A body holding
/// both a nested and a trailing ReturnStmt would make "topmost" ambiguous — the
/// fast path would pick the trailing one, a pre-order walk the nested one.
ReturnStmtPtr FindFirstReturn(const StmtPtr& body);

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

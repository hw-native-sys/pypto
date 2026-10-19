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
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace return_lineage {

/// Share callee return maps across queries into one unchanged Program.
///
/// Nesting is allowed; the outermost scope clears the cache on exit. End the
/// scope before changing callee bodies or querying a different Program, since
/// a function's lineage can depend on other functions in that Program.
class ScopedTraceCache {
 public:
  ScopedTraceCache();
  ~ScopedTraceCache();
  ScopedTraceCache(const ScopedTraceCache&) = delete;
  ScopedTraceCache& operator=(const ScopedTraceCache&) = delete;
};

/// Trace the first value of @p func's topmost ReturnStmt back to a Param.
///
/// Follows SSA rebinds (var-to-var assigns, For/While iter args and
/// return_vars), tensor IfStmt merges whose branches resolve to the same
/// param, builtin writeback ops (``tensor.assemble``, ``tile.store``,
/// ``tensor.set_validshape``), TupleGetItem of a user-call result, and
/// user calls (single- and tuple-result) — recursing into callees with
/// memoization and a cycle guard. Tensor return values must have the same type
/// as the resolved parameter, including view and valid-shape metadata.
/// Graph functions exclude IfStmt merges until their boundary analyses support them.
/// Group/Spmd wrappers with no top-level ReturnStmt resolve through their unique returning inner call; a
/// wrapper that *does* return, but returns the forwarded tuple of a multi-result inner call
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

/// One declared return position, retaining its value independently of storage lineage.
struct ReturnedValueBinding {
  /// Callee-scoped source expression, or nullptr when its source is unknown.
  /// May be an analysis-only tuple projection. Never substitute this expression
  /// directly into another function: resolve its storage or materialize an output first.
  ExprPtr value;
  /// Declared return type, retained even when the source or storage is unknown.
  TypePtr type;
  /// Complete-parameter writeback, if known; nullopt does not erase the returned value.
  std::optional<size_t> param_index;
};

/// Preserve every declared return slot before classifying parameter writebacks.
///
/// Retains original ReturnStmt expressions, expanding a single forwarded tuple
/// only for multiple flat declared returns. A single declared TupleType stays
/// one slot. Wrappers without an explicit return use their unique returning
/// inner call when recognizable; otherwise their typed slots remain unknown.
std::vector<ReturnedValueBinding> AnalyzeReturnedValues(const FunctionPtr& func, const ProgramPtr& program);

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
/// param lineage onto a scalar return once made codegen emit `const ChipTensor&` for
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

/// Trace multiple Vars in @p body back to complete parameter writebacks.
///
/// The body is indexed once, and variable roots and callee return maps are
/// memoized across all Vars. Follows SSA and call/writeback lineage without
/// IfStmt merges. Tensor results must retain the parameter's full type/view;
/// slices, views, fresh buffers, and untraceable values remain nullopt.
///
/// @return one entry per Var containing its matching param index, or nullopt.
std::vector<std::optional<size_t>> TraceToParamIndices(const std::vector<VarPtr>& vars, const StmtPtr& body,
                                                       const std::vector<VarPtr>& params,
                                                       const ProgramPtr& program);

/// Expression generalization of TraceToParamIndices.
///
/// Also follows direct tuple projections, tuple-variable aliases, and nested
/// MakeTuple projections. User-call projections use the callee's flat return
/// positions. The body index, variable roots, tuple sources, and callee return
/// maps are shared across the batch; unknown or partial-buffer values remain
/// nullopt under the same full-parameter type/view checks.
///
/// @return one entry per expression containing its matching param index, or nullopt.
std::vector<std::optional<size_t>> TraceExprsToParamIndices(const std::vector<ExprPtr>& exprs,
                                                            const StmtPtr& body,
                                                            const std::vector<VarPtr>& params,
                                                            const ProgramPtr& program);

/// Trace multiple Vars in @p body back to @p params for scope outlining.
///
/// The body is indexed once and parameter positions are shared across all Vars.
/// With @p trace_if_merges, this additionally follows tensor IfStmt merges
/// when every branch resolves to the same param. Conflicting or untraceable
/// branches remain nullopt, as do final results with a different type/view from the param.
/// ReturnedParamIndices uses the same branch consensus for non-Graph function returns.
///
/// @return one entry per Var containing its matching param index, or nullopt.
std::vector<std::optional<size_t>> TraceToParamIndicesForOutlining(const std::vector<VarPtr>& vars,
                                                                   const StmtPtr& body,
                                                                   const std::vector<VarPtr>& params,
                                                                   const ProgramPtr& program,
                                                                   bool trace_if_merges);

/// Trace @p var defined in @p body back to one of @p params (pointer identity).
///
/// Follows SSA and call/writeback lineage without IfStmt merges, scoped to an
/// arbitrary body/param-set. Use TraceToParamIndices for multiple Vars that
/// must retain the parameter's full type/view.
///
/// @return the matching param, or nullptr when untraceable.
VarPtr TraceToParam(const VarPtr& var, const StmtPtr& body, const std::vector<VarPtr>& params,
                    const ProgramPtr& program);

}  // namespace return_lineage
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_RETURN_LINEAGE_UTILS_H_

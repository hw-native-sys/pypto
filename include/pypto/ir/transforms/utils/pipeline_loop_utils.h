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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_PIPELINE_LOOP_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_PIPELINE_LOOP_UTILS_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {
namespace pipeline_loop {

/// Body-cloning primitives shared by the loop passes that replicate a loop body
/// into per-stage clones: ``LowerPipelineLoops`` (same-core `pl.pipeline`
/// replication), ``SkewCrossCorePipeline`` (cross-core prologue/steady/epilogue
/// skew), and ``UnrollLoops`` (the throwing const-int accessor only).
///
/// These are pure IR-construction helpers with no pass state. They were
/// maintained as byte-identical private copies in each pass, which is how the
/// #2002 cube-accumulator fix came to exist in one copy and not the other. The
/// membership *taggers* are deliberately NOT here — see the note at the bottom
/// of this header.
///
/// For constant evaluation, prefer ``transform_utils::EvalConstInt`` directly:
/// it is the non-throwing form, and it guards the ``INT64_MIN`` negation that
/// the former per-pass ``TryGetConstInt`` copies did not. ``GetConstIntValue``
/// below is only the throwing wrapper over it.

/// Extract a compile-time integer from a ``ConstInt`` or ``Neg(ConstInt)``
/// expression, throwing a user-facing ``ValueError`` when @p expr is not one.
///
/// @param expr  The expression to evaluate.
/// @param pass  Pass name used as the diagnostic prefix (e.g. "LowerPipelineLoops").
/// @param what  What the value is, for the message (e.g. "step").
/// @throws pypto::ValueError if @p expr is not a compile-time integer constant.
int64_t GetConstIntValue(const ExprPtr& expr, const char* pass, const std::string& what);

/// An ``INDEX``-typed ``ConstInt``.
ExprPtr MakeConstIndex(int64_t value, const Span& span);

/// `base + offset_val`, with constant-folding when `base` is a ConstInt.
/// Emitting the unfolded form trips the round-trip verifier because the
/// reparser folds `8 + 1` back to `9`.
ExprPtr OffsetIndex(const ExprPtr& base, int64_t offset_val, const Span& span);

/// Build a fresh outer loop variable mirroring `original` (same name, same type, same span).
VarPtr CloneLoopVar(const VarPtr& original);

/// Fresh IterArg mirroring `original`, with `init_value` as the initial value.
IterArgPtr MakeFreshIterArg(const IterArgPtr& original, const ExprPtr& init_value);

/// Fresh Var mirroring `original` with a suffixed name (for intermediate return_vars).
VarPtr MakeFreshVar(const VarPtr& original, const std::string& suffix);

/// Split a body into (stmts_before_yield, yield_values). If the body ends with a
/// terminal `YieldStmt` (either standalone or as the final stmt of a top-level
/// `SeqStmts`), strip it and return its values. Otherwise return the body unchanged
/// and an empty value list. Always pass through — callers that have no iter_args
/// simply see an empty yield vector and treat `stmts` as the whole body.
std::pair<StmtPtr, std::vector<ExprPtr>> SplitBodyYield(const StmtPtr& body);

/// Widen a vector of return_vars to expressions.
std::vector<ExprPtr> ReturnVarsAsExprs(const std::vector<VarPtr>& vars);

/// Collect `initValue_` expressions from a vector of IterArgs — used when the
/// tail runs without a preceding main loop, so its iter_args seed directly
/// from the source loop's init values rather than a main-loop return_var.
std::vector<ExprPtr> InitValueExprs(const std::vector<IterArgPtr>& iter_args);

/// Fresh return_vars matching the originals' types, with a suffix applied to names.
std::vector<VarPtr> MakeFreshReturnVars(const std::vector<VarPtr>& originals, const std::string& suffix);

// ---------------------------------------------------------------------------
// Deliberately NOT shared: the pipeline-membership taggers.
//
// `LowerPipelineLoops::PipelineMembershipTagger` and
// `SkewCrossCorePipeline::MembershipTagger` stamp the same
// `kPipelineMembershipAttr`, but their skip sets are genuinely different and
// each is load-bearing for its own pass. Unifying them would change buffer
// allocation on one side or the other. See the class comment on each tagger for
// the reasoning; the divergence is documented there, not resolved here.
// ---------------------------------------------------------------------------

}  // namespace pipeline_loop
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_PIPELINE_LOOP_UTILS_H_

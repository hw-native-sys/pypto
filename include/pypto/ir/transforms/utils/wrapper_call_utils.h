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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_WRAPPER_CALL_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_WRAPPER_CALL_UTILS_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace ir {

/**
 * @brief Result of a wrapper / inner-call lookup.
 *
 * Both fields are nullptr if no matching call was found.
 */
struct WrapperCallInfo {
  CallPtr inner_call;
  FunctionPtr inner_callee;
};

/**
 * @brief Find the first non-builtin Call inside @p wrapper that resolves to a
 *        Function in @p program.
 *
 * "Non-builtin" here means the Call's op is a GlobalVar that names an
 * existing user-level Function in the program. Builtin op calls
 * (`tile.*`, `tensor.*`, `system.*`) carry no GlobalVar and are skipped.
 *
 * @return {call, callee} for the first match, or {nullptr, nullptr} if none.
 */
WrapperCallInfo FindFirstInnerCall(const FunctionPtr& wrapper, const ProgramPtr& program);

/**
 * @brief Result of a Group-function callee scan.
 *
 * - `aic_name` / `aiv_name` — the names of the first AIC / AIV callees
 *   encountered (empty if none).
 * - `inner_call` / `inner_callee` — the **first** AIC, AIV, or InCore call
 *   in source order, regardless of type. Used by orchestration codegen as
 *   the parameter-order reference for wrapper arg reconciliation. After
 *   `ExpandMixedKernel`, Group bodies are emitted as `AIC → AIV` so the
 *   AIC call is naturally first in practice; the function does not enforce
 *   a type priority.
 */
struct GroupCalleeInfo {
  std::string aic_name;
  std::string aiv_name;
  CallPtr inner_call;
  FunctionPtr inner_callee;
};

/**
 * @brief Group-specific scan: locate the AIC / AIV callees and the first
 *        AIC/AIV/InCore inner call inside @p group_func.
 *
 * @return aggregated info; any field may be empty / nullptr if not present.
 */
GroupCalleeInfo FindGroupCallees(const FunctionPtr& group_func, const ProgramPtr& program);

/**
 * @brief Collect every Call inside @p wrapper that resolves to a Function
 *        of a non-Orchestration, non-Opaque type.
 *
 * Used by cross-function direction propagation in
 * `ComputeWrapperEffectiveDirections`. Visits the body in order; each inner
 * Call appears once even if its callee is called from multiple sites.
 */
std::vector<WrapperCallInfo> CollectInnerCalls(const FunctionPtr& wrapper, const ProgramPtr& program);

/**
 * @brief Recover the true param directions of every Group / Spmd wrapper in
 *        @p program, keyed by `Function*`.
 *
 * Wrapper functions synthesised by the scope outliners forward their params
 * 1:1 to an inner kernel call, but their own `param_directions_` can still
 * read `In` for a param the inner kernel writes — the outliner infers
 * directions from the *body it extracted*, and later passes
 * (`ExpandMixedKernel`, `SplitVectorKernel`, ...) rebuild wrapper bodies
 * around new callees without revisiting the signature.
 *
 * For each wrapper this walks its body's inner calls and merges each inner
 * callee's direction back onto the wrapper param the arg refers to, matching
 * by Var pointer identity. Merge lattice: `InOut` > `Out` > `In`. Nested
 * Group/Spmd callees resolve recursively, with a cycle guard that falls back
 * to the declared directions.
 *
 * The merge is **monotone**: it starts from the wrapper's declared directions,
 * so it can only promote (`In` → `Out` → `InOut`), never weaken a declared
 * writer to read-only. A wrapper with no body, no inner calls, or a param
 * written through a builtin instead of an inner call therefore keeps what its
 * signature already says.
 *
 * Whole-program rather than per-function because both callers iterate every
 * function: one shared memo visits each wrapper body once, keeping the cost
 * linear in total body size instead of O(wrappers x body size).
 *
 * `DeriveCallDirections` calls this and writes each result back into
 * `Function::param_directions_`, so every consumer downstream of that pass
 * reads the field directly instead of recomputing. Callers that run *before*
 * `DeriveCallDirections` and need the effective view must call this
 * explicitly.
 *
 * @return one entry per Group/Spmd function. Empty if @p program is null.
 */
std::unordered_map<const Function*, std::vector<ParamDirection>> ComputeWrapperEffectiveDirections(
    const ProgramPtr& program);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_WRAPPER_CALL_UTILS_H_

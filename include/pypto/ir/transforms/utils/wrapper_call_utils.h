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
 * Used by cross-function direction propagation in `ComputeGroupEffectiveDirections`.
 * Visits the body in order; each inner Call appears once even if its callee is
 * called from multiple sites.
 */
std::vector<WrapperCallInfo> CollectInnerCalls(const FunctionPtr& wrapper, const ProgramPtr& program);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_WRAPPER_CALL_UTILS_H_

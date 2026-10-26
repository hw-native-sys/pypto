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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_REGISTERED_CALL_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_REGISTERED_CALL_UTILS_H_

#include <functional>
#include <vector>

#include "pypto/ir/expr.h"

namespace pypto::ir::transform_utils {

/// Rebuild a registered, type-deducible operator from the caller's final args.
///
/// The caller owns SSA/geometry rewriting and supplies @p remap_expr for live
/// expressions in kwargs/attrs. A Substitute-based callback must treat generated
/// replacements as terminal references, rather than walking an IterArg's seed
/// back into its own replacement chain. Re-deduce from the new args/kwargs, then
/// preserve updated attrs and the original span. Never inherit the old result
/// type/MemRef: aliases, windows and copy placement remain caller/op policy.
CallPtr RebuildRegisteredCall(const CallPtr& original, const std::vector<ExprPtr>& rewritten_args,
                              const std::function<ExprPtr(const ExprPtr&)>& remap_expr);

}  // namespace pypto::ir::transform_utils

#endif  // PYPTO_IR_TRANSFORMS_UTILS_REGISTERED_CALL_UTILS_H_

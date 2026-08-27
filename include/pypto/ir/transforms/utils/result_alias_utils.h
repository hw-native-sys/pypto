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
#ifndef PYPTO_IR_TRANSFORMS_UTILS_RESULT_ALIAS_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_RESULT_ALIAS_UTILS_H_

#include <cstddef>
#include <optional>

#include "pypto/ir/expr.h"

namespace pypto {
namespace ir {

/**
 * @brief The argument index whose buffer @p call 's result names, or nullopt
 *        when the result is a fresh value.
 *
 * An SSA-pure writer returns a new Var for a buffer it updated in place, so the
 * result and that argument are two names for one tensor. Every analysis that
 * follows a value back to the parameter it came from needs the same answer
 * here: ``ConvertTensorToTileOps`` uses it to carry parameter origins across the
 * call, and ``ScopeOutliner`` uses it to decide whether reading the result
 * counts as reading the argument. Answering the question in two places let them
 * disagree — a collective with several write slots looked un-aliased to one and
 * aliased to the other, so the same program could come out ``Out`` from one
 * pass and ``InOut`` from the other.
 *
 * Two sources feed the answer, in order:
 *   1. the operator's own ``set_output_reuses_input`` declaration, and
 *   2. the explicit table below, for operators whose result aliases an argument
 *      that is not their declared reuse slot.
 *
 * The aliased argument must carry a declared effect. "Aliased" does not imply
 * "written" — ``tensor.set_validshape`` rebinds the valid extent and names the
 * same buffer without moving data — but an *unclassified* slot means nobody
 * decided, which is the state the argument-effect registry exists to remove.
 */
[[nodiscard]] std::optional<size_t> ResultAliasedArgIndex(const CallPtr& call);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_RESULT_ALIAS_UTILS_H_

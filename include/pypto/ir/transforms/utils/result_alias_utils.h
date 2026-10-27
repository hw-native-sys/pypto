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
 *        when the result has independent storage.
 *
 * Views and destination updates may return distinct SSA values sharing an
 * input's storage. This query carries parameter origins and read attribution
 * through both, without implying writes, identical windows/descriptors, or
 * interchangeable values. ConvertTensorToTileOps and ScopeOutliner use the same
 * answer so their direction inference stays consistent.
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

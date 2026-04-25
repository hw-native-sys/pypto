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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_OP_PREDICATES_H_
#define PYPTO_IR_TRANSFORMS_UTILS_OP_PREDICATES_H_

#include "pypto/ir/expr.h"

namespace pypto {
namespace ir {
namespace op_predicates {

/// True if the Call targets a tpop op (tile.tpop_from_aic / tile.tpop_from_aiv).
/// Decided by the registry's CrossCoreRole, not by op-name string matching.
bool IsTPop(const CallPtr& call);

/// True if the Call targets a tpush op (tile.tpush_to_aic / tile.tpush_to_aiv).
bool IsTPush(const CallPtr& call);

/// True if the Call targets a tfree op (system.tfree_to_aic / system.tfree_to_aiv).
bool IsTFree(const CallPtr& call);

/// True if the Call targets an initialize_pipe op
/// (system.aic_initialize_pipe / system.aiv_initialize_pipe).
bool IsInitializePipe(const CallPtr& call);

}  // namespace op_predicates
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_OP_PREDICATES_H_

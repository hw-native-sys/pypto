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

#ifndef SRC_IR_TRANSFORMS_LOWER_COMPOSITE_LOWER_COMPOSITE_RULES_H_
#define SRC_IR_TRANSFORMS_LOWER_COMPOSITE_LOWER_COMPOSITE_RULES_H_

#include <string>
#include <vector>

#include "pypto/ir/expr.h"

namespace pypto {
namespace ir {
namespace lower_composite {

// Rules take the builder by reference only, so a forward declaration keeps this
// header decoupled from lower_composite_builder.h and its include chain.
class LoweringBuilder;

// Signature for a composite-lowering rule.
//
// @param call     Original composite-op Call. Rules read ``call->kwargs_``,
//                 ``call->span_``, and ``call->op_->name_`` for diagnostics.
// @param args     Visited operand expressions (var-remap already applied).
//                 Prefer these over ``call->args_`` so the rule sees post-
//                 visitor expressions.
// @param builder  Scratchpad: rule appends intermediate temps via builder.Bind
//                 (and structured control-flow via EmitFor / EmitIf / ...) and
//                 returns the final result expression.
// @return Final result expression. The mutator binds this to the target ``Var``
//         and splices the builder's accumulated statements before it.
using CompositeLoweringFn = ExprPtr (*)(const CallPtr& call, const std::vector<ExprPtr>& args,
                                        LoweringBuilder& builder);

/// Look up a built-in recipe by operator name; return nullptr for other calls.
/// Function-context filtering and argument visitation belong to the mutator.
CompositeLoweringFn LookupCompositeRule(const std::string& op_name);

/// FP32 trigonometric recipes, defined together in lower_composite_math.cpp.
ExprPtr LowerSinRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b);
ExprPtr LowerCosRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b);

/// Packed-mask selection, defined in lower_composite_select.cpp.
ExprPtr LowerTileSelectRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b);

/// MX quantization and workspace construction, defined in lower_composite_mx.cpp.
ExprPtr LowerTileTQuantMxRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b);

// Each collective has its own translation unit. All entries use the same
// CompositeLoweringFn contract and are referenced by lower_composite_rules.cpp.

/// ``pld.tensor.allreduce`` — mesh mode, and ring mode via the ``mode="ring"``
/// kwarg. Defined in ``lower_composite_allreduce.cpp``.
ExprPtr LowerTensorAllReduceRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b);

/// ``pld.tensor.allgather`` — push-based 3-arg form. Defined in
/// ``lower_composite_allgather.cpp``.
ExprPtr LowerTensorAllGatherRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b);

/// ``pld.tensor.reduce_scatter`` — staged-window form. Defined in
/// ``lower_composite_reduce_scatter.cpp``.
ExprPtr LowerTensorReduceScatterRule(const CallPtr& call, const std::vector<ExprPtr>& args,
                                     LoweringBuilder& b);

/// ``pld.tensor.broadcast``. Defined in ``lower_composite_broadcast.cpp``.
ExprPtr LowerTensorBroadcastRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b);

/// ``pld.tensor.barrier``. Defined in ``lower_composite_barrier.cpp``.
ExprPtr LowerTensorBarrierRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b);

/// ``pld.tensor.all_to_all`` — dual-window form. Defined in
/// ``lower_composite_all_to_all.cpp``.
ExprPtr LowerTensorAllToAllRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b);

/// ``pld.tensor.all_to_all_v`` — 5-arg variable-size form (#2112). Defined in
/// ``lower_composite_all_to_all_v.cpp``.
ExprPtr LowerTensorAllToAllVRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b);

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_LOWER_COMPOSITE_LOWER_COMPOSITE_RULES_H_

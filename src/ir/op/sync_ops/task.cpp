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

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceTaskIdScalarType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs) {
  (void)args;
  (void)kwargs;
  return std::make_shared<ScalarType>(DataType::TASK_ID);
}

}  // namespace

// system.task_invalid — produces an invalid PTO2TaskId sentinel.
//
// Synthesized internally by ``LowerManualDepsToTaskId`` as the initial value
// for TaskId iter_args carrying manual-scope dep state across loop iterations.
// At codegen time it lowers to ``PTO2TaskId::invalid()``. Add_dep with an
// invalid task id is guarded by ``is_valid()`` so the runtime sees no edge
// on the first iteration.
REGISTER_OP("system.task_invalid")
    .set_description("Construct an invalid PTO2TaskId sentinel for manual_scope dep carries")
    .set_op_category("TaskOp")
    .no_argument()
    .f_deduce_type(DeduceTaskIdScalarType);

// system.task_id_of — extracts the PTO2TaskId of the kernel Call that produced
// the given Var. Synthesized internally by ``LowerManualDepsToTaskId`` to
// give the TaskId companion of a kernel Call LHS its own SSA def in the IR
// so that subsequent passes / codegen can reference it as a normal Var.
//
// Codegen lowers ``v_tid = task_id_of(v)`` to
// ``PTO2TaskId v_tid = task_<n>_outs.task_id();`` where ``task_<n>`` is the
// task counter assigned to the kernel Call that produced ``v``.
REGISTER_OP("system.task_id_of")
    .set_description("Extract the PTO2TaskId of a manual_scope kernel Call's producer Var")
    .set_op_category("TaskOp")
    .add_argument("producer", "Var produced by the kernel Call whose TaskId is wanted")
    .f_deduce_type(DeduceTaskIdScalarType);

}  // namespace ir
}  // namespace pypto

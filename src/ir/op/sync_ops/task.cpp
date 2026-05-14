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
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
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

TypePtr DeduceBoolScalarType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // ``system.task_is_valid`` is synthesized only by manual_scope lowering
  // (no user-facing surface), so a malformed Call here is a pass bug —
  // INTERNAL_CHECK is the right tool. Span comes from args[0] which is
  // guaranteed non-null by the size check that runs first.
  INTERNAL_CHECK(args.size() == 1) << "Internal error: system.task_is_valid expects 1 argument, got "
                                   << args.size();
  auto arg_type = As<ScalarType>(args[0]->GetType());
  INTERNAL_CHECK_SPAN(arg_type, args[0]->span_)
      << "Internal error: system.task_is_valid argument must be a Scalar, got "
      << args[0]->GetType()->TypeName();
  INTERNAL_CHECK_SPAN(arg_type->dtype_ == DataType::TASK_ID, args[0]->span_)
      << "Internal error: system.task_is_valid argument must be Scalar[TASK_ID], got "
      << arg_type->dtype_.ToString();
  (void)kwargs;
  return std::make_shared<ScalarType>(DataType::BOOL);
}

}  // namespace

// system.task_invalid — produces an invalid PTO2TaskId sentinel.
//
// Surfaced as the Python literal ``None`` in TaskId-typed positions: a
// ``prev_tid = None`` loop-carry seed or a ``deps=[None]`` entry. At codegen
// time it lowers to ``PTO2TaskId::invalid()``; downstream
// ``set_dependencies`` calls skip invalid entries via an ``is_valid()``
// guard so the runtime sees no edge on the first iteration.
REGISTER_OP("system.task_invalid")
    .set_description("Construct an invalid PTO2TaskId sentinel for manual_scope dep carries")
    .set_op_category("TaskOp")
    .no_argument()
    .f_deduce_type(DeduceTaskIdScalarType);

// system.task_is_valid — boolean predicate over a Scalar[TASK_ID]. Returns
// true when the task id refers to a real dispatched task and false for the
// sentinel produced by ``system.task_invalid()``.
//
// Used by manual_scope phase-fence lowering to guard each ``add_dep`` on an
// array-carry slot: a first-iteration init slot still holds the invalid
// sentinel, and the runtime must not see an edge to it. The IR-explicit
// guard makes the codegen a thin emitter — see ``ExpandManualPhaseFence``.
//
// Codegen lowers ``b = task_is_valid(t)`` to ``bool b = t.is_valid();``.
REGISTER_OP("system.task_is_valid")
    .set_description("Predicate: returns true when the Scalar[TASK_ID] refers to a real task")
    .set_op_category("TaskOp")
    .add_argument("task_id", "Scalar[TASK_ID] to test")
    .f_deduce_type(DeduceBoolScalarType);

}  // namespace ir
}  // namespace pypto

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

/**
 * @file array_op_codegen.cpp
 * @brief Codegen for ArrayType operations.
 *
 * ArrayType lives on the on-core scalar register file / C stack. Codegen
 * lowers to a bare C-style stack array ``dtype name[N]`` — no STL types,
 * matching the device CPU runtime which compiles as C (not C++ with STL).
 *
 * The SSA-functional update semantics in the IR are realized at codegen time
 * by aliasing the LHS Var of an ``array.update_element`` AssignStmt to the
 * input array's emit name (similar to how ``tensor.assemble`` aliases its
 * target — see HandleTensorAssembleAssign in orchestration_codegen.cpp).
 * Because the LHS resolves to the SAME C variable as the input, the emitted
 * write ``arr[i] = v;`` mutates the original storage in place — no array
 * copy is ever required, which is exactly what C-style arrays demand.
 *
 * For v1 the alias plumbing lives in orchestration_codegen.cpp's dispatch;
 * this file only emits the actual array operations.
 */

#include <sstream>
#include <string>

#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/orchestration_op_registry.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

REGISTER_ORCHESTRATION_OP(array_create, ("array.create")) {
  // array.create(extent) -> ArrayType. Emit a stack-local C-style array
  // declaration: ``dtype name[N] = {0};``.
  CHECK(op->args_.size() == 1) << "array.create requires 1 argument (extent)";
  auto array_type = As<ArrayType>(op->GetType());
  CHECK(array_type) << "array.create must return ArrayType";

  auto extent_const = As<ConstInt>(array_type->extent());
  CHECK(extent_const) << "array.create extent must be a compile-time ConstInt";

  std::string result_var = codegen.GetCurrentResultTarget();
  // TASK_ID is opaque, not numeric — emit ``PTO2TaskId`` rather than letting
  // ``DataType::ToCTypeString`` fall through to its "unknown" default.
  const bool is_task_id = array_type->dtype_ == DataType::TASK_ID;
  std::string cpp_type = is_task_id ? "PTO2TaskId" : array_type->dtype_.ToCTypeString();
  const int64_t N = extent_const->value_;

  std::ostringstream oss;
  if (is_task_id) {
    // ``PTO2TaskId`` is not a plain integer — its "invalid" sentinel is
    // ``PTO2TaskId::invalid()``, which is NOT bit-zero. Zero-initializing
    // would silently mark every slot as a real "task id 0" reference,
    // causing the runtime fence to wait on a bogus dep on the first
    // iteration. Explicitly fill with the invalid sentinel. The
    // declaration + broadcast loop emit as one logical line so the
    // orchestration codegen's single-line indent works correctly.
    oss << cpp_type << " " << result_var << "[" << N << "]; "
        << "for (int64_t __init_i = 0; __init_i < " << N << "; ++__init_i) " << result_var
        << "[__init_i] = PTO2TaskId::invalid();";
  } else {
    // Numeric integer / BOOL: ``= {0}`` zero-initializes the whole array.
    // Avoid ``std::array`` so the generated code stays on the device CPU's
    // C compiler path with no STL dependency.
    oss << cpp_type << " " << result_var << "[" << N << "] = {0};";
  }
  return oss.str();
}

REGISTER_ORCHESTRATION_OP(array_get_element, ("array.get_element")) {
  // array.get_element(array, index) -> ScalarType. Emit ``dtype tmp = arr[i];``.
  CHECK(op->args_.size() == 2) << "array.get_element requires 2 arguments";

  std::string array_name = codegen.GenerateExprString(op->args_[0]);
  std::string index_expr = codegen.GenerateExprString(op->args_[1]);

  auto result_type = As<ScalarType>(op->GetType());
  CHECK(result_type) << "array.get_element must return ScalarType";
  // TASK_ID is opaque — same special case as ``array.create``.
  std::string cpp_type =
      result_type->dtype_ == DataType::TASK_ID ? "PTO2TaskId" : result_type->dtype_.ToCTypeString();

  std::string result_var = codegen.GetCurrentResultTarget();

  std::ostringstream oss;
  oss << cpp_type << " " << result_var << " = " << array_name << "[" << index_expr << "];";
  return oss.str();
}

REGISTER_ORCHESTRATION_OP(array_update_element, ("array.update_element")) {
  // array.update_element(array, index, value) -> ArrayType.
  // Lowered to an in-place write. The orchestration codegen's AssignStmt
  // dispatch aliases the LHS to the input array's emit name BEFORE invoking
  // this handler, so writes to the result variable land on the same storage.
  CHECK(op->args_.size() == 3) << "array.update_element requires 3 arguments";

  std::string array_name = codegen.GenerateExprString(op->args_[0]);
  std::string index_expr = codegen.GenerateExprString(op->args_[1]);
  std::string value_expr = codegen.GenerateExprString(op->args_[2]);

  std::ostringstream oss;
  oss << array_name << "[" << index_expr << "] = " << value_expr << ";";
  return oss.str();
}

}  // namespace codegen
}  // namespace pypto

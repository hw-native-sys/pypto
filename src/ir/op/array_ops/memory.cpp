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
 * @file memory.cpp
 * @brief ArrayType operations: create, get_element, update_element.
 *
 * Mirrors the structure of src/ir/op/tensor_ops/memory.cpp but for the
 * ArrayType — a small on-core fixed-size 1-D homogeneous array (lives on
 * the scalar register file / C stack, never GM).
 *
 * Writes are SSA-functional: ``array.update_element(arr, i, v)`` returns a
 * new SSA value of ``ArrayType`` representing "arr with element i set to v"
 * — semantically equivalent to ``tensor.assemble``. The codegen lowers a
 * chain of update_element Calls back to in-place C-stack mutation.
 */

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

TypePtr DeduceArrayCreateType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // array.create(extent) -> ArrayType
  // dtype is provided via kwargs.
  CHECK(args.size() == 1) << "array.create requires exactly 1 argument (extent), got " << args.size();

  bool found_dtype = false;
  DataType dtype;
  for (const auto& [key, value] : kwargs) {
    if (key == "dtype") {
      dtype = AnyCast<DataType>(value, "kwarg key: dtype");
      found_dtype = true;
      break;
    }
  }
  CHECK(found_dtype) << "array.create requires 'dtype' kwarg";
  // TASK_ID is admitted alongside integer / BOOL: it is an opaque 64-bit
  // scalar used as a fence companion in manual_scope lowering, and lowers to
  // the same on-core C-stack array as integer dtypes.
  CHECK(dtype.IsInt() || dtype == DataType::BOOL || dtype == DataType::TASK_ID)
      << "array.create dtype must be integer, BOOL, or TASK_ID, got " << dtype.ToString();

  auto extent_const = As<ConstInt>(args[0]);
  CHECK(extent_const) << "array.create extent must be a compile-time ConstInt, got " << args[0]->TypeName();
  CHECK(extent_const->value_ > 0) << "array.create extent must be positive, got " << extent_const->value_;

  return std::make_shared<ArrayType>(dtype, args[0]);
}

TypePtr DeduceArrayGetElementType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // array.get_element(array, index) -> ScalarType
  CHECK(args.size() == 2) << "array.get_element requires exactly 2 arguments (array, index), got "
                          << args.size();

  auto array_type = As<ArrayType>(args[0]->GetType());
  CHECK(array_type) << "array.get_element first argument must be ArrayType, got "
                    << args[0]->GetType()->TypeName();

  auto index_type = As<ScalarType>(args[1]->GetType());
  CHECK(index_type) << "array.get_element index must be ScalarType, got " << args[1]->GetType()->TypeName();
  CHECK(index_type->dtype_.IsInt()) << "array.get_element index must have integer dtype, got "
                                    << index_type->dtype_.ToString();

  (void)kwargs;
  return std::make_shared<ScalarType>(array_type->dtype_);
}

TypePtr DeduceArrayUpdateElementType(const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // array.update_element(array, index, value) -> ArrayType (SSA-functional: returns
  // a new SSA value of the same ArrayType representing the updated array).
  CHECK(args.size() == 3) << "array.update_element requires exactly 3 arguments (array, index, value), got "
                          << args.size();

  auto array_type = As<ArrayType>(args[0]->GetType());
  CHECK(array_type) << "array.update_element first argument must be ArrayType, got "
                    << args[0]->GetType()->TypeName();

  auto index_type = As<ScalarType>(args[1]->GetType());
  CHECK(index_type) << "array.update_element index must be ScalarType, got "
                    << args[1]->GetType()->TypeName();
  CHECK(index_type->dtype_.IsInt()) << "array.update_element index must have integer dtype, got "
                                    << index_type->dtype_.ToString();

  auto value_type = As<ScalarType>(args[2]->GetType());
  CHECK(value_type) << "array.update_element value must be ScalarType, got "
                    << args[2]->GetType()->TypeName();
  // INDEX is a semantic alias for a machine-word integer used by loop variables
  // and addressing. Treat it as compatible with any integer array dtype on either
  // side — this matches the parser's _dtypes_compatible convention and lets users
  // write `arr[i] = i` where `i` comes from `pl.range(N)`.
  const bool dtype_ok = value_type->dtype_ == array_type->dtype_ ||
                        (value_type->dtype_ == DataType::INDEX && array_type->dtype_.IsInt()) ||
                        (array_type->dtype_ == DataType::INDEX && value_type->dtype_.IsInt());
  CHECK(dtype_ok) << "array.update_element value dtype (" << value_type->dtype_.ToString()
                  << ") must match array dtype (" << array_type->dtype_.ToString() << ")";

  (void)kwargs;
  // Functional update — return a *new* ArrayType instance with the same shape/dtype
  // (matching the tensor.assemble pattern: fresh type per call keeps SSA-style identity).
  return std::make_shared<ArrayType>(array_type->dtype_, array_type->extent());
}

REGISTER_OP("array.create")
    .set_op_category("ArrayOp")
    .set_description("Allocate an on-core array (C-stack local). 1-D, fixed extent, integer dtype.")
    .add_argument("extent", "Number of elements (ConstInt, positive)")
    .set_attr<DataType>("dtype")
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceArrayCreateType(args, kwargs);
    });

REGISTER_OP("array.get_element")
    .set_op_category("ArrayOp")
    .set_description("Read an element from an array at the given index")
    .add_argument("array", "Source array (ArrayType)")
    .add_argument("index", "Element index (ScalarType, integer)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceArrayGetElementType(args, kwargs);
    });

REGISTER_OP("array.update_element")
    .set_op_category("ArrayOp")
    .set_description(
        "Functional update: return a new ArrayType value with element at the given "
        "index replaced by value (SSA-pure, no in-place mutation in the IR).")
    .add_argument("array", "Source array (ArrayType)")
    .add_argument("index", "Element index (ScalarType, integer)")
    .add_argument("value", "Replacement value (ScalarType, dtype must match array)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceArrayUpdateElementType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto

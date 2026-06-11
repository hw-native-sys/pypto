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
 * @file collective.cpp
 * @brief Distributed tensor collective ops.
 */

#include <any>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

void CheckReduceOp(int op_value, const std::string& op_name) {
  CHECK(op_value == static_cast<int>(ReduceOp::kSum))
      << op_name << " op must be ReduceOp.Sum (got int " << op_value << ")";
}

void CheckSupportedBuiltinVariant(int op_value, DataType dtype, const std::string& op_name) {
  CheckReduceOp(op_value, op_name);
  CHECK(dtype == DataType::FP32) << op_name << " currently supports only (op=ReduceOp.Sum, dtype=FP32); got "
                                 << "(op=ReduceOp.Sum, dtype=" << dtype.ToString() << ")";
}

TypePtr DeduceBuiltinTensorAllReduceType(const std::vector<ExprPtr>& args,
                                         const std::vector<std::pair<std::string, std::any>>& kwargs) {
  constexpr const char* kOpName = "builtin.tensor.allreduce";
  CHECK(args.size() == 2) << kOpName << " requires exactly 2 positional arguments (src, signal), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << kOpName << " positional argument #" << i << " must not be null";
  }

  auto src_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(src_type) << kOpName << " src must be a DistributedTensor, got " << args[0]->GetType()->TypeName();
  auto signal_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(signal_type) << kOpName << " signal must be a DistributedTensor, got "
                     << args[1]->GetType()->TypeName();
  CHECK(signal_type->dtype_ == DataType::INT32)
      << kOpName << " signal dtype must be INT32, got " << signal_type->dtype_.ToString();
  CHECK(signal_type->shape_.size() == 1)
      << kOpName << " signal must be a rank-1 DistributedTensor, got rank " << signal_type->shape_.size();

  auto op_value = GetRequiredKwarg<int>(kwargs, "op", kOpName);
  auto dtype = GetRequiredKwarg<DataType>(kwargs, "dtype", kOpName);
  CHECK(dtype == src_type->dtype_) << kOpName << " dtype kwarg (" << dtype.ToString()
                                   << ") must match src dtype (" << src_type->dtype_.ToString() << ")";
  CheckSupportedBuiltinVariant(op_value, dtype, kOpName);
  return args[0]->GetType();
}

}  // namespace

REGISTER_OP("builtin.tensor.allreduce")
    .set_description("Internal chip-dispatch builtin for pld.tensor.allreduce.")
    .set_op_category("DistributedOp")
    .add_argument("src", "Window-bound DistributedTensor to reduce in place")
    .add_argument("signal", "Window-bound INT32 DistributedTensor signal buffer")
    .set_attr<int>("op")
    .set_attr<DataType>("dtype")
    .no_memory_spec()
    .set_internal_only(true)
    .set_template_dir(":pypto.runtime.builtins.collectives.allreduce")
    .f_deduce_type(DeduceBuiltinTensorAllReduceType);

}  // namespace ir
}  // namespace pypto

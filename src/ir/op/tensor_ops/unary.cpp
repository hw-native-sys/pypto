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
 * @file unary.cpp
 * @brief Unary tensor operations (neg, recip, exp, log, sqrt, rsqrt, cast, abs, sin, cos)
 *
 * This file implements unary operations for tensors that operate element-wise.
 */

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
namespace pypto {
namespace ir {

TypePtr DeduceTensorNegType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "tensor.neg requires exactly 1 argument, but got " << args.size();

  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type)
      << "tensor.neg requires first argument to be a TensorType or DistributedTensorType, but got "
      << args[0]->GetType()->TypeName();

  // Negation preserves dtype (valid for both int and float)
  // Preserve the effective valid box on the fresh result without copying source
  // allocation metadata such as strides, layout, or padding policy.
  return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, std::nullopt,
                                      MakeFreshTensorResultView(GetValidShape(tensor_type)));
}

TypePtr DeduceTensorAbsType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "tensor.abs requires exactly 1 argument, but got " << args.size();

  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type)
      << "tensor.abs requires first argument to be a TensorType or DistributedTensorType, but got "
      << args[0]->GetType()->TypeName();

  // Absolute value preserves dtype (valid for both int and float)
  // Preserve the effective valid box on the fresh result without copying source
  // allocation metadata such as strides, layout, or padding policy.
  return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, std::nullopt,
                                      MakeFreshTensorResultView(GetValidShape(tensor_type)));
}

TypePtr DeduceTensorRecipType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "tensor.recip requires exactly 1 argument, but got " << args.size();

  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type)
      << "tensor.recip requires first argument to be a TensorType or DistributedTensorType, but got "
      << args[0]->GetType()->TypeName();

  // Reciprocal (1/x) always produces floating-point output
  DataType out_dtype = tensor_type->dtype_;
  if (!out_dtype.IsFloat()) {
    out_dtype = DataType::FP32;
  }

  // Dtype may promote (e.g. int -> FP32), but only the effective valid box is
  // semantic for the fresh result; source allocation metadata is not inherited.
  return std::make_shared<TensorType>(tensor_type->shape_, out_dtype, std::nullopt,
                                      MakeFreshTensorResultView(GetValidShape(tensor_type)));
}

TypePtr DeduceTensorExpType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "tensor.exp requires exactly 1 argument, but got " << args.size();

  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type)
      << "tensor.exp requires first argument to be a TensorType or DistributedTensorType, but got "
      << args[0]->GetType()->TypeName();

  // exp should promote to float type if input is integer
  // Exponential always produces floating-point output (e.g., exp(1) = 2.718...)
  DataType out_dtype = tensor_type->dtype_;
  if (!out_dtype.IsFloat()) {
    // Promote to default float type (FP32)
    out_dtype = DataType::FP32;
  }

  // Dtype may promote (e.g. int -> FP32), but only the effective valid box is
  // semantic for the fresh result; source allocation metadata is not inherited.
  return std::make_shared<TensorType>(tensor_type->shape_, out_dtype, std::nullopt,
                                      MakeFreshTensorResultView(GetValidShape(tensor_type)));
}

TypePtr DeduceTensorLogType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "tensor.log requires exactly 1 argument, but got " << args.size();

  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type)
      << "tensor.log requires first argument to be a TensorType or DistributedTensorType, but got "
      << args[0]->GetType()->TypeName();

  // log always produces floating-point output (e.g., log(1) = 0.0).
  // Promote integer inputs to FP32; preserve existing float dtype.
  DataType out_dtype = tensor_type->dtype_;
  if (!out_dtype.IsFloat()) {
    out_dtype = DataType::FP32;
  }

  // Dtype may promote (e.g. int -> FP32), but only the effective valid box is
  // semantic for the fresh result; source allocation metadata is not inherited.
  return std::make_shared<TensorType>(tensor_type->shape_, out_dtype, std::nullopt,
                                      MakeFreshTensorResultView(GetValidShape(tensor_type)));
}

TypePtr DeduceTensorSqrtType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "tensor.sqrt requires exactly 1 argument, but got " << args.size();

  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type)
      << "tensor.sqrt requires first argument to be a TensorType or DistributedTensorType, but got "
      << args[0]->GetType()->TypeName();

  // sqrt should promote to float type if input is integer
  // Square root always produces floating-point output
  DataType out_dtype = tensor_type->dtype_;
  if (!out_dtype.IsFloat()) {
    out_dtype = DataType::FP32;
  }

  // Dtype may promote (e.g. int -> FP32), but only the effective valid box is
  // semantic for the fresh result; source allocation metadata is not inherited.
  return std::make_shared<TensorType>(tensor_type->shape_, out_dtype, std::nullopt,
                                      MakeFreshTensorResultView(GetValidShape(tensor_type)));
}

TypePtr DeduceTensorRsqrtType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "tensor.rsqrt requires exactly 1 argument, but got " << args.size();

  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type)
      << "tensor.rsqrt requires first argument to be a TensorType or DistributedTensorType, but got "
      << args[0]->GetType()->TypeName();

  // rsqrt always produces floating-point output
  DataType out_dtype = tensor_type->dtype_;
  if (!out_dtype.IsFloat()) {
    out_dtype = DataType::FP32;
  }

  // Dtype may promote (e.g. int -> FP32), but only the effective valid box is
  // semantic for the fresh result; source allocation metadata is not inherited.
  return std::make_shared<TensorType>(tensor_type->shape_, out_dtype, std::nullopt,
                                      MakeFreshTensorResultView(GetValidShape(tensor_type)));
}

// Shared FP32-only deducer for transcendental ops (tensor.sin, tensor.cos).
// These ops are intentionally FP32-only to avoid silent precision loss; callers
// must explicitly cast non-FP32 inputs via tensor.cast.
TypePtr DeduceTensorFP32OnlyType(const std::string& op_name, const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << op_name << " requires exactly 1 argument, but got " << args.size();

  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type) << op_name
                     << " requires first argument to be a TensorType or DistributedTensorType, but got "
                     << args[0]->GetType()->TypeName();

  // FP32-only: do NOT auto-promote. Reject non-FP32 inputs with an actionable error.
  CHECK(tensor_type->dtype_ == DataType::FP32)
      << op_name << " is FP32-only, but got input with dtype " << tensor_type->dtype_.ToString()
      << ". Cast the input to FP32 explicitly via pl.cast(x, pl.FP32) before applying " << op_name << ".";

  // Preserve the effective valid box on the fresh result without copying source
  // allocation metadata such as strides, layout, or padding policy.
  return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, std::nullopt,
                                      MakeFreshTensorResultView(GetValidShape(tensor_type)));
}

TypePtr DeduceTensorCastType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "tensor.cast requires exactly 1 argument (input), but got " << args.size();

  // ``AsTensorTypeLike`` accepts a ``DistributedTensorType`` (window) slice the
  // same as a plain tensor (issue #1694): an elementwise op reads its window
  // input as this rank's local GM and writes fresh local data — so the result
  // is a plain ``TensorType`` (a cast is not a view into the window).
  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type) << "tensor.cast requires first argument to be a TensorType or DistributedTensorType, "
                        "but got "
                     << args[0]->GetType()->TypeName();

  // Read target_type from kwargs
  bool found_target_type = false;
  DataType target_dtype;
  for (const auto& [key, value] : kwargs) {
    if (key == "target_type") {
      // Handle both DataType and int for backward compatibility
      if (value.type() == typeid(DataType)) {
        target_dtype = AnyCast<DataType>(value, "kwarg key: target_type");
      } else if (value.type() == typeid(int)) {
        target_dtype = static_cast<DataType>(AnyCast<int>(value, "kwarg key: target_type"));
      } else {
        throw TypeError("target_type must be a DataType or int, but got " + std::string(value.type().name()));
      }
      found_target_type = true;
      break;
    }
  }
  CHECK(found_target_type) << "tensor.cast requires 'target_type' kwarg";

  // Reject same-dtype cast: the hardware pto.tcvt instruction is for
  // cross-dtype conversion, and a same-dtype invocation can corrupt values
  // rather than acting as an identity copy. Detecting this at construction
  // time keeps malformed casts out of every downstream pass and codegen.
  CHECK(tensor_type->dtype_ != target_dtype)
      << "tensor.cast: target_type " << target_dtype.ToString()
      << " equals input dtype; same-dtype cast is not a valid operation. "
      << "Remove the cast or use a different target_type.";

  // mode kwarg is optional, not used in type deduction

  // Cast preserves shape and effective valid extent, but it materializes fresh
  // storage and therefore does not inherit source addressing or padding metadata.
  return std::make_shared<TensorType>(tensor_type->shape_, target_dtype, std::nullopt,
                                      MakeFreshTensorResultView(GetValidShape(tensor_type)));
}

// ============================================================================
// Registration Function for Tensor Unary Operations
// ============================================================================

REGISTER_OP("tensor.neg")
    .set_op_category("TensorOp")
    .set_description("Element-wise negation operation")
    .add_argument("input", "Input tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorNegType(args, kwargs);
    });

REGISTER_OP("tensor.abs")
    .set_op_category("TensorOp")
    .set_description("Element-wise absolute value operation")
    .add_argument("input", "Input tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorAbsType(args, kwargs);
    });

REGISTER_OP("tensor.recip")
    .set_op_category("TensorOp")
    .set_description("Element-wise reciprocal (1/x) operation")
    .add_argument("input", "Input tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorRecipType(args, kwargs);
    });

REGISTER_OP("tensor.exp")
    .set_op_category("TensorOp")
    .set_description("Element-wise exponential operation")
    .add_argument("input", "Input tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorExpType(args, kwargs);
    });

REGISTER_OP("tensor.log")
    .set_op_category("TensorOp")
    .set_description("Element-wise natural logarithm operation")
    .add_argument("input", "Input tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorLogType(args, kwargs);
    });

REGISTER_OP("tensor.sin")
    .set_op_category("TensorOp")
    .set_description("Element-wise sine operation (radians). FP32-only.")
    .add_argument("input", "Input tensor (TensorType, FP32)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorFP32OnlyType("tensor.sin", args, kwargs);
    });

REGISTER_OP("tensor.cos")
    .set_op_category("TensorOp")
    .set_description("Element-wise cosine operation (radians). FP32-only.")
    .add_argument("input", "Input tensor (TensorType, FP32)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorFP32OnlyType("tensor.cos", args, kwargs);
    });

REGISTER_OP("tensor.sqrt")
    .set_op_category("TensorOp")
    .set_description("Element-wise square root operation")
    .add_argument("input", "Input tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorSqrtType(args, kwargs);
    });

REGISTER_OP("tensor.rsqrt")
    .set_op_category("TensorOp")
    .set_description(
        "Element-wise reciprocal square root operation. "
        "Passing high_precision=True opts into the higher-precision PTO path that uses a scratch buffer.")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<bool>("high_precision")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorRsqrtType(args, kwargs);
    });

REGISTER_OP("tensor.cast")
    .set_op_category("TensorOp")
    .set_description("Type casting operation")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<DataType>("target_type")
    .set_attr<int>("mode")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorCastType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto

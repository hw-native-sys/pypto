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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

template <typename T>
T GetKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
           const std::optional<T>& default_value = std::nullopt) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<T>(v, "kwarg key: " + key);
    }
  }
  if (default_value) {
    return *default_value;
  }
  throw ValueError("Missing kwarg: " + key);
}

TypePtr DeduceTensorElementwiseBinaryType(const std::vector<ExprPtr>& args,
                                          const std::string& op_name, bool require_int = false) {
  CHECK(args.size() == 2) << op_name << " requires exactly 2 arguments, but got " << args.size();

  auto lhs_type = As<TensorType>(args[0]->GetType());
  auto rhs_type = As<TensorType>(args[1]->GetType());
  CHECK(lhs_type) << op_name << " requires first argument to be a TensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << op_name << " requires second argument to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();

  if (require_int) {
    CHECK(lhs_type->dtype_.IsInt()) << op_name << " requires integer tensor dtype, but got "
                                    << lhs_type->dtype_.ToString();
    CHECK(rhs_type->dtype_.IsInt()) << op_name << " requires integer tensor dtype, but got "
                                    << rhs_type->dtype_.ToString();
  }

  auto result_dtype = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
  CHECK(result_dtype) << op_name << " requires compatible data types, but got "
                      << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();

  auto broadcast_result = BroadcastShapes(lhs_type->shape_, rhs_type->shape_);
  CHECK(broadcast_result.success) << op_name << " requires compatible shapes, but got "
                                  << FormatShape(lhs_type->shape_) << " and "
                                  << FormatShape(rhs_type->shape_);
  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

TypePtr DeduceTensorScalarBinaryType(const std::vector<ExprPtr>& args,
                                     const std::string& op_name, bool require_int_scalar = false) {
  CHECK(args.size() == 2) << op_name << " requires exactly 2 arguments, but got " << args.size();

  auto lhs_type = As<TensorType>(args[0]->GetType());
  auto rhs_type = As<ScalarType>(args[1]->GetType());
  CHECK(lhs_type) << op_name << " requires first argument to be a TensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << op_name << " requires second argument to be a ScalarType, but got "
                  << args[1]->GetType()->TypeName();

  if (require_int_scalar) {
    CHECK(lhs_type->dtype_.IsInt()) << op_name << " requires integer tensor dtype, but got "
                                    << lhs_type->dtype_.ToString();
    CHECK(rhs_type->dtype_.IsInt()) << op_name << " requires integer scalar dtype, but got "
                                    << rhs_type->dtype_.ToString();
  }

  auto result_dtype = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
  CHECK(result_dtype) << op_name << " requires compatible data types, but got "
                      << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();
  return std::make_shared<TensorType>(lhs_type->shape_, *result_dtype);
}

TypePtr DeduceTensorShiftBinaryType(const std::vector<ExprPtr>& args, const std::string& op_name) {
  auto type = DeduceTensorElementwiseBinaryType(args, op_name, true);
  auto tensor_type = As<TensorType>(type);
  CHECK(tensor_type) << op_name << " internal error: expected TensorType result";
  auto lhs_type = As<TensorType>(args[0]->GetType());
  return std::make_shared<TensorType>(tensor_type->shape_, lhs_type->dtype_);
}

TypePtr DeduceTensorTernaryType(const std::vector<ExprPtr>& args,
                                const std::string& op_name, bool require_int = false) {
  CHECK(args.size() == 3) << op_name << " requires exactly 3 arguments, but got " << args.size();

  auto lhs_type = As<TensorType>(args[0]->GetType());
  auto rhs_type = As<TensorType>(args[1]->GetType());
  auto third_type = As<TensorType>(args[2]->GetType());
  CHECK(lhs_type) << op_name << " requires first argument to be a TensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << op_name << " requires second argument to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(third_type) << op_name << " requires third argument to be a TensorType, but got "
                    << args[2]->GetType()->TypeName();

  if (require_int) {
    CHECK(lhs_type->dtype_.IsInt()) << op_name << " requires integer tensor dtype, but got "
                                    << lhs_type->dtype_.ToString();
    CHECK(rhs_type->dtype_.IsInt()) << op_name << " requires integer tensor dtype, but got "
                                    << rhs_type->dtype_.ToString();
  }

  auto result_dtype = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
  CHECK(result_dtype) << op_name << " requires compatible data types";
  auto broadcast_result = BroadcastShapes(lhs_type->shape_, rhs_type->shape_);
  CHECK(broadcast_result.success) << op_name << " requires compatible shapes";
  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

TypePtr DeduceTensorTriTensorType(const std::vector<ExprPtr>& args, const std::string& op_name) {
  CHECK(args.size() == 3) << op_name << " requires exactly 3 arguments, but got " << args.size();
  auto lhs_type = As<TensorType>(args[0]->GetType());
  auto rhs_type = As<TensorType>(args[1]->GetType());
  auto rhs2_type = As<TensorType>(args[2]->GetType());
  CHECK(lhs_type) << op_name << " requires first argument to be a TensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << op_name << " requires second argument to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(rhs2_type) << op_name << " requires third argument to be a TensorType, but got "
                   << args[2]->GetType()->TypeName();

  auto result_dtype = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
  CHECK(result_dtype) << op_name << " requires compatible data types";
  result_dtype = PromoteDataTypes(*result_dtype, rhs2_type->dtype_);
  CHECK(result_dtype) << op_name << " requires compatible data types";

  auto broadcast12 = BroadcastShapes(lhs_type->shape_, rhs_type->shape_);
  CHECK(broadcast12.success) << op_name << " requires compatible shapes";
  auto broadcast123 = BroadcastShapes(broadcast12.shape, rhs2_type->shape_);
  CHECK(broadcast123.success) << op_name << " requires compatible shapes";
  return std::make_shared<TensorType>(broadcast123.shape, *result_dtype);
}

TypePtr DeduceTensorTensorScalarTensorType(const std::vector<ExprPtr>& args, const std::string& op_name) {
  CHECK(args.size() == 3) << op_name << " requires exactly 3 arguments, but got " << args.size();
  auto lhs_type = As<TensorType>(args[0]->GetType());
  auto scalar_type = As<ScalarType>(args[1]->GetType());
  auto rhs2_type = As<TensorType>(args[2]->GetType());
  CHECK(lhs_type) << op_name << " requires first argument to be a TensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(scalar_type) << op_name << " requires second argument to be a ScalarType, but got "
                     << args[1]->GetType()->TypeName();
  CHECK(rhs2_type) << op_name << " requires third argument to be a TensorType, but got "
                   << args[2]->GetType()->TypeName();

  auto result_dtype = PromoteDataTypes(lhs_type->dtype_, scalar_type->dtype_);
  CHECK(result_dtype) << op_name << " requires compatible data types";
  result_dtype = PromoteDataTypes(*result_dtype, rhs2_type->dtype_);
  CHECK(result_dtype) << op_name << " requires compatible data types";

  auto broadcast_result = BroadcastShapes(lhs_type->shape_, rhs2_type->shape_);
  CHECK(broadcast_result.success) << op_name << " requires compatible shapes";
  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

TypePtr DeduceTensorSelType(const std::vector<ExprPtr>& args, const std::string& op_name) {
  CHECK(args.size() == 3) << op_name << " requires exactly 3 arguments, but got " << args.size();
  CHECK(As<TensorType>(args[0]->GetType())) << op_name
                                            << " requires first argument (mask) to be a TensorType, but got "
                                            << args[0]->GetType()->TypeName();
  auto lhs_type = As<TensorType>(args[1]->GetType());
  auto rhs_type = As<TensorType>(args[2]->GetType());
  CHECK(lhs_type) << op_name << " requires second argument to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(rhs_type) << op_name << " requires third argument to be a TensorType, but got "
                  << args[2]->GetType()->TypeName();

  auto result_dtype = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
  CHECK(result_dtype) << op_name << " requires compatible data types";
  auto broadcast_result = BroadcastShapes(lhs_type->shape_, rhs_type->shape_);
  CHECK(broadcast_result.success) << op_name << " requires compatible shapes";
  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

TypePtr DeduceTensorSelsType(const std::vector<ExprPtr>& args, const std::string& op_name) {
  CHECK(args.size() == 3) << op_name << " requires exactly 3 arguments, but got " << args.size();
  auto lhs_type = As<TensorType>(args[0]->GetType());
  auto rhs_type = As<TensorType>(args[1]->GetType());
  CHECK(lhs_type) << op_name << " requires first argument to be a TensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << op_name << " requires second argument to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(As<ScalarType>(args[2]->GetType())) << op_name
                                            << " requires third argument to be a ScalarType, but got "
                                            << args[2]->GetType()->TypeName();

  auto result_dtype = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
  CHECK(result_dtype) << op_name << " requires compatible data types";
  auto broadcast_result = BroadcastShapes(lhs_type->shape_, rhs_type->shape_);
  CHECK(broadcast_result.success) << op_name << " requires compatible shapes";
  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

TypePtr DeduceTensorCmpType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name, bool is_scalar_rhs) {
  (void)GetKwarg<int>(kwargs, "cmp_type");
  if (is_scalar_rhs) {
    return DeduceTensorScalarBinaryType(args, op_name);
  }
  return DeduceTensorElementwiseBinaryType(args, op_name);
}

TypePtr DeduceTensorFullType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 2) << "tensor.full requires exactly 2 arguments (shape, value), but got " << args.size();
  auto shape_tuple_type = As<TupleType>(args[0]->GetType());
  CHECK(shape_tuple_type) << "tensor.full requires shape to be TupleType, but got "
                          << args[0]->GetType()->TypeName();

  for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(shape_tuple_type->types_[i]);
    CHECK(scalar_type) << "tensor.full shape tuple element " << i << " must be ScalarType, but got "
                       << shape_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tensor.full shape tuple element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  DataType dtype = GetKwarg<DataType>(kwargs, "dtype");
  std::vector<ExprPtr> shape;
  shape.reserve(shape_tuple_type->types_.size());
  if (auto make_tuple = As<MakeTuple>(args[0])) {
    shape = make_tuple->elements_;
  } else {
    for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
      shape.emplace_back(std::make_shared<TupleGetItemExpr>(args[0], static_cast<int>(i), args[0]->span_));
    }
  }
  return std::make_shared<TensorType>(shape, dtype);
}

REGISTER_OP("tensor.rem")
    .set_op_category("TensorOp")
    .set_description("Element-wise remainder of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorElementwiseBinaryType(args, "tensor.rem");
    });

REGISTER_OP("tensor.adds")
    .set_op_category("TensorOp")
    .set_description("Element-wise addition of tensor and scalar")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorScalarBinaryType(args, "tensor.adds");
    });

REGISTER_OP("tensor.subs")
    .set_op_category("TensorOp")
    .set_description("Element-wise subtraction of tensor and scalar")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorScalarBinaryType(args, "tensor.subs");
    });

REGISTER_OP("tensor.muls")
    .set_op_category("TensorOp")
    .set_description("Element-wise multiplication of tensor and scalar")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorScalarBinaryType(args, "tensor.muls");
    });

REGISTER_OP("tensor.divs")
    .set_op_category("TensorOp")
    .set_description("Element-wise division of tensor and scalar")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorScalarBinaryType(args, "tensor.divs");
    });

REGISTER_OP("tensor.rems")
    .set_op_category("TensorOp")
    .set_description("Element-wise remainder of tensor and scalar")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorScalarBinaryType(args, "tensor.rems");
    });

REGISTER_OP("tensor.shl")
    .set_op_category("TensorOp")
    .set_description("Element-wise left shift of two integer tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorShiftBinaryType(args, "tensor.shl");
    });

REGISTER_OP("tensor.shls")
    .set_op_category("TensorOp")
    .set_description("Element-wise left shift of integer tensor by integer scalar")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorScalarBinaryType(args, "tensor.shls", true);
    });

REGISTER_OP("tensor.shr")
    .set_op_category("TensorOp")
    .set_description("Element-wise right shift of two integer tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorShiftBinaryType(args, "tensor.shr");
    });

REGISTER_OP("tensor.shrs")
    .set_op_category("TensorOp")
    .set_description("Element-wise right shift of integer tensor by integer scalar")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorScalarBinaryType(args, "tensor.shrs", true);
    });

REGISTER_OP("tensor.maxs")
    .set_op_category("TensorOp")
    .set_description("Element-wise maximum of tensor and scalar")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorScalarBinaryType(args, "tensor.maxs");
    });

REGISTER_OP("tensor.mins")
    .set_op_category("TensorOp")
    .set_description("Element-wise minimum of tensor and scalar")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorScalarBinaryType(args, "tensor.mins");
    });

REGISTER_OP("tensor.and")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise and of two integer tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorElementwiseBinaryType(args, "tensor.and", true);
    });

REGISTER_OP("tensor.ands")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise and of integer tensor and integer scalar")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorScalarBinaryType(args, "tensor.ands", true);
    });

REGISTER_OP("tensor.or")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise or of two integer tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorElementwiseBinaryType(args, "tensor.or", true);
    });

REGISTER_OP("tensor.ors")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise or of integer tensor and integer scalar")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorScalarBinaryType(args, "tensor.ors", true);
    });

REGISTER_OP("tensor.xor")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise xor of two integer tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .add_argument("tmp", "Temporary tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorTernaryType(args, "tensor.xor", true);
    });

REGISTER_OP("tensor.xors")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise xor of integer tensor and integer scalar")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("tmp", "Temporary tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorTensorScalarTensorType(args, "tensor.xors");
    });

REGISTER_OP("tensor.prelu")
    .set_op_category("TensorOp")
    .set_description("Element-wise parametric relu")
    .add_argument("lhs", "Input tensor (TensorType)")
    .add_argument("rhs", "Slope tensor (TensorType)")
    .add_argument("tmp", "Temporary tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorTernaryType(args, "tensor.prelu");
    });

REGISTER_OP("tensor.addc")
    .set_op_category("TensorOp")
    .set_description("Element-wise add of three tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .add_argument("rhs2", "Third tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorTriTensorType(args, "tensor.addc");
    });

REGISTER_OP("tensor.subc")
    .set_op_category("TensorOp")
    .set_description("Element-wise subtract of three tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .add_argument("rhs2", "Third tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorTriTensorType(args, "tensor.subc");
    });

REGISTER_OP("tensor.addsc")
    .set_op_category("TensorOp")
    .set_description("Element-wise add of tensor, scalar, and tensor")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("rhs2", "Third tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorTensorScalarTensorType(args, "tensor.addsc");
    });

REGISTER_OP("tensor.subsc")
    .set_op_category("TensorOp")
    .set_description("Element-wise subtract of tensor, scalar, and tensor")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("rhs2", "Third tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorTensorScalarTensorType(args, "tensor.subsc");
    });

REGISTER_OP("tensor.lrelu")
    .set_op_category("TensorOp")
    .set_description("Element-wise leaky relu of tensor and scalar slope")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorScalarBinaryType(args, "tensor.lrelu");
    });

REGISTER_OP("tensor.sel")
    .set_op_category("TensorOp")
    .set_description("Per-element selection between two tensors using a mask tensor")
    .add_argument("mask", "Predicate mask tensor (TensorType)")
    .add_argument("lhs", "Source tensor 0 (TensorType)")
    .add_argument("rhs", "Source tensor 1 (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorSelType(args, "tensor.sel");
    });

REGISTER_OP("tensor.sels")
    .set_op_category("TensorOp")
    .set_description("Select between two tensors based on scalar mode")
    .add_argument("lhs", "Source tensor 0 (TensorType)")
    .add_argument("rhs", "Source tensor 1 (TensorType)")
    .add_argument("select_mode", "Scalar select mode (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorSelsType(args, "tensor.sels");
    });

REGISTER_OP("tensor.cmp")
    .set_op_category("TensorOp")
    .set_description("Element-wise comparison of two tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .set_attr<int>("cmp_type")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorCmpType(args, kwargs, "tensor.cmp", false);
    });

REGISTER_OP("tensor.cmps")
    .set_op_category("TensorOp")
    .set_description("Element-wise comparison of tensor and scalar")
    .add_argument("lhs", "Tensor (TensorType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_attr<int>("cmp_type")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorCmpType(args, kwargs, "tensor.cmps", true);
    });

REGISTER_OP("tensor.expands")
    .set_op_category("TensorOp")
    .set_description("Expand scalar to target tensor shape")
    .add_argument("target", "Target tensor defining output shape (TensorType)")
    .add_argument("scalar", "Scalar to expand (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceTensorScalarBinaryType(args, "tensor.expands");
    });

REGISTER_OP("tensor.full")
    .set_op_category("TensorOp")
    .set_description("Create a tensor of specified shape and filling value")
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(INT64))")
    .add_argument("value", "Filling value (ScalarType)")
    .set_attr<DataType>("dtype")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorFullType(args, kwargs);
    });

REGISTER_OP("tensor.fillpad")
    .set_op_category("TensorOp")
    .set_description("Fill destination tensor with source data and pad remaining elements")
    .add_argument("input", "Input tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      CHECK(args.size() == 1) << "tensor.fillpad requires exactly 1 argument, but got " << args.size();
      auto tensor_type = As<TensorType>(args[0]->GetType());
      CHECK(tensor_type) << "tensor.fillpad requires first argument to be a TensorType, but got "
                         << args[0]->GetType()->TypeName();
      return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_);
    });

}  // namespace ir
}  // namespace pypto

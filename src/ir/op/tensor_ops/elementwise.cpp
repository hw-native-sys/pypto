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
 * @file elementwise.cpp
 * @brief Element-wise tensor operations (Add, Sub, Mul, Div)
 *
 * This file implements element-wise tensor operations that support
 * N-dimensional tensors with NumPy-style broadcasting.
 */

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

static bool IsTDivDataType(DataType dtype) {
  return dtype == DataType::INT16 || dtype == DataType::INT32 || dtype == DataType::FP16 ||
         dtype == DataType::FP32;
}

static bool IsTSubsDataType(DataType dtype) {
  return dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
         dtype == DataType::FP16 || dtype == DataType::FP32 || dtype == DataType::BF16;
}

TypePtr DeduceTensorOpElementwiseBinaryType(
    const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
    const std::string& op_name, ElementwiseValidCombine combine = ElementwiseValidCombine::kAgree) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // ``AsTensorTypeLike`` accepts ``DistributedTensorType`` (window) operands the
  // same as plain tensors (issue #1694): an elementwise op reads a window as
  // this rank's local GM and writes fresh local data. The broadcast result is a
  // plain ``TensorType`` — the sum/product is new data, not a window view.
  auto tensor_type1 = AsTensorTypeLike(args[0]->GetType());
  auto tensor_type2 = AsTensorTypeLike(args[1]->GetType());

  CHECK(tensor_type1) << "The operator " << op_name
                      << " requires first argument to be a TensorType or DistributedTensorType, but got "
                      << args[0]->GetType()->TypeName();
  CHECK(tensor_type2) << "The operator " << op_name
                      << " requires second argument to be a TensorType or DistributedTensorType, but got "
                      << args[1]->GetType()->TypeName();

  auto result_dtype = PromoteDataTypes(tensor_type1->dtype_, tensor_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

  auto broadcast_result = BroadcastShapes(tensor_type1->shape_, tensor_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                  << FormatShape(tensor_type1->shape_) << " and "
                                  << FormatShape(tensor_type2->shape_);

  // Same rule as the tile level: the operands must agree on the region they are
  // combined over (or, for the partial-combine family, admit a representable
  // union). The sum is new data, so the result is a fresh tensor.
  std::vector<ExprPtr> valid_shape = InferElementwiseValidShape({
      /*operands=*/
      {{tensor_type1->shape_, GetValidShape(tensor_type1), "lhs"},
       {tensor_type2->shape_, GetValidShape(tensor_type2), "rhs"}},
      /*result_shape=*/broadcast_result.shape,
      /*combine=*/combine,
      /*op_name=*/op_name,
      /*span=*/args[0]->span_,
  });
  return MakeFreshTensorType(broadcast_result.shape, *result_dtype, std::move(valid_shape));
}

TypePtr DeduceTensorOpElementwiseScalarType(const std::vector<ExprPtr>& args,
                                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                                            const std::string& op_name, bool preserve_lhs_dtype = false) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  auto tensor_type1 = AsTensorTypeLike(args[0]->GetType());  // accepts a window (issue #1694)
  auto scalar_type2 = As<ScalarType>(args[1]->GetType());

  CHECK(tensor_type1) << "The operator " << op_name
                      << " requires first argument to be a TensorType or DistributedTensorType, but got "
                      << args[0]->GetType()->TypeName();
  CHECK(scalar_type2) << "The operator " << op_name
                      << " requires second argument to be a ScalarType, but got "
                      << args[1]->GetType()->TypeName();

  // Same fresh-result rule as the promoting path below; only the dtype differs.
  if (preserve_lhs_dtype) {
    return MakeFreshTensorType(tensor_type1->shape_, tensor_type1->dtype_, GetValidShape(tensor_type1));
  }

  // TensorType + ScalarType - result is TensorType with same shape as first argument
  auto result_dtype = PromoteDataTypes(tensor_type1->dtype_, scalar_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

  // The scalar has no region of its own, so the result keeps the tensor's: a
  // partial input must not widen back to the full allocation, which would make
  // padding indistinguishable from real data. Tile's equivalents (tile.adds and
  // friends) already do this; this is the tensor-level dual.
  return MakeFreshTensorType(tensor_type1->shape_, *result_dtype, GetValidShape(tensor_type1));
}

// ============================================================================
// Registration Function for Tensor Element-wise Operations
// ============================================================================

REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .set_description("Element-wise addition of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.add");
    });

REGISTER_OP("tensor.adds")
    .set_op_category("TensorOp")
    .set_description("Element-wise addition of tensor and scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.adds");
    });

REGISTER_OP("tensor.sub")
    .set_op_category("TensorOp")
    .set_description("Element-wise subtraction of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.sub");
    });

REGISTER_OP("tensor.subs")
    .set_op_category("TensorOp")
    .set_description("Element-wise subtraction of tensor and scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      auto result_type = DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.subs", true);
      auto tensor_type = AsTensorTypeLike(args[0]->GetType());
      auto scalar_type = As<ScalarType>(args[1]->GetType());
      CHECK(IsTSubsDataType(tensor_type->dtype_)) << "The operator tensor.subs requires tensor dtype in "
                                                     "{INT8, INT16, INT32, FP16, FP32, BF16}, but got "
                                                  << tensor_type->dtype_.ToString();
      CHECK(IsTSubsDataType(scalar_type->dtype_)) << "The operator tensor.subs requires scalar dtype in "
                                                     "{INT8, INT16, INT32, FP16, FP32, BF16}, but got "
                                                  << scalar_type->dtype_.ToString();
      return result_type;
    });

REGISTER_OP("tensor.mul")
    .set_op_category("TensorOp")
    .set_description("Element-wise multiplication of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.mul");
    });

REGISTER_OP("tensor.muls")
    .set_op_category("TensorOp")
    .set_description("Element-wise multiplication of tensor and scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.muls");
    });

REGISTER_OP("tensor.div")
    .set_op_category("TensorOp")
    .set_description("Element-wise division of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .set_attr<bool>("high_precision")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      auto result_type = DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.div");
      for (const auto& arg : args) {
        auto tensor_type = AsTensorTypeLike(arg->GetType());
        CHECK(IsTDivDataType(tensor_type->dtype_))
            << "The operator tensor.div requires operand dtype in {INT16, INT32, FP16, FP32}, but got "
            << tensor_type->dtype_.ToString();
      }
      auto result_tensor_type = As<TensorType>(result_type);
      CHECK(!GetKwargOr<bool>(kwargs, "high_precision", false) || result_tensor_type->dtype_.IsFloat())
          << "The operator tensor.div supports high_precision only for FP16 or FP32 because the PTOAS "
             "high-precision template does not implement integer division";
      return result_type;
    });

REGISTER_OP("tensor.divs")
    .set_op_category("TensorOp")
    .set_description("Element-wise division of tensor and scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.divs");
    });

// Partial-combine binary ops (tensor-tensor only; the hardware has no scalar
// form). At the tensor level the operands are fully valid, so these lower 1:1
// to the matching tile.part_* op where the partial valid-region semantics apply.
REGISTER_OP("tensor.part_add")
    .set_op_category("TensorOp")
    .set_description("Partial element-wise add of two tensors")
    .add_argument("src0", "First source tensor (TensorType)")
    .add_argument("src1", "Second source tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.part_add",
                                                 ElementwiseValidCombine::kUnion);
    });

REGISTER_OP("tensor.part_mul")
    .set_op_category("TensorOp")
    .set_description("Partial element-wise multiply of two tensors")
    .add_argument("src0", "First source tensor (TensorType)")
    .add_argument("src1", "Second source tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.part_mul",
                                                 ElementwiseValidCombine::kUnion);
    });

REGISTER_OP("tensor.part_max")
    .set_op_category("TensorOp")
    .set_description("Partial element-wise max of two tensors")
    .add_argument("src0", "First source tensor (TensorType)")
    .add_argument("src1", "Second source tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.part_max",
                                                 ElementwiseValidCombine::kUnion);
    });

REGISTER_OP("tensor.part_min")
    .set_op_category("TensorOp")
    .set_description("Partial element-wise min of two tensors")
    .add_argument("src0", "First source tensor (TensorType)")
    .add_argument("src1", "Second source tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.part_min",
                                                 ElementwiseValidCombine::kUnion);
    });

REGISTER_OP("tensor.fmod")
    .set_op_category("TensorOp")
    .set_description("Element-wise floating-point remainder of two tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.fmod");
    });

REGISTER_OP("tensor.fmods")
    .set_op_category("TensorOp")
    .set_description("Element-wise floating-point remainder of tensor and scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.fmods");
    });

REGISTER_OP("tensor.maximum")
    .set_op_category("TensorOp")
    .set_description("Element-wise maximum of tensor and tensor or scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType) or scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2) << "The operator tensor.maximum requires exactly 2 arguments, but got "
                              << args.size();
      if (AsTensorTypeLike(args[1]->GetType())) {  // window operand routes to binary path (issue #1694)
        return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.maximum");
      }
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.maximum");
    });

REGISTER_OP("tensor.minimum")
    .set_op_category("TensorOp")
    .set_description("Element-wise minimum of tensor and tensor or scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType) or scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2) << "The operator tensor.minimum requires exactly 2 arguments, but got "
                              << args.size();
      if (AsTensorTypeLike(args[1]->GetType())) {  // window operand routes to binary path (issue #1694)
        return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.minimum");
      }
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.minimum");
    });

REGISTER_OP("tensor.cmp")
    .set_op_category("TensorOp")
    .set_description("Element-wise comparison of tensor and tensor or scalar (returns 0/1 tensor)")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType) or scalar (ScalarType)")
    .set_attr<int>("cmp_type")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2) << "The operator tensor.cmp requires exactly 2 arguments, but got "
                              << args.size();
      if (AsTensorTypeLike(args[1]->GetType())) {  // window operand routes to binary path (issue #1694)
        return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.cmp");
      }
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.cmp");
    });

}  // namespace ir
}  // namespace pypto

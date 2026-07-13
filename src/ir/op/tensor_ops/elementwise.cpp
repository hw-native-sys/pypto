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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceTensorOpElementwiseBinaryType(const std::vector<ExprPtr>& args,
                                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                                            const std::string& op_name, bool partial_combine = false) {
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

  std::vector<ExprPtr> output_valid;
  if (partial_combine) {
    // Partial-combine ops (tensor.part_add/part_mul/part_max/part_min) copy whichever
    // source is valid where only one is, so the result is valid wherever EITHER source
    // is — the origin-anchored UNION of the two valid regions, NOT the strict agreement
    // of true elementwise ops. Mirror the tile-side part_* rule bit-for-bit: reuse the
    // assemble-union helper with a zero offset (it rejects a non-representable L-shaped
    // union per the North Star and is symmetric in the two operands so operand order
    // does not matter). A fully-valid pair unions back to out_shape, which TensorType's
    // ctor collapses to a bare result.
    const std::vector<ExprPtr> src0_valid = LiftValidShapeForBroadcast(
        tensor_type1->shape_, GetValidShape(tensor_type1), broadcast_result.shape, args[0]->span_, op_name);
    const std::vector<ExprPtr> src1_valid = LiftValidShapeForBroadcast(
        tensor_type2->shape_, GetValidShape(tensor_type2), broadcast_result.shape, args[0]->span_, op_name);
    const std::vector<ExprPtr> zero_offset(broadcast_result.shape.size(),
                                           std::make_shared<ConstInt>(0, DataType::INDEX, args[0]->span_));
    output_valid = ComputeAssembleUnionValidShape(src0_valid, src1_valid, zero_offset, broadcast_result.shape,
                                                  args[0]->span_, op_name);
  } else {
    // valid_shape agreement (valid_shape never broadcasts). Reuse the same
    // shared helper the tile-side elementwise deducers use so tensor and tile binaries
    // agree bit-for-bit: on a dim where the physical shapes coincide the operands' valid
    // extents must be equal; a shape-1 broadcast operand is exempt and the output takes
    // the non-broadcast operand's extent. A fully-valid pair yields
    // out_valid == out_shape, which TensorType's ctor collapses back to a bare result.
    std::vector<std::vector<ExprPtr>> operand_shapes = {tensor_type1->shape_, tensor_type2->shape_};
    std::vector<std::vector<ExprPtr>> operand_valids = {GetValidShape(tensor_type1),
                                                        GetValidShape(tensor_type2)};
    output_valid = ComputeBroadcastElementwiseValidShape(broadcast_result.shape, operand_shapes,
                                                         operand_valids, args[0]->span_, op_name);
  }
  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype, std::nullopt,
                                      MakeFreshTensorResultView(std::move(output_valid)));
}

TypePtr DeduceTensorOpElementwiseScalarType(const std::vector<ExprPtr>& args,
                                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                                            const std::string& op_name) {
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

  // TensorType + ScalarType - result is TensorType with same shape as first argument.
  auto result_dtype = PromoteDataTypes(tensor_type1->dtype_, scalar_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

  // The scalar has no valid region of its own, so preserve the tensor operand's
  // effective valid box. This is a fresh allocation, so do not copy the source's
  // alias-only strides/layout or padding policy.
  return std::make_shared<TensorType>(tensor_type1->shape_, *result_dtype, std::nullopt,
                                      MakeFreshTensorResultView(GetValidShape(tensor_type1)));
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
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.subs");
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
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.div");
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
// form). The result is valid wherever EITHER source is valid, so — exactly like
// the matching tile.part_* op — the valid region is the origin-anchored UNION of
// the two operands' regions, NOT the strict agreement of a true elementwise op
// (``partial_combine=true``). A fully-valid pair still yields a bare result.
REGISTER_OP("tensor.part_add")
    .set_op_category("TensorOp")
    .set_description("Partial element-wise add of two tensors")
    .add_argument("src0", "First source tensor (TensorType)")
    .add_argument("src1", "Second source tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.part_add",
                                                 /*partial_combine=*/true);
    });

REGISTER_OP("tensor.part_mul")
    .set_op_category("TensorOp")
    .set_description("Partial element-wise multiply of two tensors")
    .add_argument("src0", "First source tensor (TensorType)")
    .add_argument("src1", "Second source tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.part_mul",
                                                 /*partial_combine=*/true);
    });

REGISTER_OP("tensor.part_max")
    .set_op_category("TensorOp")
    .set_description("Partial element-wise max of two tensors")
    .add_argument("src0", "First source tensor (TensorType)")
    .add_argument("src1", "Second source tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.part_max",
                                                 /*partial_combine=*/true);
    });

REGISTER_OP("tensor.part_min")
    .set_op_category("TensorOp")
    .set_description("Partial element-wise min of two tensors")
    .add_argument("src0", "First source tensor (TensorType)")
    .add_argument("src1", "Second source tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.part_min",
                                                 /*partial_combine=*/true);
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

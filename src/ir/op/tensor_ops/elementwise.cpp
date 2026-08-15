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
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
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

/// Return the shared effective valid region for two identically shaped operands.
///
/// This deliberately does not infer a region through broadcasting or intersect
/// different regions: those cases need axis-aware mapping before Tensor-to-Tile
/// lowering. Exact-shape operands with provably equal regions need no mapping,
/// and every element-wise result cell is valid iff the corresponding input cells
/// are valid.
static std::optional<std::vector<ExprPtr>> GetMatchingElementwiseValidShape(
    const std::shared_ptr<const TensorType>& lhs, const std::shared_ptr<const TensorType>& rhs) {
  if (lhs->shape_.size() != rhs->shape_.size()) return std::nullopt;
  for (size_t i = 0; i < lhs->shape_.size(); ++i) {
    if (!DimensionsEqual(lhs->shape_[i], rhs->shape_[i])) return std::nullopt;
  }

  auto lhs_valid_shape = GetValidShape(lhs);
  auto rhs_valid_shape = GetValidShape(rhs);
  INTERNAL_CHECK(lhs_valid_shape.size() == rhs_valid_shape.size())
      << "Internal error: identically shaped tensors have different valid_shape ranks";
  for (size_t i = 0; i < lhs_valid_shape.size(); ++i) {
    if (ProveValidExtentEqual(lhs_valid_shape[i], rhs_valid_shape[i]) != ProofResult::kTrue) {
      return std::nullopt;
    }
  }
  return lhs_valid_shape;
}

TypePtr DeduceTensorOpElementwiseBinaryType(const std::vector<ExprPtr>& args,
                                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                                            const std::string& op_name,
                                            bool preserve_matching_valid_shape = true) {
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

  if (preserve_matching_valid_shape) {
    if (auto valid_shape = GetMatchingElementwiseValidShape(tensor_type1, tensor_type2)) {
      return MakeFreshTensorType(broadcast_result.shape, *result_dtype, std::move(*valid_shape));
    }
  }

  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

TypePtr DeduceTensorOpElementwiseScalarType(const std::vector<ExprPtr>& args,
                                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                                            const std::string& op_name, bool preserve_lhs_dtype = false,
                                            bool preserve_valid_shape = true) {
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

  DataType result_dtype = tensor_type1->dtype_;
  if (!preserve_lhs_dtype) {
    auto promoted_dtype = PromoteDataTypes(tensor_type1->dtype_, scalar_type2->dtype_);
    CHECK(promoted_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                          << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();
    result_dtype = *promoted_dtype;
  }

  if (preserve_valid_shape) {
    return MakeFreshTensorType(tensor_type1->shape_, result_dtype, GetValidShape(tensor_type1));
  }
  return std::make_shared<TensorType>(tensor_type1->shape_, result_dtype);
}

// Bitwise and shift ops have no row/col-expand tile counterpart (there is no
// tile.row_expand_and), so a broadcasting operand pair has nothing to lower onto.
// Require identical shapes here, where the error can name the operands the user
// wrote, rather than letting ConvertTensorToTileOps fail on IR it did not build.
static void CheckBitwiseShapesMatch(const std::shared_ptr<const TensorType>& lhs,
                                    const std::shared_ptr<const TensorType>& rhs,
                                    const std::string& op_name) {
  bool same_shape = lhs->shape_.size() == rhs->shape_.size();
  for (size_t i = 0; same_shape && i < lhs->shape_.size(); ++i) {
    same_shape = DimensionsEqual(lhs->shape_[i], rhs->shape_[i]);
  }
  CHECK(same_shape) << "The operator " << op_name
                    << " requires both operands to have the same shape, but got " << FormatShape(lhs->shape_)
                    << " and " << FormatShape(rhs->shape_)
                    << ". Broadcasting is not supported for bitwise/shift operators because the "
                       "hardware provides no row/col-expand form; reshape or expand the operand "
                       "explicitly first.";
}

// Tensor-tensor bitwise/shift ops: both operands must be integer tensors of the
// same shape. ``preserve_lhs_dtype`` mirrors the tile layer's split between
// tile.and/or/xor (promote both dtypes, DeduceTileOpElementwiseBinaryType) and
// tile.shl/shr (keep the lhs element type, DeduceTileOpShiftBinaryType) — the
// shift count does not participate in the result type.
TypePtr DeduceTensorOpBitwiseBinaryType(const std::vector<ExprPtr>& args,
                                        const std::vector<std::pair<std::string, std::any>>& kwargs,
                                        const std::string& op_name, bool preserve_lhs_dtype,
                                        bool preserve_matching_valid_shape = true) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  auto tensor_type1 = AsTensorTypeLike(args[0]->GetType());  // accepts a window (issue #1694)
  auto tensor_type2 = AsTensorTypeLike(args[1]->GetType());
  CHECK(tensor_type1) << "The operator " << op_name
                      << " requires first argument to be a TensorType or DistributedTensorType, but got "
                      << args[0]->GetType()->TypeName();
  CHECK(tensor_type2) << "The operator " << op_name
                      << " requires second argument to be a TensorType or DistributedTensorType, but got "
                      << args[1]->GetType()->TypeName();

  CHECK(tensor_type1->dtype_.IsInt())
      << "The operator " << op_name << " requires an integer tensor dtype, but got "
      << tensor_type1->dtype_.ToString();
  CHECK(tensor_type2->dtype_.IsInt())
      << "The operator " << op_name << " requires an integer tensor dtype, but got "
      << tensor_type2->dtype_.ToString();

  CheckBitwiseShapesMatch(tensor_type1, tensor_type2, op_name);

  DataType result_dtype = tensor_type1->dtype_;
  if (!preserve_lhs_dtype) {
    auto promoted_dtype = PromoteDataTypes(tensor_type1->dtype_, tensor_type2->dtype_);
    CHECK(promoted_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                          << tensor_type1->dtype_.ToString() << " and " << tensor_type2->dtype_.ToString();
    result_dtype = *promoted_dtype;
  }

  if (preserve_matching_valid_shape) {
    if (auto valid_shape = GetMatchingElementwiseValidShape(tensor_type1, tensor_type2)) {
      return MakeFreshTensorType(tensor_type1->shape_, result_dtype, std::move(*valid_shape));
    }
  }
  return std::make_shared<TensorType>(tensor_type1->shape_, result_dtype);
}

// Tensor-scalar bitwise/shift ops. Mirrors DeduceTileOpIntScalarBinaryType: the
// scalar must be an integer of any width (codegen casts it to i32 per the ISA
// ``%dst = tands/tshls %src, %scalar : !pto.tile<...>, i32`` form), and the
// result keeps the tensor's element type — a bitwise op never changes it.
// Layered on the shared scalar deducer exactly as tensor.subs is (below): it already
// validates arity and both operand kinds and, with preserve_lhs_dtype, returns the lhs
// element type unchanged.
// ``is_shift`` additionally rejects a statically negative shift count. It is scoped to
// the shift ops on purpose: a negative *mask* is meaningful (``x & -1`` sets all bits),
// whereas a negative shift distance has no defined result at any layer. Nothing
// downstream catches it — PTO codegen maps tile.shls/shrs straight onto
// pto.tshls/tshrs (src/backend/common/pto_ops_elementwise.cpp) with no range guard — so
// a constant caught here would otherwise reach the hardware as garbage.
TypePtr DeduceTensorOpBitwiseScalarType(const std::vector<ExprPtr>& args,
                                        const std::vector<std::pair<std::string, std::any>>& kwargs,
                                        const std::string& op_name, bool is_shift = false,
                                        bool preserve_valid_shape = true) {
  auto result_type = DeduceTensorOpElementwiseScalarType(args, kwargs, op_name, true, preserve_valid_shape);
  auto tensor_type = AsTensorTypeLike(args[0]->GetType());  // accepts a window (issue #1694)
  auto scalar_type = As<ScalarType>(args[1]->GetType());
  CHECK(tensor_type->dtype_.IsInt()) << "The operator " << op_name
                                     << " requires an integer tensor dtype, but got "
                                     << tensor_type->dtype_.ToString();
  CHECK(scalar_type->dtype_.IsInt()) << "The operator " << op_name
                                     << " requires the shift/bitwise scalar to be an integer type, but got "
                                     << scalar_type->dtype_.ToString();
  if (is_shift) {
    if (auto shift_count = As<ConstInt>(args[1])) {
      CHECK(shift_count->value_ >= 0)
          << "The operator " << op_name << " requires a non-negative shift count, but got "
          << shift_count->value_ << ". A negative shift distance has no defined result.";
    }
  }
  return result_type;
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
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.part_add", false);
    });

REGISTER_OP("tensor.part_mul")
    .set_op_category("TensorOp")
    .set_description("Partial element-wise multiply of two tensors")
    .add_argument("src0", "First source tensor (TensorType)")
    .add_argument("src1", "Second source tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.part_mul", false);
    });

REGISTER_OP("tensor.part_max")
    .set_op_category("TensorOp")
    .set_description("Partial element-wise max of two tensors")
    .add_argument("src0", "First source tensor (TensorType)")
    .add_argument("src1", "Second source tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.part_max", false);
    });

REGISTER_OP("tensor.part_min")
    .set_op_category("TensorOp")
    .set_description("Partial element-wise min of two tensors")
    .add_argument("src0", "First source tensor (TensorType)")
    .add_argument("src1", "Second source tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.part_min", false);
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
        return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.cmp", false);
      }
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.cmp", false, false);
    });

// ============================================================================
// Bitwise and shift ops (integer-only)
//
// Tensor-level counterparts of tile.and/or/xor/shl/shr and their `*s` scalar
// forms. Each lowers 1:1 in ConvertTensorToTileOps, except tensor.xor/xors:
// pto.txor/txors take a third `tmp` scratch operand, which the conversion
// allocates so tensor-level callers never see it (same treatment as the
// high-precision tensor.rsqrt scratch).
//
// Dtype strictness deliberately mirrors the *IR contract* of the tile op each one
// lowers onto, not the current per-backend gaps:
//   * tensor.not accepts int16/uint16 only, because tile.not's own type deduction
//     does (src/ir/op/tile_ops/unary.cpp) — TNOT is a 16-bit-element instruction.
//   * tensor.and/or/xor/shl/shr accept any integer width, because their tile
//     counterparts do (DeduceTileOpElementwiseBinaryType(require_int) /
//     DeduceTileOpShiftBinaryType).
// Narrower *device* limits exist today on a2a3 — ptoas rejects pto.tand/tor/tshl/tshr
// outright and TXOR/TXORS want int16/uint16 (see tests/st/runtime/ops/test_bitwise.py,
// tracked in #1846). Encoding those here would make the tensor op stricter than the
// tile op it lowers to, and would need reverting when the backend catches up; instead
// these ops inherit the fixes automatically.
//
// Shape strictness is the opposite call, for a reason worth stating: a missing
// row/col-expand *instruction* is permanent, not a backend that is catching up. There is
// no pto.trowexpandand and none is planned, so a broadcasting operand pair can never
// lower — rejecting it is describing the ISA, not freezing a temporary gap. (The tile
// deducers still broadcast here, so pl.tile.and_([M,N],[M,1]) type-checks and fails
// later; tightening those shared helpers touches shipped ops and is left as follow-up.
// tensor.fmod and tensor.part_* have the same missing-row-expand shape and are likewise
// unguarded today.)
// ============================================================================

REGISTER_OP("tensor.and")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise AND of two integer tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType, integer dtype)")
    .add_argument("rhs", "Right-hand side tensor (TensorType, integer dtype)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpBitwiseBinaryType(args, kwargs, "tensor.and", false);
    });

REGISTER_OP("tensor.ands")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise AND of an integer tensor and an integer scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType, integer dtype)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType, integer dtype)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpBitwiseScalarType(args, kwargs, "tensor.ands");
    });

REGISTER_OP("tensor.or")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise OR of two integer tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType, integer dtype)")
    .add_argument("rhs", "Right-hand side tensor (TensorType, integer dtype)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpBitwiseBinaryType(args, kwargs, "tensor.or", false);
    });

REGISTER_OP("tensor.ors")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise OR of an integer tensor and an integer scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType, integer dtype)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType, integer dtype)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpBitwiseScalarType(args, kwargs, "tensor.ors");
    });

// Two args only — the pto.txor scratch operand is synthesized during lowering.
REGISTER_OP("tensor.xor")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise XOR of two integer tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType, integer dtype)")
    .add_argument("rhs", "Right-hand side tensor (TensorType, integer dtype)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpBitwiseBinaryType(args, kwargs, "tensor.xor", false, false);
    });

REGISTER_OP("tensor.xors")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise XOR of an integer tensor and an integer scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType, integer dtype)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType, integer dtype)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpBitwiseScalarType(args, kwargs, "tensor.xors", false, false);
    });

// Shifts keep the lhs element type: the shift count never widens the result.
REGISTER_OP("tensor.shl")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise left shift of two integer tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType, integer dtype)")
    .add_argument("rhs", "Shift-amount tensor (TensorType, integer dtype)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpBitwiseBinaryType(args, kwargs, "tensor.shl", true);
    });

REGISTER_OP("tensor.shls")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise left shift of an integer tensor by an integer scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType, integer dtype)")
    .add_argument("rhs", "Shift amount (ScalarType, integer dtype); must be >= 0")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpBitwiseScalarType(args, kwargs, "tensor.shls", true);
    });

REGISTER_OP("tensor.shr")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise right shift of two integer tensors")
    .add_argument("lhs", "Left-hand side tensor (TensorType, integer dtype)")
    .add_argument("rhs", "Shift-amount tensor (TensorType, integer dtype)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpBitwiseBinaryType(args, kwargs, "tensor.shr", true);
    });

REGISTER_OP("tensor.shrs")
    .set_op_category("TensorOp")
    .set_description("Element-wise bitwise right shift of an integer tensor by an integer scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType, integer dtype)")
    .add_argument("rhs", "Shift amount (ScalarType, integer dtype); must be >= 0")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpBitwiseScalarType(args, kwargs, "tensor.shrs", true);
    });

}  // namespace ir
}  // namespace pypto

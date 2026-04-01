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
 * @file gather.cpp
 * @brief Gather tensor operation
 *
 * Implements tensor.gather, which extracts values from the input tensor along
 * the specified dimension using the provided index tensor (PyTorch gather semantics).
 *
 * For a 3D tensor:
 *   output[i][j][k] = input[index[i][j][k]][j][k]  if dim=0
 *   output[i][j][k] = input[i][index[i][j][k]][k]  if dim=1
 *   output[i][j][k] = input[i][j][index[i][j][k]]  if dim=2
 */

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

TypePtr DeduceTensorGatherType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.gather(input, index, dim=...) -> TensorType with index's shape and input's dtype
  CHECK(args.size() == 2) << "tensor.gather requires exactly 2 arguments (input, index), got " << args.size();

  auto input_type = As<TensorType>(args[0]->GetType());
  CHECK(input_type) << "tensor.gather: input must be TensorType, got " << args[0]->GetType()->TypeName();

  auto index_type = As<TensorType>(args[1]->GetType());
  CHECK(index_type) << "tensor.gather: index must be TensorType, got " << args[1]->GetType()->TypeName();

  CHECK(index_type->dtype_.IsInt()) << "tensor.gather: index dtype must be integer, got "
                                    << index_type->dtype_.ToString();

  int64_t ndim = static_cast<int64_t>(input_type->shape_.size());
  CHECK(ndim > 0) << "tensor.gather: input must have at least 1 dimension";

  CHECK(static_cast<int64_t>(index_type->shape_.size()) == ndim)
      << "tensor.gather: index rank (" << index_type->shape_.size() << ") must match input rank (" << ndim
      << ")";

  // Extract and validate dim
  int dim = 0;
  bool dim_found = false;
  for (const auto& [key, val] : kwargs) {
    if (key == "dim") {
      dim = AnyCast<int>(val, "kwarg key: dim");
      dim_found = true;
    }
  }
  CHECK(dim_found) << "tensor.gather: dim attribute is required";
  CHECK(dim >= -ndim && dim < ndim) << "tensor.gather: dim (" << dim << ") out of range for tensor with rank "
                                    << ndim;
  if (dim < 0) {
    dim += static_cast<int>(ndim);
  }

  // Validate index.shape[i] <= input.shape[i] for i != dim (when shapes are statically known)
  for (int64_t i = 0; i < ndim; ++i) {
    if (i == dim) continue;
    auto idx_dim = As<ConstInt>(index_type->shape_[i]);
    auto inp_dim = As<ConstInt>(input_type->shape_[i]);
    if (idx_dim && inp_dim) {
      CHECK(idx_dim->value_ <= inp_dim->value_)
          << "tensor.gather: index.shape[" << i << "] (" << idx_dim->value_ << ") must be <= input.shape["
          << i << "] (" << inp_dim->value_ << ") for non-gather dimension";
    }
  }

  return std::make_shared<TensorType>(index_type->shape_, input_type->dtype_);
}

REGISTER_OP("tensor.gather")
    .set_op_category("TensorOp")
    .set_description(
        "Gather values from input tensor along the specified dimension using index tensor. "
        "output[i][j][k] = input[i][index[i][j][k]][k] when dim=1 (3D example). "
        "index must have the same rank as input, with index.shape[i] <= input.shape[i] for i != dim. "
        "Output has the same shape as index and the same dtype as input.")
    .add_argument("input", "Input tensor to gather from")
    .add_argument("index", "Index tensor (same rank as input, integer dtype)")
    .set_attr<int>("dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorGatherType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto

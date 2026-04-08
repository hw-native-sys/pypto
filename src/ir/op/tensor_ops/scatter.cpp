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
 * @file scatter.cpp
 * @brief Element-level scatter tensor operation
 *
 * Implements tensor.scatter_, which writes values from a source tensor (or scalar)
 * into an input tensor at positions specified by a per-element index tensor along
 * a given dimension.  Follows PyTorch torch.Tensor.scatter_ semantics:
 *
 *   self[i₀]…[i_{d-1}][ index[i₀…iₙ] ][i_{d+1}]…[iₙ] = src[i₀…iₙ]
 */

#include <any>
#include <cstddef>
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

TypePtr DeduceTensorScatterType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.scatter_(input, index, src) -> TensorType same as input
  // input: N-D tensor
  // index: N-D tensor of integer dtype (same rank as input)
  // src:   N-D tensor (same shape as index, dtype matches input) OR scalar
  CHECK(args.size() == 3) << "tensor.scatter_ requires exactly 3 arguments (input, index, src), got "
                          << args.size();

  auto input_type = As<TensorType>(args[0]->GetType());
  CHECK(input_type) << "tensor.scatter_: input must be TensorType, got " << args[0]->GetType()->TypeName();
  const size_t rank = input_type->shape_.size();
  CHECK(rank >= 1) << "tensor.scatter_: input must be at least 1D, got rank 0";

  auto index_type = As<TensorType>(args[1]->GetType());
  CHECK(index_type) << "tensor.scatter_: index must be TensorType, got " << args[1]->GetType()->TypeName();
  CHECK(index_type->shape_.size() == rank) << "tensor.scatter_: index rank (" << index_type->shape_.size()
                                           << ") must match input rank (" << rank << ")";
  CHECK(index_type->dtype_.IsInt()) << "tensor.scatter_: index dtype must be integer, got "
                                    << index_type->dtype_.ToString();

  // src can be TensorType or scalar (ConstFloat / ConstInt)
  bool src_is_scalar = As<ConstFloat>(args[2]) || As<ConstInt>(args[2]);
  if (!src_is_scalar) {
    auto src_type = As<TensorType>(args[2]->GetType());
    CHECK(src_type) << "tensor.scatter_: src must be TensorType or scalar, got "
                    << args[2]->GetType()->TypeName();
    CHECK(src_type->shape_.size() == rank) << "tensor.scatter_: src rank (" << src_type->shape_.size()
                                           << ") must match input rank (" << rank << ")";
    CHECK(src_type->dtype_ == input_type->dtype_)
        << "tensor.scatter_: src dtype (" << src_type->dtype_.ToString() << ") must match input dtype ("
        << input_type->dtype_.ToString() << ")";
  } else {
    // Validate scalar dtype compatibility with input dtype
    if (As<ConstFloat>(args[2])) {
      CHECK(input_type->dtype_.IsFloat())
          << "tensor.scatter_: float scalar src requires float input dtype, got "
          << input_type->dtype_.ToString();
    } else if (As<ConstInt>(args[2])) {
      CHECK(input_type->dtype_.IsInt())
          << "tensor.scatter_: integer scalar src requires integer input dtype, got "
          << input_type->dtype_.ToString();
    }
  }

  // Validate dim kwarg
  for (const auto& [key, val] : kwargs) {
    if (key == "dim") {
      int dim_val = AnyCast<int>(val, "kwarg key: dim");
      int irank = static_cast<int>(rank);
      CHECK(dim_val >= -irank && dim_val < irank) << "tensor.scatter_: dim must be in [" << -irank << ", "
                                                  << irank << ") for " << rank << "D input, got " << dim_val;
    }
  }

  return std::make_shared<TensorType>(input_type->shape_, input_type->dtype_);
}

REGISTER_OP("tensor.scatter_")
    .set_op_category("TensorOp")
    .set_description(
        "Element-level scatter: write src values into input at positions given by index along dim. "
        "For each element position (i₀,…,iₙ) in index, sets "
        "input[i₀]…[i_{d-1}][index[i₀…iₙ]][i_{d+1}]…[iₙ] = src[i₀…iₙ]. "
        "Supports arbitrary rank and any valid dim ∈ [-rank, rank). "
        "src can be a tensor (same shape as index) or a scalar value.")
    .add_argument("input", "Destination tensor (N-D)")
    .add_argument("index", "Index tensor (N-D, same rank as input) of integer dtype")
    .add_argument("src", "Source tensor (same shape as index) or scalar value")
    .set_attr<int>("dim")
    .set_attr<std::string>("reduce")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorScatterType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto

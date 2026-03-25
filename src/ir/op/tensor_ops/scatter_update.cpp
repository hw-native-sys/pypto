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
 * @file scatter_update.cpp
 * @brief Scatter update tensor operation
 *
 * Implements tensor.scatter_update, which updates rows of an input tensor at positions
 * specified by a 2D index tensor, using values from a source tensor.
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

TypePtr DeduceTensorScatterUpdateType(const std::vector<ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.scatter_update(input, index, src) -> TensorType same as input
  // input: 2D [rows, d] or 4D [blockNum, blockSize, 1, d]
  // index: 2D [b, s] of integer dtype
  // src:   2D [b*s, d] or 4D [b, s, 1, d] (same rank as input)
  CHECK(args.size() == 3) << "tensor.scatter_update requires exactly 3 arguments (input, index, src), got "
                          << args.size();

  auto input_type = As<TensorType>(args[0]->GetType());
  CHECK(input_type) << "tensor.scatter_update: input must be TensorType, got "
                    << args[0]->GetType()->TypeName();
  CHECK(input_type->shape_.size() == 2 || input_type->shape_.size() == 4)
      << "tensor.scatter_update: input must be 2D or 4D, got rank " << input_type->shape_.size();

  auto index_type = As<TensorType>(args[1]->GetType());
  CHECK(index_type) << "tensor.scatter_update: index must be TensorType, got "
                    << args[1]->GetType()->TypeName();
  CHECK(index_type->shape_.size() == 2)
      << "tensor.scatter_update: index must be 2D [b, s], got rank " << index_type->shape_.size();
  CHECK(index_type->dtype_.IsInt()) << "tensor.scatter_update: index dtype must be integer, got "
                                    << index_type->dtype_.ToString();

  auto src_type = As<TensorType>(args[2]->GetType());
  CHECK(src_type) << "tensor.scatter_update: src must be TensorType, got " << args[2]->GetType()->TypeName();
  CHECK(src_type->shape_.size() == input_type->shape_.size())
      << "tensor.scatter_update: src rank (" << src_type->shape_.size() << ") must match input rank ("
      << input_type->shape_.size() << ")";
  CHECK(src_type->dtype_ == input_type->dtype_)
      << "tensor.scatter_update: src dtype (" << src_type->dtype_.ToString() << ") must match input dtype ("
      << input_type->dtype_.ToString() << ")";

  // Validate dim kwarg: only -2 is currently supported
  for (const auto& [key, val] : kwargs) {
    if (key == "dim") {
      int dim_val = AnyCast<int>(val, "kwarg key: dim");
      CHECK(dim_val == -2) << "tensor.scatter_update: only dim=-2 is currently supported, got " << dim_val;
    }
  }

  return std::make_shared<TensorType>(input_type->shape_, input_type->dtype_);
}

REGISTER_OP("tensor.scatter_update")
    .set_op_category("TensorOp")
    .set_description(
        "Update input tensor rows at positions given by 2D index tensor with values from src. "
        "Supports 2D input [rows, d] with 2D src [b*s, d], and 4D input [blockNum, blockSize, 1, d] "
        "with 4D src [b, s, 1, d]. Index is always 2D [b, s] of integer dtype.")
    .add_argument("input", "Destination tensor (2D [rows, d] or 4D [blockNum, blockSize, 1, d])")
    .add_argument("index", "2D index tensor [b, s] of integer dtype")
    .add_argument("src", "Source tensor (2D [b*s, d] or 4D [b, s, 1, d])")
    .set_attr<int>("dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorScatterUpdateType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto

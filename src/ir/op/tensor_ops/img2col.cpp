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
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "src/ir/op/img2col_utils.h"

namespace pypto::ir {
namespace {

TypePtr DeduceTensorImg2colType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 4) << "tensor.img2col requires src, pos_m, pos_k, shape";
  const auto& span = args[0]->span_;
  auto src = AsTensorTypeLike(args[0]->GetType());
  CHECK_SPAN(src && src->shape_.size() == 2, span) << "tensor.img2col requires a 2D source tensor";
  CHECK_SPAN(tile_view_semantics::ShapeExprListsEquivalent(GetValidShape(src), src->shape_), span)
      << "tensor.img2col requires a fully valid source tensor";
  return std::make_shared<TensorType>(GetImg2colShape(*src, args, kwargs, "tensor.img2col"), src->dtype_);
}

REGISTER_OP("tensor.img2col")
    .set_op_category("TensorOp")
    .set_description("Unfold a feature map into a matrix window for matmul using TIMG2COL")
    .add_argument("src", "Full feature map [H*W, C]")
    .add_argument("pos_m", "Starting flattened output spatial position")
    .add_argument("pos_k", "Starting C1, KH, KW, C0 position")
    .add_argument("shape", "Static destination shape [M, K]")
    .set_attr<int>("fmap_h")
    .set_attr<int>("fmap_w")
    .set_attr<int>("kernel_h")
    .set_attr<int>("kernel_w")
    .set_attr<int>("stride_h")
    .set_attr<int>("stride_w")
    .set_attr<int>("dilation_h")
    .set_attr<int>("dilation_w")
    .set_attr<int>("pad_top")
    .set_attr<int>("pad_bottom")
    .set_attr<int>("pad_left")
    .set_attr<int>("pad_right")
    .f_deduce_type(DeduceTensorImg2colType);

}  // namespace
}  // namespace pypto::ir

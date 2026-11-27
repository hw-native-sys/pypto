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

#ifndef SRC_CODEGEN_PTO_PTO_TENSOR_VIEW_PLAN_H_
#define SRC_CODEGEN_PTO_PTO_TENSOR_VIEW_PLAN_H_

#include <cstddef>
#include <set>
#include <variant>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/type.h"

namespace pypto::codegen {

struct TensorParameterView {
  struct ShapeDim {
    size_t axis;
  };
  struct Product {
    // A product can reuse an earlier stride, so fallback stride expressions
    // do not reevaluate a common prefix or a composite shape dimension.
    std::variant<ShapeDim, size_t> lhs;  // size_t denotes a previously resolved stride axis.
    size_t rhs_shape_axis;
  };
  struct Stride {
    size_t axis;
    std::variant<ir::ExprPtr, ShapeDim, Product> value;
  };

  ir::VarPtr parameter;
  ir::TensorTypePtr type;
  ir::TensorLayout layout;
  // In evaluation order, with one entry per result axis.
  std::vector<Stride> strides;
};

struct EntryParameterPlan {
  // Position in this vector is the native argument index. All declared arguments
  // keep their ABI slot, including unused tensors and raw workspaces.
  std::vector<ir::VarPtr> abi_order;
  std::vector<TensorParameterView> views;
};

EntryParameterPlan PrepareEntryParameters(const ir::Function& func, bool buffer_ir,
                                          const std::set<const ir::Var*>& ffts_workspaces);

}  // namespace pypto::codegen

#endif  // SRC_CODEGEN_PTO_PTO_TENSOR_VIEW_PLAN_H_

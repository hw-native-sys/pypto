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

#include "src/codegen/pto/pto_tensor_view_plan.h"

#include <cstddef>
#include <memory>
#include <set>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto::codegen {
namespace {

TensorParameterView PrepareTensorParameterView(const ir::VarPtr& parameter, const ir::TensorTypePtr& type) {
  using ShapeDim = TensorParameterView::ShapeDim;
  using Product = TensorParameterView::Product;
  const size_t rank = type->shape_.size();
  const auto& view = type->tensor_view_;
  auto layout = view ? view->layout : ir::TensorLayout::ND;
  const auto last_dim = rank >= 2 ? ir::As<ir::ConstInt>(type->shape_.back()) : nullptr;
  // PTOAS infers DN for ordinary column vectors. MX and FP4 carrier shapes
  // already encode their physical layout and must keep the declared layout.
  const bool force_column_dn =
      last_dim && last_dim->value_ == 1 && !ir::IsMxTensorLayout(layout) && !type->dtype_.IsPackedFp4();
  if (force_column_dn) layout = ir::TensorLayout::DN;

  TensorParameterView result{parameter, type, layout, {}};
  result.strides.reserve(rank);
  if (view && !view->stride.empty()) {
    CHECK_SPAN(view->stride.size() == rank, parameter->span_)
        << "Explicit tensor view stride rank " << view->stride.size() << " does not match tensor shape rank "
        << rank;
    for (size_t i = 0; i < rank; ++i) result.strides.push_back({i, view->stride[i]});
    return result;
  }

  const auto one = std::make_shared<ir::ConstInt>(1, DataType::INDEX, parameter->span_);
  if (layout == ir::TensorLayout::DN) {
    CHECK_SPAN(rank >= 2, parameter->span_) << "Tensor view DN layout requires rank >= 2, got " << rank;
    result.strides.push_back({rank - 2, ir::ExprPtr(one)});
    // Preserve the legacy higher-rank column-vector convention, where the
    // trailing stride is the degenerate extent (1), not the column height.
    result.strides.push_back({rank - 1, ShapeDim{force_column_dn && rank > 2 ? rank - 1 : rank - 2}});
    if (rank <= 2) return result;
    if (force_column_dn) {
      result.strides.push_back({rank - 3, ShapeDim{rank - 2}});
    } else {
      result.strides.push_back({rank - 3, Product{ShapeDim{rank - 2}, rank - 1}});
    }
    for (size_t i = rank - 3; i > 0; --i) result.strides.push_back({i - 1, Product{i, i}});
  } else if (rank != 0) {
    result.strides.push_back({rank - 1, ir::ExprPtr(one)});
    if (rank >= 2) result.strides.push_back({rank - 2, ShapeDim{rank - 1}});
    for (size_t i = rank - 1; i > 1; --i) result.strides.push_back({i - 2, Product{i - 1, i - 1}});
  }
  return result;
}

}  // namespace

EntryParameterPlan PrepareEntryParameters(const ir::Function& func, bool buffer_ir,
                                          const std::set<const ir::Var*>& ffts_workspaces) {
  ir::var_collectors::VarDefUseCollector body_vars;
  if (func.body_) body_vars.VisitStmt(func.body_);

  EntryParameterPlan plan;
  plan.abi_order.reserve(func.params_.size());
  std::vector<ir::VarPtr> trailing_params;
  for (const auto& parameter : func.params_) {
    const auto type = ir::AsTensorTypeLike(parameter->GetType());
    if (!type) {
      trailing_params.push_back(parameter);
      continue;
    }
    plan.abi_order.push_back(parameter);
    if (body_vars.var_uses.count(parameter.get()) == 0) continue;
    // Outlining shares a public signature across cores. Unused parameters need
    // no view, and PTOAS cannot infer some unused layouts (e.g. MX).
    if ((!buffer_ir && parameter->name_hint_ == "__gm_pipe_buffer") ||
        ffts_workspaces.count(parameter.get())) {
      continue;
    }
    plan.views.push_back(PrepareTensorParameterView(parameter, type));
  }
  // PTOParam dispatches tensor pointers first, preserving IR order within each group.
  plan.abi_order.insert(plan.abi_order.end(), trailing_params.begin(), trailing_params.end());
  return plan;
}

}  // namespace pypto::codegen

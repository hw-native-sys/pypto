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

#include "pypto/ir/stmt.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/structural_comparison.h"

namespace pypto {
namespace ir {

SimtLaunchStmt::SimtLaunchStmt(GlobalVarPtr callee, std::vector<ExprPtr> block_dims,
                               std::vector<ExprPtr> args, std::vector<VarPtr> results,
                               std::vector<int> result_param_indices, Span span,
                               std::vector<std::string> leading_comments)
    : Stmt(std::move(span), std::move(leading_comments)),
      callee_(std::move(callee)),
      block_dims_(std::move(block_dims)),
      args_(std::move(args)),
      results_(std::move(results)),
      result_param_indices_(std::move(result_param_indices)) {
  CHECK_SPAN(callee_, span_) << "SimtLaunchStmt requires a callee symbol";
  CHECK_SPAN(block_dims_.size() == 3, span_) << "SIMT block_dims must contain exactly three dimensions";
  int64_t threads = 1;
  for (const auto& dim : block_dims_) {
    auto value = As<ConstInt>(dim);
    CHECK_SPAN(value && (value->dtype().IsInt() || value->dtype() == DataType::INDEX) && value->value_ > 0,
               span_)
        << "SIMT block dimensions must be positive constant integers";
    CHECK_SPAN(value->value_ <= std::numeric_limits<int32_t>::max() / threads, span_)
        << "SIMT block dimension product exceeds INT32_MAX";
    threads *= value->value_;
  }
  for (const auto& arg : args_) detail::CheckValueOperand(arg, span_, "SIMT launch argument");
  CHECK_SPAN(results_.size() == result_param_indices_.size(), span_)
      << "SIMT results and result_param_indices must have the same length";
  int previous = -1;
  std::unordered_set<const Var*> definitions;
  for (size_t i = 0; i < results_.size(); ++i) {
    CHECK_SPAN(results_[i], span_) << "SIMT launch result cannot be null";
    const auto kind = results_[i]->GetType()->GetKind();
    CHECK_SPAN(kind == ObjectKind::TileType || kind == ObjectKind::TensorType, span_)
        << "SIMT launch results must be Tile or Tensor storage";
    CHECK_SPAN(definitions.insert(results_[i].get()).second, span_) << "SIMT launch results must be distinct";
    const int index = result_param_indices_[i];
    CHECK_SPAN(index > previous && index >= 0 && static_cast<size_t>(index) < args_.size(), span_)
        << "SIMT result_param_indices must be unique, increasing argument indices";
    CHECK_SPAN(structural_equal(results_[i]->GetType(), args_[index]->GetType()), span_)
        << "SIMT launch result must preserve the corresponding argument type and storage metadata";
    previous = index;
  }
}

}  // namespace ir
}  // namespace pypto

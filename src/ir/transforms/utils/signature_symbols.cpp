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

#include "pypto/ir/transforms/utils/signature_symbols.h"

#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {

// Iterative and DAG-aware: signature preparation must not expand shared shapes
// or overflow before the structural comparator can enforce its depth limit.
template <typename Visit>
void VisitShapeVars(const ExprPtr& root, std::unordered_set<const Expr*>& seen, Visit visit) {
  std::vector<ExprPtr> pending{root};
  while (!pending.empty()) {
    auto expr = pending.back();
    pending.pop_back();
    if (!expr || !seen.insert(expr.get()).second) continue;
    if (auto var = AsVarLike(expr)) {
      visit(var);
    } else if (auto binary = As<BinaryExpr>(expr)) {
      pending.push_back(binary->right_);
      pending.push_back(binary->left_);
    } else if (auto unary = As<UnaryExpr>(expr)) {
      pending.push_back(unary->operand_);
    } else if (auto call = As<Call>(expr)) {
      pending.insert(pending.end(), call->args_.rbegin(), call->args_.rend());
    } else if (auto tuple = As<MakeTuple>(expr)) {
      pending.insert(pending.end(), tuple->elements_.rbegin(), tuple->elements_.rend());
    } else if (auto item = As<TupleGetItemExpr>(expr)) {
      pending.push_back(item->tuple_);
    }
  }
}

}  // namespace

SignatureSymbols ClassifySignatureSymbols(const FunctionPtr& function) {
  SignatureSymbols result;
  std::unordered_set<const Var*> bound;
  auto bind = [&](const VarPtr& var) {
    if (bound.insert(var.get()).second) result.bound.push_back(var);
  };
  std::unordered_set<const Expr*> physical;
  for (const auto& param : function->params_) {
    if (auto shaped = As<ShapedType>(param->GetType())) {
      for (const auto& dim : shaped->shape_) VisitShapeVars(dim, physical, bind);
    } else {
      bind(param);
    }
  }
  std::unordered_map<const Var*, size_t> slots;
  for (size_t i = 0; i < function->params_.size(); ++i) {
    auto tensor = AsTensorTypeLike(function->params_[i]->GetType());
    if (!tensor || !tensor->tensor_view_) continue;
    const auto& shape = tensor->tensor_view_->valid_shape;
    for (size_t d = 0; d < shape.size(); ++d) {
      auto var = AsVarLike(shape[d]);
      if (!var || bound.count(var.get())) continue;
      auto [entry, inserted] = slots.emplace(var.get(), result.valid_shape.size());
      if (inserted) result.valid_shape.push_back({var, {}});
      result.valid_shape[entry->second].slots.emplace_back(i, d);
    }
  }
  std::unordered_set<const Expr*> valid;
  for (size_t i = 0; i < function->params_.size(); ++i) {
    auto tensor = AsTensorTypeLike(function->params_[i]->GetType());
    if (!tensor || !tensor->tensor_view_) continue;
    for (const auto& dim : tensor->tensor_view_->valid_shape) {
      VisitShapeVars(dim, valid, [&](const VarPtr& var) {
        if (!bound.count(var.get()) && !slots.count(var.get())) result.unbound.push_back({i, dim, var});
      });
    }
  }
  return result;
}

}  // namespace pypto::ir

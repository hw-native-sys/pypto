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

#include <cstddef>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tensor_subview.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {
// Validate the defining call only. The result is an ordinary Tensor: aliases,
// consumers and control flow follow the same rules as any other TensorView.
class SubviewVerifier : public IRVisitor {
 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    if (!IsOp(op, "tensor.subview")) return;
    // Reuse construction checks for explicit IR Calls, which can supply a type
    // without running deduction. Unknown dynamic geometry is deliberately deferred.
    auto expected_call = OpRegistry::GetInstance().Create(op->op_->name_, op->args_, op->kwargs_, op->span_);
    auto expected = As<TensorType>(expected_call->GetType());
    auto actual = As<TensorType>(op->GetType());
    CHECK_SPAN(actual && expected && actual->tensor_view_.has_value(), op->span_)
        << "tensor.subview result requires an ordinary Tensor with addressing metadata";
    const auto& view = actual->tensor_view_.value();
    const auto& expected_view = expected->tensor_view_.value();
    CHECK_SPAN(actual->dtype_ == expected->dtype_ && view.layout == expected_view.layout &&
                   view.pad == expected_view.pad && view.valid_shape.empty(),
               op->span_)
        << "tensor.subview result must preserve parent dtype/layout with a fully valid window";
    CheckGeometry(actual->shape_, expected->shape_, op->span_);
    CheckGeometry(view.stride, expected_view.stride, op->span_);
  }

 private:
  void CheckGeometry(const std::vector<ExprPtr>& actual, const std::vector<ExprPtr>& expected,
                     const Span& span) {
    CHECK_SPAN(actual.size() == expected.size(), span) << "tensor.subview result geometry rank mismatch";
    arith::Analyzer analyzer;
    for (size_t i = 0; i < actual.size(); ++i) {
      CHECK_SPAN(!analyzer.CanProve(MakeNe(actual[i], expected[i])), span)
          << "tensor.subview result metadata does not match its defining geometry";
    }
  }
};
}  // namespace

void VerifyTensorSubviews(const FunctionPtr& function) {
  SubviewVerifier verifier;
  verifier.VisitStmt(function->body_);
}
}  // namespace pypto::ir

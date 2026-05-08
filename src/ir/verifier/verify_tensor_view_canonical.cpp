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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Walks every Var/IterArg/Call/Function-param/return type reachable from a
/// program and asserts the TensorType canonical-view invariant on each
/// TensorType (including TensorTypes nested in TupleTypes).
class TensorViewCanonicalVisitor : public IRVisitor {
 public:
  TensorViewCanonicalVisitor(std::vector<Diagnostic>& diagnostics, std::string func_name,
                             bool require_materialized)
      : diagnostics_(diagnostics),
        func_name_(std::move(func_name)),
        require_materialized_(require_materialized) {}

  void CheckType(const TypePtr& type, const Span& span) {
    if (!type) return;
    if (auto tensor_type = As<TensorType>(type)) {
      CheckTensorType(tensor_type, span);
    } else if (auto tuple_type = As<TupleType>(type)) {
      for (const auto& sub : tuple_type->types_) {
        CheckType(sub, span);
      }
    }
  }

  void CheckFunction(const FunctionPtr& func) {
    for (const auto& param : func->params_) {
      if (param) CheckType(param->GetType(), param->span_);
    }
    for (const auto& rt : func->return_types_) {
      CheckType(rt, func->span_);
    }
    if (func->body_) {
      VisitStmt(func->body_);
    }
  }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) CheckType(op->GetType(), op->span_);
    IRVisitor::VisitVarLike_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    if (op) CheckType(op->GetType(), op->span_);
    IRVisitor::VisitExpr_(op);
  }

 private:
  void CheckTensorType(const TensorTypePtr& tensor_type, const Span& span) {
    if (!tensor_type) return;
    if (!tensor_type->tensor_view_.has_value()) {
      // Bare tensor — implicitly ND-packed; canonical by construction.
      return;
    }
    const TensorView& view = *tensor_type->tensor_view_;

    if (view.layout == TensorLayout::NZ) {
      Emit(span, "TensorType has NZ layout (NZ is tile-only and not allowed on TensorType)");
      return;
    }

    if (view.stride.empty()) {
      if (require_materialized_) {
        Emit(span,
             "TensorView.stride is empty (must be materialized via MaterializeTensorStrides "
             "before reaching this point)");
      }
      // weak mode: empty stride is acceptable; layout tag implies packed canonical.
      return;
    }

    auto result = tensor_view_semantics::CheckCanonicalView(tensor_type->shape_, view.stride, view.layout,
                                                            /*relaxed_symbolic=*/true);
    if (!result.ok) {
      Emit(span, "TensorView non-canonical: " + result.reason);
    }
  }

  void Emit(const Span& span, const std::string& msg) {
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "TensorViewCanonical",
                              /*error_code=*/1, "in function '" + func_name_ + "': " + msg, span);
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  bool require_materialized_;
};

/// Concrete PropertyVerifier dispatched by the registry.
class TensorViewCanonicalPropertyVerifierImpl : public PropertyVerifier {
 public:
  explicit TensorViewCanonicalPropertyVerifierImpl(bool require_materialized)
      : require_materialized_(require_materialized) {}

  [[nodiscard]] std::string GetName() const override { return "TensorViewCanonical"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [global_var, func] : program->functions_) {
      if (!func) continue;
      TensorViewCanonicalVisitor visitor(diagnostics, func->name_, require_materialized_);
      visitor.CheckFunction(func);
    }
  }

 private:
  bool require_materialized_;
};

}  // namespace

PropertyVerifierPtr CreateTensorViewCanonicalPropertyVerifier(bool require_materialized) {
  return std::make_shared<TensorViewCanonicalPropertyVerifierImpl>(require_materialized);
}

}  // namespace ir
}  // namespace pypto

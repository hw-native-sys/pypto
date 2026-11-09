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
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/simt_analysis.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto::ir {
namespace {
bool IsStorage(const TypePtr& type) { return As<TileType>(type) || As<TensorType>(type); }
bool IsSimtOp(const CallPtr& call) {
  const auto* entry = LookupOpEntry(call->op_);
  return entry && entry->GetOpCategory() == "SimtOp";
}

class SimtChecker : public IRVisitor {
 public:
  SimtChecker(ProgramPtr program, FunctionPtr function, std::vector<Diagnostic>& diagnostics)
      : program_(std::move(program)),
        function_(std::move(function)),
        diagnostics_(diagnostics),
        helper_(function_->func_type_ == FunctionType::SIMT) {
    for (const auto& param : function_->params_) parameters_.insert(param.get());
  }

  void Check() {
    if (helper_) {
      const auto access = AnalyzeSimtAccess(function_);
      for (size_t i = 0; i < function_->params_.size(); ++i) {
        const auto& param = function_->params_[i];
        const auto& type = param->GetType();
        if (!IsStorage(type) && !As<ScalarType>(type)) {
          Error("SIMT parameters must be scalar, Tile or Tensor", param->span_);
        }
        if (auto tile = As<TileType>(type);
            tile && tile->memory_space_ && tile->memory_space_ != MemorySpace::Vec) {
          Error("SIMT Tile parameters require Vec (UB) memory", param->span_);
        }
        const auto direction = function_->param_directions_[i];
        if (direction == ParamDirection::Out) {
          Error("SIMT helpers do not support Out parameters; use InOut", param->span_);
        }
        if (direction == ParamDirection::InOut && !IsStorage(type)) {
          Error("SIMT InOut parameters must be storage", param->span_);
        }
        if (access[i].writes && direction != ParamDirection::InOut) {
          Error("SIMT store requires an InOut parameter", param->span_);
        }
      }
    }
    VisitStmt(function_->body_);
  }

  void VisitStmt(const StmtPtr& stmt) override {
    if (helper_ && stmt) {
      switch (stmt->GetKind()) {
        case ObjectKind::AssignStmt:
        case ObjectKind::EvalStmt:
        case ObjectKind::SeqStmts:
        case ObjectKind::IfStmt:
        case ObjectKind::ForStmt:
        case ObjectKind::WhileStmt:
        case ObjectKind::YieldStmt:
        case ObjectKind::ReturnStmt:
        case ObjectKind::BreakStmt:
        case ObjectKind::ContinueStmt:
          break;
        default:
          Error(
              "SIMT helper permits only scalar control flow and SIMT builtins; nested launches/scopes are "
              "forbidden",
              stmt->span_);
      }
    }
    IRVisitor::VisitStmt(stmt);
  }

 protected:
  void VisitVarLike_(const VarPtr& var) override {
    if (helper_ && !As<ScalarType>(var->GetType())) {
      if (!parameters_.count(var.get())) {
        Error("SIMT control flow cannot define or carry storage handles", var->span_);
      }
    }
    IRVisitor::VisitVarLike_(var);
  }
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    if (helper_ && !As<ScalarType>(stmt->var_->GetType())) {
      Error("SIMT helpers cannot allocate or rebind storage handles", stmt->span_);
    }
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitStmt_(const ReturnStmtPtr& stmt) override {
    if (helper_ && !stmt->value_.empty()) Error("SIMT helpers must return void", stmt->span_);
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitStmt_(const SimtScopeStmtPtr& stmt) override {
    if (function_->func_type_ != FunctionType::AIV || helper_ || register_scope_depth_ != 0) {
      Error("simt_scope requires an AIV caller outside SIMT/register regions", stmt->span_);
    }
    // The body is a distinct execution domain. OutlineSimtScopes resolves its
    // captures and applies this verifier to the resulting canonical helper.
  }
  void VisitStmt_(const RegScopeStmtPtr& stmt) override {
    ++register_scope_depth_;
    IRVisitor::VisitStmt_(stmt);
    --register_scope_depth_;
  }
  void VisitExpr_(const CallPtr& call) override {
    const auto symbol = As<GlobalVar>(call->op_);
    const auto callee = symbol ? program_->GetFunction(symbol->name_) : nullptr;
    if (callee && callee->func_type_ == FunctionType::SIMT) {
      Error("SIMT helpers require SimtLaunchStmt, not ordinary Call", call->span_);
    }
    if (helper_ && !IsSimtOp(call)) {
      Error("SIMT helper contains an unsupported call or whole-storage operation", call->span_);
    }
    if (!helper_ && IsSimtOp(call)) Error("SIMT operations are only valid inside SIMT helpers", call->span_);
    if (helper_ && (IsOp(call, "simt.load") || IsOp(call, "simt.store"))) {
      const bool borrowed = !call->args_.empty() && parameters_.count(call->args_[0].get());
      if (!borrowed) {
        Error("SIMT memory access must refer directly to a borrowed helper parameter", call->span_);
      }
    }
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& submit) override {
    auto symbol = As<GlobalVar>(submit->op_);
    auto callee = symbol ? program_->GetFunction(symbol->name_) : nullptr;
    if (helper_ || (callee && callee->func_type_ == FunctionType::SIMT)) {
      Error("SIMT helpers cannot submit runtime tasks or be submitted as tasks", submit->span_);
    }
    IRVisitor::VisitExpr_(submit);
  }
  void VisitStmt_(const SimtLaunchStmtPtr& launch) override {
    if (register_scope_depth_ != 0) {
      Error("SIMT launch inside an active VMI register scope is not supported", launch->span_);
    }
    if (function_->func_type_ != FunctionType::AIV && function_->func_type_ != FunctionType::InCore) {
      Error("SIMT launch requires an AIV or pre-expansion InCore caller", launch->span_);
    }
    const auto callee = program_->GetFunction(launch->callee_->name_);
    if (!callee || callee->func_type_ != FunctionType::SIMT) {
      Error("SIMT launch callee must resolve to a SIMT helper", launch->span_);
      return;
    }
    if (callee->params_.size() != launch->args_.size()) {
      Error("SIMT launch argument count does not match helper signature", launch->span_);
      return;
    }
    std::unordered_map<const Expr*, size_t> occurrences;
    for (const auto& arg : launch->args_) ++occurrences[arg.get()];
    std::vector<int> expected;
    for (size_t i = 0; i < callee->params_.size(); ++i) {
      const auto& arg = launch->args_[i];
      const auto& param_type = callee->params_[i]->GetType();
      const auto& arg_type = arg->GetType();
      bool compatible = param_type->GetKind() == arg_type->GetKind();
      if (auto shape = As<ShapedType>(param_type)) {
        auto actual = As<ShapedType>(arg_type);
        compatible &=
            actual && shape->dtype_ == actual->dtype_ && shape->shape_.size() == actual->shape_.size();
        if (compatible) {
          for (size_t d = 0; d < shape->shape_.size(); ++d) {
            auto extent = As<ConstInt>(shape->shape_[d]);
            auto actual_extent = As<ConstInt>(actual->shape_[d]);
            if (extent && (!actual_extent || extent->value_ != actual_extent->value_)) compatible = false;
          }
        }
        if (!AsVarLike(arg)) Error("SIMT storage arguments must be variables", launch->span_);
        if (auto tile = As<TileType>(arg_type);
            tile && tile->memory_space_ && tile->memory_space_ != MemorySpace::Vec) {
          compatible = false;
        }
        if (auto tensor = As<TensorType>(arg_type); tensor && tensor->tensor_view_ &&
                                                    tensor->tensor_view_->layout != TensorLayout::ND &&
                                                    tensor->tensor_view_->layout != TensorLayout::DN) {
          compatible = false;
        }
      } else {
        compatible &= structural_equal(param_type, arg_type);
      }
      if (!compatible) Error("SIMT launch argument type does not match helper parameter", launch->span_);
      if (callee->param_directions_[i] == ParamDirection::InOut) {
        expected.push_back(static_cast<int>(i));
        if (occurrences.at(arg.get()) > 1) {
          Error("SIMT writable arguments must not alias another argument", launch->span_);
        }
      }
    }
    if (expected != launch->result_param_indices_) {
      Error("SIMT results must map exactly to InOut parameters in parameter order", launch->span_);
    }
    IRVisitor::VisitStmt_(launch);
  }

 private:
  void Error(const std::string& message, const Span& span) {
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "SimtValid", 0, message, span);
  }
  ProgramPtr program_;
  FunctionPtr function_;
  std::vector<Diagnostic>& diagnostics_;
  bool helper_;
  size_t register_scope_depth_ = 0;
  std::unordered_set<const Expr*> parameters_;
};

class SimtVerifier : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "SimtValid"; }
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    for (const auto& [_, function] : program->functions_) SimtChecker(program, function, diagnostics).Check();
  }
};
}  // namespace
PropertyVerifierPtr CreateSimtValidPropertyVerifier() { return std::make_shared<SimtVerifier>(); }
}  // namespace pypto::ir

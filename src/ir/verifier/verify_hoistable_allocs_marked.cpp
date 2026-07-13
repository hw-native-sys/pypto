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
#include <unordered_set>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

CallPtr AsTensorCreateAssign(const StmtPtr& stmt) {
  auto assign = As<AssignStmt>(stmt);
  if (!assign) return nullptr;
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_ || !IsOp(call, "tensor.create")) return nullptr;
  return call;
}

/// Collects every Var defined within a statement subtree — mirrors the
/// enclosing-scope-validity gate ``HoistScopeLocalAllocs`` applies.
class BodyDefCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> defs;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op->var_) defs.insert(op->var_.get());
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForStmtPtr& op) override {
    if (op->loop_var_) defs.insert(op->loop_var_.get());
    for (const auto& ia : op->iter_args_) {
      if (ia) defs.insert(ia.get());
    }
    for (const auto& rv : op->return_vars_) {
      if (rv) defs.insert(rv.get());
    }
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& rv : op->return_vars_) {
      if (rv) defs.insert(rv.get());
    }
    IRVisitor::VisitStmt_(op);
  }
};

bool ExprRefsAnyOf(const ExprPtr& expr, const std::unordered_set<const Var*>& vars) {
  if (!expr) return false;
  if (auto var = AsVarLike(expr)) return vars.count(var.get()) > 0;
  if (auto bin = As<BinaryExpr>(expr)) {
    return ExprRefsAnyOf(bin->left_, vars) || ExprRefsAnyOf(bin->right_, vars);
  }
  if (auto un = As<UnaryExpr>(expr)) return ExprRefsAnyOf(un->operand_, vars);
  if (auto cast_expr = As<Cast>(expr)) return ExprRefsAnyOf(cast_expr->operand_, vars);
  return false;
}

bool ShapeIsEnclosingScopeValid(const CallPtr& create, const std::unordered_set<const Var*>& body_defs) {
  auto result_type = AsTensorTypeLike(create->GetType());
  if (!result_type) return false;
  for (const auto& dim : result_type->shape_) {
    if (ExprRefsAnyOf(dim, body_defs)) return false;
  }
  return true;
}

/// Reports any enclosing-scope-valid ``tensor.create`` sitting directly in a
/// ``pl.manual_scope`` body that is missing the ``hoistable_alloc`` attr. Such a
/// create means ``HoistScopeLocalAllocs`` never ran — orchestration codegen
/// would then leave the buffer declared inside the manual scope, and a task
/// reading it after the block would reference an out-of-scope C++ local (#1697).
class HoistMarkChecker : public IRVisitor {
 public:
  HoistMarkChecker(std::vector<Diagnostic>& diagnostics, const std::string& func_name)
      : diagnostics_(diagnostics), func_name_(func_name) {}

 protected:
  void VisitStmt_(const RuntimeScopeStmtPtr& op) override {
    if (op->manual_ && op->body_) {
      BodyDefCollector collector;
      collector.VisitStmt(op->body_);
      CheckDirectBody(op->body_, collector.defs, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  void CheckDirectBody(const StmtPtr& body, const std::unordered_set<const Var*>& body_defs,
                       const Span& scope_span) {
    auto check_one = [&](const StmtPtr& stmt) {
      auto create = AsTensorCreateAssign(stmt);
      if (!create || !ShapeIsEnclosingScopeValid(create, body_defs)) return;
      if (create->HasAttr(kAttrHoistableAlloc)) return;
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "HoistableAllocsMarked", 0,
          "tensor.create in a pl.manual_scope body of function '" + func_name_ +
              "' is enclosing-scope-valid but carries no attrs[\"" + std::string(kAttrHoistableAlloc) +
              "\"]. HoistScopeLocalAllocs must run before orchestration codegen; without it the buffer "
              "is declared inside the manual scope and an after-scope reader references an out-of-scope "
              "C++ local (#1697).",
          scope_span);
    };
    if (auto seq = As<SeqStmts>(body)) {
      for (const auto& s : seq->stmts_) check_one(s);
    } else {
      check_one(body);
    }
  }

  std::vector<Diagnostic>& diagnostics_;
  const std::string& func_name_;
};

class HoistableAllocsMarkedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "HoistableAllocsMarked"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (func->func_type_ != FunctionType::Orchestration) continue;
      HoistMarkChecker checker(diagnostics, func->name_);
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateHoistableAllocsMarkedPropertyVerifier() {
  return std::make_shared<HoistableAllocsMarkedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

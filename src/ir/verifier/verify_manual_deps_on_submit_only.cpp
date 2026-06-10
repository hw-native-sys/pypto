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
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {
namespace {

/// Walks a function body, reporting any plain cross-function Call (GlobalVar
/// callee) that carries ``attrs["manual_dep_edges"]``. Manual dependency
/// edges belong in the typed ``Submit::deps_`` field — every producer
/// (parser ``pl.submit``, scope outlining, ExpandManualPhaseFence) emits a
/// ``Submit`` for cross-function task launches with deps. Op callees (e.g.
/// ``system.task_dummy``) legitimately keep the attr as their codegen fanin
/// contract and are exempt via the GlobalVar downcast. ``Submit`` has its own
/// ObjectKind, so it never dispatches into this handler.
class ManualDepsOnSubmitOnlyChecker : public IRVisitor {
 public:
  ManualDepsOnSubmitOnlyChecker(std::vector<Diagnostic>& diagnostics, const std::string& func_name)
      : diagnostics_(diagnostics), func_name_(func_name) {}

  void VisitExpr_(const CallPtr& op) override {
    if (As<GlobalVar>(op->op_) && op->HasAttr(kAttrManualDepEdges)) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "ManualDepsOnSubmitOnly", 0,
                                "Plain cross-function Call in function '" + func_name_ +
                                    "' carries `attrs[\"manual_dep_edges\"]`. Manual dependency edges "
                                    "belong on `Submit::deps_` — launch the task with `pl.submit(...)` "
                                    "instead of a plain call.",
                                op->span_);
    }
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    // deps_ is the single source of truth on a Submit; the attr encoding
    // exists only inside the transient SubmitToCallView, never in real IR.
    if (op->HasAttr(kAttrManualDepEdges)) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "ManualDepsOnSubmitOnly", 1,
                                "Submit in function '" + func_name_ +
                                    "' carries `attrs[\"manual_dep_edges\"]`; dependency edges must "
                                    "live only in the typed `deps_` field.",
                                op->span_);
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  const std::string& func_name_;
};

class ManualDepsOnSubmitOnlyPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "ManualDepsOnSubmitOnly"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      ManualDepsOnSubmitOnlyChecker checker(diagnostics, func->name_);
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateManualDepsOnSubmitOnlyPropertyVerifier() {
  return std::make_shared<ManualDepsOnSubmitOnlyPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

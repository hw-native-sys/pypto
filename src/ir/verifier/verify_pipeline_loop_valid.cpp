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

#include "pypto/core/error.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {
namespace {

/// Walks a function body, reporting any ForStmt that violates the bidirectional
/// invariant between ``ForStmt.kind_`` and ``ForStmt.attrs_["pipeline_stages"]``:
///   - ``kind_ == ForKind::Pipeline``  ⇒  attr present
///   - attr present                    ⇒  ``kind_ == ForKind::Pipeline``
///
/// Either direction failing means the loop is malformed: a Pipeline-kind
/// without the stage attr would print as plain ``pl.range(...)`` (round-trip
/// breaks); a Sequential loop carrying ``pipeline_stages`` would mislead
/// LowerPipelineLoops on a re-run.
class PipelineLoopValidChecker : public IRVisitor {
 public:
  PipelineLoopValidChecker(std::vector<Diagnostic>& diagnostics, const std::string& func_name)
      : diagnostics_(diagnostics), func_name_(func_name) {}

  void VisitStmt_(const ForStmtPtr& op) override {
    const bool is_pipeline = (op->kind_ == ForKind::Pipeline);
    const bool has_attr = op->HasAttr(kPipelineStagesAttr);
    if (is_pipeline && !has_attr) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "PipelineLoopValid", 0,
                                "ForStmt in function '" + func_name_ +
                                    "' has `kind_ == ForKind::Pipeline` but no `pipeline_stages` attr. "
                                    "Pipeline loops must always carry the stage attribute (kind ⇔ attr).",
                                op->span_);
    }
    if (!is_pipeline && has_attr) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "PipelineLoopValid", 1,
                                "ForStmt in function '" + func_name_ +
                                    "' carries `attrs[\"pipeline_stages\"]` but `kind_` is not "
                                    "`ForKind::Pipeline`. The attr is only meaningful on Pipeline loops.",
                                op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  const std::string& func_name_;
};

class PipelineLoopValidPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "PipelineLoopValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      PipelineLoopValidChecker checker(diagnostics, func->name_);
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreatePipelineLoopValidPropertyVerifier() {
  return std::make_shared<PipelineLoopValidPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

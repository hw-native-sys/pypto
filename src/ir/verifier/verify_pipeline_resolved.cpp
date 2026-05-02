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
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {
namespace {

/// Walks a function body, reporting any ForStmt that carries the transient
/// ``ForKind::Pipeline`` marker. By design this marker must be gone after
/// ``CanonicalizeIOOrder``; any leftover indicates either a new code path that
/// forgot to route through the Lower→Canonicalize pair, or a pass that produced
/// a Pipeline loop without being accounted for.
///
/// The bidirectional ``kind ⇔ pipeline_stages attr`` invariant is checked
/// separately by ``PipelineLoopValid`` (a structural property always-on at every
/// pass boundary), so this verifier only checks the post-canonicalize-specific
/// invariant: no ``ForKind::Pipeline`` survives.
class PipelineKindLeftoverChecker : public IRVisitor {
 public:
  PipelineKindLeftoverChecker(std::vector<Diagnostic>& diagnostics, const std::string& func_name)
      : diagnostics_(diagnostics), func_name_(func_name) {}

  void VisitStmt_(const ForStmtPtr& op) override {
    if (op->kind_ == ForKind::Pipeline) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "PipelineResolved", 0,
                                "ForKind::Pipeline survived past CanonicalizeIOOrder in function '" +
                                    func_name_ +
                                    "'. This kind is a transient marker — LowerPipelineLoops keeps it, "
                                    "and CanonicalizeIOOrder must demote it to Sequential.",
                                op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  const std::string& func_name_;
};

class PipelineResolvedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "PipelineResolved"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      PipelineKindLeftoverChecker checker(diagnostics, func->name_);
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreatePipelineResolvedPropertyVerifier() {
  return std::make_shared<PipelineResolvedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

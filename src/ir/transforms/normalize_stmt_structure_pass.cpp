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
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/normalize_stmt_structure.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

// ============================================================================
// NormalizedStmtStructure verifier
// ============================================================================

namespace {

/**
 * @brief Checks that Function/If/For bodies are SeqStmts and consecutive
 * AssignStmt/EvalStmt in SeqStmts are wrapped in OpStmts.
 */
class NormalizedStmtVerifier : public IRVisitor {
 public:
  explicit NormalizedStmtVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void CheckBody(const StmtPtr& body, const std::string& context, const Span& span) {
    if (!body) return;
    if (!As<SeqStmts>(body)) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "NormalizedStmtStructure", 0,
                                context + " body is not a SeqStmts", span);
      return;
    }
    auto seq = As<SeqStmts>(body);
    for (const auto& stmt : seq->stmts_) {
      if (As<AssignStmt>(stmt) || As<EvalStmt>(stmt)) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "NormalizedStmtStructure", 0,
                                  context + " SeqStmts contains unwrapped AssignStmt/EvalStmt", stmt->span_);
      }
    }
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (!op) return;
    CheckBody(op->then_body_, "IfStmt then", op->span_);
    if (op->else_body_.has_value()) {
      CheckBody(op->else_body_.value(), "IfStmt else", op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;
    CheckBody(op->body_, "ForStmt", op->span_);
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class NormalizedStmtPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "NormalizedStmtStructure"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func) continue;
      NormalizedStmtVerifier verifier(diagnostics);
      verifier.CheckBody(func->body_, "Function '" + func->name_ + "'", func->span_);
      if (func->body_) {
        verifier.VisitStmt(func->body_);
      }
    }
  }
};

PropertyVerifierPtr CreateNormalizedStmtPropertyVerifier() {
  return std::make_shared<NormalizedStmtPropertyVerifierImpl>();
}

// ============================================================================
// NormalizeStmtStructure pass
// ============================================================================

namespace pass {

Pass NormalizeStmtStructure() {
  return CreateFunctionPass(ir::NormalizeStmtStructure, "NormalizeStmtStructure",
                            kNormalizeStmtStructureProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto

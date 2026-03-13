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
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verification_error.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace nested_seq_stmt {
std::string ErrorTypeToString(ErrorType type) {
  switch (type) {
    case ErrorType::SEQ_STMT_IN_SEQ_STMT:
      return "SEQ_STMT_IN_SEQ_STMT";
    default:
      return "UNKNOWN";
  }
}
}  // namespace nested_seq_stmt

namespace {

class NoNestedSeqStmtVerifier : public IRVisitor {
 public:
  explicit NoNestedSeqStmtVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const SeqStmtsPtr& op) override {
    if (!op) return;
    for (const auto& stmt : op->stmts_) {
      if (As<SeqStmts>(stmt)) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "NoNestedSeqStmt",
                                  static_cast<int>(nested_seq_stmt::ErrorType::SEQ_STMT_IN_SEQ_STMT),
                                  "SeqStmts directly nested inside another SeqStmts", stmt->span_);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

class NoNestedSeqStmtPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "NoNestedSeqStmt"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      NoNestedSeqStmtVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateNoNestedSeqStmtPropertyVerifier() {
  return std::make_shared<NoNestedSeqStmtPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

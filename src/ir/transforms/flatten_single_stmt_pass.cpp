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

#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/flatten_single_stmt.h"
#include "pypto/ir/transforms/verifier.h"

namespace pypto {
namespace ir {

// ============================================================================
// FlattenedSingleStmt verifier
// ============================================================================

namespace {

/**
 * @brief Checks no single-element SeqStmts or OpStmts exist.
 */
class FlattenedSingleStmtVerifier : public IRVisitor {
 public:
  explicit FlattenedSingleStmtVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const SeqStmtsPtr& op) override {
    if (!op) return;
    if (op->stmts_.size() == 1) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "FlattenedSingleStmt", 0,
                                "SeqStmts with single element should be flattened", op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const OpStmtsPtr& op) override {
    if (!op) return;
    if (op->stmts_.size() == 1) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "FlattenedSingleStmt", 0,
                                "OpStmts with single element should be flattened", op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class FlattenedSingleStmtPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "FlattenedSingleStmt"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      FlattenedSingleStmtVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateFlattenedSingleStmtPropertyVerifier() {
  return std::make_shared<FlattenedSingleStmtPropertyVerifierImpl>();
}

// ============================================================================
// FlattenSingleStmt pass
// ============================================================================

namespace pass {

Pass FlattenSingleStmt() {
  return CreateFunctionPass(
      ir::FlattenSingleStmt, "FlattenSingleStmt",
      {.produced = {IRProperty::FlattenedSingleStmt}, .invalidated = {IRProperty::NormalizedStmtStructure}});
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto

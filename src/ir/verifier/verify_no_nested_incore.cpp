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

#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

// ============================================================================
// NoNestedInCore structural property verifier
// ============================================================================

namespace {

constexpr int kNestedIncoreCode = 501;

/// Detects nested ScopeStmt(InCore) scopes in an IR tree.
class NestedInCoreScopeDetector : public IRVisitor {
 public:
  explicit NestedInCoreScopeDetector(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const InCoreScopeStmtPtr& op) override {
    if (!op) return;
    if (inside_incore_) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "NoNestedInCore", kNestedIncoreCode,
                                "Nested InCore scope detected — InCore scopes must not contain other "
                                "InCore scopes",
                                op->span_);
    }
    bool prev = inside_incore_;
    inside_incore_ = true;
    IRVisitor::VisitStmt_(op);
    inside_incore_ = prev;
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  bool inside_incore_ = false;
};

}  // namespace

class NoNestedIncorePropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "NoNestedInCore"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      NestedInCoreScopeDetector detector(diagnostics);
      detector.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateNoNestedIncorePropertyVerifier() {
  return std::make_shared<NoNestedIncorePropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

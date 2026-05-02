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
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
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
 * @brief Checks structural invariants of SeqStmts:
 *   - no single-child SeqStmts (should be unwrapped),
 *   - no nested SeqStmts inside another SeqStmts (should be flattened),
 *   - no YieldStmt anywhere except as the trailing statement (YieldStmt is
 *     the scope-exit terminator; mid-body yields are unreachable code and
 *     break passes that assume the yield is at the tail — see
 *     ConvertToSSA::ExtractYield/ReplaceOrAppendYield).
 */
class NoRedundantBlocksVerifier : public IRVisitor {
 public:
  explicit NoRedundantBlocksVerifier(std::vector<Diagnostic>& diagnostics,
                                     std::string rule_name = "NoRedundantBlocks")
      : diagnostics_(diagnostics), rule_name_(std::move(rule_name)) {}

  void VisitStmt_(const SeqStmtsPtr& op) override {
    if (!op) return;
    if (op->stmts_.size() == 1) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, rule_name_, 0,
                                "SeqStmts with single child should be unwrapped", op->span_);
    }
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
      const auto& stmt = op->stmts_[i];
      if (As<SeqStmts>(stmt)) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, rule_name_, 0,
                                  "SeqStmts contains nested SeqStmts", stmt->span_);
      }
      if (i + 1 < op->stmts_.size() && As<YieldStmt>(stmt)) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, rule_name_, 0,
                                  "YieldStmt before the terminating position; "
                                  "YieldStmt must be the trailing statement of its scope",
                                  stmt->span_);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string rule_name_;
};

}  // namespace

class NormalizedStmtPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "NormalizedStmtStructure"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    // NormalizedStmtStructure now only ensures flat SeqStmts (no nested SeqStmts).
    // The NoRedundantBlocks check covers this.
    NoRedundantBlocksVerifier verifier(diagnostics, GetName());
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateNormalizedStmtPropertyVerifier() {
  return std::make_shared<NormalizedStmtPropertyVerifierImpl>();
}

class NoRedundantBlocksPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "NoRedundantBlocks"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      NoRedundantBlocksVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateNoRedundantBlocksPropertyVerifier() {
  return std::make_shared<NoRedundantBlocksPropertyVerifierImpl>();
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

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
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verification_error.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace break_continue {
std::string ErrorTypeToString(ErrorType type) {
  switch (type) {
    case ErrorType::BREAK_IN_PARALLEL_LOOP:
      return "BREAK_IN_PARALLEL_LOOP";
    case ErrorType::BREAK_IN_UNROLL_LOOP:
      return "BREAK_IN_UNROLL_LOOP";
    case ErrorType::CONTINUE_IN_PARALLEL_LOOP:
      return "CONTINUE_IN_PARALLEL_LOOP";
    case ErrorType::CONTINUE_IN_UNROLL_LOOP:
      return "CONTINUE_IN_UNROLL_LOOP";
    case ErrorType::BREAK_OUTSIDE_LOOP:
      return "BREAK_OUTSIDE_LOOP";
    case ErrorType::CONTINUE_OUTSIDE_LOOP:
      return "CONTINUE_OUTSIDE_LOOP";
    default:
      return "UNKNOWN";
  }
}
}  // namespace break_continue

namespace {

class BreakContinueVerifier : public IRVisitor {
 public:
  explicit BreakContinueVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

 protected:
  void VisitStmt_(const ForStmtPtr& op) override {
    loop_kind_stack_.push_back(op->kind_);
    IRVisitor::VisitStmt_(op);
    loop_kind_stack_.pop_back();
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    loop_kind_stack_.push_back(ForKind::Sequential);
    IRVisitor::VisitStmt_(op);
    loop_kind_stack_.pop_back();
  }

  void VisitStmt_(const BreakStmtPtr& op) override {
    CheckLoopControl("break", op->span_, break_continue::ErrorType::BREAK_OUTSIDE_LOOP,
                     break_continue::ErrorType::BREAK_IN_PARALLEL_LOOP,
                     break_continue::ErrorType::BREAK_IN_UNROLL_LOOP);
  }

  void VisitStmt_(const ContinueStmtPtr& op) override {
    CheckLoopControl("continue", op->span_, break_continue::ErrorType::CONTINUE_OUTSIDE_LOOP,
                     break_continue::ErrorType::CONTINUE_IN_PARALLEL_LOOP,
                     break_continue::ErrorType::CONTINUE_IN_UNROLL_LOOP);
  }

 private:
  void CheckLoopControl(const std::string& keyword, const Span& span, break_continue::ErrorType outside_error,
                        break_continue::ErrorType parallel_error, break_continue::ErrorType unroll_error) {
    if (loop_kind_stack_.empty()) {
      RecordError(outside_error, "'" + keyword + "' outside loop", span);
    } else if (loop_kind_stack_.back() == ForKind::Parallel) {
      RecordError(parallel_error, "'" + keyword + "' not supported in parallel loops", span);
    } else if (loop_kind_stack_.back() == ForKind::Unroll) {
      RecordError(unroll_error, "'" + keyword + "' not supported in unrolled loops", span);
    }
  }

  void RecordError(break_continue::ErrorType error_type, const std::string& message, const Span& span) {
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "BreakContinueCheck", static_cast<int>(error_type),
                              message, span);
  }

  std::vector<Diagnostic>& diagnostics_;
  std::vector<ForKind> loop_kind_stack_;
};

class BreakContinuePropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "BreakContinueCheck"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    for (const auto& [global_var, func] : program->functions_) {
      if (!func) {
        continue;
      }
      BreakContinueVerifier verifier(diagnostics);
      if (func->body_) {
        verifier.VisitStmt(func->body_);
      }
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateBreakContinuePropertyVerifier() {
  return std::make_shared<BreakContinuePropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

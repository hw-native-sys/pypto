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
#include <sstream>
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verification_error.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

// Implement nested_call error type to string conversion
namespace nested_call {
std::string ErrorTypeToString(ErrorType type) {
  switch (type) {
    case ErrorType::CALL_IN_CALL_ARGS:
      return "CALL_IN_CALL_ARGS";
    case ErrorType::CALL_IN_IF_CONDITION:
      return "CALL_IN_IF_CONDITION";
    case ErrorType::CALL_IN_FOR_RANGE:
      return "CALL_IN_FOR_RANGE";
    case ErrorType::CALL_IN_BINARY_EXPR:
      return "CALL_IN_BINARY_EXPR";
    case ErrorType::CALL_IN_UNARY_EXPR:
      return "CALL_IN_UNARY_EXPR";
    case ErrorType::CALL_IN_WHILE_CONDITION:
      return "CALL_IN_WHILE_CONDITION";
    default:
      return "UNKNOWN";
  }
}
}  // namespace nested_call

namespace {

/**
 * @brief Helper visitor class for nested call verification
 *
 * Traverses the IR tree and checks that call expressions do not appear
 * in nested contexts (call arguments, if conditions, for ranges, binary/unary operands).
 */
class NoNestedCallVerifier : public IRVisitor {
 public:
  explicit NoNestedCallVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitExpr_(const CallPtr& op) override;
  void VisitExpr_(const AddPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const SubPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const MulPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const FloorDivPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const FloorModPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const FloatDivPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const MinPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const MaxPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const PowPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const EqPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const NePtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const LtPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const LePtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const GtPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const GePtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const AndPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const OrPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const XorPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const BitAndPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const BitOrPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const BitXorPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const BitShiftLeftPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const BitShiftRightPtr& op) override { VisitBinaryExpr(op); }
  void VisitExpr_(const AbsPtr& op) override { VisitUnaryExpr(op); }
  void VisitExpr_(const NegPtr& op) override { VisitUnaryExpr(op); }
  void VisitExpr_(const NotPtr& op) override { VisitUnaryExpr(op); }
  void VisitExpr_(const BitNotPtr& op) override { VisitUnaryExpr(op); }
  void VisitExpr_(const CastPtr& op) override { VisitUnaryExpr(op); }
  void VisitStmt_(const IfStmtPtr& op) override;
  void VisitStmt_(const ForStmtPtr& op) override;
  void VisitStmt_(const WhileStmtPtr& op) override;

 private:
  std::vector<Diagnostic>& diagnostics_;

  /**
   * @brief Record a nested call error
   */
  void RecordError(nested_call::ErrorType type, const std::string& message, const Span& span);

  /**
   * @brief Generic handler for binary expressions
   */
  template <typename BinaryExprType>
  void VisitBinaryExpr(const std::shared_ptr<const BinaryExprType>& op) {
    if (As<Call>(op->left_)) {
      RecordError(nested_call::ErrorType::CALL_IN_BINARY_EXPR,
                  "Binary expression left operand cannot be a call expression", op->left_->span_);
    }
    if (As<Call>(op->right_)) {
      RecordError(nested_call::ErrorType::CALL_IN_BINARY_EXPR,
                  "Binary expression right operand cannot be a call expression", op->right_->span_);
    }

    // Continue visiting sub-expressions
    IRVisitor::VisitExpr_(op);
  }

  /**
   * @brief Generic handler for unary expressions
   */
  template <typename UnaryExprType>
  void VisitUnaryExpr(const std::shared_ptr<const UnaryExprType>& op) {
    if (As<Call>(op->operand_)) {
      RecordError(nested_call::ErrorType::CALL_IN_UNARY_EXPR,
                  "Unary expression operand cannot be a call expression", op->operand_->span_);
    }

    // Continue visiting sub-expression
    IRVisitor::VisitExpr_(op);
  }
};

void NoNestedCallVerifier::RecordError(nested_call::ErrorType type, const std::string& message,
                                       const Span& span) {
  diagnostics_.emplace_back(DiagnosticSeverity::Error, "NoNestedCall", static_cast<int>(type), message, span);
}

void NoNestedCallVerifier::VisitExpr_(const CallPtr& op) {
  // Check each argument of the call
  for (const auto& arg : op->args_) {
    if (As<Call>(arg)) {
      std::ostringstream msg;
      msg << "Call expression has nested call in arguments";
      RecordError(nested_call::ErrorType::CALL_IN_CALL_ARGS, msg.str(), arg->span_);
    }
    // Continue visiting to check deeper nesting
    VisitExpr(arg);
  }
}

void NoNestedCallVerifier::VisitStmt_(const IfStmtPtr& op) {
  if (!op) return;

  // Check if condition is a call
  if (As<Call>(op->condition_)) {
    RecordError(nested_call::ErrorType::CALL_IN_IF_CONDITION, "If condition cannot be a call expression",
                op->condition_->span_);
  }

  // Continue with default traversal
  IRVisitor::VisitStmt_(op);
}

void NoNestedCallVerifier::VisitStmt_(const ForStmtPtr& op) {
  if (!op) return;

  // Check if range expressions are calls
  if (As<Call>(op->start_)) {
    RecordError(nested_call::ErrorType::CALL_IN_FOR_RANGE,
                "For loop start expression cannot be a call expression", op->start_->span_);
  }
  if (As<Call>(op->stop_)) {
    RecordError(nested_call::ErrorType::CALL_IN_FOR_RANGE,
                "For loop stop expression cannot be a call expression", op->stop_->span_);
  }
  if (As<Call>(op->step_)) {
    RecordError(nested_call::ErrorType::CALL_IN_FOR_RANGE,
                "For loop step expression cannot be a call expression", op->step_->span_);
  }

  // Continue with default traversal
  IRVisitor::VisitStmt_(op);
}

void NoNestedCallVerifier::VisitStmt_(const WhileStmtPtr& op) {
  if (!op) return;

  // Check if condition is a call
  if (As<Call>(op->condition_)) {
    RecordError(nested_call::ErrorType::CALL_IN_WHILE_CONDITION,
                "While loop condition cannot be a call expression", op->condition_->span_);
  }

  // Continue with default traversal
  IRVisitor::VisitStmt_(op);
}

}  // namespace

/**
 * @brief No nested call property verifier for use with IRVerifier
 */
class NoNestedCallPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "NoNestedCall"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) {
      return;
    }

    for (const auto& [global_var, func] : program->functions_) {
      if (!func) {
        continue;
      }

      // Create verifier and run verification
      NoNestedCallVerifier verifier(diagnostics);

      // Visit function body
      if (func->body_) {
        verifier.VisitStmt(func->body_);
      }
    }
  }
};

// Factory function for creating NoNestedCall property verifier
PropertyVerifierPtr CreateNoNestedCallPropertyVerifier() {
  return std::make_shared<NoNestedCallPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto

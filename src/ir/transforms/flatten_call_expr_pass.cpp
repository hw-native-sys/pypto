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

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/flatten_single_stmt.h"
#include "pypto/ir/transforms/utils/normalize_stmt_structure.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/**
 * @brief Mutator that flattens nested call expressions into three-address code
 *
 * This pass ensures that:
 * 1. Call arguments cannot be calls
 * 2. If conditions cannot be calls
 * 3. For loop ranges (start/stop/step) cannot be calls
 * 4. Binary/unary expression operands cannot be calls
 *
 * Nested calls are extracted into temporary variables and inserted as
 * AssignStmt before the statement containing the nested call.
 *
 * For if/for statements, extracted statements are inserted into the
 * last OpStmts before the if/for, or a new OpStmts is created if needed.
 */
class FlattenCallExprMutator : public IRMutator {
 public:
  FlattenCallExprMutator() = default;

  // Statement visitors
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override;
  StmtPtr VisitStmt_(const OpStmtsPtr& op) override;
  StmtPtr VisitStmt_(const IfStmtPtr& op) override;
  StmtPtr VisitStmt_(const ForStmtPtr& op) override;
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override;

  // Expression visitors
  ExprPtr VisitExpr_(const CallPtr& op) override;
  ExprPtr VisitExpr_(const AddPtr& op) override;
  ExprPtr VisitExpr_(const SubPtr& op) override;
  ExprPtr VisitExpr_(const MulPtr& op) override;
  ExprPtr VisitExpr_(const FloorDivPtr& op) override;
  ExprPtr VisitExpr_(const FloorModPtr& op) override;
  ExprPtr VisitExpr_(const FloatDivPtr& op) override;
  ExprPtr VisitExpr_(const MinPtr& op) override;
  ExprPtr VisitExpr_(const MaxPtr& op) override;
  ExprPtr VisitExpr_(const PowPtr& op) override;
  ExprPtr VisitExpr_(const EqPtr& op) override;
  ExprPtr VisitExpr_(const NePtr& op) override;
  ExprPtr VisitExpr_(const LtPtr& op) override;
  ExprPtr VisitExpr_(const LePtr& op) override;
  ExprPtr VisitExpr_(const GtPtr& op) override;
  ExprPtr VisitExpr_(const GePtr& op) override;
  ExprPtr VisitExpr_(const AndPtr& op) override;
  ExprPtr VisitExpr_(const OrPtr& op) override;
  ExprPtr VisitExpr_(const XorPtr& op) override;
  ExprPtr VisitExpr_(const BitAndPtr& op) override;
  ExprPtr VisitExpr_(const BitOrPtr& op) override;
  ExprPtr VisitExpr_(const BitXorPtr& op) override;
  ExprPtr VisitExpr_(const BitShiftLeftPtr& op) override;
  ExprPtr VisitExpr_(const BitShiftRightPtr& op) override;
  ExprPtr VisitExpr_(const AbsPtr& op) override;
  ExprPtr VisitExpr_(const NegPtr& op) override;
  ExprPtr VisitExpr_(const NotPtr& op) override;
  ExprPtr VisitExpr_(const BitNotPtr& op) override;
  ExprPtr VisitExpr_(const CastPtr& op) override;

 private:
  int temp_var_counter_ = 0;
  std::vector<StmtPtr> pending_stmts_;

  /**
   * @brief Generate a unique temporary variable name
   */
  std::string GenerateTempVarName() {
    std::ostringstream oss;
    oss << "_t" << temp_var_counter_++;
    return oss.str();
  }

  /**
   * @brief Extract a call expression into a temporary variable
   *
   * Creates a new temporary variable, generates an assignment statement,
   * and adds it to pending_stmts_.
   *
   * @param expr Expression (must be a Call)
   * @return Var expression referring to the temporary variable
   */
  ExprPtr ExtractCallToTemp(const ExprPtr& expr) {
    if (!As<Call>(expr)) {
      return expr;
    }

    // Create temporary variable
    std::string temp_name = GenerateTempVarName();
    auto temp_var = std::make_shared<Var>(temp_name, expr->GetType(), expr->span_);

    // Create assignment statement
    auto assign = std::make_shared<AssignStmt>(temp_var, expr, expr->span_);
    pending_stmts_.push_back(assign);

    return temp_var;
  }

  /**
   * @brief Process binary expression, extracting any call operands
   */
  template <typename BinaryExprType>
  ExprPtr ProcessBinaryExpr(const std::shared_ptr<const BinaryExprType>& op) {
    auto new_left = VisitExpr(op->left_);
    auto new_right = VisitExpr(op->right_);

    // Extract calls from operands
    if (As<Call>(new_left)) {
      new_left = ExtractCallToTemp(new_left);
    }
    if (As<Call>(new_right)) {
      new_right = ExtractCallToTemp(new_right);
    }

    bool changed = (new_left.get() != op->left_.get()) || (new_right.get() != op->right_.get());
    if (changed) {
      // Extract DataType from op's type (which should be ScalarType)
      auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(op->GetType());
      INTERNAL_CHECK(scalar_type) << "Binary expression type must be ScalarType";
      return std::make_shared<const BinaryExprType>(new_left, new_right, scalar_type->dtype_, op->span_);
    }
    return op;
  }

  /**
   * @brief Process unary expression, extracting any call operand
   */
  template <typename UnaryExprType>
  ExprPtr ProcessUnaryExpr(const std::shared_ptr<const UnaryExprType>& op) {
    auto new_operand = VisitExpr(op->operand_);

    // Extract call from operand
    if (As<Call>(new_operand)) {
      new_operand = ExtractCallToTemp(new_operand);
    }

    if (new_operand.get() != op->operand_.get()) {
      // Extract DataType from op's type (which should be ScalarType)
      auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(op->GetType());
      INTERNAL_CHECK(scalar_type) << "Unary expression type must be ScalarType";
      return std::make_shared<const UnaryExprType>(new_operand, scalar_type->dtype_, op->span_);
    }
    return op;
  }
};

// Statement visitors implementation

StmtPtr FlattenCallExprMutator::VisitStmt_(const SeqStmtsPtr& op) {
  std::vector<StmtPtr> new_stmts;

  for (const auto& stmt : op->stmts_) {
    pending_stmts_.clear();

    auto new_stmt = VisitStmt(stmt);

    // If there are pending statements, insert them into an OpStmts before the current stmt
    if (!pending_stmts_.empty()) {
      // Find the last OpStmts before current position
      int last_opstmts_idx = -1;
      for (int j = static_cast<int>(new_stmts.size()) - 1; j >= 0; --j) {
        if (As<OpStmts>(new_stmts[j])) {
          last_opstmts_idx = j;
          break;
        }
      }

      if (last_opstmts_idx >= 0) {
        // Append pending stmts to the existing OpStmts
        auto old_opstmts = As<OpStmts>(new_stmts[last_opstmts_idx]);
        std::vector<StmtPtr> merged = old_opstmts->stmts_;
        merged.insert(merged.end(), pending_stmts_.begin(), pending_stmts_.end());
        new_stmts[last_opstmts_idx] = std::make_shared<const OpStmts>(merged, old_opstmts->span_);
      } else {
        // No preceding OpStmts found, create a new one
        new_stmts.push_back(std::make_shared<const OpStmts>(pending_stmts_, new_stmt->span_));
      }
    }

    new_stmts.push_back(new_stmt);
  }

  return std::make_shared<SeqStmts>(new_stmts, op->span_);
}

StmtPtr FlattenCallExprMutator::VisitStmt_(const OpStmtsPtr& op) {
  std::vector<StmtPtr> new_stmts;

  for (const auto& stmt : op->stmts_) {
    pending_stmts_.clear();

    auto new_stmt = VisitStmt(stmt);

    // Insert extracted statements (all AssignStmt, compatible with OpStmts)
    for (const auto& pending : pending_stmts_) {
      new_stmts.push_back(pending);
    }
    // Insert the original statement
    new_stmts.push_back(new_stmt);
  }

  // Clear pending_stmts_ after processing all statements
  // This prevents temporaries from leaking out of OpStmts
  pending_stmts_.clear();

  return std::make_shared<const OpStmts>(new_stmts, op->span_);
}

StmtPtr FlattenCallExprMutator::VisitStmt_(const IfStmtPtr& op) {
  // Note: Don't clear pending_stmts_, preserve previous state

  auto new_condition = VisitExpr(op->condition_);

  // If condition is a call, extract to temporary variable
  if (As<Call>(new_condition)) {
    new_condition = ExtractCallToTemp(new_condition);
  }

  // Save condition pending stmts (will be handled by parent SeqStmts)
  auto condition_pending = pending_stmts_;

  // Process then branch (after normalization, body is SeqStmts which handles its own pending)
  pending_stmts_.clear();
  auto new_then = VisitStmt(op->then_body_);

  // Process else branch
  pending_stmts_.clear();
  std::optional<StmtPtr> new_else;
  if (op->else_body_.has_value()) {
    new_else = VisitStmt(op->else_body_.value());
  }

  // Restore condition pending for parent to handle
  pending_stmts_ = condition_pending;

  return std::make_shared<IfStmt>(new_condition, new_then, new_else, op->return_vars_, op->span_);
}

StmtPtr FlattenCallExprMutator::VisitStmt_(const ForStmtPtr& op) {
  // Note: Don't clear pending_stmts_, preserve previous state

  auto new_start = VisitExpr(op->start_);
  auto new_stop = VisitExpr(op->stop_);
  auto new_step = VisitExpr(op->step_);

  // Extract calls from range
  if (As<Call>(new_start)) {
    new_start = ExtractCallToTemp(new_start);
  }
  if (As<Call>(new_stop)) {
    new_stop = ExtractCallToTemp(new_stop);
  }
  if (As<Call>(new_step)) {
    new_step = ExtractCallToTemp(new_step);
  }

  // Save range pending stmts (will be handled by parent SeqStmts)
  auto range_pending = pending_stmts_;

  // Process body (after normalization, body is SeqStmts which handles its own pending)
  pending_stmts_.clear();
  auto new_body = VisitStmt(op->body_);

  // Restore range pending for parent to handle
  pending_stmts_ = range_pending;

  return std::make_shared<ForStmt>(op->loop_var_, new_start, new_stop, new_step, op->iter_args_, new_body,
                                   op->return_vars_, op->span_, op->kind_);
}

StmtPtr FlattenCallExprMutator::VisitStmt_(const WhileStmtPtr& op) {
  // Note: Don't clear pending_stmts_, preserve previous state

  auto new_condition = VisitExpr(op->condition_);

  // Extract calls from condition
  if (As<Call>(new_condition)) {
    new_condition = ExtractCallToTemp(new_condition);
  }

  // Save condition pending stmts (will be handled by parent SeqStmts)
  auto condition_pending = pending_stmts_;

  // Process body (after normalization, body is SeqStmts which handles its own pending)
  pending_stmts_.clear();
  auto new_body = VisitStmt(op->body_);

  // Restore condition pending for parent to handle
  pending_stmts_ = condition_pending;

  return std::make_shared<WhileStmt>(new_condition, op->iter_args_, new_body, op->return_vars_, op->span_);
}

// Expression visitors implementation

ExprPtr FlattenCallExprMutator::VisitExpr_(const CallPtr& op) {
  std::vector<ExprPtr> new_args;
  bool changed = false;

  for (const auto& arg : op->args_) {
    auto visited_arg = VisitExpr(arg);

    // If argument is a call, extract to temporary variable
    if (As<Call>(visited_arg)) {
      auto temp_var = ExtractCallToTemp(visited_arg);
      new_args.push_back(temp_var);
      changed = true;
    } else {
      new_args.push_back(visited_arg);
      if (visited_arg.get() != arg.get()) {
        changed = true;
      }
    }
  }

  if (changed) {
    return std::make_shared<Call>(op->op_, new_args, op->kwargs_, op->GetType(), op->span_);
  }
  return op;
}

// Binary expression visitors
ExprPtr FlattenCallExprMutator::VisitExpr_(const AddPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const SubPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const MulPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const FloorDivPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const FloorModPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const FloatDivPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const MinPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const MaxPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const PowPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const EqPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const NePtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const LtPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const LePtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const GtPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const GePtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const AndPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const OrPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const XorPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const BitAndPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const BitOrPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const BitXorPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const BitShiftLeftPtr& op) { return ProcessBinaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const BitShiftRightPtr& op) { return ProcessBinaryExpr(op); }

// Unary expression visitors
ExprPtr FlattenCallExprMutator::VisitExpr_(const AbsPtr& op) { return ProcessUnaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const NegPtr& op) { return ProcessUnaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const NotPtr& op) { return ProcessUnaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const BitNotPtr& op) { return ProcessUnaryExpr(op); }
ExprPtr FlattenCallExprMutator::VisitExpr_(const CastPtr& op) { return ProcessUnaryExpr(op); }

/**
 * @brief Transform a function by flattening nested call expressions
 *
 * Pipeline:
 * 1. NormalizeStmtStructure - ensure bodies are SeqStmts, ops wrapped in OpStmts
 * 2. FlattenCallExprMutator - extract nested calls into temporaries
 * 3. FlattenSingleStmt - remove unnecessary single-statement wrappers
 */
FunctionPtr TransformFlattenCallExpr(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "FlattenCallExpr cannot run on null function";

  // Step 1: Normalize statement structure
  auto normalized = NormalizeStmtStructure(func);

  // Step 2: Flatten call expressions
  FlattenCallExprMutator mutator;
  auto new_body = mutator.VisitStmt(normalized->body_);
  auto result = std::make_shared<Function>(normalized->name_, normalized->params_, normalized->return_types_,
                                           new_body, normalized->span_, normalized->func_type_);

  // Step 3: Flatten single-statement blocks
  return FlattenSingleStmt(result);
}

}  // namespace

// Factory function
namespace pass {
Pass FlattenCallExpr() {
  return CreateFunctionPass(TransformFlattenCallExpr, "FlattenCallExpr", kFlattenCallExprProperties);
}
}  // namespace pass

}  // namespace ir
}  // namespace pypto

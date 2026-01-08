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

#include "pypto/ir/transform/printer.h"

#include <string>

#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {

// Precedence mapping for each expression type
Precedence GetPrecedence(const ExprPtr& expr) {
  // Logical operators
  if (std::dynamic_pointer_cast<const Or>(expr)) return Precedence::kOr;
  if (std::dynamic_pointer_cast<const Xor>(expr)) return Precedence::kXor;
  if (std::dynamic_pointer_cast<const And>(expr)) return Precedence::kAnd;
  if (std::dynamic_pointer_cast<const Not>(expr)) return Precedence::kNot;

  // Comparison operators
  if (std::dynamic_pointer_cast<const Eq>(expr)) return Precedence::kComparison;
  if (std::dynamic_pointer_cast<const Ne>(expr)) return Precedence::kComparison;
  if (std::dynamic_pointer_cast<const Lt>(expr)) return Precedence::kComparison;
  if (std::dynamic_pointer_cast<const Le>(expr)) return Precedence::kComparison;
  if (std::dynamic_pointer_cast<const Gt>(expr)) return Precedence::kComparison;
  if (std::dynamic_pointer_cast<const Ge>(expr)) return Precedence::kComparison;

  // Bitwise operators
  if (std::dynamic_pointer_cast<const BitOr>(expr)) return Precedence::kBitOr;
  if (std::dynamic_pointer_cast<const BitXor>(expr)) return Precedence::kBitXor;
  if (std::dynamic_pointer_cast<const BitAnd>(expr)) return Precedence::kBitAnd;
  if (std::dynamic_pointer_cast<const BitShiftLeft>(expr)) return Precedence::kBitShift;
  if (std::dynamic_pointer_cast<const BitShiftRight>(expr)) return Precedence::kBitShift;

  // Arithmetic operators
  if (std::dynamic_pointer_cast<const Add>(expr)) return Precedence::kAddSub;
  if (std::dynamic_pointer_cast<const Sub>(expr)) return Precedence::kAddSub;
  if (std::dynamic_pointer_cast<const Mul>(expr)) return Precedence::kMulDivMod;
  if (std::dynamic_pointer_cast<const FloorDiv>(expr)) return Precedence::kMulDivMod;
  if (std::dynamic_pointer_cast<const FloatDiv>(expr)) return Precedence::kMulDivMod;
  if (std::dynamic_pointer_cast<const FloorMod>(expr)) return Precedence::kMulDivMod;
  if (std::dynamic_pointer_cast<const Pow>(expr)) return Precedence::kPow;

  // Unary operators
  if (std::dynamic_pointer_cast<const Neg>(expr)) return Precedence::kUnary;
  if (std::dynamic_pointer_cast<const BitNot>(expr)) return Precedence::kUnary;

  // Function-like operators and atoms
  if (std::dynamic_pointer_cast<const Abs>(expr)) return Precedence::kCall;
  if (std::dynamic_pointer_cast<const Min>(expr)) return Precedence::kCall;
  if (std::dynamic_pointer_cast<const Max>(expr)) return Precedence::kCall;
  if (std::dynamic_pointer_cast<const Call>(expr)) return Precedence::kCall;
  if (std::dynamic_pointer_cast<const Var>(expr)) return Precedence::kAtom;
  if (std::dynamic_pointer_cast<const ConstInt>(expr)) return Precedence::kAtom;

  // Default: treat as atom
  return Precedence::kAtom;
}

bool IsRightAssociative(const ExprPtr& expr) {
  // Only ** (power) is right-associative in Python
  return std::dynamic_pointer_cast<const Pow>(expr) != nullptr;
}

std::string IRPrinter::Print(const ExprPtr& expr) {
  stream_.str("");  // Clear the stream
  stream_.clear();  // Clear any error flags
  VisitExpr(expr);
  return stream_.str();
}

std::string IRPrinter::Print(const StmtPtr& stmt) {
  stream_.str("");  // Clear the stream
  stream_.clear();  // Clear any error flags
  VisitStmt(stmt);
  return stream_.str();
}

// Leaf nodes
void IRPrinter::VisitExpr_(const VarPtr& op) { stream_ << op->name_; }

void IRPrinter::VisitExpr_(const ConstIntPtr& op) { stream_ << op->value_; }

void IRPrinter::VisitExpr_(const CallPtr& op) {
  stream_ << op->op_->name_ << "(";
  for (size_t i = 0; i < op->args_.size(); ++i) {
    if (i > 0) stream_ << ", ";
    VisitExpr(op->args_[i]);
  }
  stream_ << ")";
}

// Helper methods
bool IRPrinter::NeedsParens(const ExprPtr& parent, const ExprPtr& child, bool is_left) {
  Precedence parent_prec = GetPrecedence(parent);
  Precedence child_prec = GetPrecedence(child);

  // Rule 1: Child with lower precedence always needs parens
  if (child_prec < parent_prec) {
    return true;
  }

  // Rule 2: Same precedence - check associativity
  if (child_prec == parent_prec) {
    // Right-associative operators (like **): left child needs parens
    // Left-associative operators: right child needs parens
    // This preserves the tree structure: a - (b - c) vs (a - b) - c
    if (IsRightAssociative(parent)) {
      return is_left;  // Left child needs parens for right-assoc
    } else {
      return !is_left;  // Right child needs parens for left-assoc
    }
  }

  // Rule 3: Higher precedence child never needs parens
  return false;
}

void IRPrinter::PrintChild(const ExprPtr& parent, const ExprPtr& child, bool is_left) {
  bool needs_parens = NeedsParens(parent, child, is_left);

  if (needs_parens) {
    stream_ << "(";
  }

  VisitExpr(child);

  if (needs_parens) {
    stream_ << ")";
  }
}

void IRPrinter::PrintBinaryOp(const BinaryExprPtr& op, const char* op_symbol) {
  PrintChild(op, op->left_, true);
  stream_ << " " << op_symbol << " ";
  PrintChild(op, op->right_, false);
}

void IRPrinter::PrintFunctionBinaryOp(const BinaryExprPtr& op, const char* func_name) {
  stream_ << func_name << "(";
  VisitExpr(op->left_);
  stream_ << ", ";
  VisitExpr(op->right_);
  stream_ << ")";
}

// Arithmetic binary operators
void IRPrinter::VisitExpr_(const AddPtr& op) { PrintBinaryOp(op, "+"); }
void IRPrinter::VisitExpr_(const SubPtr& op) { PrintBinaryOp(op, "-"); }
void IRPrinter::VisitExpr_(const MulPtr& op) { PrintBinaryOp(op, "*"); }
void IRPrinter::VisitExpr_(const FloorDivPtr& op) { PrintBinaryOp(op, "//"); }
void IRPrinter::VisitExpr_(const FloorModPtr& op) { PrintBinaryOp(op, "%"); }
void IRPrinter::VisitExpr_(const FloatDivPtr& op) { PrintBinaryOp(op, "/"); }
void IRPrinter::VisitExpr_(const PowPtr& op) { PrintBinaryOp(op, "**"); }

// Function-style binary operators
void IRPrinter::VisitExpr_(const MinPtr& op) { PrintFunctionBinaryOp(op, "min"); }
void IRPrinter::VisitExpr_(const MaxPtr& op) { PrintFunctionBinaryOp(op, "max"); }

// Comparison operators
void IRPrinter::VisitExpr_(const EqPtr& op) { PrintBinaryOp(op, "=="); }
void IRPrinter::VisitExpr_(const NePtr& op) { PrintBinaryOp(op, "!="); }
void IRPrinter::VisitExpr_(const LtPtr& op) { PrintBinaryOp(op, "<"); }
void IRPrinter::VisitExpr_(const LePtr& op) { PrintBinaryOp(op, "<="); }
void IRPrinter::VisitExpr_(const GtPtr& op) { PrintBinaryOp(op, ">"); }
void IRPrinter::VisitExpr_(const GePtr& op) { PrintBinaryOp(op, ">="); }

// Logical operators (Python keywords)
void IRPrinter::VisitExpr_(const AndPtr& op) { PrintBinaryOp(op, "and"); }
void IRPrinter::VisitExpr_(const OrPtr& op) { PrintBinaryOp(op, "or"); }
void IRPrinter::VisitExpr_(const XorPtr& op) { PrintBinaryOp(op, "xor"); }

// Bitwise operators
void IRPrinter::VisitExpr_(const BitAndPtr& op) { PrintBinaryOp(op, "&"); }
void IRPrinter::VisitExpr_(const BitOrPtr& op) { PrintBinaryOp(op, "|"); }
void IRPrinter::VisitExpr_(const BitXorPtr& op) { PrintBinaryOp(op, "^"); }
void IRPrinter::VisitExpr_(const BitShiftLeftPtr& op) { PrintBinaryOp(op, "<<"); }
void IRPrinter::VisitExpr_(const BitShiftRightPtr& op) { PrintBinaryOp(op, ">>"); }

// Unary operators
void IRPrinter::VisitExpr_(const NegPtr& op) {
  stream_ << "-";
  // Unary operators need parens for their operands if operand has lower precedence
  Precedence operand_prec = GetPrecedence(op->operand_);
  if (operand_prec < Precedence::kUnary) {
    stream_ << "(";
    VisitExpr(op->operand_);
    stream_ << ")";
  } else {
    VisitExpr(op->operand_);
  }
}

void IRPrinter::VisitExpr_(const AbsPtr& op) {
  stream_ << "abs(";
  VisitExpr(op->operand_);
  stream_ << ")";
}

void IRPrinter::VisitExpr_(const NotPtr& op) {
  stream_ << "not ";
  // Not operator needs parens for operands with lower precedence
  Precedence operand_prec = GetPrecedence(op->operand_);
  if (operand_prec < Precedence::kNot) {
    stream_ << "(";
    VisitExpr(op->operand_);
    stream_ << ")";
  } else {
    VisitExpr(op->operand_);
  }
}

void IRPrinter::VisitExpr_(const BitNotPtr& op) {
  stream_ << "~";
  // Bitwise not needs parens for operands with lower precedence
  Precedence operand_prec = GetPrecedence(op->operand_);
  if (operand_prec < Precedence::kUnary) {
    stream_ << "(";
    VisitExpr(op->operand_);
    stream_ << ")";
  } else {
    VisitExpr(op->operand_);
  }
}

// Statement types
void IRPrinter::VisitStmt_(const AssignStmtPtr& op) {
  // Print assignment: var = value
  VisitExpr(op->var_);
  stream_ << " = ";
  VisitExpr(op->value_);
}

void IRPrinter::VisitStmt_(const IfStmtPtr& op) {
  // Print if statement: if condition:\n  then_body\nelse:\n  else_body
  stream_ << "if ";
  VisitExpr(op->condition_);
  stream_ << ":\n";
  for (size_t i = 0; i < op->then_body_.size(); ++i) {
    stream_ << "  ";
    VisitStmt(op->then_body_[i]);
    if (i < op->then_body_.size() - 1 || !op->else_body_.empty()) {
      stream_ << "\n";
    }
  }
  if (!op->else_body_.empty()) {
    stream_ << "else:\n";
    for (size_t i = 0; i < op->else_body_.size(); ++i) {
      stream_ << "  ";
      VisitStmt(op->else_body_[i]);
      if (i < op->else_body_.size() - 1) {
        stream_ << "\n";
      }
    }
  }
}

void IRPrinter::VisitStmt_(const YieldStmtPtr& op) {
  // Print yield statement: yield value1, value2, ... or yield
  stream_ << "yield";
  if (!op->value_.empty()) {
    stream_ << " ";
    for (size_t i = 0; i < op->value_.size(); ++i) {
      if (i > 0) stream_ << ", ";
      VisitExpr(op->value_[i]);
    }
  }
}

void IRPrinter::VisitStmt_(const StmtPtr& op) {
  // Base Stmt: just print the type name
  stream_ << op->TypeName();
}

}  // namespace ir
}  // namespace pypto

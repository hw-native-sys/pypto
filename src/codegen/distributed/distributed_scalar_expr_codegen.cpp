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

// Python emission for scalar arithmetic / comparison / logical / bitwise /
// unary IR nodes. Mirrors src/codegen/pto/pto_scalar_expr_codegen.cpp at the
// distributed host_orch layer. Each ``VisitExpr_`` override forwards to one
// of the small helpers below and lands its rendered Python text in
// ``current_expr_value_`` so the surrounding statement visitor can splice
// it into a ``var = ...`` line or a ``chip_args.add_scalar(...)`` arg.

#include <string>

#include "pypto/codegen/distributed/distributed_codegen.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

// ========================================================================
// Render helpers
// ========================================================================

void DistributedCodegen::EmitInfixBinaryOp(const ir::BinaryExprPtr& op, const char* symbol) {
  VisitExpr(op->left_);
  std::string left = current_expr_value_;
  current_expr_value_ = "";
  VisitExpr(op->right_);
  std::string right = current_expr_value_;
  current_expr_value_ = "(" + left + " " + symbol + " " + right + ")";
}

void DistributedCodegen::EmitCallStyleBinaryOp(const ir::BinaryExprPtr& op, const char* func_name) {
  VisitExpr(op->left_);
  std::string left = current_expr_value_;
  current_expr_value_ = "";
  VisitExpr(op->right_);
  std::string right = current_expr_value_;
  current_expr_value_ = std::string(func_name) + "(" + left + ", " + right + ")";
}

void DistributedCodegen::EmitUnaryPrefixOp(const ir::UnaryExprPtr& op, const char* prefix) {
  VisitExpr(op->operand_);
  std::string operand = current_expr_value_;
  current_expr_value_ = "(" + std::string(prefix) + operand + ")";
}

void DistributedCodegen::EmitUnaryCallOp(const ir::UnaryExprPtr& op, const char* func_name) {
  VisitExpr(op->operand_);
  std::string operand = current_expr_value_;
  current_expr_value_ = std::string(func_name) + "(" + operand + ")";
}

// ========================================================================
// Binary arithmetic
// ========================================================================

void DistributedCodegen::VisitExpr_(const ir::AddPtr& op) { EmitInfixBinaryOp(op, "+"); }
void DistributedCodegen::VisitExpr_(const ir::SubPtr& op) { EmitInfixBinaryOp(op, "-"); }
void DistributedCodegen::VisitExpr_(const ir::MulPtr& op) { EmitInfixBinaryOp(op, "*"); }
void DistributedCodegen::VisitExpr_(const ir::FloorDivPtr& op) { EmitInfixBinaryOp(op, "//"); }
void DistributedCodegen::VisitExpr_(const ir::FloorModPtr& op) { EmitInfixBinaryOp(op, "%"); }
void DistributedCodegen::VisitExpr_(const ir::FloatDivPtr& op) { EmitInfixBinaryOp(op, "/"); }
void DistributedCodegen::VisitExpr_(const ir::PowPtr& op) { EmitInfixBinaryOp(op, "**"); }
void DistributedCodegen::VisitExpr_(const ir::MinPtr& op) { EmitCallStyleBinaryOp(op, "min"); }
void DistributedCodegen::VisitExpr_(const ir::MaxPtr& op) { EmitCallStyleBinaryOp(op, "max"); }

// ========================================================================
// Comparison
// ========================================================================

void DistributedCodegen::VisitExpr_(const ir::EqPtr& op) { EmitInfixBinaryOp(op, "=="); }
void DistributedCodegen::VisitExpr_(const ir::NePtr& op) { EmitInfixBinaryOp(op, "!="); }
void DistributedCodegen::VisitExpr_(const ir::LtPtr& op) { EmitInfixBinaryOp(op, "<"); }
void DistributedCodegen::VisitExpr_(const ir::LePtr& op) { EmitInfixBinaryOp(op, "<="); }
void DistributedCodegen::VisitExpr_(const ir::GtPtr& op) { EmitInfixBinaryOp(op, ">"); }
void DistributedCodegen::VisitExpr_(const ir::GePtr& op) { EmitInfixBinaryOp(op, ">="); }

// ========================================================================
// Logical
// ========================================================================

void DistributedCodegen::VisitExpr_(const ir::AndPtr& op) { EmitInfixBinaryOp(op, "and"); }
void DistributedCodegen::VisitExpr_(const ir::OrPtr& op) { EmitInfixBinaryOp(op, "or"); }
// Python has no logical-xor keyword; ``^`` on bools matches ``a != b`` semantics,
// which is what every other PyPTO codegen layer uses for the Xor IR node.
void DistributedCodegen::VisitExpr_(const ir::XorPtr& op) { EmitInfixBinaryOp(op, "^"); }

// ========================================================================
// Bitwise
// ========================================================================

void DistributedCodegen::VisitExpr_(const ir::BitAndPtr& op) { EmitInfixBinaryOp(op, "&"); }
void DistributedCodegen::VisitExpr_(const ir::BitOrPtr& op) { EmitInfixBinaryOp(op, "|"); }
void DistributedCodegen::VisitExpr_(const ir::BitXorPtr& op) { EmitInfixBinaryOp(op, "^"); }
void DistributedCodegen::VisitExpr_(const ir::BitShiftLeftPtr& op) { EmitInfixBinaryOp(op, "<<"); }
void DistributedCodegen::VisitExpr_(const ir::BitShiftRightPtr& op) { EmitInfixBinaryOp(op, ">>"); }

// ========================================================================
// Unary
// ========================================================================

void DistributedCodegen::VisitExpr_(const ir::NegPtr& op) { EmitUnaryPrefixOp(op, "-"); }
void DistributedCodegen::VisitExpr_(const ir::NotPtr& op) { EmitUnaryPrefixOp(op, "not "); }
void DistributedCodegen::VisitExpr_(const ir::BitNotPtr& op) { EmitUnaryPrefixOp(op, "~"); }
void DistributedCodegen::VisitExpr_(const ir::AbsPtr& op) { EmitUnaryCallOp(op, "abs"); }

// Cast: route int/float/bool dtype to the matching Python builtin. Index
// (compile-time integer-like) is treated as a Python int.
void DistributedCodegen::VisitExpr_(const ir::CastPtr& op) {
  auto scalar_type = ir::As<ir::ScalarType>(op->GetType());
  INTERNAL_CHECK_SPAN(scalar_type, op->span_) << "Cast has non-scalar type";
  const DataType& dt = scalar_type->dtype_;
  const char* py_builtin = nullptr;
  if (dt.IsFloat()) {
    py_builtin = "float";
  } else if (dt == DataType::BOOL) {
    py_builtin = "bool";
  } else {
    py_builtin = "int";
  }
  EmitUnaryCallOp(op, py_builtin);
}

// DimExpr: unwrap to the inner dimension expression for codegen.
void DistributedCodegen::VisitExpr_(const ir::DimExprPtr& op) { VisitExpr(op->body_); }

}  // namespace codegen
}  // namespace pypto

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

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using ir::As;
using ir::BinaryExprPtr;
using ir::ScalarType;

// ========================================================================
// Expression visitor helpers - arithmetic and comparison
// ========================================================================

std::string PTOCodegen::EmitArithBinaryOp(const std::string& mlir_op, const std::string& lhs,
                                          const std::string& rhs, const std::string& result_type) {
  std::string result = NewTemp();
  Emit(result + " = " + mlir_op + " " + lhs + ", " + rhs + " : " + result_type);
  return result;
}

std::string PTOCodegen::EmitArithCmpi(const std::string& predicate, const std::string& lhs,
                                      const std::string& rhs, const std::string& operand_type) {
  std::string result = NewTemp();
  Emit(result + " = arith.cmpi " + predicate + ", " + lhs + ", " + rhs + " : " + operand_type);
  return result;
}

std::string PTOCodegen::EmitArithOperand(const ir::ExprPtr& expr, const std::string& wanted_mlir_type) {
  CHECK(!wanted_mlir_type.empty()) << "EmitArithOperand: empty wanted_mlir_type";

  if (wanted_mlir_type == "index") {
    if (auto ci = As<ir::ConstInt>(expr)) {
      return GetOrEmitConstant(ci->value_, DataType::INDEX);
    }
  } else if (wanted_mlir_type == "i64") {
    if (auto ci = As<ir::ConstInt>(expr)) {
      return GetOrEmitConstant(ci->value_, DataType::INT64);
    }
  } else if (wanted_mlir_type == "i32") {
    if (auto ci = As<ir::ConstInt>(expr)) {
      return GetOrEmitConstant(static_cast<int64_t>(static_cast<int32_t>(ci->value_)), DataType::INT32);
    }
  }

  VisitExpr(expr);
  std::string ssa = fs_.current_expr_value;

  const ::pypto::DataType dt = ir::GetScalarDtype(expr);
  if (dt.IsFloat()) {
    CHECK(wanted_mlir_type == GetTypeString(dt)) << "EmitArithOperand: float type mismatch (want "
                                                 << wanted_mlir_type << ", have " << dt.ToString() << ")";
    return ssa;
  }

  std::string have_mlir = (dt == ::pypto::DataType::INDEX) ? "index" : GetTypeString(dt);
  if (have_mlir == wanted_mlir_type) {
    return ssa;
  }

  std::string casted = NewTemp();
  if (have_mlir == "index") {
    Emit(casted + " = arith.index_cast " + ssa + " : index to " + wanted_mlir_type);
    return casted;
  }
  if (wanted_mlir_type == "index") {
    Emit(casted + " = arith.index_cast " + ssa + " : " + have_mlir + " to index");
    return casted;
  }
  if (have_mlir == "i32" && wanted_mlir_type == "i64") {
    Emit(casted + " = arith.extsi " + ssa + " : i32 to i64");
    return casted;
  }
  if (have_mlir == "i64" && wanted_mlir_type == "i32") {
    Emit(casted + " = arith.trunci " + ssa + " : i64 to i32");
    return casted;
  }
  CHECK(false) << "EmitArithOperand: unsupported cast from " << have_mlir << " to " << wanted_mlir_type;
  return ssa;
}

void PTOCodegen::VisitBinaryArithExpr(const BinaryExprPtr& op, const std::string& int_op,
                                      const std::string& float_op) {
  std::string result_type = "index";
  std::string mlir_op = int_op;
  if (auto scalar_type = As<ScalarType>(op->GetType())) {
    if (scalar_type->dtype_.IsFloat()) {
      result_type = GetTypeString(scalar_type->dtype_);
      mlir_op = float_op;
    } else if (scalar_type->dtype_ != ::pypto::DataType::INDEX) {
      result_type = GetTypeString(scalar_type->dtype_);
    }
  }

  if (mlir_op == float_op) {
    VisitExpr(op->left_);
    std::string lhs = fs_.current_expr_value;
    VisitExpr(op->right_);
    std::string rhs = fs_.current_expr_value;
    fs_.current_expr_value = EmitArithBinaryOp(mlir_op, lhs, rhs, result_type);
    return;
  }

  std::string lhs = EmitArithOperand(op->left_, result_type);
  std::string rhs = EmitArithOperand(op->right_, result_type);
  fs_.current_expr_value = EmitArithBinaryOp(mlir_op, lhs, rhs, result_type);
}

void PTOCodegen::VisitCmpExpr(const BinaryExprPtr& op, const std::string& predicate) {
  std::string operand_type = "index";
  bool is_float = false;
  if (auto scalar_type = As<ScalarType>(op->left_->GetType())) {
    if (scalar_type->dtype_.IsFloat()) {
      operand_type = GetTypeString(scalar_type->dtype_);
      is_float = true;
    } else if (scalar_type->dtype_ != ::pypto::DataType::INDEX) {
      operand_type = GetTypeString(scalar_type->dtype_);
    }
  }

  if (is_float) {
    VisitExpr(op->left_);
    std::string lhs = fs_.current_expr_value;
    VisitExpr(op->right_);
    std::string rhs = fs_.current_expr_value;
    static const std::map<std::string, std::string> pred_map = {
        {"eq", "oeq"}, {"ne", "one"}, {"slt", "olt"}, {"sle", "ole"}, {"sgt", "ogt"}, {"sge", "oge"}};
    auto it = pred_map.find(predicate);
    INTERNAL_CHECK_SPAN(it != pred_map.end(), op->span_) << "Unsupported float predicate for " << predicate;
    std::string float_pred = it->second;

    std::string result = NewTemp();
    Emit(result + " = arith.cmpf " + float_pred + ", " + lhs + ", " + rhs + " : " + operand_type);
    fs_.current_expr_value = result;
  } else {
    std::string lhs = EmitArithOperand(op->left_, operand_type);
    std::string rhs = EmitArithOperand(op->right_, operand_type);
    fs_.current_expr_value = EmitArithCmpi(predicate, lhs, rhs, operand_type);
  }
}

// ========================================================================
// Expression visitors - Leaf nodes
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::VarPtr& op) { fs_.current_expr_value = GetVarName(op); }

void PTOCodegen::VisitExpr_(const ir::IterArgPtr& op) {
  fs_.current_expr_value = GetVarName(std::dynamic_pointer_cast<const ir::Var>(op));
}

void PTOCodegen::VisitExpr_(const ir::ConstIntPtr& op) {
  fs_.current_expr_value = GetOrEmitConstant(op->value_, op->dtype());
}

void PTOCodegen::VisitExpr_(const ir::ConstFloatPtr& op) {
  fs_.current_expr_value = GetOrEmitConstant(op->value_, op->dtype());
}

void PTOCodegen::VisitExpr_(const ir::ConstBoolPtr& op) {
  std::string result = NewTemp();
  std::string val = op->value_ ? "1" : "0";
  Emit(result + " = arith.constant " + val + " : i1");
  fs_.current_expr_value = result;
}

// ========================================================================
// Expression visitors - Binary arithmetic
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::AddPtr& op) { VisitBinaryArithExpr(op, "arith.addi", "arith.addf"); }
void PTOCodegen::VisitExpr_(const ir::SubPtr& op) { VisitBinaryArithExpr(op, "arith.subi", "arith.subf"); }
void PTOCodegen::VisitExpr_(const ir::MulPtr& op) { VisitBinaryArithExpr(op, "arith.muli", "arith.mulf"); }
void PTOCodegen::VisitExpr_(const ir::FloorDivPtr& op) {
  VisitBinaryArithExpr(op, "arith.divsi", "arith.divf");
}
void PTOCodegen::VisitExpr_(const ir::FloorModPtr& op) {
  VisitBinaryArithExpr(op, "arith.remsi", "arith.remf");
}

// ========================================================================
// Expression visitors - Comparisons
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::EqPtr& op) { VisitCmpExpr(op, "eq"); }
void PTOCodegen::VisitExpr_(const ir::NePtr& op) { VisitCmpExpr(op, "ne"); }
void PTOCodegen::VisitExpr_(const ir::LtPtr& op) { VisitCmpExpr(op, "slt"); }
void PTOCodegen::VisitExpr_(const ir::LePtr& op) { VisitCmpExpr(op, "sle"); }
void PTOCodegen::VisitExpr_(const ir::GtPtr& op) { VisitCmpExpr(op, "sgt"); }
void PTOCodegen::VisitExpr_(const ir::GePtr& op) { VisitCmpExpr(op, "sge"); }

void PTOCodegen::VisitExpr_(const ir::CastPtr& op) {
  VisitExpr(op->operand_);
  std::string src = fs_.current_expr_value;

  ::pypto::DataType src_dtype = ir::GetScalarDtype(op->operand_);
  ::pypto::DataType dst_dtype = ir::GetScalarDtype(op);
  std::string src_type = GetTypeString(src_dtype);
  std::string dst_type = GetTypeString(dst_dtype);

  std::string result = NewTemp();
  bool src_is_index = (src_dtype == ::pypto::DataType::INDEX);
  bool dst_is_index = (dst_dtype == ::pypto::DataType::INDEX);
  bool src_is_float = src_dtype.IsFloat();
  bool dst_is_float = dst_dtype.IsFloat();

  bool src_is_uint = src_dtype.IsUnsignedInt();
  bool dst_is_uint = dst_dtype.IsUnsignedInt();

  std::string mlir_op;
  if (src_dtype == dst_dtype) {
    fs_.current_expr_value = src;
    return;
  } else if (src_is_index || dst_is_index) {
    CHECK(!src_is_float && !dst_is_float) << "Cast between float and index types is not supported";
    // arith.index_cast requires signless integer types. For unsigned source/destination,
    // bridge via builtin.unrealized_conversion_cast (mirrors GetOrEmitConstant in pto_codegen.cpp).
    if (dst_is_uint) {
      // index -> uiN : arith.index_cast index -> iN, then bridge iN -> uiN
      std::string signless_dst = dst_type.substr(1);  // "ui32" -> "i32"
      std::string signless_tmp = NewTemp();
      Emit(signless_tmp + " = arith.index_cast " + src + " : " + src_type + " to " + signless_dst);
      Emit(result + " = builtin.unrealized_conversion_cast " + signless_tmp + " : " + signless_dst + " to " +
           dst_type);
      fs_.current_expr_value = result;
      return;
    }
    if (src_is_uint) {
      // uiN -> index : bridge uiN -> iN, then arith.index_cast iN -> index
      std::string signless_src = src_type.substr(1);  // "ui32" -> "i32"
      std::string signless_tmp = NewTemp();
      Emit(signless_tmp + " = builtin.unrealized_conversion_cast " + src + " : " + src_type + " to " +
           signless_src);
      Emit(result + " = arith.index_cast " + signless_tmp + " : " + signless_src + " to " + dst_type);
      fs_.current_expr_value = result;
      return;
    }
    mlir_op = "arith.index_cast";
  } else if (src_is_float && dst_is_float) {
    mlir_op = (dst_dtype.GetBit() > src_dtype.GetBit()) ? "arith.extf" : "arith.truncf";
  } else if (!src_is_float && !dst_is_float) {
    if (dst_dtype.GetBit() > src_dtype.GetBit()) {
      mlir_op = src_is_uint ? "arith.extui" : "arith.extsi";
    } else {
      mlir_op = "arith.trunci";
    }
  } else if (!src_is_float && dst_is_float) {
    mlir_op = src_is_uint ? "arith.uitofp" : "arith.sitofp";
  } else {
    mlir_op = dst_is_uint ? "arith.fptoui" : "arith.fptosi";
  }

  Emit(result + " = " + mlir_op + " " + src + " : " + src_type + " to " + dst_type);
  fs_.current_expr_value = result;
}

std::string PTOCodegen::EmitCastToIndex(const ir::VarPtr& var, const std::string& mlir_name) {
  if (auto scalar_type = As<ScalarType>(var->GetType())) {
    CHECK(!scalar_type->dtype_.IsFloat())
        << "EmitCastToIndex does not support floating-point types (got " << GetTypeString(scalar_type->dtype_)
        << " for var '" << var->name_hint_ << "')";
    if (scalar_type->dtype_ != DataType::INDEX) {
      std::string idx_name = NewNamedTemp(var->name_hint_ + "_idx");
      std::string src_type = GetTypeString(scalar_type->dtype_);
      Emit(idx_name + " = arith.index_cast " + mlir_name + " : " + src_type + " to index");
      return idx_name;
    }
  }
  return mlir_name;
}

std::string PTOCodegen::EmitCastToI32(const ir::ExprPtr& expr, const std::string& mlir_name) {
  if (auto scalar_type = As<ScalarType>(expr->GetType())) {
    CHECK(!scalar_type->dtype_.IsFloat()) << "EmitCastToI32 does not support floating-point types (got "
                                          << GetTypeString(scalar_type->dtype_) << ")";
    if (scalar_type->dtype_ != DataType::INT32) {
      std::string i32_name = NewTemp();
      std::string src_type = GetTypeString(scalar_type->dtype_);
      // Use arith.index_cast for index→i32, arith.trunci/extui for int→int
      std::string mlir_op;
      if (scalar_type->dtype_ == DataType::INDEX) {
        mlir_op = "arith.index_cast";
      } else if (scalar_type->dtype_.GetBit() > 32) {
        mlir_op = "arith.trunci";
      } else {
        mlir_op = scalar_type->dtype_.IsUnsignedInt() ? "arith.extui" : "arith.extsi";
      }
      Emit(i32_name + " = " + mlir_op + " " + mlir_name + " : " + src_type + " to i32");
      return i32_name;
    }
  }
  return mlir_name;
}

bool PTOCodegen::HasFillpadConsumer(const ir::Var* var) const {
  return fs_.fillpad_input_vars.count(var) > 0;
}

// ========================================================================
// Expression visitors - Logical & Bitwise
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::AndPtr& op) { VisitBinaryArithExpr(op, "arith.andi", "arith.andi"); }
void PTOCodegen::VisitExpr_(const ir::OrPtr& op) { VisitBinaryArithExpr(op, "arith.ori", "arith.ori"); }
void PTOCodegen::VisitExpr_(const ir::XorPtr& op) { VisitBinaryArithExpr(op, "arith.xori", "arith.xori"); }
void PTOCodegen::VisitExpr_(const ir::BitAndPtr& op) { VisitBinaryArithExpr(op, "arith.andi", "arith.andi"); }
void PTOCodegen::VisitExpr_(const ir::BitOrPtr& op) { VisitBinaryArithExpr(op, "arith.ori", "arith.ori"); }
void PTOCodegen::VisitExpr_(const ir::BitXorPtr& op) { VisitBinaryArithExpr(op, "arith.xori", "arith.xori"); }
void PTOCodegen::VisitExpr_(const ir::BitShiftLeftPtr& op) {
  VisitBinaryArithExpr(op, "arith.shli", "arith.shli");
}
void PTOCodegen::VisitExpr_(const ir::BitShiftRightPtr& op) {
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string int_op = dtype.IsUnsignedInt() ? "arith.shrui" : "arith.shrsi";
  VisitBinaryArithExpr(op, int_op, int_op);
}

// ========================================================================
// Expression visitors - Other binary
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::FloatDivPtr& op) {
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string int_op = dtype.IsUnsignedInt() ? "arith.divui" : "arith.divsi";
  VisitBinaryArithExpr(op, int_op, "arith.divf");
}
void PTOCodegen::VisitExpr_(const ir::MinPtr& op) {
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string int_op = dtype.IsUnsignedInt() ? "arith.minui" : "arith.minsi";
  VisitBinaryArithExpr(op, int_op, "arith.minimumf");
}
void PTOCodegen::VisitExpr_(const ir::MaxPtr& op) {
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string int_op = dtype.IsUnsignedInt() ? "arith.maxui" : "arith.maxsi";
  VisitBinaryArithExpr(op, int_op, "arith.maximumf");
}

// ========================================================================
// Expression visitors - Unary
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::NotPtr& op) {
  VisitExpr(op->operand_);
  std::string src = fs_.current_expr_value;
  ::pypto::DataType src_dtype = ir::GetScalarDtype(op->operand_);
  std::string src_type = GetTypeString(src_dtype);
  std::string zero = NewTemp();
  std::string result = NewTemp();
  if (src_dtype.IsFloat()) {
    Emit(zero + " = arith.constant 0.0 : " + src_type);
    Emit(result + " = arith.cmpf oeq, " + src + ", " + zero + " : " + src_type);
  } else {
    Emit(zero + " = arith.constant 0 : " + src_type);
    Emit(result + " = arith.cmpi eq, " + src + ", " + zero + " : " + src_type);
  }
  fs_.current_expr_value = result;
}

void PTOCodegen::VisitExpr_(const ir::NegPtr& op) {
  VisitExpr(op->operand_);
  std::string src = fs_.current_expr_value;
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string type_str = GetTypeString(dtype);
  std::string result = NewTemp();
  if (dtype.IsFloat()) {
    Emit(result + " = arith.negf " + src + " : " + type_str);
  } else {
    std::string zero = NewTemp();
    Emit(zero + " = arith.constant 0 : " + type_str);
    Emit(result + " = arith.subi " + zero + ", " + src + " : " + type_str);
  }
  fs_.current_expr_value = result;
}

void PTOCodegen::VisitExpr_(const ir::AbsPtr& op) {
  VisitExpr(op->operand_);
  std::string src = fs_.current_expr_value;
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string type_str = GetTypeString(dtype);
  std::string result = NewTemp();
  if (dtype.IsFloat()) {
    Emit(result + " = math.absf " + src + " : " + type_str);
  } else {
    Emit(result + " = math.absi " + src + " : " + type_str);
  }
  fs_.current_expr_value = result;
}

void PTOCodegen::VisitExpr_(const ir::BitNotPtr& op) {
  VisitExpr(op->operand_);
  std::string src = fs_.current_expr_value;
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string type_str = GetTypeString(dtype);
  std::string all_ones = NewTemp();
  Emit(all_ones + " = arith.constant -1 : " + type_str);
  std::string result = NewTemp();
  Emit(result + " = arith.xori " + src + ", " + all_ones + " : " + type_str);
  fs_.current_expr_value = result;
}

}  // namespace codegen
}  // namespace pypto

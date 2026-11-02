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
#include <optional>
#include <string>

#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/const_fold.h"
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
namespace {
/// A literal's value when it is positive *at the width it is emitted at*.
///
/// The stored value is not the one the hardware sees: `EmitArithOperand`
/// narrows an operand into the operation's dtype, so an INDEX literal
/// 4294967293 emitted at INT32 is -3. Judging the sign before that conversion
/// would skip the floor correction for a divisor that is actually negative --
/// precisely when it is needed. Requiring the literal to fit `dtype` makes the
/// stored sign the emitted sign.
///
/// Only a positive divisor lets the truncating and flooring operations agree on
/// a non-negative dividend: for a negative one they differ in exactly the cases
/// the correction below exists to handle.
std::optional<int64_t> PositiveConstLiteral(const ir::ExprPtr& expr, DataType dtype) {
  auto literal = As<ir::ConstInt>(expr);
  if (!literal || literal->value_ <= 0) return std::nullopt;
  if (!ir::arith::detail::ConstIntFitsDtype(literal->value_, dtype)) return std::nullopt;
  return literal->value_;
}
}  // namespace

bool PTOCodegen::IsProvablyNonNegative(const ir::ExprPtr& expr, DataType dtype) const {
  if (!expr) return false;
  if (auto literal = As<ir::ConstInt>(expr)) {
    // Same narrowing caveat as above: a literal outside `dtype` is re-read at
    // the emitted width and can come back negative.
    return literal->value_ >= 0 && ir::arith::detail::ConstIntFitsDtype(literal->value_, dtype);
  }

  // An induction variable is at least its own loop's start -- but only once the
  // variable reaches the operation *unnarrowed*. `EmitArithOperand` coerces an
  // operand into the operation's dtype, and a truncated value has no relation
  // to the original's sign: over an INDEX loop from 2147483648, `i` emitted at
  // INT32 is -2147483648, so a recorded lower bound of 2147483648 proves
  // nothing about what the hardware sees. Requiring the widths to match is
  // stricter than checking the bound alone, because only the loop's *start* is
  // recorded -- a variable whose start fits the target can still run past it.
  if (auto var = AsVarLike(expr)) {
    if (ir::GetScalarDtype(var) != dtype) return false;
    auto it = fs_.loop_var_lower_bound.find(var.get());
    return it != fs_.loop_var_lower_bound.end() && it->second >= 0;
  }

  // Everything below is an operation node, and `VisitBinaryArithExpr` emits it
  // at the node's *own* dtype -- not the dtype of whatever encloses it. Proving
  // a nested operation at the enclosing width would ignore the narrowing and
  // the widening that bracket it: in an INDEX loop from 2147483648, an INT32
  // `FloorDiv(i, 2)` nested inside an INDEX `FloorMod(..., 3)` emits `i` at
  // INT32 as -2147483648, divides to -1073741824, then widens that negative
  // quotient back to INDEX. Checked at INDEX throughout, the induction variable
  // looks non-negative and the outer shortcut emits a bare `remsi`, which
  // returns -1 where FloorMod must return 2.
  //
  // Requiring each nested operation to execute at the dtype being proved makes
  // every conversion on the path a no-op, so the stored sign is the emitted
  // sign at every level -- the same rule the induction variable above is held
  // to, applied to the operations between it and the division.
  if (ir::GetScalarDtype(expr) != dtype) return false;

  // `iv - start`: the offset of this trip from the first. The subtrahend must
  // itself be non-negative and no greater than the start, which pins the
  // difference inside [0, iv] so it cannot overflow. Subtracting a negative
  // literal is an addition in disguise and is not admitted. The variable is
  // subject to the same no-narrowing rule as above.
  if (auto sub = As<ir::Sub>(expr)) {
    auto var = AsVarLike(sub->left_);
    if (!var || ir::GetScalarDtype(var) != dtype) return false;
    auto subtrahend = As<ir::ConstInt>(sub->right_);
    if (!subtrahend || subtrahend->value_ < 0) return false;
    if (!ir::arith::detail::ConstIntFitsDtype(subtrahend->value_, dtype)) return false;
    auto it = fs_.loop_var_lower_bound.find(var.get());
    return it != fs_.loop_var_lower_bound.end() && it->second >= subtrahend->value_;
  }

  // A quotient or remainder by a positive literal cannot exceed its dividend in
  // magnitude, so proving the dividend proves these too, with no room to
  // overflow. Sums and products are deliberately absent -- see the header.
  if (auto div = As<ir::FloorDiv>(expr)) {
    return PositiveConstLiteral(div->right_, dtype).has_value() && IsProvablyNonNegative(div->left_, dtype);
  }
  if (auto mod = As<ir::FloorMod>(expr)) {
    return PositiveConstLiteral(mod->right_, dtype).has_value() && IsProvablyNonNegative(mod->left_, dtype);
  }
  return false;
}

void PTOCodegen::VisitExpr_(const ir::FloorDivPtr& op) {
  // Match FloorMod's floor semantics so q * divisor + remainder reconstructs
  // the dividend even when either operand is negative. PTOAS expands the native
  // floor operation into truncating division plus a conditional correction.
  const auto dtype = ir::GetScalarDtype(op);
  const bool signed_integer = dtype == DataType::INDEX || dtype.IsSignedInt();
  // Flooring and truncating division agree whenever the quotient cannot be
  // negative, so skip that expansion and let PTOAS see a plain division. Not
  // only cheaper: an index expression that stays a division over a subtraction
  // is a shape the assembler's addressing matcher can still peel, which a
  // conditional correction is not.
  const bool truncation_is_exact = signed_integer && PositiveConstLiteral(op->right_, dtype).has_value() &&
                                   IsProvablyNonNegative(op->left_, dtype);
  VisitBinaryArithExpr(op, (signed_integer && !truncation_is_exact) ? "arith.floordivsi" : "arith.divsi",
                       "arith.divf");
}
void PTOCodegen::VisitExpr_(const ir::FloorModPtr& op) {
  // CanonicalSlotCollector proves these INDEX operands nonnegative. Preserve
  // the unsigned remainder PTOAS uses for slot synchronization, while ordinary
  // FloorMod expressions keep the signed floor correction below.
  if (fs_.unsigned_slot_remainders.count(op.get())) {
    VisitBinaryArithExpr(op, "arith.remui", "arith.remf");
    return;
  }
  const auto dtype = ir::GetScalarDtype(op);
  if (dtype != DataType::INDEX && !dtype.IsSignedInt()) {
    VisitBinaryArithExpr(op, "arith.remsi", "arith.remf");
    return;
  }

  // The correction below exists because remsi follows the dividend's sign while
  // FloorMod follows the divisor's. With a non-negative dividend and a positive
  // divisor the remainder is already non-negative, the two agree, and the
  // correction is dead code -- emit the bare remainder instead. Besides saving
  // three ops, it keeps `(symbol + constant) mod N` recognisable to PTOAS' slot
  // matcher, which a rem/cmp/add/select chain is not.
  if (PositiveConstLiteral(op->right_, dtype).has_value() && IsProvablyNonNegative(op->left_, dtype)) {
    VisitBinaryArithExpr(op, "arith.remsi", "arith.remf");
    return;
  }

  const std::string type = GetTypeString(dtype);
  const std::string lhs = EmitArithOperand(op->left_, type);
  const std::string rhs = EmitArithOperand(op->right_, type);
  const std::string remainder = EmitArithBinaryOp("arith.remsi", lhs, rhs, type);
  const std::string zero = GetOrEmitConstant(int64_t{0}, dtype);

  // FloorMod follows the divisor's sign, whereas remsi follows the dividend's.
  // Correct only nonzero remainders with the opposite sign. A constant divisor
  // needs just one comparison; for example, -97 % 24 must be 23, not -1.
  const auto divisor = As<ir::ConstInt>(op->right_);
  int64_t divisor_value = divisor ? divisor->value_ : 0;
  // Interpret signless MLIR literals at their emitted signed width, including
  // EmitArithOperand's i32 narrowing (for example, 253 : i8 represents -3).
  if (dtype == DataType::INT8) {
    divisor_value = static_cast<uint8_t>(divisor_value);
    if (divisor_value >= 128) divisor_value -= 256;
  } else if (dtype == DataType::INT16) {
    divisor_value = static_cast<int16_t>(divisor_value);
  } else if (dtype == DataType::INT32) {
    divisor_value = static_cast<int32_t>(divisor_value);
  }
  std::string needs_correction;
  if (divisor && divisor_value != 0) {
    needs_correction = EmitArithCmpi(divisor_value > 0 ? "slt" : "sgt", remainder, zero, type);
  } else {
    const std::string remainder_negative = EmitArithCmpi("slt", remainder, zero, type);
    const std::string divisor_negative = EmitArithCmpi("slt", rhs, zero, type);
    const std::string signs_differ =
        EmitArithBinaryOp("arith.xori", remainder_negative, divisor_negative, "i1");
    const std::string nonzero = EmitArithCmpi("ne", remainder, zero, type);
    needs_correction = EmitArithBinaryOp("arith.andi", nonzero, signs_differ, "i1");
  }
  const std::string corrected = EmitArithBinaryOp("arith.addi", remainder, rhs, type);
  const std::string result = NewTemp();
  Emit(result + " = arith.select " + needs_correction + ", " + corrected + ", " + remainder + " : " + type);
  fs_.current_expr_value = result;
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
  } else if (src_dtype == DataType::BOOL && (dst_dtype.IsInt() || dst_is_index || dst_is_float)) {
    // A logical BOOL is 0/1, whereas interpreting its i1 representation as
    // signed would make true equal -1. Normalize before numeric conversion.
    if (dst_is_index) {
      mlir_op = "arith.index_castui";
    } else if (dst_is_float) {
      // PTOAS can lower a direct i1-to-float cast to (float)bool, which A3
      // rejects. An i32 carrier preserves 0/1 and avoids that device cast.
      src = EmitCastToI32(op->operand_, src);
      src_type = "i32";
      mlir_op = "arith.sitofp";
    } else if (dst_is_uint) {
      // Arithmetic extensions require signless result types. Bridge to uiN
      // only after zero-extending the logical value to the matching iN.
      const std::string signless_dst = dst_type.substr(1);
      const std::string extended = NewTemp();
      Emit(extended + " = arith.extui " + src + " : i1 to " + signless_dst);
      Emit(result + " = builtin.unrealized_conversion_cast " + extended + " : " + signless_dst + " to " +
           dst_type);
      fs_.current_expr_value = result;
      return;
    } else {
      mlir_op = "arith.extui";
    }
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
    if (dst_dtype.GetBit() == src_dtype.GetBit()) {
      // MLIR arithmetic casts only change integer width and require signless
      // operands. Preserve the bits when crossing between uiN and iN.
      mlir_op = "builtin.unrealized_conversion_cast";
    } else if (dst_dtype.GetBit() > src_dtype.GetBit()) {
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

std::string PTOCodegen::EmitCastToIndex(const ir::ExprPtr& expr, const std::string& mlir_name) {
  if (auto var = As<ir::Var>(expr)) {
    return EmitCastToIndex(var, mlir_name);
  }
  if (auto scalar_type = As<ScalarType>(expr->GetType())) {
    CHECK(!scalar_type->dtype_.IsFloat()) << "EmitCastToIndex does not support floating-point types (got "
                                          << GetTypeString(scalar_type->dtype_) << ")";
    if (scalar_type->dtype_ != DataType::INDEX) {
      std::string idx_name = NewTemp();
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
      const bool is_unsigned = scalar_type->dtype_.IsUnsignedInt();
      std::string source_name = mlir_name;
      std::string src_type = GetTypeString(scalar_type->dtype_);
      if (is_unsigned) {
        std::string signless_name = NewTemp();
        std::string signless_type = src_type.substr(1);  // "uiN" -> "iN"
        Emit(signless_name + " = builtin.unrealized_conversion_cast " + source_name + " : " + src_type +
             " to " + signless_type);
        if (scalar_type->dtype_.GetBit() == 32) {
          return signless_name;
        }
        source_name = signless_name;
        src_type = signless_type;
      }

      std::string i32_name = NewTemp();
      // Use arith.index_cast for index→i32, arith.trunci/extsi/extui for int→int.
      // MLIR arithmetic casts require signless integer operands, so unsigned
      // inputs are bridged to iN above before any width conversion.
      std::string mlir_op;
      if (scalar_type->dtype_ == DataType::INDEX) {
        mlir_op = "arith.index_cast";
      } else if (scalar_type->dtype_.GetBit() > 32) {
        mlir_op = "arith.trunci";
      } else {
        // BOOL is not an unsigned integer dtype, but still represents 0/1.
        mlir_op = (is_unsigned || scalar_type->dtype_ == DataType::BOOL) ? "arith.extui" : "arith.extsi";
      }
      Emit(i32_name + " = " + mlir_op + " " + source_name + " : " + src_type + " to i32");
      return i32_name;
    }
  }
  return mlir_name;
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

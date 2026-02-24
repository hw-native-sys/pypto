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

#include "pypto/codegen/codegen_base.h"

#include <string>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

std::string CodegenBase::TryGetVarName(const ir::ExprPtr& expr) const {
  if (auto var = As<Var>(expr)) {
    return var->name_;
  }
  if (auto iter_arg = As<IterArg>(expr)) {
    return iter_arg->name_;
  }
  return "";
}

std::string CodegenBase::GenerateExprString(const ir::ExprPtr& expr) const {
  std::string var_name = TryGetVarName(expr);
  if (!var_name.empty()) {
    return var_name;
  }
  if (auto const_int = As<ConstInt>(expr)) {
    return std::to_string(const_int->value_);
  }
  if (auto add = As<Add>(expr)) {
    return "(" + GenerateExprString(add->left_) + " + " + GenerateExprString(add->right_) + ")";
  }
  if (auto sub = As<Sub>(expr)) {
    return "(" + GenerateExprString(sub->left_) + " - " + GenerateExprString(sub->right_) + ")";
  }
  if (auto mul = As<Mul>(expr)) {
    return "(" + GenerateExprString(mul->left_) + " * " + GenerateExprString(mul->right_) + ")";
  }
  if (auto floor_div = As<FloorDiv>(expr)) {
    return "(" + GenerateExprString(floor_div->left_) + " / " + GenerateExprString(floor_div->right_) + ")";
  }
  if (auto floor_mod = As<FloorMod>(expr)) {
    return "(" + GenerateExprString(floor_mod->left_) + " % " + GenerateExprString(floor_mod->right_) + ")";
  }
  if (auto min_expr = As<Min>(expr)) {
    return "std::min(" + GenerateExprString(min_expr->left_) + ", " + GenerateExprString(min_expr->right_) +
           ")";
  }
  if (auto max_expr = As<Max>(expr)) {
    return "std::max(" + GenerateExprString(max_expr->left_) + ", " + GenerateExprString(max_expr->right_) +
           ")";
  }
  if (auto eq = As<Eq>(expr)) {
    return "(" + GenerateExprString(eq->left_) + " == " + GenerateExprString(eq->right_) + ")";
  }
  if (auto ne = As<Ne>(expr)) {
    return "(" + GenerateExprString(ne->left_) + " != " + GenerateExprString(ne->right_) + ")";
  }
  if (auto lt = As<Lt>(expr)) {
    return "(" + GenerateExprString(lt->left_) + " < " + GenerateExprString(lt->right_) + ")";
  }
  if (auto le = As<Le>(expr)) {
    return "(" + GenerateExprString(le->left_) + " <= " + GenerateExprString(le->right_) + ")";
  }
  if (auto gt = As<Gt>(expr)) {
    return "(" + GenerateExprString(gt->left_) + " > " + GenerateExprString(gt->right_) + ")";
  }
  if (auto ge = As<Ge>(expr)) {
    return "(" + GenerateExprString(ge->left_) + " >= " + GenerateExprString(ge->right_) + ")";
  }
  if (auto const_float = As<ConstFloat>(expr)) {
    return std::to_string(const_float->value_);
  }
  if (auto const_bool = As<ConstBool>(expr)) {
    return const_bool->value_ ? "1" : "0";
  }
  if (auto neg = As<Neg>(expr)) {
    return "(-" + GenerateExprString(neg->operand_) + ")";
  }
  if (auto cast_expr = As<Cast>(expr)) {
    return GenerateExprString(cast_expr->operand_);
  }
  if (auto tuple_get = As<TupleGetItemExpr>(expr)) {
    return GenerateExprString(tuple_get->tuple_) + "_" + std::to_string(tuple_get->index_);
  }
  throw pypto::NotImplementedError("GenerateExprString not implemented for expression type: " +
                                   expr->TypeName());
}

std::string CodegenBase::GetRuntimeDataTypeString(const DataType& dtype) const {
  if (dtype == DataType::FP16) return "DataType::FLOAT16";
  if (dtype == DataType::FP32) return "DataType::FLOAT32";
  if (dtype == DataType::INT32) return "DataType::INT32";
  if (dtype == DataType::INT16) return "DataType::INT16";
  if (dtype == DataType::INT8) return "DataType::INT8";
  if (dtype == DataType::UINT8) return "DataType::UINT8";
  if (dtype == DataType::BF16) return "DataType::BFLOAT16";
  if (dtype == DataType::INT64) return "DataType::INT64";
  if (dtype == DataType::UINT64) return "DataType::UINT64";
  return "DataType::UNKNOWN";
}

}  // namespace codegen
}  // namespace pypto

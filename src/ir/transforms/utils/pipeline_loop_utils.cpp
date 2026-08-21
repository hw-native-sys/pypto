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

#include "pypto/ir/transforms/utils/pipeline_loop_utils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto {
namespace ir {
namespace pipeline_loop {

int64_t GetConstIntValue(const ExprPtr& expr, const char* pass, const std::string& what) {
  if (auto value = transform_utils::EvalConstInt(expr)) {
    return *value;
  }
  throw pypto::ValueError(std::string(pass) + ": " + what + " must be a compile-time integer constant, got " +
                          expr->TypeName());
}

ExprPtr MakeConstIndex(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

ExprPtr OffsetIndex(const ExprPtr& base, int64_t offset_val, const Span& span) {
  if (offset_val == 0) return base;
  if (auto ci = As<ConstInt>(base)) {
    return MakeConstIndex(ci->value_ + offset_val, span);
  }
  return MakeAdd(base, MakeConstIndex(offset_val, span), span);
}

VarPtr CloneLoopVar(const VarPtr& original) {
  return std::make_shared<Var>(original->name_hint_, original->GetType(), original->span_);
}

IterArgPtr MakeFreshIterArg(const IterArgPtr& original, const ExprPtr& init_value) {
  return std::make_shared<IterArg>(original->name_hint_, original->GetType(), init_value, original->span_);
}

VarPtr MakeFreshVar(const VarPtr& original, const std::string& suffix) {
  return std::make_shared<Var>(original->name_hint_ + suffix, original->GetType(), original->span_);
}

std::pair<StmtPtr, std::vector<ExprPtr>> SplitBodyYield(const StmtPtr& body) {
  if (auto yield = As<YieldStmt>(body)) {
    return {std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, body->span_), yield->value_};
  }
  auto seq = As<SeqStmts>(body);
  if (!seq || seq->stmts_.empty()) {
    return {body, {}};
  }
  auto yield = As<YieldStmt>(seq->stmts_.back());
  if (!yield) {
    return {body, {}};
  }
  std::vector<StmtPtr> without(seq->stmts_.begin(), seq->stmts_.end() - 1);
  return {std::make_shared<SeqStmts>(std::move(without), seq->span_), yield->value_};
}

std::vector<ExprPtr> ReturnVarsAsExprs(const std::vector<VarPtr>& vars) {
  std::vector<ExprPtr> result;
  result.reserve(vars.size());
  for (const auto& v : vars) result.push_back(v);
  return result;
}

std::vector<ExprPtr> InitValueExprs(const std::vector<IterArgPtr>& iter_args) {
  std::vector<ExprPtr> result;
  result.reserve(iter_args.size());
  for (const auto& ia : iter_args) result.push_back(ia->initValue_);
  return result;
}

std::vector<VarPtr> MakeFreshReturnVars(const std::vector<VarPtr>& originals, const std::string& suffix) {
  std::vector<VarPtr> result;
  result.reserve(originals.size());
  for (const auto& v : originals) result.push_back(MakeFreshVar(v, suffix));
  return result;
}

}  // namespace pipeline_loop
}  // namespace ir
}  // namespace pypto

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

#include "pypto/ir/transforms/composite_lowering_registry.h"

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// ============================================================================
// LoweringBuilder
// ============================================================================

LoweringBuilder::LoweringBuilder(std::string base_name, std::size_t& temp_counter)
    : base_name_(std::move(base_name)), temp_counter_(temp_counter) {}

std::string LoweringBuilder::MakeTempName(const std::string& qualifier) {
  return auto_name::BuildName(auto_name::GetBaseName(base_name_), qualifier, "tmp",
                              static_cast<int>(temp_counter_++));
}

ExprPtr LoweringBuilder::Bind(const std::string& qualifier, const ExprPtr& expr, const Span& span) {
  auto var = std::make_shared<Var>(MakeTempName(qualifier), expr->GetType(), span);
  stmts_.push_back(std::make_shared<AssignStmt>(var, expr, span));
  return var;
}

ExprPtr LoweringBuilder::Muls(const ExprPtr& x, float c, const Span& span) {
  auto tile_type = As<TileType>(x->GetType());
  INTERNAL_CHECK_SPAN(tile_type, span) << "tile.muls input must be TileType";
  auto scalar = std::make_shared<ConstFloat>(static_cast<double>(c), tile_type->dtype_, span);
  return OpRegistry::GetInstance().Create("tile.muls", {x, scalar}, {}, span);
}

ExprPtr LoweringBuilder::Adds(const ExprPtr& x, float c, const Span& span) {
  auto tile_type = As<TileType>(x->GetType());
  INTERNAL_CHECK_SPAN(tile_type, span) << "tile.adds input must be TileType";
  auto scalar = std::make_shared<ConstFloat>(static_cast<double>(c), tile_type->dtype_, span);
  return OpRegistry::GetInstance().Create("tile.adds", {x, scalar}, {}, span);
}

ExprPtr LoweringBuilder::Add(const ExprPtr& a, const ExprPtr& b, const Span& span) {
  return OpRegistry::GetInstance().Create("tile.add", {a, b}, {}, span);
}

ExprPtr LoweringBuilder::Sub(const ExprPtr& a, const ExprPtr& b, const Span& span) {
  return OpRegistry::GetInstance().Create("tile.sub", {a, b}, {}, span);
}

ExprPtr LoweringBuilder::Mul(const ExprPtr& a, const ExprPtr& b, const Span& span) {
  return OpRegistry::GetInstance().Create("tile.mul", {a, b}, {}, span);
}

ExprPtr LoweringBuilder::Cast(const ExprPtr& x, DataType to, int mode, const Span& span) {
  std::vector<std::pair<std::string, std::any>> kw = {{"target_type", to}, {"mode", mode}};
  return OpRegistry::GetInstance().Create("tile.cast", {x}, kw, span);
}

// ============================================================================
// CompositeLoweringRegistry
// ============================================================================

CompositeLoweringRegistry& CompositeLoweringRegistry::GetInstance() {
  static CompositeLoweringRegistry instance;
  return instance;
}

CompositeLoweringRegistry::CompositeLoweringRegistry() { RegisterSinCosLoweringRules(*this); }

void CompositeLoweringRegistry::Register(const std::string& op_name, CompositeLoweringFn fn) {
  rules_.emplace(op_name, std::move(fn));
}

const CompositeLoweringFn* CompositeLoweringRegistry::Lookup(const std::string& op_name) const {
  auto it = rules_.find(op_name);
  if (it == rules_.end()) return nullptr;
  return &it->second;
}

}  // namespace ir
}  // namespace pypto

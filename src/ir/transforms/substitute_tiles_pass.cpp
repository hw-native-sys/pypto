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

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

StmtPtr MakeSeqOrSingle(std::vector<StmtPtr> stmts, const Span& span) {
  if (stmts.empty()) {
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
  }
  if (stmts.size() == 1) {
    return stmts.front();
  }
  return std::make_shared<SeqStmts>(std::move(stmts), span);
}

std::vector<ExprPtr> ExtractConstShape(const ExprPtr& shape_expr) {
  auto make_tuple = As<MakeTuple>(shape_expr);
  CHECK(make_tuple) << "SubstituteTiles: tile.expand_clone shape must be a literal tuple";

  std::vector<ExprPtr> shape;
  shape.reserve(make_tuple->elements_.size());
  for (size_t i = 0; i < make_tuple->elements_.size(); ++i) {
    auto const_dim = As<ConstInt>(make_tuple->elements_[i]);
    CHECK(const_dim) << "SubstituteTiles: tile.expand_clone shape element " << i
                     << " must be ConstInt for tile.create";
    shape.push_back(make_tuple->elements_[i]);
  }
  CHECK(!shape.empty()) << "SubstituteTiles: tile.expand_clone shape must be non-empty";
  return shape;
}

ExprPtr MakeOffsetTuple(const std::vector<ExprPtr>& target_shape, int broadcast_axis, const ExprPtr& loop_var,
                        const Span& span) {
  std::vector<ExprPtr> offsets;
  offsets.reserve(target_shape.size());
  for (size_t i = 0; i < target_shape.size(); ++i) {
    if (static_cast<int>(i) == broadcast_axis) {
      offsets.push_back(loop_var);
    } else {
      offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, span));
    }
  }
  return std::make_shared<MakeTuple>(offsets, span);
}

class SubstituteTilesMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call || call->op_->name_ != "tile.expand_clone") {
      return IRMutator::VisitStmt_(op);
    }
    return RewriteExpandClone(call, op->var_, op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (!call || call->op_->name_ != "tile.expand_clone") {
      return IRMutator::VisitStmt_(op);
    }

    auto result_var =
        std::make_shared<Var>(NextTempName("expand_clone", {"unused"}), call->GetType(), op->span_);
    return RewriteExpandClone(call, result_var, op->span_);
  }

 private:
  int temp_var_id_ = 0;

  StmtPtr RewriteExpandClone(const CallPtr& call, const VarPtr& result_var, const Span& span) {
    CHECK(call->args_.size() == 2) << "SubstituteTiles: tile.expand_clone expects 2 arguments";

    auto tile_src = VisitExpr(call->args_[0]);
    auto shape_expr = call->args_[1];

    auto tile_type = As<TileType>(tile_src->GetType());
    CHECK(tile_type) << "SubstituteTiles: tile.expand_clone requires TileType input";

    auto target_shape = ExtractConstShape(shape_expr);
    const auto& input_shape = tile_type->shape_;
    CHECK(input_shape.size() == target_shape.size())
        << "SubstituteTiles: tile.expand_clone rank mismatch input " << FormatShape(input_shape)
        << " vs target " << FormatShape(target_shape);

    int broadcast_axis = -1;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (DimensionsEqual(input_shape[i], target_shape[i])) {
        continue;
      }

      auto input_const = GetConstantDimension(input_shape[i]);
      CHECK(input_const && *input_const == 1)
          << "SubstituteTiles: tile.expand_clone only allows broadcast from dimension 1, got "
          << input_shape[i] << " at axis " << i;

      CHECK(broadcast_axis == -1) << "SubstituteTiles: tile.expand_clone allows at most one broadcast axis";
      broadcast_axis = static_cast<int>(i);
    }

    std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", tile_type->dtype_}};
    if (tile_type->memory_space_.has_value()) {
      create_kwargs.emplace_back("target_memory", std::any(*tile_type->memory_space_));
    }
    auto create_call = OpRegistry::GetInstance().Create("tile.create", {shape_expr}, create_kwargs, span);
    auto create_var = std::make_shared<Var>(NextTempName(result_var->name_hint_, {"expand", "init"}),
                                            create_call->GetType(), span);

    std::vector<StmtPtr> rewritten;
    rewritten.push_back(std::make_shared<AssignStmt>(create_var, create_call, span));

    if (broadcast_axis < 0) {
      auto offsets = MakeOffsetTuple(target_shape, -1, ExprPtr{}, span);
      auto assemble_call =
          OpRegistry::GetInstance().Create("tile.assemble", {create_var, tile_src, offsets}, {}, span);
      rewritten.push_back(std::make_shared<AssignStmt>(result_var, assemble_call, span));
      return MakeSeqOrSingle(std::move(rewritten), span);
    }

    auto loop_var = std::make_shared<Var>(NextTempName(result_var->name_hint_, {"expand"}, "idx"),
                                          std::make_shared<ScalarType>(DataType::INDEX), span);
    auto iter_arg = std::make_shared<IterArg>(NextTempName(result_var->name_hint_, {"expand"}, "iter"),
                                              create_var->GetType(), create_var, span);

    auto offsets = MakeOffsetTuple(target_shape, broadcast_axis, loop_var, span);
    auto assemble_call =
        OpRegistry::GetInstance().Create("tile.assemble", {iter_arg, tile_src, offsets}, {}, span);
    auto assembled_var = std::make_shared<Var>(NextTempName(result_var->name_hint_, {"expand", "assembled"}),
                                               assemble_call->GetType(), span);

    auto assign = std::make_shared<AssignStmt>(assembled_var, assemble_call, span);
    auto yield_stmt = std::make_shared<YieldStmt>(std::vector<ExprPtr>{assembled_var}, span);
    auto body = SeqStmts::Flatten(std::vector<StmtPtr>{assign, yield_stmt}, span);

    auto start = std::make_shared<ConstInt>(0, DataType::INDEX, span);
    auto step = std::make_shared<ConstInt>(1, DataType::INDEX, span);
    auto stop = target_shape[broadcast_axis];

    auto loop = std::make_shared<ForStmt>(loop_var, start, stop, step, std::vector<IterArgPtr>{iter_arg},
                                          body, std::vector<VarPtr>{result_var}, span);

    rewritten.push_back(loop);
    return MakeSeqOrSingle(std::move(rewritten), span);
  }

  std::string NextTempName(const std::string& base, const std::vector<std::string>& qualifiers,
                           const std::optional<std::string>& role = std::nullopt) {
    return auto_name::BuildName(auto_name::GetBaseName(base), qualifiers, role,
                                static_cast<int>(temp_var_id_++));
  }
};

FunctionPtr TransformFunction(const FunctionPtr& func) {
  if (!IsInCoreType(func->func_type_)) {
    return func;
  }

  SubstituteTilesMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) {
    return func;
  }

  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_, func->level_, func->role_,
                                    func->attrs_);
}

}  // namespace

namespace pass {

Pass SubstituteTiles() {
  return CreateFunctionPass(TransformFunction, "SubstituteTiles", kSubstituteTilesProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

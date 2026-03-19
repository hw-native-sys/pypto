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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
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

TileLayout GetTileLayout(const TileTypePtr& tile_type) {
  if (!tile_type || !tile_type->tile_view_.has_value()) {
    return TileLayout::row_major;
  }
  return tile_type->tile_view_->blayout;
}

bool IsConstOne(const ExprPtr& expr) {
  auto ci = As<ConstInt>(expr);
  return ci && ci->value_ == 1;
}

bool IsColumnVectorColMajor(const TileTypePtr& tile_type) {
  return tile_type && tile_type->shape_.size() == 2 && IsConstOne(tile_type->shape_[1]) &&
         GetTileLayout(tile_type) == TileLayout::col_major;
}

bool IsColumnVector(const TileTypePtr& tile_type) {
  return tile_type && tile_type->shape_.size() == 2 && IsConstOne(tile_type->shape_[1]);
}

bool IsRowVectorRowMajor(const TileTypePtr& tile_type) {
  return tile_type && tile_type->shape_.size() == 2 && IsConstOne(tile_type->shape_[0]) &&
         GetTileLayout(tile_type) == TileLayout::row_major;
}

ExprPtr MakeShapeTuple(const std::vector<ExprPtr>& dims, const Span& span) {
  return std::make_shared<MakeTuple>(dims, span);
}

CallPtr CreateReshapeCall(const ExprPtr& input, const std::vector<ExprPtr>& shape, const Span& span) {
  auto expr =
      OpRegistry::GetInstance().Create("tile.reshape", {input, MakeShapeTuple(shape, span)}, {}, span);
  auto call = As<Call>(expr);
  CHECK(call) << "ResolveBackendOpLayouts: tile.reshape must produce a Call";
  return call;
}

std::vector<ExprPtr> MakeRowVectorShape(const TileTypePtr& tile_type) {
  CHECK(IsColumnVector(tile_type)) << "ResolveBackendOpLayouts expects a [N,1] tile for vector repair";
  return {std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown()), tile_type->shape_[0]};
}

bool IsRepairableCall(const CallPtr& call, const backend::BackendTileLayoutSpec& spec) {
  bool has_repairable_input = false;
  for (size_t i = 0; i < spec.input_layouts.size() && i < call->args_.size(); ++i) {
    const auto& required_layout = spec.input_layouts[i];
    if (!required_layout.has_value() || required_layout.value() != TileLayout::row_major) {
      continue;
    }
    auto tile_type = As<TileType>(call->args_[i]->GetType());
    if (!tile_type) {
      return false;
    }
    if (GetTileLayout(tile_type) == TileLayout::row_major) {
      continue;
    }
    if (IsColumnVectorColMajor(tile_type)) {
      has_repairable_input = true;
      continue;
    }
    return false;
  }
  return has_repairable_input;
}

class BackendLayoutRepairMutator : public IRMutator {
 public:
  std::string NextTempName(const std::string& base, const std::vector<std::string>& qualifiers) {
    return auto_name::BuildName(auto_name::GetBaseName(base), qualifiers, "tmp",
                                static_cast<int>(temp_var_id_++));
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call) {
      return IRMutator::VisitStmt_(op);
    }

    auto* layout_spec = backend::GetBackend()->GetTileLayoutSpec(call->op_->name_);
    if (!layout_spec || !IsRepairableCall(call, *layout_spec)) {
      return IRMutator::VisitStmt_(op);
    }

    auto result_tile_type = As<TileType>(op->var_->GetType());
    CHECK(result_tile_type)
        << "ResolveBackendOpLayouts expects constrained op assignment targets to be TileType";

    std::vector<StmtPtr> rewritten;
    std::vector<ExprPtr> new_args = call->args_;

    for (size_t i = 0; i < layout_spec->input_layouts.size() && i < call->args_.size(); ++i) {
      const auto& required_layout = layout_spec->input_layouts[i];
      if (!required_layout.has_value() || required_layout.value() != TileLayout::row_major) {
        continue;
      }

      auto tile_type = As<TileType>(call->args_[i]->GetType());
      if (!IsColumnVector(tile_type) || IsRowVectorRowMajor(tile_type)) {
        continue;
      }

      auto reshape_call = CreateReshapeCall(call->args_[i], MakeRowVectorShape(tile_type), call->span_);
      auto reshape_var = std::make_shared<Var>(
          NextTempName(op->var_->name_hint_, {auto_name::RowMajorQualifier(), auto_name::ArgQualifier(i)}),
          reshape_call->GetType(), call->span_);
      rewritten.push_back(std::make_shared<AssignStmt>(reshape_var, reshape_call, op->span_));
      new_args[i] = reshape_var;
    }

    auto repaired_expr =
        OpRegistry::GetInstance().Create(call->op_->name_, new_args, call->kwargs_, call->span_);
    auto repaired_call = As<Call>(repaired_expr);
    CHECK(repaired_call) << "ResolveBackendOpLayouts: repaired consumer must remain a Call";

    if (!IsRowVectorRowMajor(result_tile_type)) {
      auto row_major_var = std::make_shared<Var>(NextTempName(op->var_->name_hint_, {"row_major"}),
                                                 repaired_call->GetType(), call->span_);
      rewritten.push_back(std::make_shared<AssignStmt>(row_major_var, repaired_call, op->span_));

      auto reshape_back = CreateReshapeCall(row_major_var, result_tile_type->shape_, call->span_);
      rewritten.push_back(std::make_shared<AssignStmt>(op->var_, reshape_back, op->span_));
      return MakeSeqOrSingle(std::move(rewritten), op->span_);
    }

    rewritten.push_back(std::make_shared<AssignStmt>(op->var_, repaired_call, op->span_));
    return MakeSeqOrSingle(std::move(rewritten), op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (!call) {
      return IRMutator::VisitStmt_(op);
    }

    auto* layout_spec = backend::GetBackend()->GetTileLayoutSpec(call->op_->name_);
    if (!layout_spec || !IsRepairableCall(call, *layout_spec)) {
      return IRMutator::VisitStmt_(op);
    }

    std::vector<StmtPtr> rewritten;
    std::vector<ExprPtr> new_args = call->args_;

    for (size_t i = 0; i < layout_spec->input_layouts.size() && i < call->args_.size(); ++i) {
      const auto& required_layout = layout_spec->input_layouts[i];
      if (!required_layout.has_value() || required_layout.value() != TileLayout::row_major) {
        continue;
      }
      auto tile_type = As<TileType>(call->args_[i]->GetType());
      if (!IsColumnVector(tile_type) || IsRowVectorRowMajor(tile_type)) {
        continue;
      }
      auto reshape_call = CreateReshapeCall(call->args_[i], MakeRowVectorShape(tile_type), call->span_);
      auto reshape_var =
          std::make_shared<Var>(NextTempName("layout_fix", {"row_major", "arg" + std::to_string(i)}),
                                reshape_call->GetType(), call->span_);
      rewritten.push_back(std::make_shared<AssignStmt>(reshape_var, reshape_call, op->span_));
      new_args[i] = reshape_var;
    }

    auto repaired_expr =
        OpRegistry::GetInstance().Create(call->op_->name_, new_args, call->kwargs_, call->span_);
    rewritten.push_back(std::make_shared<EvalStmt>(repaired_expr, op->span_));
    return MakeSeqOrSingle(std::move(rewritten), op->span_);
  }

 private:
  size_t temp_var_id_ = 0;
};

FunctionPtr RewriteFunction(const FunctionPtr& func) {
  if (!IsInCoreType(func->func_type_)) {
    return func;
  }
  if (!backend::BackendConfig::IsConfigured()) {
    return func;
  }

  BackendLayoutRepairMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) {
    return func;
  }

  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_, func->level_, func->role_);
}

}  // namespace

namespace pass {

Pass ResolveBackendOpLayouts() {
  return CreateFunctionPass(RewriteFunction, "ResolveBackendOpLayouts", kResolveBackendOpLayoutsProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto

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
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/op_predicates.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

struct TpopPipeInfo {
  int split = 0;
  std::optional<int> pipe_id;
  std::string op_name;
};

// A cross-core tfree carries no split/id of its own — those live on the matching
// tpop call. This pass copies them onto the tfree op so codegen reads them
// directly (no codegen-side tpop lookup table). Covers both finalizer-created
// (mixed-kernel) and user-written (explicit AIC/AIV) tfrees: by this point the
// tfree's tile arg is the tpop result Var in both cases.
class StampTfreeSplitMutator : public IRMutator {
 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    if (auto assign = As<AssignStmt>(base)) {
      if (auto call = As<Call>(assign->value_); call && op_predicates::IsTPop(call)) {
        TpopPipeInfo info;
        info.split = call->GetKwarg<int>("split", 0);
        if (call->HasKwarg("id")) info.pipe_id = call->GetKwarg<int>("id", 0);
        info.op_name = call->op_->name_;
        tpop_info_[assign->var_.get()] = info;
      }
    }
    return base;
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto eval = As<EvalStmt>(base);
    if (!eval) return base;
    auto call = As<Call>(eval->expr_);
    if (!call || !op_predicates::IsTFree(call) || call->args_.empty()) return base;
    auto tile = AsVarLike(call->args_[0]);
    if (!tile) return base;
    auto it = tpop_info_.find(tile.get());
    if (it == tpop_info_.end()) return base;
    const auto& info = it->second;

    const std::string expected_tpop =
        call->op_->name_ == "system.tfree_to_aic" ? "tile.tpop_from_aic" : "tile.tpop_from_aiv";
    CHECK_SPAN(info.op_name == expected_tpop, call->span_)
        << call->op_->name_ << " requires its tile argument to come from " << expected_tpop << ", got "
        << info.op_name;

    std::optional<int> pipe_id = info.pipe_id;
    if (call->HasKwarg("id")) {
      const int tfree_id = call->GetKwarg<int>("id", 0);
      CHECK_SPAN(tfree_id == info.pipe_id.value_or(0), call->span_)
          << call->op_->name_ << " pipe id " << tfree_id << " does not match originating " << info.op_name
          << " pipe id " << info.pipe_id.value_or(0);
      pipe_id = tfree_id;
    }

    std::vector<std::pair<std::string, std::any>> kwargs;
    kwargs.emplace_back("split", std::any(info.split));
    if (pipe_id.has_value()) kwargs.emplace_back("id", std::any(pipe_id.value()));
    auto new_call = OpRegistry::GetInstance().Create(call->op_->name_, call->args_, kwargs, call->span_);
    return std::make_shared<EvalStmt>(new_call, eval->span_);
  }

 private:
  std::map<const Var*, TpopPipeInfo> tpop_info_;
};

}  // namespace

Pass StampTfreeSplit() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_) return func;
    StampTfreeSplitMutator mutator;
    auto new_body = mutator.VisitStmt(func->body_);
    if (new_body.get() == func->body_.get()) return func;
    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, new_body, func->span_, func->func_type_,
                                      func->level_, func->role_, func->attrs_);
  };
  return CreateFunctionPass(pass_func, "StampTfreeSplit", kStampTfreeSplitProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

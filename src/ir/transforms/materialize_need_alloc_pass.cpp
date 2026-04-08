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
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

constexpr const char* kNoopFuncPrefix = "__pypto_noop_";

static bool IsNeedAllocTensorCreate(const StmtPtr& stmt) {
  auto assign = As<AssignStmt>(stmt);
  if (!assign) return false;
  auto call = As<Call>(assign->value_);
  if (!call || call->op_->name_ != "tensor.create") return false;
  return call->GetKwarg<bool>("need_alloc", false);
}

static CallPtr StripNeedAllocKwarg(const CallPtr& call) {
  std::vector<std::pair<std::string, std::any>> new_kwargs;
  for (const auto& [k, v] : call->kwargs_) {
    if (k != "need_alloc") {
      new_kwargs.emplace_back(k, v);
    }
  }
  return std::make_shared<Call>(call->op_, call->args_, std::move(new_kwargs), call->GetType(), call->span_);
}

class MaterializeNeedAllocMutator : public IRMutator {
 public:
  explicit MaterializeNeedAllocMutator(std::string parent_func_name)
      : parent_func_name_(std::move(parent_func_name)) {}

  StmtPtr VisitStmt_(const SeqStmtsPtr& seq) override {
    std::vector<StmtPtr> new_stmts;
    bool seq_changed = false;

    size_t i = 0;
    while (i < seq->stmts_.size()) {
      // Collect consecutive need_alloc tensor.creates into a batch
      size_t j = i;
      while (j < seq->stmts_.size() && IsNeedAllocTensorCreate(seq->stmts_[j])) {
        j++;
      }

      if (j > i) {
        // Found a batch of consecutive need_alloc creates
        seq_changed = true;
        std::vector<AssignStmtPtr> batch;
        for (size_t k = i; k < j; k++) {
          batch.push_back(As<AssignStmt>(seq->stmts_[k]));
        }
        EmitBatch(batch, new_stmts);
        i = j;
      } else {
        auto visited = VisitStmt(seq->stmts_[i]);
        if (visited != seq->stmts_[i]) seq_changed = true;
        new_stmts.push_back(visited);
        i++;
      }
    }

    if (!seq_changed) return seq;
    changed_ = true;
    return SeqStmts::Flatten(std::move(new_stmts), seq->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& assign) override {
    if (!IsNeedAllocTensorCreate(assign)) {
      return IRMutator::VisitStmt_(assign);
    }

    // Fallback for isolated need_alloc creates not inside a SeqStmts
    std::vector<StmtPtr> out;
    EmitBatch({assign}, out);
    changed_ = true;
    return SeqStmts::Flatten(std::move(out), assign->span_);
  }

  [[nodiscard]] bool changed() const { return changed_; }
  [[nodiscard]] const std::vector<FunctionPtr>& noop_functions() const { return noop_functions_; }

 private:
  void EmitBatch(const std::vector<AssignStmtPtr>& batch, std::vector<StmtPtr>& out_stmts) {
    size_t n = batch.size();

    // Step 1: emit clean tensor.create assignments (without need_alloc) and collect ci_vars
    std::vector<VarPtr> ci_vars;
    std::vector<TypePtr> element_types;

    for (size_t k = 0; k < n; k++) {
      auto call = As<Call>(batch[k]->value_);
      auto tensor_type = As<TensorType>(call->GetType());
      CHECK(tensor_type) << "tensor.create with need_alloc must return TensorType";

      auto clean_call = StripNeedAllocKwarg(call);
      std::string ci_name = batch[k]->var_->name_hint_ + "__pre";
      auto ci_var = std::make_shared<Var>(ci_name, tensor_type, batch[k]->span_);
      ci_vars.push_back(ci_var);
      element_types.push_back(tensor_type);

      out_stmts.push_back(std::make_shared<AssignStmt>(ci_var, clean_call, batch[k]->span_));
    }

    // Step 2: create the noop function with N Out params
    std::string noop_name = kNoopFuncPrefix + parent_func_name_ + "_" + std::to_string(noop_counter_++);

    std::vector<VarPtr> noop_params;
    std::vector<ParamDirection> noop_dirs;
    std::vector<ExprPtr> return_exprs;

    for (size_t k = 0; k < n; k++) {
      std::string param_name = (n > 1) ? ("out" + std::to_string(k)) : "out";
      auto param = std::make_shared<Var>(param_name, element_types[k], batch[k]->span_);
      noop_params.push_back(param);
      noop_dirs.push_back(ParamDirection::Out);
      return_exprs.push_back(param);
    }

    auto noop_body = std::make_shared<ReturnStmt>(return_exprs, batch[0]->span_);
    auto noop_func = std::make_shared<Function>(noop_name, noop_params, noop_dirs, element_types, noop_body,
                                                batch[0]->span_, FunctionType::InCore);
    noop_functions_.push_back(noop_func);

    // Step 3: create the call to the noop function
    auto noop_gvar = std::make_shared<GlobalVar>(noop_name);
    std::vector<ExprPtr> noop_args(ci_vars.begin(), ci_vars.end());

    if (n == 1) {
      auto noop_call =
          std::make_shared<Call>(noop_gvar, std::move(noop_args), element_types[0], batch[0]->span_);
      out_stmts.push_back(std::make_shared<AssignStmt>(batch[0]->var_, noop_call, batch[0]->span_));
    } else {
      auto tuple_type = std::make_shared<TupleType>(element_types);
      auto noop_call = std::make_shared<Call>(noop_gvar, std::move(noop_args), tuple_type, batch[0]->span_);

      auto tuple_var_name = "__noop_result_" + std::to_string(noop_counter_ - 1);
      auto tuple_var = std::make_shared<Var>(tuple_var_name, tuple_type, batch[0]->span_);
      out_stmts.push_back(std::make_shared<AssignStmt>(tuple_var, noop_call, batch[0]->span_));

      for (size_t k = 0; k < n; k++) {
        auto get_item = std::make_shared<TupleGetItemExpr>(std::static_pointer_cast<const Expr>(tuple_var),
                                                           static_cast<int>(k), batch[k]->span_);
        out_stmts.push_back(std::make_shared<AssignStmt>(batch[k]->var_, get_item, batch[k]->span_));
      }
    }
  }

  std::string parent_func_name_;
  int noop_counter_ = 0;
  bool changed_ = false;
  std::vector<FunctionPtr> noop_functions_;
};

ProgramPtr TransformMaterializeNeedAlloc(const ProgramPtr& program) {
  bool any_changed = false;
  std::vector<FunctionPtr> new_functions;
  std::vector<FunctionPtr> all_noop_functions;

  for (const auto& [gvar, func] : program->functions_) {
    if (func->func_type_ != FunctionType::Orchestration) {
      new_functions.push_back(func);
      continue;
    }

    MaterializeNeedAllocMutator mutator(func->name_);
    auto new_body = mutator.VisitStmt(func->body_);

    if (!mutator.changed()) {
      new_functions.push_back(func);
      continue;
    }

    any_changed = true;
    new_functions.push_back(std::make_shared<Function>(
        func->name_, func->params_, func->param_directions_, func->return_types_, new_body, func->span_,
        func->func_type_, func->level_, func->role_, func->attrs_));

    const auto& noops = mutator.noop_functions();
    all_noop_functions.insert(all_noop_functions.end(), noops.begin(), noops.end());
  }

  if (!any_changed) return program;

  // Prepend noop functions before existing functions
  all_noop_functions.insert(all_noop_functions.end(), new_functions.begin(), new_functions.end());
  return std::make_shared<Program>(std::move(all_noop_functions), program->name_, program->span_);
}

}  // namespace

namespace pass {

Pass MaterializeNeedAlloc() {
  return CreateProgramPass(TransformMaterializeNeedAlloc, "MaterializeNeedAlloc", PassProperties{});
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto

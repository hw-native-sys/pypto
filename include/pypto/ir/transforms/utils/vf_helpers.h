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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_VF_HELPERS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_VF_HELPERS_H_

#include <any>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"

namespace pypto::ir {

inline FunctionPtr ResolveVFCall(const CallPtr& call, const ProgramPtr& program) {
  if (!call || !program || !As<GlobalVar>(call->op_)) return nullptr;
  auto callee = program->GetFunction(call->op_->name_);
  return callee && callee->func_type_ == FunctionType::VF ? callee : nullptr;
}

/// The logical helper borrows caller storage. Rebind only for analysis/emission;
/// the Program retains its early-outlined function boundary throughout lowering.
/// Store-version types follow the actual Tile, including its final MemRef/view.
inline RegScopeStmtPtr BindVFRegion(const FunctionPtr& function, const CallPtr& call,
                                    std::vector<std::pair<VarPtr, ExprPtr>>* borrowed_args = nullptr) {
  CHECK_SPAN(function->params_.size() == call->args_.size(), call->span_)
      << "VF argument count must match its declaration";
  RegScopeStmtPtr scope;
  auto stmts = As<SeqStmts>(function->body_);
  const auto body = stmts ? stmts->stmts_ : std::vector<StmtPtr>{function->body_};
  for (const auto& stmt : body) {
    if (auto region = As<RegScopeStmt>(stmt)) {
      CHECK_SPAN(!scope, stmt->span_) << "VF helper must contain exactly one register scope";
      scope = region;
    } else {
      CHECK_SPAN(As<ReturnStmt>(stmt), stmt->span_)
          << "VF helper may only contain a register scope and return";
    }
  }
  CHECK_SPAN(scope, function->span_) << "VF helper requires a register scope";
  std::unordered_map<const Var*, ExprPtr> substitutions;
  for (size_t i = 0; i < function->params_.size(); ++i) {
    ExprPtr argument = call->args_[i];
    if (auto tile = As<TileType>(argument->GetType())) {
      // Canonical caller types may omit default views. Materialize their
      // existing semantics for the register emitter's explicit-view contract;
      // preserve the actual physical storage, never the formal's MemRef.
      auto type = MutableCopy(tile);
      type->tile_view_ = tile_view_semantics::GetEffectiveTileView(*tile);
      auto borrowed = std::make_shared<Var>(function->params_[i]->name_hint_, type, argument->span_);
      if (borrowed_args) borrowed_args->emplace_back(borrowed, argument);
      argument = borrowed;
    }
    substitutions[function->params_[i].get()] = argument;
  }
  auto bound = DeepClone(scope, substitutions).cloned_body;
  class RetypeStores : public IRMutator {
   protected:
    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      auto value = VisitExpr(op->value_);
      auto call = As<Call>(value);
      if (IsOp(call, "reg.store") || IsOp(call, "reg.scatter")) {
        auto typed_call = std::make_shared<Call>(call->op_, call->args_, call->kwargs_, call->attrs_,
                                                 call->args_[0]->GetType(), call->span_);
        auto var = std::make_shared<Var>(op->var_->name_hint_, typed_call->GetType(), op->var_->span_);
        var_remap_[op->var_.get()] = var;
        return std::make_shared<AssignStmt>(var, typed_call, op->span_, op->leading_comments_);
      }
      return IRMutator::VisitStmt_(op);
    }
  } retype;
  return As<RegScopeStmt>(retype.VisitStmt(bound));
}

/// Resolve alias returns from the canonical declaration, never by matching
/// shape/dtype: two same-typed arguments may denote different buffers.
inline std::vector<ExprPtr> VFReturnedArguments(const FunctionPtr& function, const CallPtr& call) {
  auto indices = return_lineage::ReturnedParamIndices(function, nullptr);
  CHECK_SPAN(indices.size() == function->return_types_.size(), function->span_)
      << "VF returns must explicitly name captured Tile parameters";
  std::vector<ExprPtr> result;
  for (const auto& index : indices) {
    CHECK_SPAN(index && *index < call->args_.size(), call->span_)
        << "VF return must alias a captured Tile parameter";
    CHECK_SPAN(As<TileType>(call->args_[*index]->GetType()), call->span_)
        << "VF cannot return a register or local scalar";
    result.push_back(call->args_[*index]);
  }
  return result;
}

/// Validate both authored and transformed helper calls. Placement is lexical
/// before mixed-kernel expansion and strictly AIV at the codegen boundary.
inline void CheckVFCalls(const ProgramPtr& program, bool final_placement = false) {
  return_lineage::ScopedTraceCache lineage_cache;
  class Checker : public IRVisitor {
   public:
    Checker(ProgramPtr program, bool final_placement)
        : program_(std::move(program)), final_(final_placement) {}
    void VisitFunction(const FunctionPtr& function) override {
      function_ = function;
      device_ = function->func_type_ == FunctionType::AIV ||
                (!final_ && (function->func_type_ == FunctionType::InCore ||
                             function->func_type_ == FunctionType::Inline));
      if (function->func_type_ == FunctionType::VF) {
        CHECK_SPAN(!function->level_ && !function->role_ && !function->requires_runtime_binding_,
                   function->span_)
            << "VF helpers cannot carry runtime task metadata";
        for (const auto& param : function->params_) {
          CHECK_SPAN(As<TileType>(param->GetType()) || As<ScalarType>(param->GetType()), param->span_)
              << "VF parameters must be Tiles or Scalars";
        }
        std::vector<ExprPtr> args(function->params_.begin(), function->params_.end());
        auto call =
            std::make_shared<Call>(std::make_shared<GlobalVar>(function->name_), args,
                                   std::vector<std::pair<std::string, std::any>>{},
                                   std::make_shared<TupleType>(function->return_types_), function->span_);
        BindVFRegion(function, call);
        VFReturnedArguments(function, call);
      }
      IRVisitor::VisitFunction(function);
    }

   protected:
    void VisitStmt_(const InCoreScopeStmtPtr& scope) override {
      bool saved = device_;
      device_ = !final_ && function_->func_type_ != FunctionType::AIC;
      IRVisitor::VisitStmt_(scope);
      device_ = saved;
    }
    void VisitExpr_(const CallPtr& call) override {
      if (auto helper = ResolveVFCall(call, program_)) {
        CHECK_SPAN(device_, call->span_) << "VF calls require an InCore/AIV context and final AIV placement";
        CHECK_SPAN(call->args_.size() == helper->params_.size(), call->span_)
            << "VF argument count must match its declaration";
        VFReturnedArguments(helper, call);
      }
      CHECK_SPAN(function_->func_type_ != FunctionType::VF || !As<GlobalVar>(call->op_), call->span_)
          << "VF helpers cannot call other functions";
      IRVisitor::VisitExpr_(call);
    }
    void VisitExpr_(const SubmitPtr& submit) override {
      auto callee = program_->GetFunction(submit->op_->name_);
      CHECK_SPAN(!callee || callee->func_type_ != FunctionType::VF, submit->span_)
          << "VF helpers are synchronous device calls, not submit tasks";
      IRVisitor::VisitExpr_(submit);
    }

   private:
    ProgramPtr program_;
    FunctionPtr function_;
    bool final_;
    bool device_ = false;
  } checker(program, final_placement);
  checker.VisitProgram(program);
}
}  // namespace pypto::ir
#endif  // PYPTO_IR_TRANSFORMS_UTILS_VF_HELPERS_H_

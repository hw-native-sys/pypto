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
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/arith/ir_mutator_with_analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/alloc_batching.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/normalize_stmt_structure.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/spmd_launch_utils.h"
#include "pypto/ir/transforms/utils/wrapper_call_utils.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {

// Only immutable scalar arithmetic may be substituted into the analyzer. A
// tensor.read is a snapshot, not an expression that can be re-evaluated later.
bool IsScalarArithmetic(const ExprPtr& expr) {
  if (AsVarLike(expr) || As<ConstInt>(expr) || As<ConstBool>(expr)) return true;
  if (auto binary = As<BinaryExpr>(expr)) {
    return IsScalarArithmetic(binary->left_) && IsScalarArithmetic(binary->right_);
  }
  if (auto unary = As<UnaryExpr>(expr)) return IsScalarArithmetic(unary->operand_);
  return false;
}

struct CalleeInfo {
  FunctionPtr function;
  std::vector<ParamDirection> directions;
  std::vector<std::optional<size_t>> returned_params;
  size_t required_params = 0;
};

class ScratchUses : public IRVisitor {
 public:
  std::unordered_map<const Var*, AssignStmtPtr> creates;
  std::unordered_map<const Var*, size_t> uses;

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (alloc_batching::IsInjectedGMPipeCreateVar(op->var_) && IsOp(As<Call>(op->value_), "tensor.create")) {
      creates.emplace(op->var_.get(), op);
    }
    VisitExpr(op->value_);
  }
  void VisitExpr_(const VarPtr& op) override { ++uses[op.get()]; }
};

class RemoveMovedScratch : public IRMutator {
 public:
  std::unordered_set<const Stmt*> moved;

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    if (moved.count(op.get())) return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
    return op;
  }
};

// One traversal per function, with indexed tuple substitutions and a cached
// return-to-param map per callee. Work is linear in IR plus call argument sizes.
class LaunchLegalizer : public arith::IRMutatorWithAnalyzer {
 public:
  LaunchLegalizer(arith::Analyzer* analyzer, const std::unordered_map<std::string, CalleeInfo>& callees,
                  const ScratchUses& scratch, bool verify_only = false)
      : IRMutatorWithAnalyzer(analyzer), callees_(callees), scratch_(scratch), verify_only_(verify_only) {}

  RemoveMovedScratch scratch_cleanup;

 protected:
  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
    if (auto var = AsVarLike(op->tuple_)) {
      auto it = tuple_results_.find(var.get());
      if (it != tuple_results_.end()) return it->second.at(op->index_);
    }
    return IRMutator::VisitExpr_(op);
  }

  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto scalar = scalar_results_.find(op.get());
    if (scalar != scalar_results_.end()) return scalar->second;
    auto it = tuple_results_.find(op.get());
    if (it == tuple_results_.end()) return op;
    return std::make_shared<MakeTuple>(it->second, op->span_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto condition = VisitExpr(op->condition_);
    StmtPtr then_body;
    std::optional<StmtPtr> else_body;
    {
      auto constraint = analyzer_->GetConstraintContext(condition);
      then_body = VisitScoped(op->then_body_);
    }
    if (op->else_body_) {
      auto constraint = analyzer_->GetConstraintContext(MakeNot(condition, op->span_));
      else_body = VisitScoped(*op->else_body_);
    }
    if (condition == op->condition_ && then_body == op->then_body_ && else_body == op->else_body_) return op;
    return std::make_shared<IfStmt>(condition, then_body, else_body, op->return_vars_, op->span_,
                                    op->leading_comments_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    const auto mark = bindings_.size();
    auto result = IRMutatorWithAnalyzer::VisitStmt_(op);
    UnbindSince(mark);
    return result;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    const auto mark = bindings_.size();
    auto result = IRMutator::VisitStmt_(op);
    UnbindSince(mark);
    return result;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto value = VisitExpr(op->value_);
    if (auto guarded = Guard(value, op->var_, op->leading_comments_)) return guarded;
    if (auto scalar = As<ScalarType>(op->var_->GetType());
        scalar && scalar->dtype_ != DataType::TASK_ID && IsScalarArithmetic(value)) {
      analyzer_->Bind(op->var_, value);
      bindings_.push_back(op->var_);
    }
    if (value == op->value_) return op;
    return std::make_shared<AssignStmt>(op->var_, value, op->span_, op->leading_comments_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto expr = VisitExpr(op->expr_);
    if (auto guarded = Guard(expr, nullptr, op->leading_comments_)) return guarded;
    if (expr == op->expr_) return op;
    return std::make_shared<EvalStmt>(expr, op->span_, op->leading_comments_);
  }

 private:
  const std::unordered_map<std::string, CalleeInfo>& callees_;
  const ScratchUses& scratch_;
  bool verify_only_;
  std::unordered_map<const Var*, std::vector<ExprPtr>> tuple_results_;
  std::unordered_map<const Var*, VarPtr> scalar_results_;
  std::vector<VarPtr> bindings_;

  void UnbindSince(size_t mark) {
    while (bindings_.size() > mark) {
      analyzer_->Unbind(bindings_.back());
      bindings_.pop_back();
    }
  }

  StmtPtr VisitScoped(const StmtPtr& body) {
    const auto mark = bindings_.size();
    auto result = VisitStmt(body);
    UnbindSince(mark);
    return result;
  }

  StmtPtr Guard(const ExprPtr& expr, const VarPtr& lhs, const std::vector<std::string>& comments) {
    auto call = As<Call>(expr);
    auto submit = As<Submit>(expr);
    if (!call && !submit) return nullptr;
    auto op = call ? call->op_ : submit->op_;
    auto it = callees_.find(op->name_);
    if (it == callees_.end()) return nullptr;
    const auto& info = it->second;
    ExprPtr count =
        call ? call->GetAttr<ExprPtr>(kAttrCoreNum, nullptr) : submit->core_num_.value_or(nullptr);
    if (!count) count = info.function->GetAttr<ExprPtr>(kAttrCoreNum, nullptr);
    if (!count) return nullptr;
    auto zero = std::make_shared<ConstInt>(0, As<ScalarType>(count->GetType())->dtype_, expr->span_);
    // Range analysis can lose a constraint on a Var when simplifying its
    // binding to composite arithmetic. CanProve also consults known predicates.
    if (analyzer_->CanProveGreaterEqual(count, 1) || analyzer_->CanProve(MakeGt(count, zero))) return nullptr;
    INTERNAL_CHECK_SPAN(!verify_only_, expr->span_)
        << "Internal error: potentially non-positive SPMD launch '" << op->name_
        << "' reached codegen without a positive guard. Run LegalizeSpmdLaunches before dependency analysis.";
    const bool proven_empty = analyzer_->CanProveLess(count, 1);

    const auto& args = call ? call->args_ : submit->args_;
    const auto& params = info.function->params_;
    INTERNAL_CHECK_SPAN(args.size() <= params.size() && args.size() >= info.required_params, expr->span_)
        << "Internal error: invalid SPMD argument coverage";
    // Match the parser's optional Out/InOut binding: reserve arguments for
    // every remaining required input, including a trailing CommCtx suffix.
    // An omitted output need not be a trailing parameter.
    std::vector<ExprPtr> arguments(params.size());
    size_t arg_index = 0;
    size_t remaining_required = info.required_params;
    for (size_t i = 0; i < params.size(); ++i) {
      if (info.function->param_directions_[i] == ParamDirection::In) {
        --remaining_required;
      } else if (args.size() - arg_index <= remaining_required) {
        continue;
      }
      arguments[i] = args[arg_index++];
    }
    auto argument = [&](size_t index) -> ExprPtr { return arguments[index]; };
    std::ostringstream missing;
    for (size_t i = 0; i < params.size(); ++i) {
      if (info.directions[i] == ParamDirection::In || !AsTensorTypeLike(params[i]->GetType())) continue;
      auto arg = argument(i);
      if (!arg || !AsTensorTypeLike(arg->GetType())) {
        missing << " '" << params[i]->name_hint_ << "' (parameter " << i << ")";
      }
    }
    CHECK_SPAN(missing.str().empty(), expr->span_)
        << "LegalizeSpmdLaunches: launch '" << op->name_
        << "' may have a non-positive core_num; output tensors" << missing.str()
        << " are not preallocated. Allocate and pass every Out/InOut tensor before this launch.";

    std::vector<StmtPtr> prefix;
    if (!AsVarLike(count) && !As<ConstInt>(count)) {
      auto count_var = std::make_shared<Var>("spmd_core_num", count->GetType(), count->span_);
      prefix.push_back(std::make_shared<AssignStmt>(count_var, count, count->span_));
      count = count_var;
    }
    ExprPtr launch = expr;
    if (submit) {
      auto copy = MutableCopy(submit);
      copy->core_num_ = count;
      launch = copy;
    } else {
      auto copy = MutableCopy(call);
      bool replaced = false;
      for (auto& [key, value] : copy->attrs_) {
        if (key == kAttrCoreNum) {
          value = count;
          replaced = true;
        }
      }
      if (!replaced) copy->attrs_.emplace_back(kAttrCoreNum, count);
      launch = copy;
    }

    std::vector<StmtPtr> then_stmts, else_stmts;
    for (const auto& arg : args) {
      auto var = AsVarLike(arg);
      auto create = scratch_.creates.find(var.get());
      if (create == scratch_.creates.end()) continue;
      CHECK_SPAN(scratch_.uses.at(var.get()) == 1, expr->span_)
          << "LegalizeSpmdLaunches: compiler GM pipe buffer '" << var->name_hint_
          << "' has multiple uses; cannot move its allocation into a potentially empty launch.";
      // The placeholder has a constant shape; codegen derives its real size
      // from the following dispatch. Keep the allocation next to that dispatch
      // and avoid a zero-sized workspace allocation on the empty path.
      then_stmts.push_back(MutableCopy(create->second));
      scratch_cleanup.moved.insert(create->second.get());
    }
    std::vector<ExprPtr> then_values, else_values;
    std::vector<VarPtr> results;
    if (lhs) {
      auto tuple = As<TupleType>(lhs->GetType());
      std::vector<TypePtr> types = tuple ? tuple->types_ : std::vector<TypePtr>{lhs->GetType()};
      std::vector<ExprPtr> preserved(types.size());
      for (size_t i = 0; i < types.size(); ++i) {
        auto scalar = As<ScalarType>(types[i]);
        if (scalar && scalar->dtype_ == DataType::TASK_ID) continue;
        if (AsTensorTypeLike(types[i]) && i < info.returned_params.size()) {
          const auto& parameter = info.returned_params[i];
          if (parameter.has_value()) preserved[i] = argument(parameter.value());
        }
        CHECK_SPAN(preserved[i], expr->span_)
            << "LegalizeSpmdLaunches: launch '" << op->name_ << "' return " << i
            << " cannot be mapped to a caller-provided Tensor parameter. "
            << "Return a preallocated Out/InOut parameter for a potentially empty launch.";
        // A call's tuple type may retain callee-local shape Vars. All new SSA
        // values must instead use the mapped caller buffer's shape and view.
        types[i] = preserved[i]->GetType();
      }
      TypePtr result_type = tuple ? std::make_shared<TupleType>(types) : types.front();
      if (auto launch_submit = As<Submit>(launch)) {
        launch = std::make_shared<Submit>(
            launch_submit->op_, launch_submit->args_, launch_submit->deps_, launch_submit->kwargs_,
            launch_submit->attrs_, result_type, launch_submit->span_, launch_submit->core_num_,
            launch_submit->sync_start_, launch_submit->allow_early_resolve_, launch_submit->predicate_);
      } else {
        auto launch_call = As<Call>(launch);
        launch = std::make_shared<Call>(launch_call->op_, launch_call->args_, launch_call->kwargs_,
                                        launch_call->attrs_, result_type, launch_call->span_);
      }
      auto local = std::make_shared<Var>(lhs->name_hint_ + "_nonempty", result_type, lhs->span_);
      then_stmts.push_back(std::make_shared<AssignStmt>(local, launch, expr->span_));
      for (size_t i = 0; i < types.size(); ++i) {
        VarPtr true_value = local;
        VarPtr result = lhs;
        if (!tuple && result_type != lhs->GetType()) {
          result = std::make_shared<Var>(lhs->name_hint_, result_type, lhs->span_);
          scalar_results_[lhs.get()] = result;
        }
        if (tuple) {
          true_value =
              std::make_shared<Var>(lhs->name_hint_ + "_nonempty_" + std::to_string(i), types[i], lhs->span_);
          then_stmts.push_back(std::make_shared<AssignStmt>(
              true_value, std::make_shared<TupleGetItemExpr>(local, static_cast<int>(i), lhs->span_),
              lhs->span_));
          result =
              std::make_shared<Var>(lhs->name_hint_ + "_guarded_" + std::to_string(i), types[i], lhs->span_);
        }
        results.push_back(result);
        then_values.push_back(true_value);
        auto scalar = As<ScalarType>(types[i]);
        if (scalar && scalar->dtype_ == DataType::TASK_ID) {
          auto tid = std::make_shared<Var>("spmd_empty_tid", types[i], lhs->span_);
          std::vector<VarPtr> deps;
          if (submit) {
            for (const auto& dep : submit->deps_) {
              auto var = AsVarLike(dep);
              INTERNAL_CHECK_SPAN(var, expr->span_) << "Internal error: flattened Submit dep must be a Var";
              deps.push_back(var);
            }
          } else {
            deps = call->GetAttr<std::vector<VarPtr>>(kAttrManualDepEdges, {});
          }
          std::vector<std::pair<std::string, std::any>> dummy_attrs{{kAttrDummyTask, true}};
          if (!deps.empty()) dummy_attrs = WithManualDepEdgesAttr(std::move(dummy_attrs), std::move(deps));
          auto dummy = std::make_shared<Call>(
              OpRegistry::GetInstance().GetOp("system.task_dummy"), std::vector<ExprPtr>{},
              std::vector<std::pair<std::string, std::any>>{}, std::move(dummy_attrs), types[i], expr->span_);
          else_stmts.push_back(std::make_shared<AssignStmt>(tid, dummy, expr->span_));
          else_values.push_back(tid);
        } else {
          else_values.push_back(preserved[i]);
        }
      }
      if (tuple) tuple_results_[lhs.get()] = std::vector<ExprPtr>(results.begin(), results.end());
    } else {
      then_stmts.push_back(std::make_shared<EvalStmt>(launch, expr->span_));
    }
    then_stmts.push_back(std::make_shared<YieldStmt>(then_values, expr->span_));
    if (proven_empty) {
      // Do not leave an unreachable literal zero launch for print/reparse (the
      // source parser correctly rejects that spelling). Keep the empty-path
      // task and bind the same outer SSA definitions as the dynamic If would.
      prefix.insert(prefix.end(), else_stmts.begin(), else_stmts.end());
      for (size_t i = 0; i < results.size(); ++i) {
        prefix.push_back(std::make_shared<AssignStmt>(results[i], else_values[i], expr->span_));
      }
      return SeqStmts::Flatten(prefix, expr->span_);
    }
    else_stmts.push_back(std::make_shared<YieldStmt>(else_values, expr->span_));
    auto condition =
        MakeGt(count, std::make_shared<ConstInt>(0, As<ScalarType>(count->GetType())->dtype_, expr->span_),
               expr->span_);
    prefix.push_back(std::make_shared<IfStmt>(condition, SeqStmts::Flatten(then_stmts, expr->span_),
                                              SeqStmts::Flatten(else_stmts, expr->span_), results,
                                              expr->span_, comments));
    return SeqStmts::Flatten(prefix, expr->span_);
  }
};

std::unordered_map<std::string, CalleeInfo> BuildCalleeInfo(const ProgramPtr& program) {
  auto effective = ComputeWrapperEffectiveDirections(program);
  std::unordered_map<std::string, CalleeInfo> callees;
  for (const auto& [_, func] : program->functions_) {
    auto type = func->func_type_;
    if (type != FunctionType::InCore && type != FunctionType::AIC && type != FunctionType::AIV &&
        type != FunctionType::Group && type != FunctionType::Spmd) {
      continue;
    }
    auto dirs = effective.find(func.get());
    CalleeInfo info{func, dirs == effective.end() ? func->param_directions_ : dirs->second,
                    return_lineage::ExplicitReturnedParamIndices(func), 0};
    for (auto direction : func->param_directions_) {
      if (direction == ParamDirection::In) ++info.required_params;
    }
    callees.emplace(func->name_, std::move(info));
  }
  return callees;
}

ProgramPtr TransformProgram(const ProgramPtr& program) {
  auto callees = BuildCalleeInfo(program);
  auto functions = program->functions_;
  bool changed = false;
  for (auto& [_, func] : functions) {
    // Graph launch counts must remain constant; LegalizeGraphBoundary owns its
    // diagnostic. Never hide an invalid Graph launch behind a generated guard.
    if (func->func_type_ != FunctionType::Orchestration) continue;
    auto analyzer = std::make_shared<arith::Analyzer>();
    ScratchUses scratch;
    scratch.VisitStmt(func->body_);
    LaunchLegalizer legalizer(analyzer.get(), callees, scratch);
    auto result = legalizer.VisitFunction(func);
    result = NormalizeStmtStructure(legalizer.scratch_cleanup.VisitFunction(result));
    changed |= result != func;
    func = std::move(result);
  }
  if (!changed) return program;
  return std::make_shared<Program>(std::move(functions), program->name_, program->span_);
}

}  // namespace

void VerifySpmdLaunchGuards(const ProgramPtr& program, const FunctionPtr& function) {
  if (function->func_type_ != FunctionType::Orchestration) return;
  auto callees = BuildCalleeInfo(program);
  auto analyzer = std::make_shared<arith::Analyzer>();
  ScratchUses scratch;
  LaunchLegalizer verifier(analyzer.get(), callees, scratch, /*verify_only=*/true);
  verifier.VisitFunction(function);
}

namespace pass {
Pass LegalizeSpmdLaunches() {
  return CreateProgramPass(TransformProgram, "LegalizeSpmdLaunches", kLegalizeSpmdLaunchesProperties);
}
}  // namespace pass
}  // namespace pypto::ir

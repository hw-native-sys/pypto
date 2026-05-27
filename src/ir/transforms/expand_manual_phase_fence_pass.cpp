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

#include <algorithm>
#include <any>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {
namespace {

constexpr int64_t kPhaseFenceMinEstimatedEdgeSavings = 1;

struct BarrierDecision {
  const Var* source = nullptr;
  VarPtr source_var;
  VarPtr barrier_var;
  StmtPtr barrier_stmt;
  std::unordered_set<const Call*> consumers;
};

static std::optional<int64_t> EvalConstInt(const ExprPtr& expr) {
  if (auto ci = As<ConstInt>(expr)) return ci->value_;
  return std::nullopt;
}

static int64_t EvalConstTripCount(const ForStmtPtr& for_stmt) {
  auto start = EvalConstInt(for_stmt->start_);
  auto stop = EvalConstInt(for_stmt->stop_);
  auto step = EvalConstInt(for_stmt->step_);
  if (!start || !stop || !step || *step <= 0) return 0;
  int64_t trip = (*stop - *start + *step - 1) / *step;
  return trip > 0 ? trip : 0;
}

static bool IsTaskIdArrayVar(const VarPtr& var) {
  if (!var) return false;
  auto array_ty = As<ArrayType>(var->GetType());
  return array_ty && array_ty->dtype_ == DataType::TASK_ID;
}

static std::optional<VarPtr> GetSingleManualDepArray(const CallPtr& call) {
  if (!call) return std::nullopt;
  if (call->GetAttr<bool>(kAttrDummyTask, false)) return std::nullopt;
  for (const auto& [k, v] : call->attrs_) {
    if (k != kAttrManualDepEdges) continue;
    const auto* edges = std::any_cast<std::vector<VarPtr>>(&v);
    if (!edges || edges->size() != 1 || !(*edges)[0]) return std::nullopt;
    return (*edges)[0];
  }
  return std::nullopt;
}

static std::optional<VarPtr> GetSingleManualDepTaskIdArray(const CallPtr& call) {
  auto dep = GetSingleManualDepArray(call);
  if (!dep.has_value() || !IsTaskIdArrayVar(*dep)) return std::nullopt;
  return dep;
}

static bool ManualDepsExactlyArray(const CallPtr& call, const Var* target) {
  auto dep = GetSingleManualDepTaskIdArray(call);
  return dep.has_value() && dep->get() == target;
}

static bool ShouldEmitPhaseFenceBarrier(int64_t producer_count, int64_t consumer_count) {
  if (producer_count <= 0 || consumer_count <= 0) return false;
  const int64_t estimated_saving = producer_count * consumer_count - (producer_count + consumer_count);
  return estimated_saving >= kPhaseFenceMinEstimatedEdgeSavings;
}

static std::vector<StmtPtr> FlattenToVector(const StmtPtr& body) {
  if (auto seq = As<SeqStmts>(body)) return seq->stmts_;
  if (!body) return {};
  return {body};
}

static StmtPtr MakeSeqOrStmt(std::vector<StmtPtr> stmts, const Span& span) {
  if (stmts.empty()) return std::make_shared<const SeqStmts>(std::vector<StmtPtr>{}, span);
  if (stmts.size() == 1) return stmts[0];
  return SeqStmts::Flatten(std::move(stmts), span);
}

static std::vector<VarPtr> CollectDirectManualDepArrays(const StmtPtr& body) {
  class Collector : public IRVisitor {
   public:
    std::vector<VarPtr> arrays;
    std::unordered_set<const Var*> seen;

    void VisitStmt_(const ForStmtPtr&) override {}

    void VisitExpr_(const CallPtr& call) override {
      auto dep = GetSingleManualDepTaskIdArray(call);
      if (dep.has_value() && seen.insert(dep->get()).second) arrays.push_back(*dep);
      IRVisitor::VisitExpr_(call);
    }
  };
  Collector collector;
  collector.VisitStmt(body);
  return collector.arrays;
}

static bool BodyUpdatesArray(const StmtPtr& body, const Var* target) {
  class Finder : public IRVisitor {
   public:
    bool found = false;
    const Var* target = nullptr;

    void VisitStmt_(const ForStmtPtr&) override {}

    void VisitStmt_(const AssignStmtPtr& assign) override {
      if (found) return;
      auto call = As<Call>(assign->value_);
      if (call && call->op_->name_ == "array.update_element" && !call->args_.empty()) {
        auto base = AsVarLike(call->args_[0]);
        if (base && base.get() == target) {
          found = true;
          return;
        }
      }
      IRVisitor::VisitStmt_(assign);
    }
  };
  Finder finder;
  finder.target = target;
  finder.VisitStmt(body);
  return finder.found;
}

static std::unordered_set<const Var*> CollectBodyDefinedVars(const StmtPtr& body) {
  class Collector : public IRVisitor {
   public:
    std::unordered_set<const Var*> vars;

    void AddVars(const std::vector<VarPtr>& defined_vars) {
      for (const auto& var : defined_vars) {
        if (var) vars.insert(var.get());
      }
    }

    void VisitStmt_(const AssignStmtPtr& assign) override {
      if (assign->var_) vars.insert(assign->var_.get());
      IRVisitor::VisitStmt_(assign);
    }

    void VisitStmt_(const IfStmtPtr& if_stmt) override {
      AddVars(if_stmt->return_vars_);
      IRVisitor::VisitStmt_(if_stmt);
    }

    void VisitStmt_(const ForStmtPtr& for_stmt) override { AddVars(for_stmt->return_vars_); }

    void VisitStmt_(const WhileStmtPtr& while_stmt) override {
      AddVars(while_stmt->return_vars_);
      for (const auto& iter_arg : while_stmt->iter_args_) {
        if (iter_arg) vars.insert(iter_arg.get());
      }
      IRVisitor::VisitStmt_(while_stmt);
    }
  };
  Collector collector;
  collector.VisitStmt(body);
  return collector.vars;
}

static int64_t CountManualDepConsumersOnArray(const StmtPtr& body, const Var* target) {
  class Counter : public IRVisitor {
   public:
    int64_t count = 0;
    const Var* target = nullptr;

    void VisitStmt_(const ForStmtPtr&) override {}

    void VisitExpr_(const CallPtr& call) override {
      if (ManualDepsExactlyArray(call, target)) ++count;
      IRVisitor::VisitExpr_(call);
    }
  };
  Counter counter;
  counter.target = target;
  counter.VisitStmt(body);
  return counter.count;
}

static void CollectCoveredConsumers(const StmtPtr& body, const Var* target,
                                    std::unordered_set<const Call*>* consumers) {
  class Collector : public IRVisitor {
   public:
    const Var* target = nullptr;
    std::unordered_set<const Call*>* consumers = nullptr;

    void VisitStmt_(const ForStmtPtr&) override {}

    void VisitExpr_(const CallPtr& call) override {
      if (ManualDepsExactlyArray(call, target)) consumers->insert(call.get());
      IRVisitor::VisitExpr_(call);
    }
  };
  Collector collector;
  collector.target = target;
  collector.consumers = consumers;
  collector.VisitStmt(body);
}

static int64_t GetArrayProducerCount(const VarPtr& array_var) {
  auto array_ty = As<ArrayType>(array_var->GetType());
  if (!array_ty) return 0;
  if (auto ci = As<ConstInt>(array_ty->extent())) return ci->value_;
  return 0;
}

static CallPtr RewriteManualDepsToBarrier(const CallPtr& call, const VarPtr& barrier_var) {
  return std::make_shared<Call>(call->op_, call->args_, call->kwargs_,
                                WithManualDepEdgesAttr(call->attrs_, {barrier_var}), call->GetType(),
                                call->span_);
}

static StmtPtr MakeBarrierStmt(const VarPtr& source_var, VarPtr* barrier_var, const Span& span,
                               int64_t barrier_idx) {
  std::string name = "phase_fence_barrier_" + std::to_string(barrier_idx) + "_tid";
  auto tid_type = std::make_shared<ScalarType>(DataType::TASK_ID);
  *barrier_var = std::make_shared<Var>(name, tid_type, span);
  std::vector<std::pair<std::string, std::any>> attrs;
  attrs.emplace_back(kAttrDummyTask, true);
  attrs.emplace_back(kAttrManualDepEdges, std::vector<VarPtr>{source_var});
  auto call = std::make_shared<Call>(OpRegistry::GetInstance().GetOp("system.task_dummy"),
                                     std::vector<ExprPtr>{}, std::vector<std::pair<std::string, std::any>>{},
                                     std::move(attrs), tid_type, span);
  return std::make_shared<const AssignStmt>(*barrier_var, call, span);
}

class ManualPhaseFenceMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& op) override {
    const bool saved = in_manual_scope_;
    in_manual_scope_ = op->manual_;
    auto new_body = VisitStmt(op->body_);
    in_manual_scope_ = saved;
    if (new_body.get() != op->body_.get()) {
      return std::make_shared<const RuntimeScopeStmt>(op->manual_, op->name_hint_, std::move(new_body),
                                                      op->span_, op->leading_comments_, op->attrs_);
    }
    return op;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    std::vector<BarrierDecision> decisions;
    if (in_manual_scope_) {
      decisions = BuildDecisions(op, op->body_);
    }

    std::unordered_map<const Call*, VarPtr> consumer_to_barrier;
    for (const auto& decision : decisions) {
      for (const Call* consumer : decision.consumers) {
        consumer_to_barrier[consumer] = decision.barrier_var;
      }
    }

    auto body_with_current_rewrites = RewriteCoveredConsumers(op->body_, consumer_to_barrier);
    if (!decisions.empty() && op->kind_ != ForKind::Parallel) {
      auto body_stmts = FlattenToVector(body_with_current_rewrites);
      std::vector<StmtPtr> with_barriers;
      with_barriers.reserve(decisions.size() + body_stmts.size());
      for (const auto& decision : decisions) {
        with_barriers.push_back(decision.barrier_stmt);
      }
      with_barriers.insert(with_barriers.end(), body_stmts.begin(), body_stmts.end());
      body_with_current_rewrites = MakeSeqOrStmt(std::move(with_barriers), op->span_);
    }

    auto body_with_nested = VisitStmt(body_with_current_rewrites);
    auto new_start = VisitExpr(op->start_);
    auto new_stop = VisitExpr(op->stop_);
    auto new_step = VisitExpr(op->step_);
    const bool loop_changed = body_with_nested.get() != op->body_.get() ||
                              new_start.get() != op->start_.get() || new_stop.get() != op->stop_.get() ||
                              new_step.get() != op->step_.get();
    if (loop_changed && (decisions.empty() || op->kind_ != ForKind::Parallel)) {
      return std::make_shared<const ForStmt>(op->loop_var_, std::move(new_start), std::move(new_stop),
                                             std::move(new_step), op->iter_args_, std::move(body_with_nested),
                                             op->return_vars_, op->span_, op->kind_, op->chunk_config_,
                                             op->attrs_, op->leading_comments_);
    }
    if (!decisions.empty()) {
      auto new_for = std::make_shared<const ForStmt>(
          op->loop_var_, std::move(new_start), std::move(new_stop), std::move(new_step), op->iter_args_,
          std::move(body_with_nested), op->return_vars_, op->span_, op->kind_, op->chunk_config_, op->attrs_,
          op->leading_comments_);
      std::vector<StmtPtr> with_barriers;
      with_barriers.reserve(decisions.size() + 1);
      for (const auto& decision : decisions) {
        with_barriers.push_back(decision.barrier_stmt);
      }
      with_barriers.push_back(new_for);
      return MakeSeqOrStmt(std::move(with_barriers), op->span_);
    }
    return op;
  }

 private:
  std::vector<BarrierDecision> BuildDecisions(const ForStmtPtr& for_stmt, const StmtPtr& body) {
    std::vector<BarrierDecision> decisions;
    std::unordered_set<const Var*> already_decided;
    std::unordered_set<const Var*> current_iter_args;

    auto try_add = [&](const VarPtr& match_var, const VarPtr& barrier_source_var, int64_t consumer_count) {
      if (!match_var || !barrier_source_var || !already_decided.insert(match_var.get()).second) return;
      const int64_t producer_count = GetArrayProducerCount(match_var);
      if (!ShouldEmitPhaseFenceBarrier(producer_count, consumer_count)) return;
      BarrierDecision decision;
      decision.source = match_var.get();
      decision.source_var = barrier_source_var;
      decision.barrier_stmt =
          MakeBarrierStmt(barrier_source_var, &decision.barrier_var, for_stmt->span_, barrier_counter_++);
      CollectCoveredConsumers(body, match_var.get(), &decision.consumers);
      if (!decision.consumers.empty()) decisions.push_back(std::move(decision));
    };

    const bool is_parallel = for_stmt->kind_ == ForKind::Parallel;
    int64_t trip_count = is_parallel ? EvalConstTripCount(for_stmt) : 1;
    if (is_parallel && trip_count <= 0) return decisions;

    for (const auto& iter_arg : for_stmt->iter_args_) {
      if (iter_arg) current_iter_args.insert(iter_arg.get());
    }

    const auto body_defined_vars = CollectBodyDefinedVars(body);
    for (const auto& dep_array : CollectDirectManualDepArrays(body)) {
      if (!dep_array || current_iter_args.count(dep_array.get()) != 0 ||
          body_defined_vars.count(dep_array.get()) != 0 || BodyUpdatesArray(body, dep_array.get())) {
        continue;
      }
      int64_t consumers = CountManualDepConsumersOnArray(body, dep_array.get());
      if (is_parallel) consumers *= trip_count;
      try_add(dep_array, dep_array, consumers);
    }

    return decisions;
  }

  StmtPtr RewriteCoveredConsumers(const StmtPtr& body,
                                  const std::unordered_map<const Call*, VarPtr>& consumer_to_barrier) {
    class Rewriter : public IRMutator {
     public:
      explicit Rewriter(const std::unordered_map<const Call*, VarPtr>& consumer_to_barrier)
          : consumer_to_barrier_(consumer_to_barrier) {}

      ExprPtr VisitExpr_(const CallPtr& call) override {
        auto it = consumer_to_barrier_.find(call.get());
        if (it != consumer_to_barrier_.end()) {
          return RewriteManualDepsToBarrier(call, it->second);
        }
        return IRMutator::VisitExpr_(call);
      }

     private:
      const std::unordered_map<const Call*, VarPtr>& consumer_to_barrier_;
    };

    if (consumer_to_barrier.empty()) return body;
    Rewriter rewriter(consumer_to_barrier);
    return rewriter.VisitStmt(body);
  }

  bool in_manual_scope_ = false;
  int64_t barrier_counter_ = 0;
};

ProgramPtr TransformExpandManualPhaseFence(const ProgramPtr& program) {
  if (!program) return program;
  auto new_functions = program->functions_;
  bool changed = false;
  for (auto& [gvar, func] : new_functions) {
    if (!func || !func->body_) continue;
    if (func->func_type_ != FunctionType::Orchestration) continue;
    ManualPhaseFenceMutator mutator;
    auto new_body = mutator.VisitStmt(func->body_);
    if (new_body.get() == func->body_.get()) continue;
    func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, new_body, func->span_, func->func_type_,
                                      func->level_, func->role_, func->attrs_);
    changed = true;
  }
  if (!changed) return program;
  return std::make_shared<Program>(std::move(new_functions), program->comm_groups_, program->name_,
                                   program->span_);
}

}  // namespace

Pass ExpandManualPhaseFence() {
  return CreateProgramPass(TransformExpandManualPhaseFence, "ExpandManualPhaseFence",
                           kExpandManualPhaseFenceProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

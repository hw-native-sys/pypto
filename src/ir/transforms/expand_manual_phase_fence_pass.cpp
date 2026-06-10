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
#include <functional>
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
  // Consumers are keyed on the Expr base pointer so both Call and Submit
  // (manual-scope task launches) dispatch through the same map.
  std::unordered_set<const Expr*> consumers;
};

struct DepArrayInfo {
  VarPtr source_var;
  int64_t consumer_count = 0;
  std::unordered_set<const Expr*> consumers;
};

struct LoopBodyDepIndex {
  std::vector<const Var*> dep_array_order;
  std::unordered_map<const Var*, DepArrayInfo> dep_arrays;
  std::unordered_set<const Var*> body_defined_vars;
  std::unordered_set<const Var*> updated_arrays;
};

struct TripCountInfo {
  bool known = false;
  int64_t count = 0;
};

using ArrayAliasSet = std::unordered_set<const Var*>;
using ArrayAliasMap = std::unordered_map<const Var*, std::shared_ptr<ArrayAliasSet>>;
using NestedLoopSummaryCollector = std::function<void(const ForStmtPtr&, LoopBodyDepIndex*, bool)>;

static std::optional<int64_t> EvalConstInt(const ExprPtr& expr) {
  if (auto ci = As<ConstInt>(expr)) return ci->value_;
  return std::nullopt;
}

static TripCountInfo EvalConstTripCount(const ForStmtPtr& for_stmt) {
  auto start = EvalConstInt(for_stmt->start_);
  auto stop = EvalConstInt(for_stmt->stop_);
  auto step = EvalConstInt(for_stmt->step_);
  if (!start || !stop || !step || *step <= 0) return TripCountInfo{};
  int64_t trip = (*stop - *start + *step - 1) / *step;
  return TripCountInfo{true, trip > 0 ? trip : 0};
}

static bool IsTaskIdArrayVar(const VarPtr& var) {
  if (!var) return false;
  auto array_ty = As<ArrayType>(var->GetType());
  return array_ty && array_ty->dtype_ == DataType::TASK_ID;
}

// Returns the single manual dependency array of a Submit, or nullopt when it
// carries zero / more than one dep. Submit is the only cross-function carrier
// of manual deps (ManualDepsOnSubmitOnly invariant); the pass's own
// ``system.task_dummy`` barrier Calls keep deps in the ``manual_dep_edges``
// attr as the codegen fanin contract but are never consumers.
static std::optional<VarPtr> GetSingleManualDepArray(const ExprPtr& node) {
  auto submit = As<Submit>(node);
  if (!submit) return std::nullopt;
  if (submit->deps_.size() != 1 || !submit->deps_[0]) return std::nullopt;
  return AsVarLike(submit->deps_[0]);
}

static std::optional<VarPtr> GetSingleManualDepTaskIdArray(const ExprPtr& node) {
  auto dep = GetSingleManualDepArray(node);
  if (!dep.has_value() || !IsTaskIdArrayVar(*dep)) return std::nullopt;
  return dep;
}

static bool ShouldEmitPhaseFenceBarrier(int64_t producer_count, int64_t consumer_count) {
  if (producer_count <= 0 || consumer_count <= 0) return false;
  const int64_t estimated_saving = producer_count * consumer_count - (producer_count + consumer_count);
  return estimated_saving >= kPhaseFenceMinEstimatedEdgeSavings;
}

static void InsertWithAliases(const Var* var, const ArrayAliasMap& aliases,
                              std::unordered_set<const Var*>* out) {
  if (!var || !out) return;
  out->insert(var);
  auto it = aliases.find(var);
  if (it == aliases.end()) return;
  out->insert(it->second->begin(), it->second->end());
}

static StmtPtr MakeSeqOrStmt(std::vector<StmtPtr> stmts, const Span& span) {
  if (stmts.empty()) return std::make_shared<const SeqStmts>(std::vector<StmtPtr>{}, span);
  if (stmts.size() == 1) return stmts[0];
  return SeqStmts::Flatten(std::move(stmts), span);
}

static void MergeDepArrayInfo(LoopBodyDepIndex* dst, const Var* key, const DepArrayInfo& src,
                              int64_t consumer_multiplier = 1) {
  if (!dst || !key || consumer_multiplier <= 0) return;
  auto [it, inserted] = dst->dep_arrays.emplace(key, DepArrayInfo{});
  if (inserted) {
    dst->dep_array_order.push_back(key);
    it->second.source_var = src.source_var;
  }
  it->second.consumer_count += src.consumer_count * consumer_multiplier;
  it->second.consumers.insert(src.consumers.begin(), src.consumers.end());
}

static void MergeLoopSafetyInfo(LoopBodyDepIndex* dst, const LoopBodyDepIndex& src) {
  if (!dst) return;
  dst->body_defined_vars.insert(src.body_defined_vars.begin(), src.body_defined_vars.end());
  dst->updated_arrays.insert(src.updated_arrays.begin(), src.updated_arrays.end());
}

static LoopBodyDepIndex BuildLoopBodyDepIndex(const StmtPtr& body, const ArrayAliasMap& aliases,
                                              const NestedLoopSummaryCollector& collect_nested_loop_summary,
                                              bool collect_nested_loop_consumers = false) {
  class Collector : public IRVisitor {
   public:
    Collector(const ArrayAliasMap& aliases, const NestedLoopSummaryCollector& collect_nested_loop_summary,
              bool collect_nested_loop_consumers)
        : aliases_(aliases),
          collect_nested_loop_summary_(collect_nested_loop_summary),
          collect_nested_loop_consumers_(collect_nested_loop_consumers) {}

    LoopBodyDepIndex index;

    void AddVars(const std::vector<VarPtr>& defined_vars) {
      for (const auto& var : defined_vars) {
        if (var) index.body_defined_vars.insert(var.get());
      }
    }

    void VisitStmt_(const ForStmtPtr& for_stmt) override {
      AddVars(for_stmt->return_vars_);
      for (const auto& iter_arg : for_stmt->iter_args_) {
        if (iter_arg) index.body_defined_vars.insert(iter_arg.get());
      }
      if (collect_nested_loop_summary_) {
        collect_nested_loop_summary_(for_stmt, &index, collect_nested_loop_consumers_);
      }
    }

    void RecordDepConsumer(const ExprPtr& node) {
      auto dep = GetSingleManualDepTaskIdArray(node);
      if (!dep.has_value()) return;
      const Var* key = dep->get();
      auto [it, inserted] = index.dep_arrays.emplace(key, DepArrayInfo{});
      if (inserted) {
        index.dep_array_order.push_back(key);
        it->second.source_var = *dep;
      }
      it->second.consumer_count += consumer_multiplier_;
      it->second.consumers.insert(node.get());
    }

    void VisitStmt_(const AssignStmtPtr& assign) override {
      if (assign->var_) index.body_defined_vars.insert(assign->var_.get());
      auto call = As<Call>(assign->value_);
      if (call && call->op_->name_ == "array.update_element" && !call->args_.empty()) {
        auto base = AsVarLike(call->args_[0]);
        if (base) InsertWithAliases(base.get(), aliases_, &index.updated_arrays);
      }
      IRVisitor::VisitStmt_(assign);
    }

    void VisitStmt_(const IfStmtPtr& if_stmt) override {
      AddVars(if_stmt->return_vars_);
      IRVisitor::VisitStmt_(if_stmt);
    }

    void VisitStmt_(const WhileStmtPtr& while_stmt) override {
      AddVars(while_stmt->return_vars_);
      for (const auto& iter_arg : while_stmt->iter_args_) {
        if (iter_arg) index.body_defined_vars.insert(iter_arg.get());
      }
      IRVisitor::VisitStmt_(while_stmt);
    }

    // Only Submit can carry manual deps (ManualDepsOnSubmitOnly invariant), so
    // there is no Call consumer handler — task_dummy barrier Calls are not
    // consumers either.
    void VisitExpr_(const SubmitPtr& submit) override {
      RecordDepConsumer(submit);
      IRVisitor::VisitExpr_(submit);
    }

   private:
    const ArrayAliasMap& aliases_;
    const NestedLoopSummaryCollector& collect_nested_loop_summary_;
    const bool collect_nested_loop_consumers_;
    int64_t consumer_multiplier_ = 1;
  };

  Collector collector(aliases, collect_nested_loop_summary, collect_nested_loop_consumers);
  collector.VisitStmt(body);
  return std::move(collector.index);
}

static int64_t GetArrayProducerCount(const VarPtr& array_var) {
  auto array_ty = As<ArrayType>(array_var->GetType());
  if (!array_ty) return 0;
  if (auto ci = As<ConstInt>(array_ty->extent())) return ci->value_;
  return 0;
}

// Replace a consumer Submit's manual dep array with the single barrier TaskId
// in the typed deps_ field, preserving Submit-ness. Consumers are always
// Submits — GetSingleManualDepArray never selects another kind.
static ExprPtr RewriteManualDepsToBarrier(const ExprPtr& node, const VarPtr& barrier_var) {
  auto submit = As<Submit>(node);
  INTERNAL_CHECK(submit) << "Internal error: phase-fence consumer must be a Submit";
  return std::make_shared<Submit>(submit->op_, submit->args_, std::vector<ExprPtr>{barrier_var},
                                  submit->kwargs_, submit->attrs_, submit->GetType(), submit->span_,
                                  submit->core_num_, submit->sync_start_);
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

    std::unordered_map<const Expr*, VarPtr> consumer_to_barrier;
    for (const auto& decision : decisions) {
      for (const Expr* consumer : decision.consumers) {
        consumer_to_barrier[consumer] = decision.barrier_var;
      }
    }

    auto body_with_current_rewrites = RewriteCoveredConsumers(op->body_, consumer_to_barrier);
    auto body_with_nested = VisitStmt(body_with_current_rewrites);
    auto new_start = VisitExpr(op->start_);
    auto new_stop = VisitExpr(op->stop_);
    auto new_step = VisitExpr(op->step_);
    const bool loop_changed = body_with_nested.get() != op->body_.get() ||
                              new_start.get() != op->start_.get() || new_stop.get() != op->stop_.get() ||
                              new_step.get() != op->step_.get();
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
    if (loop_changed) {
      return std::make_shared<const ForStmt>(op->loop_var_, std::move(new_start), std::move(new_stop),
                                             std::move(new_step), op->iter_args_, std::move(body_with_nested),
                                             op->return_vars_, op->span_, op->kind_, op->chunk_config_,
                                             op->attrs_, op->leading_comments_);
    }
    return op;
  }

 private:
  void AddCachedArrayAlias(const Var* lhs, const Var* rhs) {
    if (!lhs || !rhs || lhs == rhs) return;
    auto lhs_set = GetOrCreateArrayAliasSet(lhs);
    auto rhs_set = GetOrCreateArrayAliasSet(rhs);
    if (lhs_set == rhs_set) return;
    if (lhs_set->size() < rhs_set->size()) std::swap(lhs_set, rhs_set);
    lhs_set->insert(rhs_set->begin(), rhs_set->end());
    for (const Var* var : *rhs_set) {
      array_aliases_[var] = lhs_set;
    }
  }

  std::shared_ptr<ArrayAliasSet> GetOrCreateArrayAliasSet(const Var* var) {
    auto it = array_aliases_.find(var);
    if (it != array_aliases_.end()) return it->second;
    auto alias_set = std::make_shared<ArrayAliasSet>();
    alias_set->insert(var);
    array_aliases_[var] = alias_set;
    return alias_set;
  }

  void RegisterLoopArrayAliases(const ForStmtPtr& for_stmt) {
    if (!for_stmt) return;
    for (const auto& iter_arg : for_stmt->iter_args_) {
      if (!iter_arg || !IsTaskIdArrayVar(iter_arg)) continue;
      auto init_var = AsVarLike(iter_arg->initValue_);
      if (!init_var || !IsTaskIdArrayVar(init_var)) continue;
      AddCachedArrayAlias(iter_arg.get(), init_var.get());
    }
  }

  void MergeNestedLoopSummary(const ForStmtPtr& for_stmt, LoopBodyDepIndex* index, bool include_consumers) {
    if (!for_stmt || !index) return;
    RegisterLoopArrayAliases(for_stmt);
    const auto& summary = GetLoopSummary(for_stmt);
    MergeLoopSafetyInfo(index, summary);
    if (!include_consumers) return;
    const auto trip_count = EvalConstTripCount(for_stmt);
    if (!trip_count.known || trip_count.count <= 0) return;
    for (const Var* key : summary.dep_array_order) {
      auto it = summary.dep_arrays.find(key);
      if (it == summary.dep_arrays.end()) continue;
      MergeDepArrayInfo(index, key, it->second, trip_count.count);
    }
  }

  const LoopBodyDepIndex& GetLoopSummary(const ForStmtPtr& for_stmt) {
    auto it = loop_summary_cache_.find(for_stmt.get());
    if (it != loop_summary_cache_.end()) return it->second;
    auto [inserted_it, inserted] = loop_summary_cache_.emplace(for_stmt.get(), LoopBodyDepIndex{});
    if (inserted) {
      inserted_it->second = BuildLoopBodyDepIndex(
          for_stmt->body_, array_aliases_,
          [this](const ForStmtPtr& nested_for, LoopBodyDepIndex* nested_index, bool include_consumers) {
            MergeNestedLoopSummary(nested_for, nested_index, include_consumers);
          },
          for_stmt->kind_ == ForKind::Sequential);
    }
    return inserted_it->second;
  }

  std::vector<BarrierDecision> BuildDecisions(const ForStmtPtr& for_stmt, const StmtPtr& body) {
    std::vector<BarrierDecision> decisions;
    std::unordered_set<const Var*> already_decided;
    std::unordered_set<const Var*> current_iter_args;
    RegisterLoopArrayAliases(for_stmt);
    const auto body_index = BuildLoopBodyDepIndex(
        body, array_aliases_,
        [this](const ForStmtPtr& nested_for, LoopBodyDepIndex* index, bool include_consumers) {
          MergeNestedLoopSummary(nested_for, index, include_consumers);
        },
        for_stmt->kind_ == ForKind::Sequential);

    auto try_add = [&](const VarPtr& match_var, const VarPtr& barrier_source_var, int64_t consumer_count,
                       const std::unordered_set<const Expr*>& consumers) {
      if (!match_var || !barrier_source_var || !already_decided.insert(match_var.get()).second) return;
      const int64_t producer_count = GetArrayProducerCount(match_var);
      if (!ShouldEmitPhaseFenceBarrier(producer_count, consumer_count)) return;
      BarrierDecision decision;
      decision.source = match_var.get();
      decision.source_var = barrier_source_var;
      decision.barrier_stmt =
          MakeBarrierStmt(barrier_source_var, &decision.barrier_var, for_stmt->span_, barrier_counter_++);
      decision.consumers = consumers;
      if (!decision.consumers.empty()) decisions.push_back(std::move(decision));
    };

    const bool is_parallel = for_stmt->kind_ == ForKind::Parallel;
    const auto trip_count = EvalConstTripCount(for_stmt);
    if (is_parallel && (!trip_count.known || trip_count.count <= 0)) return decisions;
    if (for_stmt->kind_ == ForKind::Sequential && trip_count.known && trip_count.count <= 0) return decisions;

    for (const auto& iter_arg : for_stmt->iter_args_) {
      if (iter_arg) current_iter_args.insert(iter_arg.get());
    }

    for (const Var* dep_array_key : body_index.dep_array_order) {
      auto info_it = body_index.dep_arrays.find(dep_array_key);
      if (info_it == body_index.dep_arrays.end()) continue;
      const auto& info = info_it->second;
      const auto& dep_array = info.source_var;
      if (!dep_array || current_iter_args.count(dep_array.get()) != 0 ||
          body_index.body_defined_vars.count(dep_array.get()) != 0 ||
          body_index.updated_arrays.count(dep_array.get()) != 0) {
        continue;
      }
      int64_t consumers = info.consumer_count;
      if (trip_count.known && trip_count.count > 0) consumers *= trip_count.count;
      try_add(dep_array, dep_array, consumers, info.consumers);
    }

    return decisions;
  }

  StmtPtr RewriteCoveredConsumers(const StmtPtr& body,
                                  const std::unordered_map<const Expr*, VarPtr>& consumer_to_barrier) {
    class Rewriter : public IRMutator {
     public:
      explicit Rewriter(const std::unordered_map<const Expr*, VarPtr>& consumer_to_barrier)
          : consumer_to_barrier_(consumer_to_barrier) {}

      ExprPtr VisitExpr_(const SubmitPtr& submit) override {
        auto it = consumer_to_barrier_.find(submit.get());
        if (it != consumer_to_barrier_.end()) {
          return RewriteManualDepsToBarrier(submit, it->second);
        }
        return IRMutator::VisitExpr_(submit);
      }

     private:
      const std::unordered_map<const Expr*, VarPtr>& consumer_to_barrier_;
    };

    if (consumer_to_barrier.empty()) return body;
    Rewriter rewriter(consumer_to_barrier);
    return rewriter.VisitStmt(body);
  }

  bool in_manual_scope_ = false;
  int64_t barrier_counter_ = 0;
  ArrayAliasMap array_aliases_;
  std::unordered_map<const ForStmt*, LoopBodyDepIndex> loop_summary_cache_;
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
  return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
}

}  // namespace

Pass ExpandManualPhaseFence() {
  return CreateProgramPass(TransformExpandManualPhaseFence, "ExpandManualPhaseFence",
                           kExpandManualPhaseFenceProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

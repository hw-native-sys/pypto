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

#include "pypto/ir/transforms/utils/nested_pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/dead_code_elimination.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/pipeline_loop_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {
// Fixed expansion/depth budgets keep specialization O(N log N), including
// nested scopes. A declined plan never leaks cloned or pinned IR to fallback.
constexpr int64_t kMaxSpecializations = 64;
constexpr int kMaxDepth = 4;

class LocalEffects : public IRVisitor {
 public:
  bool local = true;
  bool nested = false;
  bool has_call = false;
  bool bounded = true;
  bool nested_local = true;
  bool known_effects = true;
  unsigned binder_slots = 0;
  unsigned depth = 0;
  unsigned branch_depth = 0;
  void VisitStmt_(const ForStmtPtr& op) override {
    nested = true;
    binder_slots += op->return_vars_.size();
    if (++depth > kMaxDepth || binder_slots > kMaxSpecializations) {
      bounded = false;
      --depth;
      return;
    }
    IRVisitor::VisitStmt_(op);
    --depth;
  }
  void VisitStmt_(const IfStmtPtr& op) override {
    binder_slots += op->return_vars_.size();
    if (++branch_depth > 8 || binder_slots > kMaxSpecializations) {
      bounded = false;
      --branch_depth;
      return;
    }
    IRVisitor::VisitStmt_(op);
    --branch_depth;
  }
  void VisitStmt_(const WhileStmtPtr&) override { bounded = false; }
  void VisitExpr_(const SubmitPtr&) override {
    local = false;
    nested_local = false;
    has_call = true;
  }
  void VisitExpr_(const CallPtr& call) override {
    has_call = true;
    const bool registered = OpRegistry::GetInstance().IsRegistered(call->op_->name_);
    if (!registered) known_effects = false;
    if (!registered || OpRegistry::GetInstance().GetEntry(call->op_->name_).GetCrossCoreRole()) {
      local = false;
      if (depth) nested_local = false;
    } else {
      const auto& entry = OpRegistry::GetInstance().GetEntry(call->op_->name_);
      const auto evidence = entry.GetExecutionMemoryAccessEvidence();
      known_effects &= entry.HasDeclaredArgEffects() ||
                       evidence == ExecutionMemoryAccessEvidence::Functional ||
                       evidence == ExecutionMemoryAccessEvidence::NoAccess;
      const bool store = IsOp(call, "tile.store");
      if (entry.IsNoDuplicate() && !store) known_effects = false;
      for (size_t i = 0; i < call->args_.size(); ++i) {
        const auto effect = entry.GetArgEffect(i, call->kwargs_);
        if (ArgEffectWrites(effect) && !store && !(entry.IsWorkspaceArg(i) && effect == ArgEffect::Write)) {
          known_effects = false;
        }
      }
    }
    IRVisitor::VisitExpr_(call);
  }
};

class ScopeSpecializer {
 public:
  NestedPipelinePlan& plan;
  std::string& reason;
  bool valid = true;
  std::map<const Var*, const Var*> origins;
  std::unordered_set<const Var*> uses;
  int64_t phases = 0;

  ScopeSpecializer(NestedPipelinePlan& plan, std::string& reason) : plan(plan), reason(reason) {}

  StmtPtr Decline(const StmtPtr& stmt, const char* message) {
    valid = false;
    reason = message;
    return stmt;
  }

  StmtPtr Rewrite(const StmtPtr& stmt, int64_t stride = 1, int64_t phase = 0, int depth = 0,
                  int64_t storage_stride = 1, int64_t storage_phase = 0) {
    if (!valid) return stmt;
    if (auto yield = As<YieldStmt>(stmt); yield && yield->value_.empty()) {
      return SeqStmts::Flatten({}, stmt->span_);
    }
    if (auto seq = As<SeqStmts>(stmt)) {
      std::vector<StmtPtr> body;
      for (const auto& child : seq->stmts_) {
        body.push_back(Rewrite(child, stride, phase, depth, storage_stride, storage_phase));
      }
      return SeqStmts::Flatten(std::move(body), seq->span_);
    }
    if (auto branch = As<IfStmt>(stmt)) {
      auto result = MutableCopy(branch);
      result->then_body_ = Rewrite(branch->then_body_, stride, phase, depth, storage_stride, storage_phase);
      if (branch->else_body_) {
        result->else_body_ =
            Rewrite(*branch->else_body_, stride, phase, depth, storage_stride, storage_phase);
      }
      if (result->else_body_) {
        auto empty = As<SeqStmts>(*result->else_body_);
        if (empty && empty->stmts_.empty()) result->else_body_ = std::nullopt;
      }
      return result;
    }
    auto child = As<ForStmt>(stmt);
    if (!child) return stmt;
    const auto start = transform_utils::EvalConstInt(child->start_);
    const auto stop = transform_utils::EvalConstInt(child->stop_);
    const auto step = transform_utils::EvalConstInt(child->step_);
    const int slots = child->GetAttr<int>(kPipelineStagesAttr, 0);
    if (child->kind_ != ForKind::Pipeline || slots < 2 || slots > 4 || !start || !stop || !step ||
        *start < 0 || *step <= 0) {
      return Decline(stmt, "nested pipeline requires static local bounds");
    }
    for (const auto& result : child->return_vars_) {
      auto origin = origins.find(result.get());
      if (uses.count(result.get()) || (origin != origins.end() && uses.count(origin->second))) {
        return Decline(stmt, "nested pipeline result escapes its staging batch");
      }
    }
    for (const auto& arg : child->iter_args_) {
      auto scalar = As<ScalarType>(arg->GetType());
      if (!scalar || scalar->dtype_ == DataType::TASK_ID) {
        return Decline(stmt, "nested pipeline carries tensor, tile or task state");
      }
      LocalEffects init;
      init.VisitExpr(arg->initValue_);
      if (init.has_call) return Decline(stmt, "nested pipeline initializer contains a call");
    }
    const auto trips = transform_utils::ComputeStaticTripCount(*start, *stop, *step);
    if (depth >= kMaxDepth || (trips && stride > kMaxSpecializations / trips)) {
      return Decline(stmt, "nested pipeline exceeds the bounded child staging window");
    }
    LocalEffects effects;
    effects.VisitStmt(child->body_);
    if (!effects.local) return Decline(stmt, "nested pipeline contains communication or a task launch");
    plan.nested = true;
    const auto child_storage_stride = storage_stride * slots;
    plan.max_stride = std::max({plan.max_stride, stride * trips, child_storage_stride});
    std::vector<StmtPtr> expanded;
    auto carries = pipeline_loop::InitValueExprs(child->iter_args_);
    for (int64_t i = 0; i < trips; ++i) {
      if (++phases > kMaxSpecializations) return Decline(stmt, "nested pipeline exceeds phase budget");
      auto anchor = std::make_shared<Var>("pipeline_phase", child->loop_var_->GetType(), child->span_);
      plan.anchors.insert(anchor.get());
      std::unordered_map<const Var*, ExprPtr> seeds{
          {child->loop_var_.get(), pipeline_loop::MakeConstIndex(*start + i * *step, child->span_)}};
      for (size_t j = 0; j < carries.size(); ++j) seeds[child->iter_args_[j].get()] = carries[j];
      auto clone = DeepClone(child->body_, seeds);
      for (const auto& [original, fresh] : clone.var_map) {
        auto origin = origins.find(original);
        const Var* family = origin == origins.end() ? original : origin->second;
        origins[fresh.get()] = family;
        if (As<TileType>(fresh->GetType())) {
          plan.storage[fresh.get()] = {family,
                                       slots,
                                       stride * trips,
                                       phase * trips + i,
                                       anchor.get(),
                                       child_storage_stride,
                                       storage_phase * slots + i % slots};
        }
      }
      auto [body, yields] = pipeline_loop::SplitBodyYield(clone.cloned_body);
      if (yields.size() != carries.size()) return Decline(stmt, "nested pipeline has unmatched yielded data");
      for (const auto& value : yields) {
        LocalEffects yielded;
        yielded.VisitExpr(value);
        if (yielded.has_call) return Decline(stmt, "nested pipeline yield contains a call");
      }
      carries = std::move(yields);
      expanded.push_back(
          std::make_shared<AssignStmt>(anchor, pipeline_loop::MakeConstIndex(i, child->span_), child->span_));
      expanded.push_back(Rewrite(body, stride * trips, phase * trips + i, depth + 1, child_storage_stride,
                                 storage_phase * slots + i % slots));
    }
    return SeqStmts::Flatten(std::move(expanded), child->span_);
  }
};
}  // namespace

bool PrepareNestedPipeline(const ForStmtPtr& loop, NestedPipelinePlan& plan, std::string& reason,
                           bool allow_root_transfer) {
  plan.loop = loop;
  LocalEffects scan;
  scan.VisitStmt(loop->body_);
  if (!scan.nested) return true;
  if (!scan.bounded || !scan.known_effects || !scan.nested_local || (!allow_root_transfer && !scan.local)) {
    reason = "nested control flow exceeds proof budget or contains nonlocal effects";
    return false;
  }
  auto body = As<SeqStmts>(loop->body_);
  // Reuse loop-state repair for Python temporaries carried only through yields.
  // Preserve real recurrence uses and function outputs; those are rejected below.
  auto cleaned = loop_repair::StripDeadIterArgs(body ? body->stmts_ : std::vector<StmtPtr>{loop->body_});
  cleaned = dce::EliminateDeadYieldSlots(cleaned);
  cleaned = dce::EliminateDeadCode(cleaned);
  ScopeSpecializer specializer(plan, reason);
  var_collectors::VarDefUseCollector refs;
  refs.VisitStmt(loop->body_);
  specializer.uses = std::move(refs.var_uses);
  auto rewritten = specializer.Rewrite(SeqStmts::Flatten(std::move(cleaned), loop->span_));
  if (!specializer.valid) return false;
  auto seq = As<SeqStmts>(rewritten);
  auto live = dce::EliminateDeadYieldSlots(seq ? seq->stmts_ : std::vector<StmtPtr>{rewritten});
  live = dce::EliminateDeadCode(live, {plan.anchors.begin(), plan.anchors.end()});
  rewritten = specializer.Rewrite(SeqStmts::Flatten(std::move(live), loop->span_));
  auto result = MutableCopy(loop);
  result->body_ = rewritten;
  plan.loop = result;
  return true;
}
}  // namespace pypto::ir

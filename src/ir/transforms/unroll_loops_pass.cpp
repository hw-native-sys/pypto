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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/pipeline_loop_utils.h"
#include "pypto/ir/transforms/utils/unroll_loops.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Maximum number of iterations allowed for compile-time unrolling.
/// Prevents excessive memory/CPU usage from large trip counts.
constexpr int64_t kMaxUnrollIterations = 1024;

/// Throwing const-int accessor, prefixed with this pass's name. Shared with the
/// pipeline loop passes: see `utils/pipeline_loop_utils.h`.
int64_t GetConstIntValue(const ExprPtr& expr, const std::string& what) {
  return pipeline_loop::GetConstIntValue(expr, "UnrollLoops", what);
}

// A pre-SSA name has the type of its first definition. When an assignment in
// either arm selects a slot with an unrolled index, SSA needs a shared name
// independent of that slot and each definition must retain its own storage.
class BranchSlotIndex : public IRVisitor {
 public:
  std::unordered_set<const Var*> Find(const StmtPtr& body) {
    VisitStmt(body);
    std::unordered_set<const Var*> result;
    if (indices_.empty() || cross_branch_.empty()) return result;
    for (const auto& [slot, vars] : slot_vars_) {
      var_collectors::VarDefUseCollector refs;
      refs.VisitExpr(slot);
      for (const auto* ref : refs.var_uses) {
        if (indices_.count(ref)) {
          for (const auto* var : vars) {
            if (cross_branch_.count(var)) result.insert(var);
          }
          break;
        }
      }
    }
    return result;
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    VisitExpr(op->value_);
    const auto tile = As<TileType>(op->var_->GetType());
    if (!tile || !tile->memref_ || !(*tile->memref_)->is_pinned_) return;

    const auto* var = op->var_.get();
    const auto first = first_arm_.emplace(var, arm_).first;
    if (first->second != arm_) cross_branch_.insert(var);
    // A fixed first declaration still establishes the defining arm. Later explicit
    // annotations live on the RHS call or on the temporary for an annotated alias.
    RecordSlot(var, *tile->memref_);
    if (auto rhs = GetTypeMemRef(op->value_->GetType()); rhs && (*rhs)->is_pinned_) {
      RecordSlot(var, *rhs);
    }
  }

  void VisitExpr_(const VarPtr& op) override {
    const auto first = first_arm_.find(op.get());
    // A read dominated by its declaration still sees that arm's value.
    if (first != first_arm_.end() && first->second && !active_arms_.count(first->second)) {
      cross_branch_.insert(op.get());
    }
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    VisitExpr(op->condition_);
    VisitArm(op->then_body_);
    if (op->else_body_) VisitArm(*op->else_body_);
    for (const auto& var : op->return_vars_) VisitExpr(var);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (op->kind_ == ForKind::Unroll) indices_.insert(op->loop_var_.get());
    IRVisitor::VisitStmt_(op);
  }

 private:
  void VisitArm(const StmtPtr& body) {
    auto outer = arm_;
    arm_ = body.get();
    active_arms_.insert(arm_);
    VisitStmt(body);
    active_arms_.erase(arm_);
    arm_ = outer;
  }

  void RecordSlot(const Var* var, const MemRefPtr& memref) {
    if (memref->slot_index_ && *memref->slot_index_ && !As<ConstInt>(*memref->slot_index_)) {
      slot_vars_[*memref->slot_index_].insert(var);
    }
  }

  const Stmt* arm_{nullptr};
  std::unordered_set<const Stmt*> active_arms_;
  std::unordered_set<const Var*> indices_;
  std::unordered_map<const Var*, const Stmt*> first_arm_;
  std::unordered_set<const Var*> cross_branch_;
  std::unordered_map<ExprPtr, std::unordered_set<const Var*>> slot_vars_;
};

class BranchSlotMutator : public IRMutator {
 public:
  explicit BranchSlotMutator(std::unordered_set<const Var*> vars) : vars_(std::move(vars)) {}

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto stmt = IRMutator::VisitStmt_(op);
    if (!vars_.erase(op->var_.get())) return stmt;
    auto decl = As<AssignStmt>(stmt);
    auto var = std::make_shared<Var>(
        decl->var_->name_hint_, CloneTypeWithMemRef(decl->var_->GetType(), std::nullopt), decl->var_->span_);
    var_remap_[op->var_.get()] = var;
    // Keep the declared storage on the producer, even for an annotated alias.
    // All paths assign the same storage-neutral Var so ConvertToSSA can form a phi.
    if (As<Call>(decl->value_) &&
        GetTypeMemRef(decl->value_->GetType()) == GetTypeMemRef(decl->var_->GetType())) {
      auto binding = MutableCopy(decl);
      binding->var_ = var;
      return binding;
    }
    auto bind = std::make_shared<AssignStmt>(var, decl->var_, op->span_);
    return SeqStmts::Flatten({decl, bind}, op->span_);
  }

 private:
  std::unordered_set<const Var*> vars_;
};

/**
 * @brief Mutator that expands ForStmt nodes with ForKind::Unroll into
 * a SeqStmts of deep-cloned bodies, substituting the loop variable with each
 * iteration's constant value.
 *
 * With shared Var pointers (the parser reuses the same Var object for
 * reassignments of the same source-level variable), we use DeepClone with
 * clone_def_vars=false so reassignments keep their shared identity unless
 * their declared type depends on the iteration. ConvertToSSA versions them
 * after unrolling.
 */
class LoopUnrollMutator : public IRMutator {
 public:
  explicit LoopUnrollMutator(const std::unordered_set<const ForStmt*>* ancestors = nullptr)
      : ancestors_(ancestors) {}
  /// Unroll a single ForStmt with ForKind::Unroll.
  StmtPtr UnrollForStmt(const ForStmtPtr& op) {
    // Validate: no iter_args for unroll loops
    INTERNAL_CHECK_SPAN(op->iter_args_.empty(), op->span_)
        << "Unroll loops cannot have iter_args (init_values)";

    // Extract compile-time constants for start/stop/step
    int64_t start = GetConstIntValue(op->start_, "start");
    int64_t stop = GetConstIntValue(op->stop_, "stop");
    int64_t step = GetConstIntValue(op->step_, "step");
    if (step == 0) {
      throw pypto::ValueError("Unroll loop step cannot be zero");
    }

    // Compute trip count and enforce max unroll limit
    int64_t trip_count = 0;
    if (step > 0 && start < stop) {
      trip_count = (stop - start + step - 1) / step;
    } else if (step < 0 && start > stop) {
      trip_count = (start - stop + (-step) - 1) / (-step);
    }
    if (trip_count > kMaxUnrollIterations) {
      throw pypto::ValueError("Unroll loop trip count " + std::to_string(trip_count) +
                              " exceeds maximum allowed (" + std::to_string(kMaxUnrollIterations) +
                              "). Reduce the loop range or use pl.range() instead");
    }

    std::vector<StmtPtr> unrolled;

    auto emit_iteration = [&](int64_t i) {
      auto const_expr = std::make_shared<ConstInt>(i, DataType::INDEX, op->loop_var_->span_);
      std::unordered_map<const Var*, ExprPtr> sub_map;
      sub_map[op->loop_var_.get()] = const_expr;
      auto [cloned_body, def_var_map] = DeepClone(op->body_, sub_map, /*clone_def_vars=*/false);
      unrolled.push_back(VisitStmt(cloned_body));
      // A tile defined in a plain loop can be read after it. Remember its
      // latest binding, including any further specialization by a nested loop.
      for (const auto& [original, cloned] : def_var_map) {
        if (original->GetKind() == ObjectKind::Var) {
          latest_defs_[original->UniqueId()] = VisitExpr(cloned);
        }
      }
    };

    if (step > 0) {
      for (int64_t i = start; i < stop; i += step) emit_iteration(i);
    } else {
      for (int64_t i = start; i > stop; i += step) emit_iteration(i);
    }

    if (unrolled.empty()) {
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
    }
    // The ForStmt is gone. Its leading_comments would otherwise vanish with it,
    // so move them onto the first unrolled stmt so readers still see them above
    // the expanded iterations. SeqStmts itself cannot carry comments (invariant).
    if (!op->leading_comments_.empty()) {
      AttachCommentsToFirstStmt(unrolled, op->leading_comments_);
    }
    return SeqStmts::Flatten(std::move(unrolled), op->span_);
  }

  // Prepend ``comments`` to the leading_comments of the first non-SeqStmts stmt
  // reachable from ``stmts``. Used when replacing a commented compound stmt with
  // a sequence that isn't itself allowed to carry comments.
  static void AttachCommentsToFirstStmt(const std::vector<StmtPtr>& stmts,
                                        const std::vector<std::string>& comments) {
    StmtPtr first;
    for (const auto& s : stmts) {
      if (s) {
        first = s;
        break;
      }
    }
    if (!first) return;
    // Walk into nested SeqStmts to find the first terminal stmt.
    while (auto seq = std::dynamic_pointer_cast<const SeqStmts>(first)) {
      if (seq->stmts_.empty()) return;
      first = seq->stmts_[0];
    }
    std::vector<std::string> merged = comments;
    merged.insert(merged.end(), first->leading_comments_.begin(), first->leading_comments_.end());
    AttachLeadingComments(first, std::move(merged));
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (op->kind_ != ForKind::Unroll || (ancestors_ && !ancestors_->count(op.get()))) {
      return IRMutator::VisitStmt_(op);
    }
    CHECK_SPAN(op->hardware_loop_policy_ == HardwareLoopPolicy::Auto, op->span_)
        << "UnrollLoops cannot erase HardwareLoopPolicy.Require; hardware loop qualification is not "
           "supported yet";
    // Cloned descendants have new statement identities. Once an enclosing
    // loop is selected, perform ordinary recursive substitution in its body.
    auto saved = ancestors_;
    ancestors_ = nullptr;
    auto result = UnrollForStmt(op);
    ancestors_ = saved;
    return result;
  }

 private:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = latest_defs_.find(op->UniqueId());
    if (it != latest_defs_.end()) return it->second;
    return IRMutator::VisitExpr_(op);
  }

  // Nested unrolling can replace intermediate clones and release their pointers.
  std::unordered_map<uint64_t, ExprPtr> latest_defs_;
  const std::unordered_set<const ForStmt*>* ancestors_;
};

}  // namespace

namespace transform_utils {
StmtPtr PrepareUnrolledBranchBindings(const StmtPtr& body) {
  auto shared = BranchSlotIndex().Find(body);
  return shared.empty() ? body : BranchSlotMutator(std::move(shared)).VisitStmt(body);
}

FunctionPtr UnrollFunction(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "UnrollLoops cannot run on null function";

  auto new_body = LoopUnrollMutator().VisitStmt(PrepareUnrolledBranchBindings(func->body_));

  if (new_body.get() == func->body_.get()) {
    return func;  // No changes
  }

  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

FunctionPtr UnrollRegScopeAncestors(const FunctionPtr& func) {
  class Ancestors : public IRVisitor {
   public:
    std::unordered_set<const ForStmt*> loops;

   protected:
    void VisitStmt_(const RegScopeStmtPtr&) override { contains_scope_ = true; }
    void VisitStmt_(const ForStmtPtr& loop) override {
      bool previous = contains_scope_;
      contains_scope_ = false;
      IRVisitor::VisitStmt_(loop);
      if (contains_scope_ && loop->kind_ == ForKind::Unroll) loops.insert(loop.get());
      contains_scope_ = contains_scope_ || previous;
    }

   private:
    bool contains_scope_ = false;
  };
  Ancestors ancestors;
  auto body = PrepareUnrolledBranchBindings(func->body_);
  ancestors.VisitStmt(body);
  if (ancestors.loops.empty()) return func;
  auto result = MutableCopy(func);
  result->body_ = LoopUnrollMutator(&ancestors.loops).VisitStmt(body);
  return result;
}

}  // namespace transform_utils

// Factory function
namespace pass {
Pass UnrollLoops() {
  return CreateFunctionPass(transform_utils::UnrollFunction, "UnrollLoops", kUnrollLoopsProperties);
}
}  // namespace pass

}  // namespace ir
}  // namespace pypto

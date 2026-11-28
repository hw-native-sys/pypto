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

#include "pypto/ir/transforms/utils/scalar_offset_facts.h"

#include <algorithm>
#include <cstddef>
#include <string>

#include "pypto/core/logging.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/visitor.h"

namespace pypto::ir::scalar_offset_facts {

class ScalarOffsetFactStore::Collector : public IRVisitor {
 public:
  Collector(ScalarOffsetFactStore& store, const ProgramPtr& program) : store_(store), program_(program) {}

 private:
  // Before SSA, one Var can have several bindings. Discard every fact for it
  // on the second binding and never let a later binding restore a fact.
  bool RecordBinding(const VarPtr& var) {
    if (seen_bindings_.insert(var).second) return true;
    store_.definitions_.erase(var);
    store_.non_negative_vars_.erase(var);
    store_.loop_bindings_.erase(var);
    store_.unknown_params_.insert(var);
    return false;
  }

  void RecordScalarBindings(const std::vector<VarPtr>& vars) {
    for (const auto& var : vars) {
      if (var && As<ScalarType>(var->GetType())) RecordBinding(var);
    }
  }

  void VisitFunction(const FunctionPtr& func) override {
    RecordScalarBindings(func->params_);
    IRVisitor::VisitFunction(func);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op->var_ && As<ScalarType>(op->var_->GetType()) && RecordBinding(op->var_)) {
      store_.definitions_[op->var_] = op->value_;
      auto call = As<Call>(op->value_);
      if (call && (IsOp(call, "tile.get_block_idx") || IsOp(call, "tile.get_block_num") ||
                   IsOp(call, "tensor.get_block_idx") || IsOp(call, "tensor.get_block_num"))) {
        store_.non_negative_vars_.insert(op->var_);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  // Control-flow results bind fresh values even when pre-SSA IR reuses a Var.
  void VisitStmt_(const IfStmtPtr& op) override {
    RecordScalarBindings(op->return_vars_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (op->loop_var_ && RecordBinding(op->loop_var_) && op->start_ && op->step_) {
      store_.loop_bindings_[op->loop_var_] = {op->start_, op->step_};
    }
    RecordScalarBindings(op->return_vars_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    RecordScalarBindings(op->return_vars_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    RecordCall(op->op_, op->args_, false);
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    RecordCall(op->op_, op->args_, true);
    IRVisitor::VisitExpr_(op);
  }

  void RecordSource(const VarPtr& param, const ExprPtr& source) {
    if (!param || !As<ScalarType>(param->GetType())) return;
    if (!source) {
      store_.unknown_params_.insert(param);
    } else if (seen_sources_[param].insert(source.get()).second) {
      store_.parameter_sources_[param].push_back(source);
    }
  }

  struct CalleeSources {
    FunctionPtr function;
    size_t ctx_count = 0;
    size_t shortest_prefix = 0;
    bool malformed = false;
  };

  void RecordCall(const OpPtr& op, const std::vector<ExprPtr>& args, bool is_submit) {
    auto global = As<GlobalVar>(op);
    if (!global) return;
    auto [it, inserted] = callees_.try_emplace(global->name_);
    auto& callee = it->second;
    if (inserted) {
      callee.function = program_->GetFunction(global->name_);
      if (callee.function) {
        const auto& params = callee.function->params_;
        for (auto p = params.rbegin(); p != params.rend() && *p && IsA<CommCtxType>((*p)->GetType()); ++p) {
          ++callee.ctx_count;
        }
        callee.shortest_prefix = params.size() - callee.ctx_count;
      }
    }
    if (!callee.function || callee.malformed) return;
    const auto& params = callee.function->params_;
    if (args.size() > params.size() || (!is_submit && args.size() != params.size()) ||
        args.size() < callee.ctx_count) {
      for (const auto& param : params) RecordSource(param, nullptr);
      callee.malformed = true;
      return;
    }

    // A Submit may omit runtime-allocated outputs just before its CommCtx suffix.
    // Across all sites, visit each omitted parameter at most once, even for
    // repeated launches with a large output gap. For Call the gap is zero.
    const size_t prefix = args.size() - callee.ctx_count;
    const size_t gap = params.size() - args.size();
    for (size_t i = prefix; i < callee.shortest_prefix; ++i) RecordSource(params[i], nullptr);
    callee.shortest_prefix = std::min(callee.shortest_prefix, prefix);
    for (size_t i = 0; i < prefix; ++i) RecordSource(params[i], args[i]);
    for (size_t i = prefix; i < args.size(); ++i) RecordSource(params[i + gap], args[i]);
  }

  ScalarOffsetFactStore& store_;
  const ProgramPtr& program_;
  std::unordered_map<std::string, CalleeSources> callees_;
  std::unordered_set<VarPtr> seen_bindings_;
  std::unordered_map<VarPtr, std::unordered_set<const Expr*>> seen_sources_;
};

ScalarOffsetFactStore::ScalarOffsetFactStore(ProgramPtr program) {
  if (program) Collector(*this, program).VisitProgram(program);
}

ScalarOffsetFacts ScalarOffsetFactStore::Facts() const {
  ScalarOffsetFacts facts;
  facts.definition = [this](const VarPtr& var) -> ExprPtr {
    auto it = definitions_.find(var);
    return it == definitions_.end() ? nullptr : it->second;
  };
  facts.loop_range = [this](const VarPtr& var) {
    auto it = loop_bindings_.find(var);
    return it == loop_bindings_.end() ? std::pair<ExprPtr, ExprPtr>{} : it->second;
  };
  facts.is_non_negative = [this](const VarPtr& var) { return non_negative_vars_.count(var) != 0; };
  facts.parameter_sources = [this](const VarPtr& var) -> const std::vector<ExprPtr>* {
    auto it = parameter_sources_.find(var);
    return it == parameter_sources_.end() || unknown_params_.count(var) ? nullptr : &it->second;
  };
  return facts;
}

namespace {

bool IsPowerOfTwo(int64_t value) { return value > 0 && (value & (value - 1)) == 0; }

bool IsPositiveDivisor(const ConstIntPtr& divisor, const ExprPtr& expr) {
  return divisor && divisor->value_ > 0 &&
         arith::detail::ConstIntFitsDtype(divisor->value_, GetScalarDtype(expr));
}

// A proof owns its recursion stack and shares one node budget across all
// branches, including diamond-shaped definition chains and multiple callers.
class OffsetProof {
 public:
  OffsetProof(const ScalarOffsetFacts& facts, int* budget) : facts_(facts), budget_(*budget) {}

  bool MultipleOf(const ExprPtr& expr, int64_t divisor) {
    if (!expr || --budget_ < 0) return false;
    if (auto c = As<ConstInt>(expr)) return c->value_ % divisor == 0;
    auto prove = [&](const ExprPtr& e) { return MultipleOf(e, divisor); };
    // Exact Var deliberately excludes loop-carried IterArgs, whose incoming
    // value says nothing about later iterations.
    if (auto var = As<Var>(expr)) return ProveBinding(var, prove);
    auto bin = As<BinaryExpr>(expr);
    if (!bin) return false;
    switch (bin->GetKind()) {
      case ObjectKind::Mul:
        return prove(bin->left_) || prove(bin->right_);
      case ObjectKind::Add:
      case ObjectKind::Sub:
      case ObjectKind::Min:
      case ObjectKind::Max:
        return prove(bin->left_) && prove(bin->right_);
      case ObjectKind::FloorDiv: {
        auto denom = As<ConstInt>(bin->right_);
        int64_t combined;
        return IsPositiveDivisor(denom, bin) && IsPowerOfTwo(denom->value_) &&
               !__builtin_mul_overflow(denom->value_, divisor, &combined) && MultipleOf(bin->left_, combined);
      }
      default:
        return false;
    }
  }

  bool NonNegative(const ExprPtr& expr) {
    if (!expr || --budget_ < 0) return false;
    if (auto c = As<ConstInt>(expr)) return c->value_ >= 0;
    if (auto var = As<Var>(expr)) {
      return (facts_.is_non_negative && facts_.is_non_negative(var)) ||
             ProveBinding(var, [this](const ExprPtr& e) { return NonNegative(e); });
    }
    auto bin = As<BinaryExpr>(expr);
    if (!bin) return false;
    switch (bin->GetKind()) {
      case ObjectKind::Add:
      case ObjectKind::Mul:
      case ObjectKind::Min:
        return NonNegative(bin->left_) && NonNegative(bin->right_);
      case ObjectKind::Max:
        return NonNegative(bin->left_) || NonNegative(bin->right_);
      case ObjectKind::FloorDiv:
      case ObjectKind::FloorMod:
        // Keep the conservative dividend check even for a positive modulus.
        return IsPositiveDivisor(As<ConstInt>(bin->right_), bin) && NonNegative(bin->left_);
      default:
        // In particular a difference of non-negative operands can be negative.
        return false;
    }
  }

 private:
  template <typename Predicate>
  bool ProveBinding(const VarPtr& var, Predicate prove) {
    if (facts_.loop_range) {
      auto [start, step] = facts_.loop_range(var);
      if (start && step && prove(start) && prove(step)) return true;
    }
    if (facts_.definition) {
      if (auto def = facts_.definition(var)) return prove(def);
    }
    auto sources = facts_.parameter_sources ? facts_.parameter_sources(var) : nullptr;
    if (!sources || sources->empty() || !active_params_.insert(var.get()).second) return false;
    const bool proven = std::all_of(sources->begin(), sources->end(), prove);
    active_params_.erase(var.get());
    return proven;
  }

  const ScalarOffsetFacts& facts_;
  int& budget_;
  std::unordered_set<const Var*> active_params_;
};

}  // namespace

bool IsProvableMultipleOf(const ExprPtr& expr, int64_t divisor, const ScalarOffsetFacts& facts, int* budget) {
  INTERNAL_CHECK(budget != nullptr) << "Internal error: offset alignment proof requires a step budget";
  INTERNAL_CHECK(IsPowerOfTwo(divisor))
      << "Internal error: offset alignment requires a power-of-two divisor, got " << divisor;
  return OffsetProof(facts, budget).MultipleOf(expr, divisor);
}

bool IsProvableNonNegative(const ExprPtr& expr, const ScalarOffsetFacts& facts, int* budget) {
  INTERNAL_CHECK(budget != nullptr) << "Internal error: offset sign proof requires a step budget";
  return OffsetProof(facts, budget).NonNegative(expr);
}

}  // namespace pypto::ir::scalar_offset_facts

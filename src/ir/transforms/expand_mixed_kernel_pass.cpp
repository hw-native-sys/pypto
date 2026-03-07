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
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// ============================================================================
// Step 2: IR Node Coloring
// ============================================================================

enum class NodeColor : uint8_t { WHITE = 0, GREEN = 1, RED = 2 };

inline bool IsAICOperation(const std::string& op_name) {
  return op_name == "tensor.matmul";
}

inline bool IsAIVOperation(const std::string& op_name) {
  static const std::unordered_set<std::string> aiv_ops = {
      "tensor.view",         "tensor.read",         "tensor.create",
      "tensor.assemble",     "tensor.mul",          "tensor.sub",
      "tensor.div",          "tensor.add",          "tensor.exp",
      "tensor.row_max",      "tensor.row_sum",      "tensor.cast",
      "tensor.reshape",      "tensor.maximum",      "tensor.minimum",
      "tensor.deep_reshape", "tensor.deep_view",
  };
  return aiv_ops.count(op_name) > 0;
}

class ColorAnalyzer : public IRVisitor {
 public:
  std::unordered_map<std::string, NodeColor> var_color;
  std::unordered_map<std::string, std::unordered_set<NodeColor>> var_consumers;
  bool has_red = false;
  bool has_green = false;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto call = As<Call>(op->value_)) {
      auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
      if (opnode) {
        NodeColor color = NodeColor::WHITE;
        bool is_scalar_output =
            std::dynamic_pointer_cast<const ScalarType>(op->var_->GetType()) != nullptr;

        if (IsAICOperation(opnode->name_)) {
          color = NodeColor::RED;
          has_red = true;
        } else if (IsAIVOperation(opnode->name_) && !is_scalar_output) {
          color = NodeColor::GREEN;
          has_green = true;
        }
        var_color[op->var_->name_] = color;

        for (const auto& arg : call->args_) {
          if (auto var = As<Var>(arg)) {
            var_consumers[var->name_].insert(color);
          }
        }
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (auto call = As<Call>(op->expr_)) {
      auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
      if (opnode && IsAIVOperation(opnode->name_)) {
        has_green = true;
        for (const auto& arg : call->args_) {
          if (auto var = As<Var>(arg)) {
            var_consumers[var->name_].insert(NodeColor::GREEN);
          }
        }
      }
    }
    IRVisitor::VisitStmt_(op);
  }
};

// ============================================================================
// Step 3: Identify cross-color boundaries
// ============================================================================

struct CrossColorBoundary {
  std::string var_name;
  NodeColor producer_color;
  NodeColor consumer_color;
};

std::vector<CrossColorBoundary> FindCrossColorBoundaries(const ColorAnalyzer& analyzer) {
  std::vector<CrossColorBoundary> boundaries;
  for (const auto& [var_name, consumers] : analyzer.var_consumers) {
    auto prod_it = analyzer.var_color.find(var_name);
    if (prod_it == analyzer.var_color.end()) continue;
    NodeColor prod_color = prod_it->second;
    if (prod_color == NodeColor::WHITE) continue;

    for (NodeColor cons_color : consumers) {
      if (cons_color == NodeColor::WHITE) continue;
      if (prod_color != cons_color) {
        boundaries.push_back({var_name, prod_color, cons_color});
      }
    }
  }
  return boundaries;
}

// ============================================================================
// Steps 4-7: Kernel Mutators (color pruning + tpush/tpop insertion)
// ============================================================================

class KernelMutatorBase : public IRMutator {
 protected:
  enum class Action { KEEP, REMOVE, REPLACE };

  struct AssignAction {
    Action action;
    StmtPtr replacement;
  };

  virtual AssignAction ClassifyAssign(const AssignStmtPtr& op) = 0;
  virtual bool ShouldRemoveEval(const EvalStmtPtr& op) = 0;

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto result = ClassifyAssign(op);
    switch (result.action) {
      case Action::KEEP:
        return IRMutator::VisitStmt_(op);
      case Action::REMOVE:
        removed_ = true;
        return nullptr;
      case Action::REPLACE:
        removed_ = true;
        return result.replacement;
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    if (ShouldRemoveEval(op)) {
      removed_ = true;
      return nullptr;
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    return FilterStmtList(op->stmts_, op->span_, /*is_seq=*/true);
  }

  StmtPtr VisitStmt_(const OpStmtsPtr& op) override {
    return FilterStmtList(op->stmts_, op->span_, /*is_seq=*/false);
  }

 private:
  bool removed_ = false;

  StmtPtr FilterStmtList(const std::vector<StmtPtr>& stmts, const Span& span, bool is_seq) {
    std::vector<StmtPtr> new_stmts;
    bool changed = false;
    for (const auto& stmt : stmts) {
      bool prev_removed = removed_;
      removed_ = false;
      auto new_stmt = StmtFunctor<StmtPtr>::VisitStmt(stmt);
      bool was_removed = removed_;
      removed_ = prev_removed;

      if (new_stmt == nullptr) {
        changed = true;
        continue;
      }
      if (was_removed || new_stmt.get() != stmt.get()) changed = true;

      if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(new_stmt)) {
        for (const auto& s : seq->stmts_) new_stmts.push_back(s);
        changed = true;
      } else {
        new_stmts.push_back(new_stmt);
      }
    }
    if (!changed) {
      if (is_seq) return std::make_shared<SeqStmts>(stmts, span);
      return std::make_shared<OpStmts>(stmts, span);
    }
    if (is_seq) return std::make_shared<SeqStmts>(new_stmts, span);
    return std::make_shared<OpStmts>(new_stmts, span);
  }
};

class AICKernelMutator : public KernelMutatorBase {
 public:
  AICKernelMutator(const ColorAnalyzer& /*analyzer*/,
                   const std::vector<CrossColorBoundary>& boundaries) {
    for (const auto& b : boundaries) {
      if (b.producer_color == NodeColor::GREEN && b.consumer_color == NodeColor::RED) {
        green_to_red_vars_.insert(b.var_name);
      }
      if (b.producer_color == NodeColor::RED && b.consumer_color == NodeColor::GREEN) {
        red_to_green_vars_.insert(b.var_name);
      }
    }
  }

 protected:
  AssignAction ClassifyAssign(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call) return {Action::KEEP, nullptr};

    auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
    if (!opnode) return {Action::KEEP, nullptr};

    if (IsAIVOperation(opnode->name_)) {
      if (std::dynamic_pointer_cast<const ScalarType>(op->var_->GetType())) {
        return {Action::KEEP, nullptr};
      }
      if (green_to_red_vars_.count(op->var_->name_)) {
        auto tpop_op = std::make_shared<Op>("comm.tpop_from_aiv");
        auto tpop_call =
            std::make_shared<Call>(tpop_op, std::vector<ExprPtr>{}, op->var_->GetType(), op->span_);
        return {Action::REPLACE, std::make_shared<AssignStmt>(op->var_, tpop_call, op->span_)};
      }
      return {Action::REMOVE, nullptr};
    }

    if (IsAICOperation(opnode->name_)) {
      if (red_to_green_vars_.count(op->var_->name_)) {
        auto orig = IRMutator::VisitStmt_(op);
        auto tpush_op = std::make_shared<Op>("comm.tpush_to_aiv");
        auto tpush_call = std::make_shared<Call>(tpush_op, std::vector<ExprPtr>{op->var_},
                                                 op->var_->GetType(), op->span_);
        auto tpush_stmt = std::make_shared<EvalStmt>(tpush_call, op->span_);
        return {Action::REPLACE,
                std::make_shared<SeqStmts>(std::vector<StmtPtr>{orig, tpush_stmt}, op->span_)};
      }
    }

    return {Action::KEEP, nullptr};
  }

  bool ShouldRemoveEval(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (!call) return false;
    auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
    if (!opnode) return false;
    return IsAIVOperation(opnode->name_);
  }

 private:
  std::unordered_set<std::string> green_to_red_vars_;
  std::unordered_set<std::string> red_to_green_vars_;
};

class AIVKernelMutator : public KernelMutatorBase {
 public:
  AIVKernelMutator(const ColorAnalyzer& /*analyzer*/,
                   const std::vector<CrossColorBoundary>& boundaries) {
    for (const auto& b : boundaries) {
      if (b.producer_color == NodeColor::RED && b.consumer_color == NodeColor::GREEN) {
        red_to_green_vars_.insert(b.var_name);
      }
      if (b.producer_color == NodeColor::GREEN && b.consumer_color == NodeColor::RED) {
        green_to_red_vars_.insert(b.var_name);
      }
    }
  }

 protected:
  AssignAction ClassifyAssign(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call) return {Action::KEEP, nullptr};

    auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
    if (!opnode) return {Action::KEEP, nullptr};

    if (IsAICOperation(opnode->name_)) {
      if (std::dynamic_pointer_cast<const ScalarType>(op->var_->GetType())) {
        return {Action::KEEP, nullptr};
      }
      if (red_to_green_vars_.count(op->var_->name_)) {
        auto tpop_op = std::make_shared<Op>("comm.tpop_from_aic");
        auto tpop_call =
            std::make_shared<Call>(tpop_op, std::vector<ExprPtr>{}, op->var_->GetType(), op->span_);
        return {Action::REPLACE, std::make_shared<AssignStmt>(op->var_, tpop_call, op->span_)};
      }
      return {Action::REMOVE, nullptr};
    }

    if (IsAIVOperation(opnode->name_)) {
      if (green_to_red_vars_.count(op->var_->name_)) {
        auto orig = IRMutator::VisitStmt_(op);
        auto tpush_op = std::make_shared<Op>("comm.tpush_to_aic");
        auto tpush_call = std::make_shared<Call>(tpush_op, std::vector<ExprPtr>{op->var_},
                                                 op->var_->GetType(), op->span_);
        auto tpush_stmt = std::make_shared<EvalStmt>(tpush_call, op->span_);
        return {Action::REPLACE,
                std::make_shared<SeqStmts>(std::vector<StmtPtr>{orig, tpush_stmt}, op->span_)};
      }
    }

    return {Action::KEEP, nullptr};
  }

  bool ShouldRemoveEval(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (!call) return false;
    auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
    if (!opnode) return false;
    return IsAICOperation(opnode->name_);
  }

 private:
  std::unordered_set<std::string> red_to_green_vars_;
  std::unordered_set<std::string> green_to_red_vars_;
};

// ============================================================================
// Step 5: Insert pipe initialization
// ============================================================================

StmtPtr PrependPipeInit(const StmtPtr& body, const std::string& init_op_name, const Span& span) {
  auto init_op = std::make_shared<Op>(init_op_name);
  auto init_call = std::make_shared<Call>(init_op, std::vector<ExprPtr>{}, span);
  auto init_stmt = std::make_shared<EvalStmt>(init_call, span);

  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(body)) {
    auto new_stmts = seq->stmts_;
    new_stmts.insert(new_stmts.begin(), init_stmt);
    return std::make_shared<SeqStmts>(new_stmts, span);
  }
  return std::make_shared<SeqStmts>(std::vector<StmtPtr>{init_stmt, body}, span);
}

// ============================================================================
// Step 8: Dead Code Elimination (comprehensive)
// ============================================================================

class VarDefCollector : public IRVisitor {
 public:
  std::unordered_set<std::string> defined_vars;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    defined_vars.insert(op->var_->name_);
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForStmtPtr& op) override {
    defined_vars.insert(op->loop_var_->name_);
    for (const auto& ia : op->iter_args_) defined_vars.insert(ia->name_);
    for (const auto& rv : op->return_vars_) defined_vars.insert(rv->name_);
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& rv : op->return_vars_) defined_vars.insert(rv->name_);
    IRVisitor::VisitStmt_(op);
  }
};

class VarRefCollectorSimple : public IRVisitor {
 public:
  std::unordered_set<std::string> referenced_vars;

 protected:
  void VisitExpr_(const VarPtr& op) override { referenced_vars.insert(op->name_); }
  void VisitExpr_(const IterArgPtr& op) override {
    referenced_vars.insert(op->name_);
    IRVisitor::VisitExpr_(op);
  }
};

inline bool IsSideEffectingOp(const ExprPtr& expr) {
  auto call = As<Call>(expr);
  if (!call) return false;
  auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!opnode) return false;
  return opnode->name_.find("comm.") == 0 || opnode->name_ == "tensor.assemble";
}

inline bool HasUndefinedRef(const ExprPtr& expr, const std::unordered_set<std::string>& defined) {
  VarRefCollectorSimple collector;
  collector.VisitExpr(expr);
  for (const auto& ref : collector.referenced_vars) {
    if (!defined.count(ref)) return true;
  }
  return false;
}

inline bool IsEffectivelyEmpty(const StmtPtr& stmt) {
  if (!stmt) return true;
  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
    for (const auto& s : seq->stmts_) {
      if (!IsEffectivelyEmpty(s)) return false;
    }
    return true;
  }
  if (auto ops = std::dynamic_pointer_cast<const OpStmts>(stmt)) {
    return ops->stmts_.empty();
  }
  return false;
}

class ComprehensiveDCE : public IRMutator {
 public:
  ComprehensiveDCE(const std::unordered_set<std::string>& live_vars,
                   const std::unordered_set<std::string>& defined_vars)
      : live_vars_(live_vars), defined_vars_(defined_vars) {}

  bool changed() const { return changed_; }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    if (HasUndefinedRef(op->value_, defined_vars_)) {
      changed_ = true;
      return nullptr;
    }
    if (!live_vars_.count(op->var_->name_) && !IsSideEffectingOp(op->value_)) {
      changed_ = true;
      return nullptr;
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    if (HasUndefinedRef(op->expr_, defined_vars_)) {
      changed_ = true;
      return nullptr;
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    return FilterStmtList(op->stmts_, op->span_, true);
  }

  StmtPtr VisitStmt_(const OpStmtsPtr& op) override {
    return FilterStmtList(op->stmts_, op->span_, false);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto new_start = ExprFunctor<ExprPtr>::VisitExpr(op->start_);
    auto new_stop = ExprFunctor<ExprPtr>::VisitExpr(op->stop_);
    auto new_step = ExprFunctor<ExprPtr>::VisitExpr(op->step_);

    std::vector<IterArgPtr> new_iter_args;
    std::vector<VarPtr> new_return_vars;

    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (!HasUndefinedRef(op->iter_args_[i]->initValue_, defined_vars_)) {
        new_iter_args.push_back(op->iter_args_[i]);
        if (i < op->return_vars_.size()) new_return_vars.push_back(op->return_vars_[i]);
      } else {
        changed_ = true;
      }
    }

    auto new_body = StmtFunctor<StmtPtr>::VisitStmt(op->body_);
    if (!new_body) new_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);

    bool anything_changed = new_body.get() != op->body_.get() ||
                            new_iter_args.size() != op->iter_args_.size() ||
                            new_return_vars.size() != op->return_vars_.size() ||
                            new_start.get() != op->start_.get() ||
                            new_stop.get() != op->stop_.get() || new_step.get() != op->step_.get();
    if (anything_changed) {
      return std::make_shared<ForStmt>(op->loop_var_, new_start, new_stop, new_step, new_iter_args,
                                       new_body, new_return_vars, op->span_, op->kind_,
                                       op->chunk_size_, op->chunk_policy_, op->loop_origin_);
    }
    return op;
  }

  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    std::vector<ExprPtr> new_values;
    bool values_changed = false;
    for (const auto& val : op->value_) {
      if (HasUndefinedRef(val, defined_vars_)) {
        values_changed = true;
        changed_ = true;
      } else {
        new_values.push_back(val);
      }
    }
    if (new_values.empty() && !op->value_.empty()) {
      changed_ = true;
      return nullptr;
    }
    if (values_changed) {
      return std::make_shared<YieldStmt>(new_values, op->span_);
    }
    return op;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    if (HasUndefinedRef(op->condition_, defined_vars_)) {
      changed_ = true;
      return nullptr;
    }

    auto new_then = StmtFunctor<StmtPtr>::VisitStmt(op->then_body_);
    if (!new_then) new_then = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);

    std::optional<StmtPtr> new_else;
    if (op->else_body_.has_value()) {
      auto visited = StmtFunctor<StmtPtr>::VisitStmt(*op->else_body_);
      if (visited) {
        new_else = visited;
      }
    }

    bool then_empty = IsEffectivelyEmpty(new_then);
    bool else_empty = !new_else.has_value() || IsEffectivelyEmpty(*new_else);

    if (then_empty && else_empty) {
      changed_ = true;
      return nullptr;
    }

    bool then_changed = new_then.get() != op->then_body_.get();
    bool else_changed = false;
    if (new_else.has_value() && op->else_body_.has_value()) {
      else_changed = (*new_else).get() != (*op->else_body_).get();
    } else if (!new_else.has_value() && op->else_body_.has_value()) {
      else_changed = true;
    }

    if (then_changed || else_changed) {
      return std::make_shared<IfStmt>(op->condition_, new_then, new_else, op->return_vars_, op->span_);
    }
    return op;
  }

 private:
  const std::unordered_set<std::string>& live_vars_;
  const std::unordered_set<std::string>& defined_vars_;
  bool changed_ = false;

  StmtPtr FilterStmtList(const std::vector<StmtPtr>& stmts, const Span& span, bool is_seq) {
    std::vector<StmtPtr> new_stmts;
    bool list_changed = false;
    for (const auto& stmt : stmts) {
      auto new_stmt = StmtFunctor<StmtPtr>::VisitStmt(stmt);
      if (!new_stmt || IsEffectivelyEmpty(new_stmt)) {
        list_changed = true;
        continue;
      }
      if (new_stmt.get() != stmt.get()) list_changed = true;
      if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(new_stmt)) {
        for (const auto& s : seq->stmts_) {
          if (!IsEffectivelyEmpty(s)) new_stmts.push_back(s);
        }
        list_changed = true;
      } else {
        new_stmts.push_back(new_stmt);
      }
    }
    if (!list_changed) {
      if (is_seq) return std::make_shared<SeqStmts>(stmts, span);
      return std::make_shared<OpStmts>(stmts, span);
    }
    if (is_seq) return std::make_shared<SeqStmts>(new_stmts, span);
    return std::make_shared<OpStmts>(new_stmts, span);
  }
};

FunctionPtr RunDCE(FunctionPtr func) {
  for (int iteration = 0; iteration < 20; ++iteration) {
    VarDefCollector def_collector;
    def_collector.VisitStmt(func->body_);
    for (const auto& param : func->params_) {
      def_collector.defined_vars.insert(param->name_);
    }

    VarRefCollectorSimple ref_collector;
    ref_collector.VisitStmt(func->body_);
    for (const auto& param : func->params_) {
      ref_collector.referenced_vars.insert(param->name_);
    }

    ComprehensiveDCE dce(ref_collector.referenced_vars, def_collector.defined_vars);
    auto new_body = dce.VisitStmt(func->body_);
    if (!new_body || !dce.changed()) break;

    func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                       func->return_types_, new_body, func->span_, func->func_type_);
  }
  return func;
}

// ============================================================================
// Step 9b: Introduce Deep Copy Operations to Break Dependency Chains
// ============================================================================
//
// Replace shallow (zero-copy) reshape/view with deep (copy-semantic) equivalents
// in the AIV kernel.  This breaks large dependency chains at reshape/view
// boundaries so that each sub-chain can be independently analyzed for splitting.
//
// Rules:
//   tensor.reshape           → tensor.deep_reshape     (always)
//   tensor.view(local_src, …) → tensor.deep_view       (only if src is NOT a func param)

class ShallowToDeepMutator : public IRMutator {
 public:
  explicit ShallowToDeepMutator(const std::unordered_set<std::string>& func_param_names)
      : func_param_names_(func_param_names) {}

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call) return IRMutator::VisitStmt_(op);
    auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
    if (!opnode) return IRMutator::VisitStmt_(op);

    auto tt = std::dynamic_pointer_cast<const TensorType>(op->var_->GetType());
    if (!tt || tt->shape_.empty()) return IRMutator::VisitStmt_(op);

    if (opnode->name_ == "tensor.reshape") {
      auto deep_op = std::make_shared<Op>("tensor.deep_reshape");
      auto new_call = std::make_shared<Call>(deep_op, call->args_, call->kwargs_,
                                             call->GetType(), call->span_);
      return std::make_shared<AssignStmt>(op->var_, new_call, op->span_);
    }

    if (opnode->name_ == "tensor.view") {
      if (!call->args_.empty()) {
        if (auto src_var = As<Var>(call->args_[0])) {
          if (!func_param_names_.count(src_var->name_)) {
            auto deep_op = std::make_shared<Op>("tensor.deep_view");
            auto new_call = std::make_shared<Call>(deep_op, call->args_, call->kwargs_,
                                                   call->GetType(), call->span_);
            return std::make_shared<AssignStmt>(op->var_, new_call, op->span_);
          }
        }
      }
    }

    return IRMutator::VisitStmt_(op);
  }

 private:
  const std::unordered_set<std::string>& func_param_names_;
};

// ============================================================================
// Step 9c: AIV Dual-Core Split — Whole-Chain Analysis ("Duplicated by Default")
// ============================================================================

inline std::optional<int64_t> TryGetConstInt(const ExprPtr& expr) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(expr)) return ci->value_;
  return std::nullopt;
}

struct TensorSplitInfo {
  int split_axis = -1;
  int64_t original_dim = 0;
  int64_t half_dim = 0;
};

// Whole-chain splittability analyzer.
//
// Instead of analyzing each operation independently (which can produce incorrect
// splits when a downstream unsplittable op needs the full tensor), this analyzer:
//   1. Builds a def-use graph of tensor variables in the AIV kernel
//   2. Groups variables into dependency chains via union-find
//   3. For each chain, finds a common split axis valid for ALL tensors
//   4. Chains without a valid common axis stay DUPLICATED (both AIV cores
//      compute identically) — correct by construction
//
// This guarantees functional correctness for any input kernel.
class ChainSplitAnalyzer : public IRVisitor {
 public:
  std::unordered_set<std::string> duplicated_vars;
  std::unordered_map<std::string, TensorSplitInfo> var_split_info;
  std::unordered_map<std::string, std::unordered_set<int>> forbidden_axes;

  void Analyze(const StmtPtr& body, const std::unordered_set<std::string>& func_param_names) {
    func_param_names_ = &func_param_names;
    VisitStmt(body);
    BuildChains();
    AnalyzeChains();

    // Log chain analysis results
    size_t split_count = 0, dup_count = 0;
    std::unordered_map<std::string, std::vector<std::string>> chain_groups_log;
    for (const auto& [var_name, info] : var_info_) {
      if (!info.tensor_type) continue;
      chain_groups_log[Find(var_name)].push_back(var_name);
    }
    for (const auto& [rep, vars] : chain_groups_log) {
      bool is_split = var_split_info.count(vars[0]) > 0;
      if (is_split) {
        auto& si = var_split_info.at(vars[0]);
        LOG_INFO << "ExpandMixedKernel: chain [SPLIT axis=" << si.split_axis
                 << " dim=" << si.original_dim << "→" << si.half_dim
                 << "] (" << vars.size() << " vars)";
        split_count += vars.size();
      } else {
        LOG_INFO << "ExpandMixedKernel: chain [DUPLICATED] (" << vars.size() << " vars):";
        for (const auto& v : vars) LOG_INFO << "    " << v;
        dup_count += vars.size();
      }
    }
    LOG_INFO << "ExpandMixedKernel: total chains=" << chain_groups_log.size()
             << "  split_vars=" << split_count << "  duplicated_vars=" << dup_count;
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    std::string var_name = op->var_->name_;
    VarInfo info;
    info.name = var_name;

    if (auto tt = std::dynamic_pointer_cast<const TensorType>(op->var_->GetType())) {
      if (!tt->shape_.empty()) {
        info.tensor_type = tt;
      }
    }

    if (auto call = As<Call>(op->value_)) {
      auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
      if (opnode) {
        info.op_name = opnode->name_;

        if (opnode->name_ == "tensor.row_max" || opnode->name_ == "tensor.row_sum") {
          for (const auto& arg : call->args_) {
            if (auto var = As<Var>(arg)) {
              auto tt = std::dynamic_pointer_cast<const TensorType>(var->GetType());
              if (tt && !tt->shape_.empty()) {
                forbidden_axes[var->name_].insert(static_cast<int>(tt->shape_.size()) - 1);
              }
            }
          }
        }

        for (const auto& arg : call->args_) {
          if (auto var = As<Var>(arg)) {
            info.input_vars.insert(var->name_);
          }
        }
      }
    }

    var_info_[var_name] = std::move(info);
    IRVisitor::VisitStmt_(op);
  }

 private:
  struct VarInfo {
    std::string name;
    std::shared_ptr<const TensorType> tensor_type;
    std::string op_name;
    std::unordered_set<std::string> input_vars;
  };

  const std::unordered_set<std::string>* func_param_names_ = nullptr;
  std::unordered_map<std::string, VarInfo> var_info_;

  // Union-Find for grouping vars into dependency chains
  std::unordered_map<std::string, std::string> parent_;

  std::string Find(const std::string& x) {
    if (parent_.find(x) == parent_.end()) parent_[x] = x;
    if (parent_[x] != x) parent_[x] = Find(parent_[x]);
    return parent_[x];
  }

  void Union(const std::string& a, const std::string& b) {
    auto ra = Find(a), rb = Find(b);
    if (ra != rb) parent_[ra] = rb;
  }

  // Phase 1: Group tensor vars into chains via union-find.
  // Two tensor vars are in the same chain if connected through producer-consumer
  // edges of local (non-parameter) tensor operations.
  //
  // Deep copy operations (tensor.deep_reshape, tensor.deep_view) act as chain
  // boundaries: their output starts a new chain independent of the input.
  // This breaks mega-chains into sub-chains that may each have a valid common
  // split axis, even when the reshaped intermediate dimensions differ.
  void BuildChains() {
    for (const auto& [var_name, info] : var_info_) {
      if (!info.tensor_type) continue;
      if (info.op_name == "tensor.deep_reshape" || info.op_name == "tensor.deep_view") continue;
      for (const auto& input_name : info.input_vars) {
        if (func_param_names_ && func_param_names_->count(input_name)) continue;
        auto it = var_info_.find(input_name);
        if (it != var_info_.end() && it->second.tensor_type) {
          Union(var_name, input_name);
        }
      }
    }
  }

  // Phase 2: For each chain, check end-to-end splittability.
  void AnalyzeChains() {
    std::unordered_map<std::string, std::vector<std::string>> chain_groups;
    for (const auto& [var_name, info] : var_info_) {
      if (!info.tensor_type) continue;
      chain_groups[Find(var_name)].push_back(var_name);
    }

    for (const auto& [rep, vars] : chain_groups) {
      std::unordered_set<int> chain_forbidden;
      for (const auto& var_name : vars) {
        auto it = forbidden_axes.find(var_name);
        if (it != forbidden_axes.end()) {
          chain_forbidden.insert(it->second.begin(), it->second.end());
        }
      }

      int common_axis = FindCommonSplitAxis(vars, chain_forbidden);

      if (common_axis >= 0) {
        for (const auto& var_name : vars) {
          auto& vi = var_info_.at(var_name);
          auto dim_val = TryGetConstInt(vi.tensor_type->shape_[common_axis]);
          if (dim_val) {
            var_split_info[var_name] = {common_axis, *dim_val, *dim_val / 2};
          }
        }
      } else {
        for (const auto& var_name : vars) {
          duplicated_vars.insert(var_name);
        }
      }
    }
  }

  int FindCommonSplitAxis(const std::vector<std::string>& vars,
                          const std::unordered_set<int>& chain_forbidden) {
    int max_ndim = 0;
    for (const auto& var_name : vars) {
      auto& vi = var_info_.at(var_name);
      max_ndim = std::max(max_ndim, static_cast<int>(vi.tensor_type->shape_.size()));
    }

    for (int axis = 0; axis < max_ndim; ++axis) {
      if (chain_forbidden.count(axis)) continue;

      bool valid = true;
      for (const auto& var_name : vars) {
        auto& vi = var_info_.at(var_name);
        auto& shape = vi.tensor_type->shape_;
        if (axis >= static_cast<int>(shape.size())) { valid = false; break; }
        auto dim_val = TryGetConstInt(shape[axis]);
        if (!dim_val || *dim_val <= 1 || *dim_val % 2 != 0) { valid = false; break; }
      }
      if (valid) return axis;
    }
    return -1;
  }
};

class AIVSplitMutator : public IRMutator {
 public:
  AIVSplitMutator(const ChainSplitAnalyzer& chain_analyzer, VarPtr aiv_idx_var,
                  const std::unordered_set<std::string>& func_param_names)
      : duplicated_vars_(chain_analyzer.duplicated_vars),
        var_split_info_(chain_analyzer.var_split_info),
        aiv_idx_var_(std::move(aiv_idx_var)),
        func_param_names_(func_param_names) {}

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call) return IRMutator::VisitStmt_(op);
    auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
    if (!opnode) return IRMutator::VisitStmt_(op);

    auto tt = std::dynamic_pointer_cast<const TensorType>(op->var_->GetType());
    if (!tt || tt->shape_.empty()) return IRMutator::VisitStmt_(op);

    // DUPLICATED variables are not split — both AIV cores compute identically
    if (duplicated_vars_.count(op->var_->name_)) {
      return IRMutator::VisitStmt_(op);
    }

    // Look up precomputed split info from whole-chain analysis
    auto it = var_split_info_.find(op->var_->name_);
    if (it == var_split_info_.end() || it->second.split_axis < 0) {
      return IRMutator::VisitStmt_(op);
    }
    auto split_info = it->second;

    if (opnode->name_ == "tensor.view") {
      return RewriteTensorView(op, call, split_info);
    }
    if (opnode->name_ == "tensor.deep_view") {
      return RewriteTensorView(op, call, split_info);
    }
    if (opnode->name_ == "tensor.create") {
      return RewriteTensorCreate(op, call, split_info);
    }
    if (opnode->name_ == "tensor.assemble") {
      return RewriteTensorAssemble(op, call, split_info);
    }
    if (opnode->name_ == "tensor.deep_reshape") {
      return RewriteTensorReshape(op, call, split_info);
    }
    if (opnode->name_.find("comm.tpop_from_aic") != std::string::npos) {
      return RewriteTpopFromAic(op, split_info);
    }

    if (IsElementWiseOrReduction(opnode->name_)) {
      return RewriteElementWise(op, call, split_info);
    }

    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (!call) return IRMutator::VisitStmt_(op);
    auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
    if (!opnode) return IRMutator::VisitStmt_(op);

    return HandleTpushToAic(op);
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    return FlattenStmtList(op->stmts_, op->span_, true);
  }

  StmtPtr VisitStmt_(const OpStmtsPtr& op) override {
    return FlattenStmtList(op->stmts_, op->span_, false);
  }

  StmtPtr FlattenStmtList(const std::vector<StmtPtr>& stmts, const Span& span, bool is_seq) {
    std::vector<StmtPtr> new_stmts;
    bool changed = false;
    bool has_non_op = false;
    for (const auto& stmt : stmts) {
      auto new_stmt = StmtFunctor<StmtPtr>::VisitStmt(stmt);
      if (!new_stmt) { changed = true; continue; }
      if (new_stmt.get() != stmt.get()) changed = true;
      if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(new_stmt)) {
        for (const auto& s : seq->stmts_) new_stmts.push_back(s);
        changed = true;
      } else {
        new_stmts.push_back(new_stmt);
      }
      if (!std::dynamic_pointer_cast<const AssignStmt>(new_stmts.back()) &&
          !std::dynamic_pointer_cast<const EvalStmt>(new_stmts.back())) {
        has_non_op = true;
      }
    }
    if (!changed) {
      if (is_seq) return std::make_shared<SeqStmts>(stmts, span);
      return std::make_shared<OpStmts>(stmts, span);
    }
    if (!is_seq && has_non_op) {
      return std::make_shared<SeqStmts>(new_stmts, span);
    }
    if (is_seq) return std::make_shared<SeqStmts>(new_stmts, span);
    return std::make_shared<OpStmts>(new_stmts, span);
  }

 private:  // tpush_to_aic handler continues below

  StmtPtr HandleTpushToAic(const EvalStmtPtr& op) {
    auto call = As<Call>(op->expr_);
    auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
    if (opnode->name_ == "comm.tpush_to_aic") {
      // Both split and replicated vars pass AIV_IDX; AIC handles accordingly
      auto new_args = call->args_;
      new_args.push_back(aiv_idx_var_);
      auto new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->GetType(), call->span_);
      return std::make_shared<EvalStmt>(new_call, op->span_);
    }

    return IRMutator::VisitStmt_(op);
  }

 private:
  const std::unordered_set<std::string>& duplicated_vars_;
  const std::unordered_map<std::string, TensorSplitInfo>& var_split_info_;
  VarPtr aiv_idx_var_;
  const std::unordered_set<std::string>& func_param_names_;
  std::unordered_map<std::string, int64_t> var_half_dim_;

  ExprPtr MakeHalfDim(int64_t half_dim, const Span& span) {
    return std::make_shared<ConstInt>(half_dim, DataType::INDEX, span);
  }

  ExprPtr MakeAivOffset(const ExprPtr& base_offset, int64_t half_dim, const Span& span) {
    auto half_expr = std::make_shared<ConstInt>(half_dim, DataType::INDEX, span);
    auto delta = std::make_shared<Mul>(aiv_idx_var_, half_expr, DataType::INDEX, span);
    return std::make_shared<Add>(base_offset, delta, DataType::INDEX, span);
  }

  TypePtr MakeHalfType(const TensorType& tt, const TensorSplitInfo& info, const Span& span) {
    auto new_shape = tt.shape_;
    new_shape[info.split_axis] = MakeHalfDim(info.half_dim, span);
    return std::make_shared<TensorType>(new_shape, tt.dtype_);
  }

  struct TupleElements {
    std::vector<ExprPtr> elements;
    enum Kind { CALL, MAKE_TUPLE } kind;
    ExprPtr original;
  };

  std::optional<TupleElements> ExtractTupleElements(const ExprPtr& expr) {
    if (auto call = As<Call>(expr)) {
      return TupleElements{call->args_, TupleElements::CALL, expr};
    }
    if (auto mt = std::dynamic_pointer_cast<const MakeTuple>(expr)) {
      return TupleElements{mt->elements_, TupleElements::MAKE_TUPLE, expr};
    }
    return std::nullopt;
  }

  ExprPtr RebuildTuple(const TupleElements& orig, const std::vector<ExprPtr>& new_elems,
                       const Span& span) {
    if (orig.kind == TupleElements::CALL) {
      auto call = std::dynamic_pointer_cast<const Call>(orig.original);
      return std::make_shared<Call>(call->op_, new_elems, call->GetType(), span);
    }
    return std::make_shared<MakeTuple>(new_elems, span);
  }

  bool IsGlobalTensor(const ExprPtr& expr) {
    if (auto var = As<Var>(expr)) return func_param_names_.count(var->name_) > 0;
    return false;
  }

  StmtPtr RewriteTensorView(const AssignStmtPtr& op, const CallPtr& call,
                             const TensorSplitInfo& info) {
    if (call->args_.size() < 3) return IRMutator::VisitStmt_(op);

    auto shape_elems = ExtractTupleElements(call->args_[1]);
    auto offset_elems = ExtractTupleElements(call->args_[2]);
    if (!shape_elems || !offset_elems) return IRMutator::VisitStmt_(op);
    if (static_cast<int>(shape_elems->elements.size()) <= info.split_axis) return IRMutator::VisitStmt_(op);
    if (static_cast<int>(offset_elems->elements.size()) <= info.split_axis) return IRMutator::VisitStmt_(op);

    auto new_shape_elems = shape_elems->elements;
    new_shape_elems[info.split_axis] = MakeHalfDim(info.half_dim, op->span_);
    auto new_shape = RebuildTuple(*shape_elems, new_shape_elems, op->span_);

    auto new_offset_elems = offset_elems->elements;
    if (IsGlobalTensor(call->args_[0])) {
      new_offset_elems[info.split_axis] =
          MakeAivOffset(offset_elems->elements[info.split_axis], info.half_dim, op->span_);
    }
    auto new_offset = RebuildTuple(*offset_elems, new_offset_elems, op->span_);

    auto new_call_args = call->args_;
    new_call_args[1] = new_shape;
    new_call_args[2] = new_offset;
    auto new_type = MakeHalfType(
        *std::dynamic_pointer_cast<const TensorType>(op->var_->GetType()), info, op->span_);
    auto new_var = std::make_shared<Var>(op->var_->name_, new_type, op->var_->span_);
    auto new_call = std::make_shared<Call>(call->op_, new_call_args, call->kwargs_, new_type, op->span_);
    var_half_dim_[op->var_->name_] = info.half_dim;
    return std::make_shared<AssignStmt>(new_var, new_call, op->span_);
  }

  StmtPtr RewriteTensorCreate(const AssignStmtPtr& op, const CallPtr& call,
                               const TensorSplitInfo& info) {
    if (call->args_.empty()) return IRMutator::VisitStmt_(op);
    auto shape_elems = ExtractTupleElements(call->args_[0]);
    if (!shape_elems || static_cast<int>(shape_elems->elements.size()) <= info.split_axis)
      return IRMutator::VisitStmt_(op);

    auto new_shape_elems = shape_elems->elements;
    new_shape_elems[info.split_axis] = MakeHalfDim(info.half_dim, op->span_);
    auto new_shape = RebuildTuple(*shape_elems, new_shape_elems, op->span_);

    auto new_call_args = call->args_;
    new_call_args[0] = new_shape;
    auto new_type = MakeHalfType(
        *std::dynamic_pointer_cast<const TensorType>(op->var_->GetType()), info, op->span_);
    auto new_var = std::make_shared<Var>(op->var_->name_, new_type, op->var_->span_);
    auto new_call = std::make_shared<Call>(call->op_, new_call_args, call->kwargs_, new_type, op->span_);
    var_half_dim_[op->var_->name_] = info.half_dim;
    return std::make_shared<AssignStmt>(new_var, new_call, op->span_);
  }

  StmtPtr RewriteTensorAssemble(const AssignStmtPtr& op, const CallPtr& call,
                                 const TensorSplitInfo& /*info*/) {
    if (call->args_.size() < 3) return IRMutator::VisitStmt_(op);

    int64_t src_half_dim = 0;
    if (auto src_var = As<Var>(call->args_[1])) {
      auto it = var_half_dim_.find(src_var->name_);
      if (it != var_half_dim_.end()) {
        src_half_dim = it->second;
      } else {
        auto src_tt = std::dynamic_pointer_cast<const TensorType>(src_var->GetType());
        if (src_tt && !src_tt->shape_.empty()) {
          auto d = TryGetConstInt(src_tt->shape_[0]);
          if (d && *d > 1 && *d % 2 == 0) src_half_dim = *d / 2;
        }
      }
    }
    if (src_half_dim <= 0) return IRMutator::VisitStmt_(op);

    bool dest_is_global = false;
    if (auto dest_var = As<Var>(call->args_[0])) {
      auto dest_tt = std::dynamic_pointer_cast<const TensorType>(dest_var->GetType());
      if (dest_tt && !dest_tt->shape_.empty()) {
        auto dest_dim = TryGetConstInt(dest_tt->shape_[0]);
        if (dest_dim && *dest_dim > src_half_dim * 2) {
          dest_is_global = true;
        }
      }
      if (!dest_is_global && !func_param_names_.count(dest_var->name_)) {
        return IRMutator::VisitStmt_(op);
      }
    }

    auto offset_elems = ExtractTupleElements(call->args_[2]);
    if (!offset_elems || offset_elems->elements.empty()) return IRMutator::VisitStmt_(op);

    auto new_offset_elems = offset_elems->elements;
    new_offset_elems[0] = MakeAivOffset(offset_elems->elements[0], src_half_dim, op->span_);
    auto new_offset = RebuildTuple(*offset_elems, new_offset_elems, op->span_);

    auto new_call_args = call->args_;
    new_call_args[2] = new_offset;
    auto new_call = std::make_shared<Call>(call->op_, new_call_args, call->kwargs_, call->GetType(), op->span_);
    return std::make_shared<AssignStmt>(op->var_, new_call, op->span_);
  }

  StmtPtr RewriteTensorReshape(const AssignStmtPtr& op, const CallPtr& call,
                               const TensorSplitInfo& info) {
    if (call->args_.size() < 2) return IRMutator::VisitStmt_(op);
    auto shape_elems = ExtractTupleElements(call->args_[1]);
    if (!shape_elems || static_cast<int>(shape_elems->elements.size()) <= info.split_axis)
      return IRMutator::VisitStmt_(op);

    auto new_shape_elems = shape_elems->elements;
    new_shape_elems[info.split_axis] = MakeHalfDim(info.half_dim, op->span_);
    auto new_shape = RebuildTuple(*shape_elems, new_shape_elems, op->span_);

    auto new_call_args = call->args_;
    new_call_args[1] = new_shape;
    auto new_type = MakeHalfType(
        *std::dynamic_pointer_cast<const TensorType>(op->var_->GetType()), info, op->span_);
    auto new_var = std::make_shared<Var>(op->var_->name_, new_type, op->var_->span_);
    auto new_call = std::make_shared<Call>(call->op_, new_call_args, call->kwargs_, new_type, op->span_);
    var_half_dim_[op->var_->name_] = info.half_dim;
    return std::make_shared<AssignStmt>(new_var, new_call, op->span_);
  }

  static bool IsElementWiseOrReduction(const std::string& name) {
    static const std::unordered_set<std::string> ops = {
        "tensor.mul",     "tensor.sub",     "tensor.div",     "tensor.add",
        "tensor.exp",     "tensor.row_max", "tensor.row_sum", "tensor.cast",
        "tensor.maximum", "tensor.minimum",
    };
    return ops.count(name) > 0;
  }

  StmtPtr RewriteElementWise(const AssignStmtPtr& op, const CallPtr& call,
                              const TensorSplitInfo& info) {
    auto new_type = MakeHalfType(
        *std::dynamic_pointer_cast<const TensorType>(op->var_->GetType()), info, op->span_);
    auto new_var = std::make_shared<Var>(op->var_->name_, new_type, op->var_->span_);
    auto new_call = std::make_shared<Call>(call->op_, call->args_, call->kwargs_, new_type, op->span_);
    var_half_dim_[op->var_->name_] = info.half_dim;
    return std::make_shared<AssignStmt>(new_var, new_call, op->span_);
  }

  StmtPtr RewriteTpopFromAic(const AssignStmtPtr& op, const TensorSplitInfo& info) {
    auto new_type = MakeHalfType(
        *std::dynamic_pointer_cast<const TensorType>(op->var_->GetType()), info, op->span_);
    auto new_var = std::make_shared<Var>(op->var_->name_, new_type, op->var_->span_);
    auto tpop_op = std::make_shared<Op>("comm.tpop_from_aic");
    auto new_call = std::make_shared<Call>(tpop_op, std::vector<ExprPtr>{aiv_idx_var_}, new_type, op->span_);
    var_half_dim_[op->var_->name_] = info.half_dim;
    return std::make_shared<AssignStmt>(new_var, new_call, op->span_);
  }

};

// Step 9d: Double tpush/tpop in AIC kernel for dual AIV cores
class AICDoubleCommMutator : public IRMutator {
 public:
  explicit AICDoubleCommMutator(const std::unordered_set<std::string>& duplicated_vars)
      : duplicated_vars_(duplicated_vars) {}

 protected:
  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (!call) return IRMutator::VisitStmt_(op);
    auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
    if (!opnode || opnode->name_ != "comm.tpush_to_aiv") return IRMutator::VisitStmt_(op);

    if (call->args_.empty()) return IRMutator::VisitStmt_(op);
    auto tensor_arg = call->args_[0];
    auto tt = std::dynamic_pointer_cast<const TensorType>(tensor_arg->GetType());
    if (!tt || tt->shape_.empty()) return IRMutator::VisitStmt_(op);

    // Check if the pushed variable is DUPLICATED (unsplittable chain)
    bool is_duplicated = false;
    if (auto var = As<Var>(tensor_arg)) {
      is_duplicated = duplicated_vars_.count(var->name_) > 0;
    }

    if (is_duplicated) {
      // DUPLICATED: push FULL tensor to both AIV cores (no splitting)
      auto idx0 = std::make_shared<ConstInt>(0, DataType::INDEX, op->span_);
      auto idx1 = std::make_shared<ConstInt>(1, DataType::INDEX, op->span_);
      auto push_op = std::make_shared<Op>("comm.tpush_to_aiv");
      auto push0 = std::make_shared<Call>(push_op, std::vector<ExprPtr>{tensor_arg, idx0},
                                          tt, op->span_);
      auto push1 = std::make_shared<Call>(push_op, std::vector<ExprPtr>{tensor_arg, idx1},
                                          tt, op->span_);
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{
          std::make_shared<EvalStmt>(push0, op->span_),
          std::make_shared<EvalStmt>(push1, op->span_),
      }, op->span_);
    }

    // Split case: split tensor into halves, push one half to each AIV
    auto dim0_val = TryGetConstInt(tt->shape_[0]);
    if (!dim0_val || *dim0_val <= 1 || *dim0_val % 2 != 0) return IRMutator::VisitStmt_(op);

    int64_t half = *dim0_val / 2;
    auto list_op = std::make_shared<Op>("__list__");
    auto half_shape_args =
        std::vector<ExprPtr>{std::make_shared<ConstInt>(half, DataType::INDEX, op->span_)};
    for (size_t i = 1; i < tt->shape_.size(); ++i) half_shape_args.push_back(tt->shape_[i]);

    auto half_type = std::make_shared<TensorType>(
        std::vector<ExprPtr>(half_shape_args.begin(), half_shape_args.end()), tt->dtype_);
    auto half_shape = std::make_shared<Call>(list_op, half_shape_args, op->span_);
    auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, op->span_);
    auto half_const = std::make_shared<ConstInt>(half, DataType::INDEX, op->span_);

    std::vector<ExprPtr> offset0_args = {zero};
    std::vector<ExprPtr> offset1_args = {half_const};
    for (size_t i = 1; i < tt->shape_.size(); ++i) {
      offset0_args.push_back(zero);
      offset1_args.push_back(zero);
    }
    auto offset0 = std::make_shared<Call>(list_op, offset0_args, op->span_);
    auto offset1 = std::make_shared<Call>(list_op, offset1_args, op->span_);

    auto view_op = std::make_shared<Op>("tensor.view");
    auto half0_call = std::make_shared<Call>(view_op,
        std::vector<ExprPtr>{tensor_arg, half_shape, offset0}, half_type, op->span_);
    auto half1_call = std::make_shared<Call>(view_op,
        std::vector<ExprPtr>{tensor_arg, half_shape, offset1}, half_type, op->span_);

    auto half0_var = std::make_shared<Var>("__half0__", half_type, op->span_);
    auto half1_var = std::make_shared<Var>("__half1__", half_type, op->span_);
    auto assign0 = std::make_shared<AssignStmt>(half0_var, half0_call, op->span_);
    auto assign1 = std::make_shared<AssignStmt>(half1_var, half1_call, op->span_);

    auto idx0 = std::make_shared<ConstInt>(0, DataType::INDEX, op->span_);
    auto idx1 = std::make_shared<ConstInt>(1, DataType::INDEX, op->span_);

    auto push_op = std::make_shared<Op>("comm.tpush_to_aiv");
    auto push0 = std::make_shared<Call>(push_op, std::vector<ExprPtr>{half0_var, idx0},
                                        half_type, op->span_);
    auto push1 = std::make_shared<Call>(push_op, std::vector<ExprPtr>{half1_var, idx1},
                                        half_type, op->span_);
    auto push0_stmt = std::make_shared<EvalStmt>(push0, op->span_);
    auto push1_stmt = std::make_shared<EvalStmt>(push1, op->span_);

    return std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{assign0, assign1, push0_stmt, push1_stmt}, op->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call) return IRMutator::VisitStmt_(op);
    auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
    if (!opnode || opnode->name_ != "comm.tpop_from_aiv") return IRMutator::VisitStmt_(op);

    auto tt = std::dynamic_pointer_cast<const TensorType>(op->var_->GetType());
    if (!tt || tt->shape_.empty()) return IRMutator::VisitStmt_(op);

    // Check if this variable is DUPLICATED (unsplittable chain)
    bool is_duplicated = duplicated_vars_.count(op->var_->name_) > 0;

    if (is_duplicated) {
      // DUPLICATED: pop from both AIV cores, use only AIV_IDX=0 result
      auto idx0 = std::make_shared<ConstInt>(0, DataType::INDEX, op->span_);
      auto idx1 = std::make_shared<ConstInt>(1, DataType::INDEX, op->span_);
      auto pop_op = std::make_shared<Op>("comm.tpop_from_aiv");
      auto pop0 = std::make_shared<Call>(pop_op, std::vector<ExprPtr>{idx0}, tt, op->span_);
      auto pop1 = std::make_shared<Call>(pop_op, std::vector<ExprPtr>{idx1}, tt, op->span_);
      auto assign0 = std::make_shared<AssignStmt>(op->var_, pop0, op->span_);
      auto discard_var = std::make_shared<Var>(op->var_->name_ + "__discard", tt, op->span_);
      auto assign1 = std::make_shared<AssignStmt>(discard_var, pop1, op->span_);
      return std::make_shared<SeqStmts>(
          std::vector<StmtPtr>{assign0, assign1}, op->span_);
    }

    // Split case: pop halves from each AIV core, reassemble
    auto dim0_val = TryGetConstInt(tt->shape_[0]);
    if (!dim0_val || *dim0_val <= 1 || *dim0_val % 2 != 0) return IRMutator::VisitStmt_(op);

    int64_t half = *dim0_val / 2;
    std::vector<ExprPtr> half_shape_exprs = {std::make_shared<ConstInt>(half, DataType::INDEX, op->span_)};
    for (size_t i = 1; i < tt->shape_.size(); ++i) half_shape_exprs.push_back(tt->shape_[i]);
    auto half_type = std::make_shared<TensorType>(half_shape_exprs, tt->dtype_);

    auto idx0 = std::make_shared<ConstInt>(0, DataType::INDEX, op->span_);
    auto idx1 = std::make_shared<ConstInt>(1, DataType::INDEX, op->span_);

    auto pop_op = std::make_shared<Op>("comm.tpop_from_aiv");
    auto pop0 = std::make_shared<Call>(pop_op, std::vector<ExprPtr>{idx0}, half_type, op->span_);
    auto pop1 = std::make_shared<Call>(pop_op, std::vector<ExprPtr>{idx1}, half_type, op->span_);

    auto half0_var = std::make_shared<Var>(op->var_->name_ + "__h0", half_type, op->span_);
    auto half1_var = std::make_shared<Var>(op->var_->name_ + "__h1", half_type, op->span_);
    auto assign0 = std::make_shared<AssignStmt>(half0_var, pop0, op->span_);
    auto assign1 = std::make_shared<AssignStmt>(half1_var, pop1, op->span_);

    auto create_op = std::make_shared<Op>("tensor.create");
    auto list_op = std::make_shared<Op>("__list__");
    auto full_shape = std::make_shared<Call>(list_op,
        std::vector<ExprPtr>(tt->shape_.begin(), tt->shape_.end()), op->span_);
    auto create_call = std::make_shared<Call>(create_op, std::vector<ExprPtr>{full_shape},
        std::vector<std::pair<std::string, std::any>>{{"dtype", tt->dtype_}},
        op->var_->GetType(), op->span_);
    auto temp_var = std::make_shared<Var>(op->var_->name_ + "__tmp", op->var_->GetType(), op->span_);
    auto create_stmt = std::make_shared<AssignStmt>(temp_var, create_call, op->span_);

    auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, op->span_);
    auto half_const = std::make_shared<ConstInt>(half, DataType::INDEX, op->span_);
    std::vector<ExprPtr> off0_args = {zero};
    std::vector<ExprPtr> off1_args = {half_const};
    for (size_t i = 1; i < tt->shape_.size(); ++i) {
      off0_args.push_back(zero);
      off1_args.push_back(zero);
    }
    auto off0 = std::make_shared<Call>(list_op, off0_args, op->span_);
    auto off1 = std::make_shared<Call>(list_op, off1_args, op->span_);

    auto asm_op = std::make_shared<Op>("tensor.assemble");
    auto mid_var = std::make_shared<Var>(op->var_->name_ + "__mid", op->var_->GetType(), op->span_);
    auto asm0 = std::make_shared<Call>(asm_op,
        std::vector<ExprPtr>{temp_var, half0_var, off0}, op->var_->GetType(), op->span_);
    auto asm0_stmt = std::make_shared<AssignStmt>(mid_var, asm0, op->span_);

    auto asm1 = std::make_shared<Call>(asm_op,
        std::vector<ExprPtr>{mid_var, half1_var, off1}, op->var_->GetType(), op->span_);
    auto asm1_stmt = std::make_shared<AssignStmt>(op->var_, asm1, op->span_);

    return std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{assign0, assign1, create_stmt, asm0_stmt, asm1_stmt}, op->span_);
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    return FlattenFilterStmtList(op->stmts_, op->span_, true);
  }

  StmtPtr VisitStmt_(const OpStmtsPtr& op) override {
    return FlattenFilterStmtList(op->stmts_, op->span_, false);
  }

 private:
  const std::unordered_set<std::string>& duplicated_vars_;

  StmtPtr FlattenFilterStmtList(const std::vector<StmtPtr>& stmts, const Span& span, bool is_seq) {
    std::vector<StmtPtr> new_stmts;
    bool changed = false;
    for (const auto& stmt : stmts) {
      auto new_stmt = StmtFunctor<StmtPtr>::VisitStmt(stmt);
      if (!new_stmt) { changed = true; continue; }
      if (new_stmt.get() != stmt.get()) changed = true;
      if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(new_stmt)) {
        for (const auto& s : seq->stmts_) new_stmts.push_back(s);
        changed = true;
      } else {
        new_stmts.push_back(new_stmt);
      }
    }
    if (!changed) {
      if (is_seq) return std::make_shared<SeqStmts>(stmts, span);
      return std::make_shared<OpStmts>(stmts, span);
    }
    if (is_seq) return std::make_shared<SeqStmts>(new_stmts, span);
    return std::make_shared<OpStmts>(new_stmts, span);
  }
};

// ============================================================================
// Step 9e: Insert tfree instructions after last use of tpop'd tensors
// ============================================================================
//
// The tpop instruction only acquires a ring-buffer slot (wait-ready + load).
// The slot must be explicitly released via tfree after the consumer is done
// reading the data.  This pass finds the last use of each tpop variable in
// the same statement list and inserts the corresponding tfree after it.

class TfreeInserter : public IRMutator {
 public:
  TfreeInserter(const std::string& tpop_op_name, const std::string& tfree_op_name)
      : tpop_op_name_(tpop_op_name), tfree_op_name_(tfree_op_name) {}

  void PreCollect(const StmtPtr& body) {
    class TpopCollector : public IRVisitor {
     public:
      std::string tpop_op;
      std::unordered_map<std::string, ExprPtr> result;
      explicit TpopCollector(const std::string& op) : tpop_op(op) {}
     protected:
      void VisitStmt_(const AssignStmtPtr& op) override {
        auto call = As<Call>(op->value_);
        if (call) {
          auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
          if (opnode && opnode->name_ == tpop_op) {
            ExprPtr aiv_idx;
            if (!call->args_.empty()) aiv_idx = call->args_[0];
            result[op->var_->name_] = aiv_idx;
          }
        }
        IRVisitor::VisitStmt_(op);
      }
    };
    TpopCollector collector(tpop_op_name_);
    collector.VisitStmt(body);
    all_tpop_vars_ = std::move(collector.result);
  }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    return InsertTfreeInList(op->stmts_, op->span_, true);
  }

  StmtPtr VisitStmt_(const OpStmtsPtr& op) override {
    return InsertTfreeInList(op->stmts_, op->span_, false);
  }

 private:
  std::string tpop_op_name_;
  std::string tfree_op_name_;
  std::unordered_map<std::string, ExprPtr> all_tpop_vars_;
  std::unordered_set<std::string> freed_vars_;

  StmtPtr InsertTfreeInList(const std::vector<StmtPtr>& stmts, const Span& span, bool is_seq) {
    // Phase 1 (top-down): scan this level for unfreed tpop var references.
    // VarRefCollectorSimple recurses into nested scopes, so it finds references
    // in child IfStmts, ForStmts, etc.
    std::unordered_map<std::string, size_t> first_use, last_use;
    for (size_t i = 0; i < stmts.size(); ++i) {
      VarRefCollectorSimple rc;
      rc.VisitStmt(stmts[i]);
      for (const auto& ref : rc.referenced_vars) {
        if (all_tpop_vars_.count(ref) && !freed_vars_.count(ref)) {
          if (!first_use.count(ref)) first_use[ref] = i;
          last_use[ref] = i;
        }
      }
    }

    // Phase 2: Decide which vars to free at this level.
    // A var is freed at this level if its references span multiple child statements
    // (first_use != last_use). Otherwise, all references are within a single child
    // and we delegate to the recursive pass on that child for tighter placement.
    std::map<size_t, std::vector<StmtPtr>> insertions;
    for (const auto& [var_name, last_pos] : last_use) {
      bool spans_multiple = (first_use.at(var_name) != last_pos);
      if (spans_multiple) {
        auto aiv_idx = all_tpop_vars_.at(var_name);
        auto tfree_op = std::make_shared<Op>(tfree_op_name_);
        std::vector<ExprPtr> tfree_args;
        if (aiv_idx) tfree_args.push_back(aiv_idx);
        auto tfree_call = std::make_shared<Call>(tfree_op, tfree_args, span);
        insertions[last_pos].push_back(std::make_shared<EvalStmt>(tfree_call, span));
        freed_vars_.insert(var_name);
      }
    }

    // Phase 3: Build list with insertions at this level, then recurse into children.
    std::vector<StmtPtr> with_tfrees;
    bool has_insertions = !insertions.empty();
    for (size_t i = 0; i < stmts.size(); ++i) {
      with_tfrees.push_back(stmts[i]);
      auto ins_it = insertions.find(i);
      if (ins_it != insertions.end()) {
        for (const auto& tfree : ins_it->second) with_tfrees.push_back(tfree);
      }
    }

    // Phase 4: Recurse into children (they handle single-scope vars)
    std::vector<StmtPtr> result;
    bool changed = has_insertions;
    for (const auto& stmt : with_tfrees) {
      auto new_stmt = StmtFunctor<StmtPtr>::VisitStmt(stmt);
      if (!new_stmt) { changed = true; continue; }
      if (new_stmt.get() != stmt.get()) changed = true;
      result.push_back(new_stmt);
    }

    // Phase 5: Fallback — any tpop vars seen at this level but still unfreed
    // (e.g., discard vars that are never used, or vars whose only scope is a
    // leaf AssignStmt with no child container to recurse into).
    for (const auto& [var_name, last_pos] : last_use) {
      if (!freed_vars_.count(var_name)) {
        auto aiv_idx = all_tpop_vars_.at(var_name);
        auto tfree_op = std::make_shared<Op>(tfree_op_name_);
        std::vector<ExprPtr> tfree_args;
        if (aiv_idx) tfree_args.push_back(aiv_idx);
        auto tfree_call = std::make_shared<Call>(tfree_op, tfree_args, span);
        result.push_back(std::make_shared<EvalStmt>(tfree_call, span));
        freed_vars_.insert(var_name);
        changed = true;
      }
    }

    if (!changed) {
      if (is_seq) return std::make_shared<SeqStmts>(stmts, span);
      return std::make_shared<OpStmts>(stmts, span);
    }
    if (is_seq) return std::make_shared<SeqStmts>(result, span);
    return std::make_shared<OpStmts>(result, span);
  }
};

// ============================================================================
// Step 10: Update orchestration call sites
// ============================================================================

class OrchestrationCallRewriter : public IRMutator {
 public:
  explicit OrchestrationCallRewriter(
      const std::unordered_map<std::string, std::string>& func_to_group_map)
      : func_to_group_map_(func_to_group_map) {}

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto gvar = std::dynamic_pointer_cast<const GlobalVar>(op->op_);
    if (!gvar) return IRMutator::VisitExpr_(op);

    auto it = func_to_group_map_.find(gvar->name_);
    if (it == func_to_group_map_.end()) return IRMutator::VisitExpr_(op);

    // Replace call to original mixed function with call_group to the group
    auto group_gvar = std::make_shared<GlobalVar>(it->second);
    return std::make_shared<Call>(group_gvar, op->args_, op->kwargs_, op->GetType(), op->span_);
  }

 private:
  // Maps original mixed function name -> group name
  const std::unordered_map<std::string, std::string>& func_to_group_map_;
};

}  // namespace

// ============================================================================
// Pass entry point
// ============================================================================

namespace pass {

Pass ExpandMixedKernel() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> new_functions;
    // Maps original mixed function name -> {aic_name, aiv_name}
    std::unordered_map<std::string, std::pair<std::string, std::string>> expanded_map;
    // Maps original mixed function name -> group name
    std::unordered_map<std::string, std::string> func_to_group_map;
    // Collected groups
    std::vector<InCoreFunctionGroupPtr> groups;

    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ != FunctionType::InCore) {
        new_functions.push_back(func);
        continue;
      }

      ColorAnalyzer analyzer;
      analyzer.VisitStmt(func->body_);

      if (!analyzer.has_red || !analyzer.has_green) {
        new_functions.push_back(func);
        continue;
      }

      LOG_INFO << "ExpandMixedKernel: expanding mixed kernel '" << func->name_ << "'";

      auto boundaries = FindCrossColorBoundaries(analyzer);

      // --- Build AIC kernel ---
      std::string aic_name = func->name_ + "_aic";
      AICKernelMutator aic_mutator(analyzer, boundaries);
      auto aic_body = aic_mutator.VisitStmt(func->body_);

      // Step 5: prepend pipe init
      aic_body = PrependPipeInit(aic_body, "comm.aic_initialize_pipe", func->span_);

      FunctionPtr aic_func = std::make_shared<Function>(
          aic_name, func->params_, func->param_directions_, func->return_types_, aic_body,
          func->span_, FunctionType::InCore);
      aic_func = RunDCE(aic_func);

      // --- Build AIV kernel ---
      std::string aiv_name = func->name_ + "_aiv";
      AIVKernelMutator aiv_mutator(analyzer, boundaries);
      auto aiv_body = aiv_mutator.VisitStmt(func->body_);

      // Step 5: prepend pipe init
      aiv_body = PrependPipeInit(aiv_body, "comm.aiv_initialize_pipe", func->span_);

      // Step 9a: add AIV_IDX parameter
      auto aiv_idx_type = std::make_shared<ScalarType>(DataType::INDEX);
      auto aiv_idx_var = std::make_shared<Var>("AIV_IDX", aiv_idx_type, func->span_);
      auto aiv_params = func->params_;
      aiv_params.push_back(aiv_idx_var);
      auto aiv_param_dirs = func->param_directions_;
      aiv_param_dirs.push_back(ParamDirection::In);

      FunctionPtr aiv_func = std::make_shared<Function>(
          aiv_name, aiv_params, aiv_param_dirs, func->return_types_, aiv_body,
          func->span_, FunctionType::InCore);
      aiv_func = RunDCE(aiv_func);

      // Step 9b: introduce deep copies to break dependency chains
      std::unordered_set<std::string> func_param_names;
      for (const auto& p : func->params_) func_param_names.insert(p->name_);

      ShallowToDeepMutator shallow_to_deep(func_param_names);
      auto deep_body = shallow_to_deep.VisitStmt(aiv_func->body_);
      aiv_func = std::make_shared<Function>(
          aiv_func->name_, aiv_func->params_, aiv_func->param_directions_,
          aiv_func->return_types_, deep_body, aiv_func->span_, aiv_func->func_type_);

      // Step 9c: whole-chain analysis and split tensors in AIV kernel
      ChainSplitAnalyzer chain_analyzer;
      chain_analyzer.Analyze(aiv_func->body_, func_param_names);

      AIVSplitMutator aiv_split(chain_analyzer, aiv_idx_var, func_param_names);
      auto split_body = aiv_split.VisitStmt(aiv_func->body_);
      aiv_func = std::make_shared<Function>(
          aiv_func->name_, aiv_func->params_, aiv_func->param_directions_,
          aiv_func->return_types_, split_body, aiv_func->span_, aiv_func->func_type_);

      // Step 9d: double tpush/tpop in AIC kernel for two AIV cores
      AICDoubleCommMutator aic_double(chain_analyzer.duplicated_vars);
      auto doubled_body = aic_double.VisitStmt(aic_func->body_);
      aic_func = std::make_shared<Function>(
          aic_func->name_, aic_func->params_, aic_func->param_directions_,
          aic_func->return_types_, doubled_body, aic_func->span_, aic_func->func_type_);

      // Step 9e: insert tfree after last use of each tpop'd tensor
      // AIC kernel: tpop_from_aiv → tfree_to_aiv (release V2C slot back to Vector producer)
      TfreeInserter aic_tfree("comm.tpop_from_aiv", "comm.tfree_to_aiv");
      aic_tfree.PreCollect(aic_func->body_);
      auto aic_tfree_body = aic_tfree.VisitStmt(aic_func->body_);
      aic_func = std::make_shared<Function>(
          aic_func->name_, aic_func->params_, aic_func->param_directions_,
          aic_func->return_types_, aic_tfree_body, aic_func->span_, aic_func->func_type_);

      // AIV kernel: tpop_from_aic → tfree_to_aic (release C2V slot back to Cube producer)
      TfreeInserter aiv_tfree("comm.tpop_from_aic", "comm.tfree_to_aic");
      aiv_tfree.PreCollect(aiv_func->body_);
      auto aiv_tfree_body = aiv_tfree.VisitStmt(aiv_func->body_);
      aiv_func = std::make_shared<Function>(
          aiv_func->name_, aiv_func->params_, aiv_func->param_directions_,
          aiv_func->return_types_, aiv_tfree_body, aiv_func->span_, aiv_func->func_type_);

      new_functions.push_back(aic_func);
      new_functions.push_back(aiv_func);
      expanded_map[func->name_] = {aic_name, aiv_name};

      // Step 4/10: Create InCoreFunctionGroup with explicit parameter mapping
      std::string group_name = func->name_ + "_group";
      std::vector<std::string> shared_params;
      for (const auto& p : func->params_) shared_params.push_back(p->name_);
      std::vector<std::string> aiv_implicit_params = {"AIV_IDX"};
      auto group = std::make_shared<InCoreFunctionGroup>(
          group_name, aic_name, aiv_name,
          std::move(shared_params), std::move(aiv_implicit_params));
      groups.push_back(group);
      func_to_group_map[func->name_] = group_name;
    }

    if (!expanded_map.empty()) {
      std::vector<FunctionPtr> final_functions;
      // Step 10: Rewrite orchestration call sites to use call_group
      OrchestrationCallRewriter rewriter(func_to_group_map);
      for (const auto& func : new_functions) {
        if (func->func_type_ == FunctionType::Orchestration) {
          auto new_body = rewriter.VisitStmt(func->body_);
          auto new_func = std::make_shared<Function>(func->name_, func->params_,
                                                      func->param_directions_, func->return_types_,
                                                      new_body, func->span_, func->func_type_);
          final_functions.push_back(new_func);
        } else {
          final_functions.push_back(func);
        }
      }
      // Carry forward any existing groups from the input program
      auto all_groups = program->groups_;
      all_groups.insert(all_groups.end(), groups.begin(), groups.end());
      return std::make_shared<Program>(final_functions, std::move(all_groups),
                                       program->name_, program->span_);
    }

    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "ExpandMixedKernel", kExpandMixedKernelProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto

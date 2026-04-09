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

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

// ---------------------------------------------------------------------------
// Analysis: BufferRootCollector (local to this pass)
// ---------------------------------------------------------------------------

class BufferRootCollector : public IRVisitor {
 public:
  explicit BufferRootCollector(ProgramPtr program) : program_(std::move(program)) {}

  void Initialize(const std::vector<VarPtr>& params) {
    for (const auto& param : params) {
      buffer_roots_[param.get()] = param.get();
    }
  }

  [[nodiscard]] const Var* ResolveVar(const Var* var) const {
    auto it = buffer_roots_.find(var);
    return it != buffer_roots_.end() ? it->second : nullptr;
  }

  [[nodiscard]] const Var* ResolveExpr(const ExprPtr& expr) const {
    if (auto var = AsVarLike(expr)) {
      return ResolveVar(var.get());
    }
    return nullptr;
  }

  std::unordered_map<const Var*, const Var*> buffer_roots_;

 protected:
  void VisitStmt_(const ForStmtPtr& for_stmt) override {
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      const auto& iter_arg = for_stmt->iter_args_[i];
      const Var* root = ResolveExpr(iter_arg->initValue_);
      if (root) {
        buffer_roots_[iter_arg.get()] = root;
        if (i < for_stmt->return_vars_.size()) {
          buffer_roots_[for_stmt->return_vars_[i].get()] = root;
        }
      }
    }
    IRVisitor::VisitStmt_(for_stmt);
  }

  void VisitStmt_(const WhileStmtPtr& while_stmt) override {
    for (size_t i = 0; i < while_stmt->iter_args_.size(); ++i) {
      const auto& iter_arg = while_stmt->iter_args_[i];
      const Var* root = ResolveExpr(iter_arg->initValue_);
      if (root) {
        buffer_roots_[iter_arg.get()] = root;
        if (i < while_stmt->return_vars_.size()) {
          buffer_roots_[while_stmt->return_vars_[i].get()] = root;
        }
      }
    }
    IRVisitor::VisitStmt_(while_stmt);
  }

  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (auto call = As<Call>(assign->value_)) {
      const std::string& op_name = call->op_->name_;
      if (op_name == "tensor.create" || op_name == "tensor.slice") {
        buffer_roots_[assign->var_.get()] = assign->var_.get();
      } else if (op_name == "tensor.assemble") {
        if (call->args_.size() == 3) {
          if (const Var* target_root = ResolveExpr(call->args_[0])) {
            buffer_roots_[assign->var_.get()] = target_root;
          }
        }
      } else if (op_name.find("tile.") != 0 && op_name.find("tensor.") != 0 && op_name.find("system.") != 0) {
        auto out_roots = CollectCallOutputRoots(call);
        if (As<TupleType>(call->GetType())) {
          tuple_output_roots_[assign->var_.get()] = std::move(out_roots);
        } else if (!out_roots.empty() && out_roots[0]) {
          buffer_roots_[assign->var_.get()] = out_roots[0];
        }
      }
    } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
      if (auto tuple_var = AsVarLike(tuple_get->tuple_)) {
        auto it = tuple_output_roots_.find(tuple_var.get());
        if (it != tuple_output_roots_.end() && tuple_get->index_ < static_cast<int>(it->second.size()) &&
            it->second[tuple_get->index_]) {
          buffer_roots_[assign->var_.get()] = it->second[tuple_get->index_];
        }
      }
    } else if (auto src_var = AsVarLike(assign->value_)) {
      if (const Var* root = ResolveVar(src_var.get())) {
        buffer_roots_[assign->var_.get()] = root;
      }
    }
    IRVisitor::VisitStmt_(assign);
  }

 private:
  [[nodiscard]] std::vector<const Var*> CollectCallOutputRoots(const CallPtr& call) const {
    auto callee = program_->GetFunction(call->op_->name_);
    if (!callee) return {};

    std::vector<const Var*> roots;
    for (size_t i = 0; i < callee->param_directions_.size() && i < call->args_.size(); ++i) {
      if (callee->param_directions_[i] != ParamDirection::Out &&
          callee->param_directions_[i] != ParamDirection::InOut) {
        continue;
      }
      if (auto arg_var = AsVarLike(call->args_[i])) {
        roots.push_back(ResolveVar(arg_var.get()));
      } else {
        roots.push_back(nullptr);
      }
    }
    return roots;
  }

  ProgramPtr program_;
  std::unordered_map<const Var*, std::vector<const Var*>> tuple_output_roots_;
};

// ---------------------------------------------------------------------------
// Analysis: AssemblePatternCollector
// Detects tensor.create + tensor.assemble patterns eligible for fusion.
// Records (source_root -> {target_expr, offset_tuple}) for creates
// assembled exactly once; marks roots with multiple assembles.
// ---------------------------------------------------------------------------

struct FuseInfo {
  ExprPtr target_expr;
  MakeTuplePtr offset_tuple;
};

class AssemblePatternCollector : public IRVisitor {
 public:
  explicit AssemblePatternCollector(const std::unordered_map<const Var*, const Var*>& buffer_roots)
      : buffer_roots_(buffer_roots) {}

  std::unordered_map<const Var*, FuseInfo> fusible_roots;
  std::unordered_set<const Var*> non_fusible_roots;
  std::unordered_map<const Var*, const Var*> create_vars;

 protected:
  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (auto tuple_value = As<MakeTuple>(assign->value_)) {
      tuple_values_[assign->var_.get()] = tuple_value;
    } else if (auto call = As<Call>(assign->value_)) {
      if (call->op_->name_ == "tensor.create") {
        const Var* root = ResolveVar(assign->var_.get());
        if (root == assign->var_.get()) {
          create_vars[assign->var_.get()] = assign->var_.get();
        }
      } else if (call->op_->name_ == "tensor.assemble" && call->args_.size() == 3) {
        const Var* source_root = ResolveExpr(call->args_[1]);
        auto offset_tuple = ResolveTupleExpr(call->args_[2]);
        if (source_root && offset_tuple && create_vars.count(source_root) > 0) {
          RecordAssembleInfo(source_root, call->args_[0], offset_tuple);
        }
      }
    } else if (auto src_var = AsVarLike(assign->value_)) {
      if (auto it = tuple_values_.find(src_var.get()); it != tuple_values_.end()) {
        tuple_values_[assign->var_.get()] = it->second;
      }
    }
    IRVisitor::VisitStmt_(assign);
  }

 private:
  void RecordAssembleInfo(const Var* source_root, const ExprPtr& target_expr,
                          const MakeTuplePtr& offset_tuple) {
    if (non_fusible_roots.count(source_root) > 0) {
      return;
    }
    auto [it, inserted] = fusible_roots.emplace(source_root, FuseInfo{target_expr, offset_tuple});
    if (!inserted) {
      fusible_roots.erase(source_root);
      non_fusible_roots.insert(source_root);
    }
  }

  [[nodiscard]] const Var* ResolveVar(const Var* var) const {
    auto it = buffer_roots_.find(var);
    return it != buffer_roots_.end() ? it->second : nullptr;
  }

  [[nodiscard]] const Var* ResolveExpr(const ExprPtr& expr) const {
    if (auto var = AsVarLike(expr)) {
      return ResolveVar(var.get());
    }
    return nullptr;
  }

  [[nodiscard]] MakeTuplePtr ResolveTupleExpr(const ExprPtr& expr) const {
    if (auto tuple = As<MakeTuple>(expr)) {
      return tuple;
    }
    if (auto var = AsVarLike(expr)) {
      auto it = tuple_values_.find(var.get());
      if (it != tuple_values_.end()) {
        return it->second;
      }
    }
    return nullptr;
  }

  const std::unordered_map<const Var*, const Var*>& buffer_roots_;
  std::unordered_map<const Var*, MakeTuplePtr> tuple_values_;
};

// ---------------------------------------------------------------------------
// Mutator: Replace tensor.create with tensor.slice, remove tensor.assemble
// ---------------------------------------------------------------------------

class FuseCreateAssembleMutator : public IRMutator {
 public:
  FuseCreateAssembleMutator(const std::unordered_map<const Var*, FuseInfo>& fusible_roots,
                            const std::unordered_map<const Var*, const Var*>& buffer_roots)
      : fusible_roots_(fusible_roots), buffer_roots_(buffer_roots) {}

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call) return IRMutator::VisitStmt_(op);

    const std::string& op_name = call->op_->name_;

    if (op_name == "tensor.create") {
      const Var* root = ResolveVar(op->var_.get());
      if (root && fusible_roots_.count(root) > 0) {
        return RewriteCreateToSlice(op, call, fusible_roots_.at(root));
      }
    } else if (op_name == "tensor.assemble" && call->args_.size() == 3) {
      const Var* source_root = ResolveExpr(call->args_[1]);
      if (source_root && fusible_roots_.count(source_root) > 0) {
        return RewriteAssembleToAlias(op, call);
      }
    }

    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto result = IRMutator::VisitStmt_(op);
    return StripPassThroughIterArgs(result);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    auto result = IRMutator::VisitStmt_(op);
    return StripPassThroughWhileIterArgs(result);
  }

 private:
  StmtPtr RewriteCreateToSlice(const AssignStmtPtr& assign, const CallPtr& create_call,
                               const FuseInfo& info) {
    auto& op_registry = OpRegistry::GetInstance();

    auto result_type = As<TensorType>(create_call->GetType());
    if (!result_type) return assign;

    size_t ndim = result_type->shape_.size();

    std::vector<ExprPtr> shape_elements;
    shape_elements.reserve(ndim);
    for (const auto& dim : result_type->shape_) {
      shape_elements.push_back(dim);
    }
    auto shape_tuple = std::make_shared<MakeTuple>(std::move(shape_elements), assign->span_);

    auto offset_tuple = std::static_pointer_cast<const MakeTuple>(info.offset_tuple);

    ExprPtr target = VisitExpr(info.target_expr);

    auto slice_call = op_registry.Create("tensor.slice", {target, shape_tuple, offset_tuple}, assign->span_);

    return std::make_shared<AssignStmt>(assign->var_, slice_call, assign->span_);
  }

  StmtPtr RewriteAssembleToAlias(const AssignStmtPtr& assign, const CallPtr& call) {
    ExprPtr target = VisitExpr(call->args_[0]);
    var_remap_[assign->var_.get()] = target;
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, assign->span_);
  }

  // After mutation, a for loop's yield may pass an iter_arg through unchanged
  // (yield(iter_arg) instead of yield(new_value)). This happens when the
  // assemble that produced the new value was eliminated. Strip such iter_args
  // from the loop since they carry no state.
  StmtPtr StripPassThroughIterArgs(const StmtPtr& stmt) {
    auto for_stmt = As<ForStmt>(stmt);
    if (!for_stmt || for_stmt->iter_args_.empty()) return stmt;

    auto yield = GetTrailingYield(for_stmt->body_);
    if (!yield || yield->value_.size() != for_stmt->iter_args_.size()) return stmt;

    std::vector<bool> is_pass_through(for_stmt->iter_args_.size(), false);
    bool any = false;
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      auto yielded = AsVarLike(yield->value_[i]);
      if (yielded && yielded.get() == for_stmt->iter_args_[i].get()) {
        is_pass_through[i] = true;
        any = true;
      }
    }
    if (!any) return stmt;

    std::vector<IterArgPtr> new_iter_args;
    std::vector<VarPtr> new_return_vars;
    std::vector<ExprPtr> new_yield_values;
    std::unordered_map<const Var*, ExprPtr> body_subst;
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      if (is_pass_through[i]) {
        var_remap_[for_stmt->return_vars_[i].get()] = for_stmt->iter_args_[i]->initValue_;
        body_subst[for_stmt->iter_args_[i].get()] = for_stmt->iter_args_[i]->initValue_;
        continue;
      }
      new_iter_args.push_back(for_stmt->iter_args_[i]);
      new_return_vars.push_back(for_stmt->return_vars_[i]);
      new_yield_values.push_back(yield->value_[i]);
    }

    auto new_body = ReplaceTrailingYield(for_stmt->body_, new_yield_values, yield->span_);
    if (!body_subst.empty()) {
      new_body = transform_utils::Substitute(new_body, body_subst);
    }

    return std::make_shared<ForStmt>(for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_,
                                     new_iter_args, new_body, new_return_vars, for_stmt->span_,
                                     for_stmt->kind_, for_stmt->chunk_config_, for_stmt->attrs_);
  }

  StmtPtr StripPassThroughWhileIterArgs(const StmtPtr& stmt) {
    auto while_stmt = As<WhileStmt>(stmt);
    if (!while_stmt || while_stmt->iter_args_.empty()) return stmt;

    auto yield = GetTrailingYield(while_stmt->body_);
    if (!yield || yield->value_.size() != while_stmt->iter_args_.size()) return stmt;

    std::vector<bool> is_pass_through(while_stmt->iter_args_.size(), false);
    bool any = false;
    for (size_t i = 0; i < while_stmt->iter_args_.size(); ++i) {
      auto yielded = AsVarLike(yield->value_[i]);
      if (yielded && yielded.get() == while_stmt->iter_args_[i].get()) {
        is_pass_through[i] = true;
        any = true;
      }
    }
    if (!any) return stmt;

    std::vector<IterArgPtr> new_iter_args;
    std::vector<VarPtr> new_return_vars;
    std::vector<ExprPtr> new_yield_values;
    std::unordered_map<const Var*, ExprPtr> body_subst;
    for (size_t i = 0; i < while_stmt->iter_args_.size(); ++i) {
      if (is_pass_through[i]) {
        var_remap_[while_stmt->return_vars_[i].get()] = while_stmt->iter_args_[i]->initValue_;
        body_subst[while_stmt->iter_args_[i].get()] = while_stmt->iter_args_[i]->initValue_;
        continue;
      }
      new_iter_args.push_back(while_stmt->iter_args_[i]);
      new_return_vars.push_back(while_stmt->return_vars_[i]);
      new_yield_values.push_back(yield->value_[i]);
    }

    auto new_body = ReplaceTrailingYield(while_stmt->body_, new_yield_values, yield->span_);
    if (!body_subst.empty()) {
      new_body = transform_utils::Substitute(new_body, body_subst);
    }

    return std::make_shared<WhileStmt>(while_stmt->condition_, new_iter_args, new_body, new_return_vars,
                                       while_stmt->span_);
  }

  static YieldStmtPtr GetTrailingYield(const StmtPtr& body) {
    if (auto seq = As<SeqStmts>(body)) {
      if (seq->stmts_.empty()) return nullptr;
      return As<YieldStmt>(seq->stmts_.back());
    }
    return As<YieldStmt>(body);
  }

  static StmtPtr ReplaceTrailingYield(const StmtPtr& body, const std::vector<ExprPtr>& new_values,
                                      const Span& span) {
    if (As<YieldStmt>(body)) {
      if (new_values.empty()) {
        return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
      }
      return std::make_shared<YieldStmt>(new_values, span);
    }
    if (auto seq = As<SeqStmts>(body)) {
      std::vector<StmtPtr> stmts = seq->stmts_;
      if (!stmts.empty() && As<YieldStmt>(stmts.back())) {
        stmts.pop_back();
        if (!new_values.empty()) {
          stmts.push_back(std::make_shared<YieldStmt>(new_values, span));
        }
      }
      return SeqStmts::Flatten(std::move(stmts), seq->span_);
    }
    return body;
  }

  [[nodiscard]] const Var* ResolveVar(const Var* var) const {
    auto it = buffer_roots_.find(var);
    return it != buffer_roots_.end() ? it->second : nullptr;
  }

  [[nodiscard]] const Var* ResolveExpr(const ExprPtr& expr) const {
    if (auto var = AsVarLike(expr)) {
      return ResolveVar(var.get());
    }
    return nullptr;
  }

  const std::unordered_map<const Var*, FuseInfo>& fusible_roots_;
  const std::unordered_map<const Var*, const Var*>& buffer_roots_;
};

// ---------------------------------------------------------------------------
// Pass entry point
// ---------------------------------------------------------------------------

ProgramPtr TransformFuseCreateAssembleToSlice(const ProgramPtr& program) {
  bool any_changed = false;
  std::vector<FunctionPtr> new_functions;
  new_functions.reserve(program->functions_.size());

  for (const auto& [gvar, func] : program->functions_) {
    if (func->func_type_ != FunctionType::Orchestration) {
      new_functions.push_back(func);
      continue;
    }

    BufferRootCollector root_collector(program);
    root_collector.Initialize(func->params_);
    root_collector.VisitStmt(func->body_);

    AssemblePatternCollector pattern_collector(root_collector.buffer_roots_);
    pattern_collector.VisitStmt(func->body_);

    if (pattern_collector.fusible_roots.empty()) {
      new_functions.push_back(func);
      continue;
    }

    FuseCreateAssembleMutator mutator(pattern_collector.fusible_roots, root_collector.buffer_roots_);
    auto new_body = mutator.VisitStmt(func->body_);

    if (new_body.get() == func->body_.get()) {
      new_functions.push_back(func);
      continue;
    }

    any_changed = true;
    new_functions.push_back(std::make_shared<Function>(
        func->name_, func->params_, func->param_directions_, func->return_types_, new_body, func->span_,
        func->func_type_, func->level_, func->role_, func->attrs_));
  }

  if (!any_changed) return program;

  return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
}

inline const PassProperties kFuseCreateAssembleToSliceProperties{.required = {IRProperty::SplitIncoreOrch}};

}  // namespace

namespace pass {

Pass FuseCreateAssembleToSlice() {
  return CreateProgramPass(TransformFuseCreateAssembleToSlice, "FuseCreateAssembleToSlice",
                           kFuseCreateAssembleToSliceProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto

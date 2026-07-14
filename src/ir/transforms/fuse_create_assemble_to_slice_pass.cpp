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
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/comm.h"
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
#include "pypto/ir/transforms/utils/buffer_root_collector.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

using buffer_root::BufferRootCollector;

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
      if (IsOp(call, "tensor.create")) {
        const Var* root = ResolveVar(assign->var_.get());
        if (root == assign->var_.get()) {
          create_vars[assign->var_.get()] = assign->var_.get();
        }
      } else if (IsOp(call, "tensor.assemble") && call->args_.size() == 3) {
        const Var* source_root = ResolveExpr(call->args_[1]);
        auto offset_tuple = ResolveTupleExpr(call->args_[2]);
        if (source_root && offset_tuple && create_vars.count(source_root) > 0) {
          // An atomic-add assemble must survive lowering: fusing it into a
          // tensor.slice (RewriteAssembleToAlias) would silently drop the
          // atomic combine mode and degrade split-K accumulation to a plain
          // overwrite. Mark the root non-fusible so it is never eliminated.
          if (call->GetKwarg<int>("atomic", 0) != static_cast<int>(AtomicType::kNone)) {
            fusible_roots.erase(source_root);
            non_fusible_roots.insert(source_root);
          } else {
            RecordAssembleInfo(source_root, call->args_[0], offset_tuple);
          }
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

    if (IsOp(call, "tensor.create")) {
      const Var* root = ResolveVar(op->var_.get());
      if (root && fusible_roots_.count(root) > 0) {
        return RewriteCreateToSlice(op, call, fusible_roots_.at(root));
      }
    } else if (IsOp(call, "tensor.assemble") && call->args_.size() == 3) {
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

    auto offset_tuple = std::static_pointer_cast<const MakeTuple>(info.offset_tuple);
    size_t offset_ndim = offset_tuple->elements_.size();

    // When the assemble target has higher rank than the created tile
    // (e.g. 2D tile assembled into a 3D tensor), pad the shape with
    // leading singleton dimensions so that shape and offset ranks match
    // in the resulting tensor.slice.
    std::vector<ExprPtr> shape_elements;
    shape_elements.reserve(std::max(ndim, offset_ndim));
    for (size_t i = ndim; i < offset_ndim; ++i) {
      shape_elements.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, assign->span_));
    }
    for (const auto& dim : result_type->shape_) {
      shape_elements.push_back(dim);
    }
    auto shape_tuple = std::make_shared<MakeTuple>(std::move(shape_elements), assign->span_);

    ExprPtr target = VisitExpr(info.target_expr);

    auto slice_call = op_registry.Create("tensor.slice", {target, shape_tuple, offset_tuple}, assign->span_);

    if (offset_ndim > ndim) {
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, slice_call->GetType(), assign->var_->span_);
      var_remap_[assign->var_.get()] = new_var;
      return std::make_shared<AssignStmt>(new_var, slice_call, assign->span_);
    }
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

    auto new_for = MutableCopy(for_stmt);
    new_for->iter_args_ = std::move(new_iter_args);
    new_for->body_ = std::move(new_body);
    new_for->return_vars_ = std::move(new_return_vars);
    return new_for;
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

    auto new_while = MutableCopy(while_stmt);
    new_while->iter_args_ = std::move(new_iter_args);
    new_while->body_ = std::move(new_body);
    new_while->return_vars_ = std::move(new_return_vars);
    return new_while;
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

    // kSkip: fusion must never assert a false alias — an ambiguous match records
    // no root rather than risk aliasing a scratch onto the output (issue #1564).
    BufferRootCollector root_collector(program, buffer_root::AmbiguousRootPolicy::kSkip);
    root_collector.Initialize(func->params_);
    root_collector.VisitStmt(func->body_);

    AssemblePatternCollector pattern_collector(root_collector.buffer_roots);
    pattern_collector.VisitStmt(func->body_);

    if (pattern_collector.fusible_roots.empty()) {
      new_functions.push_back(func);
      continue;
    }

    FuseCreateAssembleMutator mutator(pattern_collector.fusible_roots, root_collector.buffer_roots);
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

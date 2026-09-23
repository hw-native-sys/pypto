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
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "src/ir/transforms/window_externalization/internal.h"

namespace pypto {
namespace ir {
namespace window_externalization {
using transform_utils::FlattenToStmts;

namespace {

/// Which localized output a value reads, shaped like the value: ``info`` for a
/// tensor, one entry per element for a tuple (nested for a nested tuple).
struct WindowProvenance {
  const OutputRewriteInfo* info = nullptr;
  std::vector<WindowProvenance> elements;

  /// True when nothing in the value reads a localized output -- including a
  /// tuple whose elements are all empty, which must not take the localized
  /// path (it would rewrite and re-copy state for every such loop carry).
  [[nodiscard]] bool Empty() const {
    return info == nullptr &&
           std::all_of(elements.begin(), elements.end(), [](const WindowProvenance& e) { return e.Empty(); });
  }
};

class WindowWriteLocalizer : public IRMutator {
  using TupleProvenanceMap = std::unordered_map<const Var*, WindowProvenance>;

 public:
  WindowWriteLocalizer(const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var,
                       const std::unordered_map<const Var*, ExprPtr>& new_out_vars,
                       WindowRewriteContext& rewrite_context)
      : out_info_by_var_(out_info_by_var), new_out_vars_(new_out_vars), rewrite_context_(rewrite_context) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto remap_it = result_var_remap_.find(op.get());
    if (remap_it != result_var_remap_.end()) return remap_it->second;
    auto out_it = new_out_vars_.find(op.get());
    if (out_it != new_out_vars_.end()) return out_it->second;
    return IRMutator::VisitExpr_(op);
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto out_it = new_out_vars_.find(op.get());
    if (out_it != new_out_vars_.end()) return out_it->second;
    return IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto visited_value = VisitExpr(op->value_);
    auto assign = MutableCopy(op);
    assign->value_ = visited_value;
    auto call = As<Call>(assign->value_);
    if (!call) return RetypeRewrittenValue(op, assign);

    ExprPtr rewritten_target_expr;
    const Var* target_var = nullptr;
    MakeTuplePtr offsets;
    size_t offset_arg_index = SIZE_MAX;
    size_t target_arg_index = SIZE_MAX;

    if (IsOp(call, "tile.store") && call->args_.size() >= 3) {
      rewritten_target_expr = call->args_[2];
      auto out_var = AsVarLike(rewritten_target_expr);
      if (!out_var) return assign;
      target_var = out_var.get();
      offsets = As<MakeTuple>(call->args_[1]);
      offset_arg_index = 1;
      target_arg_index = 2;
    } else if (IsOp(call, "tensor.assemble") && call->args_.size() >= 3) {
      rewritten_target_expr = call->args_[0];
      auto parent_var = AsVarLike(rewritten_target_expr);
      if (!parent_var) return assign;
      target_var = parent_var.get();
      offsets = As<MakeTuple>(call->args_[2]);
      offset_arg_index = 2;
      target_arg_index = 0;
    } else if (IsOp(call, "tile.load") && call->args_.size() >= 3) {
      rewritten_target_expr = call->args_[0];
      auto parent_var = AsVarLike(rewritten_target_expr);
      if (!parent_var) return assign;
      target_var = parent_var.get();
      offsets = As<MakeTuple>(call->args_[1]);
      offset_arg_index = 1;
      target_arg_index = 0;
    } else if (IsOp(call, "tensor.slice") && call->args_.size() >= 3) {
      rewritten_target_expr = call->args_[0];
      auto parent_var = AsVarLike(rewritten_target_expr);
      if (!parent_var) return assign;
      target_var = parent_var.get();
      offsets = As<MakeTuple>(call->args_[2]);
      offset_arg_index = 2;
      target_arg_index = 0;
    } else {
      return assign;
    }

    const OutputRewriteInfo* info = LookupOutputInfo(target_var);
    if (!info) return assign;
    if (!offsets) return assign;
    if (offsets->elements_.size() != info->callsite_offsets.size()) return assign;

    arith::Analyzer analyzer;
    std::vector<ExprPtr> local_offsets;
    local_offsets.reserve(offsets->elements_.size());
    std::vector<StmtPtr> prelude_stmts;
    for (size_t i = 0; i < offsets->elements_.size(); ++i) {
      auto local_offset = analyzer.Simplify(
          MakeSub(offsets->elements_[i], info->callsite_offsets[i], offsets->elements_[i]->span_));
      local_offsets.push_back(
          FlattenGeneratedScalarExpr(local_offset, assign->var_->name_hint_, assign->span_, &prelude_stmts));
    }
    auto new_offset_tuple = std::make_shared<MakeTuple>(std::move(local_offsets), offsets->span_);
    std::vector<ExprPtr> new_args = call->args_;
    new_args[offset_arg_index] = new_offset_tuple;
    auto new_out_it = new_out_vars_.find(target_var);
    if (new_out_it != new_out_vars_.end()) new_args[target_arg_index] = new_out_it->second;
    auto new_type = (IsOp(call, "tile.store") || IsOp(call, "tensor.assemble"))
                        ? new_args[target_arg_index]->GetType()
                        : call->GetType();
    auto new_call =
        std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_, new_type, call->span_);

    auto new_result_var = std::make_shared<Var>(assign->var_->name_hint_, new_type, assign->var_->span_);
    result_var_remap_[assign->var_.get()] = new_result_var;
    result_var_output_info_[new_result_var.get()] = info;
    assign->var_ = new_result_var;
    assign->value_ = new_call;
    if (!prelude_stmts.empty()) {
      prelude_stmts.push_back(assign);
      return SeqStmts::Flatten(std::move(prelude_stmts), assign->span_);
    }
    return assign;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto new_loop = MutableCopy(op);
    new_loop->start_ = VisitExpr(op->start_);
    new_loop->stop_ = VisitExpr(op->stop_);
    new_loop->step_ = VisitExpr(op->step_);

    std::unordered_map<const Var*, OutputRewriteInfo> nested_out_info(out_info_by_var_.begin(),
                                                                      out_info_by_var_.end());
    std::unordered_map<const Var*, ExprPtr> nested_new_out_vars(new_out_vars_.begin(), new_out_vars_.end());
    bool changed = false;

    for (size_t i = 0; i < new_loop->iter_args_.size() && i < new_loop->return_vars_.size(); ++i) {
      auto old_iter_arg = new_loop->iter_args_[i];
      auto old_return_var = new_loop->return_vars_[i];
      auto init_expr = VisitExpr(old_iter_arg->initValue_);
      auto provenance = ProvenanceOf(init_expr, old_iter_arg->initValue_);

      if (provenance.Empty()) {
        // Not localized. A rewritten init (e.g. a re-minted Var) still needs a
        // new IterArg, and the body's references must follow it.
        if (init_expr.get() != old_iter_arg->initValue_.get()) {
          auto new_iter_arg = std::make_shared<IterArg>(old_iter_arg->name_hint_, old_iter_arg->GetType(),
                                                        init_expr, old_iter_arg->span_);
          nested_new_out_vars[old_iter_arg.get()] = new_iter_arg;
          new_loop->iter_args_[i] = new_iter_arg;
          changed = true;
        }
        continue;
      }

      // Localized: the carry takes the init's narrowed window type — a tensor,
      // or a tuple holding one — and so does the value the loop returns.
      auto narrowed_type = init_expr->GetType();
      auto new_iter_arg =
          std::make_shared<IterArg>(old_iter_arg->name_hint_, narrowed_type, init_expr, old_iter_arg->span_);
      auto new_return_var =
          std::make_shared<Var>(old_return_var->name_hint_, narrowed_type, old_return_var->span_);

      nested_new_out_vars[old_iter_arg.get()] = new_iter_arg;
      nested_new_out_vars[new_iter_arg.get()] = new_iter_arg;
      result_var_remap_[old_return_var.get()] = new_return_var;
      if (provenance.info) {
        nested_out_info[old_iter_arg.get()] = *provenance.info;
        nested_out_info[new_iter_arg.get()] = *provenance.info;
        result_var_output_info_[new_return_var.get()] = provenance.info;
      } else {
        tuple_provenance_[new_iter_arg.get()] = provenance;
        tuple_provenance_[new_return_var.get()] = std::move(provenance);
      }

      new_loop->iter_args_[i] = new_iter_arg;
      new_loop->return_vars_[i] = new_return_var;
      changed = true;
    }

    if (!changed) return IRMutator::VisitStmt_(op);

    WindowWriteLocalizer nested_localizer(nested_out_info, nested_new_out_vars, result_var_remap_,
                                          result_var_output_info_, tuple_provenance_, rewrite_context_);
    new_loop->body_ = nested_localizer.VisitStmt(new_loop->body_);
    return new_loop;
  }

 private:
  const OutputRewriteInfo* LookupOutputInfo(const Var* var) const {
    auto info_it = out_info_by_var_.find(var);
    if (info_it != out_info_by_var_.end()) return &info_it->second;
    auto result_info_it = result_var_output_info_.find(var);
    return result_info_it != result_var_output_info_.end() ? result_info_it->second : nullptr;
  }

  /// Provenance of a Var, looked up by its rewritten identity first and its
  /// original one second: a tensor's output info, or a tuple's elements.
  WindowProvenance LookupVarProvenance(const ExprPtr& rewritten, const ExprPtr& original) const {
    for (const auto& expr : {rewritten, original}) {
      auto var = AsVarLike(expr);
      if (!var) continue;
      if (const auto* info = LookupOutputInfo(var.get())) return {info, {}};
      auto it = tuple_provenance_.find(var.get());
      if (it != tuple_provenance_.end()) return it->second;
    }
    return {};
  }

  /// Provenance of a non-Call value built from Vars, tuples, and projections,
  /// nested to any depth. @p original is the same value before rewriting (or
  /// null), used to find Vars the rewrite did not re-mint.
  WindowProvenance ProvenanceOf(const ExprPtr& rewritten, const ExprPtr& original) const {
    if (AsVarLike(rewritten)) return LookupVarProvenance(rewritten, original);
    if (auto tuple = As<MakeTuple>(rewritten)) {
      auto original_tuple = As<MakeTuple>(original);
      WindowProvenance result;
      result.elements.reserve(tuple->elements_.size());
      for (size_t i = 0; i < tuple->elements_.size(); ++i) {
        ExprPtr original_element =
            original_tuple && i < original_tuple->elements_.size() ? original_tuple->elements_[i] : nullptr;
        result.elements.push_back(ProvenanceOf(tuple->elements_[i], original_element));
      }
      return result;
    }
    if (auto item = As<TupleGetItemExpr>(rewritten)) {
      auto original_item = As<TupleGetItemExpr>(original);
      auto tuple = ProvenanceOf(item->tuple_, original_item ? original_item->tuple_ : nullptr);
      if (item->index_ >= 0 && static_cast<size_t>(item->index_) < tuple.elements.size()) {
        return tuple.elements[item->index_];
      }
    }
    return {};
  }

  /// A non-Call value that reads a localized result follows its narrowed
  /// window type: an alias (``out__store = out__ssa_v1``), a tuple built from
  /// it, a projection of such a tuple, or any nesting of these. Keeping the
  /// full-tensor LHS over the window-sized RHS breaks
  /// ``var.type == value.type``. Only an asymmetry this rewrite introduced is
  /// repaired. The re-minted Var also carries the value's provenance, so a
  /// later access through it rebases its offsets to the window.
  StmtPtr RetypeRewrittenValue(const AssignStmtPtr& op, const std::shared_ptr<AssignStmt>& assign) {
    if (assign->value_.get() == op->value_.get()) return assign;
    const auto& narrowed_type = assign->value_->GetType();
    if (structural_equal(assign->var_->GetType(), narrowed_type)) return assign;
    if (!structural_equal(op->var_->GetType(), op->value_->GetType())) return assign;
    auto new_var = std::make_shared<Var>(assign->var_->name_hint_, narrowed_type, assign->var_->span_);
    result_var_remap_[op->var_.get()] = new_var;
    auto provenance = ProvenanceOf(assign->value_, op->value_);
    if (provenance.info) {
      result_var_output_info_[new_var.get()] = provenance.info;
    } else if (!provenance.Empty()) {
      tuple_provenance_[new_var.get()] = std::move(provenance);
    }
    assign->var_ = new_var;
    return assign;
  }

  ExprPtr FlattenGeneratedScalarExpr(const ExprPtr& expr, const std::string& name_prefix, const Span& span,
                                     std::vector<StmtPtr>* stmts) {
    return FlattenGeneratedScalarExprWithLocalTemps(expr, name_prefix, span, stmts, rewrite_context_);
  }

  WindowWriteLocalizer(const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var,
                       const std::unordered_map<const Var*, ExprPtr>& new_out_vars,
                       std::unordered_map<const Var*, VarPtr> result_var_remap,
                       std::unordered_map<const Var*, const OutputRewriteInfo*> result_var_output_info,
                       TupleProvenanceMap tuple_provenance, WindowRewriteContext& rewrite_context)
      : out_info_by_var_(out_info_by_var),
        new_out_vars_(new_out_vars),
        result_var_remap_(std::move(result_var_remap)),
        result_var_output_info_(std::move(result_var_output_info)),
        tuple_provenance_(std::move(tuple_provenance)),
        rewrite_context_(rewrite_context) {}

  const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var_;
  const std::unordered_map<const Var*, ExprPtr>& new_out_vars_;
  std::unordered_map<const Var*, VarPtr> result_var_remap_;
  std::unordered_map<const Var*, const OutputRewriteInfo*> result_var_output_info_;
  /// Provenance of each re-minted tuple Var that holds localized values.
  TupleProvenanceMap tuple_provenance_;
  WindowRewriteContext& rewrite_context_;
};

class WindowReadLocalizer : public IRMutator {
 public:
  WindowReadLocalizer(const std::unordered_map<const Var*, InputRewriteInfo>& in_info_by_var,
                      WindowRewriteContext& rewrite_context)
      : in_info_by_var_(in_info_by_var), rewrite_context_(rewrite_context) {}

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto visited_value = VisitExpr(op->value_);
    auto assign = MutableCopy(op);
    assign->value_ = visited_value;

    auto call = As<Call>(assign->value_);
    if (!call || call->args_.empty()) return assign;

    size_t offset_arg_index = SIZE_MAX;
    if (IsOp(call, "tile.load") && call->args_.size() >= 3) {
      offset_arg_index = 1;
    } else if (IsOp(call, "tensor.slice") && call->args_.size() >= 3) {
      // Keep the localizer aligned with AnalyzeInputWindows(): only window
      // reads that are already proven as a fixed tile.load/tensor.slice are
      // rewritten, and tensor.slice only localizes the matched offset.
      offset_arg_index = 2;
    } else {
      return assign;
    }

    auto parent = AsVarLike(call->args_[0]);
    auto info_it = parent ? in_info_by_var_.find(parent.get()) : in_info_by_var_.end();
    if (info_it == in_info_by_var_.end()) return assign;

    auto old_offsets = As<MakeTuple>(call->args_[offset_arg_index]);
    if (!old_offsets) return assign;
    if (old_offsets->elements_.size() != info_it->second.callsite_offsets.size()) return assign;

    arith::Analyzer analyzer;
    std::vector<ExprPtr> local_offsets;
    local_offsets.reserve(old_offsets->elements_.size());
    std::vector<StmtPtr> prelude_stmts;
    for (size_t i = 0; i < old_offsets->elements_.size(); ++i) {
      ExprPtr base_offset = info_it->second.callsite_offsets[i];
      auto local_offset = analyzer.Simplify(
          MakeSub(old_offsets->elements_[i], base_offset, old_offsets->elements_[i]->span_));
      local_offsets.push_back(
          FlattenGeneratedScalarExpr(local_offset, assign->var_->name_hint_, assign->span_, &prelude_stmts));
    }

    std::vector<ExprPtr> new_args = call->args_;
    new_args[offset_arg_index] = std::make_shared<MakeTuple>(std::move(local_offsets), old_offsets->span_);
    assign->value_ = std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_, call->GetType(),
                                            call->span_);
    if (!prelude_stmts.empty()) {
      prelude_stmts.push_back(assign);
      return SeqStmts::Flatten(std::move(prelude_stmts), assign->span_);
    }
    return assign;
  }

 private:
  ExprPtr FlattenGeneratedScalarExpr(const ExprPtr& expr, const std::string& name_prefix, const Span& span,
                                     std::vector<StmtPtr>* stmts) {
    return FlattenGeneratedScalarExprWithLocalTemps(expr, name_prefix, span, stmts, rewrite_context_);
  }

  const std::unordered_map<const Var*, InputRewriteInfo>& in_info_by_var_;
  WindowRewriteContext& rewrite_context_;
};

}  // namespace

StmtPtr LocalizeWindowWrites(const StmtPtr& body,
                             const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var,
                             const std::unordered_map<const Var*, ExprPtr>& new_out_vars,
                             WindowRewriteContext& rewrite_context) {
  WindowWriteLocalizer localizer(out_info_by_var, new_out_vars, rewrite_context);
  return localizer.VisitStmt(body);
}

StmtPtr LocalizeWindowReads(const StmtPtr& body,
                            const std::unordered_map<const Var*, InputRewriteInfo>& in_info_by_var,
                            WindowRewriteContext& rewrite_context) {
  WindowReadLocalizer localizer(in_info_by_var, rewrite_context);
  return localizer.VisitStmt(body);
}

}  // namespace window_externalization
}  // namespace ir
}  // namespace pypto

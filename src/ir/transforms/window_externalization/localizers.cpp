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
#include <cstdint>
#include <map>
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

/// Immutable provenance DAG: tuple aliases and repeated elements share nodes.
/// A null pointer means no localized element, so emptiness is constant-time.
struct WindowProvenance;
using WindowProvenancePtr = std::shared_ptr<const WindowProvenance>;
struct WindowProvenance {
  const OutputRewriteInfo* info = nullptr;
  std::vector<WindowProvenancePtr> elements;
};

class WindowWriteLocalizer : public IRMutator {
  using TupleProvenanceMap = std::unordered_map<const Var*, WindowProvenancePtr>;

 public:
  WindowWriteLocalizer(const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var,
                       const std::unordered_map<const Var*, ExprPtr>& new_out_vars,
                       WindowRewriteContext& rewrite_context)
      : out_info_by_var_(out_info_by_var), new_out_vars_(new_out_vars), rewrite_context_(rewrite_context) {}

  [[nodiscard]] bool failed() const { return failed_; }

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto remap_it = result_var_remap_.find(op.get());
    if (remap_it != result_var_remap_.end()) return remap_it->second;
    auto out_it = new_out_vars_.find(op.get());
    if (out_it != new_out_vars_.end()) return out_it->second;
    return IRMutator::VisitExpr_(op);
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto remap_it = result_var_remap_.find(op.get());
    if (remap_it != result_var_remap_.end()) return remap_it->second;
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

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto result = As<IfStmt>(IRMutator::VisitStmt_(op));
    if (HasLocalizedYield(result->then_body_) ||
        (result->else_body_ && HasLocalizedYield(*result->else_body_))) {
      // Branch joins do not yet propagate window provenance to their results.
      // Decline the candidate before an unretyped join can escape the callee.
      failed_ = true;
      return op;
    }
    return result;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& arg : op->iter_args_) {
      auto init = VisitExpr(arg->initValue_);
      if (ProvenanceOf(init, arg->initValue_) || LookupVarProvenance(arg, arg)) {
        // While carries are not localized: their condition and zero-trip
        // results need the same type/provenance repair as their body yields.
        failed_ = true;
        return op;
      }
    }
    auto result = As<WhileStmt>(IRMutator::VisitStmt_(op));
    if (HasLocalizedYield(result->body_)) {
      failed_ = true;
      return op;
    }
    return result;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto new_loop = MutableCopy(op);
    new_loop->start_ = VisitExpr(op->start_);
    new_loop->stop_ = VisitExpr(op->stop_);
    new_loop->step_ = VisitExpr(op->step_);

    std::vector<WindowProvenancePtr> initial_provenance;
    initial_provenance.reserve(op->iter_args_.size());
    for (size_t i = 0; i < new_loop->iter_args_.size() && i < new_loop->return_vars_.size(); ++i) {
      const auto& old_iter_arg = op->iter_args_[i];
      const auto& old_return_var = op->return_vars_[i];
      auto init_expr = VisitExpr(old_iter_arg->initValue_);
      auto provenance = ProvenanceOf(init_expr, old_iter_arg->initValue_);
      // Aggregate-output analysis can seed the carry directly, after narrowing
      // its type, even though its init is not itself a localized result.
      if (!provenance) provenance = LookupVarProvenance(old_iter_arg, old_iter_arg);
      initial_provenance.push_back(provenance);
      if (!provenance && init_expr.get() == old_iter_arg->initValue_.get()) continue;

      const auto& carry_type = provenance ? init_expr->GetType() : old_iter_arg->GetType();
      auto new_iter_arg =
          std::make_shared<IterArg>(old_iter_arg->name_hint_, carry_type, init_expr, old_iter_arg->span_);
      result_var_remap_[old_iter_arg.get()] = new_iter_arg;
      RecordProvenance(new_iter_arg.get(), provenance);
      new_loop->iter_args_[i] = new_iter_arg;
      if (provenance) {
        auto new_return_var =
            std::make_shared<Var>(old_return_var->name_hint_, carry_type, old_return_var->span_);
        result_var_remap_[old_return_var.get()] = new_return_var;
        RecordProvenance(new_return_var.get(), provenance);
        new_loop->return_vars_[i] = new_return_var;
      }
    }

    // SSA identities are unique across scopes; one traversal can retain their
    // mappings without copying growing maps into every nested loop.
    new_loop->body_ = VisitStmt(op->body_);
    if (initial_provenance.empty()) return new_loop;
    auto stmts = FlattenToStmts(new_loop->body_);
    auto yield = stmts.empty() ? nullptr : As<YieldStmt>(stmts.back());
    if (!yield || yield->value_.size() != initial_provenance.size()) {
      failed_ = true;
      return op;
    }
    for (size_t i = 0; i < initial_provenance.size(); ++i) {
      if (!SameProvenance(initial_provenance[i], ProvenanceOf(yield->value_[i], nullptr))) {
        // A changing carry has no static offset valid for every iteration,
        // including zero trips. Decline the whole candidate rather than
        // silently rebasing a later access with the init's output window.
        failed_ = true;
        return op;
      }
    }
    return new_loop;
  }

 private:
  bool HasLocalizedYield(const StmtPtr& body) const {
    auto yield = transform_utils::GetLastYieldStmt(body);
    if (!yield) return false;
    for (const auto& value : yield->value_) {
      if (ProvenanceOf(value, nullptr)) return true;
    }
    return false;
  }

  const OutputRewriteInfo* LookupOutputInfo(const Var* var) const {
    auto info_it = out_info_by_var_.find(var);
    if (info_it != out_info_by_var_.end()) return &info_it->second;
    auto result_info_it = result_var_output_info_.find(var);
    return result_info_it != result_var_output_info_.end() ? result_info_it->second : nullptr;
  }

  /// Provenance of a Var, looked up by its rewritten identity first and its
  /// original one second: a tensor's output info, or a tuple's elements.
  WindowProvenancePtr LookupVarProvenance(const ExprPtr& rewritten, const ExprPtr& original) const {
    for (const auto& expr : {rewritten, original}) {
      auto var = AsVarLike(expr);
      if (!var) continue;
      if (const auto* info = LookupOutputInfo(var.get())) {
        return std::make_shared<WindowProvenance>(WindowProvenance{info, {}});
      }
      auto it = tuple_provenance_.find(var.get());
      if (it != tuple_provenance_.end()) return it->second;
    }
    return {};
  }

  /// Provenance of a non-Call value built from Vars, tuples, and projections,
  /// nested to any depth. @p original is the same value before rewriting (or
  /// null), used to find Vars the rewrite did not re-mint.
  WindowProvenancePtr ProvenanceOf(const ExprPtr& rewritten, const ExprPtr& original) const {
    if (AsVarLike(rewritten)) return LookupVarProvenance(rewritten, original);
    if (auto tuple = As<MakeTuple>(rewritten)) {
      auto original_tuple = As<MakeTuple>(original);
      std::vector<WindowProvenancePtr> elements;
      elements.reserve(tuple->elements_.size());
      bool localized = false;
      for (size_t i = 0; i < tuple->elements_.size(); ++i) {
        ExprPtr original_element =
            original_tuple && i < original_tuple->elements_.size() ? original_tuple->elements_[i] : nullptr;
        auto element = ProvenanceOf(tuple->elements_[i], original_element);
        localized |= element != nullptr;
        elements.push_back(std::move(element));
      }
      if (!localized) return nullptr;
      return std::make_shared<WindowProvenance>(WindowProvenance{nullptr, std::move(elements)});
    }
    if (auto item = As<TupleGetItemExpr>(rewritten)) {
      auto original_item = As<TupleGetItemExpr>(original);
      auto tuple = ProvenanceOf(item->tuple_, original_item ? original_item->tuple_ : nullptr);
      if (tuple && item->index_ >= 0 && static_cast<size_t>(item->index_) < tuple->elements.size()) {
        return tuple->elements[item->index_];
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
    RecordProvenance(new_var.get(), provenance);
    assign->var_ = new_var;
    return assign;
  }

  ExprPtr FlattenGeneratedScalarExpr(const ExprPtr& expr, const std::string& name_prefix, const Span& span,
                                     std::vector<StmtPtr>* stmts) {
    return FlattenGeneratedScalarExprWithLocalTemps(expr, name_prefix, span, stmts, rewrite_context_);
  }

  void RecordProvenance(const Var* var, const WindowProvenancePtr& provenance) {
    if (!provenance) return;
    if (provenance->info) {
      result_var_output_info_[var] = provenance->info;
    } else {
      tuple_provenance_[var] = provenance;
    }
  }

  /// Memoized DAG equality avoids revisiting a repeated tuple subtree.
  bool SameProvenance(const WindowProvenancePtr& lhs, const WindowProvenancePtr& rhs) {
    if (lhs == rhs) return true;
    if (!lhs || !rhs || lhs->info != rhs->info || lhs->elements.size() != rhs->elements.size()) {
      return false;
    }
    const auto key = std::make_pair(lhs, rhs);
    auto cached = provenance_equality_.find(key);
    if (cached != provenance_equality_.end()) return cached->second;
    bool equal = true;
    for (size_t i = 0; i < lhs->elements.size() && equal; ++i) {
      equal = SameProvenance(lhs->elements[i], rhs->elements[i]);
    }
    provenance_equality_.emplace(key, equal);
    return equal;
  }

  const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var_;
  const std::unordered_map<const Var*, ExprPtr>& new_out_vars_;
  std::unordered_map<const Var*, VarPtr> result_var_remap_;
  std::unordered_map<const Var*, const OutputRewriteInfo*> result_var_output_info_;
  /// Provenance of each re-minted tuple Var that holds localized values.
  TupleProvenanceMap tuple_provenance_;
  std::map<std::pair<WindowProvenancePtr, WindowProvenancePtr>, bool> provenance_equality_;
  bool failed_ = false;
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
  auto localized = localizer.VisitStmt(body);
  return localizer.failed() ? nullptr : localized;
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

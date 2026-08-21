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
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/window_externalization.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/window_externalization/internal.h"

namespace pypto {
namespace ir {
namespace window_externalization {
using transform_utils::FlattenToStmts;

namespace {

using LoopIterInitSubstMap = std::unordered_map<const Var*, ExprPtr>;

class ScopedLoopIterInitSubst {
 public:
  ScopedLoopIterInitSubst(LoopIterInitSubstMap* subst, const std::vector<IterArgPtr>& iter_args)
      : subst_(subst), saved_(*subst) {
    for (const auto& iter_arg : iter_args) {
      if (iter_arg && iter_arg->initValue_) {
        (*subst_)[iter_arg.get()] = iter_arg->initValue_;
      }
    }
  }

  ~ScopedLoopIterInitSubst() { *subst_ = std::move(saved_); }

 private:
  LoopIterInitSubstMap* subst_;
  LoopIterInitSubstMap saved_;
};

bool IsTensorAllocationOp(const CallPtr& call) {
  if (!call || std::dynamic_pointer_cast<const GlobalVar>(call->op_)) return false;
  return IsOp(call, "tensor.create") || IsOp(call, "tensor.full");
}

std::unordered_set<const Var*> CollectLoopLocalTensorAllocs(const ForStmtPtr& loop) {
  class Collector : public IRVisitor {
   public:
    [[nodiscard]] const std::unordered_set<const Var*>& result() const { return result_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (IsTensorAllocationOp(call) && As<TensorType>(op->var_->GetType())) {
        result_.insert(op->var_.get());
      }
      IRVisitor::VisitStmt_(op);
    }

   private:
    std::unordered_set<const Var*> result_;
  };

  if (!loop) return {};
  Collector collector;
  collector.VisitStmt(loop->body_);
  return collector.result();
}

bool HasMultiPieceOutput(const CalleeRewriteAnalysis& analysis) {
  return std::any_of(analysis.outputs.begin(), analysis.outputs.end(),
                     [](const OutputRewriteInfo& output) { return DensePieces(output).size() > 1; });
}

bool CanUseRuntimeViewDisjointness(const CalleeRewriteAnalysis& analysis) {
  return analysis.kind == RewriteKind::AggregateWindowLoop && HasMultiPieceOutput(analysis);
}

class OrchRewriter : public IRMutator {
 public:
  OrchRewriter(ProgramPtr program, const AnalysisMap& analyses,
               const std::unordered_map<std::string, FunctionPtr>& cloned_funcs,
               const std::unordered_map<std::string, FunctionPtr>& function_lookup,
               WindowRewriteContext& rewrite_context)
      : program_(std::move(program)),
        analyses_(analyses),
        cloned_funcs_(cloned_funcs),
        function_lookup_(function_lookup),
        rewrite_context_(rewrite_context) {}

  const std::unordered_set<std::string>& used_clone_names() const { return used_clone_names_; }

 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    bool is_sequential = op->kind_ != ForKind::Parallel;
    StmtPtr result;
    {
      ScopedLoopIterInitSubst scoped_loop_iter_init_subst(&loop_iter_init_subst_, op->iter_args_);

      loop_context_.push_back(LoopContext{op, op->loop_var_, op->start_, op->stop_, op->step_});
      if (is_sequential) {
        sequential_loops_.push_back(op);
        loop_local_allocs_.emplace_back(CollectLoopLocalTensorAllocs(op));
      }
      result = IRMutator::VisitStmt_(op);
      if (is_sequential) {
        loop_local_allocs_.pop_back();
        sequential_loops_.pop_back();
      }
      loop_context_.pop_back();
    }
    RecordLoopReturnInitAliases(op);
    return result;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    StmtPtr result;
    {
      ScopedLoopIterInitSubst scoped_loop_iter_init_subst(&loop_iter_init_subst_, op->iter_args_);
      ++while_depth_;
      result = IRMutator::VisitStmt_(op);
      --while_depth_;
    }
    auto visited_loop = As<WhileStmt>(result);
    RecordLoopReturnInitAliases(visited_loop ? visited_loop : op);
    return result;
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    new_stmts.reserve(op->stmts_.size());
    bool changed = false;
    auto saved_scalar_defs = scalar_defs_;
    auto saved_tuple_result_subst = tuple_result_subst_;
    auto saved_window_parent_subst = window_parent_subst_;
    auto saved_sibling_output_alias_roots = sibling_output_alias_roots_;
    auto saved_sibling_carrier_alias_roots = sibling_carrier_alias_roots_;
    auto saved_sibling_unwindowable_output_roots = sibling_unwindowable_output_roots_;
    auto later_assemble_source_indices = CollectAssembleSourceIndices(op->stmts_);
    sibling_output_alias_roots_.clear();
    // Carrier aliases model access-graph edges such as
    // tensor.assemble(parent, source, offset).  They must remain visible
    // when a nested writer is analyzed inside the source-producing loop.
    sibling_unwindowable_output_roots_.clear();
    CollectSiblingOutputAliases(op->stmts_);

    for (size_t stmt_index = 0; stmt_index < op->stmts_.size(); ++stmt_index) {
      const auto& stmt = op->stmts_[stmt_index];
      auto call_assign = As<AssignStmt>(stmt);
      auto bundle =
          call_assign ? TryRewriteCall(call_assign, later_assemble_source_indices, stmt_index) : std::nullopt;
      if (bundle.has_value()) {
        changed = true;
        for (const auto& new_stmt : bundle->stmts) {
          auto visited = VisitStmt(new_stmt);
          if (auto visited_assign = As<AssignStmt>(visited)) {
            if (As<ScalarType>(visited_assign->var_->GetType())) {
              scalar_defs_[visited_assign->var_.get()] = visited_assign->value_;
            }
          }
          new_stmts.push_back(visited);
        }
        for (const auto& [parent, replacement] : bundle->parent_substs) {
          window_parent_subst_[parent] = replacement;
        }
        continue;
      }

      auto visited = VisitStmt(stmt);
      changed = changed || visited.get() != stmt.get();
      new_stmts.push_back(visited);

      auto visited_assign = As<AssignStmt>(visited);
      if (visited_assign && As<ScalarType>(visited_assign->var_->GetType())) {
        scalar_defs_[visited_assign->var_.get()] = visited_assign->value_;
      }
    }

    scalar_defs_ = std::move(saved_scalar_defs);
    tuple_result_subst_ = std::move(saved_tuple_result_subst);
    window_parent_subst_ = std::move(saved_window_parent_subst);
    sibling_output_alias_roots_ = std::move(saved_sibling_output_alias_roots);
    sibling_carrier_alias_roots_ = std::move(saved_sibling_carrier_alias_roots);
    sibling_unwindowable_output_roots_ = std::move(saved_sibling_unwindowable_output_roots);
    if (!changed) return op;
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }

 private:
  struct SliceBundle {
    VarPtr slice_var;
    ExprPtr parent_expr;
    MakeTuplePtr shape_tuple;
    MakeTuplePtr offset_tuple;
  };

  struct RewriteBundle {
    std::vector<StmtPtr> stmts;
    std::vector<std::pair<const Var*, ExprPtr>> parent_substs;
  };

  struct LoopContext {
    ForStmtPtr loop;
    VarPtr loop_var;
    ExprPtr start;
    ExprPtr stop;
    ExprPtr step;
  };

  static bool IsSafeInlineScalarSubstitution(const ExprPtr& expr) {
    class Checker : public IRVisitor {
     public:
      [[nodiscard]] bool ok() const { return ok_; }

     protected:
      void VisitExpr_(const CallPtr& op) override {
        ok_ = false;
        IRVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const SubmitPtr& op) override {
        ok_ = false;
        IRVisitor::VisitExpr_(op);
      }

     private:
      bool ok_ = true;
    };

    if (!expr) return false;
    Checker checker;
    checker.VisitExpr(expr);
    return checker.ok();
  }

  ExprPtr FlattenGeneratedScalarExpr(const ExprPtr& expr, const std::string& name_prefix, const Span& span,
                                     std::vector<StmtPtr>* stmts) {
    return FlattenGeneratedScalarExprWithLocalTemps(expr, name_prefix, span, stmts, rewrite_context_);
  }

  static std::optional<std::vector<ExprPtr>> SubstituteSingletonLoopStarts(
      const std::vector<ExprPtr>& exprs, const std::vector<LoopContext>& loops) {
    std::unordered_map<const Var*, ExprPtr> subst;
    for (const auto& loop : loops) {
      if (!loop.loop || !loop.loop_var) continue;
      bool referenced = false;
      for (const auto& expr : exprs) {
        if (CountVarRefsInExpr(expr, loop.loop_var.get()) != 0) {
          referenced = true;
          break;
        }
      }
      if (!referenced) continue;
      auto trip_count = GetKnownPositiveTripCount(loop.loop);
      if (!trip_count.has_value() || *trip_count != 1) return std::nullopt;
      subst[loop.loop_var.get()] = loop.start;
    }
    return SubstituteExprVector(exprs, subst);
  }

  static bool SameCallsiteLoopContext(const std::vector<LoopContext>& lhs,
                                      const std::vector<LoopContext>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (lhs[i].loop_var.get() != rhs[i].loop_var.get()) return false;
      if (lhs[i].loop.get() != rhs[i].loop.get()) return false;
    }
    return true;
  }

  ExprPtr FindVisibleParentAfterStmt(const StmtPtr& stmt, const Var* root) const {
    if (!stmt || !root) return nullptr;
    auto find_from_loop = [&](const auto& loop) -> ExprPtr {
      if (!loop) return nullptr;
      const size_t n = std::min(loop->iter_args_.size(), loop->return_vars_.size());
      for (size_t i = 0; i < n; ++i) {
        const auto& iter_arg = loop->iter_args_[i];
        const auto& return_var = loop->return_vars_[i];
        if (!iter_arg || !iter_arg->initValue_ || !return_var) continue;
        if (ResolveCarrierParentRoot(iter_arg->initValue_) == root) return return_var;
      }
      return nullptr;
    };
    if (auto loop = As<ForStmt>(stmt)) {
      if (auto parent = find_from_loop(loop)) return parent;
    }
    if (auto loop = As<WhileStmt>(stmt)) {
      if (auto parent = find_from_loop(loop)) return parent;
    }
    auto parent_it = window_parent_subst_.find(root);
    if (parent_it != window_parent_subst_.end()) return parent_it->second;
    return nullptr;
  }

  struct LoopDisjointnessCandidate {
    ForStmtPtr loop;
    const std::unordered_set<const Var*>* loop_local_allocs = nullptr;
  };

  enum class LoopRegionRole {
    Partition,
    Reduction,
    Unknown,
  };

  template <typename LoopPtr>
  void RecordLoopReturnInitAliases(const LoopPtr& loop) {
    if (!loop) return;
    size_t n = std::min(loop->iter_args_.size(), loop->return_vars_.size());
    for (size_t i = 0; i < n; ++i) {
      const auto& iter_arg = loop->iter_args_[i];
      const auto& return_var = loop->return_vars_[i];
      if (!iter_arg || !iter_arg->initValue_ || !return_var) continue;
      if (return_var.get() == iter_arg.get()) continue;
      if (!AsTensorTypeLike(return_var->GetType())) continue;
      auto parent_expr = ResolveLoopInitExpr(iter_arg->initValue_);
      if (!AsVarLike(parent_expr)) continue;
      loop_iter_init_subst_[return_var.get()] = parent_expr;
      loop_return_init_subst_[return_var.get()] = parent_expr;
    }
  }

  const std::vector<OutParamReturnMapping>& GetOutParamReturnMappings(const FunctionPtr& func,
                                                                      bool include_inout) {
    static const std::vector<OutParamReturnMapping> kEmpty;
    if (!func) return kEmpty;
    auto key = func->name_ + (include_inout ? "#inout" : "#out");
    auto it = out_param_return_mappings_cache_.find(key);
    if (it != out_param_return_mappings_cache_.end()) return it->second;
    auto [inserted_it, _] = out_param_return_mappings_cache_.emplace(
        std::move(key), BuildOutParamReturnMappings(func, include_inout));
    return inserted_it->second;
  }

  static std::unordered_map<const Var*, size_t> CollectAssembleSourceIndices(
      const std::vector<StmtPtr>& sibling_stmts) {
    std::unordered_map<const Var*, size_t> result;
    for (size_t i = 0; i < sibling_stmts.size(); ++i) {
      auto assign = As<AssignStmt>(sibling_stmts[i]);
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (!call || !IsOp(call, "tensor.assemble") || call->args_.size() < 2) continue;
      auto source = AsVarLike(call->args_[1]);
      if (source) result[source.get()] = i;
    }
    return result;
  }

  static bool IsCallResultAssembledLater(
      const VarPtr& result_var, const std::unordered_map<const Var*, size_t>& assemble_source_indices,
      size_t stmt_index) {
    if (!result_var) return false;
    auto it = assemble_source_indices.find(result_var.get());
    return it != assemble_source_indices.end() && it->second > stmt_index;
  }

  std::optional<RewriteBundle> TryRewriteCall(
      const AssignStmtPtr& call_assign, const std::unordered_map<const Var*, size_t>& assemble_source_indices,
      size_t stmt_index) {
    // Submit (pl.submit inside pl.manual_scope) is a sibling call-like kind;
    // run the windowing analysis/rewrite on its augmented-Call view, then
    // rebuild as a Submit to preserve task-launch semantics + deps_
    // (.claude/rules/pass-submit-awareness.md). The per-callee analysis and
    // windowed clone are callee-body-driven (Analyze() over all functions),
    // so they exist regardless of the call-site kind.
    auto submit = As<Submit>(call_assign->value_);
    auto call = submit ? SubmitToCallView(submit) : As<Call>(call_assign->value_);
    if (!call) return std::nullopt;

    auto callee_name = GetCallFuncName(call);
    auto analysis_it = analyses_.find(callee_name);
    if (analysis_it == analyses_.end()) return std::nullopt;
    auto clone_it = cloned_funcs_.find(callee_name);
    if (clone_it == cloned_funcs_.end()) return std::nullopt;
    auto original_func = LookupFunction(callee_name);
    if (!original_func) return std::nullopt;

    std::string clone_usage_key = callee_name;
    FunctionPtr cloned_func = clone_it->second;
    const auto& analysis = analysis_it->second;

    if (analysis.outputs.empty() && analysis.inputs.empty()) return std::nullopt;
    if (submit && analysis.outputs.empty()) {
      return std::nullopt;
    }
    std::unordered_map<const Var*, ExprPtr> callsite_subst;
    for (size_t i = 0; i < original_func->params_.size() && i < call->args_.size(); ++i) {
      callsite_subst[original_func->params_[i].get()] = call->args_[i];
    }
    if (analysis.outputs.empty() &&
        IsCallResultAssembledLater(call_assign->var_, assemble_source_indices, stmt_index)) {
      return std::nullopt;
    }
    if (!analysis.outputs.empty() && !ProveCallsiteDisjointness(call_assign, call, analysis) &&
        !CanUseRuntimeViewDisjointness(analysis)) {
      return std::nullopt;
    }
    if (HasUnwindowableSiblingOutputWriter(call, analysis)) {
      return std::nullopt;
    }
    if (HasDuplicateExternalizedOutputParent(call, analysis)) {
      return std::nullopt;
    }
    if (HasManualDepsToMultiPieceOutput(call, analysis)) {
      return std::nullopt;
    }
    std::unordered_map<size_t, std::vector<VarPtr>> slices_by_in_index_multi;
    std::unordered_map<size_t, std::vector<SliceBundle>> slices_by_out_index;
    std::vector<StmtPtr> stmts;
    stmts.reserve((analysis.inputs.size() + analysis.outputs.size()) * 2 + 2);

    arith::Analyzer input_offset_analyzer;
    for (const auto& input : analysis.inputs) {
      if (input.in_param_index >= call->args_.size()) return std::nullopt;
      auto in_arg = AsVarLike(call->args_[input.in_param_index]);
      if (!in_arg) return std::nullopt;
      const auto& pieces = DensePieces(input);
      if (pieces.empty()) return std::nullopt;

      std::vector<VarPtr> input_slices;
      input_slices.reserve(pieces.size());
      for (size_t piece_index = 0; piece_index < pieces.size(); ++piece_index) {
        const auto& piece = pieces[piece_index];
        std::vector<ExprPtr> shape_exprs;
        shape_exprs.reserve(piece.window_shape.size());
        for (const auto& dim : piece.window_shape) {
          auto shape_expr = transform_utils::Substitute(dim, callsite_subst);
          shape_exprs.push_back(
              FlattenGeneratedScalarExpr(shape_expr, in_arg->name_hint_, call_assign->span_, &stmts));
        }
        std::vector<ExprPtr> offset_exprs;
        offset_exprs.reserve(piece.callsite_offsets.size());
        for (const auto& offset : piece.callsite_offsets) {
          auto offset_expr =
              input_offset_analyzer.Simplify(transform_utils::Substitute(offset, callsite_subst));
          offset_exprs.push_back(
              FlattenGeneratedScalarExpr(offset_expr, in_arg->name_hint_, call_assign->span_, &stmts));
        }
        auto shape_tuple = std::make_shared<MakeTuple>(shape_exprs, call_assign->span_);
        auto offset_tuple = std::make_shared<MakeTuple>(offset_exprs, call_assign->span_);

        ExprPtr parent_expr = MaterializeWindowParentExpr(call->args_[input.in_param_index]);
        auto slice_call = OpRegistry::GetInstance().Create(
            "tensor.slice", {parent_expr, shape_tuple, offset_tuple}, call_assign->span_);
        auto suffix =
            pieces.size() == 1 ? std::string("__window") : "__window_" + std::to_string(piece_index);
        auto slice_var =
            std::make_shared<Var>(in_arg->name_hint_ + suffix, slice_call->GetType(), in_arg->span_);
        stmts.push_back(std::make_shared<AssignStmt>(slice_var, slice_call, call_assign->span_));
        input_slices.push_back(slice_var);
      }
      slices_by_in_index_multi.emplace(input.in_param_index, std::move(input_slices));
    }

    arith::Analyzer output_offset_analyzer;
    for (const auto& output : analysis.outputs) {
      if (output.out_param_index >= call->args_.size() ||
          output.out_param_index >= original_func->params_.size()) {
        return std::nullopt;
      }
      auto out_arg = AsVarLike(call->args_[output.out_param_index]);
      if (!out_arg) return std::nullopt;
      const auto& pieces = DensePieces(output);
      if (pieces.empty()) return std::nullopt;

      std::vector<SliceBundle> output_slices;
      output_slices.reserve(pieces.size());
      ExprPtr parent_expr = MaterializeWindowParentExpr(call->args_[output.out_param_index]);
      for (size_t piece_index = 0; piece_index < pieces.size(); ++piece_index) {
        const auto& piece = pieces[piece_index];
        std::vector<ExprPtr> shape_exprs;
        shape_exprs.reserve(piece.window_shape.size());
        for (const auto& dim : piece.window_shape) {
          auto shape_expr = transform_utils::Substitute(dim, callsite_subst);
          shape_exprs.push_back(
              FlattenGeneratedScalarExpr(shape_expr, out_arg->name_hint_, call_assign->span_, &stmts));
        }

        std::vector<ExprPtr> offset_exprs;
        offset_exprs.reserve(piece.callsite_offsets.size());
        for (const auto& offset : piece.callsite_offsets) {
          auto offset_expr =
              output_offset_analyzer.Simplify(transform_utils::Substitute(offset, callsite_subst));
          offset_exprs.push_back(
              FlattenGeneratedScalarExpr(offset_expr, out_arg->name_hint_, call_assign->span_, &stmts));
        }
        auto output_param_type = As<TensorType>(original_func->params_[output.out_param_index]->GetType());
        if (CallsiteOutputWindowHasUnsafeStaticDynamicParent(output_param_type, shape_exprs, offset_exprs)) {
          return std::nullopt;
        }
        auto shape_tuple = std::make_shared<MakeTuple>(shape_exprs, call_assign->span_);
        auto offset_tuple = std::make_shared<MakeTuple>(offset_exprs, call_assign->span_);

        auto slice_call = OpRegistry::GetInstance().Create(
            "tensor.slice", {parent_expr, shape_tuple, offset_tuple}, call_assign->span_);
        auto suffix =
            pieces.size() == 1 ? std::string("__window") : "__window_" + std::to_string(piece_index);
        auto slice_var =
            std::make_shared<Var>(out_arg->name_hint_ + suffix, slice_call->GetType(), out_arg->span_);
        stmts.push_back(std::make_shared<AssignStmt>(slice_var, slice_call, call_assign->span_));
        output_slices.push_back(SliceBundle{slice_var, parent_expr, shape_tuple, offset_tuple});
      }
      slices_by_out_index.emplace(output.out_param_index, std::move(output_slices));
    }

    std::vector<ExprPtr> new_args;
    new_args.reserve(call->args_.size());
    for (size_t i = 0; i < call->args_.size(); ++i) {
      auto input_slice_it = slices_by_in_index_multi.find(i);
      if (input_slice_it != slices_by_in_index_multi.end()) {
        for (const auto& slice : input_slice_it->second) new_args.push_back(slice);
        continue;
      }
      auto slice_it = slices_by_out_index.find(i);
      if (slice_it != slices_by_out_index.end()) {
        for (const auto& slice : slice_it->second) new_args.push_back(slice.slice_var);
      } else {
        new_args.push_back(VisitExpr(call->args_[i]));
      }
    }

    auto cloned_gvar = std::make_shared<GlobalVar>(cloned_func->name_);
    auto rewritten_budget = EstimateCallLikeSubmitBudget(cloned_func, new_args, {});
    if (!WithinRuntimeSubmitArgLimits(rewritten_budget)) {
      return std::nullopt;
    }
    const bool is_submit_call = IsSubmitCall(call);
    std::vector<TypePtr> result_types = cloned_func->return_types_;
    std::unordered_map<const Var*, ExprPtr> cloned_param_callsite_subst;
    for (size_t i = 0; i < cloned_func->params_.size() && i < new_args.size(); ++i) {
      cloned_param_callsite_subst[cloned_func->params_[i].get()] = new_args[i];
    }
    for (auto& result_type : result_types) {
      result_type = SubstituteTypeExprs(result_type, cloned_param_callsite_subst);
    }
    std::unordered_map<size_t, std::vector<size_t>> piece_return_indices_by_out_param;
    size_t next_extra_return_index = original_func->return_types_.size();
    for (const auto& output : analysis.outputs) {
      const auto& pieces = DensePieces(output);
      if (pieces.empty()) return std::nullopt;
      std::vector<size_t> piece_return_indices;
      piece_return_indices.reserve(pieces.size());
      piece_return_indices.push_back(output.return_index);
      for (size_t piece_index = 1; piece_index < pieces.size(); ++piece_index) {
        piece_return_indices.push_back(next_extra_return_index++);
      }
      piece_return_indices_by_out_param.emplace(output.out_param_index, std::move(piece_return_indices));
    }
    if (next_extra_return_index != cloned_func->return_types_.size()) return std::nullopt;
    if (is_submit_call) {
      auto tuple_ty = As<TupleType>(call->GetType());
      if (!tuple_ty || tuple_ty->types_.size() != result_types.size() + 1) return std::nullopt;
      result_types.push_back(tuple_ty->types_.back());
    }
    auto finish_bundle = [&](RewriteBundle bundle) -> RewriteBundle {
      used_clone_names_.insert(clone_usage_key);
      return bundle;
    };
    TypePtr new_return_type =
        result_types.size() == 1 ? result_types[0] : std::make_shared<TupleType>(result_types);

    auto new_attrs = RewriteCallAttrs(call, analysis, slices_by_out_index);
    ExprPtr new_call;
    if (submit) {
      // Preserve Submit-ness and deps_ (the canonical encoding); drop the
      // view's synthesised manual_dep_edges attr so deps aren't duplicated.
      // new_return_type already carries the trailing TASK_ID (is_submit_call).
      // Drop the keys SubmitToCallView *synthesises* from first-class Submit
      // fields. RewriteCallAttrs copies every attr off the transient Call
      // view, so without this filter the rebuilt Submit would carry both the
      // real field and a stale attr copy of it — duplicated state that the
      // printer emits twice and that structural_hash silently ignores (its
      // attr codec skips Var-/Expr-valued entries). The fields themselves are
      // threaded explicitly through the constructor below.
      static const std::array<const char*, 4> kViewSynthesizedKeys = {kAttrPredicate, "core_num",
                                                                      "sync_start", "allow_early_resolve"};
      std::vector<std::pair<std::string, std::any>> submit_attrs;
      submit_attrs.reserve(new_attrs.size());
      for (const auto& attr : new_attrs) {
        // Bind the key to a plain local: capturing a structured binding in the
        // lambda below is a C++20 extension and this target builds as C++17.
        const std::string& key = attr.first;
        if (key == kAttrManualDepEdges) continue;
        if (std::any_of(kViewSynthesizedKeys.begin(), kViewSynthesizedKeys.end(),
                        [&key](const char* synth) { return key == synth; })) {
          continue;
        }
        submit_attrs.emplace_back(attr.first, attr.second);
      }
      new_call =
          std::make_shared<Submit>(cloned_gvar, new_args, submit->deps_, submit->kwargs_,
                                   std::move(submit_attrs), new_return_type, submit->span_, submit->core_num_,
                                   submit->sync_start_, submit->allow_early_resolve_, submit->predicate_);
    } else {
      new_call = std::make_shared<Call>(cloned_gvar, new_args, call->kwargs_, new_attrs, new_return_type,
                                        call->span_);
    }
    if (analysis.outputs.empty()) {
      stmts.push_back(std::make_shared<AssignStmt>(call_assign->var_, new_call, call_assign->span_));
      RewriteBundle bundle;
      bundle.stmts = std::move(stmts);
      return finish_bundle(std::move(bundle));
    }
    auto tmp_result_var = std::make_shared<Var>(call_assign->var_->name_hint_ + "__windowed", new_return_type,
                                                call_assign->var_->span_);
    stmts.push_back(std::make_shared<AssignStmt>(tmp_result_var, new_call, call_assign->span_));

    size_t total_output_pieces = 0;
    for (const auto& output : analysis.outputs) {
      total_output_pieces += DensePieces(output).size();
    }
    if (!is_submit_call && analysis.outputs.size() == 1 && total_output_pieces == 1 &&
        result_types.size() == 1) {
      const auto& output = analysis.outputs[0];
      const auto& slice_bundle = slices_by_out_index.at(output.out_param_index).front();
      auto assemble_call = OpRegistry::GetInstance().Create(
          "tensor.assemble", {slice_bundle.parent_expr, ExprPtr(tmp_result_var), slice_bundle.offset_tuple},
          call_assign->span_);
      stmts.push_back(std::make_shared<AssignStmt>(call_assign->var_, assemble_call, call_assign->span_));

      RewriteBundle bundle;
      bundle.stmts = std::move(stmts);
      if (auto parent_var = AsVarLike(slice_bundle.parent_expr)) {
        bundle.parent_substs.emplace_back(parent_var.get(), call_assign->var_);
      }
      return finish_bundle(std::move(bundle));
    }

    const size_t visible_result_count = original_func->return_types_.size() + (is_submit_call ? 1 : 0);
    std::vector<ExprPtr> assembled_result_exprs(visible_result_count);
    std::vector<StmtPtr> tail_stmts;
    tail_stmts.reserve(total_output_pieces * 3 + result_types.size() + 1);
    std::vector<std::pair<const Var*, ExprPtr>> bundle_parent_substs;

    std::unordered_map<size_t, VarPtr> tuple_items;
    for (const auto& output : analysis.outputs) {
      const auto& piece_return_indices = piece_return_indices_by_out_param.at(output.out_param_index);
      const auto& slice_bundles = slices_by_out_index.at(output.out_param_index);
      const auto& assemble_pieces = DensePieces(output);
      if (piece_return_indices.size() != slice_bundles.size()) return std::nullopt;
      if (piece_return_indices.size() != assemble_pieces.size()) return std::nullopt;
      if (output.return_index >= assembled_result_exprs.size()) return std::nullopt;

      ExprPtr current_parent_expr = slice_bundles.front().parent_expr;
      for (size_t piece_index = 0; piece_index < assemble_pieces.size(); ++piece_index) {
        const size_t piece_return_index = piece_return_indices[piece_index];
        ExprPtr item_expr;
        if (result_types.size() == 1) {
          item_expr = tmp_result_var;
        } else {
          auto item_it = tuple_items.find(piece_return_index);
          if (item_it == tuple_items.end()) {
            auto get_item = std::make_shared<TupleGetItemExpr>(
                tmp_result_var, static_cast<int>(piece_return_index), call_assign->span_);
            auto item_var = std::make_shared<Var>(
                call_assign->var_->name_hint_ + "__windowed_" + std::to_string(piece_return_index),
                result_types[piece_return_index], call_assign->var_->span_);
            tail_stmts.push_back(std::make_shared<AssignStmt>(item_var, get_item, call_assign->span_));
            item_it = tuple_items.emplace(piece_return_index, item_var).first;
          }
          item_expr = item_it->second;
        }
        const SliceBundle& slice_bundle = slice_bundles[piece_index];
        const auto& assemble_piece = assemble_pieces[piece_index];
        auto assemble_item_expr = item_expr;
        auto assemble_offset_tuple = slice_bundle.offset_tuple;
        auto assemble_call = OpRegistry::GetInstance().Create(
            "tensor.assemble", {current_parent_expr, assemble_item_expr, assemble_offset_tuple},
            call_assign->span_);
        auto parent_type = current_parent_expr->GetType();
        auto assembled_var =
            std::make_shared<Var>(call_assign->var_->name_hint_ + "__assembled_" +
                                      std::to_string(output.return_index) + "_" + std::to_string(piece_index),
                                  parent_type, call_assign->var_->span_);
        tail_stmts.push_back(std::make_shared<AssignStmt>(assembled_var, assemble_call, call_assign->span_));
        current_parent_expr = assembled_var;
      }

      assembled_result_exprs[output.return_index] = current_parent_expr;
      if (auto parent_var = AsVarLike(slice_bundles.front().parent_expr)) {
        bundle_parent_substs.emplace_back(parent_var.get(), current_parent_expr);
      }
    }

    for (size_t i = 0; i < assembled_result_exprs.size(); ++i) {
      if (!assembled_result_exprs[i]) {
        const size_t source_index =
            is_submit_call && i == assembled_result_exprs.size() - 1 ? result_types.size() - 1 : i;
        if (result_types.size() == 1) {
          assembled_result_exprs[i] = tmp_result_var;
        } else {
          auto get_item = std::make_shared<TupleGetItemExpr>(tmp_result_var, static_cast<int>(source_index),
                                                             call_assign->span_);
          auto item_var = std::make_shared<Var>(call_assign->var_->name_hint_ + "__pass_" + std::to_string(i),
                                                result_types[source_index], call_assign->var_->span_);
          tail_stmts.push_back(std::make_shared<AssignStmt>(item_var, get_item, call_assign->span_));
          assembled_result_exprs[i] = item_var;
        }
      }
    }

    if (visible_result_count == 1) {
      stmts.insert(stmts.end(), tail_stmts.begin(), tail_stmts.end());
      stmts.push_back(std::make_shared<AssignStmt>(call_assign->var_, assembled_result_exprs.front(),
                                                   call_assign->span_));
      RewriteBundle bundle;
      bundle.stmts = std::move(stmts);
      bundle.parent_substs = std::move(bundle_parent_substs);
      return finish_bundle(std::move(bundle));
    }

    tuple_result_subst_[call_assign->var_.get()] = std::move(assembled_result_exprs);
    stmts.insert(stmts.end(), tail_stmts.begin(), tail_stmts.end());
    auto rebuilt_tuple =
        std::make_shared<MakeTuple>(tuple_result_subst_.at(call_assign->var_.get()), call_assign->span_);
    stmts.push_back(std::make_shared<AssignStmt>(call_assign->var_, rebuilt_tuple, call_assign->span_));

    RewriteBundle bundle;
    bundle.stmts = std::move(stmts);
    bundle.parent_substs = std::move(bundle_parent_substs);
    return finish_bundle(std::move(bundle));
  }

  static bool IsSubmitCall(const CallPtr& call) {
    auto tuple_ty = As<TupleType>(call->GetType());
    if (!tuple_ty || tuple_ty->types_.empty()) return false;
    auto last = As<ScalarType>(tuple_ty->types_.back());
    return last != nullptr && last->dtype_ == DataType::TASK_ID;
  }

  struct SubmitArgBudget {
    int add_inout = 0;
    int add_input = 0;
    int add_output = 0;
    int add_scalar = 0;

    [[nodiscard]] int Total() const { return add_inout + add_input + add_output + add_scalar; }
  };

  static ArgDirection ParamDirectionToArgDirection(ParamDirection direction) {
    switch (direction) {
      case ParamDirection::In:
        return ArgDirection::Input;
      case ParamDirection::Out:
        return ArgDirection::Output;
      case ParamDirection::InOut:
        return ArgDirection::InOut;
    }
    INTERNAL_CHECK(false) << "Internal error: unexpected ParamDirection value";
  }

  static void AddBudgetArg(ArgDirection direction, const TypePtr& type, SubmitArgBudget* budget) {
    if (!budget) return;
    if (As<ScalarType>(type)) {
      ++budget->add_scalar;
      return;
    }
    switch (direction) {
      case ArgDirection::Input:
      case ArgDirection::NoDep:
        ++budget->add_input;
        return;
      case ArgDirection::Output:
      case ArgDirection::OutputExisting:
        ++budget->add_output;
        return;
      case ArgDirection::InOut:
        ++budget->add_inout;
        return;
      case ArgDirection::Scalar:
        ++budget->add_scalar;
        return;
    }
    INTERNAL_CHECK(false) << "Internal error: unexpected ArgDirection value";
  }

  static SubmitArgBudget EstimateCallLikeSubmitBudget(const FunctionPtr& callee,
                                                      const std::vector<ExprPtr>& args,
                                                      const std::vector<ArgDirection>& arg_directions) {
    SubmitArgBudget budget;
    if (!callee) return budget;
    const bool has_arg_directions = arg_directions.size() == args.size();
    for (size_t i = 0; i < args.size(); ++i) {
      TypePtr type = args[i] ? args[i]->GetType() : nullptr;
      ArgDirection direction = ArgDirection::Input;
      if (has_arg_directions) {
        direction = arg_directions[i];
      } else if (i < callee->param_directions_.size()) {
        direction = ParamDirectionToArgDirection(callee->param_directions_[i]);
      }
      AddBudgetArg(direction, type, &budget);
    }

    // A Submit need not cover every callee param (see Submit::args_ in
    // include/pypto/ir/expr.h). Uncovered params are runtime-allocated Out
    // outputs that codegen materializes as add_output, so they still consume
    // a runtime arg slot and must be counted against the budget here.
    for (size_t i = args.size(); i < callee->params_.size() && i < callee->param_directions_.size(); ++i) {
      if (callee->param_directions_[i] != ParamDirection::Out) continue;
      AddBudgetArg(ArgDirection::Output, callee->params_[i]->GetType(), &budget);
    }
    return budget;
  }

  static bool WithinRuntimeSubmitArgLimits(const SubmitArgBudget& budget) {
    // Mirrors runtime/src/common/task_interface/arg_direction.h without adding
    // a pass-layer dependency on runtime headers.
    constexpr int kCoreMaxTensorArgs = 32;
    constexpr int kCoreMaxScalarArgs = 16;
    return budget.add_inout + budget.add_input + budget.add_output <= kCoreMaxTensorArgs &&
           budget.add_scalar <= kCoreMaxScalarArgs;
  }

  std::vector<std::pair<std::string, std::any>> RewriteCallAttrs(
      const CallPtr& call, const CalleeRewriteAnalysis& analysis,
      const std::unordered_map<size_t, std::vector<SliceBundle>>& slices_by_out_index) const {
    std::vector<std::pair<std::string, std::any>> attrs;
    attrs.reserve(call->attrs_.size());
    for (const auto& [k, v] : call->attrs_) {
      if (k == kAttrArgDirections) continue;
      attrs.emplace_back(k, v);
    }
    for (auto& [k, v] : attrs) {
      if (k != kAttrManualDepEdges) continue;
      const auto* user_deps = std::any_cast<std::vector<VarPtr>>(&v);
      if (!user_deps) break;
      std::vector<VarPtr> rewritten;
      rewritten.reserve(user_deps->size());
      bool changed = false;
      for (const auto& dep : *user_deps) {
        bool replaced = false;
        for (const auto& output : analysis.outputs) {
          auto out_arg = AsVarLike(call->args_[output.out_param_index]);
          if (dep && out_arg && dep.get() == out_arg.get()) {
            const auto& slices = slices_by_out_index.at(output.out_param_index);
            if (slices.empty()) return attrs;
            rewritten.push_back(slices.front().slice_var);
            changed = true;
            replaced = true;
            break;
          }
        }
        if (!replaced) rewritten.push_back(dep);
      }
      if (changed) {
        return WithManualDepEdgesAttr(std::move(attrs), std::move(rewritten));
      }
      break;
    }
    return attrs;
  }

  bool HasManualDepsToMultiPieceOutput(const CallPtr& call, const CalleeRewriteAnalysis& analysis) const {
    for (const auto& [k, v] : call->attrs_) {
      if (k != kAttrManualDepEdges) continue;
      const auto* user_deps = std::any_cast<std::vector<VarPtr>>(&v);
      if (!user_deps) return false;
      for (const auto& dep : *user_deps) {
        for (const auto& output : analysis.outputs) {
          if (DensePieces(output).size() <= 1) continue;
          if (output.out_param_index >= call->args_.size()) return true;
          auto out_arg = AsVarLike(call->args_[output.out_param_index]);
          if (dep && out_arg && dep.get() == out_arg.get()) return true;
        }
      }
      return false;
    }
    return false;
  }

  const Var* ResolveOutputParentRoot(const CallPtr& call, size_t arg_index) const {
    if (!call || arg_index >= call->args_.size()) return nullptr;
    return ResolveCarrierParentRoot(call->args_[arg_index]);
  }

  const Var* ResolveOutputRootExpr(const ExprPtr& expr) const { return ResolveCarrierParentRoot(expr); }

  const Var* CanonicalizeOutputAliasRoot(const Var* root) const {
    if (!root) return nullptr;
    std::unordered_set<const Var*> seen;
    const Var* current = root;
    while (seen.insert(current).second) {
      auto it = sibling_output_alias_roots_.find(current);
      if (it == sibling_output_alias_roots_.end()) break;
      current = it->second;
      if (!current) return nullptr;
    }
    return current;
  }

  const Var* CanonicalizeCarrierParentRoot(const Var* root) const {
    if (!root) return nullptr;
    std::unordered_set<const Var*> seen;
    const Var* current = root;
    while (seen.insert(current).second) {
      if (auto it = sibling_output_alias_roots_.find(current); it != sibling_output_alias_roots_.end()) {
        current = it->second;
        if (!current) return nullptr;
        continue;
      }
      if (auto it = sibling_carrier_alias_roots_.find(current); it != sibling_carrier_alias_roots_.end()) {
        current = it->second;
        if (!current) return nullptr;
        continue;
      }
      break;
    }
    return current;
  }

  const Var* ResolveCarrierParentRoot(const ExprPtr& expr) const {
    auto parent = AsVarLike(ResolveLoopInitExpr(ResolveLoopReturnInitExpr(expr)));
    if (!parent) return nullptr;
    const Var* root = CanonicalizeCarrierParentRoot(parent.get());
    if (!root) return nullptr;
    return root;
  }

  const Var* ResolveVisibleParentRoot(const ExprPtr& expr) const {
    auto parent = AsVarLike(expr);
    if (!parent) return nullptr;
    return CanonicalizeOutputAliasRoot(parent.get());
  }

  void RecordSiblingCarrierAliasRoot(const Var* alias_root, const Var* parent_root) {
    if (!alias_root || !parent_root || alias_root == parent_root) return;
    auto [it, inserted] = sibling_carrier_alias_roots_.emplace(alias_root, parent_root);
    if (!inserted && it->second != parent_root) {
      it->second = nullptr;
    }
  }

  void CollectSiblingOutputAliases(const std::vector<StmtPtr>& sibling_stmts) {
    std::unordered_map<const Var*, std::vector<const Var*>> sibling_tuple_output_roots;

    class SiblingWriterCollector : public IRVisitor {
     public:
      SiblingWriterCollector(OrchRewriter* rewriter,
                             std::unordered_map<const Var*, std::vector<const Var*>>* tuple_output_roots)
          : rewriter_(rewriter), tuple_output_roots_(tuple_output_roots) {}

     protected:
      void VisitStmt_(const AssignStmtPtr& op) override {
        if (!op) return;
        CallPtr call;
        if (auto submit = As<Submit>(op->value_)) {
          call = SubmitToCallView(submit);
        } else {
          call = As<Call>(op->value_);
        }

        if (auto tuple_get = As<TupleGetItemExpr>(op->value_)) {
          auto tuple_var = AsVarLike(tuple_get->tuple_);
          auto tuple_it = tuple_var ? tuple_output_roots_->find(tuple_var.get()) : tuple_output_roots_->end();
          if (tuple_it != tuple_output_roots_->end() && tuple_get->index_ >= 0 &&
              static_cast<size_t>(tuple_get->index_) < tuple_it->second.size()) {
            if (const Var* root = tuple_it->second[static_cast<size_t>(tuple_get->index_)]) {
              rewriter_->sibling_output_alias_roots_[op->var_.get()] = root;
            }
          }
        }

        if (call && call->op_ && IsOp(call, "tensor.slice") && !call->args_.empty() &&
            AsTensorTypeLike(op->var_->GetType())) {
          if (const Var* parent_root = rewriter_->ResolveCarrierParentRoot(call->args_[0])) {
            rewriter_->RecordSiblingCarrierAliasRoot(op->var_.get(), parent_root);
          }
        }

        if (call && call->op_ && IsOp(call, "tensor.assemble") && call->args_.size() >= 2) {
          auto source_root_expr = rewriter_->ResolveLoopReturnInitExpr(call->args_[1]);
          auto source_root = AsVarLike(source_root_expr);
          const Var* parent_root = rewriter_->ResolveCarrierParentRoot(call->args_[0]);
          if (source_root) rewriter_->RecordSiblingCarrierAliasRoot(source_root.get(), parent_root);
        }

        if (!call || !call->op_ || op_predicates::IsBuiltinOp(call->op_->name_)) {
          IRVisitor::VisitStmt_(op);
          return;
        }

        auto callee = rewriter_->LookupFunction(call->op_->name_);
        if (!callee) {
          IRVisitor::VisitStmt_(op);
          return;
        }

        const Var* single_output_root = nullptr;
        size_t output_root_count = 0;
        auto arg_directions = call->GetArgDirections();
        bool has_callsite_directions = arg_directions.size() == call->args_.size();
        for (size_t i = 0; i < call->args_.size() && i < callee->param_directions_.size(); ++i) {
          bool is_writer = false;
          if (has_callsite_directions) {
            is_writer = IsWriterArgDirection(arg_directions[i]);
          } else {
            is_writer = callee->param_directions_[i] == ParamDirection::Out ||
                        callee->param_directions_[i] == ParamDirection::InOut;
          }
          if (!is_writer) {
            continue;
          }
          if (const Var* parent_root = rewriter_->ResolveOutputParentRoot(call, i)) {
            if (!rewriter_->HasOutputWindowAnalysis(call->op_->name_, i)) {
              rewriter_->sibling_unwindowable_output_roots_.insert(parent_root);
            }
            single_output_root = parent_root;
            ++output_root_count;
          }
        }
        if (output_root_count == 1 && AsTensorTypeLike(op->var_->GetType())) {
          rewriter_->sibling_output_alias_roots_[op->var_.get()] = single_output_root;
        }
        if (output_root_count > 0 && As<TupleType>(op->var_->GetType())) {
          std::vector<const Var*> tuple_roots(callee->return_types_.size(), nullptr);
          for (const auto& mapping : rewriter_->GetOutParamReturnMappings(callee, /*include_inout=*/true)) {
            if (mapping.return_index >= tuple_roots.size() || mapping.param_index >= call->args_.size()) {
              continue;
            }
            tuple_roots[mapping.return_index] = rewriter_->ResolveOutputParentRoot(call, mapping.param_index);
          }
          (*tuple_output_roots_)[op->var_.get()] = std::move(tuple_roots);
        }

        IRVisitor::VisitStmt_(op);
      }

      void VisitStmt_(const ForStmtPtr& op) override {
        {
          ScopedLoopIterInitSubst scoped_loop_iter_init_subst(&rewriter_->loop_iter_init_subst_,
                                                              op->iter_args_);
          IRVisitor::VisitStmt_(op);
        }
        rewriter_->RecordLoopReturnInitAliases(op);
      }

      void VisitStmt_(const WhileStmtPtr& op) override {
        {
          ScopedLoopIterInitSubst scoped_loop_iter_init_subst(&rewriter_->loop_iter_init_subst_,
                                                              op->iter_args_);
          IRVisitor::VisitStmt_(op);
        }
        rewriter_->RecordLoopReturnInitAliases(op);
      }

      void VisitStmt_(const IfStmtPtr& op) override { IRVisitor::VisitStmt_(op); }

     private:
      OrchRewriter* rewriter_;
      std::unordered_map<const Var*, std::vector<const Var*>>* tuple_output_roots_;
    };

    SiblingWriterCollector collector(this, &sibling_tuple_output_roots);
    for (const auto& sibling_stmt : sibling_stmts) {
      collector.VisitStmt(sibling_stmt);
    }
  }

  const Var* ResolveAggregateWriterFlowRoot(const Var* parent_root,
                                            const std::vector<LoopContext>& enclosing_loops) const {
    if (!parent_root) return nullptr;
    for (auto loop_it = enclosing_loops.rbegin(); loop_it != enclosing_loops.rend(); ++loop_it) {
      const auto& loop = loop_it->loop;
      if (!loop) continue;
      const size_t n = std::min(loop->iter_args_.size(), loop->return_vars_.size());
      for (size_t i = 0; i < n; ++i) {
        const auto& iter_arg = loop->iter_args_[i];
        const auto& return_var = loop->return_vars_[i];
        if (!iter_arg || !return_var) continue;
        if (iter_arg.get() == parent_root) return return_var.get();
      }
    }
    return parent_root;
  }

  static bool IsWriterArgDirection(ArgDirection direction) {
    return direction == ArgDirection::Output || direction == ArgDirection::OutputExisting ||
           direction == ArgDirection::InOut;
  }

  bool HasOutputWindowAnalysis(const std::string& callee_name, size_t out_param_index) const {
    auto analysis_it = analyses_.find(callee_name);
    if (analysis_it == analyses_.end()) return false;
    const auto& outputs = analysis_it->second.outputs;
    return std::any_of(outputs.begin(), outputs.end(), [out_param_index](const OutputRewriteInfo& output) {
      return output.out_param_index == out_param_index;
    });
  }

  bool HasUnwindowableSiblingOutputWriter(const CallPtr& call, const CalleeRewriteAnalysis& analysis) const {
    for (const auto& output : analysis.outputs) {
      const Var* parent_root = ResolveOutputParentRoot(call, output.out_param_index);
      if (!parent_root) return true;
      if (sibling_unwindowable_output_roots_.count(parent_root)) {
        return true;
      }
    }
    return false;
  }

  bool HasDuplicateExternalizedOutputParent(const CallPtr& call,
                                            const CalleeRewriteAnalysis& analysis) const {
    std::unordered_set<const Var*> seen_roots;
    for (const auto& output : analysis.outputs) {
      const Var* parent_root = ResolveOutputParentRoot(call, output.out_param_index);
      if (!parent_root) return true;
      if (!seen_roots.insert(parent_root).second) return true;
    }
    return false;
  }

  bool ProveCallsiteDisjointness(const AssignStmtPtr& call_assign, const CallPtr& call,
                                 const CalleeRewriteAnalysis& analysis) const {
    if (while_depth_ > 0) return false;
    std::vector<LoopDisjointnessCandidate> candidate_loops;
    candidate_loops.reserve(sequential_loops_.size());
    for (size_t i = 0; i < sequential_loops_.size(); ++i) {
      const auto& loop = sequential_loops_[i];
      if (!loop) continue;
      const auto* local_allocs = i < loop_local_allocs_.size() ? &loop_local_allocs_[i] : nullptr;
      candidate_loops.push_back(LoopDisjointnessCandidate{loop, local_allocs});
    }
    if (candidate_loops.empty()) return true;

    auto original_func = LookupFunction(call->op_->name_);
    if (!original_func) return false;

    std::unordered_map<const Var*, ExprPtr> callsite_subst;
    for (size_t i = 0; i < original_func->params_.size() && i < call->args_.size(); ++i) {
      callsite_subst[original_func->params_[i].get()] = call->args_[i];
    }

    for (const auto& output : analysis.outputs) {
      if (output.out_param_index >= original_func->params_.size()) return false;
      if (!ProveOutputDisjoint(candidate_loops, output, original_func->params_[output.out_param_index].get(),
                               callsite_subst)) {
        return false;
      }
    }
    return true;
  }

  bool CallsiteOutputWindowHasUnsafeStaticDynamicParent(const std::shared_ptr<const TensorType>& tensor_type,
                                                        const std::vector<ExprPtr>& window_shape,
                                                        const std::vector<ExprPtr>& offsets) const {
    if (!tensor_type || tensor_type->shape_.size() != window_shape.size() ||
        tensor_type->shape_.size() != offsets.size()) {
      return true;
    }

    std::unordered_set<const Var*> static_loop_vars;
    for (const auto& loop : loop_context_) {
      if (!loop.loop_var || !loop.loop || !GetStaticTripCount(loop.loop).has_value()) continue;
      static_loop_vars.insert(loop.loop_var.get());
    }

    for (size_t dim = 0; dim < tensor_type->shape_.size(); ++dim) {
      if (As<ConstInt>(tensor_type->shape_[dim])) continue;
      if (!As<ConstInt>(window_shape[dim])) continue;
      if (AreExprsEqual(window_shape[dim], tensor_type->shape_[dim])) continue;
      if (ExprReferencesOnlyVarsIn(offsets[dim], static_loop_vars)) return true;
    }
    return false;
  }

  bool ProveOutputDisjoint(const std::vector<LoopDisjointnessCandidate>& loops,
                           const OutputRewriteInfo& output, const Var* output_param,
                           const std::unordered_map<const Var*, ExprPtr>& callsite_subst) const {
    std::unordered_set<size_t> varying_dims_used;
    for (const auto& candidate : loops) {
      const auto role =
          ClassifyOutputLoopRole(candidate, output, output_param, callsite_subst, &varying_dims_used);
      if (role == LoopRegionRole::Unknown) {
        return false;
      }
    }
    return true;
  }

  LoopRegionRole ClassifyOutputLoopRole(const LoopDisjointnessCandidate& candidate,
                                        const OutputRewriteInfo& output, const Var* output_param,
                                        const std::unordered_map<const Var*, ExprPtr>& callsite_subst,
                                        std::unordered_set<size_t>* varying_dims_used) const {
    auto loop = candidate.loop;
    if (!loop) return LoopRegionRole::Unknown;
    if (IsOutputParentLocalToLoop(output_param, callsite_subst, candidate.loop_local_allocs)) {
      return LoopRegionRole::Reduction;
    }

    auto trip_count = GetStaticTripCount(loop);
    if (trip_count.has_value() && *trip_count <= 1) {
      return LoopRegionRole::Reduction;
    }

    if (output.window_shape.size() != output.callsite_offsets.size()) {
      return LoopRegionRole::Unknown;
    }
    std::optional<size_t> varying_dim;
    for (size_t i = 0; i < output.callsite_offsets.size(); ++i) {
      auto rewritten = transform_utils::Substitute(output.callsite_offsets[i], callsite_subst);
      rewritten = transform_utils::Substitute(rewritten, scalar_defs_);
      auto affine = ParseAffineInLoop(rewritten, loop->loop_var_.get());
      if (!affine.has_value()) return LoopRegionRole::Unknown;
      if (affine->coeff == 0) {
        continue;
      }

      auto extent_ci = As<ConstInt>(output.window_shape[i]);
      auto loop_step = transform_utils::EvalConstInt(loop->step_);
      if (!extent_ci || !loop_step.has_value()) return LoopRegionRole::Unknown;
      if (varying_dim.has_value()) return LoopRegionRole::Unknown;
      if (varying_dims_used && varying_dims_used->count(i)) return LoopRegionRole::Unknown;
      auto stride = CheckedMul(affine->coeff, *loop_step);
      if (!stride.has_value()) return LoopRegionRole::Unknown;
      auto stride_abs = CheckedAbs(*stride);
      if (!stride_abs.has_value()) return LoopRegionRole::Unknown;
      if (*stride_abs < extent_ci->value_) return LoopRegionRole::Unknown;
      varying_dim = i;
    }
    if (!varying_dim.has_value()) {
      return LoopRegionRole::Reduction;
    }
    if (varying_dims_used) varying_dims_used->insert(*varying_dim);
    return LoopRegionRole::Partition;
  }

  bool IsOutputParentLocalToLoop(const Var* output_param,
                                 const std::unordered_map<const Var*, ExprPtr>& callsite_subst,
                                 const std::unordered_set<const Var*>* loop_local_allocs) const {
    if (!loop_local_allocs || loop_local_allocs->empty()) return false;

    auto subst_it = callsite_subst.find(output_param);
    if (subst_it == callsite_subst.end()) return false;

    auto parent_expr = ResolveLoopInitExpr(subst_it->second);
    auto parent_var = AsVarLike(parent_expr);
    if (parent_var) {
      const Var* root = parent_var.get();
      std::unordered_set<const Var*> seen;
      while (seen.insert(root).second) {
        auto alias_it = sibling_output_alias_roots_.find(root);
        if (alias_it == sibling_output_alias_roots_.end()) break;
        root = alias_it->second;
      }
      return loop_local_allocs->count(root);
    }
    return false;
  }

  ExprPtr ResolveLoopInitExpr(const ExprPtr& expr) const {
    ExprPtr current = expr;
    std::unordered_set<const Var*> seen;
    while (auto var = AsVarLike(current)) {
      if (!seen.insert(var.get()).second) break;
      auto it = loop_iter_init_subst_.find(var.get());
      if (it == loop_iter_init_subst_.end()) break;
      current = it->second;
    }
    return current;
  }

  ExprPtr ResolveLoopReturnInitExpr(const ExprPtr& expr) const {
    ExprPtr current = expr;
    std::unordered_set<const Var*> seen;
    while (auto var = AsVarLike(current)) {
      if (!seen.insert(var.get()).second) break;
      auto it = loop_return_init_subst_.find(var.get());
      if (it == loop_return_init_subst_.end()) break;
      current = it->second;
    }
    return current;
  }

  ExprPtr MaterializeWindowParentExpr(const ExprPtr& expr) {
    return VisitExpr(ResolveLoopReturnInitExpr(expr));
  }

  FunctionPtr LookupFunction(const std::string& name) const {
    auto it = function_lookup_.find(name);
    if (it != function_lookup_.end()) return it->second;
    auto clone_it = cloned_funcs_.find(name);
    if (clone_it != cloned_funcs_.end()) return clone_it->second;
    return nullptr;
  }

  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
    auto tuple_var = AsVarLike(op->tuple_);
    if (tuple_var) {
      auto subst_it = tuple_result_subst_.find(tuple_var.get());
      if (subst_it != tuple_result_subst_.end() && op->index_ >= 0 &&
          static_cast<size_t>(op->index_) < subst_it->second.size()) {
        return VisitExpr(subst_it->second[static_cast<size_t>(op->index_)]);
      }
    }
    return IRMutator::VisitExpr_(op);
  }

  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = window_parent_subst_.find(op.get());
    if (it != window_parent_subst_.end()) return VisitExpr(it->second);
    return IRMutator::VisitExpr_(op);
  }

  ProgramPtr program_;
  const AnalysisMap& analyses_;
  const std::unordered_map<std::string, FunctionPtr>& cloned_funcs_;
  const std::unordered_map<std::string, FunctionPtr>& function_lookup_;
  WindowRewriteContext& rewrite_context_;
  std::unordered_set<std::string> used_clone_names_;
  std::vector<ForStmtPtr> sequential_loops_;
  std::vector<LoopContext> loop_context_;
  std::vector<std::unordered_set<const Var*>> loop_local_allocs_;
  std::unordered_map<const Var*, ExprPtr> loop_iter_init_subst_;
  std::unordered_map<const Var*, ExprPtr> loop_return_init_subst_;
  std::unordered_map<const Var*, ExprPtr> scalar_defs_;
  std::unordered_map<const Var*, std::vector<ExprPtr>> tuple_result_subst_;
  std::unordered_map<const Var*, ExprPtr> window_parent_subst_;
  std::unordered_map<const Var*, const Var*> sibling_output_alias_roots_;
  std::unordered_map<const Var*, const Var*> sibling_carrier_alias_roots_;
  std::unordered_set<const Var*> sibling_unwindowable_output_roots_;
  std::unordered_map<std::string, std::vector<OutParamReturnMapping>> out_param_return_mappings_cache_;
  int while_depth_ = 0;
};

}  // namespace

OrchRewriteResult RewriteOrchestrationBody(
    const ProgramPtr& program, const AnalysisMap& analyses,
    const std::unordered_map<std::string, FunctionPtr>& cloned_funcs,
    const std::unordered_map<std::string, FunctionPtr>& function_lookup,
    WindowRewriteContext& rewrite_context, const StmtPtr& body) {
  OrchRewriter rewriter(program, analyses, cloned_funcs, function_lookup, rewrite_context);
  OrchRewriteResult result;
  result.body = rewriter.VisitStmt(body);
  result.used_clone_names = rewriter.used_clone_names();
  return result;
}

}  // namespace window_externalization
}  // namespace ir
}  // namespace pypto

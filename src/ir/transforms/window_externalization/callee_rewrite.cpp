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
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/window_externalization/internal.h"

namespace pypto {
namespace ir {
namespace window_externalization {
using transform_utils::FlattenToStmts;

namespace {

std::string MakeUniqueFunctionName(const ProgramPtr& program, const std::string& base_name) {
  if (!program || !program->GetFunction(base_name)) return base_name;
  for (size_t suffix = 1;; ++suffix) {
    auto candidate = base_name + "_" + std::to_string(suffix);
    if (!program->GetFunction(candidate)) return candidate;
  }
}

/// Count Var/IterArg references to `target` inside an IR node.

std::unordered_map<const Var*, VarPtr> SubstituteFunctionBoundaryTypeExprs(
    std::vector<VarPtr>* params, std::vector<TypePtr>* return_types, StmtPtr* body,
    std::unordered_map<const Var*, ExprPtr>* subst) {
  std::unordered_map<const Var*, VarPtr> rebuilt_param_subst;
  if (!params || !return_types || !body || !subst || subst->empty()) return rebuilt_param_subst;

  std::unordered_map<const Var*, ExprPtr> body_rebuilt_param_subst;
  for (auto& param : *params) {
    auto new_type = SubstituteTypeExprs(param->GetType(), *subst);
    if (new_type.get() == param->GetType().get()) continue;

    auto rebuilt_param = std::make_shared<Var>(param->name_hint_, std::move(new_type), param->span_);
    rebuilt_param_subst[param.get()] = rebuilt_param;
    body_rebuilt_param_subst[param.get()] = rebuilt_param;
    param = std::move(rebuilt_param);
  }

  if (!body_rebuilt_param_subst.empty()) {
    *body = transform_utils::Substitute(*body, body_rebuilt_param_subst);
    // Visit order does not escape: keys are unique, so copying the entries
    // into `subst` yields the same map for any traversal order.
    // NOLINTNEXTLINE(bugprone-nondeterministic-pointer-iteration-order)
    for (const auto& [old_param, new_param] : body_rebuilt_param_subst) {
      (*subst)[old_param] = new_param;
    }
  }

  for (auto& return_type : *return_types) {
    return_type = SubstituteTypeExprs(return_type, *subst);
  }
  return rebuilt_param_subst;
}

class StaticPieceLoopExternalizer : public IRMutator {
 public:
  StaticPieceLoopExternalizer(ForStmtPtr target_loop, std::vector<OutputRewriteInfo> outputs,
                              std::unordered_map<size_t, std::vector<VarPtr>> piece_params_by_old_index,
                              WindowRewriteContext& rewrite_context)
      : target_loop_(std::move(target_loop)),
        outputs_(std::move(outputs)),
        piece_params_by_old_index_(std::move(piece_params_by_old_index)),
        rewrite_context_(rewrite_context) {
    for (size_t output_index = 0; output_index < outputs_.size(); ++output_index) {
      const auto& output = outputs_[output_index];
      output_by_iter_arg_index_[output.iter_arg_index] = output_index;
      if (DensePieces(output).size() > 1) {
        multi_output_by_return_index_[output.return_index] = output_index;
      }
    }
  }

  bool failed() const { return failed_; }
  bool rewrote_loop() const { return rewrote_loop_; }

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = return_var_remap_.find(op.get());
    if (it != return_var_remap_.end()) return it->second;
    return IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (op.get() != target_loop_.get()) return IRMutator::VisitStmt_(op);
    rewrote_loop_ = true;

    auto trip_count = GetKnownPositiveTripCount(op);
    if (!trip_count.has_value() || *trip_count <= 0) return MarkFailed(op);

    std::vector<ExprPtr> current_values;
    current_values.reserve(op->iter_args_.size());
    for (const auto& iter_arg : op->iter_args_) {
      current_values.push_back(iter_arg->initValue_);
    }

    std::vector<StmtPtr> unrolled_stmts;
    for (int64_t trip = 0; trip < *trip_count; ++trip) {
      auto loop_value = GetLoopValueAtTrip(op, trip);
      if (!loop_value.has_value()) return MarkFailed(op);

      for (const auto& [iter_arg_index, output_index] : output_by_iter_arg_index_) {
        if (iter_arg_index >= current_values.size()) return MarkFailed(op);
        const auto& output = outputs_[output_index];
        if (DensePieces(output).size() <= 1) continue;
        auto params_it = piece_params_by_old_index_.find(output.out_param_index);
        if (params_it == piece_params_by_old_index_.end() ||
            static_cast<size_t>(trip) >= params_it->second.size()) {
          return MarkFailed(op);
        }
        current_values[iter_arg_index] = params_it->second[static_cast<size_t>(trip)];
      }

      std::unordered_map<const Var*, ExprPtr> sub_map;
      sub_map[op->loop_var_.get()] = *loop_value;
      for (size_t i = 0; i < op->iter_args_.size(); ++i) {
        sub_map[op->iter_args_[i].get()] = current_values[i];
      }

      auto cloned = DeepClone(op->body_, sub_map);
      auto force_sub_map = sub_map;
      for (size_t i = 0; i < op->iter_args_.size() && i < current_values.size(); ++i) {
        auto cloned_iter_it = cloned.var_map.find(op->iter_args_[i].get());
        if (cloned_iter_it != cloned.var_map.end()) {
          force_sub_map[cloned_iter_it->second.get()] = current_values[i];
        }
      }
      auto iteration_body = ForceSubstituteExprRefs(cloned.cloned_body, force_sub_map);
      auto localized_body = LocalizeIteration(iteration_body, current_values, static_cast<size_t>(trip));
      if (!localized_body.has_value()) return MarkFailed(op);
      auto body_stmts = FlattenToStmts(*localized_body);
      if (body_stmts.empty()) return MarkFailed(op);
      auto yield = As<YieldStmt>(body_stmts.back());
      if (!yield || yield->value_.size() != op->iter_args_.size()) return MarkFailed(op);

      for (size_t i = 0; i + 1 < body_stmts.size(); ++i) {
        unrolled_stmts.push_back(body_stmts[i]);
      }

      for (size_t i = 0; i < yield->value_.size(); ++i) {
        auto output_it = output_by_iter_arg_index_.find(i);
        if (output_it != output_by_iter_arg_index_.end() &&
            DensePieces(outputs_[output_it->second]).size() > 1) {
          final_piece_values_[outputs_[output_it->second].return_index].push_back(yield->value_[i]);
          continue;
        }
        current_values[i] = yield->value_[i];
      }
    }

    for (size_t i = 0; i < op->return_vars_.size() && i < current_values.size(); ++i) {
      return_var_remap_[op->return_vars_[i].get()] = current_values[i];
    }
    return std::make_shared<SeqStmts>(std::move(unrolled_stmts), op->span_);
  }

  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    std::vector<ExprPtr> new_values;
    std::vector<std::pair<size_t, ExprPtr>> extra_piece_values;
    bool changed = false;
    for (size_t i = 0; i < op->value_.size(); ++i) {
      auto multi_it = multi_output_by_return_index_.find(i);
      if (multi_it != multi_output_by_return_index_.end()) {
        const auto& output = outputs_[multi_it->second];
        auto final_it = final_piece_values_.find(output.return_index);
        if (final_it == final_piece_values_.end() || final_it->second.size() != DensePieces(output).size() ||
            output.piece_return_indices.size() != DensePieces(output).size()) {
          return MarkFailed(op);
        }
        new_values.push_back(final_it->second.front());
        for (size_t piece_index = 1; piece_index < final_it->second.size(); ++piece_index) {
          extra_piece_values.emplace_back(output.piece_return_indices[piece_index],
                                          final_it->second[piece_index]);
        }
        changed = true;
        continue;
      }
      auto new_value = VisitExpr(op->value_[i]);
      if (new_value.get() != op->value_[i].get()) changed = true;
      new_values.push_back(new_value);
    }
    std::sort(extra_piece_values.begin(), extra_piece_values.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    for (const auto& [_, value] : extra_piece_values) {
      new_values.push_back(value);
    }
    if (!changed) return op;
    auto result = MutableCopy(op);
    result->value_ = std::move(new_values);
    return result;
  }

 private:
  StmtPtr MarkFailed(const StmtPtr& fallback) {
    failed_ = true;
    return fallback;
  }

  static StmtPtr ForceSubstituteExprRefs(const StmtPtr& stmt,
                                         const std::unordered_map<const Var*, ExprPtr>& replacements) {
    class Replacer : public IRMutator {
     public:
      explicit Replacer(const std::unordered_map<const Var*, ExprPtr>& replacements)
          : replacements_(replacements) {}

     protected:
      ExprPtr VisitExpr_(const VarPtr& op) override {
        auto it = replacements_.find(op.get());
        if (it != replacements_.end()) return it->second;
        return IRMutator::VisitExpr_(op);
      }

      ExprPtr VisitExpr_(const IterArgPtr& op) override {
        auto it = replacements_.find(op.get());
        if (it != replacements_.end()) return it->second;
        return IRMutator::VisitExpr_(op);
      }

     private:
      const std::unordered_map<const Var*, ExprPtr>& replacements_;
    };

    Replacer replacer(replacements);
    return replacer.VisitStmt(stmt);
  }

  std::optional<StmtPtr> LocalizeIteration(const StmtPtr& body, const std::vector<ExprPtr>& current_values,
                                           size_t trip) const {
    std::unordered_map<const Var*, OutputRewriteInfo> out_info_by_var;
    std::unordered_map<const Var*, ExprPtr> new_out_vars;
    for (const auto& [iter_arg_index, output_index] : output_by_iter_arg_index_) {
      if (iter_arg_index >= current_values.size()) return std::nullopt;
      const auto& output = outputs_[output_index];
      const auto& pieces = DensePieces(output);
      const size_t piece_index = pieces.size() > 1 ? trip : 0;
      if (piece_index >= pieces.size()) return std::nullopt;
      auto target_var = AsVarLike(current_values[iter_arg_index]);
      if (!target_var) return std::nullopt;

      OutputRewriteInfo piece_info = output;
      piece_info.window_shape = pieces[piece_index].window_shape;
      piece_info.callsite_offsets = pieces[piece_index].callsite_offsets;
      piece_info.local_store_offsets = pieces[piece_index].local_offsets;
      piece_info.region = MakeDenseRegion({pieces[piece_index]});
      out_info_by_var.emplace(target_var.get(), std::move(piece_info));
      new_out_vars.emplace(target_var.get(), target_var);
    }

    return LocalizeWindowWrites(body, out_info_by_var, new_out_vars, rewrite_context_);
  }

  ForStmtPtr target_loop_;
  std::vector<OutputRewriteInfo> outputs_;
  std::unordered_map<size_t, std::vector<VarPtr>> piece_params_by_old_index_;
  std::unordered_map<size_t, size_t> output_by_iter_arg_index_;
  std::unordered_map<size_t, size_t> multi_output_by_return_index_;
  std::unordered_map<const Var*, ExprPtr> return_var_remap_;
  std::unordered_map<size_t, std::vector<ExprPtr>> final_piece_values_;
  WindowRewriteContext& rewrite_context_;
  bool failed_ = false;
  bool rewrote_loop_ = false;
};

}  // namespace

FunctionPtr RewriteCallee(const ProgramPtr& program, const FunctionPtr& func,
                          const CalleeRewriteAnalysis& analysis, const std::string& clone_suffix,
                          WindowRewriteContext& rewrite_context) {
  if (!func) return nullptr;

  std::vector<VarPtr> new_params;
  new_params.reserve(func->params_.size());
  std::vector<TypePtr> new_return_types = func->return_types_;
  std::vector<ParamDirection> new_param_directions;
  new_param_directions.reserve(func->param_directions_.size());
  std::vector<VarPtr> primary_new_param_by_old_index(func->params_.size());
  std::unordered_map<size_t, std::vector<VarPtr>> output_piece_params_by_old_index;
  std::unordered_map<size_t, std::vector<size_t>> output_piece_return_indices_by_old_index;

  std::unordered_map<const Var*, ExprPtr> seed;
  for (size_t i = 0; i < func->params_.size(); ++i) {
    auto param_type = func->params_[i]->GetType();
    auto rewrite_it = std::find_if(analysis.outputs.begin(), analysis.outputs.end(),
                                   [i](const OutputRewriteInfo& info) { return info.out_param_index == i; });
    if (rewrite_it != analysis.outputs.end()) {
      auto out_tensor_type = As<TensorType>(func->params_[i]->GetType());
      if (!out_tensor_type) return nullptr;
      const auto& pieces = DensePieces(*rewrite_it);
      if (pieces.empty()) return nullptr;
      if (rewrite_it->return_index >= new_return_types.size()) return nullptr;

      std::vector<VarPtr> piece_params;
      std::vector<size_t> piece_return_indices;
      piece_params.reserve(pieces.size());
      piece_return_indices.reserve(pieces.size());
      for (size_t piece_index = 0; piece_index < pieces.size(); ++piece_index) {
        const auto& piece = pieces[piece_index];
        std::vector<ExprPtr> piece_window_shape = piece.window_shape;
        auto piece_type = MakeWindowTensorType(out_tensor_type, rewrite_it->parent_shape, piece_window_shape);
        if (!piece_type) return nullptr;
        auto name_hint = func->params_[i]->name_hint_;
        if (piece_index > 0) name_hint += "_piece" + std::to_string(piece_index);
        auto new_param = std::make_shared<Var>(name_hint, piece_type, func->params_[i]->span_);
        new_params.push_back(new_param);
        new_param_directions.push_back(func->param_directions_[i]);
        piece_params.push_back(new_param);

        size_t piece_return_index = rewrite_it->return_index;
        if (piece_index == 0) {
          new_return_types[piece_return_index] = piece_type;
        } else {
          piece_return_index = new_return_types.size();
          new_return_types.push_back(piece_type);
        }
        piece_return_indices.push_back(piece_return_index);
      }

      primary_new_param_by_old_index[i] = piece_params.front();
      output_piece_params_by_old_index.emplace(i, std::move(piece_params));
      output_piece_return_indices_by_old_index.emplace(i, std::move(piece_return_indices));
      seed[func->params_[i].get()] = primary_new_param_by_old_index[i];
      continue;
    }
    auto input_rewrite_it =
        std::find_if(analysis.inputs.begin(), analysis.inputs.end(),
                     [i](const InputRewriteInfo& info) { return info.in_param_index == i; });
    if (input_rewrite_it != analysis.inputs.end()) {
      auto in_tensor_type = As<TensorType>(func->params_[i]->GetType());
      if (!in_tensor_type) return nullptr;
      const auto& pieces = DensePieces(*input_rewrite_it);
      if (pieces.size() != 1) return nullptr;
      std::vector<ExprPtr> window_shape = pieces.front().window_shape;
      param_type = MakeWindowTensorType(in_tensor_type, input_rewrite_it->parent_shape, window_shape);
      if (!param_type) return nullptr;
    }

    auto new_param = std::make_shared<Var>(func->params_[i]->name_hint_, param_type, func->params_[i]->span_);
    new_params.push_back(new_param);
    new_param_directions.push_back(func->param_directions_[i]);
    primary_new_param_by_old_index[i] = new_param;
    seed[func->params_[i].get()] = new_param;
  }

  auto cloned_name = MakeUniqueFunctionName(program, func->name_ + clone_suffix);
  auto cloned = DeepClone(func->body_, seed);
  std::unordered_map<const Var*, ExprPtr> body_subst = seed;
  for (const auto& [old_var, new_var] : cloned.var_map) {
    body_subst[old_var] = new_var;
  }
  StmtPtr new_body = cloned.cloned_body;
  auto rebuilt_param_subst =
      SubstituteFunctionBoundaryTypeExprs(&new_params, &new_return_types, &new_body, &body_subst);
  auto remap_rebuilt_param = [&](VarPtr* var) {
    if (!var || !*var || rebuilt_param_subst.empty()) return;
    auto it = rebuilt_param_subst.find(var->get());
    if (it != rebuilt_param_subst.end()) {
      *var = it->second;
    }
  };
  for (auto& param : primary_new_param_by_old_index) {
    remap_rebuilt_param(&param);
  }
  for (auto& [_, params] : output_piece_params_by_old_index) {
    for (auto& param : params) {
      remap_rebuilt_param(&param);
    }
  }
  std::vector<OutputRewriteInfo> localized_outputs = analysis.outputs;
  for (auto& output : localized_outputs) {
    auto return_it = output_piece_return_indices_by_old_index.find(output.out_param_index);
    if (return_it != output_piece_return_indices_by_old_index.end()) {
      output.piece_return_indices = return_it->second;
    }
    for (auto& offset : output.callsite_offsets) {
      offset = transform_utils::Substitute(offset, body_subst);
    }
    for (auto& offset : output.local_store_offsets) {
      offset = transform_utils::Substitute(offset, body_subst);
    }
    for (auto& dim : output.window_shape) {
      dim = transform_utils::Substitute(dim, body_subst);
    }
    for (auto& piece : output.region.dense_pieces) {
      for (auto& dim : piece.window_shape) {
        dim = transform_utils::Substitute(dim, body_subst);
      }
      for (auto& offset : piece.callsite_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
      for (auto& offset : piece.local_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
    }
  }
  std::vector<InputRewriteInfo> localized_inputs = analysis.inputs;
  for (auto& input : localized_inputs) {
    for (auto& offset : input.callsite_offsets) {
      offset = transform_utils::Substitute(offset, body_subst);
    }
    for (auto& offset : input.local_read_offsets) {
      offset = transform_utils::Substitute(offset, body_subst);
    }
    for (auto& piece : input.region.dense_pieces) {
      for (auto& dim : piece.window_shape) {
        dim = transform_utils::Substitute(dim, body_subst);
      }
      for (auto& offset : piece.callsite_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
      for (auto& offset : piece.local_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
    }
  }
  if (analysis.kind == RewriteKind::AggregateWindowLoop) {
    auto find_aggregate_loop = [&](const StmtPtr& body) -> ForStmtPtr {
      auto body_stmts = FlattenToStmts(body);
      auto ret_stmt = body_stmts.empty() ? nullptr : As<ReturnStmt>(body_stmts.back());
      if (!ret_stmt) return nullptr;

      ForStmtPtr matched_loop;
      for (const auto& stmt : body_stmts) {
        auto candidate = As<ForStmt>(stmt);
        if (!candidate) continue;

        bool matches_outputs = true;
        for (const auto& output : analysis.outputs) {
          if (output.iter_arg_index >= candidate->iter_args_.size() ||
              output.iter_arg_index >= candidate->return_vars_.size() ||
              output.return_index >= ret_stmt->value_.size()) {
            matches_outputs = false;
            break;
          }
          auto init_var = AsVarLike(candidate->iter_args_[output.iter_arg_index]->initValue_);
          auto returned = AsVarLike(ret_stmt->value_[output.return_index]);
          if (!init_var || !returned) {
            matches_outputs = false;
            break;
          }
          auto direct_return_param = output.out_param_index < primary_new_param_by_old_index.size()
                                         ? primary_new_param_by_old_index[output.out_param_index]
                                         : nullptr;
          if (output.out_param_index >= primary_new_param_by_old_index.size() ||
              init_var.get() != primary_new_param_by_old_index[output.out_param_index].get() ||
              (returned.get() != candidate->return_vars_[output.iter_arg_index].get() &&
               (!direct_return_param || returned.get() != direct_return_param.get()))) {
            matches_outputs = false;
            break;
          }
        }
        if (!matches_outputs) continue;
        if (matched_loop) return nullptr;
        matched_loop = candidate;
      }
      return matched_loop;
    };

    auto cloned_loop = find_aggregate_loop(new_body);
    if (!cloned_loop) return nullptr;

    std::unordered_map<const Var*, TypePtr> narrowed_return_vars;
    for (const auto& output : analysis.outputs) {
      if (output.iter_arg_index >= cloned_loop->return_vars_.size()) {
        return nullptr;
      }
      narrowed_return_vars.emplace(cloned_loop->return_vars_[output.iter_arg_index].get(),
                                   new_return_types[output.return_index]);
    }

    class AggregateLoopTypeLocalizer : public IRMutator {
     public:
      explicit AggregateLoopTypeLocalizer(const std::unordered_map<const Var*, TypePtr>& narrowed_return_vars)
          : narrowed_return_vars_(narrowed_return_vars) {}

     protected:
      StmtPtr VisitStmt_(const ForStmtPtr& op) override {
        std::vector<const Var*> old_iter_args_to_erase;
        for (size_t i = 0; i < op->return_vars_.size() && i < op->iter_args_.size(); ++i) {
          auto it = narrowed_return_vars_.find(op->return_vars_[i].get());
          if (it == narrowed_return_vars_.end()) continue;
          auto old_iter = op->iter_args_[i];
          auto old_ret = op->return_vars_[i];
          auto new_iter = std::make_shared<IterArg>(old_iter->name_hint_, it->second, old_iter->initValue_,
                                                    old_iter->span_);
          auto new_ret = std::make_shared<Var>(old_ret->name_hint_, it->second, old_ret->span_);
          var_remap_[old_iter.get()] = new_iter;
          var_remap_[old_ret.get()] = new_ret;
          old_iter_args_to_erase.push_back(old_iter.get());
        }
        auto new_stmt = IRMutator::VisitStmt_(op);
        for (const auto* old_iter : old_iter_args_to_erase) {
          var_remap_.erase(old_iter);
        }
        return new_stmt;
      }

     private:
      const std::unordered_map<const Var*, TypePtr>& narrowed_return_vars_;
    };

    AggregateLoopTypeLocalizer type_localizer(narrowed_return_vars);
    new_body = type_localizer.VisitStmt(new_body);

    auto typed_loop = find_aggregate_loop(new_body);
    if (!typed_loop) return nullptr;

    if (std::any_of(localized_outputs.begin(), localized_outputs.end(),
                    [](const OutputRewriteInfo& output) { return DensePieces(output).size() > 1; })) {
      StaticPieceLoopExternalizer static_piece_externalizer(
          typed_loop, localized_outputs, output_piece_params_by_old_index, rewrite_context);
      new_body = static_piece_externalizer.VisitStmt(new_body);
      if (static_piece_externalizer.failed() || !static_piece_externalizer.rewrote_loop()) {
        return nullptr;
      }
    } else {
      std::unordered_map<const Var*, OutputRewriteInfo> out_info_by_var;
      std::unordered_map<const Var*, ExprPtr> new_out_vars;
      for (const auto& output : localized_outputs) {
        if (output.iter_arg_index >= typed_loop->iter_args_.size()) {
          return nullptr;
        }
        auto iter_arg = typed_loop->iter_args_[output.iter_arg_index];
        out_info_by_var.emplace(iter_arg.get(), output);
        new_out_vars.emplace(iter_arg.get(), iter_arg);
      }

      new_body = LocalizeWindowWrites(new_body, out_info_by_var, new_out_vars, rewrite_context);
    }
  } else {
    std::unordered_map<const Var*, OutputRewriteInfo> out_info_by_var;
    std::unordered_map<const Var*, ExprPtr> new_out_vars;
    for (const auto& output : localized_outputs) {
      if (output.out_param_index >= primary_new_param_by_old_index.size()) {
        return nullptr;
      }
      auto new_out = primary_new_param_by_old_index[output.out_param_index];
      out_info_by_var.emplace(new_out.get(), output);
      new_out_vars.emplace(new_out.get(), new_out);
    }
    new_body = LocalizeWindowWrites(new_body, out_info_by_var, new_out_vars, rewrite_context);
  }

  std::unordered_map<const Var*, InputRewriteInfo> in_info_by_var;
  for (const auto& input : localized_inputs) {
    if (input.in_param_index >= primary_new_param_by_old_index.size()) {
      return nullptr;
    }
    in_info_by_var.emplace(primary_new_param_by_old_index[input.in_param_index].get(), input);
  }
  if (!in_info_by_var.empty()) {
    new_body = LocalizeWindowReads(new_body, in_info_by_var, rewrite_context);
  }

  return std::make_shared<Function>(cloned_name, new_params, new_param_directions, new_return_types, new_body,
                                    func->span_, func->func_type_, func->level_, func->role_, func->attrs_,
                                    func->requires_runtime_binding_, func->ir_stage_);
}
}  // namespace window_externalization
}  // namespace ir
}  // namespace pypto

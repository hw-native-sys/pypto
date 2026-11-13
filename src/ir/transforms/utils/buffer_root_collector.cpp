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

#include "pypto/ir/transforms/utils/buffer_root_collector.h"

#include <algorithm>
#include <cstddef>
#include <optional>
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
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace buffer_root {
namespace {

using op_predicates::IsBuiltinOp;

}  // namespace

BufferRootCollector::BufferRootCollector(ProgramPtr program, AmbiguousRootPolicy ambiguous_policy,
                                         bool follow_reshape,
                                         const ReturnedRootCandidates* returned_candidates)
    : program_(std::move(program)),
      ambiguous_policy_(ambiguous_policy),
      follow_reshape_(follow_reshape),
      returned_candidates_(returned_candidates) {}

void BufferRootCollector::Initialize(const std::vector<VarPtr>& params) {
  for (const auto& param : params) {
    RecordRootCandidates(param, {param.get()});
  }
}

void BufferRootCollector::VisitStmt_(const IfStmtPtr& if_stmt) {
  IRVisitor::VisitStmt_(if_stmt);
  if (if_stmt->return_vars_.empty() || !if_stmt->else_body_.has_value()) return;

  auto then_yield = transform_utils::GetLastYieldStmt(if_stmt->then_body_);
  auto else_yield = transform_utils::GetLastYieldStmt(*if_stmt->else_body_);
  if (!then_yield || !else_yield) return;

  for (size_t i = 0;
       i < if_stmt->return_vars_.size() && i < then_yield->value_.size() && i < else_yield->value_.size();
       ++i) {
    if (returned_candidates_ && As<TupleType>(if_stmt->return_vars_[i]->GetType())) {
      tuple_joins_.emplace(if_stmt->return_vars_[i].get(),
                           std::make_pair(then_yield->value_[i], else_yield->value_[i]));
      continue;
    }
    auto roots = ResolveRootCandidates(then_yield->value_[i]);
    auto else_roots = ResolveRootCandidates(else_yield->value_[i]);
    for (const Var* root : else_roots) {
      if (std::find(roots.begin(), roots.end(), root) == roots.end()) roots.push_back(root);
    }
    const bool unresolved = roots.empty() || IsAmbiguous(then_yield->value_[i]) ||
                            IsAmbiguous(else_yield->value_[i]) ||
                            ResolveRootCandidates(then_yield->value_[i]).empty() || else_roots.empty();
    RecordRootCandidates(if_stmt->return_vars_[i], roots, unresolved);
  }
}

void BufferRootCollector::VisitStmt_(const ForStmtPtr& for_stmt) {
  auto init_roots = InitializeLoopCarryRoots(for_stmt->iter_args_);
  IRVisitor::VisitStmt_(for_stmt);
  RecordLoopReturnRoots(for_stmt->body_, for_stmt->return_vars_, init_roots,
                        transform_utils::EvalConstTripCount(for_stmt).value_or(0) > 0);
}

void BufferRootCollector::VisitStmt_(const WhileStmtPtr& while_stmt) {
  auto init_roots = InitializeLoopCarryRoots(while_stmt->iter_args_);
  IRVisitor::VisitStmt_(while_stmt);
  // A while condition may be false before the first iteration, so both the
  // initial and yielded roots remain possible.
  RecordLoopReturnRoots(while_stmt->body_, while_stmt->return_vars_, init_roots,
                        /*guaranteed_to_run=*/false);
}

std::vector<BufferRootCollector::RootCandidates> BufferRootCollector::InitializeLoopCarryRoots(
    const std::vector<IterArgPtr>& iter_args) {
  std::vector<RootCandidates> init_roots;
  init_roots.reserve(iter_args.size());
  for (const auto& iter_arg : iter_args) {
    auto roots = ResolveRootCandidates(iter_arg->initValue_);
    init_roots.push_back(roots);
    RecordRootCandidates(iter_arg, roots, IsAmbiguous(iter_arg->initValue_));
  }
  return init_roots;
}

void BufferRootCollector::RecordLoopReturnRoots(const StmtPtr& body, const std::vector<VarPtr>& return_vars,
                                                const std::vector<RootCandidates>& init_roots,
                                                bool guaranteed_to_run) {
  // Resolve after visiting the body so fresh-rebind assignments already have
  // roots. An in-place yield of the IterArg still resolves to its init root.
  auto yield = transform_utils::GetLastYieldStmt(body);
  if (!yield) return;

  for (size_t i = 0; i < return_vars.size() && i < yield->value_.size(); ++i) {
    auto roots = ResolveRootCandidates(yield->value_[i]);
    bool unresolved = roots.empty() || IsAmbiguous(yield->value_[i]);
    if (!guaranteed_to_run && i < init_roots.size()) {
      if (init_roots[i].empty()) unresolved = true;
      for (const Var* root : init_roots[i]) {
        if (std::find(roots.begin(), roots.end(), root) == roots.end()) roots.push_back(root);
      }
    }
    RecordRootCandidates(return_vars[i], roots, unresolved);
  }
}

void BufferRootCollector::VisitStmt_(const AssignStmtPtr& assign) {
  // Submit (pl.submit inside a pl.manual_scope) is a sibling call-like kind;
  // route it through the Call-shaped view so a task launch's Out/InOut roots are
  // tracked identically to a plain Call (results keyed on the stable binding
  // Var). The view preserves args_ and the TASK_ID-augmented return type, so the
  // tuple path maps the submit-result projections to the callee's Out roots and
  // leaves the trailing TASK_ID element unmapped
  // (see .claude/rules/pass-submit-awareness.md).
  if (auto call = transform_utils::AsCallOrSubmitView(assign->value_)) {
    const std::string& op_name = call->op_->name_;
    if (IsOp(call, "tensor.create") || (!returned_candidates_ && IsOp(call, "tensor.slice"))) {
      RecordRootCandidates(assign->var_, {assign->var_.get()});
    } else if (IsOp(call, "tensor.assemble") && call->args_.size() == 3) {
      RecordRootCandidates(assign->var_, ResolveRootCandidates(call->args_[0]), IsAmbiguous(call->args_[0]));
    } else if (!call->args_.empty() &&
               ((follow_reshape_ && IsOp(call, "tensor.reshape")) ||
                (returned_candidates_ && (IsOp(call, "tensor.view") || IsOp(call, "tensor.transpose") ||
                                          IsOp(call, "tensor.slice"))))) {
      ForwardReshapeCandidates(assign->var_, call->args_[0]);
    } else if (!IsBuiltinOp(op_name)) {
      if (As<TupleType>(call->GetType()) && ambiguous_policy_ == AmbiguousRootPolicy::kTraceReturn) {
        tuple_calls_[assign->var_.get()] = call;
      } else {
        auto out_roots = CollectCallOutputRoots(call);
        if (As<TupleType>(call->GetType())) {
          std::vector<RootInfo> roots;
          roots.reserve(out_roots.size());
          for (auto& entry : out_roots) roots.push_back(std::move(entry.info));
          tuple_output_roots_[assign->var_.get()] = std::move(roots);
        } else {
          auto info = SelectReturnRootInfo(out_roots, call);
          if (!info.roots.empty()) {
            RecordRootCandidates(assign->var_, info.roots, info.ambiguous);
          }
        }
      }
    }
  } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
    if (auto tuple_var = AsVarLike(tuple_get->tuple_); tuple_var && tuple_get->index_ >= 0) {
      const auto position = static_cast<size_t>(tuple_get->index_);
      if (auto lazy = tuple_calls_.find(tuple_var.get()); lazy != tuple_calls_.end()) {
        auto info = ResolveCallTupleField(lazy->second, position);
        RecordRootCandidates(assign->var_, info.roots, info.ambiguous);
      } else if (returned_candidates_ &&
                 (tuple_aliases_.count(tuple_var.get()) || tuple_joins_.count(tuple_var.get()))) {
        auto [roots, ambiguous] = ResolveReturnedValue(tuple_var, position);
        RecordRootCandidates(assign->var_, roots, ambiguous);
      } else if (auto it = tuple_output_roots_.find(tuple_var.get());
                 it != tuple_output_roots_.end() && position < it->second.size()) {
        const auto& info = it->second[position];
        RecordRootCandidates(assign->var_, info.roots, info.ambiguous);
      }
    }
  } else if (returned_candidates_ && As<TupleType>(assign->var_->GetType()) &&
             (As<MakeTuple>(assign->value_) || AsVarLike(assign->value_))) {
    auto var = AsVarLike(assign->value_);
    auto source = var ? tuple_aliases_.find(var.get()) : tuple_aliases_.end();
    tuple_aliases_[assign->var_.get()] = source != tuple_aliases_.end() ? source->second : assign->value_;
  } else if (AsVarLike(assign->value_)) {
    RecordRootCandidates(assign->var_, ResolveRootCandidates(assign->value_), IsAmbiguous(assign->value_));
  }
  IRVisitor::VisitStmt_(assign);
}

void BufferRootCollector::VisitStmt_(const SimtLaunchStmtPtr& launch) {
  // GM results are new dependency versions of the exact same reference. Tile
  // results have value semantics and are materialized separately before planning.
  for (size_t i = 0; i < launch->results_.size(); ++i) {
    if (!AsTensorTypeLike(launch->results_[i]->GetType())) continue;
    const auto& argument = launch->args_[launch->result_param_indices_[i]];
    RecordRootCandidates(launch->results_[i], ResolveRootCandidates(argument), IsAmbiguous(argument));
  }
  IRVisitor::VisitStmt_(launch);
}

std::optional<BufferRootCollector::RootInfo> BufferRootCollector::ResolveReturnedCandidates(
    const CallPtr& call, size_t position) const {
  if (!returned_candidates_) return std::nullopt;
  const auto callee = program_->GetFunction(call->op_->name_);
  if (!callee) return std::nullopt;
  const auto fields = returned_candidates_->find(callee.get());
  if (fields == returned_candidates_->end()) return std::nullopt;
  const auto candidates = fields->second.find(position);
  if (candidates == fields->second.end()) return std::nullopt;
  RootInfo result{{}, true};
  for (size_t index : candidates->second) {
    if (index >= call->args_.size()) continue;
    if (auto arg = AsVarLike(call->args_[index])) {
      const auto& roots = RootCandidatesOf(arg.get());
      result.roots.insert(result.roots.end(), roots.begin(), roots.end());
    }
  }
  return result;
}

std::pair<BufferRootCollector::RootCandidates, bool> BufferRootCollector::ResolveReturnedValue(
    const ExprPtr& expr, size_t position) const {
  ExprPtr value = expr;
  while (value) {
    if (auto tuple = As<MakeTuple>(value)) {
      if (position >= tuple->elements_.size()) return {{}, false};
      value = tuple->elements_[position];
      position = 0;
      continue;
    }
    if (auto call = As<Call>(value)) {
      if (position == 0 && !call->args_.empty() &&
          (IsOp(call, "tensor.reshape") || IsOp(call, "tensor.view") || IsOp(call, "tensor.transpose") ||
           IsOp(call, "tensor.slice"))) {
        value = call->args_[0];
        continue;
      }
      if (auto info = ResolveReturnedCandidates(call, position)) {
        return {std::move(info->roots), info->ambiguous};
      }
      return {{}, false};
    }
    auto var = AsVarLike(value);
    if (!var) return {{}, false};
    if (auto alias = tuple_aliases_.find(var.get()); alias != tuple_aliases_.end()) {
      value = alias->second;
      continue;
    }
    if (auto join = tuple_joins_.find(var.get()); join != tuple_joins_.end()) {
      auto& fields = tuple_join_fields_[var.get()];
      if (auto field = fields.find(position); field != fields.end()) {
        return {field->second.roots, field->second.ambiguous};
      }
      auto [then_roots, then_ambiguous] = ResolveReturnedValue(join->second.first, position);
      auto [else_roots, else_ambiguous] = ResolveReturnedValue(join->second.second, position);
      const bool ambiguous = then_ambiguous || else_ambiguous || then_roots != else_roots;
      std::unordered_set<const Var*> seen(then_roots.begin(), then_roots.end());
      for (const auto* root : else_roots) {
        if (seen.insert(root).second) then_roots.push_back(root);
      }
      fields.emplace(position, RootInfo{then_roots, ambiguous});
      return {std::move(then_roots), ambiguous};
    }
    if (auto call = tuple_calls_.find(var.get()); call != tuple_calls_.end()) {
      auto info = ResolveCallTupleField(call->second, position);
      return {std::move(info.roots), info.ambiguous};
    }
    if (position == 0 && AsTensorTypeLike(var->GetType())) {
      return {RootCandidatesOf(var.get()), ambiguous_buffer_vars.count(var.get()) != 0};
    }
    return {{}, false};
  }
  return {{}, false};
}

std::unordered_map<size_t, std::vector<size_t>> BufferRootCollector::CollectAmbiguousReturnedParams(
    const FunctionPtr& func) const {
  std::unordered_map<size_t, std::vector<size_t>> results;
  if (!func->body_ || func->HasAttr(kAttrExternalSource)) return results;
  const auto returns = return_lineage::CollectReturns(func->body_);
  if (returns.size() != 1) return results;
  std::unordered_map<const Var*, size_t> indices;
  for (size_t i = 0; i < func->params_.size(); ++i) indices.emplace(func->params_[i].get(), i);
  const auto record = [&](const ExprPtr& value, size_t position, size_t field) {
    auto [roots, ambiguous] = ResolveReturnedValue(value, field);
    if (!ambiguous) return;
    std::unordered_set<size_t> seen;
    for (const auto* root : roots) {
      auto index = indices.find(root);
      if (index != indices.end() && seen.insert(index->second).second) {
        results[position].push_back(index->second);
      }
    }
  };
  const auto& values = returns.front()->value_;
  if (values.size() == 1) {
    if (auto tuple = As<TupleType>(values.front()->GetType())) {
      for (size_t i = 0; i < tuple->types_.size(); ++i) {
        if (AsTensorTypeLike(tuple->types_[i])) record(values.front(), i, i);
      }
      return results;
    }
  }
  for (size_t i = 0; i < values.size(); ++i) {
    if (AsTensorTypeLike(values[i]->GetType())) record(values[i], i, 0);
  }
  return results;
}

BufferRootCollector::RootInfo BufferRootCollector::ResolveCallTupleField(const CallPtr& call,
                                                                         size_t position) const {
  const auto tuple = As<TupleType>(call->GetType());
  // Only tensor positions can name storage. This also excludes Submit's TaskId.
  if (position >= tuple->types_.size() || !AsTensorTypeLike(tuple->types_[position])) return {{}, false};
  const auto callee = program_->GetFunction(call->op_->name_);
  if (!callee) return {{}, false};
  if (auto possible = ResolveReturnedCandidates(call, position)) return std::move(*possible);
  std::optional<size_t> index;
  if (!callee->HasAttr(kAttrExternalSource)) {
    const auto& returned = return_lineage::CachedReturnedStorageParamIndices(callee, program_);
    if (!returned.empty()) {
      if (position >= returned.size() || !returned[position]) return {{}, false};
      index = returned[position];
    }
  }
  if (!index) {
    auto [outputs, inserted] = writable_param_indices_.try_emplace(callee.get());
    if (inserted) {
      for (size_t i = 0; i < callee->param_directions_.size(); ++i) {
        if (callee->param_directions_[i] != ParamDirection::In) outputs->second.push_back(i);
      }
    }
    if (position < outputs->second.size()) index = outputs->second[position];
  }
  if (!index || *index >= call->args_.size()) return {{}, false};
  const auto& arg = call->args_[*index];
  return {ResolveRootCandidates(arg), IsAmbiguous(arg)};
}

BufferRootCollector::RootCandidates BufferRootCollector::ResolveRootCandidates(const ExprPtr& expr) const {
  auto var = AsVarLike(expr);
  if (!var) return {};
  return RootCandidatesOf(var.get());
}

void BufferRootCollector::ForwardReshapeCandidates(const VarPtr& var, const ExprPtr& source) {
  auto source_var = AsVarLike(source);
  if (!var || !source_var) return;
  const Var* key = source_var.get();
  auto alias = reshape_candidate_sources_.find(key);
  if (alias != reshape_candidate_sources_.end()) key = alias->second;
  if (root_candidates_.find(key) == root_candidates_.end()) return;
  reshape_candidate_sources_[var.get()] = key;
  auto root = buffer_roots.find(source_var.get());
  if (root != buffer_roots.end()) buffer_roots[var.get()] = root->second;
  if (ambiguous_buffer_vars.count(source_var.get())) ambiguous_buffer_vars.insert(var.get());
}

bool BufferRootCollector::IsAmbiguous(const ExprPtr& expr) const {
  auto var = AsVarLike(expr);
  return var && ambiguous_buffer_vars.count(var.get()) != 0;
}

void BufferRootCollector::RecordRootCandidates(const VarPtr& var, const RootCandidates& roots,
                                               bool unresolved) {
  if (!var) return;
  RootCandidates unique;
  unique.reserve(roots.size());
  std::unordered_set<const Var*> seen;
  for (const Var* root : roots) {
    if (root && seen.insert(root).second) unique.push_back(root);
  }
  if (unique.empty()) return;

  root_candidates_[var.get()] = unique;
  const bool ambiguous = unresolved || unique.size() > 1;
  if (ambiguous) {
    ambiguous_buffer_vars.insert(var.get());
    if (ambiguous_policy_ == AmbiguousRootPolicy::kFirstOutput) {
      buffer_roots[var.get()] = unique.front();
    } else {
      buffer_roots.erase(var.get());
    }
  } else {
    ambiguous_buffer_vars.erase(var.get());
    buffer_roots[var.get()] = unique.front();
  }
}

std::vector<BufferRootCollector::OutputRoot> BufferRootCollector::CollectCallOutputRoots(
    const CallPtr& call) const {
  auto callee = program_->GetFunction(call->op_->name_);
  if (!callee) return {};

  std::vector<OutputRoot> roots;
  for (size_t i = 0; i < callee->param_directions_.size() && i < call->args_.size(); ++i) {
    if (callee->param_directions_[i] != ParamDirection::Out &&
        callee->param_directions_[i] != ParamDirection::InOut) {
      continue;
    }
    roots.push_back(OutputRoot{RootInfo{ResolveRootCandidates(call->args_[i]), IsAmbiguous(call->args_[i])},
                               call->args_[i]->GetType()});
  }
  return roots;
}

BufferRootCollector::RootInfo BufferRootCollector::SelectReturnRootInfo(
    const std::vector<OutputRoot>& out_roots, const CallPtr& call) const {
  if (ambiguous_policy_ == AmbiguousRootPolicy::kTraceReturn) {
    if (auto possible = ResolveReturnedCandidates(call, 0)) return std::move(*possible);
    const auto callee = program_->GetFunction(call->op_->name_);
    if (callee && !callee->HasAttr(kAttrExternalSource)) {
      const auto& returned = return_lineage::CachedReturnedStorageParamIndices(callee, program_);
      if (returned.size() == 1) {
        if (auto index = returned.front(); index && *index < call->args_.size()) {
          const auto& arg = call->args_[*index];
          return {ResolveRootCandidates(arg), IsAmbiguous(arg)};
        }
      }
    }
  }
  if (out_roots.empty()) return RootInfo{{}, false};
  if (out_roots.size() == 1) return out_roots[0].info;

  const OutputRoot* match = nullptr;
  bool ambiguous = false;
  for (const auto& candidate : out_roots) {
    if (!candidate.info.roots.empty() && TypesMatchShapeDtype(candidate.type, call->GetType())) {
      if (match == nullptr) {
        match = &candidate;
      } else if (match->info.roots != candidate.info.roots ||
                 match->info.ambiguous != candidate.info.ambiguous) {
        ambiguous = true;
      }
    }
  }
  if (match && !ambiguous) return match->info;
  // No provable unambiguous type match (0 matches, or >1 distinct candidates).
  // The fallback depends on what the consumer needs when the owning buffer
  // can't be pinned down:
  //   kSkip        — record no root. Fusion / aliasing is an optimization, so
  //                  skipping it (no root -> no aliasing) is always safe,
  //                  whereas guessing could re-alias a scratch onto the output.
  //   kFirstOutput — fall back to the first Out/InOut root, matching the naive
  //                  pre-dedup behavior. DeriveCallDirections needs *some* root
  //                  so a later write to the returned var still promotes to
  //                  InOut; a null root would silently drop the WAW/InOut dep.
  if (ambiguous_policy_ == AmbiguousRootPolicy::kFirstOutput) {
    return out_roots[0].info;
  }
  return RootInfo{{}, false};
}

bool BufferRootCollector::TypesMatchShapeDtype(const TypePtr& a, const TypePtr& b) {
  auto ta = As<TensorType>(a);
  auto tb = As<TensorType>(b);
  if (!ta || !tb) return false;
  if (ta->dtype_ != tb->dtype_) return false;
  if (ta->shape_.size() != tb->shape_.size()) return false;
  for (size_t i = 0; i < ta->shape_.size(); ++i) {
    if (!AreExprsEqual(ta->shape_[i], tb->shape_[i])) return false;
  }
  return true;
}

}  // namespace buffer_root
}  // namespace ir
}  // namespace pypto

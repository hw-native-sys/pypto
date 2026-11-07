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
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/result_alias_utils.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace buffer_root {
namespace {

using op_predicates::IsBuiltinOp;

std::optional<size_t> BackingAliasSlot(const CallPtr& call) {
  if (auto slot = ResultAliasedArgIndex(call)) return slot;
  if (IsOp(call, "tensor.slice")) return 0;
  const auto* entry = LookupOpEntry(call->op_);
  // A view needs explicit registry evidence of descriptor-only access.
  if (entry && !call->args_.empty() &&
      entry->GetExecutionMemoryAccessEvidence() == ExecutionMemoryAccessEvidence::NoAccess &&
      op_predicates::IsBufferAliasingViewOp(call->op_->name_)) {
    return 0;
  }
  return std::nullopt;
}

// Keys retain their IR until the outer analysis exits, avoiding pointer reuse.
// Program identity matters because the same Function can resolve different callees.
using StorageKey = std::pair<ProgramPtr, FunctionPtr>;
struct StorageReturnContext {
  size_t depth = 0;
  std::map<StorageKey, std::vector<std::vector<size_t>>> memo;
  std::set<StorageKey> active;
};

StorageReturnContext& StorageCtx() {
  thread_local StorageReturnContext ctx;
  return ctx;
}

}  // namespace

ScopedStorageRootCache::ScopedStorageRootCache() { ++StorageCtx().depth; }

ScopedStorageRootCache::~ScopedStorageRootCache() {
  auto& ctx = StorageCtx();
  if (--ctx.depth == 0) {
    ctx.memo.clear();
    ctx.active.clear();
  }
}

BufferRootCollector::BufferRootCollector(ProgramPtr program, AmbiguousRootPolicy ambiguous_policy,
                                         ViewRootPolicy view_policy)
    : storage_cache_(view_policy == ViewRootPolicy::kBackingStorage
                         ? std::make_unique<ScopedStorageRootCache>()
                         : nullptr),
      program_(std::move(program)),
      ambiguous_policy_(ambiguous_policy),
      view_policy_(view_policy) {}

void BufferRootCollector::Initialize(const std::vector<VarPtr>& params) {
  for (const auto& param : params) {
    RecordRootCandidates(param, {param.get()});
  }
}

void BufferRootCollector::VisitStmt_(const IfStmtPtr& if_stmt) {
  IRVisitor::VisitStmt_(if_stmt);
  RecordIfReturnRoots(if_stmt);
}

void BufferRootCollector::RecordIfReturnRoots(const IfStmtPtr& if_stmt) {
  const auto& else_body = if_stmt->else_body_;
  if (if_stmt->return_vars_.empty() || !else_body.has_value()) return;

  auto then_yield = transform_utils::GetLastYieldStmt(if_stmt->then_body_);
  auto else_yield = transform_utils::GetLastYieldStmt(else_body.value());
  if (!then_yield || !else_yield) return;

  for (size_t i = 0;
       i < if_stmt->return_vars_.size() && i < then_yield->value_.size() && i < else_yield->value_.size();
       ++i) {
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
  const auto trip_count = transform_utils::EvalConstTripCount(for_stmt);
  RecordLoopReturnRoots(for_stmt->body_, for_stmt->return_vars_, init_roots, trip_count.value_or(0) > 0,
                        !trip_count || *trip_count > 1);
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
    if (view_policy_ == ViewRootPolicy::kBackingStorage) {
      // Keep a symbolic identity until all repeated-iteration edges are known.
      // Eager expansion would freeze a descriptor swap to its first iteration.
      storage_successors_[iter_arg.get()] = std::move(roots);
      init_roots.push_back({iter_arg.get()});
      RecordRootCandidates(iter_arg, {iter_arg.get()});
    } else {
      init_roots.push_back(roots);
      RecordRootCandidates(iter_arg, roots, IsAmbiguous(iter_arg->initValue_));
    }
  }
  return init_roots;
}

void BufferRootCollector::RecordLoopReturnRoots(const StmtPtr& body, const std::vector<VarPtr>& return_vars,
                                                const std::vector<RootCandidates>& init_roots,
                                                bool guaranteed_to_run, bool may_repeat) {
  // Resolve after visiting the body so fresh-rebind assignments already have
  // roots. An in-place yield of the IterArg still resolves to its init root.
  auto yield = transform_utils::GetLastYieldStmt(body);
  if (!yield) return;

  if (view_policy_ == ViewRootPolicy::kBackingStorage) {
    for (size_t i = 0; i < return_vars.size() && i < yield->value_.size(); ++i) {
      auto roots = ResolveRootCandidates(yield->value_[i]);
      auto& result_roots = storage_successors_[return_vars[i].get()];
      result_roots = roots;
      RecordRootCandidates(return_vars[i], {return_vars[i].get()});
      if (i >= init_roots.size()) continue;
      if (!guaranteed_to_run) {
        result_roots.insert(result_roots.end(), init_roots[i].begin(), init_roots[i].end());
      }
      if (may_repeat) {
        for (const Var* carry : init_roots[i]) {
          auto& next = storage_successors_[carry];
          next.insert(next.end(), roots.begin(), roots.end());
        }
      }
    }
    return;
  }

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
  RecordAssignmentRoots(assign);
  IRVisitor::VisitStmt_(assign);
}

void BufferRootCollector::RecordAssignmentRoots(const AssignStmtPtr& assign) {
  // Submit (pl.submit inside a pl.manual_scope) is a sibling call-like kind;
  // route it through the Call-shaped view so a task launch's Out/InOut roots are
  // tracked identically to a plain Call (results keyed on the stable binding
  // Var). The view preserves args_ and the TASK_ID-augmented return type, so the
  // tuple path maps the submit-result projections to the callee's Out roots and
  // leaves the trailing TASK_ID element unmapped
  // (see .claude/rules/pass-submit-awareness.md).
  bool recorded = false;
  if (auto call = transform_utils::AsCallOrSubmitView(assign->value_)) {
    const std::string& op_name = call->op_->name_;
    std::optional<size_t> alias_slot;
    if (view_policy_ == ViewRootPolicy::kBackingStorage) {
      alias_slot = BackingAliasSlot(call);
    }
    if (alias_slot && *alias_slot < call->args_.size()) {
      RecordRootCandidates(assign->var_, ResolveRootCandidates(call->args_[*alias_slot]),
                           IsAmbiguous(call->args_[*alias_slot]));
      recorded = true;
    } else if (IsOp(call, "tensor.create") || IsOp(call, "tensor.slice")) {
      RecordRootCandidates(assign->var_, {assign->var_.get()});
      recorded = true;
    } else if (IsOp(call, "tensor.assemble")) {
      if (call->args_.size() == 3) {
        RecordRootCandidates(assign->var_, ResolveRootCandidates(call->args_[0]),
                             IsAmbiguous(call->args_[0]));
        recorded = true;
      }
    } else if (!IsBuiltinOp(op_name)) {
      auto out_roots = CollectCallOutputRoots(call);
      if (As<TupleType>(call->GetType())) {
        std::vector<RootInfo> roots;
        roots.reserve(out_roots.size());
        for (const auto& entry : out_roots) roots.push_back(entry.info);
        tuple_output_roots_[assign->var_.get()] = std::move(roots);
      } else {
        auto info = SelectReturnRootInfo(out_roots, call->GetType());
        RecordRootCandidates(assign->var_, info.roots, info.ambiguous);
        recorded = true;
      }
    }
  } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
    if (auto tuple_var = AsVarLike(tuple_get->tuple_)) {
      auto it = tuple_output_roots_.find(tuple_var.get());
      if (it != tuple_output_roots_.end() && tuple_get->index_ < static_cast<int>(it->second.size()) &&
          !it->second[tuple_get->index_].roots.empty()) {
        const auto& info = it->second[tuple_get->index_];
        RecordRootCandidates(assign->var_, info.roots, info.ambiguous);
        recorded = true;
      }
    }
  } else if (AsVarLike(assign->value_)) {
    RecordRootCandidates(assign->var_, ResolveRootCandidates(assign->value_), IsAmbiguous(assign->value_));
    recorded = true;
  }
  if (!recorded && view_policy_ == ViewRootPolicy::kBackingStorage) {
    // Also supports a pre-SSA name rebound to fresh/unknown storage.
    RecordRootCandidates(assign->var_, {});
  }
}

void BufferRootCollector::RecordSimtLaunchRoots(const SimtLaunchStmtPtr& launch) {
  // GM results are new dependency versions of the exact same reference. Tile
  // results have value semantics and are materialized separately before planning.
  for (size_t i = 0; i < launch->results_.size(); ++i) {
    if (!AsTensorTypeLike(launch->results_[i]->GetType())) continue;
    const auto& argument = launch->args_[launch->result_param_indices_[i]];
    RecordRootCandidates(launch->results_[i], ResolveRootCandidates(argument), IsAmbiguous(argument));
  }
}

void BufferRootCollector::VisitStmt_(const SimtLaunchStmtPtr& launch) {
  RecordSimtLaunchRoots(launch);
  IRVisitor::VisitStmt_(launch);
}

BufferRootCollector::RootCandidates BufferRootCollector::ResolveRootCandidates(const ExprPtr& expr) const {
  auto var = AsVarLike(expr);
  if (!var) return {};
  auto it = root_candidates_.find(var.get());
  return it != root_candidates_.end() ? it->second : RootCandidates{};
}

BufferRootCollector::RootCandidates BufferRootCollector::ResolveBackingDescriptorRoots(
    const ExprPtr& expr) const {
  auto value = expr;
  while (auto call = As<Call>(value)) {
    const auto slot = BackingAliasSlot(call);
    if (!slot || *slot >= call->args_.size()) return {};
    value = call->args_[*slot];
  }
  if (auto projection = As<TupleGetItemExpr>(value)) {
    if (auto tuple = AsVarLike(projection->tuple_)) {
      auto it = tuple_output_roots_.find(tuple.get());
      const auto index = static_cast<size_t>(projection->index_);
      if (it != tuple_output_roots_.end() && index < it->second.size()) return it->second[index].roots;
    }
    return {};
  }
  return ResolveRootCandidates(value);
}

BufferRootCollector::RootCandidates BufferRootCollector::ResolveStorageLeaves(const Var* var) const {
  RootCandidates pending = RootCandidatesOf(var);
  RootCandidates leaves;
  std::unordered_set<const Var*> seen;
  for (size_t cursor = 0; cursor < pending.size(); ++cursor) {
    const Var* root = pending[cursor];
    if (!seen.insert(root).second) continue;
    auto next = storage_successors_.find(root);
    if (next == storage_successors_.end() || next->second.empty()) {
      leaves.push_back(root);
    } else {
      pending.insert(pending.end(), next->second.begin(), next->second.end());
    }
  }
  return leaves;
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
  if (unique.empty()) {
    if (view_policy_ == ViewRootPolicy::kBackingStorage) {
      root_candidates_.erase(var.get());
      buffer_roots.erase(var.get());
      ambiguous_buffer_vars.erase(var.get());
    }
    return;
  }

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
  if (!program_) return {};
  auto callee = program_->GetFunction(call->op_->name_);
  if (!callee) return {};

  if (view_policy_ == ViewRootPolicy::kBackingStorage) return CollectProvenReturnRoots(call);

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

std::vector<BufferRootCollector::OutputRoot> BufferRootCollector::CollectProvenReturnRoots(
    const CallPtr& call) const {
  const auto callee = program_->GetFunction(call->op_->name_);
  INTERNAL_CHECK(callee) << "Storage return analysis requires a resolved callee";
  if (callee->return_types_.empty()) return {};
  auto& ctx = StorageCtx();
  const StorageKey key{program_, callee};
  auto cached = ctx.memo.find(key);
  if (cached == ctx.memo.end()) {
    std::vector<std::vector<size_t>> returned(callee->return_types_.size());
    if (ctx.active.insert(key).second) {
      // Reuse the general return-binding utility, but collect *backing* roots:
      // its complete-parameter writeback test intentionally excludes partial views.
      BufferRootCollector nested(program_, AmbiguousRootPolicy::kSkip, ViewRootPolicy::kBackingStorage);
      nested.Initialize(callee->params_);
      if (callee->body_) nested.VisitStmt(callee->body_);
      std::unordered_map<const Var*, size_t> positions;
      for (size_t i = 0; i < callee->params_.size(); ++i) positions.emplace(callee->params_[i].get(), i);
      const auto bindings = return_lineage::AnalyzeReturnedValues(callee, program_);
      for (size_t i = 0; i < bindings.size() && i < returned.size(); ++i) {
        if (!bindings[i].value) {
          const auto& parameter = bindings[i].param_index;
          if (parameter.has_value()) returned[i].push_back(parameter.value());
          continue;
        }
        auto value = AsVarLike(bindings[i].value);
        if (!value) {
          value = std::make_shared<Var>("storage_return", bindings[i].type, call->span_);
          nested.RecordAssignmentRoots(std::make_shared<AssignStmt>(value, bindings[i].value, call->span_));
        }
        for (const Var* root : nested.ResolveStorageLeaves(value.get())) {
          auto parameter = positions.find(root);
          if (parameter != positions.end()) returned[i].push_back(parameter->second);
        }
      }
      ctx.active.erase(key);
    } else {
      // Cyclic user call graphs are diagnosed by the pass. Until then, unknown
      // recursion must not erase a possible captured-storage dependency.
      for (auto& result : returned) {
        for (size_t i = 0; i < callee->params_.size(); ++i) result.push_back(i);
      }
    }
    cached = ctx.memo.emplace(key, std::move(returned)).first;
  }

  std::vector<OutputRoot> roots;
  const size_t ctx_count = transform_utils::TrailingCommCtxParamCount(callee);
  const size_t prefix = call->args_.size() >= ctx_count ? call->args_.size() - ctx_count : 0;
  const size_t gap =
      callee->params_.size() >= call->args_.size() ? callee->params_.size() - call->args_.size() : 0;
  for (size_t i = 0; i < cached->second.size(); ++i) {
    RootInfo info{{}, false};
    std::unordered_set<const Var*> seen;
    for (const size_t parameter : cached->second[i]) {
      if (parameter >= prefix && parameter < callee->params_.size() - ctx_count) continue;
      const size_t argument = parameter < prefix ? parameter : parameter - gap;
      if (argument >= call->args_.size()) continue;
      for (const Var* root : ResolveRootCandidates(call->args_[argument])) {
        if (seen.insert(root).second) info.roots.push_back(root);
      }
      info.ambiguous = info.ambiguous || IsAmbiguous(call->args_[argument]);
    }
    info.ambiguous = info.ambiguous || info.roots.size() > 1;
    roots.push_back(OutputRoot{std::move(info), callee->return_types_[i]});
  }
  return roots;
}

BufferRootCollector::RootInfo BufferRootCollector::SelectReturnRootInfo(
    const std::vector<OutputRoot>& out_roots, const TypePtr& return_type) const {
  if (out_roots.empty()) return RootInfo{{}, false};
  if (out_roots.size() == 1) return out_roots[0].info;

  const OutputRoot* match = nullptr;
  bool ambiguous = false;
  for (const auto& candidate : out_roots) {
    if (!candidate.info.roots.empty() && TypesMatchShapeDtype(candidate.type, return_type)) {
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

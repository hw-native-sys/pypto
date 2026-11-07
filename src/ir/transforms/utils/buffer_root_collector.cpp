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
#include "pypto/ir/transforms/structural_comparison.h"
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
struct StorageReturnValue {
  std::vector<size_t> roots;
  std::vector<StorageReturnValue> elements;
};

struct StorageSummaryNode {
  TypePtr type;
  std::optional<size_t> parameter;
  std::vector<size_t> successors;
};

// Each return refers to nodes in the same graph. A loop that rotates N
// descriptors stores N transitions, rather than N sets of N parameters.
struct StorageReturnSummary {
  std::vector<StorageSummaryNode> nodes;
  std::vector<StorageReturnValue> outputs;
};

struct StorageReturnContext {
  size_t depth = 0;
  std::map<StorageKey, StorageReturnSummary> memo;
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
    auto roots = view_policy_ == ViewRootPolicy::kBackingStorage
                     ? ResolveBackingDescriptorRoots(iter_arg->initValue_)
                     : ResolveRootCandidates(iter_arg->initValue_);
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
      auto roots = ResolveBackingDescriptorRoots(yield->value_[i]);
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
  if (view_policy_ == ViewRootPolicy::kBackingStorage && As<TupleType>(assign->value_->GetType())) {
    // Capture the RHS before replacing a pre-SSA tuple binding.
    auto value = ResolveBackingValue(assign->value_);
    if (value.elements) {
      backing_tuple_roots_[assign->var_.get()] = std::move(value.elements);
    } else {
      backing_tuple_roots_.erase(assign->var_.get());
    }
    RecordRootCandidates(assign->var_, {});
    return;
  }
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
  } else if (view_policy_ == ViewRootPolicy::kBackingStorage && As<TupleGetItemExpr>(assign->value_)) {
    auto info = ResolveBackingValue(assign->value_);
    RecordRootCandidates(assign->var_, info.roots, info.ambiguous);
    recorded = true;
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

BufferRootCollector::RootCandidates BufferRootCollector::ResolveBackingDescriptorRoots(const ExprPtr& expr) {
  return ResolveBackingValue(expr).roots;
}

BufferRootCollector::RootInfo BufferRootCollector::ResolveBackingValue(const ExprPtr& expr) {
  if (auto var = AsVarLike(expr)) {
    auto it = backing_tuple_roots_.find(var.get());
    if (it != backing_tuple_roots_.end()) return {{}, false, it->second};
    return {RootCandidatesOf(var.get()), IsAmbiguous(expr), nullptr};
  }
  if (auto projection = As<TupleGetItemExpr>(expr)) {
    auto tuple = ResolveBackingValue(projection->tuple_);
    const auto index = static_cast<size_t>(projection->index_);
    if (tuple.elements && index < tuple.elements->size()) return (*tuple.elements)[index];
  }
  if (auto tuple = As<MakeTuple>(expr)) {
    auto elements = std::make_shared<std::vector<RootInfo>>();
    elements->reserve(tuple->elements_.size());
    for (const auto& element : tuple->elements_) elements->push_back(ResolveBackingValue(element));
    return {{}, false, elements};
  }
  if (auto call = transform_utils::AsCallOrSubmitView(expr)) {
    if (auto slot = BackingAliasSlot(call); slot && *slot < call->args_.size()) {
      return ResolveBackingValue(call->args_[*slot]);
    }
    if (!IsBuiltinOp(call->op_->name_)) return BackingCallValue(call);
  }
  return {};
}

BufferRootCollector::RootInfo BufferRootCollector::BackingCallValue(const CallPtr& call) {
  auto outputs = CollectCallOutputRoots(call);
  if (!As<TupleType>(call->GetType())) return SelectReturnRootInfo(outputs, call->GetType());
  // A plain call returning one tuple preserves that tuple. A Submit wraps it
  // in the launch result, whose trailing TaskId has no storage roots.
  if (outputs.size() == 1 && structural_equal(outputs[0].type, call->GetType())) {
    return std::move(outputs[0].info);
  }
  auto elements = std::make_shared<std::vector<RootInfo>>();
  elements->reserve(outputs.size());
  for (auto& output : outputs) elements->push_back(std::move(output.info));
  return {{}, false, elements};
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
    const CallPtr& call) {
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
    const CallPtr& call) {
  const auto callee = program_->GetFunction(call->op_->name_);
  INTERNAL_CHECK(callee) << "Storage return analysis requires a resolved callee";
  if (callee->return_types_.empty()) return {};
  auto& ctx = StorageCtx();
  const StorageKey key{program_, callee};
  auto cached = ctx.memo.find(key);
  StorageReturnSummary recursive;
  const StorageReturnSummary* summary = nullptr;
  if (cached != ctx.memo.end()) {
    summary = &cached->second;
  } else if (ctx.active.insert(key).second) {
    StorageReturnSummary built;
    built.outputs.resize(callee->return_types_.size());
    BufferRootCollector nested(program_, AmbiguousRootPolicy::kSkip, ViewRootPolicy::kBackingStorage);
    nested.Initialize(callee->params_);
    if (callee->body_) nested.VisitStmt(callee->body_);

    std::unordered_map<const Var*, size_t> positions;
    for (size_t i = 0; i < callee->params_.size(); ++i) positions.emplace(callee->params_[i].get(), i);
    std::unordered_map<const Var*, size_t> node_ids;
    RootCandidates pending;
    const auto intern = [&](const Var* root) {
      auto [it, inserted] = node_ids.try_emplace(root, built.nodes.size());
      if (inserted) {
        auto parameter = positions.find(root);
        built.nodes.push_back(
            {root->GetType(),
             parameter == positions.end() ? std::nullopt : std::optional<size_t>{parameter->second},
             {}});
        pending.push_back(root);
      }
      return it->second;
    };
    const auto summarize = [&](const auto& self, const RootInfo& value) -> StorageReturnValue {
      StorageReturnValue result;
      if (value.elements) {
        result.elements.reserve(value.elements->size());
        for (const auto& element : *value.elements) result.elements.push_back(self(self, element));
      }
      for (const Var* root : value.roots) result.roots.push_back(intern(root));
      return result;
    };
    const auto bindings = return_lineage::AnalyzeReturnedValues(callee, program_);
    for (size_t i = 0; i < bindings.size() && i < built.outputs.size(); ++i) {
      const auto& binding = bindings[i];
      if (binding.value) {
        built.outputs[i] = summarize(summarize, nested.ResolveBackingValue(binding.value));
      } else if (binding.param_index) {
        built.outputs[i].roots.push_back(intern(callee->params_[*binding.param_index].get()));
      }
    }
    for (size_t i = 0; i < pending.size(); ++i) {
      auto next = nested.storage_successors_.find(pending[i]);
      if (next == nested.storage_successors_.end()) continue;
      std::vector<size_t> edges;
      edges.reserve(next->second.size());
      for (const Var* root : next->second) edges.push_back(intern(root));
      built.nodes[i].successors = std::move(edges);
    }
    ctx.active.erase(key);
    summary = &ctx.memo.emplace(key, std::move(built)).first->second;
  } else {
    // Recursive call graphs are diagnosed separately. Conservatively connect
    // the results to a shared union of the arguments until then.
    recursive.outputs.resize(callee->return_types_.size());
    recursive.nodes.reserve(callee->params_.size() + 1);
    for (size_t i = 0; i < callee->params_.size(); ++i) {
      recursive.nodes.push_back({callee->params_[i]->GetType(), i, {}});
    }
    std::vector<size_t> edges;
    edges.reserve(recursive.nodes.size());
    for (size_t i = 0; i < recursive.nodes.size(); ++i) edges.push_back(i);
    const size_t union_id = recursive.nodes.size();
    recursive.nodes.push_back({callee->return_types_[0], std::nullopt, std::move(edges)});
    for (auto& output : recursive.outputs) output.roots.push_back(union_id);
    summary = &recursive;
  }

  const size_t ctx_count = transform_utils::TrailingCommCtxParamCount(callee);
  const size_t prefix = call->args_.size() >= ctx_count ? call->args_.size() - ctx_count : 0;
  const size_t gap =
      callee->params_.size() >= call->args_.size() ? callee->params_.size() - call->args_.size() : 0;
  std::vector<RootInfo> bound(summary->nodes.size());
  for (size_t i = 0; i < summary->nodes.size(); ++i) {
    const auto& node = summary->nodes[i];
    if (node.parameter) {
      const size_t parameter = *node.parameter;
      if (parameter >= prefix && parameter < callee->params_.size() - ctx_count) continue;
      const size_t argument = parameter < prefix ? parameter : parameter - gap;
      if (argument < call->args_.size()) bound[i] = ResolveBackingValue(call->args_[argument]);
    } else if (!node.successors.empty()) {
      auto proxy = std::make_shared<Var>("storage_transition", node.type, call->span_);
      bound[i].roots.push_back(proxy.get());
      storage_proxy_vars_.push_back(std::move(proxy));
    }
  }
  for (size_t i = 0; i < summary->nodes.size(); ++i) {
    const auto& node = summary->nodes[i];
    if (node.parameter || node.successors.empty()) continue;
    auto& edges = storage_successors_[bound[i].roots.front()];
    for (size_t next : node.successors) {
      const auto& roots = bound[next].roots;
      edges.insert(edges.end(), roots.begin(), roots.end());
    }
  }
  const auto bind = [&](const auto& self, const StorageReturnValue& result) -> RootInfo {
    RootInfo info;
    if (!result.elements.empty()) {
      auto elements = std::make_shared<std::vector<RootInfo>>();
      elements->reserve(result.elements.size());
      for (const auto& element : result.elements) elements->push_back(self(self, element));
      info.elements = elements;
    }
    std::unordered_set<const Var*> seen;
    for (size_t root : result.roots) {
      const auto& actual = bound[root];
      if (actual.elements) info.elements = actual.elements;
      for (const Var* candidate : actual.roots) {
        if (seen.insert(candidate).second) info.roots.push_back(candidate);
      }
      info.ambiguous = info.ambiguous || actual.ambiguous;
    }
    info.ambiguous = info.ambiguous || info.roots.size() > 1;
    return info;
  };
  std::vector<OutputRoot> roots;
  roots.reserve(summary->outputs.size());
  for (size_t i = 0; i < summary->outputs.size(); ++i) {
    roots.push_back(OutputRoot{bind(bind, summary->outputs[i]), callee->return_types_[i]});
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

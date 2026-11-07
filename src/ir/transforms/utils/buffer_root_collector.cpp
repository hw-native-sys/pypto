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
#include <functional>
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
#include "pypto/ir/span.h"
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
  std::shared_ptr<std::vector<StorageReturnValue>> elements;
  std::optional<size_t> tuple_sources;
};

struct StorageTupleSources {
  std::shared_ptr<const TupleType> type;
  std::vector<StorageReturnValue> sources;
};

// A transfer is a shared union of formal roots. Retaining the union graph
// avoids materializing an increasingly large formal set at every output.
struct StorageFormalTransfer {
  size_t id;
  size_t first;
  size_t split;
  const StorageFormalTransfer* left = nullptr;
  const StorageFormalTransfer* right = nullptr;
};

struct StorageReturnSummary;

struct StorageSummaryCall {
  std::shared_ptr<const StorageReturnSummary> callee;
  std::vector<StorageReturnValue> arguments;
  mutable std::unordered_map<const StorageFormalTransfer*, size_t> mapped_transfers;
};

struct StorageSummaryNode {
  TypePtr type;
  std::optional<size_t> parameter;
  std::vector<size_t> successors;
  // (local call index, node in the callee graph)
  std::optional<std::pair<size_t, size_t>> projection = std::nullopt;
};

struct StorageMappedTransfer {
  size_t call;
  const StorageFormalTransfer* transfer;
};

// Each summary owns its local transitions and transfers. A call binds a
// callee's shared transfer to its arguments, using one local node per binding
// and callee transfer. Caches live as long as the enclosing summary.
struct StorageReturnSummary {
  std::vector<StorageSummaryNode> nodes;
  std::vector<std::pair<size_t, size_t>> formal_bindings;
  std::vector<StorageReturnValue> outputs;
  std::vector<StorageSummaryCall> calls;
  // Shared tuple values retain their source edges, including loop cycles.
  std::vector<StorageTupleSources> tuples;
  mutable std::vector<StorageMappedTransfer> mapped_nodes;
  mutable std::unordered_map<size_t, const StorageFormalTransfer*> formal_transfers;
  mutable std::vector<std::unique_ptr<const StorageFormalTransfer>> transfer_storage;
  mutable std::unordered_map<size_t, const StorageFormalTransfer*> formal_leaves;
  mutable std::map<std::pair<size_t, size_t>, const StorageFormalTransfer*> transfer_branches;
  mutable std::map<std::pair<size_t, size_t>, const StorageFormalTransfer*> transfer_unions;

  const StorageFormalTransfer* FormalLeaf(size_t parameter) const {
    auto [leaf, inserted] = formal_leaves.try_emplace(parameter);
    if (inserted) {
      leaf->second = OwnTransfer({transfer_storage.size(), parameter, 0});
    }
    return leaf->second;
  }

  const StorageFormalTransfer* TransferBranch(size_t split, const StorageFormalTransfer* left,
                                              const StorageFormalTransfer* right) const {
    if (!left) return right;
    if (!right) return left;
    auto [branch, inserted] = transfer_branches.try_emplace(std::pair{left->id, right->id});
    if (inserted) {
      branch->second = OwnTransfer({transfer_storage.size(), left->first, split, left, right});
    }
    return branch->second;
  }

  const StorageFormalTransfer* OwnTransfer(StorageFormalTransfer transfer) const {
    transfer_storage.push_back(std::make_unique<const StorageFormalTransfer>(transfer));
    return transfer_storage.back().get();
  }
};

// Canonical persistent sets of formal indices. The split bit is the
// highest differing bit in a branch. Swapping bindings or regrouping a
// union produces the same identity, so callers share equivalent transfers.
const StorageFormalTransfer* UnionStorageTransfers(const StorageReturnSummary& summary,
                                                   const StorageFormalTransfer* lhs,
                                                   const StorageFormalTransfer* rhs) {
  if (!lhs || lhs == rhs) return rhs;
  if (!rhs) return lhs;
  // Interning the result alone still repeats the recursive merge when
  // successive output joins share the same pair of input subtrees.
  const auto operands = std::minmax(lhs->id, rhs->id);
  const auto [memo, inserted] = summary.transfer_unions.try_emplace(operands);
  if (!inserted) {
    return memo->second;
  }
  size_t differing_bit = lhs->first ^ rhs->first;
  // Clear lower set bits until only the highest differing bit remains.
  while ((differing_bit & (differing_bit - 1)) != 0) {
    differing_bit &= differing_bit - 1;
  }
  const size_t split = std::max({lhs->split, rhs->split, differing_bit});
  const auto children = [split](const StorageFormalTransfer* transfer) {
    if (transfer->split == split) return std::pair{transfer->left, transfer->right};
    const bool right = (transfer->first & split) != 0;
    return std::pair{right ? nullptr : transfer, right ? transfer : nullptr};
  };
  const auto [left_lhs, right_lhs] = children(lhs);
  const auto [left_rhs, right_rhs] = children(rhs);
  const auto* left = UnionStorageTransfers(summary, left_lhs, left_rhs);
  const auto* right = UnionStorageTransfers(summary, right_lhs, right_rhs);
  memo->second = summary.TransferBranch(split, left, right);
  return memo->second;
}

const StorageFormalTransfer* GetStorageTransfer(const StorageReturnSummary& summary, size_t result);

size_t MapStorageTransfer(const StorageReturnSummary& summary, size_t call,
                          const StorageFormalTransfer* transfer) {
  auto& mapped = summary.calls[call].mapped_transfers;
  auto [it, inserted] = mapped.try_emplace(transfer, summary.nodes.size() + summary.mapped_nodes.size());
  if (inserted) summary.mapped_nodes.push_back({call, transfer});
  return it->second;
}

std::vector<size_t> StorageTransferSuccessors(const StorageReturnSummary& summary, size_t index) {
  if (index < summary.nodes.size()) {
    const auto& node = summary.nodes[index];
    auto successors = node.successors;
    if (node.projection) {
      const auto [call, result] = *node.projection;
      if (const auto* transfer = GetStorageTransfer(*summary.calls[call].callee, result)) {
        successors.push_back(MapStorageTransfer(summary, call, transfer));
      }
    }
    return successors;
  }

  // Mapping the inputs can grow mapped_nodes, so keep this entry by value.
  const auto [call, transfer] = summary.mapped_nodes[index - summary.nodes.size()];
  if (transfer->split == 0) return summary.calls[call].arguments[transfer->first].roots;
  return {MapStorageTransfer(summary, call, transfer->left),
          MapStorageTransfer(summary, call, transfer->right)};
}

const StorageFormalTransfer* GetStorageTransfer(const StorageReturnSummary& summary, size_t result) {
  auto& transfers = summary.formal_transfers;
  if (auto it = transfers.find(result); it != transfers.end()) {
    return it->second;
  }

  // Iterative Tarjan over newly reached local nodes. Completed components
  // already have transfers, and every loop component shares a single union.
  struct SearchNode {
    size_t order;
    size_t lowlink;
    std::vector<size_t> successors;
  };
  std::unordered_map<size_t, SearchNode> search;
  std::vector<size_t> active;
  std::vector<std::pair<size_t, size_t>> frames;
  const auto enter = [&](size_t index) {
    const auto order = search.size();
    search.emplace(index, SearchNode{order, order, StorageTransferSuccessors(summary, index)});
    active.push_back(index);
    frames.emplace_back(index, 0);
  };
  enter(result);
  while (!frames.empty()) {
    const size_t index = frames.back().first;
    auto& current = search.at(index);
    if (frames.back().second < current.successors.size()) {
      const size_t next = current.successors[frames.back().second++];
      if (transfers.count(next)) continue;
      if (auto it = search.find(next); it != search.end()) {
        current.lowlink = std::min(current.lowlink, it->second.order);
      } else {
        enter(next);
      }
      continue;
    }

    if (current.order == current.lowlink) {
      std::vector<size_t> members;
      const StorageFormalTransfer* transfer = nullptr;
      std::unordered_set<const StorageFormalTransfer*> seen;
      while (true) {
        const size_t member = active.back();
        active.pop_back();
        members.push_back(member);
        if (member < summary.nodes.size()) {
          if (const auto parameter = summary.nodes[member].parameter; parameter.has_value()) {
            transfer = UnionStorageTransfers(summary, transfer, summary.FormalLeaf(*parameter));
          }
        }
        for (size_t next : search.at(member).successors) {
          if (auto it = transfers.find(next);
              it != transfers.end() && it->second && seen.insert(it->second).second) {
            transfer = UnionStorageTransfers(summary, transfer, it->second);
          }
        }
        if (member == index) break;
      }
      for (size_t member : members) transfers.emplace(member, transfer);
    }

    frames.pop_back();
    if (!frames.empty()) {
      auto& parent = search.at(frames.back().first);
      parent.lowlink = std::min(parent.lowlink, current.lowlink);
    }
  }
  return transfers.at(result);
}

struct StorageReturnContext {
  size_t depth = 0;
  std::map<StorageKey, std::shared_ptr<const StorageReturnSummary>> memo;
  std::set<StorageKey> active;
};

StorageReturnContext& StorageCtx() {
  thread_local StorageReturnContext ctx;
  return ctx;
}

}  // namespace

struct BufferRootCollector::StorageCallInstance {
  std::shared_ptr<const StorageReturnSummary> callee;
  std::vector<RootInfo> arguments;
};

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
    const auto& result = if_stmt->return_vars_[i];
    const auto& then_value = then_yield->value_[i];
    const auto& else_value = else_yield->value_[i];
    if (view_policy_ == ViewRootPolicy::kBackingStorage) {
      RecordBackingValue(result,
                         MergeBackingValues(ResolveBackingValue(then_value), ResolveBackingValue(else_value),
                                            result->GetType(), result->span_));
      continue;
    }

    auto then_roots = ResolveRootCandidates(then_value);
    auto else_roots = ResolveRootCandidates(else_value);
    const bool unresolved =
        then_roots.empty() || else_roots.empty() || IsAmbiguous(then_value) || IsAmbiguous(else_value);
    then_roots.insert(then_roots.end(), else_roots.begin(), else_roots.end());
    RecordRootCandidates(result, then_roots, unresolved);
  }
}

BufferRootCollector::RootInfo BufferRootCollector::MergeBackingValues(const RootInfo& lhs,
                                                                      const RootInfo& rhs,
                                                                      const TypePtr& type, const Span& span) {
  if (lhs.loop_tuple || rhs.loop_tuple) {
    if (lhs.loop_tuple && lhs.loop_tuple == rhs.loop_tuple) return lhs;
    auto joined = MakeLoopStorageValue(type, span, nullptr);
    LinkLoopStorageValues(joined, lhs);
    LinkLoopStorageValues(joined, rhs);
    return joined;
  }
  if (lhs.elements || rhs.elements) {
    if (lhs.elements == rhs.elements) return lhs;
    const auto tuple = As<TupleType>(type);
    INTERNAL_CHECK(tuple) << "Storage join with tuple elements must have TupleType";
    auto elements = std::make_shared<std::vector<RootInfo>>();
    const RootInfo unknown;
    elements->reserve(tuple->types_.size());
    for (size_t i = 0; i < tuple->types_.size(); ++i) {
      const auto& left = lhs.elements && i < lhs.elements->size() ? (*lhs.elements)[i] : unknown;
      const auto& right = rhs.elements && i < rhs.elements->size() ? (*rhs.elements)[i] : unknown;
      elements->push_back(MergeBackingValues(left, right, tuple->types_[i], span));
    }
    return {{}, false, std::move(elements)};
  }

  auto roots = lhs.roots;
  std::unordered_set<const Var*> seen(roots.begin(), roots.end());
  for (const Var* root : rhs.roots) {
    if (seen.insert(root).second) roots.push_back(root);
  }
  const bool ambiguous =
      lhs.roots.empty() || rhs.roots.empty() || lhs.ambiguous || rhs.ambiguous || roots.size() > 1;
  if (roots.size() <= 1) return {std::move(roots), ambiguous};

  // One symbolic edge set per join. A later join references this identity
  // without copying its transitive alternatives.
  auto join = std::make_shared<Var>("storage_branch_join", type, span);
  const Var* root = join.get();
  storage_successors_[root] = std::move(roots);
  storage_proxy_vars_.push_back(std::move(join));
  return {{root}, true};
}

void BufferRootCollector::VisitStmt_(const ForStmtPtr& for_stmt) {
  auto init_roots = InitializeLoopCarryRoots(for_stmt->iter_args_);
  IRVisitor::VisitStmt_(for_stmt);
  const auto trip_count = transform_utils::EvalConstTripCount(for_stmt);
  RecordLoopReturnRoots(for_stmt->body_, for_stmt->iter_args_, for_stmt->return_vars_, init_roots,
                        trip_count.value_or(0) > 0, !trip_count || *trip_count > 1);
}

void BufferRootCollector::VisitStmt_(const WhileStmtPtr& while_stmt) {
  auto init_roots = InitializeLoopCarryRoots(while_stmt->iter_args_);
  IRVisitor::VisitStmt_(while_stmt);
  // A while condition may be false before the first iteration, so both the
  // initial and yielded roots remain possible.
  RecordLoopReturnRoots(while_stmt->body_, while_stmt->iter_args_, while_stmt->return_vars_, init_roots,
                        /*guaranteed_to_run=*/false);
}

BufferRootCollector::RootInfo BufferRootCollector::MakeLoopStorageValue(const TypePtr& type, const Span& span,
                                                                        const Var* identity) {
  if (const auto tuple = As<TupleType>(type)) {
    auto roots = std::make_unique<LoopTupleRoots>(LoopTupleRoots{tuple, span, {}, {}});
    auto* ptr = roots.get();
    loop_tuple_roots_.push_back(std::move(roots));
    return {{}, false, nullptr, ptr};
  }
  if (identity) return {{identity}, false};
  auto element = std::make_shared<Var>("storage_carry_element", type, span);
  const Var* root = element.get();
  storage_proxy_vars_.push_back(std::move(element));
  return {{root}, false};
}

void BufferRootCollector::LinkLoopStorageValues(const RootInfo& target, const RootInfo& source) {
  if (target.loop_tuple) {
    auto& tuple = *target.loop_tuple;
    if (source.loop_tuple == &tuple || (!source.elements && !source.loop_tuple)) return;
    tuple.sources.push_back(source);
    if (tuple.aggregate) {
      const auto roots = CollectBackingContentRoots(source);
      auto& successors = storage_successors_[tuple.aggregate];
      successors.insert(successors.end(), roots.begin(), roots.end());
    }
    for (const auto& [index, child] : tuple.projected) {
      LinkLoopStorageValues(child, ProjectBackingElement(source, index));
    }
    return;
  }
  if (target.roots.empty() || source.roots.empty()) return;
  auto& successors = storage_successors_[target.roots.front()];
  successors.insert(successors.end(), source.roots.begin(), source.roots.end());
}

void BufferRootCollector::RecordBackingValue(const VarPtr& var, const RootInfo& value) {
  if (value.elements || value.loop_tuple) {
    backing_tuple_roots_[var.get()] = value;
  } else {
    backing_tuple_roots_.erase(var.get());
  }
  RecordRootCandidates(var, value.roots, value.ambiguous);
}

std::vector<BufferRootCollector::RootCandidates> BufferRootCollector::InitializeLoopCarryRoots(
    const std::vector<IterArgPtr>& iter_args) {
  std::vector<RootCandidates> init_roots;
  if (view_policy_ == ViewRootPolicy::kBackingStorage) {
    for (const auto& iter_arg : iter_args) {
      // Scalar and tensor carries get a root immediately. Tuple carries
      // allocate roots only for fields that the analysis projects.
      auto carry = MakeLoopStorageValue(iter_arg->GetType(), iter_arg->span_, iter_arg.get());
      LinkLoopStorageValues(carry, ResolveBackingValue(iter_arg->initValue_));
      RecordBackingValue(iter_arg, carry);
    }
    return init_roots;
  }

  init_roots.reserve(iter_args.size());
  for (const auto& iter_arg : iter_args) {
    auto roots = ResolveRootCandidates(iter_arg->initValue_);
    RecordRootCandidates(iter_arg, roots, IsAmbiguous(iter_arg->initValue_));
    init_roots.push_back(std::move(roots));
  }
  return init_roots;
}

void BufferRootCollector::RecordLoopReturnRoots(const StmtPtr& body, const std::vector<IterArgPtr>& iter_args,
                                                const std::vector<VarPtr>& return_vars,
                                                const std::vector<RootCandidates>& init_roots,
                                                bool guaranteed_to_run, bool may_repeat) {
  // Resolve after visiting the body so fresh-rebind assignments already have
  // roots. An in-place yield of the IterArg still resolves to its init root.
  auto yield = transform_utils::GetLastYieldStmt(body);
  if (!yield) return;

  if (view_policy_ == ViewRootPolicy::kBackingStorage) {
    for (size_t i = 0; i < return_vars.size() && i < yield->value_.size(); ++i) {
      const auto& var = return_vars[i];
      auto yielded = ResolveBackingValue(yield->value_[i]);
      auto result = MakeLoopStorageValue(var->GetType(), var->span_, var.get());
      LinkLoopStorageValues(result, yielded);
      if (i < iter_args.size()) {
        auto carry = ResolveBackingValue(iter_args[i]);
        if (!guaranteed_to_run) LinkLoopStorageValues(result, carry);
        if (may_repeat) LinkLoopStorageValues(carry, yielded);
      }
      RecordBackingValue(var, result);
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
    value.roots.clear();
    RecordBackingValue(assign->var_, value);
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
      auto info = ResolveBackingValue(call->args_[*alias_slot]);
      RecordRootCandidates(assign->var_, info.roots, info.ambiguous);
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

void BufferRootCollector::VisitBackingStorageOperand(const ExprPtr& expr,
                                                     const std::function<void(const Var*)>& record_read) {
  // Resolve nested expressions once during this operand walk. Statement
  // boundaries are outside this scope, so pre-SSA rebindings cannot reuse it.
  struct MemoScope {
    size_t& depth;
    std::unordered_map<ExprPtr, RootInfo>& memo;
    ~MemoScope() {
      if (--depth == 0) memo.clear();
    }
  };
  ++backing_value_depth_;
  const MemoScope scope{backing_value_depth_, backing_value_memo_};
  for (const Var* root : CollectBackingContentRoots(ResolveBackingValue(expr))) record_read(root);
  VisitExpr(expr);
}

BufferRootCollector::RootInfo BufferRootCollector::ProjectBackingElement(const RootInfo& value,
                                                                         size_t index) {
  if (value.elements && index < value.elements->size()) return (*value.elements)[index];
  if (!value.loop_tuple || index >= value.loop_tuple->type->types_.size()) return {};

  auto& tuple = *value.loop_tuple;
  auto [it, inserted] = tuple.projected.try_emplace(index);
  auto& child = it->second;
  if (inserted) {
    // Register the identity before following sources, since a repeated
    // iteration can project the same field through a carry cycle.
    child = MakeLoopStorageValue(tuple.type->types_[index], tuple.span, nullptr);
    for (const auto& source : tuple.sources) {
      LinkLoopStorageValues(child, ProjectBackingElement(source, index));
    }
  }
  return child;
}

BufferRootCollector::RootCandidates BufferRootCollector::CollectBackingContentRoots(const RootInfo& value) {
  RootCandidates roots;
  std::vector<const RootInfo*> pending{&value};
  std::unordered_set<const std::vector<RootInfo>*> seen_elements;
  while (!pending.empty()) {
    const auto* current = pending.back();
    pending.pop_back();
    roots.insert(roots.end(), current->roots.begin(), current->roots.end());
    if (current->loop_tuple) roots.push_back(AggregateLoopTupleRoots(*current->loop_tuple));
    if (current->elements && seen_elements.insert(current->elements.get()).second) {
      for (const auto& element : *current->elements) pending.push_back(&element);
    }
  }
  return roots;
}

const Var* BufferRootCollector::AggregateLoopTupleRoots(LoopTupleRoots& tuple) {
  if (tuple.aggregate) return tuple.aggregate;
  auto root = std::make_shared<Var>("storage_tuple_contents", tuple.type, tuple.span);
  tuple.aggregate = root.get();
  storage_proxy_vars_.push_back(std::move(root));
  auto& successors = storage_successors_[tuple.aggregate];
  for (const auto& source : tuple.sources) {
    auto roots = CollectBackingContentRoots(source);
    successors.insert(successors.end(), roots.begin(), roots.end());
  }
  return tuple.aggregate;
}

BufferRootCollector::RootInfo BufferRootCollector::ResolveBackingValue(const ExprPtr& expr) {
  if (backing_value_depth_ == 0) return ResolveBackingValueUncached(expr);
  auto [it, inserted] = backing_value_memo_.try_emplace(expr);
  auto& value = it->second;
  if (inserted) value = ResolveBackingValueUncached(expr);
  return value;
}

BufferRootCollector::RootInfo BufferRootCollector::ResolveBackingValueUncached(const ExprPtr& expr) {
  if (auto var = AsVarLike(expr)) {
    auto it = backing_tuple_roots_.find(var.get());
    if (it != backing_tuple_roots_.end()) return it->second;
    return {RootCandidatesOf(var.get()), IsAmbiguous(expr), nullptr};
  }
  if (auto projection = As<TupleGetItemExpr>(expr)) {
    return ProjectBackingElement(ResolveBackingValue(projection->tuple_),
                                 static_cast<size_t>(projection->index_));
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
  std::shared_ptr<const StorageReturnSummary> summary;
  if (cached != ctx.memo.end()) {
    summary = cached->second;
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
        if (parameter != positions.end()) built.formal_bindings.emplace_back(it->second, parameter->second);
        built.nodes.push_back(
            {root->GetType(),
             parameter == positions.end() ? std::nullopt : std::optional<size_t>{parameter->second},
             {}});
        pending.push_back(root);
      }
      return it->second;
    };
    std::unordered_map<const LoopTupleRoots*, size_t> tuple_ids;
    std::unordered_map<std::shared_ptr<const std::vector<RootInfo>>,
                       std::shared_ptr<std::vector<StorageReturnValue>>>
        element_summaries;
    const auto summarize = [&](const auto& self, const RootInfo& value) -> StorageReturnValue {
      StorageReturnValue result;
      if (value.loop_tuple) {
        const auto& tuple = *value.loop_tuple;
        auto [entry, inserted] = tuple_ids.try_emplace(&tuple, built.tuples.size());
        const size_t index = entry->second;
        result.tuple_sources = index;
        if (inserted) {
          // Enter the tuple before following sources, which can refer back
          // to it. No fields are projected while exporting a loop result.
          built.tuples.push_back({tuple.type, {}});
          std::vector<StorageReturnValue> sources;
          sources.reserve(tuple.sources.size());
          for (const auto& source : tuple.sources) sources.push_back(self(self, source));
          built.tuples[index].sources = std::move(sources);
        }
      } else if (value.elements) {
        auto [entry, inserted] = element_summaries.try_emplace(value.elements);
        if (inserted) entry->second = std::make_shared<std::vector<StorageReturnValue>>();
        result.elements = entry->second;
        if (inserted) {
          result.elements->reserve(value.elements->size());
          for (const auto& element : *value.elements) result.elements->push_back(self(self, element));
        }
      }
      result.roots.reserve(value.roots.size());
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
    std::unordered_map<const StorageCallInstance*, size_t> local_calls;
    for (size_t i = 0; i < pending.size(); ++i) {
      const Var* root = pending[i];
      if (auto it = nested.storage_call_roots_.find(root); it != nested.storage_call_roots_.end()) {
        const auto& projection = it->second;
        auto [call, inserted] = local_calls.try_emplace(projection.instance.get(), built.calls.size());
        if (inserted) {
          StorageSummaryCall entry{projection.instance->callee, {}, {}};
          const auto& actuals = projection.instance->arguments;
          entry.arguments.reserve(actuals.size());
          for (const auto& actual : actuals) entry.arguments.push_back(summarize(summarize, actual));
          built.calls.push_back(std::move(entry));
        }
        built.nodes[i].projection = std::pair{call->second, projection.node};
        continue;
      }
      auto next = nested.storage_successors_.find(root);
      if (next == nested.storage_successors_.end()) continue;
      std::vector<size_t> edges;
      edges.reserve(next->second.size());
      for (const Var* root : next->second) edges.push_back(intern(root));
      built.nodes[i].successors = std::move(edges);
    }
    // The caller references one node per returned buffer. Keeping the
    // alternatives inside this graph avoids multiplying boundary projections
    // when successive callers merge the results of different call sites.
    std::unordered_set<const std::vector<StorageReturnValue>*> joined_elements;
    std::unordered_set<size_t> joined_tuples;
    const auto join_returns = [&](const auto& self, StorageReturnValue& value) -> void {
      if (value.roots.size() > 1) {
        const auto root = built.nodes.size();
        auto type = built.nodes[value.roots.front()].type;
        built.nodes.push_back({std::move(type), std::nullopt, std::move(value.roots)});
        value.roots = {root};
      }
      if (value.elements && joined_elements.insert(value.elements.get()).second) {
        for (auto& element : *value.elements) self(self, element);
      }
      if (const auto tuple = value.tuple_sources; tuple.has_value() && joined_tuples.insert(*tuple).second) {
        for (auto& source : built.tuples[*tuple].sources) self(self, source);
      }
    };
    for (auto& output : built.outputs) join_returns(join_returns, output);
    ctx.active.erase(key);
    summary = std::make_shared<StorageReturnSummary>(std::move(built));
    ctx.memo.emplace(key, summary);
  } else {
    // Recursive call graphs are diagnosed separately. Conservatively connect
    // the results to a shared union of the arguments until then.
    StorageReturnSummary recursive;
    recursive.outputs.resize(callee->return_types_.size());
    recursive.nodes.reserve(callee->params_.size() + 1);
    recursive.formal_bindings.reserve(callee->params_.size());
    for (size_t i = 0; i < callee->params_.size(); ++i) {
      recursive.formal_bindings.emplace_back(i, i);
      recursive.nodes.push_back({callee->params_[i]->GetType(), i, {}});
    }
    std::vector<size_t> edges;
    edges.reserve(recursive.nodes.size());
    for (size_t i = 0; i < recursive.nodes.size(); ++i) edges.push_back(i);
    const size_t union_id = recursive.nodes.size();
    recursive.nodes.push_back({callee->return_types_[0], std::nullopt, std::move(edges)});
    for (auto& output : recursive.outputs) output.roots.push_back(union_id);
    summary = std::make_shared<StorageReturnSummary>(std::move(recursive));
  }

  const size_t ctx_count = transform_utils::TrailingCommCtxParamCount(callee);
  const size_t prefix = call->args_.size() >= ctx_count ? call->args_.size() - ctx_count : 0;
  const size_t gap =
      callee->params_.size() >= call->args_.size() ? callee->params_.size() - call->args_.size() : 0;
  auto instance = std::make_shared<StorageCallInstance>();
  instance->callee = summary;
  instance->arguments.resize(callee->params_.size());
  for (const auto& binding : summary->formal_bindings) {
    const size_t parameter = binding.second;
    if (parameter >= prefix && parameter < callee->params_.size() - ctx_count) continue;
    const size_t argument = parameter < prefix ? parameter : parameter - gap;
    if (argument < call->args_.size()) {
      instance->arguments[parameter] = ResolveBackingValue(call->args_[argument]);
    }
  }
  std::unordered_map<size_t, const Var*> projections;
  const auto project = [&](size_t index) -> RootInfo {
    const auto& node = summary->nodes[index];
    if (node.parameter) return instance->arguments[*node.parameter];
    if (node.successors.empty() && !node.projection) return {};
    auto [it, inserted] = projections.try_emplace(index, nullptr);
    if (inserted) {
      auto proxy = std::make_shared<Var>("storage_projection", node.type, call->span_);
      it->second = proxy.get();
      storage_call_roots_.emplace(proxy.get(), StorageCallProjection{instance, index});
      storage_proxy_vars_.push_back(std::move(proxy));
    }
    return {{it->second}, false};
  };
  std::unordered_map<size_t, RootInfo> bound_tuples;
  std::unordered_map<const std::vector<StorageReturnValue>*, std::shared_ptr<std::vector<RootInfo>>>
      bound_elements;
  const auto bind = [&](const auto& self, const StorageReturnValue& result) -> RootInfo {
    RootInfo info;
    if (const auto index = result.tuple_sources; index.has_value()) {
      auto [entry, inserted] = bound_tuples.try_emplace(*index);
      auto& tuple = entry->second;
      if (inserted) {
        const auto& source_tuple = summary->tuples[*index];
        tuple = MakeLoopStorageValue(source_tuple.type, call->span_, nullptr);
        for (const auto& source : source_tuple.sources) LinkLoopStorageValues(tuple, self(self, source));
      }
      info = tuple;
    } else if (result.elements) {
      auto [entry, inserted] = bound_elements.try_emplace(result.elements.get());
      if (inserted) entry->second = std::make_shared<std::vector<RootInfo>>();
      auto elements = entry->second;
      if (inserted) {
        elements->reserve(result.elements->size());
        for (const auto& element : *result.elements) elements->push_back(self(self, element));
      }
      info.elements = std::move(elements);
    }
    std::unordered_set<const Var*> seen;
    for (size_t root : result.roots) {
      const auto actual = project(root);
      if (actual.elements || actual.loop_tuple) {
        info.elements = actual.elements;
        info.loop_tuple = actual.loop_tuple;
      }
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

void BufferRootCollector::PropagateStorageEffects(
    std::unordered_map<const Var*, unsigned char>& effects) const {
  std::vector<const Var*> pending;
  pending.reserve(effects.size());
  for (const auto& [root, bits] : effects) {
    if (bits) pending.push_back(root);
  }
  const auto add = [&](const Var* root, unsigned char bits) {
    auto& current = effects[root];
    if (!(bits & ~current)) return;
    current |= bits;
    pending.push_back(root);
  };

  // A call may demand several returns that share a transfer. Apply each
  // effect bit to that transfer once, keeping distinct actual bindings apart.
  using AppliedEffects = std::unordered_map<const StorageFormalTransfer*, unsigned char>;
  std::unordered_map<const StorageCallInstance*, AppliedEffects> applied;
  std::vector<const StorageFormalTransfer*> pending_transfers;
  size_t cursor = 0;
  while (cursor < pending.size()) {
    const Var* root = pending[cursor++];
    const auto bits = effects[root];
    if (auto it = storage_successors_.find(root); it != storage_successors_.end()) {
      for (const Var* target : it->second) add(target, bits);
    }
    if (auto it = storage_call_roots_.find(root); it != storage_call_roots_.end()) {
      const auto& projection = it->second;
      const auto& instance = *projection.instance;
      const auto* transfer = GetStorageTransfer(*instance.callee, projection.node);
      if (!transfer) continue;
      auto& visited = applied[&instance];
      pending_transfers.push_back(transfer);
      while (!pending_transfers.empty()) {
        const auto* next = pending_transfers.back();
        pending_transfers.pop_back();
        auto& current = visited[next];
        if (!(bits & ~current)) continue;
        current |= bits;
        if (next->split == 0) {
          for (const Var* target : instance.arguments[next->first].roots) add(target, bits);
        } else {
          pending_transfers.push_back(next->left);
          pending_transfers.push_back(next->right);
        }
      }
    }
  }
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

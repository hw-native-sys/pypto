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
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
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

struct StorageFormalPath {
  std::shared_ptr<const StorageFormalPath> parent;
  size_t index;
};

struct StorageReturnValue {
  std::vector<size_t> roots;
  std::shared_ptr<std::vector<StorageReturnValue>> elements;
  std::optional<size_t> tuple_sources;
  // The call owns the callee summary containing the referenced tuple.
  std::optional<std::pair<size_t, const StorageReturnValue*>> tuple_call;
  // Whole-tuple effects travel through these roots without projecting fields.
  std::vector<size_t> contents;
};

struct StorageTupleSources {
  std::shared_ptr<const TupleType> type;
  std::vector<StorageReturnValue> sources;
  mutable std::unordered_map<size_t, StorageReturnValue> fields = {};
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
  std::unordered_map<size_t, StorageReturnValue> parameters;
  mutable std::unordered_map<const StorageFormalTransfer*, size_t> mapped_transfers;
};

struct StorageFormalBinding {
  size_t parameter;
  std::shared_ptr<const StorageFormalPath> path;
};

struct StorageSummaryNode {
  TypePtr type;
  std::optional<size_t> binding;
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
  // Projecting a field can append nodes after transfer calculation begins.
  mutable std::deque<StorageSummaryNode> nodes;
  mutable std::vector<StorageFormalBinding> formal_bindings;
  std::vector<StorageReturnValue> outputs;
  std::optional<StorageReturnValue> packed_outputs;
  std::vector<StorageSummaryCall> calls;
  // Shared tuple values retain their source edges, including loop cycles.
  mutable std::deque<StorageTupleSources> tuples;
  mutable std::map<std::pair<size_t, size_t>, size_t> projected_roots;
  mutable std::map<std::pair<size_t, size_t>, StorageReturnValue> formal_fields;
  mutable std::unordered_map<const StorageReturnValue*, std::unordered_map<size_t, StorageReturnValue>>
      call_fields;
  mutable std::unordered_map<const StorageReturnValue*,
                             std::unordered_map<const StorageFormalPath*, const StorageReturnValue*>>
      argument_fields;
  mutable std::vector<StorageMappedTransfer> mapped_nodes;
  mutable std::unordered_map<size_t, const StorageFormalTransfer*> formal_transfers;
  mutable std::vector<std::unique_ptr<const StorageFormalTransfer>> transfer_storage;
  mutable std::unordered_map<size_t, const StorageFormalTransfer*> formal_leaves;
  mutable std::map<std::pair<size_t, size_t>, const StorageFormalTransfer*> transfer_branches;
  mutable std::map<std::pair<size_t, size_t>, const StorageFormalTransfer*> transfer_unions;

  void PackOutputs() {
    if (outputs.size() < 2) return;
    // A call exposes one tuple descriptor for multiple returns. Share its
    // fields and whole-value transfer across all callers.
    auto& packed = packed_outputs.emplace();
    packed.elements = std::make_shared<std::vector<StorageReturnValue>>(outputs);
    std::vector<size_t> edges;
    for (const auto& output : outputs) {
      edges.insert(edges.end(), output.roots.begin(), output.roots.end());
      edges.insert(edges.end(), output.contents.begin(), output.contents.end());
    }
    if (edges.empty()) return;
    packed.contents = {nodes.size()};
    nodes.push_back({GetUnknownType(), std::nullopt, std::move(edges)});
  }

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

const StorageReturnValue& ProjectSummaryField(const StorageReturnSummary& summary,
                                              const StorageReturnValue& value, size_t index);

std::vector<size_t> SummaryContentRoots(const StorageReturnValue& value) {
  return value.contents.empty() ? value.roots : value.contents;
}

size_t SummaryProjectionRoot(const StorageReturnSummary& summary, size_t call, size_t root) {
  auto [entry, inserted] = summary.projected_roots.try_emplace({call, root}, summary.nodes.size());
  if (inserted) {
    const auto& callee = *summary.calls[call].callee;
    summary.nodes.push_back({callee.nodes[root].type, std::nullopt, {}, std::pair{call, root}});
  }
  return entry->second;
}

bool HasSummaryTupleShape(const StorageReturnSummary& summary, const StorageReturnValue& value) {
  if (value.elements || value.tuple_sources || value.tuple_call) return true;
  return value.roots.size() == 1 && As<TupleType>(summary.nodes[value.roots.front()].type);
}

const StorageReturnValue& ProjectSummaryField(const StorageReturnSummary& summary,
                                              const StorageReturnValue& value, size_t index) {
  static const StorageReturnValue kUnknown;
  if (value.elements) {
    return index < value.elements->size() ? (*value.elements)[index] : kUnknown;
  }
  if (const auto source_index = value.tuple_sources; source_index.has_value()) {
    const auto& source = summary.tuples[*source_index];
    if (index >= source.type->types_.size()) return kUnknown;
    auto [entry, inserted] = source.fields.try_emplace(index);
    auto& result = entry->second;
    if (!inserted) return result;
    const auto& type = source.type->types_[index];
    const size_t root = summary.nodes.size();
    summary.nodes.push_back({type, std::nullopt, {}});
    if (const auto tuple_type = As<TupleType>(type)) {
      const size_t projected = summary.tuples.size();
      result.tuple_sources = projected;
      result.contents = {root};
      summary.tuples.push_back({tuple_type, {}});
      auto& children = summary.tuples[projected].sources;
      children.reserve(source.sources.size());
      for (const auto& parent : source.sources) {
        const auto& child = ProjectSummaryField(summary, parent, index);
        if (HasSummaryTupleShape(summary, child)) children.push_back(child);
        auto roots = SummaryContentRoots(child);
        auto& edges = summary.nodes[root].successors;
        edges.insert(edges.end(), roots.begin(), roots.end());
      }
    } else {
      result.roots = {root};
      for (const auto& parent : source.sources) {
        const auto& child = ProjectSummaryField(summary, parent, index);
        auto& edges = summary.nodes[root].successors;
        edges.insert(edges.end(), child.roots.begin(), child.roots.end());
      }
    }
    return result;
  }
  if (const auto call_ref = value.tuple_call; call_ref.has_value()) {
    auto& fields = summary.call_fields[&value];
    auto [entry, inserted] = fields.try_emplace(index);
    auto& result = entry->second;
    if (!inserted) return result;
    const auto [call, target] = *call_ref;
    const auto& callee = *summary.calls[call].callee;
    const auto& child = ProjectSummaryField(callee, *target, index);
    if (HasSummaryTupleShape(callee, child)) {
      result.tuple_call = std::pair{call, &child};
      for (size_t root : SummaryContentRoots(child)) {
        result.contents.push_back(SummaryProjectionRoot(summary, call, root));
      }
    } else {
      for (size_t root : child.roots) {
        result.roots.push_back(SummaryProjectionRoot(summary, call, root));
      }
    }
    return result;
  }
  if (value.roots.size() != 1) return kUnknown;
  const auto& node = summary.nodes[value.roots.front()];
  if (!node.binding) return kUnknown;
  const auto type = As<TupleType>(node.type);
  if (!type || index >= type->types_.size()) return kUnknown;
  auto [entry, inserted] = summary.formal_fields.try_emplace({*node.binding, index});
  auto& result = entry->second;
  if (inserted) {
    const auto parent = summary.formal_bindings[*node.binding];
    const size_t binding = summary.formal_bindings.size();
    auto path = std::make_shared<StorageFormalPath>(StorageFormalPath{parent.path, index});
    summary.formal_bindings.push_back({parent.parameter, std::move(path)});
    result.roots = {summary.nodes.size()};
    summary.nodes.push_back({type->types_[index], binding, {}});
  }
  return result;
}

const StorageReturnValue* ProjectSummaryPath(const StorageReturnSummary& summary,
                                             const StorageReturnValue* value,
                                             const std::shared_ptr<const StorageFormalPath>& path) {
  if (!path) return value;
  auto& fields = summary.argument_fields[value];
  auto [entry, inserted] = fields.try_emplace(path.get(), nullptr);
  auto& projected = entry->second;
  if (inserted) {
    projected = &ProjectSummaryField(summary, *ProjectSummaryPath(summary, value, path->parent), path->index);
  }
  return projected;
}

// Mapped transfer IDs must stay distinct from storage nodes appended on demand.
constexpr size_t kMappedStorageNode = size_t{1} << (std::numeric_limits<size_t>::digits - 1);

size_t MapStorageTransfer(const StorageReturnSummary& summary, size_t call,
                          const StorageFormalTransfer* transfer) {
  auto& mapped = summary.calls[call].mapped_transfers;
  auto [it, inserted] = mapped.try_emplace(transfer, kMappedStorageNode | summary.mapped_nodes.size());
  if (inserted) summary.mapped_nodes.push_back({call, transfer});
  return it->second;
}

std::vector<size_t> StorageTransferSuccessors(const StorageReturnSummary& summary, size_t index) {
  if ((index & kMappedStorageNode) == 0) {
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
  const auto [call, transfer] = summary.mapped_nodes[index & ~kMappedStorageNode];
  if (transfer->split == 0) {
    const auto& mapping = summary.calls[call];
    const auto& binding = mapping.callee->formal_bindings[transfer->first];
    const auto parameter = mapping.parameters.find(binding.parameter);
    if (parameter == mapping.parameters.end()) return {};
    return SummaryContentRoots(*ProjectSummaryPath(summary, &parameter->second, binding.path));
  }
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
        if ((member & kMappedStorageNode) == 0) {
          if (const auto binding = summary.nodes[member].binding; binding.has_value()) {
            transfer = UnionStorageTransfers(summary, transfer, summary.FormalLeaf(*binding));
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

struct BufferRootCollector::BoundStorageTuple {
  StorageCallInstance* instance;
  const StorageReturnValue* value;
  std::optional<RootInfo> forwarded = std::nullopt;
};

struct BufferRootCollector::StorageCallInstance {
  StorageCallInstance(std::shared_ptr<const StorageReturnSummary> callee, Span span)
      : callee(std::move(callee)), span(std::move(span)) {}

  std::shared_ptr<const StorageReturnSummary> callee;
  Span span;
  // Capture actual values before local variables can be rebound. Individual
  // formal fields are resolved only when a result or effect needs them.
  std::unordered_map<size_t, RootInfo> actual_parameters;
  std::unordered_map<size_t, RootInfo> bindings;
  StorageCallInstance* parent = nullptr;
  const StorageSummaryCall* local_call = nullptr;
  std::unordered_map<size_t, const Var*> projections;
  std::unordered_map<const StorageReturnValue*, BoundStorageTuple> bound_tuples;
  std::unordered_map<size_t, StorageCallInstance*> nested_calls;
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
    if (view_policy_ == ViewRootPolicy::kBackingStorage && As<TupleType>(param->GetType())) {
      auto value = MakeLoopStorageValue(param->GetType(), param->span_, param.get());
      value.loop_tuple->formal = param.get();
      formal_projections_.emplace(param.get(), FormalReference{param.get(), nullptr});
      backing_tuple_roots_.emplace(param.get(), std::move(value));
    }
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
  if (lhs.loop_tuple || rhs.loop_tuple || lhs.bound_tuple || rhs.bound_tuple) {
    if ((lhs.loop_tuple && lhs.loop_tuple == rhs.loop_tuple) ||
        (lhs.bound_tuple && lhs.bound_tuple == rhs.bound_tuple)) {
      return lhs;
    }
    auto joined = MakeLoopStorageValue(type, span, nullptr);
    LinkLoopStorageValues(joined, lhs);
    LinkLoopStorageValues(joined, rhs);
    return joined;
  }
  if (lhs.elements || rhs.elements) {
    if (lhs.elements == rhs.elements) return lhs;
    // Tuple element vectors are immutable. Hold them in the key so a
    // later branch can reuse a join without risking pointer reuse.
    auto [entry, inserted] = merged_tuple_roots_.try_emplace(std::tuple{lhs.elements, rhs.elements, type});
    auto& result = entry->second;
    if (!inserted) return result;
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
    result = {{}, false, std::move(elements)};
    return result;
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
    if (source.loop_tuple == &tuple || !source.IsTuple()) return;
    // Projecting a permuted feedback value can insert new fields into
    // this carry. Those fields see the new source when they are created.
    std::vector<size_t> projected_fields;
    projected_fields.reserve(tuple.projected.size());
    for (const auto& entry : tuple.projected) projected_fields.push_back(entry.first);
    tuple.sources.push_back(source);
    if (tuple.aggregate) {
      const auto roots = CollectBackingContentRoots(source);
      auto& successors = storage_successors_[tuple.aggregate];
      successors.insert(successors.end(), roots.begin(), roots.end());
    }
    for (size_t index : projected_fields) {
      LinkLoopStorageValues(tuple.projected.at(index), ProjectBackingElement(source, index));
    }
    return;
  }
  if (target.roots.empty() || source.roots.empty()) return;
  auto& successors = storage_successors_[target.roots.front()];
  successors.insert(successors.end(), source.roots.begin(), source.roots.end());
}

void BufferRootCollector::RecordBackingValue(const VarPtr& var, const RootInfo& value) {
  if (value.IsTuple()) {
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

BufferRootCollector::BackingValueMemoScope::BackingValueMemoScope(BufferRootCollector& collector)
    : collector_(collector) {
  ++collector_.backing_value_depth_;
}

BufferRootCollector::BackingValueMemoScope::~BackingValueMemoScope() {
  if (--collector_.backing_value_depth_ == 0) collector_.backing_value_memo_.clear();
}

void BufferRootCollector::VisitBackingStorageOperand(const ExprPtr& expr,
                                                     const std::function<void(const Var*)>& record_read) {
  const BackingValueMemoScope scope(*this);
  for (const Var* root : ResolveBackingContentRoots(expr)) record_read(root);
  VisitExpr(expr);
}

BufferRootCollector::RootCandidates BufferRootCollector::ResolveBackingContentRoots(const ExprPtr& expr) {
  return CollectBackingContentRoots(ResolveBackingValue(expr));
}

BufferRootCollector::RootInfo BufferRootCollector::ProjectBackingElement(const RootInfo& value,
                                                                         size_t index) {
  if (value.elements && index < value.elements->size()) return (*value.elements)[index];
  if (value.bound_tuple) {
    auto& tuple = *value.bound_tuple;
    if (tuple.value->elements) {
      const auto& elements = *tuple.value->elements;
      return index < elements.size() ? BindStorageValue(*tuple.instance, elements[index]) : RootInfo{};
    }
    if (tuple.value->tuple_sources) {
      const auto& child = ProjectSummaryField(*tuple.instance->callee, *tuple.value, index);
      return BindStorageValue(*tuple.instance, child);
    }
    return ProjectBackingElement(ForwardStorageTuple(tuple), index);
  }
  if (!value.loop_tuple || index >= value.loop_tuple->type->types_.size()) return {};

  auto& tuple = *value.loop_tuple;
  auto [it, inserted] = tuple.projected.try_emplace(index);
  auto& child = it->second;
  if (inserted) {
    // Register the identity before following sources, since a repeated
    // iteration can project the same field through a carry cycle.
    child = MakeLoopStorageValue(tuple.type->types_[index], tuple.span, nullptr);
    if (tuple.formal) {
      const auto& parent = formal_projections_.at(tuple.formal);
      auto path = std::make_shared<StorageFormalPath>(StorageFormalPath{parent.path, index});
      const Var* root = nullptr;
      if (child.loop_tuple) {
        auto proxy = std::make_shared<Var>("storage_formal_tuple", child.loop_tuple->type, tuple.span);
        root = proxy.get();
        child.loop_tuple->formal = root;
        storage_proxy_vars_.push_back(std::move(proxy));
      } else {
        root = child.roots.front();
      }
      storage_successors_[root] = {parent.parameter};
      formal_projections_.emplace(root, FormalReference{parent.parameter, std::move(path)});
    }
    for (const auto& source : tuple.sources) {
      LinkLoopStorageValues(child, ProjectBackingElement(source, index));
    }
  }
  return child;
}

BufferRootCollector::RootCandidates BufferRootCollector::CollectBackingContentRoots(const RootInfo& value) {
  auto roots = value.roots;
  if (value.loop_tuple) roots.push_back(AggregateLoopTupleRoots(*value.loop_tuple));
  if (value.elements) roots.push_back(AggregateExplicitTupleRoots(value.elements));
  if (value.bound_tuple) {
    auto& tuple = *value.bound_tuple;
    for (size_t index : tuple.value->contents) {
      auto actual = ProjectStorageRoot(*tuple.instance, index);
      auto contents = CollectBackingContentRoots(actual);
      roots.insert(roots.end(), contents.begin(), contents.end());
    }
  }
  return roots;
}

const Var* BufferRootCollector::AggregateExplicitTupleRoots(
    const std::shared_ptr<const std::vector<RootInfo>>& elements) {
  auto [it, inserted] = explicit_tuple_contents_.try_emplace(elements, nullptr);
  if (!inserted) return it->second;
  auto proxy = std::make_shared<Var>("storage_tuple_contents", GetUnknownType(), Span::unknown());
  const Var* root = proxy.get();
  it->second = root;
  storage_proxy_vars_.push_back(std::move(proxy));
  auto& successors = storage_successors_[root];
  for (const auto& element : *elements) {
    auto roots = CollectBackingContentRoots(element);
    successors.insert(successors.end(), roots.begin(), roots.end());
  }
  return root;
}

const Var* BufferRootCollector::AggregateLoopTupleRoots(LoopTupleRoots& tuple) {
  if (tuple.formal) return tuple.formal;
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
  if (outputs.size() == 1 &&
      (outputs[0].type == call->GetType() || structural_equal(outputs[0].type, call->GetType()))) {
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
        const auto projection = nested.formal_projections_.find(root);
        const auto parameter = positions.find(
            projection == nested.formal_projections_.end() ? root : projection->second.parameter);
        std::optional<size_t> binding;
        if (parameter != positions.end()) {
          binding = built.formal_bindings.size();
          built.formal_bindings.push_back({parameter->second, projection == nested.formal_projections_.end()
                                                                  ? nullptr
                                                                  : projection->second.path});
        }
        built.nodes.push_back({root->GetType(), binding, {}});
        pending.push_back(root);
      }
      return it->second;
    };
    std::unordered_map<const StorageCallInstance*, size_t> local_calls;
    std::vector<StorageCallInstance*> pending_calls;
    const auto intern_call = [&](StorageCallInstance* instance) {
      auto [it, inserted] = local_calls.try_emplace(instance, built.calls.size());
      if (inserted) {
        built.calls.push_back({instance->callee, {}, {}});
        pending_calls.push_back(instance);
      }
      return it->second;
    };
    std::unordered_map<const LoopTupleRoots*, size_t> tuple_ids;
    std::unordered_map<std::shared_ptr<const std::vector<RootInfo>>,
                       std::shared_ptr<std::vector<StorageReturnValue>>>
        element_summaries;
    const auto summarize = [&](const auto& self, const RootInfo& value) -> StorageReturnValue {
      StorageReturnValue result;
      if (value.loop_tuple && value.loop_tuple->formal) {
        result.roots.push_back(intern(value.loop_tuple->formal));
      } else if (value.loop_tuple) {
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
      } else if (value.bound_tuple) {
        auto* tuple = value.bound_tuple;
        // Export the forwarded field shape once in the shared summary.
        // Later callers bind the already-composed actuals directly.
        while (tuple->value->tuple_call) {
          const auto forwarded = nested.ForwardStorageTuple(*tuple);
          if (!forwarded.bound_tuple) break;
          tuple = forwarded.bound_tuple;
        }
        result.tuple_call = std::pair{intern_call(tuple->instance), tuple->value};
      } else if (value.elements) {
        auto [entry, inserted] = element_summaries.try_emplace(value.elements);
        if (inserted) entry->second = std::make_shared<std::vector<StorageReturnValue>>();
        result.elements = entry->second;
        if (inserted) {
          result.elements->reserve(value.elements->size());
          for (const auto& element : *value.elements) result.elements->push_back(self(self, element));
        }
      }
      if (result.elements || result.tuple_call || result.tuple_sources) {
        for (const Var* root : nested.CollectBackingContentRoots(value)) {
          result.contents.push_back(intern(root));
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
    // Export each local root and call once. Serializing an actual can
    // discover more roots or references to earlier calls.
    size_t node_cursor = 0;
    size_t call_cursor = 0;
    while (node_cursor < pending.size() || call_cursor < pending_calls.size()) {
      if (node_cursor == pending.size()) {
        const size_t index = call_cursor++;
        auto& call = *pending_calls[index];
        std::unordered_map<size_t, StorageReturnValue> parameters;
        if (call.local_call) {
          for (const auto& [parameter, recipe] : call.local_call->parameters) {
            parameters.emplace(parameter,
                               summarize(summarize, nested.ResolveStorageParameter(call, parameter)));
          }
        } else {
          for (const auto& [parameter, actual] : call.actual_parameters) {
            parameters.emplace(parameter, summarize(summarize, actual));
          }
        }
        built.calls[index].parameters = std::move(parameters);
        continue;
      }
      const size_t index = node_cursor++;
      const Var* root = pending[index];
      // A formal field is a binding in its own right. Its edge to the
      // containing parameter only applies inside the function.
      if (built.nodes[index].binding.has_value()) continue;
      if (auto it = nested.storage_call_roots_.find(root); it != nested.storage_call_roots_.end()) {
        const auto& projection = it->second;
        built.nodes[index].projection = std::pair{intern_call(projection.instance), projection.node};
        continue;
      }
      auto next = nested.storage_successors_.find(root);
      if (next == nested.storage_successors_.end()) continue;
      std::vector<size_t> edges;
      edges.reserve(next->second.size());
      for (const Var* root : next->second) edges.push_back(intern(root));
      built.nodes[index].successors = std::move(edges);
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
    built.PackOutputs();
    ctx.active.erase(key);
    summary = std::make_shared<StorageReturnSummary>(std::move(built));
    ctx.memo.emplace(key, summary);
  } else {
    // Recursive call graphs are diagnosed separately. Conservatively connect
    // the results to a shared union of the arguments until then.
    StorageReturnSummary recursive;
    recursive.outputs.resize(callee->return_types_.size());
    recursive.formal_bindings.reserve(callee->params_.size());
    for (size_t i = 0; i < callee->params_.size(); ++i) {
      recursive.formal_bindings.push_back({i, nullptr});
      recursive.nodes.push_back({callee->params_[i]->GetType(), i, {}});
    }
    std::vector<size_t> edges;
    edges.reserve(recursive.nodes.size());
    for (size_t i = 0; i < recursive.nodes.size(); ++i) edges.push_back(i);
    const size_t union_id = recursive.nodes.size();
    recursive.nodes.push_back({callee->return_types_[0], std::nullopt, std::move(edges)});
    for (auto& output : recursive.outputs) output.roots.push_back(union_id);
    recursive.PackOutputs();
    summary = std::make_shared<StorageReturnSummary>(std::move(recursive));
  }

  const size_t ctx_count = transform_utils::TrailingCommCtxParamCount(callee);
  const size_t prefix = call->args_.size() >= ctx_count ? call->args_.size() - ctx_count : 0;
  const size_t gap =
      callee->params_.size() >= call->args_.size() ? callee->params_.size() - call->args_.size() : 0;
  auto instance = std::make_shared<StorageCallInstance>(summary, call->span_);
  storage_calls_.push_back(instance);
  for (size_t argument = 0; argument < call->args_.size(); ++argument) {
    const size_t parameter = argument < prefix ? argument : argument + gap;
    if (parameter < callee->params_.size()) {
      instance->actual_parameters.emplace(parameter, ResolveBackingValue(call->args_[argument]));
    }
  }
  if (summary->packed_outputs && As<TupleType>(call->GetType())) {
    return {{BindStorageValue(*instance, *summary->packed_outputs), call->GetType()}};
  }
  std::vector<OutputRoot> roots;
  roots.reserve(summary->outputs.size());
  for (size_t i = 0; i < summary->outputs.size(); ++i) {
    roots.push_back(OutputRoot{BindStorageValue(*instance, summary->outputs[i]), callee->return_types_[i]});
  }
  return roots;
}

BufferRootCollector::RootInfo BufferRootCollector::BindStorageField(
    const RootInfo& actual, const std::shared_ptr<const StorageFormalPath>& path) {
  if (!path) return actual;
  if (!actual.IsTuple()) return {};
  // Tuple shapes identify values, so rebinding a variable to a new tuple
  // cannot reuse the former fields. Keep explicit tuples and paths alive.
  StorageTupleIdentity identity = actual.elements;
  if (actual.loop_tuple) identity = actual.loop_tuple;
  if (actual.bound_tuple) identity = actual.bound_tuple;
  auto& fields = storage_field_projections_[std::move(identity)];
  const auto bind = [&](const auto& self, const std::shared_ptr<const StorageFormalPath>& path) -> RootInfo {
    if (!path) return actual;
    auto [entry, inserted] = fields.try_emplace(path);
    auto& field = entry->second;
    if (inserted) {
      field = ProjectBackingElement(self(self, path->parent), path->index);
    }
    return field;
  };
  return bind(bind, path);
}

BufferRootCollector::RootInfo BufferRootCollector::ProjectStorageRoot(StorageCallInstance& instance,
                                                                      size_t index) {
  const auto& node = instance.callee->nodes[index];
  if (const auto binding = node.binding; binding.has_value()) {
    return ResolveStorageBinding(instance, *binding);
  }
  if (node.successors.empty() && !node.projection) return {};
  auto [it, inserted] = instance.projections.try_emplace(index, nullptr);
  if (inserted) {
    auto proxy = std::make_shared<Var>("storage_projection", node.type, instance.span);
    it->second = proxy.get();
    storage_call_roots_.emplace(proxy.get(), StorageCallProjection{&instance, index});
    storage_proxy_vars_.push_back(std::move(proxy));
  }
  return {{it->second}, false};
}

BufferRootCollector::StorageCallInstance& BufferRootCollector::BindNestedStorageCall(
    StorageCallInstance& parent, size_t index) {
  auto [it, inserted] = parent.nested_calls.try_emplace(index, nullptr);
  if (inserted) {
    const auto& call = parent.callee->calls[index];
    auto instance = std::make_shared<StorageCallInstance>(call.callee, parent.span);
    it->second = instance.get();
    storage_calls_.push_back(instance);
    instance->parent = &parent;
    instance->local_call = &call;
  }
  return *it->second;
}

const BufferRootCollector::RootInfo& BufferRootCollector::ResolveStorageBinding(StorageCallInstance& instance,
                                                                                size_t index) {
  auto [it, inserted] = instance.bindings.try_emplace(index);
  auto& result = it->second;
  if (!inserted) return result;
  const auto binding = instance.callee->formal_bindings[index];
  result = BindStorageField(ResolveStorageParameter(instance, binding.parameter), binding.path);
  return result;
}

const BufferRootCollector::RootInfo& BufferRootCollector::ResolveStorageParameter(
    StorageCallInstance& instance, size_t parameter) {
  if (const auto actual = instance.actual_parameters.find(parameter);
      actual != instance.actual_parameters.end()) {
    return actual->second;
  }
  static const RootInfo kUnknown;
  if (!instance.local_call) return kUnknown;
  const auto recipe = instance.local_call->parameters.find(parameter);
  if (recipe == instance.local_call->parameters.end()) return kUnknown;
  auto [entry, inserted] = instance.actual_parameters.try_emplace(parameter);
  auto& result = entry->second;
  result = BindStorageValue(*instance.parent, recipe->second);
  return result;
}

BufferRootCollector::RootInfo BufferRootCollector::ForwardStorageTuple(BoundStorageTuple& tuple) {
  const auto& reference = tuple.value->tuple_call;
  if (!reference) return {};
  if (!tuple.forwarded) {
    const auto& [call, result] = *reference;
    tuple.forwarded = BindStorageValue(BindNestedStorageCall(*tuple.instance, call), *result);
  }
  return *tuple.forwarded;
}

BufferRootCollector::RootInfo BufferRootCollector::BindStorageValue(StorageCallInstance& instance,
                                                                    const StorageReturnValue& result) {
  RootInfo info;
  if (result.tuple_sources || result.elements || result.tuple_call) {
    auto [entry, inserted] =
        instance.bound_tuples.try_emplace(&result, BoundStorageTuple{&instance, &result});
    info.bound_tuple = &entry->second;
  }
  std::unordered_set<const Var*> seen;
  for (size_t root : result.roots) {
    const auto actual = ProjectStorageRoot(instance, root);
    if (actual.IsTuple()) {
      info.elements = actual.elements;
      info.loop_tuple = actual.loop_tuple;
      info.bound_tuple = actual.bound_tuple;
    }
    for (const Var* candidate : actual.roots) {
      if (seen.insert(candidate).second) info.roots.push_back(candidate);
    }
    info.ambiguous = info.ambiguous || actual.ambiguous;
  }
  info.ambiguous = info.ambiguous || info.roots.size() > 1;
  return info;
}

void BufferRootCollector::PropagateStorageEffects(std::unordered_map<const Var*, unsigned char>& effects) {
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
      auto& instance = *projection.instance;
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
          const auto& actual = ResolveStorageBinding(instance, next->first);
          if (actual.IsTuple()) {
            for (const Var* target : CollectBackingContentRoots(actual)) add(target, bits);
          } else {
            for (const Var* target : actual.roots) add(target, bits);
          }
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

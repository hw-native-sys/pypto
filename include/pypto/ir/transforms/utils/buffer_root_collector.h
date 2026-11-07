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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_BUFFER_ROOT_COLLECTOR_H_
#define PYPTO_IR_TRANSFORMS_UTILS_BUFFER_ROOT_COLLECTOR_H_

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace buffer_root {

struct StorageFormalPath;
struct StorageReturnValue;

/// Share proven storage-return summaries while an unchanged Program is analyzed.
/// Nested collectors reuse the scope; the outermost exit releases all cached IR.
class ScopedStorageRootCache {
 public:
  ScopedStorageRootCache();
  ~ScopedStorageRootCache();
  ScopedStorageRootCache(const ScopedStorageRootCache&) = delete;
  ScopedStorageRootCache& operator=(const ScopedStorageRootCache&) = delete;

 private:
  return_lineage::ScopedTraceCache trace_cache_;
};

/// What SelectReturnRoot does for a single-return call when the owning buffer
/// cannot be pinned to one Out/InOut arg by shape+dtype (no match, or two+
/// distinct same-typed candidates). The right choice depends on the consumer:
enum class AmbiguousRootPolicy {
  /// Record no root. Safe for fusion / aliasing: never assert a false alias
  /// that could corrupt a differently-typed buffer (issue #1564 / #1580).
  kSkip,
  /// Fall back to the first Out/InOut root. Dependency analysis
  /// (DeriveCallDirections) must keep *some* root so a later write to the
  /// returned var still promotes to InOut and the WAW/InOut dependency is not
  /// dropped — a missing root silently degrades to OutputExisting.
  kFirstOutput,
};

/// Whether a view is a distinct logical footprint or shares its backing storage.
/// Fusion keeps the default footprint policy; direction inference follows storage.
enum class ViewRootPolicy { kLogicalFootprint, kBackingStorage };

/**
 * @brief Determine the canonical buffer root for every Var in a function body.
 *
 * Walks the IR and maps each Var* to the Var* that owns its underlying buffer,
 * propagating root identity through assignments, loop carries, and function
 * calls (Call and Submit alike). Pure structural analysis — no optimization
 * logic.
 *
 * For a call's single (non-tuple) return value the owning root is selected by
 * matching the return type's shape+dtype against the callee's Out/InOut args
 * (see SelectReturnRootInfo), so a differently-typed InOut scratch is never
 * mistaken for the real output buffer (issue #1564 / #1580). When that match is
 * ambiguous, the fallback is governed by @p ambiguous_policy (see
 * AmbiguousRootPolicy).
 */
class BufferRootCollector : public IRVisitor {
 public:
  explicit BufferRootCollector(ProgramPtr program,
                               AmbiguousRootPolicy ambiguous_policy = AmbiguousRootPolicy::kSkip,
                               ViewRootPolicy view_policy = ViewRootPolicy::kLogicalFootprint);

  void Initialize(const std::vector<VarPtr>& params);

  /// Var* -> owning buffer-root Var*. Populated by Initialize + VisitStmt.
  std::unordered_map<const Var*, const Var*> buffer_roots;

  /// Vars whose value may refer to more than one owning buffer root because of
  /// control flow. Consumers that derive dependencies must treat writes through
  /// these vars conservatively; aliasing optimizations leave them unmapped.
  std::unordered_set<const Var*> ambiguous_buffer_vars;

  /// Every owning-buffer-root candidate recorded for @p var, empty when the var
  /// has no lineage. This is what "conservatively" above means in practice: a
  /// write through an ambiguous var may land on *any* candidate, so a consumer
  /// deriving dependencies has to account for all of them rather than skip the
  /// var — skipping drops the dependency for every candidate at once.
  ///
  /// `buffer_roots` cannot express this: it holds one root, which is either the
  /// single unambiguous answer or (under `kFirstOutput`) an arbitrary pick.
  [[nodiscard]] const std::vector<const Var*>& RootCandidatesOf(const Var* var) const {
    static const std::vector<const Var*> kNone;
    auto it = root_candidates_.find(var);
    return it == root_candidates_.end() ? kNone : it->second;
  }

 protected:
  using RootCandidates = std::vector<const Var*>;

  // A single operand evaluation observes a stable set of SSA bindings.
  // Nested accesses share the memo; the outermost scope clears it.
  class BackingValueMemoScope {
   public:
    explicit BackingValueMemoScope(BufferRootCollector& collector);
    ~BackingValueMemoScope();
    BackingValueMemoScope(const BackingValueMemoScope&) = delete;
    BackingValueMemoScope& operator=(const BackingValueMemoScope&) = delete;

   private:
    BufferRootCollector& collector_;
  };

  // Effect analyses visit RHS uses before installing the new binding and skip
  // descriptor-only yields, while sharing all storage/control-flow propagation.
  void RecordRootCandidates(const VarPtr& var, const RootCandidates& roots, bool unresolved = false);
  void RecordAssignmentRoots(const AssignStmtPtr& assign);
  void VisitBackingStorageOperand(const ExprPtr& expr, const std::function<void(const Var*)>& record_read);
  [[nodiscard]] RootCandidates ResolveBackingContentRoots(const ExprPtr& expr);

  // Symbolic storage identities share edges until the consumer propagates effects.
  std::unordered_map<const Var*, RootCandidates> storage_successors_;

  // Propagate content effects through local storage edges and shared callees.
  void PropagateStorageEffects(std::unordered_map<const Var*, unsigned char>& effects);
  void RecordSimtLaunchRoots(const SimtLaunchStmtPtr& launch);
  void RecordIfReturnRoots(const IfStmtPtr& if_stmt);
  [[nodiscard]] std::vector<RootCandidates> InitializeLoopCarryRoots(
      const std::vector<IterArgPtr>& iter_args);
  void RecordLoopReturnRoots(const StmtPtr& body, const std::vector<IterArgPtr>& iter_args,
                             const std::vector<VarPtr>& return_vars,
                             const std::vector<RootCandidates>& init_roots, bool guaranteed_to_run,
                             bool may_repeat = true);

  void VisitStmt_(const IfStmtPtr& if_stmt) override;
  void VisitStmt_(const ForStmtPtr& for_stmt) override;
  void VisitStmt_(const WhileStmtPtr& while_stmt) override;
  void VisitStmt_(const AssignStmtPtr& assign) override;
  void VisitStmt_(const SimtLaunchStmtPtr& launch) override;

 private:
  struct LoopTupleRoots;
  struct BoundStorageTuple;
  struct StorageCallInstance;
  struct RootInfo {
    RootCandidates roots;
    bool ambiguous = false;
    std::shared_ptr<const std::vector<RootInfo>> elements = nullptr;
    LoopTupleRoots* loop_tuple = nullptr;
    BoundStorageTuple* bound_tuple = nullptr;

    [[nodiscard]] bool IsTuple() const { return elements || loop_tuple || bound_tuple; }
  };

  // Own loop tuple nodes in the collector; source edges can be cyclic.
  // A tuple field receives a distinct identity when it is first projected.
  struct LoopTupleRoots {
    std::shared_ptr<const TupleType> type;
    Span span;
    std::vector<RootInfo> sources;
    std::unordered_map<size_t, RootInfo> projected;
    const Var* aggregate = nullptr;
    const Var* formal = nullptr;
  };

  struct FormalReference {
    const Var* parameter;
    std::shared_ptr<const StorageFormalPath> path;
  };

  // A candidate output buffer: the resolved root of an Out/InOut arg, paired
  // with that arg's type so a single return value can be matched to the param
  // it actually aliases (see SelectReturnRoot).
  struct OutputRoot {
    RootInfo info;
    TypePtr type;
  };

  [[nodiscard]] RootCandidates ResolveRootCandidates(const ExprPtr& expr) const;
  [[nodiscard]] RootInfo ResolveBackingValue(const ExprPtr& expr);
  [[nodiscard]] RootInfo ResolveBackingValueUncached(const ExprPtr& expr);
  [[nodiscard]] RootInfo MergeBackingValues(const RootInfo& lhs, const RootInfo& rhs, const TypePtr& type,
                                            const Span& span);
  [[nodiscard]] RootInfo BackingCallValue(const CallPtr& call);
  [[nodiscard]] RootInfo MakeLoopStorageValue(const TypePtr& type, const Span& span, const Var* identity);
  void LinkLoopStorageValues(const RootInfo& target, const RootInfo& source);
  [[nodiscard]] RootInfo ProjectBackingElement(const RootInfo& tuple, size_t index);
  [[nodiscard]] RootCandidates CollectBackingContentRoots(const RootInfo& value);
  [[nodiscard]] const Var* AggregateLoopTupleRoots(LoopTupleRoots& tuple);
  [[nodiscard]] const Var* AggregateExplicitTupleRoots(
      const std::shared_ptr<const std::vector<RootInfo>>& elements);
  [[nodiscard]] RootInfo BindStorageValue(StorageCallInstance& instance, const StorageReturnValue& value);
  [[nodiscard]] RootInfo BindStorageField(const RootInfo& actual,
                                          const std::shared_ptr<const StorageFormalPath>& path);
  [[nodiscard]] RootInfo ProjectStorageRoot(StorageCallInstance& instance, size_t index);
  [[nodiscard]] const RootInfo& ResolveStorageBinding(StorageCallInstance& instance, size_t binding);
  [[nodiscard]] const RootInfo& ResolveStorageParameter(StorageCallInstance& instance, size_t parameter);
  [[nodiscard]] RootInfo ForwardStorageTuple(BoundStorageTuple& tuple);
  [[nodiscard]] StorageCallInstance& BindNestedStorageCall(StorageCallInstance& parent, size_t index);
  void RecordBackingValue(const VarPtr& var, const RootInfo& value);
  [[nodiscard]] bool IsAmbiguous(const ExprPtr& expr) const;
  [[nodiscard]] std::vector<OutputRoot> CollectCallOutputRoots(const CallPtr& call);

  // Pick the buffer root for a call's single (non-tuple) return value. A
  // SubWorker group may take an InOut scratch (e.g. a matmul's kv_final)
  // *before* its real Out param, so the first Out/InOut in param order is not
  // necessarily the one the return aliases. Match on the return type instead.
  // Issue #1564: without this, the FP32 scratch was fused onto the BF16 output,
  // making tensor.create -> tensor.slice(output) alias and corrupt the result.
  // When no unambiguous match exists, the fallback follows ambiguous_policy_.
  [[nodiscard]] RootInfo SelectReturnRootInfo(const std::vector<OutputRoot>& out_roots,
                                              const TypePtr& return_type) const;

  // Structural shape + dtype equality, ignoring memref / tensor_view: a return
  // value aliases its source buffer with the same logical shape and dtype.
  [[nodiscard]] static bool TypesMatchShapeDtype(const TypePtr& a, const TypePtr& b);

  [[nodiscard]] std::vector<OutputRoot> CollectProvenReturnRoots(const CallPtr& call);

  std::unique_ptr<ScopedStorageRootCache> storage_cache_;
  ProgramPtr program_;
  AmbiguousRootPolicy ambiguous_policy_;
  ViewRootPolicy view_policy_;
  std::unordered_map<const Var*, RootCandidates> root_candidates_;
  struct StorageCallProjection {
    StorageCallInstance* instance;
    size_t node;
  };
  std::vector<VarPtr> storage_proxy_vars_;
  std::vector<std::shared_ptr<StorageCallInstance>> storage_calls_;
  std::unordered_map<std::shared_ptr<const std::vector<RootInfo>>, const Var*> explicit_tuple_contents_;
  using TupleElements = std::shared_ptr<const std::vector<RootInfo>>;
  std::map<std::tuple<TupleElements, TupleElements, TypePtr>, RootInfo> merged_tuple_roots_;
  using StorageTupleIdentity = std::variant<std::shared_ptr<const std::vector<RootInfo>>,
                                            const LoopTupleRoots*, const BoundStorageTuple*>;
  std::unordered_map<StorageTupleIdentity,
                     std::unordered_map<std::shared_ptr<const StorageFormalPath>, RootInfo>>
      storage_field_projections_;
  std::unordered_map<const Var*, StorageCallProjection> storage_call_roots_;
  std::unordered_map<const Var*, std::vector<RootInfo>> tuple_output_roots_;
  std::unordered_map<const Var*, RootInfo> backing_tuple_roots_;
  std::vector<std::unique_ptr<LoopTupleRoots>> loop_tuple_roots_;
  std::unordered_map<const Var*, FormalReference> formal_projections_;
  size_t backing_value_depth_ = 0;
  std::unordered_map<ExprPtr, RootInfo> backing_value_memo_;
};

}  // namespace buffer_root
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_ROOT_COLLECTOR_H_

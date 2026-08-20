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

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace buffer_root {

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
                               AmbiguousRootPolicy ambiguous_policy = AmbiguousRootPolicy::kSkip);

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
  void VisitStmt_(const IfStmtPtr& if_stmt) override;
  void VisitStmt_(const ForStmtPtr& for_stmt) override;
  void VisitStmt_(const WhileStmtPtr& while_stmt) override;
  void VisitStmt_(const AssignStmtPtr& assign) override;

 private:
  using RootCandidates = std::vector<const Var*>;

  struct RootInfo {
    RootCandidates roots;
    bool ambiguous;
  };

  // A candidate output buffer: the resolved root of an Out/InOut arg, paired
  // with that arg's type so a single return value can be matched to the param
  // it actually aliases (see SelectReturnRoot).
  struct OutputRoot {
    RootInfo info;
    TypePtr type;
  };

  [[nodiscard]] RootCandidates ResolveRootCandidates(const ExprPtr& expr) const;
  [[nodiscard]] bool IsAmbiguous(const ExprPtr& expr) const;
  void RecordRootCandidates(const VarPtr& var, const RootCandidates& roots, bool unresolved = false);
  [[nodiscard]] std::vector<RootCandidates> InitializeLoopCarryRoots(
      const std::vector<IterArgPtr>& iter_args);
  void RecordLoopReturnRoots(const StmtPtr& body, const std::vector<VarPtr>& return_vars,
                             const std::vector<RootCandidates>& init_roots, bool guaranteed_to_run);
  [[nodiscard]] std::vector<OutputRoot> CollectCallOutputRoots(const CallPtr& call) const;

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

  ProgramPtr program_;
  AmbiguousRootPolicy ambiguous_policy_;
  std::unordered_map<const Var*, RootCandidates> root_candidates_;
  std::unordered_map<const Var*, std::vector<RootInfo>> tuple_output_roots_;
};

}  // namespace buffer_root
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_ROOT_COLLECTOR_H_

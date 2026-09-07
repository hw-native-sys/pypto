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

#include "pypto/ir/transforms/utils/scalar_output_hoist.h"

#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace outline_utils {

namespace {

/// A scalar the runtime would have to carry out of a kernel.
///
/// ``Scalar[TASK_ID]`` is excluded: it is never a kernel return value. The
/// outliner appends it to the *call's* tuple type only, and binds it at the
/// call site from ``task_<n>_outs.task_id()``.
bool IsTransportableScalar(const Var* var) {
  if (var == nullptr) return false;
  auto scalar_type = As<ScalarType>(var->GetType());
  return scalar_type != nullptr && scalar_type->dtype_ != DataType::TASK_ID;
}

/// Walk @p expr, appending every Var leaf to @p refs.
///
/// Returns false as soon as a node outside the caller-computable whitelist is
/// reached, leaving @p refs in an unspecified state (the caller discards it).
///
/// The whitelist is deliberately narrow. A Call is rejected wholesale: a scalar
/// read of device data (``tensor.read``) or an SPMD block index
/// (``tile.get_block_idx``) has no caller-side meaning, and mis-classifying one
/// would reintroduce exactly the silently-wrong value this hoist exists to
/// remove. Widening it is safe only for ops that are pure functions of their
/// scalar operands.
bool CollectCallerComputable(const ExprPtr& expr, const std::unordered_set<const Var*>& avail,
                             std::vector<const Var*>* refs) {
  if (!expr) return false;
  if (IsA<ConstInt>(expr) || IsA<ConstFloat>(expr) || IsA<ConstBool>(expr)) return true;
  // AsVarLike, not As<Var>: an enclosing loop's IterArg is a Var subclass with
  // its own ObjectKind, so As<Var> would miss it (see ir-kind-traits).
  if (auto var = AsVarLike(expr)) {
    if (avail.count(var.get()) == 0) return false;
    refs->push_back(var.get());
    return true;
  }
  if (auto bin = As<BinaryExpr>(expr)) {
    return CollectCallerComputable(bin->left_, avail, refs) &&
           CollectCallerComputable(bin->right_, avail, refs);
  }
  // Matches every unary kind, Cast included.
  if (auto un = As<UnaryExpr>(expr)) {
    return CollectCallerComputable(un->operand_, avail, refs);
  }
  return false;
}

/// The top-level statements of @p body, in order.
///
/// A non-SeqStmts body is a single statement and is treated as a one-element
/// list, so a bare ``AssignStmt`` body is still a hoist candidate.
std::vector<StmtPtr> TopLevelStmts(const StmtPtr& body) {
  if (auto seq = As<SeqStmts>(body)) return seq->stmts_;
  return {body};
}

}  // namespace

ScalarHoistPlan PlanScalarOutputHoist(const StmtPtr& body, const std::vector<const Var*>& live_in,
                                      const std::unordered_set<const Var*>& used_after) {
  ScalarHoistPlan plan;
  plan.new_body = body;
  if (!body) return plan;

  // The scalars the caller wants out of this scope, in definition order so the
  // diagnostic and the hoisted statement order are both deterministic.
  var_collectors::VarDefUseCollector defs;
  defs.VisitStmt(body);
  std::vector<const Var*> scalar_outputs;
  for (const Var* var : defs.var_defs_ordered) {
    if (used_after.count(var) > 0 && IsTransportableScalar(var)) {
      scalar_outputs.push_back(var);
    }
  }
  if (scalar_outputs.empty()) return plan;

  // Forward scan: a top-level scalar AssignStmt whose value the caller can
  // evaluate becomes a candidate and extends the available set for the
  // statements after it. ``avail`` grows only on success, so a scalar produced
  // by a device-dependent statement can never make a later one look hoistable.
  const std::vector<StmtPtr> stmts = TopLevelStmts(body);
  std::unordered_set<const Var*> avail(live_in.begin(), live_in.end());
  std::unordered_map<const Var*, size_t> candidate_index;
  std::unordered_map<size_t, std::vector<const Var*>> candidate_refs;
  for (size_t i = 0; i < stmts.size(); ++i) {
    auto assign = As<AssignStmt>(stmts[i]);
    if (!assign || !IsTransportableScalar(assign->var_.get())) continue;
    std::vector<const Var*> refs;
    if (!CollectCallerComputable(assign->value_, avail, &refs)) continue;
    candidate_index[assign->var_.get()] = i;
    candidate_refs[i] = std::move(refs);
    avail.insert(assign->var_.get());
  }

  // Transitive closure: hoist a candidate only when a scalar output actually
  // needs it, so the body gains no scalar parameter it would not otherwise use.
  std::unordered_set<size_t> needed;
  std::vector<const Var*> worklist = scalar_outputs;
  while (!worklist.empty()) {
    const Var* var = worklist.back();
    worklist.pop_back();
    auto it = candidate_index.find(var);
    if (it == candidate_index.end()) {
      // No caller-side definition: the value exists only on device.
      plan.blocker = var;
      plan.hoisted.clear();
      return plan;
    }
    if (!needed.insert(it->second).second) continue;
    for (const Var* ref : candidate_refs[it->second]) {
      if (candidate_index.count(ref) > 0) worklist.push_back(ref);
    }
  }

  // Split the body in original order.
  std::vector<StmtPtr> remaining;
  remaining.reserve(stmts.size() - needed.size());
  plan.hoisted.reserve(needed.size());
  for (size_t i = 0; i < stmts.size(); ++i) {
    if (needed.count(i) > 0) {
      plan.hoisted.push_back(stmts[i]);
    } else {
      remaining.push_back(stmts[i]);
    }
  }
  plan.new_body = SeqStmts::Flatten(std::move(remaining), body->span_);
  return plan;
}

}  // namespace outline_utils
}  // namespace ir
}  // namespace pypto

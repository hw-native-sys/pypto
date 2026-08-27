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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_ALLOC_BATCHING_H_
#define PYPTO_IR_TRANSFORMS_UTILS_ALLOC_BATCHING_H_

#include <cstddef>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto {
namespace ir {
namespace alloc_batching {

/// How many `TensorCreateInfo` operands one `alloc_tensors` call carries.
///
/// Orchestration codegen packs eligible `tensor.create`s up to this many per
/// call, and the host_build_graph runtime records one node per call. Both the
/// emitter and `LegalizeGraphBoundary`'s node-limit check reason about how many
/// nodes a region's allocations become, so the rule lives here rather than in
/// either — an estimate that drifts from the emitter is a limit check that
/// rejects legal Graphs or waves through uncacheable ones.
inline constexpr size_t kAllocTensorsArgs = 16;

/// True for a `tensor.create` that `InjectGmPipeBuffer` synthesised.
[[nodiscard]] inline bool IsInjectedGMPipeCreateVar(const VarPtr& var) {
  return var != nullptr && var->name_hint_.rfind("gm_pipe_buffer_", 0) == 0;
}

/// True when @p expr reads any of @p vars.
[[nodiscard]] inline bool ExprRefsAnyOf(const ExprPtr& expr, const std::unordered_set<const Var*>& vars) {
  if (!expr) return false;
  if (auto var = As<Var>(expr)) return vars.count(var.get()) > 0;
  if (auto bin = As<BinaryExpr>(expr)) {
    return ExprRefsAnyOf(bin->left_, vars) || ExprRefsAnyOf(bin->right_, vars);
  }
  if (auto un = As<UnaryExpr>(expr)) return ExprRefsAnyOf(un->operand_, vars);
  if (auto cast_expr = As<Cast>(expr)) return ExprRefsAnyOf(cast_expr->operand_, vars);
  return false;
}

/// The `core_num` a launch effectively carries.
///
/// `pl.spmd_submit` puts it on the dispatch call; a scope-based `pl.spmd` or
/// Group wrapper puts it on the callee. Reading only the callee under-reads a
/// direct `pl.spmd_submit(..., core_num=N)`.
[[nodiscard]] inline ExprPtr EffectiveCoreNum(const CallPtr& call, const FunctionPtr& callee) {
  ExprPtr core_num = call->GetAttr<ExprPtr>(kAttrCoreNum, nullptr);
  if (!core_num && callee) core_num = callee->GetAttr<ExprPtr>(kAttrCoreNum, nullptr);
  return core_num;
}

/// `core_num` of the launch that consumes the GM pipe buffer defined at
/// @p create_stmt_idx, or null when there is no such launch.
///
/// Walks forward to the first call taking the buffer, exactly as the emitter
/// does, and stops at a rebind of the same name.
[[nodiscard]] inline ExprPtr ResolveGMPipeCoreNum(const std::vector<StmtPtr>& stmts, size_t create_stmt_idx,
                                                  const VarPtr& create_var, const ProgramPtr& program) {
  if (!create_var || !program) return nullptr;
  for (size_t i = create_stmt_idx + 1; i < stmts.size(); ++i) {
    auto assign = As<AssignStmt>(stmts[i]);
    if (assign && assign->var_ && assign->var_.get() == create_var.get()) break;

    CallPtr call;
    if (assign) {
      call = transform_utils::AsCallOrSubmitView(assign->value_);
    } else if (auto eval = As<EvalStmt>(stmts[i])) {
      call = transform_utils::AsCallOrSubmitView(eval->expr_);
    }
    if (!call) continue;

    bool uses_create_var = false;
    for (const auto& arg : call->args_) {
      auto arg_var = AsVarLike(arg);
      if (arg_var && arg_var.get() == create_var.get()) {
        uses_create_var = true;
        break;
      }
    }
    if (!uses_create_var) continue;

    auto gv = As<GlobalVar>(call->op_);
    if (!gv) return nullptr;
    return EffectiveCoreNum(call, program->GetFunction(gv->name_));
  }
  return nullptr;
}

/// True when the GM pipe buffer at @p create_stmt_idx keeps its place in the
/// shared batch.
///
/// The emitter pulls one out when its `core_num` reads a value defined earlier
/// in the same statement list, because the create then has to stay in body
/// order. That is the single allocation-batching decision a caller cannot make
/// from the create alone, which is why it is resolved here for both of them.
[[nodiscard]] inline bool GMPipeCreateJoinsBatch(const std::vector<StmtPtr>& stmts, size_t create_stmt_idx,
                                                 const VarPtr& create_var, const ProgramPtr& program,
                                                 const std::unordered_set<const Var*>& locally_defined) {
  auto core_num = ResolveGMPipeCoreNum(stmts, create_stmt_idx, create_var, program);
  return !ExprRefsAnyOf(core_num, locally_defined);
}

/// Nodes that @p creates batchable allocations become.
[[nodiscard]] inline size_t BatchedAllocationNodes(size_t creates) {
  return (creates + kAllocTensorsArgs - 1) / kAllocTensorsArgs;
}

}  // namespace alloc_batching
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_ALLOC_BATCHING_H_

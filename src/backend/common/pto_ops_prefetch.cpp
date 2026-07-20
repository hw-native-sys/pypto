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

/**
 * @file pto_ops_prefetch.cpp
 * @brief PTO codegen for the asynchronous GM->L2 prefetch op family.
 *
 * Maps the PyPTO ``prefetch.*`` ops onto their PTOAS counterparts::
 *
 *     prefetch.make_context    -> pto.make_prefetch_async_context
 *     prefetch.async_prefetch  -> pto.tprefetch_async
 *     prefetch.session         -> pto.get_prefetch_async_session
 *     prefetch.wait            -> pto.comm.wait_async_event
 *
 * Emitted shape for the full sequence::
 *
 *     %ctx = pto.make_prefetch_async_context(%arg1 : !pto.ptr<i8>)
 *         -> !pto.prefetch_async_context
 *     %x_pview = pto.partition_view %x_view, offsets = [%c0], sizes = [%c4096]
 *         : !pto.tensor_view<4096xf32> -> !pto.partition_tensor_view<4096xf32>
 *     %evt = pto.tprefetch_async(%x_pview, %ctx
 *         : !pto.partition_tensor_view<4096xf32>, !pto.prefetch_async_context)
 *         -> !pto.async_event
 *     %sess = pto.get_prefetch_async_session %ctx
 *         : !pto.prefetch_async_context -> !pto.async_session
 *     %done = pto.comm.wait_async_event(%evt, %sess
 *         : !pto.async_event, !pto.async_session) -> i1
 *
 * The three handle types are opaque singletons in PyPTO IR — they carry no
 * shape or buffer, so each emitter defines its result under the assignment's
 * pre-bound LHS SSA name (see :func:`ResultSSA`) and publishes it through
 * ``SetCurrentExprValue`` for the enclosing statement.
 *
 * ``pto.get_prefetch_async_session`` deliberately uses the no-parenthesis
 * assembly form — it is a pure projection (``ctx.session``), unlike the other
 * three ops which take a parenthesised operand list.
 */

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/type.h"
#include "src/backend/common/pto_ops_internal.h"

namespace pypto {
namespace backend {

using pto_ops_detail::AsPto;
using pto_ops_detail::EmitPartitionViewPTO;
using pto_ops_detail::GetDimStrings;
using pto_ops_detail::GetSizeCodes;
using pto_ops_detail::MakePartitionTensorViewType;

namespace {

constexpr const char* kCtxType = "!pto.prefetch_async_context";
constexpr const char* kEventType = "!pto.async_event";
constexpr const char* kSessionType = "!pto.async_session";

/**
 * @brief SSA name to define this op's result under.
 *
 * ``GenerateFunction`` pre-binds each assignment LHS, so consumers already
 * resolve ``ctx`` / ``evt`` / ``session`` to that name — the definition must use
 * it too. Falls back to a fresh temp for an unassigned call, which is the normal
 * shape of a bare ``pl.prefetch.wait(evt, session)`` statement.
 */
std::string ResultSSA(codegen::PTOCodegen& codegen) {
  std::string target = codegen.GetCurrentResultTarget();
  return target.empty() ? codegen.NewTemp() : target;
}

/// Resolve a prefetch-handle operand (ctx / event / session) to its SSA name.
std::string GetHandleSSA(const ir::CallPtr& op, size_t index, codegen::PTOCodegen& codegen) {
  std::string ssa = codegen.GetExprAsCode(op->args_[index]);
  INTERNAL_CHECK_SPAN(!ssa.empty(), op->span_)
      << op->op_->name_ << ": operand " << index
      << " has no SSA binding; the producing prefetch op must be assigned to a named variable";
  return ssa;
}

/**
 * @brief Build a whole-tensor ``pto.partition_view`` for the prefetch source.
 *
 * PTOAS ``tprefetch_async`` accepts a memref / tensor_view / partition_view but
 * not a bare pointer, and the region is always the entire (logically 1D) tensor
 * — the IR verifier already rejected anything else. Mirrors the view/partition
 * pair that ``tile.load`` emits.
 *
 * @return A pair of (partition-view SSA, partition-view type string).
 */
std::pair<std::string, std::string> EmitWholeTensorPartitionView(const ir::CallPtr& op,
                                                                 codegen::PTOCodegen& codegen) {
  auto src = ir::AsVarLike(op->args_[0]);
  INTERNAL_CHECK_SPAN(src, op->span_)
      << "prefetch.async_prefetch src must be a Var or IterArg, got " << op->args_[0]->TypeName();
  auto tensor_type = ir::AsTensorTypeLike(src->GetType());
  INTERNAL_CHECK_SPAN(tensor_type, op->span_)
      << "prefetch.async_prefetch src must have TensorType, got " << src->GetType()->TypeName();

  const std::string tensor_view = codegen.GetOrCreateTensorView(src);
  const std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  const std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);

  const std::vector<std::string> dims = GetDimStrings(tensor_type->shape_);
  const std::vector<std::string> size_codes = GetSizeCodes(tensor_type->shape_, codegen);
  const std::string zero = codegen.GetOrEmitConstant(static_cast<int64_t>(0), DataType::INDEX);
  const std::vector<std::string> offset_codes(tensor_type->shape_.size(), zero);

  const std::string partition_type = MakePartitionTensorViewType(dims, dtype_str);
  const std::string partition_view = EmitPartitionViewPTO(src->name_hint_, tensor_view, tensor_view_type,
                                                          partition_type, offset_codes, size_codes, codegen);
  return {partition_view, partition_type};
}

}  // namespace

void RegisterPrefetchOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops) {
  auto reg = [&](const char* op_name, BackendCodegenFunc fn) {
    if (exclude_ops.count(op_name) > 0) return;
    backend.RegisterOp(op_name).f_codegen(std::move(fn));
  };

  // prefetch.make_context(workspace) -> pto.make_prefetch_async_context.
  // The workspace is an INT8 GM tensor, whose function-parameter SSA is already
  // the `!pto.ptr<i8>` that PTOAS expects — no view needed.
  reg("prefetch.make_context", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
    auto& cg = AsPto(codegen_base);
    pto_ops_detail::CheckArity(op, "pto.make_prefetch_async_context", 1);
    auto workspace = ir::AsVarLike(op->args_[0]);
    INTERNAL_CHECK_SPAN(workspace, op->span_)
        << "prefetch.make_context workspace must be a Var or IterArg, got " << op->args_[0]->TypeName();
    auto workspace_type = ir::AsTensorTypeLike(workspace->GetType());
    INTERNAL_CHECK_SPAN(workspace_type, op->span_)
        << "prefetch.make_context workspace must have TensorType, got " << workspace->GetType()->TypeName();

    const std::string ws_ptr = cg.GetTensorBasePtr(workspace);
    INTERNAL_CHECK_SPAN(!ws_ptr.empty(), op->span_)
        << "prefetch.make_context: no base pointer bound for workspace '" << workspace->name_hint_ << "'";

    const std::string ctx = ResultSSA(cg);
    cg.Emit(ctx + " = pto.make_prefetch_async_context(" + ws_ptr + " : !pto.ptr<" +
            cg.GetTypeString(workspace_type->dtype_) + ">) -> " + kCtxType);
    cg.SetCurrentExprValue(ctx);
    return "";
  });

  // prefetch.async_prefetch(src, ctx) -> pto.tprefetch_async.
  reg("prefetch.async_prefetch",
      [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
        auto& cg = AsPto(codegen_base);
        pto_ops_detail::CheckArity(op, "pto.tprefetch_async", 2);
        auto [partition_view, partition_type] = EmitWholeTensorPartitionView(op, cg);
        const std::string ctx = GetHandleSSA(op, 1, cg);

        const std::string event = ResultSSA(cg);
        cg.Emit(event + " = pto.tprefetch_async(" + partition_view + ", " + ctx + " : " + partition_type +
                ", " + kCtxType + ") -> " + kEventType);
        cg.SetCurrentExprValue(event);
        return "";
      });

  // prefetch.session(ctx) -> pto.get_prefetch_async_session (pure projection).
  reg("prefetch.session", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
    auto& cg = AsPto(codegen_base);
    pto_ops_detail::CheckArity(op, "pto.get_prefetch_async_session", 1);
    const std::string ctx = GetHandleSSA(op, 0, cg);

    const std::string session = ResultSSA(cg);
    cg.Emit(session + " = pto.get_prefetch_async_session " + ctx + " : " + kCtxType + " -> " + kSessionType);
    cg.SetCurrentExprValue(session);
    return "";
  });

  // prefetch.wait(event, session) -> pto.comm.wait_async_event.
  // Result is an `i1` done flag; PyPTO types it as a BOOL scalar.
  reg("prefetch.wait", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
    auto& cg = AsPto(codegen_base);
    pto_ops_detail::CheckArity(op, "pto.comm.wait_async_event", 2);
    const std::string event = GetHandleSSA(op, 0, cg);
    const std::string session = GetHandleSSA(op, 1, cg);

    const std::string done = ResultSSA(cg);
    cg.Emit(done + " = pto.comm.wait_async_event(" + event + ", " + session + " : " + kEventType + ", " +
            kSessionType + ") -> i1");
    cg.SetCurrentExprValue(done);
    return "";
  });
}

}  // namespace backend
}  // namespace pypto

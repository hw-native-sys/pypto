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
 * @file comm_ctx.cpp
 * @brief CommContext accessor ops — ``pld.get_comm_ctx`` and the two field reads
 *        ``pld.comm_ctx.rank`` / ``pld.comm_ctx.nranks``.
 *
 * Parser desugars ``dist_t.comm`` → ``pld.get_comm_ctx(dist_t)`` and
 * ``ctx.rank`` / ``ctx.nranks`` → the matching field-read ops, so user code
 * reads the same way as for any other Python attribute chain:
 *
 *     my_rank = data.comm.rank          # → pld.comm_ctx.rank(pld.get_comm_ctx(data))
 *     nranks  = data.comm.nranks        # → pld.comm_ctx.nranks(pld.get_comm_ctx(data))
 *
 * IR signatures:
 *
 *     pld.get_comm_ctx  (dist_t : DistributedTensor)         -> CommCtxType
 *     pld.comm_ctx.rank  (ctx     : CommCtxType)             -> Scalar<INT32>
 *     pld.comm_ctx.nranks(ctx     : CommCtxType)             -> Scalar<INT32>
 *
 * Verifier (strict per kind-trait rules — ``As<DistributedTensorType>`` does
 * NOT match a plain ``TensorType``):
 *
 * * ``get_comm_ctx`` refuses a plain ``pl.Tensor`` argument — there is no
 *   comm group attached to a non-window-bound tensor.
 * * ``rank`` / ``nranks`` refuse anything that isn't a ``CommCtxType`` — the
 *   parser desugar guarantees this, but the verifier remains as a backstop
 *   for IR built outside the parser.
 *
 * Codegen (N6+): ``get_comm_ctx`` lowers to a no-op aliasing the synthesised
 * incore ctx pointer that ``host_orch`` codegen threaded into the kernel based
 * on ``dist_t.type.window_buffer`` / ``CommGroup`` membership; the field reads
 * become byte-offset reads against ``CommContext.rankId`` /
 * ``CommContext.rankNum`` (offsets pinned in
 * ``include/pypto/codegen/distributed/comm_layout.h``).
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceGetCommCtxType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "pld.get_comm_ctx requires exactly 1 positional argument "
                             "(target: DistributedTensor), but got "
                          << args.size();
  CHECK(args[0]) << "pld.get_comm_ctx target argument must not be null";
  CHECK(kwargs.empty()) << "pld.get_comm_ctx takes no kwargs, but got " << kwargs.size();

  // target must be a DistributedTensorType — As<DistributedTensorType> is an
  // exact ObjectKind match, so plain TensorType is correctly refused here.
  CHECK(IsA<DistributedTensorType>(args[0]->GetType()))
      << "pld.get_comm_ctx target must be a DistributedTensor (window-bound), got "
      << args[0]->GetType()->TypeName();

  return GetCommCtxType();
}

TypePtr DeduceCommCtxFieldType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const std::string& op_name) {
  CHECK(args.size() == 1) << op_name << " requires exactly 1 positional argument (ctx: CommCtxType), but got "
                          << args.size();
  CHECK(args[0]) << op_name << " ctx argument must not be null";
  CHECK(kwargs.empty()) << op_name << " takes no kwargs, but got " << kwargs.size();
  CHECK(IsA<CommCtxType>(args[0]->GetType()))
      << op_name << " ctx must have CommCtxType (result of pld.get_comm_ctx), got "
      << args[0]->GetType()->TypeName();

  // rankId / rankNum are uint32_t in the runtime CommContext struct; we expose
  // them as INT32 on the IR side to match how the rest of the DSL handles
  // small integers (peer indices, loop bounds), avoiding gratuitous unsigned
  // arithmetic in user code.
  return std::make_shared<ScalarType>(DataType::INT32);
}

TypePtr DeduceCommCtxRankType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  return DeduceCommCtxFieldType(args, kwargs, "pld.comm_ctx.rank");
}

TypePtr DeduceCommCtxNranksType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  return DeduceCommCtxFieldType(args, kwargs, "pld.comm_ctx.nranks");
}

}  // namespace

// ============================================================================
// pld.get_comm_ctx — resolve the CommContext handle behind a DistributedTensor
// ============================================================================

REGISTER_OP("pld.get_comm_ctx")
    .set_description(
        "Return the CommCtx handle for the CommGroup that owns the window backing this "
        "DistributedTensor. Result is a CommCtxType (singleton marker); codegen materialises "
        "the actual device-side CommContext pointer based on the group_idx of the source "
        "WindowBuffer (filled in by CollectCommGroups).")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor (DistributedTensorType)")
    .no_memory_spec()
    .f_deduce_type(DeduceGetCommCtxType);

// ============================================================================
// pld.comm_ctx.rank — read rankId field from a CommCtx handle
// ============================================================================

REGISTER_OP("pld.comm_ctx.rank")
    .set_description(
        "Read the local rank id (rankId field) from a CommCtx handle. Returns a scalar INT32. "
        "Lowers to a byte-offset load against CommContext.rankId at codegen time.")
    .set_op_category("DistributedOp")
    .add_argument("ctx", "CommCtx handle (result of pld.get_comm_ctx)")
    .no_memory_spec()
    .f_deduce_type(DeduceCommCtxRankType);

// ============================================================================
// pld.comm_ctx.nranks — read rankNum field from a CommCtx handle
// ============================================================================

REGISTER_OP("pld.comm_ctx.nranks")
    .set_description(
        "Read the comm-group size (rankNum field) from a CommCtx handle. Returns a scalar "
        "INT32. Lowers to a byte-offset load against CommContext.rankNum at codegen time.")
    .set_op_category("DistributedOp")
    .add_argument("ctx", "CommCtx handle (result of pld.get_comm_ctx)")
    .no_memory_spec()
    .f_deduce_type(DeduceCommCtxNranksType);

}  // namespace ir
}  // namespace pypto

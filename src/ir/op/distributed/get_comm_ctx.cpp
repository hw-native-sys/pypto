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
 * @file get_comm_ctx.cpp
 * @brief Distributed comm-context ops — ``pld.system.get_comm_ctx`` and the
 *        ``pld.system.rank`` / ``pld.system.nranks`` scalar accessors.
 *
 * DSL surface (explicit op calls, no attribute-access sugar)::
 *
 *     ctx = pld.system.get_comm_ctx(data)  # CommCtx
 *     r   = pld.system.rank(ctx)           # INT32 scalar
 *     n   = pld.system.nranks(ctx)         # INT32 scalar
 *
 * All three ops use the standard 3-segment ``pld.<category>.<op>`` shape and
 * dispatch through the parser's generic ``_parse_pld_category_op`` path. The
 * unified shim (``pld.world_size(...)``, ``pld.rank(ctx)``, ...) re-exports the
 * same builders so the short form resolves to identical IR.
 *
 * Codegen recovers the originating :class:`CommGroup` from the
 * ``DistributedTensorType`` (via its ``window_buffer_`` back-reference,
 * filled in by ``MaterializeCommDomainScopes`` in N4) when lowering these ops to
 * scalar loads from the runtime ``CommContext`` struct.
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Validate that a comm-ctx op was called with exactly one non-null positional
/// arg and no kwargs. ``op_name`` is the registered op name (used as the error
/// prefix); ``arg_role`` describes the expected argument for the count message
/// (e.g. ``"a DistributedTensor"``).
void CheckUnary(const std::string& op_name, const std::string& arg_role, const std::vector<ExprPtr>& args,
                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << op_name << " requires exactly 1 positional argument (" << arg_role
                          << "), but got " << args.size();
  CHECK(kwargs.empty()) << op_name << " takes no kwargs, but got " << kwargs.size();
  CHECK(args[0]) << op_name << " argument must not be null";
}

TypePtr DeduceGetCommCtxType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CheckUnary("pld.system.get_comm_ctx", "a DistributedTensor", args, kwargs);
  // Strict ObjectKind match — As<DistributedTensorType> won't accept a plain
  // TensorType. Cross-rank metadata only exists on window-bound tensors.
  CHECK(As<DistributedTensorType>(args[0]->GetType()))
      << "pld.system.get_comm_ctx expects a DistributedTensor (window-bound), got "
      << args[0]->GetType()->TypeName();
  return GetCommCtxType();
}

/// Shared deducer for the ``pld.system.rank`` / ``pld.system.nranks`` scalar
/// accessors — same signature (single ``CommCtx`` arg, ``INT32`` result),
/// differing only in the registered op name surfaced in error messages.
TypePtr DeduceCommCtxScalarType(const std::string& op_name, const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CheckUnary(op_name, "a CommCtx", args, kwargs);
  CHECK(IsA<CommCtxType>(args[0]->GetType()))
      << op_name << " expects a CommCtx (output of pld.system.get_comm_ctx), got "
      << args[0]->GetType()->TypeName();
  return std::make_shared<ScalarType>(DataType::INT32);
}

}  // namespace

// ============================================================================
// pld.system.get_comm_ctx — lift a DistributedTensor to its CommContext handle
// ============================================================================

REGISTER_OP("pld.system.get_comm_ctx")
    .set_description(
        "Return the communication-context handle (CommCtxType) of a window-bound "
        "DistributedTensor. The result is the input to pld.system.rank / "
        "pld.system.nranks; codegen recovers the originating CommGroup from "
        "the DistributedTensor's window_buffer back-reference.")
    .set_op_category("DistributedOp")
    .add_argument("dist_tensor", "A window-bound DistributedTensor (DistributedTensorType)")
    .no_memory_spec()
    .f_deduce_type(DeduceGetCommCtxType);

// ============================================================================
// pld.system.rank — read the local rank from a CommContext
// ============================================================================

REGISTER_OP("pld.system.rank")
    .set_description(
        "Read the local rank (INT32 scalar) from a CommContext handle. Codegen "
        "lowers this to a scalar load of CommContext::rankId.")
    .set_op_category("DistributedOp")
    .add_argument("ctx", "A CommContext handle (CommCtxType)")
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceCommCtxScalarType("pld.system.rank", args, kwargs);
    });

// ============================================================================
// pld.system.nranks — read the rank count from a CommContext
// ============================================================================

REGISTER_OP("pld.system.nranks")
    .set_description(
        "Read the rank count (INT32 scalar) of the comm group from a CommContext "
        "handle. Codegen lowers this to a scalar load of CommContext::rankNum.")
    .set_op_category("DistributedOp")
    .add_argument("ctx", "A CommContext handle (CommCtxType)")
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceCommCtxScalarType("pld.system.nranks", args, kwargs);
    });

}  // namespace ir
}  // namespace pypto

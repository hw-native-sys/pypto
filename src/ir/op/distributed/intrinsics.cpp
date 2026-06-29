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
 * @file intrinsics.cpp
 * @brief ``dist.*`` distributed runtime-orchestration intrinsics.
 *
 * These ops model host-side runtime calls (make a runtime tensor, launch a
 * worker/orchestrator task, await a future, perform a tree reduction). They are
 * produced by the distributed lowering pass and consumed by the distributed
 * codegen backend (``DistributedCodegen::EmitDistIntrinsic``), which dispatches
 * the whole family by the ``dist.`` name prefix.
 *
 * Registering them here (rather than leaving them as ad-hoc ``ir.Op("dist.*")``
 * instances) means their names resolve through ``OpRegistry::GetOp`` so callers
 * can use the typo-safe ``IsOp(call, "dist.tree_reduce")`` identity check instead
 * of a bare ``name_ == "..."`` comparison.
 *
 * Their concrete result type is assigned by the lowering pass at construction
 * time, so the registered type deducer is a passthrough that returns
 * ``UnknownType`` — the registry create path is not the source of these nodes.
 */

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Passthrough deducer for runtime-orchestration intrinsics. The result type is
/// set by the distributed lowering pass when it builds the Call; the registry
/// create path is not used for these ops, so report UnknownType here.
TypePtr DeduceDistIntrinsicType(const std::vector<ExprPtr>& /*args*/,
                                const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
  return GetUnknownType();
}

}  // namespace

// ============================================================================
// dist.make_tensor — wrap a runtime handle + element count as a runtime tensor
// ============================================================================
REGISTER_OP("dist.make_tensor")
    .set_description(
        "Distributed runtime intrinsic: materialise a runtime tensor from a runtime handle and an "
        "element count. Produced by the distributed lowering pass; lowered by the distributed codegen "
        "backend.")
    .set_op_category("DistributedOp")
    .add_argument("rt", "Runtime context handle")
    .add_argument("count", "Element count (scalar expression)")
    .no_memory_spec()
    .f_deduce_type(DeduceDistIntrinsicType);

// ============================================================================
// dist.tree_reduce — tree-structured reduction over a set of leaf contributions
// ============================================================================
REGISTER_OP("dist.tree_reduce")
    .set_description(
        "Distributed runtime intrinsic: perform a tree-structured reduction over the given leaves. "
        "Produced by the distributed lowering pass; the distributed codegen backend emits a "
        "``tree_reduce(...)`` runtime call.")
    .set_op_category("DistributedOp")
    .add_argument("rt", "Runtime context handle")
    .add_argument("leaves", "Leaf contributions to reduce")
    .no_memory_spec()
    .f_deduce_type(DeduceDistIntrinsicType);

// ============================================================================
// dist.submit_worker — launch a worker task on the runtime
// ============================================================================
REGISTER_OP("dist.submit_worker")
    .set_description(
        "Distributed runtime intrinsic: launch a worker task on the runtime, returning a future "
        "handle. Produced by the distributed lowering pass.")
    .set_op_category("DistributedOp")
    .add_argument("rt", "Runtime context handle")
    .no_memory_spec()
    .f_deduce_type(DeduceDistIntrinsicType);

// ============================================================================
// dist.submit_orchestrator — launch an orchestrator task on the runtime
// ============================================================================
REGISTER_OP("dist.submit_orchestrator")
    .set_description(
        "Distributed runtime intrinsic: launch an orchestrator task on the runtime, returning a "
        "future handle. Produced by the distributed lowering pass.")
    .set_op_category("DistributedOp")
    .add_argument("rt", "Runtime context handle")
    .no_memory_spec()
    .f_deduce_type(DeduceDistIntrinsicType);

// ============================================================================
// dist.future_get — await a future handle and yield its value
// ============================================================================
REGISTER_OP("dist.future_get")
    .set_description(
        "Distributed runtime intrinsic: await a future handle and yield its resolved value. Produced "
        "by the distributed lowering pass.")
    .set_op_category("DistributedOp")
    .add_argument("future", "Future handle to await")
    .no_memory_spec()
    .f_deduce_type(DeduceDistIntrinsicType);

}  // namespace ir
}  // namespace pypto

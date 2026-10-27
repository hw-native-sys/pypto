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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_OP_PREDICATES_H_
#define PYPTO_IR_TRANSFORMS_UTILS_OP_PREDICATES_H_

#include <cstddef>
#include <optional>
#include <string>

#include "pypto/ir/expr.h"

namespace pypto {
namespace ir {
namespace op_predicates {

/// Whether tile.cast must keep its result disjoint from its source. Widening
/// casts are unsafe in place on every target. A2/A3 narrowing casts process all
/// row heads before the tails, so later output rows can overwrite unread input.
/// Callers without a resolved target must conservatively pass is_a2a3=true.
bool CastRequiresDisjointSource(const CallPtr& call, bool is_a2a3);

/// Explicit semantic storage source of a single-result Functional operator.
/// This says nothing about writes, descriptor equality, or SSA substitutability.
/// Returns nullopt for an undeclared alias or an index absent from this call.
[[nodiscard]] std::optional<size_t> StorageAliasArgIndex(const OpPtr& op, size_t arg_count);

/// Whether this Functional call writes data through argument index, including
/// kwarg-dependent effects. Storage alias declarations do not imply a write.
[[nodiscard]] bool CallWritesArg(const CallPtr& call, size_t index);

/// True if the Call targets a tpop op (tile.tpop_from_aic / tile.tpop_from_aiv).
/// Decided by the registry's CrossCoreRole, not by op-name string matching.
bool IsTPop(const CallPtr& call);

/// True if the Call targets a tpush op (tile.tpush_to_aic / tile.tpush_to_aiv).
bool IsTPush(const CallPtr& call);

/// True if the Call targets a tfree op (system.tfree_to_aic / system.tfree_to_aiv).
bool IsTFree(const CallPtr& call);

/// True if the Call targets an initialize_pipe op
/// (system.aic_initialize_pipe / system.aiv_initialize_pipe).
bool IsInitializePipe(const CallPtr& call);

/// True when the operator explicitly declares that its result shares an input's
/// underlying storage. Physical reuse chosen from liveness does not count.
/// Memory-space inheritance and in-place safety are independent contracts.
bool OutputInheritsSourceBuffer(const std::string& op_name);

/// True for builtin ops whose name is namespaced `tile.` / `tensor.` /
/// `system.` / `array.`, or for the distributed context query
/// `pld.system.get_comm_ctx`. Builtin ops are never user Functions, so they
/// carry no callee body and no Out/InOut params to trace.
///
/// This is a deliberate string-prefix *family* check, not a registry lookup:
/// it must classify op names that may not be registered (e.g. during `.pto`
/// deserialization or partway through a lowering pass). It matches an entire
/// namespace, not a single operator, so `IsOp` / `GetOp` do not apply here
/// (see `operator-identity-checks.md`). It is the single canonical home for a
/// predicate that was previously copy-pasted across the IR and codegen layers.
bool IsBuiltinOp(const std::string& op_name);

/// True for a public managed `pld.tensor.*` collective — the seven ops a user
/// writes and a lowering rail later expands (composite in an InCore body,
/// builtin dispatch in a HOST orchestrator, one local AIV task in a CHIP
/// orchestration).
///
/// These are the only `pld.` operators that legitimately *survive* inside an
/// orchestration body: they are written there by the user and live until their
/// rail runs. Every other `pld.` op is either InCore-only (`pld.tile.put`,
/// `pld.system.notify`, ...) or HOST-only (`pld.tensor.alloc_window_buffer`,
/// `pld.tensor.window`), so an orchestration body holding one is an error the
/// reference verifier still reports.
///
/// Matched by name through `IsOp`, so a typo fails loudly at the comparison
/// site (see `operator-identity-checks.md`). Shared rather than duplicated
/// because three sites must agree on the same set: the composite rail's
/// deferral, the CHIP rail's post-condition, and the reference verifier's
/// exemption.
bool IsManagedTensorCollective(const OpPtr& op);

/// True if the Call publishes data that a peer rank may read after a subsequent
/// `pld.system.notify`, so a GM `system.fence` must separate it from that notify:
///   - remote writes: `pld.tile.remote_store` / `pld.tile.put` / `pld.tensor.put`;
///   - a local `tile.store` whose destination tensor (arg 2) is window-bound
///     (`DistributedTensorType`) — a peer can `remote_load` it;
///   - a `pld.tile.get` / `pld.tensor.get` whose local destination (arg 0) is
///     window-bound — same publishing obligation as a local store;
///   - conservatively, a call to an unregistered op name (a user function whose
///     body is not analysed interprocedurally).
bool IsPublishingWrite(const CallPtr& call);

}  // namespace op_predicates
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_OP_PREDICATES_H_

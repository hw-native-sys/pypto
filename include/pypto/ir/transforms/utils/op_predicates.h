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

#include <string>

#include "pypto/ir/expr.h"

namespace pypto {
namespace ir {
namespace op_predicates {

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

/// True if `op_name` is an inherit-input view op whose output ALIASES its input's
/// buffer in place — a zero-copy reinterpretation (slice / reshape / extract /
/// transpose_view / ...). Decided by the registry:
/// `OutputMemoryInheritsInput() && IsInplaceSafe()`.
/// Excludes `tile.transpose`: it permutes data into a FRESH buffer (pto.ttrans is
/// registered not_inplace_safe()), so its output does NOT alias the input.
///
/// Used wherever a view over a buffer-less tile (e.g. a cross-core tpop result)
/// must be treated as the SAME underlying buffer: InitMemRef's buffer-less
/// propagation and the tpop lifetime / tfree finalizer.
bool IsBufferAliasingViewOp(const std::string& op_name);

/// True for builtin ops whose name is namespaced `tile.` / `tensor.` /
/// `system.` / `array.`. Builtin ops are never user Functions, so they carry
/// no callee body and no Out/InOut params to trace.
///
/// This is a deliberate string-prefix *family* check, not a registry lookup:
/// it must classify op names that may not be registered (e.g. during `.pto`
/// deserialization or partway through a lowering pass). It matches an entire
/// namespace, not a single operator, so `IsOp` / `GetOp` do not apply here
/// (see `operator-identity-checks.md`). It is the single canonical home for a
/// predicate that was previously copy-pasted across the IR and codegen layers.
bool IsBuiltinOp(const std::string& op_name);

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

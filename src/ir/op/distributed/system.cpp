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
 * @file system.cpp
 * @brief Distributed system-level synchronisation ops — pld.system.notify / pld.system.wait.
 *
 * These ops drive cross-rank synchronisation against the per-rank signal slot
 * of a window-bound :class:`DistributedTensorType` (typically a 1-D INT32
 * "signal matrix"). Both are side-effect-only and produce :class:`UnknownType`
 * — there is no SSA result for downstream consumers to read.
 *
 * IR signatures:
 *
 *     pld.system.notify(target, peer, offsets, value, *, op: int)  -> Unknown
 *     pld.system.wait  (signal, offsets, expected,    *, cmp: int) -> Unknown
 *
 * The ``op`` / ``cmp`` integers are the underlying values of
 * :enum:`NotifyOp` / :enum:`WaitCmp` (see ``include/pypto/ir/comm.h``); the
 * deducer validates the int falls within the enum range so codegen can cast
 * back without a separate guard. The DSL surface
 * (``python/pypto/language/distributed/op/system.py``) accepts the typed
 * Python enums and the parser packs ``int(value)`` into the kwarg.
 *
 * Verifier (strict per kind-trait rules — ``As<DistributedTensorType>`` does
 * NOT match a plain :class:`TensorType`):
 *
 * * ``target`` / ``signal`` must have :class:`DistributedTensorType` — refuse
 *   plain :class:`TensorType` so users cannot accidentally feed a non-window-
 *   bound tensor into a cross-rank synchronisation primitive.
 * * For ``notify``: ``peer`` and ``value`` must be :class:`ScalarType`.
 *   ``offsets`` must be a :class:`MakeTuple` of rank equal to the target rank.
 * * For ``wait``: ``expected`` must be :class:`ScalarType`. ``offsets`` must
 *   be a :class:`MakeTuple` of rank equal to the signal rank.
 */

#include <any>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

void CheckOffsetsRankMatchesTarget(const MakeTuplePtr& offsets_tuple, size_t target_rank,
                                   const std::string& op_name) {
  CHECK(offsets_tuple->elements_.size() == target_rank)
      << op_name << " offsets rank (" << offsets_tuple->elements_.size()
      << ") must match target tensor rank (" << target_rank << ")";
}

TypePtr DeduceNotifyType(const std::vector<ExprPtr>& args,
                         const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 4) << "pld.system.notify requires exactly 4 positional arguments "
                             "(target, peer, offsets, value), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.system.notify positional argument #" << i << " must not be null";
  }

  auto dist_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(dist_type) << "pld.system.notify target must be a DistributedTensor (window-bound), got "
                   << args[0]->GetType()->TypeName();

  CHECK(IsA<ScalarType>(args[1]->GetType()))
      << "pld.system.notify peer must be a scalar (rank index), got " << args[1]->GetType()->TypeName();

  auto offsets_tuple = As<MakeTuple>(args[2]);
  CHECK(offsets_tuple) << "pld.system.notify offsets must be a tuple (MakeTuple of scalars), got "
                       << args[2]->TypeName();
  CheckOffsetsRankMatchesTarget(offsets_tuple, dist_type->shape_.size(), "pld.system.notify");

  CHECK(IsA<ScalarType>(args[3]->GetType()))
      << "pld.system.notify value must be a scalar, got " << args[3]->GetType()->TypeName();

  // Validate `op` kwarg falls in the NotifyOp range — codegen casts back
  // without a separate guard.
  auto op_value = GetRequiredKwarg<int>(kwargs, "op", "pld.system.notify");
  CHECK(op_value == static_cast<int>(NotifyOp::kAtomicAdd) || op_value == static_cast<int>(NotifyOp::kSet))
      << "pld.system.notify op must be NotifyOp.AtomicAdd or NotifyOp.Set (got int " << op_value << ")";

  // Side-effect-only — no SSA result for downstream consumers.
  return GetUnknownType();
}

TypePtr DeduceWaitType(const std::vector<ExprPtr>& args,
                       const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3) << "pld.system.wait requires exactly 3 positional arguments "
                             "(signal, offsets, expected), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.system.wait positional argument #" << i << " must not be null";
  }

  auto dist_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(dist_type) << "pld.system.wait signal must be a DistributedTensor (window-bound), got "
                   << args[0]->GetType()->TypeName();

  auto offsets_tuple = As<MakeTuple>(args[1]);
  CHECK(offsets_tuple) << "pld.system.wait offsets must be a tuple (MakeTuple of scalars), got "
                       << args[1]->TypeName();
  CheckOffsetsRankMatchesTarget(offsets_tuple, dist_type->shape_.size(), "pld.system.wait");

  CHECK(IsA<ScalarType>(args[2]->GetType()))
      << "pld.system.wait expected must be a scalar, got " << args[2]->GetType()->TypeName();

  auto cmp_value = GetRequiredKwarg<int>(kwargs, "cmp", "pld.system.wait");
  CHECK(cmp_value == static_cast<int>(WaitCmp::kEq) || cmp_value == static_cast<int>(WaitCmp::kGe))
      << "pld.system.wait cmp must be WaitCmp.Eq or WaitCmp.Ge (got int " << cmp_value << ")";

  return GetUnknownType();
}

}  // namespace

// ============================================================================
// pld.system.notify — atomically signal a peer rank's slot in a DistributedTensor
// ============================================================================

REGISTER_OP("pld.system.notify")
    .set_description(
        "Cross-rank notify: write `value` to the peer rank's slot of a window-bound "
        "DistributedTensor signal matrix. `op` selects between atomic-add and set semantics. "
        "Lowers to CommRemoteOffset(ctx, peer) + addptr + make_tensor_view + partition_view + TNOTIFY at "
        "codegen.")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor signal matrix")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("offsets", "Offsets in target tensor coordinates (MakeTuple of scalars)")
    .add_argument("value", "Scalar value to deposit at the peer slot")
    .set_attr<int>("op")
    .no_memory_spec()
    .f_deduce_type(DeduceNotifyType);

// ============================================================================
// pld.system.wait — block until a local signal slot meets a threshold
// ============================================================================

REGISTER_OP("pld.system.wait")
    .set_description(
        "Cross-rank wait: block until the local slot of a window-bound DistributedTensor "
        "signal matrix satisfies `cmp` against `expected`. Lowers to TWAIT at codegen.")
    .set_op_category("DistributedOp")
    .add_argument("signal", "Window-bound DistributedTensor signal matrix")
    .add_argument("offsets", "Offsets in signal tensor coordinates (MakeTuple of scalars)")
    .add_argument("expected", "Scalar threshold value")
    .set_attr<int>("cmp")
    .no_memory_spec()
    .f_deduce_type(DeduceWaitType);

}  // namespace ir
}  // namespace pypto

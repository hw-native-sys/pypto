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
 * @brief Distributed system-level synchronisation ops — notify, wait, and defer_wait.
 *
 * These ops drive cross-rank synchronisation against the per-rank signal slot
 * of a window-bound :class:`DistributedTensorType` (typically a 1-D INT32
 * "signal matrix"). All three are side-effect-only and produce :class:`UnknownType`
 * — there is no SSA result for downstream consumers to read.
 *
 * IR signatures:
 *
 *     pld.system.notify    (target, peer, offsets, value, *, op: int)  -> Unknown
 *     pld.system.wait      (signal, offsets, expected,    *, cmp: int) -> Unknown
 *     pld.system.defer_wait(signal, offsets, expected,    *, cmp: int) -> Unknown
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
#include <cstdint>
#include <limits>
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

TypePtr DeduceDeferWaitType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3) << "pld.system.defer_wait requires exactly 3 positional arguments "
                             "(signal, offsets, expected), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.system.defer_wait positional argument #" << i << " must not be null";
  }

  auto dist_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(dist_type) << "pld.system.defer_wait signal must be a DistributedTensor (window-bound), got "
                   << args[0]->GetType()->TypeName();
  CHECK(dist_type->dtype_ == DataType::INT32)
      << "pld.system.defer_wait signal dtype must be INT32, got " << dist_type->dtype_.ToString();
  CHECK_SPAN(!dist_type->shape_.empty(), args[0]->span_)
      << "pld.system.defer_wait signal rank must be at least 1, got 0";

  auto offsets_tuple = As<MakeTuple>(args[1]);
  CHECK(offsets_tuple) << "pld.system.defer_wait offsets must be a tuple (MakeTuple of scalars), got "
                       << args[1]->TypeName();
  CheckOffsetsRankMatchesTarget(offsets_tuple, dist_type->shape_.size(), "pld.system.defer_wait");
  for (size_t i = 0; i < offsets_tuple->elements_.size(); ++i) {
    const auto offset_type = As<ScalarType>(offsets_tuple->elements_[i]->GetType());
    CHECK_SPAN(offset_type && (offset_type->dtype_.IsInt() || offset_type->dtype_ == DataType::INDEX),
               offsets_tuple->elements_[i]->span_)
        << "pld.system.defer_wait offset " << i << " must be an integer or index scalar, got "
        << (offset_type ? offset_type->dtype_.ToString()
                        : offsets_tuple->elements_[i]->GetType()->TypeName());
  }

  auto expected_type = As<ScalarType>(args[2]->GetType());
  CHECK_SPAN(expected_type && (expected_type->dtype_.IsInt() || expected_type->dtype_ == DataType::INDEX),
             args[2]->span_)
      << "pld.system.defer_wait expected must be an integer or index scalar, got "
      << (expected_type ? expected_type->dtype_.ToString() : args[2]->GetType()->TypeName());
  if (auto expected_const = As<ConstInt>(args[2])) {
    CHECK(expected_const->value_ >= 0 &&
          expected_const->value_ <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        << "pld.system.defer_wait constant expected must be in [0, INT32_MAX], got "
        << expected_const->value_;
  }

  auto cmp_value = GetRequiredKwarg<int>(kwargs, "cmp", "pld.system.defer_wait");
  CHECK(cmp_value == static_cast<int>(WaitCmp::kGe))
      << "pld.system.defer_wait only supports WaitCmp.Ge (got int " << cmp_value << ")";

  return GetUnknownType();
}

}  // namespace

// ============================================================================
// pld.system.notify — atomically signal a peer rank's slot in a DistributedTensor
// ============================================================================
//
// Core placement: deliberately undeclared, i.e. SHARED. TNOTIFY's pto-isa
// implementation is pure scalar/GM (st_atomic + dcci + dsb) and ptoas imposes
// no core or section constraint on it, so pinning it to VECTOR would be a false
// claim about the ISA.
//
// It is, however, marked `set_no_duplicate()` unconditionally — for BOTH
// `NotifyOp` forms, not just the non-idempotent atomic-add. The hazard on the
// cube lane is not double-counting but PREMATURE RELEASE FROM THE WRONG LANE: a
// notify copied onto the AIC lane can publish the signal before the AIV lane's
// TPUT has landed the data that signal releases, so the peer reads stale bytes.
// A `NotifyOp::kSet` fires that race exactly as readily as an atomic-add, so it
// needs pinning to the vector lane just as much. The flag's only consumer is
// LowerAutoVectorSplit's region placement stamp, which uses it to keep a
// region's comm ops off the cube lane.
//
// `pld.system.wait` carries no such rule: TWAIT is a poll that BLOCKS, and its
// presence on the cube lane is load-bearing — pinning it to AIV would let the
// matmul race ahead of the peer data it was waiting for.

REGISTER_OP("pld.system.notify")
    .set_description(
        "Cross-rank notify: write `value` to the peer rank's slot of a window-bound "
        "DistributedTensor signal matrix. `op` selects between atomic-add and set semantics. "
        "Lowers to inline peer-offset arithmetic + addptr + make_tensor_view + partition_view + TNOTIFY "
        "at codegen.")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor signal matrix")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("offsets", "Offsets in target tensor coordinates (MakeTuple of scalars)")
    .add_argument("value", "Scalar value to deposit at the peer slot")
    .set_attr<int>("op")
    .set_no_duplicate()
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

// ============================================================================
// pld.system.defer_wait — register a task-completion condition without blocking
// ============================================================================

REGISTER_OP("pld.system.defer_wait")
    .set_description(
        "Register a deferred completion condition on a local INT32 signal slot. The enclosing task may "
        "finish executing, but its TaskId remains incomplete until signal >= expected. This operation does "
        "not resume the kernel after the condition becomes true.")
    .set_op_category("DistributedOp")
    .add_argument("signal", "Window-bound INT32 DistributedTensor signal matrix")
    .add_argument("offsets", "Offsets in signal tensor coordinates (MakeTuple of integer/index scalars)")
    .add_argument("expected", "Integer or index scalar threshold value")
    .set_attr<int>("cmp")
    .no_memory_spec()
    .f_deduce_type(DeduceDeferWaitType);

}  // namespace ir
}  // namespace pypto

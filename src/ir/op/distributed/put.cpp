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
 * @file put.cpp
 * @brief Distributed cross-rank tensor write — ``pld.tensor.put``.
 *
 * Synchronously writes the local window-bound :class:`DistributedTensorType`
 * ``src`` into the ``peer`` rank's slice of the window-bound
 * :class:`DistributedTensorType` ``dst`` (HCCL TPUT). Both operands live at
 * the **tensor** (GM) level — the VEC staging tile that TPUT bounces through
 * is synthesised at codegen and never appears on the DSL surface — so the op
 * is a sibling of ``pld.tensor.alloc_window_buffer`` / ``pld.tensor.window``,
 * not of the tile-level ``pld.tile.remote_load`` (which produces a tile).
 *
 * IR signature::
 *
 *     pld.tensor.put(dst, peer, src, *, atomic: int) -> Unknown
 *
 * The ``atomic`` integer is the underlying value of :enum:`AtomicType`
 * (``include/pypto/ir/comm.h``); the deducer validates the int against the
 * enum range so codegen can cast back without a separate guard. The DSL
 * surface (``pld.tensor.put`` in
 * ``python/pypto/language/distributed/op/tensor_ops.py``) accepts the typed
 * Python enum and packs ``int(atomic)`` into the kwarg. Side-effect-only —
 * the op produces :class:`UnknownType`, mirroring ``pld.system.notify`` /
 * ``pld.system.wait``.
 *
 * Verifier (strict per kind-trait rules — ``As<DistributedTensorType>`` does
 * NOT match a plain :class:`TensorType`):
 *
 * * ``dst`` / ``src`` must have :class:`DistributedTensorType` — refuse plain
 *   :class:`TensorType` so a non-window-bound tensor cannot be fed into a
 *   cross-rank write.
 * * ``peer`` must be a :class:`ScalarType` expression (integer rank index).
 * * ``dst`` and ``src`` must share element type and identical positive static
 *   shape (the TPUT contract — the staging VEC buffer synthesised at codegen
 *   needs compile-time extents).
 */

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeducePutType(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3) << "pld.tensor.put requires exactly 3 positional arguments "
                             "(dst, peer, src), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.put positional argument #" << i << " must not be null";
  }

  auto dst_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(dst_type) << "pld.tensor.put dst must be a DistributedTensor (window-bound), got "
                  << args[0]->GetType()->TypeName();

  CHECK(IsA<ScalarType>(args[1]->GetType()))
      << "pld.tensor.put peer must be a scalar (rank index), got " << args[1]->GetType()->TypeName();

  auto src_type = As<DistributedTensorType>(args[2]->GetType());
  CHECK(src_type) << "pld.tensor.put src must be a DistributedTensor (window-bound), got "
                  << args[2]->GetType()->TypeName();

  // TPUT contract: dst and src cover the same region — identical element type
  // and identical positive static shape. Static shape is required because the
  // staging VEC buffer synthesised at codegen needs compile-time extents.
  CHECK(dst_type->dtype_ == src_type->dtype_)
      << "pld.tensor.put dst and src must have the same element type, got dst "
      << args[0]->GetType()->TypeName() << " vs src " << args[2]->GetType()->TypeName();

  const auto& dst_shape = dst_type->shape_;
  const auto& src_shape = src_type->shape_;
  CHECK(!dst_shape.empty()) << "pld.tensor.put requires at least one dimension on dst/src";
  CHECK(dst_shape.size() == src_shape.size()) << "pld.tensor.put dst rank (" << dst_shape.size()
                                              << ") must match src rank (" << src_shape.size() << ")";
  for (size_t i = 0; i < dst_shape.size(); ++i) {
    auto d = As<ConstInt>(dst_shape[i]);
    auto s = As<ConstInt>(src_shape[i]);
    CHECK(d && s) << "pld.tensor.put requires static (compile-time constant) shapes on dst and src; "
                     "dimension "
                  << i << " is dynamic";
    CHECK(d->value_ > 0) << "pld.tensor.put shape dimension " << i << " must be positive, got " << d->value_;
    CHECK(d->value_ == s->value_) << "pld.tensor.put dst and src must have the same static shape; dimension "
                                  << i << " differs (dst=" << d->value_ << ", src=" << s->value_ << ")";
  }

  auto atomic_value = GetRequiredKwarg<int>(kwargs, "atomic", "pld.tensor.put");
  CHECK(atomic_value == static_cast<int>(AtomicType::kNone) ||
        atomic_value == static_cast<int>(AtomicType::kAdd))
      << "pld.tensor.put atomic must be AtomicType.None_ or AtomicType.Add (got int " << atomic_value << ")";

  // Side-effect-only — no SSA result for downstream consumers.
  return GetUnknownType();
}

}  // namespace

// ============================================================================
// pld.tensor.put — synchronous cross-rank bulk write into a peer rank's slice
// ============================================================================

REGISTER_OP("pld.tensor.put")
    .set_description(
        "Cross-rank put: synchronously write the local window-bound DistributedTensor `src` "
        "into the `peer` rank's slice of the window-bound DistributedTensor `dst`. `atomic` "
        "selects plain-store vs atomic-add combine semantics. Lowers to "
        "CommRemoteOffset(ctx, peer) + addptr + make_tensor_view + partition_view (dst) + "
        "partition_view (src) + a synthesised VEC staging tile + TPUT at codegen.")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Remote (peer) window-bound DistributedTensor destination")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("src", "Local window-bound DistributedTensor source (same dtype + static shape as dst)")
    .set_attr<int>("atomic")
    .no_memory_spec()
    .f_deduce_type(DeducePutType);

}  // namespace ir
}  // namespace pypto

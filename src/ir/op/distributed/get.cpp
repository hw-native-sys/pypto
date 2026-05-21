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
 * @file get.cpp
 * @brief Distributed cross-rank tensor read — ``pld.tensor.get``.
 *
 * Synchronously reads the ``peer`` rank's slice of the window-bound
 * :class:`DistributedTensorType` ``src`` into the local window-bound
 * :class:`DistributedTensorType` ``dst`` (TGET). Semantically this is the
 * tensor-level bulk form of ``remote_load + store``: it copies remote GM into
 * local GM, while the VEC staging tile that TGET bounces through is
 * synthesised at codegen and never appears on the DSL surface.
 *
 * IR signature::
 *
 *     pld.tensor.get(dst, peer, src) -> Unknown
 *
 * Side-effect-only — the op produces :class:`UnknownType`, mirroring
 * ``pld.tensor.put`` and the sync primitives.
 *
 * Verifier (strict per kind-trait rules — ``As<DistributedTensorType>`` does
 * NOT match a plain :class:`TensorType`):
 *
 * * ``dst`` / ``src`` must have :class:`DistributedTensorType` — refuse plain
 *   :class:`TensorType` so non-window-bound tensors cannot participate in a
 *   cross-rank read.
 * * ``peer`` must be a :class:`ScalarType` expression (rank index).
 * * ``dst`` and ``src`` must share element type and identical positive static
 *   shape (the TGET contract — the staging VEC buffer synthesised at codegen
 *   needs compile-time extents).
 */

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceGetType(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3) << "pld.tensor.get requires exactly 3 positional arguments "
                             "(dst, peer, src), but got "
                          << args.size();
  CHECK(kwargs.empty()) << "pld.tensor.get does not accept keyword attributes";
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.get positional argument #" << i << " must not be null";
  }

  auto dst_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(dst_type) << "pld.tensor.get dst must be a DistributedTensor (window-bound), got "
                  << args[0]->GetType()->TypeName();

  CHECK(IsA<ScalarType>(args[1]->GetType()))
      << "pld.tensor.get peer must be a scalar (rank index), got " << args[1]->GetType()->TypeName();

  auto src_type = As<DistributedTensorType>(args[2]->GetType());
  CHECK(src_type) << "pld.tensor.get src must be a DistributedTensor (window-bound), got "
                  << args[2]->GetType()->TypeName();

  // TGET contract: dst and src cover the same region — identical element type
  // and identical positive static shape. Static shape is required because the
  // staging VEC buffer synthesised at codegen needs compile-time extents.
  CHECK(dst_type->dtype_ == src_type->dtype_)
      << "pld.tensor.get dst and src must have the same element type, got dst "
      << args[0]->GetType()->TypeName() << " vs src " << args[2]->GetType()->TypeName();

  const auto& dst_shape = dst_type->shape_;
  const auto& src_shape = src_type->shape_;
  CHECK(!dst_shape.empty()) << "pld.tensor.get requires at least one dimension on dst/src";
  CHECK(dst_shape.size() == src_shape.size()) << "pld.tensor.get dst rank (" << dst_shape.size()
                                              << ") must match src rank (" << src_shape.size() << ")";
  for (size_t i = 0; i < dst_shape.size(); ++i) {
    auto d = As<ConstInt>(dst_shape[i]);
    auto s = As<ConstInt>(src_shape[i]);
    CHECK(d && s) << "pld.tensor.get requires static (compile-time constant) shapes on dst and src; "
                     "dimension "
                  << i << " is dynamic";
    CHECK(d->value_ > 0) << "pld.tensor.get shape dimension " << i << " must be positive, got " << d->value_;
    CHECK(d->value_ == s->value_) << "pld.tensor.get dst and src must have the same static shape; dimension "
                                  << i << " differs (dst=" << d->value_ << ", src=" << s->value_ << ")";
  }

  // Side-effect-only — no SSA result for downstream consumers.
  return GetUnknownType();
}

}  // namespace

// ============================================================================
// pld.tensor.get — synchronous cross-rank bulk read from a peer rank's slice
// ============================================================================

REGISTER_OP("pld.tensor.get")
    .set_description(
        "Cross-rank get: synchronously read the `peer` rank's slice of the window-bound "
        "DistributedTensor `src` into the local window-bound DistributedTensor `dst`. "
        "Semantically equivalent to remote_load + store. Lowers to "
        "CommRemoteOffset(ctx, peer) + addptr + make_tensor_view + partition_view (src) + "
        "partition_view (dst) + a synthesised VEC staging tile + TGET at codegen.")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Local window-bound DistributedTensor destination")
    .add_argument("peer", "Peer rank index (ScalarType)")
    .add_argument("src",
                  "Remote (peer) window-bound DistributedTensor source (same dtype + static shape as dst)")
    .no_memory_spec()
    .f_deduce_type(DeduceGetType);

}  // namespace ir
}  // namespace pypto

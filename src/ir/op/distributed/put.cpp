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
 * @brief Distributed cross-rank tensor write - ``pld.tensor.put``.
 *
 * Synchronously writes the local window-bound :class:`DistributedTensorType`
 * ``src`` into the ``peer`` rank's slice of the window-bound
 * :class:`DistributedTensorType` ``dst`` (HCCL TPUT). Both operands live at
 * the **tensor** (GM) level, so the user-facing op is a sibling of
 * ``pld.tensor.alloc_window_buffer`` / ``pld.tensor.window`` rather than the
 * tile-producing ``pld.tile.remote_load``.
 *
 * The VEC staging tile that TPUT bounces through is still intentionally hidden
 * from the public DSL surface. It is materialized by ``ConvertTensorToTileOps``
 * as ``tile.create`` and threaded into the internal ``pld.tile.put`` op so the
 * normal memory allocator assigns its UB address before PTO codegen.
 *
 * IR signatures::
 *
 *     pld.tensor.put(dst, peer, src, *, atomic: int) -> Unknown
 *     pld.tensor.put(dst, peer, src, dst_offsets, src_offsets, shape,
 *                    *, atomic: int) -> Unknown
 *
 * The ``atomic`` integer is the underlying value of :enum:`AtomicType`
 * (``include/pypto/ir/comm.h``); the deducer validates the int against the
 * enum range so codegen can cast back without a separate guard. The DSL
 * surface (``pld.tensor.put`` in
 * ``python/pypto/language/distributed/op/tensor_ops.py``) accepts the typed
 * Python enum and packs ``int(atomic)`` into the kwarg. Side-effect-only -
 * the op produces :class:`UnknownType`, mirroring ``pld.system.notify`` /
 * ``pld.system.wait``.
 *
 * Verifier (strict per kind-trait rules - ``As<DistributedTensorType>`` does
 * NOT match a plain :class:`TensorType`):
 *
 * * ``dst`` / ``src`` must have :class:`DistributedTensorType` - refuse plain
 *   :class:`TensorType` so a non-window-bound tensor cannot be fed into a
 *   cross-rank write.
 * * ``peer`` must be a :class:`ScalarType` expression (integer rank index).
 * * ``dst`` and ``src`` must share element type, rank, and positive static
 *   dimensions.
 * * Full-slice puts require identical static shape. Subregion puts may use
 *   different per-rank slice extents, but ``dst_offsets``, ``src_offsets``,
 *   and ``shape`` must be rank-matched static tuples and are provided
 *   together.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

void ValidatePutContract(const ExprPtr& dst, const ExprPtr& peer, const ExprPtr& src,
                         const std::vector<std::pair<std::string, std::any>>& kwargs,
                         const std::string& op_name, bool require_same_shape) {
  CHECK(dst) << op_name << " dst argument must not be null";
  CHECK(peer) << op_name << " peer argument must not be null";
  CHECK(src) << op_name << " src argument must not be null";

  auto dst_type = As<DistributedTensorType>(dst->GetType());
  CHECK(dst_type) << op_name << " dst must be a DistributedTensor (window-bound), got "
                  << dst->GetType()->TypeName();

  CHECK(IsA<ScalarType>(peer->GetType()))
      << op_name << " peer must be a scalar (rank index), got " << peer->GetType()->TypeName();

  auto src_type = As<DistributedTensorType>(src->GetType());
  CHECK(src_type) << op_name << " src must be a DistributedTensor (window-bound), got "
                  << src->GetType()->TypeName();

  // TPUT contract: dst and src always describe same-rank window slices with
  // identical element type and static positive extents. Full-slice puts also
  // require identical shapes. Subregion puts may use different full window
  // extents, because the actual transfer extent is supplied separately as
  // `shape` and validated by ValidatePutRegionArgs.
  CHECK(dst_type->dtype_ == src_type->dtype_)
      << op_name << " dst and src must have the same element type, got dst " << dst->GetType()->TypeName()
      << " vs src " << src->GetType()->TypeName();

  const auto& dst_shape = dst_type->shape_;
  const auto& src_shape = src_type->shape_;
  CHECK(!dst_shape.empty()) << op_name << " requires at least one dimension on dst/src";
  CHECK(dst_shape.size() == src_shape.size())
      << op_name << " dst rank (" << dst_shape.size() << ") must match src rank (" << src_shape.size() << ")";
  for (size_t i = 0; i < dst_shape.size(); ++i) {
    auto d = As<ConstInt>(dst_shape[i]);
    auto s = As<ConstInt>(src_shape[i]);
    CHECK(d && s) << op_name << " requires static (compile-time constant) shapes on dst and src; dimension "
                  << i << " is dynamic";
    CHECK(d->value_ > 0) << op_name << " shape dimension " << i << " must be positive, got " << d->value_;
    CHECK(s->value_ > 0) << op_name << " src shape dimension " << i << " must be positive, got " << s->value_;
    if (require_same_shape) {
      CHECK(d->value_ == s->value_) << op_name << " dst and src must have the same static shape; dimension "
                                    << i << " differs (dst=" << d->value_ << ", src=" << s->value_ << ")";
    }
  }

  auto atomic_value = GetRequiredKwarg<int>(kwargs, "atomic", op_name);
  CHECK(atomic_value == static_cast<int>(AtomicType::kNone) ||
        atomic_value == static_cast<int>(AtomicType::kAdd))
      << op_name << " atomic must be AtomicType.None_ or AtomicType.Add (got int " << atomic_value << ")";
}

std::vector<ExprPtr> ValidatePutRegionArgs(const std::vector<ExprPtr>& args, size_t region_arg_base,
                                           const std::vector<ExprPtr>& dst_shape,
                                           const std::vector<ExprPtr>& src_shape,
                                           const std::string& op_name) {
  // Subregion TPUT keeps the tensor operands at their full window-slice
  // shapes, then narrows both partition views with explicit offsets and a
  // shared static transfer shape. The stage tile is sized to this transfer
  // shape after ConvertTensorToTileOps lowers tensor.put to tile.put.
  auto dst_offsets = As<MakeTuple>(args[region_arg_base]);
  auto src_offsets = As<MakeTuple>(args[region_arg_base + 1]);
  auto transfer_shape = As<MakeTuple>(args[region_arg_base + 2]);
  CHECK(dst_offsets) << op_name << " dst_offsets must be a tuple";
  CHECK(src_offsets) << op_name << " src_offsets must be a tuple";
  CHECK(transfer_shape) << op_name << " shape must be a tuple";
  CHECK(dst_offsets->elements_.size() == dst_shape.size())
      << op_name << " dst_offsets rank must match dst rank";
  CHECK(src_offsets->elements_.size() == src_shape.size())
      << op_name << " src_offsets rank must match src rank";
  CHECK(transfer_shape->elements_.size() == dst_shape.size())
      << op_name << " shape rank must match tensor rank";
  for (size_t i = 0; i < transfer_shape->elements_.size(); ++i) {
    auto dim = As<ConstInt>(transfer_shape->elements_[i]);
    CHECK(dim) << op_name << " shape dimensions must be static constants";
    CHECK(dim->value_ > 0) << op_name << " shape dimension " << i << " must be positive, got " << dim->value_;
    auto dst_dim = As<ConstInt>(dst_shape[i]);
    auto src_dim = As<ConstInt>(src_shape[i]);
    INTERNAL_CHECK(dst_dim && src_dim) << op_name << " tensor shapes must be static before region validation";
    if (auto dst_offset = As<ConstInt>(dst_offsets->elements_[i])) {
      CHECK(dst_offset->value_ >= 0) << op_name << " dst_offsets dimension " << i
                                     << " must be non-negative, got " << dst_offset->value_;
      CHECK(dst_offset->value_ + dim->value_ <= dst_dim->value_)
          << op_name << " dst subregion dimension " << i
          << " exceeds dst shape (offset=" << dst_offset->value_ << ", shape=" << dim->value_
          << ", dst_dim=" << dst_dim->value_ << ")";
    }
    if (auto src_offset = As<ConstInt>(src_offsets->elements_[i])) {
      CHECK(src_offset->value_ >= 0) << op_name << " src_offsets dimension " << i
                                     << " must be non-negative, got " << src_offset->value_;
      CHECK(src_offset->value_ + dim->value_ <= src_dim->value_)
          << op_name << " src subregion dimension " << i
          << " exceeds src shape (offset=" << src_offset->value_ << ", shape=" << dim->value_
          << ", src_dim=" << src_dim->value_ << ")";
    }
  }
  return transfer_shape->elements_;
}

TypePtr DeducePutType(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3 || args.size() == 6)
      << "pld.tensor.put requires 3 positional arguments (dst, peer, src) or 6 "
         "(dst, peer, src, dst_offsets, src_offsets, shape), but got "
      << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.put positional argument #" << i << " must not be null";
  }
  ValidatePutContract(args[0], args[1], args[2], kwargs, "pld.tensor.put", args.size() == 3);
  if (args.size() == 6) {
    auto dst_type = As<DistributedTensorType>(args[0]->GetType());
    auto src_type = As<DistributedTensorType>(args[2]->GetType());
    ValidatePutRegionArgs(args, 3, dst_type->shape_, src_type->shape_, "pld.tensor.put");
  }
  // Side-effect-only: no SSA result for downstream consumers.
  return GetUnknownType();
}

TypePtr DeducePutTileType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 4 || args.size() == 7)
      << "pld.tile.put requires 4 positional arguments (dst, peer, src, stage) or 7 "
         "(dst, peer, src, stage, dst_offsets, src_offsets, shape), but got "
      << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tile.put positional argument #" << i << " must not be null";
  }
  ValidatePutContract(args[0], args[1], args[2], kwargs, "pld.tile.put", args.size() == 4);

  auto stage_type = As<TileType>(args[3]->GetType());
  CHECK(stage_type) << "pld.tile.put stage must be a TileType, got " << args[3]->GetType()->TypeName();
  auto dst_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(stage_type->dtype_ == dst_type->dtype_)
      << "pld.tile.put stage dtype must match dst dtype, got stage=" << stage_type->dtype_.ToString()
      << " dst=" << dst_type->dtype_.ToString();

  auto src_type = As<DistributedTensorType>(args[2]->GetType());
  std::vector<ExprPtr> transfer_shape = dst_type->shape_;
  if (args.size() == 7) {
    transfer_shape = ValidatePutRegionArgs(args, 4, dst_type->shape_, src_type->shape_, "pld.tile.put");
  }

  // The explicit stage tile is allocated by ConvertTensorToTileOps. Its element
  // count must match the actual transfer shape, not necessarily the full dst
  // shape when subregion arguments are present.
  int64_t expected_elems = 1;
  for (const auto& dim : transfer_shape) {
    auto d = As<ConstInt>(dim);
    INTERNAL_CHECK_SPAN(d, args[0]->span_)
        << "Internal error: pld.tile.put transfer shape was not static after validation";
    expected_elems *= d->value_;
  }
  int64_t stage_elems = 1;
  for (const auto& dim : stage_type->shape_) {
    auto d = As<ConstInt>(dim);
    INTERNAL_CHECK_SPAN(d, args[3]->span_) << "Internal error: pld.tile.put stage dim is not ConstInt";
    INTERNAL_CHECK_SPAN(d->value_ > 0, args[3]->span_)
        << "Internal error: pld.tile.put stage dim not positive (" << d->value_ << ")";
    stage_elems *= d->value_;
  }
  INTERNAL_CHECK_SPAN(stage_elems == expected_elems, args[3]->span_)
      << "Internal error: pld.tile.put stage holds " << stage_elems << " elements, expected "
      << expected_elems << " (prod(transfer shape))";

  return GetUnknownType();
}

}  // namespace

// ============================================================================
// pld.tensor.put - synchronous cross-rank bulk write into a peer rank's slice
// ============================================================================

REGISTER_OP("pld.tensor.put")
    .set_description(
        "Cross-rank put: synchronously write the local window-bound DistributedTensor `src` "
        "into the `peer` rank's slice of the window-bound DistributedTensor `dst`. `atomic` "
        "selects plain-store vs atomic-add combine semantics. Lowered by ConvertTensorToTileOps "
        "to a `tile.create`-allocated VEC staging tile plus a `pld.tile.put` call, so the "
        "staging tile flows through PyPTO's memory allocator (required at --pto-level=level3).")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Remote (peer) window-bound DistributedTensor destination")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("src", "Local window-bound DistributedTensor source (same dtype as dst)")
    .set_attr<int>("atomic")
    .no_memory_spec()
    .f_deduce_type(DeducePutType);

// ============================================================================
// pld.tile.put - tile-level form with explicit VEC staging tile (post-conversion)
// ============================================================================

REGISTER_OP("pld.tile.put")
    .set_description(
        "Tile-level form of pld.tensor.put with an explicit VEC staging tile. "
        "Created by ConvertTensorToTileOps; not user-facing.")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Remote (peer) window-bound DistributedTensor destination")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("src", "Local window-bound DistributedTensor source (same dtype as dst)")
    .add_argument("stage", "VEC staging TileType (rows x cols == prod(transfer shape))")
    .set_attr<int>("atomic")
    .no_memory_spec()
    .f_deduce_type(DeducePutTileType);

}  // namespace ir
}  // namespace pypto

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for distributed ops registered via OpRegistry.

After the MemRef-mirror redesign:

* ``WindowBufferType`` is a singleton (no per-instance fields).
* ``WindowBuffer`` is a :class:`Var` subclass with no ``name``/``dtype``
  fields; it wraps a base ``Var(PtrType)`` plus a per-rank byte size and
  host-staging flags. Constructed by the comm-collection pass.
* ``pld.tensor.alloc_window_buffer(size, name=...)`` is pure-allocation and
  returns the singleton :class:`PtrType` (same as ``tile.alloc``).
* ``pld.tensor.window(buf, shape, dtype=...)`` consumes a ``Ptr`` and returns
  :class:`DistributedTensorType`; ``window_buffer`` back-reference is
  ``None`` at parse time and filled in by the comm-collection pass later.
"""

import pytest
from pypto import DataType, ir


def _make_shape_tuple(values: list[int], span: ir.Span) -> ir.MakeTuple:
    return ir.MakeTuple([ir.ConstInt(v, DataType.INT64, span) for v in values], span)


# ---------------------------------------------------------------------------
# WindowBufferType singleton
# ---------------------------------------------------------------------------


def test_window_buffer_type_is_singleton():
    """``WindowBufferType.get()`` returns a structurally-equal instance every call."""
    a = ir.WindowBufferType.get()
    b = ir.WindowBufferType.get()
    assert a is b
    assert ir.structural_equal(a, ir.WindowBufferType())


# ---------------------------------------------------------------------------
# pld.tensor.alloc_window_buffer op
# ---------------------------------------------------------------------------


def test_alloc_window_buffer_returns_ptr_type():
    """Pure-allocation: alloc returns the singleton PtrType (mirrors tile.alloc)."""
    span = ir.Span.unknown()
    size = ir.ConstInt(1024, DataType.INT64, span)
    call = ir.create_op_call(
        "pld.tensor.alloc_window_buffer",
        [size],
        {"name": "buf"},
        span,
    )
    assert isinstance(call.type, ir.PtrType)
    # Op preserves the parser-injected name kwarg for downstream consumers.
    assert call.kwargs["name"] == "buf"
    # No dtype kwarg on the op surface — alloc is dtype-agnostic.
    assert "dtype" not in call.kwargs


def test_alloc_window_buffer_requires_non_empty_name():
    span = ir.Span.unknown()
    size = ir.ConstInt(4, DataType.INT64, span)
    with pytest.raises(Exception, match="non-empty 'name'"):
        ir.create_op_call(
            "pld.tensor.alloc_window_buffer",
            [size],
            {"name": ""},
            span,
        )


# ---------------------------------------------------------------------------
# WindowBuffer Var subclass
# ---------------------------------------------------------------------------


def test_window_buffer_is_var_subclass_wrapping_ptr():
    """WindowBuffer is a Var whose type is the singleton WindowBufferType,
    wrapping a base Ptr Var (mirrors MemRef wrapping a base Ptr)."""
    span = ir.Span.unknown()
    base = ir.Var("buf", ir.PtrType(), span)
    size = ir.ConstInt(64, DataType.INT64, span)
    wb = ir.WindowBuffer(base, size, span=span)
    assert isinstance(wb, ir.Var)
    assert isinstance(wb.type, ir.WindowBufferType)
    # name_hint flows from base.name_hint — no separate name field on
    # WindowBuffer (mirrors MemRef).
    assert wb.name_hint == "buf"
    assert wb.base is base
    assert isinstance(wb.size, ir.ConstInt)
    assert wb.size.value == 64
    assert wb.load_from_host is False
    assert wb.store_to_host is False


# ---------------------------------------------------------------------------
# pld.tensor.window op
# ---------------------------------------------------------------------------


def test_window_returns_distributed_tensor_with_no_buffer_at_parse_time():
    """``pld.tensor.window(ptr, shape, dtype=...)`` returns DistributedTensorType
    with shape + dtype set; ``window_buffer`` is None until the
    comm-collection pass populates it."""
    span = ir.Span.unknown()
    base = ir.Var("buf", ir.PtrType(), span)
    shape = _make_shape_tuple([64], span)
    call = ir.create_op_call("pld.tensor.window", [base, shape], {"dtype": DataType.FP16}, span)
    assert isinstance(call.type, ir.DistributedTensorType)
    assert call.type.dtype == DataType.FP16
    assert len(call.type.shape) == 1
    assert isinstance(call.type.shape[0], ir.ConstInt)
    assert call.type.shape[0].value == 64
    # window_buffer back-reference is filled in by the comm-collection pass,
    # not by the op deducer — at parse time it is None.
    assert call.type.window_buffer is None


def test_window_rejects_non_ptr_arg():
    """A Var with a non-PtrType type cannot be passed to ``pld.tensor.window``."""
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)
    bad = ir.Var("x", tensor_type, span)
    shape = _make_shape_tuple([64], span)
    with pytest.raises(Exception, match="Ptr"):
        ir.create_op_call("pld.tensor.window", [bad, shape], {"dtype": DataType.FP32}, span)


def test_window_rejects_non_make_tuple_shape():
    span = ir.Span.unknown()
    base = ir.Var("buf", ir.PtrType(), span)
    bad_shape = ir.ConstInt(8, DataType.INT64, span)
    with pytest.raises(Exception, match="shape tuple"):
        ir.create_op_call("pld.tensor.window", [base, bad_shape], {"dtype": DataType.FP32}, span)


# ---------------------------------------------------------------------------
# DistributedTensorType.window_buffer back-reference
# ---------------------------------------------------------------------------


def test_distributed_tensor_type_distinguishes_distinct_window_buffers():
    """Same shape + dtype but different window_buffer ⇒ structurally distinct."""
    span = ir.Span.unknown()
    base_a = ir.Var("buf_a", ir.PtrType(), span)
    base_b = ir.Var("buf_b", ir.PtrType(), span)
    wb_a = ir.WindowBuffer(base_a, ir.ConstInt(32, DataType.INT64, span), span=span)
    wb_b = ir.WindowBuffer(base_b, ir.ConstInt(32, DataType.INT64, span), span=span)
    shape = [ir.ConstInt(32, DataType.INT64, span)]
    dt_a = ir.DistributedTensorType(shape, DataType.FP32, wb_a)
    dt_b = ir.DistributedTensorType(shape, DataType.FP32, wb_b)
    assert dt_a.window_buffer is wb_a
    assert dt_b.window_buffer is wb_b
    assert not ir.structural_equal(dt_a, dt_b)


def test_distributed_tensor_type_with_and_without_window_buffer_differ():
    """Param-annotation form (no buffer) and bound form (with buffer) differ."""
    span = ir.Span.unknown()
    base = ir.Var("buf", ir.PtrType(), span)
    wb = ir.WindowBuffer(base, ir.ConstInt(32, DataType.INT64, span), span=span)
    shape = [ir.ConstInt(32, DataType.INT64, span)]
    dt_param = ir.DistributedTensorType(shape, DataType.FP32)
    dt_bound = ir.DistributedTensorType(shape, DataType.FP32, wb)
    assert dt_param.window_buffer is None
    assert dt_bound.window_buffer is wb
    assert not ir.structural_equal(dt_param, dt_bound)


# ---------------------------------------------------------------------------
# pld.system.world_size op
# ---------------------------------------------------------------------------


def test_world_size_returns_int64_scalar():
    """``pld.system.world_size()`` returns a scalar INT64 — the distributed device count."""
    span = ir.Span.unknown()
    call = ir.create_op_call("pld.system.world_size", [], {}, span)
    assert isinstance(call.type, ir.ScalarType)
    assert call.type.dtype == DataType.INT64
    assert call.args == []
    assert call.kwargs == {}


def test_world_size_rejects_positional_args():
    span = ir.Span.unknown()
    with pytest.raises(Exception, match="no positional arguments"):
        ir.create_op_call("pld.system.world_size", [ir.ConstInt(0, DataType.INT64, span)], {}, span)


def test_world_size_rejects_kwargs():
    span = ir.Span.unknown()
    with pytest.raises(Exception, match="no kwargs"):
        ir.create_op_call("pld.system.world_size", [], {"foo": 1}, span)


# ---------------------------------------------------------------------------
# pld.tile.remote_load op
# ---------------------------------------------------------------------------


def _make_distributed_tensor_var(name: str, shape: list[int], dtype: DataType, span: ir.Span) -> ir.Var:
    """Build a DistributedTensor-typed Var, mimicking the parser-level binding
    produced by a ``pld.DistributedTensor[[...], dtype]`` parameter annotation
    (``window_buffer`` back-reference left None until CollectCommGroups runs)."""
    shape_exprs: list[ir.Expr] = [ir.ConstInt(v, DataType.INT64, span) for v in shape]
    return ir.Var(name, ir.DistributedTensorType(shape_exprs, dtype), span)


def test_remote_load_returns_tile_type_with_target_dtype():
    """Positive: result is TileType with the requested shape + target's dtype."""
    span = ir.Span.unknown()
    target = _make_distributed_tensor_var("data", [64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    offsets = _make_shape_tuple([0], span)
    shape = _make_shape_tuple([32], span)

    call = ir.create_op_call(
        "pld.tile.remote_load",
        [target, peer, offsets, shape],
        {},
        span,
    )
    assert isinstance(call.type, ir.TileType)
    assert call.type.dtype == DataType.FP16
    assert len(call.type.shape) == 1
    assert isinstance(call.type.shape[0], ir.ConstInt)
    assert call.type.shape[0].value == 32


def test_remote_load_rejects_plain_tensor_target():
    """Negative: a plain pl.Tensor target is refused — must be window-bound."""
    span = ir.Span.unknown()
    plain = ir.Var(
        "x",
        ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
        span,
    )
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    offsets = _make_shape_tuple([0], span)
    shape = _make_shape_tuple([32], span)

    with pytest.raises(Exception, match="DistributedTensor"):
        ir.create_op_call(
            "pld.tile.remote_load",
            [plain, peer, offsets, shape],
            {},
            span,
        )


def test_remote_load_rejects_non_scalar_peer():
    """Negative: peer must be a ScalarType expression (rank index)."""
    span = ir.Span.unknown()
    target = _make_distributed_tensor_var("data", [64], DataType.FP32, span)
    bad_peer = _make_shape_tuple([0], span)  # MakeTuple, not a scalar
    offsets = _make_shape_tuple([0], span)
    shape = _make_shape_tuple([32], span)

    with pytest.raises(Exception, match="peer must be a scalar"):
        ir.create_op_call(
            "pld.tile.remote_load",
            [target, bad_peer, offsets, shape],
            {},
            span,
        )


def test_remote_load_rejects_mismatched_offsets_rank():
    """Negative: offsets rank must match target tensor rank."""
    span = ir.Span.unknown()
    target = _make_distributed_tensor_var("data", [64, 32], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    bad_offsets = _make_shape_tuple([0], span)  # 1-D, but target is 2-D
    shape = _make_shape_tuple([32, 16], span)

    with pytest.raises(Exception, match="offsets rank"):
        ir.create_op_call(
            "pld.tile.remote_load",
            [target, peer, bad_offsets, shape],
            {},
            span,
        )


def test_remote_load_rejects_mismatched_shape_rank():
    """Negative: shape rank must match target tensor rank."""
    span = ir.Span.unknown()
    target = _make_distributed_tensor_var("data", [64, 32], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    offsets = _make_shape_tuple([0, 0], span)
    bad_shape = _make_shape_tuple([16], span)  # 1-D, but target is 2-D

    with pytest.raises(Exception, match="shape rank"):
        ir.create_op_call(
            "pld.tile.remote_load",
            [target, peer, offsets, bad_shape],
            {},
            span,
        )


def test_remote_load_rejects_non_make_tuple_offsets():
    """Negative: offsets must be a MakeTuple."""
    span = ir.Span.unknown()
    target = _make_distributed_tensor_var("data", [64], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    bad_offsets = ir.ConstInt(0, DataType.INT64, span)
    shape = _make_shape_tuple([32], span)

    with pytest.raises(Exception, match="offsets must be a tuple"):
        ir.create_op_call(
            "pld.tile.remote_load",
            [target, peer, bad_offsets, shape],
            {},
            span,
        )


# ---------------------------------------------------------------------------
# pld.system.get_comm_ctx / pld.system.rank / pld.system.nranks ops (N5)
# ---------------------------------------------------------------------------


def test_comm_ctx_type_is_singleton():
    a = ir.CommCtxType.get()
    b = ir.CommCtxType.get()
    assert a is b
    assert ir.structural_equal(a, ir.CommCtxType())


def test_get_comm_ctx_returns_comm_ctx_type():
    span = ir.Span.unknown()
    target = _make_distributed_tensor_var("data", [64], DataType.FP32, span)
    ctx = ir.create_op_call("pld.system.get_comm_ctx", [target], {}, span)
    assert isinstance(ctx.type, ir.CommCtxType)
    assert ctx.type is ir.CommCtxType.get()


def test_get_comm_ctx_rejects_plain_tensor():
    """Precise ObjectKind match — As<DistributedTensorType> refuses TensorType."""
    span = ir.Span.unknown()
    shape: list[ir.Expr] = [ir.ConstInt(64, DataType.INT64, span)]
    plain = ir.Var("x", ir.TensorType(shape, DataType.FP32), span)
    with pytest.raises(Exception, match="DistributedTensor"):
        ir.create_op_call("pld.system.get_comm_ctx", [plain], {}, span)


def test_get_comm_ctx_rejects_kwargs():
    span = ir.Span.unknown()
    target = _make_distributed_tensor_var("data", [64], DataType.FP32, span)
    with pytest.raises(Exception, match="no kwargs"):
        ir.create_op_call("pld.system.get_comm_ctx", [target], {"peer": 0}, span)


def test_get_comm_ctx_rejects_extra_positional():
    span = ir.Span.unknown()
    target = _make_distributed_tensor_var("data", [64], DataType.FP32, span)
    extra = ir.ConstInt(0, DataType.INT32, span)
    with pytest.raises(Exception, match="exactly 1 positional"):
        ir.create_op_call("pld.system.get_comm_ctx", [target, extra], {}, span)


def test_comm_ctx_rank_returns_int32_scalar():
    span = ir.Span.unknown()
    target = _make_distributed_tensor_var("data", [64], DataType.FP32, span)
    ctx = ir.create_op_call("pld.system.get_comm_ctx", [target], {}, span)
    rank = ir.create_op_call("pld.system.rank", [ctx], {}, span)
    assert isinstance(rank.type, ir.ScalarType)
    assert rank.type.dtype == DataType.INT32


def test_comm_ctx_nranks_returns_int32_scalar():
    span = ir.Span.unknown()
    target = _make_distributed_tensor_var("data", [64], DataType.FP32, span)
    ctx = ir.create_op_call("pld.system.get_comm_ctx", [target], {}, span)
    nranks = ir.create_op_call("pld.system.nranks", [ctx], {}, span)
    assert isinstance(nranks.type, ir.ScalarType)
    assert nranks.type.dtype == DataType.INT32


def test_comm_ctx_rank_rejects_non_comm_ctx_arg():
    span = ir.Span.unknown()
    not_ctx = ir.Var("n", ir.ScalarType(DataType.INT64), span)
    with pytest.raises(Exception, match="CommCtx"):
        ir.create_op_call("pld.system.rank", [not_ctx], {}, span)


def test_comm_ctx_nranks_rejects_non_comm_ctx_arg():
    span = ir.Span.unknown()
    not_ctx = ir.Var("n", ir.ScalarType(DataType.INT64), span)
    with pytest.raises(Exception, match="CommCtx"):
        ir.create_op_call("pld.system.nranks", [not_ctx], {}, span)


# ---------------------------------------------------------------------------
# pld.system.notify / pld.system.wait ops (N6 cross-rank sync)
# ---------------------------------------------------------------------------


def test_notify_returns_unknown_type():
    """Positive: notify is side-effect-only — result is UnknownType."""
    span = ir.Span.unknown()
    target = _make_distributed_tensor_var("signal", [4], DataType.INT32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    offsets = _make_shape_tuple([0], span)
    value = ir.Var("v", ir.ScalarType(DataType.INT32), span)

    call = ir.create_op_call(
        "pld.system.notify",
        [target, peer, offsets, value],
        {"op": ir.NotifyOp.AtomicAdd},
        span,
    )
    assert isinstance(call.type, ir.UnknownType)


def test_notify_rejects_plain_tensor_target():
    """Negative: a plain pl.Tensor target is refused — must be window-bound."""
    span = ir.Span.unknown()
    plain = ir.Var("x", ir.TensorType([ir.ConstInt(4, DataType.INT64, span)], DataType.INT32), span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    offsets = _make_shape_tuple([0], span)
    value = ir.Var("v", ir.ScalarType(DataType.INT32), span)

    with pytest.raises(Exception, match="DistributedTensor"):
        ir.create_op_call(
            "pld.system.notify",
            [plain, peer, offsets, value],
            {"op": ir.NotifyOp.Set},
            span,
        )


def test_notify_rejects_mismatched_offsets_rank():
    """Negative: offsets rank must match target rank."""
    span = ir.Span.unknown()
    target = _make_distributed_tensor_var("signal", [4, 2], DataType.INT32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    bad_offsets = _make_shape_tuple([0], span)  # 1-D, target is 2-D
    value = ir.Var("v", ir.ScalarType(DataType.INT32), span)

    with pytest.raises(Exception, match="offsets rank"):
        ir.create_op_call(
            "pld.system.notify",
            [target, peer, bad_offsets, value],
            {"op": ir.NotifyOp.AtomicAdd},
            span,
        )


def test_wait_returns_unknown_type():
    """Positive: wait is side-effect-only — result is UnknownType."""
    span = ir.Span.unknown()
    signal = _make_distributed_tensor_var("signal", [4], DataType.INT32, span)
    offsets = _make_shape_tuple([0], span)
    expected = ir.Var("e", ir.ScalarType(DataType.INT32), span)

    call = ir.create_op_call(
        "pld.system.wait",
        [signal, offsets, expected],
        {"cmp": ir.WaitCmp.Ge},
        span,
    )
    assert isinstance(call.type, ir.UnknownType)


def test_wait_rejects_plain_tensor_signal():
    """Negative: a plain pl.Tensor signal is refused — must be window-bound."""
    span = ir.Span.unknown()
    plain = ir.Var("x", ir.TensorType([ir.ConstInt(4, DataType.INT64, span)], DataType.INT32), span)
    offsets = _make_shape_tuple([0], span)
    expected = ir.Var("e", ir.ScalarType(DataType.INT32), span)

    with pytest.raises(Exception, match="DistributedTensor"):
        ir.create_op_call(
            "pld.system.wait",
            [plain, offsets, expected],
            {"cmp": ir.WaitCmp.Eq},
            span,
        )


# ---------------------------------------------------------------------------
# pld.tensor.put op (synchronous cross-rank bulk write — TPUT)
# ---------------------------------------------------------------------------


def test_put_returns_unknown_type():
    """Positive: put is side-effect-only — result is UnknownType."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)

    call = ir.create_op_call(
        "pld.tensor.put",
        [dst, peer, src],
        {"atomic": ir.AtomicType.Add},
        span,
    )
    assert isinstance(call.type, ir.UnknownType)


def test_put_rejects_plain_tensor_dst():
    """Negative: a plain pl.Tensor dst is refused — must be window-bound."""
    span = ir.Span.unknown()
    plain = ir.Var("x", ir.TensorType([ir.ConstInt(16, DataType.INT64, span)], DataType.FP16), span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16], DataType.FP16, span)

    with pytest.raises(Exception, match="DistributedTensor"):
        ir.create_op_call(
            "pld.tensor.put",
            [plain, peer, src],
            {"atomic": ir.AtomicType.None_},
            span,
        )


def test_put_rejects_plain_tensor_src():
    """Negative: a plain pl.Tensor src is refused — must be window-bound."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    plain = ir.Var("x", ir.TensorType([ir.ConstInt(16, DataType.INT64, span)], DataType.FP16), span)

    with pytest.raises(Exception, match="DistributedTensor"):
        ir.create_op_call(
            "pld.tensor.put",
            [dst, peer, plain],
            {"atomic": ir.AtomicType.None_},
            span,
        )


def test_put_rejects_dtype_mismatch():
    """Negative: dst and src must share element type."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16], DataType.FP32, span)

    with pytest.raises(Exception, match="element type"):
        ir.create_op_call(
            "pld.tensor.put",
            [dst, peer, src],
            {"atomic": ir.AtomicType.None_},
            span,
        )


def test_put_rejects_shape_mismatch():
    """Negative: dst and src must have the same static shape."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 32], DataType.FP16, span)

    with pytest.raises(Exception, match="static shape"):
        ir.create_op_call(
            "pld.tensor.put",
            [dst, peer, src],
            {"atomic": ir.AtomicType.Add},
            span,
        )


def test_put_rejects_non_scalar_peer():
    """Negative: peer must be a scalar rank index."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16], DataType.FP16, span)
    bad_peer = _make_distributed_tensor_var("p", [16], DataType.FP16, span)
    src = _make_distributed_tensor_var("src", [16], DataType.FP16, span)

    with pytest.raises(Exception, match="scalar"):
        ir.create_op_call(
            "pld.tensor.put",
            [dst, bad_peer, src],
            {"atomic": ir.AtomicType.None_},
            span,
        )


# ---------------------------------------------------------------------------
# pld.tensor.get op (synchronous cross-rank bulk read — TGET)
# ---------------------------------------------------------------------------


def test_get_returns_unknown_type():
    """Positive: get is side-effect-only — result is UnknownType."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)

    call = ir.create_op_call(
        "pld.tensor.get",
        [dst, peer, src],
        {},
        span,
    )
    assert isinstance(call.type, ir.UnknownType)


def test_get_rejects_unexpected_kwargs():
    """Negative: get does not accept keyword attributes."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16], DataType.FP16, span)

    with pytest.raises(Exception, match="does not accept keyword"):
        ir.create_op_call(
            "pld.tensor.get",
            [dst, peer, src],
            {"atomic": 0},
            span,
        )


def test_get_rejects_plain_tensor_dst():
    """Negative: a plain pl.Tensor dst is refused — must be window-bound."""
    span = ir.Span.unknown()
    plain = ir.Var("x", ir.TensorType([ir.ConstInt(16, DataType.INT64, span)], DataType.FP16), span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16], DataType.FP16, span)

    with pytest.raises(Exception, match="DistributedTensor"):
        ir.create_op_call(
            "pld.tensor.get",
            [plain, peer, src],
            {},
            span,
        )


def test_get_rejects_plain_tensor_src():
    """Negative: a plain pl.Tensor src is refused — must be window-bound."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    plain = ir.Var("x", ir.TensorType([ir.ConstInt(16, DataType.INT64, span)], DataType.FP16), span)

    with pytest.raises(Exception, match="DistributedTensor"):
        ir.create_op_call(
            "pld.tensor.get",
            [dst, peer, plain],
            {},
            span,
        )


def test_get_rejects_dtype_mismatch():
    """Negative: dst and src must share element type."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16], DataType.FP32, span)

    with pytest.raises(Exception, match="element type"):
        ir.create_op_call(
            "pld.tensor.get",
            [dst, peer, src],
            {},
            span,
        )


def test_get_rejects_shape_mismatch():
    """Negative: dst and src must have the same static shape."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 32], DataType.FP16, span)

    with pytest.raises(Exception, match="static shape"):
        ir.create_op_call(
            "pld.tensor.get",
            [dst, peer, src],
            {},
            span,
        )


def test_get_rejects_rank_mismatch():
    """Negative: dst and src ranks must match."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64, 4], DataType.FP16, span)

    with pytest.raises(Exception, match="rank"):
        ir.create_op_call(
            "pld.tensor.get",
            [dst, peer, src],
            {},
            span,
        )


def test_get_rejects_non_positive_static_shape():
    """Negative: dst/src static shape dims must be positive."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 0], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 0], DataType.FP16, span)

    with pytest.raises(Exception, match="positive"):
        ir.create_op_call(
            "pld.tensor.get",
            [dst, peer, src],
            {},
            span,
        )


def test_get_rejects_non_scalar_peer():
    """Negative: peer must be a scalar rank index."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16], DataType.FP16, span)
    bad_peer = _make_distributed_tensor_var("p", [16], DataType.FP16, span)
    src = _make_distributed_tensor_var("src", [16], DataType.FP16, span)

    with pytest.raises(Exception, match="scalar"):
        ir.create_op_call(
            "pld.tensor.get",
            [dst, bad_peer, src],
            {},
            span,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

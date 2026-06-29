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

from importlib import resources

import pytest
from pypto import DataType, ir
from pypto.ir.op.distributed import tensor_ops as dist_tensor_ops
from pypto.ir.op.distributed import tile_ops as dist_tile_ops
from pypto.language.distributed.op import tensor_ops as dsl_tensor_ops
from pypto.language.distributed.op.tensor_ops import _validate_chunk, _validate_pipeline


def _make_shape_tuple(values: list[int], span: ir.Span) -> ir.MakeTuple:
    return ir.MakeTuple([ir.ConstInt(v, DataType.INT64, span) for v in values], span)


def _make_tile_var(name: str, shape: list[int], dtype: DataType, span: ir.Span) -> ir.Var:
    return ir.Var(
        name,
        ir.TileType([ir.ConstInt(v, DataType.INT64, span) for v in shape], dtype),
        span,
    )


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
# pld.tensor.allreduce op
# ---------------------------------------------------------------------------


def test_tensor_allreduce_returns_src_type():
    span = ir.Span.unknown()
    src = _make_distributed_tensor_var("src", [16], DataType.FP32, span)
    signal = _make_distributed_tensor_var("signal", [4], DataType.INT32, span)
    call = dist_tensor_ops.allreduce(src, signal, op=ir.ReduceOp.Sum, span=span)
    assert call.type is src.type
    assert call.kwargs["op"] == int(ir.ReduceOp.Sum)


def test_tensor_allreduce_defaults_to_sum():
    span = ir.Span.unknown()
    src = _make_distributed_tensor_var("src", [16], DataType.FP32, span)
    signal = _make_distributed_tensor_var("signal", [4], DataType.INT32, span)
    call = dist_tensor_ops.allreduce(src, signal, span=span)
    assert call.type is src.type
    assert call.kwargs["op"] == int(ir.ReduceOp.Sum)


def test_tensor_allreduce_accepts_dynamic_shape():
    span = ir.Span.unknown()
    n = ir.Var("n", ir.ScalarType(DataType.INT64), span)
    src = ir.Var("src", ir.DistributedTensorType([n], DataType.FP32), span)
    signal = _make_distributed_tensor_var("signal", [4], DataType.INT32, span)
    call = dist_tensor_ops.allreduce(src, signal, op=ir.ReduceOp.Sum, span=span)
    assert call.type is src.type


def test_tensor_allreduce_rejects_plain_tensor_src():
    span = ir.Span.unknown()
    src = ir.Var("src", ir.TensorType([ir.ConstInt(16, DataType.INT64, span)], DataType.FP32), span)
    signal = _make_distributed_tensor_var("signal", [4], DataType.INT32, span)
    with pytest.raises(Exception, match="DistributedTensor"):
        dist_tensor_ops.allreduce(src, signal, op=ir.ReduceOp.Sum, span=span)


def test_tensor_allreduce_rejects_non_int32_signal():
    span = ir.Span.unknown()
    src = _make_distributed_tensor_var("src", [16], DataType.FP32, span)
    signal = _make_distributed_tensor_var("signal", [4], DataType.FP32, span)
    with pytest.raises(Exception, match="signal must have INT32 element type"):
        dist_tensor_ops.allreduce(src, signal, op=ir.ReduceOp.Sum, span=span)


def test_tensor_allreduce_accepts_non_rank1_signal():
    span = ir.Span.unknown()
    src = _make_distributed_tensor_var("src", [16], DataType.FP32, span)
    signal = _make_distributed_tensor_var("signal", [2, 2], DataType.INT32, span)
    call = dist_tensor_ops.allreduce(src, signal, op=ir.ReduceOp.Sum, span=span)
    assert call.type is src.type


def test_tensor_allreduce_accepts_non_fp32_target_dtype():
    span = ir.Span.unknown()
    src = _make_distributed_tensor_var("src", [16], DataType.FP16, span)
    signal = _make_distributed_tensor_var("signal", [4], DataType.INT32, span)
    call = dist_tensor_ops.allreduce(src, signal, op=ir.ReduceOp.Sum, span=span)
    assert call.type is src.type


def test_builtin_tensor_allreduce_is_internal_only():
    span = ir.Span.unknown()
    src = _make_distributed_tensor_var("src", [16], DataType.FP32, span)
    signal = _make_distributed_tensor_var("signal", [4], DataType.INT32, span)
    with pytest.raises(Exception, match="internal-only"):
        ir.create_op_call(
            "builtin.tensor.allreduce",
            [src, signal],
            {"op": int(ir.ReduceOp.Sum), "dtype": DataType.FP32},
            span,
        )


def test_builtin_tensor_allreduce_template_resource_exists():
    template_root = resources.files("pypto.runtime.builtins.collectives.allreduce")
    assert (template_root / "templates" / "entry.cpp.in").is_file()
    assert (template_root / "templates" / "kernel.cpp.in").is_file()
    assert (template_root / "templates" / "kernel_config.py.in").is_file()


# ---------------------------------------------------------------------------
# pld.tile.remote_load op
# ---------------------------------------------------------------------------


def _make_distributed_tensor_var(name: str, shape: list[int], dtype: DataType, span: ir.Span) -> ir.Var:
    """Build a DistributedTensor-typed Var, mimicking the parser-level binding
    produced by a ``pld.DistributedTensor[[...], dtype]`` parameter annotation
    (``window_buffer`` back-reference left None until MaterializeCommDomainScopes runs)."""
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
# pld.tile.remote_store op (mirrors remote_load coverage)
# ---------------------------------------------------------------------------


def test_remote_store_returns_unknown_type():
    """Positive: remote_store is side-effect-only — result is UnknownType."""
    span = ir.Span.unknown()
    tile_var = ir.Var("t", ir.TileType([32, 16], DataType.FP16), span)
    target = _make_distributed_tensor_var("data", [64, 32], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    offsets = _make_shape_tuple([0, 0], span)

    call = ir.create_op_call(
        "pld.tile.remote_store",
        [tile_var, target, peer, offsets],
        {},
        span,
    )
    assert isinstance(call.type, ir.UnknownType)


def test_remote_store_rejects_plain_tensor_target():
    """Negative: a plain pl.Tensor target is refused — must be window-bound."""
    span = ir.Span.unknown()
    tile_var = ir.Var("t", ir.TileType([32, 16], DataType.FP32), span)
    plain = ir.Var(
        "x",
        ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
        span,
    )
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    offsets = _make_shape_tuple([0], span)

    with pytest.raises(Exception, match="DistributedTensor"):
        ir.create_op_call(
            "pld.tile.remote_store",
            [tile_var, plain, peer, offsets],
            {},
            span,
        )


def test_remote_store_rejects_non_scalar_peer():
    """Negative: peer must be a ScalarType expression (rank index)."""
    span = ir.Span.unknown()
    tile_var = ir.Var("t", ir.TileType([32, 16], DataType.FP32), span)
    target = _make_distributed_tensor_var("data", [64, 32], DataType.FP32, span)
    bad_peer = _make_shape_tuple([0, 0], span)  # MakeTuple, not a scalar
    offsets = _make_shape_tuple([0, 0], span)

    with pytest.raises(Exception, match="peer must be a scalar"):
        ir.create_op_call(
            "pld.tile.remote_store",
            [tile_var, target, bad_peer, offsets],
            {},
            span,
        )


def test_remote_store_rejects_mismatched_offsets_rank():
    """Negative: offsets rank must match target tensor rank."""
    span = ir.Span.unknown()
    tile_var = ir.Var("t", ir.TileType([32, 16], DataType.FP32), span)
    target = _make_distributed_tensor_var("data", [64, 32], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    bad_offsets = _make_shape_tuple([0], span)  # 1-D, but target is 2-D

    with pytest.raises(Exception, match="offsets rank"):
        ir.create_op_call(
            "pld.tile.remote_store",
            [tile_var, target, peer, bad_offsets],
            {},
            span,
        )


def test_remote_store_rejects_non_make_tuple_offsets():
    """Negative: offsets must be a MakeTuple."""
    span = ir.Span.unknown()
    tile_var = ir.Var("t", ir.TileType([32], DataType.FP32), span)
    target = _make_distributed_tensor_var("data", [64], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    bad_offsets = ir.ConstInt(0, DataType.INT64, span)

    with pytest.raises(Exception, match="offsets must be a tuple"):
        ir.create_op_call(
            "pld.tile.remote_store",
            [tile_var, target, peer, bad_offsets],
            {},
            span,
        )


def test_remote_store_rejects_non_tile_src():
    """Negative: src_tile must have TileType (a plain tensor is refused)."""
    span = ir.Span.unknown()
    not_a_tile = ir.Var(
        "x",
        ir.TensorType([ir.ConstInt(32, DataType.INT64, span)], DataType.FP32),
        span,
    )
    target = _make_distributed_tensor_var("data", [64], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    offsets = _make_shape_tuple([0], span)

    with pytest.raises(Exception, match="src_tile must be a TileType"):
        ir.create_op_call(
            "pld.tile.remote_store",
            [not_a_tile, target, peer, offsets],
            {},
            span,
        )


def test_remote_store_rejects_dtype_mismatch():
    """Negative: src_tile dtype must match target dtype."""
    span = ir.Span.unknown()
    tile_var = ir.Var("t", ir.TileType([32], DataType.FP32), span)
    target = _make_distributed_tensor_var("data", [64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    offsets = _make_shape_tuple([0], span)

    with pytest.raises(Exception, match="dtype"):
        ir.create_op_call(
            "pld.tile.remote_store",
            [tile_var, target, peer, offsets],
            {},
            span,
        )


def test_remote_store_rejects_extra_positional():
    """Negative: the dead 5-arg shapes form is no longer accepted."""
    span = ir.Span.unknown()
    tile_var = ir.Var("t", ir.TileType([32, 16], DataType.FP32), span)
    target = _make_distributed_tensor_var("data", [64, 32], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    offsets = _make_shape_tuple([0, 0], span)
    extra_shapes = _make_shape_tuple([32, 16], span)

    with pytest.raises(Exception, match="4 positional argument"):
        ir.create_op_call(
            "pld.tile.remote_store",
            [tile_var, target, peer, offsets, extra_shapes],
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


def test_put_subregion_returns_unknown_type():
    """Positive: offset put writes matching subregions and is side-effect-only."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [8, 64], DataType.FP16, span)
    dst_offsets = _make_shape_tuple([3, 0], span)
    src_offsets = _make_shape_tuple([1, 0], span)
    shape = _make_shape_tuple([1, 64], span)

    call = ir.create_op_call(
        "pld.tensor.put",
        [dst, peer, src, dst_offsets, src_offsets, shape],
        {"atomic": ir.AtomicType.None_},
        span,
    )
    assert isinstance(call.type, ir.UnknownType)


def test_put_subregion_dynamic_shape_requires_chunk():
    """A dynamic subregion transfer extent is allowed, but needs a static chunk
    to bound the VEC staging tile — a dynamic leading dim needs chunk_rows."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    n = ir.Var("n", ir.ScalarType(DataType.INT32), span)  # dynamic transfer rows
    dst_offsets = _make_shape_tuple([0, 0], span)
    src_offsets = _make_shape_tuple([0, 0], span)
    dyn_shape = ir.MakeTuple([n, ir.ConstInt(64, DataType.INT64, span)], span)

    # Without chunk_rows → rejected (can't size the staging tile).
    with pytest.raises(Exception, match="dynamic leading transfer dim needs a static chunk_rows"):
        ir.create_op_call(
            "pld.tensor.put", [dst, peer, src, dst_offsets, src_offsets, dyn_shape], {"atomic": 0}, span
        )

    # With chunk_rows → accepted.
    call = ir.create_op_call(
        "pld.tensor.put",
        [dst, peer, src, dst_offsets, src_offsets, dyn_shape],
        {"atomic": 0, "chunk_rows": 4},
        span,
    )
    assert isinstance(call.type, ir.UnknownType)


def _make_dynamic_window_var(name: str, dim0: ir.Expr, span: ir.Span) -> ir.Var:
    """A DistributedTensor whose leading window dim is a dynamic (runtime) expr."""
    return ir.Var(
        name,
        ir.DistributedTensorType([dim0, ir.ConstInt(64, DataType.INT64, span)], DataType.FP16),
        span,
    )


def test_put_full_slice_dynamic_window_requires_chunk():
    """Full-slice put with a dynamic-shaped window (dst/src) is allowed with a
    static chunk; dst/src dynamic dims must match structurally."""
    span = ir.Span.unknown()
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    n = ir.Var("n", ir.ScalarType(DataType.INT32), span)
    dst = _make_dynamic_window_var("dst", n, span)
    src = _make_dynamic_window_var("src", n, span)  # same dynamic extent

    # Dynamic leading window dim, no chunk_rows → rejected.
    with pytest.raises(Exception, match="dynamic leading transfer dim needs a static chunk_rows"):
        ir.create_op_call("pld.tensor.put", [dst, peer, src], {"atomic": 0}, span)

    # With chunk_rows → accepted.
    call = ir.create_op_call("pld.tensor.put", [dst, peer, src], {"atomic": 0, "chunk_rows": 4}, span)
    assert isinstance(call.type, ir.UnknownType)

    # Mismatched dynamic dst/src extents → rejected by the full-slice same-shape check.
    src_mismatch = _make_dynamic_window_var("src2", ir.Var("m", ir.ScalarType(DataType.INT32), span), span)
    with pytest.raises(Exception, match="must have the same shape"):
        ir.create_op_call("pld.tensor.put", [dst, peer, src_mismatch], {"atomic": 0, "chunk_rows": 4}, span)


def test_put_ir_builder_accepts_positional_atomic_compat():
    """Compatibility: raw IR builder still accepts the old positional atomic arg."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)

    call = dist_tensor_ops.put(dst, peer, src, ir.AtomicType.Add, span=span)

    assert isinstance(call.type, ir.UnknownType)
    assert call.kwargs["atomic"] == int(ir.AtomicType.Add)


def test_put_ir_builder_packs_chunk_attrs():
    """The raw IR builder packs chunk_rows/chunk_cols int attrs, only when non-zero."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)

    call = dist_tensor_ops.put(dst, peer, src, ir.AtomicType.Add, chunk_rows=4, chunk_cols=32, span=span)
    assert call.kwargs["chunk_rows"] == 4
    assert call.kwargs["chunk_cols"] == 32

    plain = dist_tensor_ops.put(dst, peer, src, span=span)
    assert "chunk_rows" not in plain.kwargs
    assert "chunk_cols" not in plain.kwargs


def test_get_ir_builder_packs_chunk_attrs():
    """get packs only the chunk attrs that are set (row-only chunk omits chunk_cols)."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)

    call = dist_tensor_ops.get(dst, peer, src, chunk_rows=8, span=span)
    assert call.kwargs["chunk_rows"] == 8
    assert "chunk_cols" not in call.kwargs

    plain = dist_tensor_ops.get(dst, peer, src, span=span)
    assert plain.kwargs == {}


def test_negative_chunk_rejected_by_deducer():
    """The C++ deducer rejects negative chunk sizes even for direct IR calls.

    The DSL ``_validate_chunk`` guards the user surface, but ``ir.create_op_call``
    bypasses it; the deducer must still reject a negative extent so it never
    reaches stage-tile creation. 0 (= full) stays valid.
    """
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)

    with pytest.raises(Exception, match="chunk_rows must be non-negative"):
        ir.create_op_call("pld.tensor.put", [dst, peer, src], {"atomic": 0, "chunk_rows": -1}, span)
    with pytest.raises(Exception, match="chunk_cols must be non-negative"):
        ir.create_op_call("pld.tensor.get", [dst, peer, src], {"chunk_cols": -1}, span)

    # 0 = full is accepted (no raise).
    ir.create_op_call("pld.tensor.put", [dst, peer, src], {"atomic": 0, "chunk_rows": 0}, span)
    ir.create_op_call("pld.tensor.get", [dst, peer, src], {"chunk_cols": 0}, span)


def test_dsl_validate_chunk():
    """The DSL chunk_rows/chunk_cols accept non-negative static ints (0 = full)."""
    # Valid: 0 (full), positive ints — no exception.
    _validate_chunk(0, 0, "pld.tensor.put")
    _validate_chunk(4, 0, "pld.tensor.put")
    _validate_chunk(4, 32, "pld.tensor.put")


def test_dsl_validate_pipeline():
    """pipeline=True requires both chunk dims; pipeline=False is unconstrained."""
    # pipeline disabled → no constraint regardless of chunk.
    _validate_pipeline(False, 0, 0, "pld.tensor.put")
    # pipeline enabled with both chunk dims → ok.
    _validate_pipeline(True, 4, 32, "pld.tensor.put")
    # pipeline enabled missing a chunk dim → rejected.
    for cr, cc in ((0, 32), (4, 0), (0, 0)):
        with pytest.raises(ValueError, match="requires both chunk_rows>0 and chunk_cols>0"):
            _validate_pipeline(True, cr, cc, "pld.tensor.put")


def test_pipeline_packs_attr_and_requires_chunk():
    """The IR builder packs pipeline as a bool attr; the C++ deducer enforces both chunk dims."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)

    call = dist_tensor_ops.put(dst, peer, src, chunk_rows=4, chunk_cols=32, pipeline=True, span=span)
    # pipeline is a bool switch attr (like tile.load's `transpose`), not an int count.
    assert call.kwargs["pipeline"] is True

    # Deducer rejects pipeline without both chunk dims.
    with pytest.raises(Exception, match="pipeline=True requires both chunk_rows>0 and chunk_cols>0"):
        ir.create_op_call(
            "pld.tensor.put", [dst, peer, src], {"atomic": 0, "pipeline": True, "chunk_rows": 4}, span
        )


def test_dsl_pipeline_requires_chunk():
    """The DSL put/get reject pipeline=True without both chunk dims, before lowering."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)

    with pytest.raises(ValueError, match="pipeline=True requires both chunk_rows>0 and chunk_cols>0"):
        dsl_tensor_ops.put(dst, peer, src, chunk_rows=4, pipeline=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pipeline=True requires both chunk_rows>0 and chunk_cols>0"):
        dsl_tensor_ops.get(dst, peer, src, chunk_cols=32, pipeline=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be non-negative"):
        _validate_chunk(-1, 0, "pld.tensor.put")
    with pytest.raises(TypeError, match="static"):
        _validate_chunk(1.5, 0, "pld.tensor.put")  # type: ignore[arg-type]


def test_tile_put_ir_builder_accepts_positional_atomic_compat():
    """Compatibility: raw tile builder still accepts the old positional atomic arg."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    stage = ir.Var(
        "stage",
        ir.TileType(
            [ir.ConstInt(16, DataType.INT64, span), ir.ConstInt(64, DataType.INT64, span)],
            DataType.FP16,
        ),
        span,
    )

    call = dist_tile_ops.put(dst, peer, src, stage, ir.AtomicType.Add, span=span)

    assert isinstance(call.type, ir.UnknownType)
    assert call.kwargs["atomic"] == int(ir.AtomicType.Add)


def test_tile_put_rejects_non_2d_stage():
    """Negative: post-conversion put stage must be the flattened [rows, cols] tile."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [1, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [1, 64], DataType.FP16, span)
    stage = _make_tile_var("stage", [1, 1, 64], DataType.FP16, span)

    with pytest.raises(Exception, match="stage must be a 2D VEC staging tile"):
        dist_tile_ops.put(dst, peer, src, stage, atomic=ir.AtomicType.None_, span=span)


def test_tile_put_accepts_stage_smaller_than_transfer():
    """Positive: a sub-tile stage is a valid chunk — pto-isa TPUT auto-chunks it."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    stage = _make_tile_var("stage", [4, 32], DataType.FP16, span)

    call = dist_tile_ops.put(dst, peer, src, stage, atomic=ir.AtomicType.None_, span=span)
    assert isinstance(call.type, ir.UnknownType)


def test_tile_put_rejects_stage_larger_than_transfer():
    """Negative: a stage exceeding the flattened transfer in either dim is rejected."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    # 128 cols > 64 transfer cols.
    stage = _make_tile_var("stage", [16, 128], DataType.FP16, span)

    with pytest.raises(Exception, match="must fit within"):
        dist_tile_ops.put(dst, peer, src, stage, atomic=ir.AtomicType.None_, span=span)


def test_tile_put_accepts_optional_second_stage():
    """Positive: a second (ping/pong) staging tile is accepted for double-buffering."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    ping = _make_tile_var("ping", [4, 32], DataType.FP16, span)
    pong = _make_tile_var("pong", [4, 32], DataType.FP16, span)

    call = dist_tile_ops.put(dst, peer, src, ping, atomic=ir.AtomicType.None_, stage2=pong, span=span)
    assert isinstance(call.type, ir.UnknownType)
    # stage2 is threaded in as the 5th positional operand.
    assert len(call.args) == 5


def test_tile_put_rejects_second_stage_shape_mismatch():
    """Negative: ping/pong staging tiles must have identical shape (pto-isa)."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    ping = _make_tile_var("ping", [4, 32], DataType.FP16, span)
    pong = _make_tile_var("pong", [8, 32], DataType.FP16, span)  # different rows

    with pytest.raises(Exception, match="ping/pong staging tiles must have identical shape"):
        dist_tile_ops.put(dst, peer, src, ping, atomic=ir.AtomicType.None_, stage2=pong, span=span)


def test_tile_put_rejects_second_stage_dtype_mismatch():
    """Negative: the second staging tile must match dst dtype like the first."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    ping = _make_tile_var("ping", [4, 32], DataType.FP16, span)
    pong = _make_tile_var("pong", [4, 32], DataType.FP32, span)  # wrong dtype

    with pytest.raises(Exception, match="stage2 dtype must match dst dtype"):
        dist_tile_ops.put(dst, peer, src, ping, atomic=ir.AtomicType.None_, stage2=pong, span=span)


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


def test_put_accepts_plain_tensor_src():
    """Positive: plain pl.Tensor src is accepted — TPUT only needs a readable
    local GM region on the source side, no window membership required."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    plain = ir.Var("x", ir.TensorType([ir.ConstInt(16, DataType.INT64, span)], DataType.FP16), span)

    call = ir.create_op_call(
        "pld.tensor.put",
        [dst, peer, plain],
        {"atomic": ir.AtomicType.None_},
        span,
    )
    assert isinstance(call.type, ir.UnknownType)


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


def test_put_subregion_rejects_mismatched_offsets_rank():
    """Negative: subregion offsets and shape must match tensor rank."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    bad_dst_offsets = _make_shape_tuple([0], span)
    src_offsets = _make_shape_tuple([0, 0], span)
    shape = _make_shape_tuple([1, 64], span)

    with pytest.raises(Exception, match="dst_offsets rank"):
        ir.create_op_call(
            "pld.tensor.put",
            [dst, peer, src, bad_dst_offsets, src_offsets, shape],
            {"atomic": ir.AtomicType.None_},
            span,
        )


def test_put_subregion_rejects_negative_offsets():
    """Negative: static subregion offsets must be non-negative."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    dst_offsets = _make_shape_tuple([-1, 0], span)
    src_offsets = _make_shape_tuple([0, 0], span)
    shape = _make_shape_tuple([1, 64], span)

    with pytest.raises(Exception, match="dst_offsets dimension 0 must be non-negative"):
        ir.create_op_call(
            "pld.tensor.put",
            [dst, peer, src, dst_offsets, src_offsets, shape],
            {"atomic": ir.AtomicType.None_},
            span,
        )


def test_put_subregion_rejects_out_of_bounds_dst():
    """Negative: static dst offset + shape must stay within dst shape."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    dst_offsets = _make_shape_tuple([15, 0], span)
    src_offsets = _make_shape_tuple([0, 0], span)
    shape = _make_shape_tuple([2, 64], span)

    with pytest.raises(Exception, match="dst subregion dimension 0 exceeds dst shape"):
        ir.create_op_call(
            "pld.tensor.put",
            [dst, peer, src, dst_offsets, src_offsets, shape],
            {"atomic": ir.AtomicType.None_},
            span,
        )


def test_put_subregion_rejects_out_of_bounds_src():
    """Negative: static src offset + shape must stay within src shape."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    dst_offsets = _make_shape_tuple([0, 0], span)
    src_offsets = _make_shape_tuple([15, 0], span)
    shape = _make_shape_tuple([2, 64], span)

    with pytest.raises(Exception, match="src subregion dimension 0 exceeds src shape"):
        ir.create_op_call(
            "pld.tensor.put",
            [dst, peer, src, dst_offsets, src_offsets, shape],
            {"atomic": ir.AtomicType.None_},
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


def test_get_subregion_returns_unknown_type():
    """Positive: offset get reads matching subregions and is side-effect-only."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [8, 64], DataType.FP16, span)
    dst_offsets = _make_shape_tuple([3, 0], span)
    src_offsets = _make_shape_tuple([1, 0], span)
    shape = _make_shape_tuple([1, 64], span)

    call = ir.create_op_call(
        "pld.tensor.get",
        [dst, peer, src, dst_offsets, src_offsets, shape],
        {},
        span,
    )
    assert isinstance(call.type, ir.UnknownType)


def test_get_subregion_dynamic_shape_requires_chunk():
    """A dynamic subregion transfer extent is allowed with a static chunk — a
    dynamic innermost dim needs chunk_cols to bound the VEC staging tile."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    m = ir.Var("m", ir.ScalarType(DataType.INT32), span)  # dynamic transfer cols
    dst_offsets = _make_shape_tuple([0, 0], span)
    src_offsets = _make_shape_tuple([0, 0], span)
    dyn_shape = ir.MakeTuple([ir.ConstInt(16, DataType.INT64, span), m], span)

    # Without chunk_cols → rejected.
    with pytest.raises(Exception, match="dynamic innermost transfer dim needs a static chunk_cols"):
        ir.create_op_call("pld.tensor.get", [dst, peer, src, dst_offsets, src_offsets, dyn_shape], {}, span)

    # With chunk_cols → accepted.
    call = ir.create_op_call(
        "pld.tensor.get",
        [dst, peer, src, dst_offsets, src_offsets, dyn_shape],
        {"chunk_cols": 32},
        span,
    )
    assert isinstance(call.type, ir.UnknownType)


def test_get_full_slice_dynamic_window_requires_chunk():
    """Full-slice get with a dynamic-shaped window (dst/src) is allowed with a
    static chunk to bound the staging tile."""
    span = ir.Span.unknown()
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    n = ir.Var("n", ir.ScalarType(DataType.INT32), span)
    dst = _make_dynamic_window_var("dst", n, span)
    src = _make_dynamic_window_var("src", n, span)

    with pytest.raises(Exception, match="dynamic leading transfer dim needs a static chunk_rows"):
        ir.create_op_call("pld.tensor.get", [dst, peer, src], {}, span)

    call = ir.create_op_call("pld.tensor.get", [dst, peer, src], {"chunk_rows": 4}, span)
    assert isinstance(call.type, ir.UnknownType)


def test_tile_get_returns_unknown_type():
    """Positive: tile get is the post-conversion form with an explicit stage tile."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    stage = _make_tile_var("stage", [16, 64], DataType.FP16, span)

    call = dist_tile_ops.get(dst, peer, src, stage, span=span)

    assert isinstance(call.type, ir.UnknownType)
    assert call.kwargs == {}


def test_tile_get_subregion_returns_unknown_type():
    """Positive: tile get subregions validate the explicit stage against transfer shape."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [8, 64], DataType.FP16, span)
    stage = _make_tile_var("stage", [1, 64], DataType.FP16, span)

    call = dist_tile_ops.get(
        dst,
        peer,
        src,
        stage,
        dst_offsets=[3, 0],
        src_offsets=[1, 0],
        shape=[1, 64],
        span=span,
    )

    assert isinstance(call.type, ir.UnknownType)
    assert len(call.args) == 7


def test_tile_get_rejects_stage_dtype_mismatch():
    """Negative: post-conversion get stage dtype must match dst/src dtype."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    stage = _make_tile_var("stage", [16, 64], DataType.FP32, span)

    with pytest.raises(Exception, match="stage dtype must match dst dtype"):
        dist_tile_ops.get(dst, peer, src, stage, span=span)


def test_tile_get_accepts_stage_smaller_than_transfer():
    """Positive: a sub-tile stage is a valid chunk — pto-isa TGET auto-chunks it."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    # Stage holds 8x64 < transfer 16x64 — fits in both flattened dims.
    stage = _make_tile_var("stage", [8, 64], DataType.FP16, span)

    call = dist_tile_ops.get(dst, peer, src, stage, span=span)
    assert isinstance(call.type, ir.UnknownType)


def test_tile_get_rejects_stage_larger_than_transfer():
    """Negative: a stage exceeding the flattened transfer in either dim is rejected."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    # 32 rows > 16 transfer rows.
    stage = _make_tile_var("stage", [32, 64], DataType.FP16, span)

    with pytest.raises(Exception, match="must fit within"):
        dist_tile_ops.get(dst, peer, src, stage, span=span)


def test_tile_get_rejects_non_2d_stage():
    """Negative: post-conversion get stage must be the flattened [rows, cols] tile."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [1, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [1, 64], DataType.FP16, span)
    stage = _make_tile_var("stage", [1, 1, 64], DataType.FP16, span)

    with pytest.raises(Exception, match="stage must be a 2D VEC staging tile"):
        dist_tile_ops.get(dst, peer, src, stage, span=span)


def test_get_rejects_unexpected_kwargs():
    """Negative: get accepts only chunk_rows/chunk_cols attrs — others are rejected."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16], DataType.FP16, span)

    with pytest.raises(Exception, match="Unknown kwarg 'atomic'"):
        ir.create_op_call(
            "pld.tensor.get",
            [dst, peer, src],
            {"atomic": 0},
            span,
        )


def test_get_accepts_plain_tensor_dst():
    """Positive: plain pl.Tensor dst is accepted — TGET only needs a writable
    local GM region to receive into."""
    span = ir.Span.unknown()
    plain = ir.Var("x", ir.TensorType([ir.ConstInt(16, DataType.INT64, span)], DataType.FP16), span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16], DataType.FP16, span)

    call = ir.create_op_call(
        "pld.tensor.get",
        [plain, peer, src],
        {},
        span,
    )
    assert isinstance(call.type, ir.UnknownType)


def test_tile_get_accepts_plain_tensor_dst():
    """Positive: plain pl.Tensor dst is accepted by pld.tile.get — TGET only
    needs a writable local GM region to receive into."""
    span = ir.Span.unknown()
    plain = ir.Var("x", ir.TensorType([ir.ConstInt(16, DataType.INT64, span)], DataType.FP16), span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16], DataType.FP16, span)
    stage = _make_tile_var("stage", [1, 16], DataType.FP16, span)

    call = dist_tile_ops.get(plain, peer, src, stage, span=span)

    assert isinstance(call.type, ir.UnknownType)


def test_tile_get_accepts_optional_second_stage():
    """Positive: a second (ping/pong) staging tile is accepted for double-buffering."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    ping = _make_tile_var("ping", [4, 32], DataType.FP16, span)
    pong = _make_tile_var("pong", [4, 32], DataType.FP16, span)

    call = dist_tile_ops.get(dst, peer, src, ping, stage2=pong, span=span)
    assert isinstance(call.type, ir.UnknownType)
    assert len(call.args) == 5


def test_tile_get_rejects_second_stage_shape_mismatch():
    """Negative: ping/pong staging tiles must have identical shape (pto-isa)."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    ping = _make_tile_var("ping", [4, 32], DataType.FP16, span)
    pong = _make_tile_var("pong", [4, 16], DataType.FP16, span)  # different cols

    with pytest.raises(Exception, match="ping/pong staging tiles must have identical shape"):
        dist_tile_ops.get(dst, peer, src, ping, stage2=pong, span=span)


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


def test_get_subregion_rejects_mismatched_offsets_rank():
    """Negative: subregion offsets and shape must match tensor rank."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    bad_dst_offsets = _make_shape_tuple([0], span)
    src_offsets = _make_shape_tuple([0, 0], span)
    shape = _make_shape_tuple([1, 64], span)

    with pytest.raises(Exception, match="dst_offsets rank"):
        ir.create_op_call(
            "pld.tensor.get",
            [dst, peer, src, bad_dst_offsets, src_offsets, shape],
            {},
            span,
        )


def test_get_subregion_rejects_negative_offsets():
    """Negative: static subregion offsets must be non-negative."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    dst_offsets = _make_shape_tuple([-1, 0], span)
    src_offsets = _make_shape_tuple([0, 0], span)
    shape = _make_shape_tuple([1, 64], span)

    with pytest.raises(Exception, match="dst_offsets dimension 0 must be non-negative"):
        ir.create_op_call(
            "pld.tensor.get",
            [dst, peer, src, dst_offsets, src_offsets, shape],
            {},
            span,
        )


def test_get_subregion_rejects_out_of_bounds_dst():
    """Negative: static dst offset + shape must stay within dst shape."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    dst_offsets = _make_shape_tuple([15, 0], span)
    src_offsets = _make_shape_tuple([0, 0], span)
    shape = _make_shape_tuple([2, 64], span)

    with pytest.raises(Exception, match="dst subregion dimension 0 exceeds dst shape"):
        ir.create_op_call(
            "pld.tensor.get",
            [dst, peer, src, dst_offsets, src_offsets, shape],
            {},
            span,
        )


def test_get_subregion_rejects_out_of_bounds_src():
    """Negative: static src offset + shape must stay within src shape."""
    span = ir.Span.unknown()
    dst = _make_distributed_tensor_var("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _make_distributed_tensor_var("src", [16, 64], DataType.FP16, span)
    dst_offsets = _make_shape_tuple([0, 0], span)
    src_offsets = _make_shape_tuple([15, 0], span)
    shape = _make_shape_tuple([2, 64], span)

    with pytest.raises(Exception, match="src subregion dimension 0 exceeds src shape"):
        ir.create_op_call(
            "pld.tensor.get",
            [dst, peer, src, dst_offsets, src_offsets, shape],
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

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Op-contract tests for the InCore async remote put family.

Covers ``pld.system.async_session`` / ``pld.tile.async_session``,
``pld.tensor.put_async`` / ``pld.tile.put_async`` and
``pld.system.wait_async_event`` / ``pld.tile.wait_async_event``.

The constraints under test are not stylistic — each mirrors a hard limit one
layer down:

* **flat-contiguous logical-1D, fully static** — PTOAS
  ``verifyAsyncFlatContiguous1DGMViewLike`` and pto-isa
  ``TPutAsyncIsFlatContiguous1D``.
* **no ``atomic`` kwarg** — ``pto.comm.tput_async`` has no atomicType operand
  at all (contrast ``pto.comm.tput``).
* **``sync_id`` <= 7** — pto-isa ``BuildSdmaSession`` returns false above that
  and PTOAS's op discards the bool, so an invalid session would reach the
  device silently.
"""

import pytest
from pypto import DataType, ir


def _span() -> ir.Span:
    return ir.Span.unknown()


def _dist_tensor(name: str, shape: list[int], dtype: DataType, span: ir.Span) -> ir.Var:
    return ir.Var(
        name,
        ir.DistributedTensorType([ir.ConstInt(v, DataType.INT64, span) for v in shape], dtype),
        span,
    )


def _tensor(name: str, shape: list[int], dtype: DataType, span: ir.Span) -> ir.Var:
    return ir.Var(
        name,
        ir.TensorType([ir.ConstInt(v, DataType.INT64, span) for v in shape], dtype),
        span,
    )


def _tile(name: str, shape: list[int], dtype: DataType, span: ir.Span) -> ir.Var:
    return ir.Var(
        name,
        ir.TileType([ir.ConstInt(v, DataType.INT64, span) for v in shape], dtype),
        span,
    )


def _tuple(values: list[int], span: ir.Span) -> ir.MakeTuple:
    return ir.MakeTuple([ir.ConstInt(v, DataType.INT64, span) for v in values], span)


def _session(span: ir.Span) -> ir.Call:
    return ir.create_op_call("pld.system.async_session", [], {}, span)


# ---------------------------------------------------------------------------
# Session build
# ---------------------------------------------------------------------------


def test_async_session_returns_session_handle():
    """The session build yields the singleton AsyncSessionType."""
    span = _span()
    call = _session(span)
    assert isinstance(call.type, ir.AsyncSessionType)


def test_async_session_takes_no_positional_args():
    span = _span()
    with pytest.raises(ValueError, match="no positional arguments"):
        ir.create_op_call("pld.system.async_session", [_tensor("x", [1, 8], DataType.FP32, span)], {}, span)


@pytest.mark.parametrize("sync_id", [0, 7])
def test_async_session_accepts_valid_sync_id(sync_id: int):
    span = _span()
    call = ir.create_op_call("pld.system.async_session", [], {"sync_id": sync_id}, span)
    assert isinstance(call.type, ir.AsyncSessionType)


def test_async_session_rejects_out_of_range_sync_id():
    """pto-isa BuildSdmaSession rejects syncId > 7 and PTOAS discards that bool."""
    span = _span()
    with pytest.raises(ValueError, match="sync_id"):
        ir.create_op_call("pld.system.async_session", [], {"sync_id": 8}, span)


def test_async_session_accepts_positive_block_bytes():
    """An explicit chunk size is a compile-time constant forwarded to PTOAS."""
    span = _span()
    call = ir.create_op_call("pld.system.async_session", [], {"block_bytes": 65536}, span)
    assert isinstance(call.type, ir.AsyncSessionType)


def test_async_session_rejects_non_positive_block_bytes():
    """Zero/negative would become an invalid SDMA session that PTOAS silently keeps."""
    span = _span()
    with pytest.raises(ValueError, match="block_bytes"):
        ir.create_op_call("pld.system.async_session", [], {"block_bytes": 0}, span)
    with pytest.raises(ValueError, match="block_bytes"):
        ir.create_op_call("pld.system.async_session", [], {"block_bytes": -1}, span)


def test_tile_async_session_requires_tile_scratch():
    span = _span()
    scratch = _tile("scratch", [1, 256], DataType.INT8, span)
    call = ir.create_op_call("pld.tile.async_session", [scratch], {}, span)
    assert isinstance(call.type, ir.AsyncSessionType)

    with pytest.raises(ValueError, match="scratch must be a TileType"):
        ir.create_op_call("pld.tile.async_session", [_tensor("t", [1, 256], DataType.INT8, span)], {}, span)


# ---------------------------------------------------------------------------
# put_async — happy path
# ---------------------------------------------------------------------------


def test_put_async_returns_event_handle():
    """A full-slice 1-D put_async yields the singleton AsyncEventType."""
    span = _span()
    dst = _dist_tensor("dst", [1, 1024], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _dist_tensor("src", [1, 1024], DataType.FP16, span)

    call = ir.create_op_call("pld.tensor.put_async", [dst, peer, src, _session(span)], {}, span)
    assert isinstance(call.type, ir.AsyncEventType)


def test_put_async_accepts_plain_tensor_source():
    """src need not be window-bound — TPUT_ASYNC only needs a readable local GM region."""
    span = _span()
    dst = _dist_tensor("dst", [1, 512], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _tensor("src", [1, 512], DataType.FP32, span)

    call = ir.create_op_call("pld.tensor.put_async", [dst, peer, src, _session(span)], {}, span)
    assert isinstance(call.type, ir.AsyncEventType)


def test_put_async_subregion_uses_explicit_1d_shape():
    """A 2-D window is fine when the moved region itself is logically 1-D."""
    span = _span()
    dst = _dist_tensor("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _dist_tensor("src", [16, 64], DataType.FP16, span)

    call = ir.create_op_call(
        "pld.tensor.put_async",
        [dst, peer, src, _session(span), _tuple([3, 0], span), _tuple([1, 0], span), _tuple([1, 64], span)],
        {},
        span,
    )
    assert isinstance(call.type, ir.AsyncEventType)


def test_put_async_subregion_rejects_mismatched_src_dst_rank():
    """Subregion validation indexes src_shape by dst rank; mismatched ranks must
    fail before that loop, not as an out-of-bounds read or a later internal error.
    """
    span = _span()
    dst = _dist_tensor("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _dist_tensor("src", [64], DataType.FP16, span)

    with pytest.raises(ValueError, match=r"dst rank \(2\) must match src rank \(1\)"):
        ir.create_op_call(
            "pld.tensor.put_async",
            [
                dst,
                peer,
                src,
                _session(span),
                _tuple([0, 0], span),
                _tuple([0], span),
                _tuple([1, 64], span),
            ],
            {},
            span,
        )
    """Unlike pld.tile.put, the tile-level async form has no buf(...) operand."""
    span = _span()
    dst = _dist_tensor("dst", [1, 256], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _dist_tensor("src", [1, 256], DataType.FP16, span)

    call = ir.create_op_call("pld.tile.put_async", [dst, peer, src, _session(span)], {}, span)
    assert isinstance(call.type, ir.AsyncEventType)


# ---------------------------------------------------------------------------
# put_async — the constraints that mirror PTOAS / pto-isa
# ---------------------------------------------------------------------------


def test_put_async_rejects_non_1d_region():
    """PTOAS's async_put_invalid_non_1d.pto lit test rejects exactly this shape."""
    span = _span()
    dst = _dist_tensor("dst", [16, 64], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _dist_tensor("src", [16, 64], DataType.FP16, span)

    with pytest.raises(ValueError, match="flat-contiguous logical-1D"):
        ir.create_op_call("pld.tensor.put_async", [dst, peer, src, _session(span)], {}, span)


def test_put_async_rejects_dynamic_shape():
    """TPUT_ASYNC has no chunking path, so a dynamic extent cannot be bounded."""
    span = _span()
    n = ir.Var("n", ir.ScalarType(DataType.INT64), span)
    dst = ir.Var(
        "dst", ir.DistributedTensorType([ir.ConstInt(1, DataType.INT64, span), n], DataType.FP16), span
    )
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = ir.Var(
        "src", ir.DistributedTensorType([ir.ConstInt(1, DataType.INT64, span), n], DataType.FP16), span
    )

    with pytest.raises(ValueError, match="fully static"):
        ir.create_op_call("pld.tensor.put_async", [dst, peer, src, _session(span)], {}, span)


def test_put_async_rejects_atomic_kwarg():
    """pto.comm.tput_async has no atomicType operand — the kwarg is inexpressible."""
    span = _span()
    dst = _dist_tensor("dst", [1, 128], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _dist_tensor("src", [1, 128], DataType.FP32, span)

    with pytest.raises(ValueError, match="atomic"):
        ir.create_op_call(
            "pld.tensor.put_async",
            [dst, peer, src, _session(span)],
            {"atomic": ir.AtomicType.Add},
            span,
        )


def test_put_async_requires_session_handle():
    span = _span()
    dst = _dist_tensor("dst", [1, 128], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _dist_tensor("src", [1, 128], DataType.FP32, span)
    not_a_session = ir.Var("nope", ir.ScalarType(DataType.INT32), span)

    with pytest.raises(ValueError, match="AsyncSession"):
        ir.create_op_call("pld.tensor.put_async", [dst, peer, src, not_a_session], {}, span)


def test_put_async_requires_window_bound_dst():
    span = _span()
    dst = _tensor("dst", [1, 128], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _tensor("src", [1, 128], DataType.FP32, span)

    with pytest.raises(ValueError, match="window-bound DistributedTensor"):
        ir.create_op_call("pld.tensor.put_async", [dst, peer, src, _session(span)], {}, span)


def test_put_async_rejects_dtype_mismatch():
    span = _span()
    dst = _dist_tensor("dst", [1, 128], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _dist_tensor("src", [1, 128], DataType.FP16, span)

    with pytest.raises(ValueError, match="element type"):
        ir.create_op_call("pld.tensor.put_async", [dst, peer, src, _session(span)], {}, span)


@pytest.mark.parametrize("nargs", [3, 5, 6, 8])
def test_put_async_rejects_bad_arity(nargs: int):
    """Only the 4-arg full-slice and 7-arg subregion forms are legal."""
    span = _span()
    dst = _dist_tensor("dst", [1, 128], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _dist_tensor("src", [1, 128], DataType.FP32, span)
    # Eight entries so nargs=8 is a genuine over-long call rather than a slice
    # that silently truncates back to the valid 7-arg form.
    args = [
        dst,
        peer,
        src,
        _session(span),
        _tuple([0, 0], span),
        _tuple([0, 0], span),
        _tuple([1, 128], span),
        _tuple([1, 128], span),
    ]

    with pytest.raises(ValueError, match="positional argument"):
        ir.create_op_call("pld.tensor.put_async", args[:nargs], {}, span)


# ---------------------------------------------------------------------------
# wait_async_event
# ---------------------------------------------------------------------------


def test_wait_async_event_returns_bool():
    span = _span()
    dst = _dist_tensor("dst", [1, 256], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _dist_tensor("src", [1, 256], DataType.FP16, span)
    sess = _session(span)
    evt = ir.create_op_call("pld.tensor.put_async", [dst, peer, src, sess], {}, span)

    done = ir.create_op_call("pld.system.wait_async_event", [evt, sess], {}, span)
    assert isinstance(done.type, ir.ScalarType)
    assert done.type.dtype == DataType.BOOL


def test_wait_async_event_rejects_non_event():
    span = _span()
    sess = _session(span)
    not_an_event = ir.Var("nope", ir.ScalarType(DataType.INT32), span)

    with pytest.raises(ValueError, match="AsyncEvent"):
        ir.create_op_call("pld.system.wait_async_event", [not_an_event, sess], {}, span)


def test_tile_wait_async_event_pins_the_scratch():
    """The tile-level wait carries the scratch so its live range spans the window.

    pto-isa reads the completion word back through ``session.tmpBufAddr``, which
    points into that Vec(UB) buffer, so it must outlive every wait — not just the
    session build, which is its only other use.
    """
    span = _span()
    scratch = _tile("scratch", [1, 256], DataType.INT8, span)
    sess = ir.create_op_call("pld.tile.async_session", [scratch], {}, span)
    dst = _dist_tensor("dst", [1, 256], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _dist_tensor("src", [1, 256], DataType.FP16, span)
    evt = ir.create_op_call("pld.tile.put_async", [dst, peer, src, sess], {}, span)

    done = ir.create_op_call("pld.tile.wait_async_event", [evt, sess, scratch], {}, span)
    assert isinstance(done.type, ir.ScalarType)
    assert done.type.dtype == DataType.BOOL


def test_tile_wait_async_event_requires_scratch():
    span = _span()
    scratch = _tile("scratch", [1, 256], DataType.INT8, span)
    sess = ir.create_op_call("pld.tile.async_session", [scratch], {}, span)
    dst = _dist_tensor("dst", [1, 256], DataType.FP16, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _dist_tensor("src", [1, 256], DataType.FP16, span)
    evt = ir.create_op_call("pld.tile.put_async", [dst, peer, src, sess], {}, span)

    with pytest.raises(ValueError, match="positional argument"):
        ir.create_op_call("pld.tile.wait_async_event", [evt, sess], {}, span)


# ---------------------------------------------------------------------------
# Handle types are the same singletons prefetch already uses
# ---------------------------------------------------------------------------


def test_handles_reuse_the_prefetch_singletons():
    """No new ObjectKind: the async put family rides prefetch's singleton types.

    This is what lets the wait emitter be shared with ``prefetch.wait``, which
    already lowers to ``pto.comm.wait_async_event``.
    """
    span = _span()
    sess = _session(span)
    assert ir.structural_equal(sess.type, ir.AsyncSessionType.get())

    dst = _dist_tensor("dst", [1, 64], DataType.FP32, span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    src = _dist_tensor("src", [1, 64], DataType.FP32, span)
    evt = ir.create_op_call("pld.tensor.put_async", [dst, peer, src, sess], {}, span)
    assert ir.structural_equal(evt.type, ir.AsyncEventType.get())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``tensor.as_layout`` op (RFC #1300, P4-a).

The op is a pure metadata reinterpret — it points at the same physical memory
as its source but exposes a different ``(shape, stride, layout)`` triple.
This file covers ``DeduceTensorAsLayoutType`` (type inference + validity
check); the Simplify-pass folding rules are covered in
``tests/ut/ir/transforms/test_simplify_pass.py`` (P4-b).
"""

import pytest
from pypto import DataType, ir


def _span():
    return ir.Span.unknown()


def _const(value: int, dtype: DataType = DataType.INDEX):
    return ir.ConstInt(value, dtype, _span())


def _tensor_var(shape, dtype=DataType.FP32, view=None, name="t"):
    span = _span()
    shape_exprs = [_const(d) for d in shape] if isinstance(shape[0], int) else shape
    if view is None:
        t = ir.TensorType(shape_exprs, dtype)
    else:
        t = ir.TensorType(shape_exprs, dtype, None, view)
    return ir.Var(name, t, span)


def _result_view(call):
    """Return the TensorView (or None) on the Call's result type."""
    t = call.type
    assert isinstance(t, ir.TensorType)
    return t.tensor_view


def _values_of(exprs):
    out = []
    for e in exprs:
        assert isinstance(e, ir.ConstInt)
        out.append(e.value)
    return out


# ============================================================================
# Happy path — RFC §4.2 canonical pairs
# ============================================================================


def test_nd_to_dn_2d_static():
    """Bare ``[N=8, K=4] ND`` reinterpreted as ``[K=4, N=8] DN`` is valid:
    both describe the same row-major buffer (RFC §4.2).
    """
    src = _tensor_var([8, 4])  # bare = ND-packed
    call = ir.op.tensor.as_layout(src, [4, 8], ir.TensorLayout.DN)

    assert call.op.name == "tensor.as_layout"
    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.DN
    # DN-packed for [4, 8]: stride=[1, 4]
    assert _values_of(view.stride) == [1, 4]
    # Output shape matches the requested target.
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert _values_of(result_type.shape) == [4, 8]


def test_dn_to_nd_2d_static():
    """Round trip: ``[K=4, N=8] DN-packed`` reinterpreted back to ``[N=8, K=4] ND``."""
    src_view = ir.TensorView([_const(1), _const(4)], ir.TensorLayout.DN)
    src = _tensor_var([4, 8], view=src_view)
    call = ir.op.tensor.as_layout(src, [8, 4], ir.TensorLayout.ND)

    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.ND
    # ND-packed for [8, 4]: stride=[4, 1]
    assert _values_of(view.stride) == [4, 1]
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert _values_of(result_type.shape) == [8, 4]


def test_3d_dn_to_nd_static():
    """3D analogue: ``[B=2, K=4, N=8] DN`` → ``[2, 8, 4] ND`` over the same buffer."""
    src_view = ir.TensorView([_const(32), _const(1), _const(4)], ir.TensorLayout.DN)
    src = _tensor_var([2, 4, 8], view=src_view)
    call = ir.op.tensor.as_layout(src, [2, 8, 4], ir.TensorLayout.ND)

    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.ND
    # ND-packed for [2, 8, 4]: stride=[32, 4, 1]
    assert _values_of(view.stride) == [32, 4, 1]


def test_idempotent_self_reinterpret():
    """``as_layout(t, t.shape, t.layout)`` produces the canonical view for the
    same shape/layout — equivalent to the input modulo packed-canonical stride
    materialization.
    """
    src = _tensor_var([8, 4])  # bare ND
    call = ir.op.tensor.as_layout(src, [8, 4], ir.TensorLayout.ND)

    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.ND
    assert _values_of(view.stride) == [4, 1]
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert _values_of(result_type.shape) == [8, 4]


# ============================================================================
# Validity checks — every CHECK must surface as ``ValueError``
# ============================================================================


def test_element_count_mismatch_rejected():
    """Total element count must match — `[8,4]` (32) ≠ `[8,5]` (40)."""
    src = _tensor_var([8, 4])
    with pytest.raises(ValueError, match="element count mismatch"):
        ir.op.tensor.as_layout(src, [8, 5], ir.TensorLayout.ND)


def test_nz_target_rejected():
    """NZ on TensorType is forbidden (NZ is tile-only / fractal)."""
    src = _tensor_var([8, 4])
    with pytest.raises(ValueError, match="NZ layout is not allowed"):
        ir.op.tensor.as_layout(src, [8, 4], ir.TensorLayout.NZ)


def test_invalid_offset_map_rejected():
    """Reinterprets that don't fall in a §4.2 canonical pair are rejected.

    Here the same shape ``[8, 4]`` ND → DN at the same shape would put DN's
    trailing dim at 8 (≠ row-major-equivalent ``[4, 8]``), so the offset map
    differs from the source.
    """
    src = _tensor_var([8, 4])  # row-major-equivalent shape = [8, 4]
    with pytest.raises(ValueError, match="canonical pairs"):
        ir.op.tensor.as_layout(src, [8, 4], ir.TensorLayout.DN)


# ============================================================================
# Symbolic shapes — accepted as long as ExprPtr identity matches
# ============================================================================


def test_symbolic_shape_passes():
    """Symbolic ``[N, K] ND`` → ``[K, N] DN`` over the same buffer.

    The helper compares ExprPtr identity; both sides reference the same
    Var nodes after the expected swap, so the offset-map check passes.
    """
    span = _span()
    n_var = ir.Var("N", ir.ScalarType(DataType.INDEX), span)
    k_var = ir.Var("K", ir.ScalarType(DataType.INDEX), span)
    src = _tensor_var([n_var, k_var])
    call = ir.op.tensor.as_layout(src, [k_var, n_var], ir.TensorLayout.DN)

    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.DN


# ============================================================================
# Op-registry sanity
# ============================================================================


def test_op_registered():
    """``tensor.as_layout`` must be discoverable through the OpRegistry."""
    assert ir.is_op_registered("tensor.as_layout")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

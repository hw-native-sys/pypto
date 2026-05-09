# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``tensor.as_layout`` op (RFC #1300, P4).

The op flips a TensorType's layout tag over the same physical memory; it
never reshapes (use ``tensor.reshape`` for that). Target shape is mechanically
derived from the source — callers do not pass a target shape.

This file covers ``DeduceTensorAsLayoutType`` (type inference + validity); the
Simplify-pass identity-elimination rule is covered in
``tests/ut/ir/transforms/test_simplify_pass.py``.
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
# Cross-layout flips — target shape is auto-derived (trailing-2-dim swap)
# ============================================================================


def test_bare_nd_to_dn_flips_trailing_dims():
    """Bare ``[N=8, K=4]`` (implicit ND) → DN auto-swaps to ``[K=4, N=8]``
    DN-packed, the §4.2 canonical pair partner."""
    src = _tensor_var([8, 4])
    call = ir.op.tensor.as_layout(src, ir.TensorLayout.DN)

    assert call.op.name == "tensor.as_layout"
    out = call.type
    assert isinstance(out, ir.TensorType)
    assert _values_of(out.shape) == [4, 8]
    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.DN
    # DN-packed for [4, 8]: stride=[1, 4]
    assert _values_of(view.stride) == [1, 4]


def test_dn_packed_to_nd_flips_back():
    """``[K=4, N=8] DN-packed`` → ND auto-swaps back to ``[N=8, K=4] ND``."""
    src_view = ir.TensorView([_const(1), _const(4)], ir.TensorLayout.DN)
    src = _tensor_var([4, 8], view=src_view)
    call = ir.op.tensor.as_layout(src, ir.TensorLayout.ND)

    out = call.type
    assert isinstance(out, ir.TensorType)
    assert _values_of(out.shape) == [8, 4]
    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.ND
    # ND-packed for [8, 4]: stride=[4, 1]
    assert _values_of(view.stride) == [4, 1]


def test_3d_nd_to_dn_swaps_trailing_pair_only():
    """Outer batch dim is preserved; only the trailing 2 dims swap."""
    src = _tensor_var([2, 4, 8])  # bare ND
    call = ir.op.tensor.as_layout(src, ir.TensorLayout.DN)

    out = call.type
    assert isinstance(out, ir.TensorType)
    # [2, 4, 8] ND → [2, 8, 4] DN (trailing pair swap)
    assert _values_of(out.shape) == [2, 8, 4]
    view = _result_view(call)
    assert view is not None
    # DN-packed for [2, 8, 4]: stride=[8*4, 1, 8] = [32, 1, 8]
    assert _values_of(view.stride) == [32, 1, 8]


# ============================================================================
# Identity flips — same layout, shape unchanged (Simplify will fold the call)
# ============================================================================


def test_identity_flip_keeps_shape():
    """``as_layout(x, x.layout)`` produces an identity Call: same shape,
    same layout (modulo packed-canonical stride materialization). The Call
    survives type inference; the Simplify pass folds it away."""
    src = _tensor_var([8, 4])  # bare ND
    call = ir.op.tensor.as_layout(src, ir.TensorLayout.ND)

    out = call.type
    assert isinstance(out, ir.TensorType)
    assert _values_of(out.shape) == [8, 4]
    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.ND
    assert _values_of(view.stride) == [4, 1]


# ============================================================================
# Validity rejections
# ============================================================================


def test_nz_target_rejected():
    """NZ on TensorType is forbidden (NZ is tile-only / fractal)."""
    src = _tensor_var([8, 4])
    with pytest.raises(ValueError, match="NZ layout is not allowed"):
        ir.op.tensor.as_layout(src, ir.TensorLayout.NZ)


def test_cross_layout_flip_below_rank_2_rejected():
    """ND ↔ DN flip needs at least 2 dims to swap; 1D is rejected."""
    src = _tensor_var([8])
    with pytest.raises(ValueError, match="rank >= 2"):
        ir.op.tensor.as_layout(src, ir.TensorLayout.DN)


def test_strided_source_rejected():
    """Strided sub-views can't ride the canonical pair; reject them so the
    caller routes through ``tensor.slice`` / ``tensor.reshape`` first."""
    # Synthesize a strided ND tensor: stride [16, 1] on shape [4, 8] (parent
    # stride preserved on a 4×8 sub-view of an 8×16 row-major buffer).
    src_view = ir.TensorView([_const(16), _const(1)], ir.TensorLayout.ND)
    src = _tensor_var([4, 8], view=src_view, name="strided")
    with pytest.raises(ValueError, match="strided sub-view"):
        ir.op.tensor.as_layout(src, ir.TensorLayout.DN)


# ============================================================================
# Symbolic shapes — accepted on the cross-layout flip; shape swap survives
# ============================================================================


def test_symbolic_shape_flips():
    """Symbolic ``[N, K] ND`` → DN swaps to ``[K, N] DN``; ExprPtr identity
    is preserved through the swap."""
    span = _span()
    n_var = ir.Var("N", ir.ScalarType(DataType.INDEX), span)
    k_var = ir.Var("K", ir.ScalarType(DataType.INDEX), span)
    src = _tensor_var([n_var, k_var])
    call = ir.op.tensor.as_layout(src, ir.TensorLayout.DN)

    out = call.type
    assert isinstance(out, ir.TensorType)
    # Trailing pair swap: [N, K] -> [K, N]; ExprPtrs preserved.
    assert out.shape[0] is k_var
    assert out.shape[1] is n_var
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

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``tensor.view`` op (RFC #1300, P4).

The op reinterprets a TensorType over the same physical memory: it can flip
the layout tag, reinterpret with a new product-preserving shape, or both.
``tensor.reshape`` is unaffected; this op covers canonical layout/shape
reinterpretation, not arbitrary strides.

This file covers ``DeduceTensorViewType`` (type inference + validity); the
Simplify-pass identity-elimination rule is covered in
``tests/ut/ir/transforms/test_simplify_pass.py``.
"""

import pytest
from pypto import DataType, ir
from pypto.language.distributed import DistributedTensor
from pypto.language.op import tensor_ops as dsl_tensor_ops


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
    assert isinstance(t, (ir.TensorType, ir.DistributedTensorType))
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
    call = ir.op.tensor.view(src, layout=ir.TensorLayout.DN)

    assert call.op.name == ir.get_op("tensor.view").name
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
    call = ir.op.tensor.view(src, layout=ir.TensorLayout.ND)

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
    call = ir.op.tensor.view(src, layout=ir.TensorLayout.DN)

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
    """``view(x, x.layout)`` produces an identity Call: same shape,
    same layout (modulo packed-canonical stride materialization). The Call
    survives type inference; the Simplify pass folds it away."""
    src = _tensor_var([8, 4])  # bare ND
    call = ir.op.tensor.view(src, layout=ir.TensorLayout.ND)

    out = call.type
    assert isinstance(out, ir.TensorType)
    assert _values_of(out.shape) == [8, 4]
    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.ND
    assert _values_of(view.stride) == [4, 1]


def test_shape_only_reinterprets_with_canonical_stride():
    """``view(x, shape=[4, 8])`` changes shape and derives packed ND stride."""
    src = _tensor_var([2, 16])
    call = ir.op.tensor.view(src, [4, 8])

    out = call.type
    assert isinstance(out, ir.TensorType)
    assert _values_of(out.shape) == [4, 8]
    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.ND
    assert _values_of(view.stride) == [8, 1]


def test_shape_and_layout_reinterprets_with_canonical_layout_stride():
    """``shape`` and ``layout`` can be combined; stride follows the target layout."""
    src = _tensor_var([2, 16])
    call = ir.op.tensor.view(src, [4, 8], layout=ir.TensorLayout.DN)

    out = call.type
    assert isinstance(out, ir.TensorType)
    assert _values_of(out.shape) == [4, 8]
    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.DN
    assert _values_of(view.stride) == [1, 4]


def test_shape_reinterpret_rejects_strided_source():
    """Shape-changing views are canonical-only and cannot discard parent stride."""
    src_view = ir.TensorView([_const(16), _const(1)], ir.TensorLayout.ND)
    src = _tensor_var([4, 8], view=src_view, name="strided")
    with pytest.raises(ValueError, match="packed source"):
        ir.op.tensor.view(src, [32])


@pytest.mark.parametrize("shape", ([0, 32], [-1, 32]))
def test_shape_reinterpret_rejects_non_positive_static_dimension(shape):
    """Static view dimensions must be positive."""
    src = _tensor_var([4, 8])
    with pytest.raises(ValueError, match="must be positive"):
        ir.op.tensor.view(src, shape)


def test_shape_reinterpret_rejects_zero_sized_source_product_mismatch():
    """A known zero-sized source must still preserve the element count."""
    src = _tensor_var([0, 8])
    with pytest.raises(ValueError, match="cannot reinterpret"):
        ir.op.tensor.view(src, [1])


def test_distributed_tensor_kind_is_preserved():
    """A distributed view preserves both its type kind and window binding."""
    span = _span()
    base = ir.Var("buffer", ir.PtrType(), span)
    window = ir.WindowBuffer(base, _const(32), span=span)
    src_type = ir.DistributedTensorType([_const(2), _const(16)], DataType.FP32, window)
    src = ir.Var("dt", src_type, span)
    call = ir.op.tensor.view(src, [32])

    out = call.type
    assert isinstance(out, ir.DistributedTensorType)
    assert out.window_buffer is window
    assert _values_of(out.shape) == [32]
    view = _result_view(call)
    assert view is not None
    assert _values_of(view.stride) == [1]


def test_dsl_view_preserves_distributed_tensor_wrapper():
    """The DSL wrapper retains the concrete DistributedTensor class."""
    span = _span()
    src = ir.Var("dt", ir.DistributedTensorType([_const(2), _const(16)], DataType.FP32), span)

    viewed = dsl_tensor_ops.view(DistributedTensor(expr=src), [32])

    assert isinstance(viewed, DistributedTensor)
    assert isinstance(viewed.unwrap().type, ir.DistributedTensorType)


# ============================================================================
# Validity rejections
# ============================================================================


def test_nz_target_rejected():
    """NZ on TensorType is forbidden (NZ is tile-only / fractal)."""
    src = _tensor_var([8, 4])
    with pytest.raises(ValueError, match="NZ layout is not allowed"):
        ir.op.tensor.view(src, layout=ir.TensorLayout.NZ)


def test_cross_layout_flip_below_rank_2_rejected():
    """ND ↔ DN flip needs at least 2 dims to swap; 1D is rejected."""
    src = _tensor_var([8])
    with pytest.raises(ValueError, match="rank >= 2"):
        ir.op.tensor.view(src, layout=ir.TensorLayout.DN)


def test_shape_reinterpret_rejects_rank_zero_view():
    """Rank-zero tensor views are not representable by either codegen backend."""
    src = _tensor_var([1])
    with pytest.raises(ValueError, match="target shape must have rank >= 1"):
        ir.op.tensor.view(src, [])


def test_unknown_view_kwarg_is_rejected():
    """The op schema must reject misspelled metadata instead of ignoring it."""
    src = _tensor_var([1])
    shape = ir.MakeTuple([_const(1)], _span())
    with pytest.raises(ValueError, match="Unknown kwarg 'layuot'"):
        ir.create_op_call("tensor.view", [src, shape], {"layuot": 0}, _span())


def test_strided_source_flips_inheriting_stride():
    """Strided sub-views ride the §4.2 canonical pair too: a strided-ND view
    ``(shape=[4, 8], stride=[16, 1])`` (4×8 sub-view of an 8×16 row-major
    buffer) flips to a strided-DN view ``(shape=[8, 4], stride=[1, 16])``
    over the same physical memory. The parent's row stride 16 is preserved
    through the flip so downstream codegen reads the correct addresses (the
    bug class behind #1212 / #1213)."""
    src_view = ir.TensorView([_const(16), _const(1)], ir.TensorLayout.ND)
    src = _tensor_var([4, 8], view=src_view, name="strided")
    call = ir.op.tensor.view(src, layout=ir.TensorLayout.DN)

    out = call.type
    assert isinstance(out, ir.TensorType)
    # Trailing-pair shape swap.
    assert _values_of(out.shape) == [8, 4]
    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.DN
    # Trailing-pair stride swap — parent row stride (16) now sits at the
    # outer slot, the inner DN stride is 1.
    assert _values_of(view.stride) == [1, 16]


def test_non_canonical_source_rejected():
    """Source views that fail the canonical-family invariants (e.g. ND with
    non-unit innermost stride) are still rejected — they don't describe a
    well-defined memory pattern."""
    # ND requires stride[-1] == 1; a stride of [16, 2] violates that.
    src_view = ir.TensorView([_const(16), _const(2)], ir.TensorLayout.ND)
    src = _tensor_var([4, 8], view=src_view, name="malformed")
    with pytest.raises(ValueError, match="not canonical"):
        ir.op.tensor.view(src, layout=ir.TensorLayout.DN)


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
    call = ir.op.tensor.view(src, layout=ir.TensorLayout.DN)

    out = call.type
    assert isinstance(out, ir.TensorType)
    # Trailing pair swap: [N, K] -> [K, N]; ExprPtrs preserved.
    assert out.shape[0] is k_var
    assert out.shape[1] is n_var
    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.DN


def test_symbolic_shape_reinterpret_accepted():
    """Symbolic dimensions where product can't be proven are accepted.

    When ``src == [N, M]`` and ``view(src, [N, M, 1])``, the shape products
    are unprovable (one or both <= 0) so the product-equality check is skipped
    and the view is accepted."""
    span = _span()
    n_var = ir.Var("N", ir.ScalarType(DataType.INDEX), span)
    m_var = ir.Var("M", ir.ScalarType(DataType.INDEX), span)
    src = _tensor_var([n_var, m_var])
    # Symbolic shape reinterpret: [N, M] -> [N, M, 1].  Product can't be
    # proven so the check at transform.cpp:361 bails early.
    call = ir.op.tensor.view(src, [n_var, m_var, _const(1)])

    out = call.type
    assert isinstance(out, ir.TensorType)
    assert len(out.shape) == 3
    assert out.shape[0] is n_var
    assert out.shape[1] is m_var
    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.ND


def test_layout_view_preserves_valid_shape():
    """Layout-only view (no shape arg) preserves ``valid_shape`` metadata,
    swapping the trailing pair for cross-layout flips."""
    src_view = ir.TensorView(
        [_const(8), _const(1)],
        ir.TensorLayout.ND,
        valid_shape=[_const(6), _const(4)],
        pad=ir.PadValue.zero,
    )
    src = _tensor_var([8, 4], view=src_view)
    # DN flip: valid_shape trailing pair swaps too.
    call = ir.op.tensor.view(src, layout=ir.TensorLayout.DN)

    out = call.type
    assert isinstance(out, ir.TensorType)
    view = _result_view(call)
    assert view is not None
    # valid_shape is preserved and trailing-pair swapped.
    assert _values_of(view.valid_shape) == [4, 6]
    # pad is unconditionally preserved.
    assert view.pad == ir.PadValue.zero


def test_shape_reinterpret_rejects_partial_valid_shape():
    """A rectangular partial-valid region cannot be safely reshaped."""
    src_view = ir.TensorView(
        [_const(8), _const(1)],
        ir.TensorLayout.ND,
        valid_shape=[_const(6), _const(4)],
        pad=ir.PadValue.zero,
    )
    src = _tensor_var([4, 8], view=src_view)
    with pytest.raises(ValueError, match="partial valid_shape"):
        ir.op.tensor.view(src, [32])


def test_shape_reinterpret_clears_inert_full_valid_padding():
    """A full valid region may be reshaped; its padding metadata is inert."""
    src_view = ir.TensorView(
        [_const(8), _const(1)],
        ir.TensorLayout.ND,
        valid_shape=[_const(4), _const(8)],
        pad=ir.PadValue.zero,
    )
    src = _tensor_var([4, 8], view=src_view)

    view = _result_view(ir.op.tensor.view(src, [32]))

    assert view is not None
    assert len(view.valid_shape) == 0
    assert view.pad == ir.PadValue.null


def test_shape_reinterpret_clears_pad_without_valid_shape():
    """Padding metadata is not carried across a shape reinterpret."""
    src_view = ir.TensorView(
        [_const(8), _const(1)],
        ir.TensorLayout.ND,
        pad=ir.PadValue.zero,
    )
    src = _tensor_var([4, 8], view=src_view)

    view = _result_view(ir.op.tensor.view(src, [32]))

    assert view is not None
    assert len(view.valid_shape) == 0
    assert view.pad == ir.PadValue.null


def test_layout_view_preserves_pad_on_bare_tensor():
    """pad is preserved even when valid_shape is empty (bare tensor)."""
    src_view = ir.TensorView(
        [_const(8), _const(1)],
        ir.TensorLayout.ND,
        pad=ir.PadValue.zero,
    )
    src = _tensor_var([4, 8], view=src_view)
    call = ir.op.tensor.view(src, layout=ir.TensorLayout.DN)

    out = call.type
    assert isinstance(out, ir.TensorType)
    view = _result_view(call)
    assert view is not None
    assert view.pad == ir.PadValue.zero


# ============================================================================
# Op-registry sanity
# ============================================================================


def test_op_registered():
    """``tensor.view`` must be discoverable through the OpRegistry."""
    assert ir.is_op_registered("tensor.view")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

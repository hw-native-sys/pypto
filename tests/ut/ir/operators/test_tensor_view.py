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
``tensor.reshape`` is unaffected. Explicit strides additionally describe
non-overlapping padded ND views within packed source storage.

This file covers ``DeduceTensorViewType`` (type inference + validity); the
Simplify-pass identity-elimination rule is covered in
``tests/ut/ir/transforms/test_simplify_pass.py``.
"""

import inspect

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.language.distributed import DistributedTensor
from pypto.language.op import tensor_ops as dsl_tensor_ops
from pypto.language.typing import Scalar, Tensor


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


@pytest.mark.parametrize("dtype", [DataType.FP4E2M1X2, DataType.FP4E1M2X2])
def test_packed_fp4_layout_view_preserves_byte_carriers(dtype):
    src = _tensor_var([8, 16], dtype=dtype)
    call = ir.op.tensor.view(src, layout=ir.TensorLayout.DN)
    assert isinstance(call.type, ir.TensorType)
    assert call.type.dtype == dtype
    assert _values_of(call.type.shape) == [16, 8]
    view = _result_view(call)
    assert view is not None
    assert view.layout == ir.TensorLayout.DN
    assert _values_of(view.stride) == [1, 16]


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
    """tensor.view cannot mint an NZ layout.

    NZ is legal on a TensorType, but only in the blocked rank-5 form that
    BlockNzTensorViews derives from the source tensor — it is not a layout a
    view site gets to choose.
    """
    src = _tensor_var([8, 4])
    with pytest.raises(ValueError, match="cannot produce an NZ layout"):
        ir.op.tensor.view(src, layout=ir.TensorLayout.NZ)


@pytest.mark.parametrize(
    ("source_layout", "source_shape", "target_layout", "target_shape"),
    [
        pytest.param(ir.TensorLayout.MX_A_ZZ, [32, 4], ir.TensorLayout.ND, [1, 128], id="a-to-nd"),
        pytest.param(ir.TensorLayout.ND, [1, 128], ir.TensorLayout.MX_A_ZZ, [32, 4], id="nd-to-a"),
        pytest.param(ir.TensorLayout.MX_B_NN, [4, 32], ir.TensorLayout.ND, [1, 128], id="b-to-nd"),
        pytest.param(ir.TensorLayout.ND, [1, 128], ir.TensorLayout.MX_B_NN, [4, 32], id="nd-to-b"),
    ],
)
def test_shaped_nd_mx_backing_view_is_allowed(source_layout, source_shape, target_layout, target_shape):
    """FP8E8M0 storage may alias packed ND and either complete MX scale layout."""
    source_view = None if source_layout == ir.TensorLayout.ND else ir.TensorView([], source_layout)
    src = _tensor_var(source_shape, dtype=DataType.FP8E8M0, view=source_view)

    call = ir.op.tensor.view(src, target_shape, layout=target_layout)

    out = call.type
    assert isinstance(out, ir.TensorType)
    assert _values_of(out.shape) == target_shape
    view = _result_view(call)
    assert view is not None
    assert view.layout == target_layout


def test_mx_backing_view_rejects_scalar_access():
    """Scalar addressing cannot flatten packed MX scale bytes as ordinary ND."""
    src = _tensor_var([1, 128], dtype=DataType.FP8E8M0)
    mx = ir.op.tensor.view(src, [32, 4], layout=ir.TensorLayout.MX_A_ZZ)
    value = ir.Var("value", ir.ScalarType(DataType.FP8E8M0), _span())

    with pytest.raises(ValueError, match="tensor.read does not support MX-layout tensors"):
        ir.op.tensor.read(mx, [0, 2])
    with pytest.raises(ValueError, match="tensor.write does not support MX-layout tensors"):
        ir.op.tensor.write(mx, [0, 2], value)


def test_mx_a_backing_view_rejects_non_scale_dtype():
    """The MX_A_ZZ backing alias is limited to E8M0 scale storage."""
    src = _tensor_var([1, 128], dtype=DataType.FP32)

    with pytest.raises(ValueError, match="does not support MX layouts except shaped ND/MX_A_ZZ/MX_B_NN"):
        ir.op.tensor.view(src, [32, 4], layout=ir.TensorLayout.MX_A_ZZ)


def test_mx_a_backing_view_requires_explicit_shape():
    """An MX_A_ZZ layout flip without a backing shape remains unsupported."""
    src = _tensor_var([1, 128], dtype=DataType.FP8E8M0)

    with pytest.raises(ValueError, match="does not support MX layouts except shaped ND/MX_A_ZZ/MX_B_NN"):
        ir.op.tensor.view(src, layout=ir.TensorLayout.MX_A_ZZ)


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


def test_shape_reinterpret_accepts_explicit_target_valid_shape():
    """An explicit target rectangle makes a compiler-owned partial reshape unambiguous."""
    src_view = ir.TensorView(
        [_const(2048), _const(128), _const(1)],
        ir.TensorLayout.ND,
        valid_shape=[_const(1), _const(16), _const(128)],
        pad=ir.PadValue.zero,
    )
    src = _tensor_var([2, 16, 128], view=src_view)

    call = ir.op.tensor.view(src, [32, 128], [16, 128])
    view = _result_view(call)

    assert view is not None
    assert _values_of(view.valid_shape) == [16, 128]
    assert view.pad == ir.PadValue.zero


def test_shape_reinterpret_rejects_invalid_explicit_valid_shape():
    src = _tensor_var([4, 8])
    with pytest.raises(ValueError, match="valid_shape rank"):
        ir.op.tensor.view(src, [2, 16], [16])
    with pytest.raises(ValueError, match="exceeds target shape"):
        ir.op.tensor.view(src, [2, 16], [3, 16])

    partial_view = ir.TensorView(
        [_const(8), _const(1)], ir.TensorLayout.ND, valid_shape=[_const(3), _const(8)]
    )
    partial_src = _tensor_var([4, 8], view=partial_view)
    with pytest.raises(ValueError, match="leading-dimension collapse"):
        ir.op.tensor.view(partial_src, [2, 16], [1, 16])
    with pytest.raises(ValueError, match="leading-dimension collapse"):
        ir.op.tensor.view(partial_src, [2, 16], [2, 12])

    dn_view = ir.TensorView(
        [_const(12), _const(1), _const(3)],
        ir.TensorLayout.DN,
        valid_shape=[_const(1), _const(3), _const(2)],
    )
    dn_src = _tensor_var([2, 3, 4], view=dn_view)
    with pytest.raises(ValueError, match="only supports ND layout"):
        ir.op.tensor.view(dn_src, [6, 4], [3, 2])


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
# Explicit strides — padded ND aliases within packed source storage
# ============================================================================


@pytest.mark.parametrize("page_bytes", [4160, 4224, 8192])
@pytest.mark.parametrize("dtype", [DataType.INT8, DataType.FP16])
def test_explicit_strides_preserve_page_gaps(page_bytes, dtype):
    """Logical Key elements exclude each page's metadata without a copy."""
    src = _tensor_var([9, page_bytes], dtype=dtype)
    call = ir.op.tensor.view(src, [9, 32, 128], strides=[page_bytes, 128, 1])

    assert len(call.args) == 4
    assert isinstance(call.args[2], ir.MakeTuple)
    assert list(call.args[2].elements) == []
    result = call.type
    assert isinstance(result, ir.TensorType)
    assert result.dtype == dtype
    assert _values_of(result.shape) == [9, 32, 128]
    view = _result_view(call)
    assert view is not None
    assert _values_of(view.stride) == [page_bytes, 128, 1]


@pytest.mark.parametrize("valid_shape", [None, [], [2, 4]])
def test_explicit_strides_accept_full_target_validity(valid_shape):
    source_view = ir.TensorView([8, 1], ir.TensorLayout.ND, [2, 8], ir.PadValue.zero)
    src = _tensor_var([2, 8], view=source_view)
    call = ir.op.tensor.view(src, [2, 4], valid_shape, strides=[8, 1])

    view = _result_view(call)
    assert view is not None
    # TensorType normalizes an explicit full valid_shape to its empty spelling.
    assert _values_of(view.valid_shape) == []
    assert view.pad == ir.PadValue.null


def test_explicit_strides_preserve_memref_and_offset():
    """Capacity belongs to the source region even when its allocation is larger."""
    memref = ir.MemRef("storage", 64, 4096, _span())
    src = ir.Var("src", ir.TensorType([2, 8], DataType.INT8, memref), _span())
    call = ir.op.tensor.view(src, [2, 4], strides=[8, 1])

    result = call.type
    assert isinstance(result, ir.TensorType)
    result_memref = result.memref
    assert result_memref is not None
    assert result_memref is memref
    assert isinstance(result_memref.byte_offset_, ir.ConstInt)
    assert result_memref.byte_offset_.value == 64
    with pytest.raises(ValueError, match="exceeds source capacity 16"):
        ir.op.tensor.view(src, [2, 4], strides=[16, 1])


def test_explicit_strides_accept_exact_storage_end():
    src = _tensor_var([12])
    call = ir.op.tensor.view(src, [2, 4], strides=[8, 1])
    view = _result_view(call)
    assert view is not None
    assert _values_of(view.stride) == [8, 1]


def test_explicit_strides_symbolic_pages_and_pitch():
    """Dynamic extent/stride expressions remain operands, with caller bounds."""
    pages = ir.Var("pages", ir.ScalarType(DataType.INDEX), _span())
    pitch = ir.Var("pitch", ir.ScalarType(DataType.INT32), _span())
    source_stride = ir.TensorView([pitch, _const(1)], ir.TensorLayout.ND)
    src = _tensor_var([pages, pitch], DataType.INT8, source_stride)
    call = ir.op.tensor.view(src, [pages, 32, 128], strides=[pitch, 128, 1])

    result = call.type
    assert isinstance(result, ir.TensorType)
    assert result.shape[0] is pages
    assert isinstance(call.args[3], ir.MakeTuple)
    assert call.args[3].elements[0] is pitch
    view = _result_view(call)
    assert view is not None
    assert view.stride[0] is pitch


def test_explicit_strides_dsl_unwraps_scalar_shape_and_stride():
    pages = ir.Var("pages", ir.ScalarType(DataType.INDEX), _span())
    pitch = ir.Var("pitch", ir.ScalarType(DataType.INDEX), _span())
    source = Tensor(expr=_tensor_var([pages, pitch], DataType.INT8))

    result = dsl_tensor_ops.view(source, [Scalar(expr=pages), 32, 128], strides=[Scalar(expr=pitch), 128, 1])

    assert isinstance(result, Tensor)
    result_type = result.unwrap().type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.shape[0] is pages
    assert result_type.tensor_view is not None
    assert result_type.tensor_view.stride[0] is pitch


def test_explicit_strides_preserve_root_read_write_provenance():
    """Root stores and alias stores must promote the same parameter to InOut.

    Use DMA loads/stores: scalar GM access has no explicit-stride contract.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, cache: pl.Tensor[[2, 16], pl.INT32], seed: pl.Tensor[[2, 8], pl.INT32]
        ) -> pl.Tensor[[2, 16], pl.INT32]:
            value = pl.tile.load(seed, [0, 0], [2, 8])
            root_written = pl.tile.store(value, [0, 0], cache)
            keys = pl.tensor.view(root_written, [2, 8], strides=[16, 1])
            from_view = pl.tile.load(keys, [0, 0], [2, 8])
            _view_written = pl.tile.store(from_view, [0, 0], keys)
            _from_root = pl.tile.load(cache, [0, 0], [2, 8])
            return cache

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, cache: pl.InOut[pl.Tensor[[2, 16], pl.INT32]], seed: pl.Tensor[[2, 8], pl.INT32]
        ) -> pl.Tensor[[2, 16], pl.INT32]:
            value = pl.tile.load(seed, [0, 0], [2, 8])
            root_written = pl.tile.store(value, [0, 0], cache)
            keys = pl.tensor.view(root_written, [2, 8], strides=[16, 1])
            from_view = pl.tile.load(keys, [0, 0], [2, 8])
            _view_written = pl.tile.store(from_view, [0, 0], keys)
            _from_root = pl.tile.load(cache, [0, 0], [2, 8])
            return cache

    after = passes.convert_tensor_to_tile_ops()(Before)
    ir.assert_structural_equal(after, Expected)


def test_explicit_strides_require_shape_and_keyword():
    src = _tensor_var([16])
    with pytest.raises(ValueError, match="strides requires shape"):
        ir.op.tensor.view(src, strides=[1])
    with pytest.raises(TypeError):
        inspect.signature(ir.op.tensor.view).bind(src, [16], [], [1])
    with pytest.raises(TypeError):
        inspect.signature(dsl_tensor_ops.view).bind(Tensor(expr=src), [16], [], [1])


@pytest.mark.parametrize(
    ("shape", "strides", "message"),
    [
        ([], [], "rank >= 1"),
        ([2, 4], [1], "strides rank"),
        ([0, 4], [4, 1], "shape element 0 must be positive"),
        ([-1, 4], [4, 1], "shape element 0 must be positive"),
        ([2, 4], [0, 1], "strides element 0 must be positive"),
        ([2, 4], [-8, 1], "strides element 0 must be positive"),
        ([2, 4], [3, 1], "not canonical ND"),
        ([2, 4], [8, 2], "innermost stride"),
        ([2, 4], [32, 1], "exceeds source capacity"),
    ],
)
def test_explicit_strides_reject_invalid_shape_and_stride(shape, strides, message):
    with pytest.raises(ValueError, match=message):
        ir.op.tensor.view(_tensor_var([32]), shape, strides=strides)


@pytest.mark.parametrize("field", ["shape", "strides", "valid_shape"])
def test_explicit_strides_reject_noninteger_elements(field):
    values: dict[str, list[int | ir.Expr]] = {"shape": [2, 4], "strides": [8, 1], "valid_shape": []}
    values[field] = [ir.ConstFloat(2.0, DataType.FP32, _span()), _const(1)]
    with pytest.raises(ValueError, match="integer scalar dtype"):
        ir.op.tensor.view(
            _tensor_var([32]), values["shape"], values["valid_shape"], strides=values["strides"]
        )


@pytest.mark.parametrize("shape", [[0, 8], [-1, 8]])
def test_explicit_strides_reject_nonpositive_source(shape):
    with pytest.raises(ValueError, match="source shape element 0 must be positive"):
        ir.op.tensor.view(_tensor_var(shape), [1], strides=[1])


@pytest.mark.parametrize("valid_shape", [[1, 4], [2]])
def test_explicit_strides_reject_partial_target_validity(valid_shape):
    with pytest.raises(ValueError, match="fully valid target"):
        ir.op.tensor.view(_tensor_var([2, 8]), [2, 4], valid_shape, strides=[8, 1])


def test_explicit_strides_reject_partial_source_validity():
    view = ir.TensorView([8, 1], ir.TensorLayout.ND, [1, 8])
    with pytest.raises(ValueError, match="fully valid source"):
        ir.op.tensor.view(_tensor_var([2, 8], view=view), [2, 4], strides=[8, 1])


def test_explicit_strides_reject_gapped_source():
    view = ir.TensorView([16, 1], ir.TensorLayout.ND)
    with pytest.raises(ValueError, match="packed ND source"):
        ir.op.tensor.view(_tensor_var([2, 8], view=view), [2, 4], strides=[8, 1])


@pytest.mark.parametrize("layout", [ir.TensorLayout.DN, ir.TensorLayout.NZ, ir.TensorLayout.MX_A_ZZ])
def test_explicit_strides_reject_non_nd_layout(layout):
    src = _tensor_var([2, 8])
    with pytest.raises(ValueError, match="ND source and target"):
        ir.op.tensor.view(src, [2, 4], strides=[8, 1], layout=layout)
    source_view = ir.TensorView([], layout)
    with pytest.raises(ValueError, match="ND source and target"):
        ir.op.tensor.view(_tensor_var([2, 8], view=source_view), [2, 4], strides=[8, 1])


@pytest.mark.parametrize("dtype", [DataType.FP4E2M1X2, DataType.FP4E1M2X2, DataType.INT4, DataType.BOOL])
def test_explicit_strides_reject_packed_and_subbyte_dtypes(dtype):
    with pytest.raises(ValueError, match="packed FP4 or sub-byte"):
        ir.op.tensor.view(_tensor_var([32], dtype=dtype), [16], strides=[1])


def test_explicit_strides_reject_distributed_source():
    src = ir.Var("distributed", ir.DistributedTensorType([2, 8], DataType.INT8), _span())
    with pytest.raises(ValueError, match="do not support distributed"):
        ir.op.tensor.view(src, [2, 4], strides=[8, 1])


@pytest.mark.parametrize("symbolic", [False, True])
def test_explicit_strides_reject_capacity_overflow_with_symbolic_prefix(symbolic):
    """A dynamic first axis must not hide overflowing known positive factors."""
    prefix = ir.Var("pages", ir.ScalarType(DataType.INDEX), _span()) if symbolic else _const(1)
    src = _tensor_var([prefix, _const(1 << 62), _const(4)])
    with pytest.raises(ValueError, match="source capacity overflows int64"):
        ir.op.tensor.view(src, [1], strides=[1])


@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("overflow", ["multiply", "add"])
def test_explicit_strides_reject_span_overflow_with_symbolic_prefix(symbolic, overflow):
    """Validate all static span terms even when another contribution is unknown."""
    prefix = ir.Var("pages", ir.ScalarType(DataType.INDEX), _span()) if symbolic else _const(1)
    if overflow == "multiply":
        shape, strides = [prefix, 3, 1], [1 << 62, 1 << 62, 1]
    else:
        shape, strides = [prefix, 2, 2, 1], [1 << 62, 1 << 62, 1 << 62, 1]
    with pytest.raises(ValueError, match="storage span overflows int64"):
        ir.op.tensor.view(_tensor_var([(1 << 63) - 1], DataType.INT8), shape, strides=strides)


def test_explicit_strides_reject_known_out_of_bounds_with_symbolic_axis():
    pages = ir.Var("pages", ir.ScalarType(DataType.INDEX), _span())
    with pytest.raises(ValueError, match="exceeds source capacity"):
        ir.op.tensor.view(_tensor_var([16]), [pages, 2, 4], strides=[32, 16, 1])


@pytest.mark.parametrize("symbolic", [False, True])
def test_explicit_strides_reject_canonical_product_overflow(symbolic):
    """A fitting last address must not hide an impossible outer-stride bound."""
    maximum = (1 << 63) - 1
    prefix = ir.Var("pages", ir.ScalarType(DataType.INDEX), _span()) if symbolic else _const(1)
    # The known span is exactly maximum, but stride[1] * shape[1] exceeds
    # int64. The shared relaxed verifier alone would treat that as symbolic.
    with pytest.raises(ValueError, match="canonical stride product overflows int64"):
        ir.op.tensor.view(
            _tensor_var([maximum], DataType.INT8), [prefix, 2, 1], strides=[maximum, maximum - 1, 1]
        )


@pytest.mark.parametrize("symbolic", [False, True])
def test_explicit_strides_reject_source_byte_capacity_overflow(symbolic):
    prefix = ir.Var("pages", ir.ScalarType(DataType.INDEX), _span()) if symbolic else _const(1)
    source = _tensor_var([prefix, _const(1 << 62)], DataType.FP16)
    with pytest.raises(ValueError, match="source capacity in bytes overflows int64"):
        ir.op.tensor.view(source, [1], strides=[1])


@pytest.mark.parametrize("symbolic", [False, True])
def test_explicit_strides_reject_storage_byte_span_overflow(symbolic):
    prefix = ir.Var("pages", ir.ScalarType(DataType.INDEX), _span()) if symbolic else _const(1)
    with pytest.raises(ValueError, match="storage span in bytes overflows int64"):
        ir.op.tensor.view(_tensor_var([16], DataType.FP16), [prefix, 2, 1], strides=[1 << 62, 1 << 62, 1])


def test_explicit_strides_do_not_relax_ordinary_view():
    src = _tensor_var([9, 4160], DataType.INT8)
    with pytest.raises(ValueError, match="cannot reinterpret tensor of size"):
        ir.op.tensor.view(src, [9, 32, 128])
    view = ir.op.tensor.view(src, [9, 4160])
    assert len(view.args) == 2
    metadata = _result_view(view)
    assert metadata is not None
    assert _values_of(metadata.stride) == [4160, 1]


# ============================================================================
# Op-registry sanity
# ============================================================================


def test_op_registered():
    """``tensor.view`` must be discoverable through the OpRegistry."""
    assert ir.is_op_registered("tensor.view")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

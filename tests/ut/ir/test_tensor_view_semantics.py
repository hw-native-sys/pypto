# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for tensor_view_semantics helpers (RFC #1300, P1).

The helpers under test live in
``include/pypto/ir/transforms/utils/tensor_view_semantics.h`` and are exposed
to Python via ``ir.tensor_view_semantics``. They define the canonical
(shape, stride, layout) invariants used by later phases (P2 verifier, P3
materialization pass).
"""

import pytest
from pypto import DataType, ir

tvs = ir.tensor_view_semantics


def _span():
    return ir.Span.unknown()


def _const(value: int, dtype: DataType = DataType.INDEX):
    return ir.ConstInt(value, dtype, _span())


def _shape(*dims):
    return [_const(d) for d in dims]


def _stride(*vals):
    return [_const(v) for v in vals]


def _const_value(expr):
    """Extract int value from a ConstInt expression for assertions."""
    assert isinstance(expr, ir.ConstInt), f"expected ConstInt, got {type(expr).__name__}"
    return expr.value


def _values_of(exprs):
    return [_const_value(e) for e in exprs]


# ============================================================================
# BuildLogicalStridesFromLayout
# ============================================================================


def test_build_nd_packed_2d():
    strides = tvs.build_logical_strides_from_layout(_shape(8, 16), ir.TensorLayout.ND)
    assert _values_of(strides) == [16, 1]


def test_build_nd_packed_3d():
    strides = tvs.build_logical_strides_from_layout(_shape(2, 4, 8), ir.TensorLayout.ND)
    # stride[2]=1, stride[1]=8, stride[0]=4*8=32
    assert _values_of(strides) == [32, 8, 1]


def test_build_dn_packed_2d():
    # K=4, N=8 -> stride[0]=1, stride[1]=K=4
    strides = tvs.build_logical_strides_from_layout(_shape(4, 8), ir.TensorLayout.DN)
    assert _values_of(strides) == [1, 4]


def test_build_dn_packed_3d():
    # B=2, K=4, N=8 -> stride[1]=1, stride[2]=K=4, stride[0]=K*N=32
    strides = tvs.build_logical_strides_from_layout(_shape(2, 4, 8), ir.TensorLayout.DN)
    assert _values_of(strides) == [32, 1, 4]


def test_build_dn_packed_4d():
    # shape [B0, B1, K, N] = [2, 3, 4, 8]
    # innermost two: stride[2]=1, stride[3]=4
    # stride[1] = K*N = 32
    # stride[0] = stride[1] * shape[1] = 32 * 3 = 96
    strides = tvs.build_logical_strides_from_layout(_shape(2, 3, 4, 8), ir.TensorLayout.DN)
    assert _values_of(strides) == [96, 32, 1, 4]


def test_build_nz_rejected():
    with pytest.raises(ValueError, match="NZ"):
        tvs.build_logical_strides_from_layout(_shape(8, 16), ir.TensorLayout.NZ)


def test_build_dn_rank1_rejected():
    with pytest.raises(ValueError, match="rank >= 2"):
        tvs.build_logical_strides_from_layout(_shape(8), ir.TensorLayout.DN)


def test_build_empty_shape_returns_empty():
    assert tvs.build_logical_strides_from_layout([], ir.TensorLayout.ND) == []


# ============================================================================
# DeriveLayoutFromStrides
# ============================================================================


def test_derive_nd_packed():
    assert tvs.derive_layout_from_strides(_shape(8, 16), _stride(16, 1)) == ir.TensorLayout.ND


def test_derive_nd_strided():
    # Sub-view of a row-major parent: stride[-1]=1 still, outer stride larger
    # than packed -> still ND family.
    assert tvs.derive_layout_from_strides(_shape(4, 8), _stride(16, 1)) == ir.TensorLayout.ND


def test_derive_dn_packed():
    assert tvs.derive_layout_from_strides(_shape(4, 8), _stride(1, 4)) == ir.TensorLayout.DN


def test_derive_dn_strided():
    # DN sub-view: stride[-2]=1, stride[-1] > shape[-2]
    assert tvs.derive_layout_from_strides(_shape(2, 4), _stride(1, 8)) == ir.TensorLayout.DN


def test_derive_unknown_for_arbitrary():
    # Neither stride[-1]==1 nor stride[-2]==1 statically.
    assert tvs.derive_layout_from_strides(_shape(8, 16), _stride(2, 4)) is None


def test_derive_unknown_for_rank_mismatch():
    assert tvs.derive_layout_from_strides(_shape(8, 16), _stride(1)) is None


def test_derive_unknown_for_empty():
    assert tvs.derive_layout_from_strides([], []) is None


# ============================================================================
# CheckCanonicalView (returns (ok, reason))
# ============================================================================


def test_check_passes_packed_nd():
    ok, reason = tvs.check_canonical_view(_shape(8, 16), _stride(16, 1), ir.TensorLayout.ND)
    assert ok, reason


def test_check_passes_packed_dn():
    ok, reason = tvs.check_canonical_view(_shape(4, 8), _stride(1, 4), ir.TensorLayout.DN)
    assert ok, reason


def test_check_passes_strided_nd_subview():
    # parent shape [8, 16] -> sub [4, 8]; stride inherited [16, 1].
    ok, reason = tvs.check_canonical_view(_shape(4, 8), _stride(16, 1), ir.TensorLayout.ND)
    assert ok, reason


def test_check_passes_strided_dn_subview():
    # parent [4, 8] DN with stride [1, 4]; sub [2, 4] inherits [1, 4].
    ok, reason = tvs.check_canonical_view(_shape(2, 4), _stride(1, 4), ir.TensorLayout.DN)
    assert ok, reason


def test_check_rejects_nz_on_tensor():
    ok, reason = tvs.check_canonical_view(_shape(8, 16), _stride(16, 1), ir.TensorLayout.NZ)
    assert not ok
    assert "NZ" in reason


def test_check_rejects_empty_stride():
    ok, reason = tvs.check_canonical_view(_shape(8, 16), [], ir.TensorLayout.ND)
    assert not ok
    assert "stride is empty" in reason


def test_check_rejects_rank_mismatch():
    ok, reason = tvs.check_canonical_view(_shape(8, 16), _stride(1), ir.TensorLayout.ND)
    assert not ok
    assert "rank" in reason


def test_check_rejects_layout_tag_mismatch_nd_with_dn_stride():
    # stride [1, 4] is DN-shaped, but layout tag claims ND -> innermost stride
    # is not 1, so ND check fails.
    ok, reason = tvs.check_canonical_view(_shape(4, 8), _stride(1, 4), ir.TensorLayout.ND)
    assert not ok
    assert "ND" in reason and "innermost" in reason


def test_check_rejects_layout_tag_mismatch_dn_with_nd_stride():
    # stride [16, 1] is ND-shaped, layout tag claims DN -> stride[-2] not 1.
    ok, reason = tvs.check_canonical_view(_shape(8, 16), _stride(16, 1), ir.TensorLayout.DN)
    assert not ok
    assert "DN" in reason and "stride[-2]" in reason


def test_check_rejects_too_small_outer_stride_nd():
    # ND with shape [4, 8]: packed stride is [8, 1]. stride [4, 1] is too small.
    ok, reason = tvs.check_canonical_view(_shape(4, 8), _stride(4, 1), ir.TensorLayout.ND)
    assert not ok
    assert "smaller than packed" in reason


def test_check_rejects_dn_trailing_stride_too_small():
    # DN with shape [4, 8]: trailing stride must be >= shape[-2] = 4.
    ok, reason = tvs.check_canonical_view(_shape(4, 8), _stride(1, 2), ir.TensorLayout.DN)
    assert not ok
    assert "DN" in reason and "shape[-2]" in reason


def test_check_zero_rank_canonical():
    ok, reason = tvs.check_canonical_view([], [], ir.TensorLayout.ND)
    assert ok, reason


# ============================================================================
# Symbolic strides — RFC Open Q2 (relaxed_symbolic mode)
# ============================================================================


def _sym(name: str):
    """Build a symbolic shape variable (Var of ScalarType INDEX)."""
    return ir.Var(name, ir.ScalarType(DataType.INDEX), _span())


def test_check_relaxed_symbolic_dn_passes():
    # [K_sym, N_sym] DN with stride [1, K_sym]: trailing stride symbolic, but
    # stride[-2]==1 structurally holds. relaxed_symbolic=True (default) should
    # accept.
    K = _sym("K")
    N = _sym("N")
    one = _const(1)
    ok, reason = tvs.check_canonical_view([K, N], [one, K], ir.TensorLayout.DN)
    assert ok, reason


def test_check_strict_symbolic_dn_fails():
    # Same input as above with relaxed_symbolic=False should refuse to certify
    # the symbolic case.
    K = _sym("K")
    N = _sym("N")
    one = _const(1)
    ok, reason = tvs.check_canonical_view([K, N], [one, K], ir.TensorLayout.DN, False)
    assert not ok
    assert "symbolic" in reason


# ============================================================================
# CanonicalizeView convenience wrapper
# ============================================================================


def test_canonicalize_view_nd_2d():
    view = tvs.canonicalize_view(_shape(8, 16), ir.TensorLayout.ND)
    assert view.layout == ir.TensorLayout.ND
    assert _values_of(view.stride) == [16, 1]
    assert list(view.valid_shape) == []


def test_canonicalize_view_dn_2d():
    view = tvs.canonicalize_view(_shape(4, 8), ir.TensorLayout.DN)
    assert view.layout == ir.TensorLayout.DN
    assert _values_of(view.stride) == [1, 4]


# ============================================================================
# ComputeShapeProduct
# ============================================================================


def test_compute_shape_product_static():
    assert tvs.compute_shape_product(_shape(2, 3, 5)) == 30


def test_compute_shape_product_empty():
    assert tvs.compute_shape_product([]) == 1


def test_compute_shape_product_dynamic_returns_minus_one():
    K = _sym("K")
    assert tvs.compute_shape_product([K, _const(8)]) == -1


# ============================================================================
# TensorType valid_shape canonicalization (unset valid_shape means fully valid)
# ============================================================================
#
# ``docs/en/dev/ir/08-valid_shape.md``: an explicit
# ``valid_shape`` equal to the physical shape carries no information and must
# canonicalize to empty, so "unset" and "fully valid" share one in-memory form —
# matching ``TileView``. When nothing else in the view is meaningful, the whole
# view collapses to ``None`` (one canonical form per semantic state). A narrower
# or dynamic ``valid_shape``, or a meaningful stride/layout/pad, is preserved.


def _tensor_view(stride, layout=ir.TensorLayout.ND, valid_shape=None, pad=ir.PadValue.null):
    return ir.TensorView(
        stride=stride,
        layout=layout,
        valid_shape=valid_shape if valid_shape is not None else [],
        pad=pad,
    )


def test_tensor_valid_shape_equal_to_shape_canonicalizes_to_empty():
    """valid_shape == physical shape carries no info: the fully-default residual view
    collapses to None and compares structurally-equal to the no-view TensorType."""
    view = _tensor_view(stride=[], valid_shape=_shape(128, 128))
    t = ir.TensorType(_shape(128, 128), DataType.FP32, None, view)
    # Only valid_shape was set, and it matched shape -> nothing meaningful remains,
    # so the whole view collapses to None (one in-memory form per semantic state).
    assert t.tensor_view is None
    no_view = ir.TensorType(_shape(128, 128), DataType.FP32)
    assert ir.structural_equal(t, no_view)
    assert ir.structural_hash(t) == ir.structural_hash(no_view)


def test_tensor_narrower_valid_shape_preserved():
    """A strictly narrower static valid_shape is real information — preserved."""
    view = _tensor_view(stride=[], valid_shape=_shape(64, 128))
    t = ir.TensorType(_shape(128, 128), DataType.FP32, None, view)
    assert t.tensor_view is not None
    assert _values_of(t.tensor_view.valid_shape) == [64, 128]
    # Distinct from the fully-valid (no-view) form.
    assert not ir.structural_equal(t, ir.TensorType(_shape(128, 128), DataType.FP32))


def test_tensor_dynamic_valid_shape_preserved():
    """A symbolic valid_shape dim is not statically equal to shape — preserved."""
    valid_len = _sym("valid_len")
    view = _tensor_view(stride=[], valid_shape=[valid_len, _const(128)])
    t = ir.TensorType(_shape(128, 128), DataType.FP32, None, view)
    assert t.tensor_view is not None
    assert len(t.tensor_view.valid_shape) == 2
    assert t.tensor_view.valid_shape[0] is valid_len


def test_tensor_valid_shape_cleared_preserves_stride_layout_pad():
    """When valid_shape == shape is cleared, meaningful stride / layout / pad on the
    same TensorView survive (the whole view is NOT reset, unlike TileType)."""
    view = _tensor_view(
        stride=_stride(128, 1),
        layout=ir.TensorLayout.DN,
        valid_shape=_shape(128, 128),
        pad=ir.PadValue.zero,
    )
    t = ir.TensorType(_shape(128, 128), DataType.FP32, None, view)
    assert t.tensor_view is not None
    assert list(t.tensor_view.valid_shape) == []  # redundant valid_shape dropped
    assert _values_of(t.tensor_view.stride) == [128, 1]  # stride preserved
    assert t.tensor_view.layout == ir.TensorLayout.DN  # layout preserved
    assert t.tensor_view.pad == ir.PadValue.zero  # pad preserved


def test_tile_type_collapses_whole_view_when_full():
    """TileType drops the whole view when every field matches implicit semantics."""
    tile_view = ir.TileView(valid_shape=_shape(128, 128), stride=[], start_offset=None)
    tile = ir.TileType(_shape(128, 128), DataType.FP32, None, tile_view)
    assert tile.tile_view is None


def test_tile_full_valid_shape_cleared_preserves_nondefault_view_fields():
    """A redundant full valid_shape is independent of other TileView metadata."""
    start_offset = _const(7)
    tile_view = ir.TileView(
        valid_shape=_shape(128, 128),
        stride=_stride(128, 1),
        start_offset=start_offset,
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        fractal=1024,
        pad=ir.PadValue.zero,
    )
    tile = ir.TileType(_shape(128, 128), DataType.FP32, None, tile_view, ir.MemorySpace.Vec)

    assert tile.tile_view is not None
    assert list(tile.tile_view.valid_shape) == []
    assert _values_of(tile.tile_view.stride) == [128, 1]
    assert _const_value(tile.tile_view.start_offset) == 7
    assert tile.tile_view.blayout == ir.TileLayout.col_major
    assert tile.tile_view.slayout == ir.TileLayout.row_major
    assert tile.tile_view.fractal == 1024
    assert tile.tile_view.pad == ir.PadValue.zero
    # The stored field is canonical-empty, but D2 still makes the effective view
    # fully valid for every consumer.
    assert _values_of(tile.get_effective_tile_view().valid_shape) == [128, 128]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

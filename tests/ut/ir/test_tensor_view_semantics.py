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

import ast

import pytest
from pypto import DataType, ir
from pypto.language.parser.expr_evaluator import ExprEvaluator
from pypto.language.parser.type_resolver import TypeResolver

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


def _assert_type_print_round_trip(original):
    printed = ir.python_print_type(original)
    node = ast.parse(printed, mode="eval").body
    restored = TypeResolver(expr_evaluator=ExprEvaluator(closure_vars={})).resolve_type(node)
    assert isinstance(restored, ir.Type)
    original_var = ir.Var("value", original, _span())
    restored_var = ir.Var("value", restored, _span())
    ir.assert_structural_equal(original_var, restored_var, enable_auto_mapping=True)


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
# TensorType and TileType valid_shape canonicalization
# ============================================================================


def test_tensor_type_without_view_stays_without_view():
    tensor_type = ir.TensorType([8, 16], DataType.FP32)
    assert tensor_type.tensor_view is None


def test_tensor_type_explicit_full_default_view_collapses():
    view = ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=[8, 16])
    tensor_type = ir.TensorType([8, 16], DataType.FP32, tensor_view=view)
    assert tensor_type.tensor_view is None


def test_tensor_type_full_valid_shape_preserves_other_view_metadata():
    view = ir.TensorView(
        stride=[1, 8],
        layout=ir.TensorLayout.DN,
        valid_shape=[8, 16],
        pad=ir.PadValue.max,
    )
    tensor_type = ir.TensorType([8, 16], DataType.FP32, tensor_view=view)

    assert tensor_type.tensor_view is not None
    assert list(tensor_type.tensor_view.valid_shape) == []
    assert _values_of(tensor_type.tensor_view.stride) == [1, 8]
    assert tensor_type.tensor_view.layout == ir.TensorLayout.DN
    assert tensor_type.tensor_view.pad == ir.PadValue.max


def test_tensor_type_partial_and_symbolic_valid_shapes_stay_explicit():
    partial_type = ir.TensorType(
        [8, 16],
        DataType.FP32,
        tensor_view=ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=[4, 16]),
    )
    assert partial_type.tensor_view is not None
    assert _values_of(partial_type.tensor_view.valid_shape) == [4, 16]

    valid_rows = _sym("valid_rows")
    symbolic_type = ir.TensorType(
        [8, 16],
        DataType.FP32,
        tensor_view=ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=[valid_rows, 16]),
    )
    assert symbolic_type.tensor_view is not None
    assert symbolic_type.tensor_view.valid_shape[0] is valid_rows


def test_tensor_type_symbolic_full_valid_shape_collapses():
    rows = _sym("rows")
    tensor_type = ir.TensorType(
        [rows, _const(16)],
        DataType.FP32,
        tensor_view=ir.TensorView(
            layout=ir.TensorLayout.ND,
            valid_shape=[rows, _const(16)],
        ),
    )
    assert tensor_type.tensor_view is None


def test_tile_type_without_view_stays_without_view():
    tile_type = ir.TileType([8, 16], DataType.FP32)
    assert tile_type.tile_view is None


def test_tile_type_explicit_full_default_view_collapses():
    view = ir.TileView(valid_shape=[8, 16])
    tile_type = ir.TileType([8, 16], DataType.FP32, tile_view=view)
    assert tile_type.tile_view is None


def test_tile_type_symbolic_full_valid_shape_collapses():
    rows = _sym("rows")
    tile_type = ir.TileType(
        [rows, _const(16)],
        DataType.FP32,
        tile_view=ir.TileView(valid_shape=[rows, _const(16)]),
    )
    assert tile_type.tile_view is None


def test_tile_type_full_valid_shape_preserves_other_view_metadata():
    start_offset = _const(3)
    view = ir.TileView(
        valid_shape=[8, 16],
        stride=[32, 2],
        start_offset=start_offset,
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        fractal=1024,
        pad=ir.PadValue.min,
    )
    tile_type = ir.TileType([8, 16], DataType.FP32, tile_view=view)

    assert tile_type.tile_view is not None
    assert list(tile_type.tile_view.valid_shape) == []
    assert _values_of(tile_type.tile_view.stride) == [32, 2]
    assert tile_type.tile_view.start_offset is start_offset
    assert tile_type.tile_view.blayout == ir.TileLayout.col_major
    assert tile_type.tile_view.slayout == ir.TileLayout.row_major
    assert tile_type.tile_view.fractal == 1024
    assert tile_type.tile_view.pad == ir.PadValue.min


def test_tile_type_partial_and_symbolic_valid_shapes_stay_explicit():
    partial_type = ir.TileType([8, 16], DataType.FP32, tile_view=ir.TileView(valid_shape=[4, 16]))
    assert partial_type.tile_view is not None
    assert _values_of(partial_type.tile_view.valid_shape) == [4, 16]

    valid_rows = _sym("valid_rows")
    symbolic_type = ir.TileType([8, 16], DataType.FP32, tile_view=ir.TileView(valid_shape=[valid_rows, 16]))
    assert symbolic_type.tile_view is not None
    assert symbolic_type.tile_view.valid_shape[0] is valid_rows


def test_canonical_view_types_survive_print_parse_round_trip():
    tensor_type = ir.TensorType(
        [8, 16],
        DataType.FP32,
        tensor_view=ir.TensorView(
            stride=[1, 8],
            layout=ir.TensorLayout.DN,
            valid_shape=[8, 16],
        ),
    )
    tile_type = ir.TileType(
        [8, 16],
        DataType.FP32,
        tile_view=ir.TileView(valid_shape=[4, 16]),
    )

    _assert_type_print_round_trip(tensor_type)
    _assert_type_print_round_trip(tile_type)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

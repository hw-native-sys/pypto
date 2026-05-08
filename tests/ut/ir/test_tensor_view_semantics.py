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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

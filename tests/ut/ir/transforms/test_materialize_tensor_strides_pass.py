# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for MaterializeTensorStrides pass (RFC #1300, P3).

The pass walks every TensorType in a Program and replaces any
``view.has_value() && view.stride.empty()`` slot with the packed canonical
stride for the carried layout. Bare TensorTypes and already-explicit views
pass through unchanged.

After this pass runs, the codegen-entry contract holds: every
``view.has_value()`` slot has explicit stride matching its layout — which
the strict ``TensorViewCanonical`` verifier enforces.

Tests follow the Before/Expected ``@pl.program`` pattern: the pass runs on
``Before`` to produce ``After``, which is compared against ``Expected`` via
``ir.assert_structural_equal``. Skip / no-op cases compare ``After`` against
``Before``.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.pypto_core import passes as _passes

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _materialize(program):
    return _passes.materialize_tensor_strides()(program)


def _verify_strict(program):
    """Run TensorViewCanonical in strict mode — empty stride is rejected."""
    return _passes.verify_tensor_view_canonical(program, require_materialized=True)


# ============================================================================
# Bare tensor stays bare; strict verifier still passes (treated as implicit ND).
# ============================================================================


def test_bare_tensor_unchanged():
    @pl.program
    class Before:
        @pl.function
        def f(self, x: pl.Tensor[[8, 16], pl.FP32]):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    # Bare TensorType has no view to materialize: pass is a no-op.
    ir.assert_structural_equal(After, Before)
    # Strict verifier accepts a bare tensor (implicit ND).
    assert _verify_strict(After) == []


# ============================================================================
# Empty stride filled with packed canonical
# ============================================================================


def test_empty_dn_stride_filled_2d():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    ir.assert_structural_equal(After, Expected)
    # Strict verifier accepts the materialized form.
    assert _verify_strict(After) == []


def test_empty_dn_stride_filled_3d():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[2, 4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            # B=2, K=4, N=8 -> stride=[K*N, 1, K]=[32, 1, 4]
            x: pl.Tensor[[2, 4, 8], pl.FP32, pl.TensorView(stride=[32, 1, 4], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    ir.assert_structural_equal(After, Expected)
    assert _verify_strict(After) == []


def test_empty_nd_stride_filled():
    # An ND view with empty stride is also materialized to row-major packed.
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[8, 16], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.ND)],
        ):
            pl.const(0, pl.INT64)

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[8, 16], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    ir.assert_structural_equal(After, Expected)


# ============================================================================
# Already-explicit view stays unchanged (no spurious rewrite)
# ============================================================================


def test_explicit_packed_nd_unchanged():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[8, 16], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    # Identity preservation: pass returns the same Program when nothing changed.
    assert After is Before
    ir.assert_structural_equal(After, Before)


def test_explicit_packed_dn_unchanged():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    assert After is Before
    ir.assert_structural_equal(After, Before)


def test_strided_dn_subview_unchanged():
    # Inherited from a parent — stride larger than DN-packed for the sub-shape.
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[2, 4], pl.FP32, pl.TensorView(stride=[1, 8], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    assert After is Before
    ir.assert_structural_equal(After, Before)


# ============================================================================
# NZ on TensorType is left untouched (verifier rejects it; pass doesn't crash)
# ============================================================================


def test_nz_on_tensor_rejected_by_paired_verifier():
    # NZ on a TensorType is invalid IR. The pass leaves the slot untouched
    # rather than CHECK-failing inside BuildLogicalStridesFromLayout — but
    # because the pass produces TensorViewCanonical, PassPipeline runs the
    # paired verifier, which surfaces the bug as a thrown ValueError.
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[8, 16], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        ):
            pl.const(0, pl.INT64)

    with pytest.raises(ValueError, match="NZ layout"):
        _materialize(Before)


# ============================================================================
# Idempotence
# ============================================================================


def test_idempotent_after_first_pass():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    once = _materialize(Before)
    twice = _materialize(once)
    # Second invocation is a no-op: nothing to materialize, identity preserved.
    assert twice is once
    ir.assert_structural_equal(twice, once)


# ============================================================================
# Symbolic shape: stride expressions stay symbolic.
# ============================================================================


def test_symbolic_dn_materialized_preserves_symbols():
    K = pl.dynamic("K")
    N = pl.dynamic("N")

    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[K, N], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            # DN-packed: stride[-2] == 1, stride[-1] == K (the symbolic Var).
            x: pl.Tensor[[K, N], pl.FP32, pl.TensorView(stride=[1, K], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    ir.assert_structural_equal(After, Expected)


# ============================================================================
# Pass plays well with the canonical verifier as a paired guarantee.
# ============================================================================


def test_strict_verifier_passes_after_materialization():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    # Before materialization, strict mode rejects empty stride.
    diags_before = _verify_strict(Before)
    assert any("stride is empty" in d.message for d in diags_before)
    # After materialization, strict mode accepts and IR matches Expected.
    After = _materialize(Before)
    ir.assert_structural_equal(After, Expected)
    assert _verify_strict(After) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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
"""

import pytest
from pypto import DataType, ir
from pypto.pypto_core import passes as _passes

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _span():
    return ir.Span.unknown()


def _const(value: int, dtype: DataType = DataType.INDEX):
    return ir.ConstInt(value, dtype, _span())


def _shape(*dims):
    return [_const(d) for d in dims]


def _stride(*vals):
    return [_const(v) for v in vals]


def _empty_body():
    return ir.EvalStmt(_const(0, DataType.INT64), _span())


def _program_with_param_type(tensor_type):
    span = _span()
    var = ir.Var("x", tensor_type, span)
    func = ir.Function("f", [var], [], _empty_body(), span)
    return ir.Program([func], "p", span)


def _materialize(program):
    return _passes.materialize_tensor_strides()(program)


def _const_value(expr):
    assert isinstance(expr, ir.ConstInt), f"expected ConstInt, got {type(expr).__name__}"
    return expr.value


def _values_of(exprs):
    return [_const_value(e) for e in exprs]


def _verify_strict(program):
    """Run TensorViewCanonical in strict mode — empty stride is rejected."""
    return _passes.verify_tensor_view_canonical(program, require_materialized=True)


def _param_tensor_type(program):
    """Return the TensorType attached to the single param of 'f'.

    Wraps the get_function/params/type chain in assertions so callers get
    a precise TensorType handle (and pyright sees the narrowing).
    """
    func = program.get_function("f")
    assert func is not None
    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    return param_type


def _param_view(program):
    """Return the TensorView (or None) attached to the single param of 'f'."""
    return _param_tensor_type(program).tensor_view


# ============================================================================
# Bare tensor stays bare; strict verifier still passes (treated as implicit ND).
# ============================================================================


def test_bare_tensor_unchanged():
    t = ir.TensorType(_shape(8, 16), DataType.FP32)
    program = _program_with_param_type(t)
    out = _materialize(program)
    assert _param_view(out) is None
    assert _verify_strict(out) == []


# ============================================================================
# Empty stride filled with packed canonical
# ============================================================================


def test_empty_dn_stride_filled_2d():
    view = ir.TensorView([], ir.TensorLayout.DN)
    t = ir.TensorType(_shape(4, 8), DataType.FP32, None, view)
    program = _program_with_param_type(t)
    out = _materialize(program)
    out_view = _param_view(out)
    assert out_view is not None
    assert out_view.layout == ir.TensorLayout.DN
    assert _values_of(out_view.stride) == [1, 4]
    # Strict verifier accepts the materialized form.
    assert _verify_strict(out) == []


def test_empty_dn_stride_filled_3d():
    view = ir.TensorView([], ir.TensorLayout.DN)
    t = ir.TensorType(_shape(2, 4, 8), DataType.FP32, None, view)
    program = _program_with_param_type(t)
    out = _materialize(program)
    out_view = _param_view(out)
    assert out_view is not None
    # B=2, K=4, N=8 -> stride=[K*N, 1, K]=[32, 1, 4]
    assert _values_of(out_view.stride) == [32, 1, 4]
    assert _verify_strict(out) == []


def test_empty_nd_stride_filled():
    # An ND view with empty stride is also materialized to row-major packed.
    view = ir.TensorView([], ir.TensorLayout.ND)
    t = ir.TensorType(_shape(8, 16), DataType.FP32, None, view)
    out_view = _param_view(_materialize(_program_with_param_type(t)))
    assert out_view is not None
    assert _values_of(out_view.stride) == [16, 1]


# ============================================================================
# Already-explicit view stays unchanged (no spurious rewrite)
# ============================================================================


def test_explicit_packed_nd_unchanged():
    view = ir.TensorView(_stride(16, 1), ir.TensorLayout.ND)
    t = ir.TensorType(_shape(8, 16), DataType.FP32, None, view)
    program = _program_with_param_type(t)
    out = _materialize(program)
    # Identity preservation: pass returns the same Program when nothing changed.
    assert out is program


def test_explicit_packed_dn_unchanged():
    view = ir.TensorView(_stride(1, 4), ir.TensorLayout.DN)
    t = ir.TensorType(_shape(4, 8), DataType.FP32, None, view)
    program = _program_with_param_type(t)
    assert _materialize(program) is program


def test_strided_dn_subview_unchanged():
    # Inherited from a parent — stride larger than DN-packed for the sub-shape.
    view = ir.TensorView(_stride(1, 8), ir.TensorLayout.DN)
    t = ir.TensorType(_shape(2, 4), DataType.FP32, None, view)
    program = _program_with_param_type(t)
    assert _materialize(program) is program


# ============================================================================
# NZ on TensorType is left untouched (verifier rejects it; pass doesn't crash)
# ============================================================================


def test_nz_on_tensor_rejected_by_paired_verifier():
    # NZ on a TensorType is invalid IR. The pass leaves the slot untouched
    # rather than CHECK-failing inside BuildLogicalStridesFromLayout — but
    # because the pass produces TensorViewCanonical, PassPipeline runs the
    # paired verifier, which surfaces the bug as a thrown ValueError.
    view = ir.TensorView([], ir.TensorLayout.NZ)
    t = ir.TensorType(_shape(8, 16), DataType.FP32, None, view)
    program = _program_with_param_type(t)
    with pytest.raises(ValueError, match="NZ layout"):
        _materialize(program)


# ============================================================================
# Idempotence
# ============================================================================


def test_idempotent_after_first_pass():
    view = ir.TensorView([], ir.TensorLayout.DN)
    t = ir.TensorType(_shape(4, 8), DataType.FP32, None, view)
    program = _program_with_param_type(t)
    once = _materialize(program)
    twice = _materialize(once)
    # Second invocation is a no-op: nothing to materialize, identity preserved.
    assert twice is once


# ============================================================================
# Symbolic shape: stride expressions stay symbolic.
# ============================================================================


def test_symbolic_dn_materialized_preserves_symbols():
    K = ir.Var("K", ir.ScalarType(DataType.INDEX), _span())
    N = ir.Var("N", ir.ScalarType(DataType.INDEX), _span())
    view = ir.TensorView([], ir.TensorLayout.DN)
    t = ir.TensorType([K, N], DataType.FP32, None, view)
    program = _program_with_param_type(t)
    out_view = _param_view(_materialize(program))
    assert out_view is not None
    assert len(out_view.stride) == 2
    # stride[-2] == ConstInt(1)
    inner = out_view.stride[0]
    assert isinstance(inner, ir.ConstInt) and inner.value == 1
    # stride[-1] == K (the symbolic Var, not a ConstInt)
    trailing = out_view.stride[1]
    assert isinstance(trailing, ir.Var)
    assert trailing.name_hint == "K"


# ============================================================================
# Pass plays well with the canonical verifier as a paired guarantee.
# ============================================================================


def test_strict_verifier_passes_after_materialization():
    view = ir.TensorView([], ir.TensorLayout.DN)
    t = ir.TensorType(_shape(4, 8), DataType.FP32, None, view)
    program = _program_with_param_type(t)
    # Before materialization, strict mode rejects empty stride.
    diags_before = _verify_strict(program)
    assert any("stride is empty" in d.message for d in diags_before)
    # After materialization, strict mode accepts.
    diags_after = _verify_strict(_materialize(program))
    assert diags_after == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

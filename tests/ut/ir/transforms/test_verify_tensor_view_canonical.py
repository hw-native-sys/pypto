# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the TensorViewCanonical verifier (RFC #1300, P2).

The verifier walks every TensorType reachable from a program (function
params, return types, body Vars, IterArgs, Call return types) and asserts
the canonical (shape, stride, layout) form per RFC #1300 §2.2.

Two modes are tested:
- weak (default): empty stride is accepted as implicitly packed canonical.
- strict (``require_materialized=True``): empty stride is rejected — the
  contract enforced at codegen entry once P3's MaterializeTensorStrides runs.
"""

import pytest
from pypto import DataType, ir
from pypto.pypto_core import passes as _passes

# ----------------------------------------------------------------------------
# Helpers — build small functions whose only meaningful state is a
# parameter's TensorType. The function body is a no-op return so the
# verifier checks only the param types.
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
    """Build a 1-function program whose single parameter has the given type."""
    span = _span()
    var = ir.Var("x", tensor_type, span)
    func = ir.Function("f", [var], [], _empty_body(), span)
    return ir.Program([func], "p", span)


def _verify(program, *, require_materialized: bool = False):
    return _passes.verify_tensor_view_canonical(program, require_materialized)


# ============================================================================
# Bare TensorType (no view) — implicitly canonical
# ============================================================================


def test_bare_tensor_passes_weak():
    t = ir.TensorType(_shape(8, 16), DataType.FP32)
    diags = _verify(_program_with_param_type(t))
    assert diags == [], [d.message for d in diags]


def test_bare_tensor_passes_strict():
    # In strict mode the verifier still passes for bare tensors — they are
    # treated as implicitly ND-packed and the materialization pass will fill
    # the explicit stride. Strict mode only rejects ``view.has_value() &&
    # stride.empty()``, not the absence of a view altogether.
    t = ir.TensorType(_shape(8, 16), DataType.FP32)
    diags = _verify(_program_with_param_type(t), require_materialized=True)
    assert diags == [], [d.message for d in diags]


# ============================================================================
# Explicit canonical views — both modes pass
# ============================================================================


def test_packed_nd_passes():
    view = ir.TensorView(_stride(16, 1), ir.TensorLayout.ND)
    t = ir.TensorType(_shape(8, 16), DataType.FP32, None, view)
    assert _verify(_program_with_param_type(t)) == []
    assert _verify(_program_with_param_type(t), require_materialized=True) == []


def test_packed_dn_passes():
    view = ir.TensorView(_stride(1, 4), ir.TensorLayout.DN)
    t = ir.TensorType(_shape(4, 8), DataType.FP32, None, view)
    assert _verify(_program_with_param_type(t)) == []
    assert _verify(_program_with_param_type(t), require_materialized=True) == []


def test_strided_nd_inherited_subview_passes():
    # Parent ND [8, 16] -> sub [4, 8] inherits parent's [16, 1] stride.
    view = ir.TensorView(_stride(16, 1), ir.TensorLayout.ND)
    t = ir.TensorType(_shape(4, 8), DataType.FP32, None, view)
    assert _verify(_program_with_param_type(t)) == []


def test_strided_dn_inherited_subview_passes():
    # Parent DN [4, 8] (stride [1, 4]) -> sub [2, 4] inherits [1, 4].
    view = ir.TensorView(_stride(1, 4), ir.TensorLayout.DN)
    t = ir.TensorType(_shape(2, 4), DataType.FP32, None, view)
    assert _verify(_program_with_param_type(t)) == []


# ============================================================================
# Empty stride — weak vs strict mode boundary
# ============================================================================


def test_empty_stride_passes_weak():
    # Layout-tagged but stride not yet materialized — pre-MaterializeTensorStrides.
    view = ir.TensorView([], ir.TensorLayout.DN)
    t = ir.TensorType(_shape(4, 8), DataType.FP32, None, view)
    assert _verify(_program_with_param_type(t)) == []


def test_empty_stride_fails_strict():
    view = ir.TensorView([], ir.TensorLayout.DN)
    t = ir.TensorType(_shape(4, 8), DataType.FP32, None, view)
    diags = _verify(_program_with_param_type(t), require_materialized=True)
    assert len(diags) == 1
    assert "stride is empty" in diags[0].message
    assert "MaterializeTensorStrides" in diags[0].message


# ============================================================================
# NZ on TensorType — always rejected
# ============================================================================


def test_nz_layout_rejected_weak():
    view = ir.TensorView(_stride(16, 1), ir.TensorLayout.NZ)
    t = ir.TensorType(_shape(8, 16), DataType.FP32, None, view)
    diags = _verify(_program_with_param_type(t))
    assert len(diags) == 1
    assert "NZ" in diags[0].message
    assert "tile-only" in diags[0].message


def test_nz_layout_rejected_strict():
    # NZ is rejected even when stride is empty — layout family is wrong.
    view = ir.TensorView([], ir.TensorLayout.NZ)
    t = ir.TensorType(_shape(8, 16), DataType.FP32, None, view)
    diags = _verify(_program_with_param_type(t), require_materialized=True)
    assert any("NZ" in d.message for d in diags)


# ============================================================================
# Non-canonical strides — caught with helpful diagnostic
# ============================================================================


def test_layout_tag_mismatch_caught():
    # Stride [1, 4] is DN-shaped, but layout tag says ND.
    view = ir.TensorView(_stride(1, 4), ir.TensorLayout.ND)
    t = ir.TensorType(_shape(4, 8), DataType.FP32, None, view)
    diags = _verify(_program_with_param_type(t))
    assert len(diags) == 1
    assert "innermost" in diags[0].message


def test_too_small_outer_stride_caught():
    # ND with shape [4, 8]: packed stride [8, 1]; [4, 1] is too small.
    view = ir.TensorView(_stride(4, 1), ir.TensorLayout.ND)
    t = ir.TensorType(_shape(4, 8), DataType.FP32, None, view)
    diags = _verify(_program_with_param_type(t))
    assert len(diags) == 1
    assert "smaller than packed" in diags[0].message


def test_diagnostic_includes_function_name():
    view = ir.TensorView(_stride(1, 4), ir.TensorLayout.ND)
    t = ir.TensorType(_shape(4, 8), DataType.FP32, None, view)
    diags = _verify(_program_with_param_type(t))
    assert len(diags) == 1
    assert "in function 'f'" in diags[0].message
    assert diags[0].rule_name == "TensorViewCanonical"


# ============================================================================
# Symbolic strides — RFC Open Q2 (relaxed_symbolic semantics)
# ============================================================================


def test_symbolic_dn_relaxed_passes():
    """[K_sym, N_sym] DN with stride [1, K_sym]: trailing-stride check is
    symbolic. Verifier is in relaxed_symbolic mode and accepts."""
    K = ir.Var("K", ir.ScalarType(DataType.INDEX), _span())
    N = ir.Var("N", ir.ScalarType(DataType.INDEX), _span())
    view = ir.TensorView([_const(1), K], ir.TensorLayout.DN)
    t = ir.TensorType([K, N], DataType.FP32, None, view)
    assert _verify(_program_with_param_type(t)) == []


# ============================================================================
# Registry path — verify_or_throw using the IRProperty
# ============================================================================


def test_registry_returns_strict_verifier():
    """The registry's TensorViewCanonical entry uses strict mode (RFC #1300
    §2.4 — codegen-entry contract). MaterializeTensorStrides produces this
    property, so the auto-verify after it enforces explicit stride. Empty
    stride on an explicit TensorView is rejected (the state
    MaterializeTensorStrides is responsible for eliminating)."""
    view = ir.TensorView([], ir.TensorLayout.DN)
    t = ir.TensorType(_shape(4, 8), DataType.FP32, None, view)
    program = _program_with_param_type(t)

    props = _passes.IRPropertySet()
    props.insert(_passes.IRProperty.TensorViewCanonical)
    diags = _passes.PropertyVerifierRegistry.verify(props, program)
    assert len(diags) >= 1
    assert any("stride is empty" in d.message for d in diags), (
        f"expected 'stride is empty' diagnostic, got: {[d.message for d in diags]}"
    )


def test_registry_accepts_bare_tensor_type():
    """Bare TensorTypes (``!view.has_value()``) are implicitly ND-packed and
    accepted by both weak and strict modes — only ``view.has_value() &&
    stride.empty()`` is flagged."""
    t = ir.TensorType(_shape(4, 8), DataType.FP32)
    program = _program_with_param_type(t)

    props = _passes.IRPropertySet()
    props.insert(_passes.IRProperty.TensorViewCanonical)
    diags = _passes.PropertyVerifierRegistry.verify(props, program)
    assert diags == []


def test_registry_catches_layout_mismatch():
    """Non-canonical IR is caught by the registry-based path too."""
    view = ir.TensorView(_stride(1, 4), ir.TensorLayout.ND)
    t = ir.TensorType(_shape(4, 8), DataType.FP32, None, view)
    program = _program_with_param_type(t)

    props = _passes.IRPropertySet()
    props.insert(_passes.IRProperty.TensorViewCanonical)
    diags = _passes.PropertyVerifierRegistry.verify(props, program)
    assert len(diags) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

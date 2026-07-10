# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the standing ``valid_shape <= shape`` invariant .

The invariant is enforced by the ``TypeChecked`` property verifier (the
``TypeCheck`` rule): for any TileType/TensorType carrying an explicit,
non-empty ``valid_shape``,

- ``rank(valid_shape) == rank(shape)`` (or ``valid_shape`` is empty), and
- ``0 <= valid_shape[i] <= shape[i]`` for every dim where **both** the valid
  extent and the physical extent are compile-time constants.

Symbolic (dynamic) valid extents are a supported feature and are skipped, not
rejected. The invariant covers both TileView and TensorView, and is placed in
the always-on ``TypeChecked`` structural property so a malformed ``valid_shape``
is caught before/after every pass rather than only on demand.
"""

import pytest
from pypto import DataType, ir
from pypto.ir import IRBuilder
from pypto.ir.op import tile as tile_ops
from pypto.pypto_core import passes as _passes


def _span():
    return ir.Span.unknown()


def _const(value: int):
    return ir.ConstInt(value, DataType.INDEX, _span())


def _program_with_param_type(t) -> ir.Program:
    """Build a 1-function program whose single parameter has the given type."""
    span = _span()
    var = ir.Var("x", t, span)
    func = ir.Function("f", [var], [], ir.EvalStmt(_const(0), span), span)
    return ir.Program([func], "p", span)


def _verify(program: ir.Program) -> list:
    """Run the TypeChecked verifier and return its diagnostics."""
    props = _passes.IRPropertySet()
    props.insert(_passes.IRProperty.TypeChecked)
    return _passes.PropertyVerifierRegistry.verify(props, program)


def _tile(shape: list[int], valid: list) -> ir.TileType:
    """TileType with an explicit valid_shape (ints stay ConstInt; Var passes through)."""
    valid_exprs = [_const(v) if isinstance(v, int) else v for v in valid]
    view = ir.TileView(valid_shape=valid_exprs)
    return ir.TileType([_const(s) for s in shape], DataType.FP32, None, view, ir.MemorySpace.Vec)


def _tensor(shape: list[int], valid: list) -> ir.TensorType:
    """TensorType with an explicit valid_shape (layout required alongside valid_shape)."""
    valid_exprs = [_const(v) if isinstance(v, int) else v for v in valid]
    view = ir.TensorView(valid_shape=valid_exprs, layout=ir.TensorLayout.ND)
    return ir.TensorType([_const(s) for s in shape], DataType.FP32, None, view)


# ============================================================================
# TileType — the [999, 999] / [128, 128] scenario and its neighbours
# ============================================================================


def test_tile_valid_shape_exceeds_shape_rejected():
    """valid_shape [999, 999] on shape [128, 128] is rejected (B3)."""
    diags = _verify(_program_with_param_type(_tile([128, 128], [999, 999])))
    assert len(diags) >= 1
    assert any("out of bounds" in d.message for d in diags), [d.message for d in diags]
    assert any(d.rule_name == "TypeCheck" for d in diags)


def test_tile_narrower_valid_shape_accepted():
    """A narrower static valid_shape [64, 128] on shape [128, 128] is accepted."""
    assert _verify(_program_with_param_type(_tile([128, 128], [64, 128]))) == []


def test_tile_zero_valid_shape_accepted():
    """valid_shape [0, 0] (the fresh-accumulator sentinel) is accepted: 0 <= 0 <= shape."""
    assert _verify(_program_with_param_type(_tile([128, 128], [0, 0]))) == []


def test_tile_dynamic_valid_shape_accepted():
    """A dynamic (symbolic Var) valid extent is skipped, not rejected."""
    m = ir.Var("m", ir.ScalarType(DataType.INDEX), _span())
    # Even a symbolic value that could exceed the shape at runtime is accepted —
    # the bound is only checked for compile-time-constant dims.
    assert _verify(_program_with_param_type(_tile([128, 128], [m, 128]))) == []


def test_tile_rank_mismatch_rejected():
    """rank(valid_shape) != rank(shape) is rejected."""
    diags = _verify(_program_with_param_type(_tile([128, 128], [64])))
    assert len(diags) >= 1
    assert any("valid_shape rank" in d.message and "does not match shape rank" in d.message for d in diags)


# ============================================================================
# TensorType — same invariant, verifying the check covers both view kinds
# ============================================================================


def test_tensor_valid_shape_exceeds_shape_rejected():
    """valid_shape [999, 999] on a [128, 128] TensorType is rejected."""
    diags = _verify(_program_with_param_type(_tensor([128, 128], [999, 999])))
    assert len(diags) >= 1
    assert any("out of bounds" in d.message and "TensorType" in d.message for d in diags)


def test_tensor_narrower_valid_shape_accepted():
    """A narrower static valid_shape on a TensorType is accepted."""
    assert _verify(_program_with_param_type(_tensor([128, 128], [64, 128]))) == []


def test_tensor_dynamic_valid_shape_accepted():
    """A dynamic valid extent on a TensorType is skipped, not rejected."""
    n = ir.Var("n", ir.ScalarType(DataType.INDEX), _span())
    assert _verify(_program_with_param_type(_tensor([128, 128], [n, n]))) == []


# ============================================================================
# Bare types (no explicit valid_shape) — fully valid, nothing to check
# ============================================================================


def test_bare_tile_accepted():
    """A TileType without a valid_shape is fully valid and passes."""
    t = ir.TileType([_const(128), _const(128)], DataType.FP32, None, None, ir.MemorySpace.Vec)
    assert _verify(_program_with_param_type(t)) == []


def test_bare_tensor_accepted():
    """A bare TensorType (no view) is fully valid and passes."""
    t = ir.TensorType([_const(128), _const(128)], DataType.FP32)
    assert _verify(_program_with_param_type(t)) == []


# ============================================================================
# Body Var path — a malformed valid_shape produced by pl.load is caught too
# ============================================================================


def test_body_var_from_load_rejected():
    """An out-of-bounds ``valid_shapes=`` on ``pl.load`` is caught on the body Var.

    ``tile.load`` sets ``valid_shape`` verbatim without a bound check, so the
    malformed tile only surfaces at the standing invariant. This exercises the
    ``VisitVarLike_``/``VisitExpr_(Call)`` body walk (not just param/return types).
    """
    span = _span()
    ib = IRBuilder()
    with ib.function("cast_incore", type=ir.FunctionType.InCore) as f:
        x = f.param("x", ir.TensorType([_const(128), _const(128)], DataType.FP32))
        out_p = f.param(
            "out", ir.TensorType([_const(128), _const(128)], DataType.FP32), direction=ir.ParamDirection.Out
        )
        f.return_type(ir.TensorType([_const(128), _const(128)], DataType.FP32))
        # 999 > 128 on both dims: valid_shape exceeds the physical tile shape.
        x_tile = ib.let("x_tile", tile_ops.load(x, [0, 0], [128, 128], valid_shapes=[999, 999], span=span))
        out_r = ib.let("out_0", tile_ops.store(x_tile, [0, 0], out_p, span=span))
        ib.return_stmt(out_r)
    program = ir.Program([f.get_result()], "p", span)

    diags = _verify(program)
    assert len(diags) >= 1
    assert any("out of bounds" in d.message for d in diags), [d.message for d in diags]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

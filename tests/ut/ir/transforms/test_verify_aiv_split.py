# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the AivSplitValid property verifier.

The verifier closes the gap left by SplitVectorKernel's split_aiv bypass: the
AUTO path rejects a vector reduce over the split axis inline, but the EXPLICIT
``split_aiv`` path skips that rewrite. A split_aiv AIV/AIC function whose body
reduces over the split axis is a partial-reduction miscompile (each AIV lane
holds only half the tile), so the verifier flags it as an Error.

These tests hand-build minimal split_aiv functions and run the verifier
directly through ``PropertyVerifierRegistry`` (no full pipeline needed).
"""

import pytest
from pypto import DataType, ir, passes
from pypto.ir.op import tile_ops as T

MS = ir.MemorySpace
FP32 = DataType.FP32
_IN = ir.ParamDirection.In
_OUT = ir.ParamDirection.Out


def _tile(shape, mem=MS.Vec):
    return ir.TileType(shape, FP32, None, None, mem)


def _aiv_split_prop_set() -> passes.IRPropertySet:
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.AivSplitValid)
    return props


def _verify(program) -> list:
    return passes.PropertyVerifierRegistry.verify(_aiv_split_prop_set(), program)


def _errors(program) -> list:
    return [d for d in _verify(program) if d.severity == passes.DiagnosticSeverity.Error]


def _make_reduce_func(
    reduce_kind: str,
    *,
    split_mode,
    split_aiv: bool,
    func_type=ir.FunctionType.AIV,
):
    """Build a minimal kernel: load [16, 128] -> reduce -> store.

    ``reduce_kind`` selects the reduce op:
      - ``"col"``  -> tile.col_max (reduces axis 0)
      - ``"row"``  -> tile.row_max (reduces the last axis, dim 1 for a 2D tile)
    """
    span = ir.Span.unknown()
    data = ir.Var("data", _tile([16, 128]), span)

    if reduce_kind == "col":
        reduce_call = T.col_max(data, span)
    elif reduce_kind == "row":
        tmp = ir.Var("tmp", _tile([16, 128]), span)
        reduce_call = T.row_max(data, tmp, span)
    else:
        raise ValueError(f"unknown reduce_kind: {reduce_kind}")

    reduced = ir.Var("reduced", reduce_call.type, span)
    out_0 = ir.Var("out_0", ir.TensorType([16, 128], FP32), span)
    store = T.store(data, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(reduced, reduce_call, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        span,
    )
    attrs = {"split": split_mode}
    if split_aiv:
        attrs["split_aiv"] = True
    func = ir.Function(
        "split_aiv",
        [(data, _IN), (out_0, _OUT)],
        [out_0.type],
        body,
        span,
        func_type,
        attrs=attrs,
    )
    return ir.Program([func], "test_aiv_split", span)


# ---------------------------------------------------------------------------
# Reduce ON the split axis -> Error
# ---------------------------------------------------------------------------


def test_up_down_col_reduce_on_split_axis_rejected():
    """UP_DOWN splits dim 0; col_max reduces dim 0 -> partial reduction -> Error."""
    program = _make_reduce_func("col", split_mode=ir.SplitMode.UP_DOWN, split_aiv=True)
    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "reduces over the split axis" in errors[0].message
    assert "tile.col_max" in errors[0].message


def test_left_right_row_reduce_on_split_axis_rejected():
    """LEFT_RIGHT splits dim 1; row_max reduces the last axis (1) -> Error."""
    program = _make_reduce_func("row", split_mode=ir.SplitMode.LEFT_RIGHT, split_aiv=True)
    errors = _errors(program)
    assert len(errors) == 1
    assert "tile.row_max" in errors[0].message


def test_aic_function_also_checked():
    """The gate covers AIC functions too, not just AIV."""
    program = _make_reduce_func(
        "col", split_mode=ir.SplitMode.UP_DOWN, split_aiv=True, func_type=ir.FunctionType.AIC
    )
    assert len(_errors(program)) == 1


# ---------------------------------------------------------------------------
# Reduce on the NON-split axis -> no false positive
# ---------------------------------------------------------------------------


def test_up_down_row_reduce_off_split_axis_passes():
    """UP_DOWN splits dim 0; row_max reduces dim 1 (flash-attn row_max) -> OK."""
    program = _make_reduce_func("row", split_mode=ir.SplitMode.UP_DOWN, split_aiv=True)
    assert _errors(program) == []


def test_left_right_col_reduce_off_split_axis_passes():
    """LEFT_RIGHT splits dim 1; col_max reduces dim 0 -> OK."""
    program = _make_reduce_func("col", split_mode=ir.SplitMode.LEFT_RIGHT, split_aiv=True)
    assert _errors(program) == []


# ---------------------------------------------------------------------------
# Not applicable -> verifier does nothing
# ---------------------------------------------------------------------------


def test_non_split_aiv_function_not_checked():
    """A function without the split_aiv marker is out of scope (verifier is a no-op)."""
    program = _make_reduce_func("col", split_mode=ir.SplitMode.UP_DOWN, split_aiv=False)
    assert _verify(program) == []


def test_split_aiv_without_split_mode_not_checked():
    """split_aiv marker but SplitMode.NONE -> gate requires a real split mode."""
    program = _make_reduce_func("col", split_mode=ir.SplitMode.NONE, split_aiv=True)
    assert _verify(program) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

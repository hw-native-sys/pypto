# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the structural AivSplitValid property verifier.

The verifier is keyed on the first-class ``SplitAivScopeStmt`` region (live
between OutlineIncoreScopes and LowerAutoVectorSplit). Per region it checks:
  (a) no cube compute inside a region (each AIV lane holds only half the tile);
  (b) no AIV reduce over the split axis inside a region (partial reduction);
  (c) the ``tile.aiv_shard`` / ``tile.aic_gather`` boundary ops appear only
      inside a region.
Full-width vector compute *outside* a region is legal (multi-mode goal) and is
deliberately NOT flagged.

These tests hand-build minimal functions and run the verifier directly through
``PropertyVerifierRegistry`` (no full pipeline needed).
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


def _program(body: ir.Stmt) -> ir.Program:
    """Wrap a body statement in a minimal AIV function + program."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tile([16, 128]), span)
    out_0 = ir.Var("out_0", ir.TensorType([16, 128], FP32), span)
    func = ir.Function(
        "split_aiv",
        [(data, _IN), (out_0, _OUT)],
        [out_0.type],
        body,
        span,
        ir.FunctionType.AIV,
    )
    return ir.Program([func], "test_aiv_split", span)


def _region(split_mode, inner_stmts: list[ir.Stmt]) -> ir.SplitAivScopeStmt:
    span = ir.Span.unknown()
    return ir.SplitAivScopeStmt(split=split_mode, body=ir.SeqStmts(inner_stmts, span), span=span)


# ---------------------------------------------------------------------------
# (a) Cube compute inside a region -> Error
# ---------------------------------------------------------------------------


def test_cube_in_region_fails():
    """A cube op (tile.matmul, Acc output) inside a region cannot be vector-split."""
    span = ir.Span.unknown()
    lhs = ir.Var("lhs", _tile([16, 128], MS.Left), span)
    rhs = ir.Var("rhs", _tile([128, 64], MS.Right), span)
    mm = T.matmul(lhs, rhs, span)
    res = ir.Var("res", mm.type, span)
    region = _region(ir.SplitMode.UP_DOWN, [ir.AssignStmt(res, mm, span)])
    program = _program(ir.SeqStmts([region], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "cube op" in errors[0].message
    assert "tile.matmul" in errors[0].message


# ---------------------------------------------------------------------------
# (b) Reduce over the split axis inside a region -> Error
# ---------------------------------------------------------------------------


def test_reduce_on_split_axis_fails():
    """UP_DOWN splits dim 0; tile.col_max reduces dim 0 inside a region -> Error."""
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128]), span)
    cm = T.col_max(data, span)
    res = ir.Var("res", cm.type, span)
    region = _region(ir.SplitMode.UP_DOWN, [ir.AssignStmt(res, cm, span)])
    program = _program(ir.SeqStmts([region], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "reduces over the split axis" in errors[0].message
    assert "tile.col_max" in errors[0].message


# ---------------------------------------------------------------------------
# (c) Boundary op outside any region -> Error
# ---------------------------------------------------------------------------


def test_boundary_outside_region_fails():
    """tile.aiv_shard at top level (no enclosing region) -> Error."""
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128]), span)
    shard = T.aiv_shard(data, split=int(ir.SplitMode.UP_DOWN.value), span=span)
    res = ir.Var("res", shard.type, span)
    program = _program(ir.SeqStmts([ir.AssignStmt(res, shard, span)], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "tile.aiv_shard" in errors[0].message
    assert "must appear inside a pl.split_aiv region" in errors[0].message


# ---------------------------------------------------------------------------
# Valid region -> no error
# ---------------------------------------------------------------------------


def test_valid_region_passes():
    """A region with vector compute + a boundary op inside it is valid."""
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128]), span)
    shard = T.aiv_shard(data, split=int(ir.SplitMode.UP_DOWN.value), span=span)
    sharded = ir.Var("sharded", shard.type, span)
    add = T.add(sharded, sharded, span)
    res = ir.Var("res", add.type, span)
    region = _region(
        ir.SplitMode.UP_DOWN,
        [ir.AssignStmt(sharded, shard, span), ir.AssignStmt(res, add, span)],
    )
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []


def test_fullwidth_vector_outside_region_passes():
    """Full-width vector compute outside any region is legal (multi-mode goal)."""
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128]), span)
    add = T.add(data, data, span)
    res = ir.Var("res", add.type, span)
    program = _program(ir.SeqStmts([ir.AssignStmt(res, add, span)], span))

    assert _errors(program) == []


# ---------------------------------------------------------------------------
# Task-parallel (NONE) regions: no split axis. Boundary ops are rejected; the
# split-axis rules (cube / reduce-on-split-axis) do NOT apply (both lanes run
# the full body).
# ---------------------------------------------------------------------------


def test_boundary_in_none_region_fails():
    """tile.aiv_shard inside a NONE region -> Error (no split axis to shard)."""
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128]), span)
    shard = T.aiv_shard(data, split=int(ir.SplitMode.UP_DOWN.value), span=span)
    res = ir.Var("res", shard.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, shard, span)])
    program = _program(ir.SeqStmts([region], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "tile.aiv_shard" in errors[0].message
    assert "task-parallel" in errors[0].message


def test_reduce_in_none_region_passes():
    """A reduce that would collapse dim 0 is fine in a NONE region: there is no
    split axis, so it is a full (not partial) reduction on both lanes."""
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128]), span)
    cm = T.col_max(data, span)
    res = ir.Var("res", cm.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, cm, span)])
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []


def test_cube_in_none_region_passes():
    """A cube op is allowed in a NONE region: nothing is halved, and the op routes
    to AIC after ExpandMixedKernel — both AIV lanes run the full body."""
    span = ir.Span.unknown()
    lhs = ir.Var("lhs", _tile([16, 128], MS.Left), span)
    rhs = ir.Var("rhs", _tile([128, 64], MS.Right), span)
    mm = T.matmul(lhs, rhs, span)
    res = ir.Var("res", mm.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, mm, span)])
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

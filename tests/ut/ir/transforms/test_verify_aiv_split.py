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
      inside a region, and never inside a task-parallel ``mode=NONE`` one;
  (d) the boundary memory contract — ``tile.aiv_shard`` is ``Acc -> Vec`` and
      ``tile.aic_gather`` is ``Vec -> Mat``, since both ops *are* the cross-core
      transfer, so the operand must sit on the producing lane and the result on
      the consuming one.
Full-width vector compute *outside* a region is legal (multi-mode goal) and is
deliberately NOT flagged.

These tests hand-build minimal functions and run the verifier directly through
``PropertyVerifierRegistry`` (no full pipeline needed).
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir.op import tile_ops as T

MS = ir.MemorySpace
FP32 = DataType.FP32
_IN = ir.ParamDirection.In
_OUT = ir.ParamDirection.Out


def _tile(shape, mem=MS.Vec):
    return ir.TileType(shape, FP32, None, None, mem)


def _tensor(shape):
    return ir.TensorType(shape, FP32)


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
    """A region with vector compute + a boundary op inside it is valid.

    The shard operand is Acc: aiv_shard carries a CUBE-produced value (a matmul
    result in L0C) across to the vector lane, so Acc is the only valid operand
    space. Sharding a Vec operand is rejected by check (d) below.
    """
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128], mem=MS.Acc), span)
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


# ---------------------------------------------------------------------------
# (d) Boundary memory contract: aiv_shard is Acc -> Vec, aic_gather is Vec -> Mat
# ---------------------------------------------------------------------------


def test_shard_vector_produced_operand_fails():
    """aiv_shard of a Vec (vector-produced) operand -> Error.

    Regression guard: ExpandMixedKernel routes the shard's tpush onto the AIC
    lane by op name, but a Vec operand's producer stays on AIV. The cube half
    then references a value it never defines, which InitMemRef turns into an
    orphan Mem.Vec allocation and PTO codegen finally rejects with
    "no MLIR mapping for MemRef base". Catch it here instead.
    """
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128], mem=MS.Vec), span)
    shard = T.aiv_shard(data, split=int(ir.SplitMode.UP_DOWN.value), span=span)
    res = ir.Var("res", shard.type, span)
    region = _region(ir.SplitMode.UP_DOWN, [ir.AssignStmt(res, shard, span)])
    program = _program(ir.SeqStmts([region], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "tile.aiv_shard" in errors[0].message
    assert "operand is in Vec" in errors[0].message
    assert "requires Acc" in errors[0].message


def test_gather_cube_produced_operand_fails():
    """aic_gather of an Acc (cube-produced) operand -> Error (the mirror case)."""
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128], mem=MS.Acc), span)
    gather = T.aic_gather(data, split=int(ir.SplitMode.UP_DOWN.value), span=span)
    res = ir.Var("res", gather.type, span)
    region = _region(ir.SplitMode.UP_DOWN, [ir.AssignStmt(res, gather, span)])
    program = _program(ir.SeqStmts([region], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "tile.aic_gather" in errors[0].message
    assert "operand is in Acc" in errors[0].message
    assert "requires Vec" in errors[0].message


def test_gather_vector_produced_operand_passes():
    """aic_gather of a Vec operand is the valid direction -> no error."""
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128], mem=MS.Vec), span)
    gather = T.aic_gather(data, split=int(ir.SplitMode.UP_DOWN.value), span=span)
    res = ir.Var("res", gather.type, span)
    region = _region(ir.SplitMode.UP_DOWN, [ir.AssignStmt(res, gather, span)])
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []


def test_boundary_result_in_wrong_memory_fails():
    """A boundary result stamped with the PRODUCER-side space -> Error.

    The declared type describes the CONSUMING lane. This is the shape the
    tensor->tile converter used to emit for aic_gather (Vec, the producer side)
    before it read the space from the op's set_output_memory declaration.
    """
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128], mem=MS.Vec), span)
    gather = T.aic_gather(data, split=int(ir.SplitMode.UP_DOWN.value), span=span)
    assert isinstance(gather.type, ir.TileType)
    # Re-type the call result to the producer-side Vec instead of the Mat the op declares.
    wrong = ir.TileType(gather.type.shape, FP32, None, gather.type.tile_view, MS.Vec)
    mistyped = ir.Call(gather.op, gather.args, gather.kwargs, wrong, span)
    res = ir.Var("res", wrong, span)
    region = _region(ir.SplitMode.UP_DOWN, [ir.AssignStmt(res, mistyped, span)])
    program = _program(ir.SeqStmts([region], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "tile.aic_gather" in errors[0].message
    assert "result is in Vec" in errors[0].message
    assert "must be Mat" in errors[0].message


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


# ---------------------------------------------------------------------------
# Tensor-form boundary ops (tensor.aiv_shard / tensor.aic_gather).
#
# These are the @pl.jit / pl.spmd author-facing pl.aiv_shard(tensor) /
# pl.aic_gather(tensor) form: still tensor.* in the window between
# OutlineIncoreScopes (which produces AivSplitValid) and ConvertTensorToTileOps
# (which lowers them 1:1 to tile.aiv_shard / tile.aic_gather). The verifier must
# recognize them as the SAME AIV-split boundary as the tile.* ops: valid inside
# a data-parallel (UP_DOWN / LEFT_RIGHT) region, rejected in a task-parallel
# (NONE) region, and rejected at top level. Mirrors the tile-form matrix above.
#
# The split attr matches the region's SplitMode value: UP_DOWN == 1 (axis 0),
# LEFT_RIGHT == 2 (axis 1). The verifier keys the boundary rejection on the
# region's split MODE, so the op's own split value is what the tile-form tests
# also use (1) — the region node is the source of truth.
# ---------------------------------------------------------------------------


def _tensor_shard(shape, split, span):
    """Hand-build a ``tensor.aiv_shard`` Call over a fresh rank-2 Tensor Var."""
    t = ir.Var("t", _tensor(shape), span)
    return ir.create_op_call("tensor.aiv_shard", [t], {"split": split}, span)


def _tensor_gather(shape, split, span):
    """Hand-build a ``tensor.aic_gather`` Call over a fresh rank-2 Tensor Var."""
    t = ir.Var("t", _tensor(shape), span)
    return ir.create_op_call("tensor.aic_gather", [t], {"split": split}, span)


def _has_tensor_call(program, op_name) -> bool:
    """Whether any ``ir.Call`` to ``op_name`` is reachable in ``program``."""
    found = []

    def walk(n):
        if isinstance(n, ir.Call) and isinstance(n.op, ir.Op) and n.op.name == op_name:
            found.append(n)
        if isinstance(n, ir.SeqStmts):
            for s in n.stmts:
                walk(s)
            return
        if isinstance(n, ir.AssignStmt):
            walk(n.value)
        body = getattr(n, "body", None)
        if body is not None:
            walk(body)

    for func in program.functions.values():
        if func.body is not None:
            walk(func.body)
    return bool(found)


# --- Accepted: data-parallel regions (UP_DOWN / LEFT_RIGHT) -----------------


def test_tensor_shard_in_up_down_region_passes():
    """tensor.aiv_shard (split axis 0) inside an UP_DOWN region is valid."""
    span = ir.Span.unknown()
    shard = _tensor_shard([16, 128], int(ir.SplitMode.UP_DOWN.value), span)
    res = ir.Var("res", shard.type, span)
    region = _region(ir.SplitMode.UP_DOWN, [ir.AssignStmt(res, shard, span)])
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []


def test_tensor_gather_in_up_down_region_passes():
    """tensor.aic_gather (split axis 0) inside an UP_DOWN region is valid."""
    span = ir.Span.unknown()
    gather = _tensor_gather([8, 128], int(ir.SplitMode.UP_DOWN.value), span)
    res = ir.Var("res", gather.type, span)
    region = _region(ir.SplitMode.UP_DOWN, [ir.AssignStmt(res, gather, span)])
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []


def test_tensor_shard_in_left_right_region_passes():
    """tensor.aiv_shard (split axis 1) inside a LEFT_RIGHT region is valid."""
    span = ir.Span.unknown()
    shard = _tensor_shard([16, 128], int(ir.SplitMode.LEFT_RIGHT.value), span)
    res = ir.Var("res", shard.type, span)
    region = _region(ir.SplitMode.LEFT_RIGHT, [ir.AssignStmt(res, shard, span)])
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []


def test_tensor_gather_in_left_right_region_passes():
    """tensor.aic_gather (split axis 1) inside a LEFT_RIGHT region is valid."""
    span = ir.Span.unknown()
    gather = _tensor_gather([16, 64], int(ir.SplitMode.LEFT_RIGHT.value), span)
    res = ir.Var("res", gather.type, span)
    region = _region(ir.SplitMode.LEFT_RIGHT, [ir.AssignStmt(res, gather, span)])
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []


# --- Rejected: task-parallel (NONE) region — no split axis to shard/gather --


def test_tensor_shard_in_none_region_fails():
    """tensor.aiv_shard inside a NONE region -> Error (no split axis)."""
    span = ir.Span.unknown()
    shard = _tensor_shard([16, 128], int(ir.SplitMode.UP_DOWN.value), span)
    res = ir.Var("res", shard.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, shard, span)])
    program = _program(ir.SeqStmts([region], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "tensor.aiv_shard" in errors[0].message
    assert "task-parallel" in errors[0].message


def test_tensor_gather_in_none_region_fails():
    """tensor.aic_gather inside a NONE region -> Error (no split axis)."""
    span = ir.Span.unknown()
    gather = _tensor_gather([8, 128], int(ir.SplitMode.UP_DOWN.value), span)
    res = ir.Var("res", gather.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, gather, span)])
    program = _program(ir.SeqStmts([region], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "tensor.aic_gather" in errors[0].message
    assert "task-parallel" in errors[0].message


# --- Rejected: boundary op escaped its region (top level) -------------------


def test_tensor_shard_outside_region_fails():
    """tensor.aiv_shard at top level (no enclosing region) -> Error."""
    span = ir.Span.unknown()
    shard = _tensor_shard([16, 128], int(ir.SplitMode.UP_DOWN.value), span)
    res = ir.Var("res", shard.type, span)
    program = _program(ir.SeqStmts([ir.AssignStmt(res, shard, span)], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "tensor.aiv_shard" in errors[0].message
    assert "must appear inside a pl.split_aiv region" in errors[0].message


def test_tensor_gather_outside_region_fails():
    """tensor.aic_gather at top level (no enclosing region) -> Error."""
    span = ir.Span.unknown()
    gather = _tensor_gather([8, 128], int(ir.SplitMode.UP_DOWN.value), span)
    res = ir.Var("res", gather.type, span)
    program = _program(ir.SeqStmts([ir.AssignStmt(res, gather, span)], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "tensor.aic_gather" in errors[0].message
    assert "must appear inside a pl.split_aiv region" in errors[0].message


# ---------------------------------------------------------------------------
# End-to-end DSL path: the author writes pl.aiv_shard(tensor) /
# pl.aic_gather(tensor) inside a `for aiv_id in pl.split_aiv(mode=...)` region.
# The parser emits the tensor.* boundary op (region-only, split inherited from
# the region mode), and the verifier accepts the resulting region. Only the
# data-parallel accept path is expressible via the DSL: the parser blocks
# pl.aiv_shard(tensor) in a NONE region (split == 0 fails the rank-2 deducer's
# split gate) and outside any region (no mode to inherit), so the NONE / top
# level rejections are covered by the hand-built matrix above.
# ---------------------------------------------------------------------------


def test_dsl_tensor_shard_up_down_region_passes():
    """DSL pl.aiv_shard(tensor) in an UP_DOWN region -> tensor.aiv_shard, accepted."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):  # noqa: B007
                h = pl.aiv_shard(a)  # noqa: F841
            return out

    assert _has_tensor_call(Prog, ir.get_op("tensor.aiv_shard").name)
    assert _errors(Prog) == []


def test_dsl_tensor_gather_left_right_region_passes():
    """DSL pl.aic_gather(tensor) in a LEFT_RIGHT region -> tensor.aic_gather, accepted."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[128, 512], pl.FP32],
            out: pl.Out[pl.Tensor[[128, 512], pl.FP32]],
        ) -> pl.Tensor[[128, 512], pl.FP32]:
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):  # noqa: B007
                g = pl.aic_gather(a)  # noqa: F841
            return out

    assert _has_tensor_call(Prog, ir.get_op("tensor.aic_gather").name)
    assert _errors(Prog) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

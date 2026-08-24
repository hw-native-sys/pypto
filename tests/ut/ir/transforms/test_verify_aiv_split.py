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
  (a) no cube compute inside ANY region — a data-parallel region cannot
      vector-split it (each AIV lane holds only half the tile) and every region,
      task-parallel included, *is* the AIV lane's body;
  (b) no AIV reduce over the split axis inside a data-parallel region (partial
      reduction). Gated on the data-parallel modes, unlike (a): a ``mode=NONE``
      region has no split axis to collapse;
  (c) the ``tile.aiv_shard`` / ``tile.aic_gather`` boundary ops appear only
      inside a region. Inside a task-parallel ``mode=NONE`` one they are
      ACCEPTED: with no split axis they still mean "this value crosses the
      AIC/AIV boundary", and their ``split=0`` deduction preserves the shape;
  (d) the boundary memory contract — ``tile.aiv_shard`` is ``Acc -> Vec`` and
      ``tile.aic_gather`` is ``Vec -> Mat``, since both ops *are* the cross-core
      transfer, so the operand must sit on the producing lane and the result on
      the consuming one. Mode-independent, so it runs in NONE regions too.

Plus the MANUAL-MODE rules, gated on a whole-function fact rather than on the
node:
  (e) in a function that opens at least one region, the regions are
      authoritative for vector placement, so a VECTOR-affine op *outside* every
      region is rejected. Three carve-outs: ``tile.load`` / ``tile.store`` are
      the compiler's own out-of-region output (ConvertTensorToTileOps hoists
      them out of the region holding the compute they feed), and an op whose
      lane is STATED rather than inferred — via ``set_core_affinity`` or via a
      ``core_type`` kwarg on a barrier / cross-core event — was never inferred
      in the first place, so a region cannot disambiguate it. A function with
      NO region is untouched — the same op is accepted there;
  (f) a value defined inside a region and read on the CUBE lane outside it must
      be a ``tile.aic_gather`` result (V->C);
  (g) a cube-produced value defined outside every region and read on the VECTOR
      lane inside one must arrive through a ``tile.aiv_shard`` (C->V).
      (f)/(g) share (e)'s ``tile.load`` / ``tile.store`` carve-out, on both the
      definer and the consumer side.
  (i) no ``pl.aiv_shard`` / ``pl.aic_gather`` result crosses a loop back-edge —
      checked at BOTH ends of the carry (the iter_arg's init and the value
      yielded back into it), since the loop type-checks either way;
  (j) a boundary result is consumed in the region that produced it. Gated on the
      data-parallel modes: only they halve the region's tiles and localize its
      store offsets, so only there does an already-per-lane value get halved and
      offset twice. A ``mode=NONE`` region rewrites nothing and MAY consume a
      boundary result from elsewhere — the cross-core comm-kernel shape.

Lane-sharding of once-only side effects (``pld.system.notify``) is deliberately
NOT a check — see the comment above
``test_wellformed_manual_mode_mixed_body_passes``.

These tests hand-build minimal functions and run the verifier directly through
``PropertyVerifierRegistry`` (no full pipeline needed). Every rule is paired
with its negative: a check that fires on everything is as broken as one that
never fires.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir.op import tile_ops as T

MS = ir.MemorySpace
FP32 = DataType.FP32
INT32 = DataType.INT32
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


def _program(body: ir.Stmt, func_type=ir.FunctionType.AIV) -> ir.Program:
    """Wrap a body statement in a minimal function + program.

    An ``InCore`` function is stamped ``split_aiv`` to match what
    ``OutlineIncoreScopes`` puts on a function it mints from a region-bearing
    CORE_GROUP scope. Check (h) reads that attr as the region's provenance — a
    region in an InCore function is legal exactly when the outliner produced the
    function — so hand-built IR standing in for a post-outlining kernel has to
    carry it, or it models a shape the pipeline never emits.
    """
    span = ir.Span.unknown()
    data = ir.Var("data", _tile([16, 128]), span)
    out_0 = ir.Var("out_0", ir.TensorType([16, 128], FP32), span)
    func = ir.Function(
        "split_aiv",
        [(data, _IN), (out_0, _OUT)],
        [out_0.type],
        body,
        span,
        func_type,
        attrs={"split_aiv": True} if func_type == ir.FunctionType.InCore else None,
    )
    return ir.Program([func], "test_aiv_split", span)


def _region(split_mode, inner_stmts: list[ir.Stmt]) -> ir.SplitAivScopeStmt:
    span = ir.Span.unknown()
    return ir.SplitAivScopeStmt(split=split_mode, body=ir.SeqStmts(inner_stmts, span), span=span)


def _notify(span, notify_op: int = 0) -> ir.Call:
    """Hand-build a ``pld.system.notify`` Call — today's only no-duplicate op.

    Signature: ``notify(target: DistributedTensor, peer: Scalar, offsets: tuple,
    value: Scalar, *, op: int)``. ``op=0`` is ``NotifyOp.AtomicAdd``, ``op=1``
    is ``NotifyOp.Set``; the verifier treats them alike.
    """
    sig = ir.Var("sig", ir.DistributedTensorType([4, 4], INT32), span)
    peer = ir.Var("peer", ir.ScalarType(INT32), span)
    offsets = _zero_offsets(span)
    value = ir.ConstInt(1, INT32, span)
    return ir.create_op_call("pld.system.notify", [sig, peer, offsets, value], {"op": notify_op}, span)


def _zero_offsets(span) -> ir.MakeTuple:
    """A rank-2 ``[0, 0]`` offset tuple, the shape both signal ops expect."""
    zero = ir.ConstInt(0, DataType.INDEX, span)
    return ir.MakeTuple([zero, zero], span)


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
    # The data-parallel arm names BOTH reasons: the region is AIV work, and each
    # lane holds only half the tile.
    assert "region body is AIV work" in errors[0].message
    assert "half the tile" in errors[0].message


def test_cube_outside_region_passes():
    """The negative for (a): the same cube op at top level is AIC work, and legal.

    Manual mode makes the region authoritative for VECTOR placement only — cube
    work belongs outside, which is exactly where check (a) tells the author to
    move it.
    """
    span = ir.Span.unknown()
    lhs = ir.Var("lhs", _tile([16, 128], MS.Left), span)
    rhs = ir.Var("rhs", _tile([128, 64], MS.Right), span)
    mm = T.matmul(lhs, rhs, span)
    res = ir.Var("res", mm.type, span)
    data = ir.Var("d", _tile([16, 128]), span)
    add = T.add(data, data, span)
    inner = ir.Var("inner", add.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(inner, add, span)])
    program = _program(ir.SeqStmts([ir.AssignStmt(res, mm, span), region], span))

    assert _errors(program) == []


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


# ---------------------------------------------------------------------------
# (e) Manual mode: a function that opens at least one region makes the regions
# authoritative for vector placement, so VECTOR-affine compute outside every
# region is rejected. A function with NO region keeps today's behaviour.
# ---------------------------------------------------------------------------


def _vector_add(span):
    """A plain VECTOR-affine call (Vec in, Vec out) and the Var it binds."""
    data = ir.Var("d", _tile([16, 128]), span)
    add = T.add(data, data, span)
    return add, ir.Var("res", add.type, span)


def test_fullwidth_vector_outside_region_passes_without_any_region():
    """The negative for (e): with NO region in the function, nothing opted into
    manual mode, so full-width vector compute at top level stays legal."""
    span = ir.Span.unknown()
    add, res = _vector_add(span)
    program = _program(ir.SeqStmts([ir.AssignStmt(res, add, span)], span))

    assert _errors(program) == []


def test_fullwidth_vector_outside_region_fails_when_a_region_exists():
    """(e): the SAME op, in a function that also opens a region -> Error.

    The only difference from the test above is the presence of a region
    elsewhere in the body, which is precisely the manual-mode opt-in.
    """
    span = ir.Span.unknown()
    add, res = _vector_add(span)
    inner_data = ir.Var("id", _tile([16, 128]), span)
    inner_add = T.add(inner_data, inner_data, span)
    inner_res = ir.Var("inner", inner_add.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(inner_res, inner_add, span)])
    program = _program(ir.SeqStmts([ir.AssignStmt(res, add, span), region], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "vector op" in errors[0].message
    assert "tile.add" in errors[0].message
    assert "outside every pl.split_aiv region" in errors[0].message
    # The diagnostic names the fix the author is expected to apply.
    assert "pl.split_aiv(2, mode=pl.SplitMode.NONE)" in errors[0].message


def test_vector_inside_region_passes_when_a_region_exists():
    """(e) is scoped to the OUTSIDE: the identical op inside the region is fine."""
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128]), span)
    add = T.add(data, data, span)
    res = ir.Var("res", add.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, add, span)])
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []


def test_load_and_store_outside_region_pass_when_a_region_exists():
    """(e) carves out tile.load / tile.store — the COMPILER's own out-of-region output.

    ConvertTensorToTileOps materialises the load/store pair for a tensor-level op
    OUTSIDE the region that holds the compute it feeds (a ``pl.exp(gm_tensor)``
    written inside a region becomes a hoisted ``tile.load`` plus an in-region
    ``tile.exp``). Without the carve-out this check would fire on IR the compiler
    itself produced, so it is load-bearing rather than a tuning knob.
    """
    span = ir.Span.unknown()
    src = ir.Var("src", ir.TensorType([16, 128], FP32), span)
    loaded = T.load(src, [0, 0], [16, 128], target_memory=MS.Vec, span=span)
    tile_var = ir.Var("t", loaded.type, span)
    dst = ir.Var("dst", ir.TensorType([16, 128], FP32), span)
    stored = T.store(tile_var, [0, 0], dst, span=span)
    stored_var = ir.Var("stored", stored.type, span)

    inner = ir.Var("id", _tile([16, 128]), span)
    inner_add = T.add(inner, inner, span)
    inner_res = ir.Var("inner", inner_add.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(inner_res, inner_add, span)])
    program = _program(
        ir.SeqStmts(
            [ir.AssignStmt(tile_var, loaded, span), region, ir.AssignStmt(stored_var, stored, span)],
            span,
        )
    )

    assert _errors(program) == []


# ---------------------------------------------------------------------------
# Task-parallel (NONE) regions: no split axis. The boundary ops are ACCEPTED
# there — without a split axis they still mean "this value crosses the AIC/AIV
# boundary", which is what checks (f)/(g) below require an author to write. The
# genuinely split-axis-specific rule (reduce-on-split-axis) does not apply,
# since both lanes run the full body.
# ---------------------------------------------------------------------------


def test_shard_in_none_region_passes_and_preserves_shape():
    """(c) accepts tile.aiv_shard in a NONE region, and split=0 preserves the shape.

    The op is the crossing, not the split. With no split axis there is nothing to
    halve, so the result type matches the operand's — which is what makes the
    explicit form usable in a task-parallel region at all.
    """
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128], MS.Acc), span)
    shard = T.aiv_shard(data, split=int(ir.SplitMode.NONE.value), span=span)
    res = ir.Var("res", shard.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, shard, span)])
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []
    shard_type = shard.type
    assert isinstance(shard_type, ir.TileType)
    assert shard_type.shape == [16, 128]


def test_gather_in_none_region_passes_and_preserves_shape():
    """Mirror of the shard case for the V->C direction."""
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128], MS.Vec), span)
    gather = T.aic_gather(data, split=int(ir.SplitMode.NONE.value), span=span)
    res = ir.Var("res", gather.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, gather, span)])
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []
    gather_type = gather.type
    assert isinstance(gather_type, ir.TileType)
    assert gather_type.shape == [16, 128]


def test_boundary_memory_contract_still_checked_in_none_region():
    """(d) is mode-independent: a NONE crossing spans the same two lanes.

    Only the shape stops depending on the mode — which lane produces the value
    and which consumes it does not, so a Vec operand is as wrong for a
    ``tile.aiv_shard`` here as in a data-parallel region. NEGATIVE control for
    the two acceptance tests above: relaxing (c) must not silently relax (d).
    """
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128], MS.Vec), span)
    shard = T.aiv_shard(data, split=int(ir.SplitMode.NONE.value), span=span)
    res = ir.Var("res", shard.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, shard, span)])
    program = _program(ir.SeqStmts([region], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "tile.aiv_shard" in errors[0].message
    assert "requires Acc" in errors[0].message


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


def test_cube_in_none_region_fails():
    """(a) fires in a task-parallel region too, for the mode-independent reason.

    The split-axis argument ("each lane holds half the tile") does not apply
    here — a NONE region halves nothing. Manual mode supplies the second,
    mode-independent reason: the region body *is* the AIV lane's body, so cube
    work does not belong in it whatever the mode. The diagnostic states only
    that reason, and does not claim the tile was halved.
    """
    span = ir.Span.unknown()
    lhs = ir.Var("lhs", _tile([16, 128], MS.Left), span)
    rhs = ir.Var("rhs", _tile([128, 64], MS.Right), span)
    mm = T.matmul(lhs, rhs, span)
    res = ir.Var("res", mm.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, mm, span)])
    program = _program(ir.SeqStmts([region], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "cube op" in errors[0].message
    assert "tile.matmul" in errors[0].message
    assert "region body is AIV work" in errors[0].message
    assert "half the tile" not in errors[0].message


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


# --- Accepted: task-parallel (NONE) region — the crossing without the split ---


def test_tensor_shard_in_none_region_passes():
    """tensor.aiv_shard inside a NONE region is accepted, shape preserved."""
    span = ir.Span.unknown()
    shard = _tensor_shard([16, 128], int(ir.SplitMode.NONE.value), span)
    res = ir.Var("res", shard.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, shard, span)])
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []
    shard_type = shard.type
    assert isinstance(shard_type, ir.TensorType)
    assert shard_type.shape == [16, 128]


def test_tensor_gather_in_none_region_passes():
    """tensor.aic_gather inside a NONE region is accepted, shape preserved."""
    span = ir.Span.unknown()
    gather = _tensor_gather([8, 128], int(ir.SplitMode.NONE.value), span)
    res = ir.Var("res", gather.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, gather, span)])
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []
    gather_type = gather.type
    assert isinstance(gather_type, ir.TensorType)
    assert gather_type.shape == [8, 128]


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


# ---------------------------------------------------------------------------
# A well-formed manual-mode mixed body: no rule fires.
#
# Every rule above is a rejection; this is the shape they must all accept —
# cube work at top level, and every vector op (including the cross-rank comm
# ops, which the region pins to the AIV lane) inside a region.
#
# There is deliberately NO check on lane-sharding of once-only side effects: a
# region cannot mean "exactly once" while the AIV body runs on both sub-lanes
# (dual_aiv_dispatch), and the correct authoring form (sharded by aiv_id) and
# the incorrect one (both lanes notifying the same peer) are structurally
# identical IR. The rule is documented for authors instead — see
# docs/en/user/language/04-scopes.md and the pl.split_aiv docstring.
# ---------------------------------------------------------------------------


def _cube_stmt(span) -> ir.Stmt:
    """A cube matmul binding (Acc output) — the CUBE half of a mixed body."""
    lhs = ir.Var("lhs", _tile([16, 128], MS.Left), span)
    rhs = ir.Var("rhs", _tile([128, 64], MS.Right), span)
    mm = T.matmul(lhs, rhs, span)
    return ir.AssignStmt(ir.Var("res", mm.type, span), mm, span)


def _vector_stmt(span) -> ir.Stmt:
    """A vector add binding (Vec output) — the VECTOR half of a mixed body."""
    data = ir.Var("d", _tile([16, 128]), span)
    add = T.add(data, data, span)
    return ir.AssignStmt(ir.Var("vec", add.type, span), add, span)


def test_wellformed_manual_mode_mixed_body_passes():
    """Cube outside, vector + comm inside a region -> no diagnostics.

    The MIXED InCore shape a real comm kernel has. It pins that neither (a)
    (cube compute is at top level, not in the region) nor (e) (every vector op,
    and the notify with it, is inside the region) misfires on the arrangement
    the docs tell authors to write.
    """
    span = ir.Span.unknown()
    region = _region(ir.SplitMode.NONE, [_vector_stmt(span), ir.EvalStmt(_notify(span), span)])
    program = _program(ir.SeqStmts([_cube_stmt(span), region], span), func_type=ir.FunctionType.InCore)

    assert _errors(program) == []


def test_declared_vector_affinity_op_outside_region_passes():
    """(e) carve-out: an op that DECLARES its lane is exempt.

    Manual mode exists to stop the compiler *inferring* AIV placement outside a
    region. ``system.syncall(core_type="aiv_only")`` states its lane outright
    via ``set_core_affinity``, so a region cannot make it less ambiguous and
    rejecting it would be a regression — this was legal before manual mode and
    stays legal. The same exemption covers ``system.sync_set`` / ``sync_wait``
    with ``core_type="aiv"`` and the ``pld.tile.put`` / ``get`` family.

    Paired with ``test_vector_compute_outside_region_fails_in_manual_mode``:
    together they pin that (e) discriminates on *how* the lane was decided, not
    merely on the resulting affinity value.
    """
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128]), span)
    add = T.add(data, data, span)
    res = ir.Var("res", add.type, span)
    barrier = ir.create_op_call("system.syncall", [], {"core_type": "aiv_only"}, span)
    # The region is what puts the function into manual mode; the barrier sits
    # outside it and must still be accepted.
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, add, span)])
    program = _program(ir.SeqStmts([region, ir.EvalStmt(barrier, span)], span))

    assert _errors(program) == []


# ---------------------------------------------------------------------------
# (f) / (g): region-edge crossings must be explicit.
#
# In manual mode the author owns the AIC/AIV boundary, so a tile value that
# crosses a region edge has to say so with a boundary op. Both directions
# already LOWER without one (the compiler emits a split=0 tpush/tpop pair for an
# implicit crossing either way) — which is precisely why the check is needed:
# the crossing was happening silently, at a place nobody chose.
#
# Every rejection below is paired with the explicit form it asks for, plus the
# two carve-outs (tile.load / tile.store) and the no-region control.
# ---------------------------------------------------------------------------


def _cube_matmul(span, shape=(16, 128)):
    """A cube matmul call producing an Acc tile of ``shape``."""
    lhs = ir.Var("lhs", _tile([shape[0], 128], MS.Left), span)
    rhs = ir.Var("rhs", _tile([128, shape[1]], MS.Right), span)
    return T.matmul(lhs, rhs, span)


def test_v2c_implicit_crossing_fails():
    """(f) A vector value defined in a region, matmul'd outside it -> Error."""
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128]), span)
    add = T.add(data, data, span)
    vec = ir.Var("vec", add.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(vec, add, span)])
    # The cube consumer outside the region reads the in-region value directly.
    rhs = ir.Var("rhs", _tile([128, 64], MS.Right), span)
    mm = T.matmul(vec, rhs, span)
    program = _program(ir.SeqStmts([region, ir.AssignStmt(ir.Var("mm", mm.type, span), mm, span)], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "'vec'" in errors[0].message
    assert "tile.matmul" in errors[0].message
    assert "pl.aic_gather" in errors[0].message


def test_v2c_explicit_gather_passes():
    """The positive of (f): the same crossing, gathered inside the region.

    Pins that the check asks for something achievable — and that the achievable
    form is accepted, so (f) is a requirement to be explicit rather than a ban on
    crossing.
    """
    span = ir.Span.unknown()
    data = ir.Var("d", _tile([16, 128]), span)
    add = T.add(data, data, span)
    vec = ir.Var("vec", add.type, span)
    gather = T.aic_gather(vec, split=int(ir.SplitMode.NONE.value), span=span)
    gathered = ir.Var("gathered", gather.type, span)
    region = _region(
        ir.SplitMode.NONE,
        [ir.AssignStmt(vec, add, span), ir.AssignStmt(gathered, gather, span)],
    )
    rhs = ir.Var("rhs", _tile([128, 64], MS.Right), span)
    mm = T.matmul(gathered, rhs, span)
    program = _program(ir.SeqStmts([region, ir.AssignStmt(ir.Var("mm", mm.type, span), mm, span)], span))

    assert _errors(program) == []


def test_c2v_implicit_crossing_fails():
    """(g) A cube value produced outside every region, read by a vector op inside."""
    span = ir.Span.unknown()
    mm = _cube_matmul(span)
    acc = ir.Var("acc", mm.type, span)
    add = T.add(acc, acc, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(ir.Var("v", add.type, span), add, span)])
    program = _program(ir.SeqStmts([ir.AssignStmt(acc, mm, span), region], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "'acc'" in errors[0].message
    assert "tile.add" in errors[0].message
    assert "pl.aiv_shard" in errors[0].message


def test_c2v_explicit_shard_passes():
    """The positive of (g): the same crossing, sharded at the top of the region."""
    span = ir.Span.unknown()
    mm = _cube_matmul(span)
    acc = ir.Var("acc", mm.type, span)
    shard = T.aiv_shard(acc, split=int(ir.SplitMode.NONE.value), span=span)
    sharded = ir.Var("sharded", shard.type, span)
    add = T.add(sharded, sharded, span)
    region = _region(
        ir.SplitMode.NONE,
        [ir.AssignStmt(sharded, shard, span), ir.AssignStmt(ir.Var("v", add.type, span), add, span)],
    )
    program = _program(ir.SeqStmts([ir.AssignStmt(acc, mm, span), region], span))

    assert _errors(program) == []


def test_hoisted_load_into_region_passes():
    """Carve-out: a compiler-hoisted ``tile.load`` feeding in-region compute.

    ConvertTensorToTileOps materialises the load OUTSIDE the region that holds
    the compute it feeds, so treating it as an implicit crossing would fire (g)
    on the compiler's own output. Load into a CUBE space to make the definer
    genuinely cube-affine — without the carve-out this is exactly the shape that
    would be reported.
    """
    span = ir.Span.unknown()
    src = ir.Var("src", ir.TensorType([16, 128], FP32), span)
    loaded = T.load(src, [0, 0], [16, 128], target_memory=MS.Mat, span=span)
    tile_var = ir.Var("t", loaded.type, span)
    add = T.add(tile_var, tile_var, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(ir.Var("v", add.type, span), add, span)])
    program = _program(ir.SeqStmts([ir.AssignStmt(tile_var, loaded, span), region], span))

    assert _errors(program) == []


def test_hoisted_store_of_region_value_passes():
    """Carve-out, V->C side: a compiler-hoisted ``tile.store`` of an in-region value.

    The store is the other half of the pair ConvertTensorToTileOps hoists. It is
    exempted as a CONSUMER, so an in-region value reaching it is not reported as
    a crossing even when the stored tile sits in a cube space.
    """
    span = ir.Span.unknown()
    mm = _cube_matmul(span)
    acc = ir.Var("acc", mm.type, span)
    region = _region(ir.SplitMode.NONE, [ir.AssignStmt(acc, mm, span)])
    dst = ir.Var("dst", ir.TensorType([16, 128], FP32), span)
    stored = T.store(acc, [0, 0], dst, span=span)
    program = _program(
        ir.SeqStmts([region, ir.AssignStmt(ir.Var("stored", stored.type, span), stored, span)], span)
    )

    # (a) still reports the cube matmul inside the region; the crossing checks add
    # nothing on top of it.
    errors = _errors(program)
    assert len(errors) == 1
    assert "cube op" in errors[0].message


def test_crossing_checks_inert_without_a_region():
    """A function with NO region is untouched: no manual mode, no crossing rule.

    The same cube-produced value read by a vector op, and the same vector value
    matmul'd — both accepted, because nothing here declares a boundary the
    author owns.
    """
    span = ir.Span.unknown()
    mm = _cube_matmul(span)
    acc = ir.Var("acc", mm.type, span)
    add = T.add(acc, acc, span)
    vec = ir.Var("vec", add.type, span)
    rhs = ir.Var("rhs", _tile([128, 64], MS.Right), span)
    mm2 = T.matmul(vec, rhs, span)
    program = _program(
        ir.SeqStmts(
            [
                ir.AssignStmt(acc, mm, span),
                ir.AssignStmt(vec, add, span),
                ir.AssignStmt(ir.Var("mm2", mm2.type, span), mm2, span),
            ],
            span,
        )
    )

    assert _errors(program) == []


# ---------------------------------------------------------------------------
# Check (h) — placement: pl.split_aiv is a CORE_GROUP-level construct.
#
# It may be opened inside a CORE_GROUP scope (`pl.at(level=...)`) or at the top
# of an Opaque function (which the parser wraps into exactly such a scope). It
# may NOT be authored inside a function already declared FunctionType.InCore.
#
# The check keys on PROVENANCE, not shape: a region reaches an InCore function
# legitimately only when OutlineIncoreScopes lifted the enclosing scope into it,
# which the outliner records with the `split_aiv` attr. Keying on shape instead
# ("region nested in a surviving InCore scope") would silently stop rejecting
# anything now that the parser emits a top-level region bare in an InCore body.
# ---------------------------------------------------------------------------

_PLACEMENT_MSG = "CORE_GROUP-level region"


def test_region_in_incore_function_without_provenance_fails():
    """(h) A region hand-authored in an InCore function is rejected.

    The function carries no `split_aiv` attr, so it is not one the outliner
    produced — exactly the authoring form the placement rule forbids.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def core(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                out = pl.store(pl.exp(pl.load(a, [0, 0], [128, 128])), [0, 0], out)
            return out

    # The parser leaves the region bare — no wrapper to lean on, which is why
    # the check cannot be shape-based.
    assert "pl.at(" not in ir.python_print(Prog)
    errs = _errors(Prog)
    assert len(errs) == 1, errs
    assert _PLACEMENT_MSG in errs[0].message, errs[0].message


def test_region_in_incore_function_under_author_scope_fails():
    """(h) The same rejection when the author writes the CORE_GROUP scope out."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def core(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    out = pl.store(pl.exp(pl.load(a, [0, 0], [128, 128])), [0, 0], out)
            return out

    errs = _errors(Prog)
    assert len(errs) == 1, errs
    assert _PLACEMENT_MSG in errs[0].message, errs[0].message


def test_region_in_opaque_function_passes():
    """(h) The canonical authoring form — Opaque function — is accepted."""

    @pl.program
    class Prog:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                out = pl.store(pl.exp(pl.load(a, [0, 0], [128, 128])), [0, 0], out)
            return out

    assert _errors(Prog) == []


def test_outlined_incore_function_passes():
    """(h) The post-pass-8 shape is accepted: InCore function, but NO scope left.

    Boundary partner to the rejection above — same InCore function type, and the
    only difference is that OutlineIncoreScopes consumed the scope. This is what
    keeps the check from firing on the compiler's own output.
    """

    @pl.program
    class Prog:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                out = pl.store(pl.exp(pl.load(a, [0, 0], [128, 128])), [0, 0], out)
            return out

    with passes.PassContext([]):
        outlined = passes.outline_incore_scopes()(Prog)
    incore = [f for f in outlined.functions.values() if f.func_type == ir.FunctionType.InCore]
    assert len(incore) == 1, "pass 8 should have produced one InCore function"
    assert _errors(outlined) == []


# ---------------------------------------------------------------------------
# (i) A boundary result carried across a loop back-edge -> Error
# ---------------------------------------------------------------------------


def _shard_of_acc(span, shape=None):
    """An ``aiv_shard`` of an Acc (cube-produced) operand — the legal spelling.

    Returns ``(assign_stmt, result_var)``. Acc is the only valid operand space
    for a shard (check (d)), so every fixture below shares this one builder
    rather than open-coding it and risking an unrelated (d) error in the count.
    """
    data = ir.Var("acc", _tile(shape or [16, 128], mem=MS.Acc), span)
    shard = T.aiv_shard(data, split=int(ir.SplitMode.UP_DOWN.value), span=span)
    sharded = ir.Var("sh", shard.type, span)
    return ir.AssignStmt(sharded, shard, span), sharded


def _loop(span, iter_args, body_stmts, return_vars=None):
    """A minimal sequential ForStmt carrying ``iter_args``."""
    zero = ir.ConstInt(0, DataType.INDEX, span)
    two = ir.ConstInt(2, DataType.INDEX, span)
    one = ir.ConstInt(1, DataType.INDEX, span)
    return ir.ForStmt(
        ir.Var("i", ir.ScalarType(DataType.INDEX), span),
        zero,
        two,
        one,
        iter_args,
        ir.SeqStmts(body_stmts, span),
        return_vars if return_vars is not None else [],
        span,
    )


def test_boundary_result_carried_by_iter_arg_fails():
    """A shard result used as a loop iter_arg's INIT is a back-edge carry."""
    span = ir.Span.unknown()
    shard_stmt, sharded = _shard_of_acc(span)
    region = _region(ir.SplitMode.UP_DOWN, [shard_stmt])

    carry = ir.IterArg("carry", sharded.type, sharded, span)
    add = T.add(carry, carry, span)
    inner = ir.Var("inner", add.type, span)
    # The consumer lives in a mode=NONE region: legal per (j)'s carve-out, and it
    # keeps the fixture from also tripping (e) (vector op outside every region).
    loop = _loop(span, [carry], [_region(ir.SplitMode.NONE, [ir.AssignStmt(inner, add, span)])])
    program = _program(ir.SeqStmts([region, loop], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "across a loop back-edge" in errors[0].message
    assert "the initial value of" in errors[0].message
    assert "pl.aiv_shard" in errors[0].message
    assert "carry" in errors[0].message


def test_boundary_result_yielded_into_iter_arg_fails():
    """The yield end of (i): the init is a plain half-shaped tile, the YIELD is a shard.

    An init-only check would miss this — the loop is seeded with a non-boundary
    tile of the same (half) shape and fed by the boundary op from iteration 1
    onwards, which defeats the passes exactly as an init-side carry does.
    """
    span = ir.Span.unknown()
    seed_call = T.full([8, 128], FP32, 0.0, span=span)
    seed = ir.Var("seed", seed_call.type, span)

    carry = ir.IterArg("carry", seed_call.type, seed, span)
    add = T.add(carry, carry, span)
    inner = ir.Var("inner", add.type, span)
    shard_stmt, sharded = _shard_of_acc(span, shape=[16, 128])
    region = _region(ir.SplitMode.UP_DOWN, [shard_stmt])
    loop = _loop(
        span,
        [carry],
        [
            _region(ir.SplitMode.NONE, [ir.AssignStmt(inner, add, span)]),
            region,
            ir.YieldStmt([sharded], span),
        ],
        return_vars=[ir.Var("rv", seed_call.type, span)],
    )
    seed_region = _region(ir.SplitMode.NONE, [ir.AssignStmt(seed, seed_call, span)])
    program = _program(ir.SeqStmts([seed_region, loop], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert "across a loop back-edge" in errors[0].message
    assert "yielded back into" in errors[0].message


def test_boundary_carry_in_while_loop_fails():
    """(i) covers WhileStmt iter_args too, not just ForStmt."""
    span = ir.Span.unknown()
    shard_stmt, sharded = _shard_of_acc(span)
    region = _region(ir.SplitMode.UP_DOWN, [shard_stmt])

    carry = ir.IterArg("carry", sharded.type, sharded, span)
    add = T.add(carry, carry, span)
    inner = ir.Var("inner", add.type, span)
    cond = ir.Var("cond", ir.ScalarType(DataType.BOOL), span)
    loop = ir.WhileStmt(
        cond,
        [carry],
        ir.SeqStmts([_region(ir.SplitMode.NONE, [ir.AssignStmt(inner, add, span)])], span),
        [],
        span,
    )
    program = _program(ir.SeqStmts([region, loop], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert "across a loop back-edge" in errors[0].message


def test_ordinary_tile_carried_by_iter_arg_passes():
    """The negative for (i): a NON-boundary tile carried across a loop is untouched.

    (i) must key on the boundary op, not on "an iter_arg exists in a split_aiv
    function" — ordinary loop-carried accumulators are the common case.
    """
    span = ir.Span.unknown()
    seed_call = T.full([16, 128], FP32, 0.0, span=span)
    seed = ir.Var("seed", seed_call.type, span)
    shard_stmt, _ = _shard_of_acc(span)
    region = _region(ir.SplitMode.UP_DOWN, [shard_stmt])

    carry = ir.IterArg("carry", seed_call.type, seed, span)
    add = T.add(carry, carry, span)
    inner = ir.Var("inner", add.type, span)
    loop = _loop(span, [carry], [_region(ir.SplitMode.NONE, [ir.AssignStmt(inner, add, span)])])
    seed_region = _region(ir.SplitMode.NONE, [ir.AssignStmt(seed, seed_call, span)])
    program = _program(ir.SeqStmts([seed_region, region, loop], span))

    assert _errors(program) == []


def test_boundary_produced_and_consumed_in_one_iteration_passes():
    """The other negative for (i): shard inside the loop body, consumed there.

    This is the form the diagnostic tells the author to move to, so it has to be
    accepted or the advice is a dead end.
    """
    span = ir.Span.unknown()
    shard_stmt, sharded = _shard_of_acc(span)
    add = T.add(sharded, sharded, span)
    inner = ir.Var("inner", add.type, span)
    region = _region(ir.SplitMode.UP_DOWN, [shard_stmt, ir.AssignStmt(inner, add, span)])
    loop = _loop(span, [], [region])
    program = _program(ir.SeqStmts([loop], span))

    assert _errors(program) == []


# ---------------------------------------------------------------------------
# (j) A boundary result consumed in a DIFFERENT data-parallel region -> Error
# ---------------------------------------------------------------------------


def test_boundary_consumed_in_other_up_down_region_fails():
    """A shard produced in one UP_DOWN region and read in another is double-halved."""
    span = ir.Span.unknown()
    shard_stmt, sharded = _shard_of_acc(span)
    producer = _region(ir.SplitMode.UP_DOWN, [shard_stmt])
    add = T.add(sharded, sharded, span)
    res = ir.Var("res", add.type, span)
    consumer = _region(ir.SplitMode.UP_DOWN, [ir.AssignStmt(res, add, span)])
    program = _program(ir.SeqStmts([producer, consumer], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert errors[0].rule_name == "AivSplitValid"
    assert "produced in a DIFFERENT pl.split_aiv region" in errors[0].message
    assert "halved a second time" in errors[0].message
    assert "pl.aiv_shard" in errors[0].message


def test_boundary_consumed_in_other_left_right_region_fails():
    """(j) is gated on data-parallel, not on UP_DOWN specifically."""
    span = ir.Span.unknown()
    shard_stmt, sharded = _shard_of_acc(span)
    producer = _region(ir.SplitMode.UP_DOWN, [shard_stmt])
    add = T.add(sharded, sharded, span)
    res = ir.Var("res", add.type, span)
    consumer = _region(ir.SplitMode.LEFT_RIGHT, [ir.AssignStmt(res, add, span)])
    program = _program(ir.SeqStmts([producer, consumer], span))

    errors = _errors(program)
    assert len(errors) == 1
    assert "produced in a DIFFERENT pl.split_aiv region" in errors[0].message


def test_boundary_consumed_in_none_region_passes():
    """The carve-out, and the most important negative in this file.

    A ``mode=NONE`` region has no split axis: its body is spliced through with no
    halving and no offset localization, so an incoming already-per-lane tile is
    used as-is on each lane. This is the shape the cross-core comm kernels use —
    if (j) ever starts firing here, the rule has been drawn in the wrong place.
    """
    span = ir.Span.unknown()
    shard_stmt, sharded = _shard_of_acc(span)
    producer = _region(ir.SplitMode.UP_DOWN, [shard_stmt])
    add = T.add(sharded, sharded, span)
    res = ir.Var("res", add.type, span)
    consumer = _region(ir.SplitMode.NONE, [ir.AssignStmt(res, add, span)])
    program = _program(ir.SeqStmts([producer, consumer], span))

    assert _errors(program) == []


def test_boundary_consumed_in_defining_region_passes():
    """The plain negative for (j): consumed in the region that produced it."""
    span = ir.Span.unknown()
    shard_stmt, sharded = _shard_of_acc(span)
    add = T.add(sharded, sharded, span)
    res = ir.Var("res", add.type, span)
    region = _region(ir.SplitMode.UP_DOWN, [shard_stmt, ir.AssignStmt(res, add, span)])
    program = _program(ir.SeqStmts([region], span))

    assert _errors(program) == []


# ---------------------------------------------------------------------------
# (i) / (j) at the DSL level.
#
# The hand-built fixtures above would all pass even if the checks never fired on
# real parser output, so these compile the actual authoring shapes. They are also
# what proves the rejection now lands at OutlineIncoreScopes rather than 12
# passes later, where the same programs used to surface as a misleading
# "plain full-width vector op" from LowerAutoVectorSplit or as an INTERNAL error
# from ExpandMixedKernel naming SSA the author never wrote.
# ---------------------------------------------------------------------------


def test_dsl_shard_carried_across_loop_back_edge_fails():
    """The software-pipelining shape: shard iteration i+1's value at the end of i."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            for _p in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):  # noqa: B007
                sh = pl.aiv_shard(a)
            for _i in pl.range(2):  # noqa: B007
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):  # noqa: B007
                    e = pl.exp(sh)  # noqa: F841
                    sh = pl.aiv_shard(a)
            return out

    # The loop carry only becomes an IterArg at ConvertToSSA (pass 4): as parsed,
    # `sh` is a plain reassignment and there is no back edge to see. Run SSA first
    # so the fixture models what the verifier actually meets in the pipeline.
    with passes.PassContext([]):
        ssa = passes.convert_to_ssa()(Prog)

    errors = _errors(ssa)
    assert len(errors) == 1
    assert "across a loop back-edge" in errors[0].message
    # The pass-20 message this one displaces was factually wrong about the tile
    # being full width; make sure it is not what the author sees any more.
    assert "plain full-width vector op" not in errors[0].message


def test_dsl_shard_consumed_in_other_region_fails():
    """A shard read in a second UP_DOWN region — no loop involved.

    This shape used to reach ptoas as a 'pto.tcvt' shape error, after the region
    halved the already-per-lane tile a second time and localized its store offset
    twice.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            for _p in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):  # noqa: B007
                sh = pl.aiv_shard(a)
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):  # noqa: B007
                e = pl.exp(sh)  # noqa: F841
            return out

    errors = _errors(Prog)
    assert len(errors) == 1
    assert "produced in a DIFFERENT pl.split_aiv region" in errors[0].message


def test_dsl_shard_consumed_in_none_region_passes():
    """The carve-out at the DSL level: a mode=NONE region halves nothing."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            for _p in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):  # noqa: B007
                sh = pl.aiv_shard(a)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):  # noqa: B007
                e = pl.exp(sh)  # noqa: F841
            return out

    assert _errors(Prog) == []


def test_dsl_shard_produced_and_consumed_in_one_iteration_passes():
    """The form check (i)'s message tells the author to move to."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            for _i in pl.range(2):  # noqa: B007
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):  # noqa: B007
                    sh = pl.aiv_shard(a)
                    e = pl.exp(sh)  # noqa: F841
            return out

    with passes.PassContext([]):
        ssa = passes.convert_to_ssa()(Prog)

    assert _errors(ssa) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

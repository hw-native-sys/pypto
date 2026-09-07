# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Unit tests for ``ClassifyCallAffinity`` (``src/ir/transforms/utils/core_affinity.cpp``).

``ClassifyCallAffinity`` decides which core executes a statement. It tries, in
order: the affinity declared at op registration, the dynamically-classified
special cases (``tile.move``, ``system.syncall``, the sync events, the
split-reshape ops), the op's output memory spec, and the first tile
*argument*'s memory space — falling back to SHARED, which ``ExpandMixedKernel``
duplicates onto both lanes.

This file characterises ``pld.tile.remote_load``, which falls off the end of
that list: it consumes a DistributedTensor plus scalar tuples and produces a
tile, so it declares no memory spec and offers no tile argument, and every rule
misses it. It therefore classifies SHARED — a **known gap**, benign today only
because the duplicated cube copy is dead and gets eliminated.

The tests below pin that behaviour deliberately, so that a future change to it
is a conscious one. See ``src/ir/op/distributed/remote_load.cpp`` for why the
two obvious fixes are both unsafe: declaring VECTOR is a false ISA claim (the
destination could be a cube-side buffer), and classifying from the *result*
tile makes ``LowerAutoVectorSplit`` treat the op as halvable when its
``offsets`` / ``shape`` tuples have no rewrite in the halving path.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.op import tile_ops as T
from pypto.pypto_core import testing


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure Ascend910B backend before each test and reset afterward."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


@pl.program
class _RemoteLoadProgram:
    """A mixed InCore kernel whose remote_load result feeds cube work."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        data: pld.DistributedTensor[[16, 64], pl.FP16],
        rhs: pl.Tensor[[16, 16], pl.FP16],
        acc_out: pl.Tensor[[16, 16], pl.FP32],
        peer: pl.Scalar[pl.INT32],
    ):
        remote = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[16, 16])
        b = pl.load(rhs, [0, 0], [16, 16])
        c = pl.matmul(remote, b)
        pl.store(c, [0, 0], acc_out)


def _leaf_call(stmt: ir.Stmt) -> ir.Call | None:
    """Return the Call an AssignStmt/EvalStmt carries, or None."""
    if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
        return stmt.value
    if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call):
        return stmt.expr
    return None


def _op_names(func: ir.Function | None) -> list[str]:
    """Return the op name of every top-level leaf Call in ``func``'s body."""
    assert func is not None, "function not found in program"
    names = []
    for stmt in ir.flatten_to_stmts(func.body):
        call = _leaf_call(stmt)
        if call is not None and isinstance(call.op, ir.Op):
            names.append(call.op.name)
    return names


def _find_call(func: ir.Function | None, op_name: str) -> ir.Call:
    """Return the single Call to ``op_name`` in ``func``'s body."""
    assert func is not None, "function not found in program"
    wanted = ir.get_op(op_name).name
    found = [
        call
        for stmt in ir.flatten_to_stmts(func.body)
        if (call := _leaf_call(stmt)) is not None and isinstance(call.op, ir.Op) and call.op.name == wanted
    ]
    assert len(found) == 1, f"expected exactly one {op_name}, got {len(found)}"
    return found[0]


def test_remote_load_classifies_shared_even_once_its_memory_space_is_resolved():
    """remote_load stays SHARED — the known gap, pinned deliberately.

    ``InferTileMemorySpace`` resolves the destination tile to ``Mem.Vec``, so
    the information needed to place this op precisely *is* available by pass 20.
    It is still not used: classifying from the result tile would also change
    what ``LowerAutoVectorSplit`` (pass 23) does, where a VECTOR-affine leaf is
    routed into the split-halving machinery. That machinery shrinks the result
    type but has no rewrite for this op's ``offsets`` / ``shape`` tuples, so the
    request would stay full-width while the destination halved.

    If you are here because you want to place remote_load properly: teach the
    halving path about the op first, then revisit the classification.
    """
    program = passes.infer_tile_memory_space()(passes.convert_to_ssa()(_RemoteLoadProgram))
    call = _find_call(program.get_function("kernel"), "pld.tile.remote_load")

    # The placement information exists...
    result_type = call.type
    assert isinstance(result_type, ir.TileType)
    assert result_type.memory_space == ir.MemorySpace.Vec

    # ...and is deliberately not consumed.
    assert testing.classify_call_affinity(call) == "shared"


def test_remote_load_is_shared_before_memory_space_is_resolved():
    """SHARED before InferTileMemorySpace too — for the more basic reason.

    Here the result tile's ``memory_space`` is not even resolved yet, so no rule
    could place the op regardless. Pinned separately from the post-pass-20 case
    so that a future fix which only handles one of the two windows shows up as a
    single failing test rather than silently half-working.
    """
    program = passes.convert_to_ssa()(_RemoteLoadProgram)
    call = _find_call(program.get_function("kernel"), "pld.tile.remote_load")

    unresolved = call.type
    assert isinstance(unresolved, ir.TileType)
    assert unresolved.memory_space is None
    assert testing.classify_call_affinity(call) == "shared"


def test_remote_load_is_placed_only_on_the_vector_lane():
    """End-to-end: no remote_load survives on the AIC function.

    This is what makes the SHARED classification benign today, and it is worth
    being precise about *why* it holds. ``ExpandMixedKernel`` does replicate the
    SHARED statement onto the cube lane; that copy is then dead — its ``Vec``
    result has no cube-lane consumer, because a cube consumer reaches the tile
    through the C/V boundary tpush/tpop instead — and ``FinalizeSplitCoreBody``
    runs DCE over the finalized body.

    So this asserts a property of the *lowered output*, not correct placement.
    It would start failing if a cube-lane statement ever came to consume the
    duplicated result, which is exactly the signal that the gap pinned by
    ``test_remote_load_classifies_shared_even_once_its_memory_space_is_resolved``
    has stopped being benign.
    """
    expanded = passes.expand_mixed_kernel()(
        passes.infer_tile_memory_space()(passes.convert_to_ssa()(_RemoteLoadProgram))
    )
    remote_load = ir.get_op("pld.tile.remote_load").name

    assert remote_load in _op_names(expanded.get_function("kernel_aiv"))
    assert remote_load not in _op_names(expanded.get_function("kernel_aic"))


# ---------------------------------------------------------------------------
# `pl.split_aiv` region placement override
#
# LowerAutoVectorSplit erases the region wrapper, so it leaves the author's
# placement behind as ``attrs["core_placement"] = "aiv"`` on the calls whose
# lane the region decides. ClassifyCallAffinity reads that as the placement
# authority, ahead of every rule above — which is what keeps a core-agnostic
# comm op off the cube lane instead of being duplicated onto both.
#
# The override refuses in three cases, each because the region is not what
# decides that call's lane. Every one is paired with its unstamped counterpart
# so the tests distinguish "the override did nothing" from "the override was
# never reached".
# ---------------------------------------------------------------------------

_PLACED = {"core_placement": "aiv"}


def _placed(call: ir.Call) -> ir.Call:
    """Re-mint ``call`` with the region placement stamp."""
    return ir.Call(call.op, call.args, call.kwargs, _PLACED, call.type, call.span)


def _tile(shape, mem):
    return ir.TileType(shape, pl.FP16, None, None, mem)


def _notify(span) -> ir.Call:
    sig = ir.Var("sig", ir.DistributedTensorType([4, 4], pl.INT32), span)
    peer = ir.Var("peer", ir.ScalarType(pl.INT32), span)
    zero = ir.ConstInt(0, pl.INDEX, span)
    value = ir.ConstInt(1, pl.INT32, span)
    return ir.create_op_call(
        "pld.system.notify", [sig, peer, ir.MakeTuple([zero, zero], span), value], {"op": 0}, span
    )


def test_region_placement_moves_a_core_agnostic_op_to_the_vector_lane():
    """SHARED + region placement -> VECTOR. The case the carrier exists for.

    ``pld.system.notify`` declares no core affinity (TNOTIFY is core-agnostic by
    ISA), so nothing but the region can place it — and SHARED is precisely what
    ExpandMixedKernel duplicates onto both lanes.
    """
    span = ir.Span.unknown()
    notify = _notify(span)

    assert testing.classify_call_affinity(notify) == "shared"
    assert testing.classify_call_affinity(_placed(notify)) == "vector"


def test_region_placement_leaves_a_stated_lane_alone():
    """A lane the op DECLARES outranks region placement.

    ``tile.create`` is SHARED by policy via ``set_core_affinity`` so both lanes
    can declare the buffer. Overriding it to VECTOR would drop the declaration
    from the cube lane. Note this is SHARED like the notify above, so it pins
    that the carve-out keys on *how* the lane was decided, not on the value.
    """
    span = ir.Span.unknown()
    create = T.create([16, 16], pl.FP16, ir.MemorySpace.Vec, span=span)

    assert testing.classify_call_affinity(create) == "shared"
    assert testing.classify_call_affinity(_placed(create)) == "shared"


def test_region_placement_leaves_the_cross_core_boundary_alone():
    """MIXED means "this call IS the transfer" — it really does need both lanes.

    ``tile.aiv_shard`` lowers to a tpush on the cube lane plus a tpop on the
    vector lane. Forcing it to VECTOR would leave the tpush without its tpop.
    """
    span = ir.Span.unknown()
    src = ir.Var("qk", _tile([128, 128], ir.MemorySpace.Acc), span)
    shard = ir.create_op_call("tile.aiv_shard", [src], {"split": 1}, span)

    assert testing.classify_call_affinity(shard) == "mixed"
    assert testing.classify_call_affinity(_placed(shard)) == "mixed"


def test_region_placement_does_not_drag_cube_work_onto_the_vector_lane():
    """CUBE stays CUBE — the verifier reports it, the classifier does not guess.

    Cube compute inside a region is an authoring error that AivSplitValid check
    (a) rejects. If verification is off, declining the override leaves the op on
    the cube lane exactly as before, rather than newly miscompiling a matmul
    onto the vector lane.
    """
    span = ir.Span.unknown()
    lhs = ir.Var("lhs", _tile([16, 128], ir.MemorySpace.Left), span)
    rhs = ir.Var("rhs", _tile([128, 16], ir.MemorySpace.Right), span)
    matmul = ir.create_op_call("tile.matmul", [lhs, rhs], {"out_dtype": pl.FP32}, span)

    assert testing.classify_call_affinity(matmul) == "cube"
    assert testing.classify_call_affinity(_placed(matmul)) == "cube"


def test_region_placement_is_a_no_op_for_ordinary_vector_compute():
    """VECTOR is already the answer; the stamp only confirms it.

    Pass 23 does not stamp these (their memory spec already places them), but
    the attr is ordinary IR that a hand-written program may carry, so the
    override must still give a defined — and unchanged — answer.
    """
    span = ir.Span.unknown()
    src = ir.Var("v", _tile([16, 128], ir.MemorySpace.Vec), span)
    add = ir.create_op_call("tile.add", [src, src], {}, span)

    assert testing.classify_call_affinity(add) == "vector"
    assert testing.classify_call_affinity(_placed(add)) == "vector"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

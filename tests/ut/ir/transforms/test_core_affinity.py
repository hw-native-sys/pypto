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


# Intrinsic affinity is independent of lexical region placement. The latter is
# covered by the ExpandMixedKernel region-consumption tests.


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


def test_notify_has_intrinsic_shared_affinity():
    span = ir.Span.unknown()
    notify = _notify(span)

    assert testing.classify_call_affinity(notify) == "shared"


def test_create_declares_shared_affinity():
    span = ir.Span.unknown()
    create = T.create([16, 16], pl.FP16, ir.MemorySpace.Vec, span=span)

    assert testing.classify_call_affinity(create) == "shared"


def test_boundary_has_mixed_affinity():
    span = ir.Span.unknown()
    src = ir.Var("qk", _tile([128, 128], ir.MemorySpace.Acc), span)
    shard = ir.create_op_call("tile.aiv_shard", [src], {"split": 1}, span)

    assert testing.classify_call_affinity(shard) == "mixed"


def test_matmul_has_cube_affinity():
    span = ir.Span.unknown()
    lhs = ir.Var("lhs", _tile([16, 128], ir.MemorySpace.Left), span)
    rhs = ir.Var("rhs", _tile([128, 16], ir.MemorySpace.Right), span)
    matmul = ir.create_op_call("tile.matmul", [lhs, rhs], {"out_dtype": pl.FP32}, span)

    assert testing.classify_call_affinity(matmul) == "cube"


def test_add_has_vector_affinity():
    span = ir.Span.unknown()
    src = ir.Var("v", _tile([16, 128], ir.MemorySpace.Vec), span)
    add = ir.create_op_call("tile.add", [src, src], {}, span)

    assert testing.classify_call_affinity(add) == "vector"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

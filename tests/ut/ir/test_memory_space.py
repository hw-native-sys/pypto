# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The "nothing moves into Acc" invariant, checked against the SoC graphs.

Two independent structures describe memory movement, and they are *not* the
same table:

* ``SoC::GetMemoryGraph()`` (``src/backend/common/soc.cpp``) models the memory
  hierarchy for ``Backend::FindMemPath``. It is coarser than tmov legality --
  it omits ``Acc -> Vec`` on Ascend910B, a move the pipeline emits and PTOAS
  accepts -- so it must not be used to decide whether a ``tile.move`` is legal.
* ``IsTileMoveEverSupported`` (``src/ir/memref.cpp``) mirrors PTOAS's
  ``TMovOp::verify``, unioned over targets.

They agree on exactly one row, and it is the row PyPTO depends on: **nothing
moves into ``Acc``**. These tests pin that agreement, so if either source ever
gains an inbound Acc edge the assumption is caught here rather than in a
miscompile.
"""

import pytest
from pypto import DataType, backend
from pypto.ir.op import tile
from pypto.pypto_core import ir, testing

MS = ir.MemorySpace

_SPACES = [
    MS.DDR,
    MS.Vec,
    MS.Mat,
    MS.Left,
    MS.Right,
    MS.Acc,
    MS.Bias,
    MS.LeftScale,
    MS.RightScale,
    MS.SRAM,
]

_TARGETS = [backend.BackendType.Ascend910B, backend.BackendType.Ascend950]


@pytest.fixture(autouse=True)
def _reset_backend():
    backend.reset_for_testing()
    yield
    backend.reset_for_testing()


def _direct_edges(backend_type) -> set[str]:
    """Direct edges of one target's memory graph.

    ``find_mem_path`` BFSs the same adjacency the graph stores, so a path of
    exactly two nodes is a single topology edge, not necessarily a ``pto.tmov``.
    """
    backend.set_backend_type(backend_type)
    be = backend.get_backend_instance(backend_type)
    edges = set()
    for src in _SPACES:
        for dst in _SPACES:
            if src == dst:
                continue
            try:
                path = be.find_mem_path(src, dst)
            except Exception:  # noqa: BLE001 - no path is the answer, not an error
                continue
            if len(path) == 2:
                edges.add(f"{src.name}->{dst.name}")
    return edges


@pytest.mark.parametrize("backend_type", _TARGETS)
def test_no_target_moves_into_acc(backend_type):
    """The invariant the operand check and MoveCollector both rely on.

    Only the matrix unit writes L0C, so an accumulator must be *created* in
    ``Acc``; no copy can put it there. ``OpRegistry::Create`` rejects an
    explicit non-Acc accumulator on this basis, and ``InferTileMemorySpace``
    Phase 1 places an unset one directly in ``Acc`` rather than leaving Phase 2
    to bridge. If this ever fails, both of those become wrong.
    """
    into_acc = {e for e in _direct_edges(backend_type) if e.endswith("->Acc")}
    assert into_acc == set(), f"{backend_type} gained a move into Acc: {sorted(into_acc)}"


def test_targets_disagree_on_some_edges():
    """The graphs are per-target, which is why no single shared table suffices.

    Recorded so a future attempt to collapse these into one table has to
    confront the difference rather than average it away.
    """
    a2a3 = _direct_edges(backend.BackendType.Ascend910B)
    backend.reset_for_testing()
    a5 = _direct_edges(backend.BackendType.Ascend950)

    assert "Vec->Mat" in a5
    assert "Vec->Mat" not in a2a3
    assert "Acc->Vec" in a5
    assert "Acc->Vec" not in a2a3


@pytest.mark.parametrize("backend_type", _TARGETS)
def test_sram_has_independent_external_routes(backend_type):
    """Shared pointer addressing does not collapse the two physical endpoints."""
    edges = _direct_edges(backend_type)
    assert {"SRAM->Vec", "SRAM->Mat", "Vec->SRAM", "Acc->SRAM"} <= edges
    assert {"DDR->Vec", "DDR->Mat", "Vec->DDR", "Acc->DDR"} <= edges
    assert {"DDR->SRAM", "SRAM->DDR"} <= edges
    be = backend.get_backend_instance(backend_type)
    assert be.find_mem_path(MS.DDR, MS.SRAM) == [MS.DDR, MS.SRAM]
    assert be.find_mem_path(MS.SRAM, MS.DDR) == [MS.SRAM, MS.DDR]
    assert be.find_mem_path(MS.SRAM, MS.Mat) == [MS.SRAM, MS.Mat]
    assert be.find_mem_path(MS.DDR, MS.Mat) == [MS.DDR, MS.Mat]
    assert be.find_mem_path(MS.SRAM, MS.Left) == [MS.SRAM, MS.Mat, MS.Left]
    assert be.find_mem_path(MS.Vec, MS.SRAM) == [MS.Vec, MS.SRAM]
    # Capacity belongs to the chip, not to a per-core allocation pool.
    assert be.get_mem_size(MS.SRAM) == 256 * 1024 * 1024


@pytest.mark.parametrize("backend_type", _TARGETS)
def test_sram_vec_mat_connections_match_ddr(backend_type):
    """Match both directions, including unsupported Mat-to-external writeback."""
    edges = _direct_edges(backend_type)
    for buffer in ("Vec", "Mat"):
        assert (f"SRAM->{buffer}" in edges) == (f"DDR->{buffer}" in edges)
        assert (f"{buffer}->SRAM" in edges) == (f"{buffer}->DDR" in edges)


@pytest.mark.parametrize("backend_type", _TARGETS)
@pytest.mark.parametrize("medium", [None, MS.DDR, MS.SRAM])
@pytest.mark.parametrize("destination", [MS.Vec, MS.Mat])
def test_external_load_pipe(backend_type, medium, destination):
    backend.set_backend_type(backend_type)
    tensor = ir.Var("input", ir.TensorType([16, 128], DataType.FP32), ir.Span.unknown())
    call = tile.load(tensor, [0, 0], [16, 128], source_memory=medium, target_memory=destination)
    assert testing.try_infer_pipe(call) == int(ir.PipeType.MTE2)
    assert testing.classify_call_affinity(call) == ("vector" if destination == MS.Vec else "cube")


@pytest.mark.parametrize("backend_type", _TARGETS)
@pytest.mark.parametrize("medium", [None, MS.DDR, MS.SRAM])
@pytest.mark.parametrize("source,pipe", [(MS.Vec, ir.PipeType.MTE3), (MS.Acc, ir.PipeType.FIX)])
def test_external_store_pipe(backend_type, medium, source, pipe):
    backend.set_backend_type(backend_type)
    span = ir.Span.unknown()
    tensor = ir.Var("output", ir.TensorType([16, 128], DataType.FP32), span)
    value = ir.Var("value", ir.TileType([16, 128], DataType.FP32, memory_space=source), span)
    call = tile.store(value, [0, 0], tensor, target_memory=medium)
    assert testing.try_infer_pipe(call) == int(pipe)
    assert testing.classify_call_affinity(call) == ("vector" if source == MS.Vec else "cube")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

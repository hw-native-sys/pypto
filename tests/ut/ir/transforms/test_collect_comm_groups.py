# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Tests for the ``CollectCommGroups`` pass.

The pass walks each ``host_orch`` function, traces
``pld.tensor.alloc_window_buffer → pld.tensor.window → dispatch(device=r)``
chains, and:

* constructs one :class:`ir.WindowBuffer` per alloc,
* rewrites every ``pld.tensor.window`` result Var so its
  :class:`ir.DistributedTensorType` carries a ``window_buffer`` back-reference,
* clusters allocs with the same device descriptor into a single
  :class:`ir.CommGroup` and writes them to ``Program.comm_groups``.

Most tests below run the pass directly on a parsed program (via
``passes.collect_comm_groups()(program)``) and assert on the produced IR by
node inspection rather than the Before/Expected ``assert_structural_equal``
pattern. The pass's two output products have no print/parse surface syntax:

* ``Program.comm_groups`` — a ``UsualField`` compared by ``structural_equal``,
  but the printer emits no ``comm_groups`` syntax and the parser parses none.
* ``DistributedTensorType.window_buffer_`` — a ``UsualField`` back-reference
  on each view Var, also compared by ``structural_equal`` but not printed.

An ``Expected`` ``@pl.program`` is built by parsing Python source, so it would
always carry an empty ``comm_groups`` and ``window_buffer``-less view types,
mismatching the pass output. ``test_no_alloc_window_buffer_no_op`` is the one
exception: it produces no such fields and uses ``assert_structural_equal``.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import ir, passes


@pytest.fixture(autouse=True)
def _basic_verification_context():
    """Override the ``ut/conftest.py`` autouse fixture to run with
    BEFORE_AND_AFTER property verification but no print/parse roundtrip.

    The pass's output materialises ``Program.comm_groups`` and
    ``DistributedTensorType.window_buffer_`` back-references on view Vars,
    but the printer / parser pair has no surface syntax for either — so the
    roundtrip-symmetric check would fail every iteration despite the
    in-memory IR being correct. Property verification still runs.
    """
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _find_window_calls(func: ir.Function) -> list[ir.AssignStmt]:
    """Return all ``AssignStmt``s whose RHS is a ``pld.tensor.window`` Call."""

    found: list[ir.AssignStmt] = []

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.AssignStmt):
            if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "pld.tensor.window":
                found.append(stmt)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        if isinstance(stmt, ir.ForStmt):
            walk(stmt.body)

    walk(func.body)
    return found


def _view_var_types(func: ir.Function) -> list[ir.DistributedTensorType]:
    """Return the type of each pld.tensor.window result Var, in source order."""
    return [
        stmt.var.type
        for stmt in _find_window_calls(func)
        if isinstance(stmt.var.type, ir.DistributedTensorType)
    ]


def _apply(program: ir.Program) -> ir.Program:
    return passes.collect_comm_groups()(program)


# ---------------------------------------------------------------------------
# Single alloc / ALL devices
# ---------------------------------------------------------------------------


def test_single_alloc_all_devices_world_size_loop():
    """``for r in pl.range(pld.world_size())`` + ``device=r`` ⇒ kAll."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(1024)
            data = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return 0

    result = _apply(P)
    assert len(result.comm_groups) == 1
    g = result.comm_groups[0]
    # devices == [] is the on-the-wire encoding for "all devices" (kAll).
    assert list(g.devices) == []
    assert len(g.slots) == 1
    wb = g.slots[0]
    assert isinstance(wb, ir.WindowBuffer)
    assert wb.name_hint == "buf"
    assert isinstance(wb.size, ir.ConstInt)
    assert wb.size.value == 1024
    # The view's window_buffer back-reference now points to the same wb.
    host = _get_func(result, "host_orch")
    view_types = _view_var_types(host)
    assert len(view_types) == 1
    assert view_types[0].window_buffer is wb


# ---------------------------------------------------------------------------
# Single alloc / explicit ConstInt subset
# ---------------------------------------------------------------------------


def test_single_alloc_subset_const_int_devices():
    """Two separate ``device=0`` / ``device=1`` dispatches ⇒ subset {0,1}."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(1024)
            data = pld.window(buf, [256], dtype=pl.FP32)
            self.chip_orch(data, device=0)
            self.chip_orch(data, device=1)
            return 0

    result = _apply(P)
    assert len(result.comm_groups) == 1
    g = result.comm_groups[0]
    assert list(g.devices) == [0, 1]
    assert len(g.slots) == 1
    assert g.slots[0].name_hint == "buf"


# ---------------------------------------------------------------------------
# Single alloc / bounded loop
# ---------------------------------------------------------------------------


def test_single_alloc_bounded_loop_devices():
    """``for r in pl.range(2)`` + ``device=r`` ⇒ subset {0, 1}."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(1024)
            data = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(2):
                self.chip_orch(data, device=r)
            return 0

    result = _apply(P)
    assert len(result.comm_groups) == 1
    assert list(result.comm_groups[0].devices) == [0, 1]


# ---------------------------------------------------------------------------
# Two allocs / same descriptor → one group, two slots in alloc order
# ---------------------------------------------------------------------------


def test_two_allocs_same_descriptor_one_group():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            data: pld.DistributedTensor[[256], pl.FP32],
            signal: pld.DistributedTensor[[8], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf_data = pld.alloc_window_buffer(1024)
            buf_signal = pld.alloc_window_buffer(32)
            data = pld.window(buf_data, [256], dtype=pl.FP32)
            signal = pld.window(buf_signal, [8], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            return 0

    result = _apply(P)
    assert len(result.comm_groups) == 1
    g = result.comm_groups[0]
    assert list(g.devices) == []  # kAll
    assert [s.name_hint for s in g.slots] == ["buf_data", "buf_signal"]


# ---------------------------------------------------------------------------
# Two allocs / different descriptors → two groups
# ---------------------------------------------------------------------------


def test_two_allocs_different_descriptors_two_groups():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_a(self, a: pld.DistributedTensor[[256], pl.FP32]):
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_b(self, b: pld.DistributedTensor[[256], pl.FP32]):
            return b

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf_a = pld.alloc_window_buffer(1024)
            buf_b = pld.alloc_window_buffer(1024)
            a = pld.window(buf_a, [256], dtype=pl.FP32)
            b = pld.window(buf_b, [256], dtype=pl.FP32)
            self.chip_orch_a(a, device=0)
            self.chip_orch_a(a, device=1)
            self.chip_orch_b(b, device=2)
            self.chip_orch_b(b, device=3)
            return 0

    result = _apply(P)
    assert len(result.comm_groups) == 2
    devices = sorted([tuple(g.devices) for g in result.comm_groups])
    assert devices == [(0, 1), (2, 3)]


# ---------------------------------------------------------------------------
# Single alloc / multiple views share the same WindowBuffer instance
# ---------------------------------------------------------------------------


def test_multi_view_per_alloc_shares_wb_instance():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            v1: pld.DistributedTensor[[256], pl.FP32],
            v2: pld.DistributedTensor[[256], pl.FP32],
        ):
            return v1

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(2048)
            view_a = pld.window(buf, [256], dtype=pl.FP32)
            view_b = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(view_a, view_b, device=r)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    types = _view_var_types(host)
    assert len(types) == 2
    wb_a = types[0].window_buffer
    wb_b = types[1].window_buffer
    assert wb_a is not None and wb_b is not None
    # Both views materialise the same alloc → identical WindowBuffer shared_ptr.
    assert wb_a is wb_b
    # And that same WindowBuffer is the (single) slot of the (single) group.
    assert len(result.comm_groups) == 1
    assert len(result.comm_groups[0].slots) == 1
    assert result.comm_groups[0].slots[0] is wb_a


# ---------------------------------------------------------------------------
# chip_orch param types remain nullopt (host_orch only is rewritten)
# ---------------------------------------------------------------------------


def test_chip_orch_param_types_keep_window_buffer_nullopt():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(1024)
            data = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return 0

    result = _apply(P)
    chip = _get_func(result, "chip_orch")
    # The param annotation came from the user; its DistributedTensorType
    # should carry no window_buffer back-reference even after the pass runs.
    assert len(chip.params) == 1
    pt = chip.params[0].type
    assert isinstance(pt, ir.DistributedTensorType)
    assert pt.window_buffer is None


# ---------------------------------------------------------------------------
# Dead alloc — alloc + window but no dispatch consumer
# ---------------------------------------------------------------------------


def test_dead_alloc_no_dispatch_raises():
    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(1024)
            _ = pld.window(buf, [256], dtype=pl.FP32)
            return 0

    with pytest.raises(Exception, match="not consumed by any chip_orch dispatch"):
        _apply(P)


# ---------------------------------------------------------------------------
# Idempotence — running the pass twice produces an equivalent program
# ---------------------------------------------------------------------------


def test_idempotent():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(1024)
            data = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return 0

    first = _apply(P)
    second = _apply(first)
    assert len(first.comm_groups) == 1
    assert len(second.comm_groups) == 1
    # Same device coverage; same number of slots.
    assert list(first.comm_groups[0].devices) == list(second.comm_groups[0].devices)
    assert len(first.comm_groups[0].slots) == len(second.comm_groups[0].slots)


# ---------------------------------------------------------------------------
# Programs without any pld.tensor.alloc_window_buffer pass straight through
# ---------------------------------------------------------------------------


def test_no_alloc_window_buffer_no_op():
    @pl.program
    class Before:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
            return x

    After = _apply(Before)
    # No alloc_window_buffer chains ⇒ the pass produces no CommGroups and
    # rewrites nothing, so the program is unchanged.
    ir.assert_structural_equal(After, Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pl.builtin.<category>.<op>`` — the machine-only surface
for compiler-internal operators.

``builtin.*`` operators are synthesized by lowering passes (today
``builtin.tensor.*``, emitted by ``LowerHostTensorCollectives`` for the host
``pld.tensor.*`` collectives) and are marked ``internal_only`` in the registry,
so no DSL wrapper spells them and users write the composite ``pld.tensor.*``
form instead. The printer nevertheless renders them as
``pl.builtin.<category>.<op>``, so the parser must read that spelling back or
IR past those passes stops round-tripping.

End-to-end coverage of the printed lowering (kwargs, ``device`` /
``arg_directions`` attrs, every collective) lives in
:file:`tests/ut/ir/transforms/test_lower_host_tensor_collectives.py`; this
module covers the namespace's own parse behaviour and diagnostics.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.language.parser.diagnostics import InvalidOperationError
from pypto.pypto_core import ir

_BUILTIN_BARRIER = ir.get_op("builtin.tensor.barrier").name


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _iter_stmts(stmt: ir.Stmt):
    """Yield ``stmt`` and every nested statement (flattening containers)."""
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            yield from _iter_stmts(s)
    if isinstance(stmt, ir.ScopeStmt):
        yield from _iter_stmts(stmt.body)
    if isinstance(stmt, ir.ForStmt):
        yield from _iter_stmts(stmt.body)


_BARRIER_SRC = """
import pypto.language as pl
import pypto.language.distributed as pld


@pl.program
class P:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(self):
        signal_buf: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(16, pl.INDEX))
        signal: pld.DistributedTensor[[4], pl.INT32] = pld.tensor.window(signal_buf, [4], dtype=pl.INT32)
        for r in pl.range(pld.system.world_size()):
            {call}
        return 0
"""


def _barrier_program(call: str) -> str:
    return _BARRIER_SRC.format(call=call)


_PRINTED_BARRIER = 'pl.builtin.tensor.barrier(signal, attrs={"device": r, "arg_directions": [pl.adir.inout]})'


def test_builtin_op_parses_to_the_internal_registry_op():
    """The printed form builds the internal-only registered op."""
    program = pl.parse_program(_barrier_program(_PRINTED_BARRIER))
    host = _get_func(program, "host_orch")

    calls = [
        stmt.expr
        for stmt in _iter_stmts(host.body)
        if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call)
    ]
    barriers = [call for call in calls if call.op.name == _BUILTIN_BARRIER]
    assert len(barriers) == 1, [call.op.name for call in calls]
    assert len(barriers[0].args) == 1


def test_builtin_op_carries_machine_only_attrs():
    """The trailing ``attrs={...}`` dict the printer emits round-trips."""
    program = pl.parse_program(_barrier_program(_PRINTED_BARRIER))
    host = _get_func(program, "host_orch")

    barriers = [
        stmt.expr
        for stmt in _iter_stmts(host.body)
        if isinstance(stmt, ir.EvalStmt)
        and isinstance(stmt.expr, ir.Call)
        and stmt.expr.op.name == _BUILTIN_BARRIER
    ]
    assert len(barriers) == 1
    call = barriers[0]
    assert set(call.attrs.keys()) == {"device", "arg_directions"}
    assert isinstance(call.attrs["device"], ir.Var)
    assert call.attrs["device"].name_hint == "r"
    assert list(call.attrs["arg_directions"]) == [ir.ArgDirection.InOut]


def test_builtin_namespace_is_scoped_to_registered_builtin_ops():
    """An unregistered ``builtin.`` name is rejected — the namespace does not
    become a back door onto arbitrary internal operators."""
    with pytest.raises(InvalidOperationError, match="Unknown builtin operation"):
        pl.parse_program(_barrier_program(_PRINTED_BARRIER.replace("barrier", "not_a_collective")))

    # ``tensor.write`` is a real op, but it is not registered under ``builtin.``,
    # so the builtin namespace must not reach it either.
    with pytest.raises(InvalidOperationError, match="Unknown builtin operation"):
        pl.parse_program(_barrier_program(_PRINTED_BARRIER.replace("barrier", "write")))


def test_builtin_namespace_rejects_a_wrong_segment_count():
    """``pl.builtin`` is spelled ``pl.builtin.<category>.<op>``; a short chain
    names the full spelling the user wrote and points at the right form, rather
    than reporting ``Unknown operation 'pl.builtin'`` from the 2-segment
    unified path."""
    with pytest.raises(InvalidOperationError) as excinfo:
        pl.parse_program(_barrier_program(_PRINTED_BARRIER.replace("tensor.barrier", "barrier")))
    assert "pl.builtin.barrier" in str(excinfo.value)
    assert excinfo.value.hint is not None
    assert "pl.builtin.<category>.<op>" in excinfo.value.hint


def test_builtin_op_reports_deduction_failures_against_its_own_name():
    """A bad payload surfaces as an error naming the builtin, not a bare
    registry message."""
    two_args = _PRINTED_BARRIER.replace("(signal,", "(signal, signal,").replace(
        "[pl.adir.inout]", "[pl.adir.inout, pl.adir.inout]"
    )
    with pytest.raises(InvalidOperationError, match=r"builtin\.tensor\.barrier"):
        pl.parse_program(_barrier_program(two_args))


def test_hand_written_builtin_call_without_device_is_a_user_error():
    """A bare `pl.builtin.tensor.barrier(signal)` is rejected at parse time.

    Orchestration codegen resolves the dispatching rank from the `device` attr
    behind an `INTERNAL_CHECK`, so accepting a hand-written call without it
    would turn bad user input into a compiler-bug diagnostic much later. The
    printer always stamps the attr, so requiring it costs the round-trip
    nothing.
    """
    with pytest.raises(InvalidOperationError) as excinfo:
        pl.parse_program(_barrier_program("pl.builtin.tensor.barrier(signal)"))
    assert "device" in str(excinfo.value)
    assert excinfo.value.hint is not None
    assert "pld.tensor.barrier" in excinfo.value.hint


def test_hand_written_builtin_call_without_arg_directions_is_a_user_error():
    """`arg_directions` is the other invariant codegen reads back internally."""
    with pytest.raises(InvalidOperationError, match="arg_directions"):
        pl.parse_program(_barrier_program('pl.builtin.tensor.barrier(signal, attrs={"device": r})'))


def test_builtin_call_arg_directions_must_cover_every_arg():
    """One direction per positional arg — codegen asserts the two lengths match."""
    with pytest.raises(InvalidOperationError, match="arg_directions entries"):
        pl.parse_program(
            _barrier_program('pl.builtin.tensor.barrier(signal, attrs={"device": r, "arg_directions": []})')
        )


def test_public_collective_still_lowers_through_pld_surface():
    """The public surface is unchanged: users write ``pld.tensor.barrier``, and
    the internal name stays out of the parsed IR until the pass runs."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            pld.tensor.barrier(signal)
            return 0

    host = _get_func(P, "host_orch")
    op_names = {
        stmt.expr.op.name
        for stmt in _iter_stmts(host.body)
        if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call)
    }
    assert _BUILTIN_BARRIER not in op_names
    assert ir.get_op("pld.tensor.barrier").name in op_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

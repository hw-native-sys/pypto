# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Tests for the ``MaterializeCommDomainScopes`` pass.

The pass walks each ``host_orch`` function, traces
``pld.tensor.alloc_window_buffer → pld.tensor.window → dispatch(device=r)``
chains, and:

* constructs one :class:`ir.WindowBuffer` per alloc,
* rewrites every ``pld.tensor.window`` result Var so its
  :class:`ir.DistributedTensorType` carries a ``window_buffer`` back-reference,
* clusters allocs with the same device descriptor into a single comm domain and
  wraps the host_orch body in nested :class:`ir.CommDomainScopeStmt` nodes
  (outer = first declared domain, inner = last).

The tests below run the late host-distributed materialization sequence directly
on a parsed program (via ``passes.synthesize_allreduce_signals()`` followed by
``passes.materialize_comm_domain_scopes()``). The materialized output products have
no full *print/parse* surface syntax — the printer emits a comment-only
descriptor of each comm-domain scope, and the parser doesn't reconstruct the
scope at all — so a whole-``@pl.program`` ``Expected`` parsed from Python
source would always carry no scope-stmt wrapping and ``window_buffer``-less
view types, mismatching the pass output.

The Before/Expected ``assert_structural_equal`` pattern is therefore applied at
the granularity of the pass's structurally-comparable output products: the
``devices`` list and ``slots`` vector of each emitted
:class:`ir.CommDomainScopeStmt`. Slot ``WindowBuffer``s are hand-built from
the pass's documented semantics (device-descriptor table + slot/alloc-order
rules in ``docs/en/dev/passes/37-materialize_comm_domain_scopes.md``) and compared with
``enable_auto_mapping=True`` so freshly-constructed Vars match the
pass-produced ones by structural isomorphism rather than identity.

``test_no_alloc_window_buffer_no_op`` is the whole-program exception: it
produces no scope wrapping and no rewritten view types, so it uses
``assert_structural_equal`` on the entire program. Error-branch tests assert via
``pytest.raises`` — the malformed-input "after" is a ``pypto::ValueError``.
"""

import ast
import textwrap
from typing import cast

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.ir.op.distributed import tensor_ops as dist_tensor_ops
from pypto.pypto_core import DataType, ir, passes


@pytest.fixture(autouse=True)
def _basic_verification_context():
    """Override the ``ut/conftest.py`` autouse fixture to run with
    BEFORE_AND_AFTER property verification but no print/parse roundtrip.

    The pass's output materialises ``CommDomainScopeStmt`` wrappers and
    ``DistributedTensorType.window_buffer_`` back-references on view Vars,
    but the printer / parser pair does not roundtrip either — so the
    roundtrip-symmetric check would fail every iteration despite the
    in-memory IR being correct. Property verification still runs.
    """
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _as_call(expr: ir.Expr) -> ir.Call:
    assert isinstance(expr, ir.Call)
    return expr


def _as_var(expr: ir.Expr) -> ir.Var:
    assert isinstance(expr, ir.Var)
    return expr


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
        if isinstance(stmt, ir.ScopeStmt):
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


def _get_comm_domain_scopes(func: ir.Function) -> list[ir.CommDomainScopeStmt]:
    """Walk ``func.body`` and return every ``CommDomainScopeStmt`` it carries,
    in nesting order (outermost first, innermost last)."""
    scopes: list[ir.CommDomainScopeStmt] = []

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.CommDomainScopeStmt):
            scopes.append(stmt)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        if isinstance(stmt, ir.ForStmt):
            walk(stmt.body)
        if isinstance(stmt, ir.ScopeStmt):
            walk(stmt.body)

    walk(func.body)
    return scopes


def _flatten_stmts(stmt: ir.Stmt) -> list[ir.Stmt]:
    result = [stmt]
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            result.extend(_flatten_stmts(child))
    if isinstance(stmt, ir.ScopeStmt):
        result.extend(_flatten_stmts(stmt.body))
    if isinstance(stmt, ir.ForStmt):
        result.extend(_flatten_stmts(stmt.body))
    return result


def _apply(program: ir.Program) -> ir.Program:
    return cast(ir.Program, passes.materialize_comm_domain_scopes()(_synthesize(program)))


def _synthesize(program: ir.Program) -> ir.Program:
    return cast(ir.Program, passes.synthesize_allreduce_signals()(program))


def _mul(left: int, right: int) -> ir.Expr:
    """Build ``Mul(ConstInt(left, INDEX), ConstInt(right, INDEX))``.

    Matches the expression the parser produces when a ``@pl.program`` body
    contains ``N * pl.<DType>.get_byte()`` (e.g. ``256 * pl.FP32.get_byte()``
    → ``256 * 4``, a ``Mul`` of two ``ConstInt`` with dtype ``INDEX``).
    """
    span = ir.Span.unknown()
    return ir.Mul(
        ir.ConstInt(left, DataType.INDEX, span),
        ir.ConstInt(right, DataType.INDEX, span),
        DataType.INDEX,
        span,
    )


def _expected_slot(name: str, size: ir.Expr) -> ir.WindowBuffer:
    """Hand-build the WindowBuffer the pass should mint for one alloc.

    Each ``pld.alloc_window_buffer(size, *, name)`` materialises
    ``WindowBuffer(base=Var(name, PtrType), size=size, load_from_host=False,
    store_to_host=False)`` with ``name_hint`` inherited from the base Ptr Var.
    The ``size`` is the alloc's first arg passed through unchanged — the DSL
    emits it as a ``ConstInt`` (bare literal) or a ``Mul`` of two ``ConstInt``
    (``N * dtype.get_byte()`` form), always with dtype ``index``.
    """
    base = ir.Var(name, ir.PtrType(), ir.Span.unknown())
    return ir.WindowBuffer(base, size)


def _assert_scope_fields(
    actual: ir.CommDomainScopeStmt,
    expected_devices: list[int],
    expected_slots: list[ir.WindowBuffer],
) -> None:
    """Compare a produced ``CommDomainScopeStmt`` against hand-derived fields.

    ``enable_auto_mapping=True`` lets the freshly-built slot Vars in
    ``expected_slots`` match the pass-produced ones by structural isomorphism
    (name_hint + size + flags) rather than by ``shared_ptr`` identity.
    """
    assert list(actual.devices) == expected_devices
    assert len(actual.slots) == len(expected_slots)
    for actual_slot, expected_slot in zip(actual.slots, expected_slots, strict=True):
        ir.assert_structural_equal(actual_slot, expected_slot, enable_auto_mapping=True)


def _assert_rank2_signal_type(value: ir.Expr, world_size_name: str) -> None:
    signal_type = value.type
    assert isinstance(signal_type, ir.DistributedTensorType)
    assert len(signal_type.shape) == 2
    assert isinstance(signal_type.shape[0], ir.Var)
    assert signal_type.shape[0].name_hint == world_size_name
    assert isinstance(signal_type.shape[1], ir.ConstInt)
    assert signal_type.shape[1].value == 1


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
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")

    # The world_size loop bound resolves the device descriptor to kAll, encoded
    # on the wire as an empty ``devices`` list, with a single slot for ``buf``.
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    _assert_scope_fields(scopes[0], [], [_expected_slot("buf", _mul(256, 4))])

    # The view's window_buffer back-reference now points to the (same) slot.
    # Pointer-identity between the scope's slot and the view type's
    # window_buffer is a load-bearing invariant that structural comparison
    # alone cannot express.
    wb = scopes[0].slots[0]
    view_types = _view_var_types(host)
    assert len(view_types) == 1
    assert view_types[0].window_buffer is wb


def test_allreduce_signal_inherits_data_comm_domain():
    """Signal buffers used only by pld.tensor.allreduce still become scope slots."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    _assert_scope_fields(
        scopes[0], [], [_expected_slot("data_buf", _mul(256, 4)), _expected_slot("signal_buf", _mul(4, 4))]
    )


def test_explicit_allreduce_assignment_keeps_user_signal():
    """Direct explicit-signal assignments stay explicit and join the data domain."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    synthetic_assigns = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduces = [
        stmt.value
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.tensor.allreduce"
    ]

    assert synthetic_assigns == []
    assert len(allreduces) == 1
    assert len(allreduces[0].args) == 2
    assert _as_var(allreduces[0].args[1]).name_hint == "signal"

    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    assert [slot.base.name_hint for slot in scopes[0].slots] == ["data_buf", "signal_buf"]


def test_implicit_allreduce_signal_is_materialized_in_data_comm_domain():
    """A host allreduce without signal gets a private signal slot in the data domain."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    signal_windows = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.tensor.window"
        and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduces = [
        stmt.value
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.tensor.allreduce"
    ]

    assert len(signal_windows) == 1
    assert len(allreduces) == 1
    assert len(allreduces[0].args) == 2
    assert _as_var(allreduces[0].args[1]).name_hint == signal_windows[0].var.name_hint

    _assert_rank2_signal_type(signal_windows[0].var, "__allreduce_signal_world_size_0")

    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    assert [slot.base.name_hint for slot in scopes[0].slots] == [
        "data_buf",
        "__allreduce_signal_buf_0",
    ]


def test_optional_signal_allreduce_round_trips_before_materialization():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    printed = P.as_python()
    reparsed = cast(ir.Program, pl.parse(printed))
    host = _get_func(reparsed, "host_orch")
    allreduces = [
        stmt.value
        for stmt in _flatten_stmts(host.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.tensor.allreduce"
    ]

    assert "pld.tensor.allreduce(" in printed
    assert "op=" in printed
    assert "__allreduce_signal_" not in printed
    assert len(allreduces) == 1
    allreduce = _as_call(allreduces[0])
    assert len(allreduce.args) == 1
    assert allreduce.kwargs["op"] == int(pld.ReduceOp.Sum)


def test_synthesize_allreduce_signals_normalizes_host_allreduce():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    synthesized = _synthesize(P)
    host = _get_func(synthesized, "host_orch")
    stmts = _flatten_stmts(host.body)
    allreduces = [
        stmt.value
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.tensor.allreduce"
    ]
    signal_windows = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.tensor.window"
        and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]

    assert len(allreduces) == 1
    assert len(allreduces[0].args) == 2
    assert len(signal_windows) == 1
    assert _as_var(allreduces[0].args[1]).name_hint == signal_windows[0].var.name_hint


def test_synthesized_allreduce_signal_round_trips_after_materialization():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    materialized = _apply(P)
    printed = materialized.as_python()
    reparsed = cast(ir.Program, pl.parse(printed))
    host = _get_func(reparsed, "host_orch")
    stmts = _flatten_stmts(host.body)
    allocs = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.tensor.alloc_window_buffer"
        and stmt.var.name_hint.startswith("__allreduce_signal_buf_")
    ]
    windows = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.tensor.window"
        and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduces = [
        stmt.value
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.tensor.allreduce"
    ]

    printed_ast = ast.parse(printed)
    alloc_call_nodes = [
        node
        for node in ast.walk(printed_ast)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "alloc_window_buffer"
    ]
    assert alloc_call_nodes
    for node in alloc_call_nodes:
        assert "name" not in {keyword.arg for keyword in node.keywords}
    assert len(allocs) == 1
    alloc_call = _as_call(allocs[0].value)
    assert alloc_call.kwargs["name"] == allocs[0].var.name_hint
    assert len(windows) == 1
    window_call = _as_call(windows[0].value)
    assert _as_var(window_call.args[0]).name_hint == allocs[0].var.name_hint
    signal_type = windows[0].var.type
    assert isinstance(signal_type, ir.DistributedTensorType)
    assert len(signal_type.shape) == 2
    assert isinstance(signal_type.shape[0], ir.Var)
    assert signal_type.shape[0].name_hint.startswith("__allreduce_signal_world_size_")
    assert isinstance(signal_type.shape[1], ir.ConstInt)
    assert signal_type.shape[1].value == 1
    assert len(allreduces) == 1
    assert len(allreduces[0].args) == 2
    assert _as_var(allreduces[0].args[1]).name_hint == windows[0].var.name_hint


def test_implicit_allreduce_signal_names_are_program_unique():
    program = cast(
        ir.Program,
        pl.parse(
            textwrap.dedent(
                """
            @pl.program
            class P:
                @pl.function(type=pl.FunctionType.Orchestration)
                def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
                    return data

                @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
                def host_orch_a(self):
                    data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
                    __allreduce_signal_world_size_0 = pld.world_size()
                    __allreduce_signal_buf_0 = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())
                    __allreduce_signal_0 = pld.window(
                        __allreduce_signal_buf_0,
                        [pld.world_size(), 1],
                        dtype=pl.INT32,
                    )
                    __allreduce_result_0 = pld.window(data_buf, [256], dtype=pl.FP32)
                    data = pld.window(data_buf, [256], dtype=pl.FP32)
                    for r in pl.range(pld.world_size()):
                        self.chip_orch(data, device=r)
                    data = pld.tensor.allreduce(data, __allreduce_signal_0, op=pld.ReduceOp.Sum)
                    data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
                    return data

                @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
                def host_orch_b(self):
                    data_buf_b = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
                    data = pld.window(data_buf_b, [256], dtype=pl.FP32)
                    for r in pl.range(pld.world_size()):
                        self.chip_orch(data, device=r)
                    data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
                    return data
            """
            )
        ),
    )

    result = _apply(program)
    hosts = [_get_func(result, "host_orch_a"), _get_func(result, "host_orch_b")]
    allreduce_signal_args = [
        _as_var(_as_call(stmt.value).args[1]).name_hint
        for host in hosts
        for stmt in _flatten_stmts(host.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.tensor.allreduce"
    ]
    world_size_vars = [
        stmt.var.name_hint
        for host in hosts
        for stmt in _flatten_stmts(host.body)
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("__allreduce_signal_world_size_")
    ]
    signal_slots = [
        slot.base.name_hint
        for host in hosts
        for scope in _get_comm_domain_scopes(host)
        for slot in scope.slots
        if slot.base.name_hint.startswith("__allreduce_signal_buf_")
    ]

    assert allreduce_signal_args == [
        "__allreduce_signal_0",
        "__allreduce_signal_1",
        "__allreduce_signal_2",
    ]
    assert world_size_vars == [
        "__allreduce_signal_world_size_0",
        "__allreduce_signal_world_size_1",
        "__allreduce_signal_world_size_2",
    ]
    assert signal_slots == [
        "__allreduce_signal_buf_0",
        "__allreduce_signal_buf_1",
        "__allreduce_signal_buf_2",
    ]
    assert len(signal_slots) == len(set(signal_slots))


def test_consecutive_implicit_allreduces_get_fresh_signal_slots():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    result = _apply(P)
    host = _get_func(result, "host_orch")
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    assert [slot.base.name_hint for slot in scopes[0].slots] == [
        "data_buf",
        "__allreduce_signal_buf_0",
        "__allreduce_signal_buf_1",
    ]


def test_synthesize_allreduce_signals_reserves_existing_alloc_names():
    span = ir.Span.unknown()
    existing_alloc = dist_tensor_ops.alloc_window_buffer(
        16,
        name="__allreduce_signal_buf_0",
        span=span,
    )
    existing_buf = ir.Var("user_named_buf", existing_alloc.type, span)
    existing_assign = ir.AssignStmt(existing_buf, existing_alloc, span)

    data_type = ir.DistributedTensorType([ir.ConstInt(256, DataType.INT64, span)], DataType.FP32)
    data = ir.Var("data", data_type, span)
    allreduce = dist_tensor_ops.allreduce(data, op=ir.ReduceOp.Sum, span=span)
    allreduce_assign = ir.AssignStmt(ir.Var("data_out", allreduce.type, span), allreduce, span)

    host = ir.Function(
        "host_orch",
        [data],
        [],
        ir.SeqStmts([existing_assign, allreduce_assign], span),
        span,
        ir.FunctionType.Opaque,
        ir.Level.HOST,
        ir.Role.Orchestrator,
    )
    synthesized = _synthesize(ir.Program([host], "P", span))
    alloc_names = [
        stmt.value.kwargs["name"]
        for stmt in _flatten_stmts(_get_func(synthesized, "host_orch").body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.tensor.alloc_window_buffer"
    ]

    assert alloc_names == ["__allreduce_signal_buf_0", "__allreduce_signal_buf_1"]


def test_ssa_allreduce_result_keeps_source_window_lineage():
    """Post-SSA allreduce result Vars still resolve to the original data window."""

    program = cast(
        ir.Program,
        pl.parse(
            textwrap.dedent(
                """
                @pl.program
                class P:
                    @pl.function(type=pl.FunctionType.Orchestration)
                    def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
                        return data

                    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
                    def host_orch(self):
                        data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
                        signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
                        data = pld.window(data_buf, [256], dtype=pl.FP32)
                        signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
                        for r in pl.range(pld.world_size()):
                            self.chip_orch(data, device=r)
                        data_1 = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
                        data_2 = pld.tensor.allreduce(data_1, op=pld.ReduceOp.Sum)
                        return data_2
                """
            )
        ),
    )

    result = _apply(program)
    host = _get_func(result, "host_orch")
    scopes = _get_comm_domain_scopes(host)

    assert len(scopes) == 1
    assert [slot.base.name_hint for slot in scopes[0].slots] == [
        "data_buf",
        "signal_buf",
        "__allreduce_signal_buf_0",
    ]


def test_return_implicit_allreduce_is_lifted_for_host_lowering():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    allreduce_assigns = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.tensor.allreduce"
    ]
    returns = [stmt for stmt in stmts if isinstance(stmt, ir.ReturnStmt)]

    assert len(allreduce_assigns) == 1
    assert allreduce_assigns[0].var.name_hint.startswith("__allreduce_result_")
    allreduce_call = _as_call(allreduce_assigns[0].value)
    assert len(allreduce_call.args) == 2
    assert _as_var(allreduce_call.args[1]).name_hint == "__allreduce_signal_0"
    _assert_rank2_signal_type(allreduce_call.args[1], "__allreduce_signal_world_size_0")
    assert len(returns) == 1
    assert isinstance(returns[0].value[0], ir.Var)
    assert returns[0].value[0] is allreduce_assigns[0].var

    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    assert [slot.base.name_hint for slot in scopes[0].slots] == [
        "data_buf",
        "__allreduce_signal_buf_0",
    ]


def test_return_explicit_allreduce_is_lifted_for_host_lowering():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    synthetic_assigns = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduce_assigns = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.tensor.allreduce"
    ]
    returns = [stmt for stmt in stmts if isinstance(stmt, ir.ReturnStmt)]

    assert synthetic_assigns == []
    assert len(allreduce_assigns) == 1
    assert allreduce_assigns[0].var.name_hint.startswith("__allreduce_result_")
    assert _as_var(_as_call(allreduce_assigns[0].value).args[1]).name_hint == "signal"
    assert len(returns) == 1
    assert isinstance(returns[0].value[0], ir.Var)
    assert returns[0].value[0] is allreduce_assigns[0].var


def test_implicit_allreduce_eval_stmt_gets_signal():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    allreduces = [
        stmt.expr
        for stmt in stmts
        if isinstance(stmt, ir.EvalStmt)
        and isinstance(stmt.expr, ir.Call)
        and stmt.expr.op.name == "pld.tensor.allreduce"
    ]

    assert len(allreduces) == 1
    assert len(allreduces[0].args) == 2
    assert _as_var(allreduces[0].args[1]).name_hint == "__allreduce_signal_0"
    _assert_rank2_signal_type(allreduces[0].args[1], "__allreduce_signal_world_size_0")

    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    assert [slot.base.name_hint for slot in scopes[0].slots] == [
        "data_buf",
        "__allreduce_signal_buf_0",
    ]


def test_explicit_allreduce_eval_stmt_keeps_user_signal():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    synthetic_assigns = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduces = [
        stmt.expr
        for stmt in stmts
        if isinstance(stmt, ir.EvalStmt)
        and isinstance(stmt.expr, ir.Call)
        and stmt.expr.op.name == "pld.tensor.allreduce"
    ]

    assert synthetic_assigns == []
    assert len(allreduces) == 1
    assert _as_var(allreduces[0].args[1]).name_hint == "signal"


def test_implicit_allreduce_in_loop_is_rejected():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            for _ in pl.range(2):
                data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    with pytest.raises(Exception, match="allreduce is not supported inside a for/while loop"):
        _apply(P)


def test_implicit_allreduce_in_while_loop_is_rejected():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            while True:
                data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    with pytest.raises(Exception, match="allreduce is not supported inside a for/while loop"):
        _apply(P)


def test_explicit_allreduce_in_loop_is_rejected():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            for _ in pl.range(2):
                data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    with pytest.raises(Exception, match="allreduce is not supported inside a for/while loop"):
        _apply(P)


def test_explicit_allreduce_in_while_loop_is_rejected():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            while True:
                data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    with pytest.raises(Exception, match="allreduce is not supported inside a for/while loop"):
        _apply(P)


def test_nested_explicit_allreduce_expression_is_rejected():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            wrapped = (pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum),)
            return wrapped

    with pytest.raises(Exception, match="must be a direct assignment"):
        _apply(P)


def test_nested_implicit_allreduce_expression_is_rejected():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            wrapped = (pld.tensor.allreduce(data, op=pld.ReduceOp.Sum),)
            return wrapped

    with pytest.raises(Exception, match="must be a direct assignment"):
        _apply(P)


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
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            self.chip_orch(data, device=0)
            self.chip_orch(data, device=1)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    # Two ConstInt dispatches contribute {0} and {1}; merged into subset {0,1}
    # over a single ``buf`` slot.
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    _assert_scope_fields(scopes[0], [0, 1], [_expected_slot("buf", _mul(256, 4))])


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
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(2):
                self.chip_orch(data, device=r)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    # ``pl.range(2)`` expands the induction-var descriptor to subset {0, 1}.
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    _assert_scope_fields(scopes[0], [0, 1], [_expected_slot("buf", _mul(256, 4))])


# ---------------------------------------------------------------------------
# Two allocs / same descriptor → one scope, two slots in alloc order
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
            buf_data = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            buf_signal = pld.alloc_window_buffer(8 * pl.INT32.get_byte())
            data = pld.window(buf_data, [256], dtype=pl.FP32)
            signal = pld.window(buf_signal, [8], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    # Both allocs are dispatched over the same world_size loop ⇒ identical kAll
    # descriptor ⇒ a single scope whose slots follow source/alloc order
    # (buf_data, then buf_signal) per Phase-6 clustering.
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    _assert_scope_fields(
        scopes[0],
        [],  # kAll
        [_expected_slot("buf_data", _mul(256, 4)), _expected_slot("buf_signal", _mul(8, 4))],
    )


# ---------------------------------------------------------------------------
# Two allocs / different descriptors → two nested scopes
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
            buf_a = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            buf_b = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            a = pld.window(buf_a, [256], dtype=pl.FP32)
            b = pld.window(buf_b, [256], dtype=pl.FP32)
            self.chip_orch_a(a, device=0)
            self.chip_orch_a(a, device=1)
            self.chip_orch_b(b, device=2)
            self.chip_orch_b(b, device=3)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    # buf_a is dispatched to {0,1}; buf_b to {2,3}. Distinct descriptors ⇒ two
    # nested scopes. The pass walks allocs in source order and opens a domain
    # on first descriptor mismatch, so scope order follows alloc order:
    # outermost = buf_a, innermost = buf_b.
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 2
    _assert_scope_fields(scopes[0], [0, 1], [_expected_slot("buf_a", _mul(256, 4))])
    _assert_scope_fields(scopes[1], [2, 3], [_expected_slot("buf_b", _mul(256, 4))])


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
            buf = pld.alloc_window_buffer(512 * pl.FP32.get_byte())
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
    # And that same WindowBuffer is the (single) slot of the (single) scope.
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    assert len(scopes[0].slots) == 1
    assert scopes[0].slots[0] is wb_a


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
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
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
# Dead alloc — alloc with no pld.tensor.window materialisation
# ---------------------------------------------------------------------------


def test_dead_alloc_no_window_materialisation_raises():
    """An alloc with no ``pld.tensor.window`` view is a dead allocation.

    Phase-3 sanity check (source ``CHECK(!allocs_with_windows[...].empty())``,
    doc "Sanity checks" bullet 1): downstream codegen would have nothing to
    point a comm-domain buffer slot at, so the pass rejects it. This is a
    distinct branch from ``test_dead_alloc_no_dispatch_raises`` (which has a
    view but no consuming dispatch).
    """

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())  # noqa: F841  # never windowed
            return 0

    with pytest.raises(Exception, match=r"no pld\.tensor\.window materialisation"):
        _apply(P)


# ---------------------------------------------------------------------------
# Dead alloc — alloc + window but no dispatch consumer
# ---------------------------------------------------------------------------


def test_dead_alloc_no_dispatch_raises():
    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            _ = pld.window(buf, [256], dtype=pl.FP32)
            return 0

    with pytest.raises(Exception, match="not consumed by any chip_orch dispatch"):
        _apply(P)


# ---------------------------------------------------------------------------
# Unsupported device= — induction var over a non-unit-step loop
# ---------------------------------------------------------------------------


def test_non_unit_step_device_loop_raises():
    """``device=r`` over ``pl.range(0, 4, 2)`` is rejected.

    The device resolver only supports unit-step ``pl.range`` induction vars
    (source ``CHECK(step == 1)``, doc device-descriptor table "other ⇒
    ValueError"). A stride-2 loop has no well-defined contiguous coverage, so
    the pass raises rather than guessing.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(0, 4, 2):
                self.chip_orch(data, device=r)
            return 0

    with pytest.raises(Exception, match="non-unit-step loop is not supported"):
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
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return 0

    first = _apply(P)
    second = _apply(first)
    first_host = _get_func(first, "host_orch")
    second_host = _get_func(second, "host_orch")
    first_scopes = _get_comm_domain_scopes(first_host)
    second_scopes = _get_comm_domain_scopes(second_host)
    assert len(first_scopes) == 1
    assert len(second_scopes) == 1
    # Same device coverage; same number of slots.
    assert list(first_scopes[0].devices) == list(second_scopes[0].devices)
    assert len(first_scopes[0].slots) == len(second_scopes[0].slots)


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
    # No alloc_window_buffer chains ⇒ the pass emits no scope wrapping and
    # rewrites nothing, so the program is unchanged.
    ir.assert_structural_equal(After, Before)


# ---------------------------------------------------------------------------
# Loop-bound through SSA temp (regression for the second N4 bug)
# ---------------------------------------------------------------------------


def test_loop_bound_via_assigned_temp_world_size():
    """``n = pld.world_size(); for r in pl.range(n): ... device=r`` ⇒ kAll.

    Mirrors the post-ConvertToSSA / NormalizeStmtStructure shape where
    ``pl.range(pld.world_size())`` has been CSE-hoisted into a named temp.
    The pass must follow the AssignStmt def chain back to the
    ``pld.system.world_size`` Call, otherwise the dispatch is rejected.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            n = pld.world_size()
            for r in pl.range(n):
                self.chip_orch(data, device=r)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    assert list(scopes[0].devices) == [], "world_size loop bound must resolve to kAll"


def test_loop_bound_via_assigned_temp_const_int():
    """``n = 2; for r in pl.range(n): ... device=r`` ⇒ subset {0, 1}.

    Same indirection as the world_size case but with a ConstInt at the
    end of the def chain — exercises the integer-bound branch of
    ``UnwrapStopExpr``.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            n = 2
            for r in pl.range(n):
                self.chip_orch(data, device=r)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    assert list(scopes[0].devices) == [0, 1]


# ---------------------------------------------------------------------------
# CommDomainScopesMaterialized property verifier
# ---------------------------------------------------------------------------


def _build_pass_output_with_two_groups() -> ir.Program:
    """Run the pass on a 2-domain program; return its (uniqueness-correct) output."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf_a = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            buf_b = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data_a = pld.window(buf_a, [256], dtype=pl.FP32)
            data_b = pld.window(buf_b, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data_a, device=r)
            self.chip_orch(data_b, device=0)
            return 0

    return _apply(P)


def test_verifier_passes_on_pass_output():
    """Verifier emits no errors on the pass's own output (slot-uniqueness holds)."""
    program = _build_pass_output_with_two_groups()
    host = _get_func(program, "host_orch")
    assert len(_get_comm_domain_scopes(host)) == 2

    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.CommDomainScopesMaterialized)
    diagnostics = passes.PropertyVerifierRegistry.verify(props, program)
    errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
    assert errors == []


def test_verifier_flags_duplicate_slot_across_scopes():
    """A WindowBuffer reused across two CommDomainScopeStmts is rejected with a clear error."""
    program = _build_pass_output_with_two_groups()
    host = _get_func(program, "host_orch")
    outer, inner = _get_comm_domain_scopes(host)

    # Build a sibling scope that reuses the outer scope's slot while the outer
    # still owns it. The body is the original (substituted) body lifted out
    # from under ``inner`` — it's only present to satisfy the ScopeStmt ctor;
    # the verifier only inspects ``slots_``.
    duplicated = ir.CommDomainScopeStmt(
        list(outer.devices),
        [outer.slots[0]],
        "comm_d_dup",
        body=inner.body,
        span=outer.span,
    )

    # Splice the duplicate into the function body in place of the original
    # scope chain so the IR carries both ``outer`` (with the slot) and
    # ``duplicated`` (reusing it).
    bad_body = ir.SeqStmts([outer, duplicated], outer.span)
    bad_host_func = ir.Function(
        host.name,
        list(host.params),
        list(host.return_types),
        bad_body,
        host.span,
        type=host.func_type,
        level=host.level,
        role=host.role,
    )
    new_functions = [bad_host_func if f.name == host.name else f for f in program.functions.values()]
    bad_program = ir.Program(new_functions, program.name, program.span)

    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.CommDomainScopesMaterialized)
    diagnostics = passes.PropertyVerifierRegistry.verify(props, bad_program)
    errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
    assert len(errors) == 1
    assert "appears in multiple CommDomainScopeStmts" in errors[0].message
    assert errors[0].rule_name == "CommDomainScopesMaterialized"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

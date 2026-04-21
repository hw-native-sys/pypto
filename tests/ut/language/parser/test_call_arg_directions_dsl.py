# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Printer + parser coverage for the ``pl.adir.<dir>(...)`` DSL surface syntax.

``Call.arg_directions_`` is normally populated by the ``DeriveCallDirections``
pass and is invisible in the DSL surface syntax. To make the field round-trip
through ``python_print`` → ``parse``, the printer wraps each cross-function
call argument with ``pl.adir.<dir>(...)`` (an identity helper provided by
``pypto.language.arg_direction``). The parser then strips these wrappers and
restores ``arg_directions_`` on the rebuilt :class:`ir.Call`.

These tests pin down both halves of that contract independently of the
``DeriveCallDirections`` pass.
"""

from __future__ import annotations

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.language.arg_direction import DIRECTION_TO_NAME, NAME_TO_DIRECTION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_calls(program: ir.Function | ir.Program, callee_name: str) -> list[ir.Call]:
    """Collect every ``self.<callee_name>(...)`` Call in *program*."""
    found: list[ir.Call] = []

    class _Collector(ir.IRVisitor):
        def visit_call(self, op):
            if op.op.name == callee_name:
                found.append(op)
            super().visit_call(op)

    assert isinstance(program, ir.Program), "expected a Program, not a bare Function"
    _Collector().visit_program(program)
    return found


def _make_two_callsite_program() -> ir.Program:
    """A minimal program with one Orchestration ``main`` calling ``kernel`` once.

    ``kernel`` has signature ``(In tensor, Out tensor)`` so that after
    ``DeriveCallDirections`` the call site has directions
    ``[Input, OutputExisting]`` (param-rooted Out).
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
            ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
            return ret

        @pl.function
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            dst: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            r: pl.Tensor[[64], pl.FP32] = self.kernel(x, dst)
            return r

    return Prog


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------


class TestPrinterEmitsAdirWrappers:
    """``IRPythonPrinter`` wraps cross-function call args with ``pl.adir.<dir>(...)``."""

    def test_no_wrapper_when_arg_directions_empty(self):
        """Legacy / pre-derive Call objects must print bare arguments."""
        Prog = _make_two_callsite_program()
        # Sanity: the freshly parsed call has no derived directions.
        calls = _user_calls(Prog, "kernel")
        assert len(calls) == 1
        assert list(calls[0].arg_directions) == []

        printed = Prog.as_python()
        assert "self.kernel(x, dst)" in printed
        assert "pl.adir." not in printed

    def test_wrappers_emitted_after_derive(self):
        """Once ``DeriveCallDirections`` has run, every arg is wrapped."""
        Prog = _make_two_callsite_program()
        out = passes.derive_call_directions()(Prog)
        calls = _user_calls(out, "kernel")
        assert len(calls) == 1
        assert [d for d in calls[0].arg_directions] == [
            ir.ArgDirection.Input,
            ir.ArgDirection.OutputExisting,
        ]

        printed = out.as_python()
        # Both args are wrapped, in the correct order, with the matching helper.
        assert "self.kernel(pl.adir.input(x), pl.adir.output_existing(dst))" in printed

    def test_wrapper_name_table_is_consistent(self):
        """``DIRECTION_TO_NAME`` covers every enum value and matches the printer's choices."""
        # Bijection between names and enum values.
        assert {DIRECTION_TO_NAME[d] for d in ir.ArgDirection} == set(NAME_TO_DIRECTION)
        for name, direction in NAME_TO_DIRECTION.items():
            assert DIRECTION_TO_NAME[direction] == name


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParserStripsAdirWrappers:
    """The parser recognizes ``pl.adir.<dir>(inner)`` and recovers ``arg_directions``."""

    def test_parse_single_wrapper_populates_arg_directions(self):
        code = """
import pypto.language as pl

@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64], pl.FP32],
        out: pl.Out[pl.Tensor[[64], pl.FP32]],
    ) -> pl.Tensor[[64], pl.FP32]:
        t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
        ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
        return ret

    @pl.function
    def main(
        self,
        x: pl.Tensor[[64], pl.FP32],
        dst: pl.Tensor[[64], pl.FP32],
    ) -> pl.Tensor[[64], pl.FP32]:
        r: pl.Tensor[[64], pl.FP32] = self.kernel(pl.adir.input(x), pl.adir.output_existing(dst))
        return r
"""
        prog = pl.parse(code)
        calls = _user_calls(prog, "kernel")
        assert len(calls) == 1
        assert [d for d in calls[0].arg_directions] == [
            ir.ArgDirection.Input,
            ir.ArgDirection.OutputExisting,
        ]

    def test_parse_legacy_call_keeps_arg_directions_empty(self):
        """A call with no wrappers must yield an empty ``arg_directions`` (legacy form)."""
        code = """
import pypto.language as pl

@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64], pl.FP32],
        out: pl.Out[pl.Tensor[[64], pl.FP32]],
    ) -> pl.Tensor[[64], pl.FP32]:
        t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
        ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
        return ret

    @pl.function
    def main(
        self,
        x: pl.Tensor[[64], pl.FP32],
        dst: pl.Tensor[[64], pl.FP32],
    ) -> pl.Tensor[[64], pl.FP32]:
        r: pl.Tensor[[64], pl.FP32] = self.kernel(x, dst)
        return r
"""
        prog = pl.parse(code)
        calls = _user_calls(prog, "kernel")
        assert len(calls) == 1
        assert list(calls[0].arg_directions) == []

    def test_all_six_wrappers_round_trip_through_parse(self):
        """Each ``pl.adir.<name>`` helper resolves to its matching ``ArgDirection``."""
        code = """
import pypto.language as pl

@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64], pl.FP32],
        b: pl.Tensor[[64], pl.FP32],
        c: pl.Tensor[[64], pl.FP32],
        d: pl.Tensor[[64], pl.FP32],
        e: pl.Tensor[[64], pl.FP32],
        f: pl.Scalar[pl.INT64],
    ):
        t: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
        pl.store(t, [0], a)

    @pl.function
    def main(
        self,
        a: pl.Tensor[[64], pl.FP32],
        b: pl.Tensor[[64], pl.FP32],
        c: pl.Tensor[[64], pl.FP32],
        d: pl.Tensor[[64], pl.FP32],
        e: pl.Tensor[[64], pl.FP32],
        f: pl.Scalar[pl.INT64],
    ):
        self.kernel(
            pl.adir.input(a),
            pl.adir.output(b),
            pl.adir.inout(c),
            pl.adir.output_existing(d),
            pl.adir.no_dep(e),
            pl.adir.scalar(f),
        )
"""
        prog = pl.parse(code)
        calls = _user_calls(prog, "kernel")
        assert len(calls) == 1
        assert [d for d in calls[0].arg_directions] == [
            ir.ArgDirection.Input,
            ir.ArgDirection.Output,
            ir.ArgDirection.InOut,
            ir.ArgDirection.OutputExisting,
            ir.ArgDirection.NoDep,
            ir.ArgDirection.Scalar,
        ]

    def test_mixing_wrapped_and_bare_args_is_rejected(self):
        code = """
import pypto.language as pl

@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64], pl.FP32],
        out: pl.Out[pl.Tensor[[64], pl.FP32]],
    ) -> pl.Tensor[[64], pl.FP32]:
        t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
        ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
        return ret

    @pl.function
    def main(
        self,
        x: pl.Tensor[[64], pl.FP32],
        dst: pl.Tensor[[64], pl.FP32],
    ) -> pl.Tensor[[64], pl.FP32]:
        r: pl.Tensor[[64], pl.FP32] = self.kernel(pl.adir.input(x), dst)
        return r
"""
        with pytest.raises(Exception, match="mixes wrapped"):
            pl.parse(code)

    def test_unknown_direction_marker_is_rejected(self):
        code = """
import pypto.language as pl

@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
        ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], x)
        return ret

    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        r: pl.Tensor[[64], pl.FP32] = self.kernel(pl.adir.bogus(x))
        return r
"""
        with pytest.raises(Exception, match="bogus"):
            pl.parse(code)

    def test_wrapper_must_take_single_positional_arg(self):
        code = """
import pypto.language as pl

@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
        ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], x)
        return ret

    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        r: pl.Tensor[[64], pl.FP32] = self.kernel(pl.adir.input(x, x))
        return r
"""
        with pytest.raises(Exception, match="exactly one positional"):
            pl.parse(code)


# ---------------------------------------------------------------------------
# End-to-end round-trip
# ---------------------------------------------------------------------------


class TestAdirRoundTrip:
    """Print → parse → structural_equal preserves ``arg_directions`` on cross-function calls."""

    def test_round_trip_preserves_arg_directions(self):
        Prog = _make_two_callsite_program()
        derived = passes.derive_call_directions()(Prog)

        printed = derived.as_python()
        reparsed = pl.parse(printed)

        # Structural equality covers ``arg_directions_`` because Call::arg_directions_
        # is declared as ``UsualField`` in the IR reflection.
        ir.assert_structural_equal(derived, reparsed, enable_auto_mapping=True)

        # And the directions on the rebuilt call match the original ones explicitly.
        original_call = _user_calls(derived, "kernel")[0]
        rebuilt_call = _user_calls(reparsed, "kernel")[0]
        assert [d for d in rebuilt_call.arg_directions] == [d for d in original_call.arg_directions]

    def test_round_trip_legacy_program_remains_legacy(self):
        """A program that has not been derived stays free of ``pl.adir.*`` after a round-trip."""
        Prog = _make_two_callsite_program()
        printed = Prog.as_python()
        assert "pl.adir." not in printed

        reparsed = pl.parse(printed)
        ir.assert_structural_equal(Prog, reparsed, enable_auto_mapping=True)
        assert list(_user_calls(reparsed, "kernel")[0].arg_directions) == []

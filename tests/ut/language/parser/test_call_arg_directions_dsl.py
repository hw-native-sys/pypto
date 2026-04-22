# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Printer + parser coverage for the ``Call.attrs['arg_directions']`` DSL surface.

``Call.attrs['arg_directions']`` is normally populated by the
``DeriveCallDirections`` pass and is invisible in the DSL surface syntax. To
make the attr round-trip through ``python_print`` -> ``parse``, the printer
emits a trailing ``attrs={"arg_directions": [pl.adir.<dir>, ...]}`` keyword on
each cross-function call. The parser recognizes this keyword and restores
``arg_directions`` on the rebuilt :class:`ir.Call`.

The per-argument ``pl.adir.<dir>(arg)`` wrapper form is *not* supported on
either the printer or the parser side -- ``pl.adir.<name>`` symbols are bare
aliases of the matching :class:`ir.ArgDirection` enum value.

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


class TestPrinterEmitsAttrsKwarg:
    """``IRPythonPrinter`` surfaces ``arg_directions`` via a trailing ``attrs=`` keyword."""

    def test_no_attrs_when_arg_directions_empty(self):
        """Legacy / pre-derive Call objects must print bare arguments without ``attrs=``."""
        Prog = _make_two_callsite_program()
        # Sanity: the freshly parsed call has no derived directions.
        calls = _user_calls(Prog, "kernel")
        assert len(calls) == 1
        assert list(calls[0].arg_directions) == []

        printed = Prog.as_python()
        assert "self.kernel(x, dst)" in printed
        assert "attrs=" not in printed
        assert "pl.adir." not in printed

    def test_attrs_kwarg_emitted_after_derive(self):
        """Once ``DeriveCallDirections`` has run, the call carries an ``attrs=`` kwarg."""
        Prog = _make_two_callsite_program()
        out = passes.derive_call_directions()(Prog)
        calls = _user_calls(out, "kernel")
        assert len(calls) == 1
        assert [d for d in calls[0].arg_directions] == [
            ir.ArgDirection.Input,
            ir.ArgDirection.OutputExisting,
        ]

        printed = out.as_python()
        assert (
            'self.kernel(x, dst, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})'
        ) in printed
        # Per-argument wrapper form is no longer emitted by the printer.
        assert "pl.adir.input(x)" not in printed

    def test_wrapper_name_table_is_consistent(self):
        """``DIRECTION_TO_NAME`` covers every enum value and matches the printer's choices."""
        # Bijection between names and enum values.
        assert {DIRECTION_TO_NAME[d] for d in ir.ArgDirection} == set(NAME_TO_DIRECTION)
        for name, direction in NAME_TO_DIRECTION.items():
            assert DIRECTION_TO_NAME[direction] == name

    def test_printer_emits_each_direction_marker(self):
        """The printer's ``pl.adir.<name>`` output covers every ``ArgDirection`` variant.

        Build the IR via the parser (using the supported ``attrs=`` kwarg form so
        every direction is exercised), then re-print it and assert that the
        emitted ``attrs={"arg_directions": [...]}`` lists each marker by its
        canonical name. This guards the printer/parser contract end-to-end and
        independently of ``DeriveCallDirections``.
        """
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
            a, b, c, d, e, f,
            attrs={"arg_directions": [
                pl.adir.input,
                pl.adir.output,
                pl.adir.inout,
                pl.adir.output_existing,
                pl.adir.no_dep,
                pl.adir.scalar,
            ]},
        )
"""
        prog = pl.parse(code)

        # Sanity: the parsed Call carries all six directions in order.
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

        printed = prog.as_python()
        assert (
            'self.kernel(a, b, c, d, e, f, attrs={"arg_directions": ['
            "pl.adir.input, pl.adir.output, pl.adir.inout, "
            "pl.adir.output_existing, pl.adir.no_dep, pl.adir.scalar]})"
        ) in printed
        # The wrapper form must never appear in printer output.
        for name in NAME_TO_DIRECTION:
            assert f"pl.adir.{name}(" not in printed


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParserExtractsAttrsKwarg:
    """The parser recognizes ``attrs={"arg_directions": [...]}`` and restores the vector."""

    def test_attrs_kwarg_populates_arg_directions(self):
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
        r: pl.Tensor[[64], pl.FP32] = self.kernel(
            x, dst, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]}
        )
        return r
"""
        prog = pl.parse(code)
        calls = _user_calls(prog, "kernel")
        assert len(calls) == 1
        assert [d for d in calls[0].arg_directions] == [
            ir.ArgDirection.Input,
            ir.ArgDirection.OutputExisting,
        ]

    def test_attrs_kwarg_unknown_marker_rejected(self):
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
        r: pl.Tensor[[64], pl.FP32] = self.kernel(x, attrs={"arg_directions": [pl.adir.bogus]})
        return r
"""
        with pytest.raises(Exception, match="bogus"):
            pl.parse(code)

    def test_attrs_kwarg_size_mismatch_rejected(self):
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
        r: pl.Tensor[[64], pl.FP32] = self.kernel(x, dst, attrs={"arg_directions": [pl.adir.input]})
        return r
"""
        with pytest.raises(Exception, match=r"(?i)length|match"):
            pl.parse(code)

    def test_attrs_kwarg_unknown_key_rejected(self):
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
        r: pl.Tensor[[64], pl.FP32] = self.kernel(x, attrs={"bogus": [pl.adir.input]})
        return r
"""
        with pytest.raises(Exception, match="bogus"):
            pl.parse(code)

    def test_other_keyword_args_still_rejected(self):
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
        r: pl.Tensor[[64], pl.FP32] = self.kernel(x, foo=1)
        return r
"""
        with pytest.raises(Exception, match="foo"):
            pl.parse(code)

    def test_per_argument_wrapper_form_no_longer_supported(self):
        """``pl.adir.<dir>(arg)`` is removed; the markers are not callable."""
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
        r: pl.Tensor[[64], pl.FP32] = self.kernel(pl.adir.input(x))
        return r
"""
        # The parser routes ``pl.adir.input(x)`` through the generic op-call
        # path, where it is rejected as an unsupported function call.
        with pytest.raises(Exception):  # noqa: B017, PT011
            pl.parse(code)

    def test_marker_alias_resolves_to_enum_value(self):
        """``pl.adir.<name>`` evaluates to the matching ``ArgDirection`` enum value."""
        from pypto.language import adir  # noqa: PLC0415

        assert adir.input is ir.ArgDirection.Input
        assert adir.output is ir.ArgDirection.Output
        assert adir.output_existing is ir.ArgDirection.OutputExisting
        assert adir.inout is ir.ArgDirection.InOut
        assert adir.no_dep is ir.ArgDirection.NoDep
        assert adir.scalar is ir.ArgDirection.Scalar


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

        # Structural equality covers ``arg_directions`` because ``Call::attrs_`` is
        # declared as ``UsualField`` in the IR reflection.
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

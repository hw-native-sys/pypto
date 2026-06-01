# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser coverage for ``pl.dump_tag(<name>)`` — the declarative per-tensor
selective tensor dump marker (simpler#844).

``pl.dump_tag(t)`` is a statement-position marker that records the bound Var;
every *subsequent* kernel dispatch consuming that exact Var gets it merged into
the dispatch's ``attrs['dump_vars']`` (the same attr the explicit ``dumps=``
kwarg writes). No IR statement is emitted and no Function-level attr is written —
the dump target is tracked by Var identity on the consuming Call / Submit nodes.
"""

from __future__ import annotations

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import ParserSyntaxError


def _kernel_calls(program: ir.Program, callee_name: str = "kernel") -> list[ir.Call]:
    """Collect every ``self.<callee_name>(...)`` Call in *program*."""
    found: list[ir.Call] = []

    class _Collector(ir.IRVisitor):
        def visit_call(self, op):
            if op.op.name == callee_name:
                found.append(op)
            super().visit_call(op)

    _Collector().visit_program(program)
    return found


def test_dump_tag_desugars_to_per_call_dump_vars() -> None:
    """Each ``pl.dump_tag(t)`` makes subsequent calls consuming ``t`` carry it
    in ``Call.attrs['dump_vars']`` (arg order, by Var identity)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
            r: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
            o: pl.Tensor[[16, 16], pl.FP32] = pl.store(r, [0, 0], output)
            return o

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
            d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            pl.dump_tag(a)
            pl.dump_tag(d)
            d = self.kernel(a, b, d)
            return d

    calls = _kernel_calls(P)
    assert len(calls) == 1
    assert "dump_vars" in calls[0].attrs
    names = {v.name_hint for v in calls[0].attrs["dump_vars"]}
    assert names == {"a", "d"}


def test_dump_tag_dedups_repeated_tags() -> None:
    """Marking the same Var twice merges it once into the call's dump_vars."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            o: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], output)
            return o

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            pl.dump_tag(a)
            pl.dump_tag(a)
            d = self.kernel(a, d)
            return d

    calls = _kernel_calls(P)
    assert len(calls) == 1
    names = [v.name_hint for v in calls[0].attrs["dump_vars"]]
    assert names == ["a"]


def test_dump_tag_is_forward_sticky() -> None:
    """A tag affects only *subsequent* calls — a call written before the marker
    does not carry the tagged Var."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            o: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], output)
            return o

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            c = self.kernel(a, c)  # before the tag — a not dumped here
            pl.dump_tag(a)
            d = self.kernel(a, d)  # after the tag — a dumped here
            return d

    calls = _kernel_calls(P)
    assert len(calls) == 2
    assert "dump_vars" not in calls[0].attrs
    assert {v.name_hint for v in calls[1].attrs["dump_vars"]} == {"a"}


def test_dump_tag_absent_when_unused() -> None:
    """No ``pl.dump_tag`` -> no ``dump_vars`` attr on the consuming call."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            o: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], output)
            return o

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            d = self.kernel(a, d)
            return d

    calls = _kernel_calls(P)
    assert len(calls) == 1
    assert "dump_vars" not in calls[0].attrs


def test_dump_tag_rejects_non_name_argument() -> None:
    """``pl.dump_tag(<attr/subscript/call>)`` is rejected with a clear error.
    Only bare variable names are valid — the codegen matches against IR Var
    base names, which are unambiguous for direct Name references but
    undefined for attribute / subscript / arbitrary expression arguments.
    """
    with pytest.raises(ParserSyntaxError, match="dump_tag.*bare variable name"):

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                o: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], output)
                return o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pl.dump_tag(self.kernel)  # type: ignore[arg-type]  # not a tensor Var
                d = self.kernel(a, d)
                return d

        _ = P


def test_dump_tag_rejects_too_many_args() -> None:
    """``pl.dump_tag(a, b)`` fails at the statement-position interceptor —
    exactly one positional arg is required."""
    with pytest.raises(ParserSyntaxError, match="dump_tag.*exactly one positional"):

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                o: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], output)
                return o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pl.dump_tag(a, b)  # two args
                d = self.kernel(a, d)
                return d

        _ = P


def test_dump_tag_rejects_zero_args() -> None:
    """``pl.dump_tag()`` fails at the statement-position interceptor —
    exactly one positional arg is required."""
    with pytest.raises(ParserSyntaxError, match="dump_tag.*exactly one positional"):

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                o: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], output)
                return o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pl.dump_tag()  # no args
                d = self.kernel(a, d)
                return d

        _ = P


def test_dump_tag_rejects_non_orch_scope() -> None:
    """``pl.dump_tag`` in a kernel (AIV/AIC/Mix) function body is a user error,
    not a silent no-op: the orchestration codegen never inspects non-orch
    function attrs, so the marker would have no effect. Raise at parse time
    so the mistake surfaces immediately."""
    with pytest.raises(ParserSyntaxError, match="dump_tag.*only valid inside an Orchestration"):

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pl.dump_tag(a)  # AIV body — not orch
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                o: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], output)
                return o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.kernel(a, d)
                return d

        _ = P


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

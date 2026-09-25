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
from pypto.language.parser.diagnostics.exceptions import ParserTypeError


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


def _tagged_spmd_as_tid() -> ir.Program:
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            pl.dump_tag(a)
            with pl.spmd(4) as tid:  # noqa: F841 — the capture selects the `as tid` form
                i = pl.tile.get_block_idx()
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                out = pl.store(t, [i * 128, 0], out)
            return out

    return P


def _tagged_spmd_as_tid_split() -> ir.Program:
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            pl.dump_tag(a)
            with pl.spmd(4, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]) as tid:  # noqa: F841
                i = pl.tile.get_block_idx()
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                out = pl.store(t, [i * 128, 0], out)
            return out

    return P


def _tagged_spmd_for() -> ir.Program:
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            pl.dump_tag(a)
            for i in pl.spmd(4, name_hint="stage1"):
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                out = pl.store(t, [i * 128, 0], out)
            return out

    return P


def _tagged_spmd_with() -> ir.Program:
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            pl.dump_tag(a)
            with pl.spmd(4):
                i = pl.tile.get_block_idx()
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                out = pl.store(t, [i * 128, 0], out)
            return out

    return P


@pytest.mark.parametrize(
    "build",
    [_tagged_spmd_as_tid, _tagged_spmd_as_tid_split, _tagged_spmd_for, _tagged_spmd_with],
    ids=["as_tid", "as_tid_split", "for", "with"],
)
def test_dump_tag_before_inline_spmd_survives_print_reparse(build) -> None:
    """A tag before an inline-body ``pl.spmd`` lands on the auto-synthesised
    InCore carrier. Every such form normally prints that carrier's header away
    (``for i in pl.spmd(...)`` / the inline ``as tid`` body), and ``pl.dump_tag``
    leaves no statement to re-print — so the printer must spell the carrier out
    as ``pl.at(level=pl.Level.CORE_GROUP, dumps=[a])`` for the mark to survive."""
    program = build()
    printed = program.as_python()
    assert "dumps=[a]" in printed, printed
    ir.assert_structural_equal(program, pl.parse_program(printed))


def _tagged_spmd_dispatch() -> ir.Program:
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            i = pl.tile.get_block_idx()
            t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
            out = pl.store(t, [i * 128, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            pl.dump_tag(a)
            with pl.spmd(4):
                out = self.kernel(a, out)
            return out

    return P


def _tagged_spmd_explicit_carrier() -> ir.Program:
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            pl.dump_tag(a)
            with pl.spmd(4):
                with pl.at(level=pl.Level.CORE_GROUP):
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(t, [i * 128, 0], out)
            return out

    return P


def _tagged_cluster() -> ir.Program:
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            b: pl.Tensor[[64, 64], pl.FP32],
            c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            pl.dump_tag(a)
            with pl.cluster():
                with pl.at(level=pl.Level.CORE_GROUP):
                    mm = pl.matmul(a, b, out_dtype=pl.FP32)
                    c = pl.add(mm, b)
            return c

    return P


def _tagged_graph() -> ir.Program:
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            pl.dump_tag(a)
            with pl.graph("g"):
                with pl.at(level=pl.Level.CORE_GROUP):
                    c = pl.add(a, a)
            return c

    return P


@pytest.mark.parametrize(
    "build",
    [_tagged_spmd_dispatch, _tagged_spmd_explicit_carrier, _tagged_cluster, _tagged_graph],
    ids=["spmd_dispatch", "spmd_explicit_carrier", "cluster", "graph"],
)
def test_dump_tag_before_container_scope_survives_print_reparse(build) -> None:
    """A tag before a ``pl.spmd`` whose body is a kernel dispatch or an explicit
    carrier, a ``pl.cluster``, or a ``pl.graph`` lands on that outer scope itself
    (each outlines to a dispatch of its own). The outer scope prints it back as
    its own ``dumps=[a]`` kwarg, so the mark survives print -> reparse."""
    program = build()
    outer = program.get_function("main").body.stmts[0]
    assert [v.name_hint for v in outer.attrs["dump_vars"]] == ["a"]
    printed = program.as_python()
    ir.assert_structural_equal(program, pl.parse_program(printed))


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


@pytest.mark.parametrize("capture", ["", " as tid"])
@pytest.mark.parametrize("optimizations", ["", ", optimizations=[pl.cross_core_slot(slot_num=2)]"])
def test_spmd_roundtrip_preserves_scoped_dump_without_tagging_next_scope(capture, optimizations):
    """Flattening an InCore must not drop its dump list or leak it to later scopes."""
    program = pl.parse(f"""
import pypto.language as pl

@pl.program
class P:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[16, 16], pl.FP32]):
        with pl.spmd(2){capture}:
            with pl.at(level=pl.Level.CORE_GROUP, dumps=[a]{optimizations}):
                i = pl.tile.get_block_idx()
                tile = pl.load(a, [i * 8, 0], [8, 16])
        with pl.spmd(2):
            with pl.at(level=pl.Level.CORE_GROUP):
                j = pl.tile.get_block_idx()
                other = pl.load(a, [j * 8, 0], [8, 16])
""")
    printed = ir.python_print(program)
    assert "dumps=[a]" in printed
    ir.assert_structural_equal(program, pl.parse(printed))


def test_cluster_scoped_dump_roundtrip():
    """A cluster's explicit dump list must not become a forward-sticky marker."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, a: pl.Tensor[[16, 16], pl.FP32]):
            with pl.cluster(dumps=[a]):
                with pl.at(level=pl.Level.CORE_GROUP):
                    _tile = pl.load(a, [0, 0], [16, 16])
            with pl.cluster():
                with pl.at(level=pl.Level.CORE_GROUP):
                    _other = pl.load(a, [0, 0], [16, 16])

    printed = ir.python_print(P)
    assert "pl.cluster(dumps=[a])" in printed
    ir.assert_structural_equal(P, pl.parse(printed))


@pytest.mark.parametrize(
    "dumps, message",
    [("[missing]", "unknown name"), ("[a, a]", "more than once"), ("[1]", "bare tensor names")],
)
def test_cluster_rejects_invalid_dump_list(dumps, message):
    with pytest.raises(ParserTypeError, match=message):
        pl.parse(f"""
import pypto.language as pl
@pl.program
class P:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[16, 16], pl.FP32]):
        with pl.cluster(dumps={dumps}):
            pass
""")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

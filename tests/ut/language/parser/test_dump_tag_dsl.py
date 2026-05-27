# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser coverage for ``pl.dump_tag(<name>)`` — the DSL marker that drives
the orchestration codegen's selective tensor dump (simpler#844).

The marker is a statement-position DSL call whose effect is captured on the
enclosing :class:`ir.Function`'s ``attrs["dump_tagged_names"]`` (a list of
Var name_hints) at parse time. No IR statement is emitted; the marker has no
runtime side effect.
"""

from __future__ import annotations

from typing import Any

import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics import ParserSyntaxError


def _get_orch(program) -> Any:
    """Return the orchestration ``Function`` from a parsed program."""
    from pypto import ir  # noqa: PLC0415

    for func in program.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            return func
    raise AssertionError("no orchestration function in program")


def test_dump_tag_attaches_names_to_orch_attrs() -> None:
    """Each ``pl.dump_tag(<name>)`` at orch scope adds the bound name to
    ``Function.attrs['dump_tagged_names']`` in source order."""

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

    orch = _get_orch(P)
    assert "dump_tagged_names" in orch.attrs, "parser failed to record dump_tag set"
    assert orch.attrs["dump_tagged_names"] == ["a", "d"]


def test_dump_tag_dedups_repeated_names() -> None:
    """Marking the same name twice records it once — the set is a name-keyed
    allowlist, not a multiset."""

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

    orch = _get_orch(P)
    assert orch.attrs["dump_tagged_names"] == ["a"]


def test_dump_tag_absent_when_unused() -> None:
    """No ``pl.dump_tag`` -> no attr on the Function. Codegen reads this as
    the legacy full-dump path."""

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

    orch = _get_orch(P)
    assert "dump_tagged_names" not in orch.attrs


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

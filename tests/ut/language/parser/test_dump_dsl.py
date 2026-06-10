# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser + printer coverage for the explicit ``dumps=`` selective-dump surface
(simpler#844).

``dumps=`` is the explicit dump kwarg, symmetric with ``deps=`` — accepted on
``pl.submit(...)`` and ``pl.at(...)``. It feeds the same IR attr
``attrs['dump_vars']`` (a ``vector<VarPtr>``) as a ``pl.dump_tag`` declaration,
so the dump target is tracked by Var identity (never by name). A plain
``self.kernel(...)`` call offers no ``dumps=`` surface; the printer surfaces a
Call's ``dump_vars`` (seeded by ``pl.dump_tag``) as a ``dumps=[...]`` kwarg, the
same way it surfaces a Submit's. The forward-sticky ``pl.dump_tag`` statement
itself is covered in ``test_dump_tag_dsl.py``.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import ParserTypeError


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


def _submit_nodes(program: ir.Program) -> list[ir.Submit]:
    """Collect every ``ir.Submit`` in *program* via the ``visit_submit`` hook."""
    found: list[ir.Submit] = []

    class _Collector(ir.IRVisitor):
        def visit_submit(self, op):
            found.append(op)
            super().visit_submit(op)

    _Collector().visit_program(program)
    return found


def test_submit_dumps_records_arg_vars_on_submit() -> None:
    """Each ``dumps=`` entry adds the arg Var to that Submit's
    ``attrs['dump_vars']`` — by identity, so the entries are the submit's own
    arg Vars."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.submit(self.kernel, a, b, dumps=[a, b])
            return out

    submits = _submit_nodes(P)
    assert len(submits) == 1
    assert "dump_vars" in submits[0].attrs
    names = {v.name_hint for v in submits[0].attrs["dump_vars"]}
    assert names == {"a", "b"}


def test_submit_dumps_absent_when_unused() -> None:
    """No ``dumps=`` kwarg -> no ``dump_vars`` attr on the Submit."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.submit(self.kernel, a)
            return out

    submits = _submit_nodes(P)
    assert len(submits) == 1
    assert "dump_vars" not in submits[0].attrs


def test_submit_dumps_dedups_repeated_arg() -> None:
    """Listing the same arg twice in ``dumps=`` records it once (dedup by
    Var identity)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.submit(self.kernel, a, dumps=[a, a])
            return out

    submits = _submit_nodes(P)
    assert len(submits) == 1
    assert list(submits[0].attrs["dump_vars"]) == [submits[0].args[0]]


def test_submit_dumps_roundtrips_through_printer() -> None:
    """``pl.submit(..., dumps=[x])`` is the submit-side selective dump surface.
    The parser records the listed args on the Submit's ``dump_vars`` and the
    printer round-trips them as a ``dumps=[...]`` kwarg."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.submit(self.kernel, a, dumps=[a])
            return out

    printed = P.as_python()
    assert "dumps=[a]" in printed, printed


def test_dump_tag_call_roundtrips_via_attrs_dict() -> None:
    """A ``pl.dump_tag`` seed on a plain ``self.kernel(...)`` Call is surfaced by
    the printer inside the machine-only ``attrs={"dump_vars": [...]}`` dict — a
    plain Call exposes no user-facing ``dumps=`` kwarg; the dump targets live in
    IR only and round-trip via the same dict as ``arg_directions``."""

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
            d = self.kernel(a, b, d)
            return d

    calls = _kernel_calls(P)
    assert len(calls) == 1
    assert {v.name_hint for v in calls[0].attrs["dump_vars"]} == {"a"}

    printed = P.as_python()
    assert '"dump_vars": [a]' in printed, printed
    # A plain Call exposes no user-facing dumps= kwarg and no arg wrapper.
    assert "dumps=[a]" not in printed, printed
    assert "pl.dump(" not in printed, printed


def test_at_dumps_records_dump_vars_on_scope() -> None:
    """``pl.at(..., dumps=[t])`` records ``t`` on the ScopeStmt's ``dump_vars``
    attr — the scope-side explicit surface, symmetric with ``deps=``."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, dumps=[a]):
                d = pl.add(a, a)
            return d

    printed = P.as_python()
    assert "dumps=[a]" in printed, printed


def test_at_dumps_rejects_non_tensor() -> None:
    """A ``pl.at(dumps=[...])`` entry that is not a tensor Var is rejected."""
    with pytest.raises(ParserTypeError, match="dumps=.*not a tensor"):

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                n: pl.Scalar[pl.INT32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, dumps=[n]):  # n is a scalar, not a tensor
                    d = pl.add(a, a)
                return d

        _ = P


def test_submit_dumps_rejects_non_arg() -> None:
    """A ``dumps=`` entry that is not a positional argument of the submit is
    rejected (strict arg-membership validation)."""
    with pytest.raises(ParserTypeError, match="dumps=.*not an argument of this submit"):

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
                return a

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                z: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                with pl.manual_scope():
                    out, _ = pl.submit(self.kernel, a, dumps=[z])  # z is not an arg of this submit
                return out

        _ = P


def test_dumps_rejected_on_plain_call() -> None:
    """``dumps=`` is only valid on ``pl.submit(...)`` / ``pl.at(...)``; a plain
    kernel call rejects it (declare the target with ``pl.dump_tag`` instead)."""
    with pytest.raises(ParserTypeError, match="does not accept keyword argument 'dumps'"):

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
                d = self.kernel(a, d, dumps=[a])  # type: ignore[call-arg]  # dumps= invalid on plain call
                return d

        _ = P


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

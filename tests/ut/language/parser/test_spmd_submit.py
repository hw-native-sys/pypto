# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser / IR / printer tests for the ``pl.spmd_submit(...)`` SPMD task-launch construct.

``pl.spmd_submit`` is the SPMD sibling of :func:`pl.submit`: it builds an
``ir.Submit`` carrying an SPMD launch spec (``core_num`` / ``sync_start``) on the
first-class ``Submit.core_num`` / ``Submit.sync_start`` fields, while keeping the
same ``(result, task_id)`` surface and ``deps=[...]`` wiring.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError, ParserTypeError


def _flatten(stmt):
    """Flatten a SeqStmts / RuntimeScopeStmt subtree into a flat statement list."""
    if isinstance(stmt, ir.SeqStmts):
        out = []
        for s in stmt.stmts:
            out.extend(_flatten(s))
        return out
    if isinstance(stmt, ir.RuntimeScopeStmt):
        return _flatten(stmt.body)
    return [stmt]


def _submits_in(stmt):
    """Collect every ``ir.Submit`` bound as an AssignStmt RHS in the subtree."""
    return [
        s.value for s in _flatten(stmt) if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Submit)
    ]


def _main_submits(prog) -> list:
    """Collect the Submit nodes in ``prog``'s ``main`` orchestration body."""
    fn = prog.get_function("main")
    assert fn is not None
    return _submits_in(fn.body)


def _two_kernel_program():
    """A program with a producer + consumer InCore kernel and an orchestrator."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
            self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            bi = pl.tile.get_block_idx()
            off = bi * 128
            t = pl.load(x, [off, 0], [128, 128])
            out = pl.store(t, [off, 0], out)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def consumer(
            self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            t = pl.load(x, [0, 0], [128, 128])
            out = pl.store(t, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            with pl.manual_scope():
                scratch = pl.create_tensor([512, 128], dtype=pl.FP32)
                scratch, tid = pl.spmd_submit(self.producer, x, scratch, core_num=4, sync_start=True)
                out, _ = pl.spmd_submit(self.consumer, scratch, out, core_num=2, deps=[tid])
            return out

    return Prog


class TestSpmdSubmitParsing:
    def test_builds_submit_with_launch_spec(self):
        Prog = _two_kernel_program()
        submits = _main_submits(Prog)
        assert len(submits) == 2
        first = submits[0]
        # core_num is the first-class launch-spec field; sync_start mirrors it.
        assert first.core_num is not None
        assert isinstance(first.core_num, ir.ConstInt) and first.core_num.value == 4
        assert first.sync_start is True
        # The return type is the flat Tuple{<kernel result>, TASK_ID}.
        assert isinstance(first.type, ir.TupleType)
        assert len(first.type.types) == 2
        assert isinstance(first.type.types[1], ir.ScalarType)
        assert first.type.types[1].dtype == pl.TASK_ID

    def test_sync_start_defaults_false(self):
        Prog = _two_kernel_program()
        second = _main_submits(Prog)[1]
        assert isinstance(second.core_num, ir.ConstInt) and second.core_num.value == 2
        assert second.sync_start is False

    def test_deps_wire_producer_task_id(self):
        Prog = _two_kernel_program()
        producer_submit, consumer_submit = _main_submits(Prog)
        # Producer has no dependency edges of its own.
        assert list(producer_submit.deps) == []
        # Consumer depends on the producer's TaskId scalar.
        deps = list(consumer_submit.deps)
        assert len(deps) == 1
        assert isinstance(deps[0].type, ir.ScalarType)
        assert deps[0].type.dtype == pl.TASK_ID

    def test_single_lhs_form(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k(
                self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
            ) -> pl.Tensor[[128], pl.FP32]:
                t = pl.load(x, [0], [128])
                out = pl.store(t, [0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
            ) -> pl.Tensor[[128], pl.FP32]:
                with pl.manual_scope():
                    res = pl.spmd_submit(self.k, x, out, core_num=8)
                    out = res[0]
                return out

        submits = _main_submits(Prog)
        assert len(submits) == 1
        assert isinstance(submits[0].core_num, ir.ConstInt) and submits[0].core_num.value == 8

    def test_core_num_accepts_closure_constant(self):
        n_blocks = 16

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k(
                self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
            ) -> pl.Tensor[[128], pl.FP32]:
                t = pl.load(x, [0], [128])
                out = pl.store(t, [0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
            ) -> pl.Tensor[[128], pl.FP32]:
                with pl.manual_scope():
                    out, _ = pl.spmd_submit(self.k, x, out, core_num=n_blocks)
                return out

        submit = _main_submits(Prog)[0]
        assert isinstance(submit.core_num, ir.ConstInt) and submit.core_num.value == 16

    def test_works_in_auto_scope(self):
        # spmd_submit is not restricted to manual_scope — like pl.submit, it
        # works in plain auto-tracked orchestration and preserves the launch spec.
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k(
                self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
            ) -> pl.Tensor[[128], pl.FP32]:
                t = pl.load(x, [0], [128])
                out = pl.store(t, [0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
            ) -> pl.Tensor[[128], pl.FP32]:
                out, _tid = pl.spmd_submit(self.k, x, out, core_num=4, sync_start=True)
                return out

        submits = _main_submits(Prog)
        assert len(submits) == 1
        assert isinstance(submits[0].core_num, ir.ConstInt) and submits[0].core_num.value == 4
        assert submits[0].sync_start is True

    def test_round_trips_through_printer(self):
        Prog = _two_kernel_program()
        text = Prog.as_python()
        assert "pl.spmd_submit(self.producer" in text
        assert "core_num=4" in text
        assert "sync_start=True" in text
        assert "pl.spmd_submit(self.consumer" in text
        assert "core_num=2" in text
        assert "deps=[tid]" in text
        # sync_start=True appears once (producer); the consumer's default
        # sync_start=False is OMITTED entirely (printer only emits the True case).
        assert text.count("sync_start=True") == 1
        assert "sync_start=False" not in text
        # Reparse and confirm full structural equality (core_num/sync_start
        # survive print -> parse).
        reparsed = pl.parse_program(text)
        ir.assert_structural_equal(reparsed, Prog)


class TestSpmdSubmitStructuralEquality:
    """Structural-equality must reflect over the new launch-spec fields.

    Both submits in a comparison share the same callee / arg / return-type
    objects so the *only* structural difference is core_num / sync_start.
    """

    def _pair(self, spec_a, spec_b):
        span = ir.Span.unknown()
        gv = ir.GlobalVar("k")
        arg = ir.Var("a", ir.ScalarType(DataType.INDEX), span)
        ret = ir.TupleType([ir.ScalarType(DataType.INDEX), ir.ScalarType(DataType.TASK_ID)])

        def build(core_num_val, sync_start):
            core_num = None if core_num_val is None else ir.ConstInt(core_num_val, DataType.INDEX, span)
            return ir.Submit(gv, [arg], [], {}, None, ret, span, core_num=core_num, sync_start=sync_start)

        return build(*spec_a), build(*spec_b)

    def test_plain_submit_not_equal_to_spmd_submit(self):
        plain, spmd = self._pair((None, False), (4, False))
        assert not ir.structural_equal(plain, spmd)

    def test_different_core_num_not_equal(self):
        a, b = self._pair((4, False), (8, False))
        assert not ir.structural_equal(a, b)

    def test_different_sync_start_not_equal(self):
        a, b = self._pair((4, False), (4, True))
        assert not ir.structural_equal(a, b)

    def test_same_spmd_submit_equal(self):
        a, b = self._pair((4, True), (4, True))
        assert ir.structural_equal(a, b)
        assert ir.structural_hash(a) == ir.structural_hash(b)

    def test_two_plain_submits_equal_with_null_core_num(self):
        # Regression: a null (nullopt) core_num field must compare cleanly,
        # not crash the reflection-driven structural-equal visitor.
        a, b = self._pair((None, False), (None, False))
        assert ir.structural_equal(a, b)


class TestSpmdSubmitConstructorValidation:
    def test_sync_start_without_core_num_rejected(self):
        span = ir.Span.unknown()
        gv = ir.GlobalVar("k")
        ret = ir.ScalarType(DataType.TASK_ID)
        with pytest.raises(ValueError, match="sync_start"):
            ir.Submit(gv, [], [], {}, None, ret, span, core_num=None, sync_start=True)

    def test_non_integer_core_num_rejected(self):
        # The IR constructor guards the launch-spec invariant at the public
        # boundary: core_num must be an integer/index expression, not float/bool.
        span = ir.Span.unknown()
        gv = ir.GlobalVar("k")
        ret = ir.TupleType([ir.ScalarType(DataType.INDEX), ir.ScalarType(DataType.TASK_ID)])
        bad_core_num = ir.ConstFloat(4.0, DataType.FP32, span)
        with pytest.raises(TypeError, match="core_num must be an integer"):
            ir.Submit(gv, [], [], {}, None, ret, span, core_num=bad_core_num, sync_start=False)


class TestSpmdSubmitErrors:
    def test_requires_core_num(self):
        with pytest.raises(ParserSyntaxError, match="requires the core_num"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k(
                    self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
                ) -> pl.Tensor[[128], pl.FP32]:
                    t = pl.load(x, [0], [128])
                    out = pl.store(t, [0], out)
                    return out

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
                ) -> pl.Tensor[[128], pl.FP32]:
                    with pl.manual_scope():
                        out, _ = pl.spmd_submit(self.k, x, out)
                    return out

    def test_core_num_must_be_positive(self):
        with pytest.raises(ParserSyntaxError, match="positive"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k(
                    self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
                ) -> pl.Tensor[[128], pl.FP32]:
                    t = pl.load(x, [0], [128])
                    out = pl.store(t, [0], out)
                    return out

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
                ) -> pl.Tensor[[128], pl.FP32]:
                    with pl.manual_scope():
                        out, _ = pl.spmd_submit(self.k, x, out, core_num=0)
                    return out

    def test_core_num_must_be_integer(self):
        with pytest.raises(ParserSyntaxError, match="integer"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k(
                    self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
                ) -> pl.Tensor[[128], pl.FP32]:
                    t = pl.load(x, [0], [128])
                    out = pl.store(t, [0], out)
                    return out

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
                ) -> pl.Tensor[[128], pl.FP32]:
                    with pl.manual_scope():
                        out, _ = pl.spmd_submit(self.k, x, out, core_num=2.5)
                    return out

    def test_sync_start_must_be_bool_literal(self):
        with pytest.raises(ParserSyntaxError, match="boolean literal"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k(
                    self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
                ) -> pl.Tensor[[128], pl.FP32]:
                    t = pl.load(x, [0], [128])
                    out = pl.store(t, [0], out)
                    return out

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
                ) -> pl.Tensor[[128], pl.FP32]:
                    with pl.manual_scope():
                        out, _ = pl.spmd_submit(self.k, x, out, core_num=4, sync_start=1)
                    return out

    def test_plain_submit_rejects_core_num(self):
        with pytest.raises(ParserTypeError, match="core_num"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k(
                    self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
                ) -> pl.Tensor[[128], pl.FP32]:
                    t = pl.load(x, [0], [128])
                    out = pl.store(t, [0], out)
                    return out

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
                ) -> pl.Tensor[[128], pl.FP32]:
                    with pl.manual_scope():
                        out, _ = pl.submit(self.k, x, out, core_num=4)
                    return out

    def test_bare_spmd_submit_must_be_unpacked(self):
        with pytest.raises(ParserSyntaxError, match="must be unpacked"):

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k(
                    self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
                ) -> pl.Tensor[[128], pl.FP32]:
                    t = pl.load(x, [0], [128])
                    out = pl.store(t, [0], out)
                    return out

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
                ) -> pl.Tensor[[128], pl.FP32]:
                    with pl.manual_scope():
                        pl.spmd_submit(self.k, x, out, core_num=4)
                    return out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

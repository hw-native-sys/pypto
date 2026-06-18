# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser / IR / printer tests for the ``allow_early_resolve=True`` submit hint.

``allow_early_resolve`` opts a task in as a speculative early-dispatch producer
(simpler#1065). It is recorded on the first-class ``Submit.allow_early_resolve``
field and is independent of the SPMD launch spec — valid on both ``pl.submit``
and ``pl.spmd_submit``. The flag lowers to ``Arg::set_allow_early_resolve(true)``
in orchestration codegen.
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


def _main_submits(prog) -> list:
    """Collect the Submit nodes bound as AssignStmt RHS in ``prog``'s main body."""
    fn = prog.get_function("main")
    assert fn is not None
    return [
        s.value for s in _flatten(fn.body) if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Submit)
    ]


def _plain_flag_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
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
                out, _ = pl.submit(self.producer, x, out, allow_early_resolve=True)
            return out

    return Prog


def _spmd_flag_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
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
                out, _ = pl.spmd_submit(self.producer, x, out, core_num=4, allow_early_resolve=True)
            return out

    return Prog


def _no_flag_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
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
                out, _ = pl.submit(self.producer, x, out)
            return out

    return Prog


def _explicit_false_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
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
                out, _ = pl.submit(self.producer, x, out, allow_early_resolve=False)
            return out

    return Prog


class TestAllowEarlyResolveParsing:
    def test_plain_submit_records_flag(self):
        (submit,) = _main_submits(_plain_flag_program())
        assert submit.allow_early_resolve is True
        # Independent of the launch spec — a plain submit has no core_num.
        assert submit.core_num is None

    def test_spmd_submit_records_flag(self):
        (submit,) = _main_submits(_spmd_flag_program())
        assert submit.allow_early_resolve is True
        # Coexists with the SPMD launch spec without interference.
        assert isinstance(submit.core_num, ir.ConstInt) and submit.core_num.value == 4

    def test_defaults_false_when_omitted(self):
        (submit,) = _main_submits(_no_flag_program())
        assert submit.allow_early_resolve is False

    def test_explicit_false_records_false(self):
        (submit,) = _main_submits(_explicit_false_program())
        assert submit.allow_early_resolve is False

    def test_non_bool_literal_rejected(self):
        with pytest.raises(ParserSyntaxError, match="allow_early_resolve must be a boolean literal"):

            @pl.program
            class Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def producer(
                    self,
                    x: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    t = pl.load(x, [0, 0], [128, 128])
                    out = pl.store(t, [0, 0], out)
                    return out

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    x: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    with pl.manual_scope():
                        out, _ = pl.submit(self.producer, x, out, allow_early_resolve=1)
                    return out

    def test_rejected_on_plain_kernel_call(self):
        # A fire-and-forget self.kernel(...) call is not a submit, so the hint
        # is not an accepted kwarg there.
        with pytest.raises(ParserTypeError, match="allow_early_resolve"):

            @pl.program
            class Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def producer(
                    self,
                    x: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    t = pl.load(x, [0, 0], [128, 128])
                    out = pl.store(t, [0, 0], out)
                    return out

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    x: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    self.producer(x, out, allow_early_resolve=True)
                    return out


class TestAllowEarlyResolveRoundTrip:
    def test_round_trips_through_printer(self):
        Prog = _plain_flag_program()
        text = Prog.as_python()
        assert "allow_early_resolve=True" in text
        # Reparse and confirm structural equality preserves the flag.
        reparsed = pl.parse_program(text)
        ir.assert_structural_equal(reparsed, Prog)
        assert _main_submits(reparsed)[0].allow_early_resolve is True

    def test_default_false_omitted_from_print(self):
        text = _no_flag_program().as_python()
        assert "allow_early_resolve" not in text


class TestAllowEarlyResolveStructuralEqual:
    """Two Submits identical except for ``allow_early_resolve`` must differ."""

    @staticmethod
    def _pair(flag_a: bool, flag_b: bool):
        # Share the callee / arg / return-type objects so the only structural
        # difference between the two Submits is allow_early_resolve.
        span = ir.Span.unknown()
        gv = ir.GlobalVar("k")
        arg = ir.Var("a", ir.ScalarType(DataType.INDEX), span)
        ret = ir.TupleType([ir.ScalarType(DataType.INDEX), ir.ScalarType(DataType.TASK_ID)])

        def build(flag):
            return ir.Submit(gv, [arg], [], {}, None, ret, span, allow_early_resolve=flag)

        return build(flag_a), build(flag_b)

    def test_differs_by_flag(self):
        a, b = self._pair(True, False)
        assert not ir.structural_equal(a, b)

    def test_same_flag_equal(self):
        a, b = self._pair(True, True)
        assert ir.structural_equal(a, b)
        assert ir.structural_hash(a) == ir.structural_hash(b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

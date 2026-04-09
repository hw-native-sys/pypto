# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for SplitChunkedLoops pass."""

import re
from typing import cast

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.printer import python_print


def _prepare_for_split(program):
    """Run prerequisite passes to produce SSA input for SplitChunkedLoops."""
    program = passes.unroll_loops()(program)
    program = passes.convert_to_ssa()(program)
    program = passes.flatten_call_expr()(program)
    return program


def _top_level_stmts(program: ir.Program) -> list[ir.Stmt]:
    """Return the first function's top-level statements."""
    func = list(program.functions.values())[0]
    return list(cast(ir.SeqStmts, func.body).stmts)


def _body_stmts(stmt: ir.Stmt) -> list[ir.Stmt]:
    """Return child statements from a SeqStmts body."""
    return list(cast(ir.SeqStmts, stmt).stmts)


def _normalize_expected(program):
    """Normalize Expected IR structure to match pass pipeline output.

    The DSL-constructed Expected programs have a different statement nesting
    than the pass pipeline output. This applies the same structural
    normalization so assert_structural_equal can compare them.
    """
    return passes.normalize_stmt_structure()(program)


class TestBasicChunking:
    """Tests for basic loop chunking with SSA iter_args propagation."""

    def test_divisible_chunk(self):
        """Chunk a loop where trip_count is divisible by chunk_size."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.range(0, 10, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i_0_out, (x_iter_1_outer,) in pl.range(
                        0, 2, 1, init_values=(x_0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        for i_0_in, (x_iter_1_inner,) in pl.range(
                            0,
                            5,
                            1,
                            init_values=(x_iter_1_outer,),
                            attrs={"loop_origin": pl.LoopOrigin.ChunkInner},
                        ):
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_inner, 1.0)
                            x_iter_1_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_iter_1_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_1_inner_rv)
                return x_iter_1_outer_rv

        ir.assert_structural_equal(After, _normalize_expected(Expected), enable_auto_mapping=True)

    def test_non_divisible_chunk(self):
        """Chunk a loop where trip_count is NOT divisible by chunk_size."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.range(0, 7, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i_0_out, (x_iter_1_outer,) in pl.range(
                        0, 1, 1, init_values=(x_0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):  # noqa: E501
                        for i_0_in, (x_iter_1_inner,) in pl.range(
                            0,
                            5,
                            1,
                            init_values=(x_iter_1_outer,),
                            attrs={"loop_origin": pl.LoopOrigin.ChunkInner},
                        ):  # noqa: E501
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_inner, 1.0)
                            x_iter_1_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_iter_1_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_1_inner_rv)
                    for i_0_rem, (x_iter_1_rem,) in pl.range(
                        0,
                        2,
                        1,
                        init_values=(x_iter_1_outer_rv,),
                        attrs={"loop_origin": pl.LoopOrigin.ChunkRemainder},
                    ):  # noqa: E501
                        x_3_f: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_rem, 1.0)
                        x_iter_1_rem_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3_f)
                return x_iter_1_rem_rv

        ir.assert_structural_equal(After, _normalize_expected(Expected), enable_auto_mapping=True)

    def test_single_chunk(self):
        """Chunk a loop where trip_count equals chunk_size."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.range(0, 5, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i_0_out, (x_iter_1_outer,) in pl.range(
                        0, 1, 1, init_values=(x_0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        for i_0_in, (x_iter_1_inner,) in pl.range(
                            0,
                            5,
                            1,
                            init_values=(x_iter_1_outer,),
                            attrs={"loop_origin": pl.LoopOrigin.ChunkInner},
                        ):
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_inner, 1.0)
                            x_iter_1_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_iter_1_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_1_inner_rv)
                return x_iter_1_outer_rv

        ir.assert_structural_equal(After, _normalize_expected(Expected), enable_auto_mapping=True)


class TestChunkingWithStep:
    """Tests for chunking with non-unit step."""

    def test_step_2(self):
        """Chunk with step=2: range(0, 20, 2, chunk=5) -> trip_count=10."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.range(0, 20, 2, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i_0_out, (x_iter_1_outer,) in pl.range(
                        0, 2, 1, init_values=(x_0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        for i_0_in, (x_iter_1_inner,) in pl.range(
                            0,
                            5,
                            1,
                            init_values=(x_iter_1_outer,),
                            attrs={"loop_origin": pl.LoopOrigin.ChunkInner},
                        ):
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_inner, 1.0)
                            x_iter_1_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_iter_1_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_1_inner_rv)
                return x_iter_1_outer_rv

        ir.assert_structural_equal(After, _normalize_expected(Expected), enable_auto_mapping=True)

    def test_chunk_all_remainder(self):
        """Chunk where trip_count < chunk_size -> only remainder loop."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.range(0, 3, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i_0_rem, (x_iter_1_rem,) in pl.range(
                        0, 3, 1, init_values=(x_0,), attrs={"loop_origin": pl.LoopOrigin.ChunkRemainder}
                    ):
                        x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_rem, 1.0)
                        x_iter_1_rem_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                return x_iter_1_rem_rv

        ir.assert_structural_equal(After, _normalize_expected(Expected), enable_auto_mapping=True)


class TestChunkingWithKind:
    """Tests for chunking with different loop kinds."""

    def test_parallel_chunk(self):
        """Chunk a parallel loop: both inner and outer loops should be Parallel."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i_0_out, (x_iter_1_outer,) in pl.parallel(
                        0, 2, 1, init_values=(x_0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        for i_0_in, (x_iter_1_inner,) in pl.parallel(
                            0,
                            4,
                            1,
                            init_values=(x_iter_1_outer,),
                            attrs={"loop_origin": pl.LoopOrigin.ChunkInner},
                        ):
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_inner, 1.0)
                            x_iter_1_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_iter_1_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_1_inner_rv)
                return x_iter_1_outer_rv

        ir.assert_structural_equal(After, _normalize_expected(Expected), enable_auto_mapping=True)

    @pytest.mark.filterwarnings("ignore:.*RoundtripInstrument.*IR not printable:UserWarning")
    def test_unroll_chunk(self):
        """Chunk an unroll loop: both inner and outer loops are Unroll.

        Since the DSL does not support pl.unroll() with init_values,
        we verify the IR structure properties directly instead of
        using structural equality.
        """

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.unroll(0, 12, 1, chunk=4):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        # Extract the function body
        stmts = _top_level_stmts(After)

        # Body should be SeqStmts: [auto_incore_scope, return]
        assert len(stmts) == 2  # auto_incore scope + return

        # The first stmt is the AutoInCore scope
        scope = cast(ir.ScopeStmt, stmts[0])
        assert scope.scope_kind == ir.ScopeKind.AutoInCore

        # Inside the scope is the outer for loop
        outer_for = cast(ir.ForStmt, scope.body)
        assert outer_for.kind == ir.ForKind.Unroll
        assert len(outer_for.iter_args) == 1
        assert len(outer_for.return_vars) == 1

        # Outer loop bounds: range(0, 3, 1) — 12/4 = 3 full chunks
        assert cast(ir.ConstInt, outer_for.start).value == 0
        assert cast(ir.ConstInt, outer_for.stop).value == 3

        # Inner loop is inside outer body (SeqStmts: [inner_for, yield])
        outer_body_stmts = _body_stmts(outer_for.body)
        inner_for = cast(ir.ForStmt, outer_body_stmts[0])
        assert inner_for.kind == ir.ForKind.Unroll
        assert len(inner_for.iter_args) == 1
        assert len(inner_for.return_vars) == 1

        # Inner loop bounds: range(0, 4, 1)
        assert cast(ir.ConstInt, inner_for.start).value == 0
        assert cast(ir.ConstInt, inner_for.stop).value == 4


class TestPrinterRoundTrip:
    """Tests for printer output with chunk kwargs."""

    def test_chunk_in_printer(self):
        """Verify that chunk kwarg is printed correctly."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.range(0, 10, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        printed = python_print(Before)
        assert "chunk=5" in printed

    def test_parallel_chunk_in_printer(self):
        """Verify parallel chunk kwarg is printed."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        x = pl.add(x, 1.0)
                return x

        printed = python_print(Before)
        assert "chunk=4" in printed
        assert "pl.parallel" in printed


class TestParserErrors:
    """Tests for parser validation of chunk arguments."""

    def test_chunk_with_init_values_allowed(self):
        """chunk + init_values should be allowed (not raise parser error)."""

        @pl.program
        class Good:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i, (s,) in pl.range(10, init_values=(x,), chunk=5):
                        s = pl.add(s, 1.0)
                        s = pl.yield_(s)
                return x

    def test_chunk_zero_error(self):
        """chunk=0 should raise parser error."""
        with pytest.raises(Exception, match="chunk must be a positive integer"):

            @pl.program
            class Bad:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for i in pl.range(0, 10, 1, chunk=0):
                            x = pl.add(x, 1.0)
                    return x

    def test_chunk_negative_error(self):
        """chunk=-1 should raise parser error."""
        with pytest.raises(Exception, match="chunk must be a positive integer"):

            @pl.program
            class Bad:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for i in pl.range(0, 10, 1, chunk=-1):
                            x = pl.add(x, 1.0)
                    return x


class TestLoopOrigin:
    """Tests for LoopOrigin annotation set by SplitChunkedLoops."""

    def _get_func_body_stmts(self, program):
        """Get the top-level statements from the first function's body."""
        return _top_level_stmts(program)

    def _get_auto_incore_body_stmts(self, program):
        """Get statements inside the AutoInCore scope."""
        stmts = self._get_func_body_stmts(program)
        # First stmt should be AutoInCore scope
        scope = cast(ir.ScopeStmt, stmts[0])
        assert scope.scope_kind == ir.ScopeKind.AutoInCore
        body = scope.body
        # Body may be a single stmt or SeqStmts
        if hasattr(body, "stmts"):
            return _body_stmts(body)
        return [body]

    def test_divisible_chunk_origin(self):
        """Verify outer=ChunkOuter, inner=ChunkInner for divisible chunks."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.range(0, 10, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        inner_stmts = self._get_auto_incore_body_stmts(After)
        outer_for = cast(ir.ForStmt, inner_stmts[0])

        assert outer_for.attrs.get("loop_origin") == ir.LoopOrigin.ChunkOuter

        # Inner loop is inside outer body
        outer_body_stmts = _body_stmts(outer_for.body)
        inner_for = cast(ir.ForStmt, outer_body_stmts[0])
        assert inner_for.attrs.get("loop_origin") == ir.LoopOrigin.ChunkInner

    def test_non_divisible_chunk_origin(self):
        """Verify outer=ChunkOuter, inner=ChunkInner, remainder=ChunkRemainder."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.range(0, 7, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        inner_stmts = self._get_auto_incore_body_stmts(After)
        # stmts: [outer_for, remainder_for]
        outer_for = cast(ir.ForStmt, inner_stmts[0])
        remainder_for = cast(ir.ForStmt, inner_stmts[1])

        assert outer_for.attrs.get("loop_origin") == ir.LoopOrigin.ChunkOuter

        outer_body_stmts = _body_stmts(outer_for.body)
        inner_for = cast(ir.ForStmt, outer_body_stmts[0])
        assert inner_for.attrs.get("loop_origin") == ir.LoopOrigin.ChunkInner

        assert remainder_for.attrs.get("loop_origin") == ir.LoopOrigin.ChunkRemainder

    def test_all_remainder_origin(self):
        """Verify remainder=ChunkRemainder when trip_count < chunk_size."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.range(0, 3, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        inner_stmts = self._get_auto_incore_body_stmts(After)
        remainder_for = cast(ir.ForStmt, inner_stmts[0])
        assert remainder_for.attrs.get("loop_origin") == ir.LoopOrigin.ChunkRemainder

    def test_non_chunked_loop_origin(self):
        """Verify regular (non-chunked) loops have Original origin."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 10, 1):
                    x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        stmts = self._get_func_body_stmts(After)
        for_stmt = cast(ir.ForStmt, stmts[0])
        assert "loop_origin" not in for_stmt.attrs


class TestNestedChunking:
    """Tests for nested chunked loops with iter_args propagation."""

    def test_nested_outer_divisible_inner_remainder(self):
        """Nested chunks: outer divisible, inner only remainder.

        Reproduces the bug where inner remainder loop's init_values
        referenced the original (unsplit) iter_arg instead of the
        inner iter_arg from the outer loop's split.
        """

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.parallel(8, chunk=4):
                        for j in pl.parallel(1, chunk=2):
                            x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i_0_out, (x_iter_1_outer,) in pl.parallel(
                        0, 2, 1, init_values=(x_0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        for i_0_in, (x_iter_1_inner,) in pl.parallel(
                            0,
                            4,
                            1,
                            init_values=(x_iter_1_outer,),
                            attrs={"loop_origin": pl.LoopOrigin.ChunkInner},
                        ):
                            for j_0_rem, (x_iter_3_rem,) in pl.parallel(
                                0,
                                1,
                                1,
                                init_values=(x_iter_1_inner,),
                                attrs={"loop_origin": pl.LoopOrigin.ChunkRemainder},
                            ):
                                x_5: pl.Tensor[[64], pl.FP32] = pl.tensor.add(x_iter_3_rem, 1.0)
                                x_iter_3_rem_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_5)
                            x_iter_1_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_3_rem_rv)
                        x_iter_1_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_1_inner_rv)
                return x_iter_1_outer_rv

        ir.assert_structural_equal(After, _normalize_expected(Expected), enable_auto_mapping=True)

    def test_nested_both_divisible(self):
        """Nested chunks: both outer and inner divisible."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.parallel(8, chunk=4):
                        for j in pl.parallel(12, chunk=4):
                            x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i_0_out, (x_iter_1_outer,) in pl.parallel(
                        0, 2, 1, init_values=(x_0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        for i_0_in, (x_iter_1_inner,) in pl.parallel(
                            0,
                            4,
                            1,
                            init_values=(x_iter_1_outer,),
                            attrs={"loop_origin": pl.LoopOrigin.ChunkInner},
                        ):
                            for j_0_out, (x_iter_3_outer,) in pl.parallel(
                                0,
                                3,
                                1,
                                init_values=(x_iter_1_inner,),
                                attrs={"loop_origin": pl.LoopOrigin.ChunkOuter},
                            ):
                                for j_0_in, (x_iter_3_inner,) in pl.parallel(
                                    0,
                                    4,
                                    1,
                                    init_values=(x_iter_3_outer,),
                                    attrs={"loop_origin": pl.LoopOrigin.ChunkInner},
                                ):
                                    x_5: pl.Tensor[[64], pl.FP32] = pl.tensor.add(x_iter_3_inner, 1.0)
                                    x_iter_3_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_5)
                                x_iter_3_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_3_inner_rv)
                            x_iter_1_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_3_outer_rv)
                        x_iter_1_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_1_inner_rv)
                return x_iter_1_outer_rv

        ir.assert_structural_equal(After, _normalize_expected(Expected), enable_auto_mapping=True)

    def test_nested_both_remainder(self):
        """Nested chunks: both outer and inner have remainders.

        Verifies init_values are correctly substituted in all paths:
        outer-inner, outer-remainder, remainder-inner, remainder-remainder.
        """

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.parallel(6, chunk=4):
                        for j in pl.parallel(3, chunk=2):
                            x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        printed = python_print(After)
        init_refs = re.findall(r"init_values=\((\w+),\)", printed)
        for ref in init_refs:
            assert ref != "x__iter_v1", (
                "Found bare 'x__iter_v1' in init_values; should be a chunk-qualified iter name."
            )
            assert ref != "x__iter_v3", (
                "Found bare 'x__iter_v3' in init_values; should be a chunk-qualified iter name."
            )


class TestDynamicChunking:
    """Tests for chunked loops where start/stop are dynamic (runtime) scalars."""

    @staticmethod
    def _split_and_simplify(program):
        """Run prerequisite passes, split chunked loops, and simplify expressions."""
        prepared = _prepare_for_split(program)
        split = passes.split_chunked_loops()(prepared)
        return passes.simplify()(split)

    def test_dynamic_stop(self):
        """Dynamic stop: outer+inner+remainder with FloorDiv/FloorMod bounds."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.range(0, n, 1, chunk=4):
                        x = pl.add(x, 1.0)
                return x

        After = self._split_and_simplify(Input)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(
                self, x_0: pl.Tensor[[64], pl.FP32], n_0: pl.Scalar[pl.INDEX]
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i_out, (x_outer,) in pl.range(
                        0, n_0 // 4, 1, init_values=(x_0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        for i_in, (x_inner,) in pl.range(
                            0, 4, 1, init_values=(x_outer,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, 1.0)
                            x_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_inner_rv)
                    for i_rem, (x_rem,) in pl.range(
                        0,
                        n_0 % 4,
                        1,
                        init_values=(x_outer_rv,),
                        attrs={"loop_origin": pl.LoopOrigin.ChunkRemainder},
                    ):
                        x_4: pl.Tensor[[64], pl.FP32] = pl.add(x_rem, 1.0)
                        x_rem_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_4)
                return x_rem_rv

        ir.assert_structural_equal(After, _normalize_expected(Expected), enable_auto_mapping=True)

    def test_dynamic_start_and_stop(self):
        """Both start and stop are dynamic."""

        @pl.program
        class Input:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                lo: pl.Scalar[pl.INDEX],
                hi: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.range(lo, hi, 1, chunk=4):
                        x = pl.add(x, 1.0)
                return x

        After = self._split_and_simplify(Input)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(
                self,
                x_0: pl.Tensor[[64], pl.FP32],
                lo_0: pl.Scalar[pl.INDEX],
                hi_0: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i_out, (x_outer,) in pl.range(
                        0,
                        pl.max(hi_0 - lo_0, 0) // 4,
                        1,
                        init_values=(x_0,),
                        attrs={"loop_origin": pl.LoopOrigin.ChunkOuter},
                    ):
                        for i_in, (x_inner,) in pl.range(
                            0, 4, 1, init_values=(x_outer,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, 1.0)
                            x_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_inner_rv)
                    for i_rem, (x_rem,) in pl.range(
                        0,
                        pl.max(hi_0 - lo_0, 0) % 4,
                        1,
                        init_values=(x_outer_rv,),
                        attrs={"loop_origin": pl.LoopOrigin.ChunkRemainder},
                    ):
                        x_4: pl.Tensor[[64], pl.FP32] = pl.add(x_rem, 1.0)
                        x_rem_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_4)
                return x_rem_rv

        ir.assert_structural_equal(After, _normalize_expected(Expected), enable_auto_mapping=True)

    def test_dynamic_stop_parallel(self):
        """Dynamic stop with pl.parallel should also work."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.parallel(0, n, 1, chunk=4):
                        x = pl.add(x, 1.0)
                return x

        After = self._split_and_simplify(Input)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(
                self, x_0: pl.Tensor[[64], pl.FP32], n_0: pl.Scalar[pl.INDEX]
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i_out, (x_outer,) in pl.parallel(
                        0, n_0 // 4, 1, init_values=(x_0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        for i_in, (x_inner,) in pl.parallel(
                            0, 4, 1, init_values=(x_outer,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, 1.0)
                            x_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_inner_rv)
                    for i_rem, (x_rem,) in pl.parallel(
                        0,
                        n_0 % 4,
                        1,
                        init_values=(x_outer_rv,),
                        attrs={"loop_origin": pl.LoopOrigin.ChunkRemainder},
                    ):
                        x_4: pl.Tensor[[64], pl.FP32] = pl.add(x_rem, 1.0)
                        x_rem_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_4)
                return x_rem_rv

        ir.assert_structural_equal(After, _normalize_expected(Expected), enable_auto_mapping=True)

    def test_static_still_works(self):
        """Regression: static bounds should continue to produce same IR as before."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.range(0, 10, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i_0_out, (x_iter_1_outer,) in pl.range(
                        0, 2, 1, init_values=(x_0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        for i_0_in, (x_iter_1_inner,) in pl.range(
                            0,
                            5,
                            1,
                            init_values=(x_iter_1_outer,),
                            attrs={"loop_origin": pl.LoopOrigin.ChunkInner},
                        ):
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_inner, 1.0)
                            x_iter_1_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_iter_1_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_1_inner_rv)
                return x_iter_1_outer_rv

        ir.assert_structural_equal(After, _normalize_expected(Expected), enable_auto_mapping=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

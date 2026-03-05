# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for SplitChunkedLoops pass."""

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


class TestBasicChunking:
    """Tests for basic loop chunking with SSA iter_args propagation."""

    def test_divisible_chunk(self):
        """Chunk a loop where trip_count is divisible by chunk_size."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.range(0, 10, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i_0_out, (x_iter_1_outer,) in pl.range(0, 2, 1, init_values=(x_0,)):
                        for i_0_in, (x_iter_1_inner,) in pl.range(0, 5, 1, init_values=(x_iter_1_outer,)):
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_inner, 1.0)
                            x_iter_1_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_iter_1_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_1_inner_rv)
                return x_iter_1_outer_rv

        ir.assert_structural_equal(After, Expected)

    def test_non_divisible_chunk(self):
        """Chunk a loop where trip_count is NOT divisible by chunk_size."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.range(0, 7, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i_0_out, (x_iter_1_outer,) in pl.range(0, 1, 1, init_values=(x_0,)):
                        for i_0_in, (x_iter_1_inner,) in pl.range(0, 5, 1, init_values=(x_iter_1_outer,)):
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_inner, 1.0)
                            x_iter_1_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_iter_1_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_1_inner_rv)
                    for i_0_rem, (x_iter_1_rem,) in pl.range(0, 2, 1, init_values=(x_iter_1_outer_rv,)):
                        x_3_f: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_rem, 1.0)
                        x_iter_1_rem_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3_f)
                return x_iter_1_rem_rv

        ir.assert_structural_equal(After, Expected)

    def test_single_chunk(self):
        """Chunk a loop where trip_count equals chunk_size."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.range(0, 5, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i_0_out, (x_iter_1_outer,) in pl.range(0, 1, 1, init_values=(x_0,)):
                        for i_0_in, (x_iter_1_inner,) in pl.range(0, 5, 1, init_values=(x_iter_1_outer,)):
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_inner, 1.0)
                            x_iter_1_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_iter_1_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_1_inner_rv)
                return x_iter_1_outer_rv

        ir.assert_structural_equal(After, Expected)


class TestChunkingWithStep:
    """Tests for chunking with non-unit step."""

    def test_step_2(self):
        """Chunk with step=2: range(0, 20, 2, chunk=5) -> trip_count=10."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.range(0, 20, 2, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i_0_out, (x_iter_1_outer,) in pl.range(0, 2, 1, init_values=(x_0,)):
                        for i_0_in, (x_iter_1_inner,) in pl.range(0, 5, 1, init_values=(x_iter_1_outer,)):
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_inner, 1.0)
                            x_iter_1_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_iter_1_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_1_inner_rv)
                return x_iter_1_outer_rv

        ir.assert_structural_equal(After, Expected)

    def test_chunk_all_remainder(self):
        """Chunk where trip_count < chunk_size -> only remainder loop."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.range(0, 3, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i_0_rem, (x_iter_1_rem,) in pl.range(0, 3, 1, init_values=(x_0,)):
                        x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_rem, 1.0)
                        x_iter_1_rem_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                return x_iter_1_rem_rv

        ir.assert_structural_equal(After, Expected)


class TestChunkingWithKind:
    """Tests for chunking with different loop kinds."""

    def test_parallel_chunk(self):
        """Chunk a parallel loop: both inner and outer loops should be Parallel."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i_0_out, (x_iter_1_outer,) in pl.parallel(0, 2, 1, init_values=(x_0,)):
                        for i_0_in, (x_iter_1_inner,) in pl.parallel(0, 4, 1, init_values=(x_iter_1_outer,)):
                            x_3: pl.Tensor[[64], pl.FP32] = pl.add(x_iter_1_inner, 1.0)
                            x_iter_1_inner_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_3)
                        x_iter_1_outer_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter_1_inner_rv)
                return x_iter_1_outer_rv

        ir.assert_structural_equal(After, Expected)

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
                with pl.auto_incore():
                    for i in pl.unroll(0, 12, 1, chunk=4):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        # Extract the function body
        func = list(After.functions.values())[0]
        body = func.body

        # Body should be SeqStmts: [auto_incore_scope, return]
        assert body.stmts is not None  # type: ignore[attr-defined]
        stmts = list(body.stmts)  # type: ignore[attr-defined]
        assert len(stmts) == 2  # auto_incore scope + return

        # The first stmt is the AutoInCore scope
        scope = stmts[0]
        assert scope.scope_kind == ir.ScopeKind.AutoInCore

        # Inside the scope is the outer for loop
        outer_for = scope.body
        assert outer_for.kind == ir.ForKind.Unroll
        assert len(outer_for.iter_args) == 1
        assert len(outer_for.return_vars) == 1

        # Outer loop bounds: range(0, 3, 1) — 12/4 = 3 full chunks
        assert outer_for.start.value == 0
        assert outer_for.stop.value == 3

        # Inner loop is inside outer body (SeqStmts: [inner_for, yield])
        outer_body_stmts = list(outer_for.body.stmts)
        inner_for = outer_body_stmts[0]
        assert inner_for.kind == ir.ForKind.Unroll
        assert len(inner_for.iter_args) == 1
        assert len(inner_for.return_vars) == 1

        # Inner loop bounds: range(0, 4, 1)
        assert inner_for.start.value == 0
        assert inner_for.stop.value == 4


class TestPrinterRoundTrip:
    """Tests for printer output with chunk kwargs."""

    def test_chunk_in_printer(self):
        """Verify that chunk kwarg is printed correctly."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
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
                with pl.auto_incore():
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        x = pl.add(x, 1.0)
                return x

        printed = python_print(Before)
        assert "chunk=4" in printed
        assert "pl.parallel" in printed


class TestParserErrors:
    """Tests for parser validation of chunk arguments."""

    def test_chunk_with_init_values_error(self):
        """chunk + init_values should raise parser error."""
        with pytest.raises(Exception, match="chunk cannot be combined with init_values"):

            @pl.program
            class Bad:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    for i, (s,) in pl.range(10, init_values=(x,), chunk=5):
                        s = pl.add(s, 1.0)  # noqa: PLW2901
                        s = pl.yield_(s)  # noqa: PLW2901
                    return x

    def test_chunk_zero_error(self):
        """chunk=0 should raise parser error."""
        with pytest.raises(Exception, match="chunk must be a positive integer"):

            @pl.program
            class Bad:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
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
                    for i in pl.range(0, 10, 1, chunk=-1):
                        x = pl.add(x, 1.0)
                    return x


class TestLoopOrigin:
    """Tests for LoopOrigin annotation set by SplitChunkedLoops."""

    def _get_func_body_stmts(self, program):
        """Get the top-level statements from the first function's body."""
        func = list(program.functions.values())[0]
        return list(func.body.stmts)  # type: ignore[attr-defined]

    def _get_auto_incore_body_stmts(self, program):
        """Get statements inside the AutoInCore scope."""
        stmts = self._get_func_body_stmts(program)
        # First stmt should be AutoInCore scope
        scope = stmts[0]
        assert scope.scope_kind == ir.ScopeKind.AutoInCore
        body = scope.body
        # Body may be a single stmt or SeqStmts
        if hasattr(body, "stmts"):
            return list(body.stmts)
        return [body]

    def test_divisible_chunk_origin(self):
        """Verify outer=ChunkOuter, inner=ChunkInner for divisible chunks."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.range(0, 10, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        inner_stmts = self._get_auto_incore_body_stmts(After)
        outer_for = inner_stmts[0]

        assert outer_for.loop_origin == ir.LoopOrigin.ChunkOuter

        # Inner loop is inside outer body
        outer_body_stmts = list(outer_for.body.stmts)
        inner_for = outer_body_stmts[0]
        assert inner_for.loop_origin == ir.LoopOrigin.ChunkInner

    def test_non_divisible_chunk_origin(self):
        """Verify outer=ChunkOuter, inner=ChunkInner, remainder=ChunkRemainder."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.range(0, 7, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        inner_stmts = self._get_auto_incore_body_stmts(After)
        # stmts: [outer_for, remainder_for]
        outer_for = inner_stmts[0]
        remainder_for = inner_stmts[1]

        assert outer_for.loop_origin == ir.LoopOrigin.ChunkOuter

        outer_body_stmts = list(outer_for.body.stmts)
        inner_for = outer_body_stmts[0]
        assert inner_for.loop_origin == ir.LoopOrigin.ChunkInner

        assert remainder_for.loop_origin == ir.LoopOrigin.ChunkRemainder

    def test_all_remainder_origin(self):
        """Verify remainder=ChunkRemainder when trip_count < chunk_size."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.range(0, 3, 1, chunk=5):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        After = passes.split_chunked_loops()(Before)

        inner_stmts = self._get_auto_incore_body_stmts(After)
        remainder_for = inner_stmts[0]
        assert remainder_for.loop_origin == ir.LoopOrigin.ChunkRemainder

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
        for_stmt = stmts[0]
        assert for_stmt.loop_origin == ir.LoopOrigin.Original


class TestAutoIncoreGating:
    """Tests for auto_incore scope gating behavior."""

    def test_chunked_loop_without_auto_incore_not_split(self):
        """Chunked loop outside auto_incore scope is NOT split."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 10, 1, chunk=5):
                    x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_split(Input)
        before_str = python_print(Before)
        After = passes.split_chunked_loops()(Before)
        after_str = python_print(After)

        # Should be unchanged — no splitting without auto_incore
        assert before_str == after_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

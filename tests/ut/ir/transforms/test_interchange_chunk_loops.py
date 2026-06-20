# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for InterchangeChunkLoops pass.

Test strategy:
  Build a ``Before`` program, run prerequisite passes + InterchangeChunkLoops,
  and compare to an explicitly-constructed ``Expected`` program using
  ``ir.assert_structural_equal(After, Expected)``.
"""

import re
from collections.abc import Callable

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.printer import python_print


def _prepare_for_interchange(program):
    """Run prerequisite passes to produce input for InterchangeChunkLoops."""
    program = passes.unroll_loops()(program)
    program = passes.convert_to_ssa()(program)
    program = passes.flatten_call_expr()(program)
    program = passes.split_chunked_loops()(program)
    return program


class TestSingleParallelChunk:
    """Tests for single parallel chunked loop (1 outer + 1 inner, InCore wrapping only)."""

    def test_single_parallel_chunk_gets_incore(self):
        """Single parallel chunked loop: outer wraps InCore around inner."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i0, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for i1, (x2,) in pl.parallel(
                            4, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x3: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x2, 1.0)
                            x4: pl.Tensor[[64], pl.FP32] = pl.yield_(x3)
                    x5: pl.Tensor[[64], pl.FP32] = pl.yield_(x4)
                return x5

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)


class TestNestedParallelChunks:
    """Tests for nested parallel chunked loops (full interchange + InCore)."""

    def test_two_nested_parallel_divisible(self):
        """Two nested parallel chunked loops, divisible: full interchange + InCore."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        for j in pl.parallel(0, 12, 1, chunk=4, chunk_policy="leading_full"):
                            x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i0, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    for j0, (x2,) in pl.parallel(
                        3, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        with pl.at(level=pl.Level.CORE_GROUP):
                            for i1, (x3,) in pl.parallel(
                                4, init_values=(x2,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                            ):
                                for j1, (x4,) in pl.parallel(
                                    4, init_values=(x3,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                                ):
                                    x5: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x4, 1.0)
                                    x6: pl.Tensor[[64], pl.FP32] = pl.yield_(x5)
                                x7: pl.Tensor[[64], pl.FP32] = pl.yield_(x6)
                        x8: pl.Tensor[[64], pl.FP32] = pl.yield_(x7)
                    x9: pl.Tensor[[64], pl.FP32] = pl.yield_(x8)
                return x9

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)

    def test_two_nested_parallel_with_iter_args(self):
        """Two nested parallel chunked loops with iter_args: verify SSA threading.

        Same Before as ``test_two_nested_parallel_divisible`` — this test also
        structurally confirms that iter_args thread correctly through every
        level of the interchanged nest.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        for j in pl.parallel(0, 12, 1, chunk=4, chunk_policy="leading_full"):
                            x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i0, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    for j0, (x2,) in pl.parallel(
                        3, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        with pl.at(level=pl.Level.CORE_GROUP):
                            for i1, (x3,) in pl.parallel(
                                4, init_values=(x2,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                            ):
                                for j1, (x4,) in pl.parallel(
                                    4, init_values=(x3,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                                ):
                                    x5: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x4, 1.0)
                                    x6: pl.Tensor[[64], pl.FP32] = pl.yield_(x5)
                                x7: pl.Tensor[[64], pl.FP32] = pl.yield_(x6)
                        x8: pl.Tensor[[64], pl.FP32] = pl.yield_(x7)
                    x9: pl.Tensor[[64], pl.FP32] = pl.yield_(x8)
                return x9

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)


class TestNestedChunkChainsInitSubstitution:
    """Tests that nested chunk chains correctly substitute init_values from parent chain."""

    def test_nested_chains_init_values_substituted(self):
        """Nested parallel chunk chains: inner chain init_values reference parent's
        rewritten iter_args, not the original pre-interchange names."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for b in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        for h in pl.parallel(0, 12, 1, chunk=4, chunk_policy="leading_full"):
                            x = pl.add(x, y)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                x0: pl.Tensor[[64], pl.FP32],
                y0: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                for b0, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    for h0, (x2,) in pl.parallel(
                        3, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        with pl.at(level=pl.Level.CORE_GROUP):
                            for b1, (x3,) in pl.parallel(
                                4, init_values=(x2,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                            ):
                                for h1, (x4,) in pl.parallel(
                                    4, init_values=(x3,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                                ):
                                    x5: pl.Tensor[[64], pl.FP32] = pl.tensor.add(x4, y0)
                                    x6: pl.Tensor[[64], pl.FP32] = pl.yield_(x5)
                                x7: pl.Tensor[[64], pl.FP32] = pl.yield_(x6)
                        x8: pl.Tensor[[64], pl.FP32] = pl.yield_(x7)
                    x9: pl.Tensor[[64], pl.FP32] = pl.yield_(x8)
                return x9

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)

    def test_nested_chains_outline_no_crash(self):
        """Nested parallel chunk chains followed by OutlineIncoreScopes must not crash.

        This is the end-to-end scenario from DeepSeekV3 decode that triggered the
        'Variable ... not found in symbol table' crash.
        """

        @pl.program
        class Input:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for b in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        for h in pl.parallel(0, 12, 1, chunk=4, chunk_policy="leading_full"):
                            x = pl.add(x, y)
                return x

        program = _prepare_for_interchange(Input)
        program = passes.interchange_chunk_loops()(program)
        # This should not raise "Variable ... not found in symbol table"
        program = passes.outline_incore_scopes()(program)

        incore_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.InCore]
        assert len(incore_funcs) >= 1

    def test_nested_chains_with_remainder_outline_no_crash(self):
        """Nested chains with remainder: outline must not crash on substituted init_values."""

        @pl.program
        class Input:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for b in pl.parallel(0, 6, 1, chunk=4, chunk_policy="leading_full"):
                        for h in pl.parallel(0, 14, 1, chunk=4, chunk_policy="leading_full"):
                            x = pl.add(x, y)
                return x

        program = _prepare_for_interchange(Input)
        program = passes.interchange_chunk_loops()(program)
        program = passes.outline_incore_scopes()(program)

        incore_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.InCore]
        assert len(incore_funcs) >= 1


class TestNestedChunksWithInterveningStatements:
    """Tests for nested chunked parallel loops with intervening statements (issue #911)."""

    @staticmethod
    def _make_input():
        @pl.program
        class Input:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for b in pl.parallel(0, 16, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, y)
                        for h in pl.parallel(0, 8, 1, chunk=2, chunk_policy="leading_full"):
                            x = pl.add(x, y)
                return x

        return Input

    def test_no_nested_incore_with_intervening_stmt(self):
        """Nested chunks with intervening add: single InCore, no nesting."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for b in pl.parallel(0, 16, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, y)
                        for h in pl.parallel(0, 8, 1, chunk=2, chunk_policy="leading_full"):
                            x = pl.add(x, y)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                x0: pl.Tensor[[64], pl.FP32],
                y0: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                for b0, (x1,) in pl.parallel(
                    4, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for b1, (x2,) in pl.parallel(
                            4, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x3: pl.Tensor[[64], pl.FP32] = pl.tensor.add(x2, y0)
                            for h0, (x4,) in pl.parallel(
                                4, init_values=(x3,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                            ):
                                for h1, (x5,) in pl.parallel(
                                    2, init_values=(x4,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                                ):
                                    x6: pl.Tensor[[64], pl.FP32] = pl.tensor.add(x5, y0)
                                    x7: pl.Tensor[[64], pl.FP32] = pl.yield_(x6)
                                x8: pl.Tensor[[64], pl.FP32] = pl.yield_(x7)
                            x9: pl.Tensor[[64], pl.FP32] = pl.yield_(x8)
                    x10: pl.Tensor[[64], pl.FP32] = pl.yield_(x9)
                return x10

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)

    def test_outline_no_crash_with_intervening_stmt(self):
        """Nested chunks with intervening stmt: outline must not crash."""
        program = _prepare_for_interchange(self._make_input())
        program = passes.interchange_chunk_loops()(program)
        # This must not crash with nested InCore or missing operator
        program = passes.outline_incore_scopes()(program)

        incore_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.InCore]
        assert len(incore_funcs) >= 1


class TestChunkWithRemainderInChain:
    """Tests for chunk chains that include remainder loops (non-divisible inner)."""

    def test_chunk_outer_inner_with_remainder_preserves_iter_args(self):
        """Chunk chain with trailing remainder: iter_args thread through inner, remainder preserved."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        for j in pl.parallel(0, 1, 1, chunk=2, chunk_policy="leading_full"):
                            x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i0, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for i1, (x2,) in pl.parallel(
                            4, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            for j0, (x3,) in pl.parallel(
                                1, init_values=(x2,), attrs={"loop_origin": pl.LoopOrigin.ChunkRemainder}
                            ):
                                x4: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x3, 1.0)
                                x5: pl.Tensor[[64], pl.FP32] = pl.yield_(x4)
                            x6: pl.Tensor[[64], pl.FP32] = pl.yield_(x5)
                    x7: pl.Tensor[[64], pl.FP32] = pl.yield_(x6)
                return x7

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)

    def test_chunk_with_remainder_body_contains_remainder_loop(self):
        """Remainder loop inside chain body is preserved after interchange.

        Same Before as ``test_chunk_outer_inner_with_remainder_preserves_iter_args``
        — the matching Expected confirms the remainder loop structurally survives.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        for j in pl.parallel(0, 1, 1, chunk=2, chunk_policy="leading_full"):
                            x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i0, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for i1, (x2,) in pl.parallel(
                            4, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            for j0, (x3,) in pl.parallel(
                                1, init_values=(x2,), attrs={"loop_origin": pl.LoopOrigin.ChunkRemainder}
                            ):
                                x4: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x3, 1.0)
                                x5: pl.Tensor[[64], pl.FP32] = pl.yield_(x4)
                            x6: pl.Tensor[[64], pl.FP32] = pl.yield_(x5)
                    x7: pl.Tensor[[64], pl.FP32] = pl.yield_(x6)
                return x7

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)


class TestRemainderLoops:
    """Tests for non-divisible cases with remainder loops."""

    def test_non_divisible_with_remainder(self):
        """Non-divisible with remainder: main chunk gets interchange, remainder gets InCore."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 6, 1, chunk=4, chunk_policy="leading_full"):
                        for j in pl.parallel(0, 14, 1, chunk=4, chunk_policy="leading_full"):
                            x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i0, (x1,) in pl.parallel(
                    1, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for i1, (x2,) in pl.parallel(
                            4, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            for j0, (x3,) in pl.parallel(
                                3, init_values=(x2,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                            ):
                                for j1, (x4,) in pl.parallel(
                                    4, init_values=(x3,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                                ):
                                    x5: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x4, 1.0)
                                    x6: pl.Tensor[[64], pl.FP32] = pl.yield_(x5)
                                x7: pl.Tensor[[64], pl.FP32] = pl.yield_(x6)
                            for j2, (x8,) in pl.parallel(
                                2, init_values=(x7,), attrs={"loop_origin": pl.LoopOrigin.ChunkRemainder}
                            ):
                                x9: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x8, 1.0)
                                x10: pl.Tensor[[64], pl.FP32] = pl.yield_(x9)
                            x11: pl.Tensor[[64], pl.FP32] = pl.yield_(x10)
                    x12: pl.Tensor[[64], pl.FP32] = pl.yield_(x11)
                for i2, (x13,) in pl.parallel(
                    2, init_values=(x12,), attrs={"loop_origin": pl.LoopOrigin.ChunkRemainder}
                ):
                    for j3, (x14,) in pl.parallel(
                        3, init_values=(x13,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        with pl.at(level=pl.Level.CORE_GROUP):
                            for j4, (x15,) in pl.parallel(
                                4, init_values=(x14,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                            ):
                                x16: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x15, 1.0)
                                x17: pl.Tensor[[64], pl.FP32] = pl.yield_(x16)
                        x18: pl.Tensor[[64], pl.FP32] = pl.yield_(x17)
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for j5, (x19,) in pl.parallel(
                            2, init_values=(x18,), attrs={"loop_origin": pl.LoopOrigin.ChunkRemainder}
                        ):
                            x20: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x19, 1.0)
                            x21: pl.Tensor[[64], pl.FP32] = pl.yield_(x20)
                    x22: pl.Tensor[[64], pl.FP32] = pl.yield_(x21)
                return x22

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)


class TestNonChunkedLoops:
    """Tests for loops that should pass through unchanged."""

    def test_non_chunked_loop_unchanged(self):
        """Regular (non-chunked) loops pass through untouched."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 10, 1):
                    x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i0, (x1,) in pl.range(10, init_values=(x0,)):
                    x2: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x1, 1.0)
                    x3: pl.Tensor[[64], pl.FP32] = pl.yield_(x2)
                return x3

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)


class TestSequentialChunks:
    """Tests for sequential chunked loops (should NOT interchange but get InCore wrapping)."""

    def test_sequential_chunk_gets_incore(self):
        """Sequential chunked loop inside auto_incore: gets InCore wrapping."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.range(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    for i0, (x1,) in pl.range(
                        2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        for i1, (x2,) in pl.range(
                            4, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x3: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x2, 1.0)
                            x4: pl.Tensor[[64], pl.FP32] = pl.yield_(x3)
                        x5: pl.Tensor[[64], pl.FP32] = pl.yield_(x4)
                return x5

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)

    def test_nested_sequential_chunks_get_incore(self):
        """Nested sequential chunked loops: no interchange, but get InCore wrapping."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.range(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        for j in pl.range(0, 12, 1, chunk=4, chunk_policy="leading_full"):
                            x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    for i0, (x1,) in pl.range(
                        2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        for i1, (x2,) in pl.range(
                            4, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            for j0, (x3,) in pl.range(
                                3, init_values=(x2,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                            ):
                                for j1, (x4,) in pl.range(
                                    4, init_values=(x3,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                                ):
                                    x5: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x4, 1.0)
                                    x6: pl.Tensor[[64], pl.FP32] = pl.yield_(x5)
                                x7: pl.Tensor[[64], pl.FP32] = pl.yield_(x6)
                            x8: pl.Tensor[[64], pl.FP32] = pl.yield_(x7)
                        x9: pl.Tensor[[64], pl.FP32] = pl.yield_(x8)
                return x9

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)


class TestExistingInCore:
    """Tests for loops with existing InCore scope (should skip interchange)."""

    def test_existing_incore_skip(self):
        """Body already has ScopeStmt(InCore): pass through unchanged by interchange."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        with pl.at(level=pl.Level.CORE_GROUP):
                            x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i0, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    for i1, (x2,) in pl.parallel(
                        4, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                    ):
                        with pl.at(level=pl.Level.CORE_GROUP):
                            x3: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x2, 1.0)
                        x4: pl.Tensor[[64], pl.FP32] = pl.yield_(x3)
                    x5: pl.Tensor[[64], pl.FP32] = pl.yield_(x4)
                return x5

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)


class TestAutoIncoreConsumed:
    """Tests that auto_incore scope is consumed by InterchangeChunkLoops."""

    def test_auto_incore_consumed(self):
        """AutoInCore scope should be removed after InterchangeChunkLoops.

        Same Before as ``TestSingleParallelChunk::test_single_parallel_chunk_gets_incore``
        — the Expected has no AutoInCore (``auto_chunk``) marker, structurally
        asserting the AutoInCore scope was consumed.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i0, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for i1, (x2,) in pl.parallel(
                            4, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x3: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x2, 1.0)
                            x4: pl.Tensor[[64], pl.FP32] = pl.yield_(x3)
                    x5: pl.Tensor[[64], pl.FP32] = pl.yield_(x4)
                return x5

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)


class TestPassProperties:
    """Tests for pass properties and factory."""

    def test_pass_name(self):
        """Pass has correct name."""
        p = passes.interchange_chunk_loops()
        assert p.get_name() == "InterchangeChunkLoops"

    def test_pass_required_properties(self):
        """Pass requires SSAForm (TypeChecked is a structural property)."""
        p = passes.interchange_chunk_loops()
        req = p.get_required_properties()
        assert req.contains(passes.IRProperty.SSAForm)

    def test_pass_produced_properties(self):
        """Pass produces SSAForm (TypeChecked is a structural property)."""
        p = passes.interchange_chunk_loops()
        prod = p.get_produced_properties()
        assert prod.contains(passes.IRProperty.SSAForm)


class TestNoNestedIncoreVerifier:
    """Tests for the NoNestedInCore structural property verifier (issue #912)."""

    def test_no_nested_incore_is_structural_property(self):
        """NoNestedInCore is in the structural property set."""
        structural = passes.get_structural_properties()
        assert structural.contains(passes.IRProperty.NoNestedInCore)

    def test_verifier_passes_on_valid_ir(self):
        """Verifier passes when InterchangeChunkLoops produces valid (non-nested) InCore."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 1.0)
                return x

        program = _prepare_for_interchange(Input)
        program = passes.interchange_chunk_loops()(program)

        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.NoNestedInCore)
        diagnostics = passes.PropertyVerifierRegistry.verify(props, program)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert len(errors) == 0

    def test_verifier_passes_with_intervening_stmts(self):
        """Verifier passes on fixed nested chunks with intervening statements."""

        @pl.program
        class Input:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for b in pl.parallel(0, 16, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, y)
                        for h in pl.parallel(0, 8, 1, chunk=2, chunk_policy="leading_full"):
                            x = pl.add(x, y)
                return x

        program = _prepare_for_interchange(Input)
        program = passes.interchange_chunk_loops()(program)

        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.NoNestedInCore)
        diagnostics = passes.PropertyVerifierRegistry.verify(props, program)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert len(errors) == 0


class TestNonChunkStatementsWrapping:
    """Tests that non-chunk statements inside auto_incore get InCore wrapping."""

    def test_standalone_tensor_op_wrapped(self):
        """Standalone tensor op inside auto_incore gets wrapped in InCore."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    x1: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x0, 1.0)
                return x1

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)

    def test_standalone_op_before_parallel_chunk(self):
        """Standalone op before parallel chunk: op wrapped separately, chunk interchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    x = pl.add(x, 1.0)
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 2.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    x1: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x0, 1.0)
                for i0, (x2,) in pl.parallel(
                    2, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for i1, (x3,) in pl.parallel(
                            4, init_values=(x2,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x4: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x3, 2.0)
                            x5: pl.Tensor[[64], pl.FP32] = pl.yield_(x4)
                    x6: pl.Tensor[[64], pl.FP32] = pl.yield_(x5)
                return x6

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)

    def test_standalone_op_after_parallel_chunk(self):
        """Standalone op after parallel chunk: chunk interchanged, op wrapped separately."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 2.0)
                    x = pl.mul(x, 3.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i0, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for i1, (x2,) in pl.parallel(
                            4, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x3: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x2, 2.0)
                            x4: pl.Tensor[[64], pl.FP32] = pl.yield_(x3)
                    x5: pl.Tensor[[64], pl.FP32] = pl.yield_(x4)
                with pl.at(level=pl.Level.CORE_GROUP):
                    x6: pl.Tensor[[64], pl.FP32] = pl.tensor.muls(x5, 3.0)
                return x6

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)

    def test_host_side_assemble_after_parallel_chunk_not_wrapped(self):
        """Host-side tail assemble after a chunk stays outside InCore."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[4], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                out_0: pl.Tensor[[8], pl.FP32] = pl.tensor.create(
                    [8], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 4, 1, chunk=2, chunk_policy="leading_full"):
                        x = pl.tensor.adds(x, 1.0)
                    out_1: pl.Tensor[[8], pl.FP32] = pl.tensor.assemble(out_0, x, [0])
                return out_1

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[4], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                out_0_0: pl.Tensor[[8], pl.FP32] = pl.tensor.create(
                    [8], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                for i0, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for i1, (x2,) in pl.parallel(
                            2, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x3: pl.Tensor[[4], pl.FP32] = pl.tensor.adds(x2, 1.0)
                            x4: pl.Tensor[[4], pl.FP32] = pl.yield_(x3)
                    x5: pl.Tensor[[4], pl.FP32] = pl.yield_(x4)
                out_1_0: pl.Tensor[[8], pl.FP32] = pl.tensor.assemble(out_0_0, x5, [0])
                return out_1_0

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)

    def test_multiple_parallel_chunks_no_regression(self):
        """Multiple parallel chunks with no standalone ops: all interchanged, no extra wrapping."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 1.0)
                    for j in pl.parallel(0, 12, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.mul(x, 2.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i0, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for i1, (x2,) in pl.parallel(
                            4, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x3: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x2, 1.0)
                            x4: pl.Tensor[[64], pl.FP32] = pl.yield_(x3)
                    x5: pl.Tensor[[64], pl.FP32] = pl.yield_(x4)
                for j0, (x6,) in pl.parallel(
                    3, init_values=(x5,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for j1, (x7,) in pl.parallel(
                            4, init_values=(x6,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x8: pl.Tensor[[64], pl.FP32] = pl.tensor.muls(x7, 2.0)
                            x9: pl.Tensor[[64], pl.FP32] = pl.yield_(x8)
                    x10: pl.Tensor[[64], pl.FP32] = pl.yield_(x9)
                return x10

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)

    def test_non_chunked_loop_inside_auto_incore_wrapped(self):
        """Non-chunked loop with tensor ops inside auto_incore gets wrapped in InCore."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.range(10):
                        x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    for i0, (x1,) in pl.range(10, init_values=(x0,)):
                        x2: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x1, 1.0)
                        x3: pl.Tensor[[64], pl.FP32] = pl.yield_(x2)
                return x3

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)

    def test_mixed_parallel_and_sequential_chunks(self):
        """Mixed parallel chunk + sequential chunk: parallel interchanged, sequential wrapped."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 1.0)
                    for j in pl.range(0, 12, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.mul(x, 2.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i0, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for i1, (x2,) in pl.parallel(
                            4, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x3: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x2, 1.0)
                            x4: pl.Tensor[[64], pl.FP32] = pl.yield_(x3)
                    x5: pl.Tensor[[64], pl.FP32] = pl.yield_(x4)
                with pl.at(level=pl.Level.CORE_GROUP):
                    for j0, (x6,) in pl.range(
                        3, init_values=(x5,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        for j1, (x7,) in pl.range(
                            4, init_values=(x6,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                        ):
                            x8: pl.Tensor[[64], pl.FP32] = pl.tensor.muls(x7, 2.0)
                            x9: pl.Tensor[[64], pl.FP32] = pl.yield_(x8)
                        x10: pl.Tensor[[64], pl.FP32] = pl.yield_(x9)
                return x10

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)


class TestScalarAssignmentNotWrapped:
    """Tests that pure scalar assignments stay outside InCore scopes."""

    def test_scalar_assign_adjacent_to_compute_not_wrapped(self):
        """Scalar assignment adjacent to tensor compute ops stays in orchestration."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for ob in pl.range(0, 8):
                        offset: pl.Scalar[pl.INDEX] = ob * 4  # noqa: F841
                        x = pl.add(x, 1.0)
                        for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                            x = pl.add(x, 2.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for ob0, (x1,) in pl.range(8, init_values=(x0,)):
                    offset0: pl.Scalar[pl.INDEX] = ob0 * 4
                    with pl.at(level=pl.Level.CORE_GROUP):
                        x2: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x1, 1.0)
                    for i0, (x3,) in pl.parallel(
                        2, init_values=(x2,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        with pl.at(level=pl.Level.CORE_GROUP):
                            for i1, (x4,) in pl.parallel(
                                4, init_values=(x3,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                            ):
                                x5: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x4, 2.0)
                                x6: pl.Tensor[[64], pl.FP32] = pl.yield_(x5)
                        x7: pl.Tensor[[64], pl.FP32] = pl.yield_(x6)
                    x8: pl.Tensor[[64], pl.FP32] = pl.yield_(x7)
                return x8

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        ir.assert_structural_equal(After, Expected)

    def test_scalar_assign_not_wrapped_outline_no_crash(self):
        """Scalar assignment stays in orchestration after outline — no undefined variable."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for ob in pl.range(0, 8):
                        offset: pl.Scalar[pl.INDEX] = ob * 4  # noqa: F841
                        for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                            x = pl.add(x, 2.0)
                return x

        program = _prepare_for_interchange(Input)
        program = passes.interchange_chunk_loops()(program)
        # This should not crash with undefined variable references
        program = passes.outline_incore_scopes()(program)

        incore_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.InCore]
        assert len(incore_funcs) >= 1


class TestEndToEndNoComputeLeaks:
    """End-to-end tests verifying no compute tensor ops leak into Orchestration."""

    def _run_through_outline(self, program):
        """Run prerequisite passes + interchange + outline."""
        program = _prepare_for_interchange(program)
        program = passes.interchange_chunk_loops()(program)
        program = passes.outline_incore_scopes()(program)
        return program

    # Host-side tensor ops that are allowed in Orchestration
    _HOST_SIDE_OPS = {
        "tensor.create",
        "tensor.read",
        "tensor.write",
        "tensor.slice",
        "tensor.assemble",
        "tensor.dim",
        "tensor.reshape",
        "tensor.transpose",
    }

    def _assert_no_compute_leaks(self, program, min_incore_funcs=1):
        """Assert no compute tensor ops in Orchestration and enough InCore functions exist."""
        for func in program.functions.values():
            if func.func_type == ir.FunctionType.Orchestration:
                func_str = python_print(func)
                for match in re.findall(r"tensor\.\w+", func_str):
                    assert match in self._HOST_SIDE_OPS, (
                        f"Compute tensor op '{match}' leaked into Orchestration"
                    )

        incore_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.InCore]
        assert len(incore_funcs) >= min_incore_funcs

    def test_standalone_op_outlined(self):
        """Standalone op inside auto_incore: outlined into InCore function."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    x = pl.add(x, 1.0)
                return x

        After = self._run_through_outline(Input)
        self._assert_no_compute_leaks(After, min_incore_funcs=1)

    def test_mix_standalone_and_parallel_chunk_outlined(self):
        """Mix of standalone + parallel chunk: two InCore functions, orchestration clean."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    x = pl.add(x, 1.0)
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 2.0)
                return x

        After = self._run_through_outline(Input)
        self._assert_no_compute_leaks(After, min_incore_funcs=2)

    def test_sequential_chunk_outlined(self):
        """Sequential chunk inside auto_incore: one InCore function containing the whole loop chain."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.range(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 1.0)
                return x

        After = self._run_through_outline(Input)
        self._assert_no_compute_leaks(After, min_incore_funcs=1)


class TestNameHintPropagation:
    """Tests that ``name_hint`` on AutoInCore is propagated to generated InCore scopes.

    The pass creates InCoreScopeStmts at several sites (interchange wrapping,
    sub-remainder wrapping, non-chunk statement wrapping, yield-aware wrapping).
    All of them must carry the AutoInCore's ``name_hint`` so the downstream
    outlining pass can name the resulting function.
    """

    @staticmethod
    def _collect_incore_scopes(program: ir.Program) -> list[ir.InCoreScopeStmt]:
        found: list[ir.InCoreScopeStmt] = []

        class Collector(ir.IRVisitor):
            def visit_in_core_scope_stmt(self, op: ir.InCoreScopeStmt) -> None:
                found.append(op)
                super().visit_in_core_scope_stmt(op)

        Collector().visit_program(program)
        return found

    def _assert_all_incore_have_hint(self, program: ir.Program, hint: str) -> None:
        scopes = self._collect_incore_scopes(program)
        assert scopes, "expected at least one InCoreScopeStmt in the program"
        for scope in scopes:
            assert scope.name_hint == hint, (
                f"InCoreScopeStmt has name_hint={scope.name_hint!r}, expected {hint!r}"
            )

    def test_single_parallel_chunk_propagates_hint(self):
        """Single parallel chunk: generated InCore carries the AutoInCore name_hint."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    optimizations=[pl.auto_chunk],
                    name_hint="my_kernel",
                ):
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 1.0)
                return x

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        self._assert_all_incore_have_hint(After, "my_kernel")

    def test_nested_parallel_chunks_propagate_hint(self):
        """Two nested parallel chunks (full interchange): all InCores get the hint."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    optimizations=[pl.auto_chunk],
                    name_hint="nested_hint",
                ):
                    for b in pl.parallel(0, 16, 1, chunk=4, chunk_policy="leading_full"):
                        for h in pl.parallel(0, 8, 1, chunk=2, chunk_policy="leading_full"):
                            x = pl.add(x, y)
                return x

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        self._assert_all_incore_have_hint(After, "nested_hint")

    def test_non_chunk_statement_wrapping_propagates_hint(self):
        """Standalone tensor op inside AutoInCore: the wrapping InCore gets the hint."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    optimizations=[pl.auto_chunk],
                    name_hint="wrap_hint",
                ):
                    x = pl.add(x, 1.0)
                return x

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        self._assert_all_incore_have_hint(After, "wrap_hint")

    def test_empty_name_hint_default(self):
        """No name_hint provided: generated InCores have empty name_hint (regression guard)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 1.0)
                return x

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        self._assert_all_incore_have_hint(After, "")

    def test_chunk_remainder_wrapping_propagates_hint(self):
        """Non-divisible nested chunks produce ChunkRemainder sub-loops; the
        remainder-wrapping path (WrapSubRemainderLoopsInInCore) must also
        propagate the hint.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    optimizations=[pl.auto_chunk],
                    name_hint="remainder_hint",
                ):
                    for b in pl.parallel(0, 6, 1, chunk=4, chunk_policy="leading_full"):
                        for h in pl.parallel(0, 14, 1, chunk=4, chunk_policy="leading_full"):
                            x = pl.add(x, y)
                return x

        After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        self._assert_all_incore_have_hint(After, "remainder_hint")


def _is_block_idx_assign(stmt: ir.Stmt) -> bool:
    if not isinstance(stmt, ir.AssignStmt):
        return False
    call = stmt.value
    if isinstance(call, ir.Call) and isinstance(call.op, ir.Op):
        return call.op.name == "tile.get_block_idx"
    return False


def _walk_stmt_subtree(node: ir.Stmt, visit: Callable[[ir.Stmt], None]) -> None:
    """Visit ``node`` and recursively walk nested statement containers."""
    visit(node)
    if isinstance(node, ir.SeqStmts):
        for child in node.stmts:
            _walk_stmt_subtree(child, visit)
    elif isinstance(node, ir.ScopeStmt):
        _walk_stmt_subtree(node.body, visit)
    elif isinstance(node, ir.ForStmt):
        _walk_stmt_subtree(node.body, visit)
    elif isinstance(node, ir.WhileStmt):
        _walk_stmt_subtree(node.body, visit)
    elif isinstance(node, ir.IfStmt):
        _walk_stmt_subtree(node.then_body, visit)
        if node.else_body is not None:
            _walk_stmt_subtree(node.else_body, visit)


def _find_spmd_scope(func: ir.Function) -> ir.SpmdScopeStmt:
    found: list[ir.SpmdScopeStmt] = []

    def visit(node: ir.Stmt) -> None:
        if isinstance(node, ir.SpmdScopeStmt):
            found.append(node)

    _walk_stmt_subtree(func.body, visit)
    assert len(found) == 1, f"expected exactly one SpmdScopeStmt, got {len(found)}"
    return found[0]


def _find_incore_whose_body_starts_with_block_idx(stmt: ir.Stmt) -> ir.InCoreScopeStmt | None:
    if isinstance(stmt, ir.InCoreScopeStmt):
        if _is_block_idx_assign(_first_stmt_in_block(stmt.body)):
            return stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            found = _find_incore_whose_body_starts_with_block_idx(child)
            if found is not None:
                return found
    if isinstance(stmt, ir.ScopeStmt):
        return _find_incore_whose_body_starts_with_block_idx(stmt.body)
    if isinstance(stmt, ir.ForStmt):
        return _find_incore_whose_body_starts_with_block_idx(stmt.body)
    if isinstance(stmt, ir.WhileStmt):
        return _find_incore_whose_body_starts_with_block_idx(stmt.body)
    if isinstance(stmt, ir.IfStmt):
        found = _find_incore_whose_body_starts_with_block_idx(stmt.then_body)
        if found is not None:
            return found
        if stmt.else_body is not None:
            return _find_incore_whose_body_starts_with_block_idx(stmt.else_body)
    return None


def _first_stmt_in_block(stmt: ir.Stmt) -> ir.Stmt:
    if isinstance(stmt, ir.SeqStmts) and stmt.stmts:
        return stmt.stmts[0]
    return stmt


def _block_idx_is_sibling_before_incore(spmd_body: ir.Stmt) -> bool:
    """True when ``tile.get_block_idx`` is a direct seq child before an InCore sibling."""
    if isinstance(spmd_body, ir.AutoInCoreScopeStmt):
        spmd_body = spmd_body.body
    if isinstance(spmd_body, ir.InCoreScopeStmt):
        return False
    if not isinstance(spmd_body, ir.SeqStmts):
        return False
    stmts = spmd_body.stmts
    if len(stmts) < 2 or not _is_block_idx_assign(stmts[0]):
        return False
    for index, stmt in enumerate(stmts):
        if isinstance(stmt, ir.InCoreScopeStmt):
            return index > 0
    return False


class TestSpmdBlockIdxPreludeMerge:
    """``InterchangeChunkLoops`` merges leading ``tile.get_block_idx`` prelude into the
    first ``InCoreScopeStmt`` directly under ``SpmdScopeStmt`` (pass-9 behavior).
    """

    def test_interchange_merges_block_idx_into_first_incore_under_spmd(self):
        """After interchange, block index binding lives inside the first InCore, not as a
        sibling ahead of it under the Spmd body."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4):
                    with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                        for j in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                            offset = i * 128
                            tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                            out = pl.store(tile_a, [offset, 0], out)
                return out

        # Pass verification/roundtrip off: Spmd+AutoInCore IR after interchange does
        # not always satisfy default post-pass checks for this orchestration shape.
        with passes.PassContext([]):
            prepared = _prepare_for_interchange(Before)
            main_func = list(prepared.functions.values())[0]
            before_spmd = _find_spmd_scope(main_func)

            After = passes.interchange_chunk_loops()(prepared)
            after_spmd = _find_spmd_scope(list(After.functions.values())[0])

        assert before_spmd.body is not after_spmd.body, "interchange should rewrite the Spmd body"
        assert not _block_idx_is_sibling_before_incore(after_spmd.body), (
            "tile.get_block_idx prelude should be merged into the first InCore, "
            "not left as a sibling before it"
        )

        incore = _find_incore_whose_body_starts_with_block_idx(after_spmd.body)
        assert incore is not None, (
            "expected an InCore under Spmd whose body starts with tile.get_block_idx after merge"
        )

    def test_merged_spmd_incore_prints_as_nested_with_blocks(self):
        """Printer regression: after the prelude-merge rewrite, the IR must
        still print as ``with pl.spmd(...):`` containing ``with pl.at(...):``.

        The merge wraps the rewritten Spmd body in a single-element SeqStmts so
        the Python printer emits the nested ``with`` shape. Any printer change
        that flattens single-element sequences could silently break this and
        emit the prelude as siblings outside the InCore again.

        Roundtrip verification is disabled inside this test because intermediate
        passes produce a ``SpmdScopeStmt(InCoreScopeStmt(...))`` shape that
        the printer currently emits as ``with pl.spmd(...):`` with a non-call
        body — a form the parser rejects. That's a separate printer bug (see
        KNOWN_ISSUES.md). We only assert the *final* printed form here.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4):
                    with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                        for j in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                            offset = i * 128
                            tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                            out = pl.store(tile_a, [offset, 0], out)
                return out

        # Override the autouse PassContext with an empty instrument set so the
        # intermediate-pass roundtrip (which trips a separate printer bug) does
        # not preempt the assertions below.
        with passes.PassContext([]):
            After = passes.interchange_chunk_loops()(_prepare_for_interchange(Before))
        main_func = list(After.functions.values())[0]
        printed = python_print(main_func)

        # Both scope markers should appear, and pl.at must be nested under pl.spmd.
        spmd_line = next(
            (i for i, line in enumerate(printed.splitlines()) if "pl.spmd(" in line),
            -1,
        )
        at_line = next(
            (i for i, line in enumerate(printed.splitlines()) if "pl.at(level=pl.Level.CORE_GROUP" in line),
            -1,
        )
        assert spmd_line >= 0, f"expected 'pl.spmd(' in printed output:\n{printed}"
        assert at_line >= 0, (
            f"expected nested 'pl.at(level=pl.Level.CORE_GROUP' in printed output:\n{printed}"
        )
        assert at_line > spmd_line, (
            f"pl.at(level=...) must appear after pl.spmd(...) in printed output:\n{printed}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

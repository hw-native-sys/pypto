# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for non-parallel code inside auto_incore losing InCore scope.

Root cause
----------
InterchangeChunkLoops consumes ``auto_incore`` and wraps each interchanged
parallel chunk body in ``ScopeStmt(InCore)``.  However, non-parallel code
(range loops, straight-line ops) that sits *between* parallel chunk loops
inside the same ``auto_incore`` scope is left without an InCore wrapper.

``WrapNonIncoreStatementsInInCore`` only operates on the direct children of
the ``auto_incore`` body.  When the body is a single ``ForStmt`` (e.g. a
``pl.range`` loop) whose body *contains* InCore scopes from the interchanged
parallel chunks, ``ContainsInCoreScope`` returns ``True`` for the entire
``ForStmt``, so the function returns it as-is — leaving non-parallel code
inside the loop body unwrapped.

Consequence: ``OutlineIncoreScopes`` cannot outline these unwrapped
operations, so they stay in the Orchestration function as bare tensor ops
(including matmul), which downstream passes (ConvertTensorToTileOps,
ExpandMixedKernel, etc.) cannot process correctly.

This reproduces the issue observed in the Qwen3SingleLayerDecode model where
the MLP gate/up projection matmuls remained in the Orchestration function.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _run_pipeline(program):
    """Run prerequisite passes plus interchange + outline.

    This is the full pipeline exercised by these tests: it reproduces the
    setup that triggered the original bug (parallel chunks + non-parallel
    code inside auto_incore).
    """
    program = passes.unroll_loops()(program)
    program = passes.convert_to_ssa()(program)
    program = passes.flatten_call_expr()(program)
    program = passes.split_chunked_loops()(program)
    program = passes.interchange_chunk_loops()(program)
    program = passes.outline_incore_scopes()(program)
    return program


class TestNonParallelCodeBetweenChunks:
    """Non-parallel code between parallel chunk loops inside auto_incore
    must be wrapped in InCore scope so that OutlineIncoreScopes can outline it."""

    def test_interleaved_scalar_op_gets_incore(self):
        """A scalar op between two parallel chunks must get an InCore scope."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for b in pl.range(0, 8, 4):
                        for i in pl.parallel(4, chunk=2, chunk_policy="leading_full"):
                            x = pl.tensor.adds(x, 1.0)
                        y: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x, 2.0)
                        for j in pl.parallel(4, chunk=2, chunk_policy="leading_full"):
                            x = pl.tensor.add(x, y)
                return x

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_0(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for i1, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                ):
                    x2: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(x1, 1.0)
                    x3: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x2)
                return x3

            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_1(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                y0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x0, 2.0)
                return y0

            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_2(
                self,
                x0: pl.Tensor[[8, 64], pl.FP32],
                y0: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                for j1, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                ):
                    x2: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.add(x1, y0)
                    x3: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x2)
                return x3

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for b0, (x1,) in pl.range(0, 8, 4, init_values=(x0,)):
                    for i0, (x2,) in pl.parallel(
                        2, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        x3: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_0(x2)
                        x4: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x3)
                    y0: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_1(x4)
                    for j0, (x5,) in pl.parallel(
                        2, init_values=(x4,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        x6: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_2(x5, y0)
                        x7: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x6)
                    x8: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x7)
                return x8

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_interleaved_range_loop_gets_incore(self):
        """A range loop between parallel chunks must get an InCore scope.

        This mirrors the Qwen3 MLP pattern: a pl.range() loop containing
        matmul sits between two pl.parallel() chunk loops.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
                w: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for b in pl.range(0, 8, 4):
                        for i in pl.parallel(4, chunk=2, chunk_policy="leading_full"):
                            x = pl.tensor.adds(x, 1.0)
                        for k in pl.range(2):
                            x = pl.tensor.matmul(x, w)
                        for j in pl.parallel(4, chunk=2, chunk_policy="leading_full"):
                            x = pl.tensor.adds(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_0(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for i1, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                ):
                    x2: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(x1, 1.0)
                    x3: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x2)
                return x3

            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_1(
                self,
                x0: pl.Tensor[[8, 64], pl.FP32],
                w0: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                for k0, (x1,) in pl.range(2, init_values=(x0,)):
                    x2: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.matmul(
                        x1, w0, a_trans=False, b_trans=False, c_matrix_nz=False
                    )
                    x3: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x2)
                return x3

            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_2(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for j1, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                ):
                    x2: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(x1, 1.0)
                    x3: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x2)
                return x3

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x0: pl.Tensor[[8, 64], pl.FP32],
                w0: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                for b0, (x1,) in pl.range(0, 8, 4, init_values=(x0,)):
                    for i0, (x2,) in pl.parallel(
                        2, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        x3: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_0(x2)
                        x4: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x3)
                    x5: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_1(x4, w0)
                    for j0, (x6,) in pl.parallel(
                        2, init_values=(x5,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        x7: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_2(x6)
                        x8: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x7)
                    x9: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x8)
                return x9

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_all_ops_outlined_end_to_end(self):
        """End-to-end: all compute ops inside auto_incore must be outlined.

        Same structure as ``test_interleaved_scalar_op_gets_incore`` — this
        test is retained as a stronger end-to-end check (same expected output).
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for b in pl.range(0, 8, 4):
                        for i in pl.parallel(4, chunk=2, chunk_policy="leading_full"):
                            x = pl.tensor.adds(x, 1.0)
                        y: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x, 2.0)
                        for j in pl.parallel(4, chunk=2, chunk_policy="leading_full"):
                            x = pl.tensor.add(x, y)
                return x

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_0(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for i1, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                ):
                    x2: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(x1, 1.0)
                    x3: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x2)
                return x3

            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_1(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                y0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x0, 2.0)
                return y0

            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_2(
                self,
                x0: pl.Tensor[[8, 64], pl.FP32],
                y0: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                for j1, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                ):
                    x2: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.add(x1, y0)
                    x3: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x2)
                return x3

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for b0, (x1,) in pl.range(0, 8, 4, init_values=(x0,)):
                    for i0, (x2,) in pl.parallel(
                        2, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        x3: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_0(x2)
                        x4: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x3)
                    y0: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_1(x4)
                    for j0, (x5,) in pl.parallel(
                        2, init_values=(x4,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        x6: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_2(x5, y0)
                        x7: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x6)
                    x8: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x7)
                return x8

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


class TestNestedForStmtRecursion:
    """The fix recurses into ForStmt bodies that contain InCore scopes.
    These tests verify the recursion works for deeper nesting and edge cases."""

    def test_doubly_nested_range_with_interleaved_op(self):
        """Non-parallel op inside a doubly nested range loop must get InCore scope."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for b in pl.range(0, 8, 4):
                        for c in pl.range(2):
                            for i in pl.parallel(4, chunk=2, chunk_policy="leading_full"):
                                x = pl.tensor.adds(x, 1.0)
                            y: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x, 3.0)
                            for j in pl.parallel(4, chunk=2, chunk_policy="leading_full"):
                                x = pl.tensor.add(x, y)
                return x

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_0(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for i1, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                ):
                    x2: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(x1, 1.0)
                    x3: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x2)
                return x3

            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_1(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                y0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x0, 3.0)
                return y0

            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_2(
                self,
                x0: pl.Tensor[[8, 64], pl.FP32],
                y0: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                for j1, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                ):
                    x2: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.add(x1, y0)
                    x3: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x2)
                return x3

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for b0, (x1,) in pl.range(0, 8, 4, init_values=(x0,)):
                    for c0, (x2,) in pl.range(2, init_values=(x1,)):
                        for i0, (x3,) in pl.parallel(
                            2, init_values=(x2,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                        ):
                            x4: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_0(x3)
                            x5: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x4)
                        y0: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_1(x5)
                        for j0, (x6,) in pl.parallel(
                            2, init_values=(x5,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                        ):
                            x7: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_2(x6, y0)
                            x8: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x7)
                        x9: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x8)
                    x10: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x9)
                return x10

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_single_forstmt_body_with_mixed_children(self):
        """auto_incore body is a single ForStmt (not SeqStmts).

        This is the exact trigger for the original bug: ContainsInCoreScope
        returns True for the ForStmt, so the old code returned it as-is
        without examining its children.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for b in pl.range(0, 8, 4):
                        for i in pl.parallel(4, chunk=2, chunk_policy="leading_full"):
                            x = pl.tensor.adds(x, 1.0)
                        x = pl.tensor.muls(x, 2.0)
                return x

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_0(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for i1, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                ):
                    x2: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(x1, 1.0)
                    x3: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x2)
                return x3

            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_1(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                x1: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x0, 2.0)
                return x1

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for b0, (x1,) in pl.range(0, 8, 4, init_values=(x0,)):
                    for i0, (x2,) in pl.parallel(
                        2, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        x3: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_0(x2)
                        x4: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x3)
                    x5: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_1(x4)
                    x6: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x5)
                return x6

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_multiple_non_parallel_ops_between_chunks(self):
        """Multiple consecutive non-parallel ops between chunks must all be wrapped."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for b in pl.range(0, 8, 4):
                        for i in pl.parallel(4, chunk=2, chunk_policy="leading_full"):
                            x = pl.tensor.adds(x, 1.0)
                        y: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x, 2.0)
                        z: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.add(x, y)
                        x = pl.tensor.muls(z, 0.5)
                        for j in pl.parallel(4, chunk=2, chunk_policy="leading_full"):
                            x = pl.tensor.adds(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_0(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for i1, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                ):
                    x2: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(x1, 1.0)
                    x3: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x2)
                return x3

            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_1(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                y0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x0, 2.0)
                z0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.add(x0, y0)
                x1: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(z0, 0.5)
                return x1

            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_2(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for j1, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                ):
                    x2: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(x1, 1.0)
                    x3: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x2)
                return x3

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for b0, (x1,) in pl.range(0, 8, 4, init_values=(x0,)):
                    for i0, (x2,) in pl.parallel(
                        2, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        x3: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_0(x2)
                        x4: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x3)
                    x5: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_1(x4)
                    for j0, (x6,) in pl.parallel(
                        2, init_values=(x5,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                    ):
                        x7: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_2(x6)
                        x8: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x7)
                    x9: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x8)
                return x9

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_no_parallel_chunks_no_wrapping(self):
        """auto_incore with only non-parallel code (no chunks) should not crash.

        When there are no interchanged parallel chunks, there are no InCore
        scopes to trigger recursion. The function should still work correctly —
        the whole body becomes a single InCore function.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for b in pl.range(0, 8, 4):
                        x = pl.tensor.adds(x, 1.0)
                        x = pl.tensor.muls(x, 2.0)
                return x

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_0(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for b0, (x1,) in pl.range(0, 8, 4, init_values=(x0,)):
                    x2: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(x1, 1.0)
                    x3: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x2, 2.0)
                    x4: pl.Tensor[[8, 64], pl.FP32] = pl.yield_(x3)
                return x4

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x0: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                x1: pl.Tensor[[8, 64], pl.FP32] = self.main_incore_0(x0)
                return x1

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


class TestHostSideTailOps:
    """Host-side tensor ops may stay in Orchestration after outline."""

    def test_tail_assemble_after_parallel_chunk_stays_in_orchestration(self):
        """A trailing tensor.assemble should remain in the Orchestration function."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[4], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                out_0: pl.Tensor[[8], pl.FP32] = pl.tensor.create(
                    [8], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for i in pl.parallel(0, 4, 1, chunk=2, chunk_policy="leading_full"):
                        x = pl.tensor.adds(x, 1.0)
                    out_1: pl.Tensor[[8], pl.FP32] = pl.tensor.assemble(out_0, x, [0])
                return out_1

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_0(self, x0: pl.Tensor[[4], pl.FP32]) -> pl.Tensor[[4], pl.FP32]:
                for i1, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                ):
                    x2: pl.Tensor[[4], pl.FP32] = pl.tensor.adds(x1, 1.0)
                    x3: pl.Tensor[[4], pl.FP32] = pl.yield_(x2)
                return x3

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x0: pl.Tensor[[4], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                out_0: pl.Tensor[[8], pl.FP32] = pl.tensor.create(
                    [8], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                for i0, (x1,) in pl.parallel(
                    2, init_values=(x0,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    x2: pl.Tensor[[4], pl.FP32] = self.main_incore_0(x1)
                    x3: pl.Tensor[[4], pl.FP32] = pl.yield_(x2)
                out_1: pl.Tensor[[8], pl.FP32] = pl.tensor.assemble(out_0, x3, [0])
                return out_1

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

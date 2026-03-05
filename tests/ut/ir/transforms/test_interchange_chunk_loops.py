# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for InterchangeChunkLoops pass."""

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
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)

        # Verify structure: outer → InCore { inner → body }
        func = list(After.functions.values())[0]
        stmts = list(func.body.stmts)  # type: ignore[attr-defined]

        outer_for = stmts[0]
        assert outer_for.loop_origin == ir.LoopOrigin.ChunkOuter
        assert outer_for.kind == ir.ForKind.Sequential

        # Outer body = SeqStmts [InCore, yield]
        outer_body_stmts = list(outer_for.body.stmts)
        scope_stmt = outer_body_stmts[0]
        assert scope_stmt.scope_kind == ir.ScopeKind.InCore

        # InCore body = inner ForStmt
        inner_for = scope_stmt.body
        assert inner_for.loop_origin == ir.LoopOrigin.ChunkInner
        assert inner_for.kind == ir.ForKind.Parallel


class TestNestedParallelChunks:
    """Tests for nested parallel chunked loops (full interchange + InCore)."""

    def test_two_nested_parallel_divisible(self):
        """Two nested parallel chunked loops, divisible: full interchange + InCore."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        for j in pl.parallel(0, 12, 1, chunk=4):
                            x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)

        # Verify structure: i_out → j_out → InCore { i_in → j_in → body }
        func = list(After.functions.values())[0]
        stmts = list(func.body.stmts)  # type: ignore[attr-defined]

        # i_out
        i_out = stmts[0]
        assert i_out.loop_origin == ir.LoopOrigin.ChunkOuter
        assert i_out.kind == ir.ForKind.Sequential

        # j_out inside i_out body
        i_out_body = list(i_out.body.stmts)
        j_out = i_out_body[0]
        assert j_out.loop_origin == ir.LoopOrigin.ChunkOuter
        assert j_out.kind == ir.ForKind.Sequential

        # InCore inside j_out body
        j_out_body = list(j_out.body.stmts)
        scope = j_out_body[0]
        assert scope.scope_kind == ir.ScopeKind.InCore

        # i_in inside InCore
        i_in = scope.body
        assert i_in.loop_origin == ir.LoopOrigin.ChunkInner
        assert i_in.kind == ir.ForKind.Parallel

        # j_in inside i_in body
        i_in_body = list(i_in.body.stmts)
        j_in = i_in_body[0]
        assert j_in.loop_origin == ir.LoopOrigin.ChunkInner
        assert j_in.kind == ir.ForKind.Parallel

    def test_two_nested_parallel_with_iter_args(self):
        """Two nested parallel chunked loops with iter_args: verify SSA threading."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        for j in pl.parallel(0, 12, 1, chunk=4):
                            x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)

        # Verify iter_args are correctly threaded
        func = list(After.functions.values())[0]
        stmts = list(func.body.stmts)  # type: ignore[attr-defined]
        i_out = stmts[0]

        # i_out should have iter_args (from x)
        assert len(i_out.iter_args) == 1
        assert len(i_out.return_vars) == 1

        # j_out should have iter_args chained from i_out
        i_out_body = list(i_out.body.stmts)
        j_out = i_out_body[0]
        assert len(j_out.iter_args) == 1
        assert len(j_out.return_vars) == 1

        # InCore → i_in → j_in all with iter_args
        j_out_body = list(j_out.body.stmts)
        scope = j_out_body[0]
        i_in = scope.body
        assert len(i_in.iter_args) == 1
        assert len(i_in.return_vars) == 1

        i_in_body = list(i_in.body.stmts)
        j_in = i_in_body[0]
        assert len(j_in.iter_args) == 1
        assert len(j_in.return_vars) == 1


class TestRemainderLoops:
    """Tests for non-divisible cases with remainder loops."""

    def test_non_divisible_with_remainder(self):
        """Non-divisible with remainder: main chunk gets interchange, remainder gets InCore."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.parallel(0, 6, 1, chunk=4):
                        for j in pl.parallel(0, 14, 1, chunk=4):
                            x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)

        func = list(After.functions.values())[0]
        stmts = list(func.body.stmts)  # type: ignore[attr-defined]

        # Main chunk pair: i_out → j_out → InCore { i_in → j_in → body }
        i_out = stmts[0]
        assert i_out.loop_origin == ir.LoopOrigin.ChunkOuter

        # Remainder: i_rem contains j_out→InCore{j_in} + InCore{j_rem}
        i_rem = stmts[1]
        assert i_rem.loop_origin == ir.LoopOrigin.ChunkRemainder

        # Inside i_rem body, look for InCore scopes
        i_rem_body = list(i_rem.body.stmts)

        # j_out should have InCore wrapping j_in inside its body
        j_out_in_rem = i_rem_body[0]
        assert j_out_in_rem.loop_origin == ir.LoopOrigin.ChunkOuter
        j_out_body = list(j_out_in_rem.body.stmts)
        assert j_out_body[0].scope_kind == ir.ScopeKind.InCore

        # j_rem should be wrapped in InCore
        j_rem_incore = i_rem_body[1]
        assert j_rem_incore.scope_kind == ir.ScopeKind.InCore


class TestNonChunkedLoops:
    """Tests for loops that should pass through unchanged."""

    def test_non_chunked_loop_unchanged(self):
        """Regular (non-chunked) loops pass through untouched."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 10, 1):
                    x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_interchange(Input)
        before_str = python_print(Before)
        After = passes.interchange_chunk_loops()(Before)
        after_str = python_print(After)

        assert before_str == after_str


class TestSequentialChunks:
    """Tests for sequential chunked loops (should NOT interchange)."""

    def test_sequential_chunk_skip(self):
        """Sequential chunked loop (pl.range with chunk): no interchange, no InCore."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.range(0, 8, 1, chunk=4):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)
        after_str = python_print(After)

        # AutoInCore is consumed, but sequential loops are not interchanged
        assert "auto_incore" not in after_str
        # Verify loop structure preserved (outer→inner, no InCore inserted)
        func = list(After.functions.values())[0]
        stmts = list(func.body.stmts)  # type: ignore[attr-defined]
        outer_for = stmts[0]
        assert outer_for.loop_origin == ir.LoopOrigin.ChunkOuter
        assert outer_for.kind == ir.ForKind.Sequential

    def test_nested_sequential_chunks_skip(self):
        """Nested sequential chunked loops: no interchange."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.range(0, 8, 1, chunk=4):
                        for j in pl.range(0, 12, 1, chunk=4):
                            x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)
        after_str = python_print(After)

        # AutoInCore consumed, sequential loops not interchanged
        assert "auto_incore" not in after_str
        assert "incore" not in after_str
        # Verify nested sequential structure preserved
        func = list(After.functions.values())[0]
        stmts = list(func.body.stmts)  # type: ignore[attr-defined]
        outer_for = stmts[0]
        assert outer_for.loop_origin == ir.LoopOrigin.ChunkOuter
        assert outer_for.kind == ir.ForKind.Sequential


class TestExistingInCore:
    """Tests for loops with existing InCore scope (should skip)."""

    def test_existing_incore_skip(self):
        """Body already has ScopeStmt(InCore): pass through unchanged."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        with pl.incore():
                            x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)
        after_str = python_print(After)

        # AutoInCore is consumed but existing InCore prevents interchange
        assert "auto_incore" not in after_str


class TestAutoIncoreConsumed:
    """Tests that auto_incore scope is consumed by InterchangeChunkLoops."""

    def test_auto_incore_consumed(self):
        """AutoInCore scope should be removed after InterchangeChunkLoops."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)
        after_str = python_print(After)

        assert "auto_incore" not in after_str

    def test_loops_outside_auto_incore_not_interchanged(self):
        """Loops outside auto_incore are not interchanged."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.parallel(0, 8, 1, chunk=4):
                    x = pl.add(x, 1.0)
                return x

        # Without auto_incore, split_chunked_loops won't split, so
        # interchange also has nothing to do
        Before = _prepare_for_interchange(Input)
        before_str = python_print(Before)
        After = passes.interchange_chunk_loops()(Before)
        after_str = python_print(After)

        assert before_str == after_str


class TestPassProperties:
    """Tests for pass properties and factory."""

    def test_pass_name(self):
        """Pass has correct name."""
        p = passes.interchange_chunk_loops()
        assert p.get_name() == "InterchangeChunkLoops"

    def test_pass_required_properties(self):
        """Pass requires TypeChecked and SSAForm."""
        p = passes.interchange_chunk_loops()
        req = p.get_required_properties()
        assert req.contains(passes.IRProperty.TypeChecked)
        assert req.contains(passes.IRProperty.SSAForm)

    def test_pass_produced_properties(self):
        """Pass produces TypeChecked and SSAForm."""
        p = passes.interchange_chunk_loops()
        prod = p.get_produced_properties()
        assert prod.contains(passes.IRProperty.TypeChecked)
        assert prod.contains(passes.IRProperty.SSAForm)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

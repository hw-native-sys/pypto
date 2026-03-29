# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for InterchangeChunkLoops pass."""

import re
from typing import cast

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


def _top_level_stmts(program: ir.Program) -> list[ir.Stmt]:
    """Return the first function's top-level statements."""
    func = list(program.functions.values())[0]
    return list(cast(ir.SeqStmts, func.body).stmts)


def _body_stmts(stmt: ir.Stmt) -> list[ir.Stmt]:
    """Return child statements from a SeqStmts body."""
    return list(cast(ir.SeqStmts, stmt).stmts)


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
        stmts = _top_level_stmts(After)

        outer_for = cast(ir.ForStmt, stmts[0])
        assert outer_for.loop_origin == ir.LoopOrigin.ChunkOuter
        assert outer_for.kind == ir.ForKind.Parallel

        # Outer body = SeqStmts [InCore, yield]
        outer_body_stmts = _body_stmts(outer_for.body)
        scope_stmt = cast(ir.ScopeStmt, outer_body_stmts[0])
        assert scope_stmt.scope_kind == ir.ScopeKind.InCore

        # InCore body = inner ForStmt
        inner_for = cast(ir.ForStmt, scope_stmt.body)
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
        stmts = _top_level_stmts(After)

        # i_out
        i_out = cast(ir.ForStmt, stmts[0])
        assert i_out.loop_origin == ir.LoopOrigin.ChunkOuter
        assert i_out.kind == ir.ForKind.Parallel

        # j_out inside i_out body
        i_out_body = _body_stmts(i_out.body)
        j_out = cast(ir.ForStmt, i_out_body[0])
        assert j_out.loop_origin == ir.LoopOrigin.ChunkOuter
        assert j_out.kind == ir.ForKind.Parallel

        # InCore inside j_out body
        j_out_body = _body_stmts(j_out.body)
        scope = cast(ir.ScopeStmt, j_out_body[0])
        assert scope.scope_kind == ir.ScopeKind.InCore

        # i_in inside InCore
        i_in = cast(ir.ForStmt, scope.body)
        assert i_in.loop_origin == ir.LoopOrigin.ChunkInner
        assert i_in.kind == ir.ForKind.Parallel

        # j_in inside i_in body
        i_in_body = _body_stmts(i_in.body)
        j_in = cast(ir.ForStmt, i_in_body[0])
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
        stmts = _top_level_stmts(After)
        i_out = cast(ir.ForStmt, stmts[0])

        # i_out should have iter_args (from x)
        assert len(i_out.iter_args) == 1
        assert len(i_out.return_vars) == 1

        # j_out should have iter_args chained from i_out
        i_out_body = _body_stmts(i_out.body)
        j_out = cast(ir.ForStmt, i_out_body[0])
        assert len(j_out.iter_args) == 1
        assert len(j_out.return_vars) == 1

        # InCore → i_in → j_in all with iter_args
        j_out_body = _body_stmts(j_out.body)
        scope = cast(ir.ScopeStmt, j_out_body[0])
        i_in = cast(ir.ForStmt, scope.body)
        assert len(i_in.iter_args) == 1
        assert len(i_in.return_vars) == 1

        i_in_body = _body_stmts(i_in.body)
        j_in = cast(ir.ForStmt, i_in_body[0])
        assert len(j_in.iter_args) == 1
        assert len(j_in.return_vars) == 1


class TestNestedChunkChainsInitSubstitution:
    """Tests that nested chunk chains correctly substitute init_values from parent chain."""

    def test_nested_chains_init_values_substituted(self):
        """Nested parallel chunk chains: inner chain init_values must reference parent's
        rewritten iter_args, not the original pre-interchange names."""

        @pl.program
        class Input:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for b in pl.parallel(0, 8, 1, chunk=4):
                        for h in pl.parallel(0, 12, 1, chunk=4):
                            x = pl.add(x, y)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)
        after_str = python_print(After)

        # SplitChunkedLoops introduces `ci` qualifiers for inner-loop
        # index vars. After InterchangeChunkLoops, init_values should reference
        # rewritten loop-threaded iter args, not the pre-interchange chunk-inner
        # iter names.
        lines = after_str.split("\n")
        for line in lines:
            if "init_values" in line:
                assert "__ci_iter_" not in line, f"Dangling chunk-inner iter in init_values: {line.strip()}"

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
                with pl.auto_incore():
                    for b in pl.parallel(0, 8, 1, chunk=4):
                        for h in pl.parallel(0, 12, 1, chunk=4):
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
                with pl.auto_incore():
                    for b in pl.parallel(0, 6, 1, chunk=4):
                        for h in pl.parallel(0, 14, 1, chunk=4):
                            x = pl.add(x, y)
                return x

        program = _prepare_for_interchange(Input)
        program = passes.interchange_chunk_loops()(program)
        program = passes.outline_incore_scopes()(program)

        incore_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.InCore]
        assert len(incore_funcs) >= 1


class TestChunkWithRemainderInChain:
    """Tests for chunk chains that include remainder loops (non-divisible inner)."""

    def test_chunk_outer_inner_with_remainder_preserves_iter_args(self):
        """Chunk chain with trailing remainder: iter_args thread through inner, remainder preserved."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        for j in pl.parallel(0, 1, 1, chunk=2):
                            x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)

        stmts = _top_level_stmts(After)

        # i_out (ChunkOuter, Parallel — preserves original kind from pl.parallel)
        i_out = cast(ir.ForStmt, stmts[0])
        assert i_out.loop_origin == ir.LoopOrigin.ChunkOuter
        assert i_out.kind == ir.ForKind.Parallel
        assert len(i_out.iter_args) == 1

        # InCore { i_in → body with remainder }
        i_out_body = _body_stmts(i_out.body)
        scope = cast(ir.ScopeStmt, i_out_body[0])
        assert scope.scope_kind == ir.ScopeKind.InCore

        i_in = cast(ir.ForStmt, scope.body)
        assert i_in.loop_origin == ir.LoopOrigin.ChunkInner
        assert len(i_in.iter_args) == 1

        # i_in's iter_arg should chain from i_out's iter_arg (not from original init)
        assert cast(ir.Var, i_in.iter_args[0].initValue).name_hint == i_out.iter_args[0].name_hint

    def test_chunk_with_remainder_body_contains_remainder_loop(self):
        """Remainder loop inside chain body must be preserved after interchange."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        for j in pl.parallel(0, 1, 1, chunk=2):
                            x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)
        after_str = python_print(After)

        # The remainder loop variable should still appear in the output
        # (it must not be dropped during interchange)
        assert "rem" in after_str or "j_" in after_str or "parallel(1" in after_str


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

        stmts = _top_level_stmts(After)

        # Main chunk pair: i_out → j_out → InCore { i_in → j_in → body }
        i_out = cast(ir.ForStmt, stmts[0])
        assert i_out.loop_origin == ir.LoopOrigin.ChunkOuter

        # Remainder: i_rem contains j_out→InCore{j_in} + InCore{j_rem}
        i_rem = cast(ir.ForStmt, stmts[1])
        assert i_rem.loop_origin == ir.LoopOrigin.ChunkRemainder

        # Inside i_rem body, look for InCore scopes
        i_rem_body = _body_stmts(i_rem.body)

        # j_out should have InCore wrapping j_in inside its body
        j_out_in_rem = cast(ir.ForStmt, i_rem_body[0])
        assert j_out_in_rem.loop_origin == ir.LoopOrigin.ChunkOuter
        j_out_body = _body_stmts(j_out_in_rem.body)
        assert cast(ir.ScopeStmt, j_out_body[0]).scope_kind == ir.ScopeKind.InCore

        # j_rem should be wrapped in InCore
        j_rem_incore = cast(ir.ScopeStmt, i_rem_body[1])
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
    """Tests for sequential chunked loops (should NOT interchange but get InCore wrapping)."""

    def test_sequential_chunk_gets_incore(self):
        """Sequential chunked loop inside auto_incore: gets InCore wrapping."""

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

        # AutoInCore is consumed, sequential chunks fail interchange guard
        # but get InCore wrapping from the non-chunk statement handler
        assert "auto_incore" not in after_str
        assert "incore" in after_str

    def test_nested_sequential_chunks_get_incore(self):
        """Nested sequential chunked loops: no interchange, but get InCore wrapping."""

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

        # AutoInCore consumed, sequential loops not interchanged but wrapped in InCore
        assert "auto_incore" not in after_str
        assert "incore" in after_str


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


class TestNonChunkStatementsWrapping:
    """Tests that non-chunk statements inside auto_incore get InCore wrapping."""

    def test_standalone_tensor_op_wrapped(self):
        """Standalone tensor op inside auto_incore gets wrapped in InCore."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)
        after_str = python_print(After)

        assert "auto_incore" not in after_str
        assert "incore" in after_str

    def test_standalone_op_before_parallel_chunk(self):
        """Standalone op before parallel chunk: op wrapped separately, chunk interchanged."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    x = pl.add(x, 1.0)
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        x = pl.add(x, 2.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)

        stmts = _top_level_stmts(After)

        # First stmt should be InCore wrapping the standalone op
        incore_scope = cast(ir.ScopeStmt, stmts[0])
        assert incore_scope.scope_kind == ir.ScopeKind.InCore

        # Second stmt should be ChunkOuter (interchanged)
        outer_for = cast(ir.ForStmt, stmts[1])
        assert outer_for.loop_origin == ir.LoopOrigin.ChunkOuter

    def test_standalone_op_after_parallel_chunk(self):
        """Standalone op after parallel chunk: chunk interchanged, op wrapped separately."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        x = pl.add(x, 2.0)
                    x = pl.mul(x, 3.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)

        stmts = _top_level_stmts(After)

        # First stmt should be ChunkOuter (interchanged)
        outer_for = cast(ir.ForStmt, stmts[0])
        assert outer_for.loop_origin == ir.LoopOrigin.ChunkOuter

        # There should be an InCore wrapping the standalone mul op
        after_str = python_print(After)
        assert "auto_incore" not in after_str
        # Count incore occurrences: one for the chunk's inner, one for the standalone op
        incore_count = after_str.count("pl.incore()")
        assert incore_count >= 2

    def test_host_side_assemble_after_parallel_chunk_not_wrapped(self):
        """Host-side tail assemble after a chunk should stay outside InCore."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[4], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                out_0: pl.Tensor[[8], pl.FP32] = pl.tensor.create(
                    [8], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                with pl.auto_incore():
                    for i in pl.parallel(0, 4, 1, chunk=2):
                        x = pl.tensor.adds(x, 1.0)
                    out_1: pl.Tensor[[8], pl.FP32] = pl.tensor.assemble(out_0, x, [0])
                return out_1

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)

        after_str = python_print(After)
        # Only the interchanged chunk body should be in InCore.
        assert after_str.count("pl.incore()") == 1
        assert "pl.tensor.assemble(" in after_str

    def test_multiple_parallel_chunks_no_regression(self):
        """Multiple parallel chunks with no standalone ops: all interchanged, no extra wrapping."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        x = pl.add(x, 1.0)
                    for j in pl.parallel(0, 12, 1, chunk=4):
                        x = pl.mul(x, 2.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)

        stmts = _top_level_stmts(After)

        # Both should be ChunkOuter loops (interchanged)
        i_out = cast(ir.ForStmt, stmts[0])
        assert i_out.loop_origin == ir.LoopOrigin.ChunkOuter
        j_out = cast(ir.ForStmt, stmts[1])
        assert j_out.loop_origin == ir.LoopOrigin.ChunkOuter

        # No extra InCore wrapping around the outers themselves
        assert "auto_incore" not in python_print(After)

    def test_non_chunked_loop_inside_auto_incore_wrapped(self):
        """Non-chunked loop with tensor ops inside auto_incore gets wrapped in InCore."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.range(10):
                        x = pl.add(x, 1.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)
        after_str = python_print(After)

        assert "auto_incore" not in after_str
        assert "incore" in after_str

    def test_mixed_parallel_and_sequential_chunks(self):
        """Mixed parallel chunk + sequential chunk: parallel interchanged, sequential wrapped."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for i in pl.parallel(0, 8, 1, chunk=4):
                        x = pl.add(x, 1.0)
                    for j in pl.range(0, 12, 1, chunk=4):
                        x = pl.mul(x, 2.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)

        stmts = _top_level_stmts(After)

        # First stmt: ChunkOuter from parallel chunk (interchanged)
        assert cast(ir.ForStmt, stmts[0]).loop_origin == ir.LoopOrigin.ChunkOuter

        # Sequential chunk should be wrapped in InCore
        after_str = python_print(After)
        assert "auto_incore" not in after_str
        # Both the interchanged chunk's inner and sequential chunk should have incore
        assert after_str.count("pl.incore()") >= 2


class TestScalarAssignmentNotWrapped:
    """Tests that pure scalar assignments stay outside InCore scopes."""

    def test_scalar_assign_adjacent_to_compute_not_wrapped(self):
        """Scalar assignment adjacent to tensor compute ops should stay in orchestration."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for ob in pl.range(0, 8):
                        offset: pl.Scalar[pl.INDEX] = ob * 4  # noqa: F841
                        x = pl.add(x, 1.0)
                        for i in pl.parallel(0, 8, 1, chunk=4):
                            x = pl.add(x, 2.0)
                return x

        Before = _prepare_for_interchange(Input)
        After = passes.interchange_chunk_loops()(Before)
        after_str = python_print(After)

        # The scalar assignment should NOT be inside any pl.incore() scope.
        scalar_assign_re = re.compile(r"offset\S*\s*:.*=.*\*\s*4")
        lines = after_str.split("\n")
        in_incore = False
        incore_depth = 0
        for line in lines:
            stripped = line.strip()
            if "pl.incore()" in stripped:
                in_incore = True
                incore_depth = len(line) - len(line.lstrip())
            elif in_incore and stripped and (len(line) - len(line.lstrip())) <= incore_depth:
                if not stripped.startswith("#"):
                    in_incore = False
            if in_incore:
                assert not scalar_assign_re.search(stripped), (
                    f"Pure scalar assignment found inside InCore scope: {stripped}"
                )

    def test_scalar_assign_not_wrapped_outline_no_crash(self):
        """Scalar assignment stays in orchestration after outline — no undefined variable."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.auto_incore():
                    for ob in pl.range(0, 8):
                        offset: pl.Scalar[pl.INDEX] = ob * 4  # noqa: F841
                        for i in pl.parallel(0, 8, 1, chunk=4):
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
                with pl.auto_incore():
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
                with pl.auto_incore():
                    x = pl.add(x, 1.0)
                    for i in pl.parallel(0, 8, 1, chunk=4):
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
                with pl.auto_incore():
                    for i in pl.range(0, 8, 1, chunk=4):
                        x = pl.add(x, 1.0)
                return x

        After = self._run_through_outline(Input)
        self._assert_no_compute_leaks(After, min_incore_funcs=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

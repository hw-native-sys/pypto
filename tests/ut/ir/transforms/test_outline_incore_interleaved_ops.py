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


def _prepare_for_interchange(program):
    """Run prerequisite passes to produce input for InterchangeChunkLoops."""
    program = passes.unroll_loops()(program)
    program = passes.convert_to_ssa()(program)
    program = passes.flatten_call_expr()(program)
    program = passes.split_chunked_loops()(program)
    return program


class TestNonParallelCodeBetweenChunks:
    """Non-parallel code between parallel chunk loops inside auto_incore
    must be wrapped in InCore scope so that OutlineIncoreScopes can outline it."""

    def test_interleaved_scalar_op_gets_incore(self):
        """A scalar op between two parallel chunks must get an InCore scope."""

        @pl.program
        class Input:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.auto_incore():
                    for b in pl.range(0, 8, 4):
                        # Parallel chunk 1 → gets InCore after interchange
                        for i in pl.parallel(4, chunk=2):
                            x = pl.tensor.adds(x, 1.0)
                        # Non-parallel op → should ALSO get InCore
                        y: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x, 2.0)
                        # Parallel chunk 2 → gets InCore after interchange
                        for j in pl.parallel(4, chunk=2):
                            x = pl.tensor.add(x, y)
                return x

        program = _prepare_for_interchange(Input)
        program = passes.interchange_chunk_loops()(program)
        program = passes.outline_incore_scopes()(program)

        # The muls op should have been outlined and not be in the Orchestration function.
        orch_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.Orchestration]
        assert len(orch_funcs) == 1
        orch_str = orch_funcs[0].as_python()

        assert "tensor.muls" not in orch_str, (
            "pl.tensor.muls appears in Orchestration function — "
            "InterchangeChunkLoops did not wrap non-parallel code"
        )

    def test_interleaved_range_loop_gets_incore(self):
        """A range loop between parallel chunks must get an InCore scope.

        This mirrors the Qwen3 MLP pattern: a pl.range() loop containing
        matmul sits between two pl.parallel() chunk loops.
        """

        @pl.program
        class Input:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
                w: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.auto_incore():
                    for b in pl.range(0, 8, 4):
                        # Parallel chunk → gets InCore
                        for i in pl.parallel(4, chunk=2):
                            x = pl.tensor.adds(x, 1.0)
                        # Non-parallel range loop with matmul → should get InCore
                        for k in pl.range(2):
                            x = pl.tensor.matmul(x, w)
                        # Parallel chunk → gets InCore
                        for j in pl.parallel(4, chunk=2):
                            x = pl.tensor.adds(x, 1.0)
                return x

        program = _prepare_for_interchange(Input)
        program = passes.interchange_chunk_loops()(program)
        program = passes.outline_incore_scopes()(program)

        orch_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.Orchestration]
        assert len(orch_funcs) == 1

        # The orchestration function should NOT contain tensor.matmul
        # (it should have been outlined into an InCore function)
        orch_str = orch_funcs[0].as_python()
        assert "tensor.matmul" not in orch_str, (
            "tensor.matmul remains in Orchestration function — "
            "non-parallel range loop was not wrapped in InCore"
        )

    def test_all_ops_outlined_end_to_end(self):
        """End-to-end: all compute ops inside auto_incore must be outlined.

        After InterchangeChunkLoops + OutlineIncoreScopes, the Orchestration
        function should contain only call sites, loop scaffolding, and yields
        — no tensor compute ops.
        """

        @pl.program
        class Input:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.auto_incore():
                    for b in pl.range(0, 8, 4):
                        for i in pl.parallel(4, chunk=2):
                            x = pl.tensor.adds(x, 1.0)
                        # Straight-line op between chunks
                        y: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x, 2.0)
                        for j in pl.parallel(4, chunk=2):
                            x = pl.tensor.add(x, y)
                return x

        program = _prepare_for_interchange(Input)
        program = passes.interchange_chunk_loops()(program)
        program = passes.outline_incore_scopes()(program)

        orch_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.Orchestration]
        assert len(orch_funcs) == 1
        orch_str = orch_funcs[0].as_python()

        # No tensor compute ops should remain in Orchestration
        forbidden_ops = ["tensor.muls", "tensor.add", "tensor.mul", "tensor.matmul"]
        for op in forbidden_ops:
            assert op not in orch_str, f"{op} remains in Orchestration — non-parallel code was not outlined"

        # At least 3 InCore functions should exist (chunk1, interleaved, chunk2)
        incore_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.InCore]
        assert len(incore_funcs) >= 3, (
            f"Expected >= 3 InCore functions, got {len(incore_funcs)} — "
            "non-parallel code was not outlined into its own InCore function"
        )


class TestNestedForStmtRecursion:
    """The fix recurses into ForStmt bodies that contain InCore scopes.
    These tests verify the recursion works for deeper nesting and edge cases."""

    def test_doubly_nested_range_with_interleaved_op(self):
        """Non-parallel op inside a doubly nested range loop must get InCore scope.

        Structure: auto_incore > range > range > [parallel, scalar_op, parallel]
        The fix must recurse through both range loops.
        """

        @pl.program
        class Input:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.auto_incore():
                    for b in pl.range(0, 8, 4):
                        for c in pl.range(2):
                            for i in pl.parallel(4, chunk=2):
                                x = pl.tensor.adds(x, 1.0)
                            # Non-parallel op deep inside nested range
                            y: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x, 3.0)
                            for j in pl.parallel(4, chunk=2):
                                x = pl.tensor.add(x, y)
                return x

        program = _prepare_for_interchange(Input)
        program = passes.interchange_chunk_loops()(program)
        program = passes.outline_incore_scopes()(program)

        orch_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.Orchestration]
        assert len(orch_funcs) == 1
        orch_str = orch_funcs[0].as_python()

        assert "tensor.muls" not in orch_str, (
            "tensor.muls remains in Orchestration — recursive descent did not reach doubly nested range loop"
        )

    def test_single_forstmt_body_with_mixed_children(self):
        """auto_incore body is a single ForStmt (not SeqStmts).

        This is the exact trigger for the original bug: ContainsInCoreScope
        returns True for the ForStmt, so the old code returned it as-is
        without examining its children.
        """

        @pl.program
        class Input:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.auto_incore():
                    for b in pl.range(0, 8, 4):
                        for i in pl.parallel(4, chunk=2):
                            x = pl.tensor.adds(x, 1.0)
                        x = pl.tensor.muls(x, 2.0)
                return x

        program = _prepare_for_interchange(Input)
        program = passes.interchange_chunk_loops()(program)
        program = passes.outline_incore_scopes()(program)

        # The muls op should have been outlined and not be in the Orchestration function.
        orch_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.Orchestration]
        assert len(orch_funcs) == 1
        orch_str = orch_funcs[0].as_python()

        assert "tensor.muls" not in orch_str, (
            "pl.tensor.muls appears in Orchestration function — "
            "single ForStmt body case not handled correctly"
        )

    def test_multiple_non_parallel_ops_between_chunks(self):
        """Multiple consecutive non-parallel ops between chunks must all be wrapped."""

        @pl.program
        class Input:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.auto_incore():
                    for b in pl.range(0, 8, 4):
                        for i in pl.parallel(4, chunk=2):
                            x = pl.tensor.adds(x, 1.0)
                        # Multiple non-parallel ops in sequence
                        y: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(x, 2.0)
                        z: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.add(x, y)
                        x = pl.tensor.muls(z, 0.5)
                        for j in pl.parallel(4, chunk=2):
                            x = pl.tensor.adds(x, 1.0)
                return x

        program = _prepare_for_interchange(Input)
        program = passes.interchange_chunk_loops()(program)
        program = passes.outline_incore_scopes()(program)

        orch_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.Orchestration]
        assert len(orch_funcs) == 1
        orch_str = orch_funcs[0].as_python()

        # None of the non-parallel ops should remain in Orchestration
        for op in ["tensor.muls", "tensor.add"]:
            assert op not in orch_str, (
                f"{op} remains in Orchestration — consecutive non-parallel ops were not all wrapped in InCore"
            )

    def test_no_parallel_chunks_no_wrapping(self):
        """auto_incore with only non-parallel code (no chunks) should not crash.

        When there are no interchanged parallel chunks, there are no InCore
        scopes to trigger recursion. The function should still work correctly.
        """

        @pl.program
        class Input:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                with pl.auto_incore():
                    for b in pl.range(0, 8, 4):
                        x = pl.tensor.adds(x, 1.0)
                        x = pl.tensor.muls(x, 2.0)
                return x

        program = _prepare_for_interchange(Input)
        # Should not crash
        program = passes.interchange_chunk_loops()(program)
        program = passes.outline_incore_scopes()(program)

        # All ops should still be outlined (auto_incore wraps everything)
        orch_funcs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.Orchestration]
        assert len(orch_funcs) == 1
        orch_str = orch_funcs[0].as_python()
        assert "tensor.adds" not in orch_str
        assert "tensor.muls" not in orch_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

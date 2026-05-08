# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for FoldNoOpReshape pass.

The pass rewrites ``lhs = tile.reshape(rhs, shape)`` AssignStmts into plain
``lhs = rhs`` whenever both sides share the same MemRef root and produce
identical TileBufSignatures. PTO codegen previously dropped emission of such
reshapes via a peephole; folding into the IR makes codegen 1:1.
"""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure Ascend910B backend; LegalizePTOBufferReuse needs one."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _count_reshape_calls(program: ir.Program) -> int:
    count = 0

    class _ReshapeCounter(ir.IRVisitor):
        def visit_call(self, op):
            nonlocal count
            if op.op.name == "tile.reshape":
                count += 1

    counter = _ReshapeCounter()
    for _, fn in program.functions.items():
        if fn.body is not None:
            counter.visit_stmt(fn.body)
    return count


def _run_to_legalize_then_fold(program: ir.Program) -> ir.Program:
    """Run the pre-required passes, then FoldNoOpReshape."""
    pipeline = passes.PassPipeline()
    for p in (
        passes.convert_to_ssa(),
        passes.outline_incore_scopes(),
        passes.flatten_tile_nd_to_2d(),
        passes.infer_tile_memory_space(),
        passes.init_mem_ref(),
        passes.memory_reuse(),
        passes.legalize_pto_buffer_reuse(),
        passes.allocate_memory_addr(),
        passes.fold_no_op_reshape(),
    ):
        pipeline.add_pass(p)
    ctx = passes.PassContext([], passes.VerificationLevel.NONE)
    with ctx:
        return pipeline.run(program)


class TestFoldNoOpReshape:
    def test_noop_reshape_is_folded(self):
        """A reshape that preserves shape and shares MemRef must be folded out."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                # Identical shape + same MemRef after LegalizePTOBufferReuse → no-op reshape.
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(tile_a, [64, 64])
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output)
                return result

        # The same-shape reshape exists in the input...
        assert _count_reshape_calls(Before) == 1
        After = _run_to_legalize_then_fold(Before)
        # ...and FoldNoOpReshape rewrites it into a Var-to-Var assignment, dropping the Call.
        assert _count_reshape_calls(After) == 0

    def test_genuine_reshape_kept(self):
        """A reshape that changes physical shape must NOT be folded."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                reshaped: pl.Tile[[4096, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(tile_a, [4096, 1])
                flat: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(reshaped, [64, 64])
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(flat, [0, 0], output)
                return result

        After = _run_to_legalize_then_fold(Before)
        # Physical-shape changing reshapes ([64,64] <-> [4096,1]) must remain.
        assert _count_reshape_calls(After) >= 2

    def test_pass_runs_without_error_on_simple_kernel(self):
        """Smoke test: pass should not crash on a kernel without trivial reshapes."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(x, [0, 0], [64, 64])
                y: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(t, t)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(y, [0, 0], output)
                return result

        After = _run_to_legalize_then_fold(Before)
        # No tile.reshape in the input — none in the output.
        assert _count_reshape_calls(After) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

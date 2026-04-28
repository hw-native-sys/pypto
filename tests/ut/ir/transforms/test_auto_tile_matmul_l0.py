# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Before / After / Expected tests for the AutoTileMatmulL0 pass.

The pass walks Mat-resident ``tile.matmul`` calls, queries
``utils::ChooseL0Tile`` against the active backend's L0 capacities, and rewrites
each call into a K-loop that branches on the loop index: the first iteration
uses ``tile.matmul`` (fresh accumulator) and subsequent iterations use
``tile.matmul_acc`` (accumulating into the iter-arg).  The loop is marked
``ForKind.Pipeline`` with ``pipeline_stages=2`` whenever it has at least two
iterations.

The conftest configures the Ascend950 backend, which advertises L0a/L0b = 64KB
and L0c = 256KB.  Tests rely on those capacities to predict the chooser's
output.

Each test is structured as Before / After / Expected:

* ``Before``  — the input program (a Mat-resident matmul).
* ``After``   — the program produced by running the pass.
* ``Expected`` — the program written out as the pass should produce it.

The comparison uses ``ir.assert_structural_equal`` with auto-mapping, so
intermediate Var names may differ between After and Expected — only types and
structural positions need to match.

The pass emits IR that is intentionally not yet type-correct: the iter-arg
init is a ``tile.create`` with ``Vec`` target while the yields are
``Acc``-typed matmul results.  ``InferTileMemorySpace`` (the next pass in
the pipeline) inserts the Acc↔Vec bridges that make the IR type-correct.
For unit testing this pass in isolation, ``_run_pass`` disables the
auto-fixture's BeforeAndAfter type-check verification, mirroring the pattern
used by ``test_lower_pipeline_loops``.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _run_pass(program: ir.Program) -> ir.Program:
    """Run AutoTileMatmulL0 with verification suppressed.

    The pass emits intermediate IR with iter-arg/yield TileView mismatches
    that ``InferTileMemorySpace`` (next pass) will resolve.  Wrapping in a
    ``VerificationLevel.NONE`` PassContext bypasses the autouse fixture's
    BeforeAndAfter check and the print/parse roundtrip check, which together
    expect type-correct IR after every pass.
    """
    with passes.PassContext([], passes.VerificationLevel.NONE):
        return passes.auto_tile_matmul_l0()(program)


class TestAutoTileMatmulL0KOnly:
    """K-tiling rewrites for Mat-resident tile.matmul."""

    def test_skinny_gemm_pipelined(self):
        """16×64 @ 2048 BF16 → ChooseL0Tile picks (m=16, n=64, k=256).

        K=2048 → 8 K-iterations → loop runs 8 times with an if-else branching
        on ``ko == 0`` between ``tile.matmul`` (first iter) and
        ``tile.matmul_acc`` (later iters).  Loop is Pipeline-marked with
        ``pipeline_stages=2``."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                # Vec-resident placeholder for the iter-arg init.
                c_init: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create(
                    [16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec
                )
                # Full K-loop with ko branching on the first iteration.
                for ko, (c_iter,) in pl.pipeline(0, 2048, 256, init_values=(c_init,), stage=2):
                    sa: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.slice(lhs_mat, [16, 256], [0, ko])
                    sb: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.slice(rhs_mat, [256, 64], [ko, 0])
                    if ko == 0:
                        c_first: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(sa, sb)
                        c_phi: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(c_first)
                    else:
                        c_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(c_iter, sa, sb)
                        c_phi: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(c_acc)
                    c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(c_phi)
                out = pl.store(c, [0, 0], out)
                return out

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_already_l0_sized_skipped(self):
        """64×64×64 BF16 → fits in L0 capacity after double-buffering →
        ChooseL0Tile returns (M, N, K) → pass leaves the matmul untouched."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[64, 64], pl.BF16],
                rhs: pl.Tensor[[64, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                lhs_mat: pl.Tile[[64, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [64, 64], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[64, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [64, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        # No tiling needed → expected = before.
        After = _run_pass(Before)
        ir.assert_structural_equal(After, Before)

    def test_pass_idempotent(self):
        """Running the pass twice produces the same result as running it once.

        After the first rewrite, the only ``tile.matmul`` left is the peeled
        first iteration — its operands are slices of [16, 256] / [256, 64],
        which are themselves L0-sized.  A second run sees ChooseL0Tile pick
        ``(16, 64, 256)`` for the slice problem and returns
        ``(m=16, n=64, k=256) == (M, N, K)``, so the pass leaves the peeled
        matmul (and the loop body's matmul_acc, which is out of v1 scope)
        untouched."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        once = _run_pass(Before)
        twice = _run_pass(once)
        ir.assert_structural_equal(twice, once)


class TestAutoTileMatmulL0Skips:
    """Cases where the pass intentionally leaves the matmul untouched."""

    def test_matmul_acc_left_untouched(self):
        """``tile.matmul_acc`` is out of scope for v1; the pass should leave
        it identical to the input (no peeled split, no K-loop)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                acc_init: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(acc_init, lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Before)

    def test_non_mat_operands_left_untouched(self):
        """Operands not in ``MemorySpace.Mat`` (e.g. default ``Vec``) are out
        of scope; the pass shouldn't try to tile them.  Verified by checking
        After is structurally identical to Before."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                # Default tile.load lands in Vec, not Mat.
                lhs_vec: pl.Tile[[16, 2048], pl.BF16] = pl.tile.load(lhs, [0, 0], [16, 2048])
                rhs_vec: pl.Tile[[2048, 64], pl.BF16] = pl.tile.load(rhs, [0, 0], [2048, 64])
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_vec, rhs_vec)
                out = pl.store(c, [0, 0], out)
                return out

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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

The pass emits an Acc-typed iter-arg init via ``tile.create(target=Acc)``
and per-iter ``tile.extract(..., target_memory=Left|Right)`` for the Mat
operand slices, so the produced IR is L0-typed end-to-end and roundtrips
cleanly through the autouse print/parse fixture.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _run_pass(program: ir.Program, *, suppress_verification: bool = False) -> ir.Program:
    """Run AutoTileMatmulL0.

    Most tests verify their result, but the ``matmul_acc`` case threads a
    user-supplied ``acc_init`` parameter into the new K-loop's iter-arg.
    The DSL parser doesn't attach an Nz tile_view to function-parameter
    Acc tiles, but ``tile.matmul_acc``'s deducer always emits one, so the
    iter_arg-init / yield-value tile_view-presence check inside the new
    ForStmt fails.  This is the same parser↔deducer asymmetry tracked in
    ``KNOWN_ISSUES.md`` ("Structural equality incompatible with Acc-typed
    iter_args …"), and it doesn't reproduce when ``acc_init`` originates
    from ``tile.create``/``tile.matmul`` upstream (the common production
    case).  Pass ``suppress_verification=True`` to bypass the autouse
    BeforeAndAfter / print-parse roundtrip for that one test.
    """
    if suppress_verification:
        with passes.PassContext([], passes.VerificationLevel.NONE):
            return passes.auto_tile_matmul_l0()(program)
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
                # Acc-resident placeholder for the iter-arg init.
                c_init: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                # Full K-loop with ko branching on the first iteration.
                for ko, (c_iter,) in pl.pipeline(0, 2048, 256, init_values=(c_init,), stage=2):
                    sa: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 0, ko, shape=[16, 256], target_memory=pl.Mem.Left
                    )
                    sb: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko, 0, shape=[256, 64], target_memory=pl.Mem.Right
                    )
                    if ko == 0:
                        c_first: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(sa, sb)
                        c_phi: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(c_first)
                    else:
                        c_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(c_iter, sa, sb)
                        c_phi: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(c_acc)
                    c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(c_phi)
                out = pl.store(c, [0, 0], out)
                return out

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_matmul_acc_pipelined(self):
        """``tile.matmul_acc`` with the same 16×64 @ 2048 BF16 shape rewrites
        into a uniform K-loop: every iteration is ``tile.matmul_acc``, with
        the iter-arg init = caller's ``acc_init`` (no Vec placeholder, no
        if-else branch since the accumulator chain is uniform from the
        first iteration)."""

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

        @pl.program
        class Expected:
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
                # No Vec placeholder: the iter-arg init is the caller's acc_init.
                for ko, (c_iter,) in pl.pipeline(0, 2048, 256, init_values=(acc_init,), stage=2):
                    sa: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 0, ko, shape=[16, 256], target_memory=pl.Mem.Left
                    )
                    sb: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko, 0, shape=[256, 64], target_memory=pl.Mem.Right
                    )
                    c_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(c_iter, sa, sb)
                    c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(c_acc)
                out = pl.store(c, [0, 0], out)
                return out

        After = _run_pass(Before, suppress_verification=True)
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

        After the first rewrite, the only ``tile.matmul`` is inside the
        K-loop's then-branch over slices of shape [16, 256] / [256, 64] which
        are already L0-sized, so the second run sees a no-op.  We also assert
        the first run *did* change the IR so a regression where the pass
        becomes a no-op overall still fails the test."""

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
        # First run must have rewritten — otherwise the idempotency check is
        # vacuously true.
        with pytest.raises(ValueError, match="Structural equality"):
            ir.assert_structural_equal(once, Before)
        twice = _run_pass(once)
        ir.assert_structural_equal(twice, once)

    def test_k_not_divisible_skipped(self):
        """When the chooser picks a ``k`` that doesn't divide ``K``, the pass
        emits a ``PerfHint`` (PH-AT-007) and leaves the matmul untouched —
        K-boundary handling (``valid_shape`` on the last slice) is not yet
        implemented."""

        # M=16, N=64, K=2050 (not divisible by 256 — chooser picks k=256 for
        # 16/64 BF16 → emits PH-AT-007).
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2050], pl.BF16],
                rhs: pl.Tensor[[2050, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 2050], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 2050], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[2050, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2050, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Before)


class TestAutoTileMatmulL0Skips:
    """Cases where the pass intentionally leaves the matmul untouched."""

    def test_non_mat_operands_left_untouched_for_matmul_acc(self):
        """``tile.matmul_acc`` whose lhs/rhs aren't Mat-resident is out of
        scope for tiling; the pass should leave it identical."""

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
                # Default tile.load lands in Vec, not Mat — pass should skip.
                lhs_vec: pl.Tile[[16, 2048], pl.BF16] = pl.tile.load(lhs, [0, 0], [16, 2048])
                rhs_vec: pl.Tile[[2048, 64], pl.BF16] = pl.tile.load(rhs, [0, 0], [2048, 64])
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(acc_init, lhs_vec, rhs_vec)
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

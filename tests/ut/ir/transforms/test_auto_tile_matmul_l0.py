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
each call into a K-loop whose body is a single predicated
``tile.matmul_acc(c_iter, sa, sb, ko == 0)``: the predicate overwrites the
accumulator on the first iteration and accumulates into it afterwards, so the
whole chain stays on one Acc buffer.  The loop is marked ``ForKind.Pipeline``
with ``pipeline_stages=2`` whenever it has at least two iterations.

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

import re

import pypto.language as pl
import pytest
from pypto import backend as _backend
from pypto import ir, passes
from pypto.backend import BackendType


def _default_allocation_snapshot(program: ir.Program) -> ir.Program:
    """Inspect planner-owned Tile allocations while still running the full pipeline."""
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

    snapshot = None
    for transform in PassManager(OptimizationStrategy.Default).passes:
        if transform.get_name() == "LowerTileToBuffer":
            snapshot = program
        program = transform(program)
    assert snapshot is not None
    return snapshot


class TestAutoTileMatmulL0ExplicitL0Diagnostics:
    """Actionable failures for manual L0 operands that cannot fit."""

    def test_oversized_right_operand_names_tile_and_fix(self):
        """The hpgemm-step0 shape is one impossible 128 KiB L0B tile.

        AutoTile must fail at the explicit ``b_right`` definition instead of
        silently skipping the already-L0 matmul and letting MemoryReuse blame
        packing before AllocateMemoryAddr reports only an aggregate overflow.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 256], pl.FP16],
                b: pl.Tensor[[256, 256], pl.FP16],
                out: pl.Out[pl.Tensor[[128, 256], pl.FP32]],
            ) -> pl.Tensor[[128, 256], pl.FP32]:
                a_mat = pl.tile.load(a, [0, 0], [128, 256], target_memory=pl.Mem.Mat)
                b_mat = pl.tile.load(b, [0, 0], [256, 256], target_memory=pl.Mem.Mat)
                a_left = pl.tile.extract(a_mat, 0, 0, [128, 256], target_memory=pl.Mem.Left)
                b_right = pl.tile.extract(b_mat, 0, 0, [256, 256], target_memory=pl.Mem.Right)
                acc = pl.tile.matmul(a_left, b_right)
                out = pl.tile.store(acc, [0, 0], out)
                return out

        before_ssa = passes.convert_to_ssa()(Before)
        with pytest.raises(ValueError) as exc_info:
            passes.auto_tile_matmul_l0()(before_ssa)

        message = str(exc_info.value)
        assert "tile.matmul right operand 'b_right'" in message
        assert "Right (L0B)" in message
        assert "physical shape [256, 256]" in message
        assert "requiring 131072 bytes" in message
        assert "provides 65536 bytes" in message
        assert "does not retile operands already placed in Left or Right" in message
        assert "Keep this operand in Mat" in message
        assert "manually extract a smaller Right tile" in message
        assert "b_right__ssa" not in message
        assert "Check failed" not in message

    def test_oversized_left_matmul_acc_operand_is_also_diagnosed(self):
        """The same check covers matmul_acc's shifted lhs/rhs argument slots."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[256, 256], pl.FP16],
                b: pl.Tensor[[256, 64], pl.FP16],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                a_mat = pl.tile.load(a, [0, 0], [256, 256], target_memory=pl.Mem.Mat)
                b_mat = pl.tile.load(b, [0, 0], [256, 64], target_memory=pl.Mem.Mat)
                a_left = pl.tile.extract(a_mat, 0, 0, [256, 256], target_memory=pl.Mem.Left)
                b_right = pl.tile.extract(b_mat, 0, 0, [256, 64], target_memory=pl.Mem.Right)
                acc = pl.tile.create([256, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                result = pl.tile.matmul_acc(acc, a_left, b_right)
                out = pl.tile.store(result, [0, 0], out)
                return out

        with pytest.raises(ValueError) as exc_info:
            passes.auto_tile_matmul_l0()(Before)

        message = str(exc_info.value)
        assert "tile.matmul_acc left operand 'a_left'" in message
        assert "Left (L0A)" in message
        assert "physical shape [256, 256]" in message
        assert "requiring 131072 bytes" in message
        assert "manually extract a smaller Left tile" in message

    def test_oversized_left_operand_on_predicated_matmul_acc_is_diagnosed(self):
        """The 4-operand ``tile.matmul_acc`` reaches the same capacity check.

        Accepting arity 4 in ``AnalyzeMatmul`` moved this call past the arity
        bail that used to short-circuit it, so a predicated call now gets the
        actionable Left/L0A diagnostic instead of silently surviving AutoTile
        and failing later in memory planning with a worse message.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[256, 256], pl.FP16],
                b: pl.Tensor[[256, 64], pl.FP16],
                first_k: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                a_mat = pl.tile.load(a, [0, 0], [256, 256], target_memory=pl.Mem.Mat)
                b_mat = pl.tile.load(b, [0, 0], [256, 64], target_memory=pl.Mem.Mat)
                a_left = pl.tile.extract(a_mat, 0, 0, [256, 256], target_memory=pl.Mem.Left)
                b_right = pl.tile.extract(b_mat, 0, 0, [256, 64], target_memory=pl.Mem.Right)
                acc = pl.tile.create([256, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                result = pl.tile.matmul_acc(acc, a_left, b_right, init_cond=(first_k == 0))
                out = pl.tile.store(result, [0, 0], out)
                return out

        with pytest.raises(ValueError) as exc_info:
            passes.auto_tile_matmul_l0()(Before)

        message = str(exc_info.value)
        assert "tile.matmul_acc left operand 'a_left'" in message
        assert "Left (L0A)" in message
        assert "physical shape [256, 256]" in message
        assert "requiring 131072 bytes" in message
        assert "manually extract a smaller Left tile" in message

    def test_exact_capacity_manual_operands_remain_untouched(self):
        """Manual L0 scheduling remains valid at the inclusive capacity boundary."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 256], pl.FP16],
                b: pl.Tensor[[256, 128], pl.FP16],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                a_mat = pl.tile.load(a, [0, 0], [128, 256], target_memory=pl.Mem.Mat)
                b_mat = pl.tile.load(b, [0, 0], [256, 128], target_memory=pl.Mem.Mat)
                a_left = pl.tile.extract(a_mat, 0, 0, [128, 256], target_memory=pl.Mem.Left)
                b_right = pl.tile.extract(b_mat, 0, 0, [256, 128], target_memory=pl.Mem.Right)
                acc = pl.tile.matmul(a_left, b_right)
                out = pl.tile.store(acc, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)


class TestAutoTileMatmulL0KOnly:
    """K-tiling rewrites for Mat-resident tile.matmul."""

    def test_skinny_gemm_pipelined(self):
        """16×64 @ 2048 BF16 → ChooseL0Tile picks (m=16, n=64, k=256).

        K=2048 → 8 K-iterations → loop runs 8 times, each iteration a
        predicated ``tile.matmul_acc`` that overwrites the accumulator on
        ``ko == 0`` and accumulates into it afterwards.  Loop is
        Pipeline-marked with ``pipeline_stages=2``."""

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
                # Full K-loop; the ko == 0 predicate overwrites the accumulator
                # on the first iteration and accumulates into it afterwards.
                for ko, (c_iter,) in pl.pipeline(0, 2048, 256, init_values=(c_init,), stage=2):
                    sa: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 0, ko, shape=[16, 256], target_memory=pl.Mem.Left
                    )
                    sb: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko, 0, shape=[256, 64], target_memory=pl.Mem.Right
                    )
                    c_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                        c_iter, sa, sb, ko == 0
                    )
                    c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(c_acc)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_plain_matmul_k_loop_keeps_one_acc_buffer(self):
        """The generated fresh K-loop reaches memory planning on ONE L0C buffer.

        This is the property the predicated body buys.  ``tile.matmul_acc``
        declares ``set_output_reuses_input(0)``, so the ``tile.create`` seed,
        the per-iteration result, the yield and the loop's ``return_var`` are
        one Acc allocation by construction — no coalescing repair, and no
        Acc->Acc ``tile.move`` (which no supported target can even emit).

        Checked on both planners: PYPTO runs MemoryReuse, DSA_RP skips it and
        leaves MaterializeSemanticAliases as the only aliasing step.
        """

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

        def assert_single_acc_buffer(after: ir.Program, planner: str) -> None:
            printed = ir.python_print(after)
            acc_bases = {
                line.strip().split(":")[0] for line in printed.splitlines() if "tile.alloc(pl.Mem.Acc" in line
            }
            assert len(acc_bases) == 1, (
                f"{planner}: expected ONE Acc allocation for the predicated K-loop, "
                f"got {len(acc_bases)}: {sorted(acc_bases)}\n{printed}"
            )
            assert "tile.move" not in printed, (
                f"{planner}: an in-place accumulator chain needs no tile.move, and an "
                f"Acc->Acc copy is not realizable:\n{printed}"
            )

        tiled = passes.auto_tile_matmul_l0()(Before)

        with passes.PassContext([], passes.VerificationLevel.BASIC):
            pypto_after = passes.memory_reuse()(
                passes.materialize_semantic_aliases()(passes.init_mem_ref()(tiled))
            )
        assert_single_acc_buffer(pypto_after, "PYPTO")

        with passes.PassContext(
            [],
            passes.VerificationLevel.BASIC,
            memory_planner=passes.MemoryPlanner.DSA_RP,
        ):
            dsa_after = passes.materialize_semantic_aliases()(passes.init_mem_ref()(tiled))
        assert_single_acc_buffer(dsa_after, "DSA_RP")

    def test_matmul_acc_pipelined(self):
        """``tile.matmul_acc`` with the same 16×64 @ 2048 BF16 shape rewrites
        into a uniform K-loop: every iteration is ``tile.matmul_acc``, with
        the iter-arg init = caller's ``acc_init`` (no Vec placeholder and no
        ``init_cond`` predicate, since the caller's accumulator is already
        live on the first iteration and must never be overwritten)."""

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

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_vec_fed_lhs_staged_to_mat_and_tiled(self):
        """Fused-attention PV / ``score·V`` pattern: the left operand is
        Vec-resident (softmax/``exp`` output crossing the cube↔vector boundary)
        while the right operand is Mat.

        The pass stages the Vec left operand into Mat via ``tile.move`` *before*
        the K-loop — so ``ExpandMixedKernel`` can lower the Vec→Mat boundary
        crossing through its ``tile.move``-based ``tpop_from_aiv`` handshake —
        then tiles symmetrically with the QK (Mat-fed) path, extracting Left
        sub-tiles from the staged Mat tile.  16×64 @ 2048 BF16 → ChooseL0Tile
        picks (m=16, n=64, k=256)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                # Default tile.load lands in Vec — the PV / score·V operand.
                lhs_vec: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Vec] = pl.tile.load(lhs, [0, 0], [16, 2048])
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_vec, rhs_mat)
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
                lhs_vec: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Vec] = pl.tile.load(lhs, [0, 0], [16, 2048])
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                # Acc-resident placeholder for the iter-arg init.
                c_init: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                # Vec lhs staged into Mat once, before the K-loop.
                lhs_mat: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.move(
                    lhs_vec, target_memory=pl.Mem.Mat
                )
                for ko, (c_iter,) in pl.pipeline(0, 2048, 256, init_values=(c_init,), stage=2):
                    # lhs sub-tile extracted from the *staged Mat* tile, not from Vec.
                    sa: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 0, ko, shape=[16, 256], target_memory=pl.Mem.Left
                    )
                    sb: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko, 0, shape=[256, 64], target_memory=pl.Mem.Right
                    )
                    c_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                        c_iter, sa, sb, ko == 0
                    )
                    c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(c_acc)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_matmul_acc_vec_lhs_staged_and_tiled(self):
        """``tile.matmul_acc`` whose left (A) operand is Vec-resident
        (fused-attention PV / ``score·V`` with a running caller accumulator).

        ``AnalyzeMatmul`` in ``auto_tile_matmul_l0/analysis.cpp`` ensures the
        Vec left operand sets ``stage_lhs_to_mat=true`` so a single
        ``tile.move(lhs_vec, target=Mat)`` is emitted before the K-loop and the
        per-iter Left extract slices from the staged Mat tile; ``acc_init`` is
        the caller's accumulator threaded into the iter-arg directly.  Because
        ``is_acc`` is true the body is the *uniform* ``matmul_acc`` shape with
        **no** ``init_cond`` predicate and **no** ``tile.create`` placeholder (``BuildKLoopRewrite``
        in ``auto_tile_matmul_l0_pass.cpp``).  16×64 @ 2048 BF16 with
        ``c_read=true`` picks (m=16, n=64, k=256) — the same tile the Mat-lhs
        ``test_matmul_acc_pipelined`` case pins."""

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
                # Default tile.load lands in Vec — the PV / score·V operand.
                lhs_vec: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Vec] = pl.tile.load(lhs, [0, 0], [16, 2048])
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(acc_init, lhs_vec, rhs_mat)
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
                lhs_vec: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Vec] = pl.tile.load(lhs, [0, 0], [16, 2048])
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                # No tile.create placeholder: the iter-arg init is the caller's
                # acc_init.  Vec lhs staged into Mat once, before the K-loop.
                lhs_mat: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.move(
                    lhs_vec, target_memory=pl.Mem.Mat
                )
                for ko, (c_iter,) in pl.pipeline(0, 2048, 256, init_values=(acc_init,), stage=2):
                    # lhs sub-tile extracted from the staged Mat tile, not Vec.
                    sa: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 0, ko, shape=[16, 256], target_memory=pl.Mem.Left
                    )
                    sb: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko, 0, shape=[256, 64], target_memory=pl.Mem.Right
                    )
                    # Uniform matmul_acc body — no init_cond predicate.
                    c_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(c_iter, sa, sb)
                    c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(c_acc)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_two_independent_matmuls_each_remapped(self):
        """Two independent Mat-resident ``tile.matmul`` calls in one function
        body are each rewritten into their own K-loop, and each downstream
        ``pl.store`` is redirected to the matching ForStmt's ``return_var``.

        This exercises the per-SeqStmts ``remap`` in
        ``AutoTileMutator::VisitStmt_(SeqStmtsPtr)`` (pass lines 561-585): the
        first rewrite records ``c0 -> for0.return_var`` and the second records
        ``c1 -> for1.return_var``; the running ``Substitute`` then rewrites the
        two ``pl.store`` uses to the new return_vars.  Each matmul is 16×64 @
        2048 BF16 (plain ``tile.matmul``, ``c_read=false``) → (m=16, n=64,
        k=256), so each loop is the standard predicated K-loop of
        ``test_skinny_gemm_pipelined``."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs0: pl.Tensor[[16, 2048], pl.BF16],
                rhs0: pl.Tensor[[2048, 64], pl.BF16],
                lhs1: pl.Tensor[[16, 2048], pl.BF16],
                rhs1: pl.Tensor[[2048, 64], pl.BF16],
                out0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
                out1: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 64], pl.FP32], pl.Tensor[[16, 64], pl.FP32]]:
                a0: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs0, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                b0: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs0, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a0, b0)
                out0 = pl.store(c0, [0, 0], out0)
                a1: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs1, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                b1: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs1, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a1, b1)
                out1 = pl.store(c1, [0, 0], out1)
                return out0, out1

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs0: pl.Tensor[[16, 2048], pl.BF16],
                rhs0: pl.Tensor[[2048, 64], pl.BF16],
                lhs1: pl.Tensor[[16, 2048], pl.BF16],
                rhs1: pl.Tensor[[2048, 64], pl.BF16],
                out0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
                out1: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 64], pl.FP32], pl.Tensor[[16, 64], pl.FP32]]:
                a0: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs0, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                b0: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs0, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c0_init: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for ko0, (c0_iter,) in pl.pipeline(0, 2048, 256, init_values=(c0_init,), stage=2):
                    sa0: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        a0, 0, ko0, shape=[16, 256], target_memory=pl.Mem.Left
                    )
                    sb0: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        b0, ko0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                    )
                    c0_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                        c0_iter, sa0, sb0, ko0 == 0
                    )
                    c0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(c0_acc)
                out0 = pl.store(c0, [0, 0], out0)
                a1: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs1, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                b1: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs1, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c1_init: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for ko1, (c1_iter,) in pl.pipeline(0, 2048, 256, init_values=(c1_init,), stage=2):
                    sa1: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        a1, 0, ko1, shape=[16, 256], target_memory=pl.Mem.Left
                    )
                    sb1: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        b1, ko1, 0, shape=[256, 64], target_memory=pl.Mem.Right
                    )
                    c1_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                        c1_iter, sa1, sb1, ko1 == 0
                    )
                    c1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(c1_acc)
                out1 = pl.store(c1, [0, 0], out1)
                return out0, out1

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_vec_right_operand_left_untouched(self):
        """The right (B) operand must be Mat — it feeds L0B from L1.  A Vec
        right operand (even with a Mat left) is out of scope: the asymmetry is
        deliberate (only the left / A operand may be Vec, for the PV pattern)."""

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
                # rhs lands in Vec — not a valid L0B source, so the pass skips.
                rhs_vec: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(rhs, [0, 0], [2048, 64])
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_vec)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)

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
        After = passes.auto_tile_matmul_l0()(Before)
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

        once = passes.auto_tile_matmul_l0()(Before)
        # First run must have rewritten — otherwise the idempotency check is
        # vacuously true.
        with pytest.raises(ValueError, match="Structural mismatch"):
            ir.assert_structural_equal(once, Before)
        twice = passes.auto_tile_matmul_l0()(once)
        ir.assert_structural_equal(twice, once)

    def test_non_aligned_K_left_untouched(self):
        """Non-16-aligned K has no valid L0 K-tiling: any peeled tail or whole-K block
        would have non-16-aligned (non-fractal) tile cols that ptoas rejects, so the
        pass leaves the matmul untouched (PH-AT-007 PerfHint) instead of emitting
        invalid extracts.  K=2050 (M=16, N=64) is not a multiple of the cube fractal
        16.  The device-valid 16-aligned-K peel is covered in the st suite
        (``tests/st/runtime/ops/test_matmul.py::...test_matmul_autol0_nonaligned_k``,
        K=688); the chooser-level rejection in
        ``test_l0_tile_chooser.py::...test_non_aligned_K_rejected``."""

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

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)  # non-aligned K -> untouched

    def test_matmul_bias_k_split_applies_bias_once(self):
        """The first K block applies bias; every later block only accumulates.

        ``tile.matmul_bias`` has no ``init_cond`` operand, so it cannot use the
        predicated body a plain ``tile.matmul`` gets.  The first K block is
        *head-peeled* out of the loop instead: it applies the bias exactly once
        and mints the accumulator, and the loop then accumulates into it
        uniformly.  An ``if ko == 0`` phi between ``tile.matmul_bias`` and
        ``tile.matmul_acc`` would give one logical value two producers on two
        L0C buffers, which no target can realize (there is no Acc->Acc copy) --
        so this test asserts the branch is *absent*.
        """
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                bias: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, 64], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        printed = ir.python_print(After)
        assert printed.count("pl.tile.matmul_bias(") == 1
        assert "pl.tile.matmul_acc(" in printed
        assert "if " not in printed, (
            "the bias first block is head-peeled, not branched: an IfStmt phi would put the "
            f"accumulator on two L0C buffers\n{printed}"
        )
        # The peel is structural: matmul_bias precedes the loop, and the loop
        # covers the *remaining* full blocks, so K is still covered exactly once.
        assert printed.index("pl.tile.matmul_bias(") < printed.index("pl.pipeline("), printed
        assert "pl.pipeline(256, 2048, 256" in printed, printed
        assert printed.count("pl.tile.move(bias_mat, target_memory=pl.Mem.Bias)") == 1
        assert "pl.tile.extract(bias_mat" not in printed
        _assert_ssa_valid(After, "test_matmul_bias_k_split_applies_bias_once")

    def test_matmul_bias_a2a3_float_mat_bias_is_supported(self):
        """The pinned A2/A3 ISA supports an FP32 Mat-to-Bias move."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                bias: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, 64], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_mat)
                out = pl.store(c, [0, 0], out)
                return out

        printed = ir.python_print(passes.auto_tile_matmul_l0()(Before))
        assert "pl.tile.matmul_acc(" in printed
        assert "pl.tile.move(bias_mat, target_memory=pl.Mem.Bias)" in printed

    def test_matmul_bias_a2a3_int_k_split_is_supported(self):
        """A2/A3 supports the INT32 Mat-to-Bias path used by INT8 matmul."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.INT8],
                rhs: pl.Tensor[[2048, 64], pl.INT8],
                bias: pl.Tensor[[1, 64], pl.INT32],
                out: pl.Out[pl.Tensor[[16, 64], pl.INT32]],
            ) -> pl.Tensor[[16, 64], pl.INT32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, 64], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_mat)
                out = pl.store(c, [0, 0], out)
                return out

        after = passes.auto_tile_matmul_l0()(Before)
        printed = ir.python_print(after)
        assert "pl.tile.matmul_acc(" in printed
        assert "pl.tile.move(bias_mat, target_memory=pl.Mem.Bias)" in printed
        # The biased head and accumulating tail share the selected K grid.
        calls = _matmul_acc_calls(after)
        assert calls
        for call in calls:
            lhs, rhs = call.args[1:3]
            assert isinstance(lhs.type, ir.TileType) and isinstance(rhs.type, ir.TileType)
            left_k, right_k = lhs.type.shape[1], rhs.type.shape[0]
            assert isinstance(left_k, ir.ConstInt) and isinstance(right_k, ir.ConstInt)
            assert left_k.value == right_k.value and left_k.value % 32 == 0


def _torch_codegen_matches_matmul(program, m_dim, n_dim, k_dim):
    """Drive ``program`` through ``torch_codegen`` and check the executed
    reference matches ``torch.matmul``.  Used to numerically validate the M/N
    + K tiled output the pass emits, independent of the device toolchain.

    Returns ``(ok, max_abs_diff)``.  The generated entry is named ``kernel``
    (the function name in the Before/After programs below).
    """
    torch = pytest.importorskip("torch")
    from pypto.debug import torch_codegen  # noqa: PLC0415

    torch.manual_seed(0)
    a = torch.randn(m_dim, k_dim, dtype=torch.float32)
    b = torch.randn(k_dim, n_dim, dtype=torch.float32)
    out = torch.zeros(m_dim, n_dim, dtype=torch.float32)

    code = torch_codegen(program)
    ns: dict = {}
    exec(code, ns)  # noqa: S102 — executing generated reference code is the point
    ns["kernel"](a, b, out)
    expected = torch.matmul(a, b)
    return torch.allclose(out, expected, rtol=1e-3, atol=1e-3), (out - expected).abs().max().item()


def _assert_ssa_valid(after, label):
    """Assert the rewritten program still satisfies ``SSAForm`` + ``UseAfterDef``.

    The snake reuses one Left/Right extract Var across several ``tile.matmul``s,
    so every reused operand must remain defined before all its uses — a check
    that would catch a stale/dangling reuse the same way the reversed-store
    regression catches a stale remap.
    """
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.SSAForm)
    props.insert(passes.IRProperty.UseAfterDef)
    passes.verify_properties(props, after, label)


def _assert_pipelined_full_k(after, n_pipeline_levels=2):
    """Assert the full-K M/N path emitted its nested pipelined **interior**:
    ``n_pipeline_levels`` ``pl.pipeline`` loops (the straight-line boundary tail,
    if any, adds no pipelines), no K-loop accumulation, at least the interior
    matmul + store, and every interior loop's trip count is exact (``full_m`` /
    ``full_n`` are multiples of ``m`` / ``n`` by construction)."""
    import re  # noqa: PLC0415

    printed = ir.python_print(after)
    assert printed.count("pl.pipeline(") == n_pipeline_levels, (
        f"expected {n_pipeline_levels} pipelined interior loops, got {printed.count('pl.pipeline(')}"
    )
    assert "matmul_acc" not in printed, "full-K body is a single matmul, no accumulation"
    assert printed.count("pl.tile.matmul(") >= 1 and printed.count("pl.tile.store(") >= 1, (
        "the full-K schedule has at least the interior matmul + store (plus any boundary tail tiles)"
    )
    bounds = re.findall(r"pl\.pipeline\(0, (\d+), (\d+)", printed)
    assert len(bounds) == n_pipeline_levels, "every pipeline loop should be pl.pipeline(0, stop, step, ...)"
    for stop, step in bounds:
        assert int(stop) % int(step) == 0, f"interior trip count must be exact: stop={stop} step={step}"


def _full_k_stationary_operand(after) -> str:
    """Which operand the full-K interior keeps stationary in the OUTER loop —
    ``"A"`` (row-outer) or ``"B"`` (column-outer).  The stationary panel is the
    single ``tile.extract`` emitted in the outer loop body, between the outer and
    inner ``pl.pipeline`` headers: a ``Mem.Left`` extract ⇒ A-stationary, a
    ``Mem.Right`` extract ⇒ B-stationary."""
    lines = ir.python_print(after).splitlines()
    outer_i = next(i for i, ln in enumerate(lines) if "pl.pipeline(" in ln)
    inner_i = next(i for i in range(outer_i + 1, len(lines)) if "pl.pipeline(" in lines[i])
    for i in range(outer_i + 1, inner_i):
        if ".extract(" in lines[i] and "pl.Mem.Left" in lines[i]:
            return "A"
        if ".extract(" in lines[i] and "pl.Mem.Right" in lines[i]:
            return "B"
    raise AssertionError("no stationary extract found in the outer loop body")


def _lower_to_tile_ops(program):
    """Run the tensor→tile lowering prefix so a tensor-level chained matmul reaches
    ``AutoTileMatmulL0`` as the real ``c = tile.matmul(a, b); d = tile.matmul(c, e)``
    it sees in the pipeline (the chained tile-matmul is not hand-constructible — the
    user-facing op guard rejects an Acc operand, but ConvertTensorToTileOps builds it
    internally)."""
    for p in (
        passes.convert_to_ssa(),
        passes.convert_tensor_to_tile_ops(),
        passes.lower_composite_ops(),
        passes.flatten_tile_nd_to_2d(),
    ):
        program = p(program)
    return program


def _assert_unchanged_by_pass(before, after):
    """Assert ``AutoTileMatmulL0`` left ``before`` structurally unchanged.

    ``after`` is the pass run over the lowered ``before``. The golden is a FRESH
    prerequisite-only lowering of ``before``: the pass under test never runs on the
    right-hand side, and the golden is not the same object the pass was handed, so
    neither a rewrite nor an in-place mutation can cancel out on both sides.

    Use this for the ``..._not_folded`` guards: they assert the pass *declines*
    to rewrite, so the whole program is the oracle. A substring probe for one op
    cannot distinguish those cases from each other -- every ``_not_folded`` body
    prints both ``pl.tile.cast(`` and no ``pl.tile.assemble(``.
    """
    ir.assert_structural_equal(after, _lower_to_tile_ops(before))


_TILE_MATMUL_ACC = ir.get_op("tile.matmul_acc").name


def _matmul_acc_calls(node) -> list:
    """Every ``tile.matmul_acc`` Call under ``node``, in visit order.

    Used by the predicated-accumulator tests to count operands per emission
    site: the tail block must stay 3-operand while the loop body carries the
    composed predicate as a 4th.
    """
    calls: list = []

    class _Collector(ir.IRVisitor):
        def visit_call(self, call):  # type: ignore[override]
            if isinstance(call.op, ir.Op) and call.op.name == _TILE_MATMUL_ACC:
                calls.append(call)
            super().visit_call(call)

    _Collector().visit_program(node)
    return calls


_CUBE_M_AXIS_TILE = re.compile(r"pl\.Tile\[\[(\d+), \d+\], [^\]]*?pl\.Mem\.(Left|Acc)")


def _cube_m_axis_rows(after) -> list[int]:
    """Static physical row counts of every ``Left`` / ``Acc`` tile in ``after``.

    Both spaces index the matmul's M axis, whose NZ fractal is 16 rows on every
    Ascend generation, so their physical row count is directly comparable against
    that one constant. ``Right`` is deliberately excluded: its rows are ``K``,
    whose granularity is ``32 bytes / sizeof(dtype)`` and therefore dtype-dependent.
    """
    return [int(rows) for rows, _ in _CUBE_M_AXIS_TILE.findall(ir.python_print(after))]


class TestAutoTileMatmulL0FractalBoundary:
    """No cube operand leaves the pass with a sub-fractal physical row count.

    Regression for an M that is not a multiple of 16. The chooser picks a
    16-aligned tile and the pass peels the remainder, so the tail used to carry
    ``M mod m`` physical rows -- 36 for ``M = 100`` -- straight into ``Left`` and
    ``Acc``. ptoas rejects such an allocation outright (``'pto.alloc_tile' op
    expects result boxed tile rows to be a multiple of innerRows (16)``), and at
    ``M = 1`` it accepts a one-row cube operand that pto-isa's ``TExtract`` cannot
    address.

    ``ConvertTensorToTileOps`` now loads the left operand into a whole number of
    boxes with the true extent in ``valid_shape``, so the M reaching this pass is
    already a multiple of 16 -- and a multiple of 16 can only be tiled into
    16-aligned pieces, tail included. The invariant therefore holds for every M,
    with no boundary special case in the pass.
    """

    @pytest.mark.parametrize("m_dim", [1, 17, 40, 100, 250])
    def test_no_sub_fractal_cube_rows_at_any_m(self, m_dim):
        k_dim, n_dim = 256, 512

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[m_dim, k_dim], pl.FP16],
                b: pl.Tensor[[k_dim, n_dim], pl.FP16],
                out: pl.Out[pl.Tensor[[m_dim, n_dim], pl.FP32]],
            ) -> pl.Tensor[[m_dim, n_dim], pl.FP32]:
                c = pl.matmul(a, b, out_dtype=pl.FP32)
                out = pl.assemble(out, c, [0, 0])
                return out

        After = passes.auto_tile_matmul_l0()(_lower_to_tile_ops(Before))
        rows = _cube_m_axis_rows(After)
        assert rows, "expected at least one Left/Acc tile in the lowered program"
        sub_fractal = sorted({r for r in rows if r % 16})
        assert not sub_fractal, f"sub-fractal cube rows for M={m_dim}: {sub_fractal}"

        # The padding is physical only: the accumulator still names the M the
        # caller asked for, so the store writes exactly m_dim rows. (An N-tiled
        # schedule splits the column extent across sub-tiles, so match the row
        # extent alone.)
        printed = ir.python_print(After)
        assert f"valid_shape=[{m_dim}, " in printed, f"the accumulator must carry the true M={m_dim} extent"

    @pytest.mark.parametrize("m_dim", [1, 17, 40, 100, 250])
    def test_no_sub_fractal_cube_rows_for_the_accumulating_spelling(self, m_dim):
        """``pl.matmul_acc`` holds the same invariant at the same set of M.

        The accumulating spelling has a second cube tile the rule has to reach:
        the accumulator, which is allocated rather than loaded. Since
        ``tile.matmul_acc`` requires it and the product to agree on physical M,
        an accumulator left at the declared extent would either reach ptoas
        sub-fractal or be rejected outright against a boxed operand -- so the
        two are boxed together, and this pass sees a 16-aligned M either way.
        """
        k_dim, n_dim = 256, 512

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[m_dim, k_dim], pl.FP16],
                b: pl.Tensor[[k_dim, n_dim], pl.FP16],
                out: pl.Out[pl.Tensor[[m_dim, n_dim], pl.FP32]],
            ) -> pl.Tensor[[m_dim, n_dim], pl.FP32]:
                acc = pl.create_tensor([m_dim, n_dim], pl.FP32)
                c = pl.matmul_acc(acc, a, b)
                out = pl.assemble(out, c, [0, 0])
                return out

        After = passes.auto_tile_matmul_l0()(_lower_to_tile_ops(Before))
        rows = _cube_m_axis_rows(After)
        assert rows, "expected at least one Left/Acc tile in the lowered program"
        sub_fractal = sorted({r for r in rows if r % 16})
        assert not sub_fractal, f"sub-fractal cube rows for M={m_dim}: {sub_fractal}"

        printed = ir.python_print(After)
        assert f"valid_shape=[{m_dim}, " in printed, f"the accumulator must carry the true M={m_dim} extent"


class TestAutoTileMatmulL0PredicatedAcc:
    """A caller-written ``init_cond`` is threaded through the K-only emitter.

    ``pl.tile.matmul_acc(acc, a, b, init_cond=...)`` is the split-K idiom: the
    predicate means "this is the first K step of the *user's* reduction", so on
    those steps the accumulator is overwritten instead of accumulated into.
    AutoTileMatmulL0 used to refuse the 4-operand spelling outright, leaving an
    oversized predicated call untiled.  It now tiles it, composing the caller's
    predicate with the ``ko == 0`` its own K-loop introduces.

    Per the pass's three emission sites (``BuildKLoopRewrite``):

    * pipelined K-loop (``num_full >= 2``) → ``user_cond and ko == 0``
    * lone straight-line full block (``num_full == 1``) → ``user_cond`` verbatim
    * peeled partial tail → no predicate at all (it is never the first block)
    """

    @staticmethod
    def _predicated_acc_program(K: int, M: int = 16, N: int = 64):
        """A predicated ``tile.matmul_acc`` over Mat-resident [M, K] x [K, N]."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.BF16],
                rhs: pl.Tensor[[K, N], pl.BF16],
                acc_init: pl.Tile[[M, N], pl.FP32, pl.Mem.Acc],
                first_k: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_acc(acc_init, lhs_mat, rhs_mat, init_cond=(first_k == 0))
                out = pl.store(c, [0, 0], out)
                return out

        return Before

    def test_predicated_matmul_acc_k_tiled_composes_predicate(self):
        """16×64 @ 2048 BF16 → (m=16, n=64, k=256), 8 full blocks and no tail.

        The whole point of B1: the call is tiled at all, and the two predicates
        are ANDed rather than either one being dropped.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                acc_init: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc],
                first_k: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                    acc_init, lhs_mat, rhs_mat, init_cond=(first_k == 0)
                )
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
                first_k: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                # Iter-arg init is the caller's acc_init, exactly as for the
                # unpredicated 3-operand spelling — only the call gains a 4th
                # operand.
                for ko, (c_iter,) in pl.pipeline(0, 2048, 256, init_values=(acc_init,), stage=2):
                    sa: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 0, ko, shape=[16, 256], target_memory=pl.Mem.Left
                    )
                    sb: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko, 0, shape=[256, 64], target_memory=pl.Mem.Right
                    )
                    c_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                        c_iter, sa, sb, first_k == 0 and ko == 0
                    )
                    c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(c_acc)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_predicated_matmul_acc_k_boundary_tail_is_unpredicated(self):
        """K=544 → (k=192): two pipelined full blocks plus a 160-wide tail.

        The tail is the one site where forwarding the predicate would be a
        correctness bug — it would re-zero the accumulator on the last K block,
        discarding the full blocks' partial sum.
        """
        Before = self._predicated_acc_program(K=544, M=64)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[64, 544], pl.BF16],
                rhs: pl.Tensor[[544, 64], pl.BF16],
                acc_init: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc],
                first_k: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                lhs_mat: pl.Tile[[64, 544], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [64, 544], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[544, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [544, 64], target_memory=pl.Mem.Mat
                )
                # Two full 192-wide blocks: the body ANDs the caller's predicate
                # with the ko == 0 this loop introduces.
                for ko, (c_iter,) in pl.pipeline(0, 384, 192, init_values=(acc_init,), stage=2):
                    sa: pl.Tile[[64, 192], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 0, ko, shape=[64, 192], target_memory=pl.Mem.Left
                    )
                    sb: pl.Tile[[192, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko, 0, shape=[192, 64], target_memory=pl.Mem.Right
                    )
                    c_acc: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                        c_iter, sa, sb, first_k == 0 and ko == 0
                    )
                    c_main: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(c_acc)
                # The 160-wide tail runs at K offset 384, so it is never the
                # first block: it carries no predicate at all.
                sa_t: pl.Tile[[64, 160], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 0, 384, shape=[64, 160], target_memory=pl.Mem.Left
                )
                sb_t: pl.Tile[[160, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 384, 0, shape=[160, 64], target_memory=pl.Mem.Right
                )
                c_tail: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(c_main, sa_t, sb_t)
                out = pl.store(c_tail, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_predicated_matmul_acc_single_full_block_carries_user_cond_unmodified(self):
        """K=272 → (k=144): one straight-line full block plus a 128-wide tail.

        The lone full block *is* the first K block, so a generated ``ko == 0``
        term would be statically true; the caller's predicate is forwarded
        verbatim instead.  The tail behind it still drops the predicate.
        """
        Before = self._predicated_acc_program(K=272, M=64)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[64, 272], pl.BF16],
                rhs: pl.Tensor[[272, 64], pl.BF16],
                acc_init: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc],
                first_k: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                lhs_mat: pl.Tile[[64, 272], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [64, 272], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[272, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [272, 64], target_memory=pl.Mem.Mat
                )
                # One 144-wide full block, straight-line: no pipeline loop, because
                # a 1-trip one would be degenerate.  It *is* the first K block, so a
                # generated ko == 0 term would be statically true — the caller's
                # predicate is forwarded verbatim.
                sa_0: pl.Tile[[64, 144], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 0, 0, shape=[64, 144], target_memory=pl.Mem.Left
                )
                sb_0: pl.Tile[[144, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[144, 64], target_memory=pl.Mem.Right
                )
                c_0: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                    acc_init, sa_0, sb_0, first_k == 0
                )
                # The 128-wide tail behind it still drops the predicate.
                sa_t: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 0, 144, shape=[64, 128], target_memory=pl.Mem.Left
                )
                sb_t: pl.Tile[[128, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 144, 0, shape=[128, 64], target_memory=pl.Mem.Right
                )
                c_tail: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(c_0, sa_t, sb_t)
                out = pl.store(c_tail, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_predicated_matmul_acc_tiling_preserves_ssa(self):
        """The predicate is re-parented into the emitted loop body.

        ``AnalyzeMatmul`` captures the caller's operand expression itself, and
        the ForStmt that consumes it replaces the original AssignStmt in the
        same SeqStmts — so the predicate's operands must still dominate their
        single new use inside the loop.
        """
        After = passes.auto_tile_matmul_l0()(self._predicated_acc_program(K=2048))
        _assert_ssa_valid(After, "test_predicated_matmul_acc_tiling_preserves_ssa")

    def test_predicated_matmul_acc_uses_one_acc_buffer(self):
        """One logical accumulator lands on one L0C buffer through Default.

        This is what the predicated idiom exists for: no ``if ko == 0`` phi
        means no second Acc tile to reconcile, and L0C has no Acc→Acc copy that
        could reconcile one.
        """
        from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                first_k: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat)
                # InCore parameters must be TensorType, so the accumulator is
                # created here rather than passed in.
                acc = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                c = pl.tile.matmul_acc(acc, lhs_mat, rhs_mat, init_cond=(first_k == 0))
                out = pl.store(c, [0, 0], out)
                return out

        allocated = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before)
        printed = ir.python_print(allocated)

        acc_allocs = [line for line in printed.splitlines() if "tile.alloc(pl.Mem.Acc" in line]
        assert len(acc_allocs) == 1, f"one logical accumulator, one L0C buffer; got {acc_allocs}"
        assert not [
            line for line in printed.splitlines() if "pl.tile.move(" in line and "pl.Mem.Acc" in line
        ], "there is no Acc→Acc copy on the machine; the chain must not need one"

    def test_literal_true_init_cond_is_composed_not_folded(self):
        """A literal ``True`` predicate is composed, never folded to ``tile.matmul``.

        Folding it back to a fresh ``tile.matmul`` would mint a second L0C
        buffer and reintroduce the divergent accumulator the predicated form
        exists to remove.  The backend emitter already selects the right
        instruction for a literal predicate, so there is nothing to gain.
        """

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
                lhs_mat = pl.tile.load(lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_acc(acc_init, lhs_mat, rhs_mat, init_cond=True)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)

        printed = ir.python_print(After)
        assert "pl.tile.matmul(" not in printed, "a true predicate must not become a fresh matmul"

        (body_call,) = _matmul_acc_calls(After)
        assert len(body_call.args) == 4
        assert ir.python_print(body_call.args[3]) == "True and c_l0_ko == 0"


class TestAutoTileMatmulL0MNTiling:
    """M/N output tiling.

    When ``ChooseL0Tile`` picks ``m < M`` or ``n < N`` the [M, N] output Acc
    overflows L0c.  The operands are already Mat-resident, so only the output
    overflows: the pass tiles the output into a ``ceil(M/m) x ceil(N/n)`` grid
    of ``[m, n]`` (partial on the boundary) sub-tiles, each computed by the
    existing pipelined K-loop and stored straight to ``out[mi:, ni:]`` (the
    direct-store / DDR-output path).  The output tensor is chained through the
    per-sub-tile stores in SSA form.
    """

    def test_matmul_bias_mn_and_k_tiling_slices_bias_by_n(self):
        """Each output-column tile reloads one Bias window and applies it once."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)
        M, K, N = 256, 1024, 512

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.INT8],
                rhs: pl.Tensor[[K, N], pl.INT8],
                bias: pl.Tensor[[1, N], pl.INT32],
                out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> pl.Tensor[[M, N], pl.INT32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, N], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        printed = ir.python_print(After)
        assert "pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_mat)" not in printed
        assert "pl.tile.matmul_bias(" in printed
        assert "pl.tile.matmul_acc(" in printed
        assert "pl.tile.slice(bias_mat" not in printed
        assert re.search(r"pl\.tile\.load\(\s*bias,", printed)
        assert not re.search(r"pl\.tile\.load\(\s*bias,\s*\[0, 0\],\s*\[1, 512\]", printed), (
            "the original redundant full bias load must be removed"
        )
        assert "pl.tile.move(" in printed and "target_memory=pl.Mem.Bias" in printed
        assert "pl.tile.extract(bias_mat" not in printed
        assert "target_memory=pl.Mem.Bias" in printed
        assert printed.count("pl.tile.store(") >= 2, "the oversized output must use a direct-store grid"
        _assert_ssa_valid(After, "test_matmul_bias_mn_and_k_tiling_slices_bias_by_n")

        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        torch.manual_seed(0)
        # Keep each unsplit torch reference dot inside INT8 range. Torch's
        # debug backend preserves the input dtype for `@`, whereas the device
        # cube accumulates INT8 products in INT32.
        lhs = torch.randint(-1, 2, (M, K), dtype=torch.int8)
        rhs = torch.randint(-1, 2, (K, N), dtype=torch.int8)
        bias = torch.randint(-20, 21, (1, N), dtype=torch.int32)
        out = torch.zeros(M, N, dtype=torch.int32)
        ns: dict = {}
        exec(torch_codegen(After), ns)  # noqa: S102 -- executing generated reference code is the point
        ns["kernel"](lhs, rhs, bias, out)
        expected = lhs.int() @ rhs.int() + bias
        assert torch.equal(out, expected), (
            f"mismatches={(out != expected).sum().item()}, max_abs={(out - expected).abs().max().item()}"
        )

    def test_matmul_bias_n_tiling_with_partial_valid_load_is_deferred(self):
        """A narrowed bias snapshot cannot be widened by reconstructed N-window loads."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)
        M, K, N = 256, 1024, 512

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.INT8],
                rhs: pl.Tensor[[K, N], pl.INT8],
                bias: pl.Tensor[[1, N], pl.INT32],
                out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> pl.Tensor[[M, N], pl.INT32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [K, N], [K, N - 16], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, N], [1, N - 16], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)

    def test_matmul_bias_n_tiling_without_store_reports_placement_hint(self, capfd):
        """A missing store is a placement gap, not a bias-snapshot ordering hazard."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)
        M, K, N = 256, 1024, 512

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.INT8],
                rhs: pl.Tensor[[K, N], pl.INT8],
                bias: pl.Tensor[[1, N], pl.INT32],
                out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> pl.Tensor[[M, N], pl.INT32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, N], target_memory=pl.Mem.Mat)
                _ = pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_mat)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)
        diagnostics = capfd.readouterr().err
        assert "PH-AT-006" in diagnostics
        assert "PH-AT-011" not in diagnostics

    def test_matmul_bias_n_tiling_with_intervening_store_is_deferred(self):
        """Reloading after an intervening effect must not replace the earlier bias snapshot."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)
        M, K, N = 256, 1024, 512

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.INT8],
                rhs: pl.Tensor[[K, N], pl.INT8],
                bias: pl.Tensor[[1, N], pl.INT32],
                out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> pl.Tensor[[M, N], pl.INT32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                snapshot_mat = pl.tile.load(bias, [0, 0], [1, N], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, N], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_mat)
                # Keep bias_mat single-use so only the post-matmul barrier, not
                # the load-use-count guard, rejects deferred reconstruction.
                out_snapshot = pl.store(snapshot_mat, [0, 0], out)
                out_final = pl.store(c, [0, 0], out_snapshot)
                return out_final

        ir.assert_structural_equal(passes.auto_tile_matmul_l0()(Before), Before)

    def test_matmul_bias_partial_n_boundary_keeps_logical_store_extent(self):
        """A 16-column N tail is sliced from bias and stored only at its logical width.

        ``Expected`` pins the whole fold: the tail's bias arrives as its own
        narrow ``tile.load(bias, [0, 512], [1, 16], [1, 16])`` routed through
        ``Mem.Bias`` — never as a ``tile.slice`` / ``tile.extract`` of the full
        ``bias_mat`` — and the tail store lands at column 512 at its logical
        16-column width.
        """
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend950)
        M, K, N = 528, 32, 528

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.BF16],
                rhs: pl.Tensor[[K, N], pl.BF16],
                bias: pl.Tensor[[1, N], pl.FP32],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, N], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_mat)
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[528, 32], pl.BF16],
                rhs: pl.Tensor[[32, 528], pl.BF16],
                bias: pl.Tensor[[1, 528], pl.FP32],
                out: pl.Out[pl.Tensor[[528, 528], pl.FP32]],
            ) -> pl.Tensor[[528, 528], pl.FP32]:
                lhs_mat: pl.Tile[[528, 32], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [528, 32], [528, 32], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[32, 528], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [32, 528], [32, 528], target_memory=pl.Mem.Mat
                )
                for c_o, (c_oc,) in pl.range(0, 528, 528, init_values=(out,)):
                    c_a: pl.Tile[
                        [528, 32], pl.BF16, pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(lhs_mat, c_o, 0, [528, 32], target_memory=pl.Mem.Left)
                    for c_i, (c_ic,) in pl.pipeline(
                        0,
                        512,
                        64,
                        stage=2,
                        init_values=(c_oc,),
                        attrs={"pipeline_overlap_stores": False},
                    ):
                        c_b: pl.Tile[[32, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                            rhs_mat, 0, c_i, [32, 64], target_memory=pl.Mem.Right
                        )
                        # The per-tile bias is loaded narrow from GM, not sliced
                        # out of the full-width bias_mat.
                        c_bias_mat: pl.Tile[
                            [1, 64],
                            pl.FP32,
                            pl.Mem.Mat,
                            pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box),
                        ] = pl.tile.load(bias, [0, c_i], [1, 64], [1, 64], target_memory=pl.Mem.Mat)
                        c_bias: pl.Tile[[1, 64], pl.FP32, pl.Mem.Bias] = pl.tile.move(
                            c_bias_mat, target_memory=pl.Mem.Bias
                        )
                        c_c: pl.Tile[[528, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_bias(c_a, c_b, c_bias)
                        out_t0: pl.Tensor[[528, 528], pl.FP32] = pl.tile.store(c_c, [c_o, c_i], c_ic)
                        c_irv = pl.yield_(out_t0)
                    c_orv = pl.yield_(c_irv)
                # The peeled 16-column tail.
                c_ta1: pl.Tile[
                    [528, 32], pl.BF16, pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                ] = pl.tile.extract(lhs_mat, 0, 0, [528, 32], target_memory=pl.Mem.Left)
                c_tb1: pl.Tile[[32, 16], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 512, [32, 16], target_memory=pl.Mem.Right
                )
                c_tbias1_mat: pl.Tile[
                    [1, 16],
                    pl.FP32,
                    pl.Mem.Mat,
                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box),
                ] = pl.tile.load(bias, [0, 512], [1, 16], [1, 16], target_memory=pl.Mem.Mat)
                c_tbias1: pl.Tile[[1, 16], pl.FP32, pl.Mem.Bias] = pl.tile.move(
                    c_tbias1_mat, target_memory=pl.Mem.Bias
                )
                c_tc1: pl.Tile[[528, 16], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_bias(c_ta1, c_tb1, c_tbias1)
                out_t1: pl.Tensor[[528, 528], pl.FP32] = pl.tile.store(c_tc1, [0, 512], c_orv)
                return out_t1

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)
        _assert_ssa_valid(After, "test_matmul_bias_partial_n_boundary")

        # Exercise the production lowering as well as the AutoTile-local IR:
        # the boundary's physical boxes and narrowed valid shapes must survive
        # memory inference and PTO codegen.
        from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415
        from pypto.pypto_core import codegen as _codegen_core  # noqa: PLC0415
        from pypto.pypto_core import ir as _ir_core  # noqa: PLC0415

        post = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before)
        pto = "\n".join(
            _codegen_core.PTOCodegen().generate(_ir_core.Program([func], func.name, post.span))
            for func in post.functions.values()
        )
        assert "pto.tmatmul.bias" in pto
        assert "pto.tload" in pto
        assert "pto.tmov" in pto

    def test_matmul_bias_window_load_uses_tensor_remapped_by_earlier_fold(self):
        """A prior M/N store fold must update a later reconstructed bias load."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)
        M, K, N = 256, 1024, 512

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a0: pl.Tensor[[M, K], pl.INT8],
                b0: pl.Tensor[[K, N], pl.INT8],
                a1: pl.Tensor[[M, K], pl.INT8],
                b1: pl.Tensor[[K, N], pl.INT8],
                scratch: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> pl.Tensor[[M, N], pl.INT32]:
                a0_mat = pl.tile.load(a0, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                b0_mat = pl.tile.load(b0, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                produced = pl.tile.matmul(a0_mat, b0_mat)
                bias_tensor = pl.store(produced, [0, 0], scratch)
                bias_mat = pl.tile.load(bias_tensor, [0, 0], [1, N], target_memory=pl.Mem.Mat)
                a1_mat = pl.tile.load(a1, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                b1_mat = pl.tile.load(b1, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                result = pl.tile.matmul_bias(a1_mat, b1_mat, bias_mat)
                out = pl.store(result, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        printed = ir.python_print(After)
        assert "pl.tile.matmul_bias(" in printed
        assert not re.search(
            r"^\s*bias_mat:.*tile\.load\(bias_tensor, \[0, 0\], \[1, 512\]", printed, re.MULTILINE
        )
        _assert_ssa_valid(After, "test_matmul_bias_window_load_uses_tensor_remapped_by_earlier_fold")

    def test_matmul_bias_capacity_caps_n_even_when_l0abc_fit(self):
        """The 910B 1 KiB bias table limits INT32 bias windows to N=256."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)
        M, K, N = 16, 32, 512

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.INT8],
                rhs: pl.Tensor[[K, N], pl.INT8],
                bias: pl.Tensor[[1, N], pl.INT32],
                out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> pl.Tensor[[M, N], pl.INT32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, N], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_mat)
                out = pl.store(c, [0, 0], out)
                return out

        printed = ir.python_print(passes.auto_tile_matmul_l0()(Before))
        assert "pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_mat)" not in printed
        windows = [
            int(n) for n in re.findall(r"pl\.tile\.load\(\s*bias,\s*\[[^]]+\],\s*\[1, (\d+)\]", printed)
        ]
        assert windows and max(windows) <= 256

    def test_matmul_bias_nonfractal_mn_is_deferred(self):
        """AutoTile does not emit physically illegal narrow cube/Bias boxes."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.INT8],
                rhs: pl.Tensor[[2048, 150], pl.INT8],
                bias: pl.Tensor[[1, 150], pl.INT32],
                out: pl.Out[pl.Tensor[[16, 150], pl.INT32]],
            ) -> pl.Tensor[[16, 150], pl.INT32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [2048, 150], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, 150], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)

    def test_matmul_bias_nondivisor_k_tail_applies_bias_once_per_output_tile(self):
        """A peeled K tail accumulates after, rather than re-applying, bias."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend950)
        M, K, N = 64, 272, 64

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.BF16],
                rhs: pl.Tensor[[K, N], pl.BF16],
                bias: pl.Tensor[[1, N], pl.FP32],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, N], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        printed = ir.python_print(After)
        assert printed.count("pl.tile.matmul_bias(") == 1
        assert printed.count("pl.tile.matmul_acc(") >= 1
        assert "_l0_bt" in printed, "expected the peeled K-tail Right extract"
        _assert_ssa_valid(After, "test_matmul_bias_nondivisor_k_tail")

        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        torch.manual_seed(2)
        lhs = torch.randn(M, K, dtype=torch.bfloat16)
        rhs = torch.randn(K, N, dtype=torch.bfloat16)
        bias = torch.randn(1, N)
        out = torch.zeros(M, N)
        ns: dict = {}
        exec(torch_codegen(After), ns)  # noqa: S102 -- executing generated reference code is the point
        ns["kernel"](lhs, rhs, bias, out)
        expected = lhs.float() @ rhs.float() + bias
        rel_err = ((out - expected).norm() / expected.norm()).item()
        assert rel_err < 5e-2, f"peeled-K matmul_bias rel_err {rel_err:.3e} exceeds 5e-2"

    def test_matmul_bias_m_only_tiling_reuses_bias_resident_source(self):
        """M-only tiling reuses one full-width Bias tile without pipeline replication."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)
        # FP32 Bias capacity is N=256 on 910B. The output needs M tiling,
        # while full N fits only when the existing Bias tile is charged once;
        # applying the Mat-window /2 pipeline bound would force unsupported N
        # tiling and incorrectly defer this legal case.
        M, K, N = 528, 64, 256

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.BF16],
                rhs: pl.Tensor[[K, N], pl.BF16],
                bias: pl.Tensor[[1, N], pl.FP32],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, N], target_memory=pl.Mem.Mat)
                bias_l0 = pl.tile.move(bias_mat, target_memory=pl.Mem.Bias)
                c = pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_l0)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        printed = ir.python_print(After)
        assert "pl.pipeline(" in printed
        assert printed.count("pl.tile.move(bias_mat, target_memory=pl.Mem.Bias)") == 1
        assert "pl.tile.extract(bias_l0" not in printed
        assert printed.count("pl.tile.matmul_bias(") == 2
        _assert_ssa_valid(After, "test_matmul_bias_m_only_tiling_reuses_bias_resident_source")

    @pytest.mark.parametrize("batch_offsets", [None, [], [1], [1, 2]])
    def test_mn_tiling_rewrites_to_subtile_grid(self, batch_offsets):
        """512×512 @ 512 FP32 on Ascend950 (L0c = 256 KB): the [512, 512] FP32
        output is 1 MB > L0c, so ChooseL0Tile picks m = n = 256, k = 32.  The
        pass unrolls the output into a 2×2 grid of [256, 256] Acc sub-tiles —
        each an independent 16-trip pipelined K-loop — and stores each straight
        to ``out[mi:, ni:]``, chaining the output tensor through the four
        stores (out → out_t0 → out_t1 → out_t2 → out_t3)."""

        # Explicit partition windows cover 2D, rank-3 and rank-4 stores.
        # Nonzero matrix/batch offsets pin the destination mapping and the SSA
        # chain, while the original three-argument store remains a baseline.
        prefix = batch_offsets or []
        explicit = batch_offsets is not None
        output_shape = [i + 1 for i in prefix] + ([544, 576] if explicit else [512, 512])
        row_base, col_base = (16, 32) if explicit else (0, 0)
        offsets = prefix + [row_base, col_base]
        offsets_m = prefix + [row_base + 256, col_base]
        offsets_n = prefix + [row_base, col_base + 256]
        offsets_mn = prefix + [row_base + 256, col_base + 256]
        full_window = [1] * len(prefix) + [512, 512] if explicit else None
        sub_window = [1] * len(prefix) + [256, 256] if explicit else None

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[512, 512], pl.FP32],
                rhs: pl.Tensor[[512, 512], pl.FP32],
                out: pl.Out[pl.Tensor[output_shape, pl.FP32]],
            ) -> pl.Tensor[output_shape, pl.FP32]:
                lhs_mat: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[512, 512], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.tile.store(c, offsets, out, shapes=full_window)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[512, 512], pl.FP32],
                rhs: pl.Tensor[[512, 512], pl.FP32],
                out: pl.Out[pl.Tensor[output_shape, pl.FP32]],
            ) -> pl.Tensor[output_shape, pl.FP32]:
                lhs_mat: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                # Sub-tile (mi=0, ni=0).
                c0_init: pl.Tile[[256, 256], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [256, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for ko0, (c0_iter,) in pl.pipeline(0, 512, 32, init_values=(c0_init,), stage=2):
                    a0: pl.Tile[[256, 32], pl.FP32, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 0, ko0, shape=[256, 32], target_memory=pl.Mem.Left
                    )
                    b0: pl.Tile[[32, 256], pl.FP32, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko0, 0, shape=[32, 256], target_memory=pl.Mem.Right
                    )
                    c0_acc: pl.Tile[[256, 256], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                        c0_iter, a0, b0, ko0 == 0
                    )
                    c0: pl.Tile[[256, 256], pl.FP32, pl.Mem.Acc] = pl.yield_(c0_acc)
                out_t0: pl.Tensor[output_shape, pl.FP32] = pl.tile.store(c0, offsets, out, shapes=sub_window)
                # Sub-tile (mi=256, ni=0).
                c1_init: pl.Tile[[256, 256], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [256, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for ko1, (c1_iter,) in pl.pipeline(0, 512, 32, init_values=(c1_init,), stage=2):
                    a1: pl.Tile[[256, 32], pl.FP32, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 256, ko1, shape=[256, 32], target_memory=pl.Mem.Left
                    )
                    b1: pl.Tile[[32, 256], pl.FP32, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko1, 0, shape=[32, 256], target_memory=pl.Mem.Right
                    )
                    c1_acc: pl.Tile[[256, 256], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                        c1_iter, a1, b1, ko1 == 0
                    )
                    c1: pl.Tile[[256, 256], pl.FP32, pl.Mem.Acc] = pl.yield_(c1_acc)
                out_t1: pl.Tensor[output_shape, pl.FP32] = pl.tile.store(
                    c1, offsets_m, out_t0, shapes=sub_window
                )
                # Sub-tile (mi=0, ni=256).
                c2_init: pl.Tile[[256, 256], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [256, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for ko2, (c2_iter,) in pl.pipeline(0, 512, 32, init_values=(c2_init,), stage=2):
                    a2: pl.Tile[[256, 32], pl.FP32, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 0, ko2, shape=[256, 32], target_memory=pl.Mem.Left
                    )
                    b2: pl.Tile[[32, 256], pl.FP32, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko2, 256, shape=[32, 256], target_memory=pl.Mem.Right
                    )
                    c2_acc: pl.Tile[[256, 256], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                        c2_iter, a2, b2, ko2 == 0
                    )
                    c2: pl.Tile[[256, 256], pl.FP32, pl.Mem.Acc] = pl.yield_(c2_acc)
                out_t2: pl.Tensor[output_shape, pl.FP32] = pl.tile.store(
                    c2, offsets_n, out_t1, shapes=sub_window
                )
                # Sub-tile (mi=256, ni=256).
                c3_init: pl.Tile[[256, 256], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [256, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for ko3, (c3_iter,) in pl.pipeline(0, 512, 32, init_values=(c3_init,), stage=2):
                    a3: pl.Tile[[256, 32], pl.FP32, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 256, ko3, shape=[256, 32], target_memory=pl.Mem.Left
                    )
                    b3: pl.Tile[[32, 256], pl.FP32, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko3, 256, shape=[32, 256], target_memory=pl.Mem.Right
                    )
                    c3_acc: pl.Tile[[256, 256], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                        c3_iter, a3, b3, ko3 == 0
                    )
                    c3: pl.Tile[[256, 256], pl.FP32, pl.Mem.Acc] = pl.yield_(c3_acc)
                out_t3: pl.Tensor[output_shape, pl.FP32] = pl.tile.store(
                    c3, offsets_mn, out_t2, shapes=sub_window
                )
                return out_t3

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_mn_tiling_numerically_correct(self):
        """The 2×2-tiled 512×512 output (clean tiles) numerically matches
        ``torch.matmul`` when driven through ``torch_codegen``."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[512, 512], pl.FP32],
                rhs: pl.Tensor[[512, 512], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 512], pl.FP32]],
            ) -> pl.Tensor[[512, 512], pl.FP32]:
                lhs_mat: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[512, 512], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        # Sanity: the pass actually tiled (otherwise the numeric check is vacuous).
        with pytest.raises(ValueError, match="Structural mismatch"):
            ir.assert_structural_equal(After, Before)
        ok, max_diff = _torch_codegen_matches_matmul(After, 512, 512, 512)
        assert ok, f"512×512 M/N-tiled output mismatch: max abs diff {max_diff:.3e}"

    def test_mn_tiling_partial_tiles_numerically_correct(self):
        """384×384 @ 512 FP32 on Ascend950: ChooseL0Tile still picks m = n = 256,
        so the output tiles into a 2×2 grid with **partial boundary sub-tiles**
        (256 + 128 on each axis → sub-tiles 256×256, 256×128, 128×256, 128×128).
        Exercises static partial-extent handling; the result must still match
        ``torch.matmul``."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[384, 512], pl.FP32],
                rhs: pl.Tensor[[512, 384], pl.FP32],
                out: pl.Out[pl.Tensor[[384, 384], pl.FP32]],
            ) -> pl.Tensor[[384, 384], pl.FP32]:
                lhs_mat: pl.Tile[[384, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [384, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[512, 384], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [512, 384], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[384, 384], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        with pytest.raises(ValueError, match="Structural mismatch"):
            ir.assert_structural_equal(After, Before)
        ok, max_diff = _torch_codegen_matches_matmul(After, 384, 384, 512)
        assert ok, f"384×384 partial-tile output mismatch: max abs diff {max_diff:.3e}"

    def test_mn_tiling_end_to_end_no_l0c_overflow(self):
        """End-to-end acceptance: a 256×256 @ 256 FP32 matmul on Ascend910
        (output 256 KB > L0c = 128 KB; operands fit L1) compiles through the
        **full** pass pipeline — M/N tiling makes it pass ``AllocateMemoryAddr``
        with no L0c overflow — and the executed ``torch_codegen`` reference
        matches ``torch.matmul``.  ChooseL0Tile picks m = 192, n = 160, so the
        output tiles into a 2×2 grid with partial boundary sub-tiles."""

        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415
        from pypto.jit.decorator import jit  # noqa: PLC0415

        # Override the autouse Ascend950 fixture: 256×256 FP32 fits L0c on 950
        # but overflows it on 910B, which is the configuration that forces M/N
        # tiling here (and matches the solver's per-tile-kernel target backend).
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @jit
        def kernel(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                ta = pl.load(a, [0, 0], [256, 256], target_memory=pl.MemorySpace.Mat)
                tb = pl.load(b, [0, 0], [256, 256], target_memory=pl.MemorySpace.Mat)
                tc = pl.matmul(ta, tb)
                pl.store(tc, [0, 0], c)
            return c

        torch.manual_seed(0)
        a = torch.randn(256, 256, dtype=torch.float32)
        b = torch.randn(256, 256, dtype=torch.float32)
        c = torch.zeros(256, 256, dtype=torch.float32)

        # lower runs the full pipeline; AllocateMemoryAddr would
        # raise on an L0c overflow if the output were not tiled.
        post = kernel.lower(a, b, c)
        code = torch_codegen(post)
        ns: dict = {}
        exec(code, ns)  # noqa: S102 — executing generated reference code is the point

        out = c.clone()
        ns["kernel"](a, b, out)
        expected = torch.matmul(a, b)
        assert torch.allclose(out, expected, rtol=1e-3, atol=1e-3), (
            f"end-to-end M/N-tiled matmul mismatch: max abs diff {(out - expected).abs().max().item():.3e}"
        )

    def test_mn_tiling_reversed_def_store_chain_stays_ssa(self):
        """Two oversized matmuls whose **definitions are in the reverse order of
        their chained stores** must still produce valid SSA.

        Ordering (all valid SSA — each matmul precedes its store): ``c2`` is
        defined first, then ``c1``; the stores chain ``out → out1`` (via ``c1``)
        → ``out2`` (via ``c2``).  Each fold is built when its matmul is visited,
        but the folded stores are only *emitted* at the consumer-store site —
        with the now-current remap applied.  So ``c2``'s fold (built before
        ``c1``'s fold redefined ``out1``) chains from ``c1``'s fold output, not a
        stale/dangling ``out1``.  Regression for that bug: assert ``SSAForm`` +
        ``UseAfterDef`` hold after the pass and the result is numerically
        correct.  Each store writes a disjoint half of the [512, 1024] output."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a1: pl.Tensor[[512, 512], pl.FP32],
                b1: pl.Tensor[[512, 512], pl.FP32],
                a2: pl.Tensor[[512, 512], pl.FP32],
                b2: pl.Tensor[[512, 512], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 1024], pl.FP32]],
            ) -> pl.Tensor[[512, 1024], pl.FP32]:
                a2m: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    a2, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                b2m: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    b2, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                c2: pl.Tile[[512, 512], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a2m, b2m)
                a1m: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    a1, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                b1m: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    b1, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                c1: pl.Tile[[512, 512], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a1m, b1m)
                out1: pl.Tensor[[512, 1024], pl.FP32] = pl.store(c1, [0, 0], out)
                out2: pl.Tensor[[512, 1024], pl.FP32] = pl.store(c2, [0, 512], out1)
                return out2

        After = passes.auto_tile_matmul_l0()(Before)

        # SSA invariants must hold — the pass declares it preserves SSAForm.
        # A stale `out1` reference (the bug) is a use-before-def and fails here.
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.SSAForm)
        props.insert(passes.IRProperty.UseAfterDef)
        passes.verify_properties(props, After, "test_reversed_def_store_chain")

        # Numerically: out[:, 0:512] = a1 @ b1, out[:, 512:1024] = a2 @ b2.
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        torch.manual_seed(0)
        a1 = torch.randn(512, 512, dtype=torch.float32)
        b1 = torch.randn(512, 512, dtype=torch.float32)
        a2 = torch.randn(512, 512, dtype=torch.float32)
        b2 = torch.randn(512, 512, dtype=torch.float32)
        out = torch.zeros(512, 1024, dtype=torch.float32)

        code = torch_codegen(After)
        ns: dict = {}
        exec(code, ns)  # noqa: S102 — executing generated reference code is the point
        ns["kernel"](a1, b1, a2, b2, out)

        expected = torch.zeros(512, 1024, dtype=torch.float32)
        expected[:, 0:512] = torch.matmul(a1, b1)
        expected[:, 512:1024] = torch.matmul(a2, b2)
        assert torch.allclose(out, expected, rtol=1e-3, atol=1e-3), (
            f"reversed def/store-chain mismatch: max abs diff {(out - expected).abs().max().item():.3e}"
        )

    def test_mn_tiling_full_k_row_outer_pipelined(self):
        """Full-K (k == K) M/N tiling emits **nested pipelined loops** — outer rows,
        inner cols, both ``ForKind::Pipeline`` stage=2 — so ``LowerPipelineLoops``
        double-buffers both operand extracts (the pto-isa cost model's ~15% win).

        384×640 @ 64 BF16 on Ascend910: the roofline chooser picks (m=192, n=160,
        k=64), a divisible 2×4 grid (output [384,640] FP32 overflows L0c). The left
        panel (192×64) is not smaller than the right (64×160), so A is stationary
        and the M-row loop is the outer one."""

        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[384, 64], pl.BF16],
                rhs: pl.Tensor[[64, 640], pl.BF16],
                out: pl.Out[pl.Tensor[[384, 640], pl.FP32]],
            ) -> pl.Tensor[[384, 640], pl.FP32]:
                lhs_mat: pl.Tile[[384, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [384, 64], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[64, 640], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [64, 640], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[384, 640], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        _assert_ssa_valid(After, "test_full_k_row_outer")
        _assert_pipelined_full_k(After, n_pipeline_levels=2)
        # The traversal-cost rule keeps A stationary (rows outer) here: the chosen
        # tile makes row traversal no more expensive than column.
        assert _full_k_stationary_operand(After) == "A", "expected A-stationary (row-outer) traversal"

        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        torch.manual_seed(0)
        a = torch.randn(384, 64, dtype=torch.bfloat16)
        b = torch.randn(64, 640, dtype=torch.bfloat16)
        out = torch.zeros(384, 640, dtype=torch.float32)
        ns: dict = {}
        exec(torch_codegen(After), ns)  # noqa: S102
        ns["kernel"](a, b, out)
        expected = torch.matmul(a, b).float()
        assert torch.allclose(out, expected, rtol=1e-2, atol=1e-2), (
            f"full-K pipelined mismatch: max abs diff {(out - expected).abs().max().item():.3e}"
        )

    def test_mn_tiling_full_k_column_outer_pipelined(self):
        """When the right panel is the larger operand, B is stationary and the
        N-col loop is the **outer** one (the column-stationary mirror of the row
        case).  Same nested-pipelined-loop structure; the outer loop iterates over
        N, the inner over M.

        384×256 @ 64 BF16 on Ascend910."""

        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[384, 64], pl.BF16],
                rhs: pl.Tensor[[64, 256], pl.BF16],
                out: pl.Out[pl.Tensor[[384, 256], pl.FP32]],
            ) -> pl.Tensor[[384, 256], pl.FP32]:
                lhs_mat: pl.Tile[[384, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [384, 64], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[64, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [64, 256], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[384, 256], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        _assert_ssa_valid(After, "test_full_k_column_outer")
        _assert_pipelined_full_k(After, n_pipeline_levels=2)
        # Here the right panel is the more expensive one and the grid makes column
        # traversal cheaper, so the cost rule keeps B stationary (cols outer).
        assert _full_k_stationary_operand(After) == "B", "expected B-stationary (column-outer) traversal"

        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        torch.manual_seed(0)
        a = torch.randn(384, 64, dtype=torch.bfloat16)
        b = torch.randn(64, 256, dtype=torch.bfloat16)
        out = torch.zeros(384, 256, dtype=torch.float32)
        ns: dict = {}
        exec(torch_codegen(After), ns)  # noqa: S102
        ns["kernel"](a, b, out)
        expected = torch.matmul(a, b).float()
        assert torch.allclose(out, expected, rtol=1e-2, atol=1e-2), (
            f"full-K column-outer mismatch: max abs diff {(out - expected).abs().max().item():.3e}"
        )

    def test_full_k_os_hoist_obeys_scored_bandwidth_weighted_choice(self):
        """The full-K OS emit must hoist the SAME operand the chooser scored the
        wall under — a bandwidth-weighted (not raw-byte) decision.

        384×512 @ 64 BF16 on Ascend910 → output-stationary full-K tile
        (m = 128, n = 256), a 3×2 grid. L0A is faster than L0B (~200 vs ~132
        B/cyc), so streaming A on the fast port while holding B is cheaper than the
        reverse: the chooser's min-hoist load scores **hold B**, recorded in
        ``os_holds_a = False``. The emit therefore hoists B (column-outer).

        This is a regression pin for the chooser/emit hoist-objective unification:
        the previous emit re-derived the hoist from raw interior-extract bytes,
        which can disagree with the bandwidth-weighted min-hoist the wall was scored
        under (``estimated_cost_cycles``). The single-source ``os_holds_a`` makes the
        emitted hoist match the scored hoist by construction, so this asserts **B**.

        (The original byte-*tie* square case — 320×320@64 → 160×160 — is no longer
        reachable: n=160 has odd(ceil(160/8))=odd(20)=5, so the FIXPIPE
        misalignment penalty now prices that tile drain-bound and the chooser
        avoids it. This aligned-N asymmetric shape exercises the same hoist path.)"""

        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[384, 64], pl.BF16],
                rhs: pl.Tensor[[64, 512], pl.BF16],
                out: pl.Out[pl.Tensor[[384, 512], pl.FP32]],
            ) -> pl.Tensor[[384, 512], pl.FP32]:
                lhs_mat: pl.Tile[[384, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [384, 64], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [64, 512], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[384, 512], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        _assert_ssa_valid(After, "test_full_k_os_hoist")
        _assert_pipelined_full_k(After, n_pipeline_levels=2)
        # Byte traffic ties on the square tile; the bandwidth-weighted scored hoist
        # is B, so the emit must hoist B (column-outer). Pre-fix (byte heuristic)
        # this was A — the assertion that pins the fix.
        assert _full_k_stationary_operand(After) == "B", (
            "OS full-K emit must obey the scored bandwidth-weighted hoist (hold B), "
            "not the raw-byte heuristic"
        )

        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        torch.manual_seed(0)
        a = torch.randn(384, 64, dtype=torch.bfloat16)
        b = torch.randn(64, 512, dtype=torch.bfloat16)
        out = torch.zeros(384, 512, dtype=torch.float32)
        ns: dict = {}
        exec(torch_codegen(After), ns)  # noqa: S102
        ns["kernel"](a, b, out)
        expected = torch.matmul(a, b).float()
        assert torch.allclose(out, expected, rtol=1e-2, atol=1e-2), (
            f"OS hoist full-K mismatch: max abs diff {(out - expected).abs().max().item():.3e}"
        )

    def test_full_k_partial_boundary_is_peeled_into_tail(self):
        """When the chosen tile does not divide M/N, the full-K emitter pipelines
        the ``[0,full_m)×[0,full_n)`` interior (full m×n blocks) and peels the
        partial boundary into straight-line tail tiles — instead of forcing a tiny
        exact-divisor tile.  272×416 @ 32 (output-stationary): the roofline chooser
        picks (m=144, n=208, k=32) → a 1×2 full-tile interior plus an M-tail strip
        ``[144:272)×[0:416)``, every tile numerically exact with no collapse to a
        tiny divisor.  (A deliberately output-stationary shape, so this exercises
        the OS nested-pipeline peel; A/B-stationary peeling is covered separately
        in ``test_a_stationary_*``.)"""

        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[272, 32], pl.BF16],
                rhs: pl.Tensor[[32, 416], pl.BF16],
                out: pl.Out[pl.Tensor[[272, 416], pl.FP32]],
            ) -> pl.Tensor[[272, 416], pl.FP32]:
                lhs_mat: pl.Tile[[272, 32], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [272, 32], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[32, 416], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [32, 416], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[272, 416], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        _assert_ssa_valid(After, "test_full_k_tail")
        printed = ir.python_print(After)
        # Interior pipelines (2 levels); the boundary is peeled into extra tiles.
        assert printed.count("pl.pipeline(") == 2, "the interior must pipeline"
        assert printed.count("pl.tile.matmul(") > 1, "the partial boundary must be peeled into tail tiles"
        # No exact-divisor collapse: every interior tile step is large (≥ 64), not 16.
        import re  # noqa: PLC0415

        steps = [int(s) for s in re.findall(r"pl\.pipeline\(0, \d+, (\d+)", printed)]
        assert steps and all(s >= 64 for s in steps), f"tile collapsed to a tiny divisor: steps={steps}"

    def test_a_stationary_single_buffers_held_operand(self):
        """A-stationary (chooser picks it for k == K when pinning A cuts load): the
        held operand A occupies the FULL L0A (single-buffered) across the moving N
        grid; B streams double-buffered. The emitter realizes it as a **Sequential**
        outer (M) loop carrying A's extract + a **pipelined** inner (N) loop — one
        pipeline, not the two nested pipelines of the output-stationary path.

        256×544 @ 128 → A-stationary (m=256, n=128, k=128) under the per-M-row drain
        cost model: A = [256, 128] = 64 KB fits L0A single-buffered (= 64 KB) but
        would overflow double-buffered, so the single-buffered Sequential outer is
        what makes the tile legal. 544 = 4*128 + 32, so the inner pipeline runs the 4
        full 128-wide blocks and a straight-line 32-wide N-peel follows. The full
        Default pipeline must allocate cleanly. (Numerics: st suite.)"""
        from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[256, 128], pl.BF16],
                rhs: pl.Tensor[[128, 544], pl.BF16],
                out: pl.Out[pl.Tensor[[256, 544], pl.FP32]],
            ) -> pl.Tensor[[256, 544], pl.FP32]:
                lhs_mat: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [256, 128], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[128, 544], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [128, 544], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[256, 544], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[256, 128], pl.BF16],
                rhs: pl.Tensor[[128, 544], pl.BF16],
                out: pl.Out[pl.Tensor[[256, 544], pl.FP32]],
            ) -> pl.Tensor[[256, 544], pl.FP32]:
                lhs_mat: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [256, 128], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[128, 544], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [128, 544], target_memory=pl.Mem.Mat
                )
                # Sequential outer (M) loop holds the single-buffered A panel (full L0A).
                for mo, (out_o,) in pl.range(0, 256, 256, init_values=(out,)):
                    a_held: pl.Tile[[256, 128], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, mo, 0, [256, 128], target_memory=pl.Mem.Left
                    )
                    # Pipelined inner (N) loop over the 4 full 128-wide blocks; B double-buffered.
                    for ni, (out_i,) in pl.pipeline(
                        0, 512, 128, stage=2, init_values=(out_o,), attrs={"pipeline_overlap_stores": False}
                    ):
                        b_mov: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                            rhs_mat, 0, ni, [128, 128], target_memory=pl.Mem.Right
                        )
                        c_sub: pl.Tile[[256, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a_held, b_mov)
                        out_s: pl.Tensor[[256, 544], pl.FP32] = pl.store(c_sub, [mo, ni], out_i)
                        out_iy = pl.yield_(out_s)
                    out_oy = pl.yield_(out_iy)
                # N-boundary peel: the last 32-wide block (544 = 4*128 + 32), straight-line.
                a_peel: pl.Tile[[256, 128], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 0, 0, [256, 128], target_memory=pl.Mem.Left
                )
                b_peel: pl.Tile[[128, 32], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 512, [128, 32], target_memory=pl.Mem.Right
                )
                c_peel: pl.Tile[[256, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a_peel, b_peel)
                out_peel: pl.Tensor[[256, 544], pl.FP32] = pl.store(c_peel, [0, 512], out_oy)
                return out_peel

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)
        # The chooser's single-buffered A tile must also allocate without an L0A
        # overflow through the full pipeline (A = 64 KB single-buffered; double-
        # buffering it, 128 KB, would exceed the 64 KB L0A).
        assert PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before) is not None

        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        torch.manual_seed(0)
        a = torch.randn(256, 128, dtype=torch.bfloat16)
        b = torch.randn(128, 544, dtype=torch.bfloat16)
        out = torch.zeros(256, 544, dtype=torch.float32)
        ns: dict = {}
        exec(torch_codegen(After), ns)  # noqa: S102
        ns["kernel"](a, b, out)
        expected = torch.matmul(a, b).float()
        assert torch.allclose(out, expected, rtol=1e-2, atol=1e-2), (
            f"A-stationary numerics mismatch: max abs diff {(out - expected).abs().max().item():.3e}"
        )

    def test_b_stationary_single_buffers_held_operand(self):
        """B-stationary mirror: the held operand B occupies the FULL L0B
        (single-buffered) across the moving M grid; A streams double-buffered. The
        held B is the outer (Sequential) panel and A the moving (pipelined) inner
        panel — the loop order flips vs A-stationary, the single-buffering does not.

        192×512 @ 64 → B-stationary (m=64, n=512, k=64) under the drain-count model
        (#1912): B = [64, 512] = 64 KB held in full L0B single-buffered (double would
        overflow), A = [64, 64] streamed across the 3 clean m-blocks. (256×272 no
        longer selects B-stationary under the drain-count model — B-stat splits the
        output over M, so on that small shape output-stationary has fewer drains.)"""
        from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[192, 64], pl.BF16],
                rhs: pl.Tensor[[64, 512], pl.BF16],
                out: pl.Out[pl.Tensor[[192, 512], pl.FP32]],
            ) -> pl.Tensor[[192, 512], pl.FP32]:
                lhs_mat: pl.Tile[[192, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [192, 64], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [64, 512], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[192, 512], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[192, 64], pl.BF16],
                rhs: pl.Tensor[[64, 512], pl.BF16],
                out: pl.Out[pl.Tensor[[192, 512], pl.FP32]],
            ) -> pl.Tensor[[192, 512], pl.FP32]:
                lhs_mat: pl.Tile[[192, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [192, 64], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[64, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [64, 512], target_memory=pl.Mem.Mat
                )
                # Sequential outer (N) loop holds the single-buffered B panel (full L0B).
                for no, (out_o,) in pl.range(0, 512, 512, init_values=(out,)):
                    b_held: pl.Tile[[64, 512], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, 0, no, [64, 512], target_memory=pl.Mem.Right
                    )
                    # Pipelined inner (M) loop streams A double-buffered over 3 m-blocks.
                    for mi, (out_i,) in pl.pipeline(
                        0, 192, 64, stage=2, init_values=(out_o,), attrs={"pipeline_overlap_stores": False}
                    ):
                        a_mov: pl.Tile[[64, 64], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                            lhs_mat, mi, 0, [64, 64], target_memory=pl.Mem.Left
                        )
                        c_sub: pl.Tile[[64, 512], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a_mov, b_held)
                        out_s: pl.Tensor[[192, 512], pl.FP32] = pl.store(c_sub, [mi, no], out_i)
                        out_iy = pl.yield_(out_s)
                    out_oy = pl.yield_(out_iy)
                return out_oy

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)
        assert PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before) is not None

        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        torch.manual_seed(0)
        a = torch.randn(192, 64, dtype=torch.bfloat16)
        b = torch.randn(64, 512, dtype=torch.bfloat16)
        out = torch.zeros(192, 512, dtype=torch.float32)
        ns: dict = {}
        exec(torch_codegen(After), ns)  # noqa: S102
        ns["kernel"](a, b, out)
        expected = torch.matmul(a, b).float()
        assert torch.allclose(out, expected, rtol=1e-2, atol=1e-2), (
            f"B-stationary numerics mismatch: max abs diff {(out - expected).abs().max().item():.3e}"
        )

    def test_multiple_geometries_reuse_full_l0b_panel_through_default_pypto(self):
        """#2633: address-level reuse keeps A5's chosen 64 KiB dense panel.

        The tail matmul creates two co-live 32 KiB Right panels. PYPTO's
        overflow fallback places them in the two halves of the expired dense
        panel instead of forcing AutoTile to re-choose the dense schedule.
        """
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend950)
        from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[64, 128], pl.INT8],
                tail: pl.Tensor[[16, 128], pl.INT8],
                rhs: pl.Tensor[[128, 512], pl.INT8],
                out: pl.Out[pl.Tensor[[64, 512], pl.INT32]],
                out_tail: pl.Out[pl.Tensor[[16, 512], pl.INT32]],
            ) -> tuple[pl.Tensor[[64, 512], pl.INT32], pl.Tensor[[16, 512], pl.INT32]]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [128, 512], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                tail_mat = pl.tile.load(tail, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
                c_tail = pl.tile.matmul(tail_mat, rhs_mat)
                out_tail = pl.store(c_tail, [0, 0], out_tail)
                return out, out_tail

        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
            tiled = passes.auto_tile_matmul_l0()(Before)
            lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before)

        # Only the tail uses the k=64 output-stationary loop. The dense call
        # remains the chooser's full-k B-stationary design point.
        assert ir.python_print(tiled).count("pl.pipeline(0, 128, 64,") == 1
        right_allocs = [
            int(size)
            for size in re.findall(r"pl\.tile\.alloc\(pl\.Mem\.Right, (\d+)\)", ir.python_print(lowered))
        ]
        assert right_allocs == [64 * 1024]

    @pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
    @pytest.mark.parametrize(
        ("M", "K", "N", "tile_k"),
        [
            (16, 128, 128, 64),
            (64, 192, 128, 64),
            (64, 256, 256, 32),
            (128, 384, 64, 64),
        ],
    )
    def test_system_shapes_emit_expected_reduction_route(self, planner, M, K, N, tile_k):
        """Structural contract for the FP32 system-test matrix.

        In-tree planners retain single-C K-only schedules. PTOAS also retains
        its historical full-K A-stationary dbC choice for 64x256x256.
        """
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.FP32],
                rhs: pl.Tensor[[K, N], pl.FP32],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                lhs_mat: pl.Tile[[M, K], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[K, N], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[M, N], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        with passes.PassContext([], memory_planner=planner):
            After = passes.auto_tile_matmul_l0()(Before)
        printed = ir.python_print(After)
        if planner == passes.MemoryPlanner.PTOAS and (M, K, N) == (64, 256, 256):
            assert '"pipeline_double_buffer_c": True' in printed
            assert "pl.range(0, 64, 64," in printed
            assert "pl.pipeline(0, 256, 32," in printed
            assert "[64, 256]" in printed
            assert "pl.tile.matmul_acc(" not in printed
            _assert_ssa_valid(After, "ptoas_full_k_stationary")
            return
        assert "pipeline_double_buffer_c" not in printed
        assert printed.count("pl.pipeline(") == 1
        assert "pl.range(" not in printed
        assert f"pl.pipeline(0, {K}, {tile_k}," in printed
        assert "pl.tile.matmul_acc(" in printed
        _assert_ssa_valid(After, f"test_system_k_split_{planner}_{M}_{K}_{N}")

    @pytest.mark.parametrize(
        ("planner", "M", "K", "N", "held_m", "outer_loop", "inner_loop", "double_buffer_c"),
        [
            (
                passes.MemoryPlanner.PYPTO,
                256,
                128,
                544,
                256,
                "pl.range(0, 256, 256,",
                "pl.pipeline(0, 512, 128,",
                False,
            ),
            (
                passes.MemoryPlanner.PTOAS,
                48,
                352,
                96,
                48,
                "pl.range(0, 48, 48,",
                "pl.pipeline(0, 96, 32,",
                True,
            ),
        ],
    )
    def test_system_a_stationary_shapes_emit_held_a(
        self, planner, M, K, N, held_m, outer_loop, inner_loop, double_buffer_c
    ):
        """Planner-specific A-stationary shapes reuse held A across the moving loop."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.BF16],
                rhs: pl.Tensor[[K, N], pl.BF16],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                lhs_mat: pl.Tile[[M, K], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[K, N], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[M, N], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        with passes.PassContext([], memory_planner=planner):
            After = passes.auto_tile_matmul_l0()(Before)
        printed = ir.python_print(After)
        assert printed.count("pl.range(") == 1
        assert printed.count("pl.pipeline(") == 1
        assert outer_loop in printed
        assert inner_loop in printed
        lines = printed.splitlines()
        outer_i = next(i for i, line in enumerate(lines) if outer_loop in line)
        inner_i = next(i for i, line in enumerate(lines) if inner_loop in line)
        assert outer_i < inner_i
        held_region = "\n".join(lines[outer_i + 1 : inner_i])
        assert "pl.tile.extract(" in held_region
        assert f"[{held_m}, {K}]" in held_region
        assert "target_memory=pl.Mem.Left" in held_region
        assert ("pipeline_double_buffer_c" in printed) == double_buffer_c
        _assert_ssa_valid(After, f"test_system_a_stationary_{planner}")

    @pytest.mark.parametrize(
        ("planner", "M", "K", "N", "held_n", "outer_loop", "inner_loop", "double_buffer_c"),
        [
            (
                passes.MemoryPlanner.PYPTO,
                192,
                64,
                512,
                512,
                "pl.range(0, 512, 512,",
                "pl.pipeline(0, 192, 64,",
                False,
            ),
            (
                passes.MemoryPlanner.PTOAS,
                64,
                80,
                256,
                256,
                "pl.range(0, 256, 256,",
                "pl.pipeline(0, 64, 32,",
                True,
            ),
        ],
    )
    def test_system_b_stationary_shapes_emit_held_b(
        self, planner, M, K, N, held_n, outer_loop, inner_loop, double_buffer_c
    ):
        """Planner-specific B-stationary system shapes keep B in the outer loop."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.BF16],
                rhs: pl.Tensor[[K, N], pl.BF16],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                lhs_mat: pl.Tile[[M, K], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[K, N], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[M, N], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        with passes.PassContext([], memory_planner=planner):
            After = passes.auto_tile_matmul_l0()(Before)
        printed = ir.python_print(After)
        assert printed.count("pl.range(") == 1
        assert printed.count("pl.pipeline(") == 1
        assert outer_loop in printed
        assert inner_loop in printed
        lines = printed.splitlines()
        outer_i = next(i for i, line in enumerate(lines) if outer_loop in line)
        inner_i = next(i for i, line in enumerate(lines) if inner_loop in line)
        assert outer_i < inner_i
        held_region = "\n".join(lines[outer_i + 1 : inner_i])
        assert "pl.tile.extract(" in held_region
        assert f"[{K}, {held_n}]" in held_region
        assert "target_memory=pl.Mem.Right" in held_region
        assert ("pipeline_double_buffer_c" in printed) == double_buffer_c
        _assert_ssa_valid(After, f"test_system_b_stationary_{planner}")

    @pytest.mark.parametrize(
        ("planner", "pypto_dbc"),
        [
            (passes.MemoryPlanner.PYPTO, True),
            (passes.MemoryPlanner.DSA_RP, False),
            (passes.MemoryPlanner.PTOAS, False),
        ],
    )
    @pytest.mark.parametrize(
        ("M", "N"),
        [
            (160, 160),
            (144, 144),
            (256, 256),
            (448, 448),
            (384, 256),
        ],
    )
    def test_system_shapes_emit_planner_selected_tile(self, planner, pypto_dbc, M, N):
        """PTOAS retains dbC; the in-tree planners search the full single-C budget."""
        K = 64
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.FP32],
                rhs: pl.Tensor[[K, N], pl.FP32],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                lhs_mat: pl.Tile[[M, K], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[K, N], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[M, N], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        with passes.PassContext(
            [],
            memory_planner=planner,
            enable_pypto_l0c_double_buffer=pypto_dbc,
        ):
            After = passes.auto_tile_matmul_l0()(Before)
        printed = ir.python_print(After)
        cfg = passes.l0_tile_chooser.L0TileConfig()
        cfg.M, cfg.N, cfg.K = M, N, K
        cfg.bytes_a = cfg.bytes_b = cfg.bytes_c = 4
        cfg.l0a_bytes = cfg.l0b_bytes = 65536
        cfg.l0c_bytes = 131072
        cfg.allow_a_stationary = cfg.allow_b_stationary = True
        cfg.allow_double_buffer_c = planner == passes.MemoryPlanner.PTOAS
        cfg.allow_k_boundary = True
        choice = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert f"[{choice.m}, {choice.k}], target_memory=pl.Mem.Left" in printed
        assert f"[{choice.k}, {choice.n}], target_memory=pl.Mem.Right" in printed
        assert ("pipeline_double_buffer_c" in printed) == choice.double_buffer_c
        assert "pipeline_membership" not in printed
        assert "slots=2" not in printed
        _assert_ssa_valid(After, f"test_system_single_c_{planner}_{M}_{N}")

    def test_full_k_direct_gm_keeps_one_l0c_accumulator(self):
        """Full-K direct-GM tiling keeps **one** L0C accumulator through the whole
        pipeline.  The stage-2 inner loop sets ``overlap_stores=false`` so
        ``CanonicalizeIOOrder`` schedules each store adjacent to its matmul
        (``matmul_i, store_i, matmul_{i+1}, …``) instead of floating both stores
        below both matmuls — the latter co-lives two ``[m,n]`` results
        (``2·m·n·bytes_c``) while the chooser budgets a single L0C buffer
        (``double_buffer_c=false``), overflowing allocation.

        320×320 @ 64 BF16: chooser picks m=n=160 → C tile = 160·160·4 = 100 KB.
        One accumulator (100 KB) fits L0C (128 KB); two (200 KB) would not.  The
        full Default pipeline raised an Acc-overflow before the one-accumulator
        schedule; here it must allocate cleanly, and the moving-operand extract
        stays double-buffered (hoisted Load tier)."""

        from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[320, 64], pl.BF16],
                rhs: pl.Tensor[[64, 320], pl.BF16],
                out: pl.Out[pl.Tensor[[320, 320], pl.FP32]],
            ) -> pl.Tensor[[320, 320], pl.FP32]:
                lhs_mat: pl.Tile[[320, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [320, 64], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[64, 320], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [64, 320], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[320, 320], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        # The full Default pipeline must allocate without an L0C (Acc) overflow.
        # (Pre-fix this raised "Acc buffer usage (204800 bytes) exceeds platform
        # limit (131072 bytes)" — 2× the 100 KB C tile.)
        result = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before)
        assert result is not None

        # Structural check on the one-accumulator schedule: after pipeline lowering
        # + IO canonicalization the operand extracts hoist (double-buffered) and
        # each store interleaves with its matmul (one C), not floating to the end.
        prog = passes.auto_tile_matmul_l0()(Before)
        prog = passes.infer_tile_memory_space()(prog)
        prog = passes.lower_pipeline_loops()(prog)
        prog = passes.canonicalize_io_order()(prog)
        seq = []
        for line in ir.python_print(prog).splitlines():
            s = line.strip()
            if ".extract(" in s and ("Left" in s or "Right" in s):
                seq.append("extract")
            elif "matmul" in s and "=" in s:
                seq.append("matmul")
            elif ".store(" in s and "=" in s:
                seq.append("store")
        # Every matmul is immediately followed by its store (interleaved one-C),
        # and no two matmuls are adjacent (which would mean two live accumulators).
        matmul_idxs = [i for i, op in enumerate(seq) if op == "matmul"]
        assert matmul_idxs, "expected matmuls in the lowered full-K schedule"
        for i in matmul_idxs:
            assert i + 1 < len(seq) and seq[i + 1] == "store", (
                f"matmul at {i} not immediately followed by its store (two-accumulator schedule): {seq}"
            )

    @pytest.mark.parametrize(
        ("M", "N"),
        [
            (320, 320),  # clean divisible interior, no tail
            (272, 272),  # 272 = 16·17 → 1×1 interior + partial-boundary tail tiles
        ],
    )
    def test_full_k_direct_gm_generates_valid_pto(self, M: int, N: int):
        """Direct-store full-K tiling lowers to valid PTO MLIR (PTOCodegen
        succeeds) — for both a clean divisible grid and a shape whose partial
        boundary is peeled into a straight-line tail (the tail reuses the same
        extract/matmul/store primitives, so it must also codegen)."""
        from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415
        from pypto.pypto_core import codegen as _codegen_core  # noqa: PLC0415
        from pypto.pypto_core import ir as _ir_core  # noqa: PLC0415

        K = 64
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.BF16],
                rhs: pl.Tensor[[K, N], pl.BF16],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                lhs_mat: pl.Tile[[M, K], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[K, N], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[M, N], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        prog = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before)
        generated = False
        for _name, func in prog.functions.items():
            mlir = _codegen_core.PTOCodegen().generate(_ir_core.Program([func], func.name, prog.span))
            if "pto." in mlir or "func.func" in mlir:
                generated = True
        assert generated, "direct-store full-K must generate valid PTO MLIR"


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

        After = passes.auto_tile_matmul_l0()(Before)
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

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)

    def test_sub_byte_dtype_skipped(self):
        """An INT4 (sub-byte) operand makes ``DTypeBytes`` return 0, so the
        pass emits ``PH-AT-003`` and leaves the matmul untouched (pass lines
        448-453).  INT4 @ INT4 deduces an INT32 accumulator, so the matmul is
        well-typed and Mat-resident — the skip is purely the sub-byte guard,
        not a residency/shape filter.  The shape (16×64 @ 2048) would otherwise
        be K-tiled, proving the sub-byte branch is what blocks the rewrite."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.INT4],
                rhs: pl.Tensor[[2048, 64], pl.INT4],
                out: pl.Out[pl.Tensor[[16, 64], pl.INT32]],
            ) -> pl.Tensor[[16, 64], pl.INT32]:
                lhs_mat: pl.Tile[[16, 2048], pl.INT4, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[2048, 64], pl.INT4, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)

    def test_chooser_rejected_config_skipped(self):
        """A K dimension below the cube minimum (K=8 < min_k=16) makes
        ``ChooseL0Tile`` throw ``pypto::ValueError`` (chooser line 192,
        ``allow_padding=false``).  The pass catches it, emits ``PH-AT-005``,
        and leaves the matmul untouched (pass lines 492-500)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 8], pl.BF16],
                rhs: pl.Tensor[[8, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 8], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 8], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[8, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [8, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)

    def test_oversized_matmul_acc_mn_deferred(self):
        """An arbitrary oversized ``tile.matmul_acc`` output (512×512 FP32 on
        Ascend950, 1 MB > L0c) would require slices of its caller-owned [M, N]
        accumulator, so it remains deferred with ``PH-AT-006``. The supported
        split-K case is instead matched as a canonical create/pipeline/store
        chain and tiled outside the K loop."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[512, 512], pl.FP32],
                rhs: pl.Tensor[[512, 512], pl.FP32],
                acc_init: pl.Tile[[512, 512], pl.FP32, pl.Mem.Acc],
                out: pl.Out[pl.Tensor[[512, 512], pl.FP32]],
            ) -> pl.Tensor[[512, 512], pl.FP32]:
                lhs_mat: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[512, 512], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(acc_init, lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)

    def test_oversized_predicated_matmul_acc_mn_still_deferred(self):
        """A predicate does not unlock M/N tiling for ``tile.matmul_acc``.

        ``TryFoldMNTiling`` / ``TryFoldMatScratch`` refuse every accumulate
        tiling, so ``BuildFullKPipelined`` and ``BuildSplitKGrid`` — neither of
        which threads a predicate — stay unreachable for the accumulate kind.
        If a refactor ever lets a predicated call reach them, the predicate
        would be dropped silently; this fails loudly instead.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[512, 512], pl.FP32],
                rhs: pl.Tensor[[512, 512], pl.FP32],
                acc_init: pl.Tile[[512, 512], pl.FP32, pl.Mem.Acc],
                first_k: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[512, 512], pl.FP32]],
            ) -> pl.Tensor[[512, 512], pl.FP32]:
                lhs_mat: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[512, 512], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                    acc_init, lhs_mat, rhs_mat, init_cond=(first_k == 0)
                )
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)

    def test_oversized_matmul_no_store_consumer_untouched(self):
        """An oversized plain ``tile.matmul`` whose result is *not* consumed by a
        2D ``tile.store`` cannot use the direct-store M/N fold.  Here the [512,
        512] Acc result feeds a ``tile.move`` (Acc→Vec) before any store, so the
        pass emits ``PH-AT-006`` and leaves the matmul untouched — the
        Mat-scratch / assemble path for on-chip consumers is deferred."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[512, 512], pl.FP32],
                rhs: pl.Tensor[[512, 512], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 512], pl.FP32]],
            ) -> pl.Tensor[[512, 512], pl.FP32]:
                lhs_mat: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[512, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [512, 512], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[512, 512], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                # Consumer is a tile.move (Acc→Vec), not a store → not foldable.
                c_vec: pl.Tile[[512, 512], pl.FP32, pl.Mem.Vec] = pl.tile.move(c, target_memory=pl.Mem.Vec)
                out = pl.store(c_vec, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)

    def test_matmul_bias_n_tiling_with_bias_resident_source_is_deferred(self):
        """The architectural bias table cannot form a Bias-to-Bias N sub-window."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[256, 64], pl.BF16],
                rhs: pl.Tensor[[64, 512], pl.BF16],
                bias: pl.Tensor[[1, 512], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 512], pl.FP32]],
            ) -> pl.Tensor[[256, 512], pl.FP32]:
                lhs_mat = pl.tile.load(lhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat)
                rhs_mat = pl.tile.load(rhs, [0, 0], [64, 512], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, 512], target_memory=pl.Mem.Mat)
                bias_l0 = pl.tile.move(bias_mat, target_memory=pl.Mem.Bias)
                c = pl.tile.matmul_bias(lhs_mat, rhs_mat, bias_l0)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)

    def test_matmul_bias_vec_left_is_out_of_scope(self):
        """The historical Vec-left K-only exception is not widened to biased matmul."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                bias: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_vec = pl.tile.load(lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Vec)
                rhs_mat = pl.tile.load(rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, 64], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_bias(lhs_vec, rhs_mat, bias_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)

    def test_non_incore_function_untouched(self):
        """The pass only walks InCore-typed functions
        (``TransformFunction`` guard, pass line 593 — ``IsInCoreType``).  An
        ``Opaque`` function carrying the *exact same* tile-able Mat matmul as
        the rewritten K-only cases is left untouched, while the InCore twin
        rewrites — isolating the function-type guard as the deciding factor."""

        @pl.program
        class OpaqueProg:
            @pl.function(type=pl.FunctionType.Opaque)
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

        # The Opaque function is left structurally identical.
        After = passes.auto_tile_matmul_l0()(OpaqueProg)
        ir.assert_structural_equal(After, OpaqueProg)

        # Twin: same body in an InCore function DOES rewrite — proves the
        # untouched-ness above is the function-type guard, not a different
        # filter.
        @pl.program
        class InCoreProg:
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

        incore_after = passes.auto_tile_matmul_l0()(InCoreProg)
        with pytest.raises(ValueError, match="Structural mismatch"):
            ir.assert_structural_equal(incore_after, InCoreProg)


class TestAutoTileMatmulL0MatScratch:
    """M/N output tiling to an L1/Mat scratch (on-chip matmul consumer), not DDR.

    When an oversized ``[M, N]`` matmul result is consumed *only* as a matmul operand
    (a chained matmul), the pass tiles the output into a ``tile.create(target=Mat)``
    scratch via per-sub-tile ``tile.assemble`` (Acc→Mat) and keeps it on-chip for the
    consumer, instead of the direct-GM store path. Split-K uses a constant-offset
    grid; full-K uses pipelined loop-variable offsets."""

    def test_matmul_bias_producer_uses_mat_scratch(self):
        """An oversized biased producer may stay on-chip for one later matmul."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend950)
        M, K, N, out_n = 256, 192, 512, 64

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, K], pl.BF16],
                b: pl.Tensor[[K, N], pl.BF16],
                bias: pl.Tensor[[1, N], pl.FP32],
                e: pl.Tensor[[N, out_n], pl.BF16],
                out: pl.Out[pl.Tensor[[M, out_n], pl.FP32]],
            ) -> pl.Tensor[[M, out_n], pl.FP32]:
                a_mat = pl.tile.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                b_mat = pl.tile.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, N], target_memory=pl.Mem.Mat)
                e_mat = pl.tile.load(e, [0, 0], [N, out_n], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_bias(a_mat, b_mat, bias_mat)
                cb = pl.cast(c, pl.BF16, mode="rint")
                d = pl.tile.matmul(cb, e_mat)
                out = pl.store(d, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        printed = ir.python_print(After)
        assert "tile.create" in printed and "Mem.Mat" in printed
        assert printed.count("pl.tile.assemble(") >= 2
        assert "pl.tile.cast(" not in printed
        assert "pl.tile.matmul_bias(" in printed and "pl.tile.matmul_acc(" in printed
        _assert_ssa_valid(After, "test_matmul_bias_producer_uses_mat_scratch")

        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        torch.manual_seed(1)
        a = torch.randn(M, K, dtype=torch.bfloat16)
        b = torch.randn(K, N, dtype=torch.bfloat16)
        bias = torch.randn(1, N)
        e = torch.randn(N, out_n, dtype=torch.bfloat16)
        out = torch.zeros(M, out_n)
        ns: dict = {}
        exec(torch_codegen(After), ns)  # noqa: S102 -- executing generated reference code is the point
        ns["kernel"](a, b, bias, e, out)
        intermediate = (a.float() @ b.float() + bias).to(torch.bfloat16).float()
        expected = intermediate @ e.float()
        rel_err = ((out - expected).norm() / expected.norm()).item()
        assert rel_err < 5e-2, f"matmul_bias Mat-scratch rel_err {rel_err:.3e} exceeds 5e-2"

    def test_matmul_bias_mat_scratch_load_removal_uses_forced_os_tile(self):
        """#1908 re-selection keeps the full bias load when forced OS only K-tiles.

        Standalone, this geometry selects A-stationary with N=128. The
        Mat-scratch path re-chooses output stationarity and keeps full N=192
        while splitting K. The emitted full-width Bias move still reads the
        original load, so load removal must follow that final OS choice rather
        than the discarded A-stationary candidate.
        """
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend950)
        M, K, N, out_n = 208, 128, 192, 64

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, K], pl.BF16],
                b: pl.Tensor[[K, N], pl.BF16],
                bias: pl.Tensor[[1, N], pl.FP32],
                e: pl.Tensor[[N, out_n], pl.BF16],
                out: pl.Out[pl.Tensor[[M, out_n], pl.FP32]],
            ) -> pl.Tensor[[M, out_n], pl.FP32]:
                a_mat = pl.tile.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                b_mat = pl.tile.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                bias_mat = pl.tile.load(bias, [0, 0], [1, N], target_memory=pl.Mem.Mat)
                e_mat = pl.tile.load(e, [0, 0], [N, out_n], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_bias(a_mat, b_mat, bias_mat)
                cb = pl.cast(c, pl.BF16, mode="rint")
                d = pl.tile.matmul(cb, e_mat)
                out = pl.store(d, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        printed = ir.python_print(After)
        assert re.search(r"pl\.tile\.load\(\s*bias,\s*\[0, 0\],\s*\[1, 192\]", printed)
        assert printed.count("pl.tile.matmul_bias(") == 1
        assert "pl.tile.matmul_acc(" in printed
        assert "pl.tile.assemble(" in printed
        _assert_ssa_valid(After, "test_matmul_bias_mat_scratch_load_removal_uses_forced_os_tile")

    def test_chained_matmul_uses_mat_scratch(self):
        """An oversized producer feeding a matmul: the pass assembles the result into an
        L1/Mat scratch via per-sub-tile Acc→Mat assembles, and the consumer reads the
        scratch on-chip (no DDR — the L0C→L1→L0A trip).  256×256 @ 256 producer: under
        the drain-count cost model (#1912) the chooser picks (256, 128, 64)
        **output-stationary** split-K (wider m halves the drain count) → a 1×2 grid
        → 2 Acc→Mat assembles at constant offsets; the consumer is also output-stationary.

        The dims are chosen so BOTH matmuls are output-stationary: their L0 operand
        buffers are the same 32 KB shape, so the producer's (sequential, dead before the
        consumer) packs cleanly into the consumer's in the current MemoryReuse.  An
        A-stationary producer would instead pin a monolithic 64 KB L0 buffer that the
        consumer's 2×32 KB double-buffer cannot pack against until MemoryReuse learns to
        subdivide a freed region (the offset-packing follow-up).  Asserts structure + SSA
        + numerics, and that the full Default pipeline allocates without an L0 overflow."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[256, 256], pl.BF16],
                b: pl.Tensor[[256, 256], pl.BF16],
                e: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                c = pl.matmul(a, b, out_dtype=pl.FP32)  # [256, 256] f32 > L0c → on-chip consumer
                cb = pl.cast(c, pl.BF16, mode="rint")  # rint -> bf16 Mat scratch (cast fused)
                d = pl.matmul(cb, e, out_dtype=pl.FP32)  # consumes the scratch only as a matmul operand
                out = pl.assemble(out, d, [0, 0])
                return out

        After = passes.auto_tile_matmul_l0()(_lower_to_tile_ops(Before))
        printed = ir.python_print(After)

        assert "tile.create" in printed and "Mem.Mat" in printed, "expected a Mat output scratch"
        assert printed.count("pl.tile.assemble(") == 2, (
            "output-stationary split-K 1×2 grid (m=256, n=128) → 2 Acc→Mat assembles at constant offsets"
        )
        assert "pl.tile.matmul(a__ssa_v0_mat, b__ssa_v0_mat)" not in printed, (
            "the oversized producer must be tiled, not left whole"
        )
        assert "pl.tile.cast(" not in printed, "the downcast must be fused into the Mat scratch"
        _assert_ssa_valid(After, "test_mat_scratch_chained")

        # The chained producer must allocate without an L0 overflow.  These dims keep
        # both matmuls output-stationary, so their L0 operand buffers are the same 32 KB
        # shape and pack in MemoryReuse; a pass-level structural check alone would not
        # catch an overflow.
        from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

        assert PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before) is not None

        # Numerically correct vs the bf16 chain, executed through torch_codegen. The
        # reference does block-wise bf16 matmuls with the intermediate downcast to bf16
        # (the FIXPIPE writeback), so it carries real bf16 rounding; with random data,
        # near-zero cancellation elements make element-wise allclose hopeless. The
        # Frobenius relative error (dominated by the large entries) is the robust metric.
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        torch.manual_seed(0)
        a = torch.randn(256, 256, dtype=torch.bfloat16)
        b = torch.randn(256, 256, dtype=torch.bfloat16)
        e = torch.randn(256, 64, dtype=torch.bfloat16)
        out = torch.zeros(256, 64)
        ns: dict = {}
        exec(torch_codegen(After), ns)  # noqa: S102 — executing generated reference code is the point
        ns["kernel"](a, b, e, out)
        c_bf16 = (a.float() @ b.float()).to(torch.bfloat16).float()  # FIXPIPE downcast
        expected = c_bf16 @ e.float()
        rel_err = ((out - expected).norm() / expected.norm()).item()
        assert rel_err < 5e-2, f"split-K Mat-scratch chained bf16 rel_err {rel_err:.3e} exceeds 5e-2"

    @pytest.mark.parametrize(
        ("planner", "pypto_opt_in", "operand_stationary", "double_buffer_c"),
        [
            (passes.MemoryPlanner.PYPTO, False, False, False),
            (passes.MemoryPlanner.DSA_RP, False, True, False),
        ],
    )
    def test_chained_mat_scratch_stationarity_matches_planner(
        self, planner, pypto_opt_in, operand_stationary, double_buffer_c
    ):
        """Apply the #1908 guard only to the legacy PyPTO allocator.

        This chained Mat-scratch producer standalone selects B-stationary
        (128×512×128). PyPTO cannot subdivide its released monolithic L0B
        panel for the consumer's smaller pipelined buffers, so AutoTile
        re-chooses OS. DSA_RP places from actual lifetimes and retains the
        B-stationary choice.

        128×512 FP32 output (256 KB) > L0c so the producer is tiled; the 128×512 bf16
        Mat scratch (128 KB) fits Mat/L1, so it reaches the fold (not the capacity gate).
        The consumer [128, 64] fits L0c (K-only tiling), so any Sequential
        ``pl.range`` in the emitted kernel is the producer's A/B-stationary
        held-operand loop."""
        from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 128], pl.BF16],
                b: pl.Tensor[[128, 512], pl.BF16],
                e: pl.Tensor[[512, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                c = pl.matmul(a, b, out_dtype=pl.FP32)  # [128, 512] f32 > L0c → Mat scratch
                cb = pl.cast(c, pl.BF16, mode="rint")
                d = pl.matmul(cb, e, out_dtype=pl.FP32)  # consumes the scratch as a matmul operand
                out = pl.assemble(out, d, [0, 0])
                return out

        with passes.PassContext([], memory_planner=planner, enable_pypto_l0c_double_buffer=pypto_opt_in):
            After = passes.auto_tile_matmul_l0()(_lower_to_tile_ops(Before))
            allocated = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before)

        printed = ir.python_print(After)
        assert "tile.create" in printed and "Mem.Mat" in printed, "expected a Mat output scratch"
        assert ("pl.range(" in printed) == operand_stationary
        assert ("pipeline_double_buffer_c" in printed) == double_buffer_c
        _assert_ssa_valid(After, f"test_mat_scratch_stationarity_{planner}")

        if not operand_stationary:
            return
        allocated_text = ir.python_print(allocated)
        assert "tile.alloc(pl.Mem.Right, 65536)" in allocated_text
        right_ranges = {
            (int(offset), int(size))
            for offset, size in re.findall(
                r"pl\.MemRef\(mem_right_[^,]+, pl\.const\((\d+), pl\.INT64\), (\d+)\), pl\.Mem\.Right",
                allocated_text,
            )
        }
        # Single-C geometry permits separate 16 KiB consumer descriptors in
        # the producer's expired arena. Their MemRef sizes need not equal the
        # 64 KiB held panel; the physical capacity/reuse contract is unchanged.
        assert (0, 65536) in right_ranges
        assert all(0 <= offset < offset + size <= 65536 for offset, size in right_ranges), (
            "DSA_RP should co-place the producer and consumer Right-buffer lifetimes "
            f"inside one 64 KiB L0B arena, got ranges {sorted(right_ranges)}"
        )

    def test_misaligned_n_mat_scratch_roundtrips(self):
        """A misaligned-N Mat-scratch boundary tail survives print -> parse.

        The 128x272 producer exceeds L0c and is consumed only by the second
        matmul, so AutoTile emits an output-stationary Mat-scratch grid with a
        partial N boundary. This exact shape once exposed a printer/parser
        mismatch on the tail variable back when the consumer K-loop was an
        if/else; the loop is a single predicated ``tile.matmul_acc`` now, and
        the roundtrip instrument is what pins the print -> parse property.
        """
        import re  # noqa: PLC0415

        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)
        M, K, N = 128, 64, 272

        cfg = passes.l0_tile_chooser.L0TileConfig()
        cfg.M, cfg.K, cfg.N = M, K, N
        cfg.l0a_bytes = cfg.l0b_bytes = 64 * 1024
        cfg.l0c_bytes = 128 * 1024
        cfg.bytes_a = cfg.bytes_b = 2
        cfg.bytes_c = 4
        cfg.allow_a_stationary = True
        cfg.allow_b_stationary = True
        cfg.allow_k_boundary = True
        choice = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert choice.stationarity == passes.l0_tile_chooser.Stationarity.OutputStationary
        assert N % choice.n != 0, f"expected a partial-N tail, but tile n={choice.n} divides N={N}"

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, K], pl.BF16],
                b: pl.Tensor[[K, N], pl.BF16],
                e: pl.Tensor[[N, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[M, 64], pl.FP32]],
            ) -> pl.Tensor[[M, 64], pl.FP32]:
                c = pl.matmul(a, b, out_dtype=pl.FP32)
                cb = pl.cast(c, pl.BF16, mode="rint")
                d = pl.matmul(cb, e, out_dtype=pl.FP32)
                out = pl.assemble(out, d, [0, 0])
                return out

        lowered = _lower_to_tile_ops(Before)
        with passes.PassContext([ir.make_roundtrip_instrument()]):
            After = passes.auto_tile_matmul_l0()(lowered)

        printed = ir.python_print(After)
        assert printed.count("pl.tile.assemble(") >= 2, "expected a multi-tile Mat-scratch placement"
        tail_offset = N - N % choice.n
        tail_match = re.search(
            rf"(?P<tail>[A-Za-z_]\w*):[^\n]*=\s*pl\.tile\.assemble\([^\n]*\[0, {tail_offset}\]\)",
            printed,
        )
        assert tail_match, "expected the partial-N Mat-scratch assemble at the boundary offset"
        tail_var = tail_match.group("tail")
        tail_extract = re.search(
            rf"pl\.tile\.extract\(\s*{re.escape(tail_var)},.*?target_memory=pl\.Mem\.Left",
            printed,
            re.DOTALL,
        )
        assert tail_extract, "the consumer K-loop must read the completed partial-N scratch variable"
        left_bind = re.search(
            rf"(?P<lhs>[A-Za-z_]\w*):[^\n]*=\s*pl\.tile\.extract\(\s*{re.escape(tail_var)},",
            printed,
        )
        assert left_bind, "the consumer K-loop's Left extract must bind a variable"
        assert re.search(
            rf"pl\.tile\.matmul_acc\([^)\n]*\b{re.escape(left_bind.group('lhs'))}\b[^)\n]*==\s*0\)",
            printed,
        ), "expected the partial-N scratch variable to feed the consumer's predicated K-loop"
        assert "pl.tile.cast(" not in printed, "the bf16 downcast must be folded into the Mat scratch"
        _assert_ssa_valid(After, "test_misaligned_n_mat_scratch_roundtrip")

    def test_chained_matmul_exceeding_mat_capacity_deferred(self):
        """The conservative Mat-capacity gate: a bf16 chained matmul whose result is
        consumed entirely as a matmul operand WOULD take the Mat-scratch path, but its
        ``[512, 1024]`` bf16 scratch (1 MiB) exceeds the backend's Mat/L1 capacity (512
        KiB on Ascend910). The pass leaves the producer on the deferred ``PH-AT-006``
        path (left whole, no Acc->Mat assemble) instead of an impossible allocation."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 512], pl.BF16],
                b: pl.Tensor[[512, 1024], pl.BF16],
                e: pl.Tensor[[1024, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[512, 64], pl.FP32]],
            ) -> pl.Tensor[[512, 64], pl.FP32]:
                c = pl.matmul(a, b, out_dtype=pl.FP32)  # [512, 1024] → 1 MiB bf16 scratch > Mat cap
                cb = pl.cast(c, pl.BF16, mode="rint")  # feeds a bf16 Mat scratch, but exceeds capacity
                d = pl.matmul(cb, e, out_dtype=pl.FP32)  # consumes c only as a matmul operand
                out = pl.assemble(out, d, [0, 0])
                return out

        After = passes.auto_tile_matmul_l0()(_lower_to_tile_ops(Before))
        printed = ir.python_print(After)

        assert printed.count("pl.tile.assemble(") == 0, (
            "a chained-matmul scratch exceeding Mat capacity must not emit any Acc->Mat assemble"
        )
        assert "pl.tile.matmul(a__ssa_v0_mat, b__ssa_v0_mat)" in printed, (
            "the gated producer matmul must be left whole (deferred), not tiled into a Mat scratch"
        )

    def test_chained_matmul_full_k_uses_pipelined_mat_scratch(self):
        """A *full-K* (K fits L0) oversized chained matmul tiles into a Mat scratch via
        the **pipelined** emitter — the Acc->Mat ``tile.assemble`` lands inside the
        ``pl.pipeline`` loop with loop-variable offsets (``tile.assemble`` accepts a
        ``MakeTuple`` of index-typed variables, not only constants)."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[256, 32], pl.BF16],
                b: pl.Tensor[[32, 256], pl.BF16],
                e: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                c = pl.matmul(a, b, out_dtype=pl.FP32)  # [256, 256] > L0c, K=32 fits L0 -> full-K
                cb = pl.cast(c, pl.BF16, mode="rint")  # FIXPIPE downcast -> bf16 Mat scratch (cast fused)
                d = pl.matmul(cb, e, out_dtype=pl.FP32)
                out = pl.assemble(out, d, [0, 0])
                return out

        After = passes.auto_tile_matmul_l0()(_lower_to_tile_ops(Before))
        printed = ir.python_print(After)
        assert "tile.create" in printed and "Mem.Mat" in printed, "expected a Mat output scratch"
        assert "pl.tile.cast(" not in printed, "the downcast must be fused into the Mat scratch"
        assemble_lines = [line for line in printed.splitlines() if "pl.tile.assemble(" in line]
        assert assemble_lines, "the Mat scratch is filled by Acc->Mat assembles"

        # Full-K → the pipelined emitter, whose interior assembles carry LOOP-VARIABLE
        # offsets. Split-K (BuildSplitKGrid) also pipelines but emits CONSTANT offsets,
        # so a bare `pl.pipeline` check cannot distinguish the two — the offset form can.
        def _offset_is_loop_variable(line: str) -> bool:
            offset = line.rsplit("[", 1)[-1].split("]", 1)[0]  # content of the final [...] (the offset)
            return any(ch.isalpha() for ch in offset)

        assert any(_offset_is_loop_variable(line) for line in assemble_lines), (
            "full-K Mat-scratch must emit loop-variable assemble offsets (the pipelined "
            "interior of BuildFullKPipelined); only constant offsets means split-K:\n"
            + "\n".join(assemble_lines)
        )
        _assert_ssa_valid(After, "test_full_k_mat_scratch_chained")

        # Numerically correct vs the bf16 chain, executed through torch_codegen. As in the
        # split-K case the reference carries real bf16 rounding (block-wise bf16 matmuls +
        # the FIXPIPE downcast), so the Frobenius relative error is the robust metric.
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        torch.manual_seed(0)
        a = torch.randn(256, 32, dtype=torch.bfloat16)
        b = torch.randn(32, 256, dtype=torch.bfloat16)
        e = torch.randn(256, 64, dtype=torch.bfloat16)
        out = torch.zeros(256, 64)
        ns: dict = {}
        exec(torch_codegen(After), ns)  # noqa: S102 — executing generated reference code is the point
        ns["kernel"](a, b, e, out)
        c_bf16 = (a.float() @ b.float()).to(torch.bfloat16).float()  # FIXPIPE downcast
        expected = c_bf16 @ e.float()
        rel_err = ((out - expected).norm() / expected.norm()).item()
        assert rel_err < 5e-2, f"full-K Mat-scratch chained bf16 rel_err {rel_err:.3e} exceeds 5e-2"


class TestAutoTileMatmulL0FitsL0cCastFold:
    """Fits-L0c chained-matmul cast-fold: a ``matmul -> cast(bf16) -> matmul`` whose
    ``[M, N]`` result *fits* L0c routes the bf16 downcast through the cube FIXPIPE
    (``tile.assemble`` -> ``pto.tinsert``) instead of the Vector (``pto.tcvt``). The
    cast is folded into a single full-window Acc->Mat assemble and dropped — the
    fits-L0c analogue of the oversized per-sub-tile Mat-scratch fold. Without it the
    standalone Vector cast overflows the Vec buffer at ``[128, 128]``."""

    def _chain(self, k_first):
        """``[128, k_first] @ [k_first, 128] -> [128, 128]`` (fits L0c), cast to bf16,
        fed to ``@ [128, 64]``. ``k_first=64`` keeps the producer un-split (corner C);
        ``k_first=512`` overflows L0a/L0b and forces a K-loop (corner D)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, k_first], pl.BF16],
                b: pl.Tensor[[k_first, 128], pl.BF16],
                e: pl.Tensor[[128, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                c = pl.matmul(a, b, out_dtype=pl.FP32)  # [128, 128] f32, fits L0c
                cb = pl.cast(c, pl.BF16, mode="rint")  # FIXPIPE downcast -> bf16 Mat scratch (folded)
                d = pl.matmul(cb, e, out_dtype=pl.FP32)  # consumes the scratch on-chip
                out = pl.assemble(out, d, [0, 0])
                return out

        return Before

    def test_no_ksplit_cast_folds_to_full_window_assemble(self):
        """Corner C: the producer fits L0a/L0b (no K-loop). The bf16 downcast folds
        into a single full-window Acc->Mat ``tile.assemble`` into a bf16 Mat scratch
        (the standalone ``tile.cast`` is dropped), and the consumer matmul reads the
        scratch on-chip. (Numerics are covered by the st suite — see
        ``tests/st/runtime/ops/test_auto_tile_matmul.py``.)"""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                a: pl.Tensor[[128, 64], pl.BF16],
                b: pl.Tensor[[64, 128], pl.BF16],
                e: pl.Tensor[[128, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                a_mat: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a,
                    [0, 0],
                    [128, 64],
                    target_memory=pl.Mem.Mat,
                    attrs={"__compiler_tensor_to_tile_mat_bridge": True},
                )
                b_mat: pl.Tile[[64, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b,
                    [0, 0],
                    [64, 128],
                    target_memory=pl.Mem.Mat,
                    attrs={"__compiler_tensor_to_tile_mat_bridge": True},
                )
                c: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a_mat, b_mat)
                # Folded downcast: a bf16 Mat scratch + one full-window Acc->Mat
                # assemble (no standalone tile.cast).
                c_mat: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.create(
                    [128, 128], dtype=pl.BF16, target_memory=pl.Mem.Mat
                )
                c_scratch: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.assemble(c_mat, c, [0, 0])
                e_mat: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    e,
                    [0, 0],
                    [128, 64],
                    target_memory=pl.Mem.Mat,
                    attrs={"__compiler_tensor_to_tile_mat_bridge": True},
                )
                d: pl.Tile[[128, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(c_scratch, e_mat)
                out_st: pl.Tensor[[128, 64], pl.FP32] = pl.store(d, [0, 0], out)
                return out_st

        After = passes.auto_tile_matmul_l0()(_lower_to_tile_ops(self._chain(k_first=64)))
        ir.assert_structural_equal(After, Expected)

    def test_ksplit_cast_folds_to_full_window_assemble(self):
        """Corner D: the producer needs a K-loop (``[128, 512] @ [512, 128]``). The
        K-loop's Acc result folds into the *same* single full-window Acc->Mat assemble
        (cast dropped) — the fold is independent of whether the producer was K-tiled.
        (Numerics are covered by the st suite.)"""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                a: pl.Tensor[[128, 512], pl.BF16],
                b: pl.Tensor[[512, 128], pl.BF16],
                e: pl.Tensor[[128, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                a_mat: pl.Tile[[128, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a,
                    [0, 0],
                    [128, 512],
                    target_memory=pl.Mem.Mat,
                    attrs={"__compiler_tensor_to_tile_mat_bridge": True},
                )
                b_mat: pl.Tile[[512, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b,
                    [0, 0],
                    [512, 128],
                    target_memory=pl.Mem.Mat,
                    attrs={"__compiler_tensor_to_tile_mat_bridge": True},
                )
                c_init: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                # Producer K-loop: the Acc result `c` is what the fold assembles.
                for ko, (c_iter,) in pl.pipeline(0, 512, 128, stage=2, init_values=(c_init,)):
                    a_sub: pl.Tile[[128, 128], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        a_mat, 0, ko, shape=[128, 128], target_memory=pl.Mem.Left
                    )
                    b_sub: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        b_mat, ko, 0, shape=[128, 128], target_memory=pl.Mem.Right
                    )
                    c_acc: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                        c_iter, a_sub, b_sub, ko == 0
                    )
                    c: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc] = pl.yield_(c_acc)
                c_mat: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.create(
                    [128, 128], dtype=pl.BF16, target_memory=pl.Mem.Mat
                )
                c_scratch: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.assemble(c_mat, c, [0, 0])
                e_mat: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    e,
                    [0, 0],
                    [128, 64],
                    target_memory=pl.Mem.Mat,
                    attrs={"__compiler_tensor_to_tile_mat_bridge": True},
                )
                d: pl.Tile[[128, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(c_scratch, e_mat)
                out_st: pl.Tensor[[128, 64], pl.FP32] = pl.store(d, [0, 0], out)
                return out_st

        After = passes.auto_tile_matmul_l0()(_lower_to_tile_ops(self._chain(k_first=512)))
        ir.assert_structural_equal(After, Expected)

    def test_cast_to_non_matmul_consumer_not_folded(self):
        """Guard: a fits-L0c matmul whose cast result is consumed by a store (not a
        matmul operand) must keep the Vector cast path — a non-matmul consumer cannot
        read the bf16 value from Mat, so the fold must not fire."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 64], pl.BF16],
                b: pl.Tensor[[64, 128], pl.BF16],
                out: pl.Out[pl.Tensor[[128, 128], pl.BF16]],
            ) -> pl.Tensor[[128, 128], pl.BF16]:
                c = pl.matmul(a, b, out_dtype=pl.FP32)  # [128, 128] f32, fits L0c
                cb = pl.cast(c, pl.BF16, mode="rint")  # consumed by a store, not a matmul operand
                out = pl.assemble(out, cb, [0, 0])
                return out

        After = passes.auto_tile_matmul_l0()(_lower_to_tile_ops(Before))

        # The pass must decline to rewrite anything: the Vector cast stays and no
        # Mat-scratch assemble appears. Pinning the whole program (rather than
        # grepping for those two ops) also catches an unrelated spurious rewrite.
        # The positive counterpart is test_no_ksplit_cast_folds_to_full_window_assemble.
        _assert_unchanged_by_pass(Before, After)

    def test_nondefault_round_mode_not_folded(self):
        """Guard: a fits-L0c chained cast with a directional round mode (e.g.
        ``mode="floor"``) must keep the Vector cast — FIXPIPE's Acc->Mat writeback
        applies a single fixed tie rule and carries no ``rmode``, so folding ``floor``
        into ``pto.tinsert`` would silently change rounding. Only ``rint``
        (round-half-to-even — FIXPIPE's fixed tie rule) is foldable."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 64], pl.BF16],
                b: pl.Tensor[[64, 128], pl.BF16],
                e: pl.Tensor[[128, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                c = pl.matmul(a, b, out_dtype=pl.FP32)  # [128, 128] f32, fits L0c
                cb = pl.cast(c, pl.BF16, mode="floor")  # non-default rounding FIXPIPE can't do
                d = pl.matmul(cb, e, out_dtype=pl.FP32)  # consumed as a matmul operand
                out = pl.assemble(out, d, [0, 0])
                return out

        After = passes.auto_tile_matmul_l0()(_lower_to_tile_ops(Before))

        # ``floor`` is unfoldable, so the whole chain must survive untouched --
        # Vector cast kept, no Mat-scratch assemble, and no other rewrite either.
        _assert_unchanged_by_pass(Before, After)

    @pytest.mark.parametrize("saturation_mode", ["on", "off"])
    def test_explicit_saturation_not_folded(self, saturation_mode):
        """Guard: a fits-L0c chained ``rint`` cast that asked for a saturation mode
        must keep the Vector cast. ``pto.tinsert`` carries no ``satmode`` any more
        than it carries an ``rmode``, so folding one would make ``"on"`` and
        ``"off"`` compile to the same instruction even though the API defines
        different finite-overflow results for them. The same cast without the
        kwarg still folds -- a float destination has no default saturation, so
        absent means "did not ask" (see ``*cast_folds*``)."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 64], pl.BF16],
                b: pl.Tensor[[64, 128], pl.BF16],
                e: pl.Tensor[[128, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                c = pl.matmul(a, b, out_dtype=pl.FP32)  # [128, 128] f32, fits L0c
                cb = pl.cast(c, pl.BF16, mode="rint", saturation_mode=saturation_mode)
                d = pl.matmul(cb, e, out_dtype=pl.FP32)  # consumed as a matmul operand
                out = pl.assemble(out, d, [0, 0])
                return out

        After = passes.auto_tile_matmul_l0()(_lower_to_tile_ops(Before))

        # The request is unrepresentable on the cube, so the whole chain survives.
        _assert_unchanged_by_pass(Before, After)

    def test_default_round_mode_not_folded(self):
        """Guard: the cast default mode is ``"round"`` (round-half-*away*), but FIXPIPE's
        fixed Acc->Mat narrowing is round-half-to-*even* (``rint``). So a default
        ``pl.cast(c, bf16)`` in a chained matmul is NOT folded — it keeps the Vector cast
        (the pass also emits a ``PH-AT-010`` hint pointing at ``mode="rint"``). Only an
        explicit ``rint`` cast folds onto the cube (see the ``*cast_folds*`` tests)."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 64], pl.BF16],
                b: pl.Tensor[[64, 128], pl.BF16],
                e: pl.Tensor[[128, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                c = pl.matmul(a, b, out_dtype=pl.FP32)  # [128, 128] f32, fits L0c
                cb = pl.cast(c, pl.BF16)  # default mode="round" (ties away); FIXPIPE does ties-even
                d = pl.matmul(cb, e, out_dtype=pl.FP32)  # consumed as a matmul operand
                out = pl.assemble(out, d, [0, 0])
                return out

        After = passes.auto_tile_matmul_l0()(_lower_to_tile_ops(Before))

        # Default ``round`` (ties-away) is not FIXPIPE's ties-even, so the chain
        # must survive untouched; only an explicit ``rint`` folds onto the cube.
        _assert_unchanged_by_pass(Before, After)

    @pytest.mark.parametrize("backend", [BackendType.Ascend910, BackendType.Ascend950])
    def test_cast_fold_lowers_cube_only_no_vector(self, backend):
        """End-to-end: the folded fits-L0c chain generates a cube-only kernel —
        ``pto.tinsert`` (FIXPIPE downcast) and zero ``pto.tcvt`` (Vector cast), with no
        ``_aiv`` Vector function. Without the fold this overflows the Vec buffer at
        ``[128, 128]``; with it the intermediate never leaves the cube."""
        from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415
        from pypto.pypto_core import codegen as _codegen_core  # noqa: PLC0415
        from pypto.pypto_core import ir as _ir_core  # noqa: PLC0415

        _backend.reset_for_testing()
        _backend.set_backend_type(backend)

        prog = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(self._chain(k_first=64))
        names = [f.name for f in prog.functions.values()]
        assert not any(n.endswith("_aiv") for n in names), f"no Vector kernel expected, got {names}"

        tinsert = tcvt = 0
        for _name, func in prog.functions.items():
            mlir = _codegen_core.PTOCodegen().generate(_ir_core.Program([func], func.name, prog.span))
            tinsert += mlir.count("pto.tinsert")
            tcvt += mlir.count("pto.tcvt")
        assert tinsert >= 1, "the bf16 downcast must lower to the cube FIXPIPE pto.tinsert"
        assert tcvt == 0, "no Vector pto.tcvt — the cast is folded into the cube writeback"


class TestAutoTileMatmulL0OperandKAlignment:
    """The chooser's K grid comes from the emitted L0 operand layouts."""

    @pytest.mark.parametrize("dbc", [False, True])
    @pytest.mark.parametrize(
        "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
    )
    def test_int8_padded_n_problem_keeps_legal_k(self, planner, dbc):
        """The sweep's logical N=400 case formerly selected illegal k=48.

        This pass-only assertion isolates K alignment; correctness of the padded
        output partitions belongs to the separate valid-output-bounds fix.
        """
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[48, 96], pl.INT8],
                b: pl.Tensor[[96, 400], pl.INT8],
                out: pl.Out[pl.Tensor[[48, 400], pl.INT32]],
            ) -> pl.Tensor[[48, 400], pl.INT32]:
                at = pl.load(a, [0, 0], [48, 96], target_memory=pl.Mem.Mat)
                bt = pl.load(b, [0, 0], [96, 416], valid_shape=[96, 400], target_memory=pl.Mem.Mat)
                acc = pl.tile.matmul(at, bt)
                return pl.store(acc, [0, 0], out)

        with passes.PassContext([], memory_planner=planner, enable_pypto_l0c_double_buffer=dbc):
            after = passes.auto_tile_matmul_l0()(Before)

        class Matmuls(ir.IRVisitor):
            def __init__(self):
                super().__init__()
                self.calls = []

            def visit_call(self, op):
                if op.op.name in {ir.get_op("tile.matmul").name, ir.get_op("tile.matmul_acc").name}:
                    self.calls.append(op)
                super().visit_call(op)

        visitor = Matmuls()
        visitor.visit_program(after)
        calls = visitor.calls
        assert calls
        for call in calls:
            start = int(call.op.name == ir.get_op("tile.matmul_acc").name)
            lhs, rhs = call.args[start : start + 2]
            assert isinstance(lhs.type, ir.TileType) and isinstance(rhs.type, ir.TileType)
            left_k, right_k = lhs.type.shape[1], rhs.type.shape[0]
            assert isinstance(left_k, ir.ConstInt) and isinstance(right_k, ir.ConstInt)
            assert left_k.value == right_k.value and left_k.value % 32 == 0

    @pytest.mark.parametrize("dtype", [pl.BF16, pl.FP16])
    def test_sixteen_aligned_floating_point_k_still_tiles(self, dtype):
        """Do not impose INT8's 32-element grid on BF16/FP16 K=80."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 80], dtype],
                b: pl.Tensor[[80, 256], dtype],
                out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                at = pl.load(a, [0, 0], [64, 80], target_memory=pl.Mem.Mat)
                bt = pl.load(b, [0, 0], [80, 256], target_memory=pl.Mem.Mat)
                acc = pl.tile.matmul(at, bt)
                return pl.store(acc, [0, 0], out)

        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
            after = passes.auto_tile_matmul_l0()(Before)
        calls = _matmul_acc_calls(after)
        assert calls, "legal 16-aligned floating-point K must not be deferred"
        for call in calls:
            lhs = call.args[1]
            assert isinstance(lhs.type, ir.TileType)
            assert lhs.type.memory_space == ir.MemorySpace.Left
            k = lhs.type.shape[1]
            assert isinstance(k, ir.ConstInt) and k.value % 16 == 0

    def test_misaligned_int8_source_k_is_not_retiled(self):
        """K=80 cannot be split into 32-aligned full blocks and a legal tail."""
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 80], pl.INT8],
                b: pl.Tensor[[80, 256], pl.INT8],
                out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
            ) -> pl.Tensor[[64, 256], pl.INT32]:
                at = pl.load(a, [0, 0], [64, 80], target_memory=pl.Mem.Mat)
                bt = pl.load(b, [0, 0], [80, 256], target_memory=pl.Mem.Mat)
                acc = pl.tile.matmul(at, bt)
                return pl.store(acc, [0, 0], out)

        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
            after = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(after, Before)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize("legacy_opt_in", [False, True])
@pytest.mark.parametrize("shape", [(128, 512, 384), (64, 288, 384), (64, 256, 80)])
def test_automatic_dbc_disabled_for_in_tree_planners(planner, legacy_opt_in, shape):
    """The legacy opt-in must not shrink a single-C tile to the dbC capacity."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    M, N, K = shape

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tensor[[M, K], pl.BF16],
            rhs: pl.Tensor[[K, N], pl.BF16],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            a = pl.tile.load(lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat)
            b = pl.tile.load(rhs, [0, 0], [K, N], target_memory=pl.Mem.Mat)
            c = pl.tile.matmul(a, b)
            out = pl.tile.store(c, [0, 0], out)
            return out

    with passes.PassContext([], memory_planner=planner, enable_pypto_l0c_double_buffer=legacy_opt_in):
        after = passes.auto_tile_matmul_l0()(Before)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        expected = passes.auto_tile_matmul_l0()(Before)
    ir.assert_structural_equal(after, expected)
    text = ir.python_print(after)
    assert "pipeline_double_buffer_c" not in text
    assert "pipeline_membership" not in text
    assert "slots=2" not in text
    assert "pl.tile.matmul" in text
    _assert_ssa_valid(after, "automatic_dbc_disabled")


@pytest.mark.parametrize("legacy_opt_in", [False, True])
def test_ptoas_retains_automatic_dbc_independently_of_legacy_flag(legacy_opt_in):
    """Keep the historical full-K dbC choice only for the native planner."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tensor[[160, 64], pl.FP32],
            rhs: pl.Tensor[[64, 160], pl.FP32],
            out: pl.Out[pl.Tensor[[160, 160], pl.FP32]],
        ) -> pl.Tensor[[160, 160], pl.FP32]:
            a = pl.tile.load(lhs, [0, 0], [160, 64], target_memory=pl.Mem.Mat)
            b = pl.tile.load(rhs, [0, 0], [64, 160], target_memory=pl.Mem.Mat)
            c = pl.tile.matmul(a, b)
            out = pl.tile.store(c, [0, 0], out)
            return out

    with passes.PassContext(
        [], memory_planner=passes.MemoryPlanner.PTOAS, enable_pypto_l0c_double_buffer=legacy_opt_in
    ):
        after = passes.auto_tile_matmul_l0()(Before)
    text = ir.python_print(after)
    assert '"pipeline_double_buffer_c": True' in text
    assert '"pipeline_overlap_stores": False' in text
    _assert_ssa_valid(after, "ptoas_automatic_dbc")


@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
@pytest.mark.parametrize("legacy_opt_in", [False, True])
@pytest.mark.parametrize("explicit_dbc", [False, True])
def test_existing_l0_pipeline_preserves_explicit_buffering_policy(planner, legacy_opt_in, explicit_dbc):
    """AutoTile preserves a manual L0 pipeline and an explicit dbC marker.

    This formerly triggered the separate automatic recognizer even with the
    legacy flag off. The output must now equal the input for every planner.
    Explicit two-slot intent still passes through the complete lowering path.
    """
    from pypto.language.parser.text_parser import parse as parse_text  # noqa: PLC0415

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tensor[[16, 128], pl.BF16],
            rhs: pl.Tensor[[128, 512], pl.BF16],
            out: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
        ) -> pl.Tensor[[16, 512], pl.FP32]:
            a_mat = pl.tile.load(lhs, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
            a = pl.tile.extract(a_mat, 0, 0, [16, 128], target_memory=pl.Mem.Left)
            b_mat = pl.tile.load(rhs, [0, 0], [128, 512], target_memory=pl.Mem.Mat)
            for ni, (out_i,) in pl.pipeline(0, 512, 128, stage=2, init_values=(out,)):
                b = pl.tile.extract(b_mat, 0, ni, [128, 128], target_memory=pl.Mem.Right)
                c = pl.tile.matmul(a, b)
                out_s = pl.tile.store(c, [0, ni], out_i)
                out_r = pl.yield_(out_s)
            return out_r

    before = Before
    if explicit_dbc:
        source = ir.python_print(Before)
        original = "stage=2, init_values=(out,)"
        assert source.count(original) == 1
        source = source.replace(
            original,
            original + ', attrs={"pipeline_double_buffer_c": True, "pipeline_overlap_stores": False}',
        )
        before = parse_text(source)
        assert isinstance(before, ir.Program)
    with passes.PassContext([], memory_planner=planner, enable_pypto_l0c_double_buffer=legacy_opt_in):
        after = passes.auto_tile_matmul_l0()(before)
        lowered = passes.lower_pipeline_loops()(after)
        lowered = passes.canonicalize_io_order()(lowered)
    ir.assert_structural_equal(after, before)

    class AccumulatorMemberships(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.memberships = []

        def visit_call(self, op):
            if op.op.name == ir.get_op("tile.matmul").name:
                self.memberships.append("pipeline_membership" in op.attrs)
            super().visit_call(op)

    # Operand extracts always carry pipeline memberships; inspect cube MADs.
    collector = AccumulatorMemberships()
    collector.visit_program(lowered)
    assert collector.memberships
    assert all(member == explicit_dbc for member in collector.memberships)
    _assert_ssa_valid(lowered, "explicit_l0_pipeline_policy")


@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
@pytest.mark.parametrize("m,n", [(64, 256), (512, 256), (64, 512)])
def test_unwindowable_mat_bias_uses_legal_k_only_fallback(planner, ascend_backend, m, n, capfd):
    """A Mat bias parameter can be reused whole, but cannot be reloaded by N window."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tile[[m, 256], pl.BF16, pl.Mem.Mat],
            rhs: pl.Tile[[256, n], pl.BF16, pl.Mem.Mat],
            bias: pl.Tile[[1, n], pl.FP32, pl.Mem.Mat],
        ) -> pl.Tile[[m, n], pl.FP32, pl.Mem.Acc]:
            result = pl.tile.matmul_bias(lhs, rhs, bias)
            return result

    with passes.PassContext([], memory_planner=planner):
        after = passes.auto_tile_matmul_l0()(Before)
    diagnostics = capfd.readouterr().err
    if (m, n) != (64, 256):
        # Splitting K cannot repair an oversized full output or Bias vector.
        ir.assert_structural_equal(after, Before)
        assert "PH-AT-011" in diagnostics or "PH-AT-006" in diagnostics
        return

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tile[[64, 256], pl.BF16, pl.Mem.Mat],
            rhs: pl.Tile[[256, 256], pl.BF16, pl.Mem.Mat],
            bias: pl.Tile[[1, 256], pl.FP32, pl.Mem.Mat],
        ) -> pl.Tile[[64, 256], pl.FP32, pl.Mem.Acc]:
            bias_l0 = pl.tile.move(bias, target_memory=pl.Mem.Bias)
            first_left = pl.tile.extract(lhs, 0, 0, shape=[64, 64], target_memory=pl.Mem.Left)
            first_right = pl.tile.extract(rhs, 0, 0, shape=[64, 256], target_memory=pl.Mem.Right)
            seed = pl.tile.matmul_bias(first_left, first_right, bias_l0)
            for ko, (acc,) in pl.pipeline(64, 256, 64, init_values=(seed,), stage=2):
                left = pl.tile.extract(lhs, 0, ko, shape=[64, 64], target_memory=pl.Mem.Left)
                right = pl.tile.extract(rhs, ko, 0, shape=[64, 256], target_memory=pl.Mem.Right)
                updated = pl.tile.matmul_acc(acc, left, right)
                result = pl.yield_(updated)
            return result

    ir.assert_structural_equal(after, Expected)
    printed = ir.python_print(after)
    assert "PH-AT-011" not in diagnostics
    assert printed.count("pl.tile.matmul_bias(") == 1
    assert "pl.tile.matmul_acc(" in printed
    assert printed.count("pl.tile.move(bias, target_memory=pl.Mem.Bias)") == 1
    assert "pl.tile.load(" not in printed
    assert "pl.tile.extract(bias" not in printed
    _assert_ssa_valid(after, "unwindowable_mat_bias_k_only_fallback")

    torch = pytest.importorskip("torch")
    from pypto.debug import torch_codegen  # noqa: PLC0415

    # Small integers make all products and sums exactly representable, catching
    # missing K blocks and repeated bias without floating-point tolerances.
    generator = torch.Generator().manual_seed(2872)
    lhs = torch.randint(-2, 3, (m, 256), generator=generator).to(torch.bfloat16)
    rhs = torch.randint(-2, 3, (256, n), generator=generator).to(torch.bfloat16)
    bias = torch.randint(1, 5, (1, n), generator=generator).float()
    ns: dict = {}
    exec(torch_codegen(after), ns)  # noqa: S102 -- execute the generated reference
    actual = ns["kernel"](lhs, rhs, bias)
    torch.testing.assert_close(actual, lhs.float() @ rhs.float() + bias, rtol=0, atol=0)


@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
def test_unfoldable_output_uses_k_only_fallback(planner, ascend_backend):
    """Preserve the whole result when no consumer can place a selected M/N grid."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tile[[64, 128], pl.FP32, pl.Mem.Mat],
            rhs: pl.Tile[[128, 512], pl.FP32, pl.Mem.Mat],
        ) -> pl.Tile[[64, 512], pl.FP32, pl.Mem.Acc]:
            result = pl.tile.matmul(lhs, rhs)
            return result

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tile[[64, 128], pl.FP32, pl.Mem.Mat],
            rhs: pl.Tile[[128, 512], pl.FP32, pl.Mem.Mat],
        ) -> pl.Tile[[64, 512], pl.FP32, pl.Mem.Acc]:
            seed = pl.tile.create([64, 512], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for ko, (acc,) in pl.pipeline(0, 128, 16, init_values=(seed,), stage=2):
                left = pl.tile.extract(lhs, 0, ko, shape=[64, 16], target_memory=pl.Mem.Left)
                right = pl.tile.extract(rhs, ko, 0, shape=[16, 512], target_memory=pl.Mem.Right)
                updated = pl.tile.matmul_acc(acc, left, right, init_cond=(ko == 0))
                result = pl.yield_(updated)
            return result

    with passes.PassContext([], memory_planner=planner):
        After = passes.auto_tile_matmul_l0()(Before)
    ir.assert_structural_equal(After, Expected)


@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
@pytest.mark.parametrize("window", [None, [2, 32, 512], [1, 32, 512]])
def test_non_page_store_keeps_k_only_fallback(planner, ascend_backend, window):
    """Do not interpret collapsed batch rows or clipped windows as matrix axes."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tile[[64, 128], pl.FP32, pl.Mem.Mat],
            rhs: pl.Tile[[128, 512], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[128, 17])],
            out: pl.Out[pl.Tensor[[2, 64, 512], pl.FP32]],
        ) -> pl.Tensor[[2, 64, 512], pl.FP32]:
            result = pl.tile.matmul(lhs, rhs)
            out = pl.tile.store(result, [0, 0, 0], out, shapes=window)
            return out

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tile[[64, 128], pl.FP32, pl.Mem.Mat],
            rhs: pl.Tile[[128, 512], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[128, 17])],
            out: pl.Out[pl.Tensor[[2, 64, 512], pl.FP32]],
        ) -> pl.Tensor[[2, 64, 512], pl.FP32]:
            storage = pl.tile.create([64, 512], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            seed = pl.tile.set_validshape(storage, 64, 17)
            for ko, (acc,) in pl.pipeline(0, 128, 16, init_values=(seed,), stage=2):
                left = pl.tile.extract(lhs, 0, ko, shape=[64, 16], target_memory=pl.Mem.Left)
                right = pl.tile.extract(rhs, ko, 0, shape=[16, 512], target_memory=pl.Mem.Right)
                updated = pl.tile.matmul_acc(acc, left, right, init_cond=(ko == 0))
                result = pl.yield_(updated)
            out = pl.tile.store(result, [0, 0, 0], out, shapes=window)
            return out

    with passes.PassContext([], memory_planner=planner):
        After = passes.auto_tile_matmul_l0()(Before)
    ir.assert_structural_equal(After, Expected)


class TestAutoTileMatmulAccPhase:
    """Each physical reduction sets its unit flag once, immediately before drain."""

    def test_final_matmul_uniform_k_peels_last_block(self):
        """An eight-block K reduction emits seven Partial blocks and one Final.

        Keeping Final on the loop body would set the unit flag on the first
        iteration and make the second iteration wait before the store can run.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                a = pl.tile.load(lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat)
                b = pl.tile.load(rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul(a, b, acc_phase=pl.AccPhase.Final)
                out = pl.store(c, [0, 0], out, st_phase=pl.STPhase.Final)
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
                a = pl.tile.load(lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat)
                b = pl.tile.load(rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat)
                seed = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                for ko, (acc,) in pl.pipeline(0, 1792, 256, init_values=(seed,), stage=2):
                    sa = pl.tile.extract(a, 0, ko, [16, 256], target_memory=pl.Mem.Left)
                    sb = pl.tile.extract(b, ko, 0, [256, 64], target_memory=pl.Mem.Right)
                    partial = pl.tile.matmul_acc(
                        acc, sa, sb, init_cond=(ko == 0), acc_phase=pl.AccPhase.Partial
                    )
                    main = pl.yield_(partial)
                sa_last = pl.tile.extract(a, 0, 1792, [16, 256], target_memory=pl.Mem.Left)
                sb_last = pl.tile.extract(b, 1792, 0, [256, 64], target_memory=pl.Mem.Right)
                final = pl.tile.matmul_acc(main, sa_last, sb_last, acc_phase=pl.AccPhase.Final)
                out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)
        self._verify_pairing(After)

    @pytest.mark.parametrize(
        "M,K,step,stop,last_width", [(16, 2048, 256, 1792, 256), (64, 544, 192, 384, 160)]
    )
    def test_final_acc_loop_preserves_init_cond_and_uses_existing_tail(self, M, K, step, stop, last_width):
        """Initialization stays on the first block; the final block only accumulates."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.BF16],
                rhs: pl.Tensor[[K, 64], pl.BF16],
                seed: pl.Tile[[M, 64], pl.FP32, pl.Mem.Acc],
                first: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[M, 64], pl.FP32]],
            ) -> pl.Tensor[[M, 64], pl.FP32]:
                a = pl.tile.load(lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                b = pl.tile.load(rhs, [0, 0], [K, 64], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_acc(seed, a, b, init_cond=(first == 0), acc_phase=pl.AccPhase.Final)
                out = pl.store(c, [0, 0], out, st_phase=pl.STPhase.Final)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, K], pl.BF16],
                rhs: pl.Tensor[[K, 64], pl.BF16],
                seed: pl.Tile[[M, 64], pl.FP32, pl.Mem.Acc],
                first: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[M, 64], pl.FP32]],
            ) -> pl.Tensor[[M, 64], pl.FP32]:
                a = pl.tile.load(lhs, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                b = pl.tile.load(rhs, [0, 0], [K, 64], target_memory=pl.Mem.Mat)
                for ko, (acc,) in pl.pipeline(0, stop, step, init_values=(seed,), stage=2):
                    sa = pl.tile.extract(a, 0, ko, [M, step], target_memory=pl.Mem.Left)
                    sb = pl.tile.extract(b, ko, 0, [step, 64], target_memory=pl.Mem.Right)
                    partial = pl.tile.matmul_acc(
                        acc, sa, sb, init_cond=(first == 0 and ko == 0), acc_phase=pl.AccPhase.Partial
                    )
                    main = pl.yield_(partial)
                sa_last = pl.tile.extract(a, 0, stop, [M, last_width], target_memory=pl.Mem.Left)
                sb_last = pl.tile.extract(b, stop, 0, [last_width, 64], target_memory=pl.Mem.Right)
                final = pl.tile.matmul_acc(main, sa_last, sb_last, acc_phase=pl.AccPhase.Final)
                out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)
        self._verify_pairing(After)

    @pytest.mark.parametrize("K,step", [(272, 144), (512, 256)])
    def test_final_acc_single_main_block_keeps_user_predicate(self, K, step):
        """A one-block prefix carries the user's initialization predicate verbatim.

        K=512 has two equal blocks, so peeling its Final leaves just one prefix
        block. K=272 already has a short tail. Neither needs a pipeline loop.
        """
        last_width = K - step

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[64, K], pl.BF16],
                rhs: pl.Tensor[[K, 64], pl.BF16],
                seed: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc],
                first: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                a = pl.tile.load(lhs, [0, 0], [64, K], target_memory=pl.Mem.Mat)
                b = pl.tile.load(rhs, [0, 0], [K, 64], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul_acc(seed, a, b, init_cond=(first == 0), acc_phase=pl.AccPhase.Final)
                out = pl.store(c, [0, 0], out, st_phase=pl.STPhase.Final)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[64, K], pl.BF16],
                rhs: pl.Tensor[[K, 64], pl.BF16],
                seed: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc],
                first: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                a = pl.tile.load(lhs, [0, 0], [64, K], target_memory=pl.Mem.Mat)
                b = pl.tile.load(rhs, [0, 0], [K, 64], target_memory=pl.Mem.Mat)
                sa = pl.tile.extract(a, 0, 0, [64, step], target_memory=pl.Mem.Left)
                sb = pl.tile.extract(b, 0, 0, [step, 64], target_memory=pl.Mem.Right)
                partial = pl.tile.matmul_acc(
                    seed, sa, sb, init_cond=(first == 0), acc_phase=pl.AccPhase.Partial
                )
                sa_last = pl.tile.extract(a, 0, step, [64, last_width], target_memory=pl.Mem.Left)
                sb_last = pl.tile.extract(b, step, 0, [last_width, 64], target_memory=pl.Mem.Right)
                final = pl.tile.matmul_acc(partial, sa_last, sb_last, acc_phase=pl.AccPhase.Final)
                out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)
        self._verify_pairing(After)

    @pytest.mark.parametrize("phase", [pl.AccPhase.Unspecified, pl.AccPhase.Partial])
    def test_nonfinal_phase_keeps_uniform_k_loop(self, phase):
        """Partial never sets a flag; Unspecified retains the default loop shape."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                a = pl.tile.load(lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat)
                b = pl.tile.load(rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul(a, b, acc_phase=phase)
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
                a = pl.tile.load(lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat)
                b = pl.tile.load(rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat)
                seed = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                for ko, (acc,) in pl.pipeline(0, 2048, 256, init_values=(seed,), stage=2):
                    sa = pl.tile.extract(a, 0, ko, [16, 256], target_memory=pl.Mem.Left)
                    sb = pl.tile.extract(b, ko, 0, [256, 64], target_memory=pl.Mem.Right)
                    partial = pl.tile.matmul_acc(acc, sa, sb, init_cond=(ko == 0), acc_phase=phase)
                    result = pl.yield_(partial)
                out = pl.store(result, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Expected)
        self._verify_pairing(After)

    @pytest.mark.parametrize("K", [64, 544])
    @pytest.mark.parametrize("alias", [False, True])
    def test_final_mn_subtiles_each_have_a_matching_drain(self, K, alias):
        """M/N tiling expands the logical pair into physical producer/store pairs.

        This also covers single-use plain aliases: after folding there is no
        whole-size accumulator left for those alias definitions to reference.
        """
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[384, K], pl.BF16],
                rhs: pl.Tensor[[K, 640], pl.BF16],
                out: pl.Out[pl.Tensor[[384, 640], pl.FP32]],
            ) -> pl.Tensor[[384, 640], pl.FP32]:
                a = pl.tile.load(lhs, [0, 0], [384, K], target_memory=pl.Mem.Mat)
                b = pl.tile.load(rhs, [0, 0], [K, 640], target_memory=pl.Mem.Mat)
                c = pl.tile.matmul(a, b, acc_phase=pl.AccPhase.Final)
                alias1 = c
                alias2 = alias1
                out = pl.store(alias2, [0, 0], out, st_phase=pl.STPhase.Final)
                return out

        if not alias:

            @pl.program
            class Direct:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    lhs: pl.Tensor[[384, K], pl.BF16],
                    rhs: pl.Tensor[[K, 640], pl.BF16],
                    out: pl.Out[pl.Tensor[[384, 640], pl.FP32]],
                ) -> pl.Tensor[[384, 640], pl.FP32]:
                    a = pl.tile.load(lhs, [0, 0], [384, K], target_memory=pl.Mem.Mat)
                    b = pl.tile.load(rhs, [0, 0], [K, 640], target_memory=pl.Mem.Mat)
                    c = pl.tile.matmul(a, b, acc_phase=pl.AccPhase.Final)
                    out = pl.store(c, [0, 0], out, st_phase=pl.STPhase.Final)
                    return out

            Before = Direct
        After = passes.auto_tile_matmul_l0()(Before)
        _assert_ssa_valid(After, "final_mn_subtiles")
        self._verify_pairing(After)
        printed = ir.python_print(After)
        assert "alias1" not in printed and "alias2" not in printed

        class PhysicalPairs(ir.IRVisitor):
            def __init__(self):
                super().__init__()
                self.final_count = 0
                self.store_count = 0

            def visit_call(self, op):
                if op.op.name in {ir.get_op("tile.matmul").name, ir.get_op("tile.matmul_acc").name}:
                    phase = dict(op.kwargs).get("acc_phase", 0)
                    assert phase in {int(pl.AccPhase.Partial), int(pl.AccPhase.Final)}
                    self.final_count += phase == int(pl.AccPhase.Final)
                    assert isinstance(op.type, ir.TileType)
                    assert all(isinstance(dim, ir.ConstInt) for dim in op.type.shape)
                    rows, cols = (int(dim.value) for dim in op.type.shape if isinstance(dim, ir.ConstInt))
                    assert rows * cols * 4 <= 131072
                if op.op.name == ir.get_op("tile.store").name:
                    self.store_count += 1
                    assert dict(op.kwargs)["st_phase"] == int(pl.STPhase.Final)
                super().visit_call(op)

        pairs = PhysicalPairs()
        pairs.visit_program(After)
        assert pairs.final_count == pairs.store_count > 0

    @staticmethod
    def _verify_pairing(program):
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.AccStorePhaseValid)
        assert passes.PropertyVerifierRegistry.verify(props, program) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

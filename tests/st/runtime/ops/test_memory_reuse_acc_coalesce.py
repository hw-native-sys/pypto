# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime st: MemoryReuse coalesces a peeled loop-carried L0C accumulator.

A ``512 x 512 x 192`` bf16 matmul. The L0 tiler picks a ``176 x 176`` output
tile whose fp32 accumulator is ~121 KB, and ``LowerPipelineLoops`` peels the
stage-2 K-loop into ``if``-phis (first block seeds with ``matmul``, later blocks
accumulate in place with ``matmul_acc``). ``MemoryReuse`` must coalesce the whole
accumulator chain onto ONE L0C buffer.

Regression: the peeled if-phi reconciliation used to treat the dead ``if k==0``
seed branch as canonical and copy the live accumulator onto the seed's *separate*
buffer with a phantom Acc->Acc ``tile.move`` -- a 2nd co-live 121 KB L0C buffer
that (1) overflowed the 128 KB L0C at ``AllocateMemoryAddr`` (this shape failed
to compile) and (2) is an Acc->Acc ``tmov`` that ptoas rejects on every target.

This is the on-device validation the unit / codegen checks cannot give: that the
coalesced accumulator compiles, ptoas accepts it, and it computes correctly.
Numerics use one f32-accumulated golden (bf16 operands, cube f32 accumulation).
"""

import dataclasses

import pytest

torch = pytest.importorskip("torch")

import pypto.language as pl  # noqa: E402
from pypto.pypto_core.passes import MemoryPlanner  # noqa: E402

M, N, K = 512, 512, 192


@pl.jit
def matmul_512x512x192_bf16(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    """``a @ b`` -> ``[512, 512]`` f32 stored to DDR. The chosen ``176 x 176`` L0
    tile + split-K (K=192) peels into the pipelined accumulator if-phi shape that
    exercises the MemoryReuse accumulator-coalescing fix."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm_512x512x192_acc_coalesce"):
        c = pl.matmul(a, b, out_dtype=pl.FP32)
        out = pl.assemble(out, c, [0, 0])
    return out


@pytest.mark.platforms("a2a3", "a2a3sim")
class TestMemoryReuseAccumulatorCoalesce:
    """End-to-end device check for the peeled-accumulator coalescing fix."""

    @pytest.mark.parametrize(
        "memory_planner",
        [MemoryPlanner.PYPTO, MemoryPlanner.PTOAS],
        ids=["pypto", "ptoas"],
    )
    def test_512x512x192_bf16_compiles_and_runs(self, test_config, memory_planner):
        """The peeled ``176 x 176`` fp32 accumulator coalesces to one L0C buffer,
        so this shape compiles (no L0C overflow), ptoas accepts it (no Acc->Acc
        ``tmov``), and the result matches the reference.

        Run under BOTH memory planners. PYPTO exercises the ``MemoryReuse``
        accumulator-coalescing fix; PTOAS is the regression guard for the L0C
        matmul K-reduction loop-carry (the cb4181e8 must-alias handoff): under
        ``memory_planner=PTOAS`` the reuse/allocation passes are skipped and
        ``MaterializeSemanticAliases`` alone must keep the peeled accumulator
        if-phi on one addr-less ``tile_buf`` handle, else ptoas plans two L0C
        buffers and the K-reduction silently drops accumulation."""
        matmul_512x512x192_bf16._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(M, K, dtype=torch.bfloat16)
        b = torch.randn(K, N, dtype=torch.bfloat16)
        out = torch.zeros((M, N), dtype=torch.float32)

        config = dataclasses.replace(test_config, memory_planner=memory_planner)
        matmul_512x512x192_bf16(a, b, out, config=config)

        # bf16 operands, cube f32 accumulation -> f32 reference on the same bf16 inputs.
        expected = a.float() @ b.float()
        rel_err = ((out - expected).norm() / expected.norm()).item()
        assert rel_err < 2e-2, (
            f"512x512x192 bf16 matmul ({memory_planner}) Frobenius rel_err = {rel_err:.3e} exceeds 2e-2"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

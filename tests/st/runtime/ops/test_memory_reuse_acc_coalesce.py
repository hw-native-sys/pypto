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

import pytest

torch = pytest.importorskip("torch")

import pypto.language as pl  # noqa: E402
from harness import st  # noqa: E402

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


def _acc_coalesce_case():
    """The peeled ``176 x 176`` fp32 accumulator coalesces to one L0C buffer.

    So this shape compiles (no L0C overflow), ptoas accepts it (no Acc->Acc
    ``tmov``), and the result matches the reference. bf16 operands with cube f32
    accumulation, so the reference is f32 over the same bf16 inputs, and the
    assertion is on the tensor's Frobenius relative error rather than per
    element — accumulation order moves individual elements much further than it
    moves the whole tensor.
    """
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)
    out = torch.zeros((M, N), dtype=torch.float32)
    return st.case(
        matmul_512x512x192_bf16,
        a,
        b,
        out,
        name="matmul_512x512x192_bf16_acc_coalesce",
        golden=lambda _: a.float() @ b.float(),
        compare=st.rel_err_under(2e-2),
    )


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(_acc_coalesce_case())
def test_512x512x192_bf16_compiles_and_runs(case_run):
    """End-to-end device check for the peeled-accumulator coalescing fix."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

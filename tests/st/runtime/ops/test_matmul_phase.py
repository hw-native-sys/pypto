# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Execute the Matmul producer/store unit-flag protocol on A2/A3.

Textual phase attributes cannot prove that a flag is cleared before an Acc slot
is reused. These cases compare real results and exercise repeated pairs, two
simultaneously live accumulators, skipped pairs, initialization, and AutoTile
boundaries. Run with the harness's bounded task-submit execution and batch size
one so a stalled case cannot poison the rest of the batch.
"""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto import passes

M, N, K = 32, 64, 64
REPEATS = 4
SENTINEL = -7.0


def _values(rows: int, cols: int, offset: int = 0) -> torch.Tensor:
    """Dyadic BF16 inputs keep all these FP32 reductions exactly representable."""
    values = (torch.arange(rows * cols).reshape(rows, cols) + offset).remainder(7) - 3
    return (values.float() / 4).to(torch.bfloat16)


@pl.jit
def _default_matmul(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs = pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
        result = pl.tile.matmul(lhs, rhs)
        pl.store(result, [0, 0], out)
    return out


def _final_matmul(m: int, n: int, k: int):
    @pl.jit
    def kernel(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            lhs = pl.load(a, [0, 0], [m, k], target_memory=pl.Mem.Mat)
            rhs = pl.load(b, [0, 0], [k, n], target_memory=pl.Mem.Mat)
            result = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)
            pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)
        return out

    return kernel


def _split_matmul(init_cond: bool | None):
    @pl.jit
    def kernel(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            lhs0 = pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
            rhs0 = pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
            acc = pl.tile.matmul(lhs0, rhs0, acc_phase=pl.AccPhase.Partial)
            lhs1 = pl.load(a, [0, K], [M, K], target_memory=pl.Mem.Mat)
            rhs1 = pl.load(b, [K, 0], [K, N], target_memory=pl.Mem.Mat)
            result = pl.tile.matmul_acc(acc, lhs1, rhs1, init_cond=init_cond, acc_phase=pl.AccPhase.Final)
            pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)
        return out

    return kernel


@pl.jit
def _runtime_initialization(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs = pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
        acc = pl.tile.create([M, N], pl.FP32, target_memory=pl.Mem.Acc)
        for step in pl.range(2):
            # Both runtime init_cond arms must keep Final. Draining each step
            # makes reusing the same physical Acc on the next step legal.
            acc = pl.tile.matmul_acc(acc, lhs, rhs, init_cond=(step == 0), acc_phase=pl.AccPhase.Final)
            pl.store(acc, [step * M, 0], out, st_phase=pl.STPhase.Final)
    return out


@pl.jit
def _loop_pairs(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs = pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
        for step in pl.range(REPEATS):
            result = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)
            pl.store(result, [step * M, 0], out, st_phase=pl.STPhase.Final)
    return out


@pl.jit
def _pipeline_pairs(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs = pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
        # Include the prologue and final drain produced by pipeline lowering.
        for step in pl.pipeline(REPEATS, stage=2):
            result = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)
            pl.store(result, [step * M, 0], out, st_phase=pl.STPhase.Final)
    return out


@pl.jit
def _skipped_pairs(a: pl.Tensor, b: pl.Tensor, out: pl.InOut[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs = pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
        for step in pl.range(REPEATS):
            # Skip the whole pair: no path may set a flag without clearing it.
            if step % 2 == 0:
                result = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)
                pl.store(result, [step * M, 0], out, st_phase=pl.STPhase.Final)
    return out


@pl.jit
def _dual_accumulators(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs0 = pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        lhs1 = pl.load(a, [M, 0], [M, K], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
        for step in pl.range(REPEATS):
            acc0: pl.Tile[[M, N], pl.FP32, pl.MemRef("phase_l0c", slots=2)[step % 2], pl.Mem.Acc] = (
                pl.tile.matmul(lhs0, rhs, acc_phase=pl.AccPhase.Partial)
            )
            acc1: pl.Tile[[M, N], pl.FP32, pl.MemRef("phase_l0c", slots=2)[(step + 1) % 2], pl.Mem.Acc] = (
                pl.tile.matmul(lhs1, rhs, acc_phase=pl.AccPhase.Partial)
            )
            # Both slots are live. Finalize/drain separately before the next
            # iteration rotates their roles; collapsing their addresses fails.
            result0 = pl.tile.matmul_acc(acc0, lhs0, rhs, acc_phase=pl.AccPhase.Final)
            pl.store(result0, [step * 2 * M, 0], out, st_phase=pl.STPhase.Final)
            result1 = pl.tile.matmul_acc(acc1, lhs1, rhs, acc_phase=pl.AccPhase.Final)
            pl.store(result1, [step * 2 * M + M, 0], out, st_phase=pl.STPhase.Final)
    return out


def _case(kernel, name: str, *, m: int = M, n: int = N, k: int = K):
    a, b = _values(m, k), _values(k, n, offset=2)
    out = torch.zeros((m, n), dtype=torch.float32)
    return st.case(
        kernel,
        a,
        b,
        out,
        name=name,
        golden=lambda tensors: tensors["a"].float() @ tensors["b"].float(),
        memory_planner=passes.MemoryPlanner.PYPTO,
        rtol=0,
        atol=0,
    )


def _split_case(init_cond: bool | None):
    a, b = _values(M, 2 * K), _values(2 * K, N, offset=2)
    expected = a.float() @ b.float()
    if init_cond is True:
        # A valid existing partial makes overwrite distinguishable from add.
        expected = a[:, K:].float() @ b[K:].float()
    return st.case(
        _split_matmul(init_cond),
        a,
        b,
        torch.zeros((M, N)),
        name=f"matmul_phase_acc_init_{init_cond}",
        golden=lambda _: expected,
        memory_planner=passes.MemoryPlanner.PYPTO,
        rtol=0,
        atol=0,
    )


def _lifecycle_case(kernel, name: str, *, runtime_init: bool = False, skip: bool = False):
    a, b = _values(M, K), _values(K, N, offset=2)
    product = a.float() @ b.float()
    count = 2 if runtime_init else REPEATS
    expected = product.repeat(count, 1)
    if runtime_init:
        expected[M:] *= 2
    elif skip:
        for step in range(1, count, 2):
            expected[step * M : (step + 1) * M] = SENTINEL
    return st.case(
        kernel,
        a,
        b,
        torch.full((count * M, N), SENTINEL),
        name=name,
        golden=lambda _: expected,
        memory_planner=passes.MemoryPlanner.PYPTO,
        rtol=0,
        atol=0,
    )


def _dual_acc_case():
    a, b = _values(2 * M, K), _values(K, N, offset=2)
    return st.case(
        _dual_accumulators,
        a,
        b,
        torch.zeros((REPEATS * 2 * M, N)),
        name="matmul_phase_dual_acc_drain",
        golden=lambda _: (2 * (a.float() @ b.float())).repeat(REPEATS, 1),
        memory_planner=passes.MemoryPlanner.PYPTO,
        rtol=0,
        atol=0,
    )


@pytest.mark.platforms("a2a3", reason="A2/A3 Cube-to-FIX unit-flag protocol")
@st.cases(
    _case(_default_matmul, "matmul_phase_default_control"),
    _case(_final_matmul(M, N, K), "matmul_phase_final"),
    # A3 chooses k=96 here: the two-block Partial prefix is followed by an
    # 80-wide natural tail, whose Final must survive AutoTile and lowering.
    _case(_final_matmul(128, 128, 272), "matmul_phase_k_boundary", m=128, n=128, k=272),
    # The output exceeds L0C and both dimensions have prime 16-block counts,
    # exercising physical output sub-tiles and their final boundary drains.
    _case(_final_matmul(272, 304, 64), "matmul_phase_mn_boundary", m=272, n=304, k=64),
    *(_split_case(init_cond) for init_cond in (None, True, False)),
    _lifecycle_case(_runtime_initialization, "matmul_phase_runtime_init", runtime_init=True),
    _lifecycle_case(_loop_pairs, "matmul_phase_loop_reuse"),
    _lifecycle_case(_pipeline_pairs, "matmul_phase_pipeline_drain"),
    _lifecycle_case(_skipped_pairs, "matmul_phase_skip_whole_pair", skip=True),
    _dual_acc_case(),
)
def test_matmul_phase_protocol(case_run):
    case_run.assert_passed()


def _dtype_case(dtype: torch.dtype):
    """Exercise each accumulator dtype with exactly representable products."""
    lhs = torch.arange(M * K).reshape(M, K).remainder(7) - 3
    rhs = (torch.arange(K * N).reshape(K, N) + 2).remainder(7) - 3
    integer = dtype == torch.int8
    if integer:
        a, b = lhs.to(dtype), rhs.to(dtype)
        accumulator_dtype = torch.int32
    else:
        a, b = (lhs.float() / 4).to(dtype), (rhs.float() / 4).to(dtype)
        accumulator_dtype = torch.float32
    return st.case(
        _final_matmul(M, N, K),
        a,
        b,
        torch.zeros((M, N), dtype=accumulator_dtype),
        name=f"matmul_phase_dtype_{str(dtype).removeprefix('torch.')}",
        golden=lambda _: a.to(accumulator_dtype) @ b.to(accumulator_dtype),
        memory_planner=passes.MemoryPlanner.PYPTO,
        rtol=0,
        atol=0,
    )


@pytest.mark.platforms("a2a3", reason="A2/A3 Cube-to-FIX unit-flag protocol")
@st.cases(*(_dtype_case(dtype) for dtype in (torch.float16, torch.float32, torch.int8)))
def test_matmul_phase_dtype_generality(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

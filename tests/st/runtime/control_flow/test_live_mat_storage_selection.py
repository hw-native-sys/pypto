# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical checks for live storage selection and current/next prefetch rotation."""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto import passes


@pl.jit.incore
def rotate(
    a: pl.Tensor[[512, 64], pl.FP16],
    b: pl.Tensor[[64, 16], pl.FP16],
    out: pl.Out[pl.Tensor[[512, 16], pl.FP32]],
):
    current = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
    weight = pl.load(b, [0, 0], [64, 16], target_memory=pl.MemorySpace.Mat)
    right = pl.tile.move(weight, target_memory=pl.MemorySpace.Right)
    for block in pl.range(8):
        if block + 1 < 8:
            following = pl.load(a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        else:
            following = current
        left = pl.tile.move(current, target_memory=pl.MemorySpace.Left)
        result = pl.tile.matmul(left, right)
        pl.store(result, [block * 64, 0], out)
        current = following
    return out


@pl.jit
def entry(
    a: pl.Tensor[[512, 64], pl.FP16],
    b: pl.Tensor[[64, 16], pl.FP16],
    out: pl.Out[pl.Tensor[[512, 16], pl.FP32]],
):
    with pl.spmd(1):
        out = rotate(a, b, out)
    return out


@pl.jit.incore
def mat_phi_core(
    x: pl.Tensor[[32, 16], pl.BF16],
    y: pl.Tensor[[16, 16], pl.BF16],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    a = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
    b = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Mat)
    rhs = pl.load(y, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
    if pl.read(flag, [0]) > 0:
        selected = pl.yield_(a)
    else:
        selected = pl.yield_(b)
    product = pl.matmul(selected, rhs, out_dtype=pl.FP32)
    pl.store(product, [0, 0], out)
    # Both inputs remain live after selection and must keep independent storage.
    keep_a = pl.matmul(a, rhs, out_dtype=pl.FP32)
    pl.store(keep_a, [16, 0], out)
    keep_b = pl.matmul(b, rhs, out_dtype=pl.FP32)
    pl.store(keep_b, [32, 0], out)
    return out


@pl.jit
def mat_phi(
    x: pl.Tensor[[32, 16], pl.BF16],
    y: pl.Tensor[[16, 16], pl.BF16],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    with pl.spmd(1):
        out = mat_phi_core(x, y, flag, out)
    return out


@pl.jit.incore
def vec_phi_core(
    x: pl.Tensor[[32, 16], pl.FP32],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    first = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
    second = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Vec)
    a = pl.add(first, 1.25)
    b = pl.mul(second, -2.0)
    if pl.read(flag, [0]) > 0:
        selected = pl.yield_(a)
    else:
        selected = pl.yield_(b)
    pl.store(pl.add(selected, 0.5), [0, 0], out)
    # The selected result and both original arithmetic producers must survive.
    pl.store(a, [16, 0], out)
    pl.store(b, [32, 0], out)
    return out


@pl.jit
def vec_phi(
    x: pl.Tensor[[32, 16], pl.FP32],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    with pl.spmd(1):
        out = vec_phi_core(x, flag, out)
    return out


def _pattern(rows, columns, modulus, dtype):
    grid = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    return ((grid.remainder(modulus) - modulus // 2) / 8).to(dtype)


def _rotation_golden(tensors):
    return {"out": tensors["a"].float() @ tensors["b"].float()}


def _selection_golden(tensors):
    products = tensors["x"].float() @ tensors["y"].float()
    selected = products[:16] if tensors["flag"].item() > 0 else products[16:]
    return {"out": torch.cat((selected, products[:16], products[16:]), dim=0)}


def _vec_selection_golden(tensors):
    a = tensors["x"][:16].float() + 1.25
    b = tensors["x"][16:].float() * -2.0
    selected = a if tensors["flag"].item() > 0 else b
    return {"out": torch.cat((selected + 0.5, a, b), dim=0)}


@st.cases(
    *[
        st.case(
            entry,
            _pattern(512, 64, 23, torch.float16),
            _pattern(64, 16, 17, torch.float16),
            torch.zeros((512, 16), dtype=torch.float32),
            name=f"rotation_{planner.name.lower()}",
            golden=_rotation_golden,
            memory_planner=planner,
            rtol=1e-5,
            atol=1e-5,
        )
        for planner in (passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS)
    ],
    *[
        st.case(
            mat_phi,
            _pattern(32, 16, 29, torch.bfloat16),
            _pattern(16, 16, 13, torch.bfloat16),
            torch.tensor([flag], dtype=torch.int32),
            torch.zeros((48, 16), dtype=torch.float32),
            name=f"live_inputs_flag{flag}_{planner.name.lower()}",
            golden=_selection_golden,
            memory_planner=planner,
            rtol=1e-5,
            atol=1e-5,
        )
        for flag in (0, 1)
        for planner in (passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS)
    ],
)
@pytest.mark.platforms("a2a3", reason="Mat storage selection is verified on the A2/A3 backend.")
def test_live_mat_storage_selection(case_run):
    case_run.assert_passed()


@st.cases(
    *[
        st.case(
            vec_phi,
            _pattern(32, 16, 31, torch.float32),
            torch.tensor([flag], dtype=torch.int32),
            torch.zeros((48, 16), dtype=torch.float32),
            name=f"vec_live_producers_flag{flag}_{planner.name.lower()}",
            golden=_vec_selection_golden,
            memory_planner=planner,
            rtol=1e-6,
            atol=1e-6,
        )
        for flag in (0, 1)
        for planner in (passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS)
    ]
)
@pytest.mark.platforms("a2a3", reason="Vec storage selection is verified on the A2/A3 backend.")
def test_live_vec_storage_selection(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

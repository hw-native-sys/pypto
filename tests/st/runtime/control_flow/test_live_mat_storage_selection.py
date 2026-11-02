# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Issue #3049: selecting a Tile must preserve both live If inputs."""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto import passes


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
    # Both inputs remain live after selection, so their storage must stay distinct.
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


def _pattern(rows, columns, modulus, dtype):
    grid = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    return ((grid.remainder(modulus) - modulus // 2) / 8).to(dtype)


def _selection_golden(tensors):
    products = tensors["x"].float() @ tensors["y"].float()
    selected = products[:16] if tensors["flag"].item() > 0 else products[16:]
    return {"out": torch.cat((selected, products[:16], products[16:]), dim=0)}


@st.cases(
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
def test_if_selection_preserves_live_inputs(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

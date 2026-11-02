# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Capacity-triggered Vec materialization preserves every conditional path."""

import itertools

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto import passes


@pl.jit.incore
def conditional_vec_core(
    x: pl.Tensor[[64, 256], pl.FP32],
    flags: pl.Tensor[[4], pl.INT32],
    out: pl.InOut[pl.Tensor[[64, 256], pl.FP32]],
):
    # Four independently retained 64-KiB candidates exceed the 184-KiB UB
    # limit. The planner must use explicit branch copies and two buffers.
    valid_rows = pl.cast(pl.read(flags, [3]), pl.INDEX)
    initial = pl.tile.load(x, [0, 0], [64, 256], [valid_rows, 256], target_memory=pl.Mem.Vec)
    if pl.read(flags, [0]) > 0:
        first = pl.yield_(pl.add(initial, 1.0))
    else:
        first = pl.yield_(initial)
    if pl.read(flags, [1]) > 0:
        second = pl.yield_(first)
    else:
        second = pl.yield_(pl.mul(first, 2.0))
    if pl.read(flags, [2]) > 0:
        third = pl.yield_(pl.add(second, 4.0))
    else:
        third = pl.yield_(second)
    return pl.store(third, [0, 0], out)


@pl.jit
def conditional_vec(
    x: pl.Tensor[[64, 256], pl.FP32],
    flags: pl.Tensor[[4], pl.INT32],
    out: pl.InOut[pl.Tensor[[64, 256], pl.FP32]],
):
    with pl.spmd(1):
        out = conditional_vec_core(x, flags, out)
    return out


def _golden(tensors):
    value = tensors["x"].clone()
    first, second, third, valid_rows = tensors["flags"].tolist()
    if first:
        value = value + 1.0
    if not second:
        value = value * 2.0
    if third:
        value = value + 4.0
    # The caller initializes the untouched tail; InOut preserves it on device.
    mask = torch.arange(64).reshape(64, 1) < valid_rows
    return {"out": torch.where(mask, value, torch.zeros_like(value))}


@st.cases(
    *[
        st.case(
            conditional_vec,
            torch.arange(64 * 256, dtype=torch.float32).reshape(64, 256) / 128,
            torch.tensor([*flags, valid_rows], dtype=torch.int32),
            torch.zeros((64, 256), dtype=torch.float32),
            name=f"rows{valid_rows}_flags_" + "".join(map(str, flags)),
            golden=_golden,
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_buffer_ir=False,
            rtol=1e-5,
            atol=1e-5,
        )
        for flags in itertools.product((0, 1), repeat=3)
        for valid_rows in (63, 64)
    ],
)
@pytest.mark.platforms("a2a3", reason="Exercise the 184-KiB Vec capacity fallback on Ascend910.")
def test_conditional_vec_capacity(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

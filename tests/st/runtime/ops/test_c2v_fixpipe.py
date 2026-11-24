# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Non-square C2V layout regression: physical DN bytes must match the view."""

import pypto.language as pl
import pytest
import torch
from harness import st


@pl.jit
def column_view(
    a: pl.Tensor[[32, 32], pl.BF16],
    b: pl.Tensor[[32, 16], pl.BF16],
    out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
        acc = pl.matmul(lhs, rhs)
        dn = pl.tile.move(
            acc, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box
        )
        pl.store(pl.tile.transpose_view(dn), [0, 0], out)
    return out


@pl.jit
def row_transpose(
    a: pl.Tensor[[32, 32], pl.BF16],
    b: pl.Tensor[[32, 16], pl.BF16],
    out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
        acc = pl.matmul(lhs, rhs)
        nd = pl.tile.move(
            acc, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box
        )
        pl.store(pl.tile.transpose(nd, 0, 1), [0, 0], out)
    return out


def _case(kernel, signed, buffer_ir):
    if signed:
        a = (torch.arange(1024).reshape(32, 32) % 7 - 3).to(torch.bfloat16)
        b = (torch.arange(512).reshape(32, 16) % 11 - 5).to(torch.bfloat16)
    else:
        a = torch.eye(32, dtype=torch.bfloat16)
        b = torch.arange(512).reshape(32, 16).to(torch.bfloat16)
    # Accumulate exact small integers independently of the device matmul path.
    expected = (a.to(torch.int64) @ b.to(torch.int64)).T.to(torch.float32)
    return st.case(
        kernel,
        a,
        b,
        torch.empty(16, 32),
        name=f"{kernel.__name__}_signed={signed}_buffer_ir={buffer_ir}",
        golden=lambda _tensors: expected,
        platform="a5",
        enable_buffer_ir=buffer_ir,
        rtol=0,
        atol=0,
    )


@st.cases(
    *[
        _case(kernel, signed, buffer_ir)
        for kernel in (column_view, row_transpose)
        for signed in (False, True)
        for buffer_ir in (False, True)
    ]
)
def test_non_square_c2v(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

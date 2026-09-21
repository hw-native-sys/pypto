# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor col_sum binary strategy, including odd rows and live input reuse."""

import pypto.language as pl
import pytest
import torch
from harness import st


def _case(rows, cols, is_binary, dtype=pl.FP32):
    torch_dtype = torch.float32 if dtype == pl.FP32 else torch.int32

    @pl.jit
    def kernel(
        x: pl.Tensor[[rows, cols], dtype],
        out: pl.Out[pl.Tensor[[1, cols], dtype]],
        preserved: pl.Out[pl.Tensor[[rows, cols], dtype]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            score = pl.col_sum(x, is_binary=is_binary)
            out = pl.assemble(out, score, [0, 0])
            saved = pl.add(x, 1)
            preserved = pl.assemble(preserved, saved, [0, 0])
        return out, preserved

    generator = torch.Generator().manual_seed(rows * cols)
    x = (
        torch.randn(rows, cols, generator=generator)
        if dtype == pl.FP32
        else torch.randint(-100, 101, (rows, cols), generator=generator, dtype=torch_dtype)
    )
    return st.case(
        kernel,
        x,
        torch.zeros(1, cols, dtype=torch_dtype),
        torch.zeros_like(x),
        name=f"tensor_col_sum_{rows}x{cols}_{dtype}_{is_binary}",
        golden=lambda t: {"out": t["x"].sum(0, keepdim=True).to(torch_dtype), "preserved": t["x"] + 1},
        rtol=1e-5,
        atol=1e-5,
    )


@st.cases(
    *[
        _case(rows, cols, mode)
        for rows, cols in [(1, 16), (3, 16), (63, 192), (64, 192), (65, 64)]
        for mode in [False, True]
    ],
    _case(65, 64, True, pl.INT32),
)
def test_tensor_col_sum_binary(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

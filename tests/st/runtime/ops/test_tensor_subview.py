# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime acceptance for descriptor-only pitched GM row sampling."""

import pypto.language as pl
import pytest
import torch
from harness import st


@pl.jit.incore
def copy_rows(
    cache: pl.Tensor[[8, 512], pl.BF16], output: pl.Out[pl.Tensor[[2, 512], pl.BF16]]
) -> pl.Tensor[[2, 512], pl.BF16]:
    selected = pl.tensor.subview(cache, [2, 512], [1, 0], steps=[3, 1])
    pair = pl.load(selected, [0, 0], [2, 512])
    pl.store(pair, [0, 0], output)
    return output


@pl.jit
def subview_rows(
    cache: pl.Tensor[[8, 512], pl.BF16], output: pl.Out[pl.Tensor[[2, 512], pl.BF16]]
) -> pl.Tensor[[2, 512], pl.BF16]:
    return copy_rows(cache, output)


# BF16-exact values distinguish every row from every other row.
_CACHE = (16 * torch.arange(8)[:, None] + torch.arange(512)[None, :] % 16).to(torch.bfloat16)


@st.cases(
    st.case(
        subview_rows,
        _CACHE,
        torch.zeros((2, 512), dtype=torch.bfloat16),
        name="tensor_subview_rows_1_4",
        golden=lambda tensors: tensors["cache"][[1, 4]],
    )
)
def test_tensor_subview_rows(case_run):
    assert case_run.passed, case_run.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

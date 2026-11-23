# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Load two device-selected BF16 cache rows through one GM window (#3042)."""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto.pypto_core.passes import MemoryPlanner


@pl.jit.incore
def _load_rows(
    cache: pl.Tensor[[1024, 512], pl.BF16],
    rows: pl.Tensor[[2], pl.INT32],
    output: pl.Out[pl.Tensor[[2, 512], pl.BF16]],
):
    lo = pl.cast(pl.tensor.read(rows, [0]), pl.INDEX)
    hi = pl.cast(pl.tensor.read(rows, [1]), pl.INDEX)
    window = pl.tensor.view(cache, [2, 512], offsets=[lo, 0], steps=[hi - lo, 1])
    pair = pl.load(window, [0, 0], [2, 512])
    return pl.store(pair, [0, 0], output)


@pl.jit
def _entry(
    cache: pl.Tensor[[1024, 512], pl.BF16],
    rows: pl.Tensor[[2], pl.INT32],
    output: pl.Out[pl.Tensor[[2, 512], pl.BF16]],
):
    return _load_rows(cache, rows, output)


def _golden(tensors):
    return {"output": tensors["cache"].index_select(0, tensors["rows"].long())}


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(
    *(
        st.case(
            _entry,
            torch.randn((1024, 512), generator=torch.Generator().manual_seed(3042)).bfloat16(),
            torch.tensor([lo, hi], dtype=torch.int32),
            torch.empty((2, 512), dtype=torch.bfloat16),
            name=f"gm_window_rows_{lo}_{hi}",
            golden=_golden,
            memory_planner=MemoryPlanner.PYPTO,
            rtol=0.0,
            atol=0.0,
        )
        for lo, hi in ((0, 2), (7, 507), (1022, 1023))
    )
)
def test_dynamic_row_pair(case_run):
    case_run.assert_passed()

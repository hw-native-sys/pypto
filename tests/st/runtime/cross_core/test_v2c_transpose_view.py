# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression for a transposed Vec operand crossing to Cube (issue #2767)."""

import pypto.language as pl
import pytest

torch = pytest.importorskip("torch")

from harness import st  # noqa: E402

M = 32
N = 64
K = 128


@pl.jit
def v2c_transpose_view(
    keys: pl.Tensor[[N, K], pl.FP32],
    query: pl.Tensor[[M, K], pl.BF16],
    output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    with pl.spmd(1):
        block = pl.tile.get_block_idx()
        key_values = pl.load(keys, [0, 0], [N, K])
        key_tile = pl.cast(key_values, pl.BF16, mode="rint")
        query_tile = pl.load(query, [0, 0], [M, K])
        transposed_key = pl.tile.transpose_view(key_tile)
        key_alias = transposed_key
        scores = pl.matmul(query_tile, key_alias)
        pl.store(pl.maximum(scores, 0.0), [block, 0], output)
    return output


def _case():
    torch.manual_seed(0)
    keys = torch.randn(N, K, dtype=torch.float32)
    query = torch.randn(M, K, dtype=torch.bfloat16)
    output = torch.zeros(M, N, dtype=torch.float32)
    return st.case(
        v2c_transpose_view,
        keys,
        query,
        output,
        name="v2c_transpose_view",
        golden=lambda _: torch.relu(torch.matmul(query.float(), keys.to(torch.bfloat16).float().T)),
        rtol=2e-2,
        atol=2e-2,
    )


@st.cases(_case())
def test_v2c_transpose_view(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

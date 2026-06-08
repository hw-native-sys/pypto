# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression for gh#1649: sub-column slice of a matmul output's vec epilogue.

Slicing sub-columns of a matmul result that has been brought into Vec ND by a
vector epilogue used to make pypto emit a materializing ``pto.tmov`` whose
destination ``alloc_tile`` carried an unset ``valid_shape`` while the source
subview carried ``v_row=16, v_col=8``. ptoas (correctly) rejected the non-mat
``tmov`` with::

    error: 'pto.tmov' op expects A2/A3 non-mat tmov to use matching src/dst shapes

The eager materializing path was removed in #1636 (``tile.slice`` is now a pure
``pto.subview`` and the consumer reads the subview directly), which also fixes
this case. This test guards that the matmul -> vec-epilogue -> sub-column-slice
-> elementwise path compiles and runs correctly.
"""

import pypto.language as pl
import pytest
import torch

M, K, N, SUB = 16, 128, 32, 8


@pl.jit
def cube_slice(
    x: pl.Tensor[[M, K], pl.BF16],
    w: pl.Tensor[[N, K], pl.FP32],
    out: pl.Out[pl.Tensor[[M, SUB], pl.FP32]],
):
    for _ in pl.spmd(1, name_hint="cube_slice_blk"):
        x_f = pl.cast(x[0:M, 0:K], target_type=pl.FP32)
        w_s = pl.slice(w, [N, K], [0, 0], valid_shape=[24, K])
        acc = pl.matmul(x_f, w_s, b_trans=True, out_dtype=pl.FP32)  # [M,N] cube/L0C
        acc_nd = pl.mul(acc, 1.0)  # vec epilogue: land cube output in Vec ND
        sub = pl.mul(acc_nd[0:M, 0:SUB], 2.0)  # in-place sub-column slice -> used to fail
        out[0:M, 0:SUB] = sub
    return out


class TestSubviewTmovValidShape:
    """End-to-end @pl.jit compile + execute for the gh#1649 repro."""

    def test_cube_slice_subcolumn(self, test_config):
        cube_slice._cache.clear()

        torch.manual_seed(0)
        x = torch.randn((M, K), dtype=torch.bfloat16)
        w = torch.randn((N, K), dtype=torch.float32)
        out = torch.zeros((M, SUB), dtype=torch.float32)

        # Compiling alone reproduces the ptoas non-mat tmov rejection if present.
        cube_slice(x, w, out, config=test_config)

        # out = 2 * (x_f @ w^T)[:, :SUB]; columns 0..SUB-1 are within w_s valid 24 rows.
        expected = 2.0 * (x.float() @ w.t())[:, :SUB]
        assert torch.allclose(out, expected, rtol=2e-2, atol=2e-2), (
            f"max diff = {(out - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

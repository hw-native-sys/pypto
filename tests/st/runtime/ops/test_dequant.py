# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise per-row INT8 and INT16 dequantization on A5."""

import pypto.language as pl
import pytest
import torch
from harness import st


def _dequant_program(src_dtype, width, offset_rows):
    physical_width = (width + 63) // 64 * 64

    @pl.jit.incore
    def kernel(
        src: pl.Tensor[[16, width], src_dtype],
        scale: pl.Tensor[[16, 1], pl.FP32],
        offset: pl.Tensor[[offset_rows, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[16, width], pl.FP32]],
    ) -> pl.Tensor[[16, width], pl.FP32]:
        values = pl.tile.load(src, [0, 0], [16, physical_width], valid_shape=[16, width])
        scales = pl.tile.load(scale, [0, 0], [16, 1])
        offsets = pl.tile.load(offset, [0, 0], [offset_rows, 1])
        if offset_rows != 16:
            offsets = pl.tile.slice(offsets, [16, 1], [0, 0])
        return pl.tile.store(pl.tile.dequant(values, scales, offsets), [0, 0], out)

    @pl.jit
    def program(
        src: pl.Tensor[[16, width], src_dtype],
        scale: pl.Tensor[[16, 1], pl.FP32],
        offset: pl.Tensor[[offset_rows, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[16, width], pl.FP32]],
    ) -> pl.Tensor[[16, width], pl.FP32]:
        return kernel(src, scale, offset, out)

    return program


def _dequant_cases():
    scale = torch.tensor([0.25, -0.5, 2.0])[torch.arange(16) % 3].reshape(16, 1)
    for src_dtype, torch_dtype, width, offset_rows in (
        (pl.INT8, torch.int8, 32, 16),
        (pl.INT16, torch.int16, 96, 16),
        (pl.INT8, torch.int8, 64, 32),
    ):
        offset = (torch.arange(offset_rows, dtype=torch.float32) - 8.5).reshape(offset_rows, 1)
        bounds = torch.iinfo(torch_dtype)
        codes = torch.tensor([bounds.min, -83, -1, 0, 1, 77, bounds.max], dtype=torch_dtype)
        source = codes[torch.arange(16 * width) % codes.numel()].reshape(16, width)
        yield st.case(
            _dequant_program(src_dtype, width, offset_rows),
            source,
            scale,
            offset,
            torch.empty((16, width), dtype=torch.float32),
            name=f"{src_dtype}_{width}_columns",
            golden=lambda t: (t["src"].float() - t["offset"][:16]) * t["scale"],
            rtol=0,
            atol=0,
        )


@pytest.mark.platforms("a5")
@st.cases(*_dequant_cases())
def test_dequantization(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])

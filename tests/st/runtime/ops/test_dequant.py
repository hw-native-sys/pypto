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


def _dequant_program(src_dtype, width, offset_rows, coeff_cols, valid_rows):
    physical_width = (width + 63) // 64 * 64
    offset_cols = 16 if offset_rows == 1 else coeff_cols

    @pl.jit.incore
    def kernel(
        src: pl.Tensor[[valid_rows, width], src_dtype],
        scale: pl.Tensor[[16, coeff_cols], pl.FP32],
        offset: pl.Tensor[[offset_rows, offset_cols], pl.FP32],
        out: pl.Out[pl.Tensor[[valid_rows, width], pl.FP32]],
    ) -> pl.Tensor[[valid_rows, width], pl.FP32]:
        values = pl.tile.load(src, [0, 0], [16, physical_width], valid_shape=[valid_rows, width])
        if coeff_cols == 1:
            scales = pl.tile.load(scale, [0, 0], [16, 1], valid_shape=[valid_rows, 1])
            offsets = pl.tile.load(offset, [0, 0], [offset_rows, offset_cols])
            if offset_rows == 1:
                offsets = pl.tile.reshape(offsets, [16, 1])
            elif offset_rows != 16:
                if valid_rows == 16:
                    offsets = pl.tile.slice(offsets, [16, 1], [0, 0])
                else:
                    offsets = pl.tile.slice(offsets, [16, 1], [0, 0], valid_shape=[valid_rows, 1])
        elif valid_rows == 16:
            scales = pl.tile.load(scale, [0, 0], [16, coeff_cols], valid_shape=[16, 1])
            offsets = pl.tile.load(offset, [0, 0], [offset_rows, coeff_cols], valid_shape=[16, 1])
            scales = pl.tile.slice(scales, [16, 1], [0, 0])
            offsets = pl.tile.slice(offsets, [16, 1], [0, 0])
        else:
            scales = pl.tile.load(scale, [0, 0], [16, coeff_cols])
            offsets = pl.tile.load(offset, [0, 0], [offset_rows, coeff_cols])
            scales = pl.tile.slice(scales, [valid_rows, 1], [1, 1])
            offsets = pl.tile.slice(offsets, [valid_rows, 1], [1, 1])
        return pl.tile.store(pl.tile.dequant(values, scales, offsets), [0, 0], out)

    @pl.jit
    def program(
        src: pl.Tensor[[valid_rows, width], src_dtype],
        scale: pl.Tensor[[16, coeff_cols], pl.FP32],
        offset: pl.Tensor[[offset_rows, offset_cols], pl.FP32],
        out: pl.Out[pl.Tensor[[valid_rows, width], pl.FP32]],
    ) -> pl.Tensor[[valid_rows, width], pl.FP32]:
        return kernel(src, scale, offset, out)

    return program


def _dequant_cases():
    for src_dtype, torch_dtype, width, offset_rows, coeff_cols, valid_rows, suffix in (
        (pl.INT8, torch.int8, 32, 16, 1, 16, "columns"),
        (pl.INT16, torch.int16, 96, 16, 1, 16, "columns"),
        (pl.INT8, torch.int8, 64, 32, 1, 16, "column_view"),
        (pl.INT8, torch.int8, 64, 32, 1, 10, "explicit_valid_view"),
        (pl.INT8, torch.int8, 64, 16, 8, 16, "row_major_view"),
        (pl.INT8, torch.int8, 64, 16, 8, 9, "unaligned_odd_row_view"),
        (pl.INT8, torch.int8, 64, 1, 1, 16, "reshape_view"),
    ):
        scale = torch.tensor([0.25, -0.5, 2.0])[torch.arange(16) % 3].reshape(16, 1)
        scale = scale + torch.arange(coeff_cols)
        offset = (torch.arange(offset_rows, dtype=torch.float32) - 8.5).reshape(offset_rows, 1)
        offset = offset + torch.arange(coeff_cols)
        if offset_rows == 1:
            offset = (torch.arange(16, dtype=torch.float32) - 8.5).reshape(1, 16)
        bounds = torch.iinfo(torch_dtype)
        codes = torch.tensor([bounds.min, -83, -1, 0, 1, 77, bounds.max], dtype=torch_dtype)
        source = codes[torch.arange(valid_rows * width) % codes.numel()].reshape(valid_rows, width)
        start = 1 if coeff_cols > 1 and valid_rows != 16 else 0
        yield st.case(
            _dequant_program(src_dtype, width, offset_rows, coeff_cols, valid_rows),
            source,
            scale,
            offset,
            torch.empty((valid_rows, width), dtype=torch.float32),
            name=f"{src_dtype}_{width}_{suffix}",
            golden=lambda t, start=start, valid_rows=valid_rows, offset_rows=offset_rows: (
                t["src"].float()
                - (
                    t["offset"].reshape(16, 1)
                    if offset_rows == 1
                    else t["offset"][start : start + valid_rows, start : start + 1]
                )
            )
            * t["scale"][start : start + valid_rows, start : start + 1],
            rtol=0,
            atol=0,
        )


@pytest.mark.platforms("a5")
@st.cases(*_dequant_cases())
def test_dequantization(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])

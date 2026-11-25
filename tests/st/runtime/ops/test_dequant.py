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


def _dequant_transpose_program(base_rows, origin_row, origin_col, chained, window_rows):
    valid_cols = 2 if chained else 1

    @pl.jit.incore
    def kernel(
        src: pl.Tensor[[16, 64], pl.INT8],
        scale: pl.Tensor[[base_rows, base_rows], pl.FP32],
        offset: pl.Tensor[[base_rows, base_rows], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        values = pl.tile.load(src, [0, 0], [16, 64])
        scale_parent = pl.tile.transpose_view(pl.tile.load(scale, [0, 0], [base_rows, base_rows]))
        offset_parent = pl.tile.transpose_view(pl.tile.load(offset, [0, 0], [base_rows, base_rows]))
        scale_window = pl.tile.slice(
            scale_parent, [window_rows, 16], [origin_row, origin_col], valid_shape=[valid_cols, 16]
        )
        offset_window = pl.tile.slice(
            offset_parent, [window_rows, 16], [origin_row, origin_col], valid_shape=[valid_cols, 16]
        )
        scales = pl.tile.transpose_view(scale_window)
        offsets = pl.tile.transpose_view(offset_window)
        if chained:
            scales = pl.tile.slice(scales, [16, 1], [0, 1])
            offsets = pl.tile.slice(offsets, [16, 1], [0, 1])
        return pl.tile.store(pl.tile.dequant(values, scales, offsets), [0, 0], out)

    @pl.jit
    def program(
        src: pl.Tensor[[16, 64], pl.INT8],
        scale: pl.Tensor[[base_rows, base_rows], pl.FP32],
        offset: pl.Tensor[[base_rows, base_rows], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        return kernel(src, scale, offset, out)

    return program


def _dequant_transpose_cases():
    for base_rows, origin_row, origin_col, chained, window_rows in (
        (16, 0, 0, False, 16),
        (32, 3, 4, False, 16),
        (32, 8, 8, True, 16),
        (32, 3, 4, False, 1),
    ):
        indices = torch.arange(base_rows, dtype=torch.float32)
        scale = indices[:, None] * 100 + indices[None, :] + 1
        offset = torch.zeros_like(scale) if base_rows == 16 else indices[:, None] * 2 + indices[None, :]
        column = origin_row + int(chained)
        yield st.case(
            _dequant_transpose_program(base_rows, origin_row, origin_col, chained, window_rows),
            torch.ones((16, 64), dtype=torch.int8),
            scale,
            offset,
            torch.empty((16, 64), dtype=torch.float32),
            name=f"int8_transpose_view_{base_rows}_{origin_row}_{origin_col}_{chained}_{window_rows}",
            golden=lambda t, row=origin_col, col=column: (
                t["src"].float() - t["offset"][row : row + 16, col : col + 1]
            )
            * t["scale"][row : row + 16, col : col + 1],
            rtol=0,
            atol=0,
        )


def _dequant_vector_view_program(drop_axis):
    @pl.jit.incore
    def kernel(
        src: pl.Tensor[[16, 64], pl.INT8],
        scale: pl.Tensor[[32, 32], pl.FP32],
        offset: pl.Tensor[[32, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        values = pl.tile.load(src, [0, 0], [16, 64])
        scales = pl.tile.load(scale, [0, 0], [32, 32])
        offsets = pl.tile.load(offset, [0, 0], [32, 32])
        if drop_axis:
            scale_window = pl.tile.slice(scales, [16, 1], [4, 3], drop_dims=[1])
            offset_window = pl.tile.slice(offsets, [16, 1], [4, 3], drop_dims=[1])
            scales = pl.tile.transpose_view(scale_window)
            offsets = pl.tile.transpose_view(offset_window)
        else:
            scale_window = pl.tile.slice(scales, [1, 16], [3, 4])
            offset_window = pl.tile.slice(offsets, [1, 16], [3, 4])
            scales = pl.tile.reshape(scale_window, [16, 1])
            offsets = pl.tile.reshape(offset_window, [16, 1])
        return pl.tile.store(pl.tile.dequant(values, scales, offsets), [0, 0], out)

    @pl.jit
    def program(
        src: pl.Tensor[[16, 64], pl.INT8],
        scale: pl.Tensor[[32, 32], pl.FP32],
        offset: pl.Tensor[[32, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        return kernel(src, scale, offset, out)

    return program


def _dequant_vector_view_cases():
    indices = torch.arange(32, dtype=torch.float32)
    scale = indices[:, None] * 100 + indices[None, :] + 1
    offset = indices[:, None] * 2 + indices[None, :]
    for drop_axis in (False, True):
        window = (slice(4, 20), slice(3, 4)) if drop_axis else (slice(3, 4), slice(4, 20))
        yield st.case(
            _dequant_vector_view_program(drop_axis),
            torch.ones((16, 64), dtype=torch.int8),
            scale,
            offset,
            torch.empty((16, 64), dtype=torch.float32),
            name=f"int8_vector_view_drop_axis_{drop_axis}",
            golden=lambda t, window=window: (t["src"].float() - t["offset"][window].reshape(16, 1))
            * t["scale"][window].reshape(16, 1),
            rtol=0,
            atol=0,
        )


@pl.jit.incore
def _dequant_loop_kernel(
    src: pl.Tensor[[16, 64], pl.INT8],
    scale: pl.Tensor[[32, 1], pl.FP32],
    offset: pl.Tensor[[16, 1], pl.FP32],
    out: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
) -> pl.Tensor[[32, 64], pl.FP32]:
    values = pl.tile.load(src, [0, 0], [16, 64])
    initial = pl.tile.slice(pl.tile.load(scale, [0, 0], [32, 1]), [16, 1], [1, 0])
    offsets = pl.tile.load(offset, [0, 0], [16, 1])
    for step, (carried, out_iter) in pl.range(2, init_values=(initial, out)):
        result = pl.tile.dequant(values, carried, offsets)
        out_next = pl.tile.store(result, [step * 16, 0], out_iter)
        updated = pl.tile.adds(carried, 1.0)
        _, out_final = pl.yield_(updated, out_next)
    return out_final


@pl.jit
def _dequant_loop_program(
    src: pl.Tensor[[16, 64], pl.INT8],
    scale: pl.Tensor[[32, 1], pl.FP32],
    offset: pl.Tensor[[16, 1], pl.FP32],
    out: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
) -> pl.Tensor[[32, 64], pl.FP32]:
    return _dequant_loop_kernel(src, scale, offset, out)


def _dequant_loop_case():
    return st.case(
        _dequant_loop_program,
        torch.ones((16, 64), dtype=torch.int8),
        torch.ones((32, 1), dtype=torch.float32),
        torch.zeros((16, 1), dtype=torch.float32),
        torch.empty((32, 64), dtype=torch.float32),
        name="int8_loop_carried_coefficient",
        golden=lambda t: torch.cat((t["src"].float(), t["src"].float() * 2), dim=0),
        rtol=0,
        atol=0,
    )


@pytest.mark.platforms("a5")
@st.cases(*_dequant_cases(), *_dequant_transpose_cases(), *_dequant_vector_view_cases(), _dequant_loop_case())
def test_dequantization(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])

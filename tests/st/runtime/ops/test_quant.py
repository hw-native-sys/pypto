# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise A5 symmetric INT8 and asymmetric UINT8 quantization."""

import pypto.language as pl
import pytest
import torch
from harness import st


@pl.jit.incore
def _quant_sym_kernel(
    src: pl.Tensor[[16, 64], pl.FP32],
    scale: pl.Tensor[[16, 1], pl.FP32],
    out: pl.Out[pl.Tensor[[16, 64], pl.INT8]],
) -> pl.Tensor[[16, 64], pl.INT8]:
    values = pl.tile.load(src, [0, 0], [16, 64])
    scales = pl.tile.load(scale, [0, 0], [16, 1])
    return pl.tile.store(pl.tile.quant(values, scales), [0, 0], out)


@pl.jit.incore
def _quant_asym_kernel(
    src: pl.Tensor[[16, 64], pl.FP32],
    scale: pl.Tensor[[16, 1], pl.FP32],
    offset: pl.Tensor[[16, 1], pl.FP32],
    out: pl.Out[pl.Tensor[[16, 64], pl.UINT8]],
) -> pl.Tensor[[16, 64], pl.UINT8]:
    values = pl.tile.load(src, [0, 0], [16, 64])
    scales = pl.tile.load(scale, [0, 0], [16, 1])
    offsets = pl.tile.load(offset, [0, 0], [16, 1])
    return pl.tile.store(pl.tile.quant(values, scales, offsets), [0, 0], out)


@pl.jit
def _quant_sym(
    src: pl.Tensor[[16, 64], pl.FP32],
    scale: pl.Tensor[[16, 1], pl.FP32],
    out: pl.Out[pl.Tensor[[16, 64], pl.INT8]],
):
    return _quant_sym_kernel(src, scale, out)


@pl.jit
def _quant_asym(
    src: pl.Tensor[[16, 64], pl.FP32],
    scale: pl.Tensor[[16, 1], pl.FP32],
    offset: pl.Tensor[[16, 1], pl.FP32],
    out: pl.Out[pl.Tensor[[16, 64], pl.UINT8]],
):
    return _quant_asym_kernel(src, scale, offset, out)


def _quant_cases():
    # Exact half-integers exercise ties-to-even. Large magnitudes cover both
    # clamps, and varying the row coefficients checks the broadcast direction.
    values = torch.tensor([-640.0, -130.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127.5, 253.5, 640.0])
    src = values[torch.arange(16 * 64) % values.numel()].reshape(16, 64)
    scale = torch.tensor([1.0, 0.5, 2.0])[torch.arange(16) % 3].reshape(16, 1)
    offset = torch.tensor([0.0, -7.0, 11.5])[torch.arange(16) % 3].reshape(16, 1)
    yield st.case(
        _quant_sym,
        src,
        scale,
        torch.empty_like(src, dtype=torch.int8),
        name="int8_symmetric",
        golden=lambda t: (t["src"] * t["scale"]).round().clamp(-128, 127).to(torch.int8),
        rtol=0,
        atol=0,
    )
    yield st.case(
        _quant_asym,
        src,
        scale,
        offset,
        torch.empty_like(src, dtype=torch.uint8),
        name="uint8_asymmetric",
        golden=lambda t: (t["src"] * t["scale"] + t["offset"]).round().clamp(0, 255).to(torch.uint8),
        rtol=0,
        atol=0,
    )


@pytest.mark.platforms("a5")
@st.cases(*_quant_cases())
def test_quant(case_run):
    case_run.assert_passed()

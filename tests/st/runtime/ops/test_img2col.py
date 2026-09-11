# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Convolution composed from TIMG2COL, TEXTRACT and existing matmul operations."""

import pypto.language as pl
import pytest
import torch
import torch.nn.functional as F
from harness import st


def _conv_case(h, w, channels, kernel, stride, padding, dilation, dtype, *, tensor_level):
    kh, kw = kernel
    sh, sw = stride
    pt, pb, pad_left, pad_right = padding
    dh, dw = dilation
    oh = (h + pt + pb - dh * (kh - 1) - 1) // sh + 1
    ow = (w + pad_left + pad_right - dw * (kw - 1) - 1) // sw + 1
    outputs = 32
    c0 = 32 // torch.empty((), dtype=dtype).element_size()
    # TEXTRACT from NZ to Right requires K divisible by 16, including for FP32.
    k_tile = max(16, c0)
    packed_channels = (channels + c0 - 1) // c0 * c0
    reduction = packed_channels * kh * kw
    acc_dtype = pl.INT32 if dtype == torch.int8 else pl.FP32

    @pl.jit.incore
    def conv_tile(x: pl.Tensor, weight: pl.Tensor, out: pl.Out[pl.Tensor]):
        fmap = pl.load(x, [0, 0], [h * w, packed_channels], target_memory=pl.MemorySpace.Mat)
        weights = pl.load(weight, [0, 0], [reduction, outputs], target_memory=pl.MemorySpace.Mat)
        for m in pl.range(0, oh * ow, 16):
            acc = pl.tile.create([16, outputs], acc_dtype, target_memory=pl.MemorySpace.Acc)
            for k in pl.range(0, reduction, k_tile):
                lhs = pl.img2col(
                    fmap,
                    m,
                    k,
                    [16, k_tile],
                    image_shape=(h, w),
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
                rhs = pl.tile.extract(weights, k, 0, [k_tile, outputs], target_memory=pl.MemorySpace.Right)
                acc = pl.tile.matmul_acc(acc, lhs, rhs, init_cond=(k == 0))
            out = pl.store(acc, [m, 0], out)
        return out

    @pl.jit.incore
    def conv_tensor(x: pl.Tensor, weight: pl.Tensor, out: pl.Out[pl.Tensor]):
        fmap = pl.slice(x, [h * w, packed_channels], [0, 0])
        weights = pl.slice(weight, [reduction, outputs], [0, 0])
        for m in pl.range(0, oh * ow, 16):
            acc = pl.create_tensor([16, outputs], acc_dtype)
            for k in pl.range(0, reduction, k_tile):
                lhs = pl.img2col(
                    fmap,
                    m,
                    k,
                    [16, k_tile],
                    image_shape=(h, w),
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
                rhs = pl.slice(weights, [k_tile, outputs], [k, 0])
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k == 0))
            out = pl.assemble(out, acc, [m, 0])
        return out

    conv = conv_tensor if tensor_level else conv_tile

    @pl.jit
    def run_conv(x: pl.Tensor, weight: pl.Tensor, out: pl.Out[pl.Tensor]):
        return conv(x, weight, out)

    generator = torch.Generator().manual_seed(2665)
    image = torch.randint(-2, 3, (1, channels, h, w), generator=generator).to(dtype)
    weight = torch.randint(-2, 3, (outputs, channels, kh, kw), generator=generator).to(dtype)
    # NZ [HW,C] has the same bytes as NC1HWC0. Hardware K order is C1,KH,KW,C0.
    # Zero-filled channel padding also permits an RGB convolution to use C0 packing.
    channel_padding = (0, 0, 0, 0, 0, packed_channels - channels)
    x = F.pad(image, channel_padding).permute(0, 2, 3, 1).reshape(h * w, packed_channels).contiguous()
    packed_weight = (
        F.pad(weight, channel_padding)
        .reshape(outputs, packed_channels // c0, c0, kh, kw)
        .permute(1, 3, 4, 2, 0)
        .reshape(reduction, outputs)
        .contiguous()
    )
    expected = (
        F.conv2d(
            F.pad(image.float(), (pad_left, pad_right, pt, pb)),
            weight.float(),
            stride=stride,
            dilation=dilation,
        )
        .permute(0, 2, 3, 1)
        .reshape(oh * ow, outputs)
    )
    output_dtype = torch.int32 if dtype == torch.int8 else torch.float32
    return st.case(
        run_conv,
        x,
        packed_weight,
        torch.zeros((oh * ow, outputs), dtype=output_dtype),
        name=(
            f"img2col_{'tensor' if tensor_level else 'tile'}_conv_{h}x{w}_c{channels}_k{kh}x{kw}_s{sh}x{sw}_"
            f"p{pt}_{pb}_{pad_left}_{pad_right}_d{dh}x{dw}_{dtype}"
        ),
        golden=lambda _: expected.to(output_dtype),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.platforms("a2a3", reason="TIMG2COL lowering currently supports A2/A3 device execution")
@st.cases(
    *[
        _conv_case(h, w, c, kernel, stride, padding, dilation, dtype, tensor_level=tensor_level)
        for tensor_level in (False, True)
        for h, w, c, kernel, stride, padding, dilation, dtype in (
            (8, 8, 3, (3, 3), (2, 2), (1, 1, 1, 1), (1, 1), torch.float16),
            (8, 8, 32, (3, 3), (1, 1), (1, 1, 1, 1), (1, 1), torch.float16),
            (8, 16, 32, (3, 3), (2, 2), (1, 1, 1, 1), (1, 1), torch.float16),
            (8, 8, 32, (3, 3), (1, 1), (2, 2, 2, 2), (2, 2), torch.bfloat16),
            (8, 8, 32, (3, 3), (1, 1), (0, 2, 1, 1), (1, 1), torch.bfloat16),
            (8, 8, 32, (1, 1), (1, 1), (0, 0, 0, 0), (1, 1), torch.int8),
            (8, 8, 16, (1, 1), (1, 1), (0, 0, 0, 0), (1, 1), torch.float32),
        )
    ]
)
def test_img2col_convolution(case_run):
    """Check spatial/channel ordering, dynamic M/K positions and accumulation."""
    case_run.assert_passed()


@pl.jit.incore
def _causal_conv3d(x: pl.Tensor, weight: pl.Tensor, out: pl.Out[pl.Tensor]):
    # x contains two leading zero frames: temporal padding is ordinary GM data.
    for t in pl.range(3):
        acc = pl.tile.create([16, 32], pl.FP32, target_memory=pl.MemorySpace.Acc)
        for kt in pl.range(3):
            fmap = pl.load(x, [(t + kt) * 16, 0], [16, 32], target_memory=pl.MemorySpace.Mat)
            weights = pl.load(weight, [kt * 288, 0], [288, 32], target_memory=pl.MemorySpace.Mat)
            for k in pl.range(0, 288, 16):
                lhs = pl.img2col(
                    fmap,
                    0,
                    k,
                    [16, 16],
                    image_shape=(4, 4),
                    kernel_size=(3, 3),
                    padding=(1, 1, 1, 1),
                )
                rhs = pl.tile.extract(weights, k, 0, [16, 32], target_memory=pl.MemorySpace.Right)
                acc = pl.tile.matmul_acc(acc, lhs, rhs, init_cond=((kt == 0) and (k == 0)))
        out = pl.store(acc, [t * 16, 0], out)
    return out


@pl.jit
def _run_causal_conv3d(x: pl.Tensor, weight: pl.Tensor, out: pl.Out[pl.Tensor]):
    return _causal_conv3d(x, weight, out)


def _causal_case():
    generator = torch.Generator().manual_seed(2731)
    image = torch.randint(-2, 3, (1, 32, 3, 4, 4), generator=generator).to(torch.float16)
    weight = torch.randint(-2, 3, (32, 32, 3, 3, 3), generator=generator).to(torch.float16)
    padded = F.pad(image, (0, 0, 0, 0, 2, 0))
    x = padded.permute(0, 2, 3, 4, 1).reshape(80, 32).contiguous()
    packed = weight.reshape(32, 2, 16, 3, 3, 3).permute(3, 1, 4, 5, 2, 0).reshape(864, 32).contiguous()
    expected = (
        F.conv3d(F.pad(image.float(), (1, 1, 1, 1, 2, 0)), weight.float())
        .permute(0, 2, 3, 4, 1)
        .reshape(48, 32)
    )
    return st.case(
        _run_causal_conv3d,
        x,
        packed,
        torch.zeros((48, 32)),
        name="img2col_causal_conv3d",
        golden=lambda _: expected,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.platforms("a2a3", reason="TIMG2COL lowering currently supports A2/A3 device execution")
@st.cases(_causal_case())
def test_img2col_causal_convolution(case_run):
    """Each output frame accumulates only its current and two preceding frames."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

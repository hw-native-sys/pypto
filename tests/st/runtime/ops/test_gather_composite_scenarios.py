# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Conv2D, causal Conv3D, and bilinear GridSample coverage for issue #2665.

Gather indices and interpolation weights are computed on-device; PyTorch goldens check
gathered patches and final outputs. These are correctness tests, not library ops.
"""

import pypto.language as pl
import pytest
import torch
import torch.nn.functional as F
from harness import st

pytestmark = pytest.mark.platforms(
    "a2a3", "a2a3sim", reason="Issue #2665 gather-dependent scenarios; A5 is out of scope."
)


def _conv_kernel(shape: tuple[int, ...], kernel_depth: int, stride: int, padding: int, replicate_past: bool):
    batches, channels, depth, height, width = shape
    out_height = (height + 2 * padding - 3) // stride + 1
    out_width = (width + 2 * padding - 3) // stride + 1
    positions = batches * depth * out_height * out_width
    reduction = channels * kernel_depth * 3 * 3
    temporal_mask = 0.0 if replicate_past else 1.0

    @pl.jit
    def convolution(
        src: pl.Tensor,
        weight: pl.Tensor,
        patches: pl.InOut[pl.Tensor],
        out: pl.Out[pl.Tensor],
    ):
        _, padded_positions = patches.shape
        out_channels, padded_reduction = weight.shape
        with pl.at(level=pl.Level.CORE_GROUP):
            p = pl.cast(pl.arange(0, [1, padded_positions]), pl.FP32)
            batch = pl.cast(
                pl.cast(pl.div(p, depth * out_height * out_width), pl.INT32, mode="floor"), pl.FP32
            )
            spatial = pl.sub(p, pl.mul(batch, depth * out_height * out_width))
            oz = pl.cast(pl.cast(pl.div(spatial, out_height * out_width), pl.INT32, mode="floor"), pl.FP32)
            spatial = pl.sub(spatial, pl.mul(oz, out_height * out_width))
            oy = pl.cast(pl.cast(pl.div(spatial, out_width), pl.INT32, mode="floor"), pl.FP32)
            ox = pl.sub(spatial, pl.mul(oy, out_width))
            base_y = pl.sub(pl.mul(oy, stride), padding)
            base_x = pl.sub(pl.mul(ox, stride), padding)
            valid_position = pl.minimum(pl.maximum(pl.add(pl.neg(p), positions), 0.0), 1.0)

            for k in pl.range(reduction):
                channel = k // (kernel_depth * 9)
                kz = (k // 9) % kernel_depth
                ky = (k // 3) % 3
                kx = k % 3
                z = pl.add(oz, pl.cast(pl.cast(kz - (kernel_depth - 1), pl.INT32), pl.FP32))
                y = pl.add(base_y, pl.cast(pl.cast(ky, pl.INT32), pl.FP32))
                x = pl.add(base_x, pl.cast(pl.cast(kx, pl.INT32), pl.FP32))
                # Clamp before linearization to prevent cross-row aliases.
                safe_batch = pl.minimum(batch, batches - 1)
                safe_z = pl.minimum(pl.maximum(z, 0.0), depth - 1)
                safe_y = pl.minimum(pl.maximum(y, 0.0), height - 1)
                safe_x = pl.minimum(pl.maximum(x, 0.0), width - 1)
                outside = pl.add(pl.mul(pl.abs(pl.sub(z, safe_z)), temporal_mask), pl.abs(pl.sub(y, safe_y)))
                outside = pl.add(outside, pl.abs(pl.sub(x, safe_x)))
                valid = pl.mul(valid_position, pl.maximum(pl.add(pl.neg(outside), 1.0), 0.0))
                index = pl.add(pl.mul(safe_batch, channels), pl.cast(pl.cast(channel, pl.INT32), pl.FP32))
                index = pl.add(pl.mul(index, depth), safe_z)
                index = pl.add(pl.mul(index, height), safe_y)
                index = pl.add(pl.mul(index, width), safe_x)
                values = pl.gather(src, index=pl.cast(index, pl.INT32))
                values = pl.cast(pl.mul(pl.cast(values, pl.FP32), valid), pl.FP16)
                patches = pl.assemble(patches, values, [k, 0])

        with pl.at(level=pl.Level.CORE_GROUP):
            lhs = pl.load(weight, [0, 0], [out_channels, padded_reduction], target_memory=pl.Mem.Mat)
            rhs = pl.load(patches, [0, 0], [padded_reduction, padded_positions], target_memory=pl.Mem.Mat)
            lhs = pl.move(lhs, target_memory=pl.Mem.Left)
            rhs = pl.move(rhs, target_memory=pl.Mem.Right)
            pl.store(pl.matmul(lhs, rhs), [0, 0], out)
        return patches, out

    return convolution


def _conv_case(causal: bool, padding: int = 1, future_impulse: bool = False, replicate_past: bool = False):
    shape = (2, 2, 3, 4, 4) if causal else (2, 2, 1, 9, 9)
    batches, channels, depth, height, width = shape
    kernel_depth, stride = (3, 1) if causal else (1, 2)
    out_height = (height + 2 * padding - 3) // stride + 1
    out_width = (width + 2 * padding - 3) // stride + 1
    positions = batches * depth * out_height * out_width
    reduction = channels * kernel_depth * 9
    padded_positions = (positions + 15) // 16 * 16
    padded_reduction = (reduction + 15) // 16 * 16
    out_channels = 16
    generator = torch.Generator().manual_seed(2665)
    src = (torch.randn(shape, generator=generator) * 0.25).half()
    if future_impulse:
        src.zero_()
        src[:, :, -1] = torch.randn((batches, channels, height, width), generator=generator).half()
    weights = (torch.randn((out_channels, reduction), generator=generator) * 0.25).half()
    packed_weight = torch.zeros((out_channels, padded_reduction), dtype=torch.float16)
    packed_weight[:, :reduction] = weights

    def golden(tensors):
        image = tensors["src"].reshape(shape).float()
        weight = tensors["weight"][:, :reduction].reshape(out_channels, channels, kernel_depth, 3, 3).float()
        temporal_mode = "replicate" if replicate_past else "constant"
        past_padded = F.pad(image, (0, 0, 0, 0, kernel_depth - 1, 0), mode=temporal_mode)
        padded = F.pad(past_padded, (padding, padding, padding, padding))
        windows = padded.unfold(2, kernel_depth, 1).unfold(3, 3, stride).unfold(4, 3, stride)
        columns = windows.permute(1, 5, 6, 7, 0, 2, 3, 4).reshape(reduction, positions)
        expected_patches = torch.zeros_like(tensors["patches"])
        expected_patches[:reduction, :positions] = columns.half()
        if causal:
            result = F.conv3d(padded, weight)
            if future_impulse:
                assert torch.count_nonzero(result[:, :, :-1]) == 0
        else:
            result = F.conv2d(image[:, :, 0], weight[:, :, 0], stride=2, padding=padding)
        expected = torch.zeros_like(tensors["out"])
        expected[:, :positions] = result.transpose(0, 1).reshape(out_channels, positions)
        return {"patches": expected_patches, "out": expected}

    def compare(actual, expected):
        # Gather is exact; only matmul needs a reduction-order tolerance.
        torch.testing.assert_close(actual["patches"], expected["patches"], rtol=0, atol=0)
        torch.testing.assert_close(actual["out"], expected["out"], rtol=1e-4, atol=1e-4)

    name = (
        f"causal_conv3d_replicate{replicate_past}_future{future_impulse}"
        if causal
        else f"conv2d_stride2_padding{padding}"
    )
    return st.case(
        _conv_kernel(shape, kernel_depth, stride, padding, replicate_past),
        src.flatten().contiguous(),
        packed_weight,
        torch.zeros((padded_reduction, padded_positions), dtype=torch.float16),
        torch.zeros((out_channels, padded_positions), dtype=torch.float32),
        name=name,
        golden=golden,
        compare=compare,
    )


@st.cases(_conv_case(False, padding=0), _conv_case(False, padding=1))
def test_conv2d_stride2(case_run):
    """Multi-batch/channel im2col and complete convolution, with and without padding."""
    case_run.assert_passed()


@st.cases(
    *(
        _conv_case(True, future_impulse=future, replicate_past=replicate)
        for replicate in (False, True)
        for future in (False, True)
    )
)
def test_causal_conv3d(case_run):
    """Temporal left padding is causal: a last-frame impulse cannot affect past outputs."""
    case_run.assert_passed()


def _grid_kernel(align_corners: bool, border: bool):
    batches, channels, height, width = 2, 2, 5, 7
    scale_x = (width - 1) / 2.0 if align_corners else width / 2.0
    scale_y = (height - 1) / 2.0 if align_corners else height / 2.0
    # The one-pixel halo preserves zero-padding interpolation and bounds the INT32 cast.
    lower_bound = 0.0 if border else -1.0
    upper_x = float(width - 1) if border else float(width)
    upper_y = float(height - 1) if border else float(height)

    @pl.jit
    def grid_sample(src: pl.Tensor, grid: pl.Tensor, out: pl.Out[pl.Tensor]):
        _, _, points = grid.shape
        for batch in pl.range(batches):
            for channel in pl.range(channels):
                with pl.at(level=pl.Level.CORE_GROUP):
                    gx = pl.reshape(pl.slice(grid, [1, 1, points], [batch, 0, 0]), [1, points])
                    gy = pl.reshape(pl.slice(grid, [1, 1, points], [batch, 1, 0]), [1, points])
                    pixel_x = pl.add(pl.mul(gx, scale_x), (width - 1) / 2.0)
                    pixel_y = pl.add(pl.mul(gy, scale_y), (height - 1) / 2.0)
                    x = pl.minimum(pl.maximum(pixel_x, lower_bound), upper_x)
                    y = pl.minimum(pl.maximum(pixel_y, lower_bound), upper_y)
                    x0 = pl.cast(pl.cast(x, pl.INT32, mode="floor"), pl.FP32)
                    y0 = pl.cast(pl.cast(y, pl.INT32, mode="floor"), pl.FP32)
                    fraction_x = pl.sub(x, x0)
                    fraction_y = pl.sub(y, y0)
                    result = pl.full([1, points], dtype=pl.FP32, value=0.0)
                    for dy in pl.unroll(2):
                        for dx in pl.unroll(2):
                            dx_f = pl.cast(pl.cast(dx, pl.INT32), pl.FP32)
                            dy_f = pl.cast(pl.cast(dy, pl.INT32), pl.FP32)
                            ix = pl.add(x0, dx_f)
                            iy = pl.add(y0, dy_f)
                            safe_x = pl.minimum(pl.maximum(ix, 0.0), width - 1)
                            safe_y = pl.minimum(pl.maximum(iy, 0.0), height - 1)
                            outside = pl.add(pl.abs(pl.sub(ix, safe_x)), pl.abs(pl.sub(iy, safe_y)))
                            valid = pl.maximum(pl.add(pl.neg(outside), 1.0), 0.0)
                            index = pl.add(pl.mul(safe_y, width), safe_x)
                            offset = pl.cast((batch * channels + channel) * height * width, pl.INT32)
                            index = pl.add(index, pl.cast(offset, pl.FP32))
                            value = pl.gather(src, index=pl.cast(index, pl.INT32))
                            wx = pl.add(pl.mul(fraction_x, 2.0 * dx_f - 1.0), 1.0 - dx_f)
                            wy = pl.add(pl.mul(fraction_y, 2.0 * dy_f - 1.0), 1.0 - dy_f)
                            weighted = pl.mul(pl.mul(value, valid), pl.mul(wx, wy))
                            result = pl.add(result, weighted)
                    out = pl.assemble(out, pl.reshape(result, [1, 1, points]), [batch, channel, 0])
        return out

    return grid_sample


def _grid_case(align_corners: bool, padding_mode: str):
    generator = torch.Generator().manual_seed(2665)
    image = torch.randn((2, 2, 5, 7), generator=generator)
    # Distinct rows and images expose aliases from incorrectly flattened coordinates.
    image += torch.arange(20, dtype=torch.float32).reshape(2, 2, 5, 1)
    grid = torch.rand((2, 4, 8, 2), generator=generator) * 3.0 - 1.5
    boundary_points = torch.tensor(
        [
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [0.0, 0.0],
            [0.15, -0.35],
            [1.25, 0.0],
            [-1.25, 0.0],
            [0.0, 1.25],
            [0.0, -1.25],
            [2.0, 2.0],
            [-2.0, -2.0],
            [1.0001, 0.0],
            [-1.0001, 0.0],
            [0.0, 1.0001],
            [0.0, -1.0001],
        ],
        dtype=torch.float32,
    )
    grid.reshape(2, 32, 2)[:, :16] = boundary_points

    def golden(tensors):
        original_grid = tensors["grid"].transpose(1, 2).reshape(2, 4, 8, 2)
        result = F.grid_sample(
            tensors["src"].reshape(2, 2, 5, 7),
            original_grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return result.reshape(2, 2, 32)

    return st.case(
        _grid_kernel(align_corners, padding_mode == "border"),
        image.flatten().contiguous(),
        grid.reshape(2, 32, 2).transpose(1, 2).contiguous(),
        torch.zeros((2, 2, 32), dtype=torch.float32),
        name=f"grid_sample_bilinear_{padding_mode}_align{align_corners}",
        golden=golden,
        rtol=1e-5,
        atol=1e-5,
    )


@st.cases(*(_grid_case(align, mode) for align in (False, True) for mode in ("zeros", "border")))
def test_grid_sample_bilinear(case_run):
    """Runtime normalized coordinates, four neighbors, fractional weights, and boundary modes."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manual and whole-function AutoTile forms of four vector kernels.

The manual kernels fix their core grid and UB chunks explicitly.  Their paired
``AutoTile*Program`` classes express only the tensor DAG and ask the compiler to
choose those scheduling details.  The pairs intentionally have identical
shapes, dtypes, formulas, and output contracts so they can be compared on
device without changing the algorithm.

The manual forms mirror the maintained PyPTO/PyPTO-Lib softmax, RMSNorm,
LayerNorm, and SiLU examples.  See
``tests/st/examples/02_intermediate/test_auto_tile_vector_performance.py`` for
register-once correctness and performance comparisons.
"""

import pypto.language as pl

SOFTMAX_ROWS = 512
SOFTMAX_COLS = 256
SOFTMAX_ROW_TILE = 64

RMS_ROWS = 512
RMS_HIDDEN = 512
RMS_ROW_TILE = 64
RMS_HIDDEN_TILE = 64
RMS_EPS = 1.0e-6

LAYER_ROWS = 512
LAYER_HIDDEN = 256
LAYER_ROW_TILE = 32
LAYER_EPS = 1.0e-5

SILU_ROWS = 512
SILU_COLS = 256
SILU_ROW_TILE = 32

_EXAMPLE_WARMUP = 3
_EXAMPLE_ROUNDS = 10


@pl.jit
def manual_softmax(
    x: pl.Tensor[[SOFTMAX_ROWS, SOFTMAX_COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[SOFTMAX_ROWS, SOFTMAX_COLS], pl.FP32]],
):
    """Explicitly tile stable row softmax across eight core groups."""
    for row in pl.parallel(0, SOFTMAX_ROWS, SOFTMAX_ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_rows"):
            tile_x = x[row : row + SOFTMAX_ROW_TILE, :]
            maximum = pl.row_max(tile_x)
            shifted = pl.row_expand_sub(tile_x, maximum)
            exponent = pl.exp(shifted)
            total = pl.row_sum(exponent)
            y[row : row + SOFTMAX_ROW_TILE, :] = pl.row_expand_div(exponent, total)
    return y


@pl.program
class AutoTileSoftmaxProgram:
    """The same stable softmax as :func:`manual_softmax`, without a schedule."""

    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        x: pl.Tensor[[SOFTMAX_ROWS, SOFTMAX_COLS], pl.FP32],
        y: pl.Out[pl.Tensor[[SOFTMAX_ROWS, SOFTMAX_COLS], pl.FP32]],
    ) -> pl.Tensor[[SOFTMAX_ROWS, SOFTMAX_COLS], pl.FP32]:
        maximum: pl.Tensor[[SOFTMAX_ROWS, 1], pl.FP32] = pl.row_max(x)
        shifted: pl.Tensor[[SOFTMAX_ROWS, SOFTMAX_COLS], pl.FP32] = pl.sub(x, maximum)
        exponent: pl.Tensor[[SOFTMAX_ROWS, SOFTMAX_COLS], pl.FP32] = pl.exp(shifted)
        total: pl.Tensor[[SOFTMAX_ROWS, 1], pl.FP32] = pl.row_sum(exponent)
        y = pl.div(exponent, total)
        return y


@pl.jit
def manual_rms_norm(
    x: pl.Tensor[[RMS_ROWS, RMS_HIDDEN], pl.FP32],
    gamma: pl.Tensor[[1, RMS_HIDDEN], pl.FP32],
    y: pl.Out[pl.Tensor[[RMS_ROWS, RMS_HIDDEN], pl.FP32]],
):
    """Explicitly tile a two-pass RMSNorm across rows and hidden chunks."""
    for row in pl.parallel(0, RMS_ROWS, RMS_ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_norm_rows"):
            square_sum = pl.full([1, RMS_ROW_TILE], dtype=pl.FP32, value=0.0)
            for hidden_block in pl.range(RMS_HIDDEN // RMS_HIDDEN_TILE):
                hidden = hidden_block * RMS_HIDDEN_TILE
                tile_x = x[row : row + RMS_ROW_TILE, hidden : hidden + RMS_HIDDEN_TILE]
                square = pl.mul(tile_x, tile_x)
                partial = pl.reshape(pl.row_sum(square), [1, RMS_ROW_TILE])
                square_sum = pl.add(square_sum, partial)

            mean_square = pl.mul(square_sum, 1.0 / RMS_HIDDEN)
            inverse_rms = pl.rsqrt(pl.add(mean_square, RMS_EPS))
            inverse_rms = pl.reshape(inverse_rms, [RMS_ROW_TILE, 1])

            for hidden_block in pl.range(RMS_HIDDEN // RMS_HIDDEN_TILE):
                hidden = hidden_block * RMS_HIDDEN_TILE
                tile_x = x[row : row + RMS_ROW_TILE, hidden : hidden + RMS_HIDDEN_TILE]
                tile_gamma = gamma[:, hidden : hidden + RMS_HIDDEN_TILE]
                normalized = pl.row_expand_mul(tile_x, inverse_rms)
                y[row : row + RMS_ROW_TILE, hidden : hidden + RMS_HIDDEN_TILE] = pl.col_expand_mul(
                    normalized, tile_gamma
                )
    return y


@pl.program
class AutoTileRmsNormProgram:
    """The same RMSNorm as :func:`manual_rms_norm`, without a schedule."""

    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        x: pl.Tensor[[RMS_ROWS, RMS_HIDDEN], pl.FP32],
        gamma: pl.Tensor[[1, RMS_HIDDEN], pl.FP32],
        y: pl.Out[pl.Tensor[[RMS_ROWS, RMS_HIDDEN], pl.FP32]],
    ) -> pl.Tensor[[RMS_ROWS, RMS_HIDDEN], pl.FP32]:
        square: pl.Tensor[[RMS_ROWS, RMS_HIDDEN], pl.FP32] = pl.mul(x, x)
        total: pl.Tensor[[RMS_ROWS, 1], pl.FP32] = pl.row_sum(square)
        mean_square: pl.Tensor[[RMS_ROWS, 1], pl.FP32] = pl.mul(total, 1.0 / RMS_HIDDEN)
        inverse_rms: pl.Tensor[[RMS_ROWS, 1], pl.FP32] = pl.rsqrt(pl.add(mean_square, RMS_EPS))
        normalized: pl.Tensor[[RMS_ROWS, RMS_HIDDEN], pl.FP32] = pl.mul(x, inverse_rms)
        y = pl.mul(normalized, gamma)
        return y


@pl.jit
def manual_layer_norm(
    x: pl.Tensor[[LAYER_ROWS, LAYER_HIDDEN], pl.FP32],
    gamma: pl.Tensor[[1, LAYER_HIDDEN], pl.FP32],
    beta: pl.Tensor[[1, LAYER_HIDDEN], pl.FP32],
    y: pl.Out[pl.Tensor[[LAYER_ROWS, LAYER_HIDDEN], pl.FP32]],
):
    """Explicitly tile LayerNorm across independent row groups."""
    for row in pl.parallel(0, LAYER_ROWS, LAYER_ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="layer_norm_rows"):
            tile_x = x[row : row + LAYER_ROW_TILE, :]
            tile_gamma = gamma[:, :]
            tile_beta = beta[:, :]

            mean = pl.row_sum(pl.mul(tile_x, 1.0 / LAYER_HIDDEN))
            centered = pl.row_expand_sub(tile_x, mean)
            variance = pl.mul(pl.row_sum(pl.mul(centered, centered)), 1.0 / LAYER_HIDDEN)
            normalized = pl.row_expand_div(centered, pl.sqrt(pl.add(variance, LAYER_EPS)))
            scaled = pl.col_expand_mul(normalized, tile_gamma)
            y[row : row + LAYER_ROW_TILE, :] = pl.col_expand_add(scaled, tile_beta)
    return y


@pl.program
class AutoTileLayerNormProgram:
    """The same LayerNorm as :func:`manual_layer_norm`, without a schedule."""

    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        x: pl.Tensor[[LAYER_ROWS, LAYER_HIDDEN], pl.FP32],
        gamma: pl.Tensor[[1, LAYER_HIDDEN], pl.FP32],
        beta: pl.Tensor[[1, LAYER_HIDDEN], pl.FP32],
        y: pl.Out[pl.Tensor[[LAYER_ROWS, LAYER_HIDDEN], pl.FP32]],
    ) -> pl.Tensor[[LAYER_ROWS, LAYER_HIDDEN], pl.FP32]:
        scaled_x: pl.Tensor[[LAYER_ROWS, LAYER_HIDDEN], pl.FP32] = pl.mul(x, 1.0 / LAYER_HIDDEN)
        mean: pl.Tensor[[LAYER_ROWS, 1], pl.FP32] = pl.row_sum(scaled_x)
        centered: pl.Tensor[[LAYER_ROWS, LAYER_HIDDEN], pl.FP32] = pl.sub(x, mean)
        squared: pl.Tensor[[LAYER_ROWS, LAYER_HIDDEN], pl.FP32] = pl.mul(centered, centered)
        square_sum: pl.Tensor[[LAYER_ROWS, 1], pl.FP32] = pl.row_sum(squared)
        variance: pl.Tensor[[LAYER_ROWS, 1], pl.FP32] = pl.mul(square_sum, 1.0 / LAYER_HIDDEN)
        std: pl.Tensor[[LAYER_ROWS, 1], pl.FP32] = pl.sqrt(pl.add(variance, LAYER_EPS))
        normalized: pl.Tensor[[LAYER_ROWS, LAYER_HIDDEN], pl.FP32] = pl.div(centered, std)
        scaled: pl.Tensor[[LAYER_ROWS, LAYER_HIDDEN], pl.FP32] = pl.mul(normalized, gamma)
        y = pl.add(scaled, beta)
        return y


@pl.jit
def manual_silu(
    x: pl.Tensor[[SILU_ROWS, SILU_COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[SILU_ROWS, SILU_COLS], pl.FP32]],
):
    """Explicitly tile SiLU across independent row groups."""
    for row in pl.parallel(0, SILU_ROWS, SILU_ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu_rows"):
            tile_x = x[row : row + SILU_ROW_TILE, :]
            denominator = pl.add(pl.exp(pl.mul(tile_x, -1.0)), 1.0)
            y[row : row + SILU_ROW_TILE, :] = pl.mul(tile_x, pl.recip(denominator))
    return y


@pl.program
class AutoTileSiluProgram:
    """The same SiLU as :func:`manual_silu`, without a schedule."""

    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        x: pl.Tensor[[SILU_ROWS, SILU_COLS], pl.FP32],
        y: pl.Out[pl.Tensor[[SILU_ROWS, SILU_COLS], pl.FP32]],
    ) -> pl.Tensor[[SILU_ROWS, SILU_COLS], pl.FP32]:
        negative: pl.Tensor[[SILU_ROWS, SILU_COLS], pl.FP32] = pl.mul(x, -1.0)
        exponential: pl.Tensor[[SILU_ROWS, SILU_COLS], pl.FP32] = pl.exp(negative)
        denominator: pl.Tensor[[SILU_ROWS, SILU_COLS], pl.FP32] = pl.add(exponential, 1.0)
        sigmoid: pl.Tensor[[SILU_ROWS, SILU_COLS], pl.FP32] = pl.recip(denominator)
        y = pl.mul(x, sigmoid)
        return y


def main() -> None:
    """Run correctness-gated, register-once manual/AutoTile comparisons."""
    import torch  # noqa: PLC0415
    from pypto import ir  # noqa: PLC0415
    from pypto.runtime import RunConfig, benchmark  # noqa: PLC0415

    config = RunConfig(platform="a2a3")

    torch.manual_seed(0)
    softmax_x = torch.randn((SOFTMAX_ROWS, SOFTMAX_COLS), dtype=torch.float32)
    softmax_manual_out = torch.zeros_like(softmax_x)
    softmax_auto_out = torch.zeros_like(softmax_x)
    softmax_manual = manual_softmax.compile(softmax_x, softmax_manual_out, config=config)
    softmax_auto = ir.compile(AutoTileSoftmaxProgram, platform=config.platform)
    softmax_manual_stats = benchmark(
        softmax_manual,
        [softmax_x, softmax_manual_out],
        warmup=_EXAMPLE_WARMUP,
        rounds=_EXAMPLE_ROUNDS,
        config=config,
    )
    softmax_auto_stats = benchmark(
        softmax_auto,
        [softmax_x, softmax_auto_out],
        warmup=_EXAMPLE_WARMUP,
        rounds=_EXAMPLE_ROUNDS,
        config=config,
    )
    softmax_expected = torch.softmax(softmax_x, dim=1)
    torch.testing.assert_close(softmax_manual_out, softmax_expected, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(softmax_auto_out, softmax_expected, rtol=1.0e-5, atol=1.0e-5)
    print(
        "softmax: "
        f"manual={softmax_manual_stats.device_us_median:.3f} us, "
        f"auto_tile={softmax_auto_stats.device_us_median:.3f} us, "
        f"speedup={softmax_manual_stats.device_us_median / softmax_auto_stats.device_us_median:.3f}x"
    )

    torch.manual_seed(1)
    rms_x = torch.randn((RMS_ROWS, RMS_HIDDEN), dtype=torch.float32)
    gamma = torch.randn((1, RMS_HIDDEN), dtype=torch.float32)
    rms_manual_out = torch.zeros_like(rms_x)
    rms_auto_out = torch.zeros_like(rms_x)
    rms_manual = manual_rms_norm.compile(rms_x, gamma, rms_manual_out, config=config)
    rms_auto = ir.compile(AutoTileRmsNormProgram, platform=config.platform)
    rms_manual_stats = benchmark(
        rms_manual,
        [rms_x, gamma, rms_manual_out],
        warmup=_EXAMPLE_WARMUP,
        rounds=_EXAMPLE_ROUNDS,
        config=config,
    )
    rms_auto_stats = benchmark(
        rms_auto,
        [rms_x, gamma, rms_auto_out],
        warmup=_EXAMPLE_WARMUP,
        rounds=_EXAMPLE_ROUNDS,
        config=config,
    )
    rms_expected = rms_x * torch.rsqrt(rms_x.square().mean(dim=1, keepdim=True) + RMS_EPS) * gamma
    torch.testing.assert_close(rms_manual_out, rms_expected, rtol=1.0e-2, atol=1.0e-2)
    torch.testing.assert_close(rms_auto_out, rms_expected, rtol=1.0e-2, atol=1.0e-2)
    print(
        "RMSNorm: "
        f"manual={rms_manual_stats.device_us_median:.3f} us, "
        f"auto_tile={rms_auto_stats.device_us_median:.3f} us, "
        f"speedup={rms_manual_stats.device_us_median / rms_auto_stats.device_us_median:.3f}x"
    )

    torch.manual_seed(2)
    layer_x = torch.randn((LAYER_ROWS, LAYER_HIDDEN), dtype=torch.float32)
    layer_gamma = torch.randn((1, LAYER_HIDDEN), dtype=torch.float32)
    layer_beta = torch.randn((1, LAYER_HIDDEN), dtype=torch.float32)
    layer_manual_out = torch.zeros_like(layer_x)
    layer_auto_out = torch.zeros_like(layer_x)
    layer_manual = manual_layer_norm.compile(
        layer_x, layer_gamma, layer_beta, layer_manual_out, config=config
    )
    layer_auto = ir.compile(AutoTileLayerNormProgram, platform=config.platform)
    layer_manual_stats = benchmark(
        layer_manual,
        [layer_x, layer_gamma, layer_beta, layer_manual_out],
        warmup=_EXAMPLE_WARMUP,
        rounds=_EXAMPLE_ROUNDS,
        config=config,
    )
    layer_auto_stats = benchmark(
        layer_auto,
        [layer_x, layer_gamma, layer_beta, layer_auto_out],
        warmup=_EXAMPLE_WARMUP,
        rounds=_EXAMPLE_ROUNDS,
        config=config,
    )
    layer_mean = layer_x.mean(dim=1, keepdim=True)
    layer_variance = (layer_x - layer_mean).square().mean(dim=1, keepdim=True)
    layer_expected = (layer_x - layer_mean) / torch.sqrt(
        layer_variance + LAYER_EPS
    ) * layer_gamma + layer_beta
    torch.testing.assert_close(layer_manual_out, layer_expected, rtol=1.0e-2, atol=1.0e-2)
    torch.testing.assert_close(layer_auto_out, layer_expected, rtol=1.0e-2, atol=1.0e-2)
    print(
        "LayerNorm: "
        f"manual={layer_manual_stats.device_us_median:.3f} us, "
        f"auto_tile={layer_auto_stats.device_us_median:.3f} us, "
        f"speedup={layer_manual_stats.device_us_median / layer_auto_stats.device_us_median:.3f}x"
    )

    torch.manual_seed(3)
    silu_x = torch.randn((SILU_ROWS, SILU_COLS), dtype=torch.float32)
    silu_manual_out = torch.zeros_like(silu_x)
    silu_auto_out = torch.zeros_like(silu_x)
    silu_manual = manual_silu.compile(silu_x, silu_manual_out, config=config)
    silu_auto = ir.compile(AutoTileSiluProgram, platform=config.platform)
    silu_manual_stats = benchmark(
        silu_manual,
        [silu_x, silu_manual_out],
        warmup=_EXAMPLE_WARMUP,
        rounds=_EXAMPLE_ROUNDS,
        config=config,
    )
    silu_auto_stats = benchmark(
        silu_auto,
        [silu_x, silu_auto_out],
        warmup=_EXAMPLE_WARMUP,
        rounds=_EXAMPLE_ROUNDS,
        config=config,
    )
    silu_expected = torch.nn.functional.silu(silu_x)
    torch.testing.assert_close(silu_manual_out, silu_expected, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(silu_auto_out, silu_expected, rtol=1.0e-5, atol=1.0e-5)
    print(
        "SiLU: "
        f"manual={silu_manual_stats.device_us_median:.3f} us, "
        f"auto_tile={silu_auto_stats.device_us_median:.3f} us, "
        f"speedup={silu_manual_stats.device_us_median / silu_auto_stats.device_us_median:.3f}x"
    )


if __name__ == "__main__":
    main()

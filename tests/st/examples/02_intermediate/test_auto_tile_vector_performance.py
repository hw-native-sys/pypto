# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Silicon A/Bs for manual and whole-function AutoTile vector kernels.

Each pair uses identical tensor shapes, dtypes, inputs, formula, output, worker
configuration, warmup count, and measured launch count.  Correctness is a hard
gate.  Timing is deliberately reported rather than thresholded: a relative
performance result is useful evidence, while a fixed pass/fail speedup would be
fragile across firmware, runtime, PTOAS, and shared-device conditions.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
import torch
from examples.advanced.auto_tile_vector import (
    LAYER_EPS,
    LAYER_HIDDEN,
    LAYER_ROWS,
    RMS_EPS,
    RMS_HIDDEN,
    RMS_ROWS,
    SILU_COLS,
    SILU_ROWS,
    SOFTMAX_COLS,
    SOFTMAX_ROWS,
    AutoTileLayerNormProgram,
    AutoTileRmsNormProgram,
    AutoTileSiluProgram,
    AutoTileSoftmaxProgram,
    manual_layer_norm,
    manual_rms_norm,
    manual_silu,
    manual_softmax,
)
from pypto import ir
from pypto.runtime import BenchmarkStats, RunConfig, benchmark

_WARMUP = 5
_ROUNDS = 15
_REPETITIONS = 2


@dataclass(frozen=True)
class _SoftmaxPerformanceCase:
    """One preselected shape and its paired manual/AutoTile programs."""

    name: str
    rows: int
    cols: int
    manual: Any
    auto_program: Any


@pl.jit
def _manual_softmax_256x512(
    x: pl.Tensor[[256, 512], pl.FP32],
    y: pl.Out[pl.Tensor[[256, 512], pl.FP32]],
):
    for row in pl.parallel(0, 256, 32):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_rows"):
            tile_x = x[row : row + 32, :]
            maximum = pl.row_max(tile_x)
            shifted = pl.row_expand_sub(tile_x, maximum)
            exponent = pl.exp(shifted)
            total = pl.row_sum(exponent)
            y[row : row + 32, :] = pl.row_expand_div(exponent, total)
    return y


@pl.program
class _AutoTileSoftmax256x512Program:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        x: pl.Tensor[[256, 512], pl.FP32],
        y: pl.Out[pl.Tensor[[256, 512], pl.FP32]],
    ) -> pl.Tensor[[256, 512], pl.FP32]:
        maximum: pl.Tensor[[256, 1], pl.FP32] = pl.row_max(x)
        shifted: pl.Tensor[[256, 512], pl.FP32] = pl.sub(x, maximum)
        exponent: pl.Tensor[[256, 512], pl.FP32] = pl.exp(shifted)
        total: pl.Tensor[[256, 1], pl.FP32] = pl.row_sum(exponent)
        y = pl.div(exponent, total)
        return y


@pl.jit
def _manual_softmax_128x1024(
    x: pl.Tensor[[128, 1024], pl.FP32],
    y: pl.Out[pl.Tensor[[128, 1024], pl.FP32]],
):
    for row in pl.parallel(0, 128, 16):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_rows"):
            tile_x = x[row : row + 16, :]
            maximum = pl.row_max(tile_x)
            shifted = pl.row_expand_sub(tile_x, maximum)
            exponent = pl.exp(shifted)
            total = pl.row_sum(exponent)
            y[row : row + 16, :] = pl.row_expand_div(exponent, total)
    return y


@pl.program
class _AutoTileSoftmax128x1024Program:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        x: pl.Tensor[[128, 1024], pl.FP32],
        y: pl.Out[pl.Tensor[[128, 1024], pl.FP32]],
    ) -> pl.Tensor[[128, 1024], pl.FP32]:
        maximum: pl.Tensor[[128, 1], pl.FP32] = pl.row_max(x)
        shifted: pl.Tensor[[128, 1024], pl.FP32] = pl.sub(x, maximum)
        exponent: pl.Tensor[[128, 1024], pl.FP32] = pl.exp(shifted)
        total: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(exponent)
        y = pl.div(exponent, total)
        return y


@pl.jit
def _manual_softmax_32x8192(
    x: pl.Tensor[[32, 8192], pl.FP32],
    y: pl.Out[pl.Tensor[[32, 8192], pl.FP32]],
):
    # Eight rows keep the [8, 1] running statistics 32-byte aligned.  Stream
    # the wide axis so the reference remains within UB instead of materializing
    # an 8x8192 tile.
    for row in pl.parallel(0, 32, 8):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_rows"):
            first = x[row : row + 8, 0:480]
            running_max = pl.row_max(first)
            first_exp = pl.exp(pl.row_expand_sub(first, running_max))
            running_sum = pl.row_sum(first_exp)

            for chunk_index in pl.range(1, 17):
                col = chunk_index * 480
                tile_x = x[row : row + 8, col : col + 480]
                local_max = pl.row_max(tile_x)
                next_max = pl.maximum(running_max, local_max)
                correction = pl.exp(pl.sub(running_max, next_max))
                local_exp = pl.exp(pl.row_expand_sub(tile_x, next_max))
                local_sum = pl.row_sum(local_exp)
                running_sum = pl.add(pl.mul(running_sum, correction), local_sum)
                running_max = next_max

            tail = x[row : row + 8, 8160:8192]
            tail_max = pl.row_max(tail)
            final_max = pl.maximum(running_max, tail_max)
            correction = pl.exp(pl.sub(running_max, final_max))
            tail_exp = pl.exp(pl.row_expand_sub(tail, final_max))
            final_sum = pl.add(pl.mul(running_sum, correction), pl.row_sum(tail_exp))

            for chunk_index in pl.range(17):
                col = chunk_index * 480
                tile_x = x[row : row + 8, col : col + 480]
                exponent = pl.exp(pl.row_expand_sub(tile_x, final_max))
                y[row : row + 8, col : col + 480] = pl.row_expand_div(exponent, final_sum)
            tail = x[row : row + 8, 8160:8192]
            tail_exp = pl.exp(pl.row_expand_sub(tail, final_max))
            y[row : row + 8, 8160:8192] = pl.row_expand_div(tail_exp, final_sum)
    return y


@pl.program
class _AutoTileSoftmax32x8192Program:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        x: pl.Tensor[[32, 8192], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 8192], pl.FP32]],
    ) -> pl.Tensor[[32, 8192], pl.FP32]:
        maximum: pl.Tensor[[32, 1], pl.FP32] = pl.row_max(x)
        shifted: pl.Tensor[[32, 8192], pl.FP32] = pl.sub(x, maximum)
        exponent: pl.Tensor[[32, 8192], pl.FP32] = pl.exp(shifted)
        total: pl.Tensor[[32, 1], pl.FP32] = pl.row_sum(exponent)
        y = pl.div(exponent, total)
        return y


_SOFTMAX_PERFORMANCE_CASES = (
    _SoftmaxPerformanceCase(
        "softmax_512x256",
        SOFTMAX_ROWS,
        SOFTMAX_COLS,
        manual_softmax,
        AutoTileSoftmaxProgram,
    ),
    _SoftmaxPerformanceCase(
        "softmax_256x512", 256, 512, _manual_softmax_256x512, _AutoTileSoftmax256x512Program
    ),
    _SoftmaxPerformanceCase(
        "softmax_128x1024", 128, 1024, _manual_softmax_128x1024, _AutoTileSoftmax128x1024Program
    ),
    _SoftmaxPerformanceCase(
        "softmax_32x8192", 32, 8192, _manual_softmax_32x8192, _AutoTileSoftmax32x8192Program
    ),
)


def _compile_manual(jit_function, args: Sequence[torch.Tensor], config: RunConfig, output_dir: Path):
    """Compile one explicit reference while preserving its generated artifacts."""
    jit_function._cache.clear()
    compile_config = replace(config, save_kernels=True, save_kernels_dir=str(output_dir))
    return jit_function.compile(*args, config=compile_config)


def _compile_auto(program: Any, config: RunConfig, output_dir: Path):
    """Compile one tensor-level AutoTile program through the default pipeline."""
    return ir.compile(
        program,
        output_dir=str(output_dir),
        strategy=config.strategy,
        backend_type=config.backend_type,
        dump_passes=config.dump_passes,
        diagnostic_phase=config.diagnostic_phase,
        disabled_diagnostics=config.disabled_diagnostics,
        platform=config.platform,
        profiling=config.compile_profiling,
        analyze_auto_scopes_for_deps=config.analyze_auto_scopes_for_deps,
        memory_planner=config.memory_planner,
    )


def _effective_samples(runs: Sequence[BenchmarkStats]) -> list[float]:
    """Return sched/orch-union samples from single-dispatch benchmark runs."""
    samples = [invocation.effective_us for stats in runs for invocation in stats.invocations]
    assert len(samples) == _ROUNDS * len(runs)
    assert all(sample > 0.0 and math.isfinite(sample) for sample in samples)
    return samples


def _device_samples(runs: Sequence[BenchmarkStats]) -> list[float]:
    """Return device-wall samples and verify every run used the requested protocol."""
    for stats in runs:
        assert len(stats.device_wall_us) == _ROUNDS
        assert not stats.all_zero_device
    return [sample for stats in runs for sample in stats.device_wall_us]


def _compare(
    *,
    name: str,
    reference,
    auto_program: Any,
    reference_args: list[torch.Tensor],
    auto_args: list[torch.Tensor],
    expected: torch.Tensor,
    config: RunConfig,
    output_dir: Path,
    tolerance: tuple[float, float],
    record_property,
) -> None:
    benchmark_config = replace(
        config,
        enable_l2_swimlane=False,
        enable_dump_args=0,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
    )
    reference_compiled = _compile_manual(reference, reference_args, config, output_dir / f"{name}_manual")
    auto_compiled = _compile_auto(auto_program, config, output_dir / f"{name}_auto_tile")

    rtol, atol = tolerance

    def correctness_launch(compiled, args: list[torch.Tensor]) -> None:
        """Check a fresh launch so a later good timing launch cannot hide a bad one."""
        args[-1].fill_(float("nan"))
        compiled(*args, config=benchmark_config)
        torch.testing.assert_close(args[-1], expected, rtol=rtol, atol=atol)

    correctness_launch(reference_compiled, reference_args)
    correctness_launch(auto_compiled, auto_args)

    reference_runs = [
        benchmark(
            reference_compiled,
            reference_args,
            warmup=_WARMUP,
            rounds=_ROUNDS,
            config=benchmark_config,
        )
    ]
    correctness_launch(reference_compiled, reference_args)
    auto_runs = [
        benchmark(
            auto_compiled,
            auto_args,
            warmup=_WARMUP,
            rounds=_ROUNDS,
            config=benchmark_config,
        )
    ]
    correctness_launch(auto_compiled, auto_args)
    # Repeat in reverse order so page-in, thermal, and run-order effects do not
    # systematically favor either implementation.
    auto_runs.append(
        benchmark(
            auto_compiled,
            auto_args,
            warmup=_WARMUP,
            rounds=_ROUNDS,
            config=benchmark_config,
        )
    )
    correctness_launch(auto_compiled, auto_args)
    reference_runs.append(
        benchmark(
            reference_compiled,
            reference_args,
            warmup=_WARMUP,
            rounds=_ROUNDS,
            config=benchmark_config,
        )
    )
    correctness_launch(reference_compiled, reference_args)

    torch.testing.assert_close(reference_args[-1], expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(auto_args[-1], expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(auto_args[-1], reference_args[-1], rtol=rtol, atol=atol)

    reference_device = statistics.median(_device_samples(reference_runs))
    auto_device = statistics.median(_device_samples(auto_runs))
    reference_effective = statistics.median(_effective_samples(reference_runs))
    auto_effective = statistics.median(_effective_samples(auto_runs))
    assert all(
        value > 0.0 and math.isfinite(value)
        for value in (reference_device, auto_device, reference_effective, auto_effective)
    )

    device_speedup = reference_device / auto_device
    effective_speedup = reference_effective / auto_effective
    record_property(f"{name}.manual.device_wall_us", reference_device)
    record_property(f"{name}.auto_tile.device_wall_us", auto_device)
    record_property(f"{name}.auto_tile.device_wall_speedup", device_speedup)
    record_property(f"{name}.manual.effective_us", reference_effective)
    record_property(f"{name}.auto_tile.effective_us", auto_effective)
    record_property(f"{name}.auto_tile.effective_speedup", effective_speedup)
    for repetition, (reference_run, auto_run) in enumerate(zip(reference_runs, auto_runs, strict=True)):
        record_property(
            f"{name}.rep{repetition}.device_wall_speedup",
            reference_run.device_us_median / auto_run.device_us_median,
        )
        record_property(
            f"{name}.rep{repetition}.effective_speedup",
            statistics.median(_effective_samples([reference_run]))
            / statistics.median(_effective_samples([auto_run])),
        )
    print(
        f"AUTOTILE_PERF pair={name} repetitions={_REPETITIONS} "
        f"rounds={_ROUNDS} warmup={_WARMUP} "
        f"manual_device_wall_us={reference_device:.3f} "
        f"auto_tile_device_wall_us={auto_device:.3f} "
        f"device_wall_speedup={device_speedup:.4f} "
        f"manual_effective_us={reference_effective:.3f} "
        f"auto_tile_effective_us={auto_effective:.3f} "
        f"effective_speedup={effective_speedup:.4f}"
    )


@pytest.mark.slow
@pytest.mark.platforms("a2a3")
class TestAutoTileVectorPerformance:
    """Correctness-gated, register-once performance comparisons on Ascend 910B."""

    @pytest.mark.parametrize("case", _SOFTMAX_PERFORMANCE_CASES, ids=lambda case: case.name)
    def test_softmax_manual_vs_auto_tile(self, test_config, tmp_path, record_property, case):
        torch.manual_seed(0)
        x = torch.randn((case.rows, case.cols), dtype=torch.float32)
        expected = torch.softmax(x, dim=1)
        reference_out = torch.zeros_like(x)
        auto_out = torch.zeros_like(x)
        _compare(
            name=case.name,
            reference=case.manual,
            auto_program=case.auto_program,
            reference_args=[x, reference_out],
            auto_args=[x, auto_out],
            expected=expected,
            config=test_config,
            output_dir=tmp_path,
            tolerance=(1.0e-5, 1.0e-5),
            record_property=record_property,
        )

    def test_rms_norm_manual_vs_auto_tile(self, test_config, tmp_path, record_property):
        torch.manual_seed(1)
        x = torch.randn((RMS_ROWS, RMS_HIDDEN), dtype=torch.float32)
        gamma = torch.randn((1, RMS_HIDDEN), dtype=torch.float32)
        expected = x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + RMS_EPS) * gamma
        reference_out = torch.zeros_like(x)
        auto_out = torch.zeros_like(x)
        _compare(
            name="rms_norm",
            reference=manual_rms_norm,
            auto_program=AutoTileRmsNormProgram,
            reference_args=[x, gamma, reference_out],
            auto_args=[x, gamma, auto_out],
            expected=expected,
            config=test_config,
            output_dir=tmp_path,
            tolerance=(1.0e-2, 1.0e-2),
            record_property=record_property,
        )

    def test_layer_norm_manual_vs_auto_tile(self, test_config, tmp_path, record_property):
        torch.manual_seed(2)
        x = torch.randn((LAYER_ROWS, LAYER_HIDDEN), dtype=torch.float32)
        gamma = torch.randn((1, LAYER_HIDDEN), dtype=torch.float32)
        beta = torch.randn((1, LAYER_HIDDEN), dtype=torch.float32)
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        expected = (x - mean) / torch.sqrt(variance + LAYER_EPS) * gamma + beta
        reference_out = torch.zeros_like(x)
        auto_out = torch.zeros_like(x)
        _compare(
            name="layer_norm",
            reference=manual_layer_norm,
            auto_program=AutoTileLayerNormProgram,
            reference_args=[x, gamma, beta, reference_out],
            auto_args=[x, gamma, beta, auto_out],
            expected=expected,
            config=test_config,
            output_dir=tmp_path,
            tolerance=(1.0e-2, 1.0e-2),
            record_property=record_property,
        )

    def test_silu_manual_vs_auto_tile(self, test_config, tmp_path, record_property):
        torch.manual_seed(3)
        x = torch.randn((SILU_ROWS, SILU_COLS), dtype=torch.float32)
        expected = torch.nn.functional.silu(x)
        reference_out = torch.zeros_like(x)
        auto_out = torch.zeros_like(x)
        _compare(
            name="silu",
            reference=manual_silu,
            auto_program=AutoTileSiluProgram,
            reference_args=[x, reference_out],
            auto_args=[x, auto_out],
            expected=expected,
            config=test_config,
            output_dir=tmp_path,
            tolerance=(1.0e-5, 1.0e-5),
            record_property=record_property,
        )

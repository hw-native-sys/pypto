# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical coverage for compiler-generated runtime-bound local pipelines."""

from dataclasses import replace

import pypto.language as pl
import pytest
import torch
from codegen.dsl.software_pipeline_cases import (
    make_epilogue,
    make_mixed,
    make_nested,
    make_scratch_pressure,
)
from pypto import passes


def _run_pipeline(kernel, args, test_config, software):
    """Keep the IR path identical when comparing software pipelining on and off."""
    # The default non-pipelined path uses Buffer IR. Force Tile IR in both
    # variants so numerical differences isolate scheduling and slot reuse.
    config = replace(test_config, enable_software_pipeline=None)
    with passes.PassContext([], enable_buffer_ir=False, enable_software_pipeline=software):
        kernel(*args, config=config)


@pl.jit
def dynamic_pipeline(
    x: pl.Tensor[[144, 64], pl.FP32],
    stop: pl.Scalar[pl.INDEX],
    y: pl.InOut[pl.Tensor[[144, 64], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.pipeline(2, stop, 2, stage=3):
            tile = pl.load(x, [i, 0], [1, 64])
            y = pl.store(pl.add(tile, 0.25), [i, 0], y)
    return y


@pytest.mark.parametrize("trips", [-4611686018427387905, -1, 0, 1, 2, 3, 65])
def test_runtime_pipeline_preserves_unwritten_rows(test_config, trips):
    """Check wraparound, nonzero start/step, empty loops and guarded tails."""
    x = torch.arange(144 * 64, dtype=torch.float32).reshape(144, 64)
    y = torch.full_like(x, -777.0)
    expected = y.clone()
    stop = 2 + trips * 2
    expected[2:stop:2] = x[2:stop:2] + 0.25
    dynamic_pipeline(x, stop, y, config=replace(test_config, enable_software_pipeline=True))
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


@pytest.mark.parametrize("trips", [0, 1, 65])
@pytest.mark.parametrize(
    ("children", "leaves", "depth", "guarded"),
    [(5, 0, 3, False), (2, 2, 2, False), (2, 2, 2, True)],
)
def test_nested_pipeline_preserves_values_and_inactive_groups(
    test_config, trips, children, leaves, depth, guarded
):
    rows_per_group = children * max(leaves, 1)
    x = torch.arange(65 * rows_per_group * 64, dtype=torch.float32).reshape(-1, 64)
    y = torch.full_like(x, -777.0)
    expected = y.clone()
    active = trips * rows_per_group
    expected[:active] = x[:active] + 1.0
    if guarded:
        for g in range(trips):
            for p in range(children):
                if (g + p) % 2:
                    row = (g * children + p) * leaves
                    expected[row : row + leaves] = -777.0
    kernel = make_nested(
        children=children, inner_stage=depth, leaves=leaves, leaf_stage=depth, guarded=guarded
    )
    kernel(x, trips, y, config=replace(test_config, enable_software_pipeline=True))
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


@pytest.mark.parametrize("software", [False, True])
def test_static_pipeline_reuses_slots_without_touching_tail(test_config, software):
    """Check two-input slot wraparound and untouched output rows."""
    capacity, trips, stage = 9, 7, 3

    @pl.jit
    def binary_pipeline(
        x: pl.Tensor[[capacity, 1024], pl.FP32],
        z: pl.Tensor[[capacity, 1024], pl.FP32],
        y: pl.InOut[pl.Tensor[[capacity, 1024], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            for row in pl.pipeline(trips, stage=stage):
                a = pl.load(x, [row, 0], [1, 1024])
                b = pl.load(z, [row, 0], [1, 1024])
                pl.store(pl.add(pl.add(a, b), 0.25), [row, 0], y)
        return y

    generator = torch.Generator().manual_seed(2026)
    x = torch.randn((capacity, 1024), generator=generator)
    z = torch.randn((capacity, 1024), generator=generator)
    y = torch.full_like(x, -777.0)
    expected = y.clone()
    expected[:trips] = x[:trips] + z[:trips] + 0.25
    _run_pipeline(binary_pipeline, (x, z, y), test_config, software)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


@pytest.mark.parametrize("software", [False, True])
@pytest.mark.parametrize(
    ("capacity", "active", "dynamic"),
    [(3, 3, False), (3, 2, True)],
)
def test_score_epilogue_keeps_reduction_workspace_values(test_config, software, capacity, active, dynamic):
    """Validate cast, broadcast, workspace reduction and inactive output canaries."""
    generator = torch.Generator().manual_seed(2026)
    acc = torch.randint(-127, 128, (capacity * 64, 64), generator=generator, dtype=torch.int32)
    scale = torch.randn((capacity * 64, 1), generator=generator)
    coef = torch.randn((1, 64), generator=generator)
    out = torch.full((capacity * 64, 1), -777.0)
    expected = out.clone()
    rows = active * 64
    expected[:rows] = (acc[:rows].float().clamp_min(0) * coef).sum(dim=1, keepdim=True) * scale[:rows]
    _run_pipeline(make_epilogue(capacity, dynamic), (acc, scale, coef, active, out), test_config, software)
    torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-4)


@pytest.mark.parametrize("software", [False, True])
@pytest.mark.parametrize("split_mode", [pl.SplitMode.UP_DOWN, pl.SplitMode.LEFT_RIGHT])
def test_mixed_pipeline_matches_matmul_and_dependent_task(test_config, software, split_mode):
    """Check AIC/AIV FIFO ownership, local reuse and the following GM consumer."""
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn((512, 32), generator=generator, dtype=torch.float16)
    weight = torch.randn((32, 16), generator=generator, dtype=torch.float16)
    scale = torch.randn((512, 1), generator=generator)
    y = torch.empty((512, 16), dtype=torch.float32)
    z = torch.empty_like(y)
    expected = (x.float() @ weight.float() + 1.0) * scale
    # All four groups are active: the dependent task consumes the entire y.
    _run_pipeline(make_mixed(3, 3, split_mode), (x, weight, scale, 4, y, z), test_config, software)
    torch.testing.assert_close(y, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(z, expected + 1.0, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("software", [False, True])
def test_retired_scratch_allows_larger_later_tiles(test_config, software):
    """Fixed stage depths must fit and preserve both phases' observable outputs."""
    rows, total_rows = 56, 896
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn((total_rows, 128), generator=generator)
    reduced = torch.empty((512, 1))
    processed = torch.empty_like(x)
    _run_pipeline(make_scratch_pressure(rows, total_rows), (x, reduced, processed), test_config, software)
    torch.testing.assert_close(reduced, x[:512].sum(dim=1, keepdim=True), rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(processed, x + 1.0, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

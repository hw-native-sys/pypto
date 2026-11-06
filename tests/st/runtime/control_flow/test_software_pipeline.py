# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical coverage for single-level static and runtime-bound local pipelines."""

from dataclasses import replace

import pypto.language as pl
import pytest
import torch
from pypto import passes

from tests.software_pipeline_cases import make_epilogue


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

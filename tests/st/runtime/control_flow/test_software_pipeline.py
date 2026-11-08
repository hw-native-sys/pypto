# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical coverage for static, runtime-bound and bounded nested local pipelines."""

from dataclasses import replace

import pypto.language as pl
import pytest
import torch
from pypto import passes

from tests.software_pipeline_cases import make_epilogue, make_nested, make_single_parent_nested


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


@pytest.mark.parametrize(
    ("groups", "children", "outer_stage", "inner_stage", "leaves", "leaf_stage", "guarded", "software"),
    [
        (0, 5, 3, 3, 0, 3, False, True),
        (1, 5, 3, 3, 0, 3, False, True),
        (65, 5, 3, 3, 0, 3, False, True),
        (0, 2, 3, 2, 2, 2, False, True),
        (1, 2, 3, 2, 2, 2, False, True),
        (65, 2, 3, 2, 2, 2, False, True),
        (0, 2, 3, 2, 2, 2, True, True),
        (1, 2, 3, 2, 2, 2, True, True),
        (65, 2, 3, 2, 2, 2, True, True),
        (65, 1, 2, 4, 0, 4, False, True),
        (65, 2, 3, 2, 2, 4, True, True),
        (65, 2, 3, 2, 2, 2, True, False),
    ],
)
def test_nested_pipeline_preserves_values_and_inactive_groups(
    test_config, groups, children, outer_stage, inner_stage, leaves, leaf_stage, guarded, software
):
    """Check parent wraparound, short children, guarded leaves and untouched canaries."""
    rows_per_group = children * max(leaves, 1)
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn((65 * rows_per_group, 64), generator=generator)
    y = torch.full_like(x, -777.0)
    expected = y.clone()
    active = groups * rows_per_group
    expected[:active] = x[:active] + 1.0
    if guarded:
        # Preserve canaries for inactive child batches as well as inactive
        # outer groups; both can otherwise hide speculative slot overwrites.
        for g in range(groups):
            for p in range(children):
                if (g + p) % 2:
                    row = (g * children + p) * leaves
                    expected[row : row + leaves] = -777.0
    kernel = make_nested(children, outer_stage, inner_stage, leaves, leaf_stage, guarded)
    _run_pipeline(kernel, (x, groups, y), test_config, software)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


def test_static_single_parent_preserves_all_child_values(test_config):
    """Check constant slot indices after eliminating a one-trip parent loop."""
    x = torch.randn((6, 64), generator=torch.Generator().manual_seed(2026))
    y = torch.full_like(x, -777.0)
    _run_pipeline(make_single_parent_nested(), (x, y), test_config, True)
    torch.testing.assert_close(y, x + 1.0, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

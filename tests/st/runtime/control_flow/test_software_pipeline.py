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
from codegen.dsl.software_pipeline_cases import make_nested


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

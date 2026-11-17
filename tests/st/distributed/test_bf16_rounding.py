# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact BF16 midpoint checks through guarded L2 execution."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

import pypto.language as pl
import pytest
import torch
from pypto.runtime import RunConfig


@pl.jit.incore
def casts(
    x: pl.Tensor[[1, 32], pl.FP32],
    away: pl.Out[pl.Tensor[[1, 32], pl.BF16]],
    even: pl.Out[pl.Tensor[[1, 32], pl.BF16]],
):
    away = pl.assemble(away, pl.cast(x, pl.BF16), [0, 0])
    even = pl.assemble(even, pl.cast(x, pl.BF16, mode="rint"), [0, 0])
    return away, even


@pl.jit
def casts_chip(
    x: pl.Tensor[[1, 32], pl.FP32],
    away: pl.Out[pl.Tensor[[1, 32], pl.BF16]],
    even: pl.Out[pl.Tensor[[1, 32], pl.BF16]],
):
    away, even = casts(x, away, even)
    return away, even


def test_bf16_midpoints(test_config: RunConfig, device_ids: list[int]) -> None:
    config = replace(test_config, device_id=device_ids[0])
    if config.platform != "a2a3":
        pytest.skip("BF16 midpoint hardware proof targets a2a3")
    # Exactly representable FP32 midpoints between BF16 values around +/-1.
    lower = 1 + torch.arange(16, dtype=torch.float32) / 128
    positive = lower + 1 / 256
    x = torch.cat((positive, -positive)).reshape(1, 32)
    expected_even = x.bfloat16()
    expected_away = torch.cat((lower + 1 / 128, -lower - 1 / 128)).reshape(1, 32).bfloat16()
    away = torch.empty((1, 32), dtype=torch.bfloat16)
    even = torch.empty_like(away)
    meta = [torch.empty(t.shape, dtype=t.dtype, device="meta") for t in (x, away, even)]
    with patch(
        "pypto.runtime.device_runner.ThreadPoolExecutor",
        side_effect=lambda **kw: ThreadPoolExecutor(max_workers=1),
    ):
        compiled = casts_chip.compile(*meta, config=config)
        _ = compiled.chip_callable
    # The standard compiled call acquires the suite's Worker card lock and
    # transfers CPU operands; no framework device is opened before that lock.
    compiled(x, away, even, config=config)
    assert torch.equal(even, expected_even)
    assert torch.equal(away, expected_away)
    assert (away != even).sum().item() == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

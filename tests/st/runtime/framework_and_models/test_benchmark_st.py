# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end test for the register-once ``benchmark`` helper (issue #1858).

The unit test (``tests/ut/runtime/test_benchmark.py``) mocks the worker and the
stderr capture, so it only proves the parse / warmup-discard / aggregation
logic. This system test runs a real kernel on device and exercises the full
on-device path that the UT cannot: ``benchmark`` raising the runtime log level
to ``v9``, fd-level capturing the host runtime's ``[STRACE]`` stderr markers
(simpler PR #1177), and parsing the *measured* per-launch ``device_wall_us``
out of them.

Timing semantics asserted here (L2 single-task, default ``SIMPLER_PROFILING``
build): every measured launch carries a real on-NPU ``device_wall_us > 0``, and
the ``warmup`` launches are excluded from the sample count.
"""

import sys

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.runtime import RunConfig, benchmark

_M = 128


@pl.program
class AddProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[_M, _M], pl.FP32],
        b: pl.Tensor[[_M, _M], pl.FP32],
        c: pl.Out[pl.Tensor[[_M, _M], pl.FP32]],
    ) -> pl.Tensor[[_M, _M], pl.FP32]:
        ta = pl.load(a, [0, 0], [_M, _M])
        tb = pl.load(b, [0, 0], [_M, _M])
        return pl.store(pl.add(ta, tb), [0, 0], c)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch_add(
        self,
        a: pl.Tensor[[_M, _M], pl.FP32],
        b: pl.Tensor[[_M, _M], pl.FP32],
        c: pl.Out[pl.Tensor[[_M, _M], pl.FP32]],
    ) -> pl.Tensor[[_M, _M], pl.FP32]:
        return self.tile_add(a, b, c)


_EXPECTED = torch.full((_M, _M), 5.0, dtype=torch.float32)


def _inputs():
    a = torch.full((_M, _M), 2.0, dtype=torch.float32)
    b = torch.full((_M, _M), 3.0, dtype=torch.float32)
    c = torch.zeros((_M, _M), dtype=torch.float32)
    return a, b, c


def test_benchmark_register_once_surfaces_timing(test_config, tmp_path):
    """``benchmark`` registers once and surfaces per-launch device time (#1858).

    One ``ChipWorker`` / one ``register``, then ``warmup + rounds`` cheap
    launches whose ``device_wall_us`` are read off the ``[STRACE]`` markers and
    aggregated. Asserts each measured sample is a real L2 device wall (default
    ``SIMPLER_PROFILING`` build) and that warmup launches are excluded.
    """
    compiled = ir.compile(AddProgram, output_dir=str(tmp_path), platform=test_config.platform)

    a, b, c = _inputs()
    worker_cfg = RunConfig(platform=test_config.platform, device_id=test_config.device_id)
    rounds, warmup = 5, 2
    stats = benchmark(compiled, [a, b, c], rounds=rounds, warmup=warmup, config=worker_cfg)

    # Output is correct after the final measured launch.
    torch.testing.assert_close(c, _EXPECTED, rtol=1e-5, atol=1e-5)

    assert len(stats.device_wall_us) == rounds, (
        f"expected {rounds} measured samples (warmup excluded), got {len(stats.device_wall_us)}"
    )
    assert stats.rounds == rounds and stats.warmup == warmup
    assert not stats.all_zero_device, "device_wall_us must be > 0 on the default SIMPLER_PROFILING build"
    assert stats.device_us_min > 0.0
    assert stats.device_us_max >= stats.device_us_min
    assert stats.device_us_min <= stats.device_us_median <= stats.device_us_max


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])

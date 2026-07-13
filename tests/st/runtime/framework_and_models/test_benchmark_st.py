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

Timing semantics asserted here (L2 single-task, default ``SIMPLER_HOST_STRACE``
build): every measured launch carries a real on-NPU ``device_wall_us > 0``, and
the ``warmup`` launches are excluded from the sample count.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig
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
    ``SIMPLER_HOST_STRACE`` build) and that warmup launches are excluded.
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
    assert not stats.all_zero_device, "device_wall_us must be > 0 on the default SIMPLER_HOST_STRACE build"
    assert stats.device_us_min > 0.0
    assert stats.device_us_max >= stats.device_us_min
    assert stats.device_us_min <= stats.device_us_median <= stats.device_us_max


_L3_ROWS = 16
_L3_COLS = 32


def _build_per_rank_add_one():
    """A minimal L3 program: one rank-pinned ``+ 1`` child dispatch per rank."""

    @pl.program
    class PerRankAddOne:
        @pl.function(type=pl.FunctionType.InCore)
        def add_one(
            self,
            x: pl.Tensor[[_L3_ROWS, _L3_COLS], pl.FP32],
            y: pl.Out[pl.Tensor[[_L3_ROWS, _L3_COLS], pl.FP32]],
        ) -> pl.Tensor[[_L3_ROWS, _L3_COLS], pl.FP32]:
            for row in pl.parallel(_L3_ROWS):
                x_row = pl.slice(x, [1, _L3_COLS], [row, 0])
                y_row = pl.add(x_row, 1.0)
                y = pl.assemble(y, y_row, [row, 0])
            return y

        @pl.function(type=pl.FunctionType.Orchestration)
        def child(
            self,
            x: pl.Tensor[[_L3_ROWS, _L3_COLS], pl.FP32],
            y: pl.Out[pl.Tensor[[_L3_ROWS, _L3_COLS], pl.FP32]],
        ) -> pl.Tensor[[_L3_ROWS, _L3_COLS], pl.FP32]:
            y = self.add_one(x, y)
            return y

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            x: pl.Tensor[[pl.dynamic("NR"), _L3_ROWS, _L3_COLS], pl.FP32],
            y: pl.Out[pl.Tensor[[pl.dynamic("NR"), _L3_ROWS, _L3_COLS], pl.FP32]],
        ):
            for r in pl.range(pld.world_size()):
                self.child(x[r], y[r], device=r)

    return PerRankAddOne


def test_benchmark_l3_surfaces_per_rank_timing(test_config, device_ids):
    """``benchmark`` on an L3 program aggregates per-rank ``[STRACE]`` markers.

    Routes through ``DistributedWorker`` (``compiled.prepare()``), dispatches
    ``warmup + rounds`` DAG launches, and folds each rank's chip-child markers
    into per-round samples. The per-rank ``+1`` program dispatches exactly one
    child per rank per round (deterministic shape → no flatten fallback), so the
    headline ``device_wall_us`` is the per-round max across ranks and
    ``per_rank("device")`` carries one list per rank.
    """
    n_ranks = 2
    if len(device_ids) < n_ranks:
        pytest.skip(f"L3 benchmark needs >= {n_ranks} devices, got {device_ids}")

    program = _build_per_rank_add_one()
    compiled = ir.compile(
        program,
        platform=test_config.platform,
        distributed_config=DistributedConfig(device_ids=device_ids[:n_ranks], num_sub_workers=0),
    )

    # L3 requires shared-memory host tensors (the forked chip workers read them
    # through the inherited mapping). Reused in place across launches.
    inputs = torch.randn((n_ranks, _L3_ROWS, _L3_COLS), dtype=torch.float32).share_memory_()
    outputs = torch.zeros((n_ranks, _L3_ROWS, _L3_COLS), dtype=torch.float32).share_memory_()

    rounds, warmup = 5, 2
    # platform/device_id are rejected for L3 (device set is compile-fixed).
    stats = benchmark(compiled, [inputs, outputs], rounds=rounds, warmup=warmup)

    # Output is correct after the final measured launch.
    assert torch.allclose(outputs, inputs + 1.0), (
        f"L3 benchmark output mismatch: max diff = {(outputs - (inputs + 1.0)).abs().max().item()}"
    )

    assert not stats.fallback_flattened, "deterministic 1-dispatch/rank/round shape must segment cleanly"
    assert len(stats.device_wall_us) == rounds, (
        f"expected {rounds} per-round samples (warmup excluded), got {len(stats.device_wall_us)}"
    )
    # One per-rank series per device, each with one entry per measured round.
    rank_dev = stats.per_rank("device")
    rank_host = stats.per_rank("host")
    rank_eff = stats.per_rank("effective")
    union = stats.per_round("union")
    assert len(rank_dev) == n_ranks, f"expected {n_ranks} ranks, got {sorted(rank_dev)}"
    for pid, series in rank_dev.items():
        assert len(series) == rounds, f"rank {pid} has {len(series)} rounds, expected {rounds}"
    # Headline is the per-round max across ranks.
    for k in range(rounds):
        assert stats.device_wall_us[k] == max(v[k] for v in rank_dev.values())

    # Cross-rank host-timeline union window is populated (one per round) and, as a
    # window spanning all ranks' host spans, is >= the per-round host max.
    assert len(union) == rounds
    for k in range(rounds):
        assert union[k] >= max(v[k] for v in rank_host.values()) - 1e-6

    assert not stats.all_zero_device, "device_wall_us must be > 0 on the default SIMPLER_HOST_STRACE build"
    assert stats.device_us_min > 0.0

    # Per-card L2 Effective (orch union sched window): one series per rank, > 0, and
    # bounded by that rank's device wall (Effective ⊆ device_wall).
    assert set(rank_eff) == set(rank_dev)
    for pid, eff in rank_eff.items():
        assert len(eff) == rounds
        for k in range(rounds):
            assert 0.0 < eff[k] <= rank_dev[pid][k] + 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])

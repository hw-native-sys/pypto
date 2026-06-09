# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end ``RunTiming`` surfacing across the L3 distributed dispatch paths (issue #1679).

The L3 half of ``tests/st/runtime/framework_and_models/test_run_timing_st.py``.
A level-2 ``Worker`` session leaks runtime state that breaks a later level-3
``Worker`` init in the same process, so the L2 and L3 timing tests must never
share a ``pytest`` process (same split documented in ``test_l2_multi_orch.py``).
Distributed tests already run in their own CI invocation — keep this file there.

Covers the two user-reachable L3 dispatch entry points:

* ``DistributedCompiledProgram.__call__`` — one-shot ``Worker(level=3)`` →
  ``compiled.last_run_timing``
* ``DistributedWorker`` (``compiled.prepare()`` → ``rt(...)``) — reusable
  handle → ``rt.last_run_timing``

Timing semantics asserted here (L3 DAG):

* ``host_wall_us > 0`` — the host wall wraps the whole fork/dispatch, always nonzero.
* ``device_wall_us == 0`` — per-task device cycles are **not** aggregated in the
  ring scheduler, so the L3 DAG reports no device wall. This is the defining
  contrast against the L2 paths (where ``device_wall_us > 0``).
"""

import sys

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.runtime.task_interface import RunTiming

_M = 128
_EXPECTED = torch.full((_M, _M), 5.0, dtype=torch.float32)


@pl.program
class L3AddProgram:
    """L3: HOST orch → CHIP orch → InCore (``f = a + b``)."""

    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[_M, _M], pl.FP32],
        b: pl.Tensor[[_M, _M], pl.FP32],
        f: pl.Out[pl.Tensor[[_M, _M], pl.FP32]],
    ) -> pl.Tensor[[_M, _M], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [_M, _M])
        tile_b = pl.load(b, [0, 0], [_M, _M])
        return pl.store(pl.add(tile_a, tile_b), [0, 0], f)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        a: pl.Tensor[[_M, _M], pl.FP32],
        b: pl.Tensor[[_M, _M], pl.FP32],
        f: pl.Out[pl.Tensor[[_M, _M], pl.FP32]],
    ) -> pl.Tensor[[_M, _M], pl.FP32]:
        return self.tile_add(a, b, f)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        a: pl.Tensor[[_M, _M], pl.FP32],
        b: pl.Tensor[[_M, _M], pl.FP32],
        f: pl.Out[pl.Tensor[[_M, _M], pl.FP32]],
    ) -> pl.Tensor[[_M, _M], pl.FP32]:
        return self.chip_orch(a, b, f)


def _assert_l3_semantics(timing, label: str) -> None:
    """L3 DAG semantics: host wall present, device wall unaggregated (== 0)."""
    assert isinstance(timing, RunTiming), (
        f"expected a RunTiming from a real dispatch, got {type(timing).__name__}: {timing!r}"
    )
    # Print so the wall times surface in the CI log (``-s``).
    print(
        f"[run-timing] {label}: host_wall_us={timing.host_wall_us:.3f} "
        f"device_wall_us={timing.device_wall_us:.3f}"
    )
    assert timing.host_wall_us > 0.0, f"host_wall_us must be > 0, got {timing.host_wall_us}"
    assert timing.host_wall_ns > 0, f"host_wall_ns must be > 0, got {timing.host_wall_ns}"
    # The ring scheduler does not aggregate per-task device cycles for an L3 DAG,
    # so device wall is reported as exactly zero (contrast: L2 has it > 0).
    assert timing.device_wall_us == 0.0, (
        f"device_wall_us must be 0 for the L3 DAG, got {timing.device_wall_us}"
    )
    assert timing.device_wall_ns == 0, f"device_wall_ns must be 0 for the L3 DAG, got {timing.device_wall_ns}"


def _compile_l3(test_config, device_ids):
    return ir.compile(
        L3AddProgram,
        platform=test_config.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids[:1],
            num_sub_workers=0,
            block_dim=3,
            aicpu_thread_num=4,
        ),
    )


class TestL3RunTimingSurface:
    """``RunTiming`` surfaces on both user-reachable L3 dispatch entry points."""

    def test_distributed_compiled_program_call_surfaces_timing(self, test_config, device_ids):
        """(G) one-shot ``DistributedCompiledProgram(*args)`` → ``last_run_timing``."""
        if not device_ids:
            pytest.skip("L3 run-timing test needs at least one device")

        compiled = _compile_l3(test_config, device_ids)

        a = torch.full((_M, _M), 2.0, dtype=torch.float32)
        b = torch.full((_M, _M), 3.0, dtype=torch.float32)
        f = torch.zeros((_M, _M), dtype=torch.float32)

        compiled(a, b, f)

        torch.testing.assert_close(f, _EXPECTED, rtol=1e-5, atol=1e-5)
        _assert_l3_semantics(compiled.last_run_timing, "DistributedCompiledProgram.__call__")

    def test_distributed_worker_prepared_surfaces_timing(self, test_config, device_ids):
        """(H) ``compiled.prepare()`` → ``rt(*args)`` → ``rt.last_run_timing``.

        Per-call IO uses shared-memory host tensors allocated *before*
        ``prepare()`` so the forked chip worker inherits their mappings.
        """
        if not device_ids:
            pytest.skip("L3 run-timing test needs at least one device")

        compiled = _compile_l3(test_config, device_ids)

        host_a = torch.full((_M, _M), 2.0, dtype=torch.float32).share_memory_()
        host_b = torch.full((_M, _M), 3.0, dtype=torch.float32).share_memory_()
        host_out = torch.zeros((_M, _M), dtype=torch.float32).share_memory_()

        with compiled.prepare() as rt:
            rt(host_a, host_b, host_out)

        torch.testing.assert_close(host_out, _EXPECTED, rtol=1e-5, atol=1e-5)
        _assert_l3_semantics(rt.last_run_timing, "DistributedWorker (prepared)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])

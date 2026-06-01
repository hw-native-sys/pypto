# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board verification of the L3 explicit-dispatch runtime API (PR #1599).

The runtime refactor made ``DistributedWorker`` symmetric with the L2
``ChipWorker``: an L3 program is prepared once via
``DistributedCompiledProgram.prepare()`` and then dispatched explicitly through
``register(compiled) -> RegistrationHandle`` / ``run(compiled, *args)``, with
the same centralized ``DeviceTensor`` lifecycle (``alloc_tensor`` tracked in an
owned-set, reclaimed by ``close()``) and the same post-close invalidation.

This drives those paths on real silicon. A single-chip distributed runtime is
used: it still forks the full HOST → chip hierarchy, so every L3 dispatch path
is exercised, while keeping the device footprint and failure surface minimal.
The strict-identity rejection (``register``/``run`` with a different compiled)
is already covered by the mocked unit tests in
``tests/ut/runtime/test_distributed_worker.py``; here we verify the
hardware-specific behaviors: real dispatch correctness, real device-memory
auto-free, and the register-after-close guard on a genuinely closed worker.

Run on hardware via ``task-submit`` (one chip)::

    task-submit --device auto --device-num 1 --run 'cd <repo> && \
        export PYTHONPATH=<repo>/python:$PYTHONPATH && \
        export PTO_ISA_ROOT=/path/to/pto-isa && \
        python -m pytest tests/st/distributed/test_l3_explicit_dispatch_onboard.py \
        -v --platform a2a3 --device $TASK_DEVICE'

or on the host simulator with ``--platform a2a3sim``.
"""

import logging
import sys

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

M = 128
_LEAK_LOGGER = "pypto.runtime.runtime_base"


@pl.program
class L3AddProgram:
    """Single-chip L3: HOST orch → CHIP orch → InCore tile add (f = a + b)."""

    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[M, M], pl.FP32],
        b: pl.Tensor[[M, M], pl.FP32],
        f: pl.Out[pl.Tensor[[M, M], pl.FP32]],
    ) -> pl.Tensor[[M, M], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [M, M])
        tile_b = pl.load(b, [0, 0], [M, M])
        return pl.store(pl.add(tile_a, tile_b), [0, 0], f)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        a: pl.Tensor[[M, M], pl.FP32],
        b: pl.Tensor[[M, M], pl.FP32],
        f: pl.Out[pl.Tensor[[M, M], pl.FP32]],
    ) -> pl.Tensor[[M, M], pl.FP32]:
        return self.tile_add(a, b, f)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        a: pl.Tensor[[M, M], pl.FP32],
        b: pl.Tensor[[M, M], pl.FP32],
        f: pl.Out[pl.Tensor[[M, M], pl.FP32]],
    ) -> pl.Tensor[[M, M], pl.FP32]:
        return self.chip_orch(a, b, f)


@pl.program
class L3MultiChipProgram:
    """Two-chip L3: chip 0 computes ``a + b``, chip 1 computes ``a - b``."""

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        a: pl.Tensor[[M, M], pl.FP32],
        b: pl.Tensor[[M, M], pl.FP32],
        sum_buf: pl.Out[pl.Tensor[[M, M], pl.FP32]],
        sub_buf: pl.Out[pl.Tensor[[M, M], pl.FP32]],
    ) -> pl.Tensor[[M, M], pl.FP32]:
        with pl.at(level=pl.Level.CHIP, role=pl.Role.Orchestrator):
            with pl.at(level=pl.Level.CORE_GROUP):
                out_sum = pl.assemble(sum_buf, pl.add(a, b), [0, 0])
        with pl.at(level=pl.Level.CHIP, role=pl.Role.Orchestrator):
            with pl.at(level=pl.Level.CORE_GROUP):
                pl.assemble(sub_buf, pl.sub(a, b), [0, 0])
        return out_sum


@pytest.fixture
def _no_leak_warning(caplog):
    """Fail if ``_close_owned_tensors`` logs a free-failure ("leaking") warning."""
    with caplog.at_level(logging.WARNING, logger=_LEAK_LOGGER):
        yield caplog
    leaks = [r for r in caplog.records if "leaking" in r.getMessage()]
    assert not leaks, f"close() emitted owned-tensor leak warnings: {[r.getMessage() for r in leaks]}"


def test_l3_explicit_dispatch_single_chip(test_config, device_ids, _no_leak_warning):
    """prepare → register → handle/run dispatch; close auto-frees; reuse-after-close raises."""
    if not device_ids:
        pytest.skip(f"L3 explicit-dispatch test needs >= 1 device, got {device_ids}")

    compiled = ir.compile(
        L3AddProgram,
        platform=test_config.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids[:1],
            num_sub_workers=1,
            block_dim=3,
            aicpu_thread_num=4,
        ),
    )

    # L3 uploads run inside the forked chip worker, so IO and init sources must
    # be CPU shared-memory tensors allocated BEFORE prepare().
    host_a = torch.full((M, M), 2.0, dtype=torch.float32).share_memory_()
    host_weight = torch.full((M, M), 3.0, dtype=torch.float32).share_memory_()
    host_f = torch.zeros((M, M), dtype=torch.float32).share_memory_()
    expected = torch.full((M, M), 5.0, dtype=torch.float32)

    rt = compiled.prepare()
    try:
        # Static weight uploaded once into device memory; not explicitly freed,
        # so close() must reclaim it via _close_owned_tensors.
        weight = rt.alloc_tensor((M, M), torch.float32, init=host_weight)
        assert len(rt._owned_tensors) == 1

        # Explicit register → handle dispatch (symmetric with the L2 path).
        handle = rt.register(compiled)
        assert handle.compiled is compiled

        handle(host_a, weight, host_f)
        assert torch.allclose(host_f, expected, rtol=1e-5, atol=1e-5), (
            f"L3 handle dispatch wrong: max diff {(host_f - expected).abs().max().item()}"
        )

        # run(compiled, *args) is the other half of the explicit surface.
        host_f.zero_()
        rt.run(compiled, host_a, weight, host_f)
        assert torch.allclose(host_f, expected, rtol=1e-5, atol=1e-5), (
            f"L3 run() dispatch wrong: max diff {(host_f - expected).abs().max().item()}"
        )
    finally:
        rt.close()

    # close() auto-freed the weight without an explicit free_tensor.
    assert len(rt._owned_tensors) == 0, "close() must reclaim all owned L3 DeviceTensors"

    # Post-close reuse is rejected (symmetric with L2 worker/handle invalidation).
    with pytest.raises(RuntimeError, match="called after close"):
        rt.register(compiled)
    assert handle.closed, "handles must be marked closed by DistributedWorker.close()"


def test_l3_explicit_dispatch_multi_chip(test_config, device_ids, _no_leak_warning):
    """Explicit register/handle/run across a genuine 2-chip program (chip 0 add, chip 1 sub)."""
    if len(device_ids) < 2:
        pytest.skip(f"multi-chip L3 explicit-dispatch test needs >= 2 devices, got {device_ids}")

    compiled = ir.compile(
        L3MultiChipProgram,
        platform=test_config.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids[:2],
            num_sub_workers=1,
            block_dim=3,
            aicpu_thread_num=4,
        ),
    )

    a = torch.full((M, M), 2.0, dtype=torch.float32).share_memory_()
    b = torch.full((M, M), 3.0, dtype=torch.float32).share_memory_()
    sum_buf = torch.zeros((M, M), dtype=torch.float32).share_memory_()
    sub_buf = torch.zeros((M, M), dtype=torch.float32).share_memory_()
    expected_sum = torch.full((M, M), 5.0, dtype=torch.float32)
    expected_sub = torch.full((M, M), -1.0, dtype=torch.float32)

    rt = compiled.prepare()
    try:
        handle = rt.register(compiled)
        handle(a, b, sum_buf, sub_buf)
        assert torch.allclose(sum_buf, expected_sum, rtol=1e-5, atol=1e-5), (
            f"chip-0 sum wrong: max diff {(sum_buf - expected_sum).abs().max().item()}"
        )
        assert torch.allclose(sub_buf, expected_sub, rtol=1e-5, atol=1e-5), (
            f"chip-1 diff wrong: max diff {(sub_buf - expected_sub).abs().max().item()}"
        )

        sum_buf.zero_()
        sub_buf.zero_()
        rt.run(compiled, a, b, sum_buf, sub_buf)
        assert torch.allclose(sum_buf, expected_sum, rtol=1e-5, atol=1e-5)
        assert torch.allclose(sub_buf, expected_sub, rtol=1e-5, atol=1e-5)
    finally:
        rt.close()

    with pytest.raises(RuntimeError, match="called after close"):
        rt.register(compiled)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))

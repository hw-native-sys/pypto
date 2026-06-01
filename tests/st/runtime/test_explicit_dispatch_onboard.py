# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board verification of the explicit-dispatch runtime API (PR #1599).

The ``Worker`` / ``ChipWorker`` / ``DistributedWorker`` refactor introduced an
explicit dispatch surface: ``register(compiled) -> RegistrationHandle`` and
``run(compiled, *args)``, plus centralized ``DeviceTensor`` lifecycle
(``alloc_tensor`` tracked in an owned-set, reclaimed by ``close()``). The unit
tests under ``tests/ut/runtime/`` cover the contracts with the simpler backend
mocked; this file drives the same surface on real silicon so the diagnostic
counters, device-memory free, and dispatch loop are exercised end to end.

The training-loop scenario covered here uploads static weights once via
``alloc_tensor``, registers two distinct callables (a forward + backward
stand-in) through a helper typed against the ``Worker`` ABC — so the shared
base-type surface is exercised, not just the concrete ``ChipWorker`` — then
dispatches them in a sustained loop and asserts:

1. ``aicpu_dlopen_count == 2`` (one dlopen per distinct callable) and it does
   **not** grow across the loop — repeated dispatch of an already dlopened
   callable must not re-dlopen.
2. ``close()`` auto-frees every still-owned ``DeviceTensor`` and emits no
   ``_close_owned_tensors`` "leaking" warning.
3. Per-dispatch host latency of the registered-handle path is reported
   (informational, not a hard gate).

Run on hardware via ``task-submit`` (which sets the locked device id)::

    task-submit --device auto --device-num 1 --run 'cd <repo> && \
        export PYTHONPATH=<repo>/python:$PYTHONPATH && \
        python -m pytest tests/st/runtime/test_explicit_dispatch_onboard.py \
        -v --platform a2a3 --device $TASK_DEVICE'

or on the host simulator with ``--platform a2a3sim`` (no device lock).
"""

import logging
import os
import sys
import tempfile
import time

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.runtime import ChipWorker, Worker
from pypto.runtime.runner import RunConfig

# Compiled programs default to the tensormap_and_ringbuffer runtime; the
# ChipWorker must be constructed with the same runtime or the binding check
# in ``register`` rejects the compiled program.
_RUNTIME = "tensormap_and_ringbuffer"

_log = logging.getLogger(__name__)

M = 128
_LEAK_LOGGER = "pypto.runtime.runtime_base"

# A sustained dispatch loop proves the dlopen count and owned-tensor set stay
# flat across many calls; a few dozen iterations exercise the scenario without
# a long device hold. Overridable via env so the host simulator (seconds per
# dispatch) can validate the flow with an even smaller count.
LOOP_ITERS = int(os.environ.get("PYPTO_DISPATCH_LOOP_ITERS", "100"))


@pl.program
class ForwardAddProgram:
    """Forward stand-in: ``c = a + b`` on one 128x128 tile."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, M], pl.FP32],
        b: pl.Tensor[[M, M], pl.FP32],
        c: pl.Out[pl.Tensor[[M, M], pl.FP32]],
    ) -> pl.Tensor[[M, M], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            tile_a = pl.load(a, [0, 0], [M, M])
            tile_b = pl.load(b, [0, 0], [M, M])
            pl.store(pl.add(tile_a, tile_b), [0, 0], c)
        return c


@pl.program
class BackwardMulProgram:
    """Backward stand-in: ``c = a * b`` on one 128x128 tile (distinct callable)."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, M], pl.FP32],
        b: pl.Tensor[[M, M], pl.FP32],
        c: pl.Out[pl.Tensor[[M, M], pl.FP32]],
    ) -> pl.Tensor[[M, M], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            tile_a = pl.load(a, [0, 0], [M, M])
            tile_b = pl.load(b, [0, 0], [M, M])
            pl.store(pl.mul(tile_a, tile_b), [0, 0], c)
        return c


@pytest.fixture(autouse=True)
def _pto_isa_root(test_config):
    """Resolve PTO_ISA_ROOT before any on-board runtime build.

    Compiling the tensormap_and_ringbuffer runtime for real silicon needs the
    PTO ISA sources. The PTOTestCase harness resolves this once up front; tests
    here drive ChipWorker directly, so mirror that bootstrap. Honors an existing
    PTO_ISA_ROOT env var and otherwise auto-clones (same as the harness). The
    simulator build does not need it, so skip the resolution for ``*sim``.
    """
    if str(test_config.platform).endswith("sim"):
        return
    from pypto.runtime.device_runner import ensure_pto_isa_root

    ensure_pto_isa_root(commit=test_config.pto_isa_commit, clone_protocol="https")


@pytest.fixture
def _no_leak_warning(caplog):
    """Fail if ``_close_owned_tensors`` logs a free-failure ("leaking") warning."""
    with caplog.at_level(logging.WARNING, logger=_LEAK_LOGGER):
        yield caplog
    leaks = [r for r in caplog.records if "leaking" in r.getMessage()]
    assert not leaks, f"close() emitted owned-tensor leak warnings: {[r.getMessage() for r in leaks]}"


def _register_through_worker(rt: Worker, compiled):
    """Register via the shared ``Worker`` ABC surface (not the concrete subtype).

    A library helper that only depends on the base type still drives a concrete
    ``ChipWorker`` — the unified-signature property of the refactor.
    """
    return rt.register(compiled)


def test_l2_training_loop_explicit_dispatch(test_config, _no_leak_warning):
    """Weights uploaded once, two callables registered, sustained dispatch loop."""
    platform = test_config.platform
    with tempfile.TemporaryDirectory(prefix="pypto_l2_dispatch_") as work_dir:
        fwd = ir.compile(ForwardAddProgram, output_dir=f"{work_dir}/fwd", platform=platform, dump_passes=False)
        bwd = ir.compile(BackwardMulProgram, output_dir=f"{work_dir}/bwd", platform=platform, dump_passes=False)

        a = torch.full((M, M), 2.0, dtype=torch.float32)
        b = torch.full((M, M), 3.0, dtype=torch.float32)

        worker = ChipWorker(config=RunConfig(platform=platform, device_id=test_config.device_id), runtime=_RUNTIME)
        try:
            assert worker.aicpu_dlopen_count == 0, "no callable dlopened before any dispatch"

            # Static weights: uploaded once, reused for every dispatch in the loop.
            a_dev = worker.alloc_tensor((M, M), torch.float32, init=a)
            b_dev = worker.alloc_tensor((M, M), torch.float32, init=b)
            assert len(worker._owned_tensors) == 2

            # Register through the Worker-ABC-typed helper to exercise the
            # shared base-type surface, not just the concrete ChipWorker.
            h_fwd = _register_through_worker(worker, fwd)
            h_bwd = _register_through_worker(worker, bwd)

            c_fwd = torch.zeros((M, M), dtype=torch.float32)
            c_bwd = torch.zeros((M, M), dtype=torch.float32)

            # First dispatch of each distinct callable triggers its (single) dlopen.
            h_fwd(a_dev, b_dev, c_fwd)
            h_bwd(a_dev, b_dev, c_bwd)
            assert torch.allclose(c_fwd, a + b, rtol=1e-5, atol=1e-5)
            assert torch.allclose(c_bwd, a * b, rtol=1e-5, atol=1e-5)

            dlopen_after_first = worker.aicpu_dlopen_count
            assert dlopen_after_first == 2, (
                f"two distinct callables must dlopen exactly twice, got {dlopen_after_first}"
            )

            # Sustained loop: dispatch both handles repeatedly. The dlopen count
            # must stay flat (no re-dlopen on cached callables) and no tensor is
            # leaked or double-allocated.
            start = time.perf_counter()
            for _ in range(LOOP_ITERS):
                h_fwd(a_dev, b_dev, c_fwd)
                h_bwd(a_dev, b_dev, c_bwd)
            elapsed = time.perf_counter() - start

            assert worker.aicpu_dlopen_count == 2, (
                f"dlopen count grew across loop: {dlopen_after_first} -> {worker.aicpu_dlopen_count}"
            )
            assert torch.allclose(c_fwd, a + b, rtol=1e-5, atol=1e-5)
            assert torch.allclose(c_bwd, a * b, rtol=1e-5, atol=1e-5)
            assert len(worker._owned_tensors) == 2, "owned-tensor set must be stable across the loop"

            per_dispatch_us = elapsed / (LOOP_ITERS * 2) * 1e6
            _log.info(
                "%d iters x 2 handles: %.3fs total, %.1f us/dispatch "
                "(host wall, registered-handle path); aicpu_dlopen_count=%d",
                LOOP_ITERS,
                elapsed,
                per_dispatch_us,
                worker.aicpu_dlopen_count,
            )
        finally:
            worker.close()

        # close() auto-freed both weights without an explicit free_tensor.
        assert len(worker._owned_tensors) == 0, "close() must reclaim all owned DeviceTensors"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))

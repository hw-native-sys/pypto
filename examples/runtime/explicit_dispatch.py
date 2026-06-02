# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit runtime dispatch with ChipWorker.run / register.

Three end-to-end patterns showing how to use ChipWorker explicitly instead of
relying on ``with ChipWorker(): compiled(...)`` ContextVar discovery:

  mode_a_inference_service — pre-register N compiled kernels, dispatch by name
  mode_b_training_loop      — persistent weight DeviceTensor + hot register/run loop
  mode_c_register_dispatch_overhead — register warms the callable cache once
                                       (uses its own ChipWorker — see note in the mode)

All three modes use a ``@pl.jit`` kernel and the new
:meth:`JITFunction.compile` entry point so the same script demonstrates:

  1. ``my_kernel.compile(*sample_args) -> CompiledProgram`` (stage compile)
  2. ``worker.register(compiled) -> RegistrationHandle`` (eager registration)
  3. ``handle(*args)`` (dispatch — repeat in a hot loop)
  4. Long-lived ``worker.alloc_tensor`` for weights / KV cache
  5. ``worker.close()`` cleaning up cids + DeviceTensors in one shot

Run:  python examples/runtime/explicit_dispatch.py
"""

from __future__ import annotations

import pypto.language as pl
import torch
from pypto.runtime import ChipWorker, RunConfig

# ---------------------------------------------------------------------------
# Kernels (shared across the three modes)
# ---------------------------------------------------------------------------


@pl.jit
def tile_add_128(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """c = a + b on 128x128 fp32 tiles."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, tile_b)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit
def tile_mul_128(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """c = a * b on 128x128 fp32 tiles."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_c = pl.mul(tile_a, tile_b)
        pl.store(tile_c, [0, 0], c)
    return c


# ---------------------------------------------------------------------------
# Mode A — inference service: pre-register multiple kernels, dispatch by name
# ---------------------------------------------------------------------------


def mode_a_inference_service(worker: ChipWorker) -> None:
    """Pattern: long-lived service holds one ``ChipWorker`` and a dict of
    pre-registered handles. Per-request work is just ``handles[op](...)``.

    Compile the JIT kernels once with sample args, ``worker.register`` each
    CompiledProgram, then dispatch many times without re-checking the cid
    cache.
    """
    # Sample inputs only used to drive JIT specialization (shape/dtype only,
    # values are not read by ``compile``).
    sample = torch.zeros((128, 128), dtype=torch.float32)
    handles = {
        "add": worker.register(tile_add_128.compile(sample, sample, sample.clone())),
        "mul": worker.register(tile_mul_128.compile(sample, sample, sample.clone())),
    }

    # Per-request dispatch — no cid lookup, no compile-and-assemble.
    a = torch.full((128, 128), 2.0, dtype=torch.float32)
    b = torch.full((128, 128), 3.0, dtype=torch.float32)
    out = torch.zeros((128, 128), dtype=torch.float32)

    handles["add"](a, b, out)
    assert torch.allclose(out, a + b, rtol=1e-5, atol=1e-5)

    handles["mul"](a, b, out)
    assert torch.allclose(out, a * b, rtol=1e-5, atol=1e-5)

    print("mode A OK — pre-registered handles dispatched correctly")


# ---------------------------------------------------------------------------
# Mode B — training loop: persistent weight DeviceTensor, hot register/run loop
# ---------------------------------------------------------------------------


def mode_b_training_loop(worker: ChipWorker) -> None:
    """Pattern: training / inference loop with a static weight tensor uploaded
    to device once and reused across many dispatches.

    ``worker.alloc_tensor(init=host_weight)`` performs malloc + H2D in one
    rollback-safe step and tracks the DeviceTensor in ``worker._owned_tensors``
    so ``worker.close()`` releases it even if the loop forgets ``free_tensor``.
    """
    # Pretend ``host_weight`` is a per-model static buffer (e.g. an MLP weight).
    host_weight = torch.full((128, 128), 0.5, dtype=torch.float32)
    w_dev = worker.alloc_tensor(host_weight.shape, host_weight.dtype, init=host_weight)

    # Compile the kernel once for the shape/dtype combination we'll dispatch
    # with — DeviceTensor and torch.Tensor are interchangeable shape-wise.
    sample = torch.zeros((128, 128), dtype=torch.float32)
    compiled = tile_add_128.compile(sample, sample, sample.clone())
    h = worker.register(compiled)

    # Hot loop: per-step input + persistent weight + per-step output.
    # The weight DeviceTensor has ``child_memory=True`` semantics — no H2D
    # cost per step.
    for step in range(5):
        x = torch.full((128, 128), float(step), dtype=torch.float32)
        out = torch.zeros((128, 128), dtype=torch.float32)
        h(x, w_dev, out)
        expected = x + host_weight
        assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), f"step {step} mismatch"

    # Explicit free even though ``worker.close()`` would auto-free —
    # documented best practice (and shown in 00-getting_started.md caveats).
    worker.free_tensor(w_dev)
    print("mode B OK — 5 dispatches with persistent weight DeviceTensor")


# ---------------------------------------------------------------------------
# Mode C — register warms the callable cache once
# ---------------------------------------------------------------------------


def mode_c_register_dispatch_overhead() -> None:
    """Demonstrate that ``worker.register(compiled)`` triggers exactly one
    AICPU dlopen for that callable, and subsequent ``handle(*args)`` calls
    reuse it.

    The diagnostic counter ``ChipWorker.aicpu_dlopen_count`` is a direct
    passthrough to the underlying simpler worker — useful for benchmarking
    and for verifying the cid cache is doing its job.

    **Uses its own ChipWorker** rather than sharing with the other modes:
    ``ChipWorker._cid_cache`` is keyed by ``id(chip_callable)``, so once
    ``tile_add_128`` is registered on a worker (e.g. by ``mode_a`` above),
    a second ``worker.register`` of the same kernel would be a cache hit
    and would NOT bump ``aicpu_dlopen_count``. A fresh worker guarantees
    the first ``register`` is observable.
    """
    worker = ChipWorker(config=RunConfig())
    try:
        before = worker.aicpu_dlopen_count

        sample = torch.zeros((128, 128), dtype=torch.float32)
        compiled = tile_add_128.compile(sample, sample, sample.clone())
        h = worker.register(compiled)

        after_register = worker.aicpu_dlopen_count

        a = torch.full((128, 128), 1.0, dtype=torch.float32)
        b = torch.full((128, 128), 2.0, dtype=torch.float32)
        out = torch.zeros((128, 128), dtype=torch.float32)
        for _ in range(20):
            h(a, b, out)

        after_runs = worker.aicpu_dlopen_count

        assert after_register == before + 1, (
            f"register should dlopen exactly once: before={before}, after_register={after_register}"
        )
        assert after_runs == after_register, (
            f"20 dispatches should not re-dlopen: after_register={after_register}, after_runs={after_runs}"
        )
        print(
            f"mode C OK — aicpu_dlopen_count: "
            f"{before} -> {after_register} (1 register) -> {after_runs} (20 runs)"
        )
    finally:
        worker.close()


# ---------------------------------------------------------------------------
# Entry point — run all three modes against one worker
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Modes A and B share a worker (init / close amortised); mode C uses
    # its own to keep the ``aicpu_dlopen_count`` assertion meaningful.
    # ``RunConfig()`` picks the default simulator platform; pass
    # ``RunConfig(platform="a2a3")`` to target a real device.
    worker = ChipWorker(config=RunConfig())
    try:
        mode_a_inference_service(worker)
        mode_b_training_loop(worker)
    finally:
        worker.close()

    mode_c_register_dispatch_overhead()

    print("OK")

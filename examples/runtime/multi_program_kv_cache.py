# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Multi-program L3 dispatch sharing one worker and one resident KV cache (JIT style).

Serving skeleton (Qwen3-style): ``prefill`` and ``decode`` are two separate
``@pl.jit.host`` kernels that must share **one** L3 worker and **one**
worker-resident :class:`~pypto.runtime.DeviceTensor` KV cache. ``prefill``
writes the KV cache once; each ``decode`` step reads and updates it — so the KV
cache must survive across dispatches from *different* compiled programs, which a
worker-per-program design cannot do.

Key APIs:

  1. ``@pl.jit.host`` → ``@pl.jit`` (chip orch) → ``@pl.jit.incore`` — the JIT
     three-level L3 kernel; shapes specialize from the sample args.
  2. ``kernel.compile(*sample_args, config=RunConfig(distributed_config=...))``
     — compile without dispatching; returns a ``DistributedCompiledProgram``.
  3. ``DistributedWorker([prefill, decode])`` — prepare both programs on one
     worker (or, symmetrically, ``prefill.prepare(extra_compiled=[decode])``).
  4. ``rt.alloc_tensor(...)`` — a resident KV-cache DeviceTensor, valid across
     dispatches from either program.
  5. ``rt.run(compiled, *args)`` — pick which prepared program to dispatch.
     ``rt(*args)`` is intentionally disabled in multi-program mode (ambiguous).
  6. ``rt.close()`` — release the worker + all DeviceTensors in one shot.

To keep the math trivial and the focus on the dispatch/sharing API:

  prefill:  kv = prompt + prompt            (writes the KV cache)
  decode:   logits = token + kv             (reads the KV cache, per step)

This needs a real device + the ``simpler`` runtime; without one it prints a
skip notice instead of failing.

Run:  python examples/runtime/multi_program_kv_cache.py
"""

from __future__ import annotations

import pypto.language as pl
import torch
from pypto.ir.distributed_compiled_program import DistributedCompiledProgram, DistributedConfig
from pypto.runtime import DistributedWorker, RunConfig

TILE = 128


# ---------------------------------------------------------------------------
# Program 1 — prefill: write the KV cache (kv = prompt + prompt)
# ---------------------------------------------------------------------------


@pl.jit.incore
def write_kv(prompt: pl.Tensor, kv: pl.Out[pl.Tensor]):
    tile_p = pl.load(prompt, [0, 0], [TILE, TILE])
    tile_kv = pl.add(tile_p, tile_p)
    return pl.store(tile_kv, [0, 0], kv)


@pl.jit
def prefill_chip(prompt: pl.Tensor, kv: pl.Out[pl.Tensor]):
    out_kv = write_kv(prompt, kv)
    return out_kv


@pl.jit.host
def prefill(prompt: pl.Tensor, kv: pl.Out[pl.Tensor]):
    out_kv = prefill_chip(prompt, kv)
    return out_kv


# ---------------------------------------------------------------------------
# Program 2 — decode: read the KV cache (logits = token + kv)
# ---------------------------------------------------------------------------


@pl.jit.incore
def read_kv(token: pl.Tensor, kv: pl.Tensor, logits: pl.Out[pl.Tensor]):
    tile_t = pl.load(token, [0, 0], [TILE, TILE])
    tile_kv = pl.load(kv, [0, 0], [TILE, TILE])
    tile_o = pl.add(tile_t, tile_kv)
    return pl.store(tile_o, [0, 0], logits)


@pl.jit
def decode_chip(token: pl.Tensor, kv: pl.Tensor, logits: pl.Out[pl.Tensor]):
    out = read_kv(token, kv, logits)
    return out


@pl.jit.host
def decode(token: pl.Tensor, kv: pl.Tensor, logits: pl.Out[pl.Tensor]):
    out = decode_chip(token, kv, logits)
    return out


# ---------------------------------------------------------------------------
# Serving loop — one worker, one resident KV cache, two programs
# ---------------------------------------------------------------------------


def serve(platform: str = "a2a3", device_ids: list[int] | None = None) -> None:
    dc = DistributedConfig(device_ids=device_ids or [0], num_sub_workers=0, block_dim=3)
    cfg = RunConfig(platform=platform, distributed_config=dc)

    # Per-call IO buffers must be shared-memory host tensors allocated BEFORE the
    # worker forks, so the chip worker sees them through the inherited mapping.
    host_prompt = torch.full((TILE, TILE), 2.0, dtype=torch.float32).share_memory_()
    host_token = torch.zeros((TILE, TILE), dtype=torch.float32).share_memory_()
    host_logits = torch.zeros((TILE, TILE), dtype=torch.float32).share_memory_()

    # compile() specializes on the sample shapes/dtypes without dispatching;
    # passing distributed_config makes it return a DistributedCompiledProgram
    # (assert to narrow the CompiledProgram | DistributedCompiledProgram union).
    kv_sample = torch.zeros((TILE, TILE), dtype=torch.float32)
    prefill_c = prefill.compile(host_prompt, kv_sample, config=cfg)
    decode_c = decode.compile(host_token, kv_sample, host_logits, config=cfg)
    assert isinstance(prefill_c, DistributedCompiledProgram)
    assert isinstance(decode_c, DistributedCompiledProgram)

    # Both programs prepared on ONE worker. Equivalent symmetric form:
    #   with prefill_c.prepare(extra_compiled=[decode_c]) as rt:
    with DistributedWorker([prefill_c, decode_c]) as rt:
        # Resident KV cache: written by prefill, read by every decode step.
        kv_cache = rt.alloc_tensor((TILE, TILE), torch.float32)

        rt.run(prefill_c, host_prompt, kv_cache)  # kv_cache = 2 * prompt = 4.0

        for step in range(3):
            host_token.fill_(float(step))  # refresh per-step input in place
            host_logits.zero_()
            rt.run(decode_c, host_token, kv_cache, host_logits)  # logits = token + kv

            expected = torch.full((TILE, TILE), float(step) + 4.0, dtype=torch.float32)
            torch.testing.assert_close(host_logits, expected, rtol=1e-5, atol=1e-5)

        rt.free_tensor(kv_cache)

    print("OK — prefill wrote the KV cache; 3 decode steps read it on one shared worker")


if __name__ == "__main__":
    # L3 distributed dispatch needs a real device + the ``simpler`` runtime.
    # Pass the platform / device ids your host exposes (e.g. serve("a2a3", [0])).
    try:
        serve()
    except Exception as exc:  # noqa: BLE001 — example: degrade gracefully without a device
        print(f"skipped (needs a device + simpler runtime): {type(exc).__name__}: {exc}")

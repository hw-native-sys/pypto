# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Multi-program L3 dispatch sharing one worker and one resident KV cache.

Serving skeleton (Qwen3-style): ``prefill`` and ``decode`` are two separate
distributed HOST programs that must share **one** L3 worker and **one**
worker-resident :class:`~pypto.runtime.DeviceTensor` KV cache. ``prefill``
writes the KV cache once; each ``decode`` step reads and updates it — so the KV
cache must survive across dispatches from *different* compiled programs, which a
worker-per-program design cannot do.

Key APIs:

  1. ``DistributedWorker([prefill, decode])`` — prepare both programs on one
     worker (or, symmetrically, ``prefill.prepare(extra_compiled=[decode])``).
  2. ``rt.alloc_tensor(...)`` — a resident KV-cache DeviceTensor, valid across
     dispatches from either program.
  3. ``rt.run(compiled, *args)`` — pick which prepared program to dispatch.
     ``rt(*args)`` is intentionally disabled in multi-program mode (ambiguous).
  4. ``rt.close()`` — release the worker + all DeviceTensors in one shot.

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
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedCompiledProgram, DistributedConfig
from pypto.runtime import DistributedWorker

TILE = 128


# ---------------------------------------------------------------------------
# Program 1 — prefill: write the KV cache (kv = prompt + prompt)
# ---------------------------------------------------------------------------


@pl.program
class PrefillProgram:
    """L3: HOST orch → CHIP worker. Writes the KV cache from the prompt."""

    @pl.function(type=pl.FunctionType.InCore)
    def write_kv(
        self,
        prompt: pl.Tensor[[TILE, TILE], pl.FP32],
        kv: pl.Out[pl.Tensor[[TILE, TILE], pl.FP32]],
    ) -> pl.Tensor[[TILE, TILE], pl.FP32]:
        tile_p = pl.load(prompt, [0, 0], [TILE, TILE])
        tile_kv = pl.add(tile_p, tile_p)
        return pl.store(tile_kv, [0, 0], kv)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        prompt: pl.Tensor[[TILE, TILE], pl.FP32],
        kv: pl.Out[pl.Tensor[[TILE, TILE], pl.FP32]],
    ) -> pl.Tensor[[TILE, TILE], pl.FP32]:
        return self.write_kv(prompt, kv)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        prompt: pl.Tensor[[TILE, TILE], pl.FP32],
        kv: pl.Out[pl.Tensor[[TILE, TILE], pl.FP32]],
    ) -> pl.Tensor[[TILE, TILE], pl.FP32]:
        out_kv: pl.Tensor[[TILE, TILE], pl.FP32] = self.chip_orch(prompt, kv)
        return out_kv


# ---------------------------------------------------------------------------
# Program 2 — decode: read the KV cache (logits = token + kv)
# ---------------------------------------------------------------------------


@pl.program
class DecodeProgram:
    """L3: HOST orch → CHIP worker. Reads the KV cache to produce logits."""

    @pl.function(type=pl.FunctionType.InCore)
    def read_kv(
        self,
        token: pl.Tensor[[TILE, TILE], pl.FP32],
        kv: pl.Tensor[[TILE, TILE], pl.FP32],
        logits: pl.Out[pl.Tensor[[TILE, TILE], pl.FP32]],
    ) -> pl.Tensor[[TILE, TILE], pl.FP32]:
        tile_t = pl.load(token, [0, 0], [TILE, TILE])
        tile_kv = pl.load(kv, [0, 0], [TILE, TILE])
        tile_o = pl.add(tile_t, tile_kv)
        return pl.store(tile_o, [0, 0], logits)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        token: pl.Tensor[[TILE, TILE], pl.FP32],
        kv: pl.Tensor[[TILE, TILE], pl.FP32],
        logits: pl.Out[pl.Tensor[[TILE, TILE], pl.FP32]],
    ) -> pl.Tensor[[TILE, TILE], pl.FP32]:
        return self.read_kv(token, kv, logits)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        token: pl.Tensor[[TILE, TILE], pl.FP32],
        kv: pl.Tensor[[TILE, TILE], pl.FP32],
        logits: pl.Out[pl.Tensor[[TILE, TILE], pl.FP32]],
    ) -> pl.Tensor[[TILE, TILE], pl.FP32]:
        out: pl.Tensor[[TILE, TILE], pl.FP32] = self.chip_orch(token, kv, logits)
        return out


# ---------------------------------------------------------------------------
# Serving loop — one worker, one resident KV cache, two programs
# ---------------------------------------------------------------------------


def serve(platform: str = "a2a3", device_ids: list[int] | None = None) -> None:
    dc = DistributedConfig(device_ids=device_ids or [0], num_sub_workers=0, block_dim=3)
    # Passing distributed_config makes compile() return DistributedCompiledProgram;
    # assert to narrow the CompiledProgram | DistributedCompiledProgram union.
    prefill = ir.compile(PrefillProgram, platform=platform, distributed_config=dc)
    decode = ir.compile(DecodeProgram, platform=platform, distributed_config=dc)
    assert isinstance(prefill, DistributedCompiledProgram)
    assert isinstance(decode, DistributedCompiledProgram)

    # Per-call IO buffers must be shared-memory host tensors allocated BEFORE the
    # worker forks, so the chip worker sees them through the inherited mapping.
    host_prompt = torch.full((TILE, TILE), 2.0, dtype=torch.float32).share_memory_()
    host_token = torch.zeros((TILE, TILE), dtype=torch.float32).share_memory_()
    host_logits = torch.zeros((TILE, TILE), dtype=torch.float32).share_memory_()

    # Both programs prepared on ONE worker. Equivalent symmetric form:
    #   with prefill.prepare(extra_compiled=[decode]) as rt:
    with DistributedWorker([prefill, decode]) as rt:
        # Resident KV cache: written by prefill, read by every decode step.
        kv_cache = rt.alloc_tensor((TILE, TILE), torch.float32)

        rt.run(prefill, host_prompt, kv_cache)  # kv_cache = 2 * prompt = 4.0

        for step in range(3):
            host_token.fill_(float(step))  # refresh per-step input in place
            host_logits.zero_()
            rt.run(decode, host_token, kv_cache, host_logits)  # logits = token + kv

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

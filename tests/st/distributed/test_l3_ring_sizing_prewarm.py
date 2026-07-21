# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end L3 arena prewarm via ``DistributedCompiledProgram.prepare(config=...)``.

Exercises the runtime-arena prewarm path: ``prepare(rc)`` eagerly builds the
prebuilt runtime arena for ``rc``'s ring sizing at worker ``init``, and each
dispatch passes the **same** ``rc`` so its sizing key matches the prewarmed
arena (first dispatch hits the cache instead of paying the cold build). Keeping
the sizing constant across the worker's dispatches is the single-slot-cache
guidance from ``docs/en/dev/05-runtime-ring-sizing.md``.

This is a **functional** check — it asserts the prewarm-then-dispatch path with
an explicit, non-default ring sizing produces correct results
(``f = a + b``). It does not assert timing (the ~800ms cold-build saving is not
observable without flakiness). On the host simulator the prewarm is a no-op
(no prebuilt arena); the dispatch path still runs and is validated there.
"""

import sys

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.runtime import RunConfig


@pl.program
class L3AddProgram:
    """L3: HOST orch → CHIP worker (``f = a + b``)."""

    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_f = pl.add(tile_a, tile_b)
        return pl.store(tile_f, [0, 0], f)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_f = self.tile_add(a, b, f)
        return out_f

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_f: pl.Tensor[[128, 128], pl.FP32] = self.chip_orch(a, b, f)
        return out_f


class TestL3RingSizingPrewarm:
    """prepare(config=ring-sized) prewarms; dispatches reuse that sizing."""

    def test_prewarmed_sizing_matches_dispatch(self, test_config, device_ids):
        """``f = a + b`` over two dispatches under an explicit, prewarmed ring sizing.

        Verifies that:
        1. ``prepare(rc)`` accepts a ring-sized ``RunConfig`` and prewarms it at
           ``init`` (the prebuilt-arena build lands in setup, onboard).
        2. Dispatching with the **same** ``rc`` — the sizing the arena was
           prewarmed for — computes correctly across repeated reuse of the
           handle (first dispatch hits the prewarmed arena onboard).
        """
        if not device_ids:
            pytest.skip("L3 ring-sizing prewarm test needs at least one device")

        compiled = ir.compile(
            L3AddProgram,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:1],
                num_sub_workers=0,
                block_dim=3,
                aicpu_thread_num=4,
            ),
        )

        # A non-default, valid ring sizing (power-of-2 window/heap; dep_pool in
        # range). The same rc is passed to prepare() (prewarm) and to every
        # dispatch, so the prewarmed arena's sizing key matches the run's.
        rc = RunConfig(
            platform=test_config.platform,
            ring_task_window=64,
            ring_heap=8 * 1024 * 1024,
            ring_dep_pool=256,
        )

        # Shared-memory host buffers MUST be allocated before prepare() so the
        # forked chip worker inherits their mappings; per-call IO is reused in
        # place across dispatches.
        host_a = torch.zeros((128, 128), dtype=torch.float32).share_memory_()
        host_b = torch.zeros((128, 128), dtype=torch.float32).share_memory_()
        host_out = torch.zeros((128, 128), dtype=torch.float32).share_memory_()

        with compiled.prepare(rc) as rt:
            for a_val, b_val in ((2.0, 3.0), (7.0, 4.0)):
                host_a.fill_(a_val)
                host_b.fill_(b_val)
                host_out.zero_()
                rt(host_a, host_b, host_out, config=rc)

                expected = torch.full((128, 128), a_val + b_val, dtype=torch.float32)
                torch.testing.assert_close(host_out, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])

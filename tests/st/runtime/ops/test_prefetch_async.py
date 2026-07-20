# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System test for the ``pl.prefetch.*`` async GM->L2 prefetch op family.

Mirrors the pto-isa single-card ST
(``tests/npu/<arch>/src/st/testcase/tprefetch_async``): prefetch a GM region,
wait on the event, then copy the region out and check the bytes match. The
prefetch is a pure cache hint that changes no tensor value, so the property
under test is **non-interference** — plus the fact that the event/session wait
actually completes rather than hanging.

The workspace follows the same contract as that ST::

    SdmaWorkspaceManager mgr; bool ok = mgr.Init();
    uint8_t *ws = ok ? (uint8_t *)mgr.GetWorkspaceAddr() : nullptr;

i.e. a *host-initialized* SDMA context, obtained here via
:meth:`Worker.sdma_prefetch_workspace_addr`. It must never be a plain
allocation: the device-side session init only rejects a null workspace, so an
uninitialized non-null buffer yields garbage SQ base addresses and hangs the
kernel (AICore 507018).
"""

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.runtime import ChipWorker, DeviceTensor, RunConfig

ROWS = 1
COLS = 128

# pto-isa allocates the SDMA context itself (`kSdmaWorkspaceBytes` = 16KB); this
# shape only has to cover it so the IR-level byte extent is not smaller than
# what the device side addresses.
WORKSPACE_BYTES = 65536


@pl.program
class PrefetchAsyncProgram:
    """Warm L2 with `a`, wait for completion, then copy `a` to `out`."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[ROWS, COLS], pl.FP32],
        ws: pl.Tensor[[WORKSPACE_BYTES], pl.INT8],
        out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        ctx = pl.prefetch.make_context(ws)
        evt = pl.prefetch.async_prefetch(a, ctx)
        session = pl.prefetch.session(ctx)
        # Blocks until the prefetch lands, so `a` is resident in L2 below.
        pl.prefetch.wait(evt, session)

        tile_a: pl.Tile[[ROWS, COLS], pl.FP32] = pl.load(a, [0, 0], [ROWS, COLS])
        out = pl.store(tile_a, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[ROWS, COLS], pl.FP32],
        ws: pl.Tensor[[WORKSPACE_BYTES], pl.INT8],
        out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        out = self.kernel(a, ws, out)
        return out


class TestPrefetchAsync:
    """End-to-end async GM->L2 prefetch on device."""

    def test_prefetch_async_does_not_perturb_data(self, test_config):
        """Prefetching a region then copying it yields a bit-exact copy."""
        compiled = ir.compile(PrefetchAsyncProgram, backend_type=test_config.backend_type)

        a = torch.randn(ROWS, COLS, dtype=torch.float32)
        out = torch.zeros(ROWS, COLS, dtype=torch.float32)

        worker_cfg = RunConfig(platform=test_config.platform, device_id=test_config.device_id)
        with ChipWorker(config=worker_cfg) as w:
            ws_addr = w.sdma_prefetch_workspace_addr()
            if ws_addr == 0:
                pytest.skip(
                    "SDMA prefetch workspace unavailable on this platform "
                    "(needs a CANN exposing aclnnShmemSdmaStarsQuery)"
                )
            # Runtime-owned, host-initialized SDMA context. Bound as a
            # DeviceTensor so the runtime passes the pointer straight through
            # (child_memory: no H2D, no copy-back). Deliberately NOT freed here
            # and NOT a plain alloc_tensor -- see the module docstring.
            ws = DeviceTensor(ws_addr, (WORKSPACE_BYTES,), torch.int8)
            compiled(a, ws, out, config=test_config)

        # The prefetch must not perturb the data -- a plain copy is the golden.
        assert torch.equal(out, a), f"max|err| = {(out - a).abs().max().item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

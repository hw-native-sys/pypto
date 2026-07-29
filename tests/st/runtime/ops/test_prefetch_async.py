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

The generated artifact records that it requires SDMA. Normal compiled-program
execution uses that metadata to create an SDMA-enabled worker; the runtime owns
and injects the workspace consumed by the generated kernel wrapper. Runtime
provisioning is currently covered only on onboard a2a3, so this test deliberately
excludes simulator and a5 targets instead of treating either as a no-op path.
"""

import pypto.language as pl
import pytest
import torch
from pypto import ir

ROWS = 1
COLS = 128


@pl.program
class PrefetchAsyncProgram:
    """Warm L2 with `a`, wait for completion, then copy `a` to `out`."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[ROWS, COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        ctx = pl.prefetch.make_context()
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
        out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        out = self.kernel(a, out)
        return out


@pytest.mark.platforms("a2a3")
class TestPrefetchAsync:
    """End-to-end async GM->L2 prefetch on device."""

    def test_prefetch_async_does_not_perturb_data(self, test_config):
        """Prefetching a region then copying it yields a bit-exact copy."""
        compiled = ir.compile(PrefetchAsyncProgram, backend_type=test_config.backend_type)

        a = torch.randn(ROWS, COLS, dtype=torch.float32)
        out = torch.zeros(ROWS, COLS, dtype=torch.float32)

        compiled(a, out, config=test_config)

        # The prefetch must not perturb the data -- a plain copy is the golden.
        assert torch.equal(out, a), f"max|err| = {(out - a).abs().max().item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

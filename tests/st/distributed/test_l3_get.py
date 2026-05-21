# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: ``pld.tensor.get`` cross-rank read (TGET).

End-to-end contract for the N6 cross-rank GM-to-GM read primitive. ``get`` is
the tensor-level bulk form of ``remote_load + store``: rank ``r`` reads the
``peer`` rank's window-bound ``src`` slice directly into its own window-bound
``dst`` slice.

Scenario:

* Rank ``r`` stages ``inputs[r]`` into its own ``src`` window slice.
* All ranks barrier on the stage-in with ``pld.system.notify`` /
  ``pld.system.wait``.
* Rank ``r`` calls ``pld.tensor.get(dst, peer=(r + 1) % nranks, src=src)``.
* Rank ``r`` reads its local ``dst`` slice back to ``outputs[r]``.

Golden: ``outputs[r] == inputs[(r + 1) % nranks]``.

The test is currently skipped for the same host-side reasons as
``test_l3_remote_load.py`` and PR #1442's ``test_l3_put.py``: the InCore PTO
codegen for the N6 op is present, but the host/runtime distributed glue is not
complete yet.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64


def _build_ring_get_program():
    """Build the ring-get program at call time.

    Deferred construction lets this file collect even when the parser rejects
    the embedded body, for example while ``tile.store`` still rejects a
    ``DistributedTensorType`` destination. The class-level skip marker ensures
    the body does not run until the pending host-side work lands.
    """

    @pl.program
    class RingGet:
        @pl.function(type=pl.FunctionType.InCore)
        def ring_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            src: pld.DistributedTensor[[1, SIZE], pl.FP32],
            dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Phase 1: stage-in — local input -> this rank's own src window slice.
            local = pl.load(inp, [0, 0], [1, SIZE])
            _ = pl.store(local, [0, 0], src)

            # Phase 2: signal the peer that our src slice has been staged, then
            # wait until our own signal cell has been written by the rank that
            # will be our peer in the ring.
            pld.system.notify(
                target=signal,
                peer=peer,
                offsets=[0, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(
                signal=signal,
                offsets=[0, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

            # Phase 3: pull the peer's GM src slice directly into local GM dst.
            pld.tensor.get(dst, peer=peer, src=src)

            # Phase 4: read our local dst slice back and surface it as output.
            recv = pl.load(dst, [0, 0], [1, SIZE])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            src: pld.DistributedTensor[[1, SIZE], pl.FP32],
            dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.ring_step(inp, out, src, dst, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, 1, SIZE], pl.FP32]:
            src_buf = pld.alloc_window_buffer(SIZE * 4)  # 1xSIZE x FP32
            dst_buf = pld.alloc_window_buffer(SIZE * 4)
            signal_buf = pld.alloc_window_buffer(4)  # 1x1 x INT32

            for r in pl.range(pld.world_size()):
                src = pld.window(src_buf, [1, SIZE], dtype=pl.FP32)
                dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], src, dst, signal, (r + 1) % pld.world_size(), device=r)
            return outputs

    return RingGet


@pytest.mark.skip(
    reason=(
        "pld.tensor.get end-to-end requires: (a) tile.store accepting "
        "DistributedTensor destinations (Phase-1 stage-in), (b) N7 host_orch "
        "python codegen emitting add_scalar(ctx) per DistributedTensor, "
        "(c) N8 driver wiring CommGroup window buffers. The InCore PTO codegen "
        "(N6 P1) is in place — drop this skip once (a)-(c) land."
    )
)
class TestL3Get:
    """L3 distributed runtime: cross-rank read via pld.tensor.get."""

    def test_ring_shuffle(self, test_config, device_ids):
        """Rank r reads peer (r + 1) % nranks into local dst."""
        if len(device_ids) < 2:
            pytest.skip(f"ring get needs 2 devices, got {device_ids}")

        program = _build_ring_get_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        inputs = torch.stack(
            [
                torch.arange(SIZE, dtype=torch.float32).reshape(1, SIZE),
                torch.arange(100.0, 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE),
            ]
        )
        outputs = torch.zeros((2, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = torch.stack([inputs[1], inputs[0]])
        assert torch.allclose(outputs, expected), (
            f"ring get mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])

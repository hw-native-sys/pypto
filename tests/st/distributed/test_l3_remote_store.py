# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: ``pld.tile.remote_store`` cross-rank subview write.

Subview-aware push dual of :func:`pld.tile.remote_load`. Where ``remote_load``
reads a region of a peer's tensor into a local tile, ``remote_store`` writes a
local tile into a region of a peer's tensor — the missing tile-level primitive
that lets ``TPUT``-style kernels target a sub-region of a larger remote tensor
(e.g. ``peer.recv_x[loc_e, slot, :]`` keyed by runtime routing decisions).

This ST exercises the same ring-shuffle protocol as :file:`test_l3_put.py`'s
``test_ring_shuffle`` — rank ``r`` pushes its input to ``peer = (r + 1) % nranks``
so each rank's slice is overwritten by ``(r - 1) % nranks`` — but uses
``remote_store`` instead of the whole-tensor :func:`pld.tensor.put`. Golden:
``outputs[r] == inputs[(r - 1) % nranks]``.

In addition to the whole-tile push (matching ``put``'s behavior), a second
case exercises the **subview** capability that ``put`` does not have: the
local input is half the size of the remote window, and each rank pushes its
half to a specific offset of the peer's window. Whichever rank lands at
``offsets=[0, 0]`` covers the lower half; the other half stays unchanged.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
HALF = SIZE // 2


def _build_ring_remote_store_program():
    """Build the ring remote_store program at call time.

    Deferred construction mirrors :file:`test_l3_put.py` — keeps module
    collection independent of parser/verifier edge cases on the embedded body.
    """

    @pl.program
    class RingRemoteStore:
        @pl.function(type=pl.FunctionType.InCore)
        def ring_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Phase 1: push our local input directly into the peer's dst slice
            # via remote_store (no stage-in window needed — remote_store reads
            # from a local tile, not from a DistributedTensor source).
            local = pl.load(inp, [0, 0], [1, SIZE])
            pld.tile.remote_store(local, target=dst, peer=peer, offsets=[0, 0])

            # Phase 2: signal the peer that our write to it has landed, then
            # wait for the rank that targets us ((r - 1) % nranks) to have
            # done the same.
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

            # Phase 3: read our own dst slice back locally — it was written by
            # the rank whose peer is us — and surface it as the output.
            recv = pl.load(dst, [0, 0], [1, SIZE])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.ring_step(inp, out, dst, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, 1, SIZE], pl.FP32]:
            dst_buf = pld.alloc_window_buffer(SIZE * 4)
            signal_buf = pld.alloc_window_buffer(4)

            for r in pl.range(pld.world_size()):
                dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                # Ring partner: rank r pushes to peer = (r + 1) % nranks.
                self.chip_orch(inputs[r], outputs[r], dst, signal, (r + 1) % pld.world_size(), device=r)
            return outputs

    return RingRemoteStore


def _build_subview_remote_store_program():
    """Push half-size local tiles into different halves of the peer's window.

    Exercises the subview capability that :func:`pld.tensor.put` does not have:
    each rank pushes its local ``[1, HALF]`` tile to peer's ``dst[0:HALF]``
    (rank 0 → peer) or peer's ``dst[HALF:SIZE]`` (rank 1 → peer), depending on
    which half the rank is responsible for.

    The two halves are pushed concurrently to the same peer's dst window but
    to disjoint offsets, so the final state on each rank's dst is the
    concatenation of two halves coming from the partner rank. The HCCL window
    is zero-initialised, so the receiving rank's lower half (written by
    partner) and upper half (written by partner) together cover the full
    window — no race condition because the offsets are disjoint.
    """

    @pl.program
    class SubviewRemoteStore:
        @pl.function(type=pl.FunctionType.InCore)
        def half_push_step(
            self,
            inp_low: pl.Tensor[[1, HALF], pl.FP32],
            inp_high: pl.Tensor[[1, HALF], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Push two half-sized tiles into peer's dst at offsets [0, 0] and
            # [0, HALF]. Each remote_store writes a [1, HALF] subview — the
            # capability put doesn't have.
            tile_low = pl.load(inp_low, [0, 0], [1, HALF])
            pld.tile.remote_store(tile_low, target=dst, peer=peer, offsets=[0, 0])
            tile_high = pl.load(inp_high, [0, 0], [1, HALF])
            pld.tile.remote_store(tile_high, target=dst, peer=peer, offsets=[0, HALF])

            pld.system.notify(
                target=signal,
                peer=peer,
                offsets=[0, 0],
                value=1,
                op=pld.NotifyOp.Set,
            )
            pld.system.wait(
                signal=signal,
                offsets=[0, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

            recv = pl.load(dst, [0, 0], [1, SIZE])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp_low: pl.Tensor[[1, HALF], pl.FP32],
            inp_high: pl.Tensor[[1, HALF], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.half_push_step(inp_low, inp_high, out, dst, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs_low: pl.Tensor[[2, 1, HALF], pl.FP32],
            inputs_high: pl.Tensor[[2, 1, HALF], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, 1, SIZE], pl.FP32]:
            dst_buf = pld.alloc_window_buffer(SIZE * 4)
            signal_buf = pld.alloc_window_buffer(4)

            for r in pl.range(pld.world_size()):
                dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                self.chip_orch(
                    inputs_low[r],
                    inputs_high[r],
                    outputs[r],
                    dst,
                    signal,
                    (r + 1) % pld.world_size(),
                    device=r,
                )
            return outputs

    return SubviewRemoteStore


class TestL3RemoteStore:
    """L3 distributed runtime: cross-rank subview write via pld.tile.remote_store."""

    def test_ring_shuffle(self, test_config, device_ids):
        """Non-atomic overwrite: rank r pushes its input to peer (r + 1) % nranks."""
        if len(device_ids) < 2:
            pytest.skip(f"ring remote_store needs 2 devices, got {device_ids}")

        program = _build_ring_remote_store_program()
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

        # outputs[r] = inputs[(r - 1) % nranks] → outputs[0] = inputs[1], etc.
        expected = torch.stack([inputs[1], inputs[0]])
        assert torch.allclose(outputs, expected), (
            f"ring remote_store mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.skip(
        reason="Flaky: PTOAS does not drain MTE pipes before pto.comm.tnotify, "
        "so the second remote tstore can be in flight when the signal lands on peer. "
        "Tracked upstream at hw-native-sys/PTOAS#711."
    )
    def test_subview_halves(self, test_config, device_ids):
        """Subview push: each rank writes two half-tiles to disjoint offsets of peer's dst."""
        if len(device_ids) < 2:
            pytest.skip(f"subview remote_store needs 2 devices, got {device_ids}")

        program = _build_subview_remote_store_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        # rank 0 sends low=[0..HALF) and high=[HALF..SIZE) to rank 1's dst at
        # offsets [0,0] and [0,HALF] — rank 1 ends up with the full sequence.
        # rank 1 sends low=[100..100+HALF) and high=[100+HALF..100+SIZE) to
        # rank 0's dst — rank 0 ends up with the shifted sequence.
        inputs_low = torch.stack(
            [
                torch.arange(HALF, dtype=torch.float32).reshape(1, HALF),
                torch.arange(100.0, 100.0 + HALF, dtype=torch.float32).reshape(1, HALF),
            ]
        )
        inputs_high = torch.stack(
            [
                torch.arange(HALF, SIZE, dtype=torch.float32).reshape(1, HALF),
                torch.arange(100.0 + HALF, 100.0 + SIZE, dtype=torch.float32).reshape(1, HALF),
            ]
        )
        outputs = torch.zeros((2, 1, SIZE), dtype=torch.float32)

        compiled(inputs_low, inputs_high, outputs)

        # Each rank's dst is the concatenation of partner's low + high halves.
        expected_0 = torch.cat([inputs_low[1], inputs_high[1]], dim=1)
        expected_1 = torch.cat([inputs_low[0], inputs_high[0]], dim=1)
        expected = torch.stack([expected_0, expected_1])
        assert torch.allclose(outputs, expected), (
            f"subview remote_store mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 regression for an explicit dependency between communication dispatches.

Issue #2397 exposed a deterministic deadlock when a rank's send and wait lived
in different L3 dispatches. The wait had no tensor producer, became READY at
submission, and entered the rank's worker FIFO before the send, which was still
waiting on an earlier compute result.

This test preserves that exact shape on two ranks:

1. compute;
2. SEND (``remote_store`` + ``notify``);
3. compute;
4. WAIT in a separate dispatch;
5. compute.

Both ranks explicitly make WAIT depend on their own SEND. The dependency is
declared with ``pl.submit(..., deps=[send_task])`` and lowers to the L3
``TaskArgs.add_dep_wait`` interface, on top of the automatic per-rank comm
ordering token, so the two mechanisms must coexist on one dispatch.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

D = 16
P = 2
SIGNAL_ROWS = 8


def _build_split_send_wait_program():
    """Build the two-rank split SEND/WAIT reproducer from issue #2397."""

    @pl.program
    class SplitSendWait:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            inp: pl.Tensor[[D, D], pl.FP32],
            out: pl.Out[pl.Tensor[[D, D], pl.FP32]],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            tile = pl.load(inp, [0, 0], [D, D])
            return pl.store(pl.add(tile, tile), [0, 0], out)

        @pl.function(type=pl.FunctionType.InCore)
        def send(
            self,
            inp: pl.Tensor[[D, D], pl.FP32],
            echo: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            dst: pld.DistributedTensor[[D, D], pl.FP32],
            signal: pld.DistributedTensor[[SIGNAL_ROWS, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            tile = pl.load(inp, [0, 0], [D, D])
            echo = pl.store(tile, [0, 0], echo)
            pld.tile.remote_store(tile, target=dst, peer=peer, offsets=[0, 0])
            pld.system.notify(
                target=signal,
                peer=peer,
                offsets=[0, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            return echo

        @pl.function(type=pl.FunctionType.InCore)
        def recv(
            self,
            out: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            dst: pld.DistributedTensor[[D, D], pl.FP32],
            signal: pld.DistributedTensor[[SIGNAL_ROWS, 1], pl.INT32],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            return pl.store(pl.load(dst, [0, 0], [D, D]), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def compute_orch(
            self,
            inp: pl.Tensor[[D, D], pl.FP32],
            out: pl.Out[pl.Tensor[[D, D], pl.FP32]],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            return self.compute(inp, out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def send_orch(
            self,
            inp: pl.Tensor[[D, D], pl.FP32],
            echo: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            dst: pld.DistributedTensor[[D, D], pl.FP32],
            signal: pld.DistributedTensor[[SIGNAL_ROWS, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            return self.send(inp, echo, dst, signal, peer)

        @pl.function(type=pl.FunctionType.Orchestration)
        def recv_orch(
            self,
            out: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            dst: pld.DistributedTensor[[D, D], pl.FP32],
            signal: pld.DistributedTensor[[SIGNAL_ROWS, 1], pl.INT32],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            return self.recv(out, dst, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[P, D, D], pl.FP32],
            echoes: pl.Out[pl.Tensor[[P, D, D], pl.FP32]],
            forward: pl.Out[pl.Tensor[[P, D, D], pl.FP32]],
            reverse: pl.Out[pl.Tensor[[P, D, D], pl.FP32]],
        ):
            forward_dst_buf = pld.alloc_window_buffer(D * D * pl.FP32.get_byte())
            forward_signal_buf = pld.alloc_window_buffer(SIGNAL_ROWS * pl.INT32.get_byte())
            reverse_dst_buf = pld.alloc_window_buffer(D * D * pl.FP32.get_byte())
            reverse_signal_buf = pld.alloc_window_buffer(SIGNAL_ROWS * pl.INT32.get_byte())

            compute_0 = pl.create_tensor([P, D, D], dtype=pl.FP32)
            compute_1 = pl.create_tensor([P, D, D], dtype=pl.FP32)
            compute_2 = pl.create_tensor([P, D, D], dtype=pl.FP32)

            for rank in pl.range(P):
                forward_dst = pld.window(forward_dst_buf, [D, D], dtype=pl.FP32)
                forward_signal = pld.window(forward_signal_buf, [SIGNAL_ROWS, 1], dtype=pl.INT32)
                reverse_dst = pld.window(reverse_dst_buf, [D, D], dtype=pl.FP32)
                reverse_signal = pld.window(reverse_signal_buf, [SIGNAL_ROWS, 1], dtype=pl.INT32)

                if rank == 0:
                    first = self.compute_orch(inputs[rank], compute_0[rank], device=rank)
                    _sent, send_task = pl.submit(
                        self.send_orch,
                        first,
                        echoes[rank],
                        forward_dst,
                        forward_signal,
                        rank + 1,
                        device=rank,
                    )
                    second = self.compute_orch(first, compute_1[rank], device=rank)
                    _received, _recv_task = pl.submit(
                        self.recv_orch,
                        reverse[rank],
                        reverse_dst,
                        reverse_signal,
                        device=rank,
                        deps=[send_task],
                    )
                    self.compute_orch(second, compute_2[rank], device=rank)
                else:
                    first = self.compute_orch(inputs[rank], compute_0[rank], device=rank)
                    _sent, send_task = pl.submit(
                        self.send_orch,
                        first,
                        echoes[rank],
                        reverse_dst,
                        reverse_signal,
                        rank - 1,
                        device=rank,
                    )
                    second = self.compute_orch(first, compute_1[rank], device=rank)
                    _received, _recv_task = pl.submit(
                        self.recv_orch,
                        forward[rank],
                        forward_dst,
                        forward_signal,
                        device=rank,
                        deps=[send_task],
                    )
                    self.compute_orch(second, compute_2[rank], device=rank)

            return echoes, forward, reverse

    return SplitSendWait


def test_split_send_wait_dispatch_order(test_config, device_ids):
    """Separate WAIT dispatches must not enter a rank FIFO before its SEND."""
    if len(device_ids) < P:
        pytest.skip(f"split SEND/WAIT regression needs {P} devices, got {device_ids}")

    compiled = ir.compile(
        _build_split_send_wait_program(),
        platform=test_config.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids[:P],
            num_sub_workers=0,
        ),
    )

    orch_src = (compiled.output_dir / "orchestration" / "host_orch.py").read_text()
    dep_waits = [line for line in orch_src.splitlines() if ".add_dep_wait(" in line]
    assert len(dep_waits) == P, orch_src
    assert all("send_task" in line for line in dep_waits), dep_waits
    assert "_last_task" not in orch_src

    inputs = torch.arange(P * D * D, dtype=torch.float32).reshape(P, D, D)
    echoes = torch.zeros_like(inputs)
    forward = torch.zeros_like(inputs)
    reverse = torch.zeros_like(inputs)

    compiled(inputs, echoes, forward, reverse)

    doubled = inputs * 2.0
    assert torch.equal(echoes, doubled)
    assert torch.equal(forward[1], doubled[0])
    assert torch.count_nonzero(forward[0]) == 0
    assert torch.equal(reverse[0], doubled[1])
    assert torch.count_nonzero(reverse[1]) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"] + sys.argv[1:]))

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 EP dispatch/combine smoke test.

This is the PyPTO counterpart of the data movement pattern in
``runtime/examples/workers/l3/ep_dispatch_combine``:

* dispatch pushes a row subview into the peer rank's expert input window;
* a local expert placeholder consumes the received rows;
* combine pushes the routed row back to the original owner.

The runtime example also carries dynamic expert histograms, weights, route
indices, and TOPK reduction. This ST intentionally keeps the routing static so
it focuses on the PyPTO capability that EP needs first: offset ``pld.tensor.put``
for row/subview traffic in both dispatch and combine directions.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

N_RANKS = 2
ROWS = 2
D = 64


def _build_ep_dispatch_combine_program():
    """Build a 2-rank EP dispatch/combine program at call time."""

    @pl.program
    class EPDispatchCombine:
        @pl.function(type=pl.FunctionType.InCore)
        def ep_step(
            self,
            routes: pl.Tensor[[ROWS, D], pl.FP32],
            out: pl.Out[pl.Tensor[[ROWS, D], pl.FP32]],
            send_x: pl.InOut[pld.DistributedTensor[[ROWS, D], pl.FP32]],
            recv_x: pl.InOut[pld.DistributedTensor[[ROWS, D], pl.FP32]],
            routed_y: pl.InOut[pld.DistributedTensor[[ROWS, D], pl.FP32]],
            dispatch_signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            combine_signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[ROWS, D], pl.FP32]:
            # Dispatch: row 0 is local; row 1 is routed to the peer's expert.
            local = pl.load(routes, [0, 0], [ROWS, D])
            _ = pl.store(local, [0, 0], send_x)
            _ = pl.store(local, [0, 0], recv_x)
            pld.tensor.put(
                recv_x,
                peer=peer,
                src=send_x,
                dst_offsets=[1, 0],
                src_offsets=[1, 0],
                shape=[1, D],
                atomic=pld.AtomicType.None_,
            )
            pld.system.notify(dispatch_signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
            pld.system.wait(signal=dispatch_signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

            # Local expert placeholder: mirrors the runtime demo's simple
            # per-row expert by doubling the received route payloads.
            expert_in = pl.load(recv_x, [0, 0], [ROWS, D])
            expert_out = pl.add(expert_in, expert_in)
            _ = pl.store(expert_out, [0, 0], recv_x)

            # Combine: keep row 0 locally and return row 1 to the original
            # owner, then read the completed routed buffer.
            local_y = pl.load(recv_x, [0, 0], [ROWS, D])
            _ = pl.store(local_y, [0, 0], routed_y)
            # Row 1 currently belongs to the peer's original token stream.
            # Send that row back to the owner, mirroring EP combine.
            pld.tensor.put(
                routed_y,
                peer=peer,
                src=recv_x,
                dst_offsets=[1, 0],
                src_offsets=[1, 0],
                shape=[1, D],
                atomic=pld.AtomicType.None_,
            )
            pld.system.notify(combine_signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
            pld.system.wait(signal=combine_signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

            result = pl.load(routed_y, [0, 0], [ROWS, D])
            return pl.store(result, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            routes: pl.Tensor[[ROWS, D], pl.FP32],
            out: pl.Out[pl.Tensor[[ROWS, D], pl.FP32]],
            send_x: pl.InOut[pld.DistributedTensor[[ROWS, D], pl.FP32]],
            recv_x: pl.InOut[pld.DistributedTensor[[ROWS, D], pl.FP32]],
            routed_y: pl.InOut[pld.DistributedTensor[[ROWS, D], pl.FP32]],
            dispatch_signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            combine_signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[ROWS, D], pl.FP32]:
            return self.ep_step(routes, out, send_x, recv_x, routed_y, dispatch_signal, combine_signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            routes: pl.Tensor[[N_RANKS, ROWS, D], pl.FP32],
            outputs: pl.Out[pl.Tensor[[N_RANKS, ROWS, D], pl.FP32]],
        ) -> pl.Tensor[[N_RANKS, ROWS, D], pl.FP32]:
            row_bytes = ROWS * D * 4
            send_x_buf = pld.alloc_window_buffer(row_bytes)
            recv_x_buf = pld.alloc_window_buffer(row_bytes)
            routed_y_buf = pld.alloc_window_buffer(row_bytes)
            dispatch_signal_buf = pld.alloc_window_buffer(4)
            combine_signal_buf = pld.alloc_window_buffer(4)

            for r in pl.range(pld.world_size()):
                send_x = pld.window(send_x_buf, [ROWS, D], dtype=pl.FP32)
                recv_x = pld.window(recv_x_buf, [ROWS, D], dtype=pl.FP32)
                routed_y = pld.window(routed_y_buf, [ROWS, D], dtype=pl.FP32)
                dispatch_signal = pld.window(dispatch_signal_buf, [1, 1], dtype=pl.INT32)
                combine_signal = pld.window(combine_signal_buf, [1, 1], dtype=pl.INT32)
                self.chip_orch(
                    routes[r],
                    outputs[r],
                    send_x,
                    recv_x,
                    routed_y,
                    dispatch_signal,
                    combine_signal,
                    (r + 1) % pld.world_size(),
                    device=r,
                )
            return outputs

    return EPDispatchCombine


class TestL3EPDispatchCombine:
    """L3 distributed runtime: EP-style dispatch, local expert, and combine."""

    def test_dispatch_combine_roundtrip(self, test_config, device_ids):
        if len(device_ids) < N_RANKS:
            pytest.skip(f"ep dispatch/combine needs {N_RANKS} devices, got {device_ids}")

        program = _build_ep_dispatch_combine_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            ),
        )

        routes = torch.stack(
            [
                torch.stack(
                    [
                        torch.arange(D, dtype=torch.float32),
                        torch.arange(100.0, 100.0 + D, dtype=torch.float32),
                    ]
                ),
                torch.stack(
                    [
                        torch.arange(1000.0, 1000.0 + D, dtype=torch.float32),
                        torch.arange(2000.0, 2000.0 + D, dtype=torch.float32),
                    ]
                ),
            ]
        )
        outputs = torch.zeros((N_RANKS, ROWS, D), dtype=torch.float32)

        compiled(routes, outputs)

        expected = routes * 2.0
        torch.testing.assert_close(outputs, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])

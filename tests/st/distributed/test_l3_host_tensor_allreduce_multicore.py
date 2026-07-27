# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 ST for HOST AllReduce with multiple synchronized AIV blocks.

Each case checks the reduced FP32 output and, on device, waits for every
requested signal lane.  The lane waits prove that every requested block
started; output correctness additionally proves that active blocks processed
their block-strided chunks.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig


def _make_rank_inputs(n_ranks: int, size: int) -> torch.Tensor:
    rows = [
        torch.arange(r * 100.0, r * 100.0 + size, dtype=torch.float32).reshape(1, size)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _expected_allreduce(inputs: torch.Tensor) -> torch.Tensor:
    reduced = inputs.sum(dim=0)
    return torch.stack([reduced] * inputs.shape[0])


def _build_multicore_allreduce_program(
    n_ranks: int,
    size: int,
    core_num: int,
    signal_stride: int,
):
    nr = n_ranks
    sz = size
    cores = core_num
    stride = signal_stride
    stage_rows = 8 if size == 1 else 1
    stage_cols = 1 if size == 1 else ((size + 7) // 8) * 8
    signal_stage_cols = ((signal_stride + 7) // 8) * 8

    @pl.program
    class HostTensorAllReduceMulticore:
        @pl.function(type=pl.FunctionType.InCore)
        def publish_step(
            self,
            inp: pl.Tensor[[1, sz], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[1, sz], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, stride], pl.INT32]],
        ) -> pld.DistributedTensor[[1, sz], pl.FP32]:
            local = pl.load(
                inp,
                [0, 0],
                [stage_rows, stage_cols],
                valid_shape=[1, sz],
            )
            return pl.store(local, [0, 0], data)

        @pl.function(type=pl.FunctionType.Orchestration)
        def publish_orch(
            self,
            inp: pl.Tensor[[1, sz], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[1, sz], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, stride], pl.INT32]],
        ) -> pld.DistributedTensor[[1, sz], pl.FP32]:
            return self.publish_step(inp, data, signal)

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pld.DistributedTensor[[1, sz], pl.FP32],
            signal: pld.DistributedTensor[[nr, stride], pl.INT32],
            out: pl.Out[pl.Tensor[[1, sz], pl.FP32]],
            signal_out: pl.Out[pl.Tensor[[nr, stride], pl.INT32]],
        ) -> pl.Tensor[[1, sz], pl.FP32]:
            ctx = pld.get_comm_ctx(signal)
            my_rank = pld.rank(ctx)
            for peer in pl.range(nr):
                if peer != my_rank:
                    for lane in pl.range(cores):
                        # The builtin's start barrier publishes one signal per
                        # launched block, including blocks with no data chunk.
                        # TWAIT performs the cache invalidation required for a
                        # reliable device-side observation.
                        pld.system.wait(
                            signal=signal,
                            offsets=[peer, lane],
                            expected=1,
                            cmp=pld.WaitCmp.Ge,
                        )

            reduced = pl.load(
                data,
                [0, 0],
                [stage_rows, stage_cols],
                valid_shape=[1, sz],
            )
            out = pl.store(reduced, [0, 0], out)

            signal_values = pl.load(
                signal,
                [0, 0],
                [nr, signal_stage_cols],
                valid_shape=[nr, stride],
            )
            pl.store(signal_values, [0, 0], signal_out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            data: pld.DistributedTensor[[1, sz], pl.FP32],
            signal: pld.DistributedTensor[[nr, stride], pl.INT32],
            out: pl.Out[pl.Tensor[[1, sz], pl.FP32]],
            signal_out: pl.Out[pl.Tensor[[nr, stride], pl.INT32]],
        ) -> pl.Tensor[[1, sz], pl.FP32]:
            return self.consume_step(data, signal, out, signal_out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, sz], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, sz], pl.FP32]],
            signal_outputs: pl.Out[pl.Tensor[[nr, nr, stride], pl.INT32]],
        ) -> pl.Tensor[[nr, 1, sz], pl.FP32]:
            data_buf = pld.alloc_window_buffer(sz * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pld.world_size() * stride * pl.INT32.get_byte())

            for rank in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, sz], dtype=pl.FP32)
                signal = pld.window(signal_buf, [pld.world_size(), stride], dtype=pl.INT32)
                self.publish_orch(inputs[rank], data, signal, device=rank)

            data = pld.window(data_buf, [1, sz], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), stride], dtype=pl.INT32)
            data = pld.tensor.allreduce(
                data,
                signal,
                op=pld.ReduceOp.Sum,
                core_num=cores,
            )

            for rank in pl.range(pld.world_size()):
                self.consume_orch(
                    data,
                    signal,
                    outputs[rank],
                    signal_outputs[rank],
                    device=rank,
                )
            return outputs

    return HostTensorAllReduceMulticore


class TestL3HostTensorAllReduceMulticore:
    @pytest.mark.parametrize(
        ("n_ranks", "core_num", "size", "signal_stride"),
        [
            pytest.param(2, 2, 1, 2, id="p2-c2-idle-lane"),
            pytest.param(2, 4, 4127, 6, id="p2-c4-wide-stride"),
            pytest.param(4, 2, 1041, 2, id="p4-c2-multichunk"),
        ],
    )
    def test_multicore_output_and_signal_lanes(
        self,
        test_config,
        device_ids,
        n_ranks,
        core_num,
        size,
        signal_stride,
    ):
        if len(device_ids) < n_ranks:
            pytest.skip(f"multicore host allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_multicore_allreduce_program(n_ranks, size, core_num, signal_stride)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks, size)
        outputs = torch.zeros((n_ranks, 1, size), dtype=torch.float32)
        signal_outputs = torch.zeros((n_ranks, n_ranks, signal_stride), dtype=torch.int32)

        compiled(inputs, outputs, signal_outputs)

        expected_outputs = _expected_allreduce(inputs)
        assert torch.allclose(outputs, expected_outputs, rtol=1e-4, atol=1e-5), (
            f"multicore host allreduce P={n_ranks}, C={core_num}, size={size} mismatch: "
            f"max diff = {(outputs - expected_outputs).abs().max().item()}"
        )

        for rank in range(n_ranks):
            for peer in range(n_ranks):
                if peer == rank:
                    continue
                assert torch.all(signal_outputs[rank, peer, :core_num] >= 1), (
                    f"missing signal lane for P={n_ranks}, C={core_num}, size={size}, "
                    f"stride={signal_stride}, receiver={rank}, sender={peer}: "
                    f"got {signal_outputs[rank, peer].tolist()}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])

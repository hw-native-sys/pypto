# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 ST for HOST broadcast/reduce_scatter with multiple synchronized AIV blocks.

RFC #2521 plan 85/B6b: extends #2160's block-indexed signal-lane multicore
contract — proven by allreduce and
``test_l3_host_tensor_allreduce_multicore.py`` — to broadcast and
reduce_scatter.  Each case checks the collective's output and, like the
allreduce test, infers each active block's lane completed its epilogue from
output correctness plus signal-lane self-clearing (mirrored here with the
same ``TWAIT(Eq 0)`` device-side proof).
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig


def _build_multicore_broadcast_program(n_ranks: int, size: int, core_num: int, signal_stride: int, root: int):
    nr = n_ranks
    sz = size
    cores = core_num
    stride = signal_stride
    stage_rows = 8 if size == 1 else 1
    stage_cols = 1 if size == 1 else ((size + 7) // 8) * 8
    signal_stage_cols = ((signal_stride + 7) // 8) * 8

    @pl.jit.incore
    def publish_step(
        inp: pl.Tensor[[1, sz], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, sz], pl.FP32]],
    ) -> pld.DistributedTensor[[1, sz], pl.FP32]:
        local = pl.load(inp, [0, 0], [stage_rows, stage_cols], valid_shape=[1, sz])
        return pl.store(local, [0, 0], data)

    @pl.jit
    def publish_orch(
        inp: pl.Tensor[[1, sz], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, sz], pl.FP32]],
    ) -> pld.DistributedTensor[[1, sz], pl.FP32]:
        return publish_step(inp, data)

    @pl.jit.incore
    def consume_step(
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
                    pld.system.wait(
                        signal=signal,
                        offsets=[peer, lane],
                        expected=0,
                        cmp=pld.WaitCmp.Eq,
                    )
        reduced = pl.load(data, [0, 0], [stage_rows, stage_cols], valid_shape=[1, sz])
        out = pl.store(reduced, [0, 0], out)
        signal_values = pl.load(signal, [0, 0], [nr, signal_stage_cols], valid_shape=[nr, stride])
        pl.store(signal_values, [0, 0], signal_out)
        return out

    @pl.jit
    def consume_orch(
        data: pld.DistributedTensor[[1, sz], pl.FP32],
        signal: pld.DistributedTensor[[nr, stride], pl.INT32],
        out: pl.Out[pl.Tensor[[1, sz], pl.FP32]],
        signal_out: pl.Out[pl.Tensor[[nr, stride], pl.INT32]],
    ) -> pl.Tensor[[1, sz], pl.FP32]:
        return consume_step(data, signal, out, signal_out)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[nr, 1, sz], pl.FP32],
        outputs: pl.Out[pl.Tensor[[nr, 1, sz], pl.FP32]],
        signal_outputs: pl.Out[pl.Tensor[[nr, nr, stride], pl.INT32]],
    ) -> pl.Tensor[[nr, 1, sz], pl.FP32]:
        data_buf = pld.alloc_window_buffer(sz * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(pld.world_size() * stride * pl.INT32.get_byte())

        for rank in pl.range(pld.world_size()):
            data = pld.window(data_buf, [1, sz], dtype=pl.FP32)
            publish_orch(inputs[rank], data, device=rank)

        data = pld.window(data_buf, [1, sz], dtype=pl.FP32)
        signal = pld.window(signal_buf, [pld.world_size(), stride], dtype=pl.INT32)
        data = pld.tensor.broadcast(data, signal, root=root, core_num=cores)

        for rank in pl.range(pld.world_size()):
            consume_orch(data, signal, outputs[rank], signal_outputs[rank], device=rank)
        return outputs

    return host_orch


def _build_multicore_reduce_scatter_program(n_ranks: int, size: int, core_num: int, signal_stride: int):
    nr = n_ranks
    sz = size
    cores = core_num
    stride = signal_stride
    signal_stage_cols = ((signal_stride + 7) // 8) * 8
    row_stage_cols = ((sz + 7) // 8) * 8
    # A size-one trailing dim is forced to DN; stage it as a padded column vector
    # (the same shape the broadcast helper uses) instead of a 1x8 row vector.
    stage_rows = 8 if sz == 1 else nr
    stage_cols = 1 if sz == 1 else row_stage_cols
    consume_rows = 8 if sz == 1 else 1
    consume_cols = 1 if sz == 1 else row_stage_cols

    @pl.jit.incore
    def publish_step(
        inp: pl.Tensor[[nr, sz], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[nr, sz], pl.FP32]],
    ) -> pld.DistributedTensor[[nr, sz], pl.FP32]:
        local = pl.load(inp, [0, 0], [stage_rows, stage_cols], valid_shape=[nr, sz])
        return pl.store(local, [0, 0], data)

    @pl.jit
    def publish_orch(
        inp: pl.Tensor[[nr, sz], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[nr, sz], pl.FP32]],
    ) -> pld.DistributedTensor[[nr, sz], pl.FP32]:
        return publish_step(inp, data)

    @pl.jit.incore
    def consume_step(
        data: pld.DistributedTensor[[nr, sz], pl.FP32],
        signal: pld.DistributedTensor[[nr, stride], pl.INT32],
        out: pl.Out[pl.Tensor[[1, sz], pl.FP32]],
        signal_out: pl.Out[pl.Tensor[[nr, stride], pl.INT32]],
    ) -> pl.Tensor[[1, sz], pl.FP32]:
        ctx = pld.get_comm_ctx(signal)
        my_rank = pld.rank(ctx)
        for peer in pl.range(nr):
            if peer != my_rank:
                for lane in pl.range(cores):
                    pld.system.wait(
                        signal=signal,
                        offsets=[peer, lane],
                        expected=0,
                        cmp=pld.WaitCmp.Eq,
                    )
        reduced = pl.load(data, [my_rank, 0], [consume_rows, consume_cols], valid_shape=[1, sz])
        out = pl.store(reduced, [0, 0], out)
        signal_values = pl.load(signal, [0, 0], [nr, signal_stage_cols], valid_shape=[nr, stride])
        pl.store(signal_values, [0, 0], signal_out)
        return out

    @pl.jit
    def consume_orch(
        data: pld.DistributedTensor[[nr, sz], pl.FP32],
        signal: pld.DistributedTensor[[nr, stride], pl.INT32],
        out: pl.Out[pl.Tensor[[1, sz], pl.FP32]],
        signal_out: pl.Out[pl.Tensor[[nr, stride], pl.INT32]],
    ) -> pl.Tensor[[1, sz], pl.FP32]:
        return consume_step(data, signal, out, signal_out)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[nr, nr, sz], pl.FP32],
        outputs: pl.Out[pl.Tensor[[nr, 1, sz], pl.FP32]],
        signal_outputs: pl.Out[pl.Tensor[[nr, nr, stride], pl.INT32]],
    ) -> pl.Tensor[[nr, 1, sz], pl.FP32]:
        target_buf = pld.alloc_window_buffer(nr * sz * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(pld.world_size() * stride * pl.INT32.get_byte())

        for rank in pl.range(pld.world_size()):
            target = pld.window(target_buf, [nr, sz], dtype=pl.FP32)
            publish_orch(inputs[rank], target, device=rank)

        target = pld.window(target_buf, [nr, sz], dtype=pl.FP32)
        signal = pld.window(signal_buf, [pld.world_size(), stride], dtype=pl.INT32)
        target = pld.tensor.reduce_scatter(target, signal, op=pld.ReduceOp.Sum, core_num=cores)

        for rank in pl.range(pld.world_size()):
            consume_orch(target, signal, outputs[rank], signal_outputs[rank], device=rank)
        return outputs

    return host_orch


class TestL3HostTensorBroadcastMulticore:
    @pytest.mark.parametrize(
        ("n_ranks", "core_num", "size", "signal_stride", "root"),
        [
            pytest.param(2, 2, 1, 2, 0, id="p2-c2-idle-lane"),
            pytest.param(2, 4, 4127, 6, 1, id="p2-c4-wide-stride"),
            pytest.param(4, 2, 1041, 2, 2, id="p4-c2-multichunk"),
        ],
    )
    def test_multicore_output_and_signal_lanes(
        self, test_config, device_ids, n_ranks, core_num, size, signal_stride, root
    ):
        if len(device_ids) < n_ranks:
            pytest.skip(f"multicore host broadcast P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        inputs = torch.stack(
            [
                torch.arange(r * 100.0, r * 100.0 + size, dtype=torch.float32).reshape(1, size)
                for r in range(n_ranks)
            ]
        )
        outputs = torch.zeros((n_ranks, 1, size), dtype=torch.float32)
        signal_outputs = torch.zeros((n_ranks, n_ranks, signal_stride), dtype=torch.int32)

        host_orch = _build_multicore_broadcast_program(n_ranks, size, core_num, signal_stride, root)
        compiled = host_orch.compile(
            inputs,
            outputs,
            signal_outputs,
            config=RunConfig(
                platform=test_config.platform,
                distributed_config=DistributedConfig(device_ids=device_ids[:n_ranks], num_sub_workers=0),
            ),
        )
        compiled(inputs, outputs, signal_outputs, config=RunConfig(platform=test_config.platform))

        expected = torch.stack([inputs[root, 0]] * n_ranks).unsqueeze(1)
        assert torch.allclose(outputs, expected, rtol=1e-4, atol=1e-5), (
            f"multicore host broadcast P={n_ranks}, C={core_num}, size={size}, root={root} mismatch: "
            f"max diff = {(outputs - expected).abs().max().item()}"
        )
        for rank in range(n_ranks):
            for peer in range(n_ranks):
                if peer == rank:
                    continue
                assert torch.all(signal_outputs[rank, peer, :core_num] == 0), (
                    f"signal lane not self-cleared for P={n_ranks}, C={core_num}, size={size}, "
                    f"stride={signal_stride}, receiver={rank}, sender={peer}: "
                    f"got {signal_outputs[rank, peer].tolist()}"
                )


class TestL3HostTensorReduceScatterMulticore:
    @pytest.mark.parametrize(
        ("n_ranks", "core_num", "size", "signal_stride"),
        [
            pytest.param(2, 2, 1, 2, id="p2-c2-idle-lane"),
            pytest.param(2, 4, 4127, 6, id="p2-c4-wide-stride"),
            pytest.param(4, 2, 1041, 2, id="p4-c2-multichunk"),
        ],
    )
    def test_multicore_output_and_signal_lanes(
        self, test_config, device_ids, n_ranks, core_num, size, signal_stride
    ):
        if len(device_ids) < n_ranks:
            pytest.skip(
                f"multicore host reduce_scatter P={n_ranks} needs {n_ranks} devices, got {device_ids}"
            )

        # inputs[s, r, :] = what source rank s stages for chunk r. Every rank
        # stages the full [NR, SIZE] matrix (reduce_scatter semantics), so the
        # expected result for chunk r is the sum over s of inputs[s, r, :].
        inputs = torch.stack(
            [
                torch.stack(
                    [
                        torch.arange(r * 10.0 + s * 1000.0, r * 10.0 + s * 1000.0 + size, dtype=torch.float32)
                        for r in range(n_ranks)
                    ]
                )
                for s in range(n_ranks)
            ]
        )
        outputs = torch.zeros((n_ranks, 1, size), dtype=torch.float32)
        signal_outputs = torch.zeros((n_ranks, n_ranks, signal_stride), dtype=torch.int32)

        host_orch = _build_multicore_reduce_scatter_program(n_ranks, size, core_num, signal_stride)
        compiled = host_orch.compile(
            inputs,
            outputs,
            signal_outputs,
            config=RunConfig(
                platform=test_config.platform,
                distributed_config=DistributedConfig(device_ids=device_ids[:n_ranks], num_sub_workers=0),
            ),
        )
        compiled(inputs, outputs, signal_outputs, config=RunConfig(platform=test_config.platform))

        expected = inputs.sum(dim=0).unsqueeze(1)  # [NR, 1, SIZE], row r = sum_s inputs[s, r, :]
        assert torch.allclose(outputs, expected, rtol=1e-4, atol=1e-5), (
            f"multicore host reduce_scatter P={n_ranks}, C={core_num}, size={size} mismatch: "
            f"max diff = {(outputs - expected).abs().max().item()}"
        )
        for rank in range(n_ranks):
            for peer in range(n_ranks):
                if peer == rank:
                    continue
                assert torch.all(signal_outputs[rank, peer, :core_num] == 0), (
                    f"signal lane not self-cleared for P={n_ranks}, C={core_num}, size={size}, "
                    f"stride={signal_stride}, receiver={rank}, sender={peer}: "
                    f"got {signal_outputs[rank, peer].tolist()}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])

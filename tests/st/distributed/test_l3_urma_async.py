# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: asynchronous URMA put/get — task-dependency completion.

V1 completion is owned by the enclosing task: returning from ``put_async`` /
``get_async`` does not mean the data has moved. A later task consumes the
destination as ``In``, and that task dependency is the ordering point.

One exchange covers both directions. Each rank fills its own row in ``src`` and
``data``, pushes the ``data`` row into every peer (PUT), overwrites its local
``data`` row only after the transfer task has retired, then pulls every peer's
``src`` row into ``recv`` (GET) and copies both windows to host-visible output.
The overwrite makes an early release observable: if the PUT task were released
at its in-core finish, the peer would receive the overwritten bytes. The local
row therefore carries the overwrite value while the peer rows carry the first
value; the golden reflects both.

**Hardware.** The generated helper includes ``backend/urma/urma_completion_kernel.h``,
which exists only for A5 (Ascend 950) silicon; there is no simulator backend.
Run on A5 TMR with two or four cards:

    pytest tests/st/distributed/test_l3_urma_async.py -v \
        --platform=a5 --device=0,1

Compiled with CANN 9.1.0; record the runtime and pypto commits with the result.
A skip on a non-A5 host is not evidence of correctness.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

SIZE = 4096
REPEATS = 2


def _rank_inputs(n_ranks: int, *, round_index: int = 0) -> torch.Tensor:
    """Rank r fills only its own row: input[r, r, j] = base + r*1000 + j."""
    inputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    base = round_index * 10000
    for r in range(n_ranks):
        inputs[r, r, :] = torch.arange(SIZE, dtype=torch.float32) + base + r * 1000
    return inputs


def _expected_exchange(n_ranks: int, *, round_index: int = 0) -> torch.Tensor:
    """Every rank ends with every row: out[rank, d, j] = base + d*1000 + j."""
    base = round_index * 10000
    d = torch.arange(n_ranks, dtype=torch.float32).view(1, -1, 1)
    j = torch.arange(SIZE, dtype=torch.float32).view(1, 1, -1)
    return (d * 1000 + j + base).expand(n_ranks, n_ranks, SIZE).contiguous()


def _expected_put(n_ranks: int, *, round_index: int = 0) -> torch.Tensor:
    """Peer rows keep v1; the local row was overwritten with v2 after completion."""
    expected = _expected_exchange(n_ranks, round_index=round_index)
    overwritten = _expected_exchange(n_ranks, round_index=round_index + 1)
    for r in range(n_ranks):
        expected[r, r, :] = overwritten[r, r, :]
    return expected


def _build_urma_exchange(n_ranks: int):
    """Build the host orchestrator; the kernels are shared by every case."""

    @pl.jit.incore
    def fill_src(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        src: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(src)
        my_rank = pld.rank(ctx)
        row = pl.load(inp, [my_rank, 0], [1, SIZE])
        return pl.store(row, [my_rank, 0], src)

    @pl.jit.incore
    def stage_put(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        row = pl.load(inp, [my_rank, 0], [1, SIZE])
        data = pl.store(row, [my_rank, 0], data)
        for peer in pl.range(nranks):
            if peer != my_rank:
                pld.tensor.put_async(
                    data,
                    peer=peer,
                    src=data,
                    dst_offsets=[my_rank, 0],
                    src_offsets=[my_rank, 0],
                    shape=[1, SIZE],
                    engine=pld.CommEngine.URMA,
                )
        return data

    @pl.jit.incore
    def overwrite_after_completion(
        inp2: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        row = pl.load(inp2, [my_rank, 0], [1, SIZE])
        return pl.store(row, [my_rank, 0], data)

    @pl.jit.incore
    def announce(
        src: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        data: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pld.DistributedTensor[[n_ranks, 1], pl.INT32]:
        # `src` is unused: passing the filled window makes this task depend on
        # fill_src, so the notification cannot reach a peer before this rank
        # publishes its own source row.
        ctx = pld.get_comm_ctx(signal)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        for peer in pl.range(nranks):
            if peer != my_rank:
                pld.system.notify(
                    signal,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.Set,
                )
        return signal

    @pl.jit.incore
    def await_peers(
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pld.DistributedTensor[[n_ranks, 1], pl.INT32]:
        ctx = pld.get_comm_ctx(signal)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        for peer in pl.range(nranks):
            if peer != my_rank:
                pld.system.wait(
                    signal=signal,
                    offsets=[peer, 0],
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )
        return signal

    @pl.jit.incore
    def fetch(
        src: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        signal: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        recv: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(recv)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        own = pl.load(src, [my_rank, 0], [1, SIZE])
        recv = pl.store(own, [my_rank, 0], recv)
        for peer in pl.range(nranks):
            if peer != my_rank:
                pld.tensor.get_async(
                    recv,
                    peer=peer,
                    src=src,
                    dst_offsets=[peer, 0],
                    src_offsets=[peer, 0],
                    shape=[1, SIZE],
                    engine=pld.CommEngine.URMA,
                )
        return recv

    @pl.jit.incore
    def readback_put(
        data: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        signal: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        for row in pl.range(pld.nranks(ctx)):
            chunk = pl.load(data, [row, 0], [1, SIZE])
            out = pl.store(chunk, [row, 0], out)
        return out

    @pl.jit.incore
    def readback_get(
        recv: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        signal: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(recv)
        for row in pl.range(pld.nranks(ctx)):
            chunk = pl.load(recv, [row, 0], [1, SIZE])
            # Mixed compute after completion: the read is only valid once the
            # GET task has retired.
            value = pl.add(chunk, 1.0)
            out = pl.store(value, [row, 0], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        inp2: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        out_put: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        out_get: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
        src: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
        recv: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        filled = fill_src(inp, src)
        staged = stage_put(inp, data)
        # Ordered after stage_put's completion by the InOut -> In dependency.
        overwritten = overwrite_after_completion(inp2, staged)
        announced = announce(filled, overwritten, signal)
        waited = await_peers(announced)
        fetched = fetch(filled, waited, recv)
        readback_put(overwritten, waited, out_put)
        return readback_get(fetched, waited, out_get)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32],
        inputs2: pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32],
        outputs_put: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
        outputs_get: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        src_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        recv_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(n_ranks * pl.INT32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [n_ranks, SIZE], dtype=pl.FP32)
            src = pld.window(src_buf, [n_ranks, SIZE], dtype=pl.FP32)
            recv = pld.window(recv_buf, [n_ranks, SIZE], dtype=pl.FP32)
            sig = pld.window(signal_buf, [n_ranks, 1], dtype=pl.INT32)
            chip_orch(inputs[r], inputs2[r], outputs_put[r], outputs_get[r], data, src, recv, sig, device=r)
        return outputs_put

    return host_orch


def _run(
    test_config: RunConfig, device_ids: list[int], n_ranks: int, round_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = _rank_inputs(n_ranks, round_index=round_index)
    inputs2 = _rank_inputs(n_ranks, round_index=round_index + 1)
    outputs_put = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    outputs_get = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    distributed_config = DistributedConfig(device_ids=device_ids[:n_ranks], num_sub_workers=0)
    host_orch = _build_urma_exchange(n_ranks)
    compiled = host_orch.compile(
        inputs,
        inputs2,
        outputs_put,
        outputs_get,
        config=RunConfig(platform=test_config.platform, distributed_config=distributed_config),
    )
    compiled(
        inputs,
        inputs2,
        outputs_put,
        outputs_get,
        config=RunConfig(platform=test_config.platform),
    )
    return outputs_put, outputs_get


@pytest.mark.platforms("a5", reason="URMA async transfers require the A5 backend")
class TestL3UrmaAsync:
    """Asynchronous URMA PUT/GET and task-completion ordering on A5 silicon.

    The cases execute on A5 TMR hardware, and are skipped by the platform
    filter on a2a3 hosts. A skip is not evidence of correctness.
    """

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_exchange_values(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async exchange P={n_ranks} needs {n_ranks} devices")

        outputs_put, outputs_get = _run(test_config, device_ids, n_ranks, round_index=0)

        expected_put = _expected_put(n_ranks)
        expected_get = _expected_exchange(n_ranks) + 1.0
        assert torch.allclose(outputs_put, expected_put), (
            f"PUT P={n_ranks} mismatch: max diff = {(outputs_put - expected_put).abs().max().item()}"
        )
        assert torch.allclose(outputs_get, expected_get), (
            f"GET P={n_ranks} mismatch: max diff = {(outputs_get - expected_get).abs().max().item()}"
        )

    def test_exchange_repeated(self, test_config, device_ids):
        """Repeated calls must each observe their own payload."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async repeated exchange needs {n_ranks} devices")

        for round_index in range(REPEATS):
            outputs_put, outputs_get = _run(test_config, device_ids, n_ranks, round_index)
            expected_put = _expected_put(n_ranks, round_index=round_index)
            expected_get = _expected_exchange(n_ranks, round_index=round_index) + 1.0
            assert torch.allclose(outputs_put, expected_put), (
                f"repeated PUT round {round_index} mismatch: "
                f"max diff = {(outputs_put - expected_put).abs().max().item()}"
            )
            assert torch.allclose(outputs_get, expected_get), (
                f"repeated GET round {round_index} mismatch: "
                f"max diff = {(outputs_get - expected_get).abs().max().item()}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])

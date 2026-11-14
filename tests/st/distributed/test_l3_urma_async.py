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

The reduced cases pin the transfer stages: ``test_put_only`` runs one PUT per
peer with a minimal notification handshake, and
``test_put_get_no_handshake`` runs PUT and GET together with no notification.
Both keep the program small enough to attribute a device fault to a single
stage.

**Hardware.** The generated helper includes ``backend/urma/urma_completion_kernel.h``,
which exists only for A5 (Ascend 950) silicon; there is no simulator backend.
Run on A5 TMR with two or four cards:

    pytest tests/st/distributed/test_l3_urma_async.py -v \
        --platform=a5 --device=0,1

Compiled with CANN 9.1.0; record the runtime and pypto commits with the result.
This hardware acceptance suite skips under ``--codegen-only``; compiler-only
coverage lives in ``tests/ut/codegen/distributed/``. A skip on a non-A5 host or
in codegen-only mode is not evidence of runtime correctness.
"""

import json
import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig
from pypto.runtime._artifact_runtime import runtime_output_directory

SIZE = 4096
REPEATS = 3


@pytest.fixture(autouse=True)
def _require_device_execution(test_config: RunConfig) -> None:
    """Skip hardware acceptance before compilation or warmup in codegen-only mode."""
    if test_config.codegen_only:
        pytest.skip("URMA async hardware acceptance requires device execution; --codegen-only is unsupported")


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


def _build_put_only(n_ranks: int):
    """Build the canonical FP32 PUT-only exchange."""
    return _build_put_only_typed(n_ranks, pl.FP32)


def _build_put_get_no_handshake(n_ranks: int):
    """PUT and GET in one program without a cross-rank notification.

    Both readbacks are ordered after this rank's own PUT, but nothing orders
    them after the peers' fills or incoming PUTs. Only each rank's local row
    is deterministic in either result; the handshake cases check peer rows.
    """

    @pl.jit.incore
    def publish(
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
    def fetch(
        data: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        recv: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(recv)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        own = pl.load(data, [my_rank, 0], [1, SIZE])
        recv = pl.store(own, [my_rank, 0], recv)
        for peer in pl.range(nranks):
            if peer != my_rank:
                pld.tensor.get_async(
                    recv,
                    peer=peer,
                    src=data,
                    dst_offsets=[peer, 0],
                    src_offsets=[peer, 0],
                    shape=[1, SIZE],
                    engine=pld.CommEngine.URMA,
                )
        return recv

    @pl.jit.incore
    def readback(
        window: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(window)
        for row in pl.range(pld.nranks(ctx)):
            chunk = pl.load(window, [row, 0], [1, SIZE])
            out = pl.store(chunk, [row, 0], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        out_put: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        out_get: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
        recv: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        published = publish(inp, data)
        fetched = fetch(published, recv)
        readback(published, out_put)
        return readback(fetched, out_get)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32],
        outputs_put: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
        outputs_get: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        recv_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [n_ranks, SIZE], dtype=pl.FP32)
            recv = pld.window(recv_buf, [n_ranks, SIZE], dtype=pl.FP32)
            chip_orch(inputs[r], outputs_put[r], outputs_get[r], data, recv, device=r)
        return outputs_put

    return host_orch


def _run_single_output(test_config: RunConfig, device_ids: list[int], n_ranks: int, builder):
    inputs = _rank_inputs(n_ranks)
    outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    host_orch = builder(n_ranks)
    compiled = host_orch.compile(
        inputs,
        outputs,
        config=RunConfig(
            platform=test_config.platform,
            distributed_config=DistributedConfig(device_ids=device_ids[:n_ranks], num_sub_workers=0),
        ),
    )
    compiled(inputs, outputs, config=RunConfig(platform=test_config.platform))
    return outputs


def _run_two_outputs(test_config: RunConfig, device_ids: list[int], n_ranks: int, builder):
    inputs = _rank_inputs(n_ranks)
    outputs_put = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    outputs_get = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    host_orch = builder(n_ranks)
    compiled = host_orch.compile(
        inputs,
        outputs_put,
        outputs_get,
        config=RunConfig(
            platform=test_config.platform,
            distributed_config=DistributedConfig(device_ids=device_ids[:n_ranks], num_sub_workers=0),
        ),
    )
    compiled(
        inputs,
        outputs_put,
        outputs_get,
        config=RunConfig(platform=test_config.platform),
    )
    return outputs_put, outputs_get


def _compile(test_config: RunConfig, device_ids: list[int], n_ranks: int):
    """Compile the exchange once; execution reuses the returned program."""
    inputs = _rank_inputs(n_ranks)
    inputs2 = _rank_inputs(n_ranks, round_index=1)
    outputs_put = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    outputs_get = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    distributed_config = DistributedConfig(device_ids=device_ids[:n_ranks], num_sub_workers=0)
    host_orch = _build_urma_exchange(n_ranks)
    return host_orch.compile(
        inputs,
        inputs2,
        outputs_put,
        outputs_get,
        config=RunConfig(platform=test_config.platform, distributed_config=distributed_config),
    )


def _execute(test_config: RunConfig, compiled, n_ranks: int, round_index: int):
    inputs = _rank_inputs(n_ranks, round_index=round_index)
    inputs2 = _rank_inputs(n_ranks, round_index=round_index + 1)
    outputs_put = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    outputs_get = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    compiled(
        inputs,
        inputs2,
        outputs_put,
        outputs_get,
        config=RunConfig(platform=test_config.platform),
    )
    return outputs_put, outputs_get


def _run(test_config: RunConfig, device_ids: list[int], n_ranks: int, round_index: int):
    compiled = _compile(test_config, device_ids, n_ranks)
    return _execute(test_config, compiled, n_ranks, round_index)


def _build_put_only_typed(n_ranks: int, dtype):
    """PUT-only exchange in an arbitrary element type.

    Mirrors the FP32 ladder with a single transfer set, so a second dtype can
    be checked without a second full exchange.
    """

    @pl.jit.incore
    def publish(
        inp: pl.Tensor[[n_ranks, SIZE], dtype],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], dtype]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], dtype]:
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
    def announce(
        data: pld.DistributedTensor[[n_ranks, SIZE], dtype],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pld.DistributedTensor[[n_ranks, 1], pl.INT32]:
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
    def readback(
        data: pld.DistributedTensor[[n_ranks, SIZE], dtype],
        signal: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], dtype]],
    ) -> pl.Tensor[[n_ranks, SIZE], dtype]:
        ctx = pld.get_comm_ctx(data)
        for row in pl.range(pld.nranks(ctx)):
            chunk = pl.load(data, [row, 0], [1, SIZE])
            out = pl.store(chunk, [row, 0], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[n_ranks, SIZE], dtype],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], dtype]],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], dtype]],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pl.Tensor[[n_ranks, SIZE], dtype]:
        published = publish(inp, data)
        announced = announce(published, signal)
        waited = await_peers(announced)
        return readback(published, waited, out)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[n_ranks, n_ranks, SIZE], dtype],
        outputs: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], dtype]],
    ) -> pl.Tensor[[n_ranks, n_ranks, SIZE], dtype]:
        data_buf = pld.alloc_window_buffer(n_ranks * SIZE * dtype.get_byte())
        signal_buf = pld.alloc_window_buffer(n_ranks * pl.INT32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [n_ranks, SIZE], dtype=dtype)
            sig = pld.window(signal_buf, [n_ranks, 1], dtype=pl.INT32)
            chip_orch(inputs[r], outputs[r], data, sig, device=r)
        return outputs

    return host_orch


def _build_bad_peer(n_ranks: int):
    """A PUT whose peer is the domain's own rank count, which is out of range."""

    @pl.jit.incore
    def publish(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        row = pl.load(inp, [my_rank, 0], [1, SIZE])
        data = pl.store(row, [my_rank, 0], data)
        pld.tensor.put_async(
            data,
            peer=pld.nranks(ctx),
            src=data,
            dst_offsets=[my_rank, 0],
            src_offsets=[my_rank, 0],
            shape=[1, SIZE],
            engine=pld.CommEngine.URMA,
        )
        return data

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        return publish(inp, data)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [n_ranks, SIZE], dtype=pl.FP32)
            chip_orch(inputs[r], outputs[r], data, device=r)
        return outputs

    return host_orch


def _build_undersized_window(n_ranks: int):
    """A window view whose backing allocation is four bytes short."""

    @pl.jit.incore
    def publish(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        return data

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        return publish(inp, data)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte() - 4)
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [n_ranks, SIZE], dtype=pl.FP32)
            chip_orch(inputs[r], outputs[r], data, device=r)
        return outputs

    return host_orch


def _build_out_of_range(n_ranks: int):
    """A PUT whose runtime column offset lands past the tensor's own extent."""

    @pl.jit.incore
    def publish(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        row = pl.load(inp, [my_rank, 0], [1, SIZE])
        data = pl.store(row, [my_rank, 0], data)
        # nranks * SIZE is one full row past the last valid column.
        col = nranks * SIZE
        pld.tensor.put_async(
            data,
            peer=1 - my_rank,
            src=data,
            dst_offsets=[my_rank, 0],
            src_offsets=[0, col],
            shape=[1, SIZE],
            engine=pld.CommEngine.URMA,
        )
        return data

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        return publish(inp, data)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [n_ranks, SIZE], dtype=pl.FP32)
            chip_orch(inputs[r], outputs[r], data, device=r)
        return outputs

    return host_orch


def _invalid_chip_orch(n_ranks: int, publish):
    """Wrap a single publish kernel as the chip orchestrator."""

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        return publish(inp, data)

    return chip_orch


def _invalid_host_orch(n_ranks: int, chip_orch):
    """Host orchestrator for a one-window rejection program."""

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [n_ranks, SIZE], dtype=pl.FP32)
            chip_orch(inputs[r], outputs[r], data, device=r)
        return outputs

    return host_orch


def _build_self_peer(n_ranks: int):
    """A PUT whose peer is the issuing rank."""

    @pl.jit.incore
    def publish(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        row = pl.load(inp, [my_rank, 0], [1, SIZE])
        data = pl.store(row, [my_rank, 0], data)
        pld.tensor.put_async(
            data,
            peer=my_rank,
            src=data,
            dst_offsets=[my_rank, 0],
            src_offsets=[my_rank, 0],
            shape=[1, SIZE],
            engine=pld.CommEngine.URMA,
        )
        return data

    return _invalid_host_orch(n_ranks, _invalid_chip_orch(n_ranks, publish))


def _build_zero_length(n_ranks: int):
    """A PUT whose runtime transfer extent is zero."""

    @pl.jit.incore
    def publish(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        row = pl.load(inp, [my_rank, 0], [1, SIZE])
        data = pl.store(row, [my_rank, 0], data)
        zero = my_rank - my_rank
        pld.tensor.put_async(
            data,
            peer=1 - my_rank,
            src=data,
            dst_offsets=[my_rank, 0],
            src_offsets=[my_rank, 0],
            shape=[1, zero],
            engine=pld.CommEngine.URMA,
        )
        return data

    return _invalid_host_orch(n_ranks, _invalid_chip_orch(n_ranks, publish))


def _build_negative_offset(n_ranks: int):
    """A PUT with a negative runtime column offset on the odd rank."""

    @pl.jit.incore
    def publish(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        row = pl.load(inp, [my_rank, 0], [1, SIZE])
        data = pl.store(row, [my_rank, 0], data)
        col = 0 - my_rank
        pld.tensor.put_async(
            data,
            peer=1 - my_rank,
            src=data,
            dst_offsets=[my_rank, 0],
            src_offsets=[my_rank, col],
            shape=[1, HALF],
            engine=pld.CommEngine.URMA,
        )
        return data

    return _invalid_host_orch(n_ranks, _invalid_chip_orch(n_ranks, publish))


def _typed_inputs(n_ranks: int, dtype, step: int):
    inputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=dtype)
    for r in range(n_ranks):
        inputs[r, r, :] = torch.arange(SIZE, dtype=dtype) + r * step
    return inputs


def _typed_expected(n_ranks: int, dtype, step: int):
    d = torch.arange(n_ranks, dtype=dtype).view(1, -1, 1)
    j = torch.arange(SIZE, dtype=dtype).view(1, 1, -1)
    return (d * step + j).expand(n_ranks, n_ranks, SIZE).contiguous()


def _run_put_only_typed(test_config: RunConfig, device_ids: list[int], n_ranks: int, dtype, torch_dtype):
    inputs = _typed_inputs(n_ranks, torch_dtype, step=100)
    outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch_dtype)
    host_orch = _build_put_only_typed(n_ranks, dtype)
    compiled = host_orch.compile(
        inputs,
        outputs,
        config=RunConfig(
            platform=test_config.platform,
            distributed_config=DistributedConfig(device_ids=device_ids[:n_ranks], num_sub_workers=0),
        ),
    )
    compiled(inputs, outputs, config=RunConfig(platform=test_config.platform))
    return outputs


def _compile_invalid(test_config: RunConfig, device_ids: list[int], n_ranks: int, builder):
    """Compile and assemble an invalid program before asserting its rejection."""
    inputs = _rank_inputs(n_ranks)
    outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    host_orch = builder(n_ranks)
    compiled = host_orch.warmup(
        inputs,
        outputs,
        config=RunConfig(
            platform=test_config.platform,
            distributed_config=DistributedConfig(device_ids=device_ids[:n_ranks], num_sub_workers=0),
        ),
    )

    return compiled, inputs, outputs


def _execute_invalid(test_config: RunConfig, compiled, inputs, outputs):
    """Execute a previously compiled invalid program."""
    compiled(inputs, outputs, config=RunConfig(platform=test_config.platform))


def _assert_invalid(
    test_config: RunConfig,
    device_ids: list[int],
    n_ranks: int,
    builder,
    capfd,
    expected_error: str = "ASYNC_COMPLETION_INVALID",
):
    """Require a runtime rejection after device warmup.

    A5 reports the scheduler classification in the device log, but the
    distributed Python runner can surface the subsequent lane-poisoned or
    cleanup exception instead.  Match the stable host-level contract and
    verify the scheduler-specific code in captured device diagnostics.
    """
    compiled, inputs, outputs = _compile_invalid(test_config, device_ids, n_ranks, builder)
    with pytest.raises(RuntimeError):
        _execute_invalid(test_config, compiled, inputs, outputs)
    assert expected_error in capfd.readouterr().err


HALF = SIZE // 2


def _build_get_only(n_ranks: int):
    """Each rank fills its own row, then pulls every peer's row into ``recv``."""

    @pl.jit.incore
    def fill(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        src: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(src)
        my_rank = pld.rank(ctx)
        row = pl.load(inp, [my_rank, 0], [1, SIZE])
        return pl.store(row, [my_rank, 0], src)

    @pl.jit.incore
    def announce(
        src: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pld.DistributedTensor[[n_ranks, 1], pl.INT32]:
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
    def readback(
        recv: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(recv)
        for row in pl.range(pld.nranks(ctx)):
            chunk = pl.load(recv, [row, 0], [1, SIZE])
            out = pl.store(chunk, [row, 0], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        src: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
        recv: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        filled = fill(inp, src)
        announced = announce(filled, signal)
        waited = await_peers(announced)
        fetched = fetch(filled, waited, recv)
        return readback(fetched, out)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]:
        src_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        recv_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(n_ranks * pl.INT32.get_byte())
        for r in pl.range(pld.world_size()):
            src = pld.window(src_buf, [n_ranks, SIZE], dtype=pl.FP32)
            recv = pld.window(recv_buf, [n_ranks, SIZE], dtype=pl.FP32)
            sig = pld.window(signal_buf, [n_ranks, 1], dtype=pl.INT32)
            chip_orch(inputs[r], outputs[r], src, recv, sig, device=r)
        return outputs

    return host_orch


def _build_dynamic_subregion(n_ranks: int):
    """PUT a half-row whose column offset is a runtime value.

    Each rank writes its own row's upper half at a column derived from its
    rank, so the offset is not a compile-time constant and the runtime
    per-dimension guard has to accept it.
    """

    @pl.jit.incore
    def publish(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        row = pl.load(inp, [my_rank, 0], [1, SIZE])
        data = pl.store(row, [my_rank, 0], data)
        col = my_rank
        for peer in pl.range(nranks):
            if peer != my_rank:
                pld.tensor.put_async(
                    data,
                    peer=peer,
                    src=data,
                    dst_offsets=[my_rank, col],
                    src_offsets=[my_rank, col],
                    shape=[1, HALF],
                    engine=pld.CommEngine.URMA,
                )
        return data

    @pl.jit.incore
    def announce(
        data: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pld.DistributedTensor[[n_ranks, 1], pl.INT32]:
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
    def readback(
        data: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        signal: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        for row in pl.range(pld.nranks(ctx)):
            chunk = pl.load(data, [row, 0], [1, SIZE])
            out = pl.store(chunk, [row, 0], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        published = publish(inp, data)
        announced = announce(published, signal)
        waited = await_peers(announced)
        return readback(published, waited, out)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(n_ranks * pl.INT32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [n_ranks, SIZE], dtype=pl.FP32)
            sig = pld.window(signal_buf, [n_ranks, 1], dtype=pl.INT32)
            chip_orch(inputs[r], outputs[r], data, sig, device=r)
        return outputs

    return host_orch


def _build_burst(n_ranks: int, repeats: int = 8):
    """Submit several transfers back to back from one task."""

    @pl.jit.incore
    def publish(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        row = pl.load(inp, [my_rank, 0], [1, SIZE])
        data = pl.store(row, [my_rank, 0], data)
        for _ in pl.range(repeats):
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
    def announce(
        data: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pld.DistributedTensor[[n_ranks, 1], pl.INT32]:
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
    def readback(
        data: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        signal: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        for row in pl.range(pld.nranks(ctx)):
            chunk = pl.load(data, [row, 0], [1, SIZE])
            out = pl.store(chunk, [row, 0], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        published = publish(inp, data)
        announced = announce(published, signal)
        waited = await_peers(announced)
        return readback(published, waited, out)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(n_ranks * pl.INT32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [n_ranks, SIZE], dtype=pl.FP32)
            sig = pld.window(signal_buf, [n_ranks, 1], dtype=pl.INT32)
            chip_orch(inputs[r], outputs[r], data, sig, device=r)
        return outputs

    return host_orch


@pytest.mark.platforms("a5", reason="URMA async transfers require the A5 backend")
class TestL3UrmaAsync:
    """Asynchronous URMA PUT/GET and task-completion ordering on A5 silicon.

    The cases execute on A5 TMR hardware, and are skipped by the platform
    filter on a2a3 hosts. A skip is not evidence of correctness.

    The ordering checks are value-based: an early release of the consumer
    would change the observed rows, but the assertions do not read scheduler
    state. A scheduler-side trace is the direct evidence and is not wired.
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

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_put_only(self, test_config, device_ids, n_ranks):
        """A single URMA PUT per peer, with no GET and a minimal handshake."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async PUT P={n_ranks} needs {n_ranks} devices")

        outputs = _run_single_output(test_config, device_ids, n_ranks, _build_put_only)
        expected = _expected_exchange(n_ranks)
        assert torch.allclose(outputs, expected), (
            f"PUT-only P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    def test_put_get_no_handshake(self, test_config, device_ids):
        """PUT and GET without a notification: only local rows are deterministic."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async PUT/GET needs {n_ranks} devices")

        outputs_put, outputs_get = _run_two_outputs(
            test_config, device_ids, n_ranks, _build_put_get_no_handshake
        )
        expected = _expected_exchange(n_ranks)
        # Local completion orders this rank's readbacks, not incoming PUTs or
        # peer fills. Full peer-row assertions require the handshake cases.
        for r in range(n_ranks):
            assert torch.allclose(outputs_put[r, r], expected[r, r]), (
                f"PUT local row mismatch on rank {r}: "
                f"max diff = {(outputs_put[r, r] - expected[r, r]).abs().max().item()}"
            )
            assert torch.allclose(outputs_get[r, r], expected[r, r]), (
                f"GET local row mismatch on rank {r}: "
                f"max diff = {(outputs_get[r, r] - expected[r, r]).abs().max().item()}"
            )

    def test_exchange_repeated(self, test_config, device_ids):
        """Repeated calls must each observe their own payload."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async repeated exchange needs {n_ranks} devices")

        compiled = _compile(test_config, device_ids, n_ranks)
        for round_index in range(REPEATS):
            outputs_put, outputs_get = _execute(test_config, compiled, n_ranks, round_index)
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

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_put_only_fp16(self, test_config, device_ids, n_ranks):
        """A second element type crosses the same PUT path.

        The values exercise FP16 rounding (some exceed 2048 in the wider
        ladder); the input and golden are built with the same dtype
        arithmetic, so this checks the transfer path bit-exactly rather than
        acting as an independent FP16 math oracle.
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async FP16 PUT P={n_ranks} needs {n_ranks} devices")

        outputs = _run_put_only_typed(test_config, device_ids, n_ranks, pl.FP16, torch.float16)
        expected = _typed_expected(n_ranks, torch.float16, step=100)
        assert torch.equal(outputs, expected), (
            f"FP16 PUT P={n_ranks} mismatch: max diff = "
            f"{(outputs.float() - expected.float()).abs().max().item()}"
        )

    def test_invalid_peer_fails_loudly(self, test_config, device_ids, capfd):
        """A runtime peer equal to the domain's rank count is rejected."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async invalid-peer case needs {n_ranks} devices")

        _assert_invalid(test_config, device_ids, n_ranks, _build_bad_peer, capfd)

    def test_out_of_range_region_fails_loudly(self, test_config, device_ids, capfd):
        """A runtime subregion past the tensor's own extent is rejected."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async out-of-range case needs {n_ranks} devices")

        _assert_invalid(test_config, device_ids, n_ranks, _build_out_of_range, capfd)

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_get_only(self, test_config, device_ids, n_ranks):
        """GET on its own: every rank pulls every peer's row into recv."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async GET-only P={n_ranks} needs {n_ranks} devices")

        outputs = _run_single_output(test_config, device_ids, n_ranks, _build_get_only)
        expected = _expected_exchange(n_ranks)
        assert torch.allclose(outputs, expected), (
            f"GET-only P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_dynamic_subregion(self, test_config, device_ids, n_ranks):
        """A half-row transferred at a runtime column offset.

        The offset is `rank`, so the runtime per-dimension guard sees a value
        that is not a compile-time constant. Only the transferred half of each
        row is written; the rest of the window stays zero.
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async dynamic subregion P={n_ranks} needs {n_ranks} devices")

        outputs = _run_single_output(test_config, device_ids, n_ranks, _build_dynamic_subregion)

        expected = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
        # Every rank's own row is written locally in full; peer rows only carry
        # the half-row slice that rank transferred at its runtime column offset.
        for r in range(n_ranks):
            expected[r, r, :] = torch.arange(SIZE, dtype=torch.float32) + r * 1000
        for d in range(n_ranks):
            for r in range(n_ranks):
                if r == d:
                    continue
                for j in range(HALF):
                    expected[r, d, d + j] = d * 1000 + d + j
        assert torch.allclose(outputs, expected), (
            f"dynamic subregion mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    def test_burst_submissions(self, test_config, device_ids):
        """Eight transfers submitted from one task share the completion slab."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async burst case needs {n_ranks} devices")

        outputs = _run_single_output(test_config, device_ids, n_ranks, _build_burst)
        expected = _expected_exchange(n_ranks)
        assert torch.allclose(outputs, expected), (
            f"burst mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    def test_put_only_int32(self, test_config, device_ids):
        """A third element type crosses the PUT path."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async INT32 PUT needs {n_ranks} devices")

        outputs = _run_put_only_typed(test_config, device_ids, n_ranks, pl.INT32, torch.int32)
        expected = _typed_expected(n_ranks, torch.int32, step=100)
        assert torch.equal(outputs, expected), (
            f"INT32 PUT mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    def test_self_peer_fails_loudly(self, test_config, device_ids, capfd):
        """A PUT addressed to the issuing rank is rejected."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async self-peer case needs {n_ranks} devices")

        _assert_invalid(test_config, device_ids, n_ranks, _build_self_peer, capfd)

    def test_zero_length_fails_loudly(self, test_config, device_ids, capfd):
        """A runtime transfer extent of zero is rejected."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async zero-length case needs {n_ranks} devices")

        _assert_invalid(test_config, device_ids, n_ranks, _build_zero_length, capfd)

    def test_negative_offset_fails_loudly(self, test_config, device_ids, capfd):
        """A negative runtime column offset is rejected."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async negative-offset case needs {n_ranks} devices")

        _assert_invalid(test_config, device_ids, n_ranks, _build_negative_offset, capfd)

    def test_capacity_overflow_fails_loudly(self, test_config, device_ids, capfd):
        """More than 64 registered submissions from one task is rejected."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async capacity case needs {n_ranks} devices")

        _assert_invalid(
            test_config,
            device_ids,
            n_ranks,
            lambda n: _build_burst(n, repeats=65),
            capfd,
            expected_error="ASYNC_REGISTRATION_FAILED",
        )

    def test_undersized_window_fails_at_compile_time(self, test_config, device_ids):
        """A window view larger than its backing allocation is rejected before codegen."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA async capacity case needs {n_ranks} devices")

        with pytest.raises(ValueError, match="exceeds backing window slot"):
            _compile_invalid(test_config, device_ids, n_ranks, _build_undersized_window)


def _build_early_resolve(n_ranks: int):
    """Build a PUT exchange whose transfer dispatch opts into early resolve.

    The ``pl.at`` block is outlined into the transfer task; the hint lets the
    scheduler pre-stage its consumers, but the consumer body must not enter
    before the transfer retires. The overwrite-after-completion pattern makes
    an early body entry observable in the golden.
    """

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
        data: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pld.DistributedTensor[[n_ranks, 1], pl.INT32]:
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
    def readback(
        data: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        signal: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        for row in pl.range(pld.nranks(ctx)):
            chunk = pl.load(data, [row, 0], [1, SIZE])
            out = pl.store(chunk, [row, 0], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        inp2: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        # A17: the scheduler may pre-stage this task's consumers; the consumer
        # body must still not enter before the transfer retires.
        with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=True):
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
        overwritten = overwrite_after_completion(inp2, data)
        announced = announce(overwritten, signal)
        waited = await_peers(announced)
        return readback(overwritten, waited, out)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32],
        inputs2: pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(n_ranks * pl.INT32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [n_ranks, SIZE], dtype=pl.FP32)
            sig = pld.window(signal_buf, [n_ranks, 1], dtype=pl.INT32)
            chip_orch(inputs[r], inputs2[r], outputs[r], data, sig, device=r)
        return outputs

    return host_orch


@pytest.mark.platforms("a5", reason="URMA async transfers require the A5 backend")
class TestL3UrmaAsyncEarlyResolve:
    def test_early_resolve_does_not_release_early(self, test_config, device_ids):
        """A17: early dispatch must not let the consumer enter its body early."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA early-resolve needs {n_ranks} devices, got {device_ids}")

        inputs = _rank_inputs(n_ranks)
        inputs2 = _rank_inputs(n_ranks, round_index=1)
        outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
        compiled = _build_early_resolve(n_ranks).compile(
            inputs,
            inputs2,
            outputs,
            config=RunConfig(
                platform=test_config.platform,
                distributed_config=DistributedConfig(device_ids=device_ids[:n_ranks], num_sub_workers=0),
            ),
        )
        compiled(
            inputs,
            inputs2,
            outputs,
            config=RunConfig(platform=test_config.platform, enable_dep_gen=True),
        )

        expected = _expected_put(n_ranks)
        assert torch.allclose(outputs, expected), (
            f"early-resolve PUT mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

        dfx_base = runtime_output_directory(compiled) / "dfx_outputs"
        for r in range(n_ranks):
            deps_path = dfx_base / f"rank{r}" / "d0" / "deps.json"
            assert deps_path.is_file(), f"missing deps.json: {deps_path}"
            deps = json.loads(deps_path.read_text())
            assert any(task.get("early_dispatch") is True for task in deps.get("tasks", [])), (
                f"rank {r}: no early_dispatch task in {deps_path}"
            )


def _build_producer_compute(n_ranks: int):
    """A PUT exchange with compute before and after the async submission (A15)."""

    @pl.jit.incore
    def compute_put(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
        extra: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        row = pl.load(inp, [my_rank, 0], [1, SIZE])
        row = pl.add(row, 1.0)  # compute before submission
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
        after = pl.add(row, 2.0)  # compute after submission
        extra = pl.store(after, [my_rank, 0], extra)
        return data

    @pl.jit.incore
    def announce(
        data: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pld.DistributedTensor[[n_ranks, 1], pl.INT32]:
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
    def readback(
        data: pld.DistributedTensor[[n_ranks, SIZE], pl.FP32],
        signal: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        for row in pl.range(pld.nranks(ctx)):
            chunk = pl.load(data, [row, 0], [1, SIZE])
            out = pl.store(chunk, [row, 0], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[n_ranks, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        extra: pl.Out[pl.Tensor[[n_ranks, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[n_ranks, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pl.Tensor[[n_ranks, SIZE], pl.FP32]:
        staged = compute_put(inp, data, extra)
        announced = announce(staged, signal)
        waited = await_peers(announced)
        return readback(staged, waited, out)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
        extras: pl.Out[pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, n_ranks, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(n_ranks * SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(n_ranks * pl.INT32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [n_ranks, SIZE], dtype=pl.FP32)
            sig = pld.window(signal_buf, [n_ranks, 1], dtype=pl.INT32)
            chip_orch(inputs[r], outputs[r], extras[r], data, sig, device=r)
        return outputs, extras

    return host_orch


@pytest.mark.platforms("a5", reason="URMA async transfers require the A5 backend")
class TestL3UrmaAsyncProducerCompute:
    def test_producer_compute_before_and_after_submission(self, test_config, device_ids):
        """A15: compute before and after the async submission stays correct."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA producer-compute case needs {n_ranks} devices, got {device_ids}")

        inputs = _rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
        extras = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
        compiled = _build_producer_compute(n_ranks).compile(
            inputs,
            outputs,
            extras,
            config=RunConfig(
                platform=test_config.platform,
                distributed_config=DistributedConfig(device_ids=device_ids[:n_ranks], num_sub_workers=0),
            ),
        )
        compiled(inputs, outputs, extras, config=RunConfig(platform=test_config.platform))

        expected = _expected_exchange(n_ranks) + 1.0
        assert torch.allclose(outputs, expected), (
            f"producer-compute PUT mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )
        for r in range(n_ranks):
            assert torch.allclose(extras[r, r, :], inputs[r, r, :] + 3.0), (
                f"rank {r}: post-submission compute mismatch"
            )
            others = [d for d in range(n_ranks) if d != r]
            assert torch.count_nonzero(extras[r, others, :]) == 0, f"rank {r}: unexpected extra rows"


CHUNK_ROWS = 65537
CHUNK_COLS = 1024
# 65537 x 1024 x 4 = 268,439,552 bytes = 256 MiB + 4 KiB, just above the
# runtime's kUrmaMaxWqeTransferBytes chunk limit.
CHUNK_BYTES = CHUNK_ROWS * CHUNK_COLS * pl.FP32.get_byte()


def _chunk_inputs(n_ranks: int) -> torch.Tensor:
    """A row-varying pattern so a misordered or short chunk is visible."""
    rows = (torch.arange(CHUNK_ROWS, dtype=torch.float32) % CHUNK_COLS).view(-1, 1)
    inputs = torch.empty((n_ranks, CHUNK_ROWS, CHUNK_COLS), dtype=torch.float32)
    for r in range(n_ranks):
        inputs[r] = rows + r * 10000.0
    return inputs


def _build_chunked_put(n_ranks: int):
    """A full-window PUT larger than the runtime's 256 MiB chunk limit (A16)."""

    @pl.jit.incore
    def fill_and_put(
        inp: pl.Tensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32],
        src: pl.InOut[pld.DistributedTensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32]],
        dst: pl.InOut[pld.DistributedTensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32]],
    ) -> pld.DistributedTensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32]:
        ctx = pld.get_comm_ctx(src)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        for r in pl.range(CHUNK_ROWS):
            row = pl.load(inp, [r, 0], [1, CHUNK_COLS])
            src = pl.store(row, [r, 0], src)
        for peer in pl.range(nranks):
            if peer != my_rank:
                # Full-slice form: nbytes is CHUNK_BYTES, above the chunk limit.
                pld.tensor.put_async(dst, peer=peer, src=src)
        return src

    @pl.jit.incore
    def announce(
        src: pld.DistributedTensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pld.DistributedTensor[[n_ranks, 1], pl.INT32]:
        # `src` is unused: it makes this task depend on the transfer task.
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
        my_rank = pld.rank(pld.get_comm_ctx(signal))
        nranks = pld.nranks(pld.get_comm_ctx(signal))
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
    def readback(
        dst: pld.DistributedTensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32],
        signal: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32]],
    ) -> pl.Tensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32]:
        # `signal` is unused: it makes this task depend on await_peers.
        for r in pl.range(CHUNK_ROWS):
            row = pl.load(dst, [r, 0], [1, CHUNK_COLS])
            out = pl.store(row, [r, 0], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32]],
        dst: pl.InOut[pld.DistributedTensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32]],
        src: pl.InOut[pld.DistributedTensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[n_ranks, 1], pl.INT32]],
    ) -> pl.Tensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32]:
        staged = fill_and_put(inp, src, dst)
        announced = announce(staged, signal)
        waited = await_peers(announced)
        return readback(dst, waited, out)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[n_ranks, CHUNK_ROWS, CHUNK_COLS], pl.FP32],
        outputs: pl.Out[pl.Tensor[[n_ranks, CHUNK_ROWS, CHUNK_COLS], pl.FP32]],
    ) -> pl.Tensor[[n_ranks, CHUNK_ROWS, CHUNK_COLS], pl.FP32]:
        src_buf = pld.alloc_window_buffer(CHUNK_BYTES)
        dst_buf = pld.alloc_window_buffer(CHUNK_BYTES)
        signal_buf = pld.alloc_window_buffer(n_ranks * pl.INT32.get_byte())
        for r in pl.range(pld.world_size()):
            src = pld.window(src_buf, [CHUNK_ROWS, CHUNK_COLS], dtype=pl.FP32)
            dst = pld.window(dst_buf, [CHUNK_ROWS, CHUNK_COLS], dtype=pl.FP32)
            sig = pld.window(signal_buf, [n_ranks, 1], dtype=pl.INT32)
            chip_orch(inputs[r], outputs[r], dst, src, sig, device=r)
        return outputs

    return host_orch


@pytest.mark.platforms("a5", reason="URMA async transfers require the A5 backend")
class TestL3UrmaAsyncChunking:
    def test_put_above_urma_chunk_limit(self, test_config, device_ids):
        """A16: a >256 MiB PUT is chunked by the runtime and arrives intact."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA chunking case needs {n_ranks} devices, got {device_ids}")

        inputs = _chunk_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, CHUNK_ROWS, CHUNK_COLS), dtype=torch.float32)
        compiled = _build_chunked_put(n_ranks).compile(
            inputs,
            outputs,
            config=RunConfig(
                platform=test_config.platform,
                distributed_config=DistributedConfig(device_ids=device_ids[:n_ranks], num_sub_workers=0),
            ),
        )
        compiled(inputs, outputs, config=RunConfig(platform=test_config.platform))

        for r in range(n_ranks):
            peer = (r + 1) % n_ranks
            assert torch.equal(outputs[r], inputs[peer]), (
                f"rank {r}: chunked PUT mismatch against rank {peer}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])

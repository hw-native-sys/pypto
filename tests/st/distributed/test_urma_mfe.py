# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Two-rank URMA MFEs copied from #3379 at c97aa3c1; no Host-domain changes.

Keep the original transfer, notification and source-overwrite checks. The
separate diagnostic runner repeats each test in a fresh process.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

SIZE = 4096


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


@pytest.fixture(autouse=True)
def require_onboard_pair(test_config, device_ids, monkeypatch):
    assert test_config.platform == "a5", "This diagnostic requires onboard A5"
    assert not test_config.codegen_only, "A codegen-only result is not hardware evidence"
    assert len(device_ids) == 2, "Pass exactly the two allocated devices"

    def reject_cold_build(*args, **kwargs):
        raise RuntimeError("Diagnostic cache miss: device compilation is forbidden while holding cards")

    # A missed warmup is a setup failure, never a transfer timeout. READY cache
    # loads use prebuilt bytes and do not reach this compilation boundary.
    monkeypatch.setattr("pypto.runtime.device_runner._compile_and_assemble", reject_cold_build)


def test_put(test_config, device_ids):
    outputs = _run_single_output(test_config, device_ids, 2, _build_put_only)
    assert torch.allclose(outputs, _expected_exchange(2)), "PUT payload mismatch"


def test_get(test_config, device_ids):
    outputs = _run_single_output(test_config, device_ids, 2, _build_get_only)
    assert torch.allclose(outputs, _expected_exchange(2)), "GET payload mismatch"


def test_exchange(test_config, device_ids):
    outputs_put, outputs_get = _run(test_config, device_ids, 2, round_index=0)
    assert torch.allclose(outputs_put, _expected_put(2)), "PUT source-overwrite ordering mismatch"
    assert torch.allclose(outputs_get, _expected_exchange(2) + 1.0), "GET consumer mismatch"

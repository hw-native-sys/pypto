# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: ``pld.tensor.put_async`` asynchronous cross-rank write (TPUT_ASYNC).

The async dual of :file:`test_l3_put.py`. Rank ``r`` issues an SDMA write into
``peer = (r + 1) % nranks``, does local work while it is in flight, then drains
the event before the notify that publishes it. Golden is the same ring shuffle:
``outputs[r] == f(inputs[(r - 1) % nranks])``.

**Why this is a2a3-only, and why sim is not merely weaker but unavailable.**
It is tempting to assume the simulator can at least check the data, since
pto-isa's CPU ``TPUT_ASYNC`` performs a synchronous ``Copy_Data`` and returns
``AsyncEvent(0)`` (``include/pto/cpu/comm/TPut.hpp``). It cannot: a kernel that
builds an SDMA session makes the compiled artifact declare ``enable_sdma``, and
the runtime provisions that workspace **only on a2a3 onboard with the
tensormap_and_ringbuffer runtime** — "host-build-graph, simulation, a5, and
builds without the a2a3 PTO-SDMA provider reject non-empty requirements at
registration" (``runtime/.../common/dma_workspace.h``). So the artifact fails at
worker registration on sim, before any kernel runs. Same reason
``test_prefetch_async.py`` is ``@pytest.mark.platforms("a2a3")``-only.

Everything that *can* be checked without hardware is covered in the unit suite
instead — the emitted PTO (including a real ``ptoas`` assembly round-trip), the
fence placement, and the conversion — so this file is deliberately thin and
carries only what needs two ranks and a live SDMA engine.

**What the second test adds.** :meth:`test_ring_shuffle_async` pins the golden.
:meth:`test_async_matches_sync_result` compares against a synchronous-``put``
control running identical local compute, which is the assertion that would
actually catch a misplaced release marker: if the peer cache invalidate or the
GM fence were emitted at the issue instead of after the wait, the peer could
observe a partially-landed buffer, and the two results would diverge. No timing
assertion is made — a shared box makes wall-clock comparison too noisy to gate
on; use the profiling harness for the overlap number.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

SIZE = 1024  # logically 1-D: TPUT_ASYNC requires a flat-contiguous 1-D region


@pl.jit.incore
def put_async_step(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    src: pld.DistributedTensor[[1, SIZE], pl.FP32],
    dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[1, 1], pl.INT32],
):
    """Issue the remote write, compute locally while it flies, then drain it."""
    ctx = pld.get_comm_ctx(src)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    local = pl.load(x, [0, 0], [1, SIZE])
    src = pl.store(local, [0, 0], src)

    peer = (my_rank + 1) % nranks

    # Build the SDMA session once, then fire without blocking.
    sess = pld.system.async_session()
    evt = pld.tensor.put_async(dst, peer, src, sess)

    # Local work that overlaps the in-flight transfer. It reads `x` rather than
    # `src` so it cannot alias the region SDMA is reading, and it is deliberately
    # more than a single op so there is a real window to overlap.
    scratch = pl.load(x, [0, 0], [1, SIZE])
    scratch = pl.add(scratch, scratch)
    scratch = pl.mul(scratch, scratch)

    # Drain before publishing: the peer-region cache invalidate and the GM
    # release fence are both emitted at this wait, so the notify below must
    # follow it or the peer can observe stale data.
    pld.system.wait_async_event(evt, sess)

    pld.system.notify(
        signal,
        peer=peer,
        offsets=[0, 0],
        value=1,
        op=pld.NotifyOp.AtomicAdd,
    )
    pld.system.wait(
        signal,
        offsets=[0, 0],
        expected=1,
        cmp=pld.WaitCmp.Ge,
    )

    # Our dst slice was written by the rank whose peer is us.
    recv = pl.load(dst, [0, 0], [1, SIZE])
    y = pl.store(recv, [0, 0], y)
    return y


@pl.jit
def per_rank_put_async(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    src: pld.DistributedTensor[[1, SIZE], pl.FP32],
    dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[1, 1], pl.INT32],
):
    return put_async_step(x, y, src, dst, signal)


@pl.jit.host
def ring_put_async(
    x: pl.Tensor[[2, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
):
    src_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    dst_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    signal_buf = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        src = pld.window(src_buf, [1, SIZE], dtype=pl.FP32)
        dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
        per_rank_put_async(x[r], y[r], src, dst, signal, device=r)


@pl.jit.incore
def put_sync_step(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    src: pld.DistributedTensor[[1, SIZE], pl.FP32],
    dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[1, 1], pl.INT32],
):
    """Synchronous control for the overlap comparison — identical local compute."""
    ctx = pld.get_comm_ctx(src)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    local = pl.load(x, [0, 0], [1, SIZE])
    src = pl.store(local, [0, 0], src)

    peer = (my_rank + 1) % nranks
    pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_)

    scratch = pl.load(x, [0, 0], [1, SIZE])
    scratch = pl.add(scratch, scratch)
    scratch = pl.mul(scratch, scratch)

    pld.system.notify(signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    pld.system.wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

    recv = pl.load(dst, [0, 0], [1, SIZE])
    y = pl.store(recv, [0, 0], y)
    return y


@pl.jit
def per_rank_put_sync(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    src: pld.DistributedTensor[[1, SIZE], pl.FP32],
    dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[1, 1], pl.INT32],
):
    return put_sync_step(x, y, src, dst, signal)


@pl.jit.host
def ring_put_sync(
    x: pl.Tensor[[2, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
):
    src_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    dst_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    signal_buf = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        src = pld.window(src_buf, [1, SIZE], dtype=pl.FP32)
        dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
        per_rank_put_sync(x[r], y[r], src, dst, signal, device=r)


def _ring_inputs() -> torch.Tensor:
    """Rank 0 holds [0, 1, …]; rank 1 holds [1000, 1001, …] — distinguishable."""
    return torch.stack(
        [
            torch.arange(SIZE, dtype=torch.float32).reshape(1, SIZE),
            torch.arange(1000.0, 1000.0 + SIZE, dtype=torch.float32).reshape(1, SIZE),
        ]
    )


@pytest.mark.platforms("a2a3")
class TestL3PutAsync:
    """Onboard a2a3: asynchronous cross-rank write via pld.tensor.put_async.

    Sim is excluded deliberately, not incidentally — see the module docstring.
    """

    def test_ring_shuffle_async(self, test_config, device_ids):
        """Rank r's slice ends up holding rank (r-1)'s input, transferred by SDMA."""
        if len(device_ids) < 2:
            pytest.skip(f"async ring put needs 2 devices, got {device_ids}")

        inputs = _ring_inputs()
        outputs = torch.zeros((2, 1, SIZE), dtype=torch.float32)

        compiled = ring_put_async.compile(
            inputs,
            outputs,
            config=RunConfig(
                platform=test_config.platform,
                distributed_config=DistributedConfig(device_ids=device_ids[:2], num_sub_workers=0),
            ),
        )
        compiled(inputs, outputs)

        expected = torch.stack([inputs[1], inputs[0]])
        assert torch.allclose(outputs, expected), (
            f"async ring put mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    def test_async_matches_sync_result(self, test_config, device_ids):
        """The async path produces bit-identical data to the synchronous control.

        Guards the release-marker placement specifically: if the peer-region
        cache invalidate or the GM fence were emitted at the issue rather than
        after the wait, the peer could read a partially-landed buffer, and this
        comparison against the synchronous put is what would catch it.
        """
        if len(device_ids) < 2:
            pytest.skip(f"async ring put needs 2 devices, got {device_ids}")

        inputs = _ring_inputs()
        cfg = RunConfig(
            platform=test_config.platform,
            distributed_config=DistributedConfig(device_ids=device_ids[:2], num_sub_workers=0),
        )

        out_async = torch.zeros((2, 1, SIZE), dtype=torch.float32)
        ring_put_async.compile(inputs, out_async, config=cfg)(inputs, out_async)

        out_sync = torch.zeros((2, 1, SIZE), dtype=torch.float32)
        ring_put_sync.compile(inputs, out_sync, config=cfg)(inputs, out_sync)

        assert torch.equal(out_async, out_sync), (
            "async and synchronous put disagree — the async release markers are likely "
            f"misplaced; max diff = {(out_async - out_sync).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Replay CPU-prepared SHMEM kernels in real Torch graphs on two, four or eight allocated NPUs.

Run under torchrun exactly like test_gloo_shmem_eager.py, after CPU preparation
of capture_kernels.py too. Test segmented replay first, then a complete collective
graph with reusable device barriers and no Host barriers between replays.
"""

import gc
import importlib
import os
import weakref
from dataclasses import replace
from pathlib import Path

import pypto.torch
import pytest
import torch
import torch.distributed as dist
from pypto.runtime import ChipWorker
from pypto.runtime.kernel.context import get_process_kernel_state
from pypto.runtime.kernel_compiler import KernelCompiler
from pypto.runtime.shmem_gloo import acquire_gloo_shmem_window

from examples.distributed.gloo_shmem_utils import load_eager_kernel, run_eager_kernel


def _prepare(monkeypatch, device):
    def forbidden(*args, **kwargs):
        pytest.fail("capture device stage must neither compile nor create a PROGRAM worker")

    monkeypatch.setattr(ChipWorker, "init", forbidden)
    monkeypatch.setattr(KernelCompiler, "compile_incore", forbidden)
    monkeypatch.setattr(KernelCompiler, "compile_orchestration", forbidden)
    directory = Path(os.environ["PYPTO_SHMEM_EAGER_ARTIFACTS"])
    artifacts = {
        name: load_eager_kernel(directory / f"{name}_chip")
        for name in ("local_compute", "tstore", "tput", "publish", "consume", "allreduce", "capture_barrier")
    }
    importlib.import_module("torch_npu")
    torch.npu.set_device(device)
    dist.init_process_group("gloo")
    pypto.torch.init(device=device, runtime=artifacts["local_compute"].runtime_name)
    state = get_process_kernel_state()
    config = state.require_config()
    for artifact in artifacts.values():
        state.ensure_callable(artifact, config)  # Registration only; no business warmup.
    return state, artifacts


def _check_unplayed_graph(record, reset_graphs, call, x, data, out, window):
    # With the Torch task queue disabled, the ordinary tensor itself must be
    # retired. With the queue enabled, Torch can retain its TensorImpl.
    if os.environ.get("TASK_QUEUE_ENABLE", "1") == "0":
        orphan = torch.full_like(x, 3)
        tensor_ref = torch._C._WeakTensorRef(orphan)
        record(lambda value=orphan: call("local_compute", value, out))
        del orphan
        gc.collect()
        assert not tensor_ref.expired()
        reset_graphs()
        gc.collect()
        assert tensor_ref.expired()

    # A separate window owner tests PyPTO retirement independently of Torch.
    owner = replace(window)
    reference = weakref.ref(owner)
    slot = owner.as_torch_tensor("data", tuple(data.shape), data.dtype)
    record(lambda value=slot: call("consume", value, out))
    del slot
    del owner
    gc.collect()
    assert reference() is not None
    reset_graphs()
    gc.collect()
    assert reference() is None


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") not in {"2", "4", "8"} or not os.environ.get("PYPTO_SHMEM_EAGER_ARTIFACTS"),
    reason="requires 2/4/8 allocated NPUs and CPU-prepared artifacts",
)
def test_shmem_graph_replay_and_retirement(monkeypatch):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = int(os.environ["LOCAL_RANK"])
    state, artifacts = _prepare(monkeypatch, device)
    window = acquire_gloo_shmem_window(
        rank=rank,
        world_size=world,
        device=f"npu:{device}",
        slot_names=("data", "src", "dst", "signal"),
        slot_nbytes=(2048, 2048, 2048, world * 4),
    )
    data, src, dst = (
        window.as_torch_tensor(name, (1, 1024), torch.float16) for name in ("data", "src", "dst")
    )
    signal = window.as_torch_tensor("signal", (world, 1), torch.int32)
    signal.zero_()
    x = torch.full((1, 1024), rank + 1, dtype=torch.float16, device=f"npu:{device}")
    mid, out, result = (torch.full_like(x, -999) for _ in range(3))
    stream = torch.npu.Stream(device=device)
    torch.npu.synchronize()
    graphs = []

    def call(name, *args):
        return run_eager_kernel(artifacts[name], *args)

    def record(fn):
        graph = torch.npu.NPUGraph()
        graphs.append(graph)  # Preserve even a failed recording until checked reset.
        with torch.npu.graph(graph, stream=stream):
            fn()
        stream.synchronize()
        return graph

    def replay(graph):
        with torch.npu.stream(stream):
            graph.replay()
        stream.synchronize()
        state.drain()  # Must retain live graph tickets rather than wait on a replay.

    def check(tensor, expected):
        torch.testing.assert_close(
            tensor.cpu(), torch.full((1, 1024), expected, dtype=torch.float16), rtol=0, atol=0
        )

    def reset_graphs():
        for graph in graphs:
            graph.reset()
        graphs.clear()
        state.drain()
        assert not state._submissions, "destroyed graphs still retain submissions"

    try:
        # Two PyPTO nodes in one graph must share the actual ACL destruction notice.
        def send():
            call("local_compute", x, mid)
            torch.add(mid, mid, out=mid)  # Torch/PyPTO ordering inside the recorded graph.
            call("tstore", mid, data)

        send_graph = record(send)
        check(mid, -999)  # Recording must not execute the user's computation.

        def receive():
            call("consume", data, out)
            call("local_compute", out, result)

        receive_graph = record(receive)
        for value in (2, 7, 19):
            x.fill_(rank + value)
            torch.npu.synchronize()
            replay(send_graph)
            dist.barrier()
            replay(receive_graph)
            check(result, ((rank - 1) % world + value) * 8)
            assert all(not t.retired for t in state._submissions if t.capture is not None)
            dist.barrier()
        reset_graphs()

        put_graph = record(lambda: call("tput", x, src, dst))
        consume_graph = record(lambda: call("consume", dst, out))
        for value in (3, 11, 23):
            x.fill_(rank + value)
            torch.npu.synchronize()
            replay(put_graph)
            dist.barrier()
            replay(consume_graph)
            check(out, (rank - 1) % world + value)
            dist.barrier()
        reset_graphs()

        publish_graph = record(lambda: call("publish", x, data))
        sum_graph = record(lambda: call("allreduce", data, out))
        for value in (5, 13, 29):
            x.fill_(rank + value)
            torch.npu.synchronize()
            replay(publish_graph)
            dist.barrier()
            replay(sum_graph)
            check(out, world * value + world * (world - 1) // 2)
            dist.barrier()
        reset_graphs()

        def full_collective():
            call("local_compute", x, mid)
            call("publish", mid, data)
            call("capture_barrier", signal)
            call("allreduce", data, out)
            call("capture_barrier", signal)
            call("local_compute", out, result)

        # Two device barriers order publication and prevent next-replay writes
        # racing remote reads. No Host barrier separates any phase or replay.
        full_graph = record(full_collective)
        for value in (17, 31, 43):
            x.fill_(rank + value)
            torch.npu.synchronize()
            replay(full_graph)
            check(result, (world * value + world * (world - 1) // 2) * 4)
        dist.barrier()  # Only after the complete replay experiment, before teardown.
        reset_graphs()

        _check_unplayed_graph(record, reset_graphs, call, x, data, out, window)
        call("local_compute", x, out)
        torch.npu.synchronize()
        state.drain()
        check(out, (rank + 43) * 2)
        print(
            f"[rank {rank}] GLOO_SHMEM_CAPTURE_PASS world={world} runtime={state.require_config().runtime}",
            flush=True,
        )
    finally:
        torch.npu.synchronize()
        reset_graphs()
        state.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

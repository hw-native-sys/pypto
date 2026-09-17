# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real DSL kernel submission through torch_npu with taskQueue on and off."""

import ctypes
import gc
import multiprocessing
import os
import threading
import traceback

import pypto.language as pl
import pytest
from pypto import CacheConfig
from pypto.pypto_core import passes
from pypto.runtime.runner import RunConfig


@pl.jit
def scaled_kernel(
    x: pl.Tensor[[16, 16], pl.FP32],
    scale: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.mul(pl.load(x, [0, 0], [16, 16]), scale), [0, 0], out)
    return out


def _execute_case(case, platform, device, directory):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415
    from pypto.runtime.kernel.abi import KernelConfig  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch.launch import enqueue  # noqa: PLC0415

    torch_npu.npu.set_device(device)
    os.environ["PYPTO_PROG_BUILD_DIR"] = directory
    config = RunConfig(platform=platform, device_id=device, cache_config=CacheConfig(enabled=False))
    with passes.PassContext([], runtime=passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER):
        artifact = scaled_kernel._resolve_kernel_artifact((), {"config": config}, allow_signature_mode=True)
    state = get_process_kernel_state()
    worker_config = KernelConfig(platform, "tensormap_and_ringbuffer", device)
    # The internal counterpart of pypto.torch.init: registration never initializes.
    state.ensure_worker(worker_config)
    registration = state.ensure_callable(artifact, worker_config)
    stream = torch_npu.npu.Stream(device=device)
    alternate = torch_npu.npu.Stream(device=device)
    outputs = []
    try:
        with torch_npu.npu.stream(stream):
            if case == "delayed":
                _delayed_case(registration, state, device)
            elif case == "failure":
                _failure_case(registration, state, device)
                return
            elif case == "invalid":
                x = torch.ones((16, 16), device=f"npu:{device}")
                with pytest.raises(ValueError, match="contiguous"):
                    enqueue(registration, (x.t(), 2.0, torch.empty_like(x)))
                assert not state._submissions
            else:
                scalar = ctypes.c_float(2.0)
                for step in range(8):
                    current = alternate if case == "streams" and step % 2 else stream
                    with torch_npu.npu.stream(current):
                        # Offset inputs/outputs and fresh allocations on every call.
                        x = torch.full((257,), float(step), device=f"npu:{device}")[1:].view(16, 16)
                        x.add_(1)  # A: queued framework producer before PyPTO.
                        storage = torch.empty((257,), device=f"npu:{device}")
                        out = storage[1:].view(16, 16)
                        scalar.value = float(step + 2)
                        assert enqueue(registration, (x, scalar, out)) is out
                        scalar.value = -99.0
                        outputs.append((out + 3, (step + 1) * (step + 2) + 3))  # B: framework consumer.
                        del x, out, storage
                        gc.collect()
                        pressure = [torch.empty((257,), device=f"npu:{device}") for _ in range(32)]
                        for tensor in pressure:
                            tensor.fill_(-1000)
                        del pressure
        if case == "close":
            # Internal close must drain queued callbacks before releasing Worker.
            state.close()
        else:
            state.drain()
        for actual, expected in outputs:
            torch.testing.assert_close(actual.cpu(), torch.full((16, 16), float(expected)))
    finally:
        if case != "failure":
            state.close()
    # Framework storage/device remains usable after Simpler closes.
    assert torch.ones(1, device=f"npu:{device}").item() == 1


def _delayed_case(registration, state, device):
    import _torch_npu_test  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from pypto.torch.launch import enqueue  # noqa: PLC0415

    x = torch.full((16, 16), 7.0, device=f"npu:{device}")
    out = torch.empty_like(x)
    # Materialize inputs first, then hold the *host* queue before PyPTO's callback.
    torch.npu.synchronize()
    gate = _torch_npu_test.block_queue()
    gate.wait_entered()
    expired = threading.Event()

    def timeout():
        expired.set()
        gate.release()

    watchdog = threading.Timer(10, timeout)
    watchdog.start()
    try:
        scalar = ctypes.c_float(3.0)
        enqueue(registration, (x, scalar, out))
        assert not expired.is_set(), "enqueue drained the blocked host queue"
        assert not state._submissions[0].done()
        scalar.value = -100
        del x
        gc.collect()
    finally:
        gate.release()
        watchdog.cancel()
    state.drain()
    torch.testing.assert_close(out.cpu(), torch.full((16, 16), 21.0))


def _failure_case(registration, state, device):
    import torch  # noqa: PLC0415
    from pypto.torch import launch  # noqa: PLC0415

    native = launch._load_native()
    prepare = native.prepare

    def invalid_callable(worker, callable_id, *args):
        return prepare(worker, 8191, *args)

    x = torch.ones((16, 16), device=f"npu:{device}")
    out = torch.empty_like(x)
    # taskQueue may surface the framework wrapper before the ticket's SDK error.
    error_pattern = "simpler_kernel_mode_launch failed|working operator name is PyPTOKernel"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native, "prepare", invalid_callable)
        with pytest.raises(RuntimeError, match=error_pattern):
            launch.enqueue(registration, (x, 2.0, out))
            state.drain()
    assert len(state._submissions) == 1
    with pytest.raises(RuntimeError, match=error_pattern):
        state.close()
    assert state._worker is None and not state._submissions


def _child(case, platform, device, directory, queue):
    try:
        os.chdir(directory)
        _execute_case(case, platform, device, directory)
        queue.put(None)
    except BaseException:
        queue.put(traceback.format_exc())


def _isolated(case, queue_enabled, platform, device, directory):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    old = os.environ.get("TASK_QUEUE_ENABLE")
    os.environ["TASK_QUEUE_ENABLE"] = str(queue_enabled)
    child = ctx.Process(target=_child, args=(case, platform, device, directory, queue))
    child.start()
    if old is None:
        os.environ.pop("TASK_QUEUE_ENABLE", None)
    else:
        os.environ["TASK_QUEUE_ENABLE"] = old
    try:
        result = queue.get(timeout=180)
        child.join(timeout=15)
        assert result is None, result
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=15)
        queue.close()


@pytest.mark.parametrize("queue_enabled", [0, 1])
@pytest.mark.parametrize("case", ["ordering", "streams", "close", "invalid", "delayed", "failure"])
def test_torch_kernel_launch(test_config, tmp_path, queue_enabled, case):
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("Launch validation requires an A2/A3 NPU and the optional native adapter")
    if case == "delayed" and not queue_enabled:
        pytest.skip("A blocked host callback requires taskQueue enabled")
    pytest.importorskip("torch_npu")
    _isolated(case, queue_enabled, test_config.platform, test_config.device_id, str(tmp_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

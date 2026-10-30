# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate call frames against actual torch_npu storage and stream contexts."""

import ctypes
import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest
import torch
from pypto.ir.param_info import ParamInfo
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.torch.interop import CallSignature


@pytest.fixture
def npu_context(test_config):
    """Use the selected real NPU; simulator/codegen jobs cannot test framework storage."""
    if test_config.codegen_only or test_config.platform.endswith("sim"):
        pytest.skip("torch_npu metadata checks require a real NPU")
    npu = pytest.importorskip("torch_npu")
    if not npu.npu.is_available():
        pytest.skip("torch_npu reports no available NPU")
    with npu.npu.device(test_config.device_id):
        yield npu


def test_real_npu_offset_alias_and_stream_snapshots(npu_context):
    """Two calls borrow the correct views and current streams without a PyPTO launch."""
    npu = npu_context
    base = torch.empty((4, 3), dtype=torch.float32, device="npu")
    x = base[1:3]
    out = torch.empty_like(x)
    signature = CallSignature(
        [
            ParamInfo("x", ParamDirection.In, [2, 3], DataType.FP32),
            ParamInfo("step", ParamDirection.In, None, DataType.INT32),
            ParamInfo("out", ParamDirection.InOut, [2, 3], DataType.FP32),
        ],
        return_aliases=(2,),
    )
    first_stream, second_stream = npu.npu.Stream(), npu.npu.Stream()
    step = ctypes.c_int32(1)
    with npu.npu.stream(first_stream):
        first = signature.describe_call((x, step, out))
    step.value = 2
    with npu.npu.stream(second_stream):
        second = signature.describe_call((x, step, out))
    assert first.stream == first_stream and second.stream == second_stream
    assert first.scalars[0].value == 1 and second.scalars[0].value == 2
    assert first.tensors[0].metadata.data_ptr == base.data_ptr() + 3 * base.element_size()
    assert first.tensors[0].metadata.storage_offset == 3
    assert first.tensors[0].metadata.format == int(npu.get_npu_format(x))
    assert first.alias_result() is out and second.alias_result() is out


@pytest.mark.parametrize("column_start", [0, 1, 3])
def test_real_npu_empty_slice_storage_offset(npu_context, column_start):
    """Legal empty NPU slices may point at or beyond their storage's end."""
    base = torch.empty((2, 3), device="npu")
    view = base[2:, column_start:]
    signature = CallSignature(
        [ParamInfo("x", ParamDirection.In, [-1, -1], DataType.FP32)], return_aliases=(0,)
    )
    frame = signature.describe_call((view,))
    metadata = frame.tensors[0].metadata
    assert metadata.shape == (0, 3 - column_start)
    assert metadata.storage_offset == 6 + column_start
    assert metadata.nbytes == 0
    assert metadata.storage_ptr == base.untyped_storage().data_ptr()
    assert frame.alias_result() is view


def test_real_npu_rejects_transposed_input(npu_context):
    """A noncontiguous NPU view is rejected rather than copied or normalized."""
    value = torch.empty((3, 2), device="npu").t()
    signature = CallSignature([ParamInfo("x", ParamDirection.In, [2, 3], DataType.FP32)])
    with pytest.raises(ValueError, match="contiguous strided tensor"):
        signature.describe_call((value,))


@pytest.mark.parametrize("drain", ["query", "empty_cache", "full_queue"])
def test_python_queue_callback_survives_gil_held_drain(npu_context, test_config, drain):
    """Queue drains work in a fresh process without external ACL-op warmup."""
    if importlib.util.find_spec("pypto._torch_npu") is None:
        pytest.skip("optional native adapter was not built")
    # An installed but unloadable adapter is a regression, not an optional skip.
    importlib.import_module("pypto._torch_npu")
    script = textwrap.dedent("""
        import sys
        import threading
        import time
        import torch
        import torch_npu
        from pypto import _torch_npu

        device = int(sys.argv[1])
        torch.npu.set_device(device)
        stream = torch.npu.current_stream()
        context = _torch_npu.borrow_context(device)
        events = [torch.npu.Event() for _ in range(8192)] if sys.argv[2] == "full_queue" else []
        # The first enqueue drains the INIT queue synchronously. Complete a
        # nonblocking OpApi callback before testing a later GIL-held drain.
        warmed = threading.Event()

        def warmup(caller_stream):
            assert caller_stream > 0
            warmed.set()

        _torch_npu.enqueue([], stream.stream_id, device, context, warmup)
        assert warmed.wait(10)
        started, unblock, finished = threading.Event(), threading.Event(), threading.Event()

        def callback(caller_stream):
            assert caller_stream > 0
            started.set()
            assert unblock.wait(10)
            finished.set()

        # The process has no external ACL-op warmup: the adapter must arrange
        # GIL-safe queue drains before admitting any Python callback.
        _torch_npu.enqueue([], stream.stream_id, device, context, callback)
        assert started.wait(10)
        filled, progress, failures = threading.Event(), [0], []

        def release_full_queue():
            # Do not release the consumer merely because a fixed timer fired:
            # first prove the producer reached capacity and then stopped there.
            if not filled.wait(10):
                failures.append("producer did not reach queue capacity")
            else:
                time.sleep(0.2)
                if not 4090 <= progress[0] < 8192:
                    failures.append(f"producer did not block at capacity: {progress[0]}")
            unblock.set()

        timer = (threading.Thread(target=release_full_queue) if events
                 else threading.Timer(0.1, unblock.set))
        timer.start()
        if sys.argv[2] == "query":
            stream.query()
        elif sys.argv[2] == "empty_cache":
            torch.npu.empty_cache()
        else:
            # Exceed the framework's 4096-entry queue while the consumer is
            # blocked in a Python callback. Event.record enters with the GIL.
            for index, event in enumerate(events):
                event.record(stream)
                progress[0] = index + 1
                if progress[0] == 4090:
                    filled.set()
        timer.join()
        assert not failures, failures
        assert finished.wait(10)
        stream.synchronize()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(test_config.device_id), drain],
        env={**os.environ, "TASK_QUEUE_ENABLE": "1", "ASCEND_LAUNCH_BLOCKING": "0"},
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

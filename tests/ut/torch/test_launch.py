# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Queue ordering, public TaskArgs and ownership across delayed eager submissions."""

import gc
import threading
import weakref
from dataclasses import replace

import pytest
import torch
from pypto._artifact_contract import ArtifactExecutionMode, ExecutionCapabilities
from pypto.backend import BackendType
from pypto.ir._artifact_metadata import ArtifactMetadata
from pypto.ir.compiled_artifact import CompiledArtifact
from pypto.ir.param_info import ParamInfo
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.runtime.kernel.arguments import BorrowedArguments
from pypto.runtime.kernel.context import KernelState
from pypto.torch import interop, launch


@pytest.fixture
def prepared(eager_env, tmp_path):
    env = eager_env
    env.init()
    params = (
        ParamInfo("x", ParamDirection.In, (16, 16), DataType.FP32),
        ParamInfo("step", ParamDirection.In, None, DataType.INT32),
        ParamInfo("out", ParamDirection.InOut, (16, 16), DataType.FP32),
    )
    meta = ArtifactMetadata(
        params,
        params,
        1,
        "a2a3",
        BackendType.Ascend910,
        (2,),
        ExecutionCapabilities((ArtifactExecutionMode.KERNEL,)),
    )
    artifact = CompiledArtifact(tmp_path, "a2a3", meta)
    # These are state/interop tests. Real JIT compilation has separate coverage.
    artifact._chip_callable, artifact._runtime_name = object(), "tensormap_and_ringbuffer"
    config = env.state.require_config()
    signature = interop.CallSignature(params, return_aliases=(2,), platform="a2a3")
    return artifact, config, signature


def _invoke(env, prepared, *, x=None, out=None):
    artifact, config, signature = prepared
    x = env.tensor() if x is None else x
    out = env.tensor() if out is None else out
    frame = signature.describe_call((x, -2, out))
    assert launch.invoke(artifact, frame, config) is out
    return frame


@pytest.mark.parametrize("queued", [False, True])
def test_callback_order_and_stream_snapshot(eager_env, prepared, queued):
    env = eager_env
    env.queued = queued
    if queued:
        env.queue.append(lambda: env.events.append("torch.before"))
    else:
        env.events.append("torch.before")
    _invoke(env, prepared)
    if queued:
        assert not env.runs
        env.queue.append(lambda: env.events.append("torch.after"))
    else:
        env.events.append("torch.after")
    env.stream_id, env.rts_stream = 8, 67890
    env.run_queue()
    submit_index = next(i for i, event in enumerate(env.events) if isinstance(event, tuple))
    assert env.events.index("torch.before") < submit_index < env.events.index("torch.after")
    assert env.events[submit_index][4] == 12345  # Captured RTS stream, not current/logical ID.
    assert not env.runs[0].done and not env.state._submissions[0].done()
    assert "wait" not in env.events and "synchronize" not in env.events
    env.state.close()


def test_storage_offset_and_aliases_share_one_source(eager_env, prepared):
    env = eager_env
    base = env.tensor((32, 16))
    view = base[16:]
    _invoke(env, prepared, x=view, out=view)
    env.run_queue()
    args = next(e[2] for e in env.events if isinstance(e, tuple))
    assert len(env.buffers) == 1
    assert env.buffers[0].address == base.untyped_storage().data_ptr()
    assert env.buffers[0].nbytes == base.untyped_storage().nbytes()
    assert [view.byte_offset for view, _ in args.tensors] == [16 * 16 * 4] * 2
    assert args.tensors[0][0].buffer is args.tensors[1][0].buffer
    assert [tag for _, tag in args.tensors] == ["IN", "INOUT"]
    assert type(args.scalars[0]).__name__ == "c_int" and args.scalars[0].value == -2
    env.state.close()


def test_repeated_retirement_releases_worker_imports(eager_env, prepared):
    env = eager_env
    worker = env.workers[0]
    x, out = env.tensor(), env.tensor()
    for _ in range(3):
        _invoke(env, prepared, x=x, out=out)
        _invoke(env, prepared, x=x, out=out)
        env.run_queue()
        assert len(worker.imports) == 2
        first, second = env.state._submissions[-2:]
        env.runs[-2].complete = True
        assert first.done() and len(worker.imports) == 2
        env.runs[-1].complete = True
        assert second.done() and not worker.imports
    env.state.close()


def test_owners_live_until_retirement_and_closures_are_released(eager_env, prepared):
    env = eager_env
    frame = _invoke(env, prepared)
    owner = weakref.ref(frame.tensors[0].tensor)
    ticket = env.state._submissions[0]
    del frame
    gc.collect()
    assert owner() is not None and not ticket.done()
    env.run_queue()
    assert not ticket.done()
    gc.collect()
    assert owner() is not None
    env.runs[0].complete = True
    assert ticket.done()
    gc.collect()
    assert owner() is None and ticket._callback is None and ticket._owners == ()
    env.state.close()


def test_partial_enqueue_failure_keeps_callback_until_explicit_close(eager_env, prepared, monkeypatch):
    env = eager_env
    native_enqueue = env.native.enqueue

    def enqueue_then_fail(*args):
        native_enqueue(*args)
        raise RuntimeError("failed after queue admission")

    monkeypatch.setattr(env.native, "enqueue", enqueue_then_fail)
    with pytest.raises(RuntimeError, match="after queue admission"):
        _invoke(env, prepared)
    assert not env.runs and env.queue
    ticket = env.state._submissions[0]
    assert ticket._callback is not None and ticket._owners
    assert not any(buffer.closed for buffer in env.buffers)
    env.state.close()
    assert env.events.index("synchronize") < env.events.index("wait") < env.events.index("source.close")
    assert ticket._callback is None and not ticket._owners


@pytest.mark.parametrize("failure", ["enqueue", "submit", "context", "stream"])
def test_submission_failure_retains_owners_and_original_error(eager_env, prepared, failure):
    env = eager_env
    error = ValueError("test submission failure")
    if failure == "enqueue":
        env.enqueue_error = error
        with pytest.raises(ValueError) as caught:
            _invoke(env, prepared)
        assert caught.value is error
    else:
        if failure == "stream":
            env.rts_stream = 0
        _invoke(env, prepared)
        if failure == "submit":
            env.submit_error = error
        elif failure == "context":
            env.context += 1
        with pytest.raises((RuntimeError, ValueError)):
            env.run_queue()
    ticket = env.state._submissions[0]
    assert ticket._owners and not any(b.closed for b in env.buffers)
    with pytest.raises((RuntimeError, ValueError)):
        env.state.close()
    assert env.state.state is KernelState.FAILED
    assert "close" not in env.events and not any(b.closed for b in env.buffers)


@pytest.mark.parametrize("failure", ["wait", "source"])
def test_retirement_failure_preserves_resources_for_retry(eager_env, prepared, failure):
    env = eager_env
    _invoke(env, prepared)
    env.run_queue()
    run = env.runs[0]
    run.complete = True
    if failure == "wait":
        run.wait_error = RuntimeError("retirement failed")
    else:
        env.buffers[0].close_error = RuntimeError("retirement failed")
    ticket = env.state._submissions[0]
    with pytest.raises(RuntimeError, match="retirement failed"):
        ticket.done()
    assert ticket._owners
    assert not env.buffers[0].closed
    assert env.workers[0].imports
    run.wait_error = env.buffers[0].close_error = None
    assert ticket.done()
    env.state.close()
    assert all(b.closed for b in env.buffers)
    assert not env.workers[0].imports


def test_close_can_finish_queued_callback_without_admission_lock_deadlock(eager_env, prepared):
    env = eager_env
    _invoke(env, prepared)
    errors = []

    def close():
        try:
            env.state.close()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=close)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    assert env.events.index("synchronize") < env.events.index("wait") < env.events.index("close")
    assert env.state.state is KernelState.CLOSED


def test_close_synchronizes_the_retained_stream(eager_env, prepared):
    env = eager_env
    _invoke(env, prepared)
    assert not env.runs
    env.stream_id = 8
    env.state.close()
    assert env.synchronized_streams == [7]


def test_sources_reject_another_worker_location(eager_env, prepared):
    env = eager_env
    frame = _invoke(env, prepared)
    session = env.state.bound_worker(prepared[1])
    with pytest.raises(ValueError, match="different Worker"):
        BorrowedArguments(frame, session.sources, object())
    env.state.close()


def test_native_nz_uses_physical_shape_and_no_transfer(eager_env, prepared, monkeypatch):
    env = eager_env
    monkeypatch.setattr(env.framework, "get_npu_format", lambda tensor: 29)
    monkeypatch.setattr(interop, "_native_storage_shape", lambda tensor: (1, 2, 1, 16, 16))
    info = ParamInfo("x", ParamDirection.In, (16, 32), DataType.FP16, layout="NZ")
    tensor = env.tensor((16, 32), dtype=torch.float16)
    frame = interop.CallSignature([info], platform="a2a3").describe_call((tensor,))
    session = env.state.bound_worker(prepared[1])
    args = BorrowedArguments(frame, session.sources, session.device_location)
    view, tag = args.task_args.tensors[0]
    assert view.shapes == (1, 2, 1, 16, 16) and view.strides == (512, 256, 256, 16, 1)
    assert view.byte_offset == 0 and view.buffer.nbytes == 1024 and tag == "IN"
    args.release()
    env.state.close()


def test_empty_frame_and_changed_device_rejected(eager_env, prepared):
    env = eager_env
    frame = _invoke(env, prepared)
    with pytest.raises(ValueError, match="device"):
        launch.check_frame(replace(frame, device_index=1), prepared[1])
    bad = replace(frame.tensors[0], metadata=replace(frame.tensors[0].metadata, shape=(0, 16)))
    with pytest.raises(ValueError, match="positive"):
        launch.check_frame(replace(frame, tensors=(bad,)), prepared[1])
    env.state.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

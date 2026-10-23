# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Submit eager kernels at the Torch host-queue boundary using public Worker APIs."""

import importlib
from typing import Any


def _load_native() -> Any:
    try:
        native = importlib.import_module("pypto._torch_npu")
    except ModuleNotFoundError as exc:
        if exc.name != "pypto._torch_npu":
            raise
        raise RuntimeError(
            "Torch kernel initialization requires the optional pypto._torch_npu adapter; "
            "build with PYPTO_BUILD_TORCH_NPU=ON and matching torch/torch_npu/CANN"
        ) from exc
    for name in ("check_call", "borrow_context", "bind_context", "framework_alive", "retain_until_exit"):
        if not callable(getattr(native, name, None)):
            raise NotImplementedError(f"Torch lifecycle adapter requires {name}; rebuild the adapter")
    return native


def _capture_supported() -> bool:
    # The production backend has no graph lifetime contract. Keep the policy
    # separate so fake-capable UTs can exercise lookup-only capture admission.
    return False


def _check_capture(stream_id: int, device: int) -> bool:
    capturing = bool(_load_native().check_call(stream_id, device))
    if capturing and not _capture_supported():
        raise NotImplementedError("Torch kernel graph capture/replay is not supported, even after warmup")
    return capturing


def check_eager_call(config: Any) -> bool:
    """Reject capture/device mismatch before specialization or compilation."""
    from .interop import _load_torch_npu  # noqa: PLC0415

    npu = _load_torch_npu().npu
    device = npu.current_device()
    if device != config.device_id:
        raise ValueError(f"Kernel call device {device} differs from initialized NPU {config.device_id}")
    stream = npu.current_stream(device)
    return _check_capture(stream.stream_id, device)


def check_frame(frame: Any, config: Any) -> bool:
    if frame.device_index != config.device_id:
        raise ValueError("Kernel call device differs from its initialized Worker device")
    for argument in frame.tensors:
        meta = argument.metadata
        if not 1 <= len(meta.shape) <= 5 or any(
            extent <= 0 or extent > 0xFFFFFFFF for extent in (*meta.shape, *meta.strides)
        ):
            raise ValueError("Kernel tensors require rank 1..5 and positive u32 extents/strides")
    return _check_capture(frame.stream.stream_id, frame.device_index)


def invoke(artifact: Any, frame: Any, config: Any) -> Any:
    """Register a shared artifact, retain a ticket, then enqueue without waiting."""
    from pypto.runtime.kernel.arguments import BorrowedArguments  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.runtime.kernel.pending import PendingTicket  # noqa: PLC0415
    from pypto.runtime.kernel.session import make_call_config  # noqa: PLC0415

    capturing = check_frame(frame, config)
    state = get_process_kernel_state()
    native = _load_native()
    if not callable(getattr(native, "prepare", None)):
        raise NotImplementedError("Torch eager adapter requires prepare; rebuild the optional adapter")
    registration = (
        state.require_callable(artifact, config) if capturing else state.ensure_callable(artifact, config)
    )

    def prepare(session: Any) -> PendingTicket:
        # Capture stream/context and native Tensor owners on the calling thread.
        bridge = native.prepare(
            [arg.tensor for arg in frame.tensors], frame.stream.stream_id, frame.device_index
        )
        call_config = make_call_config(config, session._api.CallConfig)
        arguments = BorrowedArguments(frame, session.sources, session.device_location)

        def enqueue(ticket: PendingTicket) -> None:
            def submit(caller_stream: int) -> None:
                try:
                    if type(caller_stream) is not int or caller_stream <= 0:
                        raise ValueError("Kernel submission requires a nonzero RTS stream handle")
                    session.check_context()
                    run = session.worker.submit(
                        registration.handle, arguments.task_args, call_config, caller_stream=caller_stream
                    )
                    ticket.publish(run)
                except BaseException as error:
                    ticket.fail(error)
                    raise  # Preserve the framework's synchronous/asynchronous error channel.

            # Native callback state holds only a borrowed PyObject pointer. The
            # process pending set already owns this ticket, including on failure.
            ticket.retain_callback(submit)
            bridge.enqueue(submit)

        def release() -> None:
            arguments.release()
            bridge.release()

        return PendingTicket(
            registration,
            enqueue=enqueue,
            synchronize=bridge.synchronize,
            release=release,
            owners=(session, arguments, call_config, frame, bridge),
        )

    state.submit(registration, prepare)
    return frame.alias_result()


__all__ = ["_load_native"]

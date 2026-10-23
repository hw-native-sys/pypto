# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit process initialization for the borrowed-context kernel runtime."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path


def init(  # noqa: PLR0913
    *,
    device: int | None = None,
    platform: str = "a2a3",
    runtime: str = "tensormap_and_ringbuffer",
    aicpu_thread_num: int = 0,
    enable_chip_swimlane: int | bool = 0,
    enable_dep_gen: bool = False,
    output_dir: str | Path | None = None,
    ring_task_window: int | Sequence[int] | None = None,
    ring_heap: int | Sequence[int] | None = None,
    ring_dep_pool: int | Sequence[int] | None = None,
    workspace_budget_bytes: int | None = None,
) -> None:
    """Initialize one process Worker borrowing the current torch_npu context.

    Repeating the same effective configuration is idempotent. The current NPU
    device must match ``device``; this function never switches it. Kernel and
    program execution are mutually exclusive, including after a failed native
    initialization or shutdown. Compilation alone does not claim either mode.

    Ring sizing accepts a positive scalar or four per-depth values. Diagnostics
    require ``output_dir``. The optional workspace budget is in bytes and is
    passed to the public Worker. Graph capture is unsupported. Direct JIT calls
    use this initialized Worker and the current Torch stream; compile() retains
    explicit program semantics.
    """
    from pypto.runtime.kernel import session  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.runtime.kernel.session import KernelConfig, make_call_config  # noqa: PLC0415

    from . import interop, launch, shutdown  # noqa: PLC0415

    state = get_process_kernel_state()  # Reject inherited state before touching NPU APIs.
    if device is not None and type(device) is not int:
        raise TypeError(f"device must be an int NPU index or None, got {device!r}")
    config = KernelConfig(
        platform,
        runtime,
        0 if device is None else device,
        aicpu_thread_num=aicpu_thread_num,
        enable_chip_swimlane=enable_chip_swimlane,
        enable_dep_gen=enable_dep_gen,
        output_dir=output_dir,
        ring_task_window=ring_task_window,
        ring_heap=ring_heap,
        ring_dep_pool=ring_dep_pool,
        workspace_budget_bytes=workspace_budget_bytes,
    )
    # Validate static settings before importing optional framework/runtime modules.
    framework = interop._load_torch_npu()
    current = framework.npu.current_device()
    if device is not None and device != current:
        raise ValueError(
            f"pypto.torch.init(device={device}) must match current torch_npu device {current}; "
            f"call torch.npu.set_device({device}) first"
        )
    config = replace(config, device_id=current)
    shutdown.require_supported_framework(framework)
    api = session.require_runtime()
    make_call_config(config, api.CallConfig)
    native = launch._load_native()
    stream = framework.npu.current_stream(current)
    if native.check_call(stream.stream_id, current):
        raise NotImplementedError("Torch kernel graph capture is not supported; initialize outside capture")
    native.borrow_context(current)  # Check live context before claiming kernel mode.
    state.ensure_worker(config)

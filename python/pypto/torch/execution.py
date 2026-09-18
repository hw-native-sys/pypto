# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit process-level execution information for kernel calls."""

from pathlib import Path

from pypto._kernel_abi import EAGER_KERNEL_TARGETS, KernelABI


def init(
    *,
    device: int | None = None,
    platform: str = "a2a3",
    runtime: str = "tensormap_and_ringbuffer",
    aicpu_thread_num: int = 0,
    enable_chip_swimlane: int | bool = 0,
    enable_dep_gen: bool = False,
    output_dir: str | Path | None = None,
) -> None:
    """Fix this process's kernel execution target and initialize its Worker.

    Call once before the first direct ``@pl.jit`` or registered ``torch.ops``
    kernel call, outside graph capture. Execution information is not passed per
    call, so graph capture, compiler tracing and dispatcher registration never
    snapshot a device or runtime choice. Registration, Fake/Meta calls and
    explicit program compilation do not require this call.

    Repeating ``init`` outside capture with the same effective configuration is
    a no-op. Kernel and program execution remain mutually exclusive within one
    process.

    Args:
        device: NPU index. None uses torch_npu's current device; an explicit
            value must equal it because init never switches the framework device.
        platform: Kernel target platform family.
        runtime: Simpler runtime bound by the process Worker.
        aicpu_thread_num: 0 selects the runtime default; otherwise 2..5.
        enable_chip_swimlane: Collection level 0..4; True selects full (4).
            Use begin_dfx/end_dfx outside capture to select measured launches.
        enable_dep_gen: Collect the task graph independently of timing.
        output_dir: Artifact directory, required when either diagnostic is enabled.

    Raises:
        TypeError: ``device`` is neither None nor an int.
        ValueError: The target, device or thread count is invalid, differs from
            the configuration already bound in this process, or the native
            Simpler binding does not match the kernel ABI revision.
        RuntimeError: Called inside graph capture, with an unverified framework
            version, after kernel shutdown, in a process that claimed program
            mode, or in a forked child.

    Examples:
        >>> import torch_npu, pypto.torch
        >>> torch_npu.npu.set_device(3)
        >>> pypto.torch.init(aicpu_thread_num=4)
        >>> op(x, 2.0, out)  # @pl.jit kernel on NPU 3; no per-call execution config
    """
    from pypto.runtime import _kernel_artifact  # noqa: PLC0415
    from pypto.runtime.kernel.abi import KernelConfig  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415

    from . import interop, launch, shutdown  # noqa: PLC0415

    # Reject a forked child before touching the framework or native state.
    state = get_process_kernel_state()
    if (platform, runtime) not in EAGER_KERNEL_TARGETS:
        raise ValueError(
            f"Kernel execution supports (platform, runtime) in {sorted(EAGER_KERNEL_TARGETS)}, "
            f"got ({platform!r}, {runtime!r})"
        )
    if device is not None and type(device) is not int:
        raise TypeError(
            f"pypto.torch.init device must be an int NPU index or None, got {type(device).__name__}"
        )
    framework = interop._load_torch_npu()
    current: int = framework.npu.current_device()
    if device is not None and device != current:
        raise ValueError(
            f"pypto.torch.init(device={device}) must match the current torch_npu device {current}; "
            f"call torch.npu.set_device({device}) first"
        )
    config = KernelConfig(
        platform, runtime, current, aicpu_thread_num, enable_chip_swimlane, enable_dep_gen, output_dir
    )
    # Every check below runs before ensure_worker claims kernel mode.
    shutdown.require_supported_framework(framework)
    _kernel_artifact.require_kernel_native(KernelABI(platform, runtime, ()))
    stream = framework.npu.current_stream(current)
    if launch._load_native().check_call(stream.stream_id, current):
        raise RuntimeError("pypto.torch.init must be called outside graph capture")
    state.ensure_worker(config)


def _dfx_window(*, begin: bool) -> None:
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415

    from . import interop, launch  # noqa: PLC0415

    state = get_process_kernel_state()
    config = state.require_config()
    framework = interop._load_torch_npu()
    if framework.npu.current_device() != config.device_id:
        raise ValueError(f"Kernel DFX requires the initialized device {config.device_id} to be current")
    stream = framework.npu.current_stream(config.device_id)
    if launch._load_native().check_call(stream.stream_id, config.device_id):
        raise RuntimeError("Kernel DFX begin/end must be called outside graph capture")
    state.collect_dfx(stream, begin=begin)


def begin_dfx() -> None:
    """Open a swimlane window on the current stream, outside graph capture.

    Requires init with swimlane or dep_gen enabled and output_dir. Warm up operators
    first. This drains earlier stream work before opening the window. Serialize
    collection with other kernel work and keep all measured launches/replays on
    this stream. Nested windows are rejected.
    """
    _dfx_window(begin=True)


def end_dfx() -> None:
    """Drain the window's current stream and write its swimlane artifact.

    Must run outside capture, on the same stream as begin_dfx. Drains torch_npu's
    host task queue before the runtime collects device records. The first window
    writes into init's output_dir; later windows use window_1, window_2, etc.
    """
    _dfx_window(begin=False)

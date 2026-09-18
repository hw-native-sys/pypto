# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal eager/captured submission of a prepared kernel through the torch_npu queue."""

import importlib
import struct
from collections.abc import Hashable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from pypto._kernel_abi import SCALAR_FORMATS, SIMPLER_KERNEL_REVISION, TENSOR_DTYPE_TAGS, KernelABI
from pypto.ir.param_info import ParamInfo
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.runtime.kernel.callable import KernelRegistration

from .interop import CallFrame, CallSignature

if TYPE_CHECKING:
    from pypto.runtime.kernel.abi import KernelConfig


def _load_native() -> Any:
    try:
        native = importlib.import_module("pypto._torch_npu")
    except ImportError as exc:
        raise RuntimeError(
            "Kernel submission requires the optional PyPTO torch_npu adapter; "
            "rebuild with -DPYPTO_BUILD_TORCH_NPU=ON using matching torch, torch_npu and Simpler"
        ) from exc
    if native.simpler_revision != SIMPLER_KERNEL_REVISION:
        raise RuntimeError("The PyPTO torch_npu adapter uses a different Simpler revision; rebuild it")
    return native


def enqueue(registration: KernelRegistration, args: Sequence[Any]) -> Any:
    """Snapshot complete arguments and enqueue; return the declared output aliases.

    This internal entry accepts a prepared registration, never compiles or prepares
    a callable, and does not wait for device completion. The process manager owns
    pending submissions even when the caller drops every tensor reference.
    """
    registration.require_live()
    abi = registration.artifact.kernel_abi
    frame = _describe(abi, args)
    return _enqueue_frame(registration, frame)


def _describe(abi: KernelABI, args: Sequence[Any]) -> CallFrame:
    params = [
        ParamInfo(
            p.name,
            getattr(ParamDirection, p.direction),
            list(p.shape) if p.shape is not None else None,
            getattr(DataType, "BF16" if p.dtype == "bfloat16" else p.dtype.upper()),
        )
        for p in abi.parameters
    ]
    return CallSignature(params, return_aliases=abi.return_aliases).describe_call(args)


def describe_eager_call(abi: KernelABI, args: Sequence[Any], bound: "KernelConfig") -> CallFrame:
    """Validate against the target bound by ``pypto.torch.init`` before compilation or registration."""
    frame = _describe(abi, args)
    for tensor in frame.tensors:
        meta = tensor.metadata
        if not 1 <= len(meta.shape) <= 5 or any(
            x <= 0 or x > 0xFFFFFFFF for x in (*meta.shape, *meta.strides)
        ):
            raise ValueError("JIT eager tensors require rank 1..5 and positive u32 extents/strides")
    if frame.device_index != bound.device_id:
        raise ValueError(
            f"Kernel call tensors are on NPU {frame.device_index}, but pypto.torch.init bound NPU "
            f"{bound.device_id}; kernel mode uses one device per process"
        )
    capture_id = _load_native().check_call(frame.stream.stream_id, frame.device_index)
    return replace(frame, capture_id=capture_id or 0)


def invoke(
    artifact: Any, frame: CallFrame, bound: "KernelConfig", *, specialization: Hashable | None = None
) -> Any:
    """Register once on the initialized process Worker and submit one prevalidated eager call."""
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415

    state = get_process_kernel_state()
    registration = (
        state.require_callable(artifact, bound)
        if frame.capture_id
        else state.ensure_callable(artifact, bound)
    )
    if specialization is not None and not frame.capture_id:
        state.publish_specialization(specialization, registration)
    return _enqueue_frame(registration, frame)


def _enqueue_frame(registration: KernelRegistration, frame: CallFrame) -> Any:
    registration.require_live()
    if frame.device_index != registration.owner.config.device_id:
        raise ValueError("Kernel call device differs from its process Worker device")
    native = _load_native()
    capture_id = native.check_call(frame.stream.stream_id, frame.device_index)
    if capture_id:
        from .capture import install_capture  # noqa: PLC0415

        install_capture(registration.owner).notice(capture_id)
    scalar_bits = [
        int.from_bytes(struct.pack(f"<{SCALAR_FORMATS[str(s.dtype)]}", s.value), "little")
        for s in frame.scalars
    ]

    # The native constructor copies metadata and holds native Tensor/Storage owners.
    # No Python object is captured by the eventual taskQueue callback.
    def prepare(worker: Any) -> Any:
        return native.prepare(
            worker.native_launch_target,
            registration.handle,
            [t.tensor for t in frame.tensors],
            [TENSOR_DTYPE_TAGS[str(t.metadata.dtype)] for t in frame.tensors],
            scalar_bits,
            frame.stream.stream_id,
            frame.device_index,
        )

    registration.owner.submit(registration, prepare)
    return frame.alias_result()

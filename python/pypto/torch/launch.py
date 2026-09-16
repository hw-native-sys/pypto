# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal eager submission of a prepared kernel through the torch_npu queue."""

import importlib
import struct
from collections.abc import Sequence
from typing import Any

from pypto._kernel_abi import SCALAR_FORMATS, SIMPLER_KERNEL_REVISION, TENSOR_DTYPE_TAGS
from pypto.ir.param_info import ParamInfo
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.runtime.kernel.callable import KernelRegistration

from .interop import CallSignature


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
    params = [
        ParamInfo(
            p.name,
            getattr(ParamDirection, p.direction),
            list(p.shape) if p.shape is not None else None,
            getattr(DataType, "BF16" if p.dtype == "bfloat16" else p.dtype.upper()),
        )
        for p in abi.parameters
    ]
    frame = CallSignature(params, return_aliases=abi.return_aliases).describe_call(args)
    if frame.device_index != registration.owner.config.device_id:
        raise ValueError("Kernel call device differs from its process Worker device")
    native = _load_native()
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

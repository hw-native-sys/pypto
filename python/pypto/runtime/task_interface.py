# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Re-exports from ``simpler.task_interface`` and ``simpler.worker``.

All C++ nanobind types (DataType, ChipCallable, ChipStorageTaskArgs, etc.) and
torch-aware helpers (make_tensor_arg, scalar_to_uint64) come from the
``simpler`` package installed via ``pip install simpler``.
"""

from simpler.task_interface import (  # pyright: ignore[reportMissingImports]
    CallConfig,
    ChipCallable,
    ChipStorageTaskArgs,
    CoreCallable,
    DataType,
    Tensor,
    scalar_to_uint64,
)

# ``RunTiming`` is a native ``_task_interface`` nanobind type in older Simpler
# builds. Newer Simpler revisions no longer export it because timing is emitted
# through profiling logs. Keep a small compatibility stand-in so compile-only
# flows and callers that only read the attributes can still import PyPTO.
try:
    from simpler.worker import RunTiming, Worker  # pyright: ignore[reportMissingImports]
except ImportError:
    from simpler.worker import Worker  # pyright: ignore[reportMissingImports]

    class RunTiming:  # pyright: ignore[reportRedeclaration]
        def __init__(self, host_wall_ns: int = 0, device_wall_ns: int = 0) -> None:
            self.host_wall_ns = host_wall_ns
            self.device_wall_ns = device_wall_ns

        @property
        def host_wall_us(self) -> float:
            return self.host_wall_ns / 1000.0

        @property
        def device_wall_us(self) -> float:
            return self.device_wall_ns / 1000.0
from simpler_setup.torch_interop import (  # pyright: ignore[reportMissingImports]
    make_tensor_arg,
    torch_dtype_to_datatype,
)

from .device_tensor import DeviceTensor


def device_tensor_to_tensor(dt: DeviceTensor) -> Tensor:
    """Wrap a worker-resident :class:`DeviceTensor` as a simpler ``Tensor``.

    ``child_memory=True`` tells the runtime the buffer is already on the device,
    so it skips the H2D/D2H copies — the buffer stays caller-managed. Shared by
    the L2 (:func:`pypto.runtime.runner.execute_compiled`) and L3
    (:func:`pypto.runtime.tensor_arg.make_tensor_arg`) calling conventions.
    """
    try:
        dt_enum = torch_dtype_to_datatype(dt.dtype)
    except KeyError as e:
        raise ValueError(f"Unsupported DeviceTensor dtype: {dt.dtype}") from e
    return Tensor.make(data=dt.data_ptr, shapes=dt.shape, dtype=dt_enum, child_memory=True)


__all__ = [
    "CallConfig",
    "ChipCallable",
    "ChipStorageTaskArgs",
    "CoreCallable",
    "DataType",
    "RunTiming",
    "Tensor",
    "Worker",
    "device_tensor_to_tensor",
    "make_tensor_arg",
    "scalar_to_uint64",
    "torch_dtype_to_datatype",
]

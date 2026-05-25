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
    ContinuousTensor,
    CoreCallable,
    DataType,
    scalar_to_uint64,
)
from simpler.worker import Worker  # pyright: ignore[reportMissingImports]
from simpler_setup.torch_interop import (  # pyright: ignore[reportMissingImports]
    make_tensor_arg,
    torch_dtype_to_datatype,
)

from .device_tensor import DeviceTensor


def device_tensor_to_continuous(dt: DeviceTensor) -> ContinuousTensor:
    """Wrap a worker-resident :class:`DeviceTensor` as a ``ContinuousTensor``.

    ``child_memory=True`` tells the runtime the buffer is already on the device,
    so it skips the H2D/D2H copies — the buffer stays caller-managed. Shared by
    the L2 (:func:`pypto.runtime.runner.execute_compiled`) and L3
    (:func:`pypto.runtime.tensor_arg.make_tensor_arg`) calling conventions.
    """
    try:
        dt_enum = torch_dtype_to_datatype(dt.dtype)
    except KeyError as e:
        raise ValueError(f"Unsupported DeviceTensor dtype: {dt.dtype}") from e
    return ContinuousTensor.make(data=dt.data_ptr, shapes=dt.shape, dtype=dt_enum, child_memory=True)


__all__ = [
    "CallConfig",
    "ChipCallable",
    "ChipStorageTaskArgs",
    "ContinuousTensor",
    "CoreCallable",
    "DataType",
    "Worker",
    "device_tensor_to_continuous",
    "make_tensor_arg",
    "scalar_to_uint64",
    "torch_dtype_to_datatype",
]

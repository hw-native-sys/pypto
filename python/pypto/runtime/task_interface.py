# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Re-exports from ``simpler.task_interface`` and ``simpler.worker``.

All C++ nanobind types (DataType, ChipCallable, TaskArgs, etc.) and torch-aware
helpers come from the ``simpler`` package installed via ``pip install
simpler``.
"""

from simpler.task_interface import (  # pyright: ignore[reportMissingImports]
    CallConfig,  # pyright: ignore[reportAttributeAccessIssue]
    ChipCallable,  # pyright: ignore[reportAttributeAccessIssue]
    ChipStorageTaskArgs,  # pyright: ignore[reportAttributeAccessIssue]
    ChipTensor,  # pyright: ignore[reportAttributeAccessIssue]
    CoreCallable,  # pyright: ignore[reportAttributeAccessIssue]
    DataType,  # pyright: ignore[reportAttributeAccessIssue]
    TaskArgs,  # pyright: ignore[reportAttributeAccessIssue]
    Tensor,  # pyright: ignore[reportAttributeAccessIssue]
    scalar_to_uint64,  # pyright: ignore[reportAttributeAccessIssue]
)
from simpler.worker import Worker  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from simpler_setup.torch_interop import (  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
    make_chip_tensor_arg,
    torch_dtype_to_datatype,
)

from .device_tensor import DeviceTensor


def device_tensor_to_chip_tensor(dt: DeviceTensor) -> ChipTensor:
    """Legacy direct-chip adapter from :class:`DeviceTensor` to ``ChipTensor``.

    ``child_memory=True`` tells the runtime the buffer is already on the device,
    so it skips the H2D/D2H copies and leaves lifetime caller-managed. This is
    retained for low-level compatibility only; high-level public dispatch uses
    address-free ``TaskArgs`` and does not call this helper.
    """
    try:
        dt_enum = torch_dtype_to_datatype(dt.dtype)
    except KeyError as e:
        raise ValueError(f"Unsupported DeviceTensor dtype: {dt.dtype}") from e
    return ChipTensor.make(data=dt.data_ptr, shapes=dt.shape, dtype=dt_enum, child_memory=True)


# Compatibility aliases for existing imports. New internal call sites use the
# explicit chip-scoped names so they cannot be confused with the worker-aware,
# address-free helper in :mod:`pypto.runtime.tensor_arg`.
device_tensor_to_tensor = device_tensor_to_chip_tensor
make_tensor_arg = make_chip_tensor_arg


__all__ = [
    "CallConfig",
    "ChipCallable",
    "ChipStorageTaskArgs",
    "ChipTensor",
    "CoreCallable",
    "DataType",
    "TaskArgs",
    "Tensor",
    "Worker",
    "device_tensor_to_chip_tensor",
    "device_tensor_to_tensor",
    "make_chip_tensor_arg",
    "make_tensor_arg",
    "scalar_to_uint64",
    "torch_dtype_to_datatype",
]

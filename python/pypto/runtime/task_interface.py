# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLW0603, PLC0415

"""Python API for task_interface nanobind bindings.

Re-exports C++ types (DataType, ContinuousTensor, ChipStorageTaskArgs, etc.)
from the ``_task_interface`` nanobind module (installed via ``pip install simpler``)
and adds torch-aware convenience helpers.
"""

from _task_interface import (  # pyright: ignore[reportMissingImports]
    CONTINUOUS_TENSOR_MAX_DIMS,
    ArgDirection,
    ChipCallable,
    ChipCallConfig,
    ChipStorageTaskArgs,
    ContinuousTensor,
    CoreCallable,
    DataType,
    DynamicTaskArgs,
    TaggedTaskArgs,
    TensorArgType,
    _ChipWorker,
    arg_direction_name,
    get_dtype_name,
    get_element_size,
)

__all__ = [
    "DataType",
    "get_element_size",
    "get_dtype_name",
    "CONTINUOUS_TENSOR_MAX_DIMS",
    "ContinuousTensor",
    "ChipStorageTaskArgs",
    "TensorArgType",
    "DynamicTaskArgs",
    "TaggedTaskArgs",
    "ArgDirection",
    "CoreCallable",
    "ChipCallable",
    "ChipCallConfig",
    "ChipWorker",
    "arg_direction_name",
    "torch_dtype_to_datatype",
    "make_tensor_arg",
    "scalar_to_uint64",
]


# Lazy-loaded torch dtype → DataType map (avoids importing torch at module load)
_TORCH_DTYPE_MAP = None


def _ensure_torch_map():
    global _TORCH_DTYPE_MAP
    if _TORCH_DTYPE_MAP is not None:
        return
    import torch

    _TORCH_DTYPE_MAP = {
        torch.float32: DataType.FLOAT32,
        torch.float16: DataType.FLOAT16,
        torch.int32: DataType.INT32,
        torch.int16: DataType.INT16,
        torch.int8: DataType.INT8,
        torch.uint8: DataType.UINT8,
        torch.bfloat16: DataType.BFLOAT16,
        torch.int64: DataType.INT64,
    }


def torch_dtype_to_datatype(dt) -> DataType:
    """Convert a ``torch.dtype`` to a ``DataType`` enum value.

    Raises ``KeyError`` for unsupported dtypes.
    """
    _ensure_torch_map()
    return _TORCH_DTYPE_MAP[dt]  # pyright: ignore[reportOptionalSubscript]


def make_tensor_arg(tensor) -> ContinuousTensor:
    """Create a ``ContinuousTensor`` from a ``torch.Tensor``.

    The tensor must be CPU-contiguous.
    """
    _ensure_torch_map()
    dt = _TORCH_DTYPE_MAP.get(tensor.dtype)  # pyright: ignore[reportOptionalMemberAccess]
    if dt is None:
        raise ValueError(f"Unsupported tensor dtype for ContinuousTensor: {tensor.dtype}")
    shapes = tuple(int(s) for s in tensor.shape)
    return ContinuousTensor.make(tensor.data_ptr(), shapes, dt)


def scalar_to_uint64(value) -> int:
    """Convert a scalar value to ``uint64``.

    *value* can be a Python int, float, a ctypes scalar (``c_int64``,
    ``c_float``, etc.), or any object convertible to ``int``.

    Python float values are converted to IEEE 754 single precision (32-bit)
    and their bit pattern is zero-extended to uint64.
    """
    import struct as _struct

    if isinstance(value, float):
        bits = _struct.unpack("<I", _struct.pack("<f", value))[0]
        return bits
    import ctypes as _ct

    if isinstance(value, _ct._SimpleCData):
        if isinstance(value, (_ct.c_float, _ct.c_double)):
            uint_type = _ct.c_uint32 if isinstance(value, _ct.c_float) else _ct.c_uint64
            return uint_type.from_buffer_copy(value).value
        return int(value.value) & 0xFFFFFFFFFFFFFFFF
    return int(value) & 0xFFFFFFFFFFFFFFFF


class ChipWorker:
    """Unified execution interface wrapping the host runtime C API.

    Usage::

        worker = ChipWorker()
        worker.init(host_path="build/lib/.../host.so",
                    aicpu_path="build/lib/.../aicpu.so",
                    aicore_path="build/lib/.../aicore.o")
        worker.set_device(device_id=0)
        worker.run(chip_callable, orch_args, block_dim=24)
        worker.reset_device()
        worker.finalize()
    """

    def __init__(self):
        self._impl = _ChipWorker()

    def init(self, host_path, aicpu_path, aicore_path, sim_context_lib_path=""):
        """Load host runtime library and cache platform binaries."""
        self._impl.init(str(host_path), str(aicpu_path), str(aicore_path), str(sim_context_lib_path))

    def set_device(self, device_id):
        """Set the target NPU device."""
        self._impl.set_device(device_id)

    def reset_device(self):
        """Release device resources. The runtime binding remains intact."""
        self._impl.reset_device()

    def finalize(self):
        """Tear down everything: device resources and runtime library."""
        self._impl.finalize()

    def run(self, callable, args, config=None, **kwargs):
        """Execute a callable synchronously.

        Args:
            callable: ChipCallable built from orchestration + kernel binaries.
            args: ChipStorageTaskArgs for this invocation.
            config: Optional ChipCallConfig. If ``None``, a default is created.
            **kwargs: Overrides applied to config (e.g. ``block_dim=24``).
        """
        if config is None:
            config = ChipCallConfig()
        for k, v in kwargs.items():
            setattr(config, k, v)
        self._impl.run(callable, args, config)

    @property
    def device_id(self):
        return self._impl.device_id

    @property
    def initialized(self):
        return self._impl.initialized

    @property
    def device_set(self):
        return self._impl.device_set

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate borrowed NPU arguments and capture one call's metadata and owners.

This internal adapter does not compile, initialize a Simpler Worker, register
callables or enqueue work. Native ABI encoding and asynchronous allocator
protection belong to the launch adapter, not to these Python frames.
"""

import ctypes
import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor

from pypto.ir.param_info import _DATATYPE_TO_CTYPE, ParamInfo, _to_torch_dtype, bind_complete_args
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection


@dataclass(frozen=True)
class TensorMetadata:
    """Snapshot of one tensor's logical view and its borrowed storage bounds."""

    name: str
    param_index: int
    direction: ParamDirection
    dtype: DataType
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    device_index: int
    format: int
    data_ptr: int
    storage_ptr: int
    storage_offset: int
    storage_nbytes: int
    nbytes: int


@dataclass(frozen=True)
class TensorArgument:
    """Keep the tensor and its storage alive alongside immutable view metadata."""

    metadata: TensorMetadata
    tensor: torch.Tensor = field(repr=False, compare=False)
    storage: torch.UntypedStorage = field(repr=False, compare=False)


@dataclass(frozen=True)
class ScalarArgument:
    """Typed primitive value copied from this call, retaining its signature slot."""

    name: str
    param_index: int
    dtype: DataType
    value: int | float | bool


@dataclass(frozen=True)
class CallFrame:
    """One call's independent values, current stream and strong tensor references.

    Tensor and scalar tuples each preserve signature order; ``param_index``
    retains their original mixed positions. These are not native ABI pools.
    Keeping this frame alive protects Python owners only, not asynchronous
    device uses after the frame is released.
    """

    tensors: tuple[TensorArgument, ...]
    scalars: tuple[ScalarArgument, ...]
    device_index: int
    stream: Any = field(repr=False, compare=False)
    return_tensors: tuple[torch.Tensor, ...] = field(repr=False, compare=False)

    def alias_result(self) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Return the exact externally supplied objects named by return aliases."""
        if not self.return_tensors:
            return None
        return self.return_tensors[0] if len(self.return_tensors) == 1 else self.return_tensors


def _load_torch_npu() -> Any:
    """Load the optional framework extension only for a real call description."""
    try:
        return importlib.import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError("Describing NPU calls requires a compatible torch_npu installation") from exc


def _scalar_value(value: Any, info: ParamInfo) -> int | float | bool:
    """Copy a scalar using the same ctypes representation as program parameters."""
    ctype = _DATATYPE_TO_CTYPE.get(str(info.dtype))
    if ctype is None:
        raise TypeError(f"Parameter {info.name!r} has unsupported scalar dtype {info.dtype}")
    if isinstance(value, ctypes._SimpleCData):
        if type(value) is not ctype:
            raise TypeError(f"Parameter {info.name!r} expects {ctype.__name__}, got {type(value).__name__}")
        value = value.value
    if ctype is ctypes.c_bool:
        if type(value) is not bool:
            raise TypeError(f"Parameter {info.name!r} expects bool, got {type(value).__name__}")
    elif ctype in (ctypes.c_float, ctypes.c_double):
        if type(value) not in (int, float):
            raise TypeError(f"Parameter {info.name!r} expects a numeric scalar, got {type(value).__name__}")
    else:
        if type(value) is not int:
            raise TypeError(f"Parameter {info.name!r} expects an integer scalar, got {type(value).__name__}")
        bits = ctypes.sizeof(ctype) * 8
        signed = ctype(-1).value < 0
        minimum, maximum = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
        if not minimum <= value <= maximum:
            raise ValueError(
                f"Parameter {info.name!r} expects {info.dtype} in [{minimum}, {maximum}], got {value}"
            )
    return ctype(value).value


def _validate_tensor(tensor: Any, info: ParamInfo, dtype: torch.dtype) -> None:
    """Reject incompatible inputs before loading torch_npu or reading pointers."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Parameter {info.name!r} requires an NPU torch.Tensor, got {type(tensor).__name__}")
    if isinstance(tensor, FakeTensor) or tensor.is_meta:
        raise TypeError(f"Parameter {info.name!r} requires real NPU storage, got a Fake/Meta tensor")
    if tensor.device.type != "npu":
        raise TypeError(
            f"Parameter {info.name!r} requires an NPU torch.Tensor, got device {tensor.device}; "
            "use explicit compile() and the program execution path for host or Worker-owned tensors"
        )
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise ValueError(f"Parameter {info.name!r} requires a contiguous strided tensor; no copy is made")
    if tensor.is_conj() or tensor.is_neg():
        raise ValueError(f"Parameter {info.name!r} has an unresolved conjugate or negative view")
    if tensor.requires_grad and torch.is_grad_enabled():
        raise ValueError(f"Parameter {info.name!r} requires gradients; this adapter has no autograd contract")
    if tensor.dtype != dtype:
        raise TypeError(f"Parameter {info.name!r} expects dtype {dtype}, got {tensor.dtype}")
    shape = tuple(tensor.shape)
    assert info.shape is not None
    if len(shape) != len(info.shape) or any(
        expected not in (-1, actual) for expected, actual in zip(info.shape, shape)
    ):
        raise ValueError(f"Parameter {info.name!r} expects shape {tuple(info.shape)}, got {shape}")


def _describe_tensor(tensor: torch.Tensor, info: ParamInfo, index: int, npu: Any) -> TensorArgument:
    """Capture a contiguous base-format view without moving or normalizing it."""
    tensor_format = int(npu.get_npu_format(tensor))
    if tensor_format not in (0, 2):
        raise ValueError(
            f"Parameter {info.name!r} requires base format NCHW (0) or ND (2), got {tensor_format}"
        )
    storage = tensor.untyped_storage()
    pointer, base = tensor.data_ptr(), storage.data_ptr()
    offset, itemsize = int(tensor.storage_offset()), tensor.element_size()
    nbytes, capacity = tensor.numel() * itemsize, storage.nbytes()
    # Empty slices may have offsets beyond capacity because they access no elements.
    if offset < 0 or (nbytes and offset * itemsize + nbytes > capacity):
        raise ValueError(f"Parameter {info.name!r} has a view outside its storage bounds")
    if nbytes and (base <= 0 or pointer != base + offset * itemsize):
        raise ValueError(
            f"Parameter {info.name!r} has an invalid logical data pointer for its storage offset"
        )
    device = tensor.device.index
    if device is None or device < 0:
        raise ValueError(f"Parameter {info.name!r} requires an indexed NPU device, got {tensor.device}")
    metadata = TensorMetadata(
        info.name,
        index,
        info.direction,
        info.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        device,
        tensor_format,
        pointer,
        base,
        offset,
        capacity,
        nbytes,
    )
    return TensorArgument(metadata, tensor, storage)


def _validate_aliases(tensors: Sequence[TensorArgument]) -> None:
    """Allow exact byte-span aliases and read-only overlaps; reject partial writes."""
    views: dict[tuple[Any, ...], tuple[TensorMetadata, bool]] = {}
    for tensor in tensors:
        m = tensor.metadata
        if not m.nbytes:
            continue
        # A page pool may be viewed as FP32 state and BF16 KV with different
        # shapes. Equal pointer and byte span are still an exact alias.
        key = (m.data_ptr, m.nbytes)
        writable = m.direction != ParamDirection.In
        previous = views.get(key)
        views[key] = (m, writable or (previous is not None and previous[1]))
    end = writable_end = 0
    for m, writable in sorted(views.values(), key=lambda entry: entry[0].data_ptr):
        if m.data_ptr < writable_end or (writable and m.data_ptr < end):
            raise ValueError(f"Parameter {m.name!r} partially overlaps another tensor with a writable alias")
        end = max(end, m.data_ptr + m.nbytes)
        if writable:
            writable_end = max(writable_end, m.data_ptr + m.nbytes)


class CallSignature:
    """Pre-parse shared parameter metadata for repeated internal torch calls.

    Shapes use ParamInfo's carrier dimensions, including ``-1`` dynamic axes.
    Return aliases are indices into the full parameter list and must refer to
    tensors. No framework, compiler or Worker state is initialized here.
    """

    def __init__(
        self,
        params: Sequence[ParamInfo],
        *,
        return_aliases: Sequence[int] = (),
        caller_name: str = "torch call",
    ) -> None:
        """Copy stable parameter metadata and validate declared return aliases."""
        self._params = tuple(
            ParamInfo(p.name, p.direction, list(p.shape) if p.shape is not None else None, p.dtype)
            for p in params
        )
        self._caller_name = caller_name
        self._return_aliases = tuple(return_aliases)
        self._torch_dtypes = tuple(_to_torch_dtype(p.dtype) for p in self._params)
        for p, dtype in zip(self._params, self._torch_dtypes, strict=True):
            if p.shape is not None and dtype is None:
                raise TypeError(f"Parameter {p.name!r} has unsupported tensor dtype {p.dtype}")
            if p.shape is None and str(p.dtype) not in _DATATYPE_TO_CTYPE:
                raise TypeError(f"Parameter {p.name!r} has unsupported scalar dtype {p.dtype}")
        for index in self._return_aliases:
            if (
                type(index) is not int
                or not 0 <= index < len(self._params)
                or self._params[index].shape is None
            ):
                raise ValueError(f"Return alias must reference a tensor parameter, got {index!r}")

    def describe_call(self, args: Sequence[Any]) -> CallFrame:
        """Validate all arguments, then capture this call's device, stream and owners.

        Does not allocate output tensors, convert layouts, initialize a Worker,
        encode native ABI arguments, read a raw stream handle or submit work.
        """
        bound = bind_complete_args(args, self._params, caller_name=self._caller_name)
        scalars: list[ScalarArgument] = []
        device_indices: set[int | None] = set()
        for index, (arg, info, dtype) in enumerate(zip(bound, self._params, self._torch_dtypes, strict=True)):
            if info.shape is None:
                scalars.append(ScalarArgument(info.name, index, info.dtype, _scalar_value(arg, info)))
            else:
                assert dtype is not None
                _validate_tensor(arg, info, dtype)
                device_indices.add(arg.device.index)
        if len(device_indices) > 1:
            raise ValueError(f"{self._caller_name} requires one NPU device, got {device_indices}")
        npu = _load_torch_npu()
        tensors = tuple(
            _describe_tensor(arg, info, index, npu)
            for index, (arg, info) in enumerate(zip(bound, self._params, strict=True))
            if info.shape is not None
        )
        _validate_aliases(tensors)
        device = npu.npu.current_device()
        if type(device) is not int or device < 0:
            raise ValueError(f"{self._caller_name} requires a valid current NPU device, got {device!r}")
        if device_indices and device_indices != {device}:
            raise ValueError(
                f"{self._caller_name} tensor devices {device_indices} differ from current device {device}"
            )
        stream = npu.npu.current_stream(device)
        if stream.device.type != "npu" or stream.device.index != device:
            raise ValueError(
                f"{self._caller_name} current stream belongs to {stream.device}, expected npu:{device}"
            )
        return CallFrame(
            tensors, tuple(scalars), device, stream, tuple(bound[i] for i in self._return_aliases)
        )

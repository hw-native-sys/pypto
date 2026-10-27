# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Borrow Torch storage as public Buffer/Tensor/TaskArgs without device copies."""

import importlib
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pypto.ir.param_info import _DATATYPE_TO_CTYPE, block_nz_shape
from pypto.pypto_core.ir import ParamDirection
from pypto.runtime.runtime_pin import check_runtime_pin


def _load_api() -> Any:
    check_runtime_pin()
    return importlib.import_module("simpler.task_interface")


def check_tensor_dtype(dtype: Any) -> None:
    """Reject unsupported borrowed tensor types before registration or enqueue."""
    import torch  # noqa: PLC0415

    if dtype == torch.bool:
        raise ValueError("Torch eager kernels do not support torch.bool tensors in the pinned runtime")


def _tensor_dtype(dtype: Any) -> Any:
    check_tensor_dtype(dtype)
    from simpler_setup.torch_interop import torch_dtype_to_datatype  # noqa: PLC0415

    return torch_dtype_to_datatype(dtype)


@dataclass
class _Source:
    buffer: Any
    storage: Any  # Keep the actual allocation owner even if preparation fails.
    uses: int = 1


class SourcePool:
    """One session's source registrations, shared by all pending calls.

    Buffer.wrap rejects overlapping registrations. Equal storage ranges reuse a
    source even when Tensor views differ; each call takes one lease per storage.
    A failed close keeps both the registration and allocation owner for retry.
    """

    def __init__(self, release_buffer: Callable[[Any], None]) -> None:
        self._release_buffer = release_buffer
        self._location: Any = None
        self._sources: dict[tuple[int, int], _Source] = {}
        self._lock = threading.Lock()

    def acquire(self, argument: Any, location: Any, api: Any) -> tuple[tuple[int, int], Any]:
        meta = argument.metadata
        key = (meta.storage_ptr, meta.storage_nbytes)
        with self._lock:
            if self._location is None:
                self._location = location
            elif self._location != location:
                raise ValueError("Borrowed sources belong to a different Worker location")
            source = self._sources.get(key)
            if source is None:
                buffer = api.Buffer.wrap(address=key[0], nbytes=key[1], location=location)
                source = self._sources[key] = _Source(buffer, argument.storage)
            else:
                source.uses += 1
            return key, source.buffer

    def release(self, key: tuple[int, int]) -> None:
        with self._lock:
            source = self._sources[key]
            if source.uses == 1:
                self._release_buffer(source.buffer)
                del self._sources[key]
            else:
                source.uses -= 1

    def close(self) -> None:
        """Revoke remaining preparation-only sources after pending runs retire."""
        with self._lock:
            for key in list(self._sources):
                self._release_buffer(self._sources[key].buffer)
                del self._sources[key]


class BorrowedArguments:
    """A call's TaskArgs and retryable leases; release only after run retirement."""

    def __init__(self, frame: Any, pool: SourcePool, location: Any) -> None:
        self._pool = pool
        self._keys: list[tuple[int, int]] = []
        api = _load_api()
        self.task_args = api.TaskArgs()
        sources: dict[tuple[int, int], Any] = {}
        tags = {
            ParamDirection.In: api.TensorArgType.INPUT,
            ParamDirection.Out: api.TensorArgType.OUTPUT_EXISTING,
            ParamDirection.InOut: api.TensorArgType.INOUT,
        }
        try:
            # TaskArgs has independent tensor/scalar pools. Preserve signature
            # order within each pool, adding all tensors before typed scalars.
            for argument in frame.tensors:
                meta = argument.metadata
                key = (meta.storage_ptr, meta.storage_nbytes)
                if key not in sources:
                    key, source = pool.acquire(argument, location, api)
                    self._keys.append(key)
                    sources[key] = source
                shape, strides = meta.shape, meta.strides
                if meta.format == 29:
                    shape = tuple(block_nz_shape(shape, argument.tensor.dtype))
                    strides = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
                view = api.Tensor(
                    sources[key],
                    shapes=shape,
                    strides=strides,
                    dtype=_tensor_dtype(argument.tensor.dtype),
                    byte_offset=meta.storage_offset * argument.tensor.element_size(),
                )
                self.task_args.add_tensor(view, tags[meta.direction], transfer=api.TensorTransfer.NONE)
            for scalar in frame.scalars:
                self.task_args.add_scalar(_DATATYPE_TO_CTYPE[str(scalar.dtype)](scalar.value))
        except BaseException:
            # Nothing was enqueued. It is safe to revoke these leases; a failed
            # revocation still retains its storage in the session's source pool.
            self.release()
            raise

    def release(self) -> None:
        while self._keys:
            self._pool.release(self._keys[-1])
            self._keys.pop()

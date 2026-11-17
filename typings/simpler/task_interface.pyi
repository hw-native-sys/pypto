# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Consumed surface of ``simpler.task_interface``.

Mirrors the real module's re-exports: the nanobind wire types from
``_task_interface`` plus ``Buffer``/``Tensor`` re-exported through
``simpler.buffer``.
"""

from collections.abc import Callable
from dataclasses import dataclass

from _task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    ChipStorageTaskArgs,
    ChipTensor,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
    scalar_to_uint64,
)

from .buffer import Buffer, Tensor, TensorTransfer

__all__ = [
    "ArgDirection",
    "Buffer",
    "CallConfig",
    "ChipCallable",
    "ChipDomainContext",
    "CommDomainHandle",
    "CommBufferSpec",
    "ChipStorageTaskArgs",
    "ChipTensor",
    "CoreCallable",
    "DataType",
    "TaskArgs",
    "Tensor",
    "TensorArgType",
    "TensorTransfer",
    "scalar_to_uint64",
]

@dataclass
class CommBufferSpec:
    name: str
    dtype: str
    count: int
    nbytes: int
    load_from_host: bool = ...
    store_to_host: bool = ...

@dataclass
class ChipDomainContext:
    name: str
    domain_rank: int
    domain_size: int
    device_ctx: int
    local_window_base: int
    actual_window_size: int
    buffers: dict[str, Buffer]

class CommDomainHandle:
    contexts: dict[int, ChipDomainContext]
    def __init__(
        self,
        *,
        name: str,
        workers: tuple[int, ...] = ...,
        contexts: dict[int, ChipDomainContext] | None = ...,
        allocation_id: int = ...,
        _release_fn: Callable[[CommDomainHandle], None],
        _domain_size: int | None = ...,
        _domain_ranks: dict[int, int] | None = ...,
    ) -> None: ...
    def __getitem__(self, chip_idx: int) -> ChipDomainContext: ...
    def __enter__(self) -> CommDomainHandle: ...
    def __exit__(self, *args: object) -> None: ...
    def release(self) -> None: ...

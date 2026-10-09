# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Consumed surface of ``simpler.worker``."""

from _task_interface import (
    CallConfig,
    DeviceMemoryInfo,
)

from .buffer import Buffer
from .task_interface import Tensor

class CallableHandle:
    """Opaque public token returned by ``Worker.register``."""

    hashid: str
    kind: str
    target_namespace: str

class RunHandle:
    """Completion handle returned by ``Worker.submit``."""

    def done(self) -> bool: ...
    def wait(self, timeout: float | None = ...) -> None: ...
    def result(self, timeout: float | None = ...) -> None: ...

class Worker:
    def __init__(self, level: int, **config: object) -> None: ...
    def init(
        self,
        prewarm_config: CallConfig | None = ...,
        *,
        _startup_deadline: float | None = ...,
    ) -> None: ...
    def register(self, target: object, *, workers: list[int] | None = ...) -> CallableHandle: ...
    def unregister(self, handle_or_slot: object) -> None: ...
    def run(self, callable: object, args: object = ..., config: object = ...) -> None: ...
    def submit(self, callable: object, args: object = ..., config: object = ...) -> RunHandle: ...
    def close(self) -> None: ...
    def malloc(self, size: int) -> Buffer: ...
    def free(self, handle: Buffer) -> None: ...
    def release_buffer(self, buffer: Buffer) -> None: ...
    def create_buffer(self, nbytes: int) -> Buffer: ...
    def alloc_child_tensor(self, worker_id: int, shapes: tuple[int, ...], dtype: object) -> Buffer: ...
    def copy_to(
        self,
        dst: Buffer,
        src: object,
        *,
        dst_offset: int = ...,
        src_offset: int = ...,
        nbytes: int | None = ...,
    ) -> None: ...
    def copy_from(
        self,
        dst: object,
        src: Buffer,
        *,
        dst_offset: int = ...,
        src_offset: int = ...,
        nbytes: int | None = ...,
    ) -> None: ...
    def make_tensor_arg(
        self,
        tensor: object,
        shapes: tuple[int, ...],
        dtype: int,
        *,
        strides: tuple[int, ...] | None = ...,
    ) -> Tensor: ...
    def committed_device_memory(self, worker_id: int = ...) -> int: ...
    def device_memory_info(self, worker_id: int = ...) -> DeviceMemoryInfo: ...
    def aicpu_dlopen_count(self) -> int: ...
    def host_dlopen_count(self) -> int: ...

__all__ = ["CallableHandle", "RunHandle", "Worker"]

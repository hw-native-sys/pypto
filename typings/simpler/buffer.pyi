# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Consumed surface of ``simpler.buffer``."""

from collections.abc import Iterable

from _task_interface import (
    AccessMode as AccessMode,
)
from _task_interface import (
    AddressSpace as AddressSpace,
)
from _task_interface import (
    BackendKind as BackendKind,
)
from _task_interface import (
    BufferDescriptor as BufferDescriptor,
)
from _task_interface import (
    CanonicalIdentity as CanonicalIdentity,
)
from _task_interface import (
    DataType,
)
from _task_interface import (
    Tensor as Tensor,
)

class Buffer:
    """Owner-side registry object for one shared backing."""

    identity: CanonicalIdentity
    address_space: AddressSpace
    access: AccessMode
    backend_kind: BackendKind
    nbytes: int
    body: bytes
    owner_worker_path_id: int
    owner_worker_id: int
    closed: bool
    unlinked: bool
    base: int

    def freeze_descriptor(self) -> None: ...
    def to_descriptor(self) -> BufferDescriptor: ...
    def tensor(
        self,
        shapes: Iterable[int],
        dtype: int | DataType,
        strides: Iterable[int] | None = ...,
        byte_offset: int = ...,
    ) -> Tensor: ...
    def close(self) -> None: ...

def mint_owner_instance_id() -> bytes: ...
def wrap_fork_inherited(
    data_ptr: int,
    nbytes: int,
    owner_instance_id: bytes,
    buffer_id: int,
    owner_worker_path: str = ...,
    generation: int = ...,
    access: AccessMode = ...,
    backend_kind: BackendKind = ...,
) -> Buffer: ...

__all__ = [
    "AccessMode",
    "BackendKind",
    "Buffer",
    "Tensor",
    "mint_owner_instance_id",
    "wrap_fork_inherited",
]

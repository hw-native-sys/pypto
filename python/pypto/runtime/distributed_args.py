# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Encode L3 argument pools and resolve external windows inside the active run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .comm_domain import CommDomain, DomainTensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from simpler.comm_domain import CommDomainRef
    from simpler.task_interface import TaskArgs

    from pypto.ir.param_info import ParamInfo

    from .distributed_runner import DistributedWorker


def _uint64(value: int) -> int:
    if type(value) is not int or not 0 <= value < 1 << 64:
        raise ValueError("expected an unsigned 64-bit domain reference field")
    return value


def encode_domain_ref(ref: CommDomainRef) -> tuple[int, int, int]:
    if len(ref.owner_instance_id) != 8:
        raise ValueError("expected an eight-byte domain owner")
    return int.from_bytes(ref.owner_instance_id, "little"), _uint64(ref.domain_id), _uint64(ref.generation)


def decode_domain_ref(args: TaskArgs, offset: int) -> CommDomainRef:
    from simpler.comm_domain import CommDomainRef  # noqa: PLC0415

    return CommDomainRef(
        args.scalar(offset).to_bytes(8, "little"), args.scalar(offset + 1), args.scalar(offset + 2)
    )


WindowRequirement = tuple[bool, tuple[int, ...], str | None]


def entry_window_requirements(entry: Any, params: Sequence[ParamInfo]) -> dict[int, WindowRequirement]:
    """Check the generated entry's window indices against its public parameter metadata."""
    requirements: dict[int, WindowRequirement] = cast(dict[str, Any], vars(entry)).get(
        "_window_requirements", {}
    )
    if set(requirements) != {i for i, info in enumerate(params) if info.is_distributed}:
        raise ValueError("External window parameters disagree with the generated HOST entry")
    return requirements


def validate_window_use(
    owner: CommDomain, name: str, members: Sequence[int], requirement: WindowRequirement, world_size: int
) -> None:
    all_workers, workers, engine = requirement
    for rank in range(world_size) if all_workers else workers:
        if rank not in members:
            raise ValueError(f"window {name!r}: target worker {rank} is not a domain member")
    if engine == "mapped":
        if "urma" in owner._engines or "rdma" in owner._engines:
            raise ValueError(f"window {name!r} requires mapped peers")
        if owner._memory_source == "host" and owner._worker._compiled.platform == "a2a3":
            engine = "sdma"
    if engine in ("urma", "sdma") and engine not in owner._engines:
        raise ValueError(f"window {name!r} requires {engine} workspace")


def pack_window_args(
    params: Sequence[ParamInfo],
    args: Sequence[Any],
    worker: DistributedWorker,
    requirements: dict[int, WindowRequirement],
) -> tuple[TaskArgs, tuple[CommDomain, ...]]:
    """Pack scalar values and the context suffix for the existing HOST callback argument."""
    from simpler.task_interface import TaskArgs  # noqa: PLC0415

    from pypto.ir.param_info import _DATATYPE_TO_CTYPE  # noqa: PLC0415

    entry_args = TaskArgs()
    contexts: list[int] = []
    owners: dict[CommDomainRef, CommDomain] = {}
    for i, (info, value) in enumerate(zip(params, args, strict=True)):
        if info.is_distributed:
            if not isinstance(value, DomainTensor):
                raise TypeError(
                    f"parameter {info.name!r} requires DomainTensor, received {type(value).__name__}"
                )
            owner = value._owner
            owner._require_live()
            if owner._worker is not worker or worker._external_domains.get(owner.ref) is not owner:
                raise ValueError(f"parameter {info.name!r} belongs to a different worker")
            if (
                type(value.buffer_index) is not int
                or not 0 <= value.buffer_index < len(owner.buffers)
                or owner._views[owner.buffers[value.buffer_index].name] is not value
            ):
                raise ValueError(f"parameter {info.name!r} has an invalid buffer_index or owner")
            if (value.shape, value.dtype, value.layout) != (info.shape, info.dtype, info.layout):
                raise ValueError(f"parameter {info.name!r} requires {info.shape} {info.dtype} {info.layout}")
            validate_window_use(owner, info.name, owner.members, requirements[i], len(worker.dc.device_ids))
            entry_args.add_scalar(value.buffer_index)
            contexts.extend(encode_domain_ref(owner.ref))
            owners[owner.ref] = owner
        elif isinstance(value, DomainTensor):
            raise TypeError(f"parameter {info.name!r} does not accept a DomainTensor")
        elif info.shape is None:
            ctype = _DATATYPE_TO_CTYPE.get(str(info.dtype))
            if ctype is not None and isinstance(value, (int, float)):
                value = ctype(value)
            entry_args.add_scalar(value)
    for scalar in contexts:
        entry_args.add_scalar(scalar)
    return entry_args, tuple(owners.values())

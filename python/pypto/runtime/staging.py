# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Request-owned ordinary CPU I/O backed by independent Device domain storage."""

from typing import Any

import torch


class _StagedTensor:
    def __init__(self, invocation: Any, host: torch.Tensor) -> None:
        self.invocation, self.host = invocation, host
        self.shape, self.dtype = tuple(host.shape), host.dtype

    def __getitem__(self, index: Any) -> Any:
        view = self.host[index]
        if view.ndim == 0:
            raise ValueError("staged CHIP arguments must retain tensor dimensions")
        return _StagedTensor(self.invocation, view)

    def reshape(self, *shape: Any) -> Any:
        view = self.host.view(*shape)
        return _StagedTensor(self.invocation, view)

    view = reshape

    def _pypto_chip_view(self, rank: int, tag: Any) -> Any:
        from simpler.task_interface import (  # noqa: PLC0415 -- optional runtime or import-cycle boundary
            TensorArgType,  # noqa: PLC0415 -- avoid importing the optional runtime during compilation
        )
        from simpler_setup.torch_interop import (  # noqa: PLC0415 -- optional runtime or import-cycle boundary
            torch_dtype_to_datatype,  # noqa: PLC0415 -- avoid importing the optional runtime during compilation
        )

        if type(rank) is not int or not 0 <= rank < self.invocation.worker.world_size:
            raise ValueError("staged tensor requires an explicit logical target rank")
        if self.host.numel() == 0:
            raise ValueError("staged CHIP arguments must be nonempty")
        storage = self.host.untyped_storage()
        name = self.invocation.names[storage.data_ptr()]
        offset = self.host.storage_offset() * self.host.element_size()
        span = (
            1 + sum((d - 1) * s for d, s in zip(self.host.shape, self.host.stride(), strict=True))
        ) * self.host.element_size()
        write = tag != TensorArgType.INPUT
        read = tag in (TensorArgType.INPUT, TensorArgType.INOUT, TensorArgType.OUTPUT_EXISTING)
        access = (rank, name, offset, offset + span, read, write)
        for previous in self.invocation.accesses:
            other, slot, start, end, previous_read, previous_write = previous
            if other != rank and slot == name and start < offset + span and offset < end:
                if (write and (previous_read or previous_write)) or (read and previous_write):
                    raise ValueError("cross-rank ordinary I/O aliases require explicit device communication")
        self.invocation.accesses.append(access)
        member = self.invocation.domain.member(rank)
        return member.buffers[name].tensor(
            shapes=tuple(self.host.shape),
            strides=tuple(self.host.stride()),
            dtype=torch_dtype_to_datatype(self.dtype),
            byte_offset=offset,
        )


class _StagedInvocation:  # pyright: ignore[reportUnusedClass] -- imported by distributed_runner
    """Replicate CPU storage per rank, preserving aliases and publishing written spans.

    This path is synchronous. Large reusable weights should use resident shards;
    ordinary storage is uploaded once per rank and invocation, not per CHIP call.
    """

    def __init__(self, worker: Any, args: tuple[Any, ...]) -> None:
        from simpler.task_interface import (  # noqa: PLC0415 -- optional runtime or import-cycle boundary
            CommBufferSpec,  # noqa: PLC0415 -- avoid importing the optional runtime during compilation
        )

        self.worker = worker
        self.storages: dict[int, torch.Tensor] = {}
        self.names: dict[int, str] = {}
        self.accesses: list[tuple[int, str, int, int, bool, bool]] = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if arg.device.type != "cpu" or not arg.is_contiguous() or arg.numel() == 0:
                    raise ValueError("automatic staging requires nonempty contiguous CPU tensors")
                storage = arg.untyped_storage()
                if storage.data_ptr() not in self.names:
                    self.names[storage.data_ptr()] = f"storage{len(self.names)}"
                    raw = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
                    self.storages[storage.data_ptr()] = raw
        buffers = [
            CommBufferSpec(self.names[key], "opaque", raw.numel(), (raw.numel() + 63) // 64 * 64)
            for key, raw in self.storages.items()
        ]
        with worker._dispatch_submit_mu:
            counter = worker.__dict__.get("_staging_counter", 0)
            worker._staging_counter = counter + 1
        self.domain = worker.allocate_domain(
            f"pypto:staging:{counter}", window_size=sum(b.nbytes for b in buffers), buffers=buffers
        )
        self.args = tuple(_StagedTensor(self, arg) if isinstance(arg, torch.Tensor) else arg for arg in args)

    def upload(self) -> None:
        for rank in range(self.worker.world_size):
            for key, raw in self.storages.items():
                self.worker.copy_domain_to(self.domain, rank, self.names[key], raw)

    def publish(self) -> None:
        # Read every output first. A failed download must not partially publish
        # one rank's results into caller storage.
        downloaded: list[tuple[torch.Tensor, int, int, torch.Tensor]] = []
        by_name = {self.names[key]: raw for key, raw in self.storages.items()}
        spans: dict[tuple[int, str], list[tuple[int, int]]] = {}
        for rank, name, start, end, _read, write in self.accesses:
            if write:
                spans.setdefault((rank, name), []).append((start, end))
        for (rank, name), ranges in spans.items():
            merged: list[tuple[int, int]] = []
            for start, end in sorted(ranges):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            for start, end in merged:
                raw = torch.empty(end - start, dtype=torch.uint8)
                self.worker.copy_domain_from(self.domain, rank, name, raw, byte_offset=start)
                downloaded.append((by_name[name], start, end, raw))
        for target, start, end, raw in downloaded:
            target[start:end].copy_(raw)

    def release(self) -> None:
        self.domain.release()
        self.worker._owned_domains.pop(self.domain.name, None)

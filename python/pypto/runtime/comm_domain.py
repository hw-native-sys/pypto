# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Worker-owned communication domains and their logical, named windows."""

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, BinaryIO

import torch

from pypto.pypto_core import DataType

from .window_sources import make_window_initializer, normalize_window_sources

if TYPE_CHECKING:
    from simpler.comm_domain import CommDomainRef

    from .distributed_runner import DistributedWorker

_WINDOW_DTYPES = (
    DataType.FP32,
    DataType.FP16,
    DataType.BF16,
    DataType.INT8,
    DataType.INT16,
    DataType.INT32,
    DataType.INT64,
    DataType.UINT8,
    DataType.UINT16,
    DataType.UINT32,
    DataType.UINT64,
)


@dataclass(frozen=True)
class DomainBufferSpec:
    """Logical shape and element type of one dense communication buffer."""

    name: str
    shape: tuple[int, ...]
    dtype: DataType
    layout: str = "ND"

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))


@dataclass(frozen=True)
class _Slot:
    spec: DomainBufferSpec
    offset: int
    logical: int
    physical: int


def build_domain_layout(
    buffers: Sequence[object], max_size: int = sys.maxsize
) -> tuple[tuple[_Slot, ...], int]:
    """Compute each padded allocation while preserving its exact logical extent."""
    slots: list[_Slot] = []
    names: set[str] = set()
    offset = 0
    for spec in buffers:
        if not isinstance(spec, DomainBufferSpec):
            raise TypeError("buffers must contain DomainBufferSpec objects")
        if type(spec.name) is not str or not spec.name or spec.name in names:
            raise ValueError("domain buffer names must be nonempty and unique")
        names.add(spec.name)
        if spec.layout != "ND" or spec.dtype not in _WINDOW_DTYPES:
            raise ValueError(f"buffer {spec.name!r} requires a supported byte-addressable ND type")
        if not spec.shape or any(type(d) is not int or d <= 0 for d in spec.shape):
            raise ValueError(f"buffer {spec.name!r} requires positive static integer dimensions")
        logical = spec.dtype.get_byte()
        for dimension in reversed(spec.shape):
            if dimension >= 1 << 32 or logical // spec.dtype.get_byte() >= 1 << 32:
                raise ValueError(f"buffer {spec.name!r} shape and stride must fit uint32")
            logical *= dimension
            if logical > max_size:
                raise ValueError("domain buffer exceeds size limit")
        physical = (logical + 63) // 64 * 64
        if offset + physical > max_size:
            raise ValueError("aligned domain window exceeds size limit")
        slots.append(_Slot(spec, offset, logical, physical))
        offset += physical
    if not slots:
        raise ValueError("a communication domain requires at least one buffer")
    return tuple(slots), offset


@dataclass(frozen=True, init=False, eq=False)
class DomainTensor:
    """Logical window view issued by CommDomain.tensor()."""

    _owner: "CommDomain"
    buffer_index: int

    def __init__(self) -> None:
        raise TypeError("obtain a DomainTensor with domain.tensor(name)")

    @classmethod
    def _from_slot(cls, owner: "CommDomain", index: int) -> "DomainTensor":
        view = object.__new__(cls)
        object.__setattr__(view, "_owner", owner)
        object.__setattr__(view, "buffer_index", index)
        return view

    @property
    def domain_ref(self) -> "CommDomainRef":
        return self._owner.ref

    @property
    def shape(self) -> tuple[int, ...]:
        return self._owner.buffers[self.buffer_index].shape

    @property
    def dtype(self) -> DataType:
        return self._owner.buffers[self.buffer_index].dtype

    @property
    def layout(self) -> str:
        return self._owner.buffers[self.buffer_index].layout


class DomainState(Enum):
    CREATING = "creating"
    LIVE = "live"
    CLOSING = "closing"
    INVALID = "invalid"
    CLEANUP_FAILED = "cleanup_failed"
    RELEASED = "released"


class CommDomain:
    """Own a retained native domain and its per-member ordering buffers."""

    def __init__(
        self,
        worker: "DistributedWorker",
        members: tuple[int, ...],
        slots: tuple[_Slot, ...],
        memory_source: str,
        engines: tuple[str, ...],
    ) -> None:
        self._worker = worker
        self._members = members
        self._buffers = tuple(slot.spec for slot in slots)
        self._views = {s.name: DomainTensor._from_slot(self, i) for i, s in enumerate(self._buffers)}
        self._memory_source = memory_source
        self._engines = engines
        self._tokens: dict[int, Any] = {}
        self._native: Any = None
        self._creation_run: Any = None
        self._ref: CommDomainRef | None = None
        self._state = DomainState.CREATING

    @property
    def ref(self) -> "CommDomainRef":
        if self._ref is None:
            raise RuntimeError("domain has no published reference")
        return self._ref

    @property
    def members(self) -> tuple[int, ...]:
        return self._members

    @property
    def buffers(self) -> tuple[DomainBufferSpec, ...]:
        return self._buffers

    def tensor(self, name: str) -> DomainTensor:
        self._require_live()
        return self._views[name]

    def _require_live(self) -> None:
        if self._state is not DomainState.LIVE:
            raise RuntimeError(f"communication domain is {self._state.value}")

    def release(self) -> None:
        self._worker._release_external_domain(self)

    def __enter__(self) -> "CommDomain":
        self._require_live()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.release()

    def _free(self) -> None:
        self._state = DomainState.CLOSING
        try:
            if self._creation_run is not None:
                # An interrupted submit can have accepted the callback before the owner
                # received the handle. Fence it before reclaiming any transaction resources.
                try:
                    self._creation_run.result()
                except Exception:
                    if (
                        not self._creation_run.done
                        or getattr(self._creation_run, "_finalization_abandoned", False) is True
                    ):
                        raise
                self._creation_run = None
            if self._native is not None and not self._native.freed:
                self._native.release()
                if not self._native.freed:
                    raise RuntimeError("communication domain release has not completed")
            for rank, token in list(self._tokens.items()):
                self._worker._w.release_buffer(token)
                del self._tokens[rank]
        except BaseException:
            self._state = DomainState.CLEANUP_FAILED
            raise
        self._state = DomainState.RELEASED

    def _after_worker_close(self) -> None:
        # Simpler's whole-tree close owns abandoned runs and remaining buffers.
        # A successful control close is the final fence for transaction debt.
        if self._native is not None and not self._native.freed:
            raise RuntimeError("worker closed without confirming external domain reclamation")
        self._creation_run = None
        self._tokens.clear()
        self._state = DomainState.RELEASED


def allocate_external_domain(
    worker: "DistributedWorker",
    members: Sequence[int],
    buffers: Sequence[DomainBufferSpec],
    memory_source: str,
    workspace_engines: Sequence[str],
    window_sources: Mapping[str, Sequence[BinaryIO | torch.Tensor]] | None,
) -> CommDomain:
    from simpler.comm_domain import normalize_workspace_engines  # noqa: PLC0415
    from simpler.task_interface import MAX_TENSOR_DIMS, AddressSpace, CommBufferSpec  # noqa: PLC0415

    ordered = tuple(members)
    if (
        not ordered
        or len(ordered) > 64
        or any(type(m) is not int or not 0 <= m < len(worker.dc.device_ids) for m in ordered)
        or len(set(ordered)) != len(ordered)
    ):
        raise ValueError("members must be distinct local chip worker indices in range")
    slots, size = build_domain_layout(buffers)
    if any(len(slot.spec.shape) > MAX_TENSOR_DIMS for slot in slots):
        raise ValueError(f"domain buffer rank must be at most {MAX_TENSOR_DIMS}")
    if isinstance(workspace_engines, Mapping):
        raise TypeError("workspace_engines must be a sequence of engine names")
    engines = normalize_workspace_engines(workspace_engines)
    initializer = None
    if memory_source == "host":
        sources = normalize_window_sources(window_sources)
        if sources.keys() != {slot.spec.name for slot in slots}:
            raise ValueError("Host window_sources must cover exactly the declared buffers")
        initializer = make_window_initializer(
            sources, [(s.spec.name, s.spec.name, s.logical) for s in slots], len(ordered)
        )
    elif memory_source != "device" or window_sources is not None:
        raise ValueError("memory_source must be host or device; only Host windows accept window_sources")
    specs = [
        CommBufferSpec(name=s.spec.name, dtype="opaque", count=s.logical, nbytes=s.physical) for s in slots
    ]
    domain = CommDomain(worker, ordered, slots, memory_source, engines)
    name = f"external:{worker._next_external_domain}"
    worker._next_external_domain += 1

    def create(orch: Any, _args: Any, _config: Any) -> None:
        domain._native = orch.allocate_domain(
            name=name,
            members=ordered,
            buffers=specs,
            window_size=size,
            retain_after_run=True,
            memory_source=AddressSpace.HOST if memory_source == "host" else AddressSpace.DEVICE,
            workspace_engines=engines,
            initializer=initializer,
        )
        domain._ref = domain._native.ref

    try:
        for member in ordered:
            domain._tokens[member] = worker._w.create_buffer(64)
        before = worker._accepted_native_handles()
        try:
            domain._creation_run = worker._w.submit(create)
        except BaseException:
            domain._creation_run = worker._recover_accepted_native_handle(before)
            raise
        domain._creation_run.result()
        domain._creation_run = None
        worker._external_domains[domain.ref] = domain
        domain._state = DomainState.LIVE
        return domain
    except BaseException:
        if domain._ref is not None:
            worker._external_domains.pop(domain._ref, None)
        try:
            domain._free()
        except BaseException:
            worker._external_cleanup.add(domain)
            raise
        raise

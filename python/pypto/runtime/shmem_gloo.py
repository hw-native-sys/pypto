# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Gloo + torch_npu symmetric-memory windows for pypto comm ops.

The host process group is Gloo (CPU). Data windows come from
``torch.distributed._symmetric_memory`` (NPUSHMEM), not from
``orch.allocate_domain`` / HCCL Fabric. The bundled example dispatches directly
through an in-process ChipWorker. :func:`make_shmem_domain_provider` adapts
slots for an in-process HOST integration; it must not be passed across an L3
fork. The generic ``RunConfig.domain_provider`` hook does not make SHMEM
addresses valid in another process.

``CommContext`` packing matches ``runtime/src/common/platform_comm/comm_context.h``
(1088 bytes). Device kernels read ``windowsIn[peer]`` for TPUT/TSTORE.
"""

import importlib
import os
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from .device_tensor import DeviceTensor

# Mirrors comm_context.h. Do not reorder without coordinating pto-isa.
COMM_CONTEXT_SIZE = 1088
COMM_CTX_SLOT = "_comm_ctx"
COMM_MAX_RANK_NUM = 64
_OFF_RANK_ID = 0
_OFF_RANK_NUM = 4
_OFF_WIN_SIZE = 8
_OFF_WINDOWS_IN = 16
_OFF_WINDOWS_OUT = 528

# Match comm_layout::kCommBufferAlignmentBytes: separate slots must not share
# a 64-byte scalar cache line, since DCCI writes back the entire line.
COMM_BUFFER_ALIGN = 64


def align_up(nbytes: int, align: int = COMM_BUFFER_ALIGN) -> int:
    """Round *nbytes* up to a multiple of *align*."""
    if nbytes < 0:
        raise ValueError(f"nbytes must be non-negative, got {nbytes}")
    if align <= 0:
        raise ValueError(f"align must be positive, got {align}")
    return ((nbytes + align - 1) // align) * align


def carve_window_layout(
    slot_nbytes: Sequence[int], *, align: int = COMM_BUFFER_ALIGN
) -> tuple[list[int], int]:
    """Return ``(offsets, window_bytes)`` for consecutive aligned slots."""
    offsets: list[int] = []
    cursor = 0
    for n in slot_nbytes:
        if n <= 0:
            raise ValueError(f"slot nbytes must be positive, got {n}")
        offsets.append(cursor)
        cursor += align_up(n, align)
    return offsets, cursor


def pack_comm_context(
    rank: int,
    world_size: int,
    win_size: int,
    window_bases: Sequence[int],
) -> bytes:
    """Pack a host ``CommContext`` blob (little-endian, 1088 bytes).

    The SDMA, URMA and RDMA workspace descriptors at byte 1040 remain zero.
    This context supports peer-window remote loads/stores; operations requiring
    a communication-engine workspace must not use it.
    """
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} out of range for world_size={world_size}")
    if world_size <= 0 or world_size > COMM_MAX_RANK_NUM:
        raise ValueError(f"world_size {world_size} out of range (1..{COMM_MAX_RANK_NUM})")
    if len(window_bases) != world_size:
        raise ValueError(f"expected {world_size} window bases, got {len(window_bases)}")
    if win_size <= 0:
        raise ValueError(f"win_size must be positive, got {win_size}")

    # Zero unused rank slots and all three 16-byte (addr, size) workspaces.
    blob = bytearray(COMM_CONTEXT_SIZE)
    struct.pack_into("<I", blob, _OFF_RANK_ID, int(rank))
    struct.pack_into("<I", blob, _OFF_RANK_NUM, int(world_size))
    struct.pack_into("<Q", blob, _OFF_WIN_SIZE, int(win_size))
    for i, base in enumerate(window_bases):
        struct.pack_into("<Q", blob, _OFF_WINDOWS_IN + i * 8, int(base))
        struct.pack_into("<Q", blob, _OFF_WINDOWS_OUT + i * 8, int(base))
    return bytes(blob)


def unpack_comm_context(blob: bytes) -> dict[str, Any]:
    """Decode a packed ``CommContext`` for tests / debug dumps."""
    if len(blob) != COMM_CONTEXT_SIZE:
        raise ValueError(f"CommContext blob must be {COMM_CONTEXT_SIZE} bytes, got {len(blob)}")
    rank = struct.unpack_from("<I", blob, _OFF_RANK_ID)[0]
    world = struct.unpack_from("<I", blob, _OFF_RANK_NUM)[0]
    win_size = struct.unpack_from("<Q", blob, _OFF_WIN_SIZE)[0]
    bases = [struct.unpack_from("<Q", blob, _OFF_WINDOWS_IN + i * 8)[0] for i in range(world)]
    return {"rank": rank, "world_size": world, "win_size": win_size, "window_bases": bases}


@dataclass
class ShmemWindow:
    """One process-lifetime SHMEM allocation plus carved named slices."""

    tensor: Any
    handle: Any
    window_bytes: int
    local_base: int
    peer_bases: list[int]
    offsets: dict[str, int]
    device_ctx_tensor: Any
    device_ctx_ptr: int
    owner_pid: int = field(default_factory=os.getpid)

    def as_device_tensor(self, slot: str, shape: Sequence[int], dtype: Any) -> "DeviceTensor":
        """Wrap a carved SHMEM slice as a worker-resident :class:`DeviceTensor`."""
        if os.getpid() != self.owner_pid:
            raise RuntimeError("SHMEM window cannot be used in a forked process")
        if slot not in self.offsets:
            raise KeyError(f"unknown SHMEM slot {slot!r}; have {sorted(self.offsets)}")
        offset = self.offsets[slot]
        end = min((o for o in self.offsets.values() if o > offset), default=self.window_bytes)
        tensor = DeviceTensor(self.local_base + offset, shape, dtype)
        if tensor.nbytes > end - offset:
            raise ValueError(
                f"Tensor needs {tensor.nbytes} bytes, but SHMEM slot {slot!r} has {end - offset}"
            )
        return tensor


def acquire_gloo_shmem_window(
    *,
    rank: int,
    world_size: int,
    device: str,
    slot_names: Sequence[str],
    slot_nbytes: Sequence[int],
    group_name: str | None = None,
) -> ShmemWindow:
    """``empty`` + ``rendezvous`` one uint8 window and carve named slices.

    Requires an initialized Gloo process group. Does not call
    ``orch.allocate_domain``.
    """
    # The symmetric-memory backend is optional until a live window is requested.
    symm_mem = importlib.import_module("torch.distributed._symmetric_memory")

    if not dist.is_initialized() or dist.get_backend() != "gloo":
        raise RuntimeError("SHMEM acquisition requires an initialized Gloo process group")
    if (rank, world_size) != (dist.get_rank(), dist.get_world_size()):
        raise ValueError("rank/world_size must match the initialized Gloo process group")
    if not 0 <= rank < world_size <= COMM_MAX_RANK_NUM:
        raise ValueError(f"Invalid rank={rank}, world_size={world_size}")
    if len(slot_names) != len(slot_nbytes):
        raise ValueError("slot_names and slot_nbytes length mismatch")
    if len(set(slot_names)) != len(slot_names):
        raise ValueError(f"duplicate slot names: {slot_names}")
    if COMM_CTX_SLOT in slot_names:
        raise ValueError(f"{COMM_CTX_SLOT!r} is reserved for the packed CommContext")

    # CommContext lives in the SHMEM heap so TSTORE/TPUT can load windowsIn[]
    # from the same GVA space as the data slices (not a private torch.empty).
    all_names = (COMM_CTX_SLOT, *slot_names)
    all_nbytes = (COMM_CONTEXT_SIZE, *slot_nbytes)
    offsets_list, window_bytes = carve_window_layout(all_nbytes)
    offsets = dict(zip(all_names, offsets_list, strict=True))

    if group_name is None:
        group = dist.group.WORLD
        if group is None:
            raise RuntimeError("Gloo default process group is unavailable")
        group_name = group.group_name
    if not symm_mem.is_symm_mem_enabled_for_group(group_name):
        symm_mem.enable_symm_mem_for_group(group_name)

    tensor = symm_mem.empty((window_bytes,), dtype=torch.uint8, device=device)
    handle = symm_mem.rendezvous(tensor, group=group_name)
    peer_bases = [int(p) for p in handle.buffer_ptrs]
    if len(peer_bases) != world_size:
        raise RuntimeError(f"rendezvous returned {len(peer_bases)} ptrs, expected world_size={world_size}")
    local_base = peer_bases[rank]
    if local_base != int(tensor.data_ptr()):
        raise RuntimeError("SHMEM rendezvous local pointer does not match the allocated tensor")

    ctx_host = pack_comm_context(rank, world_size, window_bytes, peer_bases)
    ctx_off = offsets[COMM_CTX_SLOT]
    device_ctx_tensor = tensor[ctx_off : ctx_off + COMM_CONTEXT_SIZE]
    device_ctx_tensor.copy_(torch.frombuffer(bytearray(ctx_host), dtype=torch.uint8).to(device))
    getattr(torch, "npu").synchronize()

    return ShmemWindow(
        tensor=tensor,
        handle=handle,
        window_bytes=window_bytes,
        local_base=local_base,
        peer_bases=peer_bases,
        offsets=offsets,
        device_ctx_tensor=device_ctx_tensor,
        device_ctx_ptr=int(device_ctx_tensor.data_ptr()),
    )


def make_shmem_domain_provider(
    window: ShmemWindow,
    *,
    rank: int,
    world_size: int,
    simpler_worker: Any,
    local_chip: int = 0,
) -> Callable[..., Any]:
    """Bind external window slots to a live, in-process simpler L2 worker.

    ``simpler_worker`` must be the initialized direct L2 worker, not an L3
    parent. Public ``Buffer.wrap`` source registrations replace the obsolete
    child-provenance hook. The window owns the allocation; domain release only
    revokes the registrations. Keep each domain alive until its dispatch drains.
    Engine workspaces are unsupported: use peer-window loads/stores only.
    """
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import ChipDomainContext, CommDomainHandle  # noqa: PLC0415

    if os.getpid() != window.owner_pid:
        raise RuntimeError("SHMEM window cannot be used in a forked process")
    if not 0 <= rank < world_size <= COMM_MAX_RANK_NUM:
        raise ValueError(f"Invalid rank={rank}, world_size={world_size}")
    if len(window.peer_bases) != world_size or window.peer_bases[rank] != window.local_base:
        raise ValueError("SHMEM window rank mapping does not match the requested domain")
    if simpler_worker.level != 2:
        raise ValueError("SHMEM requires an in-process L2 worker; L3 children cannot access its mappings")
    location = simpler_worker.device_location
    if window.tensor is not None and window.tensor.device.index != location.device_id:
        raise ValueError("SHMEM allocation and L2 worker must use the same device")
    owner_pid = window.owner_pid

    def provider(**kwargs: Any) -> Any:
        if os.getpid() != owner_pid:
            raise RuntimeError("SHMEM domain provider cannot be used in a forked process")
        if kwargs.get("workspace_engines"):
            raise ValueError("SHMEM domain has no SDMA/URMA/RDMA workspaces")
        if tuple(kwargs.get("workers", (local_chip,))) != (local_chip,):
            raise ValueError(f"SHMEM domain must target only local chip {local_chip}")
        if kwargs.get("window_size", 0) > window.window_bytes:
            raise ValueError("Requested domain window exceeds the SHMEM allocation")
        buffers: dict[str, Any] = {}
        try:
            for spec in kwargs["buffers"]:
                logical = spec.name.split("__ssa_v", 1)[0]
                if logical == COMM_CTX_SLOT or logical not in window.offsets:
                    raise KeyError(f"Unknown SHMEM data slot {logical!r}; have {sorted(window.offsets)}")
                if spec.name in buffers:
                    raise ValueError(f"Duplicate SHMEM buffer name {spec.name!r}")
                if spec.load_from_host or spec.store_to_host:
                    raise ValueError("External SHMEM slots do not support implicit host copies")
                offset = window.offsets[logical]
                end = min((o for o in window.offsets.values() if o > offset), default=window.window_bytes)
                if spec.nbytes <= 0 or spec.nbytes % COMM_BUFFER_ALIGN or offset + spec.nbytes > end:
                    raise ValueError(
                        f"Buffer {spec.name!r} needs {spec.nbytes} bytes; slot capacity is {end - offset}"
                    )
                buffers[spec.name] = Buffer.wrap(
                    address=window.local_base + offset, nbytes=spec.nbytes, location=location
                )
        except BaseException:
            for buffer in buffers.values():
                buffer.close()
            raise

        def release(_handle: Any) -> None:
            # Retain the allocation through the handle, even if the caller drops
            # the provider closure before the final synchronous dispatch drains.
            _ = window
            for buffer in buffers.values():
                buffer.close()

        name = str(kwargs.get("name", "shmem"))
        ctx = ChipDomainContext(
            name=name,
            domain_rank=rank,
            domain_size=world_size,
            device_ctx=window.device_ctx_ptr,
            local_window_base=window.local_base,
            actual_window_size=window.window_bytes,
            buffers=buffers,
        )
        return CommDomainHandle(
            name=name,
            workers=(local_chip,),
            contexts={local_chip: ctx},
            _release_fn=release,
            _domain_size=world_size,
            _domain_ranks={local_chip: rank},
        )

    return provider


def peer_copy_expected(local: Any, peer: Any) -> Any:
    """Golden for a ring-shift / TSTORE / TPUT of *peer* into *local*'s view."""
    return peer.detach().clone()


def allreduce_sum_expected(mine: Any, peer: Any) -> Any:
    """Elementwise sum golden for two-rank allreduce."""
    return mine.to(dtype=mine.dtype) + peer.to(dtype=mine.dtype)

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
through an in-process L2 worker. SHMEM addresses must remain in the process
that owns the symmetric-memory mapping; no HOST/L3 domain provider is exposed.

``CommContext`` packing matches ``runtime/src/common/platform_comm/comm_context.h``
(1104 bytes). Device kernels read ``windowsIn[peer]`` for TPUT/TSTORE.
"""

import importlib
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import torch
import torch.distributed as dist

from pypto.pypto_core.codegen import _shmem_comm_layout

from .device_tensor import DeviceTensor

# The native extension reads these from the same pinned runtime layout as codegen.
(
    COMM_CONTEXT_SIZE,
    COMM_MAX_RANK_NUM,
    _OFF_RANK_ID,
    _OFF_RANK_NUM,
    _OFF_WIN_SIZE,
    _OFF_WINDOWS_IN,
    _OFF_WINDOWS_OUT,
    COMM_BUFFER_ALIGN,
) = _shmem_comm_layout()
COMM_CTX_SLOT = "_comm_ctx"


# NPUSHMEM uses one process-wide heap. Registering allocator group 0 a
# second time or switching its Store midway would cross communication domains.
@dataclass
class _Bootstrap:
    groups: tuple[dist.ProcessGroup, dist.ProcessGroup] | None = None


_bootstrap = _Bootstrap()
_bootstrap_lock = Lock()


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
    """Pack a host ``CommContext`` blob (little-endian, 1104 bytes).

    The SDMA, URMA, RDMA and CCU workspace descriptors at byte 1040 remain zero.
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

    # Zero unused rank slots and all four 16-byte (addr, size) workspaces.
    blob = bytearray(COMM_CONTEXT_SIZE)
    struct.pack_into("<I", blob, _OFF_RANK_ID, int(rank))
    struct.pack_into("<I", blob, _OFF_RANK_NUM, int(world_size))
    struct.pack_into("<Q", blob, _OFF_WIN_SIZE, int(win_size))
    for i, base in enumerate(window_bases):
        struct.pack_into("<Q", blob, _OFF_WINDOWS_IN + i * 8, int(base))
        struct.pack_into("<Q", blob, _OFF_WINDOWS_OUT + i * 8, int(base))
    return bytes(blob)


@dataclass
class ShmemWindow:
    """A process-local SHMEM mapping and its carved slices.

    Graphs and pending submissions retain this owner. Complete device work,
    reset any graphs, drain their tickets, then drop the window and its views.
    The Torch symmetric-memory backend manages the backing allocation lifetime.
    """

    tensor: Any
    handle: Any
    window_bytes: int
    local_base: int
    peer_bases: list[int]
    offsets: dict[str, int]
    device_ctx_tensor: Any
    device_ctx_ptr: int
    owner_pid: int = field(default_factory=os.getpid)

    def _contains_data_tensor(self, tensor: torch.Tensor) -> bool:
        if (
            not tensor.is_contiguous()
            or tensor.untyped_storage()._cdata != self.tensor.untyped_storage()._cdata
        ):
            return False
        # Keep tensor views out of the reserved ABI slot and within this
        # mapping, including views with nonzero storage offsets.
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        first_data = self.local_base + self.offsets[COMM_CTX_SLOT] + align_up(COMM_CONTEXT_SIZE)
        return first_data <= start < end <= self.local_base + self.window_bytes

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

    def as_torch_tensor(self, slot: str, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """Return a Torch view associated with this window for direct JIT calls."""
        view = self.as_device_tensor(slot, shape, dtype)
        offset = self.offsets[slot]
        tensor = self.tensor.narrow(0, offset, view.nbytes).view(dtype).view(tuple(shape))
        # The window owns the mapping, and the view owns the window. The
        # window does not hold the view, so dropping both releases the mapping.
        tensor.__dict__["_pypto_shmem_window"] = self
        return tensor


def _enable_gloo_bootstrap(symm_mem: Any, world: dist.ProcessGroup, group: dist.ProcessGroup) -> None:
    """Provide the fixed NPUSHMEM WORLD bootstrap through vLLM's Gloo Store."""
    if world.group_name != "0":
        raise NotImplementedError(f"NPUSHMEM requires WORLD group '0', got {world.group_name!r}")
    if dist.get_process_group_ranks(group) != dist.get_process_group_ranks(world):
        raise NotImplementedError("NPUSHMEM requires the Gloo group to match WORLD membership and rank order")
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    if (rank, size) != (dist.get_rank(world), dist.get_world_size(world)):
        raise ValueError("The Gloo group and WORLD must use the same rank numbering")

    with _bootstrap_lock:
        if _bootstrap.groups is None:
            if symm_mem.is_symm_mem_enabled_for_group(world.group_name):
                raise RuntimeError("NPUSHMEM WORLD bootstrap was already registered by another caller")
            # TorchNPU allocates the heap using the context stored under "0".
            # The real c10d WORLD survives with its original backend.
            c10d = importlib.import_module("torch.distributed.distributed_c10d")
            store = dist.PrefixStore("pypto-npushmem", c10d._get_process_group_store(group))
            symm_mem._SymmetricMemory.set_group_info(world.group_name, rank, size, store)
            _bootstrap.groups = world, group
        elif _bootstrap.groups[0] is not world or _bootstrap.groups[1] is not group:
            raise RuntimeError("NPUSHMEM is already bound to a different WORLD or Gloo group")
        if not symm_mem.is_symm_mem_enabled_for_group(group.group_name):
            symm_mem.enable_symm_mem_for_group(group.group_name)


def acquire_gloo_shmem_window(
    *,
    rank: int,
    world_size: int,
    device: str,
    slot_names: Sequence[str],
    slot_nbytes: Sequence[int],
    group_name: str | None = None,
    gloo_group: dist.ProcessGroup | None = None,
) -> ShmemWindow:
    """``empty`` + ``rendezvous`` one uint8 window and carve named slices.

    With no explicit group, use Gloo WORLD. A caller such as vLLM Ascend may
    instead supply an existing Gloo group with the same members and rank order
    as its HCCL WORLD. Each WORLD rank must enter allocation in the same order.
    """
    # The symmetric-memory backend is optional until a live window is requested.
    symm_mem = importlib.import_module("torch.distributed._symmetric_memory")

    if not dist.is_initialized():
        raise RuntimeError("SHMEM acquisition requires an initialized Gloo process group")
    world = dist.group.WORLD
    if world is None:
        raise RuntimeError("SHMEM acquisition requires an initialized WORLD process group")
    group = world if gloo_group is None else gloo_group
    if dist.get_backend(group) != "gloo":
        if gloo_group is None:
            raise RuntimeError("SHMEM acquisition requires Gloo WORLD or an explicit full-WORLD Gloo group")
        raise ValueError("gloo_group must use the Gloo backend")
    if group_name is not None and group_name != group.group_name:
        raise NotImplementedError("SHMEM group_name must match the selected full-WORLD Gloo process group")
    if (rank, world_size) != (dist.get_rank(group), dist.get_world_size(group)):
        raise ValueError("rank/world_size must match the selected Gloo process group")
    group_name = group.group_name
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

    if group is not world:
        _enable_gloo_bootstrap(symm_mem, world, group)
    elif not symm_mem.is_symm_mem_enabled_for_group(group_name):
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
    getattr(torch, "npu").synchronize(device_ctx_tensor.device)

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


def _require_shmem_window(tensor: object, parameter: str, device_index: int) -> ShmemWindow:
    """Resolve and validate the public view's backing mapping."""
    window = getattr(tensor, "_pypto_shmem_window", None)
    if not isinstance(tensor, torch.Tensor) or not isinstance(window, ShmemWindow):
        raise TypeError(
            f"Distributed tensor {parameter!r} requires a view from ShmemWindow.as_torch_tensor()"
        )
    if window.owner_pid != os.getpid():
        raise RuntimeError("SHMEM window belongs to another process")
    if window.tensor.device.index != device_index:
        raise ValueError(f"Distributed tensor {parameter!r} requires a SHMEM window on npu:{device_index}")
    if not window._contains_data_tensor(tensor):
        raise ValueError(
            f"Distributed tensor {parameter!r} must be a contiguous data view of its SHMEM window"
        )
    return window

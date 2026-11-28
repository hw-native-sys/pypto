# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Two/four/eight-rank Gloo + torch_npu SHMEM windows driving pypto TSTORE, TPUT, allreduce.

Prepare and launch (requires the optional ``PYPTO_BUILD_TORCH_NPU=ON`` adapter)::

    # Source the same PTOAS, CANN and SHMEM environments for both stages.
    # CPU preparation, before requesting devices.
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python examples/distributed/08_gloo_shmem_collectives.py --prepare-only
    # Launch on the allocated devices with the same artifact directory.
    torchrun --standalone --nproc_per_node=2 \\
        examples/distributed/08_gloo_shmem_collectives.py -p a2a3

Windows are ``symm_mem.empty`` / ``rendezvous``. Chip kernels run in-process
through the existing ``pypto.torch.init()`` KERNEL worker so TSTORE/TPUT see the same SHMEM GVA that
``rendezvous().buffer_ptrs`` published (L3 fork children do not inherit those
mappings). Each Gloo barrier follows explicit device completion.
Prepare CPU artifacts with ``--prepare-only`` before launching torchrun.
The sum uses remote_load + add; no workspace-dependent builtin is launched.
"""

import argparse
import importlib
import os
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto import CacheConfig, configure_cache
from pypto._artifact_contract import ArtifactExecutionMode
from pypto.jit.decorator import JITFunction
from pypto.runtime import CompileOptions
from pypto.runtime.kernel.session import KernelConfig
from pypto.runtime.shmem_gloo import acquire_gloo_shmem_window

dist: Any = importlib.import_module("torch.distributed")

SUPPORTED_RANKS = (2, 4, 8)
SIZE = 1024
TORCH_DTYPE = torch.float16
PASS_TOKEN = "GLOO_SHMEM_COLLECTIVES_PASS"


@pl.jit.incore
def tstore_step(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP16]],
):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)
    local = pl.load(x, [0, 0], [1, SIZE])
    peer = (my_rank + 1) % nranks
    pld.tile.remote_store(local, data, peer=peer, offsets=[0, 0])


@pl.jit
def tstore_chip(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP16]],
):
    tstore_step(x, data)


@pl.jit.incore
def tput_step(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    src: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP16]],
    dst: pl.Out[pld.DistributedTensor[[1, SIZE], pl.FP16]],
):
    ctx = pld.get_comm_ctx(src)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)
    local = pl.load(x, [0, 0], [1, SIZE])
    src = pl.store(local, [0, 0], src)
    peer = (my_rank + 1) % nranks
    pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_)
    return dst


@pl.jit
def tput_chip(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    src: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP16]],
    dst: pl.Out[pld.DistributedTensor[[1, SIZE], pl.FP16]],
):
    dst = tput_step(x, src, dst)
    return dst


@pl.jit.incore
def publish_step(
    inp: pl.Tensor[[1, SIZE], pl.FP16],
    data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP16]],
) -> pld.DistributedTensor[[1, SIZE], pl.FP16]:
    data = pl.store(pl.load(inp, [0, 0], [1, SIZE]), [0, 0], data)
    return data


@pl.jit
def publish_chip(
    inp: pl.Tensor[[1, SIZE], pl.FP16],
    data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP16]],
):
    data = publish_step(inp, data)
    return data


@pl.jit.incore
def consume_step(
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
) -> pl.Tensor[[1, SIZE], pl.FP16]:
    out = pl.store(pl.load(data, [0, 0], [1, SIZE]), [0, 0], out)
    return out


@pl.jit
def consume_chip(
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    out = consume_step(data, out)
    return out


@pl.jit.incore
def allreduce_step(
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
) -> pl.Tensor[[1, SIZE], pl.FP16]:
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)
    summed = pl.cast(pl.load(data, [0, 0], [1, SIZE]), pl.FP32)
    for offset in pl.range(1, nranks):
        peer = (my_rank + offset) % nranks
        remote = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
        summed = pl.add(summed, pl.cast(remote, pl.FP32))
    out = pl.store(pl.cast(summed, pl.FP16, mode="rint"), [0, 0], out)
    return out


@pl.jit
def allreduce_chip(
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    out = allreduce_step(data, out)
    return out


@pl.jit.incore
def local_compute_step(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
) -> pl.Tensor[[1, SIZE], pl.FP16]:
    value = pl.load(x, [0, 0], [1, SIZE])
    out = pl.store(pl.add(value, value), [0, 0], out)
    return out


@pl.jit
def local_compute_chip(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    out = local_compute_step(x, out)
    return out


def _barrier() -> None:
    # Gloo only orders the host ranks. Complete local writes and reads first.
    getattr(torch, "npu").synchronize()
    dist.barrier()


def collective_kernels() -> tuple[tuple[JITFunction, int], ...]:
    """The entry points and tensor arities used by CPU preparation."""
    return (
        (tstore_chip, 2),
        (consume_chip, 2),
        (tput_chip, 3),
        (publish_chip, 2),
        (allreduce_chip, 2),
        (local_compute_chip, 2),
    )


def prepare(artifact_dir: Path) -> None:
    """Populate the ordinary KERNEL JIT cache without initializing a device."""
    configure_cache(CacheConfig(enabled=True, root=artifact_dir))
    config = KernelConfig("a2a3", "tensormap_and_ringbuffer", 0)
    meta = torch.empty((1, SIZE), dtype=TORCH_DTYPE)
    for kernel, count in collective_kernels():
        artifact, _, _ = kernel._resolve_compiled(
            (meta,) * count,
            {"config": CompileOptions()},
            _contract=ArtifactExecutionMode.KERNEL,
            _kernel_config=config,
        )
        artifact.load()


def _log(msg: str) -> None:
    print(f"[rank {os.environ.get('RANK', '?')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--artifacts", type=Path, default=Path("build/gloo-shmem-eager"))
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.artifacts)
        return 0
    platform = args.platform
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world not in SUPPORTED_RANKS:
        raise SystemExit(f"need WORLD_SIZE in {SUPPORTED_RANKS}, got {world}")

    configure_cache(CacheConfig(enabled=True, root=args.artifacts, readonly=True))

    dist.init_process_group(backend="gloo", rank=rank, world_size=world, timeout=timedelta(seconds=60))
    _log(f"backend={dist.get_backend()} group={dist.group.WORLD.group_name!r}")

    gen = torch.Generator()
    gen.manual_seed(1000 + rank)
    x = torch.randn((1, SIZE), dtype=TORCH_DTYPE, generator=gen)
    gathered = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(gathered, x)
    peer_x = gathered[(rank - 1) % world]
    y = torch.zeros((1, SIZE), dtype=TORCH_DTYPE)
    # Register the NPU backend only when executing, so CPU codegen checks can import this module.
    importlib.import_module("torch_npu")

    getattr(torch, "npu").set_device(local_rank)
    device = f"npu:{local_rank}"
    import pypto.torch  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415

    pypto.torch.init(device=local_rank, platform=platform)
    state = get_process_kernel_state()

    window = acquire_gloo_shmem_window(
        rank=rank,
        world_size=world,
        device=device,
        slot_names=("data_buf", "src_buf", "dst_buf"),
        slot_nbytes=(SIZE * 2, SIZE * 2, SIZE * 2),
    )
    _log(
        f"SHMEM window bytes={window.window_bytes} local={hex(window.local_base)} "
        f"peers={[hex(p) for p in window.peer_bases]} ctx={hex(window.device_ctx_ptr)} device={device}"
    )

    try:
        x_device = x.to(device)
        y_device = y.to(device)
        data_dt = window.as_torch_tensor("data_buf", (1, SIZE), TORCH_DTYPE)
        src_dt = window.as_torch_tensor("src_buf", (1, SIZE), TORCH_DTYPE)
        dst_dt = window.as_torch_tensor("dst_buf", (1, SIZE), TORCH_DTYPE)
        computed = torch.empty_like(x_device)
        result = torch.empty_like(y_device)
        local_compute_chip(x_device, computed)
        torch.testing.assert_close(computed.cpu(), x * 2, rtol=0, atol=0)
        x_device = computed
        peer_x = peer_x * 2
        _log("eager local compute before SHMEM OK")

        _log("run TSTORE")
        tstore_chip(x_device, data_dt)
        _barrier()
        consume_chip(data_dt, y_device)
        torch.testing.assert_close(y_device.cpu(), peer_x, rtol=1e-3, atol=1e-3)
        local_compute_chip(y_device, result)
        torch.testing.assert_close(result.cpu(), peer_x * 2, rtol=1e-3, atol=1e-3)
        _log("TSTORE + eager local compute after SHMEM golden OK")
        _barrier()

        y_device.zero_()
        _log("run TPUT")
        tput_chip(x_device, src_dt, dst_dt)
        _barrier()
        consume_chip(dst_dt, y_device)
        torch.testing.assert_close(y_device.cpu(), peer_x, rtol=1e-3, atol=1e-3)
        local_compute_chip(y_device, result)
        torch.testing.assert_close(result.cpu(), peer_x * 2, rtol=1e-3, atol=1e-3)
        _log("TPUT + eager local compute after SHMEM golden OK")
        _barrier()

        for iteration in range(args.iterations):
            # Change every publication so stale reads cannot pass a reuse check.
            current = (x.float() + iteration / 8).to(TORCH_DTYPE)
            x_device.copy_(current)
            expected = (
                torch.stack([(value.float() + iteration / 8).to(TORCH_DTYPE).float() for value in gathered])
                .sum(dim=0)
                .to(TORCH_DTYPE)
            )
            y_device.zero_()
            publish_chip(x_device, data_dt)
            _barrier()  # All rank publications must be visible before reading.
            allreduce_chip(data_dt, y_device)
            torch.testing.assert_close(y_device.cpu(), expected, rtol=1e-3, atol=1e-3)
            _barrier()  # Retire every remote read before any rank overwrites.
            _log(f"allreduce golden OK world={world} iteration={iteration}")

        _log(PASS_TOKEN)
    finally:
        state.close()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        raise SystemExit(1)

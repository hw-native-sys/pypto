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

Launch::

    # Source your SHMEM installation environment first.
    torchrun --standalone --nproc_per_node=2 \\
        examples/distributed/08_gloo_shmem_collectives.py -p a2a3

Windows are ``symm_mem.empty`` / ``rendezvous``. Chip kernels run in-process
via :class:`ChipWorker` so TSTORE/TPUT see the same SHMEM GVA that
``rendezvous().buffer_ptrs`` published (L3 fork children do not inherit those
mappings). Host sync is a Gloo barrier after each synchronous dispatch.
The sum uses remote_load + add; no workspace-dependent builtin is launched.
"""

import argparse
import importlib
import os
import traceback
from typing import Any

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto.ir import CompiledProgram
from pypto.jit.decorator import JITFunction
from pypto.runtime import ChipWorker, RunConfig
from pypto.runtime.device_tensor import DeviceTensor
from pypto.runtime.shmem_gloo import (
    acquire_gloo_shmem_window,
    peer_copy_expected,
)

dist: Any = importlib.import_module("torch.distributed")

SUPPORTED_RANKS = (2, 4, 8)
SIZE = 1024
DTYPE = pl.FP16
TORCH_DTYPE = torch.float16
PASS_TOKEN = "GLOO_SHMEM_COLLECTIVES_PASS"


@pl.jit.incore
def tstore_step(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
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
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
):
    return tstore_step(x, data)


@pl.jit.host
def tstore_host(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    data_buf = pld.import_window_buffer([1, SIZE], dtype=pl.FP16)
    data = pld.window(data_buf, [1, SIZE], dtype=pl.FP16)
    tstore_chip(x, data, device=0)


@pl.jit.host
def consume_host(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    data_buf = pld.import_window_buffer([1, SIZE], dtype=pl.FP16)
    data = pld.window(data_buf, [1, SIZE], dtype=pl.FP16)
    consume_chip(data, y, device=0)


@pl.jit.host
def consume_dst_host(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    dst_buf = pld.import_window_buffer([1, SIZE], dtype=pl.FP16)
    dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP16)
    consume_chip(dst, y, device=0)


@pl.jit.incore
def tput_step(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    src: pld.DistributedTensor[[1, SIZE], pl.FP16],
    dst: pld.DistributedTensor[[1, SIZE], pl.FP16],
):
    ctx = pld.get_comm_ctx(src)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)
    local = pl.load(x, [0, 0], [1, SIZE])
    src = pl.store(local, [0, 0], src)
    peer = (my_rank + 1) % nranks
    pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_)


@pl.jit
def tput_chip(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    src: pld.DistributedTensor[[1, SIZE], pl.FP16],
    dst: pld.DistributedTensor[[1, SIZE], pl.FP16],
):
    return tput_step(x, src, dst)


@pl.jit.host
def tput_host(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    src_buf = pld.import_window_buffer([1, SIZE], dtype=pl.FP16)
    dst_buf = pld.import_window_buffer([1, SIZE], dtype=pl.FP16)
    src = pld.window(src_buf, [1, SIZE], dtype=pl.FP16)
    dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP16)
    tput_chip(x, src, dst, device=0)


@pl.jit.incore
def publish_step(
    inp: pl.Tensor[[1, SIZE], pl.FP16],
    data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP16]],
) -> pld.DistributedTensor[[1, SIZE], pl.FP16]:
    return pl.store(pl.load(inp, [0, 0], [1, SIZE]), [0, 0], data)


@pl.jit
def publish_chip(
    inp: pl.Tensor[[1, SIZE], pl.FP16],
    data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP16]],
):
    return publish_step(inp, data)


@pl.jit.incore
def consume_step(
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
) -> pl.Tensor[[1, SIZE], pl.FP16]:
    return pl.store(pl.load(data, [0, 0], [1, SIZE]), [0, 0], out)


@pl.jit
def consume_chip(
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    return consume_step(data, out)


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
    return pl.store(pl.cast(summed, pl.FP16, mode="rint"), [0, 0], out)


@pl.jit
def allreduce_chip(
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    return allreduce_step(data, out)


def _run_kernel(
    chip: ChipWorker, compiled: CompiledProgram, *values: torch.Tensor | DeviceTensor | int
) -> None:
    """Bind caller-owned device allocations to address-free TaskArgs.

    Tensor and scalar pools follow the lowered chip ABI. The raw context pointer
    is a scalar; every data operand carries a live Buffer registration. The
    synchronous run retires uses before registrations are revoked.
    """
    task_interface = importlib.import_module("pypto.runtime.task_interface")
    buffer_type = importlib.import_module("simpler.buffer").Buffer
    tags = importlib.import_module("simpler.task_interface").TensorArgType
    args = task_interface.TaskArgs()
    buffers: list[Any] = []
    tensors = [v for v in values if not isinstance(v, int)]
    scalars = [v for v in values if isinstance(v, int)]
    infos, _, _ = compiled._get_metadata()
    tensor_infos = [info for info in infos if info.shape is not None]
    directions = {"In": tags.INPUT, "Out": tags.OUTPUT_EXISTING, "InOut": tags.INOUT}
    try:
        for val, info in zip(tensors, tensor_infos, strict=True):
            if isinstance(val, DeviceTensor):
                address, nbytes, shape, dtype = val.data_ptr, val.nbytes, val.shape, val.dtype
            else:
                if val.device.type != "npu" or not val.is_contiguous():
                    raise ValueError("Borrowed execution requires contiguous NPU tensors")
                address, nbytes = val.data_ptr(), val.numel() * val.element_size()
                shape, dtype = tuple(val.shape), val.dtype
            buffer = buffer_type.wrap(address=address, nbytes=nbytes, location=chip._impl.device_location)
            buffers.append(buffer)
            args.add_tensor(
                buffer.tensor(shape, task_interface.torch_dtype_to_datatype(dtype)),
                directions[info.direction.name],
            )
        for val in scalars:
            args.add_scalar(int(val))
        # Drain torch_npu's host emission queue and device work before entering
        # simpler on this Python thread. This example deliberately runs serially.
        getattr(torch, "npu").synchronize()
        stream = int(getattr(torch, "npu").current_stream().npu_stream)
        chip._run_chip(
            compiled.chip_callable,
            args,
            compiled._build_call_config(RunConfig(platform=compiled.platform)),
            caller_stream=stream,
        )
    except BaseException:
        # A failed dispatch can retain native uses. Teardown must retire those
        # uses before registrations may be revoked; a failed close leaves them live.
        chip.close()
        for buffer in buffers:
            buffer.close()
        raise
    else:
        for buffer in buffers:
            buffer.close()


def _log(msg: str) -> None:
    print(f"[rank {os.environ.get('RANK', '?')}] {msg}", flush=True)


def _compile_chip(program: JITFunction, *meta: torch.Tensor, platform: str) -> CompiledProgram:
    compiled = program.compile(*meta, config=RunConfig(platform=platform, save_kernels=True))
    if not isinstance(compiled, CompiledProgram):
        raise TypeError("Path B requires a single-chip compiled program")
    return compiled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    platform = args.platform
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world not in SUPPORTED_RANKS:
        raise SystemExit(f"need WORLD_SIZE in {SUPPORTED_RANKS}, got {world}")

    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    _log(f"backend={dist.get_backend()} group={dist.group.WORLD.group_name!r}")

    gen = torch.Generator()
    gen.manual_seed(1000 + rank)
    x = torch.randn((1, SIZE), dtype=TORCH_DTYPE, generator=gen)
    gathered = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(gathered, x)
    peer_x = gathered[(rank - 1) % world]
    y = torch.zeros((1, SIZE), dtype=TORCH_DTYPE)
    meta = torch.empty((1, SIZE), dtype=TORCH_DTYPE)

    _log("compile chip programs")
    tstore_prog = consume_prog = tput_prog = publish_prog = allreduce_prog = None
    # Serialize rank compilation to respect the shared host's worker budget.
    for compiling_rank in range(world):
        if rank == compiling_rank:
            tstore_prog = _compile_chip(tstore_chip, meta, meta, platform=platform)
            consume_prog = _compile_chip(consume_chip, meta, meta, platform=platform)
            tput_prog = _compile_chip(tput_chip, meta, meta, meta, platform=platform)
            publish_prog = _compile_chip(publish_chip, meta, meta, platform=platform)
            allreduce_prog = _compile_chip(allreduce_chip, meta, meta, platform=platform)
        dist.barrier()
    assert tstore_prog is not None and consume_prog is not None and tput_prog is not None
    assert publish_prog is not None and allreduce_prog is not None
    # Register the NPU backend only when executing, so CPU codegen checks can import this module.
    importlib.import_module("torch_npu")

    getattr(torch, "npu").set_device(local_rank)
    device = f"npu:{local_rank}"
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

    # In-process L2: same ACL context as torch_npu SHMEM. Do not fork L3.
    chip = ChipWorker(config=RunConfig(platform=platform, device_id=local_rank), auto_init=False)
    try:
        # One torch context generation for this process; never reset it during this run.
        chip.init(borrowed_context_generation=1)
        x_device = x.to(device)
        y_device = y.to(device)
        data_dt = window.as_device_tensor("data_buf", (1, SIZE), TORCH_DTYPE)
        src_dt = window.as_device_tensor("src_buf", (1, SIZE), TORCH_DTYPE)
        dst_dt = window.as_device_tensor("dst_buf", (1, SIZE), TORCH_DTYPE)
        ctx = window.device_ctx_ptr

        _log("run TSTORE")
        _run_kernel(chip, tstore_prog, x_device, data_dt, ctx)
        dist.barrier()
        _run_kernel(chip, consume_prog, data_dt, y_device, ctx)
        torch.testing.assert_close(y_device.cpu(), peer_copy_expected(x, peer_x), rtol=1e-3, atol=1e-3)
        _log("TSTORE golden OK")
        dist.barrier()

        y_device.zero_()
        _log("run TPUT")
        _run_kernel(chip, tput_prog, x_device, src_dt, dst_dt, ctx, ctx)
        dist.barrier()
        _run_kernel(chip, consume_prog, dst_dt, y_device, ctx)
        torch.testing.assert_close(y_device.cpu(), peer_copy_expected(x, peer_x), rtol=1e-3, atol=1e-3)
        _log("TPUT golden OK")
        dist.barrier()

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
            _run_kernel(chip, publish_prog, x_device, data_dt, ctx)
            dist.barrier()  # All rank publications must be visible before reading.
            _run_kernel(chip, allreduce_prog, data_dt, y_device, ctx)
            torch.testing.assert_close(y_device.cpu(), expected, rtol=1e-3, atol=1e-3)
            dist.barrier()  # Retire every remote read before any rank overwrites.
            _log(f"allreduce golden OK world={world} iteration={iteration}")

        _log(PASS_TOKEN)
    finally:
        chip.close()
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

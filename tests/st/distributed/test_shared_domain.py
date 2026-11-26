# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent L3 writer, reader and remote GET sharing named Host windows.

The CLI separates card-free fixture/binary preparation from run-only acceptance.
"""

import argparse
import io
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import DataType
from pypto.ir import DistributedCompiledProgram, DistributedConfig
from pypto.runtime import DistributedWorker, DomainBufferSpec, RunConfig, runtime_pin_status
from pypto.runtime._prebuilt import BuildKind, prepare_prebuilt


@pl.jit.incore
def copy_table(
    src: pld.DistributedTensor[[1, 65], pl.UINT8], dest: pl.Out[pl.Tensor[[1, 65], pl.UINT8]]
) -> pl.Tensor[[1, 65], pl.UINT8]:
    return pl.store(pl.load(src, [0, 0], [1, 96], valid_shape=[1, 65]), [0, 0], dest)


@pl.jit
def read_chip(
    src: pld.DistributedTensor[[1, 65], pl.UINT8], dest: pl.Out[pl.Tensor[[1, 65], pl.UINT8]]
) -> pl.Tensor[[1, 65], pl.UINT8]:
    return copy_table(src, dest)


@pl.jit.host
def read_tables(
    a: pld.DistributedTensor[[1, 65], pl.UINT8],
    swap: pl.Scalar[pl.INT32],
    b: pld.DistributedTensor[[1, 65], pl.UINT8],
    out: pl.Out[pl.Tensor[[4, 1, 65], pl.UINT8]],
) -> pl.Tensor[[4, 1, 65], pl.UINT8]:
    for chip in pl.range(pld.world_size()):
        if swap != 0:
            read_chip(b, out[chip], device=chip)
        else:
            read_chip(a, out[chip], device=chip)
    return out


@pl.jit.incore
def store_table(
    src: pl.Tensor[[1, 65], pl.UINT8], dest: pl.InOut[pld.DistributedTensor[[1, 65], pl.UINT8]]
) -> pld.DistributedTensor[[1, 65], pl.UINT8]:
    return pl.store(pl.load(src, [0, 0], [1, 96], valid_shape=[1, 65]), [0, 0], dest)


@pl.jit
def write_chip(
    src: pl.Tensor[[1, 65], pl.UINT8], dest: pl.InOut[pld.DistributedTensor[[1, 65], pl.UINT8]]
) -> pld.DistributedTensor[[1, 65], pl.UINT8]:
    return store_table(src, dest)


@pl.jit.host
def write_table(
    src: pl.Tensor[[4, 1, 65], pl.UINT8], dest: pl.InOut[pld.DistributedTensor[[1, 65], pl.UINT8]]
) -> pld.DistributedTensor[[1, 65], pl.UINT8]:
    for chip in pl.range(pld.world_size()):
        dest = write_chip(src[chip], dest, device=chip)
    return dest


@pl.jit.incore
def copy_pair(
    a: pld.DistributedTensor[[1, 65], pl.UINT8],
    b: pld.DistributedTensor[[1, 65], pl.UINT8],
    out: pl.Out[pl.Tensor[[2, 64], pl.UINT8]],
) -> pl.Tensor[[2, 64], pl.UINT8]:
    out = pl.store(pl.load(a, [0, 0], [1, 64]), [0, 0], out)
    return pl.store(pl.load(b, [0, 0], [1, 64]), [1, 0], out)


@pl.jit
def pair_chip(
    a: pld.DistributedTensor[[1, 65], pl.UINT8],
    b: pld.DistributedTensor[[1, 65], pl.UINT8],
    out: pl.Out[pl.Tensor[[2, 64], pl.UINT8]],
) -> pl.Tensor[[2, 64], pl.UINT8]:
    return copy_pair(a, b, out)


@pl.jit.host
def read_pair(
    a: pld.DistributedTensor[[1, 65], pl.UINT8],
    b: pld.DistributedTensor[[1, 65], pl.UINT8],
    out: pl.Out[pl.Tensor[[4, 2, 64], pl.UINT8]],
) -> pl.Tensor[[4, 2, 64], pl.UINT8]:
    for chip in pl.range(pld.world_size()):
        pair_chip(a, b, out[chip], device=chip)
    return out


@pl.jit.incore
def peer_get(
    src: pld.DistributedTensor[[1, 64], pl.UINT8], dest: pl.InOut[pl.Tensor[[1, 64], pl.UINT8]]
) -> pl.Tensor[[1, 64], pl.UINT8]:
    peer = 1 - pld.rank(pld.get_comm_ctx(src))
    return pld.tensor.get_async(dest, peer, src)


@pl.jit
def get_chip(
    src: pld.DistributedTensor[[1, 64], pl.UINT8], dest: pl.InOut[pl.Tensor[[1, 64], pl.UINT8]]
) -> pl.Tensor[[1, 64], pl.UINT8]:
    return peer_get(src, dest)


@pl.jit.host
def read_peer(
    src: pld.DistributedTensor[[1, 64], pl.UINT8], dest: pl.InOut[pl.Tensor[[2, 1, 64], pl.UINT8]]
) -> pl.Tensor[[2, 1, 64], pl.UINT8]:
    for chip in pl.range(2):
        get_chip(src, dest[chip], device=chip)
    return dest


# Four equal shards of the 183-GiB capacity acceptance table. A row is one cache line.
LARGE_ROWS = 183 * (1 << 30) // (4 * 64)


@pl.jit.incore
def read_boundary(
    src: pld.DistributedTensor[[LARGE_ROWS, 64], pl.UINT8],
    row: pl.Scalar[pl.INT64],
    out: pl.Out[pl.Tensor[[1, 64], pl.UINT8]],
) -> pl.Tensor[[1, 64], pl.UINT8]:
    return pl.store(pl.load(src, [row, 0], [1, 64]), [0, 0], out)


@pl.jit
def boundary_chip(
    src: pld.DistributedTensor[[LARGE_ROWS, 64], pl.UINT8],
    row: pl.Scalar[pl.INT64],
    out: pl.Out[pl.Tensor[[1, 64], pl.UINT8]],
) -> pl.Tensor[[1, 64], pl.UINT8]:
    return read_boundary(src, row, out)


@pl.jit.host
def first_rows(
    src: pld.DistributedTensor[[LARGE_ROWS, 64], pl.UINT8], out: pl.Out[pl.Tensor[[4, 1, 64], pl.UINT8]]
) -> pl.Tensor[[4, 1, 64], pl.UINT8]:
    for rank in pl.range(4):
        boundary_chip(src, 0, out[rank], device=rank)
    return out


@pl.jit.host
def last_rows(
    src: pld.DistributedTensor[[LARGE_ROWS, 64], pl.UINT8], out: pl.Out[pl.Tensor[[4, 1, 64], pl.UINT8]]
) -> pl.Tensor[[4, 1, 64], pl.UINT8]:
    for rank in pl.range(4):
        boundary_chip(src, LARGE_ROWS - 1, out[rank], device=rank)
    return out


def prepare_large(root: Path, platform: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for entry in (first_rows, last_rows):
        directory = root / entry.__name__
        entry.compile(config=RunConfig(platform=platform, save_kernels_dir=str(directory)))
        prepare_prebuilt(directory, platform, BuildKind.DISTRIBUTED)
    for rank in range(4):
        with (root / f"shard{rank}.bin").open("wb") as stream:
            stream.write(bytes([rank + 3]) * 64)
            stream.seek((LARGE_ROWS - 1) * 64)
            stream.write(bytes([rank + 73]) * 64)


def run_large(root: Path, devices: list[int]) -> None:
    assert len(devices) == 4
    dc = DistributedConfig(device_ids=devices)
    programs = [
        DistributedCompiledProgram.from_dir(root / name, distributed_config=dc)
        for name in ("first_rows", "last_rows")
    ]
    out = torch.zeros((4, 1, 64), dtype=torch.uint8).share_memory_()
    members = tuple(range(4))
    with (
        patch("pypto.runtime.device_runner._compile_and_assemble", side_effect=AssertionError("run-only")),
        DistributedWorker(programs) as worker,
        ExitStack() as stack,
    ):
        sources = [stack.enter_context((root / f"shard{rank}.bin").open("rb")) for rank in members]
        with worker.create_comm_domain(
            members=members,
            memory_source="host",
            workspace_engines=("urma",),
            buffers=[DomainBufferSpec("table", (LARGE_ROWS, 64), DataType.UINT8)],
            window_sources={"table": sources},
        ) as domain:
            table = domain.tensor("table")
            paths = [
                Path("/dev/shm") / domain._native.member(rank).buffers["table"].shm.name for rank in members
            ]

            def allocated():
                return {p.stat().st_ino: p.stat().st_blocks * 512 for p in paths}

            storage = allocated()
            assert len(storage) == 4 and sum(storage.values()) >= 183 * (1 << 30)
            for prog, tag in zip(programs, (3, 73), strict=True):
                worker.submit(prog, table, out).result()
                assert all(torch.all(out[r] == tag + r) for r in members)
                assert allocated() == storage
            print(
                json.dumps(
                    {
                        "devices": devices,
                        "domain_id": domain.ref.domain_id,
                        "allocated_bytes_by_inode": storage,
                        "bytes_per_shard": LARGE_ROWS * 64,
                    }
                ),
                flush=True,
            )
        assert domain._native.freed
        assert all(not path.exists() for path in paths)


_PROGRAMS = (write_table, read_tables, read_pair, read_peer)


def prepare(root, platform, codegen_only=False):
    root.mkdir(exist_ok=True, parents=True)
    for entry in _PROGRAMS:
        if entry is read_peer and platform != "a5":
            continue
        directory = root / entry.__name__
        entry.compile(config=RunConfig(platform=platform, save_kernels_dir=str(directory)))
        if not codegen_only:
            prepare_prebuilt(directory, platform, BuildKind.DISTRIBUTED)
    pattern = torch.arange(4 * 65, dtype=torch.int64).reshape(4, 1, 65)
    torch.save([(pattern * 7 + tag).to(torch.uint8) for tag in (3, 115, 42)], root / "tables.pt")


class CountingSource(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.bytes_read = 0

    def readinto(self, target):
        count = super().readinto(target)
        self.bytes_read += count
        return count


def report_release(case, root, names, devices, domain, sources):
    assert domain._native.freed
    initialized = {name: [stream.bytes_read for stream in streams] for name, streams in sources.items()}
    print(
        json.dumps(
            {
                "case": case,
                "devices": devices,
                "members": domain.members,
                "schemas": {
                    name: json.loads((root / name / "distributed_meta.json").read_text())["schema"]
                    for name in names
                },
                "runtime": runtime_pin_status().installed,
                "domain_id": domain.ref.domain_id,
                "generation": domain.ref.generation,
                "initialized_bytes_per_member": initialized,
                "correct_local_workers": list(range(len(devices))),
                "released": True,
            }
        ),
        flush=True,
    )


def run_prebuilt(root, devices, *, remote=False, persistent=False):
    dc = DistributedConfig(device_ids=devices)
    names = ("read_peer",) if remote else ("write_table", "read_tables", "read_pair")
    programs = [DistributedCompiledProgram.from_dir(root / n, distributed_config=dc) for n in names]
    assert all(p.program is None for p in programs)
    tables = torch.load(root / "tables.pt", weights_only=True)
    n = len(devices)
    count = 64 if remote else 65
    out = torch.zeros((2 if remote else 4, 1, count), dtype=torch.uint8).share_memory_()
    src = tables[2].clone().share_memory_()
    pair_out = torch.zeros((4, 2, 64), dtype=torch.uint8).share_memory_()
    order = tuple(range(n)) if remote else tuple(reversed(range(n)))
    sources = {
        name: [CountingSource(table[r, :, :count].numpy().tobytes()) for r in order]
        for name, table in zip(("a", "b"), tables[:2], strict=True)
    }
    specs = [DomainBufferSpec(name, (1, count), DataType.UINT8) for name in sources]
    with (
        patch(
            "pypto.runtime.device_runner._compile_and_assemble",
            side_effect=AssertionError("compiler used in run-only mode"),
        ),
        DistributedWorker(programs, persistent=persistent) as worker,
    ):
        with worker.create_comm_domain(
            members=order,
            memory_source="host",
            buffers=specs,
            workspace_engines=("urma",) if remote else (),
            window_sources=sources,
        ) as domain:
            a, b = domain.tensor("a"), domain.tensor("b")
            for streams in sources.values():
                for stream in streams:
                    assert stream.bytes_read == count
                    stream.close()
            if remote:
                worker.submit(programs[0], a, out).result()
                assert torch.equal(out, tables[0][:2, :, :64].flip(0))
                domain.release()
                report_release("peer", root, names, devices, domain, sources)
                return
            writer, reader, pair = programs
            worker.submit(pair, a, b, pair_out).result()
            assert torch.equal(pair_out[:n, 0], tables[0][:n, 0, :64])
            assert torch.equal(pair_out[:n, 1], tables[1][:n, 0, :64])
            for first, second, select, expected in [
                (a, b, 0, tables[0]),
                (b, a, 0, tables[1]),
                (a, b, 1, tables[1]),
            ]:
                out[:n].copy_(expected[:n].bitwise_not())
                worker.submit(reader, first, select, second, out).result()
                assert torch.equal(out[:n], expected[:n])
            # Bind two physically different domains to the same reader. All initialization
            # of the Device window happens through the separately compiled writer.
            with worker.create_comm_domain(members=order, buffers=[specs[1]]) as device_domain:
                device_b = device_domain.tensor("b")
                worker.submit(writer, src, device_b)
                worker.submit(reader, a, 1, device_b, out).result()
                assert torch.equal(out[:n], src[:n])
                worker.submit(reader, a, 0, device_b, out).result()
                assert torch.equal(out[:n], tables[0][:n])
                worker.submit(pair, a, device_b, pair_out).result()
                assert torch.equal(pair_out[:n, 0], tables[0][:n, 0, :64])
                assert torch.equal(pair_out[:n, 1], src[:n, 0, :64])
                assert device_b.domain_ref != b.domain_ref
            write = worker.submit(writer, src, b)
            read = worker.submit(reader, b, 1, b, out)
            domain.release()  # Finish all accepted work without a user-side wait.
            read.result()
            write.result()
            assert torch.equal(out[:n], src[:n])
            with pytest.raises(RuntimeError, match="released"):
                worker.submit(reader, b, 0, a, out)
            report_release("persistent" if persistent else "local", root, names, devices, domain, sources)


@pytest.mark.platforms("a5")
def test_shared_domain(test_config, device_ids, tmp_path):
    prepare(tmp_path, "a5", test_config.codegen_only)
    if test_config.codegen_only:
        return
    if len(device_ids) < 2:
        pytest.skip("two devices required")
    run_prebuilt(tmp_path, device_ids[:2])
    run_prebuilt(tmp_path, device_ids[:2], remote=True)
    run_prebuilt(tmp_path, device_ids[:2], persistent=True)


def _run_phase() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "local", "persistent", "peer", "prepare-large", "large"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--platform", default="a5")
    parser.add_argument("--devices")
    options = parser.parse_args()
    if options.phase == "prepare":
        prepare(options.directory, options.platform)
    elif options.phase == "prepare-large":
        prepare_large(options.directory, options.platform)
    else:
        if not options.devices:
            parser.error(f"{options.phase} requires --devices")
        try:
            devices = [int(value) for value in options.devices.split(",")]
        except ValueError:
            parser.error("--devices expects comma-separated integer device IDs")
        if any(device < 0 for device in devices) or len(set(devices)) != len(devices):
            parser.error("--devices expects distinct non-negative device IDs")
        if options.phase == "large":
            if len(devices) != 4:
                parser.error("large requires exactly four devices")
            run_large(options.directory, devices)
        else:
            if options.phase == "peer" and len(devices) != 2:
                parser.error("peer requires exactly two devices")
            if len(devices) > 4:
                parser.error(f"{options.phase} supports up to four devices")
            run_prebuilt(
                options.directory,
                devices,
                remote=options.phase == "peer",
                persistent=options.phase == "persistent",
            )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _run_phase()
    else:
        pytest.main([__file__, "-v"])

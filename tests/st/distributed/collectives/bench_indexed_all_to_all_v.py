# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare indexed and prepacked URMA collectives with clean AIV timings."""

import hashlib
import json
import os
import shutil
import statistics
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig
from pypto.runtime._artifact_runtime import runtime_output_directory
from pypto.runtime._binary_cache import invalidate_binary_context


def build_program(nr: int, capacity: int, width: int, cores: int, indexed: bool, repetitions: int):
    total = nr * capacity
    # Keep baseline control stores off payload cache lines as well as each other.
    control_bytes = ((nr * 4 + 63) // 64) * 64
    signal_bytes = ((nr * cores * 4 + 63) // 64) * 64

    @pl.jit.incore
    def stage_step(
        inp: pl.Tensor[[total, width], pl.FP32],
        cnt: pl.Tensor[[nr, 1], pl.INT32],
        x: pl.InOut[pld.DistributedTensor[[total, width], pl.FP32]],
        counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
    ) -> tuple[
        pld.DistributedTensor[[total, width], pl.FP32],
        pld.DistributedTensor[[nr, 1], pl.INT32],
    ]:
        for i in pl.range(total):
            row = pl.load(inp, [i, 0], [1, width])
            x = pl.store(row, [i, 0], x)
        for d in pl.range(nr):
            pl.write(counts, [d, 0], pl.read(cnt, [d, 0]))
        return x, counts

    @pl.jit.incore
    def consume_step(
        target: pld.DistributedTensor[[total, width], pl.FP32],
        recv: pl.Tensor[[nr, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[total, width], pl.FP32]],
        rc: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
    ) -> tuple[pl.Tensor[[total, width], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
        for i in pl.range(total):
            row = pl.load(target, [i, 0], [1, width])
            out = pl.store(row, [i, 0], out)
        for s in pl.range(nr):
            pl.write(rc, [s, 0], pl.read(recv, [s, 0]))
        return out, rc

    @pl.jit
    def chip_pipeline(
        inp: pl.Tensor[[total, width], pl.FP32],
        ids: pl.Tensor[[nr, capacity], pl.INT32],
        cnt: pl.Tensor[[nr, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[total, width], pl.FP32]],
        rc: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
        x: pl.InOut[pld.DistributedTensor[[total, width], pl.FP32]],
        counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        target: pl.InOut[pld.DistributedTensor[[total, width], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[nr, cores], pl.INT32]],
        recv_window: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
    ) -> tuple[pl.Tensor[[total, width], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
        x, counts = stage_step(inp, cnt, x, counts)
        if indexed:
            recv: pl.Tensor[[nr, 1], pl.INT32] = pl.create_tensor([nr, 1], dtype=pl.INT32)
            for _ in pl.range(repetitions):
                target = pld.tensor.indexed_all_to_all_v(x, target, signal, counts, recv, ids, core_num=cores)
            return consume_step(target, recv, out, rc)
        else:
            for _ in pl.range(repetitions):
                target = pld.tensor.all_to_all_v(x, target, signal, counts, recv_window, core_num=cores)
            return consume_step(target, recv_window, out, rc)

    @pl.jit.host
    def main(
        inputs: pl.Tensor[[nr, total, width], pl.FP32],
        ids: pl.Tensor[[nr, nr, capacity], pl.INT32],
        cnt: pl.Tensor[[nr, nr, 1], pl.INT32],
        outputs: pl.Out[pl.Tensor[[nr, total, width], pl.FP32]],
        received: pl.Out[pl.Tensor[[nr, nr, 1], pl.INT32]],
    ) -> tuple[pl.Tensor[[nr, total, width], pl.FP32], pl.Tensor[[nr, nr, 1], pl.INT32]]:
        xb = pld.alloc_window_buffer(total * width * 4)
        cb = pld.alloc_window_buffer(control_bytes)
        tb = pld.alloc_window_buffer(total * width * 4)
        sb = pld.alloc_window_buffer(signal_bytes)
        rb = pld.alloc_window_buffer(control_bytes)
        for r in pl.range(nr):
            x = pld.window(xb, [total, width], dtype=pl.FP32)
            counts = pld.window(cb, [nr, 1], dtype=pl.INT32)
            target = pld.window(tb, [total, width], dtype=pl.FP32)
            signal = pld.window(sb, [nr, cores], dtype=pl.INT32)
            recv = pld.window(rb, [nr, 1], dtype=pl.INT32)
            chip_pipeline(
                inputs[r], ids[r], cnt[r], outputs[r], received[r], x, counts, target, signal, recv, device=r
            )
        return outputs, received

    return main


def summarize(root: Path, capture: Path, nr: int, cores: int, rounds: int, warmup: int, peer_bytes: int):
    samples = {}
    for rank in range(nr):
        paths = [
            p
            for p in capture.rglob("chip_swimlane_records.json")
            if f"rank{rank}" in p.parts and "samples" not in p.parts
        ]
        assert len(paths) == 1, paths
        path = paths[0]
        records = json.loads(path.read_text())
        deps = json.loads((path.parent / "deps.json").read_text())
        # The only repeated task is the collective; staging and consume each occur once.
        tasks = sorted(deps["tasks"], key=lambda task: int(task["task_id"]))
        assert len(tasks) == warmup + rounds + 2, len(tasks)
        tasks = tasks[1:-1][warmup:]
        freq = records["metadata"]["clock_freq_hz"]
        times = []
        for task in tasks:
            task_id = int(task["task_id"])
            rows = [row for row in records["aicore_tasks"] if row[1] == task_id]
            assert len(rows) == cores, (task, rows)
            assert all(records["metadata"]["core_types"][row[0]] == "aiv" for row in rows)
            times.append((max(row[4] for row in rows) - min(row[3] for row in rows)) * 1e6 / freq)
        samples[f"rank{rank}"] = times
        dest = root / "samples" / f"rank{rank}"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("chip_swimlane_records.json", "deps.json", "func_names.json"):
            if (path.parent / name).exists():
                shutil.copy2(path.parent / name, dest / name)
    means = {rank: statistics.fmean(times) for rank, times in samples.items()}
    # Keep the historical fastest-card statistic, and also report the whole-group bound.
    fastest = min(means.values())
    group = [max(samples[rank][i] for rank in samples) for i in range(rounds)]
    return {
        "rank_mean_us": means,
        "rank_times_us": samples,
        "fastest_mean_us": fastest,
        "slowest_mean_us": max(means.values()),
        "group_mean_us": statistics.fmean(group),
        "historical_remote_send_GB_s": (nr - 1) * peer_bytes / fastest / 1000,
        "group_remote_send_GB_s": (nr - 1) * peer_bytes / statistics.fmean(group) / 1000,
    }


def main():
    compile_only = os.environ.get("BENCH_COMPILE_ONLY") == "1"
    device_text = os.environ.get("TASK_DEVICE", "0,1") if compile_only else os.environ["TASK_DEVICE"]
    devices = [int(x) for x in device_text.split(",")]
    nr = len(devices)
    cores = int(os.environ.get("BENCH_CORE_NUM", str(nr)))
    peer_mib = int(os.environ.get("BENCH_PEER_MIB", "1"))
    width, warmup, rounds = int(os.environ.get("BENCH_WIDTH", "1024")), 5, 20
    remote_only = os.environ.get("BENCH_REMOTE_ONLY") == "1"
    peer_bytes = peer_mib * 1024 * 1024
    capacity = peer_bytes // (width * 4)
    total = nr * capacity
    base = (torch.arange(nr * total * width, dtype=torch.float32) % 4093).reshape(nr, total, width)
    counts = torch.full((nr, nr, 1), capacity, dtype=torch.int32).share_memory_()
    if remote_only:
        for rank in range(nr):
            counts[rank, rank, 0] = 0
    results = []
    for mode in os.environ.get("BENCH_MODES", "baseline,indexed_contiguous,indexed_shuffled").split(","):
        assert mode in ("baseline", "baseline_optimized", "indexed_contiguous", "indexed_shuffled"), mode
        indexed = not mode.startswith("baseline")
        ids = torch.arange(total, dtype=torch.int32).reshape(nr, capacity)
        if mode == "indexed_shuffled":
            ids = (ids * 7 + 13) % total
        ids = ids.expand(nr, nr, capacity).contiguous().share_memory_()
        inputs = base.clone().share_memory_()
        output = torch.empty_like(inputs).share_memory_()
        recv = torch.empty_like(counts).share_memory_()
        args = [inputs, ids, counts, output, recv]
        root = Path(f"build_output/bench_indexed_{nr}card_{cores}core_{peer_mib}mib_{mode}")
        root = root.with_name(root.name + (f"_h{width}" if width != 1024 else ""))
        if remote_only:
            root = root.with_name(root.name + "_remote_only")
        root.mkdir(parents=True, exist_ok=True)
        compiled = build_program(nr, capacity, width, cores, indexed, warmup + rounds).compile(
            *args,
            config=RunConfig(
                platform="a5",
                save_kernels=True,
                save_kernels_dir=str(root / "compiled"),
                distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
            ),
        )
        source_sha256 = None
        if mode == "baseline_optimized":
            source = Path(os.environ["BENCH_BASELINE_SOURCE"]).read_bytes()
            destination = (
                compiled.output_dir
                / "next_levels"
                / "chip_pipeline"
                / "kernels"
                / "aiv"
                / "__builtin_all_to_all_v__fp32.cpp"
            )
            destination.write_bytes(source)
            source_sha256 = hashlib.sha256(source).hexdigest()
        if compile_only:
            print(f"compile PASS {mode}", flush=True)
            continue
        # A reused save directory may still hold binaries built from an older template.
        # Invalidate only this benchmark's generated chip builds before loading them.
        for kernel_config in compiled.output_dir.rglob("kernel_config.py"):
            invalidate_binary_context(kernel_config.parent)
        capture = runtime_output_directory(compiled)
        with compiled.prepare(persistent=True, reset_persistent_windows=False) as worker:
            handle = worker.register(compiled)
            handle(*args, config=RunConfig(enable_chip_swimlane=1, save_kernels=True))
        if indexed:
            selected = torch.stack([inputs[s, ids[s].reshape(-1).long()] for s in range(nr)])
        else:
            selected = inputs
        golden = selected.reshape(nr, nr, capacity, width).transpose(0, 1).reshape_as(output)
        if remote_only:
            for rank in range(nr):
                for src in range(nr):
                    if src != rank:
                        block = slice(src * capacity, (src + 1) * capacity)
                        torch.testing.assert_close(output[rank, block], golden[rank, block], rtol=0, atol=0)
        else:
            torch.testing.assert_close(output, golden, rtol=0, atol=0)
        assert torch.equal(recv, counts.transpose(0, 1))
        result = summarize(root, capture, nr, cores, rounds, warmup, peer_bytes)
        result["kernel_binary_sha256"] = {
            str(path.relative_to(compiled.output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in compiled.output_dir.rglob("__builtin*all_to_all_v__fp32.o")
        }
        result.update(
            mode=mode,
            devices=devices,
            cores=cores,
            bytes_per_peer=peer_bytes,
            width=width,
            sge_bytes=width * 4 if indexed else None,
            sges_per_peer=capacity if indexed else None,
            warmup=warmup,
            rounds=rounds,
            persistent=True,
            reset_persistent_windows=False,
            indexed_recv_counts_local=indexed,
            remote_only=remote_only,
            indexed_self_transport="MTE 8-stage all-AIV pipeline between URMA submit and CQ wait"
            if indexed
            else None,
            local_copy_workers=cores if indexed else None,
            baseline_source_sha256=source_sha256,
            correctness="PASS",
            timing="AIV task span; excludes staging and consume",
        )
        (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print("BENCH_RESULT " + json.dumps(result), flush=True)
    print("BENCH_SUMMARY " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()

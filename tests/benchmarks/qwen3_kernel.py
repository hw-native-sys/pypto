# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Measure Qwen3 KERNEL eager or capture with fixed pre-tokenized requests.

CPU preparation: examples.models.qwen3_tp.prepare. Run this module under
queue-allocated torchrun. Cold initialization and graph recording are excluded;
reported request latency includes prefill, decode, CPU logits and greedy sampling.
"""

import argparse
import cProfile
import importlib
import json
import logging
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from pypto.runtime import ChipWorker
from pypto.runtime.kernel_compiler import KernelCompiler

from examples.models.qwen3_tp import kernels as k
from examples.models.qwen3_tp.runner import Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "capture"), required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    paired = parser.add_mutually_exclusive_group()
    paired.add_argument(
        "--paired-attention", action="store_true", help="Alternate fused and unfused eager requests"
    )
    paired.add_argument(
        "--paired-layer-grouping",
        action="store_true",
        help="Alternate grouped and baseline HBG eager requests",
    )
    parser.add_argument("--unfused-attention", action="store_true")
    parser.add_argument(
        "--profile-host", action="store_true", help="Diagnostic run; timings include profiler overhead"
    )
    args = parser.parse_args()
    if (args.paired_attention or args.paired_layer_grouping) and (
        args.mode != "eager" or args.unfused_attention
    ):
        parser.error("Paired runs require eager mode and no --unfused-attention")
    tokens = json.loads(args.prompt_ids.read_text())
    if not 1 <= len(tokens) <= k.TOK or not 2 <= args.steps <= k.MAX_CACHE - len(tokens) + 1:
        parser.error("Invalid prompt or generation length")
    if args.rounds < 2:
        parser.error("At least two measured rounds are required")
    if args.warmups < 0:
        parser.error("Warmup count must be nonnegative")
    return args


def request_schedule(
    args: argparse.Namespace, grouping_available: bool
) -> list[tuple[int, tuple[bool, bool]]]:
    fused = not args.unfused_attention
    variants = [(fused, grouping_available and fused)]
    if args.paired_attention:
        variants = [(False, False), (True, False)]
    elif args.paired_layer_grouping:
        if not grouping_available:
            raise ValueError("Paired layer grouping needs prepared TP2 HBG eager artifacts")
        variants = [(True, False), (True, True)]
    schedule = [(0, variant) for _ in range(args.warmups) for variant in variants]
    for trial in range(1, args.rounds + 1):
        schedule.extend((trial, variant) for variant in (variants if trial % 2 else reversed(variants)))
    return schedule


def main() -> None:
    args = parse_args()
    tokens = json.loads(args.prompt_ids.read_text())
    importlib.import_module("torch_npu")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(1)
    logging.getLogger("simpler").setLevel(logging.WARNING)
    npu = getattr(torch, "npu")
    npu.set_device(local_rank)
    dist.init_process_group("gloo")
    # Fail rather than hide a build or PROGRAM fallback inside device allocation.

    def forbidden(*unused, **kwargs):
        raise AssertionError("Device benchmark cannot compile or create a PROGRAM worker")

    ChipWorker.init = forbidden
    KernelCompiler.compile_incore = forbidden
    KernelCompiler.compile_orchestration = forbidden
    runner = Runner(args.artifacts, args.weights, rank, local_rank, capture=args.mode == "capture")
    rows, expected = [], None
    try:
        schedule = request_schedule(args, runner.group_eager_layers)
        runner.load()
        print(f"[rank {rank}] Model loaded mode={args.mode}", flush=True)
        reference_logits = None
        for trial, (fused, grouped) in schedule:
            runner.fuse_attention = fused
            runner.group_eager_layers = grouped
            runner.reset()
            npu.synchronize()
            dist.barrier()
            generated, times, logits_history = [], [], []
            profiler = cProfile.Profile() if args.profile_host and trial == 1 else None
            if profiler is not None:
                profiler.enable()
            start = perf_counter()
            for step in range(args.steps):
                step_start = perf_counter()
                logits = runner.forward(tokens if step == 0 else [generated[-1]])
                generated.append(int(logits.argmax()))
                times.append((perf_counter() - step_start) * 1000)
                logits_history.append(logits)
                if trial == 0:
                    print(
                        f"[rank {rank}] Warmup step {step + 1}/{args.steps}: "
                        f"token={generated[-1]}, elapsed_ms={times[-1]:.3f}",
                        flush=True,
                    )
            elapsed = (perf_counter() - start) * 1000
            if profiler is not None:
                profiler.disable()
                suffix = f".rank{rank}.fused{int(fused)}"
                if args.paired_layer_grouping:
                    suffix += f".grouped{int(grouped)}"
                profiler.dump_stats(str(args.output.with_suffix(suffix + ".prof")))
            assert all(torch.isfinite(value).all() for value in logits_history)
            if reference_logits is None:
                reference_logits = logits_history
            for actual, reference in zip(logits_history, reference_logits, strict=True):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            if expected is None:
                expected = generated
            assert generated == expected, "Greedy tokens changed between repetitions"
            if not rows and rank == 0:
                torch.save(
                    {
                        "world_size": k.TP_WORLD,
                        "prompt_ids": tokens,
                        "generated_ids": generated,
                        "logits": logits_history,
                        "traces": [],
                    },
                    args.output.with_suffix(".pt"),
                )
            local = {
                "request_wall_ms": elapsed,
                "prefill_ms": times[0],
                "decode_mean_ms": sum(times[1:]) / (args.steps - 1),
                "step_ms": times,
                "tokens": generated,
            }
            peers = [None] * k.TP_WORLD
            dist.all_gather_object(peers, local)
            assert all(isinstance(peer, dict) and peer["tokens"] == generated for peer in peers)
            rows.append(
                {
                    "trial": trial,
                    "warmup": trial == 0,
                    "fuse_attention": fused,
                    "grouped_layers": grouped,
                    "ranks": peers,
                }
            )
            if rank == 0:
                args.output.write_text(
                    json.dumps(
                        {
                            "backend": f"kernel_{args.mode}",
                            "profile_host": args.profile_host,
                            "runtime": runner.state.require_config().runtime,
                            "ring_task_window": runner.state.require_config().ring_task_window,
                            "ring_heap": runner.state.require_config().ring_heap,
                            "ring_dep_pool": runner.state.require_config().ring_dep_pool,
                            "tp": k.TP_WORLD,
                            "prompt_ids": tokens,
                            "steps": args.steps,
                            "rows": rows,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(json.dumps(rows[-1]), flush=True)
    finally:
        runner.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

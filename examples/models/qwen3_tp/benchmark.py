# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Prepare artifacts on CPU, then measure one warm Qwen workload per allocation."""

import argparse
import ast
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
from time import monotonic_ns, perf_counter
from typing import Any, cast

import torch
from pypto.ir import CompiledProgram
from pypto.runtime import RunConfig

from examples.models.qwen3_tp import kernels as k
from examples.models.qwen3_tp import runner as model_runner


def source_digest() -> str:
    return hashlib.sha256(Path(k.__file__).read_bytes()).hexdigest()


def prepare(directory: Path) -> None:
    """Lower all entries before acquiring any device."""
    directory.mkdir(parents=True, exist_ok=True)
    entries: dict[str, str] = {}
    dtypes = {"BF16": torch.bfloat16, "FP32": torch.float32, "INT32": torch.int32}
    for node in ast.parse(Path(k.__file__).read_text()).body:
        if not isinstance(node, ast.FunctionDef) or not node.name.endswith("_chip"):
            continue
        values: list[torch.Tensor | int] = []
        for arg in node.args.args:
            annotation = cast(ast.Subscript, arg.annotation)
            if cast(ast.Attribute, annotation.value).attr in ("InOut", "Out"):
                annotation = cast(ast.Subscript, annotation.slice)
            if cast(ast.Attribute, annotation.value).attr == "Scalar":
                values.append(1)
            else:
                shape_node, dtype_node = cast(ast.Tuple, annotation.slice).elts
                shape = eval(compile(ast.Expression(shape_node), k.__file__, "eval"), dict(vars(k)))
                values.append(
                    torch.empty(shape, dtype=dtypes[cast(ast.Attribute, dtype_node).attr], device="meta")
                )
        compiled = getattr(k, node.name).compile(
            *values,
            config=RunConfig(platform="a2a3", save_kernels=True, save_kernels_dir=str(directory / node.name)),
        )
        entries[node.name] = str(Path(compiled.output_dir).resolve())
    (directory / "manifest.json").write_text(
        json.dumps({"world_size": k.TP_WORLD, "source_sha256": source_digest(), "entries": entries}, indent=2)
    )


class BenchmarkRunner(model_runner.Runner):
    """Reuse prepared artifacts and time inclusive collective dispatch."""

    collective_seconds: float = 0.0

    def allreduce(self) -> None:
        start = perf_counter()
        super().allreduce()
        self.collective_seconds += perf_counter() - start


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the slowest rank per step; preserve raw samples separately."""
    result: dict[str, Any] = {}
    for phase in ("prefill", "decode"):
        selected = [row for row in rows if row["phase"] == phase]
        times = torch.tensor([max(row["rank_ms"]) for row in selected], dtype=torch.float64)
        result[phase] = {
            "samples": len(selected),
            "median_ms": float(torch.quantile(times, 0.5)),
            "p95_ms": float(torch.quantile(times, 0.95)),
            "mean_ms": float(times.mean()),
            "mean_rank_collective_ms": sum(
                sum(row["rank_collective_ms"]) / len(row["rank_collective_ms"]) for row in selected
            )
            / len(selected),
        }
    result["decode"]["tokens_per_second"] = 1000 / result["decode"]["mean_ms"]
    return result


def configure_trace_output(enabled: bool, output: Path) -> None:
    if enabled:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Give each native async logger a direct file, avoiding shared pipe backpressure.
        trace_path = output.parent / f"native-rank{os.environ['RANK']}.log"
        with trace_path.open("w") as trace_file:
            os.dup2(trace_file.fileno(), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model-dir")
    parser.add_argument("--prompt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    fusion = parser.add_mutually_exclusive_group()
    fusion.add_argument("--fuse-swiglu-down", action="store_true")
    fusion.add_argument("--fuse-ffn", action="store_true")
    parser.add_argument("--cache-dispatch-plan", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--profile-dispatch", action="store_true", help="Record native spans and dispatch costs"
    )
    parser.add_argument("--profile-setup", action="store_true", help="Time dispatch preparation stages")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.artifacts)
        return
    if not args.model_dir or not args.prompt or not args.output:
        parser.error("Measurement requires --model-dir, --prompt and --output")
    if args.rounds < 2 or not 2 <= args.steps <= 96:
        parser.error("Require at least two rounds and 2..96 steps")
    manifest = json.loads((args.artifacts / "manifest.json").read_text())
    if manifest["world_size"] != k.TP_WORLD or manifest["source_sha256"] != source_digest():
        parser.error("Prepared artifacts do not match TP size or kernel source")
    configure_trace_output(args.profile_dispatch, args.output)
    torch.set_num_threads(1)
    logging.getLogger("simpler").setLevel(25 if args.profile_dispatch else logging.WARNING)
    importlib.import_module("torch_npu")
    getattr(torch, "npu").set_device(int(os.environ["LOCAL_RANK"]))
    dist = importlib.import_module("torch.distributed")
    dist.init_process_group("gloo")
    tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True
    )
    tokens = tokenizer.encode(args.prompt)
    if not 1 <= len(tokens) <= k.TOK:
        parser.error("Prompt must have 1..32 tokens")
    rank = dist.get_rank()
    runner = BenchmarkRunner(
        args.model_dir,
        rank,
        int(os.environ["LOCAL_RANK"]),
        cache_dispatch_plan=args.cache_dispatch_plan,
        fuse_swiglu_down=args.fuse_swiglu_down,
        fuse_ffn=args.fuse_ffn,
        artifacts=args.artifacts,
    )
    dispatch: dict[str, dict[str, float]] = {}
    setup: dict[str, float] = {}
    original_dispatch = model_runner.run_kernel

    def timed_dispatch(chip: Any, compiled: CompiledProgram, *values: Any, **kwargs: Any) -> None:
        name = next(name for name, program in runner.programs.items() if program is compiled)
        start = perf_counter()
        if args.profile_setup:
            kwargs["timings"] = setup
        original_dispatch(chip, compiled, *values, **kwargs)
        stats = dispatch.setdefault(name, {"calls": 0, "ms": 0.0})
        stats["calls"] += 1
        stats["ms"] += (perf_counter() - start) * 1000

    model_runner.run_kernel = (
        timed_dispatch if (args.profile_dispatch or args.profile_setup) else original_dispatch
    )
    rows: list[dict[str, Any]] = []
    expected_tokens: list[int] = []
    model_runner.log = lambda message: None
    try:
        runner.load()
        for trial in range(args.rounds + 1):
            runner.reset()
            generated: list[int] = []
            for step in range(args.steps):
                getattr(torch, "npu").synchronize()
                dist.barrier()
                runner.collective_seconds = 0.0
                dispatch = {}
                setup = {}
                start_ns = monotonic_ns()
                start = perf_counter()
                logits = runner.forward(tokens if step == 0 else [generated[-1]])
                elapsed = (perf_counter() - start) * 1000
                end_ns = monotonic_ns()
                collective_ms = runner.collective_seconds * 1000
                assert torch.isfinite(logits).all()
                peers = [torch.empty_like(logits) for _ in range(k.TP_WORLD)]
                dist.all_gather(peers, logits)
                for peer in peers[1:]:
                    torch.testing.assert_close(peers[0], peer, rtol=1e-4, atol=1e-4)
                generated.append(int(logits.argmax()))
                samples: list[Any] = [None] * k.TP_WORLD
                profile = (
                    {
                        "pid": os.getpid(),
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "dispatch": dispatch,
                        "setup_ms": setup,
                    }
                    if args.profile_dispatch or args.profile_setup
                    else None
                )
                dist.all_gather_object(samples, [elapsed, collective_ms, profile])
                if trial:
                    rows.append(
                        {
                            "round": trial,
                            "step": step,
                            "phase": "prefill" if step == 0 else "decode",
                            "rank_ms": [sample[0] for sample in samples],
                            "rank_collective_ms": [sample[1] for sample in samples],
                            "rank_profiles": [sample[2] for sample in samples],
                        }
                    )
            if trial == 0:
                expected_tokens = generated
            else:
                assert generated == expected_tokens, "Warm/measured token mismatch"
            if rank == 0:
                print(f"Completed {'warm-up' if trial == 0 else f'round {trial}'}", flush=True)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "world_size": k.TP_WORLD,
                        "prompt_tokens": len(tokens),
                        "steps": args.steps,
                        "rounds": args.rounds,
                        "warmup_sequences": 1,
                        "profile_dispatch": args.profile_dispatch,
                        "profile_setup": args.profile_setup,
                        "cache_dispatch_plan": args.cache_dispatch_plan,
                        "fuse_swiglu_down": args.fuse_swiglu_down,
                        "fuse_ffn": args.fuse_ffn,
                        "source_sha256": source_digest(),
                        "generated_ids": expected_tokens,
                        "summary": summarize(rows),
                        "rows": rows,
                    },
                    indent=2,
                )
            )
    finally:
        model_runner.run_kernel = original_dispatch
        runner.chip.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure host submission latency for an eight-rank L3 program.

The benchmark intentionally keeps the device program trivial. It measures the
time for :meth:`DistributedWorker.submit` to return separately from the time
spent waiting for the submitted work, making host dispatch regressions visible
without any model or serving dependency.

Run on eight devices::

    python examples/runtime/distributed_dispatch_latency.py --devices 0,1,2,3,4,5,6,7
"""

import argparse
import importlib.util
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

WORLD_SIZE = 8
DIM = 128


def load_rank_add_program(directory: Path, tensor_args: int) -> Any:
    """Generate a model-free program with a configurable chip ABI width.

    PyPTO parses decorated functions from their Python source, so the generated
    module is written into the benchmark's temporary build directory before it
    is imported. Only the first two inputs participate in the add; the remaining
    inputs exist solely to reproduce wide generated TaskArgs construction.
    """
    input_count = tensor_args - 1
    chip_params = ",\n            ".join(
        f"x{index}: pl.Tensor[[DIM, DIM], pl.FP32]" for index in range(input_count)
    )
    host_params = ",\n            ".join(
        f"x{index}: pl.Tensor[[WORLD_SIZE, DIM, DIM], pl.FP32]" for index in range(input_count)
    )
    rank_args = ", ".join(f"x{index}[rank]" for index in range(input_count))
    source = f"""import pypto.language as pl
import pypto.language.distributed as pld

WORLD_SIZE = {WORLD_SIZE}
DIM = {DIM}

@pl.program
class RankAdd:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
            self,
            a: pl.Tensor[[DIM, DIM], pl.FP32],
            b: pl.Tensor[[DIM, DIM], pl.FP32],
            out: pl.Out[pl.Tensor[[DIM, DIM], pl.FP32]]):
        tile_a = pl.load(a, [0, 0], [DIM, DIM])
        tile_b = pl.load(b, [0, 0], [DIM, DIM])
        return pl.store(pl.add(tile_a, tile_b), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
            self,
            {chip_params},
            out: pl.Out[pl.Tensor[[DIM, DIM], pl.FP32]]):
        return self.tile_add(x0, x1, out)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
            self,
            {host_params},
            out: pl.Out[pl.Tensor[[WORLD_SIZE, DIM, DIM], pl.FP32]]):
        for rank in pl.range(pld.world_size()):
            self.chip_orch({rank_args}, out[rank], device=rank)
"""
    module_path = directory / "generated_dispatch_program.py"
    module_path.write_text(source, encoding="utf-8")
    module_name = "_pypto_generated_dispatch_program"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.RankAdd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True, help="Exactly eight comma-separated device IDs")
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument(
        "--tensor-args",
        type=int,
        default=142,
        help="Total chip tensor arguments per rank, including the output",
    )
    return parser.parse_args()


def median_and_range(samples: list[float]) -> str:
    """Format a latency sample set in milliseconds."""
    return f"median={statistics.median(samples):.3f} ms range={min(samples):.3f}-{max(samples):.3f} ms"


def main() -> None:
    """Compile the minimal program and report submit and wait latency."""
    args = parse_args()
    devices = [int(item) for item in args.devices.split(",")]
    if len(devices) != WORLD_SIZE or len(set(devices)) != WORLD_SIZE:
        raise ValueError(f"--devices must contain exactly {WORLD_SIZE} unique IDs, got {devices}")
    if args.rounds <= 0:
        raise ValueError(f"--rounds must be positive, got {args.rounds}")
    if args.tensor_args < 3:
        raise ValueError(f"--tensor-args must be at least 3, got {args.tensor_args}")

    host_source = torch.ones((WORLD_SIZE, DIM, DIM), dtype=torch.float32).share_memory_()
    host_out = torch.zeros_like(host_source).share_memory_()

    with tempfile.TemporaryDirectory(prefix="pypto-dispatch-latency-") as output_dir:
        program = load_rank_add_program(Path(output_dir), args.tensor_args)
        compiled = ir.compile(
            program,
            platform=args.platform,
            output_dir=output_dir,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
        )
        submit_ms: list[float] = []
        wait_ms: list[float] = []
        with compiled.prepare() as worker:
            worker.register(compiled)
            resident_inputs = [worker.alloc_stacked_tensor(host_source) for _ in range(args.tensor_args - 1)]
            try:
                worker.submit(compiled, *resident_inputs, host_out).wait()
                for _ in range(args.rounds):
                    started_ns = time.perf_counter_ns()
                    pending = worker.submit(compiled, *resident_inputs, host_out)
                    submitted_ns = time.perf_counter_ns()
                    pending.wait()
                    finished_ns = time.perf_counter_ns()
                    submit_ms.append((submitted_ns - started_ns) / 1e6)
                    wait_ms.append((finished_ns - submitted_ns) / 1e6)
            finally:
                for tensor in reversed(resident_inputs):
                    worker.free_stacked_tensor(tensor)

    torch.testing.assert_close(host_out, torch.full_like(host_out, 2.0))
    print(f"tensor args per rank: {args.tensor_args}")
    print(f"submit: {median_and_range(submit_ms)}")
    print(f"wait:   {median_and_range(wait_ms)}")


if __name__ == "__main__":
    main()

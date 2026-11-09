# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure scratch reuse at fixed stages and workload on each compiler revision.

Run as ``uv run -m tests.ci.benchmark_software_pipeline --device N --output DIR``.
The default tile sweep is 32, 44 and 56 rows, with a fixed 36960-row workload.
The runtime benchmark excludes compilation and warmup from its raw samples.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import pypto
import torch
from pypto import passes
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig, benchmark

from tests.software_pipeline_cases import make_scratch_pressure


def main() -> int:
    """Check both outputs and retain timings, input and compiler identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tile-rows", type=int, nargs="+", default=[32, 44, 56])
    args = parser.parse_args()
    if args.rounds <= 0 or args.warmup < 0:
        parser.error("--rounds must be positive and --warmup must be nonnegative")
    if any(rows <= 0 or 36960 % rows for rows in args.tile_rows):
        parser.error("--tile-rows must contain positive divisors of the fixed 36960 input rows")

    assert pypto.__file__ is not None
    root = Path(pypto.__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    x = torch.randn((36960, 128), generator=torch.Generator().manual_seed(2026), dtype=torch.float32)
    expected_reduced, expected_processed = x[:512].sum(dim=1, keepdim=True), x + 1.0
    torch.save(x, args.output / "input.pt")
    source_diff = subprocess.check_output(
        ["git", "-C", str(root), "diff", "HEAD", "--", "include", "src", "python"]
    )
    report: dict[str, Any] = {
        "compiler_revision": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "compiler_diff_sha256": hashlib.sha256(source_diff).hexdigest(),
        "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
        "rounds": args.rounds,
        "warmup": args.warmup,
        "device": args.device,
        "software_pipeline": True,
        "cases": [],
    }
    try:
        for rows in args.tile_rows:
            output = args.output / f"rows-{rows}"
            config = RunConfig(
                backend_type=BackendType.Ascend910,
                execution_mode=ExecutionMode.ONBOARD,
                device_id=args.device,
                save_kernels=True,
                save_kernels_dir=str(output),
                dump_passes=True,
            )
            reduced, processed = torch.empty((512, 1)), torch.empty_like(x)
            with passes.PassContext([], enable_buffer_ir=False, enable_software_pipeline=True):
                compiled = make_scratch_pressure(rows).compile(x, reduced, processed, config=config)
            dumps = list(output.rglob("*_after_LowerPipelineToSlots.py"))
            assert len(dumps) == 1, dumps
            assert '"software_pipeline_slots": 2' in dumps[0].read_text(), (
                "scratch loop did not retain its two software-pipeline slots"
            )
            stats = benchmark(
                compiled, [x, reduced, processed], rounds=args.rounds, warmup=args.warmup, config=config
            )
            torch.testing.assert_close(reduced, expected_reduced, rtol=2e-4, atol=2e-4)
            torch.testing.assert_close(processed, expected_processed, rtol=2e-4, atol=2e-4)
            samples = stats.device_wall_us
            assert len(samples) == args.rounds and all(value > 0 for value in samples), samples
            report["cases"].append(
                {
                    "tile_rows": rows,
                    "software_pipeline_slots": 2,
                    "device_wall_us": samples,
                    "effective_us": stats.per_round("effective"),
                    "median_us": statistics.median(samples),
                }
            )
            print(f"rows={rows}: {stats}", flush=True)
    except Exception as error:
        report["error"] = {"tile_rows": rows, "message": str(error)}
        raise
    finally:
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure CPU-only cross-process READY latency for the installed PyPTO.

Run with the interpreter and PYTHONPATH layout to test. Imports, tensor creation
and kernel definition are outside the interval; first identity capture and the
first warmup (including callable restoration) are inside it. No NPU is started.
The first child populates the cache, and every measured child must hit READY.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def worker(root: Path) -> dict:
    import runpy  # noqa: PLC0415
    import time  # noqa: PLC0415
    from dataclasses import asdict  # noqa: PLC0415

    import pypto  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from pypto.jit._toolchain import capture_toolchain  # noqa: PLC0415
    from pypto.runtime import RunConfig  # noqa: PLC0415

    example = Path(__file__).resolve().parents[2] / "examples/beginner/01_hello_world.py"
    kernel = runpy.run_path(str(example))["tile_add"]
    tensor = torch.zeros(128, 128)
    config = RunConfig(platform="a2a3", cache_config=pypto.CacheConfig(enabled=True, root=root))
    start = time.perf_counter_ns()
    identity = capture_toolchain("a2a3", "tensormap_and_ringbuffer")
    captured = time.perf_counter_ns()
    assert identity.usable, identity.failures
    kernel.warmup(tensor, tensor, tensor, config=config)
    end = time.perf_counter_ns()
    return {
        "identity_ms": (captured - start) / 1e6,
        "warmup_ms": (end - captured) / 1e6,
        "total_ms": (end - start) / 1e6,
        "stats": asdict(pypto.cache_stats()),
        "python_package": pypto.__file__,
        "native_module": pypto.pypto_core.__file__,
        "identity": asdict(identity),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.cache_root)))
        return
    if args.runs < 1:
        parser.error("--runs must be positive")
    samples = []
    for index in range(args.runs + 1):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", "--cache-root", str(args.cache_root)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"Benchmark child failed ({result.returncode}):\n{result.stderr}")
        sample = json.loads(result.stdout.splitlines()[-1])
        if index:
            stats = sample["stats"]
            assert stats["ready_hits"] == 1, sample
            assert all(
                stats[key] == 0
                for key in (
                    "misses",
                    "bypasses",
                    "generation_builds",
                    "binary_builds",
                    "invalid_entries",
                    "storage_errors",
                )
            ), sample
            samples.append(sample)
    report = {
        "interpreter": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "policy": os.environ.get("PYPTO_CACHE_IDENTITY", "build"),
        "samples": samples,
        "summary_ms": {
            key: {
                "median": statistics.median(s[key] for s in samples),
                "min": min(s[key] for s in samples),
                "max": max(s[key] for s in samples),
            }
            for key in ("identity_ms", "warmup_ms", "total_ms")
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary_ms"], indent=2))


if __name__ == "__main__":
    main()

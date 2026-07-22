# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare compact and loose standalone NPU kernels with paired ABBA batches."""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_manifest(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "standalone_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"standalone manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("run_mode") != "npu":
        raise ValueError(f"{path} is not a schema-v1 NPU standalone manifest")
    return manifest


def _abi_signature(manifest: dict[str, Any]) -> tuple[Any, ...]:
    parameters = tuple(
        (
            parameter["name"],
            parameter["cpp_type"],
            parameter["kind"],
            parameter.get("elements"),
            parameter.get("value"),
        )
        for parameter in manifest["parameters"]
    )
    mixed = manifest.get("mixed_runner", {})
    mixed_runner = (mixed.get("kind"), mixed.get("generator_sha256"))
    return manifest["kernel"], manifest["aicore_arch"], manifest["block_dim"], mixed_runner, parameters


def _pointer_names(manifest: dict[str, Any]) -> list[str]:
    return [parameter["name"] for parameter in manifest["parameters"] if parameter["kind"] == "pointer"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cases(compact_dir: Path, loose_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Validate ABI, launch metadata, and real inputs shared by two cases."""
    compact = _load_manifest(compact_dir)
    loose = _load_manifest(loose_dir)
    if compact.get("mixed") or loose.get("mixed"):
        expected_runner = "ptoas_validation_group_wrapper"
        runners = {
            compact.get("mixed_runner", {}).get("kind"),
            loose.get("mixed_runner", {}).get("kind"),
        }
        if runners != {expected_runner}:
            raise ValueError("mixed AIC/AIV kernels require the canonical PTOAS group-level runner")
    if _abi_signature(compact) != _abi_signature(loose):
        raise ValueError("compact and loose standalone cases have different ABI or launch metadata")
    pointers = _pointer_names(compact)
    for name in pointers:
        compact_input = compact_dir / f"{name}.bin"
        loose_input = loose_dir / f"{name}.bin"
        if not compact_input.is_file() or not loose_input.is_file():
            raise FileNotFoundError(f"both cases must contain the real input buffer {name}.bin")
        if _sha256(compact_input) != _sha256(loose_input):
            raise ValueError(f"compact and loose inputs differ for ABI buffer {name}.bin")
    return compact, pointers


def _read_samples(path: Path) -> list[float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "sample\telapsed_us":
        raise ValueError(f"unexpected timing header in {path}")
    samples: list[float] = []
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"invalid timing row {line_number} in {path}: {line!r}")
        value = float(fields[1])
        if value <= 0:
            raise ValueError(f"non-positive device duration at row {line_number} in {path}: {value}")
        samples.append(value)
    if not samples:
        raise ValueError(f"timing file contains no samples: {path}")
    return samples


def _run_variant(
    executable: Path,
    case_dir: Path,
    timing_path: Path,
    *,
    device_id: int,
    warmup: int,
    rounds: int,
    timeout: int,
    dump_dir: Path | None = None,
) -> list[float]:
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "ACL_DEVICE_ID": str(device_id),
            "PYPTO_BENCH_WARMUP": str(warmup),
            "PYPTO_BENCH_ROUNDS": str(rounds),
            "PYPTO_BENCH_OUTPUT": str(timing_path),
        }
    )
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        env["PYPTO_BENCH_DUMP_DIR"] = str(dump_dir)
    result = subprocess.run(
        [str(executable.resolve())],
        cwd=case_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"standalone kernel failed with exit code {result.returncode}: {executable}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return _read_samples(timing_path)


def _compare_outputs(
    compact_dump: Path,
    loose_dump: Path,
    names: list[str],
    *,
    expected_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    hashes: dict[str, dict[str, str]] = {}
    for name in names:
        compact_output = compact_dump / f"{name}.bin"
        loose_output = loose_dump / f"{name}.bin"
        if not compact_output.is_file() or not loose_output.is_file():
            raise FileNotFoundError(f"standalone output dump is missing {name}.bin")
        compact_hash = _sha256(compact_output)
        loose_hash = _sha256(loose_output)
        if compact_hash != loose_hash:
            raise ValueError(f"compact and loose standalone outputs differ for ABI buffer {name}.bin")
        record = {"compact": compact_hash, "loose": loose_hash}
        if expected_dir is not None:
            expected_output = expected_dir / f"{name}.bin"
            if not expected_output.is_file():
                raise FileNotFoundError(f"captured expected output is missing {name}.bin")
            expected_hash = _sha256(expected_output)
            if compact_hash != expected_hash:
                raise ValueError(f"standalone output differs from the captured model output for {name}.bin")
            record["captured_expected"] = expected_hash
        hashes[name] = record
    return hashes


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = percentile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(
    compact_samples: list[float],
    loose_samples: list[float],
    quartet_differences: list[float],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize raw samples and bootstrap the paired quartet effect."""
    if not compact_samples or not loose_samples or not quartet_differences:
        raise ValueError("compact, loose, and paired-quartet samples must be non-empty")
    compact_median = statistics.median(compact_samples)
    loose_median = statistics.median(loose_samples)
    difference = loose_median - compact_median
    rng = random.Random(seed)
    bootstrapped = [
        statistics.mean(rng.choice(quartet_differences) for _ in quartet_differences)
        for _ in range(bootstrap_samples)
    ]
    return {
        "compact": {
            "samples": len(compact_samples),
            "median_us": compact_median,
            "p10_us": _percentile(compact_samples, 0.10),
            "p90_us": _percentile(compact_samples, 0.90),
        },
        "loose": {
            "samples": len(loose_samples),
            "median_us": loose_median,
            "p10_us": _percentile(loose_samples, 0.10),
            "p90_us": _percentile(loose_samples, 0.90),
        },
        "loose_minus_compact_us": difference,
        "loose_minus_compact_percent": 100.0 * difference / compact_median,
        "paired_quartet_mean_us": statistics.mean(quartet_differences),
        "paired_bootstrap_95_ci_us": [
            _percentile(bootstrapped, 0.025),
            _percentile(bootstrapped, 0.975),
        ],
    }


def _infer_executable(case_dir: Path, manifest: dict[str, Any]) -> Path:
    return case_dir / "build" / f"{manifest['testcase']}_npu"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare compact and loose standalone NPU kernels")
    parser.add_argument("--compact-case", type=Path, required=True)
    parser.add_argument("--loose-case", type=Path, required=True)
    parser.add_argument("--compact-exe", type=Path)
    parser.add_argument("--loose-exe", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", action="append", default=[], help="ABI buffer to compare; repeatable")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--quartets", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)

    if args.quartets <= 0 or args.warmup <= 0 or args.rounds <= 0 or args.timeout <= 0:
        parser.error("quartets, warmup, rounds, and timeout must all be positive")
    manifest, pointers = validate_cases(args.compact_case, args.loose_case)
    capture = manifest.get("capture", {})
    recommended_outputs = capture.get("recommended_outputs", []) if isinstance(capture, dict) else []
    outputs = args.output or recommended_outputs or pointers
    unknown_outputs = sorted(set(outputs) - set(pointers))
    if unknown_outputs:
        parser.error(f"output names are absent from the kernel ABI: {unknown_outputs}")

    compact_exe = args.compact_exe or _infer_executable(args.compact_case, manifest)
    loose_manifest = _load_manifest(args.loose_case)
    loose_exe = args.loose_exe or _infer_executable(args.loose_case, loose_manifest)
    for executable in (compact_exe, loose_exe):
        if not executable.is_file():
            raise FileNotFoundError(f"standalone NPU executable not found: {executable}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    compact_dump = args.output_root / "correctness" / "compact"
    loose_dump = args.output_root / "correctness" / "loose"
    _run_variant(
        compact_exe,
        args.compact_case,
        args.output_root / "correctness" / "compact.tsv",
        device_id=args.device_id,
        warmup=1,
        rounds=1,
        timeout=args.timeout,
        dump_dir=compact_dump,
    )
    _run_variant(
        loose_exe,
        args.loose_case,
        args.output_root / "correctness" / "loose.tsv",
        device_id=args.device_id,
        warmup=1,
        rounds=1,
        timeout=args.timeout,
        dump_dir=loose_dump,
    )
    compact_expected = args.compact_case / "captured_expected"
    loose_expected = args.loose_case / "captured_expected"
    expected_dir: Path | None = None
    if compact_expected.is_dir() or loose_expected.is_dir():
        if not compact_expected.is_dir() or not loose_expected.is_dir():
            raise ValueError("compact and loose cases do not both carry captured expected outputs")
        for name in outputs:
            if _sha256(compact_expected / f"{name}.bin") != _sha256(loose_expected / f"{name}.bin"):
                raise ValueError(f"compact and loose captured expectations differ for {name}.bin")
        expected_dir = compact_expected
    output_hashes = _compare_outputs(compact_dump, loose_dump, outputs, expected_dir=expected_dir)

    raw: dict[str, list[float]] = {"compact": [], "loose": []}
    quartet_differences: list[float] = []
    sample_rows = ["quartet\tposition\tvariant\tsample\telapsed_us"]
    order = ("compact", "loose", "loose", "compact")
    executables = {"compact": compact_exe, "loose": loose_exe}
    case_dirs = {"compact": args.compact_case, "loose": args.loose_case}
    for quartet in range(args.quartets):
        within: dict[str, list[float]] = {"compact": [], "loose": []}
        for position, variant in enumerate(order):
            timing_path = args.output_root / "runs" / f"q{quartet:02d}_{position}_{variant}.tsv"
            samples = _run_variant(
                executables[variant],
                case_dirs[variant],
                timing_path,
                device_id=args.device_id,
                warmup=args.warmup,
                rounds=args.rounds,
                timeout=args.timeout,
            )
            raw[variant].extend(samples)
            within[variant].extend(samples)
            sample_rows.extend(
                f"{quartet}\t{position}\t{variant}\t{sample}\t{elapsed_us}"
                for sample, elapsed_us in enumerate(samples)
            )
        quartet_differences.append(statistics.median(within["loose"]) - statistics.median(within["compact"]))

    summary = summarize(raw["compact"], raw["loose"], quartet_differences)
    report = {
        "schema_version": 1,
        "kernel": manifest["kernel"],
        "block_dim": manifest["block_dim"],
        "device_id": args.device_id,
        "quartets": args.quartets,
        "warmup_per_process": args.warmup,
        "rounds_per_process": args.rounds,
        "abba_order": list(order),
        "compared_output_hashes": output_hashes,
        "quartet_differences_us": quartet_differences,
        "summary": summary,
    }
    (args.output_root / "samples.tsv").write_text("\n".join(sample_rows) + "\n", encoding="utf-8")
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

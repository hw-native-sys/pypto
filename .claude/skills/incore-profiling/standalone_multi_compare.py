# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare multiple standalone NPU placements with balanced execution orders."""

import argparse
import itertools
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import standalone_compare as pair


def balanced_orders(names: list[str]) -> list[tuple[str, ...]]:
    """Return cyclic orders and their reverses, balanced by position."""
    if len(names) < 2:
        raise ValueError("multi-case comparison requires at least two variants")
    forward = [tuple(names[index:] + names[:index]) for index in range(len(names))]
    reverse_names = list(reversed(names))
    reverse = [tuple(reverse_names[index:] + reverse_names[:index]) for index in range(len(names))]
    return forward + reverse


def _variant_argument(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    if not name.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError(f"variant name must be alphanumeric: {name!r}")
    return name, Path(raw_path)


def _variant_map(values: list[tuple[str, Path]], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, path in values:
        if name in result:
            raise ValueError(f"duplicate {option} variant: {name}")
        result[name] = path
    return result


def summarize_variants(
    samples: dict[str, list[float]],
    block_medians: list[dict[str, float]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize each arm and all paired block-level differences."""
    if not samples or not block_medians:
        raise ValueError("samples and block medians must be non-empty")
    names = list(samples)
    if any(not values for values in samples.values()):
        raise ValueError("every variant must contain timing samples")
    if any(set(block) != set(names) for block in block_medians):
        raise ValueError("every timing block must contain every variant")

    variants = {
        name: {
            "samples": len(values),
            "median_us": statistics.median(values),
            "p10_us": pair._percentile(values, 0.10),
            "p90_us": pair._percentile(values, 0.90),
        }
        for name, values in samples.items()
    }
    comparisons: dict[str, Any] = {}
    for reference, candidate in itertools.combinations(names, 2):
        differences = [block[candidate] - block[reference] for block in block_medians]
        rng = random.Random(f"{seed}:{reference}:{candidate}")
        bootstrapped = [
            statistics.mean(rng.choice(differences) for _ in differences) for _ in range(bootstrap_samples)
        ]
        reference_median = variants[reference]["median_us"]
        candidate_median = variants[candidate]["median_us"]
        delta = candidate_median - reference_median
        comparisons[f"{candidate}_minus_{reference}"] = {
            "reference": reference,
            "candidate": candidate,
            "median_delta_us": delta,
            "median_delta_percent": 100.0 * delta / reference_median,
            "paired_block_mean_us": statistics.mean(differences),
            "paired_bootstrap_95_ci_us": [
                pair._percentile(bootstrapped, 0.025),
                pair._percentile(bootstrapped, 0.975),
            ],
            "block_differences_us": differences,
        }
    return {"variants": variants, "comparisons": comparisons}


def _validate_cases(case_dirs: dict[str, Path]) -> tuple[dict[str, Any], list[str]]:
    names = list(case_dirs)
    baseline_dir = case_dirs[names[0]]
    baseline = pair._load_manifest(baseline_dir)
    pointers = pair._pointer_names(baseline)
    for name in names[1:]:
        pair.validate_cases(baseline_dir, case_dirs[name])
    return baseline, pointers


def _compare_all_outputs(
    dumps: dict[str, Path], outputs: list[str], expected_dir: Path | None
) -> dict[str, dict[str, str]]:
    hashes: dict[str, dict[str, str]] = {}
    for output in outputs:
        output_hashes: dict[str, str] = {}
        for name, dump in dumps.items():
            path = dump / f"{output}.bin"
            if not path.is_file():
                raise FileNotFoundError(f"standalone output dump is missing {path}")
            output_hashes[name] = pair._sha256(path)
        if len(set(output_hashes.values())) != 1:
            raise ValueError(f"standalone outputs differ for ABI buffer {output}.bin")
        if expected_dir is not None:
            expected = expected_dir / f"{output}.bin"
            if not expected.is_file():
                raise FileNotFoundError(f"captured expected output is missing {output}.bin")
            expected_hash = pair._sha256(expected)
            if expected_hash not in set(output_hashes.values()):
                raise ValueError(f"standalone output differs from captured output for {output}.bin")
            output_hashes["captured_expected"] = expected_hash
        hashes[output] = output_hashes
    return hashes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare multiple standalone NPU placements")
    parser.add_argument("--case", action="append", type=_variant_argument, required=True)
    parser.add_argument("--exe", action="append", type=_variant_argument, default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", action="append", default=[], help="ABI output; repeatable")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args(argv)

    if args.warmup <= 0 or args.rounds <= 0 or args.timeout <= 0 or args.bootstrap_samples <= 0:
        parser.error("warmup, rounds, timeout, and bootstrap-samples must be positive")
    case_dirs = _variant_map(args.case, "case")
    if len(case_dirs) < 2:
        parser.error("at least two --case NAME=PATH arguments are required")
    executable_overrides = _variant_map(args.exe, "executable")
    unknown_executables = sorted(set(executable_overrides) - set(case_dirs))
    if unknown_executables:
        parser.error(f"executable variants have no matching case: {unknown_executables}")

    manifest, pointers = _validate_cases(case_dirs)
    capture = manifest.get("capture", {})
    recommended = capture.get("recommended_outputs", []) if isinstance(capture, dict) else []
    outputs = args.output or recommended or pointers
    unknown_outputs = sorted(set(outputs) - set(pointers))
    if unknown_outputs:
        parser.error(f"output names are absent from the kernel ABI: {unknown_outputs}")

    manifests = {name: pair._load_manifest(case) for name, case in case_dirs.items()}
    executables = {
        name: executable_overrides.get(name, pair._infer_executable(case_dirs[name], manifests[name]))
        for name in case_dirs
    }
    for executable in executables.values():
        if not executable.is_file():
            raise FileNotFoundError(f"standalone NPU executable not found: {executable}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    dumps: dict[str, Path] = {}
    for name in case_dirs:
        dump = args.output_root / "correctness" / name
        pair._run_variant(
            executables[name],
            case_dirs[name],
            args.output_root / "correctness" / f"{name}.tsv",
            device_id=args.device_id,
            warmup=1,
            rounds=1,
            timeout=args.timeout,
            dump_dir=dump,
        )
        dumps[name] = dump

    expected_dirs = [case / "captured_expected" for case in case_dirs.values()]
    expected_dir: Path | None = None
    if any(path.is_dir() for path in expected_dirs):
        if not all(path.is_dir() for path in expected_dirs):
            raise ValueError("standalone cases do not all carry captured expected outputs")
        expected_dir = expected_dirs[0]
        for output in outputs:
            expected_hashes = {pair._sha256(path / f"{output}.bin") for path in expected_dirs}
            if len(expected_hashes) != 1:
                raise ValueError(f"captured expectations differ for {output}.bin")
    output_hashes = _compare_all_outputs(dumps, outputs, expected_dir)

    names = list(case_dirs)
    orders = balanced_orders(names)
    raw: dict[str, list[float]] = {name: [] for name in names}
    block_medians: list[dict[str, float]] = []
    sample_rows = ["block\tposition\tvariant\tsample\telapsed_us"]
    for block, order in enumerate(orders):
        medians: dict[str, float] = {}
        for position, name in enumerate(order):
            timing_path = args.output_root / "runs" / f"b{block:02d}_{position}_{name}.tsv"
            samples = pair._run_variant(
                executables[name],
                case_dirs[name],
                timing_path,
                device_id=args.device_id,
                warmup=args.warmup,
                rounds=args.rounds,
                timeout=args.timeout,
            )
            raw[name].extend(samples)
            medians[name] = statistics.median(samples)
            sample_rows.extend(
                f"{block}\t{position}\t{name}\t{sample}\t{elapsed_us}"
                for sample, elapsed_us in enumerate(samples)
            )
        block_medians.append(medians)

    summary = summarize_variants(raw, block_medians, bootstrap_samples=args.bootstrap_samples, seed=0)
    report = {
        "schema_version": 1,
        "kernel": manifest["kernel"],
        "block_dim": manifest["block_dim"],
        "device_id": args.device_id,
        "warmup_per_process": args.warmup,
        "rounds_per_process": args.rounds,
        "balanced_orders": [list(order) for order in orders],
        "compared_output_hashes": output_hashes,
        "block_medians_us": block_medians,
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

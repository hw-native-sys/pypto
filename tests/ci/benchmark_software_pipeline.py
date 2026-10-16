# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Benchmark Tile IR software pipelines on library kernels and scratch pressure.

Both variants run against the same saved inputs and Torch reference. Compilation
and warmup are excluded from the reported device timings; raw samples and pass
dumps remain available for review. Run one model family per process because the
library's model modules use unqualified imports such as ``config``.
"""

import argparse
import hashlib
import importlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from pypto import passes
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig


def _revision(path: Path) -> str:
    """Identify the exact source revision without changing either checkout."""
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _source_hash(root: Path) -> str:
    """Include uncommitted compiler sources in the benchmark identity."""
    paths = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "include",
                "src",
                "python",
            ],
        )
        .decode()
        .split("\0")
    )
    digest = hashlib.sha256()
    for name in sorted(set(paths) - {""}):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    return digest.hexdigest()


def _summary(samples: list[float]) -> dict[str, float]:
    """Keep the median and dispersion alongside every raw sample series."""
    assert samples and all(sample > 0 for sample in samples), samples
    quantiles = statistics.quantiles(samples, n=4, method="inclusive")
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "p25": quantiles[0],
        "p75": quantiles[2],
    }


def _run_pair(args, report, name, kernel, specs, golden_fn, compare_fn=None):
    """Validate before timing and freeze the baseline golden for the enabled run."""
    golden = importlib.import_module("golden")

    golden_data = None
    pair: dict[str, Any] = {
        "case": name,
        "tensors": [
            {"name": spec.name, "shape": spec.shape, "dtype": str(spec.dtype)}
            for spec in specs
            if hasattr(spec, "shape")
        ],
        "variants": [],
    }
    report["cases"].append(pair)
    # A clean main checkout predates the option. Measure its unchanged path
    # without passing an unsupported keyword or inventing an enabled variant.
    supports_software = hasattr(passes.PassContext, "get_enable_software_pipeline")
    for enabled in (False, True) if supports_software else (False,):
        output = args.output / name / ("on" if enabled else "off")
        config = RunConfig(
            backend_type=BackendType.Ascend910B,
            execution_mode=ExecutionMode.ONBOARD,
            device_id=args.device,
            save_kernels=True,
            save_kernels_dir=str(output),
            dump_passes=True,
        )
        record: dict[str, Any] = {"software_pipeline": enabled, "passed": False}
        pair["variants"].append(record)
        # Fix both variants to Tile IR. The normal flag-off default is Buffer IR,
        # which would confound the scheduling comparison.
        context = (
            passes.PassContext([], enable_buffer_ir=False, enable_software_pipeline=enabled)
            if supports_software
            else passes.PassContext([], enable_buffer_ir=False)
        )
        with context:
            result = golden.run(
                fn=kernel,
                specs=specs,
                golden_fn=golden_fn,
                golden_data=golden_data,
                save_data=golden_data is None,
                config=config,
                rtol=5e-3,
                atol=2e-2,
                compare_fn=compare_fn,
            )
        record.update(passed=bool(result.passed), error=result.error, work_dir=str(result.work_dir))
        if not result.passed:
            raise RuntimeError(f"{name} software_pipeline={enabled}: {result.error}")
        assert result.work_dir is not None and result.bench is not None
        if golden_data is None:
            golden_data = str(Path(result.work_dir) / "data")
        stats = result.bench
        samples = {"device_wall": list(stats.device_wall_us), "effective": list(stats.per_round("effective"))}
        assert all(len(series) == args.rounds for series in samples.values())
        record.update(
            samples_us=samples,
            summary_us={metric: _summary(values) for metric, values in samples.items()},
            golden_replayed=enabled,
            golden_data=golden_data,
        )
        # Ordinary slices also emit subviews. The dedicated pass marker proves
        # that an eligible loop actually entered software-slot lowering.
        sources = list((Path(result.work_dir) / "ptoas").glob("*.pto"))
        record["pto_files"] = len(sources)
        record["pto_subview_count"] = sum(path.read_text().count("pto.subview") for path in sources)
        dumps = list((Path(result.work_dir) / "passes_dump").glob("*_after_LowerPipelineToSlots.py"))
        assert len(dumps) == 1
        record["software_slot_markers"] = dumps[0].read_text().count("software_pipeline_slots")
        print(f"[PIPELINE_BENCH] {name} enabled={enabled} {record['summary_us']}", flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if not supports_software:
        return
    off, on = pair["variants"]
    pair["device_speedup"] = (
        off["summary_us"]["device_wall"]["median"] / on["summary_us"]["device_wall"]["median"]
    )
    pair["effective_speedup"] = (
        off["summary_us"]["effective"]["median"] / on["summary_us"]["effective"]["median"]
    )


def _rmsnorm(args, report):
    """Use the model's decode/prefill shapes and its published precision contract."""
    golden = importlib.import_module("golden")

    sys.path.insert(0, str(args.lib / "models" / "deepseek_v4_flash_mtp"))
    module = importlib.import_module("rmsnorm")
    modes = (
        ("decode", module.DECODE_BATCH, module.DECODE_SEQ),
        ("prefill", module.PREFILL_BATCH, module.PREFILL_SEQ),
    )
    for seed in (0, 2026):
        for mode, batch, sequence in modes:
            torch.manual_seed(seed)
            specs = module.build_tensor_specs(batch, sequence)
            name = f"rmsnorm-{mode}-seed{seed}"
            _run_pair(
                args,
                report,
                name,
                module.rms_norm_test,
                specs,
                module.golden_rms_norm_test,
                {"x_normed": golden.ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
            )


def _paged_attention(args, report):
    """Run the library's boundary, ragged, mapping, GQA and cache-canary matrix."""
    sys.path.insert(0, str(args.lib / "models" / "qwen3_14b"))
    module = importlib.import_module("test_paged_attention_pypto")
    for case in module.matrix_cases(args.matrix):
        fixture = module.make_fixture(case)
        specs = module.build_specs(case, fixture)
        _run_pair(
            args,
            report,
            case.name,
            module.paged_attention_pypto_dynamic,
            specs,
            lambda values: module.golden_attention(values, case),
            {
                "out": module._compare_output,
                "key_cache": lambda actual, expected, **kw: module._compare_cache(
                    actual, expected, case=case, **kw
                ),
                "value_cache": lambda actual, expected, **kw: module._compare_cache(
                    actual, expected, case=case, **kw
                ),
            },
        )


def _qwen(args, report):
    """Measure the original stage=4 model kernel as an untransformed control."""
    golden = importlib.import_module("golden")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "st"))
    cases = importlib.import_module("codegen.dsl.software_pipeline_cases")

    def reference(values):
        x = values["x"].float()
        values["y"][:] = (x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + 1e-6) * values["weight"]).to(
            torch.bfloat16
        )

    for seed in (0, 2026):
        torch.manual_seed(seed)
        specs = [
            golden.TensorSpec("x", [16, 8192], torch.bfloat16, init_value=torch.randn),
            golden.TensorSpec("weight", [1, 8192], torch.float32, init_value=torch.randn),
            golden.TensorSpec("y", [16, 8192], torch.bfloat16),
        ]
        _run_pair(
            args, report, f"qwen-input-rmsnorm-stage4-seed{seed}", cases.qwen_input_rmsnorm, specs, reference
        )


def _pipeline(args, report):
    """Measure kernels known to lower into software slots, beside model controls."""
    golden = importlib.import_module("golden")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "st"))
    cases = importlib.import_module("codegen.dsl.software_pipeline_cases")
    language = importlib.import_module("pypto.language")

    def epilogue_reference(values):
        values["out"][:] = (values["acc"].float().clamp_min(0) * values["coef"]).sum(
            dim=1, keepdim=True
        ) * values["scale"]

    for dynamic in (False, True):
        torch.manual_seed(2026)
        specs = [
            golden.TensorSpec(
                "acc",
                [4160, 64],
                torch.int32,
                init_value=lambda: torch.randint(-127, 128, (4160, 64), dtype=torch.int32),
            ),
            golden.TensorSpec("scale", [4160, 1], torch.float32, init_value=torch.randn),
            golden.TensorSpec("coef", [1, 64], torch.float32, init_value=torch.randn),
            golden.ScalarSpec("n", torch.int64, 65),
            golden.TensorSpec("out", [4160, 1], torch.float32, init_value=-777.0),
        ]
        _run_pair(
            args,
            report,
            f"epilogue-dynamic{dynamic}",
            cases.make_epilogue(dynamic=dynamic),
            specs,
            epilogue_reference,
        )

    def mixed_reference(values):
        values["y"][:] = (values["x"].float() @ values["w"].float() + 1.0) * values["scale"]
        values["z"][:] = values["y"] + 1.0

    for mode in (language.SplitMode.UP_DOWN, language.SplitMode.LEFT_RIGHT):
        for stage, slots in ((2, 3), (3, 3), (4, 3), (3, 2)):
            torch.manual_seed(2026)
            specs = [
                golden.TensorSpec("x", [512, 32], torch.float16, init_value=torch.randn),
                golden.TensorSpec("w", [32, 16], torch.float16, init_value=torch.randn),
                golden.TensorSpec("scale", [512, 1], torch.float32, init_value=torch.randn),
                golden.ScalarSpec("n", torch.int64, 4),
                golden.TensorSpec("y", [512, 16], torch.float32),
                golden.TensorSpec("z", [512, 16], torch.float32),
            ]
            _run_pair(
                args,
                report,
                f"mixed-{mode.name}-stage{stage}-fifo{slots}",
                cases.make_mixed(stage, slots, mode),
                specs,
                mixed_reference,
            )
    # Reject a timing-only success if no eligible loop was transformed.
    for pair in report["cases"]:
        if len(pair["variants"]) == 1:
            continue
        enabled = pair["variants"][1]
        dumps = list((Path(enabled["work_dir"]) / "passes_dump").glob("*_after_LowerPipelineToSlots.py"))
        assert len(dumps) == 1
        assert "software_pipeline_slots" in dumps[0].read_text(), pair["case"]


def _scratch_pressure(args, report):
    """Measure the same requested stages before/after scratch lifetime reuse."""
    golden = importlib.import_module("golden")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "st"))
    cases = importlib.import_module("codegen.dsl.software_pipeline_cases")
    total_rows, cols, scratch_rows = 36960, 128, 64
    for rows in args.tile_rows:
        torch.manual_seed(2026)
        specs = [
            golden.TensorSpec("x", [total_rows, cols], torch.float32, init_value=torch.randn),
            golden.TensorSpec("reduced", [8 * scratch_rows, 1], torch.float32),
            golden.TensorSpec("processed", [total_rows, cols], torch.float32),
        ]

        def reference(values):
            values["reduced"][:] = values["x"][: 8 * scratch_rows].sum(dim=1, keepdim=True)
            values["processed"][:] = values["x"] + 1.0

        _run_pair(
            args,
            report,
            f"scratch-pressure-{scratch_rows}-{rows}x{cols}",
            cases.make_scratch_pressure(rows, total_rows),
            specs,
            reference,
        )


def main() -> int:
    """Require an explicit device and record enough metadata to reproduce timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=("rmsnorm", "paged-attention", "pipeline", "qwen", "scratch-pressure"),
        required=True,
    )
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix", choices=("pr", "full"), default="full")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tile-rows", type=int, nargs="+", default=[32, 44, 56])
    args = parser.parse_args()
    if args.rounds < 4 or args.warmup < 0:
        parser.error("--rounds must be >= 4 and --warmup must be >= 0")
    if args.case == "scratch-pressure" and any(rows <= 0 or 36960 % rows for rows in args.tile_rows):
        parser.error("--tile-rows must contain positive divisors of the fixed 36960 input rows")
    args.lib, args.output = args.lib.resolve(), args.output.resolve()
    sys.path.insert(0, str(args.lib))
    os.environ.update(
        PYPTO_BENCH="1", PYPTO_BENCH_ROUNDS=str(args.rounds), PYPTO_BENCH_WARMUP=str(args.warmup)
    )
    # Attribute results to the imported compiler, including baseline checkouts.
    compiler = importlib.import_module("pypto")
    assert compiler.__file__ is not None, "Benchmarking requires a source-backed compiler package"
    root = Path(compiler.__file__).resolve().parents[2]
    report: dict[str, Any] = {
        "pypto_revision": _revision(root),
        "compiler_source_sha256": _source_hash(root),
        "library_revision": _revision(args.lib),
        "runtime_revision": _revision(root / "runtime"),
        "pto_isa_pin": (root / "runtime" / "pto_isa.pin").read_text().strip(),
        "cann_root": os.environ.get("ASCEND_HOME_PATH"),
        "platform": "a2a3",
        "device": args.device,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "ir": "Tile",
        "cases": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        runners = {
            "rmsnorm": _rmsnorm,
            "paged-attention": _paged_attention,
            "pipeline": _pipeline,
            "qwen": _qwen,
            "scratch-pressure": _scratch_pressure,
        }
        runners[args.case](args, report)
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

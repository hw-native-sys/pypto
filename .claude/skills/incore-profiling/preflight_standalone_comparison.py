# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare and locally compile a standalone fixed-placement comparison."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gen_profiling_case as generator
import standalone_compare as comparison

_PTO_FUNCTION_RE = re.compile(r"func\.func\s+@([A-Za-z_]\w*)\s*\(")
_ALLOC_TILE_ADDR_RE = re.compile(r"^\s*%\S+\s*=\s*pto\.alloc_tile\b.*\baddr\s*=", re.MULTILINE)


@dataclass(frozen=True)
class PtoUnit:
    """One persisted PTOAS compilation unit containing a target function."""

    path: Path
    functions: tuple[str, ...]
    target_function: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_target_sync_summary(path: Path, function: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid synchronization-summary JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: synchronization summary must be a JSON object")
        if record.get("function") == function:
            records.append(record)
    if len(records) != 1:
        raise ValueError(
            f"{path}: expected exactly one synchronization summary for {function!r}, got {len(records)}"
        )
    return records[0]


def _pto_dirs(build_dir: Path) -> list[Path]:
    roots = [build_dir / "ptoas"]
    next_levels = build_dir / "next_levels"
    if next_levels.is_dir():
        roots.extend(path / "ptoas" for path in sorted(next_levels.iterdir()) if path.is_dir())
    return [path for path in roots if path.is_dir()]


def resolve_pto_unit(build_dir: Path, function: str, unit_name: str | None = None) -> PtoUnit:
    """Resolve exactly one persisted PTOAS unit containing ``function``."""
    candidates: list[PtoUnit] = []
    for pto_dir in _pto_dirs(build_dir):
        paths = [pto_dir / f"{unit_name}.pto"] if unit_name is not None else sorted(pto_dir.glob("*.pto"))
        for path in paths:
            if not path.is_file():
                continue
            functions = tuple(_PTO_FUNCTION_RE.findall(path.read_text(encoding="utf-8")))
            if function in functions:
                candidates.append(PtoUnit(path.resolve(), functions, function))
    if not candidates:
        qualifier = f" in unit {unit_name!r}" if unit_name is not None else ""
        raise FileNotFoundError(
            f"no persisted PTOAS unit contains function {function!r}{qualifier}: {build_dir}"
        )
    if len(candidates) != 1:
        paths = [str(candidate.path) for candidate in candidates]
        raise ValueError(
            f"function {function!r} occurs in {len(candidates)} PTOAS units; select one with --unit: {paths}"
        )
    return candidates[0]


def compile_pto_unit(
    unit: PtoUnit,
    output_dir: Path,
    ptoas_bin: Path,
    *,
    timeout: int,
) -> tuple[Path, Path, Path]:
    """Compile one persisted PTO unit through InsertSync into a fresh C++ source."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pto_path = output_dir / unit.path.name
    cpp_path = output_dir / f"{unit.path.stem}.cpp"
    summary_path = output_dir / f"{unit.path.stem}.sync.jsonl"
    debug_path = output_dir / f"{unit.path.stem}.sync.debug.txt"
    shutil.copy2(unit.path, pto_path)
    pto_text = pto_path.read_text(encoding="utf-8")
    level = "level3" if _ALLOC_TILE_ADDR_RE.search(pto_text) else "level2"
    command = [
        str(ptoas_bin.resolve()),
        str(pto_path),
        "-o",
        str(cpp_path),
        "--enable-insert-sync",
        f"--pto-level={level}",
        f"--pto-insert-sync-summary={summary_path}",
        "--pto-insert-sync-debug=3",
    ]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    debug_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"PTOAS failed for {unit.path} with exit code {result.returncode}; see {debug_path}"
        )
    if not cpp_path.is_file() or not summary_path.is_file():
        raise RuntimeError(f"PTOAS did not emit both C++ and synchronization summary for {unit.path}")
    emitted = cpp_path.read_text(encoding="utf-8")
    if not re.search(rf"\b{re.escape(unit.target_function)}\s*\(", emitted):
        raise RuntimeError(f"PTOAS C++ output omits target function {unit.target_function!r}")
    return pto_path, cpp_path, summary_path


def _run_checked(command: list[str], *, cwd: Path, log_path: Path, timeout: int) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}; see {log_path}")


def _build_npu_case(
    case_dir: Path,
    *,
    pto_isa_root: Path,
    soc_version: str,
    timeout: int,
) -> Path:
    build_dir = case_dir / "build"
    _run_checked(
        [
            "cmake",
            "-S",
            str(case_dir),
            "-B",
            str(build_dir),
            f"-DPTO_ISA_ROOT={pto_isa_root.resolve()}",
            f"-DSOC_VERSION={soc_version}",
            "-DENABLE_SIM_GOLDEN=OFF",
            "-DENABLE_NPU_BENCHMARK=ON",
        ],
        cwd=case_dir,
        log_path=case_dir / "configure.log",
        timeout=timeout,
    )
    _run_checked(
        ["cmake", "--build", str(build_dir), "--parallel", "2"],
        cwd=case_dir,
        log_path=case_dir / "build.log",
        timeout=timeout,
    )
    return build_dir


def prepare(  # noqa: PLR0913
    baseline_build: Path,
    candidate_build: Path,
    function: str,
    invocation_profile: Path,
    output_root: Path,
    ptoas_bin: Path,
    *,
    ptoas_root: Path | None = None,
    unit_name: str | None = None,
    aicore_arch: str = "dav-c220",
    build_npu: bool = False,
    pto_isa_root: Path | None = None,
    soc_version: str = "Ascend910B2",
    timeout: int = 600,
) -> Path:
    """Prepare, validate, and optionally build both standalone endpoints.

    Args:
        baseline_build: PyPTO build-output directory for the baseline placement.
        candidate_build: PyPTO build-output directory for the candidate placement.
        function: PTO function to compile and compare.
        invocation_profile: Portable launch inputs and output selection.
        output_root: Fresh destination for sources, cases, and the preflight record.
        ptoas_bin: Instrumented PTOAS executable.
        ptoas_root: Optional PTOAS checkout required by mixed validation groups.
        unit_name: Optional PTOAS unit stem used to disambiguate a function.
        aicore_arch: Target architecture passed to the standalone generator.
        build_npu: Whether to compile both generated NPU executables.
        pto_isa_root: pto-isa checkout used by the NPU build.
        soc_version: CMake device target for the NPU build.
        timeout: Per-command timeout in seconds.

    Returns:
        Path to the generated ``preflight.json``.
    """
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")
    if build_npu and pto_isa_root is None:
        raise ValueError("build_npu requires pto_isa_root")
    profile = generator.load_invocation_profile(invocation_profile)
    output_root.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, Any]] = {}
    case_dirs: dict[str, Path] = {}
    for label, build_dir in (("baseline", baseline_build), ("candidate", candidate_build)):
        unit = resolve_pto_unit(build_dir, function, unit_name)
        source_dir = output_root / "sources" / label
        pto_path, cpp_path, summary_path = compile_pto_unit(
            unit,
            source_dir,
            ptoas_bin,
            timeout=timeout,
        )
        case_dir = generator.generate(
            cpp_path,
            f"{label}_{function}",
            output_root / "cases" / label,
            aicore_arch,
            run_mode="npu",
            block_dim=profile.block_dim,
            input_dir=profile.input_dir,
            scalar_values=profile.scalar_values,
            synthetic_seed=profile.synthetic_seed,
            pointer_fills=profile.pointer_fills,
            recommended_outputs=profile.outputs,
            invocation_profile=profile.source_path,
            ptoas_root=ptoas_root,
        )
        build_dir_out = None
        if build_npu:
            assert pto_isa_root is not None
            build_dir_out = _build_npu_case(
                case_dir,
                pto_isa_root=pto_isa_root,
                soc_version=soc_version,
                timeout=timeout,
            )
        records[label] = {
            "input_build_dir": str(build_dir.resolve()),
            "unit": str(unit.path),
            "functions": list(unit.functions),
            "pto": {"path": str(pto_path), "sha256": _sha256(pto_path)},
            "cpp": {"path": str(cpp_path), "sha256": _sha256(cpp_path)},
            "sync_summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "target_sync_summary": _load_target_sync_summary(summary_path, function),
            "case_dir": str(case_dir),
            "npu_build_dir": str(build_dir_out) if build_dir_out is not None else None,
        }
        case_dirs[label] = case_dir

    manifest, pointer_names = comparison.validate_cases(case_dirs["baseline"], case_dirs["candidate"])
    if not profile.outputs:
        raise ValueError("invocation profile must name at least one output for the device correctness gate")
    preflight = {
        "schema_version": 1,
        "function": function,
        "invocation_profile": {
            "path": invocation_profile.name,
            "sha256": _sha256(invocation_profile),
        },
        "ptoas": {"path": str(ptoas_bin.resolve()), "sha256": _sha256(ptoas_bin)},
        "kernel": manifest["kernel"],
        "block_dim": manifest["block_dim"],
        "pointer_names": pointer_names,
        "recommended_outputs": profile.outputs,
        "post_insert_sync_compiled": True,
        "npu_cases_built": build_npu,
        "endpoints": records,
    }
    output = output_root / "preflight.json"
    output.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate post-InsertSync C++ and preflight two fixed-placement standalone cases"
    )
    parser.add_argument("--baseline-build", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--unit", help="PTOAS unit stem when the function occurs more than once")
    parser.add_argument("--invocation-profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ptoas-bin", type=Path, required=True)
    parser.add_argument("--ptoas-root", type=Path)
    parser.add_argument("--aicore-arch", default="dav-c220")
    parser.add_argument("--build-npu", action="store_true")
    parser.add_argument("--pto-isa-root", type=Path)
    parser.add_argument("--soc-version", default="Ascend910B2")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    output = prepare(
        args.baseline_build,
        args.candidate_build,
        args.function,
        args.invocation_profile,
        args.output_root,
        args.ptoas_bin,
        ptoas_root=args.ptoas_root,
        unit_name=args.unit,
        aicore_arch=args.aicore_arch,
        build_npu=args.build_npu,
        pto_isa_root=args.pto_isa_root,
        soc_version=args.soc_version,
        timeout=args.timeout,
    )
    print(f"[preflight_standalone_comparison] wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

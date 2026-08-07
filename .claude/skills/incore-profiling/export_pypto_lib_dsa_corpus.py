# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile PyPTO-Lib programs without execution and export a DSA corpus."""

import argparse
import copy
import csv
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211
_UINT64_MASK = (1 << 64) - 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_problem_fingerprint(document: dict[str, Any]) -> str:
    """Match ``dsa::FingerprintStructuredProblem`` alias-label normalization."""
    semantic = copy.deepcopy(document)
    structure = semantic.get("problem", {}).get("pypto_structure")
    if structure is not None:
        for alias_class in structure.get("alias_classes", []):
            alias_class["members"] = [f"member_{index}" for index, _ in enumerate(alias_class["members"])]
    canonical = (json.dumps(semantic, indent=2, sort_keys=True) + "\n").encode()
    value = _FNV_OFFSET
    for byte in canonical:
        value ^= byte
        value = value * _FNV_PRIME & _UINT64_MASK
    return f"{value:016x}"


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def read_export_plan(manifest: Path, requested_platform: str) -> list[tuple[str, str]]:
    """Recover the previously successful launch mode for each program.

    Most model entry points accept simulator platform names.  A few production
    drivers intentionally accept only a real architecture name even when the
    golden runner is replaced by a compile-only shim.  The prior export status
    records that distinction in its launch-mode column.
    """
    plan: list[tuple[str, str]] = []
    with manifest.open(encoding="utf-8", newline="") as source:
        for row in csv.reader(source, delimiter="\t"):
            if not row or row[0].startswith("#") or len(row) < 2 or row[1] != "EXPORTED":
                continue
            mode = row[2] if len(row) > 2 else ""
            platform = requested_platform
            if mode.endswith("-a2a3"):
                platform = "a2a3"
            elif mode.endswith("-a5"):
                platform = "a5"
            plan.append((row[0], platform))
    if not plan:
        raise ValueError(f"manifest contains no EXPORTED scripts: {manifest}")
    return plan


def _patch_golden(
    golden: Any,
    passes: Any,
    export_root: Path,
) -> None:
    call_index = 0

    def wrap(original: Callable[..., Any], *, jit: bool) -> Callable[..., Any]:
        def compile_only(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_index
            call_index += 1
            call_root = export_root / f"call-{call_index:03d}"
            call_root.mkdir(parents=True, exist_ok=False)
            compile_cfg = dict(kwargs.get("compile_cfg") or {})
            compile_cfg.update(
                {
                    "memory_planner": passes.MemoryPlanner.DSA,
                    "dsa_export_dir": str(call_root),
                    "dsa_reuse_penalty_recognizer": passes.DsaReusePenaltyRecognizer.QUADRATIC,
                    "dump_passes": False,
                }
            )
            if jit:
                compile_cfg["codegen_only"] = True
            else:
                compile_cfg["skip_ptoas"] = True
            kwargs["compile_cfg"] = compile_cfg
            kwargs["compile_only"] = True
            kwargs["save_data"] = False
            return original(*args, **kwargs)

        return compile_only

    golden.run = wrap(golden.run, jit=False)
    golden.run_jit = wrap(golden.run_jit, jit=True)


def _patch_compile_for_test(jit_decorator: Any, passes: Any, export_root: Path) -> None:
    """Export programs compiled directly through ``JITFunction.compile_for_test``.

    PyPTO-Lib has two compile-only drivers that intentionally bypass the golden
    harness.  ``compile_for_test`` has no RunConfig argument, but it is specified
    to inherit an ambient PassContext.  Wrapping only that entry point keeps the
    production driver unchanged while exercising the same DSA pipeline.
    """
    original = jit_decorator.JITFunction.compile_for_test
    call_index = 0

    def compile_for_test(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal call_index
        call_index += 1
        call_root = export_root / f"direct-call-{call_index:03d}"
        call_root.mkdir(parents=True, exist_ok=False)
        with passes.PassContext(
            [],
            memory_planner=passes.MemoryPlanner.DSA,
            dsa_export_dir=str(call_root),
            dsa_reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
        ):
            return original(self, *args, **kwargs)

    jit_decorator.JITFunction.compile_for_test = compile_for_test


def run_one(script: Path, export_root: Path, platform: str) -> int:
    """Execute one model entry point after replacing its golden runner."""
    import golden  # noqa: PLC0415
    import pypto.jit.decorator as jit_decorator  # noqa: PLC0415
    from pypto.pypto_core import passes  # noqa: PLC0415

    export_root.mkdir(parents=True, exist_ok=False)
    _patch_golden(golden, passes, export_root)
    _patch_compile_for_test(jit_decorator, passes, export_root)
    # JIT compilation decides whether to invoke PTOAS independently of
    # RunConfig.codegen_only.  Export stops at PTO source generation.
    jit_decorator._ptoas_available = lambda: False
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), "-p", platform]
    runpy.run_path(str(script), run_name="__main__")
    count = sum(1 for _ in export_root.rglob("*.dsa.json"))
    print(f"EXPORTED_DSA_PROBLEMS={count}")
    return 0


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_inventory(output_root: Path, script_by_root: dict[Path, str]) -> tuple[int, int, int]:
    """Index invocations and materialize one representative per fingerprint."""
    invocation_rows: list[dict[str, Any]] = []
    unique: dict[str, dict[str, Any]] = {}
    for export_root, script in sorted(script_by_root.items(), key=lambda item: item[1]):
        for path in sorted(export_root.rglob("*.dsa.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            fingerprint = semantic_problem_fingerprint(document)
            body = document.get("problem", {})
            penalties = len((body.get("cost_model") or {}).get("reuse_penalties", []))
            pools = body.get("pools", [])
            buffers = body.get("buffers", [])
            instance = str(document.get("instance", path.stem))
            row = {
                "script": script,
                "instance": instance,
                "problem_fingerprint": fingerprint,
                "document_sha256": _sha256(path),
                "document": str(path.relative_to(output_root)),
                "buffers": len(buffers),
                "pools": len(pools),
                "pool_names": ";".join(str(pool.get("name", pool.get("id", ""))) for pool in pools),
                "reuse_penalties": penalties,
                "recognizer": document.get("metadata", {}).get("reuse_penalty_recognizer", ""),
            }
            invocation_rows.append(row)
            unique.setdefault(fingerprint, {**row, "source": path})

    corpus_root = output_root / "corpus"
    penalty_root = corpus_root / "penalty-bearing"
    control_root = corpus_root / "no-penalty"
    penalty_root.mkdir(parents=True)
    control_root.mkdir()
    unique_rows: list[dict[str, Any]] = []
    for fingerprint, row in sorted(unique.items()):
        destination_root = penalty_root if int(row["reuse_penalties"]) > 0 else control_root
        tag = f"{_slug(str(row['script']).removesuffix('.py'))}__{_slug(str(row['instance']))}"
        destination = destination_root / f"{tag}-{fingerprint}.dsa.json"
        _link_or_copy(Path(row.pop("source")), destination)
        row["corpus_document"] = str(destination.relative_to(output_root))
        unique_rows.append(row)

    def write_rows(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError(f"cannot write an empty inventory: {name}")
        with (output_root / name).open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    write_rows("invocations.tsv", invocation_rows)
    write_rows("unique-problems.tsv", unique_rows)
    penalty_count = sum(int(row["reuse_penalties"]) > 0 for row in unique_rows)
    return len(invocation_rows), len(unique_rows), penalty_count


def _run_export_subprocess(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
) -> tuple[str, int | str]:
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""
    return ("EXPORTED" if completed.returncode == 0 else "FAILED"), completed.returncode


def _classify_export_status(status: str, problem_count: int) -> str:
    if status == "EXPORTED" and problem_count == 0:
        return "NO_DSA"
    return status


def export_corpus(args: argparse.Namespace) -> int:
    args.output_root = args.output_root.resolve()
    args.pypto_lib_root = args.pypto_lib_root.resolve()
    args.pypto_python = args.pypto_python.resolve()
    if args.manifest is not None:
        args.manifest = args.manifest.resolve()
    args.python = args.python.resolve()
    if args.output_root.exists():
        raise ValueError(f"output root already exists: {args.output_root}")
    if not args.pypto_lib_root.is_dir():
        raise FileNotFoundError(args.pypto_lib_root)
    if not args.pypto_python.is_dir():
        raise FileNotFoundError(args.pypto_python)
    if args.manifest is not None:
        plan = read_export_plan(args.manifest, args.platform)
    elif args.scripts:
        plan = [(script, args.platform) for script in args.scripts]
    else:
        raise ValueError("provide --manifest or at least one --script")
    if args.limit is not None:
        plan = plan[: args.limit]
    args.output_root.mkdir(parents=True)
    captures = args.output_root / "captures"
    logs = args.output_root / "logs"
    builds = args.output_root / "builds"
    captures.mkdir()
    logs.mkdir()
    builds.mkdir()

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(args.pypto_python), str(args.pypto_lib_root)])
    environment.update(
        {
            "PYPTO_CODEGEN_MAX_WORKERS": "1",
            "PYPTO_GOLDEN_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    status_rows: list[dict[str, Any]] = []
    script_by_root: dict[Path, str] = {}
    for index, (relative, platform) in enumerate(plan, start=1):
        script = args.pypto_lib_root / relative
        if not script.is_file():
            raise FileNotFoundError(script)
        tag = _slug(relative.removesuffix(".py"))
        export_root = captures / tag
        script_by_root[export_root] = relative
        child_environment = dict(environment)
        child_environment["PYPTO_PROG_BUILD_DIR"] = str(builds / tag)
        command = [
            str(args.python),
            str(Path(__file__).resolve()),
            "_run-one",
            "--script",
            str(script),
            "--export-root",
            str(export_root),
            "--platform",
            platform,
        ]
        stdout_path = logs / f"{tag}.stdout.log"
        stderr_path = logs / f"{tag}.stderr.log"
        status, returncode = _run_export_subprocess(
            command,
            cwd=args.pypto_lib_root,
            environment=child_environment,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=args.timeout,
        )
        count = sum(1 for _ in export_root.rglob("*.dsa.json")) if export_root.exists() else 0
        status = _classify_export_status(status, count)
        status_rows.append(
            {
                "script": relative,
                "status": status,
                "launch_mode": f"final-{platform}",
                "returncode": returncode,
                "problems": count,
            }
        )
        print(f"[{index}/{len(plan)}] {relative}: {status} ({count} problems)", flush=True)

    with (args.output_root / "export-status.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(status_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(status_rows)
    successful_roots = {
        root: script
        for root, script in script_by_root.items()
        if any(row["script"] == script and row["status"] == "EXPORTED" for row in status_rows)
    }
    invocations, unique, penalties = build_inventory(args.output_root, successful_roots)
    print(
        f"CORPUS invocations={invocations} unique={unique} penalty_bearing={penalties}",
        flush=True,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--pypto-lib-root", type=Path, required=True)
    export.add_argument("--pypto-python", type=Path, required=True)
    export.add_argument("--python", type=Path, required=True)
    export.add_argument("--manifest", type=Path)
    export.add_argument("--script", dest="scripts", action="append")
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--platform", default="a2a3sim")
    export.add_argument("--timeout", type=int, default=1800)
    export.add_argument("--limit", type=int)

    one = subparsers.add_parser("_run-one")
    one.add_argument("--script", type=Path, required=True)
    one.add_argument("--export-root", type=Path, required=True)
    one.add_argument("--platform", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "_run-one":
        return run_one(args.script, args.export_root, args.platform)
    return export_corpus(args)


if __name__ == "__main__":
    sys.exit(main())

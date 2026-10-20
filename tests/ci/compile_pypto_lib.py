# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile every formal pypto-lib entry without executing device code.

Selection is owned by pypto-lib's detect_changes.py and ci_platforms.json.
The subprocess adapter supports older entries without a --compile-only CLI.
"""

import argparse
import ast
import importlib
import inspect
import json
import os
import re
import runpy
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# The library selector also includes these CPU-only reference and checkpoint tests.
# They define no PyPTO kernels and cannot contribute compilation coverage.
NON_KERNEL_ENTRIES = {
    "models/glm5_3_flash/prefill_ref.py",
    "models/glm5_3_flash/weights.py",
}


def select_entries(lib: Path, platform: str) -> list[str]:
    """Use the library's full formal-entry selector, including device-only cases."""
    selected = subprocess.run(
        [sys.executable, str(lib / ".github/scripts/detect_changes.py"), "--all", "--platform", platform],
        cwd=lib,
        check=True,
        capture_output=True,
        text=True,
    )
    entries = sorted(set(selected.stdout.split()) - NON_KERNEL_ENTRIES)
    if not entries:
        raise ValueError(f"No formal pypto-lib entries selected for {platform}")
    for entry in entries:
        path = Path(entry)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.parts[0] not in ("examples", "models")
            or path.name.endswith("_draft.py")
            or not (lib / path).is_file()
        ):
            raise ValueError(f"Invalid formal entry from pypto-lib selector: {entry}")
    return entries


def entry_args(path: Path, platform: str) -> list[str]:
    """Supply platform and declared logical ranks without acquiring devices."""
    source = path.read_text()
    args = [str(path), "-p", platform]
    devices = re.search(r"^#\s*ci:\s*devices=(\d+)\b", source, re.MULTILINE)
    if devices:
        count = int(devices.group(1))
        if count < 1:
            raise ValueError(f"Invalid device count in {path}: {count}")
        args += ["-d", ",".join(str(rank) for rank in range(count))]
    options = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]
    if not devices and any(
        any(isinstance(arg, ast.Constant) and arg.value == "--device" for arg in node.args)
        and any(
            kw.arg == "required" and isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in node.keywords
        )
        for node in options
    ):
        args += ["-d", "0"]
    if any(
        any(isinstance(arg, ast.Constant) and arg.value == "--compile-only" for arg in node.args)
        for node in options
    ):
        args += ["--compile-only"]
    return args


def compile_entry(lib: Path, entry: str, platform: str, receipt: Path) -> None:
    """Force every golden.run call through its real compile-only pipeline."""
    sys.path[:0] = [str(lib), str((lib / entry).parent)]
    golden: Any = importlib.import_module("golden")
    runner: Any = importlib.import_module("golden.runner")

    original = runner.run
    signature = inspect.signature(original)
    results: list[dict[str, Any]] = []
    compile_module: Any = importlib.import_module("pypto.ir.compile")
    ir_module: Any = importlib.import_module("pypto.ir")
    original_compile = compile_module.compile
    in_golden = False

    def full_compile(*args: Any, **kwargs: Any) -> Any:
        # JIT otherwise silently emits only MLIR when PTOAS is unavailable.
        kwargs["skip_ptoas"] = False
        record = {"passed": False, "error": "Compilation did not return successfully"}
        if not in_golden:
            results.append(record)
        compiled = original_compile(*args, **kwargs)
        record.update(passed=True, error="")
        return compiled

    def compile_only(*args: Any, **kwargs: Any) -> Any:
        nonlocal in_golden
        bound = signature.bind(*args, **kwargs)
        bound.arguments["compile_only"] = True
        bound.arguments["runtime_dir"] = None
        record = {"passed": False, "error": "Compilation did not return successfully"}
        results.append(record)
        in_golden = True
        try:
            result = original(*bound.args, **bound.kwargs)
        finally:
            in_golden = False
        record.update(passed=bool(result.passed), error=str(result.error or ""))
        return result

    def reject_execution(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Compile-only audit attempted to prepare runtime inputs")

    # Entries import either spelling. Install both before executing their main.
    golden.run = runner.run = compile_only
    runner._prepare_inputs = reject_execution
    compile_module.compile = ir_module.compile = full_compile
    sys.argv = entry_args(lib / entry, platform)
    try:
        runpy.run_path(str(lib / entry), run_name="__main__")
    finally:
        receipt.write_text(json.dumps(results, indent=2) + "\n")


def run_case(lib: Path, entry: str, platform: str, output: Path, timeout: int) -> dict[str, Any]:
    """Run one isolated interpreter; kill its entire compiler process group on timeout."""
    case_dir = output / Path(entry).with_suffix("")
    case_dir.mkdir(parents=True, exist_ok=True)
    receipt = case_dir / "compilations.json"
    receipt.unlink(missing_ok=True)
    log_path = case_dir / "compile.log"
    env = os.environ | {"PYPTO_PROG_BUILD_DIR": str(case_dir / "build_output")}
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--lib",
        str(lib),
        "--platform",
        platform,
        "--entry",
        entry,
        "--receipt",
        str(receipt),
    ]
    start = time.monotonic()
    timed_out = False
    with log_path.open("w") as log:
        with subprocess.Popen(
            command, cwd=lib, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        ) as proc:
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait()
                log.write(f"\nTimed out after {timeout}s\n")
    try:
        results = json.loads(receipt.read_text()) if receipt.exists() else []
    except (json.JSONDecodeError, OSError):
        results = []
    if not isinstance(results, list) or any(
        not isinstance(result, dict) or not isinstance(result.get("passed"), bool) for result in results
    ):
        results = []
    passed = code == 0 and bool(results) and all(result["passed"] for result in results)
    return {
        "entry": entry,
        "platform": platform,
        "passed": passed,
        "exit_code": code,
        "timed_out": timed_out,
        "seconds": round(time.monotonic() - start, 2),
        "compilations": results,
        "log": str(log_path.relative_to(output)),
    }


def positive_int(value: str) -> int:
    """Reject zero and negative worker counts and timeouts."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}")
    return parsed


def main() -> int:
    """Run the full matrix leg and write a reproducible manifest and result summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--platform", choices=("a2a3", "a5"), required=True)
    parser.add_argument("--output", type=Path, default=Path("build/lib-compile"))
    parser.add_argument("--jobs", type=positive_int, default=4)
    parser.add_argument("--timeout", type=positive_int, default=1200)
    parser.add_argument("--entry", help=argparse.SUPPRESS)
    parser.add_argument("--receipt", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    lib = args.lib.resolve()
    if args.entry:
        if args.receipt is None:
            parser.error("--entry requires --receipt")
        compile_entry(lib, args.entry, args.platform, args.receipt)
        return 0

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries = select_entries(lib, args.platform)
    revision = subprocess.check_output(["git", "-C", str(lib), "rev-parse", "HEAD"], text=True).strip()
    manifest = {"lib_revision": revision, "platform": args.platform, "entries": entries}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Compiling {len(entries)} entries on {args.platform}; pypto-lib {revision}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_case, lib, entry, args.platform, output, args.timeout): entry for entry in entries
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status} {result['entry']} ({result['seconds']}s) [{result['log']}]", flush=True)
            if not result["passed"]:
                print((output / result["log"]).read_text(errors="replace")[-12000:], flush=True)
    results.sort(key=lambda result: result["entry"])
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    passed = sum(result["passed"] for result in results)
    summary = f"pypto-lib {args.platform}: {passed}/{len(results)} passed (library {revision})"
    print(summary, flush=True)
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a") as summary_file:
            summary_file.write(summary + "\n")
            for result in results:
                if not result["passed"]:
                    summary_file.write(f"- FAIL: `{result['entry']}` — `{result['log']}`\n")
    return int(passed != len(results))


if __name__ == "__main__":
    sys.exit(main())

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
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
from pathlib import Path
from typing import Any

# The library selector also includes these CPU-only reference and checkpoint tests.
# Their default CLI paths run CPU checks and cannot contribute compilation coverage.
NON_KERNEL_ENTRIES = {
    "models/glm5_3_flash/prefill_fwd.py",  # main() defaults to CPU weight checks.
    "models/glm5_3_flash/prefill_ref.py",
    "models/glm5_3_flash/weights.py",
}


# These entry contracts use runtime-dir as a compile OUTPUT. Other library
# CLIs use that spelling for replay inputs or only in checkpoint branches;
# discovering an add_argument call is not sufficient to enable it (#3205).
# No current kernel entry uses runtime-dir as a compile output.
COMPILE_OUTPUT_ENTRIES: set[str] = set()

# pypto-lib #1622 retained these kernels but changed the module CLI to bounded
# CPU reference/weight checks. Audit its HOST roots directly from their annotated
# signatures; each root compiles the associated CHIP kernel as a dependency.
SIGNATURE_ENTRIES = {
    "models/glm5_3_flash/prefill_fwd.py": ("l3_copy_request_snapshot", "l3_reset_request_caches"),
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


def entry_args(path: Path, platform: str, runtime_dir: Path | None = None) -> list[str]:
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
        if runtime_dir is not None:
            args += ["--runtime-dir", str(runtime_dir)]
    return args


def read_receipt(receipt: Path) -> list[dict[str, Any]]:
    """Reject absent or malformed compilation records without aborting the matrix."""
    try:
        results = json.loads(receipt.read_text()) if receipt.exists() else []
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(results, list) or any(
        not isinstance(result, dict) or not isinstance(result.get("passed"), bool) for result in results
    ):
        return []
    return results


def compile_entry(lib: Path, entry: str, platform: str, receipt: Path, argv: list[str] | None = None) -> None:
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
    original_popen = subprocess.Popen
    popen_signature = inspect.signature(original_popen)
    child_ids = count()
    children: list[tuple[Path, subprocess.Popen[Any]]] = []
    entry_path = (lib / entry).resolve()

    class CompileOnlyPopen(subprocess.Popen):
        def __init__(self, args: Any, *popenargs: Any, **kwargs: Any) -> None:
            command = args
            child_receipt = None
            if (
                isinstance(command, (list, tuple))
                and len(command) >= 2
                and Path(os.fsdecode(command[0])).resolve() == Path(sys.executable).resolve()
            ):
                options = popen_signature.bind(args, *popenargs, **kwargs).arguments
                script = Path(os.fsdecode(options.get("cwd") or Path.cwd())) / os.fsdecode(command[1])
                if (
                    not options.get("shell")
                    and not options.get("executable")
                    and script.resolve() == entry_path
                ):
                    # Self-dispatching entries need the same audit in each fresh interpreter.
                    child_receipt = receipt.with_name(f"{receipt.stem}.{next(child_ids)}.json")
                    child_receipt.unlink(missing_ok=True)
                    command = [
                        command[0],
                        str(Path(__file__).resolve()),
                        "--lib",
                        str(lib),
                        "--platform",
                        platform,
                        "--entry",
                        entry,
                        "--receipt",
                        str(child_receipt),
                        "--entry-args",
                        json.dumps([os.fsdecode(arg) for arg in command[2:]]),
                    ]
            try:
                super().__init__(command, *popenargs, **kwargs)
            except OSError as error:
                if child_receipt is not None:
                    results.append({"passed": False, "error": f"Child process failed to start: {error}"})
                raise
            self.args = args
            if child_receipt is not None:
                children.append((child_receipt, self))

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
    subprocess.Popen = CompileOnlyPopen
    # A fresh per-invocation directory prevents a retained manifest from turning
    # this audit into a cache-load check. Keep artifacts beside the receipt.
    if argv is None:
        runtime_dir = (
            Path(tempfile.mkdtemp(prefix="runtime-", dir=receipt.parent))
            if entry in COMPILE_OUTPUT_ENTRIES
            else None
        )
        sys.argv = entry_args(entry_path, platform, runtime_dir)
    else:
        # Self-dispatched children inherit explicit CLI arguments. Replace the
        # inherited manifest directory while preserving their case selection.
        child_args = list(argv)
        for index, arg in enumerate(child_args):
            if entry not in COMPILE_OUTPUT_ENTRIES:
                break
            if arg == "--runtime-dir" and index + 1 < len(child_args):
                child_args[index + 1] = tempfile.mkdtemp(prefix="runtime-", dir=receipt.parent)
            elif arg.startswith("--runtime-dir="):
                child_args[index] = "--runtime-dir=" + tempfile.mkdtemp(prefix="runtime-", dir=receipt.parent)
        sys.argv = [str(entry_path), *child_args]
    signature_entries = SIGNATURE_ENTRIES.get(entry, ())
    if signature_entries:
        tree = ast.parse(entry_path.read_text())
        definitions = {
            node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        } | {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        if not set(signature_entries) <= definitions:
            signature_entries = ()
    try:
        if signature_entries and "--compile-only" not in sys.argv:
            from pypto.backend import BackendType  # noqa: PLC0415
            from pypto.runtime import RunConfig  # noqa: PLC0415

            namespace = runpy.run_path(str(entry_path), run_name="_pypto_compile_audit_")
            backend = BackendType.Ascend950 if platform == "a5" else BackendType.Ascend910
            config = RunConfig(backend_type=backend)
            for name in signature_entries:
                function = namespace.get(name)
                if function is None or not callable(getattr(function, "compile", None)):
                    raise ValueError(f"Missing compile-only entry {name} in {entry}")
                function.compile(config=config)
        else:
            runpy.run_path(str(entry_path), run_name="__main__")
    finally:
        subprocess.Popen = original_popen
        for child_receipt, child in children:
            child_results = read_receipt(child_receipt)
            child_code = child.poll()
            if not child_results or child_code != 0:
                child_results.append(
                    {
                        "passed": False,
                        "error": "Child process failed or produced no compilation receipts: "
                        f"{child_receipt.name}, exit_code={child_code}",
                    }
                )
            results.extend(child_results)
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
    results = read_receipt(receipt)
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
    parser.add_argument("--entry-args", type=json.loads, help=argparse.SUPPRESS)
    args = parser.parse_args()
    lib = args.lib.resolve()
    if args.entry:
        if args.receipt is None:
            parser.error("--entry requires --receipt")
        compile_entry(lib, args.entry, args.platform, args.receipt, args.entry_args)
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

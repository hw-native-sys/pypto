# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Repeat five A5 URMA probes on one queue-owned device pair; never retry away failures."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROL = "tests/st/a5/tensormap_and_ringbuffer/urma_deferred_completion/test_urma_deferred_completion.py"
CASES = (
    ("pypto-put", ROOT, "tests/st/distributed/test_urma_mfe.py::test_put"),
    ("simpler-put", ROOT / "runtime", f"{CONTROL}::TestUrmaPut::test_run"),
    ("pypto-get", ROOT, "tests/st/distributed/test_urma_mfe.py::test_get"),
    ("simpler-get", ROOT / "runtime", f"{CONTROL}::TestUrmaGet::test_run"),
    ("pypto-exchange", ROOT, "tests/st/distributed/test_urma_mfe.py::test_exchange"),
)


def board_matches(board: dict, expected: str, cann: Path) -> bool:
    if board["returncode"] != 0:
        return False
    text = board["stdout"]
    if f"950{expected}" in text:
        return True
    # Some driver versions print the chip family and SKU separately. Match
    # the exact installed platform INI, as the runtime's architecture gate does.
    chip = re.search(r"Chip Name\s*:\s*(\S+)", text)
    sku = re.search(r"NPU Name\s*:\s*(\S+)", text)
    if chip is None or sku is None or chip[1] != "Ascend950" or not sku[1].isalnum():
        return False
    return any(
        (cann / arch / "data/platform_config" / f"Ascend950{expected}_{sku[1]}.ini").is_file()
        for arch in ("aarch64-linux", "x86_64-linux")
    )


def interrupt_run(signum, frame):
    # Raising unwinds run_case's finally block before the queue releases cards.
    raise KeyboardInterrupt(f"Diagnostic interrupted by signal {signum}")


def capture(command: list[str], cwd: Path = ROOT) -> dict:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False, timeout=30)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def validate_junit(path: Path) -> bool:
    """A zero pytest exit alone cannot turn collection/skip into execution evidence."""
    try:
        cases = list(ET.parse(path).getroot().iter("testcase"))
    except (OSError, ET.ParseError):
        return False
    return len(cases) == 1 and not any(
        case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")
    )


def run_case(command: list[str], cwd: Path, directory: Path, env: dict, timeout: int = 180) -> dict:
    directory.mkdir(parents=True)
    ascend = directory / "ascend"
    ascend.mkdir()
    env = {**env, "ASCEND_PROCESS_LOG_PATH": str(ascend), "PYTHONUNBUFFERED": "1"}
    timed_out = False
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        child = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True
        )
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Own the whole process group, including forked L2 workers. The queue
            # allocation remains held until these children have been terminated.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
    result = {
        "command": command,
        "returncode": child.returncode,
        "timed_out": timed_out,
        "passed": not timed_out and child.returncode == 0 and validate_junit(directory / "junit.xml"),
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def diagnostics(directory: Path) -> list[str]:
    signatures = ("ASYNC_WAIT", "sub_class=", "sched_error_code=", "CQE status", "URMA completion")
    found = []
    for path in sorted(directory.rglob("*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if any(signature in line for signature in signatures):
                found.append(f"{path.relative_to(directory)}: {line}")
    return found


def main() -> int:
    signal.signal(signal.SIGTERM, interrupt_run)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--expected-soc", choices=("PR", "DT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    devices = args.devices.split(",")
    if len(devices) != 2 or any(not device.isdecimal() for device in devices) or devices[0] == devices[1]:
        parser.error("--devices must name two distinct allocated devices")
    if os.environ.get("TASK_DEVICE") != args.devices:
        parser.error("Run inside task-submit using its exact TASK_DEVICE pair")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def progress(**state):
        state["updated_at"] = time.time()
        temporary = output / "current.json.tmp"
        temporary.write_text(json.dumps(state) + "\n")
        temporary.replace(output / "current.json")
        print(f"PROGRESS {state}", flush=True)

    progress(phase="preflight", devices=args.devices)
    identity = {
        "runner": os.environ.get("RUNNER_NAME"),
        "host_arch": platform.machine(),
        "devices_by_rank": devices,
        "expected_soc": args.expected_soc,
        "pypto": capture(["git", "rev-parse", "HEAD"]),
        "runtime": capture(["git", "rev-parse", "HEAD"], ROOT / "runtime"),
        "isa_pin": (ROOT / "runtime/pto_isa.pin").read_text().strip(),
        "toolchain": (ROOT / "toolchain/versions.env").read_text(),
        "python": sys.version,
    }
    identity["board"] = [capture(["npu-smi", "info", "-t", "board", "-i", device]) for device in devices]
    identity["driver"] = capture(["npu-smi", "info"])
    identity["installed"] = capture(
        [
            sys.executable,
            "-c",
            (
                "import json, torch, pypto, _task_interface; "
                "print(json.dumps({'pypto':pypto.__file__, 'torch':torch.__version__, "
                "'simpler_build':getattr(_task_interface,'__build_commit__',None), "
                "'simpler_extension':_task_interface.__file__}))"
            ),
        ]
    )
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent"))
    identity["cann_metadata"] = {
        str(path.relative_to(cann)): path.read_text(errors="replace")
        for pattern in ("compiler/version.info", "opp/version.info", "*-linux/ascend_toolkit_install.info")
        for path in cann.glob(pattern)
        if path.is_file()
    }
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    for board in identity["board"]:
        if not board_matches(board, args.expected_soc, cann):
            raise RuntimeError("Allocated SoC does not match the requested class; inspect identity.json")
    if identity["installed"]["returncode"] != 0:
        raise RuntimeError("Cannot identify the installed runtime; inspect identity.json")
    installed = json.loads(identity["installed"]["stdout"].strip().splitlines()[-1])
    if installed["simpler_build"] != identity["runtime"]["stdout"].strip():
        raise RuntimeError("Installed Simpler build does not match the pinned runtime")
    rows = []
    # Three independent trials per case, interleaved with the corresponding
    # standalone controls. Never stop after a red case or erase an earlier red.
    for attempt in range(1, 4):
        for name, cwd, node in CASES:
            directory = output / f"round-{attempt}" / name
            command = [
                sys.executable,
                "-m",
                "pytest",
                node,
                "--platform=a5",
                f"--device={args.devices}",
                "-v",
                "-s",
                "-o",
                "addopts=",
                f"--junitxml={directory / 'junit.xml'}",
            ]
            if name.startswith("simpler"):
                # These L3 classes are resource jobs in runtime/conftest.py.
                # The diagnostic owns an exact two-device pair already, so run
                # the scheduled child directly; otherwise pytest dispatches a
                # second process and writes a zero-test parent JUnit report.
                command.extend(
                    [
                        "--resource-child",
                        "--runtime",
                        "tensormap_and_ringbuffer",
                        "--level",
                        "3",
                        "--manual=include",
                        "--require-pto-isa",
                    ]
                )
            progress(phase="running", case=name, attempt=attempt, devices=args.devices)
            started = time.monotonic()
            result = run_case(command, cwd, directory, dict(os.environ))
            result["elapsed_seconds"] = round(time.monotonic() - started, 2)
            rows.append({"case": name, "attempt": attempt, **result})
            (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
            (directory / "diagnostics.txt").write_text("\n".join(diagnostics(directory)) + "\n")
            print(f"round={attempt} case={name}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    summary = [f"{args.expected_soc} URMA MFEs — devices {args.devices}"]
    for name, _, _ in CASES:
        passed = sum(row["passed"] for row in rows if row["case"] == name)
        summary.append(f"{name}: {passed}/3 passed")
    (output / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    progress(phase="complete", passed=sum(row["passed"] for row in rows), total=len(rows))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

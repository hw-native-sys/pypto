# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Warm and queue card pairs incrementally, streaming progress and attempt evidence."""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def emit(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())} UTC] {message}", flush=True)


def run_logged(command: list[str], log_path: Path, label: str, timeout: float) -> int:
    """Tee output immediately; a silent child still gets a bounded wait and heartbeat."""
    started = time.monotonic()
    emit(f"START {label} timeout={timeout}s")
    with log_path.open("w") as log:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        def forward():
            for line in child.stdout:
                log.write(line)
                log.flush()
                emit(f"{label}: {line.rstrip()}")

        reader = threading.Thread(target=forward, daemon=True)
        reader.start()
        code = None
        try:
            while code is None:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    emit(f"TIMEOUT {label} elapsed={time.monotonic() - started:.1f}s")
                    code = 124
                    break
                try:
                    code = child.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    emit(f"HEARTBEAT {label} elapsed={time.monotonic() - started:.1f}s")
        finally:
            # Reap compiler descendants as well as the launcher before moving on.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            reader.join(timeout=5)
        emit(f"END {label} returncode={code} elapsed={time.monotonic() - started:.1f}s")
        return code


def watch_results(output: Path, stop: threading.Event) -> None:
    # Queue clients may buffer their remote stdout. Read the shared files so a
    # completed attempt and the active phase remain visible before queue exit.
    seen = {}
    heartbeat = time.monotonic()
    offsets = {}
    while True:
        for path in sorted(output.glob("pair-*/current.json")) + sorted(output.glob("pair-*/results.json")):
            try:
                content = path.read_text()
                if seen.get(path) == content:
                    continue
                data = json.loads(content)
                old = json.loads(seen.get(path, "[]")) if path.name == "results.json" else []
                seen[path] = content
                if path.name == "current.json":
                    emit(f"{path.parent.name} progress: {data}")
                else:
                    for row in data[len(old) :]:
                        emit(
                            f"{path.parent.name} round={row['attempt']} case={row['case']} "
                            f"{'PASS' if row['passed'] else 'FAIL'} timeout={row['timed_out']}"
                        )
            except (OSError, ValueError):
                continue
        # Keep active pytest/device evidence visible even if task-submit only
        # forwards its command's stdout after the entire allocation exits.
        for path in sorted(output.glob("pair-*/round-*/*/*.log")):
            try:
                with path.open(errors="replace") as log:
                    log.seek(offsets.get(path, 0))
                    chunk = log.read(8192)
                    offsets[path] = log.tell()
                if chunk:
                    emit(f"{path.relative_to(output)}: {chunk.rstrip()}")
            except OSError:
                continue
        if time.monotonic() - heartbeat >= 30:
            emit("HEARTBEAT all-card controller; queue/warmup labels and latest pair phases above")
            heartbeat = time.monotonic()
        if stop.wait(1):
            return


def card_pairs(text: str) -> list[list[int]]:
    cards = sorted({int(value) for value in re.findall(r"NPU ID\s*[:：]\s*(\d+)", text)})
    if len(cards) < 2:
        raise RuntimeError("Need at least two discoverable NPUs; inspect inventory.log")
    pairs = [cards[index : index + 2] for index in range(0, len(cards) - 1, 2)]
    if len(cards) % 2:
        # The queue serializes this overlapping pair with its neighbor. The
        # final card must not silently disappear on an odd-sized host.
        pairs.append(cards[-2:])
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-soc", choices=("PR", "DT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    query = subprocess.run(["npu-smi", "info", "-l"], capture_output=True, text=True, check=False, timeout=30)
    inventory = query.stdout + query.stderr
    if query.returncode != 0:
        query = subprocess.run(
            ["task-submit", "--timeout", "90", "--run", "npu-smi info -l"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        inventory += "\nPrivileged query:\n" + query.stdout + query.stderr
    (output / "inventory.log").write_text(inventory)
    if query.returncode != 0:
        raise RuntimeError("Cannot enumerate the host cards; inspect inventory.log")
    pairs = card_pairs(query.stdout)
    (output / "coverage.json").write_text(json.dumps({"planned_pairs": pairs}, indent=2) + "\n")
    print(f"Testing all discovered cards in pairs: {pairs}", flush=True)
    results = []

    def execute(pair):
        devices, pair_output = pair
        command = (
            # task-submit inherits this controller's activated environment.
            # Re-sourcing would prepend loader paths again and invalidate the
            # JIT toolchain fingerprint of the prewarmed READY artifacts.
            f"cd {shlex.quote(str(ROOT))} && "
            "bash runtime/.claude/skills/onboard-arch-precheck/check.sh a5 && "
            f"{shlex.quote(sys.executable)} tests/diagnostics/run_urma_mfe.py "
            f'--devices "$TASK_DEVICE" --expected-soc {args.expected_soc} '
            f"--output {shlex.quote(str(pair_output))}"
        )
        code = run_logged(
            ["task-submit", "--device", devices, "--timeout", "4800", "--max-time", "3300", "--run", command],
            pair_output / "queue.log",
            f"pair={devices} queue/allocation/execution",
            4920,
        )
        return {"devices": devices, "phase": "execute", "returncode": code}

    stop = threading.Event()
    monitor = threading.Thread(target=watch_results, args=(output, stop), daemon=True)
    monitor.start()
    try:
        # Submit immediately after each warmup: first-pair evidence must not wait
        # for every other pair's compilation. The queue still owns device locks.
        with ThreadPoolExecutor(max_workers=len(pairs)) as executor:
            futures = []
            for pair in pairs:
                devices = ",".join(map(str, pair))
                pair_output = output / f"pair-{devices.replace(',', '-')}"
                pair_output.mkdir(parents=True, exist_ok=True)
                code = run_logged(
                    [sys.executable, "-u", "tests/diagnostics/prepare_urma_mfe.py", "--devices", devices],
                    pair_output / "prepare.log",
                    f"pair={devices} warmup",
                    600,
                )
                if code == 0:
                    futures.append(executor.submit(execute, (devices, pair_output)))
                else:
                    results.append({"devices": devices, "phase": "prepare", "returncode": code})
                (output / "pair-results.json").write_text(json.dumps(results, indent=2) + "\n")
            for future in as_completed(futures):
                results.append(future.result())
                (output / "pair-results.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        stop.set()
        monitor.join(timeout=5)
    summaries = [path.read_text() for path in sorted(output.glob("pair-*/summary.txt"))]
    (output / "summary.txt").write_text("\n".join(summaries) + "\n" + json.dumps(results, indent=2) + "\n")
    print((output / "summary.txt").read_text(), flush=True)
    return 0 if len(results) == len(pairs) and all(result["returncode"] == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

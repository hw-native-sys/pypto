# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Discover every host card, warm each pair, then queue independent pair diagnostics."""

import argparse
import json
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
    prepared = []
    results = []
    for pair in pairs:
        devices = ",".join(map(str, pair))
        pair_output = output / f"pair-{devices.replace(',', '-')}"
        pair_output.mkdir(parents=True, exist_ok=True)
        with (pair_output / "prepare.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, "tests/diagnostics/prepare_urma_mfe.py", "--devices", devices],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0:
            prepared.append((devices, pair_output))
        else:
            results.append({"devices": devices, "phase": "prepare", "returncode": result.returncode})

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
        with (pair_output / "queue.log").open("w") as log:
            result = subprocess.run(
                [
                    "task-submit",
                    "--device",
                    devices,
                    "--timeout",
                    "4800",
                    "--max-time",
                    "3300",
                    "--run",
                    command,
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return {"devices": devices, "phase": "execute", "returncode": result.returncode}

    # Disjoint pairs may execute concurrently; task-submit owns all locks and
    # arbitrates overlap or cards occupied by other users. No reset or eviction.
    with ThreadPoolExecutor(max_workers=max(1, len(prepared))) as executor:
        for result in executor.map(execute, prepared):
            results.append(result)
            (output / "pair-results.json").write_text(json.dumps(results, indent=2) + "\n")
    summaries = [path.read_text() for path in sorted(output.glob("pair-*/summary.txt"))]
    (output / "summary.txt").write_text("\n".join(summaries) + "\n" + json.dumps(results, indent=2) + "\n")
    print((output / "summary.txt").read_text(), flush=True)
    return 0 if len(results) == len(pairs) and all(result["returncode"] == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

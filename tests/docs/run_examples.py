# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run the example ladders on the simulator, several at a time.

Every example must compile AND run with a passing golden, so the teaching
material cannot silently rot. Two ladders are covered:

- ``examples/distributed`` — run at the default rank pair, plus every alternate
  mode and rank count the walkthroughs document. A documented invocation that
  nothing runs is teaching material nobody checks.
- ``examples/{beginner,intermediate,advanced,utils}`` — the manual's
  beginner/intermediate/advanced/utils citations. They default to ``a2a3sim``
  and take no arguments.

Each invocation is an independent process: its compile output goes to a fresh
``mkdtemp`` under ``build_output``, its reusable binaries are guarded by
``binary_context_lock``, and a distributed run's shared-memory segment is named
after its own driver PID (``comm_sim.cpp``'s ``make_shm_name``). Nothing is
shared, so ``--jobs`` simply runs N of them at once.

``--jobs`` defaults to 1: the width belongs to the caller, who knows what else
the machine is doing. CI passes it explicitly.

Usage:
    python tests/docs/run_examples.py [--jobs 16] [-p a2a3sim]
    python tests/docs/run_examples.py --list        # show what would run
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_TIMEOUT_SECONDS = 1800
_FAILURE_TAIL_LINES = 25

#: Ladders whose every example runs bare — no platform, no ranks.
_TEACHING_DIRS = ("beginner", "intermediate", "advanced", "utils")

#: The default rank pair every distributed example is run at.
_DEFAULT_RANKS = ("-d", "0,1")

#: Distributed invocations beyond the default-mode P=2 pass above. Every mode
#: and rank count the walkthroughs document has to be CI-validated too, or the
#: teaching material rots.
_DISTRIBUTED_EXTRAS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Alternate modes documented alongside the default one.
    ("04_barrier.py", ("-d", "0,1", "--use-builtin")),
    ("05_remote_load_store.py", ("-d", "0,1", "--mode", "store")),
    ("06_put_get.py", ("-d", "0,1", "--mode", "get")),
    # Step 07 is rank-count-agnostic: the P=3 and P=4 legs are the committed
    # >=3-rank coverage for the general ring pattern (P=2 ran in the default
    # pass above).
    ("07_dynamic_rank_count.py", ("-d", "0,1,2")),
    ("07_dynamic_rank_count.py", ("-d", "0,1,2,3")),
    # Steps 08-11: the all-reduce comparisons are only observable at P=4 (P=2
    # collapses mesh/two-phase/ring into the same exchange); the reveal runs
    # both modes.
    ("08_allreduce_mesh.py", ("-d", "0,1,2,3")),
    ("09_allreduce_two_phase.py", ("-d", "0,1,2,3")),
    ("10_allreduce_ring.py", ("-d", "0,1,2,3")),
    ("11_allreduce_reveal.py", ("-d", "0,1,2,3")),
    ("11_allreduce_reveal.py", ("-d", "0,1,2,3", "--mode", "ring")),
    # Steps 12-15: the collective zoo. P=2 default ran in the pass above; add
    # the P=2 builtin reveal and the P=4 legs (both modes; the per-rank
    # patterns are most distinct at P=4).
    ("12_broadcast.py", ("-d", "0,1", "--mode", "builtin")),
    ("12_broadcast.py", ("-d", "0,1,2,3")),
    ("12_broadcast.py", ("-d", "0,1,2,3", "--mode", "builtin")),
    ("13_allgather.py", ("-d", "0,1", "--mode", "builtin")),
    ("13_allgather.py", ("-d", "0,1,2,3")),
    ("13_allgather.py", ("-d", "0,1,2,3", "--mode", "builtin")),
    ("14_reduce_scatter.py", ("-d", "0,1", "--mode", "builtin")),
    ("14_reduce_scatter.py", ("-d", "0,1,2,3")),
    ("14_reduce_scatter.py", ("-d", "0,1,2,3", "--mode", "builtin")),
    ("15_all_to_all.py", ("-d", "0,1", "--mode", "builtin")),
    ("15_all_to_all.py", ("-d", "0,1,2,3")),
    ("15_all_to_all.py", ("-d", "0,1,2,3", "--mode", "builtin")),
    # Step 16: the composition capstone at P=4.
    ("16_putting_it_together.py", ("-d", "0,1,2,3")),
)


def _ladder_scripts(directory: str) -> list[Path]:
    """Return the runnable examples of ``examples/<directory>``, in step order."""
    root = REPO_ROOT / "examples" / directory
    if not root.is_dir():
        raise FileNotFoundError(f"No such example ladder: {root}")
    return sorted(p for p in root.glob("*.py") if p.name != "__init__.py")


def invocations(platform: str) -> list[list[str]]:
    """Build every example invocation, as argv tails relative to the repo root.

    Args:
        platform: Substituted for the distributed ladder's ``-p``. The teaching
            ladders take no arguments at all, so it does not reach them.

    Returns:
        One argv per invocation, script path first. Ordered longest-ladder
        first so the slowest work is dealt out before the short tail.
    """
    sim = ["-p", platform]
    out: list[list[str]] = []

    # Globbed, not listed: a newly added step is covered the day it lands.
    for script in _ladder_scripts("distributed"):
        out.append([script.relative_to(REPO_ROOT).as_posix(), *sim, *_DEFAULT_RANKS])
    for name, extra in _DISTRIBUTED_EXTRAS:
        script = REPO_ROOT / "examples" / "distributed" / name
        if not script.is_file():
            raise FileNotFoundError(f"Documented invocation names a missing example: {script}")
        out.append([script.relative_to(REPO_ROOT).as_posix(), *sim, *extra])

    for directory in _TEACHING_DIRS:
        for script in _ladder_scripts(directory):
            out.append([script.relative_to(REPO_ROOT).as_posix()])

    return out


def run_one(argv: list[str]) -> tuple[bool, float, str]:
    """Execute one invocation; return (passed, seconds, output tail)."""
    started = time.monotonic()
    proc = subprocess.run(  # noqa: S603 - fixed argv, paths from the repo tree
        [sys.executable, *argv],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=_TIMEOUT_SECONDS,
        check=False,
    )
    elapsed = time.monotonic() - started
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    return proc.returncode == 0, elapsed, "\n".join(tail[-_FAILURE_TAIL_LINES:])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the example ladders on the simulator.")
    parser.add_argument("-p", "--platform", default="a2a3sim", help="platform for the distributed ladder")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="how many examples to run at once (default: 1). Each is an independent process.",
    )
    parser.add_argument("--list", action="store_true", help="list the invocations and exit")
    args = parser.parse_args()

    if args.jobs < 1:
        print(f"--jobs must be at least 1, got {args.jobs}", file=sys.stderr)
        return 1

    todo = invocations(args.platform)

    if args.list:
        for argv in todo:
            print(f"  would run  {' '.join(argv)}")
        print(f"\n{len(todo)} invocation(s)")
        return 0

    failures: list[tuple[str, str]] = []
    started = time.monotonic()
    # Results are printed as they land, so a long run shows progress rather
    # than going silent; the failing tails are replayed at the end, where they
    # are not buried under the passes that finished after them.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, argv): argv for argv in todo}
        for future in as_completed(futures):
            label = " ".join(futures[future])
            ok, elapsed, tail = future.result()
            print(f"{'PASS' if ok else 'FAIL'}  {elapsed:6.1f}s  {label}", flush=True)
            if not ok:
                failures.append((label, tail))

    for label, tail in failures:
        print(f"\n=== {label}\n{tail}", file=sys.stderr)
    wall = time.monotonic() - started
    passed = len(todo) - len(failures)
    print(f"\n{passed}/{len(todo)} example invocations passed in {wall:.1f}s at -j {args.jobs}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run the pypto-lib model matrix, several task-submit leases at a time.

Every model must still compile AND run with a passing golden on real cards, so
a pypto change that breaks a shipped model is caught before it reaches
pypto-serving. The matrix lives here rather than in ci.yml so that one command
reproduces any of it locally, and so the drift below can be checked at all.

Why several at a time: a lease holds its cards for the whole ``python
models/...`` invocation -- venv boot, compile, the torch reference golden,
the device run, validation -- but only the device run needs a card. Measured
over four main runs, 63-83% of the held time was card-free work, the torch
golden alone being 400s of a ~615s job, while the queue wait for a card was
~0. The serialisation was the cost, not card scarcity.

Nothing is shared between two leases: each compile lands in its own mkdtemp
under the clone's ``build_output``, inputs and goldens live under that same
work_dir, and pypto's JIT artifact store is flock-guarded with no-replace
renames. So they simply run at once, bounded by a card budget.

Unlike the old per-model workflow steps, a failure no longer stops the matrix:
every model runs and every failure is reported, which costs a few card-seconds
on a broken tree and saves a round trip per additional breakage.

Usage:
    python tests/models/run_lib_models.py --lib-root ../pypto-lib [--cards 6]
    python tests/models/run_lib_models.py --list       # the exact task-submit lines
"""

import argparse
import contextlib
import dataclasses
import difflib
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

#: How long task-submit may wait for the cards, and how long the task itself may
#: run once it has them. Both match what the per-model workflow steps passed.
_QUEUE_TIMEOUT_SECONDS = 3600
_TASK_MAX_TIME_SECONDS = 1800

#: Python-side backstop above task-submit's own ``--timeout``, so a wedged
#: task-submit cannot hang the job forever. Killing it here orphans the queued
#: task, which the workflow's `kill-orphaned-tasks` step reaps.
_WATCHDOG_SLACK_SECONDS = 600

#: A model needing more cards than this is a host-impact change that pypto has
#: to make deliberately, not one pypto-lib can land on pypto's behalf.
_MAX_CARDS_PER_MODEL = 4

#: Cards this job may hold at once, summed over every in-flight lease. Measured
#: by list-scheduling the four runs above: 6 already reaches the critical path
#: (prefill_csa, ~250s), and 8 or 13 are not faster -- so this is the smallest
#: width that costs nothing, and it leaves 10 of the host's 16 cards alone.
_DEFAULT_CARD_BUDGET = 6

#: `# ci: devices=N` near the top of a model file. Same expression pypto-lib's
#: own CI greps for, so both repos read one marker the same way.
_DEVICE_MARKER = re.compile(r"#\s*ci:\s*devices=(\d+)")


@dataclasses.dataclass(frozen=True)
class Model:
    """One CI leg: a model in the pypto-lib clone, run with its own cards."""

    #: Path inside the pypto-lib clone.
    path: str
    #: argv tail beyond the shared `-p <platform> -d $TASK_DEVICE`.
    extra: tuple[str, ...]
    #: Cards the `# ci: devices=N` marker stated when this leg was reviewed.
    #: The marker in the clone is what actually runs; this is the expectation
    #: a drift is reported against.
    cards: int
    #: Why this leg is in the matrix at all.
    why: str


#: The matrix, in report order. Submission order is by card count (see
#: `_submission_order`); this order is what `--list` prints and what the logs
#: are replayed in, so it stays aligned with the workflow it replaced.
_MATRIX: tuple[Model, ...] = (
    Model(
        "models/qwen3_14b/decode_fwd.py",
        (),
        1,
        "the shipped Qwen3-14B decode path, end to end on one card",
    ),
    Model(
        "models/qwen3_14b/prefill_fwd.py",
        (),
        1,
        "the dynamic-shape host path decode does not exercise: the token axis is a "
        "pl.dynamic() symbol, the per-layer temp is sized from pl.tensor.dim(...), and "
        "the layer result is reassigned to the same loop-carried var. Cross-function "
        "dynamic-shape typing regressions surface here, and only here, before serving",
    ),
    Model("models/deepseek_v4_flash_mtp/decode_swa.py", (), 1, "sliding-window attention decode"),
    Model("models/deepseek_v4_flash_mtp/decode_hca.py", (), 1, "hierarchical compressed attention decode"),
    Model("models/deepseek_v4_flash_mtp/decode_csa.py", (), 1, "compressed sparse attention decode"),
    Model(
        "models/deepseek_v4_flash_mtp/prefill_csa.py",
        (),
        2,
        "context-parallel prefill; the longest leg of the matrix and so the one "
        "that sets the job's wall clock once the legs overlap",
    ),
    Model("models/deepseek_v4_flash_mtp/prefill_hca.py", (), 2, "context-parallel prefill, HCA variant"),
    Model("models/deepseek_v4_flash_mtp/prefill_swa.py", (), 2, "context-parallel prefill, SWA variant"),
    Model(
        "models/deepseek_v4_flash_mtp/decode_moe.py",
        ("--ep", "2"),
        2,
        "EP=2 routed experts -- the only multi-card decode leg, and the only one "
        "whose world size comes from an argument rather than the fixture",
    ),
)


def read_device_marker(source: str) -> int:
    """Cards the `# ci: devices=N` marker asks for; 1 when a file carries none.

    Args:
        source: Full text of a pypto-lib model file.

    Returns:
        The first marker's value, or 1. Matching the first occurrence over the
        whole file is what pypto-lib's own CI does, so both repos agree even
        when a model mentions the marker again further down.
    """
    match = _DEVICE_MARKER.search(source)
    return int(match.group(1)) if match else 1


def _suggest_rename(missing: Path) -> str | None:
    """Closest surviving basename to a model that has moved, if there is one.

    pypto-lib renames its models without any change on this side -- `moe.py` ->
    `decode_moe.py`, and a directory flattening before that -- so a missing path
    is much more often a rename than a deletion.
    """
    candidates: list[Path] = []
    if missing.parent.is_dir():
        candidates = sorted(missing.parent.glob("*.py"))
    else:
        # The directory itself moved: look for the same basename anywhere under
        # the models tree before giving up.
        for models_root in (missing.parents[1], missing.parents[2]):
            if models_root.is_dir():
                candidates = sorted(models_root.rglob(missing.name))
                break
    if not candidates:
        return None
    names = [path.name for path in candidates]
    close = difflib.get_close_matches(missing.name, names, n=1, cutoff=0.5)
    if not close:
        return None
    return next(path.name for path in candidates if path.name == close[0])


def preflight(lib_root: Path, matrix: tuple[Model, ...] = _MATRIX) -> None:
    """Reject a matrix the clone has moved on from, before any card is borrowed.

    Args:
        lib_root: The pypto-lib clone.
        matrix: Models to check.

    Raises:
        FileNotFoundError: One or more models are no longer in the clone. Every
            missing entry is reported at once -- fixing them one CI round trip
            at a time is what this replaces.
    """
    missing: list[str] = []
    for model in matrix:
        path = lib_root / model.path
        if path.is_file():
            continue
        suggestion = _suggest_rename(path)
        hint = f"   (renamed to {path.parent.name}/{suggestion}?)" if suggestion else ""
        missing.append(f"  {model.path}{hint}")
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} model(s) named by the matrix are not in {lib_root}:\n"
            + "\n".join(missing)
            + "\n\nUpdate _MATRIX in tests/models/run_lib_models.py: pypto-lib moves its "
            "models without any change on this side."
        )


def resolve_cards(lib_root: Path, model: Model, *, budget: int, warn: Callable[[str], None]) -> int:
    """Cards to borrow for *model*, taken from the marker in the clone.

    The marker is the authority -- borrowing too few does not degrade
    gracefully, it aborts with `CP2 requires 2 devices, got [1]` -- while
    ``Model.cards`` is the value a reviewer last agreed to. A difference is
    reported and the marker wins.

    Args:
        lib_root: The pypto-lib clone.
        model: Matrix entry to resolve.
        budget: Total cards this run may hold, so a request nothing can ever
            schedule is rejected here rather than deadlocking the budget.
        warn: Sink for the drift notice.

    Returns:
        Cards the marker asks for.

    Raises:
        ValueError: The marker asks for more than `_MAX_CARDS_PER_MODEL` or
            more than the whole budget.
    """
    actual = read_device_marker((lib_root / model.path).read_text(encoding="utf-8", errors="replace"))
    if actual != model.cards:
        warn(
            f"{model.path}: marker says devices={actual}, the matrix records "
            f"{model.cards} -- using {actual}. Update Model.cards once this is reviewed."
        )
    ceiling = min(_MAX_CARDS_PER_MODEL, budget)
    if actual > ceiling:
        raise ValueError(
            f"{model.path} asks for {actual} cards, above the {ceiling}-card ceiling "
            f"(max-per-model {_MAX_CARDS_PER_MODEL}, budget {budget}). Raising either is a "
            "host-impact decision -- make it deliberately rather than inheriting it."
        )
    return actual


def submit_argv(
    model: Model,
    *,
    lib_root: Path,
    activate: Path,
    device: str,
    cards: int,
    platform: str,
) -> list[str]:
    """The task-submit lease for one model, as an argv.

    The inner `--run` string is handed to task-submit as a single argument and
    run by the queue's own shell, which is where `$TASK_DEVICE` -- the cards it
    lent, one id or a comma-separated set -- expands. Nothing here goes through
    a shell of ours.
    """
    device_num = ["--device-num", str(cards)] if cards > 1 else []
    inner = f"source {activate} && cd {lib_root} && python {model.path} -p {platform} -d $TASK_DEVICE" + (
        "".join(f" {arg}" for arg in model.extra)
    )
    return [
        "task-submit",
        "--device",
        device,
        *device_num,
        "--timeout",
        str(_QUEUE_TIMEOUT_SECONDS),
        "--max-time",
        str(_TASK_MAX_TIME_SECONDS),
        "--run",
        inner,
    ]


def child_env(lib_root: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a lease: PYTHONPATH is the clone, and only the clone.

    The inner command re-sources activate.sh, which *prepends* the job venv's
    site-packages, so the child ends up with `<venv>:<clone>`. Prepending the
    clone to an inherited PYTHONPATH instead would leave whatever the outer
    shell had in front of the venv -- which is how a checkout's stale build
    silently shadows the installed pypto.
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = str(lib_root)
    return env


class CardBudget:
    """All-or-nothing card accounting across the in-flight leases.

    Granting partial cards would let two 2-card leases each hold one and wait
    forever for the other, so a lease takes every card it needs or none.

    `acquire` and `release` are separate rather than only a context manager
    because the driver acquires in submission order on one thread and releases
    on the worker that finishes -- which is what keeps the widest leases from
    being starved by a stream of narrow ones.
    """

    def __init__(self, total: int) -> None:
        self._total = total
        self._free = total
        self._peak = 0
        self._condition = threading.Condition()

    @property
    def free(self) -> int:
        with self._condition:
            return self._free

    @property
    def peak(self) -> int:
        """Most cards held at once over this budget's life."""
        with self._condition:
            return self._peak

    def acquire(self, cards: int) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self._free >= cards)
            self._free -= cards
            self._peak = max(self._peak, self._total - self._free)

    def release(self, cards: int) -> None:
        with self._condition:
            self._free += cards
            self._condition.notify_all()

    @contextlib.contextmanager
    def hold(self, cards: int) -> Iterator[None]:
        self.acquire(cards)
        try:
            yield
        finally:
            self.release(cards)


def run_lease(argv: list[str], log_path: Path, env: dict[str, str]) -> int:
    """Run one task-submit lease, capturing its output; return the exit code.

    A lease that has to be killed is a failure, not a missing result: reporting
    "no exit code" as success is how a task that never ran gets a green tick.
    """
    with log_path.open("wb") as log:
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=_QUEUE_TIMEOUT_SECONDS + _WATCHDOG_SLACK_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.write(
                f"\n[run_lib_models] killed after "
                f"{_QUEUE_TIMEOUT_SECONDS + _WATCHDOG_SLACK_SECONDS}s; the queued task may "
                f"outlive it (the workflow's kill-orphaned-tasks step reaps those)\n".encode()
            )
            return 124
        except OSError as error:
            log.write(f"\n[run_lib_models] could not launch task-submit: {error}\n".encode())
            return 127
    return completed.returncode


def _submission_order(plan: list[tuple[Model, int]]) -> list[tuple[Model, int]]:
    """Widest leases first, so the long poles start and the narrow ones fill in.

    Python's sort is stable, so equal-width models keep matrix order. Starting
    the 1-card leases first would let them hold the budget below what a 2-card
    lease needs while the longest models wait.
    """
    return sorted(plan, key=lambda entry: -entry[1])


@dataclasses.dataclass(frozen=True)
class Result:
    """What one lease did."""

    model: Model
    cards: int
    returncode: int
    seconds: float
    log: Path

    @property
    def passed(self) -> bool:
        return self.returncode == 0


#: A lease runner: argv, where to capture its output, the environment to run it
#: in -> exit code. Injected so the scheduling can be tested without a card.
Submitter = Callable[[list[str], Path, dict[str, str]], int]


class _Annotator:
    """GitHub workflow commands, or plain lines when nothing is reading them."""

    def __init__(self, github: bool, sink: Callable[[str], None]) -> None:
        self._github = github
        self._sink = sink

    def warning(self, file: str, message: str) -> None:
        self._sink(f"::warning file={file}::{message}" if self._github else f"WARNING  {message}")

    def error(self, file: str, message: str) -> None:
        self._sink(f"::error file={file}::{message}" if self._github else f"ERROR  {message}")

    @contextlib.contextmanager
    def group(self, title: str) -> Iterator[None]:
        self._sink(f"::group::{title}" if self._github else f"----- {title}")
        try:
            yield
        finally:
            if self._github:
                self._sink("::endgroup::")


def _log_path(log_dir: Path, model: Model) -> Path:
    """Where one lease's captured output lands. Flattened, so two models with
    the same basename in different directories cannot share a file."""
    return log_dir / (model.path.replace("/", "_") + ".log")


def run_matrix(
    *,
    lib_root: Path,
    activate: Path,
    device: str,
    budget: int,
    platform: str,
    log_dir: Path,
    matrix: tuple[Model, ...] = _MATRIX,
    submit: Submitter = run_lease,
    annotator: _Annotator | None = None,
    report: Callable[[str], None] = print,
) -> list[Result]:
    """Run every model in *matrix*, at most *budget* cards in flight.

    Args:
        lib_root: The pypto-lib clone the models are read and run from.
        activate: activate.sh the lease sources before running python.
        device: Passed straight to `task-submit --device`. Concurrency only
            materialises on `auto`; a fixed card id serialises every lease on
            that one card's lock, which is slower but still correct.
        budget: Cards held at once, summed over in-flight leases.
        platform: `-p` for every model.
        log_dir: Directory for the per-lease capture files.
        matrix: Models to run.
        submit: Lease runner; injected by the tests.
        annotator: Workflow-command sink.
        report: Where progress lines go.

    Returns:
        One Result per model, in *matrix* order.

    Raises:
        FileNotFoundError: A model named by *matrix* is not in the clone.
        ValueError: A model's marker asks for more cards than it may have.
    """
    notes = annotator or _Annotator(github=False, sink=report)
    preflight(lib_root, matrix)
    log_dir.mkdir(parents=True, exist_ok=True)

    plan = [
        (
            model,
            resolve_cards(
                lib_root, model, budget=budget, warn=lambda message, m=model: notes.warning(m.path, message)
            ),
        )
        for model in matrix
    ]
    env = child_env(lib_root)
    accounting = CardBudget(budget)
    results: dict[str, Result] = {}
    results_lock = threading.Lock()

    def _lease(model: Model, cards: int) -> None:
        """Run one model, then hand its cards back."""
        argv = submit_argv(
            model,
            lib_root=lib_root,
            activate=activate,
            device=device,
            cards=cards,
            platform=platform,
        )
        log = _log_path(log_dir, model)
        started = time.monotonic()
        try:
            try:
                code = submit(argv, log, env)
            except Exception as error:  # noqa: BLE001 - one lease must not take the matrix down
                # Losing this to a swallowed Future would leave the model with
                # no Result at all, which the fallback below reports as a
                # failure -- correct, but with nothing to debug from.
                code = 1
                with log.open("a", encoding="utf-8") as sink:
                    sink.write(f"\n[run_lib_models] lease raised: {error!r}\n")
        finally:
            accounting.release(cards)
        elapsed = time.monotonic() - started
        with results_lock:
            results[model.path] = Result(model, cards, code, elapsed, log)
        report(f"{'PASS' if code == 0 else 'FAIL'}  {elapsed:7.1f}s  {cards}c  {model.path}")

    # Acquire on this thread, in submission order, so the widest leases are
    # never starved: a worker only starts once its cards are already reserved.
    with ThreadPoolExecutor(max_workers=max(len(plan), 1)) as pool:
        for model, cards in _submission_order(plan):
            accounting.acquire(cards)
            report(f"[submit] {model.path} (cards={cards})")
            pool.submit(_lease, model, cards)

    # A worker that died before recording a Result left no exit code; treating
    # that as success is how a model that never ran gets a green tick.
    ordered = [
        results.get(model.path) or Result(model, cards, 1, 0.0, _log_path(log_dir, model))
        for model, cards in plan
    ]
    report(f"\npeak {accounting.peak}/{budget} cards held")
    return ordered


def _replay(results: list[Result], notes: _Annotator, report: Callable[[str], None]) -> None:
    """Print each lease's captured output, in matrix order.

    Concurrent leases share this step's stdout, so nothing is printed while they
    run -- interleaved lines would shred the groups and make every log unusable.
    """
    for result in results:
        verdict = "PASS" if result.passed else f"FAIL (exit {result.returncode})"
        with notes.group(f"{result.model.path}  {verdict}  {result.seconds:.1f}s"):
            if result.log.is_file():
                report(result.log.read_text(encoding="utf-8", errors="replace").rstrip("\n"))
            else:
                report("(no output captured -- the lease never started)")
        if not result.passed:
            notes.error(result.model.path, f"FAIL (exit {result.returncode})")


def _summarise(results: list[Result], budget: int, wall: float, report: Callable[[str], None]) -> None:
    """One table, so the per-step timings the workflow used to show survive."""
    report("")
    report(f"{'cards':>6}  {'wall':>8}  {'verdict':<8}  model")
    for result in results:
        report(
            f"{result.cards:>6}  {result.seconds:>7.1f}s  "
            f"{'PASS' if result.passed else 'FAIL':<8}  {result.model.path}"
        )
    card_seconds = sum(result.cards * result.seconds for result in results)
    passed = sum(1 for result in results if result.passed)
    report(
        f"\n{passed}/{len(results)} models passed in {wall:.1f}s "
        f"at a {budget}-card budget ({card_seconds:.0f} card-seconds)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pypto-lib model matrix on borrowed cards.")
    parser.add_argument("--lib-root", type=Path, help="the pypto-lib clone (required unless --list)")
    parser.add_argument("--activate", type=Path, help="activate.sh each lease sources")
    parser.add_argument(
        "--device",
        default="auto",
        help="task-submit --device (default: auto). A fixed card id serialises every lease on it.",
    )
    parser.add_argument(
        "-c",
        "--cards",
        type=int,
        default=_DEFAULT_CARD_BUDGET,
        help=f"cards held at once across all leases (default: {_DEFAULT_CARD_BUDGET})",
    )
    parser.add_argument("-p", "--platform", default="a2a3", help="platform passed to every model")
    parser.add_argument("--log-dir", type=Path, help="where lease logs are captured (default: a temp dir)")
    parser.add_argument(
        "--annotations",
        choices=("auto", "on", "off"),
        default="auto",
        help="emit GitHub workflow commands (default: auto, on under GITHUB_ACTIONS)",
    )
    parser.add_argument("--list", action="store_true", help="print the task-submit lines and exit")
    args = parser.parse_args(argv)

    if args.cards < 1:
        print(f"--cards must be at least 1, got {args.cards}", file=sys.stderr)
        return 2

    if args.list:
        lib_root = args.lib_root or Path("<pypto-lib>")
        activate = args.activate or Path("<activate.sh>")
        if args.lib_root is not None:
            # Given a clone, --list is also the cheap drift check: it reads the
            # same markers the run would, so it must refuse the same plan.
            try:
                preflight(args.lib_root)
            except FileNotFoundError as error:
                print(f"\n{error}", file=sys.stderr)
                return 1
        for model in _MATRIX:
            cards = (
                resolve_cards(lib_root, model, budget=args.cards, warn=lambda message: None)
                if args.lib_root
                else model.cards
            )
            argv_line = submit_argv(
                model,
                lib_root=lib_root,
                activate=activate,
                device=args.device,
                cards=cards,
                platform=args.platform,
            )
            print(" ".join(argv_line[:-1]) + f' "{argv_line[-1]}"')
        print(f"\n{len(_MATRIX)} model invocation(s)")
        return 0

    if args.lib_root is None or args.activate is None:
        print("--lib-root and --activate are required unless --list is given", file=sys.stderr)
        return 2

    github = args.annotations == "on" or (
        args.annotations == "auto" and bool(os.environ.get("GITHUB_ACTIONS"))
    )
    notes = _Annotator(github=github, sink=print)

    with contextlib.ExitStack() as stack:
        log_dir = args.log_dir
        if log_dir is None:
            log_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="pypto-lib-models-")))
        started = time.monotonic()
        try:
            results = run_matrix(
                lib_root=args.lib_root.resolve(),
                activate=args.activate.resolve(),
                device=args.device,
                budget=args.cards,
                platform=args.platform,
                log_dir=log_dir,
                annotator=notes,
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"\n{error}", file=sys.stderr)
            return 1
        wall = time.monotonic() - started
        _replay(results, notes, print)
        _summarise(results, args.cards, wall, print)

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())

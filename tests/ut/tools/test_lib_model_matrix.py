# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for the pypto-lib model matrix runner.

The runner borrows real NPUs in CI, so the properties that decide how many and
on whose behalf are pinned here instead of only on hardware: the invocation set
it submits, the card count it reads out of pypto-lib, the budget it never
exceeds, and the two ways a lease can fail without being reported green.
"""

import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def runner(monkeypatch):
    """Import tests/models/run_lib_models.py, which is a script rather than a package."""
    path = _REPO_ROOT / "tests" / "models" / "run_lib_models.py"
    spec = importlib.util.spec_from_file_location("pypto_lib_model_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clone(runner, tmp_path):
    """A stand-in pypto-lib clone carrying each matrix model's reviewed marker."""
    root = tmp_path / "pypto-lib"
    for model in runner._MATRIX:
        path = root / model.path
        path.parent.mkdir(parents=True, exist_ok=True)
        marker = f"# ci: devices={model.cards}\n" if model.cards > 1 else ""
        path.write_text(f'"""{model.path}"""\n{marker}', encoding="utf-8")
    return root


class _FakeQueue:
    """A task-submit stand-in that records what was asked of it.

    Cards come out of the argv rather than being handed in, so the recorded
    concurrency also proves the lease actually carries `--device-num`.
    """

    def __init__(self, *, exit_codes: dict[str, int] | None = None, raises: str | None = None) -> None:
        self._exit_codes = exit_codes or {}
        self._raises = raises
        self._lock = threading.Lock()
        self.calls: list[list[str]] = []
        self.in_flight = 0
        self.peak = 0

    def __call__(self, argv: list[str], log: Path, env: dict[str, str]) -> int:
        cards = int(argv[argv.index("--device-num") + 1]) if "--device-num" in argv else 1
        model = argv[-1].rsplit("python ", 1)[1].split(" ")[0]
        with self._lock:
            self.calls.append(argv)
            self.in_flight += cards
            self.peak = max(self.peak, self.in_flight)
        try:
            log.write_text(f"fake queue ran {model}\n", encoding="utf-8")
            time.sleep(0.02)
            if self._raises is not None and model.endswith(self._raises):
                raise RuntimeError("task-submit is not on PATH")
            return self._exit_codes.get(model, 0)
        finally:
            with self._lock:
                self.in_flight -= cards


def _run(runner, clone, fake, **overrides):
    kwargs = {
        "lib_root": clone,
        "activate": Path("/ws/activate.sh"),
        "device": "auto",
        "budget": 6,
        "platform": "a2a3",
        "log_dir": clone.parent / "logs",
        "submit": fake,
        "report": lambda _line: None,
    }
    kwargs.update(overrides)
    return runner.run_matrix(**kwargs)


# --------------------------------------------------------------------------- the invocation set


def test_matrix_is_the_invocation_set_the_workflow_ran(runner):
    """The matrix replaced nine workflow steps; a silent edit to it is a silent
    change to what per-PR CI proves about the shipped models."""
    assert [(m.path, m.extra, m.cards) for m in runner._MATRIX] == [
        ("models/qwen3_14b/decode_fwd.py", (), 1),
        ("models/qwen3_14b/prefill_fwd.py", (), 1),
        ("models/deepseek_v4_flash_mtp/decode_swa.py", (), 1),
        ("models/deepseek_v4_flash_mtp/decode_hca.py", (), 1),
        ("models/deepseek_v4_flash_mtp/decode_csa.py", (), 1),
        ("models/deepseek_v4_flash_mtp/prefill_csa.py", (), 2),
        ("models/deepseek_v4_flash_mtp/prefill_hca.py", (), 2),
        ("models/deepseek_v4_flash_mtp/prefill_swa.py", (), 2),
        ("models/deepseek_v4_flash_mtp/decode_moe.py", ("--ep", "2"), 2),
    ]
    assert all(model.why for model in runner._MATRIX), "every leg states why it is in the matrix"


def test_lease_is_the_command_the_workflow_step_ran(runner):
    """One card: no --device-num, and the cards task-submit lends expand inside
    the queue's own shell, not ours."""
    argv = runner.submit_argv(
        runner._MATRIX[0],
        lib_root=Path("/ws/pypto-lib"),
        activate=Path("/ws/lib-checkout/activate.sh"),
        device="auto",
        cards=1,
        platform="a2a3",
    )
    assert argv == [
        "task-submit",
        "--device",
        "auto",
        "--timeout",
        "3600",
        "--max-time",
        "1800",
        "--run",
        "source /ws/lib-checkout/activate.sh && cd /ws/pypto-lib && "
        "python models/qwen3_14b/decode_fwd.py -p a2a3 -d $TASK_DEVICE",
    ]


def test_multi_card_lease_borrows_and_passes_its_extra_argv(runner):
    argv = runner.submit_argv(
        runner._MATRIX[-1],
        lib_root=Path("/ws/pypto-lib"),
        activate=Path("/ws/a.sh"),
        device="auto",
        cards=2,
        platform="a2a3",
    )
    assert argv[:5] == ["task-submit", "--device", "auto", "--device-num", "2"]
    assert argv[-1].endswith(
        "python models/deepseek_v4_flash_mtp/decode_moe.py -p a2a3 -d $TASK_DEVICE --ep 2"
    )


# --------------------------------------------------------------------------- the marker


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('"""A model with no marker."""\nimport sys\n', 1),
        ("# ci: devices=2\n", 2),
        ('"""Docstring first."""\n# ci: devices=4\n', 4),
        ("#ci:devices=8\n", 8),
        ("# ci: devices=2\n# ci: devices=4\n", 2),
    ],
)
def test_marker_parsing(runner, source, expected):
    """Same first-match-wins reading as pypto-lib's own CI grep, so one marker
    means one thing in both repos."""
    assert runner.read_device_marker(source) == expected


def test_marker_wins_over_the_reviewed_count_and_says_so(runner, clone):
    """Borrowing too few cards does not degrade -- the model aborts with
    `CP2 requires 2 devices, got [1]` -- so the clone decides and we report it."""
    model = runner._MATRIX[0]
    (clone / model.path).write_text("# ci: devices=2\n", encoding="utf-8")
    warnings: list[str] = []
    assert runner.resolve_cards(clone, model, budget=6, warn=warnings.append) == 2
    assert len(warnings) == 1
    assert "marker says devices=2" in warnings[0]
    assert "the matrix records 1" in warnings[0]


def test_matching_marker_is_not_reported_as_drift(runner, clone):
    warnings: list[str] = []
    for model in runner._MATRIX:
        assert runner.resolve_cards(clone, model, budget=6, warn=warnings.append) == model.cards
    assert warnings == []


@pytest.mark.parametrize(("marker", "budget"), [(8, 6), (3, 2)])
def test_marker_above_the_ceiling_stops_the_run_before_any_lease(runner, clone, marker, budget):
    """A model that grew past the per-model ceiling, or past the whole budget,
    is a host-impact change pypto makes deliberately -- and one nothing could
    ever schedule, which would otherwise wedge the budget."""
    (clone / runner._MATRIX[0].path).write_text(f"# ci: devices={marker}\n", encoding="utf-8")
    fake = _FakeQueue()
    with pytest.raises(ValueError, match="cards, above the"):
        _run(runner, clone, fake, budget=budget)
    assert fake.calls == [], "no card may be borrowed once the plan is known to be unschedulable"


# --------------------------------------------------------------------------- preflight


def test_a_model_pypto_lib_moved_fails_before_any_lease(runner, clone):
    """pypto-lib renames its models without any change here. Failing in the
    plan costs nothing; failing per-step costs a queue wait and two cards."""
    (clone / "models/deepseek_v4_flash_mtp/prefill_hca.py").unlink()
    fake = _FakeQueue()
    with pytest.raises(FileNotFoundError, match="prefill_hca.py"):
        _run(runner, clone, fake)
    assert fake.calls == []


def test_every_missing_model_is_reported_at_once(runner, clone):
    """One CI round trip per missing model is exactly what this replaces."""
    (clone / "models/qwen3_14b/decode_fwd.py").unlink()
    (clone / "models/deepseek_v4_flash_mtp/decode_swa.py").unlink()
    with pytest.raises(FileNotFoundError) as caught:
        runner.preflight(clone)
    assert "2 model(s)" in str(caught.value)
    assert "decode_fwd.py" in str(caught.value)
    assert "decode_swa.py" in str(caught.value)


def test_a_renamed_model_is_named_in_the_error(runner, clone):
    """`moe.py` -> `decode_moe.py` is the rename this actually shipped against."""
    moved = clone / "models/deepseek_v4_flash_mtp/decode_moe.py"
    moved.rename(moved.with_name("decode_moe_v2.py"))
    with pytest.raises(FileNotFoundError, match="decode_moe_v2.py"):
        runner.preflight(clone)


# --------------------------------------------------------------------------- scheduling


def test_the_card_budget_is_never_exceeded(runner, clone):
    """The whole point of a budget: this job must not take the host's cards out
    from under the nine other runner instances sharing them."""
    for budget in (2, 4, 6):
        fake = _FakeQueue()
        results = _run(runner, clone, fake, budget=budget)
        assert fake.peak <= budget, f"held {fake.peak} cards against a budget of {budget}"
        assert len(results) == len(runner._MATRIX)
        assert all(result.passed for result in results)


def test_the_budget_is_actually_used(runner, clone):
    """A budget nothing reaches would make the test above vacuous."""
    fake = _FakeQueue()
    _run(runner, clone, fake, budget=6)
    assert fake.peak > 2, "leases are meant to overlap, not queue behind each other"


def test_widest_leases_are_submitted_first(runner, clone):
    """A stream of 1-card leases must not hold the budget below what a 2-card
    lease needs while the longest models wait for it."""
    plan = [(model, model.cards) for model in runner._MATRIX]
    widths = [cards for _model, cards in runner._submission_order(plan)]
    assert widths == sorted(widths, reverse=True)
    # Stable within a width, so the report order still tracks the matrix.
    two_card = [m.path for m, c in runner._submission_order(plan) if c == 2]
    assert two_card == [m.path for m in runner._MATRIX if m.cards == 2]


def test_budget_grants_all_cards_or_none(runner):
    """Partial grants would let two 2-card leases each hold one and wait forever."""
    budget = runner.CardBudget(3)
    budget.acquire(2)
    holder_started = threading.Event()

    def _second():
        holder_started.set()
        budget.acquire(2)
        budget.release(2)

    waiter = threading.Thread(target=_second)
    waiter.start()
    holder_started.wait(timeout=5)
    time.sleep(0.05)
    assert budget.free == 1, "the waiter must not have taken the single free card"
    budget.release(2)
    waiter.join(timeout=5)
    assert not waiter.is_alive()


# --------------------------------------------------------------------------- failure reporting


def test_a_failing_model_does_not_stop_the_others(runner, clone):
    """Sequential steps only ever showed the first failure; every leg now runs."""
    fake = _FakeQueue(exit_codes={"models/deepseek_v4_flash_mtp/decode_hca.py": 1})
    results = _run(runner, clone, fake)
    assert len(fake.calls) == len(runner._MATRIX)
    failed = [result.model.path for result in results if not result.passed]
    assert failed == ["models/deepseek_v4_flash_mtp/decode_hca.py"]


def test_results_come_back_in_matrix_order(runner, clone):
    """Logs are replayed in this order, so it has to be the matrix's, not the
    order nine concurrent leases happened to finish in."""
    results = _run(runner, clone, _FakeQueue())
    assert [result.model.path for result in results] == [m.path for m in runner._MATRIX]


def test_a_lease_that_raises_is_a_failure_not_a_missing_result(runner, clone):
    """A swallowed Future would leave the model with no exit code at all."""
    fake = _FakeQueue(raises="prefill_swa.py")
    results = _run(runner, clone, fake)
    broken = next(r for r in results if r.model.path.endswith("prefill_swa.py"))
    assert not broken.passed
    assert "task-submit is not on PATH" in broken.log.read_text(encoding="utf-8")
    assert all(r.passed for r in results if r is not broken)


def test_a_lease_that_recorded_nothing_is_a_failure(runner, clone):
    """Defaulting a missing exit code to 0 reports a model that never ran green."""

    def _never_records(argv, log, env):
        # A BaseException escapes the lease's `except Exception` and is parked
        # in the Future, so nothing ever records a Result -- the "worker died"
        # shape (runner OOM, cancellation) the fallback exists for.
        raise SystemExit(0)

    results = _run(runner, clone, _FakeQueue(), matrix=runner._MATRIX[3:4], submit=_never_records)
    assert len(results) == 1
    assert not results[0].passed
    assert results[0].returncode != 0


# --------------------------------------------------------------------------- environment


def test_pythonpath_is_the_clone_and_only_the_clone(runner):
    """The lease re-sources activate.sh, which prepends the venv's
    site-packages. Prepending to an inherited PYTHONPATH instead would leave a
    stale checkout in front of the installed pypto."""
    env = runner.child_env(Path("/ws/pypto-lib"), {"PYTHONPATH": "/somewhere/stale", "HOME": "/root"})
    assert env["PYTHONPATH"] == "/ws/pypto-lib"
    assert env["HOME"] == "/root", "the rest of the environment is passed through"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

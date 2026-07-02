# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the harness task-submit execute path.

``subprocess.run`` is mocked so these run without a device, without
``task-submit``, and without compiling anything. They pin the argv the harness
hands to ``task-submit``, the pass/fail classification, and the sim guard that
keeps simulator platforms off the borrow-a-card path.

The harness package (``harness.core.test_runner``) lives under ``tests/st``; add
that dir to ``sys.path`` so this device-free unit test can import it directly.
"""

import importlib
import json
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from pypto.runtime.runner import _DfxOpts

_ST_DIR = Path(__file__).resolve().parents[2] / "st"
if str(_ST_DIR) not in sys.path:
    sys.path.insert(0, str(_ST_DIR))

test_runner = importlib.import_module("harness.core.test_runner")


@pytest.fixture(autouse=True)
def _reset_pipeline_ctx():
    """Isolate the module-global pipeline ctx / pools between tests."""
    saved_ctx = dict(test_runner._pipeline_ctx)
    saved_pool = test_runner._device_pool
    test_runner._pipeline_ctx.clear()
    test_runner._executed_device.clear()
    yield
    test_runner._pipeline_ctx.clear()
    test_runner._pipeline_ctx.update(saved_ctx)
    # Direct assignment (not setattr); pyright sees the importlib-loaded module as
    # bare ModuleType, so ignore its spurious unknown-attribute error.
    test_runner._device_pool = saved_pool  # pyright: ignore[reportAttributeAccessIssue]


def _proc(returncode, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# _dfx_to_cli
# ---------------------------------------------------------------------------


def test_dfx_to_cli_empty_for_default():
    assert test_runner._dfx_to_cli(_DfxOpts()) == []


def test_dfx_to_cli_emits_only_enabled_flags():
    dfx = _DfxOpts(enable_l2_swimlane=True, enable_dump_tensor=2, enable_pmu=5, enable_dep_gen=True)
    argv = test_runner._dfx_to_cli(dfx)
    assert argv == [
        "--enable-l2-swimlane",
        "--dump-tensor",
        "2",
        "--enable-pmu",
        "5",
        "--enable-dep-gen",
    ]


# ---------------------------------------------------------------------------
# _parse_executed_device
# ---------------------------------------------------------------------------


def test_parse_executed_device():
    assert test_runner._parse_executed_device("noise\nPYPTO_EXEC_RESULT=PASS device=7\n") == 7
    assert test_runner._parse_executed_device("PYPTO_EXEC_RESULT=FAIL\n") is None


# ---------------------------------------------------------------------------
# _run_artifact_via_task_submit — argv + result handling
# ---------------------------------------------------------------------------


def test_task_submit_argv_and_pass(tmp_path):
    dfx = _DfxOpts(enable_l2_swimlane=True)
    with patch.object(
        test_runner.subprocess,
        "run",
        return_value=_proc(0, "PYPTO_EXEC_RESULT=PASS device=4\n"),
    ) as run:
        passed, error, device = test_runner._run_artifact_via_task_submit(
            tmp_path, "a2a3", dfx, "abc123", max_time=600, queue_timeout=1800
        )
    assert (passed, error, device) == (True, None, 4)
    argv = run.call_args.args[0]
    assert argv[0] == "task-submit"
    assert "--device" in argv and "auto" in argv
    assert "--timeout" in argv and "1800" in argv
    assert "--max-time" in argv and "600" in argv
    # No --env / --ptoas: rely on task-submit preserving the caller's env, so the
    # minimal CI task-submit (which lacks those options) works.
    assert "--env" not in argv
    assert "--ptoas" not in argv
    run_cmd = argv[-1]
    assert "pypto.runtime.execute_artifact" in run_cmd
    assert "--device-id $TASK_DEVICE" in run_cmd
    assert "--enable-l2-swimlane" in run_cmd
    assert "--pto-isa-commit abc123" in run_cmd
    # Device run only; the harness validates with the real tolerance afterwards.
    assert "--no-validate" in run_cmd
    # full child output persisted next to the artifact
    assert (tmp_path / "execute.log").exists()


def test_task_submit_pins_device_when_requested(tmp_path):
    """Test mode: a specific --device pins the card instead of borrowing auto."""
    with patch.object(
        test_runner.subprocess, "run", return_value=_proc(0, "PYPTO_EXEC_RESULT=PASS device=2\n")
    ) as run:
        passed, _, device = test_runner._run_artifact_via_task_submit(
            tmp_path, "a2a3", _DfxOpts(), None, 600, 1800, device="2"
        )
    assert passed is True
    assert device == 2
    argv = run.call_args.args[0]
    assert argv[argv.index("--device") + 1] == "2"


def test_task_submit_real_failure(tmp_path):
    with patch.object(
        test_runner.subprocess, "run", return_value=_proc(1, "PYPTO_EXEC_RESULT=FAIL\n", "Traceback: boom")
    ):
        passed, error, device = test_runner._run_artifact_via_task_submit(
            tmp_path, "a2a3", _DfxOpts(), None, 600, 1800
        )
    assert passed is False
    assert device is None
    assert "Test failed on device" in error
    assert "boom" in error


def test_task_submit_queue_timeout_is_distinguished(tmp_path):
    with patch.object(test_runner.subprocess, "run", return_value=_proc(1, "", "")):
        passed, error, _ = test_runner._run_artifact_via_task_submit(
            tmp_path, "a2a3", _DfxOpts(), None, 600, 1800
        )
    assert passed is False
    assert "queue wait timed out" in error


def test_task_submit_watchdog_kill_is_distinguished(tmp_path):
    with patch.object(test_runner.subprocess, "run", return_value=_proc(137, "", "")):
        passed, error, _ = test_runner._run_artifact_via_task_submit(
            tmp_path, "a2a3", _DfxOpts(), None, 600, 1800
        )
    assert passed is False
    assert "--max-time" in error


@pytest.mark.parametrize("exc", [FileNotFoundError(), PermissionError("not executable")])
def test_task_submit_exec_failure(tmp_path, exc):
    # OSError (missing binary *or* not-executable) is reported as an exec failure,
    # not a device test failure — with the "do not pass --execute-via-task-submit"
    # hint so the operator knows to drop the flag on this host.
    with patch.object(test_runner.subprocess, "run", side_effect=exc):
        passed, error, _ = test_runner._run_artifact_via_task_submit(
            tmp_path, "a2a3", _DfxOpts(), None, 600, 1800
        )
    assert passed is False
    assert "do not pass --execute-via-task-submit" in error


# ---------------------------------------------------------------------------
# _fused_execute_task dispatch — sim guard
# ---------------------------------------------------------------------------


def _artifact(platform):
    return test_runner.CompileArtifact(
        work_dir=Path("/tmp/x"),
        resolved_platform=platform,
        error=None,
        runtime_name="rt",
        chip_callable=object(),
    )


def test_sim_platform_never_borrows_a_card():
    """task-submit mode + a *sim* platform must stay on the in-process pool."""
    test_runner._pipeline_ctx["execute_mode"] = "task-submit"
    pool: queue.Queue = queue.Queue()
    pool.put(0)
    test_runner._device_pool = pool  # pyright: ignore[reportAttributeAccessIssue]
    tc = Mock()
    tc.get_name.return_value = "case_sim"
    timing = SimpleNamespace(device_wall_us=1.0, host_wall_us=2.0)
    with (
        patch.object(test_runner, "_execute_on_device", return_value=timing) as on_dev,
        patch.object(test_runner, "_run_artifact_via_task_submit") as via_ts,
    ):
        result = test_runner._fused_execute_task(tc, "case_sim@a2a3sim", _artifact("a2a3sim"))
    assert result.passed is True
    on_dev.assert_called_once()
    via_ts.assert_not_called()


# ---------------------------------------------------------------------------
# _run_batch_via_task_submit — one task-submit task per batch, marker parsing
# ---------------------------------------------------------------------------


def test_batch_argv_and_per_artifact_results(tmp_path):
    wd1 = tmp_path / "a@a2a3"
    wd2 = tmp_path / "b@a2a3"
    wd3 = tmp_path / "c@a2a3"
    entries = [(wd1, "a2a3"), (wd2, "a2a3"), (wd3, "a2a3")]
    manifest = tmp_path / "batch_0.json"
    # wd2 fails, wd3 has no marker (process died before reaching it).
    stdout = f"PYPTO_EXEC_RESULT=PASS work_dir={wd1} device=2\nPYPTO_EXEC_RESULT=FAIL work_dir={wd2}\n"
    with patch.object(test_runner.subprocess, "run", return_value=_proc(1, stdout, "boom")) as run:
        results = test_runner._run_batch_via_task_submit(
            entries, manifest, "auto", _DfxOpts(), "abc123", 600, 1800
        )
    # manifest written + batch command shape
    assert json.loads(manifest.read_text()) == [
        {"work_dir": str(wd1), "platform": "a2a3"},
        {"work_dir": str(wd2), "platform": "a2a3"},
        {"work_dir": str(wd3), "platform": "a2a3"},
    ]
    run_cmd = run.call_args.args[0][-1]
    assert "--batch-manifest" in run_cmd
    assert "--no-validate" in run_cmd
    assert "--pto-isa-commit abc123" in run_cmd
    # per-artifact verdicts
    assert results[str(wd1)] == (True, None, 2)
    assert results[str(wd2)][0] is False  # FAIL marker
    assert results[str(wd3)][0] is False  # no marker -> failed
    assert "no result marker" in results[str(wd3)][1]


def test_batch_missing_task_submit(tmp_path):
    entries = [(tmp_path / "a@a2a3", "a2a3")]
    with patch.object(test_runner.subprocess, "run", side_effect=FileNotFoundError):
        results = test_runner._run_batch_via_task_submit(
            entries, tmp_path / "b.json", "auto", _DfxOpts(), None, 600, 1800
        )
    ok, error, _ = results[str(tmp_path / "a@a2a3")]
    assert ok is False
    assert "do not pass --execute-via-task-submit" in error


def test_marker_value():
    line = "PYPTO_EXEC_RESULT=PASS work_dir=/x/y device=4"
    assert test_runner._marker_value(line, "work_dir=") == "/x/y"
    assert test_runner._marker_value(line, "device=") == "4"
    assert test_runner._marker_value(line, "missing=") is None


# ---------------------------------------------------------------------------
# _classify_task_submit_failure — marker / exit-code triage
# ---------------------------------------------------------------------------


def test_classify_infra_marker_is_not_a_test_failure():
    # An INFRA marker (reconstruction/setup miss) must read as infra, never as a
    # device test failure — even at rc=1.
    err = test_runner._classify_task_submit_failure(1, "boom\nPYPTO_EXEC_RESULT=INFRA\n", "")
    assert "infra" in err.lower()
    assert "Test failed on device" not in err


def test_classify_rc1_no_output_is_queue_timeout():
    # No child output at all → task-submit never got a card within --timeout.
    err = test_runner._classify_task_submit_failure(1, "", "")
    assert "queue wait timed out" in err


def test_classify_rc1_with_output_no_marker_is_marker_missing():
    # The child ran and printed (e.g. an import crash) but emitted no marker —
    # don't mislabel it as a queue timeout.
    err = test_runner._classify_task_submit_failure(1, "ImportError: boom", "traceback")
    assert "without a result marker" in err
    assert "queue wait timed out" not in err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

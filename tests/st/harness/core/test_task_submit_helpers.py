# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the ST harness task-submit execution helpers.

``subprocess.run`` is mocked so these tests never invoke the real
``task-submit`` binary and need no device. ``tests/st`` is on ``sys.path``
(inserted by ``tests/st/conftest.py``), so ``harness.core.test_runner`` imports.
"""

import queue
import shlex
import subprocess
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from pypto.runtime.execute_artifact import _build_parser
from pypto.runtime.runner import _DfxOpts

from harness.core import test_runner as tr
from harness.core.test_runner import (
    _TASK_SUBMIT_FORWARD_ENV,
    CompileArtifact,
    _dfx_to_cli,
    _parse_exec_marker,
)


@pytest.fixture
def pipeline_state():
    """Save/restore the harness module globals the helpers read/write."""
    saved = (dict(tr._pipeline_ctx), dict(tr._executed_device), tr._device_pool)
    tr._pipeline_ctx = {"execute_mode": "task-submit", "task_max_time": 600, "dfx": _DfxOpts()}
    tr._executed_device = {}
    tr._device_pool = None
    yield
    tr._pipeline_ctx, tr._executed_device, tr._device_pool = saved[0], saved[1], saved[2]


def _fake_proc(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=["task-submit"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _run_str(mock_run) -> str:
    argv = mock_run.call_args.args[0]
    return argv[argv.index("--run") + 1]


# ---------------------------------------------------------------------------
# _dfx_to_cli round-trip with the execute_artifact parser
# ---------------------------------------------------------------------------


def _roundtrip(dfx: _DfxOpts) -> _DfxOpts:
    ns = _build_parser().parse_args(
        ["--work-dir", "x", "--platform", "a2a3", "--device-id", "0", *_dfx_to_cli(dfx)]
    )
    return _DfxOpts(
        enable_l2_swimlane=ns.enable_l2_swimlane,
        enable_dump_tensor=ns.dump_tensor,
        enable_pmu=ns.enable_pmu,
        enable_dep_gen=ns.enable_dep_gen,
        enable_scope_stats=ns.enable_scope_stats,
    )


@pytest.mark.parametrize(
    "dfx",
    [
        _DfxOpts(),
        _DfxOpts(enable_dump_tensor=1),
        _DfxOpts(enable_dump_tensor=2),
        _DfxOpts(enable_pmu=5),
        _DfxOpts(enable_l2_swimlane=True, enable_dep_gen=True, enable_scope_stats=True),
        _DfxOpts(
            enable_l2_swimlane=True,
            enable_dump_tensor=2,
            enable_pmu=3,
            enable_dep_gen=True,
            enable_scope_stats=True,
        ),
    ],
)
def test_dfx_to_cli_round_trip(dfx: _DfxOpts) -> None:
    assert _roundtrip(dfx) == dfx


def test_dfx_to_cli_empty_is_no_flags() -> None:
    assert _dfx_to_cli(_DfxOpts()) == []


# ---------------------------------------------------------------------------
# _parse_exec_marker
# ---------------------------------------------------------------------------


def test_parse_exec_marker_pass() -> None:
    assert _parse_exec_marker("noise\n__PYPTO_EXEC__ result=PASS device=3\n") == ("PASS", 3)


def test_parse_exec_marker_fail() -> None:
    assert _parse_exec_marker("__PYPTO_EXEC__ result=FAIL device=0") == ("FAIL", 0)


def test_parse_exec_marker_absent() -> None:
    assert _parse_exec_marker("only device output, no marker\n") == (None, None)


def test_parse_exec_marker_picks_last() -> None:
    out = "__PYPTO_EXEC__ result=PASS device=1\n__PYPTO_EXEC__ result=FAIL device=2\n"
    assert _parse_exec_marker(out) == ("FAIL", 2)


def test_parse_exec_marker_handles_empty() -> None:
    assert _parse_exec_marker("") == (None, None)


# ---------------------------------------------------------------------------
# _execute_via_task_submit — argv construction
# ---------------------------------------------------------------------------


def test_builds_task_submit_argv(pipeline_state, monkeypatch) -> None:
    monkeypatch.delenv("DEVICE_ID", raising=False)  # exercise the auto fallback
    monkeypatch.delenv("PYPTO_TASK_SUBMIT_SETUP", raising=False)
    for var in _TASK_SUBMIT_FORWARD_ENV:
        monkeypatch.delenv(var, raising=False)  # start clean
    for var in ("PYTHONPATH", "PTO_ISA_ROOT", "ASCEND_HOME_PATH"):
        monkeypatch.setenv(var, "x")  # exactly these three forwarded
    tr._pipeline_ctx["pto_isa_commit"] = "cafe1234"
    dfx = _DfxOpts(enable_dump_tensor=2, enable_l2_swimlane=True)
    with patch.object(
        tr.subprocess, "run", return_value=_fake_proc(0, "__PYPTO_EXEC__ result=PASS device=2\n")
    ) as m_run:
        res = tr._execute_via_task_submit("t", "k", Path("/shared/wd"), "a2a3", dfx, 0.0)

    assert res.passed is True
    argv = m_run.call_args.args[0]
    assert argv[0] == "task-submit"
    assert argv[1:3] == ["--device", "auto"]
    assert "--max-time" in argv and "600" in argv
    # env forwarding for the task-submit hop (all three are set above)
    assert argv.count("--env") == 3
    for var in ("PYTHONPATH", "PTO_ISA_ROOT", "ASCEND_HOME_PATH"):
        assert var in argv
    run_str = _run_str(m_run)
    assert "-m pypto.runtime.execute_artifact" in run_str
    assert "--device-id $TASK_DEVICE" in run_str  # literal, not expanded
    assert "--platform a2a3" in run_str
    assert "--dump-tensor 2" in run_str
    assert "--enable-l2-swimlane" in run_str
    assert "--pto-isa-commit cafe1234" in run_str


def test_device_pinned_to_device_id_env(pipeline_state, monkeypatch) -> None:
    # When DEVICE_ID is set (CI container / validation), pin task-submit to it
    # instead of --device auto, so it locks the job's reserved card.
    monkeypatch.setenv("DEVICE_ID", "3")
    with patch.object(
        tr.subprocess, "run", return_value=_fake_proc(0, "__PYPTO_EXEC__ result=PASS device=3\n")
    ) as m_run:
        tr._execute_via_task_submit("t", "k", Path("/shared/wd"), "a2a3", _DfxOpts(), 0.0)
    argv = m_run.call_args.args[0]
    assert argv[1:3] == ["--device", "3"]


def test_env_forwarding_skips_unset_vars(pipeline_state, monkeypatch) -> None:
    # Only vars that are actually set are forwarded — no --env for unset ones,
    # so task-submit doesn't warn "环境变量 X 未定义".
    for var in _TASK_SUBMIT_FORWARD_ENV:
        monkeypatch.delenv(var, raising=False)  # start clean
    monkeypatch.setenv("PTO_ISA_ROOT", "/p")
    monkeypatch.setenv("ASCEND_HOME_PATH", "/a")
    with patch.object(
        tr.subprocess, "run", return_value=_fake_proc(0, "__PYPTO_EXEC__ result=PASS device=0\n")
    ) as m_run:
        tr._execute_via_task_submit("t", "k", Path("/wd"), "a2a3", _DfxOpts(), 0.0)
    argv = m_run.call_args.args[0]
    assert argv.count("--env") == 2
    assert "PYTHONPATH" not in argv
    assert "PTO_ISA_ROOT" in argv and "ASCEND_HOME_PATH" in argv


def test_setup_prefix_injected_into_run_str(pipeline_state, monkeypatch) -> None:
    # PYPTO_TASK_SUBMIT_SETUP is run after the cd and before the python call so
    # the borrowed-card subprocess can re-activate the host env.
    monkeypatch.setenv("PYPTO_TASK_SUBMIT_SETUP", "source activate.sh")
    with patch.object(
        tr.subprocess, "run", return_value=_fake_proc(0, "__PYPTO_EXEC__ result=PASS device=0\n")
    ) as m_run:
        tr._execute_via_task_submit("t", "k", Path("/shared/wd"), "a2a3", _DfxOpts(), 0.0)
    run_str = _run_str(m_run)
    assert "&& source activate.sh &&" in run_str
    # ordering: cd ... before setup before the python invocation
    assert run_str.index("cd ") < run_str.index("source activate.sh") < run_str.index("-m pypto")


def test_no_setup_prefix_by_default(pipeline_state, monkeypatch) -> None:
    monkeypatch.delenv("PYPTO_TASK_SUBMIT_SETUP", raising=False)
    with patch.object(
        tr.subprocess, "run", return_value=_fake_proc(0, "__PYPTO_EXEC__ result=PASS device=0\n")
    ) as m_run:
        tr._execute_via_task_submit("t", "k", Path("/shared/wd"), "a2a3", _DfxOpts(), 0.0)
    run_str = _run_str(m_run)
    # cd directly into the python invocation, no extra && segment
    assert "source" not in run_str
    assert run_str.count(" && ") == 1


def test_argv_shlex_quotes_spaced_and_bracketed_paths(pipeline_state) -> None:
    work_dir = Path("/shared/a b[shape=128]/case")
    with patch.object(
        tr.subprocess, "run", return_value=_fake_proc(0, "__PYPTO_EXEC__ result=PASS device=0\n")
    ) as m_run:
        tr._execute_via_task_submit("t", "k", work_dir, "a2a3", _DfxOpts(), 0.0)
    run_str = _run_str(m_run)
    assert shlex.quote(str(work_dir)) in run_str
    # bare, unquoted path must NOT appear (would word-split in task-submit's shell)
    assert f"--work-dir {work_dir} " not in run_str


def test_no_pto_isa_commit_omits_flag(pipeline_state) -> None:
    tr._pipeline_ctx.pop("pto_isa_commit", None)
    with patch.object(
        tr.subprocess, "run", return_value=_fake_proc(0, "__PYPTO_EXEC__ result=PASS device=0\n")
    ) as m_run:
        tr._execute_via_task_submit("t", "k", Path("/wd"), "a2a3", _DfxOpts(), 0.0)
    assert "--pto-isa-commit" not in _run_str(m_run)


# ---------------------------------------------------------------------------
# _execute_via_task_submit — pass/fail semantics + device reporting
# ---------------------------------------------------------------------------


def test_returncode_zero_marker_pass(pipeline_state) -> None:
    with patch.object(
        tr.subprocess, "run", return_value=_fake_proc(0, "__PYPTO_EXEC__ result=PASS device=4\n")
    ):
        res = tr._execute_via_task_submit("t", "k", Path("/wd"), "a2a3", _DfxOpts(), 0.0)
    assert res.passed is True
    assert tr._executed_device["k"] == 4


def test_returncode_nonzero_marker_fail(pipeline_state) -> None:
    with patch.object(
        tr.subprocess, "run", return_value=_fake_proc(1, "__PYPTO_EXEC__ result=FAIL device=4\n", "boom")
    ):
        res = tr._execute_via_task_submit("t", "k", Path("/wd"), "a2a3", _DfxOpts(), 0.0)
    assert res.passed is False
    assert "boom" in res.error
    assert tr._executed_device["k"] == 4


def test_marker_fail_overrides_false_green_returncode(pipeline_state) -> None:
    # task-submit returned 0 but the inner run actually failed — marker wins.
    with patch.object(
        tr.subprocess, "run", return_value=_fake_proc(0, "__PYPTO_EXEC__ result=FAIL device=2\n")
    ):
        res = tr._execute_via_task_submit("t", "k", Path("/wd"), "a2a3", _DfxOpts(), 0.0)
    assert res.passed is False


def test_no_marker_falls_back_to_returncode(pipeline_state) -> None:
    # subprocess crashed before emitting a marker → rc decides, device unset.
    with patch.object(tr.subprocess, "run", return_value=_fake_proc(1, "partial\n", "segfault")):
        res = tr._execute_via_task_submit("t", "k", Path("/wd"), "a2a3", _DfxOpts(), 0.0)
    assert res.passed is False
    assert "k" not in tr._executed_device


def test_no_marker_returncode_zero_passes(pipeline_state) -> None:
    with patch.object(tr.subprocess, "run", return_value=_fake_proc(0, "no marker here\n")):
        res = tr._execute_via_task_submit("t", "k", Path("/wd"), "a2a3", _DfxOpts(), 0.0)
    assert res.passed is True
    assert "k" not in tr._executed_device  # device unknown without a marker


# ---------------------------------------------------------------------------
# branch wiring: sim stays in-process, non-sim routes to task-submit
# ---------------------------------------------------------------------------


def _fake_tc(name="case"):
    return types.SimpleNamespace(get_name=lambda: name)


def test_fused_execute_sim_stays_in_process(pipeline_state) -> None:
    tr._device_pool = queue.Queue()
    tr._device_pool.put(0)
    artifact = CompileArtifact(
        work_dir=Path("/wd"),
        resolved_platform="a2a3sim",
        error=None,
        runtime_name="rt",
        chip_callable="cc",
    )
    with patch.object(tr, "_execute_on_device") as m_exec, patch.object(tr.subprocess, "run") as m_run:
        res = tr._fused_execute_task(_fake_tc(), "k", artifact)
    assert res.passed is True
    m_run.assert_not_called()  # sim never borrows a card
    m_exec.assert_called_once()  # in-process device path used


def test_fused_execute_non_sim_routes_to_task_submit(pipeline_state) -> None:
    artifact = CompileArtifact(
        work_dir=Path("/shared/wd"),
        resolved_platform="a2a3",
        error=None,
        runtime_name="rt",
        chip_callable="cc",
    )
    with (
        patch.object(
            tr.subprocess, "run", return_value=_fake_proc(0, "__PYPTO_EXEC__ result=PASS device=6\n")
        ) as m_run,
        patch.object(tr, "_execute_on_device") as m_exec,
    ):
        res = tr._fused_execute_task(_fake_tc(), "k", artifact)
    assert res.passed is True
    m_run.assert_called_once()  # borrowed a card via task-submit
    m_exec.assert_not_called()  # in-process path bypassed
    assert tr._executed_device["k"] == 6


def test_fused_execute_codegen_only_skips_borrow(pipeline_state) -> None:
    tr._pipeline_ctx["codegen_only"] = True
    artifact = CompileArtifact(
        work_dir=Path("/wd"),
        resolved_platform="a2a3",
        error=None,
        runtime_name="rt",
        chip_callable="cc",
    )
    with patch.object(tr.subprocess, "run") as m_run:
        res = tr._fused_execute_task(_fake_tc(), "k", artifact)
    assert res.passed is True
    m_run.assert_not_called()  # codegen-only never borrows a card


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

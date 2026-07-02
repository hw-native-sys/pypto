# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for :mod:`pypto.runtime.execute_artifact`.

``compile_and_assemble`` and ``_execute_on_device`` are mocked so these tests
run without a device, without ``simpler``, and without any compiled artifact.
They pin the CLI contract the harness relies on: arg parsing, DFX passthrough,
and the ``PYPTO_EXEC_RESULT`` marker + exit code on pass / fail.
"""

import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pypto.runtime.runner import _DfxOpts

execute_artifact = importlib.import_module("pypto.runtime.execute_artifact")
main = execute_artifact.main
execute_artifact_dir = execute_artifact.execute_artifact_dir
execute_batch_manifest = execute_artifact.execute_batch_manifest


@pytest.fixture
def stub_compile_and_assemble():
    """Inject a stub ``pypto.runtime.device_runner`` so these tests don't import it.

    ``execute_artifact`` imports ``compile_and_assemble`` lazily from
    ``pypto.runtime.device_runner``; importing that module pulls in the optional
    ``simpler`` package, which is absent on the unit-test runners. A stub module
    lets the lazy import resolve to a mock without it.
    """
    ca = MagicMock(return_value=(object(), "tensormap_and_ringbuffer", {}))
    stub = SimpleNamespace(compile_and_assemble=ca)
    with patch.dict(sys.modules, {"pypto.runtime.device_runner": stub}):
        yield ca


def _argv(work_dir, *extra):
    return ["--work-dir", str(work_dir), "--platform", "a2a3", "--device-id", "3", *extra]


def test_pass_prints_marker_and_returns_zero(tmp_path, capsys):
    with (
        patch.object(execute_artifact, "execute_artifact_dir") as run,
    ):
        rc = main(_argv(tmp_path))
    assert rc == 0
    run.assert_called_once()
    out = capsys.readouterr().out
    assert "PYPTO_EXEC_RESULT=PASS device=3" in out


def test_failure_prints_fail_marker_and_returns_one(tmp_path, capsys):
    with patch.object(execute_artifact, "execute_artifact_dir", side_effect=RuntimeError("golden mismatch")):
        rc = main(_argv(tmp_path))
    assert rc == 1
    captured = capsys.readouterr()
    assert "PYPTO_EXEC_RESULT=FAIL" in captured.out
    # The traceback (with the original message) must reach stderr for the
    # harness to surface it.
    assert "golden mismatch" in captured.err
    assert "PYPTO_EXEC_RESULT=PASS" not in captured.out


def test_setup_error_prints_infra_marker_not_fail(tmp_path, capsys):
    """A reconstruction/setup failure emits INFRA (infra), never FAIL (test failure)."""
    with patch.object(
        execute_artifact,
        "execute_artifact_dir",
        side_effect=execute_artifact.ArtifactSetupError("stale cache"),
    ):
        rc = main(_argv(tmp_path))
    assert rc == 1
    captured = capsys.readouterr()
    assert "PYPTO_EXEC_RESULT=INFRA" in captured.out
    assert "PYPTO_EXEC_RESULT=FAIL" not in captured.out
    assert "stale cache" in captured.err


def test_execute_artifact_dir_wraps_compile_failure_as_setup_error(tmp_path, stub_compile_and_assemble):
    """A compile_and_assemble failure surfaces as ArtifactSetupError (infra)."""
    stub_compile_and_assemble.side_effect = RuntimeError("missing .so")
    with pytest.raises(execute_artifact.ArtifactSetupError):
        execute_artifact_dir(tmp_path, "a2a3", 0)


def test_dfx_flags_parsed_into_dfx_opts(tmp_path):
    with patch.object(execute_artifact, "execute_artifact_dir") as run:
        rc = main(
            _argv(
                tmp_path,
                "--enable-l2-swimlane",
                "--dump-tensor",
                "2",
                "--enable-pmu",
                "5",
                "--enable-dep-gen",
                "--enable-scope-stats",
                "--pto-isa-commit",
                "abc123",
            )
        )
    assert rc == 0
    _, kwargs = run.call_args
    assert kwargs["pto_isa_commit"] == "abc123"
    assert kwargs["dfx"] == _DfxOpts(
        enable_l2_swimlane=True,
        enable_dump_tensor=2,
        enable_pmu=5,
        enable_dep_gen=True,
        enable_scope_stats=True,
    )


def test_plain_run_has_default_dfx(tmp_path):
    with patch.object(execute_artifact, "execute_artifact_dir") as run:
        main(_argv(tmp_path))
    _, kwargs = run.call_args
    assert kwargs["dfx"] == _DfxOpts()
    assert kwargs["pto_isa_commit"] is None
    # Default (manual repro): validate in-process.
    assert kwargs["validate"] is True


def test_no_validate_flag_defers_validation(tmp_path):
    """--no-validate (the harness split path) runs the device but skips allclose."""
    with patch.object(execute_artifact, "execute_artifact_dir") as run:
        rc = main(_argv(tmp_path, "--no-validate"))
    assert rc == 0
    _, kwargs = run.call_args
    assert kwargs["validate"] is False


def test_execute_artifact_dir_wires_compile_then_execute(tmp_path, stub_compile_and_assemble):
    chip = object()
    stub_compile_and_assemble.return_value = (chip, "tensormap_and_ringbuffer", {})
    with patch.object(execute_artifact, "_execute_on_device") as exec_on_dev:
        execute_artifact_dir(tmp_path, "a2a3", 1, pto_isa_commit="deadbeef")
    stub_compile_and_assemble.assert_called_once_with(tmp_path, "a2a3", pto_isa_commit="deadbeef")
    args, _ = exec_on_dev.call_args
    # work_dir, golden_path, chip_callable, runtime_name, platform, device_id
    assert args[0] == tmp_path
    assert args[1] == tmp_path / "golden.py"
    assert args[2] is chip
    assert args[3] == "tensormap_and_ringbuffer"
    assert args[4] == "a2a3"
    assert args[5] == 1


def test_work_dir_and_platform_are_required():
    with pytest.raises(SystemExit):
        main(["--device-id", "0"])


def _manifest(tmp_path, *work_dirs):
    p = tmp_path / "m.json"
    p.write_text(json.dumps([{"work_dir": str(wd), "platform": "a2a3"} for wd in work_dirs]))
    return p


def test_execute_batch_runs_all_in_one_worker(tmp_path, capsys, stub_compile_and_assemble):
    """A batch opens ONE ChipWorker and runs every artifact under it."""
    wd1, wd2 = tmp_path / "a", tmp_path / "b"
    manifest = _manifest(tmp_path, wd1, wd2)
    chipworker = MagicMock()
    with (
        patch("pypto.runtime.ChipWorker", return_value=chipworker) as cw,
        patch.object(execute_artifact, "_execute_on_device") as on_dev,
    ):
        all_ok = execute_batch_manifest(manifest, 3, validate=False)
    assert all_ok is True
    cw.assert_called_once()  # ONE ChipWorker for the whole batch
    chipworker.__enter__.assert_called_once()
    assert on_dev.call_count == 2  # one device run per artifact, reusing the worker
    out = capsys.readouterr().out
    assert f"PYPTO_EXEC_RESULT=PASS work_dir={wd1} device=3" in out
    assert f"PYPTO_EXEC_RESULT=PASS work_dir={wd2} device=3" in out


def test_execute_batch_one_failure_does_not_abort_rest(tmp_path, capsys, stub_compile_and_assemble):
    wd1, wd2 = tmp_path / "a", tmp_path / "b"
    manifest = _manifest(tmp_path, wd1, wd2)
    with (
        patch("pypto.runtime.ChipWorker", return_value=MagicMock()),
        # First artifact's device run fails; second still runs.
        patch.object(execute_artifact, "_execute_on_device", side_effect=[RuntimeError("dev boom"), None]),
    ):
        all_ok = execute_batch_manifest(manifest, 0, validate=False)
    assert all_ok is False
    out = capsys.readouterr().out
    assert f"PYPTO_EXEC_RESULT=FAIL work_dir={wd1}" in out
    assert f"PYPTO_EXEC_RESULT=PASS work_dir={wd2} device=0" in out


def test_execute_batch_first_rebind_failure_marks_infra_and_continues(
    tmp_path, capsys, stub_compile_and_assemble
):
    """A leading un-rebindable artifact gets its own INFRA marker; the rest still run.

    Regression: previously the batch's runtime probe ran outside the per-entry
    try, so a first-entry rebind failure escaped as a marker-less batch crash and
    the harness failed *every* entry in the batch.
    """
    wd1, wd2 = tmp_path / "a", tmp_path / "b"
    manifest = _manifest(tmp_path, wd1, wd2)
    # 1st rebind (probe wd1) fails; 2nd (probe wd2) + 3rd (run wd2) succeed.
    stub_compile_and_assemble.side_effect = [
        RuntimeError("bad .so"),
        (object(), "tensormap_and_ringbuffer", {}),
        (object(), "tensormap_and_ringbuffer", {}),
    ]
    with (
        patch("pypto.runtime.ChipWorker", return_value=MagicMock()),
        patch.object(execute_artifact, "_execute_on_device"),
    ):
        all_ok = execute_batch_manifest(manifest, 0, validate=False)
    assert all_ok is False
    out = capsys.readouterr().out
    assert f"PYPTO_EXEC_RESULT=INFRA work_dir={wd1}" in out
    assert f"PYPTO_EXEC_RESULT=PASS work_dir={wd2} device=0" in out


def test_execute_batch_setup_failure_marks_infra_not_fail(tmp_path, capsys, stub_compile_and_assemble):
    """A mid-batch rebind failure is INFRA (infra), a device-run failure is FAIL."""
    wd1, wd2 = tmp_path / "a", tmp_path / "b"
    manifest = _manifest(tmp_path, wd1, wd2)
    # Probe wd1 ok (opens worker); run wd1 ok; rebind wd2 fails → INFRA for wd2.
    good = (object(), "tensormap_and_ringbuffer", {})
    stub_compile_and_assemble.side_effect = [good, good, RuntimeError("wd2 cache miss")]
    with (
        patch("pypto.runtime.ChipWorker", return_value=MagicMock()),
        patch.object(execute_artifact, "_execute_on_device"),
    ):
        all_ok = execute_batch_manifest(manifest, 0, validate=False)
    assert all_ok is False
    out = capsys.readouterr().out
    assert f"PYPTO_EXEC_RESULT=PASS work_dir={wd1} device=0" in out
    assert f"PYPTO_EXEC_RESULT=INFRA work_dir={wd2}" in out
    assert f"PYPTO_EXEC_RESULT=FAIL work_dir={wd2}" not in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for :mod:`pypto.runtime.execute_artifact`.

``_compile_and_assemble`` and ``_execute_golden_case`` are mocked so these tests
run without a device, without ``simpler``, and without any compiled artifact.
They pin the CLI contract the harness relies on: arg parsing, DFX passthrough,
and the ``PYPTO_EXEC_RESULT`` marker + exit code on pass / fail.
"""

import importlib
import json
from unittest.mock import MagicMock, patch

import pytest
from pypto.runtime.runner import _DfxOpts

execute_artifact = importlib.import_module("pypto.runtime.execute_artifact")
main = execute_artifact.main
execute_artifact_dir = execute_artifact.execute_artifact_dir
execute_batch_manifest = execute_artifact.execute_batch_manifest


@pytest.fixture
def stub_compile_and_assemble(stub_device_runner):
    """The assembly-layer mock these tests drive.

    ``execute_artifact`` reaches it through a lazy import, so the stub module
    ``stub_device_runner`` installs is what the import binds to; see that
    fixture in ``tests/ut/conftest.py``.
    """
    return stub_device_runner._compile_and_assemble


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
    """A _compile_and_assemble failure surfaces as ArtifactSetupError (infra)."""
    stub_compile_and_assemble.side_effect = RuntimeError("missing .so")
    with pytest.raises(execute_artifact.ArtifactSetupError):
        execute_artifact_dir(tmp_path, "a2a3", 0)


def test_dfx_flags_parsed_into_dfx_opts(tmp_path):
    with patch.object(execute_artifact, "execute_artifact_dir") as run:
        rc = main(
            _argv(
                tmp_path,
                "--enable-chip-swimlane",
                "--dump-args",
                "2",
                "--enable-pmu",
                "5",
                "--enable-dep-gen",
                "--enable-scope-stats",
            )
        )
    assert rc == 0
    _, kwargs = run.call_args
    assert kwargs["dfx"] == _DfxOpts(
        enable_chip_swimlane=True,
        enable_dump_args=2,
        enable_pmu=5,
        enable_dep_gen=True,
        enable_scope_stats=True,
    )


def test_plain_run_has_default_dfx(tmp_path):
    with patch.object(execute_artifact, "execute_artifact_dir") as run:
        main(_argv(tmp_path))
    _, kwargs = run.call_args
    assert kwargs["dfx"] == _DfxOpts()
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
    stub_compile_and_assemble.return_value = (
        chip,
        "tensormap_and_ringbuffer",
        {"enable_sdma": True},
    )
    with patch.object(execute_artifact, "_execute_golden_case") as exec_on_dev:
        execute_artifact_dir(tmp_path, "a2a3", 1)
    stub_compile_and_assemble.assert_called_once_with(tmp_path, "a2a3")
    args, kwargs = exec_on_dev.call_args
    # work_dir, golden_path, chip_callable, runtime_name, platform, device_id
    assert args[0] == tmp_path
    assert args[1] == tmp_path / "golden.py"
    assert args[2] is chip
    assert args[3] == "tensormap_and_ringbuffer"
    assert args[4] == "a2a3"
    assert args[5] == 1
    assert kwargs["enable_sdma"] is True


def test_execute_artifact_dir_defaults_sdma_off(tmp_path, stub_compile_and_assemble):
    with patch.object(execute_artifact, "_execute_golden_case") as exec_on_dev:
        execute_artifact_dir(tmp_path, "a2a3", 1)

    assert exec_on_dev.call_args.kwargs["enable_sdma"] is False


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
        patch.object(execute_artifact, "_execute_golden_case") as on_dev,
    ):
        all_ok = execute_batch_manifest(manifest, 3, validate=False)
    assert all_ok is True
    cw.assert_called_once()  # ONE ChipWorker for the whole batch
    assert cw.call_args.kwargs["enable_sdma"] is False
    chipworker.__enter__.assert_called_once()
    assert on_dev.call_count == 2  # one device run per artifact, reusing the worker
    assert all(call.kwargs["enable_sdma"] is False for call in on_dev.call_args_list)
    out = capsys.readouterr().out
    assert f"PYPTO_EXEC_RESULT=PASS work_dir={wd1} device=3" in out
    assert f"PYPTO_EXEC_RESULT=PASS work_dir={wd2} device=3" in out


def test_execute_batch_enables_sdma_worker_for_prefetch_artifact(
    tmp_path,
    stub_compile_and_assemble,
):
    wd = tmp_path / "prefetch"
    manifest = _manifest(tmp_path, wd)
    stub_compile_and_assemble.return_value = (
        object(),
        "tensormap_and_ringbuffer",
        {"enable_sdma": True},
    )

    with (
        patch("pypto.runtime.ChipWorker", return_value=MagicMock()) as worker_cls,
        patch.object(execute_artifact, "_execute_golden_case") as on_dev,
    ):
        all_ok = execute_batch_manifest(manifest, 3, validate=False)

    assert all_ok is True
    assert worker_cls.call_args.kwargs["enable_sdma"] is True
    assert on_dev.call_args.kwargs["enable_sdma"] is True


@pytest.mark.parametrize(
    "runtime_configs",
    [
        pytest.param(({}, {"enable_sdma": True}), id="ordinary-then-sdma"),
        pytest.param(({"enable_sdma": True}, {}), id="sdma-then-ordinary"),
    ],
)
def test_execute_batch_mixed_sdma_capability_is_order_independent(
    tmp_path,
    capsys,
    stub_compile_and_assemble,
    runtime_configs,
):
    """Any same-binding SDMA artifact enables the shared worker, independent of manifest order."""
    wd1, wd2 = tmp_path / "a", tmp_path / "b"
    manifest = _manifest(tmp_path, wd1, wd2)
    configs_by_work_dir = dict(zip((wd1, wd2), runtime_configs, strict=True))

    def reconstruct(work_dir, _platform):
        return object(), "tensormap_and_ringbuffer", configs_by_work_dir[work_dir]

    stub_compile_and_assemble.side_effect = reconstruct
    with (
        patch("pypto.runtime.ChipWorker", return_value=MagicMock()) as worker_cls,
        patch.object(execute_artifact, "_execute_golden_case") as on_dev,
    ):
        all_ok = execute_batch_manifest(manifest, 3, validate=False)

    assert all_ok is True
    assert worker_cls.call_args.kwargs["enable_sdma"] is True
    assert [call.kwargs["enable_sdma"] for call in on_dev.call_args_list] == [
        bool(config.get("enable_sdma", False)) for config in runtime_configs
    ]
    out = capsys.readouterr().out
    markers = [line for line in out.splitlines() if line.startswith("PYPTO_EXEC_RESULT=")]
    assert markers == [
        f"PYPTO_EXEC_RESULT=PASS work_dir={wd1} device=3",
        f"PYPTO_EXEC_RESULT=PASS work_dir={wd2} device=3",
    ]


@pytest.mark.parametrize(
    ("platforms", "runtimes"),
    [
        pytest.param(
            ("a2a3sim", "a2a3"),
            ("tensormap_and_ringbuffer", "tensormap_and_ringbuffer"),
            id="different-platform",
        ),
        pytest.param(
            ("a2a3", "a2a3"),
            ("first_runtime", "second_runtime"),
            id="different-runtime",
        ),
    ],
)
def test_execute_batch_ignores_fallback_artifact_for_shared_sdma_capability(
    tmp_path,
    capsys,
    stub_compile_and_assemble,
    platforms,
    runtimes,
):
    """An SDMA artifact on another binding must not configure the shared worker."""
    wd1, wd2 = tmp_path / "ordinary", tmp_path / "sdma"
    manifest = tmp_path / "mixed-bindings.json"
    manifest.write_text(
        json.dumps(
            [
                {"work_dir": str(wd1), "platform": platforms[0]},
                {"work_dir": str(wd2), "platform": platforms[1]},
            ]
        )
    )
    artifacts_by_work_dir = {
        wd1: (object(), runtimes[0], {}),
        wd2: (object(), runtimes[1], {"enable_sdma": True}),
    }

    def reconstruct(work_dir, _platform):
        return artifacts_by_work_dir[work_dir]

    stub_compile_and_assemble.side_effect = reconstruct
    with (
        patch("pypto.runtime.ChipWorker", return_value=MagicMock()) as worker_cls,
        patch.object(execute_artifact, "_execute_golden_case") as on_dev,
    ):
        all_ok = execute_batch_manifest(manifest, 3, validate=False)

    assert all_ok is True
    assert worker_cls.call_args.kwargs["enable_sdma"] is False
    assert [call.kwargs["enable_sdma"] for call in on_dev.call_args_list] == [False, True]
    markers = [line for line in capsys.readouterr().out.splitlines() if line.startswith("PYPTO_EXEC_RESULT=")]
    assert markers == [
        f"PYPTO_EXEC_RESULT=PASS work_dir={wd1} device=3",
        f"PYPTO_EXEC_RESULT=PASS work_dir={wd2} device=3",
    ]


def test_execute_batch_one_failure_does_not_abort_rest(tmp_path, capsys, stub_compile_and_assemble):
    wd1, wd2 = tmp_path / "a", tmp_path / "b"
    manifest = _manifest(tmp_path, wd1, wd2)
    with (
        patch("pypto.runtime.ChipWorker", return_value=MagicMock()),
        # First artifact's device run fails; second still runs.
        patch.object(execute_artifact, "_execute_golden_case", side_effect=[RuntimeError("dev boom"), None]),
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

    def reconstruct(work_dir, _platform):
        if work_dir == wd1:
            raise RuntimeError("bad .so")
        return object(), "tensormap_and_ringbuffer", {}

    stub_compile_and_assemble.side_effect = reconstruct
    with (
        patch("pypto.runtime.ChipWorker", return_value=MagicMock()),
        patch.object(execute_artifact, "_execute_golden_case"),
    ):
        all_ok = execute_batch_manifest(manifest, 0, validate=False)
    assert all_ok is False
    out = capsys.readouterr().out
    markers = [line for line in out.splitlines() if line.startswith("PYPTO_EXEC_RESULT=")]
    assert markers == [
        f"PYPTO_EXEC_RESULT=INFRA work_dir={wd1}",
        f"PYPTO_EXEC_RESULT=PASS work_dir={wd2} device=0",
    ]


def test_execute_batch_setup_failure_marks_infra_not_fail(tmp_path, capsys, stub_compile_and_assemble):
    """A mid-batch rebind failure is INFRA (infra), a device-run failure is FAIL."""
    wd1, wd2 = tmp_path / "a", tmp_path / "b"
    manifest = _manifest(tmp_path, wd1, wd2)

    def reconstruct(work_dir, _platform):
        if work_dir == wd2:
            raise RuntimeError("wd2 cache miss")
        return object(), "tensormap_and_ringbuffer", {}

    stub_compile_and_assemble.side_effect = reconstruct
    with (
        patch("pypto.runtime.ChipWorker", return_value=MagicMock()),
        patch.object(execute_artifact, "_execute_golden_case"),
    ):
        all_ok = execute_batch_manifest(manifest, 0, validate=False)
    assert all_ok is False
    out = capsys.readouterr().out
    markers = [line for line in out.splitlines() if line.startswith("PYPTO_EXEC_RESULT=")]
    assert markers == [
        f"PYPTO_EXEC_RESULT=PASS work_dir={wd1} device=0",
        f"PYPTO_EXEC_RESULT=INFRA work_dir={wd2}",
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

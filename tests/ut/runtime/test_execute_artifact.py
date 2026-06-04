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
run without a device and without the optional ``simpler`` runtime package.
``compile_and_assemble`` is imported lazily from ``device_runner`` (which eagerly
pulls in ``simpler``), so it is stubbed via ``sys.modules`` rather than patched
on an attribute.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pypto.runtime import execute_artifact
from pypto.runtime.execute_artifact import _EXEC_MARKER, _build_parser, main
from pypto.runtime.runner import _DfxOpts


@pytest.fixture
def mocks():
    """Stub the lazily-imported device_runner.compile_and_assemble and patch
    _execute_on_device. Yields ``(assemble_mock, execute_mock)``."""
    assemble = MagicMock(return_value=("CHIP_CALLABLE", "rt_name", {}))
    fake_device_runner = types.ModuleType("pypto.runtime.device_runner")
    fake_device_runner.compile_and_assemble = assemble
    saved = sys.modules.get("pypto.runtime.device_runner")
    sys.modules["pypto.runtime.device_runner"] = fake_device_runner
    try:
        with patch.object(execute_artifact, "_execute_on_device") as execute:
            yield assemble, execute
    finally:
        if saved is not None:
            sys.modules["pypto.runtime.device_runner"] = saved
        else:
            sys.modules.pop("pypto.runtime.device_runner", None)


# ---------------------------------------------------------------------------
# arg parsing → helper calls
# ---------------------------------------------------------------------------


def test_main_parses_core_args_and_calls_helpers(tmp_path: Path, mocks) -> None:
    assemble, execute = mocks
    rc = main(["--work-dir", str(tmp_path), "--platform", "a2a3", "--device-id", "3"])

    assert rc == 0
    assemble.assert_called_once_with(tmp_path, "a2a3", pto_isa_commit=None)
    # _execute_on_device(work_dir, golden_path, chip_callable, runtime_name, platform, device_id, dfx=...)
    args, kwargs = execute.call_args
    assert args[0] == tmp_path
    assert args[1] == tmp_path / "golden.py"
    assert args[2] == "CHIP_CALLABLE"
    assert args[3] == "rt_name"
    assert args[4] == "a2a3"
    assert args[5] == 3
    assert kwargs["dfx"] == _DfxOpts()


def test_pto_isa_commit_passthrough(tmp_path: Path, mocks) -> None:
    assemble, _ = mocks
    main(
        ["--work-dir", str(tmp_path), "--platform", "a2a3", "--device-id", "0", "--pto-isa-commit", "abc123"]
    )
    assemble.assert_called_once_with(tmp_path, "a2a3", pto_isa_commit="abc123")


def test_dfx_flags_build_dfx_opts(tmp_path: Path, mocks) -> None:
    _, execute = mocks
    main(
        [
            "--work-dir",
            str(tmp_path),
            "--platform",
            "a2a3",
            "--device-id",
            "0",
            "--dump-tensor",
            "2",
            "--enable-pmu",
            "5",
            "--enable-l2-swimlane",
            "--enable-dep-gen",
            "--enable-scope-stats",
        ]
    )
    dfx = execute.call_args.kwargs["dfx"]
    assert dfx == _DfxOpts(
        enable_l2_swimlane=True,
        enable_dump_tensor=2,
        enable_pmu=5,
        enable_dep_gen=True,
        enable_scope_stats=True,
    )


# ---------------------------------------------------------------------------
# return codes + result marker
# ---------------------------------------------------------------------------


def test_main_returns_0_and_prints_pass_marker(tmp_path: Path, mocks, capsys) -> None:
    rc = main(["--work-dir", str(tmp_path), "--platform", "a2a3", "--device-id", "7"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == f"{_EXEC_MARKER} result=PASS device=7"


def test_main_returns_1_and_prints_fail_marker_on_exception(tmp_path: Path, mocks, capsys) -> None:
    _, execute = mocks
    execute.side_effect = AssertionError("golden mismatch")
    rc = main(["--work-dir", str(tmp_path), "--platform", "a2a3", "--device-id", "4"])
    assert rc == 1
    captured = capsys.readouterr()
    assert f"{_EXEC_MARKER} result=FAIL device=4" in captured.out
    # Traceback goes to stderr, not stdout.
    assert "AssertionError" in captured.err
    assert "golden mismatch" in captured.err


def test_main_returns_1_when_assemble_fails(tmp_path: Path, mocks, capsys) -> None:
    assemble, execute = mocks
    assemble.side_effect = RuntimeError("no kernel_config.py")
    rc = main(["--work-dir", str(tmp_path), "--platform", "a2a3", "--device-id", "1"])
    assert rc == 1
    execute.assert_not_called()  # never reached execution
    assert f"{_EXEC_MARKER} result=FAIL device=1" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# execute_artifact_dir threads the assembled callable through
# ---------------------------------------------------------------------------


def test_execute_artifact_dir_threads_callable(tmp_path: Path, mocks) -> None:
    _, execute = mocks
    execute_artifact.execute_artifact_dir(
        tmp_path, "a5", 2, pto_isa_commit="deadbeef", dfx=_DfxOpts(enable_pmu=1)
    )
    args, kwargs = execute.call_args
    assert args[2] == "CHIP_CALLABLE" and args[3] == "rt_name"
    assert args[5] == 2
    assert kwargs["dfx"] == _DfxOpts(enable_pmu=1)


# ---------------------------------------------------------------------------
# parser sanity: required args
# ---------------------------------------------------------------------------


def test_parser_requires_core_args() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # missing required --work-dir/--platform/--device-id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

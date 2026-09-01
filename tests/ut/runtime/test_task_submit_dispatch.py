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
from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime import execute_artifact
from pypto.runtime.runner import RunConfig, _DfxOpts

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


def _planner_case():
    return SimpleNamespace(
        get_name=lambda: "planner_case",
        get_program=lambda: object(),
        get_strategy=lambda: None,
        get_backend_type=lambda: test_runner.BackendType.Ascend910B,
        get_memory_planner=lambda: None,
        get_enable_pypto_l0c_double_buffer=lambda: None,
        config=RunConfig(memory_planner=None),
    )


def _write_minimal_compile_output(_program, *, output_dir, **_kwargs):
    work_dir = Path(output_dir)
    (work_dir / "kernels").mkdir(parents=True)
    (work_dir / "kernels" / "kernel.cpp").touch()
    (work_dir / "orchestration").mkdir()
    (work_dir / "orchestration" / "orch.cpp").touch()


# ---------------------------------------------------------------------------
# System-test memory-planner precedence
# ---------------------------------------------------------------------------


def test_case_memory_planner_is_authoritative():
    case = SimpleNamespace(
        get_memory_planner=lambda: MemoryPlanner.PTOAS,
        config=RunConfig(memory_planner=MemoryPlanner.PYPTO),
    )
    assert test_runner._resolve_case_memory_planner(case, MemoryPlanner.DSA_RP) == MemoryPlanner.PTOAS


def test_case_run_config_precedes_session_memory_planner():
    case = SimpleNamespace(
        get_memory_planner=lambda: None,
        config=RunConfig(memory_planner=MemoryPlanner.PYPTO),
    )
    assert test_runner._resolve_case_memory_planner(case, MemoryPlanner.DSA_RP) == MemoryPlanner.PYPTO


def test_session_memory_planner_is_fallback():
    case = SimpleNamespace(
        get_memory_planner=lambda: None,
        config=RunConfig(memory_planner=None),
    )
    assert test_runner._resolve_case_memory_planner(case, MemoryPlanner.DSA_RP) == MemoryPlanner.DSA_RP


def test_system_test_cache_key_separates_memory_planners():
    case = SimpleNamespace(
        get_name=lambda: "case",
        get_platform=lambda: "a2a3",
        get_backend_type=lambda: None,
        get_memory_planner=lambda: None,
        config=RunConfig(memory_planner=None),
    )
    assert test_runner._cache_key(case, "a2a3", MemoryPlanner.PYPTO).endswith("@pypto")
    assert test_runner._cache_key(case, "a2a3", MemoryPlanner.DSA_RP).endswith("@dsa_rp")


def test_precompile_forwards_session_memory_planner(tmp_path):
    case = _planner_case()
    with (
        patch.object(
            test_runner.ir,
            "compile",
            side_effect=_write_minimal_compile_output,
        ) as ir_compile,
        patch.object(test_runner, "_write_golden_for_test_case"),
    ):
        test_runner._compile_for_cache(
            case,
            tmp_path,
            "a2a3",
            dump_passes=False,
            analyze_auto_scopes_for_deps=False,
            session_memory_planner=MemoryPlanner.DSA_RP,
        )
    assert ir_compile.call_args.kwargs["memory_planner"] == MemoryPlanner.DSA_RP


def test_inline_compile_forwards_session_memory_planner():
    case = _planner_case()
    config = RunConfig(
        platform="a2a3sim",
        memory_planner=MemoryPlanner.DSA_RP,
        codegen_only=True,
    )
    with (
        patch.object(
            test_runner.ir,
            "compile",
            side_effect=_write_minimal_compile_output,
        ) as ir_compile,
        patch.object(test_runner, "_write_golden_for_test_case"),
    ):
        result = test_runner.TestRunner(config)._run_inline(case, "a2a3sim")
    assert result.passed, result.error
    assert ir_compile.call_args.kwargs["memory_planner"] == MemoryPlanner.DSA_RP


# ---------------------------------------------------------------------------
# _dfx_to_cli
# ---------------------------------------------------------------------------


def test_dfx_to_cli_empty_for_default():
    assert test_runner._dfx_to_cli(_DfxOpts()) == []


def test_dfx_to_cli_emits_only_enabled_flags():
    dfx = _DfxOpts(enable_chip_swimlane=True, enable_dump_args=2, enable_pmu=5, enable_dep_gen=True)
    argv = test_runner._dfx_to_cli(dfx)
    assert argv == [
        "--enable-chip-swimlane",
        "4",
        "--dump-args",
        "2",
        "--enable-pmu",
        "5",
        "--enable-dep-gen",
    ]


def _load_st_conftest():
    """Import tests/st/conftest.py under a private name.

    Loaded by path rather than ``import conftest`` so pytest's own conftest
    collection is not disturbed.
    """
    import importlib.util  # noqa: PLC0415

    path = _ST_DIR / "conftest.py"
    spec = importlib.util.spec_from_file_location("_st_conftest_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeConfig:
    """Minimal stand-in for ``pytest.Config.getoption``."""

    def __init__(self, bare=0, level=None, deprecated=False):
        self._values = {
            "--enable-chip-swimlane": bare,
            "--chip-swimlane-level": level,
            "enable_l2_swimlane_deprecated": deprecated,
        }

    def getoption(self, name):
        return self._values[name]


def test_st_conftest_resolves_the_swimlane_options():
    resolve = _load_st_conftest()._resolve_swimlane_option
    assert resolve(_FakeConfig()) == 0
    assert resolve(_FakeConfig(bare=4)) == 4
    assert resolve(_FakeConfig(level=2)) == 2
    assert resolve(_FakeConfig(level=0)) == 0  # explicit off
    # An explicit level wins over the bare enable flag.
    assert resolve(_FakeConfig(bare=4, level=1)) == 1


def test_st_conftest_still_accepts_the_deprecated_flag():
    # CI passes --enable-l2-swimlane, so it must keep resolving (with a warning).
    resolve = _load_st_conftest()._resolve_swimlane_option
    with pytest.warns(DeprecationWarning, match="--enable-l2-swimlane is deprecated"):
        assert resolve(_FakeConfig(deprecated=True)) == 4


def test_execute_artifact_accepts_the_deprecated_swimlane_flag():
    # CI and existing scripts still pass --enable-l2-swimlane; it must keep
    # working (with a DeprecationWarning) and land on the canonical level.
    parser = execute_artifact._build_parser()
    args = parser.parse_args(["--enable-l2-swimlane", "2", "--device-id", "0"])
    with pytest.warns(DeprecationWarning, match="--enable-l2-swimlane is deprecated"):
        assert execute_artifact._resolve_swimlane_args(parser, args) == 2

    bare = parser.parse_args(["--enable-l2-swimlane", "--device-id", "0"])
    with pytest.warns(DeprecationWarning):
        assert execute_artifact._resolve_swimlane_args(parser, bare) == 4

    absent = parser.parse_args(["--enable-chip-swimlane", "3", "--device-id", "0"])
    assert execute_artifact._resolve_swimlane_args(parser, absent) == 3


def test_execute_artifact_rejects_conflicting_swimlane_flags():
    parser = execute_artifact._build_parser()
    args = parser.parse_args(["--enable-chip-swimlane", "1", "--enable-l2-swimlane", "3", "--device-id", "0"])
    with pytest.raises(SystemExit), pytest.warns(DeprecationWarning):
        execute_artifact._resolve_swimlane_args(parser, args)


def test_dfx_to_cli_round_trips_the_swimlane_level():
    # Regression (issue #2385): a level 1-3 capture must survive the harness ->
    # execute_artifact CLI hop instead of being flattened to the bare flag.
    for level in (1, 2, 3, 4):
        argv = test_runner._dfx_to_cli(_DfxOpts(enable_chip_swimlane=level))
        assert argv == ["--enable-chip-swimlane", str(level)]
        # ``--device-id`` is the parser's only required argument.
        parsed = execute_artifact._build_parser().parse_args([*argv, "--device-id", "0"])
        assert parsed.enable_chip_swimlane == level


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
    dfx = _DfxOpts(enable_chip_swimlane=True)
    with patch.object(
        test_runner.subprocess,
        "run",
        return_value=_proc(0, "PYPTO_EXEC_RESULT=PASS device=4\n"),
    ) as run:
        passed, error, device = test_runner._run_artifact_via_task_submit(
            tmp_path, "a2a3", dfx, max_time=600, queue_timeout=1800
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
    assert "--enable-chip-swimlane" in run_cmd
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
            tmp_path, "a2a3", _DfxOpts(), 600, 1800, device="2"
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
            tmp_path, "a2a3", _DfxOpts(), 600, 1800
        )
    assert passed is False
    assert device is None
    assert "Test failed on device" in error
    assert "boom" in error


def test_task_submit_queue_timeout_is_distinguished(tmp_path):
    with patch.object(test_runner.subprocess, "run", return_value=_proc(1, "", "")):
        passed, error, _ = test_runner._run_artifact_via_task_submit(tmp_path, "a2a3", _DfxOpts(), 600, 1800)
    assert passed is False
    assert "queue wait timed out" in error


def test_task_submit_watchdog_kill_is_distinguished(tmp_path):
    with patch.object(test_runner.subprocess, "run", return_value=_proc(137, "", "")):
        passed, error, _ = test_runner._run_artifact_via_task_submit(tmp_path, "a2a3", _DfxOpts(), 600, 1800)
    assert passed is False
    assert "--max-time" in error


@pytest.mark.parametrize("exc", [FileNotFoundError(), PermissionError("not executable")])
def test_task_submit_exec_failure(tmp_path, exc):
    # OSError (missing binary *or* not-executable) is reported as an exec failure,
    # not a device test failure — with the "do not pass --execute-via-task-submit"
    # hint so the operator knows to drop the flag on this host.
    with patch.object(test_runner.subprocess, "run", side_effect=exc):
        passed, error, _ = test_runner._run_artifact_via_task_submit(tmp_path, "a2a3", _DfxOpts(), 600, 1800)
    assert passed is False
    assert "do not pass --execute-via-task-submit" in error


# ---------------------------------------------------------------------------
# _fused_execute_task dispatch — sim guard
# ---------------------------------------------------------------------------


def _artifact(platform, *, enable_sdma=False):
    return test_runner.CompileArtifact(
        work_dir=Path("unused_work_dir"),
        resolved_platform=platform,
        error=None,
        runtime_name="rt",
        chip_callable=object(),
        enable_sdma=enable_sdma,
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
        result = test_runner._fused_execute_task(
            tc,
            "case_sim@a2a3sim",
            _artifact("a2a3sim", enable_sdma=True),
        )
    assert result.passed is True
    on_dev.assert_called_once()
    assert on_dev.call_args.kwargs["enable_sdma"] is True
    via_ts.assert_not_called()


def test_fused_compile_records_sdma_capability(tmp_path):
    fake_compile_and_assemble = Mock(
        return_value=(object(), "tensormap_and_ringbuffer", {"enable_sdma": True})
    )
    fake_device_runner = SimpleNamespace(compile_and_assemble=fake_compile_and_assemble)
    tc = Mock()

    with (
        patch.object(test_runner, "_resolve_platform", return_value="a2a3"),
        patch.object(test_runner, "_cache_key", return_value="prefetch@a2a3"),
        patch.object(test_runner, "_compile_for_cache"),
        patch.dict(sys.modules, {"pypto.runtime.device_runner": fake_device_runner}),
    ):
        artifact = test_runner._fused_compile_task(tc, tmp_path, "a2a3", False, False)

    assert artifact.enable_sdma is True


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
        results = test_runner._run_batch_via_task_submit(entries, manifest, "auto", _DfxOpts(), 600, 1800)
    # manifest written + batch command shape
    assert json.loads(manifest.read_text()) == [
        {"work_dir": str(wd1), "platform": "a2a3"},
        {"work_dir": str(wd2), "platform": "a2a3"},
        {"work_dir": str(wd3), "platform": "a2a3"},
    ]
    run_cmd = run.call_args.args[0][-1]
    assert "--batch-manifest" in run_cmd
    assert "--no-validate" in run_cmd
    # per-artifact verdicts
    assert results[str(wd1)] == (True, None, 2)
    assert results[str(wd2)][0] is False  # FAIL marker
    assert results[str(wd3)][0] is False  # no marker -> failed
    assert "no result marker" in results[str(wd3)][1]


def test_batch_missing_task_submit(tmp_path):
    entries = [(tmp_path / "a@a2a3", "a2a3")]
    with patch.object(test_runner.subprocess, "run", side_effect=FileNotFoundError):
        results = test_runner._run_batch_via_task_submit(
            entries, tmp_path / "b.json", "auto", _DfxOpts(), 600, 1800
        )
    ok, error, _ = results[str(tmp_path / "a@a2a3")]
    assert ok is False
    assert "do not pass --execute-via-task-submit" in error


# ---------------------------------------------------------------------------
# _batch_submitter — bucketing keeps each batch single (platform, runtime)
# ---------------------------------------------------------------------------


def test_batch_submitter_never_mixes_runtimes_within_a_batch(tmp_path):
    """A batch child opens ONE ChipWorker keyed on (platform, runtime); if a
    batch mixed runtimes, a differing artifact would miss ChipWorker.current()
    and open a second Worker.init() on the same card (halMemCtl EACCES). Two
    same-platform but different-runtime artifacts must land in SEPARATE batches.
    """
    from concurrent.futures import Future  # noqa: PLC0415

    runtime_by_wd: dict[str, str] = {}
    compile_futures: dict[str, Future] = {}
    for name, runtime in [("a", "rtA"), ("b", "rtB"), ("c", "rtA"), ("d", "rtB")]:
        wd = tmp_path / f"{name}@a2a3"
        runtime_by_wd[str(wd)] = runtime
        fut: Future = Future()
        fut.set_result(
            test_runner.CompileArtifact(
                work_dir=wd,
                resolved_platform="a2a3",
                error=None,
                runtime_name=runtime,
                chip_callable=object(),
            )
        )
        compile_futures[f"{name}@a2a3"] = fut

    submitted: list[list[tuple[Path, str]]] = []

    def _fake_submit(_fn, entries, *_args):
        submitted.append(list(entries))
        done: Future = Future()
        done.set_result({})
        return done

    fake_pool = SimpleNamespace(submit=_fake_submit)
    test_runner._case_to_batch.clear()
    test_runner._batches_ready.clear()
    with (
        patch.object(test_runner, "_execute_pool", fake_pool),
        patch.dict(test_runner._compile_futures, compile_futures, clear=True),
    ):
        # A batch_size larger than either bucket forces the flush through the
        # per-(platform, runtime) leftover path, exercising the bucket key.
        test_runner._batch_submitter(batch_size=10, cache_dir=tmp_path)

    assert test_runner._batches_ready.is_set()
    # Two runtimes → two batches, each pure in its runtime.
    assert len(submitted) == 2
    for batch in submitted:
        runtimes = {runtime_by_wd[str(wd)] for wd, _plat in batch}
        assert len(runtimes) == 1, f"batch mixed runtimes: {runtimes}"
    # All four cases assigned; the two runtimes split 2/2.
    assert sorted(len(b) for b in submitted) == [2, 2]


def test_batch_submitter_never_mixes_sdma_capabilities_within_a_batch(tmp_path):
    """An ordinary worker cannot safely host a later SDMA-required artifact."""
    from concurrent.futures import Future  # noqa: PLC0415

    compile_futures: dict[str, Future] = {}
    for name, enable_sdma in [("plain", False), ("prefetch", True)]:
        wd = tmp_path / f"{name}@a2a3"
        fut: Future = Future()
        fut.set_result(
            test_runner.CompileArtifact(
                work_dir=wd,
                resolved_platform="a2a3",
                error=None,
                runtime_name="tensormap_and_ringbuffer",
                chip_callable=object(),
                enable_sdma=enable_sdma,
            )
        )
        compile_futures[f"{name}@a2a3"] = fut

    submitted: list[list[tuple[Path, str]]] = []

    def _fake_submit(_fn, entries, *_args):
        submitted.append(list(entries))
        done: Future = Future()
        done.set_result({})
        return done

    test_runner._case_to_batch.clear()
    test_runner._batches_ready.clear()
    with (
        patch.object(test_runner, "_execute_pool", SimpleNamespace(submit=_fake_submit)),
        patch.dict(test_runner._compile_futures, compile_futures, clear=True),
    ):
        test_runner._batch_submitter(batch_size=10, cache_dir=tmp_path)

    assert test_runner._batches_ready.is_set()
    assert len(submitted) == 2
    assert all(len(batch) == 1 for batch in submitted)


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

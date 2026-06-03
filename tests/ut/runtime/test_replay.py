# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for :mod:`pypto.runtime.debug.replay`.

``execute_compiled`` is mocked so these tests run without a device and
without the optional ``simpler`` runtime package.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from pypto.runtime import RunConfig

# ``pypto.runtime.debug.__init__`` re-exports ``replay`` (the function),
# which shadows the ``replay`` submodule on attribute lookup. Resolve the
# module via importlib so ``patch.object(replay_module, "...")`` works.
replay_module = importlib.import_module("pypto.runtime.debug.replay")
_load_named_inputs_from_golden = replay_module._load_named_inputs_from_golden
_main = replay_module._main
invalidate_binary_cache = replay_module.invalidate_binary_cache
replay = replay_module.replay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _make_build_output(tmp_path: Path) -> Path:
    """Minimal build_output skeleton with just the marker file."""
    (tmp_path / "kernel_config.py").write_text("KERNELS = []\nORCHESTRATION = {}\n")
    return tmp_path


# ---------------------------------------------------------------------------
# invalidate_binary_cache
# ---------------------------------------------------------------------------


def test_invalidate_binary_cache_removes_bin_files(tmp_path: Path) -> None:
    bin_a = _touch(tmp_path / "cache" / "incore_aiv_foo.bin")
    bin_b = _touch(tmp_path / "cache" / "orch_main.bin")
    invalidate_binary_cache(tmp_path)
    assert not bin_a.exists()
    assert not bin_b.exists()


def test_invalidate_binary_cache_removes_sibling_so_and_o(tmp_path: Path) -> None:
    so_aiv = _touch(tmp_path / "kernels" / "aiv" / "foo.so")
    o_aic = _touch(tmp_path / "kernels" / "aic" / "bar.o")
    so_orch = _touch(tmp_path / "orchestration" / "main.so")
    cpp = _touch(tmp_path / "kernels" / "aiv" / "foo.cpp")  # must survive
    invalidate_binary_cache(tmp_path)
    assert not so_aiv.exists()
    assert not o_aic.exists()
    assert not so_orch.exists()
    assert cpp.exists(), "cpp source must not be deleted"


def test_invalidate_binary_cache_noop_on_empty_dir(tmp_path: Path) -> None:
    invalidate_binary_cache(tmp_path)  # must not raise


def test_invalidate_binary_cache_prints_status_when_files_removed(tmp_path: Path, capsys) -> None:
    _touch(tmp_path / "cache" / "foo.bin")
    _touch(tmp_path / "kernels" / "aiv" / "bar.so")
    invalidate_binary_cache(tmp_path)
    out = capsys.readouterr().out
    assert "[cpp->.so] invalidated 2 cached binary file(s)" in out


def test_invalidate_binary_cache_prints_status_when_nothing_to_do(tmp_path: Path, capsys) -> None:
    invalidate_binary_cache(tmp_path)
    out = capsys.readouterr().out
    assert "[cpp->.so] no cached binaries to invalidate" in out


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def test_replay_routes_to_execute_compiled(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    a = torch.zeros(2)
    b = torch.zeros(2)
    with patch.object(replay_module, "execute_compiled") as ec:
        replay(work_dir, a, b, config=RunConfig(platform="a2a3sim", device_id=3))
    ec.assert_called_once()
    call_args = ec.call_args
    assert call_args.args[0] == work_dir
    assert call_args.args[1] == [a, b]
    assert call_args.kwargs["platform"] == "a2a3sim"
    assert call_args.kwargs["device_id"] == 3


def test_replay_forwards_dfx_flags(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    config = RunConfig(
        platform="a2a3sim",
        enable_l2_swimlane=True,
        enable_pmu=2,
        enable_dump_tensor=True,
        enable_dep_gen=True,
        enable_scope_stats=True,
    )
    with patch.object(replay_module, "execute_compiled") as ec:
        replay(work_dir, config=config)
    dfx = ec.call_args.kwargs["dfx"]
    assert dfx.enable_l2_swimlane is True
    assert dfx.enable_pmu == 2
    assert dfx.enable_dump_tensor is True
    assert dfx.enable_dep_gen is True
    assert dfx.enable_scope_stats is True


def test_replay_invalidates_by_default(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    bin_file = _touch(work_dir / "cache" / "incore_aiv_foo.bin")
    so_file = _touch(work_dir / "kernels" / "aiv" / "foo.so")
    with patch.object(replay_module, "execute_compiled"):
        replay(work_dir)
    assert not bin_file.exists()
    assert not so_file.exists()


def test_replay_skips_invalidation_when_recompile_false(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    bin_file = _touch(work_dir / "cache" / "incore_aiv_foo.bin")
    so_file = _touch(work_dir / "kernels" / "aiv" / "foo.so")
    with patch.object(replay_module, "execute_compiled"):
        replay(work_dir, recompile=False)
    assert bin_file.exists()
    assert so_file.exists()


def test_replay_calls_rebuild_before_invalidate(tmp_path: Path) -> None:
    """rebuild_from_pto must run *before* invalidate_binary_cache so the
    freshly-spliced cpp is the one that drives the subsequent cpp -> .so
    rebuild step."""
    work_dir = _make_build_output(tmp_path)
    call_order: list[str] = []

    def _record_rebuild(*_a, **_kw):
        call_order.append("rebuild")
        return []

    def _record_invalidate(*_a, **_kw):
        call_order.append("invalidate")

    with (
        patch.object(replay_module, "execute_compiled"),
        patch.object(replay_module, "rebuild_kernel_cpp_from_pto", side_effect=_record_rebuild),
        patch.object(replay_module, "invalidate_binary_cache", side_effect=_record_invalidate),
    ):
        replay(work_dir)
    assert call_order == ["rebuild", "invalidate"]


def test_replay_skips_rebuild_from_pto_when_disabled(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    with (
        patch.object(replay_module, "execute_compiled"),
        patch.object(replay_module, "rebuild_kernel_cpp_from_pto") as rb,
    ):
        replay(work_dir, rebuild_from_pto=False)
    rb.assert_not_called()


def test_replay_prints_execute_banner(tmp_path: Path, capsys) -> None:
    work_dir = _make_build_output(tmp_path)
    with patch.object(replay_module, "execute_compiled"):
        replay(work_dir, rebuild_from_pto=False, recompile=False)
    out = capsys.readouterr().out
    assert "[execute] running on device..." in out


def test_replay_missing_kernel_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"kernel_config\.py"):
        replay(tmp_path)


def test_replay_uses_default_run_config_when_none(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    with patch.object(replay_module, "execute_compiled") as ec:
        replay(work_dir)
    default_cfg = RunConfig()
    assert ec.call_args.kwargs["platform"] == default_cfg.platform
    assert ec.call_args.kwargs["device_id"] == default_cfg.device_id


# ---------------------------------------------------------------------------
# validate=True path
# ---------------------------------------------------------------------------


def _write_add_golden(work_dir: Path) -> None:
    """Write a minimal golden.py: outputs[0] = inputs[0] + inputs[1]."""
    (work_dir / "golden.py").write_text(
        "import torch\n"
        "__outputs__ = ['c']\n"
        "RTOL = 1e-5\n"
        "ATOL = 1e-5\n"
        "def generate_inputs(params):\n"
        "    return [('a', torch.zeros(4)), ('b', torch.zeros(4)), ('c', torch.zeros(4))]\n"
        "def compute_golden(tensors, params=None):\n"
        "    tensors['c'].copy_(tensors['a'] + tensors['b'])\n"
    )


def test_replay_validate_passes_when_outputs_match(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    _write_add_golden(work_dir)
    a = torch.full((4,), 2.0)
    b = torch.full((4,), 3.0)
    c = torch.full((4,), 5.0)  # already correct: a+b

    with patch.object(replay_module, "execute_compiled"):
        replay(work_dir, a, b, c, validate=True)  # must not raise


def test_replay_validate_fails_when_outputs_mismatch(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    _write_add_golden(work_dir)
    a = torch.full((4,), 2.0)
    b = torch.full((4,), 3.0)
    c = torch.zeros(4)  # wrong: should be 5.0

    with patch.object(replay_module, "execute_compiled"):
        with pytest.raises(AssertionError, match="does not match golden"):
            replay(work_dir, a, b, c, validate=True)


def test_replay_validate_missing_golden_raises(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"golden\.py"):
        replay(work_dir, torch.zeros(1), validate=True)


def test_replay_validate_tensor_count_mismatch_raises(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    _write_add_golden(work_dir)
    with pytest.raises(ValueError, match="expected 3 tensors"):
        replay(work_dir, torch.zeros(4), torch.zeros(4), validate=True)


def test_replay_validate_false_does_not_open_golden(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    # No golden.py exists; validate=False must not care.
    with patch.object(replay_module, "execute_compiled"):
        replay(work_dir, torch.zeros(4), validate=False)


# ---------------------------------------------------------------------------
# _load_named_inputs_from_golden
# ---------------------------------------------------------------------------


def test_load_inputs_from_golden_returns_named_tuples_in_order(tmp_path: Path) -> None:
    (tmp_path / "golden.py").write_text(
        "import torch\n"
        "def generate_inputs(params):\n"
        "    return [\n"
        "        ('x', torch.zeros(2)),\n"
        "        ('y', torch.ones(3)),\n"
        "        ('z', torch.full((4,), 7.0)),\n"
        "    ]\n"
    )
    named = _load_named_inputs_from_golden(tmp_path)
    assert [n for n, _ in named] == ["x", "y", "z"]
    assert named[0][1].shape == (2,)
    assert named[1][1].shape == (3,)
    assert named[2][1].shape == (4,)
    assert named[2][1][0].item() == 7.0


def test_load_inputs_from_golden_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"golden\.py"):
        _load_named_inputs_from_golden(tmp_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_invokes_replay_with_dfx_flags(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    captured: dict = {}

    def fake_replay(wd, *tensors, config, recompile, rebuild_from_pto, validate):
        captured["work_dir"] = wd
        captured["tensors"] = tensors
        captured["config"] = config
        captured["recompile"] = recompile
        captured["validate"] = validate

    with (
        patch.object(replay_module, "replay", side_effect=fake_replay),
        patch.object(replay_module, "_load_named_inputs_from_golden", return_value=[("a", torch.zeros(1))]),
    ):
        rc = _main([str(work_dir), "--pmu", "2", "--swimlane", "--scope-stats", "--device-id", "5"])
    assert rc == 0
    assert captured["work_dir"] == work_dir
    assert captured["recompile"] is True
    assert captured["validate"] is False
    cfg = captured["config"]
    assert cfg.enable_pmu == 2
    assert cfg.enable_l2_swimlane is True
    assert cfg.enable_scope_stats is True
    assert cfg.device_id == 5


def test_cli_no_recompile_flag(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    captured: dict = {}

    def fake_replay(wd, *tensors, config, recompile, rebuild_from_pto, validate):
        captured["recompile"] = recompile

    with (
        patch.object(replay_module, "replay", side_effect=fake_replay),
        patch.object(replay_module, "_load_named_inputs_from_golden", return_value=[]),
    ):
        _main([str(work_dir), "--no-recompile"])
    assert captured["recompile"] is False


def test_cli_validate_flag(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    captured: dict = {}

    def fake_replay(wd, *tensors, config, recompile, rebuild_from_pto, validate):
        captured["validate"] = validate

    with (
        patch.object(replay_module, "replay", side_effect=fake_replay),
        patch.object(replay_module, "_load_named_inputs_from_golden", return_value=[]),
    ):
        _main([str(work_dir), "--validate"])
    assert captured["validate"] is True


def test_cli_log_level_calls_configure_log(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    captured: dict = {}

    def fake_configure_log(level, *, sync_pypto):
        captured["level"] = level
        captured["sync_pypto"] = sync_pypto

    with (
        patch.object(replay_module, "replay"),
        patch.object(replay_module, "_load_named_inputs_from_golden", return_value=[]),
        patch("pypto.runtime.log_config.configure_log", side_effect=fake_configure_log),
    ):
        _main([str(work_dir), "--log-level", "debug", "--log-sync-pypto"])
    assert captured["level"] == "debug"
    assert captured["sync_pypto"] is True


def test_cli_no_log_level_skips_configure_log(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    called: dict = {"count": 0}

    def fake_configure_log(*a, **kw):
        called["count"] += 1

    with (
        patch.object(replay_module, "replay"),
        patch.object(replay_module, "_load_named_inputs_from_golden", return_value=[]),
        patch("pypto.runtime.log_config.configure_log", side_effect=fake_configure_log),
    ):
        _main([str(work_dir)])
    assert called["count"] == 0


# ---------------------------------------------------------------------------
# CLI: auto-runner kwargs (inline_inputs / user_compare / default_platform)
# ---------------------------------------------------------------------------


def test_cli_inline_inputs_fallback_when_golden_missing(tmp_path: Path) -> None:
    """When the auto-runner passes ``inline_inputs`` and ``golden.py`` is
    absent, ``_main`` must use the inline tensors and skip validation —
    instead of raising FileNotFoundError as the standalone path does."""
    work_dir = _make_build_output(tmp_path)
    inline_tensors = [torch.zeros(2), torch.ones(3)]
    captured: dict = {}

    def fake_replay(wd, *tensors, config, recompile, rebuild_from_pto, validate):
        captured["tensors"] = tensors
        captured["validate"] = validate

    with patch.object(replay_module, "replay", side_effect=fake_replay):
        rc = _main([str(work_dir)], inline_inputs=lambda: inline_tensors)
    assert rc == 0
    assert list(captured["tensors"]) == inline_tensors
    assert captured["validate"] is False


def test_cli_inline_inputs_default_validates_when_golden_exists(tmp_path: Path) -> None:
    """Auto-runner default is validate=on; when golden.py exists, use golden
    tensors and propagate validate=True to ``replay`` so ``compute_golden``
    actually runs."""
    work_dir = _make_build_output(tmp_path)
    _write_add_golden(work_dir)
    captured: dict = {}

    def fake_replay(wd, *tensors, config, recompile, rebuild_from_pto, validate):
        captured["validate"] = validate
        captured["tensor_count"] = len(tensors)

    with patch.object(replay_module, "replay", side_effect=fake_replay):
        _main([str(work_dir)], inline_inputs=lambda: [torch.zeros(99)])
    assert captured["validate"] is True
    assert captured["tensor_count"] == 3  # came from golden, not inline


def test_cli_no_validate_uses_inline_even_with_golden(tmp_path: Path) -> None:
    """``--no-validate`` from the auto-runner must short-circuit golden
    loading entirely and use the inline tensors. Otherwise a JIT user
    skipping validation would still pay the golden import cost."""
    work_dir = _make_build_output(tmp_path)
    _write_add_golden(work_dir)
    inline_marker = [torch.full((4,), 42.0)]
    captured: dict = {}

    def fake_replay(wd, *tensors, config, recompile, rebuild_from_pto, validate):
        captured["tensors"] = tensors
        captured["validate"] = validate

    with patch.object(replay_module, "replay", side_effect=fake_replay):
        _main([str(work_dir), "--no-validate"], inline_inputs=lambda: inline_marker)
    assert captured["validate"] is False
    assert captured["tensors"][0][0].item() == 42.0


def test_cli_user_compare_invoked_when_validation_skipped(tmp_path: Path) -> None:
    """``user_compare`` is the JIT comparison hook — it must run after
    replay when validation did NOT happen, with the same tensors that
    replay saw, so hand-written asserts can read in-place outputs."""
    work_dir = _make_build_output(tmp_path)
    inline_tensors = [torch.zeros(4), torch.ones(4)]
    seen: dict = {}

    def fake_compare(*tensors):
        seen["tensors"] = tensors

    with patch.object(replay_module, "replay"):
        _main([str(work_dir)], inline_inputs=lambda: inline_tensors, user_compare=fake_compare)
    assert seen["tensors"] == tuple(inline_tensors)


def test_cli_user_compare_not_invoked_when_validation_runs(tmp_path: Path) -> None:
    """Validation success and the user-compare hook are mutually exclusive
    — running both would make a hand-edited inline assertion fire against
    golden tensors and break the JIT user's mental model."""
    work_dir = _make_build_output(tmp_path)
    _write_add_golden(work_dir)
    seen: dict = {"called": False}

    def fake_compare(*tensors):
        seen["called"] = True

    with patch.object(replay_module, "replay"):
        _main(
            [str(work_dir)],
            inline_inputs=lambda: [torch.zeros(1)],
            user_compare=fake_compare,
        )
    assert seen["called"] is False


def test_cli_default_platform_kwarg_used_when_flag_omitted(tmp_path: Path) -> None:
    """The auto-runner bakes the compile-time platform into the
    ``default_platform`` kwarg so ``python debug/run.py`` (no flags) picks
    the right target. ``--platform`` on the CLI must still override."""
    work_dir = _make_build_output(tmp_path)
    captured: dict = {}

    def fake_replay(wd, *tensors, config, recompile, rebuild_from_pto, validate):
        captured["platform"] = config.platform

    with (
        patch.object(replay_module, "replay", side_effect=fake_replay),
        patch.object(replay_module, "_load_named_inputs_from_golden", return_value=[]),
    ):
        _main([str(work_dir)], default_platform="a5sim")
    assert captured["platform"] == "a5sim"


def test_cli_default_platform_overridden_by_explicit_flag(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    captured: dict = {}

    def fake_replay(wd, *tensors, config, recompile, rebuild_from_pto, validate):
        captured["platform"] = config.platform

    with (
        patch.object(replay_module, "replay", side_effect=fake_replay),
        patch.object(replay_module, "_load_named_inputs_from_golden", return_value=[]),
    ):
        _main([str(work_dir), "--platform", "a2a3sim"], default_platform="a5sim")
    assert captured["platform"] == "a2a3sim"


def test_cli_standalone_missing_golden_still_raises(tmp_path: Path) -> None:
    """Back-compat: when ``inline_inputs`` is not provided (standalone CLI),
    a missing ``golden.py`` must still raise — otherwise the standalone
    invocation would silently run with no tensors."""
    work_dir = _make_build_output(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"golden\.py"):
        _main([str(work_dir)])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

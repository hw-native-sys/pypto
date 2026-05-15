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

from pathlib import Path
import importlib
from unittest.mock import patch

import pytest
import torch
from pypto.runtime import RunConfig

# ``pypto.runtime.debug.__init__`` re-exports ``replay`` (the function),
# which shadows the ``replay`` submodule on attribute lookup. Resolve the
# module via importlib so ``patch.object(replay_module, "...")`` works.
replay_module = importlib.import_module("pypto.runtime.debug.replay")
_load_inputs_from_golden = replay_module._load_inputs_from_golden
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
    )
    with patch.object(replay_module, "execute_compiled") as ec:
        replay(work_dir, config=config)
    dfx = ec.call_args.kwargs["dfx"]
    assert dfx.enable_l2_swimlane is True
    assert dfx.enable_pmu == 2
    assert dfx.enable_dump_tensor is True
    assert dfx.enable_dep_gen is True


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


def test_replay_missing_kernel_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="kernel_config.py"):
        replay(tmp_path)


def test_replay_uses_default_run_config_when_none(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    with patch.object(replay_module, "execute_compiled") as ec:
        replay(work_dir)
    default_cfg = RunConfig()
    assert ec.call_args.kwargs["platform"] == default_cfg.platform
    assert ec.call_args.kwargs["device_id"] == default_cfg.device_id


# ---------------------------------------------------------------------------
# _load_inputs_from_golden
# ---------------------------------------------------------------------------


def test_load_inputs_from_golden_returns_values_in_order(tmp_path: Path) -> None:
    (tmp_path / "golden.py").write_text(
        "import torch\n"
        "def generate_inputs(params):\n"
        "    return {'x': torch.zeros(2), 'y': torch.ones(3), 'z': torch.full((4,), 7.0)}\n"
    )
    tensors = _load_inputs_from_golden(tmp_path)
    assert len(tensors) == 3
    assert tensors[0].shape == (2,)
    assert tensors[1].shape == (3,)
    assert tensors[2].shape == (4,)
    assert tensors[2][0].item() == 7.0


def test_load_inputs_from_golden_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="golden.py"):
        _load_inputs_from_golden(tmp_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_invokes_replay_with_dfx_flags(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    captured: dict = {}

    def fake_replay(wd, *tensors, config, recompile):
        captured["work_dir"] = wd
        captured["tensors"] = tensors
        captured["config"] = config
        captured["recompile"] = recompile

    with (
        patch.object(replay_module, "replay", side_effect=fake_replay),
        patch.object(replay_module, "_load_inputs_from_golden", return_value=[torch.zeros(1)]),
    ):
        rc = _main([str(work_dir), "--pmu", "2", "--swimlane", "--device-id", "5"])
    assert rc == 0
    assert captured["work_dir"] == work_dir
    assert captured["recompile"] is True
    cfg = captured["config"]
    assert cfg.enable_pmu == 2
    assert cfg.enable_l2_swimlane is True
    assert cfg.device_id == 5


def test_cli_no_recompile_flag(tmp_path: Path) -> None:
    work_dir = _make_build_output(tmp_path)
    captured: dict = {}

    def fake_replay(wd, *tensors, config, recompile):
        captured["recompile"] = recompile

    with (
        patch.object(replay_module, "replay", side_effect=fake_replay),
        patch.object(replay_module, "_load_inputs_from_golden", return_value=[]),
    ):
        _main([str(work_dir), "--no-recompile"])
    assert captured["recompile"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

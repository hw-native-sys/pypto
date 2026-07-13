# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for :mod:`pypto.runtime.debug.run_script_writer`."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import pytest
from pypto.ir.compiled_program import ParamInfo
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.runtime.debug.run_script_writer import write_run_script


class _FakeDataType:
    """Stand-in for an IR ``DataType`` — only ``str(dt)`` is consulted."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return self._name


def _info(name: str, direction: ParamDirection, shape: list[int] | None, dtype: str = "fp32") -> ParamInfo:
    return ParamInfo(name=name, direction=direction, shape=shape, dtype=cast(DataType, _FakeDataType(dtype)))


def test_writes_to_debug_subdir(tmp_path: Path) -> None:
    out = write_run_script(tmp_path, [_info("a", ParamDirection.In, [4, 4])])
    assert out == tmp_path / "debug" / "run.py"
    assert out.exists()


def test_emitted_script_is_syntactically_valid(tmp_path: Path) -> None:
    out = write_run_script(
        tmp_path,
        [
            _info("a", ParamDirection.In, [128, 128]),
            _info("b", ParamDirection.In, [128, 128]),
            _info("c", ParamDirection.Out, [128, 128]),
        ],
    )
    ast.parse(out.read_text())


def test_input_tensors_use_randn_outputs_use_zeros(tmp_path: Path) -> None:
    out = write_run_script(
        tmp_path,
        [
            _info("x", ParamDirection.In, [64], dtype="fp32"),
            _info("y", ParamDirection.Out, [64], dtype="fp32"),
            _info("z", ParamDirection.InOut, [64], dtype="fp32"),
        ],
    )
    text = out.read_text()
    assert "x = torch.randn((64,), dtype=torch.float32)" in text
    assert "y = torch.zeros((64,), dtype=torch.float32)" in text
    # InOut is treated as input — caller must provide initial values.
    assert "z = torch.randn((64,), dtype=torch.float32)" in text


def test_dtype_mapping_covers_common_types(tmp_path: Path) -> None:
    out = write_run_script(
        tmp_path,
        [
            _info("a", ParamDirection.In, [2], dtype="fp16"),
            _info("b", ParamDirection.In, [2], dtype="bfloat16"),
            _info("c", ParamDirection.In, [2], dtype="int32"),
            _info("d", ParamDirection.In, [2], dtype="bool"),
        ],
    )
    text = out.read_text()
    assert "dtype=torch.float16" in text
    assert "dtype=torch.bfloat16" in text
    assert "dtype=torch.int32" in text
    assert "dtype=torch.bool" in text


def test_dynamic_dim_filled_with_one_and_commented(tmp_path: Path) -> None:
    out = write_run_script(tmp_path, [_info("a", ParamDirection.In, [-1, 32])])
    text = out.read_text()
    assert "a = torch.randn((1, 32), dtype=torch.float32)" in text
    assert "dynamic dim" in text


def test_platform_baked_into_runner_kwarg(tmp_path: Path) -> None:
    out = write_run_script(tmp_path, [_info("a", ParamDirection.In, [4])], platform="a5sim")
    assert 'default_platform="a5sim"' in out.read_text()


def test_platform_defaults_to_a2a3sim(tmp_path: Path) -> None:
    out = write_run_script(tmp_path, [_info("a", ParamDirection.In, [4])])
    assert 'default_platform="a2a3sim"' in out.read_text()


def test_scalar_param_emits_placeholder(tmp_path: Path) -> None:
    # shape=None marks a scalar parameter — we punt on auto-materialisation
    # and emit a TODO line so the user can edit by hand.
    out = write_run_script(tmp_path, [_info("k", ParamDirection.In, None)])
    text = out.read_text()
    assert "k = None" in text
    assert "TODO" in text and "scalar" in text


def test_user_compare_hook_uses_param_names(tmp_path: Path) -> None:
    """JIT path has no golden.py — users edit ``_user_compare`` to write their
    own assertions. The hook must expose the IR's parameter names directly so
    the body can reference ``a``, ``b``, ``c`` without indexing into a list."""
    out = write_run_script(
        tmp_path,
        [
            _info("a", ParamDirection.In, [64]),
            _info("b", ParamDirection.In, [64]),
            _info("c", ParamDirection.Out, [64]),
        ],
    )
    text = out.read_text()
    assert "def _user_compare(a, b, c):" in text


def test_user_compare_with_no_params(tmp_path: Path) -> None:
    """Edge case: a kernel with zero IR params still produces a syntactically
    valid ``_user_compare()`` definition."""
    out = write_run_script(tmp_path, [])
    text = out.read_text()
    ast.parse(text)
    assert "def _user_compare():" in text


def test_delegates_cli_to_replay_main(tmp_path: Path) -> None:
    """The generated script must not re-implement the replay CLI — it just
    forwards work_dir + sys.argv into ``replay._main`` with its two hooks.
    Re-running the full argparse block would mean every new flag added to
    replay needs a parallel edit in run_script_writer."""
    out = write_run_script(tmp_path, [_info("a", ParamDirection.In, [4])])
    text = out.read_text()
    assert "from pypto.runtime.debug.replay import _main" in text
    assert "inline_inputs=_inline_inputs" in text
    assert "user_compare=_user_compare" in text
    # No CLI flags should be DEFINED locally — they all live in replay._main.
    # (Docstring mentions of flag names are fine; the regression we guard
    # against is a re-declared argparse block.)
    assert "argparse" not in text
    assert "parser.add_argument" not in text
    assert "configure_log" not in text
    assert "RunConfig(" not in text


def test_generated_script_stays_compact(tmp_path: Path) -> None:
    """Guard against the duplication regressing — the shim should be well
    under the line count of the original copy-paste version (~150 lines).
    Account for the in-file CLI flag listing (~25 docstring lines)."""
    out = write_run_script(tmp_path, [_info("a", ParamDirection.In, [4])])
    lines = out.read_text().splitlines()
    assert len(lines) < 110, f"generated script ballooned to {len(lines)} lines"


def test_docstring_lists_common_cli_flags(tmp_path: Path) -> None:
    """The whole point of the auto-runner is to give users a one-command
    debug entry. After moving the CLI to ``replay._main``, the flags aren't
    visible in the file anymore — so the docstring must enumerate them.
    Otherwise users have to ``--help`` or open replay.py to discover
    ``--pmu`` / ``--swimlane`` / ``--log-level`` / ``--no-validate``."""
    out = write_run_script(tmp_path, [_info("a", ParamDirection.In, [4])])
    text = out.read_text()
    for flag in (
        "--platform",
        "--pmu",
        "--swimlane",
        "--dump-args",
        "--dep-gen",
        "--no-recompile",
        "--no-rebuild-from-pto",
        "--log-level",
        "--log-sync-pypto",
        "--validate",
        "--no-validate",
    ):
        assert flag in text, f"docstring missing flag {flag}"
    # And the baked-in platform must show up in the listing so users know
    # what default they're running against.
    assert "default: a2a3sim" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

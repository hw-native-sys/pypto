# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproduce #3098's legacy producer bytes on silicon with the correct golden.

Only the emitted epilogue attributes are removed. The current compiler,
protocol, nosplit policy, input data and independent transpose golden stay fixed.
"""

import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from harness.core.test_runner import TestRunner
from pypto.backend import pto_backend
from test_c2v_fixpipe import _case, column_view

_PtoasCompiler = Callable[[str, str, list[str] | None], None]


def _legacy_compiler(real_ptoas: _PtoasCompiler, replay_root: Path) -> _PtoasCompiler:
    """Rewrite the replay's PTO source while forwarding unrelated compilations."""
    replay_root = replay_root.resolve()

    def compile_replay(pto_path: str, output_path: str, ptoas_flags: list[str] | None = None) -> None:
        source = Path(pto_path)
        # Both the harness and PTOAS itself compile concurrently. Artifact
        # ownership, not the worker thread or layout, identifies the replay.
        if not source.resolve().is_relative_to(replay_root):
            return real_ptoas(pto_path, output_path, ptoas_flags)
        text = source.read_text()
        assert "layout = nz2dn" in text
        text, count = re.subn(r", acc_push_epilogue = #pto\.acc_push_epilogue<[^>]+>", "", text)
        assert count == 2
        source.write_text(text)
        return real_ptoas(pto_path, output_path, ptoas_flags)

    return compile_replay


@pytest.mark.platforms("a5", reason="The legacy physical-byte oracle is validated on A5 hardware only.")
@pytest.mark.inline_case(reason="The replay must compile after its PTOAS monkeypatch is installed.")
def test_legacy_column_major_bytes_are_detected(test_runner, monkeypatch):
    def detect_legacy_bytes(actual, expected):
        correct = expected["out"]
        legacy = correct.T.contiguous().reshape(correct.shape)
        assert not torch.equal(legacy, correct)
        assert torch.equal(actual["out"], legacy), "unexpected legacy transport byte pattern"
        raise AssertionError("legacy row-major producer bytes reproduced; transpose golden rejected")

    case = _case(column_view, signed=False, buffer_ir=False)
    case.name = "legacy_column_major_byte_oracle"
    case.compare = detect_legacy_bytes
    # Keep the replay's artifact namespace independent of the active compile pool.
    # A copied config leaves the session runner and background cases unchanged.
    base = test_runner.config.save_kernels_dir or Path(__file__).resolve().parents[4] / "build_output"
    replay_root = Path(base) / f"legacy_replay_{uuid.uuid4().hex}"
    replay_root.mkdir(parents=True, mode=0o755)
    try:
        config = replace(test_runner.config, save_kernels=True, save_kernels_dir=str(replay_root))
        runner = TestRunner(config)
        hook = _legacy_compiler(pto_backend._run_ptoas, replay_root)
        monkeypatch.setattr(pto_backend, "_run_ptoas", hook)
        result = runner.run(case)
        assert not result.passed
        assert "legacy row-major producer bytes reproduced" in result.error, result.error
    finally:
        if not test_runner.config.save_kernels:
            shutil.rmtree(replay_root)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

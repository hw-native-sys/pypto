# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression checks for retaining evidence when hardware subprocesses fail."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

RUN = runpy.run_path(str(Path(__file__).with_name("test_validation.py")))["run"]


def test_nonzero_exit_keeps_diagnostics(tmp_path):
    with pytest.raises(AssertionError, match="exited 7"):
        RUN(tmp_path, "failure", [sys.executable, "-c", "print('launch failed'); raise SystemExit(7)"])
    assert "launch failed" in (tmp_path / "failure.log").read_text()
    assert sys.executable in (tmp_path / "failure.command.txt").read_text()


def test_timeout_keeps_partial_diagnostics(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        RUN(
            tmp_path,
            "timeout",
            [
                sys.executable,
                "-c",
                "import time; print('before launch', flush=True); time.sleep(30)",
            ],
            timeout=1,
        )
    assert (tmp_path / "timeout.log").read_text() == "TIMEOUT after 1s\nbefore launch\n"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

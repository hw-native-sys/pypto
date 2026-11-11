# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check trusted distributed-job routing without an NPU or runtime build."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("has_suite", [False, True])
@pytest.mark.parametrize("exit_code", [0, 1])
def test_distributed_job_routes_and_propagates_failure(
    tmp_path: Path, has_suite: bool, exit_code: int
) -> None:
    """Run the actual workflow shell with recording substitutes for execution."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["onboard-tests-a5-distributed"]["steps"]
    command = next(step["run"] for step in steps if "-m pytest" in step.get("run", ""))
    checkout = tmp_path / "a5-dist-checkout"
    checkout.mkdir()
    (checkout / "activate.sh").touch()
    if has_suite:
        suite = checkout / "tests/st/distributed/test_l3_urma_async.py"
        suite.parent.mkdir(parents=True)
        suite.touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "commands.log"
    for name in ["python", "task-submit"]:
        executable = bin_dir / name
        executable.write_text(
            f'#!/bin/bash\nprintf "%s %s\\n" "{name}" "$*" >> "$COMMAND_LOG"\nexit {exit_code}\n'
        )
        executable.chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", command],
        cwd=checkout,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GITHUB_WORKSPACE": str(tmp_path),
            "COMMAND_LOG": str(log),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    executed = log.read_text()
    if has_suite:
        assert "task-submit --device auto --device-num 4" in executed
        assert "tests/st/distributed/test_l3_urma_async.py" in executed
        assert "test_ci_placeholder.py" not in executed
    else:
        assert "python -m pytest tests/lint/test_ci_placeholder.py" in executed
        assert "task-submit" not in executed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

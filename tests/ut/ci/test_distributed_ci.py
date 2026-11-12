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

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("exit_code", [0, 1])
def test_distributed_job_routes_and_propagates_failure(tmp_path: Path, exit_code: int) -> None:
    """Run the actual workflow shell with recording substitutes for execution."""
    workflow_path = ROOT / ".github/workflows/ci-full.yml"
    workflow_text = workflow_path.read_text()
    workflow = yaml.safe_load(workflow_text)
    steps = workflow["jobs"]["onboard-tests-a5-distributed"]["steps"]
    command = next(step["run"] for step in steps if "-m pytest" in step.get("run", ""))
    # A rename/move of the suite must not leave the unconditional lane green.
    assert (ROOT / "tests/st/distributed/test_l3_urma_async.py").is_file()
    # The placeholder fallback is retired: the real suite is the only path.
    assert "test_ci_placeholder.py" not in workflow_text
    assert "if [ -f tests/st/distributed/test_l3_urma_async.py ]" not in command
    checkout = tmp_path / "a5-dist-checkout"
    checkout.mkdir()
    (checkout / "activate.sh").touch()
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
    assert "task-submit --device auto --device-num 4" in executed
    assert "tests/st/distributed/test_l3_urma_async.py" in executed
    assert "test_ci_placeholder.py" not in executed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

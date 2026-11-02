# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device preflight and independent artifact directories for VF acceptance tests."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def vpto_onboard_session() -> Path | None:
    if os.environ.get("PYPTO_RUN_VPTO_ONBOARD") != "1":
        return None
    device = int(os.environ["PYPTO_VPTO_DEVICE_ID"])
    assert device >= 0, "VF acceptance requires an explicit nonnegative device ID"
    output = Path(
        os.environ.get(
            "PYPTO_VPTO_RUNTIME_OUTPUT", Path(__file__).resolve().parents[4] / "build/vpto-onboard"
        )
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Probe in a bounded child so the pytest parent never retains an NPU context.
    probe = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("device_probe.py")),
            str(device),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (output / "device.txt").write_text(f"device_id={device}\n{probe.stdout}\n{probe.stderr}")
    assert probe.returncode == 0, f"Cannot probe VF device {device}: {probe.stdout}\n{probe.stderr}"
    names = [
        line.removeprefix("VPTO_DEVICE_NAME=")
        for line in probe.stdout.splitlines()
        if line.startswith("VPTO_DEVICE_NAME=")
    ]
    assert len(names) == 1 and names[0].startswith("Ascend950"), (
        f"VF acceptance requires an A5 (Ascend950) device; device {device} reported {names}"
    )
    return output


@pytest.fixture(autouse=True)
def vpto_test_output(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, vpto_onboard_session: Path | None
) -> None:
    if vpto_onboard_session is not None:
        # Several cases compile into 'original' and relocate it. Sharing that
        # directory would make a suite run reuse or overwrite another case's evidence.
        output = tempfile.mkdtemp(prefix=f"{request.node.name}-", dir=vpto_onboard_session)
        monkeypatch.setenv("PYPTO_VPTO_RUNTIME_OUTPUT", output)

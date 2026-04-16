# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Environment configuration for Simpler dependency.

Simpler is bundled as a git submodule at the repository root (``runtime/``).
"""

from pathlib import Path

_SIMPLER_ROOT = Path(__file__).resolve().parents[4] / "runtime"


def get_simpler_root() -> Path:
    """Return the Simpler submodule root directory."""
    return _SIMPLER_ROOT


def get_simpler_python_path() -> Path:
    """Get Simpler Python package path (simpler/python directory)."""
    return _SIMPLER_ROOT / "python"


def get_simpler_scripts_path() -> Path:
    """Get Simpler scripts path (simpler/examples/scripts directory)."""
    return _SIMPLER_ROOT / "examples" / "scripts"


def is_hardware_available() -> bool:
    """Check if Ascend NPU hardware is available.

    Checks for common Ascend NPU device nodes:
    - /dev/davinci*
    - /dev/npu*
    - /dev/ascend*

    Returns:
        True if any Ascend NPU device files exist, False otherwise.
    """
    dev_path = Path("/dev")
    if not dev_path.exists():
        return False

    # Check for various Ascend NPU device node patterns
    device_patterns = ["davinci*", "npu*", "ascend*"]
    for pattern in device_patterns:
        if any(dev_path.glob(pattern)):
            return True

    return False

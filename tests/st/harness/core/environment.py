# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Environment configuration for Simpler dependency.

This module manages the Simpler runtime dependency, which is external
to the PyPTO project and required for test execution.
"""

import os
from pathlib import Path
from typing import Optional


class PtoEnvironmentError(Exception):
    """Environment configuration error exception"""

    pass


def get_simpler_root() -> Optional[Path]:
    """Get Simpler root directory from SIMPLER_ROOT environment variable.

    Returns:
        Simpler root directory, or None if not set.
    """
    if "SIMPLER_ROOT" in os.environ:
        return Path(os.environ["SIMPLER_ROOT"])
    return None


def get_simpler_python_path() -> Optional[Path]:
    """Get Simpler Python package path (simpler/python directory)."""
    root = get_simpler_root()
    if root is None:
        return None
    return root / "python"


def get_simpler_scripts_path() -> Optional[Path]:
    """Get Simpler scripts path (simpler/examples/scripts directory)."""
    root = get_simpler_root()
    if root is None:
        return None
    return root / "examples" / "scripts"


def ensure_simpler_available() -> Path:
    """Ensure Simpler is available, raise error if SIMPLER_ROOT is not set.

    Returns:
        Simpler root directory

    Raises:
        PtoEnvironmentError: When SIMPLER_ROOT is not set
    """
    root = get_simpler_root()
    if root is not None:
        return root

    raise PtoEnvironmentError(
        "Simpler runtime is not available.\n\n"
        "Please set the SIMPLER_ROOT environment variable:\n"
        "  export SIMPLER_ROOT=/path/to/your/simpler"
    )


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

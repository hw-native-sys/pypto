# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""pytest configuration for runtime tests.

Runtime tests execute on hardware or simulator and are automatically
skipped when hardware is not available.
"""

import pytest
from harness.core.environment import is_hardware_available
from harness.core.harness import ONBOARD_PLATFORM_IDS


@pytest.fixture(scope="session", autouse=True)
def check_hardware_availability(request):
    """Skip all runtime tests if hardware is not available.

    This fixture checks for Ascend NPU device nodes (/dev/davinci*, /dev/npu*,
    /dev/ascend*). If none are found and the requested platform set only
    contains onboard (real-hardware) platforms (e.g. ``a2a3``/``a5``), all
    tests in the runtime directory are skipped with a hint to switch to a
    simulator platform.
    """
    raw = request.config.getoption("--platform")
    requested = {p.strip() for p in str(raw).split(",") if p.strip()}
    onboard_set = set(ONBOARD_PLATFORM_IDS)
    if requested and requested.issubset(onboard_set) and not is_hardware_available():
        pytest.skip(
            "Hardware not available: Ascend NPU device nodes not found "
            "(checked /dev/davinci*, /dev/npu*, /dev/ascend*). "
            "Use --platform=a2a3sim or --platform=a5sim to run on simulator."
        )

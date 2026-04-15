# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fuzz-specific conftest ensuring simpler paths survive pytest-forked forks."""

import os
import sys


def pytest_configure(config):
    """Inject simpler paths into sys.path at the earliest hook.

    pytest-forked forks the process after collection but inherits the
    parent's sys.path.  By setting the paths here (before any fork)
    rather than in a session fixture, we guarantee that forked children
    see the correct paths for ``simpler_setup``, ``simpler``, etc.
    """
    simpler_root = os.environ.get("SIMPLER_ROOT")
    if not simpler_root:
        return

    # simpler root for simpler_setup package; python/ for the simpler package.
    for p in (simpler_root, os.path.join(simpler_root, "python")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

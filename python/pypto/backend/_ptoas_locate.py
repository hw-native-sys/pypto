# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared discovery of the ``ptoas`` executable under ``$PTOAS_ROOT``."""

import os
import shutil

# Probed in order under $PTOAS_ROOT — the release tarball layout changed at
# v0.51 and the two entries are NOT interchangeable:
#
# - up to v0.50, `<root>/ptoas` is a shell launcher that exports
#   `LD_LIBRARY_PATH=<root>/lib` before exec'ing `<root>/bin/ptoas`. That bare
#   binary has no RUNPATH, so invoking it directly dies with
#   "libMLIR*.so: cannot open shared object file". The launcher must win.
# - from v0.51, `<root>/ptoas` is a Python package *directory* (not executable),
#   and `<root>/bin/ptoas` links self-sufficiently — so the probe falls through
#   to it.
#
# Hence launcher-first: it is correct on both layouts, whereas `bin/ptoas`-first
# silently breaks every pre-v0.51 toolchain.
PTOAS_RELATIVE_PATHS = ("ptoas", "bin/ptoas")


def find_ptoas_binary() -> str | None:
    """Locate the ``ptoas`` executable.

    When ``PTOAS_ROOT`` is set only that directory is searched — falling back to
    ``PATH`` would silently compile with a different PTOAS than the pinned one.

    Returns:
        Path to the executable, or ``None`` when no executable ``ptoas`` exists.
    """
    ptoas_root = os.environ.get("PTOAS_ROOT")
    if not ptoas_root:
        return shutil.which("ptoas")

    for relative in PTOAS_RELATIVE_PATHS:
        candidate = os.path.join(ptoas_root, relative)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None

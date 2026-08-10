# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Helpers for asserting on generated ``.pto`` lines that carry MLIR locations.

PTO codegen suffixes every emitted operation with ``loc("file":line:col)`` (see
``tests/ut/codegen/test_pto_codegen_source_loc.py``). Assertions that anchor to
the *end* of an operation -- ``endswith(...)``, ``strip() == ...`` -- must drop
that suffix first, or they assert on the source path of the test file.
"""

import re

# Trailing ` loc("<path>":<line>:<col>)`; the path may contain escaped quotes.
_TRAILING_LOC_RE = re.compile(r'\s+loc\("(?:[^"\\]|\\.)*":\d+:\d+\)\s*$')


def strip_loc(line: str) -> str:
    """Return ``line`` without its trailing MLIR location and surrounding space.

    Safe to call on lines that carry no location -- they are returned stripped
    of whitespace only, so a call site reads the same either way.

    Args:
        line: One line of generated ``.pto`` text.

    Returns:
        The operation text with any trailing ``loc(...)`` removed and both ends
        whitespace-stripped.
    """
    return _TRAILING_LOC_RE.sub("", line).strip()

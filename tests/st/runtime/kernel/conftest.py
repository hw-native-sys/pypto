# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Kernel device test controls; keep PR checks small and profiling explicit."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Select a small correctness run or a full latency measurement."""
    parser.addoption(
        "--kernel-perf-samples",
        type=int,
        choices=(64, 1024),
        default=64,
        help="Samples per kernel hot-path variant: 64 for regression, 1024 for performance measurement",
    )

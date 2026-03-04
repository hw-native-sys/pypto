# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Core module for test case definitions and execution.

Re-exports from pypto.runtime for backward compatibility.
"""

from pypto.runtime import (
    PTOTestCase,
    TensorSpec,
    ensure_simpler_available,
)
from pypto.runtime import (
    RunConfig as TestConfig,
)
from pypto.runtime import (
    Runner as TestRunner,
)
from pypto.runtime import (
    RunResult as TestResult,
)

__all__ = [
    "PTOTestCase",
    "TensorSpec",
    "TestConfig",
    "TestResult",
    "TestRunner",
    "ensure_simpler_available",
]

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PTO Testing Framework

End-to-end testing framework for PyPTO frontend and Simpler runtime.
Re-exports from pypto.runtime for backward compatibility.
"""

from pypto.runtime import (
    DataType,
    GoldenGenerator,
    ProgramCodeGenerator,
    PTOTestCase,
    TensorSpec,
    TestSuite,
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

__version__ = "0.1.0"
__all__ = [
    # Core
    "PTOTestCase",
    "TensorSpec",
    "TestConfig",
    "TestResult",
    "DataType",
    "TestRunner",
    "TestSuite",
    # Adapters
    "ProgramCodeGenerator",
    "GoldenGenerator",
]

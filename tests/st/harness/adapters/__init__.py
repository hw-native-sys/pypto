# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Adapters module for PTO testing framework.

This module provides adapters that bridge PyPTO programs and test specifications
to the files required by simpler's CodeRunner:
- program_generator: PyPTO Program -> CCE C++ kernel files + orchestration
- golden_generator: PTOTestCase -> golden.py

Note: kernel_config.py is generated directly by ir.compile() in C++ code.
"""

from harness.adapters.golden_generator import GoldenGenerator
from harness.adapters.program_generator import ProgramCodeGenerator

__all__ = [
    "ProgramCodeGenerator",
    "GoldenGenerator",
]

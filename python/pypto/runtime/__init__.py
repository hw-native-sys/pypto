# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO Runtime

Runtime framework for compiling and executing PyPTO programs on
simulation and hardware platforms via Simpler's CodeRunner.
"""

from pypto.runtime.environment import (
    PtoEnvironmentError,
    ensure_simpler_available,
    get_simpler_python_path,
    get_simpler_root,
    get_simpler_scripts_path,
    is_hardware_available,
    setup_simpler_paths,
)
from pypto.runtime.golden_generator import GoldenGenerator
from pypto.runtime.harness import (
    DataType,
    PTOTestCase,
    RunConfig,
    RunResult,
    TensorSpec,
    TestConfig,
    TestResult,
)
from pypto.runtime.program_generator import ProgramCodeGenerator
from pypto.runtime.runner import Runner, TestRunner, TestSuite, run

__version__ = "0.1.0"
__all__ = [
    # Top-level convenience function
    "run",
    # Core data structures
    "DataType",
    "TensorSpec",
    "RunConfig",
    "RunResult",
    "PTOTestCase",
    # Runner
    "Runner",
    "TestSuite",
    # Code generators
    "ProgramCodeGenerator",
    "GoldenGenerator",
    # Environment
    "PtoEnvironmentError",
    "ensure_simpler_available",
    "setup_simpler_paths",
    "get_simpler_root",
    "get_simpler_python_path",
    "get_simpler_scripts_path",
    "is_hardware_available",
    # Backward-compatible aliases
    "TestConfig",
    "TestResult",
    "TestRunner",
]

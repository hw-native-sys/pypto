# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Internal implementation modules for the fuzzer framework.

Subpackage layout:

    core/           - OpSpec, shape utilities, OpFuzzer
    body/           - Composable body AST, generator, codegen, golden

Top-level modules:

    kernel_generator           - KernelGenerator: kernel code generation
    golden_generator           - Golden reference dispatch
    orchestrator_generator     - Orchestration function generation
    multi_kernel_test_generator - MultiKernelTestGenerator: test class assembly
"""

from .body.ast import BodyNode, ForBlock, IfElseBlock, OpBlock
from .body.generator import BodyGenerator
from .core.fuzzer import OpChainConfig, OpFuzzer
from .core.op_specs import OpSpec
from .kernel_generator import KernelGenerator
from .multi_kernel_test_generator import MultiKernelTestGenerator
from .orchestrator_generator import OrchestratorGenerator

__all__ = [
    "BodyNode",
    "BodyGenerator",
    "ForBlock",
    "IfElseBlock",
    "KernelGenerator",
    "MultiKernelTestGenerator",
    "OpBlock",
    "OpChainConfig",
    "OpFuzzer",
    "OpSpec",
    "OrchestratorGenerator",
]

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Body AST layer: composable control flow with nested for-loops and if/else."""

from .ast import BodyNode, ForBlock, IfElseBlock, OpBlock
from .codegen import generate_body_code
from .generator import BodyGenerator
from .golden import generate_body_golden_lines

__all__ = [
    "BodyGenerator",
    "BodyNode",
    "ForBlock",
    "IfElseBlock",
    "OpBlock",
    "generate_body_code",
    "generate_body_golden_lines",
]

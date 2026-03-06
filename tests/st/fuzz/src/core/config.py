# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Configuration dataclasses for fuzz test generation."""

from dataclasses import dataclass


@dataclass
class ControlFlowConfig:
    """Configuration for control flow generation in fuzz tests.

    Groups parameters that control for-loop and if/else generation,
    nesting depth, and probability decay.
    """

    enable_for_loop: bool = False
    max_for_loop_iterations: int = 4
    enable_if_else: bool = False
    for_loop_probability: float = 1.0
    if_else_probability: float = 1.0
    max_depth: int = 1
    depth_decay: float = 0.5

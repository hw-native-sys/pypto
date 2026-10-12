# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit low-level Buffer pipeline for recipes beyond default acceptance."""

from pypto import passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def buffer_recipe_pipeline() -> passes.PassPipeline:
    """Exercise all production recipes without the normal rollout selection pass.

    These tests validate the explicit lowering API, including unsupported-recipe
    diagnostics. Default selection is tested separately through PassManager.
    """
    manager = PassManager(OptimizationStrategy.Default)
    pipeline = passes.PassPipeline()
    for transform in manager.passes:
        if transform.get_name() != "SelectBufferIRFunctions":
            pipeline.add_pass(transform)
    return pipeline

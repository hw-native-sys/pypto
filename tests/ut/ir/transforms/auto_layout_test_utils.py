# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pipeline utilities shared by AutoLayout tests."""

from pypto import ir, passes


def run_through_auto_layout(program: ir.Program) -> ir.Program:
    """Run the AutoLayout pipeline up to and including layout selection."""
    prefix = passes.PassPipeline()
    for step in ir.PassManager.get_strategy(ir.OptimizationStrategy.AutoLayout).passes:
        prefix.add_pass(step)
        if step.get_name() == "AutoLayout":
            return prefix.run(program)
    raise AssertionError("AutoLayout is missing from the pipeline")

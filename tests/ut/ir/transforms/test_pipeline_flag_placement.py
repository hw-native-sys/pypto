# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reject explicit flags before software-pipeline scheduling can reorder them."""

import pypto
import pypto.language as pl
import pytest
from pypto import passes


def test_flags_inside_pipeline_rejected_before_skew():
    @pl.program
    class FlaggedPipeline:
        @pl.function(strict_ssa=True)
        def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
            for _ in pl.pipeline(0, 4, stage=2):
                for _ in pl.range(1):
                    pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.V, 0)
                    pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.V, 0)
            return x

    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.PipelineLoopValid)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, FlaggedPipeline)
    assert len(diagnostics) == 2
    assert all(d.rule_name == "PipelineLoopValid" for d in diagnostics)

    pipeline = passes.PassPipeline()
    pipeline.add_pass(passes.skew_cross_core_pipeline())
    with pytest.raises(pypto.Error, match="cannot be used inside pl.pipeline"):
        pipeline.run(FlaggedPipeline)


def test_flags_in_ordinary_loop_are_allowed():
    @pl.program
    class OrdinaryLoop:
        @pl.function(strict_ssa=True)
        def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
            for _ in pl.range(4):
                pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.V, 0)
                pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.V, 0)
            return x

    pipeline = passes.PassPipeline()
    pipeline.add_pass(passes.skew_cross_core_pipeline())
    assert pipeline.run(OrdinaryLoop) is not None

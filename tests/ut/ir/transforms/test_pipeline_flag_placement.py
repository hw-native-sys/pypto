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


def test_flags_inside_pipeline_rejected_before_scheduling():
    @pl.program
    class FlaggedPipeline:
        @pl.function(strict_ssa=True)
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            for _ in pl.pipeline(0, 4, stage=2):
                _tile: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.load(x, [0, 0], [16, 16])
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

    for scheduler in (
        passes.skew_cross_core_pipeline,
        passes.lower_pipeline_loops,
        passes.lower_pipeline_to_slots,
        passes.canonicalize_io_order,
    ):
        pipeline = passes.PassPipeline()
        pipeline.add_pass(scheduler())
        with passes.PassContext([], passes.VerificationLevel.NONE, memory_planner=passes.MemoryPlanner.PTOAS):
            with pytest.raises(ValueError, match="cannot be used inside pl.pipeline"):
                pipeline.run(FlaggedPipeline)


@pytest.mark.parametrize("flag_name", ["set_flag", "wait_flag"])
def test_standalone_io_canonicalization_rejects_flags_before_demotion(flag_name):
    program = pl.parse_program(f"""
@pl.program
class FlaggedPipeline:
    @pl.function(strict_ssa=True)
    def main(self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Tensor[[16, 16], pl.FP32]):
        for i in pl.pipeline(0, 4, stage=1):
            tile = pl.tile.load(x, [0, 0], [16, 16])
            pl.tile.store(tile, [0, 0], out)
            pl.system.{flag_name}(pl.PipeType.MTE3, pl.PipeType.MTE2, 0)
""")
    with passes.PassContext([], passes.VerificationLevel.NONE):
        with pytest.raises(ValueError, match="cannot be used inside pl.pipeline"):
            passes.canonicalize_io_order()(program)


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
    assert passes.canonicalize_io_order()(OrdinaryLoop) is OrdinaryLoop


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

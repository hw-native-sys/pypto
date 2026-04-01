# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end test for tensor.gather orchestration codegen pipeline.

Verifies the complete compilation pipeline: DSL → IR → PassManager → codegen
for programs using tensor.gather in orchestration functions.

Test scenarios:
  1. 2D gather with dim=0 (row selection)
  2. 2D gather with dim=1 (MoE top-k expert selection pattern)
  3. 3D gather with dim=1 (batch attention pattern)
  4. 3D gather with dim=2 (last-dim selection)
  5. 2D gather with negative dim (dim=-1 → dim=1)
"""

import pypto.language as pl
import pytest
from pypto import backend, codegen
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import ir


def _generate_orch_code(program) -> str:
    """Generate orchestration code through the full pipeline."""
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    optimized = pm.run_passes(program)
    for func in optimized.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            result = codegen.generate_orchestration(optimized, func)
            return result.code
    raise ValueError("No orchestration function found in program")


class TestGatherOrchCodegenPipeline:
    """End-to-end tests for tensor.gather through the full codegen pipeline."""

    def test_gather_2d_dim0_pipeline(self):
        """Full pipeline: 2D gather with dim=0 (row selection).

        Pattern: select specific rows from a table.
        Input [8, 4], Index [3, 4] → Output [3, 4]
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class GatherDim0:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_process(
                self,
                t: pl.Tensor[[3, 4], pl.FP32],
                out: pl.Out[pl.Tensor[[3, 4], pl.FP32]],
            ) -> pl.Tensor[[3, 4], pl.FP32]:
                tile: pl.Tile[[3, 4], pl.FP32] = pl.load(t, [0, 0], [3, 4])
                return pl.store(tile, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                inp: pl.Tensor[[8, 4], pl.FP32],
                idx: pl.Tensor[[3, 4], pl.INT32],
                out: pl.Out[pl.Tensor[[3, 4], pl.FP32]],
            ) -> pl.Tensor[[3, 4], pl.FP32]:
                gathered: pl.Tensor[[3, 4], pl.FP32] = pl.gather(inp, 0, idx)
                out = self.kernel_process(gathered, out)
                return out

        code = _generate_orch_code(GatherDim0)

        # Verify output tensor creation
        assert "make_tensor(" in code
        assert "DataType::FLOAT32" in code

        # Verify pointer casts for input and index
        assert "static_cast<const float*>" in code
        assert "static_cast<const int32_t*>" in code
        assert "static_cast<float*>" in code

        # Verify gather loop structure
        assert "for (size_t" in code

        # Verify dim=0 coordinate replacement
        assert "[0] = static_cast<size_t>" in code

        # Verify kernel task submission
        assert "pto2_rt_submit_aiv_task" in code

    def test_gather_2d_dim1_moe_pipeline(self):
        """Full pipeline: 2D gather with dim=1 (MoE top-k expert selection).

        Pattern: weights [16, 64] (16 tokens, 64 experts),
                 topk_ids [16, 4] → selected weights [16, 4]
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class GatherMoE:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_process(
                self,
                t: pl.Tensor[[16, 4], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 4], pl.FP16]],
            ) -> pl.Tensor[[16, 4], pl.FP16]:
                tile: pl.Tile[[16, 4], pl.FP16] = pl.load(t, [0, 0], [16, 4])
                return pl.store(tile, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                weights: pl.Tensor[[16, 64], pl.FP16],
                topk_ids: pl.Tensor[[16, 4], pl.INT32],
                out: pl.Out[pl.Tensor[[16, 4], pl.FP16]],
            ) -> pl.Tensor[[16, 4], pl.FP16]:
                selected: pl.Tensor[[16, 4], pl.FP16] = pl.gather(weights, 1, topk_ids)
                out = self.kernel_process(selected, out)
                return out

        code = _generate_orch_code(GatherMoE)

        # Output tensor dtype matches input (FP16), not index (INT32)
        assert "DataType::FLOAT16" in code

        # Output tensor shape matches index [16, 4], not input [16, 64]
        assert "{(uint32_t)(16), (uint32_t)(4)}" in code
        assert "make_tensor(" in code

        # Input shape [16, 64] used for stride computation (distinct from output)
        assert "{(size_t)(16), (size_t)(64)}" in code

        # Verify dim=1 coordinate replacement
        assert "[1] = static_cast<size_t>" in code

    def test_gather_3d_pipeline(self):
        """Full pipeline: 3D gather with dim=1 (batch attention pattern).

        Pattern: Input [4, 8, 16], Index [4, 3, 16] → Output [4, 3, 16]
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Gather3D:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_process(
                self,
                t: pl.Tensor[[4, 3, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 3, 16], pl.FP32]],
            ) -> pl.Tensor[[4, 3, 16], pl.FP32]:
                tile: pl.Tile[[4, 3, 16], pl.FP32] = pl.load(t, [0, 0, 0], [4, 3, 16])
                return pl.store(tile, [0, 0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                inp: pl.Tensor[[4, 8, 16], pl.FP32],
                idx: pl.Tensor[[4, 3, 16], pl.INT64],
                out: pl.Out[pl.Tensor[[4, 3, 16], pl.FP32]],
            ) -> pl.Tensor[[4, 3, 16], pl.FP32]:
                gathered: pl.Tensor[[4, 3, 16], pl.FP32] = pl.gather(inp, 1, idx)
                out = self.kernel_process(gathered, out)
                return out

        code = _generate_orch_code(Gather3D)

        # Verify 3D tensor handling
        assert "[3]" in code  # 3-element shape/stride arrays

        # Verify dim=1 coordinate replacement for 3D
        assert "[1] = static_cast<size_t>" in code

        # Verify INT64 index pointer cast
        assert "static_cast<const int64_t*>" in code

        # Verify stride computation loop for 3D
        assert "ist_" in code

    def test_gather_3d_dim2_pipeline(self):
        """Full pipeline: 3D gather with dim=2 (last-dim selection).

        Pattern: Input [4, 8, 16], Index [4, 8, 5] → Output [4, 8, 5]
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Gather3DDim2:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_process(
                self,
                t: pl.Tensor[[4, 8, 5], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8, 5], pl.FP32]],
            ) -> pl.Tensor[[4, 8, 5], pl.FP32]:
                tile: pl.Tile[[4, 8, 5], pl.FP32] = pl.load(t, [0, 0, 0], [4, 8, 5])
                return pl.store(tile, [0, 0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                inp: pl.Tensor[[4, 8, 16], pl.FP32],
                idx: pl.Tensor[[4, 8, 5], pl.INT32],
                out: pl.Out[pl.Tensor[[4, 8, 5], pl.FP32]],
            ) -> pl.Tensor[[4, 8, 5], pl.FP32]:
                gathered: pl.Tensor[[4, 8, 5], pl.FP32] = pl.gather(inp, 2, idx)
                out = self.kernel_process(gathered, out)
                return out

        code = _generate_orch_code(Gather3DDim2)

        # Verify dim=2 coordinate replacement
        assert "[2] = static_cast<size_t>" in code

        # Verify output shape matches index [4, 8, 5], not input [4, 8, 16]
        assert "{(uint32_t)(4), (uint32_t)(8), (uint32_t)(5)}" in code

        # Verify input shape [4, 8, 16] used for stride computation
        assert "{(size_t)(4), (size_t)(8), (size_t)(16)}" in code

    def test_gather_negative_dim_pipeline(self):
        """Full pipeline: 2D gather with dim=-1 (normalizes to dim=1).

        Pattern: Input [8, 16], Index [8, 4] → Output [8, 4]
        Verifies negative dim is correctly normalized through the pipeline.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class GatherNegDim:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_process(
                self,
                t: pl.Tensor[[8, 4], pl.FP32],
                out: pl.Out[pl.Tensor[[8, 4], pl.FP32]],
            ) -> pl.Tensor[[8, 4], pl.FP32]:
                tile: pl.Tile[[8, 4], pl.FP32] = pl.load(t, [0, 0], [8, 4])
                return pl.store(tile, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                inp: pl.Tensor[[8, 16], pl.FP32],
                idx: pl.Tensor[[8, 4], pl.INT32],
                out: pl.Out[pl.Tensor[[8, 4], pl.FP32]],
            ) -> pl.Tensor[[8, 4], pl.FP32]:
                gathered: pl.Tensor[[8, 4], pl.FP32] = pl.gather(inp, -1, idx)
                out = self.kernel_process(gathered, out)
                return out

        code = _generate_orch_code(GatherNegDim)

        # dim=-1 normalizes to dim=1 for 2D tensor
        assert "[1] = static_cast<size_t>" in code

        # Verify output shape matches index [8, 4]
        assert "{(uint32_t)(8), (uint32_t)(4)}" in code

        # Verify output dtype matches input (FP32)
        assert "DataType::FLOAT32" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

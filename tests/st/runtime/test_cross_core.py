# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
a2a3 Cross-Core Communication (TPUSH/TPOP) System Test.

Program under test:
  Vector (AIV): loads tiles a and b, computes (a+b) and (a-b), pushes both to Cube.
  Cube  (AIC) : pops both tiles, performs matmul((a+b), (a-b)), stores result.
  Golden      : output = (a + b) @ (a - b)

Artifact-level split/layout validation is covered by dedicated unit tests and
on-board execution checks.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec


@pl.program
class CrossCoreTpushTpopProgram:
    """V2C updown-split cross-core program.

    Vector producer: loads tiles a and b, computes add and sub, pushes both to Cube.
    Cube consumer: pops tiles, performs matmul, stores result.
    """

    @pl.function(type=pl.FunctionType.AIV)
    def vector_producer(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
        pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=v2c_peer)

        tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
        tile_b: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
        result_add: pl.Tile[[16, 16], pl.FP32] = pl.add(tile_a, tile_b)
        result_sub: pl.Tile[[16, 16], pl.FP32] = pl.sub(tile_a, tile_b)

        pl.tpush_to_aic(result_add, split=1)
        pl.tpush_to_aic(result_sub, split=1)

    @pl.function(type=pl.FunctionType.AIC)
    def cube_consumer(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
        pl.aic_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=pipe_buf)

        # Chain 1: tpop -> move (use) -> tfree
        received_add: pl.Tile[
            [16, 16],
            pl.FP32,
            pl.MemorySpace.Mat,
            pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
        ] = pl.tpop_from_aiv(split=1)
        received_add_left = pl.move(received_add, target_memory=pl.Mem.Left)
        pl.tfree_to_aiv(received_add)

        # Chain 2: tpop -> move (use) -> tfree
        received_sub: pl.Tile[
            [16, 16],
            pl.FP32,
            pl.MemorySpace.Mat,
            pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
        ] = pl.tpop_from_aiv(split=1)
        received_sub_right = pl.move(received_sub, target_memory=pl.Mem.Right)
        pl.tfree_to_aiv(received_sub)

        mm_result: pl.Tile[[16, 16], pl.FP32] = pl.matmul(received_add_left, received_sub_right)

        updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(mm_result, [0, 0], output)
        return updated

    @pl.function(type=pl.FunctionType.Group)
    def group_func(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        updated = self.cube_consumer(a, b, output)
        self.vector_producer(a, b, output)
        return updated

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        out = self.group_func(a, b, output)
        return out


class CrossCoreTpushTpop(PTOTestCase):
    """a2a3 cross-core V2C: output = (a + b) @ (a - b)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_tpush_tpop_v2c_updown_16x16"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [16, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [16, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [16, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return CrossCoreTpushTpopProgram

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].float()
        b = tensors["b"].float()
        tensors["output"][:] = torch.matmul(a + b, a - b)


class TestCrossCore:
    """a2a3 cross-core communication system tests"""

    def test_tpush_tpop_v2c_updown(self, test_runner):
        """V2C updown pipe: compile through full pipeline and verify kernel artifacts."""
        test_case = CrossCoreTpushTpop()
        result = test_runner.run(test_case)
        assert result.passed, f"Cross-core V2C compilation failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

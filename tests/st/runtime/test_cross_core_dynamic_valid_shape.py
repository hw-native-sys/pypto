# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime test for C2V tpop with dynamic valid_shape operands."""

import sys
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

ROWS = 16
COLS = 16
VALID_ROWS = 8
VALID_COLS = 12
SLOT_SIZE_BYTES = ROWS * COLS * 4
BUFFER_SIZE_BYTES = SLOT_SIZE_BYTES * 4


class C2VDynamicTpopValidShapeTestCase(PTOTestCase):
    """Cube pushes a full tile; vector tpop consumes only dynamic valid_shape."""

    __test__ = False

    def __init__(
        self,
        *,
        platform: str | None = None,
        config: RunConfig | None = None,
    ):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"cross_core_c2v_dynamic_tpop_valid_shape_{VALID_ROWS}x{VALID_COLS}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [ROWS, COLS], DataType.BF16, init_value=1.0),
            TensorSpec("b", [ROWS, COLS], DataType.BF16, init_value=2.0),
            TensorSpec(
                "valid_shape",
                [2],
                DataType.INT64,
                init_value=torch.tensor([VALID_ROWS, VALID_COLS], dtype=torch.int64),
            ),
            TensorSpec("output", [ROWS, COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class C2VDynamicTpopValidShapeProgram:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def cube_producer(
                self,
                a: pl.Tensor[[ROWS, COLS], pl.BF16],
                b: pl.Tensor[[ROWS, COLS], pl.BF16],
                output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                valid_rows: pl.Scalar[pl.INDEX],
                valid_cols: pl.Scalar[pl.INDEX],
            ):
                c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_consumer")
                pl.aic_initialize_pipe(
                    dir_mask=1,
                    slot_size=SLOT_SIZE_BYTES,
                    c2v_consumer_buf=c2v_peer,
                )

                a_mat: pl.Tile[[ROWS, COLS], pl.BF16, pl.Mem.Mat] = pl.load(
                    a, [0, 0], [ROWS, COLS], target_memory=pl.MemorySpace.Mat
                )
                b_mat: pl.Tile[[ROWS, COLS], pl.BF16, pl.Mem.Mat] = pl.load(
                    b, [0, 0], [ROWS, COLS], target_memory=pl.MemorySpace.Mat
                )
                a_left: pl.Tile[[ROWS, COLS], pl.BF16, pl.Mem.Left] = pl.move(
                    a_mat, target_memory=pl.MemorySpace.Left
                )
                b_right: pl.Tile[[ROWS, COLS], pl.BF16, pl.Mem.Right] = pl.move(
                    b_mat, target_memory=pl.MemorySpace.Right
                )
                acc: pl.Tile[[ROWS, COLS], pl.FP32] = pl.matmul(a_left, b_right)
                pl.tpush_to_aiv(acc, split=1)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def vector_consumer(
                self,
                a: pl.Tensor[[ROWS, COLS], pl.BF16],
                b: pl.Tensor[[ROWS, COLS], pl.BF16],
                output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                valid_rows: pl.Scalar[pl.INDEX],
                valid_cols: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                c2v_buf = pl.reserve_buffer(
                    name="c2v_slot_buffer",
                    size=BUFFER_SIZE_BYTES,
                    base=0x2000,
                )
                pl.aiv_initialize_pipe(
                    dir_mask=1,
                    slot_size=SLOT_SIZE_BYTES,
                    c2v_consumer_buf=c2v_buf,
                )

                popped: pl.Tile[
                    [ROWS, COLS],
                    pl.FP32,
                    pl.Mem.Vec,
                    pl.TileView(valid_shape=[valid_rows, valid_cols]),
                ] = pl.tpop_from_aic(split=1)
                incremented: pl.Tile[[ROWS, COLS], pl.FP32] = pl.add(popped, 1.0)
                pl.tfree_to_aic(popped)
                return pl.store(incremented, [0, 0], output)

            @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.UP_DOWN})
            def group_func(
                self,
                a: pl.Tensor[[ROWS, COLS], pl.BF16],
                b: pl.Tensor[[ROWS, COLS], pl.BF16],
                output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                valid_rows: pl.Scalar[pl.INDEX],
                valid_cols: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                self.cube_producer(a, b, output, valid_rows, valid_cols)
                result = self.vector_consumer(a, b, output, valid_rows, valid_cols)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[ROWS, COLS], pl.BF16],
                b: pl.Tensor[[ROWS, COLS], pl.BF16],
                valid_shape: pl.Tensor[[2], pl.INDEX],
                output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                valid_rows: pl.Scalar[pl.INDEX] = pl.tensor.read(valid_shape, [0])
                valid_cols: pl.Scalar[pl.INDEX] = pl.tensor.read(valid_shape, [1])
                result = self.group_func(a, b, output, valid_rows, valid_cols)
                return result

        return C2VDynamicTpopValidShapeProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        valid_rows = int(tensors["valid_shape"][0])
        valid_cols = int(tensors["valid_shape"][1])
        matmul = torch.matmul(tensors["a"].float(), tensors["b"].float())
        tensors["output"][:valid_rows, :valid_cols] = matmul[:valid_rows, :valid_cols] + 1.0


class TestCrossCoreDynamicValidShape:
    """Cross-core dynamic valid_shape runtime tests."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_c2v_tpop_dynamic_valid_shape(self, test_runner, platform):
        """C2V tpop passes runtime valid_shape to vector-core compute and store."""
        result = test_runner.run(C2VDynamicTpopValidShapeTestCase(platform=platform))
        assert result.passed, f"C2V dynamic valid_shape tpop failed: {result.error}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cross-core 1C2V split runtime parity probes.

These probes exercise ``UP_DOWN`` and ``LEFT_RIGHT`` splits on an even static
[16, 16] tile with the runtime ``valid_shape`` swept through every parity
regime (small / below-half / at-half / above-half / near-full):

  VR/VC ∈ {1, 7, 8, 9, 15}

For each subblock, ``SplitVectorKernel``'s ``LocalizeValidDimForSplit`` computes
the per-subblock valid extent as
``max(min(valid_dim - subblock_idx * half_dim, half_dim), 0)`` (see
``src/ir/transforms/split_vector_kernel_pass.cpp:233``). The store
auto-adjusts its offset to the subblock-relative slot via ``AdjustOffsets``,
so the user-written store offset stays ``[0, 0]`` and the compiler places
subblock 0 at the origin and subblock 1 at the split-axis half.

This is the canonical pattern users follow when they need to ship an odd
extent (e.g. ``valid_rows = 17``) through ``tpush_to_aiv`` / ``tpop_from_aic``:
declare a box dim that is a multiple of the producer's ``innerDim`` and use
``pl.tile.set_validshape(...)`` to carry the truthful odd extent. The full
producer box is transported over the slot (so both halves of the consumer
receive complete data), and each subblock's store writes only the localized
valid region back to GM.
"""

import sys
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

ROWS = 16
COLS = 16
HALF = ROWS // 2
SLOT_SIZE_BYTES = ROWS * COLS * 4
BUFFER_SIZE_BYTES = SLOT_SIZE_BYTES * 4


def _build_c2v_ud_program() -> Any:
    """C2V tpush + set_validshape program, hardcoded UP_DOWN split."""

    @pl.program
    class C2VParityProgramUD:
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
            narrowed: pl.Tile[
                [ROWS, COLS],
                pl.FP32,
                pl.Mem.Acc,
                pl.TileView(valid_shape=[valid_rows, valid_cols]),
            ] = pl.tile.set_validshape(acc, valid_rows, valid_cols)
            pl.tpush_to_aiv(narrowed, split=1)

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
            # Per-subblock store offset is injected by SplitVectorKernel's
            # AdjustOffsets; user offset stays at the subblock-local origin.
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

    return C2VParityProgramUD


def _build_c2v_lr_program() -> Any:
    """C2V tpush + set_validshape program, hardcoded LEFT_RIGHT split."""

    @pl.program
    class C2VParityProgramLR:
        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.LEFT_RIGHT})
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
            narrowed: pl.Tile[
                [ROWS, COLS],
                pl.FP32,
                pl.Mem.Acc,
                pl.TileView(valid_shape=[valid_rows, valid_cols]),
            ] = pl.tile.set_validshape(acc, valid_rows, valid_cols)
            pl.tpush_to_aiv(narrowed, split=2)

        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
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
            ] = pl.tpop_from_aic(split=2)
            incremented: pl.Tile[[ROWS, COLS], pl.FP32] = pl.add(popped, 1.0)
            pl.tfree_to_aic(popped)
            return pl.store(incremented, [0, 0], output)

        @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.LEFT_RIGHT})
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

    return C2VParityProgramLR


class _C2VValidShapeParityCase(PTOTestCase):
    """C2V tpush with dynamic valid_shape, parametric over (vr, vc, split)."""

    __test__ = False

    def __init__(
        self,
        valid_rows: int,
        valid_cols: int,
        split_mode: pl.SplitMode,
        *,
        platform: str | None = None,
        config: RunConfig | None = None,
    ):
        self._valid_rows = valid_rows
        self._valid_cols = valid_cols
        self._split_mode = split_mode
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        sm = "ud" if self._split_mode == pl.SplitMode.UP_DOWN else "lr"
        return f"cross_core_split_parity_{sm}_vr{self._valid_rows}_vc{self._valid_cols}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [ROWS, COLS], DataType.BF16, init_value=1.0),
            TensorSpec("b", [ROWS, COLS], DataType.BF16, init_value=2.0),
            TensorSpec(
                "valid_shape",
                [2],
                DataType.INT64,
                init_value=torch.tensor([self._valid_rows, self._valid_cols], dtype=torch.int64),
            ),
            TensorSpec("output", [ROWS, COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        if self._split_mode == pl.SplitMode.UP_DOWN:
            return _build_c2v_ud_program()
        return _build_c2v_lr_program()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        vr = int(tensors["valid_shape"][0])
        vc = int(tensors["valid_shape"][1])
        matmul = torch.matmul(tensors["a"].float(), tensors["b"].float())
        tensors["output"][:vr, :vc] = matmul[:vr, :vc] + 1.0


# Valid-shape parity sweep on box [16, 16]: covers below-half (1, 7), at-half
# (8), above-half (9, 15) — the regimes that distinguish "subblock 1 no-op"
# from "subblock 1 contributes a partial extent".
_UP_DOWN_PARITY_CASES = [
    (1, 16, "vr1"),
    (7, 16, "vr7"),
    (8, 16, "vr8"),
    (9, 16, "vr9"),
    (15, 16, "vr15"),
]

_LEFT_RIGHT_PARITY_CASES = [
    (16, 1, "vc1"),
    (16, 7, "vc7"),
    (16, 8, "vc8"),
    (16, 9, "vc9"),
    (16, 15, "vc15"),
]


class TestSplitParityRuntime:
    """Runtime valid_shape parity probes across UP_DOWN and LEFT_RIGHT splits."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
    @pytest.mark.parametrize(
        "vr,vc",
        [c[:2] for c in _UP_DOWN_PARITY_CASES],
        ids=[c[2] for c in _UP_DOWN_PARITY_CASES],
    )
    def test_up_down_valid_shape_parity(self, test_runner, vr, vc, platform):
        case = _C2VValidShapeParityCase(vr, vc, pl.SplitMode.UP_DOWN, platform=platform)
        result = test_runner.run(case)
        assert result.passed, f"UP_DOWN (VR={vr},VC={vc}) failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
    @pytest.mark.parametrize(
        "vr,vc",
        [c[:2] for c in _LEFT_RIGHT_PARITY_CASES],
        ids=[c[2] for c in _LEFT_RIGHT_PARITY_CASES],
    )
    def test_left_right_valid_shape_parity(self, test_runner, vr, vc, platform):
        case = _C2VValidShapeParityCase(vr, vc, pl.SplitMode.LEFT_RIGHT, platform=platform)
        result = test_runner.run(case)
        assert result.passed, f"LEFT_RIGHT (VR={vr},VC={vc}) failed: {result.error}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))

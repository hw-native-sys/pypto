# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""What a Cube -> Vector boundary can carry in its COLUMN extent.

The column is not a free field. The FIFO slot is written at the producer's
PHYSICAL column pitch, while pto-isa rebuilds the read geometry from the popped
tile's own extents (``gmStrideR = validCol``, doubled for the left-right codes;
``subAIVOffset`` likewise scales by ``validCol``). A narrowed column therefore
mis-strides the read -- unless codegen transports the full box and restores the
logical extents afterwards, which is what it does for a STATIC narrowing
(``use_full_box`` + ``pto.treshape`` in ``MakeTpopCodegenPTO``).

So exactly one narrowed-column shape has a carrier, and this probe pins it from
both sides: the static narrowing is carried, and the two shapes that would need
``treshape`` to express something it cannot are rejected with their span. The
operands are ramps (``a[i,k] = i+1``, ``b[k,j] = j+1``) so that every row AND
column of the product is distinct -- uniform operands make a mis-stride
indistinguishable from a correct read, which is how this went unnoticed.

The runtime-valued and LEFT_RIGHT narrowings are the other two no-carrier shapes;
they are covered by ``test_cross_core_split_parity.py``.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

ROWS, COLS = 16, 16
SLOT = ROWS * COLS * 4
BUF = SLOT * 4


def _row_ramp() -> torch.Tensor:
    return torch.arange(1, 17, dtype=torch.bfloat16).reshape(16, 1).expand(16, 16).contiguous()


def _col_ramp() -> torch.Tensor:
    return torch.arange(1, 17, dtype=torch.bfloat16).expand(16, 16).contiguous()


def _program(vr: int, vc: int) -> Any:
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
        def cube_producer(
            self,
            a: pl.Tensor[[ROWS, COLS], pl.BF16],
            b: pl.Tensor[[ROWS, COLS], pl.BF16],
            output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ):
            peer = pl.import_peer_buffer(name="slot_buf", peer_func="vector_consumer")
            pl.aic_initialize_pipe(dir_mask=1, slot_size=SLOT, c2v_consumer_buf=peer)
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
            narrowed: pl.Tile[[ROWS, COLS], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[vr, vc])] = (
                pl.tile.set_validshape(acc, vr, vc)
            )
            pl.tpush_to_aiv(narrowed, split=1)

        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def vector_consumer(
            self,
            a: pl.Tensor[[ROWS, COLS], pl.BF16],
            b: pl.Tensor[[ROWS, COLS], pl.BF16],
            output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
            buf = pl.reserve_buffer(name="slot_buf", size=BUF, base=0x2000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=SLOT, c2v_consumer_buf=buf)
            popped: pl.Tile[[ROWS, COLS], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, vc])] = (
                pl.tpop_from_aic(split=1)
            )
            incremented: pl.Tile[[ROWS, COLS], pl.FP32] = pl.add(popped, 1.0)
            pl.tfree_to_aic(popped)
            return pl.store(incremented, [0, 0], output)

        @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.UP_DOWN})
        def group_func(
            self,
            a: pl.Tensor[[ROWS, COLS], pl.BF16],
            b: pl.Tensor[[ROWS, COLS], pl.BF16],
            output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
            self.cube_producer(a, b, output)
            return self.vector_consumer(a, b, output)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[ROWS, COLS], pl.BF16],
            b: pl.Tensor[[ROWS, COLS], pl.BF16],
            output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
            return self.group_func(a, b, output)

    return P


class _Case(PTOTestCase):
    __test__ = False

    def __init__(self, vr, vc, *, platform=None, config=None):
        self._vr, self._vc = vr, vc
        super().__init__(config, platform=platform)

    def get_name(self):
        return f"zz_static_col_vr{self._vr}_vc{self._vc}"

    def define_tensors(self):
        return [
            TensorSpec("a", [ROWS, COLS], DataType.BF16, init_value=_row_ramp),
            TensorSpec("b", [ROWS, COLS], DataType.BF16, init_value=_col_ramp),
            TensorSpec("output", [ROWS, COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self):
        return _program(self._vr, self._vc)

    def compute_expected(self, tensors, params=None):
        mm = torch.matmul(tensors["a"].float(), tensors["b"].float())
        tensors["output"][:] = float("nan")
        tensors["output"][: self._vr, : self._vc] = mm[: self._vr, : self._vc] + 1.0


@pytest.mark.platforms("a2a3")
@pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
@pytest.mark.parametrize("vr,vc", [(16, 16), (16, 12), (16, 8)], ids=["full", "vc12", "vc8"])
def test_static_narrowed_column_is_carried(test_runner, vr, vc, platform):
    """A STATIC column narrowing rides the full-box transport and is restored."""
    result = test_runner.run(_Case(vr, vc, platform=platform))
    assert result.passed, f"static (VR={vr},VC={vc}): {result.error}"


@pytest.mark.platforms("a2a3")
@pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
def test_per_lane_row_with_narrowed_column_is_rejected(test_runner, platform):
    """Both axes narrowed has no carrier: treshape rebuilds them from one type.

    The per-lane row extent rides on the TPOP valid_row operand; the narrowed
    column can only come back through ``pto.treshape``, which carries no operands
    and rewrites BOTH axes -- overwriting the row the lane needs.
    """
    result = test_runner.run(_Case(8, 12, platform=platform))

    assert not result.passed, "a per-lane row beside a narrowed column must not compile"
    assert "narrowed column extent" in result.error, result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

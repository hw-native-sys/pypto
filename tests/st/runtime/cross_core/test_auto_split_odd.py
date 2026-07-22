# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board coverage for odd automatic LEFT_RIGHT vector splitting."""

import pypto.language as pl
import pytest
import torch

ROWS = 127
COLS = 255
OUTPUT_BOX = 128


@pl.jit
def odd_auto_split_left_right(
    lhs: pl.Tensor[[ROWS, COLS], pl.FP16],
    rhs: pl.Tensor[[COLS, ROWS], pl.FP16],
    output: pl.Out[pl.Tensor[[OUTPUT_BOX, OUTPUT_BOX], pl.FP32]],
):
    """Multiply two vector-produced tensors with odd trailing axes."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT, slot_num=1)],
    ):
        lhs_doubled = pl.add(lhs, lhs)
        rhs_doubled = pl.add(rhs, rhs)

        # Both operands cross V2C after unequal LEFT_RIGHT splits. The lhs
        # lanes cover 127/128 columns and the rhs lanes cover 63/64 columns;
        # lowering pads them to compatible 128x256 and 256x128 Cube boxes.
        result = pl.matmul(lhs_doubled, rhs_doubled, out_dtype=pl.FP32)
        output = pl.assemble(output, result, [0, 0])
    return output


@pytest.mark.platforms("a2a3")
def test_odd_auto_split_left_right_on_board(test_config):
    """Both unequal AIV shards must survive the GM rendezvous exactly once."""
    odd_auto_split_left_right._cache.clear()
    torch.manual_seed(2072)
    lhs = torch.randn(ROWS, COLS, dtype=torch.float16)
    rhs = torch.randn(COLS, ROWS, dtype=torch.float16)
    output = torch.zeros(OUTPUT_BOX, OUTPUT_BOX, dtype=torch.float32)

    odd_auto_split_left_right(lhs, rhs, output, config=test_config)

    expected = torch.matmul((lhs + lhs).float(), (rhs + rhs).float())
    assert torch.allclose(output[:ROWS, :ROWS], expected, rtol=1e-3, atol=1e-3)
    assert torch.count_nonzero(output[ROWS:, :]) == 0
    assert torch.count_nonzero(output[:, ROWS:]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

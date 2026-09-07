# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile.assemble using @pl.jit kernels.

tile.assemble lowers to TINSERT (Ascend 950 only). Mode is inferred from
operand memory spaces:

  Acc->Mat (TInsertMode::NZ):
    source: Acc (L0C), FP32, fractal layout (output of tile.matmul)
    target: Mat (L1), FP32, fractal layout

  Vec->Vec (TInsertMode::ND_VEC):
    source: Vec (UB), FP32, ND/RowMajor layout
    target: Vec (UB), FP32, ND/RowMajor layout
"""

import pytest
import torch
from examples.intermediate.assemble import (
    tile_assemble_acc_mat,
    tile_assemble_double_loop,
    tile_assemble_double_loop_broadcast,
    tile_assemble_loop_col_broadcast,
    tile_assemble_row_by_row,
    tile_assemble_vec,
)
from harness import st

# Both sim bugs below produce the same wrong output, so they share one reason.
_SLICE_SIM_BUG = "Sim bug: Vec->Vec assemble with pl.slice produces wrong output (496/1024 mismatch)"


def _left_half_case(kernel, name, **kwargs):
    """``src`` (32x16) assembled into the left half of a 32x32 ``x``."""
    torch.manual_seed(0)
    x = torch.rand(32, 32, dtype=torch.float32)
    src = torch.rand(32, 16, dtype=torch.float32)
    y = torch.zeros((32, 32), dtype=torch.float32)

    def golden(_):
        expected = x.clone()
        expected[:, :16] = src
        return expected

    return st.case(kernel, x, src, y, name=name, golden=golden, **kwargs)


def _acc_mat_case():
    """Acc->Mat (NZ mode): matmul result assembled into the right half of a Mat target."""
    torch.manual_seed(0)
    x = torch.rand(32, 32, dtype=torch.float32)
    a = torch.rand(32, 16, dtype=torch.float32)
    b = torch.rand(16, 16, dtype=torch.float32)
    y = torch.zeros((32, 32), dtype=torch.float32)

    def golden(_):
        expected = x.clone()
        expected[:, 16:] = a @ b
        return expected

    return st.case(
        tile_assemble_acc_mat, x, a, b, y, name="assemble_acc_mat", golden=golden, rtol=1e-3, atol=1e-3
    )


def _loop_col_broadcast_case():
    """Vec->Vec single loop, no pl.slice: the same 32x8 src at each c*8 column offset."""
    torch.manual_seed(0)
    x = torch.rand(32, 32, dtype=torch.float32)
    src = torch.rand(32, 8, dtype=torch.float32)
    y = torch.zeros((32, 32), dtype=torch.float32)

    def golden(_):
        expected = x.clone()
        for c in range(4):
            expected[:, c * 8 : (c + 1) * 8] = src
        return expected

    return st.case(
        tile_assemble_loop_col_broadcast, x, src, y, name="assemble_loop_col_broadcast", golden=golden
    )


def _double_loop_broadcast_case():
    """Vec->Vec nested loops, no pl.slice: the same 16x16 src fills all four quadrants."""
    torch.manual_seed(0)
    x = torch.rand(32, 32, dtype=torch.float32)
    src = torch.rand(16, 16, dtype=torch.float32)
    y = torch.zeros((32, 32), dtype=torch.float32)

    def golden(_):
        expected = x.clone()
        for b in range(2):
            for c in range(2):
                expected[b * 16 : (b + 1) * 16, c * 16 : (c + 1) * 16] = src
        return expected

    return st.case(
        tile_assemble_double_loop_broadcast, x, src, y, name="assemble_double_loop_broadcast", golden=golden
    )


# tile.assemble lowers to TINSERT, which is only available on Ascend 950.
@pytest.mark.platforms("a5", "a5sim")
@st.cases(
    pytest.param(
        _acc_mat_case(),
        marks=pytest.mark.skip(reason="Codegen bug: MemRef not found in mapping for Acc->Mat assemble"),
    ),
    _left_half_case(tile_assemble_vec, "assemble_vec"),
    pytest.param(
        _left_half_case(tile_assemble_row_by_row, "assemble_row_by_row"),
        marks=pytest.mark.skip(reason=_SLICE_SIM_BUG),
    ),
    pytest.param(
        _left_half_case(tile_assemble_double_loop, "assemble_double_loop"),
        marks=pytest.mark.skip(reason=_SLICE_SIM_BUG),
    ),
    _loop_col_broadcast_case(),
    _double_loop_broadcast_case(),
)
def test_tile_assemble(case_run):
    """Each tile.assemble pattern matches its torch reference."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

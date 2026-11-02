# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile.assemble using @pl.jit kernels.

Floating-point ND Vec insertion uses TINSERT on A2/A3 and A5. The issue #2717
cases verify column offsets, distinct row pitches, narrowed valid regions, and
preservation of surrounding target data. Historical examples keep their separate
platform restrictions and known skips below.
"""

import pypto.language as pl
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


# Preserve the historical example matrix; the Vec regression matrix below also
# exercises A2/A3.
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


def _vec_columns_case(dtype, rows, row_offset, col_offset, valid_rows, valid_cols):
    """Distinct source rows expose pitch errors; nonzero targets expose clobbers."""

    @pl.jit
    def insert_columns(x: pl.Tensor, initial: pl.Tensor, out: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            source = pl.load(x, [0, 0], [16, 128], valid_shape=[valid_rows, valid_cols])
            target = pl.load(initial, [0, 0], [rows, 512])
            result = pl.assemble(target, source, [row_offset, col_offset])
            pl.store(result, [0, 0], out)
        return out

    x = (torch.arange(16 * 128).reshape(16, 128) % 127).to(dtype)
    initial = torch.full((rows, 512), 7, dtype=dtype)
    out = torch.empty_like(initial)

    def golden(tensors):
        expected = tensors["initial"].clone()
        expected[row_offset : row_offset + valid_rows, col_offset : col_offset + valid_cols] = tensors["x"][
            :valid_rows, :valid_cols
        ]
        return expected

    return st.case(
        insert_columns,
        x,
        initial,
        out,
        name=f"vec_columns_{dtype}_{rows}_{row_offset}_{col_offset}_{valid_rows}_{valid_cols}",
        golden=golden,
        rtol=0,
        atol=0,
    )


@pytest.mark.platforms("a2a3", "a2a3sim", "a5", "a5sim")
@st.cases(
    *[
        _vec_columns_case(dtype, *geometry)
        for dtype in (torch.float32, torch.float16, torch.bfloat16)
        for geometry in ((16, 0, 128, 16, 128), (16, 0, 0, 16, 128), (32, 16, 128, 8, 96))
    ],
    _loop_col_broadcast_case(),
    _double_loop_broadcast_case(),
)
def test_vec_assemble_columns(case_run):
    """Issue #2717: insertion preserves every element outside the valid window."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

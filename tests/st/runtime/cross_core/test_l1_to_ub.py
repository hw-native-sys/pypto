# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run automatic Mat-to-Vec transfers through the normal A5 mixed-kernel path."""

import sys

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto import passes

_FRAMES = 12
_ROWS = 32
_COLS = 64


def _transfer(mode: pl.SplitMode, flat: bool, add: bool):
    row_split = mode == pl.SplitMode.UP_DOWN
    col_split = mode == pl.SplitMode.LEFT_RIGHT
    rows = 1 if flat else _ROWS
    shape = (_FRAMES * rows, _COLS)

    @pl.jit(auto_scope=False)
    def transfer(
        data: pl.Tensor[shape, pl.INT16], out: pl.Out[pl.Tensor[shape, pl.INT16]]
    ) -> pl.Tensor[shape, pl.INT16]:
        with pl.at(level=pl.Level.CORE_GROUP):
            for frame in pl.range(_FRAMES):
                row = frame * rows
                # Mat loads use NZ for the 32-row Tile and ND for one row.
                l1 = pl.tile.load(data, [row, 0], [rows, _COLS], target_memory=pl.Mem.Mat)
                if row_split:
                    for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        row_shard = pl.aiv_shard(l1)
                        out = pl.tile.store(row_shard, [row + lane * (_ROWS // 2), 0], out)
                elif col_split:
                    for lane in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                        col_shard = pl.aiv_shard(l1)
                        out = pl.tile.store(col_shard, [row, lane * (_COLS // 2)], out)
                elif flat:
                    row_tile = pl.tile.move(l1, pl.Mem.Vec)
                    out = pl.tile.store(row_tile, [row, 0], out)
                else:
                    ub = pl.tile.move(
                        l1, pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box
                    )
                    if add:
                        doubled = pl.tile.add(ub, ub)
                        out = pl.tile.store(doubled, [row, 0], out)
                    else:
                        out = pl.tile.store(ub, [row, 0], out)
        return out

    return transfer


def _case(
    mode: pl.SplitMode, planner: passes.MemoryPlanner = passes.MemoryPlanner.PYPTO, *, flat=False, add=False
):
    # Unique tags across rows, column boxes, and frames catch layout errors,
    # swapped lanes and overwriting a value before its consumer has copied it.
    shape = (_FRAMES if flat else _FRAMES * _ROWS, _COLS)
    entries = shape[0] * shape[1]
    first = -(entries // 2) if add else 1
    data = torch.arange(first, first + entries, dtype=torch.int16).reshape(shape)
    suffix = "_flat" if flat else "_add" if add else ""
    return st.case(
        _transfer(mode, flat, add),
        data,
        torch.zeros_like(data),
        name=f"a5_l1_to_ub_{mode.name.lower()}_{planner.name.lower()}{suffix}",
        memory_planner=planner,
        golden=lambda tensors: {"out": tensors["data"] * 2 if add else tensors["data"]},
        rtol=0,
        atol=0,
    )


@pytest.mark.platforms("a5")
@st.cases(
    _case(pl.SplitMode.NONE),
    _case(pl.SplitMode.UP_DOWN),
    _case(pl.SplitMode.LEFT_RIGHT),
    _case(pl.SplitMode.NONE, passes.MemoryPlanner.PTOAS),
    _case(pl.SplitMode.NONE, flat=True),
    _case(pl.SplitMode.NONE, add=True),
    _case(pl.SplitMode.NONE, passes.MemoryPlanner.PTOAS, add=True),
)
def test_l1_to_ub(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--platform=a5", *sys.argv[1:]]))

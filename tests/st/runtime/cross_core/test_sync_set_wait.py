# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board odd-shape V2C split using explicit cross-core events.

Two AIV lanes partition a logical ``[5, K]`` tensor unevenly: lane 0 computes
rows ``[0:2]`` and lane 1 computes rows ``[2:5]``. Both write into one GM
transfer tensor and signal the same mode-2 event with ``sync_set``. One AIC
``sync_wait`` therefore completes only after both the 2-row and 3-row stores
have landed, after which Cube consumes the complete tensor. No ``tpush`` or
``tpop`` participates in the transfer or rendezvous.

The GM transfer is physically padded to 16 rows because AIC Mat/Acc tiles must
be box-aligned, while ``valid_shape`` keeps the logical payload at five rows.
The single ``pl.jit`` function is expanded into AIV and AIC kernels that share
one mixed-kernel launch argument layout.
"""

import pypto.language as pl
import pytest
import torch

ROWS = 5
LANE0_ROWS = 2
LANE1_ROWS = ROWS - LANE0_ROWS
K = 16
N = 16
CUBE_PHYSICAL_ROWS = 16
V2C_EVENT_ID = 4
FFTS_WORKSPACE_ELEMENTS = 256


@pl.jit
def sync_set_wait_odd_shape(
    a: pl.Tensor[[ROWS, K], pl.FP32],
    b: pl.Tensor[[ROWS, K], pl.FP32],
    weight: pl.Tensor[[K, N], pl.FP32],
    transfer: pl.InOut[pl.Tensor[[CUBE_PHYSICAL_ROWS, K], pl.FP32]],
    ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
    output: pl.Out[pl.Tensor[[CUBE_PHYSICAL_ROWS, N], pl.FP32]],
):
    """Uneven 2/3-row AIV split followed by one AIC consumer."""
    for _ in pl.spmd(1, name_hint="sync_set_wait"):
        pl.system.set_ffts(ffts_workspace)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            if aiv_id == 0:
                a_lane0: pl.Tile[[LANE0_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [LANE0_ROWS, K])
                b_lane0: pl.Tile[[LANE0_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.load(b, [0, 0], [LANE0_ROWS, K])
                sum_lane0: pl.Tile[[LANE0_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.add(a_lane0, b_lane0)
                pl.store(sum_lane0, [0, 0], transfer)
            else:
                a_lane1: pl.Tile[[LANE1_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.load(
                    a, [LANE0_ROWS, 0], [LANE1_ROWS, K]
                )
                b_lane1: pl.Tile[[LANE1_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.load(
                    b, [LANE0_ROWS, 0], [LANE1_ROWS, K]
                )
                sum_lane1: pl.Tile[[LANE1_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.add(a_lane1, b_lane1)
                pl.store(sum_lane1, [LANE0_ROWS, 0], transfer)

            # Mode 2 is a V-to-C reduction: AIC unblocks only after both AIV
            # lanes have signalled, so the complete five-row GM tensor is ready.
            pl.system.sync_set(
                V2C_EVENT_ID,
                pipe=pl.PipeType.MTE3,
                ffts_mode=2,
                core_type="aiv",
            )

        pl.system.sync_wait(V2C_EVENT_ID, pipe=pl.PipeType.MTE2, core_type="aic")
        transfer_mat: pl.Tile[
            [CUBE_PHYSICAL_ROWS, K],
            pl.FP32,
            pl.Mem.Mat,
            pl.TileView(valid_shape=[ROWS, K]),
        ] = pl.load(
            transfer,
            [0, 0],
            [CUBE_PHYSICAL_ROWS, K],
            valid_shapes=[ROWS, K],
            target_memory=pl.Mem.Mat,
        )
        weight_mat: pl.Tile[[K, N], pl.FP32, pl.Mem.Mat] = pl.load(
            weight,
            [0, 0],
            [K, N],
            target_memory=pl.Mem.Mat,
        )
        transfer_left = pl.move(transfer_mat, target_memory=pl.Mem.Left)
        weight_right = pl.move(weight_mat, target_memory=pl.Mem.Right)
        result: pl.Tile[
            [CUBE_PHYSICAL_ROWS, N],
            pl.FP32,
            pl.Mem.Acc,
            pl.TileView(valid_shape=[ROWS, N]),
        ] = pl.matmul(transfer_left, weight_right)
        output = pl.store(result, [0, 0], output)
    return output


class TestSyncSetWait:
    """Explicit cross-core sync event system test."""

    @pytest.mark.platforms("a2a3")
    def test_static_event_id_on_board(self, test_config):
        """Synchronize one 2/3-row two-AIV GM write with one AIC wait."""
        sync_set_wait_odd_shape._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(ROWS, K, dtype=torch.float32)
        b = torch.randn(ROWS, K, dtype=torch.float32)
        weight = torch.randn(K, N, dtype=torch.float32)
        transfer = torch.zeros(CUBE_PHYSICAL_ROWS, K, dtype=torch.float32)
        ffts_workspace = torch.zeros(FFTS_WORKSPACE_ELEMENTS, dtype=torch.int64)
        output = torch.zeros(CUBE_PHYSICAL_ROWS, N, dtype=torch.float32)

        sync_set_wait_odd_shape(a, b, weight, transfer, ffts_workspace, output, config=test_config)

        expected = torch.zeros_like(output)
        expected[:ROWS] = torch.matmul(a + b, weight)
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3), (
            f"odd-shape sync_set/sync_wait max diff = {(output - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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
from examples.kernels.assemble import (
    tile_assemble_acc_mat,
    tile_assemble_double_loop,
    tile_assemble_double_loop_broadcast,
    tile_assemble_loop_col_broadcast,
    tile_assemble_row_by_row,
    tile_assemble_vec,
)


# tile.assemble lowers to TINSERT, which is only available on Ascend 950.
@pytest.mark.platforms("a5", "a5sim")
class TestAssembleOperations:
    """Test suite for tile.assemble: one test per distinct pattern."""

    @pytest.mark.skip(reason="Codegen bug: MemRef not found in mapping for Acc->Mat assemble")
    def test_tile_assemble_acc_mat(self, test_config):
        """Acc->Mat (NZ mode): matmul result assembled into right half of Mat target."""
        tile_assemble_acc_mat._cache.clear()
        torch.manual_seed(0)
        x = torch.rand(32, 32, dtype=torch.float32)
        a = torch.rand(32, 16, dtype=torch.float32)
        b = torch.rand(16, 16, dtype=torch.float32)
        y = torch.zeros((32, 32), dtype=torch.float32)
        tile_assemble_acc_mat(x, a, b, y, config=test_config)

        expected = x.clone()
        expected[:, 16:] = a @ b
        assert torch.allclose(y, expected, rtol=1e-3, atol=1e-3), (
            f"acc_mat assemble failed: max diff = {(y - expected).abs().max().item()}"
        )

    def test_tile_assemble_vec(self, test_config):
        """Vec->Vec single-shot (ND_VEC mode): src assembled into left half of target."""
        tile_assemble_vec._cache.clear()
        torch.manual_seed(0)
        x = torch.rand(32, 32, dtype=torch.float32)
        src = torch.rand(32, 16, dtype=torch.float32)
        y = torch.zeros((32, 32), dtype=torch.float32)
        tile_assemble_vec(x, src, y, config=test_config)

        expected = x.clone()
        expected[:, :16] = src
        assert torch.allclose(y, expected, rtol=1e-5, atol=1e-5), (
            f"vec assemble failed: max diff = {(y - expected).abs().max().item()}"
        )

    @pytest.mark.skip(
        reason="Sim bug: Vec->Vec assemble with pl.slice produces wrong output (496/1024 mismatch)"
    )
    def test_tile_assemble_row_by_row(self, test_config):
        """Vec->Vec single loop + pl.slice: dynamic row gather into left half."""
        tile_assemble_row_by_row._cache.clear()
        torch.manual_seed(0)
        x = torch.rand(32, 32, dtype=torch.float32)
        src = torch.rand(32, 16, dtype=torch.float32)
        y = torch.zeros((32, 32), dtype=torch.float32)
        tile_assemble_row_by_row(x, src, y, config=test_config)

        expected = x.clone()
        expected[:, :16] = src
        assert torch.allclose(y, expected, rtol=1e-5, atol=1e-5), (
            f"row_by_row assemble failed: max diff = {(y - expected).abs().max().item()}"
        )

    @pytest.mark.skip(
        reason="Sim bug: Vec->Vec assemble with pl.slice produces wrong output (496/1024 mismatch)"
    )
    def test_tile_assemble_double_loop(self, test_config):
        """Vec->Vec nested loops + pl.slice: batch x head two-level index (b*8+i)."""
        tile_assemble_double_loop._cache.clear()
        torch.manual_seed(0)
        x = torch.rand(32, 32, dtype=torch.float32)
        src = torch.rand(32, 16, dtype=torch.float32)
        y = torch.zeros((32, 32), dtype=torch.float32)
        tile_assemble_double_loop(x, src, y, config=test_config)

        expected = x.clone()
        expected[:, :16] = src
        assert torch.allclose(y, expected, rtol=1e-5, atol=1e-5), (
            f"double_loop assemble failed: max diff = {(y - expected).abs().max().item()}"
        )

    def test_tile_assemble_loop_col_broadcast(self, test_config):
        """Vec->Vec single loop, no pl.slice: same src column-block at each c*8 offset."""
        tile_assemble_loop_col_broadcast._cache.clear()
        torch.manual_seed(0)
        x = torch.rand(32, 32, dtype=torch.float32)
        src = torch.rand(32, 8, dtype=torch.float32)
        y = torch.zeros((32, 32), dtype=torch.float32)
        tile_assemble_loop_col_broadcast(x, src, y, config=test_config)

        expected = x.clone()
        for c in range(4):
            expected[:, c * 8 : (c + 1) * 8] = src
        assert torch.allclose(y, expected, rtol=1e-5, atol=1e-5), (
            f"loop_col_broadcast assemble failed: max diff = {(y - expected).abs().max().item()}"
        )

    def test_tile_assemble_double_loop_broadcast(self, test_config):
        """Vec->Vec nested loops, no pl.slice: same src[16,16] fills all four quadrants."""
        tile_assemble_double_loop_broadcast._cache.clear()
        torch.manual_seed(0)
        x = torch.rand(32, 32, dtype=torch.float32)
        src = torch.rand(16, 16, dtype=torch.float32)
        y = torch.zeros((32, 32), dtype=torch.float32)
        tile_assemble_double_loop_broadcast(x, src, y, config=test_config)

        expected = x.clone()
        for b in range(2):
            for c in range(2):
                expected[b * 16 : (b + 1) * 16, c * 16 : (c + 1) * 16] = src
        assert torch.allclose(y, expected, rtol=1e-5, atol=1e-5), (
            f"double_loop_broadcast assemble failed: max diff = {(y - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

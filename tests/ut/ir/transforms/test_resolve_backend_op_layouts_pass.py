# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ResolveBackendOpLayouts pass."""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType


class TestResolveBackendOpLayouts:
    """Test backend-driven layout repair for constrained tile ops."""

    def test_rewrites_column_vector_add_through_row_major_reshape(self):
        """`tile.add` on `[N, 1]` vectors should be repaired through `[1, N] row_major`."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def repro(
                self,
                data: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                acc_0: pl.Tile[[16, 1], pl.FP32] = pl.tile.create(
                    [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                acc_1: pl.Tile[[16, 1], pl.FP32] = pl.tile.muls(acc_0, 0.0)
                chunk: pl.Tile[[16, 256], pl.FP32] = pl.load(data, [0, 0], [16, 256])
                tmp: pl.Tile[[16, 256], pl.FP32] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                partial: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_sum(chunk, tmp)
                updated: pl.Tile[[16, 1], pl.FP32] = pl.tile.add(acc_1, partial)
                stored: pl.Tensor[[16, 1], pl.FP32] = pl.store(updated, [0, 0], out)
                return stored

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        try:
            after = passes.resolve_backend_op_layouts()(Before)
        finally:
            backend.reset_for_testing()

        printed = ir.python_print(after)
        # tile.muls is now also constrained to row_major, so acc_0 gets reshaped too
        assert "pl.tile.reshape(acc_0, [1, 16])" in printed
        assert "pl.tile.muls(" in printed
        assert "pl.tile.reshape(acc_1, [1, 16])" in printed
        assert "pl.tile.reshape(partial, [1, 16])" in printed
        assert "pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(" in printed
        assert "updated:" in printed and "= pl.tile.reshape(" in printed

    def test_rewrites_column_vector_abs_through_row_major_reshape(self):
        """`tile.abs` (unary) on `[N, 1]` col_major vector should be repaired."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def repro(
                self,
                data: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                chunk: pl.Tile[[16, 256], pl.FP32] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tmp: pl.Tile[[16, 256], pl.FP32] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                partial: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_sum(chunk, tmp)
                result: pl.Tile[[16, 1], pl.FP32] = pl.tile.abs(partial)
                stored: pl.Tensor[[16, 1], pl.FP32] = pl.store(result, [0, 0], out)
                return stored

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        try:
            after = passes.resolve_backend_op_layouts()(Before)
        finally:
            backend.reset_for_testing()

        printed = ir.python_print(after)
        assert "pl.tile.reshape(partial, [1, 16])" in printed
        assert "pl.tile.abs(" in printed
        assert "result:" in printed and "= pl.tile.reshape(" in printed

    def test_rewrites_column_vector_muls_through_row_major_reshape(self):
        """`tile.muls` (tile x scalar) on `[N, 1]` col_major should repair only the tile input."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def repro(
                self,
                data: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                chunk: pl.Tile[[16, 256], pl.FP32] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tmp: pl.Tile[[16, 256], pl.FP32] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                partial: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_sum(chunk, tmp)
                scaled: pl.Tile[[16, 1], pl.FP32] = pl.tile.muls(partial, 2.0)
                stored: pl.Tensor[[16, 1], pl.FP32] = pl.store(scaled, [0, 0], out)
                return stored

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        try:
            after = passes.resolve_backend_op_layouts()(Before)
        finally:
            backend.reset_for_testing()

        printed = ir.python_print(after)
        assert "pl.tile.reshape(partial, [1, 16])" in printed
        assert "pl.tile.muls(" in printed
        assert "scaled:" in printed and "= pl.tile.reshape(" in printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

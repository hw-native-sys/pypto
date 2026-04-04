# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for pl.runtime_print() — runtime tile/tensor printing."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics.exceptions import InvalidOperationError


class TestRuntimePrintTile:
    """Tests for pl.runtime_print() with tile arguments."""

    def test_runtime_print_tile_creates_eval_stmt(self):
        """runtime_print(tile) should create an EvalStmt wrapping a tile.runtime_print Call."""

        @pl.function
        def func(x: pl.Tensor[[32, 32], pl.FP32]) -> pl.Tensor[[32, 32], pl.FP32]:
            tile: pl.Tile[[32, 32], pl.FP32] = pl.load(x, [0, 0], [32, 32])
            pl.runtime_print(tile)
            return pl.store(tile, [0, 0], x)

        body = func.body
        assert isinstance(body, ir.SeqStmts)
        # AssignStmt (load) + EvalStmt (runtime_print) + ReturnStmt
        assert len(body.stmts) == 3
        eval_stmt = body.stmts[1]
        assert isinstance(eval_stmt, ir.EvalStmt)
        call = eval_stmt.expr
        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.runtime_print"
        assert len(call.args) == 1

    def test_tile_namespace_runtime_print(self):
        """pl.tile.runtime_print(tile) should produce the same IR."""

        @pl.function
        def func(x: pl.Tensor[[32, 32], pl.FP32]) -> pl.Tensor[[32, 32], pl.FP32]:
            tile: pl.Tile[[32, 32], pl.FP32] = pl.load(x, [0, 0], [32, 32])
            pl.tile.runtime_print(tile)
            return pl.store(tile, [0, 0], x)

        body = func.body
        assert isinstance(body, ir.SeqStmts)
        eval_stmt = body.stmts[1]
        assert isinstance(eval_stmt, ir.EvalStmt)
        call = eval_stmt.expr
        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.runtime_print"

    def test_runtime_print_tile_does_not_affect_data_flow(self):
        """runtime_print should not change data flow — only adds an EvalStmt."""

        @pl.function
        def with_print(x: pl.Tensor[[32, 32], pl.FP32]) -> pl.Tensor[[32, 32], pl.FP32]:
            tile: pl.Tile[[32, 32], pl.FP32] = pl.load(x, [0, 0], [32, 32])
            pl.runtime_print(tile)
            return pl.store(tile, [0, 0], x)

        @pl.function
        def without_print(x: pl.Tensor[[32, 32], pl.FP32]) -> pl.Tensor[[32, 32], pl.FP32]:
            tile: pl.Tile[[32, 32], pl.FP32] = pl.load(x, [0, 0], [32, 32])
            return pl.store(tile, [0, 0], x)

        # Should NOT be structurally equal (with_print has an extra EvalStmt)
        with_body = with_print.body
        without_body = without_print.body
        assert isinstance(with_body, ir.SeqStmts)
        assert isinstance(without_body, ir.SeqStmts)
        assert len(with_body.stmts) == len(without_body.stmts) + 1

    def test_runtime_print_tile_roundtrip(self):
        """Print IR → reparse → should produce structurally equal IR."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[32, 32], pl.FP32]) -> pl.Tensor[[32, 32], pl.FP32]:
                tile: pl.Tile[[32, 32], pl.FP32] = pl.load(x, [0, 0], [32, 32])
                pl.runtime_print(tile)
                return pl.store(tile, [0, 0], x)

        printed = Before.as_python()
        assert "pl.tile.runtime_print(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_runtime_print_tile_type_preserved(self):
        """The Call expression should have TileType matching the input tile."""

        @pl.function
        def func(x: pl.Tensor[[32, 32], pl.FP16]) -> pl.Tensor[[32, 32], pl.FP16]:
            tile: pl.Tile[[32, 32], pl.FP16] = pl.load(x, [0, 0], [32, 32])
            pl.runtime_print(tile)
            return pl.store(tile, [0, 0], x)

        body = func.body
        assert isinstance(body, ir.SeqStmts)
        eval_stmt = body.stmts[1]
        assert isinstance(eval_stmt, ir.EvalStmt)
        call = eval_stmt.expr
        assert isinstance(call, ir.Call)
        assert isinstance(call.type, ir.TileType)


class TestRuntimePrintTensor:
    """Tests for pl.runtime_print() with tensor arguments."""

    def test_runtime_print_tensor_creates_eval_stmt(self):
        """runtime_print(tensor) should create an EvalStmt wrapping a tensor.runtime_print Call."""

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            pl.runtime_print(x)
            return x

        body = func.body
        assert isinstance(body, ir.SeqStmts)
        # EvalStmt (runtime_print) + ReturnStmt
        assert len(body.stmts) == 2
        eval_stmt = body.stmts[0]
        assert isinstance(eval_stmt, ir.EvalStmt)
        call = eval_stmt.expr
        assert isinstance(call, ir.Call)
        assert call.op.name == "tensor.runtime_print"
        assert len(call.args) == 1

    def test_tensor_namespace_runtime_print(self):
        """pl.tensor.runtime_print(tensor) should produce the same IR."""

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            pl.tensor.runtime_print(x)
            return x

        body = func.body
        assert isinstance(body, ir.SeqStmts)
        eval_stmt = body.stmts[0]
        assert isinstance(eval_stmt, ir.EvalStmt)
        call = eval_stmt.expr
        assert isinstance(call, ir.Call)
        assert call.op.name == "tensor.runtime_print"

    def test_runtime_print_tensor_roundtrip(self):
        """Print IR → reparse → should produce structurally equal IR."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.runtime_print(x)
                return x

        printed = Before.as_python()
        assert "pl.tensor.runtime_print(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_runtime_print_tensor_type_preserved(self):
        """The Call expression should have TensorType matching the input tensor."""

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
            pl.runtime_print(x)
            return x

        body = func.body
        assert isinstance(body, ir.SeqStmts)
        eval_stmt = body.stmts[0]
        assert isinstance(eval_stmt, ir.EvalStmt)
        call = eval_stmt.expr
        assert isinstance(call, ir.Call)
        assert isinstance(call.type, ir.TensorType)


class TestRuntimePrintErrors:
    """Tests for error cases."""

    def test_runtime_print_requires_tile_or_tensor(self):
        """runtime_print with scalar should raise an error."""
        with pytest.raises(InvalidOperationError):

            @pl.function
            def func(x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                pl.runtime_print(x)  # type: ignore
                return x


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

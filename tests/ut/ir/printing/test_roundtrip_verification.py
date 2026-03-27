# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for print-parse roundtrip verification helpers."""

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.ir import OptimizationStrategy, PassManager
from pypto.ir.op import tile
from pypto.pypto_core import passes


class TestVerifyRoundtrip:
    """Tests for ir.verify_roundtrip()."""

    def test_simple_program_passes(self):
        """A well-formed program should pass roundtrip verification."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

        ir.verify_roundtrip(Prog)

    def test_program_with_multiple_functions_passes(self):
        """Programs with multiple functions should roundtrip."""

        @pl.program
        class Prog:
            @pl.function
            def add(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                one: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                result: pl.Tensor[[64], pl.FP32] = self.add(one, x)
                return result

        ir.verify_roundtrip(Prog)

    def test_ctrl_flow_transformed_program_passes(self):
        """Lowered while-style IR should still pass roundtrip verification."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                    if i > 5:
                        break
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    x_iter = pl.yield_(y)
                return x_iter

        transformed = passes.ctrl_flow_transform()(Prog)
        ir.verify_roundtrip(transformed)

    def test_program_with_explicit_tile_move_layout_passes(self):
        """Explicit tile.move layout kwargs should survive print-parse roundtrip."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                x_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                x_vec: pl.Tile[[16, 128], pl.BF16] = pl.move(
                    x_mat,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.store(x_vec, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.BF16]) -> pl.Tensor[[16, 128], pl.BF16]:
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.create_tensor([16, 128], dtype=pl.BF16)
                y: pl.Tensor[[16, 128], pl.BF16] = self.main_incore_0(x, out_0)
                return y

        printed = ir.python_print(Prog)
        assert "blayout=pl.TileLayout.row_major" in printed
        assert "slayout=pl.TileLayout.none_box" in printed
        ir.verify_roundtrip(Prog)

    def test_sparse_bias_move_program_passes_roundtrip(self):
        """Sparse manual Bias moves should roundtrip with canonicalized TileView metadata."""

        span = ir.Span.unknown()
        dim_1 = ir.ConstInt(1, DataType.INT64, span)
        dim_128 = ir.ConstInt(128, DataType.INT64, span)

        bias = ir.Var("bias", ir.TensorType([1, 128], DataType.FP32), span)
        out = ir.Var("out", ir.TensorType([1, 128], DataType.FP32), span)
        stored = ir.Var("stored", ir.TensorType([1, 128], DataType.FP32), span)

        bias_vec = ir.Var(
            "bias_vec", ir.TileType([dim_1, dim_128], DataType.FP32, memory_space=ir.MemorySpace.Vec), span
        )
        bias_mat = ir.Var(
            "bias_mat", ir.TileType([dim_1, dim_128], DataType.FP32, memory_space=ir.MemorySpace.Mat), span
        )
        bias_bias = ir.Var(
            "bias_bias",
            ir.TileType([dim_1, dim_128], DataType.FP32, memory_space=ir.MemorySpace.Bias),
            span,
        )
        bias_back = ir.Var(
            "bias_back",
            ir.TileType([dim_1, dim_128], DataType.FP32, memory_space=ir.MemorySpace.Vec),
            span,
        )

        body = ir.SeqStmts(
            [
                ir.AssignStmt(bias_vec, tile.load(bias, offsets=[0, 0], shapes=[1, 128]), span),
                ir.AssignStmt(bias_mat, tile.move(bias_vec, target_memory=ir.MemorySpace.Mat), span),
                ir.AssignStmt(bias_bias, tile.move(bias_mat, target_memory=ir.MemorySpace.Bias), span),
                ir.AssignStmt(
                    bias_back,
                    tile.move(
                        bias_bias,
                        target_memory=ir.MemorySpace.Vec,
                        blayout=ir.TileLayout.row_major,
                        slayout=ir.TileLayout.none_box,
                    ),
                    span,
                ),
                ir.AssignStmt(stored, tile.store(bias_back, offsets=[0, 0], output_tensor=out), span),
                ir.ReturnStmt(span),
            ],
            span,
        )
        func = ir.Function("main_incore_0", [bias, out], [], body, span, ir.FunctionType.InCore)
        prog = ir.Program([func], "BiasRoundtrip", span)

        printed = ir.python_print(prog)
        assert "pl.Mem.Bias" in printed
        ir.verify_roundtrip(prog)


class TestRoundtripInstrument:
    """Tests for ir.RoundtripInstrument()."""

    def test_instrument_name(self):
        """RoundtripInstrument should expose a stable instrument name."""
        instrument = ir.RoundtripInstrument()
        assert isinstance(instrument, passes.CallbackInstrument)
        assert instrument.get_name() == "RoundtripInstrument"

    def test_instrument_runs_in_pass_context(self):
        """RoundtripInstrument should run successfully in a pass context."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        with passes.PassContext([ir.RoundtripInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            result = pm.run_passes(Prog)

        assert isinstance(result, ir.Program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

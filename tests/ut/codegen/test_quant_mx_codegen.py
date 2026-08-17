# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen tests for MX quantization operators."""

import re

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

_TQUANT_MX_DPS = ir.get_op("tile.tquant_mx_dps").name


def _optimize(program):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


def _generate_mlir(program):
    optimized = _optimize(program)
    target = next(
        function for function in optimized.functions.values() if ir.is_incore_type(function.func_type)
    )
    return codegen.PTOCodegen().generate(ir.Program([target], target.name, optimized.span))


def _allocation_range(mlir, name_fragment):
    line = next(line for line in mlir.splitlines() if name_fragment in line and "pto.alloc_tile" in line)
    match = re.search(r"addr = %c(\d+)_i64.*dtype=([^,]+), rows=(\d+), cols=(\d+)", line)
    assert match is not None, line
    address, dtype, rows, cols = match.groups()
    element_bytes = {"i8": 1, "ui8": 1, "!pto.f4E2M1x2": 1, "f16": 2, "bf16": 2, "f32": 4}[dtype]
    return int(address), int(address) + int(rows) * int(cols) * element_bytes


class TestQuantMxCodegen:
    @pytest.mark.parametrize(
        ("layout", "dtype", "src_shape", "out_shape", "axis", "quant_type"),
        [
            (pl.MX_A_ZZ, pl.FP8E4M3FN, (16, 64), (16, 64), 1, "MXFP8"),
            (pl.MX_B_NN, pl.FP8E4M3FN, (32, 64), (64, 32), 0, "MXFP8"),
            (pl.MX_A_ZZ, pl.FP4, (16, 64), (16, 64), 1, "MXFP4_E2M1"),
            (pl.MX_B_NN, pl.FP4, (64, 64), (64, 64), 0, "MXFP4_E2M1"),
        ],
    )
    def test_packed_quant_codegen(self, layout, dtype, src_shape, out_shape, axis, quant_type):
        src_rows, src_cols = src_shape
        out_rows, out_cols = out_shape

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[src_rows, src_cols], pl.BF16],
                out: pl.Tensor[[out_rows, out_cols], dtype],
            ):
                quant, _scale = pl.quant_mx(
                    pl.load(src, [0, 0], [src_rows, src_cols]), layout=layout, dtype=dtype
                )
                pl.store(quant, [0, 0], out)

        mlir = _generate_mlir(Program)
        tquant = next(line for line in mlir.splitlines() if "pto.tquant.mx" in line)
        x2zz = next(line for line in mlir.splitlines() if "pto.tmov" in line and "x2zz_tmp_static" in line)
        assert tquant.split("outs(", 1)[1].split(" : ", 1)[0].count(",") == 3
        assert f"quant_type {quant_type}" in tquant
        assert f"grpAxis = #pto<mx_group_axis axis{axis}>" in tquant
        assert f"grpAxis = #pto<mx_group_axis axis{axis}>" in x2zz
        assert ("tq_src_kn" in mlir) == (layout == pl.MX_B_NN)
        assert ("!pto.f4E2M1x2" in tquant) == (dtype == pl.FP4)

        ranges = [_allocation_range(mlir, name) for name in ("tq_max", "tq_scaling", "tq_dst", "tq_exp")]
        for index, lhs in enumerate(ranges):
            assert all(lhs[1] <= rhs[0] or rhs[1] <= lhs[0] for rhs in ranges[index + 1 :])

    def test_dps_remains_eval_after_optimization(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, src: pl.Tensor[[16, 64], pl.FP16]):
                _quant, _scale = pl.quant_mx(pl.load(src, [0, 0], [16, 64]), layout=pl.MX_A_ZZ)

        class Collector(ir.IRVisitor):
            def __init__(self):
                super().__init__()
                self.eval_count = 0
                self.assign_count = 0

            def visit_eval_stmt(self, stmt):
                self.eval_count += isinstance(stmt.expr, ir.Call) and stmt.expr.op.name == _TQUANT_MX_DPS
                super().visit_eval_stmt(stmt)

            def visit_assign_stmt(self, stmt):
                self.assign_count += isinstance(stmt.value, ir.Call) and stmt.value.op.name == _TQUANT_MX_DPS
                super().visit_assign_stmt(stmt)

        collector = Collector()
        collector.visit_program(_optimize(Program))
        assert (collector.eval_count, collector.assign_count) == (1, 0)

    def test_tuple_alias_chain_reaches_codegen(self):
        """Ordinary aliases of the composite tuple must preserve projection lowering."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
            ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
                pair = pl.quant_mx(pl.load(src, [0, 0], [16, 64]), layout=pl.MX_A_ZZ)
                alias = pair
                second_alias = alias
                quant = second_alias[0]
                return pl.store(quant, [0, 0], out)

        mlir = _generate_mlir(Program)
        assert "pto.tquant.mx" in mlir

    def test_not_registered_on_ascend910b(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, src: pl.Tensor[[16, 64], pl.FP16]):
                _quant, _scale = pl.quant_mx(pl.load(src, [0, 0], [16, 64]), layout=pl.MX_A_ZZ)

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Program)
        target = next(
            function for function in optimized.functions.values() if ir.is_incore_type(function.func_type)
        )
        with pytest.raises(ValueError, match=r"tile\.tquant_mx_dps"):
            codegen.PTOCodegen().generate(ir.Program([target], target.name, optimized.span))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

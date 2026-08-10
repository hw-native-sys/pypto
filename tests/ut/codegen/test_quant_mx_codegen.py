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


def _optimize(program: ir.Program) -> ir.Program:
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


def _generate_mlir(program: ir.Program) -> str:
    optimized = _optimize(program)
    functions = list(optimized.functions.values())
    target = next(function for function in functions if ir.is_incore_type(function.func_type))
    return codegen.PTOCodegen().generate(ir.Program([target], target.name, optimized.span))


def _allocation_range(mlir: str, name_fragment: str) -> tuple[int, int]:
    line = next(line for line in mlir.splitlines() if name_fragment in line and "pto.alloc_tile" in line)
    match = re.search(r"addr = %c(\d+)_i64.*dtype=([^,]+), rows=(\d+), cols=(\d+)", line)
    assert match is not None, line
    address, dtype, rows, cols = match.groups()
    element_bytes = {"i8": 1, "ui8": 1, "f8E4M3FN": 1, "f8E8M0": 1, "f16": 2, "bf16": 2, "f32": 4}[dtype]
    start = int(address)
    return start, start + int(rows) * int(cols) * element_bytes


class TestQuantMxCodegen:
    def test_tquant_lowers_to_four_output_pto_op(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                out_q: pl.Tensor[[16, 64], pl.INT8],
                out_s: pl.Tensor[[1, 32], pl.UINT8],
            ):
                src_tile: pl.Tile[[16, 64], pl.FP32] = pl.load(src, [0, 0], [16, 64])
                quant, scale = pl.tile._quant_mx_nd(src_tile)
                pl.store(pl.reinterpret_view(quant, pl.INT8), [0, 0], out_q)
                pl.store(pl.reinterpret_view(scale, pl.UINT8), [0, 0], out_s)

        mlir = _generate_mlir(Program)

        line = next(line for line in mlir.splitlines() if "pto.tquant.mx" in line)
        assert line.count("outs(") == 1
        assert line.split("outs(", 1)[1].split(" : ", 1)[0].count(",") == 3
        assert "quant_type MXFP8" in line
        ranges = [
            _allocation_range(mlir, "src_tile__ssa"),
            _allocation_range(mlir, "tq_max"),
            _allocation_range(mlir, "tq_scaling"),
            _allocation_range(mlir, "tq_dst"),
            _allocation_range(mlir, "tq_exp"),
        ]
        for index, lhs in enumerate(ranges):
            for rhs in ranges[index + 1 :]:
                assert lhs[1] <= rhs[0] or rhs[1] <= lhs[0]

    def test_tquant_rejects_when_required_buffers_exceed_vec_budget(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 2944], pl.FP32],
                out_q: pl.Tensor[[16, 2944], pl.INT8],
                out_s: pl.Tensor[[1, 1472], pl.UINT8],
            ):
                src_tile = pl.load(src, [0, 0], [16, 2944])
                quant, scale = pl.tile._quant_mx_nd(src_tile)
                pl.store(pl.reinterpret_view(quant, pl.INT8), [0, 0], out_q)
                pl.store(pl.reinterpret_view(scale, pl.UINT8), [0, 0], out_s)

        # TQuant needs src, dst, exp, max, and scaling live simultaneously.
        with pytest.raises(ValueError, match="Vec buffer usage"):
            _generate_mlir(Program)

    def test_fp8e4m3fn_dtype_selects_mxfp8_quant_type(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                out_q: pl.Tensor[[16, 64], pl.INT8],
                out_s: pl.Tensor[[1, 32], pl.UINT8],
            ):
                src_tile: pl.Tile[[16, 64], pl.FP16] = pl.load(src, [0, 0], [16, 64])
                quant, scale = pl.tile._quant_mx_nd(src_tile, dtype=pl.FP8E4M3FN)
                pl.store(pl.reinterpret_view(quant, pl.INT8), [0, 0], out_q)
                pl.store(pl.reinterpret_view(scale, pl.UINT8), [0, 0], out_s)

        mlir = _generate_mlir(Program)

        line = next(line for line in mlir.splitlines() if "pto.tquant.mx" in line)
        assert "quant_type MXFP8" in line

    def test_tquant_rejects_direct_inline_indexing(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                out_q: pl.Tensor[[16, 64], pl.INT8],
            ):
                src_tile = pl.load(src, [0, 0], [16, 64])
                result = pl.tile._quant_mx_nd(src_tile)[0]
                pl.store(result, [0, 0], out_q)

        with pytest.raises(ValueError, match=r"Direct indexing of pl\.quant_mx"):
            _generate_mlir(Program)

    def test_tquant_rejects_pair_carried_through_if_result(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                cond: pl.Scalar[pl.BOOL],
                out_q: pl.Tensor[[16, 64], pl.INT8],
            ):
                src_tile = pl.load(src, [0, 0], [16, 64])
                if cond:
                    pair = pl.tile._quant_mx_nd(src_tile)
                else:
                    pair = pl.tile._quant_mx_nd(src_tile)
                quant = pair[0]
                pl.store(pl.reinterpret_view(quant, pl.INT8), [0, 0], out_q)

        with pytest.raises(ValueError, match=r"pl\.quant_mx result pair through if/loop"):
            _generate_mlir(Program)

    def test_tquant_rejects_pair_carried_through_loop_result(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                n: pl.Scalar[pl.INDEX],
                out_q: pl.Tensor[[16, 64], pl.INT8],
            ):
                src_tile = pl.load(src, [0, 0], [16, 64])
                pair = pl.tile._quant_mx_nd(src_tile)
                for _i in pl.range(n):
                    pair = pl.tile._quant_mx_nd(src_tile)
                quant = pair[0]
                pl.store(pl.reinterpret_view(quant, pl.INT8), [0, 0], out_q)

        with pytest.raises(ValueError, match=r"pl\.quant_mx result pair through if/loop"):
            _generate_mlir(Program)

    def test_tquant_rejects_pair_carried_through_scoped_loop_result(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore, auto_scope=False)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                n: pl.Scalar[pl.INDEX],
                out_q: pl.Tensor[[16, 64], pl.INT8],
            ):
                src_tile = pl.load(src, [0, 0], [16, 64])
                pair = pl.tile._quant_mx_nd(src_tile)
                for _i in pl.range(n):
                    with pl.scope():
                        pair = pl.tile._quant_mx_nd(src_tile)
                quant = pair[0]
                pl.store(pl.reinterpret_view(quant, pl.INT8), [0, 0], out_q)

        with pytest.raises(ValueError, match=r"pl\.quant_mx result pair through if/loop"):
            _generate_mlir(Program)

    def test_tquant_rejects_pair_carried_through_split_aiv_loop_result(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                n: pl.Scalar[pl.INDEX],
                out_q: pl.Tensor[[16, 64], pl.INT8],
            ):
                src_tile = pl.load(src, [0, 0], [16, 64])
                pair = pl.tile._quant_mx_nd(src_tile)
                for _i in pl.range(n):
                    for _aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                        pair = pl.tile._quant_mx_nd(src_tile)
                quant = pair[0]
                pl.store(pl.reinterpret_view(quant, pl.INT8), [0, 0], out_q)

        with pytest.raises(ValueError, match=r"pl\.quant_mx result pair through if/loop"):
            _generate_mlir(Program)

    def test_tquant_mx_dps_remains_an_eval_stmt_after_optimization(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                out_q: pl.Tensor[[16, 64], pl.INT8],
            ):
                src_tile = pl.load(src, [0, 0], [16, 64])
                quant, _scale = pl.tile._quant_mx_nd(src_tile)
                pl.store(pl.reinterpret_view(quant, pl.INT8), [0, 0], out_q)

        class TQuantMxDpsStmtCollector(ir.IRVisitor):
            def __init__(self) -> None:
                super().__init__()
                self.eval_count = 0
                self.assign_count = 0

            def visit_eval_stmt(self, stmt: ir.EvalStmt) -> None:
                if isinstance(stmt.expr, ir.Call) and stmt.expr.op.name == "tile.tquant_mx_dps":
                    self.eval_count += 1
                super().visit_eval_stmt(stmt)

            def visit_assign_stmt(self, stmt: ir.AssignStmt) -> None:
                if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "tile.tquant_mx_dps":
                    self.assign_count += 1
                super().visit_assign_stmt(stmt)

        collector = TQuantMxDpsStmtCollector()
        collector.visit_program(_optimize(Program))

        assert collector.eval_count == 1
        assert collector.assign_count == 0

    def test_tquant_tuple_element_nested_in_store_is_named(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                out_q: pl.Tensor[[16, 64], pl.INT8],
            ):
                src_tile = pl.load(src, [0, 0], [16, 64])
                result = pl.tile._quant_mx_nd(src_tile)
                pl.store(pl.reinterpret_view(result[0], pl.INT8), [0, 0], out_q)

        mlir = _generate_mlir(Program)

        assert "pto.tquant.mx" in mlir
        assert "tq_exp" in mlir
        assert "tq_dst" in mlir

    def test_tquant_tuple_element_nested_in_loop_store_is_named(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                out_q: pl.Tensor[[16, 64], pl.INT8],
            ):
                src_tile = pl.load(src, [0, 0], [16, 64])
                result = pl.tile._quant_mx_nd(src_tile)
                for _i in pl.range(1):
                    pl.store(pl.reinterpret_view(result[0], pl.INT8), [0, 0], out_q)

        mlir = _generate_mlir(Program)

        assert "pto.tquant.mx" in mlir
        assert "tq_exp" in mlir
        assert "tq_dst" in mlir

    def test_tquant_repeated_tuple_index_aliases_one_output(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                out_q0: pl.Tensor[[16, 64], pl.INT8],
                out_q1: pl.Tensor[[16, 64], pl.INT8],
            ):
                src_tile = pl.load(src, [0, 0], [16, 64])
                result = pl.tile._quant_mx_nd(src_tile)
                first = result[0]
                second = result[0]
                pl.store(pl.reinterpret_view(first, pl.INT8), [0, 0], out_q0)
                pl.store(pl.reinterpret_view(second, pl.INT8), [0, 0], out_q1)

        mlir = _generate_mlir(Program)

        dst_range = _allocation_range(mlir, "tq_dst")
        assert _allocation_range(mlir, "first__ssa") == dst_range
        assert _allocation_range(mlir, "second__ssa") == dst_range

    def test_tquant_tuple_alias_preserves_output_mapping(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                out_q: pl.Tensor[[16, 64], pl.INT8],
            ):
                src_tile = pl.load(src, [0, 0], [16, 64])
                result = pl.tile._quant_mx_nd(src_tile)
                alias1 = result
                alias2 = alias1
                quant = alias2[0]
                pl.store(pl.reinterpret_view(quant, pl.INT8), [0, 0], out_q)

        mlir = _generate_mlir(Program)

        assert _allocation_range(mlir, "quant__ssa") == _allocation_range(mlir, "tq_dst")
        assert "alias1" not in mlir
        assert "alias2" not in mlir

    def test_tquant_is_not_registered_on_ascend910b(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, src: pl.Tensor[[16, 64], pl.FP16]):
                src_tile = pl.load(src, [0, 0], [16, 64])
                _quant, _scale = pl.tile._quant_mx_nd(src_tile)

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Program)
        target = next(f for f in optimized.functions.values() if ir.is_incore_type(f.func_type))
        with pytest.raises(ValueError, match=r"tile\.tquant_mx_dps"):
            codegen.PTOCodegen().generate(ir.Program([target], target.name, optimized.span))


class TestTDequantCodegen:
    def test_tdequant_emits_pto_tdequant(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.INT8],
                scale: pl.Tensor[[16, 1], pl.FP32],
                offset: pl.Tensor[[16, 1], pl.FP32],
                out: pl.Tensor[[16, 64], pl.FP32],
            ):
                src_tile = pl.load(src, [0, 0], [16, 64])
                scale_tile = pl.load(scale, [0, 0], [16, 1])
                offset_tile = pl.load(offset, [0, 0], [16, 1])
                result = pl.tdequant(src_tile, scale_tile, offset_tile)
                pl.store(result, [0, 0], out)

        mlir = _generate_mlir(Program)

        line = next(line for line in mlir.splitlines() if "pto.tdequant" in line)
        assert "dtype=i8" in line
        assert line.count("dtype=f32") == 3
        assert line.count("blayout=row_major") == 2  # src and dst
        assert line.count("blayout=col_major") == 2  # per-row scale and offset

        src_alloc = next(
            line for line in mlir.splitlines() if "%src_tile__ssa" in line and "alloc_tile" in line
        )
        dst_alloc = next(
            line for line in mlir.splitlines() if "%result__ssa" in line and "alloc_tile" in line
        )
        src_addr = src_alloc.split("addr = ", 1)[1].split()[0]
        dst_addr = dst_alloc.split("addr = ", 1)[1].split()[0]
        assert src_addr != dst_addr

    def test_tdequant_is_not_registered_on_ascend910b(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.INT8],
                scale: pl.Tensor[[16, 1], pl.FP32],
                offset: pl.Tensor[[16, 1], pl.FP32],
            ):
                src_tile = pl.load(src, [0, 0], [16, 64])
                scale_tile = pl.load(scale, [0, 0], [16, 1])
                offset_tile = pl.load(offset, [0, 0], [16, 1])
                _result = pl.tdequant(src_tile, scale_tile, offset_tile)

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Program)
        target = next(f for f in optimized.functions.values() if ir.is_incore_type(f.func_type))
        with pytest.raises(ValueError, match=r"tile\.tdequant"):
            codegen.PTOCodegen().generate(ir.Program([target], target.name, optimized.span))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

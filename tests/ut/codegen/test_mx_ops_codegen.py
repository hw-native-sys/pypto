# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen / IR smoke tests for MX DSL ops."""

import pypto.language as pl
from pypto import ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import DataType, codegen, passes


def _emit_incore_mlir(program) -> str:
    """Run Default pipeline on Ascend950 and concatenate AIC/AIV MLIR."""
    reset_for_testing()
    set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    parts: list[str] = []
    for func in optimized.functions.values():
        if func.func_type in (pl.FunctionType.Orchestration, pl.FunctionType.Group):
            continue
        single = ir.Program([func], func.name, optimized.span)
        result = codegen.PTOCodegen().generate(single, emit_tile_addr=True)
        parts.append(result if isinstance(result, str) else "".join(result.values()))
    return "\n".join(parts)


class TestMatmulMxIR:
    def test_matmul_mx_call_name(self):
        # Shapes match PTO-ISA tmatmul_mx case1: 128x64x64, scales [128,2]/[2,64].
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([128, 64], DataType.FP8E5M2), span)
        lhs_scale = ir.Var("lhs_scale", ir.TileType([128, 2], DataType.FP8E8M0), span)
        rhs = ir.Var("rhs", ir.TileType([64, 64], DataType.FP8E5M2), span)
        rhs_scale = ir.Var("rhs_scale", ir.TileType([2, 64], DataType.FP8E8M0), span)
        call = ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)
        assert call.op.name == "tile.matmul_mx"


class TestMxDslProgram:
    def test_mx_chain_parses(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 64], pl.FP8E5M2],
                a_s: pl.Tensor[[128, 2], pl.FP8E8M0],
                b: pl.Tensor[[64, 64], pl.FP8E5M2],
                b_s: pl.Tensor[[2, 64], pl.FP8E8M0],
                out: pl.Tensor[[128, 64], pl.FP32],
            ):
                ta = pl.load(a, [0, 0], [128, 64], target_memory=pl.Mem.Mat)
                tas = pl.load(a_s, [0, 0], [128, 2], target_memory=pl.Mem.Mat, mx_layout="mx_a_zz")
                tb = pl.load(b, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
                tbs = pl.load(b_s, [0, 0], [2, 64], target_memory=pl.Mem.Mat, mx_layout="mx_b_nn")
                la = pl.move(ta, target_memory=pl.Mem.Left)
                las = pl.move(tas, target_memory=pl.Mem.LeftScale)
                rb = pl.move(tb, target_memory=pl.Mem.Right)
                rbs = pl.move(tbs, target_memory=pl.Mem.RightScale)
                las = pl.tget_scale_addr(las, la)
                rbs = pl.tget_scale_addr(rbs, rb)
                c = pl.matmul_mx(la, las, rb, rbs)
                pl.store(c, [0, 0], out)

        text = str(Program)
        assert "tile.matmul_mx" in text
        assert "tile.tget_scale_addr" in text
        assert "LeftScale" in text or "leftscale" in text.lower()

    def test_matmul_mx_emits_pto_tmatmul_mx(self):
        # Canonical sample = PTO-ISA a5 tmatmul_mx case1 (128x64x64, e5m2+e8m0).
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 64], pl.FP8E5M2],
                a_s: pl.Tensor[[128, 2], pl.FP8E8M0],
                b: pl.Tensor[[64, 64], pl.FP8E5M2],
                b_s: pl.Tensor[[2, 64], pl.FP8E8M0],
                out: pl.Tensor[[128, 64], pl.FP32],
            ):
                ta = pl.load(a, [0, 0], [128, 64], target_memory=pl.Mem.Mat)
                tas = pl.load(a_s, [0, 0], [128, 2], target_memory=pl.Mem.Mat, mx_layout="mx_a_zz")
                tb = pl.load(b, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
                tbs = pl.load(b_s, [0, 0], [2, 64], target_memory=pl.Mem.Mat, mx_layout="mx_b_nn")
                la = pl.move(ta, target_memory=pl.Mem.Left)
                las = pl.move(tas, target_memory=pl.Mem.LeftScale)
                rb = pl.move(tb, target_memory=pl.Mem.Right)
                rbs = pl.move(tbs, target_memory=pl.Mem.RightScale)
                las = pl.tget_scale_addr(las, la)
                rbs = pl.tget_scale_addr(rbs, rb)
                c = pl.matmul_mx(la, las, rb, rbs)
                pl.store(c, [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        assert "pto.tmatmul.mx" in mlir
        assert "pto.tget_scale_addr" in mlir
        # PTOAS v0.48: LeftScale/RightScale → loc=scaling; FP8E8M0 → !pto.f8E8M0
        # (EmitC ScaleLeft/Right); tload carries mx_* GM layout.
        assert "loc=scaling" in mlir
        assert "f8E5M2" in mlir
        assert "!pto.f8E8M0" in mlir
        assert "make_tensor_view" in mlir and "#pto.layout<mx_a_zz>" in mlir
        assert "#pto.layout<mx_b_nn>" in mlir
        assert "pto.tload" in mlir


class TestTQuantProgram:
    def test_tquant_program_builds(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                out_q: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                out_s: pl.Tensor[[16, 2], pl.FP8E8M0],
            ):
                t = pl.load(src, [0, 0], [16, 64])
                q, s = pl.tquant(t, mode="mxfp8_e4m3")
                pl.store(q, [0, 0], out_q)
                pl.store(s, [0, 0], out_s)

        assert "tile.tquant" in str(Program)

    def test_mx_quant_chain_emits_pto_ops(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP32],
                b_t: pl.Tensor[[32, 64], pl.FP32],
                out: pl.Tensor[[16, 32], pl.FP32],
            ):
                ta = pl.load(a, [0, 0], [16, 64])
                tb_t = pl.load(b_t, [0, 0], [32, 64])
                qa, sa = pl.mx_quant(ta, mode="mxfp8_e4m3")
                qb_t, sb_t = pl.mx_quant(tb_t, mode="mxfp8_e4m3")
                qb = pl.tile.transpose_view(qb_t)
                sb = pl.tile.transpose_view(sb_t)
                qa_m = pl.move(qa, target_memory=pl.Mem.Mat)
                sa_m = pl.move(sa, target_memory=pl.Mem.Mat)
                qb_m = pl.move(qb, target_memory=pl.Mem.Mat)
                sb_m = pl.move(sb, target_memory=pl.Mem.Mat)
                la = pl.move(qa_m, target_memory=pl.Mem.Left)
                las = pl.move(sa_m, target_memory=pl.Mem.LeftScale)
                rb = pl.move(qb_m, target_memory=pl.Mem.Right)
                rbs = pl.move(sb_m, target_memory=pl.Mem.RightScale)
                las = pl.tget_scale_addr(las, la)
                rbs = pl.tget_scale_addr(rbs, rb)
                c = pl.matmul_mx(la, las, rb, rbs)
                pl.store(c, [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        assert "pto.tquant.mx" in mlir
        assert "pto.tmatmul.mx" in mlir
        assert "pto.tget_scale_addr" in mlir

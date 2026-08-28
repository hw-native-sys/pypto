# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen smoke tests for MXFP8 matmul and explicit FP4-to-FP8 legalization."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


@pytest.fixture(autouse=True)
def _reset_backend_after_test():
    """Do not leak an explicitly selected backend into later test modules."""
    yield
    reset_for_testing()


def _run_default_pipeline(program, backend_type=BackendType.Ascend950):
    """Run the Default pipeline for an explicitly selected backend."""
    reset_for_testing()
    set_backend_type(backend_type)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


def _emit_incore_mlir(program, backend_type=BackendType.Ascend950) -> str:
    """Run Default pipeline and concatenate AIC/AIV MLIR."""
    optimized = _run_default_pipeline(program, backend_type)
    parts: list[str] = []
    for func in optimized.functions.values():
        if func.func_type in (pl.FunctionType.Orchestration, pl.FunctionType.Group):
            continue
        single = ir.Program([func], func.name, optimized.span)
        result = codegen.PTOCodegen().generate(single, emit_tile_addr=True)
        parts.append(result if isinstance(result, str) else "".join(result.values()))
    return "\n".join(parts)


class TestMxMatmulCodegen:
    def test_fp4_set_validshape_uses_physical_packed_extent(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self):
                created = pl.tile.create([16, 64], dtype=pl.FP4, target_memory=pl.Mem.Vec)
                pl.set_validshape(created, 8, 64)

        mlir = _emit_incore_mlir(Program)
        alloc = next(line for line in mlir.splitlines() if "pto.alloc_tile" in line and "f4E2M1x2" in line)
        set_valid = next(line for line in mlir.splitlines() if "pto.set_validshape" in line)
        assert "valid_row = %c16_index valid_col = %c32_index" in alloc
        assert "v_row=?, v_col=?" in alloc
        assert ", %c8_index, %c32_index :" in set_valid
        assert "%c64_index" not in set_valid

    def test_rejects_ascend910b_before_codegen(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Tensor[[16, 32], pl.FP32],
            ):
                lhs = pl.move(pl.load(a, [0, 0], [16, 64]), target_memory=pl.Mem.Left)
                lhs_scale = pl.move(
                    pl.load(a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.LeftScale,
                )
                rhs = pl.move(pl.load(b, [0, 0], [64, 32]), target_memory=pl.Mem.Right)
                rhs_scale = pl.move(
                    pl.load(b_s, [0, 0], [2, 32], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.RightScale,
                )
                pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)

        with pytest.raises(ValueError, match=r"matmul_mx.*only supported.*Ascend950.*a5.*a2a3"):
            _run_default_pipeline(Program, BackendType.Ascend910B)

    def test_emits_tmatmul_mx_and_tget(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[128, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 64], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 64], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Tensor[[128, 64], pl.FP32],
            ):
                ta = pl.load(a, [0, 0], [128, 64], target_memory=pl.Mem.Mat)
                tas = pl.load(a_s, [0, 0], [128, 2], target_memory=pl.Mem.Mat)
                tb = pl.load(b, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
                tbs = pl.load(b_s, [0, 0], [2, 64], target_memory=pl.Mem.Mat)
                la = pl.move(ta, target_memory=pl.Mem.Left)
                las = pl.move(tas, target_memory=pl.Mem.LeftScale)
                rb = pl.move(tb, target_memory=pl.Mem.Right)
                rbs = pl.move(tbs, target_memory=pl.Mem.RightScale)
                c = pl.matmul_mx(la, las, rb, rbs)
                pl.store(c, [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        assert "pto.tmatmul.mx" in mlir
        assert mlir.count("pto.tget_scale_addr") == 2
        assert "loc=scaling" in mlir
        assert "f8E4M3FN" in mlir or "f8E4M3" in mlir
        assert "!pto.f8E8M0" in mlir
        assert "make_tensor_view" in mlir and "#pto.layout<mx_a_zz>" in mlir
        assert "#pto.layout<mx_b_nn>" in mlir
        assert "pto.tload" in mlir
        mx_a_view = next(line for line in mlir.splitlines() if "mx5d_view" in line and "mx_a_zz" in line)
        mx_b_view = next(line for line in mlir.splitlines() if "mx5d_view" in line and "mx_b_nn" in line)
        assert "shape = [%c1_index, %c8_index, %c1_index, %c16_index, %c2_index]" in mx_a_view
        assert "strides = [%c256_index, %c32_index, %c32_index, %c2_index, %c1_index]" in mx_a_view
        assert "shape = [%c1_index, %c4_index, %c1_index, %c16_index, %c2_index]" in mx_b_view
        assert "strides = [%c128_index, %c32_index, %c32_index, %c2_index, %c1_index]" in mx_b_view
        # Mat→scale fill stays source-order; PTOAS later reorders bind-before-fill.
        lines = mlir.splitlines()
        first_tget = next(i for i, line in enumerate(lines) if "pto.tget_scale_addr" in line)
        assert any("pto.tmov" in line and "scaling" in line for i, line in enumerate(lines) if i < first_tget)

    def test_nd_backing_alias_only_emits_physical_rank5_mx_views(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[1, 32], pl.FP8E8M0],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[1, 64], pl.FP8E8M0],
                out: pl.Tensor[[16, 32], pl.FP32],
            ):
                a_s_mx = pl.tensor.view(a_s, [16, 2], layout=pl.MX_A_ZZ)
                b_s_mx = pl.tensor.view(b_s, [2, 32], layout=pl.MX_B_NN)
                lhs = pl.move(
                    pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Left
                )
                lhs_scale = pl.move(
                    pl.load(a_s_mx, [0, 0], [16, 2], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.LeftScale,
                )
                rhs = pl.move(
                    pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Right
                )
                rhs_scale = pl.move(
                    pl.load(b_s_mx, [0, 0], [2, 32], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.RightScale,
                )
                pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        mx_views = [
            line for line in mlir.splitlines() if "pto.make_tensor_view" in line and "#pto.layout<mx_" in line
        ]
        assert len(mx_views) == 2
        assert all(line.count("?") == 5 for line in mx_views)
        assert any("#pto.layout<mx_a_zz>" in line for line in mx_views)
        assert any("#pto.layout<mx_b_nn>" in line for line in mx_views)

    def test_mxfp4_times_mxfp8_uses_explicit_lhs_cast(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 64], pl.FP4],
                a_s: pl.Tensor[[128, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 64], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 64], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Tensor[[128, 64], pl.FP32],
            ):
                ta = pl.cast(pl.load(a, [0, 0], [128, 64]), pl.FP8E4M3FN)
                tas = pl.load(a_s, [0, 0], [128, 2], target_memory=pl.Mem.Mat)
                tb = pl.load(b, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
                tbs = pl.load(b_s, [0, 0], [2, 64], target_memory=pl.Mem.Mat)
                la = pl.move(ta, target_memory=pl.Mem.Left)
                las = pl.move(tas, target_memory=pl.Mem.LeftScale)
                rb = pl.move(tb, target_memory=pl.Mem.Right)
                rbs = pl.move(tbs, target_memory=pl.Mem.RightScale)
                pl.store(pl.matmul_mx(la, las, rb, rbs), [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        assert mlir.count("pto.tcvt") == 3
        assert "dtype=!pto.f4E2M1x2, rows=128, cols=32" in mlir
        assert "pto.tmatmul.mx" in mlir
        assert mlir.count("pto.tget_scale_addr") == 2

    @pytest.mark.parametrize("dtype", [pl.FP4, pl.INT4, pl.UINT4, pl.HF4])
    def test_a2a3_rejects_all_four_bit_incore_abis(self, dtype):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], dtype],
                out: pl.Tensor[[16, 64], dtype],
            ):
                pl.store(pl.load(src, [0, 0], [16, 64]), [0, 0], out)

        with pytest.raises(ValueError, match=r"4-bit dtype.*not supported.*a2a3.*load/store.*ABI"):
            _emit_incore_mlir(Program, BackendType.Ascend910B)

    @pytest.mark.parametrize("dtype", [pl.INT4, pl.UINT4, pl.HF4])
    def test_a5_rejects_non_fp4_four_bit_incore_abis(self, dtype):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], dtype],
                out: pl.Tensor[[16, 64], dtype],
            ):
                pl.store(pl.load(src, [0, 0], [16, 64]), [0, 0], out)

        with pytest.raises(ValueError, match=r"4-bit dtype.*not supported.*a5.*only FP4"):
            _emit_incore_mlir(Program)

    def test_matmul_mx_acc_ins_equals_outs(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Tensor[[16, 32], pl.FP32],
            ):
                lhs = pl.move(
                    pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Left
                )
                lhs_scale = pl.move(
                    pl.load(a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.LeftScale,
                )
                rhs = pl.move(
                    pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Right
                )
                rhs_scale = pl.move(
                    pl.load(b_s, [0, 0], [2, 32], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.RightScale,
                )
                acc = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
                result = pl.matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale)
                pl.store(result, [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        # Each consumer needs fresh LeftScale and RightScale bindings because
        # tget_scale_addr mutates a physical buffer shared by SSA aliases.
        assert mlir.count("pto.tget_scale_addr") == 4
        acc_line = next(line for line in mlir.splitlines() if "pto.tmatmul.mx.acc" in line)
        ins_acc = acc_line.split("ins(", 1)[1].split(",", 1)[0].strip()
        outs_acc = acc_line.split("outs(", 1)[1].split(":", 1)[0].strip()
        assert ins_acc == outs_acc

    def test_mx_rank5_view_maps_aligned_logical_offsets(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[32, 128], pl.FP8E4M3FN],
                a_s: pl.Tensor[[32, 4], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[128, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[4, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Tensor[[16, 32], pl.FP32],
            ):
                lhs = pl.move(
                    pl.load(a, [16, 64], [16, 64], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Left
                )
                lhs_scale = pl.move(
                    pl.load(a_s, [16, 2], [16, 2], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.LeftScale,
                )
                rhs = pl.move(
                    pl.load(b, [64, 0], [64, 32], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Right
                )
                rhs_scale = pl.move(
                    pl.load(b_s, [2, 0], [2, 32], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.RightScale,
                )
                pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        partitions = [line for line in mlir.splitlines() if "partition_view" in line and "mx5d_view" in line]
        assert any(
            "offsets = [%c0_index, %c1_index, %c1_index, %c0_index, %c0_index]" in line for line in partitions
        )
        assert any(
            "offsets = [%c0_index, %c0_index, %c1_index, %c0_index, %c0_index]" in line for line in partitions
        )

    def test_matmul_mx_bias_emits_pto_op(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                bias: pl.Tensor[[1, 32], pl.FP32],
                out: pl.Tensor[[16, 32], pl.FP32],
            ):
                lhs = pl.move(
                    pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Left
                )
                lhs_scale = pl.move(
                    pl.load(a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.LeftScale,
                )
                rhs = pl.move(
                    pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Right
                )
                rhs_scale = pl.move(
                    pl.load(b_s, [0, 0], [2, 32], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.RightScale,
                )
                bias_mat = pl.move(
                    pl.load(bias, [0, 0], [1, 32], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.Mat,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                bias_tile = pl.move(bias_mat, target_memory=pl.Mem.Bias)
                result = pl.matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias_tile)
                pl.store(result, [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        assert "pto.tmatmul.mx.bias" in mlir
        assert mlir.count("pto.tget_scale_addr") == 2

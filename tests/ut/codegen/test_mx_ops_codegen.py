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
from pypto.backend import BackendType, pto_backend, reset_for_testing, set_backend_type
from pypto.backend._ptoas_locate import find_ptoas_binary
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
    @pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
    @pytest.mark.parametrize("hint", [pl.CachePolicy.NORMAL_FIRST_VICTIM, pl.CachePolicy.NOT_ALLOC_KEEP])
    def test_mx_scale_load_preserves_layout_with_explicit_l2_hint(self, tmp_path, planner, hint):
        if find_ptoas_binary() is None:
            pytest.skip("PTOAS is not available")

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, a_s: pl.Tensor[[128, 8], pl.FP8E8M0, pl.MX_A_ZZ]):
                _ = pl.load(a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat, cache=hint)

        reset_for_testing()
        set_backend_type(BackendType.Ascend950)
        with passes.PassContext([], memory_planner=planner):
            optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Program)
        mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
        cpp = pto_backend._compile_pto_module(mlir, "mx_hinted_load", str(tmp_path), planner)
        assert "{layout = #pto.layout<mx_a_zz>}" in mlir
        assert "cache_policy" not in mlir
        expected = "NormalFirstVictim" if hint == pl.CachePolicy.NORMAL_FIRST_VICTIM else "NotAllocKeep"
        assert f"TLOAD<pto::TLoadL2Hint::{expected}>" in cpp
        assert "__pypto_l2_hint" not in cpp

    def test_fp4_set_validshape_uses_physical_packed_extent(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self):
                created = pl.tile.create([16, 32], dtype=pl.FP4E2M1X2, target_memory=pl.Mem.Vec)
                pl.set_validshape(created, 8, 32)

        mlir = _emit_incore_mlir(Program)
        alloc = next(line for line in mlir.splitlines() if "pto.alloc_tile" in line and "f4E2M1x2" in line)
        set_valid = next(line for line in mlir.splitlines() if "pto.set_validshape" in line)
        # Carrier extents are emitted without unit conversion.
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
            _run_default_pipeline(Program, BackendType.Ascend910)

    def test_mx_scale_load_accepts_provable_dynamic_offset(self):
        """Provable MX offsets become FloorDiv in IR; codegen sees floordivsi + maxsi."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a_s: pl.Tensor[[128, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ):
                for mt in pl.parallel(128 // 16):
                    row_off = mt * 16
                    for kg in pl.pipeline(2, 8, 2, stage=1):
                        _ = pl.load(a_s, [row_off, kg], [16, 2], target_memory=pl.Mem.Mat)

        mlir = _emit_incore_mlir(Program)
        assert "cf.assert" not in mlir
        assert "arith.remui" not in mlir
        assert "arith.floordivsi" in mlir
        assert "pto.tload" in mlir
        assert "{layout = #pto.layout<mx_a_zz>}" in mlir
        partitions = [line for line in mlir.splitlines() if "partition_view" in line]
        assert any(
            "sizes = [%c1_index, %c1_index, %c1_index, %c16_index, %c2_index]" in line for line in partitions
        )

    def test_mx_scale_load_preserves_layout_with_a5_bypass(self):
        """A5 NOT_ALLOC_KEEP keeps the MX layout and selects an ISA hint without an offset."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a_s: pl.Tensor[[128, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ):
                _ = pl.load(
                    a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat, cache=pl.CachePolicy.NOT_ALLOC_KEEP
                )

        mlir = _emit_incore_mlir(Program)
        tloads = [line for line in mlir.splitlines() if "pto.tload" in line]
        assert len(tloads) == 1, f"expected one MX scale load:\n{mlir}"
        assert "layout = #pto.layout<mx_a_zz>" in tloads[0]
        assert "cache_policy" not in mlir
        assert "l2_hint = #pto.load_l2_hint<not_alloc_keep>" in tloads[0]

    def test_mx_scale_load_rejects_unprovable_dynamic_offset(self):
        """Unaligned / unprovable dynamic MX offsets fail at BlockMxScaleTensorViews."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a_s: pl.Tensor[[128, 8], pl.FP8E8M0, pl.MX_A_ZZ],
                row_off: pl.Scalar[pl.INDEX],
                col_off: pl.Scalar[pl.INDEX],
            ):
                _ = pl.load(a_s, [row_off, col_off], [16, 2], target_memory=pl.Mem.Mat)

        with pytest.raises(ValueError, match=r"MX_A_ZZ.*cannot be proven|multiple of"):
            _emit_incore_mlir(Program)

    def test_mx_scale_load_accepts_narrowed_valid_shape(self):
        """Physical shapes stay fractal-aligned; valid_shape may narrow M."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a_s: pl.Tensor[[128, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ):
                _ = pl.load(
                    a_s,
                    [0, 0],
                    [16, 2],
                    valid_shape=[8, 2],
                    target_memory=pl.Mem.Mat,
                )

        mlir = _emit_incore_mlir(Program)
        assert "pto.tload" in mlir
        partitions = [line for line in mlir.splitlines() if "partition_view" in line]
        # Physical TLoad box remains shapes=[16,2] -> one SFractal block row.
        assert any(
            "sizes = [%c1_index, %c1_index, %c1_index, %c16_index, %c2_index]" in line for line in partitions
        )

    def test_mx_scale_load_rejects_static_misaligned_offset(self):
        with pytest.raises(ValueError, match=r"MX_A_ZZ.*multiple of 16"):

            @pl.program
            class Program:
                @pl.function(type=pl.FunctionType.InCore)
                def main(self, a_s: pl.Tensor[[128, 8], pl.FP8E8M0, pl.MX_A_ZZ]):
                    _ = pl.load(a_s, [8, 0], [16, 2], target_memory=pl.Mem.Mat)

            _emit_incore_mlir(Program)

    def test_mx_scale_load_rejects_static_misaligned_group_offset(self):
        with pytest.raises(ValueError, match=r"MX_A_ZZ.*multiple of 2"):

            @pl.program
            class Program:
                @pl.function(type=pl.FunctionType.InCore)
                def main(self, a_s: pl.Tensor[[128, 8], pl.FP8E8M0, pl.MX_A_ZZ]):
                    _ = pl.load(a_s, [0, 1], [16, 2], target_memory=pl.Mem.Mat)

            _emit_incore_mlir(Program)

    def test_mx_scale_load_floordiv_ks_is_provable(self):
        """Expert-style ``ks = k0 // 32`` must prove group-axis alignment."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a_s: pl.Tensor[[16, 64], pl.FP8E8M0, pl.MX_A_ZZ],
                b_s: pl.Tensor[[64, 32], pl.FP8E8M0, pl.MX_B_NN],
            ):
                for k0 in pl.pipeline(256, 2048, 256, stage=1):
                    ks = k0 // 32
                    _a = pl.load(a_s, [0, ks], [16, 8], target_memory=pl.Mem.Mat)
                    _b = pl.load(b_s, [ks, 0], [8, 32], target_memory=pl.Mem.Mat)

        mlir = _emit_incore_mlir(Program)
        assert "cf.assert" not in mlir
        assert "pto.tload" in mlir
        assert "{layout = #pto.layout<mx_a_zz>}" in mlir
        assert "{layout = #pto.layout<mx_b_nn>}" in mlir

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
                ta = pl.load(a, [0, 0], [128, 64])
                tas = pl.load(a_s, [0, 0], [128, 2])
                tb = pl.load(b, [0, 0], [64, 64])
                tbs = pl.load(b_s, [0, 0], [2, 64])
                c = pl.matmul_mx(ta, tas, tb, tbs)
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
        mx_views = [
            line for line in mlir.splitlines() if "pto.make_tensor_view" in line and "#pto.layout<mx_" in line
        ]
        assert len(mx_views) >= 2
        mx_a_view = next(line for line in mx_views if "mx_a_zz" in line)
        mx_b_view = next(line for line in mx_views if "mx_b_nn" in line)
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
                a: pl.Tensor[[128, 32], pl.FP4E2M1X2],
                a_s: pl.Tensor[[128, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 64], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 64], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Tensor[[128, 64], pl.FP32],
            ):
                ta = pl.cast(pl.load(a, [0, 0], [128, 32]), pl.FP8E4M3FN)
                tas = pl.load(a_s, [0, 0], [128, 2])
                tb = pl.load(b, [0, 0], [64, 64])
                tbs = pl.load(b_s, [0, 0], [2, 64])
                pl.store(pl.matmul_mx(ta, tas, tb, tbs), [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        assert mlir.count("pto.tcvt") == 3
        assert "dtype=!pto.f4E2M1x2, rows=128, cols=32" in mlir
        assert "pto.tmatmul.mx" in mlir
        assert mlir.count("pto.tget_scale_addr") == 2

    @pytest.mark.parametrize("dtype", [pl.INT4, pl.UINT4, pl.HF4])
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

        with pytest.raises(ValueError, match=r"(?:4-bit )?dtype .*not supported.*a2a3"):
            _emit_incore_mlir(Program, BackendType.Ascend910)

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

        with pytest.raises(ValueError, match=r"4-bit dtype.*not supported.*a5.*packed FP4E2M1X2"):
            _emit_incore_mlir(Program)

    @pytest.mark.parametrize(
        "compile_with_ptoas", [False, pytest.param(True, marks=pytest.mark.ptoas_compile)]
    )
    def test_matmul_mx_acc_ins_equals_outs(self, tmp_path, compile_with_ptoas):
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
        # Reuse unchanged bindings: a second bind would hoist across the first
        # MX consumer and is rejected by PTOAS v0.70 (PTOAS #1691).
        assert mlir.count("pto.tget_scale_addr") == 2
        acc_line = next(line for line in mlir.splitlines() if "pto.tmatmul.mx.acc" in line)
        ins_acc = acc_line.split("ins(", 1)[1].split(",", 1)[0].strip()
        outs_acc = acc_line.split("outs(", 1)[1].split(":", 1)[0].strip()
        assert ins_acc == outs_acc

        if compile_with_ptoas:
            # PTOAS #1691: v0.70 rejects the second binding after the first MX
            # consumer. Compile with the installed release so a toolchain bump
            # cannot pass solely on textual IR checks or an unreleased repair.
            if find_ptoas_binary() is None:
                pytest.skip("PTOAS is not available")
            cpp = pto_backend._compile_pto_module(mlir, "main", str(tmp_path))
            # Base and accumulating forms use two overloads of TMATMUL_MX.
            assert cpp.count("TMATMUL_MX(") == 2

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
        partitions = [
            line
            for line in mlir.splitlines()
            if "partition_view" in line and "offsets = [%c0_index" in line and "%c16_index" in line
        ]
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


# The four MX data pairs, with the tile_buf dtype spelling PTO emits for each.
_MX_FP8_PAIRS = [
    pytest.param(pl.FP8E4M3FN, pl.FP8E4M3FN, id="e4m3-x-e4m3"),
    pytest.param(pl.FP8E5M2, pl.FP8E5M2, id="e5m2-x-e5m2"),
    pytest.param(pl.FP8E4M3FN, pl.FP8E5M2, id="e4m3-x-e5m2"),
    pytest.param(pl.FP8E5M2, pl.FP8E4M3FN, id="e5m2-x-e4m3"),
]
_PTO_FP8_SPELLING = {pl.FP8E4M3FN: "f8E4M3FN", pl.FP8E5M2: "f8E5M2"}

# Physical K for a 1-byte MX GEMV operand: the Mat-to-Left extract of the
# single-row lhs is a vector load whose pto-isa static_assert requires
# `Cols % (512 / sizeof(DType)) == 0` (npu/a5/TExtractCommon.hpp:182-183).
_GEMV_K = 512
_GEMV_N = 96
_GEMV_G = _GEMV_K // 32


def _gemv_mx_program(
    lhs_dtype=pl.FP8E4M3FN,
    rhs_dtype=pl.FP8E4M3FN,
    *,
    n_phys=_GEMV_N,
    n_valid=_GEMV_N,
    base_phase=pl.AccPhase.Unspecified,
):
    """All three MX GEMV forms in one InCore function.

    Operand placement is left to the compiler (no hand-written ``pl.move``),
    which is the form the family is documented to accept. The A scale is a plain
    ND one-row FP8E8M0 tensor and the B scale an ``MX_B_NN`` one.

    ``base_phase`` is read as a closure constant rather than splatted from a
    dict: the kernel body is parsed from source, so a ``**kwargs`` call site is
    not something the DSL parser can resolve.
    """

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[1, _GEMV_K], lhs_dtype],
            a_s: pl.Tensor[[1, _GEMV_G], pl.FP8E8M0],
            b: pl.Tensor[[_GEMV_K, n_phys], rhs_dtype],
            b_s: pl.Tensor[[_GEMV_G, n_phys], pl.FP8E8M0, pl.MX_B_NN],
            bias: pl.Tensor[[1, n_phys], pl.FP32],
            out: pl.Tensor[[1, n_phys], pl.FP32],
            out_acc: pl.Tensor[[1, n_phys], pl.FP32],
            out_bias: pl.Tensor[[1, n_phys], pl.FP32],
        ):
            lhs = pl.load(a, [0, 0], [1, _GEMV_K], target_memory=pl.Mem.Mat)
            lhs_s = pl.load(a_s, [0, 0], [1, _GEMV_G], target_memory=pl.Mem.Mat)
            rhs = pl.load(b, [0, 0], [_GEMV_K, n_phys], valid_shape=[_GEMV_K, n_valid])
            # The B scale's valid cols track the rhs's valid N: PTOAS wants
            # b_scale valid_shape [ceil(K/32), N_valid] against its physical
            # [ceil(K/32), N_phys].
            rhs_s = pl.load(b_s, [0, 0], [_GEMV_G, n_phys], valid_shape=[_GEMV_G, n_valid])
            bias_tile = pl.load(bias, [0, 0], [1, n_phys], valid_shape=[1, n_valid], target_memory=pl.Mem.Mat)
            base = pl.tile.gemv_mx(lhs, lhs_s, rhs, rhs_s, acc_phase=base_phase)
            pl.store(base, [0, 0], out)
            pl.store(pl.tile.gemv_mx_acc(base, lhs, lhs_s, rhs, rhs_s), [0, 0], out_acc)
            pl.store(pl.tile.gemv_mx_bias(lhs, lhs_s, rhs, rhs_s, bias_tile), [0, 0], out_bias)

    return Program


class TestGemvMxCodegen:
    """`tile.gemv_mx` / `_acc` / `_bias` reach PTOAS as the `tgemv.mx` family."""

    @pytest.mark.parametrize(("lhs_dtype", "rhs_dtype"), _MX_FP8_PAIRS)
    @pytest.mark.parametrize(
        "compile_with_ptoas", [False, pytest.param(True, marks=pytest.mark.ptoas_compile)]
    )
    def test_every_fp8_pair_emits_the_tgemv_mx_family(
        self, tmp_path, compile_with_ptoas, lhs_dtype, rhs_dtype
    ):
        """Each pair reaches PTOAS as `pto.tgemv.mx*`, with per-side dtypes.

        Asserting the instruction *by name* is the point: satisfying the MX GEMV
        surface by lowering to an ordinary `pto.tgemv` would compute unscaled
        products and is not an acceptable implementation, so the name is pinned
        rather than only the numeric result.
        """
        mlir = _emit_incore_mlir(_gemv_mx_program(lhs_dtype, rhs_dtype))
        lhs_pto = _PTO_FP8_SPELLING[lhs_dtype]
        rhs_pto = _PTO_FP8_SPELLING[rhs_dtype]

        # The MX family, not the ordinary one.
        for pto_op in ("pto.tgemv.mx ", "pto.tgemv.mx.acc ", "pto.tgemv.mx.bias "):
            line = next(line for line in mlir.splitlines() if pto_op in line)
            assert f"loc=left, dtype={lhs_pto}," in line
            assert f"loc=right, dtype={rhs_pto}," in line
            assert "loc=acc, dtype=f32," in line
            assert f"loc=scaling, dtype=!pto.f8E8M0, rows=1, cols={_GEMV_G}" in line
            assert f"loc=scaling, dtype=!pto.f8E8M0, rows={_GEMV_G}, cols={_GEMV_N}" in line
        assert "pto.tmatmul" not in mlir
        # An ordinary `pto.tgemv` would mean the scales were dropped.
        assert "pto.tgemv " not in mlir and "pto.tgemv.acc" not in mlir
        assert "pto.tgemv.bias" not in mlir

        # One scale-address binding per side; each reads its own data dtype.
        tget_lines = [line for line in mlir.splitlines() if "pto.tget_scale_addr" in line]
        assert len(tget_lines) == 2
        left_tget = [line for line in tget_lines if "loc=left," in line]
        right_tget = [line for line in tget_lines if "loc=right," in line]
        assert len(left_tget) == 1 and len(right_tget) == 1
        assert f"loc=left, dtype={lhs_pto}," in left_tget[0]
        assert f"loc=right, dtype={rhs_pto}," in right_tget[0]

        if compile_with_ptoas:
            # Textual IR cannot prove PTOAS accepts these shapes: the scale
            # geometry, the single-row a_scale and the bridged accumulator are
            # all verifier contracts, so compile with the real release.
            if find_ptoas_binary() is None:
                pytest.skip("PTOAS is not available")
            cpp = pto_backend._compile_pto_module(mlir, "main", str(tmp_path))
            # Base, acc and bias are three overloads of one intrinsic.
            assert cpp.count("TGEMV_MX(") == 3

    def test_accumulator_is_bridged_to_a_static_valid_view_on_a5(self):
        """Every form re-types the Acc box so PTOAS compares *logical* extents.

        Without this, `verifyA5MxGemvTileOperands` chains into the matmul operand
        verifier, whose shape rule reads the physical extent when valid is
        dynamic -- comparing lhs rows 1 against the Acc box's 16 and rejecting
        the op at every dtype.
        """
        mlir = _emit_incore_mlir(_gemv_mx_program())
        reshapes = [line for line in mlir.splitlines() if "pto.treshape" in line]
        assert len(reshapes) == 3, f"expected one bridge per form, got {len(reshapes)}"
        for line in reshapes:
            assert "v_row=?, v_col=?" in line.split("->")[0]
            assert f"v_row=1, v_col={_GEMV_N}" in line.split("->")[1]

        # The accumulating form must pass the *same* SSA value as its ins()
        # accumulator and its outs() destination, which is what PTOAS requires
        # of an in-place accumulate.
        acc_line = next(line for line in mlir.splitlines() if "pto.tgemv.mx.acc " in line)
        ins_acc = acc_line.split("ins(", 1)[1].split(",", 1)[0].strip()
        outs_acc = acc_line.split("outs(", 1)[1].split(":", 1)[0].strip()
        assert ins_acc == outs_acc

    def test_bridge_declines_for_a_narrowed_valid_n(self):
        """A partially-valid N is left exactly as unassemblable as it was.

        Making the dst's logical N static turns on the same rule's
        `rhs logical N == dst logical N` comparison, and the rhs type's valid
        stays dynamic so that rule reads its *physical* N -- a fresh mismatch the
        bridge would have introduced. Fixing it needs every operand re-typed, not
        just the accumulator.
        """
        mlir = _emit_incore_mlir(_gemv_mx_program(n_phys=_GEMV_N, n_valid=64))
        assert "pto.tgemv.mx " in mlir
        assert "pto.treshape" not in mlir

    def test_a2a3_rejects_the_mx_gemv_family(self):
        """The three ops are A5-only, and that is enforced before codegen."""
        with pytest.raises(ValueError, match="only supported on the Ascend950|a5"):
            _emit_incore_mlir(_gemv_mx_program(), backend_type=BackendType.Ascend910)

    def test_acc_phase_rides_the_base_and_acc_forms_but_never_the_bias_form(self):
        """`pto.tgemv.mx.bias` has no accPhase attribute.

        MLIR would keep an emitted one as a *discardable* attribute and
        `TGEMV_MX` would ignore it, so a phase there is silently dropped rather
        than rejected. Emitting it is therefore the bug this pins.
        """
        mlir = _emit_incore_mlir(_gemv_mx_program(base_phase=pl.AccPhase.Partial))
        base_line = next(line for line in mlir.splitlines() if "pto.tgemv.mx " in line)
        assert "accPhase = #pto<acc_phase partial>" in base_line
        bias_line = next(line for line in mlir.splitlines() if "pto.tgemv.mx.bias " in line)
        assert "accPhase" not in bias_line

    def test_matmul_mx_acc_emission_is_unchanged(self):
        """Regression guard for the shared MX accumulate emitter.

        `make_mx_acc_codegen` gained the accumulator bridge and the attribute
        suffix for the GEMV form; neither may touch `pto.tmatmul.mx.acc`, whose
        Acc box has no physical/logical row split to bridge and which exposes no
        `acc_phase`.
        """

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
                lhs = pl.load(a, [0, 0], [16, 64])
                lhs_s = pl.load(a_s, [0, 0], [16, 2])
                rhs = pl.load(b, [0, 0], [64, 32])
                rhs_s = pl.load(b_s, [0, 0], [2, 32])
                acc = pl.matmul_mx(lhs, lhs_s, rhs, rhs_s)
                pl.store(pl.matmul_mx_acc(acc, lhs, lhs_s, rhs, rhs_s), [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        acc_line = next(line for line in mlir.splitlines() if "pto.tmatmul.mx.acc " in line)
        assert "accPhase" not in acc_line
        assert "pto.treshape" not in mlir

    def test_single_row_scale_reaches_leftscale_through_mat(self):
        """The plain-ND one-row A scale is staged Mat -> LeftScale as a scale box.

        `MX_A_ZZ` cannot carry it (its block dimension must be a multiple of 16),
        and a one-row FP8E8M0 Mat tile with the data-vector view would be refused
        by the move into LeftScale, so this pins both the layout and the staging.
        """
        mlir = _emit_incore_mlir(_gemv_mx_program())
        scale_box = (
            f"rows=1, cols={_GEMV_G}, v_row=?, v_col=?, blayout=row_major, slayout=row_major, fractal=32"
        )
        mov_lines = [
            line
            for line in mlir.splitlines()
            if "pto.tmov" in line and "loc=scaling" in line and "rows=1," in line
        ]
        assert len(mov_lines) == 1
        assert f"loc=mat, dtype=!pto.f8E8M0, {scale_box}" in mov_lines[0]
        assert f"loc=scaling, dtype=!pto.f8E8M0, {scale_box}" in mov_lines[0]

    def test_segmented_k_mx_gemv_is_rejected_rather_than_silently_wrong(self):
        """A segmented-K MX GEMV must not compile at all (PTOAS #1691).

        A multi-chunk MX reduction needs a fresh `tile.tget_scale_addr` per chunk,
        because each chunk carries its own scale tiles. `InsertMxScaleAddr` can
        reuse a binding across an MX consumer only when it is *unchanged*
        ("repeating it after a consumer would make PTOAS hoist it across that
        consumer to the original scale fill"), and segmented K makes that reuse
        impossible. The emission would be `bind, bind, consumer, bind, bind,
        consumer` -- the shape PTOAS #1691 hoists back to the first scale fill, so
        chunk 1's MAD reads chunk 0's scales for part of its output.

        On device that is nondeterministic: a two-chunk K=1024 GEMV case passed only
        3 of 6 runs, and failed identically with the phase bits removed, so it is
        not an `acc_phase` problem. The device case was removed from
        `tests/st/runtime/ops/test_gemv_mx_a5.py` rather than shipped as a ~50%
        flake into the shared `onboard-tests-a5` gate.

        This assertion was inverted on purpose. It used to PIN the hazardous
        emission -- documenting a shape that compiles and then produces a wrong
        number. A silently wrong result is worse than a compile error, so the pass
        now refuses to emit it and this test requires the refusal. When #1691 is
        fixed, drop the rejection in `insert_mx_scale_addr_pass.cpp`, restore the
        emission-order assertion kept below, and re-add the device case.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[1, 2 * _GEMV_K], pl.FP8E4M3FN],
                a_s: pl.Tensor[[1, 2 * _GEMV_G], pl.FP8E8M0],
                b: pl.Tensor[[2 * _GEMV_K, _GEMV_N], pl.FP8E5M2],
                b_s: pl.Tensor[[2 * _GEMV_G, _GEMV_N], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Tensor[[1, _GEMV_N], pl.FP32],
            ):
                a0 = pl.load(a, [0, 0], [1, _GEMV_K], target_memory=pl.Mem.Mat)
                a0_s = pl.load(a_s, [0, 0], [1, _GEMV_G], target_memory=pl.Mem.Mat)
                b0 = pl.load(b, [0, 0], [_GEMV_K, _GEMV_N], target_memory=pl.Mem.Mat)
                b0_s = pl.load(b_s, [0, 0], [_GEMV_G, _GEMV_N])
                acc = pl.tile.gemv_mx(a0, a0_s, b0, b0_s, acc_phase=pl.AccPhase.Partial)
                a1 = pl.load(a, [0, _GEMV_K], [1, _GEMV_K], target_memory=pl.Mem.Mat)
                a1_s = pl.load(a_s, [0, _GEMV_G], [1, _GEMV_G], target_memory=pl.Mem.Mat)
                b1 = pl.load(b, [_GEMV_K, 0], [_GEMV_K, _GEMV_N], target_memory=pl.Mem.Mat)
                b1_s = pl.load(b_s, [_GEMV_G, 0], [_GEMV_G, _GEMV_N])
                acc = pl.tile.gemv_mx_acc(acc, a1, a1_s, b1, b1_s, acc_phase=pl.AccPhase.Final)
                pl.store(acc, [0, 0], out, st_phase=pl.STPhase.Final)

        # The rejection fires on exactly the program whose device runs produced the
        # wrong numbers: two K chunks, distinct per-chunk scales, Partial/Final.
        with pytest.raises(ValueError, match="PTOAS #1691"):
            _emit_incore_mlir(Program)

        # The emission the rejection replaces, kept as the re-enable recipe: when
        # #1691 is fixed, this is the order that must come back.
        #   order == ["bind", "bind", "consume", "bind", "bind", "consume"]
        #   mlir.count("pto.tget_scale_addr") == 4

    def test_single_window_mx_gemv_with_reused_scales_still_compiles(self):
        """The #1691 rejection must not catch the shape users are told to write.

        The guidance for a multi-step MX GEMV is to accumulate over ONE K window,
        i.e. reuse the same scale tiles. `InsertMxScaleAddr` then reuses the
        binding instead of emitting a second one, so no re-binding lands after a
        consumer and nothing is rejected. Without this control the rejection could
        be over-broad and every `gemv_mx` + `gemv_mx_acc` pair would stop
        compiling, which the inverted test above cannot detect on its own.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[1, _GEMV_K], pl.FP8E4M3FN],
                a_s: pl.Tensor[[1, _GEMV_G], pl.FP8E8M0],
                b: pl.Tensor[[_GEMV_K, _GEMV_N], pl.FP8E5M2],
                b_s: pl.Tensor[[_GEMV_G, _GEMV_N], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Tensor[[1, _GEMV_N], pl.FP32],
            ):
                a0 = pl.load(a, [0, 0], [1, _GEMV_K], target_memory=pl.Mem.Mat)
                a0_s = pl.load(a_s, [0, 0], [1, _GEMV_G], target_memory=pl.Mem.Mat)
                b0 = pl.load(b, [0, 0], [_GEMV_K, _GEMV_N], target_memory=pl.Mem.Mat)
                b0_s = pl.load(b_s, [0, 0], [_GEMV_G, _GEMV_N])
                acc = pl.tile.gemv_mx(a0, a0_s, b0, b0_s, acc_phase=pl.AccPhase.Partial)
                acc = pl.tile.gemv_mx_acc(acc, a0, a0_s, b0, b0_s, acc_phase=pl.AccPhase.Final)
                pl.store(acc, [0, 0], out, st_phase=pl.STPhase.Final)

        mlir = _emit_incore_mlir(Program)
        # One binding per side, reused by both consumers -- never re-bound.
        assert mlir.count("pto.tget_scale_addr") == 2
        assert mlir.count("pto.tgemv.mx") == 2

    def test_segmented_k_matmul_mx_is_not_rejected(self):
        """The rejection is scoped to the MX GEMV family, deliberately.

        `tile.matmul_mx` emits the identical hazardous sequence, but it predates
        this family and pypto-lib's `deepseek_v4_1_flash` MX models are built on
        exactly this chunked `matmul_mx` + `matmul_mx_acc` shape. Rejecting it
        would break existing callers, which is a contract decision rather than a
        fix to this PR. This test pins the scope so the split reads as intentional;
        it is NOT an assertion that the matmul form is safe on device.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 4], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[128, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[4, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Tensor[[16, 32], pl.FP32],
            ):
                a0 = pl.move(
                    pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Left
                )
                a0_s = pl.move(
                    pl.load(a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.LeftScale,
                )
                b0 = pl.move(
                    pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Right
                )
                b0_s = pl.move(
                    pl.load(b_s, [0, 0], [2, 32], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.RightScale,
                )
                acc = pl.matmul_mx(a0, a0_s, b0, b0_s)
                a1 = pl.move(
                    pl.load(a, [0, 64], [16, 64], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Left
                )
                a1_s = pl.move(
                    pl.load(a_s, [0, 2], [16, 2], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.LeftScale,
                )
                b1 = pl.move(
                    pl.load(b, [64, 0], [64, 32], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Right
                )
                b1_s = pl.move(
                    pl.load(b_s, [2, 0], [2, 32], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.RightScale,
                )
                acc = pl.matmul_mx_acc(acc, a1, a1_s, b1, b1_s)
                pl.store(acc, [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        assert mlir.count("pto.tget_scale_addr") == 4

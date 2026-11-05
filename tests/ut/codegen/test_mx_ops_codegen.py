# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen smoke tests for MXFP8 matmul and explicit FP4-to-FP8 legalization."""

from pathlib import Path

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType, pto_backend, reset_for_testing, set_backend_type
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.language.parser.diagnostics import InvalidOperationError
from pypto.pypto_core import codegen, passes
from pypto.runtime import RunConfig


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


class TestMxScaleStoreCodegen:
    """Storing an MX scale tile back to its MX_A_ZZ / MX_B_NN GM tensor.

    Two separate defects used to make every one of these cases unassemblable, so
    each test both inspects the emitted descriptors and (optionally) assembles
    the module with the installed ptoas:

    1. ``pto.partition_view`` sizes were emitted as bare integers
       (``sizes = [1, 4, 2, 16, 2]``). ``sizes`` is an index-typed *operand*
       list, so ptoas failed to parse the module at all ("expected ']'").
    2. The scale tile is boxed ZZ (``row_major`` / ``row_major``, fractal 32),
       which ``pto.tstore`` rejects — it takes ND, DN or NZ Vec sources only.
       Codegen now aliases the tile's UB address as one flat ND row, which is a
       pure descriptor change because the ZZ bytes already sit in the
       destination's ``[rows/16][groups/2][16][2]`` SFractal order.

    3. The GM-side descriptor was the tensor's cached MX-layout
       ``pto.make_tensor_view``, which the Cube-side ``pto.tload`` needs but the
       pinned pto-isa ``TSTORE`` has no overload for (ND, DN and NZ only), so
       the kernel C++ failed a static assertion in ``pto/npu/a5/TStore.hpp``.
       The store now emits its own ND view of the same pointer, which is an
       honest description of a linear copy of already-boxed bytes.
    """

    @staticmethod
    def _definition_above(lines: list[str], use_index: int, ssa_name: str) -> str:
        """The line defining ``ssa_name`` nearest above ``use_index``.

        An AIC/AIV pair is concatenated into one string here and both halves
        reuse SSA names, so a forward search would answer from the wrong
        function. Definitions precede their uses inside a function, so scanning
        upward from the use is unambiguous.
        """
        prefix = ssa_name + " = "
        for index in range(use_index, -1, -1):
            if lines[index].strip().startswith(prefix):
                return lines[index]
        raise AssertionError(f"no definition of {ssa_name} above line {use_index}")

    @classmethod
    def _nd_store_view(cls, mlir: str, store_line: str) -> str:
        """Return the ``pto.make_tensor_view`` the scale store writes through."""
        lines = mlir.splitlines()
        store_index = lines.index(store_line)
        partition_name = store_line.split("outs(", 1)[1].split(" ", 1)[0]
        partition = cls._definition_above(lines, store_index, partition_name)
        view_name = partition.split("pto.partition_view ", 1)[1].split(",", 1)[0]
        return cls._definition_above(lines, lines.index(partition), view_name)

    @classmethod
    def _scale_store_lines(cls, mlir: str) -> tuple[str, str, str]:
        """Return the (ND alias, rank-5 partition, tstore) lines of the scale store."""
        lines = mlir.splitlines()
        alias = next(line for line in lines if "_mx_nd" in line and "pto.alloc_tile" in line)
        store = next(line for line in lines if "pto.tstore" in line and "_mx_nd" in line)
        partition_name = store.split("outs(", 1)[1].split(" ", 1)[0]
        partition = cls._definition_above(lines, lines.index(store), partition_name)
        return alias, partition, store

    @pytest.mark.parametrize(
        "compile_with_ptoas", [False, pytest.param(True, marks=pytest.mark.ptoas_compile)]
    )
    @pytest.mark.parametrize(
        ("group_axis", "dtype", "scale_shape", "layout"),
        [
            (1, pl.FP8E4M3FN, (64, 4), pl.MX_A_ZZ),
            (0, pl.FP8E4M3FN, (4, 64), pl.MX_B_NN),
            (1, pl.FP4E2M1X2, (64, 4), pl.MX_A_ZZ),
            (0, pl.FP4E2M1X2, (4, 64), pl.MX_B_NN),
        ],
    )
    def test_tensor_quant_mx_scale_store(
        self, tmp_path, compile_with_ptoas, group_axis, dtype, scale_shape, layout
    ):
        scale_rows, scale_cols = scale_shape

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self, src: pl.Tensor[[64, 128], pl.BF16]
            ) -> pl.Tensor[[scale_rows, scale_cols], pl.FP8E8M0, layout]:
                _data, scale = pl.quant_mx(src, group_axis=group_axis, dtype=dtype)
                return scale

        mlir = _emit_incore_mlir(Program)
        alias, partition, store = self._scale_store_lines(mlir)

        # The ND alias re-describes the ZZ bytes; it must not allocate anything
        # new, so its flat extent equals the scale-group count.
        assert "blayout=row_major, slayout=none_box" in alias
        assert f"rows=1, cols={scale_rows * scale_cols}, v_row=1, v_col={scale_rows * scale_cols}" in alias
        # The window is one contiguous run of whole 16x2 boxes, with index-typed
        # size operands: pto.tstore is row-oriented, so a box-shaped window
        # against a flat source moves only the first destination row.
        elements = scale_rows * scale_cols
        assert f"sizes = [%c1_index, %c1_index, %c1_index, %c1_index, %c{elements}_index]" in partition
        assert f"partition_tensor_view<1x1x1x1x{elements}x!pto.f8E8M0>" in partition
        assert "fractal=32" not in store
        # The destination view is the store's own ND one, not the tensor's
        # MX-layout view: TSTORE has no MX-layout GlobalTensor overload.
        view = self._nd_store_view(mlir, store)
        assert "{layout = #pto.layout<nd>}" in view
        assert f"shape = [%c1_index, %c1_index, %c1_index, %c1_index, %c{elements}_index]" in view
        assert (
            f"strides = [%c{elements}_index, %c{elements}_index, %c{elements}_index, "
            f"%c{elements}_index, %c1_index]" in view
        )

        if compile_with_ptoas:
            if find_ptoas_binary() is None:
                pytest.skip("PTOAS is not available")
            pto_backend._compile_pto_module(mlir, "main", str(tmp_path))

    @pytest.mark.ptoas_compile
    def test_tile_scale_store_at_block_offset(self, tmp_path):
        """A per-block store addresses a sub-window of the blocked scale tensor."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[64, 128], pl.BF16],
                out_scale: pl.Out[pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]],
            ) -> pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]:
                _data, scale = pl.quant_mx(pl.load(src, [32, 0], [32, 128]), group_axis=1, dtype=pl.FP8E4M3FN)
                return pl.store(scale, [32, 0], out_scale)

        mlir = _emit_incore_mlir(Program)
        _alias, partition, _store = self._scale_store_lines(mlir)
        # Row offset 32 is box-row 2 of the destination, and the flat window
        # folds the box coordinates into one element index: the box-row stride is
        # groups/2 * 16 * 2 = 64 elements, so this window starts at 2 * 64 = 128
        # and covers 32 rows * 4 groups = 128 elements.
        assert "offsets = [%c0_index, %c0_index, %c0_index, %c0_index, " in partition
        assert "sizes = [%c1_index, %c1_index, %c1_index, %c1_index, %c128_index]" in partition
        assert "partition_tensor_view<1x1x1x1x128x!pto.f8E8M0>" in partition
        # The offset is computed, not a constant, so check the arithmetic that
        # feeds it rather than a brittle SSA name.
        offset_ssa = partition.split("offsets = [", 1)[1].split("]", 1)[0].split(", ")[-1]
        lines = mlir.splitlines()
        offset_def = self._definition_above(lines, lines.index(partition), offset_ssa)
        assert "arith.muli" in offset_def or "arith.addi" in offset_def

        if find_ptoas_binary() is None:
            pytest.skip("PTOAS is not available")
        pto_backend._compile_pto_module(mlir, "main", str(tmp_path))

    def test_scale_tensor_is_loaded_as_mx_and_stored_as_nd(self):
        """One pointer, two descriptors — the case the ND store view has to not break.

        The Cube-side ``pto.tload`` needs the MX layout to gather 16x2 boxes into
        a Mat tile, while ``pto.tstore`` needs ND. Both views are emitted against
        the same ``%arg``, so a kernel that quantizes into an MX scale tensor and
        then feeds that same tensor to ``matmul_mx`` is the real test of the
        store-local view: one allocation, two descriptors, no copy.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[64, 128], pl.BF16],
                b: pl.Tensor[[128, 64], pl.FP8E4M3FN],
                b_scale: pl.Tensor[[4, 64], pl.FP8E8M0, pl.MX_B_NN],
                a_quant: pl.Out[pl.Tensor[[64, 128], pl.FP8E4M3FN]],
                a_scale: pl.Out[pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ):
                data, scale = pl.quant_mx(pl.load(src, [0, 0], [64, 128]), group_axis=1)
                pl.store(data, [0, 0], a_quant)
                pl.store(scale, [0, 0], a_scale)
                lhs = pl.load(a_quant, [0, 0], [64, 128])
                lhs_scale = pl.load(a_scale, [0, 0], [64, 4], target_memory=pl.Mem.Mat)
                rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.Mem.Mat)
                rhs_scale = pl.load(b_scale, [0, 0], [4, 64], target_memory=pl.Mem.Mat)
                pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        views = [line for line in mlir.splitlines() if "pto.make_tensor_view" in line]
        a_scale_views = [line for line in views if "a_scale" in line]
        # Both descriptors exist and address the same base pointer.
        mx_views = [line for line in a_scale_views if "#pto.layout<mx_a_zz>" in line]
        nd_views = [line for line in a_scale_views if "#pto.layout<nd>" in line]
        assert mx_views and nd_views

        def base_of(line: str) -> str:
            return line.split("pto.make_tensor_view ", 1)[1].split(",", 1)[0]

        assert {base_of(line) for line in mx_views} == {base_of(line) for line in nd_views}
        # The MX view feeds the Cube load; the ND view feeds the store.
        load = next(line for line in mlir.splitlines() if "pto.tload" in line and "mx_a_zz" in line)
        assert "loc=mat" in load
        store = next(line for line in mlir.splitlines() if "pto.tstore" in line and "_mx_nd" in line)
        assert "blayout=row_major, slayout=none_box" in store
        assert "{layout = #pto.layout<nd>}" in self._nd_store_view(mlir, store)
        assert "pto.tmatmul.mx" in mlir

    @pytest.mark.ptoas_compile
    def test_mx_scale_store_kernel_builds_end_to_end(self, tmp_path):
        """The same quantize-store-reload shape, through the whole toolchain.

        A full ``compile()`` runs ptoas *and* the kernel C++ compiler, which is
        where an MX-layout destination would trip the ``TStore.hpp`` static
        assertion — a module-level ptoas check alone cannot see that. The kernel
        also splits across AIV and AIC, so both halves emit the descriptors.
        """
        if find_ptoas_binary() is None:
            pytest.skip("PTOAS is not available")

        @pl.jit(auto_scope=False)
        def entry(
            src: pl.Tensor[[64, 128], pl.BF16],
            b: pl.Tensor[[128, 64], pl.FP8E4M3FN],
            b_scale: pl.Tensor[[4, 64], pl.FP8E8M0, pl.MX_B_NN],
            a_quant: pl.Out[pl.Tensor[[64, 128], pl.FP8E4M3FN]],
            a_scale: pl.Out[pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ):
            with pl.spmd(1):
                _block = pl.tile.get_block_idx()
                data, scale = pl.quant_mx(pl.tile.load(src, [0, 0], [64, 128]), group_axis=1)
                pl.tile.store(data, [0, 0], a_quant)
                pl.tile.store(scale, [0, 0], a_scale)
                lhs = pl.tile.load(a_quant, [0, 0], [64, 128])
                lhs_scale = pl.tile.load(a_scale, [0, 0], [64, 4], target_memory=pl.Mem.Mat)
                rhs = pl.tile.load(b, [0, 0], [128, 64], target_memory=pl.Mem.Mat)
                rhs_scale = pl.tile.load(b_scale, [0, 0], [4, 64], target_memory=pl.Mem.Mat)
                out = pl.tile.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)
            return out

        reset_for_testing()
        set_backend_type(BackendType.Ascend950)
        compiled = entry.compile(config=RunConfig(platform="a5", save_kernels_dir=str(tmp_path)))
        kernels = list(Path(compiled.output_dir).rglob("*.pto"))
        assert kernels
        mlir = "\n".join(path.read_text() for path in kernels)
        assert "{layout = #pto.layout<nd>}" in mlir
        assert "{layout = #pto.layout<mx_a_zz>}" in mlir

    @pytest.mark.ptoas_compile
    def test_four_argument_store_takes_the_mx_path(self, tmp_path):
        """An explicit shapes tuple must not route around the MX descriptors.

        `pl.store(tile, offsets, tensor, shapes)` supplies args[3], and
        BlockMxScaleTensorViews rewrites that tuple to rank 5 as well, so the
        generic N-rank partition path would otherwise run and emit the
        MX-layout destination that TSTORE has no overload for.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[64, 128], pl.BF16],
                out_scale: pl.Out[pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]],
            ) -> pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]:
                _data, scale = pl.quant_mx(pl.load(src, [0, 0], [64, 128]), group_axis=1)
                return pl.store(scale, [0, 0], out_scale, [64, 4])

        mlir = _emit_incore_mlir(Program)
        _alias, partition, store = self._scale_store_lines(mlir)
        assert "sizes = [%c1_index, %c1_index, %c1_index, %c1_index, %c256_index]" in partition
        assert "{layout = #pto.layout<nd>}" in self._nd_store_view(mlir, store)

        if find_ptoas_binary() is None:
            pytest.skip("PTOAS is not available")
        pto_backend._compile_pto_module(mlir, "main", str(tmp_path))

    def test_four_argument_store_rejects_a_narrower_window(self):
        """An explicit store shape smaller than the tile must not be ignored.

        The MX branch derives its single flat run from the tile's own valid
        shape, so a `shapes` tuple asking for fewer rows than the tile holds
        would be silently dropped and the store would overwrite the rows the
        caller excluded. Measured before the check: a `[64, 4]` scale with
        `shapes=[32, 4]` still emitted the full 256-element window.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[64, 128], pl.BF16],
                out_scale: pl.Out[pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]],
            ) -> pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]:
                _data, scale = pl.quant_mx(pl.load(src, [0, 0], [64, 128]), group_axis=1)
                return pl.store(scale, [0, 0], out_scale, [32, 4])

        with pytest.raises(ValueError, match=r"must describe exactly that tile's boxes \[1, 4, 2, 16, 2\]"):
            _emit_incore_mlir(Program)

    def test_partial_group_coverage_is_rejected(self):
        """A store must cover every group box of the rows it writes.

        The flat run is only contiguous when it does. A `[32, 2]` scale into a
        `[32, 4]` MX_A_ZZ tensor selects packed boxes 0 and 2, which a flat run
        would write as boxes 0 and 1 — silently corrupting the scales — so the
        case is rejected rather than emitted.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[32, 64], pl.BF16],
                out_scale: pl.Out[pl.Tensor[[32, 4], pl.FP8E8M0, pl.MX_A_ZZ]],
            ) -> pl.Tensor[[32, 4], pl.FP8E8M0, pl.MX_A_ZZ]:
                # K=64 gives 2 groups, against the destination's 4.
                _data, scale = pl.quant_mx(pl.load(src, [0, 0], [32, 64]), group_axis=1)
                return pl.store(scale, [0, 0], out_scale)

        with pytest.raises(ValueError, match="cover every scale group of the rows it writes"):
            _emit_incore_mlir(Program)

    @pytest.mark.parametrize(
        ("group_axis", "layout", "src_shape", "slice_shape", "slice_offsets", "nested", "alias"),
        [
            # Partial group coverage. Consecutive group boxes of one row are not
            # adjacent in packed order, so this is a strided set of 32-byte runs.
            pytest.param(1, "a", [32, 128], [32, 2], [0, 0], False, None, id="azz_partial_groups"),
            pytest.param(1, "a", [32, 128], [32, 2], [0, 0], False, "bare", id="azz_partial_alias"),
            # The B-NN dual, and the sharpest case: with the group axis
            # OUTERMOST this window is perfectly row-major contiguous — `[2, 64]`
            # of a `[4, 64]` scale spans exactly its own 128 bytes — while its
            # data is four 32-byte runs at 0, 64, 128 and 192.
            pytest.param(0, "b", [64, 128], [2, 64], [0, 0], False, "bare", id="bnn_partial_groups"),
            # Starts part-way into a box: spans every column so it reads as
            # contiguous, while its 64 bytes are the back half of packed box row
            # 0 and the front half of box row 1.
            pytest.param(1, "a", [32, 128], [16, 4], [8, 0], False, None, id="azz_box_crossing"),
            pytest.param(1, "a", [32, 128], [16, 4], [8, 0], False, "bare", id="azz_box_crossing_alias"),
            # Genuinely contiguous in packed order, yet still carrying a logical
            # address: `[4, 32]` at `[0, 32]` of `[4, 64]` B-NN records +32
            # where its packed bytes start at +128.
            pytest.param(0, "b", [64, 128], [4, 32], [0, 32], False, None, id="bnn_offset_window"),
            # The same window re-sliced identically. That recomputes the span
            # against the immediate parent, resetting it from 224 to a tight
            # 128, which is what defeated an earlier span-based test at the
            # store — and a bare alias then strips the codegen view marker too.
            pytest.param(0, "b", [64, 128], [4, 32], [0, 32], True, "bare", id="bnn_nested_reslice"),
        ],
    )
    def test_boxed_subview_source_is_rejected(
        self, group_axis, layout, src_shape, slice_shape, slice_offsets, nested, alias
    ):
        """Only the whole MX scale tile is a representable boxed view.

        A scale's bytes are stored in packed box order while a `tile.slice`
        records a *logical* row-major address. The two agree only for the whole
        tile, so every sub-window is rejected at the slice — the one place where
        the parent's geometry is still visible. Downstream it is just an offset
        into a shared buffer pool, indistinguishable from a legitimately placed
        whole tile.

        Each case defeats a weaker rule that was implemented and measured first
        (see the ids and the comment at the check): a logical-span test is blind
        to B-NN group windows, a group-coverage plus box-alignment test is blind
        to `bnn_offset_window`, and the span test that did catch that one is
        reset by `bnn_nested_reslice`. A bare alias strips the codegen-side view
        marker in every case, which is why the rule cannot live at the store.
        """
        src_rows, src_cols = src_shape
        dest_rows, dest_cols = slice_shape
        tile_shape = list(slice_shape)
        tile_offsets = list(slice_offsets)
        tensor_layout = pl.MX_A_ZZ if layout == "a" else pl.MX_B_NN

        def direct():
            @pl.program
            class Program:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    src: pl.Tensor[[src_rows, src_cols], pl.BF16],
                    out_scale: pl.Out[pl.Tensor[[dest_rows, dest_cols], pl.FP8E8M0, tensor_layout]],
                ) -> pl.Tensor[[dest_rows, dest_cols], pl.FP8E8M0, tensor_layout]:
                    _data, scale = pl.quant_mx(
                        pl.load(src, [0, 0], [src_rows, src_cols]), group_axis=group_axis
                    )
                    return pl.store(pl.tile.slice(scale, tile_shape, tile_offsets), [0, 0], out_scale)

            return Program

        def bare():
            @pl.program
            class Program:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    src: pl.Tensor[[src_rows, src_cols], pl.BF16],
                    out_scale: pl.Out[pl.Tensor[[dest_rows, dest_cols], pl.FP8E8M0, tensor_layout]],
                ) -> pl.Tensor[[dest_rows, dest_cols], pl.FP8E8M0, tensor_layout]:
                    _data, scale = pl.quant_mx(
                        pl.load(src, [0, 0], [src_rows, src_cols]), group_axis=group_axis
                    )
                    part = pl.tile.slice(scale, tile_shape, tile_offsets)
                    reused = part
                    return pl.store(reused, [0, 0], out_scale)

            return Program

        def nested_bare():
            @pl.program
            class Program:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    src: pl.Tensor[[src_rows, src_cols], pl.BF16],
                    out_scale: pl.Out[pl.Tensor[[dest_rows, dest_cols], pl.FP8E8M0, tensor_layout]],
                ) -> pl.Tensor[[dest_rows, dest_cols], pl.FP8E8M0, tensor_layout]:
                    _data, scale = pl.quant_mx(
                        pl.load(src, [0, 0], [src_rows, src_cols]), group_axis=group_axis
                    )
                    part = pl.tile.slice(scale, tile_shape, tile_offsets)
                    whole = pl.tile.slice(part, tile_shape, [0, 0])
                    reused = whole
                    return pl.store(reused, [0, 0], out_scale)

            return Program

        build = nested_bare if nested else {None: direct, "bare": bare}[alias]
        # The slice-level rule fires while the kernel body is traced, i.e. when
        # the program class is built, so construction belongs inside the
        # `raises` block as well as the lowering.
        with pytest.raises((ValueError, InvalidOperationError), match="must select the whole tile"):
            _emit_incore_mlir(build())

    def test_whole_tile_slice_is_rejected_as_out_of_scope(self):
        """A whole-tile slice is correct, and still declined — on purpose.

        It is byte-identical to the scale tile, so storing it would work. The
        store keeps the supported surface to the tile the quantizer produced,
        which is the only shape with on-board coverage, and this pins that as a
        deliberate scope choice rather than a correctness claim. Relaxing it
        needs its own device test, not just deleting the check.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[64, 128], pl.BF16],
                out_scale: pl.Out[pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]],
            ) -> pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]:
                _data, scale = pl.quant_mx(pl.load(src, [0, 0], [64, 128]), group_axis=1)
                return pl.store(pl.tile.slice(scale, [64, 4], [0, 0]), [0, 0], out_scale)

        with pytest.raises(ValueError, match="quantizer produced, not a view of one"):
            _emit_incore_mlir(Program)

    def test_scale_store_requires_static_tile_addresses(self):
        """PTOAS planning cannot supply the ND alias address the store needs."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, src: pl.Tensor[[64, 128], pl.BF16]) -> pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]:
                _data, scale = pl.quant_mx(src, group_axis=1)
                return scale

        reset_for_testing()
        set_backend_type(BackendType.Ascend950)
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
            optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Program)
        with pytest.raises(ValueError, match="requires a static PYPTO or DSA_RP memory allocation"):
            codegen.PTOCodegen().generate(optimized, emit_tile_addr=False)

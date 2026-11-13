# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen tests for MXFP8, native FP4, and explicit FP4-to-FP8 legalization."""

import re

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType, pto_backend, reset_for_testing, set_backend_type
from pypto.backend._ptoas_fp4 import uses_native_fp4_mx
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


@pytest.fixture(autouse=True)
def _reset_backend_after_test():
    """Do not leak an explicitly selected backend into later test modules."""
    yield
    reset_for_testing()


def _run_default_pipeline(
    program,
    backend_type=BackendType.Ascend950,
    *,
    planner=passes.MemoryPlanner.PYPTO,
    enable_buffer_ir=True,
):
    """Run the Default pipeline for an explicitly selected backend."""
    reset_for_testing()
    set_backend_type(backend_type)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


def _emit_incore_mlir(
    program,
    backend_type=BackendType.Ascend950,
    *,
    planner=passes.MemoryPlanner.PYPTO,
    enable_buffer_ir=True,
) -> str:
    """Run Default pipeline and concatenate AIC/AIV MLIR."""
    optimized = _run_default_pipeline(
        program, backend_type, planner=planner, enable_buffer_ir=enable_buffer_ir
    )
    parts: list[str] = []
    for func in optimized.functions.values():
        if func.func_type in (pl.FunctionType.Orchestration, pl.FunctionType.Group):
            continue
        single = ir.Program([func], func.name, optimized.span)
        result = codegen.PTOCodegen().generate(single, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
        parts.append(result if isinstance(result, str) else "".join(result.values()))
    return "\n".join(parts)


def _assert_outer_scale_fills_are_prebound(mlir: str, expected_fills: int = 4) -> None:
    """Check the exact scale handles consumed by MX, not unrelated nearby bindings."""
    lines = mlir.splitlines()
    aliases = {}
    for line in lines:
        reshaped = re.search(r"(%[\w.$-]+)\s*=\s*pto\.treshape\s+(%[\w.$-]+)", line)
        if reshaped:
            # A metadata-only view preserves the physical address used by MX
            # binding even when the producer and consumer use distinct SSAs.
            aliases[reshaped.group(1)] = reshaped.group(2)
        if "iter_args(" in line:
            # The carried regression below self-yields these handles unchanged.
            # Canonicalize its descriptor names to the initial resident tiles.
            aliases.update(re.findall(r"(%[\w.$-]+)\s*=\s*(%[\w.$-]+)", line))

    def canonical(value: str) -> str:
        seen = set()
        while value in aliases:
            assert value not in seen, "cyclic loop descriptor aliases in generated PTO"
            seen.add(value)
            value = aliases[value]
        return value

    pairs: dict[str, set[str]] = {}
    for line in lines:
        if "pto.tmatmul.mx" not in line:
            continue
        operands = line.split("ins(", 1)[1].split(":", 1)[0].split(",")
        operands = [operand.strip() for operand in operands]
        if "pto.tmatmul.mx.acc" in line:
            operands = operands[1:]
        for data, scale in ((operands[0], operands[1]), (operands[2], operands[3])):
            pairs.setdefault(canonical(scale), set()).add(canonical(data))

    first_loop = next(i for i, line in enumerate(lines) if "scf.for" in line)
    bindings: dict[str, tuple[str, int]] = {}
    fills = []
    for i, line in enumerate(lines):
        if "pto.tget_scale_addr" in line:
            data = line.split("ins(", 1)[1].split(":", 1)[0].strip()
            scale = line.split("outs(", 1)[1].split(":", 1)[0].strip()
            bindings[canonical(scale)] = (canonical(data), i)
        elif "pto.tmov" in line:
            destination = line.split("outs(", 1)[1]
            if "loc=scaling" not in destination:
                continue
            scale = canonical(destination.split(":", 1)[0].strip())
            assert i < first_loop, f"regression fixture must fill {scale} outside the loop"
            assert scale in bindings, f"{scale} is filled before its inferred address is bound:\n{line}"
            data, binding_line = bindings[scale]
            assert pairs[scale] == {data}, f"{scale} was bound to the wrong resident data tile"
            assert binding_line < i
            fills.append(scale)
    assert len(fills) == expected_fills


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

    @pytest.mark.parametrize("dtype, token", [(pl.FP4E2M1X2, "f4E2M1x2"), (pl.FP4E1M2X2, "f4E1M2x2")])
    def test_native_fp4_matmul_keeps_packed_operands(self, dtype, token, monkeypatch, tmp_path):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[32, 64], dtype],
                sa: pl.Tensor[[32, 4], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 64], dtype],
                sb: pl.Tensor[[4, 64], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Tensor[[32, 64], pl.FP32],
            ):
                a_mat = pl.load(a, [0, 0], [32, 64], target_memory=pl.MemorySpace.Mat)
                b_mat = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                lhs = pl.move(a_mat, pl.MemorySpace.Left)
                rhs = pl.move(pl.tile.transpose_view(b_mat), pl.MemorySpace.Right)
                lhs_scale = pl.move(
                    pl.load(sa, [0, 0], [32, 4], target_memory=pl.MemorySpace.Mat), pl.MemorySpace.LeftScale
                )
                rhs_scale = pl.move(
                    pl.load(sb, [0, 0], [4, 64], target_memory=pl.MemorySpace.Mat), pl.MemorySpace.RightScale
                )
                base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
                result = pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale)
                pl.store(result, [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        assert "pto.tcvt" not in mlir
        assert f"loc=left, dtype=!pto.{token}, rows=32, cols=128" in mlir
        assert f"loc=right, dtype=!pto.{token}, rows=128, cols=64" in mlir
        assert "loc=acc, dtype=f32, rows=32, cols=64" in mlir
        assert "pto.tmatmul.mx.acc" in mlir
        assert mlir.count("pto.tget_scale_addr") == 2

        optimized = _run_default_pipeline(Program)
        assert uses_native_fp4_mx(optimized)

        def unsupported():
            raise RuntimeError("missing native FP4 capability")

        monkeypatch.setattr(pto_backend, "check_native_fp4_mx_support", unsupported)

        def generate_bundle(skip_ptoas: bool):
            return pto_backend._generate_pto_bundle(
                optimized,
                skip_ptoas=skip_ptoas,
                memory_planner=passes.MemoryPlanner.PYPTO,
                emit_source_loc=False,
            )

        # IR inspection remains available without an assembler; executable
        # generation must pass the capability gate before producing a bundle.
        generate_bundle(True)
        with pytest.raises(RuntimeError, match="missing native FP4 capability"):
            generate_bundle(False)
        monkeypatch.setattr(pto_backend, "check_native_fp4_mx_support", lambda: None)
        source, _ = generate_bundle(False)
        assert "pto.tmatmul.mx.acc" in source
        if find_ptoas_binary() is not None:
            cpp = pto_backend._compile_pto_module(source, "native_fp4", str(tmp_path))
            cpp_dtype = "float4_e2m1x2_t" if dtype == pl.FP4E2M1X2 else "float4_e1m2x2_t"
            assert f"Tile<TileType::Left, {cpp_dtype}, 32, 128," in cpp
            assert f"Tile<TileType::Right, {cpp_dtype}, 128, 64," in cpp
            assert "Tile<TileType::Acc, float, 32, 64," in cpp

    @pytest.mark.parametrize(
        "memory, physical, valid",
        [
            (pl.Mem.Vec, (16, 32), (8, 32)),
            (pl.Mem.Mat, (32, 32), (16, 32)),
            (pl.Mem.Left, (32, 32), (16, 32)),
            (pl.Mem.Right, (32, 64), (8, 64)),
        ],
    )
    def test_fp4_set_validshape_uses_memory_space_units(self, memory, physical, valid):
        carrier_rows = 32 if memory == pl.Mem.Right else 16

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self):
                created = pl.tile.create([carrier_rows, 32], dtype=pl.FP4E2M1X2, target_memory=memory)
                pl.set_validshape(created, 8, 32)

        mlir = _emit_incore_mlir(Program)
        alloc = next(line for line in mlir.splitlines() if "pto.alloc_tile" in line and "f4E2M1x2" in line)
        set_valid = next(line for line in mlir.splitlines() if "pto.set_validshape" in line)
        # Vec remains packed; Cube descriptors and their mutable validity use
        # logical units. Storage planning still uses the original IR carriers.
        assert f"valid_row = %c{physical[0]}_index valid_col = %c{physical[1]}_index" in alloc
        assert "v_row=?, v_col=?" in alloc
        assert f", %c{valid[0]}_index, %c{valid[1]}_index :" in set_valid

    def test_fp4_cube_slice_converts_offset_shape_and_validity(self, tmp_path):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self):
                tile = pl.tile.create([32, 128], dtype=pl.FP4E2M1X2, target_memory=pl.Mem.Mat)
                _ = pl.tile.slice(tile, [16, 128], [16, 0], valid_shape=[8, 64])

        mlir = _emit_incore_mlir(Program)
        view = next(line for line in mlir.splitlines() if "pto.subview" in line)
        # Mat is column-major: rows, row offsets, and valid rows all expand.
        assert "[%c32_index, %c0_index] sizes [32, 128] valid [%c16_index, %c64_index]" in view
        assert "rows=32, cols=128, v_row=16, v_col=64" in view
        if find_ptoas_binary() is not None:
            pto_backend._compile_pto_module(mlir, "fp4_cube_slice", str(tmp_path))

    @pytest.mark.parametrize("dtype", [pl.FP4E2M1X2, pl.FP4E1M2X2])
    @pytest.mark.parametrize("transpose", [False, True])
    def test_fp4_gather_row_converts_only_cube_coordinates(self, dtype, transpose, tmp_path):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, src: pl.Tensor[[128, 128], dtype]):
                dst = pl.create_l1([128, 128], dtype, transpose=transpose)
                _ = pl.gather_row(dst, src, [32, 32], [0, 0], [1, 64], transpose=transpose)

        mlir = _emit_incore_mlir(Program)
        if find_ptoas_binary() is not None:
            # This used to fail PTOAS verification: the result type had logical
            # validity, but the subview operands still counted carriers.
            pto_backend._compile_pto_module(mlir, "fp4_gather", str(tmp_path))
        subview = next(line for line in mlir.splitlines() if "pto.subview" in line)
        partition = next(line for line in mlir.splitlines() if "pto.partition_view" in line)
        if transpose:
            assert "[%c32_index, %c64_index] sizes [64, 32] valid [%c64_index, %c2_index]" in subview
            assert "v_row=64, v_col=2" in subview
            assert "sizes = [%c64_index, %c1_index]" in partition
        else:
            assert "[%c64_index, %c32_index] sizes [32, 64] valid [%c2_index, %c64_index]" in subview
            assert "v_row=2, v_col=64" in subview
            assert "sizes = [%c1_index, %c64_index]" in partition

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
        # This straight-line case remains eligible for PTOAS's same-block
        # normalization. Cross-scope fills must be prebound by PyPTO itself.
        lines = mlir.splitlines()
        first_tget = next(i for i, line in enumerate(lines) if "pto.tget_scale_addr" in line)
        assert any("pto.tmov" in line and "scaling" in line for i, line in enumerate(lines) if i < first_tget)

    @pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
    @pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer-fallback"])
    @pytest.mark.parametrize("conditional", [False, True], ids=["loop", "conditional-loop"])
    def test_resident_mx_scale_bindings_precede_outer_fills(
        self, tmp_path, planner, enable_buffer_ir, conditional
    ):
        """Resident Left/Right scales must be filled at their eventual MX addresses."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[32, 64], pl.FP8E4M3FN],
                sa: pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[128, 32], pl.FP8E4M3FN],
                sb: pl.Tensor[[4, 32], pl.FP8E8M0, pl.MX_B_NN],
                flag: pl.Scalar[pl.INT32],
                out: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ):
                lhs0 = pl.move(pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), pl.Mem.Left)
                ls0 = pl.move(pl.load(sa, [0, 0], [16, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale)
                lhs1 = pl.move(pl.load(a, [16, 0], [16, 64], target_memory=pl.Mem.Mat), pl.Mem.Left)
                ls1 = pl.move(pl.load(sa, [16, 0], [16, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale)
                rhs0 = pl.move(pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs0 = pl.move(pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                rhs1 = pl.move(pl.load(b, [64, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs1 = pl.move(pl.load(sb, [2, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                for panel in pl.range(2):
                    if conditional:
                        if flag > 0:
                            dot0 = pl.matmul_mx(lhs0, ls0, rhs0, rs0)
                            out = pl.store(dot0, [panel * 32, 0], out)
                            dot1 = pl.matmul_mx(lhs1, ls1, rhs1, rs1)
                            out = pl.store(dot1, [panel * 32 + 16, 0], out)
                    else:
                        dot0 = pl.matmul_mx(lhs0, ls0, rhs0, rs0)
                        out = pl.store(dot0, [panel * 32, 0], out)
                        dot1 = pl.matmul_mx(lhs1, ls1, rhs1, rs1)
                        out = pl.store(dot1, [panel * 32 + 16, 0], out)
                return out

        mlir = _emit_incore_mlir(Program, planner=planner, enable_buffer_ir=enable_buffer_ir)
        # Two pairs on each side prevent offset zero from accidentally hiding
        # the resident scale's missing derived-address binding.
        _assert_outer_scale_fills_are_prebound(mlir)
        if conditional:
            assert "scf.if" in mlir
        if find_ptoas_binary() is not None:
            # Assembly must also accept the producer prebinding plus original
            # consumer bindings; checking PTO order alone would miss a later
            # normalization rejection from repeated scale bindings.
            cpp = pto_backend._compile_pto_module(mlir, "main", str(tmp_path), planner)
            assert "TMATMUL_MX(" in cpp

    @pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer-fallback"])
    def test_loop_carried_resident_mx_scale_bindings_precede_initial_fills(self, enable_buffer_ir):
        """An invariant descriptor carry still names its outer resident producer."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                sa: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                sb: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ):
                lhs = pl.move(pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), pl.Mem.Left)
                ls = pl.move(pl.load(sa, [0, 0], [16, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale)
                rhs = pl.move(pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs = pl.move(pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                for panel, (lhs_i, ls_i) in pl.range(2, init_values=(lhs, ls)):
                    dot = pl.matmul_mx(lhs_i, ls_i, rhs, rs)
                    pl.store(dot, [panel * 16, 0], out)
                    _lhs_out, _ls_out = pl.yield_(lhs_i, ls_i)
                return out

        mlir = _emit_incore_mlir(Program, enable_buffer_ir=enable_buffer_ir)
        _assert_outer_scale_fills_are_prebound(mlir, expected_fills=2)

    @pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer-fallback"])
    def test_cross_scope_scale_binding_preserves_consumer_valid_shape(self, tmp_path, enable_buffer_ir):
        """A dominating data allocation can have wider metadata than its MX alias."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                sa: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                sb: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ):
                lhs_root = pl.move(pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), pl.Mem.Left)
                # Only eight rows are valid in the paired scale. The consumer
                # narrows the data view to match, without changing its address.
                ls = pl.move(
                    pl.load(sa, [0, 0], [16, 2], valid_shape=[8, 2], target_memory=pl.Mem.Mat),
                    pl.Mem.LeftScale,
                )
                rhs = pl.move(pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs = pl.move(pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                for panel in pl.range(2):
                    lhs_view = pl.set_validshape(lhs_root, 8, 64)
                    dot = pl.matmul_mx(lhs_view, ls, rhs, rs)
                    out = pl.store(dot, [panel * 16, 0], out)
                return out

        mlir = _emit_incore_mlir(Program, enable_buffer_ir=enable_buffer_ir)
        _assert_outer_scale_fills_are_prebound(mlir, expected_fills=2)
        assert "pto.set_validshape" in mlir
        if find_ptoas_binary() is not None:
            cpp = pto_backend._compile_pto_module(mlir, "main", str(tmp_path))
            assert "TMATMUL_MX(" in cpp

    @pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer-fallback"])
    def test_cross_scope_scale_binding_preserves_both_metadata_aliases(self, tmp_path, enable_buffer_ir):
        """Producer geometry stays valid when both MX consumer views are narrower."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                sa: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                sb: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ):
                lhs_root = pl.move(pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), pl.Mem.Left)
                ls_root = pl.move(pl.load(sa, [0, 0], [16, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale)
                rhs = pl.move(pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs = pl.move(pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                for panel in pl.range(2):
                    # The outer fill is valid for all sixteen rows. Reusing
                    # the first consumer's narrower metadata for that producer
                    # would incorrectly reject the early address binding.
                    lhs_view = pl.set_validshape(lhs_root, 8, 64)
                    ls_view = pl.set_validshape(ls_root, 8, 2)
                    dot = pl.matmul_mx(lhs_view, ls_view, rhs, rs)
                    out = pl.store(dot, [panel * 16, 0], out)
                return out

        mlir = _emit_incore_mlir(Program, enable_buffer_ir=enable_buffer_ir)
        _assert_outer_scale_fills_are_prebound(mlir, expected_fills=2)
        assert mlir.count("pto.set_validshape") >= 2
        if find_ptoas_binary() is not None:
            cpp = pto_backend._compile_pto_module(mlir, "main", str(tmp_path))
            assert "TMATMUL_MX(" in cpp

    @pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer-fallback"])
    def test_cross_scope_scale_with_multiple_data_pairs_is_rejected(self, enable_buffer_ir):
        """One outer fill cannot populate distinct data-derived scale addresses."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[32, 64], pl.FP8E4M3FN],
                sa: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                sb: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ):
                lhs0 = pl.move(pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), pl.Mem.Left)
                lhs1 = pl.move(pl.load(a, [16, 0], [16, 64], target_memory=pl.Mem.Mat), pl.Mem.Left)
                ls = pl.move(pl.load(sa, [0, 0], [16, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale)
                rhs = pl.move(pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs = pl.move(pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                for panel in pl.range(2):
                    dot0 = pl.matmul_mx(lhs0, ls, rhs, rs)
                    out = pl.store(dot0, [panel * 32, 0], out)
                    dot1 = pl.matmul_mx(lhs1, ls, rhs, rs)
                    out = pl.store(dot1, [panel * 32 + 16, 0], out)
                return out

        with pytest.raises(ValueError, match="MX scale cross-scope fill requires a unique paired data tile"):
            _emit_incore_mlir(Program, enable_buffer_ir=enable_buffer_ir)

    @pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer-fallback"])
    def test_nested_changing_scale_carry_is_rejected(self, enable_buffer_ir):
        """Nested changing carries must still trace back to the unsafe outer fill."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                sa: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                sb: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ):
                lhs = pl.move(pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), pl.Mem.Left)
                initial_scale = pl.move(
                    pl.load(sa, [0, 0], [16, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale
                )
                rhs = pl.move(pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs = pl.move(pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                for outer, (outer_scale,) in pl.range(2, init_values=(initial_scale,)):
                    for inner, (inner_scale,) in pl.range(2, init_values=(outer_scale,)):
                        dot = pl.matmul_mx(lhs, inner_scale, rhs, rs)
                        pl.store(dot, [(outer * 2 + inner) * 16, 0], out)
                        # Both carries change: a one-hop search stops at the
                        # outer IterArg instead of finding initial_scale.
                        replacement_scale = pl.move(
                            pl.load(sa, [0, 0], [16, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale
                        )
                        inner_result = pl.yield_(replacement_scale)
                    _outer_result = pl.yield_(inner_result)
                return out

        with pytest.raises(ValueError, match="MX scale cross-scope fill requires a unique paired data tile"):
            _emit_incore_mlir(Program, enable_buffer_ir=enable_buffer_ir)

    @pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer-fallback"])
    def test_cross_scope_scale_with_inner_data_is_rejected(self, enable_buffer_ir):
        """Binding an outer fill must not refer to a later inner-loop tile handle."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[32, 64], pl.FP8E4M3FN],
                sa: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                sb: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ):
                ls = pl.move(pl.load(sa, [0, 0], [16, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale)
                rhs = pl.move(pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs = pl.move(pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                for panel in pl.range(2):
                    lhs = pl.move(
                        pl.load(a, [panel * 16, 0], [16, 64], target_memory=pl.Mem.Mat), pl.Mem.Left
                    )
                    dot = pl.matmul_mx(lhs, ls, rhs, rs)
                    out = pl.store(dot, [panel * 16, 0], out)
                return out

        with pytest.raises(
            ValueError, match="MX scale cross-scope fill requires its paired data tile to dominate the fill"
        ):
            _emit_incore_mlir(Program, enable_buffer_ir=enable_buffer_ir)

    @pytest.mark.parametrize("dtype", [pl.FP4E2M1X2, pl.FP4E1M2X2])
    @pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer-fallback"])
    def test_resident_packed_fp4_scale_validity_uses_carrier_units(self, tmp_path, dtype, enable_buffer_ir):
        """Logical scale M/N must not double the packed Left/Right carrier extents."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[32, 32], dtype],
                sa: pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], dtype],
                sb: pl.Tensor[[2, 64], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ):
                # 32 logical rows give a complete 16-row packed Mat box.
                lhs = pl.move(pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat), pl.Mem.Left)
                rhs_mat = pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat)
                rhs = pl.move(pl.tile.transpose_view(rhs_mat), pl.Mem.Right)
                ls = pl.move(pl.load(sa, [0, 0], [32, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale)
                rs = pl.move(pl.load(sb, [0, 0], [2, 64], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                for panel in pl.range(2):
                    out = pl.store(pl.matmul_mx(lhs, ls, rhs, rs), [panel * 32, 0], out)
                return out

        mlir = _emit_incore_mlir(Program, enable_buffer_ir=enable_buffer_ir)
        _assert_outer_scale_fills_are_prebound(mlir, expected_fills=2)
        if find_ptoas_binary() is not None:
            cpp = pto_backend._compile_pto_module(mlir, "main", str(tmp_path))
            assert "TMATMUL_MX(" in cpp

    @pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer-fallback"])
    def test_resident_scale_binding_accepts_dominating_data_slices(self, enable_buffer_ir):
        """A slice declared before a fill dominates it, and its offset belongs in the binding."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP8E4M3FN],
                sa: pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 64], pl.FP8E4M3FN],
                sb: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ):
                lhs_root = pl.move(pl.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat), pl.Mem.Left)
                lhs = pl.tile.slice(lhs_root, [32, 64], [32, 0])
                rhs_root = pl.move(pl.load(b, [0, 0], [64, 64], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rhs = pl.tile.slice(rhs_root, [64, 32], [0, 0])
                ls = pl.move(
                    pl.load(sa, [0, 0], [32, 2], target_memory=pl.Mem.Mat),
                    pl.Mem.LeftScale,
                )
                rs = pl.move(
                    pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat),
                    pl.Mem.RightScale,
                )
                for panel in pl.range(2):
                    out = pl.store(pl.matmul_mx(lhs, ls, rhs, rs), [panel * 32, 0], out)
                return out

        # The PTOAS planner retains these as actual view handles; the addressed
        # planner can fold slices into allocations and no longer isolates the
        # missing view-declaration registration.
        planner = passes.MemoryPlanner.PTOAS
        mlir = _emit_incore_mlir(Program, planner=planner, enable_buffer_ir=enable_buffer_ir)
        _assert_outer_scale_fills_are_prebound(mlir, expected_fills=2)
        assert "pto.subview" in mlir

    @pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer-fallback"])
    def test_resident_scale_binding_rejects_later_data_reshape(self, enable_buffer_ir):
        """A shared allocation does not make a later reshape descriptor available."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.FP8E4M3FN],
                sa: pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                sb: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ):
                lhs_root = pl.move(pl.load(a, [0, 0], [16, 128], target_memory=pl.Mem.Mat), pl.Mem.Left)
                ls = pl.move(pl.load(sa, [0, 0], [32, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale)
                lhs = pl.tile.reshape(lhs_root, [32, 64])
                rhs = pl.move(pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs = pl.move(pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                for panel in pl.range(2):
                    out = pl.store(pl.matmul_mx(lhs, ls, rhs, rs), [panel * 32, 0], out)
                return out

        with pytest.raises(
            ValueError, match="MX scale cross-scope fill requires its paired data tile to dominate the fill"
        ):
            _emit_incore_mlir(Program, planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=enable_buffer_ir)

    @pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer-fallback"])
    def test_resident_scale_binding_accepts_dominating_data_reshape(self, enable_buffer_ir):
        """Bind the emitted reshape descriptor, with its own rows, before filling scales."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.FP8E4M3FN],
                sa: pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                sb: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ):
                lhs_root = pl.move(pl.load(a, [0, 0], [16, 128], target_memory=pl.Mem.Mat), pl.Mem.Left)
                lhs = pl.tile.reshape(lhs_root, [32, 64])
                ls = pl.move(pl.load(sa, [0, 0], [32, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale)
                rhs = pl.move(pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs = pl.move(pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                for panel in pl.range(2):
                    out = pl.store(pl.matmul_mx(lhs, ls, rhs, rs), [panel * 32, 0], out)
                return out

        mlir = _emit_incore_mlir(
            Program, planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=enable_buffer_ir
        )
        _assert_outer_scale_fills_are_prebound(mlir, expected_fills=2)
        # Do not canonicalize reshape aliases here: the base has 16 rows, while
        # this scale needs the actual 32-row consumer descriptor.
        reshape = re.search(r"(%[\w.$-]+)\s*=\s*pto\.treshape[^\n]*loc=left[^\n]*rows=32, cols=64", mlir)
        assert reshape is not None
        binding = re.search(
            rf"pto\.tget_scale_addr ins\({re.escape(reshape.group(1))}\s*:[^\n]*outs\((%[\w.$-]+)",
            mlir,
        )
        assert binding is not None
        fill = re.search(rf"pto\.tmov[^\n]*outs\({re.escape(binding.group(1))}\s*:", mlir)
        assert fill is not None
        assert reshape.start() < binding.start() < fill.start() < mlir.index("scf.for")

    @pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer-fallback"])
    def test_branch_returned_scale_requires_a_proven_binding(self, enable_buffer_ir):
        """A phi must not hide both branch fills from the cross-region safety check."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                sa: pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                sb: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                flag: pl.Scalar[pl.INT32],
                out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ):
                lhs = pl.move(pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), pl.Mem.Left)
                rhs = pl.move(pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs = pl.move(pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                scale_mat0 = pl.load(sa, [0, 0], [16, 2], target_memory=pl.Mem.Mat)
                scale_mat1 = pl.load(sa, [16, 0], [16, 2], target_memory=pl.Mem.Mat)
                if flag > 0:
                    ls = pl.move(scale_mat0, pl.Mem.LeftScale)
                else:
                    ls = pl.move(scale_mat1, pl.Mem.LeftScale)
                for panel in pl.range(2):
                    out = pl.store(pl.matmul_mx(lhs, ls, rhs, rs), [panel * 16, 0], out)
                return out

        with pytest.raises(ValueError, match="MX scale bindings through branch or changing loop results"):
            # PTOAS retains the branch-result descriptors instead of applying
            # the legacy MemoryReuse branch-phi coalescing first.
            _emit_incore_mlir(Program, planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=enable_buffer_ir)

    @pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer-fallback"])
    def test_changing_loop_scale_result_requires_a_proven_binding(self, enable_buffer_ir):
        """A post-loop consumer must not lose the provenance of its replacement scale fill."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                sa: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                sb: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            ):
                lhs = pl.move(pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), pl.Mem.Left)
                initial = pl.move(pl.load(sa, [0, 0], [16, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale)
                rhs = pl.move(pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs = pl.move(pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                for _panel, (_carried,) in pl.range(2, init_values=(initial,)):
                    replacement = pl.move(
                        pl.load(sa, [0, 0], [16, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale
                    )
                    result = pl.yield_(replacement)
                out = pl.store(pl.matmul_mx(lhs, result, rhs, rs), [0, 0], out)
                return out

        with pytest.raises(ValueError, match="MX scale bindings through branch or changing loop results"):
            _emit_incore_mlir(Program, enable_buffer_ir=enable_buffer_ir)

    def test_cross_scope_scale_subview_requires_a_consumer_scope_fill(self):
        """A scale subview cannot silently lose the root fill's address provenance."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                sa: pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                sb: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ):
                lhs = pl.move(pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), pl.Mem.Left)
                scale_root = pl.move(pl.load(sa, [0, 0], [32, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale)
                ls = pl.tile.slice(scale_root, [16, 2], [16, 0])
                rhs = pl.move(pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs = pl.move(pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                for panel in pl.range(2):
                    out = pl.store(pl.matmul_mx(lhs, ls, rhs, rs), [panel * 16, 0], out)
                return out

        with pytest.raises(ValueError, match="MX scale views consumed across a loop or branch"):
            _emit_incore_mlir(Program, planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=False)

    def test_cross_scope_reinterpret_scale_view_requires_a_consumer_scope_fill(self):
        """Zero-copy dtype views must retain the resident fill's provenance."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                sa: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                sb: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ):
                lhs = pl.move(pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat), pl.Mem.Left)
                scale_root = pl.move(pl.load(sa, [0, 0], [16, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale)
                scale_bytes = pl.tile.reinterpret_view(scale_root, pl.UINT8)
                scale_view = pl.tile.reinterpret_view(scale_bytes, pl.FP8E8M0)
                rhs = pl.move(pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
                rs = pl.move(pl.load(sb, [0, 0], [2, 32], target_memory=pl.Mem.Mat), pl.Mem.RightScale)
                for panel in pl.range(2):
                    out = pl.store(pl.matmul_mx(lhs, scale_view, rhs, rs), [panel * 16, 0], out)
                return out

        with pytest.raises(ValueError, match="MX scale views consumed across a loop or branch"):
            _emit_incore_mlir(Program, planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=False)

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

        assert not uses_native_fp4_mx(_run_default_pipeline(Program))
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

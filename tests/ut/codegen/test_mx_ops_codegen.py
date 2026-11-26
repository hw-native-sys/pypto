# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen tests for MXFP8, native FP4, and explicit FP4-to-FP8 legalization."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType, pto_backend, reset_for_testing, set_backend_type
from pypto.backend._ptoas_fp4 import uses_native_fp4_mx
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


@pytest.fixture(autouse=True)
def _reset_backend_after_test():
    """Do not leak an explicitly selected backend into later test modules."""
    yield
    reset_for_testing()


def _run_default_pipeline(program, backend_type=BackendType.Ascend950, planner=passes.MemoryPlanner.PYPTO):
    """Run the Default pipeline for an explicitly selected backend."""
    reset_for_testing()
    set_backend_type(backend_type)
    with passes.PassContext([], memory_planner=planner):
        return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


def _emit_incore_mlir(program, backend_type=BackendType.Ascend950, planner=passes.MemoryPlanner.PYPTO) -> str:
    """Run Default pipeline and concatenate AIC/AIV MLIR."""
    optimized = _run_default_pipeline(program, backend_type, planner)
    parts: list[str] = []
    for func in optimized.functions.values():
        if func.func_type in (pl.FunctionType.Orchestration, pl.FunctionType.Group):
            continue
        single = ir.Program([func], func.name, optimized.span)
        result = codegen.PTOCodegen().generate(single, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
        parts.append(result if isinstance(result, str) else "".join(result.values()))
    return "\n".join(parts)


def _assemble_mx(mlir, tmp_path, planner=passes.MemoryPlanner.PYPTO, *, require_native_views=True):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source = tmp_path / "mx.pto"
    output = tmp_path / "mx.cpp"
    source.write_text(mlir)
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(output), ["--pto-arch=a5", f"--pto-level={level}"])
    cpp = output.read_text()
    # Acceptance alone is insufficient: MX must retain its packed ISA layout,
    # rather than silently falling back to an ordinary ND GlobalTensor.
    for layout in ("mx_a_zz", "mx_b_nn"):
        if require_native_views and f"#pto.layout<{layout}>" in mlir:
            assert f"pto::Layout::{layout.upper()}" in cpp
    return cpp


_MX_FP8_PAIRS = [
    pytest.param(pl.FP8E4M3FN, pl.FP8E4M3FN, id="e4m3-x-e4m3"),
    pytest.param(pl.FP8E5M2, pl.FP8E5M2, id="e5m2-x-e5m2"),
    pytest.param(pl.FP8E4M3FN, pl.FP8E5M2, id="e4m3-x-e5m2"),
    pytest.param(pl.FP8E5M2, pl.FP8E4M3FN, id="e5m2-x-e4m3"),
]

# tile_buf dtype spelling in emitted PTO, and the pto-isa C++ element type.
_PTO_FP8_SPELLING = {pl.FP8E4M3FN: "f8E4M3FN", pl.FP8E5M2: "f8E5M2"}
_ISA_FP8_SPELLING = {pl.FP8E4M3FN: "float8_e4m3_t", pl.FP8E5M2: "float8_e5m2_t"}


def _simulate_mx_load(program, tmp_path, backing, tile_shape, major, scalar_args=()):
    """Execute the generated TLOAD and return the Mat tile in logical order."""
    compiler = shutil.which("g++-15")
    if compiler is None:
        pytest.skip("GCC 15 is required for the CPU simulator")
    cpp = _assemble_mx(_emit_incore_mlir(program), tmp_path)

    isa_root = os.environ.get("PTO_ISA_ROOT")
    if isa_root:
        include_dir = Path(isa_root) / "include"
    else:
        pytest.importorskip("simpler_setup")
        from pypto.runtime import pto_isa_include_dir  # noqa: PLC0415

        include_dir = pto_isa_include_dir()
    assert (include_dir / "pto/pto-inst.hpp").is_file(), f"PTO-ISA headers missing in {include_dir}"

    rows, cols = tile_shape
    input_bytes = ", ".join(str(value) for value in backing)
    args = "".join(f", {value}" for value in scalar_args)
    source = tmp_path / "probe.cpp"
    source.write_text(
        "#include <array>\n#include <cstdint>\n#include <cstdio>\n"
        + cpp
        + f"""
int main() {{
    pto::NPU_MEMORY_INIT(pto::NPUArch::A5);
    const std::array<uint8_t, {len(backing)}> raw{{{input_bytes}}};
    std::array<float8_e8m0_t, raw.size()> input{{}};
    for (size_t i = 0; i < input.size(); ++i) {{
        input[i] = float8_e8m0_t::FromRaw(raw[i]);
    }}
    load_window(input.data(){args});

    // The generated kernel places its sole Mat tile at byte offset zero.
    Tile<TileType::Mat, float8_e8m0_t, {rows}, {cols}, BLayout::{major}, {rows}, {cols},
         SLayout::{major}, 32, PadValue::Null, CompactMode::Null> tile;
    TASSIGN(tile, 0);
    for (int row = 0; row < {rows}; ++row) {{
        for (int col = 0; col < {cols}; ++col) {{
            std::printf("%u ", static_cast<unsigned>(tile.GetElement(row, col).RawData()));
        }}
    }}
}}
"""
    )
    binary = tmp_path / "probe"
    result = subprocess.run(
        [
            compiler,
            "-std=c++23",
            "-pthread",
            "-D__CPU_SIM",
            "-DPTO_CPU_SIM_TARGET_A5",
            "-D__DAV_CUBE__",
            f"-I{include_dir}",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return [int(value) for value in subprocess.check_output([str(binary)], timeout=30).split()]


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("layout", ["MX_A_ZZ", "MX_B_NN"])
@pytest.mark.parametrize("nonzero_offset", [False, True], ids=["parent-stride", "nonzero-offset"])
def test_packed_mx_scale_subwindow_loads_expected_bytes(tmp_path, layout, nonzero_offset):
    """Test origin within one block and parent stride across two blocks independently."""
    # Both layouts store 16 outer lanes by 2 scale groups per physical fractal.
    # A's outer dimension is rows; B's outer dimension is columns.
    outer, groups = 64, 8
    outer_start, group_start = (16, 2) if nonzero_offset else (0, 0)
    outer_size, group_size = (16, 4) if nonzero_offset else (32, 4)
    logical = np.arange(outer * groups).reshape(outer, groups) % 251
    backing = logical.reshape(outer // 16, 16, groups // 2, 2).transpose(0, 2, 1, 3).ravel()
    window = logical[outer_start : outer_start + outer_size, group_start : group_start + group_size]
    if layout == "MX_A_ZZ":
        shape, offset, sizes, major = (
            (outer, groups),
            (outer_start, group_start),
            (outer_size, group_size),
            "RowMajor",
        )
    else:
        shape, offset, sizes, major = (
            (groups, outer),
            (group_start, outer_start),
            (group_size, outer_size),
            "ColMajor",
        )
        window = window.T
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def load_window(self, scales: pl.Tensor[[{shape[0]}, {shape[1]}], pl.FP8E8M0, pl.{layout}]):
        _ = pl.load(scales, [{offset[0]}, {offset[1]}], [{sizes[0]}, {sizes[1]}], target_memory=pl.Mem.Mat)
""")
    actual = _simulate_mx_load(program, tmp_path, backing, sizes, major)
    np.testing.assert_array_equal(np.array(actual).reshape(sizes), window)


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
        source, _, _ = generate_bundle(False)
        assert "pto.tmatmul.mx.acc" in source
        if find_ptoas_binary() is not None:
            cpp = pto_backend._compile_pto_module(source, "native_fp4", str(tmp_path))
            cpp_dtype = "float4_e2m1x2_t" if dtype == pl.FP4E2M1X2 else "float4_e1m2x2_t"
            assert f"Tile<TileType::Left, {cpp_dtype}, 32, 128," in cpp
            assert f"Tile<TileType::Right, {cpp_dtype}, 128, 64," in cpp
            assert "Tile<TileType::Acc, float, 32, 64," in cpp

    @pytest.mark.ptoas_compile
    @pytest.mark.parametrize(
        "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
    )
    @pytest.mark.parametrize("m", [32, 64])
    @pytest.mark.parametrize("lhs_dtype", [pl.FP4E2M1X2, pl.FP4E1M2X2])
    @pytest.mark.parametrize("rhs_dtype", [pl.FP4E2M1X2, pl.FP4E1M2X2])
    @pytest.mark.parametrize("lhs_layout", [pl.ND, pl.DN])
    @pytest.mark.parametrize("rhs_layout", [pl.ND, pl.DN])
    def test_tensor_native_fp4_matmul(
        self, lhs_dtype, rhs_dtype, lhs_layout, rhs_layout, planner, m, tmp_path
    ):
        lhs_rows, lhs_cols = (m // 2, 128) if lhs_layout == pl.DN else (m, 64)
        rhs_rows, rhs_cols = (64, 192) if rhs_layout == pl.DN else (128, 96)
        lhs_s0, lhs_s1 = (1, lhs_rows) if lhs_layout == pl.DN else (lhs_cols, 1)
        rhs_s0, rhs_s1 = (1, rhs_rows) if rhs_layout == pl.DN else (rhs_cols, 1)

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[
                    [lhs_rows, lhs_cols], lhs_dtype, pl.TensorView(stride=[lhs_s0, lhs_s1], layout=lhs_layout)
                ],
                sa: pl.Tensor[[m, 4], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[
                    [rhs_rows, rhs_cols], rhs_dtype, pl.TensorView(stride=[rhs_s0, rhs_s1], layout=rhs_layout)
                ],
                sb: pl.Tensor[[4, 192], pl.FP8E8M0, pl.MX_B_NN],
            ) -> pl.Tensor[[m, 192], pl.FP32]:
                return pl.matmul_mx(a, sa, b, sb)

        mlir = _emit_incore_mlir(Program, planner=planner)
        lhs_token = "f4E2M1x2" if lhs_dtype == pl.FP4E2M1X2 else "f4E1M2x2"
        rhs_token = "f4E2M1x2" if rhs_dtype == pl.FP4E2M1X2 else "f4E1M2x2"
        assert "pto.tcvt" not in mlir
        assert f"loc=left, dtype=!pto.{lhs_token}, rows={m}, cols=128" in mlir
        assert f"loc=right, dtype=!pto.{rhs_token}, rows=128, cols=192" in mlir
        assert f"loc=acc, dtype=f32, rows={m}, cols=192" in mlir
        assert "pto.tmatmul.mx" in mlir
        assert mlir.count("pto.tget_scale_addr") == 2
        cpp = _assemble_mx(mlir, tmp_path, planner)
        assert "TMATMUL_MX(" in cpp

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

    @pytest.mark.ptoas_compile
    def test_mx_scale_load_accepts_provable_dynamic_offset(self, tmp_path):
        """Provable MX offsets remain logical; PTOAS performs the physical addressing."""

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
        _assemble_mx(mlir, tmp_path)
        assert "cf.assert" not in mlir
        assert "arith.remui" not in mlir
        assert "arith.floordivsi" not in mlir
        assert "pto.tload" in mlir
        assert "{layout = #pto.layout<mx_a_zz>}" in mlir
        partitions = [line for line in mlir.splitlines() if "partition_view" in line]
        assert any("sizes = [%c16_index, %c2_index]" in line for line in partitions)

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
        """Unaligned / unprovable dynamic MX offsets fail at LegalizeMxScaleTensorViews."""

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

    @pytest.mark.ptoas_compile
    def test_mx_scale_load_accepts_narrowed_valid_shape(self, tmp_path):
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
        _assemble_mx(mlir, tmp_path)
        assert "pto.tload" in mlir
        partitions = [line for line in mlir.splitlines() if "partition_view" in line]
        # Physical TLoad box remains shapes=[16,2] -> one SFractal block row.
        assert any("sizes = [%c16_index, %c2_index]" in line for line in partitions)

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

    @pytest.mark.ptoas_compile
    def test_emits_tmatmul_mx_and_tget(self, tmp_path):
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
        _assemble_mx(mlir, tmp_path)
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
        assert "shape = [%c128_index, %c2_index]" in mx_a_view
        assert "strides = [%c2_index, %c1_index]" in mx_a_view
        assert "shape = [%c2_index, %c64_index]" in mx_b_view
        assert "strides = [%c64_index, %c1_index]" in mx_b_view
        # Mat→scale fill stays source-order; PTOAS later reorders bind-before-fill.
        lines = mlir.splitlines()
        first_tget = next(i for i, line in enumerate(lines) if "pto.tget_scale_addr" in line)
        assert any("pto.tmov" in line and "scaling" in line for i, line in enumerate(lines) if i < first_tget)

    def test_mx_scale_extract_binds_write_destination(self):
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
                am = pl.load(a, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
                asm = pl.load(a_s, [0, 0], [16, 4], target_memory=pl.Mem.Mat)
                bm = pl.load(b, [0, 0], [128, 32], target_memory=pl.Mem.Mat)
                bsm = pl.load(b_s, [0, 0], [4, 32], target_memory=pl.Mem.Mat)
                al = pl.tile.extract(am, 0, 64, [16, 64], target_memory=pl.Mem.Left)
                br = pl.tile.extract(bm, 64, 0, [64, 32], target_memory=pl.Mem.Right)
                asl: pl.Tile[[16, 2], pl.FP8E8M0, pl.MemRef("left_scale", slots=2)[1], pl.Mem.LeftScale] = (
                    pl.tile.extract(asm, 0, 2, [16, 2], target_memory=pl.Mem.LeftScale)
                )
                bsr = pl.tile.extract(bsm, 2, 0, [2, 32], target_memory=pl.Mem.RightScale)
                result = pl.matmul_mx(al, asl, br, bsr)
                pl.store(result, [0, 0], out)

        rewritten = passes.insert_mx_scale_addr()(passes.infer_tile_memory_space()(Program))
        rewritten_source = ir.python_print(rewritten)
        assert rewritten_source.count("tile.bind_scale_addr") == 2
        assert rewritten_source.count("tile.extract_into") == 2
        assert "tile.tget_scale_addr(" not in rewritten_source
        ir.assert_structural_equal(rewritten, pl.parse_program(rewritten_source))

        mlir = _emit_incore_mlir(Program)
        lines = mlir.splitlines()
        scale_extracts = [
            i for i, line in enumerate(lines) if "pto.textract" in line and "loc=scaling" in line
        ]
        bindings = [i for i, line in enumerate(lines) if "pto.tget_scale_addr" in line]
        assert len(scale_extracts) == len(bindings) == 2
        for extract, bind in zip(scale_extracts, bindings, strict=True):
            assert bind < extract
            bind_destination = lines[bind].split(" outs(", 1)[1].split(" : ", 1)[0]
            extract_destination = lines[extract].split(" outs(", 1)[1].split(" : ", 1)[0]
            assert bind_destination == extract_destination

    @pytest.mark.ptoas_compile
    def test_nd_backing_alias_emits_logical_mx_views(self, tmp_path):
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
        _assemble_mx(mlir, tmp_path)
        mx_views = [
            line for line in mlir.splitlines() if "pto.make_tensor_view" in line and "#pto.layout<mx_" in line
        ]
        assert len(mx_views) == 2
        assert all(line.count("?") == 2 for line in mx_views)
        assert any("#pto.layout<mx_a_zz>" in line for line in mx_views)
        assert any("#pto.layout<mx_b_nn>" in line for line in mx_views)

    @pytest.mark.ptoas_compile
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("dtype", [pl.FP8E4M3FN, pl.FP8E5M2, pl.FP4E2M1X2])
    def test_tensor_quant_mx_returns_payload_and_packed_scale(self, axis, dtype, tmp_path):
        rows = 16 if axis == 1 else (64 if dtype == pl.FP4E2M1X2 else 32)
        src_dtype = pl.BF16 if dtype == pl.FP4E2M1X2 else pl.FP32
        payload_rows, payload_cols = (rows, 64) if axis == 1 else (64, rows)
        if dtype == pl.FP4E2M1X2:
            payload_cols //= 2
        scale_shape = (rows, 2) if axis == 1 else (2, rows)
        scale_layout = pl.MX_A_ZZ if axis == 1 else pl.MX_B_NN

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self, src: pl.Tensor[[rows, 64], src_dtype]
            ) -> tuple[
                pl.Tensor[[payload_rows, payload_cols], dtype],
                pl.Tensor[scale_shape, pl.FP8E8M0, scale_layout],
            ]:
                data, scale = pl.quant_mx(src, group_axis=axis, dtype=dtype)
                return data, scale

        optimized = _run_default_pipeline(Program)
        kernel = next(func for func in optimized.functions.values() if ir.is_incore_type(func.func_type))
        scale_params = [
            param.type
            for param in kernel.params
            if isinstance(param.type, ir.TensorType) and param.type.dtype == pl.FP8E8M0
        ]
        (scale_type,) = scale_params
        assert isinstance(scale_type, ir.TensorType) and scale_type.tensor_view is not None
        assert scale_type.tensor_view.layout == scale_layout
        assert tuple(dim.value for dim in scale_type.shape if isinstance(dim, ir.ConstInt)) == scale_shape
        mlir = _emit_incore_mlir(Program)
        cpp = _assemble_mx(mlir, tmp_path, require_native_views=False)
        # The packed public GM view is metadata-only in this kernel; TSTORE
        # uses the flat backing bytes, not an MX-layout GlobalTensor.
        assert "TSTORE" in cpp and "pto::Layout::ND" in cpp
        stores = [line for line in mlir.splitlines() if "pto.tstore" in line and "!pto.f8E8M0" in line]
        assert len(stores) == 1
        assert "slayout=none_box" in stores[0]
        assert f"!pto.partition_tensor_view<1x{rows * 2}x!pto.f8E8M0>" in stores[0]
        assert f"#pto.layout<{'mx_a_zz' if axis == 1 else 'mx_b_nn'}>" in mlir

    @pytest.mark.ptoas_compile
    def test_full_quant_mx_scale_store_to_explicit_output(self, tmp_path):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self, src: pl.Tensor[[32, 128], pl.BF16], out: pl.Tensor[[32, 4], pl.FP8E8M0, pl.MX_A_ZZ]
            ):
                _data, scale = pl.quant_mx(pl.load(src, [0, 0], [32, 128]), group_axis=1)
                pl.store(scale, [0, 0], out)

        mlir = _emit_incore_mlir(Program)
        _assemble_mx(mlir, tmp_path, require_native_views=False)
        stores = [line for line in mlir.splitlines() if "pto.tstore" in line and "!pto.f8E8M0" in line]
        assert len(stores) == 1
        assert "!pto.partition_tensor_view<1x128x!pto.f8E8M0>" in stores[0]

    @pytest.mark.ptoas_compile
    def test_partial_quant_mx_scale_store_preserves_native_layout_diagnostic(self, tmp_path):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self, src: pl.Tensor[[16, 64], pl.FP16], out: pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ]
            ):
                _data, scale = pl.quant_mx(pl.load(src, [0, 0], [16, 64]), group_axis=1)
                pl.store(scale, [16, 0], out)

        mlir = _emit_incore_mlir(Program)
        # A subset of a packed tensor need not be contiguous. Do not silently
        # treat a partial window as a full ND byte-buffer write.
        with pytest.raises(RuntimeError, match="expects A5 vec tstore src layout"):
            _assemble_mx(mlir, tmp_path)

    @pytest.mark.parametrize("cast_layout", [None, pl.NZ], ids=["nd", "fused-nz"])
    def test_mxfp4_times_mxfp8_uses_explicit_lhs_cast(self, cast_layout):
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
                ta = pl.cast(pl.load(a, [0, 0], [128, 32]), pl.FP8E4M3FN, layout=cast_layout)
                tas = pl.load(a_s, [0, 0], [128, 2])
                tb = pl.load(b, [0, 0], [64, 64])
                tbs = pl.load(b_s, [0, 0], [2, 64])
                pl.store(pl.matmul_mx(ta, tas, tb, tbs), [0, 0], out)

        assert not uses_native_fp4_mx(_run_default_pipeline(Program))
        mlir = _emit_incore_mlir(Program)
        # E2M1 now widens exactly in one pass. The explicit NZ destination
        # also removes the layout-only Vec move before the MX peer transfer.
        assert mlir.count("pto.tcvt") == 1
        vec_allocs = [line for line in mlir.splitlines() if "pto.alloc_tile" in line and "loc=vec" in line]
        assert all("dtype=bf16" not in line and "dtype=f32" not in line for line in vec_allocs)
        vec_layout_moves = [
            line for line in mlir.splitlines() if "pto.tmov" in line and line.count("loc=vec") == 2
        ]
        assert len(vec_layout_moves) == (1 if cast_layout is None else 0)
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

    @pytest.mark.ptoas_compile
    def test_mx_view_preserves_aligned_logical_offsets(self, tmp_path):
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
        _assemble_mx(mlir, tmp_path)
        partitions = [
            line for line in mlir.splitlines() if "partition_view" in line and "!pto.f8E8M0" in line
        ]
        assert any("offsets = [%c16_index, %c2_index]" in line for line in partitions)
        assert any("offsets = [%c2_index, %c0_index]" in line for line in partitions)

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

    @pytest.mark.parametrize(("lhs_dtype", "rhs_dtype"), _MX_FP8_PAIRS)
    @pytest.mark.parametrize(
        "compile_with_ptoas", [False, pytest.param(True, marks=pytest.mark.ptoas_compile)]
    )
    def test_every_fp8_pair_emits_its_own_operand_dtypes(
        self, tmp_path, compile_with_ptoas, lhs_dtype, rhs_dtype
    ):
        """Each MX FP8 pair reaches PTOAS with per-side dtypes on data, scale-addr and matmul.

        Operand placement is left to the compiler (no hand-written ``pl.move``), which
        is the form the ``matmul_mx`` family is documented to accept and the only one
        whose Bias staging PTOAS lowers.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 64], lhs_dtype],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], rhs_dtype],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                bias: pl.Tensor[[1, 32], pl.FP32],
                out: pl.Tensor[[16, 32], pl.FP32],
                out_acc: pl.Tensor[[16, 32], pl.FP32],
                out_bias: pl.Tensor[[16, 32], pl.FP32],
            ):
                lhs = pl.load(a, [0, 0], [16, 64])
                lhs_scale = pl.load(a_s, [0, 0], [16, 2])
                rhs = pl.load(b, [0, 0], [64, 32])
                rhs_scale = pl.load(b_s, [0, 0], [2, 32])
                bias_tile = pl.load(bias, [0, 0], [1, 32])
                base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
                pl.store(base, [0, 0], out)
                pl.store(pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale), [0, 0], out_acc)
                pl.store(pl.matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias_tile), [0, 0], out_bias)

        mlir = _emit_incore_mlir(Program)
        lhs_pto = _PTO_FP8_SPELLING[lhs_dtype]
        rhs_pto = _PTO_FP8_SPELLING[rhs_dtype]
        assert f"loc=left, dtype={lhs_pto}," in mlir
        assert f"loc=right, dtype={rhs_pto}," in mlir
        # Each side's scale-addr derivation reads that side's own data dtype; a
        # dtype hardcoded in the MX path would make the two sides agree wrongly.
        tget_lines = [line for line in mlir.splitlines() if "pto.tget_scale_addr" in line]
        left_tget = [line for line in tget_lines if "loc=left," in line]
        right_tget = [line for line in tget_lines if "loc=right," in line]
        assert len(left_tget) + len(right_tget) == len(tget_lines) == 2
        assert f"loc=left, dtype={lhs_pto}," in left_tget[0]
        assert f"loc=right, dtype={rhs_pto}," in right_tget[0]
        for pto_op in ("pto.tmatmul.mx ", "pto.tmatmul.mx.acc ", "pto.tmatmul.mx.bias "):
            line = next(line for line in mlir.splitlines() if pto_op in line)
            assert f"loc=left, dtype={lhs_pto}," in line
            assert f"loc=right, dtype={rhs_pto}," in line
            assert "loc=acc, dtype=f32," in line
        assert "loc=bias, dtype=f32," in mlir

        if compile_with_ptoas:
            cpp = _assemble_mx(mlir, tmp_path)
            assert f"Tile<TileType::Left, {_ISA_FP8_SPELLING[lhs_dtype]}" in cpp
            assert f"Tile<TileType::Right, {_ISA_FP8_SPELLING[rhs_dtype]}" in cpp
            # base + acc + bias all lower to TMATMUL_MX overloads.
            assert cpp.count("TMATMUL_MX(") == 3


class TestMxANDScaleCodegen:
    @pytest.mark.parametrize("row_offset", [0, 5])
    def test_clamped_source_rows_bound_the_dma_partition(self, row_offset):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, scales: pl.Tensor[[19, 72], pl.FP8E8M0, pl.MX_A_ND]):
                _ = pl.load(scales, [row_offset, 0], [32, 72], clamp=True)

        mlir = _emit_incore_mlir(Program)
        (tload,) = [line for line in mlir.splitlines() if "pto.tload " in line]
        # Allocating a 32-row tile is valid; transferring 32 rows from the
        # 19-row source would overread, including when valid_shape was omitted.
        assert f"!pto.partition_tensor_view<{19 - row_offset}x72x!pto.f8E8M0>" in tload
        assert "!pto.partition_tensor_view<32x72x!pto.f8E8M0>" not in tload
        assert "loc=mat, dtype=!pto.f8E8M0, rows=32, cols=72" in mlir

    @pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
    def test_compact_a_scales_load_directly_to_zz_mat(self, tmp_path, planner):
        """Keep row-major scale ABI and load the dynamic tail on AIC, without Vec packing."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a_s: pl.Tensor[[37, 72], pl.FP8E8M0, pl.MX_A_ND],
                row_base: pl.Scalar[pl.INDEX],
                valid_rows: pl.Scalar[pl.INDEX],
            ):
                scale = pl.load(a_s, [row_base, 0], [16, 72], valid_shape=[valid_rows, 72])
                _ = pl.move(scale, pl.Mem.LeftScale)

        reset_for_testing()
        set_backend_type(BackendType.Ascend950)
        with passes.PassContext([], memory_planner=planner):
            optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Program)
        mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
        assert "#pto.layout<mx_a_nd>" in mlir
        assert "loc=mat, dtype=!pto.f8E8M0, rows=16, cols=72" in mlir
        assert "slayout=row_major, fractal=32" in mlir
        assert "pto.tgather" not in mlir
        assert "pto.tpush" not in mlir
        assert "arith.floordivsi" not in mlir
        assert "!pto.partition_tensor_view<?x72x!pto.f8E8M0>" in mlir
        if find_ptoas_binary() is None:
            pytest.skip("PTOAS is not available")
        cpp = pto_backend._compile_pto_module(mlir, "mx_a_nd_compact", str(tmp_path), planner)
        assert "pto::Layout::MX_A_ND" in cpp
        assert "TLOAD(" in cpp
        assert "TMOV(" in cpp
        assert "ScaleLeft" in cpp
        assert "TileShape2D<float8_e8m0_t" in cpp

    def test_nd_backing_view_keeps_row_major_abi(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, backing: pl.Tensor[[1, 37 * 72], pl.FP8E8M0]):
                scales = pl.tensor.view(backing, [37, 72], layout=pl.MX_A_ND)
                _ = pl.load(scales, [5, 0], [16, 72], valid_shape=[7, 72])

        mlir = _emit_incore_mlir(Program)
        views = [line for line in mlir.splitlines() if "pto.make_tensor_view" in line]
        assert any("#pto.layout<mx_a_nd>" in line for line in views)
        assert "#pto.layout<mx_a_zz>" not in mlir
        assert "pto.tload" in mlir
        assert "pto.tmov" not in mlir

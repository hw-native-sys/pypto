# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen checks for the PTOAS v0.67 packed FP4 GM carrier ABI."""

import re

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _get_ptoas_flags, _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.language.parser.diagnostics import InvalidOperationError
from pypto.pypto_core import codegen, passes

_FP4_BYTE_ALIAS_PAIRS = [
    (pl.UINT8, pl.FP4E2M1X2),
    (pl.INT8, pl.FP4E2M1X2),
    (pl.FP4E2M1X2, pl.UINT8),
    (pl.FP4E2M1X2, pl.INT8),
]


@pytest.fixture(autouse=True)
def _reset_backend_after_test():
    yield
    reset_for_testing()


def _emit_incore_mlir(program, planner=passes.MemoryPlanner.PYPTO, *, enable_buffer_ir=True) -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    parts: list[str] = []
    for func in optimized.functions.values():
        if func.func_type in (pl.FunctionType.Orchestration, pl.FunctionType.Group):
            continue
        single = ir.Program([func], func.name, optimized.span)
        result = codegen.PTOCodegen().generate(single, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
        parts.append(result if isinstance(result, str) else "".join(result.values()))
    return "\n".join(parts)


def _uint8_ub_fp4_decode_program(dtype=pl.fp4e2m1x2):
    @pl.program
    class Decode:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            packed: pl.Tensor[[96, 320], pl.UINT8],
            output: pl.Out[pl.Tensor[[96, 640], pl.BF16]],
        ) -> pl.Tensor[[96, 640], pl.BF16]:
            payload = pl.tile.load(packed, [32, 32], [32, 256])
            fp4 = pl.reinterpret_view(payload, dtype)
            decoded = pl.cast(fp4, pl.BF16)
            return pl.tile.store(decoded, [32, 64], output)

    return Decode


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize(("dtype", "pto_type"), [(pl.fp4e2m1x2, "f4E2M1x2"), (pl.fp4e1m2x2, "f4E1M2x2")])
def test_uint8_ub_fp4_alias_is_consumed_by_native_cast_and_store(planner, dtype, pto_type):
    mlir = _emit_incore_mlir(_uint8_ub_fp4_decode_program(dtype), planner)
    assert mlir.count("pto.tcvt ") == 1, mlir
    assert mlir.count("pto.tload ") == mlir.count("pto.tstore ") == 1, mlir
    cast = next(line for line in mlir.splitlines() if "pto.tcvt " in line)
    assert f"dtype=!pto.{pto_type}, rows=32, cols=256" in cast, cast
    assert "dtype=bf16, rows=32, cols=512" in cast, cast
    gm_views = [
        line for line in mlir.splitlines() if "pto.make_tensor_view" in line or "pto.partition_view" in line
    ]
    assert gm_views and all(pto_type not in line for line in gm_views), mlir
    assert not any(op in mlir for op in ("pto.tand", "pto.tshr", "pto.tshl", "pto.tscatter")), mlir

    assert "pto.treshape" not in mlir, mlir
    aliases = [
        line
        for line in mlir.splitlines()
        if "pto.alloc_tile" in line and ("dtype=ui8" in line or f"dtype=!pto.{pto_type}" in line)
    ]
    assert len(aliases) == 2, mlir
    addresses = [match.group(1) for line in aliases if (match := re.search(r"addr = ([^, ]+)", line))]
    assert len(addresses) == 2 and addresses[0] == addresses[1], aliases


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize("transpose", [False, True])
def test_fp4_decode_fp8_byte_alias_is_consumed(planner, transpose):
    @pl.program
    class DecodeToFp8Bytes:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            packed: pl.Tensor[[128, 64], pl.UINT8],
            output: pl.Out[pl.Tensor[[128, 128], pl.INT8]],
        ) -> pl.Tensor[[128, 128], pl.INT8]:
            payload = pl.load(packed, [0, 0], [128, 64])
            fp4 = pl.reinterpret_view(payload, pl.FP4E2M1X2)
            decoded = pl.cast(fp4, pl.BF16)
            fp8 = pl.cast(decoded, pl.FP8E4M3FN, mode="rint")
            byte_view = pl.reinterpret_view(fp8, pl.INT8)
            if transpose:
                result = pl.transpose(byte_view, 0, 1)
            else:
                result = byte_view
            return pl.store(result, [0, 0], output)

    mlir = _emit_incore_mlir(DecodeToFp8Bytes, planner)
    casts = [line for line in mlir.splitlines() if "pto.tcvt " in line]
    # Ascend950 legalizes BF16 -> FP8 through FP32; the final byte alias
    # must remain zero-copy after that existing conversion sequence.
    assert len(casts) == 3, mlir
    assert "dtype=!pto.f4E2M1x2, rows=128, cols=64" in casts[0], casts[0]
    assert "dtype=bf16, rows=128, cols=128" in casts[0], casts[0]
    assert "dtype=bf16, rows=128, cols=128" in casts[1], casts[1]
    assert "dtype=f32, rows=128, cols=128" in casts[1], casts[1]
    assert "dtype=f32, rows=128, cols=128" in casts[2], casts[2]
    assert "dtype=f8E4M3FN, rows=128, cols=128" in casts[2], casts[2]
    assert "rmode = #pto<round_mode RINT>" in casts[1], casts[1]
    assert "rmode = #pto<round_mode RINT>" in casts[2], casts[2]
    assert mlir.count("pto.tload ") == mlir.count("pto.tstore ") == 1, mlir
    assert mlir.count("pto.ttrans ") == int(transpose), mlir
    assert "pto.treshape" not in mlir, mlir

    consumer_op = "pto.ttrans " if transpose else "pto.tstore "
    consumer = next(line for line in mlir.splitlines() if consumer_op in line)
    source = re.search(r"ins\((%[\w.$-]+)", consumer)
    assert source is not None, consumer
    allocations = [line for line in mlir.splitlines() if "pto.alloc_tile" in line]
    fp8_alloc = next(line for line in allocations if "dtype=f8E4M3FN" in line)
    byte_alloc = next(line for line in allocations if line.strip().startswith(f"{source.group(1)} = "))
    assert "dtype=i8, rows=128, cols=128" in byte_alloc, byte_alloc
    addresses = [
        match.group(1) for line in (fp8_alloc, byte_alloc) if (match := re.search(r"addr = ([^, ]+)", line))
    ]
    assert len(addresses) == 2 and addresses[0] == addresses[1], (fp8_alloc, byte_alloc)


@pytest.mark.parametrize("dtype", [pl.FP4E2M1X2, pl.FP4E1M2X2])
def test_uint8_ub_fp4_alias_with_ptoas_memory_planner(tmp_path, dtype):
    mlir = _emit_incore_mlir(_uint8_ub_fp4_decode_program(dtype), passes.MemoryPlanner.PTOAS)
    assert "pto.treshape " in mlir
    if find_ptoas_binary() is not None:
        pto, cpp = tmp_path / "packed_alias.pto", tmp_path / "packed_alias.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", "--pto-level=level2", "--pto-arch=a5"])


def _uint8_slice_fp4_decode_program(
    parent_cols, column_offset, source_dtype=pl.UINT8, target_dtype=pl.FP4E2M1X2
):
    # Always bridge through a distinct byte dtype before native FP4 decoding;
    # A5 has no general UINT8/INT8-to-BF16 tile.cast legalization.
    bridge_dtype = pl.INT8 if target_dtype == pl.UINT8 else pl.UINT8

    @pl.program
    class DecodeSlice:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            packed: pl.Tensor[[32, parent_cols], source_dtype],
            output: pl.Out[pl.Tensor[[32, 512], pl.BF16]],
        ) -> pl.Tensor[[32, 512], pl.BF16]:
            payload = pl.tile.load(packed, [0, 0], [32, parent_cols])
            window = pl.tile.slice(payload, [32, 256], [0, column_offset])
            alias = pl.reinterpret_view(window, target_dtype)
            bridge = pl.reinterpret_view(alias, bridge_dtype)
            fp4 = pl.reinterpret_view(bridge, pl.FP4E2M1X2)
            return pl.tile.store(pl.cast(fp4, pl.BF16), [0, 0], output)

    return DecodeSlice


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize(("parent_cols", "column_offset"), [(512, 256), (320, 32)])
@pytest.mark.parametrize(("source_dtype", "target_dtype"), _FP4_BYTE_ALIAS_PAIRS)
def test_uint8_ub_fp4_alias_rejects_noncontiguous_slice(
    planner, parent_cols, column_offset, source_dtype, target_dtype
):
    # Both directions would erase the parent pitch if their alloc_tile alias
    # were accepted; signed byte labels have the same storage requirement.
    program = _uint8_slice_fp4_decode_program(parent_cols, column_offset, source_dtype, target_dtype)
    with pytest.raises(ValueError, match="FP4E2M1X2 UB alias requires a contiguous byte-carrier tile"):
        _emit_incore_mlir(program, planner)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_uint8_ub_fp4_alias_rejects_noncontiguous_slice_through_views(planner):
    @pl.program
    class DecodeAliasedSlice:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            packed: pl.Tensor[[32, 512], pl.UINT8],
            output: pl.Out[pl.Tensor[[16, 1024], pl.BF16]],
        ) -> pl.Tensor[[16, 1024], pl.BF16]:
            payload = pl.tile.load(packed, [0, 0], [32, 512])
            window = pl.tile.slice(payload, [32, 256], [0, 256])
            alias = window
            reshaped = pl.tile.reshape(alias, [16, 512])
            fp4 = pl.reinterpret_view(reshaped, pl.FP4E2M1X2)
            return pl.tile.store(pl.cast(fp4, pl.BF16), [0, 0], output)

    with pytest.raises(ValueError, match="FP4E2M1X2 UB alias requires a contiguous byte-carrier tile"):
        _emit_incore_mlir(DecodeAliasedSlice, planner)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_uint8_ub_fp4_alias_rejects_nested_noncontiguous_slice(planner):
    @pl.program
    class DecodeNestedSlice:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            packed: pl.Tensor[[32, 512], pl.UINT8],
            output: pl.Out[pl.Tensor[[16, 512], pl.BF16]],
        ) -> pl.Tensor[[16, 512], pl.BF16]:
            payload = pl.tile.load(packed, [0, 0], [32, 512])
            first = pl.tile.slice(payload, [32, 256], [0, 256])
            second = pl.tile.slice(first, [16, 256], [0, 0])
            fp4 = pl.reinterpret_view(second, pl.FP4E2M1X2)
            return pl.tile.store(pl.cast(fp4, pl.BF16), [0, 0], output)

    with pytest.raises(ValueError, match="FP4E2M1X2 UB alias requires a contiguous byte-carrier tile"):
        _emit_incore_mlir(DecodeNestedSlice, planner)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize(("source_dtype", "target_dtype"), _FP4_BYTE_ALIAS_PAIRS)
def test_uint8_ub_fp4_alias_rejects_dynamic_ub_offset_through_reshape(planner, source_dtype, target_dtype):
    # Select the authored program before parsing: a packed tile must become a
    # byte view before reshaping, while byte input can reshape before aliasing.
    if source_dtype == pl.FP4E2M1X2:

        @pl.program
        class DecodeDynamicSlice:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                packed: pl.Tensor[[64, 256], source_dtype],
                start: pl.Scalar[pl.INT64],
                output: pl.Out[pl.Tensor[[16, 1024], pl.BF16]],
            ) -> pl.Tensor[[16, 1024], pl.BF16]:
                payload = pl.tile.load(packed, [0, 0], [64, 256])
                window = pl.tile.slice(payload, [32, 256], [start, 0])
                byte_alias = pl.reinterpret_view(window, target_dtype)
                reshaped = pl.tile.reshape(byte_alias, [16, 512])
                fp4 = pl.reinterpret_view(reshaped, pl.FP4E2M1X2)
                return pl.tile.store(pl.cast(fp4, pl.BF16), [0, 0], output)

    else:

        @pl.program
        class DecodeDynamicSlice:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                packed: pl.Tensor[[64, 256], source_dtype],
                start: pl.Scalar[pl.INT64],
                output: pl.Out[pl.Tensor[[16, 1024], pl.BF16]],
            ) -> pl.Tensor[[16, 1024], pl.BF16]:
                payload = pl.tile.load(packed, [0, 0], [64, 256])
                window = pl.tile.slice(payload, [32, 256], [start, 0])
                reshaped = pl.tile.reshape(window, [16, 512])
                alias = pl.reinterpret_view(reshaped, target_dtype)
                return pl.tile.store(pl.cast(alias, pl.BF16), [0, 0], output)

    with pytest.raises(ValueError, match="FP4E2M1X2 UB alias requires a contiguous byte-carrier tile"):
        _emit_incore_mlir(DecodeDynamicSlice, planner)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize(("source_dtype", "target_dtype"), _FP4_BYTE_ALIAS_PAIRS)
def test_uint8_ub_fp4_alias_accepts_contiguous_slice(planner, source_dtype, target_dtype):
    mlir = _emit_incore_mlir(_uint8_slice_fp4_decode_program(256, 0, source_dtype, target_dtype), planner)
    assert mlir.count("pto.tcvt ") == 1, mlir
    assert "pto.treshape" not in mlir, mlir


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize(("source_dtype", "target_dtype"), _FP4_BYTE_ALIAS_PAIRS)
def test_uint8_ub_fp4_alias_rejects_memref_less_tpop(planner, source_dtype, target_dtype):
    bridge_dtype = pl.INT8 if target_dtype == pl.UINT8 else pl.UINT8

    @pl.program
    class DecodePop:
        @pl.function(type=pl.FunctionType.AIV)
        def main(self, output: pl.Out[pl.Tensor[[32, 512], pl.BF16]]) -> pl.Tensor[[32, 512], pl.BF16]:
            buf = pl.reserve_buffer(name="c2v_slot_buffer", size=16384, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=8192, c2v_consumer_buf=buf)
            payload: pl.Tile[[32, 256], source_dtype, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
            alias = pl.reinterpret_view(payload, target_dtype)
            bridge = pl.reinterpret_view(alias, bridge_dtype)
            fp4 = pl.reinterpret_view(bridge, pl.FP4E2M1X2)
            decoded = pl.cast(fp4, pl.BF16)
            pl.tfree_to_aic(payload)
            return pl.tile.store(decoded, [0, 0], output)

    with pytest.raises(ValueError, match="MemRef-less inputs"):
        _emit_incore_mlir(DecodePop, planner)


def test_fp4e2m1x2_gm_views_preserve_carrier_units():
    """Parameters, tensor.view, loads and stores retain packed carrier coordinates."""

    @pl.program
    class Rank2:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[2, 256], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[2, 256], pl.FP4E2M1X2]],
        ) -> pl.Tensor[[2, 256], pl.FP4E2M1X2]:
            viewed: pl.Tensor[[2, 256], pl.FP4E2M1X2] = pl.tensor.view(src, [2, 256])
            return pl.store(pl.load(viewed, [0, 0], [2, 256]), [0, 0], out)

    mlir = _emit_incore_mlir(Rank2)
    views = [line for line in mlir.splitlines() if "pto.make_tensor_view" in line and "f4E2M1x2" in line]
    assert views and all("shape = [%c2_index, %c256_index]" in line for line in views), mlir
    assert all("strides = [%c256_index, %c1_index]" in line for line in views), mlir
    # PTOAS expands the DMA descriptor itself. Pre-expansion doubles the row pitch.
    partitions = [line for line in mlir.splitlines() if "partition_view" in line]
    assert len(partitions) == 2, mlir
    assert all("sizes = [%c2_index, %c256_index]" in line for line in partitions), mlir
    assert all("!pto.partition_tensor_view<2x256x!pto.f4E2M1x2>" in line for line in partitions), mlir

    @pl.program
    class Rank3:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[2, 16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[2, 16, 32], pl.FP4E2M1X2]],
        ) -> pl.Tensor[[2, 16, 32], pl.FP4E2M1X2]:
            return pl.store(pl.load(src, [0, 0, 0], [2, 16, 32]), [0, 0, 0], out)

    mlir3 = _emit_incore_mlir(Rank3)
    assert "!pto.f4E2M1x2" in mlir3
    views3 = [line for line in mlir3.splitlines() if "pto.make_tensor_view" in line and "f4E2M1x2" in line]
    assert views3, mlir3
    assert all("shape = [%c2_index, %c16_index, %c32_index]" in line for line in views3), mlir3
    assert "arith.muli %c32_index, %c16_index : index" in mlir3, mlir3
    assert "arith.muli %c32_index, %c2_index : index" not in mlir3, mlir3
    assert all(re.search(r"strides = \[%\w+_s0, %c32_index, %c1_index\]", line) for line in views3), mlir3
    partitions3 = [line for line in mlir3.splitlines() if "partition_view" in line]
    assert len(partitions3) == 2, mlir3
    assert all("sizes = [%c2_index, %c16_index, %c32_index]" in line for line in partitions3), mlir3


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
def test_packed_fp4_preserves_nd_for_one_carrier_width(dtype):
    """One carrier is two nibbles, so the GM layout must stay ND."""

    @pl.program
    class ColVecPacked:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[32, 1], dtype],
            out: pl.Out[pl.Tensor[[32, 1], dtype]],
        ) -> pl.Tensor[[32, 1], dtype]:
            return pl.store(pl.load(src, [0, 0], [32, 1]), [0, 0], out)

    mlir = _emit_incore_mlir(ColVecPacked)
    views = [line for line in mlir.splitlines() if "pto.make_tensor_view" in line]
    assert views and all("layout = #pto.layout<nd>" in line for line in views), mlir
    allocations = [line for line in mlir.splitlines() if "pto.alloc_tile" in line]
    assert allocations and all("blayout=row_major" in line for line in allocations), mlir


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
def test_packed_fp4_preserves_dn_carrier_geometry(dtype):
    """DN packs the row axis; GM offsets and strides remain carrier units."""

    @pl.program
    class DnAnnotated:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[64, 32], dtype, pl.TensorView(stride=[1, 64], layout=pl.TensorLayout.DN)],
            out: pl.Out[pl.Tensor[[64, 32], dtype, pl.TensorView(stride=[1, 64], layout=pl.TensorLayout.DN)]],
        ) -> pl.Tensor[[64, 32], dtype, pl.TensorView(stride=[1, 64], layout=pl.TensorLayout.DN)]:
            return pl.tile.store(pl.tile.load(src, [32, 0], [32, 32]), [0, 0], out)

    mlir = _emit_incore_mlir(DnAnnotated)
    views = [line for line in mlir.splitlines() if "pto.make_tensor_view" in line]
    assert views and all("strides = [%c1_index, %c64_index]" in line for line in views), mlir
    allocations = [line for line in mlir.splitlines() if "pto.alloc_tile" in line]
    assert allocations and all("blayout=col_major" in line for line in allocations), mlir


@pytest.mark.parametrize("planner", list(passes.MemoryPlanner))
@pytest.mark.parametrize("dtype", ["FP4E2M1X2", "FP4E1M2X2"])
@pytest.mark.parametrize("valid", [(0, 0), (0, 32), (32, 0)])
@pytest.mark.parametrize("inferred_valid", [False, True])
@pytest.mark.parametrize("buffer_ir", [False, True])
@pytest.mark.parametrize("column_offset", [0, 32])
def test_empty_fp4_slice_compiles_with_initialized_validity(
    tmp_path, planner, dtype, valid, inferred_valid, buffer_ir, column_offset
):
    """Empty FP4 slices use addressed descriptors or fail with an actionable diagnostic."""
    rows, cols = valid
    window_cols = 64 - column_offset
    load_valid = f", valid_shape=[32 + {rows}, {column_offset} + {cols}]" if inferred_valid else ""
    slice_valid = "" if inferred_valid else f", valid_shape=[{rows}, {cols}]"
    program = pl.parse(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, src: pl.Tensor[[64, 64], pl.{dtype}],
        out: pl.Out[pl.Tensor[[32, {window_cols * 2}], pl.BF16]]
    ) -> pl.Tensor[[32, {window_cols * 2}], pl.BF16]:
        tile = pl.load(src, [0, 0], [64, 64]{load_valid})
        window = pl.tile.slice(tile, [32, {window_cols}], [32, {column_offset}]{slice_valid})
        decoded = pl.cast(window, pl.BF16)
        return pl.store(decoded, [0, 0], out)
""")
    if planner == passes.MemoryPlanner.PTOAS or column_offset:
        message = (
            "static PYPTO or DSA_RP address" if planner == passes.MemoryPlanner.PTOAS else "contiguous Vec"
        )
        with pytest.raises(ValueError, match=message):
            _emit_incore_mlir(program, planner, enable_buffer_ir=buffer_ir)
        return
    mlir = _emit_incore_mlir(program, planner, enable_buffer_ir=buffer_ir)
    if find_ptoas_binary() is not None:
        pto = tmp_path / "empty_fp4_slice.pto"
        cpp = tmp_path / "empty_fp4_slice.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), _get_ptoas_flags(planner))
        assert cpp.is_file()
    assert "pto.treshape" not in mlir, mlir
    descriptor = next(line for line in mlir.splitlines() if "%window__ssa_v0 = pto.alloc_tile" in line)
    source = next(line for line in mlir.splitlines() if "%tile__ssa_v0 = pto.alloc_tile" in line)
    source_address = re.search(r"addr = %c(\d+)_i64", source)
    slice_address = re.search(r"addr = %c(\d+)_i64", descriptor)
    assert source_address and slice_address, (source, descriptor)
    assert int(slice_address[1]) - int(source_address[1]) == 32 * 64, (source, descriptor)
    assert f"valid_row = %c{rows}_index valid_col = %c{cols}_index" in descriptor, descriptor
    cast = next(line for line in mlir.splitlines() if "pto.tcvt" in line)
    assert "ins(%window__ssa_v0 " in cast, cast
    assert "v_row=0" not in mlir and "v_col=0" not in mlir, mlir
    assert "pto.tstore" not in mlir, mlir


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize("dtype", ["FP4E2M1X2", "FP4E1M2X2"])
def test_empty_fp4_replay_slice_uses_initialized_descriptor(tmp_path, planner, dtype):
    # SplitVectorKernel replay stamps [0, 0] onto local tiles. Preserve it
    # through a slice that has no explicit valid_shape operand.
    program = pl.parse(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, out: pl.Out[pl.Tensor[[32, 128], pl.BF16]]) -> pl.Tensor[[32, 128], pl.BF16]:
        tile: pl.Tile[
            [64, 64], pl.{dtype}, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
        ] = pl.tile.create([64, 64], dtype=pl.{dtype}, target_memory=pl.MemorySpace.Vec)
        window = pl.tile.slice(tile, [32, 64], [32, 0])
        decoded = pl.cast(window, pl.BF16)
        return pl.store(decoded, [0, 0], out)
""")
    mlir = _emit_incore_mlir(program, planner)
    assert "pto.treshape" not in mlir and "pto.tstore" not in mlir, mlir
    descriptor = next(line for line in mlir.splitlines() if "%window__ssa_v0 = pto.alloc_tile" in line)
    assert "valid_row = %c0_index valid_col = %c0_index" in descriptor, descriptor
    if find_ptoas_binary() is not None:
        pto = tmp_path / "empty_fp4_replay.pto"
        cpp = tmp_path / "empty_fp4_replay.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), _get_ptoas_flags(planner))
        assert cpp.is_file()


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize("dtype", ["FP4E2M1X2", "FP4E1M2X2"])
@pytest.mark.parametrize("nested", [False, True])
def test_empty_fp4_slice_rejects_unproven_address(planner, dtype, nested):
    # AllocateMemoryAddr can fold a dynamic slice offset down to its base.
    # A nested slice must not mistake that base for a proven static address.
    first = "base = pl.tile.slice(tile, [32, 64], [start, 0])" if nested else ""
    source = "base" if nested else "tile"
    offset = "0" if nested else "start"
    program = pl.parse(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, src: pl.Tensor[[64, 64], pl.{dtype}], start: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[32, 128], pl.BF16]]) -> pl.Tensor[[32, 128], pl.BF16]:
        tile = pl.load(src, [0, 0], [64, 64])
        {first}
        window = pl.tile.slice({source}, [32, 64], [{offset}, 0], valid_shape=[0, 0])
        decoded = pl.cast(window, pl.BF16)
        return pl.store(decoded, [0, 0], out)
""")
    with pytest.raises(ValueError, match="Empty packed FP4 tile.slice requires .*static"):
        _emit_incore_mlir(program, planner)


def test_fp4e2m1x2_slice_cast_preserves_carrier_offset():
    """An E2M1 carrier slice retains its byte offset before the existing FP8 cast."""

    @pl.program
    class CastProg:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 32], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 32], pl.FP8E4M3FN]:
            return pl.store(pl.cast(pl.load(src, [0, 16], [16, 16]), pl.FP8E4M3FN), [0, 0], out)

    mlir = _emit_incore_mlir(CastProg)
    assert "!pto.f4E2M1x2" in mlir and "pto.tcvt" in mlir
    assert "pto.ttrans" not in mlir
    fp4_load = next(line for line in mlir.splitlines() if "partition_view" in line and "f4E2M1x2" in line)
    assert "offsets = [%c0_index, %c16_index]" in fp4_load, mlir
    assert "sizes = [%c16_index, %c16_index]" in fp4_load, mlir


@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
@pytest.mark.parametrize("byte_dtype", [pl.UINT8, pl.INT8])
@pytest.mark.parametrize("tensor_surface", [True, False])
def test_fp4e2m1x2_byte_reshape_round_trip_consumed(planner, byte_dtype, tensor_surface):
    """The alias chain is consumed by load, cast and store with every planner."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 64], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[8, 256], pl.BF16]],
        ) -> pl.Tensor[[8, 256], pl.BF16]:
            if tensor_surface:
                bytes_view = pl.tensor.reinterpret_view(src, byte_dtype)
                reshaped = pl.tensor.reshape(bytes_view, [8, 128])
                packed = pl.tensor.reinterpret_view(reshaped, pl.FP4E2M1X2)
                cast = pl.tensor.cast(packed, pl.BF16)
                result = pl.tensor.assemble(out, cast, [0, 0])
            else:
                source = pl.load(src, [0, 0], [16, 64])
                bytes_tile = pl.tile.reinterpret_view(source, byte_dtype)
                reshaped_tile = pl.tile.reshape(bytes_tile, [8, 128])
                tile = pl.tile.reinterpret_view(reshaped_tile, pl.FP4E2M1X2)
                result = pl.store(pl.cast(tile, pl.BF16), [0, 0], out)
            return result

    mlir = _emit_incore_mlir(Program, planner)
    assert "pto.tload" in mlir and "pto.tcvt" in mlir and "pto.tstore" in mlir, mlir
    assert "8x128x!pto.f4E2M1x2" in mlir or "rows=8, cols=128" in mlir, mlir


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
def test_fp4_native_vec_copy_rejects_unsafe_byte_strides(dtype):
    with pytest.raises(InvalidOperationError, match="Vec-to-Vec moves support only ND-to-NZ"):

        @pl.program
        class MoveProg:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 32], dtype],
                out: pl.Out[pl.Tensor[[16, 32], dtype]],
            ) -> pl.Tensor[[16, 32], dtype]:
                return pl.store(
                    pl.move(pl.load(src, [0, 0], [16, 32]), target_memory=pl.Mem.Vec), [0, 0], out
                )


@pytest.mark.parametrize(("dtype", "pto_type"), [(pl.fp4e2m1x2, "f4E2M1x2"), (pl.fp4e1m2x2, "f4E1M2x2")])
def test_fp4_make_tensor_view_preserves_carrier_units(dtype, pto_type):
    """Rank-2/3 parameter views and partitions retain their carrier dimensions."""

    @pl.program
    class Rank2:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[2, 256], dtype],
            out: pl.Out[pl.Tensor[[2, 256], dtype]],
        ) -> pl.Tensor[[2, 256], dtype]:
            viewed: pl.Tensor[[2, 256], dtype] = pl.tensor.view(src, [2, 256])
            return pl.store(pl.load(viewed, [0, 0], [2, 256]), [0, 0], out)

    mlir = _emit_incore_mlir(Rank2)
    views = [line for line in mlir.splitlines() if "pto.make_tensor_view" in line and pto_type in line]
    assert views and all("shape = [%c2_index, %c256_index]" in line for line in views), mlir
    assert all("strides = [%c256_index, %c1_index]" in line for line in views), mlir
    partitions = [line for line in mlir.splitlines() if "partition_view" in line]
    assert partitions and all(f"2x256x!pto.{pto_type}" in line for line in partitions), mlir

    @pl.program
    class Rank3:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[2, 16, 32], dtype],
            out: pl.Out[pl.Tensor[[2, 16, 32], dtype]],
        ) -> pl.Tensor[[2, 16, 32], dtype]:
            return pl.store(pl.load(src, [0, 0, 0], [2, 16, 32]), [0, 0, 0], out)

    mlir3 = _emit_incore_mlir(Rank3)
    assert f"!pto.{pto_type}" in mlir3
    views3 = [line for line in mlir3.splitlines() if "pto.make_tensor_view" in line]
    assert views3 and all("shape = [%c2_index, %c16_index, %c32_index]" in line for line in views3), mlir3
    assert "arith.muli %c32_index, %c16_index : index" in mlir3, mlir3
    assert "arith.muli %c32_index, %c2_index : index" not in mlir3, mlir3


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
@pytest.mark.parametrize("carrier_width", [1, 2, 32])
@pytest.mark.parametrize("tile_api", [False, True], ids=["tensor", "tile"])
@pytest.mark.parametrize("enable_buffer_ir", [False, True])
def test_fp4_subload_preserves_row_packing_axis(dtype, carrier_width, tile_api, enable_buffer_ir):
    """A narrow ND window keeps row packing even at one carrier per row."""
    tensor_width = 32
    if tile_api:

        @pl.program
        class TileProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[32, tensor_width], dtype],
                out: pl.Out[pl.Tensor[[32, tensor_width], dtype]],
            ) -> pl.Tensor[[32, tensor_width], dtype]:
                tile = pl.tile.load(src, [0, 0], [32, carrier_width])
                return pl.tile.store(tile, [0, 0], out)

        program = TileProgram
    else:

        @pl.program
        class TensorProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[32, tensor_width], dtype],
                out: pl.Out[pl.Tensor[[32, tensor_width], dtype]],
            ) -> pl.Tensor[[32, tensor_width], dtype]:
                tile = pl.load(src, [0, 0], [32, carrier_width])
                return pl.store(tile, [0, 0], out)

        program = TensorProgram
    mlir = _emit_incore_mlir(program, enable_buffer_ir=enable_buffer_ir)
    allocations = [line for line in mlir.splitlines() if "pto.alloc_tile" in line]
    assert allocations, mlir
    assert all(f"rows=32, cols={carrier_width}" in line for line in allocations), mlir
    assert all("blayout=row_major" in line for line in allocations), mlir


@pytest.mark.parametrize(("dtype", "pto_type"), [(pl.fp4e2m1x2, "f4E2M1x2"), (pl.fp4e1m2x2, "f4E1M2x2")])
def test_fp4_cacheinvalid_preserves_carrier_partition(dtype, pto_type):
    """system.cacheinvalid preserves the requested byte-carrier window."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[16, 32], dtype],
            out: pl.Out[pl.Tensor[[16, 32], dtype]],
        ) -> pl.Tensor[[16, 32], dtype]:
            updated: pl.Tensor[[16, 32], dtype] = pl.store(pl.load(x, [0, 0], [16, 32]), [0, 0], out)
            pl.system.cacheinvalid(updated, [16, 16], [0, 0])
            return updated

    mlir = _emit_incore_mlir(Prog)
    cmo_lines = [line for line in mlir.splitlines() if "pto.cmo.cacheinvalid" in line]
    assert len(cmo_lines) == 1, mlir
    assert f"16x16x!pto.{pto_type}" in cmo_lines[0], cmo_lines[0]


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
def test_fp4_nz_assemble_uses_native_insert(dtype, tmp_path):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV)
        def fp4_nz_insert(
            self,
            original: pl.Tensor[[64, 128], dtype],
            update: pl.Tensor[[32, 64], dtype],
        ):
            original_nd = pl.tile.load(original, [0, 0], [64, 128])
            original_nz = pl.tile.move(
                original_nd,
                target_memory=pl.Mem.Vec,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            target = pl.tile.move(original_nz, target_memory=pl.Mem.Mat)
            update_nd = pl.tile.load(update, [0, 0], [32, 64])
            update_nz = pl.tile.move(
                update_nd,
                target_memory=pl.Mem.Vec,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            assembled = pl.tile.assemble(target, update_nz, [0, 64])
            pl.tile.set_validshape(assembled, 32, 256)

    mlir = _emit_incore_mlir(Program)
    inserts = [line for line in mlir.splitlines() if "pto.tinsert " in line]
    assert inserts, mlir
    assert any(
        "rows=16, cols=128" in line and "rows=64, cols=256" in line and "%c64_index" in line
        for line in inserts
    ), mlir
    # The Vec source still counts carriers; the Mat target uses PTOAS v0.72
    # logical dimensions. Verify that the cross-space transfer also assembles.
    if find_ptoas_binary() is not None:
        source = tmp_path / "fp4_insert.pto"
        source.write_text(mlir)
        _run_ptoas(str(source), str(tmp_path / "fp4_insert.cpp"), ["--pto-arch=a5", "--pto-level=level3"])


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
def test_fp4_nz_assemble_rejects_automatic_mixed_expansion(dtype):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def fp4_nz_insert(
            self,
            original: pl.Tensor[[64, 128], dtype],
            update: pl.Tensor[[32, 64], dtype],
        ):
            # Direct GM-to-Mat loading isolates the insert guard: no earlier
            # Vec-to-Mat move can reject the function before the insertion.
            target = pl.tile.load(original, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
            update_nd = pl.tile.load(update, [0, 0], [32, 64])
            update_nz = pl.tile.move(
                update_nd,
                target_memory=pl.Mem.Vec,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            assembled = pl.tile.assemble(target, update_nz, [0, 64])
            pl.tile.set_validshape(assembled, 32, 256)

    with pytest.raises(ValueError, match="use an explicit FunctionType.AIV"):
        _emit_incore_mlir(Program)


def _dense_dn_fp4_conversion_program(source, target):
    @pl.program
    class ConvertDn:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[64, 32], source, pl.TensorView(stride=[1, 64], layout=pl.TensorLayout.DN)],
            out: pl.Out[
                pl.Tensor[[32, 32], target, pl.TensorView(stride=[1, 32], layout=pl.TensorLayout.DN)]
            ],
        ) -> pl.Tensor[[32, 32], target, pl.TensorView(stride=[1, 32], layout=pl.TensorLayout.DN)]:
            # Nonzero GM offsets do not change the freshly loaded UB tile's
            # zero origin. The packed-to-packed cast must retain DN through BF16.
            packed = pl.tile.load(src, [32, 0], [32, 32])
            return pl.tile.store(pl.cast(packed, target), [0, 0], out)

    return ConvertDn


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize(("source", "target"), [(pl.fp4e2m1x2, pl.fp4e1m2x2), (pl.fp4e1m2x2, pl.fp4e2m1x2)])
def test_dense_dn_fp4_cast_uses_static_valid_descriptors(source, target, planner):
    mlir = _emit_incore_mlir(_dense_dn_fp4_conversion_program(source, target), planner)
    casts = [line for line in mlir.splitlines() if "pto.tcvt " in line]
    assert len(casts) == 2, mlir
    for cast in casts:
        assert "blayout=col_major" in cast, cast
        assert "dtype=bf16, rows=64, cols=32, v_row=64, v_col=32" in cast, cast
        assert "rows=32, cols=32, v_row=32, v_col=32" in cast, cast
        assert "v_row=?" not in cast and "v_col=?" not in cast, cast
    # Dynamic-valid DN TCVT faults on A5. Both packed operands must be given
    # addressed static aliases because PTOAS cannot treshape custom FP4 types.
    packed_views = [
        line
        for line in mlir.splitlines()
        if "pto.alloc_tile" in line and "f4" in line and "v_row=32, v_col=32" in line
    ]
    assert len(packed_views) == 2, mlir
    assert all("addr = " in line for line in packed_views), mlir
    assert "pto.tmov " not in mlir and "pto.ttrans " not in mlir, mlir


@pytest.mark.parametrize(("source", "target"), [(pl.fp4e2m1x2, pl.fp4e1m2x2), (pl.fp4e1m2x2, pl.fp4e2m1x2)])
def test_dense_dn_fp4_cast_requires_addressed_planner(source, target):
    with pytest.raises(ValueError, match="Packed FP4 DN cast requires static PYPTO or DSA_RP"):
        _emit_incore_mlir(_dense_dn_fp4_conversion_program(source, target), passes.MemoryPlanner.PTOAS)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
@pytest.mark.parametrize("direction", ["decode", "encode", "cross"])
def test_dense_dn_tensor_cast_preserves_layout_through_default_pipeline(dtype, direction, planner):
    source = pl.BF16 if direction == "encode" else dtype
    other = pl.fp4e1m2x2 if dtype == pl.fp4e2m1x2 else pl.fp4e2m1x2
    target = pl.BF16 if direction == "decode" else other if direction == "cross" else dtype
    source_rows = 64 if source == pl.BF16 else 32
    target_rows = 64 if target == pl.BF16 else 32

    @pl.program
    class TensorCastDn:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[
                [source_rows, 32], source, pl.TensorView(stride=[1, source_rows], layout=pl.TensorLayout.DN)
            ],
            out: pl.Out[
                pl.Tensor[
                    [target_rows, 32],
                    target,
                    pl.TensorView(stride=[1, target_rows], layout=pl.TensorLayout.DN),
                ]
            ],
        ) -> pl.Tensor[
            [target_rows, 32], target, pl.TensorView(stride=[1, target_rows], layout=pl.TensorLayout.DN)
        ]:
            converted = pl.tensor.cast(src, target, mode="rint")
            return pl.tensor.assemble(out, converted, [0, 0])

    mlir = _emit_incore_mlir(TensorCastDn, planner)
    assert mlir.count("pto.tload ") == mlir.count("pto.tstore ") == 1, mlir
    casts = [line for line in mlir.splitlines() if "pto.tcvt " in line]
    assert len(casts) == (2 if direction == "cross" else 1), mlir
    for cast in casts:
        assert cast.count("blayout=col_major") == 2, cast
        assert "dtype=bf16, rows=64, cols=32, v_row=64, v_col=32" in cast, cast
        assert "v_row=?" not in cast and "v_col=?" not in cast, cast
    # Encoding loads the contiguous ND storage and flips only its metadata.
    # Neither path is allowed to materialize a numerical transpose or move.
    assert "pto.ttrans " not in mlir and "pto.tmov " not in mlir, mlir


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
@pytest.mark.parametrize("direction", ["decode", "encode", "cross"])
def test_dense_dn_tensor_slice_cast_preserves_packing_axis(tmp_path, dtype, direction, planner):
    source = pl.BF16 if direction == "encode" else dtype
    other = pl.fp4e1m2x2 if dtype == pl.fp4e2m1x2 else pl.fp4e2m1x2
    target = pl.BF16 if direction == "decode" else other if direction == "cross" else dtype
    source_rows = 128 if source == pl.BF16 else 64
    target_rows = 128 if target == pl.BF16 else 64

    @pl.program
    class SliceCastDn:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[
                [source_rows, 128], source, pl.TensorView(stride=[1, source_rows], layout=pl.TensorLayout.DN)
            ],
            out: pl.Out[
                pl.Tensor[
                    [target_rows, 64],
                    target,
                    pl.TensorView(stride=[1, target_rows], layout=pl.TensorLayout.DN),
                ]
            ],
        ) -> pl.Tensor[
            [target_rows, 64], target, pl.TensorView(stride=[1, target_rows], layout=pl.TensorLayout.DN)
        ]:
            selected = pl.tensor.slice(src, [source_rows, 64], [0, 64])
            converted = pl.tensor.cast(selected, target, mode="rint")
            return pl.tensor.assemble(out, converted, [0, 0])

    mlir = _emit_incore_mlir(SliceCastDn, planner)
    assert mlir.count("pto.tload ") == 1, mlir
    casts = [line for line in mlir.splitlines() if "pto.tcvt " in line]
    assert len(casts) == (2 if direction == "cross" else 1), mlir
    for cast in casts:
        assert cast.count("blayout=col_major") == 2, cast
        assert "dtype=bf16, rows=128, cols=64, v_row=128, v_col=64" in cast, cast
        assert "rows=64, cols=64, v_row=64, v_col=64" in cast, cast
        assert "v_row=?" not in cast and "v_col=?" not in cast, cast
    # A BF16 slice is already materialized as row-major values. Encoding must
    # transpose that snapshot before viewing it as the original DN matrix.
    assert mlir.count("pto.ttrans ") == (1 if direction == "encode" else 0), mlir
    assert "pto.tmov " not in mlir, mlir
    if find_ptoas_binary() is not None:
        pto = tmp_path / "slice_cast_dn.pto"
        cpp = tmp_path / "slice_cast_dn.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", "--pto-level=level3", "--pto-arch=a5"])
        assert cpp.is_file()


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
def test_dense_dn_tensor_slice_cast_preserves_other_consumer_layout(tmp_path, dtype, planner):
    @pl.program
    class SharedSlice:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[128, 128], pl.BF16, pl.TensorView(stride=[1, 128], layout=pl.TensorLayout.DN)],
            out: pl.Out[pl.Tensor[[64, 64], dtype, pl.TensorView(stride=[1, 64], layout=pl.TensorLayout.DN)]],
            sums: pl.Out[pl.Tensor[[128, 64], pl.BF16]],
        ) -> tuple[
            pl.Tensor[[64, 64], dtype, pl.TensorView(stride=[1, 64], layout=pl.TensorLayout.DN)],
            pl.Tensor[[128, 64], pl.BF16],
        ]:
            selected = pl.tensor.slice(src, [128, 64], [0, 64])
            added = pl.tensor.add(selected, selected)
            converted = pl.tensor.cast(selected, dtype, mode="rint")
            return pl.tensor.assemble(out, converted, [0, 0]), pl.tensor.assemble(sums, added, [0, 0])

    mlir = _emit_incore_mlir(SharedSlice, planner)
    # Both consumers must use the same snapshot, without reloading its GM source.
    assert mlir.count("pto.tload ") == 1, mlir
    cast = next(line for line in mlir.splitlines() if "pto.tcvt " in line)
    assert cast.count("blayout=col_major") == 2, cast
    assert "dtype=bf16, rows=128, cols=64, v_row=128, v_col=64" in cast, cast
    assert "rows=64, cols=64, v_row=64, v_col=64" in cast, cast
    assert mlir.count("pto.ttrans ") == 1, mlir
    addition = next(line for line in mlir.splitlines() if "pto.tadd " in line)
    assert "blayout=col_major" not in addition, addition
    assert addition.count("blayout=row_major") == 3, addition
    if find_ptoas_binary() is not None:
        pto = tmp_path / "shared_slice_cast_dn.pto"
        cpp = tmp_path / "shared_slice_cast_dn.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", "--pto-level=level3", "--pto-arch=a5"])
        assert cpp.is_file()


def _direct_fp4_to_fp8_program(layout=None, *, partial=False, carrier_cols=128):
    output_layout = pl.NZ if layout == pl.NZ else pl.ND
    source_valid = [17, carrier_cols - 8] if partial else [32, carrier_cols]
    observation_valid = [32, carrier_cols * 2] if layout == pl.NZ else [source_valid[0], source_valid[1] * 2]

    @pl.program
    class DirectDecode:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[32, carrier_cols], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[32, carrier_cols * 2], pl.FP8E4M3FN, output_layout]],
        ) -> pl.Tensor[[32, carrier_cols * 2], pl.FP8E4M3FN, output_layout]:
            packed = pl.tile.load(src, [0, 0], [32, carrier_cols], valid_shape=source_valid)
            decoded = pl.cast(packed, pl.FP8E4M3FN, layout=layout)
            # NZ GM stores transfer complete 16x32 boxes. Only the observation
            # alias is widened after decoding. The codegen test does not assert
            # values outside the decoded valid region.
            observed = pl.tile.set_validshape(decoded, observation_valid[0], observation_valid[1])
            return pl.tile.store(observed, [0, 0], out)

    return DirectDecode


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("layout", [None, pl.ND, pl.NZ])
def test_direct_fp4_to_fp8_cast_emits_one_fused_tcvt(tmp_path, planner, partial, layout):
    # Before this path, casting generated BF16 and FP32 temporaries and a
    # separate ND-to-NZ move. The destination descriptor must reach TCVT itself.
    mlir = _emit_incore_mlir(_direct_fp4_to_fp8_program(layout, partial=partial), planner)
    casts = [line for line in mlir.splitlines() if "pto.tcvt " in line]
    assert len(casts) == 1, mlir
    cast = casts[0]
    assert "dtype=!pto.f4E2M1x2, rows=32, cols=128" in cast, cast
    assert "dtype=f8E4M3FN, rows=32, cols=256" in cast, cast
    assert "dtype=bf16" not in mlir and "dtype=f32" not in mlir, mlir
    assert "pto.tmov " not in mlir, mlir
    # FP4 and FP8 both report eight storage bits, but the decoder expands two
    # values per carrier. Automatic reuse must not alias this numerical widening.
    source = re.search(r"ins\((%[\w.$-]+)", cast)
    target = re.search(r"outs\((%[\w.$-]+)", cast)
    assert source and target, cast
    allocations = {
        match.group(1): match.group(2)
        for line in mlir.splitlines()
        if (match := re.search(r"(%[\w.$-]+) = pto.alloc_tile addr = (%[\w.$-]+)", line))
    }
    assert allocations[source.group(1)] != allocations[target.group(1)], mlir
    if layout == pl.NZ:
        assert "blayout=col_major, slayout=row_major" in cast, cast
        fp8_allocations = [
            line for line in mlir.splitlines() if "pto.alloc_tile" in line and "f8E4M3FN" in line
        ]
        assert fp8_allocations and all("slayout=row_major" in line for line in fp8_allocations), mlir
    if find_ptoas_binary() is not None:
        pto = tmp_path / "direct_fp4_to_fp8.pto"
        cpp = tmp_path / "direct_fp4_to_fp8.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", "--pto-level=level3", "--pto-arch=a5"])
        assert cpp.read_text().count("TCVT(") == 1


def test_direct_fp4_to_fp8_nz_cast_rejects_unaligned_physical_source():
    # A valid rectangle can be narrow; the physical ND row must still be one
    # complete 32-byte source block for the ISA's residual-block load.
    with pytest.raises(ValueError, match="physical carrier columns divisible by 32"):
        _emit_incore_mlir(_direct_fp4_to_fp8_program(layout=pl.NZ, carrier_cols=16))


@pytest.mark.parametrize(
    ("source", "target", "layout", "error"),
    [
        (pl.FP4E1M2X2, pl.FP8E4M3FN, pl.NZ, "layout=NZ supports only"),
        (pl.FP4E2M1X2, pl.FP8E5M2, pl.NZ, "layout=NZ supports only"),
        (pl.FP4E2M1X2, pl.FP8E4M3FN, pl.TensorLayout.DN, "layout must be ND or NZ"),
    ],
)
def test_fp4_to_fp8_cast_rejects_unsupported_output_options(source, target, layout, error):
    value = ir.Var("source", ir.TileType([32, 128], source), ir.Span.unknown())
    with pytest.raises(ValueError, match=error):
        ir.op.tile.cast(value, target, layout=layout)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_direct_fp4_to_fp8_named_raw_nz_slots_remain_disjoint(tmp_path, planner):
    raw_slots = pl.MemRef("RAW", slots=2)
    nz_slots = pl.MemRef("NZ", slots=2)

    @pl.program
    class NamedDecode:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            src: pl.Tensor[[128, 128], pl.UINT8],
            out: pl.Out[pl.Tensor[[128, 256], pl.FP8E4M3FN, pl.NZ]],
        ) -> pl.Tensor[[128, 256], pl.FP8E4M3FN, pl.NZ]:
            raw: pl.Tile[[128, 128], pl.UINT8, raw_slots[1], pl.Mem.Vec] = pl.load(src, [0, 0], [128, 128])
            decoded: pl.Tile[
                [128, 256],
                pl.FP8E4M3FN,
                nz_slots[1],
                pl.Mem.Vec,
                pl.TileView(
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                ),
            ] = pl.cast(
                pl.reinterpret_view(raw, pl.FP4E2M1X2),
                pl.FP8E4M3FN,
                layout=pl.NZ,
            )
            return pl.store(decoded, [0, 0], out)

    mlir = _emit_incore_mlir(NamedDecode, planner)
    cast = next(line for line in mlir.splitlines() if "pto.tcvt " in line)
    assert mlir.count("pto.tcvt ") == 1 and "pto.tmov " not in mlir, mlir
    # Canonical tile buffer types omit the default (null) compact mode.
    assert "compact=" not in cast, cast
    assert "blayout=col_major, slayout=row_major" in cast, cast
    source = re.search(r"ins\((%[\w.$-]+)", cast)
    target = re.search(r"outs\((%[\w.$-]+)", cast)
    assert source and target, cast
    allocations = {
        match.group(1): match.group(2)
        for line in mlir.splitlines()
        if (match := re.search(r"(%[\w.$-]+) = pto.alloc_tile addr = (%[\w.$-]+)", line))
    }
    constants = {
        match.group(1): int(match.group(2))
        for line in mlir.splitlines()
        if (match := re.search(r"(%[\w.$-]+) = arith.constant (-?\d+) : i64", line))
    }
    source_address = constants[allocations[source.group(1)]]
    target_address = constants[allocations[target.group(1)]]
    # Check the actual byte windows independently of multi-buffer SSA naming.
    # One FP4 carrier expands into two FP8 bytes; neither window may overlap.
    assert source_address + 128 * 128 <= target_address or target_address + 128 * 256 <= source_address
    if find_ptoas_binary() is not None:
        pto = tmp_path / "named_direct_fp4_to_fp8.pto"
        cpp = tmp_path / "named_direct_fp4_to_fp8.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", "--pto-level=level3", "--pto-arch=a5"])
        assert cpp.read_text().count("TCVT(") == 1


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_narrow_fp4_to_fp8_nd_cast_retains_existing_legalization(tmp_path, planner):
    mlir = _emit_incore_mlir(_direct_fp4_to_fp8_program(carrier_cols=16), planner)
    # The fused decoder reads full 32-byte source blocks; a narrower physical
    # row keeps its existing numerical chain instead of acquiring a new error.
    assert mlir.count("pto.tcvt ") == 3, mlir
    assert "dtype=bf16" in mlir and "dtype=f32" in mlir, mlir
    if find_ptoas_binary() is not None:
        pto = tmp_path / "narrow_fp4_to_fp8.pto"
        cpp = tmp_path / "narrow_fp4_to_fp8.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", "--pto-level=level3", "--pto-arch=a5"])
        assert cpp.read_text().count("TCVT(") == 3


@pytest.mark.parametrize("destination_offset", [0, 32, 4096])
def test_direct_fp4_to_fp8_rejects_explicitly_overlapping_memrefs(destination_offset):
    span = ir.Span.unknown()
    source_shape, result_shape = [32, 128], [32, 256]
    source_base = ir.Var("source_base", ir.PtrType(), span)
    result_base = ir.Var("result_base", ir.PtrType(), span)
    source_type = ir.TileType(
        source_shape,
        pl.FP4E2M1X2,
        ir.MemRef(source_base, ir.ConstInt(0, pl.INT64, span), 4096, span),
        None,
        ir.MemorySpace.Vec,
    )
    result_type = ir.TileType(
        result_shape,
        pl.FP8E4M3FN,
        ir.MemRef(result_base, ir.ConstInt(destination_offset, pl.INT64, span), 8192, span),
        None,
        ir.MemorySpace.Vec,
    )
    src = ir.Var("src", ir.TensorType(source_shape, pl.FP4E2M1X2), span)
    out = ir.Var("out", ir.TensorType(result_shape, pl.FP8E4M3FN), span)
    source, result = ir.Var("source", source_type, span), ir.Var("result", result_type, span)
    output = ir.Var("output", out.type, span)
    offsets = ir.MakeTuple([ir.ConstInt(0, pl.INDEX, span)] * 2, span)
    sizes = ir.MakeTuple([ir.ConstInt(n, pl.INDEX, span) for n in source_shape], span)
    load = ir.Call(ir.get_op("tile.load"), [src, offsets, sizes], {}, source_type, span)
    cast = ir.Call(
        ir.get_op("tile.cast"), [source], {"target_type": pl.FP8E4M3FN, "mode": 1}, result_type, span
    )
    store = ir.Call(ir.get_op("tile.store"), [result, offsets, out], out.type, span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(source, load, span),
            ir.AssignStmt(result, cast, span),
            ir.AssignStmt(output, store, span),
            ir.ReturnStmt([output], span),
        ],
        span,
    )
    function = ir.Function(
        "direct_cast_overlap",
        [(src, ir.ParamDirection.In), (out, ir.ParamDirection.Out)],
        [out.type],
        body,
        span,
        ir.FunctionType.AIV,
    )
    reset_for_testing()
    set_backend_type(BackendType.Ascend950)
    program = ir.Program([function], "direct_cast_overlap", span)
    if destination_offset < 4096:
        with pytest.raises(ValueError, match="tile.cast.*requires non-overlapping"):
            codegen.PTOCodegen().generate(program)
    else:
        assert "pto.tcvt" in codegen.PTOCodegen().generate(program)


def test_rank_one_fp4_to_fp8_direct_cast_keeps_nd_vector_support(tmp_path):
    @pl.program
    class VectorDecode:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            src: pl.Tensor[[128], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[1, 256], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[1, 256], pl.FP8E4M3FN]:
            packed = pl.load(src, [0], [128])
            decoded = pl.cast(packed, pl.FP8E4M3FN)
            observed = pl.reshape(decoded, [1, 256])
            return pl.store(observed, [0, 0], out)

    mlir = _emit_incore_mlir(VectorDecode)
    casts = [line for line in mlir.splitlines() if "pto.tcvt " in line]
    assert len(casts) == 1, mlir
    assert "dtype=!pto.f4E2M1x2, rows=1, cols=128" in casts[0]
    assert "dtype=f8E4M3FN, rows=1, cols=256" in casts[0]
    if find_ptoas_binary() is not None:
        pto = tmp_path / "vector_fp4_to_fp8.pto"
        cpp = tmp_path / "vector_fp4_to_fp8.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", "--pto-level=level3", "--pto-arch=a5"])
        assert cpp.read_text().count("TCVT(") == 1


def _vec_fp8_nz_store_ir(*, source_dtype=pl.FP8E4M3FN, nz_source=True, valid_rows=32, options=None):
    span = ir.Span.unknown()
    source_view = ir.TileView(
        valid_shape=[valid_rows, 256],
        blayout=ir.TileLayout.col_major if nz_source else ir.TileLayout.row_major,
        slayout=ir.TileLayout.row_major if nz_source else ir.TileLayout.none_box,
    )
    source = ir.Var(
        "source", ir.TileType([32, 256], source_dtype, None, source_view, ir.MemorySpace.Vec), span
    )
    output = ir.Var(
        "output",
        ir.TensorType([32, 256], pl.FP8E4M3FN, tensor_view=ir.TensorView(layout=ir.TensorLayout.NZ)),
        span,
    )
    offsets = ir.MakeTuple([ir.ConstInt(0, pl.INDEX, span)] * 2, span)
    # Explicit malformed kwargs exercise backend validation independently of
    # the frontend's earlier atomic/epilogue dtype checks.
    store = ir.Call(ir.get_op("tile.store"), [source, offsets, output], options or {}, output.type, span)
    function = ir.Function(
        "vec_nz_store",
        [(source, ir.ParamDirection.In), (output, ir.ParamDirection.Out)],
        [],
        ir.EvalStmt(store, span),
        span,
        ir.FunctionType.AIV,
    )
    return ir.Program([function], "vec_nz_store", span)


@pytest.mark.parametrize(
    ("options", "valid_rows", "nz_source", "source_dtype", "error"),
    [
        ({"atomic": 1}, 32, True, pl.FP8E4M3FN, "requires a plain store"),
        ({"pre_quant": 1.0}, 32, True, pl.FP8E4M3FN, "requires a plain store"),
        ({"pre_relu": True}, 32, True, pl.FP8E4M3FN, "requires a plain store"),
        ({"st_phase": 1}, 32, True, pl.FP8E4M3FN, "requires a plain store"),
        ({}, 17, True, pl.FP8E4M3FN, "complete 16x32 boxes"),
        ({}, 32, False, pl.FP8E4M3FN, "requires a standard 512-byte NZ tile"),
        ({}, 32, True, pl.FP8E5M2, "supports only FP8E4M3FN"),
    ],
)
def test_vec_fp8_nz_store_rejects_unsupported_modes(options, valid_rows, nz_source, source_dtype, error):
    reset_for_testing()
    set_backend_type(BackendType.Ascend950)
    program = _vec_fp8_nz_store_ir(
        source_dtype=source_dtype, nz_source=nz_source, valid_rows=valid_rows, options=options
    )
    with pytest.raises(ValueError, match=error):
        passes.block_nz_tensor_views()(program)


@pytest.mark.parametrize("backend_type", [BackendType.Ascend910, BackendType.Ascend950])
def test_vec_fp8_nz_store_backend_capability_is_independent_of_acc_store(backend_type):
    from pypto import backend  # noqa: PLC0415

    reset_for_testing()
    set_backend_type(backend_type)
    assert backend.get_handler().supports_vec_fp8_nz_gm_store() == (backend_type == BackendType.Ascend950)
    program = _vec_fp8_nz_store_ir()
    if backend_type == BackendType.Ascend950:
        assert "[1, 8, 2, 16, 32]" in passes.block_nz_tensor_views()(program).as_python()
    else:
        with pytest.raises(ValueError, match="requires the A5 FP8 NZ store capability"):
            passes.block_nz_tensor_views()(program)


def test_direct_fp4_to_fp8_keeps_runtime_rows_with_static_columns(tmp_path):
    @pl.program
    class DynamicRows:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            src: pl.Tensor[[32, 128], pl.FP4E2M1X2],
            rows: pl.Tensor[[1], pl.INT64],
            out: pl.Out[pl.Tensor[[32, 256], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[32, 256], pl.FP8E4M3FN]:
            valid_rows = pl.tensor.read(rows, [0])
            packed = pl.load(src, [0, 0], [32, 128], valid_shape=[valid_rows, 128])
            decoded = pl.cast(packed, pl.FP8E4M3FN)
            # ND has a fixed physical row pitch. Observe the whole envelope
            # after conversion; only the compile-time descriptor is checked.
            observed = pl.tile.set_validshape(decoded, 32, 256)
            return pl.store(observed, [0, 0], out)

    mlir = _emit_incore_mlir(DynamicRows)
    casts = [line for line in mlir.splitlines() if "pto.tcvt " in line]
    assert len(casts) == 1, mlir
    assert "v_row=?, v_col=128" in casts[0] and "v_row=?, v_col=256" in casts[0], casts[0]
    if find_ptoas_binary() is not None:
        pto = tmp_path / "dynamic_rows_fp4_to_fp8.pto"
        cpp = tmp_path / "dynamic_rows_fp4_to_fp8.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", "--pto-level=level3", "--pto-arch=a5"])
        assert cpp.read_text().count("TCVT(") == 1


def test_dynamic_valid_columns_keep_existing_fp4_cast_bridge():
    @pl.program
    class DynamicColumns:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            src: pl.Tensor[[32, 128], pl.FP4E2M1X2],
            columns: pl.Tensor[[1], pl.INT64],
            out: pl.Out[pl.Tensor[[32, 256], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[32, 256], pl.FP8E4M3FN]:
            valid_columns = pl.tensor.read(columns, [0])
            packed = pl.load(src, [0, 0], [32, 128], valid_shape=[32, valid_columns])
            return pl.store(pl.cast(packed, pl.FP8E4M3FN), [0, 0], out)

    mlir = _emit_incore_mlir(DynamicColumns)
    assert mlir.count("pto.tcvt ") == 3, mlir
    assert "dtype=bf16" in mlir and "dtype=f32" in mlir, mlir


def test_ptoas_memory_planner_keeps_existing_fp4_to_fp8_nd_bridge(tmp_path):
    mlir = _emit_incore_mlir(_direct_fp4_to_fp8_program(), passes.MemoryPlanner.PTOAS)
    assert mlir.count("pto.tcvt ") == 3, mlir
    if find_ptoas_binary() is not None:
        pto = tmp_path / "planned_fp4_to_fp8_bridge.pto"
        cpp = tmp_path / "planned_fp4_to_fp8_bridge.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", "--pto-level=level2", "--pto-arch=a5"])
        assert cpp.read_text().count("TCVT(") == 3


def test_direct_fp4_to_fp8_nz_requires_addressed_memory_planner():
    with pytest.raises(ValueError, match="require addressed PYPTO or DSA_RP memory planning"):
        _emit_incore_mlir(_direct_fp4_to_fp8_program(layout=pl.NZ), passes.MemoryPlanner.PTOAS)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_direct_fp4_to_fp8_static_columns_preserve_runtime_slot_address(tmp_path, planner):
    raw_slots = pl.MemRef("RAW", slots=2)
    nz_slots = pl.MemRef("NZ", slots=2)

    @pl.program
    class RuntimeSlots:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            src: pl.Tensor[[64, 128], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[64, 256], pl.FP8E4M3FN, pl.NZ]],
        ) -> pl.Tensor[[64, 256], pl.FP8E4M3FN, pl.NZ]:
            # A packed Tensor accepts UINT8 carriers without changing bytes.
            # Load that declared format directly: UB byte reinterpretation has
            # a separate existing static-offset restriction.
            for packet in pl.range(2):
                raw: pl.Tile[[32, 128], pl.FP4E2M1X2, raw_slots[packet], pl.Mem.Vec] = pl.load(
                    src, [packet * 32, 0], [32, 128]
                )
                decoded: pl.Tile[
                    [32, 256],
                    pl.FP8E4M3FN,
                    nz_slots[packet],
                    pl.Mem.Vec,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.cast(raw, pl.FP8E4M3FN, layout=pl.NZ)
                out = pl.store(decoded, [packet * 32, 0], out)
            return out

    mlir = _emit_incore_mlir(RuntimeSlots, planner)
    cast = next(line for line in mlir.splitlines() if "pto.tcvt " in line)
    source = re.search(r"ins\((%[\w.$-]+)", cast)
    assert source, cast
    source_allocation = next(
        line for line in mlir.splitlines() if line.strip().startswith(source.group(1) + " = ")
    )
    address = re.search(r"addr = (%[\w.$-]+)", source_allocation)
    constants = {
        match.group(1)
        for line in mlir.splitlines()
        if (match := re.search(r"(%[\w.$-]+) = arith.constant", line))
    }
    assert address and address.group(1) not in constants, mlir
    assert "v_col=128" in cast and "v_col=256" in cast, cast
    if find_ptoas_binary() is not None:
        pto = tmp_path / "runtime_slot_fp4_to_fp8.pto"
        cpp = tmp_path / "runtime_slot_fp4_to_fp8.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", "--pto-level=level3", "--pto-arch=a5"])
        assert "TCVT(" in cpp.read_text()


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize("dynamic_offset", [False, True])
def test_direct_fp4_to_fp8_slice_preserves_parent_pitch_and_offset(tmp_path, planner, dynamic_offset):
    offset_scale, offset_base = (1, 0) if dynamic_offset else (0, 8)

    @pl.program
    class SliceDecode:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            src: pl.Tensor[[32, 128], pl.FP4E2M1X2],
            start: pl.Scalar[pl.INT64],
            out: pl.Out[pl.Tensor[[16, 128], pl.FP8E4M3FN, pl.NZ]],
        ) -> pl.Tensor[[16, 128], pl.FP8E4M3FN, pl.NZ]:
            packed = pl.load(src, [0, 0], [32, 128])
            row = start * offset_scale + offset_base
            window = pl.tile.slice(packed, [16, 64], [row, 32], valid_shape=[13, 56])
            decoded = pl.cast(window, pl.FP8E4M3FN, layout=pl.NZ)
            observed = pl.tile.set_validshape(decoded, 16, 128)
            return pl.store(observed, [0, 0], out)

    mlir = _emit_incore_mlir(SliceDecode, planner)
    cast = next(line for line in mlir.splitlines() if "pto.tcvt " in line)
    assert mlir.count("pto.tcvt ") == 1 and "dtype=bf16" not in mlir and "dtype=f32" not in mlir, mlir
    source = re.search(r"ins\((%[\w.$-]+)", cast)
    assert source, cast
    source_view = next(line for line in mlir.splitlines() if line.strip().startswith(source.group(1) + " = "))
    parent = re.search(r"= pto.subview (%[\w.$-]+)\[", source_view)
    assert parent, source_view
    parent_view = next(line for line in mlir.splitlines() if line.strip().startswith(parent.group(1) + " = "))
    offsets = re.search(r"= pto.subview %[\w.$-]+\[(%[\w.$-]+), (%[\w.$-]+)\]", parent_view)
    assert offsets and "valid [" in parent_view, parent_view
    constants = {
        match.group(1)
        for line in mlir.splitlines()
        if (match := re.search(r"(%[\w.$-]+) = arith.constant", line))
    }
    assert (offsets.group(1) not in constants) == dynamic_offset, mlir
    assert "v_col=56" in cast and "v_col=112" in cast, cast
    if find_ptoas_binary() is not None:
        pto, cpp = tmp_path / "slice_fp4_to_fp8.pto", tmp_path / "slice_fp4_to_fp8.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", "--pto-level=level3", "--pto-arch=a5"])
        emitted = cpp.read_text()
        assert emitted.count("TCVT(") == 1, emitted
        # The actual C++ source descriptor must retain the parent's 128-byte row
        # pitch (256 logical FP4 values), even though the selected window is 64 bytes.
        source_name = re.search(r"TCVT\([^,]+,\s*([A-Za-z_]\w*)\s*,", emitted)
        assert source_name, emitted
        assert re.search(
            r"Tile<TileType::Vec, float4_e2m1x2_t, (?:16|32), 256, BLayout::RowMajor, 13, 112,[^>]+>\s*"
            + re.escape(source_name.group(1))
            + r"\s*[;({]",
            emitted,
        ), emitted


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize("dynamic_column", [False, True])
def test_direct_fp4_to_fp8_slice_requires_aligned_source(planner, dynamic_column):
    column_scale, column_base = (1, 0) if dynamic_column else (0, 1)

    @pl.program
    class UnalignedSliceDecode:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            src: pl.Tensor[[32, 512], pl.FP4E2M1X2],
            column: pl.Scalar[pl.INT64],
            out: pl.Out[pl.Tensor[[16, 512], pl.FP8E4M3FN, pl.NZ]],
        ) -> pl.Tensor[[16, 512], pl.FP8E4M3FN, pl.NZ]:
            packed = pl.load(src, [0, 0], [32, 512])
            start = column * column_scale + column_base
            window = pl.tile.slice(packed, [16, 256], [0, start], valid_shape=[16, 256])
            return pl.store(pl.cast(window, pl.FP8E4M3FN, layout=pl.NZ), [0, 0], out)

    # NORM register loads use the source window's actual UB address. A known
    # one-byte offset and an unknown runtime column are both unsafe to emit.
    with pytest.raises(ValueError, match="32-byte-aligned source subview"):
        _emit_incore_mlir(UnalignedSliceDecode, planner)


def test_standalone_left_right_fp4_nd_cast_keeps_bridge_after_split(tmp_path):
    @pl.program
    class StandaloneSplitDecode:
        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def main(
            self,
            src: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            packed = pl.load(src, [0, 0], [16, 32])
            return pl.store(pl.cast(packed, pl.FP8E4M3FN), [0, 0], out)

    mlir = _emit_incore_mlir(StandaloneSplitDecode)
    assert mlir.count("pto.tcvt ") == 3, mlir
    assert "dtype=!pto.f4E2M1x2, rows=16, cols=16" in mlir, mlir
    assert "dtype=bf16" in mlir and "dtype=f32" in mlir, mlir
    if find_ptoas_binary() is not None:
        pto, cpp = tmp_path / "split_nd_bridge.pto", tmp_path / "split_nd_bridge.cpp"
        pto.write_text(mlir)
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", "--pto-level=level3", "--pto-arch=a5"])
        assert cpp.read_text().count("TCVT(") == 3


@pytest.mark.parametrize("dynamic_column", [False, True])
@pytest.mark.parametrize("planner", list(passes.MemoryPlanner))
def test_plain_nd_fp4_slice_keeps_bridge_without_direct_alignment_error(tmp_path, dynamic_column, planner):
    column_scale, column_base = (1, 0) if dynamic_column else (0, 1)

    @pl.program
    class SliceNdDecode:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            src: pl.Tensor[[32, 512], pl.FP4E2M1X2],
            column: pl.Scalar[pl.INT64],
            out: pl.Out[pl.Tensor[[16, 512], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 512], pl.FP8E4M3FN]:
            packed = pl.load(src, [0, 0], [32, 512])
            start = column * column_scale + column_base
            window = pl.tile.slice(packed, [16, 256], [0, start], valid_shape=[16, 256])
            return pl.store(pl.cast(window, pl.FP8E4M3FN), [0, 0], out)

    mlir = _emit_incore_mlir(SliceNdDecode, planner)
    extract = next(line for line in mlir.splitlines() if "pto.textract ins(" in line)
    assert "dtype=i8, rows=32, cols=512" in extract and "dtype=i8, rows=16, cols=256" in extract
    if not dynamic_column:
        assert ", %c0_index, %c1_index :" in extract
    assert mlir.count("pto.tcvt ") == 3, mlir
    assert "dtype=bf16" in mlir and "dtype=f32" in mlir, mlir
    if find_ptoas_binary() is not None:
        pto, cpp = tmp_path / "slice_nd_bridge.pto", tmp_path / "slice_nd_bridge.cpp"
        pto.write_text(mlir)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", f"--pto-level={level}", "--pto-arch=a5"])
        assert cpp.read_text().count("TCVT(") == 3


@pytest.mark.parametrize("planner", list(passes.MemoryPlanner))
def test_nested_fp4_window_preserves_partial_validity(tmp_path, planner):
    @pl.program
    class Partial:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self, src: pl.Tensor[[8, 128], pl.FP4E1M2X2], out: pl.Out[pl.Tensor[[3, 64], pl.BF16]]
        ) -> pl.Tensor[[3, 64], pl.BF16]:
            packed = pl.load(src, [0, 0], [8, 128])
            outer = pl.tile.slice(packed, [6, 96], [1, 1])
            inner = pl.tile.slice(outer, [3, 32], [1, 2], valid_shape=[2, 23])
            return pl.store(pl.cast(inner, pl.BF16), [0, 0], out)

    mlir = _emit_incore_mlir(Partial, planner)
    extract = next(line for line in mlir.splitlines() if "pto.textract ins(" in line)
    assert ", %c2_index, %c3_index :" in extract and "dtype=i8, rows=8, cols=128" in extract
    store = next(line for line in mlir.splitlines() if "pto.tstore ins(" in line)
    assert "2x46xbf16" in store, store
    if find_ptoas_binary() is not None:
        pto, cpp = tmp_path / "partial.pto", tmp_path / "partial.cpp"
        pto.write_text(mlir)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", f"--pto-level={level}", "--pto-arch=a5"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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
from pypto.backend.pto_backend import _run_ptoas
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


def test_uint8_ub_fp4_alias_rejects_ptoas_memory_planner():
    with pytest.raises(ValueError, match="FP4E2M1X2 tile aliases require PYPTO or DSA_RP"):
        _emit_incore_mlir(_uint8_ub_fp4_decode_program(), passes.MemoryPlanner.PTOAS)


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

    with pytest.raises(ValueError, match="FP4E2M1X2 UB alias requires an address-backed tile"):
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

    if planner == passes.MemoryPlanner.PTOAS:
        with pytest.raises(ValueError, match="FP4E2M1X2 tile aliases require PYPTO or DSA_RP"):
            _emit_incore_mlir(Program, planner)
        return
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
def test_fp4_nz_assemble_uses_native_insert(dtype):
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
        "rows=16, cols=128" in line and "rows=32, cols=256" in line and "%c64_index" in line
        for line in inserts
    ), mlir


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

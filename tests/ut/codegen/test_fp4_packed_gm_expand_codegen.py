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
from pypto import DataType, ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
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


def _emit_incore_mlir(program, planner=passes.MemoryPlanner.PYPTO) -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    parts: list[str] = []
    for func in optimized.functions.values():
        if func.func_type in (pl.FunctionType.Orchestration, pl.FunctionType.Group):
            continue
        single = ir.Program([func], func.name, optimized.span)
        result = codegen.PTOCodegen().generate(single, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
        parts.append(result if isinstance(result, str) else "".join(result.values()))
    return "\n".join(parts)


def _uint8_ub_fp4_decode_program():
    @pl.program
    class Decode:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            packed: pl.Tensor[[96, 320], pl.UINT8],
            output: pl.Out[pl.Tensor[[96, 640], pl.BF16]],
        ) -> pl.Tensor[[96, 640], pl.BF16]:
            payload = pl.tile.load(packed, [32, 32], [32, 256])
            fp4 = pl.reinterpret_view(payload, pl.FP4E2M1X2)
            decoded = pl.cast(fp4, pl.BF16)
            return pl.tile.store(decoded, [32, 64], output)

    return Decode


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_uint8_ub_fp4_alias_is_consumed_by_native_cast_and_store(planner):
    mlir = _emit_incore_mlir(_uint8_ub_fp4_decode_program(), planner)
    assert mlir.count("pto.tcvt ") == 1, mlir
    assert mlir.count("pto.tload ") == mlir.count("pto.tstore ") == 1, mlir
    cast = next(line for line in mlir.splitlines() if "pto.tcvt " in line)
    assert "dtype=!pto.f4E2M1x2, rows=32, cols=256" in cast, cast
    assert "dtype=bf16, rows=32, cols=512" in cast, cast
    gm_views = [
        line for line in mlir.splitlines() if "pto.make_tensor_view" in line or "pto.partition_view" in line
    ]
    assert gm_views and all("f4E2M1x2" not in line for line in gm_views), mlir
    assert not any(op in mlir for op in ("pto.tand", "pto.tshr", "pto.tshl", "pto.tscatter")), mlir

    assert "pto.treshape" not in mlir, mlir
    aliases = [
        line
        for line in mlir.splitlines()
        if "pto.alloc_tile" in line and ("dtype=ui8" in line or "dtype=!pto.f4E2M1x2" in line)
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


def test_fp4e2m1x2_rejects_column_vector_last_carrier_dim_one():
    """Carrier last-dim 1 would force DN; packed FP4E2M1X2 bans that path."""

    @pl.program
    class ColVecPacked:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[32, 1], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[32, 1], pl.FP4E2M1X2]],
        ) -> pl.Tensor[[32, 1], pl.FP4E2M1X2]:
            return pl.store(pl.load(src, [0, 0], [32, 1]), [0, 0], out)

    with pytest.raises(ValueError, match=r"last carrier dimension 1|column-vector"):
        _emit_incore_mlir(ColVecPacked)


def test_fp4e2m1x2_rejects_explicit_dn_param_annotation():
    """Non-ND annotations on packed FP4E2M1X2 are rejected at make_tensor_view."""

    @pl.program
    class DnAnnotated:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[
                [16, 32],
                pl.FP4E2M1X2,
                pl.TensorView(stride=[1, 16], layout=pl.TensorLayout.DN),
            ],
            out: pl.Out[pl.Tensor[[16, 32], pl.FP4E2M1X2]],
        ) -> pl.Tensor[[16, 32], pl.FP4E2M1X2]:
            return pl.store(pl.load(src, [0, 0], [16, 32]), [0, 0], out)

    with pytest.raises(ValueError, match=r"FP4E2M1X2 supports ND layout only"):
        _emit_incore_mlir(DnAnnotated)


def test_fp4e2m1x2_slice_cast_and_vec_move():
    """Even last-axis slice + cast emits f4x2; move stays on Vec."""

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

    @pl.program
    class MoveProg:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 32], pl.FP4E2M1X2]],
        ) -> pl.Tensor[[16, 32], pl.FP4E2M1X2]:
            return pl.store(pl.move(pl.load(src, [0, 0], [16, 32]), target_memory=pl.Mem.Vec), [0, 0], out)

    mlir_m = _emit_incore_mlir(MoveProg)
    assert "!pto.f4E2M1x2" in mlir_m
    assert "pto.tmov" in mlir_m or "pto.tload" in mlir_m


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


def _fp4_gm_metadata(
    mlir: str, runtime_indices: dict[str, int] | None = None
) -> tuple[list[tuple], list[tuple]]:
    """Resolve static index arithmetic in emitted views and their partitions.

    Assertions use the actual descriptor strides and offsets to compute typed
    pointer byte addresses, so a compensating pair of wrong units cannot pass.
    """
    values: dict[str, int] = dict(runtime_indices or {})
    views: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    descriptors, accesses = [], []
    ssa = r"%[\w.$]+"

    def indices(text: str) -> tuple[int, ...]:
        return tuple(values[name.strip()] for name in text.split(","))

    for line in mlir.splitlines():
        if "func.func @" in line:
            values, views = dict(runtime_indices or {}), {}
        constant = re.search(rf"({ssa}) = arith.constant (-?\d+) : index", line)
        if constant:
            values[constant[1]] = int(constant[2])
        arithmetic = re.search(rf"({ssa}) = arith.(muli|divsi|addi|maxsi) ({ssa}), ({ssa}) : index", line)
        if arithmetic and arithmetic[3] in values and arithmetic[4] in values:
            lhs, rhs = values[arithmetic[3]], values[arithmetic[4]]
            if arithmetic[2] == "muli":
                values[arithmetic[1]] = lhs * rhs
            elif arithmetic[2] == "divsi":
                values[arithmetic[1]] = lhs // rhs
            elif arithmetic[2] == "addi":
                values[arithmetic[1]] = lhs + rhs
            else:
                # Dynamic partition offsets are clamped to zero in the emitted
                # descriptor; this does not rescale their carrier-byte units.
                values[arithmetic[1]] = max(lhs, rhs)
        if "f4E2M1x2" not in line:
            continue
        view = re.search(
            rf"({ssa}) = pto.make_tensor_view .*?, shape = \[([^\]]+)\], strides = \[([^\]]+)\]",
            line,
        )
        if view:
            metadata = indices(view[2]), indices(view[3])
            views[view[1]] = metadata
            descriptors.append(metadata)
        partition = re.search(
            rf"pto.partition_view ({ssa}), offsets = \[([^\]]+)\], sizes = \[([^\]]+)\]", line
        )
        if partition:
            shape, strides = views[partition[1]]
            offsets, sizes = indices(partition[2]), indices(partition[3])
            assert len(offsets) == len(strides), line
            byte_offset = sum(offset * stride for offset, stride in zip(offsets, strides))
            accesses.append((shape, strides, offsets, sizes, byte_offset))
    return descriptors, accesses


def _assert_fp4_gm_window(mlir, shape, strides, offsets, sizes, byte_offset):
    descriptors, accesses = _fp4_gm_metadata(mlir)
    assert descriptors and all(item == (shape, strides) for item in descriptors), mlir
    assert accesses and all(item == (shape, strides, offsets, sizes, byte_offset) for item in accesses), mlir
    partition_type = "x".join(str(size) for size in sizes)
    assert f"!pto.partition_tensor_view<{partition_type}x!pto.f4E2M1x2>" in mlir, mlir


def _partition_line(mlir: str, dtype_fragment: str) -> str:
    lines = [ln for ln in mlir.splitlines() if "pto.partition_view" in ln and dtype_fragment in ln]
    assert lines, mlir
    return lines[0]


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_fp4_make_tensor_view_and_partition_use_carrier_units(planner):
    """Parameter views and tensor.view agree with the unchanged packed ABI."""
    columns = 256

    @pl.program
    class Rank2:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[2, columns], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[2, columns], pl.FP4E2M1X2]],
        ) -> pl.Tensor[[2, columns], pl.FP4E2M1X2]:
            viewed: pl.Tensor[[2, columns], pl.FP4E2M1X2] = pl.tensor.view(src, [2, columns])
            return pl.store(pl.load(viewed, [0, 0], [2, columns]), [0, 0], out)

    mlir = _emit_incore_mlir(Rank2, planner)
    _assert_fp4_gm_window(mlir, (2, 256), (256, 1), (0, 0), (2, 256), 0)
    assert "pto.tload" in mlir and "pto.tstore" in mlir


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_fp4_rank3_keeps_odd_batch_and_row_coordinates(planner):
    """The rank-3 load and explicit-shapes store both address byte 616."""
    columns = 32
    column_offset = 8
    window_columns = 16

    @pl.program
    class Rank3:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[3, 16, columns], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[3, 16, columns], pl.FP4E2M1X2]],
        ) -> pl.Tensor[[3, 16, columns], pl.FP4E2M1X2]:
            tile = pl.load(src, [1, 3, column_offset], [1, 8, window_columns])
            return pl.store(tile, [1, 3, column_offset], out)

    mlir = _emit_incore_mlir(Rank3, planner)
    _assert_fp4_gm_window(mlir, (3, 16, 32), (512, 32, 1), (1, 3, 8), (1, 8, 16), 616)
    assert "pto.tload" in mlir and "pto.tstore" in mlir


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_fp4_strided_nd_load_and_standard_store_keep_carrier_addresses(planner):
    """A padded carrier pitch must remain unchanged in both GM descriptors."""
    columns = 32
    row_stride = 80
    column_offset = 16
    window_columns = 16

    @pl.program
    class Strided:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[
                [16, columns], pl.FP4E2M1X2, pl.TensorView(stride=[row_stride, 1], layout=ir.TensorLayout.ND)
            ],
            out: pl.Out[
                pl.Tensor[
                    [16, columns],
                    pl.FP4E2M1X2,
                    pl.TensorView(stride=[row_stride, 1], layout=ir.TensorLayout.ND),
                ]
            ],
        ) -> pl.Tensor[
            [16, columns], pl.FP4E2M1X2, pl.TensorView(stride=[row_stride, 1], layout=ir.TensorLayout.ND)
        ]:
            return pl.store(pl.load(src, [3, column_offset], [8, window_columns]), [3, column_offset], out)

    mlir = _emit_incore_mlir(Strided, planner)
    _assert_fp4_gm_window(mlir, (16, 32), (80, 1), (3, 16), (8, 16), 256)
    assert "pto.tload" in mlir and "pto.tstore" in mlir


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_fp4_dense_512_window_uses_256_byte_row_pitch(planner):
    """The failing dense KN geometry must yield 256-byte rows, not 512."""
    columns = 256
    column_offset = 64
    window_columns = 128

    @pl.program
    class Dense512:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[512, columns], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[128, 256], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[128, 256], pl.FP8E4M3FN]:
            tile = pl.load(src, [128, column_offset], [128, window_columns])
            return pl.store(pl.cast(tile, pl.FP8E4M3FN), [0, 0], out)

    mlir = _emit_incore_mlir(Dense512, planner)
    _assert_fp4_gm_window(mlir, (512, 256), (256, 1), (128, 64), (128, 128), 32832)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_packed_fp4_odd_carrier_columns_and_stride_are_preserved(planner):
    """Odd packed metadata already counts bytes and needs no parity guard."""

    @pl.program
    class OddCarriers:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 65], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[8, 34], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[8, 34], pl.FP8E4M3FN]:
            return pl.store(pl.cast(pl.load(src, [3, 17], [8, 17]), pl.FP8E4M3FN), [0, 0], out)

    mlir = _emit_incore_mlir(OddCarriers, planner)
    _assert_fp4_gm_window(mlir, (16, 65), (65, 1), (3, 17), (8, 17), 212)


def test_fp4_to_fp8_lowers_to_single_native_cast():
    """Packed FP4 lowers to one native conversion with no compiler LUT."""

    @pl.program
    class BitopsCast:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            return pl.store(pl.cast(pl.load(src, [0, 0], [16, 32]), pl.FP8E4M3FN), [0, 0], out)

    mlir = _emit_incore_mlir(BitopsCast)
    assert "pto.tgather" not in mlir and "pto.tci" not in mlir, mlir
    data_tcvt = [ln for ln in mlir.splitlines() if "pto.tcvt" in ln]
    assert len(data_tcvt) == 1, mlir
    assert "dtype=!pto.f4E2M1x2" in data_tcvt[0] and "dtype=f8E4M3FN" in data_tcvt[0]


def test_fp4_partition_offsets_use_carrier_coordinates():
    """Packed windows use carrier coordinates; FP8 coordinates stay unchanged."""

    @pl.program
    class PackedWindow:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 64], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[8, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[8, 64], pl.FP8E4M3FN]:
            return pl.store(pl.cast(pl.load(src, [4, 16], [8, 32]), pl.FP8E4M3FN), [0, 0], out)

    mlir = _emit_incore_mlir(PackedWindow)
    line = _partition_line(mlir, "f4E2M1x2")
    # Packed row/column coordinates and window extents already count carriers.
    assert "offsets = [%c4_index, %c16_index]" in line, line
    assert "sizes = [%c8_index, %c32_index]" in line, line
    _assert_fp4_gm_window(mlir, (16, 64), (64, 1), (4, 16), (8, 32), 272)

    @pl.program
    class Fp8Control:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 64], pl.FP8E4M3FN],
            out: pl.Out[pl.Tensor[[8, 32], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[8, 32], pl.FP8E4M3FN]:
            return pl.store(pl.load(src, [4, 32], [8, 32]), [0, 0], out)

    mlir_ctl = _emit_incore_mlir(Fp8Control)
    line_ctl = _partition_line(mlir_ctl, "f8E4M3FN")
    assert "offsets = [%c4_index, %c32_index]" in line_ctl, line_ctl


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("row_stride", [64, 65, 160, None])
@pytest.mark.parametrize("offset_kind", ["row", "column", "named_row", "chained_row", "odd_row"])
def test_packed_dynamic_offsets_and_strides_are_already_bytes(planner, row_stride, offset_kind):
    """Dynamic carrier coordinates need no old nibble-parity proof or division."""
    span = ir.Span.unknown()

    def index(n):
        return ir.ConstInt(n, DataType.INDEX, span)

    runtime = ir.Var("runtime", ir.ScalarType(DataType.INDEX), span)
    named = ir.Var("named", runtime.type, span)
    chained = ir.Var("chained", runtime.type, span)
    statements: list[ir.Stmt] = [
        ir.AssignStmt(
            named, ir.Mul(runtime, index(3 if offset_kind == "odd_row" else 64), DataType.INDEX, span), span
        ),
        ir.AssignStmt(chained, ir.Add(named, index(0), DataType.INDEX, span), span),
    ]
    row = {"row": runtime, "named_row": named, "chained_row": chained, "odd_row": named}.get(
        offset_kind, index(3)
    )
    column = runtime if offset_kind == "column" else index(17)
    tensor_type = ir.TensorType(
        [256, 64],
        DataType.FP4E2M1X2,
        tensor_view=ir.TensorView(
            stride=[runtime if row_stride is None else row_stride, 1], layout=ir.TensorLayout.ND
        ),
    )
    gm = ir.Var("gm", tensor_type, span)
    tile_type = ir.TileType(
        [8, 32],
        DataType.FP4E2M1X2,
        ir.MemRef("tile_base", 0, 256),
        ir.TileView(valid_shape=[8, 17]),
        ir.MemorySpace.Vec,
    )
    tile = ir.Var("tile", tile_type, span)
    shape = ir.MakeTuple([index(8), index(32)], span)
    valid = ir.MakeTuple([index(8), index(17)], span)
    load = ir.Call(
        ir.get_op("tile.load"), [gm, ir.MakeTuple([row, column], span), shape, valid], {}, tile_type, span
    )
    statements += [ir.AssignStmt(tile, load, span), ir.ReturnStmt([], span)]
    function = ir.Function(
        "packed_dynamic", [gm, runtime], [], ir.SeqStmts(statements, span), span, ir.FunctionType.AIV
    )
    set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=planner):
        mlir = codegen.PTOCodegen().generate(
            ir.Program([function], "packed_dynamic", span),
            emit_tile_addr=planner == passes.MemoryPlanner.PYPTO,
        )
    # Scalar arguments use ABI names and aliases use expression SSA names.
    # Evaluate actual emitted descriptors instead of guessing those names.
    runtime_value = 3
    descriptors, accesses = _fp4_gm_metadata(mlir, {"%arg1": runtime_value})
    pitch = runtime_value if row_stride is None else row_stride
    expected_row = {
        "row": runtime_value,
        "column": 3,
        "named_row": runtime_value * 64,
        "chained_row": runtime_value * 64,
        "odd_row": runtime_value * 3,
    }[offset_kind]
    expected_column = runtime_value if offset_kind == "column" else 17
    expected_view = ((256, 64), (pitch, 1))
    expected_access = (
        *expected_view,
        (expected_row, expected_column),
        (8, 17),  # DMA describes the valid carrier window, not the UB allocation.
        expected_row * pitch + expected_column,
    )
    assert descriptors and all(view == expected_view for view in descriptors), mlir
    assert accesses and all(access == expected_access for access in accesses), mlir
    assert "arith.div" not in mlir and "arith.rem" not in mlir, mlir
    assert "dtype=!pto.f4E2M1x2, rows=8, cols=32" in mlir, mlir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

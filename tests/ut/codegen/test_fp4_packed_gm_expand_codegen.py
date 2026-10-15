# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen checks for hand-written FP4E2M1X2 GM expand (carrier → nibble ABI)."""

import re

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


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


def test_uint8_ub_fp4_alias_rejects_ptoas_memory_planner():
    with pytest.raises(ValueError, match="FP4E2M1X2 UB alias requires memory_planner=PYPTO or DSA_RP"):
        _emit_incore_mlir(_uint8_ub_fp4_decode_program(), passes.MemoryPlanner.PTOAS)


def _uint8_slice_fp4_decode_program(parent_cols, column_offset):
    @pl.program
    class DecodeSlice:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            packed: pl.Tensor[[32, parent_cols], pl.UINT8],
            output: pl.Out[pl.Tensor[[32, 512], pl.BF16]],
        ) -> pl.Tensor[[32, 512], pl.BF16]:
            payload = pl.tile.load(packed, [0, 0], [32, parent_cols])
            window = pl.tile.slice(payload, [32, 256], [0, column_offset])
            fp4 = pl.reinterpret_view(window, pl.FP4E2M1X2)
            return pl.tile.store(pl.cast(fp4, pl.BF16), [0, 0], output)

    return DecodeSlice


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize(("parent_cols", "column_offset"), [(512, 256), (320, 32)])
def test_uint8_ub_fp4_alias_rejects_noncontiguous_slice(planner, parent_cols, column_offset):
    program = _uint8_slice_fp4_decode_program(parent_cols, column_offset)
    with pytest.raises(ValueError, match="FP4E2M1X2 UB alias requires a contiguous UINT8 tile"):
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

    with pytest.raises(ValueError, match="FP4E2M1X2 UB alias requires a contiguous UINT8 tile"):
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

    with pytest.raises(ValueError, match="FP4E2M1X2 UB alias requires a contiguous UINT8 tile"):
        _emit_incore_mlir(DecodeNestedSlice, planner)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_uint8_ub_fp4_alias_rejects_dynamic_ub_offset_through_reshape(planner):
    @pl.program
    class DecodeDynamicSlice:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            packed: pl.Tensor[[64, 256], pl.UINT8],
            start: pl.Scalar[pl.INT64],
            output: pl.Out[pl.Tensor[[16, 1024], pl.BF16]],
        ) -> pl.Tensor[[16, 1024], pl.BF16]:
            payload = pl.tile.load(packed, [0, 0], [64, 256])
            window = pl.tile.slice(payload, [32, 256], [start, 0])
            reshaped = pl.tile.reshape(window, [16, 512])
            fp4 = pl.reinterpret_view(reshaped, pl.FP4E2M1X2)
            return pl.tile.store(pl.cast(fp4, pl.BF16), [0, 0], output)

    with pytest.raises(ValueError, match="FP4E2M1X2 UB alias requires a contiguous UINT8 tile"):
        _emit_incore_mlir(DecodeDynamicSlice, planner)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_uint8_ub_fp4_alias_accepts_contiguous_slice(planner):
    mlir = _emit_incore_mlir(_uint8_slice_fp4_decode_program(256, 0), planner)
    assert mlir.count("pto.tcvt ") == 1, mlir
    assert "pto.treshape" not in mlir, mlir


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_uint8_ub_fp4_alias_rejects_memref_less_tpop(planner):
    @pl.program
    class DecodePop:
        @pl.function(type=pl.FunctionType.AIV)
        def main(self, output: pl.Out[pl.Tensor[[32, 512], pl.BF16]]) -> pl.Tensor[[32, 512], pl.BF16]:
            buf = pl.reserve_buffer(name="c2v_slot_buffer", size=16384, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=8192, c2v_consumer_buf=buf)
            payload: pl.Tile[[32, 256], pl.UINT8, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
            fp4 = pl.reinterpret_view(payload, pl.FP4E2M1X2)
            decoded = pl.cast(fp4, pl.BF16)
            pl.tfree_to_aic(payload)
            return pl.tile.store(decoded, [0, 0], output)

    with pytest.raises(ValueError, match="FP4E2M1X2 UB alias requires an address-backed tile"):
        _emit_incore_mlir(DecodePop, planner)


def test_fp4e2m1x2_make_tensor_view_expands_to_nibble_units():
    """Param + InCore tensor.view + rank-3 leading strides expand carrier→nibble."""

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
    assert views and all("512" in line for line in views), mlir
    # Carrier IR last-axis 256 must not appear unexpanded on make_tensor_view.
    assert all("%c256_index" not in line for line in views), mlir
    # Static ConstInt partition last-axis expand folds *2 (carrier 256 → nibble 512).
    partitions = [line for line in mlir.splitlines() if "partition_view" in line]
    assert partitions and all("%c512_index" in line for line in partitions), mlir

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
    # Carrier [2,16,32] → nibble shape last-axis 32*2=64 (ConstInt fold).
    assert all("shape = [%c2_index, %c16_index, %c64_index]" in line for line in views3), mlir3
    # Leading strides expand via *2: row pitch 32*16 then *2 → 1024; mid stride 32*2 → 64.
    assert "arith.muli %c32_index, %c16_index : index" in mlir3, mlir3
    assert "arith.muli %c32_index, %c2_index : index" in mlir3, mlir3
    assert any(
        "arith.muli" in line and "_s0," in line and "%c2_index" in line for line in mlir3.splitlines()
    ), mlir3
    assert all("%c1_index]" in line for line in views3), mlir3


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

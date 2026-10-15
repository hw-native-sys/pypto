# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen checks for hand-written FP4E2M1X2 GM expand (carrier → nibble ABI)."""

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


def _emit_incore_mlir(program) -> str:
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


def test_fp4_to_fp8_loweres_to_identity_bitop_chain():
    """FP4 -> FP8E4M3FN lowering follows the LowersFp4ToFp8InKernel default.

    The default is the FP4->BF16->FP32->FP8 hops (t12 measured the single-pass
    bit-op chain as a 1.87-6.16x end-to-end regression; t18 owns cutting its
    op count before it may return). With the hook off the program carries
    three tcvt hops and the BF16/FP32 intermediates.
    """

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
    # Gather route: 1 tgather on the data path; tcvt ops only on the tiny
    # [1, 256] LUT construction tile + the u8->u16 zero-extend.
    assert "pto.tgather" in mlir, mlir
    assert "pto.tci" in mlir, mlir  # the LUT ramp generation
    # No BF16/FP32 in the data path (the 3-hop chain is not taken).
    data_tcvt = [ln for ln in mlir.splitlines() if "pto.tcvt" in ln and "rows=16" in ln]
    assert len(data_tcvt) <= 1, mlir  # only the u8->u16 zero-extend on data

    @pl.program
    class LogicalBitopsCast:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 64], pl.FP4],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            return pl.store(pl.cast(pl.load(src, [0, 0], [16, 64]), pl.FP8E4M3FN), [0, 0], out)

    mlir_log = _emit_incore_mlir(LogicalBitopsCast)
    assert "pto.tgather" in mlir_log, mlir_log


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


def _partition_line(mlir: str, dtype_fragment: str) -> str:
    lines = [ln for ln in mlir.splitlines() if "pto.partition_view" in ln and dtype_fragment in ln]
    assert lines, mlir
    return lines[0]


def test_fp4_partition_offsets_emit_prehalved():
    """PTOAS advances the partition base in carrier-byte units; offsets halve.

    PTOAS lowers ``partition_view`` to ``data_ptr + sum(offset * stride)`` in
    element (byte) units while pto-isa walks the same view's strides in nibble
    units, so PyPTO emits every FP4-family window offset pre-halved: row
    offsets halve from nibble rows, and the packed last axis nets back to the
    carrier value (expand *2 then halve). Sizes keep nibble units.
    """

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
    # Rows 4 -> 2; carrier cols 16 -> expand 32 -> halve 16. Size stays nibble-doubled (64).
    assert "offsets = [%c2_index, %c16_index]" in line, line
    assert "sizes = [%c8_index, %c64_index]" in line, line

    @pl.program
    class LogicalWindow:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 64], pl.FP4],
            out: pl.Out[pl.Tensor[[8, 32], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[8, 32], pl.FP8E4M3FN]:
            return pl.store(pl.cast(pl.load(src, [4, 32], [8, 32]), pl.FP8E4M3FN), [0, 0], out)

    mlir_log = _emit_incore_mlir(LogicalWindow)
    line_log = _partition_line(mlir_log, "f4E2M1x2")
    # Nibble offsets [4, 32] halve to [2, 16]; logical sizes already nibbles.
    assert "offsets = [%c2_index, %c16_index]" in line_log, line_log
    assert "sizes = [%c8_index, %c32_index]" in line_log, line_log

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


def test_fp4_partition_rejects_odd_offsets():
    """Odd nibble-unit offsets have no integer byte encoding under PTOAS."""

    @pl.program
    class OddRows:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 64], pl.FP4],
            out: pl.Out[pl.Tensor[[8, 32], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[8, 32], pl.FP8E4M3FN]:
            return pl.store(pl.cast(pl.load(src, [3, 32], [8, 32]), pl.FP8E4M3FN), [0, 0], out)

    with pytest.raises(ValueError, match="odd in nibble units"):
        _emit_incore_mlir(OddRows)

    @pl.program
    class OddCols:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 96], pl.FP4],
            out: pl.Out[pl.Tensor[[8, 32], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[8, 32], pl.FP8E4M3FN]:
            return pl.store(pl.cast(pl.load(src, [4, 33], [8, 32]), pl.FP8E4M3FN), [0, 0], out)

    with pytest.raises(ValueError, match="odd in nibble units"):
        _emit_incore_mlir(OddCols)


def test_fp4_partition_dynamic_offset_provably_even_halves():
    """Codex P1 (PR#2987 round 2): a dynamic offset provably even (an even
    constant times a runtime value) keeps the exact symbolic halving."""

    @pl.program
    class DynamicRows:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[256, 64], pl.FP4],
            out: pl.Out[pl.Tensor[[8, 32], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[8, 32], pl.FP8E4M3FN]:
            out_t: pl.Tensor[[8, 32], pl.FP8E4M3FN] = pl.store(
                pl.cast(pl.load(src, [0, 32], [8, 32]), pl.FP8E4M3FN), [0, 0], out
            )
            for kb in pl.pipeline(0, 4, stage=1):
                t = pl.cast(pl.load(src, [kb * 64, 32], [8, 32]), pl.FP8E4M3FN)
                out_t = pl.store(t, [0, 0], out)
            return out_t

    mlir = _emit_incore_mlir(DynamicRows)
    assert "arith.divsi" in mlir, mlir[:1500]


def test_fp4_partition_dynamic_offset_unprovable_halves_silently():
    """t36 (PR#2987 device validation): a dynamic offset whose evenness
    cannot be proven locally — every real tiled kernel's runtime block
    coordinate arrives as an opaque device-function scalar (the loop lives
    in the orchestration driver) — keeps the documented exact-for-even
    symbolic halving; the static odd offset stays hard-rejected (see
    test_fp4_partition_rejects_odd_offsets)."""

    @pl.program
    class ScalarRows:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            s: pl.Tensor[[1, 1], pl.INT32],
            src: pl.Tensor[[256, 64], pl.FP4],
            out: pl.Out[pl.Tensor[[8, 32], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[8, 32], pl.FP8E4M3FN]:
            off = pl.read(s, [0, 0])
            return pl.store(pl.cast(pl.load(src, [off, 32], [8, 32]), pl.FP8E4M3FN), [0, 0], out)

    mlir = _emit_incore_mlir(ScalarRows)
    assert "arith.divsi" in mlir, mlir[:1500]


def test_fp4_partition_named_scalar_offset_follows_definition():
    """Codex P2 (PR#2987 round 3): `row = kb * 64` then
    ``pl.load(src, [row, 0], ...)`` must prove even by following the named
    scalar's definition — Simplify deliberately keeps symbolic assignments,
    so the prover resolves them instead of rejecting the Var."""

    @pl.program
    class NamedScalarRows:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[256, 64], pl.FP4],
            out: pl.Out[pl.Tensor[[8, 32], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[8, 32], pl.FP8E4M3FN]:
            out_t: pl.Tensor[[8, 32], pl.FP8E4M3FN] = pl.store(
                pl.cast(pl.load(src, [0, 32], [8, 32]), pl.FP8E4M3FN), [0, 0], out
            )
            for kb in pl.pipeline(0, 4, stage=1):
                row = kb * 64
                t = pl.cast(pl.load(src, [row, 32], [8, 32]), pl.FP8E4M3FN)
                out_t = pl.store(t, [0, 0], out)
            return out_t

    mlir = _emit_incore_mlir(NamedScalarRows)
    assert "arith.divsi" in mlir, mlir[:1500]


def test_fp4_partition_named_scalar_chain_follows_definitions():
    """A chained named scalar (`row2 = row + 0` where `row = kb * 64`)
    resolves transitively; an odd product through a name still rejects."""

    @pl.program
    class ChainedScalar:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[256, 64], pl.FP4],
            out: pl.Out[pl.Tensor[[8, 32], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[8, 32], pl.FP8E4M3FN]:
            out_t: pl.Tensor[[8, 32], pl.FP8E4M3FN] = pl.store(
                pl.cast(pl.load(src, [0, 32], [8, 32]), pl.FP8E4M3FN), [0, 0], out
            )
            for kb in pl.pipeline(0, 4, stage=1):
                row = kb * 64
                row2 = row + 0
                t = pl.cast(pl.load(src, [row2, 32], [8, 32]), pl.FP8E4M3FN)
                out_t = pl.store(t, [0, 0], out)
            return out_t

    mlir = _emit_incore_mlir(ChainedScalar)
    assert "arith.divsi" in mlir, mlir[:1500]

    @pl.program
    class OddNamedScalar:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            s: pl.Tensor[[1, 1], pl.INT32],
            src: pl.Tensor[[256, 64], pl.FP4],
            out: pl.Out[pl.Tensor[[8, 32], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[8, 32], pl.FP8E4M3FN]:
            off = pl.read(s, [0, 0])
            odd = off * 3
            return pl.store(pl.cast(pl.load(src, [odd, 32], [8, 32]), pl.FP8E4M3FN), [0, 0], out)

    mlir_odd = _emit_incore_mlir(OddNamedScalar)
    assert "arith.divsi" in mlir_odd, mlir_odd[:1500]


def test_fp4_partition_parallel_domain_coordinate_offset_accepts_even_step():
    """t36 (PR#2987 device validation): `for nb in pl.parallel(0, N, N_TILE)`
    with an EVEN static step — the domain-coordinate offset pattern every
    real tiled kernel uses — must prove even (loop values are multiples of
    the step); this is the form the CPU UTs missed by testing only explicit
    kb * K_TILE products."""

    @pl.program
    class ParallelOffset:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[256, 128], pl.FP4],
            out: pl.Out[pl.Tensor[[8, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[8, 64], pl.FP8E4M3FN]:
            out_t: pl.Tensor[[8, 64], pl.FP8E4M3FN] = pl.store(
                pl.cast(pl.load(src, [0, 64], [8, 64]), pl.FP8E4M3FN), [0, 0], out
            )
            for nb in pl.parallel(0, 128, 64):
                t = pl.cast(pl.load(src, [0, nb], [8, 64]), pl.FP8E4M3FN)
                out_t = pl.store(t, [0, 0], out)
            return out_t

    mlir = _emit_incore_mlir(ParallelOffset)
    assert "arith.divsi" in mlir, mlir[:1500]

    # A static-provable-odd step is a compile-time layout bug and stays
    # rejected (see test_fp4_partition_rejects_odd_offsets for the static
    # form); an odd DYNAMIC value is not statically visible here and keeps
    # the documented exact-for-even divsi semantics (t36 captain ruling:
    # odd offsets can only come from bugs; no runtime defense).

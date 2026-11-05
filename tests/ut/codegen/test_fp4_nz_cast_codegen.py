# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen for the A5 packed FP4 cast with an NZ destination layout.

`pl.tile.cast(..., blayout=col_major, slayout=row_major)` asks the native A5
`pto.tcvt` to write its FP8 result straight into the standard 512-byte NZ box
grid, which is the layout the matrix unit consumes. PTOAS v0.72 implements
exactly that for `!pto.f4E2M1x2 -> f8E4M3FN` (see the `f4E2M1x2 -> f8E4M3FN`
paragraph of its `PTO_IR_manual.md`, and its own
`test/lit/pto/tcvt_fp4_direct_nz_a5.pto`), with these constraints:

* source ND (`row_major` / `none_box`), `compact=0`, packed `cols % 32 == 0`
* destination `col_major` / `row_major` / `fractal=512`, physical `rows % 16 == 0`
* both tiles in `loc=vec`, valid columns static

What the fusion buys is asserted as a *delta* against the same kernel without
the keyword: the ND form needs a Vec-to-Vec ND-to-NZ `pto.tmov` (and a Vec
buffer for it) before the operand can be staged to Mat, and the NZ form does
not. Counting only the NZ form would not show that anything was saved.

The `rows % 16` and `cols % 32` granularities also mean the tile is a whole
number of 512-byte boxes, so its byte footprint stays the plain `rows * cols`
product the allocator already computes -- there is no boxed-padding term to
add, and `test_nz_destination_allocation_size_is_unchanged` pins that.
"""

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes

_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]

M = 32
PACKED_K = 128  # packed FP4 bytes; 2 * PACKED_K logical FP8 columns
N = 64


@pytest.fixture(autouse=True)
def _a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def _emit(program: ir.Program, planner: passes.MemoryPlanner, compile_dir=None) -> str:
    """Run the production pipeline and, when PTOAS is available, assemble it."""
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    functions = [
        function for function in optimized.functions.values() if ir.is_incore_type(function.func_type)
    ]
    assert functions, "Expected an emitted InCore kernel"
    # Mixed kernels reference their peer's reserved buffers, so both sides must
    # assemble in one module for PTOAS to resolve every peer_func symbol.
    kernels = ir.Program(functions, optimized.name, optimized.span)
    source = codegen.PTOCodegen().generate(kernels, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    if compile_dir is not None and find_ptoas_binary() is not None:
        pto = compile_dir / "kernels.pto"
        cpp = compile_dir / "kernels.cpp"
        pto.write_text(source)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", f"--pto-level={level}", "--pto-arch=a5"])
        assert cpp.is_file()
    return source


def _nz_matmul_program():
    """Cast packed FP4 straight into NZ, then feed the FP8 matmul."""

    @pl.program
    class NzCastMatmul:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[M, PACKED_K], pl.fp4e2m1x2],
            b: pl.Tensor[[2 * PACKED_K, N], pl.FP8E4M3FN],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            packed = pl.tile.load(a, [0, 0], [M, PACKED_K])
            lhs = pl.tile.cast(
                packed,
                pl.FP8E4M3FN,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            rhs = pl.tile.load(b, [0, 0], [2 * PACKED_K, N])
            return pl.tile.store(pl.matmul(lhs, rhs), [0, 0], out)

    return NzCastMatmul


def _nd_matmul_program():
    """The same kernel without the keyword -- today's lowering, for the delta."""

    @pl.program
    class NdCastMatmul:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[M, PACKED_K], pl.fp4e2m1x2],
            b: pl.Tensor[[2 * PACKED_K, N], pl.FP8E4M3FN],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            packed = pl.tile.load(a, [0, 0], [M, PACKED_K])
            lhs = pl.tile.cast(packed, pl.FP8E4M3FN)
            rhs = pl.tile.load(b, [0, 0], [2 * PACKED_K, N])
            return pl.tile.store(pl.matmul(lhs, rhs), [0, 0], out)

    return NdCastMatmul


def _casts(mlir: str) -> list[str]:
    return [line for line in mlir.splitlines() if "pto.tcvt " in line]


def _vec_to_vec_relayout_moves(mlir: str) -> list[str]:
    """Vec-to-Vec `pto.tmov`s, i.e. the pure layout repairs this feature removes.

    Operand staging moves (Vec/Mat to Mat/Left/Right) change memory space and
    are unaffected, so they must not be counted.
    """
    moves = []
    for line in mlir.splitlines():
        if "pto.tmov " not in line:
            continue
        ins, _, outs = line.partition(") outs(")
        if "loc=vec" in ins and "loc=vec" in outs:
            moves.append(line.strip())
    return moves


@pytest.mark.parametrize("planner", _PLANNERS, ids=lambda p: p.name)
def test_nz_destination_cast_is_one_tcvt_writing_a_boxed_tile(planner, tmp_path):
    """One `pto.tcvt`, NZ destination type, no float intermediate -- and it assembles."""
    mlir = _emit(_nz_matmul_program(), planner, tmp_path)
    casts = _casts(mlir)
    assert len(casts) == 1, mlir
    cast = casts[0]
    assert "f4E2M1x2" in cast and "f8E4M3FN" in cast, cast
    # The destination type is PTOAS's standard NZ triple, rendered from the
    # cast's own recorded result view.
    destination = cast.partition(") outs(")[2]
    assert "blayout=col_major" in destination, cast
    assert "slayout=row_major" in destination, cast
    assert "fractal=512" in destination, cast
    assert f"rows={M}, cols={2 * PACKED_K}" in destination, cast
    # A BF16/FP32 bridge would mean the native edge was not selected at all.
    # Scoped to Vec: the FP32 Acc accumulator and the FP32 output are expected.
    assert "bf16" not in mlir, mlir
    assert "loc=vec, dtype=f32" not in mlir, mlir


@pytest.mark.parametrize("planner", _PLANNERS, ids=lambda p: p.name)
def test_nz_destination_removes_the_vec_relayout_move(planner):
    """The delta: requesting NZ drops exactly one Vec-to-Vec relayout and its buffer.

    Without the keyword the cast writes ND and a `pto.tmov` converts the whole
    tile to NZ before it can be staged to Mat. With it, the conversion is part
    of the `pto.tcvt` the kernel already had.
    """
    nd = _emit(_nd_matmul_program(), planner)
    nz = _emit(_nz_matmul_program(), planner)

    nd_moves = _vec_to_vec_relayout_moves(nd)
    nz_moves = _vec_to_vec_relayout_moves(nz)
    assert len(nd_moves) == 1, nd_moves
    assert nz_moves == [], nz_moves
    # The ND form's move is specifically the ND-to-NZ conversion being fused.
    assert "blayout=row_major" in nd_moves[0].partition(") outs(")[0]
    assert "blayout=col_major" in nd_moves[0].partition(") outs(")[2]

    # One fewer Vec buffer, and the conversion count is unchanged.
    assert nz.count("pto.alloc_tile ") == nd.count("pto.alloc_tile ") - 1
    assert len(_casts(nz)) == len(_casts(nd)) == 1


@pytest.mark.parametrize("planner", _PLANNERS, ids=lambda p: p.name)
def test_omitting_the_keyword_reproduces_todays_lowering(planner):
    """Strict no-op: the ND kernel's emitted module must not mention NZ anywhere
    the NZ kernel introduced it.

    Paired with ``test_nz_destination_removes_the_vec_relayout_move``, this is
    what makes the feature opt-in: the two kernels differ only by the keyword,
    and the ND one keeps its ND cast destination.
    """
    nd = _emit(_nd_matmul_program(), planner)
    cast = _casts(nd)
    assert len(cast) == 1, nd
    destination = cast[0].partition(") outs(")[2]
    assert "blayout=row_major" in destination, cast[0]
    assert "slayout=none_box" in destination, cast[0]


def test_nz_destination_allocation_size_is_unchanged():
    """A whole number of boxes occupies exactly `rows * cols` bytes.

    The allocator sizes a non-accumulator tile from its logical element count
    and is layout-blind. That is correct here *because* the NZ contract forces
    `rows % 16 == 0` and `cols % 32 == 0`, so there is no fractal padding to
    account for. If either granularity were relaxed, this assertion is the one
    that should fail first.
    """
    nz = _emit(_nz_matmul_program(), passes.MemoryPlanner.PYPTO)
    nd = _emit(_nd_matmul_program(), passes.MemoryPlanner.PYPTO)

    def fp8_cast_destination_bytes(mlir):
        cast = _casts(mlir)[0]
        destination = cast.partition(") outs(")[2]
        assert f"rows={M}, cols={2 * PACKED_K}" in destination, cast
        return M * 2 * PACKED_K  # FP8 is one byte per element

    assert fp8_cast_destination_bytes(nz) == fp8_cast_destination_bytes(nd) == M * 2 * PACKED_K


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

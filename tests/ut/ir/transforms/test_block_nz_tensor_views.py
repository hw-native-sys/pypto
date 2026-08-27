# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the BlockNzTensorViews pass.

``pl.Tensor[[..., R, C], dtype, pl.NZ]`` asserts that the GM bytes are already
in PTO-native NZ fractal order. pto-isa describes such a buffer with a blocked
rank-(r+2) GlobalTensor (``pto/common/pto_tile.hpp``)::

    TileShape2D<T, R, C, Layout::NZ> = Shape< 1, C/c0, R/16, 16, c0>
    BaseShape2D<T, R, C, Layout::NZ> = Stride<C*R, R*c0, 16*c0, c0, 1>

with ``c0 = 32 / sizeof(T)``. This pass rewrites the IR into that form while
keeping the destination tile logical 2-D.

Every shape below is written as a literal: a closure variable indexed inside a
``@pl.program`` body (``shape[1]``) parses to a ``TupleGetItemExpr`` rather than
a constant, which breaks the printer round-trip check.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.language.parser.diagnostics import ParserError
from pypto.pypto_core import codegen, passes

_PREFIX = [
    passes.inline_functions,
    passes.unroll_loops,
    passes.ctrl_flow_transform,
    passes.convert_to_ssa,
    passes.simplify,
    passes.normalize_stmt_structure,
    passes.flatten_call_expr,
    passes.outline_hierarchy_scopes,
    passes.outline_incore_scopes,
    passes.outline_cluster_scopes,
    passes.convert_tensor_to_tile_ops,
    passes.optimize_orch_tensors,
    passes.lower_composite_ops,
    passes.flatten_tile_nd_to_2d,
    passes.block_nz_tensor_views,
]


@pytest.fixture(autouse=True)
def _reset_backend_after_test():
    yield
    reset_for_testing()


def _run(program: ir.Program) -> ir.Program:
    """Run the Default pipeline prefix up to and including BlockNzTensorViews."""
    for factory in _PREFIX:
        program = factory()(program)
    return program


def _emit_pto(program: ir.Program, backend_type=BackendType.Ascend910B) -> str:
    reset_for_testing()
    set_backend_type(backend_type)
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


def _values(exprs) -> list[int]:
    return [e.value for e in exprs]


def _nz_param(program: ir.Program) -> ir.TensorType:
    """The single NZ-annotated parameter's TensorType."""
    for func in program.functions.values():
        for param in func.params:
            param_type = param.type
            if not isinstance(param_type, ir.TensorType):
                continue
            view = param_type.tensor_view
            if view is not None and view.layout == ir.TensorLayout.NZ:
                return param_type
    raise AssertionError("no NZ-annotated param found")


def _walk(stmt):
    """Yield every statement in a body, descending into SeqStmts."""
    if stmt is None:
        return
    if isinstance(stmt, ir.SeqStmts):
        for inner in stmt.stmts:
            yield from _walk(inner)
        return
    yield stmt


def _nz_load(program: ir.Program):
    """The single tile.load whose source tensor carries the NZ layout."""
    load_name = ir.get_op("tile.load").name
    found = []
    for func in program.functions.values():
        for stmt in _walk(func.body):
            if not isinstance(stmt, ir.AssignStmt):
                continue
            call = stmt.value
            if not isinstance(call, ir.Call) or call.op.name != load_name:
                continue
            view = getattr(call.args[0].type, "tensor_view", None)
            if view is not None and view.layout == ir.TensorLayout.NZ:
                found.append(call)
    assert len(found) == 1, f"expected exactly one NZ tile.load, got {len(found)}"
    return found[0]


# ============================================================================
# Programs under test
# ============================================================================


@pl.program
class NzMatmul:
    """[256, 512] INT8 NZ weight consumed as a matmul B operand."""

    @pl.function(type=pl.FunctionType.InCore)
    def main(
        self,
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[256, 512], pl.INT8, pl.NZ],
        out: pl.Tensor[[64, 256], pl.INT32],
    ):
        xt = pl.load(x, [0, 0], [64, 512], target_memory=pl.Mem.Mat)
        wt = pl.load(w, [0, 0], [256, 512], target_memory=pl.Mem.Mat)
        acc = pl.matmul(xt, pl.tile.transpose_view(wt), out_dtype=pl.INT32)
        pl.store(acc, [0, 0], out)
        return out


@pl.jit
def _batched_nz_mm(
    x: pl.Tensor[[64, 512], pl.INT8],
    w: pl.Tensor[[4, 256, 512], pl.INT8, pl.NZ],
    out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
):
    """Tensor-level grouped-matmul shape: logical [E, N, K] weight, logical slice."""
    for _ in pl.spmd(1, name_hint="batched_nz"):
        xt = pl.slice(x, [64, 512], [0, 0])
        wt = w[0:1, 0:256, 0:512]
        acc = pl.matmul(xt, wt, b_trans=True, out_dtype=pl.INT32)
        out[0:64, 0:256] = pl.reshape(acc, [64, 256])
    return out


@pl.program
class NdMatmul:
    """Same kernel with an ordinary ND weight — the pass must not touch it."""

    @pl.function(type=pl.FunctionType.InCore)
    def main(
        self,
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[256, 512], pl.INT8, pl.ND],
        out: pl.Tensor[[64, 256], pl.INT32],
    ):
        xt = pl.load(x, [0, 0], [64, 512], target_memory=pl.Mem.Mat)
        wt = pl.load(w, [0, 0], [256, 512], target_memory=pl.Mem.Mat)
        acc = pl.matmul(xt, pl.tile.transpose_view(wt), out_dtype=pl.INT32)
        pl.store(acc, [0, 0], out)
        return out


# ============================================================================
# Phase 1 — the TensorType shape is blocked
# ============================================================================


def test_nz_tensor_shape_is_blocked():
    """[256, 512] INT8 becomes [512/32, 256/16, 16, 32] = [16, 16, 16, 32]."""
    param_type = _nz_param(_run(NzMatmul))
    assert _values(param_type.shape) == [16, 16, 16, 32]


def test_leading_dims_are_preserved():
    """Only the trailing (R, C) pair is decomposed; batch dims ride along.

    Written against the tensor-level DSL (the shape the real grouped-matmul
    weight uses): a tile-level ``pl.load`` of a rank-3 window would produce a
    rank-3 tile, which ``tile.matmul`` rejects at parse time — before this pass
    ever runs.
    """
    _, _, tm, sv, sd, dyn = _batched_nz_mm._bind_args_from_signature({})
    program = _batched_nz_mm._compile_to_program(tm, sv, sd, dyn, pl)
    param_type = _nz_param(_run(program))
    assert _values(param_type.shape) == [4, 16, 16, 16, 32]


def test_nd_tensor_is_untouched():
    """An ND weight keeps its logical shape — the pass is NZ-only."""
    after = _run(NdMatmul)
    for func in after.functions.values():
        for param in func.params:
            if param.name_hint.startswith("w"):
                param_type = param.type
                assert isinstance(param_type, ir.TensorType)
                assert _values(param_type.shape) == [256, 512]


# ============================================================================
# Phase 2 — the consuming tile.load is retargeted, the tile stays 2-D
# ============================================================================


def test_tile_load_coordinates_are_blocked_and_tile_stays_2d():
    """The GM window becomes rank-4; the destination tile stays [256, 512]."""
    call = _nz_load(_run(NzMatmul))
    assert _values(call.args[1].elements) == [0, 0, 0, 0]  # offsets
    assert _values(call.args[2].elements) == [16, 16, 16, 32]  # shapes
    # The tile the load produces is the logical 2-D operand, not the GM window.
    assert _values(call.type.shape) == [256, 512]


def test_slice_offsets_are_mapped_to_fractal_coordinates():
    """A logical [n0, k0] offset becomes [k0/c0, n0/16, 0, 0]."""

    @pl.program
    class Sliced:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[64, 1024], pl.INT8],
            w: pl.Tensor[[512, 1024], pl.INT8, pl.NZ],
            out: pl.Tensor[[64, 256], pl.INT32],
        ):
            xt = pl.load(x, [0, 512], [64, 512], target_memory=pl.Mem.Mat)
            wt = pl.load(w, [256, 512], [256, 512], target_memory=pl.Mem.Mat)
            acc = pl.matmul(xt, pl.tile.transpose_view(wt), out_dtype=pl.INT32)
            pl.store(acc, [0, 0], out)
            return out

    call = _nz_load(_run(Sliced))
    # n0 = 256 -> 256/16 = 16 ; k0 = 512 -> 512/32 = 16
    assert _values(call.args[1].elements) == [16, 16, 0, 0]
    assert _values(call.args[2].elements) == [16, 16, 16, 32]


# ============================================================================
# The stride equality the whole design rests on
# ============================================================================


def test_blocked_nz_strides_match_pto_isa():
    """Row-major over the blocked shape == pto-isa's BaseShape2D<..., NZ>.

    For [256, 512] INT8 (c0 = 32) pto-isa gives ``Stride<C*R, R*c0, 16*c0, c0, 1>``
    which, with the leading batch dim dropped, is
    ``[256*32, 16*32, 32, 1] = [8192, 512, 32, 1]``.

    If this ever diverges, NZ can no longer reuse the ND row-major stride rule
    and the premise of the whole design is broken — hence a dedicated test.
    """
    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(NzMatmul)
    param_type = _nz_param(optimized)
    view = param_type.tensor_view
    assert view is not None
    assert _values(view.stride) == [8192, 512, 32, 1]


# ============================================================================
# Codegen — the emitted descriptor and its rank consistency
# ============================================================================


def test_codegen_emits_blocked_nz_descriptor():
    text = _emit_pto(NzMatmul)
    view_lines = [
        line for line in text.splitlines() if "make_tensor_view" in line and "#pto.layout<nz>" in line
    ]
    assert len(view_lines) == 1, text
    line = view_lines[0]
    assert "%c16_index, %c16_index, %c16_index, %c32_index" in line  # shape
    assert "%c8192_index, %c512_index, %c32_index, %c1_index" in line  # pto-isa NZ strides


def test_codegen_rank_is_consistent_across_all_three_sites():
    """make_tensor_view, its !pto.tensor_view type and partition_view must agree.

    Each is derived independently from ``TensorType::shape_``; a disagreement is
    what PTOAS rejects outright.
    """
    text = _emit_pto(NzMatmul)
    nz_view_line = next(
        line for line in text.splitlines() if "make_tensor_view" in line and "layout<nz>" in line
    )
    assert "!pto.tensor_view<?x?x?x?xi8>" in nz_view_line
    ssa = nz_view_line.strip().split(" ")[0]
    pview_line = next(line for line in text.splitlines() if "partition_view" in line and f"{ssa}," in line)
    assert "offsets = [%c0_index, %c0_index, %c0_index, %c0_index]" in pview_line
    assert "sizes = [%c16_index, %c16_index, %c16_index, %c32_index]" in pview_line
    assert "!pto.partition_tensor_view<16x16x16x32xi8>" in pview_line


def test_codegen_tile_keeps_logical_2d_nz_layout():
    """The destination tile is the logical 2-D NZ Mat operand."""
    text = _emit_pto(NzMatmul)
    nz_tload = [line for line in text.splitlines() if "pto.tload" in line and "rows=256, cols=512" in line]
    assert len(nz_tload) == 1, text
    assert "blayout=col_major" in nz_tload[0]
    assert "slayout=row_major" in nz_tload[0]
    assert "fractal=512" in nz_tload[0]


# ============================================================================
# Rejections — never silently mis-address
# ============================================================================


def test_rejects_unaligned_rows():
    """R must be a multiple of 16 — a partial fractal has no representation."""

    @pl.program
    class Unaligned:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[64, 512], pl.INT8],
            w: pl.Tensor[[200, 512], pl.INT8, pl.NZ],
            out: pl.Tensor[[64, 200], pl.INT32],
        ):
            xt = pl.load(x, [0, 0], [64, 512], target_memory=pl.Mem.Mat)
            wt = pl.load(w, [0, 0], [200, 512], target_memory=pl.Mem.Mat)
            acc = pl.matmul(xt, pl.tile.transpose_view(wt), out_dtype=pl.INT32)
            pl.store(acc, [0, 0], out)
            return out

    with pytest.raises(ValueError, match="multiple of 16"):
        _run(Unaligned)


def test_rejects_unaligned_cols():
    """C must be a multiple of c0 (32 elements for INT8)."""

    @pl.program
    class Unaligned:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[64, 400], pl.INT8],
            w: pl.Tensor[[256, 400], pl.INT8, pl.NZ],
            out: pl.Tensor[[64, 256], pl.INT32],
        ):
            xt = pl.load(x, [0, 0], [64, 400], target_memory=pl.Mem.Mat)
            wt = pl.load(w, [0, 0], [256, 400], target_memory=pl.Mem.Mat)
            acc = pl.matmul(xt, pl.tile.transpose_view(wt), out_dtype=pl.INT32)
            pl.store(acc, [0, 0], out)
            return out

    with pytest.raises(ValueError, match="multiple of c0"):
        _run(Unaligned)


def test_rejects_unaligned_slice_offset():
    """A slice must start on a fractal boundary."""

    @pl.program
    class BadOffset:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[64, 1024], pl.INT8],
            w: pl.Tensor[[512, 1024], pl.INT8, pl.NZ],
            out: pl.Tensor[[64, 256], pl.INT32],
        ):
            xt = pl.load(x, [0, 0], [64, 512], target_memory=pl.Mem.Mat)
            # 8 is not a multiple of the 16-row fractal.
            wt = pl.load(w, [8, 0], [256, 512], target_memory=pl.Mem.Mat)
            acc = pl.matmul(xt, pl.tile.transpose_view(wt), out_dtype=pl.INT32)
            pl.store(acc, [0, 0], out)
            return out

    with pytest.raises(ValueError, match="must be a non-negative multiple of 16"):
        _run(BadOffset)


def test_rejects_dynamic_slice_offset():
    """A loop-derived slice offset is not mapped, even when provably aligned.

    Milestone 1 maps only `ConstInt` trailing offsets: turning `nb * 256` into
    `nb * 16` for the 16-row axis needs a divisibility proof plus an algebraic
    rewrite, which is not implemented. This pins the refusal, and with it the
    gap between this milestone and the grouped-matmul weight path that motivated
    it — `n0 = nb * N_TILE` is exactly the shape that does not compile yet.
    """

    @pl.jit
    def _sym(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[512, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 512], pl.INT32]],
    ):
        for nb in pl.spmd(2, name_hint="nz_sym"):
            n0 = nb * 256  # a multiple of 16, but not a constant
            xt = pl.slice(x, [64, 512], [0, 0])
            wt = w[n0 : n0 + 256, 0:512]
            acc = pl.matmul(xt, wt, b_trans=True, out_dtype=pl.INT32)
            out[0:64, n0 : n0 + 256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sv, sd, dyn = _sym._bind_args_from_signature({})
    program = _sym._compile_to_program(tm, sv, sd, dyn, pl)
    with pytest.raises(ValueError, match="does not support a dynamic offset on shape.-2."):
        _run(program)


def test_rejects_vec_target():
    """pto-isa offers NZ->NZ into Mat for the matmul path; Vec is unimplemented."""

    @pl.program
    class VecLoad:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            w: pl.Tensor[[256, 512], pl.INT8, pl.NZ],
            out: pl.Tensor[[256, 512], pl.INT8],
        ):
            wt = pl.load(w, [0, 0], [256, 512], target_memory=pl.Mem.Vec)
            pl.store(wt, [0, 0], out)
            return out

    with pytest.raises(ValueError, match="target_memory=pl.Mem.Mat"):
        _run(VecLoad)


def test_rejects_sub_byte_dtype():
    """A 4-bit dtype has no NZ C0 line, and must not be blocked with a byte-derived c0.

    `DataType.GetByte()` is `ceil(bits/8)`, so FP4 reports 1 byte and a
    byte-based `c0` would come out as 32 instead of the 64 elements that fit in
    a 32-byte C0 line. That is doubly wrong: the trailing dim would be `[.., 16,
    32]`, and the alignment check would accept extents that are not multiples of
    64 (`C = 544` passes `% 32`). pto-isa's `TLOAD` lists no 4-bit dtype for the
    NZ path at all — 4-bit operands use the `HIF4_A_ZZ` / `HIF4_B_NN` layouts —
    so the annotation is refused rather than silently mis-addressed.

    The rejection fires in phase 1, while the *parameter* type is blocked, so a
    plain load/store kernel is enough to reach it — no matmul needed (FP4
    operands would trip the Cube accumulator's dtype rule at parse time first).
    """

    @pl.program
    class Fp4Nz:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            w: pl.Tensor[[256, 512], pl.FP4, pl.NZ],
            out: pl.Tensor[[256, 512], pl.FP4],
        ):
            wt = pl.load(w, [0, 0], [256, 512], target_memory=pl.Mem.Vec)
            pl.store(wt, [0, 0], out)
            return out

    with pytest.raises(ValueError, match="sub-byte dtype"):
        _run(Fp4Nz)


def test_c0_is_derived_from_bit_width():
    """c0 halves as the element width doubles: 32 for INT8, 16 for FP16.

    The blocked trailing dim and the stride both follow, so this pins the
    arithmetic against a regression to a byte-rounded `c0`.
    """

    @pl.program
    class Fp16Nz:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[64, 512], pl.FP16],
            w: pl.Tensor[[256, 512], pl.FP16, pl.NZ],
            out: pl.Tensor[[64, 256], pl.FP32],
        ):
            xt = pl.load(x, [0, 0], [64, 512], target_memory=pl.Mem.Mat)
            wt = pl.load(w, [0, 0], [256, 512], target_memory=pl.Mem.Mat)
            acc = pl.matmul(xt, pl.tile.transpose_view(wt), out_dtype=pl.FP32)
            pl.store(acc, [0, 0], out)
            return out

    # FP16: c0 = 256/16 = 16, so [256, 512] -> [512/16, 256/16, 16, 16].
    param_type = _nz_param(_run(Fp16Nz))
    assert _values(param_type.shape) == [32, 16, 16, 16]


def test_rejects_nz_store_destination():
    """An NZ tensor cannot be written.

    ``tile.store``'s destination is argument 2, not argument 0, so a guard that
    inspects only the first operand would let this through: phase 1 would block
    the destination's type while the store kept logical offsets, producing a
    rank-inconsistent, silently mis-addressed write instead of a diagnostic.
    """

    @pl.program
    class StoreToNz:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[256, 512], pl.INT8],
            out: pl.Tensor[[256, 512], pl.INT8, pl.NZ],
        ):
            t = pl.load(x, [0, 0], [256, 512], target_memory=pl.Mem.Vec)
            pl.store(t, [0, 0], out)
            return out

    with pytest.raises(ValueError, match="NZ layout is read-only"):
        _run(StoreToNz)


def test_rejects_tensor_view_of_nz():
    """Re-viewing a fractal decomposition would break the addressing.

    ``tensor.view`` deduces its type while the DSL body is parsed, so the
    rejection lands on the ``@pl.program`` decorator (wrapped as a
    ``ParserError``) rather than on a later pass — the whole class definition is
    therefore inside ``pytest.raises``.
    """
    with pytest.raises(ParserError, match="does not support an NZ source"):

        @pl.program
        class Viewed:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                w: pl.Tensor[[256, 512], pl.INT8, pl.NZ],
                out: pl.Tensor[[256, 512], pl.INT8],
            ):
                v = pl.tensor.view(w, [512, 256])
                wt = pl.load(v, [0, 0], [512, 256], target_memory=pl.Mem.Mat)
                pl.store(wt, [0, 0], out)
                return out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

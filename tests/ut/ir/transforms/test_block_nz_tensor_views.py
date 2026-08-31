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

from collections.abc import Sequence

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
        # The PTO backend only accepts InCore-variant functions; name them
        # positively so an SPMD wrapper is skipped as readily as an
        # Orchestration or Group one.
        if func.func_type not in (pl.FunctionType.InCore, pl.FunctionType.AIC, pl.FunctionType.AIV):
            continue
        single = ir.Program([func], func.name, optimized.span)
        result = codegen.PTOCodegen().generate(single, emit_tile_addr=True)
        parts.append(result if isinstance(result, str) else "".join(result.values()))
    return "\n".join(parts)


def _const(expr: ir.Expr) -> int:
    """The value of a `ConstInt`, asserting the expression is one."""
    assert isinstance(expr, ir.ConstInt), f"expected a ConstInt, got {type(expr).__name__}"
    return expr.value


def _values(exprs: Sequence[ir.Expr]) -> list[int]:
    return [_const(e) for e in exprs]


def _elements(expr: ir.Expr) -> Sequence[ir.Expr]:
    """The elements of a `MakeTuple` coordinate argument."""
    assert isinstance(expr, ir.MakeTuple), f"expected a MakeTuple, got {type(expr).__name__}"
    return expr.elements


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
    """Yield every statement in a body, descending into SeqStmts and loop bodies."""
    if stmt is None:
        return
    if isinstance(stmt, ir.SeqStmts):
        for inner in stmt.stmts:
            yield from _walk(inner)
        return
    if isinstance(stmt, ir.ForStmt):
        yield stmt
        yield from _walk(stmt.body)
        return
    yield stmt


def _nz_loads(program: ir.Program) -> list[ir.Call]:
    """Every tile.load whose source tensor carries the NZ layout, in body order."""
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
    return found


def _nz_load(program: ir.Program) -> ir.Call:
    """The single tile.load whose source tensor carries the NZ layout."""
    found = _nz_loads(program)
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
    assert _values(_elements(call.args[1])) == [0, 0, 0, 0]  # offsets
    assert _values(_elements(call.args[2])) == [16, 16, 16, 32]  # shapes
    # The tile the load produces is the logical 2-D operand, not the GM window.
    tile_type = call.type
    assert isinstance(tile_type, ir.TileType)
    assert _values(tile_type.shape) == [256, 512]


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
    assert _values(_elements(call.args[1])) == [16, 16, 0, 0]
    assert _values(_elements(call.args[2])) == [16, 16, 16, 32]


def test_maps_an_spmd_derived_slice_offset():
    """`n0 = nb * 256` becomes `n0 // 16` on the 16-row axis.

    The offset reaches the pass as the SSA name `n0`, not as the `Mul`, so the
    divisibility proof has to follow the definition chain to see the `* 256`.
    What it emits is the whole offset divided, never the re-associated
    `nb * 16` — see `test_does_not_reassociate_the_offset_arithmetic`.
    """

    @pl.jit
    def _spmd_offset(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[512, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 512], pl.INT32]],
    ):
        for nb in pl.spmd(2, name_hint="nz_spmd"):
            n0 = nb * 256
            xt = pl.slice(x, [64, 512], [0, 0])
            wt = w[n0 : n0 + 256, 0:512]
            acc = pl.matmul(xt, wt, b_trans=True, out_dtype=pl.INT32)
            out[0:64, n0 : n0 + 256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sv, sd, dyn = _spmd_offset._bind_args_from_signature({})
    call = _nz_load(_run(_spmd_offset._compile_to_program(tm, sv, sd, dyn, pl)))
    col_off, row_off, in_fractal_row, in_c0_line = _elements(call.args[1])
    assert _const(col_off) == 0
    assert isinstance(row_off, ir.FloorDiv)
    assert isinstance(row_off.left, ir.Var) and row_off.left.name_hint.startswith("n0")
    assert _const(row_off.right) == 16
    assert (_const(in_fractal_row), _const(in_c0_line)) == (0, 0)


def test_does_not_reassociate_the_offset_arithmetic():
    """The blocked coordinate divides the offset's *result*, never its arithmetic.

    Rewriting `n0 = nb * 256` into `nb * 16` looks equivalent and is not: IR
    arithmetic wraps at its declared width, and the re-associated form does not
    wrap at the same point. With `nb: INT32 = 1 << 24`, `nb * 256` wraps to 0 in
    i32, so the true coordinate is 0 while `nb * 16` is 268435456 — a read from
    the wrong fractal with no diagnostic anywhere downstream. Matching the
    original width does not rescue it either: for `a * b = q * 2^W + r` the
    original yields `r / d` and any re-association yields
    `r / d + q * 2^(W - log2 d)`.

    So the emitted expression must still *contain* the original offset. This
    pins that: the quotient's operand is the offset `Var` itself, and no
    multiply by the reduced factor 256/16 was synthesized.
    """

    @pl.jit
    def _no_reassoc(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[512, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 512], pl.INT32]],
    ):
        for nb in pl.spmd(2, name_hint="nz_no_reassoc"):
            n0 = nb * 256
            xt = pl.slice(x, [64, 512], [0, 0])
            wt = w[n0 : n0 + 256, 0:512]
            acc = pl.matmul(xt, wt, b_trans=True, out_dtype=pl.INT32)
            out[0:64, n0 : n0 + 256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sv, sd, dyn = _no_reassoc._bind_args_from_signature({})
    after = _run(_no_reassoc._compile_to_program(tm, sv, sd, dyn, pl))
    row_off = _elements(_nz_load(after).args[1])[1]

    # A division of the offset itself, not a product of the block index.
    assert isinstance(row_off, ir.FloorDiv), f"offset was re-associated into {type(row_off).__name__}"
    assert isinstance(row_off.left, ir.Var)

    # `n0` keeps its own definition; nothing multiplied the block index by 16.
    definitions = {
        stmt.var.name_hint: stmt.value
        for func in after.functions.values()
        for stmt in _walk(func.body)
        if isinstance(stmt, ir.AssignStmt)
    }
    n0_def = definitions[row_off.left.name_hint]
    assert isinstance(n0_def, ir.Mul) and _const(n0_def.right) == 256
    assert not [
        value
        for value in definitions.values()
        if isinstance(value, ir.Mul) and isinstance(value.right, ir.ConstInt) and value.right.value == 16
    ], "a reduced-factor multiply was synthesized"


def test_maps_a_loop_variable_slice_offset():
    """A `pl.pipeline` index divides by c0 once start and step are both multiples.

    `k0` steps 512 from 512, and both are multiples of c0 = 32, so every value it
    takes is one. Nothing in the IR names the trip count, so unlike the `nb * 256`
    case there is no folded quotient to build — the exact `k0 // 32` is the
    answer, and it is exact precisely because the divisibility was proven first.
    """

    @pl.jit
    def _loop_offset(
        x: pl.Tensor[[64, 1024], pl.INT8],
        w: pl.Tensor[[256, 1024], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
    ):
        for _ in pl.spmd(1, name_hint="nz_loop"):
            xt = pl.slice(x, [64, 512], [0, 0])
            acc = pl.matmul(xt, w[0:256, 0:512], b_trans=True, out_dtype=pl.INT32)
            for k0 in pl.pipeline(512, 1024, 512, stage=2):
                x_k = pl.slice(x, [64, 512], [0, k0])
                acc = pl.matmul_acc(acc, x_k, w[0:256, k0 : k0 + 512], b_trans=True)
            out[0:64, 0:256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sv, sd, dyn = _loop_offset._bind_args_from_signature({})
    after = _run(_loop_offset._compile_to_program(tm, sv, sd, dyn, pl))

    # Two NZ loads share the weight: the k0 = 0 prologue and the loop body.
    offsets = [_elements(call.args[1]) for call in _nz_loads(after)]
    assert len(offsets) == 2, f"expected two NZ loads, got {len(offsets)}"

    prologue, in_loop = offsets
    assert _values(prologue) == [0, 0, 0, 0]
    col_off = in_loop[0]
    assert isinstance(col_off, ir.FloorDiv)
    assert isinstance(col_off.left, ir.Var) and col_off.left.name_hint.startswith("k0")
    assert _const(col_off.right) == 32  # c0 for INT8
    assert _values(in_loop[1:]) == [0, 0, 0]


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


def test_codegen_emits_the_divided_offset():
    """The blocked coordinate reaches `partition_view` as the divided offset.

    This is the end-to-end statement of the rewrite: the descriptor is the
    blocked rank-4 NZ one, and the row-fractal coordinate is `n0 / 16` — the
    offset the kernel computed, divided, rather than a re-derived index.
    """

    @pl.jit
    def _spmd_offset(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[512, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 512], pl.INT32]],
    ):
        for nb in pl.spmd(2, name_hint="nz_spmd_cg"):
            n0 = nb * 256
            xt = pl.slice(x, [64, 512], [0, 0])
            wt = w[n0 : n0 + 256, 0:512]
            acc = pl.matmul(xt, wt, b_trans=True, out_dtype=pl.INT32)
            out[0:64, n0 : n0 + 256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sv, sd, dyn = _spmd_offset._bind_args_from_signature({})
    text = _emit_pto(_spmd_offset._compile_to_program(tm, sv, sd, dyn, pl))
    lines = text.splitlines()

    nz_view_line = next(line for line in lines if "make_tensor_view" in line and "layout<nz>" in line)
    assert "%c16_index, %c32_index, %c16_index, %c32_index" in nz_view_line  # [C/c0, R/16, 16, c0]

    ssa = nz_view_line.strip().split(" ")[0]
    pview_line = next(line for line in lines if "partition_view" in line and f"{ssa}," in line)
    assert "sizes = [%c16_index, %c16_index, %c16_index, %c32_index]" in pview_line
    # offsets = [c0-block, row-fractal, 0, 0] — only the row fractal is symbolic.
    offsets = pview_line.split("offsets = [", 1)[1].split("]", 1)[0].split(", ")
    assert [offsets[0], offsets[2], offsets[3]] == ["%c0_index"] * 3, pview_line

    # The NZ coordinate is the kernel's own offset divided: offsets[1] traces
    # back through the negative clamp to a division by 16, whose dividend is the
    # `n0 = nb * 256` the ND store on `out` also uses.
    def defining_line(operand: str) -> str:
        return next(line for line in lines if line.strip().startswith(f"{operand} = "))

    clamp = defining_line(offsets[1])
    assert "arith.maxsi" in clamp, clamp
    divide = defining_line(clamp.split("arith.maxsi ", 1)[1].split(",", 1)[0])
    assert "arith.divsi" in divide and "%c16_index" in divide, divide

    dividend = divide.split("arith.divsi ", 1)[1].split(",", 1)[0]
    offset_def = defining_line(dividend)
    assert "arith.muli" in offset_def and "%c256_index" in offset_def, offset_def


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


def test_rejects_a_slice_offset_whose_alignment_cannot_be_proven():
    """A symbolic offset that is not a provable multiple is refused, not guessed.

    `nb * 8` is a multiple of 8, never of the 16-row fractal, so no exact
    quotient exists. The pass must say so rather than divide anyway — an NZ
    tensor addressed from a guessed coordinate reads the wrong fractal with no
    diagnostic anywhere downstream.
    """

    @pl.jit
    def _unprovable(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[512, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 512], pl.INT32]],
    ):
        for nb in pl.spmd(2, name_hint="nz_unprovable"):
            n0 = nb * 8  # not a multiple of the 16-row fractal
            xt = pl.slice(x, [64, 512], [0, 0])
            wt = w[n0 : n0 + 256, 0:512]
            acc = pl.matmul(xt, wt, b_trans=True, out_dtype=pl.INT32)
            out[0:64, 0:256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sv, sd, dyn = _unprovable._bind_args_from_signature({})
    program = _unprovable._compile_to_program(tm, sv, sd, dyn, pl)
    with pytest.raises(ValueError, match=r"offset on shape\[-2\] to be a multiple of 16"):
        _run(program)


def test_rejects_a_loop_variable_whose_step_breaks_alignment():
    """A loop variable is only divisible when *both* its start and step are.

    Start 0 is a multiple of c0 = 32 while step 16 is not, so the offset is
    aligned on the first iteration and misaligned on the second. Proving from
    the start alone would silently mis-address every later iteration.
    """

    @pl.jit
    def _bad_step(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[256, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
    ):
        for _ in pl.spmd(1, name_hint="nz_bad_step"):
            for k0 in pl.pipeline(0, 32, 16, stage=2):
                xt = pl.slice(x, [64, 16], [0, k0])
                wt = w[0:256, k0 : k0 + 16]
                acc = pl.matmul(xt, wt, b_trans=True, out_dtype=pl.INT32)
                out[0:64, 0:256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sv, sd, dyn = _bad_step._bind_args_from_signature({})
    program = _bad_step._compile_to_program(tm, sv, sd, dyn, pl)
    with pytest.raises(ValueError, match=r"offset on shape\[-1\] to be a multiple of c0 = 32"):
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

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the BlockNzTensorViews pass.

``pl.Tensor[[R, C], dtype, pl.NZ]`` (or ``[[B, R, C], ...]``) asserts that the GM bytes are already
in PTO-native NZ fractal order. pto-isa describes such a buffer with a blocked
rank-5 GlobalTensor (``pto/common/pto_tile.hpp``)::

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
from pypto import DataType, ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.op import tile
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


def _emit_pto(program: ir.Program, backend_type=BackendType.Ascend910) -> str:
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


def _nz_slices(program: ir.Program) -> list[ir.Call]:
    """Every tensor.slice whose source tensor carries the NZ layout, in body order."""
    slice_name = ir.get_op("tensor.slice").name
    found = []
    for func in program.functions.values():
        for stmt in _walk(func.body):
            if not isinstance(stmt, ir.AssignStmt):
                continue
            call = stmt.value
            if not isinstance(call, ir.Call) or call.op.name != slice_name:
                continue
            view = getattr(call.args[0].type, "tensor_view", None)
            if view is not None and view.layout == ir.TensorLayout.NZ:
                found.append(call)
    return found


def _nz_reshapes(program: ir.Program) -> list[ir.Call]:
    """Every tensor.reshape whose source tensor carries the NZ layout, in body order."""
    reshape_name = ir.get_op("tensor.reshape").name
    found = []
    for func in program.functions.values():
        for stmt in _walk(func.body):
            if not isinstance(stmt, ir.AssignStmt):
                continue
            call = stmt.value
            if not isinstance(call, ir.Call) or call.op.name != reshape_name:
                continue
            view = getattr(call.args[0].type, "tensor_view", None)
            if view is not None and view.layout == ir.TensorLayout.NZ:
                found.append(call)
    return found


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
    """[256, 512] INT8 becomes [1, 512/32, 256/16, 16, 32] = [1, 16, 16, 16, 32].

    The leading 1 is pto-isa's batch slot, materialised because the logical
    tensor has no leading axis. Blocking to rank 4 instead yields a view PTOAS
    rejects outright ("layout=nz requires a rank-5 view").
    """
    param_type = _nz_param(_run(NzMatmul))
    assert _values(param_type.shape) == [1, 16, 16, 16, 32]


def test_leading_dims_are_preserved():
    """Only the trailing (R, C) pair is decomposed; batch dims ride along.

    Written against the tensor-level DSL (the shape the real grouped-matmul
    weight uses): a tile-level ``pl.load`` of a rank-3 window would produce a
    rank-3 tile, which ``tile.matmul`` rejects at parse time — before this pass
    ever runs.
    """
    _, _, tm, sd, cx, dyn = _batched_nz_mm._bind_args_from_signature({})
    program = _batched_nz_mm._compile_to_program(tm, sd, cx, dyn, pl)
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
    """The GM window becomes rank-5; the destination tile stays [256, 512]."""
    call = _nz_load(_run(NzMatmul))
    assert _values(_elements(call.args[1])) == [0, 0, 0, 0, 0]  # offsets
    assert _values(_elements(call.args[2])) == [1, 16, 16, 16, 32]  # shapes
    # The tile the load produces is the logical 2-D operand, not the GM window.
    tile_type = call.type
    assert isinstance(tile_type, ir.TileType)
    assert _values(tile_type.shape) == [256, 512]


def test_slice_offsets_are_mapped_to_fractal_coordinates():
    """A logical [n0, k0] offset becomes [0, k0/c0, n0/16, 0, 0]."""

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
    assert _values(_elements(call.args[1])) == [0, 16, 16, 0, 0]
    assert _values(_elements(call.args[2])) == [1, 16, 16, 16, 32]


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

    _, _, tm, sd, cx, dyn = _spmd_offset._bind_args_from_signature({})
    call = _nz_load(_run(_spmd_offset._compile_to_program(tm, sd, cx, dyn, pl)))
    batch_off, col_off, row_off, in_fractal_row, in_c0_line = _elements(call.args[1])
    assert _const(batch_off) == 0
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

    _, _, tm, sd, cx, dyn = _no_reassoc._bind_args_from_signature({})
    after = _run(_no_reassoc._compile_to_program(tm, sd, cx, dyn, pl))
    # Blocked offsets are [batch, C/c0, R/16, 0, 0] — index 2 is the row axis.
    row_off = _elements(_nz_load(after).args[1])[2]

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

    _, _, tm, sd, cx, dyn = _loop_offset._bind_args_from_signature({})
    after = _run(_loop_offset._compile_to_program(tm, sd, cx, dyn, pl))

    # Two NZ loads share the weight: the k0 = 0 prologue and the loop body.
    offsets = [_elements(call.args[1]) for call in _nz_loads(after)]
    assert len(offsets) == 2, f"expected two NZ loads, got {len(offsets)}"

    prologue, in_loop = offsets
    assert _values(prologue) == [0, 0, 0, 0, 0]
    # [batch, C/c0, R/16, 0, 0] — index 1 is the column-block axis k0 divides.
    assert _const(in_loop[0]) == 0
    col_off = in_loop[1]
    assert isinstance(col_off, ir.FloorDiv)
    assert isinstance(col_off.left, ir.Var) and col_off.left.name_hint.startswith("k0")
    assert _const(col_off.right) == 32  # c0 for INT8
    assert _values(in_loop[2:]) == [0, 0, 0]


# ============================================================================
# The stride equality the whole design rests on
# ============================================================================


def test_blocked_nz_strides_match_pto_isa():
    """Row-major over the blocked shape == pto-isa's BaseShape2D<..., NZ>.

    For [256, 512] INT8 (c0 = 32) pto-isa gives ``Stride<C*R, R*c0, 16*c0, c0, 1>``
    = ``[512*256, 256*32, 16*32, 32, 1] = [131072, 8192, 512, 32, 1]`` — all five,
    including the leading batch stride. The batch extent is the materialised 1,
    so that stride is never multiplied by a non-zero coordinate; it still has to
    be *present*, because pto-isa reads NZ strides at a fixed arity.

    If this ever diverges, NZ can no longer reuse the ND row-major stride rule
    and the premise of the whole design is broken — hence a dedicated test.
    """
    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(NzMatmul)
    param_type = _nz_param(optimized)
    view = param_type.tensor_view
    assert view is not None
    assert _values(view.stride) == [131072, 8192, 512, 32, 1]


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
    assert "%c1_index, %c16_index, %c16_index, %c16_index, %c32_index" in line  # shape
    # pto-isa NZ strides, batch stride included
    assert "%c131072_index, %c8192_index, %c512_index, %c32_index, %c1_index" in line


def test_codegen_rank_is_consistent_across_all_three_sites():
    """make_tensor_view, its !pto.tensor_view type and partition_view must agree.

    Each is derived independently from ``TensorType::shape_``; a disagreement is
    what PTOAS rejects outright. The rank they must agree *on* is 5 — PTOAS
    checks the NZ arity itself, so three mutually consistent rank-4 sites would
    still be refused.
    """
    text = _emit_pto(NzMatmul)
    nz_view_line = next(
        line for line in text.splitlines() if "make_tensor_view" in line and "layout<nz>" in line
    )
    assert "!pto.tensor_view<?x?x?x?x?xi8>" in nz_view_line
    ssa = nz_view_line.strip().split(" ")[0]
    pview_line = next(line for line in text.splitlines() if "partition_view" in line and f"{ssa}," in line)
    assert "offsets = [%c0_index, %c0_index, %c0_index, %c0_index, %c0_index]" in pview_line
    assert "sizes = [%c1_index, %c16_index, %c16_index, %c16_index, %c32_index]" in pview_line
    assert "!pto.partition_tensor_view<1x16x16x16x32xi8>" in pview_line


def test_codegen_emits_the_divided_offset():
    """The blocked coordinate reaches `partition_view` as the divided offset.

    This is the end-to-end statement of the rewrite: the descriptor is the
    blocked rank-5 NZ one, and the row-fractal coordinate is `n0 / 16` — the
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

    _, _, tm, sd, cx, dyn = _spmd_offset._bind_args_from_signature({})
    text = _emit_pto(_spmd_offset._compile_to_program(tm, sd, cx, dyn, pl))
    lines = text.splitlines()

    nz_view_line = next(line for line in lines if "make_tensor_view" in line and "layout<nz>" in line)
    # [batch, C/c0, R/16, 16, c0]
    assert "%c1_index, %c16_index, %c32_index, %c16_index, %c32_index" in nz_view_line

    ssa = nz_view_line.strip().split(" ")[0]
    pview_line = next(line for line in lines if "partition_view" in line and f"{ssa}," in line)
    assert "sizes = [%c1_index, %c16_index, %c16_index, %c16_index, %c32_index]" in pview_line
    # offsets = [batch, c0-block, row-fractal, 0, 0] — only the row fractal is symbolic.
    offsets = pview_line.split("offsets = [", 1)[1].split("]", 1)[0].split(", ")
    assert [offsets[0], offsets[1], offsets[3], offsets[4]] == ["%c0_index"] * 4, pview_line

    # The NZ coordinate is the kernel's own offset divided: offsets[2] traces
    # back through the negative clamp to a division by 16, whose dividend is the
    # `n0 = nb * 256` the ND store on `out` also uses.
    def defining_line(operand: str) -> str:
        return next(line for line in lines if line.strip().startswith(f"{operand} = "))

    clamp = defining_line(offsets[2])
    assert "arith.maxsi" in clamp, clamp
    divide = defining_line(clamp.split("arith.maxsi ", 1)[1].split(",", 1)[0])
    assert "arith.floordivsi" in divide and "%c16_index" in divide, divide

    dividend = divide.split("arith.floordivsi ", 1)[1].split(",", 1)[0]
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


def test_folds_leading_axes_into_the_batch_slot():
    """A logical ``[G, E, N, K]`` weight blocks with ``B = G*E``.

    pto-isa's NZ ``GlobalTensor`` has exactly one batch slot, and dense
    row-major leading axes collapse into it exactly — the fold removes only the
    strides it multiplies back in.
    """

    @pl.jit
    def _rank4_nz(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[2, 4, 256, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
    ):
        for _ in pl.spmd(1, name_hint="rank4_nz"):
            xt = pl.slice(x, [64, 512], [0, 0])
            wt = w[1:2, 3:4, 0:256, 0:512]
            acc = pl.matmul(xt, wt, b_trans=True, out_dtype=pl.INT32)
            out[0:64, 0:256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sd, cx, dyn = _rank4_nz._bind_args_from_signature({})
    after = _run(_rank4_nz._compile_to_program(tm, sd, cx, dyn, pl))
    assert _values(_nz_param(after).shape) == [8, 16, 16, 16, 32]
    call = _nz_load(after)
    # [1, 3, 0, 0] into [2, 4, ...] addresses batch 1*4 + 3 = 7.
    assert _values(_elements(call.args[1])) == [7, 0, 0, 0, 0]
    assert _values(_elements(call.args[2])) == [1, 16, 16, 16, 32]


def test_blocks_a_leading_axis_slice():
    """``pl.slice`` along the batch axis keeps the NZ view addressable.

    A layer-stacked weight reaches its kernel as one layer's window, so the
    slice has to block like the load that follows it: whole trailing matrix,
    batch offset carried through.
    """

    @pl.jit(auto_scope=False)
    def _sliced_nz(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[4, 256, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
    ):
        w_layer: pl.Tensor[[2, 256, 512], pl.INT8, pl.NZ] = pl.slice(w, [2, 256, 512], [2, 0, 0])
        with pl.scope():
            for _ in pl.spmd(1, name_hint="sliced_nz"):
                xt = pl.slice(x, [64, 512], [0, 0])
                acc = pl.matmul(xt, w_layer[1:2, 0:256, 0:512], b_trans=True, out_dtype=pl.INT32)
                out[0:64, 0:256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sd, cx, dyn = _sliced_nz._bind_args_from_signature({})
    after = _run(_sliced_nz._compile_to_program(tm, sd, cx, dyn, pl))
    slices = _nz_slices(after)
    assert len(slices) == 1
    assert _values(_elements(slices[0].args[1])) == [2, 16, 16, 16, 32]  # shapes
    assert _values(_elements(slices[0].args[2])) == [2, 0, 0, 0, 0]  # offsets
    slice_type = slices[0].type
    assert isinstance(slice_type, ir.TensorType)
    assert _values(slice_type.shape) == [2, 16, 16, 16, 32]


def test_rejects_a_leading_window_under_a_spanning_axis():
    """Two narrowed leading axes do not fold into one contiguous batch run.

    ``[2, 4, R, C]`` sliced to ``[2, 2, R, C]`` at ``[0, 1, 0, 0]`` names
    batches ``{1, 2, 5, 6}``, but the fold multiplies extents and flattens
    offsets row-major, so it would emit extent 4 at offset 1 — the run
    ``{1, 2, 3, 4}``, a different four layers read as if they were the right
    ones.
    """

    @pl.jit(auto_scope=False)
    def _two_windows(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[2, 4, 256, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
    ):
        w_win: pl.Tensor[[2, 2, 256, 512], pl.INT8, pl.NZ] = pl.slice(w, [2, 2, 256, 512], [0, 1, 0, 0])
        with pl.scope():
            for _ in pl.spmd(1, name_hint="two_windows"):
                xt = pl.slice(x, [64, 512], [0, 0])
                acc = pl.matmul(xt, w_win[0:1, 0:1, 0:256, 0:512], b_trans=True, out_dtype=pl.INT32)
                out[0:64, 0:256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sd, cx, dyn = _two_windows._bind_args_from_signature({})
    program = _two_windows._compile_to_program(tm, sd, cx, dyn, pl)
    with pytest.raises(ValueError, match="a window on one leading axis only"):
        _run(program)


def test_rejects_a_slice_inside_the_fractal_plane():
    """A row window of an NZ tensor has no contiguous stride, so it is refused.

    One layer's rows sit inside every fractal column block, so ``[layer*R, 0]``
    selects ``C/c0`` disjoint runs — exactly what the blocked view's derived
    row-major stride cannot describe.
    """

    @pl.jit(auto_scope=False)
    def _row_sliced_nz(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[512, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
    ):
        w_layer: pl.Tensor[[256, 512], pl.INT8, pl.NZ] = pl.slice(w, [256, 512], [256, 0])
        with pl.scope():
            for _ in pl.spmd(1, name_hint="row_sliced_nz"):
                xt = pl.slice(x, [64, 512], [0, 0])
                acc = pl.matmul(xt, w_layer[0:256, 0:512], b_trans=True, out_dtype=pl.INT32)
                out[0:64, 0:256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sd, cx, dyn = _row_sliced_nz._bind_args_from_signature({})
    with pytest.raises(ValueError, match="slicing the leading axes only"):
        _run(_row_sliced_nz._compile_to_program(tm, sd, cx, dyn, pl))


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

    _, _, tm, sd, cx, dyn = _unprovable._bind_args_from_signature({})
    program = _unprovable._compile_to_program(tm, sd, cx, dyn, pl)
    with pytest.raises(ValueError, match=r"offset on shape\[-2\] to be a multiple of 16"):
        _run(program)


def test_rejects_a_slice_offset_whose_sign_cannot_be_proven():
    """Divisibility is not enough — the offset must also be provably non-negative.

    `nb * 256 - 256` is a clean multiple of 16 and is negative on the first
    block. A negative offset is *clamped* to 0 at `pto.partition_view` rather
    than rejected, so it would read fractal 0 and return silently wrong data —
    the shape #2543 fixed for row indices. A literal `-16` is already refused by
    the constant path, so accepting this one would mean the same value is caught
    written inline and waved through once bound to a name.
    """

    @pl.jit
    def _maybe_negative(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[512, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
    ):
        for nb in pl.spmd(2, name_hint="nz_maybe_neg"):
            n0 = nb * 256 - 256  # a multiple of 16, but negative when nb == 0
            xt = pl.slice(x, [64, 512], [0, 0])
            wt = w[n0 : n0 + 256, 0:512]
            acc = pl.matmul(xt, wt, b_trans=True, out_dtype=pl.INT32)
            out[0:64, 0:256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sd, cx, dyn = _maybe_negative._bind_args_from_signature({})
    program = _maybe_negative._compile_to_program(tm, sd, cx, dyn, pl)
    with pytest.raises(ValueError, match=r"offset on shape\[-2\] to be non-negative"):
        _run(program)


def test_maps_a_split_k_offset_built_from_a_remainder():
    """``(block % OK) * K_SLICE`` is the split-K half of a K loop.

    A non-negative block index and a positive constant divisor make the
    remainder non-negative, and the constant factor carries the alignment.
    Without the remainder rule the split-K idiom used by attention projections
    is refused.
    """

    @pl.jit
    def _split_k(
        x: pl.Tensor[[64, 1024], pl.INT8],
        w: pl.Tensor[[256, 1024], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
    ):
        for blk in pl.spmd(4, name_hint="split_k_mm"):
            k_base = (blk % 2) * 512
            n0 = (blk // 2) * 128
            acc = pl.create_tensor([64, 128], dtype=pl.INT32)
            for kb in pl.pipeline(0, 512, 512, stage=2):
                k0 = k_base + kb
                acc = pl.matmul_acc(
                    acc,
                    x[0:64, k0 : k0 + 512],
                    w[n0 : n0 + 128, k0 : k0 + 512],
                    b_trans=True,
                    init_cond=(kb == 0),
                )
            out[:, n0 : n0 + 128] = acc
        return out

    _, _, tm, sd, cx, dyn = _split_k._bind_args_from_signature({})
    call = _nz_load(_run(_split_k._compile_to_program(tm, sd, cx, dyn, pl)))
    # Both trailing offsets are divided; neither is refused for an unprovable sign.
    assert len(_elements(call.args[1])) == 5


def test_flattens_a_whole_nz_tensor():
    """A rank-1 view of *every* element is layout-invariant, so it is kept as written.

    The blocked form permutes the index space, not the memory: both spellings
    walk the same contiguous GM range. An SDMA L2 warm
    (``prefetch.async_prefetch``) wants a flat logical-1D source, so refusing
    this would mean an NZ weight can never be prefetched — which is what the
    DeepSeek V4 o-projection weights do while they stream.
    """

    @pl.jit
    def _flatten(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[256, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
        head: pl.Out[pl.Tensor[[512], pl.INT8]],
    ):
        w_flat = pl.reshape(w, [256 * 512])
        for nb in pl.spmd(2, name_hint="nz_flat_mm"):
            n0 = nb * 128
            acc = pl.matmul(x[0:64, 0:512], w[n0 : n0 + 128, 0:512], b_trans=True, out_dtype=pl.INT32)
            out[0:64, n0 : n0 + 128] = pl.reshape(acc, [64, 128])
        for _flat_blk in pl.spmd(1, name_hint="nz_flat_head"):
            pl.store(pl.load(w_flat, [0], [512], target_memory=pl.Mem.Vec), [0], head)
        return out, head

    _, _, tm, sd, cx, dyn = _flatten._bind_args_from_signature({})
    program = _run(_flatten._compile_to_program(tm, sd, cx, dyn, pl))
    flatten = _nz_reshapes(program)
    assert len(flatten) == 1
    # The target shape is untouched, and the source now carries the blocked type.
    assert _values(_elements(flatten[0].args[1])) == [256 * 512]
    source_type = flatten[0].args[0].type
    assert isinstance(source_type, ir.TensorType)
    assert len(source_type.shape) == 5
    # The matmul operand is still blocked, so the flatten did not disable the pass.
    assert len(_elements(_nz_loads(program)[0].args[1])) == 5


def test_rejects_a_reshape_that_is_not_a_whole_tensor_flatten():
    """A partial reshape reinterprets coordinates, which the blocked form breaks.

    ``[256, 512] -> [128, 1024]`` pairs rows in logical row-major order; in the
    blocked form those elements are scattered across fractal blocks. There is no
    coordinate rewrite that makes it mean the same thing, so it is refused
    rather than silently addressed as ND.
    """

    @pl.jit
    def _partial(
        w: pl.Tensor[[256, 512], pl.INT8, pl.NZ],
        head: pl.Out[pl.Tensor[[512], pl.INT8]],
    ):
        w_half = pl.reshape(w, [128, 1024])
        for _half_blk in pl.spmd(1, name_hint="nz_partial_head"):
            pl.store(pl.load(w_half, [0, 0], [1, 512], target_memory=pl.Mem.Vec), [0], head)
        return head

    _, _, tm, sd, cx, dyn = _partial._bind_args_from_signature({})
    program = _partial._compile_to_program(tm, sd, cx, dyn, pl)
    with pytest.raises(ValueError, match="whole-tensor 'tensor.reshape' flatten"):
        _run(program)


def test_rejects_a_remainder_of_an_unproven_dividend():
    """The conservative proof still requires a non-negative dividend.

    ``FloorMod`` codegen corrects negative remainders, so ``(blk - 1) % 2``
    is 1 on the first block. This pass does not use that stronger fact: a
    difference proves no sign, so the whole offset remains unproven.
    """

    @pl.jit
    def _unproven_dividend(
        x: pl.Tensor[[64, 1024], pl.INT8],
        w: pl.Tensor[[128, 1024], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 128], pl.INT32]],
    ):
        for blk in pl.spmd(4, name_hint="unproven_mod"):
            k0 = ((blk - 1) % 2) * 512
            acc = pl.matmul(x[0:64, k0 : k0 + 512], w[0:128, k0 : k0 + 512], b_trans=True, out_dtype=pl.INT32)
            out[0:64, 0:128] = pl.reshape(acc, [64, 128])
        return out

    _, _, tm, sd, cx, dyn = _unproven_dividend._bind_args_from_signature({})
    program = _unproven_dividend._compile_to_program(tm, sd, cx, dyn, pl)
    with pytest.raises(ValueError, match=r"offset on shape\[-1\] to be non-negative"):
        _run(program)


def test_maps_an_offset_from_a_strided_loop_that_starts_at_the_block_index():
    """A loop variable's facts come from its start and step, symbolic or not.

    ``for ob in pl.range(core, TILES, CORES)`` hands each core every CORES-th
    tile starting at its own block index -- the vocab split an LM head uses. The
    start is a block index, so it is non-negative, and the step is a positive
    constant, so ``ob * 256`` is a non-negative multiple of 16. Before the start
    could be symbolic this offset was refused for its sign.
    """

    @pl.jit
    def _strided(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[1024, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 1024], pl.INT32]],
    ):
        with pl.spmd(2, name_hint="nz_strided"):
            core = pl.tile.get_block_idx()
            for ob in pl.range(core, 4, 2):
                o0 = ob * 256
                acc = pl.matmul(x[0:64, 0:512], w[o0 : o0 + 256, 0:512], b_trans=True, out_dtype=pl.INT32)
                out[0:64, o0 : o0 + 256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sd, cx, dyn = _strided._bind_args_from_signature({})
    program = _run(_strided._compile_to_program(tm, sd, cx, dyn, pl))
    assert len(_elements(_nz_loads(program)[0].args[1])) == 5


def test_blocks_a_ragged_tail_whose_valid_rows_are_a_min():
    """A valid region that covers whole fractals blocks to fewer row fractals.

    The last of four 256-row tiles over an 800-row weight holds 32 rows, named at
    run time as ``min(800 - o0, 256)``. Both operands are multiples of 16, so the
    region ends on a fractal boundary and blocks to ``valid // 16`` row
    fractals -- the one axis left dynamic in the blocked valid shape.
    """

    @pl.jit
    def _ragged(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[800, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 1024], pl.INT32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="nz_ragged"):
            for ob in pl.range(0, 4):
                o0 = ob * 256
                valid = pl.min(800 - o0, 256)
                wt = pl.slice(w, [256, 512], [o0, 0], valid_shape=[valid, 512])
                acc = pl.matmul(x[0:64, 0:512], wt, b_trans=True, out_dtype=pl.INT32)
                out[0:64, o0 : o0 + 256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sd, cx, dyn = _ragged._bind_args_from_signature({})
    program = _run(_ragged._compile_to_program(tm, sd, cx, dyn, pl))
    load = _nz_loads(program)[0]
    valid = _elements(load.args[3])
    assert len(valid) == 5
    # Column blocks, fractal rows and the C0 line stay static; the row-fractal
    # count is the run-time quotient.
    assert _values([valid[1], valid[3], valid[4]]) == [16, 16, 32]
    assert not isinstance(valid[2], ir.ConstInt)


def test_rejects_a_ragged_tail_that_ends_inside_a_fractal():
    """A dynamic valid row count must be a provable multiple of 16.

    ``min(800 - o0, 250)`` can end mid-fractal, and a partial fractal has no
    blocked form, so the load is refused rather than rounded.
    """

    @pl.jit
    def _mid_fractal(
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[800, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 1024], pl.INT32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="nz_mid_fractal"):
            for ob in pl.range(0, 4):
                o0 = ob * 256
                valid = pl.min(800 - o0, 250)
                wt = pl.slice(w, [256, 512], [o0, 0], valid_shape=[valid, 512])
                acc = pl.matmul(x[0:64, 0:512], wt, b_trans=True, out_dtype=pl.INT32)
                out[0:64, o0 : o0 + 256] = pl.reshape(acc, [64, 256])
        return out

    _, _, tm, sd, cx, dyn = _mid_fractal._bind_args_from_signature({})
    program = _mid_fractal._compile_to_program(tm, sd, cx, dyn, pl)
    with pytest.raises(ValueError, match=r"dynamic valid_shape\[-2\] to be a provable multiple of 16"):
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

    _, _, tm, sd, cx, dyn = _bad_step._bind_args_from_signature({})
    program = _bad_step._compile_to_program(tm, sd, cx, dyn, pl)
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


@pytest.mark.parametrize(
    ("dtype", "error"),
    [(pl.INT4, "sub-byte dtype"), (pl.UINT4, "sub-byte dtype"), (pl.HF4, "sub-byte dtype")],
)
def test_rejects_sub_byte_dtype(dtype, error):
    """Sub-byte dtypes cannot use the byte-based NZ C0 geometry."""

    @pl.program
    class SubByteNz:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            w: pl.Tensor[[256, 512], dtype, pl.NZ],
            out: pl.Tensor[[256, 512], dtype],
        ):
            wt = pl.load(w, [0, 0], [256, 512], target_memory=pl.Mem.Vec)
            pl.store(wt, [0, 0], out)
            return out

    with pytest.raises(ValueError, match=error):
        _run(SubByteNz)


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

    # FP16: c0 = 256/16 = 16, so [256, 512] -> [1, 512/16, 256/16, 16, 16].
    param_type = _nz_param(_run(Fp16Nz))
    assert _values(param_type.shape) == [1, 32, 16, 16, 16]


@pytest.mark.usefixtures("ascend_backend")
def test_rejects_vec_nz_store_destination():
    """The Acc writeback extension must not admit an ordinary vector store.

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

    with pytest.raises(ValueError, match="NZ"):
        _run(StoreToNz)


def _nz_store_program(
    *,
    tile_shape=(64, 16),
    valid_shape=None,
    compact=ir.CompactMode.null,
    source_dtype=DataType.FP32,
    output_dtype=DataType.FP32,
    memory_space=ir.MemorySpace.Acc,
    output_shape=(64, 64),
    offsets=(0, 16),
    partition_shape=None,
    kwargs=None,
):
    """Build a store at the blocking boundary, without running memory inference.

    A Tile parameter lets the rejection tests vary the precise physical/valid
    extent and resolved memory space. The real matmul/codegen test below covers
    the public pipeline that produces this boundary IR.
    """
    span = ir.Span.unknown()
    tile_view = None
    if valid_shape is not None or compact != ir.CompactMode.null:
        tile_view = ir.TileView(
            valid_shape=valid_shape,
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
            compact=compact,
        )
    source = ir.Var(
        "acc", ir.TileType(tile_shape, source_dtype, tile_view=tile_view, memory_space=memory_space), span
    )
    output_type = ir.TensorType(
        output_shape, output_dtype, tensor_view=ir.TensorView(layout=ir.TensorLayout.NZ)
    )
    output = ir.Var("out", output_type, span)

    def coordinates(values):
        return ir.MakeTuple(
            [
                ir.ConstInt(value, DataType.INDEX, span) if isinstance(value, int) else value
                for value in values
            ],
            span,
        )

    args = [source, coordinates(offsets), output]
    if partition_shape is not None:
        args.append(coordinates(partition_shape))
    call = ir.Call(
        ir.get_op("tile.store"), args, kwargs or {}, {"nz_store_test_tag": "preserve"}, output_type, span
    )
    result = ir.Var("stored", output_type, span)
    body = ir.SeqStmts([ir.AssignStmt(result, call, span), ir.ReturnStmt([result], span)], span)
    # Unknown offsets are legal scalar inputs. Keeping them bound ensures a
    # rejection tests the NZ proof requirement rather than malformed SSA.
    parameters = [source, output, *(value for value in offsets if isinstance(value, ir.Var))]
    function = ir.Function("main", parameters, [output_type], body, span, ir.FunctionType.InCore)
    return ir.Program([function], "NzStore", span)


def _store(program: ir.Program) -> ir.AssignStmt:
    """The sole store and its assignment, with the same traversal as NZ loads."""
    store_name = ir.get_op("tile.store").name
    found = [
        stmt
        for func in program.functions.values()
        for stmt in _walk(func.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == store_name
    ]
    assert len(found) == 1
    return found[0]


def _store_call(program: ir.Program) -> ir.Call:
    call = _store(program).value
    assert isinstance(call, ir.Call)
    return call


def _shape_values(node: ir.Expr | ir.Type) -> list[int]:
    node_type = node.type if isinstance(node, ir.Expr) else node
    assert isinstance(node_type, (ir.TileType, ir.TensorType))
    return _values(node_type.shape)


@pytest.mark.parametrize(
    ("output_shape", "offsets", "partition_shape", "blocked_shape", "blocked_offsets"),
    [
        ((64, 64), (0, 16), None, [1, 8, 4, 16, 8], [0, 2, 0, 0, 0]),
        ((3, 64, 64), (2, 0, 16), (1, 64, 16), [3, 8, 4, 16, 8], [2, 2, 0, 0, 0]),
    ],
)
@pytest.mark.usefixtures("ascend_backend")
def test_blocks_acc_nz_store_and_preserves_its_alias(
    output_shape, offsets, partition_shape, blocked_shape, blocked_offsets
):
    """GM coordinates become rank five while the source and returned alias stay intact.

    The rank-three case models FlattenTileNdTo2D's explicit logical partition:
    replacing that operand with a fresh 2-D window would lose its batch slot.
    """
    before = _nz_store_program(
        output_shape=output_shape,
        offsets=offsets,
        partition_shape=partition_shape,
        kwargs={"st_phase": int(pl.STPhase.Final)},
    )
    after = passes.block_nz_tensor_views()(before)
    original = _store_call(before)
    assignment = _store(after)
    store = assignment.value
    assert isinstance(store, ir.Call)
    function = next(iter(after.functions.values()))
    destination = function.params[1]
    assert _shape_values(destination) == blocked_shape
    assert store.args[0] is original.args[0]
    assert _shape_values(store.args[0]) == [64, 16]
    assert store.args[2] is destination
    assert _values(_elements(store.args[1])) == blocked_offsets
    assert _values(_elements(store.args[3])) == [1, 2, 4, 16, 8]
    assert dict(store.kwargs) == dict(original.kwargs)
    assert dict(store.attrs) == dict(original.attrs)
    ir.assert_structural_equal(assignment.var.type, destination.type)
    ir.assert_structural_equal(function.return_types[0], destination.type)
    # The stamp must prevent a second invocation from treating the blocked
    # trailing [16,8] as a new logical matrix and blocking it again.
    ir.assert_structural_equal(passes.block_nz_tensor_views()(after), after)


@pytest.mark.usefixtures("ascend_backend")
def test_acc_nz_store_remaps_expression_valued_attrs_with_the_destination():
    """An attribute referencing the old output must follow the rewritten operand.

    Copying the attribute dictionary unchanged would leave a logical-rank Var
    attached to a blocked-rank store, even though the ordinary args are correct.
    """

    class AddOutputReference(ir.IRMutator):
        def visit_call(self, op: ir.Call) -> ir.Expr:
            call = op
            if call.op.name != ir.get_op("tile.store").name:
                return super().visit_call(call)
            return ir.Call(
                call.op,
                call.args,
                dict(call.kwargs),
                {"nz_output_ref": call.args[2]},
                call.type,
                call.span,
            )

    before = AddOutputReference().visit_program(_nz_store_program())
    old_output = _store_call(before).args[2]
    # The synthetic compiler attribute is not a public store keyword.
    with passes.PassContext([]):
        after = passes.block_nz_tensor_views()(before)
    store = _store_call(after)
    assert store.args[2] is not old_output
    assert store.attrs["nz_output_ref"] is store.args[2]
    assert _shape_values(store.attrs["nz_output_ref"]) == [1, 8, 4, 16, 8]


@pytest.mark.usefixtures("ascend_backend")
def test_acc_nz_store_accepts_compact_full_tiles():
    """For a fully valid aligned tile, compact and physical row pitches coincide."""
    before = _nz_store_program(compact=ir.CompactMode.normal)
    after = passes.block_nz_tensor_views()(before)
    store = _store_call(after)
    source_type = store.args[0].type
    assert isinstance(source_type, ir.TileType)
    assert source_type.get_effective_tile_view().compact == ir.CompactMode.normal
    assert _shape_values(source_type) == [64, 16]
    assert _values(_elements(store.args[3])) == [1, 2, 4, 16, 8]


@pytest.mark.parametrize(("destination_cols", "column_offset"), [(512, 0), (2048, 512)])
@pytest.mark.usefixtures("ascend_backend")
def test_acc_nz_store_accepts_exact_l0c_capacity(destination_cols, column_offset):
    """Capacity bounds the source Acc allocation, not the entire GM destination.

    On A2/A3, 64*512*4 bytes exactly fills the 128 KiB L0C. The same tile
    can write a column panel of a larger NZ tensor without a larger Acc.
    """
    before = _nz_store_program(
        tile_shape=(64, 512), output_shape=(64, destination_cols), offsets=(0, column_offset)
    )
    store = _store_call(passes.block_nz_tensor_views()(before))
    assert _shape_values(store.args[0]) == [64, 512]
    assert _shape_values(store.args[2]) == [1, destination_cols // 8, 4, 16, 8]
    assert _values(_elements(store.args[1])) == [0, column_offset // 8, 0, 0, 0]
    assert _values(_elements(store.args[3])) == [1, 64, 4, 16, 8]


@pytest.mark.parametrize(("source_cols", "required_bytes"), [(528, 135168), (2048, 524288)])
@pytest.mark.usefixtures("ascend_backend")
def test_acc_nz_store_defers_capacity_until_the_physical_codegen_boundary(source_cols, required_bytes):
    """Blocking accepts logical extents; an unretiled physical source is still refused.

    This source is a Tile parameter, with no matmul producer for AutoTile to
    split. The ordinary matmul tests below cover the supported producer path;
    bypassing that tiler must not allow an oversized physical Acc to reach ISA.
    """
    before = _nz_store_program(tile_shape=(64, source_cols), output_shape=(64, source_cols), offsets=(0, 0))
    blocked = passes.block_nz_tensor_views()(before)
    assert _shape_values(_store_call(blocked).args[0]) == [64, source_cols]
    with pytest.raises(ValueError, match="NZ tile.store") as error:
        codegen.PTOCodegen().generate(blocked, emit_tile_addr=True)
    message = str(error.value)
    assert "L0C" in message
    assert str(required_bytes) in message
    assert "131072" in message
    assert "pl.ND" in message


def _wide_acc_nz_program(rows=64, cols=2048, column_offset=0, batch=None, split_k=False, layout=pl.NZ):
    """A real producer lets the default pipeline legalize the oversized Acc."""
    output_cols = cols + 32 if column_offset else cols
    output_shape = [rows, output_cols] if batch is None else [3, rows, output_cols]
    offsets = [0, column_offset] if batch is None else [batch, 0, column_offset]
    reduction = 128 if split_k else 16

    if split_k:

        @pl.program
        class SplitK:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                x: pl.Tensor[[rows, reduction], pl.FP16],
                w: pl.Tensor[[reduction, cols], pl.FP16],
                out: pl.Tensor[output_shape, pl.FP32, layout],
            ):
                # Keep only one K-panel in L1; loading all 128x2048 weights at
                # once would exceed L1 before the L0 tiler could help.
                initial = pl.tile.create([rows, cols], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                for ko, (carry,) in pl.pipeline(0, reduction, 32, init_values=(initial,), stage=2):
                    lhs = pl.load(x, [0, ko], [rows, 32], target_memory=pl.Mem.Mat)
                    rhs = pl.load(w, [ko, 0], [32, cols], target_memory=pl.Mem.Mat)
                    current = pl.tile.matmul_acc(carry, lhs, rhs, ko == 0)
                    acc = pl.yield_(current)
                return pl.store(acc, offsets, out)

        return SplitK

    @pl.program
    class FullK:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[rows, reduction], pl.FP16],
            w: pl.Tensor[[reduction, cols], pl.FP16],
            out: pl.Tensor[output_shape, pl.FP32, layout],
        ):
            lhs = pl.load(x, [0, 0], [rows, reduction], target_memory=pl.Mem.Mat)
            rhs = pl.load(w, [0, 0], [reduction, cols], target_memory=pl.Mem.Mat)
            return pl.store(pl.matmul(lhs, rhs, out_dtype=pl.FP32), offsets, out)

    return FullK


def _assert_nz_column_grid(program, rows, cols, column_offset=0, batch=0):
    """Enumerate the generated static grid to check coverage without fixing its tile N.

    These kernels have constant loop bounds. Evaluating only index arithmetic
    and loop induction variables proves each C0 column is written once, even
    when the chooser changes its tile width or uses a loop plus a boundary tail.
    Numerical matmul and the physical byte layout are checked separately on NPU.
    """
    covered = [0] * (cols // 8)
    bindings = {}
    store_name = ir.get_op("tile.store").name

    def value(expr):
        if isinstance(expr, ir.ConstInt):
            return expr.value
        if isinstance(expr, ir.Var):
            assert expr in bindings, f"unbound generated index {expr.name_hint}"
            return bindings[expr]
        if isinstance(expr, ir.Add):
            return value(expr.left) + value(expr.right)
        if isinstance(expr, ir.Sub):
            return value(expr.left) - value(expr.right)
        if isinstance(expr, ir.Mul):
            return value(expr.left) * value(expr.right)
        if isinstance(expr, ir.FloorDiv):
            return value(expr.left) // value(expr.right)
        raise AssertionError(f"unexpected generated index expression: {type(expr).__name__}")

    def check_store(op):
        assert len(op.args) == 4
        source = op.args[0].type
        assert isinstance(source, ir.TileType)
        shape = _values(source.shape)
        valid = _values(source.get_effective_tile_view().valid_shape)
        assert shape == valid
        assert shape[0] == rows and shape[1] % 16 == 0
        assert shape[1] <= 4095 and shape[0] * shape[1] * 4 <= 131072
        offsets = [value(expr) for expr in _elements(op.args[1])]
        sizes = _values(_elements(op.args[3]))
        assert offsets[0] == batch and offsets[2:] == [0, 0, 0]
        assert sizes == [1, shape[1] // 8, rows // 16, 16, 8]
        begin = offsets[1] - column_offset // 8
        end = begin + sizes[1]
        assert 0 <= begin < end <= len(covered)
        for index in range(begin, end):
            covered[index] += 1

    def execute(op):
        # Re-entering visit_stmt from a Python visit_for_stmt override leaves
        # that nanobind trampoline active: a nested For falls back to the C++
        # visitor, skips this evaluator's induction binding, and visits its
        # body only once. Walk the statement structure directly so every loop
        # is interpreted and index bindings retain their actual Var identity.
        if isinstance(op, ir.SeqStmts):
            for stmt in op.stmts:
                execute(stmt)
        elif isinstance(op, ir.ForStmt):
            for index in range(value(op.start), value(op.stop), value(op.step)):
                bindings[op.loop_var] = index
                execute(op.body)
            bindings.pop(op.loop_var, None)
        elif isinstance(op, (ir.AssignStmt, ir.EvalStmt)):
            expression = op.value if isinstance(op, ir.AssignStmt) else op.expr
            if (
                isinstance(op, ir.AssignStmt)
                and isinstance(op.var.type, ir.ScalarType)
                and op.var.type.dtype == DataType.INDEX
            ):
                bindings[op.var] = value(expression)
            if isinstance(expression, ir.Call) and expression.op.name == store_name:
                check_store(expression)
        else:
            assert isinstance(op, (ir.YieldStmt, ir.ReturnStmt)), type(op).__name__

    for function in program.functions.values():
        execute(function.body)
    assert covered == [1] * len(covered), "the NZ output grid has a gap or overlapping writes"
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.SSAForm)
    properties.insert(passes.IRProperty.UseAfterDef)
    passes.verify_properties(properties, program, "NZ automatic column tiling")


@pytest.mark.parametrize(
    ("cols", "column_offset", "batch"),
    [(528, 0, None), (2048, 0, None), (4096, 0, None), (2048, 8, None), (2048, 8, 2)],
    ids=["boundary-tail", "review-64x2048", "logical-4096-cols", "nonzero-col", "nonzero-batch-and-col"],
)
@pytest.mark.usefixtures("ascend_backend")
def test_acc_nz_store_automatically_tiles_only_columns(cols, column_offset, batch):
    """Oversized logical Accs become legal full-row stores in blocked coordinates."""
    before = _wide_acc_nz_program(cols=cols, column_offset=column_offset, batch=batch)
    blocked = _run(before)
    assert _shape_values(_store_call(blocked).args[0]) == [64, cols]
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        tiled = passes.auto_tile_matmul_l0()(passes.legalize_tile_cast()(blocked))
    _assert_nz_column_grid(tiled, 64, cols, column_offset, 0 if batch is None else batch)
    text = _emit_pto(before, BackendType.Ascend910B)
    assert "#pto.layout<nz>" in text and "pto.tstore" in text
    assert f"rows=64, cols={cols}," not in text, "the oversized physical Acc survived allocation"


@pytest.mark.usefixtures("ascend_backend")
def test_acc_nz_store_automatically_tiles_canonical_split_k():
    """The output grid encloses the complete K reduction, so partial sums never store."""
    before = _wide_acc_nz_program(column_offset=8, split_k=True)
    blocked = _run(before)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        tiled = passes.auto_tile_matmul_l0()(passes.legalize_tile_cast()(blocked))
    _assert_nz_column_grid(tiled, 64, 2048, column_offset=8)
    assert "pl.tile.matmul_acc(" in ir.python_print(tiled)
    text = _emit_pto(before, BackendType.Ascend910B)
    assert "#pto.layout<nz>" in text and "pto.tstore" in text
    assert "rows=64, cols=2048," not in text


@pytest.mark.parametrize("representation", ["logical-nz-3", "logical-nz-4", "blocked-nz-4", "nd-3"])
@pytest.mark.usefixtures("ascend_backend")
def test_auto_tiled_direct_store_preserves_expression_attrs(representation):
    """All supported spellings remap metadata to each new source and output chain.

    Production reaches AutoTile with blocked NZ coordinates; isolated-pass
    callers can still supply logical NZ coordinates, with or without an
    explicit window. ND uses type deduction to retain its valid-region union.
    A shared Call rebuild must preserve those representations and update Expr
    metadata rather than retain references to the eliminated wide accumulator.
    """
    is_nz = representation != "nd-3"
    batch = 2 if representation in ("logical-nz-4", "blocked-nz-4") else None
    before = _wide_acc_nz_program(column_offset=8, batch=batch, layout=pl.NZ if is_nz else pl.ND)
    for factory in _PREFIX[:-1]:
        before = factory()(before)
    if representation == "blocked-nz-4":
        before = passes.block_nz_tensor_views()(before)
    expected_arity = 4 if representation.endswith("-4") else 3
    assert len(_store_call(before).args) == expected_arity

    class AttachStoreMetadata(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name != ir.get_op("tile.store").name:
                return super().visit_call(op)
            return ir.Call(
                op.op,
                op.args,
                {**op.kwargs, "st_phase": int(pl.STPhase.Unspecified)},
                {
                    **op.attrs,
                    "source_ref": op.args[0],
                    "out_ref": op.args[2],
                    "dump_vars": [op.args[0], op.args[2]],
                    "scalar_tag": 3036,
                    "bool_tag": True,
                    "string_tag": "store-metadata",
                },
                op.type,
                op.span,
            )

    # These are synthetic compiler attributes, so skip printer round-tripping
    # unknown metadata keys while retaining explicit SSA/UseAfterDef checks.
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        annotated = AttachStoreMetadata().visit_program(before)
        original = _store_call(annotated)
        tiled = passes.auto_tile_matmul_l0()(annotated)
        stores = [
            stmt.value
            for function in tiled.functions.values()
            for stmt in _walk(function.body)
            if isinstance(stmt, ir.AssignStmt)
            and isinstance(stmt.value, ir.Call)
            and stmt.value.op.name == ir.get_op("tile.store").name
        ]
        assert stores
        for call in stores:
            assert len(call.args) == expected_arity
            assert dict(call.kwargs) == dict(original.kwargs)
            assert call.attrs["source_ref"].same_as(call.args[0])
            assert call.attrs["out_ref"].same_as(call.args[2])
            # dump_vars is stored as vector<VarPtr>, whereas the ordinary
            # source/out fields are ExprPtr. Both must retain their payload
            # kinds and follow the same terminal SSA replacements.
            refs = call.attrs["dump_vars"]
            assert len(refs) == 2
            assert all(isinstance(ref, ir.Var) for ref in refs)
            assert refs[0].same_as(call.args[0])
            assert refs[1].same_as(call.args[2])
            assert call.attrs["scalar_tag"] == 3036
            assert call.attrs["bool_tag"] is True
            assert call.attrs["string_tag"] == "store-metadata"
            assert not call.args[0].same_as(original.args[0])
            ir.assert_structural_equal(call.type, call.args[2].type)
            if representation == "logical-nz-4":
                shape = _shape_values(call.args[0])
                assert _values(_elements(call.args[3])) == [1, shape[0], shape[1]]
        if is_nz:
            blocked = passes.block_nz_tensor_views()(tiled)
            _assert_nz_column_grid(blocked, 64, 2048, column_offset=8, batch=0 if batch is None else batch)
        else:
            properties = passes.IRPropertySet()
            properties.insert(passes.IRProperty.SSAForm)
            properties.insert(passes.IRProperty.UseAfterDef)
            passes.verify_properties(properties, tiled, "ND direct-store metadata remapping")


@pytest.mark.parametrize("layout", [ir.TensorLayout.ND, ir.TensorLayout.NZ])
@pytest.mark.parametrize("partition", [False, True])
@pytest.mark.parametrize("explicit_layout", [False, True])
@pytest.mark.usefixtures("ascend_backend")
def test_store_layout_is_independent_of_arity_and_tile_configuration(layout, partition, explicit_layout):
    """GM layout comes from the tensor; the source remains an NZ-boxed Acc.

    A fourth store argument is a destination partition in either layout. It
    never changes the source's physical shape or its implicit/explicit config.
    """
    span = ir.Span.unknown()
    config = (
        ir.TileView(blayout=ir.TileLayout.col_major, slayout=ir.TileLayout.row_major, fractal=1024)
        if explicit_layout
        else None
    )
    source = ir.Var(
        "acc", ir.TileType([64, 16], DataType.FP32, tile_view=config, memory_space=ir.MemorySpace.Acc), span
    )
    output_type = ir.TensorType([64, 64], DataType.FP32, tensor_view=ir.TensorView(layout=layout))
    output = ir.Var("out", output_type, span)
    # Exercise the registry's actual 3/4-argument type deduction in both layouts.
    store = tile.store(source, [0, 16], output, [64, 16] if partition else None, span=span)
    result = ir.Var("stored", store.type, span)
    body = ir.SeqStmts([ir.AssignStmt(result, store, span), ir.ReturnStmt([result], span)], span)
    function = ir.Function("main", [source, output], [store.type], body, span, ir.FunctionType.InCore)
    before = ir.Program([function], "StoreLayouts", span)
    after = passes.block_nz_tensor_views()(before)
    call = _store_call(after)
    target = call.args[2].type
    source = call.args[0].type
    assert isinstance(target, ir.TensorType)
    assert isinstance(source, ir.TileType)
    assert (target.tensor_view.layout if target.tensor_view else ir.TensorLayout.ND) == layout
    assert len(call.args) == (4 if layout == ir.TensorLayout.NZ or partition else 3)
    assert _shape_values(call.args[0]) == [64, 16]
    view = source.get_effective_tile_view()
    assert (view.blayout, view.slayout, view.fractal) == (
        ir.TileLayout.col_major,
        ir.TileLayout.row_major,
        1024,
    )
    assert _values(view.valid_shape) == [64, 16]
    assert _shape_values(call.args[2]) == ([1, 8, 4, 16, 8] if layout == ir.TensorLayout.NZ else [64, 64])


@pytest.mark.usefixtures("ascend_backend")
def test_registered_rebuild_remaps_canonical_kwargs_attrs_and_span():
    """Rebuilt MAD metadata must follow clone and narrowing in the same SSA.

    Matmul has an open metadata schema, unlike store's closed kwargs. PTOAS's
    canonical compatibility path retains the rebuilt MAD calls, letting this
    check observe their Expr-valued kwargs without expanding the K emitter's
    separate metadata policy.
    """
    before = _run(_wide_acc_nz_program(column_offset=8, split_k=True))
    seed = next(
        stmt.var
        for function in before.functions.values()
        for stmt in _walk(function.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("tile.create").name
    )
    source_span = ir.Span("registered_call_rebuild.py", 17, 4)

    class AttachMadMetadata(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name != ir.get_op("tile.matmul_acc").name:
                return super().visit_call(op)
            return ir.Call(
                op.op,
                op.args,
                {
                    **op.kwargs,
                    "lhs_ref": op.args[1],
                    "rhs_ref": op.args[2],
                    "carry_ref": op.args[0],
                    "seed_ref": seed,
                },
                {**op.attrs, "lhs_ref": op.args[1], "rhs_ref": op.args[2]},
                op.type,
                source_span,
            )

    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
        tiled = passes.auto_tile_matmul_l0()(AttachMadMetadata().visit_program(before))
    calls = [
        stmt.value
        for function in tiled.functions.values()
        for stmt in _walk(function.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("tile.matmul_acc").name
    ]
    assert calls
    for call in calls:
        for key, position in (("carry_ref", 0), ("lhs_ref", 1), ("rhs_ref", 2)):
            assert call.kwargs[key].same_as(call.args[position])
        assert call.attrs["lhs_ref"].same_as(call.args[1])
        assert call.attrs["rhs_ref"].same_as(call.args[2])
        carry = call.args[0]
        assert isinstance(carry, ir.IterArg)
        assert call.kwargs["seed_ref"].same_as(carry.initValue)
        assert not call.kwargs["seed_ref"].same_as(seed)
        assert _shape_values(call) == [64, _shape_values(call.args[2])[1]]
        assert _shape_values(call)[1] < 2048
        assert call.span.filename == source_span.filename
        assert call.span.begin_line == source_span.begin_line
    _assert_nz_column_grid(tiled, 64, 2048, column_offset=8)


@pytest.mark.usefixtures("ascend_backend")
def test_acc_nz_store_rejects_when_no_full_row_column_tile_fits():
    """Even the minimum 16-column Acc exceeds L0C at 2064 complete rows."""
    before = _wide_acc_nz_program(rows=2064, cols=16)
    with pytest.raises(ValueError, match="NZ") as error:
        _emit_pto(before, BackendType.Ascend910B)
    message = str(error.value)
    assert "row" in message.lower()
    assert "pl.ND" in message


@pytest.mark.usefixtures("ascend_backend")
def test_acc_nz_store_full_row_tiling_keeps_stationary_a_candidates():
    """At M=2048 only stationary-A can keep full M while fitting a legal N tile."""
    before = _wide_acc_nz_program(rows=2048, cols=32)
    blocked = _run(before)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        tiled = passes.auto_tile_matmul_l0()(passes.legalize_tile_cast()(blocked))
    _assert_nz_column_grid(tiled, 2048, 32)
    assert "pto.tstore" in _emit_pto(before, BackendType.Ascend910B)


@pytest.mark.usefixtures("ascend_backend")
def test_acc_nz_store_exact_l0c_capacity_codegen():
    """The exact capacity boundary still compiles through the complete pipeline."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[64, 16], pl.FP16],
            w: pl.Tensor[[16, 512], pl.FP16],
            out: pl.Tensor[[64, 512], pl.FP32, pl.NZ],
        ):
            lhs = pl.load(x, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
            rhs = pl.load(w, [0, 0], [16, 512], target_memory=pl.Mem.Mat)
            return pl.store(pl.matmul(lhs, rhs, out_dtype=pl.FP32), [0, 0], out)

    text = _emit_pto(Before, BackendType.Ascend910B)
    tstore = next(line for line in text.splitlines() if "pto.tstore" in line)
    assert "#pto.layout<nz>" in text
    assert "loc=acc" in tstore
    assert "rows=64, cols=512" in tstore
    assert "!pto.partition_tensor_view<1x64x4x16x8xf32>" in tstore


def _acc_nz_store_program():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[64, 64], pl.FP16],
            w: pl.Tensor[[64, 16], pl.FP16],
            out: pl.Tensor[[64, 32], pl.FP32, pl.NZ],
        ):
            lhs = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
            rhs = pl.load(w, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
            acc = pl.matmul(lhs, rhs, out_dtype=pl.FP32)
            stored = pl.store(acc, [0, 16], out)
            return stored

    return Before


@pytest.mark.usefixtures("ascend_backend")
def test_acc_nz_store_codegen_uses_nz_destination_and_2d_accumulator():
    """A real matmul writes its FP32 Acc directly through a five-dimensional NZ view."""
    before = _acc_nz_store_program()
    after = _run(before)
    store = _store_call(after)
    assert _values(_elements(store.args[1])) == [0, 2, 0, 0, 0]
    assert _values(_elements(store.args[3])) == [1, 2, 4, 16, 8]
    assert _shape_values(store.args[0]) == [64, 16]
    text = _emit_pto(before, BackendType.Ascend910B)
    assert "#pto.layout<nz>" in text
    tstore = next(line for line in text.splitlines() if "pto.tstore" in line)
    assert "loc=acc" in tstore
    assert "rows=64, cols=16" in tstore
    assert "!pto.partition_tensor_view<1x2x4x16x8xf32>" in tstore
    assert sum("pto.tload" in line for line in text.splitlines()) == 2


@pytest.mark.usefixtures("ascend_backend")
def test_codegen_rejects_a_store_split_into_partial_parent_rows_after_blocking():
    """A later row split must not bypass the full-parent-row FIXPIPE restriction.

    Model the split after the complete legal pipeline: a half-height slice
    aliases the original Acc allocation and the store window matches that
    smaller source, but the NZ parent still has all 64 rows. Only the final
    emission check can catch this new unsupported destination stride.
    """
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(_acc_nz_store_program())

    class SplitStoreRows(ir.IRMutator):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> ir.Stmt:
            stmt = op
            call = stmt.value
            if not isinstance(call, ir.Call) or call.op.name != ir.get_op("tile.store").name:
                return super().visit_assign_stmt(stmt)
            sliced = ir.op.tile.slice(call.args[0], [32, 16], [0, 0])
            half = ir.Var("half_acc", sliced.type, stmt.span)
            sizes = ir.MakeTuple(
                [ir.ConstInt(value, DataType.INDEX, stmt.span) for value in [1, 2, 2, 16, 8]],
                stmt.span,
            )
            store = ir.Call(
                call.op,
                [half, call.args[1], call.args[2], sizes],
                dict(call.kwargs),
                dict(call.attrs),
                call.type,
                call.span,
            )
            return ir.SeqStmts(
                [ir.AssignStmt(half, sliced, stmt.span), ir.AssignStmt(stmt.var, store, stmt.span)],
                stmt.span,
            )

    split = SplitStoreRows().visit_program(optimized)
    kernel = next(func for func in split.functions.values() if ir.is_incore_type(func.func_type))
    with pytest.raises(ValueError, match="NZ tile.store must cover all rows"):
        codegen.PTOCodegen().generate(ir.Program([kernel], "SplitNzStore", split.span), emit_tile_addr=True)


@pytest.mark.usefixtures("ascend_backend")
def test_acc_nz_store_accepts_provably_aligned_loop_columns():
    """Loop facts prove sign/alignment without specializing the column offset."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[64, 64], pl.FP16],
            w: pl.Tensor[[64, 16], pl.FP16],
            out: pl.Tensor[[64, 64], pl.FP32, pl.NZ],
        ):
            lhs = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
            rhs = pl.load(w, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
            acc = pl.matmul(lhs, rhs, out_dtype=pl.FP32)
            for column in pl.range(0, 64, 16):
                out = pl.store(acc, [0, column], out)
            return out

    after = _run(Before)
    store = _store_call(after)
    column = _elements(store.args[1])[1]
    assert isinstance(column, ir.FloorDiv)
    assert isinstance(column.left, ir.Var)
    assert _const(column.right) == 8
    assert _values(_elements(store.args[3])) == [1, 2, 4, 16, 8]
    loop = next(
        stmt for func in after.functions.values() for stmt in _walk(func.body) if isinstance(stmt, ir.ForStmt)
    )
    assert len(loop.iter_args) == len(loop.return_vars) == 1
    carried = loop.iter_args[0]
    assert store.args[2] is carried
    for tensor in (carried, carried.initValue, loop.return_vars[0]):
        assert _shape_values(tensor) == [1, 8, 4, 16, 8]


@pytest.mark.parametrize(
    "arguments",
    [
        {"memory_space": ir.MemorySpace.Vec},
        {"source_dtype": DataType.INT32},
        {"output_dtype": DataType.FP16},
        {"valid_shape": (48, 16)},
        {"tile_shape": (64, 32), "valid_shape": (64, 16)},
        {"tile_shape": (64, 8)},  # C0=8 alone does not satisfy the ISA's N%16 contract.
        {"tile_shape": (32, 16)},  # A row window has a FIXPIPE destination stride the ISA cannot express.
        {"output_shape": (128, 64), "offsets": (64, 16)},
        {"output_shape": (3, 64, 64), "offsets": (0, 0, 16), "partition_shape": (2, 64, 16)},
        {"offsets": (0, 3)},
        {"offsets": (0, -16)},
        {"kwargs": {"atomic": int(pl.AtomicType.Add)}},
        {"kwargs": {"pre_quant": 1.0}},
        {"kwargs": {"pre_relu": True}},
    ],
    ids=[
        "vec",
        "int32-source",
        "dtype-conversion",
        "partial-valid",
        "partial-valid-columns",
        "eight-cols",
        "partial-parent-rows",
        "nonzero-row-offset",
        "multiple-panels",
        "unaligned-cols",
        "negative-cols",
        "atomic",
        "pre-quant",
        "pre-relu",
    ],
)
@pytest.mark.usefixtures("ascend_backend")
def test_rejects_unsupported_acc_nz_store(arguments):
    """Aligned coordinates alone do not make an unsupported FIXPIPE mode safe."""
    before = _nz_store_program(**arguments)
    with pytest.raises(ValueError, match="NZ"):
        passes.block_nz_tensor_views()(before)


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
def test_rejects_acc_nz_store_on_a5(ascend_backend):
    with pytest.raises(ValueError, match="NZ"):
        passes.block_nz_tensor_views()(_nz_store_program())


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
def test_isolated_auto_tiling_rejects_unsupported_nz_store_backend(ascend_backend):
    """The isolated logical-NZ entry must diagnose an unsupported backend too."""
    before = _wide_acc_nz_program()
    for factory in _PREFIX[:-1]:
        before = factory()(before)
    with pytest.raises(ValueError, match="NZ tile.store is supported only on A2/A3"):
        passes.auto_tile_matmul_l0()(before)


@pytest.mark.usefixtures("ascend_backend")
def test_rejects_acc_nz_store_with_unknown_column_offset():
    """An arbitrary INDEX parameter is neither aligned nor provably non-negative."""
    column = ir.Var("column", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    with pytest.raises(ValueError, match="NZ"):
        passes.block_nz_tensor_views()(_nz_store_program(offsets=(0, column)))


@pytest.mark.usefixtures("ascend_backend")
def test_rejects_acc_nz_store_with_dynamic_source_shape():
    rows = ir.Var("rows", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    with pytest.raises(ValueError, match="NZ"):
        passes.block_nz_tensor_views()(_nz_store_program(tile_shape=(rows, 16)))


@pytest.mark.parametrize("negative", [False, True], ids=["unknown", "negative"])
@pytest.mark.usefixtures("ascend_backend")
def test_rejects_acc_nz_store_with_unproven_batch_offset(negative):
    """Partition clamping must not redirect a write to batch panel zero."""
    batch = -1 if negative else ir.Var("batch", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    before = _nz_store_program(output_shape=(3, 64, 64), offsets=(batch, 0, 16), partition_shape=(1, 64, 16))
    with pytest.raises(ValueError, match="NZ tile.store requires provably non-negative leading batch"):
        passes.block_nz_tensor_views()(before)


@pytest.mark.usefixtures("ascend_backend")
def test_nz_submit_blocks_returned_alias_and_preserves_launch_fields():
    """Forwarded NZ outputs keep their blocked type across a task submission."""
    span = ir.Span.unknown()
    nz_type = ir.TensorType([64, 64], DataType.FP32, tensor_view=ir.TensorView(layout=ir.TensorLayout.NZ))
    output = ir.Var("out", nz_type, span)
    dep = ir.Var("dep", ir.ScalarType(DataType.TASK_ID), span)
    submit_type = ir.TupleType([nz_type, ir.ScalarType(DataType.TASK_ID)])
    core_num = ir.ConstInt(4, DataType.INDEX, span)
    submit = ir.Submit(
        ir.GlobalVar("writer"),
        [output],
        [dep],
        {"launch_tag": 3},
        {"test_marker": 7},
        submit_type,
        span,
        core_num=core_num,
        sync_start=True,
        allow_early_resolve=True,
    )
    result_var = ir.Var("result", submit_type, span)
    caller = ir.Function("caller", [output, dep], [], ir.AssignStmt(result_var, submit, span), span)
    # Opaque launch kwargs are intentional; they are metadata this pass must
    # preserve, rather than public DSL keywords the printer can round-trip.
    with passes.PassContext([]):
        after = passes.block_nz_tensor_views()(ir.Program([caller], "NzSubmit", span))
    statement = next(iter(after.functions.values())).body
    assert isinstance(statement, ir.AssignStmt)
    rewritten = statement.value
    assert isinstance(rewritten, ir.Submit)
    assert _shape_values(rewritten.args[0]) == [1, 8, 4, 16, 8]
    assert isinstance(rewritten.type, ir.TupleType)
    assert _shape_values(rewritten.type.types[0]) == [1, 8, 4, 16, 8]
    assert dict(rewritten.kwargs) == dict(submit.kwargs)
    assert dict(rewritten.attrs) == dict(submit.attrs)
    assert rewritten.deps == submit.deps
    assert rewritten.core_num is core_num
    assert rewritten.sync_start is True
    assert rewritten.allow_early_resolve is True


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


# -- A sliced shard handed to a separate kernel -------------------------------
#
# `BlockNzTensorViews` blocks a leading-axis slice, but the driver shape that
# needs it passes the slice as an argument to an outlined kernel. That route
# goes through `OptimizeOrchTensors`, which used to stamp the parent's logical
# row-major strides onto the kernel's NZ param -- strides that describe nothing
# for fractal-blocked bytes -- and blocking then refused the explicit stride.
# The in-function slice of `test_blocks_a_leading_axis_slice` never reaches
# that optimizer, so only a call boundary exercises this.


@pl.program
class SlicedNzShardToKernel:
    """An orchestration body handing one plane of a stacked NZ weight to a kernel."""

    @pl.function(type=pl.FunctionType.InCore)
    def shard_mm(
        self,
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[256, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
    ) -> pl.Tensor[[64, 256], pl.INT32]:
        xt = pl.load(x, [0, 0], [64, 512], target_memory=pl.Mem.Mat)
        wt = pl.load(w, [0, 0], [256, 512], target_memory=pl.Mem.Mat)
        acc = pl.matmul(xt, pl.tile.transpose_view(wt), out_dtype=pl.INT32)
        return pl.store(acc, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[64, 512], pl.INT8],
        w: pl.Tensor[[2, 256, 512], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
    ) -> pl.Tensor[[64, 256], pl.INT32]:
        shard = w[1]
        return self.shard_mm(x, shard, out)


def test_a_sliced_shard_reaches_its_kernel_unstrided():
    """`w[b]` passed to an outlined kernel blocks on both sides of the call."""
    after = _run(SlicedNzShardToKernel)

    # The kernel's own param, not the driver's: `_nz_param` would return
    # whichever NZ param sorts first, which here is the stacked `[2, ...]` one.
    kernel = after.get_function("shard_mm")
    assert kernel is not None
    param_type = kernel.params[1].type
    assert isinstance(param_type, ir.TensorType)
    assert _values(param_type.shape) == [1, 16, 16, 16, 32]
    # An explicit stride here is what blocking refuses: the NZ stride is derived
    # from the blocked shape, never carried from the logical parent.
    assert param_type.tensor_view is not None
    assert list(param_type.tensor_view.stride) == []

    slices = _nz_slices(after)
    assert len(slices) == 1
    assert _values(_elements(slices[0].args[1])) == [1, 16, 16, 16, 32]  # shapes
    assert _values(_elements(slices[0].args[2])) == [1, 0, 0, 0, 0]  # offsets


# -- Temporary guard for hw-native-sys/pto-isa#317 ----------------------------
# Delete this block together with ``CheckNzGmGapFitsBurstStride`` once the
# upstream truncation is fixed.


def _gap_program(rows: int, tile_rows: int):
    """A 16-column-block NZ load of *tile_rows* out of a *rows*-row weight.

    The GM gap ``TLoadGm2L1Nz2nz`` computes for it is exactly ``rows -
    tile_rows`` 32-byte blocks, for every dtype.
    """

    @pl.program
    class GmGap:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[64, 512], pl.INT8],
            w: pl.Tensor[[rows, 512], pl.INT8, pl.NZ],
            out: pl.Tensor[[64, tile_rows], pl.INT32],
        ):
            xt = pl.load(x, [0, 0], [64, 512], target_memory=pl.Mem.Mat)
            wt = pl.load(w, [0, 0], [tile_rows, 512], target_memory=pl.Mem.Mat)
            acc = pl.matmul(xt, pl.tile.transpose_view(wt), out_dtype=pl.INT32)
            pl.store(acc, [0, 0], out)
            return out

    return GmGap


def _single_column_block_program(rows: int, tile_rows: int):
    """The same load narrowed to one C0 column block (32 INT8 columns).

    ``TLoadGm2L1Nz2nz`` passes the column-block extent as ``nBurst``, so this
    load issues a single burst and never consumes ``gmGap``.
    """

    @pl.program
    class OneBlock:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[64, 32], pl.INT8],
            w: pl.Tensor[[rows, 32], pl.INT8, pl.NZ],
            out: pl.Tensor[[64, tile_rows], pl.INT32],
        ):
            xt = pl.load(x, [0, 0], [64, 32], target_memory=pl.Mem.Mat)
            wt = pl.load(w, [0, 0], [tile_rows, 32], target_memory=pl.Mem.Mat)
            acc = pl.matmul(xt, pl.tile.transpose_view(wt), out_dtype=pl.INT32)
            pl.store(acc, [0, 0], out)
            return out

    return OneBlock


def test_accepts_the_largest_encodable_gm_gap():
    """65520 blocks is the largest gap a fractal-aligned load can produce."""
    assert 65536 - 16 == 65520
    _run(_gap_program(rows=65536, tile_rows=16))


def test_rejects_a_gm_gap_above_the_burst_stride_field():
    """Above 65535 blocks pto-isa truncates the gap and reads wrong fractals.

    The message must say this is a temporary guard, not an NZ limitation — a
    user who reads it as "NZ tops out here" would design around a cap that is
    about to disappear.
    """
    assert 65552 - 16 == 65536
    with pytest.raises(ValueError) as excinfo:
        _run(_gap_program(rows=65552, tile_rows=16))
    message = str(excinfo.value)
    assert "65536 32-byte blocks" in message
    assert "NOT an expected pl.NZ limitation" in message
    assert "pto-isa#317" in message


def test_a_taller_row_tile_rescues_the_same_tensor():
    """The gap, not the row extent, is the limit.

    Same 65552-row weight the test above refuses, read in a 32-row tile instead
    of a 16-row one: the gap drops to 65520 and it compiles. This is why the
    diagnostic offers a *wider* tile as the workaround.
    """
    assert 65552 - 32 == 65520
    _run(_gap_program(rows=65552, tile_rows=32))


def test_a_single_column_block_load_is_exempt_from_the_gap_limit():
    """``gmGap`` is the stride *between* bursts, so one burst never reads it.

    Verified on device in pto-isa's own tload_gm2mat ST suite: an NZ int16 load
    with ``gShape1 = 1`` and a 65536-block gap returns bit-exact data, while the
    identical load at ``gShape1 = 2`` corrupts 1837/4096 elements. Rejecting the
    single-block case would refuse a load that is actually correct.
    """
    assert 65552 - 16 == 65536
    _run(_single_column_block_program(rows=65552, tile_rows=16))


def _valid_shape_program(rows: int, tile_rows: int, valid_rows: int):
    """An NZ load whose ``valid_shape`` narrows the row window below ``shapes``.

    Codegen builds the ``pto.partition_view`` from ``valid_shape`` when the load
    carries one, so this window — not ``shapes`` — is pto-isa's ``gShape``.
    """

    @pl.program
    class NarrowedValidShape:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[64, 64], pl.INT8],
            w: pl.Tensor[[rows, 64], pl.INT8, pl.NZ],
            out: pl.Tensor[[64, tile_rows], pl.INT32],
        ):
            xt = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
            wt = pl.load(w, [0, 0], [tile_rows, 64], valid_shape=[valid_rows, 64], target_memory=pl.Mem.Mat)
            acc = pl.matmul(xt, pl.tile.transpose_view(wt), out_dtype=pl.INT32)
            pl.store(acc, [0, 0], out)
            return out

    return NarrowedValidShape


def test_a_narrowed_valid_shape_drives_the_gap_check():
    """The gap follows ``valid_shape``, not ``shapes``.

    ``shapes`` alone puts this load at 65520 blocks — under the bound — while
    the partition codegen actually emits (``1x2x1x16x32``, from ``valid_shape``)
    puts it at 65536. Measuring ``shapes`` would wave through a load that
    truncates on device, which is the whole failure this guard exists to stop.
    """
    assert 65552 - 32 == 65520  # what `shapes` alone would report
    assert 65552 - 16 == 65536  # what pto-isa actually computes
    with pytest.raises(ValueError) as excinfo:
        _run(_valid_shape_program(rows=65552, tile_rows=32, valid_rows=16))
    assert "65536 32-byte blocks" in str(excinfo.value)


def test_a_valid_shape_load_within_the_bound_is_accepted():
    """A narrowed valid_shape is not rejected per se — only an oversized gap."""
    assert 65536 - 16 == 65520
    _run(_valid_shape_program(rows=65536, tile_rows=32, valid_rows=16))


@pytest.mark.parametrize("dynamic_axis", [0, 1])
def test_rejects_a_dynamic_leading_extent_when_folding(dynamic_axis):
    """A rank-4 fold must reject either dynamic leading axis before codegen."""
    batch = pl.dynamic("NZ_BATCH")
    shape = [batch, 4, 256, 512] if dynamic_axis == 0 else [2, batch, 256, 512]

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, w: pl.Tensor[shape, pl.INT8, pl.NZ]):
            return

    with pytest.raises(ValueError, match=f"axis {dynamic_axis} is dynamic"):
        passes.block_nz_tensor_views()(Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

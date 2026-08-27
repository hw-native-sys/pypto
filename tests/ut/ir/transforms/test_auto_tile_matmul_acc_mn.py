# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Issue #2232: M/N-tile a canonical loop-carried ``tile.matmul_acc`` reduction.

The canonical form has two source spellings and both must reach the same
loop-level M/N tiling: the peeled ``if k0 == 0: matmul else: matmul_acc`` and
the predicated single ``matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))``. Each
peeled kernel below therefore has a predicated twin, and
``test_peeled_and_predicated_retile_to_structurally_equal_output`` pins that the
two retiled programs differ only in that reduction statement.
"""

import re

import pypto.language as pl
import pytest
from pypto import backend as _backend
from pypto import ir, passes
from pypto.backend import BackendType

_TILE_STORE_OP = ir.get_op("tile.store").name
_TILE_MATMUL_OP = ir.get_op("tile.matmul").name
_TILE_MATMUL_ACC_OP = ir.get_op("tile.matmul_acc").name

M = 16
N = 1152
K_TOTAL = 1024
K_TILE = 128
N_TOTAL = N * 8

WIDE_M = 272
WIDE_N = 144
WIDE_K_TOTAL = 256
WIDE_K_TILE = 128

# Keep the same output boundary while making each source panel large enough for
# AutoTile to apply its ordinary inner-K rewrite after the enclosing-loop fold.
COMPOSE_K_TOTAL = 768
COMPOSE_K_TILE = 384

# Full-pipeline counterpart to the reviewer's (656,80,768) chooser case. The
# old logical candidate (576,48,32) boxes to physical Acc [576,64] = 144 KiB,
# while these smaller source panels also fit together in the 512 KiB Mat arena.
BOX_CAP_M = 576
BOX_CAP_N = 48
BOX_CAP_K_TOTAL = 256
BOX_CAP_K_TILE = 128


@pl.jit
def issue_2232_repro(
    a: pl.Tensor[[M, K_TOTAL], pl.INT8],
    b: pl.Tensor[[K_TOTAL, N_TOTAL], pl.INT8],
    c: pl.Out[pl.Tensor[[M, N_TOTAL], pl.INT32]],
):
    """The exact PyPTO-only reproducer attached to issue #2232."""
    for i in pl.spmd(N_TOTAL // N, name_hint="mm"):
        n0 = i * N
        acc = pl.create_tensor([M, N], dtype=pl.INT32)
        for kb in pl.pipeline(0, K_TOTAL // K_TILE, stage=2):
            k0 = kb * K_TILE
            at = a[0:M, k0 : k0 + K_TILE]
            bt = b[k0 : k0 + K_TILE, n0 : n0 + N]
            if k0 == 0:
                acc = pl.matmul(at, bt, out_dtype=pl.INT32)
            else:
                acc = pl.matmul_acc(acc, at, bt)
        c[0:M, n0 : n0 + N] = acc
    return c


@pl.jit
def issue_2232_repro_predicated(
    a: pl.Tensor[[M, K_TOTAL], pl.INT8],
    b: pl.Tensor[[K_TOTAL, N_TOTAL], pl.INT8],
    c: pl.Out[pl.Tensor[[M, N_TOTAL], pl.INT32]],
):
    """``issue_2232_repro`` written with the predicated accumulator instead."""
    for i in pl.spmd(N_TOTAL // N, name_hint="mm"):
        n0 = i * N
        acc = pl.create_tensor([M, N], dtype=pl.INT32)
        for kb in pl.pipeline(0, K_TOTAL // K_TILE, stage=2):
            k0 = kb * K_TILE
            at = a[0:M, k0 : k0 + K_TILE]
            bt = b[k0 : k0 + K_TILE, n0 : n0 + N]
            acc = pl.matmul_acc(acc, at, bt, init_cond=(k0 == 0))
        c[0:M, n0 : n0 + N] = acc
    return c


@pl.jit
def canonical_split_k_mn(
    a: pl.Tensor[[WIDE_M, WIDE_K_TOTAL], pl.INT8],
    b: pl.Tensor[[WIDE_K_TOTAL, WIDE_N], pl.INT8],
    c: pl.Out[pl.Tensor[[WIDE_M, WIDE_N], pl.INT32]],
):
    """A non-issue-specific case that requires both M and N output tiling."""
    for _ in pl.spmd(1):
        acc = pl.create_tensor([WIDE_M, WIDE_N], dtype=pl.INT32)
        for kb in pl.pipeline(0, WIDE_K_TOTAL // WIDE_K_TILE, stage=2):
            k0 = kb * WIDE_K_TILE
            at = a[0:WIDE_M, k0 : k0 + WIDE_K_TILE]
            bt = b[k0 : k0 + WIDE_K_TILE, 0:WIDE_N]
            if k0 == 0:
                acc = pl.matmul(at, bt, out_dtype=pl.INT32)
            else:
                acc = pl.matmul_acc(acc, at, bt)
        c[0:WIDE_M, 0:WIDE_N] = acc
    return c


@pl.jit
def canonical_split_k_mn_predicated(
    a: pl.Tensor[[WIDE_M, WIDE_K_TOTAL], pl.INT8],
    b: pl.Tensor[[WIDE_K_TOTAL, WIDE_N], pl.INT8],
    c: pl.Out[pl.Tensor[[WIDE_M, WIDE_N], pl.INT32]],
):
    """``canonical_split_k_mn`` with the predicated accumulator."""
    for _ in pl.spmd(1):
        acc = pl.create_tensor([WIDE_M, WIDE_N], dtype=pl.INT32)
        for kb in pl.pipeline(0, WIDE_K_TOTAL // WIDE_K_TILE, stage=2):
            k0 = kb * WIDE_K_TILE
            at = a[0:WIDE_M, k0 : k0 + WIDE_K_TILE]
            bt = b[k0 : k0 + WIDE_K_TILE, 0:WIDE_N]
            acc = pl.matmul_acc(acc, at, bt, init_cond=(k0 == 0))
        c[0:WIDE_M, 0:WIDE_N] = acc
    return c


@pl.jit
def canonical_split_k_n_boundary_retiles_k(
    a: pl.Tensor[[WIDE_M, COMPOSE_K_TOTAL], pl.INT8],
    b: pl.Tensor[[COMPOSE_K_TOTAL, WIDE_N], pl.INT8],
    c: pl.Out[pl.Tensor[[WIDE_M, WIDE_N], pl.INT32]],
):
    """Compose an N-tail padded output with the ordinary inner-K rewrite."""
    for _ in pl.spmd(1):
        acc = pl.create_tensor([WIDE_M, WIDE_N], dtype=pl.INT32)
        for kb in pl.pipeline(0, COMPOSE_K_TOTAL // COMPOSE_K_TILE, stage=2):
            k0 = kb * COMPOSE_K_TILE
            at = a[0:WIDE_M, k0 : k0 + COMPOSE_K_TILE]
            bt = b[k0 : k0 + COMPOSE_K_TILE, 0:WIDE_N]
            if k0 == 0:
                acc = pl.matmul(at, bt, out_dtype=pl.INT32)
            else:
                acc = pl.matmul_acc(acc, at, bt)
        c[0:WIDE_M, 0:WIDE_N] = acc
    return c


@pl.jit
def canonical_split_k_n_boundary_retiles_k_predicated(
    a: pl.Tensor[[WIDE_M, COMPOSE_K_TOTAL], pl.INT8],
    b: pl.Tensor[[COMPOSE_K_TOTAL, WIDE_N], pl.INT8],
    c: pl.Out[pl.Tensor[[WIDE_M, WIDE_N], pl.INT32]],
):
    """``canonical_split_k_n_boundary_retiles_k`` with the predicated form."""
    for _ in pl.spmd(1):
        acc = pl.create_tensor([WIDE_M, WIDE_N], dtype=pl.INT32)
        for kb in pl.pipeline(0, COMPOSE_K_TOTAL // COMPOSE_K_TILE, stage=2):
            k0 = kb * COMPOSE_K_TILE
            at = a[0:WIDE_M, k0 : k0 + COMPOSE_K_TILE]
            bt = b[k0 : k0 + COMPOSE_K_TILE, 0:WIDE_N]
            acc = pl.matmul_acc(acc, at, bt, init_cond=(k0 == 0))
        c[0:WIDE_M, 0:WIDE_N] = acc
    return c


@pl.program
class BoxedCapacityBefore:
    """Tile-level canonical input for a realizable boxed-capacity counterexample.

    The reviewer's exact (656,80,768) panels exceed the 910B Mat arena when
    co-resident. This equivalent shape exercises the same post-selection N-box
    overflow while keeping unrelated operand capacity out of the regression.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[BOX_CAP_M, BOX_CAP_K_TOTAL], pl.INT8],
        b: pl.Tensor[[BOX_CAP_K_TOTAL, BOX_CAP_N], pl.INT8],
        c: pl.Out[pl.Tensor[[BOX_CAP_M, BOX_CAP_N], pl.INT32]],
    ) -> pl.Tensor[[BOX_CAP_M, BOX_CAP_N], pl.INT32]:
        acc_init: pl.Tile[[BOX_CAP_M, BOX_CAP_N], pl.INT32, pl.Mem.Acc] = pl.tile.create(
            [BOX_CAP_M, BOX_CAP_N], dtype=pl.INT32, target_memory=pl.Mem.Acc
        )
        for k0, (acc_iter,) in pl.pipeline(
            0, BOX_CAP_K_TOTAL, BOX_CAP_K_TILE, init_values=(acc_init,), stage=2
        ):
            at: pl.Tile[[BOX_CAP_M, BOX_CAP_K_TILE], pl.INT8, pl.Mem.Mat] = pl.tile.load(
                a, [0, k0], [BOX_CAP_M, BOX_CAP_K_TILE], target_memory=pl.Mem.Mat
            )
            bt: pl.Tile[[BOX_CAP_K_TILE, BOX_CAP_N], pl.INT8, pl.Mem.Mat] = pl.tile.load(
                b, [k0, 0], [BOX_CAP_K_TILE, BOX_CAP_N], target_memory=pl.Mem.Mat
            )
            if k0 == 0:
                acc_first: pl.Tile[[BOX_CAP_M, BOX_CAP_N], pl.INT32, pl.Mem.Acc] = pl.tile.matmul(at, bt)
                acc_phi: pl.Tile[[BOX_CAP_M, BOX_CAP_N], pl.INT32, pl.Mem.Acc] = pl.yield_(acc_first)
            else:
                acc_next: pl.Tile[[BOX_CAP_M, BOX_CAP_N], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(
                    acc_iter, at, bt
                )
                acc_phi: pl.Tile[[BOX_CAP_M, BOX_CAP_N], pl.INT32, pl.Mem.Acc] = pl.yield_(acc_next)
            acc: pl.Tile[[BOX_CAP_M, BOX_CAP_N], pl.INT32, pl.Mem.Acc] = pl.yield_(acc_phi)
        c = pl.tile.store(acc, [0, 0], c)
        return c


@pl.program
class BoxedCapacityPredicatedBefore:
    """``BoxedCapacityBefore`` with the predicated accumulator.

    The loop variable here *is* the K offset (``pl.pipeline`` steps it by
    ``BOX_CAP_K_TILE``), so ``init_cond`` names it directly.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[BOX_CAP_M, BOX_CAP_K_TOTAL], pl.INT8],
        b: pl.Tensor[[BOX_CAP_K_TOTAL, BOX_CAP_N], pl.INT8],
        c: pl.Out[pl.Tensor[[BOX_CAP_M, BOX_CAP_N], pl.INT32]],
    ) -> pl.Tensor[[BOX_CAP_M, BOX_CAP_N], pl.INT32]:
        acc_init: pl.Tile[[BOX_CAP_M, BOX_CAP_N], pl.INT32, pl.Mem.Acc] = pl.tile.create(
            [BOX_CAP_M, BOX_CAP_N], dtype=pl.INT32, target_memory=pl.Mem.Acc
        )
        for k0, (acc_iter,) in pl.pipeline(
            0, BOX_CAP_K_TOTAL, BOX_CAP_K_TILE, init_values=(acc_init,), stage=2
        ):
            at: pl.Tile[[BOX_CAP_M, BOX_CAP_K_TILE], pl.INT8, pl.Mem.Mat] = pl.tile.load(
                a, [0, k0], [BOX_CAP_M, BOX_CAP_K_TILE], target_memory=pl.Mem.Mat
            )
            bt: pl.Tile[[BOX_CAP_K_TILE, BOX_CAP_N], pl.INT8, pl.Mem.Mat] = pl.tile.load(
                b, [k0, 0], [BOX_CAP_K_TILE, BOX_CAP_N], target_memory=pl.Mem.Mat
            )
            acc_next: pl.Tile[[BOX_CAP_M, BOX_CAP_N], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(
                acc_iter, at, bt, k0 == 0
            )
            acc: pl.Tile[[BOX_CAP_M, BOX_CAP_N], pl.INT32, pl.Mem.Acc] = pl.yield_(acc_next)
        c = pl.tile.store(acc, [0, 0], c)
        return c


@pl.program
class OpaquePredicateBefore:
    """A canonical triplet whose ``init_cond`` is a caller-supplied flag.

    Nothing in the loop says the accumulator is overwritten on its first K
    block, so the reduction must stay untouched rather than be duplicated per
    output tile on an unproven assumption. The ``[272, 144]`` INT32 output is
    153 KiB, above the 128 KiB L0C, so M/N tiling is what the kernel needs and
    what declining the match therefore withholds.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[WIDE_M, WIDE_K_TOTAL], pl.INT8],
        b: pl.Tensor[[WIDE_K_TOTAL, WIDE_N], pl.INT8],
        c: pl.Out[pl.Tensor[[WIDE_M, WIDE_N], pl.INT32]],
        seed: pl.Scalar[pl.BOOL],
    ) -> pl.Tensor[[WIDE_M, WIDE_N], pl.INT32]:
        acc_init: pl.Tile[[WIDE_M, WIDE_N], pl.INT32, pl.Mem.Acc] = pl.tile.create(
            [WIDE_M, WIDE_N], dtype=pl.INT32, target_memory=pl.Mem.Acc
        )
        for k0, (acc_iter,) in pl.pipeline(0, WIDE_K_TOTAL, WIDE_K_TILE, init_values=(acc_init,), stage=2):
            at: pl.Tile[[WIDE_M, WIDE_K_TILE], pl.INT8, pl.Mem.Mat] = pl.tile.load(
                a, [0, k0], [WIDE_M, WIDE_K_TILE], target_memory=pl.Mem.Mat
            )
            bt: pl.Tile[[WIDE_K_TILE, WIDE_N], pl.INT8, pl.Mem.Mat] = pl.tile.load(
                b, [k0, 0], [WIDE_K_TILE, WIDE_N], target_memory=pl.Mem.Mat
            )
            acc_next: pl.Tile[[WIDE_M, WIDE_N], pl.INT32, pl.Mem.Acc] = pl.tile.matmul_acc(
                acc_iter, at, bt, seed
            )
            acc: pl.Tile[[WIDE_M, WIDE_N], pl.INT32, pl.Mem.Acc] = pl.yield_(acc_next)
        c = pl.tile.store(acc, [0, 0], c)
        return c


def _collect_store_calls(program):
    """Every ``tile.store`` in the program, in traversal order."""
    collector = _StoreCallCollector()
    collector.visit_program(program)
    return collector.calls


def _incore_bodies(program):
    """The bodies AutoTileMatmulL0 actually rewrites, in declaration order."""
    return [func.body for func in program.functions.values() if func.func_type == pl.FunctionType.InCore]


def _jit_program(kernel):
    """Specialize a fully annotated JIT function without running passes."""
    _, _, tensor_meta, scalar_values, scalar_dtypes, per_func_dyn = kernel._bind_args_from_signature({})
    return kernel._compile_to_program(tensor_meta, scalar_values, scalar_dtypes, per_func_dyn, pl)


def _lower_to_auto_tile_input(program):
    """Run the Default prefix through LegalizeTileCast, stopping before AutoTile."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    for make_pass in (
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
        passes.legalize_tile_cast,
    ):
        program = make_pass()(program)
    return program


class _StampStoreAttrs(ir.IRMutator):
    """Attach opaque compiler metadata to source stores before AutoTile."""

    def visit_call(self, op: ir.Call) -> ir.Expr:
        expr = super().visit_call(op)
        call = expr if isinstance(expr, ir.Call) else op
        if call.op.name != _TILE_STORE_OP:
            return expr
        attrs = dict(call.attrs)
        attrs["test_store_marker"] = 2232
        return ir.Call(call.op, list(call.args), dict(call.kwargs), attrs, call.type, call.span)


class _StoreAttrCollector(ir.IRVisitor):
    """Collect attrs from every tile.store in a rewritten program."""

    def __init__(self) -> None:
        super().__init__()
        self.attrs: list[dict] = []

    def visit_call(self, op: ir.Call) -> None:
        if op.op.name == _TILE_STORE_OP:
            self.attrs.append(dict(op.attrs))
        super().visit_call(op)


class _StoreCallCollector(ir.IRVisitor):
    """Collect every ``tile.store`` call, in traversal order."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[ir.Call] = []

    def visit_call(self, op: ir.Call) -> None:
        if op.op.name == _TILE_STORE_OP:
            self.calls.append(op)
        super().visit_call(op)


class _PeelToPredicate(ir.IRMutator):
    """Rewrite a peeled split-K reduction into its predicated equivalent.

    ``if c: x = matmul(a, b) / else: x = matmul_acc(acc, a, b)`` becomes
    ``x = matmul_acc(acc, a, b, c)``, reusing the ``IfStmt``'s phi as the new
    definition so the enclosing yield needs no rewrite.

    This exists only so the two spellings' AutoTile output can be compared for
    structural equality *everywhere except* the reduction statement itself --
    the one place they are supposed to differ, because the retiler clones
    whichever spelling the source used. Any peel it does not recognize is
    returned untouched, which makes the comparison fail loudly rather than
    silently pass.
    """

    def visit_if_stmt(self, op: ir.IfStmt) -> ir.Stmt:
        stmt = super().visit_if_stmt(op)
        if not isinstance(stmt, ir.IfStmt) or len(stmt.return_vars) != 1 or stmt.else_body is None:
            return stmt
        arms = []
        for body in (stmt.then_body, stmt.else_body):
            if not isinstance(body, ir.SeqStmts) or len(body.stmts) != 2:
                return stmt
            assign = body.stmts[0]
            if not isinstance(assign, ir.AssignStmt) or not isinstance(assign.value, ir.Call):
                return stmt
            arms.append(assign.value)
        seed_call, acc_call = arms
        if seed_call.op.name != _TILE_MATMUL_OP or acc_call.op.name != _TILE_MATMUL_ACC_OP:
            return stmt
        phi = stmt.return_vars[0]
        predicated = ir.Call(
            acc_call.op,
            [*acc_call.args, stmt.condition],
            dict(acc_call.kwargs),
            dict(acc_call.attrs),
            phi.type,
            acc_call.span,
        )
        return ir.AssignStmt(phi, predicated, stmt.span)


def test_issue_2232_canonical_input_shape():
    """Pin the real loop/if/matmul_acc shape seen by AutoTileMatmulL0."""
    before = _lower_to_auto_tile_input(_jit_program(issue_2232_repro))
    printed = ir.python_print(before)
    assert "pl.tile.matmul_acc(" in printed
    assert "pl.pipeline(" in printed
    assert "if " in printed


def test_predicated_canonical_input_shape():
    """The predicated twin reaches AutoTile as one 4-operand MAD, no branch."""
    before = _lower_to_auto_tile_input(_jit_program(issue_2232_repro_predicated))
    printed = ir.python_print(before)
    assert "pl.pipeline(" in printed
    assert "if " not in printed
    # acc, lhs, rhs, init_cond -- the predicate is the fourth operand.
    match = re.search(r"pl\.tile\.matmul_acc\(([^)]*)\)", printed)
    assert match, printed
    assert len(match.group(1).split(", ")) == 4, printed
    assert "== 0)" in printed


def test_issue_2232_loop_level_mn_tiling():
    """The full [16, 1152] accumulator disappears. AutoTile clones the source
    split-K loop once per output-N tile, narrows the GM loads, completes all
    eight K blocks, and only then stores that output tile."""
    before = _lower_to_auto_tile_input(_jit_program(issue_2232_repro))
    with passes.PassContext([ir.make_roundtrip_instrument()]):
        after = passes.auto_tile_matmul_l0()(before)

    printed = ir.python_print(after)
    assert "pl.Tile[[16, 1152], pl.INT32" not in printed
    assert "[128, 1152], [128, 1152]" not in printed
    assert printed.count("in pl.pipeline(8, stage=2") == 2
    assert printed.count("pl.tile.store(") == 2
    assert "n0__ssa_v0 + " in printed
    # The local matmul/matmul_acc path still applies the supported inner K
    # blocking after the enclosing loop has been output-tiled.
    assert "target_memory=pl.Mem.Right" in printed
    assert "pl.tile.matmul_acc(" in printed


def test_predicated_issue_2232_loop_level_mn_tiling():
    """The predicated spelling is not a tiling cliff: same grid, same loads,
    same stores as the peeled spelling, and no branch in the result."""
    before = _lower_to_auto_tile_input(_jit_program(issue_2232_repro_predicated))
    with passes.PassContext([ir.make_roundtrip_instrument()]):
        after = passes.auto_tile_matmul_l0()(before)

    printed = ir.python_print(after)
    assert "pl.Tile[[16, 1152], pl.INT32" not in printed
    assert "[128, 1152], [128, 1152]" not in printed
    assert printed.count("in pl.pipeline(8, stage=2") == 2
    assert printed.count("pl.tile.store(") == 2
    assert "n0__ssa_v0 + " in printed
    assert "target_memory=pl.Mem.Right" in printed
    assert "pl.tile.matmul_acc(" in printed
    assert "if " not in printed


def test_canonical_split_k_tiles_both_m_and_n_with_boundaries():
    """The enclosing-loop rewrite is a general 2D output grid, not an N-only
    special case for issue #2232. Every generated output tile reruns both source
    K blocks; no full-shape Acc or operand load survives."""
    before = _lower_to_auto_tile_input(_jit_program(canonical_split_k_mn))
    with passes.PassContext([ir.make_roundtrip_instrument()]):
        after = passes.auto_tile_matmul_l0()(before)

    printed = ir.python_print(after)
    assert "pl.Tile[[272, 144], pl.INT32, pl.Mem.Acc]" not in printed
    assert "[272, 128], [272, 128]" not in printed
    assert "[128, 144], [128, 144]" not in printed
    source_k_loops = printed.count("in pl.pipeline(2, stage=2")
    output_stores = printed.count("pl.tile.store(")
    assert source_k_loops >= 4
    assert output_stores == source_k_loops
    assert "[144, 0]" in printed  # M boundary tile
    assert "[0, 128]" in printed  # N boundary tile
    # The logical 16-column N tail occupies a legal 32-column INT8 Mat box;
    # that same physical/logical split propagates through the Acc chain.
    assert "[128, 32], [128, 16], target_memory=pl.Mem.Mat" in printed
    assert "pl.Tile[[128, 32], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[128, 16])]" in printed
    # Stores keep the logical output offsets and rely on valid_shape to avoid
    # transferring padded columns.
    assert "pl.tile.store(acc__rv_v2_mn3, [144, 128]" in printed


def test_predicated_canonical_split_k_tiles_both_m_and_n_with_boundaries():
    """The 2D output grid, the box-padded N tail and the store offsets are the
    same for the predicated spelling."""
    before = _lower_to_auto_tile_input(_jit_program(canonical_split_k_mn_predicated))
    with passes.PassContext([ir.make_roundtrip_instrument()]):
        after = passes.auto_tile_matmul_l0()(before)

    printed = ir.python_print(after)
    assert "pl.Tile[[272, 144], pl.INT32, pl.Mem.Acc]" not in printed
    assert "[272, 128], [272, 128]" not in printed
    assert "[128, 144], [128, 144]" not in printed
    source_k_loops = printed.count("in pl.pipeline(2, stage=2")
    assert source_k_loops >= 4
    assert printed.count("pl.tile.store(") == source_k_loops
    assert "[144, 0]" in printed  # M boundary tile
    assert "[0, 128]" in printed  # N boundary tile
    assert "[128, 32], [128, 16], target_memory=pl.Mem.Mat" in printed
    assert "pl.Tile[[128, 32], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[128, 16])]" in printed
    assert "pl.tile.store(acc__rv_v2_mn3, [144, 128]" in printed
    assert "if " not in printed


def test_padded_n_boundary_retains_valid_shape_through_inner_k_rewrite():
    """A box-padded 16-column output tail remains logically 16 columns when
    the post-fold matmul is K-tiled again. In particular, the inner loop's Acc
    initializer must not widen its valid N extent back to the physical 32."""
    before = _lower_to_auto_tile_input(_jit_program(canonical_split_k_n_boundary_retiles_k))
    with passes.PassContext([ir.make_roundtrip_instrument()]):
        after = passes.auto_tile_matmul_l0()(before)

    printed = ir.python_print(after)
    assert printed.count("in pl.pipeline(2, stage=2") == 4
    assert printed.count("pl.tile.store(") == 4
    assert "[384, 32], [384, 16], target_memory=pl.Mem.Mat" in printed
    assert (
        "[192, 32], pl.INT8, pl.Mem.Right, pl.TileView(valid_shape=[192, 16], compact=pl.CompactMode.normal)"
    ) in printed
    assert "pl.Tile[[144, 32], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[144, 16])]" in printed
    assert "pl.tile.set_validshape(" in printed
    assert "pl.tile.store(acc__rv_v2_mn2, [0, 128]" in printed


def test_predicated_padded_n_boundary_retains_valid_shape_through_inner_k_rewrite():
    """Composing the predicated fold with the ordinary inner-K rewrite keeps the
    16-column logical tail, exactly as the peeled spelling does."""
    before = _lower_to_auto_tile_input(_jit_program(canonical_split_k_n_boundary_retiles_k_predicated))
    with passes.PassContext([ir.make_roundtrip_instrument()]):
        after = passes.auto_tile_matmul_l0()(before)

    printed = ir.python_print(after)
    assert printed.count("in pl.pipeline(2, stage=2") == 4
    assert printed.count("pl.tile.store(") == 4
    assert "[384, 32], [384, 16], target_memory=pl.Mem.Mat" in printed
    assert (
        "[192, 32], pl.INT8, pl.Mem.Right, pl.TileView(valid_shape=[192, 16], compact=pl.CompactMode.normal)"
    ) in printed
    assert "pl.Tile[[144, 32], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[144, 16])]" in printed
    assert "pl.tile.set_validshape(" in printed
    assert "pl.tile.store(acc__rv_v2_mn2, [0, 128]" in printed


def test_peeled_and_predicated_retile_to_structurally_equal_output():
    """The two spellings differ only in the reduction statement they carry.

    The retiler clones whichever body it matched, so the peeled output keeps its
    ``if`` / phi and the predicated output keeps its single 4-operand MAD -- that
    difference is expected and is asserted below. Everything the fold actually
    decides (the output grid, each narrowed load's offsets / shape /
    ``valid_shape``, every Acc initializer, the loop trip counts, the store
    chain) must be identical, which is what collapsing the peel back into the
    predicated form and comparing structurally proves.

    ``canonical_split_k_mn`` is the kernel where the comparison can be made
    whole-body: its narrowed per-tile K fits one L0 block, so no secondary K
    rewrite runs. Where one does run the two spellings stop being collapsible
    -- the peel becomes *two* inner K-loops behind a branch while the predicate
    becomes one -- and
    ``test_peeled_and_predicated_emit_the_same_output_grid`` covers the grid
    parity for those instead.
    """
    outputs = []
    for kernel in (canonical_split_k_mn, canonical_split_k_mn_predicated):
        before = _lower_to_auto_tile_input(_jit_program(kernel))
        with passes.PassContext([ir.make_roundtrip_instrument()]):
            outputs.append(passes.auto_tile_matmul_l0()(before))
    after_peeled, after_predicated = outputs

    assert "if " in ir.python_print(after_peeled)
    assert "if " not in ir.python_print(after_predicated)

    normalized = _PeelToPredicate().visit_program(after_peeled)
    assert "if " not in ir.python_print(normalized)
    # AutoTile only rewrites the InCore body; the enclosing SPMD / orchestration
    # wrappers differ solely in the callee name the two kernels were declared
    # with. Compare bodies and let auto-mapping pair up the (necessarily
    # distinct) Var objects.
    peeled_bodies = _incore_bodies(normalized)
    predicated_bodies = _incore_bodies(after_predicated)
    assert len(peeled_bodies) == 1
    assert len(predicated_bodies) == 1
    ir.assert_structural_equal(peeled_bodies[0], predicated_bodies[0], True)


@pytest.mark.parametrize(
    ("peeled", "predicated"),
    [
        (issue_2232_repro, issue_2232_repro_predicated),
        (canonical_split_k_mn, canonical_split_k_mn_predicated),
        (canonical_split_k_n_boundary_retiles_k, canonical_split_k_n_boundary_retiles_k_predicated),
    ],
)
def test_peeled_and_predicated_emit_the_same_output_grid(peeled, predicated):
    """Neither spelling is a tiling cliff: same grid, tile by tile.

    Every ``tile.store`` the fold emits carries the whole per-tile decision --
    which Acc tile is drained (its physical shape and ``valid_shape`` come from
    the compared Var's type), at which output offset, and onto which link of the
    output chain. Comparing the store sequences structurally therefore pins that
    both spellings picked the same output grid, without requiring the reduction
    bodies to be collapsible onto each other.
    """
    grids = []
    for kernel in (peeled, predicated):
        before = _lower_to_auto_tile_input(_jit_program(kernel))
        with passes.PassContext([ir.make_roundtrip_instrument()]):
            grids.append(_collect_store_calls(passes.auto_tile_matmul_l0()(before)))
    peeled_stores, predicated_stores = grids

    assert len(peeled_stores) >= 2
    assert len(peeled_stores) == len(predicated_stores)
    for peeled_store, predicated_store in zip(peeled_stores, predicated_stores):
        ir.assert_structural_equal(peeled_store, predicated_store, True)


def test_already_padded_output_localizes_valid_shape_across_mn_grid():
    """Explicit M/N grid offsets intersect, rather than reset, valid_shape.

    The physical output is 288 columns but only 272 are logical. AutoTile must
    therefore keep the final physical 32-column panel at valid N=16 even though
    that panel is emitted by BuildSplitKGrid with a nonzero output offset.
    """
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tensor[[512, 384], pl.INT8],
            rhs: pl.Tensor[[384, 288], pl.INT8],
            out: pl.Out[pl.Tensor[[512, 288], pl.INT32]],
        ) -> pl.Tensor[[512, 288], pl.INT32]:
            lhs_mat: pl.Tile[[512, 384], pl.INT8, pl.Mem.Mat] = pl.tile.load(
                lhs, [0, 0], [512, 384], target_memory=pl.Mem.Mat
            )
            rhs_mat: pl.Tile[
                [384, 288],
                pl.INT8,
                pl.Mem.Mat,
                pl.TileView(valid_shape=[384, 272]),
            ] = pl.tile.load(
                rhs,
                [0, 0],
                [384, 288],
                valid_shape=[384, 272],
                target_memory=pl.Mem.Mat,
            )
            product: pl.Tile[
                [512, 288],
                pl.INT32,
                pl.Mem.Acc,
                pl.TileView(valid_shape=[512, 272]),
            ] = pl.tile.matmul(lhs_mat, rhs_mat)
            out = pl.tile.store(product, [0, 0], out)
            return out

    after = passes.auto_tile_matmul_l0()(Before)
    printed = ir.python_print(after)
    assert printed.count("pl.tile.store(") >= 4
    assert "pl.TileView(valid_shape=[" in printed
    assert re.search(
        r"pl\.Tile\[\[\d+, 32\], pl\.INT32, pl\.Mem\.Acc, pl\.TileView\(valid_shape=\[\d+, 16\]\)\]",
        printed,
    ), printed
    assert re.search(
        r"pl\.Tile\[\[\d+, 32\], pl\.INT8, pl\.Mem\.Right, "
        r"pl\.TileView\(valid_shape=\[\d+, 16\], compact=pl\.CompactMode\.normal\)\]",
        printed,
    ), printed


def test_symbolic_padded_output_localizes_valid_shape_across_mn_grid():
    """The sub-grid intersection remains symbolic when valid N is dynamic."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tensor[[512, 384], pl.INT8],
            rhs: pl.Tensor[[384, 288], pl.INT8],
            out: pl.Out[pl.Tensor[[512, 288], pl.INT32]],
            valid_n: pl.Scalar[pl.UINT64],
        ) -> pl.Tensor[[512, 288], pl.INT32]:
            lhs_mat: pl.Tile[[512, 384], pl.INT8, pl.Mem.Mat] = pl.tile.load(
                lhs, [0, 0], [512, 384], target_memory=pl.Mem.Mat
            )
            rhs_mat: pl.Tile[
                [384, 288],
                pl.INT8,
                pl.Mem.Mat,
                pl.TileView(valid_shape=[384, valid_n]),
            ] = pl.tile.load(
                rhs,
                [0, 0],
                [384, 288],
                valid_shape=[384, valid_n],
                target_memory=pl.Mem.Mat,
            )
            product: pl.Tile[
                [512, 288],
                pl.INT32,
                pl.Mem.Acc,
                pl.TileView(valid_shape=[512, valid_n]),
            ] = pl.tile.matmul(lhs_mat, rhs_mat)
            out = pl.tile.store(product, [0, 0], out)
            return out

    after = passes.auto_tile_matmul_l0()(Before)
    printed = ir.python_print(after)
    assert "pl.max(valid_n, pl.cast(256, pl.UINT64)) - pl.cast(256, pl.UINT64)" in printed
    assert "valid_n - pl.cast(256, pl.UINT64)" not in printed


def test_canonical_split_k_preserves_store_attrs_on_every_output_tile():
    """The one source store becomes one store per output tile without losing
    compiler metadata carried in ``Call.attrs``."""
    before = _lower_to_auto_tile_input(_jit_program(canonical_split_k_mn))
    stamped = _StampStoreAttrs().visit_program(before)
    after = passes.auto_tile_matmul_l0()(stamped)

    collector = _StoreAttrCollector()
    collector.visit_program(after)
    assert len(collector.attrs) >= 4
    assert all(attrs.get("test_store_marker") == 2232 for attrs in collector.attrs)


def test_issue_2232_full_default_pipeline_allocates():
    """After loop-level M/N tiling, both the issue reproducer and the general
    two-axis grid reach concrete allocation in the complete Default pipeline."""
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    pass_manager = PassManager.get_strategy(OptimizationStrategy.Default)
    for kernel in (issue_2232_repro, canonical_split_k_mn):
        result = pass_manager.run_passes(_jit_program(kernel))
        assert result is not None


def test_predicated_full_default_pipeline_allocates():
    """Without loop-level M/N tiling the predicated spelling does not merely
    lose an optimization -- its full ``[16, 1152]`` INT32 accumulator overflows
    L0C and allocation fails outright. Reaching the end of the Default pipeline
    is therefore the sharpest statement that the predicated form is matched."""
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    pass_manager = PassManager.get_strategy(OptimizationStrategy.Default)
    for kernel in (issue_2232_repro_predicated, canonical_split_k_mn_predicated):
        result = pass_manager.run_passes(_jit_program(kernel))
        assert result is not None


def test_non_seed_predicate_leaves_the_reduction_untouched(capfd):
    """An ``init_cond`` that is not a test of this loop's induction variable
    against 0 is not evidence of a split-K seed, so the triplet is declined --
    loudly, with the same ``PH-AT-006`` the other unsupported placements use."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    before = _lower_to_auto_tile_input(OpaquePredicateBefore)
    after = passes.auto_tile_matmul_l0()(before)

    ir.assert_structural_equal(after, before)
    assert "PH-AT-006" in capfd.readouterr().err


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("source", [BoxedCapacityBefore, BoxedCapacityPredicatedBefore])
def test_canonical_split_k_chooser_accounts_for_full_window_boxing(planner, source):
    """The pre-phase must not emit an Acc that overflows only after N boxing.

    Running the complete Default pipeline is the allocation regression: the
    PyPTO planner rejects an L0C arena above 128 KiB, so this test also proves
    the corrected candidate survives all downstream physical accounting. The
    chooser unit test separately pins the reviewer's exact (656,80,768) case.
    Both source spellings feed the same chooser, so both are checked.
    """
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    before = _lower_to_auto_tile_input(source)
    after_auto_tile = passes.auto_tile_matmul_l0()(before)
    printed = ir.python_print(after_auto_tile)

    assert "pl.Tile[[576, 64], pl.INT32, pl.Mem.Acc" not in printed
    assert "pl.Tile[[576, 48], pl.INT32, pl.Mem.Acc" not in printed
    with passes.PassContext([], memory_planner=planner):
        assert PassManager.get_strategy(OptimizationStrategy.Default).run_passes(source) is not None


def test_canonical_split_k_boundary_codegen_uses_box_aligned_physical_width():
    """After secondary K tiling, PTO still allocates N=32 with valid N=16."""
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415
    from pypto.pypto_core import codegen  # noqa: PLC0415

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
        _jit_program(canonical_split_k_n_boundary_retiles_k)
    )
    incore = [func for func in optimized.functions.values() if func.func_type == pl.FunctionType.AIC]
    assert len(incore) == 1
    single = ir.Program([incore[0]], incore[0].name, optimized.span)
    pto = codegen.PTOCodegen().generate(single)

    assert re.search(
        r"valid_col = %c16_index : !pto\.tile_buf<loc=mat, dtype=i8, rows=384, cols=32,",
        pto,
    ), pto
    # The boundary sub-tile's Acc is the one buffer the whole predicated K-loop
    # writes: allocated at the box-aligned physical width (cols=32) and narrowed
    # to the logical N=16 by the explicit set_validshape the accumulator seed
    # carries.  Pin the pair on the buffer the boundary tstore actually drains.
    tail_store = re.search(
        r"pto\.tstore ins\((?P<acc>%[\w.]+) : !pto\.tile_buf<loc=acc, dtype=i32, "
        r"rows=(?P<rows>128|144), cols=32,[^)]*\) outs\([^)]*<(?P=rows)x16xi32>\)",
        pto,
    )
    assert tail_store, pto
    tail_acc = re.escape(tail_store.group("acc"))
    tail_rows = tail_store.group("rows")
    assert re.search(
        rf"{tail_acc} = pto\.alloc_tile [^\n]*: !pto\.tile_buf<loc=acc, dtype=i32, "
        rf"rows={tail_rows}, cols=32,",
        pto,
    ), pto
    assert re.search(
        rf"pto\.set_validshape {tail_acc}, %c{tail_rows}_index, %c16_index : "
        rf"!pto\.tile_buf<loc=acc, dtype=i32, rows={tail_rows}, cols=32,",
        pto,
    ), pto
    assert re.search(
        r"!pto\.tile_buf<loc=right, dtype=i8, rows=192, cols=32, "
        r"v_row=\?, v_col=\?, blayout=row_major, slayout=col_major, fractal=512, pad=0, compact=1>",
        pto,
    ), pto
    assert "!pto.tile_buf<loc=mat, dtype=i8, rows=128, cols=16," not in pto
    assert "pto.tmov" not in pto, f"accumulator chains must coalesce without tile.move:\n{pto}"


def test_row_narrowed_matmul_declares_a_compact_accumulator_seed():
    """A K-split matmul whose lhs is row-narrowed must keep CompactMode through the chain.

    ``mad`` takes M from the L0A operand's valid rows and lays L0C out with an
    N-fractal stride of ``ceil(M/16)*16``; a reader that is not told the tile is
    compact walks it at the physical row count instead and picks up the wrong
    fractal (issues #2470, #2510). ``tile.matmul`` gets the mode from
    ``StampCompactForNarrowedAccRows``, but when the pass has to split K it also
    synthesizes the accumulator seed -- and ``tile.matmul_acc`` inherits its
    accumulator operand's mode, so a non-compact seed drags the whole chain, and
    the store after the loop, back to the physical pitch.

    The seed therefore *declares* the mode on its ``tile.create``. A declaration
    is what survives: a type stamped by the pass is discarded the moment
    ``InferTileMemorySpace`` re-deduces the call, whereas a kwarg is re-read.
    """
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            lhs: pl.Tensor[[64, 2048], pl.INT8],
            rhs: pl.Tensor[[2048, 128], pl.INT8],
            out: pl.Out[pl.Tensor[[64, 128], pl.INT32]],
        ) -> pl.Tensor[[64, 128], pl.INT32]:
            lhs_mat: pl.Tile[
                [64, 2048],
                pl.INT8,
                pl.Mem.Mat,
                pl.TileView(valid_shape=[16, 2048]),
            ] = pl.tile.load(lhs, [0, 0], [64, 2048], valid_shape=[16, 2048], target_memory=pl.Mem.Mat)
            rhs_mat: pl.Tile[[2048, 128], pl.INT8, pl.Mem.Mat] = pl.tile.load(
                rhs, [0, 0], [2048, 128], target_memory=pl.Mem.Mat
            )
            product: pl.Tile[
                [64, 128],
                pl.INT32,
                pl.Mem.Acc,
                pl.TileView(valid_shape=[16, 128], compact=pl.CompactMode.normal),
            ] = pl.tile.matmul(lhs_mat, rhs_mat)
            out = pl.tile.store(product, [0, 0], out)
            return out

    after = passes.auto_tile_matmul_l0()(Before)
    printed = ir.python_print(after)

    assert re.search(r"pl\.tile\.create\(\s*\[64, 128\][^)]*compact=True", printed, re.S), (
        f"the synthesized accumulator seed must declare compact=True:\n{printed}"
    )
    acc_views = re.findall(r"pl\.Mem\.Acc,\s*pl\.TileView\((valid_shape=\[16,[^)]*)\)", printed, re.S)
    assert acc_views, printed
    for view in acc_views:
        assert "compact=pl.CompactMode.normal" in view, (
            f"every row-narrowed Acc tile in the K chain must stay compact, got {view!r}:\n{printed}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

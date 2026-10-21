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
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

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
# AutoTile to emit the selected inner-K schedule within the enclosing-loop fold.
COMPOSE_K_TOTAL = 768
COMPOSE_K_TILE = 384


def _canonical_choice(M, N, K, iterations, *, int8=False, stages=2):
    cfg = passes.l0_tile_chooser.L0TileConfig()
    cfg.M, cfg.N, cfg.K = M, N, K
    cfg.l0a_bytes = cfg.l0b_bytes = 65536
    cfg.l0c_bytes = 131072
    cfg.bytes_a = cfg.bytes_b = 1 if int8 else 2
    cfg.align_k = cfg.min_k = 32 if int8 else 16
    cfg.l0c_align_m = 32 if int8 else 16
    cfg.box_align_m, cfg.box_align_n = 16, 32 if int8 else 16
    cfg.allow_k_boundary = True
    cfg.independent_output_tiles = True
    cfg.reduction_iterations = iterations
    cfg.reduction_pipeline_stages = stages
    return passes.l0_tile_chooser.choose_l0_tile(cfg)


def _canonical_grid_size(M, N, choice):
    return ((M + choice.m - 1) // choice.m) * ((N + choice.n - 1) // choice.n)


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
    """Compose an N-tail padded output with selected inner-K emission."""
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
    _, _, tensor_meta, scalar_dtypes, cx, per_func_dyn = kernel._bind_args_from_signature({})
    return kernel._compile_to_program(tensor_meta, scalar_dtypes, cx, per_func_dyn, pl)


def _lower_to_auto_tile_input(program):
    """Run the Default prefix through LegalizeTileCast, stopping before AutoTile."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)
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
        prefixes = []
        for body in (stmt.then_body, stmt.else_body):
            if not isinstance(body, ir.SeqStmts) or len(body.stmts) not in (2, 4):
                return stmt
            assign = body.stmts[-2]
            if not isinstance(assign, ir.AssignStmt) or not isinstance(assign.value, ir.Call):
                return stmt
            arms.append(assign.value)
            prefixes.append(list(body.stmts[:-2]))
        seed_call, acc_call = arms
        if seed_call.op.name != _TILE_MATMUL_OP or acc_call.op.name != _TILE_MATMUL_ACC_OP:
            return stmt
        if len(prefixes[0]) != len(prefixes[1]):
            return stmt
        remap = []
        for seed_extract, acc_extract in zip(*prefixes):
            if not isinstance(seed_extract, ir.AssignStmt) or not isinstance(acc_extract, ir.AssignStmt):
                return stmt
            if not ir.structural_equal(seed_extract.value, acc_extract.value):
                return stmt
            remap.append((acc_extract.var, seed_extract.var))
        remapped = ir.substitute_expr(acc_call, remap)
        assert isinstance(remapped, ir.Call)
        acc_call = remapped
        phi = stmt.return_vars[0]
        predicated = ir.Call(
            acc_call.op,
            [*acc_call.args, stmt.condition],
            dict(acc_call.kwargs),
            dict(acc_call.attrs),
            phi.type,
            acc_call.span,
        )
        return ir.SeqStmts([*prefixes[0], ir.AssignStmt(phi, predicated, stmt.span)], stmt.span)


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


def _assert_int8_l0_k_alignment(program):
    """Canonical rewrites must emit the native 32-element INT8 K box."""

    class Operands(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.count = 0

        def visit_call(self, op):
            if op.op.name in {_TILE_MATMUL_OP, _TILE_MATMUL_ACC_OP}:
                start = int(op.op.name == _TILE_MATMUL_ACC_OP)
                lhs, rhs = op.args[start : start + 2]
                assert isinstance(lhs.type, ir.TileType) and isinstance(rhs.type, ir.TileType)
                assert lhs.type.memory_space == ir.MemorySpace.Left
                assert rhs.type.memory_space == ir.MemorySpace.Right
                left_k, right_k = lhs.type.shape[1], rhs.type.shape[0]
                assert isinstance(left_k, ir.ConstInt) and isinstance(right_k, ir.ConstInt)
                assert left_k.value == right_k.value and left_k.value % 32 == 0
                self.count += 1
            super().visit_call(op)

    operands = Operands()
    operands.visit_program(program)
    assert operands.count > 0


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
    choice = _canonical_choice(M, N, K_TILE, K_TOTAL // K_TILE, int8=True)
    count = _canonical_grid_size(M, N, choice)
    assert printed.count("in pl.pipeline(8, stage=2") == count
    assert printed.count("pl.tile.store(") == count
    assert "n0__ssa_v0 + " in printed
    # The enclosing-loop fold emits the selected inner-K blocking directly.
    assert "target_memory=pl.Mem.Right" in printed
    assert "pl.tile.matmul_acc(" in printed
    _assert_int8_l0_k_alignment(after)


def test_predicated_issue_2232_loop_level_mn_tiling():
    """The predicated spelling is not a tiling cliff: same grid, same loads,
    same stores as the peeled spelling, and no branch in the result."""
    before = _lower_to_auto_tile_input(_jit_program(issue_2232_repro_predicated))
    with passes.PassContext([ir.make_roundtrip_instrument()]):
        after = passes.auto_tile_matmul_l0()(before)

    printed = ir.python_print(after)
    assert "pl.Tile[[16, 1152], pl.INT32" not in printed
    assert "[128, 1152], [128, 1152]" not in printed
    choice = _canonical_choice(M, N, K_TILE, K_TOTAL // K_TILE, int8=True)
    count = _canonical_grid_size(M, N, choice)
    assert printed.count("in pl.pipeline(8, stage=2") == count
    assert printed.count("pl.tile.store(") == count
    assert "n0__ssa_v0 + " in printed
    assert "target_memory=pl.Mem.Right" in printed
    assert "pl.tile.matmul_acc(" in printed
    assert "if " not in printed
    _assert_int8_l0_k_alignment(after)


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
    the selected K schedule is emitted. In particular, the inner loop's Acc
    initializer must not widen its valid N extent back to the physical 32."""
    before = _lower_to_auto_tile_input(_jit_program(canonical_split_k_n_boundary_retiles_k))
    with passes.PassContext([ir.make_roundtrip_instrument()]):
        after = passes.auto_tile_matmul_l0()(before)

    printed = ir.python_print(after)
    choice = _canonical_choice(WIDE_M, WIDE_N, COMPOSE_K_TILE, 2, int8=True)
    count = _canonical_grid_size(WIDE_M, WIDE_N, choice)
    assert printed.count("in pl.pipeline(2, stage=2") == count
    assert printed.count("pl.tile.store(") == count
    assert "[384, 32], [384, 16], target_memory=pl.Mem.Mat" in printed
    assert (
        f"[{choice.k}, 32], pl.INT8, pl.Mem.Right, pl.TileView(valid_shape=[{choice.k}, 16], "
        "compact=pl.CompactMode.normal)"
    ) in printed
    tail_m = WIDE_M % choice.m or choice.m
    assert (
        f"pl.Tile[[{tail_m}, 32], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[{tail_m}, 16])]" in printed
    )
    assert "pl.tile.set_validshape(" in printed
    assert "pl.tile.store(acc__rv_v2_mn2, [0, 128]" in printed


def test_predicated_padded_n_boundary_retains_valid_shape_through_inner_k_rewrite():
    """Emitting the selected K schedule in the predicated fold keeps the
    16-column logical tail, exactly as the peeled spelling does."""
    before = _lower_to_auto_tile_input(_jit_program(canonical_split_k_n_boundary_retiles_k_predicated))
    with passes.PassContext([ir.make_roundtrip_instrument()]):
        after = passes.auto_tile_matmul_l0()(before)

    printed = ir.python_print(after)
    choice = _canonical_choice(WIDE_M, WIDE_N, COMPOSE_K_TILE, 2, int8=True)
    count = _canonical_grid_size(WIDE_M, WIDE_N, choice)
    assert printed.count("in pl.pipeline(2, stage=2") == count
    assert printed.count("pl.tile.store(") == count
    assert "[384, 32], [384, 16], target_memory=pl.Mem.Mat" in printed
    assert (
        f"[{choice.k}, 32], pl.INT8, pl.Mem.Right, pl.TileView(valid_shape=[{choice.k}, 16], "
        "compact=pl.CompactMode.normal)"
    ) in printed
    tail_m = WIDE_M % choice.m or choice.m
    assert (
        f"pl.Tile[[{tail_m}, 32], pl.INT32, pl.Mem.Acc, pl.TileView(valid_shape=[{tail_m}, 16])]" in printed
    )
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
    whole-body: its selected per-tile K fits one L0 block, so the identical
    operand extracts in both branches can be hoisted for normalization.
    Where an inner K pipeline is needed the spellings stop being collapsible
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


def test_padding_only_output_columns_are_not_grid_work():
    """The chooser and emitter must agree on a complete, nonempty valid output grid."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat],
            b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[256, 16])],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            product = pl.tile.matmul(a, b)
            stored = pl.tile.store(product, [0, 0], out)
            return stored

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat],
            b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[256, 16])],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            for mi, (outer,) in pl.pipeline(
                0, 16, 16, init_values=(out,), stage=2, attrs={"pipeline_overlap_stores": False}
            ):
                left = pl.tile.extract(a, mi, 0, [16, 256], target_memory=pl.Mem.Left)
                for ni, (inner,) in pl.pipeline(
                    0, 16, 16, init_values=(outer,), stage=2, attrs={"pipeline_overlap_stores": False}
                ):
                    right = pl.tile.extract(b, 0, ni, [256, 16], target_memory=pl.Mem.Right)
                    product = pl.tile.matmul(left, right)
                    stored = pl.tile.store(product, [mi, ni], inner)
                    inner_result = pl.yield_(stored)
                outer_result = pl.yield_(inner_result)
            return outer_result

    after = passes.auto_tile_matmul_l0()(Before)
    ir.assert_structural_equal(after, Expected)


@pytest.mark.parametrize("accumulate", [False, True], ids=["returned-result", "caller-accumulator"])
def test_padded_result_without_a_grid_consumer_keeps_physical_shape(accumulate):
    """Logical bounds must not change a caller-owned accumulator or a returned tile's ABI."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat],
            b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[256, 16])],
            seed: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[16, 16])],
        ) -> pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[16, 16])]:
            product = pl.tile.matmul(a, b)
            return product

    @pl.program
    class Accumulating:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat],
            b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[256, 16])],
            seed: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[16, 16])],
        ) -> pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[16, 16])]:
            product = pl.tile.matmul_acc(seed, a, b)
            return product

    before = Accumulating if accumulate else Before
    after = passes.auto_tile_matmul_l0()(before)
    ir.assert_structural_equal(after, before)


def test_padded_accumulator_with_a_store_keeps_caller_shape():
    """Even a sole store does not authorize slicing an arbitrary accumulator."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat],
            b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[256, 16])],
            seed: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[16, 16])],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            product = pl.tile.matmul_acc(seed, a, b)
            stored = pl.tile.store(product, [0, 0], out)
            return stored

    after = passes.auto_tile_matmul_l0()(Before)
    ir.assert_structural_equal(after, Before)


def test_padded_accumulator_k_only_rewrite_keeps_physical_shape():
    """K tiling must preserve the full caller-owned Acc box, not just its valid prefix."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat],
            b: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[2048, 16])],
            seed: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[16, 16])],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            product = pl.tile.matmul_acc(seed, a, b)
            stored = pl.tile.store(product, [0, 0], out)
            return stored

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat],
            b: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[2048, 16])],
            seed: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[16, 16])],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            for ko, (acc,) in pl.pipeline(0, 2048, 256, stage=2, init_values=(seed,)):
                left = pl.tile.extract(a, 0, ko, [16, 256], target_memory=pl.Mem.Left)
                right = pl.tile.extract(b, ko, 0, [256, 64], target_memory=pl.Mem.Right)
                updated = pl.tile.matmul_acc(acc, left, right)
                product = pl.yield_(updated)
            stored = pl.tile.store(product, [0, 0], out)
            return stored

    after = passes.auto_tile_matmul_l0()(Before)
    ir.assert_structural_equal(after, Expected)


def test_padded_bias_resident_operand_keeps_full_n():
    """A resident Bias vector cannot be windowed just because output columns are padding."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat],
            b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[256, 16])],
            bias: pl.Tile[[1, 64], pl.FP32, pl.Mem.Bias, pl.TileView(valid_shape=[1, 16])],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            product = pl.tile.matmul_bias(a, b, bias)
            stored = pl.tile.store(product, [0, 0], out)
            return stored

    after = passes.auto_tile_matmul_l0()(Before)
    ir.assert_structural_equal(after, Before)


def test_padded_mat_bias_without_a_snapshot_keeps_its_k_only_rewrite():
    """Trimming N must not discard a legal full-width bias/K schedule when reload is impossible."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat],
            b: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[512, 16])],
            bias: pl.Tile[[1, 64], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[1, 16])],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            product = pl.tile.matmul_bias(a, b, bias)
            stored = pl.tile.store(product, [0, 0], out)
            return stored

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat],
            b: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[512, 16])],
            bias: pl.Tile[[1, 64], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[1, 16])],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            bias_l0 = pl.tile.move(bias, target_memory=pl.Mem.Bias)
            left0 = pl.tile.extract(a, 0, 0, [16, 256], target_memory=pl.Mem.Left)
            right0 = pl.tile.extract(b, 0, 0, [256, 64], target_memory=pl.Mem.Right)
            first = pl.tile.matmul_bias(left0, right0, bias_l0)
            left1 = pl.tile.extract(a, 0, 256, [16, 256], target_memory=pl.Mem.Left)
            right1 = pl.tile.extract(b, 256, 0, [256, 64], target_memory=pl.Mem.Right)
            product = pl.tile.matmul_acc(first, left1, right1)
            stored = pl.tile.store(product, [0, 0], out)
            return stored

    after = passes.auto_tile_matmul_l0()(Before)
    ir.assert_structural_equal(after, Expected)


def test_physical_output_below_minimum_n_is_not_expanded():
    """The existing unsupported sub-fractal shape must not grow to the minimum tile width."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat],
            b: pl.Tile[[128, 8], pl.BF16, pl.Mem.Mat],
            out: pl.Out[pl.Tensor[[16, 8], pl.FP32]],
        ) -> pl.Tensor[[16, 8], pl.FP32]:
            product = pl.tile.matmul(a, b)
            return pl.tile.store(product, [0, 0], out)

    after = passes.auto_tile_matmul_l0()(Before)
    ir.assert_structural_equal(after, Before)


def test_mat_scratch_restores_valid_reduction_extent():
    """A full-size Mat allocation must not expose unwritten columns to its consumer."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend950)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tile[[64, 64], pl.FP32, pl.Mem.Mat],
            b: pl.Tile[[64, 64], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[64, 13])],
            e: pl.Tile[[64, 16], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[13, 16])],
            out: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
        ) -> pl.Tensor[[64, 16], pl.FP32]:
            product = pl.tile.matmul(a, b)
            result = pl.tile.matmul(product, e)
            stored = pl.tile.store(result, [0, 0], out)
            return stored

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tile[[64, 64], pl.FP32, pl.Mem.Mat],
            b: pl.Tile[[64, 64], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[64, 13])],
            e: pl.Tile[[64, 16], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[13, 16])],
            out: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
        ) -> pl.Tensor[[64, 16], pl.FP32]:
            scratch = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Mat)
            for mi, (outer,) in pl.pipeline(
                0, 64, 64, init_values=(scratch,), stage=2, attrs={"pipeline_overlap_stores": False}
            ):
                left = pl.tile.extract(a, mi, 0, [64, 64], target_memory=pl.Mem.Left)
                for ni, (inner,) in pl.pipeline(
                    0, 16, 16, init_values=(outer,), stage=2, attrs={"pipeline_overlap_stores": False}
                ):
                    right = pl.tile.extract(b, 0, ni, [64, 16], target_memory=pl.Mem.Right)
                    product = pl.tile.matmul(left, right)
                    assembled = pl.tile.assemble(inner, product, [mi, ni])
                    inner_result = pl.yield_(assembled)
                outer_result = pl.yield_(inner_result)
            valid_scratch = pl.tile.set_validshape(outer_result, 64, 13)
            result = pl.tile.matmul(valid_scratch, e)
            stored = pl.tile.store(result, [0, 0], out)
            return stored

    after = passes.auto_tile_matmul_l0()(Before)
    ir.assert_structural_equal(after, Expected)


def test_already_padded_output_localizes_valid_shape_across_mn_grid():
    """Explicit M/N grid offsets intersect, rather than reset, valid_shape.

    The physical output is 288 columns but only 272 are logical. The emitted
    stores must cover those 272 columns exactly once, not turn the final
    complete padding block into additional work. Source storage stays 288 wide.
    """
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)

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
    rectangles = []
    for store in _collect_store_calls(after):
        offsets = store.args[1]
        tile_type = store.args[0].type
        assert isinstance(offsets, ir.MakeTuple)
        assert isinstance(tile_type, ir.TileType)
        extents = tile_type.shape
        if tile_type.tile_view is not None and tile_type.tile_view.valid_shape:
            extents = tile_type.tile_view.valid_shape
        assert len(offsets.elements) == len(extents) == 2
        dimensions = []
        for value in [*offsets.elements, *extents]:
            assert isinstance(value, ir.ConstInt)
            dimensions.append(int(value.value))
        row, col, height, width = dimensions
        assert height > 0 and width > 0
        assert row + height <= 512 and col + width <= 272, printed
        rectangles.append((row, col, height, width))
    for row in range(512):
        spans = sorted(
            (col, col + width) for top, col, height, width in rectangles if top <= row < top + height
        )
        cursor = 0
        for start, end in spans:
            assert start == cursor, f"gap or overlap at row {row}: {spans}"
            cursor = end
        assert cursor == 272, f"incomplete row {row}: {spans}"


def test_symbolic_padded_output_localizes_valid_shape_across_mn_grid():
    """The sub-grid intersection remains symbolic when valid N is dynamic."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)

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


@pytest.mark.parametrize(
    "kernel",
    [issue_2232_repro, canonical_split_k_mn],
    ids=["issue-2232-wide-n", "general-mn-grid"],
)
def test_issue_2232_full_default_pipeline_allocates(kernel):
    """After loop-level M/N tiling, each peeled split-K pattern reaches
    concrete allocation in the complete Default pipeline.

    Keep these as separate cases: the wide-N reproducer and general two-axis
    grid exercise different physical accumulator windows, and a failure in one
    must not prevent the other from reaching MemoryReuse.
    """
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)
    pass_manager = PassManager.get_strategy(OptimizationStrategy.Default)
    result = pass_manager.run_passes(_jit_program(kernel))
    assert result is not None


def test_predicated_full_default_pipeline_allocates():
    """Without loop-level M/N tiling the predicated spelling does not merely
    lose an optimization -- its full ``[16, 1152]`` INT32 accumulator overflows
    L0C and allocation fails outright. Reaching the end of the Default pipeline
    is therefore the sharpest statement that the predicated form is matched."""
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)
    pass_manager = PassManager.get_strategy(OptimizationStrategy.Default)
    for kernel in (issue_2232_repro_predicated, canonical_split_k_mn_predicated):
        result = pass_manager.run_passes(_jit_program(kernel))
        assert result is not None


def test_non_seed_predicate_leaves_the_reduction_untouched(capfd):
    """An ``init_cond`` that is not a test of this loop's induction variable
    against 0 is not evidence of a split-K seed, so the triplet is declined --
    loudly, with the same ``PH-AT-006`` the other unsupported placements use."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)
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
    _backend.set_backend_type(BackendType.Ascend910)
    before = _lower_to_auto_tile_input(source)
    after_auto_tile = passes.auto_tile_matmul_l0()(before)
    printed = ir.python_print(after_auto_tile)

    assert "pl.Tile[[576, 64], pl.INT32, pl.Mem.Acc" not in printed
    assert "pl.Tile[[576, 48], pl.INT32, pl.Mem.Acc" not in printed
    with passes.PassContext([], memory_planner=planner):
        assert PassManager.get_strategy(OptimizationStrategy.Default).run_passes(source) is not None


def test_canonical_split_k_boundary_codegen_uses_box_aligned_physical_width():
    """Selected K emission still allocates physical N=32 with valid N=16."""
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415
    from pypto.pypto_core import codegen  # noqa: PLC0415

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)
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
    choice = _canonical_choice(WIDE_M, WIDE_N, COMPOSE_K_TILE, 2, int8=True)
    expected_tail_m = WIDE_M % choice.m or choice.m
    tail_store = re.search(
        r"pto\.tstore ins\((?P<acc>%[\w.]+) : !pto\.tile_buf<loc=acc, dtype=i32, "
        rf"rows=(?P<rows>{expected_tail_m}), cols=32,[^)]*\) outs\([^)]*<(?P=rows)x16xi32>\)",
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
        rf"!pto\.tile_buf<loc=right, dtype=i8, rows={choice.k}, cols=32, "
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
    _backend.set_backend_type(BackendType.Ascend910)

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
                pl.TileView(valid_shape=[13, 2048]),
            ] = pl.tile.load(lhs, [0, 0], [64, 2048], valid_shape=[13, 2048], target_memory=pl.Mem.Mat)
            rhs_mat: pl.Tile[[2048, 128], pl.INT8, pl.Mem.Mat] = pl.tile.load(
                rhs, [0, 0], [2048, 128], target_memory=pl.Mem.Mat
            )
            product: pl.Tile[
                [64, 128],
                pl.INT32,
                pl.Mem.Acc,
                pl.TileView(valid_shape=[13, 128], compact=pl.CompactMode.normal),
            ] = pl.tile.matmul(lhs_mat, rhs_mat)
            out = pl.tile.store(product, [0, 0], out)
            return out

    after = passes.auto_tile_matmul_l0()(Before)
    printed = ir.python_print(after)

    assert re.search(r"pl\.tile\.create\(\s*\[16, 128\][^)]*compact=True", printed, re.S), (
        f"the synthesized accumulator seed must declare compact=True:\n{printed}"
    )
    acc_views = re.findall(r"pl\.Mem\.Acc,\s*pl\.TileView\((valid_shape=\[13,[^)]*)\)", printed, re.S)
    assert acc_views, printed
    for view in acc_views:
        assert "compact=pl.CompactMode.normal" in view, (
            f"every row-narrowed Acc tile in the K chain must stay compact, got {view!r}:\n{printed}"
        )


def _canonical_selected_k_input(M, N, source_k, iterations, stages=2, *, int8=False):
    total_k = source_k * iterations
    operand_dtype = pl.INT8 if int8 else pl.BF16
    accumulator_dtype = pl.INT32 if int8 else pl.FP32

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[M, total_k], operand_dtype],
            b: pl.Tensor[[total_k, N], operand_dtype],
            out: pl.Out[pl.Tensor[[M, N], accumulator_dtype]],
        ) -> pl.Tensor[[M, N], accumulator_dtype]:
            seed = pl.tile.create([M, N], dtype=accumulator_dtype, target_memory=pl.Mem.Acc)
            for k0, (carry,) in pl.pipeline(0, total_k, source_k, init_values=(seed,), stage=stages):
                at = pl.tile.load(a, [0, k0], [M, source_k], target_memory=pl.Mem.Mat)
                bt = pl.tile.load(b, [k0, 0], [source_k, N], target_memory=pl.Mem.Mat)
                product = pl.tile.matmul_acc(carry, at, bt, init_cond=(k0 == 0))
                result = pl.yield_(product)
            out = pl.tile.store(result, [0, 0], out)
            return out

    return Before


def test_k_only_canonical_reduction_emits_one_selected_inner_schedule():
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[128, 1024], pl.BF16],
            b: pl.Tensor[[1024, 256], pl.BF16],
            out: pl.Out[pl.Tensor[[128, 256], pl.FP32]],
        ) -> pl.Tensor[[128, 256], pl.FP32]:
            seed = pl.tile.create([128, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for k0, (carry,) in pl.pipeline(0, 1024, 128, init_values=(seed,), stage=2):
                at = pl.tile.load(a, [0, k0], [128, 128], target_memory=pl.Mem.Mat)
                bt = pl.tile.load(b, [k0, 0], [128, 256], target_memory=pl.Mem.Mat)
                product = pl.tile.matmul_acc(carry, at, bt, init_cond=(k0 == 0))
                result = pl.yield_(product)
            out = pl.tile.store(result, [0, 0], out)
            return out

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[128, 1024], pl.BF16],
            b: pl.Tensor[[1024, 256], pl.BF16],
            out: pl.Out[pl.Tensor[[128, 256], pl.FP32]],
        ) -> pl.Tensor[[128, 256], pl.FP32]:
            seed = pl.tile.create([128, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for k0, (carry,) in pl.pipeline(0, 1024, 128, init_values=(seed,), stage=2):
                at = pl.tile.load(a, [0, k0], [128, 128], target_memory=pl.Mem.Mat)
                bt = pl.tile.load(b, [k0, 0], [128, 256], target_memory=pl.Mem.Mat)
                for ko, (inner,) in pl.pipeline(0, 128, 32, init_values=(carry,), stage=2):
                    sa = pl.tile.extract(at, 0, ko, [128, 32], target_memory=pl.Mem.Left)
                    sb = pl.tile.extract(bt, ko, 0, [32, 256], target_memory=pl.Mem.Right)
                    product = pl.tile.matmul_acc(inner, sa, sb, init_cond=((k0 == 0) and (ko == 0)))
                    blocked = pl.yield_(product)
                result = pl.yield_(blocked)
            stored = pl.tile.store(result, [0, 0], out)
            return stored

    choice = _canonical_choice(128, 256, 128, 8)
    assert (choice.m, choice.n, choice.k) == (128, 256, 32)
    with passes.PassContext([ir.make_roundtrip_instrument()]):
        after = passes.auto_tile_matmul_l0()(Before)
    ir.assert_structural_equal(after, Expected)


class _CanonicalMadOperands(ir.IRVisitor):
    def __init__(self):
        super().__init__()
        self.ks = set()

    def visit_call(self, op):
        if op.op.name in {_TILE_MATMUL_OP, _TILE_MATMUL_ACC_OP}:
            start = int(op.op.name == _TILE_MATMUL_ACC_OP)
            lhs, rhs = op.args[start : start + 2]
            assert isinstance(lhs.type, ir.TileType)
            assert isinstance(rhs.type, ir.TileType)
            assert lhs.type.memory_space == ir.MemorySpace.Left
            assert rhs.type.memory_space == ir.MemorySpace.Right
            k = lhs.type.shape[1]
            rhs_k = rhs.type.shape[0]
            assert isinstance(k, ir.ConstInt) and isinstance(rhs_k, ir.ConstInt)
            assert rhs_k.value == k.value
            self.ks.add(k.value)
        super().visit_call(op)


@pytest.mark.parametrize(
    "planner,dbc",
    [
        (passes.MemoryPlanner.PYPTO, False),
        (passes.MemoryPlanner.PYPTO, True),
        (passes.MemoryPlanner.DSA_RP, False),
    ],
)
@pytest.mark.parametrize(
    "M,N,K,R,int8",
    [
        (128, 256, 128, 8, False),  # K-only rewrite: source output already fits L0C.
        (64, 384, 96, 3, False),  # Old recursive call discarded selected k=16.
        (80, 288, 80, 3, False),  # A boxed N boundary with full-K blocks.
        (128, 256, 80, 3, False),  # Selected k=32 with a 16-wide K tail.
        (272, 384, 176, 3, False),  # M boundary plus separately allocated K-tail operands.
        (16, 1152, 128, 8, True),  # #2232: INT32 accumulator has 32 physical rows.
        (272, 144, 384, 2, True),  # INT8 boundary: physical N=32, valid N=16.
    ],
)
def test_canonical_selected_k_survives_boundaries_and_native_codegen(
    M, N, K, R, int8, planner, dbc, tmp_path
):
    """Preserve selected K with PyPTO-owned placement, not PTOAS-managed allocation."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)
    expected = _canonical_choice(M, N, K, R, int8=int8)
    before = _canonical_selected_k_input(M, N, K, R, int8=int8)
    with passes.PassContext(
        [ir.make_roundtrip_instrument()], memory_planner=planner, enable_pypto_l0c_double_buffer=dbc
    ):
        after = passes.auto_tile_matmul_l0()(before)
        operands = _CanonicalMadOperands()
        operands.visit_program(after)
        expected_ks = {expected.k}
        if K % expected.k:
            expected_ks.add(K % expected.k)
        assert operands.ks == expected_ks
        assert len(_collect_store_calls(after)) == ((M + expected.m - 1) // expected.m) * (
            (N + expected.n - 1) // expected.n
        )
        allocated = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(before)
    mlir = codegen.PTOCodegen().generate(allocated, emit_tile_addr=True)
    assert "pto.tmatmul" in mlir
    native_ks = {int(k) for k in re.findall(r"!pto.tile_buf<loc=left,[^>]*cols=(\d+)", mlir)}
    assert native_ks == expected_ks
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source = tmp_path / "canonical_selected_k.pto"
    output = tmp_path / "canonical_selected_k.cpp"
    source.write_text(mlir)
    _run_ptoas(str(source), str(output), ["--pto-arch=a2", "--pto-level=level3", "--enable-insert-sync"])
    assert output.is_file()


def test_k_only_canonical_fold_preserves_declared_accumulator_slots():
    slots = pl.MemRef(slots=2)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 1024], pl.BF16],
            b: pl.Tensor[[1024, 256], pl.BF16],
            out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
        ) -> pl.Tensor[[64, 256], pl.FP32]:
            seed: pl.Tile[[64, 256], pl.FP32, slots[0], pl.Mem.Acc] = pl.tile.create(
                [64, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            for k0, (carry,) in pl.pipeline(0, 1024, 128, init_values=(seed,), stage=2):
                at = pl.tile.load(a, [0, k0], [64, 128], target_memory=pl.Mem.Mat)
                bt = pl.tile.load(b, [k0, 0], [128, 256], target_memory=pl.Mem.Mat)
                product = pl.tile.matmul_acc(carry, at, bt, init_cond=(k0 == 0))
                result = pl.yield_(product)
            out = pl.tile.store(result, [0, 0], out)
            return out

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([ir.make_roundtrip_instrument()]):
        after = passes.auto_tile_matmul_l0()(Before)
    before_body = next(func for func in Before.functions.values() if func.name == "kernel").body
    after_body = next(func for func in after.functions.values() if func.name == "kernel").body
    assert isinstance(before_body, ir.SeqStmts) and isinstance(after_body, ir.SeqStmts)
    ir.assert_structural_equal(after_body.stmts[0], before_body.stmts[0])
    assert "slots=2" in ir.python_print(after)


@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
@pytest.mark.parametrize("operand", ["lhs", "rhs"])
@pytest.mark.parametrize("slot_scale", [0, 1], ids=["constant", "rotating"])
def test_k_only_canonical_fold_preserves_declared_operand_slots(planner, operand, slot_scale):
    """Declining the canonical fold must preserve declared Mat storage exactly."""
    slots = pl.MemRef(slots=2)

    @pl.program
    class LeftBound:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[128, 1024], pl.BF16],
            b: pl.Tensor[[1024, 256], pl.BF16],
            out: pl.Out[pl.Tensor[[128, 256], pl.FP32]],
        ) -> pl.Tensor[[128, 256], pl.FP32]:
            seed = pl.tile.create([128, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for k0, (carry,) in pl.pipeline(0, 1024, 128, init_values=(seed,), stage=2):
                at: pl.Tile[[128, 128], pl.BF16, slots[slot_scale * (k0 // 128 % 2)], pl.Mem.Mat] = (
                    pl.tile.load(a, [0, k0], [128, 128], target_memory=pl.Mem.Mat)
                )
                bt = pl.tile.load(b, [k0, 0], [128, 256], target_memory=pl.Mem.Mat)
                product = pl.tile.matmul_acc(carry, at, bt, init_cond=(k0 == 0))
                result = pl.yield_(product)
            out = pl.tile.store(result, [0, 0], out)
            return out

    @pl.program
    class RightBound:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[128, 1024], pl.BF16],
            b: pl.Tensor[[1024, 256], pl.BF16],
            out: pl.Out[pl.Tensor[[128, 256], pl.FP32]],
        ) -> pl.Tensor[[128, 256], pl.FP32]:
            seed = pl.tile.create([128, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for k0, (carry,) in pl.pipeline(0, 1024, 128, init_values=(seed,), stage=2):
                at = pl.tile.load(a, [0, k0], [128, 128], target_memory=pl.Mem.Mat)
                bt: pl.Tile[[128, 256], pl.BF16, slots[slot_scale * (k0 // 128 % 2)], pl.Mem.Mat] = (
                    pl.tile.load(b, [k0, 0], [128, 256], target_memory=pl.Mem.Mat)
                )
                product = pl.tile.matmul_acc(carry, at, bt, init_cond=(k0 == 0))
                result = pl.yield_(product)
            out = pl.tile.store(result, [0, 0], out)
            return out

    before = LeftBound if operand == "lhs" else RightBound
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([ir.make_roundtrip_instrument()], memory_planner=planner):
        after = passes.auto_tile_matmul_l0()(before)
    before_func = next(func for func in before.functions.values() if func.name == "kernel")
    after_func = next(func for func in after.functions.values() if func.name == "kernel")
    assert isinstance(before_func.body, ir.SeqStmts) and isinstance(after_func.body, ir.SeqStmts)
    before_loop, after_loop = before_func.body.stmts[1], after_func.body.stmts[1]
    assert isinstance(before_loop, ir.ForStmt) and isinstance(after_loop, ir.ForStmt)
    assert isinstance(before_loop.body, ir.SeqStmts) and isinstance(after_loop.body, ir.SeqStmts)
    # Compare both complete load definitions, not just their printed shapes:
    # MemRef identity, selected slot, view, and load operands must all survive.
    for old, new in zip(before_loop.body.stmts[:2], after_loop.body.stmts[:2], strict=True):
        ir.assert_structural_equal(old, new)
    assert "slots=2" in ir.python_print(after)
    # For this shape ordinary per-call tiling also declines M/N placement into
    # a caller-owned accumulator. Retain the entire manual-storage program,
    # rather than inventing a new allocation merely to force a K rewrite.
    ir.assert_structural_equal(after, before)


def test_ptoas_canonical_boundary_keeps_compatible_legacy_schedule(tmp_path):
    """Retain PTOAS's working schedule instead of its known 76-KiB Left overflow."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)
    before = _canonical_selected_k_input(272, 384, 176, 3)
    with passes.PassContext([ir.make_roundtrip_instrument()], memory_planner=passes.MemoryPlanner.PTOAS):
        allocated = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(before)
    mlir = codegen.PTOCodegen().generate(allocated, emit_tile_addr=False)
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source, output = tmp_path / "compatible.pto", tmp_path / "compatible.cpp"
    source.write_text(mlir)
    _run_ptoas(str(source), str(output), ["--pto-arch=a2", "--pto-level=level2", "--enable-insert-sync"])
    assert output.is_file()


def test_unknown_canonical_trip_count_keeps_selected_k_without_assuming_repetition(capfd):
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 288], pl.BF16],
            b: pl.Tensor[[288, 384], pl.BF16],
            out: pl.Out[pl.Tensor[[64, 384], pl.FP32]],
            stop: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[64, 384], pl.FP32]:
            seed = pl.tile.create([64, 384], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for k0, (carry,) in pl.pipeline(0, stop, 96, init_values=(seed,), stage=2):
                at = pl.tile.load(a, [0, k0], [64, 96], target_memory=pl.Mem.Mat)
                bt = pl.tile.load(b, [k0, 0], [96, 384], target_memory=pl.Mem.Mat)
                product = pl.tile.matmul_acc(carry, at, bt, init_cond=(k0 == 0))
                result = pl.yield_(product)
            out = pl.tile.store(result, [0, 0], out)
            return out

    with passes.PassContext([ir.make_roundtrip_instrument()]):
        after = passes.auto_tile_matmul_l0()(Before)
    operands = _CanonicalMadOperands()
    operands.visit_program(after)
    assert operands.ks
    assert "stop" in ir.python_print(after)
    assert "single-C cost lower bound" in capfd.readouterr().err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Issue #2470: a loop carry must be typed by what its body yields, not by its seed.

An accumulator seeded with ``pl.create_tensor`` before a K-loop is typed from that
seed alone -- ``ConvertToSSA`` mints the ``IterArg`` from it and
``ConvertTensorToTileOps`` re-mints it from the converted seed, and both force the
loop's ``return_var`` to the same type. When every value the body yields is a matmul
over a row-narrowed left operand, the carry keeps advertising the full box height that
no ``mad`` ever wrote: the product lands in L0C at an N-fractal stride of
``ceil(validRow/16)*16`` while the reader after the loop walks the buffer at the
physical row pitch, scrambling every N-fractal above the first.

The repair (``narrow_loop_carry::NarrowAccCarries``) re-declares such a seed at the
extent the yields prove, so the existing deducers carry the narrowed, compact type
through the body and out to the reader on their own. It runs inside the two passes that
create the mismatch -- ``ConvertTensorToTileOps`` for a 2D seed and ``FlattenTileNdTo2D``
for an ND one -- so no pipeline stage publishes a carry its own verifiers reject.
"""

import pypto
import pypto.language as pl
import pytest
from pypto import backend as _backend
from pypto import codegen, ir, passes
from pypto.backend import BackendType

BLOCKS = 4
M_TILE = 64
K = 1024
N_TILE = 256
K_TILE = 512
FRACTAL_ROWS = 16  # one L0C fractal block: every valid row count packs to the box
FRACTAL_N_TILE = 128  # pypto-lib's projection tile; keeps the Mat arena within budget
FRACTAL_K_TILE = 256

_TILE_STORE_OP = ir.get_op("tile.store").name
_TILE_CREATE_OP = ir.get_op("tile.create").name
_TILE_SET_VALIDSHAPE_OP = ir.get_op("tile.set_validshape").name


@pl.jit
def create_tensor_seeded_acc(
    x: pl.Tensor[[BLOCKS * M_TILE, K], pl.INT8],
    w: pl.Tensor[[BLOCKS, N_TILE, K], pl.INT8],
    rows: pl.Tensor[[BLOCKS, 1], pl.INT32],
    y: pl.Out[pl.Tensor[[BLOCKS * M_TILE, N_TILE], pl.INT32]],
):
    """The issue #2470 shape: a full-height seed, narrowed matmuls, a peeled first K."""
    for b in pl.spmd(BLOCKS, name_hint="mm_ct", allow_early_resolve=True):
        m0 = b * M_TILE
        v = pl.min(M_TILE, pl.read(rows, [b, 0]))
        acc = pl.create_tensor([1, M_TILE, N_TILE], dtype=pl.INT32)
        for k0 in pl.pipeline(0, K, K_TILE, stage=2):
            xk = pl.slice(x, [M_TILE, K_TILE], [m0, k0], valid_shape=[v, K_TILE])
            wk = w[b : b + 1, 0:N_TILE, k0 : k0 + K_TILE]
            if k0 == 0:
                acc = pl.matmul(xk, wk, b_trans=True, out_dtype=pl.INT32)
            else:
                acc = pl.matmul_acc(acc, xk, wk, b_trans=True)
        y[m0 : m0 + M_TILE, :] = pl.reshape(acc, [M_TILE, N_TILE])
    return y


@pl.jit
def create_tensor_seeded_acc_2d(
    x: pl.Tensor[[BLOCKS * M_TILE, K], pl.INT8],
    w: pl.Tensor[[N_TILE, K], pl.INT8],
    rows: pl.Tensor[[BLOCKS, 1], pl.INT32],
    y: pl.Out[pl.Tensor[[BLOCKS * M_TILE, N_TILE], pl.INT32]],
):
    """The same defect one pass earlier: a 2D seed narrows already in ConvertTensorToTileOps."""
    for b in pl.spmd(BLOCKS, name_hint="mm_2d", allow_early_resolve=True):
        m0 = b * M_TILE
        v = pl.min(M_TILE, pl.read(rows, [b, 0]))
        acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
        for k0 in pl.pipeline(0, K, K_TILE, stage=2):
            xk = pl.slice(x, [M_TILE, K_TILE], [m0, k0], valid_shape=[v, K_TILE])
            wk = pl.slice(w, [N_TILE, K_TILE], [0, k0])
            if k0 == 0:
                acc = pl.matmul(xk, wk, b_trans=True, out_dtype=pl.INT32)
            else:
                acc = pl.matmul_acc(acc, xk, wk, b_trans=True)
        y[m0 : m0 + M_TILE, :] = acc
    return y


@pl.jit
def carry_flows_into_a_later_loop(
    x: pl.Tensor[[BLOCKS * M_TILE, K], pl.INT8],
    w: pl.Tensor[[N_TILE, K], pl.INT8],
    rows: pl.Tensor[[BLOCKS, 1], pl.INT32],
    y: pl.Out[pl.Tensor[[BLOCKS * M_TILE, N_TILE], pl.INT32]],
):
    """Two K loops in sequence: the second carries the accumulator the first repaired.

    The second loop's carry is initialised by a value this repair re-typed, so its own
    ``IterArg`` -- and everything its body deduces from it -- has to follow, or the body
    keeps referring to the carry at the width the seed used to have.
    """
    for b in pl.spmd(BLOCKS, name_hint="mm_two_loops", allow_early_resolve=True):
        m0 = b * M_TILE
        v = pl.min(M_TILE, pl.read(rows, [b, 0]))
        acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
        for k0 in pl.pipeline(0, K // 2, K_TILE, stage=2):
            xk = pl.slice(x, [M_TILE, K_TILE], [m0, k0], valid_shape=[v, K_TILE])
            wk = pl.slice(w, [N_TILE, K_TILE], [0, k0])
            if k0 == 0:
                acc = pl.matmul(xk, wk, b_trans=True, out_dtype=pl.INT32)
            else:
                acc = pl.matmul_acc(acc, xk, wk, b_trans=True)
        for k1 in pl.pipeline(K // 2, K, K_TILE, stage=2):
            xk2 = pl.slice(x, [M_TILE, K_TILE], [m0, k1], valid_shape=[v, K_TILE])
            wk2 = pl.slice(w, [N_TILE, K_TILE], [0, k1])
            acc = pl.matmul_acc(acc, xk2, wk2, b_trans=True)
        y[m0 : m0 + M_TILE, :] = acc
    return y


@pl.jit
def single_fractal_block_acc(
    x: pl.Tensor[[BLOCKS * FRACTAL_ROWS, K], pl.BF16],
    w: pl.Tensor[[K, FRACTAL_N_TILE], pl.BF16],
    rows: pl.Tensor[[BLOCKS, 1], pl.INT32],
    y: pl.Out[pl.Tensor[[BLOCKS * FRACTAL_ROWS, FRACTAL_N_TILE], pl.FP32]],
):
    """A [16, N] accumulator: `ceil(v/16)*16` is 16 for every valid row count it can hold.

    This is the shape pypto-lib's `qkv_proj_rope` projections use, down to the runtime row
    count computed next to the slice it bounds.
    """
    for b in pl.spmd(BLOCKS, name_hint="mm_fractal", allow_early_resolve=True):
        m0 = b * FRACTAL_ROWS
        acc = pl.create_tensor([FRACTAL_ROWS, FRACTAL_N_TILE], dtype=pl.FP32)
        for k0 in pl.pipeline(0, K, FRACTAL_K_TILE, stage=2):
            v = pl.min(FRACTAL_ROWS, pl.read(rows, [b, 0]))
            xk = pl.slice(x, [FRACTAL_ROWS, FRACTAL_K_TILE], [m0, k0], valid_shape=[v, FRACTAL_K_TILE])
            wk = pl.slice(w, [FRACTAL_K_TILE, FRACTAL_N_TILE], [k0, 0])
            if k0 == 0:
                acc = pl.matmul(xk, wk, out_dtype=pl.FP32)
            else:
                acc = pl.matmul_acc(acc, xk, wk)
        y[m0 : m0 + FRACTAL_ROWS, :] = acc
    return y


@pl.jit
def extent_computed_inside_the_loop(
    x: pl.Tensor[[BLOCKS * M_TILE, K], pl.INT8],
    w: pl.Tensor[[N_TILE, K], pl.INT8],
    rows: pl.Tensor[[BLOCKS, 1], pl.INT32],
    y: pl.Out[pl.Tensor[[BLOCKS * M_TILE, N_TILE], pl.INT32]],
):
    """The pitches differ, but the row count is only computed inside the loop body."""
    for b in pl.spmd(BLOCKS, name_hint="mm_inner_v", allow_early_resolve=True):
        m0 = b * M_TILE
        acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
        for k0 in pl.pipeline(0, K, K_TILE, stage=2):
            v = pl.min(M_TILE, pl.read(rows, [b, 0]))
            xk = pl.slice(x, [M_TILE, K_TILE], [m0, k0], valid_shape=[v, K_TILE])
            wk = pl.slice(w, [N_TILE, K_TILE], [0, k0])
            if k0 == 0:
                acc = pl.matmul(xk, wk, b_trans=True, out_dtype=pl.INT32)
            else:
                acc = pl.matmul_acc(acc, xk, wk, b_trans=True)
        y[m0 : m0 + M_TILE, :] = acc
    return y


@pl.jit
def full_height_acc(
    x: pl.Tensor[[BLOCKS * M_TILE, K], pl.INT8],
    w: pl.Tensor[[BLOCKS, N_TILE, K], pl.INT8],
    y: pl.Out[pl.Tensor[[BLOCKS * M_TILE, N_TILE], pl.INT32]],
):
    """The same kernel with no narrowing: every yield fills the seed's box."""
    for b in pl.spmd(BLOCKS, name_hint="mm_full", allow_early_resolve=True):
        m0 = b * M_TILE
        acc = pl.create_tensor([1, M_TILE, N_TILE], dtype=pl.INT32)
        for k0 in pl.pipeline(0, K, K_TILE, stage=2):
            xk = pl.slice(x, [M_TILE, K_TILE], [m0, k0])
            wk = w[b : b + 1, 0:N_TILE, k0 : k0 + K_TILE]
            if k0 == 0:
                acc = pl.matmul(xk, wk, b_trans=True, out_dtype=pl.INT32)
            else:
                acc = pl.matmul_acc(acc, xk, wk, b_trans=True)
        y[m0 : m0 + M_TILE, :] = pl.reshape(acc, [M_TILE, N_TILE])
    return y


@pl.jit
def vec_carry_narrowed_by_yield(
    x: pl.Tensor[[M_TILE, N_TILE], pl.FP32],
    y: pl.Out[pl.Tensor[[M_TILE, N_TILE], pl.FP32]],
):
    """A Vec carry whose yields are narrower -- outside this pass's remit."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="vec_carry"):
        seed: pl.Tile[[M_TILE, N_TILE], pl.FP32, pl.Mem.Vec] = pl.tile.create(
            [M_TILE, N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        for _, (carry,) in pl.pipeline(0, 4, 1, init_values=(seed,), stage=2):
            narrowed: pl.Tile[
                [M_TILE, N_TILE],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(valid_shape=[16, N_TILE]),
            ] = pl.tile.set_validshape(carry, 16, N_TILE)
            acc: pl.Tile[[M_TILE, N_TILE], pl.FP32, pl.Mem.Vec] = pl.yield_(narrowed)
        y = pl.tile.store(acc, [0, 0], y)
    return y


def _jit_program(kernel):
    """Specialize a fully annotated JIT function without running passes."""
    _, _, tensor_meta, scalar_values, scalar_dtypes, per_func_dyn = kernel._bind_args_from_signature({})
    return kernel._compile_to_program(tensor_meta, scalar_values, scalar_dtypes, per_func_dyn, pl)


_TENSOR_PREFIX = (
    "inline_functions",
    "unroll_loops",
    "ctrl_flow_transform",
    "convert_to_ssa",
    "simplify",
    "normalize_stmt_structure",
    "flatten_call_expr",
    "outline_hierarchy_scopes",
    "outline_incore_scopes",
    "outline_cluster_scopes",
    "convert_tensor_to_tile_ops",
    "optimize_orch_tensors",
    "lower_composite_ops",
    "flatten_tile_nd_to_2d",
)


def _lower(kernel, stop_after="flatten_tile_nd_to_2d"):
    """Run the Default prefix up to and including @p stop_after.

    Every pass runs under the UT-wide ``BEFORE_AND_AFTER`` verification installed by
    ``tests/ut/conftest.py``, so these helpers double as the regression: before the
    repair, the pass that narrows the matmul left a ``ForStmt`` declaring 64 valid rows
    and yielding ``min(v, 64)``, and verification rejected it on the spot.
    """
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    program = _jit_program(kernel)
    for name in _TENSOR_PREFIX:
        program = getattr(passes, name)()(program)
        if name == stop_after:
            break
    return program


class _CallCollector(ir.IRVisitor):
    """Every ``Call`` to a named operator, in traversal order."""

    def __init__(self, op_name):
        super().__init__()
        self.op_name = op_name
        self.calls = []

    def visit_call(self, op: ir.Call) -> None:
        if op.op.name == self.op_name:
            self.calls.append(op)
        super().visit_call(op)


def _calls(program, op_name):
    collector = _CallCollector(op_name)
    collector.visit_program(program)
    return collector.calls


class _LoopCollector(ir.IRVisitor):
    """Every ``ForStmt`` in the program, in traversal order."""

    def __init__(self):
        super().__init__()
        self.loops = []

    def visit_for_stmt(self, op: ir.ForStmt) -> None:
        self.loops.append(op)
        super().visit_for_stmt(op)


def _loops(program):
    collector = _LoopCollector()
    collector.visit_program(program)
    return collector.loops


def _incore_functions(program):
    """The device side of a lowered program; the host orchestration has its own backend."""
    return [
        func
        for func in program.functions.values()
        if func.func_type in (pl.FunctionType.InCore, pl.FunctionType.AIC, pl.FunctionType.AIV)
    ]


class _AssignVarCollector(ir.IRVisitor):
    """Every ``AssignStmt`` var whose name starts with a prefix."""

    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.vars = []

    def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
        if op.var.name_hint.startswith(self.prefix):
            self.vars.append(op.var)
        super().visit_assign_stmt(op)


def _assign_vars(program, prefix):
    collector = _AssignVarCollector(prefix)
    collector.visit_program(program)
    return collector.vars


def _stored_tile_type(program):
    """The TileType the single ``tile.store`` reads."""
    stores = _calls(program, _TILE_STORE_OP)
    assert len(stores) == 1, f"expected one tile.store, got {len(stores)}"
    tile_type = stores[0].args[0].type
    assert isinstance(tile_type, ir.TileType)
    return tile_type


def _tile_view(tile_type):
    """The tile's explicit view, which every assertion below requires it to carry."""
    view = tile_type.tile_view
    assert view is not None, f"expected an explicit TileView on {tile_type}"
    return view


def _valid_rows(tile_type):
    view = tile_type.tile_view
    if view is None or not view.valid_shape:
        return tile_type.shape[0]
    return view.valid_shape[0]


def _is_const(expr, value):
    return isinstance(expr, ir.ConstInt) and expr.value == value


def test_flatten_repairs_an_nd_seeded_carry():
    """The issue #2470 reproducer: the store reads the narrowed, compact accumulator.

    An ND ``pl.create_tensor`` seed is only narrowed when ``FlattenTileNdTo2D`` unrolls
    the ``tile.batch_matmul`` into 2D matmuls, so that is the pass that has to repair the
    carry. Reaching this assertion at all means the repaired program also passed the
    ``BEFORE_AND_AFTER`` verification the UT suite installs.
    """
    after = _lower(create_tensor_seeded_acc)
    stored = _stored_tile_type(after)

    assert not _is_const(_valid_rows(stored), M_TILE), (
        f"the store must read the accumulator at the extent the matmuls wrote, got {_valid_rows(stored)}"
    )
    assert _tile_view(stored).compact == ir.CompactMode.normal, (
        "a row-narrowed accumulator is packed at ceil(validRow/16)*16; without compact "
        "its reader recomputes the physical row pitch instead"
    )
    # The identity `tile.reshape` between the loop and the store must not re-derive the
    # layout from the shape -- that yields the flat default and loses Acc's NZ box.
    assert _tile_view(stored).blayout == ir.TileLayout.col_major
    assert _tile_view(stored).slayout == ir.TileLayout.row_major
    assert stored.memory_space == ir.MemorySpace.Acc


def test_convert_tensor_to_tile_ops_repairs_a_2d_seeded_carry():
    """A 2D seed is narrowed one pass earlier, so it is repaired one pass earlier.

    ``tensor.matmul`` drops its operands' ``valid_shape``, so the carry is still
    consistent at pipeline input; it stops being consistent the moment
    ``ConvertTensorToTileOps`` produces a ``tile.matmul`` over a row-narrowed left
    operand. Stopping the prefix right there pins that the repair happens in the same
    pass rather than several passes downstream.
    """
    after = _lower(create_tensor_seeded_acc_2d, stop_after="convert_tensor_to_tile_ops")
    stored = _stored_tile_type(after)

    assert not _is_const(_valid_rows(stored), M_TILE)
    assert _tile_view(stored).compact == ir.CompactMode.normal


def test_seed_is_redeclared_as_a_compact_narrowed_box():
    """The narrowed seed is declared, not stamped: ``tile.create(compact=True)``.

    A pass-applied type refinement is discarded the moment a later pass re-deduces the
    call (``InferTileMemorySpace`` does), whereas the kwarg is re-read by the deducer.
    ``tile.set_validshape`` then inherits the mode onto the narrowed view without
    re-interpreting bytes it did not write.
    """
    after = _lower(create_tensor_seeded_acc)

    compact_creates = [call for call in _calls(after, _TILE_CREATE_OP) if call.kwargs.get("compact")]
    assert len(compact_creates) == 1, "the re-declared seed is the only compact tile.create"
    assert compact_creates[0].kwargs.get("target_memory") == ir.MemorySpace.Acc

    aliases = _calls(after, _TILE_SET_VALIDSHAPE_OP)
    assert aliases, "the compact box is narrowed through tile.set_validshape"
    assert any(_tile_view(alias.type).compact == ir.CompactMode.normal for alias in aliases)


def test_a_repaired_carry_reaches_a_later_loops_carry():
    """A carry initialised by a re-typed value is re-minted, and its body follows.

    Substituting the init alone would leave the second loop's ``IterArg`` at the seed's
    width while its init is narrow -- the body would deduce against the stale type and the
    ``TypeCheck`` diagnostic (which this suite installs for every pass) would reject the
    program before this assertion ran.
    """
    after = _lower(carry_flows_into_a_later_loop)
    stored = _stored_tile_type(after)

    assert not _is_const(_valid_rows(stored), M_TILE)
    assert _tile_view(stored).compact == ir.CompactMode.normal
    # Both loops carry the accumulator, and both must declare the narrowed extent.
    carries = [
        arg
        for loop in _loops(after)
        for arg in loop.iter_args
        if isinstance(arg.type, ir.TileType) and arg.type.memory_space == ir.MemorySpace.Acc
    ]
    assert len(carries) == 2, f"expected both K loops to carry the accumulator, got {len(carries)}"
    assert all(not _is_const(_valid_rows(arg.type), M_TILE) for arg in carries)


_TILE_BATCH_MATMUL_ACC_OP = ir.get_op("tile.batch_matmul_acc").name
_ALIAS_NAME = "acc_alias"


class _InsertCarryAlias(ir.IRMutator):
    """Put a bare SSA copy of the carry between the ``IterArg`` and its accumulate.

    ``alias = acc`` is a legal assignment whose value is not an operator call. A repair
    that only re-types call results would rewrite the copy's right-hand side and leave the
    Var it binds at the seed's width — an asymmetry `AssignTypeSymmetry` and the
    `TypeCheck` diagnostic both reject, and a dead end for the propagation, since the
    accumulate downstream still reads the old type through the alias.
    """

    def visit_assign_stmt(self, op: ir.AssignStmt) -> ir.Stmt:
        rebuilt = super().visit_assign_stmt(op)
        if not isinstance(rebuilt, ir.AssignStmt):
            return rebuilt
        call = rebuilt.value
        if not isinstance(call, ir.Call) or call.op.name != _TILE_BATCH_MATMUL_ACC_OP:
            return rebuilt
        carry = call.args[0]
        if not isinstance(carry, ir.IterArg):
            return rebuilt

        alias = ir.Var(_ALIAS_NAME, carry.type, rebuilt.span)
        through_alias = ir.Call(
            call.op,
            [alias, *list(call.args[1:])],
            dict(call.kwargs),
            call.type,
            call.span,
        )
        return ir.SeqStmts(
            [
                ir.AssignStmt(alias, carry, rebuilt.span),
                ir.AssignStmt(rebuilt.var, through_alias, rebuilt.span),
            ],
            rebuilt.span,
        )


def test_a_bare_alias_of_the_carry_is_re_typed():
    """The propagation must cross an assignment whose value is not a call.

    Reaching the assertions at all means the repaired program passed the
    ``BEFORE_AND_AFTER`` verification this suite installs — which is where an alias left
    at the old type is caught, as an assign-type asymmetry.
    """
    before = _lower(create_tensor_seeded_acc, stop_after="lower_composite_ops")
    with_alias = _InsertCarryAlias().visit_program(before)
    assert _assign_vars(with_alias, _ALIAS_NAME), "the alias was not injected"

    after = passes.flatten_tile_nd_to_2d()(with_alias)

    aliases = _assign_vars(after, _ALIAS_NAME)
    assert aliases, "the repair dropped the alias"
    for var in aliases:
        assert isinstance(var.type, ir.TileType)
        assert not _is_const(_valid_rows(var.type), M_TILE), (
            f"the alias still binds the seed's full height: {var.type}"
        )
    assert not _is_const(_valid_rows(_stored_tile_type(after)), M_TILE)


def test_single_fractal_block_accumulator_is_left_alone():
    """`ceil(validRow/16)*16 == Rows` leaves writer and reader agreeing anyway.

    A `[16, N]` accumulator packs to its own box whatever its valid rows, so the compact
    flag cannot change a reader's pitch — the same exemption `AccCompactValid` makes. This
    is pypto-lib's `qkv_proj_rope` shape, and re-declaring it there was both unnecessary
    and harmful: its row count is computed inside the loop, so the seed could not name it,
    and codegen was left with a symbol it could not bind.

    The carry still outlives a narrower yield, which is why this lowers without the
    verification instrument — that general defect is not what this repair claims.
    """
    with passes.PassContext([]):
        after = _lower(single_fractal_block_acc)
    stored = _stored_tile_type(after)

    assert not [call for call in _calls(after, _TILE_CREATE_OP) if call.kwargs.get("compact")]
    assert not _calls(after, _TILE_SET_VALIDSHAPE_OP)
    assert stored.tile_view is None or stored.tile_view.compact != ir.CompactMode.normal


def test_single_fractal_block_accumulator_still_reaches_codegen():
    """And it compiles all the way to PTO, which is how pypto-lib builds.

    Property verification stays on here (it is what `PassContext([])` keeps); only the
    per-pass diagnostic instrument is dropped, exactly as a production compile runs.
    Codegen is the step that catches a hoisted extent: re-declaring the seed with a row
    count computed inside the loop leaves `pto.tstore`'s valid extent naming a symbol the
    kernel cannot bind to a dimension, a scalar parameter, or a loop variable.
    """
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([]):
        lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
            _jit_program(single_fractal_block_acc)
        )
    device_side = ir.Program(_incore_functions(lowered), lowered.name, ir.Span.unknown())
    mlir = codegen.PTOCodegen().generate(device_side)

    assert "pto.tstore" in mlir


def test_extent_computed_inside_the_loop_is_declined_loudly():
    """An extent the seed cannot name is declined, not hoisted.

    The re-declared seed sits before the loop, so an extent computed in the body is not in
    scope there. Hoisting it would leave codegen with a symbol it cannot bind to a
    dimension, a scalar parameter, or a loop variable — a worse failure than the one this
    repair exists to fix. Here the pitches *do* differ, so declining leaves a program that
    would corrupt data; `AccCompactValid` is what makes that a compile error instead, and
    this test pins that the two decisions compose into a loud failure rather than a silent
    one.
    """
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

    with passes.PassContext([]):
        after = _lower(extent_computed_inside_the_loop)
    assert not [call for call in _calls(after, _TILE_CREATE_OP) if call.kwargs.get("compact")]
    assert _is_const(_valid_rows(_stored_tile_type(after)), M_TILE)

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    with pytest.raises(pypto.Error, match="AccCompactValid"):
        with passes.PassContext([]):
            PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
                _jit_program(extent_computed_inside_the_loop)
            )


def test_full_height_carry_is_left_alone():
    """Nothing narrows, so nothing is re-declared and the historical form survives."""
    after = _lower(full_height_acc)
    stored = _stored_tile_type(after)

    assert _is_const(_valid_rows(stored), M_TILE)
    assert not [call for call in _calls(after, _TILE_CREATE_OP) if call.kwargs.get("compact")]
    assert not _calls(after, _TILE_SET_VALIDSHAPE_OP)


def test_vec_carry_is_left_alone():
    """Only L0C carries are re-declared.

    A Vec seed may hold bytes the first iteration is entitled to read at full height,
    and no Vec reader derives a stride from the valid row count, so narrowing one would
    risk changing what a program computes to fix nothing. The kernel below therefore
    stays type-check invalid, which is why it lowers without the verification instrument:
    a carry that outlives a narrower yield is a general defect, and this repair only
    claims the case where it also corrupts data.
    """
    with passes.PassContext([]):
        after = _lower(vec_carry_narrowed_by_yield)

    carries = [
        iter_arg
        for loop in _loops(after)
        for iter_arg in loop.iter_args
        if isinstance(iter_arg.type, ir.TileType)
    ]
    assert carries, "the Vec kernel must still have its carry"
    assert all(_is_const(_valid_rows(carry.type), M_TILE) for carry in carries)
    assert not [call for call in _calls(after, _TILE_CREATE_OP) if call.kwargs.get("compact")]


@pytest.mark.parametrize("kernel", [create_tensor_seeded_acc, create_tensor_seeded_acc_2d])
def test_default_pipeline_accepts_the_narrowed_carry(kernel):
    """The whole Default pipeline, verification on, must accept both reproducers.

    Before the repair the ``AccCompactValid`` property verifier rejected them after
    ``InferTileMemorySpace``: every ``tile.matmul_acc`` accumulated a runtime row count
    into a full-height, non-compact buffer.
    """
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    pass_manager = PassManager.get_strategy(OptimizationStrategy.Default)
    result = pass_manager.run_passes(_jit_program(kernel))

    assert result is not None


def test_emitted_pto_stores_at_the_pitch_mad_wrote_at():
    """End to end: the ``pto.tstore`` reads a compact tile at a runtime row extent.

    This is the shape of the defect as it reached the device -- ``TSTORE`` walked L0C at
    ``srcStride = TileData::Rows`` (the compile-time 64) while ``mad`` had written it at
    ``ceil(validRow/16)*16``, so store N-fractal j picked up matmul N-fractal 4j and only
    the first 16 columns of each block survived. Both halves of the agreement are visible
    in the emitted MLIR: ``compact=1`` on the stored tile, and a dynamic (``?``) row
    extent on the destination view, which is what makes
    ``TStoreAccNz2nd``'s ``validRow == gShape3`` precondition hold.
    """
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
        _jit_program(create_tensor_seeded_acc)
    )
    device_side = ir.Program(_incore_functions(lowered), lowered.name, ir.Span.unknown())
    mlir = codegen.PTOCodegen().generate(device_side)

    stores = [line.strip() for line in mlir.splitlines() if "pto.tstore" in line]
    assert len(stores) == 1, mlir
    assert "compact=1" in stores[0], (
        f"the stored accumulator must carry the packed pitch mad wrote at:\n{stores[0]}"
    )
    assert "partition_tensor_view<?x" in stores[0], (
        f"the destination must follow the tile's runtime valid rows:\n{stores[0]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

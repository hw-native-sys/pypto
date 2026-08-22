# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Issue #2232: M/N-tile a canonical loop-carried ``tile.matmul_acc`` reduction."""

import re

import pypto.language as pl
import pytest
from pypto import backend as _backend
from pypto import ir, passes
from pypto.backend import BackendType

_TILE_STORE_OP = ir.get_op("tile.store").name

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


def test_issue_2232_canonical_input_shape():
    """Pin the real loop/if/matmul_acc shape seen by AutoTileMatmulL0."""
    before = _lower_to_auto_tile_input(_jit_program(issue_2232_repro))
    printed = ir.python_print(before)
    assert "pl.tile.matmul_acc(" in printed
    assert "pl.pipeline(" in printed
    assert "if " in printed


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


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_canonical_split_k_chooser_accounts_for_full_window_boxing(planner):
    """The pre-phase must not emit an Acc that overflows only after N boxing.

    Running the complete Default pipeline is the allocation regression: the
    PyPTO planner rejects an L0C arena above 128 KiB, so this test also proves
    the corrected candidate survives all downstream physical accounting. The
    chooser unit test separately pins the reviewer's exact (656,80,768) case.
    """
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    before = _lower_to_auto_tile_input(BoxedCapacityBefore)
    after_auto_tile = passes.auto_tile_matmul_l0()(before)
    printed = ir.python_print(after_auto_tile)

    assert "pl.Tile[[576, 64], pl.INT32, pl.Mem.Acc" not in printed
    assert "pl.Tile[[576, 48], pl.INT32, pl.Mem.Acc" not in printed
    with passes.PassContext([], memory_planner=planner):
        assert (
            PassManager.get_strategy(OptimizationStrategy.Default).run_passes(BoxedCapacityBefore) is not None
        )


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
    # The N=32/valid-16 Acc box may be spelled either on the `alloc_tile` itself or
    # restored by a following `set_validshape`: the predicated `tile.matmul_acc` the
    # K-loop now carries has no if-phi handle to hang the declaration off.
    assert re.search(
        r"(valid_col = %c16_index|set_validshape [^\n]*, %c16_index) : "
        r"!pto\.tile_buf<loc=acc, dtype=i32, rows=(128|144), cols=32,",
        pto,
    ), pto
    assert re.search(
        r"!pto\.tile_buf<loc=right, dtype=i8, rows=192, cols=32, "
        r"v_row=\?, v_col=\?, blayout=row_major, slayout=col_major, fractal=512, pad=0, compact=1>",
        pto,
    ), pto
    assert "!pto.tile_buf<loc=mat, dtype=i8, rows=128, cols=16," not in pto
    assert "pto.tmov" not in pto, f"accumulator chains must coalesce without tile.move:\n{pto}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

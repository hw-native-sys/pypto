# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Issue #2232: M/N-tile a canonical loop-carried ``tile.matmul_acc`` reduction."""

import pypto.language as pl
import pytest
from pypto import backend as _backend
from pypto import ir, passes
from pypto.backend import BackendType

M = 16
N = 1152
K_TOTAL = 1024
K_TILE = 128
N_TOTAL = N * 8

WIDE_M = 272
WIDE_N = 144
WIDE_K_TOTAL = 256
WIDE_K_TILE = 128


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
        if call.op.name != "tile.store":
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
        if op.op.name == "tile.store":
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

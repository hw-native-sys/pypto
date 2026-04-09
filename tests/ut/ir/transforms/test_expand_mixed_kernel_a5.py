# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ExpandMixedKernel pass.

Most tests use Before/After style with ir.assert_structural_equal.
Tests involving MemorySpace.Bias (not expressible in the DSL) use per-function
structural equality for AIV plus string-based assertions for AIC.
"""

import re

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType

# ---------------------------------------------------------------------------
# Backend fixture: expand_mixed_kernel now requires Ascend950 backend
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure Ascend950 backend before each test and reset afterward."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    yield
    backend.reset_for_testing()


# ---------------------------------------------------------------------------
# Shared helpers: program builders and pass invocation
# ---------------------------------------------------------------------------


_AUTO_PIPE_SETUP_OPS = {
    "system.reserve_buffer",
    "system.import_peer_buffer",
    "system.aic_initialize_pipe",
    "system.aiv_initialize_pipe",
}

_AUTO_TFREE_OPS = {
    "system.tfree_to_aic",
    "system.tfree_to_aiv",
}


def _expand_raw(program):
    """Run convert_to_ssa, infer_tile_memory_space then expand_mixed_kernel."""
    return passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(passes.convert_to_ssa()(program)))


def _get_stmt_op_name(stmt):
    """Return the op name for AssignStmt/EvalStmt calls, or empty string."""
    if (
        isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and isinstance(stmt.value.op, ir.Op)
    ):
        return stmt.value.op.name
    if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call) and isinstance(stmt.expr.op, ir.Op):
        return stmt.expr.op.name
    return ""


def _flatten_top_level_stmts(body):
    """Return top-level statements as a Python list."""
    if isinstance(body, ir.SeqStmts):
        return list(body.stmts)
    return [body]


def _make_stmt_body(stmts, span):
    """Rebuild a body from a top-level statement list."""
    if not stmts:
        return ir.SeqStmts([], span)
    if len(stmts) == 1:
        return stmts[0]
    return ir.SeqStmts(stmts, span)


def _strip_auto_tfree_from_stmt(stmt):
    """Strip auto-generated tfree statements recursively."""
    if _get_stmt_op_name(stmt) in _AUTO_TFREE_OPS:
        return None
    if isinstance(stmt, ir.ForStmt):
        new_body = _strip_auto_tfree_from_body(stmt.body)
        return ir.ForStmt(
            stmt.loop_var,
            stmt.start,
            stmt.stop,
            stmt.step,
            stmt.iter_args,
            new_body,
            stmt.return_vars,
            stmt.span,
            stmt.kind,
            stmt.chunk_size,
            stmt.chunk_policy,
            stmt.attrs,
        )
    if isinstance(stmt, ir.IfStmt):
        new_then = _strip_auto_tfree_from_body(stmt.then_body)
        new_else = _strip_auto_tfree_from_body(stmt.else_body) if stmt.else_body is not None else None
        return ir.IfStmt(stmt.condition, new_then, new_else, stmt.return_vars, stmt.span)
    if isinstance(stmt, ir.WhileStmt):
        new_body = _strip_auto_tfree_from_body(stmt.body)
        return ir.WhileStmt(stmt.condition, stmt.iter_args, new_body, stmt.return_vars, stmt.span)
    return stmt


def _strip_auto_tfree_from_body(body):
    """Strip auto-generated tfree statements from a statement body."""
    rebuilt = []
    for stmt in _flatten_top_level_stmts(body):
        new_stmt = _strip_auto_tfree_from_stmt(stmt)
        if new_stmt is not None:
            rebuilt.append(new_stmt)
    return _make_stmt_body(rebuilt, body.span)


def _strip_auto_pipe_setup_function(func):
    """Strip auto-generated pipe setup prologue from AIC/AIV functions for legacy comparisons."""
    if func.func_type not in (pl.FunctionType.AIC, pl.FunctionType.AIV):
        return func
    stmts = _flatten_top_level_stmts(func.body)
    index = 0
    while index < len(stmts) and _get_stmt_op_name(stmts[index]) in _AUTO_PIPE_SETUP_OPS:
        index += 1
    if index == 0:
        body = _strip_auto_tfree_from_body(func.body)
    else:
        body = _strip_auto_tfree_from_body(_make_stmt_body(stmts[index:], func.body.span))
    params = list(zip(func.params, func.param_directions))
    return ir.Function(
        func.name, params, func.return_types, body, func.span, func.func_type, func.level, func.role
    )


def _strip_auto_pipe_setup(program):
    """Strip auto-generated pipe setup from all functions in a program."""
    rebuilt_funcs = [_strip_auto_pipe_setup_function(func) for func in program.functions.values()]
    return ir.Program(rebuilt_funcs, program.name, program.span)


def _expand(program):
    """Expand mixed kernels, then strip auto-generated pipe setup for structural tests."""
    return _strip_auto_pipe_setup(_expand_raw(program))


def _assert_function_equal(actual_program, expected_program, func_name):
    """Assert that a named function matches between two programs."""
    actual = actual_program.get_function(func_name)
    expected_program_ssa = passes.convert_to_ssa()(expected_program)
    expected = expected_program_ssa.get_function(func_name)
    assert actual is not None, f"Function '{func_name}' not found in actual program"
    assert expected is not None, f"Function '{func_name}' not found in expected program"
    ir.assert_structural_equal(actual, expected)


def _make_matmul_program():
    """Standard mixed kernel: load->Mat->Left/Right, matmul, move->Vec, store."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 128], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z_tile = pl.matmul(x_left, y_right)
            z_vec = pl.move(
                z_tile,
                target_memory=pl.MemorySpace.Vec,
                blayout=pl.TileLayout.row_major,
                slayout=pl.TileLayout.none_box,
            )
            out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
            return out_0

    return P


# ---------------------------------------------------------------------------
# Pass-through: programs that should NOT be split
# ---------------------------------------------------------------------------


class TestPassthrough:
    """Tests where the program is not split (pure vector, orchestration, pure cube).

    Non-mixed InCore functions get their FunctionType converted to AIC or AIV.
    """

    def test_pure_vector_becomes_aiv(self):
        """InCore with only vector ops -> no split, FunctionType becomes AIV."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile = pl.load(x, [0], [64])
                y_tile = pl.add(x_tile, x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile = pl.load(x, [0], [64])
                y_tile = pl.add(x_tile, x_tile)
                out_0_store: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0_store

        ir.assert_structural_equal(After, Expected)

    def test_orchestration_unchanged(self):
        """Non-InCore functions pass through unchanged."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

        Inferred = passes.infer_tile_memory_space()(Before)
        After = passes.expand_mixed_kernel()(Inferred)
        ir.assert_structural_equal(After, Inferred)

    def test_pure_cube_becomes_aic(self):
        """InCore with only cube ops (no Acc->Vec boundary) -> no split, FunctionType becomes AIC."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0_store

        ir.assert_structural_equal(After, Expected)

    def test_pure_vector_inside_loop_becomes_aiv(self):
        """InCore with only vector ops inside a loop -> no split, FunctionType becomes AIV."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(4):
                    x_tile = pl.load(x, [0], [64])
                    y_tile = pl.add(x_tile, x_tile)
                    out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(4):
                    x_tile = pl.load(x, [0], [64])
                    y_tile = pl.add(x_tile, x_tile)
                    out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

        ir.assert_structural_equal(After, passes.convert_to_ssa()(Expected))

    def test_pure_vector_with_split_attr_clears_split(self):
        """InCore with split=UP_DOWN but only vector ops -> AIV with split cleared (Fixes #841)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile = pl.load(x, [0], [64])
                y_tile = pl.add(x_tile, x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

        After = _expand_raw(Before)

        func = After.get_function("main_incore_0")
        assert func is not None
        assert func.func_type == pl.FunctionType.AIV
        assert func.split is None, f"Expected split=None for pure AIV, got {func.split}"

    def test_pure_cube_with_split_attr_clears_split(self):
        """InCore with split=UP_DOWN but only cube ops -> AIC with split cleared (Fixes #841)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0_store

        After = _expand_raw(Before)

        func = After.get_function("main_incore_0")
        assert func is not None
        assert func.func_type == pl.FunctionType.AIC
        assert func.split is None, f"Expected split=None for pure AIC, got {func.split}"


# ---------------------------------------------------------------------------
# Split structure: AIC / AIV / Group function properties
# ---------------------------------------------------------------------------


class TestSplitStructure:
    """Test the structure of generated AIC, AIV, and Group functions using Before/After."""

    def test_matmul_split_before_after(self):
        """Standard matmul split: Before/After with ir.assert_structural_equal."""
        Before = _make_matmul_program()
        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.main_incore_0_aic(x, y, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0_aiv(x, y, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_function_count_after_split(self):
        """After splitting 1 mixed InCore: original 1 func -> 3 (AIC + AIV + Group)."""
        Before = _make_matmul_program()
        assert len(Before.functions) == 1

        After = _expand(Before)
        assert len(After.functions) == 3


# ---------------------------------------------------------------------------
# Cross-core boundary detection and TPUSH/TPOP insertion
# ---------------------------------------------------------------------------


class TestCrossCoreBoundaries:
    """Test C<->V boundary detection and cross-core communication ops."""

    def test_matmul_exp_split_before_after(self):
        """matmul + exp: C->V boundary produces tpush/tpop, exp stays in AIV."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                w_tile = pl.exp(z_vec)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(w_tile, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                w_tile = pl.exp(z_vec)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(w_tile, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.main_incore_0_aic(x, y, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0_aiv(x, y, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_v2c_boundary_add_to_matmul(self):
        """Pre-matmul vector op: add(x,x) produces V->C boundary to matmul."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x_tile = pl.load(x, [0, 0], [16, 128])
                x_sum = pl.add(x_tile, x_tile)
                x_sum_mat = pl.move(
                    x_sum,
                    target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                x_sum_left = pl.move(x_sum_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_sum_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[16, 64], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ):
                x_sum_mat: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemorySpace.Mat,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.tpop_from_aiv(split=0)
                x_sum_left = pl.move(x_sum_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_sum_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x_tile = pl.load(x, [0, 0], [16, 128])
                x_sum = pl.add(x_tile, x_tile)
                x_sum_nz = pl.move(
                    x_sum,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                pl.tpush_to_aic(x_sum_nz, split=0)
                z_vec: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 64], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                self.main_incore_0_aic(x, y, out_0)
                result: pl.Tensor[[16, 64], pl.FP32] = self.main_incore_0_aiv(x, y, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_v2c_boundary_direct_to_left_keeps_mat_tpop(self):
        """Direct V->C move to Left must become Mat tpop followed by move."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x_tile = pl.load(x, [0, 0], [16, 128])
                x_left = pl.move(x_tile, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[16, 64], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ):
                x_left_mat: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemorySpace.Mat,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.tpop_from_aiv(split=0)
                x_left = pl.move(
                    x_left_mat,
                    target_memory=pl.MemorySpace.Left,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x_tile = pl.load(x, [0, 0], [16, 128])
                x_tile_nz = pl.move(
                    x_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                pl.tpush_to_aic(x_tile_nz, split=0)
                z_vec: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 64], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                self.main_incore_0_aic(x, y, out_0)
                result: pl.Tensor[[16, 64], pl.FP32] = self.main_incore_0_aiv(x, y, out_0)
                return result

        ir.assert_structural_equal(After, Expected)


# ---------------------------------------------------------------------------
# Cube op variant classification
# ---------------------------------------------------------------------------


class TestCubeOpVariants:
    """Test that all cube op variants are correctly classified and placed in AIC."""

    def test_matmul_acc_in_aic(self):
        """matmul + matmul_acc -> both in AIC, none in AIV."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                c_tile = pl.matmul(a_left, b_right)
                d_tile = pl.matmul_acc(c_tile, a_left, b_right)
                d_vec = pl.move(
                    d_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(d_vec, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                c_tile = pl.matmul(a_left, b_right)
                d_tile = pl.matmul_acc(c_tile, a_left, b_right)
                pl.tpush_to_aiv(d_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                d_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(d_vec, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.main_incore_0_aic(a, b, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0_aiv(a, b, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_matmul_bias_in_aic(self):
        """tile.matmul_bias is a CUBE op -> triggers split with bias V->C boundary."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[1, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                bias: pl.Tensor[[1, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                a_mat = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
                a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                bias_tile = pl.load(bias, [0, 0], [1, 128])
                bias_mat = pl.move(
                    bias_tile,
                    target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                c_tile = pl.matmul_bias(a_left, b_right, bias_mat)
                c_vec = pl.move(
                    c_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[1, 128], pl.FP32] = pl.store(c_vec, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        # AIV Expected (no Bias MemorySpace, so DSL can express it)
        @pl.program
        class ExpectedAIV:
            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                a: pl.Tensor[[1, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                bias: pl.Tensor[[1, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                bias_tile = pl.load(bias, [0, 0], [1, 128])
                bias_tile_nz = pl.move(
                    bias_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                pl.tpush_to_aic(bias_tile_nz, split=0)
                c_vec: pl.Tile[[1, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[1, 128], pl.FP32] = pl.store(c_vec, [0, 0], out_0)
                return out_0_store

        _assert_function_equal(After, ExpectedAIV, "main_incore_0_aiv")

        # AIC uses MemorySpace.Bias (not expressible in DSL), verify via string
        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        assert aic_func.func_type == pl.FunctionType.AIC
        aic_str = aic_func.as_python()
        assert "matmul_bias" in aic_str
        assert "tpop_from_aiv" in aic_str
        assert "tpush_to_aiv" in aic_str
        assert "Mem.Bias" in aic_str

        # Group calls AIC then AIV
        group_func = After.get_function("main_incore_0")
        assert group_func is not None
        assert group_func.func_type == pl.FunctionType.Group
        group_str = group_func.as_python()
        assert "main_incore_0_aic" in group_str
        assert "main_incore_0_aiv" in group_str

    def test_gemv_in_aic(self):
        """tile.gemv is a CUBE op -> triggers split."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                c_tile = pl.gemv(a_left, b_right)
                c_vec = pl.move(
                    c_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(c_vec, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                c_tile = pl.gemv(a_left, b_right)
                pl.tpush_to_aiv(c_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                c_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(c_vec, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.main_incore_0_aic(a, b, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0_aiv(a, b, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_gemv_acc_in_aic(self):
        """tile.gemv_acc is a CUBE op -> triggers split."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                c_tile = pl.gemv(a_left, b_right)
                a_left2 = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_right2 = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                d_tile = pl.gemv_acc(c_tile, a_left2, b_right2)
                d_vec = pl.move(
                    d_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(d_vec, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                c_tile = pl.gemv(a_left, b_right)
                a_left2 = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_right2 = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                d_tile = pl.gemv_acc(c_tile, a_left2, b_right2)
                pl.tpush_to_aiv(d_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                d_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(d_vec, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.main_incore_0_aic(a, b, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0_aiv(a, b, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_gemv_bias_in_aic(self):
        """tile.gemv_bias is a CUBE op -> triggers split with bias V->C boundary."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[1, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                bias: pl.Tensor[[1, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                a_mat = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
                a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                bias_tile = pl.load(bias, [0, 0], [1, 128])
                bias_mat = pl.move(
                    bias_tile,
                    target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                c_tile = pl.gemv_bias(a_left, b_right, bias_mat)
                c_vec = pl.move(
                    c_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[1, 128], pl.FP32] = pl.store(c_vec, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        # AIV Expected (no Bias MemorySpace, so DSL can express it)
        @pl.program
        class ExpectedAIV:
            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                a: pl.Tensor[[1, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                bias: pl.Tensor[[1, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                bias_tile = pl.load(bias, [0, 0], [1, 128])
                bias_tile_nz = pl.move(
                    bias_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                pl.tpush_to_aic(bias_tile_nz, split=0)
                c_vec: pl.Tile[[1, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[1, 128], pl.FP32] = pl.store(c_vec, [0, 0], out_0)
                return out_0_store

        _assert_function_equal(After, ExpectedAIV, "main_incore_0_aiv")

        # AIC uses MemorySpace.Bias (not expressible in DSL), verify via string
        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        assert aic_func.func_type == pl.FunctionType.AIC
        aic_str = aic_func.as_python()
        assert "gemv_bias" in aic_str
        assert "tpop_from_aiv" in aic_str
        assert "tpush_to_aiv" in aic_str
        assert "Mem.Bias" in aic_str

        # Group calls AIC then AIV
        group_func = After.get_function("main_incore_0")
        assert group_func is not None
        assert group_func.func_type == pl.FunctionType.Group
        group_str = group_func.as_python()
        assert "main_incore_0_aic" in group_str
        assert "main_incore_0_aiv" in group_str


# ---------------------------------------------------------------------------
# Vector op classification
# ---------------------------------------------------------------------------


class TestVectorOpClassification:
    """Test that vector ops are correctly classified and placed in AIV."""

    def test_sub_is_vector(self):
        """tile.sub should be in AIV with V->C boundary, not AIC."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x_tile = pl.load(x, [0, 0], [16, 128])
                x_sub = pl.sub(x_tile, x_tile)
                x_sub_mat = pl.move(
                    x_sub,
                    target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                x_sub_left = pl.move(x_sub_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_sub_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[16, 64], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ):
                x_sub_mat: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemorySpace.Mat,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.tpop_from_aiv(split=0)
                x_sub_left = pl.move(x_sub_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_sub_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x_tile = pl.load(x, [0, 0], [16, 128])
                x_sub = pl.sub(x_tile, x_tile)
                x_sub_nz = pl.move(
                    x_sub,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                pl.tpush_to_aic(x_sub_nz, split=0)
                z_vec: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 64], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                self.main_incore_0_aic(x, y, out_0)
                result: pl.Tensor[[16, 64], pl.FP32] = self.main_incore_0_aiv(x, y, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_dn_transpose_moves_in_aic(self):
        """Cube moves (Mat->Left/Right) with DN layout and transpose stay in AIC."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16, pl.DN],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_l1 = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_l1, target_memory=pl.MemorySpace.Left)
                y_l1 = pl.load(
                    y,
                    [0, 0],
                    [128, 128],
                    target_memory=pl.MemorySpace.Mat,
                    transpose=True,
                )
                y_right = pl.move(y_l1, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                z_exp = pl.exp(z_vec)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_exp, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16, pl.DN],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                x_l1 = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_l1, target_memory=pl.MemorySpace.Left)
                y_l1 = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat, transpose=True)
                y_right = pl.move(y_l1, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16, pl.DN],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                z_exp = pl.exp(z_vec)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_exp, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16, pl.DN],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.main_incore_0_aic(x, y, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0_aiv(x, y, out_0)
                return result

        ir.assert_structural_equal(After, Expected)


# ---------------------------------------------------------------------------
# Realistic computation patterns
# ---------------------------------------------------------------------------


class TestRealisticPatterns:
    """Test realistic computation patterns (attention, post-processing chains)."""

    def test_attention_pattern_split(self):
        """matmul -> exp -> add: AIC gets matmul, AIV gets exp+add+store."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                exp_tile = pl.exp(z_vec)
                sum_tile = pl.add(exp_tile, exp_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(sum_tile, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                exp_tile = pl.exp(z_vec)
                sum_tile = pl.add(exp_tile, exp_tile)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(sum_tile, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.main_incore_0_aic(x, y, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0_aiv(x, y, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_matmul_chain_vector_postprocessing(self):
        """matmul -> exp -> mul -> store: multiple vector post-ops in AIV."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                z_exp = pl.exp(z_vec)
                z_mul = pl.mul(z_exp, z_exp)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_mul, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                z_exp = pl.exp(z_vec)
                z_mul = pl.mul(z_exp, z_exp)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_mul, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.main_incore_0_aic(x, y, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0_aiv(x, y, out_0)
                return result

        ir.assert_structural_equal(After, Expected)


# ---------------------------------------------------------------------------
# Multiple InCore functions
# ---------------------------------------------------------------------------


class TestMultipleInCore:
    """Test behavior with multiple InCore functions in a program."""

    def test_multiple_mixed_functions(self):
        """Two mixed InCore functions -> both are split independently."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def compute_a_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.InCore)
            def compute_b_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                c_tile = pl.gemv(a_left, b_right)
                c_vec = pl.move(
                    c_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(c_vec, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def compute_a_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def compute_a_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def compute_a_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.compute_a_incore_0_aic(x, y, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.compute_a_incore_0_aiv(x, y, out_0)
                return result

            @pl.function(type=pl.FunctionType.AIC)
            def compute_b_incore_0_aic(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                c_tile = pl.gemv(a_left, b_right)
                pl.tpush_to_aiv(c_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def compute_b_incore_0_aiv(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                c_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(c_vec, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def compute_b_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.compute_b_incore_0_aic(a, b, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.compute_b_incore_0_aiv(a, b, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_mixed_plus_pure_incore(self):
        """One mixed + one pure vector InCore -> only mixed is split."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def pure_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile = pl.load(x, [0], [64])
                y_tile = pl.add(x_tile, x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.InCore)
            def mixed_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV)
            def pure_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile = pl.load(x, [0], [64])
                y_tile = pl.add(x_tile, x_tile)
                out_0_store: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.AIC)
            def mixed_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def mixed_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def mixed_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.mixed_incore_0_aic(x, y, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.mixed_incore_0_aiv(x, y, out_0)
                return result

        ir.assert_structural_equal(After, Expected)


# ---------------------------------------------------------------------------
# Property verification
# ---------------------------------------------------------------------------


class TestPropertyVerification:
    """Test property verification behavior with ExpandMixedKernel."""

    def test_produces_mixed_kernel_expanded_property(self):
        """After pass runs, MixedKernelExpanded property should be verifiable."""
        After = _expand_raw(_make_matmul_program())

        prop_set = passes.IRPropertySet()
        prop_set.insert(passes.IRProperty.MixedKernelExpanded)
        passes.verify_properties(prop_set, After, "test")

    def test_verification_with_after_mode_instrument(self):
        """Property verification instrument works after expand."""
        Before = _make_matmul_program()

        instrument = passes.VerificationInstrument(passes.VerificationMode.AFTER)
        with passes.PassContext([instrument]):
            After = _expand_raw(Before)

        assert After.get_function("main_incore_0_aic") is not None

    def test_verifier_rejects_aic_tpop_from_aiv_into_left(self):
        """AIC tpop_from_aiv must land in Mat, not Left."""

        @pl.program
        class BadProgram:
            @pl.function(type=pl.FunctionType.AIC)
            def bad_aic(self):
                _: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemorySpace.Left,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.tpop_from_aiv(split=0)

        prop_set = passes.IRPropertySet()
        prop_set.insert(passes.IRProperty.MixedKernelExpanded)

        with pytest.raises(Exception, match=re.escape("tile.tpop_from_aiv result in MemorySpace::Mat")):
            passes.verify_properties(prop_set, BadProgram, "test")

    def test_verifier_rejects_aiv_tpop_from_aic_into_mat(self):
        """AIV tpop_from_aic must land in Vec, not Mat."""

        @pl.program
        class BadProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def bad_aiv(self):
                _: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat, pl.TileView()] = pl.tpop_from_aic(split=0)

        prop_set = passes.IRPropertySet()
        prop_set.insert(passes.IRProperty.MixedKernelExpanded)

        with pytest.raises(Exception, match=re.escape("tile.tpop_from_aic result in MemorySpace::Vec")):
            passes.verify_properties(prop_set, BadProgram, "test")

    def test_verifier_rejects_cross_core_missing_pipe_setup(self):
        """AIC/AIV kernels with tpush/tpop must provide pipe setup."""

        @pl.program
        class BadProgram:
            @pl.function(type=pl.FunctionType.AIC)
            def bad_aic(self):
                _: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemorySpace.Mat,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.tpop_from_aiv(split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def bad_aiv(self):
                x: pl.Tile[[16, 128], pl.FP32, pl.TileView()] = pl.tpop_from_aic(split=0)
                _ = pl.exp(x)

        prop_set = passes.IRPropertySet()
        prop_set.insert(passes.IRProperty.MixedKernelExpanded)

        with pytest.raises(
            Exception,
            match=re.escape("uses cross-core tile ops but has no 'system.aic_initialize_pipe' call"),
        ):
            passes.verify_properties(prop_set, BadProgram, "test")

    def test_verifier_rejects_inconsistent_bidirectional_slot_sizes(self):
        """Bidirectional kernels must use a single slot size across both directions."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile = pl.load(x, [0, 0], [16, 128])
                x_sum = pl.add(x_tile, x_tile)
                x_sum_mat = pl.move(x_sum, target_memory=pl.MemorySpace.Mat)
                x_sum_left = pl.move(x_sum_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_sum_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0

        with pytest.raises(Exception, match="single initialize_pipe slot_size is unsupported"):
            _expand_raw(Before)

    def test_verifier_rejects_missing_tfree_for_tpop(self):
        """Cross-core tpop chains must release their slot with a matching tfree."""

        @pl.program
        class BadProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def bad_aiv(self):
                pipe_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=pipe_buf)
                _: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(split=0)

        prop_set = passes.IRPropertySet()
        prop_set.insert(passes.IRProperty.MixedKernelExpanded)

        with pytest.raises(
            Exception,
            match=re.escape("uses tile.tpop_from_aic but has no matching 'system.tfree_to_aic' call"),
        ):
            passes.verify_properties(prop_set, BadProgram, "test")

    def test_verifier_rejects_next_tpop_before_tfree(self):
        """A second tpop cannot appear before the previous chain is freed."""

        @pl.program
        class BadProgram:
            @pl.function(type=pl.FunctionType.AIC)
            def bad_aic(self):
                pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
                pl.aic_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=pipe_buf)
                first: pl.Tile[
                    [16, 16],
                    pl.FP16,
                    pl.MemorySpace.Mat,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.tpop_from_aiv(split=0)
                second: pl.Tile[
                    [16, 16],
                    pl.FP16,
                    pl.MemorySpace.Mat,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.tpop_from_aiv(split=0)
                first_left = pl.move(first, target_memory=pl.MemorySpace.Left)
                second_right = pl.move(second, target_memory=pl.MemorySpace.Right)
                pl.tfree_to_aiv(first)
                pl.tfree_to_aiv(second)
                _ = pl.matmul(first_left, second_right)

        prop_set = passes.IRPropertySet()
        prop_set.insert(passes.IRProperty.MixedKernelExpanded)

        with pytest.raises(
            Exception, match="must order cross-core tpop chains as 'tpop -> use -> tfree -> next tpop'"
        ):
            passes.verify_properties(prop_set, BadProgram, "test")

    def test_verifier_rejects_late_pipe_setup(self):
        """Pipe setup must appear before the first cross-core op."""

        @pl.program
        class BadProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def bad_aiv(self):
                popped: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                processed = pl.exp(popped)
                pl.tfree_to_aic(popped)
                pipe_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=pipe_buf)
                _ = processed

        prop_set = passes.IRPropertySet()
        prop_set.insert(passes.IRProperty.MixedKernelExpanded)

        with pytest.raises(
            Exception,
            match=re.escape("uses cross-core tile ops but has no 'system.aiv_initialize_pipe' call"),
        ):
            passes.verify_properties(prop_set, BadProgram, "test")


class TestAutoPipeSetup:
    """Test auto-generated reserve/import/initialize_pipe setup."""

    def test_auto_pipe_setup_inserted_for_c2v_kernel(self):
        """C2V-only kernels should auto-insert import on AIC and reserve on AIV."""
        After = _expand_raw(_make_matmul_program())

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()

        assert "import_peer_buffer" in aic_str
        assert "main_incore_0_c2v_slot_buffer" in aic_str
        assert (
            "pl.system.aic_initialize_pipe(main_incore_0_c2v_slot_buffer_import, pl.const(0, pl.INT32), "
            "dir_mask=1, slot_size=8192)" in aic_str
        )
        assert "reserve_buffer" in aiv_str
        assert "main_incore_0_c2v_slot_buffer" in aiv_str
        assert (
            "pl.system.aiv_initialize_pipe(main_incore_0_c2v_slot_buffer, pl.const(0, pl.INT32), "
            "dir_mask=1, slot_size=8192)" in aiv_str
        )
        assert "tfree_to_aic" in aiv_str

    def test_auto_tfree_inserted_after_post_tpop_move(self):
        """Auto-generated tfree should appear after the post-tpop move on the consumer side."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x_tile = pl.load(x, [0, 0], [16, 128])
                x_left = pl.move(x_tile, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[16, 64], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0

        After = _expand_raw(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ):
                main_incore_0_v2c_slot_buffer = pl.reserve_buffer(
                    name="main_incore_0_v2c_slot_buffer", size=16384
                )
                main_incore_0_c2v_slot_buffer_import = pl.import_peer_buffer(
                    name="main_incore_0_c2v_slot_buffer",
                    peer_func="main_incore_0_aiv",
                )
                pl.aic_initialize_pipe(
                    main_incore_0_c2v_slot_buffer_import,
                    main_incore_0_v2c_slot_buffer,
                    dir_mask=3,
                    slot_size=4096,
                )
                x_left_mat: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemorySpace.Mat,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.tpop_from_aiv(split=0)
                x_left = pl.move(
                    x_left_mat,
                    target_memory=pl.MemorySpace.Left,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                pl.tfree_to_aiv(x_left_mat)
                y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

        _assert_function_equal(After, Expected, "main_incore_0_aic")

    def test_auto_tfree_stays_after_last_use_with_intermediate_dependency(self):
        """Auto tfree should stay after the true last use of the popped tile."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 16], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_tile = pl.load(y, [0, 0], [128, 16], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_tile, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                z_exp = pl.exp(z_vec)
                z_sum = pl.add(z_vec, z_exp)
                out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(z_sum, [0, 0], out_0)
                return out_0

        After = _expand_raw(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 16], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                main_incore_0_c2v_slot_buffer = pl.reserve_buffer(
                    name="main_incore_0_c2v_slot_buffer", size=8192
                )
                pl.aiv_initialize_pipe(
                    main_incore_0_c2v_slot_buffer,
                    pl.const(0, pl.INT32),
                    dir_mask=1,
                    slot_size=1024,
                )
                z_vec: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                z_exp = pl.exp(z_vec)
                z_sum = pl.add(z_vec, z_exp)
                pl.tfree_to_aic(z_vec)
                out_0_store: pl.Tensor[[16, 16], pl.FP32] = pl.store(z_sum, [0, 0], out_0)
                return out_0_store

        _assert_function_equal(After, Expected, "main_incore_0_aiv")

    def test_auto_tfree_does_not_hoist_user_before_if_defined_tile(self):
        """A later tpop user must stay after an if-defined tile result."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                flag: pl.Scalar[pl.INDEX],
                x: pl.Tensor[[4, 32], pl.BF16],
                y: pl.Tensor[[32, 32], pl.BF16],
                out_0: pl.Out[pl.Tensor[[4, 32], pl.FP32]],
            ) -> pl.Tensor[[4, 32], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [4, 32], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                if flag == 0:
                    branch_tile = pl.exp(z_vec)
                else:
                    branch_tile = z_vec
                mixed = pl.add(z_vec, branch_tile)
                out_0: pl.Tensor[[4, 32], pl.FP32] = pl.store(mixed, [0, 0], out_0)
                return out_0

        After = _expand_raw(Before)
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()

        # Keep this as an order assertion: the PL DSL cannot express this Expected
        # function without triggering a TileView mismatch between the if branches
        # before structural_equal runs.
        assert aiv_str.index("tpop_from_aic") < aiv_str.index("if flag__ssa_v0 == 0:")
        assert aiv_str.index("if flag__ssa_v0 == 0:") < aiv_str.index("add(")


# ---------------------------------------------------------------------------
# Nested structures (for loops)
# ---------------------------------------------------------------------------


class TestNestedStructures:
    """Test that mixed ops inside ForStmt are handled recursively."""

    def test_for_loop_split_and_boundaries(self):
        """Mixed ops inside a for loop -> AIC/AIV each get the loop; TPUSH/TPOP inside."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                for i in pl.range(4):
                    x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                    x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                    y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                    y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                    z_tile = pl.matmul(x_left, y_right)
                    z_vec = pl.move(
                        z_tile,
                        target_memory=pl.MemorySpace.Vec,
                        blayout=pl.TileLayout.row_major,
                        slayout=pl.TileLayout.none_box,
                    )
                    out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        # Per-function comparison avoids loop-variable clash in structural equality
        @pl.program
        class ExpAIC:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                for i in pl.range(4):
                    x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                    x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                    y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                    y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                    z_tile = pl.matmul(x_left, y_right)
                    pl.tpush_to_aiv(z_tile, split=0)

        @pl.program
        class ExpAIV:
            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                for i in pl.range(4):
                    z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                        split=0
                    )
                    out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0

        _assert_function_equal(After, ExpAIC, "main_incore_0_aic")
        _assert_function_equal(After, ExpAIV, "main_incore_0_aiv")

        group_func = After.get_function("main_incore_0")
        assert group_func is not None
        assert group_func.func_type == pl.FunctionType.Group

    def test_bidirectional_inside_for_loop(self):
        """V->C and C->V boundaries inside same loop body.

        Pattern: load(Vec) -> add (V) -> move(Vec->Mat->Left) -> matmul (C)
                 -> move(Acc->Vec) -> exp (V) -> store
        V->C: add result flows to matmul via tpush_to_aic / tpop_from_aiv
        C->V: matmul result flows to exp via tpush_to_aiv / tpop_from_aic
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                for i in pl.range(4):
                    x_tile = pl.load(x, [0, 0], [16, 128])
                    x_sum = pl.add(x_tile, x_tile)
                    x_sum_mat = pl.move(
                        x_sum,
                        target_memory=pl.MemorySpace.Mat,
                        blayout=pl.TileLayout.col_major,
                        slayout=pl.TileLayout.row_major,
                    )
                    x_sum_left = pl.move(x_sum_mat, target_memory=pl.MemorySpace.Left)
                    y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                    y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                    z_tile = pl.matmul(x_sum_left, y_right)
                    z_vec = pl.move(
                        z_tile,
                        target_memory=pl.MemorySpace.Vec,
                        blayout=pl.TileLayout.row_major,
                        slayout=pl.TileLayout.none_box,
                    )
                    w_tile = pl.exp(z_vec)
                    out_0: pl.Tensor[[16, 64], pl.FP32] = pl.store(w_tile, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        # Per-function comparison avoids loop-variable clash in structural equality
        @pl.program
        class ExpAIC:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ):
                for i in pl.range(4):
                    x_sum_mat: pl.Tile[
                        [16, 128],
                        pl.BF16,
                        pl.MemorySpace.Mat,
                        pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                    ] = pl.tpop_from_aiv(split=0)
                    x_sum_left = pl.move(x_sum_mat, target_memory=pl.MemorySpace.Left)
                    y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                    y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                    z_tile = pl.matmul(x_sum_left, y_right)
                    pl.tpush_to_aiv(z_tile, split=0)

        @pl.program
        class ExpAIV:
            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                for i in pl.range(4):
                    x_tile = pl.load(x, [0, 0], [16, 128])
                    x_sum = pl.add(x_tile, x_tile)
                    x_sum_nz = pl.move(
                        x_sum,
                        target_memory=pl.MemorySpace.Vec,
                        blayout=pl.TileLayout.col_major,
                        slayout=pl.TileLayout.row_major,
                    )
                    pl.tpush_to_aic(x_sum_nz, split=0)
                    z_vec: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                        split=0
                    )
                    w_tile = pl.exp(z_vec)
                    out_0: pl.Tensor[[16, 64], pl.FP32] = pl.store(w_tile, [0, 0], out_0)
                return out_0

        _assert_function_equal(After, ExpAIC, "main_incore_0_aic")
        _assert_function_equal(After, ExpAIV, "main_incore_0_aiv")

        group_func = After.get_function("main_incore_0")
        assert group_func is not None
        assert group_func.func_type == pl.FunctionType.Group

    def test_mixed_loop_plus_flat_ops(self):
        """load(Mat) outside loop + mixed ops inside loop."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                for i in pl.range(2):
                    x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                    y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                    y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                    z_tile = pl.matmul(x_left, y_right)
                    z_vec = pl.move(
                        z_tile,
                        target_memory=pl.MemorySpace.Vec,
                        blayout=pl.TileLayout.row_major,
                        slayout=pl.TileLayout.none_box,
                    )
                    out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        # Per-function comparison avoids loop-variable clash in structural equality
        @pl.program
        class ExpAIC:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                for i in pl.range(2):
                    x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                    y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                    y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                    z_tile = pl.matmul(x_left, y_right)
                    pl.tpush_to_aiv(z_tile, split=0)

        @pl.program
        class ExpAIV:
            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                for i in pl.range(2):
                    z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                        split=0
                    )
                    out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0

        _assert_function_equal(After, ExpAIC, "main_incore_0_aic")
        _assert_function_equal(After, ExpAIV, "main_incore_0_aiv")

        group_func = After.get_function("main_incore_0")
        assert group_func is not None
        assert group_func.func_type == pl.FunctionType.Group


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_existing_group_from_cluster_outline(self):
        """Existing Group caller is rewritten to call AIC+AIV; no redundant Group wrapper."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def compute_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Group)
            def compute_group(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                result: pl.Tensor[[16, 128], pl.FP32] = self.compute_incore_0(x, y, out_0)
                return result

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def compute_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def compute_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def compute_group(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.compute_incore_0_aic(x, y, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.compute_incore_0_aiv(x, y, out_0)
                return result

        ir.assert_structural_equal(After, Expected)


# ---------------------------------------------------------------------------
# Regression: DCE must preserve loop iter_args, yield values, and tpop ops
# ---------------------------------------------------------------------------


class TestDCERegression:
    """Regression tests for DCE and loop-state cleanup in mixed loops."""

    def test_nested_loop_store_result_remapped_to_param(self):
        """Regression for nested AIC-side stores not seen by top-level remap logic.

        The loop-local cube store stays on AIC, while the trailing cast+store stays on
        AIV. The AIV body must treat the loop store result as the updated output tensor
        parameter rather than leaving a dangling Var reference from inside the loop.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                for i in pl.range(2):
                    x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                    x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                    y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                    y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                    z_tile = pl.matmul(x_left, y_right)
                    out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                x_tile = pl.load(x, [0, 0], [16, 128])
                x_fp32 = pl.tile.cast(x_tile, target_type=pl.FP32, mode="round")
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(x_fp32, [0, 0], out_0)
                return out_0

        After = _expand_raw(Before)

        prop_set = passes.IRPropertySet()
        prop_set.insert(passes.IRProperty.MixedKernelExpanded)
        passes.verify_properties(prop_set, After, "test")
        passes.run_verifier()(After)

        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "pl.tile.store" in aiv_str
        assert "return" in aiv_str

    def test_loop_accumulation_preserves_yield_and_init_values(self):
        """Regression for bugs 1+2: mixed loop with iter_args.

        Pattern: acc=0; for i { acc += matmul(x, w) }; store(acc)
        Before fix: DCE removed tile.create, tile.muls, tpop, tile.add.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                acc_0 = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                acc_1 = pl.tile.muls(acc_0, 0.0)
                for i, (acc_iter,) in pl.range(4, init_values=(acc_1,)):
                    x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                    x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                    w_mat = pl.load(w, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                    w_right = pl.move(w_mat, target_memory=pl.MemorySpace.Right)
                    z = pl.matmul(x_left, w_right)
                    z_vec = pl.move(
                        z,
                        target_memory=pl.MemorySpace.Vec,
                        blayout=pl.TileLayout.row_major,
                        slayout=pl.TileLayout.none_box,
                    )
                    acc_new = pl.tile.add(acc_iter, z_vec)
                    acc_out = pl.yield_(acc_new)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(acc_out, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                for i in pl.range(4):
                    x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                    x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                    w_mat = pl.load(w, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                    w_right = pl.move(w_mat, target_memory=pl.MemorySpace.Right)
                    z = pl.matmul(x_left, w_right)
                    pl.tpush_to_aiv(z, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                acc_0 = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                acc_1 = pl.tile.muls(acc_0, 0.0)
                for i, (acc_iter,) in pl.range(4, init_values=(acc_1,)):
                    z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                        split=0
                    )
                    acc_new = pl.tile.add(acc_iter, z_vec)
                    acc_out = pl.yield_(acc_new)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(acc_out, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.main_incore_0_aic(x, w, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0_aiv(x, w, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_bidirectional_loop_accumulation(self):
        """Regression for bugs 1+2: V->C and C->V boundaries inside accumulation loop.

        Pattern: for i { normed = add(x,x); matmul(normed, w); acc += result }
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                acc_0 = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                acc_1 = pl.tile.muls(acc_0, 0.0)
                for i, (acc_iter,) in pl.range(4, init_values=(acc_1,)):
                    x_tile = pl.load(x, [0, 0], [16, 128])
                    x_sum = pl.add(x_tile, x_tile)
                    x_mat = pl.move(
                        x_sum,
                        target_memory=pl.MemorySpace.Mat,
                        blayout=pl.TileLayout.col_major,
                        slayout=pl.TileLayout.row_major,
                    )
                    x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                    w_mat = pl.load(w, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                    w_right = pl.move(w_mat, target_memory=pl.MemorySpace.Right)
                    z = pl.matmul(x_left, w_right)
                    z_vec = pl.move(
                        z,
                        target_memory=pl.MemorySpace.Vec,
                        blayout=pl.TileLayout.row_major,
                        slayout=pl.TileLayout.none_box,
                    )
                    acc_new = pl.tile.add(acc_iter, z_vec)
                    acc_out = pl.yield_(acc_new)
                out_0: pl.Tensor[[16, 64], pl.FP32] = pl.store(acc_out, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ):
                for i in pl.range(4):
                    x_sum_mat: pl.Tile[
                        [16, 128],
                        pl.BF16,
                        pl.MemorySpace.Mat,
                        pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                    ] = pl.tpop_from_aiv(split=0)
                    x_left = pl.move(x_sum_mat, target_memory=pl.MemorySpace.Left)
                    w_mat = pl.load(w, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                    w_right = pl.move(w_mat, target_memory=pl.MemorySpace.Right)
                    z = pl.matmul(x_left, w_right)
                    pl.tpush_to_aiv(z, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                acc_0 = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                acc_1 = pl.tile.muls(acc_0, 0.0)
                for i, (acc_iter,) in pl.range(4, init_values=(acc_1,)):
                    x_tile = pl.load(x, [0, 0], [16, 128])
                    x_sum = pl.add(x_tile, x_tile)
                    x_sum_nz = pl.move(
                        x_sum,
                        target_memory=pl.MemorySpace.Vec,
                        blayout=pl.TileLayout.col_major,
                        slayout=pl.TileLayout.row_major,
                    )
                    pl.tpush_to_aic(x_sum_nz, split=0)
                    z_vec: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                        split=0
                    )
                    acc_new = pl.tile.add(acc_iter, z_vec)
                    acc_out = pl.yield_(acc_new)
                out_0_store: pl.Tensor[[16, 64], pl.FP32] = pl.store(acc_out, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                self.main_incore_0_aic(x, w, out_0)
                result: pl.Tensor[[16, 64], pl.FP32] = self.main_incore_0_aiv(x, w, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_tpop_preserved_when_result_unused(self):
        """Regression for bug 4: tpop must be preserved even when its result is unused.

        If AIC pushes a value (tpush is a side effect, always kept) but AIV's
        tpop result is dead, removing the tpop desynchronizes the communication
        queues, causing deadlock at runtime.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                w_mat = pl.load(w, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                w_right = pl.move(w_mat, target_memory=pl.MemorySpace.Right)
                z = pl.matmul(x_left, w_right)
                _z_vec = pl.move(
                    z,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                # z_vec is dead — only a separate load+cast+store returns a value
                x_tile = pl.load(x, [0, 0], [16, 128])
                x_fp32 = pl.tile.cast(x_tile, target_type=pl.FP32, mode="round")
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(x_fp32, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                w_mat = pl.load(w, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                w_right = pl.move(w_mat, target_memory=pl.MemorySpace.Right)
                z = pl.matmul(x_left, w_right)
                pl.tpush_to_aiv(z, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                _z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                x_tile = pl.load(x, [0, 0], [16, 128])
                x_fp32 = pl.tile.cast(x_tile, target_type=pl.FP32, mode="round")
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(x_fp32, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.main_incore_0_aic(x, w, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0_aiv(x, w, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_multiple_iter_args_preserved(self):
        """Regression for bugs 1+2: multiple iter_args (gate+up accumulators).

        Models the MLP gate/up accumulation pattern from Qwen3.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[4, 256], pl.BF16],
                wg: pl.Tensor[[256, 32], pl.BF16],
                wu: pl.Tensor[[256, 32], pl.BF16],
                out_0: pl.Out[pl.Tensor[[4, 32], pl.FP32]],
            ) -> pl.Tensor[[4, 32], pl.FP32]:
                gate_0 = pl.tile.create([4, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                up_0 = pl.tile.create([4, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                gate_1 = pl.tile.muls(gate_0, 0.0)
                up_1 = pl.tile.muls(up_0, 0.0)
                for i, (gate_iter, up_iter) in pl.range(2, init_values=(gate_1, up_1)):
                    x_mat = pl.load(x, [0, 0], [4, 256], target_memory=pl.MemorySpace.Mat)
                    x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                    wg_mat = pl.load(wg, [0, 0], [256, 32], target_memory=pl.MemorySpace.Mat)
                    wg_right = pl.move(wg_mat, target_memory=pl.MemorySpace.Right)
                    g_tile = pl.matmul(x_left, wg_right)
                    g_vec = pl.move(g_tile, target_memory=pl.MemorySpace.Vec)
                    wu_mat = pl.load(wu, [0, 0], [256, 32], target_memory=pl.MemorySpace.Mat)
                    wu_right = pl.move(wu_mat, target_memory=pl.MemorySpace.Right)
                    u_tile = pl.matmul(x_left, wu_right)
                    u_vec = pl.move(u_tile, target_memory=pl.MemorySpace.Vec)
                    gate_new = pl.tile.add(gate_iter, g_vec)
                    up_new = pl.tile.add(up_iter, u_vec)
                    gate_out, up_out = pl.yield_(gate_new, up_new)
                result = pl.tile.add(gate_out, up_out)
                out_0: pl.Tensor[[4, 32], pl.FP32] = pl.store(result, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        # AIV Expected — both accumulators fully preserved
        @pl.program
        class ExpAIV:
            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[4, 256], pl.BF16],
                wg: pl.Tensor[[256, 32], pl.BF16],
                wu: pl.Tensor[[256, 32], pl.BF16],
                out_0: pl.Out[pl.Tensor[[4, 32], pl.FP32]],
            ) -> pl.Tensor[[4, 32], pl.FP32]:
                gate_0 = pl.tile.create([4, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                up_0 = pl.tile.create([4, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                gate_1 = pl.tile.muls(gate_0, 0.0)
                up_1 = pl.tile.muls(up_0, 0.0)
                for i, (gate_iter, up_iter) in pl.range(2, init_values=(gate_1, up_1)):
                    g_vec: pl.Tile[[4, 32], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                        split=0
                    )
                    gate_new = pl.tile.add(gate_iter, g_vec)
                    u_vec: pl.Tile[[4, 32], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                        split=0
                    )
                    up_new = pl.tile.add(up_iter, u_vec)
                    gate_out, up_out = pl.yield_(gate_new, up_new)
                result = pl.tile.add(gate_out, up_out)
                out_0_store: pl.Tensor[[4, 32], pl.FP32] = pl.store(result, [0, 0], out_0)
                return out_0_store

        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "for i__idx_v0, (gate_iter, up_iter) in pl.range(2, init_values=(" in aiv_str
        assert aiv_str.count("pl.tile.tpop_from_aic") == 2
        assert "gate_new" in aiv_str
        assert "up_new" in aiv_str
        assert "gate_out, up_out = pl.yield_(" in aiv_str
        assert "result" in aiv_str and "pl.tile.add(gate_out, up_out)" in aiv_str
        assert "pl.tile.store(" in aiv_str and "result" in aiv_str

        # AIC — dead iter_args stripped, clean counted loop
        aic_str = str(After.get_function("main_incore_0_aic"))
        assert "tile.matmul" in aic_str
        assert "tile.tpush_to_aiv" in aic_str
        assert "init_values=" not in aic_str
        assert "pl.yield_(" not in aic_str

    def test_alive_cube_iter_arg_keeps_init_value_defs(self):
        """Regression for issue #533: alive AIC iter_arg keeps its init-value defs.

        Pattern: matmul_acc accumulates into a CUBE iter_arg across iterations.
        The return_var feeds a C→V boundary move after the loop.
        Dead VECTOR iter_arg is stripped; alive CUBE iter_arg is kept with
        init value definitions pulled from the original body and proper yield.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                # First matmul to create Acc-typed init value for CUBE iter_arg
                a_mat_0 = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                a_left_0 = pl.move(a_mat_0, target_memory=pl.MemorySpace.Left)
                b_mat_0 = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                b_right_0 = pl.move(b_mat_0, target_memory=pl.MemorySpace.Right)
                cube_init = pl.matmul(a_left_0, b_right_0)
                # Vec init for VECTOR iter_arg
                vec_init = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                vec_zero = pl.tile.muls(vec_init, 0.0)
                for i, (vec_acc, cube_carry) in pl.range(
                    4,
                    init_values=(vec_zero, cube_init),
                ):
                    a_mat = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                    a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                    b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                    b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                    # CUBE: matmul_acc accumulates into cube_carry
                    z = pl.matmul_acc(cube_carry, a_left, b_right)
                    z_vec = pl.move(z, target_memory=pl.MemorySpace.Vec)
                    # VECTOR: accumulate in Vec
                    vec_new = pl.tile.add(vec_acc, z_vec)
                    vec_out, cube_out = pl.yield_(vec_new, z)
                # After loop: BOUNDARY move uses cube return_var
                final_vec = pl.move(cube_out, target_memory=pl.MemorySpace.Vec)
                result = pl.tile.add(vec_out, final_vec)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(result, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        # AIC: cube_carry alive (matmul_acc uses it + boundary move after loop)
        # vec_acc dead (only VECTOR consumers) → stripped
        # Note: full structural equality not feasible here because the pass infers
        # Acc-typed TileViews on iter_args that the DSL cannot construct directly.
        aic_str = str(After.get_function("main_incore_0_aic"))
        assert "init_values=" in aic_str, "alive CUBE iter_arg must keep init_values"
        assert "pl.yield_(" in aic_str, "alive CUBE iter_arg must keep yield"
        assert "tile.matmul_acc" in aic_str
        assert "tile.tpush_to_aiv" in aic_str
        assert "cube_init" in aic_str, "alive iter_arg init-value definition must stay available"

        # AIV: vec_acc alive, cube_carry dead → stripped
        aiv_str = str(After.get_function("main_incore_0_aiv"))
        assert "tile.tpop_from_aic" in aiv_str
        assert "tile.add" in aiv_str
        assert "init_values=" in aiv_str, "alive VEC iter_arg must keep init_values"
        assert "pl.yield_(" in aiv_str, "alive VEC iter_arg must keep yield"
        assert "tile.store" in aiv_str

    def test_conditional_branch_yield_falls_back_to_iter_arg(self):
        """Regression for issue #534: branch-local dangling yields become identity yields.

        The AIC side prunes the VECTOR accumulation in the `if` branch, so the
        branch yield must fall back to `acc_iter` instead of referencing the
        stripped `acc_then` value.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                acc_0 = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                acc_1 = pl.tile.muls(acc_0, 0.0)
                for i, (acc_iter,) in pl.range(4, init_values=(acc_1,)):
                    if i == 0:
                        x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                        x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                        w_mat = pl.load(w, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                        w_right = pl.move(w_mat, target_memory=pl.MemorySpace.Right)
                        z = pl.matmul(x_left, w_right)
                        z_vec = pl.move(
                            z,
                            target_memory=pl.MemorySpace.Vec,
                            blayout=pl.TileLayout.row_major,
                            slayout=pl.TileLayout.none_box,
                        )
                        acc_then = pl.tile.add(acc_iter, z_vec)
                        branch_out = pl.yield_(acc_then)
                    else:
                        branch_out = pl.yield_(acc_iter)
                    acc_out = pl.yield_(branch_out)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(acc_out, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                acc_0 = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                acc_1 = pl.tile.muls(acc_0, 0.0)
                for i, (acc_iter,) in pl.range(4, init_values=(acc_1,)):
                    if i == 0:
                        x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                        x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                        w_mat = pl.load(w, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                        w_right = pl.move(w_mat, target_memory=pl.MemorySpace.Right)
                        z = pl.matmul(x_left, w_right)
                        pl.tpush_to_aiv(z, split=0)
                        branch_out = pl.yield_(acc_iter)
                    else:
                        branch_out = pl.yield_(acc_iter)
                    acc_out = pl.yield_(  # noqa: F841
                        branch_out
                    )

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                acc_0 = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                acc_1 = pl.tile.muls(acc_0, 0.0)
                for i, (acc_iter,) in pl.range(4, init_values=(acc_1,)):
                    if i == 0:
                        z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = (
                            pl.tpop_from_aic(split=0)
                        )
                        acc_then = pl.tile.add(acc_iter, z_vec)
                        branch_out = pl.yield_(acc_then)
                    else:
                        branch_out = pl.yield_(acc_iter)
                    acc_out = pl.yield_(branch_out)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(acc_out, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.main_incore_0_aic(x, w, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0_aiv(x, w, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_transient_shared_alias_does_not_keep_dead_iter_arg_alive(self):
        """Dead iter_args should be re-stripped after DCE removes shared-only aliases.

        Pattern:
        1. loop carries a Vec value that is only meaningful on AIV
        2. a shared alias assignment after the loop temporarily references the loop return_var
        3. that alias is dead on AIC and disappears in DCE

        AIC must not keep the Vec init-value chain or the loop iter_arg after the
        shared alias is eliminated.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                vec_init = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                acc_init = pl.tile.muls(vec_init, 0.0)
                for i, (acc_iter,) in pl.range(4, init_values=(acc_init,)):
                    x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                    x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                    w_mat = pl.load(w, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                    w_right = pl.move(w_mat, target_memory=pl.MemorySpace.Right)
                    z = pl.matmul(x_left, w_right)
                    tmp = pl.move(
                        z,
                        target_memory=pl.MemorySpace.Vec,
                        blayout=pl.TileLayout.row_major,
                        slayout=pl.TileLayout.none_box,
                    )
                    carried_next = pl.tile.add(tmp, tmp)
                    acc_out = pl.yield_(carried_next)
                carried = acc_out
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(carried, [0, 0], out_0)
                return out_0

        After = _expand(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def main_incore_0_aic(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ):
                for i in pl.range(4):
                    x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                    x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                    w_mat = pl.load(w, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                    w_right = pl.move(w_mat, target_memory=pl.MemorySpace.Right)
                    z = pl.matmul(x_left, w_right)
                    pl.tpush_to_aiv(z, split=0)

            @pl.function(type=pl.FunctionType.AIV)
            def main_incore_0_aiv(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                vec_init = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                acc_init = pl.tile.muls(vec_init, 0.0)
                for i, (acc_iter,) in pl.range(4, init_values=(acc_init,)):
                    tmp: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                        split=0
                    )
                    carried_next = pl.tile.add(tmp, tmp)
                    acc_out = pl.yield_(carried_next)
                carried = acc_out
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(carried, [0, 0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Group)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                self.main_incore_0_aic(x, w, out_0)
                result: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0_aiv(x, w, out_0)
                return result

        ir.assert_structural_equal(After, Expected)

    def test_conditional_branch_references_source_tile_of_boundary_move(self):
        """Regression for issue #584: source tile of boundary move becomes dangling.

        When a conditional branch on the AIV (pop) side references the source tile
        of a C→V boundary move (the pre-move variable), that reference must be
        remapped to the tpop result. Without the fix, only dest_var was mapped in
        tpop_var_remap, leaving the source_tile reference dangling after the split.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                w: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                acc_0 = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                acc_1 = pl.tile.muls(acc_0, 0.0)
                for i, (acc_iter,) in pl.range(4, init_values=(acc_1,)):
                    x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                    x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                    w_mat = pl.load(w, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                    w_right = pl.move(w_mat, target_memory=pl.MemorySpace.Right)
                    z = pl.matmul(x_left, w_right)
                    # Conditional branch references z (source tile), not z_vec (dest)
                    if i == 0:
                        acc_then = pl.tile.add(acc_iter, z)
                        branch_out = pl.yield_(acc_then)
                    else:
                        branch_out = pl.yield_(acc_iter)
                    acc_out = pl.yield_(branch_out)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(acc_out, [0, 0], out_0)
                return out_0

        After = _expand(Before)
        aic_func = After.get_function("main_incore_0_aic")
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aic_func is not None
        assert aiv_func is not None

        aic_str = aic_func.as_python()
        aiv_str = aiv_func.as_python()

        assert aic_str.count("pl.tile.tpush_to_aiv") == 1
        assert aiv_str.count("pl.tile.tpop_from_aic") == 1
        assert "pl.tile.add(" in aiv_str and "acc_iter" in aiv_str and "z__ssa_v0_Vec" in aiv_str
        assert "pl.yield_(acc_iter)" in aiv_str
        # tpop_from_aic must appear before the add that uses its result
        assert aiv_str.index("pl.tile.tpop_from_aic") < aiv_str.index("pl.tile.add(")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

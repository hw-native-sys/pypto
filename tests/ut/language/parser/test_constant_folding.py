# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for compile-time constant folding in parse_binop and parse_unaryop.

When all operands of a binary/unary expression are resolvable from closure
variables, the parser should fold the expression into a single ConstInt or
ConstFloat IR node rather than emitting a compound expression tree.
"""

import pypto.language as pl
import pytest
from pypto.pypto_core import ir


def _collect_call_args(func: ir.Function, op_name: str) -> list[list]:
    """Collect argument lists of all Calls matching *op_name* anywhere in the
    function body, recursing into ``ForStmt``/``WhileStmt``/``IfStmt``/``SeqStmts``
    bodies so loop-local calls are not missed.
    """
    results: list[list] = []

    def visit(node: object) -> None:
        if isinstance(node, ir.AssignStmt) and isinstance(node.value, ir.Call):
            if node.value.op.name == op_name:
                results.append(list(node.value.args))
            return
        if isinstance(node, ir.SeqStmts):
            for s in node.stmts:
                visit(s)
            return
        body = getattr(node, "body", None)
        if body is not None:
            visit(body)
        for attr in ("then_body", "else_body"):
            branch = getattr(node, attr, None)
            if branch is not None:
                visit(branch)

    visit(func.body)
    return results


class TestBinopConstantFolding:
    """Binary operations with closure-only operands fold to constants."""

    def test_floordiv_folds_to_constint(self):
        """ROPE_DIM // 2 folds to ConstInt(4), not FloorDiv(ConstInt(8), ConstInt(2))."""
        ROPE_DIM = 8

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result = pl.mul(x, ROPE_DIM // 2)
            return result

        assert isinstance(func, ir.Function)
        mul_calls = _collect_call_args(func, "tensor.muls")
        assert len(mul_calls) == 1
        scalar_arg = mul_calls[0][1]
        assert isinstance(scalar_arg, ir.ConstInt), (
            f"Expected ConstInt after folding, got {type(scalar_arg).__name__}"
        )
        assert scalar_arg.value == 4

    def test_add_folds_to_constint(self):
        """Closure constant addition A + B folds to a single ConstInt."""
        A = 10
        B = 20

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result = pl.mul(x, A + B)
            return result

        assert isinstance(func, ir.Function)
        mul_calls = _collect_call_args(func, "tensor.muls")
        assert len(mul_calls) == 1
        scalar_arg = mul_calls[0][1]
        assert isinstance(scalar_arg, ir.ConstInt), (
            f"Expected ConstInt after folding, got {type(scalar_arg).__name__}"
        )
        assert scalar_arg.value == 30

    def test_mul_folds_to_constint(self):
        """Closure constant multiplication folds correctly."""
        BASE = 64
        FACTOR = 2

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result = pl.mul(x, BASE * FACTOR)
            return result

        assert isinstance(func, ir.Function)
        mul_calls = _collect_call_args(func, "tensor.muls")
        assert len(mul_calls) == 1
        scalar_arg = mul_calls[0][1]
        assert isinstance(scalar_arg, ir.ConstInt)
        assert scalar_arg.value == 128

    def test_mod_folds_to_constint(self):
        """Closure constant modulo N % M folds correctly."""
        N = 17
        M = 5

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result = pl.mul(x, N % M)
            return result

        assert isinstance(func, ir.Function)
        mul_calls = _collect_call_args(func, "tensor.muls")
        assert len(mul_calls) == 1
        scalar_arg = mul_calls[0][1]
        assert isinstance(scalar_arg, ir.ConstInt)
        assert scalar_arg.value == 2

    def test_nested_binop_folds_to_constint(self):
        """Nested expression (A + B) // C folds to a single ConstInt."""
        A = 100
        B = 28
        C = 4

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result = pl.mul(x, (A + B) // C)
            return result

        assert isinstance(func, ir.Function)
        mul_calls = _collect_call_args(func, "tensor.muls")
        assert len(mul_calls) == 1
        scalar_arg = mul_calls[0][1]
        assert isinstance(scalar_arg, ir.ConstInt)
        assert scalar_arg.value == 32

    def test_float_div_folds_to_constfloat(self):
        """Float division A / B folds to ConstFloat."""
        A = 10.0
        B = 4.0

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result = pl.mul(x, A / B)
            return result

        assert isinstance(func, ir.Function)
        mul_calls = _collect_call_args(func, "tensor.muls")
        assert len(mul_calls) == 1
        scalar_arg = mul_calls[0][1]
        assert isinstance(scalar_arg, ir.ConstFloat), (
            f"Expected ConstFloat after folding, got {type(scalar_arg).__name__}"
        )
        assert scalar_arg.value == pytest.approx(2.5)


class TestUnaryopConstantFolding:
    """Unary operations with closure-only operands fold to constants."""

    def test_neg_folds_to_constint(self):
        """-CLOSURE_VAR folds to ConstInt(-val), not Neg(ConstInt(val))."""
        VAL = 42

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result = pl.mul(x, -VAL)
            return result

        assert isinstance(func, ir.Function)
        mul_calls = _collect_call_args(func, "tensor.muls")
        assert len(mul_calls) == 1
        scalar_arg = mul_calls[0][1]
        # After folding, -42 should become a single ConstInt(-42) or Neg(ConstInt(42)).
        # With the eval-based folder, Python evaluates -42 → int(-42) → ConstInt(-42).
        assert isinstance(scalar_arg, ir.ConstInt), (
            f"Expected ConstInt after folding, got {type(scalar_arg).__name__}"
        )
        assert scalar_arg.value == -42


class TestMixedExpressionFallback:
    """Expressions involving DSL variables must NOT be folded — they should
    produce compound IR nodes (Add, Sub, etc.) instead of ConstInt."""

    def test_dsl_var_plus_closure_not_folded(self):
        """dsl_scalar + CLOSURE produces an IR Add node, not a constant."""
        OFFSET = 10

        @pl.function
        def func(
            x: pl.Tensor[[64], pl.FP32],
            cfg: pl.Tensor[[1], pl.INDEX],
        ) -> pl.Tensor[[64], pl.FP32]:
            idx: pl.Scalar[pl.INDEX] = pl.tensor.read(cfg, [0])
            shifted: pl.Scalar[pl.INDEX] = idx + OFFSET
            result = pl.mul(x, shifted)
            return result

        assert isinstance(func, ir.Function)
        # The `idx + OFFSET` must remain an Add node, not folded
        body = func.body
        assert isinstance(body, ir.SeqStmts)
        found_add = False
        for stmt in body.stmts:
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Add):
                found_add = True
                break
        assert found_add, "Expected an ir.Add node for dsl_var + closure_const"

    def test_pure_dsl_binop_not_folded(self):
        """Operations on DSL-defined variables should not attempt folding."""

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            a = pl.add(x, x)
            b = pl.sub(a, x)
            return b

        assert isinstance(func, ir.Function)
        body = func.body
        assert isinstance(body, ir.SeqStmts)
        call_ops = [
            stmt.value.op.name
            for stmt in body.stmts
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
        ]
        assert "tensor.add" in call_ops
        assert "tensor.sub" in call_ops


class TestDimensionEqualityAfterFolding:
    """The motivating scenario: folded dimensions from the same closure
    expression compare equal in DimensionsEqual, enabling shape-dependent
    ops like pl.sub to succeed."""

    def test_same_closure_binop_in_multiple_shapes(self):
        """Two occurrences of ROPE_DIM // 2 fold to ConstInt(4), enabling DimensionsEqual."""
        ROPE_DIM = 8

        @pl.program
        class FoldedDims:
            @pl.function
            def main(
                self,
                src: pl.Tensor[[16, 8], pl.FP32],
                out: pl.Tensor[[1, 4], pl.FP32],
            ) -> pl.Tensor[[1, 4], pl.FP32]:
                lo = pl.slice(src, [1, ROPE_DIM // 2], [0, 0])
                hi = pl.slice(src, [1, ROPE_DIM // 2], [0, ROPE_DIM // 2])
                lo_scaled = pl.col_expand_mul(lo, lo)
                hi_scaled = pl.col_expand_mul(hi, hi)
                result = pl.sub(lo_scaled, hi_scaled)
                out = pl.assemble(out, result, [0, 0])
                return out

        assert isinstance(FoldedDims, ir.Program)
        printed = FoldedDims.as_python()
        assert "[1, 4]" in printed
        assert "8 // 2" not in printed
        # Roundtrip: re-parse the printed text and verify structural equality
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(FoldedDims, reparsed)

    def test_different_closure_expressions_same_value(self):
        """Different closure expressions that evaluate to the same int produce
        ConstInt nodes with equal values, so DimensionsEqual succeeds."""
        A = 16
        B = 8

        @pl.function
        def func(
            src: pl.Tensor[[16, 8], pl.FP32],
            out: pl.Tensor[[1, 8], pl.FP32],
        ) -> pl.Tensor[[1, 8], pl.FP32]:
            lo = pl.slice(src, [1, A // 2], [0, 0])
            hi = pl.slice(src, [1, B * 1], [0, 0])
            result = pl.sub(lo, hi)
            out = pl.assemble(out, result, [0, 0])
            return out

        assert isinstance(func, ir.Function)


class TestScopeShadowingSafety:
    """Folding must respect DSL scope: if a Name in the expression is already
    defined in the DSL scope, folding must be skipped even if the same name
    exists as a closure variable."""

    def test_dsl_var_shadows_closure_in_binop(self):
        """DSL-scoped variable N shadows closure N — folding must not happen."""
        N = 8  # noqa: F841 — deliberately shadowed by DSL assignment below

        @pl.function
        def func(
            x: pl.Tensor[[64], pl.FP32],
            cfg: pl.Tensor[[1], pl.INDEX],
        ) -> pl.Tensor[[64], pl.FP32]:
            N: pl.Scalar[pl.INDEX] = pl.tensor.read(cfg, [0])
            result = pl.mul(x, N // 2)
            return result

        assert isinstance(func, ir.Function)
        body = func.body
        assert isinstance(body, ir.SeqStmts)
        mul_calls = _collect_call_args(func, "tensor.muls")
        assert len(mul_calls) == 1
        scalar_arg = mul_calls[0][1]
        # Must NOT be ConstInt(4) — N is a DSL runtime variable
        assert not isinstance(scalar_arg, ir.ConstInt) or scalar_arg.value != 4, (
            "Folding incorrectly used closure value for DSL-scoped variable N"
        )

    def test_nested_shadow_partial(self):
        """If one operand is DSL-scoped and the other is closure, skip folding."""
        M = 100

        @pl.function
        def func(
            x: pl.Tensor[[64], pl.FP32],
            cfg: pl.Tensor[[1], pl.INDEX],
        ) -> pl.Tensor[[64], pl.FP32]:
            idx: pl.Scalar[pl.INDEX] = pl.tensor.read(cfg, [0])
            result = pl.mul(x, idx + M)
            return result

        assert isinstance(func, ir.Function)
        mul_calls = _collect_call_args(func, "tensor.muls")
        assert len(mul_calls) == 1
        scalar_arg = mul_calls[0][1]
        # idx + M must remain an Add node, not a folded constant
        assert isinstance(scalar_arg, ir.Add), (
            f"Expected ir.Add for mixed DSL+closure expression, got {type(scalar_arg).__name__}"
        )


def _assert_all_slice_extents_are_constint(func: ir.Function, expected_dims: list[int]) -> None:
    """Verify every ``tensor.slice`` call in *func* has shape_tuple = expected_dims."""
    slice_calls = _collect_call_args(func, "tensor.slice")
    assert slice_calls, "expected tensor.slice calls to be emitted"
    for args in slice_calls:
        extent = args[1]  # tensor.slice(tensor, shape_tuple, offset_tuple)
        assert isinstance(extent, ir.MakeTuple)
        actual = []
        for dim in extent.elements:
            assert isinstance(dim, ir.ConstInt), (
                f"slice extent dim should be folded to ConstInt, got {type(dim).__name__}: {dim}"
            )
            actual.append(dim.value)
        assert actual == expected_dims, f"expected slice shape {expected_dims}, got {actual}"


class TestSymbolicShapeEquality:
    """Symbolic dimension expressions that simplify to the same value should
    compare equal. Covers the subscript-slice pattern where extent is built
    as ``upper - lower`` with a loop induction variable on both sides."""

    def test_slice_extent_simplifies_to_constant(self):
        """``x[:, k : k + C]`` inside a loop produces a literal-C extent."""
        C = 64

        @pl.function
        def func(a: pl.Tensor[[8, 256], pl.BF16]) -> pl.Tensor[[8, 64], pl.BF16]:
            out = a[:, 0:C]
            for k in pl.range(C, 256, C):
                out = a[:, k : k + C]
            return out

        assert isinstance(func, ir.Function)
        _assert_all_slice_extents_are_constint(func, [8, C])

    def test_loop_induction_slice_extent_folds(self):
        """``x[:, i * s : (i + 1) * s]`` folds to extent ``s``."""
        S = 32

        @pl.function
        def func(a: pl.Tensor[[8, 256], pl.BF16]) -> pl.Tensor[[8, 32], pl.BF16]:
            out = a[:, 0:S]
            for i in pl.range(8):
                out = a[:, i * S : (i + 1) * S]
            return out

        assert isinstance(func, ir.Function)
        _assert_all_slice_extents_are_constint(func, [8, S])

    def test_reassign_across_loop_with_symbolic_extent(self):
        """Reassignment of a tensor variable inside a loop with a symbolic
        slice extent should succeed — the pre-fix parser rejected this with
        'Cannot reassign with a different type' because the slice shape built
        as ``k + C - k`` did not match the initial ``C`` shape."""
        C = 64

        @pl.function
        def func(a: pl.Tensor[[8, 256], pl.BF16]) -> pl.Tensor[[8, 64], pl.BF16]:
            chunk = a[:, 0:C]
            for k in pl.range(C, 256, C):
                chunk = a[:, k : k + C]
            return chunk

        assert isinstance(func, ir.Function)

    def test_symbolic_cancellation_in_broadcast_sub(self):
        """``pl.sub`` of two slices whose extents simplify to the same constant
        but differ structurally should pass shape broadcasting.  Pre-fix this
        raised 'requires compatible shapes'."""
        HALF = 32

        @pl.function
        def func(
            src: pl.Tensor[[1, 128], pl.FP32],
            out: pl.Tensor[[1, 32], pl.FP32],
        ) -> pl.Tensor[[1, 32], pl.FP32]:
            for k in pl.range(0, 64, HALF):
                lo = src[:, k : k + HALF]
                hi = src[:, k + HALF : k + HALF + HALF]
                out = pl.sub(lo, hi)
            return out

        assert isinstance(func, ir.Function)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

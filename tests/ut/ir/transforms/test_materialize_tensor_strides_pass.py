# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for MaterializeTensorStrides pass (RFC #1300, P3).

The pass walks every TensorType in a Program and replaces any
``view.has_value() && view.stride.empty()`` slot with the packed canonical
stride for the carried layout. Bare TensorTypes and already-explicit views
pass through unchanged.

After this pass runs, the codegen-entry contract holds: every
``view.has_value()`` slot has explicit stride matching its layout — which
the strict ``TensorViewCanonical`` verifier enforces.

Tests follow the Before/Expected ``@pl.program`` pattern: the pass runs on
``Before`` to produce ``After``, which is compared against ``Expected`` via
``ir.assert_structural_equal``. Skip / no-op cases compare ``After`` against
``Before``.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.pypto_core import passes as _passes

_SPAN = ir.Span.unknown()

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _materialize(program):
    return _passes.materialize_tensor_strides()(program)


def _verify_strict(program):
    """Run TensorViewCanonical in strict mode — empty stride is rejected."""
    return _passes.verify_tensor_view_canonical(program, require_materialized=True)


def _dims(shape):
    return [ir.ConstInt(s, DataType.INDEX, _SPAN) for s in shape]


def _dn_tensor(shape, stride):
    """Build a TensorType with an explicit DN-stride TensorView.

    ``stride=[]`` yields the implicit (empty-stride) form that the pass must
    materialize.
    """
    view = ir.TensorView(_dims(stride), ir.TensorLayout.DN)
    return ir.TensorType(_dims(shape), DataType.FP32, None, view)


# ============================================================================
# Bare tensor stays bare; strict verifier still passes (treated as implicit ND).
# ============================================================================


def test_bare_tensor_unchanged():
    @pl.program
    class Before:
        @pl.function
        def f(self, x: pl.Tensor[[8, 16], pl.FP32]):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    # Bare TensorType has no view to materialize: pass is a no-op.
    ir.assert_structural_equal(After, Before)
    # Strict verifier accepts a bare tensor (implicit ND).
    assert _verify_strict(After) == []


# ============================================================================
# Empty stride filled with packed canonical
# ============================================================================


def test_empty_dn_stride_filled_2d():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    ir.assert_structural_equal(After, Expected)
    # Strict verifier accepts the materialized form.
    assert _verify_strict(After) == []


def test_empty_dn_stride_filled_3d():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[2, 4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            # B=2, K=4, N=8 -> stride=[K*N, 1, K]=[32, 1, 4]
            x: pl.Tensor[[2, 4, 8], pl.FP32, pl.TensorView(stride=[32, 1, 4], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    ir.assert_structural_equal(After, Expected)
    assert _verify_strict(After) == []


def test_empty_stride_materialization_preserves_pad():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[
                [8, 16],
                pl.FP32,
                pl.TensorView(stride=[], layout=pl.TensorLayout.ND, pad=pl.PadValue.zero),
            ],
        ):
            pl.const(0, pl.INT64)

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            x: pl.Tensor[
                [8, 16],
                pl.FP32,
                pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND, pad=pl.PadValue.zero),
            ],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    ir.assert_structural_equal(After, Expected)
    assert _verify_strict(After) == []


def test_empty_default_nd_view_canonicalizes_absent():
    # Empty ND is the default TensorView and canonicalizes to no explicit view.
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[8, 16], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.ND)],
        ):
            pl.const(0, pl.INT64)

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[8, 16], pl.FP32],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    assert After is Before
    ir.assert_structural_equal(After, Expected)


# ============================================================================
# Already-explicit view stays unchanged (no spurious rewrite)
# ============================================================================


def test_explicit_packed_nd_unchanged():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[8, 16], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    # Identity preservation: pass returns the same Program when nothing changed.
    assert After is Before
    ir.assert_structural_equal(After, Before)


def test_explicit_packed_dn_unchanged():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    assert After is Before
    ir.assert_structural_equal(After, Before)


def test_strided_dn_subview_unchanged():
    # Inherited from a parent — stride larger than DN-packed for the sub-shape.
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[2, 4], pl.FP32, pl.TensorView(stride=[1, 8], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    assert After is Before
    ir.assert_structural_equal(After, Before)


# ============================================================================
# NZ on TensorType is left untouched (verifier rejects it; pass doesn't crash)
# ============================================================================


def test_nz_on_tensor_rejected_by_paired_verifier():
    # NZ on a TensorType is invalid IR. The pass leaves the slot untouched
    # rather than CHECK-failing inside BuildLogicalStridesFromLayout — but
    # because the pass produces TensorViewCanonical, PassPipeline runs the
    # paired verifier, which surfaces the bug as a thrown ValueError.
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[8, 16], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        ):
            pl.const(0, pl.INT64)

    with pytest.raises(ValueError, match="NZ layout"):
        _materialize(Before)


# ============================================================================
# Idempotence
# ============================================================================


def test_idempotent_after_first_pass():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    once = _materialize(Before)
    twice = _materialize(once)
    # Second invocation is a no-op: nothing to materialize, identity preserved.
    assert twice is once
    ir.assert_structural_equal(twice, once)


# ============================================================================
# Symbolic shape: stride expressions stay symbolic.
# ============================================================================


def test_symbolic_dn_materialized_preserves_symbols():
    K = pl.dynamic("K")
    N = pl.dynamic("N")

    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[K, N], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            # DN-packed: stride[-2] == 1, stride[-1] == K (the symbolic Var).
            x: pl.Tensor[[K, N], pl.FP32, pl.TensorView(stride=[1, K], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    ir.assert_structural_equal(After, Expected)


# ============================================================================
# Pass plays well with the canonical verifier as a paired guarantee.
# ============================================================================


def test_strict_verifier_passes_after_materialization():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)],
        ):
            pl.const(0, pl.INT64)

    # Before materialization, strict mode rejects empty stride.
    diags_before = _verify_strict(Before)
    assert any("stride is empty" in d.message for d in diags_before)
    # After materialization, strict mode accepts and IR matches Expected.
    After = _materialize(Before)
    ir.assert_structural_equal(After, Expected)
    assert _verify_strict(After) == []


# ============================================================================
# TupleType recursion: MaterializeType recurses into every element of a
# TupleType return signature (pass source: MaterializeType TupleType branch,
# materialize_tensor_strides_pass.cpp:90-101 — "recursively into TupleType").
# ============================================================================


def test_tuple_return_type_materialized():
    # A function whose single return is a Tuple of two empty-stride DN tensors.
    # Both elements must be materialized to their packed DN canonical stride:
    #   [4, 8]    -> [1, 4]
    #   [2, 4, 8] -> [32, 1, 4]
    # (DN formula, doc 27-materialize_tensor_strides.md "Stride Formulas".)
    def build(stride_2d, stride_3d):
        x = ir.Var("x", _dn_tensor([4, 8], stride_2d), _SPAN)
        y = ir.Var("y", _dn_tensor([2, 4, 8], stride_3d), _SPAN)
        ret_tuple = ir.TupleType([_dn_tensor([4, 8], stride_2d), _dn_tensor([2, 4, 8], stride_3d)])
        body = ir.ReturnStmt([x, y], _SPAN)
        func = ir.Function("f", [x, y], [ret_tuple], body, _SPAN)
        return ir.Program([func], "p", _SPAN)

    Before = build([], [])
    Expected = build([1, 4], [32, 1, 4])

    After = _materialize(Before)
    ir.assert_structural_equal(After, Expected)


# ============================================================================
# IterArg recursion: VisitExpr_(IterArgPtr) materializes the IterArg's own
# carried type, and recurses into its init_value (pass source:
# materialize_tensor_strides_pass.cpp:133-149). A loop-carried DN tensor with
# empty stride must come out packed, and its init (a reference to the
# already-materialized param) must follow.
# ============================================================================


def test_iter_arg_type_and_init_materialized():
    def build(stride):
        init = ir.Var("init", _dn_tensor([4, 8], stride), _SPAN)
        acc = ir.IterArg("acc", _dn_tensor([4, 8], stride), init, _SPAN)
        i = ir.Var("i", ir.ScalarType(DataType.INDEX), _SPAN)
        ret = ir.Var("r", _dn_tensor([4, 8], stride), _SPAN)
        # A ForStmt carrying iter_args must end its body with a YieldStmt that
        # yields the loop-carried values (SSA invariant enforced by SSAVerify).
        # A single-child body is the YieldStmt directly (NormalizedStmtStructure
        # rejects a SeqStmts wrapping a single child).
        loop_body = ir.YieldStmt([acc], _SPAN)
        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INDEX, _SPAN),
            ir.ConstInt(4, DataType.INDEX, _SPAN),
            ir.ConstInt(1, DataType.INDEX, _SPAN),
            [acc],
            loop_body,
            [ret],
            _SPAN,
        )
        body = ir.SeqStmts([for_stmt, ir.ReturnStmt([ret], _SPAN)], _SPAN)
        func = ir.Function("f", [init], [_dn_tensor([4, 8], stride)], body, _SPAN)
        return ir.Program([func], "p", _SPAN)

    Before = build([])
    Expected = build([1, 4])

    After = _materialize(Before)
    ir.assert_structural_equal(After, Expected)


# ============================================================================
# Submit return-type materialization (FOCUS — suspected bug).
#
# The pass overrides VisitExpr_(CallPtr) to route Call return types through
# MaterializeType, but provides NO VisitExpr_(SubmitPtr) override. Submit
# therefore falls to the base IRMutator::VisitExpr_(SubmitPtr), which only
# runs RemapTypeViaVisitor (remaps embedded *expressions*, NOT empty-stride
# views) on the return type. Per the pass docstring ("Walks every TensorType
# reachable from a Program ... recursively into TupleType ... after this pass
# runs, every TensorType that carries a TensorView has explicit stride") and
# .claude/rules/pass-submit-awareness.md rule 4 (Submit return types must be
# accounted for), the Submit node's own return TupleType element MUST be
# materialized to [1, 4] just like the equivalent Call/param/return-type slots
# (which this same program DOES materialize). The Submit node's type_ slot is
# left with empty stride instead.
# ============================================================================


def test_submit_return_type_materialized():
    def build(stride):
        kx = ir.Var("x", _dn_tensor([4, 8], stride), _SPAN)
        kernel = ir.Function("kernel", [kx], [_dn_tensor([4, 8], stride)], ir.ReturnStmt([kx], _SPAN), _SPAN)
        kgv = ir.GlobalVar("kernel")

        a = ir.Var("a", _dn_tensor([4, 8], stride), _SPAN)
        submit_ret = ir.TupleType([_dn_tensor([4, 8], stride), ir.ScalarType(DataType.TASK_ID)])
        res = ir.Var("res", submit_ret, _SPAN)
        submit = ir.Submit(kgv, [a], [], submit_ret, _SPAN)
        body = ir.SeqStmts([ir.AssignStmt(res, submit, _SPAN), ir.ReturnStmt([res], _SPAN)], _SPAN)
        caller = ir.Function("caller", [a], [submit_ret], body, _SPAN)
        return ir.Program([kernel, caller], "p", _SPAN)

    Before = build([])
    # Every reachable TensorType — kernel params/return, caller param/return,
    # AND the Submit node's own tuple-return element — should be DN-packed [1, 4].
    Expected = build([1, 4])

    After = _materialize(Before)
    ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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

import re
from collections.abc import Sequence
from typing import cast

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import DataType, ir
from pypto.ir import IRBuilder
from pypto.pypto_core import InternalError
from pypto.pypto_core import passes as _passes

_SPAN = ir.Span.unknown()

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _materialize(program: ir.Program) -> ir.Program:
    return _passes.materialize_tensor_strides()(program)


def _materialize_basic(program: ir.Program) -> ir.Program:
    ctx = _passes.PassContext(
        [_passes.VerificationInstrument(_passes.VerificationMode.BEFORE_AND_AFTER)],
        _passes.VerificationLevel.BASIC,
    )
    with ctx:
        return _passes.materialize_tensor_strides()(program)


def _verify_strict(program: ir.Program):
    """Run TensorViewCanonical in strict mode — empty stride is rejected."""
    return _passes.verify_tensor_view_canonical(program, require_materialized=True)


def _dims(shape: Sequence[int]) -> list[ir.ConstInt]:
    return [ir.ConstInt(s, DataType.INDEX, _SPAN) for s in shape]


def _values_of(exprs: Sequence[ir.Expr]) -> list[int]:
    return [cast(ir.ConstInt, expr).value for expr in exprs]


def _dn_tensor(shape: Sequence[int], stride: Sequence[int]) -> ir.TensorType:
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


def test_distributed_tensor_param_preserves_memref_and_pad_metadata():
    """Materializing a distributed tensor view keeps non-stride metadata.

    The ``MemRef`` binding and the ``pad`` value ride through untouched; only
    the empty stride slot is filled with the packed DN canonical stride.
    """

    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pld.DistributedTensor[
                [4, 8],
                pl.FP32,
                pl.TensorView(stride=[], layout=pl.TensorLayout.DN, pad=pl.PadValue.zero),
                pl.MemRef("base", 0, 128),
            ],
        ):
            return  # noqa: PLR1711 - DSL spelling of an empty body

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            x: pld.DistributedTensor[
                [4, 8],
                pl.FP32,
                pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN, pad=pl.PadValue.zero),
                pl.MemRef("base", 0, 128),
            ],
        ):
            return  # noqa: PLR1711 - DSL spelling of an empty body

    ir.assert_structural_equal(_materialize(Before), Expected)


def test_distributed_tensor_view_preserves_window_buffer_metadata():
    """A materialized distributed ``tensor.view`` keeps its WindowBuffer binding.

    Hand-built rather than DSL-authored, and necessarily so: the DSL parser
    fills a ``pl.tensor.view`` result's stride at parse time, so a DSL
    ``Before`` would arrive here already materialized and the comparison would
    be vacuous. ``ir.op.tensor.view`` leaves the stride empty, which is the
    input shape this pass exists to fix.
    """
    base = ir.Var("buf", ir.PtrType(), _SPAN)
    window = ir.WindowBuffer(base, ir.ConstInt(128, DataType.INT64, _SPAN), span=_SPAN)
    src_type = ir.DistributedTensorType(_dims([4, 8]), DataType.FP32, window)

    ib = IRBuilder()
    with ib.program("main") as prog:
        with ib.function("f") as f:
            x = f.param("x", src_type)
            viewed = ib.let("viewed", ir.op.tensor.view(x, layout=ir.TensorLayout.DN))
            f.return_type(viewed.type)
            ib.return_stmt(viewed)
        prog.add_function(f.get_result())
    After = _materialize_basic(prog.get_result())
    func = After.get_function("f")
    assert func is not None
    body = cast(ir.SeqStmts, func.body)
    viewed_stmt = cast(ir.AssignStmt, body.stmts[0])

    viewed_call = cast(ir.Call, viewed_stmt.value)
    assert len(func.return_types) == 1
    for type_ in (func.return_types[0], viewed_stmt.var.type, viewed_call.type):
        assert isinstance(type_, ir.DistributedTensorType)
        assert type_.window_buffer is window
        assert type_.tensor_view is not None
        assert _values_of(type_.tensor_view.stride) == [1, 8]


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
# An *unblocked* NZ TensorType is rejected by the pass itself
# ============================================================================
#
# NZ is legal on a TensorType, but only in the blocked rank-(r+2) form that
# ``BlockNzTensorViews`` produces — ``[..., C/c0, R/16, 16, c0]``, whose plain
# row-major strides are exactly pto-isa's ``BaseShape2D<..., Layout::NZ>``.
# A logical-shaped NZ view reaching this pass means BlockNzTensorViews did not
# run or missed a slot, so the rejection is a pass invariant (INTERNAL_CHECK,
# surfacing as InternalError), not a user error — the user-facing alignment
# diagnostics live in BlockNzShape, far earlier in the pipeline.
#
# The pass still rejects directly instead of leaving the slot for the paired
# TensorViewCanonical verifier: delegating meant the malformed slot survived
# silently whenever verification was disabled (``PYPTO_VERIFY_LEVEL=none``),
# even though the pass declares TensorViewCanonical as produced.


def test_unblocked_nz_on_tensor_rejected_by_pass():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[8, 16], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        ):
            pl.const(0, pl.INT64)

    with pytest.raises(InternalError, match="not blocked") as excinfo:
        _materialize(Before)
    # The pass threads the carrying node's Span into INTERNAL_CHECK_SPAN so the
    # message points at the offending annotation. Assert the location is
    # present, not just the text — otherwise dropping the span would go
    # unnoticed.
    assert re.search(r"\[[^]\s]+:\d+:\d+\]", str(excinfo.value)), str(excinfo.value)


def test_unblocked_nz_on_distributed_tensor_rejected_by_pass():
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pld.DistributedTensor[[8, 16], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        ):
            pl.const(0, pl.INT64)

    with pytest.raises(InternalError, match="not blocked"):
        _materialize(Before)


def test_unblocked_nz_with_explicit_stride_rejected_by_pass():
    # An explicit stride does not make an unblocked NZ view valid. The check
    # runs before the "already explicit, nothing to materialize" short-circuit,
    # so this slot cannot slip through the pass claiming TensorViewCanonical.
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[8, 16], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.NZ)],
        ):
            pl.const(0, pl.INT64)

    with pytest.raises(InternalError, match="not blocked"):
        _materialize(Before)


def test_unblocked_nz_rejected_under_verification_disabled():
    # Regression guard for the delegation bug: with no VerificationInstrument
    # installed, nothing but the pass itself can reject the invalid slot.
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[8, 16], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        ):
            pl.const(0, pl.INT64)

    with _passes.PassContext([], _passes.VerificationLevel.NONE):
        with pytest.raises(InternalError, match="not blocked"):
            _materialize(Before)


def test_blocked_nz_gets_row_major_strides():
    # The positive counterpart: once the shape is blocked, NZ is an ordinary
    # row-major family member. For [256, 512] INT8 (c0 = 32) the blocked shape
    # is [16, 16, 16, 32] and pto-isa's BaseShape2D<int8_t, 256, 512, NZ> is
    # Stride<256*32, 16*32, 32, 1> = [8192, 512, 32, 1].
    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[16, 16, 16, 32], pl.INT8, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        ):
            pl.const(0, pl.INT64)

    After = _materialize(Before)
    param_type = list(After.functions.values())[0].params[0].type
    assert isinstance(param_type, ir.TensorType)
    view = param_type.tensor_view
    assert view is not None
    assert view.layout == ir.TensorLayout.NZ
    assert _values_of(view.stride) == [8192, 512, 32, 1]


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
    """Both elements of a Tuple return signature are DN-packed.

    ``[4, 8] -> [1, 4]`` and ``[2, 4, 8] -> [32, 1, 4]`` per the DN formula
    (doc 31-materialize_tensor_strides.md "Stride Formulas").
    """

    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
            y: pl.Tensor[[2, 4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ) -> tuple[
            pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
            pl.Tensor[[2, 4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ]:
            return x, y

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)],
            y: pl.Tensor[[2, 4, 8], pl.FP32, pl.TensorView(stride=[32, 1, 4], layout=pl.TensorLayout.DN)],
        ) -> tuple[
            pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)],
            pl.Tensor[[2, 4, 8], pl.FP32, pl.TensorView(stride=[32, 1, 4], layout=pl.TensorLayout.DN)],
        ]:
            return x, y

    ir.assert_structural_equal(_materialize(Before), Expected)


# ============================================================================
# IterArg recursion: VisitExpr_(IterArgPtr) materializes the IterArg's own
# carried type, and recurses into its init_value (pass source:
# materialize_tensor_strides_pass.cpp:133-149). A loop-carried DN tensor with
# empty stride must come out packed, and its init (a reference to the
# already-materialized param) must follow.
# ============================================================================


def test_iter_arg_type_and_init_materialized():
    """A loop-carried DN tensor comes out packed, and so does its init value."""

    @pl.program
    class Before:
        @pl.function
        def f(
            self,
            init: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ) -> pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)]:
            for _i, (acc,) in pl.range(0, 4, init_values=(init,)):
                r = pl.yield_(acc)
            return r

    @pl.program
    class Expected:
        @pl.function
        def f(
            self,
            init: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)],
        ) -> pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)]:
            for _i, (acc,) in pl.range(0, 4, init_values=(init,)):
                r = pl.yield_(acc)
            return r

    ir.assert_structural_equal(_materialize(Before), Expected)


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
    """Every reachable TensorType — kernel params/return, caller param, AND the
    Submit node's own tuple-return element — comes out DN-packed ``[1, 4]``."""

    @pl.program
    class Before:
        @pl.function
        def kernel(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ) -> pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)]:
            return x

        @pl.function
        def caller(
            self,
            a: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
        ):
            with pl.manual_scope():
                res, tid = pl.submit(self.kernel, a)
            return res, tid

    @pl.program
    class Expected:
        @pl.function
        def kernel(
            self,
            x: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)],
        ) -> pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)]:
            return x

        @pl.function
        def caller(
            self,
            a: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)],
        ):
            with pl.manual_scope():
                res, tid = pl.submit(self.kernel, a)
            return res, tid

    ir.assert_structural_equal(_materialize(Before), Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

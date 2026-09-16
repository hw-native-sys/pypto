# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Call-boundary layout checking in the TypeChecked verifier.

A layout annotation is a claim about byte order in global memory, so a call
argument and the callee parameter it binds must make the *same* claim. Nothing
else in the pipeline compares them: `InlineFunctions` substitutes the parameter
away, and a non-inline callee keeps its own claim while only the caller's base
pointer crosses the boundary. Either way the disagreement is silent, so it has
to be caught before the first pass runs.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes

_SPAN = ir.Span.unknown()
_LAYOUT_MISMATCH = 113


def _verify(program: ir.Program) -> list[passes.Diagnostic]:
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.TypeChecked)
    return passes.PropertyVerifierRegistry.verify(properties, program)


def _tensor_type(shape: list[int], layout: ir.TensorLayout | None = None) -> ir.TensorType:
    view = None if layout is None else ir.TensorView([], layout)
    return ir.TensorType(shape, DataType.BF16, None, view)


def _callee(name: str, param_type: ir.Type) -> ir.Function:
    """A minimal callee: one tensor parameter, returned unchanged."""
    param = ir.Var("b", param_type, _SPAN)
    body = ir.SeqStmts([ir.ReturnStmt([param], _SPAN)], _SPAN)
    return ir.Function(name, [param], [param_type], body, _SPAN)


def _caller(callee: ir.Function, arg_type: ir.Type, *, name: str = "main") -> ir.Function:
    """A caller binding its own parameter straight into a call to `callee`."""
    arg = ir.Var("a", arg_type, _SPAN)
    call = ir.Call(ir.GlobalVar(callee.name), [arg], callee.return_types[0], _SPAN)
    out = ir.Var("out", callee.return_types[0], _SPAN)
    body = ir.SeqStmts([ir.AssignStmt(out, call, _SPAN), ir.ReturnStmt([out], _SPAN)], _SPAN)
    return ir.Function(name, [arg], [callee.return_types[0]], body, _SPAN)


def _program(callee: ir.Function, arg_type: ir.Type) -> ir.Program:
    return ir.Program([callee, _caller(callee, arg_type)], "test", _SPAN)


def test_rejects_nd_argument_for_an_nz_parameter():
    """The reported bug: NZ-packed bytes addressed row-major, with nothing to warn."""
    callee = _callee("nz_helper", _tensor_type([512, 256], ir.TensorLayout.NZ))
    diagnostics = _verify(_program(callee, _tensor_type([512, 256])))

    assert len(diagnostics) == 1
    assert diagnostics[0].error_code == _LAYOUT_MISMATCH
    assert "argument 0 of call to 'nz_helper'" in diagnostics[0].message
    assert "parameter 'b' is declared NZ but the argument is ND" in diagnostics[0].message


def test_rejects_nd_argument_for_an_mx_parameter():
    """Not NZ-specific: every layout a parameter *can* declare is checked."""
    callee = _callee("mx_helper", _tensor_type([16, 32], ir.TensorLayout.MX_B_NN))
    diagnostics = _verify(_program(callee, _tensor_type([16, 32])))

    assert len(diagnostics) == 1
    assert diagnostics[0].error_code == _LAYOUT_MISMATCH
    assert "declared MX_B_NN but the argument is ND" in diagnostics[0].message


def test_accepts_a_dn_argument_for_an_nd_parameter():
    """DN is the carve-out: a parameter cannot declare it, so ND is not a rival claim.

    `pl.Tensor[..., pl.DN]` raises ParserTypeError -- DN is *derived* at the use
    site by `pl.transpose`, never annotated. An ND parameter is simply the only
    thing the author can write, and `OptimizeOrchTensors` materialises the
    strides that make the pattern lower correctly.
    """
    callee = _callee("nd_helper", _tensor_type([16, 32]))
    assert _verify(_program(callee, _tensor_type([16, 32], ir.TensorLayout.DN))) == []


def test_accepts_an_nd_argument_for_a_dn_parameter():
    """The carve-out is symmetric, for DN parameters that only printed IR can carry."""
    callee = _callee("dn_helper", _tensor_type([16, 32], ir.TensorLayout.DN))
    assert _verify(_program(callee, _tensor_type([16, 32]))) == []


def test_accepts_a_transposed_view_passed_to_an_nd_parameter():
    """The DSL form the carve-out exists for: `pl.transpose` produces a DN view."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def add_kernel(
            self,
            a: pl.Tensor[[32, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
        ) -> pl.Tensor[[32, 16], pl.FP32]:
            tile = pl.load(a, [0, 0], [32, 16], target_memory=pl.MemorySpace.Vec)
            return pl.store(pl.add(tile, tile), [0, 0], c)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            a: pl.Tensor[[16, 32], pl.FP32],
            c: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
        ) -> pl.Tensor[[32, 16], pl.FP32]:
            a_t: pl.Tensor[[32, 16], pl.FP32] = pl.transpose(a, axis1=0, axis2=1)
            return self.add_kernel(a_t, c)

    assert _verify(Prog) == []


def test_rejects_nz_argument_for_an_nd_parameter():
    """The mismatch is symmetric -- an NZ argument is just as wrong the other way."""
    callee = _callee("nd_helper", _tensor_type([512, 256]))
    diagnostics = _verify(_program(callee, _tensor_type([512, 256], ir.TensorLayout.NZ)))

    assert len(diagnostics) == 1
    assert "declared ND but the argument is NZ" in diagnostics[0].message


def test_accepts_matching_layouts():
    callee = _callee("nz_helper", _tensor_type([512, 256], ir.TensorLayout.NZ))
    assert _verify(_program(callee, _tensor_type([512, 256], ir.TensorLayout.NZ))) == []


def test_absent_view_and_explicit_nd_view_are_the_same_claim():
    """`TensorType` canonicalizes an ND-only view away, so the two spellings must agree."""
    explicit_nd = _tensor_type([16, 32], ir.TensorLayout.ND)
    assert explicit_nd.tensor_view is None

    callee = _callee("nd_helper", explicit_nd)
    assert _verify(_program(callee, _tensor_type([16, 32]))) == []


def test_ignores_a_callee_outside_the_program():
    """An opaque / external callee has no signature here; the check stays silent."""
    callee = _callee("nz_helper", _tensor_type([512, 256], ir.TensorLayout.NZ))
    caller = _caller(callee, _tensor_type([512, 256]))
    assert _verify(ir.Program([caller], "test", _SPAN)) == []


def test_ignores_non_tensor_arguments():
    """A scalar parameter carries no layout, so there is nothing to compare."""
    scalar = ir.ScalarType(DataType.INDEX)
    param = ir.Var("n", scalar, _SPAN)
    callee = ir.Function(
        "scalar_helper", [param], [scalar], ir.SeqStmts([ir.ReturnStmt([param], _SPAN)], _SPAN), _SPAN
    )
    assert _verify(_program(callee, scalar)) == []


def test_rejects_the_mismatch_through_an_inline_callee():
    """The DSL form that motivated this: the annotation would be erased by pass 01."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Inline)
        def nz_helper(
            self,
            b: pl.Tensor[[512, 256], pl.BF16, pl.TensorView(layout=pl.TensorLayout.NZ)],
        ) -> pl.Tensor[[512, 256], pl.BF16, pl.TensorView(layout=pl.TensorLayout.NZ)]:
            return b

        @pl.function
        def main(self, b: pl.Tensor[[512, 256], pl.BF16]) -> pl.Tensor[[512, 256], pl.BF16]:
            out = self.nz_helper(b)
            return out

    diagnostics = _verify(Prog)

    assert len(diagnostics) == 1
    assert diagnostics[0].error_code == _LAYOUT_MISMATCH
    assert "call to 'nz_helper'" in diagnostics[0].message


def test_rejects_the_mismatch_on_a_submit():
    """`Submit` is a call-like sibling of `Call` and gets the same check."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def nz_kernel(
            self,
            b: pl.Tensor[[512, 256], pl.BF16, pl.TensorView(layout=pl.TensorLayout.NZ)],
        ) -> pl.Tensor[[512, 256], pl.BF16, pl.TensorView(layout=pl.TensorLayout.NZ)]:
            return b

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, b: pl.Tensor[[512, 256], pl.BF16]) -> pl.Tensor[[512, 256], pl.BF16]:
            with pl.scope(mode=pl.ScopeMode.MANUAL):
                out, _tid = pl.submit(self.nz_kernel, b)
            return out

    diagnostics = _verify(Prog)

    assert len(diagnostics) == 1
    assert diagnostics[0].error_code == _LAYOUT_MISMATCH
    assert "call to 'nz_kernel'" in diagnostics[0].message


def test_a_leading_axis_slice_keeps_its_layout_across_the_call():
    """The shape a sharded driver needs: `b[r]` of an NZ weight stays NZ, so both ends agree.

    This is a check on the *call boundary* only. `BlockNzTensorViews` separately
    refuses to lower a slice of an NZ tensor today, so the full pipeline still
    rejects this program -- with its own diagnostic naming `tensor.slice`, not
    with silently mis-addressed bytes.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Inline)
        def nz_helper(
            self,
            b: pl.Tensor[[512, 256], pl.BF16, pl.TensorView(layout=pl.TensorLayout.NZ)],
        ) -> pl.Tensor[[512, 256], pl.BF16, pl.TensorView(layout=pl.TensorLayout.NZ)]:
            return b

        @pl.function
        def main(
            self,
            b: pl.Tensor[[2, 512, 256], pl.BF16, pl.TensorView(layout=pl.TensorLayout.NZ)],
        ) -> pl.Tensor[[512, 256], pl.BF16, pl.TensorView(layout=pl.TensorLayout.NZ)]:
            shard = b[1]
            out = self.nz_helper(shard)
            return out

    assert _verify(Prog) == []


def test_the_same_slice_out_of_an_nd_driver_is_rejected():
    """The L3 shape that silently mis-addresses: ND driver, NZ per-shard callee."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Inline)
        def nz_helper(
            self,
            b: pl.Tensor[[512, 256], pl.BF16, pl.TensorView(layout=pl.TensorLayout.NZ)],
        ) -> pl.Tensor[[512, 256], pl.BF16, pl.TensorView(layout=pl.TensorLayout.NZ)]:
            return b

        @pl.function
        def main(self, b: pl.Tensor[[2, 512, 256], pl.BF16]) -> pl.Tensor[[512, 256], pl.BF16]:
            shard = b[1]
            out = self.nz_helper(shard)
            return out

    diagnostics = _verify(Prog)

    assert len(diagnostics) == 1
    assert diagnostics[0].error_code == _LAYOUT_MISMATCH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

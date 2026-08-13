# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the MaterializeValidShapeSymbols pass."""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors
# from type-checking annotations that reference module-level DynVar names.
# pyright: reportUndefinedVariable=false, reportInvalidTypeForm=false

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics.exceptions import ParserTypeError
from pypto.pypto_core import passes

Q = pl.dynamic("Q")
BLK = pl.dynamic("BLK")
VALID = pl.dynamic("VALID")
# Annotations are evaluated in the enclosing scope, so a symbol a signature names
# has to be resolvable there — these two back the shadowing tests below.
SHADOW_SCALE = pl.dynamic("SHADOW_SCALE")
LATE_VALID = pl.dynamic("LATE_VALID")


def _kernel_calls(func) -> list[ir.Call]:
    """Every Call to a Function (not an operator) in ``func``'s body."""
    calls: list[ir.Call] = []

    class _CallCollector(ir.IRVisitor):
        def visit_call(self, call):  # type: ignore[override]
            if isinstance(call.op, ir.GlobalVar):
                calls.append(call)
            super().visit_call(call)

    _CallCollector().visit_stmt(func.body)
    return calls


def _kernel_and_caller():
    """An InCore kernel whose valid_shape names a symbol only the caller knows."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        sij: pl.Tensor[
            [Q, BLK],
            pl.FP32,
            pl.TensorView(valid_shape=[Q, VALID], layout=pl.TensorLayout.ND),
        ],
        out: pl.Out[pl.Tensor[[Q, BLK], pl.FP32]],
    ) -> pl.Tensor[[Q, BLK], pl.FP32]:
        t = pl.load(sij, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec)
        return pl.store(t, [0, 0], out)

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            sij: pl.Tensor[[Q, BLK], pl.FP32],
            valid_len: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[Q, BLK], pl.FP32]],
        ) -> pl.Tensor[[Q, BLK], pl.FP32]:
            narrowed = pl.slice(sij, [16, 128], [0, 0], valid_shape=[16, valid_len])
            return kernel(narrowed, out)

    return Prog


def test_symbol_becomes_leading_scalar_param_and_call_arg():
    """The symbol is added as a parameter and fed the caller's actual extent."""
    prog = _kernel_and_caller()
    after = passes.materialize_valid_shape_symbols()(prog)

    kernel = after.get_function("kernel")
    assert kernel is not None
    # Leading, because the tensor parameter's annotation names it and the text
    # form declares parameters left to right.
    assert kernel.params[0].name_hint == "VALID"
    assert isinstance(kernel.params[0].type, ir.ScalarType)
    assert kernel.params[0].type.dtype == pl.INDEX
    # The tensor parameter's valid_shape now reads that very parameter.
    sij_type = kernel.params[1].type
    assert isinstance(sij_type, ir.TensorType)
    assert sij_type.tensor_view is not None
    assert sij_type.tensor_view.valid_shape[1].same_as(kernel.params[0])

    # The caller passes the actual extent, ahead of the original arguments.
    main = after.get_function("main")
    assert main is not None
    calls = _kernel_calls(main)
    assert len(calls) == 1, f"expected one kernel call, got {len(calls)}"
    args = list(calls[0].args)
    assert len(args) == 3, f"expected valid_len + 2 original args, got {len(args)}"
    valid_len_param = next(p for p in main.params if p.name_hint == "valid_len")
    assert args[0].same_as(valid_len_param), "the extent must be the caller's scalar, first"


def test_pass_is_idempotent():
    """Re-running finds nothing left to bind."""
    prog = _kernel_and_caller()
    once = passes.materialize_valid_shape_symbols()(prog)
    twice = passes.materialize_valid_shape_symbols()(once)
    ir.assert_structural_equal(twice, once)


def test_kernel_without_unbindable_symbol_is_untouched():
    """A valid_shape built from a scalar parameter already binds; leave it alone."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            sij: pl.Tensor[[Q, BLK], pl.FP32],
            valid_len: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[Q, BLK], pl.FP32]],
        ) -> pl.Tensor[[Q, BLK], pl.FP32]:
            t = pl.load(sij, [0, 0], [16, 128], valid_shape=[16, valid_len])
            return pl.store(t, [0, 0], out)

    after = passes.materialize_valid_shape_symbols()(Prog)
    ir.assert_structural_equal(after, Prog)


def test_physical_dim_symbol_is_not_materialized():
    """A symbol that is also a physical dimension is recoverable at run time."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            sij: pl.Tensor[
                [Q, BLK],
                pl.FP32,
                pl.TensorView(valid_shape=[Q, BLK], layout=pl.TensorLayout.ND),
            ],
            out: pl.Out[pl.Tensor[[Q, BLK], pl.FP32]],
        ) -> pl.Tensor[[Q, BLK], pl.FP32]:
            t = pl.load(sij, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec)
            return pl.store(t, [0, 0], out)

    after = passes.materialize_valid_shape_symbols()(Prog)
    ir.assert_structural_equal(after, Prog)


def test_non_index_scalar_param_does_not_capture_a_dynamic_symbol():
    """Only INDEX scalars may absorb a same-named ``pl.dynamic()`` symbol.

    A ``DynVar`` is an INDEX-valued dimension, so rebinding one to an FP32
    parameter would put an FP32 Var into a shape expression.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            SHADOW_SCALE: pl.Scalar[pl.FP32],
            sij: pl.Tensor[[Q, SHADOW_SCALE], pl.FP32],
            out: pl.Out[pl.Tensor[[Q, BLK], pl.FP32]],
        ) -> pl.Tensor[[Q, BLK], pl.FP32]:
            t = pl.load(sij, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec)
            return pl.store(t, [0, 0], out)

    kernel = Prog.get_function("kernel")
    assert kernel is not None
    # The FP32 parameter and the dimension symbol stay distinct Vars.
    sij_type = kernel.params[1].type
    assert isinstance(sij_type, ir.TensorType)
    assert not sij_type.shape[1].same_as(kernel.params[0])


def test_scalar_param_after_the_annotation_that_uses_it_is_rejected():
    """The shadowing parameter must precede the annotation that names it.

    Annotations resolve in declaration order, so a later parameter cannot rebind
    a symbol an earlier annotation already read; silently allowing it would
    materialize a second, redundant extent argument.
    """
    with pytest.raises(ParserTypeError, match="LATE_VALID") as exc_info:

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                sij: pl.Tensor[
                    [Q, BLK],
                    pl.FP32,
                    pl.TensorView(valid_shape=[Q, LATE_VALID], layout=pl.TensorLayout.ND),
                ],
                LATE_VALID: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[Q, BLK], pl.FP32]],
            ) -> pl.Tensor[[Q, BLK], pl.FP32]:
                t = pl.load(sij, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec)
                return pl.store(t, [0, 0], out)

    assert "earlier parameter" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

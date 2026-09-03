# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the NoScalarKernelReturn property verifier (#631).

The runtime has no scalar output channel: ``Arg::add_scalar`` passes a scalar
*in* by value and ``TaskOutputTensors`` returns only tensors. A device function
declaring a Scalar return therefore has no carrier for that value —
orchestration codegen used to emit an undefined identifier for it, and later a
silently wrong ``= 0``.

The check is on the declaration because ``FunctionType.InCore`` / ``AIC`` /
``AIV`` / ``Group`` / ``Spmd`` *means* "a dispatchable task": a helper meant to
run inside a kernel is written ``FunctionType.Inline`` and spliced away before
codegen. ``Scalar[TASK_ID]`` is rejected too: the scheduler handle is appended
to the *call's* tuple type and bound at the call site, so a declaration that
carries one has no carrier either — and makes codegen read an ordinary call to
it as a task launch.

A ``-> pl.Tuple[...]`` annotation is a *single* ``return_types_`` entry holding
a ``TupleType``, so the check descends into tuple elements: a scalar hidden in
one has no more of a carrier than a bare scalar return.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes

_SPAN = ir.Span.unknown()


def _verify(program: ir.Program) -> list:
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.NoScalarKernelReturn)
    return passes.PropertyVerifierRegistry.verify(props, program)


def _errors(program: ir.Program) -> list[str]:
    return [d.message for d in _verify(program) if d.severity == passes.DiagnosticSeverity.Error]


def _returning(name: str, return_type: ir.Type, func_type: ir.FunctionType) -> ir.Function:
    """A function of @p func_type whose sole return is a constant of that type."""
    value = ir.ConstInt(0, ir.DataType.INDEX, _SPAN)
    return ir.Function(name, [], [return_type], ir.ReturnStmt([value], _SPAN), _SPAN, func_type)


def test_tensor_returning_kernel_passes():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            x: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
            return pl.store(t, [0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            return self.kernel(x, out)

    assert _errors(P) == []


@pytest.mark.parametrize(
    "func_type",
    [
        ir.FunctionType.InCore,
        ir.FunctionType.AIC,
        ir.FunctionType.AIV,
        ir.FunctionType.Group,
        ir.FunctionType.Spmd,
    ],
)
def test_scalar_returning_device_function_is_rejected(func_type):
    """Every dispatchable function type is covered, not just InCore."""
    kernel = _returning("kernel", ir.ScalarType(ir.DataType.INDEX), func_type)

    errors = _errors(ir.Program([kernel], "ScalarReturn", _SPAN))
    assert len(errors) == 1
    assert "cannot return a scalar" in errors[0]
    assert "kernel" in errors[0]


def test_task_id_return_is_rejected():
    """A TaskId rides the *call*, never the callee's declaration.

    ``pl.submit`` and the outliner append the trailing ``Scalar[TASK_ID]`` to
    the Submit's tuple type and bind it at the call site; the callee's
    ``return_types_`` stay untouched. A declaration that carries one is not
    merely uncarried, it is misread: orchestration codegen's ``IsSubmitCall``
    keys on "TupleType whose last element is ``Scalar[TASK_ID]``".
    """
    kernel = _returning("kernel", ir.ScalarType(ir.DataType.TASK_ID), ir.FunctionType.AIV)

    errors = _errors(ir.Program([kernel], "TaskIdReturn", _SPAN))
    assert len(errors) == 1
    assert "Scalar[TASK_ID]" in errors[0]
    assert "pl.submit" in errors[0], f"the TaskId diagnostic must name the call-site fix: {errors[0]}"


def test_task_id_tail_of_a_tuple_return_is_rejected():
    """The exact shape ``IsSubmitCall`` keys on: an ordinary multi-output kernel
    whose declared tuple ends in a TaskId would be lowered as a task launch.
    """
    tuple_type = ir.TupleType([ir.TensorType([64], ir.DataType.FP32), ir.ScalarType(ir.DataType.TASK_ID)])
    kernel = _returning("kernel", tuple_type, ir.FunctionType.AIV)

    errors = _errors(ir.Program([kernel], "TupleTaskIdReturn", _SPAN))
    assert len(errors) == 1
    assert "#0 element #1" in errors[0]


def test_scalar_nested_in_a_tuple_return_is_rejected():
    """``-> pl.Tuple[Tensor, Scalar]`` is ONE return type holding a TupleType.

    Inspecting only the top level lets the scalar through, and the launch then
    trips orchestration codegen's internal check instead of this user-facing
    rejection.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            x: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Scalar[pl.INDEX]]:
            t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
            n: pl.Scalar[pl.INDEX] = pl.const(7, pl.INDEX)
            return pl.store(t, [0], out), n

    errors = _errors(P)
    assert len(errors) == 1, errors
    assert "cannot return a scalar" in errors[0]
    # The diagnostic names the offending element, not just the return slot.
    assert "#0 element #1" in errors[0], errors[0]


@pytest.mark.parametrize(
    "func_type",
    [ir.FunctionType.Orchestration, ir.FunctionType.Graph, ir.FunctionType.Inline],
)
def test_non_device_function_may_return_a_scalar(func_type):
    """Only a dispatched task lacks a carrier.

    ``Inline`` matters in its own right: it is how a device-side scalar helper is
    written, and ``InlineFunctions`` splices it long before codegen.
    """
    callee = _returning("callee", ir.ScalarType(ir.DataType.INDEX), func_type)

    assert _errors(ir.Program([callee], "NonDeviceReturn", _SPAN)) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Cross-function call return-type recovery on print -> parse.

A ``@pl.function`` callee that declares no ``-> <type>`` annotation but does
``return <value>`` (e.g. an ``InCore`` kernel returning its ``pl.Out`` tensor
param) has an empty ``return_types``. A plain cross-function call
``r = self.kernel(...)`` must still recover the callee's effective result type
(derived from the ``return`` statement) instead of ``UnknownType`` — otherwise
the printer emits the assignment target's type as an LHS annotation, the parser
upgrades the reparsed call to that concrete type, and the print -> parse
round-trip diverges at ``<fn>.body[i].value.type``.

The fix derives the call's return type from the callee body when the callee
declares no annotation, so both the original build and the reparse agree.
"""

import pypto.language as pl
import pytest
from pypto import ir


def _user_call_types(program: ir.Program, callee_name: str) -> list[ir.Type]:
    """Collect the value types of every plain Call to ``callee_name``."""
    types: list[ir.Type] = []

    class _Collector(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == callee_name:
                types.append(op.type)
            super().visit_call(op)

    _Collector().visit_program(program)
    return types


def _assert_roundtrips(program: ir.Program) -> None:
    ir.assert_structural_equal(program, pl.parse_program(ir.python_print(program)))


def test_annotationless_callee_reassign_existing_var_roundtrips():
    """``c = self.kernel(...)`` reassigning an existing typed var round-trips.

    This is the failing case: the printer emits ``c: pl.Tensor[...] = ...`` from
    the target var's type, so without return-type recovery the original call
    (``UnknownType``) and the reparsed call (``TensorType``) diverge.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ):
            return c

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ):
            c = self.kernel(a, c)
            return c

    # The original call recovers the concrete callee return type, not UnknownType.
    (call_type,) = _user_call_types(Prog, "kernel")
    assert isinstance(call_type, ir.TensorType)
    assert not isinstance(call_type, ir.UnknownType)

    _assert_roundtrips(Prog)


def test_annotationless_callee_fresh_var_binding_roundtrips():
    """``out = self.kernel(...)`` binding a fresh var round-trips and is typed."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ):
            return c

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ):
            out = self.kernel(a, c)
            return out

    (call_type,) = _user_call_types(Prog, "kernel")
    assert isinstance(call_type, ir.TensorType)
    _assert_roundtrips(Prog)


def test_annotationless_callee_dynamic_shape_is_substituted():
    """A derived return type referencing a callee shape var is deduced per call.

    The kernel's ``return c`` type references the callee param's dynamic dim
    ``N``; ``deduce_call_return_type`` substitutes it with the caller's concrete
    ``32`` so the recovered call type is fully static.
    """
    N = pl.dynamic("N")

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[N, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[N, 16], pl.FP32]],
        ):
            return c

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[32, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
        ):
            c = self.kernel(a, c)
            return c

    (call_type,) = _user_call_types(Prog, "kernel")
    assert isinstance(call_type, ir.TensorType)
    assert [str(d) for d in call_type.shape] == ["32", "16"]
    _assert_roundtrips(Prog)


def test_explicit_return_annotation_still_roundtrips():
    """A callee with an explicit ``-> `` annotation is unaffected (control)."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return c

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ):
            c = self.kernel(a, c)
            return c

    (call_type,) = _user_call_types(Prog, "kernel")
    assert isinstance(call_type, ir.TensorType)
    _assert_roundtrips(Prog)


def test_submit_to_annotationless_callee_return_unchanged():
    """``pl.submit`` to an annotation-less callee keeps its ``Tuple[TASK_ID]``.

    Submit return augmentation is governed by pl.submit conventions, not the
    callee's implicit return, so the recovery must not widen the submit tuple.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ):
            return c

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ):
            with pl.manual_scope():
                _tid = pl.submit(self.kernel, a, c)
            return c

    submit_types: list[ir.Type] = []

    def walk(stmt: ir.Stmt) -> None:
        value = getattr(stmt, "value", None)
        if isinstance(value, ir.Submit):
            submit_types.append(value.type)
        for attr in ("body", "stmts"):
            sub = getattr(stmt, attr, None)
            if sub is None:
                continue
            for child in sub if isinstance(sub, (list, tuple)) else [sub]:
                walk(child)

    orch = Prog.get_function("orch")
    assert orch is not None
    walk(orch.body)
    assert len(submit_types) == 1
    submit_type = submit_types[0]
    # Tuple holds exactly the producer TASK_ID — no widening from the callee.
    assert isinstance(submit_type, ir.TupleType)
    assert len(submit_type.types) == 1
    assert isinstance(submit_type.types[0], ir.ScalarType)

    _assert_roundtrips(Prog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

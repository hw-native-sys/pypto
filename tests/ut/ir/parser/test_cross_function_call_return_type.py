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
import pypto.language.distributed as pld
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


def test_symbolic_valid_shape_and_composite_return_metadata_are_substituted():
    """A DSL call rebuilds a composite valid-shape expression for the actual tensor shape."""
    n = pl.dynamic("N")

    @pl.program
    class Prog:
        @pl.function
        def view(self, arg: pl.Tensor[[n, 64], pl.FP32]):
            return pl.tensor.slice(arg, [n, 64], [0, 0], valid_shape=[n - 1, 64])

        @pl.function
        def main(self, actual: pl.Tensor[[32, 64], pl.FP32]):
            return self.view(actual)

    (deduced,) = _user_call_types(Prog, "view")
    assert isinstance(deduced, ir.TensorType)
    assert [str(dim) for dim in deduced.shape] == ["32", "64"]
    assert deduced.tensor_view is not None
    valid_expr = deduced.tensor_view.valid_shape[0]
    assert isinstance(valid_expr, ir.Sub)
    assert isinstance(valid_expr.left, ir.ConstInt) and valid_expr.left.value == 32
    assert isinstance(valid_expr.right, ir.ConstInt) and valid_expr.right.value == 1
    _assert_roundtrips(Prog)


def test_self_binding_is_refined_by_later_concrete_argument():
    """A shared placeholder does not hide a concrete binding from a later argument."""
    n = pl.dynamic("N")

    @pl.program
    class Prog:
        @pl.function
        def select(self, symbolic: pl.Tensor[[n], pl.FP32], concrete: pl.Tensor[[n], pl.FP32]):
            return concrete

        @pl.function
        def main(self, symbolic: pl.Tensor[[n], pl.FP32], concrete: pl.Tensor[[8], pl.FP32]):
            return self.select(symbolic, concrete)

    (deduced,) = _user_call_types(Prog, "select")
    assert isinstance(deduced, ir.TensorType)
    assert [str(dim) for dim in deduced.shape] == ["8"]
    _assert_roundtrips(Prog)


def test_repeated_composite_binding_is_checked_after_transitive_substitution():
    """A repeated extent can be proved after a later parameter binds its nested variable."""
    factor = pl.dynamic("FACTOR")
    total = pl.dynamic("TOTAL")
    actual_factor = pl.dynamic("ACTUAL_FACTOR")

    @pl.program
    class Prog:
        @pl.function
        def select(
            self,
            first: pl.Tensor[[total], pl.FP32],
            second: pl.Tensor[[total], pl.FP32],
            marker: pl.Tensor[[factor], pl.FP32],
        ):
            return first

        @pl.function
        def main(
            self,
            first: pl.Tensor[[factor * 64], pl.FP32],
            second: pl.Tensor[[actual_factor * 64], pl.FP32],
            marker: pl.Tensor[[actual_factor], pl.FP32],
        ):
            return self.select(first, second, marker)

    (deduced,) = _user_call_types(Prog, "select")
    assert isinstance(deduced, ir.TensorType)
    assert [str(dim) for dim in deduced.shape] == ["ACTUAL_FACTOR * 64"]
    _assert_roundtrips(Prog)


def test_composite_binding_skips_mismatched_nonvariable_operands():
    """A compatible call can bind from a later argument without a partial composite binding."""
    n = pl.dynamic("N")
    m = pl.dynamic("M")

    @pl.program
    class Prog:
        @pl.function
        def select(
            self,
            shifted: pl.Tensor[[n + 1], pl.FP32],
            marker: pl.Tensor[[n], pl.FP32],
        ):
            return marker

        @pl.function
        def main(
            self,
            shifted: pl.Tensor[[m + 2], pl.FP32],
            marker: pl.Tensor[[m + 1], pl.FP32],
        ):
            return self.select(shifted, marker)

    (deduced,) = _user_call_types(Prog, "select")
    assert isinstance(deduced, ir.TensorType)
    assert [str(dim) for dim in deduced.shape] == ["M + 1"]
    _assert_roundtrips(Prog)


def test_recursive_tuple_and_tile_return_metadata_are_substituted():
    """A DSL call recursively substitutes metadata on a Tile nested inside tuples."""
    n = pl.dynamic("N")

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def load_tile(self, arg: pl.Tensor[[n, 64], pl.FP32]):
            tile = pl.load(arg, [0, 0], [n, 64], valid_shapes=[n - 1, 64])
            status = pl.const(0, pl.INT64)
            return status, (tile,)

        @pl.function
        def main(self, actual: pl.Tensor[[32, 64], pl.FP32]):
            return self.load_tile(actual)

    (deduced,) = _user_call_types(Prog, "load_tile")
    assert isinstance(deduced, ir.TupleType)
    assert isinstance(deduced.types[1], ir.TupleType)
    deduced_tile = deduced.types[1].types[0]
    assert isinstance(deduced_tile, ir.TileType)
    assert [str(dim) for dim in deduced_tile.shape] == ["32", "64"]
    assert deduced_tile.tile_view is not None
    valid_expr = deduced_tile.tile_view.valid_shape[0]
    assert isinstance(valid_expr, ir.Sub)
    assert isinstance(valid_expr.left, ir.ConstInt) and valid_expr.left.value == 32
    assert isinstance(valid_expr.right, ir.ConstInt) and valid_expr.right.value == 1
    _assert_roundtrips(Prog)


def test_distributed_return_kind_and_valid_shape_are_preserved():
    """A DSL call preserves DistributedTensorType while substituting its valid shape."""
    n = pl.dynamic("N")

    @pl.program
    class Prog:
        @pl.function
        def view(
            self,
            shape_source: pl.Tensor[[n], pl.FP32],
            arg: pld.DistributedTensor[[64, 64], pl.FP32],
        ):
            return pl.tensor.slice(arg, [64, 64], [0, 0], valid_shape=[n - 1, 64])

        @pl.function
        def main(
            self,
            actual_shape_source: pl.Tensor[[32], pl.FP32],
            actual: pld.DistributedTensor[[64, 64], pl.FP32],
        ):
            return self.view(actual_shape_source, actual)

    (deduced,) = _user_call_types(Prog, "view")
    assert isinstance(deduced, ir.DistributedTensorType)
    assert [str(dim) for dim in deduced.shape] == ["64", "64"]
    assert deduced.tensor_view is not None
    valid_expr = deduced.tensor_view.valid_shape[0]
    assert isinstance(valid_expr, ir.Sub)
    assert isinstance(valid_expr.left, ir.ConstInt) and valid_expr.left.value == 32
    assert isinstance(valid_expr.right, ir.ConstInt) and valid_expr.right.value == 1
    _assert_roundtrips(Prog)


def test_tensor_dim_folds_onto_the_signature_dyn_symbol():
    """``pl.tensor.dim(x, i)`` in an Orchestration body IS the extent x's type names.

    Re-reading a declared extent at runtime would mint a second scalar for the
    same quantity, and nothing downstream can prove the copy equal to the symbol
    the tensor types carry. Fold instead: one runtime extent, one IR name.
    """
    tokens = pl.dynamic("TOKENS_DYN")

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[tokens, 128], pl.FP32]):
            n = pl.tensor.dim(x, 0)
            return pl.create_tensor([n, 128], dtype=pl.FP32)

    body = ir.python_print(Prog)
    # The dim read folded away, and the temp is shaped by the symbol itself --
    # not by a second scalar that happens to hold the same number.
    assert "pl.tensor.dim(" not in body, body
    assert "pl.tensor.create([TOKENS_DYN, 128]" in body, body
    _assert_roundtrips(Prog)


def test_dyn_symbol_survives_call_returning_into_a_loop_carried_var():
    """The Qwen3-14B prefill shape: a shared dyn symbol + a `tensor.dim`-sized temp.

    ``hidden_states`` is reassigned from a callee that declares the *same*
    ``pl.dynamic`` symbol, while the callee's ``out`` argument is a temp the
    caller sized from ``pl.tensor.dim(hidden_states, 0)``. Before the fold, that
    temp carried a second name for the token extent, the call's deduced return
    type came back in terms of that name, and reassigning ``hidden_states`` was
    rejected as a type change (``was pl.Tensor[[TOKENS_DYN, ...]], got
    pl.Tensor[[n, ...]]``).
    """
    tokens = pl.dynamic("TOKENS_DYN")

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
        def layer(
            self,
            hidden_states: pl.Tensor[[tokens, 128], pl.BF16],
            out: pl.Tensor[[tokens, 128], pl.BF16],
        ) -> pl.Tensor[[tokens, 128], pl.BF16]:
            return out

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, hidden_states: pl.Tensor[[tokens, 128], pl.BF16]) -> pl.Tensor[[tokens, 128], pl.BF16]:
            n = pl.tensor.dim(hidden_states, 0)
            for _ in pl.range(2):
                with pl.scope():
                    nxt = pl.create_tensor([n, 128], dtype=pl.BF16)
                    hidden_states = self.layer(hidden_states, nxt)
            return hidden_states

    # The call's return type keeps the caller's own symbol, so the reassignment
    # of `hidden_states` type-checks against the parameter it was declared with.
    (call_type,) = _user_call_types(Prog, "layer")
    assert isinstance(call_type, ir.TensorType)
    assert [str(dim) for dim in call_type.shape] == ["TOKENS_DYN", "128"]
    _assert_roundtrips(Prog)


def test_reassigning_a_folded_name_never_writes_into_the_symbol():
    """Rebinding the folded name must not assign *through* it into the symbol.

    The name is bound to the symbol, and the symbol names an argument's extent —
    it is immutable. A non-leaking scope (an ``if`` with explicit yields) rebinds
    ``n`` and then exits, restoring the outer ``n -> symbol`` binding, so the
    guard has to test the Var the name is bound to rather than track names.
    Writing to the symbol would corrupt the extent for every later use of it.
    """
    tokens = pl.dynamic("TOKENS_DYN")
    cols = pl.dynamic("COLS_DYN")

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[tokens, cols], pl.FP32], flag: pl.Scalar[pl.BOOL]):
            n = pl.tensor.dim(x, 0)  # n -> TOKENS_DYN
            if flag:
                n = 3  # rebind inside a scope that does not leak its vars
                k = pl.yield_(1)
            else:
                k = pl.yield_(2)
            n = 5  # outer scope: n is bound to the symbol again
            return pl.create_tensor([n, k], dtype=pl.FP32)

    printed = ir.python_print(Prog)
    # The symbol is never an assignment target -- it stays a pure extent name.
    assert "TOKENS_DYN: pl.Scalar" not in printed, printed
    assert "TOKENS_DYN = 5" not in printed, printed
    _assert_roundtrips(Prog)


def test_tensor_dim_folds_for_every_call_spelling():
    """Keyword args and an annotated LHS fold too.

    The printer normalizes every spelling to ``n: pl.Scalar[pl.INDEX] =
    pl.tensor.dim(x, 0)``, so a spelling that did not fold would fold on reparse
    and break the round-trip -- and would quietly reintroduce the
    second-name-for-one-extent bug for the source that used it.
    """
    tokens = pl.dynamic("TOKENS_DYN")
    row_axis = 0

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def keyword_axis(self, x: pl.Tensor[[tokens, 128], pl.FP32]):
            n = pl.tensor.dim(x, axis=0)
            return pl.create_tensor([n, 128], dtype=pl.FP32)

        @pl.function(type=pl.FunctionType.Orchestration)
        def keyword_tensor(self, x: pl.Tensor[[tokens, 128], pl.FP32]):
            n = pl.tensor.dim(tensor=x, axis=0)
            return pl.create_tensor([n, 128], dtype=pl.FP32)

        @pl.function(type=pl.FunctionType.Orchestration)
        def named_const_axis(self, x: pl.Tensor[[tokens, 128], pl.FP32]):
            n = pl.tensor.dim(x, row_axis)
            return pl.create_tensor([n, 128], dtype=pl.FP32)

        @pl.function(type=pl.FunctionType.Orchestration)
        def annotated_target(self, x: pl.Tensor[[tokens, 128], pl.FP32]):
            n: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
            return pl.create_tensor([n, 128], dtype=pl.FP32)

    printed = ir.python_print(Prog)
    assert "pl.tensor.dim(" not in printed, printed
    assert printed.count("pl.tensor.create([TOKENS_DYN, 128]") == 4, printed
    _assert_roundtrips(Prog)


def test_tensor_dim_is_not_folded_in_an_inline_callee():
    """An Inline callee's ``tensor.dim`` must stay a runtime read.

    Its placeholder is not the caller's: the same callee can be inlined into a
    caller that passes a statically-shaped tensor, and ``InlineFunctions`` does
    not rewrite the callee's dyn symbols. Only an Orchestration body -- where
    codegen defines the symbol from the task-arg descriptor -- may fold.
    """
    tokens = pl.dynamic("TOKENS_DYN")

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Inline)
        def layer(self, x: pl.Tensor[[tokens, 128], pl.FP32]):
            n = pl.tensor.dim(x, 0)
            return pl.create_tensor([n, 128], dtype=pl.FP32)

    body = ir.python_print(Prog)
    assert "pl.tensor.dim(x, 0)" in body
    _assert_roundtrips(Prog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

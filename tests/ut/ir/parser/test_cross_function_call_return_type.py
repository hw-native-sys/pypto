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


@pytest.mark.parametrize(
    "type_text",
    [
        "pl.Scalar[pl.INDEX]",
        "pl.Ptr",
        "pl.Tensor[[4], pl.FP32]",
        "pl.Tile[[4], pl.FP32, pl.Mem.Vec]",
        "pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.FP32]]",
    ],
)
def test_recorded_call_supports_non_distributed_result_types(type_text):
    """The private Call syntax preserves ordinary types and call metadata too."""
    program = pl.parse_program(f"""
@pl.program
class P:
    @pl.function
    def identity(self, x: {type_text}) -> {type_text}:
        return x

    @pl.function
    def caller(self, x: {type_text}):
        y = pl._dump.call(self.identity, [x], {type_text}, attrs={{"saved": True}})
        return y
""")
    caller_gvar = program.get_global_var("caller")
    assert caller_gvar is not None
    caller = program.functions[caller_gvar]
    assert isinstance(caller.body, ir.SeqStmts)
    assignment = caller.body.stmts[0]
    assert isinstance(assignment, ir.AssignStmt)
    assert isinstance(assignment.value, ir.Call)
    callee_gvar = program.get_global_var("identity")
    assert callee_gvar is not None
    assert isinstance(assignment.value.op, ir.GlobalVar)
    assert assignment.value.op.name == callee_gvar.name
    assert assignment.value.args[0] is caller.params[0]
    ir.assert_structural_equal(assignment.value.type, caller.params[0].type)
    assert assignment.value.attrs == {"saved": True}
    _assert_roundtrips(program)


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
            tile = pl.load(arg, [0, 0], [n, 64], valid_shape=[n - 1, 64])
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
    caller that passes a statically-shaped tensor. Shape specialization happens
    later in ``InlineFunctions``; parsing must preserve the runtime read. Only
    an Orchestration body -- where codegen defines the symbol from the task-arg
    descriptor -- may fold at this stage.
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


@pytest.mark.parametrize(
    "expression, extent", [("signed_shift", 21), ("same_symbol", 16), ("constant_actual", 16)]
)
def test_call_return_type_solves_single_variable_integer_affine_extents(expression, extent):
    """Infer a concrete return extent through signed offsets or repeated uses of one symbol."""
    span = ir.Span.unknown()
    rows = ir.Var("rows", ir.ScalarType(pl.INDEX), span)
    if expression == "signed_shift":
        scaled = ir.Mul(ir.ConstInt(2, pl.INDEX, span), rows, pl.INDEX, span)
        pattern = ir.Sub(ir.ConstInt(5, pl.INDEX, span), ir.Neg(scaled, pl.INDEX, span), pl.INDEX, span)
    else:
        pattern = ir.Add(rows, rows, pl.INDEX, span)
    param = ir.Var("tile", ir.TileType([pattern, 64], pl.FP32), span)
    actual_extent = (
        ir.Add(ir.ConstInt(8, pl.INDEX, span), ir.ConstInt(8, pl.INDEX, span), pl.INDEX, span)
        if expression == "constant_actual"
        else extent
    )
    actual = ir.Var("actual", ir.TileType([actual_extent, 64], pl.FP32), span)
    return_type = ir.TileType([rows, 64], pl.FP32)

    (deduced,) = ir.deduce_call_return_type([param], [actual], [return_type])

    assert isinstance(deduced, ir.TileType)
    assert isinstance(deduced.shape[0], ir.ConstInt)
    assert deduced.shape[0].value == 8


@pytest.mark.parametrize("expression", ["multiple_symbols", "nonlinear", "narrow_overflow"])
def test_call_return_type_does_not_invent_unsafe_affine_bindings(expression):
    """Ambiguous, nonlinear, and out-of-range solutions leave the return symbol unbound."""
    span = ir.Span.unknown()
    dtype = pl.INT8 if expression == "narrow_overflow" else pl.INDEX
    rows = ir.Var("rows", ir.ScalarType(dtype), span)
    if expression == "multiple_symbols":
        other = ir.Var("other", ir.ScalarType(dtype), span)
        pattern = ir.Add(rows, other, dtype, span)
    elif expression == "nonlinear":
        pattern = ir.Mul(rows, rows, dtype, span)
    else:
        pattern = ir.Mul(ir.ConstInt(2, dtype, span), rows, dtype, span)
    extent = 256 if expression == "narrow_overflow" else 16
    param = ir.Var("tile", ir.TileType([pattern, 64], pl.FP32), span)
    actual = ir.Var("actual", ir.TileType([extent, 64], pl.FP32), span)
    return_type = ir.TileType([rows, 64], pl.FP32)

    (deduced,) = ir.deduce_call_return_type([param], [actual], [return_type])

    ir.assert_structural_equal(deduced, return_type)


@pytest.mark.parametrize("contract", ["surviving_tile", "inline_tensor", "inline_explicit_tile_view"])
def test_call_return_type_keeps_shared_shape_contracts_strict(contract):
    """Inline physical placeholders do not relax surviving calls or explicit shared metadata."""
    span = ir.Span.unknown()
    rows = ir.Var("rows", ir.ScalarType(pl.INDEX), span)
    pattern = ir.Mul(ir.ConstInt(2, pl.INDEX, span), rows, pl.INDEX, span)
    if contract == "inline_tensor":
        formal_type = ir.TensorType([pattern, 64], pl.FP32)
        first_type = ir.TensorType([16, 64], pl.FP32)
        second_type = ir.TensorType([32, 64], pl.FP32)
    elif contract == "inline_explicit_tile_view":
        formal_type = ir.TileType([64, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[rows, 64]))
        first_type = ir.TileType([64, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[16, 64]))
        second_type = ir.TileType([64, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[32, 64]))
    else:
        formal_type = ir.TileType([pattern, 64], pl.FP32)
        first_type = ir.TileType([16, 64], pl.FP32)
        second_type = ir.TileType([32, 64], pl.FP32)
    params = [ir.Var("first", formal_type, span), ir.Var("second", formal_type, span)]
    args = [ir.Var("first_actual", first_type, span), ir.Var("second_actual", second_type, span)]

    with pytest.raises(ValueError, match="conflicting bindings"):
        if contract == "surviving_tile":
            ir.deduce_call_return_type(params, args, [formal_type])
        else:
            ir.deduce_call_return_type(params, args, [formal_type], is_inline=True)


@pytest.mark.parametrize("is_inline", [False, True], ids=["surviving", "inline"])
def test_inline_return_deduction_preserves_transitive_shared_tensor_bindings(is_inline):
    """A = 2*B and A = 32 must retain their proven binding after a later argument binds B = 16."""
    span = ir.Span.unknown()
    first_dim = ir.Var("A", ir.ScalarType(pl.INDEX), span)
    later_dim = ir.Var("B", ir.ScalarType(pl.INDEX), span)
    two = ir.ConstInt(2, pl.INDEX, span)
    shared_type = ir.TensorType([first_dim], pl.FP32)
    params = [
        ir.Var("first", shared_type, span),
        ir.Var("second", shared_type, span),
        ir.Var("marker", ir.TensorType([later_dim], pl.FP32), span),
    ]
    args = [
        ir.Var("first_actual", ir.TensorType([ir.Mul(two, later_dim, pl.INDEX, span)], pl.FP32), span),
        ir.Var("second_actual", ir.TensorType([32], pl.FP32), span),
        ir.Var("marker_actual", ir.TensorType([16], pl.FP32), span),
    ]

    (deduced,) = ir.deduce_call_return_type(params, args, [shared_type], is_inline=is_inline)

    expected = ir.TensorType([ir.Mul(two, ir.ConstInt(16, pl.INDEX, span), pl.INDEX, span)], pl.FP32)
    ir.assert_structural_equal(deduced, expected)


def test_external_inline_call_uses_independent_tile_dimension_deduction():
    """A separately decorated Inline function uses the same parser policy as self.method calls."""
    rows = pl.dynamic("EXTERNAL_INLINE_GENERIC_ROWS")

    @pl.function(type=pl.FunctionType.Inline)
    def select_first(first: pl.Tile[[2 * rows, 64], pl.FP32], second: pl.Tile[[2 * rows, 64], pl.FP32]):
        return first

    @pl.program
    class Prog:
        @pl.function
        def main(self, first: pl.Tile[[16, 64], pl.FP32], second: pl.Tile[[32, 64], pl.FP32]):
            return select_first(first, second)

    (deduced,) = _user_call_types(Prog, "select_first")
    assert isinstance(deduced, ir.TileType)
    assert isinstance(deduced.shape[0], ir.ConstInt)
    assert deduced.shape[0].value == 16


def test_rebound_inline_parameter_does_not_claim_the_incoming_tile_view():
    """Returning a rebound formal is not evidence that its original actual Tile is returned."""
    rows = pl.dynamic("REBOUND_INLINE_GENERIC_ROWS")

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Inline)
        def replace(self, tile: pl.Tile[[rows, 64], pl.FP32], replacement: pl.Tile[[rows, 64], pl.FP32]):
            tile = replacement
            return tile

        @pl.function
        def main(
            self,
            source: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[4, 64])],
            replacement: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[3, 64])],
        ):
            return self.replace(source, replacement)

    (deduced,) = _user_call_types(Prog, "replace")
    assert isinstance(deduced, ir.TileType)
    # Native deduction remains provisional here. Inlining later recovers the
    # replacement's valid extent 3; direct-return provenance must not claim 4.
    valid_rows = deduced.get_effective_tile_view().valid_shape[0]
    assert isinstance(valid_rows, ir.ConstInt)
    assert valid_rows.value == 16


@pytest.mark.parametrize("extent", ["ROWS", "2 * ROWS"], ids=["plain", "affine"])
@pytest.mark.parametrize("returned", ["first", "second", "chain", "tuple"])
def test_inline_immutable_alias_return_keeps_actual_tile_type(extent, returned):
    """An alias of an independently specialized Tile retains the chosen caller operand's type."""
    bodies = {
        "first": "alias = first\n        return alias",
        "second": "alias = second\n        return alias",
        "chain": "alias = second\n        another = alias\n        result = another\n        return result",
        "tuple": "left = first\n        right = second\n        return left, right",
    }
    target = "first" if returned == "first" else "second"
    call = "_left, selected" if returned == "tuple" else "selected"
    program = pl.parse_program(f"""
ROWS = pl.dynamic("IMMUTABLE_ALIAS_ROWS")
@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.Inline)
    def choose(self, first: pl.Tile[[{extent}, 64], pl.FP32], second: pl.Tile[[{extent}, 64], pl.FP32]):
        {bodies[returned]}

    @pl.function
    def main(self, first: pl.Tile[[16, 64], pl.FP32], second: pl.Tile[[32, 64], pl.FP32]):
        {call} = self.choose(first, second)
        return pl.add(selected, {target})
""")
    (deduced,) = _user_call_types(program, "choose")
    if returned == "tuple":
        assert isinstance(deduced, ir.TupleType)
        deduced = deduced.types[1]
    assert isinstance(deduced, ir.TileType)
    rows = deduced.shape[0]
    assert isinstance(rows, ir.ConstInt)
    assert rows.value == (16 if returned == "first" else 32)


def test_inline_alias_of_tuple_projection_keeps_actual_tile_type():
    """A transparent tuple alias and projection remain tied to the original argument."""
    program = pl.parse_program("""
ROWS = pl.dynamic("TUPLE_ALIAS_ROWS")
@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.Inline)
    def choose(self, pair: pl.Tuple[pl.Tile[[ROWS, 64], pl.FP32], pl.Tile[[ROWS, 64], pl.FP32]]):
        alias = pair
        selected = alias[1]
        result = selected
        return result

    @pl.function
    def main(self, first: pl.Tile[[16, 64], pl.FP32], second: pl.Tile[[32, 64], pl.FP32]):
        selected = self.choose((first, second))
        return pl.add(selected, second)
""")
    (deduced,) = _user_call_types(program, "choose")
    assert isinstance(deduced, ir.TileType)
    rows = deduced.shape[0]
    assert isinstance(rows, ir.ConstInt)
    assert rows.value == 32


@pytest.mark.parametrize("mutation", ["alias", "formal", "conditional", "computed"])
def test_inline_alias_recovery_keeps_mutation_and_computation_guards(mutation):
    """Rebound, conditional, and computed results must keep native provisional metadata."""
    bodies = {
        "alias": "alias = second\n        alias = first\n        return alias",
        "formal": "alias = second\n        second = first\n        return alias",
        "conditional": (
            "alias = second\n        if condition > 0:\n            alias = first\n        return alias"
        ),
        "computed": "alias = pl.add(first, second)\n        return alias",
    }
    program = pl.parse_program(f"""
ROWS = pl.dynamic("GUARDED_ALIAS_ROWS")
@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.Inline)
    def choose(self, first: pl.Tile[[ROWS, 64], pl.FP32], second: pl.Tile[[ROWS, 64], pl.FP32],
               condition: pl.Scalar[pl.INDEX]):
        {bodies[mutation]}

    @pl.function
    def main(self, first: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[4, 64])],
             second: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[3, 64])]):
        return self.choose(first, second, 1)
""")
    (deduced,) = _user_call_types(program, "choose")
    assert isinstance(deduced, ir.TileType)
    rows = deduced.get_effective_tile_view().valid_shape[0]
    assert isinstance(rows, ir.ConstInt)
    assert rows.value == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

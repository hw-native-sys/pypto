# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Caller-owned storage for computed Tensor results consumed inside a mixed Group."""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.op import tensor as tensor_ops
from pypto.ir.op import tile as tile_ops


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _function(program: ir.Program, name: str) -> ir.Function:
    function = program.get_function(name)
    assert function is not None, f"Expected function {name!r}"
    return function


def _select(program: ir.Program, names: tuple[str, ...]) -> ir.Program:
    return ir.Program(
        [function for function in program.functions.values() if function.name in names],
        program.name,
        program.span,
    )


def _global_launches(function: ir.Function) -> list[ir.Call | ir.Submit]:
    return [
        statement.value
        for statement in ir.flatten_to_stmts(function.body)
        if isinstance(statement, ir.AssignStmt)
        and isinstance(statement.value, (ir.Call, ir.Submit))
        and isinstance(statement.value.op, ir.GlobalVar)
    ]


def _builtin_calls(function: ir.Function, name: str) -> list[ir.Call]:
    return [
        statement.value
        for statement in ir.flatten_to_stmts(function.body)
        if isinstance(statement, ir.AssignStmt)
        and isinstance(statement.value, ir.Call)
        and statement.value.op.name == ir.get_op(name).name
    ]


def _all_builtin_calls(function: ir.Function, name: str) -> list[ir.Call]:
    """Include unassigned scalar writes as well as assignment RHS calls."""
    calls: list[ir.Call] = []
    op_name = ir.get_op(name).name

    class CollectCalls(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == op_name:
                calls.append(op)
            super().visit_call(op)

    CollectCalls().visit_function(function)
    return calls


def _lower(program: ir.Program) -> ir.Program:
    return passes.convert_tensor_to_tile_ops()(passes.convert_to_ssa()(program))


def _expand(program: ir.Program) -> ir.Program:
    after = passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(program))
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.SSAForm)
    properties.insert(passes.IRProperty.UseAfterDef)
    passes.run_verifier(properties)(after)
    assert "FREE_VAR" not in after.as_python()
    return after


@pytest.mark.parametrize("use_submit", [False, True])
def test_computed_tuple_result_uses_its_own_group_storage(use_submit):
    """Return position one is a fresh add result; position zero remains the original input."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def producer(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> tuple[pl.Tensor[[16, 16], pl.FP16], pl.Tensor[[16, 16], pl.FP16]]:
            fresh = pl.add(x, y)
            return x, fresh

        @pl.function(type=pl.FunctionType.AIC)
        def consumer(
            self, fresh: pl.Tensor[[16, 16], pl.FP16], original: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            result = pl.matmul(fresh, original, out_dtype=pl.FP32)
            return result

        @pl.function(type=pl.FunctionType.Group)
        def mixed_call(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            pair = self.producer(x, y)
            result = self.consumer(pl.no_dep(pair[1]), pair[0])
            return result

        @pl.function(type=pl.FunctionType.Group)
        def mixed_submit(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            pair = pl.submit(self.producer, x, y)
            result, _consumer_tid = pl.submit(self.consumer, pl.no_dep(pair[1]), pair[0])
            return result

        @pl.function(type=pl.FunctionType.Orchestration)
        def main_call(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            result = self.mixed_call(x, y)
            return result

        @pl.function(type=pl.FunctionType.Orchestration)
        def main_submit(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            _first, first_tid = pl.submit(self.mixed_submit, x, y)
            result, _second_tid = pl.submit(self.mixed_submit, y, x, deps=[first_tid])
            return result

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def producer(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            y: pl.Tensor[[16, 16], pl.FP16],
            scratch: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> tuple[pl.Tensor[[16, 16], pl.FP16], pl.Tensor[[16, 16], pl.FP16]]:
            return x, scratch

        @pl.function(type=pl.FunctionType.AIC)
        def consumer(
            self,
            fresh: pl.Tensor[[16, 16], pl.FP16],
            original: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Group)
        def mixed_call(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            y: pl.Tensor[[16, 16], pl.FP16],
            scratch: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            pair = self.producer(x, y, scratch)
            result = self.consumer(pl.no_dep(pair[1]), pair[0], out)
            return result

        @pl.function(type=pl.FunctionType.Group)
        def mixed_submit(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            y: pl.Tensor[[16, 16], pl.FP16],
            scratch: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            pair = pl.submit(self.producer, x, y, scratch)
            result, _consumer_tid = pl.submit(self.consumer, pl.no_dep(pair[1]), pair[0], out)
            return result

        @pl.function(type=pl.FunctionType.Orchestration)
        def main_call(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            scratch = pl.tensor.create([16, 16], dtype=pl.FP16, layout=pl.TensorLayout.ND)
            out = pl.tensor.create([16, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            result = self.mixed_call(x, y, scratch, out)
            return result

        @pl.function(type=pl.FunctionType.Orchestration)
        def main_submit(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            first_scratch = pl.tensor.create([16, 16], dtype=pl.FP16, layout=pl.TensorLayout.ND)
            first_out = pl.tensor.create([16, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            _first, first_tid = pl.submit(self.mixed_submit, x, y, first_scratch, first_out)
            second_scratch = pl.tensor.create([16, 16], dtype=pl.FP16, layout=pl.TensorLayout.ND)
            second_out = pl.tensor.create([16, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            result, _second_tid = pl.submit(
                self.mixed_submit, y, x, second_scratch, second_out, deps=[first_tid]
            )
            return result

    kind = "submit" if use_submit else "call"
    group_name, main_name = f"mixed_{kind}", f"main_{kind}"
    before = _select(Before, ("producer", "consumer", group_name, main_name))
    lowered = _lower(before)
    for name in (group_name, main_name):
        ir.assert_structural_equal(_function(lowered, name), _function(Expected, name))
    assert len(_builtin_calls(_function(lowered, "producer"), "tile.add")) == 1
    assert len(_builtin_calls(_function(lowered, "consumer"), "tile.matmul")) == 1

    after = _expand(lowered)
    group = _function(after, group_name)
    producer, consumer = _function(after, "producer"), _function(after, "consumer")
    assert producer.func_type == ir.FunctionType.AIV
    assert consumer.func_type == ir.FunctionType.AIC
    assert _builtin_calls(producer, "tile.store")[0].args[2].same_as(producer.params[2])
    loads = _builtin_calls(consumer, "tile.load")
    assert [load.args[0] for load in loads] == [consumer.params[2], consumer.params[0]]
    assert _builtin_calls(consumer, "tile.store")[0].args[2].same_as(consumer.params[3])
    launches = _global_launches(group)
    assert len(launches) == 2
    assert all(isinstance(launch, ir.Submit if use_submit else ir.Call) for launch in launches)
    assert all(list(launch.args) == list(group.params) for launch in launches)
    assert launches[1].attrs["arg_direction_overrides"] == [2]
    if use_submit:
        consumer_submit = launches[1]
        assert isinstance(consumer_submit, ir.Submit)
        assert len(consumer_submit.deps) == 0
        assert isinstance(launches[0].type, ir.TupleType)
        task_id_type = launches[0].type.types[-1]
        assert isinstance(task_id_type, ir.ScalarType)
        assert task_id_type.dtype == ir.DataType.TASK_ID
        outer_submits = _global_launches(_function(after, main_name))
        assert len(outer_submits) == 2
        assert all(isinstance(submit, ir.Submit) for submit in outer_submits)
        second_submit = outer_submits[1]
        assert isinstance(second_submit, ir.Submit)
        assert len(second_submit.deps) == 1


def test_raw_expand_reports_computed_output_without_caller_storage():
    """Skipping output materialization must name the producer result before emitting FREE_VAR."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def producer(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            fresh = pl.add(x, y)
            return fresh

        @pl.function(type=pl.FunctionType.AIC)
        def consumer(self, fresh: pl.Tensor[[16, 16], pl.FP16]) -> pl.Tensor[[16, 16], pl.FP16]:
            return fresh

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            fresh = self.producer(x, y)
            result = self.consumer(fresh)
            return result

    # This negative test intentionally omits conversion/output materialization;
    # let ExpandMixedKernel diagnose it instead of checking its tile-only input contract.
    with passes.PassContext([], passes.VerificationLevel.NONE):
        with pytest.raises(
            ValueError,
            match=(
                "Group 'mixed' member 'consumer' argument 0 consumes output 0 of producer 'producer' "
                "without a caller-visible output binding.*ConvertTensorToTileOps"
            ),
        ):
            passes.expand_mixed_kernel()(passes.convert_to_ssa()(Before))


def test_existing_out_and_fresh_output_keep_independent_return_positions():
    """A fresh return preceding an existing Out must not select that existing output's storage."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def producer(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            y: pl.Tensor[[16, 16], pl.FP16],
            existing: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> tuple[pl.Tensor[[16, 16], pl.FP16], pl.Tensor[[16, 16], pl.FP16]]:
            fresh = pl.add(x, y)
            copied = pl.store(pl.load(x, [0, 0], [16, 16]), [0, 0], existing)
            return fresh, copied

        @pl.function(type=pl.FunctionType.AIC)
        def consumer(
            self, fresh: pl.Tensor[[16, 16], pl.FP16], copied: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            product = pl.matmul(fresh, copied, out_dtype=pl.FP32)
            return product

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            y: pl.Tensor[[16, 16], pl.FP16],
            existing: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            pair = self.producer(x, y, existing)
            product = self.consumer(pl.no_dep(pair[0]), pair[1])
            return product

    lowered = _lower(Before)
    lowered_group = _function(lowered, "mixed")
    assert len(lowered_group.params) == 5
    assert lowered_group.param_directions[2:] == [ir.ParamDirection.Out] * 3
    launches = _global_launches(lowered_group)
    assert list(launches[0].args) == list(lowered_group.params[:4])
    assert launches[1].args[-1].same_as(lowered_group.params[4])

    after = _expand(lowered)
    consumer = _function(after, "consumer")
    loads = _builtin_calls(consumer, "tile.load")
    assert [load.args[0] for load in loads] == [consumer.params[3], consumer.params[2]]
    assert _builtin_calls(consumer, "tile.store")[0].args[2].same_as(consumer.params[4])
    assert _global_launches(_function(after, "mixed"))[1].attrs["arg_direction_overrides"] == [3]


def test_repeated_computed_producer_calls_have_distinct_group_storage():
    """Each call instance owns an output buffer, even when the same callee is used twice."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def producer(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            fresh = pl.add(x, y)
            return fresh

        @pl.function(type=pl.FunctionType.AIC)
        def consumer(
            self, left: pl.Tensor[[16, 16], pl.FP16], right: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            product = pl.matmul(left, right, out_dtype=pl.FP32)
            return product

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            left = self.producer(x, y)
            right = self.producer(y, x)
            product = self.consumer(left, right)
            return product

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            result = self.mixed(x, y)
            return result

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def producer(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            y: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            return out

        @pl.function(type=pl.FunctionType.AIC)
        def consumer(
            self,
            left: pl.Tensor[[16, 16], pl.FP16],
            right: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            y: pl.Tensor[[16, 16], pl.FP16],
            left_out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            right_out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            result_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            left = self.producer(x, y, left_out)
            right = self.producer(y, x, right_out)
            product = self.consumer(left, right, result_out)
            return product

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            left_out = pl.tensor.create([16, 16], dtype=pl.FP16, layout=pl.TensorLayout.ND)
            right_out = pl.tensor.create([16, 16], dtype=pl.FP16, layout=pl.TensorLayout.ND)
            result_out = pl.tensor.create([16, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            result = self.mixed(x, y, left_out, right_out, result_out)
            return result

    after = _lower(Before)
    for name in ("mixed", "main"):
        ir.assert_structural_equal(_function(after, name), _function(Expected, name))
    group = _function(after, "mixed")
    launches = _global_launches(group)
    assert launches[0].args[-1].same_as(group.params[2])
    assert launches[1].args[-1].same_as(group.params[3])
    assert not launches[0].args[-1].same_as(launches[1].args[-1])
    assert len(_builtin_calls(_function(after, "main"), "tensor.create")) == 3


def test_shared_computed_callees_keep_each_groups_storage_private():
    """Two Groups and an ordinary producer call must never share caller-allocated output buffers."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def producer(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            fresh = pl.add(x, y)
            return fresh

        @pl.function(type=pl.FunctionType.AIC)
        def consumer(
            self, left: pl.Tensor[[16, 16], pl.FP16], right: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            product = pl.matmul(left, right, out_dtype=pl.FP32)
            return product

        @pl.function(type=pl.FunctionType.Group)
        def first_group(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            fresh = self.producer(x, y)
            product = self.consumer(pl.no_dep(fresh), x)
            return product

        @pl.function(type=pl.FunctionType.Group)
        def second_group(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            fresh = self.producer(y, x)
            product = self.consumer(pl.no_dep(fresh), y)
            return product

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
            external = self.producer(x, y)
            first = self.first_group(external, y)
            second = self.second_group(y, external)
            return first, second

    lowered = _lower(Before)
    main = _function(lowered, "main")
    launches = _global_launches(main)
    assert [launch.op.name for launch in launches] == ["producer", "first_group", "second_group"]
    storage = [launches[0].args[-1], *launches[1].args[-2:], *launches[2].args[-2:]]
    storage_ids = set()
    for value in storage:
        assert isinstance(value, ir.Var)
        storage_ids.add(value.unique_id)
    assert len(storage_ids) == 5
    assert len(_builtin_calls(main, "tensor.create")) == 5

    after = _expand(lowered)
    original_producer = _function(after, "producer")
    original_consumer = _function(after, "consumer")
    assert len(original_producer.params) == len(original_consumer.params) == 3
    assert _builtin_calls(original_producer, "tile.store")[0].args[2].same_as(original_producer.params[2])
    for name, input_indices, old_index in (
        ("first_group", [0, 1], 0),
        ("second_group", [1, 0], 1),
    ):
        group = _function(after, name)
        producer = _function(after, f"{name}_aiv_adapter")
        consumer = _function(after, f"{name}_aic_adapter")
        assert len(group.params) == len(producer.params) == len(consumer.params) == 4
        assert [load.args[0] for load in _builtin_calls(producer, "tile.load")] == [
            producer.params[index] for index in input_indices
        ]
        assert _builtin_calls(producer, "tile.store")[0].args[2].same_as(producer.params[2])
        assert [load.args[0] for load in _builtin_calls(consumer, "tile.load")] == [
            consumer.params[2],
            consumer.params[old_index],
        ]
        assert _builtin_calls(consumer, "tile.store")[0].args[2].same_as(consumer.params[3])
        group_launches = _global_launches(group)
        assert all(list(launch.args) == list(group.params) for launch in group_launches)
        assert group_launches[1].attrs["arg_direction_overrides"] == [2]


@pytest.mark.parametrize("add_fresh_output", [False, True])
def test_omitted_existing_out_precedes_added_output_and_comm_ctx(add_fresh_output):
    """The old Submit output gap is filled before new outputs, with ctx still at the end."""
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([16, 16], ir.DataType.FP16)
    ctx_type = ir.CommCtxType()
    x, old_out, ctx = (
        ir.Var("x", tensor_type, span),
        ir.Var("old_out", tensor_type, span),
        ir.Var("ctx", ctx_type, span),
    )
    loaded = tile_ops.load(x, [0, 0], [16, 16])
    tile = ir.Var("tile", loaded.type, span)
    stored = tile_ops.store(tile, [0, 0], old_out)
    copied = ir.Var("copied", stored.type, span)
    producer_body: list[ir.Stmt] = [ir.AssignStmt(tile, loaded, span), ir.AssignStmt(copied, stored, span)]
    returned: list[ir.Expr] = [copied]
    if add_fresh_output:
        computed = tensor_ops.add(x, x)
        fresh = ir.Var("fresh", computed.type, span)
        producer_body.append(ir.AssignStmt(fresh, computed, span))
        # Logical result zero belongs to the NEW output parameter; result one
        # belongs to the original omitted Out parameter.
        returned.insert(0, fresh)
    producer_body.append(ir.ReturnStmt(returned, span))
    producer = ir.Function(
        "producer",
        [(x, ir.ParamDirection.In), (old_out, ir.ParamDirection.Out), (ctx, ir.ParamDirection.In)],
        [tensor_type] * len(returned),
        ir.SeqStmts(producer_body, span),
        span,
        ir.FunctionType.AIV,
    )
    submit_type = ir.TupleType([tensor_type] * len(returned) + [ir.ScalarType(ir.DataType.TASK_ID)])
    group_x, group_ctx = ir.Var("x", tensor_type, span), ir.Var("ctx", ctx_type, span)
    pair, result = ir.Var("pair", submit_type, span), ir.Var("result", tensor_type, span)
    submit = ir.Submit(
        ir.GlobalVar("producer"),
        [group_x, group_ctx],
        [],
        {},
        {"arg_directions": [ir.ArgDirection.Input, ir.ArgDirection.Scalar]},
        submit_type,
        span,
    )
    group = ir.Function(
        "mixed",
        [group_x, group_ctx],
        [tensor_type],
        ir.SeqStmts(
            [
                ir.AssignStmt(pair, submit, span),
                ir.AssignStmt(result, ir.TupleGetItemExpr(pair, 0, span), span),
                ir.ReturnStmt([result], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Group,
    )
    main_x, main_ctx = ir.Var("x", tensor_type, span), ir.Var("ctx", ctx_type, span)
    main_result = ir.Var("result", tensor_type, span)
    main = ir.Function(
        "main",
        [main_x, main_ctx],
        [tensor_type],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    main_result, ir.Call(ir.GlobalVar("mixed"), [main_x, main_ctx], tensor_type, span), span
                ),
                ir.ReturnStmt([main_result], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Orchestration,
    )
    before = ir.Program([producer, group, main], "omitted_group_output_with_ctx", span)

    expected_x, expected_ctx = ir.Var("x", tensor_type, span), ir.Var("ctx", ctx_type, span)
    storage = [ir.Var("old_buffer", tensor_type, span)]
    if add_fresh_output:
        storage.append(ir.Var("new_buffer", tensor_type, span))
    expected_args = [expected_x, *storage, expected_ctx]
    expected_pair = ir.Var("pair", submit_type, span)
    expected_result = ir.Var("result", tensor_type, span)
    expected_submit = ir.Submit(
        ir.GlobalVar("producer"),
        expected_args,
        [],
        {},
        {
            "arg_directions": [
                ir.ArgDirection.Input,
                *([ir.ArgDirection.OutputExisting] * len(storage)),
                ir.ArgDirection.Scalar,
            ]
        },
        submit_type,
        span,
    )
    expected_group = ir.Function(
        "mixed",
        [expected_x, *[(value, ir.ParamDirection.Out) for value in storage], expected_ctx],
        [tensor_type],
        ir.SeqStmts(
            [
                ir.AssignStmt(expected_pair, expected_submit, span),
                ir.AssignStmt(expected_result, ir.TupleGetItemExpr(expected_pair, 0, span), span),
                ir.ReturnStmt([expected_result], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Group,
    )

    after = _lower(before)
    ir.assert_structural_equal(_function(after, "mixed"), expected_group)
    lowered_producer = _function(after, "producer")
    assert len(lowered_producer.params) == 3 + int(add_fresh_output)
    assert isinstance(lowered_producer.params[-1].type, ir.CommCtxType)
    caller = _function(after, "main")
    creates = [
        statement.var
        for statement in ir.flatten_to_stmts(caller.body)
        if isinstance(statement, ir.AssignStmt)
        and isinstance(statement.value, ir.Call)
        and statement.value.op.name == ir.get_op("tensor.create").name
    ]
    assert len(creates) == 1 + int(add_fresh_output)
    caller_launch = _global_launches(caller)[0]
    assert list(caller_launch.args) == [caller.params[0], *creates, caller.params[1]]
    assert len({value.unique_id for value in creates}) == len(creates)


@pytest.mark.parametrize("store_shapes", [None, [16, 16]], ids=["implicit_shape", "explicit_shape"])
def test_returned_local_allocation_and_aliases_reuse_one_output_buffer(store_shapes):
    """Promoting a stored GM allocation must not add a second allocation, copy or output for its alias."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def producer(
            self, x: pl.Tensor[[16, 16], pl.FP16]
        ) -> tuple[pl.Tensor[[16, 16], pl.FP16], pl.Tensor[[16, 16], pl.FP16]]:
            buffer = pl.tensor.create([16, 16], dtype=pl.FP16)
            tile = pl.load(x, [0, 0], [16, 16])
            stored = pl.store(tile, [0, 0], buffer, shapes=store_shapes)
            alias = stored
            return stored, alias

        @pl.function(type=pl.FunctionType.AIC)
        def consumer(
            self, left: pl.Tensor[[16, 16], pl.FP16], right: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            product = pl.matmul(left, right, out_dtype=pl.FP32)
            return product

        @pl.function(type=pl.FunctionType.Group)
        def mixed(self, x: pl.Tensor[[16, 16], pl.FP16]) -> pl.Tensor[[16, 16], pl.FP32]:
            pair = self.producer(x)
            product = self.consumer(pl.no_dep(pair[1]), pair[0])
            return product

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16, 16], pl.FP16]) -> pl.Tensor[[16, 16], pl.FP32]:
            product = self.mixed(x)
            return product

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def producer(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            scratch: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> tuple[pl.Tensor[[16, 16], pl.FP16], pl.Tensor[[16, 16], pl.FP16]]:
            return scratch, scratch

        @pl.function(type=pl.FunctionType.AIC)
        def consumer(
            self,
            left: pl.Tensor[[16, 16], pl.FP16],
            right: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            scratch: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            pair = self.producer(x, scratch)
            product = self.consumer(pl.no_dep(pair[1]), pair[0], out)
            return product

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16, 16], pl.FP16]) -> pl.Tensor[[16, 16], pl.FP32]:
            scratch = pl.tensor.create([16, 16], dtype=pl.FP16, layout=pl.TensorLayout.ND)
            out = pl.tensor.create([16, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            product = self.mixed(x, scratch, out)
            return product

    lowered = _lower(Before)
    for name in ("mixed", "main"):
        ir.assert_structural_equal(_function(lowered, name), _function(Expected, name))
    lowered_producer = _function(lowered, "producer")
    assert len(lowered_producer.params) == 2
    assert len(_builtin_calls(lowered_producer, "tile.store")) == 1
    assert not _builtin_calls(lowered_producer, "tensor.create")
    assert not _builtin_calls(_function(lowered, "mixed"), "tensor.create")
    assert len(_builtin_calls(_function(lowered, "main"), "tensor.create")) == 2

    after = _expand(lowered)
    producer = _function(after, "producer")
    stores = _builtin_calls(producer, "tile.store")
    assert len(stores) == 1
    assert len(stores[0].args) == (3 if store_shapes is None else 4)
    if store_shapes is not None:
        shape = stores[0].args[3]
        assert isinstance(shape, ir.MakeTuple)
        assert len(shape.elements) == 2
        for extent in shape.elements:
            assert isinstance(extent, ir.ConstInt)
            assert extent.value == 16
    assert stores[0].args[2].same_as(producer.params[1])
    definitions = {
        statement.var.unique_id: statement.value
        for statement in ir.flatten_to_stmts(producer.body)
        if isinstance(statement, ir.AssignStmt)
    }
    returns = [
        statement for statement in ir.flatten_to_stmts(producer.body) if isinstance(statement, ir.ReturnStmt)
    ]
    assert len(returns) == 1 and len(returns[0].value) == 2
    for value in returns[0].value:
        visited = set()
        while isinstance(value, ir.Var) and value.unique_id in definitions:
            assert value.unique_id not in visited, "Returned output aliases must not contain a cycle"
            visited.add(value.unique_id)
            value = definitions[value.unique_id]
        assert value.same_as(stores[0])
    consumer = _function(after, "consumer")
    assert [load.args[0] for load in _builtin_calls(consumer, "tile.load")] == [consumer.params[1]] * 2
    assert _global_launches(_function(after, "mixed"))[1].attrs["arg_direction_overrides"] == [1]


@pytest.mark.parametrize("outer_sorts_first", [False, True])
def test_nested_spmd_group_outputs_propagate_in_call_graph_order(outer_sorts_first):
    """Wrapper output propagation follows callees before callers regardless of function-name order."""
    span = ir.Span.unknown()
    input_type = ir.TensorType([16, 16], ir.DataType.FP16)
    output_type = ir.TensorType([16, 16], ir.DataType.FP32)
    outer_name, inner_name = ("a_outer", "z_inner") if outer_sorts_first else ("z_outer", "a_inner")
    producer_x, producer_y = ir.Var("x", input_type, span), ir.Var("y", input_type, span)
    add = tensor_ops.add(producer_x, producer_y)
    fresh = ir.Var("fresh", add.type, span)
    producer = ir.Function(
        "producer",
        [producer_x, producer_y],
        [input_type],
        ir.SeqStmts([ir.AssignStmt(fresh, add, span), ir.ReturnStmt([fresh], span)], span),
        span,
        ir.FunctionType.AIV,
    )
    consumer_left, consumer_right = ir.Var("left", input_type, span), ir.Var("right", input_type, span)
    matmul = tensor_ops.matmul(consumer_left, consumer_right, out_dtype=ir.DataType.FP32)
    product = ir.Var("product", matmul.type, span)
    consumer = ir.Function(
        "consumer",
        [consumer_left, consumer_right],
        [output_type],
        ir.SeqStmts([ir.AssignStmt(product, matmul, span), ir.ReturnStmt([product], span)], span),
        span,
        ir.FunctionType.AIC,
    )
    group_x, group_y = ir.Var("x", input_type, span), ir.Var("y", input_type, span)
    group_fresh, group_product = ir.Var("fresh", input_type, span), ir.Var("product", output_type, span)
    group = ir.Function(
        inner_name,
        [group_x, group_y],
        [output_type],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    group_fresh, ir.Call(ir.GlobalVar("producer"), [group_x, group_y], input_type, span), span
                ),
                ir.AssignStmt(
                    group_product,
                    ir.Call(ir.GlobalVar("consumer"), [group_fresh, group_x], output_type, span),
                    span,
                ),
                ir.ReturnStmt([group_product], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Group,
    )

    def forwarding_function(name: str, callee: str, function_type: ir.FunctionType) -> ir.Function:
        x, y = ir.Var("x", input_type, span), ir.Var("y", input_type, span)
        result = ir.Var("result", output_type, span)
        return ir.Function(
            name,
            [x, y],
            [output_type],
            ir.SeqStmts(
                [
                    ir.AssignStmt(result, ir.Call(ir.GlobalVar(callee), [x, y], output_type, span), span),
                    ir.ReturnStmt([result], span),
                ],
                span,
            ),
            span,
            function_type,
            attrs={"core_num": 2} if function_type == ir.FunctionType.Spmd else None,
        )

    outer = forwarding_function(outer_name, inner_name, ir.FunctionType.Spmd)
    main = forwarding_function("main", outer_name, ir.FunctionType.Orchestration)
    before = ir.Program([producer, consumer, group, outer, main], "nested_group_outputs", span)
    lowered = passes.convert_tensor_to_tile_ops()(before)

    expected_x, expected_y = ir.Var("x", input_type, span), ir.Var("y", input_type, span)
    scratch, out = ir.Var("scratch", input_type, span), ir.Var("out", output_type, span)
    result = ir.Var("result", output_type, span)
    expected_outer = ir.Function(
        outer_name,
        [expected_x, expected_y, (scratch, ir.ParamDirection.Out), (out, ir.ParamDirection.Out)],
        [output_type],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    result,
                    ir.Call(
                        ir.GlobalVar(inner_name), [expected_x, expected_y, scratch, out], output_type, span
                    ),
                    span,
                ),
                ir.ReturnStmt([result], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Spmd,
        attrs={"core_num": 2},
    )
    ir.assert_structural_equal(_function(lowered, outer_name), expected_outer)
    for name in (inner_name, outer_name):
        wrapper = _function(lowered, name)
        assert len(wrapper.params) == 4
        assert not _builtin_calls(wrapper, "tensor.create")
    inner = _function(lowered, inner_name)
    inner_launches = _global_launches(inner)
    assert inner_launches[0].args[-1].same_as(inner.params[2])
    assert inner_launches[1].args[-1].same_as(inner.params[3])
    caller = _function(lowered, "main")
    assert len(_builtin_calls(caller, "tensor.create")) == 2
    outer_call = _global_launches(caller)[0]
    assert len(outer_call.args) == 4
    assert not outer_call.args[2].same_as(outer_call.args[3])
    _expand(lowered)


def test_returned_local_assemble_stages_tensor_parameter_source():
    """A returned local assemble must retain the staging path that loads its Tensor source."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[32], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
            buffer = pl.tensor.create([32], dtype=pl.FP32)
            assembled = pl.tensor.assemble(buffer, x, [0])
            return assembled

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[32], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
            result = self.kernel(x)
            return result

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32], pl.FP32], out: pl.Out[pl.Tensor[[32], pl.FP32]]
        ) -> pl.Tensor[[32], pl.FP32]:
            buffer = pl.tile.create([32], dtype=pl.FP32)
            source = pl.load(x, [0], [32], [32], target_memory=pl.Mem.Vec)
            assembled = pl.tile.assemble(buffer, source, [0])
            stored = pl.store(assembled, [0], out)
            return stored

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[32], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
            out = pl.tensor.create([32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            result = self.kernel(x, out)
            return result

    lowered = _lower(Before)
    ir.assert_structural_equal(lowered, Expected)
    kernel = _function(lowered, "kernel")
    assert kernel.func_type == ir.FunctionType.InCore
    assert len(kernel.params) == 2
    assert not _all_builtin_calls(kernel, "tensor.assemble")
    assert not _all_builtin_calls(kernel, "tensor.create")
    assert len(_all_builtin_calls(kernel, "tile.create")) == 1
    assert len(_all_builtin_calls(kernel, "tile.assemble")) == 1
    stores = _all_builtin_calls(kernel, "tile.store")
    assert len(stores) == 1
    assert stores[0].args[2].same_as(kernel.params[-1])
    assert len(_builtin_calls(_function(lowered, "main"), "tensor.create")) == 1


@pytest.mark.parametrize("write_through_alias", [False, True])
def test_scalar_updated_local_assemble_retains_tile_staging(write_through_alias):
    """An unassigned scalar write, including through an alias, stays on-chip before one exit store."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def direct(
            self,
            x: pl.Tensor[[32], pl.FP32],
            y: pl.Tensor[[32], pl.FP32],
            index: pl.Scalar[pl.INDEX],
            value: pl.Scalar[pl.FP32],
        ) -> pl.Tensor[[32], pl.FP32]:
            buffer = pl.tensor.create([32], dtype=pl.FP32)
            computed = pl.add(x, y)
            buffer = pl.tensor.assemble(buffer, computed, [0])
            pl.tensor.write(buffer, [index], value)
            return buffer

        @pl.function(type=pl.FunctionType.InCore)
        def alias(
            self,
            x: pl.Tensor[[32], pl.FP32],
            y: pl.Tensor[[32], pl.FP32],
            index: pl.Scalar[pl.INDEX],
            value: pl.Scalar[pl.FP32],
        ) -> pl.Tensor[[32], pl.FP32]:
            buffer = pl.tensor.create([32], dtype=pl.FP32)
            computed = pl.add(x, y)
            buffer = pl.tensor.assemble(buffer, computed, [0])
            target = buffer
            pl.tensor.write(target, [index], value)
            return buffer

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def direct(
            self,
            x: pl.Tensor[[32], pl.FP32],
            y: pl.Tensor[[32], pl.FP32],
            index: pl.Scalar[pl.INDEX],
            value: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[32], pl.FP32]],
        ) -> pl.Tensor[[32], pl.FP32]:
            buffer = pl.tile.create([32], dtype=pl.FP32)
            x_tile = pl.load(x, [0], [32], [32])
            y_tile = pl.load(y, [0], [32], [32])
            computed = pl.tile.add(x_tile, y_tile)
            assembled = pl.tile.assemble(buffer, computed, [0])
            pl.tile.write(assembled, [index], value)
            stored = pl.store(assembled, [0], out)
            return stored

        @pl.function(type=pl.FunctionType.InCore)
        def alias(
            self,
            x: pl.Tensor[[32], pl.FP32],
            y: pl.Tensor[[32], pl.FP32],
            index: pl.Scalar[pl.INDEX],
            value: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[32], pl.FP32]],
        ) -> pl.Tensor[[32], pl.FP32]:
            buffer = pl.tile.create([32], dtype=pl.FP32)
            x_tile = pl.load(x, [0], [32], [32])
            y_tile = pl.load(y, [0], [32], [32])
            computed = pl.tile.add(x_tile, y_tile)
            assembled = pl.tile.assemble(buffer, computed, [0])
            target = assembled
            pl.tile.write(target, [index], value)
            stored = pl.store(assembled, [0], out)
            return stored

    name = "alias" if write_through_alias else "direct"
    lowered = _lower(_select(Before, (name,)))
    ir.assert_structural_equal(_function(lowered, name), _function(Expected, name))
    kernel = _function(lowered, name)
    assert len(kernel.params) == 5
    assert kernel.param_directions[-1] == ir.ParamDirection.Out
    assert not _all_builtin_calls(kernel, "tensor.create")
    assert not _all_builtin_calls(kernel, "tensor.assemble")
    assert not _all_builtin_calls(kernel, "tensor.write")
    assert len(_all_builtin_calls(kernel, "tile.create")) == 1
    assert len(_all_builtin_calls(kernel, "tile.assemble")) == 1
    writes, stores = _all_builtin_calls(kernel, "tile.write"), _all_builtin_calls(kernel, "tile.store")
    assert len(writes) == len(stores) == 1
    assert isinstance(writes[0].args[0].type, ir.TileType)
    assert stores[0].args[2].same_as(kernel.params[-1])


def test_scalar_updated_tensor_source_assemble_reaches_vector_expansion():
    """The supported 2D staging flow reaches Expand with its scalar write and one final GM store."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[1, 32], pl.FP32],
            index: pl.Scalar[pl.INDEX],
            value: pl.Scalar[pl.FP32],
        ) -> pl.Tensor[[1, 32], pl.FP32]:
            buffer = pl.tensor.create([1, 32], dtype=pl.FP32)
            assembled = pl.tensor.assemble(buffer, x, [0, 0])
            alias = assembled
            pl.tensor.write(alias, [0, index], value)
            return assembled

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[1, 32], pl.FP32],
            index: pl.Scalar[pl.INDEX],
            value: pl.Scalar[pl.FP32],
        ) -> pl.Tensor[[1, 32], pl.FP32]:
            result = self.kernel(x, index, value)
            return result

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[1, 32], pl.FP32],
            index: pl.Scalar[pl.INDEX],
            value: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
        ) -> pl.Tensor[[1, 32], pl.FP32]:
            buffer = pl.tile.create([1, 32], dtype=pl.FP32)
            source = pl.load(x, [0, 0], [1, 32], [1, 32], target_memory=pl.Mem.Vec)
            assembled = pl.tile.assemble(buffer, source, [0, 0])
            alias = assembled
            pl.tile.write(alias, [0, index], value)
            stored = pl.store(assembled, [0, 0], out)
            return stored

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[1, 32], pl.FP32],
            index: pl.Scalar[pl.INDEX],
            value: pl.Scalar[pl.FP32],
        ) -> pl.Tensor[[1, 32], pl.FP32]:
            out = pl.tensor.create([1, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            result = self.kernel(x, index, value, out)
            return result

    lowered = _lower(Before)
    ir.assert_structural_equal(lowered, Expected)
    after = _expand(passes.flatten_tile_nd_to_2d()(lowered))
    kernel = _function(after, "kernel")
    assert kernel.func_type == ir.FunctionType.AIV
    assert kernel.param_directions == [ir.ParamDirection.In] * 3 + [ir.ParamDirection.Out]
    for name in ("tensor.assemble", "tensor.write", "tensor.create"):
        assert not _all_builtin_calls(kernel, name)
    assert len(_all_builtin_calls(kernel, "tile.assemble")) == 1
    writes, stores = _all_builtin_calls(kernel, "tile.write"), _all_builtin_calls(kernel, "tile.store")
    assert len(writes) == len(stores) == 1
    assert isinstance(writes[0].args[0].type, ir.TileType)
    assert stores[0].args[2].same_as(kernel.params[-1])
    assert len(_builtin_calls(_function(after, "main"), "tensor.create")) == 1


def test_scalar_write_through_if_join_retains_both_local_tiles():
    """A phi that may select either allocation requires staging for both roots, including the return."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, condition: pl.Scalar[pl.BOOL], value: pl.Scalar[pl.FP32]
        ) -> pl.Tensor[[1, 32], pl.FP32]:
            first = pl.tensor.create([1, 32], dtype=pl.FP32)
            second = pl.tensor.create([1, 32], dtype=pl.FP32)
            if condition:
                selected = pl.yield_(first)
            else:
                selected = pl.yield_(second)
            pl.tensor.write(selected, [0, 0], value)
            return first

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            condition: pl.Scalar[pl.BOOL],
            value: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
        ) -> pl.Tensor[[1, 32], pl.FP32]:
            first = pl.tile.create([1, 32], dtype=pl.FP32)
            second = pl.tile.create([1, 32], dtype=pl.FP32)
            if condition:
                selected = pl.yield_(first)
            else:
                selected = pl.yield_(second)
            pl.tile.write(selected, [0, 0], value)
            stored = pl.store(first, [0, 0], out)
            return stored

    lowered = _lower(Before)
    ir.assert_structural_equal(lowered, Expected)
    after = _expand(lowered)
    kernel = _function(after, "kernel")
    assert len(_all_builtin_calls(kernel, "tile.create")) == 2
    assert len(_all_builtin_calls(kernel, "tile.write")) == 1
    assert len(_all_builtin_calls(kernel, "tile.store")) == 1
    assert not _all_builtin_calls(kernel, "tensor.write")


def test_if_join_staging_does_not_block_independent_explicit_gm_output():
    """Ambiguous alias uses must not force an unrelated explicit GM allocation onto the tile path."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[1, 32], pl.FP32],
            condition: pl.Scalar[pl.BOOL],
            value: pl.Scalar[pl.FP32],
        ) -> tuple[pl.Tensor[[1, 32], pl.FP32], pl.Tensor[[1, 32], pl.FP32]]:
            first = pl.tensor.create([1, 32], dtype=pl.FP32)
            second = pl.tensor.create([1, 32], dtype=pl.FP32)
            if condition:
                selected = pl.yield_(first)
            else:
                selected = pl.yield_(second)
            pl.tensor.write(selected, [0, 0], value)
            independent = pl.tensor.create([1, 32], dtype=pl.FP32)
            source = pl.load(x, [0, 0], [1, 32])
            stored = pl.store(source, [0, 0], independent)
            return first, stored

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[1, 32], pl.FP32],
            condition: pl.Scalar[pl.BOOL],
            value: pl.Scalar[pl.FP32],
            independent: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
            first_out: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
        ) -> tuple[pl.Tensor[[1, 32], pl.FP32], pl.Tensor[[1, 32], pl.FP32]]:
            first = pl.tile.create([1, 32], dtype=pl.FP32)
            second = pl.tile.create([1, 32], dtype=pl.FP32)
            if condition:
                selected = pl.yield_(first)
            else:
                selected = pl.yield_(second)
            pl.tile.write(selected, [0, 0], value)
            source = pl.load(x, [0, 0], [1, 32])
            stored = pl.store(source, [0, 0], independent)
            first_stored = pl.store(first, [0, 0], first_out)
            return first_stored, stored

    lowered = _lower(Before)
    ir.assert_structural_equal(lowered, Expected)
    after = _expand(lowered)
    kernel = _function(after, "kernel")
    assert len(kernel.params) == 5
    assert len(_all_builtin_calls(kernel, "tile.create")) == 2
    assert not _all_builtin_calls(kernel, "tensor.create")
    stores = _all_builtin_calls(kernel, "tile.store")
    assert len(stores) == 2
    assert [store.args[2] for store in stores] == list(kernel.params[-2:])
    assert all(isinstance(store.args[2].type, ir.TensorType) for store in stores)


@pytest.mark.parametrize("kernel_name", ["incore", "aiv"])
def test_unannotated_computed_tensor_return_infers_output_storage(kernel_name):
    """A nonempty return without an annotation still gains one inferred Tensor Out and exit store."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def incore(self, x: pl.Tensor[[32], pl.FP32], y: pl.Tensor[[32], pl.FP32]):
            fresh = pl.add(x, y)
            return fresh

        @pl.function(type=pl.FunctionType.AIV)
        def aiv(self, x: pl.Tensor[[32], pl.FP32], y: pl.Tensor[[32], pl.FP32]):
            fresh = pl.add(x, y)
            return fresh

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def incore(
            self,
            x: pl.Tensor[[32], pl.FP32],
            y: pl.Tensor[[32], pl.FP32],
            out: pl.Out[pl.Tensor[[32], pl.FP32]],
        ) -> pl.Tensor[[32], pl.FP32]:
            x_tile = pl.load(x, [0], [32], [32])
            y_tile = pl.load(y, [0], [32], [32])
            fresh = pl.tile.add(x_tile, y_tile)
            stored = pl.store(fresh, [0], out)
            return stored

        @pl.function(type=pl.FunctionType.AIV)
        def aiv(
            self,
            x: pl.Tensor[[32], pl.FP32],
            y: pl.Tensor[[32], pl.FP32],
            out: pl.Out[pl.Tensor[[32], pl.FP32]],
        ) -> pl.Tensor[[32], pl.FP32]:
            x_tile = pl.load(x, [0], [32], [32])
            y_tile = pl.load(y, [0], [32], [32])
            fresh = pl.tile.add(x_tile, y_tile)
            stored = pl.store(fresh, [0], out)
            return stored

    before = _select(Before, (kernel_name,))
    assert not _function(before, kernel_name).return_types
    lowered = _lower(before)
    kernel = _function(lowered, kernel_name)
    ir.assert_structural_equal(kernel, _function(Expected, kernel_name))
    assert len(kernel.return_types) == 1
    assert isinstance(kernel.return_types[0], ir.TensorType)
    assert kernel.param_directions == [ir.ParamDirection.In, ir.ParamDirection.In, ir.ParamDirection.Out]
    assert len(_all_builtin_calls(kernel, "tile.add")) == 1
    stores = _all_builtin_calls(kernel, "tile.store")
    assert len(stores) == 1 and stores[0].args[2].same_as(kernel.params[-1])
    assert _function(_expand(lowered), kernel_name).func_type == ir.FunctionType.AIV


@pytest.mark.parametrize("function_type", [ir.FunctionType.AIC, ir.FunctionType.AIV])
def test_tensor_conversion_preserves_already_lowered_buffer_functions(function_type):
    """CTT must skip final Buffer bodies instead of querying functional effects for buffer.alloc."""
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([16, 32], ir.DataType.FP32)
    descriptor = ir.BufferType([16, 32], ir.DataType.FP32, ir.MemorySpace.Vec)
    source, output = ir.Var("source", tensor_type, span), ir.Var("output", tensor_type, span)
    buffer = ir.Var("buffer", descriptor, span)
    offsets = ir.MakeTuple([ir.ConstInt(0, ir.DataType.INDEX, span)] * 2, span)
    valid = ir.MakeTuple(
        [ir.ConstInt(16, ir.DataType.INDEX, span), ir.ConstInt(32, ir.DataType.INDEX, span)], span
    )
    allocation = ir.Call(ir.get_op("buffer.alloc"), [ir.MakeTuple([], span)], {}, None, descriptor, span)
    load = ir.Call(ir.get_op("buffer.load"), [source, offsets, valid, buffer], {}, None, ir.VoidType(), span)
    store = ir.Call(
        ir.get_op("buffer.store"), [buffer, offsets, valid, output], {}, None, ir.VoidType(), span
    )
    kernel = ir.Function(
        "kernel",
        [source, (output, ir.ParamDirection.Out)],
        [tensor_type],
        ir.SeqStmts(
            [
                ir.AssignStmt(buffer, allocation, span),
                ir.EvalStmt(load, span),
                ir.EvalStmt(store, span),
                ir.ReturnStmt([output], span),
            ],
            span,
        ),
        span,
        type=function_type,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    before = ir.Program([kernel], "already_lowered_buffer", span)
    after = passes.convert_tensor_to_tile_ops()(before)
    ir.assert_structural_equal(after, before)
    preserved = _function(after, "kernel")
    assert preserved.same_as(kernel)
    assert preserved.ir_stage == ir.FunctionIRStage.Buffer
    assert preserved.func_type == function_type
    for name in ("buffer.alloc", "buffer.load", "buffer.store"):
        assert len(_all_builtin_calls(preserved, name)) == 1


def _unannotated_dynamic_output_program(function_type, *, local_extent, gm_storage):
    """Exercise both inferred exit stores and pre-lifted GM outputs at a real call site."""
    span = ir.Span.unknown()
    index_type = ir.ScalarType(ir.DataType.INDEX)
    n = ir.Var("n", index_type, span)
    source = ir.Var("source", ir.TensorType([32, 16], ir.DataType.FP32), span)
    statements: list[ir.Stmt] = []
    extent = n
    if local_extent:
        extent = ir.Var("local_extent", index_type, span)
        # Even pure arithmetic is body-local until explicitly reconstructed in
        # the caller. Copying this Var into an Out type would violate SSA scope.
        statements.append(
            ir.AssignStmt(
                extent, ir.Add(n, ir.ConstInt(0, ir.DataType.INDEX, span), ir.DataType.INDEX, span), span
            )
        )
    sliced = tensor_ops.slice(source, [extent, 16], [0, 0])
    result = ir.Var("sliced", sliced.type, span)
    statements.append(ir.AssignStmt(result, sliced, span))
    if gm_storage:
        allocation = tensor_ops.create([extent, 16], ir.DataType.FP32)
        buffer = ir.Var("buffer", allocation.type, span)
        tile = tile_ops.load(source, [0, 0], [extent, 16])
        loaded = ir.Var("loaded", tile.type, span)
        store = tile_ops.store(loaded, [0, 0], buffer)
        result = ir.Var("stored", store.type, span)
        statements.extend(
            [
                ir.AssignStmt(buffer, allocation, span),
                ir.AssignStmt(loaded, tile, span),
                ir.AssignStmt(result, store, span),
            ]
        )
    statements.append(ir.ReturnStmt([result], span))
    kernel = ir.Function("kernel", [n, source], [], ir.SeqStmts(statements, span), span, function_type)
    caller_n = ir.Var("caller_n", index_type, span)
    caller_source = ir.Var("caller_source", source.type, span)
    caller_type = ir.TensorType([caller_n, 16], ir.DataType.FP32)
    caller_result = ir.Var("result", caller_type, span)
    caller = ir.Function(
        "main",
        [caller_n, caller_source],
        [caller_type],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    caller_result,
                    ir.Call(ir.GlobalVar("kernel"), [caller_n, caller_source], caller_type, span),
                    span,
                ),
                ir.ReturnStmt([caller_result], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Orchestration,
    )
    return ir.Program([kernel, caller], "unannotated_dynamic_output", span)


@pytest.mark.parametrize("function_type", [ir.FunctionType.InCore, ir.FunctionType.AIC, ir.FunctionType.AIV])
@pytest.mark.parametrize("gm_storage", [False, True], ids=["exit_store", "gm_allocation"])
def test_inferred_output_rejects_body_local_dimensions(function_type, gm_storage):
    """Reject invalid signatures before a local extent can escape into caller tensor.create."""
    before = _unannotated_dynamic_output_program(function_type, local_extent=True, gm_storage=gm_storage)
    with pytest.raises(ValueError, match="output type depends on body-local dimension 'local_extent'"):
        passes.convert_tensor_to_tile_ops()(before)


@pytest.mark.parametrize("function_type", [ir.FunctionType.InCore, ir.FunctionType.AIC, ir.FunctionType.AIV])
@pytest.mark.parametrize("gm_storage", [False, True], ids=["exit_store", "gm_allocation"])
def test_inferred_output_instantiates_scalar_parameter_dimensions(function_type, gm_storage):
    """A signature-visible extent still maps to the caller's own scalar parameter."""
    before = _unannotated_dynamic_output_program(function_type, local_extent=False, gm_storage=gm_storage)
    after = passes.convert_tensor_to_tile_ops()(before)
    kernel = _function(after, "kernel")
    output_type = kernel.params[-1].type
    assert isinstance(output_type, ir.TensorType)
    assert output_type.shape[0].same_as(kernel.params[0])
    caller = _function(after, "main")
    allocations = _builtin_calls(caller, "tensor.create")
    assert len(allocations) == 1
    allocated_type = allocations[0].type
    assert isinstance(allocated_type, ir.TensorType)
    assert allocated_type.shape[0].same_as(caller.params[0])
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.SSAForm)
    properties.insert(passes.IRProperty.UseAfterDef)
    passes.run_verifier(properties)(after)


def test_dynamic_computed_outputs_bind_to_the_adapters_scalar_parameter():
    """Remapping dynamic tensor types must retain the final canonical scalar's identity."""
    span = ir.Span.unknown()
    index_type = ir.ScalarType(ir.DataType.INDEX)
    rhs_type = ir.TensorType([16, 16], ir.DataType.FP16)

    def input_vars():
        n = ir.Var("n", index_type, span)
        tensor_type = ir.TensorType([n, 16], ir.DataType.FP16)
        return n, ir.Var("x", tensor_type, span), ir.Var("y", tensor_type, span)

    producer_n, producer_x, producer_y = input_vars()
    add = tensor_ops.add(producer_x, producer_y)
    fresh = ir.Var("fresh", add.type, span)
    producer = ir.Function(
        "producer",
        [producer_n, producer_x, producer_y],
        [producer_x.type],
        ir.SeqStmts([ir.AssignStmt(fresh, add, span), ir.ReturnStmt([fresh], span)], span),
        span,
        ir.FunctionType.AIV,
    )
    consumer_n = ir.Var("n", index_type, span)
    consumer_x = ir.Var("fresh", ir.TensorType([consumer_n, 16], ir.DataType.FP16), span)
    consumer_rhs = ir.Var("rhs", rhs_type, span)
    matmul = tensor_ops.matmul(consumer_x, consumer_rhs, out_dtype=ir.DataType.FP32)
    product = ir.Var("product", matmul.type, span)
    consumer = ir.Function(
        "consumer",
        [consumer_n, consumer_x, consumer_rhs],
        [matmul.type],
        ir.SeqStmts([ir.AssignStmt(product, matmul, span), ir.ReturnStmt([product], span)], span),
        span,
        ir.FunctionType.AIC,
    )
    group_n, group_x, group_y = input_vars()
    group_rhs = ir.Var("rhs", rhs_type, span)
    group_output_type = ir.TensorType([group_n, 16], ir.DataType.FP32)
    group_fresh = ir.Var("fresh", group_x.type, span)
    group_product = ir.Var("product", group_output_type, span)
    group = ir.Function(
        "mixed",
        [group_n, group_x, group_y, group_rhs],
        [group_output_type],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    group_fresh,
                    ir.Call(ir.GlobalVar("producer"), [group_n, group_x, group_y], group_x.type, span),
                    span,
                ),
                ir.AssignStmt(
                    group_product,
                    ir.Call(
                        ir.GlobalVar("consumer"),
                        [group_n, group_fresh, group_rhs],
                        {},
                        {"arg_direction_overrides": [1]},
                        group_output_type,
                        span,
                    ),
                    span,
                ),
                ir.ReturnStmt([group_product], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Group,
    )
    main_n, main_x, main_y = input_vars()
    main_rhs = ir.Var("rhs", rhs_type, span)
    main_output_type = ir.TensorType([main_n, 16], ir.DataType.FP32)
    main_product = ir.Var("product", main_output_type, span)
    main = ir.Function(
        "main",
        [main_n, main_x, main_y, main_rhs],
        [main_output_type],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    main_product,
                    ir.Call(
                        ir.GlobalVar("mixed"), [main_n, main_x, main_y, main_rhs], main_output_type, span
                    ),
                    span,
                ),
                ir.ReturnStmt([main_product], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Orchestration,
    )
    # Already-SSA input isolates adapter remapping from ConvertToSSA's separate
    # handling of scalar references inside a function's return annotations.
    before = ir.Program([producer, consumer, group, main], "dynamic_group_output_bindings", span)
    lowered = passes.convert_tensor_to_tile_ops()(before)
    assert len(_builtin_calls(_function(lowered, "main"), "tensor.create")) == 2
    after = _expand(lowered)
    for name in ("mixed", "producer", "consumer"):
        function = _function(after, name)
        assert len(function.params) == 6
        for index in (1, 2, 4, 5):
            tensor_type = function.params[index].type
            assert isinstance(tensor_type, ir.TensorType)
            assert tensor_type.shape[0].same_as(function.params[0])
        returned_type = function.return_types[-1]
        assert isinstance(returned_type, ir.TensorType)
        assert returned_type.shape[0].same_as(function.params[0])
    group = _function(after, "mixed")
    assert _global_launches(group)[1].attrs["arg_direction_overrides"] == [4]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

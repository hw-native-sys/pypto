# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression coverage for empty dynamic SPMD launches and escaping SSA values."""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes


def _program():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
            self,
            a: pl.Out[pl.Tensor[[16], pl.FP32]],
            b: pl.Out[pl.Tensor[[16], pl.FP32]],
        ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
            ta = pl.tile.full([16], pl.FP32, 3.0)
            tb = pl.tile.full([16], pl.FP32, 7.0)
            a = pl.store(ta, [0], a)
            b = pl.store(tb, [0], b)
            return b, a

        @pl.function(type=pl.FunctionType.InCore)
        def consumer(
            self, a: pl.Tensor[[16], pl.FP32], b: pl.Tensor[[16], pl.FP32]
        ) -> pl.Tensor[[16], pl.FP32]:
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[16], pl.FP32],
            b: pl.Tensor[[16], pl.FP32],
            n: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16], pl.FP32]:
            with pl.manual_scope():
                prior = pl.system.task_dummy(deps=[])
                (ob, oa), tid = pl.spmd_submit(
                    self.producer,
                    a,
                    b,
                    core_num=n,
                    deps=[prior],
                    sync_start=True,
                    allow_early_resolve=True,
                )
                result, _ = pl.submit(self.consumer, oa, ob, deps=[tid])
            return result

    return Program


def _prepare(program):
    program = passes.convert_to_ssa()(program)
    program = passes.normalize_stmt_structure()(program)
    program = passes.flatten_call_expr()(program)
    return passes.normalize_return_order()(program)


def _nodes(program, kind):
    found = []

    class Collector(ir.IRVisitor):
        def visit_if_stmt(self, op):
            if isinstance(op, kind):
                found.append(op)
            super().visit_if_stmt(op)

        def visit_return_stmt(self, op):
            if isinstance(op, kind):
                found.append(op)
            super().visit_return_stmt(op)

        def visit_yield_stmt(self, op):
            if isinstance(op, kind):
                found.append(op)
            super().visit_yield_stmt(op)

        def visit_call(self, op):
            if isinstance(op, kind):
                found.append(op)
            super().visit_call(op)

        def visit_submit(self, op):
            if isinstance(op, kind):
                found.append(op)
            super().visit_submit(op)

    Collector().visit_program(program)
    return found


def _verify(program):
    properties = passes.IRPropertySet()
    for prop in [
        passes.IRProperty.SSAForm,
        passes.IRProperty.UseAfterDef,
        passes.IRProperty.AssignTypeSymmetry,
        passes.IRProperty.NoNestedCalls,
    ]:
        properties.insert(prop)
    passes.run_verifier(properties)(program)


class RewriteLaunch(ir.IRMutator):
    def __init__(self, *, supplied=2, count=None):
        super().__init__()
        self.supplied = supplied
        self.count = count

    def visit_submit(self, op):
        if op.core_num is None:
            return super().visit_submit(op)
        return ir.Submit(
            op.op,
            list(op.args)[: self.supplied],
            list(op.deps),
            kwargs=op.kwargs,
            attrs=op.attrs,
            type=op.type,
            span=op.span,
            core_num=self.count if self.count is not None else op.core_num,
            sync_start=op.sync_start,
            allow_early_resolve=op.allow_early_resolve,
            predicate=op.predicate,
        )


def test_multi_output_no_input_preserves_mapping_and_task_id():
    before = _prepare(_program())
    original = next(s for s in _nodes(before, ir.Submit) if s.core_num is not None)
    callee = before.get_function(original.op.name)
    assert callee is not None
    returned = _nodes(ir.Program([callee], "callee", callee.span), ir.ReturnStmt)[-1]
    expected = [
        original.args[next(i for i, p in enumerate(callee.params) if p.same_as(v))] for v in returned.value
    ]
    after = passes.legalize_spmd_launches()(before)
    _verify(after)
    guard = _nodes(after, ir.IfStmt)[0]
    yields = _nodes(after, ir.YieldStmt)
    assert len(guard.return_vars) == 3
    empty = yields[-1]
    assert all(actual.same_as(wanted) for actual, wanted in zip(empty.value[:2], expected))
    assert guard.return_vars[-1].type.dtype == DataType.TASK_ID
    launch = next(s for s in _nodes(after, ir.Submit) if s.core_num is not None)
    assert launch.sync_start and launch.allow_early_resolve
    dummy = [c for c in _nodes(after, ir.Call) if c.op.name == ir.get_op("system.task_dummy").name][-1]
    assert dummy.attrs["manual_dep_edges"][0].same_as(original.deps[0])
    # Running it again under its own n > 0 constraint must not nest guards.
    twice = passes.legalize_spmd_launches()(after)
    ir.assert_structural_equal(after, twice)


@pytest.mark.parametrize("supplied,missing", [(0, ["a", "b"]), (1, ["b"])])
def test_missing_outputs_name_every_unallocated_tensor(supplied, missing):
    program = RewriteLaunch(supplied=supplied).visit_program(_prepare(_program()))
    with pytest.raises(ValueError, match="not preallocated") as error:
        passes.legalize_spmd_launches()(program)
    for name in missing:
        assert f"'{name}__ssa_v0' (parameter" in str(error.value)


def test_positive_constant_does_not_restrict_runtime_allocated_outputs():
    program = RewriteLaunch(
        supplied=0, count=ir.ConstInt(4, DataType.INDEX, ir.Span.unknown())
    ).visit_program(_prepare(_program()))
    after = passes.legalize_spmd_launches()(program)
    assert not _nodes(after, ir.IfStmt)


def test_nonpositive_constant_preserves_outputs_after_simplify():
    program = RewriteLaunch(count=ir.ConstInt(0, DataType.INDEX, ir.Span.unknown())).visit_program(
        _prepare(_program())
    )
    after = passes.simplify()(passes.legalize_spmd_launches()(program))
    _verify(after)
    assert not [s for s in _nodes(after, ir.Submit) if s.core_num is not None]
    assert any(c.op.name == ir.get_op("system.task_dummy").name for c in _nodes(after, ir.Call))


def test_existing_positive_guard_allows_omitted_output():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, out: pl.Out[pl.Tensor[[16], pl.FP32]]) -> pl.Tensor[[16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, n: pl.Scalar[pl.INDEX]):
            with pl.manual_scope():
                if n > 0:
                    out, tid = pl.spmd_submit(self.kernel, core_num=n)

    after = passes.legalize_spmd_launches()(_prepare(Program))
    assert len(_nodes(after, ir.IfStmt)) == 1


def test_arith_positive_expression_and_loop_range_need_no_guard():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, out: pl.Out[pl.Tensor[[16], pl.FP32]]) -> pl.Tensor[[16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self):
            with pl.manual_scope():
                for i in pl.range(0, 4):
                    n = i + 1
                    out, tid = pl.spmd_submit(self.kernel, core_num=n)

    after = passes.legalize_spmd_launches()(_prepare(Program))
    assert not _nodes(after, ir.IfStmt)


def test_unreturned_out_must_also_be_preallocated():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, a: pl.Out[pl.Tensor[[16], pl.FP32]], hidden: pl.Out[pl.Tensor[[16], pl.FP32]]
        ) -> pl.Tensor[[16], pl.FP32]:
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, a: pl.Tensor[[16], pl.FP32], n: pl.Scalar[pl.INDEX]):
            with pl.manual_scope():
                out, tid = pl.spmd_submit(self.kernel, a, core_num=n)

    with pytest.raises(ValueError, match="hidden.*parameter 1"):
        passes.legalize_spmd_launches()(_prepare(Program))


def test_positive_then_constraint_does_not_leak_into_else():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, out: pl.Out[pl.Tensor[[16], pl.FP32]]) -> pl.Tensor[[16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, n: pl.Scalar[pl.INDEX]):
            with pl.manual_scope():
                if n > 0:
                    a, ta = pl.spmd_submit(self.kernel, core_num=n)
                else:
                    b, tb = pl.spmd_submit(self.kernel, core_num=n)

    with pytest.raises(ValueError, match="not preallocated"):
        passes.legalize_spmd_launches()(_prepare(Program))


def test_auto_dependencies_use_outer_task_id_for_original_buffer_consumers():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def write(self, a: pl.Out[pl.Tensor[[16], pl.FP32]]) -> pl.Tensor[[16], pl.FP32]:
            return a

        @pl.function(type=pl.FunctionType.InCore)
        def read(self, a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, a: pl.Tensor[[16], pl.FP32], n: pl.Scalar[pl.INDEX]):
            before, prior = pl.submit(self.write, a)
            changed, tid = pl.spmd_submit(self.write, a, core_num=n, deps=[prior])
            result, final_tid = pl.submit(self.read, a)

    program = passes.legalize_spmd_launches()(_prepare(Program))
    program = passes.derive_call_directions()(program)
    program = passes.auto_derive_task_dependencies(analyze_auto_scopes=True)(program)
    _verify(program)
    guard = _nodes(program, ir.IfStmt)[0]
    consumer = [c for c in _nodes(program, ir.Submit) if c.op.name == "read"][0]
    edges = consumer.attrs.get("compiler_manual_dep_edges", [])
    assert any(edge.same_as(guard.return_vars[-1]) for edge in edges)
    assert len(edges) >= 2  # Earlier producer still matters when the launch is skipped.


def test_composite_scalar_binding_keeps_existing_positive_proof():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, out: pl.Out[pl.Tensor[[16], pl.FP32]]) -> pl.Tensor[[16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, size: pl.Scalar[pl.INDEX]):
            n = size * 3 // 4
            with pl.manual_scope():
                if n > 0:
                    out, tid = pl.spmd_submit(self.kernel, core_num=n)

    after = passes.legalize_spmd_launches()(_prepare(Program))
    assert len(_nodes(after, ir.IfStmt)) == 1


@pytest.mark.parametrize("count", [None, 0, -1])
def test_omitted_outputs_before_required_inputs_are_rejected(count):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Out[pl.Tensor[[16], pl.FP32]],
            x: pl.Tensor[[16], pl.FP32],
            b: pl.Out[pl.Tensor[[16], pl.FP32]],
            y: pl.Tensor[[16], pl.FP32],
        ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
            return a, b

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[16], pl.FP32],
            y: pl.Tensor[[16], pl.FP32],
            n: pl.Scalar[pl.INDEX],
        ):
            with pl.manual_scope():
                (a, b), tid = pl.spmd_submit(self.kernel, x, y, core_num=n)

    program = _prepare(Program)
    if count is not None:
        program = RewriteLaunch(count=ir.ConstInt(count, DataType.INDEX, ir.Span.unknown())).visit_program(
            program
        )
    with pytest.raises(ValueError, match="not preallocated") as error:
        passes.legalize_spmd_launches()(program)
    assert "'a__ssa_v0' (parameter 0)" in str(error.value)
    assert "'b__ssa_v0' (parameter 2)" in str(error.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

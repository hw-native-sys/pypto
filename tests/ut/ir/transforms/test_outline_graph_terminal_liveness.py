# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Terminal Graph scopes have known-empty liveness, not all-local-def outputs."""

import pypto
import pypto.language as pl
import pytest
from pypto import ir, passes


def _graph(program):
    functions = [f for f in program.functions.values() if f.func_type == ir.FunctionType.Graph]
    assert len(functions) == 1
    return functions[0]


@pytest.mark.parametrize("direct_body", [False, True])
def test_terminal_void_graph_keeps_external_store_not_local_storage(direct_body):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, src: pl.Tensor[[16], pl.FP32], dst: pl.InOut[pl.Tensor[[16], pl.FP32]]):
            with pl.graph("terminal"):
                scratch = pl.create_tensor([16], dtype=pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP):
                    value = pl.load(src, [0], [16])
                    scratch = pl.store(value, [0], scratch)
                    dst = pl.store(value, [0], dst)

    before = passes.convert_to_ssa()(Before)
    main = before.get_function("main")
    assert main is not None
    body = main.body
    if isinstance(body, ir.SeqStmts):
        assert len(body.stmts) == 1
        body = body.stmts[0]
    assert isinstance(body, ir.GraphScopeStmt)
    if not direct_body:
        # A legal SeqStmts needs more than one child; its terminal Graph still
        # has a known-empty tail. The prefix is outside the capture.
        prefix = ir.Var("prefix", ir.ScalarType(pypto.DataType.INDEX), main.span)
        body = ir.SeqStmts(
            [ir.AssignStmt(prefix, ir.ConstInt(0, pypto.DataType.INDEX, main.span), main.span), body],
            main.span,
        )
    replaced = ir.Function(
        main.name,
        list(zip(main.params, main.param_directions, strict=True)),
        main.return_types,
        body,
        main.span,
        type=main.func_type,
        level=main.level,
        role=main.role,
        attrs=main.attrs,
    )
    before = ir.Program([replaced], before.name, before.span)
    after = passes.outline_graph_scopes()(before)
    graph = _graph(after)
    # Only caller-owned dst is exported. Neither scratch nor its local Tile escapes.
    assert len(graph.return_types) == 1
    assert isinstance(graph.return_types[0], ir.TensorType)
    assert all(not p.name_hint.startswith("scratch") for p in graph.params)
    assert len(graph.params) == 2
    directions = {p.name_hint: d for p, d in zip(graph.params, graph.param_directions, strict=True)}
    assert next(d for n, d in directions.items() if n.startswith("dst")) != ir.ParamDirection.In


def test_real_tensor_tail_use_survives_without_dead_locals():
    @pl.program
    class Before:
        @pl.function
        def main(
            self, src: pl.Tensor[[16], pl.FP32], dst: pl.InOut[pl.Tensor[[16], pl.FP32]]
        ) -> pl.Tensor[[16], pl.FP32]:
            with pl.graph("used"):
                _unused = pl.create_tensor([16], dtype=pl.FP32)
                value = pl.load(src, [0], [16])
                dst = pl.store(value, [0], dst)
            return dst

    after = passes.outline_graph_scopes()(passes.convert_to_ssa()(Before))
    graph = _graph(after)
    assert len(graph.return_types) == 1
    assert isinstance(graph.return_types[0], ir.TensorType)
    assert all(not p.name_hint.startswith("_unused") for p in graph.params)


def test_real_tile_tail_is_not_silently_dropped_or_legalized():
    @pl.program
    class Before:
        @pl.function
        def main(self, src: pl.Tensor[[16], pl.FP32]) -> pl.Tile[[16], pl.FP32]:
            with pl.graph("bad_tile_return"):
                value = pl.load(src, [0], [16])
            return value

    after = passes.outline_graph_scopes()(passes.convert_to_ssa()(Before))
    assert any(isinstance(t, ir.TileType) for t in _graph(after).return_types)
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        directed = passes.derive_call_directions()(after)
        with pytest.raises(ValueError, match="returns a value it computed|computed rather than"):
            passes.legalize_graph_boundary()(directed)


def test_terminal_graph_inside_if_keeps_only_external_effect():
    @pl.program
    class Before:
        @pl.function
        def main(self, flag: pl.Scalar[pl.INT32], dst: pl.InOut[pl.Tensor[[16], pl.FP32]]):
            if flag > 0:
                with pl.graph("conditional"):
                    scratch = pl.create_tensor([16], dtype=pl.FP32)
                    with pl.at(level=pl.Level.CORE_GROUP):
                        value = pl.create_tile([16], dtype=pl.FP32)
                        scratch = pl.store(value, [0], scratch)
                        dst = pl.store(value, [0], dst)

    after = passes.outline_graph_scopes()(passes.convert_to_ssa()(Before))
    graph = _graph(after)
    assert len(graph.return_types) == 1
    assert isinstance(graph.return_types[0], ir.TensorType)
    assert all(not p.name_hint.startswith("scratch") for p in graph.params)


def test_terminal_graph_inside_for_keeps_external_write():
    @pl.program
    class Before:
        @pl.function
        def main(self, dst: pl.InOut[pl.Tensor[[16], pl.FP32]]):
            for i in pl.range(2):
                with pl.graph("repeated"):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        value = pl.create_tile([16], dtype=pl.FP32)
                        dst = pl.store(value, [0], dst)

    after = passes.outline_graph_scopes()(passes.convert_to_ssa()(Before))
    graph = _graph(after)
    assert len(graph.return_types) == 1
    assert isinstance(graph.return_types[0], ir.TensorType)


def test_graph_real_loop_carried_yield_is_preserved():
    @pl.program
    class Before:
        @pl.function
        def main(self, dst: pl.InOut[pl.Tensor[[16], pl.FP32]]) -> pl.Tensor[[16], pl.FP32]:
            for i, (acc,) in pl.range(2, init_values=(dst,)):
                with pl.graph("carried"):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        value = pl.load(acc, [0], [16])
                        updated = pl.store(pl.add(value, 1.0), [0], acc)
                result = pl.yield_(updated)
            return result

    after = passes.outline_graph_scopes()(passes.convert_to_ssa()(Before))
    graph = _graph(after)
    assert len(graph.return_types) == 1
    assert isinstance(graph.return_types[0], ir.TensorType)
    main = after.get_function("main")
    assert main is not None
    assert isinstance(main.body, ir.SeqStmts)
    loop = main.body.stmts[0]
    assert isinstance(loop, ir.ForStmt)
    assert len(loop.iter_args) == len(loop.return_vars) == 1
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.SSAForm)
    props.insert(passes.IRProperty.UseAfterDef)
    diagnostics = passes.PropertyVerifierRegistry.verify(props, after)
    assert not diagnostics, [d.message for d in diagnostics]


def test_terminal_graph_keeps_taskid_dependencies_between_side_effects():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, src: pl.Tensor[[16], pl.FP32], dst: pl.InOut[pl.Tensor[[16], pl.FP32]]):
            with pl.graph("ordered_writes"):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="first") as ready:
                    value = pl.load(src, [0], [16])
                    dst = pl.store(value, [0], dst)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="second", deps=[ready]):
                    value = pl.load(dst, [0], [16])
                    dst = pl.store(pl.add(value, 1.0), [0], dst)

    after = passes.outline_graph_scopes()(passes.convert_to_ssa()(Before))
    after = passes.outline_incore_scopes()(after)
    graph = _graph(after)
    assert all(isinstance(t, ir.TensorType) for t in graph.return_types)
    launches = []

    class Collect(ir.IRVisitor):
        def visit_submit(self, op):
            launches.append(op)
            super().visit_submit(op)

    Collect().visit_function(graph)
    assert len(launches) == 2
    assert len(launches[1].deps) == 1
    assert isinstance(launches[0].type, ir.TupleType)
    assert isinstance(launches[0].type.types[-1], ir.ScalarType)
    assert launches[0].type.types[-1].dtype == pypto.DataType.TASK_ID


def test_non_graph_terminal_incore_fallback_is_unchanged():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, src: pl.Tensor[[16], pl.FP32]):
            with pl.at(level=pl.Level.CORE_GROUP):
                _result = pl.add(src, src)

    after = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
    kernels = [f for f in after.functions.values() if f.func_type == ir.FunctionType.InCore]
    assert len(kernels) == 1
    assert any(isinstance(t, ir.TensorType) for t in kernels[0].return_types)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

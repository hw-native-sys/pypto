# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""JIT dependency discovery and binding through explicit submissions."""

import ast
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import passes
from pypto.backend import BackendType
from pypto.ir.compiled_program import CompiledProgram
from pypto.jit import _arguments, _ast_analysis, _tensor_meta
from pypto.jit.decorator import _discover_dep_bindings
from pypto.jit.specializer import TensorMeta
from pypto.pypto_core import ir
from pypto.runtime.runner import ExecutionMode, RunConfig

# The callee constexpr must take precedence over this module shape constant.
_RETURN_WIDTH = 64
_FACTORY_ROWS = 64


def _submissions(program: ir.Program) -> list[ir.Submit]:
    submitted: list[ir.Submit] = []

    class Visitor(ir.IRVisitor):
        def visit_submit(self, op: ir.Submit) -> None:
            submitted.append(op)
            super().visit_submit(op)

    Visitor().visit_program(program)
    return submitted


def _graph_chain(rows=16, cols=32):
    @pl.jit.graph
    def stage(x: pl.Tensor, y: pl.InOut[pl.Tensor], factor: pl.constexpr) -> pl.Tensor[[rows, cols], pl.FP32]:
        rows, cols = x.shape
        with pl.at(level=pl.Level.CORE_GROUP):
            value = pl.load(x, [0, 0], [rows, cols])
            pl.store(pl.mul(value, factor), [0, 0], y)
        return y

    alias = stage

    @pl.jit
    def chain(x: pl.Tensor, y: pl.InOut[pl.Tensor], z: pl.InOut[pl.Tensor]):
        first, first_done = pl.submit(stage, x, y, 2)
        second, second_done = pl.submit(alias, first, z, 2, deps=[first_done])
        third, third_done = pl.submit(stage, second, y, factor=3, deps=[second_done])
        return third

    return stage, chain


def test_discovers_submitted_graph_dependencies_and_aliases():
    stage, chain = _graph_chain()
    bindings = _discover_dep_bindings(chain._func)
    assert [binding.call_name for binding in bindings] == ["stage", "alias"]
    assert all(binding.dep is stage for binding in bindings)


@pytest.mark.parametrize("shape", [(16, 32), (32, 128)])
def test_submitted_graph_variants_keep_tensor_results_and_dependencies(shape):
    _, chain = _graph_chain(*shape)
    program = chain.specialize(*(torch.empty(shape) for _ in range(3)))
    submitted = _submissions(program)
    assert len(submitted) == 3
    graphs = [f for f in program.functions.values() if f.func_type == ir.FunctionType.Graph]
    assert len(graphs) == 2
    assert all(len(f.params) == 2 for f in graphs)
    assert submitted[0].op.name == submitted[1].op.name
    assert submitted[2].op.name != submitted[0].op.name
    assert [len(s.deps) for s in submitted] == [0, 1, 1]
    assert all(len(s.args) == 2 for s in submitted)


@pytest.mark.parametrize("spmd", [False, True])
def test_submitted_incore_constexpr_and_dispatch_metadata(spmd):
    @pl.jit.incore
    def kernel(x: pl.Tensor, out: pl.InOut[pl.Tensor], width: pl.constexpr) -> pl.Tensor[[16, 32], pl.FP32]:
        value = pl.load(x, [0, 0], [16, width])
        out = pl.store(value, [0, 0], out)
        return out

    if spmd:

        @pl.jit
        def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
            result, done = pl.spmd_submit(kernel, x, out, width=32, core_num=2)
            return result
    else:

        @pl.jit
        def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
            result, done = pl.submit(kernel, x, out, width=32, allow_early_resolve=True)
            return result

    program = entry.specialize(torch.empty(16, 32), torch.empty(16, 32))
    kernels = [f for f in program.functions.values() if f.func_type == ir.FunctionType.InCore]
    assert len(kernels) == 1
    assert len(kernels[0].params) == 2
    launches = _submissions(program)
    assert len(launches) == 1
    assert launches[0].allow_early_resolve is (not spmd)
    if spmd:
        core_num = launches[0].core_num
        assert isinstance(core_num, ir.ConstInt)
        assert core_num.value == 2
    else:
        assert launches[0].core_num is None


def test_submit_keeps_keyword_constexpr_after_omitted_output():
    @pl.jit.incore
    def kernel(x: pl.Tensor, out: pl.Out[pl.Tensor[[16, 32], pl.FP32]], width: pl.constexpr):
        value = pl.load(x, [0, 0], [16, width])
        out = pl.store(value, [0, 0], out)
        return out

    @pl.jit
    def entry(x: pl.Tensor, out: pl.Out[pl.Tensor]):
        seeded, first = pl.submit(kernel, x, out, width=32)
        result, done = pl.submit(kernel, x, width=32, deps=[first])
        return result

    program = entry.specialize(torch.empty(16, 32), torch.empty(16, 32))
    submitted = _submissions(program)
    assert [len(s.args) for s in submitted] == [2, 1]


def test_submit_preserves_returned_store_expression():
    @pl.jit.incore
    def kernel(x: pl.Tensor, out: pl.InOut[pl.Tensor]) -> pl.Tensor[[16, 32], pl.FP32]:
        value = pl.load(x, [0, 0], [16, 32])
        return pl.store(value, [0, 0], out)

    @pl.jit
    def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
        result, done = pl.submit(kernel, x, out)
        return result

    assert len(_submissions(entry.specialize(torch.empty(16, 32), torch.empty(16, 32)))) == 1


def test_submitted_graph_metadata_follows_return_order():
    @pl.jit.graph
    def reorder(a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor]):
        return b, a

    @pl.jit.graph
    def consume(value: pl.Tensor):
        return value

    @pl.jit
    def entry(a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor]):
        (first, second), done = pl.submit(reorder, a, b)
        result, consumed = pl.submit(consume, first, deps=[done])
        return result

    program = entry.specialize(torch.empty(16, 32), torch.empty(32, 64))
    consumer = program.get_function("consume")
    assert consumer is not None
    assert ir.structural_equal(consumer.params[0].type, ir.TensorType([32, 64], pl.FP32))


@pytest.mark.parametrize("keyword_store", [False, True])
def test_graph_local_store_alias_preserves_return_metadata(keyword_store):
    @pl.jit.graph
    def stage(x: pl.Tensor, scratch: pl.InOut[pl.Tensor], out: pl.InOut[pl.Tensor], keyword: pl.constexpr):
        with pl.at(level=pl.Level.CORE_GROUP):
            value = pl.load(x, [0, 0], [32, 64])
            if keyword:
                stored = pl.tile.store(value, [0, 0], output_tensor=out)
            else:
                stored = pl.store(value, [0, 0], out)
        return stored

    @pl.jit.graph
    def consume(value: pl.Tensor):
        return value

    @pl.jit
    def entry(x: pl.Tensor, scratch: pl.InOut[pl.Tensor], out: pl.InOut[pl.Tensor]):
        result = stage(x, scratch, out, keyword=keyword_store)
        return consume(result)

    program = entry.specialize(torch.empty(32, 64), torch.empty(16, 32), torch.empty(32, 64))
    consumer = program.get_function("consume")
    assert consumer is not None
    assert ir.structural_equal(consumer.params[0].type, ir.TensorType([32, 64], pl.FP32))


@pytest.mark.parametrize("qualified", [False, True])
def test_graph_assemble_alias_preserves_target_metadata(qualified):
    @pl.jit.graph
    def stage(x: pl.Tensor, scratch: pl.InOut[pl.Tensor], out: pl.InOut[pl.Tensor], tensor_op: pl.constexpr):
        if tensor_op:
            assembled = pl.tensor.assemble(target=out, source=x, offset=[0, 0])
        else:
            assembled = pl.assemble(out, x, [0, 0])
        return assembled

    @pl.jit.graph
    def consume(value: pl.Tensor):
        return value

    @pl.jit
    def entry(x: pl.Tensor, scratch: pl.InOut[pl.Tensor], out: pl.InOut[pl.Tensor]):
        result = stage(x, scratch, out, tensor_op=qualified)
        return consume(result)

    program = entry.specialize(torch.empty(16, 32), torch.empty(8, 16), torch.empty(32, 64))
    consumer = program.get_function("consume")
    assert consumer is not None
    assert ir.structural_equal(consumer.params[0].type, ir.TensorType([32, 64], pl.FP32))


def test_single_lhs_submit_projection_keeps_tensor_metadata():
    @pl.jit.graph
    def stage(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
        return out

    @pl.jit.graph
    def consume(value: pl.Tensor):
        return value

    @pl.jit
    def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
        launch = pl.submit(stage, x, out)
        value = launch[0]
        result, done = pl.submit(consume, value)
        return result

    assert len(_submissions(entry.specialize(torch.empty(16, 32), torch.empty(16, 32)))) == 2


def _assert_codegen_artifacts(entry, samples, tmp_path):
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        compiled = entry.compile(
            *samples,
            config=RunConfig(
                backend_type=BackendType.Ascend910,
                execution_mode=ExecutionMode.ONBOARD,
                save_kernels_dir=str(tmp_path / "compiled"),
                codegen_only=True,
            ),
        )
    assert isinstance(compiled, CompiledProgram)
    assert list(Path(compiled.output_dir).rglob("*.pto"))
    assert list(Path(compiled.output_dir).rglob("*.cpp"))


@pytest.mark.parametrize("kind", ["graph", "submit", "spmd_submit"])
def test_jit_dependencies_compile_through_codegen(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))
    if kind == "graph":
        stage, _ = _graph_chain()

        @pl.jit
        def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor], other: pl.InOut[pl.Tensor]):
            first = stage(x, out, 2)
            result = stage(first, other, 3)
            return result

        samples = [torch.empty(16, 32) for _ in range(3)]
    else:

        @pl.jit.incore
        def kernel(x: pl.Tensor, out: pl.InOut[pl.Tensor], width: pl.constexpr):
            value = pl.load(x, [0, 0], [16, width])
            out = pl.store(value, [0, 0], out)
            return out

        if kind == "spmd_submit":

            @pl.jit
            def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
                result, done = pl.spmd_submit(kernel, x, out, width=32, core_num=2)
                return result
        else:

            @pl.jit
            def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
                result, done = pl.submit(kernel, x, out, width=32)
                return result

        samples = [torch.empty(16, 32) for _ in range(2)]
    _assert_codegen_artifacts(entry, samples, tmp_path)


@pytest.mark.parametrize("binding", ["alias", "join", "tensor_rebind"])
def test_submit_tuple_metadata_tracks_binding_lifetime(binding):
    @pl.jit.graph
    def stage(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
        return out

    @pl.jit.graph
    def consume(value: pl.Tensor):
        return value

    samples: list[object]
    expected = [16, 32]
    if binding == "alias":

        @pl.jit
        def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
            launch = pl.submit(stage, x, out)
            saved = launch
            value = saved[0]
            result, done = pl.submit(consume, value)
            return result

        samples = [torch.empty(16, 32), torch.empty(16, 32)]
    elif binding == "join":

        @pl.jit
        def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor], flag: pl.Scalar[pl.INT32]):
            launch = pl.submit(stage, x, out)
            if flag > 0:
                saved = launch
            else:
                saved = launch
            value = saved[0]
            result, done = pl.submit(consume, value)
            return result

        samples = [torch.empty(16, 32), torch.empty(16, 32), 1]
    else:

        @pl.jit
        def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor], other: pl.Tensor):
            launch = pl.submit(stage, x, out)
            saved = launch
            saved = other
            value = saved[0]
            result, done = pl.submit(consume, value)
            return result

        samples = [torch.empty(16, 32), torch.empty(16, 32), torch.empty(32, 64)]
        expected = [64]
    program = entry.specialize(*samples)
    consumer = program.get_function("consume")
    assert consumer is not None
    assert ir.structural_equal(consumer.params[0].type, ir.TensorType(expected, pl.FP32))


def test_submitted_expression_return_resolves_constexpr_annotation():
    @pl.jit.incore
    def kernel(
        x: pl.Tensor, out: pl.InOut[pl.Tensor], _RETURN_WIDTH: pl.constexpr
    ) -> "pl.Tensor[[16, _RETURN_WIDTH], pl.FP32]":
        value = pl.load(x, [0, 0], [16, _RETURN_WIDTH])
        return pl.store(value, [0, 0], out)

    @pl.jit
    def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
        result, done = pl.submit(kernel, x, out, _RETURN_WIDTH=32)
        return result

    assert len(_submissions(entry.specialize(torch.empty(16, 32), torch.empty(16, 32)))) == 1


def test_graph_call_expression_retains_single_output_metadata():
    @pl.jit.incore
    def kernel(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
        value = pl.load(x, [0, 0], [16, 32])
        out = pl.store(value, [0, 0], out)
        return out

    _FACTORY_ROWS = 16

    @pl.jit.graph
    def stage(x: pl.Tensor, out: pl.InOut[pl.Tensor]) -> pl.Tensor[[_FACTORY_ROWS, 32], pl.FP32]:
        return kernel(x, out)

    @pl.jit.graph
    def consume(value: pl.Tensor):
        return value

    @pl.jit
    def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
        value, done = pl.submit(stage, x, out)
        result, consumed = pl.submit(consume, value, deps=[done])
        return result

    program = entry.specialize(torch.empty(16, 32), torch.empty(16, 32))
    assert len(_submissions(program)) == 2
    stage_ir = program.get_function("stage")
    assert stage_ir is not None
    assert ir.structural_equal(stage_ir.return_types[0], ir.TensorType([16, 32], pl.FP32))


@pytest.mark.parametrize("capture", ["shape", "dtype", "shadow"])
def test_ordinary_expression_return_keeps_annotation_only_factory_captures(capture, tmp_path, monkeypatch):
    monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))
    output_shape = [16, 32]
    output_dtype = pl.FP32
    _FACTORY_ROWS = 16

    @pl.jit.incore
    def body(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
        value = pl.load(x, [0, 0], [16, 32])
        out = pl.store(value, [0, 0], out)
        return out

    if capture == "shape":

        @pl.jit
        def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor]) -> pl.Tensor[output_shape, pl.FP32]:
            return body(x, out)
    elif capture == "dtype":

        @pl.jit
        def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor]) -> pl.Tensor[[16, 32], output_dtype]:
            return body(x, out)

    else:

        @pl.jit
        def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor]) -> pl.Tensor[[_FACTORY_ROWS, 32], pl.FP32]:
            return body(x, out)

    samples = [torch.empty(32, 64), torch.empty(32, 64)]
    program = entry.specialize(*samples)
    main = program.get_function("entry")
    assert main is not None
    assert not main.return_types
    _assert_codegen_artifacts(entry, samples, tmp_path)


def test_first_submission_allocates_typed_omitted_output(tmp_path, monkeypatch):
    monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))

    @pl.jit.incore
    def kernel(x: pl.Tensor, out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]):
        value = pl.load(x, [0, 0], [16, 32])
        out = pl.store(value, [0, 0], out)
        return out

    @pl.jit
    def entry(x: pl.Tensor):
        value, done = pl.submit(kernel, x)
        return value

    _assert_codegen_artifacts(entry, [torch.empty(16, 32)], tmp_path)


def test_submitted_unannotated_store_expression_returns_tensor(tmp_path, monkeypatch):
    monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))

    @pl.jit.incore
    def kernel(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
        value = pl.load(x, [0, 0], [16, 32])
        return pl.store(value, [0, 0], out)

    @pl.jit
    def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
        value, done = pl.submit(kernel, x, out)
        return value

    _assert_codegen_artifacts(entry, [torch.empty(16, 32), torch.empty(16, 32)], tmp_path)


def test_single_lhs_forwarded_tuple_contract_preserves_all_slots():
    @pl.jit.graph
    def pair(a: pl.Tensor, b: pl.Tensor):
        return a, b

    @pl.jit.graph
    def forward(
        a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor]
    ) -> tuple[pl.Tensor[[16, 32], pl.FP32], pl.Tensor[[32, 64], pl.FP32]]:
        return pair(a, b)

    @pl.jit.graph
    def consume(value: pl.Tensor):
        return value

    @pl.jit
    def entry(a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor]):
        launch = pl.submit(forward, a, b)
        second = launch[1]
        result, done = pl.submit(consume, second)
        return result

    program = entry.specialize(torch.empty(16, 32), torch.empty(32, 64))
    consumer = program.get_function("consume")
    assert consumer is not None
    assert ir.structural_equal(consumer.params[0].type, ir.TensorType([32, 64], pl.FP32))


@pytest.mark.parametrize("projection", ["literal", "folded"])
def test_direct_submit_projection_argument_preserves_tensor_metadata(projection):
    @pl.jit.graph
    def stage(first: pl.InOut[pl.Tensor], second: pl.InOut[pl.Tensor]):
        return first, second

    @pl.jit.graph
    def consume(value: pl.Tensor):
        return value

    if projection == "literal":

        @pl.jit
        def entry(selector: pl.Tensor, first: pl.InOut[pl.Tensor], second: pl.InOut[pl.Tensor]):
            launch = pl.submit(stage, first, second)
            result, done = pl.submit(consume, launch[1])
            return result
    else:

        @pl.jit
        def entry(selector: pl.Tensor, first: pl.InOut[pl.Tensor], second: pl.InOut[pl.Tensor]):
            launch = pl.submit(stage, first, second)
            slot = selector.shape[0]  # pyright: ignore[reportOptionalSubscript]
            result, done = pl.submit(consume, launch[slot])
            return result

    program = entry.specialize(torch.empty(1, 32), torch.empty(8, 32), torch.empty(16, 32))
    assert len(_submissions(program)) == 2
    consumer = program.get_function("consume")
    assert consumer is not None
    assert ir.structural_equal(consumer.params[0].type, ir.TensorType([16, 32], pl.FP32))


def test_first_omitted_output_uses_call_owned_constexpr_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))

    rows = 16
    output_dtype = pl.FP32
    output_layout = pl.ND

    @pl.jit.incore
    def kernel(
        x: pl.Tensor,
        out: pl.Out[pl.Tensor[[rows, _RETURN_WIDTH], output_dtype, output_layout]],
        _RETURN_WIDTH: pl.constexpr,
    ):
        value = pl.load(x, [0, 0], [16, _RETURN_WIDTH])
        out = pl.store(value, [0, 0], out)
        return out

    @pl.jit
    def entry(x: pl.Tensor):
        result, done = pl.submit(kernel, x, _RETURN_WIDTH=32)
        return result

    program = entry.specialize(torch.empty(16, 32))
    outputs = [
        param.type
        for function in program.functions.values()
        for param in function.params
        if param.name_hint == "out"
    ]
    assert outputs and all(ir.structural_equal(typ, ir.TensorType([16, 32], pl.FP32)) for typ in outputs)
    _assert_codegen_artifacts(entry, [torch.empty(16, 32)], tmp_path)


def test_omitted_output_uses_eager_factory_capture_even_when_global_resolves():
    _FACTORY_ROWS = 16

    @pl.jit.incore
    def kernel(x: pl.Tensor, out: pl.Out[pl.Tensor[[_FACTORY_ROWS, 32], pl.FP32]]):
        return pl.store(pl.load(x, [0, 0], [16, 32]), [0, 0], out)

    @pl.jit
    def entry(x: pl.Tensor):
        value, done = pl.submit(kernel, x)
        return value

    program = entry.specialize(torch.empty(16, 32))
    kernel_ir = program.get_function("kernel")
    assert kernel_ir is not None
    assert ir.structural_equal(kernel_ir.params[1].type, ir.TensorType([16, 32], pl.FP32))


def test_factory_annotation_changes_the_compiler_source_key():
    def factory(rows):
        @pl.jit.incore
        def body(x: pl.Tensor, out: pl.Out[pl.Tensor[[rows, 32], pl.FP32]]):
            return out

        return body

    assert factory(16)._get_source_hash() != factory(64)._get_source_hash()


def test_submit_rejects_runtime_argument_after_omitted_output():
    @pl.jit.incore
    def kernel(
        x: pl.Tensor, left: pl.Out[pl.Tensor[[16, 32], pl.FP32]], right: pl.Out[pl.Tensor[[16, 32], pl.FP32]]
    ):
        return left, right

    @pl.jit
    def entry(x: pl.Tensor, supplied: pl.InOut[pl.Tensor]):
        launch = pl.submit(kernel, x, right=supplied)
        return launch[0]

    with pytest.raises(ValueError, match="cannot skip a runtime argument"):
        entry.specialize(torch.empty(16, 32), torch.empty(16, 32))


def test_returned_submit_projection_uses_rebound_tuple():
    @pl.jit.incore
    def stage(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
        return pl.store(pl.load(out, [0, 0], [16, 64]), [0, 0], out)

    @pl.jit.graph
    def relay(x: pl.Tensor, out: pl.InOut[pl.Tensor]):
        launch = (x, x)
        launch = pl.submit(stage, x, out)
        return launch[0], launch[-1]

    definition = _ast_analysis.get_func_def(relay._func)
    site = _tensor_meta._return_site(definition)
    assert site is not None and isinstance(site.value, ast.Tuple)
    returned = _tensor_meta._ReturnMeta(site, list(site.value.elts))
    out = TensorMeta((16, 64), pl.FP32)
    _tensor_meta.extract_local_tensor_metas(
        relay._func,
        seed_meta={"x": TensorMeta((16, 32), pl.FP32), "out": out},
        caller_func_type="graph",
        discover_deps=_discover_dep_bindings,
        outputs=_tensor_meta._MetadataOutputs(return_meta=returned),
    )
    result, task_id = site.value.elts
    assert returned.metas[result] == out
    assert returned.metas[task_id] is None


def test_distributed_eager_annotation_keeps_its_kind_with_constexpr():
    source = ast.parse("pld.DistributedTensor[[16, WIDTH], pl.FP32]", mode="eval").body
    eager = pld.DistributedTensor[[16, 64], pl.FP32]
    result = _arguments.resolve_call_annotation(
        source, eager, {"pld": pld, "pl": pl, "WIDTH": 32}, {"WIDTH": "32"}
    )
    assert type(result) is pld.DistributedTensor
    assert result.shape == [16, 32]


def test_submit_tuple_metadata_joins_zero_trip_loop_path():
    @pl.jit.graph
    def stage(out: pl.InOut[pl.Tensor]):
        return out

    @pl.jit
    def entry(first: pl.InOut[pl.Tensor], next_out: pl.InOut[pl.Tensor], trips: pl.Scalar[pl.INDEX]):
        launch = pl.submit(stage, first)
        for i in pl.range(trips):
            launch = pl.submit(stage, next_out)
        return launch[0]

    definition = _ast_analysis.get_func_def(entry._func)
    site = _tensor_meta._return_site(definition)
    assert site is not None and site.value is not None
    returned = _tensor_meta._ReturnMeta(site, [site.value])
    first = TensorMeta((16, 32), pl.FP32)
    next_out = TensorMeta((16, 64), pl.FP32)
    _tensor_meta.extract_local_tensor_metas(
        entry._func,
        seed_meta={"first": first, "next_out": next_out},
        discover_deps=_discover_dep_bindings,
        outputs=_tensor_meta._MetadataOutputs(return_meta=returned),
    )
    assert returned.metas[site.value] is None


def test_submit_task_id_projection_is_not_inferred_as_a_tensor():
    @pl.jit.graph
    def stage(x: pl.Tensor):
        return x

    @pl.jit.graph
    def consume(value: pl.Tensor):
        return value

    @pl.jit
    def entry(x: pl.Tensor):
        launch = pl.submit(stage, x)
        result, done = pl.submit(consume, launch[-1])
        return result

    with pytest.raises(ValueError, match="missing inferred tensor metadata"):
        entry.specialize(torch.empty(16, 32))


@pytest.mark.parametrize("keyword_store", [False, True])
def test_submitted_store_returns_actual_destination_not_first_out(keyword_store, tmp_path, monkeypatch):
    monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))
    if keyword_store:

        @pl.jit.incore
        def kernel(x: pl.Tensor, scratch: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor]):
            value = pl.load(x, [0, 0], [32, 64])
            return pl.store(tile=value, offsets=[0, 0], output_tensor=out)
    else:

        @pl.jit.incore
        def kernel(x: pl.Tensor, scratch: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor]):
            value = pl.load(x, [0, 0], [32, 64])
            return pl.store(value, [0, 0], out)

    @pl.jit.incore
    def consume(value: pl.Tensor, sink: pl.InOut[pl.Tensor]):
        tile = pl.load(value, [0, 0], [32, 64])
        sink = pl.store(tile, [0, 0], sink)
        return sink

    @pl.jit
    def entry(
        x: pl.Tensor, scratch: pl.InOut[pl.Tensor], out: pl.InOut[pl.Tensor], sink: pl.InOut[pl.Tensor]
    ):
        value, done = pl.submit(kernel, x, scratch, out)
        result, consumed = pl.submit(consume, value, sink, deps=[done])
        return result

    samples = [torch.empty(32, 64), torch.empty(16, 32), torch.empty(32, 64), torch.empty(32, 64)]
    program = entry.specialize(*samples)
    consumer = program.get_function("consume")
    assert consumer is not None
    assert ir.structural_equal(consumer.params[0].type, ir.TensorType([32, 64], pl.FP32))
    _assert_codegen_artifacts(entry, samples, tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

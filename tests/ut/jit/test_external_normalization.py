# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Source-level external effects through JIT specialization, SSA, and codegen."""

import ast
import importlib.util
import re
import textwrap

import pypto.language as pl
import pytest
import torch
from pypto.backend.pto_backend import generate
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit._signature_variants import _signature
from pypto.jit._tensor_meta import extract_local_tensor_metas
from pypto.jit.decorator import _discover_dep_bindings
from pypto.jit.external._alias_analysis import AliasTracker
from pypto.jit.external.compatibility import request_from_context
from pypto.jit.external.contracts import (
    Access,
    AliasPermission,
    CompileTarget,
    Coverage,
    KernelContract,
    ParameterSpec,
    ReturnBinding,
    TensorEffect,
    TensorSpec,
    WritebackBinding,
)
from pypto.jit.specializer import DynDim, Specializer, TensorMeta
from pypto.pypto_core import DataType, ir


def _specialize(tmp_path, body, *, core="aiv", public=True, contract=False, permit_alias=False):
    native = tmp_path / "kernel.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    source_args = (
        f"aic_source={str(native)!r}, aiv_source={str(native)!r}"
        if core == "mixed"
        else f"source={str(native)!r}"
    )
    result = "return out" if public else "..."
    out_direction = "pl.Out[pl.Tensor]" if public else "pl.InOut[pl.Tensor]"
    path = tmp_path / "normalization_case.py"
    path.write_text(f"""import pypto.language as pl
@pl.jit.extern(core_type={core!r}, {source_args})
def producer(x: pl.Tensor, state: pl.InOut[pl.Tensor], out: {out_direction}):
    {result}
@pl.jit.extern(core_type="aiv", source={str(native)!r})
def consumer(state: pl.Tensor, out: pl.Tensor, result: pl.Out[pl.Tensor]):
    return result
@pl.jit.inline
def inner(x: pl.Tensor, state: pl.Tensor, out: pl.Tensor):
    producer(x, state, out)
@pl.jit.inline
def helper(x: pl.Tensor, state: pl.Tensor, out: pl.Tensor):
    inner(x, state, out)
@pl.jit(auto_scope=False)
def entry(x: pl.Tensor, state: pl.InOut[pl.Tensor], out: pl.InOut[pl.Tensor],
          result: pl.Out[pl.Tensor], flag: pl.Scalar[pl.BOOL]):
{textwrap.indent(body, "    ")}
    result = consumer(state, out, result)
    return result, state, out
""")
    spec = importlib.util.spec_from_file_location("normalization_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = module.entry
    args = (*(torch.empty(32) for _ in range(4)), True)
    _, _, tensors, scalars, constants, dynamics = entry._bind_args(args, {})
    contexts = entry._build_contexts(tensors, scalars, constants, dynamics)
    if contract:
        provider = KernelContract(
            tuple(ParameterSpec(name, TensorSpec((32,), DataType.FP32)) for name in ("x", "state", "out")),
            (
                TensorEffect(0, Access.READ),
                TensorEffect(1, Access.WRITE, Coverage.PARTIAL),
                TensorEffect(2, Access.WRITE, Coverage.FULL),
            ),
            public_returns=(ReturnBinding(2),) if public else (),
            writebacks=(WritebackBinding(1), WritebackBinding(2)),
            alias_permissions=(AliasPermission(0, 1),) if permit_alias else (),
        )
        for context in contexts:
            if context.source_def_name == "producer":
                context.external_contract = provider
    source = Specializer("Normalized", contexts).specialize()
    program = pl.parse(source)
    assert isinstance(program, ir.Program)
    return source, program


@pytest.mark.parametrize("public", [False, True], ids=["void", "public-return"])
@pytest.mark.parametrize("core", ["aiv", "aic", "mixed"])
@pytest.mark.parametrize(
    "body",
    [
        "helper(x, state, out)",
        "for i in pl.range(2):\n    helper(x, state, out)",
        "for i in pl.range(0):\n    helper(x, state, out)",
        "if flag:\n    helper(x, state, out)",
        "for i in pl.range(2):\n    if flag:\n        helper(x, state, out)",
    ],
)
def test_hidden_writes_cross_inline_and_control_flow(tmp_path, public, core, body):
    _, program = _specialize(tmp_path, body, public=public, core=core, contract=True)
    producer = program.get_function("producer_aic" if core == "mixed" else "producer")
    assert producer is not None
    assert ir.returned_param_indices(producer, program) == [1, 2]
    assert producer.param_directions[1] == ir.ParamDirection.InOut
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    files = generate(after, str(tmp_path / "generated"), skip_ptoas=True)
    code = files["orchestration/entry.cpp"]
    consumer = re.split(r"// Task \d+: consumer", code)[-1]
    inputs = re.findall(r"\.add_input\((\w+)\)", consumer)
    assert len(inputs) == 2
    for name, argument in zip(("state", "out"), inputs, strict=True):
        assert argument == f"ext_{name}" or f"{argument} = ext_{name};" in code
    assert "add_output(ext_result)" in code
    if "range(0)" in body:
        assert "add_inout" not in code
    else:
        assert "add_inout" in code
        if core == "mixed":
            assert "MixedKernels" in code


def test_legacy_void_writeback_is_materialized(tmp_path):
    source, program = _specialize(tmp_path, "helper(x, state, out)", public=False)
    assert "state = _external_result_" in source
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    code = generate(after, str(tmp_path / "generated"), skip_ptoas=True)["orchestration/entry.cpp"]
    assert "add_inout(ext_state)" in code
    assert "add_input(ext_state)" in code


@pytest.mark.parametrize(
    "annotation", ["pl.Tensor[[16, 32], pl.FP32]", "tuple[pl.Tensor[[16, 32], pl.FP32]]"]
)
def test_singleton_legacy_return_reaches_consumer(tmp_path, annotation):
    native = tmp_path / "singleton.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    path = tmp_path / "singleton_case.py"
    path.write_text(f"""import pypto.language as pl
@pl.jit.extern(source={str(native)!r}, core_type="aiv")
def producer(out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> {annotation}:
    ...
@pl.jit.extern(source={str(native)!r}, core_type="aiv")
def consumer(x: pl.Tensor):
    return x
@pl.jit
def entry(out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]):
    value = producer(out)
    return consumer(value)
""")
    spec = importlib.util.spec_from_file_location("singleton_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program = module.entry.specialize()
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    code = generate(after, str(tmp_path / "generated"), skip_ptoas=True)["orchestration/entry.cpp"]
    assert "add_input(ext_out)" in code


@pytest.mark.parametrize("direct", [False, True])
def test_inline_extern_return_reaches_consumer(tmp_path, direct):
    native = tmp_path / "inline_return.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    path = tmp_path / "inline_return_case.py"
    body = "return producer(x)" if direct else "value = producer(x)\nreturn value"
    path.write_text(f"""import pypto.language as pl
@pl.jit.extern(source={str(native)!r}, core_type="aiv")
def producer(x: pl.Tensor):
    return x
@pl.jit.extern(source={str(native)!r}, core_type="aiv")
def consumer(x: pl.Tensor):
    return x
@pl.jit.inline
def helper(x: pl.Tensor):
{textwrap.indent(body, "    ")}
@pl.jit
def entry(x: pl.Tensor[[16, 32], pl.FP32]):
    value = helper(x)
    return consumer(value)
""")
    spec = importlib.util.spec_from_file_location("inline_return_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program = module.entry.specialize()
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    code = generate(after, str(tmp_path / "generated"), skip_ptoas=True)["orchestration/entry.cpp"]
    assert "add_input(ext_x)" in code


@pytest.mark.parametrize("explicit_first", [False, True])
def test_omitted_layout_does_not_reuse_explicit_nd_variant(tmp_path, explicit_first):
    native = tmp_path / "layout.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')

    @pl.jit.extern(source=native, core_type="aiv")
    def external(x: pl.Tensor[[16, 32], pl.FP32, pl.NZ]):
        return x

    @pl.jit.inline
    def helper(x: pl.Tensor):
        value = external(x)
        return value

    @pl.jit
    def entry(a: pl.Tensor, b: pl.Tensor):
        first = helper(a)
        second = helper(b)
        local = pl.create_tensor([16, 32], pl.FP32, layout=pl.ND)
        third = helper(local)
        return first, second, third

    layouts = (ir.TensorLayout.ND, None) if explicit_first else (None, ir.TensorLayout.ND)
    metas = {
        name: TensorMeta((16, 32), DataType.FP32, layout=layout)
        for name, layout in zip(("a", "b"), layouts, strict=True)
    }
    local_metas = extract_local_tensor_metas(entry._func, metas, discover_deps=_discover_dep_bindings)
    assert local_metas["local"].layout == ir.TensorLayout.ND
    contexts = entry._build_contexts(metas, {}, {}, {})
    helpers = [context for context in contexts if context.source_def_name == "helper"]
    externs = [context for context in contexts if context.source_def_name == "external"]
    assert len(helpers) == len(externs) == 2
    program = pl.parse(Specializer("Layouts", contexts).specialize())
    assert isinstance(program, ir.Program)
    for context, layout in zip(externs, layouts, strict=True):
        function = program.get_function(context.func_name)
        assert function is not None
        tensor_type = function.params[0].type
        assert isinstance(tensor_type, ir.TensorType)
        actual_layout = tensor_type.tensor_view.layout if tensor_type.tensor_view else ir.TensorLayout.ND
        assert actual_layout == (ir.TensorLayout.NZ if layout is None else layout)


@pytest.mark.parametrize(
    "body",
    [
        "out = producer(x, state, out)",
        "alias = state\nhelper(x, alias, out)",
        "view = pl.slice(state, [16], [0])\nhelper(pl.slice(x, [16], [0]), view, pl.slice(out, [16], [0]))",
        "alias = state\nalias = pl.create_tensor([32], dtype=pl.FP32)\nhelper(x, alias, out)",
    ],
)
def test_public_return_alias_view_and_rebinding(tmp_path, body):
    _, program = _specialize(tmp_path, body)
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    code = generate(after, str(tmp_path / "generated"), skip_ptoas=True)["orchestration/entry.cpp"]
    assert "add_input(ext_state)" in code
    assert "add_input(ext_out)" in code
    if "create_tensor" in body:
        assert "add_inout(ext_state)" not in code
    elif "slice" not in body:
        assert "add_inout(ext_state)" in code


@pytest.mark.parametrize(
    "body",
    [
        "if producer(x, state, out):\n    pass",
        "out = consumer(state, producer(x, state, out), out)",
    ],
)
def test_nested_external_evaluation_is_diagnosed(tmp_path, body):
    with pytest.raises(ValueError, match="External calls must be standalone"):
        _specialize(tmp_path, body)


@pytest.mark.parametrize("difference", ["layout", "scalar", "dynamic-equality"])
def test_signature_variants_preserve_argument_constraints(tmp_path, difference):
    native = tmp_path / "signature.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')

    @pl.jit.extern(source=native, core_type="aiv")
    def external(x: pl.Tensor, y: pl.Tensor, count: pl.Scalar):
        return y

    @pl.jit.inline
    def helper(x: pl.Tensor, y: pl.Tensor, count: pl.Scalar):
        value = external(x, y, count)
        return value

    @pl.jit
    def entry(a: pl.Tensor, b: pl.Tensor, c: pl.Tensor, n: pl.Scalar, m: pl.Scalar):
        first = helper(a, a, n)
        second = helper(b, c, m)
        third = helper(a, a, n)
        return first, second, third

    tensors = {name: TensorMeta((16, 32), DataType.FP32) for name in ("a", "b", "c")}
    scalars = {"n": DataType.INT32, "m": DataType.INT32}
    if difference == "layout":
        tensors["b"].layout = tensors["c"].layout = ir.TensorLayout.NZ
    elif difference == "scalar":
        scalars["m"] = DataType.INT64
    else:
        tensors["a"].shape = (DynDim("M", "M", 16), 32)
        tensors["b"].shape = (DynDim("K", "K", 16), 32)
        tensors["c"].shape = (DynDim("N", "N", 16), 32)
    contexts = entry._build_contexts(tensors, scalars, {}, {})
    source = Specializer("Signatures", contexts).specialize()
    program = pl.parse(source)
    assert isinstance(program, ir.Program)
    variants = [context for context in contexts if context.source_def_name == "external"]
    assert len(variants) == 2
    assert len([context for context in contexts if context.source_def_name == "helper"]) == 2
    first, second = (program.get_function(context.func_name) for context in variants)
    assert first is not None and second is not None
    if difference == "scalar":
        assert isinstance(first.params[2].type, ir.ScalarType)
        assert isinstance(second.params[2].type, ir.ScalarType)
        assert first.params[2].type.dtype == DataType.INT32
        assert second.params[2].type.dtype == DataType.INT64
    elif difference == "layout":
        assert not ir.structural_equal(first.params[0].type, second.params[0].type)
    else:
        assert ir.structural_equal(first.params[0].type, first.params[1].type)
        assert not ir.structural_equal(second.params[0].type, second.params[1].type)


def test_signature_reuses_renamed_dynamic_symbols(tmp_path):
    def signature(left, right):
        return _signature(
            ["x", "y"],
            {"x": TensorMeta((left, 16), DataType.FP32), "y": TensorMeta((right, 16), DataType.FP32)},
            {},
        )

    m, n = DynDim("M", "M", 32), DynDim("N", "N", 64)
    assert signature(m, m) == signature(n, n)
    assert signature(m, m) != signature(m, n)


@pytest.mark.parametrize(
    "body",
    [
        "producer(state, state, out)",
        "alias = state\nproducer(alias, state, out)",
        "view = pl.slice(state, [32], [0])\nproducer(view, state, out)",
        "view = pl.tensor.slice(tensor=state, shape=[32], offset=[0])\nproducer(view, state, out)",
        "view = pl.tensor.reshape(tensor=state, shape=[32])\nproducer(view, state, out)",
        "alias, other = state, x\nproducer(alias, state, out)",
        "updated = pl.assemble(state, x, [0])\nproducer(updated, state, out)",
        "updated = pl.assemble(target=state, source=x, offset=[0])\nproducer(updated, state, out)",
        "updated = pl.write(state, [0], 1.0)\nproducer(updated, state, out)",
        "with pl.scope():\n    alias = state\nproducer(alias, state, out)",
        "alias = x\nwith pl.scope():\n    alias = state\nproducer(alias, state, out)",
        "with pl.manual_scope():\n    alias = state\nproducer(alias, state, out)",
        "alias = x\nwith pl.manual_scope():\n    alias = state\nproducer(alias, state, out)",
        "local = pl.tensor.create_tensor([32], pl.FP32)\n"
        "view = pl.slice(local, [32], [0])\nproducer(view, local, out)",
        "pair = (state, x)\nalias = pair[0]\nproducer(alias, state, out)",
        "pair = (x, state)\nalias = pair[-1]\nproducer(alias, state, out)",
        "pair = (state, x)\ncopy = pair\nalias = copy[0]\nproducer(alias, state, out)",
        "pair = (state, x)\nwith pl.scope():\n    alias = pair[0]\nproducer(alias, state, out)",
        # Clamping changes the actual region; requested intervals cannot prove separation.
        "lo = pl.slice(state, [16], [16])\n"
        "hi = pl.slice(state, [16], [32], clamp=True)\nproducer(lo, hi, out)",
        # Different parents/coordinate systems must not use sibling intervals.
        "lo = pl.slice(state, [16], [0])\n"
        "shifted = pl.slice(state, [24], [8])\n"
        "hi = pl.slice(shifted, [8], [0])\nproducer(lo, hi, out)",
        "lo = pl.slice(state, [16], [0])\n"
        "reshaped = pl.reshape(state, [32])\n"
        "hi = pl.slice(reshaped, [16], [0])\nproducer(lo, hi, out)",
    ],
)
def test_provider_overlap_requires_explicit_permission(tmp_path, body):
    with pytest.raises(ValueError, match="overlapping.*without an AliasPermission"):
        _specialize(tmp_path, body, contract=True)


@pytest.mark.parametrize("offset", [0, 8, 16])
@pytest.mark.parametrize("keyword", [False, True])
def test_provider_disjoint_constant_slices(tmp_path, offset, keyword):
    second = (
        f"pl.tensor.slice(tensor=state, shape=[16], offset=[{offset}])"
        if keyword
        else f"pl.slice(state, [16], [{offset}])"
    )
    body = f"lo = pl.slice(state, [16], [0])\nhi = {second}\nproducer(lo, hi, out)"
    if offset < 16:
        with pytest.raises(ValueError, match="overlapping.*without an AliasPermission"):
            _specialize(tmp_path, body, contract=True)
    else:
        _, program = _specialize(tmp_path, body, contract=True)
        assert program.get_function("entry") is not None


def test_provider_permitted_overlap_and_local_rebinding(tmp_path):
    _, program = _specialize(tmp_path, "producer(state, state, out)", contract=True, permit_alias=True)
    assert program.get_function("entry") is not None

    _, program = _specialize(
        tmp_path,
        "alias = state\nalias = pl.create_tensor([32], dtype=pl.FP32)\nproducer(alias, state, out)",
        contract=True,
    )
    assert program.get_function("entry") is not None


def test_branch_joined_view_extent_remains_dynamic(tmp_path):
    native = tmp_path / "branch.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')

    @pl.jit.extern(source=native, core_type="aiv")
    def consume(root: pl.Tensor, left: pl.Tensor, right: pl.Tensor):
        return left

    @pl.jit
    def entry(x: pl.Tensor, flag: pl.Scalar[pl.BOOL], limit: pl.Scalar[pl.INDEX], other: pl.Scalar[pl.INDEX]):
        if flag:
            view = pl.slice(x, [limit, 16], [0, 0])
        else:
            view = pl.slice(x, [other, 16], [0, 0])
        return consume(x, view, view)

    _, _, tensors, scalars, constants, dynamics = entry._bind_args((torch.empty(32, 16), True, 8, 12), {})
    contexts = entry._build_contexts(tensors, scalars, constants, dynamics)
    context = next(context for context in contexts if context.source_def_name == "consume")
    request = request_from_context(context, CompileTarget("a5", "test", "1"))
    root, left, right = (parameter.type for parameter in request.parameters)
    assert isinstance(root, TensorSpec) and isinstance(left, TensorSpec) and isinstance(right, TensorSpec)
    assert root.shape == (32, 16)
    assert left.shape == right.shape == ("d0", 16)
    assert isinstance(context.tensor_meta["left"].shape[0], DynDim)
    assert context.tensor_meta["left"].shape[0] is context.tensor_meta["right"].shape[0]


@pytest.mark.parametrize(
    "view, expected, input_shape",
    [
        ("pl.slice(x, [16], [0])", [16], (32,)),
        ("pl.reshape(x, [4, 8])", [4, 8], (32,)),
        ("x[1:]", [31], (32,)),
        ("pl.reshape(pl.slice(x, [16], [0]), [4, 4])", [4, 4], (32,)),
        ("pl.slice(pl.reshape(x, [4, 8]), [2, 8], [0, 0])", [2, 8], (32,)),
        ("pl.reshape(pl.slice(x, [16, 16], [0, 0]), [256])", [256], (32, 16)),
    ],
)
@pytest.mark.parametrize("keyword", [False, True])
def test_direct_view_return_flows_to_dependency(tmp_path, view, expected, input_shape, keyword):
    native = tmp_path / "view.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    path = tmp_path / "view_case.py"
    path.write_text(f"""import pypto.language as pl
@pl.jit.extern(source={str(native)!r}, core_type="aiv")
def external(x: pl.Tensor):
    return x
@pl.jit.inline
def consume(x: pl.Tensor):
    return x
@pl.jit
def entry(x: pl.Tensor):
    value = external({"x=" if keyword else ""}{view})
    return consume(value)
""")
    spec = importlib.util.spec_from_file_location("view_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program = module.entry.specialize(torch.empty(input_shape))
    consumer = program.get_function("consume")
    assert consumer is not None
    assert [dim.value for dim in consumer.params[0].type.shape] == expected


def test_sample_scalars_flow_through_bare_helper_and_external(tmp_path):
    native = tmp_path / "scalar.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')

    @pl.jit.extern(source=native, core_type="aiv")
    def external(x: pl.Tensor, count: pl.Scalar):
        return x

    @pl.jit.inline
    def helper(x: pl.Tensor, count: pl.Scalar):
        return external(x, count)

    @pl.jit
    def entry(x: pl.Tensor, n: pl.Scalar[pl.INT32]):
        return helper(x, n)

    sample = torch.empty(32)
    first = entry.specialize(sample, 1)
    second = entry.specialize(sample, 7)
    assert ir.structural_equal(first, second)
    for name in ("external", "helper"):
        function = first.get_function(name)
        assert function is not None
        assert function.params[1].type.dtype == DataType.INT32
    assert isinstance(entry.lower(sample, 1), ir.Program)


@pytest.mark.parametrize("annotation", ["", ": int"])
@pytest.mark.parametrize("return_annotation", ["", " -> pl.Tensor[[32], pl.FP32]"])
def test_missing_external_annotation_reports_parameter(tmp_path, annotation, return_annotation):
    native = tmp_path / "invalid.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    path = tmp_path / "invalid_case.py"
    path.write_text(f"""import pypto.language as pl
@pl.jit.extern(source={str(native)!r}, core_type="aiv")
def external(x{annotation}){return_annotation}:
    ...
@pl.jit
def entry(x: pl.Tensor):
    external(x)
    return x
""")
    spec = importlib.util.spec_from_file_location("invalid_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="External function 'external' parameter 'x'.*annotation"):
        module.entry.specialize(torch.empty(32))


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("alias, other = state, x", {"alias": "state", "other": "x"}),
        ("[alias, other] = [state, x]", {"alias": "state", "other": "x"}),
        ("alias, (other, third) = state, (x, state)", {"alias": "state", "other": "x", "third": "state"}),
        ("state, x = x, state", {"state": "x", "x": "state"}),
        ("whole = alias, other = state, x", {"whole": ("state", "x"), "alias": "state", "other": "x"}),
        ("alias = pl.reshape(input=state, shape=[32])", {"alias": "state"}),
        ("alias = pl.store(x, [0], state)", {"alias": "state"}),
        ("alias = pl.tile.store(x, [0], output_tensor=state)", {"alias": "state"}),
        ("alias = pl.tensor.assemble(target=state, source=x, offsets=[0])", {"alias": "state"}),
    ],
)
def test_alias_assignment_snapshots_each_operand(statement, expected):
    tracker = AliasTracker(["state", "x"], lambda: "allocation")
    tracker.record_assignment(ast.parse(statement).body[0])
    for name, value in expected.items():
        if isinstance(value, tuple):
            assert tracker.aliases[name] == value
        else:
            assert tracker.root_of(ast.Name(id=name, ctx=ast.Load())) == value


@pytest.mark.parametrize(
    "body, expected",
    [
        ("pair = ((state, x), x)\nalias = pair[0][0]", "state"),
        ("pair = (state, x)\nstate = x\nalias = pair[0]", "state"),
        ("pair = (state, x)\npair = (x, state)\nalias = pair[0]", "x"),
        ("pair = (state, x)\nalias = pair[2]", None),
        ("pair = (state, x)\nalias = pair[index]", None),
        ("pair = [state, x]\nalias = pair[0]", None),
        ("pair = ([state, x], x)\nalias = pair[0][0]", None),
    ],
)
def test_tuple_alias_projection_preserves_snapshots(body, expected):
    tracker = AliasTracker(["state", "x"], lambda: "allocation")
    for statement in ast.parse(body).body:
        tracker.record_assignment(statement)
    assert tracker.root_of(ast.Name(id="alias", ctx=ast.Load())) == expected


def test_legacy_inout_stub_assignment(tmp_path):
    native = tmp_path / "inout.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')

    @pl.jit.extern(source=native, core_type="aiv")
    def external(cache: pl.InOut[pl.Tensor], scales: pl.InOut[pl.Tensor]) -> pl.Tensor: ...

    @pl.jit.inline
    def consume(x: pl.Tensor):
        return x

    @pl.jit
    def entry(cache: pl.InOut[pl.Tensor], scales: pl.InOut[pl.Tensor]):
        copied = external(cache, scales)
        copied = consume(copied)
        return copied, scales

    assert isinstance(entry.specialize(torch.empty(32), torch.empty(16)), ir.Program)


@pytest.mark.parametrize("core", ["aiv", "mixed"])
def test_hidden_writeback_preserves_dynamic_argument_type(tmp_path, core):
    native = tmp_path / "dynamic.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    dim = pl.dynamic("Rows")

    decorator = (
        pl.jit.extern(source=native, core_type=core)
        if core == "aiv"
        else pl.jit.extern(aic_source=native, aiv_source=native, core_type=core)
    )

    @decorator
    def external(x: pl.Tensor, out: pl.InOut[pl.Tensor], indices: pl.Tensor): ...

    @pl.jit.inline(auto_scope=False)
    def helper(x: pl.Tensor[[dim, 16], pl.FP32], out: pl.Tensor[[dim, 16], pl.FP32]):
        rows = pl.tensor.dim(x, 0)
        indices = pl.create_tensor([rows], dtype=pl.INT32)
        with pl.spmd(1):
            external(x, out, indices)
        return out

    @pl.jit
    def entry(x: pl.Tensor[[dim, 16], pl.FP32], out: pl.InOut[pl.Tensor[[dim, 16], pl.FP32]]):
        return helper(x, out)

    assert isinstance(entry.specialize(), ir.Program)


@pytest.mark.parametrize("shape", ["a.shape", "[a.shape[0], 16]", "[pl.tensor.dim(a, 0), 16]"])
@pytest.mark.parametrize("operation", ["slice", "reshape"])
def test_chained_view_extent_is_valid_at_external_call(tmp_path, shape, operation):
    # The parser keeps descriptor reads distinct from the scalar n in a's type.
    native = tmp_path / "chained.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    path = tmp_path / "chained_case.py"
    offsets = ", [0, 0]" if operation == "slice" else ""
    path.write_text(f"""import pypto.language as pl
@pl.jit.extern(source={str(native)!r}, core_type="aiv")
def external(a: pl.Tensor, b: pl.Tensor):
    return a
@pl.jit
def entry(x: pl.Tensor[[32, 16], pl.FP32], n: pl.Scalar[pl.INDEX]):
    a = pl.slice(x, [n, 16], [0, 0])
    b = pl.{operation}(a, {shape}{offsets})
    return external(a, b)
""")
    spec = importlib.util.spec_from_file_location("chained_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module.entry.specialize(), ir.Program)


def test_views_of_same_runtime_count_share_extent(tmp_path):
    native = tmp_path / "extent.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    dim = pl.dynamic("Rows")

    @pl.jit.extern(source=native, core_type="aiv")
    def external(a: pl.Tensor, b: pl.Tensor):
        return a

    @pl.jit.inline
    def helper(a: pl.Tensor[[dim, 16], pl.FP32], b: pl.Tensor[[dim, 16], pl.FP32]):
        return external(a, b)

    @pl.jit.inline
    def intermediate(a: pl.Tensor, b: pl.Tensor):
        return helper(a, b)

    @pl.jit
    def entry(x: pl.Tensor, n: pl.Scalar[pl.INT32]):
        count = pl.min(n, 16)
        a = pl.slice(x, [count, 16], [0, 0])
        b = pl.slice(x, [count, 16], [0, 0])
        return intermediate(a, b)

    _, _, tensors, scalars, constants, dynamics = entry._bind_args((torch.empty(32, 16), 8), {})
    metas = extract_local_tensor_metas(entry._func, tensors, discover_deps=_discover_dep_bindings)
    assert metas["a"].actual_dim(0) is metas["b"].actual_dim(0)
    assert isinstance(entry.specialize(torch.empty(32, 16), 8), ir.Program)


def test_views_of_same_computed_dimension_share_extent(tmp_path):
    """A runtime division has one value even when it sizes different source views."""
    native = tmp_path / "computed_extent.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    rows = pl.dynamic("Rows")

    @pl.jit.extern(source=native, core_type="aiv")
    def external(a: pl.Tensor, b: pl.Tensor):
        return a

    @pl.jit.inline
    def helper(a: pl.Tensor[[rows, 16], pl.FP32], b: pl.Tensor[[rows, 16], pl.FP32]):
        return external(a, b)

    @pl.jit
    def entry(x: pl.Tensor[[rows, 16], pl.FP32], y: pl.Tensor[[rows, 16], pl.FP32]):
        count = pl.tensor.dim(x, 0) // 2
        a = pl.slice(x, [count, 16], [0, 0])
        b = pl.slice(y, [count, 16], [0, 0])
        return helper(a, b)

    assert isinstance(entry.specialize(), ir.Program)


@pytest.mark.parametrize(
    "rebind",
    ["count = other", "if flag:\n        count = other", "for i in pl.range(2):\n        count = other"],
)
def test_rebound_runtime_count_does_not_equate_view_extents(tmp_path, rebind):
    native = tmp_path / "independent.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    path = tmp_path / "independent_case.py"
    path.write_text(f"""import pypto.language as pl
@pl.jit.extern(source={str(native)!r}, core_type="aiv")
def external(a: pl.Tensor, b: pl.Tensor):
    return a
@pl.jit
def entry(x: pl.Tensor, count: pl.Scalar[pl.INT32], other: pl.Scalar[pl.INT32], flag: pl.Scalar[pl.BOOL]):
    a = pl.slice(x, [count, 16], [0, 0])
    {rebind}
    b = pl.slice(x, [count, 16], [0, 0])
    return external(a, b)
""")
    spec = importlib.util.spec_from_file_location("independent_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, _, tensors, scalars, constants, dynamics = module.entry._bind_args(
        (torch.empty(32, 16), 8, 12, True), {}
    )
    contexts = module.entry._build_contexts(tensors, scalars, constants, dynamics)
    context = next(ctx for ctx in contexts if ctx.source_def_name == "external")
    request = request_from_context(context, CompileTarget("a5", "test", "1"))
    left, right = (parameter.type for parameter in request.parameters)
    assert isinstance(left, TensorSpec) and isinstance(right, TensorSpec)
    assert left.shape == ("d0", 16) and right.shape == ("d1", 16)


@pytest.mark.parametrize("literal", [False, True], ids=["inherited-type", "literal"])
def test_helper_declared_scalar_type_reaches_external(tmp_path, literal):
    native = tmp_path / "declared.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    path = tmp_path / "declared_case.py"
    arguments = ("1", "2") if literal else ("n", "n")
    path.write_text(f"""import pypto.language as pl
@pl.jit.extern(source={str(native)!r}, core_type="aiv")
def external(x: pl.Tensor, count: pl.Scalar):
    return x
@pl.jit.inline
def helper(x: pl.Tensor, count: pl.Scalar[pl.INDEX]):
    return external(x, count)
@pl.jit
def entry(x: pl.Tensor, n: pl.Scalar[pl.INT32]):
    first = helper(x, {arguments[0]})
    second = helper(x, {arguments[1]})
    return first, second
""")
    spec = importlib.util.spec_from_file_location("declared_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, _, tensors, scalars, constants, dynamics = module.entry._bind_args((torch.empty(32), 1), {})
    contexts = module.entry._build_contexts(tensors, scalars, constants, dynamics)
    for name in ("helper", "external"):
        variants = [ctx for ctx in contexts if ctx.source_def_name == name]
        assert len(variants) == 1
        assert variants[0].scalar_dtypes == {"count": DataType.INDEX}
    program = module.entry.specialize(torch.empty(32), 1)
    external = program.get_function("external")
    assert external is not None
    assert external.params[1].type.dtype == DataType.INDEX


def test_constexpr_arguments_have_only_compile_time_bindings(tmp_path):
    native = tmp_path / "constant.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')

    @pl.jit.extern(source=native, core_type="aiv")
    def external(x: pl.Tensor, count: pl.constexpr):
        return x

    @pl.jit.inline
    def helper(x: pl.Tensor, count: pl.constexpr):
        return external(x, count)

    @pl.jit
    def entry(x: pl.Tensor):
        first = external(x, 7)
        second = helper(x, count=8)
        return first, second

    _, _, tensors, scalars, constants, dynamics = entry._bind_args((torch.empty(32),), {})
    contexts = entry._build_contexts(tensors, scalars, constants, dynamics)
    values = set()
    for context in contexts:
        assert "count" not in context.scalar_dtypes
        if context.source_def_name == "external":
            request = request_from_context(context, CompileTarget("a5", "test", "1"))
            values.add(dict(request.constants)["count"])
    assert values == {7, 8}
    assert isinstance(entry.specialize(torch.empty(32)), ir.Program)


@pytest.mark.parametrize("extent", ["Rows", "32"])
def test_external_shared_extent_survives_nested_inline_reads(tmp_path, extent):
    """Separate dim reads of one extent must agree when hidden writes are inlined."""
    native = tmp_path / "dimensions.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    path = tmp_path / "dimensions_case.py"
    path.write_text(f"""import pypto.language as pl
Rows = pl.dynamic("Rows")
@pl.jit.extern(source={str(native)!r}, core_type="aiv")
def external(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    ...
@pl.jit.inline
def helper(x: pl.Tensor, other: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    out = pl.create_tensor([rows, 16], dtype=pl.FP32)
    external(other, out)
    return out
@pl.jit.inline
def outer(x: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    other = pl.create_tensor([rows, 16], dtype=pl.FP32)
    return helper(x, other)
@pl.jit
def entry(x: pl.Tensor[[{extent}, 16], pl.FP32]):
    return outer(x)
""")
    spec = importlib.util.spec_from_file_location("dimensions_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program = module.entry.specialize()
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    code = generate(after, str(tmp_path / "generated"), skip_ptoas=True)["orchestration/entry.cpp"]
    assert "add_output" in code


@pytest.mark.parametrize(
    "setup, argument, dtype",
    [
        ("", "1", DataType.INDEX),
        ("", "1.5", DataType.FP32),
        ("", "True", DataType.BOOL),
        ("", "n + 1", DataType.INDEX),
        ("count = n + 1", "count", DataType.INDEX),
        ("count: pl.Scalar[pl.INT64] = n + 1", "count", DataType.INT64),
        ("count = pl.min(n, 16)", "count", DataType.INDEX),
        ("count = pl.tensor.dim(x, 0)", "count", DataType.INDEX),
        ("count = pl.read(x, [0])", "count", DataType.FP32),
        ("count = pl.read(src=x, offset=[0])", "count", DataType.FP32),
        ("count = pl.tensor.read(tensor=x, indices=[0])", "count", DataType.FP32),
        ("", "pl.const(1, pl.INT32)", DataType.INT32),
        ("count = pl.const(1, pl.INT32)", "count", DataType.INT32),
        ("", "pl.const(-1.5, pl.FP16)", DataType.FP16),
        ("count = pl.const(-1.5, pl.FP16)", "count", DataType.FP16),
        ("if flag:\n    count = n + 1", "n", DataType.INT32),
        ("if flag:\n    count = n + 1\nelse:\n    count = n - 1", "count", DataType.INDEX),
        (
            "count = pl.cast(n, pl.INDEX)\nfor i in pl.range(2):\n    count = count + 1",
            "count",
            DataType.INDEX,
        ),
    ],
)
def test_bare_external_scalar_expression_types(tmp_path, setup, argument, dtype):
    native = tmp_path / "scalar_expression.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    path = tmp_path / "scalar_expression_case.py"
    path.write_text(f"""import pypto.language as pl
@pl.jit.extern(source={str(native)!r}, core_type="aiv")
def external(x: pl.Tensor, count: pl.Scalar):
    return x
@pl.jit
def entry(x: pl.Tensor, n: pl.Scalar[pl.INT32], flag: pl.Scalar[pl.BOOL]):
{textwrap.indent(setup, "    ")}
    return external(x, {argument})
""")
    spec = importlib.util.spec_from_file_location("scalar_expression_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program = module.entry.specialize(torch.empty(32), 4, True)
    external = program.get_function("external")
    assert external is not None
    assert external.params[1].type.dtype == dtype


@pytest.mark.parametrize("declaration", ["...", "return out"])
def test_prepared_contract_cannot_change_declared_return_mapping(tmp_path, declaration):
    native = tmp_path / "contract_return.cpp"
    native.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    path = tmp_path / "contract_return_case.py"
    path.write_text(f"""import pypto.language as pl
@pl.jit.extern(source={str(native)!r}, core_type="aiv")
def external(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    {declaration}
@pl.jit
def entry(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    return external(x, out)
""")
    spec = importlib.util.spec_from_file_location("contract_return_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, _, tensors, scalars, constants, dynamics = module.entry._bind_args(
        (torch.empty(16), torch.empty(32)), {}
    )
    contexts = module.entry._build_contexts(tensors, scalars, constants, dynamics)
    external = next(ctx for ctx in contexts if ctx.source_def_name == "external")
    external.external_contract = KernelContract(
        (
            ParameterSpec("x", TensorSpec((16,), DataType.FP32)),
            ParameterSpec("out", TensorSpec((32,), DataType.FP32)),
        ),
        (TensorEffect(0, Access.READ), TensorEffect(1, Access.WRITE, Coverage.FULL)),
        public_returns=(ReturnBinding(0),),
        writebacks=(WritebackBinding(1),),
    )
    with pytest.raises(ValueError, match="contract public returns differ from its declaration"):
        Specializer("ContractReturns", contexts).specialize()

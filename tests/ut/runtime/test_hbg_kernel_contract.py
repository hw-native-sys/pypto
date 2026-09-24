# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""HBG kernel Host orchestration cannot dereference Tensor storage."""

import importlib
import sys
from enum import Enum
from types import SimpleNamespace

import pypto.language as pl
import pytest
from pypto import DataType
from pypto.ir._kernel_compile import kernel_abi_for_program, validate_hbg_kernel_orchestration
from pypto.pypto_core import ir, passes


@pytest.fixture
def stub_codegen_dependencies(monkeypatch):
    # Preserve the real pass pipeline, Host codegen, config and ABI stamping.
    # Generated configs only need this enum from the optional runtime SDK.
    task_interface = SimpleNamespace(ArgDirection=Enum("ArgDirection", ["SCALAR", "IN", "OUT", "INOUT"]))
    monkeypatch.setitem(sys.modules, "simpler", SimpleNamespace(task_interface=task_interface))
    monkeypatch.setitem(sys.modules, "simpler.task_interface", task_interface)
    # Replace the external Device C++ translator for hardware-free UTs.
    monkeypatch.setattr(
        "pypto.backend.pto_backend._compile_pto_module",
        lambda _code, _name, _directory, _planner=None: '#include "pto/pto-inst.hpp"\n'
        "using namespace pto;\n"
        "__global__ AICORE void stub_kernel(__gm__ float* v1) {}\n",
    )


@pytest.fixture(params=["read", "write"])
def access_program(request):
    @pl.program
    class Read:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[8], pl.INT32]) -> pl.Tensor[[8], pl.INT32]:
            n: pl.Scalar[pl.INT32] = pl.tensor.read(x, [0])
            if n > 0:
                pl.tensor.write(x, [1], n)
            return x

    @pl.program
    class Write:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[8], pl.INT32]) -> pl.Tensor[[8], pl.INT32]:
            value: pl.Scalar[pl.INT32] = 7
            pl.tensor.write(x, [0], value)
            return x

    return (Read if request.param == "read" else Write), request.param


@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_kernel_abi_derivation_does_not_restrict_host_access(access_program, runtime):
    program, _ = access_program
    abi = kernel_abi_for_program(program, platform="a2a3", runtime=runtime)
    assert abi.runtime == runtime
    assert abi.return_aliases == (0,)


def test_direct_kernel_compile_cannot_bypass_validation(access_program, tmp_path, monkeypatch):
    program, operation = access_program
    abi = kernel_abi_for_program(program, platform="a2a3", runtime="host_build_graph")
    compiler = importlib.import_module("pypto.ir.compile")

    def unexpected_pipeline(*args, **kwargs):
        pytest.fail("Invalid Host access reached the optimization pipeline")

    monkeypatch.setattr(compiler, "_run_pass_pipeline", unexpected_pipeline)
    with pytest.raises(ValueError, match=f"cannot use tensor.{operation}") as caught:
        compiler._compile_impl(program, output_dir=str(tmp_path), _kernel_abi=abi)
    message = str(caught.value)
    assert "test_hbg_kernel_contract.py:" in message
    assert "'main'" in message
    assert ("explicit scalar argument" if operation == "read" else "Device task") in message


@pytest.mark.parametrize("runtime", list(passes.RuntimeKind))
def test_program_compilation_keeps_tensor_access(access_program, tmp_path, runtime):
    program, _ = access_program
    compiler = importlib.import_module("pypto.ir.compile")
    with passes.PassContext([], runtime=runtime):
        compiled = compiler.compile(program, output_dir=str(tmp_path), skip_ptoas=True)
    assert compiled is not None
    sources = list(tmp_path.rglob("*.cpp"))
    assert any("set_tensor_data" in source.read_text() for source in sources)


@pytest.mark.parametrize("call_kind", ["call", "submit"])
@pytest.mark.parametrize("function_type", [ir.FunctionType.Inline, ir.FunctionType.Graph])
def test_reachable_host_helpers_are_checked(access_program, call_kind, function_type):
    program, operation = access_program
    original = program.get_function("main")
    helper = ir.Function(
        "helper", original.params, original.return_types, original.body, original.span, type=function_type
    )
    span = original.span
    target = ir.GlobalVar("helper")
    if call_kind == "call":
        call = ir.Call(target, original.params, span)
    else:
        call = ir.Submit(
            target,
            original.params,
            [],
            ir.TupleType([*original.return_types, ir.ScalarType(DataType.TASK_ID)]),
            span,
        )
    entry = ir.Function(
        "main", original.params, [], ir.EvalStmt(call, span), span, type=ir.FunctionType.Orchestration
    )
    with pytest.raises(ValueError, match=f"'helper' cannot use tensor.{operation}"):
        validate_hbg_kernel_orchestration(ir.Program([entry, helper], "Helpers", span))


@pytest.mark.parametrize("function_type", [ir.FunctionType.InCore, ir.FunctionType.AIC, ir.FunctionType.AIV])
def test_device_function_bodies_are_not_host_access(access_program, function_type):
    program, _ = access_program
    original = program.get_function("main")
    device = ir.Function(
        "device", original.params, original.return_types, original.body, original.span, type=function_type
    )
    call = ir.Call(ir.GlobalVar("device"), original.params, original.span)
    entry = ir.Function(
        "main",
        original.params,
        [],
        ir.EvalStmt(call, original.span),
        original.span,
        type=ir.FunctionType.Orchestration,
    )
    validate_hbg_kernel_orchestration(ir.Program([entry, device], "Device", original.span))


def test_unreachable_host_helper_does_not_restrict_entry(access_program):
    program, _ = access_program
    original = program.get_function("main")
    helper = ir.Function(
        "unused",
        original.params,
        original.return_types,
        original.body,
        original.span,
        type=ir.FunctionType.Inline,
    )
    entry = ir.Function(
        "main",
        original.params,
        original.return_types,
        ir.ReturnStmt(original.params, original.span),
        original.span,
        type=ir.FunctionType.Orchestration,
    )
    validate_hbg_kernel_orchestration(ir.Program([entry, helper], "Unused", original.span))


@pytest.mark.parametrize("access_program", ["read"], indirect=True)
def test_device_call_arguments_are_evaluated_on_host(access_program):
    program, _ = access_program
    original = program.get_function("main")
    calls = []

    class Reads(ir.IRVisitor):
        def visit_call(self, op):
            if op.op.name == ir.get_op("tensor.read").name:
                calls.append(op)
            super().visit_call(op)

    Reads().visit_stmt(original.body)
    assert len(calls) == 1
    span = original.span
    device = ir.Function("device", [], [], ir.SeqStmts([], span), span, type=ir.FunctionType.InCore)
    call = ir.Call(ir.GlobalVar("device"), [calls[0]], span)
    entry = ir.Function(
        "main", original.params, [], ir.EvalStmt(call, span), span, type=ir.FunctionType.Orchestration
    )
    with pytest.raises(ValueError, match="cannot use tensor.read"):
        validate_hbg_kernel_orchestration(ir.Program([entry, device], "Arguments", span))


def test_device_scope_is_allowed_but_runtime_branch_is_checked():
    @pl.program
    class Device:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[8], pl.INT32]) -> pl.Tensor[[8], pl.INT32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                value: pl.Scalar[pl.INT32] = 7
                pl.tensor.write(x, [0], value)
            return x

    validate_hbg_kernel_orchestration(Device)

    @pl.program
    class Conditional:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[8], pl.INT32], flag: pl.Scalar[pl.INT32]) -> pl.Tensor[[8], pl.INT32]:
            if flag > 0:
                value: pl.Scalar[pl.INT32] = 7
                pl.tensor.write(x, [0], value)
            return x

    with pytest.raises(ValueError, match="cannot use tensor.write"):
        validate_hbg_kernel_orchestration(Conditional)


def test_scalar_graph_decision_and_tensor_metadata_are_allowed():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[8], pl.INT32], count: pl.Scalar[pl.INT32]) -> pl.Tensor[[8], pl.INT32]:
            size: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
            for i in pl.range(count):
                if i < size:
                    with pl.at(level=pl.Level.CORE_GROUP):
                        value: pl.Scalar[pl.INT32] = 7
                        pl.tensor.write(x, [i], value)
            return x

    abi = kernel_abi_for_program(Program, platform="a2a3", runtime="host_build_graph")
    assert [p.shape for p in abi.parameters] == [(8,), None]
    assert abi.return_aliases == (0,)


@pytest.mark.parametrize("form", ["scope", "submit"])
def test_device_dispatch_predicate_is_allowed(tmp_path, form, stub_codegen_dependencies):
    @pl.program
    class Scope:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            control: pl.Tensor[[1], pl.INT32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.manual_scope():
                with pl.at(level=pl.Level.CORE_GROUP, predicate=(control[0] > 0)):
                    pl.store(pl.load(x, [0, 0], [16, 16]), [0, 0], out)
            return out

    @pl.program
    class Submit:
        @pl.function(type=pl.FunctionType.InCore)
        def copy(
            self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            out = pl.store(pl.load(x, [0, 0], [16, 16]), [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            control: pl.Tensor[[1], pl.INT32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.spmd_submit(self.copy, x, out, core_num=1, predicate=(control[0] > 0))
            return out

    program = Scope if form == "scope" else Submit
    abi = kernel_abi_for_program(program, platform="a2a3", runtime="host_build_graph")
    compiler = importlib.import_module("pypto.ir.compile")
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        compiler._compile_impl(
            program,
            output_dir=str(tmp_path),
            platform="a2a3",
            skip_ptoas=False,
            _kernel_abi=abi,
        )
    sources = "\n".join(source.read_text() for source in tmp_path.rglob("*.cpp"))
    assert ".set_predicate(" in sources
    assert "get_tensor_data" not in sources


@pytest.mark.parametrize("name", ["add", "repeat_add", "read_on_host"])
def test_hbg_integration_kernels_compile(tmp_path, monkeypatch, name, stub_codegen_dependencies):
    from pypto.ir import _kernel_compile  # noqa: PLC0415

    from tests.st.runtime.kernel import test_hbg_contract  # noqa: PLC0415

    program = getattr(test_hbg_contract, name).specialize()
    if name == "read_on_host":
        # Match the hardware negative test: preserve the actual runtime accessor.
        monkeypatch.setattr(_kernel_compile, "validate_hbg_kernel_orchestration", lambda program: None)
    abi = kernel_abi_for_program(program, platform="a2a3", runtime="host_build_graph")
    compiler = importlib.import_module("pypto.ir.compile")
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        artifact = compiler._compile_impl(
            program,
            output_dir=str(tmp_path),
            platform="a2a3",
            skip_ptoas=False,
            _kernel_abi=abi,
        )
    assert artifact.kernel_abi == abi
    sources = "\n".join(source.read_text() for source in tmp_path.rglob("*.cpp"))
    assert "pypto_orchestration_requirements_v1" not in sources
    if name == "read_on_host":
        assert "get_tensor_data<int32_t>" in sources
    elif name == "repeat_add":
        assert "orch_args.scalar<int32_t>" in sources
        assert "get_tensor_data" not in sources


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

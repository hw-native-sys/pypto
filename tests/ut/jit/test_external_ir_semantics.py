# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Executable IR prototypes for the external contract, without a DSL compiler.

The generated source is the normalized form that task 34.3 must produce. It
uses the same result projections for public values and hidden tensor writes.
No runtime execution is claimed by these CPU pass/codegen checks.
"""

import re
import textwrap

import pypto.language as pl
import pytest
from pypto import passes
from pypto.backend.pto_backend import generate
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit.external.contracts import (
    Access,
    Coverage,
    KernelContract,
    ParameterSpec,
    ReturnBinding,
    TensorEffect,
    TensorSpec,
    WritebackBinding,
)
from pypto.jit.external.normalization import plan_call
from pypto.pypto_core import DataType, ir

_TENSOR = "pl.Tensor[[32], pl.FP32]"


def _contract():
    return KernelContract(
        tuple(ParameterSpec(name, TensorSpec((32,), DataType.FP32)) for name in ("x", "state", "out")),
        (
            TensorEffect(0, Access.READ),
            TensorEffect(1, Access.READ_WRITE, Coverage.PARTIAL),
            TensorEffect(2, Access.WRITE, Coverage.FULL),
        ),
        public_returns=(ReturnBinding(2),),
        writebacks=(WritebackBinding(1), WritebackBinding(2)),
    )


def _declaration(name, source, core, contract):
    plan = plan_call(contract)
    names = [parameter.name for parameter in contract.parameters]
    annotations = []
    for parameter, direction in zip(contract.parameters, plan.parameter_directions):
        spec = parameter.type
        tensor = f"pl.Tensor[{list(spec.shape)}, pl.{spec.dtype.to_string().upper()}]"
        annotation = tensor if direction == ir.ParamDirection.In else f"pl.{direction.name}[{tensor}]"
        annotations.append(f"{parameter.name}: {annotation}")
    returns = ", ".join(
        f"pl.Tensor[{list(contract.parameters[index].type.shape)}, pl.FP32]"
        for index in plan.result_parameters
    )
    result_type = f"tuple[{returns}]" if len(plan.result_parameters) > 1 else returns
    result_names = ", ".join(names[index] for index in plan.result_parameters)
    return f"""
    @pl.function(type=pl.FunctionType.{core}, external_source={str(source)!r})
    def {name}(self, {", ".join(annotations)}) -> {result_type}:
        pl.func_attr({{"external_return_params": [{result_names}]}})
        ...
"""


def _program(tmp_path, variant, core="AIV"):
    source = tmp_path / "external.cpp"
    source.write_text('extern "C" void kernel_entry(long long* args) {}\n')
    contract = _contract()
    plan = plan_call(contract)
    declarations = _declaration("producer", source, core if core != "MIX" else "AIC", contract)
    if core == "MIX":
        declarations = _declaration("producer_aic", source, "AIC", contract)
        declarations += _declaration("producer_aiv", source, "AIV", contract)
        declarations += f"""
    @pl.function(type=pl.FunctionType.Group)
    def producer(self, x: {_TENSOR}, state: pl.InOut[{_TENSOR}],
                 out: pl.Out[{_TENSOR}]) -> tuple[{_TENSOR}, {_TENSOR}]:
        updated = self.producer_aic(x, state, out)
        self.producer_aiv(x, state, out)
        return updated
"""
    # Public and writeback slot selection comes from the plan, not Out ordering.
    declarations += f"""
    @pl.function(type=pl.FunctionType.Inline)
    def helper(self, x: {_TENSOR}, state: pl.InOut[{_TENSOR}],
               out: pl.Out[{_TENSOR}]) -> tuple[{_TENSOR}, {_TENSOR}]:
        updated = self.producer(x, state, out)
        return updated[{plan.public_slots[0]}], updated[{dict(plan.writeback_slots)[1]}]

    @pl.function(type=pl.FunctionType.AIV, external_source={str(source)!r})
    def consumer(self, state: {_TENSOR}, data: {_TENSOR}, result: pl.Out[{_TENSOR}]) -> {_TENSOR}:
        pl.func_attr({{"external_return_params": [result]}})
        ...
"""
    bodies = {
        "straight": "out, state = self.helper(x, state, out)",
        "loop": "for i in pl.range(2):\n    out, state = self.helper(x, state, out)",
        "zero_trip": "for i in pl.range(0):\n    out, state = self.helper(x, state, out)",
        "branch": "if flag:\n    out, state = self.helper(x, state, out)",
        "nested": "for i in pl.range(2):\n    if flag:\n        out, state = self.helper(x, state, out)",
    }
    body = textwrap.indent(bodies[variant], "        ")
    declarations += f"""
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: {_TENSOR}, state: pl.InOut[{_TENSOR}], out: pl.InOut[{_TENSOR}],
             result: pl.Out[{_TENSOR}], flag: pl.Scalar[pl.BOOL]
             ) -> tuple[{_TENSOR}, {_TENSOR}, {_TENSOR}]:
{body}
        result = self.consumer(state, out, result)
        return result, state, out
"""
    program = pl.parse("import pypto.language as pl\n@pl.program\nclass Prototype:\n" + declarations)
    assert isinstance(program, ir.Program)
    return program


def _round_trip(program):
    for restored in (pl.parse(str(program)), ir.deserialize(ir.serialize(program))):
        assert isinstance(restored, ir.Program)
        ir.assert_structural_equal(program, restored)


@pytest.mark.parametrize("variant", ["straight", "loop", "zero_trip", "branch", "nested"])
@pytest.mark.parametrize("core", ["AIC", "AIV", "MIX"])
def test_return_and_writeback_survive_full_pipeline(tmp_path, variant, core):
    program = _program(tmp_path, variant, core)
    _round_trip(program)
    main = program.get_function("main")
    assert main is not None
    assert ir.returned_param_indices(main, program) == [3, 1, 2]
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    files = generate(after, str(tmp_path / "generated"), skip_ptoas=True)
    config = files["kernel_config.py"]
    assert "'external': True" in config or '"external": True' in config
    code = files["orchestration/main.cpp"]
    # Memory annotation can make whole-parameter lineage conservative across
    # loop yields. Check the actual storage passed to the downstream task.
    consumer = re.split(r"// Task \d+: consumer", code)[-1]
    inputs = re.findall(r"\.add_input\((\w+)\)", consumer)
    assert len(inputs) == 2
    for name, argument in zip(("state", "out"), inputs):
        if variant == "branch":
            # Both the taken and untaken branch retain the same allocation.
            assert code.count(f"{argument} = ext_{name};") == 2
        elif variant == "nested":
            assert f"Tensor {argument} = ext_{name};" in code
            carried = re.findall(rf"(\w+) = {re.escape(argument)};", code)
            assert len(carried) == 2 and carried[0] == carried[1]
            assert f"{argument} = {carried[0]};" in code
        else:
            assert argument == f"ext_{name}"
    assert "add_output(ext_result)" in code
    if variant == "zero_trip":
        assert "// Task 0: consumer" in code
        assert "add_inout" not in code
    else:
        state = inputs[0] if variant == "nested" else "ext_state"
        assert f"add_inout({state})" in code
    if variant != "zero_trip" and core == "MIX":
        assert "MixedKernels" in code


def test_view_write_propagates_dependency_to_root(tmp_path):
    """Writing a subview must order a subsequent read of the whole allocation."""
    source = tmp_path / "external.cpp"
    source.write_text('extern "C" void kernel_entry(long long* args) {}\n')

    Views = pl.parse(f"""
import pypto.language as pl
@pl.program
class Views:
    @pl.function(type=pl.FunctionType.AIV, external_source={str(source)!r})
    def update(self, view: pl.InOut[pl.Tensor[[16], pl.FP32]]) -> pl.Tensor[[16], pl.FP32]:
        pl.func_attr({{"external_return_params": [view]}})
        ...

    @pl.function(type=pl.FunctionType.AIV, external_source={str(source)!r})
    def consume(
        self, root: pl.Tensor[[32], pl.FP32], out: pl.Out[pl.Tensor[[32], pl.FP32]]
    ) -> pl.Tensor[[32], pl.FP32]:
        pl.func_attr({{"external_return_params": [out]}})
        ...

    @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
    def main(
        self, root: pl.InOut[pl.Tensor[[32], pl.FP32]], out: pl.Out[pl.Tensor[[32], pl.FP32]]
    ) -> pl.Tensor[[32], pl.FP32]:
        with pl.scope(mode=pl.ScopeMode.AUTO):
            view = pl.slice(root, [16], [8])
            updated, writer = pl.submit(self.update, view)
            result, reader = pl.submit(self.consume, root, out)
        return result

""")
    assert isinstance(Views, ir.Program)

    _round_trip(Views)
    dependencies = passes.auto_derive_task_dependencies(analyze_auto_scopes=True)(
        passes.derive_call_directions()(Views)
    )
    consumers: list[ir.Submit] = []

    class CollectConsumer(ir.IRVisitor):
        def visit_submit(self, op):
            if op.op.name == "consume":
                consumers.append(op)
            super().visit_submit(op)

    CollectConsumer().visit_program(dependencies)
    assert len(consumers) == 1
    edges = consumers[0].attrs.get("compiler_manual_dep_edges", [])
    assert [edge.name_hint for edge in edges] == ["writer"]
    # All known hazards are covered by this explicit edge, so runtime input
    # tracking can be removed even though the producer writes only a subview.
    assert consumers[0].arg_directions[0] == ir.ArgDirection.NoDep
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Views)
    files = generate(after, str(tmp_path / "generated"), skip_ptoas=True)
    assert "add_inout" in files["orchestration/main.cpp"]
    update = after.get_function("update")
    assert update is not None
    assert ir.returned_param_indices(update, after) == [0]
    result_type = update.return_types[0]
    assert isinstance(result_type, ir.TensorType)
    assert str(result_type.shape[0]) == "16"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

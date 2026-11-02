# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile computed Group outputs through the complete CPU codegen pipeline.

These checks do not execute hardware or establish numerical correctness. Group
member order does not synchronize its co-scheduled cores; an executable version
also needs an explicit cross-core protocol around the materialized store/load.
"""

import re

import pypto.language as pl
import pytest
import torch
from pypto import backend, ir
from pypto.backend import BackendType, pto_backend
from pypto.ir import OptimizationStrategy, PassManager


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


class _CallCollector(ir.IRVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.launches: list[ir.Call | ir.Submit] = []
        self.allocations: list[ir.AssignStmt] = []

    def visit_expr(self, expr: ir.Expr) -> None:
        if isinstance(expr, (ir.Call, ir.Submit)) and isinstance(expr.op, ir.GlobalVar):
            self.launches.append(expr)
        super().visit_expr(expr)

    def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
        if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tensor.create").name:
            self.allocations.append(op)
        super().visit_assign_stmt(op)


def _function_section(pto: str, name: str) -> str:
    start = pto.find(f"func.func @{name}(")
    assert start >= 0, pto
    end = pto.find("\n  func.func @", start + 1)
    return pto[start:] if end < 0 else pto[start:end]


def test_group_computed_tensor_output_full_codegen(tmp_path):
    """A fresh AIV result gets one outer buffer shared with the AIC consumer."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def producer(
            self,
            p: pl.Tensor[[16, 16], pl.FP16],
            q: pl.Tensor[[16, 16], pl.FP16],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            summed = pl.add(p, q)
            return summed

        @pl.function(type=pl.FunctionType.AIC)
        def consumer(
            self,
            summed: pl.Tensor[[16, 16], pl.FP16],
            weight: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            left_mat = pl.load(summed, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            right_mat = pl.load(weight, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            left = pl.move(left_mat, target_memory=pl.Mem.Left)
            right = pl.move(right_mat, target_memory=pl.Mem.Right)
            product = pl.matmul(left, right)
            out = pl.store(product, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            p: pl.Tensor[[16, 16], pl.FP16],
            q: pl.Tensor[[16, 16], pl.FP16],
            weight: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            summed = self.producer(p, q)
            return self.consumer(summed, weight, out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            p: pl.Tensor[[16, 16], pl.FP16],
            q: pl.Tensor[[16, 16], pl.FP16],
            weight: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return self.mixed(p, q, weight, out)

    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before)
    group = after.get_function("mixed")
    main = after.get_function("main")
    producer = after.get_function("producer")
    consumer = after.get_function("consumer")
    assert group is not None and main is not None and producer is not None and consumer is not None
    assert producer.func_type == ir.FunctionType.AIV
    assert consumer.func_type == ir.FunctionType.AIC
    assert len(main.params) == 4
    assert len(group.params) == 5
    assert len(group.return_types) == 1

    group_calls = _CallCollector()
    group_calls.visit_stmt(group.body)
    assert [call.op.name for call in group_calls.launches] == ["producer", "consumer"]
    assert all(list(call.args) == list(group.params) for call in group_calls.launches)
    for member in (producer, consumer):
        assert [param.name_hint for param in member.params] == [param.name_hint for param in group.params]
        assert member.param_directions == group.param_directions
        for member_param, group_param in zip(member.params, group.params, strict=True):
            ir.assert_structural_equal(member_param.type, group_param.type, enable_auto_mapping=True)

    outer = _CallCollector()
    outer.visit_stmt(main.body)
    assert len(outer.allocations) == 1
    assert len(outer.launches) == 1
    launch = outer.launches[0]
    assert launch.op.name == "mixed"
    assert len(launch.args) == len(group.params)
    scratch = outer.allocations[0].var
    assert any(scratch.same_as(argument) for argument in launch.args)
    assert all(not scratch.same_as(param) for param in main.params)

    files = pto_backend.generate(after, str(tmp_path), skip_ptoas=True)
    kernel_files = [content for path, content in files.items() if path.endswith(".pto")]
    assert len(kernel_files) == 1
    pto = kernel_files[0]
    producer_pto = _function_section(pto, "producer")
    consumer_pto = _function_section(pto, "consumer")
    assert "pto.tadd" in producer_pto
    assert "pto.tload" in producer_pto and "pto.tstore" in producer_pto
    assert "pto.tmatmul" in consumer_pto
    assert "pto.tload" in consumer_pto and "pto.tstore" in consumer_pto

    orchestration = files["orchestration/main.cpp"]
    assert orchestration.count("alloc_tensors(") == 1
    dispatch = re.search(r"rt_submit_task\(\s*(\w+)\s*,\s*(\w+)\s*\)", orchestration)
    assert dispatch is not None, orchestration
    mixed_name, params_name = dispatch.groups()
    assert f"MixedKernels {mixed_name}" in orchestration
    arguments = re.findall(rf"{re.escape(params_name)}\.add_\w+\([^)]*\)", orchestration)
    assert len(arguments) == len(group.params), orchestration
    allocated = re.search(r"const Tensor&\s+(\w+)\s*=\s*alloc_\w+\.get_ref\(0\)", orchestration)
    assert allocated is not None, orchestration
    assert re.search(rf"\.add_output\(\s*{re.escape(allocated[1])}\s*\)", orchestration), orchestration
    assert "FREE_VAR" not in after.as_python()
    assert all("FREE_VAR" not in content for content in files.values())


def test_recompiling_a_lowered_jit_program_preserves_buffer_stage(tmp_path):
    """The documented compile(kernel.lower(...)) path must accept already lowered kernels."""

    @pl.jit
    def add_kernel(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            left = pl.load(a, [0, 0], [16, 16])
            right = pl.load(b, [0, 0], [16, 16])
            pl.store(pl.add(left, right), [0, 0], out)
        return out

    inputs = [torch.zeros((16, 16), dtype=torch.float32) for _ in range(3)]
    lowered = add_kernel.lower(*inputs)
    kernels = [function for function in lowered.functions.values() if ir.is_incore_type(function.func_type)]
    assert kernels and all(function.ir_stage == ir.FunctionIRStage.Buffer for function in kernels)

    # ir.compile() runs Default again on this input. Emit its artifacts without
    # invoking PTOAS or a device, just as the first compilation's codegen did.
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(lowered)
    files = pto_backend.generate(after, str(tmp_path), skip_ptoas=True)
    assert "orchestration/add_kernel.cpp" in files
    pto = "\n".join(content for path, content in files.items() if path.endswith(".pto"))
    assert "pto.tadd" in pto and "pto.tstore" in pto
    assert all("FREE_VAR" not in content for content in files.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

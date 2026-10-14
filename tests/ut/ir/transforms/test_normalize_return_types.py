# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Heterogeneous return tuples must keep their types and output identities (#2981)."""

import pypto.language as pl
import pytest
import torch
from pypto import ir, passes
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _verify_types(program):
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.AssignTypeSymmetry)
    properties.insert(passes.IRProperty.TypeChecked)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, program)
    assert not diagnostics, "\n".join(d.message for d in diagnostics)


def _program(*, rows=4, dtype="FP16", caller="Orchestration", submit=False, expected=False, use="index"):
    first = "pl.Tensor[[4, 16], pl.FP32]"
    second = f"pl.Tensor[[{rows}, 16], pl.{dtype}]"
    returns = f"{first}, {second}" if expected else f"{second}, {first}"
    kernel_return = "out_a, out_b" if expected else "b, a"
    b_index, a_index = (1, 0) if expected else (0, 1)
    if submit:
        body = f"""with pl.manual_scope():
            ret = pl.submit(self.kernel, x, y, out_a, out_b)
            b = ret[{b_index}]
            a = ret[{a_index}]
            tid = ret[2]
            ret2 = pl.submit(self.kernel, x, y, out_a, out_b, deps=[tid])
            b2 = ret2[{b_index}]
            a2 = ret2[{a_index}]
            tid2 = ret2[2]"""
    else:
        body = f"""ret = self.kernel(x, y, out_a, out_b)
        b = ret[{b_index}]
        a = ret[{a_index}]"""
    returned = "a, b"
    if use == "alias":
        alias = "(ret[1], ret[0])" if expected else "ret"
        body = f"""ret = self.kernel(x, y, out_a, out_b)
        alias = {alias}
        b = alias[0]
        a = alias[1]"""
    elif use == "whole":
        body = "ret = self.kernel(x, y, out_a, out_b)"
        returned = "(ret[1], ret[0])" if expected else "ret"
    caller_returns = f"{second}, {first}" if use == "whole" else f"{first}, {second}"
    if expected and caller == "Spmd":
        returned = "out_a, out_b"
    return pl.parse_program(f"""
@pl.program
class {"Expected" if expected else "Before"}:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, x: {first}, y: {second}, out_a: pl.Out[{first}], out_b: pl.Out[{second}]
    ) -> tuple[{returns}]:
        tx = pl.load(x, [0, 0], [4, 16])
        ty = pl.load(y, [0, 0], [{rows}, 16])
        a = pl.store(tx, [0, 0], out_a)
        b = pl.store(ty, [0, 0], out_b)
        return {kernel_return}

    @pl.function(type=pl.FunctionType.{caller})
    def main(
        self, x: {first}, y: {second}, out_a: pl.Out[{first}], out_b: pl.Out[{second}]
    ) -> tuple[{caller_returns}]:
        {body}
        return {returned}
""")


@pytest.mark.parametrize("rows,dtype", [(4, "FP16"), (8, "FP32"), (8, "FP16")])
@pytest.mark.parametrize("caller", ["Orchestration", "Spmd"])
@pytest.mark.parametrize("submit", [False, True])
def test_heterogeneous_result_types_follow_return_permutation(rows, dtype, caller, submit):
    before = _program(rows=rows, dtype=dtype, caller=caller, submit=submit)
    expected = _program(rows=rows, dtype=dtype, caller=caller, submit=submit, expected=True)
    _verify_types(before)
    after = passes.normalize_return_order()(before)
    _verify_types(after)
    ir.assert_structural_equal(after, expected, enable_auto_mapping=True)
    ir.assert_structural_equal(passes.normalize_return_order()(after), after)
    restored = ir.deserialize(ir.serialize(after))
    ir.assert_structural_equal(after, restored, enable_auto_mapping=True)


@pytest.mark.parametrize("use", ["alias", "whole"])
def test_whole_tuple_uses_preserve_logical_return_order(use):
    before = _program(use=use)
    expected = _program(use=use, expected=True)
    _verify_types(before)
    after = passes.normalize_return_order()(before)
    _verify_types(after)
    ir.assert_structural_equal(after, expected, enable_auto_mapping=True)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("planner", list(passes.MemoryPlanner))
def test_default_pipeline_preserves_heterogeneous_output_identity(ascend_backend, enabled, planner):
    before = _program(rows=8)
    with passes.PassContext([], enable_buffer_ir=enabled, memory_planner=planner):
        after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(before)
    kernel = after.get_function("kernel")
    assert kernel is not None
    assert kernel.ir_stage == (ir.FunctionIRStage.Buffer if enabled else ir.FunctionIRStage.Functional)
    _verify_types(after)
    namespace = {}
    exec(torch_codegen(after), namespace)  # noqa: S102
    x = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    y = torch.arange(128, dtype=torch.float16).reshape(8, 16) + 100
    a, b = namespace["main"](x, y, torch.empty_like(x), torch.empty_like(y))
    torch.testing.assert_close(a, x)
    torch.testing.assert_close(b, y)


def test_three_way_permutation_preserves_caller_shape_symbols():
    def program(expected):
        types = [
            "pl.Tensor[[inner_n, 16], pl.FP32]",
            "pl.Tensor[[8, 16], pl.FP16]",
            "pl.Tensor[[4, 16], pl.INT32]",
        ]
        kernel_types = ", ".join(types if expected else [types[2], types[0], types[1]])
        kernel_values = "a, b, c" if expected else "c, a, b"
        first, second, third = (2, 0, 1) if expected else (0, 1, 2)
        return pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, inner_n: pl.Scalar[pl.INDEX],
        a: pl.Out[pl.Tensor[[inner_n, 16], pl.FP32]],
        b: pl.Out[pl.Tensor[[8, 16], pl.FP16]],
        c: pl.Out[pl.Tensor[[4, 16], pl.INT32]],
    ) -> tuple[{kernel_types}]:
        tile = pl.load(b, [0, 0], [8, 16])
        written = pl.store(tile, [0, 0], b)
        return {kernel_values}

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, outer_n: pl.Scalar[pl.INDEX],
        a: pl.Out[pl.Tensor[[outer_n, 16], pl.FP32]],
        b: pl.Out[pl.Tensor[[8, 16], pl.FP16]],
        c: pl.Out[pl.Tensor[[4, 16], pl.INT32]],
    ) -> tuple[
        pl.Tensor[[4, 16], pl.INT32], pl.Tensor[[outer_n, 16], pl.FP32], pl.Tensor[[8, 16], pl.FP16]
    ]:
        ret = self.kernel(outer_n, a, b, c)
        first = ret[{first}]
        second = ret[{second}]
        third = ret[{third}]
        return first, second, third
""")

    before = program(False)
    expected = program(True)
    _verify_types(before)
    after = passes.normalize_return_order()(before)
    _verify_types(after)
    ir.assert_structural_equal(after, expected, enable_auto_mapping=True)


@pytest.mark.parametrize("flow", ["if", "if_else", "mixed", "loop"])
@pytest.mark.parametrize("whole", [False, True])
def test_pre_ssa_reassignments_preserve_control_flow(flow, whole):
    initial = "ret = self.kernel(a, b)"
    if flow == "if_else":
        body = (
            "if flag > 0:\n            ret = self.kernel(a, b)"
            "\n        else:\n            ret = self.kernel(other_a, other_b)"
        )
    elif flow == "loop":
        body = initial + (
            "\n        for i in pl.range(flag):"
            "\n            ret = self.kernel(other_a, other_b)"
            "\n            other_a = ret[1]"
            "\n            other_b = ret[0]"
        )
    else:
        rhs = "(other_b, other_a)" if flow == "mixed" else "self.kernel(other_a, other_b)"
        body = initial + f"\n        if flag > 0:\n            ret = {rhs}"
    returned = "ret" if whole else "ret[0], ret[1]"
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
               b: pl.Out[pl.Tensor[[4, 16], pl.FP16]]) -> tuple[
                   pl.Tensor[[4, 16], pl.FP16], pl.Tensor[[4, 16], pl.FP32]]:
        tile = pl.load(a, [0, 0], [4, 16])
        written = pl.store(tile, [0, 0], a)
        return b, a

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, flag: pl.Scalar[pl.INDEX], a: pl.Tensor[[4, 16], pl.FP32],
             b: pl.Tensor[[4, 16], pl.FP16], other_a: pl.Tensor[[4, 16], pl.FP32],
             other_b: pl.Tensor[[4, 16], pl.FP16]) -> tuple[
                 pl.Tensor[[4, 16], pl.FP16], pl.Tensor[[4, 16], pl.FP32]]:
        {body}
        return {returned}
""")
    after = passes.normalize_return_order()(before)
    kernel = after.get_function("kernel")
    assert kernel is not None
    assert isinstance(kernel.return_types[0], ir.TensorType)
    assert kernel.return_types[0].dtype == pl.FP32
    _verify_types(after)
    after = passes.convert_to_ssa()(after)
    _verify_types(after)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.SSAForm)
    assert not passes.PropertyVerifierRegistry.verify(properties, after)
    reference, actual = {}, {}
    exec(torch_codegen(before), reference)  # noqa: S102
    exec(torch_codegen(after), actual)  # noqa: S102
    tensors = [
        torch.full((4, 16), i, dtype=dtype)
        for i, dtype in enumerate([torch.float32, torch.float16, torch.float32, torch.float16], start=1)
    ]
    for flag in [0, 1, 2]:
        expected = reference["main"](flag, *tensors)
        result = actual["main"](flag, *tensors)
        for got, want in zip(result, expected):
            torch.testing.assert_close(got, want)


def test_pre_ssa_submit_reassignment_preserves_merge_and_task_id():
    before = pl.parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
               b: pl.Out[pl.Tensor[[4, 16], pl.FP16]]) -> tuple[
                   pl.Tensor[[4, 16], pl.FP16], pl.Tensor[[4, 16], pl.FP32]]:
        tile = pl.load(a, [0, 0], [4, 16])
        written = pl.store(tile, [0, 0], a)
        return b, a

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, flag: pl.Scalar[pl.BOOL], a: pl.Tensor[[4, 16], pl.FP32],
             b: pl.Tensor[[4, 16], pl.FP16]) -> tuple[
                 pl.Tensor[[4, 16], pl.FP16], pl.Tensor[[4, 16], pl.FP32]]:
        with pl.manual_scope():
            ret = pl.submit(self.kernel, a, b)
            if flag:
                ret = pl.submit(self.kernel, a, b)
            b1 = ret[0]
            a1 = ret[1]
            tid = ret[2]
            followup = pl.submit(self.kernel, a1, b1, deps=[tid])
        return b1, a1
""")
    after = passes.convert_to_ssa()(passes.normalize_return_order()(before))
    _verify_types(after)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.SSAForm)
    properties.insert(passes.IRProperty.ManualDepsOnSubmitOnly)
    assert not passes.PropertyVerifierRegistry.verify(properties, after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

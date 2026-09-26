# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dynamic zero-block guards through the complete compiler pipeline."""

import re

import pypto
import pypto.language as pl
import pytest
from _orchestration_codegen_common import _generate_orch_code
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _compile(program, auto_deps):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    pm = PassManager.get_strategy(OptimizationStrategy.Default, analyze_auto_scopes_for_deps=auto_deps)
    result = pm.run_passes(program)
    return result, _generate_orch_code(result)


@pytest.mark.parametrize("auto_deps", [False, True])
def test_plain_dynamic_launch_merges_output_before_consumer(auto_deps):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def consume(
            self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t = pl.load(x, [0, 0], [16, 16])
            return pl.store(t, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            ctrl: pl.Tensor[[1], pl.INT32],
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            n = pl.tensor.read(ctrl, [0])
            for bi in pl.spmd(n):
                a = pl.add(a, 1.0)
            b = self.consume(a, b)
            return b

    result, code = _compile(Program, auto_deps)
    assert "if (" in code, code
    assert code.count(".launch_spec.set_block_num(") == 1
    assert "} else {" in code
    assert "_guarded_" in ir.python_print(result) or "_nonempty" in ir.python_print(result)


@pytest.mark.parametrize("auto_deps", [False, True])
def test_manual_multi_output_empty_task_keeps_dependency_chain(auto_deps):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
            self, a: pl.Out[pl.Tensor[[16], pl.FP32]], b: pl.Out[pl.Tensor[[16], pl.FP32]]
        ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
            ta = pl.tile.full([16], pl.FP32, 3.0)
            tb = pl.tile.full([16], pl.FP32, 7.0)
            a = pl.store(ta, [0], a)
            b = pl.store(tb, [0], b)
            return b, a

        @pl.function(type=pl.FunctionType.InCore)
        def consumer(
            self,
            a: pl.Tensor[[16], pl.FP32],
            b: pl.Tensor[[16], pl.FP32],
            out: pl.Out[pl.Tensor[[16], pl.FP32]],
        ) -> pl.Tensor[[16], pl.FP32]:
            ta = pl.load(a, [0], [16])
            tb = pl.load(b, [0], [16])
            return pl.store(pl.tile.sub(ta, tb), [0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            ctrl: pl.Tensor[[1], pl.INT32],
            a: pl.Tensor[[16], pl.FP32],
            b: pl.Tensor[[16], pl.FP32],
            out: pl.Tensor[[16], pl.FP32],
        ) -> pl.Tensor[[16], pl.FP32]:
            n = pl.tensor.read(ctrl, [0])
            with pl.manual_scope():
                prior = pl.system.task_dummy(deps=[])
                (ob, oa), tid = pl.spmd_submit(self.producer, a, b, core_num=n, deps=[prior])
                out, _ = pl.submit(self.consumer, oa, ob, out, deps=[tid])
            return out

    result, code = _compile(Program, auto_deps)
    assert code.count("rt_submit_dummy_task(") == 2
    assert "} else {" in code
    assert re.search(r"TaskId \w+guarded\w* = TaskId::invalid\(\);", code)
    assert "spmd_empty_tid" in code
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.SSAForm)
    properties.insert(passes.IRProperty.UseAfterDef)
    assert not passes.PropertyVerifierRegistry.verify(properties, result)


def test_mixed_dynamic_launch_keeps_gm_pipe_allocation_inside_guard():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            ctrl: pl.Tensor[[1], pl.INT32],
            a: pl.Tensor[[16, 64], pl.FP32],
            b: pl.Tensor[[64, 16], pl.FP32],
            out: pl.Tensor[[16, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            n = pl.tensor.read(ctrl, [0])
            for bi in pl.spmd(n):
                shifted = pl.add(a, 1.0)
                product = pl.matmul(shifted, b)
                out = pl.add(product, 1.0)
            return out

    _, code = _compile(Program, False)
    guard_position = code.index("if (")
    allocation_position = code.index("gm_pipe_buffer_")
    dispatch_position = code.index(".launch_spec.set_block_num(")
    assert guard_position < allocation_position < dispatch_position
    assert "rt_submit_task(mixed_" in code


def test_codegen_rejects_dynamic_launch_when_legalization_was_omitted():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, out: pl.Out[pl.Tensor[[16], pl.FP32]]) -> pl.Tensor[[16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, out: pl.Tensor[[16], pl.FP32], n: pl.Scalar[pl.INDEX]) -> pl.Tensor[[16], pl.FP32]:
            out, tid = pl.spmd_submit(self.kernel, out, core_num=n)
            return out

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    program = passes.normalize_return_order()(passes.convert_to_ssa()(Program))
    with pytest.raises(pypto.InternalError, match="Run LegalizeSpmdLaunches"):
        _generate_orch_code(program)


@pytest.mark.parametrize("capture_task", [False, True])
def test_dynamic_output_shape_uses_caller_scalar_after_guard(capture_task):
    program = pl.parse(f"""
n_dim = pl.dynamic("n_dim")
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[n_dim, 16], pl.FP32]):
        n = pl.tensor.dim(x, 0)
        padded = (n + 15) // 16 * 16
        out = pl.tensor.create([padded, 16], dtype=pl.FP32)
        for bi in pl.spmd(padded // 16, allow_early_resolve={capture_task}):
            tile = pl.slice(x, [16, 16], [bi * 16, 0])
            out = pl.assemble(out, tile, [bi * 16 % padded, 0])
        return out
""")

    # Intermediate outlined return types retain callee shape Vars and cannot
    # yet print/parse as Python annotations. Use production pipeline checks
    # plus binary roundtrip here; verify SSA and use-after-def on the final IR.
    def check_binary_roundtrip(_pass, intermediate):
        ir.assert_structural_equal(
            intermediate, ir.deserialize(ir.serialize(intermediate)), enable_auto_mapping=True
        )

    with passes.PassContext([passes.CallbackInstrument(after_pass=check_binary_roundtrip)]):
        result, code = _compile(program, True)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.SSAForm)
    properties.insert(passes.IRProperty.UseAfterDef)
    assert not passes.PropertyVerifierRegistry.verify(properties, result)
    assert "if (" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

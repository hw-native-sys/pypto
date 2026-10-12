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
from _orchestration_codegen_common import _generate_orch_code, _out_of_scope_tensor_refs
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.runtime import RunConfig


def _compile(program, auto_deps):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    pm = PassManager.get_strategy(OptimizationStrategy.Default, analyze_auto_scopes_for_deps=auto_deps)
    result = pm.run_passes(program)
    return result, _generate_orch_code(result)


@pytest.mark.parametrize("inside_scope", [False, True])
def test_scope_detector_checks_tensor_assignment_targets(inside_scope):
    assignment = "carry = input;"
    code = "Tensor input;\nSIMPLER_SCOPE() {\nTensor carry = input;\n"
    code += assignment + "\n}\n" if inside_scope else "}\n" + assignment + "\n"
    assert _out_of_scope_tensor_refs(code) == ([] if inside_scope else ["carry"])


@pytest.mark.parametrize("auto_deps", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_dynamic_launch_inside_explicit_scope_loop_carry(auto_deps, dynamic):
    """A guarded output must remain visible to the yield after an explicit scope."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(
            self,
            ctrl: pl.Tensor[[1], pl.INT32],
            a: pl.Tensor[[16, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            if dynamic:
                n = pl.tensor.read(ctrl, [0])
            else:
                n = pl.const(4, pl.INT32)
            for i in pl.range(2):
                # Keep work before the scope: the loop yield then sits outside it.
                count = pl.min(n - i, 16)
                with pl.scope():
                    for bi in pl.spmd(count, allow_early_resolve=True):
                        a[0:16, 0:16] = pl.add(a, 1.0)
            return a

    _, code = _compile(Program, auto_deps)
    assert not _out_of_scope_tensor_refs(code), code


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


@pytest.mark.parametrize("auto_deps", [False, True])
def test_composite_positive_bound_survives_late_simplify(auto_deps):
    """Codegen must retain Simplify's proof for a guarded ceil-div launch count."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            out: pl.Tensor[[16, 16], pl.FP32],
            total: pl.Scalar[pl.INDEX],
            active: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            remaining = total - active
            if remaining > 0:
                tiles = (remaining + 15) // 16
                for bi in pl.spmd(tiles * 4):
                    out = pl.add(out, 1.0)
            return out

    result, code = _compile(Program, auto_deps)
    assert code.count(".launch_spec.set_block_num(") == 1
    assert "if (" in code
    # The author guard suffices; no generated launch-count guard is needed.
    assert "_nonempty" not in ir.python_print(result)


@pytest.mark.parametrize("conditional", [False, True])
def test_nested_inline_conditional_output_keeps_dynamic_shape(conditional, tmp_path):
    """Nested inline allocation and padding must retain the output's actual shape."""
    n_dim = pl.dynamic("n_dim")

    @pl.jit.inline
    def pad(ids: pl.Tensor[[n_dim], pl.INT32], out: pl.Tensor[[n_dim, 16], pl.FP32]):
        n = pl.tensor.dim(ids, 0)
        with pl.spmd(n, name_hint="pad_output_init") as tid:
            bi = pl.tile.get_block_idx()
            if conditional:
                if pl.read(ids, [bi]) < 0:
                    out[bi : bi + 1, :] = pl.full([1, 16], dtype=pl.FP32, value=0.0)
            else:
                out[bi : bi + 1, :] = pl.full([1, 16], dtype=pl.FP32, value=0.0)
        _fence = pl.system.task_dummy(deps=[tid])
        return out

    @pl.jit.inline
    def allocate(ids: pl.Tensor[[n_dim], pl.INT32], positions: pl.Tensor[[n_dim], pl.INT32]):
        rows = pl.tensor.dim(positions, 0)
        out = pl.create_tensor([rows, 16], dtype=pl.FP32)
        out = pad(ids, out)
        return out

    @pl.jit
    def main(ids: pl.Tensor[[n_dim], pl.INT32], positions: pl.Tensor[[n_dim], pl.INT32]):
        return allocate(ids, positions)

    # Text roundtrip normalizes redundant pre-SSA aliases in the nested inline
    # body. Keep every pass's property checks and lossless binary roundtrip.
    def check_binary_roundtrip(_pass, intermediate):
        ir.assert_structural_equal(
            intermediate, ir.deserialize(ir.serialize(intermediate)), enable_auto_mapping=True
        )

    with passes.PassContext(
        [
            passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER),
            passes.CallbackInstrument(after_pass=check_binary_roundtrip),
        ]
    ):
        compiled = main.compile(
            config=RunConfig(codegen_only=True, save_kernels=True, save_kernels_dir=str(tmp_path))
        )
    assert compiled is not None


@pytest.mark.parametrize("platform", ["a2a3", "a2a3sim"])
def test_dynamic_spmd_branch_writebacks_preserve_preallocated_buffer(platform, tmp_path):
    """Both branch writebacks retain context's identity through the SPMD wrapper (#2966)."""
    n_dim = pl.dynamic("N")

    @pl.jit
    def branch_buffer_identity(
        flags: pl.Tensor[[n_dim], pl.INT32],
        output: pl.Out[pl.Tensor[[n_dim, 16], pl.FP32]],
    ):
        flags.bind_dynamic(0, n_dim)
        output.bind_dynamic(0, n_dim)
        n = pl.tensor.dim(flags, 0)
        context = pl.create_tensor([4 * n, 16], dtype=pl.FP32)
        for row in pl.spmd(n, name_hint="branch_store"):
            if pl.read(flags, [row]) >= 0:
                value = pl.full([1, 16], dtype=pl.FP32, value=1.0)
                for head in pl.unroll(4):
                    position = head * n + row
                    context[position : position + 1, 0:16] = value
            else:
                empty_value = pl.full([1, 16], dtype=pl.FP32, value=0.0)
                for head in pl.unroll(4):
                    position = head * n + row
                    context[position : position + 1, 0:16] = empty_value
        for row in pl.spmd(n, name_hint="consume"):
            output[row : row + 1, 0:16] = context[row : row + 1, 0:16]
        return output

    # Outlined call-result types retain callee-local shape Vars until legalization.
    # Use binary roundtrip for intermediates and verify SSA/use-after-def on final IR.
    lowered_programs: list[ir.Program] = []

    def check_binary_roundtrip(_pass, intermediate):
        ir.assert_structural_equal(
            intermediate, ir.deserialize(ir.serialize(intermediate)), enable_auto_mapping=True
        )
        lowered_programs[:] = [intermediate]

    with passes.PassContext([passes.CallbackInstrument(after_pass=check_binary_roundtrip)]):
        compiled = branch_buffer_identity.compile(
            config=RunConfig(
                platform=platform, codegen_only=True, save_kernels=True, save_kernels_dir=str(tmp_path)
            )
        )
    assert compiled is not None
    assert lowered_programs
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.SSAForm)
    properties.insert(passes.IRProperty.UseAfterDef)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, lowered_programs[0])
    assert not diagnostics, [diagnostic.message for diagnostic in diagnostics]
    code = (compiled.output_dir / "orchestration" / "branch_buffer_identity.cpp").read_text()
    assert code.count(".launch_spec.set_block_num(") == 2
    assert code.count("} else {") == 2
    assert list(compiled.output_dir.rglob("*.pto"))


@pytest.mark.parametrize("auto_deps", [False, True])
@pytest.mark.parametrize("scope_inside_loop", [False, True])
@pytest.mark.parametrize("nested_scopes", [0, 1, 3])
def test_guarded_tuple_launch_result_survives_auto_scope(auto_deps, scope_inside_loop, nested_scopes):
    source = """
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
    def main(
        self,
        ctrl: pl.Tensor[[1], pl.INT32],
        a: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        n = pl.tensor.read(ctrl, [0])
        with pl.scope():
            for layer in pl.range(2):
                for bi in pl.spmd(n):
                    a = pl.assemble(a, pl.add(a, 1.0), [0, 0])
                    out = pl.assemble(out, pl.add(out, 2.0), [0, 0])
        with pl.scope():
            for bi in pl.spmd(1):
                out = pl.assemble(out, pl.add(a, out), [0, 0])
        return out
"""
    if scope_inside_loop:
        source = source.replace(
            "with pl.scope():\n            for layer in pl.range(2):",
            "for layer in pl.range(2):\n            with pl.scope():",
        )
    for _ in range(nested_scopes):
        lines: list[str] = source.splitlines()
        start = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("        with pl.scope():") or line.startswith("        for layer")
        )
        end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("        with pl.scope():"))
        lines[start:end] = ["        with pl.scope():"] + ["    " + line for line in lines[start:end]]
        source = "\n".join(lines)
    _, code = _compile(pl.parse_program(source), auto_deps)
    assert not _out_of_scope_tensor_refs(code), code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

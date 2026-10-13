# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Default selection must prove a complete logical kernel before storage mutation."""

import pypto.language as pl
import pytest
import torch
from pypto import ir, passes
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

pytestmark = pytest.mark.usefixtures("ascend_backend")

_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
_VECTOR = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
    "abs": torch.abs,
    "exp": torch.exp,
    "sqrt": torch.sqrt,
    "neg": torch.neg,
    "relu": torch.relu,
    "log": torch.log,
    "recip": torch.reciprocal,
}
_BINARY = {"add", "sub", "mul", "div", "maximum", "minimum"}
_SCALAR = {
    "adds": torch.add,
    "subs": torch.sub,
    "muls": torch.mul,
    "divs": torch.div,
    "maximums": torch.maximum,
    "minimums": torch.minimum,
    "lrelu": lambda x, s: torch.where(x > 0, x, x * s),
}


def _program(operation, dtype="FP32", scalar_dtype="FP32"):
    if operation == "full":
        body = f"value = pl.tile.full([16, 32], pl.{dtype}, 0.5)"
    else:
        args = "x, y" if operation in _BINARY else "x, factor" if operation in _SCALAR else "x"
        if scalar_dtype == "INDEX":
            args = args.replace("factor", "pl.cast(factor, pl.INT64)")
        body = (
            "x = pl.load(a, [0, 0], [16, 32])\n"
            "        y = pl.load(b, [0, 0], [16, 32])\n"
            f"        value = pl.tile.{operation}({args})"
        )
    return pl.parse_program(f"""
@pl.program
class Recipes:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.{dtype}], b: pl.Tensor[[16, 32], pl.{dtype}],
               factor: pl.Scalar[pl.{scalar_dtype}], out: pl.Out[pl.Tensor[[16, 32], pl.{dtype}]]
               ) -> pl.Tensor[[16, 32], pl.{dtype}]:
        {body}
        return pl.store(value, [0, 0], out)
""")


def _lower(program, planner, enabled=None):
    options = {} if enabled is None else {"enable_buffer_ir": enabled}
    with passes.PassContext([], memory_planner=planner, **options):
        return PassManager(OptimizationStrategy.Default).run_passes(program)


def _kernel(program: ir.Program) -> ir.Function:
    function = program.get_function("kernel")
    assert function is not None
    return function


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enabled", [None, True, False])
@pytest.mark.parametrize("operation", [*_VECTOR, *_SCALAR, "full"])
def test_default_recipes_select_emit_roundtrip_and_execute(planner, enabled, operation):
    program = _lower(_program(operation), planner, enabled)
    function = _kernel(program)
    assert function is not None
    expected_stage = ir.FunctionIRStage.Functional if enabled is False else ir.FunctionIRStage.Buffer
    assert function.ir_stage == expected_stage
    if enabled is False:
        return
    restored = ir.deserialize(ir.serialize(program))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(program, restored, enable_auto_mapping=True)
    text = codegen.PTOCodegen().generate(restored, emit_tile_addr=False, emit_source_loc=False)
    assert "pto.tstore" in text
    a = torch.linspace(0.25, 1.25, 512).reshape(16, 32)
    if operation in ("abs", "neg", "relu", "lrelu"):
        a -= 0.75
    b = torch.linspace(0.5, 2, 512).reshape(16, 32)
    scalar = 0.5
    out = torch.full_like(a, -777)
    if operation == "full":
        expected = torch.full_like(a, scalar)
    elif operation in _SCALAR:
        expected = _SCALAR[operation](a, torch.tensor(scalar))
    else:
        expected = _VECTOR[operation](a, b) if operation in _BINARY else _VECTOR[operation](a)
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    namespace["kernel"](a, b, scalar, out)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enabled", [None, True, False])
@pytest.mark.parametrize("rhs_rows", [1, 16])
def test_elementwise_selection_requires_matching_physical_shapes(planner, enabled, rhs_rows):
    program = pl.parse_program(f"""
@pl.program
class PhysicalShapes:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32], b: pl.Tensor[[16, 32], pl.FP32],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        x = pl.load(a, [0, 0], [16, 32], valid_shape=[1, 32])
        y = pl.load(b, [0, 0], [{rhs_rows}, 32], valid_shape=[1, 32])
        value = pl.add(x, y)
        return pl.store(value, [0, 0], out)
""")
    lowered = _lower(program, planner, enabled)
    selected = enabled is not False and rhs_rows == 16
    assert _kernel(lowered).ir_stage == (
        ir.FunctionIRStage.Buffer if selected else ir.FunctionIRStage.Functional
    )
    text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=False, emit_source_loc=False)
    assert "pto.tadd" in text and "pto.tstore" in text


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("dtype", ["FP16", "INT32", "BF16"])
@pytest.mark.parametrize("operation", ["add", "mul"])
def test_arithmetic_dtype_contract_is_narrower_than_transfer(planner, dtype, operation):
    program = _lower(_program(operation, dtype), planner)
    function = _kernel(program)
    assert function is not None
    assert function.ir_stage == (
        ir.FunctionIRStage.Functional if dtype == "BF16" else ir.FunctionIRStage.Buffer
    )


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("scalar_dtype", ["INT32", "INT64", "INDEX", "FP16", "BF16", "FP32"])
def test_element_scalar_conversion_runs_in_debug(planner, scalar_dtype):
    program = _lower(_program("muls", scalar_dtype=scalar_dtype), planner)
    assert _kernel(program).ir_stage == ir.FunctionIRStage.Buffer
    a = torch.linspace(-1, 1, 512).reshape(16, 32)
    out = torch.empty_like(a)
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    namespace["kernel"](a, a, 2, out)
    torch.testing.assert_close(out, a * 2)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("dtype", ["FP16", "BF16", "FP32", "INT32"])
def test_partial_typed_transfer_preserves_unwritten_rows_and_columns(planner, dtype):
    program = pl.parse_program(f"""
@pl.program
class Transfer:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.{dtype}],
               out: pl.InOut[pl.Tensor[[24, 48], pl.{dtype}]]) -> pl.Tensor[[24, 48], pl.{dtype}]:
        x = pl.load(a, [0, 0], [16, 32], valid_shape=[13, 24])
        return pl.store(x, [3, 8], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    torch_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
        "FP32": torch.float32,
        "INT32": torch.int32,
    }[dtype]
    a = (torch.arange(512).reshape(16, 32) * 13 - 1777).to(torch_dtype)
    out = torch.full((24, 48), -777, dtype=torch_dtype)
    expected = out.clone()
    expected[3:16, 8:32] = a[:13, :24]
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    namespace["kernel"](a, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,selected", [("FP32", True), ("FP16", False), ("BF16", False)])
def test_row_alignment_depends_on_element_width(dtype, selected):
    program = pl.parse_program(f"""
@pl.program
class Transfer:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 8], pl.{dtype}],
               out: pl.Out[pl.Tensor[[16, 8], pl.{dtype}]]) -> pl.Tensor[[16, 8], pl.{dtype}]:
        x = pl.load(a, [0, 0], [16, 8])
        return pl.store(x, [0, 0], out)
""")
    lowered = _lower(program, passes.MemoryPlanner.PYPTO)
    assert (_kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer) is selected


@pytest.mark.parametrize("planner", _PLANNERS)
def test_unsupported_scalar_power_keeps_the_whole_kernel_on_tile(planner):
    program = pl.parse_program("""
@pl.program
class ScalarPower:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32], factor: pl.Scalar[pl.FP32],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        x = pl.load(a, [0, 0], [16, 32])
        value = pl.tile.muls(x, factor ** 2.0)
        return pl.store(value, [0, 0], out)
""")
    program = _lower(program, planner)
    assert _kernel(program).ir_stage == ir.FunctionIRStage.Functional


def _control_flow_program(kind, step="1", change_alias=False):
    header = (
        "for left, right, row, gm in pl.while_(init_values=(lhs, rhs, 0, out)):\n"
        "            pl.cond(row < count)"
        if kind == "while"
        else f"for i, (left, right, row, gm) in pl.range(0, count, {step}, init_values=(lhs, rhs, 0, out)):"
    )
    backedge = "other" if change_alias else "stored"
    return pl.parse_program(f"""
@pl.program
class ControlFlow:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32], b: pl.Tensor[[16, 32], pl.FP32],
               out: pl.InOut[pl.Tensor[[32, 32], pl.FP32]], other: pl.InOut[pl.Tensor[[32, 32], pl.FP32]],
               count: pl.Scalar[pl.INT64], flag: pl.Scalar[pl.BOOL], step: pl.Scalar[pl.INT64]
               ) -> tuple[pl.Tensor[[32, 32], pl.FP32], pl.Tensor[[32, 32], pl.FP32]]:
        lhs = pl.load(a, [0, 0], [16, 32])
        rhs = pl.load(b, [0, 0], [16, 32])
        {header}
            if flag:
                chosen = pl.yield_(left)
            else:
                chosen = pl.yield_(right)
            stored = pl.store(chosen, [row, 0], gm)
            final_left, final_right, final_row, final_gm = pl.yield_(right, left, row + 1, {backedge})
        value = pl.sub(final_left, pl.mul(final_right, rhs))
        result = pl.store(value, [final_row, 0], final_gm)
        saved = pl.store(lhs, [0, 0], other)
        return result, saved
""")


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["for", "while"])
@pytest.mark.parametrize("enabled", [None, True, False])
def test_default_control_flow_preserves_swaps_live_inputs_and_gm_carries(planner, kind, enabled):
    if enabled is False:
        # Selection is disabled before planning. The legacy DSA-RP while path
        # independently rejects this storage pattern; it is not a Buffer fallback.
        with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
            selected = passes.select_buffer_ir_functions()(_control_flow_program(kind))
        assert _kernel(selected).attrs["buffer_ir_selected"] is False
        return
    program = _lower(_control_flow_program(kind), planner, enabled)
    assert _kernel(program).ir_stage == (
        ir.FunctionIRStage.Functional if enabled is False else ir.FunctionIRStage.Buffer
    )
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    a = torch.linspace(-1, 1, 512).reshape(16, 32)
    b = torch.linspace(0.5, 2, 512).reshape(16, 32)
    for count in range(4):
        for flag in (False, True):
            out = torch.full((32, 32), -777.0)
            saved = torch.full((32, 32), -555.0)
            expected = out.clone()
            left, right = a, b
            for row in range(count):
                expected[row : row + 16] = left if flag else right
                left, right = right, left
            expected[count : count + 16] = left - right * b
            namespace["kernel"](a, b, out, saved, count, flag, 1)
            torch.testing.assert_close(out, expected)
            torch.testing.assert_close(saved[:16], a)
            assert torch.all(saved[16:] == -555)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(
    "kind,step,change_alias",
    [
        ("for", "step", False),
        ("for", "-1", False),
        ("for", "1", True),
        ("while", "1", True),
    ],
)
def test_unsupported_step_and_changing_gm_carry_are_rejected_before_storage(
    planner, kind, step, change_alias
):
    with passes.PassContext([], memory_planner=planner):
        manager = PassManager(OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        program = _control_flow_program(kind, step, change_alias)
        for transform in manager.passes[: boundary + 1]:
            program = transform(program)
    assert _kernel(program).attrs["buffer_ir_selected"] is False


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["for", "while"])
def test_scalar_carry_swaps_are_simultaneous_in_buffer_debug(planner, kind):
    header = (
        "for row, column, i in pl.while_(init_values=(0, 16, 0)):\n            pl.cond(i < count)"
        if kind == "while"
        else "for j, (row, column, i) in pl.range(count, init_values=(0, 16, 0)):"
    )
    program = pl.parse_program(f"""
@pl.program
class ScalarSwap:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32], out: pl.InOut[pl.Tensor[[32, 64], pl.FP32]],
               count: pl.Scalar[pl.INT64]) -> pl.Tensor[[32, 64], pl.FP32]:
        x = pl.load(a, [0, 0], [16, 32])
        {header}
            final_row, final_column, final_i = pl.yield_(column, row, i + 1)
        return pl.store(x, [final_row, final_column], out)
""")
    program = _lower(program, planner)
    assert _kernel(program).ir_stage == ir.FunctionIRStage.Buffer
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    a = torch.arange(512, dtype=torch.float32).reshape(16, 32)
    for count in range(4):
        out = torch.full((32, 64), -777.0)
        expected = out.clone()
        row, column = (0, 16) if count % 2 == 0 else (16, 0)
        expected[row : row + 16, column : column + 32] = a
        namespace["kernel"](a, out, count)
        torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("same_alias", [True, False])
def test_branch_gm_aliases_must_resolve_to_one_parameter(planner, same_alias):
    other = "a" if same_alias else "b"
    program = pl.parse_program(f"""
@pl.program
class SelectGM:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32], b: pl.Tensor[[16, 32], pl.FP32],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]], flag: pl.Scalar[pl.BOOL]
               ) -> pl.Tensor[[16, 32], pl.FP32]:
        if flag:
            source = pl.yield_(a)
        else:
            source = pl.yield_({other})
        x = pl.load(source, [0, 0], [16, 32])
        return pl.store(x, [0, 0], out)
""")
    result = _lower(program, planner)
    assert _kernel(result).ir_stage == (
        ir.FunctionIRStage.Buffer if same_alias else ir.FunctionIRStage.Functional
    )
    restored = ir.deserialize(ir.serialize(result))
    assert isinstance(restored, ir.Program)
    namespace = {}
    exec(torch_codegen(restored), namespace)  # noqa: S102
    a = torch.arange(512, dtype=torch.float32).reshape(16, 32)
    b = -a - 1
    for flag in (False, True):
        out = torch.empty_like(a)
        namespace["kernel"](a, b, out, flag)
        torch.testing.assert_close(out, a if flag or same_alias else b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

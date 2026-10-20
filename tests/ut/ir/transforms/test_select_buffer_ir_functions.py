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
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
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


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("enabled", [None, False])
@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
def test_static_windows_use_default_pipeline_and_preserve_source(
    tmp_path, ascend_backend, planner, nested, enabled
):
    window = (
        "outer = pl.tile.slice(whole, [32, 48], [16, 16])\n"
        "        window = pl.tile.slice(outer, [16, 32], [0, 16])"
        if nested
        else "window = pl.tile.slice(whole, [16, 32], [16, 32])"
    )
    program = pl.parse_program(f"""
@pl.program
class StaticWindows:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32],
               out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
               placed_out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
               ) -> pl.Tensor[[64, 64], pl.FP32]:
        whole = pl.load(x, [0, 0], [64, 64])
        {window}
        doubled = pl.add(window, window)
        canvas = pl.tile.full([32, 32], dtype=pl.FP32, value=-777.0)
        placed = pl.tile.assemble(canvas, doubled, [16, 0])
        pl.store(placed, [0, 0], placed_out)
        return pl.store(whole, [0, 0], out)
""")
    lowered = _lower(program, planner, enabled)
    assert _kernel(lowered).ir_stage == (
        ir.FunctionIRStage.Functional
        if enabled is False or ascend_backend == BackendType.Ascend950
        else ir.FunctionIRStage.Buffer
    )
    restored = ir.deserialize(ir.serialize(lowered))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
    text = codegen.PTOCodegen().generate(
        restored, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS, emit_source_loc=False
    )
    assert "pto.subview" in text
    namespace = {}
    exec(torch_codegen(restored), namespace)  # noqa: S102
    x = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    out = torch.empty_like(x)
    placed = torch.empty((32, 32))
    namespace["kernel"](x, out, placed)
    torch.testing.assert_close(out, x)
    expected = torch.full_like(placed, -777)
    expected[16:] = x[16:32, 32:] * 2
    torch.testing.assert_close(placed, expected)
    if find_ptoas_binary() is not None:
        source = tmp_path / "window.pto"
        source.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        arch = "a2" if ascend_backend == BackendType.Ascend910 else "a5"
        _run_ptoas(str(source), str(tmp_path / "window.cpp"), [f"--pto-arch={arch}", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("column", [0, 16, 32])
def test_overlapping_assemble_snapshots_source_before_write(tmp_path, planner, column):
    program = pl.parse_program(f"""
@pl.program
class OverlappingAssemble:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[32, 64], pl.FP32],
               out: pl.Out[pl.Tensor[[32, 64], pl.FP32]]) -> pl.Tensor[[32, 64], pl.FP32]:
        whole = pl.load(x, [0, 0], [32, 64])
        source = pl.tile.slice(whole, [32, 32], [0, 0])
        placed = pl.tile.assemble(whole, source, [0, {column}])
        return pl.store(placed, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    # Exact self-writes need no copy; other cases still have overlapping byte envelopes.
    assert text.count("pto.alloc_tile") == (1 if column == 0 else 2)
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(2048, dtype=torch.float32).reshape(32, 64)
    out = torch.empty_like(x)
    namespace["kernel"](x, out)
    expected = x.clone()
    expected[:, column : column + 32] = x[:, :32]
    torch.testing.assert_close(out, expected)
    if find_ptoas_binary() is not None:
        source = tmp_path / "assemble.pto"
        source.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(source), str(tmp_path / "assemble.cpp"), ["--pto-arch=a2", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_disjoint_assemble_fits_one_large_allocation(tmp_path, planner, reverse, nested):
    row = 64 if reverse else 0
    target_row = 0 if reverse else 64
    window = (
        f"outer = pl.tile.slice(whole, [96, 256], [{row // 2}, 0])\n"
        f"        source = pl.tile.slice(outer, [64, 256], [{row // 2}, 0])"
        if nested
        else f"source = pl.tile.slice(whole, [64, 256], [{row}, 0])"
    )
    program = pl.parse_program(f"""
@pl.program
class DisjointAssemble:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[128, 256], pl.FP32],
               out: pl.Out[pl.Tensor[[128, 256], pl.FP32]]) -> pl.Tensor[[128, 256], pl.FP32]:
        whole = pl.load(x, [0, 0], [128, 256])
        {window}
        placed = pl.tile.assemble(whole, source, [{target_row}, 0])
        return pl.store(placed, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    # One parent is 128 KiB; an unnecessary snapshot exceeds the 184 KiB Vec limit.
    assert text.count("pto.alloc_tile") == 1
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(128 * 256, dtype=torch.float32).reshape(128, 256)
    out = torch.empty_like(x)
    namespace["kernel"](x, out)
    expected = x.clone()
    expected[target_row : target_row + 64] = x[row : row + 64]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if find_ptoas_binary() is not None:
        source = tmp_path / "disjoint.pto"
        source.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(source), str(tmp_path / "disjoint.cpp"), ["--pto-arch=a2", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("reslice", [False, True])
def test_chained_disjoint_assembles_preserve_allocation_origin(tmp_path, planner, reverse, reslice):
    row = 96 if reverse else 0
    targets = [64, 32, 0] if reverse else [32, 64, 96]
    source = f"source = pl.tile.slice(first, [32, 256], [{targets[0]}, 0])" if reslice else ""
    source_name = "source" if reslice else "window"
    program = pl.parse_program(f"""
@pl.program
class ChainedAssemble:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[128, 256], pl.FP32],
               out: pl.Out[pl.Tensor[[128, 256], pl.FP32]]) -> pl.Tensor[[128, 256], pl.FP32]:
        whole = pl.load(x, [0, 0], [128, 256])
        window = pl.tile.slice(whole, [32, 256], [{row}, 0])
        first = pl.tile.assemble(whole, window, [{targets[0]}, 0])
        {source}
        second = pl.tile.assemble(first, {source_name}, [{targets[1]}, 0])
        third = pl.tile.assemble(second, {source_name}, [{targets[2]}, 0])
        return pl.store(third, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    assert text.count("pto.alloc_tile") == 1
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(128 * 256, dtype=torch.float32).reshape(128, 256)
    out = torch.empty_like(x)
    namespace["kernel"](x, out)
    torch.testing.assert_close(out, x[row : row + 32].repeat(4, 1), rtol=0, atol=0)
    if find_ptoas_binary() is not None:
        source_file = tmp_path / "chained.pto"
        source_file.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(source_file), str(tmp_path / "chained.cpp"), ["--pto-arch=a2", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
def test_chained_overlapping_assemble_snapshots_updated_parent(tmp_path, planner):
    @pl.program
    class ChainedOverlap:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 64], pl.FP32], out: pl.Out[pl.Tensor[[32, 64], pl.FP32]]
        ) -> pl.Tensor[[32, 64], pl.FP32]:
            whole = pl.load(x, [0, 0], [32, 64])
            window = pl.tile.slice(whole, [8, 64], [0, 0])
            first = pl.tile.assemble(whole, window, [8, 0])
            updated = pl.tile.slice(first, [16, 64], [8, 0])
            second = pl.tile.assemble(first, updated, [16, 0])
            return pl.store(second, [0, 0], out)

    lowered = _lower(ChainedOverlap, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    assert text.count("pto.alloc_tile") == 2
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(2048, dtype=torch.float32).reshape(32, 64)
    out = torch.empty_like(x)
    namespace["kernel"](x, out)
    expected = x.clone()
    expected[8:16] = x[:8]
    expected[16:] = expected[8:24].clone()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if find_ptoas_binary() is not None:
        source_file = tmp_path / "chained_overlap.pto"
        source_file.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(
            str(source_file), str(tmp_path / "chained_overlap.cpp"), ["--pto-arch=a2", f"--pto-level={level}"]
        )


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("view_kind", ["allocation", "full_width"])
@pytest.mark.parametrize("source_dtype,target_dtype", [("FP32", "INT32"), ("FP16", "FP32")])
def test_contiguous_reinterpret_preserves_bit_patterns(
    tmp_path, planner, view_kind, source_dtype, target_dtype
):
    rows = 16 if view_kind == "full_width" else 32
    cols = 32 if source_dtype == "FP16" else 64
    view = {
        "allocation": "window = whole",
        "full_width": "window = pl.tile.slice(whole, [16, 64], [16, 0])",
    }[view_kind]
    program = pl.parse_program(f"""
@pl.program
class Reinterpret:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[32, 64], pl.{source_dtype}],
               out: pl.Out[pl.Tensor[[{rows}, {cols}], pl.{target_dtype}]]
               ) -> pl.Tensor[[{rows}, {cols}], pl.{target_dtype}]:
        whole = pl.load(x, [0, 0], [32, 64])
        {view}
        bits = pl.tile.reinterpret_view(window, pl.{target_dtype})
        return pl.store(bits, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    dtype = torch.float16 if source_dtype == "FP16" else torch.float32
    target = torch.int32 if target_dtype == "INT32" else torch.float32
    x = torch.arange(2048, dtype=dtype).reshape(32, 64)
    out = torch.empty((rows, cols), dtype=target)
    namespace["kernel"](x, out)
    expected = (x[16:] if view_kind == "full_width" else x).view(target)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if find_ptoas_binary() is not None:
        source = tmp_path / "reinterpret.pto"
        source.write_text(codegen.PTOCodegen().generate(lowered, emit_source_loc=False))
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(source), str(tmp_path / "reinterpret.cpp"), ["--pto-arch=a2", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("rows", [1, 16])
@pytest.mark.parametrize("operation", ["reinterpret", "reshape"])
def test_strided_relabels_remain_outside_default_buffer_contract(planner, rows, operation):
    dtype = "INT32" if operation == "reinterpret" else "FP32"
    expression = (
        "pl.tile.reinterpret_view(window, pl.INT32)"
        if operation == "reinterpret"
        else f"pl.reshape(window, [{rows}, 32])"
    )
    program = pl.parse_program(f"""
@pl.program
class StridedRelabel:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[32, 64], pl.FP32],
               out: pl.Out[pl.Tensor[[{rows}, 32], pl.{dtype}]]
               ) -> pl.Tensor[[{rows}, 32], pl.{dtype}]:
        whole = pl.load(x, [0, 0], [32, 64])
        window = pl.tile.slice(whole, [{rows}, 32], [0, 0])
        bits = {expression}
        return pl.store(bits, [0, 0], out)
""")
    assert _kernel(_lower(program, planner)).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("planner", _PLANNERS)
def test_reshaped_allocation_is_a_native_window_parent(tmp_path, planner):
    @pl.program
    class ReshapedParent:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 64], pl.FP32], out: pl.Out[pl.Tensor[[8, 64], pl.FP32]]
        ) -> pl.Tensor[[8, 64], pl.FP32]:
            whole = pl.load(x, [0, 0], [32, 64])
            parent = pl.reshape(whole, [16, 128])
            window = pl.tile.slice(parent, [8, 64], [8, 32])
            return pl.store(window, [0, 0], out)

    lowered = _lower(ReshapedParent, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(2048, dtype=torch.float32).reshape(32, 64)
    out = torch.empty((8, 64))
    namespace["kernel"](x, out)
    torch.testing.assert_close(out, x.reshape(16, 128)[8:, 32:96])
    if find_ptoas_binary() is not None:
        source = tmp_path / "parent.pto"
        source.write_text(codegen.PTOCodegen().generate(lowered, emit_source_loc=False))
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(source), str(tmp_path / "parent.cpp"), ["--pto-arch=a2", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
def test_conflicting_typed_window_parents_stay_tile(planner):
    @pl.program
    class ConflictingParents:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[32, 64], pl.FP32]]
        ) -> pl.Tensor[[32, 64], pl.FP32]:
            whole = pl.load(x, [0, 0], [64, 64])
            first = pl.tile.slice(whole, [16, 64], [0, 0])
            out = pl.store(first, [0, 0], out)
            parent = pl.reshape(whole, [32, 128])
            second = pl.tile.slice(parent, [16, 64], [16, 32])
            return pl.store(second, [16, 0], out)

    assert _kernel(_lower(ConflictingParents, planner)).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("dynamic", ["offset", "valid"])
def test_dynamic_window_forms_select_buffer(planner, dynamic):
    offset = "rows" if dynamic == "offset" else "0"
    valid = ", valid_shape=[rows, 32]" if dynamic == "valid" else ""
    program = pl.parse_program(f"""
@pl.program
class DynamicWindow:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], rows: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        whole = pl.load(x, [0, 0], [64, 64])
        window = pl.tile.slice(whole, [16, 32], [{offset}, 0]{valid})
        return pl.store(window, [0, 0], out)
""")
    assert _kernel(_lower(program, planner)).ir_stage == ir.FunctionIRStage.Buffer


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("dynamic", ["static", "offset", "valid", "both"])
def test_default_window_aliases_preserve_storage(tmp_path, planner, dynamic):
    offsets = "[row, col * 8]" if dynamic in ("offset", "both") else "[0, 0]"
    valid = ", valid_shape=[16, cols]" if dynamic in ("valid", "both") else ""
    program = pl.parse_program(f"""
@pl.program
class WindowAliases:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], row: pl.Scalar[pl.INDEX],
               col: pl.Scalar[pl.INDEX], cols: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        whole = pl.load(x, [0, 0], [64, 64])
        window = pl.tile.slice(whole, [16, 32], {offsets}{valid})
        alias = window
        chained = alias
        value = pl.add(alias, chained)
        out = pl.store(whole, [0, 0], out)
        return pl.store(value, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    ir.assert_structural_equal(lowered, ir.deserialize(ir.serialize(lowered)))
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    for row, col, cols in [(0, 0, 0), (1, 1, 1), (32, 2, 32), (8, 1, 0)]:
        out = torch.full_like(x, -777)
        namespace["kernel"](x, row, col, cols, out)
        start_row, start_col = (row, col * 8) if dynamic in ("offset", "both") else (0, 0)
        width = cols if dynamic in ("valid", "both") else 32
        expected = x.clone()
        expected[:16, :width] = 2 * x[start_row : start_row + 16, start_col : start_col + width]
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if find_ptoas_binary() is not None:
        source = tmp_path / "window_aliases.pto"
        source.write_text(codegen.PTOCodegen().generate(lowered, emit_source_loc=False))
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(
            str(source), str(tmp_path / "window_aliases.cpp"), ["--pto-arch=a2", f"--pto-level={level}"]
        )


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["nested", "assemble", "nested_assemble"])
def test_default_runtime_offsets_preserve_source_and_overlap(tmp_path, planner, kind):
    body = {
        "nested": """outer = pl.tile.slice(whole, [32, 48], [row, 0])
        window = pl.tile.slice(outer, [16, 32], [8, col * 8])
        value = pl.add(window, window)
        out = pl.store(whole, [0, 0], out)
        out = pl.store(value, [0, 0], out)""",
        "assemble": """window = pl.tile.slice(whole, [16, 64], [row, 0])
        value = pl.tile.assemble(whole, window, [col, 0])
        out = pl.store(value, [0, 0], out)""",
        "nested_assemble": """outer = pl.tile.slice(whole, [32, 64], [row, 0])
        window = pl.tile.slice(outer, [16, 64], [8, 0])
        value = pl.tile.assemble(whole, window, [col, 0])
        out = pl.store(value, [0, 0], out)""",
    }[kind]
    program = pl.parse_program(f"""
@pl.program
class RuntimeOffsets:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], row: pl.Scalar[pl.INDEX],
               col: pl.Scalar[pl.INDEX], out: pl.InOut[pl.Tensor[[64, 64], pl.FP32]]
               ) -> pl.Tensor[[64, 64], pl.FP32]:
        whole = pl.load(x, [0, 0], [64, 64])
        {body}
        return out
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    ir.assert_structural_equal(lowered, ir.deserialize(ir.serialize(lowered)))
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    # Reuse one compiled body with changing offsets; a frozen first offset
    # or wrong nested pitch must change both the data and untouched sentinel.
    samples = [(0, 0), (1, 1), (16, 2), (32, 0)]
    for row, col in samples:
        out = torch.full_like(x, -777)
        namespace["kernel"](x, row, col, out)
        if kind == "nested":
            expected = x.clone()
            expected[:16, :32] = 2 * x[row + 8 : row + 24, col * 8 : col * 8 + 32]
        else:
            expected = x.clone()
            start = row + (8 if kind == "nested_assemble" else 0)
            expected[col : col + 16] = x[start : start + 16].clone()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert _kernel(_lower(program, planner, False)).ir_stage == ir.FunctionIRStage.Functional
    if find_ptoas_binary() is not None:
        source = tmp_path / "runtime_offsets.pto"
        source.write_text(codegen.PTOCodegen().generate(lowered, emit_source_loc=False))
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(
            str(source), str(tmp_path / "runtime_offsets.cpp"), ["--pto-arch=a2", f"--pto-level={level}"]
        )


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(
    "kind",
    [
        "load",
        "setter",
        "mixed",
        "zero",
        "mixed_zero",
        "chain",
        "two_axes",
        "different_capacity",
        "static_setter_capacity",
    ],
)
def test_default_runtime_valid_empty_and_nonempty_paths(tmp_path, planner, kind):
    valid = (
        "[rows, cols]"
        if kind == "two_axes"
        else "[16, 0]"
        if kind in ("zero", "mixed_zero")
        else "[16, cols]"
    )
    body = f"loaded = pl.load(x, [0, 0], [16, 32], valid_shape={valid})"
    operation = "pl.add(loaded, loaded)"
    if kind == "setter":
        body = """whole = pl.load(x, [0, 0], [16, 32])
        loaded = pl.tile.set_validshape(whole, 16, cols)"""
    elif kind in ("mixed", "mixed_zero"):
        body = f"""whole = pl.load(x, [0, 0], [64, 64])
        window = pl.tile.slice(whole, [16, 32], [0, 0], valid_shape={valid})
        {body}"""
        operation = "pl.add(window, loaded)"
    elif kind == "chain":
        body = """whole = pl.load(x, [0, 0], [16, 32])
        narrowed = pl.tile.set_validshape(whole, 16, cols)
        out = pl.store(narrowed, [32, 0], out)
        loaded = pl.tile.set_validshape(narrowed, 16, 32)"""
    elif kind in ("different_capacity", "static_setter_capacity"):
        # The larger allocation dies before the runtime-valid load. PYPTO must
        # not turn the smaller dynamic allocation into an interior byte view.
        body = f"""whole = pl.load(x, [0, 0], [64, 64])
        out = pl.store(whole, [0, 0], out)
        {body}"""
        if kind == "static_setter_capacity":
            body = """whole = pl.load(x, [0, 0], [64, 64])
        out = pl.store(whole, [0, 0], out)
        full = pl.load(x, [0, 0], [16, 32])
        loaded = pl.tile.set_validshape(full, 16, 7)"""
    program = pl.parse_program(f"""
@pl.program
class RuntimeValid:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], rows: pl.Scalar[pl.INDEX],
               cols: pl.Scalar[pl.INDEX], out: pl.InOut[pl.Tensor[[64, 64], pl.FP32]]
               ) -> pl.Tensor[[64, 64], pl.FP32]:
        {body}
        computed = {operation}
        return pl.store(computed, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    ir.assert_structural_equal(lowered, ir.deserialize(ir.serialize(lowered)))
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    for rows, cols in [(0, 0), (0, 32), (16, 0), (1, 1), (16, 32), (7, 13), (0, 0)]:
        out = torch.full_like(x, -777)
        namespace["kernel"](x, rows, cols, out)
        expected = (
            x.clone()
            if kind in ("different_capacity", "static_setter_capacity")
            else torch.full_like(x, -777)
        )
        vr = rows if kind == "two_axes" else 16
        vc = 32 if kind == "chain" else 0 if kind in ("zero", "mixed_zero") else cols
        if kind == "static_setter_capacity":
            vc = 7
        expected[:vr, :vc] = 2 * x[:vr, :vc]
        if kind == "chain":
            expected[32:48, :cols] = x[:16, :cols]
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if find_ptoas_binary() is not None:
        source = tmp_path / "runtime_valid.pto"
        text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
        source.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(
            str(source), str(tmp_path / "runtime_valid.cpp"), ["--pto-arch=a2", f"--pto-level={level}"]
        )


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("alias", ["direct", "alias", "window"])
def test_runtime_valid_old_metadata_aliases_remain_tile(planner, alias):
    saved = {
        "direct": "",
        "alias": "old = loaded",
        "window": "old = pl.tile.slice(loaded, [16, 32], [0, 0])",
    }[alias]
    old = "loaded" if alias == "direct" else "old"
    program = pl.parse_program(f"""
@pl.program
class OldValid:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32], cols: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[32, 32], pl.FP32]]) -> pl.Tensor[[32, 32], pl.FP32]:
        loaded = pl.load(x, [0, 0], [16, 32])
        {saved}
        narrowed = pl.tile.set_validshape(loaded, 16, cols)
        small = pl.add(narrowed, narrowed)
        out = pl.store(small, [0, 0], out)
        full = pl.add({old}, {old})
        return pl.store(full, [16, 0], out)
""")
    assert _kernel(_lower(program, planner)).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(
    "kind", ["unaligned_offset", "window_setter", "nested_valid", "different_valid", "if", "for", "while"]
)
def test_dynamic_unsupported_forms_remain_tile(planner, kind):
    body = {
        "unaligned_offset": "value = pl.tile.slice(whole, [16, 32], [0, cols])",
        "window_setter": """window = pl.tile.slice(whole, [16, 32], [0, 0])
        value = pl.tile.set_validshape(window, 16, cols)""",
        "nested_valid": """window = pl.tile.slice(whole, [32, 64], [0, 0], valid_shape=[32, cols])
        value = pl.tile.slice(window, [16, 32], [0, 0], valid_shape=[16, cols])""",
        "if": """loaded = pl.load(x, [0, 0], [16, 32], valid_shape=[16, cols])
        if other > 0:
            value = pl.add(loaded, loaded)
        else:
            value = pl.sub(loaded, loaded)""",
        "for": """loaded = pl.load(x, [0, 0], [16, 32], valid_shape=[16, cols])
        for index in pl.range(2):
            loaded = pl.add(loaded, loaded)
        value = loaded""",
        "while": """loaded = pl.load(x, [0, 0], [16, 32], valid_shape=[16, cols])
        count = other
        while count > 0:
            loaded = pl.add(loaded, loaded)
            count = count - 1
        value = loaded""",
        "different_valid": """left = pl.load(x, [0, 0], [16, 32], valid_shape=[16, cols])
        right = pl.load(x, [0, 0], [16, 32], valid_shape=[16, other])
        value = pl.add(left, right)""",
    }[kind]
    program = pl.parse_program(f"""
@pl.program
class UnsupportedDynamic:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], cols: pl.Scalar[pl.INDEX],
               other: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]
               ) -> pl.Tensor[[16, 32], pl.FP32]:
        whole = pl.load(x, [0, 0], [64, 64])
        {body}
        return pl.store(value, [0, 0], out)
""")
    assert _kernel(_lower(program, planner)).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("dtype,scale", [("FP16", 16), ("FP32", 8)])
@pytest.mark.parametrize("kind", ["slice_valid", "slice_drop", "assemble"])
@pytest.mark.parametrize("offset_kind", ["raw", "aligned", "misaligned", "constant"])
def test_window_selection_proves_byte_alignment(planner, dtype, scale, kind, offset_kind):
    offset = {
        "raw": "col",
        "aligned": "scaled + step",
        "misaligned": "scaled + 1",
        "constant": "1",
    }[offset_kind]
    if kind == "assemble":
        body = f"""window = pl.tile.slice(whole, [16, 32], [0, 0])
        value = pl.tile.assemble(whole, window, [row, {offset}])"""
    else:
        drop = ", drop_dims=[]" if kind == "slice_drop" else ""
        body = f"value = pl.tile.slice(whole, [16, 32], [row, {offset}], valid_shape=[16, cols]{drop})"
    program = pl.parse_program(f"""
@pl.program
class WindowAlignment:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.{dtype}], row: pl.Scalar[pl.INDEX],
               col: pl.Scalar[pl.INDEX], cols: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[64, 64], pl.{dtype}]]) -> pl.Tensor[[64, 64], pl.{dtype}]:
        step = {scale}
        scaled = col * step
        whole = pl.load(x, [0, 0], [64, 64])
        {body}
        return pl.store(value, [0, 0], out)
""")
    expected = ir.FunctionIRStage.Buffer if offset_kind == "aligned" else ir.FunctionIRStage.Functional
    assert _kernel(_lower(program, planner)).ir_stage == expected


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("valid_cols", ["16", "cols"])
def test_standalone_valid_shape_setter_remains_tile(tmp_path, planner, valid_cols):
    program = pl.parse_program(f"""
@pl.program
class StandaloneSetter:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32], cols: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        whole = pl.load(x, [0, 0], [16, 32])
        pl.tile.set_validshape(whole, 16, {valid_cols})
        return pl.store(whole, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Functional
    if find_ptoas_binary() is not None:
        source = tmp_path / "standalone_setter.pto"
        source.write_text(
            codegen.PTOCodegen().generate(
                lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS, emit_source_loc=False
            )
        )
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(
            str(source), str(tmp_path / "standalone_setter.cpp"), ["--pto-arch=a2", f"--pto-level={level}"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for PTOCodegen - MLIR generation from PyPTO IR.

The new PTOCodegen generates PTO-ISA MLIR dialect instead of PTO assembly.
Tests verify:
- Correct MLIR module structure
- Proper function signatures with tensor pointers
- make_tensor_view generation for tensor parameters
- alloc_tile generation for tile buffers
- Operator lowering (tile.load/store/mul/adds -> pto.tload/tstore/tmul/tadds)
- SSA form with correct variable naming
"""

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir
from pypto.backend import BackendType
from pypto.backend.pto_backend import (
    _format_error_report,
    _generate_arg_unpacking,
    _generate_kernel_wrapper,
    _preprocess_ptoas_output,
    generate,
)
from pypto.ir import OptimizationStrategy, PassManager
from pypto.ir.builder import IRBuilder
from pypto.ir.op import tile

PTOCodegen = codegen.PTOCodegen

# Dynamic shape variables for wrapper dispatch tests
# pyright: reportUndefinedVariable=false
_TH = pl.dynamic("TH")
_TW = pl.dynamic("TW")


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure PTO backend before each test."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)
    yield
    backend.reset_for_testing()


@pl.program
class _DynKernel:
    """Dynamic shape kernel used in wrapper dispatch tests."""

    @pl.function(type=pl.FunctionType.InCore)
    def dyn_func(
        self,
        a: pl.Tensor[[_TH, _TW], pl.FP32],
        b: pl.Tensor[[_TH, _TW], pl.FP32],
        output: pl.Tensor[[_TH, _TW], pl.FP32],
    ) -> pl.Tensor[[_TH, _TW], pl.FP32]:
        a_tile = pl.load(a, [0, 0], [128, 128])
        b_tile = pl.load(b, [0, 0], [128, 128])
        result = pl.add(a_tile, b_tile)
        return pl.store(result, [0, 0], output)


def _get_dyn_incore_func():
    """Return the transformed InCore function from _DynKernel."""
    transformed = _run_default_passes(_DynKernel)
    for func in transformed.functions.values():
        if ir.is_incore_type(func.func_type):
            return func
    raise RuntimeError("No InCore function found in _DynKernel")


def _get_mlir_code(result):
    """Normalize generate() result to MLIR string (support both str and dict)."""
    return result if isinstance(result, str) else "".join(result.values())


def _run_default_passes(program_cls):
    """Run the default pass pipeline for a program class."""
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    return pm.run_passes(program_cls)


def _generate_mlir(program: ir.Program) -> str:
    """Generate MLIR for an already-built program."""
    return _get_mlir_code(PTOCodegen().generate(program))


def _generate_default_mlir(program_cls) -> str:
    """Run default passes then generate MLIR for a program class."""
    return _generate_mlir(_run_default_passes(program_cls))


def _get_mlir_lines(mlir_code: str) -> list[str]:
    """Return stripped MLIR lines for line-oriented assertions."""
    return [line.strip() for line in mlir_code.splitlines()]


def _get_alloc_tile_lines(mlir_code: str) -> list[str]:
    """Return normalized alloc_tile lines from generated MLIR."""
    return [line.strip() for line in mlir_code.splitlines() if "pto.alloc_tile" in line]


def _find_lines(lines: list[str], token: str, *, startswith: bool = False) -> list[str]:
    """Return MLIR lines matching a token."""
    if startswith:
        return [line for line in lines if line.startswith(token)]
    return [line for line in lines if token in line]


def _single_line(lines: list[str], token: str, *, startswith: bool = False) -> str:
    """Return the single MLIR line matching a token."""
    matched = _find_lines(lines, token, startswith=startswith)
    assert len(matched) == 1, f"Expected one line containing {token!r}, got: {matched}"
    return matched[0]


SAMPLE_PTOAS_OUTPUT = """\
#include "pto/pto-inst.hpp"
using namespace pto;

\ttemplate <typename To, typename From>
\tstatic inline To ptoas_bitcast(From from) {
\t  static_assert(sizeof(To) == sizeof(From), "ptoas_bitcast: size mismatch");
\t  To to;
\t  __builtin_memcpy(&to, &from, sizeof(To));
\t  return to;
\t}
\t
__global__ AICORE void test_func(__gm__ float* v1, float v2, __gm__ float* v3) {
  TLOAD(v1);
  TADDS(v2);
  TSTORE(v3);
  return;
}
"""


def _make_func(name, params_spec):
    """Build a Function from parameter specs.

    Args:
        name: Function name.
        params_spec: list of (param_name, "tensor"|"scalar") tuples.

    Returns:
        ir.Function with InCore type.
    """
    ib = IRBuilder()
    with ib.function(name, type=ir.FunctionType.InCore) as f:
        param_vars = []
        for pname, kind in params_spec:
            if kind == "tensor":
                param_vars.append(f.param(pname, ir.TensorType([16, 16], DataType.FP32)))
            elif kind == "scalar":
                param_vars.append(f.param(pname, ir.ScalarType(DataType.FP32)))

        # Minimal body: load first tensor param → store
        tensor_params = [v for v, (_, k) in zip(param_vars, params_spec) if k == "tensor"]
        if len(tensor_params) >= 2:
            t = ib.let("t", tile.load(tensor_params[0], [0, 0], [16, 16]))
            result = ib.let("result", tile.store(t, [0, 0], tensor_params[-1]))
            f.return_type(ir.TensorType([16, 16], DataType.FP32))
            ib.return_stmt(result)
        elif len(tensor_params) == 1:
            t = ib.let("t", tile.load(tensor_params[0], [0, 0], [16, 16]))
            result = ib.let("result", tile.store(t, [0, 0], tensor_params[0]))
            f.return_type(ir.TensorType([16, 16], DataType.FP32))
            ib.return_stmt(result)
        else:
            f.return_type(ir.ScalarType(DataType.FP32))
            ib.return_stmt(param_vars[0])

    return f.get_result()


def test_pto_codegen_basic_mlir_structure():
    """Test that PTOCodegen generates valid MLIR module structure."""

    @pl.program
    class BasicProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def test_func(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.add(tile_a, 1.0)
            pl.store(tile_b, offsets=[0, 0], output_tensor=b)

    mlir_code = _generate_default_mlir(BasicProgram)

    # Verify MLIR module structure
    assert "module attributes {pto.target_arch =" in mlir_code
    assert "func.func @test_func" in mlir_code
    assert "return" in mlir_code
    assert "}" in mlir_code


def test_pto_codegen_tensor_parameters():
    """Test that tensor parameters generate correct make_tensor_view."""

    @pl.program
    class TensorParamProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def tensor_param_func(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ):
            tile_a = pl.load(input_a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.load(input_b, offsets=[0, 0], shapes=[32, 32])
            tile_c = pl.mul(tile_a, tile_b)
            pl.store(tile_c, offsets=[0, 0], output_tensor=output)

    mlir_code = _generate_default_mlir(TensorParamProgram)

    # Verify function signature with pointer types
    assert "%arg0: !pto.ptr<f32>" in mlir_code
    assert "%arg1: !pto.ptr<f32>" in mlir_code
    assert "%arg2: !pto.ptr<f32>" in mlir_code

    # Verify make_tensor_view generation
    assert "pto.make_tensor_view" in mlir_code
    assert "shape = [%c64, %c64]" in mlir_code or "shape = [%c32, %c32]" in mlir_code
    assert "strides = " in mlir_code
    assert "!pto.tensor_view<?x?xf32>" in mlir_code


def test_pto_codegen_alloc_tile():
    """Test that tile buffers generate alloc_tile operations."""

    @pl.program
    class AllocTileProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def alloc_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_c = pl.mul(tile_a, tile_b)
            pl.store(tile_c, offsets=[0, 0], output_tensor=b)

    alloc_lines = _get_alloc_tile_lines(_generate_default_mlir(AllocTileProgram))
    assert len(alloc_lines) > 0, "Expected at least one alloc_tile"
    for alloc_line in alloc_lines:
        assert "loc=vec" in alloc_line, f"Expected vector alloc_tile, got: {alloc_line}"
        assert "dtype=f32" in alloc_line, f"Expected f32 alloc_tile, got: {alloc_line}"
        assert "rows=32, cols=32" in alloc_line, f"Expected 32x32 alloc_tile, got: {alloc_line}"


def test_pto_codegen_fillpad_shared_memref_uses_single_alloc_tile():
    """Test that shared MemRef tiles emit one alloc_tile and preserve merged TileView info."""
    span = ir.Span.unknown()
    zero = ir.ConstInt(0, DataType.INDEX, span)
    size = ir.ConstInt(128, DataType.INDEX, span)

    input_tensor = ir.Var("a", ir.TensorType([128, 128], DataType.FP32), span)
    output_tensor = ir.Var("output", ir.TensorType([128, 128], DataType.FP32), span)
    m_var = ir.Var("m", ir.ScalarType(DataType.INDEX), span)
    n_var = ir.Var("n", ir.ScalarType(DataType.INDEX), span)
    shared_memref = ir.MemRef(ir.MemorySpace.Vec, zero, 128 * 128 * 4, 0)

    load_view = ir.TileView()
    load_view.valid_shape = [m_var, n_var]
    load_tile_type = ir.TileType([128, 128], DataType.FP32, shared_memref, load_view, ir.MemorySpace.Vec)
    load_tile = ir.Var("tile_a", load_tile_type, span)

    padded_view = ir.TileView()
    padded_view.valid_shape = [size, size]
    padded_view.pad = ir.PadValue.max
    padded_tile_type = ir.TileType([128, 128], DataType.FP32, shared_memref, padded_view, ir.MemorySpace.Vec)
    padded_tile = ir.Var("padded", padded_tile_type, span)

    result_var = ir.Var("result", ir.TensorType([128, 128], DataType.FP32), span)
    offsets = ir.MakeTuple([zero, zero], span)
    shapes = ir.MakeTuple([size, size], span)

    load_call = ir.Call(ir.Op("tile.load"), [input_tensor, offsets, shapes], {}, load_tile_type, span)
    fillpad_call = ir.Call(
        ir.Op("tile.fillpad"),
        [load_tile],
        {"pad_value": ir.PadValue.max},
        padded_tile_type,
        span,
    )
    assert fillpad_call.kwargs["pad_value"] == ir.PadValue.max
    store_call = ir.Call(ir.Op("tile.store"), [padded_tile, offsets, output_tensor], result_var.type, span)

    body = ir.SeqStmts(
        [
            ir.SeqStmts(
                [
                    ir.AssignStmt(load_tile, load_call, span),
                    ir.AssignStmt(padded_tile, fillpad_call, span),
                    ir.AssignStmt(result_var, store_call, span),
                ],
                span,
            ),
            ir.ReturnStmt([result_var], span),
        ],
        span,
    )
    func = ir.Function(
        "fillpad_test",
        [
            (input_tensor, ir.ParamDirection.In),
            (output_tensor, ir.ParamDirection.Out),
            (m_var, ir.ParamDirection.In),
            (n_var, ir.ParamDirection.In),
        ],
        [ir.TensorType([128, 128], DataType.FP32)],
        body,
        span,
        ir.FunctionType.InCore,
    )
    program = ir.Program([func], "fillpad_test_program", span)

    mlir_code = _generate_mlir(program)
    alloc_lines = _get_alloc_tile_lines(mlir_code)

    assert len(alloc_lines) == 2, f"Expected two alloc_tiles for per-var alloc model, got: {alloc_lines}"
    # Both share the same addr (same MemRef)
    assert "addr = %c0i" in alloc_lines[0]
    assert "addr = %c0i" in alloc_lines[1]
    # One should carry valid_row/valid_col dynamic shapes
    dynamic_allocs = [line for line in alloc_lines if "valid_row = %arg2 valid_col = %arg3" in line]
    assert len(dynamic_allocs) >= 1
    assert "v_row=?" in alloc_lines[0]
    assert "v_col=?" in alloc_lines[0]
    assert "pad=" in alloc_lines[1]
    assert "pad=2>" in alloc_lines[1], f"Expected fillpad pad metadata to be preserved: {alloc_lines[1]}"


def test_pto_codegen_dynamic_valid_shape_scalar_defined_in_body():
    """Dynamic valid_shape scalars defined in-body should still reach alloc_tile."""

    @pl.program
    class DynamicValidShapeScalarProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def body_valid_shape(
            self,
            input: pl.Tensor[[1, 120], pl.FP32],
            ctx_len: pl.Scalar[pl.INDEX],
            output: pl.Tensor[[1, 120], pl.FP32],
        ) -> pl.Tensor[[1, 120], pl.FP32]:
            valid_len: pl.Scalar[pl.INDEX] = ctx_len + 0
            tile: pl.Tile[[1, 120], pl.FP32] = pl.tile.load(
                input,
                [0, 0],
                [1, 120],
                [1, valid_len],
                target_memory=pl.MemorySpace.Vec,
                transpose=False,
            )
            result: pl.Tensor[[1, 120], pl.FP32] = pl.tile.store(tile, [0, 0], output)
            return result

    mlir_code = _generate_default_mlir(DynamicValidShapeScalarProgram)
    alloc_lines = _get_alloc_tile_lines(mlir_code)

    assert len(alloc_lines) == 1, f"Expected one alloc_tile, got: {alloc_lines}"
    alloc_line = alloc_lines[0]
    assert "valid_col = %" in alloc_line, (
        f"Expected alloc_tile to reference in-body valid_shape SSA, got: {alloc_line}"
    )
    assert "valid_row = %" not in alloc_line, f"Did not expect dynamic valid_row in alloc_tile: {alloc_line}"
    assert "v_row=1" in alloc_line, f"Expected static v_row=1 in tile_buf type, got: {alloc_line}"
    assert "v_col=?" in alloc_line, f"Expected dynamic v_col in tile_buf type, got: {alloc_line}"
    assert "valid_col = %arg" not in alloc_line, (
        f"Expected valid_shape SSA from body, not direct arg reuse: {alloc_line}"
    )
    assert "%c-1" not in mlir_code


def test_pto_codegen_dynamic_valid_shape_row_defined_in_body():
    """Dynamic valid_shape rows defined in-body should still reach alloc_tile."""

    @pl.program
    class DynamicValidShapeRowProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def body_valid_row(
            self,
            input: pl.Tensor[[120, 16], pl.FP32],
            ctx_rows: pl.Scalar[pl.INDEX],
            output: pl.Tensor[[120, 16], pl.FP32],
        ) -> pl.Tensor[[120, 16], pl.FP32]:
            valid_rows: pl.Scalar[pl.INDEX] = ctx_rows + 0
            tile: pl.Tile[[120, 16], pl.FP32] = pl.tile.load(
                input,
                [0, 0],
                [120, 16],
                [valid_rows, 16],
                target_memory=pl.MemorySpace.Vec,
                transpose=False,
            )
            result: pl.Tensor[[120, 16], pl.FP32] = pl.tile.store(tile, [0, 0], output)
            return result

    mlir_code = _generate_default_mlir(DynamicValidShapeRowProgram)
    alloc_lines = _get_alloc_tile_lines(mlir_code)

    assert len(alloc_lines) == 1, f"Expected one alloc_tile, got: {alloc_lines}"
    alloc_line = alloc_lines[0]
    assert "valid_row = %" in alloc_line, (
        f"Expected alloc_tile to reference in-body valid_shape SSA, got: {alloc_line}"
    )
    assert "valid_col = %" not in alloc_line, f"Did not expect dynamic valid_col in alloc_tile: {alloc_line}"
    assert "v_row=?" in alloc_line, f"Expected dynamic v_row in tile_buf type, got: {alloc_line}"
    assert "v_col=16" in alloc_line, f"Expected static v_col=16 in tile_buf type, got: {alloc_line}"
    assert "valid_row = %arg" not in alloc_line, (
        f"Expected valid_shape SSA from body, not direct arg reuse: {alloc_line}"
    )


def test_pto_codegen_tile_load_lowering():
    """Test that tile.load generates partition_view + tload."""

    @pl.program
    class LoadProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def load_test(self, input: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]):
            tile = pl.load(input, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], output_tensor=output)

    mlir_code = _generate_default_mlir(LoadProgram)

    # Verify partition_view generation
    assert "pto.partition_view" in mlir_code
    assert "offsets = [%c0, %c0]" in mlir_code
    assert "sizes = [%c32, %c32]" in mlir_code
    assert "!pto.partition_tensor_view<32x32xf32>" in mlir_code

    # Verify tload generation
    assert "pto.tload" in mlir_code
    assert "ins(" in mlir_code
    assert "outs(" in mlir_code
    assert "!pto.tile_buf<" in mlir_code


def test_pto_codegen_tile_store_lowering():
    """Test that tile.store generates partition_view + tstore."""

    @pl.program
    class StoreProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def store_test(self, input: pl.Tensor[[32, 32], pl.FP32], output: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(input, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], output_tensor=output)

    mlir_code = _generate_default_mlir(StoreProgram)

    # Verify tstore generation
    assert "pto.tstore" in mlir_code
    assert "ins(" in mlir_code
    assert "outs(" in mlir_code


def test_pto_codegen_tile_mul():
    """Test that tile.mul generates pto.tmul."""

    @pl.program
    class MulProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def mul_test(
            self,
            a: pl.Tensor[[32, 32], pl.FP32],
            b: pl.Tensor[[32, 32], pl.FP32],
            c: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 32])
            tile_c = pl.mul(tile_a, tile_b)
            pl.store(tile_c, offsets=[0, 0], output_tensor=c)

    mlir_code = _generate_default_mlir(MulProgram)

    # Verify tmul generation
    assert "pto.tmul" in mlir_code
    assert "ins(" in mlir_code
    assert "outs(" in mlir_code


def test_pto_codegen_tile_adds():
    """Test that tile.adds generates pto.tadds with scalar constant."""

    @pl.program
    class AddsProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def adds_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.add(tile_a, 3.14)
            pl.store(tile_b, offsets=[0, 0], output_tensor=b)

    mlir_code = _generate_default_mlir(AddsProgram)

    # Verify tadds generation
    assert "pto.tadds" in mlir_code

    # Verify scalar constant generation
    assert "arith.constant" in mlir_code
    assert ": f32" in mlir_code


def test_pto_codegen_constants():
    """Test that constants are generated correctly."""

    @pl.program
    class ConstantProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def const_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile_a, offsets=[0, 0], output_tensor=b)

    mlir_code = _generate_default_mlir(ConstantProgram)

    # Verify index constants
    assert "arith.constant" in mlir_code
    assert ": index" in mlir_code
    assert "%c0" in mlir_code or "%c32" in mlir_code


def test_pto_codegen_ssa_naming():
    """Test that SSA value names are correct."""

    @pl.program
    class SSAProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def ssa_test(
            self,
            a: pl.Tensor[[32, 32], pl.FP32],
            b: pl.Tensor[[32, 32], pl.FP32],
            c: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 32])
            tile_c = pl.mul(tile_a, tile_b)
            pl.store(tile_c, offsets=[0, 0], output_tensor=c)

    mlir_code = _generate_default_mlir(SSAProgram)

    # Verify SSA value naming pattern
    assert "%arg0" in mlir_code  # Function parameters
    # SSA variables use IR-derived names (e.g., %a_0_view) or numeric fallbacks
    assert "%" in mlir_code  # SSA values present
    assert "%c" in mlir_code  # Constants


def test_pto_codegen_code_generation_order():
    """Test that code is generated in correct order: constants, views, allocs, body."""

    @pl.program
    class OrderProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def order_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], output_tensor=b)

    lines = _get_mlir_lines(_generate_default_mlir(OrderProgram))

    # Find indices of key operations
    const_idx = next((i for i, line in enumerate(lines) if "arith.constant" in line), -1)
    view_idx = next((i for i, line in enumerate(lines) if "make_tensor_view" in line), -1)
    alloc_idx = next((i for i, line in enumerate(lines) if "alloc_tile" in line), -1)
    load_idx = next((i for i, line in enumerate(lines) if "tload" in line), -1)

    # Verify order: constants < make_tensor_view < alloc_tile < operations
    assert const_idx < view_idx, "Constants should come before make_tensor_view"
    assert view_idx < alloc_idx, "make_tensor_view should come before alloc_tile"
    assert alloc_idx < load_idx, "alloc_tile should come before tload"


def test_pto_codegen_multiple_functions():
    """Test PTOCodegen with multiple functions."""

    @pl.program
    class MultiFunc:
        @pl.function(type=pl.FunctionType.InCore)
        def func1(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], output_tensor=b)

        @pl.function(type=pl.FunctionType.InCore)
        def func2(self, x: pl.Tensor[[32, 32], pl.FP32], y: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(x, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], output_tensor=y)

    mlir_code = _generate_default_mlir(MultiFunc)

    # Verify both functions are present
    assert "func.func @func1" in mlir_code
    assert "func.func @func2" in mlir_code


def test_pto_codegen_reusability():
    """Test that the same PTOCodegen instance can be used multiple times."""

    @pl.program
    class ReusableProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def test_func(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], output_tensor=b)

    transformed_program = _run_default_passes(ReusableProgram)

    # Use the same codegen instance multiple times
    codegen = PTOCodegen()

    code1 = _get_mlir_code(codegen.generate(transformed_program))
    code2 = _get_mlir_code(codegen.generate(transformed_program))

    # Verify both calls produce valid code
    assert isinstance(code1, str)
    assert isinstance(code2, str)
    assert "func.func @test_func" in code1
    assert "func.func @test_func" in code2
    assert code1 == code2  # Should produce identical output


# --- Kernel wrapper generation tests ---


class TestPreprocessPtoasOutput:
    """Tests for _preprocess_ptoas_output."""

    def test_strips_include(self):
        result = _preprocess_ptoas_output(SAMPLE_PTOAS_OUTPUT)
        assert '#include "pto/pto-inst.hpp"' not in result

    def test_strips_using_namespace(self):
        result = _preprocess_ptoas_output(SAMPLE_PTOAS_OUTPUT)
        assert "using namespace pto;" not in result

    def test_replaces_global_aicore(self):
        result = _preprocess_ptoas_output(SAMPLE_PTOAS_OUTPUT)
        assert "__global__ AICORE void" not in result
        assert "static __aicore__ void test_func" in result

    def test_preserves_function_body(self):
        result = _preprocess_ptoas_output(SAMPLE_PTOAS_OUTPUT)
        assert "TLOAD(v1);" in result
        assert "TADDS(v2);" in result
        assert "TSTORE(v3);" in result

    def test_preserves_helpers(self):
        result = _preprocess_ptoas_output(SAMPLE_PTOAS_OUTPUT)
        assert "ptoas_bitcast" in result


class TestGenerateArgUnpacking:
    """Tests for _generate_arg_unpacking."""

    def test_tensor_only(self):
        func = _make_func("test_fn", [("a", "tensor"), ("b", "tensor"), ("out", "tensor")])
        code, names = _generate_arg_unpacking(func)
        assert "reinterpret_cast<__gm__ TensorData*>(args[0])" in code
        assert "reinterpret_cast<__gm__ TensorData*>(args[1])" in code
        assert "reinterpret_cast<__gm__ TensorData*>(args[2])" in code
        assert names == ["a", "b", "out"]

    def test_mixed_tensor_scalar(self):
        func = _make_func("test_fn", [("input", "tensor"), ("scale", "scalar"), ("output", "tensor")])
        code, names = _generate_arg_unpacking(func)
        assert "reinterpret_cast<__gm__ TensorData*>(args[0])" in code
        assert "scale_conv.u64 = args[1];" in code
        assert "float scale = scale_conv.val;" in code
        assert "reinterpret_cast<__gm__ TensorData*>(args[2])" in code
        assert names == ["input", "scale", "output"]

    def test_scalar_only(self):
        func = _make_func("test_fn", [("x", "scalar"), ("y", "scalar")])
        code, names = _generate_arg_unpacking(func)
        assert "x_conv.u64 = args[0];" in code
        assert "y_conv.u64 = args[1];" in code
        assert names == ["x", "y"]

    def test_dynamic_tensor_extracts_shapes_dims(self):
        func = _get_dyn_incore_func()
        code, names = _generate_arg_unpacking(func)
        # TH is dim 0 of first tensor a__ssa_v0 — read from a__ssa_v0_tensor->shapes[0]
        assert "a__ssa_v0_tensor->shapes[0]" in code
        assert "int64_t TH" in code
        # TW is dim 1 of first tensor a__ssa_v0 — read from a__ssa_v0_tensor->shapes[1]
        assert "a__ssa_v0_tensor->shapes[1]" in code
        assert "int64_t TW" in code
        # dynamic dims appended after tensor params
        assert names == ["a__ssa_v0", "b__ssa_v0", "output__ssa_v0", "TH", "TW"]

    def test_dynamic_tensor_deduplicates_vars(self):
        # TH and TW each appear in a__ssa_v0, b__ssa_v0, and output__ssa_v0 but should be extracted only once
        func = _get_dyn_incore_func()
        code, names = _generate_arg_unpacking(func)
        assert code.count("int64_t TH") == 1
        assert code.count("int64_t TW") == 1


class TestGenerateKernelWrapper:
    """Tests for _generate_kernel_wrapper."""

    def test_contains_kernel_entry(self):
        func = _make_func("my_kernel", [("a", "tensor"), ("s", "scalar"), ("out", "tensor")])
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        assert "void kernel_entry(__gm__ int64_t* args)" in wrapper

    def test_contains_includes(self):
        func = _make_func("my_kernel", [("a", "tensor"), ("s", "scalar"), ("out", "tensor")])
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        assert "#include <cstdint>" in wrapper
        assert "#include <pto/pto-inst.hpp>" in wrapper
        assert '#include "tensor.h"' in wrapper

    def test_contains_forward_call(self):
        func = _make_func("my_kernel", [("a", "tensor"), ("s", "scalar"), ("out", "tensor")])
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        assert "my_kernel(a, s, out);" in wrapper

    def test_ptoas_code_made_static(self):
        func = _make_func("my_kernel", [("a", "tensor"), ("s", "scalar"), ("out", "tensor")])
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        assert "__global__ AICORE" not in wrapper
        assert "static __aicore__ void test_func" in wrapper

    def test_no_duplicate_includes(self):
        func = _make_func("my_kernel", [("a", "tensor"), ("s", "scalar"), ("out", "tensor")])
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        count = wrapper.count("#include <pto/pto-inst.hpp>")
        assert count == 1, f"Expected 1 pto-inst include, found {count}"

    def test_dynamic_shape_forward_call_includes_dims(self):
        func = _get_dyn_incore_func()
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        # Forward call must include dynamic dims TH and TW after tensor args.
        assert "dyn_func(a__ssa_v0, b__ssa_v0, output__ssa_v0, TH, TW);" in wrapper

    def test_dynamic_shape_shapes_extraction_in_wrapper(self):
        func = _get_dyn_incore_func()
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        assert "a__ssa_v0_tensor->shapes[0]" in wrapper
        assert "a__ssa_v0_tensor->shapes[1]" in wrapper


class TestGenerateSkipPtoas:
    """Tests for generate() with skip_ptoas=True."""

    def test_returns_pto_files(self, tmp_path):
        """When skip_ptoas=True, result keys for InCore functions end with .pto, not .cpp."""

        @pl.program
        class SkipPtoasProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def skip_test(
                self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
                out = pl.store(tile, offsets=[0, 0], output_tensor=b)
                return out

        transformed_program = _run_default_passes(SkipPtoasProgram)

        result = generate(transformed_program, str(tmp_path), skip_ptoas=True)

        kernel_keys = [k for k in result if k.startswith("kernels/")]
        assert len(kernel_keys) > 0, "Expected at least one kernel file"
        for key in kernel_keys:
            assert key.endswith(".pto"), f"Expected .pto extension, got: {key}"
            assert not key.endswith(".cpp"), f"Unexpected .cpp extension: {key}"


def test_compile_writes_orchestration_on_partial_codegen_failure(tmp_path):
    """compile() should preserve generated files when some InCore functions fail."""

    @pl.program
    class PartialFailureProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def good_kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            tile = pl.load(a, offsets=[0, 0], shapes=[16, 16])
            out = pl.store(tile, offsets=[0, 0], output_tensor=output)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def bad_kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            tile = pl.load(a, offsets=[0, 0], shapes=[16, 16])
            result = pl.tile.sum(tile, axis=1)
            out = pl.store(result, offsets=[0, 0], output_tensor=output)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out = pl.create_tensor([16, 16], dtype=pl.FP32)
            out = self.good_kernel(a, out)
            return out

    output_dir = tmp_path / "partial_codegen"
    with pytest.raises(RuntimeError, match="bad_kernel"):
        ir.compile(
            PartialFailureProgram,
            output_dir=str(output_dir),
            strategy=OptimizationStrategy.Default,
            dump_passes=False,
            backend_type=BackendType.Ascend910B_PTO,
            skip_ptoas=True,
        )

    assert (output_dir / "orchestration" / "orch.cpp").exists()
    assert (output_dir / "kernels" / "aiv" / "good_kernel.pto").exists()


class TestFormatErrorReport:
    """Tests for codegen error summary formatting."""

    def test_summary_lists_function_name_first(self, tmp_path):
        report = _format_error_report(
            [
                ("vector_func", RuntimeError("vector_func invalid tile shape\n\nC++ Traceback:\n...")),
                ("cube_func", ValueError("cube_func unsupported memory space")),
            ],
            str(tmp_path),
        )

        assert "2 function(s) failed to compile:" in report
        assert "  Function" in report
        assert "| Error" in report
        assert "  vector_func" in report
        assert "  cube_func" in report
        assert "| vector_func" not in report
        assert "| cube_func" not in report
        assert "invalid tile shape | vector_func" not in report
        assert "unsupported memory space | cube_func" not in report
        assert "| invalid tile shape" in report
        assert "| unsupported memory space" in report

    def test_summary_does_not_group_same_error(self, tmp_path):
        report = _format_error_report(
            [
                ("func_a", RuntimeError("func_a same failure")),
                ("func_b", RuntimeError("func_b same failure")),
            ],
            str(tmp_path),
        )

        assert report.count("| same failure") == 2
        assert "  func_a" in report
        assert "  func_b" in report


def test_pto_codegen_for_loop_tensor_iter_arg():
    """Test that tensor-typed iter_args are excluded from PTO scf.for iter_args/yield.

    In PTO, tensor views are reference types. Only scalar types need iter_args/yield
    for loop-carried value semantics. Tensor iter_args are mapped directly to their
    init values (the output tensor view), and the generated scf.for should not contain
    iter_args or scf.yield for tensor types.
    """

    @pl.program
    class ForTensorIterArgProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def loop_store(
            self,
            a: pl.Tensor[[128, 64], pl.FP32],
            output: pl.Tensor[[128, 64], pl.FP32],
        ) -> pl.Tensor[[128, 64], pl.FP32]:
            for i, (out_iter,) in pl.range(2, init_values=(output,)):
                offset_i: pl.Scalar[pl.INDEX] = i * 64
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [offset_i, 0], [64, 64])
                updated: pl.Tensor[[128, 64], pl.FP32] = pl.store(tile, [offset_i, 0], out_iter)
                result = pl.yield_(updated)
            return result

    lines = _get_mlir_lines(_generate_default_mlir(ForTensorIterArgProgram))

    # The output tensor parameter (%arg1) must have a make_tensor_view
    output_view_line = _single_line(lines, "pto.make_tensor_view %arg1")
    output_view_name = output_view_line.split("=")[0].strip()

    # scf.for should NOT have iter_args (tensor is non-scalar, excluded)
    for_line = _single_line(lines, "scf.for")
    assert "iter_args(" not in for_line, f"scf.for should not have iter_args for tensor types: {for_line}"

    # No scf.yield should be present (tensor yields are excluded)
    yield_lines = _find_lines(lines, "scf.yield")
    assert len(yield_lines) == 0, f"No scf.yield expected for tensor-only iter_args: {yield_lines}"

    # pto.partition_view must use the output tensor view directly (mapped from iter_arg)
    partition_lines = _find_lines(lines, "pto.partition_view")
    assert len(partition_lines) >= 2, "Expected at least 2 partition_view ops (load + store)"
    store_partitions = [line for line in partition_lines if f"pto.partition_view {output_view_name}," in line]
    assert len(store_partitions) >= 1, (
        f"Expected partition_view on output tensor view {output_view_name} for store path"
    )

    # pto.tstore must still be present
    _single_line(lines, "pto.tstore", startswith=True)


def test_pto_codegen_for_loop_tile_iter_arg_no_ddr_alloc():
    """Test that tile-typed iter_args are excluded from PTO scf.for iter_args/yield.

    In PTO, tile buffers are mutable references written in-place via outs().
    Only scalar types need iter_args/yield for loop-carried value semantics.
    Tile-typed iter_args should be mapped directly to their init values, and
    the generated scf.for should not contain iter_args or scf.yield for tiles.
    """

    @pl.program
    class TileIterArgProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def accumulate(
            self,
            data: pl.Tensor[[16, 512], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            acc_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.create(
                [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            init_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.muls(acc_tile, 0.0)
            for i, (acc_iter,) in pl.range(2, init_values=(init_tile,)):
                offset: pl.Scalar[pl.INDEX] = i * 256
                chunk: pl.Tile[[16, 256], pl.FP32] = pl.load(data, [0, offset], [16, 256])
                tmp: pl.Tile[[16, 1], pl.FP32] = pl.tile.create(
                    [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                partial: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_sum(chunk, tmp)
                updated: pl.Tile[[16, 1], pl.FP32] = pl.tile.add(acc_iter, partial)
                result = pl.yield_(updated)
            final: pl.Tensor[[16, 1], pl.FP32] = pl.store(result, [0, 0], out)
            return final

    mlir_code = _generate_default_mlir(TileIterArgProgram)
    lines = _get_mlir_lines(mlir_code)

    # All alloc_tile must be loc=vec (no spurious loc=gm allocation)
    alloc_lines = _get_alloc_tile_lines(mlir_code)
    assert len(alloc_lines) > 0, "Expected at least one pto.alloc_tile"
    for alloc_line in alloc_lines:
        assert "loc=vec" in alloc_line, f"Expected loc=vec in alloc_tile, got: {alloc_line}"
        assert "loc=gm" not in alloc_line, f"Unexpected loc=gm in alloc_tile: {alloc_line}"

    # scf.for should NOT have iter_args (all iter_args are tile type)
    for_line = _single_line(lines, "scf.for")
    assert "iter_args(" not in for_line, f"scf.for should not have iter_args for tile types: {for_line}"

    # No scf.yield should be present (tile yields are excluded)
    yield_lines = _find_lines(lines, "scf.yield")
    assert len(yield_lines) == 0, f"No scf.yield expected for tile-only iter_args: {yield_lines}"

    # pto.tadd (the accumulation op) must have loc=vec for all tile_buf operands
    tadd_line = _single_line(lines, "pto.tadd")
    assert "loc=gm" not in tadd_line, f"pto.tadd should not have loc=gm operands: {tadd_line}"
    assert tadd_line.count("loc=vec") >= 2, (
        f"pto.tadd should have at least 2 loc=vec annotations: {tadd_line}"
    )


def test_pto_codegen_repairs_row_sum_add_layout_mismatch():
    """`row_sum -> add` should lower through row-major reshape repair."""

    @pl.program
    class LayoutRepairProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def repro(
            self,
            data: pl.Tensor[[16, 256], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            acc_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.create(
                [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            init_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.muls(acc_tile, 0.0)
            chunk: pl.Tile[[16, 256], pl.FP32] = pl.load(data, [0, 0], [16, 256])
            tmp: pl.Tile[[16, 256], pl.FP32] = pl.tile.create(
                [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            partial: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_sum(chunk, tmp)
            updated: pl.Tile[[16, 1], pl.FP32] = pl.tile.add(init_tile, partial)
            final: pl.Tensor[[16, 1], pl.FP32] = pl.store(updated, [0, 0], out)
            return final

    mlir_code = _generate_default_mlir(LayoutRepairProgram)
    lines = _get_mlir_lines(mlir_code)

    # With per-var alloc model, tile.reshape becomes a no-op: each variable
    # gets its own alloc_tile with the correct shape/layout and shared addr.
    # The reshape operations are expressed at the declaration level, not as
    # runtime pto.treshape instructions.
    alloc_lines = _get_alloc_tile_lines(mlir_code)
    row_vec_allocs = [line for line in alloc_lines if "rows=1, cols=16" in line]
    col_vec_allocs = [line for line in alloc_lines if "rows=16, cols=1" in line]
    assert len(row_vec_allocs) >= 1, (
        f"Expected at least one row-vector alloc_tile (rows=1, cols=16), got: {alloc_lines}"
    )
    assert len(col_vec_allocs) >= 1, (
        f"Expected at least one col-vector alloc_tile (rows=16, cols=1), got: {alloc_lines}"
    )

    tadd_line = _single_line(lines, "pto.tadd")
    assert tadd_line.count("blayout=row_major") >= 3, (
        f"Expected row-major operands/results after repair, got: {tadd_line}"
    )
    assert "rows=1, cols=16" in tadd_line, f"Expected repaired row-vector add, got: {tadd_line}"


def test_pto_codegen_keeps_loop_carried_tile_distinct_from_reshape_result():
    """Loop-carried tile and reshape result must not collapse to one SSA mapping."""

    @pl.program
    class LoopReshapeRepairProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def repro(
            self,
            data: pl.Tensor[[16, 512], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            acc_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.create(
                [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            acc_row_major_seed: pl.Tile[[1, 16], pl.FP32] = pl.tile.reshape(acc_tile, [1, 16])
            zero_row_major: pl.Tile[[1, 16], pl.FP32] = pl.tile.muls(acc_row_major_seed, 0.0)
            init_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.reshape(zero_row_major, [16, 1])
            for i, (acc_iter,) in pl.range(2, init_values=(init_tile,)):
                offset: pl.Scalar[pl.INDEX] = i * 256
                chunk: pl.Tile[[16, 256], pl.FP32] = pl.load(data, [0, offset], [16, 256])
                tmp: pl.Tile[[16, 256], pl.FP32] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                partial: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_sum(chunk, tmp)
                acc_row_major: pl.Tile[[1, 16], pl.FP32] = pl.tile.reshape(acc_iter, [1, 16])
                partial_row_major: pl.Tile[[1, 16], pl.FP32] = pl.tile.reshape(partial, [1, 16])
                updated_row_major: pl.Tile[[1, 16], pl.FP32] = pl.tile.add(acc_row_major, partial_row_major)
                updated: pl.Tile[[16, 1], pl.FP32] = pl.tile.reshape(updated_row_major, [16, 1])
                result = pl.yield_(updated)
            final: pl.Tensor[[16, 1], pl.FP32] = pl.store(result, [0, 0], out)
            return final

    mlir_code = _generate_default_mlir(LoopReshapeRepairProgram)
    lines = _get_mlir_lines(mlir_code)

    # With per-var alloc model, tile.reshape becomes a no-op: each variable
    # (including reshape results) gets its own alloc_tile at the shared addr.
    # Verify the structural properties instead of pto.treshape presence.
    alloc_lines = _get_alloc_tile_lines(mlir_code)

    # Both row-vector (1x16) and col-vector (16x1) allocs should exist
    row_vec_allocs = [line for line in alloc_lines if "rows=1, cols=16" in line]
    col_vec_allocs = [line for line in alloc_lines if "rows=16, cols=1" in line]
    assert len(row_vec_allocs) >= 1, (
        f"Expected at least one row-vector alloc (rows=1, cols=16), got: {alloc_lines}"
    )
    assert len(col_vec_allocs) >= 1, (
        f"Expected at least one col-vector alloc (rows=16, cols=1), got: {alloc_lines}"
    )

    tadd_line = _single_line(lines, "pto.tadd ", startswith=True)

    # The tadd should operate on row-major operands (from per-var alloc declarations)
    assert "blayout=row_major" in tadd_line, (
        f"Expected row-major operands in tadd after reshape-via-alloc, got: {tadd_line}"
    )
    assert "rows=1, cols=16" in tadd_line, f"Expected row-vector operands in tadd, got: {tadd_line}"


def test_pto_codegen_mixed_scalar_and_tile_iter_args():
    """Test that mixed iter_args (tile + scalar) emit only scalar iter_args in PTO.

    In PTO, only scalar types need iter_args/yield for loop-carried value
    semantics. When a for loop has both tile and scalar iter_args, the generated
    scf.for should contain iter_args/yield only for the scalar entries, while
    tile iter_args are mapped directly to their init values.
    """

    @pl.program
    class MixedIterArgProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def mixed(
            self,
            data: pl.Tensor[[16, 512], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            acc_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.create(
                [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            init_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.muls(acc_tile, 0.0)
            init_offset: pl.Scalar[pl.INDEX] = 0
            for i, (acc_iter, offset) in pl.range(2, init_values=(init_tile, init_offset)):
                chunk: pl.Tile[[16, 256], pl.FP32] = pl.load(data, [0, offset], [16, 256])
                tmp: pl.Tile[[16, 1], pl.FP32] = pl.tile.create(
                    [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                partial: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_sum(chunk, tmp)
                updated: pl.Tile[[16, 1], pl.FP32] = pl.tile.add(acc_iter, partial)
                new_offset: pl.Scalar[pl.INDEX] = offset + 256
                result_tile, result_offset = pl.yield_(updated, new_offset)
            final: pl.Tensor[[16, 1], pl.FP32] = pl.store(result_tile, [0, 0], out)
            return final

    lines = _get_mlir_lines(_generate_default_mlir(MixedIterArgProgram))

    # scf.for should have iter_args for the scalar type only
    for_line = _single_line(lines, "scf.for")
    assert "iter_args(" in for_line, f"Expected scalar iter_args: {for_line}"

    # iter_args type should be index (scalar), not tile_buf
    assert "tile_buf" not in for_line, f"tile_buf should not appear in iter_args: {for_line}"
    assert "index" in for_line, f"Expected index type in iter_args: {for_line}"

    # scf.yield should have index type only, not tile_buf
    yield_line = _single_line(lines, "scf.yield")
    assert "tile_buf" not in yield_line, f"tile_buf should not appear in scf.yield: {yield_line}"
    assert "index" in yield_line, f"Expected index type in scf.yield: {yield_line}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

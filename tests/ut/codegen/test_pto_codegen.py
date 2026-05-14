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

import re

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir
from pypto.backend import BackendType
from pypto.backend.pto_backend import (
    _emit_group_output,
    _format_error_report,
    _generate_arg_unpacking,
    _generate_kernel_wrapper,
    _get_error_summary,
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
    backend.set_backend_type(BackendType.Ascend910B)
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
    assert "shape = [%c64_index, %c64_index]" in mlir_code or "shape = [%c32_index, %c32_index]" in mlir_code
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


def test_pto_codegen_mat_tile_uses_col_major_blayout():
    """Mat tiles must lower with blayout=col_major even after tile_view canonicalization.

    The ``Mat`` memory space's implicit view is ``blayout=col_major, slayout=row_major``.
    When the constructor canonicalizes a Mat tile's tile_view to ``None``, codegen must
    still surface ``col_major`` via the effective view; otherwise ``pto.tmatmul`` rejects
    the operand on Ascend backends ("expects dst to use the col_major blayout").
    """

    @pl.program
    class MatTileProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def f(self, qi: pl.Tensor[[16, 128], pl.BF16]):
            _q_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(
                qi, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
            )

    alloc_lines = _get_alloc_tile_lines(_generate_default_mlir(MatTileProgram))
    mat_allocs = [line for line in alloc_lines if "loc=mat" in line]
    assert len(mat_allocs) >= 1, f"Expected at least one mat alloc_tile, got: {alloc_lines}"
    for line in mat_allocs:
        assert "blayout=col_major" in line, f"Expected col_major blayout for Mat tile, got: {line}"
        assert "slayout=row_major" in line, f"Expected row_major slayout for Mat tile, got: {line}"


def test_pto_codegen_fillpad_shared_memref_uses_single_alloc_tile():
    """Test that shared MemRef tiles emit one alloc_tile and preserve merged TileView info."""
    span = ir.Span.unknown()
    zero = ir.ConstInt(0, DataType.INDEX, span)
    size = ir.ConstInt(128, DataType.INDEX, span)
    # MemRef byte_offset must be INT64 (PTOAS addr operand requires i64) — see
    # MemRef(VarPtr, int64_t, ...) ctor and AllocateMemoryAddrPass.
    byte_offset_zero = ir.ConstInt(0, DataType.INT64, span)

    input_tensor = ir.Var("a", ir.TensorType([128, 128], DataType.FP32), span)
    output_tensor = ir.Var("output", ir.TensorType([128, 128], DataType.FP32), span)
    m_var = ir.Var("m", ir.ScalarType(DataType.INDEX), span)
    n_var = ir.Var("n", ir.ScalarType(DataType.INDEX), span)
    shared_memref = ir.MemRef(ir.MemorySpace.Vec, byte_offset_zero, 128 * 128 * 4, 0)

    load_view = ir.TileView(valid_shape=[m_var, n_var])
    load_tile_type = ir.TileType([128, 128], DataType.FP32, shared_memref, load_view, ir.MemorySpace.Vec)
    load_tile = ir.Var("tile_a", load_tile_type, span)

    padded_view = ir.TileView(valid_shape=[size, size], pad=ir.PadValue.max)
    padded_tile_type = ir.TileType([128, 128], DataType.FP32, shared_memref, padded_view, ir.MemorySpace.Vec)
    padded_tile = ir.Var("padded", padded_tile_type, span)

    result_var = ir.Var("result", ir.TensorType([128, 128], DataType.FP32), span)
    offsets = ir.MakeTuple([zero, zero], span)
    shapes = ir.MakeTuple([size, size], span)
    valid_shapes = ir.MakeTuple([m_var, n_var], span)

    load_call = ir.Call(
        ir.Op("tile.load"), [input_tensor, offsets, shapes, valid_shapes], {}, load_tile_type, span
    )
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
    # Both share the same addr (same MemRef). The PTOAS dialect requires `i64`
    # for the alloc_tile addr operand, so AllocateMemoryAddrPass declares the
    # final byte_offset_ ConstInt as INT64 and codegen emits it 1:1.
    assert "addr = %c0_i64" in alloc_lines[0]
    assert "addr = %c0_i64" in alloc_lines[1]
    # All alloc_tile types use dynamic v_row=?, v_col=?; the actual extents
    # are conveyed via the valid_row / valid_col operands.
    assert "v_row=?" in alloc_lines[0], f"Expected dynamic v_row=? in alloc: {alloc_lines[0]}"
    assert "v_col=?" in alloc_lines[0], f"Expected dynamic v_col=? in alloc: {alloc_lines[0]}"
    # Padded tile carries dynamic v_row=?/v_col=? type, retains pad=2,
    # and sources its valid_row/valid_col operands from the physical dims.
    assert "pad=2>" in alloc_lines[1], f"Expected fillpad pad metadata to be preserved: {alloc_lines[1]}"
    assert "v_row=?" in alloc_lines[1], f"Expected dynamic v_row=? in padded tile: {alloc_lines[1]}"
    assert "v_col=?" in alloc_lines[1], f"Expected dynamic v_col=? in padded tile: {alloc_lines[1]}"
    assert "valid_row = %c128_index" in alloc_lines[1], (
        f"Expected valid_row = %c128_index operand in padded tile: {alloc_lines[1]}"
    )
    assert "valid_col = %c128_index" in alloc_lines[1], (
        f"Expected valid_col = %c128_index operand in padded tile: {alloc_lines[1]}"
    )


def test_pto_codegen_fillpad_inplace():
    """Test that tile.fillpad_inplace emits pto.tfillpad and shares MemRef with input."""
    span = ir.Span.unknown()
    zero = ir.ConstInt(0, DataType.INDEX, span)
    size = ir.ConstInt(128, DataType.INDEX, span)
    # MemRef byte_offset must be INT64 (PTOAS addr operand requires i64) — see
    # MemRef(VarPtr, int64_t, ...) ctor and AllocateMemoryAddrPass.
    byte_offset_zero = ir.ConstInt(0, DataType.INT64, span)

    input_tensor = ir.Var("a", ir.TensorType([128, 128], DataType.FP32), span)
    output_tensor = ir.Var("output", ir.TensorType([128, 128], DataType.FP32), span)
    m_var = ir.Var("m", ir.ScalarType(DataType.INDEX), span)
    n_var = ir.Var("n", ir.ScalarType(DataType.INDEX), span)
    shared_memref = ir.MemRef(ir.MemorySpace.Vec, byte_offset_zero, 128 * 128 * 4, 0)

    load_view = ir.TileView(valid_shape=[m_var, n_var])
    load_tile_type = ir.TileType([128, 128], DataType.FP32, shared_memref, load_view, ir.MemorySpace.Vec)
    load_tile = ir.Var("tile_a", load_tile_type, span)

    padded_view = ir.TileView(valid_shape=[size, size], pad=ir.PadValue.zero)
    padded_tile_type = ir.TileType([128, 128], DataType.FP32, shared_memref, padded_view, ir.MemorySpace.Vec)
    padded_tile = ir.Var("padded", padded_tile_type, span)

    result_var = ir.Var("result", ir.TensorType([128, 128], DataType.FP32), span)
    offsets = ir.MakeTuple([zero, zero], span)
    shapes = ir.MakeTuple([size, size], span)

    # Intentionally use the 3-arg form to exercise the backend fallback when
    # valid_shapes is omitted (equivalent to `pl.load(..., valid_shapes=None)`).
    load_call = ir.Call(ir.Op("tile.load"), [input_tensor, offsets, shapes], {}, load_tile_type, span)
    fillpad_inplace_call = ir.Call(
        ir.Op("tile.fillpad_inplace"),
        [load_tile],
        {"pad_value": ir.PadValue.zero},
        padded_tile_type,
        span,
    )
    store_call = ir.Call(ir.Op("tile.store"), [padded_tile, offsets, output_tensor], result_var.type, span)

    body = ir.SeqStmts(
        [
            ir.SeqStmts(
                [
                    ir.AssignStmt(load_tile, load_call, span),
                    ir.AssignStmt(padded_tile, fillpad_inplace_call, span),
                    ir.AssignStmt(result_var, store_call, span),
                ],
                span,
            ),
            ir.ReturnStmt([result_var], span),
        ],
        span,
    )
    func = ir.Function(
        "fillpad_inplace_test",
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
    program = ir.Program([func], "fillpad_inplace_test_program", span)

    mlir_code = _generate_mlir(program)
    alloc_lines = _get_alloc_tile_lines(mlir_code)

    # Both allocs share the same addr (same MemRef). Address dtype tracks
    # MemRef::byte_offset_ — declared INT64 by AllocateMemoryAddrPass so the
    # codegen emits `i64` as required by the PTOAS dialect's addr operand.
    assert len(alloc_lines) == 2, f"Expected two alloc_tiles for per-var alloc model, got: {alloc_lines}"
    assert "addr = %c0_i64" in alloc_lines[0]
    assert "addr = %c0_i64" in alloc_lines[1]
    # Dynamic valid_shape tile: type has v_row=?, v_col=? (both dynamic per PTOAS requirement)
    assert "v_row=?" in alloc_lines[0], f"Expected dynamic v_row=? in alloc: {alloc_lines[0]}"
    assert "v_col=?" in alloc_lines[0], f"Expected dynamic v_col=? in alloc: {alloc_lines[0]}"
    # fillpad_inplace emits pto.tfillpad; inplace semantics come from shared UB addr above.
    assert "pto.tfillpad " in mlir_code, "Expected pto.tfillpad in MLIR output"


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
    # Unified always-dynamic alloc_tile: type carries v_row=?/v_col=? with
    # explicit valid_row/valid_col operands. valid_row comes from a constant
    # (1) and valid_col from the user-defined dynamic scalar.
    assert "v_row=?" in alloc_line, f"Expected dynamic v_row=? in tile_buf type, got: {alloc_line}"
    assert "v_col=?" in alloc_line, f"Expected dynamic v_col=? in tile_buf type, got: {alloc_line}"
    assert "valid_row = %c1_index" in alloc_line, (
        f"Expected valid_row = %c1_index operand in alloc: {alloc_line}"
    )
    assert "valid_col" in alloc_line, f"Expected valid_col operand in alloc: {alloc_line}"
    assert "%c-1" not in mlir_code
    # alloc_tile already carries the runtime valid_col, so no separate
    # pto.set_validshape is emitted.
    assert "pto.set_validshape" not in mlir_code


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
    # Unified always-dynamic alloc_tile: type carries v_row=?/v_col=? with
    # explicit valid_row/valid_col operands. valid_row comes from the user-
    # defined dynamic scalar; valid_col is a constant (16).
    assert "v_row=?" in alloc_line, f"Expected dynamic v_row=? in tile_buf type, got: {alloc_line}"
    assert "v_col=?" in alloc_line, f"Expected dynamic v_col=? in tile_buf type, got: {alloc_line}"
    assert "valid_row" in alloc_line, f"Expected valid_row operand in alloc: {alloc_line}"
    assert "valid_col = %c16_index" in alloc_line, (
        f"Expected valid_col = %c16_index operand in alloc: {alloc_line}"
    )
    # alloc_tile already carries the runtime valid_row, so no separate
    # pto.set_validshape is emitted.
    assert "pto.set_validshape" not in mlir_code


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
    assert "offsets = [%c0_index, %c0_index]" in mlir_code
    assert "sizes = [%c32_index, %c32_index]" in mlir_code
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
    assert "%c0_index" in mlir_code or "%c32_index" in mlir_code


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
        assert "reinterpret_cast<__gm__ Tensor*>(args[0])" in code
        assert "reinterpret_cast<__gm__ Tensor*>(args[1])" in code
        assert "reinterpret_cast<__gm__ Tensor*>(args[2])" in code
        assert names == ["a", "b", "out"]

    def test_mixed_tensor_scalar(self):
        func = _make_func("test_fn", [("input", "tensor"), ("scale", "scalar"), ("output", "tensor")])
        code, names = _generate_arg_unpacking(func)
        # Tensors-first: input=args[0], output=args[1], scale=args[2]
        assert "reinterpret_cast<__gm__ Tensor*>(args[0])" in code
        assert "reinterpret_cast<__gm__ Tensor*>(args[1])" in code
        assert "scale_conv.u64 = args[2];" in code
        assert "float scale = scale_conv.val;" in code
        assert names == ["input", "output", "scale"]

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
        # Tensors-first: a=arg0, out=arg1, s=arg2
        assert "my_kernel(a, out, s);" in wrapper

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

    def test_split_aiv_wrapper_uses_runtime_subblock_bridge_on_a2a3(self):
        @pl.program
        class SplitWrapperProgram:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def split_vec(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pipe_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=pipe_buf)
                z_vec: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=1
                )
                scaled: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.muls(z_vec, 2.0)
                pl.tfree_to_aic(z_vec)
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(scaled, [0, 0], out)
                return updated

        transformed = _run_default_passes(SplitWrapperProgram)
        func = transformed.get_function("split_vec")
        assert func is not None
        assert transformed.get_function("split_vec__aiv1") is None

        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        assert "PYPTO_FIXED_SUBBLOCK_ID" not in wrapper
        assert wrapper.count("#if !defined(__CPU_SIM)") == 2
        assert '#if !defined(__CPU_SIM)\n#include "intrinsic.h"' in wrapper
        assert "[[block_local]] static int32_t pypto_runtime_subblock_id;" in wrapper
        assert '#include "intrinsic.h"' in wrapper
        assert "#define get_subblockid() pypto_runtime_subblock_id" in wrapper
        assert (
            "#if !defined(__CPU_SIM)\n"
            "    // Read A2A3 mixed-task subblock id from runtime dispatch context\n"
            "    pypto_runtime_subblock_id = get_sub_block_id(args);\n"
            "#endif"
        ) in wrapper

    def test_no_split_dual_dispatch_wrapper_uses_runtime_subblock_bridge_on_a2a3(self):
        @pl.program
        class NoSplitDualDispatchProgram:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def nosplit_vec(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pl.tile.get_subblock_idx()
                zero_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.full([16, 16], dtype=pl.FP32, value=0.0)
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(zero_tile, [0, 0], out)
                return updated

        func = NoSplitDualDispatchProgram.get_function("nosplit_vec")
        assert func is not None
        assert func.attrs.get("dual_aiv_dispatch") is True

        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        assert "PYPTO_FIXED_SUBBLOCK_ID" not in wrapper
        assert "[[block_local]] static int32_t pypto_runtime_subblock_id;" in wrapper
        assert "#define get_subblockid() pypto_runtime_subblock_id" in wrapper
        assert "pypto_runtime_subblock_id = get_sub_block_id(args);" in wrapper

    def test_split_aiv_wrapper_uses_runtime_subblock_bridge_in_group_output_on_a2a3(
        self, tmp_path, monkeypatch
    ):
        @pl.program
        class MixedGroupWrapperProgram:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def split_cube(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="split_vec")
                pl.aic_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=c2v_peer)
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=1)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def split_vec(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pipe_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=pipe_buf)
                z_vec: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=1
                )
                scaled: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.muls(z_vec, 2.0)
                pl.tfree_to_aic(z_vec)
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(scaled, [0, 0], out)
                return updated

        transformed = _run_default_passes(MixedGroupWrapperProgram)
        split_cube = transformed.get_function("split_cube")
        split_vec = transformed.get_function("split_vec")
        assert split_cube is not None
        assert split_vec is not None

        monkeypatch.setattr(
            "pypto.backend.pto_backend._compile_pto_module",
            lambda _pto_code, _module_name, _output_dir: SAMPLE_PTOAS_OUTPUT,
        )

        result_files = {}
        _emit_group_output(
            result_files,
            "mixed_group",
            [split_cube, split_vec],
            "unused grouped pto",
            str(tmp_path),
            skip_ptoas=False,
        )

        split_cube_wrapper = next(
            content for path, content in result_files.items() if path.endswith("split_cube.cpp")
        )
        split_vec_wrapper = next(
            content for path, content in result_files.items() if path.endswith("split_vec.cpp")
        )
        assert "static __aicore__ void test_func" in split_cube_wrapper
        assert "static __aicore__ void test_func" in split_vec_wrapper
        assert '#if !defined(__CPU_SIM)\n#include "intrinsic.h"' in split_vec_wrapper
        assert "[[block_local]] static int32_t pypto_runtime_subblock_id;" in split_vec_wrapper
        assert "#define get_subblockid() pypto_runtime_subblock_id" in split_vec_wrapper
        assert "pypto_runtime_subblock_id = get_sub_block_id(args);" in split_vec_wrapper

    def test_spmd_wrapper_drops_macro_bridge_and_appends_block_args(self):
        @pl.program
        class SpmdWrapperProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def spmd_kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                block_idx = pl.tile.get_block_idx()
                _block_num = pl.tile.get_block_num()
                offset = block_idx * 128
                tile_a = pl.load(a, [offset, 0], [128, 128])
                out = pl.store(tile_a, [offset, 0], out)
                return out

        transformed = _run_default_passes(SpmdWrapperProgram)
        func = transformed.get_function("spmd_kernel")
        assert func is not None

        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        # Macro shadow / dual storage class / push_pop dance are all gone.
        assert "[[block_local]] static int32_t __pypto_spmd_block_idx" not in wrapper
        assert "static thread_local int32_t __pypto_spmd_block_idx" not in wrapper
        assert "#define get_block_idx()" not in wrapper
        assert "#define get_block_num()" not in wrapper
        assert 'push_macro("get_block_idx")' not in wrapper
        assert 'pop_macro("get_block_idx")' not in wrapper
        assert 'push_macro("get_block_num")' not in wrapper
        assert 'pop_macro("get_block_num")' not in wrapper
        # Plain locals, computed once from intrinsic.h overloads.
        assert "int32_t __pypto_spmd_block_idx = get_block_idx(args);" in wrapper
        assert "int32_t __pypto_spmd_block_num = get_block_num(args);" in wrapper
        # Inner function call APPENDS the two block args after user args.
        # SSA conversion may rename params (e.g. `a` → `a__ssa_v0`), so match
        # the call-site suffix rather than the full arg list.
        assert "__pypto_spmd_block_idx, __pypto_spmd_block_num);" in wrapper
        # And the call entry uses the inner function name.
        assert "spmd_kernel(" in wrapper

    def test_non_spmd_func_in_spmd_group_keeps_locals_but_not_call_suffix(self):
        # Sibling-only SPMD usage: func itself does not call get_block_idx /
        # get_block_num, so PTOCodegen does not append SPMD params to its
        # signature — and the wrapper must NOT append them to the inner call
        # either. But __gm_pipe_buffer sharding in _generate_arg_unpacking
        # still consumes the two locals, so they remain in the wrapper.
        func = _make_func("plain_kernel", [("a", "tensor"), ("out", "tensor")])
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT, group_uses_spmd=True)
        assert "int32_t __pypto_spmd_block_idx = get_block_idx(args);" in wrapper
        assert "int32_t __pypto_spmd_block_num = get_block_num(args);" in wrapper
        # Call site does not append block args.
        assert "__pypto_spmd_block_idx" not in wrapper.split("plain_kernel(", 1)[1].split(");")[0]


def test_pto_codegen_spmd_block_params_appended_with_named_ssas():
    """tile.get_block_idx/num lower to arith.index_cast of two named i32 params
    appended at the end of the func.func signature."""

    @pl.program
    class SpmdCodegenProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def spmd_func(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            block_idx = pl.tile.get_block_idx()
            _block_num = pl.tile.get_block_num()
            offset = block_idx * 128
            tile_a = pl.load(a, [offset, 0], [128, 128])
            out = pl.store(tile_a, [offset, 0], out)
            return out

    mlir = _generate_default_mlir(SpmdCodegenProgram)
    # User-defined tensor params stay at the front under the existing
    # tensors-first convention (indices unchanged).
    assert "%arg0: !pto.ptr<f32>" in mlir
    assert "%arg1: !pto.ptr<f32>" in mlir
    # SPMD block identity params are appended at the end with named SSAs.
    assert "%__pypto_spmd_block_idx: i32" in mlir
    assert "%__pypto_spmd_block_num: i32" in mlir
    # tile.get_block_idx / num lower to arith.index_cast of those named params.
    assert "arith.index_cast %__pypto_spmd_block_idx : i32 to index" in mlir
    assert "arith.index_cast %__pypto_spmd_block_num : i32 to index" in mlir
    # The legacy pto.get_block_idx / pto.get_block_num pseudo-ops are gone.
    assert "pto.get_block_idx" not in mlir
    assert "pto.get_block_num" not in mlir
    # SPMD params come AFTER the user-defined params in textual signature order.
    signature_line = next(line for line in mlir.splitlines() if "func.func @spmd_func(" in line)
    arg0_pos = signature_line.find("%arg0")
    spmd_idx_pos = signature_line.find("%__pypto_spmd_block_idx")
    assert arg0_pos != -1 and spmd_idx_pos != -1
    assert spmd_idx_pos > arg0_pos, f"SPMD param must come after user params: {signature_line}"


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
            backend_type=BackendType.Ascend910B,
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

    def test_summary_prefers_real_ptoas_error_line(self, tmp_path):
        summary = _get_error_summary(
            RuntimeError(
                """ptoas compilation failed: module attributes {pto.target_arch = "a5"} {
  func.func @main() {
    return
  }
}
"""
                + 'loc("build_output/qwen3_decode_layer_incore_2.pto":23:50): error: '
                + "'pto.reserve_buffer' op expects 'base' to be resolved "
                + "before address materialization\n"
                + "Error: Pass execution failed."
            ),
            "qwen3_decode_layer_incore_2",
        )

        assert "module attributes" not in summary
        assert "qwen3_decode_layer_incore_2.pto" in summary
        assert "'pto.reserve_buffer' op expects 'base'" in summary

        report = _format_error_report(
            [("qwen3_decode_layer_incore_2", RuntimeError(summary))],
            str(tmp_path),
        )
        assert "module attributes" not in report


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


def test_pto_codegen_if_stmt_only_returns_scalars_for_tile_phi():
    """IfStmt should materialize tile phi values via branch-local copies, not scf.if results."""

    @pl.program
    class IfTilePhiProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def repro(
            self,
            flag: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            seed: pl.Tile[[16, 1], pl.FP32] = pl.tile.create(
                [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            partial: pl.Tile[[16, 1], pl.FP32] = pl.tile.muls(seed, 0.0)
            updated: pl.Tile[[16, 1], pl.FP32] = pl.tile.muls(seed, 2.0)
            if flag == 0:
                result = partial
            else:
                result = updated
            final: pl.Tensor[[16, 1], pl.FP32] = pl.store(result, [0, 0], out)
            return final

    mlir_code = _generate_default_mlir(IfTilePhiProgram)
    lines = _get_mlir_lines(mlir_code)

    if_line = _single_line(lines, "scf.if")
    assert "tile_buf" not in if_line, f"IfStmt should not return tile_buf values: {if_line}"
    assert "-> (" not in if_line, f"IfStmt should not expose non-scalar results: {if_line}"

    tmov_lines = _find_lines(lines, "pto.tmov")
    assert any("rows=16, cols=1" in line for line in tmov_lines), (
        f"Expected branch-local tile materialization for the tile phi, got: {tmov_lines}"
    )

    phi_tmov_line = next(
        (line for line in tmov_lines if "rows=16, cols=1" in line),
        None,
    )
    assert phi_tmov_line is not None, f"Expected a tile-phi tmov, got: {tmov_lines}"

    match = re.search(r"outs\((%[\w\d_]+) :", phi_tmov_line)
    assert match is not None, f"Expected tmov outs target in line: {phi_tmov_line}"
    phi_target = match.group(1)
    phi_alloc_line = _single_line(lines, f"{phi_target} = pto.alloc_tile", startswith=True)
    assert "addr =" in phi_alloc_line, f"Expected IfStmt tile phi alloc to carry addr: {phi_alloc_line}"


def test_pto_codegen_if_stmt_tile_phi_preserves_dynamic_valid_shape():
    """IfStmt tile phi alloc should preserve dynamic valid_shape operands."""

    @pl.program
    class IfDynamicTilePhiProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def repro(
            self,
            flag: pl.Scalar[pl.INDEX],
            input: pl.Tensor[[1, 120], pl.FP32],
            ctx_len: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[1, 120], pl.FP32]],
        ) -> pl.Tensor[[1, 120], pl.FP32]:
            valid_len: pl.Scalar[pl.INDEX] = ctx_len + 0
            seed: pl.Tile[[1, 120], pl.FP32] = pl.tile.load(
                input,
                [0, 0],
                [1, 120],
                [1, valid_len],
                target_memory=pl.MemorySpace.Vec,
                transpose=False,
            )
            updated: pl.Tile[[1, 120], pl.FP32] = pl.tile.muls(seed, 1.0)
            if flag == 0:
                result = seed
            else:
                result = updated
            final: pl.Tensor[[1, 120], pl.FP32] = pl.tile.store(result, [0, 0], out)
            return final

    mlir_code = _generate_default_mlir(IfDynamicTilePhiProgram)
    lines = _get_mlir_lines(mlir_code)
    phi_tmov_line = next(line for line in _find_lines(lines, "pto.tmov") if "rows=1, cols=120" in line)
    match = re.search(r"outs\((%[\w\d_]+) :", phi_tmov_line)
    assert match is not None, f"Expected tmov outs target in line: {phi_tmov_line}"
    phi_target = match.group(1)
    phi_alloc_line = _single_line(lines, f"{phi_target} = pto.alloc_tile", startswith=True)
    assert "valid_col = %" in phi_alloc_line, (
        f"Expected IfStmt tile phi alloc to carry dynamic valid_col, got: {phi_alloc_line}"
    )
    assert "v_col=?" in phi_alloc_line, f"Expected dynamic v_col in tile phi alloc, got: {phi_alloc_line}"


def test_pto_codegen_if_stmt_tile_no_redundant_tmov():
    """IfStmt emit_branch must not generate a codegen-level tmov for tile return_vars.

    MemoryReuse's YieldFixupMutator inserts an IR-level tile.move in the else-branch
    to unify its yield MemRef with the then-branch's canonical MemRef. The codegen
    should emit exactly one pto.tmov (from that IR tile.move), not a second redundant
    one from emit_branch (which would copy addr B → addr B, a no-op).
    """

    @pl.program
    class IfTileNoRedundantTmovProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def repro(
            self,
            flag: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            seed: pl.Tile[[64, 64], pl.FP32] = pl.tile.create(
                [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            then_tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.muls(seed, 2.0)
            else_tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.muls(seed, 3.0)
            if flag == 0:
                result = then_tile
            else:
                result = else_tile
            final: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], out)
            return final

    mlir_code = _generate_default_mlir(IfTileNoRedundantTmovProgram)
    lines = _get_mlir_lines(mlir_code)

    tmov_lines = _find_lines(lines, "pto.tmov")
    # After the fix: exactly one tmov — the IR-level tile.move (else branch only).
    # The then-branch has no tmov; the codegen-level tmov from emit_branch is removed.
    assert len(tmov_lines) == 1, (
        f"Expected exactly 1 pto.tmov (from IR tile.move), got {len(tmov_lines)}: {tmov_lines}"
    )

    # The single tmov copies from the else computation result to the canonical tile_buf.
    tmov = tmov_lines[0]
    assert "pto.tmov" in tmov, f"Expected a pto.tmov line, got: {tmov}"
    # Verify the tmov targets are at different addresses (it's a real copy, not a no-op).
    ins_match = re.search(r"ins\((%[\w\d_]+)", tmov)
    outs_match = re.search(r"outs\((%[\w\d_]+)", tmov)
    assert ins_match and outs_match, f"Expected ins/outs operands in tmov: {tmov}"
    assert ins_match.group(1) != outs_match.group(1), (
        f"tmov src and dst must differ (not a self-copy): {tmov}"
    )


def test_pto_codegen_if_stmt_scalar_result_preserves_integer_dtype():
    """IfStmt scalar results should use their real scalar dtype in scf.if results."""

    @pl.program
    class IfScalarIntProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def repro(
            self,
            flag: pl.Scalar[pl.INDEX],
            value: pl.Scalar[pl.INT32],
            delta_one: pl.Scalar[pl.INT32],
            delta_two: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1], pl.INT32]],
        ) -> pl.Tensor[[1], pl.INT32]:
            if flag == 0:
                result = value + delta_one
            else:
                result = value + delta_two
            combined: pl.Scalar[pl.INT32] = result + value
            final: pl.Tensor[[1], pl.INT32] = pl.tensor.write(out, [0], combined)
            return final

    lines = _get_mlir_lines(_generate_default_mlir(IfScalarIntProgram))
    if_line = _single_line(lines, "scf.if")
    assert "-> (i32)" in if_line, f"Expected INT32 if-result type, got: {if_line}"
    assert "-> (index)" not in if_line, f"Did not expect index-typed if-result: {if_line}"


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


def test_pto_codegen_slice_fillpad_partial_dynamic_valid_shape():
    """Slice with partially dynamic valid_shape followed by fillpad must lower
    without a scratch slice alloc, and feed a dynamic valid_shape tile into
    pto.tfillpad."""

    @pl.program
    class SliceFillpadProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            scores_in: pl.Tensor[[16, 64], pl.FP32],
            valid_len: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            scores: pl.Tile[[16, 64], pl.FP32] = pl.load(scores_in, [0, 0], [16, 64])
            sliced: pl.Tile[[16, 64], pl.FP32] = pl.tile.slice(
                scores, [16, 64], [0, 0], valid_shape=[16, valid_len]
            )
            padded: pl.Tile[[16, 64], pl.FP32] = pl.fillpad(sliced, pad_value=pl.PadValue.min)
            return pl.store(padded, [0, 0], out)

    mlir_code = _generate_default_mlir(SliceFillpadProgram)

    # tile.slice now lowers to a pto.subview view rather than a textract
    # data-movement op; no slice_buf scratch allocation should appear.
    assert "= pto.alloc_tile" not in "\n".join(
        line for line in mlir_code.splitlines() if "slice_buf" in line
    ), f"Unexpected slice_buf alloc_tile — pto.subview is a view, no extra buffer needed.\n{mlir_code}"
    assert "pto.textract" not in mlir_code, f"tile.slice no longer emits pto.textract, got:\n{mlir_code}"

    # Full-window slice with explicit valid_shape lowers via data-preserving
    # pto.tmov plus a follow-up pto.set_validshape.
    subview_lines = [line.strip() for line in mlir_code.splitlines() if "pto.subview" in line]
    assert len(subview_lines) == 0, f"Full-window slice should not emit pto.subview, got: {subview_lines}"
    tmov_lines = [line.strip() for line in mlir_code.splitlines() if "pto.tmov" in line]
    assert len(tmov_lines) == 1, f"Expected one pto.tmov for full-window slice, got: {tmov_lines}"
    set_validshape_lines = [line.strip() for line in mlir_code.splitlines() if "pto.set_validshape" in line]
    assert len(set_validshape_lines) == 1, f"Expected one pto.set_validshape, got: {set_validshape_lines}"

    # All tile_buf types use the always-dynamic `v_row=?, v_col=?` form.
    fillpad_lines = [line.strip() for line in mlir_code.splitlines() if "pto.tfillpad" in line]
    assert len(fillpad_lines) == 1, f"Expected one tfillpad, got: {fillpad_lines}"
    assert "v_row=?" in fillpad_lines[0], f"fillpad input should have v_row=?: {fillpad_lines[0]}"
    assert "v_col=?" in fillpad_lines[0], f"fillpad input should have v_col=?: {fillpad_lines[0]}"


def test_pto_codegen_slice_full_window_dynamic_valid_shape_uses_set_validshape():
    """Full-window slice with runtime valid rows should lower via tmov +
    set_validshape rather than a subview `valid [...]` clause."""

    @pl.program
    class FullWindowSliceProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            scores_in: pl.Tensor[[16, 256], pl.BF16],
            valid_rows: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[16, 256], pl.BF16]],
        ) -> pl.Tensor[[16, 256], pl.BF16]:
            scores: pl.Tile[[16, 256], pl.BF16] = pl.load(scores_in, [0, 0], [16, 256])
            trimmed: pl.Tile[[16, 256], pl.BF16] = pl.tile.slice(
                scores, [16, 256], [0, 0], valid_shape=[valid_rows, 256]
            )
            return pl.store(trimmed, [0, 0], out)

    mlir_code = _generate_default_mlir(FullWindowSliceProgram)
    subview_lines = [line.strip() for line in mlir_code.splitlines() if "pto.subview" in line]
    assert len(subview_lines) == 0, f"Full-window slice should not emit pto.subview, got: {subview_lines}"
    tmov_lines = [line.strip() for line in mlir_code.splitlines() if "pto.tmov" in line]
    assert len(tmov_lines) == 1, f"Expected one pto.tmov for full-window slice, got: {tmov_lines}"

    set_validshape_lines = [line.strip() for line in mlir_code.splitlines() if "pto.set_validshape" in line]
    assert len(set_validshape_lines) == 1, f"Expected one pto.set_validshape, got: {set_validshape_lines}"
    assert "valid_rows" in set_validshape_lines[0] or "%arg2" in set_validshape_lines[0], (
        f"Runtime valid_rows should feed pto.set_validshape: {set_validshape_lines[0]}"
    )


def test_pto_codegen_slice_static_subwindow_binds_subview_result():
    """Static subwindow slice without explicit valid_shape should remain a
    true pto.subview SSA result and avoid a materializing pto.tmov."""

    @pl.program
    class StaticSubviewProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            scores_in: pl.Tensor[[5, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[5, 64], pl.FP32]],
        ) -> pl.Tensor[[5, 64], pl.FP32]:
            scores: pl.Tile[[5, 128], pl.FP32] = pl.load(scores_in, [0, 0], [5, 128])
            sliced: pl.Tile[[5, 64], pl.FP32] = pl.tile.slice(scores, [5, 64], [0, 0])
            return pl.store(sliced, [0, 0], out)

    mlir_code = _generate_default_mlir(StaticSubviewProgram)
    subview_lines = [line.strip() for line in mlir_code.splitlines() if "pto.subview" in line]
    assert len(subview_lines) == 1, f"Expected one pto.subview, got: {subview_lines}"
    assert "rows=5, cols=64" in subview_lines[0], f"Subview result type should be 5x64: {subview_lines[0]}"
    tmov_lines = [line.strip() for line in mlir_code.splitlines() if "pto.tmov" in line]
    assert len(tmov_lines) == 0, f"Static subwindow view should not materialize a pto.tmov, got: {tmov_lines}"


class TestColumnVectorCodegen:
    """[M, 1] column-vector tensors auto-emit DN layout in make_tensor_view."""

    def test_column_vector_auto_dn_layout(self):
        """[M, 1] tensor without DN annotation emits shape=[M,1] strides=[1,M] layout=dn."""

        @pl.program
        class ColVecProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                col_vec: pl.Tensor[[16, 1], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                v = pl.load(col_vec, [0, 0], [16, 1])
                return pl.store(v, [0, 0], out)

        mlir_code = _generate_default_mlir(ColVecProgram)
        lines = _get_mlir_lines(mlir_code)
        col_vec_view = _single_line(lines, "pto.make_tensor_view %arg0")
        assert "shape = [%c16_index, %c1_index]" in col_vec_view
        assert "strides = [%c1_index, %c16_index]" in col_vec_view
        assert "layout = #pto.layout<dn>" in col_vec_view

    def test_column_vector_with_explicit_dn(self):
        """[M, 1] tensor with explicit DN also emits shape=[M,1] strides=[1,M] layout=dn."""

        @pl.program
        class ColVecDNProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                # Explicit DN view with the canonical-packed strides for shape
                # [16, 1] (stride[-2]=1, stride[-1]=shape[-2]=16). Using the
                # explicit TensorView form (RFC #1300 supplementary 1 escape
                # hatch) instead of the deprecated pl.Tensor[..., pl.DN]
                # shorthand. This test specifically verifies the
                # column-vector DN codegen path, so the DN view is the test
                # subject — not a load-time alias.
                col_vec: pl.Tensor[
                    [16, 1], pl.FP32, pl.TensorView(stride=[1, 16], layout=pl.TensorLayout.DN)
                ],
                out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                v = pl.load(col_vec, [0, 0], [16, 1])
                return pl.store(v, [0, 0], out)

        mlir_code = _generate_default_mlir(ColVecDNProgram)
        lines = _get_mlir_lines(mlir_code)
        col_vec_view = _single_line(lines, "pto.make_tensor_view %arg0")
        assert "shape = [%c16_index, %c1_index]" in col_vec_view
        assert "strides = [%c1_index, %c16_index]" in col_vec_view
        assert "layout = #pto.layout<dn>" in col_vec_view

    def test_regular_2d_nd_unchanged(self):
        """Non-column-vector ND [R,C] tensor keeps ND layout."""

        @pl.program
        class RegularProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                t = pl.load(a, [0, 0], [16, 128])
                return pl.store(t, [0, 0], out)

        mlir_code = _generate_default_mlir(RegularProgram)
        lines = _get_mlir_lines(mlir_code)
        a_view = _single_line(lines, "pto.make_tensor_view %arg0")
        assert "shape = [%c16_index, %c128_index]" in a_view
        assert "strides = [%c128_index, %c1_index]" in a_view
        assert "layout = #pto.layout<nd>" in a_view

    def test_row_vector_stays_nd(self):
        """[1, N] row-vector tensor stays ND (only [M, 1] gets forced DN)."""

        @pl.program
        class RowVecProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                row_vec: pl.Tensor[[1, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                v = pl.load(row_vec, [0, 0], [1, 128])
                return pl.store(v, [0, 0], out)

        mlir_code = _generate_default_mlir(RowVecProgram)
        lines = _get_mlir_lines(mlir_code)
        row_view = _single_line(lines, "pto.make_tensor_view %arg0")
        assert "shape = [%c1_index, %c128_index]" in row_view
        assert "strides = [%c128_index, %c1_index]" in row_view
        assert "layout = #pto.layout<nd>" in row_view


def test_pto_codegen_3d_dn_tensor_view_uses_canonical_stride():
    """3D DN tensor emits the canonical RFC #1300 (shape, stride, layout) triple.

    For ``[B, K, N] DN`` (RFC §2.3 canonical-form interpretation), packed strides are::

        stride[n-2] = 1
        stride[n-1] = shape[n-2]       (= K)
        stride[n-3] = shape[n-2] * shape[n-1]   (= K * N — the per-batch volume)

    After ``MaterializeTensorStrides`` activates in the default pipeline (RFC #1300 P6),
    the codegen reads ``(shape, stride, layout)`` directly from the materialized
    TensorView rather than going through the legacy ``dn_swap`` post-emit path.
    """

    @pl.program
    class DN3DProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            # Explicit DN view with canonical-packed strides per RFC §2.3:
            # shape [B=2, K=48, N=64] → stride[1]=1, stride[2]=K=48,
            # stride[0]=K*N=3072. Uses the explicit TensorView form (RFC
            # #1300 supplementary 1 escape hatch); the test subject is the
            # 3-D DN codegen path, so the DN view is part of the fixture.
            b: pl.Tensor[
                [2, 48, 64],
                pl.FP32,
                pl.TensorView(stride=[3072, 1, 48], layout=pl.TensorLayout.DN),
            ],
            out: pl.Out[pl.Tensor[[2, 48, 64], pl.FP32]],
        ) -> pl.Tensor[[2, 48, 64], pl.FP32]:
            tile_b = pl.load(b, [0, 0, 0], [2, 48, 64])
            return pl.store(tile_b, [0, 0, 0], out)

    mlir_code = _generate_default_mlir(DN3DProgram)
    lines = _get_mlir_lines(mlir_code)
    b_view = _single_line(lines, "pto.make_tensor_view %arg0")

    # Canonical form preserves the user-written logical shape — no swap.
    assert "shape = [%c2_index, %c48_index, %c64_index]" in b_view, (
        f"3D DN canonical shape must match the logical IR shape [2, 48, 64]: {b_view}"
    )
    # Canonical DN strides: stride[n-2]=1, stride[n-1]=shape[n-2]=48.
    assert "strides = [" in b_view and ", %c1_index, %c48_index]" in b_view, (
        f"3D DN canonical stride must end with [1, 48] for shape [2, 48, 64]: {b_view}"
    )
    # The batch stride (= 3072 = 48 * 64) is materialized as a single constant
    # ``%c3072_index`` by ``MaterializeTensorStrides``, so it should NOT show up
    # as an ``arith.muli`` at codegen.
    assert "%c3072_index" in b_view, f"Batch stride must be the constant 3072 (= 48 * 64): {b_view}"
    assert "layout = #pto.layout<dn>" in b_view


def test_pto_codegen_constant_indent_consistency():
    """All arith.constant lines must have consistent 2-space indent.

    Regression test for #812: constants first encountered inside a nested scope
    (for loop) used the loop's deeper indent level instead of the function-body
    indent level.
    """

    @pl.program
    class NestedConstantProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def nested_const(
            self,
            a: pl.Tensor[[128, 128], pl.FP32],
            output: pl.Tensor[[128, 128], pl.FP32],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            for i, (out_iter,) in pl.range(2, init_values=(output,)):
                offset_i: pl.Scalar[pl.INDEX] = i * 64
                tile: pl.Tile[[64, 128], pl.FP32] = pl.load(a, [offset_i, 0], [64, 128])
                updated: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile, [offset_i, 0], out_iter)
                result = pl.yield_(updated)
            return result

    mlir_code = _generate_default_mlir(NestedConstantProgram)

    # Collect all arith.constant lines with original indentation
    const_lines = [line for line in mlir_code.splitlines() if "arith.constant" in line]
    assert len(const_lines) >= 2, f"Expected at least 2 arith.constant lines, got {len(const_lines)}"

    # All should have exactly 2-space indent (function-body level)
    for line in const_lines:
        leading_spaces = len(line) - len(line.lstrip())
        assert leading_spaces == 2, f"arith.constant has {leading_spaces}-space indent (expected 2): {line!r}"


def test_pto_codegen_view_output_uses_physical_stride():
    """Output tensor with explicit tensor_view_.stride uses physical stride in make_tensor_view.

    When a tensor param has tensor_view_.stride set (e.g. from assemble-view pattern
    where the view shape is [32,32] but the physical tensor is [128,128]),
    the codegen must emit stride based on tensor_view_.stride, not shape.
    """
    span = ir.Span.unknown()

    # Create a tensor type with view shape [32,32] but physical stride [128, 1]
    stride = [
        ir.ConstInt(128, DataType.INDEX, span),
        ir.ConstInt(1, DataType.INDEX, span),
    ]
    tv = ir.TensorView(stride=stride, layout=ir.TensorLayout.ND)
    view_tensor_type = ir.TensorType([32, 32], DataType.FP32, memref=None, tensor_view=tv)

    # Build a minimal InCore function: load from a, store to out (view tensor)
    a_type = ir.TensorType([128, 128], DataType.FP32)
    a_param = ir.Var("a", a_type, span)
    out_param = ir.Var("out", view_tensor_type, span)

    load_call = ir.op.tile.load(a_param, [0, 0], [32, 32])
    tile_var = ir.Var("t", load_call.type, span)
    store_call = ir.op.tile.store(tile_var, [0, 0], out_param)
    result_var = ir.Var("result", store_call.type, span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(tile_var, load_call, span),
            ir.AssignStmt(result_var, store_call, span),
            ir.ReturnStmt([result_var], span),
        ],
        span,
    )

    func = ir.Function(
        "kernel",
        [a_param, out_param],
        [view_tensor_type],
        body,
        span,
        ir.FunctionType.InCore,
    )
    program = ir.Program([func], "ViewStrideTest", span)
    mlir_code = _generate_mlir(program)
    lines = _get_mlir_lines(mlir_code)

    # The out param (arg1) should use explicit stride [128, 1], not shape-based [32, 1]
    out_view_lines = _find_lines(lines, "pto.make_tensor_view %arg1")
    assert len(out_view_lines) == 1, f"Expected one make_tensor_view for out param, got: {out_view_lines}"
    out_view = out_view_lines[0]
    assert "strides = [%c128_index, %c1_index]" in out_view, (
        f"Out param stride should be [128, 1] (physical stride), not [32, 1] (view shape). Got: {out_view}"
    )

    # The a param (arg0) should still use shape-based stride [128, 1]
    a_view_lines = _find_lines(lines, "pto.make_tensor_view %arg0")
    assert len(a_view_lines) == 1
    assert "strides = [%c128_index, %c1_index]" in a_view_lines[0]


def test_pto_codegen_make_tensor_view_accepts_dynamic_shape_expressions():
    """make_tensor_view should lower non-Var dynamic shape/stride expressions via index casts."""
    span = ir.Span.unknown()
    th = ir.Var("TH", ir.ScalarType(DataType.INT64), span)
    tw = ir.Var("TW", ir.ScalarType(DataType.INT64), span)
    one = ir.ConstInt(1, DataType.INT64, span)
    th_plus_one = ir.add(th, one, span)
    tw_plus_one = ir.add(tw, one, span)

    tv = ir.TensorView(stride=[tw_plus_one, one], layout=ir.TensorLayout.ND)
    dyn_tensor_type = ir.TensorType([th_plus_one, tw], DataType.FP32, memref=None, tensor_view=tv)

    inp = ir.Var("inp", dyn_tensor_type, span)
    out = ir.Var("out", dyn_tensor_type, span)
    tile_type = ir.op.tile.load(inp, [0, 0], [8, 8]).type
    tile_var = ir.Var("t", tile_type, span)
    result_var = ir.Var("result", dyn_tensor_type, span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(tile_var, ir.op.tile.load(inp, [0, 0], [8, 8]), span),
            ir.AssignStmt(result_var, ir.op.tile.store(tile_var, [0, 0], out), span),
            ir.ReturnStmt([result_var], span),
        ],
        span,
    )

    func = ir.Function(
        "kernel",
        [inp, out, th, tw],
        [dyn_tensor_type],
        body,
        span,
        ir.FunctionType.InCore,
    )
    program = ir.Program([func], "DynamicViewExprTest", span)
    mlir_code = _generate_mlir(program)
    lines = _get_mlir_lines(mlir_code)

    assert "Expected ConstInt expression" not in mlir_code
    view_line = _single_line(lines, "pto.make_tensor_view %arg0")
    assert "shape = [" in view_line and "strides = [" in view_line
    index_cast_lines = _find_lines(lines, "arith.index_cast")
    assert index_cast_lines, (
        f"Expected dynamic shape/stride expressions to be cast to index. Got:\n{mlir_code}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

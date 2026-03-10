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
from pypto.ir import OptimizationStrategy, PassManager
from pypto.ir.builder import IRBuilder
from pypto.ir.op import tile
from pypto.ir.pto_codegen import (
    _generate_arg_unpacking,
    _generate_kernel_wrapper,
    _preprocess_ptoas_output,
    generate,
)

PTOCodegen = codegen.PTOCodegen

# Dynamic shape variables for wrapper dispatch tests
# pyright: reportUndefinedVariable=false
_TH = pl.dynamic("TH")
_TW = pl.dynamic("TW")


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
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed = pm.run_passes(_DynKernel)
    for func in transformed.functions.values():
        if func.func_type == ir.FunctionType.InCore:
            return func
    raise RuntimeError("No InCore function found in _DynKernel")


def _get_mlir_code(result):
    """Normalize generate() result to MLIR string (support both str and dict)."""
    return result if isinstance(result, str) else "".join(result.values())


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
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    @pl.program
    class BasicProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def test_func(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.add(tile_a, 1.0)
            pl.store(tile_b, offsets=[0, 0], output_tensor=b)
            return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    # Compile with Default strategy (applies necessary passes + codegen)
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed_program = pm.run_passes(BasicProgram)

    # Generate MLIR
    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify MLIR module structure
    assert "module {" in mlir_code
    assert "func.func @test_func" in mlir_code
    assert "return" in mlir_code
    assert "}" in mlir_code


def test_pto_codegen_tensor_parameters():
    """Test that tensor parameters generate correct make_tensor_view."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

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
            return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed_program = pm.run_passes(TensorParamProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

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
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    @pl.program
    class AllocTileProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def alloc_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_c = pl.mul(tile_a, tile_b)
            pl.store(tile_c, offsets=[0, 0], output_tensor=b)
            return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed_program = pm.run_passes(AllocTileProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify alloc_tile operations
    assert "pto.alloc_tile" in mlir_code
    assert "loc=vec" in mlir_code  # Vector buffer (PTO address space)
    assert "dtype=f32" in mlir_code
    assert "rows=32, cols=32" in mlir_code


def test_pto_codegen_tile_load_lowering():
    """Test that tile.load generates partition_view + tload."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    @pl.program
    class LoadProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def load_test(self, input: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]):
            tile = pl.load(input, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], output_tensor=output)
            return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed_program = pm.run_passes(LoadProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

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
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    @pl.program
    class StoreProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def store_test(self, input: pl.Tensor[[32, 32], pl.FP32], output: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(input, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], output_tensor=output)
            return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed_program = pm.run_passes(StoreProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify tstore generation
    assert "pto.tstore" in mlir_code
    assert "ins(" in mlir_code
    assert "outs(" in mlir_code


def test_pto_codegen_tile_mul():
    """Test that tile.mul generates pto.tmul."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

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
            return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed_program = pm.run_passes(MulProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify tmul generation
    assert "pto.tmul" in mlir_code
    assert "ins(" in mlir_code
    assert "outs(" in mlir_code


def test_pto_codegen_tile_adds():
    """Test that tile.adds generates pto.tadds with scalar constant."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    @pl.program
    class AddsProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def adds_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.add(tile_a, 3.14)
            pl.store(tile_b, offsets=[0, 0], output_tensor=b)
            return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed_program = pm.run_passes(AddsProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify tadds generation
    assert "pto.tadds" in mlir_code

    # Verify scalar constant generation
    assert "arith.constant" in mlir_code
    assert ": f32" in mlir_code


def test_pto_codegen_constants():
    """Test that constants are generated correctly."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    @pl.program
    class ConstantProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def const_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile_a, offsets=[0, 0], output_tensor=b)
            return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed_program = pm.run_passes(ConstantProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify index constants
    assert "arith.constant" in mlir_code
    assert ": index" in mlir_code
    assert "%c0" in mlir_code or "%c32" in mlir_code


def test_pto_codegen_ssa_naming():
    """Test that SSA value names are correct."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

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
            return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed_program = pm.run_passes(SSAProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify SSA value naming pattern
    assert "%arg0" in mlir_code  # Function parameters
    assert "%0" in mlir_code or "%1" in mlir_code  # Temporary values
    assert "%c" in mlir_code  # Constants


def test_pto_codegen_code_generation_order():
    """Test that code is generated in correct order: constants, views, allocs, body."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    @pl.program
    class OrderProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def order_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], output_tensor=b)
            return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed_program = pm.run_passes(OrderProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    lines = mlir_code.split("\n")

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
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    @pl.program
    class MultiFunc:
        @pl.function(type=pl.FunctionType.InCore)
        def func1(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], output_tensor=b)
            return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

        @pl.function(type=pl.FunctionType.InCore)
        def func2(self, x: pl.Tensor[[32, 32], pl.FP32], y: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(x, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], output_tensor=y)
            return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed_program = pm.run_passes(MultiFunc)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify both functions are present
    assert "func.func @func1" in mlir_code
    assert "func.func @func2" in mlir_code


def test_pto_codegen_reusability():
    """Test that the same PTOCodegen instance can be used multiple times."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    @pl.program
    class ReusableProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def test_func(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], output_tensor=b)
            return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed_program = pm.run_passes(ReusableProgram)

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
        # TH is dim 0 of first tensor a_0 — read from a_0_tensor->shapes[0]
        assert "a_0_tensor->shapes[0]" in code
        assert "int64_t TH" in code
        # TW is dim 1 of first tensor a_0 — read from a_0_tensor->shapes[1]
        assert "a_0_tensor->shapes[1]" in code
        assert "int64_t TW" in code
        # dynamic dims appended after tensor params
        assert names == ["a_0", "b_0", "output_0", "TH", "TW"]

    def test_dynamic_tensor_deduplicates_vars(self):
        # TH and TW each appear in a_0, b_0, and output_0 but should be extracted only once
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
        # Forward call must include dynamic dims TH and TW after tensor args (SSA-renamed with _0 suffix)
        assert "dyn_func(a_0, b_0, output_0, TH, TW);" in wrapper

    def test_dynamic_shape_shapes_extraction_in_wrapper(self):
        func = _get_dyn_incore_func()
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        assert "a_0_tensor->shapes[0]" in wrapper
        assert "a_0_tensor->shapes[1]" in wrapper


class TestGenerateSkipPtoas:
    """Tests for generate() with skip_ptoas=True."""

    def test_returns_pto_files(self, tmp_path):
        """When skip_ptoas=True, result keys for InCore functions end with .pto, not .cpp."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B_PTO)

        @pl.program
        class SkipPtoasProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def skip_test(
                self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
                out = pl.store(tile, offsets=[0, 0], output_tensor=b)
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed_program = pm.run_passes(SkipPtoasProgram)

        result = generate(transformed_program, str(tmp_path), skip_ptoas=True)

        kernel_keys = [k for k in result if k.startswith("kernels/")]
        assert len(kernel_keys) > 0, "Expected at least one kernel file"
        for key in kernel_keys:
            assert key.endswith(".pto"), f"Expected .pto extension, got: {key}"
            assert not key.endswith(".cpp"), f"Unexpected .cpp extension: {key}"


def test_pto_codegen_for_loop_tensor_iter_arg():
    """Test that tensor-typed iter_args in for loops generate correct tensor_view propagation.

    Regression test: before the fix, block.store with a tensor iter_arg (loop-carried
    output tensor) would crash with 'Tensor view not found for parameter: out_iter'.
    The codegen now propagates tensor_to_view_ mappings through ForStmt iter_args,
    return_vars, and IfStmt return_vars.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

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

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed_program = pm.run_passes(ForTensorIterArgProgram)

    codegen_inst = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_inst.generate(transformed_program))
    lines = [line.strip() for line in mlir_code.split("\n")]

    # The output tensor parameter (%arg1) must have a make_tensor_view
    view_lines = [line for line in lines if "pto.make_tensor_view %arg1" in line]
    assert len(view_lines) == 1, "Expected one make_tensor_view for output tensor (%arg1)"
    # Extract the SSA name of the output tensor view (e.g., "%2")
    output_view_name = view_lines[0].split("=")[0].strip()

    # scf.for iter_args must reference the output tensor view with tensor_view type
    for_lines = [line for line in lines if "scf.for" in line and "iter_args(" in line]
    assert len(for_lines) == 1, "Expected exactly one scf.for with iter_args"
    for_line = for_lines[0]
    assert f"= {output_view_name})" in for_line, (
        f"iter_args init value should be the output tensor view {output_view_name}"
    )
    assert "!pto.tensor_view<?x?xf32>" in for_line, "iter_args type should be !pto.tensor_view<?x?xf32>"

    # pto.partition_view must operate on the iter_arg (loop-carried tensor view)
    partition_lines = [line for line in lines if "pto.partition_view" in line]
    assert len(partition_lines) >= 2, "Expected at least 2 partition_view ops (load + store)"
    # The store's partition_view should use the iter_arg SSA name, not %arg1 directly
    # Extract the iter_arg SSA name from the scf.for line (e.g., "%4" from "iter_args(%4 = %2)")
    iter_arg_match = re.search(r"iter_args\((%\d+)\s*=", for_line)
    assert iter_arg_match, "Could not extract iter_arg SSA name from scf.for"
    iter_arg_name = iter_arg_match.group(1)
    store_partitions = [line for line in partition_lines if f"pto.partition_view {iter_arg_name}," in line]
    assert len(store_partitions) == 1, (
        f"Expected one partition_view on iter_arg {iter_arg_name} for the store path"
    )

    # pto.tstore must be present
    tstore_lines = [line for line in lines if line.startswith("pto.tstore")]
    assert len(tstore_lines) == 1, "Expected exactly one pto.tstore"

    # scf.yield must yield a tensor_view type value
    yield_lines = [line for line in lines if line.startswith("scf.yield")]
    assert len(yield_lines) == 1, "Expected exactly one scf.yield"
    assert "!pto.tensor_view<?x?xf32>" in yield_lines[0], "scf.yield type should be !pto.tensor_view<?x?xf32>"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

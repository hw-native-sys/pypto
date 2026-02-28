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
- Operator lowering (block.load/store/mul/adds -> pto.tload/tstore/tmul/tadds)
- SSA form with correct variable naming
"""

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir import OptimizationStrategy, PassManager
from pypto.ir.builder import IRBuilder
from pypto.ir.op import block
from pypto.ir.pto_codegen import (
    _generate_arg_unpacking,
    _generate_kernel_wrapper,
    _preprocess_ptoas_output,
    generate,
)

PTOCodegen = codegen.PTOCodegen


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
            t = ib.let("t", block.load(tensor_params[0], [0, 0], [16, 16]))
            result = ib.let("result", block.store(t, [0, 0], [16, 16], tensor_params[-1]))
            f.return_type(ir.TensorType([16, 16], DataType.FP32))
            ib.return_stmt(result)
        elif len(tensor_params) == 1:
            t = ib.let("t", block.load(tensor_params[0], [0, 0], [16, 16]))
            result = ib.let("result", block.store(t, [0, 0], [16, 16], tensor_params[0]))
            f.return_type(ir.TensorType([16, 16], DataType.FP32))
            ib.return_stmt(result)
        else:
            f.return_type(ir.ScalarType(DataType.FP32))
            ib.return_stmt(param_vars[0])

    return f.get_result()


def test_pto_codegen_basic_mlir_structure():
    """Test that PTOCodegen generates valid MLIR module structure."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class BasicProgram:
        @pl.function
        def test_func(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.add(tile_a, 1.0)
            pl.store(tile_b, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

    # Compile with PTOAS strategy (applies necessary passes + codegen)
    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
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
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class TensorParamProgram:
        @pl.function
        def tensor_param_func(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ):
            tile_a = pl.load(input_a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.load(input_b, offsets=[0, 0], shapes=[32, 32])
            tile_c = pl.mul(tile_a, tile_b)
            pl.store(tile_c, offsets=[0, 0], shapes=[32, 32], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
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
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class AllocTileProgram:
        @pl.function
        def alloc_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_c = pl.mul(tile_a, tile_b)
            pl.store(tile_c, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(AllocTileProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify alloc_tile operations
    assert "pto.alloc_tile" in mlir_code
    assert "loc=vec" in mlir_code  # Vector buffer (PTO address space)
    assert "dtype=f32" in mlir_code
    assert "rows=32, cols=32" in mlir_code


def test_pto_codegen_block_load_lowering():
    """Test that block.load generates partition_view + tload."""

    @pl.program
    class LoadProgram:
        @pl.function
        def load_test(self, input: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]):
            tile = pl.load(input, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
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


def test_pto_codegen_block_store_lowering():
    """Test that block.store generates partition_view + tstore."""

    @pl.program
    class StoreProgram:
        @pl.function
        def store_test(self, input: pl.Tensor[[32, 32], pl.FP32], output: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(input, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(StoreProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify tstore generation
    assert "pto.tstore" in mlir_code
    assert "ins(" in mlir_code
    assert "outs(" in mlir_code


def test_pto_codegen_block_mul():
    """Test that block.mul generates pto.tmul."""

    @pl.program
    class MulProgram:
        @pl.function
        def mul_test(
            self,
            a: pl.Tensor[[32, 32], pl.FP32],
            b: pl.Tensor[[32, 32], pl.FP32],
            c: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 32])
            tile_c = pl.mul(tile_a, tile_b)
            pl.store(tile_c, offsets=[0, 0], shapes=[32, 32], output_tensor=c)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(MulProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify tmul generation
    assert "pto.tmul" in mlir_code
    assert "ins(" in mlir_code
    assert "outs(" in mlir_code


def test_pto_codegen_block_adds():
    """Test that block.adds generates pto.tadds with scalar constant."""

    @pl.program
    class AddsProgram:
        @pl.function
        def adds_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.add(tile_a, 3.14)
            pl.store(tile_b, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
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
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class ConstantProgram:
        @pl.function
        def const_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile_a, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
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
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class SSAProgram:
        @pl.function
        def ssa_test(
            self,
            a: pl.Tensor[[32, 32], pl.FP32],
            b: pl.Tensor[[32, 32], pl.FP32],
            c: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 32])
            tile_c = pl.mul(tile_a, tile_b)
            pl.store(tile_c, offsets=[0, 0], shapes=[32, 32], output_tensor=c)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
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
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class OrderProgram:
        @pl.function
        def order_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
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

    @pl.program
    class MultiFunc:
        @pl.function
        def func1(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

        @pl.function
        def func2(self, x: pl.Tensor[[32, 32], pl.FP32], y: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(x, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=y)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(MultiFunc)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify both functions are present
    assert "func.func @func1" in mlir_code
    assert "func.func @func2" in mlir_code


def test_pto_codegen_reusability():
    """Test that the same PTOCodegen instance can be used multiple times."""

    @pl.program
    class ReusableProgram:
        @pl.function
        def test_func(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
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


class TestGenerateSkipPtoas:
    """Tests for generate() with skip_ptoas=True."""

    def test_returns_pto_files(self, tmp_path):
        """When skip_ptoas=True, result keys for InCore functions end with .pto, not .cpp."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.PTO)

        @pl.program
        class SkipPtoasProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def skip_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
                tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
                pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

        pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
        transformed_program = pm.run_passes(SkipPtoasProgram)

        result = generate(transformed_program, str(tmp_path), skip_ptoas=True)

        kernel_keys = [k for k in result if k.startswith("kernels/")]
        assert len(kernel_keys) > 0, "Expected at least one kernel file"
        for key in kernel_keys:
            assert key.endswith(".pto"), f"Expected .pto extension, got: {key}"
            assert not key.endswith(".cpp"), f"Unexpected .cpp extension: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

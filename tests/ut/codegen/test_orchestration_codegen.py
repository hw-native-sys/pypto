# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for orchestration code generation, including tuple return value handling."""

import difflib
import re
import textwrap

import pypto.language as pl
import pytest
from pypto import backend, codegen, passes
from pypto.backend import BackendType
from pypto.ir.builder import IRBuilder
from pypto.ir.op import tensor as tensor_ops
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import DataType, ir


def assert_code_equal(actual: str, expected: str) -> None:
    """Compare generated code against expected output, with unified diff on failure."""
    actual_stripped = actual.strip()
    expected_stripped = textwrap.dedent(expected).strip()
    if actual_stripped != expected_stripped:
        diff = "\n".join(
            difflib.unified_diff(
                expected_stripped.splitlines(),
                actual_stripped.splitlines(),
                fromfile="expected",
                tofile="actual",
                lineterm="",
            )
        )
        raise AssertionError(f"Code mismatch:\n{diff}")


def _ensure_arg_directions(program):
    """Phase-5 invariant: codegen requires Call.arg_directions to be populated.

    Tests that hand-build IR (without going through PassManager) need to invoke
    DeriveCallDirections before codegen so the Call sites carry explicit
    ArgDirection vectors. This helper makes that step a no-op when the program
    was already produced by the pass pipeline.
    """
    return passes.derive_call_directions()(program)


def _generate_orch_code(program) -> str:
    """Generate orchestration code using backend-agnostic codegen."""
    program = _ensure_arg_directions(program)
    for func in program.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            result = codegen.generate_orchestration(program, func)
            return result.code
    raise ValueError("No orchestration function found in program")


def _generate_orch_result(program) -> "codegen.OrchestrationResult":
    """Generate orchestration result using backend-agnostic codegen."""
    program = _ensure_arg_directions(program)
    for func in program.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            return codegen.generate_orchestration(program, func)
    raise ValueError("No orchestration function found in program")


class TestOrchestration:
    """Test orchestration codegen format."""

    def test_basic_structure(self):
        """Test codegen produces PTO2 format: make_tensor_external, Arg, pto2_rt_submit_aiv_task."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class BasicProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_basic(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                c: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                c = self.kernel_add(a, b, c)
                d = self.kernel_add(c, b, d)
                return d

        code = _generate_orch_code(BasicProgram)

        expected = """\
            #include <stddef.h>
            #include <stdint.h>
            #include <stdio.h>

            #include "pto_orchestration_api.h"

            extern "C" {

            __attribute__((visibility("default")))
            PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
                (void)orch_args;
                return PTO2OrchestrationConfig{
                    .expected_arg_count = 3,
                };
            }

            __attribute__((visibility("default")))
            void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
                // External tensors
                Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
                Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
                Tensor ext_d = from_tensor_arg(orch_args.tensor(2));

                PTO2_SCOPE() {
                    uint32_t c_ci_shapes[2] = {16, 16};
                    TensorCreateInfo c_ci(c_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_0 = alloc_tensors(c_ci);
                    const Tensor& c = alloc_0.get_ref(0);

                    // Task 0: kernel_add
                    Arg params_t0;
                    params_t0.add_input(ext_a);
                    params_t0.add_input(ext_b);
                    params_t0.add_output(c);
                    pto2_rt_submit_aiv_task(0, params_t0);

                    // Task 1: kernel_add
                    Arg params_t1;
                    params_t1.add_input(c);
                    params_t1.add_input(ext_b);
                    params_t1.add_output(ext_d);
                    pto2_rt_submit_aiv_task(0, params_t1);
                }
            }

            }  // extern "C"
        """
        assert_code_equal(code, expected)

    def test_tensor_read(self):
        """Test tensor.read uses orch_args.tensor().data_as<void>(), not host_t."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TensorReadProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_read(
                self,
                t: pl.Tensor[[4, 8], pl.FP32],
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                result: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                val: pl.Scalar[pl.FP32] = pl.tensor.read(t, [1, 3])  # noqa: F841
                result = self.kernel_add(a, b, result)
                return result

        code = _generate_orch_code(TensorReadProgram)

        # tensor.read uses orch_args.tensor(0).data_as<void>(), not host_t
        assert "idx_val" in code
        assert "static_cast<float*>(orch_args.tensor(0).data_as<void>())" in code
        assert "host_t" not in code

    def test_config_file(self):
        """Test orchestration result contains kernel function metadata."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ConfigProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_cfg(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                c = self.kernel_add(a, b, c)
                return c

        result = _generate_orch_result(ConfigProgram)

        assert "kernel_add" in result.func_name_to_id
        assert "kernel_add" in result.func_name_to_core_type

    def test_independent_tasks(self):
        """Test codegen with independent tasks (no dependencies needed)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class IndependentProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_indep(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                c = self.kernel_add(a, b, c)
                d = self.kernel_add(a, b, d)
                return c, d

        code = _generate_orch_code(IndependentProgram)

        # Two return tensors: c and d are both external
        assert "ext_c" in code
        assert "ext_d" in code
        assert "from_tensor_arg(" in code

        # Two tasks submitted
        assert code.count("pto2_rt_submit_aiv_task") == 2

        # PTO2_SCOPE wraps all task submissions
        assert "PTO2_SCOPE" in code

    def test_vector_example_dag(self):
        """Test codegen matching vector_example DAG structure.

        DAG:
          t0: c = kernel_add(a, b)           [outer scope]
          t1: d = kernel_add_scalar(c, 1.0)  [inner scope]
          t2: e = kernel_add_scalar(c, 2.0)  [inner scope]
          t3: g = kernel_mul(d, e)           [inner scope]
          t4: f = kernel_add(g, c)           [inner scope]
        Formula: f = (a + b + 1)(a + b + 2) + (a + b)
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class VectorExampleProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add_scalar(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                scalar: pl.Scalar[pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                x: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(x, scalar)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_mul(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.mul(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_vector(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                f: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                c: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                c = self.kernel_add(a, b, c)
                d: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                d = self.kernel_add_scalar(c, 1.0, d)
                e: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                e = self.kernel_add_scalar(c, 2.0, e)
                g: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                g = self.kernel_mul(d, e, g)
                f = self.kernel_add(g, c, f)
                return f

        code = _generate_orch_code(VectorExampleProgram)

        expected = """\
            #include <stddef.h>
            #include <stdint.h>
            #include <stdio.h>

            #include "pto_orchestration_api.h"

            extern "C" {

            __attribute__((visibility("default")))
            PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
                (void)orch_args;
                return PTO2OrchestrationConfig{
                    .expected_arg_count = 3,
                };
            }

            __attribute__((visibility("default")))
            void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
                // External tensors
                Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
                Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
                Tensor ext_f = from_tensor_arg(orch_args.tensor(2));

                PTO2_SCOPE() {
                    uint32_t c_ci_shapes[2] = {16, 16};
                    TensorCreateInfo c_ci(c_ci_shapes, 2, DataType::FLOAT32);
                    uint32_t d_ci_shapes[2] = {16, 16};
                    TensorCreateInfo d_ci(d_ci_shapes, 2, DataType::FLOAT32);
                    uint32_t e_ci_shapes[2] = {16, 16};
                    TensorCreateInfo e_ci(e_ci_shapes, 2, DataType::FLOAT32);
                    uint32_t g_ci_shapes[2] = {16, 16};
                    TensorCreateInfo g_ci(g_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_0 = alloc_tensors(c_ci, d_ci, e_ci, g_ci);
                    const Tensor& c = alloc_0.get_ref(0);
                    const Tensor& d = alloc_0.get_ref(1);
                    const Tensor& e = alloc_0.get_ref(2);
                    const Tensor& g = alloc_0.get_ref(3);

                    // Task 0: kernel_add
                    Arg params_t0;
                    params_t0.add_input(ext_a);
                    params_t0.add_input(ext_b);
                    params_t0.add_output(c);
                    pto2_rt_submit_aiv_task(0, params_t0);

                    // Task 1: kernel_add_scalar
                    Arg params_t1;
                    params_t1.add_input(c);
                    params_t1.add_output(d);
                    params_t1.add_scalar(to_u64(1.000000f));
                    pto2_rt_submit_aiv_task(1, params_t1);

                    // Task 2: kernel_add_scalar
                    Arg params_t2;
                    params_t2.add_input(c);
                    params_t2.add_output(e);
                    params_t2.add_scalar(to_u64(2.000000f));
                    pto2_rt_submit_aiv_task(1, params_t2);

                    // Task 3: kernel_mul
                    Arg params_t3;
                    params_t3.add_input(d);
                    params_t3.add_input(e);
                    params_t3.add_output(g);
                    pto2_rt_submit_aiv_task(2, params_t3);

                    // Task 4: kernel_add
                    Arg params_t4;
                    params_t4.add_input(g);
                    params_t4.add_input(c);
                    params_t4.add_output(ext_f);
                    pto2_rt_submit_aiv_task(0, params_t4);
                }
            }

            }  // extern "C"
        """
        assert_code_equal(code, expected)

    def test_tuple_intermediate(self):
        """Test tuple return as intermediate tensors: kernel_pair -> kernel_add."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TupleIntermediateProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_pair(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                out_s: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                out_d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                s: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                d: pl.Tile[[16, 16], pl.FP32] = pl.sub(a_tile, b_tile)
                rs: pl.Tensor[[16, 16], pl.FP32] = pl.store(s, [0, 0], out_s)
                rd: pl.Tensor[[16, 16], pl.FP32] = pl.store(d, [0, 0], out_d)
                return rs, rd

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_tuple_mid(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                result: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                x: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                y: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                x, y = self.kernel_pair(a, b, x, y)
                result = self.kernel_add(x, y, result)
                return result

        code = _generate_orch_code(TupleIntermediateProgram)

        # Tuple elements x, y are intermediate: TensorCreateInfo (not external)
        assert "TensorCreateInfo x_ci(" in code
        assert "TensorCreateInfo y_ci(" in code
        assert "DataType::FLOAT32" in code

        # Return tensor result is external
        assert "from_tensor_arg(orch_args.tensor(2))" in code

        # Two tasks: kernel_pair + kernel_add
        assert code.count("pto2_rt_submit_aiv_task") == 2

        # PTO2_SCOPE wraps all task submissions
        assert "PTO2_SCOPE" in code

    def test_tuple_output(self):
        """Test tuple return as final output: all elements are external tensors."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TupleOutputProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_pair(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                out_s: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                out_d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                s: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                d: pl.Tile[[16, 16], pl.FP32] = pl.sub(a_tile, b_tile)
                rs: pl.Tensor[[16, 16], pl.FP32] = pl.store(s, [0, 0], out_s)
                rd: pl.Tensor[[16, 16], pl.FP32] = pl.store(d, [0, 0], out_d)
                return rs, rd

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_tuple_out(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                x: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                y: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                x, y = self.kernel_pair(a, b, x, y)
                return x, y

        code = _generate_orch_code(TupleOutputProgram)

        # Both x and y are return tensors: from_tensor_arg(orch_args.tensor())
        assert "ext_x" in code
        assert "ext_y" in code
        assert "from_tensor_arg(orch_args.tensor(2))" in code
        assert "from_tensor_arg(orch_args.tensor(3))" in code

        # Only one task: kernel_pair
        assert code.count("pto2_rt_submit_aiv_task") == 1

        # PTO2_SCOPE wraps all task submissions
        assert "PTO2_SCOPE" in code

    def test_four_element_tuple(self):
        """Test 4-element tuple unpacking with mixed shapes as intermediate."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class FourTupleProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def online_update(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 16], pl.FP32],
                mi: pl.InOut[pl.Tensor[[16, 1], pl.FP32]],
                li: pl.InOut[pl.Tensor[[16, 1], pl.FP32]],
                oi: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
                dst: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 16], pl.FP32],
                pl.Tensor[[16, 16], pl.FP32],
            ]:
                mi_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(mi, [0, 0], [16, 1])
                li_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(li, [0, 0], [16, 1])
                oi_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(oi, [0, 0], [16, 16])
                dst_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(dst, [0, 0], [16, 16])
                mi_out: pl.Tensor[[16, 1], pl.FP32] = pl.store(mi_tile, [0, 0], mi)
                li_out: pl.Tensor[[16, 1], pl.FP32] = pl.store(li_tile, [0, 0], li)
                oi_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(oi_tile, [0, 0], oi)
                dst_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(dst_tile, [0, 0], dst)
                return mi_out, li_out, oi_out, dst_out

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_four_tuple(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 16], pl.FP32],
                mi_in: pl.Tensor[[16, 1], pl.FP32],
                li_in: pl.Tensor[[16, 1], pl.FP32],
                oi_in: pl.Tensor[[16, 16], pl.FP32],
                dst_in: pl.Tensor[[16, 16], pl.FP32],
                final: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                mi_in, li_in, oi_in, dst_in = self.online_update(
                    mij, lij, oi_new, mi_in, li_in, oi_in, dst_in
                )
                final = self.kernel_add(oi_in, dst_in, final)
                return final

        code = _generate_orch_code(FourTupleProgram)

        # All orch params are external tensors:
        # mij=0, lij=1, oi_new=2, mi_in=3, li_in=4, oi_in=5, dst_in=6, final=7
        assert "Tensor ext_mi_in = from_tensor_arg(orch_args.tensor(3))" in code
        assert "Tensor ext_li_in = from_tensor_arg(orch_args.tensor(4))" in code
        assert "Tensor ext_oi_in = from_tensor_arg(orch_args.tensor(5))" in code
        assert "Tensor ext_dst_in = from_tensor_arg(orch_args.tensor(6))" in code

        # Final return tensor is external
        assert "Tensor ext_final = from_tensor_arg(orch_args.tensor(7))" in code

        # Two tasks: online_update + kernel_add
        assert code.count("pto2_rt_submit_aiv_task") == 2

        # online_update: 3 In + 3 InOut + 1 Out = 7 params
        assert "params_t0.add_input(ext_mij)" in code
        assert "params_t0.add_inout(ext_mi_in)" in code
        assert "params_t0.add_output(ext_dst_in)" in code

        # kernel_add: 2 In + 1 Out = 3 params
        assert "params_t1.add_input(ext_oi_in)" in code
        assert "params_t1.add_output(ext_final)" in code

        # PTO2_SCOPE wraps all task submissions
        assert "PTO2_SCOPE" in code

    def test_tensor_create(self):
        """Test tensor.create generates TensorCreateInfo with shape/dtype."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TensorCreateProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_fill(
                self,
                a: pl.Tensor[[32, 32], pl.FP16],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP16]],
            ) -> pl.Tensor[[32, 32], pl.FP16]:
                t: pl.Tile[[32, 32], pl.FP16] = pl.load(a, [0, 0], [32, 32])
                out: pl.Tensor[[32, 32], pl.FP16] = pl.store(t, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_create(
                self,
                a: pl.Tensor[[32, 32], pl.FP16],
                result: pl.Out[pl.Tensor[[32, 32], pl.FP16]],
            ) -> pl.Tensor[[32, 32], pl.FP16]:
                buf: pl.Tensor[[32, 32], pl.FP16] = pl.create_tensor([32, 32], dtype=pl.FP16)
                result = self.kernel_fill(buf, result)
                return result

        code = _generate_orch_code(TensorCreateProgram)

        # tensor.create generates TensorCreateInfo; const Tensor& binding emitted at submit site
        # FP16 = DataType::FLOAT16
        assert "uint32_t buf_ci_shapes[2] = {32, 32};" in code
        assert "TensorCreateInfo buf_ci(buf_ci_shapes, 2, DataType::FLOAT16)" in code
        assert "const Tensor& buf = " in code
        assert "make_tensor_external(nullptr, buf_ci_shapes, 2, DataType::FLOAT16)" not in code

    def test_inplace_tensor(self):
        """Test inplace tensors use make_inout_param when a tensor is both input and output.

        Pattern from OnlineUpdateMultiOut: mi, li, oi are passed as input args
        and also appear as output (tuple return elements) of the same kernel call.
        The codegen should emit make_inout_param for these inplace tensors.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class InplaceProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def online_update(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 16], pl.FP32],
                mi: pl.InOut[pl.Tensor[[16, 1], pl.FP32]],
                li: pl.InOut[pl.Tensor[[16, 1], pl.FP32]],
                oi: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
                dst: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 16], pl.FP32],
                pl.Tensor[[16, 16], pl.FP32],
            ]:
                mi_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(mi, [0, 0], [16, 1])
                li_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(li, [0, 0], [16, 1])
                oi_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(oi, [0, 0], [16, 16])
                dst_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(dst, [0, 0], [16, 16])
                mi_out: pl.Tensor[[16, 1], pl.FP32] = pl.store(mi_tile, [0, 0], mi)
                li_out: pl.Tensor[[16, 1], pl.FP32] = pl.store(li_tile, [0, 0], li)
                oi_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(oi_tile, [0, 0], oi)
                dst_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(dst_tile, [0, 0], dst)
                return mi_out, li_out, oi_out, dst_out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_inplace(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 16], pl.FP32],
                mi: pl.InOut[pl.Tensor[[16, 1], pl.FP32]],
                li: pl.InOut[pl.Tensor[[16, 1], pl.FP32]],
                oi: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
                dst: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 16], pl.FP32],
                pl.Tensor[[16, 16], pl.FP32],
            ]:
                mi, li, oi, dst = self.online_update(mij, lij, oi_new, mi, li, oi, dst)
                return mi, li, oi, dst

        code = _generate_orch_code(InplaceProgram)

        expected = """\
            #include <stddef.h>
            #include <stdint.h>
            #include <stdio.h>

            #include "pto_orchestration_api.h"

            extern "C" {

            __attribute__((visibility("default")))
            PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
                (void)orch_args;
                return PTO2OrchestrationConfig{
                    .expected_arg_count = 7,
                };
            }

            __attribute__((visibility("default")))
            void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
                // External tensors
                Tensor ext_mij = from_tensor_arg(orch_args.tensor(0));
                Tensor ext_lij = from_tensor_arg(orch_args.tensor(1));
                Tensor ext_oi_new = from_tensor_arg(orch_args.tensor(2));
                Tensor ext_mi = from_tensor_arg(orch_args.tensor(3));
                Tensor ext_li = from_tensor_arg(orch_args.tensor(4));
                Tensor ext_oi = from_tensor_arg(orch_args.tensor(5));
                Tensor ext_dst = from_tensor_arg(orch_args.tensor(6));

                PTO2_SCOPE() {

                    // Task 0: online_update
                    Arg params_t0;
                    params_t0.add_input(ext_mij);
                    params_t0.add_input(ext_lij);
                    params_t0.add_input(ext_oi_new);
                    params_t0.add_inout(ext_mi);
                    params_t0.add_inout(ext_li);
                    params_t0.add_inout(ext_oi);
                    params_t0.add_output(ext_dst);
                    pto2_rt_submit_aiv_task(0, params_t0);
                }
            }

            }  // extern "C"
        """
        assert_code_equal(code, expected)

    def test_tensor_dim(self):
        """Test tensor.dim generates int64_t assignment with shape value."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TensorDimProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_dim(
                self,
                a: pl.Tensor[[64, 128], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                result: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                d0: pl.Scalar[pl.INT64] = pl.tensor.dim(a, 0)  # noqa: F841
                result_out = self.kernel_add(a, b, result)
                return result_out

        code = _generate_orch_code(TensorDimProgram)

        # tensor.dim generates int64_t assignment
        assert "int64_t d0 = 64" in code

    def test_for_loop_with_slice(self):
        """Test for loop + tensor.slice: simplified paged attention pattern.

        Exercises: for loop with dynamic bound, tensor.slice with dynamic offsets,
        kernel calls inside loop body.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ForViewProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_for_view(
                self,
                data: pl.Tensor[[64, 16], pl.FP32],
                bias: pl.Tensor[[16, 16], pl.FP32],
                config: pl.Tensor[[4], pl.INT64],
            ) -> pl.Tensor[[64, 16], pl.FP32]:
                n_blocks: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                out: pl.Tensor[[64, 16], pl.FP32] = data
                for i in pl.range(n_blocks):
                    chunk: pl.Tensor[[16, 16], pl.FP32] = pl.slice(data, [16, 16], [i * 16, 0])
                    result: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                    result = self.kernel_add(chunk, bias, result)  # noqa: F841
                return out

        code = _generate_orch_code(ForViewProgram)

        # For loop with dynamic bound from tensor.read
        assert "for (int64_t i = 0; i < n_blocks; i += 1)" in code

        # PTO2_SCOPE wraps the for loop body
        assert "PTO2_SCOPE()" in code

        # tensor.slice generates array variables and runtime .view() call with dynamic offset
        assert "uint32_t chunk_shapes[2] = {16, 16};" in code
        assert "uint32_t chunk_offsets[2] = {static_cast<uint32_t>((i * 16)), 0};" in code
        assert "Tensor chunk = ext_data.view(chunk_shapes, chunk_offsets);" in code

        # tensor.read generates host pointer access
        assert "static_cast<int64_t*>(orch_args.tensor(2).data_as<void>())" in code

        # kernel_add task submitted inside loop
        assert "pto2_rt_submit_aiv_task" in code

    def test_tensor_slice_with_valid_shape(self):
        """tensor.slice(valid_shape=...) should still emit a runtime tensor view."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ValidShapeSliceProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_slice(
                self,
                data: pl.Tensor[[64, 16], pl.FP32],
                valid_rows: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                chunk: pl.Tensor[[16, 16], pl.FP32] = pl.slice(
                    data, [16, 16], [0, 0], valid_shape=[valid_rows, 16]
                )
                return chunk

        code = _generate_orch_code(ValidShapeSliceProgram)

        assert "uint32_t chunk_shapes[2] = {16, 16};" in code
        assert "uint32_t chunk_offsets[2] = {0, 0};" in code
        assert "Tensor chunk = ext_data.view(chunk_shapes, chunk_offsets);" in code

    def test_tensor_reshape_external_input(self):
        """tensor.reshape on an external orchestration input emits Tensor::reshape on ext_<name>."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ReshapeExternalProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_reshape(
                self,
                data: pl.Tensor[[256], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                r: pl.Tensor[[16, 16], pl.FP32] = pl.reshape(data, [16, 16])
                return r

        code = _generate_orch_code(ReshapeExternalProgram)

        # Shape array emitted with the result variable name as prefix.
        assert "uint32_t r_shapes[2] = {16, 16};" in code
        # Reshape lowers to runtime Tensor::reshape on the external tensor handle.
        assert "Tensor r = ext_data.reshape(r_shapes, 2);" in code

    def test_tensor_reshape_after_slice(self):
        """slice -> reshape chain: reshape input is a local Tensor (no ext_ prefix)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ReshapeAfterSliceProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_slice_reshape(
                self,
                data: pl.Tensor[[4, 16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                chunk: pl.Tensor[[1, 16, 16], pl.FP32] = pl.slice(data, [1, 16, 16], [0, 0, 0])
                r: pl.Tensor[[16, 16], pl.FP32] = pl.reshape(chunk, [16, 16])
                return r

        code = _generate_orch_code(ReshapeAfterSliceProgram)

        # slice still emits view on the external tensor.
        assert "uint32_t chunk_shapes[3] = {1, 16, 16};" in code
        assert "Tensor chunk = ext_data.view(chunk_shapes, chunk_offsets);" in code
        # reshape emits its shape array and calls .reshape on the local Tensor (no ext_ prefix).
        assert "uint32_t r_shapes[2] = {16, 16};" in code
        assert "Tensor r = chunk.reshape(r_shapes, 2);" in code

    def test_tensor_transpose_external_input(self):
        """tensor.transpose on an external orchestration input emits Tensor::transpose on ext_<name>."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TransposeExternalProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_transpose(
                self,
                data: pl.Tensor[[16, 32], pl.FP16],
            ) -> pl.Tensor[[32, 16], pl.FP16]:
                t: pl.Tensor[[32, 16], pl.FP16] = pl.transpose(data, axis1=0, axis2=1)
                return t

        code = _generate_orch_code(TransposeExternalProgram)

        # transpose lowers to runtime Tensor::transpose on the external tensor handle.
        assert "Tensor t = ext_data.transpose(0, 1);" in code

    def test_tensor_transpose_negative_axis(self):
        """tensor.transpose with negative axis indices is normalized at codegen time."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TransposeNegativeProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_transpose_neg(
                self,
                data: pl.Tensor[[4, 8, 16], pl.FP16],
            ) -> pl.Tensor[[4, 16, 8], pl.FP16]:
                t: pl.Tensor[[4, 16, 8], pl.FP16] = pl.transpose(data, axis1=-1, axis2=-2)
                return t

        code = _generate_orch_code(TransposeNegativeProgram)

        # -1 / -2 on a 3D tensor should normalize to axes 2 and 1 respectively.
        assert "Tensor t = ext_data.transpose(2, 1);" in code

    def test_tensor_transpose_after_slice(self):
        """slice -> transpose chain: transpose input is a local Tensor (no ext_ prefix)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TransposeAfterSliceProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_slice_transpose(
                self,
                data: pl.Tensor[[4, 16, 32], pl.FP16],
            ) -> pl.Tensor[[1, 32, 16], pl.FP16]:
                chunk: pl.Tensor[[1, 16, 32], pl.FP16] = pl.slice(data, [1, 16, 32], [0, 0, 0])
                t: pl.Tensor[[1, 32, 16], pl.FP16] = pl.transpose(chunk, axis1=1, axis2=2)
                return t

        code = _generate_orch_code(TransposeAfterSliceProgram)

        # slice still emits view on the external tensor.
        assert "Tensor chunk = ext_data.view(chunk_shapes, chunk_offsets);" in code
        # transpose calls .transpose on the local Tensor (no ext_ prefix).
        assert "Tensor t = chunk.transpose(1, 2);" in code

    def test_if_statement(self):
        """Test if/else codegen with conditional scalar values."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class IfProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_process(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                flag: pl.Scalar[pl.INT64],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_if(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                for i in pl.range(4):
                    if i == 0:
                        is_first: pl.Scalar[pl.INT64] = pl.yield_(1)
                    else:
                        is_first: pl.Scalar[pl.INT64] = pl.yield_(0)
                    result: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                    result = self.kernel_process(a, is_first, result)
                return result

        code = _generate_orch_code(IfProgram)

        # If statement with comparison
        assert "if ((i == 0))" in code

        # PTO2_SCOPE wraps for loop body and if/else bodies
        assert "PTO2_SCOPE()" in code

        # Scalar assignment in both branches
        assert "is_first = 1" in code
        assert "is_first = 0" in code

    def test_multiple_tuple_calls(self):
        """Test that multiple tuple-returning calls produce correct per-call params.

        When two different kernel calls both return tuples, each call's Arg
        array should only contain outputs from that specific call, not outputs
        from other calls. Regression test for SSA base name collision in
        tuple_var_to_elements_ (all _tuple_tmp_N collapsed to _tuple_tmp).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class MultipleTupleProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_a(
                self,
                x: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
                y: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[16, 16], pl.FP32],
                pl.Tensor[[16, 16], pl.FP32],
            ]:
                xt: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                yt: pl.Tile[[16, 16], pl.FP32] = pl.load(y, [0, 0], [16, 16])
                x_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(xt, [0, 0], x)
                y_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(yt, [0, 0], y)
                return x_out, y_out

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_b(
                self,
                a: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
                b: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[16, 16], pl.FP32],
                pl.Tensor[[16, 16], pl.FP32],
            ]:
                at: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                bt: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                a_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(at, [0, 0], a)
                b_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(bt, [0, 0], b)
                return a_out, b_out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_multi_tuple(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                y: pl.Tensor[[16, 16], pl.FP32],
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                # First tuple-returning call
                x, y = self.kernel_a(x, y)
                # Second tuple-returning call
                a, b = self.kernel_b(a, b)
                return x

        code = _generate_orch_code(MultipleTupleProgram)

        # kernel_a should only have x and y params (2 inout), not a or b
        assert code.count("params_t0") >= 2
        # kernel_b should only have a and b params (2 inout), not x or y
        assert code.count("params_t1") >= 2

        # Count add_inout per task block: each should have exactly 2
        lines = code.split("\n")
        task0_params = []
        task1_params = []
        in_task0 = False
        in_task1 = False
        for line in lines:
            if "Arg params_t0" in line:
                in_task0 = True
            elif "Arg params_t1" in line:
                in_task1 = True
            elif "pto2_rt_submit" in line:
                in_task0 = False
                in_task1 = False
            if in_task0 and ("params_t0.add_" in line):
                task0_params.append(line.strip())
            if in_task1 and ("params_t1.add_" in line):
                task1_params.append(line.strip())

        # kernel_a: x, y as inout → 2 params
        assert len(task0_params) == 2, (
            f"kernel_a should have 2 params (x, y inout), got {len(task0_params)}: {task0_params}"
        )
        # kernel_b: a, b as inout → 2 params
        assert len(task1_params) == 2, (
            f"kernel_b should have 2 params (a, b inout), got {len(task1_params)}: {task1_params}"
        )

    def test_tuple_in_for_loop(self):
        """Test tuple-returning call inside for-loop produces no self-assignments.

        When a tuple-returning kernel is called both before and inside a for-loop,
        SSA conversion creates iter_args for the tuple intermediate (_tuple_tmp) and
        its unpacked elements. After SSA base name collapsing, these would produce
        self-assignments like `auto _tuple_tmp = _tuple_tmp;` (C++ UB) and
        `oi = oi;` (NOP). The codegen should skip these.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TupleForLoopProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_init(
                self,
                a: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
                b: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[16, 16], pl.FP32],
                pl.Tensor[[16, 16], pl.FP32],
            ]:
                at: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                bt: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                a_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(at, [0, 0], a)
                b_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(bt, [0, 0], b)
                return a_out, b_out

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_update(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                a: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
                b: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[16, 16], pl.FP32],
                pl.Tensor[[16, 16], pl.FP32],
            ]:
                xt: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                at: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                a_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(xt, [0, 0], a)
                b_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(at, [0, 0], b)
                return a_out, b_out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_tuple_loop(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                a_acc: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                b_acc: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                # Tuple call BEFORE the loop — makes _tuple_tmp/a_acc/b_acc loop-carried
                a_acc, b_acc = self.kernel_init(a_acc, b_acc)
                for i in pl.range(4):
                    # Tuple call INSIDE the loop — triggers iter_arg self-assignment
                    a_acc, b_acc = self.kernel_update(x, a_acc, b_acc)
                return a_acc

        code = _generate_orch_code(TupleForLoopProgram)

        # No self-assignment in iter_arg init
        assert "auto _tuple_tmp = _tuple_tmp" not in code
        assert "auto a_acc = a_acc" not in code
        assert "auto b_acc = b_acc" not in code

        # No self-assignment in yield
        assert "_tuple_tmp = _tuple_tmp;" not in code
        assert "a_acc = a_acc;" not in code
        assert "b_acc = b_acc;" not in code

        # TensorCreateInfo declarations exist (exactly once each)
        # a_acc is a return value → external (from_tensor_arg(orch_args.tensor()))
        assert code.count("Tensor ext_a_acc = from_tensor_arg(orch_args.tensor(1))") == 1
        assert code.count("TensorCreateInfo b_acc_ci(") == 1

        # For loop exists with correct structure
        assert "for (int64_t i = 0; i < 4; i += 1)" in code
        assert "PTO2_SCOPE()" in code

        # Both tasks submitted
        assert "kernel_init" in code
        assert "kernel_update" in code
        assert code.count("pto2_rt_submit_aiv_task") == 2

    def test_loop_carried_internal_tensor_uses_hoisted_state_after_loop(self):
        """Loop-carried internal tensors should remain consumable after the loop."""

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class LoopCarriedStateProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                out_tensor: pl.Tensor[[16, 16], pl.FP32] = pl.store(x_tile, [0, 0], out)
                return out_tensor

            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                out_tensor: pl.Tensor[[16, 16], pl.FP32] = pl.store(x_tile, [0, 0], out)
                return out_tensor

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                acc: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                for i in pl.range(2):
                    acc = self.fill(x, acc)
                out = self.consume(acc, out)
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(LoopCarriedStateProgram)
        code = _generate_orch_code(transformed)

        assert "alloc_tensors(acc_ci)" in code
        assert "const Tensor& acc = alloc_0.get_ref(0);" in code
        assert "make_tensor_external(nullptr" not in code
        assert "acc__loop_state" not in code
        assert "params_t1.add_input(acc);" in code

    def test_for_loop_with_inplace_return_after_passes(self):
        """Test inplace detection when return var has compound auto-name suffixes from pass pipeline.

        When an Opaque function with auto_incore + parallel(chunk=) goes through the full
        pass pipeline (SSA → split_chunked_loops → interchange_chunk_loops → outline), the
        return var acquires compound suffixes like "__co_l0_rv_v1". GetSSABaseName must
        strip all of these to match the return var back to the original param name for correct
        inplace detection (2 arg slots, not 3).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ChunkedInplaceProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def add_one(
                self,
                input_tensor: pl.Tensor[[1024, 256], pl.FP32],
                output_tensor: pl.Tensor[[1024, 256], pl.FP32],
            ) -> pl.Tensor[[1024, 256], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for r in pl.parallel(0, 1024, 1, chunk=64, chunk_policy="leading_full"):
                        row_tile = pl.slice(input_tensor, [1, 256], [r, 0])
                        row_result = pl.add(row_tile, 1.0)
                        output_tensor = pl.assemble(output_tensor, row_result, [r, 0])
                return output_tensor

        # Run the full pass pipeline to produce compound SSA suffixes
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(ChunkedInplaceProgram)

        code = _generate_orch_code(transformed)

        # Inplace detection: output_tensor return var should match the param,
        # so only 2 orch arg slots (input_tensor + output_tensor), not 3
        assert "expected_arg_count = 2" in code
        assert "from_tensor_arg(orch_args.tensor(0))" in code  # input_tensor
        assert "from_tensor_arg(orch_args.tensor(1))" in code  # output_tensor

        # No third orch entry for the compound-named return var
        assert "orch_args.tensor(2)" not in code

        # Task params should use ext_output_tensor (the inplace param), not a separate buffer
        assert "ext_output_tensor)" in code
        assert "ext_output_tensor_iter" not in code

    def test_tensor_assemble_uses_precomputed_view(self):
        """tensor.assemble should lower to a pre-generated target view, not a host copy."""

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class AssembleViewProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def fill_row(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                r: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            ) -> pl.Tensor[[1, 8], pl.FP32]:
                row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(x, [r, 0], [1, 8])
                out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
                return out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_assemble_view(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                for r in pl.range(4):
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                    row_done = self.fill_row(x, r, row)
                    out = pl.assemble(out, row_done, [r, 0])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(AssembleViewProgram)

        code = _generate_orch_code(transformed)

        assert "Tensor row = ext_out.view(row_shapes, row_offsets);" in code
        assert "params_t0.add_inout(row)" in code
        assert "Tensor row = make_tensor(" not in code
        assert "memcpy(" not in code
        assert "ext_out = out;" not in code

    def test_tensor_assemble_duplicate_source_root_skips_view_rewrite(self):
        """A source buffer assembled more than once must keep its standalone allocation."""

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class DuplicateAssembleProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def fill_row(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                r: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            ) -> pl.Tensor[[1, 8], pl.FP32]:
                row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(x, [r, 0], [1, 8])
                out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
                return out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_duplicate_assemble(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                zero: pl.Scalar[pl.INDEX] = 0
                row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                row = self.fill_row(x, zero, row)
                out = pl.assemble(out, row, [0, 0])
                out = pl.assemble(out, row, [1, 0])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(DuplicateAssembleProgram)

        code = _generate_orch_code(transformed)

        assert "TensorCreateInfo row_ci(row_ci_shapes, 2, DataType::FLOAT32);" in code
        assert "const Tensor& row = " in code
        assert "make_tensor_external(nullptr, row_ci_shapes, 2, DataType::FLOAT32)" not in code
        assert "Tensor row = ext_out.view(row_shapes, row_offsets);" not in code

    def test_tensor_assemble_slice_source_does_not_require_view_fast_path(self):
        """tensor.assemble should stay codegenable when the source is not a rewritten tensor.create."""

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class SliceAssembleProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_slice_source(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                chunk: pl.Tensor[[1, 8], pl.FP32] = pl.slice(x, [1, 8], [0, 0])
                out = pl.assemble(out, chunk, [0, 0])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(SliceAssembleProgram)

        code = _generate_orch_code(transformed)

        assert "Tensor chunk = ext_x.view(chunk_shapes, chunk_offsets);" in code
        assert "Tensor chunk = ext_out.view(chunk_shapes, chunk_offsets);" not in code

    def test_param_with_numeric_suffix(self):
        """Regression test for issue #573: params with numeric suffixes must not be collapsed.

        When function params have names like `out_0` and `out_1`,
        GetSSABaseName previously stripped the numeric suffix, collapsing
        both to `out`. This caused duplicate ARG_PTR defines and merged
        external tensors. With VarPtr-based identity, each param retains
        its distinct identity regardless of name patterns.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class NumericSuffixProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
                out_0: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
                out_1: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[16, 16], pl.FP32],
                pl.Tensor[[16, 16], pl.FP32],
            ]:
                xt: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                r0: pl.Tensor[[16, 16], pl.FP32] = pl.store(xt, [0, 0], out_0)
                r1: pl.Tensor[[16, 16], pl.FP32] = pl.store(xt, [0, 0], out_1)
                return r0, r1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_numeric(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out_0: pl.Tensor[[16, 16], pl.FP32],
                out_1: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out_0, out_1 = self.kernel(x, out_0, out_1)
                return out_0

        code = _generate_orch_code(NumericSuffixProgram)

        # Each param must get a distinct orch index
        assert "from_tensor_arg(orch_args.tensor(0))" in code  # x
        assert "from_tensor_arg(orch_args.tensor(1))" in code  # out_0
        assert "from_tensor_arg(orch_args.tensor(2))" in code  # out_1

        # No collapsed names
        assert "ARG_PTR" not in code

        # Each param gets its own make_tensor_external
        assert "ext_out_0" in code
        assert "ext_out_1" in code

        # 3 tensor params expected
        assert "expected_arg_count = 3" in code

        # Tuple-return elements must not be collapsed into a single alias
        assert "Tensor& out =" not in code

    def test_repeated_auto_output_buffers_get_unique_names(self):
        """Repeated auto-generated output buffers should keep distinct emitted names."""

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class RepeatedAutoOutputProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                return z

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_repeat(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                first: pl.Tensor[[64], pl.FP32] = self.kernel_add(x, y)
                second: pl.Tensor[[64], pl.FP32] = self.kernel_add(first, y)
                return second

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(RepeatedAutoOutputProgram)

        code = _generate_orch_code(transformed)

        assert code.count("TensorCreateInfo ret0__out_ci(") == 1
        assert code.count("TensorCreateInfo ret0__out_1_ci(") == 1
        assert "alloc_tensors(ret0__out_ci, ret0__out_1_ci)" in code
        # Each ret0__out{,_1} is the unique writer of its local root in this scope
        # and has no sequential ancestor, so DeriveCallDirections keeps both as
        # OutputExisting (→ add_output) rather than promoting to InOut.
        assert "params_t0.add_output(ret0__out)" in code
        assert "params_t1.add_output(ret0__out_1)" in code
        assert "const Tensor& first = ret0__out;" in code
        assert "const Tensor& second" not in code

    def test_unused_alias_not_emitted(self):
        """Alias for a kernel result that is never consumed downstream should be omitted."""

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class UnusedAliasProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_op(
                self,
                a: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], a)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_unused(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                result: pl.Tensor[[16, 16], pl.FP32] = self.kernel_op(a, b)
                return result

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(UnusedAliasProgram)
        code = _generate_orch_code(transformed)

        assert "pto2_rt_submit_" in code
        assert "const Tensor& result" not in code

    def test_multi_scope_alloc_tensors_batching(self):
        """Each scope (function body, for body) batches its own alloc_tensors independently."""

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class MultiScopeAllocProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_op(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                y: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                # Outer scope: 2 create_tensor -> batched into alloc_0
                t1: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                t2: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                t1 = self.kernel_op(x, y, t1)
                t2 = self.kernel_op(t1, x, t2)
                for i in pl.range(4):
                    # Inner scope (for body): 1 create_tensor -> batched into alloc_1
                    tmp: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                    tmp = self.kernel_op(t2, y, tmp)
                    t2 = self.kernel_op(tmp, t1, t2)
                out = self.kernel_op(t2, t1, out)
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(MultiScopeAllocProgram)
        code = _generate_orch_code(transformed)

        # Outer scope batches t1 and t2 together
        assert "alloc_tensors(t1_ci, t2_ci)" in code
        # Inner scope (for body) has its own alloc_tensors for tmp
        assert "alloc_tensors(tmp_ci)" in code
        # Two separate alloc_tensors calls (alloc_0 and alloc_1)
        assert "alloc_0 = alloc_tensors(" in code
        assert "alloc_1 = alloc_tensors(" in code
        # Verify bindings from each alloc
        assert "const Tensor& t1 = alloc_0.get_ref(0);" in code
        assert "const Tensor& t2 = alloc_0.get_ref(1);" in code
        assert "const Tensor& tmp = alloc_1.get_ref(0);" in code

    def test_alloc_tensors_splits_at_16(self):
        """More than 16 create_tensor in one scope are split into multiple alloc_tensors calls."""

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ManyCreateProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_copy(
                self,
                x: pl.Tensor[[8], pl.FP32],
                out: pl.Out[pl.Tensor[[8], pl.FP32]],
            ) -> pl.Tensor[[8], pl.FP32]:
                t: pl.Tile[[8], pl.FP32] = pl.load(x, [0], [8])
                r: pl.Tensor[[8], pl.FP32] = pl.store(t, [0], out)
                return r

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                src: pl.Tensor[[8], pl.FP32],
                dst: pl.Out[pl.Tensor[[8], pl.FP32]],
            ) -> pl.Tensor[[8], pl.FP32]:
                t0: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t1: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t2: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t3: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t4: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t5: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t6: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t7: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t8: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t9: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t10: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t11: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t12: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t13: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t14: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t15: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t16: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t17: pl.Tensor[[8], pl.FP32] = pl.create_tensor([8], dtype=pl.FP32)
                t0 = self.kernel_copy(src, t0)
                t1 = self.kernel_copy(t0, t1)
                t2 = self.kernel_copy(t1, t2)
                t3 = self.kernel_copy(t2, t3)
                t4 = self.kernel_copy(t3, t4)
                t5 = self.kernel_copy(t4, t5)
                t6 = self.kernel_copy(t5, t6)
                t7 = self.kernel_copy(t6, t7)
                t8 = self.kernel_copy(t7, t8)
                t9 = self.kernel_copy(t8, t9)
                t10 = self.kernel_copy(t9, t10)
                t11 = self.kernel_copy(t10, t11)
                t12 = self.kernel_copy(t11, t12)
                t13 = self.kernel_copy(t12, t13)
                t14 = self.kernel_copy(t13, t14)
                t15 = self.kernel_copy(t14, t15)
                t16 = self.kernel_copy(t15, t16)
                t17 = self.kernel_copy(t16, t17)
                dst = self.kernel_copy(t17, dst)
                return dst

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(ManyCreateProgram)
        code = _generate_orch_code(transformed)

        # First batch: 16 tensors
        assert "alloc_0 = alloc_tensors(" in code
        first_alloc_line = [line for line in code.splitlines() if "alloc_0 = alloc_tensors(" in line][0]
        assert first_alloc_line.count("_ci") == 16

        # Second batch: remaining 2 tensors
        assert "alloc_1 = alloc_tensors(" in code
        second_alloc_line = [line for line in code.splitlines() if "alloc_1 = alloc_tensors(" in line][0]
        assert second_alloc_line.count("_ci") == 2

        # Second batch get_ref indices reset to 0
        assert "alloc_1.get_ref(0)" in code
        assert "alloc_1.get_ref(1)" in code

    def test_create_tensor_with_local_shape_dep_not_hoisted(self):
        """create_tensor whose shape depends on a locally-defined variable is not hoisted.

        Constructs IR directly where a tensor.create has a shape referencing a
        Var defined by an earlier statement. Verifies the create is emitted
        in-place (its own alloc_tensors) rather than batched at scope entry.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        span = ir.Span.unknown()
        INDEX = DataType.INDEX
        FP32 = DataType.FP32
        dyn_n = ir.Var("n", ir.ScalarType(INDEX), span)
        dim_expr = ir.Mul(dyn_n, ir.ConstInt(16, INDEX, span), INDEX, span)

        tensor_create_op = ir.Op("tensor.create")
        static_create_call = ir.Call(
            tensor_create_op,
            [],
            ir.TensorType([16, 8], FP32),
            span,
        )
        dyn_create_call = ir.Call(
            tensor_create_op,
            [],
            ir.TensorType([dim_expr, ir.ConstInt(8, INDEX, span)], FP32),
            span,
        )

        t1_var = ir.Var("t1", ir.TensorType([16, 8], FP32), span)
        t2_var = ir.Var(
            "t2",
            ir.TensorType([dim_expr, ir.ConstInt(8, INDEX, span)], FP32),
            span,
        )

        stmts = [
            ir.AssignStmt(t1_var, static_create_call, span),
            ir.AssignStmt(dyn_n, ir.ConstInt(4, INDEX, span), span),
            ir.AssignStmt(t2_var, dyn_create_call, span),
            ir.ReturnStmt(span),
        ]
        body = ir.SeqStmts(stmts, span)

        func = ir.Function(
            "orch",
            [],
            [],
            body,
            span,
            type=ir.FunctionType.Orchestration,
        )
        program = ir.Program([func], "test_prog", span)

        code = codegen.generate_orchestration(program, func).code
        lines = code.splitlines()

        def line_index_containing(text):
            for i, line in enumerate(lines):
                if text in line:
                    return i
            return -1

        # t1 (constant shape) is batched at scope entry
        assert line_index_containing("alloc_tensors(t1_ci)") >= 0

        # n must be defined BEFORE t2_ci
        n_line = line_index_containing("int64_t n =")
        t2_ci_line = line_index_containing("TensorCreateInfo t2_ci(")
        assert n_line >= 0 and t2_ci_line >= 0
        assert n_line < t2_ci_line, f"n definition (line {n_line}) must precede t2_ci (line {t2_ci_line})"

        # t2 gets its own alloc_tensors, separate from t1
        t2_alloc_line = line_index_containing("alloc_tensors(t2_ci)")
        assert t2_alloc_line >= 0, "t2 should get its own alloc_tensors(t2_ci)"
        assert t2_alloc_line > n_line, "t2 alloc must come after n definition"

    def test_scalar_taskarg(self):
        """Scalar params get ChipStorageTaskArgs scalar slots (0-indexed) via from_u64<T>()."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class MultiScalarProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                factor: pl.Scalar[pl.INT64],
                count: pl.Scalar[pl.INT32],
                scale: pl.Scalar[pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t = pl.load(a, [0, 0], [16, 16])
                r = pl.store(t, [0, 0], out)
                return r

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_multi(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                factor: pl.Scalar[pl.INT64],
                count: pl.Scalar[pl.INT32],
                scale: pl.Scalar[pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out = self.kernel(a, out, factor, count, scale)
                return out

        code = _generate_orch_code(MultiScalarProgram)

        # Tensors at orch_args.tensor(0..1), scalars at orch_args.scalar(0..2)
        assert "from_tensor_arg(orch_args.tensor(0))" in code
        assert "from_tensor_arg(orch_args.tensor(1))" in code
        assert "from_u64<int64_t>(orch_args.scalar(0))" in code
        assert "from_u64<int32_t>(orch_args.scalar(1))" in code
        assert "from_u64<float>(orch_args.scalar(2))" in code
        assert ".expected_arg_count = 5," in code


class TestTensorReadWriteOffsetCodegen:
    """Tests verifying that multi-dimensional indices are correctly converted to flat offsets in codegen."""

    def test_tensor_read_constant_1d(self):
        """1D tensor [8], read(t, [3]) -> flat offset 3 (inlined constant)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(self, t: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                val: pl.Scalar[pl.FP32] = pl.tensor.read(t, [3])  # noqa: F841
                return t

        code = _generate_orch_code(Prog)
        assert "static_cast<float*>(orch_args.tensor(0).data_as<void>())[3]" in code

    def test_tensor_read_constant_2d(self):
        """2D tensor [4, 8], read(t, [1, 3]) -> flat offset 1*8+3=11 (computed correctly)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(self, t: pl.Tensor[[4, 8], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                val: pl.Scalar[pl.FP32] = pl.tensor.read(t, [1, 3])  # noqa: F841
                return t

        code = _generate_orch_code(Prog)
        # The flat offset expression 1*8+3=11 is generated (either inlined or via idx_val)
        assert ("orch_args.tensor(0).data_as<void>())[11]" in code) or ("1 * 8 + 3" in code)
        assert "orch_args.tensor(0).data_as<void>())" in code

    def test_tensor_read_constant_3d(self):
        """3D tensor [2, 4, 8], read(t, [1, 2, 3]) -> flat offset 1*32+2*8+3=51."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(self, t: pl.Tensor[[2, 4, 8], pl.FP32]) -> pl.Tensor[[2, 4, 8], pl.FP32]:
                val: pl.Scalar[pl.FP32] = pl.tensor.read(t, [1, 2, 3])  # noqa: F841
                return t

        code = _generate_orch_code(Prog)
        # The flat offset expression is generated (either inlined as 51 or as computed expression)
        assert ("orch_args.tensor(0).data_as<void>())[51]" in code) or (
            "1 * 4 * 8" in code and "2 * 8" in code
        )
        assert "orch_args.tensor(0).data_as<void>())" in code

    def test_tensor_read_variable_index(self):
        """2D tensor [4, 8], read(t, [i, j]) -> generates idx_val = i * 8 + j."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                t: pl.Tensor[[4, 8], pl.FP32],
                config: pl.Tensor[[2], pl.INT64],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                row: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                col: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                val: pl.Scalar[pl.FP32] = pl.tensor.read(t, [row, col])  # noqa: F841
                return t

        code = _generate_orch_code(Prog)
        assert "idx_val" in code
        assert "* 8" in code

    def test_tensor_write_constant_2d(self):
        """2D tensor [4, 8], write(t, [1, 3], val) -> flat offset 11."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(self, t: pl.Tensor[[4, 8], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                val: pl.Scalar[pl.FP32] = pl.tensor.read(t, [0, 0])
                pl.tensor.write(t, [1, 3], val)
                return t

        code = _generate_orch_code(Prog)
        # Write generates flat offset 11 or the expression 1*8+3
        assert ("orch_args.tensor(0).data_as<void>())[11]" in code) or ("1 * 8 + 3" in code)
        assert "orch_args.tensor(0).data_as<void>())" in code

    def test_infer_output_param_from_loop_carried_store(self):
        """Loop-carried store to a default-In tensor should emit output params."""

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class OutputProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                for i, (out_iter,) in pl.range(0, 64, 16, init_values=(out,)):
                    x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [i], [16])
                    out_next: pl.Tensor[[64], pl.FP32] = pl.store(x_tile, [i], out_iter)
                    result = pl.yield_(out_next)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                out = self.fill(x, out)
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(OutputProgram)
        code = _generate_orch_code(transformed)

        assert "params_t0.add_input(ext_x)" in code
        assert "params_t0.add_output(ext_out)" in code

    def test_infer_inout_param_from_loop_carried_read_modify_write(self):
        """Loop-carried read-modify-write should emit inout params."""

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class InOutProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def accumulate(
                self,
                x: pl.Tensor[[64], pl.FP32],
                acc: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc_iter,) in pl.range(0, 64, 16, init_values=(acc,)):
                    x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [i], [16])
                    acc_tile: pl.Tile[[16], pl.FP32] = pl.load(acc_iter, [i], [16])
                    sum_tile: pl.Tile[[16], pl.FP32] = pl.add(x_tile, acc_tile)
                    acc_next: pl.Tensor[[64], pl.FP32] = pl.store(sum_tile, [i], acc_iter)
                    result = pl.yield_(acc_next)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[64], pl.FP32],
                acc: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc = self.accumulate(x, acc)
                return acc

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(InOutProgram)
        code = _generate_orch_code(transformed)

        assert "params_t0.add_input(ext_x)" in code
        assert "params_t0.add_inout(ext_acc)" in code

    def test_mixed_loop_carried_and_full_tuple_return(self):
        """ForStmt yield + tile.store outputs in same kernel get correct return-to-param mapping.

        The NormalizeReturnOrder pass reorders ReturnStmt values so that
        return[i] corresponds to the i-th Out/InOut parameter in declaration
        order.  This test verifies that mixed ForStmt yield and tile.store
        returns produce distinct get_ref indices and distinct consumer inputs.
        """

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class MixedReturnProgram:
            @pl.function
            def main(
                self,
                src: pl.Tensor[[4, 16], pl.FP32],
                final_out: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                dst = pl.create_tensor([4, 16], dtype=pl.FP32)
                acc = pl.create_tensor([4, 16], dtype=pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP):
                    # ForStmt: assemble rows into dst (produces yield return).
                    for i in pl.range(4):
                        row = pl.slice(src, [1, 16], [i, 0])
                        dst = pl.assemble(dst, row, [i, 0])
                    # Top-level assemble into acc (produces tile.store return).
                    full_view = pl.slice(src, [4, 16], [0, 0])
                    acc = pl.assemble(acc, full_view, [0, 0])
                with pl.at(level=pl.Level.CORE_GROUP):
                    # Consumer: uses both dst and acc from previous kernel.
                    dst_tile = pl.slice(dst, [4, 16], [0, 0])
                    acc_tile = pl.slice(acc, [4, 16], [0, 0])
                    result = pl.add(dst_tile, acc_tile)
                    final_out = pl.assemble(final_out, result, [0, 0])
                return final_out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(MixedReturnProgram)
        code = _generate_orch_code(transformed)

        # Two tasks: mixed_kernel + consumer
        assert code.count("pto2_rt_submit_aiv_task") == 2

        # The mixed kernel returns a tuple of (acc, dst).
        # acc comes from tile.store to an acc Out param.
        # dst comes from ForStmt yield tracing back to a dst Out param.
        # Before the fix, dst would incorrectly alias to the acc Out param.

        # Ensure the two alloc_tensors aliases reference DIFFERENT get_ref indices.
        get_ref_matches = re.findall(r"alloc_\d+\.get_ref\((\d+)\)", code)
        assert len(set(get_ref_matches)) >= 2, (
            f"Expected at least 2 distinct get_ref indices, got {get_ref_matches}"
        )

        # Verify the consumer receives both tuple outputs as distinct inputs.
        t1_inputs = re.findall(r"params_t1\.add_input\(([^)]+)\)", code)
        assert len(t1_inputs) >= 2, (
            f"Consumer kernel should have at least 2 inputs (acc + dst), got {len(t1_inputs)}"
        )
        assert len(set(t1_inputs)) == len(t1_inputs), (
            f"Consumer inputs should all be distinct tensors, got {t1_inputs}"
        )

    def test_group_submit_uses_both_aiv_slots_for_split_vector_kernel(self):
        """Cross-core split inferred from pipe ops should reuse one AIV kernel across both slots."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class SplitGroupProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def vector_producer(
                self,
                a: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            ):
                v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=v2c_peer)
                tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
                pl.tpush_to_aic(tile_a, split=1)

            @pl.function(type=pl.FunctionType.AIC)
            def cube_consumer(
                self,
                a: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
                pl.aic_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=pipe_buf)
                received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=1)
                pl.tfree_to_aiv(received)
                updated: pl.Tensor[[16, 16], pl.FP16] = pl.store(received, [0, 0], out)
                return updated

            @pl.function(type=pl.FunctionType.Group)
            def group_func(
                self,
                a: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                updated = self.cube_consumer(a, out)
                self.vector_producer(a, out)
                return updated

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                updated = self.group_func(a, out)
                return updated

        with passes.PassContext([], passes.VerificationLevel.NONE):
            transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(SplitGroupProgram)
        vector_producer = transformed.get_function("vector_producer")
        cube_consumer = transformed.get_function("cube_consumer")
        assert vector_producer is not None
        assert transformed.get_function("vector_producer__aiv1") is None
        assert cube_consumer is not None
        assert vector_producer.split == ir.SplitMode.UP_DOWN
        assert cube_consumer.split == ir.SplitMode.UP_DOWN

        orch_result = _generate_orch_result(transformed)
        code = orch_result.code
        expected_ids = (
            orch_result.func_name_to_id["cube_consumer"],
            orch_result.func_name_to_id["vector_producer"],
            orch_result.func_name_to_id["vector_producer"],
        )

        assert f"MixedKernels mixed_0 = {{{expected_ids[0]}, {expected_ids[1]}, {expected_ids[2]}}};" in code
        assert "pto2_rt_submit_task(mixed_0, params_t0);" in code

    def test_no_split_mixed_group_dispatches_same_aiv_on_both_lanes(self):
        """Ascend910B no-split mixed kernels should still launch both AIV lanes."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class NoSplitGroupProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                b: pl.Tensor[[32, 32], pl.FP32],
                out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    a_plus_b = pl.add(a, b)
                    sub = pl.sub(a, b)
                    result = pl.matmul(a_plus_b, sub)
                    out = pl.assemble(out, result, [0, 0])
                return out

        with passes.PassContext([], passes.VerificationLevel.NONE):
            transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
                NoSplitGroupProgram
            )

        aic_funcs = [func for func in transformed.functions.values() if func.func_type == pl.FunctionType.AIC]
        aiv_funcs = [func for func in transformed.functions.values() if func.func_type == pl.FunctionType.AIV]
        assert len(aic_funcs) == 1
        assert len(aiv_funcs) == 1
        assert aiv_funcs[0].attrs.get("dual_aiv_dispatch") is True

        orch_result = _generate_orch_result(transformed)
        code = orch_result.code
        expected_ids = (
            orch_result.func_name_to_id[aic_funcs[0].name],
            orch_result.func_name_to_id[aiv_funcs[0].name],
            orch_result.func_name_to_id[aiv_funcs[0].name],
        )

        assert f"MixedKernels mixed_0 = {{{expected_ids[0]}, {expected_ids[1]}, {expected_ids[2]}}};" in code
        assert "pto2_rt_submit_task(mixed_0, params_t0);" in code

    def test_standalone_spmd_dispatches_group_with_spmd_launch_spec(self):
        """Standalone Spmd should remain a wrapper and carry launch spec into Group dispatch."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class SpmdMixedProgram:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_mm = pl.matmul(tile_a_l0a, tile_b_l0b)
                tile_bias = pl.load(bias, [0, 0], [64, 64])
                tile_out = pl.add(tile_mm, tile_bias)
                out = pl.store(tile_out, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.spmd(4, sync_start=True):
                    out = self.kernel(a, b, bias, out)
                return out

        with passes.PassContext([], passes.VerificationLevel.NONE):
            transformed = passes.expand_mixed_kernel()(
                passes.infer_tile_memory_space()(
                    passes.outline_cluster_scopes()(passes.convert_to_ssa()(SpmdMixedProgram))
                )
            )
        spmd_func = transformed.get_function("main_spmd_0")
        group_func = transformed.get_function("kernel")
        assert spmd_func is not None
        assert group_func is not None
        assert spmd_func.func_type == pl.FunctionType.Spmd
        assert group_func.func_type == pl.FunctionType.Group

        code = _generate_orch_code(transformed)

        assert "MixedKernels mixed_0" in code
        assert "pto2_rt_submit_task(mixed_0, params_t0);" in code
        assert "params_t0.launch_spec.set_block_num(4);" in code
        assert "params_t0.launch_spec.set_require_sync_start(true);" in code


class TestUnregisteredOpError:
    """Test that unregistered/misplaced ops in Orchestration functions raise errors."""

    def test_unregistered_tensor_op_raises_error(self):
        """Unregistered tensor op (tensor.full) in Orchestration must raise RuntimeError."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ib = IRBuilder()
        with ib.function("orch", type=ir.FunctionType.Orchestration) as orch_f:
            orch_f.param("x", ir.TensorType([16, 16], pl.FP32))
            orch_f.return_type(ir.TensorType([16, 16], pl.FP32))
            filled = ib.let("filled", tensor_ops.full([16, 16], pl.FP32, 0.0))
            ib.return_stmt(filled)
        orch_func = orch_f.get_result()

        program = ir.Program([orch_func], "test_prog", ir.Span.unknown())

        with pytest.raises(RuntimeError, match="Misplaced tensor op.*tensor.full"):
            codegen.generate_orchestration(program, orch_func)


class TestLocalAllocWAWPromotion:
    """Test that locally allocated tensors get add_inout instead of add_output.

    Issue #1022: when a tensor is pre-allocated via alloc_tensors and then
    passed as Out to multiple InCore tasks in separate loops, the codegen
    must use add_inout (not add_output) to establish WAW dependencies.

    The promotion is now performed by the ``DeriveCallDirections`` IR pass,
    which writes ``ArgDirection::InOut`` into ``Call.attrs['arg_directions']`` for
    locally allocated buffers (replacing the legacy ``CallSiteDirectionResolver``
    analysis that lived in orchestration codegen).
    """

    def test_alloc_tensor_two_loops_gets_inout(self):
        """Two loops writing to the same alloc_tensors buffer must use add_inout."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TwoLoopAllocProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def task_init(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                buf: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], buf)
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def task_compute(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                buf: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], buf)
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def task_read(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                buf: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                for _i in pl.range(4):
                    buf = self.task_init(x, buf)
                for _i in pl.range(4):
                    buf = self.task_compute(x, buf)
                out = self.task_read(buf, out)
                return out

        code = _generate_orch_code(TwoLoopAllocProgram)

        assert "add_inout(buf)" in code, (
            "Locally allocated tensor 'buf' passed as Out must generate "
            "add_inout (not add_output) to establish WAW dependencies. "
            f"Generated code:\n{code}"
        )
        assert "add_output(buf)" not in code, (
            f"Locally allocated tensor 'buf' must NOT use add_output. Generated code:\n{code}"
        )

    def test_external_tensor_keeps_add_output(self):
        """Function parameter tensors with Out direction keep add_output."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ExternalOutProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out = self.kernel(a, out)
                return out

        code = _generate_orch_code(ExternalOutProgram)

        assert "add_output(ext_out)" in code, (
            f"External (parameter) tensor should keep add_output. Generated code:\n{code}"
        )

    def test_parallel_loop_local_buf_keeps_add_output(self):
        """Issue #1086: a single ``pl.parallel`` writer of a local buffer must
        emit ``add_output`` (not ``add_inout``).

        Promoting Out → InOut here injects a spurious WAW dependency that
        forces the runtime to serialize otherwise independent iterations of
        the parallel loop, causing the regression observed in Qwen3 decode.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class SingleParallelProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def task(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                buf: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                for _i in pl.parallel(4):
                    buf = self.task(a, buf)
                out = self.task(buf, out)
                return out

        code = _generate_orch_code(SingleParallelProgram)

        assert "add_output(buf)" in code, (
            f"Local buf written from a single pl.parallel loop must use add_output, "
            f"not add_inout (issue #1086). Generated code:\n{code}"
        )
        assert "add_inout(buf)" not in code, (
            f"Local buf must not be promoted to add_inout when only a single "
            f"pl.parallel loop writes it. Generated code:\n{code}"
        )

    def test_two_parallel_loops_promote_only_second(self):
        """Two consecutive ``pl.parallel`` loops writing the same local buffer.

        The first loop is the only writer-unit at its scope and stays
        ``add_output``; the second loop hits R-prior so it is promoted to
        ``add_inout`` to keep the cross-loop WAW dependency.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TwoParallelProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def task(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                buf: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                for _i in pl.parallel(4):
                    buf = self.task(a, buf)
                for _j in pl.parallel(4):
                    buf = self.task(a, buf)
                out = self.task(buf, out)
                return out

        code = _generate_orch_code(TwoParallelProgram)

        # Both add_output (first loop, R-prior not yet active) and add_inout
        # (second loop, R-prior fires) must be present for the same `buf`.
        assert "add_output(buf)" in code, (
            f"First pl.parallel writer of buf should remain add_output. Generated code:\n{code}"
        )
        assert "add_inout(buf)" in code, (
            f"Second pl.parallel writer of buf should be promoted to add_inout via R-prior. "
            f"Generated code:\n{code}"
        )


class TestArgDirectionsCodegen:
    """Verify that orchestration codegen prefers Call.attrs['arg_directions'] when present.

    These tests exercise the new ArgDirection-driven path in BuildTaskParams:
    every recognised ArgDirection enum value is mapped to the matching runtime
    method (add_input / add_output / add_inout / add_no_dep / add_scalar) and
    the value emitted at the call site reflects the per-argument direction
    written by the DeriveCallDirections pass — independently of the callee's
    ParamDirection.
    """

    @staticmethod
    def _generate_orch_direct(program) -> str:
        """Bypass ``_ensure_arg_directions`` so explicit overrides survive."""
        for func in program.functions.values():
            if func.func_type == ir.FunctionType.Orchestration:
                return codegen.generate_orchestration(program, func).code
        raise ValueError("No orchestration function found in program")

    def _build_program_with_arg_directions(self, arg_dirs):
        """Build a tiny Orchestration program where the call site has explicit arg_directions.

        The callee declares ``Out`` for the second parameter, and the orchestration
        body pre-allocates the tensor with ``tensor.create``. We then patch the
        call expression with the requested ``arg_directions`` so that codegen
        consumes them directly (bypassing the legacy ParamDirection mapping).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ArgDirProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                buf: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                buf = self.kernel(a, buf)
                out = self.kernel(buf, out)
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        program = pm.run_passes(ArgDirProgram)

        rewritten = self._rewrite_kernel_calls(program, arg_dirs)
        return rewritten

    @staticmethod
    def _rewrite_kernel_calls(program, arg_dirs):
        """Replace every ``self.kernel(...)`` Call with a copy carrying the given arg_directions."""

        class _RewriteKernel(ir.IRMutator):
            def visit_call(self, op: ir.Call) -> ir.Expr:
                expr = super().visit_call(op)
                call = expr if isinstance(expr, ir.Call) else op
                if call.op.name != "kernel" or len(call.args) != len(arg_dirs):
                    return expr
                attrs = {"arg_directions": list(arg_dirs)}
                return ir.Call(call.op, list(call.args), dict(call.kwargs), attrs, call.type, call.span)

        return _RewriteKernel().visit_program(program)

    def test_arg_direction_inout_emits_add_inout(self):
        program = self._build_program_with_arg_directions([ir.ArgDirection.Input, ir.ArgDirection.InOut])
        code = self._generate_orch_direct(program)
        assert "add_inout(buf)" in code, (
            f"ArgDirection::InOut on the second argument must produce add_inout(...). Generated code:\n{code}"
        )

    def test_arg_direction_output_existing_emits_add_output(self):
        program = self._build_program_with_arg_directions(
            [ir.ArgDirection.Input, ir.ArgDirection.OutputExisting]
        )
        code = self._generate_orch_direct(program)
        assert "add_output(" in code, (
            f"ArgDirection::OutputExisting must produce add_output(...). Generated code:\n{code}"
        )
        assert "add_inout(" not in code or code.count("add_inout(") < code.count("add_output("), (
            f"Expected add_output to dominate over add_inout. Generated code:\n{code}"
        )

    def test_arg_direction_no_dep_emits_add_no_dep(self):
        program = self._build_program_with_arg_directions([ir.ArgDirection.Input, ir.ArgDirection.NoDep])
        code = self._generate_orch_direct(program)
        assert "add_no_dep(" in code, (
            f"ArgDirection::NoDep must produce add_no_dep(...). Generated code:\n{code}"
        )

    def test_arg_direction_input_emits_add_input(self):
        program = self._build_program_with_arg_directions([ir.ArgDirection.Input, ir.ArgDirection.Input])
        code = self._generate_orch_direct(program)
        assert "add_input(" in code, (
            f"ArgDirection::Input must produce add_input(...). Generated code:\n{code}"
        )
        assert "add_output(" not in code and "add_inout(" not in code, (
            "When all tensor args are ArgDirection::Input the codegen must not emit add_output/add_inout. "
            f"Generated code:\n{code}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

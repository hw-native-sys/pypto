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


def _materialize_scopes(program):
    """Codegen requires explicit RuntimeScopeStmt wrappers (PTO2_SCOPE blocks).

    Tests that hand-build IR (without going through PassManager) need to invoke
    MaterializeRuntimeScopes before codegen so the orchestration function body
    and for/if bodies carry explicit AUTO RuntimeScopeStmt nodes. Codegen no
    longer emits implicit PTO2_SCOPE() wrappers. This is a no-op when the program
    was already produced by the pass pipeline. Must run after DeriveCallDirections
    (a declared requirement of the pass).
    """
    return passes.materialize_runtime_scopes()(program)


def _generate_orch_code(program) -> str:
    """Generate orchestration code using backend-agnostic codegen."""
    program = _materialize_scopes(_ensure_arg_directions(program))
    for func in program.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            result = codegen.generate_orchestration(program, func)
            return result.code
    raise ValueError("No orchestration function found in program")


def _generate_orch_result(program) -> "codegen.OrchestrationResult":
    """Generate orchestration result using backend-agnostic codegen."""
    program = _materialize_scopes(_ensure_arg_directions(program))
    for func in program.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            return codegen.generate_orchestration(program, func)
    raise ValueError("No orchestration function found in program")


class TestOrchestration:
    """Test orchestration codegen format."""

    def test_basic_structure(self):
        """Test codegen produces PTO2 format: make_tensor_external, Arg, rt_submit_aiv_task."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class BasicProgram:
            @pl.function(type=pl.FunctionType.AIV)
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
                    rt_submit_aiv_task(0, params_t0);

                    // Task 1: kernel_add
                    Arg params_t1;
                    params_t1.add_input(c);
                    params_t1.add_input(ext_b);
                    params_t1.add_output(ext_d);
                    rt_submit_aiv_task(0, params_t1);
                }
            }

            }  // extern "C"
        """
        assert_code_equal(code, expected)

    def test_tensor_read(self):
        """Test tensor.read emits get_tensor_data<T>() so the runtime spin-waits
        on the producer task before reading (no raw host buffer deref)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TensorReadProgram:
            @pl.function(type=pl.FunctionType.AIV)
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

        # tensor.read emits a typed get_tensor_data<T>() call (the runtime
        # spin-waits on TensorMap producers before reading), and packs the
        # multi-dim indices into a uint32_t indices_<var>[N] = {...} array.
        # ConstInt indices are emitted bare via EmitAsUint32 (no redundant cast).
        assert "uint32_t indices_val[2] = {1, 3};" in code
        assert "float val = get_tensor_data<float>(ext_t, 2, indices_val);" in code
        # The old raw-deref path must not return.
        assert "data_as<void>" not in code
        assert "host_t" not in code
        assert "buffer.addr" not in code

    def test_orch_internal_tensor_read_uses_get_tensor_data(self):
        """Regression for #1487: an orch-level read of an internally-allocated
        tensor (produced by a device-scope task) must go through
        ``get_tensor_data<T>()`` so the runtime spin-waits on the producer's
        TensorMap entry. A raw ``buffer.addr`` deref returns stale/zero data
        before the producer has finished writing.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class InternalReadProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel_copy(
                self,
                src: pl.Tensor[[8, 1], pl.INT32],
                output: pl.Out[pl.Tensor[[8, 1], pl.INT32]],
            ) -> pl.Tensor[[8, 1], pl.INT32]:
                t: pl.Tile[[8, 1], pl.INT32] = pl.load(src, [0, 0], [8, 1])
                out: pl.Tensor[[8, 1], pl.INT32] = pl.store(t, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_internal_read(
                self,
                src_count: pl.Tensor[[8, 1], pl.INT32],
            ) -> pl.Tensor[[8, 1], pl.INT32]:
                cnt: pl.Tensor[[8, 1], pl.INT32] = pl.create_tensor([8, 1], dtype=pl.INT32)
                cnt = self.kernel_copy(src_count, cnt)
                n_rows: pl.Scalar[pl.INT32] = pl.tensor.read(cnt, [0, 0])  # noqa: F841
                return cnt

        code = _generate_orch_code(InternalReadProgram)

        # The runtime API call is what gives us producer-sync; it must be present.
        assert "get_tensor_data<int32_t>(cnt" in code
        # The pre-fix raw-deref shapes must not return — including the
        # buffer.addr / reinterpret_cast path from the dead #1479 attempt.
        assert "buffer.addr" not in code
        assert "reinterpret_cast<void*>(static_cast<uintptr_t>" not in code
        assert "static_cast<int32_t*>(reinterpret_cast" not in code

    def test_config_file(self):
        """Test orchestration result contains kernel function metadata."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ConfigProgram:
            @pl.function(type=pl.FunctionType.AIV)
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
            @pl.function(type=pl.FunctionType.AIV)
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
        assert code.count("rt_submit_aiv_task") == 2

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
            @pl.function(type=pl.FunctionType.AIV)
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

            @pl.function(type=pl.FunctionType.AIV)
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

            @pl.function(type=pl.FunctionType.AIV)
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
                    rt_submit_aiv_task(0, params_t0);

                    // Task 1: kernel_add_scalar
                    Arg params_t1;
                    params_t1.add_input(c);
                    params_t1.add_output(d);
                    params_t1.add_scalar(to_u64(1.000000f));
                    rt_submit_aiv_task(1, params_t1);

                    // Task 2: kernel_add_scalar
                    Arg params_t2;
                    params_t2.add_input(c);
                    params_t2.add_output(e);
                    params_t2.add_scalar(to_u64(2.000000f));
                    rt_submit_aiv_task(1, params_t2);

                    // Task 3: kernel_mul
                    Arg params_t3;
                    params_t3.add_input(d);
                    params_t3.add_input(e);
                    params_t3.add_output(g);
                    rt_submit_aiv_task(2, params_t3);

                    // Task 4: kernel_add
                    Arg params_t4;
                    params_t4.add_input(g);
                    params_t4.add_input(c);
                    params_t4.add_output(ext_f);
                    rt_submit_aiv_task(0, params_t4);
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
            @pl.function(type=pl.FunctionType.AIV)
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

            @pl.function(type=pl.FunctionType.AIV)
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
        assert code.count("rt_submit_aiv_task") == 2

        # PTO2_SCOPE wraps all task submissions
        assert "PTO2_SCOPE" in code

    def test_tuple_output(self):
        """Test tuple return as final output: all elements are external tensors."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TupleOutputProgram:
            @pl.function(type=pl.FunctionType.AIV)
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
        assert code.count("rt_submit_aiv_task") == 1

        # PTO2_SCOPE wraps all task submissions
        assert "PTO2_SCOPE" in code

    def test_four_element_tuple(self):
        """Test 4-element tuple unpacking with mixed shapes as intermediate."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class FourTupleProgram:
            @pl.function(type=pl.FunctionType.AIV)
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

            @pl.function(type=pl.FunctionType.AIV)
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
        assert code.count("rt_submit_aiv_task") == 2

        # online_update: 3 In + 3 InOut + 1 Out = 7 params
        assert "params_t0.add_input(ext_mij)" in code
        assert "params_t0.add_inout(ext_mi_in)" in code
        assert "params_t0.add_output(ext_dst_in)" in code

        # kernel_add: 2 In + 1 Out = 3 params
        assert "params_t1.add_input(ext_oi_in)" in code
        assert "params_t1.add_output(ext_final)" in code

        # PTO2_SCOPE wraps all task submissions
        assert "PTO2_SCOPE" in code

    def test_inout_not_returned_three_outputs_alias(self):
        """Regression for #1573: 3+ tuple outputs + an InOut that is not returned.

        ``kernel`` takes ``inout_t`` (InOut, written in place but NOT part of the
        return tuple) followed by three ``Out`` params that ARE returned. The
        legacy tail-alignment heuristic put ``inout_t`` in the Out/InOut index
        list, so ``tuple_arity (3) < out_indices (4)`` shifted every result alias
        by one: each tuple element bound to the wrong source tensor
        (``o1 = ext_inout_t``, ``o2 = ext_ta``, ``o3 = ext_tb``). Downstream that
        feeds a reshape/consumer the wrong tensor (AICPU ``valid_reshape`` assert
        / scheduler timeout). Each result must alias to its own arg, recovered
        precisely from the callee's ReturnStmt.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class InOutNotReturnedProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                inout_t: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
                out_a: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                out_c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[16, 16], pl.FP32],
                pl.Tensor[[16, 16], pl.FP32],
                pl.Tensor[[16, 16], pl.FP32],
            ]:
                it: pl.Tile[[16, 16], pl.FP32] = pl.load(inout_t, [0, 0], [16, 16])
                _io: pl.Tensor[[16, 16], pl.FP32] = pl.store(it, [0, 0], inout_t)
                a_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(it, [0, 0], out_a)
                b_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(it, [0, 0], out_b)
                c_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(it, [0, 0], out_c)
                return a_out, b_out, c_out

            @pl.function(type=pl.FunctionType.AIV)
            def combine(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                c: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                at: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                r: pl.Tensor[[16, 16], pl.FP32] = pl.store(at, [0, 0], out)
                return r

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                inout_t: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
                ta: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                tb: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                tc: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                final: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                o1, o2, o3 = self.kernel(inout_t, ta, tb, tc)
                final = self.combine(o1, o2, o3, final)
                return final

        code = _generate_orch_code(InOutNotReturnedProgram)

        # inout_t is InOut (written in place) but not part of the return tuple.
        assert "params_t0.add_inout(ext_inout_t)" in code

        # Each tuple result aliases to its OWN arg — not shifted onto inout_t.
        assert "const Tensor& o1 = ext_ta;" in code
        assert "const Tensor& o2 = ext_tb;" in code
        assert "const Tensor& o3 = ext_tc;" in code
        # The scrambled (shifted-by-one) bindings must NOT appear.
        assert "const Tensor& o1 = ext_inout_t;" not in code
        assert "const Tensor& o2 = ext_ta;" not in code
        assert "const Tensor& o3 = ext_tb;" not in code

    def test_scalar_tuple_return_materialized(self):
        """Regression for #631: a kernel returning ``tuple[Tensor, Scalar]``.

        ``producer`` returns a Tensor (writeback to its Out param) AND a Scalar
        ``off = base * 32`` that is *not* backed by any Out param. The orchestration
        unpacks both and forwards the Scalar to ``consumer``. The Scalar tuple
        element has no runtime output slot to alias, so codegen used to drop it,
        emitting code that referenced an undefined ``off`` (or, after a stop-gap,
        a wrong ``off = 0``). The fix re-materializes the Scalar by inlining the
        callee's defining expression with the callee param ``base`` substituted by
        the call-site arg ``base`` -> ``int64_t off = (base * 32);``.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ScalarTupleReturnProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def producer(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                base: pl.Scalar[pl.INDEX],
                out_t: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Scalar[pl.INDEX]]:
                at: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                rt: pl.Tensor[[16, 16], pl.FP32] = pl.store(at, [0, 0], out_t)
                off: pl.Scalar[pl.INDEX] = base * 32
                return rt, off

            @pl.function(type=pl.FunctionType.AIV)
            def consumer(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                off: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                at: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                r: pl.Tensor[[16, 16], pl.FP32] = pl.store(at, [0, 0], out)
                return r

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_scalar_tuple(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                base: pl.Scalar[pl.INDEX],
                mid: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                final: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t, off = self.producer(a, base, mid)
                final = self.consumer(t, off, final)
                return final

        code = _generate_orch_code(ScalarTupleReturnProgram)

        # The Scalar tuple element is re-materialized from the callee's defining
        # expression with the param substituted by the call-site arg.
        assert "int64_t off = (base * 32);" in code
        # The stop-gap zero default must NOT be emitted for a traceable scalar.
        assert "int64_t off = 0;" not in code
        # The Tensor tuple element still aliases its own Out arg.
        assert "const Tensor& t = ext_mid;" in code
        # The downstream kernel receives the materialized scalar.
        assert "add_scalar(off)" in code

    def test_scalar_tuple_return_materialized_submit(self):
        """Regression for #631: the ``pl.submit`` return-alias path.

        Companion to ``test_scalar_tuple_return_materialized``. The #631 fix
        re-materializes Scalar tuple-return elements in *both* the plain
        tuple-return path (``GenerateTupleReturnAliases``) and the ``pl.submit``
        path (``GenerateSubmitReturnAliases``). Here ``producer`` is launched via
        ``pl.submit`` inside a ``pl.manual_scope``: its declared
        ``tuple[Tensor, Scalar]`` return is augmented with a trailing TaskId, so
        the unpack is ``t, off, _tid = pl.submit(...)``. The Scalar ``off`` has no
        runtime output slot, so the submit path must inline the callee's
        ``off = base * 32`` with the call-site arg ``base`` -> ``int64_t off =
        (base * 32);`` exactly as the plain path does.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ScalarTupleReturnSubmitProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def producer(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                base: pl.Scalar[pl.INDEX],
                out_t: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Scalar[pl.INDEX]]:
                at: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                rt: pl.Tensor[[16, 16], pl.FP32] = pl.store(at, [0, 0], out_t)
                off: pl.Scalar[pl.INDEX] = base * 32
                return rt, off

            @pl.function(type=pl.FunctionType.InCore)
            def consumer(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                off: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                at: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                r: pl.Tensor[[16, 16], pl.FP32] = pl.store(at, [0, 0], out)
                return r

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_scalar_tuple_submit(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                base: pl.Scalar[pl.INDEX],
                mid: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                final: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                with pl.manual_scope():
                    t, off, _tid = pl.submit(self.producer, a, base, mid)
                    final, _ctid = pl.submit(self.consumer, t, off, final)
                return final

        code = _generate_orch_code(ScalarTupleReturnSubmitProgram)

        # The submit path re-materializes the Scalar tuple element from the
        # callee's defining expression with the param substituted by the arg.
        assert "int64_t off = (base * 32);" in code
        # The stop-gap zero default must NOT be emitted for a traceable scalar.
        assert "int64_t off = 0;" not in code
        # The downstream kernel receives the materialized scalar.
        assert "add_scalar(off)" in code

    def test_tensor_create(self):
        """Test tensor.create generates TensorCreateInfo with shape/dtype."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TensorCreateProgram:
            @pl.function(type=pl.FunctionType.AIV)
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

    def test_tensor_create_with_manual_dep(self):
        """``pl.create_tensor(..., manual_dep=True)`` opts a tensor out of OverlapMap
        auto-dep tracking for its entire lifetime. Codegen forwards the flag to the
        ``TensorCreateInfo`` ctor's trailing ``manual_dep`` argument.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ManualDepProgram:
            @pl.function(type=pl.FunctionType.AIV)
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
                scratch: pl.Tensor[[32, 32], pl.FP16] = pl.create_tensor(
                    [32, 32], dtype=pl.FP16, manual_dep=True
                )
                scratch = self.kernel_fill(a, scratch)
                result = self.kernel_fill(scratch, result)
                return result

        code = _generate_orch_code(ManualDepProgram)

        # The trailing /*manual_dep=*/true on TensorCreateInfo is the codegen hook
        # the runtime reads to skip OverlapMap insert/lookup for this tensor.
        assert (
            "TensorCreateInfo scratch_ci(scratch_ci_shapes, 2, DataType::FLOAT16, /*manual_dep=*/true)"
            in code
        )

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
            @pl.function(type=pl.FunctionType.AIV)
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
                    rt_submit_aiv_task(0, params_t0);
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
            @pl.function(type=pl.FunctionType.AIV)
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
            @pl.function(type=pl.FunctionType.AIV)
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

        # tensor.slice generates array variables and runtime .view() call with dynamic offset.
        # Shape dims are clamped to the source extent (offset already emitted above) so the
        # strided runtime never sees an over-extent view. The clamp guards the unsigned
        # subtraction with a ternary so an offset past the source extent saturates to 0u.
        assert (
            "uint32_t chunk_shapes[2] = {"
            "(chunk_offsets[0] >= ext_data.shapes[0] ? 0u : std::min<uint32_t>(16, ext_data.shapes[0] - chunk_offsets[0])), "
            "(chunk_offsets[1] >= ext_data.shapes[1] ? 0u : std::min<uint32_t>(16, ext_data.shapes[1] - chunk_offsets[1]))};"
            in code
        )
        assert "uint32_t chunk_offsets[2] = {static_cast<uint32_t>((i * 16)), 0};" in code
        assert "Tensor chunk = ext_data.view(chunk_shapes, chunk_offsets);" in code

        # tensor.read now goes through get_tensor_data<T>() (producer-sync
        # via TensorMap) instead of a raw orch_args.tensor().data_as<void>()
        # deref. See #1487.
        assert "uint32_t indices_n_blocks[1] = {0};" in code
        assert "int64_t n_blocks = get_tensor_data<int64_t>(ext_config, 1, indices_n_blocks);" in code

        # kernel_add task submitted inside loop
        assert "rt_submit_aiv_task" in code

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

        assert (
            "uint32_t chunk_shapes[2] = {"
            "(chunk_offsets[0] >= ext_data.shapes[0] ? 0u : std::min<uint32_t>(16, ext_data.shapes[0] - chunk_offsets[0])), "
            "(chunk_offsets[1] >= ext_data.shapes[1] ? 0u : std::min<uint32_t>(16, ext_data.shapes[1] - chunk_offsets[1]))};"
            in code
        )
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

        # slice still emits view on the external tensor (shape dims clamped to source extent).
        assert (
            "uint32_t chunk_shapes[3] = {"
            "(chunk_offsets[0] >= ext_data.shapes[0] ? 0u : std::min<uint32_t>(1, ext_data.shapes[0] - chunk_offsets[0])), "
            "(chunk_offsets[1] >= ext_data.shapes[1] ? 0u : std::min<uint32_t>(16, ext_data.shapes[1] - chunk_offsets[1])), "
            "(chunk_offsets[2] >= ext_data.shapes[2] ? 0u : std::min<uint32_t>(16, ext_data.shapes[2] - chunk_offsets[2]))};"
            in code
        )
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

    def test_tensor_as_layout_cross_flip_lowers_to_transpose(self):
        """Cross-layout flip (ND→DN) lowers to runtime Tensor::transpose on the
        trailing pair (shapes + strides swapped, start_offset preserved)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ib = IRBuilder()
        with ib.function("orch_as_layout", type=ir.FunctionType.Orchestration) as f:
            b = f.param("b", ir.TensorType([8, 4], DataType.FP16))
            f.return_type(ir.TensorType([4, 8], DataType.FP16))
            b_dn = ib.let("b_dn", tensor_ops.as_layout(b, ir.TensorLayout.DN))
            ib.return_stmt(b_dn)
        orch = f.get_result()
        program = ir.Program([orch], "test_as_layout_cross_flip", ir.Span.unknown())

        code = _generate_orch_code(program)

        # Cross-layout flip swaps the trailing pair via runtime Tensor::transpose
        # on the external tensor handle (start_offset preserved).
        assert "Tensor b_dn = ext_b.transpose(0, 1);" in code
        # The deleted pre-#808 fields must never be emitted.
        assert "raw_shapes" not in code
        assert "is_raw_eq_shapes" not in code

    def test_tensor_as_layout_identity_flip_aliases(self):
        """tensor.as_layout with target == source layout emits a plain alias, not a transpose."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ib = IRBuilder()
        with ib.function("orch_as_layout_id", type=ir.FunctionType.Orchestration) as f:
            b = f.param("b", ir.TensorType([8, 4], DataType.FP16))
            f.return_type(ir.TensorType([8, 4], DataType.FP16))
            b_same = ib.let("b_same", tensor_ops.as_layout(b, ir.TensorLayout.ND))
            ib.return_stmt(b_same)
        orch = f.get_result()
        program = ir.Program([orch], "test_as_layout_identity", ir.Span.unknown())

        code = _generate_orch_code(program)

        assert "Tensor b_same = ext_b;" in code
        assert ".transpose(" not in code

    def test_if_statement(self):
        """Test if/else codegen with conditional scalar values."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class IfProgram:
            @pl.function(type=pl.FunctionType.AIV)
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
            @pl.function(type=pl.FunctionType.AIV)
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

            @pl.function(type=pl.FunctionType.AIV)
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
            if "params_t0;" in line and "params_t0." not in line:
                in_task0 = True
            elif "params_t1;" in line and "params_t1." not in line:
                in_task1 = True
            elif "rt_submit" in line:
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
            @pl.function(type=pl.FunctionType.AIV)
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

            @pl.function(type=pl.FunctionType.AIV)
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
        assert code.count("rt_submit_aiv_task") == 2

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

        # Task params should use the inplace param, either directly or through
        # a window view of that param after OptimizeOrchTensors.
        assert "ext_output_tensor" in code
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
            @pl.function(type=pl.FunctionType.AIV)
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

        assert "rt_submit_" in code
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
            @pl.function(type=pl.FunctionType.AIV)
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
        """1D tensor [8], read(t, [3]) -> get_tensor_data<float>(ext_t, 1, indices_val)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(self, t: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                val: pl.Scalar[pl.FP32] = pl.tensor.read(t, [3])  # noqa: F841
                return t

        code = _generate_orch_code(Prog)
        assert "uint32_t indices_val[1] = {3};" in code
        assert "float val = get_tensor_data<float>(ext_t, 1, indices_val);" in code
        assert "data_as<void>" not in code
        assert "buffer.addr" not in code

    def test_tensor_read_constant_2d(self):
        """2D tensor [4, 8], read(t, [1, 3]) -> get_tensor_data<float>(ext_t, 2, indices_val)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(self, t: pl.Tensor[[4, 8], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                val: pl.Scalar[pl.FP32] = pl.tensor.read(t, [1, 3])  # noqa: F841
                return t

        code = _generate_orch_code(Prog)
        # Multi-dim indices are passed as a uint32_t[N] array — the runtime
        # computes the flat offset itself, so no `1 * 8 + 3` arithmetic appears.
        assert "uint32_t indices_val[2] = {1, 3};" in code
        assert "float val = get_tensor_data<float>(ext_t, 2, indices_val);" in code
        assert "data_as<void>" not in code

    def test_tensor_read_constant_3d(self):
        """3D tensor [2, 4, 8], read(t, [1, 2, 3]) -> get_tensor_data<float>(ext_t, 3, indices_val)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(self, t: pl.Tensor[[2, 4, 8], pl.FP32]) -> pl.Tensor[[2, 4, 8], pl.FP32]:
                val: pl.Scalar[pl.FP32] = pl.tensor.read(t, [1, 2, 3])  # noqa: F841
                return t

        code = _generate_orch_code(Prog)
        assert "uint32_t indices_val[3] = {1, 2, 3};" in code
        assert "float val = get_tensor_data<float>(ext_t, 3, indices_val);" in code
        assert "data_as<void>" not in code

    def test_tensor_read_variable_index(self):
        """2D tensor [4, 8], read(t, [i, j]) -> indices array carries the runtime expressions."""
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
        # Each read emits its own typed get_tensor_data<T> call.
        assert "int64_t row = get_tensor_data<int64_t>(ext_config, 1, indices_row);" in code
        assert "int64_t col = get_tensor_data<int64_t>(ext_config, 1, indices_col);" in code
        # The variable indices ride through unchanged inside static_cast<uint32_t>(...).
        assert "uint32_t indices_val[2] = {static_cast<uint32_t>(row), static_cast<uint32_t>(col)};" in code
        assert "float val = get_tensor_data<float>(ext_t, 2, indices_val);" in code
        assert "data_as<void>" not in code

    def test_tensor_write_constant_2d(self):
        """2D tensor [4, 8], write(t, [1, 3], val) -> set_tensor_data<float>(ext_t, 2, indices_t, val)."""
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
        # Read uses get_tensor_data<T>; write goes through the symmetric
        # set_tensor_data<T> API so the runtime can spin-wait on producers /
        # tracked INOUT consumers before writing.
        assert "float val = get_tensor_data<float>(ext_t, 2, indices_val);" in code
        assert "uint32_t indices_t[2] = {1, 3};" in code
        assert "set_tensor_data<float>(ext_t, 2, indices_t, val);" in code
        # Old raw-store form must not return.
        assert "data_as<void>" not in code
        assert "buffer.addr" not in code

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
        assert code.count("rt_submit_aiv_task") == 2

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

    def test_windowed_tuple_outputs_rebind_loop_carried_tensor_without_redeclaration(self):
        """OutWindowExternalizer tuple outputs must rebind loop-carried tensors instead of redeclaring them."""

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class WindowedTupleLoopCarryProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_proj(
                self,
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                ob_chunk: pl.Scalar[pl.INDEX],
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                for ob, (k_proj_iter, v_proj_iter) in pl.range(
                    ob_chunk, ob_chunk + 4, init_values=(k_proj, v_proj)
                ):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128]
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16] = pl.tile.load(wk, [0, kv0], [128, 64], [128, 64])
                    k_acc: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(tile_a, tile_wk)
                    k_proj_next: pl.Tensor[[16, 512], pl.FP32] = pl.tile.store(k_acc, [0, kv0], k_proj_iter)

                    tile_wv: pl.Tile[[128, 64], pl.BF16] = pl.tile.load(wv, [0, kv0], [128, 64], [128, 64])
                    v_acc: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(tile_a, tile_wv)
                    v_proj_next: pl.Tensor[[16, 512], pl.FP32] = pl.tile.store(v_acc, [0, kv0], v_proj_iter)
                    k_proj_rv, v_proj_rv = pl.yield_(k_proj_next, v_proj_next)
                return k_proj_rv, v_proj_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                for ob_chunk, (k_proj_iter, v_proj_iter) in pl.range(0, 8, 4, init_values=(k_proj, v_proj)):
                    result: tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]] = (
                        self.kv_proj(k_proj_iter, v_proj_iter, ob_chunk, normed_tile, wk, wv)
                    )
                    k_proj_next: pl.Tensor[[16, 512], pl.FP32] = result[0]
                    v_proj_next: pl.Tensor[[16, 512], pl.FP32] = result[1]
                    k_proj_rv, v_proj_rv = pl.yield_(k_proj_next, v_proj_next)
                return k_proj_rv, v_proj_rv

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(WindowedTupleLoopCarryProgram)
        code = _generate_orch_code(transformed)

        assert "kv_proj__windowed" in code, code

        declared_names = re.findall(
            r"^\s*(?:const\s+Tensor&|Tensor|PTO2TaskId|auto)\s+([A-Za-z_]\w*)\s*=",
            code,
            flags=re.MULTILINE,
        )
        duplicate_declarations = {name for name in declared_names if declared_names.count(name) > 1}
        assert not duplicate_declarations, (
            f"generated C++ redeclared names {sorted(duplicate_declarations)}:\n{code}"
        )

        mutable_tensor_names = set(re.findall(r"^\s*Tensor\s+([A-Za-z_]\w*)\s*=", code, flags=re.MULTILINE))
        const_alias_names = set(
            re.findall(r"^\s*const\s+Tensor&\s+([A-Za-z_]\w*)\s*=", code, flags=re.MULTILINE)
        )
        assert not (mutable_tensor_names & const_alias_names), code

        rv_carry_names = {
            name for name in mutable_tensor_names if name.endswith("_rv") or re.search(r"__rv(?:_|$)", name)
        }
        assert rv_carry_names, code
        assert any(
            re.search(rf"^\s*{re.escape(name)}\s*=\s*[^;]+;", code, flags=re.MULTILINE)
            for name in rv_carry_names
        ), code

    def test_windowed_writer_before_full_parent_reader_stays_unwindowed(self):
        """Issue #1444: window writes followed by full-parent reads must not be externalized.

        The unsafe codegen shape is:
            producer writes score_flat.view(...) with add_output/add_inout
            later consumer reads score_flat with add_input
            no explicit set_dependencies edge bridges view -> parent

        Until runtime/codegen has a generic root-aware dependency bridge,
        OutWindowExternalizer must keep this producer unwindowed so auto deps
        operate on the same parent Tensor object.
        """

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N, M, W = 64, 2048, 8

        @pl.program
        class WindowedWriteFullParentReadProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def produce(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                score: pl.Out[pl.Tensor[[N, M], pl.FP32]],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                tile: pl.Tile[[N, W], pl.FP32] = pl.tile.load(x, [0, col], [N, W], [N, W])
                ret: pl.Tensor[[N, M], pl.FP32] = pl.tile.store(tile, [0, col], score)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                score: pl.Tensor[[N, M], pl.FP32],
                probe: pl.Out[pl.Tensor[[N, M], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                tile: pl.Tile[[1, M], pl.FP32] = pl.tile.load(score, [row, 0], [1, M], [1, M])
                ret: pl.Tensor[[N, M], pl.FP32] = pl.tile.store(tile, [row, 0], probe)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                score: pl.Out[pl.Tensor[[N, M], pl.FP32]],
                probe: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                score_flat: pl.Tensor[[N, M], pl.FP32] = pl.reshape(score, [N, M])
                for c0, (score_iter,) in pl.range(0, M, W, init_values=(score_flat,)):
                    score_next: pl.Tensor[[N, M], pl.FP32] = self.produce(x, score_iter, c0)
                    score_rv = pl.yield_(score_next)
                for r, (probe_iter,) in pl.range(N, init_values=(probe,)):
                    probe_next: pl.Tensor[[N, M], pl.FP32] = self.consume(score_rv, probe_iter, r)
                    probe_rv = pl.yield_(probe_next)
                return probe_rv

        transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
            WindowedWriteFullParentReadProgram
        )
        code = _generate_orch_code(transformed)

        assert "produce__windowed" not in code, code
        assert "params_t0.add_inout(score_flat)" in code, code
        assert "params_t1.add_input(score_flat)" in code, code
        assert "score_flat.view(" not in code, code

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
        assert "rt_submit_task(mixed_0, params_t0);" in code

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
        assert "rt_submit_task(mixed_0, params_t0);" in code

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
        assert "rt_submit_task(mixed_0, params_t0);" in code
        assert "params_t0.launch_spec.set_block_num(4);" in code
        assert "params_t0.launch_spec.set_require_sync_start(true);" in code

    def test_spmd_mixed_multi_out_single_return_alias_targets_actual_return(self):
        """SPMD mixed kernel with multiple Out params + single return must alias the
        call-site result SSA to the Out parameter that the kernel actually returns,
        not the first Out (which would route downstream consumers into a scratch
        buffer).

        Regression for the multi-Out SPMD mixed-kernel orchestration codegen bug
        where ``GenerateSingleReturnAlias`` always picked ``out_indices[0]``: a
        downstream kernel reading the SPMD result would silently see the first
        Out's storage (e.g. a per-block scratch tensor) instead of the actual
        accumulator. The fix tracks ``ReturnStmt`` value lineage back through
        the callee body to the source Param and uses that index for the alias.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class SpmdMultiOutSingleReturnProgram:
            # Mixed kernel with multiple Out params: a scratch buffer (1st Out)
            # and the real result (2nd Out, the one that the kernel returns).
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                scratch: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_b_l1 = pl.load(bias, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_mm = pl.matmul(tile_a_l0a, tile_b_l0b)
                tile_bias = pl.load(bias, [0, 0], [64, 64])
                tile_out = pl.add(tile_mm, tile_bias)
                scratch = pl.store(tile_mm, [0, 0], scratch)
                out = pl.store(tile_out, [0, 0], out)
                return out

            # Downstream kernel consumes the SPMD result so the SSA alias is
            # forced into existence; if the bug regresses, this consumer reads
            # the scratch buffer instead of `out`.
            @pl.function(type=pl.FunctionType.InCore)
            def consumer(
                self,
                in_buf: pl.Tensor[[64, 64], pl.FP32],
                final: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile = pl.load(in_buf, [0, 0], [64, 64])
                final = pl.store(tile, [0, 0], final)
                return final

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                scratch: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                final: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.spmd(4):
                    out = self.kernel(a, bias, scratch, out)
                final = self.consumer(out, final)
                return final

        with passes.PassContext([], passes.VerificationLevel.NONE):
            transformed = passes.expand_mixed_kernel()(
                passes.infer_tile_memory_space()(
                    passes.outline_cluster_scopes()(passes.convert_to_ssa()(SpmdMultiOutSingleReturnProgram))
                )
            )

        code = _generate_orch_code(transformed)

        # The mixed SPMD dispatch and the downstream consumer must be present.
        assert "MixedKernels mixed_0" in code, f"Expected mixed-kernel dispatch:\n{code}"
        assert "params_t0.launch_spec.set_block_num(4);" in code

        # The downstream consumer must read from the kernel's actual return
        # value (``ext_out``), not from the scratch buffer (``ext_scratch``).
        # Pre-fix, the multi-Out aliasing bug made the SPMD result SSA point
        # at ``ext_scratch`` (the first Out), and the consumer's first input
        # was rewritten to read from scratch.
        consumer_input_lines = [line for line in code.splitlines() if "params_t1.add_input" in line]
        assert consumer_input_lines, f"Expected a consumer task reading the SPMD result, got:\n{code}"
        first_consumer_input = consumer_input_lines[0]
        assert "ext_out" in first_consumer_input, (
            "Downstream consumer of a multi-Out SPMD mixed kernel should read the "
            "returned Out param (ext_out). "
            f"Got: {first_consumer_input}\n\nFull code:\n{code}"
        )
        assert "ext_scratch" not in first_consumer_input, (
            "Downstream consumer is reading from the scratch buffer (multi-Out "
            f"aliasing bug):\n{first_consumer_input}\n\nFull code:\n{code}"
        )

        # If the codegen emits an explicit SSA alias for the SPMD result,
        # it must bind to ext_out and never to ext_scratch.
        out_alias_lines = [
            line for line in code.splitlines() if line.lstrip().startswith("const Tensor& out__")
        ]
        for line in out_alias_lines:
            assert "ext_out" in line and "ext_scratch" not in line, (
                f"SSA alias for the multi-Out SPMD result must bind to ext_out:\n{line}\n\nFull code:\n{code}"
            )

    def test_spmd_multi_assemble(self):
        """SPMD multi-output call with assemble should preserve both OutputExisting tuple aliases."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class SpmdMultiAssembleProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b0: pl.Tensor[[16, 16], pl.FP32],
                b1: pl.Tensor[[16, 16], pl.FP32],
                out0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                out1: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                tile_b0: pl.Tile[[16, 16], pl.FP32] = pl.load(b0, [0, 0], [16, 16])
                tile_b1: pl.Tile[[16, 16], pl.FP32] = pl.load(b1, [0, 0], [16, 16])
                acc0: pl.Tile[[16, 16], pl.FP32] = pl.matmul(tile_a, tile_b0)
                res0: pl.Tensor[[16, 16], pl.FP32] = pl.store(acc0, [0, 0], out0)
                acc1: pl.Tile[[16, 16], pl.FP32] = pl.matmul(tile_a, tile_b1)
                res1: pl.Tensor[[16, 16], pl.FP32] = pl.store(acc1, [0, 0], out1)
                return res0, res1

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b0: pl.Tensor[[16, 16], pl.FP32],
                b1: pl.Tensor[[16, 16], pl.FP32],
                out0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                out1: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                with pl.spmd(4):
                    out0, out1 = self.kernel(a, b0, b1, out0, out1)
                return out0, out1

        with passes.PassContext([], passes.VerificationLevel.NONE):
            transformed = passes.expand_mixed_kernel()(
                passes.infer_tile_memory_space()(
                    passes.outline_cluster_scopes()(passes.convert_to_ssa()(SpmdMultiAssembleProgram))
                )
            )
        code = _generate_orch_code(transformed)

        assert "add_output(ext_out0)" in code and "add_output(ext_out1)" in code, (
            f"SPMD tuple outputs must remain OutputExisting at call site. Generated code:\n{code}"
        )

    def test_spmd_gm_pipe_buffer_tensor_create_scales_with_core_num(self):
        """SPMD gm_pipe_buffer allocation should scale by launch core_num in orchestration codegen."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class SpmdGMPipeProgram:
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
                with pl.spmd(4):
                    out = self.kernel(a, b, bias, out)
                return out

        with passes.PassContext([], passes.VerificationLevel.NONE):
            transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(SpmdGMPipeProgram)

        code = _generate_orch_code(transformed)
        assert "params_t0.launch_spec.set_block_num(4);" in code
        assert re.search(
            r"gm_pipe_buffer_\d+_ci_shapes\[1\]\s*=\s*\{static_cast<uint32_t>\(\(\d+\) \* \(4\)\)\};",
            code,
        ), f"Expected gm_pipe_buffer tensor.create shape to scale by core_num. Generated code:\n{code}"

    def test_gm_pipe_buffer_tensor_create_uses_callee_workspace(self):
        """Each injected gm_pipe_buffer tensor.create is sized from its callee pipe layout."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class PerCalleeGMPipeProgram:
            @pl.function(type=pl.FunctionType.AIC)
            def small_cube(self):
                buf = pl.reserve_buffer(name="small_v2c_slot_buffer", size=4096, base=pl.AUTO)
                pl.aic_initialize_pipe(pl.const(0, pl.INT32), buf, dir_mask=2, slot_size=512)

            @pl.function(type=pl.FunctionType.AIV)
            def small_vector(self):
                peer = pl.import_peer_buffer(name="small_v2c_slot_buffer", peer_func="small_cube")
                pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size=512)

            @pl.function(type=pl.FunctionType.Group)
            def small_group(self):
                self.small_cube()
                self.small_vector()

            @pl.function(type=pl.FunctionType.AIC)
            def large_cube(self):
                buf0 = pl.reserve_buffer(name="large_v2c_slot_buffer_0", size=8192, base=pl.AUTO)
                buf1 = pl.reserve_buffer(name="large_v2c_slot_buffer_1", size=16384, base=pl.AUTO)
                pl.aic_initialize_pipe(pl.const(0, pl.INT32), buf0, dir_mask=2, slot_size=1024, id=0)
                pl.aic_initialize_pipe(pl.const(0, pl.INT32), buf1, dir_mask=2, slot_size=2048, id=1)

            @pl.function(type=pl.FunctionType.AIV)
            def large_vector(self):
                peer0 = pl.import_peer_buffer(name="large_v2c_slot_buffer_0", peer_func="large_cube")
                peer1 = pl.import_peer_buffer(name="large_v2c_slot_buffer_1", peer_func="large_cube")
                pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer0, dir_mask=2, slot_size=1024, id=0)
                pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer1, dir_mask=2, slot_size=2048, id=1)

            @pl.function(type=pl.FunctionType.Group)
            def large_group(self):
                self.large_cube()
                self.large_vector()

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self):
                self.small_group()
                self.large_group()

        with passes.PassContext([], passes.VerificationLevel.NONE):
            transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
                PerCalleeGMPipeProgram
            )

        code = _generate_orch_code(transformed)
        shape_values = re.findall(r"gm_pipe_buffer_\d+_ci_shapes\[1\]\s*=\s*\{(\d+)\};", code)
        assert shape_values == ["1024", "6144"], (
            "Expected per-callee GM workspace shapes (small=512*8*1 side / f32, "
            f"large=(1024*8+2048*8) / f32), got {shape_values}. Generated code:\n{code}"
        )


class TestTaskIsValidCodegen:
    """``system.task_is_valid`` lowers to ``<expr>.is_valid()`` in C++.

    The op guards each per-slot fill of a manual_scope array-carry TaskId
    into the ``set_dependencies`` stack array.
    Codegen is hand-tested here on minimal IR rather than waiting for the
    end-to-end pass, so the emitter contract is pinned independently of the
    pass implementation.
    """

    def test_task_is_valid_emits_dot_is_valid(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ib = IRBuilder()
        with ib.function("orch", type=ir.FunctionType.Orchestration) as orch_f:
            tid = orch_f.param("tid", ir.ScalarType(DataType.TASK_ID))
            orch_f.return_type(ir.ScalarType(DataType.BOOL))
            # b = task_is_valid(tid)
            check = ir.create_op_call("system.task_is_valid", [tid], {}, ir.Span.unknown())
            b = ib.let("b", check)
            ib.return_stmt(b)
        orch_func = orch_f.get_result()
        program = ir.Program([orch_func], "test_task_is_valid", ir.Span.unknown())

        code = codegen.generate_orchestration(program, orch_func).code
        assert "bool b = tid.is_valid();" in code, code


class TestTupleLineagePointerKeying:
    """Tuple return-alias lineage must be keyed by Var identity, not name_hint.

    Regression for issue #1463: after inlining + OutWindowExternalizer, two
    distinct tuple-producing assignments can share a ``name_hint`` (e.g. several
    rebuilt ``ret__tmp_v0`` MakeTuples). When the orchestration codegen keyed its
    tuple lineage maps by ``name_hint``, the colliding tuples' TupleGetItem
    consumers were cross-wired: the emit names of one tuple's elements were
    propagated onto the other tuple's consumers. In the DeepSeek-V4 KV compressor
    this made the ``kv_state`` / ``score_state`` return aliases reuse the
    externalized ``kv_cache`` / ``kv`` window reshape names, so the generated
    orchestration C++ declared those names twice (``Tensor X = ...`` then
    ``const Tensor& X = ...``) and failed to compile with ``conflicting
    declaration``.
    """

    def test_same_name_tuple_vars_do_not_cross_wire_return_aliases(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        t2d = ir.TensorType([16, 16], pl.FP32)
        tflat = ir.TensorType([256, 1], pl.FP32)
        span = ir.Span.unknown()

        ib = IRBuilder()
        with ib.function("orch", type=ir.FunctionType.Orchestration) as orch_f:
            o0 = orch_f.param("o0", t2d)
            o1 = orch_f.param("o1", t2d)
            orch_f.return_type(ir.TupleType([t2d, t2d]))

            # Two tuple-producing MakeTuple assignments that deliberately share
            # the SAME name_hint "ret" (distinct Var objects) — exactly what the
            # OutWindowExternalizer rebuild produces after inlining. Each tuple
            # wraps a distinct reshape local (rsh0 / rsh1); its TupleGetItem
            # consumer is then reshaped again, so the consumer's lineage must
            # resolve to its OWN tuple's element. Old name_hint keying collapsed
            # both "ret" tuples onto one key, so the first tuple's consumer (a0)
            # lost its lineage and was emitted as the undeclared ``a0.reshape``.
            rsh0 = ib.let("rsh0", tensor_ops.reshape(o0, [256, 1]))
            ret_a = ib.let("ret", ib.make_tuple([rsh0]))
            a0 = ib.let("a0", ir.TupleGetItemExpr(ret_a, 0, span), type=tflat)
            rsh1 = ib.let("rsh1", tensor_ops.reshape(o1, [256, 1]))
            ret_b = ib.let("ret", ib.make_tuple([rsh1]))
            b0 = ib.let("b0", ir.TupleGetItemExpr(ret_b, 0, span), type=tflat)
            fa = ib.let("fa", tensor_ops.reshape(a0, [16, 16]))
            fb = ib.let("fb", tensor_ops.reshape(b0, [16, 16]))
            ib.return_stmt(ib.make_tuple([fa, fb]))

        orch_func = orch_f.get_result()
        program = ir.Program([orch_func], "test_tuple_pointer_keying", span)
        code = codegen.generate_orchestration(program, orch_func).code

        # No declared name may appear twice (the conflicting-declaration bug).
        declared = re.findall(
            r"^\s*(?:const\s+Tensor&|Tensor|PTO2TaskId|auto)\s+([A-Za-z_]\w*)\s*=",
            code,
            flags=re.MULTILINE,
        )
        dups = sorted({n for n in declared if declared.count(n) > 1})
        assert not dups, f"duplicate declarations {dups} in:\n{code}"

        # Each consumer must reshape from its OWN tuple's element, not a stale /
        # undeclared getitem name. Before the fix, ``fa`` read undeclared ``a0``.
        assert "Tensor fa = rsh0.reshape" in code, code
        assert "Tensor fb = rsh1.reshape" in code, code
        assert "a0.reshape" not in code and "b0.reshape" not in code, code


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
            @pl.function(type=pl.FunctionType.AIV)
            def task_init(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                buf: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], buf)
                return out

            @pl.function(type=pl.FunctionType.AIV)
            def task_compute(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                buf: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], buf)
                return out

            @pl.function(type=pl.FunctionType.AIV)
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
            @pl.function(type=pl.FunctionType.AIV)
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
            @pl.function(type=pl.FunctionType.AIV)
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
            @pl.function(type=pl.FunctionType.AIV)
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


SELF_ALIAS_RE = re.compile(r"\bauto\s+(\w+)\s*=\s*\1\s*;")


class TestNoOpAliasSkip:
    """Regression coverage for issue #1281 sub-problem 2.

    When VarLineageCollector collapses several Vars onto the same param-rooted
    emit name (because they all alias the same buffer), a chained Var-RHS
    AssignStmt like ``u = t`` reaches the catch-all emit branch with both LHS
    and RHS resolving to the same C++ identifier. Pre-fix, the codegen emitted
    ``auto X = X;`` literally, which gcc rejects with
    ``use of 'X' before deduction of 'auto'``.

    The trigger is: an Orchestration entry that

      1. Calls a kernel with an Out/InOut param,
      2. Passes the entry's own pl.Out param as the actual arg, and
      3. Binds the call result to one local AND aliases it to a second local
         before returning.

    Forms A and B below are control cases that already worked; form C is the
    one that used to emit the self-alias.
    """

    def test_form_a_direct_return_no_alias(self):
        """Control: `return self.kern(...)` — single return path, never tripped the bug."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class FormA:
            @pl.function(type=pl.FunctionType.AIV)
            def kern(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def entry(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                q_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                return self.kern(x, q_out)

        code = _generate_orch_code(FormA)
        assert not SELF_ALIAS_RE.search(code), f"unexpected `auto X = X;` in form A. Code:\n{code}"

    def test_form_b_single_bind_no_alias(self):
        """Control: `t = self.kern(...); return t` — single AssignStmt, handled by
        GenerateSingleReturnAlias's existing ``alias_name != out_arg`` guard."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class FormB:
            @pl.function(type=pl.FunctionType.AIV)
            def kern(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def entry(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                q_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tensor[[16, 16], pl.FP32] = self.kern(x, q_out)
                return t

        code = _generate_orch_code(FormB)
        assert not SELF_ALIAS_RE.search(code), f"unexpected `auto X = X;` in form B. Code:\n{code}"

    def test_form_c_chained_alias_drops_no_op(self):
        """`t = self.kern(...); u = t; return u` — the bug trigger.

        Pre-fix: emits `auto q_out = q_out;` for the `u = t` AssignStmt because
        VarLineageCollector collapses both `t` and `u` onto the entry's `q_out`
        param emit name. Post-fix: the catch-all Var-RHS branch detects
        LHS-name == RHS-name and drops the AssignStmt entirely.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class FormC:
            @pl.function(type=pl.FunctionType.AIV)
            def kern(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def entry(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                q_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tensor[[16, 16], pl.FP32] = self.kern(x, q_out)
                u: pl.Tensor[[16, 16], pl.FP32] = t
                return u

        code = _generate_orch_code(FormC)
        assert not SELF_ALIAS_RE.search(code), (
            f"`auto X = X;` regression — issue #1281 sub-problem 2 is back. Code:\n{code}"
        )
        # Sanity: the task submission must still be present; the fix only
        # drops the no-op alias, not the actual kernel call.
        assert "rt_submit_aiv_task" in code, f"task submission missing from form C output. Code:\n{code}"


class TestManualScopeCodegen:
    """Codegen for ``with pl.manual_scope():`` and the ``deps=[var]`` Call kwarg."""

    @pytest.fixture(autouse=True)
    def _no_roundtrip_verification(self):
        """The python_printer doesn't surface ``Call.attrs['manual_dep_edges']``,
        so the pipeline's print -> parse -> assert_structural_equal roundtrip
        fails for programs that use ``deps=[...]`` inside manual_scope.
        Property verification still runs.
        """
        from pypto.pypto_core import passes as _core_passes  # noqa: PLC0415

        instruments: list[_core_passes.PassInstrument] = [
            _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
        ]
        with _core_passes.PassContext(instruments):
            yield

    def test_manual_scope_emits_manual_pto2_scope_and_task_id_capture(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, a_tid = pl.submit(self.k1, x)
                    b, _ = pl.submit(self.k2, a, deps=[a_tid])
                return b

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" in code
        # Every kernel submit inside a manual scope captures the submit's
        # outputs handle so downstream code can call ``.task_id()`` on it.
        # We no longer pre-emit a ``PTO2TaskId task_<n>`` variable — the
        # ``pl.submit`` producer-TaskId tuple element binds it on demand.
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code
        assert "TaskOutputTensors task_1_outs = rt_submit_aiv_task(" in code
        # The ``pl.submit`` producer TaskId binds to ``task_0_outs.task_id()``
        # and is wired into the consumer via ``set_dependencies``.
        assert "task_0_outs.task_id()" in code
        assert ".set_dependencies(" in code

    def test_manual_scope_emits_explicit_user_deps(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k3(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, a_tid = pl.submit(self.k1, x)
                    b, b_tid = pl.submit(self.k2, x)
                    # Explicit deps even though `c` doesn't consume `a`/`b` data.
                    c, _ = pl.submit(self.k3, x, deps=[a_tid, b_tid])
                return c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # Each ``pl.submit`` producer TaskId binds to ``task_<n>_outs.task_id()``.
        assert "PTO2TaskId a_tid = task_0_outs.task_id();" in code
        assert "PTO2TaskId b_tid = task_1_outs.task_id();" in code
        # The consumer's deps are packed into a fixed-size stack array and
        # attached with a single ``set_dependencies(arr, count)`` call. Both
        # entries are unconditionally filled (plain TaskId bindings, not
        # iter-arg carries, so no is_valid() guard).
        assert "Arg params_t2;" in code
        assert "PTO2TaskId params_t2_deps[2];" in code
        assert "params_t2_deps[params_t2_deps_count++] = a_tid;" in code
        assert "params_t2_deps[params_t2_deps_count++] = b_tid;" in code
        assert "params_t2.set_dependencies(params_t2_deps, params_t2_deps_count);" in code

    def test_user_written_task_dummy_lowers_to_dummy_submit(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N_BRANCHES = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(N_BRANCHES, pl.TASK_ID)
                    for j in pl.parallel(N_BRANCHES):
                        _branch_out, tid = pl.submit(self.k1, x)
                        tids[j] = tid
                    barrier = pl.system.task_dummy(deps=[tids])
                    b, _consumer_tid = pl.submit(self.k2, x, deps=[barrier])
                return b

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "rt_submit_dummy_task(params_phase_fence_barrier_0)" in code, code
        assert f"PTO2TaskId params_phase_fence_barrier_0_deps[{N_BRANCHES}];" in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[1\];", code), code
        assert re.search(
            r"if \(barrier.*\.is_valid\(\)\) (params_t\d+)_deps\[\1_deps_count\+\+\] = barrier",
            code,
        ), code

    def test_user_written_task_dummy_accepts_scalar_dep_codegen(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, tid = pl.submit(self.k1, x)
                    barrier = pl.system.task_dummy(deps=[tid])
                    b, _consumer_tid = pl.submit(self.k2, a, deps=[barrier])
                return b

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "rt_submit_dummy_task(params_phase_fence_barrier_0)" in code, code
        assert "PTO2TaskId params_phase_fence_barrier_0_deps[1];" in code, code
        assert re.search(
            r"if \(tid.*\.is_valid\(\)\) params_phase_fence_barrier_0_deps"
            r"\[params_phase_fence_barrier_0_deps_count\+\+\] = tid",
            code,
        ), code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[1\];", code), code

    def test_user_written_empty_task_dummy_keeps_invalid_task_id(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    barrier = pl.system.task_dummy(deps=[])
                    y, _tid = pl.submit(self.k1, x, deps=[barrier])
                return y

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "uint32_t params_phase_fence_barrier_0_deps_count = 0;" in code, code
        assert "PTO2TaskId barrier = PTO2TaskId::invalid();" in code, code
        assert "if (params_phase_fence_barrier_0_deps_count > 0)" in code, code
        assert re.search(
            r"if \(barrier.*\.is_valid\(\)\) (params_t\d+)_deps\[\1_deps_count\+\+\] = barrier",
            code,
        ), code

    def test_auto_scope_does_not_emit_task_id_capture(self):
        """Sanity: plain ``self.kernel(...)`` in auto scope stays fire-and-forget."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a = self.k1(x)
                return a

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)
        assert "PTO2ScopeMode::MANUAL" not in code
        assert "TaskOutputTensors task_0_outs" not in code
        assert "set_dependencies(" not in code

    def test_pl_at_with_deps_and_as_tid_emits_submit_call(self):
        """``with pl.at(..., deps=[tid]) as tid:`` extends the dep interface
        to the ``pl.at``-block style (no explicit ``self.kernel(...)`` calls).

        The outlined kernel ``Call``'s return type is augmented with
        ``Scalar[TASK_ID]`` so codegen's ``IsSubmitCall`` detection fires:
        the call captures a ``TaskOutputTensors`` handle, binds the producer
        TaskId Var, and downstream ``deps=[tid]`` flows through the same
        stack-array + ``set_dependencies`` codegen path used by
        ``pl.submit(...)``. Equivalent to writing two ``pl.submit`` calls,
        but matches the ``pl.at``-block programming style.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                scratch: pl.Out[pl.Tensor[[64], pl.FP32]],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                # Two back-to-back pl.at blocks writing *disjoint* output
                # tensors (scratch / out) so the test pins the deps wiring
                # without tripping the pre-existing SSA rename limitation
                # for two pl.at blocks writing the same buffer.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage1") as t1:
                    t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                    r: pl.Tile[[64], pl.FP32] = pl.add(t, t)
                    scratch = pl.store(r, [0], scratch)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage2", deps=[t1]) as _t2:
                    t2t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                    r2: pl.Tile[[64], pl.FP32] = pl.add(t2t, t2t)
                    out = pl.store(r2, [0], out)
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # The outlined ``stage1`` Call captures the TaskOutputTensors handle
        # and binds ``t1`` to the producer TaskId.
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
        assert "PTO2TaskId t1 = task_0_outs.task_id();" in code, code
        # ``stage2`` carries the explicit dep on ``t1`` via the stack-array
        # + set_dependencies path.
        assert "PTO2TaskId params_t1_deps[1];" in code, code
        assert "if (t1.is_valid()) params_t1_deps[params_t1_deps_count++] = t1;" in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code
        # The parser-emitted ``t1 = system.task_invalid()`` placeholder is
        # dropped by the outliner once the real TupleGetItem binding is
        # generated.
        assert "PTO2TaskId t1 = PTO2TaskId::invalid();" not in code, code

    def test_inline_pl_at_task_id_return_feeds_downstream_deps(self):
        """Regression for issue #1456: a ``TaskId`` produced inside an inline
        helper and returned to the caller must remain a valid scheduling
        dependency for a later ``pl.at(..., deps=[tid])`` in the caller.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Inline)
            def stage1(
                self,
                x: pl.Tensor[[64], pl.FP32],
                scratch: pl.Tensor[[64], pl.FP32],
            ) -> pl.Scalar[pl.TASK_ID]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="inline_stage1") as stage1_tid:
                    v: pl.Scalar[pl.FP32] = pl.read(x, [0])
                    pl.write(scratch, [0], v)
                return stage1_tid

            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                scratch = pl.create_tensor([64], dtype=pl.FP32)
                dep_tid: pl.Scalar[pl.TASK_ID] = self.stage1(x, scratch)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage2", deps=[dep_tid]):
                    t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                    r: pl.Tile[[64], pl.FP32] = pl.add(t, t)
                    out = pl.store(r, [0], out)

                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
        assert "task_0_outs.task_id()" in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code
        assert "params_t1_deps[params_t1_deps_count++]" in code, code

    def test_submit_with_deps_in_auto_scope_emits_set_dependencies(self):
        """``pl.submit(..., deps=[tid])`` works in auto scope too.

        The runtime's ``Arg::set_dependencies`` is orthogonal to OverlapMap
        auto-tracking (final fanin = auto ∪ explicit), so the codegen emits
        the task-output capture and the deps stack array without requiring a
        ``with pl.manual_scope():`` wrapper. The implicit ``PTO2_SCOPE()``
        (auto OverlapMap) stays on.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a, a_tid = pl.submit(self.k1, x)
                b, _ = pl.submit(self.k2, x, deps=[a_tid])
                return b

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # Stays in auto scope — no MANUAL wrapper.
        assert "PTO2ScopeMode::MANUAL" not in code, code
        # Both submits capture their TaskOutputTensors handle for downstream
        # ``.task_id()``.
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
        assert "PTO2TaskId a_tid = task_0_outs.task_id();" in code, code
        # k2's explicit dep on a_tid is wired through a stack deps array +
        # set_dependencies, exactly like in manual scope.
        assert "PTO2TaskId params_t1_deps[1];" in code, code
        assert "if (a_tid.is_valid()) params_t1_deps[params_t1_deps_count++] = a_tid;" in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

    def test_auto_scope_task_id_array_slot_dep_uses_scalar_snapshot(self):
        """A TaskId array slot read is a valid explicit dep in auto scope."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k3(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                tids = pl.array.create(1, pl.TASK_ID)
                a, first_tid = pl.submit(self.k1, x)
                tids[0] = first_tid
                prev = tids[0]
                b, second_tid = pl.submit(self.k2, x)
                tids[0] = second_tid
                c, _ = pl.submit(self.k3, x, deps=[prev])
                return c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "PTO2ScopeMode::MANUAL" not in code, code
        assert re.search(r"PTO2TaskId\s+prev\s*=\s*tids\[0\];", code), code
        assert "tids[0] = second_tid;" in code, code
        assert "PTO2TaskId params_t2_deps[1];" in code, code
        assert "if (prev.is_valid()) params_t2_deps[params_t2_deps_count++] = prev;" in code, code
        assert "params_t2.set_dependencies(params_t2_deps, params_t2_deps_count);" in code, code

    def test_manual_scope_seq_outer_parallel_inner_two_stage_pipeline(self):
        """End-to-end: ``with pl.manual_scope():`` wrapping
        ``for i in pl.range(M): for j in pl.parallel(N): stage1; stage2``.

        Each iteration runs a 2-stage pipeline (stage2 reads stage1's output
        on the same tile). This shapes a verifiable dependency graph where
        manual_scope must:

        1. **Establish** the stage1 → stage2 dependency *within* an iteration
           (otherwise stage2 races stage1's write on the shared scratch
           buffer); and
        2. **Avoid serializing** across iterations: different ``(i, j)`` tiles
           write disjoint regions of ``out`` and ``scratch``, so the inner
           ``pl.parallel(N)`` loop and the outer ``pl.range(M)`` loop should
           submit tasks at maximum concurrency.

        Because MANUAL mode skips the runtime OverlapMap, the only ordering
        comes from explicit ``set_dependencies`` calls. The codegen output
        therefore proves correct parallelism: present where required
        (stage2→stage1 within each iteration), absent everywhere else.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        M, N = 4, 8
        TILE_R, TILE_C = 32, 32
        ROWS, COLS = M * TILE_R, N * TILE_C

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def stage1(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], scratch)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def stage2(
                self,
                scratch: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(scratch, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                with pl.manual_scope():
                    for i in pl.range(M):
                        row: pl.Scalar[pl.INDEX] = i * TILE_R
                        for j in pl.parallel(N):
                            col: pl.Scalar[pl.INDEX] = j * TILE_C
                            scratch, stage1_tid = pl.submit(self.stage1, x, scratch, row, col)
                            out, _ = pl.submit(self.stage2, scratch, out, row, col, deps=[stage1_tid])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # MANUAL wrapper + both loops survive without extra auto-scope wrappers.
        assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" in code, code
        assert "for (int64_t i = 0; i < 4; i += 1)" in code, code
        assert "for (int64_t j = 0; j < 8; j += 1)" in code, code
        assert code.count("PTO2_SCOPE() {") == 1, code
        assert code.count("PTO2_SCOPE(PTO2ScopeMode::MANUAL)") == 1, code

        # The producer TaskId is preserved through windowed rewriting and is
        # threaded into the consumer dependency edge.
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
        producer_tid = re.search(r"PTO2TaskId (\w+) = task_0_outs\.task_id\(\);", code)
        assert producer_tid, code
        assert "TaskOutputTensors task_1_outs = rt_submit_aiv_task(" in code, code

        # *** Manual dep correctly established WITHIN each iteration ***
        # stage2 reads what stage1 just wrote to ``scratch``: this dep is
        # required for correctness.
        assert f"params_t1_deps[params_t1_deps_count++] = {producer_tid.group(1)};" in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

        # *** Correct parallelism ACROSS iterations ***
        # The only ``set_dependencies`` in the manual scope is the one above;
        # there is no cross-iteration serialization. Different (i, j) tiles
        # run in parallel as MANUAL mode allows.
        assert code.count("set_dependencies(") == 1, code

    def test_manual_scope_parallel_outer_seq_inner_two_stage_pipeline(self):
        """End-to-end: outer ``pl.parallel`` + inner ``pl.range`` inside
        ``with pl.manual_scope():`` with the same 2-stage pipeline.

        Mirror of the previous case with the loop kinds swapped. The same
        manual-dep contract holds: stage1 → stage2 within an iteration is
        explicit; cross-iteration is unconstrained so the parallel outer
        loop dispatches tiles concurrently.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        M, N = 8, 4
        TILE_R, TILE_C = 32, 32
        ROWS, COLS = M * TILE_R, N * TILE_C

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def stage1(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], scratch)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def stage2(
                self,
                scratch: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(scratch, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                with pl.manual_scope():
                    for i in pl.parallel(M):
                        row: pl.Scalar[pl.INDEX] = i * TILE_R
                        for j in pl.range(N):
                            col: pl.Scalar[pl.INDEX] = j * TILE_C
                            scratch, stage1_tid = pl.submit(self.stage1, x, scratch, row, col)
                            out, _ = pl.submit(self.stage2, scratch, out, row, col, deps=[stage1_tid])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" in code, code
        assert "for (int64_t i = 0; i < 8; i += 1)" in code, code
        assert "for (int64_t j = 0; j < 4; j += 1)" in code, code
        assert code.count("PTO2_SCOPE() {") == 1, code
        assert code.count("PTO2_SCOPE(PTO2ScopeMode::MANUAL)") == 1, code

        # The producer TaskId is preserved through windowed rewriting and is
        # threaded into the consumer dependency edge.
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
        producer_tid = re.search(r"PTO2TaskId (\w+) = task_0_outs\.task_id\(\);", code)
        assert producer_tid, code
        assert "TaskOutputTensors task_1_outs = rt_submit_aiv_task(" in code, code

        # Manual dep WITHIN each iteration: stage2 follows stage1.
        assert f"params_t1_deps[params_t1_deps_count++] = {producer_tid.group(1)};" in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

        # Cross-iteration parallel: the ONLY set_dependencies is the
        # intra-iteration one.
        assert code.count("set_dependencies(") == 1, code

    def test_manual_scope_parallel_dynamic_trip_count_rejected(self):
        """``pl.parallel(<dynamic>)`` carrying a manual_scope dep must error.

        Array-carry codegen needs a const trip count to allocate a fixed-size
        ``PTO2TaskId[N]`` fence array. With a dynamic trip count we cannot
        emit correct multi-deps lowering; silently falling back to a scalar
        ``last-dispatched`` fence would be wrong. The codegen surfaces this
        as a clear user-facing CHECK.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ROWS, COLS = 128, 128
        TILE_R, TILE_C = 32, 32

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                n_branches: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                with pl.manual_scope():
                    prev_tid = None
                    for i in pl.range(4):
                        row: pl.Scalar[pl.INDEX] = i * TILE_R
                        for j in pl.parallel(n_branches):
                            col: pl.Scalar[pl.INDEX] = j * TILE_C
                            out, prev_tid = pl.submit(self.kern, x, out, row, col, deps=[prev_tid])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        with pytest.raises(Exception, match="statically-known trip count"):
            _generate_orch_code(transformed)

    def test_manual_scope_double_buffered_array_carry_above_legacy_16_cap(self):
        """A stable full-array dep with ``N > 16`` lowers through a dummy barrier.

        The runtime's ``Arg::set_dependencies(ptr, count)`` primitive has no
        upper bound on explicit deps. The phase-fence compression keeps the
        N-slot dependency fanin on a synthetic dummy barrier, then makes each
        real downstream task depend on the barrier's single TaskId. The witness
        uses a double-buffered carrier so the dependency source is read-only
        inside the parallel body.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ROWS, COLS = 128, 128
        TILE_R, TILE_C = 32, 32
        ABOVE_LEGACY_CAP = 17  # past the legacy 16-dep cap that no longer exists

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(ABOVE_LEGACY_CAP, pl.TASK_ID)
                    for i, (tids_iter,) in pl.range(4, init_values=(tids,)):
                        tids_next = pl.array.create(ABOVE_LEGACY_CAP, pl.TASK_ID)
                        row: pl.Scalar[pl.INDEX] = i * TILE_R
                        for j in pl.parallel(ABOVE_LEGACY_CAP):
                            col: pl.Scalar[pl.INDEX] = j * TILE_C
                            out, tid = pl.submit(self.kern, x, out, row, col, deps=[tids_iter])
                            tids_next[j] = tid
                        tids = pl.yield_(tids_next)
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)
        assert "rt_submit_dummy_task(params_phase_fence_barrier_0)" in code, code
        assert f"PTO2TaskId params_phase_fence_barrier_0_deps[{ABOVE_LEGACY_CAP}];" in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[1\];", code), code
        assert re.search(
            r"if \(phase_fence_barrier_0_tid\.is_valid\(\)\) "
            r"(params_t\d+)_deps\[\1_deps_count\+\+\] = phase_fence_barrier_0_tid;",
            code,
        ), code
        assert not re.search(rf"PTO2TaskId params_t\d+_deps\[{ABOVE_LEGACY_CAP}\];", code), code

    def test_manual_scope_phase_fence_scalar_dep_does_not_emit_dummy_barrier(self):
        """Scalar TaskId deps remain on the legacy single-edge lowering path."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, tid = pl.submit(self.k1, x)
                    b, _ = pl.submit(self.k2, a, deps=[tid])
                return b

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "rt_submit_dummy_task" not in code, code
        assert "PTO2TaskId params_t1_deps[1];" in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

    def test_manual_scope_phase_fence_mixed_deps_do_not_emit_dummy_barrier(self):
        """Mixed array + scalar deps are intentionally outside first-version scope."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ROWS, COLS = 128, 128
        TILE_R, TILE_C = 32, 32
        N_BRANCHES = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                with pl.manual_scope():
                    seed_out, seed_tid = pl.submit(self.kern, x, out, 0, 0)
                    tids = pl.array.create(N_BRANCHES, pl.TASK_ID)
                    for i in pl.range(2):
                        row: pl.Scalar[pl.INDEX] = i * TILE_R
                        for j in pl.parallel(N_BRANCHES):
                            col: pl.Scalar[pl.INDEX] = j * TILE_C
                            seed_out, tid = pl.submit(
                                self.kern,
                                x,
                                seed_out,
                                row,
                                col,
                                deps=[tids, seed_tid],
                            )
                            tids[j] = tid
                return seed_out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "rt_submit_dummy_task" not in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[5\];", code), code

    def test_auto_scope_array_dep_does_not_emit_dummy_barrier(self):
        """Array deps outside manual_scope keep the existing explicit-deps lowering."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ROWS, COLS = 128, 128
        TILE_R, TILE_C = 32, 32
        N_BRANCHES = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                tids = pl.array.create(N_BRANCHES, pl.TASK_ID)
                for i in pl.range(2):
                    row: pl.Scalar[pl.INDEX] = i * TILE_R
                    for j in pl.parallel(N_BRANCHES):
                        col: pl.Scalar[pl.INDEX] = j * TILE_C
                        out, tid = pl.submit(self.kern, x, out, row, col, deps=[tids])
                        tids[j] = tid
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "rt_submit_dummy_task" not in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[4\];", code), code

    def test_manual_scope_submit_task_id_dep(self):
        """The producer TaskId of a ``pl.submit(...)`` threaded into a later
        submit's ``deps=[...]`` reaches the dependency edge.

        Pattern:
            scratch, tid = pl.submit(self.stage1, x, scratch, row, col)
            out, _       = pl.submit(self.stage2, scratch, out, row, col, deps=[tid])

        Expected codegen for the dep chain:
            Arg params_t1;
            PTO2TaskId params_t1_deps[1];
            uint32_t params_t1_deps_count = 0;
            params_t1_deps[params_t1_deps_count++] = <producer TaskId>;
            params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        M, N = 4, 4
        TILE_R, TILE_C = 32, 32
        ROWS, COLS = M * TILE_R, N * TILE_C

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def stage1(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], scratch)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def stage2(
                self,
                scratch: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(scratch, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                with pl.manual_scope():
                    for i in pl.range(M):
                        row: pl.Scalar[pl.INDEX] = i * TILE_R
                        for j in pl.parallel(N):
                            col: pl.Scalar[pl.INDEX] = j * TILE_C
                            scratch, tid = pl.submit(self.stage1, x, scratch, row, col)
                            out, _ = pl.submit(self.stage2, scratch, out, row, col, deps=[tid])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # The producer TaskId is attached to the consumer dependency edge.
        producer_tid = re.search(r"PTO2TaskId (\w+) = task_0_outs\.task_id\(\);", code)
        assert producer_tid, code
        # The dep edge is filled into the consumer's stack deps array and
        # attached with a single ``set_dependencies`` call.
        assert f"params_t1_deps[params_t1_deps_count++] = {producer_tid.group(1)};" in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

    def test_manual_scope_submit_iter_arg_taskid_carry(self):
        """A ``pl.submit`` producer TaskId threaded through a ``pl.range``
        iter_arg seeds the next iteration's ``deps=[...]``.

        Pattern (per-branch linear chain):
            prev_tid = None
            for step in pl.range(N):
                out, prev_tid = pl.submit(self.kern, ..., deps=[prev_tid])

        ``prev_tid`` starts as the ``None`` sentinel (``PTO2TaskId::invalid()``)
        and is rebound each iteration to the submit's producer TaskId. Because
        it is a loop iter_arg, the dep fill is guarded by ``is_valid()`` so the
        first iteration contributes no edge.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N_BRANCHES, N_STEPS = 4, 4
        TILE_M, BIG_N = 32, 32
        BIG_M = N_BRANCHES * N_STEPS * TILE_M

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                data: pl.Tensor[[BIG_M, BIG_N], pl.FP32],
                row: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[BIG_M, BIG_N], pl.FP32]],
            ) -> pl.Tensor[[BIG_M, BIG_N], pl.FP32]:
                t: pl.Tile[[TILE_M, BIG_N], pl.FP32] = pl.load(data, [row, 0], [TILE_M, BIG_N])
                r: pl.Tile[[TILE_M, BIG_N], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[BIG_M, BIG_N], pl.FP32] = pl.store(r, [row, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[BIG_M, BIG_N], pl.FP32],
                out: pl.Out[pl.Tensor[[BIG_M, BIG_N], pl.FP32]],
            ) -> pl.Tensor[[BIG_M, BIG_N], pl.FP32]:
                with pl.manual_scope():
                    for branch in pl.parallel(N_BRANCHES):
                        prev_tid = None
                        for step in pl.range(N_STEPS):
                            row: pl.Scalar[pl.INDEX] = (step * N_BRANCHES + branch) * TILE_M
                            out, prev_tid = pl.submit(self.kern, data, row, out, deps=[prev_tid])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # The ``None`` seed lowers to an invalid PTO2TaskId sentinel.
        assert "PTO2TaskId::invalid()" in code, code
        # The iter-arg TaskId carry is filled into the deps array under an
        # ``is_valid()`` guard (first iteration's sentinel is skipped).
        assert ".is_valid())" in code, code
        assert ".set_dependencies(" in code, code

    def test_manual_scope_inner_deps_can_reference_outer_scope_tid(self):
        """Regression: a kernel inside ``with pl.manual_scope():`` can carry
        ``deps=[outer_tid]`` referencing a TaskId produced by an *outer*
        (auto-scope) task.

        Previously ``OrchestrationCodegen::VisitStmt_(RuntimeScopeStmtPtr)``
        wiped ``manual_task_id_map_`` on manual-scope entry (``std::move``
        the outer map into ``saved_map`` then ``clear()``), so any
        ``deps=`` edge pointing to an outer-scope producer was silently
        dropped in ``CountManualDeps`` (early return on the
        ``manual_task_id_map_.find(edge) == end()`` miss) and no
        ``set_dependencies(...)`` call was emitted for that inner task.
        With no explicit edge and no auto-dep (manual scope disables
        OverlapMap), the inner kernel could race the outer producer.

        The fix snapshots the outer map by *copy* instead of moving + clearing,
        so the live map keeps the outer entries visible inside the inner scope.
        The outer-only snapshot is restored on exit so the inner-scope adds —
        which name C++ identifiers that die with the inner ``{`` block — do
        not leak back into the parent scope.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k_outer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k_inner(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Outer producer in auto scope — captures TaskId for the
                # cross-scope fence. ``outer_a`` is bound but unused; the
                # only edge from outer to inner is the explicit TaskId dep.
                outer_a, outer_tid = pl.submit(self.k_outer, x)
                with pl.manual_scope():
                    # Inner consumer in manual scope. The two kernels share
                    # no data input (both read ``x``), so the ``deps=`` edge
                    # is the *only* thing fencing inner behind outer.
                    inner_b, _ = pl.submit(self.k_inner, x, deps=[outer_tid])
                return inner_b

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # Outer producer (Task 0, in auto scope) captures its TaskId. Use the
        # same regex-discovery idiom as nearby tests (lines 3730, 3819, 4010)
        # so the assertion is not brittle to any future SSA renaming /
        # suffixing the codegen may apply to the producer TaskId emit name.
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
        producer_tid = re.search(r"PTO2TaskId (\w+) = task_0_outs\.task_id\(\);", code)
        assert producer_tid, code
        # Inner kernel runs inside a MANUAL scope.
        assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" in code, code
        # The inner kernel's deps array MUST include the producer TaskId edge —
        # this is what the bug used to silently drop. Without these lines the
        # inner ``rt_submit_*_task`` would have no explicit fence on the outer
        # task and the regression would re-emerge.
        assert "Arg params_t1;" in code, code
        assert "PTO2TaskId params_t1_deps[1];" in code, code
        assert (
            f"if ({producer_tid.group(1)}.is_valid()) params_t1_deps[params_t1_deps_count++] = {producer_tid.group(1)};"
            in code
        ), code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code


class TestTupleReturnNoDepAliasing:
    """``GenerateTupleReturnAliases`` must classify output slots by the
    callee's ``ParamDirection`` — same convention as the submit path. If it
    classified by call-site ``ArgDirection`` instead, a ``pl.no_dep(out)``
    on a tuple-return non-submit Call would drop the alias for that slot
    (``NoDep`` is excluded from the writer set), and downstream uses of the
    tuple-element SSA var would emit undeclared ``__rv_*`` symbols.
    """

    def test_no_dep_on_tuple_out_param_preserves_alias(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TupleNoDepProgram:
            @pl.function(type=pl.FunctionType.AIV)
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

            @pl.function(type=pl.FunctionType.AIV)
            def kernel_consume(
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
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                result: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                x: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                y: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                # ``pl.no_dep(y)`` rewrites the y slot's ArgDirection to NoDep
                # on a multi-output (tuple-returning) non-submit call. The
                # codegen must still treat y as a writer — otherwise the
                # downstream `kernel_consume(x, y, ...)` would reference an
                # undeclared y.
                x, y = self.kernel_pair(a, b, x, pl.no_dep(y))
                result = self.kernel_consume(x, y, result)
                return result

        # Same trick as ``test_flatten_call_expr_pass.TestFlattenPreservesAttrs``
        # / ``TestOutlineNoDepArgs``: ``derive_call_directions`` produces a
        # Call whose ``attrs[arg_directions]`` includes a NoDep slot; the
        # printer does not surface that attr, so the default
        # RoundtripInstrument check fails. Use VerificationInstrument only.
        from pypto.pypto_core import passes as _core_passes  # noqa: PLC0415

        ctx = _core_passes.PassContext(
            [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
        )
        with ctx:
            code = _generate_orch_code(TupleNoDepProgram)

        # The y slot must be marked NoDep on the kernel_pair call.
        assert "add_no_dep(" in code, (
            f"expected add_no_dep(...) on the NoDep y slot of kernel_pair; generated code:\n{code}"
        )
        # Both x and y must have aliases bound (otherwise the consume call
        # below would reference undeclared symbols). The aliasing path uses
        # ``from_tensor_arg(...)`` on the args array.
        assert code.count("from_tensor_arg(") >= 2, (
            "expected at least two from_tensor_arg(...) bindings (one per "
            f"tuple element); generated code:\n{code}"
        )
        # Both tasks must submit.
        assert code.count("rt_submit_aiv_task") == 2, code


class TestTupleReturnNameHintCollision:
    """Tuple metadata must track tuple Vars by identity, not name_hint."""

    def test_same_name_hint_tuple_calls_keep_distinct_elements(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        span = ir.Span.unknown()
        tensor_type = ir.TensorType([16, 16], DataType.FP32)
        tuple_type = ir.TupleType([tensor_type, tensor_type])

        input_a = ir.Var("input_a", tensor_type, span)
        input_b = ir.Var("input_b", tensor_type, span)
        first_out = ir.Var("first_out", tensor_type, span)
        second_out = ir.Var("second_out", tensor_type, span)
        kernel_a_input = ir.Var("input_a", tensor_type, span)
        kernel_a_first = ir.Var("first_out", tensor_type, span)
        kernel_a_second = ir.Var("second_out", tensor_type, span)
        kernel_b_input = ir.Var("input_b", tensor_type, span)
        kernel_b_first = ir.Var("first_out", tensor_type, span)
        kernel_b_second = ir.Var("second_out", tensor_type, span)

        kernel_a_body = ir.SeqStmts([ir.ReturnStmt([kernel_a_first, kernel_a_second], span)], span)
        kernel_a = ir.Function(
            "kernel_a",
            [
                (kernel_a_input, ir.ParamDirection.In),
                (kernel_a_first, ir.ParamDirection.Out),
                (kernel_a_second, ir.ParamDirection.Out),
            ],
            [tensor_type, tensor_type],
            kernel_a_body,
            span,
            ir.FunctionType.AIV,
        )

        kernel_b_body = ir.SeqStmts([ir.ReturnStmt([kernel_b_first, kernel_b_second], span)], span)
        kernel_b = ir.Function(
            "kernel_b",
            [
                (kernel_b_input, ir.ParamDirection.In),
                (kernel_b_first, ir.ParamDirection.Out),
                (kernel_b_second, ir.ParamDirection.Out),
            ],
            [tensor_type, tensor_type],
            kernel_b_body,
            span,
            ir.FunctionType.AIV,
        )

        consume_a = ir.Var("consume_a", tensor_type, span)
        consume_b = ir.Var("consume_b", tensor_type, span)
        consume_c = ir.Var("consume_c", tensor_type, span)
        consume_d = ir.Var("consume_d", tensor_type, span)
        consume_out = ir.Var("consume_out", tensor_type, span)
        consume_body = ir.SeqStmts([ir.ReturnStmt([consume_out], span)], span)
        kernel_consume = ir.Function(
            "kernel_consume",
            [
                (consume_a, ir.ParamDirection.In),
                (consume_b, ir.ParamDirection.In),
                (consume_c, ir.ParamDirection.In),
                (consume_d, ir.ParamDirection.In),
                (consume_out, ir.ParamDirection.Out),
            ],
            [tensor_type],
            consume_body,
            span,
            ir.FunctionType.AIV,
        )

        tmp_first = ir.Var("ret__tmp_v0", tuple_type, span)
        tmp_second = ir.Var("ret__tmp_v0", tuple_type, span)
        first_a = ir.Var("first_a", tensor_type, span)
        second_a = ir.Var("second_a", tensor_type, span)
        first_b = ir.Var("first_b", tensor_type, span)
        second_b = ir.Var("second_b", tensor_type, span)
        consume_result = ir.Var("consume_result", tensor_type, span)

        call_a = ir.Call(
            ir.GlobalVar("kernel_a"),
            [input_a, first_out, second_out],
            {},
            {
                "arg_directions": [
                    ir.ArgDirection.Input,
                    ir.ArgDirection.OutputExisting,
                    ir.ArgDirection.OutputExisting,
                ]
            },
            tuple_type,
            span,
        )
        call_b = ir.Call(
            ir.GlobalVar("kernel_b"),
            [input_b, first_out, second_out],
            {},
            {
                "arg_directions": [
                    ir.ArgDirection.Input,
                    ir.ArgDirection.OutputExisting,
                    ir.ArgDirection.OutputExisting,
                ]
            },
            tuple_type,
            span,
        )
        call_consume = ir.Call(
            ir.GlobalVar("kernel_consume"),
            [first_a, second_a, first_b, second_b, first_out],
            {},
            {
                "arg_directions": [
                    ir.ArgDirection.Input,
                    ir.ArgDirection.Input,
                    ir.ArgDirection.Input,
                    ir.ArgDirection.Input,
                    ir.ArgDirection.OutputExisting,
                ]
            },
            tensor_type,
            span,
        )

        orch_body = ir.SeqStmts(
            [
                ir.AssignStmt(tmp_first, call_a, span),
                ir.AssignStmt(tmp_second, call_b, span),
                ir.AssignStmt(first_a, ir.TupleGetItemExpr(tmp_first, 0, span), span),
                ir.AssignStmt(second_a, ir.TupleGetItemExpr(tmp_first, 1, span), span),
                ir.AssignStmt(first_b, ir.TupleGetItemExpr(tmp_second, 0, span), span),
                ir.AssignStmt(second_b, ir.TupleGetItemExpr(tmp_second, 1, span), span),
                ir.AssignStmt(consume_result, call_consume, span),
                ir.ReturnStmt([consume_result], span),
            ],
            span,
        )
        orch = ir.Function(
            "orch",
            [
                (input_a, ir.ParamDirection.In),
                (input_b, ir.ParamDirection.In),
                (first_out, ir.ParamDirection.Out),
                (second_out, ir.ParamDirection.Out),
            ],
            [tensor_type],
            orch_body,
            span,
            ir.FunctionType.Orchestration,
        )
        program = ir.Program(
            [kernel_a, kernel_b, kernel_consume, orch],
            "TupleNameHintCollisionProgram",
            span,
        )

        code = codegen.generate_orchestration(program, orch).code

        assert "const Tensor& first_a = ext_first_out;" in code, code
        assert "const Tensor& second_a = ext_second_out;" in code, code
        assert "const Tensor& first_b = ext_first_out;" in code, code
        assert "const Tensor& second_b = ext_second_out;" in code, code

        task_0 = code.index("// Task 0: kernel_a")
        task_1 = code.index("// Task 1: kernel_b")
        task_2 = code.index("// Task 2: kernel_consume")
        # With name_hint-keyed tuple metadata, first_a/second_a are attached
        # to the second call because tmp_first and tmp_second share name_hint.
        assert code.index("const Tensor& first_a", task_0) < task_1, code
        assert task_1 < code.index("const Tensor& first_b", task_1) < task_2, code

        declared_names = re.findall(
            r"^\s*(?:const\s+Tensor&|Tensor)\s+([A-Za-z_]\w*)\s*=",
            code,
            flags=re.MULTILINE,
        )
        duplicate_declarations = {name for name in declared_names if declared_names.count(name) > 1}
        assert not duplicate_declarations, (
            f"generated C++ redeclared tensor names {sorted(duplicate_declarations)}:\n{code}"
        )


class TestScalarCarryPhiCodegen:
    """Regression tests for scalar loop carries in orchestration codegen."""

    def test_scalar_carry_phi_not_emitted_as_tensor(self):
        """Regression for #1580: Scalar loop carry must not be aliased as const Tensor&.

        When a Scalar variable is defined before a pl.parallel loop and then
        reused (reassigned) inside it, alongside Tensor carries, ConvertToSSA
        promotes the scalar into the parallel-loop carry tuple.  The orchestration
        codegen must emit the Scalar carry phi as ``int64_t = 0`` (untraced scalar
        default), NOT as ``const Tensor& = <carry_var>`` (type mismatch that causes
        a C++ compile error).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N, TILE = 16, 4

        @pl.program
        class ScalarCarryProg:
            @pl.function(type=pl.FunctionType.AIV)
            def scope_b_kernel(
                self,
                x: pl.Tensor[[N, N], pl.FP32],
                out_b: pl.Out[pl.Tensor[[N, N], pl.FP32]],
                out_c: pl.Out[pl.Tensor[[N, N], pl.FP32]],
            ) -> tuple[pl.Tensor[[N, N], pl.FP32], pl.Tensor[[N, N], pl.FP32]]:
                t: pl.Tile[[N, N], pl.FP32] = pl.load(x, [0, 0], [N, N])
                out_b = pl.store(t, [0, 0], out_b)
                out_c = pl.store(t, [0, 0], out_c)
                return out_b, out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[N, N], pl.FP32],
                out_b: pl.Out[pl.Tensor[[N, N], pl.FP32]],
                out_c: pl.Out[pl.Tensor[[N, N], pl.FP32]],
                row_start: pl.Scalar[pl.INDEX],
            ) -> tuple[pl.Tensor[[N, N], pl.FP32], pl.Tensor[[N, N], pl.FP32]]:
                # global_c_idx is assigned from a scalar param before the
                # parallel loop — ConvertToSSA sees it in 'before' and adds it
                # as a carry when it is reassigned inside the loop body.
                global_c_idx = row_start

                # The parallel loop carries global_c_idx (Scalar) mixed with
                # Tensor carries out_b, out_c.  Before the fix, the Scalar carry
                # phi was emitted as ``const Tensor&`` causing a C++ compile error.
                for batch_idx in pl.parallel(0, N // TILE):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="scope_b"):
                        for inner in pl.range(TILE):
                            global_c_idx = batch_idx + inner
                            out_b, out_c = self.scope_b_kernel(x, out_b, out_c)

                return out_b, out_c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(ScalarCarryProg)
        code = _generate_orch_code(transformed)

        # The Scalar carry phi must be emitted as int64_t = 0 (untraced scalar
        # default), never as const Tensor& = <carry> (type mismatch / #1580).
        assert "int64_t global_c_idx__rv" in code, (
            "global_c_idx carry phi should be emitted as int64_t, not const Tensor&\n" + code
        )
        assert "const Tensor& global_c_idx" not in code, (
            "global_c_idx must not be aliased as const Tensor& (scalar/tensor type mismatch)\n" + code
        )

        # out_b and out_c Tensor carries must each alias to their own carry.
        # The scrambled (shifted-by-one) bindings must NOT appear.
        for line in code.splitlines():
            stripped = line.strip()
            if "=" not in stripped:
                continue
            lhs, _, rhs = stripped.partition("=")
            # out_c phi must not be initialized from out_b's carry value
            if "out_c" in lhs and "out_b" in rhs and "out_c" not in rhs:
                raise AssertionError(f"Wrong phi: out_c assigned from out_b value (scrambled):\n  {stripped}")
            # out_b phi must not be initialized from out_c's carry value
            if "out_b" in lhs and "out_c" in rhs and "out_b" not in rhs:
                raise AssertionError(f"Wrong phi: out_b assigned from out_c value (scrambled):\n  {stripped}")


def _generate_orch_full_pipeline(program_cls) -> str:
    """Run the Default pass pipeline + orchestration codegen on the orch func.

    The issue #1577 witnesses use ``pl.submit`` against ``@pl.function(InCore)``
    kernels, which only reach orchestration form after the full Default pipeline
    (inline / outline / SSA / materialize scopes). Mirrors
    ``test_array_codegen._generate_pto`` but targets the Orchestration function.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    # Scope to BASIC (property) verification, overriding conftest's default
    # roundtrip instrument. ``pl.submit(..., deps=[...])`` in an auto
    # orchestration does not survive a print->parse roundtrip after
    # DeriveCallDirections (a separate printer/parser bug, unrelated to the
    # orchestration codegen behaviour under test here). Property verification
    # still runs so real IR invariant violations are caught.
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        optimized = pm.run_passes(program_cls)
    for func in optimized.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            return codegen.generate_orchestration(optimized, func).code
    raise AssertionError("no Orchestration function found in program")


def test_array_slot_task_id_usable_as_submit_dep():
    """Regression for issue #1577.

    ``prev = tids[k]`` reads one slot of a ``pl.array.create(N, pl.TASK_ID)``
    into a ``Scalar[TASK_ID]``. Using ``prev`` as a ``pl.submit(..., deps=[prev])``
    source must wire the consumer's ``set_dependencies`` array from that snapshot
    local. Before the fix, orchestration codegen aborted with "manual_dep_edge
    var '...' has no producer task in current manual scope".
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.InCore)
        def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.InCore)
        def k3(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            tids = pl.array.create(1, pl.TASK_ID)
            _seed, seed_tid = pl.submit(self.k1, x)
            _a, tid = pl.submit(self.k2, x)
            tids[0] = tid
            prev = tids[0]
            b, _ = pl.submit(self.k3, x, deps=[seed_tid, prev])
            return b

    code = _generate_orch_full_pipeline(P)

    # ``prev = tids[0]`` lowers to a scalar PTO2TaskId snapshot local read from
    # the array slot (the dep site references the snapshot, not a slot re-read).
    assert re.search(r"PTO2TaskId\s+prev\w*\s*=\s*tids\w*\[0\];", code), code
    # The consumer gets exactly one dependency array, filled from BOTH the direct
    # producer tid (seed_tid) and the array-slot snapshot (prev), each guarded.
    assert code.count("set_dependencies(") == 1, code
    assert re.search(r"if \(seed_tid\w*\.is_valid\(\)\)", code), code
    assert re.search(r"if \(prev\w*\.is_valid\(\)\)", code), code


def test_task_id_binding_does_not_leak_past_pl_scope():
    """Regression for the issue #1577 lifetime hazard.

    A producer TaskId declared inside a nested AUTO ``pl.scope()`` names a
    ``PTO2TaskId`` C++ local that dies at the block's closing brace. Codegen must
    not reference it after the block — before the fix it emitted a
    ``set_dependencies`` fill from the out-of-scope local (uncompilable C++).
    The TaskId binding is now scoped to the ``PTO2_SCOPE`` it is produced in, so
    its identifier appears exactly once (its declaration) in the generated code.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.InCore)
        def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            with pl.scope():
                _a, scoped_tid = pl.submit(self.k1, x)
            b, _ = pl.submit(self.k2, x, deps=[scoped_tid])
            return b

    code = _generate_orch_full_pipeline(P)

    # The producer tid is declared inside the inner PTO2_SCOPE block.
    m = re.search(r"PTO2TaskId\s+(\w+)\s*=\s*task_0_outs\.task_id\(\);", code)
    assert m, code
    tid_name = m.group(1)
    # It must NOT be referenced after its block closes — the binding is scoped
    # out, so the only occurrence is its declaration (no dangling dep fill).
    assert code.count(tid_name) == 1, f"TaskId local '{tid_name}' leaked past its pl.scope():\n{code}"


def test_mixed_in_and_out_of_scope_deps_does_not_crash_codegen():
    """A deps= list mixing an in-scope TaskId with one scoped out of a closed
    ``pl.scope()`` must not abort codegen.

    ``CountManualDeps`` skips the out-of-scope edge when sizing the dep stack
    array, so the count is non-zero (the in-scope edge). ``EmitManualDeps`` must
    skip the same out-of-scope edge rather than asserting on it — otherwise the
    mixed case trips an INTERNAL_CHECK and crashes the compiler. The in-scope
    edge is still wired; the out-of-scope edge is dropped.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.InCore)
        def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.InCore)
        def k3(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            with pl.scope():
                _a, scoped_tid = pl.submit(self.k1, x)
            _seed, seed_tid = pl.submit(self.k2, x)
            b, _ = pl.submit(self.k3, x, deps=[seed_tid, scoped_tid])
            return b

    # Must not raise (regression: EmitManualDeps used to INTERNAL_CHECK here).
    code = _generate_orch_full_pipeline(P)

    # The in-scope edge is wired; the out-of-scope ``scoped_tid`` is dropped.
    assert code.count("set_dependencies(") == 1, code
    assert re.search(r"if \(seed_tid\w*\.is_valid\(\)\)", code), code
    m = re.search(r"PTO2TaskId\s+(\w+)\s*=\s*task_0_outs\.task_id\(\);", code)
    assert m, code
    assert code.count(m.group(1)) == 1, code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

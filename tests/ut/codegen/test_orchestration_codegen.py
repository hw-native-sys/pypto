# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for orchestration code generation (core cases, part 1)."""

import re

import pypto.language as pl
import pytest
from _orchestration_codegen_common import (
    _generate_orch_code,
    _generate_orch_result,
    _out_of_scope_tensor_refs,
    assert_code_equal,
)
from pypto import backend, codegen, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import ir


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
            PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs& orch_args) {
                (void)orch_args;
                return PTO2OrchestrationConfig{
                    .expected_arg_count = 3,
                };
            }

            __attribute__((visibility("default")))
            void aicpu_orchestration_entry(const L2TaskArgs& orch_args) {
                // External tensors
                const Tensor& ext_a = orch_args.tensor(0).ref();
                const Tensor& ext_b = orch_args.tensor(1).ref();
                const Tensor& ext_d = orch_args.tensor(2).ref();

                PTO2_SCOPE() {
                    uint32_t c_ci_shapes[2] = {16, 16};
                    TensorCreateInfo c_ci(c_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_0 = alloc_tensors(c_ci);
                    const Tensor& c = alloc_0.get_ref(0);

                    // Task 0: kernel_add
                    L0TaskArgs params_t0;
                    params_t0.add_input(ext_a);
                    params_t0.add_input(ext_b);
                    params_t0.add_output(c);
                    rt_submit_aiv_task(0, params_t0);

                    // Task 1: kernel_add
                    L0TaskArgs params_t1;
                    params_t1.add_input(c);
                    params_t1.add_input(ext_b);
                    params_t1.add_output(ext_d);
                    rt_submit_aiv_task(0, params_t1);
                }
            }

            }  // extern "C"
        """
        assert_code_equal(code, expected)

    @staticmethod
    def _init_value_program(dtype, init_value):
        """Build a minimal orch program whose runtime-allocated output `c` is
        created with `init_value`, then consumed by a kernel."""

        @pl.program
        class InitValueProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], dtype],
                b: pl.Tensor[[16, 16], dtype],
                output: pl.Out[pl.Tensor[[16, 16], dtype]],
            ) -> pl.Tensor[[16, 16], dtype]:
                a_tile: pl.Tile[[16, 16], dtype] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], dtype] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], dtype] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], dtype] = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], dtype],
                b: pl.Tensor[[16, 16], dtype],
                d: pl.Out[pl.Tensor[[16, 16], dtype]],
            ) -> pl.Tensor[[16, 16], dtype]:
                c: pl.Tensor[[16, 16], dtype] = pl.create_tensor([16, 16], dtype=dtype, init_value=init_value)
                c = self.kernel_add(a, b, c)
                d = self.kernel_add(c, b, d)
                return d

        return InitValueProgram

    def test_create_tensor_init_value_zero(self):
        """init_value=0 emits the dtype-agnostic uint64 set_initial_value(0) call
        right after the TensorCreateInfo declaration."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        code = _generate_orch_code(self._init_value_program(pl.FP32, 0))
        assert "TensorCreateInfo c_ci(c_ci_shapes, 2, DataType::FLOAT32);" in code
        assert "c_ci.set_initial_value(0);" in code

    def test_create_tensor_init_value_nonzero_float(self):
        """Non-zero fp32 init_value emits a typed static_cast so the runtime
        packs the correct element bytes."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        code = _generate_orch_code(self._init_value_program(pl.FP32, 2.5))
        assert "c_ci.set_initial_value(static_cast<float>(2.5" in code

    def test_create_tensor_init_value_nonzero_int(self):
        """Non-zero integer init_value truncates the stored double back to the
        integer C type."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        code = _generate_orch_code(self._init_value_program(pl.INT32, 7))
        assert "c_ci.set_initial_value(static_cast<int32_t>(7));" in code

    def test_create_tensor_init_value_fractional_int_rejected(self):
        """A fractional init_value into an integer dtype would be silently
        truncated, so codegen rejects it."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        with pytest.raises(ValueError, match="is not an integer but the tensor"):
            _generate_orch_code(self._init_value_program(pl.INT32, 2.5))

    def test_create_tensor_init_value_large_int_rejected(self):
        """An integer init_value beyond 2**53 loses precision through the double
        attr, so it is rejected at the IR boundary."""
        with pytest.raises(ValueError, match="exactly-representable"):
            pl.create_tensor([16, 16], dtype=pl.INT64, init_value=2**53 + 1)

    def test_create_tensor_init_value_nonfinite_rejected(self):
        """NaN / Inf init_value would emit invalid C++ ("nan"/"inf") or be UB to
        cast to an integer, so they are rejected at the IR boundary."""
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="must be finite"):
                pl.create_tensor([16, 16], dtype=pl.FP32, init_value=bad)

    def test_create_tensor_init_value_fp16_nonzero_rejected(self):
        """Non-zero fp16 fills are not representable in the orchestration TU
        (no half type), so codegen raises a clear user-facing error."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        with pytest.raises(ValueError, match="non-zero init_value is not supported"):
            _generate_orch_code(self._init_value_program(pl.FP16, 1.0))

    def test_create_tensor_init_value_fp16_zero_allowed(self):
        """init_value=0 is valid for fp16 (zero packs to zero bytes for any
        dtype)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        code = _generate_orch_code(self._init_value_program(pl.FP16, 0))
        assert "c_ci.set_initial_value(0);" in code

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

        # The kernel's ArgDirection signature is exported so kernel_config.py can
        # build a non-empty CoreCallable signature (issue #1458 — required for
        # the runtime tensor dump to match the task payload tensor_count).
        signature = result.func_name_to_signature["kernel_add"]
        # a, b, output are all tensors -> 3 tensor directions, no SCALAR. The
        # CoreCallable signature is a per-tensor-arg list, so scalars are excluded.
        assert "SCALAR" not in signature
        assert len(signature) == 3
        assert all(d in {"IN", "OUT", "INOUT"} for d in signature)

    def test_signature_excludes_scalar_args(self):
        """Scalar args are excluded from a kernel's CoreCallable signature.

        The CoreCallable signature_[] array is sized to CORE_MAX_TENSOR_ARGS and
        is a per-tensor-arg direction list. Recording scalars would inflate
        sig_count past that cap and trip make_callable's "sig_count exceeds
        MaxSig" guard for kernels with many params (issue #1458 follow-up). Only
        the tensor args appear, in tensors-first order.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ScalarKernelProgram:
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

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_scalar(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.kernel_add_scalar(a, 1.0, d)
                return d

        result = _generate_orch_result(ScalarKernelProgram)

        signature = result.func_name_to_signature["kernel_add_scalar"]
        # 2 tensor args (a, output); the scalar literal is not recorded.
        assert "SCALAR" not in signature
        assert len(signature) == 2
        assert all(d in {"IN", "OUT", "INOUT"} for d in signature)

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
        assert ".ref()" in code

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
            PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs& orch_args) {
                (void)orch_args;
                return PTO2OrchestrationConfig{
                    .expected_arg_count = 3,
                };
            }

            __attribute__((visibility("default")))
            void aicpu_orchestration_entry(const L2TaskArgs& orch_args) {
                // External tensors
                const Tensor& ext_a = orch_args.tensor(0).ref();
                const Tensor& ext_b = orch_args.tensor(1).ref();
                const Tensor& ext_f = orch_args.tensor(2).ref();

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
                    L0TaskArgs params_t0;
                    params_t0.add_input(ext_a);
                    params_t0.add_input(ext_b);
                    params_t0.add_output(c);
                    rt_submit_aiv_task(0, params_t0);

                    // Task 1: kernel_add_scalar
                    L0TaskArgs params_t1;
                    params_t1.add_input(c);
                    params_t1.add_output(d);
                    params_t1.add_scalar(to_u64(1.000000f));
                    rt_submit_aiv_task(1, params_t1);

                    // Task 2: kernel_add_scalar
                    L0TaskArgs params_t2;
                    params_t2.add_input(c);
                    params_t2.add_output(e);
                    params_t2.add_scalar(to_u64(2.000000f));
                    rt_submit_aiv_task(1, params_t2);

                    // Task 3: kernel_mul
                    L0TaskArgs params_t3;
                    params_t3.add_input(d);
                    params_t3.add_input(e);
                    params_t3.add_output(g);
                    rt_submit_aiv_task(2, params_t3);

                    // Task 4: kernel_add
                    L0TaskArgs params_t4;
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
        assert "orch_args.tensor(2).ref()" in code

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

        # Both x and y are return tensors: orch_args.tensor(i).ref()
        assert "ext_x" in code
        assert "ext_y" in code
        assert "orch_args.tensor(2).ref()" in code
        assert "orch_args.tensor(3).ref()" in code

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
        assert "const Tensor& ext_mi_in = orch_args.tensor(3).ref()" in code
        assert "const Tensor& ext_li_in = orch_args.tensor(4).ref()" in code
        assert "const Tensor& ext_oi_in = orch_args.tensor(5).ref()" in code
        assert "const Tensor& ext_dst_in = orch_args.tensor(6).ref()" in code

        # Final return tensor is external
        assert "const Tensor& ext_final = orch_args.tensor(7).ref()" in code

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

        # Each tuple result is the in-place arg it writes, so it remaps to that
        # arg (no per-output ``const Tensor&`` alias is minted). The consumer
        # ``combine`` reads ta/tb/tc — each result mapped to its OWN arg, NOT
        # shifted onto inout_t (issue #1573).
        ia = code.index("params_t1.add_input(ext_ta)")
        ib = code.index("params_t1.add_input(ext_tb)")
        ic = code.index("params_t1.add_input(ext_tc)")
        assert ia < ib < ic, code
        # The scrambled (shifted-by-one) mapping must NOT appear: combine must
        # not read inout_t, and no const-ref output alias is emitted.
        assert "add_input(ext_inout_t)" not in code
        assert all(f"const Tensor& o{i}" not in code for i in (1, 2, 3)), code

    @staticmethod
    def _manual_cross_scope_code(create_inside: bool) -> str:
        """Orchestration code for a tensor written inside a ``pl.manual_scope``
        and read by a task placed after it, with the ``pl.create_tensor`` placed
        either before or inside the scope (issue #1697)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class CrossScopeProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def producer(
                self,
                x: pl.Tensor[[16, 256], pl.FP32],
                buf: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
            ) -> pl.Tensor[[16, 256], pl.FP32]:
                t: pl.Tile[[16, 256], pl.FP32] = pl.load(x, [0, 0], [16, 256])
                out: pl.Tensor[[16, 256], pl.FP32] = pl.store(t, [0, 0], buf)
                return out

            @pl.function(type=pl.FunctionType.AIV)
            def consumer(
                self,
                buf: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
            ) -> pl.Tensor[[16, 256], pl.FP32]:
                t: pl.Tile[[16, 256], pl.FP32] = pl.load(buf, [0, 0], [16, 256])
                r: pl.Tensor[[16, 256], pl.FP32] = pl.store(t, [0, 0], out)
                return r

            @pl.function(type=pl.FunctionType.Orchestration)
            def main_before(
                self,
                x: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
            ) -> pl.Tensor[[16, 256], pl.FP32]:
                buf: pl.Tensor[[16, 256], pl.FP32] = pl.create_tensor([16, 256], dtype=pl.FP32)
                with pl.manual_scope():
                    buf, _ptid = pl.submit(self.producer, x, buf)
                out = self.consumer(buf, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main_inside(
                self,
                x: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
            ) -> pl.Tensor[[16, 256], pl.FP32]:
                with pl.manual_scope():
                    buf: pl.Tensor[[16, 256], pl.FP32] = pl.create_tensor([16, 256], dtype=pl.FP32)
                    buf, _ptid = pl.submit(self.producer, x, buf)
                out = self.consumer(buf, out)
                return out

        which = "main_inside" if create_inside else "main_before"
        program = passes.materialize_runtime_scopes()(
            passes.derive_call_directions()(
                PassManager.get_strategy(OptimizationStrategy.Default).run_passes(CrossScopeProgram)
            )
        )
        for func in program.functions.values():
            if func.func_type == ir.FunctionType.Orchestration and func.name == which:
                return codegen.generate_orchestration(program, func).code
        raise AssertionError(f"orchestration function {which} not found")

    @staticmethod
    def _assert_cross_scope_resolves(code: str) -> None:
        """A tensor written inside a manual_scope and read after it must resolve:
        no add_* may name an out-of-scope identifier, the buffer is declared in
        the enclosing scope, and the after-scope reader references it directly
        (issue #1697 — remap to the canonical name, not a per-SSA alias)."""
        assert _out_of_scope_tensor_refs(code) == [], code
        manual_open = code.index("PTO2_SCOPE(PTO2ScopeMode::MANUAL)")
        manual_close = code.index("}", manual_open)
        # The buffer AND the alloc handle backing it are declared in the
        # enclosing scope, ahead of the block — if only ``const Tensor& buf`` were
        # hoisted while ``TaskOutputTensors alloc_0 = ...`` stayed inside, ``buf``
        # would reference an out-of-scope handle and the .cpp would not compile.
        assert code.index("TaskOutputTensors alloc_") < manual_open, code
        assert code.index("alloc_tensors(") < manual_open, code
        decl = code.index("const Tensor& buf = ")
        assert decl < manual_open, code
        # The after-scope consumer reads ``buf`` directly — no const-ref alias
        # is minted for the producer's SSA output.
        assert "add_input(buf)" in code[manual_close:], code
        assert "const Tensor& buf__" not in code, code

    def test_manual_scope_tensor_created_before_read_after(self):
        """Regression for #1697: a tensor created BEFORE a ``pl.manual_scope``,
        written by a submit inside it, and read by a task after it. The output
        previously minted ``const Tensor& buf__ssa_v1 = buf;`` at the deep block
        indent; the after-scope ``add_input(buf__ssa_v1)`` then named an
        out-of-scope identifier and the orchestration ``.cpp`` failed to compile.
        The output is now remapped to read ``buf`` directly."""
        self._assert_cross_scope_resolves(self._manual_cross_scope_code(create_inside=False))

    def test_manual_scope_tensor_created_inside_read_after(self):
        """Companion to #1697: the same after-scope read when the
        ``pl.create_tensor`` is INSIDE the manual scope. Its ``alloc_tensors``
        declaration (a storage reservation with no scheduling dependency) is
        hoisted to the enclosing scope, so the after-scope reader still resolves
        ``buf``."""
        self._assert_cross_scope_resolves(self._manual_cross_scope_code(create_inside=True))

    @staticmethod
    def _manual_scope_loop_carry_code(fresh_carry: bool) -> str:
        """Orchestration code for a tensor carried through a ``pl.range`` loop
        *inside* a ``pl.manual_scope`` and read by a task after the scope
        (issue #1713). The loop body submits a windowed (sub-region) write of the
        carried tensor, so OptimizeOrchTensors Pattern-5 externalizes it
        (``produce__windowed`` + ``.view`` slicing).

        ``fresh_carry`` selects which lowering shape the loop carry takes:
          * False — the carry threads the before-scope tensor in place, so the
            post-loop ``score = score_rv`` rebind lowers to a catch-all
            ``Tensor score__ssa_v1 = score;`` copy emitted at the deep block
            indent; the after-scope ``pl.reshape`` reader then named the
            out-of-scope ``score__ssa_v1``.
          * True — the loop yields a freshly created tensor each iteration
            (``is_rebind``), so codegen mints a mutable carry ``Tensor acc_rv =
            acc;`` *inside* the block AND a chained ``acc__ssa_v1 = acc_rv;``
            post-loop copy. The after-scope kernel read named the out-of-scope
            chain. The carry decl is now hoisted out and the copy collapses.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N, M, W = 64, 512, 64

        @pl.program
        class LoopCarryProgram:
            @pl.function(type=pl.FunctionType.AIV, attrs={"windowize": True})
            def produce(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                col: pl.Scalar[pl.INDEX],
                score: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                t: pl.Tile[[N, W], pl.FP32] = pl.load(x, [0, col], [N, W])
                r: pl.Tensor[[N, M], pl.FP32] = pl.store(t, [0, col], score)
                return r

            @pl.function(type=pl.FunctionType.AIV)
            def consume(
                self,
                score: pl.Tensor[[N, M], pl.FP32],
                out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                t: pl.Tile[[N, M], pl.FP32] = pl.load(score, [0, 0], [N, M])
                r: pl.Tensor[[N, M], pl.FP32] = pl.store(t, [0, 0], out)
                return r

            @pl.function(type=pl.FunctionType.Orchestration)
            def main_inplace(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                score: pl.Tensor[[N, M], pl.FP32] = pl.create_tensor([N, M], dtype=pl.FP32)
                with pl.manual_scope():
                    for col, (score_iter,) in pl.range(0, M, W, init_values=(score,)):
                        score_next: pl.Tensor[[N, M], pl.FP32] = self.produce(x, col, score_iter)
                        (score_rv,) = pl.yield_(score_next)
                    score = score_rv
                # After-scope read via a method receiver (pl.reshape) — the shape
                # an ``add_*``-only scope check would miss.
                score_flat: pl.Tensor[[N, M], pl.FP32] = pl.reshape(score, [N, M])
                out = self.consume(score_flat, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main_fresh(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                seed: pl.Tensor[[N, M], pl.FP32],
                out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                acc: pl.Tensor[[N, M], pl.FP32] = pl.create_tensor([N, M], dtype=pl.FP32)
                with pl.manual_scope():
                    for col, (acc_iter,) in pl.range(0, M, W, init_values=(acc,)):
                        # A fresh per-iteration tensor makes the yield value not
                        # alias the carry -> a true rebind -> mutable carry decl.
                        fresh: pl.Tensor[[N, M], pl.FP32] = pl.create_tensor([N, M], dtype=pl.FP32)
                        fresh2: pl.Tensor[[N, M], pl.FP32] = self.produce(x, col, fresh)
                        (acc_rv,) = pl.yield_(fresh2)
                    acc = acc_rv
                out = self.consume(acc, out)
                return out

        which = "main_fresh" if fresh_carry else "main_inplace"
        program = passes.materialize_runtime_scopes()(
            passes.derive_call_directions()(
                PassManager.get_strategy(OptimizationStrategy.Default).run_passes(LoopCarryProgram)
            )
        )
        for func in program.functions.values():
            if func.func_type == ir.FunctionType.Orchestration and func.name == which:
                return codegen.generate_orchestration(program, func).code
        raise AssertionError(f"orchestration function {which} not found")

    def test_manual_scope_loop_carry_read_after_reshape(self):
        """Regression for #1713: a tensor carried through a ``pl.range`` loop
        inside a ``pl.manual_scope`` and read after the scope via ``pl.reshape``
        (a method-receiver use the old ``add_*``-only scope checker missed).

        The post-loop ``score = score_rv`` rebind lowered to a catch-all
        ``Tensor score__ssa_v1 = score;`` copy at the deep block indent; the
        after-scope ``score_flat = score__ssa_v1.reshape(...)`` then named an
        out-of-scope identifier and the ``.cpp`` failed to C++-compile. The copy
        is now collapsed onto the enclosing ``score``."""
        code = self._manual_scope_loop_carry_code(fresh_carry=False)
        # No identifier — including a ``.reshape`` receiver — names an
        # out-of-scope name.
        assert _out_of_scope_tensor_refs(code) == [], code
        # The post-loop rebind collapsed: the after-scope reshape reads the
        # enclosing ``score`` directly, never a scope-local ``score__ssa_v<N>``.
        assert re.search(r"=\s*score\.reshape\(", code), code
        assert not re.search(r"score__ssa_v\d+\s*\.reshape", code), code

    def test_manual_scope_fresh_loop_carry_chained_read_after(self):
        """Regression for #1713: a *fresh-rebind* loop carry inside a
        ``pl.manual_scope`` (the loop yields a freshly created tensor each
        iteration, so OptimizeOrchTensors Pattern-5 externalizes the windowed
        write) read by a kernel after the scope. Codegen minted a mutable carry
        ``Tensor acc_rv = acc;`` inside the block plus a chained
        ``Tensor acc__ssa_v1 = acc_rv;`` post-loop copy; the after-scope
        ``add_input`` named the out-of-scope chain.

        The carry decl is now hoisted to the enclosing scope and the chained copy
        collapses onto it, so the reader resolves a single enclosing name."""
        code = self._manual_scope_loop_carry_code(fresh_carry=True)
        assert _out_of_scope_tensor_refs(code) == [], code
        # The windowed externalization actually fired (exercises Pattern-5).
        assert "produce__windowed" in code, code
        manual_open = code.index("PTO2_SCOPE(PTO2ScopeMode::MANUAL)")
        # The mutable carry for ``acc`` (``Tensor acc_rv = acc;``) is hoisted AHEAD
        # of the manual block header (declared in the enclosing scope). Anchor the
        # match to the ``acc`` carry initialised from ``acc`` so an unrelated
        # ``_rv`` temp emitted earlier can't be picked by mistake.
        carry_decls = list(
            re.finditer(r"^\s*Tensor\s+(acc\w*_rv\w*)\s*=\s*acc;", code[:manual_open], flags=re.MULTILINE)
        )
        assert len(carry_decls) == 1, code[:manual_open]
        carry_name = carry_decls[0].group(1)
        # The after-scope kernel reads the hoisted carry directly; the chained
        # ``acc__ssa_v<N>`` post-loop copy collapsed away entirely.
        assert f"add_input({carry_name})" in code, code
        assert "acc__ssa_v" not in code, code

    def test_manual_scope_loop_carry_not_hoisted_outside_manual(self):
        """Negative control for #1713: an identical loop carry NOT inside a
        ``pl.manual_scope`` keeps its in-place ``Tensor <carry> = <init>;`` decl
        (the hoist is gated on a manual-scope body). Guards against the hoist
        firing in ordinary AUTO-scope codegen."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N, M, W = 64, 512, 64

        @pl.program
        class NoManualScopeProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def produce(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                col: pl.Scalar[pl.INDEX],
                acc: pl.Tensor[[N, M], pl.FP32],
                score: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                t: pl.Tile[[N, W], pl.FP32] = pl.load(x, [0, col], [N, W])
                r: pl.Tensor[[N, M], pl.FP32] = pl.store(t, [0, col], score)
                return r

            @pl.function(type=pl.FunctionType.AIV)
            def consume(
                self,
                score: pl.Tensor[[N, M], pl.FP32],
                out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                t: pl.Tile[[N, M], pl.FP32] = pl.load(score, [0, 0], [N, M])
                r: pl.Tensor[[N, M], pl.FP32] = pl.store(t, [0, 0], out)
                return r

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                seed: pl.Tensor[[N, M], pl.FP32],
                out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                acc: pl.Tensor[[N, M], pl.FP32] = pl.create_tensor([N, M], dtype=pl.FP32)
                for col, (acc_iter,) in pl.range(0, M, W, init_values=(acc,)):
                    fresh: pl.Tensor[[N, M], pl.FP32] = pl.create_tensor([N, M], dtype=pl.FP32)
                    fresh2: pl.Tensor[[N, M], pl.FP32] = self.produce(x, col, acc_iter, fresh)
                    (acc_rv,) = pl.yield_(fresh2)
                out = self.consume(acc_rv, out)
                return out

        code = _generate_orch_code(NoManualScopeProgram)
        assert _out_of_scope_tensor_refs(code) == [], code
        # No manual scope present -> no hoist machinery engaged; the mutable carry
        # decl stays in place (a `Tensor <carry> = <init>;` exists) and is
        # reassigned in the loop.
        assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" not in code, code
        assert re.search(r"^\s*Tensor\s+\w*_rv\w*\s*=\s*\w+;", code, flags=re.MULTILINE), code

    def test_manual_scope_windowed_submit_read_after_reshape(self):
        """Regression for #1713 (the issue's headline shape): a tensor created
        BEFORE a ``pl.manual_scope``, written INSIDE it by a ``pl.submit`` whose
        callee writes a param-offset sub-window in a loop (so OptimizeOrchTensors
        Pattern-5 externalizes it into ``produce__windowed`` + ``score.view(...)``
        + ``tensor.assemble``), and read AFTER the scope via ``pl.reshape``.

        The assemble result's SSA rebind lowered to ``Tensor score__ssa_v1 =
        score;`` at the deep block indent; the after-scope ``score__ssa_v1.reshape
        (...)`` named an out-of-scope identifier (``'<name>__ssa_v<N>' was not
        declared in this scope``). The copy now collapses onto the enclosing
        ``score``."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N, M, W = 64, 2048, 8

        @pl.program
        class WindowedSubmitProgram:
            @pl.function(type=pl.FunctionType.AIV, attrs={"windowize": True})
            def produce(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                base: pl.Scalar[pl.INDEX],
                score: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                # Internal loop writes a contiguous sub-window [base, base+4W) of
                # `score`, so the callee is windowable at the orchestration site.
                for c, (score_iter,) in pl.range(base, base + 4 * W, W, init_values=(score,)):
                    tile: pl.Tile[[N, W], pl.FP32] = pl.load(x, [0, c], [N, W])
                    score_next: pl.Tensor[[N, M], pl.FP32] = pl.store(tile, [0, c], score_iter)
                    (score_rv,) = pl.yield_(score_next)
                return score_rv

            @pl.function(type=pl.FunctionType.AIV)
            def consume(
                self,
                score: pl.Tensor[[N, M], pl.FP32],
                out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                tile: pl.Tile[[N, W], pl.FP32] = pl.load(score, [0, 0], [N, W])
                ret: pl.Tensor[[N, M], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                score: pl.Tensor[[N, M], pl.FP32] = pl.create_tensor([N, M], dtype=pl.FP32)
                with pl.manual_scope():
                    score, _tid = pl.submit(self.produce, x, 0, score)
                score_flat: pl.Tensor[[N, M], pl.FP32] = pl.reshape(score, [N, M])
                out = self.consume(score_flat, out)
                return out

        program = passes.materialize_runtime_scopes()(
            passes.derive_call_directions()(
                PassManager.get_strategy(OptimizationStrategy.Default).run_passes(WindowedSubmitProgram)
            )
        )
        code = next(
            codegen.generate_orchestration(program, f).code
            for f in program.functions.values()
            if f.func_type == ir.FunctionType.Orchestration and f.name == "main"
        )
        # Windowing fired (the test exercises the Pattern-5 ``.view()`` path).
        assert "produce__windowed" in code and ".view(" in code, code
        # The after-scope ``.reshape`` reader resolves: no out-of-scope name, and
        # the windowed-assemble SSA rebind collapsed onto the enclosing ``score``.
        assert _out_of_scope_tensor_refs(code) == [], code
        assert re.search(r"=\s*score\.reshape\(", code), code
        assert not re.search(r"score__ssa_v\d+\s*\.reshape", code), code

    def test_manual_scope_in_loop_carry_copy_keeps_snapshot(self):
        """Snapshot-safety guard for the #1713 collapse: a bare copy of a loop
        carry taken INSIDE the loop body (a deeper indent than the manual-scope
        body) must NOT collapse onto the hoisted carry — otherwise a reader of
        the copy placed before the loop's yield rebind would alias the carry's
        later value. The copy keeps a distinct ``Tensor snap = <carry>;`` decl.

        (The post-loop ``acc = acc_rv`` rebind, at the manual-scope body indent,
        still collapses — exercised by the other #1713 tests; this guards the
        indent condition that distinguishes the two.)"""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N, M, W = 64, 512, 64

        @pl.program
        class SnapshotProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def produce(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                col: pl.Scalar[pl.INDEX],
                s: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                t: pl.Tile[[N, W], pl.FP32] = pl.load(x, [0, col], [N, W])
                return pl.store(t, [0, col], s)

            @pl.function(type=pl.FunctionType.AIV)
            def snapshot_use(
                self,
                snap: pl.Tensor[[N, M], pl.FP32],
                o: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                t: pl.Tile[[N, M], pl.FP32] = pl.load(snap, [0, 0], [N, M])
                return pl.store(t, [0, 0], o)

            @pl.function(type=pl.FunctionType.AIV)
            def consume(
                self,
                s: pl.Tensor[[N, M], pl.FP32],
                out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                t: pl.Tile[[N, M], pl.FP32] = pl.load(s, [0, 0], [N, M])
                return pl.store(t, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                side: pl.Out[pl.Tensor[[N, M], pl.FP32]],
                out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                acc: pl.Tensor[[N, M], pl.FP32] = pl.create_tensor([N, M], dtype=pl.FP32)
                with pl.manual_scope():
                    for col, (acc_iter,) in pl.range(0, M, W, init_values=(acc,)):
                        fresh: pl.Tensor[[N, M], pl.FP32] = pl.create_tensor([N, M], dtype=pl.FP32)
                        snap: pl.Tensor[[N, M], pl.FP32] = acc_iter  # in-loop copy of the carry
                        acc_next: pl.Tensor[[N, M], pl.FP32] = self.produce(x, col, fresh)
                        side = self.snapshot_use(snap, side)  # read snap before the yield rebind
                        (acc_rv,) = pl.yield_(acc_next)
                    acc = acc_rv
                out = self.consume(acc, out)
                return out

        code = next(
            codegen.generate_orchestration(program, f).code
            for program in [
                passes.materialize_runtime_scopes()(
                    passes.derive_call_directions()(
                        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(SnapshotProgram)
                    )
                )
            ]
            for f in program.functions.values()
            if f.func_type == ir.FunctionType.Orchestration and f.name == "main"
        )
        assert _out_of_scope_tensor_refs(code) == [], code
        # The in-loop snapshot is materialised as its own ``Tensor snap = ...;``
        # value and read as ``add_input(snap)`` — NOT collapsed onto the carry,
        # so the later ``<carry> = ...;`` yield rebind cannot change what the
        # snapshot reader sees.
        assert re.search(r"^\s*Tensor\s+snap\s*=\s*\w+;", code, flags=re.MULTILINE), code
        assert "add_input(snap)" in code, code

    def test_manual_scope_ifstmt_phi_read_after(self):
        """Regression for #1713 (IfStmt phi sibling of the loop-carry hoist): an
        ``if`` inside a ``pl.manual_scope`` that conditionally rewrites a tensor
        produces a phi placeholder ``Tensor <buf>__phi_v<N> = <init>;`` at the
        block indent (reassigned in each branch); a task after the scope reading
        the phi then named an out-of-scope identifier.

        The phi decl is now hoisted to the enclosing scope (its init is the
        enclosing param/buffer), so the after-scope reader resolves it."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N, M = 16, 256

        @pl.program
        class IfPhiProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def producer(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                buf: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                t: pl.Tile[[N, M], pl.FP32] = pl.load(x, [0, 0], [N, M])
                return pl.store(t, [0, 0], buf)

            @pl.function(type=pl.FunctionType.AIV)
            def consumer(
                self,
                buf: pl.Tensor[[N, M], pl.FP32],
                out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                t: pl.Tile[[N, M], pl.FP32] = pl.load(buf, [0, 0], [N, M])
                return pl.store(t, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                flag: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                buf: pl.Tensor[[N, M], pl.FP32] = pl.create_tensor([N, M], dtype=pl.FP32)
                with pl.manual_scope():
                    if flag > 0:
                        buf, _t = pl.submit(self.producer, x, buf)
                out = self.consumer(buf, out)
                return out

        program = passes.materialize_runtime_scopes()(
            passes.derive_call_directions()(
                PassManager.get_strategy(OptimizationStrategy.Default).run_passes(IfPhiProgram)
            )
        )
        code = next(
            codegen.generate_orchestration(program, f).code
            for f in program.functions.values()
            if f.func_type == ir.FunctionType.Orchestration and f.name == "main"
        )
        # The IfStmt actually produced a Tensor phi placeholder (exercises the path).
        assert re.search(r"Tensor\s+\w+__phi_v\d+\s*=", code), code
        # No identifier names an out-of-scope name (including the after-scope read).
        assert _out_of_scope_tensor_refs(code) == [], code
        # The phi decl is hoisted AHEAD of the manual block header; the branch
        # ``<phi> = ...;`` merges stay inside the block (resolving through the
        # enclosing frame).
        manual_open = code.index("PTO2_SCOPE(PTO2ScopeMode::MANUAL)")
        phi_decl = re.search(r"^\s*Tensor\s+(\w+__phi_v\d+)\s*=", code[:manual_open], flags=re.MULTILINE)
        assert phi_decl, code[:manual_open]
        phi_name = phi_decl.group(1)
        # The after-scope consumer reads the hoisted phi directly (in scope).
        assert f"add_input({phi_name})" in code, code

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
            PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs& orch_args) {
                (void)orch_args;
                return PTO2OrchestrationConfig{
                    .expected_arg_count = 7,
                };
            }

            __attribute__((visibility("default")))
            void aicpu_orchestration_entry(const L2TaskArgs& orch_args) {
                // External tensors
                const Tensor& ext_mij = orch_args.tensor(0).ref();
                const Tensor& ext_lij = orch_args.tensor(1).ref();
                const Tensor& ext_oi_new = orch_args.tensor(2).ref();
                const Tensor& ext_mi = orch_args.tensor(3).ref();
                const Tensor& ext_li = orch_args.tensor(4).ref();
                const Tensor& ext_oi = orch_args.tensor(5).ref();
                const Tensor& ext_dst = orch_args.tensor(6).ref();

                PTO2_SCOPE() {

                    // Task 0: online_update
                    L0TaskArgs params_t0;
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


class TestOrchestrationOutputDeclaration:
    """The orchestration signature is built from the entry's declared
    ``ParamDirection``s, so an output a kernel writes into an entry parameter must
    be marked ``pl.Out`` / ``pl.InOut`` — otherwise it stays ``In``, is marked IN
    in the signature, and the runtime skips its D2H copy-back (silent all-zero
    output). Codegen logs a non-fatal warning when the entry declares no output
    parameter at all, but compilation still proceeds (returning a runtime-allocated
    output is a legitimate no-output-param pattern).
    """

    def test_no_output_param_is_non_fatal(self) -> None:
        # An entry whose only tensor params are declared read-only (plain
        # pl.Tensor) still compiles — the missing-output-param warning is
        # non-fatal. The declared directions flow straight into the signature.
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class NoOutputParamProgram:
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
            def orch_return_output(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                c: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                c = self.kernel_add(a, b, c)
                return c

        # No exception — the missing-output-param check is a non-fatal warning.
        result = _generate_orch_result(NoOutputParamProgram)
        assert list(result.orchestration_signature) == ["IN", "IN"]

    def test_written_param_declared_out_ok(self) -> None:
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class GoodProgram:
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
            def orch_good(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.kernel_add(a, b, d)
                return d

        result = _generate_orch_result(GoodProgram)
        # a, b are read-only inputs; d is the written output.
        assert list(result.orchestration_signature) == ["IN", "IN", "OUT"]

    def test_read_only_input_not_flagged(self) -> None:
        # A read-only param is fine as plain pl.Tensor even when consumed by a
        # kernel — only Out/InOut kernel args trigger the write check.
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ReadOnlyProgram:
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
            def orch_ro(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.kernel_add(a, b, d)
                return d

        # No exception: a, b (plain pl.Tensor, only read) are not flagged.
        result = _generate_orch_result(ReadOnlyProgram)
        assert list(result.orchestration_signature) == ["IN", "IN", "OUT"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

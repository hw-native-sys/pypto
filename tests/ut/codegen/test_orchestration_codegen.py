# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for orchestration code generation, including tuple return value handling."""

import pypto.language as pl
from pypto import backend
from pypto.backend import BackendType
from pypto.pypto_core import codegen


class TestOrchestration:
    """Test orchestration codegen format."""

    def test_basic_structure(self):
        """Test codegen produces PTO2 format: make_tensor_external, PTOParam, pto2_rt_submit_task."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class BasicProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_basic(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                c: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)
                d: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(c, b)
                return d

        generator = codegen.CCECodegen()
        files = generator.generate(BasicProgram)
        code = files["orchestration/orch_basic.cpp"]

        # Includes
        assert "#include <stddef.h>" in code
        assert "#include <stdint.h>" in code
        assert "#include <stdio.h>" in code
        assert '#include "pto_orchestration_api.h"' in code

        # ARG defines for params (a, b) and return (d)
        assert "#define ARG_PTR_A 0" in code
        assert "#define ARG_PTR_B 1" in code
        assert "#define ARG_PTR_D 2" in code
        assert "#define ARG_SIZE_A" in code
        assert "#define ARG_SIZE_B" in code
        assert "#define ARG_SIZE_D" in code

        # Helper function
        assert "float_to_u64" in code

        # Config function
        assert "aicpu_orchestration_config" in code
        assert "PTO2OrchestrationConfig" in code
        assert "expected_arg_count" in code

        # Entry function signature
        assert "aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count)" in code
        assert 'extern "C"' in code

        # Argument extraction
        assert "arg_a_ptr" in code
        assert "arg_b_ptr" in code
        assert "arg_d_ptr" in code

        # External tensors (params + return)
        assert "Tensor ext_a = make_tensor_external(arg_a_ptr, size_a)" in code
        assert "Tensor ext_b = make_tensor_external(arg_b_ptr, size_b)" in code
        assert "Tensor ext_d = make_tensor_external(arg_d_ptr, size_d)" in code

        # Intermediate tensor c: make_tensor with byte size
        assert "make_tensor(" in code
        assert "16 * 16 * 4" in code

        # PTOParam arrays and task submission
        assert "PTOParam" in code
        assert "make_input_param" in code
        assert "make_output_param" in code
        assert "pto2_rt_submit_task" in code
        assert "PTO2_WORKER_VECTOR" in code

        # PTO2_SCOPE: task 1 depends on intermediate c, so it goes in inner scope
        assert "PTO2_SCOPE(rt)" in code

        # Should NOT have V1 constructs
        assert "add_successor" not in code
        assert "add_task" not in code
        assert "Runtime* runtime" not in code
        assert "device_malloc" not in code
        assert "copy_to_device" not in code

    def test_tensor_read(self):
        """Test tensor.read uses arg_<name>_ptr."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class TensorReadProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_read(
                self,
                t: pl.Tensor[[4, 8], pl.FP32],
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                val: pl.Scalar[pl.FP32] = pl.tensor.read(t, [1, 3])  # noqa: F841
                result: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)
                return result

        generator = codegen.CCECodegen()
        files = generator.generate(TensorReadProgram)
        code = files["orchestration/orch_read.cpp"]

        # tensor.read uses arg_t_ptr, not host_t
        assert "idx_val" in code
        assert "static_cast<float*>(arg_t_ptr)" in code
        assert "host_t" not in code

    def test_config_file(self):
        """Test kernel_config.py is generated."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class ConfigProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_cfg(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                c: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)
                return c

        generator = codegen.CCECodegen()
        files = generator.generate(ConfigProgram)

        assert "kernel_config.py" in files
        config = files["kernel_config.py"]
        assert "aicpu_orchestration_entry" in config
        assert "kernel_add" in config

    def test_independent_tasks(self):
        """Test codegen with independent tasks (no dependencies needed)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class IndependentProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_indep(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                c: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)
                d: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)
                return c, d

        generator = codegen.CCECodegen()
        files = generator.generate(IndependentProgram)
        code = files["orchestration/orch_indep.cpp"]

        # Two return tensors: c and d are both external
        assert "ext_c" in code
        assert "ext_d" in code
        assert "make_tensor_external" in code

        # Two tasks submitted
        assert code.count("pto2_rt_submit_task") == 2

        # No PTO2_SCOPE needed: all tasks use only external tensors
        assert "PTO2_SCOPE" not in code

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
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class VectorExampleProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add_scalar(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                scalar: pl.Scalar[pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                x: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(x, scalar)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_mul(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.mul(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_vector(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                c: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)
                d: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add_scalar(c, 1.0)  # type: ignore[reportArgumentType]
                e: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add_scalar(c, 2.0)  # type: ignore[reportArgumentType]
                g: pl.Tensor[[16, 16], pl.FP32] = self.kernel_mul(d, e)
                f: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(g, c)
                return f

        generator = codegen.CCECodegen()
        files = generator.generate(VectorExampleProgram)
        code = files["orchestration/orch_vector.cpp"]

        # Includes
        assert "#include <stddef.h>" in code
        assert '#include "pto_orchestration_api.h"' in code

        # ARG defines: params (a, b) + return (f)
        assert "#define ARG_PTR_A 0" in code
        assert "#define ARG_PTR_B 1" in code
        assert "#define ARG_PTR_F 2" in code
        assert "#define ARG_SIZE_A 3" in code
        assert "#define ARG_SIZE_B 4" in code
        assert "#define ARG_SIZE_F 5" in code

        # Config
        assert "aicpu_orchestration_config" in code
        assert ".expected_arg_count = 6" in code

        # Entry function
        assert "aicpu_orchestration_entry(PTO2Runtime* rt" in code

        # External tensors: a, b (params) + f (return)
        assert "Tensor ext_a = make_tensor_external(arg_a_ptr, size_a)" in code
        assert "Tensor ext_b = make_tensor_external(arg_b_ptr, size_b)" in code
        assert "Tensor ext_f = make_tensor_external(arg_f_ptr, size_f)" in code

        # 4 intermediate tensors: c, d, e, g (all make_tensor)
        for name in ["c", "d", "e", "g"]:
            assert f"Tensor {name} = make_tensor(" in code, f"Missing make_tensor for {name}"

        # 5 tasks submitted
        assert code.count("pto2_rt_submit_task") == 5

        # Scalar params: kernel_add_scalar uses float_to_u64
        assert "make_scalar_param(float_to_u64(" in code
        assert "1.0" in code
        assert "2.0" in code

        # Three different kernel functions
        assert '"kernel_add"' in code
        assert '"kernel_add_scalar"' in code
        assert '"kernel_mul"' in code
        assert "PTO2_WORKER_VECTOR" in code

        # PTO2_SCOPE: inner scope wraps tasks that depend on intermediate tensors
        assert "PTO2_SCOPE(rt)" in code

        # No V1 constructs
        assert "add_successor" not in code
        assert "add_task" not in code
        assert "device_malloc" not in code

    def test_tuple_intermediate(self):
        """Test tuple return as intermediate tensors: kernel_pair -> kernel_add."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class TupleIntermediateProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_pair(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                out_s: pl.Tensor[[16, 16], pl.FP32],
                out_d: pl.Tensor[[16, 16], pl.FP32],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                s: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                d: pl.Tile[[16, 16], pl.FP32] = pl.sub(a_tile, b_tile)
                rs: pl.Tensor[[16, 16], pl.FP32] = pl.store(s, [0, 0], [16, 16], out_s)
                rd: pl.Tensor[[16, 16], pl.FP32] = pl.store(d, [0, 0], [16, 16], out_d)
                return rs, rd

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_tuple_mid(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                x, y = self.kernel_pair(a, b)
                result: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(x, y)
                return result

        generator = codegen.CCECodegen()
        files = generator.generate(TupleIntermediateProgram)
        code = files["orchestration/orch_tuple_mid.cpp"]

        # Tuple elements x, y are intermediate: make_tensor (not external)
        assert "Tensor x = make_tensor(" in code
        assert "Tensor y = make_tensor(" in code
        assert "16 * 16 * 4" in code

        # Return tensor result is external
        assert "make_tensor_external(arg_result_ptr" in code

        # Two tasks: kernel_pair + kernel_add
        assert code.count("pto2_rt_submit_task") == 2

        # PTO2_SCOPE needed: kernel_add depends on intermediate x, y
        assert "PTO2_SCOPE(rt)" in code

    def test_tuple_output(self):
        """Test tuple return as final output: all elements are external tensors."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class TupleOutputProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_pair(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                out_s: pl.Tensor[[16, 16], pl.FP32],
                out_d: pl.Tensor[[16, 16], pl.FP32],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                s: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                d: pl.Tile[[16, 16], pl.FP32] = pl.sub(a_tile, b_tile)
                rs: pl.Tensor[[16, 16], pl.FP32] = pl.store(s, [0, 0], [16, 16], out_s)
                rd: pl.Tensor[[16, 16], pl.FP32] = pl.store(d, [0, 0], [16, 16], out_d)
                return rs, rd

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_tuple_out(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                x, y = self.kernel_pair(a, b)
                return x, y

        generator = codegen.CCECodegen()
        files = generator.generate(TupleOutputProgram)
        code = files["orchestration/orch_tuple_out.cpp"]

        # Both x and y are return tensors: make_tensor_external
        assert "ext_x" in code
        assert "ext_y" in code
        assert "make_tensor_external(arg_x_ptr" in code
        assert "make_tensor_external(arg_y_ptr" in code

        # Only one task: kernel_pair
        assert code.count("pto2_rt_submit_task") == 1

        # No PTO2_SCOPE needed: single task, all external
        assert "PTO2_SCOPE" not in code

    def test_four_element_tuple(self):
        """Test 4-element tuple unpacking with mixed shapes as intermediate."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class FourTupleProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def online_update(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 16], pl.FP32],
                mi: pl.Tensor[[16, 1], pl.FP32],
                li: pl.Tensor[[16, 1], pl.FP32],
                oi: pl.Tensor[[16, 16], pl.FP32],
                dst: pl.Tensor[[16, 16], pl.FP32],
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
                mi_out: pl.Tensor[[16, 1], pl.FP32] = pl.store(mi_tile, [0, 0], [16, 1], mi)
                li_out: pl.Tensor[[16, 1], pl.FP32] = pl.store(li_tile, [0, 0], [16, 1], li)
                oi_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(oi_tile, [0, 0], [16, 16], oi)
                dst_out: pl.Tensor[[16, 16], pl.FP32] = pl.store(dst_tile, [0, 0], [16, 16], dst)
                return mi_out, li_out, oi_out, dst_out

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
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
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                mi, li, oi, dst = self.online_update(mij, lij, oi_new, mi_in, li_in, oi_in, dst_in)
                final: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(oi, dst)
                return final

        generator = codegen.CCECodegen()
        files = generator.generate(FourTupleProgram)
        code = files["orchestration/orch_four_tuple.cpp"]

        # All 4 tuple elements are intermediate tensors with correct sizes
        # [16, 1] FP32 = 16 * 1 * 4 = 64 bytes
        assert "16 * 1 * 4" in code
        # [16, 16] FP32 = 16 * 16 * 4 = 1024 bytes
        assert "16 * 16 * 4" in code
        for name in ["mi", "li", "oi", "dst"]:
            assert f"Tensor {name} = make_tensor(" in code, f"Missing make_tensor for {name}"

        # Final return tensor is external
        assert "make_tensor_external(arg_final_ptr" in code

        # Two tasks: online_update + kernel_add
        assert code.count("pto2_rt_submit_task") == 2

        # PTO2_SCOPE needed: kernel_add depends on intermediate oi, dst
        assert "PTO2_SCOPE(rt)" in code

    def test_tensor_create(self):
        """Test tensor.create generates make_tensor with correct size."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class TensorCreateProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_fill(
                self,
                a: pl.Tensor[[32, 32], pl.FP16],
                output: pl.Tensor[[32, 32], pl.FP16],
            ) -> pl.Tensor[[32, 32], pl.FP16]:
                t: pl.Tile[[32, 32], pl.FP16] = pl.load(a, [0, 0], [32, 32])
                out: pl.Tensor[[32, 32], pl.FP16] = pl.store(t, [0, 0], [32, 32], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_create(
                self,
                a: pl.Tensor[[32, 32], pl.FP16],
            ) -> pl.Tensor[[32, 32], pl.FP16]:
                buf: pl.Tensor[[32, 32], pl.FP16] = pl.create_tensor([32, 32], dtype=pl.FP16)
                result: pl.Tensor[[32, 32], pl.FP16] = self.kernel_fill(buf)
                return result

        generator = codegen.CCECodegen()
        files = generator.generate(TensorCreateProgram)
        code = files["orchestration/orch_create.cpp"]

        # tensor.create generates make_tensor with byte size
        # FP16 = 2 bytes per element
        assert "32 * 32 * 2" in code
        assert "Tensor buf = make_tensor(" in code

    def test_tensor_dim(self):
        """Test tensor.dim generates int64_t assignment with shape value."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class TensorDimProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_dim(
                self,
                a: pl.Tensor[[64, 128], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                d0: pl.Scalar[pl.INT64] = pl.tensor.dim(a, 0)  # noqa: F841
                result: pl.Tensor[[64, 128], pl.FP32] = self.kernel_add(a, b)
                return result

        generator = codegen.CCECodegen()
        files = generator.generate(TensorDimProgram)
        code = files["orchestration/orch_dim.cpp"]

        # tensor.dim generates int64_t assignment
        assert "int64_t d0 = 64" in code

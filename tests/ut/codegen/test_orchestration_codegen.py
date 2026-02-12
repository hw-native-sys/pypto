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


class TestOrchestrationSingleReturn:
    """Test orchestration codegen with single tensor return (regression)."""

    def test_single_return_basic(self):
        """Test basic orchestration with single tensor return and intermediate tensors."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class SingleReturnProgram:
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
            def orch_single(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                c: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)
                d: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(c, b)
                return d

        generator = codegen.CCECodegen()
        files = generator.generate(SingleReturnProgram)
        code = files["orchestration/orch_single.cpp"]

        # Header and signature
        assert "// Orchestration Function: orch_single" in code
        assert "#include <cstdint>" in code
        assert 'extern "C"' in code
        assert "BuildOrch_single(Runtime* runtime, uint64_t* args, int arg_count)" in code

        # Argument validation
        assert "arg_count < 7" in code
        assert "return -1" in code

        # Input parameter extraction and device memory
        assert "host_a" in code
        assert "host_b" in code
        assert "copy_to_device(dev_a, host_a, size_a)" in code
        assert "copy_to_device(dev_b, host_b, size_b)" in code

        # Output tensor: has host pointer and record_tensor_pair
        assert "host_d" in code
        assert "record_tensor_pair(host_d, dev_d, size_d)" in code

        # Intermediate tensor c: device memory only, no host pointer
        assert "size_c = 16 * 16 * 4" in code
        assert "dev_c" in code
        assert "host_c" not in code

        # Task creation
        assert code.count("add_task") == 2
        assert "CoreType::AIV" in code

        # Data-flow dependency between tasks
        assert "add_successor(t0, t1)" in code
        assert "return 0;" in code


class TestOrchestrationTupleIntermediate:
    """Test orchestration codegen with tuple return as intermediate tensors."""

    def test_tuple_intermediate_basic(self):
        """Test that tuple elements are individually allocated as intermediate tensors."""
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

        # Each tuple element gets individual device memory allocation
        assert "size_x = 16 * 16 * 4" in code
        assert "size_y = 16 * 16 * 4" in code
        assert "dev_x" in code
        assert "dev_y" in code

        # x and y are intermediate (no host pointers)
        assert "host_x" not in code
        assert "host_y" not in code

        # Return tensor has host pointer and record_tensor_pair
        assert "host_result" in code
        assert "record_tensor_pair(host_result, dev_result, size_result)" in code

        # Two tasks: kernel_pair + kernel_add
        assert code.count("add_task") == 2


class TestOrchestrationTupleOutput:
    """Test orchestration codegen with tuple return as final output."""

    def test_tuple_output_basic(self):
        """Test that tuple return values each get host pointers and record_tensor_pair."""
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

        # Both x and y are return tensors - each should have host pointer
        assert "host_x" in code
        assert "host_y" in code
        assert "record_tensor_pair(host_x, dev_x, size_x)" in code
        assert "record_tensor_pair(host_y, dev_y, size_y)" in code

        # Both should have device memory
        assert "dev_x" in code
        assert "dev_y" in code

        # Only one task: kernel_pair
        assert code.count("add_task") == 1

        # No dependencies (single task)
        assert "add_successor" not in code


class TestOrchestrationTupleFourElements:
    """Test orchestration codegen with 4-element tuple (paged attention pattern)."""

    def test_four_element_tuple_intermediate(self):
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
        assert "size_mi = 16 * 1 * 4" in code
        assert "size_li = 16 * 1 * 4" in code
        assert "size_oi = 16 * 16 * 4" in code
        assert "size_dst = 16 * 16 * 4" in code
        for name in ["mi", "li", "oi", "dst"]:
            assert f"dev_{name}" in code, f"Missing dev_{name}"

        # Final return tensor has host pointer
        assert "host_final" in code
        assert "record_tensor_pair(host_final, dev_final, size_final)" in code

        # Two tasks: online_update + kernel_add
        assert code.count("add_task") == 2


class TestOrchestrationDependencies:
    """Test task dependency generation in orchestration codegen."""

    def test_chain_three_tasks(self):
        """Test 3 chained tasks produce 2 add_successor calls."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class ChainProgram:
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
            def orch_chain(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                c: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)
                d: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(c, b)
                e: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(d, b)
                return e

        generator = codegen.CCECodegen()
        files = generator.generate(ChainProgram)
        code = files["orchestration/orch_chain.cpp"]

        # Three tasks
        assert code.count("add_task") == 3

        # Two intermediate tensors (c, d), no host pointers
        assert "host_c" not in code
        assert "host_d" not in code
        assert "dev_c" in code
        assert "dev_d" in code

        # Chain dependency: t0->t1, t1->t2
        assert code.count("add_successor") == 2
        assert "add_successor(t0, t1)" in code
        assert "add_successor(t1, t2)" in code

    def test_independent_tasks(self):
        """Test independent tasks produce no add_successor calls."""
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
            def orch_independent(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                c: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)
                d: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)
                return c, d

        generator = codegen.CCECodegen()
        files = generator.generate(IndependentProgram)
        code = files["orchestration/orch_independent.cpp"]

        # Two tasks, no data dependency between them
        assert code.count("add_task") == 2
        assert "add_successor" not in code

    def test_codegen_metadata(self):
        """Test that kernel_config.py contains func_name_to_id mapping."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class MetadataProgram:
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
            def orch_meta(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                c: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)
                return c

        generator = codegen.CCECodegen()
        files = generator.generate(MetadataProgram)

        # kernel_config.py should contain function-to-id mapping
        assert "kernel_config.py" in files
        config = files["kernel_config.py"]
        assert "kernel_add" in config


class TestOrchestrationTensorOps:
    """Test tensor operations (create, dim, read) in orchestration codegen."""

    def test_tensor_create(self):
        """Test tensor.create generates device_malloc with correct size."""
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

        # tensor.create generates inline size calculation + device_malloc
        # FP16 = 2 bytes per element
        assert "size_buf = 32 * 32 * 2" in code
        assert "device_malloc(size_buf)" in code

        # Created tensor has no host pointer (device-only)
        assert "host_buf" not in code

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

    def test_tensor_read(self):
        """Test tensor.read generates linear index computation and cast."""
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

        # tensor.read generates index computation and typed cast
        assert "idx_val" in code
        assert "static_cast<float*>(host_t)" in code


class TestOrchestrationScalarArgs:
    """Test scalar/constant argument passing in orchestration codegen."""

    def test_tensor_read_as_scalar_arg(self):
        """Test that tensor.read result generates inline scalar code in orchestration."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class ScalarReadProgram:
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
            def orch_read_scalar(
                self,
                t: pl.Tensor[[4, 8], pl.FP32],
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                val: pl.Scalar[pl.FP32] = pl.tensor.read(t, [1, 3])  # noqa: F841
                result: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)
                return result

        generator = codegen.CCECodegen()
        files = generator.generate(ScalarReadProgram)
        code = files["orchestration/orch_read_scalar.cpp"]

        # tensor.read generates inline scalar assignment
        assert "idx_val" in code
        assert "static_cast<float*>(host_t)" in code
        # Linear index for [1, 3] on shape [4, 8]: 1 * 8 + 3
        assert "1 * 8 + 3" in code

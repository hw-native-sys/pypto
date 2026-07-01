# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for orchestration code generation (core cases, part 2)."""

import re

import pypto.language as pl
import pytest
from _orchestration_codegen_common import (
    _generate_orch_code,
)
from pypto import backend, codegen, passes
from pypto.backend import BackendType
from pypto.ir.builder import IRBuilder
from pypto.ir.op import tensor as tensor_ops
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import DataType, ir


class TestOrchestrationMore:
    """Orchestration codegen — additional core cases (continued)."""

    def test_dynamic_gm_pipe_buffer_alloc_follows_its_size_local(self):
        """A dynamically-sized injected GM pipe buffer must not be hoisted above
        the body-local it is sized by (issue #1768).

        A cube matmul feeding a vector op, fused into ONE ``pl.spmd`` scope with
        a dynamic block count, makes ``InjectGMPipeBuffer`` add a placeholder
        ``tensor.create([1])`` whose *real* size is ``slot_size * (m // ROW)``.
        That size references the body-local ``m = orch_args.tensor(0).ref().shapes[0]``.
        The placeholder's IR shape is constant, so the generic hoist guard cannot
        see the dependency; the size-override branch must route the alloc to the
        per-op path so it is emitted after ``m`` rather than at the scope top
        (which produced ``'m' was not declared in this scope``).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        K = 64
        SPMD_N = 16
        ROW_TILE = 16
        M_DYN = pl.dynamic("M_DYN")

        @pl.program
        class DynPipeProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                a: pl.Tensor[[M_DYN, K], pl.FP32],
                b: pl.Tensor[[K, SPMD_N], pl.FP32],
                out: pl.Tensor[[M_DYN, SPMD_N], pl.FP32],
            ) -> pl.Tensor[[M_DYN, SPMD_N], pl.FP32]:
                m = pl.tensor.dim(a, 0)
                for ob in pl.spmd(m // ROW_TILE, name_hint="hc"):
                    m0 = ob * ROW_TILE
                    a_slice = pl.slice(a, [ROW_TILE, K], [m0, 0])
                    a_add = pl.add(a_slice, 1.0)  # vector produces matmul operand (V->C)
                    c_tile = pl.matmul(a_add, b)  # cube
                    c_vec = pl.add(c_tile, 1.0)  # vector consumes matmul result (C->V)
                    out = pl.assemble(out, c_vec, [m0, 0])
                return out

        # VerificationLevel.NONE: a dynamic ``pl.spmd`` block count lands on the
        # outlined Spmd function as a ``core_num`` attr that references a local
        # defined in the *caller* Orchestration function. The printer emits that
        # attr verbatim, but the roundtrip parser rejects the standalone function
        # (the var is out of scope) — a known print/parse gap unrelated to the
        # codegen ordering under test here.
        with passes.PassContext([], passes.VerificationLevel.NONE):
            program = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(DynPipeProgram)
        orch_func = next(
            f for f in program.functions.values() if f.func_type == ir.FunctionType.Orchestration
        )
        code = codegen.generate_orchestration(program, orch_func).code

        # The pipe-buffer size must genuinely be dynamic (references the local),
        # otherwise the test would pass vacuously on a constant size.
        size_decl = re.search(r"gm_pipe_buffer_\w+_ci_shapes\[1\] = \{(.+?)\};", code)
        assert size_decl is not None, code
        assert "/ 16" in size_decl.group(1), size_decl.group(1)

        # The body-local that sizes the buffer must be declared before the alloc.
        m_decl = re.search(r"int64_t (\w+) = \(int64_t\)orch_args\.tensor\(0\)\.ref\(\)\.shapes\[0\];", code)
        assert m_decl is not None, code
        m_name = m_decl.group(1)
        assert m_name in size_decl.group(1), (m_name, size_decl.group(1))
        assert code.index(f"int64_t {m_name} =") < code.index("gm_pipe_buffer"), code

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
            "(chunk_offsets[0] >= ext_data.shapes[0] ? 0u : std::min<uint32_t>(16, ext_data.shapes[0] - chunk_offsets[0])), "  # noqa: E501
            "(chunk_offsets[1] >= ext_data.shapes[1] ? 0u : std::min<uint32_t>(16, ext_data.shapes[1] - chunk_offsets[1]))};"  # noqa: E501
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
            "(chunk_offsets[0] >= ext_data.shapes[0] ? 0u : std::min<uint32_t>(16, ext_data.shapes[0] - chunk_offsets[0])), "  # noqa: E501
            "(chunk_offsets[1] >= ext_data.shapes[1] ? 0u : std::min<uint32_t>(16, ext_data.shapes[1] - chunk_offsets[1]))};"  # noqa: E501
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
            "(chunk_offsets[0] >= ext_data.shapes[0] ? 0u : std::min<uint32_t>(1, ext_data.shapes[0] - chunk_offsets[0])), "  # noqa: E501
            "(chunk_offsets[1] >= ext_data.shapes[1] ? 0u : std::min<uint32_t>(16, ext_data.shapes[1] - chunk_offsets[1])), "  # noqa: E501
            "(chunk_offsets[2] >= ext_data.shapes[2] ? 0u : std::min<uint32_t>(16, ext_data.shapes[2] - chunk_offsets[2]))};"  # noqa: E501
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
        # a_acc is a return value → external (orch_args.tensor(i).ref())
        assert code.count("const Tensor& ext_a_acc = orch_args.tensor(1).ref()") == 1
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
        """Test inplace detection when the return var has compound auto-name suffixes from the pipeline.

        When an Opaque function with a hand-tiled loop nest over the iteration space goes
        through the full pass pipeline (SSA → outline), the return var acquires compound
        suffixes from the nested loop-carried renames (e.g. "__rv_v1"). GetSSABaseName must
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
                with pl.at(level=pl.Level.CORE_GROUP):
                    for rc in pl.range(0, 1024 // 64):
                        for ri in pl.range(0, 64):
                            r = rc * 64 + ri
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
        assert "orch_args.tensor(0).ref()" in code  # input_tensor
        assert "orch_args.tensor(1).ref()" in code  # output_tensor

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
        assert "orch_args.tensor(0).ref()" in code  # x
        assert "orch_args.tensor(1).ref()" in code  # out_0
        assert "orch_args.tensor(2).ref()" in code  # out_1

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
        # ``first`` is kernel_add's in-place output buffer ret0__out, so it
        # remaps to ret0__out (no ``const Tensor& first = ...`` alias is minted);
        # the second call reads that buffer directly.
        assert "params_t1.add_input(ret0__out)" in code
        assert "const Tensor& first" not in code
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
        """Scalar params get L2TaskArgs scalar slots (0-indexed) via from_u64<T>()."""
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
        assert "orch_args.tensor(0).ref()" in code
        assert "orch_args.tensor(1).ref()" in code
        assert "from_u64<int64_t>(orch_args.scalar(0))" in code
        assert "from_u64<int32_t>(orch_args.scalar(1))" in code
        assert "from_u64<float>(orch_args.scalar(2))" in code
        assert ".expected_arg_count = 5," in code

    def test_dump_tag_emits_toggle_and_per_task_dump(self):
        """``pl.dump_tag(t)`` at orchestration scope makes codegen emit a per-task
        ``Arg::dump(...)`` carrying only the tagged tensors. No orch-body toggle
        is emitted (simpler#953): the runtime latches the dump level host-side.

        Two kernel calls both consume ``a`` and ``b``; only ``a`` is tagged,
        so both tasks should dump ``ext_a`` and neither should dump ``ext_b``.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class DumpTagProgram:
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
            def orch_with_tag(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pl.dump_tag(a)
                c: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                c = self.kernel_add(a, b, c)
                d = self.kernel_add(a, b, d)
                return d

        code = _generate_orch_code(DumpTagProgram)

        # No orch-body toggle (simpler#953): the runtime latches the dump level
        # (off / partial / full) host-side; codegen only emits ``.dump(...)``.
        assert "enable_dump_tensor_selective" not in code

        # Both tasks dump only the tagged arg (ext_a), never ext_b.
        assert code.count("params_t0.dump(ext_a);") == 1
        assert code.count("params_t1.dump(ext_a);") == 1
        assert "ext_b" not in [line.strip() for line in code.split("\n") if ".dump(" in line]
        # Stronger check: no dump call references ext_b anywhere.
        for line in code.split("\n"):
            if ".dump(" in line:
                assert "ext_b" not in line, f"Untagged ext_b should not be dumped: {line!r}"

    def test_no_dump_tag_emits_no_toggle_or_dump(self):
        """Without any ``pl.dump_tag`` no ``.dump(...)`` calls are emitted. The
        runtime's ``CallConfig::enable_dump_tensor`` level then drives the dump
        behaviour: level 2 (full) dumps every tensor of every task; level 1
        (partial) dumps only ``.dump(...)``-marked tensors (here: none).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class NoDumpTagProgram:
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
            def orch_plain(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.kernel_add(a, b, d)
                return d

        code = _generate_orch_code(NoDumpTagProgram)

        assert "enable_dump_tensor_selective" not in code
        for line in code.split("\n"):
            assert ".dump(" not in line, f"Plain orch should not emit any dump call: {line!r}"

    def test_dump_tag_with_no_kernel_use_emits_nothing(self):
        """Tagging a Var that no kernel call consumes is a user mistake we
        keep quiet about: zero hits → no ``.dump(...)`` call emitted. The orch
        falls back to whatever the runtime dump level dictates.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class StrayTagProgram:
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
            def orch_stray(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                unused: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pl.dump_tag(unused)  # unused never reaches a kernel call
                d = self.kernel_add(a, b, d)
                return d

        code = _generate_orch_code(StrayTagProgram)

        assert "enable_dump_tensor_selective" not in code
        for line in code.split("\n"):
            assert ".dump(" not in line, f"Stray tag should not emit any dump call: {line!r}"

    def test_dump_tag_inside_inline_function_propagates_to_caller(self):
        """``pl.dump_tag(<inline param>)`` written inside ``@pl.function(type=Inline)``
        desugars to ``dump_vars`` on the inline body's kernel calls; after
        ``InlineFunctions`` splices the body in, the mutator substitutes the
        caller's arg for the inline param, so the dump rides through to the
        inlined call site. Codegen then emits the per-task ``.dump(...)``.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class InlineDumpTagProgram:
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

            @pl.function(type=pl.FunctionType.Inline)
            def inline_helper(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pl.dump_tag(a)
                d = self.kernel_add(a, b, d)
                return d

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.inline_helper(a, b, d)
                return d

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(InlineDumpTagProgram)
        code = _generate_orch_code(transformed)

        assert "enable_dump_tensor_selective" not in code
        assert code.count("params_t0.dump(ext_a);") == 1
        for line in code.split("\n"):
            if ".dump(" in line:
                assert "ext_b" not in line, f"Untagged ext_b should not be dumped: {line!r}"

    def test_dump_tag_on_inline_body_local_var_after_freshname_rename(self):
        """``pl.create_tensor`` inside an inline function is alpha-renamed by
        the inline pass (``FreshName`` appends ``_inline<N>``) and versioned by
        SSA. Because the dump target rides on the call's ``dump_vars`` (a Var
        ref, not a name), it follows the rename / versioning automatically and
        the dump is still emitted for the right slot after inlining.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class InlineBodyVarDumpTagProgram:
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

            @pl.function(type=pl.FunctionType.Inline)
            def inline_helper(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                tmp: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                pl.dump_tag(tmp)
                tmp = self.kernel_add(a, b, tmp)
                d = self.kernel_add(tmp, b, d)
                return d

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.inline_helper(a, b, d)
                return d

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(InlineBodyVarDumpTagProgram)
        code = _generate_orch_code(transformed)

        assert "enable_dump_tensor_selective" not in code
        # The dump rides on the call's ``dump_vars`` Var ref, which follows the
        # FreshName rename (e.g. ``tmp_inline0``) and SSA versioning, so the
        # emitted dump references the renamed local — at least one
        # ``.dump(tmp...);`` line must appear.
        dump_lines = [line for line in code.split("\n") if ".dump(" in line]
        assert dump_lines, f"expected at least one dump call, got code:\n{code}"
        assert any("tmp" in line for line in dump_lines), (
            f"renamed inline body var should be dumped; got dump lines: {dump_lines}"
        )

    def test_dump_tag_stacked_inline_renames_body_local_var(self):
        """A body-local ``pl.create_tensor`` declared in the innermost inline
        function survives several layers of inlining. Each ``InlineFunctions``
        pass iteration appends a fresh ``_inline<N>`` suffix to the Var name,
        so a Var that starts as ``tmp`` can land in the orch body as
        ``tmp_inlineA_inlineB_inlineC``. Because the dump target is a Var ref on
        the call's ``dump_vars`` (not a name), it follows every rename and the
        per-task dump still emits for the right slot.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class StackedInlineProgram:
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

            @pl.function(type=pl.FunctionType.Inline)
            def inner(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                tmp: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                pl.dump_tag(tmp)
                tmp = self.kernel_add(a, b, tmp)
                d = self.kernel_add(tmp, b, d)
                return d

            @pl.function(type=pl.FunctionType.Inline)
            def middle(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.inner(a, b, d)
                return d

            @pl.function(type=pl.FunctionType.Inline)
            def outer(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.middle(a, b, d)
                return d

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.outer(a, b, d)
                return d

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(StackedInlineProgram)
        code = _generate_orch_code(transformed)

        # The Var that started as ``tmp`` in ``inner`` is renamed at every
        # outer inlining step. Confirm we see at least one ``_inline``-
        # suffixed emit name in the generated code (sanity that the multi-
        # level stack happened at all), then assert the dump emit picks it up
        # via the Var ref riding through the renames.
        assert "_inline" in code, "expected at least one inline-renamed Var in the code"
        assert "enable_dump_tensor_selective" not in code
        dump_lines = [line for line in code.split("\n") if ".dump(" in line]
        assert dump_lines, f"expected dump calls after multi-level inline, got code:\n{code}"
        assert any("tmp" in line for line in dump_lines), (
            f"stacked inline rename should still be dumped; got: {dump_lines}"
        )

    def test_dump_tag_two_level_inline_propagates(self):
        """Two-level inlining (orch → middle → inner): the ``dump_vars`` set
        written inside the innermost inline rides on the spliced kernel calls
        through each ``InlineFunctions`` fixpoint iteration, so the per-task
        dump still emits at the orchestration entry.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TwoLevelInlineDumpTagProgram:
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

            @pl.function(type=pl.FunctionType.Inline)
            def inner(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pl.dump_tag(a)
                d = self.kernel_add(a, b, d)
                return d

            @pl.function(type=pl.FunctionType.Inline)
            def middle(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.inner(a, b, d)
                return d

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.middle(a, b, d)
                return d

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(TwoLevelInlineDumpTagProgram)
        code = _generate_orch_code(transformed)

        assert "enable_dump_tensor_selective" not in code
        assert code.count("params_t0.dump(ext_a);") == 1

    def test_mixed_group_emission_order(self):
        """MixedKernels, launch_spec, and set_dependencies must emit in canonical order.

        Canonical order: ``L0TaskArgs → params → dump → MixedKernels → launch_spec → set_dependencies → submit``.
        Guards against unintentional reorder in ``TaskDispatchPlan::Emit``.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class MixedOrderProgram:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_mm = pl.matmul(tile_l0a, tile_l0b)
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

        transformed = passes.expand_mixed_kernel()(
            passes.infer_tile_memory_space()(
                passes.outline_cluster_scopes()(passes.convert_to_ssa()(MixedOrderProgram))
            )
        )
        code = _generate_orch_code(transformed)

        assert "MixedKernels mixed_0" in code, code
        assert "params_t0.launch_spec." in code, code
        assert "rt_submit_task(mixed_0" in code, code

        # MixedKernels must appear after L0TaskArgs (params/dump block ends with
        # add_* calls; MixedKernels comes next in the old canonical order).
        assert "L0TaskArgs params_t0;" in code, code
        l0_index = code.index("L0TaskArgs params_t0;")
        mixed_index = code.index("MixedKernels mixed_0")
        launch_index = code.index("params_t0.launch_spec.")
        submit_index = code.index("rt_submit_task(mixed_0")
        assert l0_index < mixed_index, "MixedKernels must appear after L0TaskArgs"
        assert mixed_index < launch_index, "MixedKernels must appear before launch_spec"
        assert launch_index < submit_index, "launch_spec must appear before submit"

    def test_direct_call_with_deps_emission_order(self):
        """Direct-call path with deps: L0TaskArgs → deps array → set_dependencies → submit.

        The dependent task (params_t1) carries the deps. Guards the canonical
        order for ``pl.submit(..., deps=[...])`` so a future change to
        ``TaskDispatchPlan::Emit`` cannot silently reorder it.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class DepsOrderProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def k(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                x, tid = pl.submit(self.k, x)
                x, _ = pl.submit(self.k, x, deps=[tid])
                return x

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(DepsOrderProgram)
        code = _generate_orch_code(transformed)

        # Task 0 (no deps) and Task 1 (carries deps) must both be present.
        assert "L0TaskArgs params_t0;" in code, code
        assert "params_t1.set_dependencies(" in code, code
        assert "rt_submit_aiv_task(" in code, code

        l0_t0 = code.index("L0TaskArgs params_t0;")
        deps_t1 = code.index("params_t1.set_dependencies(")
        submit_t1 = code.rindex("rt_submit_aiv_task(")
        assert l0_t0 < deps_t1, "L0TaskArgs must appear before set_dependencies"
        assert deps_t1 < submit_t1, "set_dependencies must appear before submit"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

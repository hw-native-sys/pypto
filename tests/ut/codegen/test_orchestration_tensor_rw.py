# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor read/write offset orchestration-codegen tests."""

import re

import pypto.language as pl
import pytest
from _orchestration_codegen_common import (
    _generate_orch_code,
    _generate_orch_result,
)
from pypto import backend, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import ir


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
        """OutWindowExternalizer tuple outputs must rebind loop carries instead of redeclaring them."""

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class WindowedTupleLoopCarryProgram:
            @pl.function(type=pl.FunctionType.InCore, attrs={"windowize": True})
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

    def test_windowed_writer_before_full_parent_reader_exposes_windowed_writer(self):
        """Window writers may stay windowed before a later full-parent reader.

        The old #1468 guard disabled this producer-side windowing shape because
        #1444 exposed a runtime TensorMap overlap bug. Runtime overlap is fixed
        by simpler#808, so Pattern 5 should expose the precise writer window and
        leave the later global reader as a full tensor input.
        """

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N, M, W = 64, 2048, 8

        @pl.program
        class WindowedWriteFullParentReadProgram:
            @pl.function(type=pl.FunctionType.InCore, attrs={"windowize": True})
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
                fence: pl.Tile[[1, M], pl.FP32] = pl.tile.load(score, [0, 0], [1, M], [1, M])
                merged: pl.Tile[[1, M], pl.FP32] = pl.tile.add(tile, fence)
                ret: pl.Tensor[[N, M], pl.FP32] = pl.tile.store(merged, [row, 0], probe)
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

        assert "produce__windowed" in code, code
        assert "params_t0.add_inout(score_iter)" in code, code
        assert "params_t1.add_input(score_flat)" in code, code
        assert "score_flat.view(" in code, code

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
                with pl.at(level=pl.Level.CORE_GROUP):
                    a_plus_b = pl.add(a, b)
                    sub = pl.sub(a, b)
                    result = pl.matmul(a_plus_b, sub)
                    out = pl.assemble(out, result, [0, 0])
                return out

        transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(NoSplitGroupProgram)

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

    def test_spmd_multi_out_return_value_aliases_actual_output(self):
        """Multi-output SPMD scope whose return value is the LAST output must alias
        to that output, never the first Out/InOut arg (issue #1702).

        The kernel writes three GM tensors (two InOut + one Out, the Out last and
        returned). After cluster outlining, the Spmd wrapper's single return must
        be traced to the ``out`` param; pre-#1702 the wrapper's return value was a
        TupleGetItem of the inner call, the lineage trace failed and
        ``GenerateSingleReturnAlias`` aliased to ``out_indices[0]`` (= ``pre``).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class SpmdMultiOutReturnLast:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                pre: pl.InOut[pl.Tensor[[64, 64], pl.FP32]],
                post: pl.InOut[pl.Tensor[[64, 64], pl.FP32]],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[64, 64], pl.FP32],
                pl.Tensor[[64, 64], pl.FP32],
                pl.Tensor[[64, 64], pl.FP32],
            ]:
                tile = pl.load(a, [0, 0], [64, 64])
                pre = pl.store(tile, [0, 0], pre)
                post = pl.store(tile, [0, 0], post)
                tile_out = pl.add(tile, tile)
                out = pl.store(tile_out, [0, 0], out)
                return pre, post, out

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
                pre: pl.InOut[pl.Tensor[[64, 64], pl.FP32]],
                post: pl.InOut[pl.Tensor[[64, 64], pl.FP32]],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                final: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.spmd(4):
                    pre, post, out = self.kernel(a, pre, post, out)
                final = self.consumer(out, final)
                return final

        # VerificationLevel.NONE: tuple destructuring inside `with pl.spmd(N):`
        # desugars to a multi-statement body the printer emits verbatim but the
        # parser rejects (same known roundtrip gap as test_spmd_multi_assemble).
        with passes.PassContext([], passes.VerificationLevel.NONE):
            transformed = passes.expand_mixed_kernel()(
                passes.infer_tile_memory_space()(
                    passes.outline_cluster_scopes()(passes.convert_to_ssa()(SpmdMultiOutReturnLast))
                )
            )

        code = _generate_orch_code(transformed)

        # The downstream consumer must read the actually-returned output
        # (``ext_out``), never the first InOut (``ext_pre``) or ``ext_post``.
        consumer_input_lines = [line for line in code.splitlines() if "params_t1.add_input" in line]
        assert consumer_input_lines, f"Expected a consumer task reading the SPMD result, got:\n{code}"
        first_consumer_input = consumer_input_lines[0]
        assert "ext_out" in first_consumer_input, (
            "Consumer of the SPMD scope result must read the returned Out param (ext_out). "
            f"Got: {first_consumer_input}\n\nFull code:\n{code}"
        )
        for wrong in ("ext_pre", "ext_post"):
            assert wrong not in first_consumer_input, (
                f"Consumer reads {wrong} — multi-output return alias bug (#1702):\n"
                f"{first_consumer_input}\n\nFull code:\n{code}"
            )

        # Any explicit SSA alias for the result must bind to ext_out as well.
        alias_lines = [line for line in code.splitlines() if line.lstrip().startswith("const Tensor& out")]
        for line in alias_lines:
            assert "ext_pre" not in line and "ext_post" not in line, (
                f"SPMD return alias bound to the wrong output:\n{line}\n\nFull code:\n{code}"
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

        # NOTE: bypass tracks a known print->parse round-trip limitation — a
        # multi-output `out0, out1 = self.kernel(...)` inside `with pl.spmd(N):`
        # desugars to a 3-statement body the printer emits verbatim, which the
        # parser then rejects (spmd body must be a single statement). The IR is
        # valid (passes BEFORE_AND_AFTER property verification); only roundtrip
        # fails. Remove NONE once the printer/parser round-trips this shape.
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

        transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
            PerCalleeGMPipeProgram
        )

        code = _generate_orch_code(transformed)
        shape_values = re.findall(r"gm_pipe_buffer_\d+_ci_shapes\[1\]\s*=\s*\{(\d+)\};", code)
        assert shape_values == ["1024", "6144"], (
            "Expected per-callee GM workspace shapes (small=512*8*1 side / f32, "
            f"large=(1024*8+2048*8) / f32), got {shape_values}. Generated code:\n{code}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

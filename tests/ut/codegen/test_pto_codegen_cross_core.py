# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
910B PTO Backend: Cross-Core Communication (TPUSH/TPOP) Codegen Test.

This test validates code generation for the complete TPUSH/TPOP cross-core
communication protocol. Elementwise ops (add, exp) run on Vector cores, while
matmul runs on Cube cores — matching the hardware architecture.

Protocol under test (V2C unidirectional):
  1. Vector (producer): load + add, tpush_to_aic → Cube
  2. Cube (consumer):   tpop_from_aiv, matmul, store

Bidirectional:
  1. Vector → Cube (V2C): Vector preprocesses (add), pushes to Cube
  2. Cube → Vector (C2V): Cube does matmul, pushes result back to Vector for post-processing
"""

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType

# ============================================================================
# Test Program: Vector Producer + Cube Consumer (V2C unidirectional)
# ============================================================================


@pl.program
class CrossCoreTpushTpopProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def vector_producer(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
    ):
        v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
        pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=v2c_peer.base)

        tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
        tile_b: pl.Tile[[16, 16], pl.FP16] = pl.load(b, [0, 0], [16, 16])
        result_add: pl.Tile[[16, 16], pl.FP16] = pl.add(tile_a, tile_b)
        result_sub: pl.Tile[[16, 16], pl.FP16] = pl.sub(tile_a, tile_b)

        pl.tpush_to_aic(result_add, aiv_idx=0)
        pl.tpush_to_aic(result_sub, aiv_idx=0)

    @pl.function(type=pl.FunctionType.InCore)
    def cube_consumer(
        self,
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
        pl.aic_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=pipe_buf.base)

        received_add: pl.Tile[[16, 16], pl.FP16] = pl.tpop_from_aiv(aiv_idx=0)
        received_sub: pl.Tile[[16, 16], pl.FP16] = pl.tpop_from_aiv(aiv_idx=0)

        mm_result: pl.Tile[[16, 16], pl.FP32] = pl.matmul(received_add, received_sub)

        pl.tfree_to_aiv(aiv_idx=0)
        pl.tfree_to_aiv(aiv_idx=0)

        updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(mm_result, [0, 0], output)
        return updated


# ============================================================================
# Bidirectional Test Program
# ============================================================================


@pl.program
class BidirectionalCrossCorProgram:
    """Bidirectional cross-core: Vector preprocesses → Cube matmul → Vector post-processes.

    vector_bidir: Loads + adds (V2C push), receives matmul result (C2V pop), applies exp, stores.
    cube_bidir: Receives preprocessed data (V2C pop), does matmul, pushes result back (C2V push).
    """

    @pl.function(type=pl.FunctionType.InCore)
    def vector_bidir(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        # C2V consumer: reserve buffer for incoming data from Cube (explicit base)
        c2v_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=2048, base=0x2000)
        # V2C producer: import cube's reserved buffer
        v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_bidir")
        # Bidirectional init with consumer buffer addresses
        pl.aiv_initialize_pipe(
            dir_mask=3, slot_size=512, c2v_consumer_buf=c2v_buf.base, v2c_consumer_buf=v2c_peer.base
        )

        # Preprocess: elementwise add (Vector op)
        tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
        tile_b: pl.Tile[[16, 16], pl.FP16] = pl.load(b, [0, 0], [16, 16])
        sum_tile: pl.Tile[[16, 16], pl.FP16] = pl.add(tile_a, tile_b)

        # Push preprocessed data to Cube for matmul (V2C direction)
        pl.tpush_to_aic(sum_tile, aiv_idx=0)

        # Receive matmul result back from Cube (C2V direction)
        mm_result: pl.Tile[[16, 16], pl.FP32] = pl.tpop_from_aic(aiv_idx=0)

        # Post-process: apply exp (Vector op)
        processed: pl.Tile[[16, 16], pl.FP32] = pl.exp(mm_result)

        # Release C2V slot
        pl.tfree_to_aic(aiv_idx=0)

        # Store final result
        updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(processed, [0, 0], output)
        return updated

    @pl.function(type=pl.FunctionType.InCore)
    def cube_bidir(
        self,
        weight: pl.Tensor[[16, 16], pl.FP16],
    ):
        # V2C consumer: reserve buffer for incoming data from Vector (explicit base)
        v2c_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=2048, base=0x1000)
        # C2V producer: import vector's reserved buffer
        c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_bidir")
        # Bidirectional init with explicit consumer buffer addresses
        pl.aic_initialize_pipe(
            dir_mask=3, slot_size=512, c2v_consumer_buf=c2v_peer.base, v2c_consumer_buf=v2c_buf.base
        )

        # Receive preprocessed tile from Vector (V2C direction)
        received: pl.Tile[[16, 16], pl.FP16] = pl.tpop_from_aiv(aiv_idx=0)

        # Matmul (Cube op)
        w_tile: pl.Tile[[16, 16], pl.FP16] = pl.load(weight, [0, 0], [16, 16])
        mm_result: pl.Tile[[16, 16], pl.FP32] = pl.matmul(received, w_tile)

        # Release V2C slot
        pl.tfree_to_aiv(aiv_idx=0)

        # Push matmul result back to Vector for post-processing (C2V direction)
        pl.tpush_to_aiv(mm_result, aiv_idx=0)


# ============================================================================
# Test Suite
# ============================================================================


class TestCrossCoreTpushTpopCodegen:
    """Tests for cross-core TPUSH/TPOP PTO code generation."""

    @staticmethod
    def _compile_and_generate(program) -> dict[str, str]:
        """Compile program and return dict of {func_name: mlir_code}.

        Uses a custom pipeline matching the Default strategy but without
        ExpandMixedKernel, since these programs already have manual cross-core
        setup with explicit AIC/AIV tpush/tpop ops.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B_PTO)

        pipeline = passes.PassPipeline()
        for factory in [
            passes.unroll_loops,
            passes.convert_to_ssa,
            passes.flatten_call_expr,
            passes.split_chunked_loops,
            passes.interchange_chunk_loops,
            passes.outline_incore_scopes,
            passes.outline_cluster_scopes,
            passes.convert_tensor_to_tile_ops,
            passes.flatten_tile_nd_to_2d,
            passes.infer_tile_memory_space,
            passes.resolve_transpose_layout,
            passes.init_mem_ref,
            passes.basic_memory_reuse,
            passes.allocate_memory_addr,
        ]:
            pipeline.add_pass(factory())
        optimized = pipeline.run(program)

        result = {}
        codegen_instance = codegen.PTOCodegen()
        for func in optimized.functions.values():
            single = ir.Program([func], func.name, optimized.span)
            mlir_code = codegen_instance.generate(single)
            result[func.name] = mlir_code
        return result

    def test_unidirectional_v2c_vector_producer(self):
        """Test Vector producer generates correct V2C cross-core PTO ops."""
        codes = self._compile_and_generate(CrossCoreTpushTpopProgram)
        vector_code = codes["vector_producer"]

        assert vector_code, "Vector producer MLIR should not be empty"
        assert "pto.import_peer_buffer" in vector_code, "Should contain pto.import_peer_buffer"
        assert 'peer_func = "cube_consumer"' in vector_code, "Should reference cube_consumer"
        assert "pto.aiv_initialize_pipe" in vector_code, "Should contain pto.aiv_initialize_pipe"
        assert "dir_mask = 2" in vector_code, "Should have dir_mask = 2 (V2C)"
        assert "v2c_consumer_buf = 4096" in vector_code, "Should have v2c_consumer_buf = 4096 (0x1000)"
        assert "pto.tpush_to_aic" in vector_code, "Should contain pto.tpush_to_aic"
        assert "pto.tadd" in vector_code, "Should contain elementwise add (Vector op)"

    def test_unidirectional_v2c_cube_consumer(self):
        """Test Cube consumer generates correct V2C cross-core PTO ops."""
        codes = self._compile_and_generate(CrossCoreTpushTpopProgram)
        cube_code = codes["cube_consumer"]

        assert cube_code, "Cube consumer MLIR should not be empty"
        assert "pto.reserve_buffer" in cube_code, "Should contain pto.reserve_buffer"
        assert 'name = "v2c_slot_buffer"' in cube_code, "Should reference v2c_slot_buffer"
        assert "base = 4096" in cube_code, "Should have explicit base address (0x1000 = 4096)"
        assert "pto.aic_initialize_pipe" in cube_code, "Should contain pto.aic_initialize_pipe"
        assert "dir_mask = 2" in cube_code, "Should have dir_mask = 2 (V2C)"
        assert "v2c_consumer_buf = 4096" in cube_code, "Should have v2c_consumer_buf = 4096 (0x1000)"
        assert "pto.tpop_from_aiv" in cube_code, "Should contain pto.tpop_from_aiv"
        assert "pto.tfree_to_aiv" in cube_code, "Should contain pto.tfree_to_aiv"
        assert "pto.tmatmul" in cube_code, "Should contain matmul (Cube op)"

    def test_bidirectional_vector(self):
        """Test Vector kernel with bidirectional communication."""
        codes = self._compile_and_generate(BidirectionalCrossCorProgram)
        vector_code = codes["vector_bidir"]

        assert vector_code, "Vector bidir MLIR should not be empty"
        # Buffer setup: C2V consumer reserves buffer, V2C producer imports peer buffer
        assert "pto.reserve_buffer" in vector_code, "Should reserve buffer for C2V"
        assert 'name = "c2v_slot_buffer"' in vector_code, "Should reference c2v_slot_buffer"
        assert "base = 8192" in vector_code, "Should have explicit base address (0x2000 = 8192)"
        assert "pto.import_peer_buffer" in vector_code, "Should import peer buffer for V2C"
        assert 'peer_func = "cube_bidir"' in vector_code, "Should reference cube_bidir"
        # Bidirectional init
        assert "pto.aiv_initialize_pipe" in vector_code, "Should contain aiv_initialize_pipe"
        assert "dir_mask = 3" in vector_code, "Should have dir_mask = 3 (bidirectional)"
        assert "c2v_consumer_buf = 8192" in vector_code, "Should have c2v_consumer_buf = 8192 (0x2000)"
        assert "v2c_consumer_buf = 4096" in vector_code, "Should have v2c_consumer_buf = 4096 (0x1000)"
        # V2C producer side: preprocess + push
        assert "pto.tadd" in vector_code, "Should do elementwise add (Vector op)"
        assert "pto.tpush_to_aic" in vector_code, "Should push to AIC"
        # C2V consumer side: receive matmul result + post-process
        assert "pto.tpop_from_aic" in vector_code, "Should pop from AIC"
        assert "pto.texp" in vector_code, "Should do exp post-processing (Vector op)"
        assert "pto.tfree_to_aic" in vector_code, "Should free C2V slot"

    def test_bidirectional_cube(self):
        """Test Cube kernel with bidirectional communication."""
        codes = self._compile_and_generate(BidirectionalCrossCorProgram)
        cube_code = codes["cube_bidir"]

        assert cube_code, "Cube bidir MLIR should not be empty"
        # Buffer setup: V2C consumer reserves buffer with explicit base, C2V producer imports peer buffer
        assert "pto.reserve_buffer" in cube_code, "Should reserve buffer for V2C"
        assert 'name = "v2c_slot_buffer"' in cube_code, "Should reference v2c_slot_buffer"
        assert "base = 4096" in cube_code, "Should have explicit base address (0x1000 = 4096)"
        assert "pto.import_peer_buffer" in cube_code, "Should import peer buffer for C2V"
        assert 'peer_func = "vector_bidir"' in cube_code, "Should reference vector_bidir"
        # Bidirectional init with explicit consumer buffer addresses
        assert "pto.aic_initialize_pipe" in cube_code, "Should contain aic_initialize_pipe"
        assert "dir_mask = 3" in cube_code, "Should have dir_mask = 3 (bidirectional)"
        assert "c2v_consumer_buf = 8192" in cube_code, "Should have c2v_consumer_buf = 8192 (0x2000)"
        assert "v2c_consumer_buf = 4096" in cube_code, "Should have v2c_consumer_buf = 4096 (0x1000)"
        # V2C consumer side: receive preprocessed data
        assert "pto.tpop_from_aiv" in cube_code, "Should pop from AIV"
        assert "pto.tfree_to_aiv" in cube_code, "Should free V2C slot"
        # C2V producer side: matmul + push back
        assert "pto.tpush_to_aiv" in cube_code, "Should push to AIV"
        assert "pto.tmatmul" in cube_code, "Should do matmul (Cube op)"

    def test_all_cross_core_pto_ops_covered(self):
        """Verify all 10 cross-core PTO operations are exercised across both test programs."""
        unidir_codes = self._compile_and_generate(CrossCoreTpushTpopProgram)
        bidir_codes = self._compile_and_generate(BidirectionalCrossCorProgram)
        all_code = "\n".join(unidir_codes.values()) + "\n" + "\n".join(bidir_codes.values())

        expected_ops = [
            "pto.tpush_to_aiv",
            "pto.tpush_to_aic",
            "pto.tpop_from_aic",
            "pto.tpop_from_aiv",
            "pto.tfree_to_aic",
            "pto.tfree_to_aiv",
            "pto.aic_initialize_pipe",
            "pto.aiv_initialize_pipe",
            "pto.reserve_buffer",
            "pto.import_peer_buffer",
        ]
        for op in expected_ops:
            assert op in all_code, f"Expected PTO op '{op}' not found in generated MLIR"


class TestExpandMixedKernelCodegen:
    """Tests that PTO codegen works on AIC/AIV functions produced by expand_mixed_kernel."""

    @staticmethod
    def _expand_and_generate(program) -> dict[str, str]:
        """Apply full PTOAS passes with expand_mixed_kernel, then generate PTO MLIR per function.

        Uses all PTOAS strategy passes with expand_mixed_kernel inserted after
        ConvertTensorToTileOps (its intended pipeline position).

        Returns:
            dict mapping function name to generated MLIR code for InCore-variant functions.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B_PTO)

        # Full PTOAS pipeline with expand_mixed_kernel at its intended position
        pipeline = passes.PassPipeline()
        pipeline.add_pass(passes.unroll_loops())
        pipeline.add_pass(passes.convert_to_ssa())
        pipeline.add_pass(passes.flatten_call_expr())
        pipeline.add_pass(passes.split_chunked_loops())
        pipeline.add_pass(passes.interchange_chunk_loops())
        pipeline.add_pass(passes.outline_incore_scopes())
        pipeline.add_pass(passes.outline_cluster_scopes())
        pipeline.add_pass(passes.convert_tensor_to_tile_ops())
        pipeline.add_pass(passes.flatten_tile_nd_to_2d())
        pipeline.add_pass(passes.infer_tile_memory_space())
        pipeline.add_pass(passes.expand_mixed_kernel())
        pipeline.add_pass(passes.init_mem_ref())
        pipeline.add_pass(passes.basic_memory_reuse())
        pipeline.add_pass(passes.allocate_memory_addr())
        optimized = pipeline.run(program)

        result = {}
        codegen_instance = codegen.PTOCodegen()
        for func in optimized.functions.values():
            if not ir.is_incore_type(func.func_type):
                continue
            single = ir.Program([func], func.name, optimized.span)
            mlir_code = codegen_instance.generate(single)
            result[func.name] = mlir_code
        return result

    def test_tile_sub_is_vector_codegen(self):
        """tile.sub in mixed kernel should generate pto.tsub in AIV, pto.tmatmul in AIC."""

        @pl.program
        class MixedSubMatmul:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
                x_sub: pl.Tile[[16, 128], pl.BF16] = pl.sub(x_tile, x_tile)
                x_sub_l1: pl.Tile[[16, 128], pl.BF16] = pl.move(x_sub, target_memory=pl.MemorySpace.Mat)
                s_sub_l0a: pl.Tile[[16, 128], pl.BF16] = pl.move(x_sub_l1, target_memory=pl.MemorySpace.Left)
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(s_sub_l0a, y_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        codes = self._expand_and_generate(MixedSubMatmul)

        # AIV function should contain pto.tsub (vector op)
        assert "main_incore_0_aiv" in codes, "AIV function should be generated"
        aiv_code = codes["main_incore_0_aiv"]
        print(aiv_code)
        assert "pto.tsub" in aiv_code, "AIV should contain pto.tsub for tile.sub"

        # AIC function should contain pto.tmatmul (cube op)
        assert "main_incore_0_aic" in codes, "AIC function should be generated"
        aic_code = codes["main_incore_0_aic"]
        assert "pto.tmatmul" in aic_code, "AIC should contain pto.tmatmul for tile.matmul"

        # tile.sub should NOT be in AIC
        assert "pto.tsub" not in aic_code, "AIC should not contain pto.tsub"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

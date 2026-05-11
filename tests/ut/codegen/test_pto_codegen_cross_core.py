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

Multiple explicit pipes:
  1. Vector → Cube (V2C): Vector pushes one tile on pipe id 0
  2. Vector → Cube (V2C): Vector pushes another tile on pipe id 1
"""

import re

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.backend.pto_backend import _build_group_mapping


def _extract_func_section(mlir_code: str, func_name: str) -> str:
    """Return the MLIR slice for a single func.func body."""
    start_token = f"func.func @{func_name}"
    start = mlir_code.find(start_token)
    assert start != -1, f"Expected function {func_name!r} in MLIR:\n{mlir_code}"
    next_start = mlir_code.find("\n  func.func @", start + len(start_token))
    if next_start == -1:
        return mlir_code[start:]
    return mlir_code[start:next_start]


# ============================================================================
# Test Program: Vector Producer + Cube Consumer (V2C unidirectional)
# ============================================================================


@pl.program
class CrossCoreTpushTpopProgram:
    """V2C unidirectional cross-core program with orchestration wrapper.

    Vector producer: loads tiles a and b, computes add and sub, pushes both to Cube.
    Cube consumer: pops tiles, performs matmul, stores result.
    """

    @pl.function(type=pl.FunctionType.AIV)
    def vector_producer(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
        pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=v2c_peer)

        tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
        tile_b: pl.Tile[[16, 16], pl.FP16] = pl.load(b, [0, 0], [16, 16])
        result_add: pl.Tile[[16, 16], pl.FP16] = pl.add(tile_a, tile_b)
        result_sub: pl.Tile[[16, 16], pl.FP16] = pl.sub(tile_a, tile_b)

        pl.tpush_to_aic(result_add, split=1)
        pl.tpush_to_aic(result_sub, split=1)

    @pl.function(type=pl.FunctionType.AIC)
    def cube_consumer(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
        pl.aic_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=pipe_buf)

        received_add: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=1)
        received_sub: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=1)
        received_add_left = pl.move(received_add, target_memory=pl.Mem.Left)
        received_sub_right = pl.move(received_sub, target_memory=pl.Mem.Right)

        mm_result: pl.Tile[[16, 16], pl.FP32] = pl.matmul(received_add_left, received_sub_right)

        pl.tfree_to_aiv(received_add)
        pl.tfree_to_aiv(received_sub)

        updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(mm_result, [0, 0], output)
        return updated

    @pl.function(type=pl.FunctionType.Group)
    def group_func(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        updated = self.cube_consumer(a, b, output)
        self.vector_producer(a, b, output)
        return updated

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        out = self.group_func(a, b, output)
        return out


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
        pl.aiv_initialize_pipe(dir_mask=3, slot_size=512, c2v_consumer_buf=c2v_buf, v2c_consumer_buf=v2c_peer)

        # Preprocess: elementwise add (Vector op)
        tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
        tile_b: pl.Tile[[16, 16], pl.FP16] = pl.load(b, [0, 0], [16, 16])
        sum_tile: pl.Tile[[16, 16], pl.FP16] = pl.add(tile_a, tile_b)

        # Push preprocessed data to Cube for matmul (V2C direction)
        pl.tpush_to_aic(sum_tile, split=0)

        # Receive matmul result back from Cube (C2V direction)
        mm_result: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)

        # Post-process: apply exp (Vector op)
        processed: pl.Tile[[16, 16], pl.FP32] = pl.exp(mm_result)

        # Release C2V slot
        pl.tfree_to_aic(mm_result)

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
        pl.aic_initialize_pipe(dir_mask=3, slot_size=512, c2v_consumer_buf=c2v_peer, v2c_consumer_buf=v2c_buf)

        # Receive preprocessed tile from Vector (V2C direction)
        received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)

        # Matmul (Cube op)
        w_tile: pl.Tile[[16, 16], pl.FP16] = pl.load(weight, [0, 0], [16, 16])
        mm_result: pl.Tile[[16, 16], pl.FP32] = pl.matmul(received, w_tile)

        # Release V2C slot
        pl.tfree_to_aiv(received)

        # Push matmul result back to Vector for post-processing (C2V direction)
        pl.tpush_to_aiv(mm_result, split=0)


# ============================================================================
# Multiple-Pipe Cross-Core Test Program
# ============================================================================


@pl.program
class MultiPipeSameDirectionCrossCoreProgram:
    """Two explicit V2C frontend pipes from Vector to Cube."""

    @pl.function(type=pl.FunctionType.InCore)
    def vector_multi_pipe(
        self,
        a: pl.Tensor[[16, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
    ):
        v2c_peer_0 = pl.import_peer_buffer(name="v2c_slot_buffer_0", peer_func="cube_multi_pipe")
        v2c_peer_1 = pl.import_peer_buffer(name="v2c_slot_buffer_1", peer_func="cube_multi_pipe")
        pl.aiv_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_peer_0,
            dir_mask=2,
            slot_size=512,
            id=0,
        )
        pl.aiv_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_peer_1,
            dir_mask=2,
            slot_size=512,
            id=1,
        )

        tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
        tile_b: pl.Tile[[16, 16], pl.FP16] = pl.load(b, [0, 0], [16, 16])
        sum_tile: pl.Tile[[16, 16], pl.FP16] = pl.add(tile_a, tile_b)
        diff_tile: pl.Tile[[16, 16], pl.FP16] = pl.sub(tile_a, tile_b)
        pl.tpush_to_aic(sum_tile, split=0, id=0)
        pl.tpush_to_aic(diff_tile, split=0, id=1)

    @pl.function(type=pl.FunctionType.InCore)
    def cube_multi_pipe(self):
        v2c_buf_0 = pl.reserve_buffer(name="v2c_slot_buffer_0", size=4096, base=0x1000)
        v2c_buf_1 = pl.reserve_buffer(name="v2c_slot_buffer_1", size=4096, base=0x2000)
        pl.aic_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_buf_0,
            dir_mask=2,
            slot_size=512,
            id=0,
        )
        pl.aic_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_buf_1,
            dir_mask=2,
            slot_size=512,
            id=1,
        )

        sum_tile: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0, id=0)
        diff_tile: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0, id=1)
        sum_left = pl.move(sum_tile, target_memory=pl.Mem.Left)
        diff_right = pl.move(diff_tile, target_memory=pl.Mem.Right)
        _ = pl.matmul(sum_left, diff_right)
        pl.tfree_to_aiv(sum_tile)
        pl.tfree_to_aiv(diff_tile)


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
        backend.set_backend_type(BackendType.Ascend910B)
        # backend.set_backend_type(BackendType.Ascend950)

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
            passes.lower_transpose_load_param_layout,
            passes.resolve_backend_op_layouts,
            passes.init_mem_ref,
            passes.memory_reuse,
            passes.allocate_memory_addr,
        ]:
            pipeline.add_pass(factory())
        optimized = pipeline.run(program)

        result = {}
        codegen_instance = codegen.PTOCodegen()
        groups, ungrouped = _build_group_mapping(optimized)

        # Grouped: one module per group
        for group_name, members in groups.items():
            grouped_program = ir.Program(members, group_name, optimized.span)
            mlir_code = codegen_instance.generate(grouped_program)
            result[group_name] = mlir_code
            for func in members:
                result[func.name] = mlir_code

        # Ungrouped: one module per function (existing behavior)
        for func in ungrouped:
            single = ir.Program([func], func.name, optimized.span)
            mlir_code = codegen_instance.generate(single)
            result[func.name] = mlir_code
        return result

    def test_unidirectional_v2c_vector_producer(self):
        """Test Vector producer generates correct V2C cross-core PTO ops."""
        codes = self._compile_and_generate(CrossCoreTpushTpopProgram)
        vector_code = codes["vector_producer"]

        assert vector_code, "Vector producer MLIR should not be empty"
        assert "pto.import_reserved_buffer" in vector_code, "Should contain pto.import_reserved_buffer"
        assert "peer_func = @cube_consumer" in vector_code, (
            "Should reference cube_consumer with MLIR symbol syntax"
        )
        assert "-> i32" in vector_code, "import_reserved_buffer should return i32"
        assert "pto.aiv_initialize_pipe" in vector_code, "Should contain pto.aiv_initialize_pipe"
        assert "dir_mask = 2" in vector_code, "Should have dir_mask = 2 (V2C)"
        assert "v2c_consumer_buf = " in vector_code, "Should have v2c_consumer_buf as SSA reference"
        assert "c2v_consumer_buf = " in vector_code, "Should have c2v_consumer_buf as SSA reference"
        assert "pto.tpush_to_aic" in vector_code, "Should contain pto.tpush_to_aic"
        assert "pto.tadd" in vector_code, "Should contain elementwise add (Vector op)"

    def test_unidirectional_v2c_cube_consumer(self):
        """Test Cube consumer generates correct V2C cross-core PTO ops."""
        codes = self._compile_and_generate(CrossCoreTpushTpopProgram)
        cube_code = codes["cube_consumer"]

        assert cube_code, "Cube consumer MLIR should not be empty"
        assert "pto.reserve_buffer" in cube_code, "Should contain pto.reserve_buffer"
        assert 'name = "v2c_slot_buffer"' in cube_code, "Should reference v2c_slot_buffer"
        assert "auto = false" in cube_code, "Should have auto = false for explicit base address"
        assert "base = 4096" in cube_code, "Should have explicit base address (0x1000 = 4096)"
        assert "location = #pto.address_space<" in cube_code, "Should have location attribute"
        assert "-> i32" in cube_code, "reserve_buffer should return i32"
        assert "pto.aic_initialize_pipe" in cube_code, "Should contain pto.aic_initialize_pipe"
        assert "dir_mask = 2" in cube_code, "Should have dir_mask = 2 (V2C)"
        assert "v2c_consumer_buf = " in cube_code, "Should have v2c_consumer_buf as SSA reference"
        assert "c2v_consumer_buf = " in cube_code, "Should have c2v_consumer_buf as SSA reference"
        assert "arith.constant 0 : i32" in cube_code, "Should emit i32 constant for default consumer buf"
        assert "pto.tpop_from_aiv" in cube_code, "Should contain pto.tpop_from_aiv"
        assert "= pto.tpop_from_aiv" in cube_code, "tpop should produce SSA result"
        assert "-> !pto.tile_buf<" in cube_code, "tpop should use -> result type syntax"
        assert "pto.tfree_from_aiv" in cube_code, "Should contain pto.tfree_from_aiv"
        tfree_line = next(line for line in cube_code.splitlines() if "pto.tfree_from_aiv" in line)
        assert "{split = " in tfree_line, f"tfree should have split attribute: {tfree_line}"
        assert "pto.tmatmul" in cube_code, "Should contain matmul (Cube op)"

    def test_tpop_dynamic_valid_shape_operands(self):
        """Dynamic tpop result valid_shape should emit PTOAS frontend operands."""
        span = ir.Span.unknown()
        valid_row = ir.Var("valid_row", ir.ScalarType(pl.INDEX), span)
        valid_col = ir.Var("valid_col", ir.ScalarType(pl.INDEX), span)
        memref = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, pl.INT64, span), 16 * 64 * 4, 0)
        tile_view = ir.TileView(valid_shape=[valid_row, valid_col])
        tile_type = ir.TileType([16, 64], pl.FP32, memref, tile_view, ir.MemorySpace.Vec)
        recv_tile = ir.Var("recv_tile", tile_type, span)
        tpop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"split": 2}, tile_type, span)
        body = ir.SeqStmts([ir.AssignStmt(recv_tile, tpop_call, span)], span)
        func = ir.Function(
            "dynamic_tpop",
            [(valid_row, ir.ParamDirection.In), (valid_col, ir.ParamDirection.In)],
            [],
            body,
            span,
            ir.FunctionType.AIV,
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        mlir_code = codegen.PTOCodegen().generate(ir.Program([func], "dynamic_tpop_program", span))
        tpop_line = next(line.strip() for line in mlir_code.splitlines() if "pto.tpop_from_aic" in line)

        assert "pto.tpop_from_aic(%arg0, %arg1) {split = 2}" in tpop_line
        assert "v_row=?" in tpop_line
        assert "v_col=?" in tpop_line
        assert "pto.alloc_tile" not in mlir_code

    def test_tpop_dynamic_valid_shape_keeps_static_counterpart_operand(self):
        """If one tpop valid_shape dim is dynamic, the other dim is still passed explicitly."""
        span = ir.Span.unknown()
        valid_col = ir.Var("valid_col", ir.ScalarType(pl.INDEX), span)
        memref = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, pl.INT64, span), 16 * 64 * 4, 0)
        tile_view = ir.TileView(valid_shape=[ir.ConstInt(8, pl.INT64, span), valid_col])
        tile_type = ir.TileType([16, 64], pl.FP32, memref, tile_view, ir.MemorySpace.Vec)
        recv_tile = ir.Var("recv_tile", tile_type, span)
        tpop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"split": 0}, tile_type, span)
        body = ir.SeqStmts([ir.AssignStmt(recv_tile, tpop_call, span)], span)
        func = ir.Function(
            "dynamic_tpop_static_row",
            [(valid_col, ir.ParamDirection.In)],
            [],
            body,
            span,
            ir.FunctionType.AIV,
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        mlir_code = codegen.PTOCodegen().generate(ir.Program([func], "dynamic_tpop_static_row_program", span))
        tpop_line = next(line.strip() for line in mlir_code.splitlines() if "pto.tpop_from_aic" in line)

        assert "pto.tpop_from_aic(%c8_index, %arg0) {split = 0}" in tpop_line
        assert "v_row=?" in tpop_line
        assert "v_col=?" in tpop_line

    def test_tpop_static_non_full_valid_shape_operands(self):
        """Static tpop valid_shape smaller than physical shape should emit explicit operands."""
        span = ir.Span.unknown()
        memref = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, pl.INT64, span), 16 * 64 * 4, 0)
        tile_view = ir.TileView(valid_shape=[ir.ConstInt(0, pl.INDEX, span), ir.ConstInt(0, pl.INDEX, span)])
        tile_type = ir.TileType([16, 64], pl.FP32, memref, tile_view, ir.MemorySpace.Vec)
        recv_tile = ir.Var("recv_tile", tile_type, span)
        tpop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"split": 0}, tile_type, span)
        body = ir.SeqStmts([ir.AssignStmt(recv_tile, tpop_call, span)], span)
        func = ir.Function("empty_tpop", [], [], body, span, ir.FunctionType.AIV)

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        mlir_code = codegen.PTOCodegen().generate(ir.Program([func], "empty_tpop_program", span))
        tpop_line = next(line.strip() for line in mlir_code.splitlines() if "pto.tpop_from_aic" in line)

        assert "pto.tpop_from_aic(%c0_index, %c0_index) {split = 0}" in tpop_line
        assert "v_row=?" in tpop_line
        assert "v_col=?" in tpop_line

    def test_tpop_dynamic_valid_shape_rejects_bool_operand(self):
        """Dynamic tpop valid_shape operands must be integer or index typed."""
        span = ir.Span.unknown()
        valid_row = ir.Var("valid_row", ir.ScalarType(pl.BOOL), span)
        memref = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, pl.INT64, span), 16 * 64 * 4, 0)
        tile_view = ir.TileView(valid_shape=[valid_row, ir.ConstInt(64, pl.INT64, span)])
        tile_type = ir.TileType([16, 64], pl.FP32, memref, tile_view, ir.MemorySpace.Vec)
        recv_tile = ir.Var("recv_tile", tile_type, span)
        tpop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"split": 0}, tile_type, span)
        body = ir.SeqStmts([ir.AssignStmt(recv_tile, tpop_call, span)], span)
        func = ir.Function(
            "dynamic_tpop_bool_row",
            [(valid_row, ir.ParamDirection.In)],
            [],
            body,
            span,
            ir.FunctionType.AIV,
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        with pytest.raises(Exception, match="tpop valid_shape operand must be integer or index type, got i1"):
            codegen.PTOCodegen().generate(ir.Program([func], "dynamic_tpop_bool_row_program", span))

    @pytest.mark.parametrize(
        ("tpush_op_name", "func_type", "memory_space"),
        [
            ("tile.tpush_to_aiv", ir.FunctionType.AIC, ir.MemorySpace.Acc),
            ("tile.tpush_to_aic", ir.FunctionType.AIV, ir.MemorySpace.Vec),
        ],
    )
    def test_tpush_uses_validshape_aliased_tile(self, tpush_op_name, func_type, memory_space):
        """set_validshape + cross-core tpush should push the in-place validShape tile handle."""
        span = ir.Span.unknown()

        src = ir.Var("src", ir.TensorType([32, 32], pl.FP32), span)
        valid_row = ir.Var("valid_row", ir.ScalarType(pl.INDEX), span)
        valid_col = ir.Var("valid_col", ir.ScalarType(pl.INDEX), span)

        zero = ir.ConstInt(0, pl.INDEX, span)
        shape_32 = ir.ConstInt(32, pl.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple([shape_32, shape_32], span)

        load_memref = ir.MemRef(memory_space, ir.ConstInt(0, pl.INT64, span), 32 * 32 * 4, 0)
        load_view = ir.TileView(
            valid_shape=[shape_32, shape_32],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
        )
        load_type = ir.TileType([32, 32], pl.FP32, load_memref, load_view, memory_space)
        src_tile = ir.Var("src_tile", load_type, span)

        narrowed_memref = ir.MemRef(memory_space, ir.ConstInt(0, pl.INT64, span), 32 * 32 * 4, 0)
        narrowed_view = ir.TileView(
            valid_shape=[valid_row, valid_col],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
        )
        narrowed_type = ir.TileType([32, 32], pl.FP32, narrowed_memref, narrowed_view, memory_space)
        narrowed_tile = ir.Var("narrowed_tile", narrowed_type, span)

        load_call = ir.Call(
            ir.Op("tile.load"),
            [src, offsets, shapes, shapes],
            {"target_memory": memory_space},
            load_type,
            span,
        )
        set_validshape_call = ir.Call(
            ir.Op("tile.set_validshape"),
            [src_tile, valid_row, valid_col],
            {},
            narrowed_type,
            span,
        )
        tpush_call = ir.Call(ir.Op(tpush_op_name), [narrowed_tile], {"split": 0}, ir.UnknownType(), span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(src_tile, load_call, span),
                ir.AssignStmt(narrowed_tile, set_validshape_call, span),
                ir.EvalStmt(tpush_call, span),
            ],
            span,
        )
        func = ir.Function(
            "narrow_then_push",
            [
                (src, ir.ParamDirection.In),
                (valid_row, ir.ParamDirection.In),
                (valid_col, ir.ParamDirection.In),
            ],
            [],
            body,
            span,
            func_type,
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        mlir_code = codegen.PTOCodegen().generate(ir.Program([func], "narrow_then_push_program", span))

        narrowed_alloc_lines = [
            line.strip()
            for line in mlir_code.splitlines()
            if "%narrowed_tile" in line and "pto.alloc_tile" in line
        ]
        set_validshape_line = next(
            line.strip() for line in mlir_code.splitlines() if "pto.set_validshape" in line
        )
        pto_tpush_op_name = tpush_op_name.replace("tile.", "pto.")
        tpush_line = next(line.strip() for line in mlir_code.splitlines() if pto_tpush_op_name in line)

        assert not narrowed_alloc_lines, (
            "Expected set_validshape result to alias the payload tile without a second alloc_tile, got:\n"
            + "\n".join(narrowed_alloc_lines)
        )
        assert "%src_tile" in set_validshape_line, (
            f"Expected set_validshape to target source payload tile SSA, got:\n{set_validshape_line}"
        )
        assert "v_row=?" in set_validshape_line and "v_col=?" in set_validshape_line, (
            f"Expected dynamic tile type on set_validshape, got:\n{set_validshape_line}"
        )
        assert "%src_tile" in tpush_line, (
            f"Expected tpush to push the source tile after in-place set_validshape, got:\n{tpush_line}"
        )
        assert "%narrowed_tile" not in tpush_line, (
            f"Expected tpush not to use a payload-less set_validshape alias alloc, got:\n{tpush_line}"
        )
        assert "v_row=?" in tpush_line and "v_col=?" in tpush_line, (
            f"Expected tpush to follow main's always-dynamic alloc_tile type annotation, got:\n{tpush_line}"
        )

    @pytest.mark.parametrize(
        (
            "tpush_op_name",
            "func_type",
            "memory_space",
            "split",
            "transport_valid_shape",
        ),
        [
            (
                "tile.tpush_to_aiv",
                ir.FunctionType.AIC,
                ir.MemorySpace.Acc,
                1,
                ", %arg1, %c16_index :",
            ),
            (
                "tile.tpush_to_aic",
                ir.FunctionType.AIV,
                ir.MemorySpace.Vec,
                2,
                ", %c16_index, %arg2 :",
            ),
        ],
    )
    def test_split_tpush_uses_full_non_split_transport_dim(
        self, tpush_op_name, func_type, memory_space, split, transport_valid_shape
    ):
        """Split tpush keeps logical validShape but transports a full non-split dimension."""
        span = ir.Span.unknown()

        src = ir.Var("src", ir.TensorType([16, 16], pl.FP32), span)
        valid_row = ir.Var("valid_row", ir.ScalarType(pl.INDEX), span)
        valid_col = ir.Var("valid_col", ir.ScalarType(pl.INDEX), span)

        zero = ir.ConstInt(0, pl.INDEX, span)
        shape_16 = ir.ConstInt(16, pl.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple([shape_16, shape_16], span)

        src_memref = ir.MemRef(memory_space, ir.ConstInt(0, pl.INT64, span), 16 * 16 * 4, 0)
        src_view = ir.TileView(
            valid_shape=[shape_16, shape_16],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
        )
        src_type = ir.TileType([16, 16], pl.FP32, src_memref, src_view, memory_space)
        src_tile = ir.Var("src_tile", src_type, span)

        narrowed_memref = ir.MemRef(memory_space, ir.ConstInt(0, pl.INT64, span), 16 * 16 * 4, 0)
        narrowed_view = ir.TileView(
            valid_shape=[valid_row, valid_col],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
        )
        narrowed_type = ir.TileType([16, 16], pl.FP32, narrowed_memref, narrowed_view, memory_space)
        narrowed_tile = ir.Var("narrowed_tile", narrowed_type, span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(
                    src_tile,
                    ir.Call(
                        ir.Op("tile.load"),
                        [src, offsets, shapes, shapes],
                        {"target_memory": memory_space},
                        src_type,
                        span,
                    ),
                    span,
                ),
                ir.AssignStmt(
                    narrowed_tile,
                    ir.Call(
                        ir.Op("tile.set_validshape"),
                        [src_tile, valid_row, valid_col],
                        {},
                        narrowed_type,
                        span,
                    ),
                    span,
                ),
                ir.EvalStmt(
                    ir.Call(
                        ir.Op(tpush_op_name),
                        [narrowed_tile],
                        {"split": split},
                        ir.UnknownType(),
                        span,
                    ),
                    span,
                ),
            ],
            span,
        )
        func = ir.Function(
            "split_narrow_then_push",
            [
                (src, ir.ParamDirection.In),
                (valid_row, ir.ParamDirection.In),
                (valid_col, ir.ParamDirection.In),
            ],
            [],
            body,
            span,
            func_type,
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        mlir_code = codegen.PTOCodegen().generate(ir.Program([func], "split_narrow_then_push_program", span))

        set_validshape_lines = [
            line.strip() for line in mlir_code.splitlines() if "pto.set_validshape" in line
        ]
        tpush_line = next(
            line.strip() for line in mlir_code.splitlines() if tpush_op_name.replace("tile.", "pto.") in line
        )

        assert len(set_validshape_lines) == 3, (
            "Expected logical set_validshape, transport normalization, and logical restore, got:\n"
            + "\n".join(set_validshape_lines)
        )
        assert ", %arg1, %arg2 :" in set_validshape_lines[0], (
            f"Expected the initial logical validShape update, got:\n{set_validshape_lines[0]}"
        )
        assert transport_valid_shape in set_validshape_lines[1], (
            "Expected split tpush to normalize the non-split transport dimension, got:\n"
            f"{set_validshape_lines[1]}"
        )
        assert ", %arg1, %arg2 :" in set_validshape_lines[2], (
            f"Expected the logical validShape restore after tpush, got:\n{set_validshape_lines[2]}"
        )
        assert "%src_tile" in tpush_line and "%narrowed_tile" not in tpush_line, (
            f"Expected tpush to use the aliased source tile, got:\n{tpush_line}"
        )

    def test_tfree_stays_after_nested_control_flow_use(self):
        """Nested control-flow users of a tpop result must stay before tfree."""

        @pl.program
        class ConditionalTpopProgram:
            @pl.function(type=pl.FunctionType.AIC)
            def cube_consumer(
                self,
                flag: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
                pl.aic_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=pipe_buf)

                received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=1)
                if flag == 0:
                    received_left = pl.move(received, target_memory=pl.MemorySpace.Left)
                    received_mat = pl.move(received_left, target_memory=pl.MemorySpace.Mat)
                    _: pl.Tensor[[16, 16], pl.FP16] = pl.store(received_mat, [0, 0], output)
                else:
                    received_right = pl.move(received, target_memory=pl.MemorySpace.Right)
                    received_mat_else = pl.move(received_right, target_memory=pl.MemorySpace.Mat)
                    _: pl.Tensor[[16, 16], pl.FP16] = pl.store(received_mat_else, [0, 0], output)

                pl.tfree_to_aiv(received)
                return output

        codes = self._compile_and_generate(ConditionalTpopProgram)
        aic_body = _extract_func_section(codes["cube_consumer"], "cube_consumer")

        assert aic_body.index("pto.tpop_from_aiv") < aic_body.index("scf.if"), (
            "The nested control-flow user should remain after the tpop"
        )
        assert aic_body.index("scf.if") < aic_body.index("pto.tfree_from_aiv"), (
            "tfree must remain after nested control-flow uses of the popped tile"
        )

    def test_tfree_rejects_mismatched_tpop_direction(self):
        """tfree must validate that its tile came from the matching tpop direction."""

        @pl.program
        class MismatchedTfreeProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def vector_consumer(self):
                pipe_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=pipe_buf)
                received: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
                pl.tfree_to_aiv(received)

        with pytest.raises(
            Exception,
            match=re.escape(
                "system.tfree_to_aiv requires its tile argument to come from tile.tpop_from_aiv, "
                "got tile.tpop_from_aic"
            ),
        ):
            self._compile_and_generate(MismatchedTfreeProgram)

    def test_tfree_rejects_mismatched_pipe_id(self):
        """Explicit tfree id must match the originating tpop pipe id."""

        @pl.program
        class MismatchedTfreeIdProgram:
            @pl.function(type=pl.FunctionType.AIC)
            def cube_consumer(self):
                pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
                pl.aic_initialize_pipe(
                    pl.const(0, pl.INT32),
                    pipe_buf,
                    dir_mask=2,
                    slot_size=512,
                    id=1,
                )
                received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(
                    split=0,
                    id=1,
                )
                pl.tfree_to_aiv(received, id=0)

        with pytest.raises(
            Exception,
            match=re.escape(
                "system.tfree_to_aiv pipe id 0 does not match originating tile.tpop_from_aiv pipe id 1"
            ),
        ):
            self._compile_and_generate(MismatchedTfreeIdProgram)

    def test_tpop_user_stays_after_if_defined_scalar_dependency(self):
        """A tpop user that depends on an if-defined scalar must not be hoisted before the if."""

        @pl.program
        class IfDefinedScalarProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def vector_consumer(
                self,
                flag: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pipe_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=pipe_buf)

                received: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
                if flag == 0:
                    scale = 1.0
                else:
                    scale = 2.0
                scaled = pl.tile.muls(received, scale)
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(scaled, [0, 0], output)
                pl.tfree_to_aic(received)
                return updated

        codes = self._compile_and_generate(IfDefinedScalarProgram)
        aiv_body = _extract_func_section(codes["vector_consumer"], "vector_consumer")

        assert aiv_body.index("pto.tpop_from_aic") < aiv_body.index("scf.if"), (
            "The scalar-defining if should remain after the tpop"
        )
        assert aiv_body.index("scf.if") < aiv_body.index("pto.tmuls"), (
            "The tpop user must remain after the if-defined scalar dependency"
        )

    def test_bidirectional_vector(self):
        """Test Vector kernel keeps bidirectional communication on one default pipe."""
        codes = self._compile_and_generate(BidirectionalCrossCorProgram)
        vector_code = codes["vector_bidir"]

        assert vector_code, "Vector bidir MLIR should not be empty"
        # Buffer setup: C2V consumer reserves buffer, V2C producer imports peer buffer
        assert "pto.reserve_buffer" in vector_code, "Should reserve buffer for C2V"
        assert 'name = "c2v_slot_buffer"' in vector_code, "Should reference c2v_slot_buffer"
        assert "auto = false" in vector_code, "Should have auto = false for explicit base"
        assert "base = 8192" in vector_code, "Should have explicit base address (0x2000 = 8192)"
        assert "-> i32" in vector_code, "Buffer ops should return i32"
        assert "pto.import_reserved_buffer" in vector_code, "Should import peer buffer for V2C"
        assert "peer_func = @cube_bidir" in vector_code, "Should reference cube_bidir"
        # Bidirectional init
        assert "pto.aiv_initialize_pipe" in vector_code, "Should contain aiv_initialize_pipe"
        assert "dir_mask = 3" in vector_code, "Should have dir_mask = 3 (bidirectional)"
        assert "{id =" not in vector_code, "Legacy bidirectional default pipe should not emit explicit id"
        assert "c2v_consumer_buf = " in vector_code, "Should have c2v_consumer_buf as SSA reference"
        assert "v2c_consumer_buf = " in vector_code, "Should have v2c_consumer_buf as SSA reference"
        # V2C producer side: preprocess + push
        assert "pto.tadd" in vector_code, "Should do elementwise add (Vector op)"
        assert "pto.tpush_to_aic" in vector_code, "Should push to AIC"
        # C2V consumer side: receive matmul result + post-process
        assert "pto.tpop_from_aic" in vector_code, "Should pop from AIC"
        assert "pto.texp" in vector_code, "Should do exp post-processing (Vector op)"
        assert "pto.tfree_from_aic" in vector_code, "Should free C2V slot"

    def test_bidirectional_cube(self):
        """Test Cube kernel keeps bidirectional communication on one default pipe."""
        codes = self._compile_and_generate(BidirectionalCrossCorProgram)
        cube_code = codes["cube_bidir"]

        assert cube_code, "Cube bidir MLIR should not be empty"
        # Buffer setup: V2C consumer reserves buffer with explicit base, C2V producer imports peer buffer
        assert "pto.reserve_buffer" in cube_code, "Should reserve buffer for V2C"
        assert 'name = "v2c_slot_buffer"' in cube_code, "Should reference v2c_slot_buffer"
        assert "auto = false" in cube_code, "Should have auto = false for explicit base"
        assert "base = 4096" in cube_code, "Should have explicit base address (0x1000 = 4096)"
        assert "-> i32" in cube_code, "Buffer ops should return i32"
        assert "pto.import_reserved_buffer" in cube_code, "Should import peer buffer for C2V"
        assert "peer_func = @vector_bidir" in cube_code, "Should reference vector_bidir"
        # Bidirectional init with SSA consumer buffer references
        assert "pto.aic_initialize_pipe" in cube_code, "Should contain aic_initialize_pipe"
        assert "dir_mask = 3" in cube_code, "Should have dir_mask = 3 (bidirectional)"
        assert "{id =" not in cube_code, "Legacy bidirectional default pipe should not emit explicit id"
        assert "c2v_consumer_buf = " in cube_code, "Should have c2v_consumer_buf as SSA reference"
        assert "v2c_consumer_buf = " in cube_code, "Should have v2c_consumer_buf as SSA reference"
        # V2C consumer side: receive preprocessed data
        assert "pto.tpop_from_aiv" in cube_code, "Should pop from AIV"
        assert "pto.tfree_from_aiv" in cube_code, "Should free V2C slot"
        # C2V producer side: matmul + push back
        assert "pto.tpush_to_aiv" in cube_code, "Should push to AIV"
        assert "pto.tmatmul" in cube_code, "Should do matmul (Cube op)"

    def test_multiple_pipe_vector(self):
        """Test Vector kernel with two explicit V2C frontend pipes."""
        codes = self._compile_and_generate(MultiPipeSameDirectionCrossCoreProgram)
        vector_code = codes["vector_multi_pipe"]

        assert vector_code, "Vector multi-pipe MLIR should not be empty"
        # Buffer setup: V2C producer imports two peer buffers
        assert vector_code.count("pto.import_reserved_buffer") == 2
        assert 'name = "v2c_slot_buffer_0"' in vector_code
        assert 'name = "v2c_slot_buffer_1"' in vector_code
        assert "peer_func = @cube_multi_pipe" in vector_code, "Should reference cube_multi_pipe"
        # Multiple explicit V2C init ops
        assert "pto.aiv_initialize_pipe" in vector_code, "Should contain aiv_initialize_pipe"
        assert "dir_mask = 3" not in vector_code, "Explicit multi-pipe program should not combine dirs"
        assert "dir_mask = 1" not in vector_code
        assert "pto.aiv_initialize_pipe {id = 0, dir_mask = 2" in vector_code
        assert "pto.aiv_initialize_pipe {id = 1, dir_mask = 2" in vector_code
        assert "v2c_consumer_buf = " in vector_code, "Should have v2c_consumer_buf as SSA reference"
        # V2C producer side: preprocess + push
        assert "pto.tadd" in vector_code, "Should do elementwise add (Vector op)"
        assert "pto.tsub" in vector_code, "Should do elementwise sub (Vector op)"
        assert "pto.tpush_to_aic" in vector_code, "Should push to AIC"
        assert "pto.tpop_from_aic" not in vector_code
        assert "pto.tfree_from_aic" not in vector_code
        assert vector_code.count("{id = 0, split = 0}") == 1
        assert vector_code.count("{id = 1, split = 0}") == 1

    def test_multiple_pipe_cube(self):
        """Test Cube kernel with two explicit V2C frontend pipes."""
        codes = self._compile_and_generate(MultiPipeSameDirectionCrossCoreProgram)
        cube_code = codes["cube_multi_pipe"]

        assert cube_code, "Cube multi-pipe MLIR should not be empty"
        # Buffer setup: V2C consumer reserves two buffers with explicit bases
        assert cube_code.count("pto.reserve_buffer") == 2
        assert 'name = "v2c_slot_buffer_0"' in cube_code
        assert 'name = "v2c_slot_buffer_1"' in cube_code
        assert "auto = false" in cube_code, "Should have auto = false for explicit base"
        assert "base = 4096" in cube_code, "Should have explicit base address (0x1000 = 4096)"
        assert "base = 8192" in cube_code, "Should have explicit base address (0x2000 = 8192)"
        assert "-> i32" in cube_code, "Buffer ops should return i32"
        assert "pto.import_reserved_buffer" not in cube_code
        # Multiple explicit V2C init ops with SSA consumer buffer references
        assert "pto.aic_initialize_pipe" in cube_code, "Should contain aic_initialize_pipe"
        assert "dir_mask = 3" not in cube_code, "Explicit multi-pipe program should not combine dirs"
        assert "dir_mask = 1" not in cube_code
        assert "pto.aic_initialize_pipe {id = 0, dir_mask = 2" in cube_code
        assert "pto.aic_initialize_pipe {id = 1, dir_mask = 2" in cube_code
        assert "v2c_consumer_buf = " in cube_code, "Should have v2c_consumer_buf as SSA reference"
        # V2C consumer side: receive preprocessed data
        assert "pto.tpop_from_aiv" in cube_code, "Should pop from AIV"
        assert "pto.tfree_from_aiv" in cube_code, "Should free V2C slot"
        assert "pto.tmatmul" in cube_code, "Should do matmul (Cube op)"
        assert "pto.tpush_to_aiv" not in cube_code
        # tpop ids are inherited by tfree even when user code omits tfree id.
        assert cube_code.count("{id = 0, split = 0}") == 2
        assert cube_code.count("{id = 1, split = 0}") == 2

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
            "pto.tfree_from_aic",
            "pto.tfree_from_aiv",
            "pto.aic_initialize_pipe",
            "pto.aiv_initialize_pipe",
            "pto.reserve_buffer",
            "pto.import_reserved_buffer",
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
        backend.set_backend_type(BackendType.Ascend950)

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
        pipeline.add_pass(passes.memory_reuse())
        pipeline.add_pass(passes.allocate_memory_addr())
        optimized = pipeline.run(program)

        result = {}
        codegen_instance = codegen.PTOCodegen()
        groups, ungrouped = _build_group_mapping(optimized)

        # Grouped: one module per group
        for group_name, members in groups.items():
            grouped_program = ir.Program(members, group_name, optimized.span)
            mlir_code = codegen_instance.generate(grouped_program)
            result[group_name] = mlir_code
            for func in members:
                result[func.name] = mlir_code

        # Ungrouped: one module per function
        for func in ungrouped:
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
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
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
        assert "pto.tsub" in aiv_code, "AIV should contain pto.tsub for tile.sub"
        assert "pto.import_reserved_buffer" in aiv_code, "AIV should import the peer V2C buffer"
        assert "pto.aiv_initialize_pipe" in aiv_code, "AIV should initialize cross-core pipe"

        # AIC function should contain pto.tmatmul (cube op)
        assert "main_incore_0_aic" in codes, "AIC function should be generated"
        aic_code = codes["main_incore_0_aic"]
        aic_body = _extract_func_section(aic_code, "main_incore_0_aic")
        assert "pto.tmatmul" in aic_code, "AIC should contain pto.tmatmul for tile.matmul"
        assert "pto.reserve_buffer" in aic_code, "AIC should reserve the V2C consumer buffer"
        assert "auto = false" in aic_code, "Auto reserve_buffer should be resolved before PTO emission"
        assert "base = 0" in aic_code, "The first auto reserve_buffer should start from base 0"
        assert "pto.aic_initialize_pipe" in aic_code, "AIC should initialize cross-core pipe"
        assert aic_body.index("pto.aic_initialize_pipe") < aic_body.index("pto.tpop_from_aiv"), (
            "AIC initialize_pipe should be emitted before the first V2C tpop"
        )
        assert "pto.tfree_from_aiv" in aic_body, "AIC consumer should free the V2C slot"
        assert aic_body.index("pto.tpop_from_aiv") < aic_body.index("pto.tmov"), (
            "AIC should move the popped tile before freeing it"
        )
        assert aic_body.index("pto.tmov") < aic_body.index("pto.tfree_from_aiv"), (
            "AIC should free the popped tile after its direct use"
        )

        # tile.sub should NOT be in AIC
        assert "pto.tsub" not in aic_body, "AIC should not contain pto.tsub"

    def test_acc_slice_before_c2v_boundary_codegen(self):
        """Cube-side tile.slice feeding C2V boundary must survive mixed-kernel expansion."""

        @pl.program
        class SliceBeforeC2V:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                x_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                x_left: pl.Tile[[16, 128], pl.BF16] = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                acc: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_left, y_right)
                row: pl.Tile[[1, 128], pl.FP32] = pl.slice(acc, [1, 128], [0, 0])
                row_vec: pl.Tile[[1, 128], pl.FP32] = pl.move(
                    row,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0: pl.Tensor[[1, 128], pl.FP32] = pl.store(row_vec, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                out_0: pl.Tensor[[1, 128], pl.FP32] = pl.create_tensor([1, 128], dtype=pl.FP32)
                z: pl.Tensor[[1, 128], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        codes = self._expand_and_generate(SliceBeforeC2V)

        assert "main_incore_0_aic" in codes, "AIC function should be generated"
        aic_body = _extract_func_section(codes["main_incore_0_aic"], "main_incore_0_aic")
        assert "pto.subview" in aic_body, "AIC should keep the tile.slice producer before C2V push"
        assert "pto.tpush_to_aiv" in aic_body, "AIC should push the sliced row to AIV"
        assert aic_body.index("pto.subview") < aic_body.index("pto.tpush_to_aiv"), (
            "AIC should subview the row tile before pushing it across cores"
        )

        assert "main_incore_0_aiv" in codes, "AIV function should be generated"
        aiv_body = _extract_func_section(codes["main_incore_0_aiv"], "main_incore_0_aiv")
        assert "pto.tpop_from_aic" in aiv_body, "AIV should pop the sliced row from AIC"
        assert "pto.tstore" in aiv_body, "AIV should store the popped row to the output tensor"

    def test_bidirectional_mixed_kernel_keeps_combined_pipe(self):
        """Automatic bidirectional mixed kernels keep the legacy combined pipe."""

        @pl.program
        class BidirectionalMixed:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x_tile = pl.load(x, [0, 0], [16, 128])
                x_sum = pl.add(x_tile, x_tile)
                x_sum_mat = pl.move(
                    x_sum,
                    target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                x_left = pl.move(x_sum_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                z_vec = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                out_0 = pl.store(z_vec, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 64], pl.BF16],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                out_0: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
                z: pl.Tensor[[16, 64], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        codes = self._expand_and_generate(BidirectionalMixed)
        aic_body = _extract_func_section(codes["main_incore_0_aic"], "main_incore_0_aic")
        aiv_body = _extract_func_section(codes["main_incore_0_aiv"], "main_incore_0_aiv")

        assert "pto.aic_initialize_pipe {dir_mask = 3, slot_size = 4096}" in aic_body
        assert "pto.aiv_initialize_pipe {dir_mask = 3, slot_size = 4096}" in aiv_body
        assert "{id =" not in aic_body
        assert "{id =" not in aiv_body
        assert "pto.tpush_to_aiv" in aic_body and "{split = 0}" in aic_body
        assert "pto.tpop_from_aiv {split = 0}" in aic_body
        assert "pto.tfree_from_aiv {split = 0}" in aic_body
        assert "pto.tpush_to_aic" in aiv_body and "{split = 0}" in aiv_body
        assert "pto.tpop_from_aic {split = 0}" in aiv_body
        assert "pto.tfree_from_aic {split = 0}" in aiv_body

    def test_multiple_pipe_ids_offset_gm_buffers_on_a2a3(self):
        """Explicit pipe ids share one GM workspace but point at disjoint regions."""

        @pl.program
        class MultiplePipeSetup:
            @pl.function(type=pl.FunctionType.AIC)
            def cube_kernel(self):
                c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_kernel")
                v2c_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=32768, base=0x1000)
                pl.aic_initialize_pipe(c2v_peer, pl.const(0, pl.INT32), dir_mask=1, slot_size=8192, id=0)
                pl.aic_initialize_pipe(pl.const(0, pl.INT32), v2c_buf, dir_mask=2, slot_size=4096, id=1)

            @pl.function(type=pl.FunctionType.AIV)
            def vector_kernel(self):
                c2v_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=65536, base=0x2000)
                v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_kernel")
                pl.aiv_initialize_pipe(c2v_buf, pl.const(0, pl.INT32), dir_mask=1, slot_size=8192, id=0)
                pl.aiv_initialize_pipe(pl.const(0, pl.INT32), v2c_peer, dir_mask=2, slot_size=4096, id=1)

            @pl.function(type=pl.FunctionType.Group)
            def group_func(self):
                self.cube_kernel()
                self.vector_kernel()

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self):
                self.group_func()

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        try:
            transformed = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(
                MultiplePipeSetup
            )
            codegen_instance = codegen.PTOCodegen()
            groups, _ = _build_group_mapping(transformed)
            group_funcs = next(iter(groups.values()))
            mlir_code = codegen_instance.generate(
                ir.Program(group_funcs, "multiple_pipe_a2a3", transformed.span)
            )
        finally:
            backend.reset_for_testing()

        aic_body = _extract_func_section(mlir_code, "cube_kernel")
        aiv_body = _extract_func_section(mlir_code, "vector_kernel")

        for body in (aic_body, aiv_body):
            assert "pto.aic_initialize_pipe {id = 0, dir_mask = 1" in body or (
                "pto.aiv_initialize_pipe {id = 0, dir_mask = 1" in body
            )
            assert "id = 1, dir_mask = 2" in body
            assert re.search(r"pto.addptr %arg\d+", body)
            assert "arith.constant 16384 : index" in body
            assert re.search(r"gm_slot_buffer = %arg\d+ : !pto.ptr<f32>", body)
            assert body.count("gm_slot_buffer = %") == 2
            assert "dir_mask = 3" not in body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

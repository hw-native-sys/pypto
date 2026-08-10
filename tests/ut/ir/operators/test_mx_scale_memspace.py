# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for MX LeftScale/RightScale memspace and MX-layout load/move IR."""

import pypto
import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.ir.op import tensor_ops as tensor
from pypto.ir.op import tile_ops as tile
from pypto.language.parser.diagnostics import InvalidOperationError


@pytest.fixture(autouse=True)
def _ascend950():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        yield
    finally:
        backend.reset_for_testing()


def _const_shape(rows: int, cols: int):
    span = ir.Span.unknown()
    return [ir.ConstInt(rows, DataType.INDEX, span), ir.ConstInt(cols, DataType.INDEX, span)]


def _tile_var(shape, dtype, *, memory=None, view=None, name="t"):
    tt = ir.TileType(shape, dtype, None, view, memory)
    return ir.Var(name, tt, ir.Span.unknown())


def _mx_tensor_var(name, rows, cols, layout=ir.TensorLayout.MX_A_ZZ, dtype=DataType.FP8E8M0):
    return ir.Var(
        name,
        ir.TensorType(
            _const_shape(rows, cols),
            dtype,
            tensor_view=ir.TensorView([], layout),
        ),
        ir.Span.unknown(),
    )


class TestMxScaleMemSpaces:
    def test_leftscale_move_layout_row_major_fractal32(self):
        src = _tile_var(
            _const_shape(16, 8),
            DataType.UINT8,
            memory=ir.MemorySpace.Mat,
            view=ir.TileView(
                blayout=ir.TileLayout.row_major,
                slayout=ir.TileLayout.row_major,
                fractal=32,
            ),
        )
        call = tile.move(src, target_memory=ir.MemorySpace.LeftScale)
        out = call.type
        assert isinstance(out, ir.TileType)
        assert out.memory_space == ir.MemorySpace.LeftScale
        assert out.dtype == DataType.FP8E8M0  # ui8 promotes for loc=scaling
        view = out.get_effective_tile_view()
        assert view.blayout == ir.TileLayout.row_major
        assert view.slayout == ir.TileLayout.row_major
        assert view.fractal == 32

    def test_rightscale_move_layout_col_major_fractal32(self):
        src = _tile_var(
            _const_shape(8, 16),
            DataType.UINT8,
            memory=ir.MemorySpace.Mat,
            view=ir.TileView(
                blayout=ir.TileLayout.col_major,
                slayout=ir.TileLayout.col_major,
                fractal=32,
            ),
        )
        call = tile.move(src, target_memory=ir.MemorySpace.RightScale)
        out = call.type
        assert isinstance(out, ir.TileType)
        assert out.memory_space == ir.MemorySpace.RightScale
        assert out.dtype == DataType.FP8E8M0
        view = out.get_effective_tile_view()
        assert view.blayout == ir.TileLayout.col_major
        assert view.slayout == ir.TileLayout.col_major
        assert view.fractal == 32

    def test_mx_layout_load_requires_explicit_mat(self):
        tensor = _mx_tensor_var("s", 16, 8)
        with pytest.raises(ValueError, match="requires explicit target_memory=MemorySpace.Mat"):
            tile.load(tensor, [0, 0], [16, 8])

    def test_mx_b_layout_load_col_major(self):
        tensor = _mx_tensor_var("w_s", 8, 16, ir.TensorLayout.MX_B_NN)
        call = tile.load(tensor, [0, 0], [8, 16], target_memory=ir.MemorySpace.Mat)
        out = call.type
        assert isinstance(out, ir.TileType)
        assert out.memory_space == ir.MemorySpace.Mat
        view = out.get_effective_tile_view()
        assert view.blayout == ir.TileLayout.col_major
        assert view.slayout == ir.TileLayout.col_major
        assert view.fractal == 32

    def test_mx_layout_rejects_vec_target(self):
        tensor = _mx_tensor_var("s", 16, 8)
        with pytest.raises(ValueError, match="Mat|Vec"):
            tile.load(tensor, [0, 0], [16, 8], target_memory=ir.MemorySpace.Vec)

    def test_mx_layout_rejects_non_2d_load_window(self):
        tensor = _mx_tensor_var("s", 16, 8)
        with pytest.raises(ValueError, match="2D load window"):
            tile.load(tensor, [0], [16], target_memory=ir.MemorySpace.Mat)

    def test_scale_move_rejects_non_scale_dtype(self):
        src = _tile_var(
            _const_shape(16, 8),
            DataType.FP16,
            memory=ir.MemorySpace.Mat,
        )
        with pytest.raises(ValueError, match="UINT8|FP8E8M0"):
            tile.move(src, target_memory=ir.MemorySpace.LeftScale)

    def test_scale_move_rejects_non_mat_input(self):
        src = _tile_var(
            _const_shape(16, 8),
            DataType.FP8E8M0,
            memory=ir.MemorySpace.Vec,
        )
        with pytest.raises(ValueError, match="Mat memory"):
            tile.move(src, target_memory=ir.MemorySpace.LeftScale)

    def test_scale_move_rejects_non_2d_input(self):
        span = ir.Span.unknown()
        src = _tile_var(
            [ir.ConstInt(16, DataType.INDEX, span)],
            DataType.FP8E8M0,
            memory=ir.MemorySpace.Mat,
        )
        with pytest.raises(ValueError, match="2D tile"):
            tile.move(src, target_memory=ir.MemorySpace.RightScale)

    @pytest.mark.parametrize(
        ("space", "view"),
        [
            (
                ir.MemorySpace.LeftScale,
                ir.TileView(
                    blayout=ir.TileLayout.col_major,
                    slayout=ir.TileLayout.col_major,
                    fractal=32,
                ),
            ),
            (
                ir.MemorySpace.RightScale,
                ir.TileView(
                    blayout=ir.TileLayout.row_major,
                    slayout=ir.TileLayout.row_major,
                    fractal=32,
                ),
            ),
            (
                ir.MemorySpace.LeftScale,
                ir.TileView(
                    blayout=ir.TileLayout.row_major,
                    slayout=ir.TileLayout.row_major,
                    fractal=512,
                ),
            ),
        ],
    )
    def test_scale_move_rejects_mismatched_source_layout(self, space, view):
        src = _tile_var(
            _const_shape(16, 8),
            DataType.FP8E8M0,
            memory=ir.MemorySpace.Mat,
            view=view,
        )
        with pytest.raises(ValueError, match="matching.*layout"):
            tile.move(src, target_memory=space)

    @pytest.mark.parametrize(
        ("space", "blayout", "slayout"),
        [
            (ir.MemorySpace.LeftScale, ir.TileLayout.col_major, ir.TileLayout.row_major),
            (ir.MemorySpace.RightScale, ir.TileLayout.col_major, ir.TileLayout.row_major),
        ],
    )
    def test_scale_move_rejects_layout_override(self, space, blayout, slayout):
        src = _tile_var(_const_shape(16, 8), DataType.FP8E8M0, memory=ir.MemorySpace.Mat)
        with pytest.raises(ValueError, match="hardware-fixed layout"):
            tile.move(src, target_memory=space, blayout=blayout, slayout=slayout)

    @pytest.mark.parametrize("space", [ir.MemorySpace.LeftScale, ir.MemorySpace.RightScale])
    def test_tile_create_rejects_scale_memory(self, space):
        with pytest.raises(ValueError, match="does not support target_memory=LeftScale/RightScale"):
            tile.create([16, 8], DataType.FP8E8M0, target_memory=space)

    def test_mx_load_rejects_strided_tensor_view(self):
        shape = _const_shape(16, 8)
        tensor = ir.Var(
            "s",
            ir.TensorType(
                shape,
                DataType.FP8E8M0,
                tensor_view=ir.TensorView([16, 1], ir.TensorLayout.MX_A_ZZ),
            ),
            ir.Span.unknown(),
        )
        with pytest.raises(ValueError, match="packed 2D"):
            tile.load(tensor, [0, 0], [16, 8], target_memory=ir.MemorySpace.Mat)

    def test_tensor_slice_rejects_mx_source(self):
        source = _mx_tensor_var("source", 16, 8)
        with pytest.raises(ValueError, match="tensor.slice does not support MX-layout tensors"):
            tensor.slice(source, [8, 8], [1, 0])

    def test_store_rejects_mx_destination(self):
        destination = _mx_tensor_var("destination", 16, 8)
        source = _tile_var(_const_shape(16, 8), DataType.FP8E8M0, memory=ir.MemorySpace.Vec)
        with pytest.raises(ValueError, match="tile.store does not support MX-layout"):
            tile.store(source, [0, 0], destination)

    def test_mscatter_rejects_mx_destination(self):
        destination = _mx_tensor_var("destination", 16, 8, dtype=DataType.FP16)
        source = _tile_var(_const_shape(16, 8), DataType.FP16, memory=ir.MemorySpace.Vec)
        indices = _tile_var(_const_shape(16, 8), DataType.INT32, memory=ir.MemorySpace.Vec)
        with pytest.raises(ValueError, match="tile.mscatter does not support MX-layout"):
            tile.mscatter(source, indices, destination)

    def test_vec_mgather_rejects_mx_source(self):
        source = _mx_tensor_var("source", 16, 8, dtype=DataType.FP8E4M3FN)
        indices = _tile_var(_const_shape(1, 4), DataType.INT32, memory=ir.MemorySpace.Vec)
        with pytest.raises(ValueError, match="Vec output does not support MX-layout"):
            tile.mgather(source, indices)

    def test_cacheinvalid_rejects_mx_region(self):
        source = _mx_tensor_var("source", 16, 8)
        span = ir.Span.unknown()
        shape = ir.MakeTuple(_const_shape(16, 8), span)
        offsets = ir.MakeTuple(_const_shape(0, 0), span)
        with pytest.raises(ValueError, match="system.cacheinvalid does not support MX-layout"):
            ir.create_op_call("system.cacheinvalid", [source, shape, offsets], {}, span)

    def test_mx_load_rejects_ssa_bound_slice_in_dsl(self):
        with pytest.raises(InvalidOperationError, match="tensor.slice does not support MX-layout tensors"):

            @pl.program
            class Input:
                @pl.function
                def kernel(
                    self,
                    source: pl.Tensor[[16, 8], pl.FP8E8M0, pl.MX_A_ZZ],
                ) -> pl.Tensor[[16, 8], pl.FP8E8M0, pl.MX_A_ZZ]:
                    sliced = pl.slice(source, [8, 8], [1, 0])
                    _scale = pl.load(sliced, [0, 0], [8, 8], target_memory=pl.Mem.Mat)
                    return source

    @pytest.mark.parametrize(
        "make_view",
        [
            lambda source: tensor.reshape(source, [16, 8]),
            lambda source: tensor.transpose(source, 0, 1),
            lambda source: tensor.reinterpret_view(source, dtype=DataType.UINT8),
            lambda source: tensor.view(source, layout=ir.TensorLayout.ND),
        ],
    )
    def test_zero_copy_tensor_views_reject_mx_source(self, make_view):
        source = _mx_tensor_var("source", 16, 8)
        with pytest.raises(ValueError, match="does not support MX"):
            make_view(source)

    def test_mx_load_accepts_direct_orchestration_argument_precondition(self):
        @pl.program
        class Input:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                source: pl.Tensor[[16, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[16, 8], pl.FP8E8M0, pl.MX_A_ZZ]:
                _scale = pl.load(source, [0, 0], [16, 8], target_memory=pl.Mem.Mat)
                return source

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestration(
                self,
                source: pl.Tensor[[16, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[16, 8], pl.FP8E8M0, pl.MX_A_ZZ]:
                result, _ = pl.submit(self.kernel, source)
                return result

        program = passes.classify_iter_arg_carry()(
            passes.materialize_runtime_scopes()(
                passes.derive_call_directions()(passes.convert_to_ssa()(Input))
            )
        )
        orchestration = next(
            function
            for function in program.functions.values()
            if function.func_type == pl.FunctionType.Orchestration
        )
        # Reaching the later core-type check proves the MX argument precondition
        # accepted this packed, unsliced MX tensor.
        with pytest.raises(pypto.InternalError, match="InferFunctionCoreType"):
            codegen.generate_orchestration(program, orchestration)

    def test_mx_load_accepts_materialized_packed_nd_tensor_view(self):
        shape = _const_shape(16, 8)
        tensor = ir.Var(
            "s",
            ir.TensorType(
                shape,
                DataType.FP8E8M0,
                tensor_view=ir.TensorView([8, 1], ir.TensorLayout.MX_A_ZZ),
            ),
            ir.Span.unknown(),
        )
        call = tile.load(
            tensor,
            [0, 0],
            [16, 8],
            target_memory=ir.MemorySpace.Mat,
        )
        assert isinstance(call.type, ir.TileType)

    def test_mx_load_pto_codegen_uses_packed_source_view(self):
        span = ir.Span.unknown()
        tensor_type = ir.TensorType(
            _const_shape(16, 2),
            DataType.FP8E8M0,
            tensor_view=ir.TensorView([], ir.TensorLayout.MX_A_ZZ),
        )
        builder = ir.IRBuilder()
        with builder.function("mx_load_codegen", type=ir.FunctionType.InCore) as function:
            source = function.param("source", tensor_type)
            scale = builder.let(
                "scale",
                tile.load(
                    source,
                    [0, 0],
                    [16, 2],
                    target_memory=ir.MemorySpace.Mat,
                ),
            )
            function.return_type(scale.type)
            builder.return_stmt(scale)

        program = ir.Program([function.get_result()], "mx_load_codegen", span)
        pto = codegen.PTOCodegen().generate(program)
        assert pto.count("pto.make_tensor_view") == 1
        assert "pto.make_tensor_view %arg0" in pto
        assert "shape = [%c1_index, %c1_index, %c1_index, %c16_index, %c2_index]" in pto
        assert (
            "strides = [%c32_index, %c32_index, %c32_index, %c2_index, %c1_index] "
            "{layout = #pto.layout<mx_a_zz>}"
        ) in pto
        assert "pto.tload" in pto


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

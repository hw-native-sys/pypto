# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Non-square C2V layout regression: physical DN bytes must match the view."""

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import TensorSpec
from harness.core.kernel_source import datatype_from_torch
from pypto import ir


@pl.jit.incore
def _column_view(
    a: pl.Tensor[[32, 32], pl.BF16],
    b: pl.Tensor[[32, 16], pl.BF16],
    out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
):
    lhs = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
    rhs = pl.load(b, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
    acc = pl.matmul(lhs, rhs)
    dn = pl.tile.move(
        acc, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box
    )
    return pl.store(pl.tile.transpose_view(dn), [0, 0], out)


@pl.jit.incore
def _row_transpose(
    a: pl.Tensor[[32, 32], pl.BF16],
    b: pl.Tensor[[32, 16], pl.BF16],
    out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
):
    lhs = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
    rhs = pl.load(b, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
    acc = pl.matmul(lhs, rhs)
    nd = pl.tile.move(
        acc, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box
    )
    return pl.store(pl.tile.transpose(nd, 0, 1), [0, 0], out)


@pl.jit
def column_view(
    a: pl.Tensor[[32, 32], pl.BF16],
    b: pl.Tensor[[32, 16], pl.BF16],
    out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
):
    return _column_view(a, b, out)


@pl.jit
def row_transpose(
    a: pl.Tensor[[32, 32], pl.BF16],
    b: pl.Tensor[[32, 16], pl.BF16],
    out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
):
    return _row_transpose(a, b, out)


def _case(kernel, signed, buffer_ir):
    if signed:
        a = (torch.arange(1024).reshape(32, 32) % 7 - 3).to(torch.bfloat16)
        b = (torch.arange(512).reshape(32, 16) % 11 - 5).to(torch.bfloat16)
    else:
        a = torch.eye(32, dtype=torch.bfloat16)
        b = torch.arange(512).reshape(32, 16).to(torch.bfloat16)
    # Accumulate exact small integers independently of the device matmul path.
    expected = (a.to(torch.int64) @ b.to(torch.int64)).T.to(torch.float32)
    return st.case(
        kernel,
        a,
        b,
        torch.empty(16, 32),
        name=f"{kernel.__name__}_signed={signed}_buffer_ir={buffer_ir}",
        golden=lambda _tensors: expected,
        platform="a5",
        enable_buffer_ir=buffer_ir,
        rtol=0,
        atol=0,
    )


@st.cases(
    *[
        _case(kernel, signed, buffer_ir)
        for kernel in (column_view, row_transpose)
        for signed in (False, True)
        for buffer_ir in (False, True)
    ]
)
def test_non_square_c2v(case_run):
    case_run.assert_passed()


def _explicit_case(quant_mode, use_relu, negative_scale=False, layout_mode="nz2nd", integer_no_convert=False):
    relu_mode = "normal_relu" if use_relu else "no_relu"
    scalar_scale = -0.5 if negative_scale else 0.5
    quant_scale = ir.ConstFloat(scalar_scale, pl.FP32, ir.Span.unknown())
    scalar_quant = quant_mode.endswith("scalar")
    vector_quant = quant_mode == "deqf16_vec"
    column_major = layout_mode == "nz2dn"
    out_rows, out_cols = 160, 16
    input_dtype = torch.int8 if quant_mode.startswith("deq") or integer_no_convert else torch.bfloat16
    output_dtype = {"no_convert": torch.float32, "f32_bf16": torch.bfloat16}.get(quant_mode, torch.float16)
    if integer_no_convert:
        output_dtype = torch.int32
    a = (torch.arange(160 * 64).reshape(160, 64) % 7 - 3).to(input_dtype)
    b = (torch.arange(64 * 16).reshape(16, 64) % 11 - 5).to(input_dtype)
    # Vary channel scales and iteration inputs to detect stale slots or controls.
    channel_scales = torch.tensor([0.5, 1.0] * 16, dtype=torch.float32)
    control_words = channel_scales.view(torch.int32).to(torch.int64).reshape(1, 32)
    expected = a.to(torch.int64) @ b.to(torch.int64).T
    if use_relu:
        expected = expected.clamp_min(0)
    expected = expected.to(torch.float32)
    if quant_mode.endswith("scalar"):
        expected *= scalar_scale
    if quant_mode.endswith("vec"):
        expected *= channel_scales[:16]
    expected = expected.to(output_dtype)

    ir_input_dtype = pl.INT8 if input_dtype == torch.int8 else pl.BF16
    ir_output_dtype = {
        torch.float32: pl.FP32,
        torch.bfloat16: pl.BF16,
        torch.float16: pl.FP16,
        torch.int32: pl.INT32,
    }[output_dtype]

    # These fixtures exercise explicit AIC/AIV IR; jit.incore cannot declare
    # those function types. The automatic non-square regression above uses jit.
    @pl.program
    class ExplicitPipe:
        @pl.function(type=pl.FunctionType.AIC)
        def fixpipe_producer(
            self,
            a: pl.Tensor[[160, 64], ir_input_dtype],
            b: pl.Tensor[[16, 64], ir_input_dtype],
            out: pl.Out[pl.Tensor[[out_rows, out_cols], ir_output_dtype]],
            words: pl.Tensor[[1, 32], pl.INT64],
        ):
            peer = pl.import_peer_buffer(name="fixpipe_slots", peer_func="fixpipe_consumer")
            pl.aic_initialize_pipe(
                peer,
                dir_mask=1,
                slot_size=2048,
                slot_num=2,
                acc_push_epilogue={"layout": layout_mode, "quant": quant_mode, "relu": relu_mode},
            )
            packed: pl.Tile[
                [1, 32],
                pl.INT64,
                pl.Mem.Mat,
                pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box),
            ] = pl.load(words, [0, 0], [1, 32], target_memory=pl.Mem.Mat)
            controls = pl.tile.move(packed, target_memory=pl.Mem.Scaling)
            for iteration in pl.range(5):
                lhs = pl.load(a, [iteration * 32, 0], [32, 64], target_memory=pl.Mem.Mat)
                rhs = pl.load(b, [0, 0], [16, 64], target_memory=pl.Mem.Mat)
                acc = pl.matmul(lhs, pl.tile.transpose_view(rhs))
                if scalar_quant:
                    pl.set_quant_scalar(quant_scale)
                if vector_quant:
                    pl.set_quant_vector(controls)
                pl.tpush_to_aiv(acc, split=0)

        @pl.function(type=pl.FunctionType.AIV)
        def fixpipe_consumer(
            self,
            a: pl.Tensor[[160, 64], ir_input_dtype],
            b: pl.Tensor[[16, 64], ir_input_dtype],
            out: pl.Out[pl.Tensor[[out_rows, out_cols], ir_output_dtype]],
            words: pl.Tensor[[1, 32], pl.INT64],
        ):
            buf = pl.reserve_buffer(name="fixpipe_slots", size=4096)
            pl.aiv_initialize_pipe(
                buf,
                dir_mask=1,
                slot_size=2048,
                slot_num=2,
                acc_push_epilogue={"layout": layout_mode, "quant": quant_mode, "relu": relu_mode},
            )
            for iteration in pl.range(5):
                if column_major:
                    tile_dn: pl.Tile[
                        [32, 16],
                        ir_output_dtype,
                        pl.Mem.Vec,
                        pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box),
                    ] = pl.tpop_from_aic(shape=[32, 16], dtype=ir_output_dtype, split=0)
                    pl.store(
                        pl.tile.transpose(pl.tile.transpose_view(tile_dn), 0, 1), [iteration * 32, 0], out
                    )
                    pl.tfree_to_aic(tile_dn)
                else:
                    tile_nd = pl.tpop_from_aic(shape=[32, 16], dtype=ir_output_dtype, split=0)
                    pl.store(tile_nd, [iteration * 32, 0], out)
                    pl.tfree_to_aic(tile_nd)

        @pl.function(type=pl.FunctionType.Group)
        def group(
            self,
            a: pl.Tensor[[160, 64], ir_input_dtype],
            b: pl.Tensor[[16, 64], ir_input_dtype],
            out: pl.Out[pl.Tensor[[out_rows, out_cols], ir_output_dtype]],
            words: pl.Tensor[[1, 32], pl.INT64],
        ) -> pl.Tensor[[out_rows, out_cols], ir_output_dtype]:
            self.fixpipe_producer(a, b, out, words)
            self.fixpipe_consumer(a, b, out, words)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[160, 64], ir_input_dtype],
            b: pl.Tensor[[16, 64], ir_input_dtype],
            out: pl.Out[pl.Tensor[[out_rows, out_cols], ir_output_dtype]],
            words: pl.Tensor[[1, 32], pl.INT64],
        ) -> pl.Tensor[[out_rows, out_cols], ir_output_dtype]:
            return self.group(a, b, out, words)

    return st.case(
        ExplicitPipe,
        tensors=[
            TensorSpec("a", [160, 64], datatype_from_torch(input_dtype), init_value=a),
            TensorSpec("b", [16, 64], datatype_from_torch(input_dtype), init_value=b),
            TensorSpec("out", [out_rows, out_cols], datatype_from_torch(output_dtype), is_output=True),
            TensorSpec("words", [1, 32], datatype_from_torch(torch.int64), init_value=control_words),
        ],
        name=f"fixpipe_{layout_mode}_{quant_mode}_relu={use_relu}_negative={negative_scale}_i32={integer_no_convert}",
        golden=lambda _tensors: expected,
        platform="a5",
        rtol=0,
        atol=0,
    )


@st.cases(
    *[
        _explicit_case(quant, relu, negative, layout)
        for layout in ("nz2nd", "nz2dn")
        for quant in (
            "no_convert",
            "f32_f16",
            "f32_bf16",
            "deqf16_scalar",
            "deqf16_vec",
            "qf322f16_pre_scalar",
        )
        for relu in (False, True)
        for negative in ((False, True) if quant.endswith("scalar") else (False,))
    ]
    + [
        _explicit_case("no_convert", relu, layout_mode=layout, integer_no_convert=True)
        for layout in ("nz2nd", "nz2dn")
        for relu in (False, True)
    ]
)
def test_explicit_quant_relu_ring_reuse(case_run):
    case_run.assert_passed()


def _fractal_case(use_relu):
    relu_mode = "normal_relu" if use_relu else "no_relu"
    a = (torch.arange(1024).reshape(32, 32) % 7 - 3).to(torch.bfloat16)
    b = (torch.arange(512).reshape(16, 32) % 11 - 5).to(torch.bfloat16)
    expected = (a.to(torch.int64) @ b.to(torch.int64).T).to(torch.float32)
    if use_relu:
        expected = expected.clamp_min(0)

    @pl.program
    class FractalPipe:
        @pl.function(type=pl.FunctionType.AIC)
        def cube(
            self,
            a: pl.Tensor[[32, 32], pl.BF16],
            b: pl.Tensor[[16, 32], pl.BF16],
            eye: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
        ):
            peer = pl.import_peer_buffer(name="nz_forward", peer_func="vector")
            back = pl.reserve_buffer(name="nz_return", size=4096)
            pl.aic_initialize_pipe(
                peer,
                dir_mask=1,
                slot_size=2048,
                slot_num=2,
                id=0,
                acc_push_epilogue={"layout": "nz2nz", "quant": "no_convert", "relu": relu_mode},
            )
            pl.aic_initialize_pipe(0, back, dir_mask=2, slot_size=2048, slot_num=2, id=1)
            lhs = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
            rhs = pl.load(b, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
            acc = pl.matmul(lhs, pl.tile.transpose_view(rhs))
            pl.tpush_to_aiv(acc, split=0, id=0)
            received = pl.tpop_from_aiv(shape=[32, 16], dtype=pl.FP32, split=0, id=1)
            identity = pl.load(eye, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            # Reading the returned NZ bytes through MAD detects physical packing.
            checked = pl.matmul(received, identity)
            pl.store(checked, [0, 0], out)
            pl.tfree_to_aiv(received, id=1)

        @pl.function(type=pl.FunctionType.AIV)
        def vector(
            self,
            a: pl.Tensor[[32, 32], pl.BF16],
            b: pl.Tensor[[16, 32], pl.BF16],
            eye: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
        ):
            local = pl.reserve_buffer(name="nz_forward", size=4096)
            peer = pl.import_peer_buffer(name="nz_return", peer_func="cube")
            pl.aiv_initialize_pipe(
                local,
                dir_mask=1,
                slot_size=2048,
                slot_num=2,
                id=0,
                acc_push_epilogue={"layout": "nz2nz", "quant": "no_convert", "relu": relu_mode},
            )
            pl.aiv_initialize_pipe(0, peer, dir_mask=2, slot_size=2048, slot_num=2, id=1)
            tile: pl.Tile[
                [32, 16],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
            ] = pl.tpop_from_aic(shape=[32, 16], dtype=pl.FP32, split=0, id=0)
            pl.tpush_to_aic(tile, split=0, id=1)
            pl.tfree_to_aic(tile, id=0)

        @pl.function(type=pl.FunctionType.Group)
        def group(
            self,
            a: pl.Tensor[[32, 32], pl.BF16],
            b: pl.Tensor[[16, 32], pl.BF16],
            eye: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
        ) -> pl.Tensor[[32, 16], pl.FP32]:
            self.cube(a, b, eye, out)
            self.vector(a, b, eye, out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[32, 32], pl.BF16],
            b: pl.Tensor[[16, 32], pl.BF16],
            eye: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
        ) -> pl.Tensor[[32, 16], pl.FP32]:
            return self.group(a, b, eye, out)

    return st.case(
        FractalPipe,
        name=f"fixpipe_nz2nz_relu={use_relu}",
        tensors=[
            TensorSpec("a", [32, 32], datatype_from_torch(torch.bfloat16), init_value=a),
            TensorSpec("b", [16, 32], datatype_from_torch(torch.bfloat16), init_value=b),
            TensorSpec("eye", [16, 16], datatype_from_torch(torch.float32), init_value=torch.eye(16)),
            TensorSpec("out", [32, 16], datatype_from_torch(torch.float32), is_output=True),
        ],
        golden=lambda _tensors: expected,
        platform="a5",
        rtol=0,
        atol=0,
    )


@st.cases(_fractal_case(False), _fractal_case(True))
def test_no_convert_fractal_round_trip(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

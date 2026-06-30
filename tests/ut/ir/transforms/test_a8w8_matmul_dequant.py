# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lowering tests for A8W8 matmul/dequant fusion."""

import pypto.language as pl
from pypto import ir, passes


def test_a8w8_matmul_dequant_lowers_to_int8_matmul_dequant_chain():
    """The explicit tensor fusion op lowers to INT8 matmul plus vector dequant."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs_i8: pl.Tensor[[16, 512], pl.INT8],
            rhs_i8: pl.Tensor[[512, 256], pl.INT8],
            act_scale: pl.Tensor[[16, 1], pl.FP32],
            weight_scale: pl.Tensor[[1, 256], pl.FP32],
        ) -> pl.Tensor[[16, 256], pl.BF16]:
            out = pl.a8w8_matmul_dequant(lhs_i8, rhs_i8, act_scale, weight_scale, out_dtype=pl.BF16)
            return out

    after = passes.convert_tensor_to_tile_ops()(Program)
    text = ir.python_print(after, format=False)

    assert "pl.tile.matmul(" in text
    assert "pl.tile.move(" in text
    assert "target_memory=pl.Mem.Vec" in text
    assert "target_type=pl.FP32" in text
    assert "a8w8_mm_vec_i32" not in text
    assert "pl.tile.cast(" in text
    assert "pl.tile.row_expand_mul(" in text
    assert "pl.tile.col_expand_mul(" in text
    assert "target_type=pl.BF16" in text
    assert "pl.tile.matmul_mx(" not in text

    after_memory = passes.infer_tile_memory_space()(after)
    memory_text = ir.python_print(after_memory, format=False)
    assert "target_memory=pl.Mem.Left" in memory_text
    assert "target_memory=pl.Mem.Right" in memory_text
    assert memory_text.count("target_memory=pl.Mem.Vec") >= 3


def test_a8w8_matmul_dequant_decode_m1_uses_scalar_activation_scale():
    """Decode M=1 can use a scalar activation scale instead of row broadcast."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs_i8: pl.Tensor[[1, 512], pl.INT8],
            rhs_i8: pl.Tensor[[512, 256], pl.INT8],
            act_scale: pl.Tensor[[1, 1], pl.FP32],
            weight_scale: pl.Tensor[[1, 256], pl.FP32],
        ) -> pl.Tensor[[1, 256], pl.BF16]:
            out = pl.a8w8_matmul_dequant(lhs_i8, rhs_i8, act_scale, weight_scale, out_dtype=pl.BF16)
            return out

    after = passes.convert_tensor_to_tile_ops()(Program)
    text = ir.python_print(after, format=False)

    assert "pl.tile.read(" in text
    assert "pl.tile.muls(" in text
    assert "pl.tile.row_expand_mul(" not in text
    assert "pl.tile.col_expand_mul(" in text

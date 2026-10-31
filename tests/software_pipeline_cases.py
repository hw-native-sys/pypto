# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Software-pipeline kernels shared by CPU codegen and device numerical tests."""

import pypto.language as pl


def make_epilogue(trips: int = 65, dynamic: bool = False):
    """Build a reduction epilogue; dynamic inactive rows retain their input values."""

    @pl.jit
    def kernel(
        acc: pl.Tensor[[trips * 64, 64], pl.INT32],
        scale: pl.Tensor[[trips * 64, 1], pl.FP32],
        coef: pl.Tensor[[1, 64], pl.FP32],
        n: pl.Scalar[pl.INDEX],
        out: pl.InOut[pl.Tensor[[trips * 64, 1], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            weights = pl.load(coef, [0, 0], [1, 64])
            scratch = pl.create_tile([64, 64], dtype=pl.FP32)
            limit = trips
            if dynamic:
                limit = pl.min(n, trips)
            for i in pl.pipeline(limit, stage=3):
                value = pl.load(acc, [i * 64, 0], [64, 64])
                factor = pl.load(scale, [i * 64, 0], [64, 1])
                fp = pl.cast(value, target_type=pl.FP32, mode="none")
                weighted = pl.col_expand_mul(pl.maximum(fp, 0.0), weights)
                reduced = pl.row_sum(weighted, scratch)
                pl.store(pl.mul(reduced, factor), [i * 64, 0], out)
        return out

    return kernel

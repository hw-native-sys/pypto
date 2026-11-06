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


def make_scratch_pressure(rows: int = 44, total_rows: int = 36960):
    """Keep stages and total work fixed while varying the second phase's tile.

    The reduction uses software slots. The following loop carries its source
    cursor, retaining the existing unrolled pipeline. Retiring the reduction's
    scratch can therefore permit larger tiles without reducing pipeline depth.
    Both outputs are observable and checked by the manual benchmark.
    """
    if rows <= 0 or total_rows < 512 or total_rows % rows:
        raise ValueError(f"Expected positive tile rows dividing total_rows >= 512, got {rows}, {total_rows}")
    trips, cols, scratch_rows = total_rows // rows, 128, 64

    @pl.jit
    def kernel(
        x: pl.Tensor[[total_rows, cols], pl.FP32],
        reduced: pl.Out[pl.Tensor[[8 * scratch_rows, 1], pl.FP32]],
        processed: pl.Out[pl.Tensor[[total_rows, cols], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            scratch = pl.create_tile([scratch_rows, cols], dtype=pl.FP32)
            for i in pl.pipeline(8, stage=2):
                value = pl.load(x, [i * scratch_rows, 0], [scratch_rows, cols])
                row_total = pl.row_sum(value, scratch)
                pl.store(row_total, [i * scratch_rows, 0], reduced)
            cursor = pl.const(0, pl.INDEX)
            for j in pl.pipeline(trips, stage=4):
                next_value = pl.load(x, [cursor * rows, 0], [rows, cols])
                pl.store(pl.add(next_value, 1.0), [j * rows, 0], processed)
                cursor = (cursor + 1) % trips
        return reduced, processed

    return kernel

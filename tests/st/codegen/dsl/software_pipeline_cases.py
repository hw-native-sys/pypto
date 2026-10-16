# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Small kernels shared by software-pipeline codegen and numerical tests."""

import pypto.language as pl

from examples.models.qwen3_jit.kernels.rmsnorm import input_rmsnorm


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


def make_nested(children=2, outer_stage=3, inner_stage=2, leaves=0, leaf_stage=4, guarded=False):
    rows = 65 * children * max(leaves, 1)
    if leaves:

        @pl.jit
        def nested_deep(
            x: pl.Tensor[[rows, 64], pl.FP32],
            groups: pl.Scalar[pl.INDEX],
            y: pl.InOut[pl.Tensor[[rows, 64], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                for g in pl.pipeline(pl.min(groups, 65), stage=outer_stage):
                    for p in pl.pipeline(children, stage=inner_stage):
                        if not guarded or (g + p) % 2 == 0:
                            for q in pl.pipeline(leaves, stage=leaf_stage):
                                row = (g * children + p) * leaves + q
                                value = pl.load(x, [row, 0], [1, 64])
                                pl.store(pl.add(value, 1.0), [row, 0], y)
            return y

        return nested_deep

    @pl.jit
    def nested_local(
        x: pl.Tensor[[rows, 64], pl.FP32],
        groups: pl.Scalar[pl.INDEX],
        y: pl.InOut[pl.Tensor[[rows, 64], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            for g in pl.pipeline(pl.min(groups, 65), stage=outer_stage):
                for p in pl.pipeline(children, stage=inner_stage):
                    row = g * children + p
                    value = pl.load(x, [row, 0], [1, 64])
                    pl.store(pl.add(value, 1.0), [row, 0], y)
        return y

    return nested_local


def make_mixed(stage: int = 3, fifo_slots: int = 3, split_mode=pl.SplitMode.UP_DOWN):
    """Exercise a C2V FIFO with row- or column-partitioned vector consumers."""
    rows_per_lane = 64 if split_mode == pl.SplitMode.UP_DOWN else 128
    columns_per_lane = 16 if split_mode == pl.SplitMode.UP_DOWN else 8
    lane_row_step = 64 if split_mode == pl.SplitMode.UP_DOWN else 0
    lane_column_step = 0 if split_mode == pl.SplitMode.UP_DOWN else 8
    child_rows = rows_per_lane // 2

    @pl.jit
    def kernel(
        x: pl.Tensor[[512, 32], pl.FP16],
        w: pl.Tensor[[32, 16], pl.FP16],
        scale: pl.Tensor[[512, 1], pl.FP32],
        n: pl.Scalar[pl.INDEX],
        y: pl.Out[pl.Tensor[[512, 16], pl.FP32]],
        z: pl.Out[pl.Tensor[[512, 16], pl.FP32]],
    ):
        with pl.spmd(1, name_hint="mixed", optimizations=[pl.cross_core_slot(slot_num=fifo_slots)]):
            block = pl.tile.get_block_idx()
            weight = pl.load(w, [block * 32, 0], [32, 16], target_memory=pl.Mem.Mat)
            for g in pl.pipeline(pl.min(n, 4), stage=stage):
                value = pl.load(x, [g * 128, 0], [128, 32], target_memory=pl.Mem.Mat)
                acc = pl.matmul(value, weight)
                for lane in pl.split_aiv(2, mode=split_mode):
                    shard = pl.aiv_shard(acc)
                    biased = pl.add(shard, 1.0)
                    for p in pl.pipeline(2, stage=2):
                        row = g * 128 + pl.cast(lane * lane_row_step, pl.INDEX) + p * child_rows
                        column = pl.cast(lane * lane_column_step, pl.INDEX)
                        factor = pl.load(scale, [row, 0], [child_rows, 1])
                        part = pl.slice(biased, [child_rows, columns_per_lane], [p * child_rows, 0])
                        pl.store(pl.row_expand_mul(part, factor), [row, column], y)
        with pl.spmd(1, name_hint="dependent"):
            block = pl.tile.get_block_idx()
            dependent_value = pl.load(y, [block * 512, 0], [512, 16])
            pl.store(pl.add(dependent_value, 1.0), [block * 512, 0], z)
        return y, z

    return kernel


def make_scratch_pressure(rows: int = 44, total_rows: int = 36960):
    """Expose retired reduction scratch consuming the next pipeline's budget.

    The first loop enters software-slot lowering. The second uses a loop-carried
    source cursor and retains the existing unrolled pipeline. At fixed stages
    and total work, scratch isolation limits its feasible tile size. Both phase
    outputs are observable so neither computation can be eliminated.
    """
    assert total_rows % rows == 0
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


@pl.jit
def qwen_input_rmsnorm(
    x: pl.Tensor[[16, 8192], pl.BF16],
    weight: pl.Tensor[[1, 8192], pl.FP32],
    y: pl.Out[pl.Tensor[[16, 8192], pl.BF16]],
):
    """Run the model's original four-stage input RMSNorm for regression checks."""
    return input_rmsnorm(x, weight, y)

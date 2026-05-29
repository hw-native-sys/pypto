# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manual phase-fence dependency compression example.

The chained snapshot program submits four flattened stages. Each stage fans out
over ``branches`` tasks, records their ``TaskId`` values in an array, and uses
that array as the full-array manual dependency for the next stage.

This example intentionally uses ``@pl.program`` because ``pl.submit`` takes
``self.<kernel>`` method references.
"""

import pypto.language as pl

DEFAULT_BRANCHES = 4
TILE_M = 32
BIG_N = 32


def chained_snapshot_shape(*, branches: int = DEFAULT_BRANCHES) -> tuple[int, int]:
    """Return the input/output tensor shape for the chained snapshot example.

    Args:
        branches: Number of parallel branches in each flattened stage.

    Returns:
        Tensor shape as ``(rows, cols)``.

    Raises:
        ValueError: If ``branches`` is not positive.
    """
    if branches <= 0:
        raise ValueError(f"branches must be positive, got {branches}")
    return 4 * branches * TILE_M, BIG_N


def build_chained_snapshot_phase_fence(*, branches: int = DEFAULT_BRANCHES):
    """Build a chained manual-scope phase-fence program.

    Args:
        branches: Number of parallel branches in each flattened stage.

    Returns:
        A ``@pl.program`` class that can be compiled or run by PyPTO tooling.
    """
    big_m, big_n = chained_snapshot_shape(branches=branches)
    tile_m = TILE_M

    @pl.program
    class ChainedSnapshotPhaseFence:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_stripe(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            bias: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, bias)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            with pl.manual_scope():
                tids = pl.array.create(branches, pl.TASK_ID)
                for a_phase, (tids_a,) in pl.range(2, init_values=(tids,)):
                    tids_next = pl.array.create(branches, pl.TASK_ID)
                    for branch in pl.parallel(branches):
                        row: pl.Scalar[pl.INDEX] = (a_phase * branches + branch) * tile_m
                        out, tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[tids_a])
                        tids_next[branch] = tid
                    tids = pl.yield_(tids_next)
                for b_phase, (tids_b,) in pl.range(2, init_values=(tids,)):
                    tids_next = pl.array.create(branches, pl.TASK_ID)
                    for branch in pl.parallel(branches):
                        row: pl.Scalar[pl.INDEX] = ((2 + b_phase) * branches + branch) * tile_m
                        out, tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[tids_b])
                        tids_next[branch] = tid
                    tids = pl.yield_(tids_next)
            return out

    return ChainedSnapshotPhaseFence


def build_chained_snapshot_manual_dummy_phase_fence(*, branches: int = DEFAULT_BRANCHES):
    """Build a chained phase-fence program with explicit dummy-task barriers.

    Args:
        branches: Number of parallel branches in each flattened stage.

    Returns:
        A ``@pl.program`` class that can be compiled or run by PyPTO tooling.
    """
    big_m, big_n = chained_snapshot_shape(branches=branches)
    tile_m = TILE_M

    @pl.program
    class ChainedSnapshotManualDummyPhaseFence:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_stripe(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            row_offset: pl.Scalar[pl.INDEX],
            bias: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            tile: pl.Tile[[tile_m, big_n], pl.FP32] = pl.load(data, [row_offset, 0], [tile_m, big_n])
            result: pl.Tile[[tile_m, big_n], pl.FP32] = pl.add(tile, bias)
            ret: pl.Tensor[[big_m, big_n], pl.FP32] = pl.store(result, [row_offset, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[big_m, big_n], pl.FP32],
            out: pl.Out[pl.Tensor[[big_m, big_n], pl.FP32]],
        ) -> pl.Tensor[[big_m, big_n], pl.FP32]:
            with pl.manual_scope():
                tids = pl.array.create(branches, pl.TASK_ID)
                for a_phase, (tids_a,) in pl.range(2, init_values=(tids,)):
                    tids_next = pl.array.create(branches, pl.TASK_ID)
                    deps = pl.system.task_dummy(deps=[tids_a])
                    for branch in pl.parallel(branches):
                        row: pl.Scalar[pl.INDEX] = (a_phase * branches + branch) * tile_m
                        out, tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[deps])
                        tids_next[branch] = tid
                    tids = pl.yield_(tids_next)
                for b_phase, (tids_b,) in pl.range(2, init_values=(tids,)):
                    tids_next = pl.array.create(branches, pl.TASK_ID)
                    deps = pl.system.task_dummy(deps=[tids_b])
                    for branch in pl.parallel(branches):
                        row: pl.Scalar[pl.INDEX] = ((2 + b_phase) * branches + branch) * tile_m
                        out, tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[deps])
                        tids_next[branch] = tid
                    tids = pl.yield_(tids_next)
            return out

    return ChainedSnapshotManualDummyPhaseFence


if __name__ == "__main__":
    rows, cols = chained_snapshot_shape()
    for builder in (build_chained_snapshot_phase_fence, build_chained_snapshot_manual_dummy_phase_fence):
        program = builder()
        print(f"{program.name}: shape=({rows}, {cols}), branches={DEFAULT_BRANCHES}")
        for fn in program.functions.values():
            print(f"  {fn.name}: {fn.func_type}")

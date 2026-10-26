# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU compilation regression for outer scalar FIFO captures in pipeline loops (#3130)."""

import pypto.language as pl
import pytest
from pypto import backend, passes
from pypto.runtime import ExecutionMode, RunConfig


@pl.jit.inline(auto_scope=False)
def conditional_offset(step: pl.Scalar[pl.INDEX]):
    if step % 2 == 0:
        offset = pl.yield_(pl.cast(0, pl.INDEX))
    else:
        offset = pl.yield_(pl.cast(16, pl.INDEX))
    return offset


@pl.jit.inline(auto_scope=False)
def direct_offset(step: pl.Scalar[pl.INDEX]):
    return step % 2 * 16


def make_entry(offset_for_step):
    @pl.jit(auto_scope=False)
    def entry(
        x: pl.Tensor[[32, 128], pl.BF16],
        weight: pl.Tensor[[128, 32], pl.BF16],
        out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
        marks: pl.Out[pl.Tensor[[2, 16], pl.INT32]],
    ):
        with pl.spmd(1):
            worker = pl.tile.get_block_idx()
            for query, (previous, recent) in pl.range(
                worker,
                2,
                init_values=(pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX)),
            ):
                for step, (old, new) in pl.range(3, init_values=(previous, recent)):
                    source = offset_for_step(step)
                    if step < 2:
                        saved = pl.yield_(source)
                    else:
                        saved = pl.yield_(new)
                    if step >= 1:
                        for chunk in pl.pipeline(2, stage=2):
                            lhs = pl.load(x, [old, chunk * 64], [16, 64], target_memory=pl.MemorySpace.Mat)
                            rhs = pl.load(weight, [chunk * 64, 0], [64, 32], target_memory=pl.MemorySpace.Mat)
                            product = pl.matmul(lhs, rhs, out_dtype=pl.FP32)
                            pl.store(product, [0, 0], out)
                    old_done, new_done = pl.yield_(new, saved)
                _previous_done, _recent_done = pl.yield_(old_done, new_done)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                marker = pl.tile.full([1, 16], dtype=pl.INT32, value=1)
                pl.store(marker, [lane, 0], marks)
        return out, marks

    return entry


def make_entry_no_pipeline(offset_for_step):
    @pl.jit(auto_scope=False)
    def entry(
        x: pl.Tensor[[32, 128], pl.BF16],
        weight: pl.Tensor[[128, 32], pl.BF16],
        out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
        marks: pl.Out[pl.Tensor[[2, 16], pl.INT32]],
    ):
        with pl.spmd(1):
            worker = pl.tile.get_block_idx()
            for query, (previous, recent) in pl.range(
                worker, 2, init_values=(pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX))
            ):
                for step, (old, new) in pl.range(3, init_values=(previous, recent)):
                    source = offset_for_step(step)
                    if step < 2:
                        saved = pl.yield_(source)
                    else:
                        saved = pl.yield_(new)
                    if step >= 1:
                        for chunk in pl.range(2):
                            lhs = pl.load(x, [old, chunk * 64], [16, 64], target_memory=pl.MemorySpace.Mat)
                            rhs = pl.load(weight, [chunk * 64, 0], [64, 32], target_memory=pl.MemorySpace.Mat)
                            product = pl.matmul(lhs, rhs, out_dtype=pl.FP32)
                            pl.store(product, [0, 0], out)
                    (old_done, new_done) = pl.yield_(new, saved)
                (_previous_done, _recent_done) = pl.yield_(old_done, new_done)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                marker = pl.tile.full([1, 16], dtype=pl.INT32, value=1)
                pl.store(marker, [lane, 0], marks)
        return (out, marks)

    return entry


@pytest.mark.parametrize("factory", [make_entry, make_entry_no_pipeline], ids=["pipeline", "range"])
@pytest.mark.parametrize("offset", [conditional_offset, direct_offset], ids=["conditional", "direct"])
def test_fifo_offset_compiles(factory, offset, tmp_path):
    # Repeated stores only exercise compilation; this kernel is never executed.
    compiled = factory(offset).compile(
        config=RunConfig(
            backend_type=backend.BackendType.Ascend910,
            execution_mode=ExecutionMode.ONBOARD,
            memory_planner=passes.MemoryPlanner.PYPTO,
            dump_passes=True,
            codegen_only=True,
            save_kernels=True,
            save_kernels_dir=str(tmp_path / "compiled"),
        )
    )
    assert compiled is not None
    assert list(tmp_path.rglob("*.pto"))
    dumps = list(tmp_path.rglob("*after_LowerPipelineLoops*.py"))
    assert dumps
    assert all("__FREE_VAR" not in path.read_text() for path in dumps)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""在 A5 上验证外部 DSL 的逻辑 SPMD 上下文及全核同步。"""

import importlib
import sys
from dataclasses import replace

import pypto.language as pl
import pytest
import torch
from pypto.pypto_core import passes

pytestmark = pytest.mark.platforms("a5")


@pytest.fixture(
    autouse=True,
    params=[passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER, passes.RuntimeKind.HOST_BUILD_GRAPH],
    ids=["tmr", "hbg"],
)
def dsl_runtime(request):
    """将编译、装配与执行置于同一运行时上下文。"""
    with passes.PassContext([], runtime=request.param):
        yield request.param


def _context_kernel():
    cb = pytest.importorskip("cannbotdsl")

    @cb.kernel
    def context(output):
        # 各核独占缓存行，避免标量写入时的缓存行覆盖。
        row = cb.get_block_idx()
        output[row, 0] = row
        output[row, 1] = cb.get_block_num()
        output[row, 2] = cb.get_subblock_id()
        output[row, 3] = cb.get_subblock_dim()

    return context


def _exchange_kernel(mixed):
    cb = pytest.importorskip("cannbotdsl")
    cube_fill_l1_zero = importlib.import_module("cannbotdsl.ops.memcpy").cube_fill_l1_zero

    @cb.kernel
    def exchange(source, scratch, output, metadata):
        index = cb.get_block_idx()
        count = cb.get_block_num()
        if mixed:
            # 保留真实的 AIC 工作，使两条 AIV 和 AIC 都参与 MIX 屏障。
            l1 = cb.Buffer(cb.MemLoc.L1, cb.dtypes.float32, cb.make_layout((1, 8)))
            cube_fill_l1_zero(l1)
            index = index * 2 + cb.get_subblock_id()
            count = count * 2
        data = cb.Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, 16)))
        for epoch in range(3):
            cb.mem_copy(data, cb.tile_slice(source[epoch, None, None], (1, 16), (index, 0)))
            cb.mem_copy(cb.tile_slice(scratch, (1, 16), (index, 0)), data)
            cb.global_sync_all()
            neighbor = (index + 1) % count
            cb.mem_copy(data, cb.tile_slice(scratch, (1, 16), (neighbor, 0)))
            cb.mem_copy(cb.tile_slice(output[epoch, None, None], (1, 16), (index, 0)), data)
            # 第二个屏障防止下一轮覆盖尚未读取的邻核数据；重复使用相同屏障 ID。
            cb.global_sync_all()
        metadata[index, 0] = count
        metadata[index, 1] = cb.get_block_idx()
        metadata[index, 2] = cb.get_subblock_id()
        metadata[index, 3] = cb.get_subblock_dim()

    return exchange


def test_cannbotdsl_partial_spmd_context(test_config):
    kernel = _context_kernel()

    @pl.jit
    def program(output: pl.Out[pl.Tensor]):
        with pl.spmd(4):
            kernel(output)
        return output

    output = torch.full((4, 16), -1, dtype=torch.int64)
    config = replace(test_config, platform="a5", rtol=0, atol=0)
    if test_config.codegen_only:
        program.compile(output, config=config)
        return
    program(output, config=config)
    expected = torch.full_like(output, -1)
    expected[:, 0] = torch.arange(4)
    expected[:, 1] = 4
    expected[:, 2] = 0
    expected[:, 3] = 1
    # 只写输出不会上传 CPU 初值，未写入的填充列不属于结果契约。
    torch.testing.assert_close(output[:, :4], expected[:, :4], rtol=0, atol=0)


@pytest.mark.parametrize("active", [0, 3, 17])
def test_cannbotdsl_runtime_slice_preserves_tail(active, test_config):
    """静态父容量上的动态切片必须保留有效区外的数据，覆盖 #3324。"""
    cb = pytest.importorskip("cannbotdsl")

    @cb.kernel
    def fill(output):
        for row in cb.range(cb.get_block_idx(), output.shape[0], cb.get_block_num()):
            for col in cb.range(output.shape[1]):
                output[row, col] = 1.0

    @pl.jit
    def program(output: pl.InOut[pl.Tensor[[17, 32], pl.FP32]], rows: pl.Scalar[pl.INDEX]):
        view = output[0:rows, :]
        with pl.spmd(4):
            fill(view)
        return output

    output = torch.full((17, 32), -37.0)
    config = replace(test_config, platform="a5", rtol=0, atol=0)
    if test_config.codegen_only:
        program.compile(output, active, config=config)
        return
    program(output, active, config=config)
    expected = torch.full_like(output, -37.0)
    expected[:active] = 1.0
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mixed", [False, True])
def test_cannbotdsl_full_grid_barrier(mixed, test_config):
    kernel = _exchange_kernel(mixed)

    @pl.jit
    def program(
        source: pl.Tensor, scratch: pl.Tensor, output: pl.Out[pl.Tensor], metadata: pl.Out[pl.Tensor]
    ):
        if mixed:
            count = pl.system.available_cluster_count()
        else:
            count = pl.system.available_aiv_count()
        with pl.spmd(count, sync_start=True):
            kernel(source, scratch, output, metadata)
        return output, metadata

    config = replace(test_config, platform="a5", rtol=0, atol=0)
    # 上限容纳 A5 的可用向量核；实际参与数量由任务返回，按运行时规模检查完整结果。
    capacity = 128
    source = torch.arange(3 * capacity * 16, dtype=torch.float32).reshape(3, capacity, 16)
    for repeat in range(2):
        values = source + repeat * 8192
        scratch = torch.full((capacity, 16), -1.0)
        output = torch.full_like(source, -1)
        metadata = torch.full((capacity, 16), -1, dtype=torch.int64)
        if test_config.codegen_only:
            program.compile(values, scratch, output, metadata, config=config)
            return
        program(values, scratch, output, metadata, config=config)
        count = int(metadata[0, 0])
        assert 1 < count <= capacity
        torch.testing.assert_close(metadata[:count, 0], torch.full((count,), count, dtype=torch.int64))
        rows = torch.arange(count)
        torch.testing.assert_close(metadata[:count, 1], rows // 2 if mixed else rows)
        torch.testing.assert_close(metadata[:count, 2], rows % 2 if mixed else torch.zeros_like(rows))
        torch.testing.assert_close(metadata[:count, 3], torch.full_like(rows, 2 if mixed else 1))
        expected = values[:, :count].roll(-1, dims=1)
        torch.testing.assert_close(output[:, :count], expected, rtol=0, atol=0)


def test_cannbotdsl_rejects_partial_dynamic_barrier(test_config, dsl_runtime):
    kernel = _exchange_kernel(True)

    @pl.jit
    def program(
        source: pl.Tensor,
        scratch: pl.Tensor,
        output: pl.Out[pl.Tensor],
        metadata: pl.Out[pl.Tensor],
        workers: pl.Scalar[pl.INDEX],
    ):
        with pl.spmd(workers, sync_start=True):
            kernel(source, scratch, output, metadata)
        return output, metadata

    args = (
        torch.zeros(3, 128, 16),
        torch.zeros(128, 16),
        torch.zeros(3, 128, 16),
        torch.zeros(128, 16, dtype=torch.int64),
    )
    config = replace(test_config, platform="a5")
    if test_config.codegen_only:
        program.compile(*args, 4, config=config)
        return
    # 核数来自运行时标量，必须在提交前以 simpler 的 INVALID_ARGS（-5）终止，
    # 不能让部分核进入屏障并造成 AICore 超时。
    # HBG 在 Host 准备图时拒绝非法启动；TMR 在设备编排结束时返回错误。
    phase = "prepare" if dsl_runtime == passes.RuntimeKind.HOST_BUILD_GRAPH else "finalize"
    with pytest.raises(RuntimeError, match=rf"{phase}_native_run failed with code -5\b"):
        program(*args, 4, config=config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))

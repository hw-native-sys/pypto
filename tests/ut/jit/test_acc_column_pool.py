# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the original physical-16 / valid-1 Acc column-pool reproducer (#3132)."""

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.runtime import ExecutionMode, RunConfig

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pl.jit.incore
def acc_pool_kernel(
    a: pl.Tensor[[1, 64], pl.FP16],
    b: pl.Tensor[[64, 64], pl.FP16],
    out: pl.InOut[pl.Tensor[[1, 128], pl.FP32]],
):
    lhs = pl.load(a, [0, 0], [1, 256], valid_shape=[1, 64], target_memory=pl.MemorySpace.Mat)
    rhs = pl.load(b, [0, 0], [256, 64], valid_shape=[64, 64], target_memory=pl.MemorySpace.Mat)
    left = pl.move(lhs, target_memory=pl.MemorySpace.Left)
    right = pl.move(rhs, target_memory=pl.MemorySpace.Right)
    pool = pl.tile.create([16, 128], pl.FP32, target_memory=pl.MemorySpace.Acc)
    for i in pl.unroll(2):
        slot = pl.tile.slice(pool, [16, 64], [0, i * 64], valid_shape=[1, 64])
        result = pl.tile.gemv_acc(slot, left, right, init_cond=True)
        pool = pl.tile.assemble(pool, result, [0, i * 64])
    return pl.store(pl.set_validshape(pool, 1, 128), [0, 0], out)


@pl.jit
def acc_pool_entry(
    a: pl.Tensor[[1, 64], pl.FP16],
    b: pl.Tensor[[64, 64], pl.FP16],
    out: pl.InOut[pl.Tensor[[1, 128], pl.FP32]],
):
    with pl.spmd(1):
        out = acc_pool_kernel(a, b, out)
    return out


@pytest.mark.parametrize("buffer_ir", [False, True])
def test_acc_column_pool_compiles_without_acc_to_acc_tmov(tmp_path: Path, buffer_ir):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=buffer_ir):
        compiled = acc_pool_entry.compile(
            config=RunConfig(
                backend_type=BackendType.Ascend910,
                execution_mode=ExecutionMode.ONBOARD,
                save_kernels=True,
                save_kernels_dir=str(tmp_path),
                dump_passes=True,
            )
        )
    sources = [path for path in compiled.output_dir.rglob("*.pto") if path.parent.name == "ptoas"]
    assert sources
    text = "\n".join(path.read_text() for path in sources)
    assert "pto.tgemv" in text
    # Shared MemRef identity alone is insufficient: check the actual PTO sent
    # to PTOAS, including both the zero and nonzero column windows.
    constants = dict(re.findall(r"(%\w+) = arith.constant (-?\d+) : index", text))
    windows = {
        name: tuple(int(constants.get(offset.strip(), offset.strip())) for offset in offsets.split(","))
        for name, offsets in re.findall(r"(%\w+) = pto.subview %\w+\[([^]]+)\]", text)
    }
    destinations = re.findall(r"pto.tgemv[^\n]*outs\((%\w+)", text)
    assert [windows[name] for name in destinations] == [(0, 0), (0, 64)]
    for line in text.splitlines():
        if "pto.tmov" in line:
            assert len(re.findall(r"loc\s*=\s*acc(?:,|>)", line)) < 2, line
    generated = "\n".join(path.read_text() for path in compiled.output_dir.rglob("*.cpp"))
    assert "TGEMV" in generated


@pl.jit.incore
def _mixed_vec_layouts(
    x: pl.Tensor[[16, 8], pl.FP32],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.InOut[pl.Tensor[[16, 1], pl.FP32]],
):
    local = pl.load(x, [0, 0], [16, 8], target_memory=pl.Mem.Vec)
    if pl.read(flag, [0]) > 0:
        window = pl.tile.slice(local, [16, 1], [0, 1])
        partial = pl.tile.set_validshape(window, 8, 1)
        chosen = pl.yield_(partial)
    else:
        window = pl.tile.slice(local, [16, 1], [0, 0])
        partial = pl.tile.set_validshape(window, 8, 1)
        chosen = pl.yield_(partial)
    return pl.store(chosen, [0, 0], out)


@pl.jit
def _mixed_vec_layouts_entry(
    x: pl.Tensor[[16, 8], pl.FP32],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.InOut[pl.Tensor[[16, 1], pl.FP32]],
):
    with pl.spmd(1):
        out = _mixed_vec_layouts(x, flag, out)
    return out


@pytest.mark.parametrize("buffer_ir", [False, True])
def test_mixed_partial_branch_layouts_compile(tmp_path: Path, buffer_ir):
    """Type coherence at a merge must also survive actual PTOAS lowering."""
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=buffer_ir):
        compiled = _mixed_vec_layouts_entry.compile(
            config=RunConfig(
                backend_type=BackendType.Ascend910,
                execution_mode=ExecutionMode.ONBOARD,
                save_kernels=True,
                save_kernels_dir=str(tmp_path),
            )
        )
    sources = [path for path in compiled.output_dir.rglob("*.pto") if path.parent.name == "ptoas"]
    assert sources
    assert all("pto.tstore" in path.read_text() for path in sources)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

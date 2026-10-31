# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile issue #3049 without copying either live If input."""

import re
import subprocess

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _get_ptoas_flags
from pypto.runtime import ExecutionMode


@pl.jit.incore
def mat_phi_core(
    x: pl.Tensor[[32, 16], pl.BF16],
    y: pl.Tensor[[16, 16], pl.BF16],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    a = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
    b = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Mat)
    rhs = pl.load(y, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
    if pl.read(flag, [0]) > 0:
        selected = pl.yield_(a)
    else:
        selected = pl.yield_(b)
    product = pl.matmul(selected, rhs, out_dtype=pl.FP32)
    pl.store(product, [0, 0], out)
    # Both inputs remain live after selection, so their storage must stay distinct.
    keep_a = pl.matmul(a, rhs, out_dtype=pl.FP32)
    pl.store(keep_a, [16, 0], out)
    keep_b = pl.matmul(b, rhs, out_dtype=pl.FP32)
    pl.store(keep_b, [32, 0], out)
    return out


@pl.jit
def mat_phi(
    x: pl.Tensor[[32, 16], pl.BF16],
    y: pl.Tensor[[16, 16], pl.BF16],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    with pl.spmd(1):
        out = mat_phi_core(x, y, flag, out)
    return out


@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
def test_if_selection_does_not_copy_live_inputs(
    tmp_path, ascend_backend, pass_verification_instruments, planner, enable_buffer_ir
):
    """Keep the public producer/consumer order and compile the resulting PTO."""
    binary = find_ptoas_binary()
    if binary is None:
        pytest.skip("PTOAS is unavailable")
    with passes.PassContext(
        pass_verification_instruments, memory_planner=planner, enable_buffer_ir=enable_buffer_ir
    ):
        ir.compile(
            mat_phi.specialize(),
            output_dir=str(tmp_path),
            backend_type=BackendType.Ascend910,
            execution_mode=ExecutionMode.ONBOARD,
            dump_passes=True,
            skip_ptoas=True,
        )
    kernels = list(tmp_path.rglob("*.pto"))
    assert kernels
    for kernel in kernels:
        text = kernel.read_text()
        assert re.search(r"scf\.if .* -> \(.*!pto\.tile_buf", text), text
        for line in text.splitlines():
            if "pto.tmov" in line:
                spaces = re.findall(r"\bloc\s*=\s*(\w+)", line)
                assert len(spaces) == 2 and spaces[0] != spaces[1], line
        result = subprocess.run(
            [binary, str(kernel), "-o", str(kernel.with_suffix(".checked.cpp")), *_get_ptoas_flags(planner)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr

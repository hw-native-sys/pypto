# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end regression for issue #2131 explicit nested-pipeline buffers."""

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.pypto_core import codegen, passes


@pl.program
class NestedExplicitBuffers:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(
        self,
        source: pl.Tensor[[64, 256], pl.BF16],
        out: pl.Tensor[[32, 256], pl.FP32],
    ):
        l1_buffers = pl.create_tile_buffers(2, [32, 256], pl.BF16, pl.Mem.Mat)
        l0b_buffers = pl.create_tile_buffers(2, [32, 128], pl.BF16, pl.Mem.Right)
        l0c_buffers = pl.create_tile_buffers(2, [16, 128], pl.FP32, pl.Mem.Acc)
        lhs = pl.create_tile([16, 32], pl.BF16, pl.Mem.Left)

        for stack in pl.pipeline(0, 2, 1, stage=2):
            l1_slot = l1_buffers[stack % 2]
            l1_value = pl.load(
                source,
                [stack * 32, 0],
                [32, 256],
                target_memory=pl.Mem.Mat,
                out=l1_slot,
            )
            for col in pl.pipeline(0, 256, 128, stage=2):
                l0_index: pl.Scalar[pl.INDEX] = (col // 128) % 2
                l0b_slot = l0b_buffers[l0_index]
                l0c_slot = l0c_buffers[l0_index]
                right = pl.tile.extract(
                    l1_value,
                    0,
                    col,
                    [32, 128],
                    target_memory=pl.Mem.Right,
                    out=l0b_slot,
                )
                acc = pl.tile.matmul(lhs, right, out=l0c_slot)
                pl.tile.store(acc, [stack * 16, col], out)
                pl.tile.release(l0b_slot)
                pl.tile.release(l0c_slot)
            pl.tile.release(l1_slot)


def _lower_and_codegen(emit_tile_addr: bool) -> str:
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    lowered = passes.lower_pipeline_loops()(NestedExplicitBuffers)
    lowered = passes.canonicalize_io_order()(lowered)
    lowered = passes.init_mem_ref()(lowered)
    if emit_tile_addr:
        lowered = passes.allocate_memory_addr()(lowered)
    func = lowered.get_function("kernel")
    assert func is not None
    return codegen.PTOCodegen().generate(ir.Program([func], "kernel", lowered.span), emit_tile_addr)


@pytest.mark.parametrize("emit_tile_addr", [True, False])
def test_nested_l1_l0_pipelines_keep_independent_explicit_rotations(emit_tile_addr):
    mlir = _lower_and_codegen(emit_tile_addr)

    allocations = [line for line in mlir.splitlines() if "pto.alloc_multi_tile" in line]
    assert len(allocations) == 3, mlir
    assert all(("addr =" in line) is emit_tile_addr for line in allocations)

    # Outer stage=2 selects two L1 slots. Each outer clone contains an inner
    # stage=2 loop selecting Right+Acc, hence 2 + 2*2*2 = 10 selections.
    assert mlir.count("pto.multi_tile_get") == 10, mlir
    assert mlir.count("pto.tload") == 2, mlir
    assert mlir.count("pto.textract") == 4, mlir
    assert mlir.count("pto.tmatmul") == 4, mlir
    assert mlir.count("pto.tstore") == 4, mlir
    assert "tile.release" not in mlir

    destination_ops = [
        line
        for line in mlir.splitlines()
        if any(op in line for op in ("pto.tload", "pto.textract", "pto.tmatmul"))
    ]
    assert destination_ops and all("outs(%" in line for line in destination_ops), mlir

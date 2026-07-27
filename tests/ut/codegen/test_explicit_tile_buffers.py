# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen tests for first-class explicit tile buffer sets."""

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.pypto_core import codegen, passes


@pl.program
class ExplicitAccBuffers:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self):
        buffers = pl.create_tile_buffers(2, [16, 128], pl.FP32, pl.Mem.Acc)
        slot = buffers[1]
        lhs = pl.create_tile([16, 32], pl.BF16, pl.Mem.Left)
        rhs = pl.create_tile([32, 128], pl.BF16, pl.Mem.Right)
        _result = pl.tile.matmul(lhs, rhs, out=slot)


@pytest.mark.parametrize("emit_tile_addr", [True, False])
def test_codegen_emits_one_multi_tile_allocation_and_slot_selection(emit_tile_addr):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    initialized = passes.init_mem_ref()(ExplicitAccBuffers)
    lowered = passes.allocate_memory_addr()(initialized) if emit_tile_addr else initialized
    func = lowered.get_function("kernel")
    assert func is not None

    mlir = codegen.PTOCodegen().generate(ir.Program([func], "kernel", lowered.span), emit_tile_addr)
    multi_allocs = [line for line in mlir.splitlines() if "pto.alloc_multi_tile" in line]
    assert len(multi_allocs) == 1, mlir
    assert ("addr =" in multi_allocs[0]) is emit_tile_addr
    assert mlir.count("pto.multi_tile_get") == 1
    assert "pto.tmatmul" in mlir

    slot_allocs = [line for line in mlir.splitlines() if "pto.alloc_tile" in line and "slot" in line]
    assert not slot_allocs, mlir
    tmatmul = next(line for line in mlir.splitlines() if "pto.tmatmul" in line)
    assert "outs(%slot" in tmatmul

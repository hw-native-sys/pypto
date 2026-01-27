# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for InitMemRefPass."""

from pypto import ir
from pypto.ir import builder
from pypto.ir.op import block
from pypto.pypto_core import DataType, passes
from pypto.pypto_core import ir as core_ir


def test_init_memref_simple():
    """Test InitMemRefPass with a simple load-compute-store sequence."""
    ib = builder.IRBuilder()

    with ib.function("test_init_memref_simple") as f:
        # Define input and output parameters (Global Tensors -> DDR)
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        # Constants for tile
        tile_height = 64
        tile_width = 64

        # Load (should infer input_a/b as DDR)
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, tile_height, tile_width))

        # Compute (UB)
        tile_sum = ib.let("tile_sum", block.add(tile_a, tile_b))

        # Store (should infer output as DDR)
        result = ib.let("result", block.store(tile_sum, 0, 0, tile_height, tile_width, output))

        ib.return_stmt(result)

    func = f.get_result()

    # Run Pass
    pass_instance = passes.InitMemRefPass()
    new_func = pass_instance.run(func)

    # --- Assertions ---

    # 1. Check Params (DDR)
    # input_a, input_b, output should all be DDR with size 64*64*4 = 16384
    params = {p.name: p for p in new_func.params}
    for name in ["input_a", "input_b", "output"]:
        p = params[name]
        # Cast to ShapedType to access memref
        assert isinstance(p.type, core_ir.ShapedType)
        assert p.type.memref is not None
        assert p.type.memref.memory_space_ == core_ir.MemorySpace.DDR
        assert p.type.memref.size_ == 16384
        assert isinstance(p.type.memref.addr_, core_ir.ConstInt)
        assert p.type.memref.addr_.value == 0

    # 2. Check Body Variables (UB)
    assert isinstance(new_func.body, ir.SeqStmts)
    stmts = new_func.body.stmts

    # tile_a, tile_b, tile_sum should all be UB with size 64*64*4 = 16384
    # stmts[0] is tile_a, stmts[1] is tile_b, stmts[2] is tile_sum
    for i, name in enumerate(["tile_a", "tile_b", "tile_sum"]):
        stmt = stmts[i]
        assert isinstance(stmt, ir.AssignStmt)
        var = stmt.var
        assert var.name == name
        assert isinstance(var.type, core_ir.ShapedType)
        assert var.type.memref is not None
        assert var.type.memref.memory_space_ == core_ir.MemorySpace.UB
        assert var.type.memref.size_ == 16384
        assert isinstance(var.type.memref.addr_, core_ir.ConstInt)
        assert var.type.memref.addr_.value == 0

    # 3. Verify Var Identity (Identity check is stronger than property check)
    # input_a in block.load must be the EXACT same object as in params
    stmt0 = stmts[0]
    assert isinstance(stmt0, ir.AssignStmt)
    call_load_a = stmt0.value
    assert isinstance(call_load_a, ir.Call)
    assert call_load_a.args[0] is params["input_a"]

    # input_b in block.load must be the EXACT same object as in params
    stmt1 = stmts[1]
    assert isinstance(stmt1, ir.AssignStmt)
    call_load_b = stmt1.value
    assert isinstance(call_load_b, ir.Call)
    assert call_load_b.args[0] is params["input_b"]

    # output in block.store must be the EXACT same object as in params
    stmt3 = stmts[3]
    assert isinstance(stmt3, ir.AssignStmt)
    call_store = stmt3.value
    assert isinstance(call_store, ir.Call)
    assert call_store.args[5] is params["output"]

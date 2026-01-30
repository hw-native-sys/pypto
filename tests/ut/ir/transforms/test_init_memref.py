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
    pass_instance = passes.init_mem_ref()
    program = core_ir.Program([func], "test_init_memref_simple", ir.Span.unknown())
    program = pass_instance(program)
    new_func = list(program.functions.values())[0]

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


def test_init_memref_matmul():
    """Test InitMemRefPass with load->move->matmul->store sequence."""
    ib = builder.IRBuilder()

    with ib.function("test_init_memref_matmul") as f:
        # Define input and output parameters (Global Tensors -> DDR)
        input_a = f.param("input_a", ir.TensorType([32, 32], DataType.FP16))
        input_b = f.param("input_b", ir.TensorType([32, 32], DataType.FP16))
        output = f.param("output", ir.TensorType([32, 32], DataType.FP16))
        f.return_type(ir.TensorType([32, 32], DataType.FP16))

        # Constants for tile
        tile_height = 32
        tile_width = 32

        # Load tile_a to UB (target_memory=1, default)
        tile_a_ub = ib.let("tile_a_ub", block.load(input_a, 0, 0, tile_height, tile_width, target_memory=1))

        # Load tile_b to L1 (target_memory=2)
        tile_b_l1 = ib.let("tile_b_l1", block.load(input_b, 0, 0, tile_height, tile_width, target_memory=2))

        # Move tile_a from UB to L0A (target_memory=3)
        tile_a_l0a = ib.let("tile_a_l0a", block.move(tile_a_ub, target_memory=3))

        # Move tile_b from L1 to L0B (target_memory=4)
        tile_b_l0b = ib.let("tile_b_l0b", block.move(tile_b_l1, target_memory=4))

        # Compute matmul (result in L0C by default)
        tile_result = ib.let("tile_result", block.matmul(tile_a_l0a, tile_b_l0b))

        # Store result back to DDR
        result = ib.let("result", block.store(tile_result, 0, 0, tile_height, tile_width, output))

        ib.return_stmt(result)

    func = f.get_result()

    # Run Pass
    pass_instance = passes.init_mem_ref()
    program = core_ir.Program([func], "test_init_memref_matmul", ir.Span.unknown())
    program = pass_instance(program)
    new_func = list(program.functions.values())[0]

    # --- Assertions ---

    # 1. Check Params (DDR)
    # input_a, input_b, output should all be DDR with size 32*32*2 = 2048 (FP16)
    params = {p.name: p for p in new_func.params}
    for name in ["input_a", "input_b", "output"]:
        p = params[name]
        assert isinstance(p.type, core_ir.ShapedType)
        assert p.type.memref is not None
        assert p.type.memref.memory_space_ == core_ir.MemorySpace.DDR
        assert p.type.memref.size_ == 2048
        assert isinstance(p.type.memref.addr_, core_ir.ConstInt)
        assert p.type.memref.addr_.value == 0

    # 2. Check Body Variables with correct memory spaces
    assert isinstance(new_func.body, ir.SeqStmts)
    stmts = new_func.body.stmts

    # Expected memory spaces for each variable
    expected_memory_spaces = [
        ("tile_a_ub", core_ir.MemorySpace.UB),  # load with target_space=0
        ("tile_b_l1", core_ir.MemorySpace.L1),  # load with target_space=1
        ("tile_a_l0a", core_ir.MemorySpace.L0A),  # move to target_space=2
        ("tile_b_l0b", core_ir.MemorySpace.L0B),  # move to target_space=3
        ("tile_result", core_ir.MemorySpace.L0C),  # matmul result (default L0C)
        ("result", core_ir.MemorySpace.DDR),  # store returns output tensor (DDR)
    ]

    for i, (expected_name, expected_space) in enumerate(expected_memory_spaces):
        stmt = stmts[i]
        assert isinstance(stmt, ir.AssignStmt)
        var = stmt.var
        assert var.name == expected_name, f"Expected {expected_name}, got {var.name}"
        assert isinstance(var.type, core_ir.ShapedType)
        assert var.type.memref is not None
        assert var.type.memref.memory_space_ == expected_space, (
            f"Variable {expected_name} expected {expected_space}, got {var.type.memref.memory_space_}"
        )
        assert var.type.memref.size_ == 2048
        assert isinstance(var.type.memref.addr_, core_ir.ConstInt)
        assert var.type.memref.addr_.value == 0

    # 3. Verify data flow: input tensors are referenced correctly
    # tile_a_ub load should reference input_a (DDR)
    stmt0 = stmts[0]
    assert isinstance(stmt0, ir.AssignStmt)
    call_load_a = stmt0.value
    assert isinstance(call_load_a, ir.Call)
    assert call_load_a.args[0] is params["input_a"]

    # tile_b_l1 load should reference input_b (DDR)
    stmt1 = stmts[1]
    assert isinstance(stmt1, ir.AssignStmt)
    call_load_b = stmt1.value
    assert isinstance(call_load_b, ir.Call)
    assert call_load_b.args[0] is params["input_b"]

    # result store should reference output (DDR)
    stmt5 = stmts[5]
    assert isinstance(stmt5, ir.AssignStmt)
    call_store = stmt5.value
    assert isinstance(call_store, ir.Call)
    assert call_store.args[5] is params["output"]

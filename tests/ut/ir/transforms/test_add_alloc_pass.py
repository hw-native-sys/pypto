# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
from pypto import DataType, ir, passes
from pypto.ir import builder
from pypto.ir.op import block
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def count_alloc_operations(func):
    """Count the number of block.alloc operations in a function.

    Args:
        func: Function to analyze

    Returns:
        Number of block.alloc operations found
    """
    if not isinstance(func.body, ir.SeqStmts):
        return 0

    count = 0
    for stmt in func.body.stmts:
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            if stmt.value.op.name == "block.alloc":
                count += 1
    return count


def get_alloc_statement_indices(func):
    """Get the indices of all block.alloc statements in a function.

    Args:
        func: Function to analyze

    Returns:
        List of statement indices where block.alloc operations are found
    """
    if not isinstance(func.body, ir.SeqStmts):
        return []

    indices = []
    for i, stmt in enumerate(func.body.stmts):
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            if stmt.value.op.name == "block.alloc":
                indices.append(i)
    return indices


def get_alloc_addresses(func):
    """Get addresses from all block.alloc operations in a function.

    Args:
        func: Function to analyze

    Returns:
        List of (var_name, addr) tuples in the order they appear
    """
    if not isinstance(func.body, ir.SeqStmts):
        return []

    addrs = []
    for stmt in func.body.stmts:
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            if stmt.value.op.name == "block.alloc" and len(stmt.value.args) >= 2:
                # Second argument is the address
                addr_expr = stmt.value.args[1]
                if isinstance(addr_expr, ir.ConstInt):
                    addrs.append((stmt.var.name, addr_expr.value))
    return addrs


def get_memref_addresses_from_tiles(func):
    """Get MemRef addresses from TileType variables in the function body.

    Args:
        func: Function to analyze

    Returns:
        Dict mapping variable name to MemRef address
    """
    if not isinstance(func.body, ir.SeqStmts):
        return {}

    memref_addrs = {}
    for stmt in func.body.stmts:
        if isinstance(stmt, ir.AssignStmt):
            # Skip alloc statements
            if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "block.alloc":
                continue

            var_type = stmt.var.type
            if isinstance(var_type, ir.TileType) and var_type.memref is not None:
                memref = var_type.memref
                if isinstance(memref.addr_, ir.ConstInt):
                    memref_addrs[stmt.var.name] = memref.addr_.value

    return memref_addrs


def test_add_alloc_pass_simple():
    """Test AddAllocPass with a simple function containing TileType variables.

    Verifies that:
    1. Alloc operations are created for each unique MemRef
    2. Alloc operations are placed at the beginning of the function
    3. Addresses are 32-byte aligned
    4. MemRef addr_ fields are updated with allocated addresses
    """
    ib = builder.IRBuilder()

    with ib.function("test_simple_alloc", type=ir.FunctionType.InCore) as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_height = 64
        tile_width = 64

        tile_a = ib.let("tile_a", block.load(input_a, [0, 0], [tile_height, tile_width]))
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))
        result = ib.let("result", block.store(tile_b, [0, 0], [tile_height, tile_width], output))

        ib.return_stmt(result)

    func = f.get_result()

    # Wrap function in Program
    program = ir.Program([func], "test_simple_alloc", ir.Span.unknown())

    # Run InitMemRefPass first to initialize MemRef for tiles
    init_pass = passes.init_mem_ref()
    program_with_memref = init_pass(program)

    # Run the AddAllocPass
    add_alloc_pass = passes.add_alloc()
    optimized_program = add_alloc_pass(program_with_memref)

    # Extract the function from the program
    optimized_func = list(optimized_program.functions.values())[0]

    # Verify alloc operations were added
    alloc_count = count_alloc_operations(optimized_func)
    assert alloc_count > 0, "AddAllocPass should create at least one alloc operation"

    # Verify alloc operations are at the beginning
    alloc_indices = get_alloc_statement_indices(optimized_func)
    assert len(alloc_indices) > 0, "Should have alloc operations"

    # First statement should be an alloc
    assert alloc_indices[0] == 0, "First alloc should be at index 0"

    # All alloc operations should be consecutive at the beginning
    for i, idx in enumerate(alloc_indices):
        assert idx == i, f"Alloc operations should be at the beginning, but found at index {idx}"

    # Verify addresses are 32-byte aligned
    alloc_addrs = get_alloc_addresses(optimized_func)
    assert len(alloc_addrs) > 0, "Should have alloc addresses"

    for var_name, addr in alloc_addrs:
        # Check 32-byte alignment
        assert addr % 32 == 0, f"Address {addr} for {var_name} should be 32-byte aligned"

    # Verify MemRef addr_ fields are updated
    memref_addrs = get_memref_addresses_from_tiles(optimized_func)
    assert len(memref_addrs) > 0, "Should have MemRef addresses in TileType variables"

    # Verify MemRef addresses match alloc addresses and check specific values
    # Expected addresses: tile_a=0, tile_b=16384 (64*64*4 bytes per tile)
    expected_addrs = {"tile_a": 0, "tile_b": 16384}
    for var_name, expected_addr in expected_addrs.items():
        assert var_name in memref_addrs, f"Variable {var_name} not found in MemRef addresses"
        actual_addr = memref_addrs[var_name]
        assert actual_addr == expected_addr, f"{var_name}: expected addr={expected_addr}, got {actual_addr}"
        assert actual_addr % 32 == 0, f"MemRef address {actual_addr} for {var_name} should be 32-byte aligned"


def test_add_alloc_pass_multiple_tiles():
    """Test AddAllocPass with multiple TileType variables.

    Verifies that:
    1. Each unique MemRef gets its own alloc operation
    2. Multiple alloc operations are created for multiple tiles
    3. Addresses are 32-byte aligned
    """
    ib = builder.IRBuilder()

    with ib.function("test_multiple_tiles") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_height = 64
        tile_width = 64

        # Create 4 tiles to test multiple allocs
        tile_a = ib.let("tile_a", block.load(input_a, [0, 0], [tile_height, tile_width]))
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))
        tile_c = ib.let("tile_c", block.add(tile_b, tile_b))
        result = ib.let("result", block.store(tile_c, [0, 0], [tile_height, tile_width], output))

        ib.return_stmt(result)

    func = f.get_result()

    # Wrap function in Program
    program = ir.Program([func], "test_multiple_tiles", ir.Span.unknown())

    # Run InitMemRefPass first to initialize MemRef for tiles
    init_pass = passes.init_mem_ref()
    program_with_memref = init_pass(program)

    # Run the AddAllocPass
    add_alloc_pass = passes.add_alloc()
    optimized_program = add_alloc_pass(program_with_memref)

    # Extract the function from the program
    optimized_func = list(optimized_program.functions.values())[0]

    # Verify multiple alloc operations were created
    alloc_count = count_alloc_operations(optimized_func)
    # We expect 3 allocs for the 3 TileType variables (tile_a, tile_b, tile_c)
    # The result variable is TensorType and reuses the output parameter's MemRef
    assert alloc_count == 3, f"Expected 3 alloc operations for 3 tiles, but got {alloc_count}"

    # Verify alloc operations are at the beginning
    alloc_indices = get_alloc_statement_indices(optimized_func)
    for i, idx in enumerate(alloc_indices):
        assert idx == i, "All alloc operations should be at the beginning"

    # Verify addresses are aligned
    alloc_addrs = get_alloc_addresses(optimized_func)
    assert len(alloc_addrs) == 3, f"Expected 3 alloc addresses, got {len(alloc_addrs)}"

    for var_name, addr in alloc_addrs:
        # Check 32-byte alignment
        assert addr % 32 == 0, f"Address {addr} for {var_name} should be 32-byte aligned"

    # Verify specific address values
    # Expected addresses: tile_a=0, tile_b=16384, tile_c=32768 (64*64*4 bytes per tile)
    memref_addrs = get_memref_addresses_from_tiles(optimized_func)
    expected_addrs = {"tile_a": 0, "tile_b": 16384, "tile_c": 32768}
    for var_name, expected_addr in expected_addrs.items():
        assert var_name in memref_addrs, f"Variable {var_name} not found in MemRef addresses"
        actual_addr = memref_addrs[var_name]
        assert actual_addr == expected_addr, f"{var_name}: expected addr={expected_addr}, got {actual_addr}"


def test_add_alloc_pass_with_ptoas_strategy():
    """Test AddAllocPass as part of PTOAS optimization strategy.

    Verifies that:
    1. AddAllocPass runs after InitMemRefPass and BasicMemoryReusePass
    2. All three passes work together correctly
    """
    ib = builder.IRBuilder()

    with ib.function("test_ptoas") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_height = 64
        tile_width = 64

        tile_a = ib.let("tile_a", block.load(input_a, [0, 0], [tile_height, tile_width]))
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))
        result = ib.let("result", block.store(tile_b, [0, 0], [tile_height, tile_width], output))

        ib.return_stmt(result)

    func = f.get_result()

    # Wrap function in Program for PassManager
    program = ir.Program([func], "test_ptoas", ir.Span.unknown())

    # Run PTOAS strategy (which includes AddAllocPass)
    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    optimized_result = pm.run_passes(program)
    assert isinstance(optimized_result, ir.Program), "Result should be a Program"

    # Extract the function from the program
    optimized_func = list(optimized_result.functions.values())[0]

    # Verify alloc operations were added
    alloc_count = count_alloc_operations(optimized_func)
    assert alloc_count > 0, "PTOAS strategy should include AddAllocPass which creates alloc operations"

    # Verify the function is still valid
    assert optimized_func is not None
    assert optimized_func.name == "test_ptoas"
    assert isinstance(optimized_func.body, ir.SeqStmts)


def test_add_alloc_pass_with_memory_reuse():
    """Test AddAllocPass behavior when memory reuse happens.

    Verifies that:
    1. AddAllocPass runs after BasicMemoryReusePass
    2. When variables share MemRef due to reuse, only one alloc is created for that MemRef
    """
    ib = builder.IRBuilder()

    with ib.function("test_with_reuse") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_height = 64
        tile_width = 64

        # Sequential operations allow memory reuse
        tile_a = ib.let("tile_a", block.load(input_a, [0, 0], [tile_height, tile_width]))
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))
        tile_c = ib.let("tile_c", block.add(tile_b, tile_b))
        result = ib.let("result", block.store(tile_c, [0, 0], [tile_height, tile_width], output))

        ib.return_stmt(result)

    func = f.get_result()

    # Wrap function in Program for PassManager
    program = ir.Program([func], "test_with_reuse", ir.Span.unknown())

    # Run PTOAS strategy
    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    optimized_result = pm.run_passes(program)
    assert isinstance(optimized_result, ir.Program), "Result should be a Program"

    # Extract the function from the program
    optimized_func = list(optimized_result.functions.values())[0]

    # Verify alloc operations were added
    alloc_count = count_alloc_operations(optimized_func)
    assert alloc_count > 0, "Should create alloc operations even with memory reuse"

    # Verify the function structure
    assert isinstance(optimized_func.body, ir.SeqStmts)
    stmts = optimized_func.body.stmts

    # Verify alloc operations come before other operations
    alloc_indices = get_alloc_statement_indices(optimized_func)
    if alloc_indices:
        last_alloc_idx = max(alloc_indices)
        first_non_alloc_idx = None
        for i, stmt in enumerate(stmts):
            if i > last_alloc_idx and isinstance(stmt, ir.AssignStmt):
                if not (isinstance(stmt.value, ir.Call) and stmt.value.op.name == "block.alloc"):
                    first_non_alloc_idx = i
                    break

        if first_non_alloc_idx is not None:
            assert last_alloc_idx < first_non_alloc_idx, (
                "All alloc operations should come before other operations"
            )

    # Verify addresses are 32-byte aligned
    alloc_addrs = get_alloc_addresses(optimized_func)
    for var_name, addr in alloc_addrs:
        assert addr % 32 == 0, f"Address {addr} for {var_name} should be 32-byte aligned"

    # Verify MemRef addresses are aligned
    memref_addrs = get_memref_addresses_from_tiles(optimized_func)
    for var_name, addr in memref_addrs.items():
        assert addr % 32 == 0, f"MemRef address {addr} for {var_name} should be 32-byte aligned"


def test_add_alloc_pass_empty_function():
    """Test AddAllocPass with a function that has no TileType variables.

    Verifies that:
    1. The pass handles functions with no tiles gracefully
    2. No alloc operations are created for non-TileType variables
    """
    ib = builder.IRBuilder()

    with ib.function("test_empty") as f:
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))
        ib.return_stmt(output)

    func = f.get_result()

    # Wrap function in Program
    program = ir.Program([func], "test_empty", ir.Span.unknown())

    # Run the AddAllocPass
    add_alloc_pass = passes.add_alloc()
    optimized_program = add_alloc_pass(program)

    # Extract the function from the program
    optimized_func = list(optimized_program.functions.values())[0]

    # Verify no alloc operations were created (since there are no TileType variables)
    alloc_count = count_alloc_operations(optimized_func)
    assert alloc_count == 0, "Should not create alloc operations for non-TileType variables"

    # Verify the function is still valid
    assert optimized_func is not None
    assert optimized_func.name == "test_empty"


@pytest.mark.xfail(reason="Need to double check if AddAllocPass depends on InitMemRefPass")
def test_add_alloc_pass_alloc_placement():
    """Test that AddAllocPass correctly places alloc operations at the function beginning.

    Verifies that:
    1. All alloc statements are placed at the very beginning
    2. No alloc statements are intermixed with other operations
    3. The order of operations after alloc is preserved
    """
    ib = builder.IRBuilder()

    with ib.function("test_placement") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_a = ib.let("tile_a", block.load(input_a, offsets=[0, 0], shapes=[64, 64]))
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))
        result = ib.let("result", block.store(tile_b, offsets=[0, 0], shapes=[64, 64], output_tensor=output))

        ib.return_stmt(result)

    func = f.get_result()

    # Wrap function in Program
    program = ir.Program([func], "test_placement", ir.Span.unknown())

    # Run the AddAllocPass
    add_alloc_pass = passes.add_alloc()
    optimized_program = add_alloc_pass(program)

    # Extract the function from the program
    optimized_func = list(optimized_program.functions.values())[0]

    assert isinstance(optimized_func.body, ir.SeqStmts)
    stmts = optimized_func.body.stmts

    # Find first non-alloc statement index
    first_non_alloc_idx = None
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, ir.AssignStmt):
            if not (isinstance(stmt.value, ir.Call) and stmt.value.op.name == "mem.alloc"):
                first_non_alloc_idx = i
                break

    # All statements before first_non_alloc_idx should be alloc operations
    if first_non_alloc_idx is not None:
        for i in range(first_non_alloc_idx):
            stmt = stmts[i]
            assert isinstance(stmt, ir.AssignStmt), f"Statement {i} should be AssignStmt"
            assert isinstance(stmt.value, ir.Call), f"Statement {i} should have a Call value"
            assert stmt.value.op.name == "block.alloc", f"Statement {i} should be a block.alloc operation"

    # Verify the original operation order is preserved
    tile_a_found = False
    tile_a_idx = None
    tile_b_idx = None
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, ir.AssignStmt):
            if stmt.var.name == "tile_a":
                tile_a_found = True
                tile_a_idx = i
            elif stmt.var.name == "tile_b":
                assert tile_a_found, "tile_b should come after tile_a"
                assert tile_a_idx is not None, "tile_a_idx should be set"
                tile_b_idx = i
                assert tile_a_idx < tile_b_idx, "Operations order should be preserved"


def test_add_alloc_pass_raw_pointer_uniqueness():
    """Test that AddAllocPass uses raw pointer comparison for MemRef uniqueness.

    Verifies that:
    1. Only one alloc is created for the same shared_ptr MemRef
    2. Different shared_ptr objects result in different alloc operations
    """
    ib = builder.IRBuilder()

    with ib.function("test_pointer_uniqueness") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        # Create 4 tiles with different MemRef objects
        tile_a = ib.let("tile_a", block.load(input_a, offsets=[0, 0], shapes=[64, 64]))
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))
        tile_c = ib.let("tile_c", block.add(tile_b, tile_b))
        result = ib.let("result", block.store(tile_c, offsets=[0, 0], shapes=[64, 64], output_tensor=output))

        ib.return_stmt(result)

    func = f.get_result()

    # Before any pass, each tile should have a unique MemRef
    # Wrap function in Program
    program = ir.Program([func], "test_pointer_uniqueness", ir.Span.unknown())

    # Run InitMemRefPass first to initialize MemRef
    init_pass = passes.init_mem_ref()
    program_with_memref = init_pass(program)

    # Now run AddAllocPass
    add_alloc_pass = passes.add_alloc()
    optimized_program = add_alloc_pass(program_with_memref)

    # Extract the function from the program
    optimized_func = list(optimized_program.functions.values())[0]

    # Count alloc operations
    alloc_count = count_alloc_operations(optimized_func)

    # We expect 3 allocs for the 3 TileType variables (tile_a, tile_b, tile_c)
    # The result variable is TensorType and reuses the output parameter's MemRef
    assert alloc_count == 3, f"Expected 3 unique MemRef objects, but got {alloc_count} allocs"

    # Verify alloc operations are placed at the beginning
    alloc_indices = get_alloc_statement_indices(optimized_func)
    assert len(alloc_indices) == alloc_count, "All alloc operations should be identified"

    for i, idx in enumerate(alloc_indices):
        assert idx == i, "Alloc operations should be consecutive at the beginning"

    # Verify specific address values and 32-byte alignment
    # Expected addresses: tile_a=0, tile_b=16384, tile_c=32768 (64*64*4 bytes per tile)
    memref_addrs = get_memref_addresses_from_tiles(optimized_func)
    expected_addrs = {"tile_a": 0, "tile_b": 16384, "tile_c": 32768}
    for var_name, expected_addr in expected_addrs.items():
        assert var_name in memref_addrs, f"Variable {var_name} not found in MemRef addresses"
        actual_addr = memref_addrs[var_name]
        assert actual_addr == expected_addr, f"{var_name}: expected addr={expected_addr}, got {actual_addr}"
        assert actual_addr % 32 == 0, f"Address {actual_addr} for {var_name} should be 32-byte aligned"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

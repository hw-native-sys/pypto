# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl
import pytest
from pypto import ir, passes


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


def _prepare_and_run_add_alloc(program):
    """Prepare IR with memrefs (test setup), then run the pass under test.

    init_mem_ref() is test setup that attaches memrefs to tiles.
    add_alloc() is the pass under test.
    """
    program = passes.init_mem_ref()(program)  # Test setup: attach memrefs
    program = passes.add_alloc()(program)  # Pass under test
    return program


def test_add_alloc_pass_simple():
    """Test AddAllocPass with a simple function containing TileType variables.

    Verifies that:
    1. Alloc operations are created for each unique MemRef
    2. Alloc operations are placed at the beginning of the function
    3. Addresses are 32-byte aligned
    4. MemRef addr_ fields are updated with allocated addresses
    """

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], [64, 64], output)
            return result

    optimized_program = _prepare_and_run_add_alloc(Before)
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

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_b, tile_b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], [64, 64], output)
            return result

    optimized_program = _prepare_and_run_add_alloc(Before)
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


def test_add_alloc_pass_empty_function():
    """Test AddAllocPass with a function that has no TileType variables.

    Verifies that:
    1. The pass handles functions with no tiles gracefully
    2. No alloc operations are created for non-TileType variables
    """

    @pl.program
    class Before:
        @pl.function
        def main(self, output: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
            return output

    optimized_program = passes.add_alloc()(Before)
    optimized_func = list(optimized_program.functions.values())[0]

    # Verify no alloc operations were created (since there are no TileType variables)
    alloc_count = count_alloc_operations(optimized_func)
    assert alloc_count == 0, "Should not create alloc operations for non-TileType variables"

    # Verify the function is still valid
    assert optimized_func is not None
    assert optimized_func.name == "main"


def test_add_alloc_pass_alloc_placement():
    """Test that AddAllocPass correctly places alloc operations at the function beginning.

    Verifies that:
    1. All alloc statements are placed at the very beginning
    2. No alloc statements are intermixed with other operations
    3. The order of operations after alloc is preserved
    """

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], [64, 64], output)
            return result

    optimized_program = _prepare_and_run_add_alloc(Before)
    optimized_func = list(optimized_program.functions.values())[0]

    assert isinstance(optimized_func.body, ir.SeqStmts)
    stmts = optimized_func.body.stmts

    # Find first non-alloc statement index
    first_non_alloc_idx = None
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, ir.AssignStmt):
            if not (isinstance(stmt.value, ir.Call) and stmt.value.op.name == "block.alloc"):
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

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_b, tile_b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], [64, 64], output)
            return result

    optimized_program = _prepare_and_run_add_alloc(Before)
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

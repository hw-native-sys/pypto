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


def _iter_all_stmts(func):
    """Iterate all AssignStmt/EvalStmt in function body (handles SeqStmts/OpStmts)."""
    if not isinstance(func.body, ir.SeqStmts):
        return
    for child in func.body.stmts:
        if isinstance(child, ir.OpStmts):
            yield from child.stmts
        elif isinstance(child, (ir.AssignStmt, ir.EvalStmt)):
            yield child


def count_alloc_operations(func):
    """Count the number of block.alloc operations in a function."""
    count = 0
    for stmt in _iter_all_stmts(func):
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            if stmt.value.op.name == "block.alloc":
                count += 1
    return count


def get_alloc_addresses(func):
    """Get addresses from all block.alloc operations in a function.

    Returns:
        List of (var_name, addr) tuples in the order they appear
    """
    addrs = []
    for stmt in _iter_all_stmts(func):
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            if stmt.value.op.name == "block.alloc" and len(stmt.value.args) >= 2:
                addr_expr = stmt.value.args[1]
                if isinstance(addr_expr, ir.ConstInt):
                    addrs.append((stmt.var.name, addr_expr.value))
    return addrs


def get_memref_addresses_from_tiles(func):
    """Get MemRef addresses from TileType variables in the function body.

    Returns:
        Dict mapping variable name to MemRef address
    """
    memref_addrs = {}
    for stmt in _iter_all_stmts(func):
        if isinstance(stmt, ir.AssignStmt):
            if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "block.alloc":
                continue
            var_type = stmt.var.type
            if isinstance(var_type, ir.TileType) and var_type.memref is not None:
                memref = var_type.memref
                if isinstance(memref.addr_, ir.ConstInt):
                    memref_addrs[stmt.var.name] = memref.addr_.value
    return memref_addrs


def _prepare_and_run_allocate_memory_addr(program):
    """Prepare IR with memrefs and alloc ops, then run the address allocation pass.

    init_mem_ref() creates MemRefs and alloc ops with addr=-1.
    allocate_memory_addr() assigns real addresses.
    """
    program = passes.init_mem_ref()(program)
    program = passes.allocate_memory_addr()(program)
    return program


def test_allocate_memory_addr_simple():
    """Test AllocateMemoryAddr with a simple function containing TileType variables.

    Verifies that:
    1. Alloc operations exist with real addresses
    2. Addresses are 32-byte aligned
    3. MemRef addr_ fields are updated with allocated addresses
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

    optimized_program = _prepare_and_run_allocate_memory_addr(Before)
    optimized_func = list(optimized_program.functions.values())[0]

    alloc_count = count_alloc_operations(optimized_func)
    assert alloc_count > 0, "Should create at least one alloc operation"

    alloc_addrs = get_alloc_addresses(optimized_func)
    assert len(alloc_addrs) > 0, "Should have alloc addresses"

    for var_name, addr in alloc_addrs:
        assert addr % 32 == 0, f"Address {addr} for {var_name} should be 32-byte aligned"

    memref_addrs = get_memref_addresses_from_tiles(optimized_func)
    assert len(memref_addrs) > 0, "Should have MemRef addresses in TileType variables"

    expected_addrs = {"tile_a": 0, "tile_b": 16384}
    for var_name, expected_addr in expected_addrs.items():
        assert var_name in memref_addrs, f"Variable {var_name} not found in MemRef addresses"
        actual_addr = memref_addrs[var_name]
        assert actual_addr == expected_addr, f"{var_name}: expected addr={expected_addr}, got {actual_addr}"


def test_allocate_memory_addr_multiple_tiles():
    """Test AllocateMemoryAddr with multiple TileType variables.

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

    optimized_program = _prepare_and_run_allocate_memory_addr(Before)
    optimized_func = list(optimized_program.functions.values())[0]

    alloc_count = count_alloc_operations(optimized_func)
    assert alloc_count == 3, f"Expected 3 alloc operations for 3 tiles, but got {alloc_count}"

    alloc_addrs = get_alloc_addresses(optimized_func)
    assert len(alloc_addrs) == 3, f"Expected 3 alloc addresses, got {len(alloc_addrs)}"

    for var_name, addr in alloc_addrs:
        assert addr % 32 == 0, f"Address {addr} for {var_name} should be 32-byte aligned"

    memref_addrs = get_memref_addresses_from_tiles(optimized_func)
    expected_addrs = {"tile_a": 0, "tile_b": 16384, "tile_c": 32768}
    for var_name, expected_addr in expected_addrs.items():
        assert var_name in memref_addrs, f"Variable {var_name} not found in MemRef addresses"
        actual_addr = memref_addrs[var_name]
        assert actual_addr == expected_addr, f"{var_name}: expected addr={expected_addr}, got {actual_addr}"


def test_allocate_memory_addr_empty_function():
    """Test AllocateMemoryAddr with a function that has no TileType variables.

    Verifies that:
    1. The pass handles functions with no tiles gracefully
    2. No alloc operations are created for non-TileType variables
    """

    @pl.program
    class Before:
        @pl.function
        def main(self, output: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
            return output

    optimized_program = passes.allocate_memory_addr()(Before)
    optimized_func = list(optimized_program.functions.values())[0]

    alloc_count = count_alloc_operations(optimized_func)
    assert alloc_count == 0, "Should not create alloc operations for non-TileType variables"

    assert optimized_func is not None
    assert optimized_func.name == "main"


def test_allocate_memory_addr_alloc_in_first_opstmts():
    """Test that alloc operations are placed in the first OpStmts.

    Verifies that:
    1. Alloc statements are inside an OpStmts
    2. Alloc statements come before other statements in the same OpStmts
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

    optimized_program = _prepare_and_run_allocate_memory_addr(Before)
    optimized_func = list(optimized_program.functions.values())[0]

    assert isinstance(optimized_func.body, ir.SeqStmts)
    first_child = optimized_func.body.stmts[0]
    assert isinstance(first_child, ir.OpStmts), "First child of SeqStmts should be OpStmts"

    # Within the first OpStmts, alloc statements should come first
    found_non_alloc = False
    for stmt in first_child.stmts:
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            if stmt.value.op.name == "block.alloc":
                assert not found_non_alloc, "Alloc statements should precede non-alloc statements"
                continue
        found_non_alloc = True


def test_allocate_memory_addr_raw_pointer_uniqueness():
    """Test that only unique MemRef objects get alloc operations.

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

    optimized_program = _prepare_and_run_allocate_memory_addr(Before)
    optimized_func = list(optimized_program.functions.values())[0]

    alloc_count = count_alloc_operations(optimized_func)
    assert alloc_count == 3, f"Expected 3 unique MemRef objects, but got {alloc_count} allocs"

    memref_addrs = get_memref_addresses_from_tiles(optimized_func)
    expected_addrs = {"tile_a": 0, "tile_b": 16384, "tile_c": 32768}
    for var_name, expected_addr in expected_addrs.items():
        assert var_name in memref_addrs, f"Variable {var_name} not found in MemRef addresses"
        actual_addr = memref_addrs[var_name]
        assert actual_addr == expected_addr, f"{var_name}: expected addr={expected_addr}, got {actual_addr}"
        assert actual_addr % 32 == 0, f"Address {actual_addr} for {var_name} should be 32-byte aligned"


def test_allocated_memory_addr_verifier_passes_after_add_alloc():
    """Test that AllocatedMemoryAddr verifier passes on a correctly allocated program.

    After running init_mem_ref + add_alloc, all non-DDR memrefs should have
    valid (non-negative) addresses.
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

    program = passes.init_mem_ref()(Before)
    program = passes.allocate_memory_addr()(program)

    func = list(program.functions.values())[0]
    memref_addrs = get_memref_addresses_from_tiles(func)

    for var_name, addr in memref_addrs.items():
        assert addr >= 0, (
            f"MemRef address for '{var_name}' should be non-negative after AllocateMemoryAddr, got {addr}"
        )


def test_memrefs_before_allocate_have_unallocated_addr():
    """Test that before AllocateMemoryAddr, MemRef addresses are -1 (unallocated).

    After init_mem_ref only (without allocate_memory_addr), all non-DDR
    MemRefs have addr=-1. The AllocatedMemoryAddr verifier would flag these.
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

    program = passes.init_mem_ref()(Before)
    func = list(program.functions.values())[0]

    memref_addrs = get_memref_addresses_from_tiles(func)
    assert len(memref_addrs) > 0, "Should have MemRef addresses after init_mem_ref"
    for var_name, addr in memref_addrs.items():
        assert addr == -1, (
            f"MemRef address for '{var_name}' should be -1 before AllocateMemoryAddr, got {addr}"
        )


def test_allocated_memory_addr_verifier_via_pipeline():
    """Test that the AllocatedMemoryAddr property is verified through the PassPipeline.

    Uses VerificationInstrument in AFTER mode to confirm that add_alloc
    correctly produces the AllocatedMemoryAddr property.
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

    pipeline = passes.PassPipeline()
    pipeline.add_pass(passes.init_mem_ref())
    pipeline.add_pass(passes.allocate_memory_addr())

    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
        result = pipeline.run(Before)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

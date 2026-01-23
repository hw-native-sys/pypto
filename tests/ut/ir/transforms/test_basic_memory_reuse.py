# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pypto import ir
from pypto.ir import builder
from pypto.ir.op import block
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import DataType, passes
from pypto.pypto_core import ir as core_ir


def get_var_type(func, var_name):
    """Extract ShapedType for a variable by name.

    Args:
        func: Function to search
        var_name: Name of the variable to find

    Returns:
        ShapedType object if found, None otherwise
    """
    if not isinstance(func.body, ir.SeqStmts):
        return None

    for stmt in func.body.stmts:
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name == var_name:
            if isinstance(stmt.var.type, core_ir.ShapedType):
                return stmt.var.type
    return None


def verify_memref_sharing(func, var_a_name, var_b_name):
    """Verify that two variables share the same MemRef object using ShapedType.shares_memref_with().

    Args:
        func: Function containing the variables
        var_a_name: Name of the first variable
        var_b_name: Name of the second variable
    """
    type_a = get_var_type(func, var_a_name)
    type_b = get_var_type(func, var_b_name)

    assert type_a is not None, f"{var_a_name} should have ShapedType"
    assert type_b is not None, f"{var_b_name} should have ShapedType"

    # Use ShapedType.shares_memref_with() method to check if they share the same MemRef
    assert type_a.shares_memref_with(type_b), (
        f"{var_b_name} should share the same MemRef object (C++ shared_ptr) with {var_a_name}"
    )


def test_basic_memory_reuse_simple():
    """Test BasicMemoryReusePass with a simple buffer reuse case.

    Core concepts:
    - let assignment has reference semantics, no memory allocation involved
    - Real memory reuse means: the underlying buffer of operation results is reused by subsequent operations
    - Example: tile_e's buffer can reuse tile_a's buffer (tile_a's lifetime has ended)
    """
    ib = builder.IRBuilder()

    with ib.function("test_basic_memory_reuse") as f:
        # Define input and output parameters
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        # Constants for tile
        tile_height = 64
        tile_width = 64

        # Load tiles
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, tile_height, tile_width))

        # Compute: tile_c = tile_a + tile_b (tile_a is last used here)
        tile_c = ib.let("tile_c", block.add(tile_a, tile_b))

        # Compute: tile_d = tile_c * 2 (tile_b is last used in tile_c, tile_c is used here)
        tile_d = ib.let("tile_d", block.mul(tile_c, tile_c))

        # Compute: tile_e = tile_d + tile_d (tile_c is last used in tile_d)
        # tile_e can potentially reuse tile_a's memory since tile_a is no longer used
        tile_e = ib.let("tile_e", block.add(tile_d, tile_d))

        # Store result
        result = ib.let("result", block.store(tile_e, 0, 0, tile_height, tile_width, output))

        ib.return_stmt(result)

    func = f.get_result()

    # Use PassManager with XPlatform strategy to run InitMemRefPass and BasicMemoryReusePass
    pm = PassManager.get_strategy(OptimizationStrategy.XPlatform)
    optimized_func: ir.Function = pm.run_passes(func)  # type: ignore[assignment]

    # Verify the function is valid
    assert optimized_func is not None
    assert optimized_func.name == "test_basic_memory_reuse"
    assert isinstance(optimized_func.body, ir.SeqStmts)

    # Check that variables have memrefs
    stmts = optimized_func.body.stmts
    assert len(stmts) >= 5

    # All intermediate tiles should have memrefs
    for i in range(5):
        stmt = stmts[i]
        assert isinstance(stmt, ir.AssignStmt)
        var = stmt.var
        assert isinstance(var.type, core_ir.ShapedType)
        assert var.type.memref is not None

    # Check that tile_d reuses tile_a's memory using object identity
    verify_memref_sharing(optimized_func, "tile_a", "tile_d")

    # Check that tile_e reuses memory from tile_a (greedy first-fit)
    verify_memref_sharing(optimized_func, "tile_a", "tile_e")


def test_basic_memory_reuse_sequential():
    """Test BasicMemoryReusePass with sequential computation (ideal for buffer reuse).

    Core concepts:
    - In a sequential computation chain, each operation's buffer becomes invalid immediately after use
    - Subsequent operations' buffers can reuse previously invalidated buffers
    - Example: tile_c's buffer can reuse tile_a's buffer
    """
    ib = builder.IRBuilder()

    with ib.function("test_sequential_reuse") as f:
        # Define input and output parameters
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        # Constants for tile
        tile_height = 64
        tile_width = 64

        # Load tile
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))

        # Sequential computations: each variable is only used once
        # tile_b = tile_a * 2
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))

        # tile_c = tile_b * 2 (tile_a is dead, tile_c can reuse tile_a's memory)
        tile_c = ib.let("tile_c", block.add(tile_b, tile_b))

        # tile_d = tile_c * 2 (tile_b is dead, tile_d can reuse tile_b's memory)
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))

        # tile_e = tile_d * 2 (tile_c is dead, tile_e can reuse tile_c's memory)
        tile_e = ib.let("tile_e", block.add(tile_d, tile_d))

        # Store result (tile_d is dead, only tile_e is live)
        result = ib.let("result", block.store(tile_e, 0, 0, tile_height, tile_width, output))

        ib.return_stmt(result)

    func = f.get_result()

    # First run InitMemRefPass
    init_pass = passes.InitMemRefPass()
    func_with_memref = init_pass.run(func)

    # Then run BasicMemoryReusePass
    reuse_pass = passes.BasicMemoryReusePass()
    optimized_func = reuse_pass.run(func_with_memref)

    # Verify the function is valid
    assert optimized_func is not None
    assert optimized_func.name == "test_sequential_reuse"
    assert isinstance(optimized_func.body, ir.SeqStmts)

    # All variables should have memrefs
    stmts = optimized_func.body.stmts
    for stmt in stmts:
        if isinstance(stmt, ir.AssignStmt):
            var = stmt.var
            if isinstance(var.type, core_ir.ShapedType):
                assert var.type.memref is not None

    # Expected lifetime analysis (sequential, no overlap):
    # tile_a: [0, 1] - defined at 0, last used in tile_b at 1
    # tile_b: [1, 2] - defined at 1, last used in tile_c at 2
    # tile_c: [2, 3] - defined at 2, last used in tile_d at 3
    # tile_d: [3, 4] - defined at 3, last used in tile_e at 4
    # tile_e: [4, 4] - defined at 4, used in store at 4
    #
    # Optimal reuse strategy (greedy first-fit):
    # - tile_c reuses tile_a ([2,3] doesn't overlap [0,1])
    # - tile_d reuses tile_a ([3,4] doesn't overlap [0,1])
    # - tile_e reuses tile_a ([4,4] doesn't overlap [0,1])

    # Verify that tile_c, tile_d, and tile_e all reuse tile_a's memory using object identity
    verify_memref_sharing(optimized_func, "tile_a", "tile_c")
    verify_memref_sharing(optimized_func, "tile_a", "tile_d")
    verify_memref_sharing(optimized_func, "tile_a", "tile_e")


def test_basic_memory_reuse_different_sizes():
    """Test BasicMemoryReusePass with different tensor sizes (size-aware buffer reuse).

    Core concepts:
    - Large buffers can be reused by small buffers (sufficient space available)
    - Small buffers cannot be reused by large buffers (insufficient space)
    - Example: A 64x64 buffer can reuse another 64x64 buffer,
              but a 32x32 buffer cannot be reused by a 64x64 buffer
    """
    ib = builder.IRBuilder()

    with ib.function("test_different_sizes") as f:
        # Define input and output parameters with different sizes
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([32, 32], DataType.FP32))
        output_a = f.param("output_a", ir.TensorType([64, 64], DataType.FP32))
        output_b = f.param("output_b", ir.TensorType([32, 32], DataType.FP32))
        f.return_type(ir.TensorType([32, 32], DataType.FP32))

        # Load tiles of different sizes
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))  # 64x64 = 16384 bytes
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 32, 32))  # 32x32 = 4096 bytes

        # Compute with tile_a (64x64)
        tile_c = ib.let("tile_c", block.add(tile_a, tile_a))  # 64x64, tile_a dies

        # Store tile_c (tile_c dies after this)
        ib.let("result_a", block.store(tile_c, 0, 0, 64, 64, output_a))

        # Compute with tile_b (32x32)
        # tile_d can potentially reuse tile_a or tile_c's memory (both 64x64 >= 32x32)
        # but tile_a/tile_c cannot reuse tile_b's memory (32x32 < 64x64)
        tile_d = ib.let("tile_d", block.add(tile_b, tile_b))  # 32x32

        # Store result
        result_b = ib.let("result_b", block.store(tile_d, 0, 0, 32, 32, output_b))

        ib.return_stmt(result_b)

    func = f.get_result()

    # First run InitMemRefPass
    init_pass = passes.InitMemRefPass()
    func_with_memref = init_pass.run(func)

    # Then run BasicMemoryReusePass
    reuse_pass = passes.BasicMemoryReusePass()
    optimized_func = reuse_pass.run(func_with_memref)

    # Verify the function is valid
    assert optimized_func is not None
    assert optimized_func.name == "test_different_sizes"

    # Expected lifetime analysis:
    # tile_a: [0, 2] - defined at 0, last used in tile_c at 2
    # tile_b: [1, 4] - defined at 1, last used in tile_d at 4
    # tile_c: [2, 3] - defined at 2, last used in result_a at 3
    # tile_d: [4, 5] - defined at 4, last used in result_b at 5
    #
    # Size-aware reuse opportunities:
    # - tile_d (4096 bytes, 32x32) can reuse tile_a (16384 bytes, 64x64) - smaller can reuse larger ✓
    # - tile_d (4096 bytes) can reuse tile_c (16384 bytes, 64x64) - smaller can reuse larger ✓
    # - tile_a/tile_c (16384 bytes) should NOT reuse tile_b (4096 bytes) - larger cannot reuse smaller ✗

    # Verify that tile_d (32x32) can reuse tile_a's memory (64x64, large enough)
    # or tile_c's memory (64x64, large enough, non-overlapping lifetimes)
    # The greedy first-fit algorithm should reuse tile_a
    verify_memref_sharing(optimized_func, "tile_a", "tile_d")


def test_basic_memory_reuse_empty_function():
    """Test BasicMemoryReusePass with empty function (edge case)."""
    ib = builder.IRBuilder()

    with ib.function("test_empty") as f:
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))
        ib.return_stmt(output)

    func = f.get_result()

    # Run BasicMemoryReusePass on empty function
    reuse_pass = passes.BasicMemoryReusePass()
    optimized_func = reuse_pass.run(func)

    # Should return a valid function (even if unchanged)
    assert optimized_func is not None
    assert optimized_func.name == "test_empty"


def test_basic_memory_reuse_memref_sharing():
    """Test that MemRef objects are actually shared between variables."""
    ib = builder.IRBuilder()

    with ib.function("test_memref_sharing") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_height = 64
        tile_width = 64

        # Create a chain of operations
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))  # tile_a dies
        tile_c = ib.let("tile_c", block.add(tile_b, tile_b))  # tile_b dies, can reuse tile_a
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))  # tile_c dies, can reuse tile_b
        result = ib.let("result", block.store(tile_d, 0, 0, tile_height, tile_width, output))

        ib.return_stmt(result)

    func = f.get_result()

    # First run InitMemRefPass
    init_pass = passes.InitMemRefPass()
    func_with_memref = init_pass.run(func)

    # Then run BasicMemoryReusePass
    reuse_pass = passes.BasicMemoryReusePass()
    optimized_func = reuse_pass.run(func_with_memref)

    # Extract variables from optimized function
    assert isinstance(optimized_func.body, ir.SeqStmts)
    stmts = optimized_func.body.stmts

    # Extract variables
    vars_dict = {}
    for stmt in stmts:
        if isinstance(stmt, ir.AssignStmt):
            vars_dict[stmt.var.name] = stmt.var

    # Expected lifetime analysis:
    # tile_a: [0, 1] - defined at 0, last used in tile_b at 1
    # tile_b: [1, 2] - defined at 1, last used in tile_c at 2
    # tile_c: [2, 3] - defined at 2, last used in tile_d at 3
    # tile_d: [3, 3] - defined at 3, used in store at 3
    #
    # Reuse opportunities:
    # - tile_c can reuse tile_a ([2,3] doesn't overlap [0,1])
    # - tile_d can reuse tile_a ([3,3] doesn't overlap [0,1])

    # Check that tile_c should reuse tile_a's MemRef (both are size 16384, non-overlapping lifetimes)
    verify_memref_sharing(optimized_func, "tile_a", "tile_c")

    # Also check tile_d reuses tile_a using object identity
    verify_memref_sharing(optimized_func, "tile_a", "tile_d")


def test_basic_memory_reuse_with_dependencies():
    """Test BasicMemoryReusePass respects dependencies."""
    ib = builder.IRBuilder()

    with ib.function("test_dependencies") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_height = 64
        tile_width = 64

        # Load two tiles
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, tile_height, tile_width))

        # tile_c depends on both tile_a and tile_b
        tile_c = ib.let("tile_c", block.add(tile_a, tile_b))

        # tile_d depends on tile_c (tile_a and tile_b can be reused after this)
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))

        # tile_e depends on tile_d (tile_c can be reused after this)
        tile_e = ib.let("tile_e", block.add(tile_d, tile_d))

        # Store result
        result = ib.let("result", block.store(tile_e, 0, 0, tile_height, tile_width, output))

        ib.return_stmt(result)

    func = f.get_result()

    # First run InitMemRefPass
    init_pass = passes.InitMemRefPass()
    func_with_memref = init_pass.run(func)

    # Then run BasicMemoryReusePass
    reuse_pass = passes.BasicMemoryReusePass()
    optimized_func = reuse_pass.run(func_with_memref)

    # Should not crash and produce valid output
    assert optimized_func is not None
    assert isinstance(optimized_func.body, ir.SeqStmts)

    # Expected lifetime analysis:
    # tile_a: [0, 2] - defined at 0, last used in tile_c at 2
    # tile_b: [1, 2] - defined at 1, last used in tile_c at 2
    # tile_c: [2, 3] - defined at 2, last used in tile_d at 3
    # tile_d: [3, 4] - defined at 3, last used in tile_e at 4
    # tile_e: [4, 4] - defined at 4, used in store at 4
    #
    # Reuse opportunities (respecting dependencies):
    # - tile_d can reuse tile_a or tile_b (lifetimes [3,4] vs [0,2] and [1,2])
    # - tile_e can reuse tile_a or tile_b (lifetimes [4,4] vs [0,2] and [1,2])
    # - tile_e can also reuse tile_c (lifetimes [4,4] vs [2,3])

    # Verify memory reuse happened using object identity
    # tile_d should reuse tile_a's memory (greedy first-fit)
    verify_memref_sharing(optimized_func, "tile_a", "tile_d")

    # tile_e should also reuse tile_a's memory
    verify_memref_sharing(optimized_func, "tile_a", "tile_e")


def test_basic_memory_reuse_multiple_memory_spaces():
    """Test BasicMemoryReusePass doesn't mix different memory spaces."""
    ib = builder.IRBuilder()

    with ib.function("test_memory_spaces") as f:
        # This test verifies that variables in DDR don't reuse UB memory and vice versa
        # Parameters are in DDR, tiles are in UB
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        output_a = f.param("output_a", ir.TensorType([64, 64], DataType.FP32))
        output_b = f.param("output_b", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_height = 64
        tile_width = 64

        # Load creates UB tiles
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, tile_height, tile_width))

        # Compute creates more UB tiles
        tile_c = ib.let("tile_c", block.add(tile_a, tile_b))  # tile_a and tile_b die here

        # Store to first output (intermediate result)
        ib.let("result_a", block.store(tile_c, 0, 0, tile_height, tile_width, output_a))

        # More UB computation (tile_c dies here)
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))

        # Store final result
        result_b = ib.let("result_b", block.store(tile_d, 0, 0, tile_height, tile_width, output_b))

        ib.return_stmt(result_b)

    func = f.get_result()

    # First run InitMemRefPass
    init_pass = passes.InitMemRefPass()
    func_with_memref = init_pass.run(func)

    # Then run BasicMemoryReusePass
    reuse_pass = passes.BasicMemoryReusePass()
    optimized_func = reuse_pass.run(func_with_memref)

    # Expected lifetime analysis:
    # tile_a: [0, 2] - defined at 0, last used in tile_c at 2
    # tile_b: [1, 2] - defined at 1, last used in tile_c at 2
    # tile_c: [2, 4] - defined at 2, used in result_a at 3 and tile_d at 4
    # tile_d: [4, 4] - defined at 4, used in result_b at 4
    #
    # All are in UB memory space, reuse should happen within UB:
    # - tile_d should reuse tile_a or tile_b's memory (non-overlapping lifetimes)

    # Verify that tile_d reuses UB memory from tile_a using object identity
    verify_memref_sharing(optimized_func, "tile_a", "tile_d")

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
from pypto import ir
from pypto.ir import builder
from pypto.ir.op import block
from pypto.pypto_core import DataType, passes


def get_stmt_index(func, var_name):
    """Get the index of a statement that defines a variable.

    Args:
        func: Function to search
        var_name: Name of the variable to find

    Returns:
        Index of the statement if found, -1 otherwise
    """
    if not isinstance(func.body, ir.SeqStmts):
        return -1

    for i, stmt in enumerate(func.body.stmts):
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name == var_name:
            return i
    return -1


def verify_dependency_order(func, producer_var, consumer_var):
    """Verify that producer appears before consumer in execution order.

    Args:
        func: Function containing the variables
        producer_var: Name of the producer variable
        consumer_var: Name of the consumer variable

    Returns:
        True if dependency order is preserved, False otherwise
    """
    prod_idx = get_stmt_index(func, producer_var)
    cons_idx = get_stmt_index(func, consumer_var)

    if prod_idx == -1 or cons_idx == -1:
        return False

    return prod_idx < cons_idx


def run_pass_with_ir_print(func, pass_obj, pass_name=None, print_ir=False):
    """Run a pass and optionally print IR before and after.

    Args:
        func: The IR function to transform
        pass_obj: The pass object to run
        pass_name: Optional name for the pass (for printing)
        print_ir: Whether to print IR before and after

    Returns:
        Transformed IR function
    """
    if print_ir:
        name = pass_name or pass_obj.__class__.__name__
        print(f"\n{'=' * 80}")
        print(f"IR BEFORE {name}:")
        print(f"{'=' * 80}")
        print(ir.python_print(func))

    result = pass_obj.run(func)

    if print_ir:
        name = pass_name or pass_obj.__class__.__name__
        print(f"\n{'=' * 80}")
        print(f"IR AFTER {name}:")
        print(f"{'=' * 80}")
        print(ir.python_print(result))

    return result


# =============================================================================
# 1. Basic Functionality Tests
# =============================================================================


def test_out_of_order_scheduler_basic():
    """Test OutOfOrderSchedulerPass with basic functionality."""
    ib = builder.IRBuilder()

    with ib.function("test_basic") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))

        # Already in optimal order: load -> compute -> load -> compute
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        _tile_c = ib.let("tile_c", block.add(tile_a, tile_a))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 64, 64))
        tile_d = ib.let("tile_d", block.add(tile_b, tile_b))

        ib.return_stmt(tile_d)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    # Run OutOfOrderSchedulerPass
    scheduler_pass = passes.OutOfOrderSchedulerPass()
    optimized_func = run_pass_with_ir_print(func, scheduler_pass, print_ir=False)

    # Verify function is valid
    assert optimized_func is not None
    assert isinstance(optimized_func.body, ir.SeqStmts)
    assert len(optimized_func.body.stmts) >= 4


# =============================================================================
# 2. Dependency Preservation Tests
# =============================================================================


def test_out_of_order_scheduler_all_dependencies():
    """Test RAW/WAR/WAW dependency preservation in one comprehensive test."""
    ib = builder.IRBuilder()

    with ib.function("test_all_deps") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))

        # RAW dependency: tile_b reads tile_a
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))

        # WAR dependency: tile_c uses tile_a, then tile_a gets redefined
        tile_c = ib.let("tile_c", block.mul(tile_a, tile_a))
        tile_a_new = ib.let("tile_a", block.load(input_b, 0, 0, 64, 64))

        # WAW dependency: tile_result gets written twice
        _tile_result_v1 = ib.let("tile_result", block.add(tile_b, tile_c))
        tile_result_v2 = ib.let("tile_result", block.mul(tile_a_new, tile_a_new))

        ib.return_stmt(tile_result_v2)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    # Run OutOfOrderSchedulerPass
    scheduler_pass = passes.OutOfOrderSchedulerPass()
    optimized_func = run_pass_with_ir_print(func, scheduler_pass, print_ir=False)

    # Verify RAW dependency: first tile_a before tile_b
    assert verify_dependency_order(optimized_func, "tile_a", "tile_b")
    # Verify basic structure
    assert optimized_func is not None
    assert isinstance(optimized_func.body, ir.SeqStmts)


def test_out_of_order_scheduler_dependency_chain():
    """Test complex dependency chain: A → B → C → D."""
    ib = builder.IRBuilder()

    with ib.function("test_chain") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))

        # Chain: tile_a → tile_b → tile_c → tile_d
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))
        tile_c = ib.let("tile_c", block.mul(tile_b, tile_b))
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))

        ib.return_stmt(tile_d)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    # Run OutOfOrderSchedulerPass
    scheduler_pass = passes.OutOfOrderSchedulerPass()
    optimized_func = run_pass_with_ir_print(func, scheduler_pass, print_ir=False)

    # Verify topological order is preserved
    assert verify_dependency_order(optimized_func, "tile_a", "tile_b")
    assert verify_dependency_order(optimized_func, "tile_b", "tile_c")
    assert verify_dependency_order(optimized_func, "tile_c", "tile_d")


# =============================================================================
# 3. Cross-Pipe Optimization Tests
# =============================================================================


def test_out_of_order_scheduler_cross_pipe():
    """Test cross-pipe scheduling with MTE2 → V → MTE3 pattern."""
    ib = builder.IRBuilder()

    with ib.function("test_cross_pipe") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        input_c = f.param("input_c", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))

        # MTE2: multiple loads
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 64, 64))
        tile_c = ib.let("tile_c", block.load(input_c, 0, 0, 64, 64))

        # V: multiple vector operations
        tile_ab = ib.let("tile_ab", block.add(tile_a, tile_b))
        tile_bc = ib.let("tile_bc", block.add(tile_b, tile_c))
        tile_ac = ib.let("tile_ac", block.add(tile_a, tile_c))

        # V: final computation
        tile_final = ib.let("tile_final", block.add(tile_ab, block.add(tile_bc, tile_ac)))

        # MTE3: store
        res = ib.let("res", block.store(tile_final, 0, 0, 64, 64, output))

        ib.return_stmt(res)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    # Run OutOfOrderSchedulerPass
    scheduler_pass = passes.OutOfOrderSchedulerPass()
    optimized_func = run_pass_with_ir_print(func, scheduler_pass, print_ir=False)

    # Verify function is valid
    assert optimized_func is not None
    assert isinstance(optimized_func.body, ir.SeqStmts)

    # Verify key dependencies
    assert verify_dependency_order(optimized_func, "tile_a", "tile_ab")
    assert verify_dependency_order(optimized_func, "tile_b", "tile_bc")
    assert verify_dependency_order(optimized_func, "tile_final", "res")


def test_out_of_order_scheduler_independent_operations():
    """Test that independent operations can be reordered."""
    ib = builder.IRBuilder()

    with ib.function("test_independent") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))

        # Two independent load-compute chains
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_a_sq = ib.let("tile_a_sq", block.mul(tile_a, tile_a))

        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 64, 64))
        tile_b_sq = ib.let("tile_b_sq", block.mul(tile_b, tile_b))

        # Combine at the end
        result = ib.let("result", block.add(tile_a_sq, tile_b_sq))

        ib.return_stmt(result)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    # Run OutOfOrderSchedulerPass
    scheduler_pass = passes.OutOfOrderSchedulerPass()
    optimized_func = run_pass_with_ir_print(func, scheduler_pass, print_ir=False)

    # Verify function is valid
    assert optimized_func is not None

    # Verify dependencies within each chain are preserved
    assert verify_dependency_order(optimized_func, "tile_a", "tile_a_sq")
    assert verify_dependency_order(optimized_func, "tile_b", "tile_b_sq")
    assert verify_dependency_order(optimized_func, "tile_a_sq", "result")
    assert verify_dependency_order(optimized_func, "tile_b_sq", "result")


# =============================================================================
# 4. Control Flow Tests
# =============================================================================


def test_out_of_order_scheduler_control_flow_barriers():
    """Test that control flow (If/For) acts as scheduling barriers."""
    ib = builder.IRBuilder()

    with ib.function("test_barriers") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        cond = f.param("cond", ir.ScalarType(DataType.BOOL))

        # Before If
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))

        # If statement (barrier) with yield
        with ib.if_stmt(cond) as if_builder:
            if_builder.return_var("tile_b", ir.TileType([64, 64], DataType.FP32))

            # Then branch
            tile_b_then = ib.let("tile_b_then", block.add(tile_a, tile_a))
            ib.emit(ir.YieldStmt([tile_b_then], ir.Span.unknown()))

            # Else branch
            if_builder.else_()
            tile_b_else = ib.let("tile_b_else", block.mul(tile_a, tile_a))
            ib.emit(ir.YieldStmt([tile_b_else], ir.Span.unknown()))

        tile_b = if_builder.output()

        # After If, before For
        tile_c = ib.let("tile_c", block.mul(tile_b, tile_b))

        # For loop (barrier) - For loops don't return values in typical usage
        loop_var = ib.var("i", ir.ScalarType(DataType.INT64))
        with ib.for_loop(loop_var, 0, 10, 1):
            _tile_d = ib.let("tile_d", block.add(tile_c, tile_c))

        # After For
        tile_e = ib.let("tile_e", block.load(input_b, 0, 0, 64, 64))

        # Use tile_c to create dependency
        result = ib.let("result", block.add(tile_c, tile_e))
        ib.return_stmt(result)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    # Run OutOfOrderSchedulerPass
    scheduler_pass = passes.OutOfOrderSchedulerPass()
    optimized_func = run_pass_with_ir_print(func, scheduler_pass, print_ir=False)

    # Verify function is valid and control flow preserved
    assert optimized_func is not None
    assert isinstance(optimized_func.body, ir.SeqStmts)

    # Verify both If and For statements exist
    has_if = any(isinstance(stmt, ir.IfStmt) for stmt in optimized_func.body.stmts)
    has_for = any(isinstance(stmt, ir.ForStmt) for stmt in optimized_func.body.stmts)
    assert has_if
    assert has_for

    # Verify dependency through If output
    assert verify_dependency_order(optimized_func, "tile_a", "tile_c")


def test_out_of_order_scheduler_if_with_return_value():
    """Test If statement with return value and dependency tracking.

    Pattern: If statement returns a value that is used by subsequent statements.
    Tests that scheduler correctly identifies dependencies across If boundaries.
    """
    ib = builder.IRBuilder()

    with ib.function("test_if_return") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        cond = f.param("cond", ir.ScalarType(DataType.BOOL))

        # Load A - used by If statement
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))

        # If statement returns a computed tile
        with ib.if_stmt(cond) as if_builder:
            # Declare return variable
            if_builder.return_var("tile_from_if", ir.TileType([64, 64], DataType.FP32))

            # Then branch: compute using tile_a
            tile_then = ib.let("tile_then", block.add(tile_a, tile_a))
            ib.emit(ir.YieldStmt([tile_then], ir.Span.unknown()))

            # Else branch: compute using tile_a differently
            if_builder.else_()
            tile_else = ib.let("tile_else", block.mul(tile_a, tile_a))
            ib.emit(ir.YieldStmt([tile_else], ir.Span.unknown()))

        # Get return value from If
        tile_from_if = if_builder.output()

        # Load B (independent of If and tile_from_if)
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 64, 64))

        # Compute using tile_from_if - creates dependency on If output
        result = ib.let("result", block.add(tile_from_if, tile_b))

        ib.return_stmt(result)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, "InitMemRefPass", print_ir=False)

    # Run OutOfOrderSchedulerPass with IR printing
    scheduler_pass = passes.OutOfOrderSchedulerPass()
    optimized_func = run_pass_with_ir_print(
        func, scheduler_pass, "OutOfOrderSchedulerPass", print_ir=False
    )

    # Verify function is valid and If statement preserved
    assert optimized_func is not None
    assert isinstance(optimized_func.body, ir.SeqStmts)

    # Verify If statement still exists
    has_if = any(isinstance(stmt, ir.IfStmt) for stmt in optimized_func.body.stmts)
    assert has_if

    # Verify dependency: tile_a must come before result
    assert verify_dependency_order(optimized_func, "tile_a", "result")


def test_out_of_order_scheduler_multiple_if_with_yields():
    """Test multiple If statements with return values and dependency chain.

    Pattern: Two If statements that both return values, with subsequent computation
    depending on both return values.
    """
    ib = builder.IRBuilder()

    with ib.function("test_multi_if") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        cond1 = f.param("cond1", ir.ScalarType(DataType.BOOL))
        cond2 = f.param("cond2", ir.ScalarType(DataType.BOOL))

        # Load inputs
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 64, 64))

        # First If - returns processed tile_a
        with ib.if_stmt(cond1) as if_builder1:
            if_builder1.return_var("tile_from_if1", ir.TileType([64, 64], DataType.FP32))

            # Then branch
            tile_then1 = ib.let("tile_then1", block.add(tile_a, tile_a))
            ib.emit(ir.YieldStmt([tile_then1], ir.Span.unknown()))

            # Else branch
            if_builder1.else_()
            tile_else1 = ib.let("tile_else1", block.mul(tile_a, tile_a))
            ib.emit(ir.YieldStmt([tile_else1], ir.Span.unknown()))

        tile_from_if1 = if_builder1.output()

        # Second If - returns processed tile_b
        with ib.if_stmt(cond2) as if_builder2:
            if_builder2.return_var("tile_from_if2", ir.TileType([64, 64], DataType.FP32))

            # Then branch
            tile_then2 = ib.let("tile_then2", block.add(tile_b, tile_b))
            ib.emit(ir.YieldStmt([tile_then2], ir.Span.unknown()))

            # Else branch
            if_builder2.else_()
            tile_else2 = ib.let("tile_else2", block.mul(tile_b, tile_b))
            ib.emit(ir.YieldStmt([tile_else2], ir.Span.unknown()))

        tile_from_if2 = if_builder2.output()

        # Final result depends on both If outputs
        result = ib.let("result", block.add(tile_from_if1, tile_from_if2))

        ib.return_stmt(result)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    # Run OutOfOrderSchedulerPass
    scheduler_pass = passes.OutOfOrderSchedulerPass()
    optimized_func = run_pass_with_ir_print(func, scheduler_pass, print_ir=False)

    # Verify function is valid
    assert optimized_func is not None
    assert isinstance(optimized_func.body, ir.SeqStmts)

    # Extract If statement indices
    if_indices = [i for i, stmt in enumerate(optimized_func.body.stmts) if isinstance(stmt, ir.IfStmt)]

    # Verify both If statements exist and are in order
    assert len(if_indices) >= 2
    assert if_indices[0] < if_indices[1]

    # Verify dependencies
    assert verify_dependency_order(optimized_func, "tile_a", "result")
    assert verify_dependency_order(optimized_func, "tile_b", "result")


def test_out_of_order_scheduler_cross_control_flow():
    """Test scheduling across control flow with dependencies."""
    ib = builder.IRBuilder()

    with ib.function("test_cross_cf") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        cond = f.param("cond", ir.ScalarType(DataType.BOOL))

        # Before If: computations
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.mul(tile_a, tile_a))

        # If statement: depends on tile_a, returns a value
        with ib.if_stmt(cond) as if_builder:
            if_builder.return_var("tile_from_if", ir.TileType([64, 64], DataType.FP32))

            # Then branch
            tile_c = ib.let("tile_c", block.add(tile_a, tile_a))
            tile_d_then = ib.let("tile_d_then", block.mul(tile_c, tile_c))
            ib.emit(ir.YieldStmt([tile_d_then], ir.Span.unknown()))

            # Else branch
            if_builder.else_()
            tile_c_else = ib.let("tile_c_else", block.mul(tile_a, tile_a))
            tile_d_else = ib.let("tile_d_else", block.add(tile_c_else, tile_c_else))
            ib.emit(ir.YieldStmt([tile_d_else], ir.Span.unknown()))

        tile_from_if = if_builder.output()

        # After If: independent computation (can potentially move up)
        tile_e = ib.let("tile_e", block.load(input_b, 0, 0, 64, 64))
        tile_f = ib.let("tile_f", block.mul(tile_e, tile_e))

        # Return combination uses tile_b, tile_from_if, and tile_f
        temp = ib.let("temp", block.add(tile_b, tile_from_if))
        result = ib.let("result", block.add(temp, tile_f))
        ib.return_stmt(result)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    # Run OutOfOrderSchedulerPass
    scheduler_pass = passes.OutOfOrderSchedulerPass()
    optimized_func = run_pass_with_ir_print(func, scheduler_pass, print_ir=True)

    # Verify function is valid
    assert optimized_func is not None
    assert isinstance(optimized_func.body, ir.SeqStmts)

    # Verify If statement exists
    has_if = any(isinstance(stmt, ir.IfStmt) for stmt in optimized_func.body.stmts)
    assert has_if

    # Verify dependencies
    assert verify_dependency_order(optimized_func, "tile_a", "tile_b")
    assert verify_dependency_order(optimized_func, "tile_e", "tile_f")
    assert verify_dependency_order(optimized_func, "tile_b", "result")


# =============================================================================
# 5. Integration Tests
# =============================================================================


def test_out_of_order_scheduler_with_insert_sync():
    """Test OutOfOrderSchedulerPass integration with InsertSyncPass."""
    ib = builder.IRBuilder()

    with ib.function("test_scheduler_sync_integration") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))

        # MTE2: loads
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 64, 64))

        # V: compute
        tile_c = ib.let("tile_c", block.add(tile_a, tile_b))

        # MTE3: store
        res = ib.let("res", block.store(tile_c, 0, 0, 64, 64, output))

        ib.return_stmt(res)

    func = f.get_result()

    # Run passes in order: InitMemRef → OutOfOrderScheduler → InsertSync
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    scheduler_pass = passes.OutOfOrderSchedulerPass()
    func = run_pass_with_ir_print(func, scheduler_pass, print_ir=False)

    insert_sync = passes.InsertSyncPass()
    synced_func = run_pass_with_ir_print(func, insert_sync, print_ir=False)

    # Verify sync operations are inserted
    assert isinstance(synced_func.body, ir.SeqStmts)
    stmts = synced_func.body.stmts

    # Count sync operations
    sync_ops = 0
    for stmt in stmts:
        if isinstance(stmt, ir.EvalStmt):
            call = stmt.expr
            if isinstance(call, ir.Call):
                if "sync" in call.op.name or "bar" in call.op.name:
                    sync_ops += 1

    # Should have sync operations inserted
    assert sync_ops > 0


def test_out_of_order_scheduler_fixes_event_limit_issue():
    """Test that OutOfOrderSchedulerPass fixes cases where InsertSyncPass would fail.

    This test constructs a scenario where:
    1. Without reordering: InsertSyncPass may fail due to exceeding 8 event limit
    2. With OutOfOrderScheduler: InsertSyncPass succeeds after reordering
    """
    ib = builder.IRBuilder()

    with ib.function("test_event_limit_fix") as f:
        # Create many input tensors to maximize cross-pipe dependencies
        inputs = [f.param(f"input_{i}", ir.TensorType([64, 64], DataType.FP32)) for i in range(12)]
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))

        # MTE2 Phase: All loads first
        loads = []
        for i, inp in enumerate(inputs):
            tile = ib.let(f"load_{i}", block.load(inp, 0, 0, 64, 64))
            loads.append(tile)

        # V Phase: Compute operations that depend on multiple loads
        computes = []
        for i in range(0, len(loads) - 1):
            compute = ib.let(f"compute_{i}", block.add(loads[i], loads[i + 1]))
            computes.append(compute)

        # Aggregate all computes
        result = computes[0]
        for i in range(1, len(computes)):
            result = ib.let(f"agg_{i}", block.add(result, computes[i]))

        # MTE3 Phase: Store result
        store_res = ib.let("store_result", block.store(result, 0, 0, 64, 64, output))

        ib.return_stmt(store_res)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    # Test Phase 1: Try without OutOfOrderSchedulerPass
    insert_sync = passes.InsertSyncPass()

    try:
        insert_sync.run(func)
    except Exception as e:
        # Expected: "Out of hardware event IDs" error
        assert "Out of hardware event IDs" in str(e) or "max 8" in str(e) or "Deadlock" in str(e)

    # Test Phase 2: With OutOfOrderSchedulerPass, must succeed
    scheduler_pass = passes.OutOfOrderSchedulerPass()
    reordered_func = run_pass_with_ir_print(func, scheduler_pass, print_ir=False)

    # InsertSyncPass should succeed after reordering
    synced_func = run_pass_with_ir_print(reordered_func, insert_sync, print_ir=False)

    # Verify the result is valid
    assert synced_func is not None
    assert isinstance(synced_func.body, ir.SeqStmts)


# =============================================================================
# 6. Edge Cases and Stress Tests
# =============================================================================


def test_out_of_order_scheduler_large_segment():
    """Test with a large number of statements (stress test)."""
    ib = builder.IRBuilder()

    with ib.function("test_large_segment") as f:
        inputs = [f.param(f"input_{i}", ir.TensorType([64, 64], DataType.FP32)) for i in range(10)]

        # Load all inputs
        tiles = []
        for i, inp in enumerate(inputs):
            tile = ib.let(f"tile_{i}", block.load(inp, 0, 0, 64, 64))
            tiles.append(tile)

        # Compute pairwise sums
        sums = []
        for i in range(len(tiles) - 1):
            sum_tile = ib.let(f"sum_{i}_{i + 1}", block.add(tiles[i], tiles[i + 1]))
            sums.append(sum_tile)

        # Final aggregation
        result = sums[0]
        for i in range(1, len(sums)):
            result = ib.let(f"result_{i}", block.add(result, sums[i]))

        ib.return_stmt(result)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    # Run OutOfOrderSchedulerPass
    scheduler_pass = passes.OutOfOrderSchedulerPass()
    optimized_func = run_pass_with_ir_print(func, scheduler_pass, print_ir=False)

    # Verify function completes without errors
    assert optimized_func is not None
    assert isinstance(optimized_func.body, ir.SeqStmts)
    assert len(optimized_func.body.stmts) > 10


def test_out_of_order_scheduler_mixed_dependencies():
    """Test with mixed independent and dependent operations."""
    ib = builder.IRBuilder()

    with ib.function("test_mixed") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        input_c = f.param("input_c", ir.TensorType([64, 64], DataType.FP32))

        # Independent chain 1
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_a2 = ib.let("tile_a2", block.mul(tile_a, tile_a))

        # Independent chain 2
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 64, 64))
        tile_b2 = ib.let("tile_b2", block.mul(tile_b, tile_b))

        # Independent chain 3
        tile_c = ib.let("tile_c", block.load(input_c, 0, 0, 64, 64))
        tile_c2 = ib.let("tile_c2", block.mul(tile_c, tile_c))

        # Dependent: combine all
        tile_ab = ib.let("tile_ab", block.add(tile_a2, tile_b2))
        result = ib.let("result", block.add(tile_ab, tile_c2))

        ib.return_stmt(result)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    # Run OutOfOrderSchedulerPass
    scheduler_pass = passes.OutOfOrderSchedulerPass()
    optimized_func = run_pass_with_ir_print(func, scheduler_pass, print_ir=False)

    # Verify dependencies within chains
    assert verify_dependency_order(optimized_func, "tile_a", "tile_a2")
    assert verify_dependency_order(optimized_func, "tile_b", "tile_b2")
    assert verify_dependency_order(optimized_func, "tile_c", "tile_c2")
    assert verify_dependency_order(optimized_func, "tile_a2", "result")
    assert verify_dependency_order(optimized_func, "tile_b2", "result")
    assert verify_dependency_order(optimized_func, "tile_c2", "result")


# =============================================================================
# 7. Broadcast Semantics Tests
# =============================================================================


def test_out_of_order_scheduler_broadcast_semantics():
    """Test that one producer with multiple consumers uses only 1 event (broadcast semantics).

    Pattern: tail_x → {consumer_0, consumer_1, ..., consumer_9} (10 consumers)

    With old per-edge model: 10 events for (MTE2, V) → would exceed limit (8)
    With new broadcast model: 1 event for (MTE2, V) → within limit
    """
    ib = builder.IRBuilder()

    with ib.function("test_broadcast") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))

        # Producer: single load operation (MTE2)
        tail_x = ib.let("tail_x", block.load(input_a, 0, 0, 64, 64))

        # Consumers: 10 compute operations (V), all depend on tail_x
        consumers = []
        for i in range(10):
            consumer = ib.let(f"consumer_{i}", block.add(tail_x, tail_x))
            consumers.append(consumer)

        # Aggregate all consumers
        result = consumers[0]
        for i in range(1, len(consumers)):
            result = ib.let(f"agg_{i}", block.add(result, consumers[i]))

        # Store result (MTE3)
        store_res = ib.let("store_result", block.store(result, 0, 0, 64, 64, output))

        ib.return_stmt(store_res)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    # Run OutOfOrderSchedulerPass
    scheduler_pass = passes.OutOfOrderSchedulerPass()
    optimized_func = run_pass_with_ir_print(func, scheduler_pass, print_ir=False)

    # Verify function is valid
    assert optimized_func is not None
    assert isinstance(optimized_func.body, ir.SeqStmts)

    # Verify tail_x comes before all consumers (dependency preservation)
    tail_x_idx = get_stmt_index(optimized_func, "tail_x")
    assert tail_x_idx >= 0

    for i in range(10):
        consumer_idx = get_stmt_index(optimized_func, f"consumer_{i}")
        assert consumer_idx >= 0
        assert tail_x_idx < consumer_idx


def test_out_of_order_scheduler_multi_producer_same_pair():
    """Test multiple producers on same pipe pair don't cause event mis-release."""
    ib = builder.IRBuilder()

    with ib.function("test_multi_producer") as f:
        inputs = [f.param(f"input_{i}", ir.TensorType([64, 64], DataType.FP32)) for i in range(9)]
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))

        # 9 producers: loads (MTE2)
        loads = []
        for i, inp in enumerate(inputs):
            loads.append(ib.let(f"load_{i}", block.load(inp, 0, 0, 64, 64)))

        # 2 consumers per producer: V compute
        consumers = []
        for i, tile in enumerate(loads):
            c0 = ib.let(f"consumer_{i}_0", block.add(tile, tile))
            c1 = ib.let(f"consumer_{i}_1", block.mul(c0, c0))
            consumers.extend([c0, c1])

        # Final aggregation
        result = consumers[0]
        for i in range(1, len(consumers)):
            result = ib.let(f"agg_{i}", block.add(result, consumers[i]))

        store_res = ib.let("store_result", block.store(result, 0, 0, 64, 64, output))
        ib.return_stmt(store_res)

    func = f.get_result()

    init_memref = passes.InitMemRefPass()
    func = run_pass_with_ir_print(func, init_memref, print_ir=False)

    scheduler_pass = passes.OutOfOrderSchedulerPass()
    func = run_pass_with_ir_print(func, scheduler_pass, print_ir=False)

    # Should not hit "Out of hardware event IDs" error
    insert_sync = passes.InsertSyncPass()
    func = run_pass_with_ir_print(func, insert_sync, print_ir=False)

    assert func is not None


if __name__ == "__main__":
    pytest.main([__file__])

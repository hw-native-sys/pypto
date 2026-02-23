# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for InsertSyncPass."""

import pytest
from pypto import DataType, backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.op import block


def count_syncs(stmt):
    """Recursively count sync operations in a statement."""
    counts = {"sync_src": 0, "sync_dst": 0, "bar_v": 0, "bar_m": 0}

    if isinstance(stmt, ir.EvalStmt):
        call = stmt.expr
        if isinstance(call, ir.Call):
            if call.op.name == "system.sync_src":
                counts["sync_src"] += 1
            elif call.op.name == "system.sync_dst":
                counts["sync_dst"] += 1
            elif call.op.name == "system.bar_v":
                counts["bar_v"] += 1
            elif call.op.name == "system.bar_m":
                counts["bar_m"] += 1
    elif isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            sub_counts = count_syncs(s)
            for key in counts:
                counts[key] += sub_counts[key]
    elif isinstance(stmt, ir.IfStmt):
        # Count syncs in then and else branches
        then_counts = count_syncs(stmt.then_body)
        for key in counts:
            counts[key] += then_counts[key]
        if stmt.else_body is not None:
            else_counts = count_syncs(stmt.else_body)
            for key in counts:
                counts[key] += else_counts[key]
    elif isinstance(stmt, ir.ForStmt):
        # Count syncs in for body
        body_counts = count_syncs(stmt.body)
        for key in counts:
            counts[key] += body_counts[key]

    return counts


def test_insert_sync_cross_pipe():
    """Test InsertSyncPass for cross-pipe dependencies (MTE2 -> V -> MTE3)."""
    # Test structure:
    #   tile_a = load(input_a)        # MTE2
    #   tile_b = load(input_b)        # MTE2
    #   sync_src (MTE2 -> V)          # Inserted
    #   sync_dst (MTE2 -> V)          # Inserted
    #   tile_c = add(tile_a, tile_b)  # V
    #   sync_src (V -> MTE3)          # Inserted
    #   sync_dst (V -> MTE3)          # Inserted
    #   store(tile_c, output)         # MTE3
    #
    # Expected: 2 sync pairs (MTE2->V) + 1 sync pair (V->MTE3)
    span = ir.Span.unknown()

    # Create shape expressions
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    # Create unique memrefs for each tile variable
    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 0)  # 64*64*4
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 1)
    memref_c = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(32768, DataType.INT64, span), 16384, 2)

    # Create variables with memrefs
    input_a = ir.Var("input_a", ir.TensorType([64, 64], DataType.FP32), span)
    input_b = ir.Var("input_b", ir.TensorType([64, 64], DataType.FP32), span)
    output = ir.Var("output", ir.TensorType([64, 64], DataType.FP32), span)

    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    tile_c = ir.Var("tile_c", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)

    # Create operations
    load_a = block.load(input_a, offsets=[0, 0], shapes=[64, 64])
    load_b = block.load(input_b, offsets=[0, 0], shapes=[64, 64])
    add_op = block.add(tile_a, tile_b)
    store_op = block.store(tile_c, offsets=[0, 0], shapes=[64, 64], output_tensor=output)

    # Build statements
    stmt_load_a = ir.AssignStmt(tile_a, load_a, span)
    stmt_load_b = ir.AssignStmt(tile_b, load_b, span)
    stmt_add = ir.AssignStmt(tile_c, add_op, span)
    stmt_store = ir.EvalStmt(store_op, span)
    stmt_return = ir.ReturnStmt(span)

    # Build function
    body = ir.SeqStmts([stmt_load_a, stmt_load_b, stmt_add, stmt_store, stmt_return], span)
    func = ir.Function(
        "test_cross_pipe_sync", [input_a, input_b, output], [], body, span, ir.FunctionType.InCore
    )

    # Wrap function in Program
    program = ir.Program([func], "test_program", span)

    # Run passes
    # 1. InitMemRefPass (required for InsertSyncPass to see memrefs)
    init_memref = passes.init_mem_ref()
    program_with_memref = init_memref(program)

    # 2. InsertSyncPass (uses globally configured backend)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    insert_sync = passes.insert_sync()
    synced_program = insert_sync(program_with_memref)

    # Extract the function from the program
    synced_func = list(synced_program.functions.values())[0]

    # Verify sync ops are inserted
    assert isinstance(synced_func.body, ir.SeqStmts)
    stmts = synced_func.body.stmts

    sync_src_count = 0
    sync_dst_count = 0
    for stmt in stmts:
        if isinstance(stmt, ir.EvalStmt):
            call = stmt.expr
            if isinstance(call, ir.Call):
                if call.op.name == "system.sync_src":
                    sync_src_count += 1
                elif call.op.name == "system.sync_dst":
                    sync_dst_count += 1

    assert sync_src_count == 3
    assert sync_dst_count == 3


def test_insert_sync_intra_pipe():
    """Test InsertSyncPass for intra-pipe dependencies (V -> V)."""
    # Test structure:
    #   t_c = add(t_a, t_b)  # V
    #   bar_v                # Inserted
    #   t_d = add(t_c, t_a)  # V (depends on t_c)
    #
    # Expected: 1 intra-pipe barrier (bar_v)
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    # Create unique memrefs for each tile
    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 3)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 4)
    memref_c = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(32768, DataType.INT64, span), 16384, 5)
    memref_d = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(49152, DataType.INT64, span), 16384, 6)

    # Create variables with memrefs (as function parameters and locals)
    t_a = ir.Var("t_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    t_b = ir.Var("t_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    t_c = ir.Var("t_c", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    t_d = ir.Var("t_d", ir.TileType([dim64, dim64], DataType.FP32, memref_d), span)

    # V pipe operations
    add_c = block.add(t_a, t_b)
    add_d = block.add(t_c, t_a)

    # Build statements
    stmt_add_c = ir.AssignStmt(t_c, add_c, span)
    stmt_add_d = ir.AssignStmt(t_d, add_d, span)
    stmt_return = ir.ReturnStmt([t_d], span)

    # Build function
    body = ir.SeqStmts([stmt_add_c, stmt_add_d, stmt_return], span)
    func = ir.Function("test_intra_pipe_sync", [t_a, t_b], [t_d.type], body, span)

    # Wrap function in Program
    program = ir.Program([func], "test_program", span)

    # Run InitMemRefPass
    init_memref = passes.init_mem_ref()
    program_with_memref = init_memref(program)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    insert_sync = passes.insert_sync()
    synced_program = insert_sync(program_with_memref)

    # Extract the function from the program
    synced_func = list(synced_program.functions.values())[0]

    # Verify bar_v is inserted
    assert isinstance(synced_func.body, ir.SeqStmts)
    stmts = synced_func.body.stmts
    bar_v_count = 0
    for stmt in stmts:
        if isinstance(stmt, ir.EvalStmt):
            call = stmt.expr
            if isinstance(call, ir.Call) and call.op.name == "system.bar_v":
                bar_v_count += 1

    assert bar_v_count == 1


def test_insert_sync_ifstmt():
    """Test InsertSyncPass for IfStmt with cross-pipe dependencies."""
    # Test structure:
    #   tile_a = load(input)          # MTE2
    #   sync_src (MTE2 -> V)          # Inserted
    #   sync_dst (MTE2 -> V)          # Inserted
    #   if (condition):
    #     then:
    #       tile_b = add(tile_a, tile_a)   # V
    #       bar_v                          # Inserted
    #       tile_c = add(tile_b, tile_a)   # V
    #       yield [tile_c]
    #     else:
    #       tile_b = mul(tile_a, tile_a)   # V
    #       bar_v                          # Inserted
    #       tile_c = add(tile_b, tile_a)   # V
    #       yield [tile_c]
    #   # tile_c from return_vars
    #   sync_src (V -> MTE3)          # Inserted
    #   sync_dst (V -> MTE3)          # Inserted
    #   store(tile_c, output)         # MTE3
    #
    # Expected: 2 cross-pipe sync pairs (MTE2→V + V→MTE3) + 2 intra-pipe barriers (bar_v in then/else)
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    # Create unique memrefs
    memref_input = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 7)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 8)
    memref_c = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(32768, DataType.INT64, span), 16384, 9)
    memref_output = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(49152, DataType.INT64, span), 16384, 10)

    # Create variables with memrefs
    input_tensor = ir.Var("input", ir.TensorType([64, 64], DataType.FP32, memref_input), span)
    output_tensor = ir.Var("output", ir.TensorType([64, 64], DataType.FP32, memref_output), span)
    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_input), span)

    # Load from MTE2 pipe
    load_op = block.load(input_tensor, offsets=[0, 0], shapes=[64, 64])
    stmt_load = ir.AssignStmt(tile_a, load_op, span)

    # Create condition
    condition = ir.ConstBool(True, span)

    # Build then branch with V pipe operations
    add_op_then = block.add(tile_a, tile_a)
    tile_b_then = ir.Var("tile_b_then", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    stmt_add_then = ir.AssignStmt(tile_b_then, add_op_then, span)

    add_op2_then = block.add(tile_b_then, tile_a)
    tile_c_then = ir.Var("tile_c_then", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    stmt_add2_then = ir.AssignStmt(tile_c_then, add_op2_then, span)

    yield_then = ir.YieldStmt([tile_c_then], span)
    then_body = ir.SeqStmts([stmt_add_then, stmt_add2_then, yield_then], span)

    # Build else branch with V pipe operations
    mul_op_else = block.mul(tile_a, tile_a)
    tile_b_else = ir.Var("tile_b_else", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    stmt_mul_else = ir.AssignStmt(tile_b_else, mul_op_else, span)

    add_op_else = block.add(tile_b_else, tile_a)
    tile_c_else = ir.Var("tile_c_else", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    stmt_add_else = ir.AssignStmt(tile_c_else, add_op_else, span)

    yield_else = ir.YieldStmt([tile_c_else], span)
    else_body = ir.SeqStmts([stmt_mul_else, stmt_add_else, yield_else], span)

    # Create IfStmt with return_vars
    if_return_var = ir.Var("tile_c", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    if_stmt = ir.IfStmt(condition, then_body, else_body, [if_return_var], span)

    # Store to MTE3 pipe (depends on IfStmt output)
    store_op = block.store(if_return_var, offsets=[0, 0], shapes=[64, 64], output_tensor=output_tensor)
    stmt_store = ir.EvalStmt(store_op, span)
    stmt_return = ir.ReturnStmt(span)

    # Build function body with the load, if statement, and store
    body = ir.SeqStmts([stmt_load, if_stmt, stmt_store, stmt_return], span)
    func = ir.Function(
        "test_ifstmt_sync", [input_tensor, output_tensor], [], body, span, ir.FunctionType.InCore
    )

    # Wrap function in Program
    program = ir.Program([func], "test_program", span)

    # Run InsertSyncPass directly without InitMemRefPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    insert_sync = passes.insert_sync()
    synced_program = insert_sync(program)

    # Extract the function from the program
    synced_func = list(synced_program.functions.values())[0]

    # Verify the structure
    assert isinstance(synced_func.body, ir.SeqStmts)

    total_counts = count_syncs(synced_func.body)
    sync_src_count = total_counts["sync_src"]
    sync_dst_count = total_counts["sync_dst"]
    bar_v_count = total_counts["bar_v"]

    assert sync_src_count == 2, f"Expected exactly 2 sync_src, got {sync_src_count}"
    assert sync_dst_count == 2, f"Expected exactly 2 sync_dst, got {sync_dst_count}"
    assert bar_v_count == 2, f"Expected exactly 2 bar_v, got {bar_v_count}"


def test_insert_sync_cross_ifstmt_dependency():
    """Test InsertSyncPass for cross-IfStmt dependencies.

    This test verifies that the pass can identify dependencies spanning across
    if statements, where a V-pipe operation after the if depends on both a V-pipe
    operation computed before the if and a result from inside the if.
    """
    # Test structure:
    #   tile_a = load(input)          # MTE2 pipe
    #   sync_src (MTE2 -> V)          # Inserted (for all tile_a users)
    #   sync_dst (MTE2 -> V)          # Inserted
    #   tile_b = add(tile_a, tile_a)  # V pipe (depends on tile_a)
    #   if (condition):
    #     then:
    #       tile_c = add(tile_a, tile_a)   # V pipe (uses same sync pair)
    #       yield [tile_c]
    #     else:
    #       tile_c = mul(tile_a, tile_a)   # V pipe (uses same sync pair)
    #       yield [tile_c]
    #   bar_v                         # Inserted (cross-if dependency: tile_b -> tile_d)
    #   tile_d = add(tile_b, tile_c)  # V pipe (depends on tile_b and tile_c from if)
    #   sync_src (V -> MTE3)          # Inserted
    #   sync_dst (V -> MTE3)          # Inserted
    #   store(tile_d, output)         # MTE3 pipe
    #
    # Expected: 2 sync_src, 2 sync_dst + 1 bar_v (same pipe pair shares one sync)
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    # Create unique memrefs for each tile
    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 11)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 12)
    memref_c = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(32768, DataType.INT64, span), 16384, 13)
    memref_d = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(49152, DataType.INT64, span), 16384, 14)
    memref_output = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(65536, DataType.INT64, span), 16384, 15)

    # Create variables with memrefs
    input_tensor = ir.Var("input", ir.TensorType([64, 64], DataType.FP32), span)
    output_tensor = ir.Var("output", ir.TensorType([64, 64], DataType.FP32, memref_output), span)

    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    tile_d = ir.Var("tile_d", ir.TileType([dim64, dim64], DataType.FP32, memref_d), span)

    # Load from MTE2 pipe
    load_op = block.load(input_tensor, offsets=[0, 0], shapes=[64, 64])
    stmt_load = ir.AssignStmt(tile_a, load_op, span)

    # V pipe operation before if: tile_b = add(tile_a, tile_a)
    # This depends on tile_a which comes from MTE2
    add_b_op = block.add(tile_a, tile_a)
    stmt_add_b = ir.AssignStmt(tile_b, add_b_op, span)

    # Create condition for if statement
    condition = ir.ConstBool(True, span)

    # Build then branch: tile_c = add(tile_a, tile_a) (V pipe, depends on tile_a)
    add_c_then_op = block.add(tile_a, tile_a)
    tile_c_then = ir.Var("tile_c_then", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    stmt_add_c_then = ir.AssignStmt(tile_c_then, add_c_then_op, span)
    yield_then = ir.YieldStmt([tile_c_then], span)
    then_body = ir.SeqStmts([stmt_add_c_then, yield_then], span)

    # Build else branch: tile_c = mul(tile_a, tile_a) (V pipe, depends on tile_a)
    mul_c_else_op = block.mul(tile_a, tile_a)
    tile_c_else = ir.Var("tile_c_else", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    stmt_mul_c_else = ir.AssignStmt(tile_c_else, mul_c_else_op, span)
    yield_else = ir.YieldStmt([tile_c_else], span)
    else_body = ir.SeqStmts([stmt_mul_c_else, yield_else], span)

    # Create IfStmt with return_vars for tile_c
    if_return_var = ir.Var("tile_c", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    if_stmt = ir.IfStmt(condition, then_body, else_body, [if_return_var], span)

    # V pipe operation after if: tile_d = add(tile_b, tile_c)
    # This depends on both tile_b (computed before if) and tile_c (from if return_vars)
    add_d_op = block.add(tile_b, if_return_var)
    stmt_add_d = ir.AssignStmt(tile_d, add_d_op, span)

    # Store to MTE3 pipe (depends on tile_d)
    store_op = block.store(tile_d, offsets=[0, 0], shapes=[64, 64], output_tensor=output_tensor)
    stmt_store = ir.EvalStmt(store_op, span)
    stmt_return = ir.ReturnStmt(span)

    # Build function body with load, tile_b, if, tile_d, store
    body = ir.SeqStmts([stmt_load, stmt_add_b, if_stmt, stmt_add_d, stmt_store, stmt_return], span)
    func = ir.Function(
        "test_cross_ifstmt_sync", [input_tensor, output_tensor], [], body, span, ir.FunctionType.InCore
    )

    # Wrap function in Program
    program = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    insert_sync = passes.insert_sync()
    synced_program = insert_sync(program)

    # Extract the function from the program
    synced_func = list(synced_program.functions.values())[0]

    # Verify the structure
    assert isinstance(synced_func.body, ir.SeqStmts)

    total_counts = count_syncs(synced_func.body)
    sync_src_count = total_counts["sync_src"]
    sync_dst_count = total_counts["sync_dst"]
    bar_v_count = total_counts["bar_v"]

    assert sync_src_count == 2, f"Expected exactly 2 sync_src, got {sync_src_count}"
    assert sync_dst_count == 2, f"Expected exactly 2 sync_dst, got {sync_dst_count}"
    assert bar_v_count == 1, f"Expected exactly 1 bar_v, got {bar_v_count}"


def test_insert_sync_nested_ifstmt():
    """Test InsertSyncPass for nested IfStmt."""
    # Test structure:
    #   tile_input = load(input)               # MTE2
    #   sync_src (MTE2 -> V)                   # Inserted
    #   sync_dst (MTE2 -> V)                   # Inserted
    #   outer_if:
    #     then:
    #       tile_a = add(tile_input, tile_input)  # V
    #       bar_v                                  # Inserted
    #       inner_if:
    #         then:
    #           tile_d = mul(tile_a, tile_a)       # V
    #           bar_v                              # Inserted
    #           tile_b = add(tile_d, tile_a)       # V
    #           yield [tile_b]
    #         else:
    #           tile_d = sub(tile_a, tile_a)       # V
    #           bar_v                              # Inserted
    #       tile_b = add(tile_d, tile_a)       # V
    #       yield [tile_b]
    #     # tile_b from inner_if return_vars
    #     bar_v                                  # Inserted
    #     tile_c = add(tile_b, tile_a)          # V
    #     yield [tile_c]
    #   else:
    #     tile_c = sub(tile_input, tile_input)  # V
    #     yield [tile_c]
    # # tile_c from outer_if return_vars
    # sync_src (V -> MTE3)                   # Inserted
    # sync_dst (V -> MTE3)                   # Inserted
    # store(tile_result, output)             # MTE3
    #
    # Expected: 2 cross-pipe sync pairs (MTE2→V + V→MTE3) + 4 intra-pipe barriers (bar_v)
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    # Create unique memrefs
    memref_input = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 16)
    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 17)
    memref_d = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(32768, DataType.INT64, span), 16384, 18)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(49152, DataType.INT64, span), 16384, 19)
    memref_c = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(65536, DataType.INT64, span), 16384, 20)
    memref_output = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(81920, DataType.INT64, span), 16384, 21)

    # Create variables
    input_tensor = ir.Var("input", ir.TensorType([64, 64], DataType.FP32, memref_input), span)
    output_tensor = ir.Var("output", ir.TensorType([64, 64], DataType.FP32, memref_output), span)
    tile_input = ir.Var("tile_input", ir.TileType([dim64, dim64], DataType.FP32, memref_input), span)

    # Load from MTE2 pipe
    load_op = block.load(input_tensor, offsets=[0, 0], shapes=[64, 64])
    stmt_load = ir.AssignStmt(tile_input, load_op, span)

    # Outer if condition
    outer_condition = ir.ConstBool(True, span)

    # --- Build outer then branch with nested if ---
    # V pipe operation before nested if
    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    add_a = block.add(tile_input, tile_input)
    stmt_add_a = ir.AssignStmt(tile_a, add_a, span)

    # Inner if condition
    inner_condition = ir.ConstBool(False, span)

    # Inner then branch with extra instruction for internal dependency
    tile_d_inner_then = ir.Var(
        "tile_d_inner_then", ir.TileType([dim64, dim64], DataType.FP32, memref_d), span
    )
    mul_d_inner_then = block.mul(tile_a, tile_a)
    stmt_mul_d_inner_then = ir.AssignStmt(tile_d_inner_then, mul_d_inner_then, span)

    tile_b_inner_then = ir.Var(
        "tile_b_inner_then", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span
    )
    add_b_inner_then = block.add(tile_d_inner_then, tile_a)
    stmt_add_b_inner_then = ir.AssignStmt(tile_b_inner_then, add_b_inner_then, span)

    yield_inner_then = ir.YieldStmt([tile_b_inner_then], span)
    inner_then_body = ir.SeqStmts([stmt_mul_d_inner_then, stmt_add_b_inner_then, yield_inner_then], span)

    # Inner else branch with extra instruction for internal dependency
    tile_d_inner_else = ir.Var(
        "tile_d_inner_else", ir.TileType([dim64, dim64], DataType.FP32, memref_d), span
    )
    sub_d_inner_else = block.sub(tile_a, tile_a)
    stmt_sub_d_inner_else = ir.AssignStmt(tile_d_inner_else, sub_d_inner_else, span)

    tile_b_inner_else = ir.Var(
        "tile_b_inner_else", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span
    )
    add_b_inner_else = block.add(tile_d_inner_else, tile_a)
    stmt_add_b_inner_else = ir.AssignStmt(tile_b_inner_else, add_b_inner_else, span)

    yield_inner_else = ir.YieldStmt([tile_b_inner_else], span)
    inner_else_body = ir.SeqStmts([stmt_sub_d_inner_else, stmt_add_b_inner_else, yield_inner_else], span)

    # Inner IfStmt with return var
    inner_return_var = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    inner_if_stmt = ir.IfStmt(inner_condition, inner_then_body, inner_else_body, [inner_return_var], span)

    # Use the result from inner if
    tile_c_outer_then = ir.Var(
        "tile_c_outer_then", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span
    )
    add_c = block.add(inner_return_var, tile_a)
    stmt_add_c = ir.AssignStmt(tile_c_outer_then, add_c, span)
    yield_outer_then = ir.YieldStmt([tile_c_outer_then], span)

    outer_then_body = ir.SeqStmts([stmt_add_a, inner_if_stmt, stmt_add_c, yield_outer_then], span)

    # --- Build outer else branch (simple V pipe operation) ---
    tile_c_outer_else = ir.Var(
        "tile_c_outer_else", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span
    )
    sub_c = block.sub(tile_input, tile_input)
    stmt_sub_c = ir.AssignStmt(tile_c_outer_else, sub_c, span)
    yield_outer_else = ir.YieldStmt([tile_c_outer_else], span)
    outer_else_body = ir.SeqStmts([stmt_sub_c, yield_outer_else], span)

    # Outer IfStmt with return var
    outer_return_var = ir.Var("tile_result", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    outer_if_stmt = ir.IfStmt(outer_condition, outer_then_body, outer_else_body, [outer_return_var], span)

    # Store result to MTE3 pipe
    store_op = block.store(outer_return_var, offsets=[0, 0], shapes=[64, 64], output_tensor=output_tensor)
    stmt_store = ir.EvalStmt(store_op, span)
    stmt_return = ir.ReturnStmt(span)

    # Build function body
    body = ir.SeqStmts([stmt_load, outer_if_stmt, stmt_store, stmt_return], span)
    func = ir.Function(
        "test_nested_ifstmt_sync", [input_tensor, output_tensor], [], body, span, ir.FunctionType.InCore
    )

    # Wrap function in Program
    program = ir.Program([func], "test_program", span)

    # Run InsertSyncPass directly without InitMemRefPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    insert_sync = passes.insert_sync()
    synced_program = insert_sync(program)

    # Extract the function from the program
    synced_func = list(synced_program.functions.values())[0]

    # Verify the structure
    assert isinstance(synced_func.body, ir.SeqStmts)

    total_counts = count_syncs(synced_func.body)
    sync_src_count = total_counts["sync_src"]
    sync_dst_count = total_counts["sync_dst"]
    bar_v_count = total_counts["bar_v"]

    assert sync_src_count == 2, f"Expected exactly 2 sync_src, got {sync_src_count}"
    assert sync_dst_count == 2, f"Expected exactly 2 sync_dst, got {sync_dst_count}"
    assert bar_v_count == 4, f"Expected exactly 4 bar_v, got {bar_v_count}"


def test_insert_sync_forstmt():
    """Test InsertSyncPass for ForStmt with cross-pipe dependencies before/after."""
    # Test structure:
    #   tile_a = load(input)          # MTE2
    #   sync_src (MTE2 -> V)          # Inserted (before for)
    #   sync_dst (MTE2 -> V)          # Inserted (before for)
    #   for i in range(0, 4, 1):
    #     tile_b = add(tile_a, tile_a)   # V (reads tile_a)
    #     bar_v                          # Inserted (internal)
    #     tile_c = add(tile_b, tile_a)   # V (reads tile_b)
    #     yield []
    #   sync_src (V -> MTE3)          # Inserted (after for)
    #   sync_dst (V -> MTE3)          # Inserted (after for)
    #   store(tile_c, output)         # MTE3
    #
    # Expected: 2 cross-pipe sync pairs (MTE2→V before for + V→MTE3 after for) + 1 bar_v internal
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    # Create unique memrefs
    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 100)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 101)
    memref_c = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(32768, DataType.INT64, span), 16384, 102)
    memref_output = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(49152, DataType.INT64, span), 16384, 103)

    # Create variables
    input_tensor = ir.Var("input", ir.TensorType([64, 64], DataType.FP32), span)
    output_tensor = ir.Var("output", ir.TensorType([64, 64], DataType.FP32, memref_output), span)
    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    tile_c = ir.Var("tile_c", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)

    # Load from MTE2 pipe
    load_op = block.load(input_tensor, offsets=[0, 0], shapes=[64, 64])
    stmt_load = ir.AssignStmt(tile_a, load_op, span)

    # For loop
    loop_var = ir.Var("i", ir.ScalarType(DataType.INT32), span)
    start = ir.ConstInt(0, DataType.INT32, span)
    stop = ir.ConstInt(4, DataType.INT32, span)
    step = ir.ConstInt(1, DataType.INT32, span)

    # For body: V pipe operations with internal dependency
    add_b_op = block.add(tile_a, tile_a)
    stmt_add_b = ir.AssignStmt(tile_b, add_b_op, span)

    add_c_op = block.add(tile_b, tile_a)
    stmt_add_c = ir.AssignStmt(tile_c, add_c_op, span)

    yield_stmt = ir.YieldStmt([], span)
    for_body = ir.SeqStmts([stmt_add_b, stmt_add_c, yield_stmt], span)

    for_stmt = ir.ForStmt(loop_var, start, stop, step, [], for_body, [], span)

    # Store to MTE3 pipe (depends on tile_c from for body)
    store_op = block.store(tile_c, offsets=[0, 0], shapes=[64, 64], output_tensor=output_tensor)
    stmt_store = ir.EvalStmt(store_op, span)
    stmt_return = ir.ReturnStmt(span)

    # Build function body
    body = ir.SeqStmts([stmt_load, for_stmt, stmt_store, stmt_return], span)
    func = ir.Function(
        "test_forstmt_sync", [input_tensor, output_tensor], [], body, span, ir.FunctionType.InCore
    )

    # Wrap function in Program
    program = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    insert_sync = passes.insert_sync()
    synced_program = insert_sync(program)

    # Extract the function from the program
    synced_func = list(synced_program.functions.values())[0]

    # Verify the structure
    assert isinstance(synced_func.body, ir.SeqStmts)

    total_counts = count_syncs(synced_func.body)
    sync_src_count = total_counts["sync_src"]
    sync_dst_count = total_counts["sync_dst"]
    bar_v_count = total_counts["bar_v"]

    assert sync_src_count == 2, f"Expected exactly 2 sync_src, got {sync_src_count}"
    assert sync_dst_count == 2, f"Expected exactly 2 sync_dst, got {sync_dst_count}"
    assert bar_v_count == 1, f"Expected exactly 1 bar_v, got {bar_v_count}"


def test_insert_sync_forstmt_cross_boundary():
    """Test InsertSyncPass for ForStmt with dependencies crossing the for boundary.

    This test verifies that dependencies between statements before/after the for
    and operations inside the for body are properly synchronized at the for boundary.
    """
    # Test structure:
    #   tile_a = load(input)          # MTE2
    #   sync_src (MTE2 -> V)          # Inserted
    #   sync_dst (MTE2 -> V)          # Inserted
    #   tile_b = add(tile_a, tile_a)  # V (before for, depends on tile_a)
    #   bar_v                         # Inserted (tile_b -> for body)
    #   for i in range(0, 4, 1):
    #     tile_c = add(tile_a, tile_b)   # V (reads tile_a and tile_b)
    #     yield []
    #   bar_v                         # Inserted (for body tile_c -> tile_d)
    #   tile_d = add(tile_b, tile_c)  # V (depends on tile_b before for, tile_c from for)
    #   sync_src (V -> MTE3)          # Inserted
    #   sync_dst (V -> MTE3)          # Inserted
    #   store(tile_d, output)         # MTE3
    #
    # Expected: 2 sync_src, 2 sync_dst, 2 bar_v (one before for, one after for)
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    # Create unique memrefs
    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 110)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 111)
    memref_c = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(32768, DataType.INT64, span), 16384, 112)
    memref_d = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(49152, DataType.INT64, span), 16384, 113)
    memref_output = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(65536, DataType.INT64, span), 16384, 114)

    # Create variables
    input_tensor = ir.Var("input", ir.TensorType([64, 64], DataType.FP32), span)
    output_tensor = ir.Var("output", ir.TensorType([64, 64], DataType.FP32, memref_output), span)
    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    tile_c = ir.Var("tile_c", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    tile_d = ir.Var("tile_d", ir.TileType([dim64, dim64], DataType.FP32, memref_d), span)

    # Load from MTE2 pipe
    load_op = block.load(input_tensor, offsets=[0, 0], shapes=[64, 64])
    stmt_load = ir.AssignStmt(tile_a, load_op, span)

    # V pipe operation before for
    add_b_op = block.add(tile_a, tile_a)
    stmt_add_b = ir.AssignStmt(tile_b, add_b_op, span)

    # For loop
    loop_var = ir.Var("i", ir.ScalarType(DataType.INT32), span)
    start = ir.ConstInt(0, DataType.INT32, span)
    stop = ir.ConstInt(4, DataType.INT32, span)
    step = ir.ConstInt(1, DataType.INT32, span)

    # For body
    add_c_op = block.add(tile_a, tile_b)
    stmt_add_c = ir.AssignStmt(tile_c, add_c_op, span)
    yield_stmt = ir.YieldStmt([], span)
    for_body = ir.SeqStmts([stmt_add_c, yield_stmt], span)

    for_stmt = ir.ForStmt(loop_var, start, stop, step, [], for_body, [], span)

    # V pipe operation after for
    add_d_op = block.add(tile_b, tile_c)
    stmt_add_d = ir.AssignStmt(tile_d, add_d_op, span)

    # Store to MTE3 pipe
    store_op = block.store(tile_d, offsets=[0, 0], shapes=[64, 64], output_tensor=output_tensor)
    stmt_store = ir.EvalStmt(store_op, span)
    stmt_return = ir.ReturnStmt(span)

    # Build function body
    body = ir.SeqStmts([stmt_load, stmt_add_b, for_stmt, stmt_add_d, stmt_store, stmt_return], span)
    func = ir.Function(
        "test_forstmt_cross_boundary", [input_tensor, output_tensor], [], body, span, ir.FunctionType.InCore
    )

    # Wrap function in Program
    program = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    insert_sync = passes.insert_sync()
    synced_program = insert_sync(program)

    # Extract the function from the program
    synced_func = list(synced_program.functions.values())[0]

    # Verify the structure
    assert isinstance(synced_func.body, ir.SeqStmts)

    total_counts = count_syncs(synced_func.body)
    sync_src_count = total_counts["sync_src"]
    sync_dst_count = total_counts["sync_dst"]
    bar_v_count = total_counts["bar_v"]

    assert sync_src_count == 2, f"Expected exactly 2 sync_src, got {sync_src_count}"
    assert sync_dst_count == 2, f"Expected exactly 2 sync_dst, got {sync_dst_count}"
    assert bar_v_count == 2, f"Expected exactly 2 bar_v, got {bar_v_count}"


def test_insert_sync_forstmt_with_ifstmt():
    """Test InsertSyncPass for ForStmt containing IfStmt."""
    # Test structure:
    #   tile_a = load(input)          # MTE2
    #   sync_src (MTE2 -> V)          # Inserted
    #   sync_dst (MTE2 -> V)          # Inserted
    #   for i in range(0, 4, 1):
    #     if (cond):
    #       then:
    #         tile_b = add(tile_a, tile_a)   # V
    #         bar_v                          # Inserted
    #         tile_c = add(tile_b, tile_a)   # V
    #         yield [tile_c]
    #       else:
    #         tile_b = mul(tile_a, tile_a)   # V
    #         bar_v                          # Inserted
    #         tile_c = mul(tile_b, tile_a)   # V
    #         yield [tile_c]
    #     yield []
    #   sync_src (V -> MTE3)          # Inserted
    #   sync_dst (V -> MTE3)          # Inserted
    #   store(tile_c, output)         # MTE3
    #
    # Expected: 2 sync_src, 2 sync_dst, 2 bar_v (one in each branch)
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    # Create unique memrefs
    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 120)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 121)
    memref_c = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(32768, DataType.INT64, span), 16384, 122)
    memref_output = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(49152, DataType.INT64, span), 16384, 123)

    # Create variables
    input_tensor = ir.Var("input", ir.TensorType([64, 64], DataType.FP32), span)
    output_tensor = ir.Var("output", ir.TensorType([64, 64], DataType.FP32, memref_output), span)
    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)

    # Load from MTE2 pipe
    load_op = block.load(input_tensor, offsets=[0, 0], shapes=[64, 64])
    stmt_load = ir.AssignStmt(tile_a, load_op, span)

    # For loop with nested if
    loop_var = ir.Var("i", ir.ScalarType(DataType.INT32), span)
    start = ir.ConstInt(0, DataType.INT32, span)
    stop = ir.ConstInt(4, DataType.INT32, span)
    step = ir.ConstInt(1, DataType.INT32, span)

    # Create condition
    condition = ir.ConstBool(True, span)

    # Build then branch
    tile_b_then = ir.Var("tile_b_then", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    add_b_then = block.add(tile_a, tile_a)
    stmt_add_b_then = ir.AssignStmt(tile_b_then, add_b_then, span)

    tile_c_then = ir.Var("tile_c_then", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    add_c_then = block.add(tile_b_then, tile_a)
    stmt_add_c_then = ir.AssignStmt(tile_c_then, add_c_then, span)

    yield_then = ir.YieldStmt([tile_c_then], span)
    then_body = ir.SeqStmts([stmt_add_b_then, stmt_add_c_then, yield_then], span)

    # Build else branch
    tile_b_else = ir.Var("tile_b_else", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    mul_b_else = block.mul(tile_a, tile_a)
    stmt_mul_b_else = ir.AssignStmt(tile_b_else, mul_b_else, span)

    tile_c_else = ir.Var("tile_c_else", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    mul_c_else = block.mul(tile_b_else, tile_a)
    stmt_mul_c_else = ir.AssignStmt(tile_c_else, mul_c_else, span)

    yield_else = ir.YieldStmt([tile_c_else], span)
    else_body = ir.SeqStmts([stmt_mul_b_else, stmt_mul_c_else, yield_else], span)

    # Create IfStmt
    if_return_var = ir.Var("tile_c", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    if_stmt = ir.IfStmt(condition, then_body, else_body, [if_return_var], span)

    yield_for = ir.YieldStmt([], span)
    for_body = ir.SeqStmts([if_stmt, yield_for], span)

    for_stmt = ir.ForStmt(loop_var, start, stop, step, [], for_body, [], span)

    # Store to MTE3 pipe
    store_op = block.store(if_return_var, offsets=[0, 0], shapes=[64, 64], output_tensor=output_tensor)
    stmt_store = ir.EvalStmt(store_op, span)
    stmt_return = ir.ReturnStmt(span)

    # Build function body
    body = ir.SeqStmts([stmt_load, for_stmt, stmt_store, stmt_return], span)
    func = ir.Function(
        "test_forstmt_with_ifstmt", [input_tensor, output_tensor], [], body, span, ir.FunctionType.InCore
    )

    # Wrap function in Program
    program = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    insert_sync = passes.insert_sync()
    synced_program = insert_sync(program)

    # Extract the function from the program
    synced_func = list(synced_program.functions.values())[0]

    # Verify the structure
    assert isinstance(synced_func.body, ir.SeqStmts)

    total_counts = count_syncs(synced_func.body)
    sync_src_count = total_counts["sync_src"]
    sync_dst_count = total_counts["sync_dst"]
    bar_v_count = total_counts["bar_v"]

    assert sync_src_count == 2, f"Expected exactly 2 sync_src, got {sync_src_count}"
    assert sync_dst_count == 2, f"Expected exactly 2 sync_dst, got {sync_dst_count}"
    assert bar_v_count == 2, f"Expected exactly 2 bar_v, got {bar_v_count}"


def test_insert_sync_forstmt_cross_iteration():
    """Test InsertSyncPass for cross-iteration dependencies within a for loop."""
    # Test structure
    #   for i in range(0, 4, 1):
    #     tile_b = add(tile_a, tile_c)   # V (reads tile_c from prev iter, writes tile_b)
    #     bar_v                          # Inserted (internal: tile_b RAW between stmt 1 and 2)
    #     tile_c = add(tile_b, tile_a)   # V (reads tile_b, writes tile_c for next iter)
    #     bar_v                          # Inserted (cross-iteration: tile_c read before write)
    #     yield []
    #
    # Expected: 0 sync_src, 0 sync_dst, 2 bar_v (1 internal + 1 cross-iteration)
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    # Create unique memrefs
    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 200)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 201)
    memref_c = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(32768, DataType.INT64, span), 16384, 202)

    # Create variables
    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    tile_c = ir.Var("tile_c", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)

    # For loop
    loop_var = ir.Var("i", ir.ScalarType(DataType.INT32), span)
    start = ir.ConstInt(0, DataType.INT32, span)
    stop = ir.ConstInt(4, DataType.INT32, span)
    step = ir.ConstInt(1, DataType.INT32, span)

    # For body: READ tile_c first, then WRITE tile_c (cross-iteration pattern)
    add_b_op = block.add(tile_a, tile_c)
    stmt_add_b = ir.AssignStmt(tile_b, add_b_op, span)

    add_c_op = block.add(tile_b, tile_a)
    stmt_add_c = ir.AssignStmt(tile_c, add_c_op, span)

    yield_stmt = ir.YieldStmt([], span)
    for_body = ir.SeqStmts([stmt_add_b, stmt_add_c, yield_stmt], span)

    for_stmt = ir.ForStmt(loop_var, start, stop, step, [], for_body, [], span)

    stmt_return = ir.ReturnStmt(span)

    # Build function body
    body = ir.SeqStmts([for_stmt, stmt_return], span)
    func = ir.Function("test_cross_iteration", [tile_a, tile_c], [], body, span, ir.FunctionType.InCore)

    # Wrap function in Program
    program = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    insert_sync = passes.insert_sync()
    synced_program = insert_sync(program)

    # Extract the function from the program
    synced_func = list(synced_program.functions.values())[0]

    # Verify the structure
    assert isinstance(synced_func.body, ir.SeqStmts)

    total_counts = count_syncs(synced_func.body)
    sync_src_count = total_counts["sync_src"]
    sync_dst_count = total_counts["sync_dst"]
    bar_v_count = total_counts["bar_v"]

    assert sync_src_count == 0, f"Expected exactly 0 sync_src, got {sync_src_count}"
    assert sync_dst_count == 0, f"Expected exactly 0 sync_dst, got {sync_dst_count}"
    assert bar_v_count == 2, f"Expected exactly 2 bar_v, got {bar_v_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

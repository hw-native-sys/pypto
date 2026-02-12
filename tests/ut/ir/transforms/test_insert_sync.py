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

_span = ir.Span.unknown()


def make_sync_src(set_pipe: ir.PipeType, wait_pipe: ir.PipeType, event_id: int) -> ir.EvalStmt:
    """Create EvalStmt for system.sync_src."""
    call = ir.create_op_call(
        "system.sync_src", [], {"set_pipe": set_pipe, "wait_pipe": wait_pipe, "event_id": event_id}, _span
    )
    return ir.EvalStmt(call, _span)


def make_sync_dst(set_pipe: ir.PipeType, wait_pipe: ir.PipeType, event_id: int) -> ir.EvalStmt:
    """Create EvalStmt for system.sync_dst."""
    call = ir.create_op_call(
        "system.sync_dst", [], {"set_pipe": set_pipe, "wait_pipe": wait_pipe, "event_id": event_id}, _span
    )
    return ir.EvalStmt(call, _span)


def make_bar_v() -> ir.EvalStmt:
    """Create EvalStmt for system.bar_v."""
    return ir.EvalStmt(ir.create_op_call("system.bar_v", [], {}, _span), _span)


# Shorthand aliases for PipeType
MTE1 = ir.PipeType.MTE1
MTE2 = ir.PipeType.MTE2
MTE3 = ir.PipeType.MTE3
M = ir.PipeType.M
V = ir.PipeType.V


def test_insert_sync_cross_pipe():
    """Test InsertSyncPass for cross-pipe dependencies (MTE2 -> V -> MTE3).

    Expected IR after pass:
        tile_a = load(input_a)              # MTE2
        tile_b = load(input_b)              # MTE2
        sync_src(MTE2 -> V, event=0)
        sync_dst(MTE2 -> V, event=0)
        tile_c = add(tile_a, tile_b)        # V
        sync_src(V -> MTE3, event=0)
        sync_dst(V -> MTE3, event=0)
        store(tile_c, output)               # MTE3
        return
    """
    span = _span
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 0)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 1)
    memref_c = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(32768, DataType.INT64, span), 16384, 2)

    input_a = ir.Var("input_a", ir.TensorType([64, 64], DataType.FP32), span)
    input_b = ir.Var("input_b", ir.TensorType([64, 64], DataType.FP32), span)
    output = ir.Var("output", ir.TensorType([64, 64], DataType.FP32), span)

    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    tile_c = ir.Var("tile_c", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)

    store_result = ir.Var("store_result", ir.TensorType([64, 64], DataType.FP32), span)

    # Build Before IR
    body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_a, block.load(input_a, offsets=[0, 0], shapes=[64, 64]), span),
                    ir.AssignStmt(tile_b, block.load(input_b, offsets=[0, 0], shapes=[64, 64]), span),
                    ir.AssignStmt(tile_c, block.add(tile_a, tile_b), span),
                    ir.AssignStmt(
                        store_result,
                        block.store(tile_c, offsets=[0, 0], shapes=[64, 64], output_tensor=output),
                        span,
                    ),
                ],
                span,
            ),
            ir.ReturnStmt(span),
        ],
        span,
    )
    func = ir.Function(
        "test_cross_pipe_sync", [input_a, input_b, output], [], body, span, ir.FunctionType.InCore
    )
    Before = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    After = passes.insert_sync()(Before)

    # Build Expected IR (reuse vars from Before since auto_mapping handles mapping)
    expected_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_a, block.load(input_a, offsets=[0, 0], shapes=[64, 64]), span),
                    ir.AssignStmt(tile_b, block.load(input_b, offsets=[0, 0], shapes=[64, 64]), span),
                    make_sync_src(MTE2, V, 0),
                    make_sync_dst(MTE2, V, 0),
                    ir.AssignStmt(tile_c, block.add(tile_a, tile_b), span),
                    make_sync_src(V, MTE3, 0),
                    make_sync_dst(V, MTE3, 0),
                    ir.AssignStmt(
                        store_result,
                        block.store(tile_c, offsets=[0, 0], shapes=[64, 64], output_tensor=output),
                        span,
                    ),
                ],
                span,
            ),
            ir.ReturnStmt(span),
        ],
        span,
    )
    expected_func = ir.Function(
        "test_cross_pipe_sync", [input_a, input_b, output], [], expected_body, span, ir.FunctionType.InCore
    )
    Expected = ir.Program([expected_func], "test_program", span)

    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_insert_sync_intra_pipe():
    """Test InsertSyncPass for intra-pipe dependencies (V -> V).

    Expected IR after pass:
        t_c = add(t_a, t_b)    # V
        bar_v                   # Inserted
        t_d = add(t_c, t_a)    # V (depends on t_c)
        return t_d
    """
    span = _span
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 3)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 4)
    memref_c = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(32768, DataType.INT64, span), 16384, 5)
    memref_d = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(49152, DataType.INT64, span), 16384, 6)

    t_a = ir.Var("t_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    t_b = ir.Var("t_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    t_c = ir.Var("t_c", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    t_d = ir.Var("t_d", ir.TileType([dim64, dim64], DataType.FP32, memref_d), span)

    # Build Before IR
    body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(t_c, block.add(t_a, t_b), span),
                    ir.AssignStmt(t_d, block.add(t_c, t_a), span),
                ],
                span,
            ),
            ir.ReturnStmt([t_d], span),
        ],
        span,
    )
    func = ir.Function("test_intra_pipe_sync", [t_a, t_b], [t_d.type], body, span)
    Before = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    After = passes.insert_sync()(Before)

    # Build Expected IR
    expected_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(t_c, block.add(t_a, t_b), span),
                    make_bar_v(),
                    ir.AssignStmt(t_d, block.add(t_c, t_a), span),
                ],
                span,
            ),
            ir.ReturnStmt([t_d], span),
        ],
        span,
    )
    expected_func = ir.Function("test_intra_pipe_sync", [t_a, t_b], [t_d.type], expected_body, span)
    Expected = ir.Program([expected_func], "test_program", span)

    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_insert_sync_cube_pipe():
    """Test InsertSyncPass for CUBE (M pipe) operations: load-move-matmul-store.

    Expected IR after pass:
        tile_a = load(input_a)                          # MTE2
        sync_src(MTE2 -> MTE1, event=0)
        tile_b = load(input_b)                          # MTE2
        sync_src(MTE2 -> MTE1, event=1)
        sync_dst(MTE2 -> MTE1, event=0)
        tile_a_cube = move(tile_a)                      # MTE1
        sync_dst(MTE2 -> MTE1, event=1)
        tile_b_cube = move(tile_b)                      # MTE1
        sync_src(MTE1 -> M, event=0)
        sync_dst(MTE1 -> M, event=0)
        tile_c = matmul(tile_a_cube, tile_b_cube)       # CUBE/M
        sync_src(M -> MTE3, event=0)
        sync_dst(M -> MTE3, event=0)
        store(tile_c, output)                           # MTE3
        return
    """
    span = _span
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    memref_a_l1 = ir.MemRef(ir.MemorySpace.L1, ir.ConstInt(0, DataType.INT64, span), 16384, 100)
    memref_b_l1 = ir.MemRef(ir.MemorySpace.L1, ir.ConstInt(16384, DataType.INT64, span), 16384, 101)
    memref_a_l0a = ir.MemRef(ir.MemorySpace.L0A, ir.ConstInt(0, DataType.INT64, span), 16384, 102)
    memref_b_l0b = ir.MemRef(ir.MemorySpace.L0B, ir.ConstInt(0, DataType.INT64, span), 16384, 103)
    memref_c_l0c = ir.MemRef(ir.MemorySpace.L0C, ir.ConstInt(0, DataType.INT64, span), 16384, 104)

    input_a = ir.Var("input_a", ir.TensorType([64, 64], DataType.FP16), span)
    input_b = ir.Var("input_b", ir.TensorType([64, 64], DataType.FP16), span)
    output = ir.Var("output", ir.TensorType([64, 64], DataType.FP32), span)

    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP16, memref_a_l1), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP16, memref_b_l1), span)
    tile_a_cube = ir.Var("tile_a_cube", ir.TileType([dim64, dim64], DataType.FP16, memref_a_l0a), span)
    tile_b_cube = ir.Var("tile_b_cube", ir.TileType([dim64, dim64], DataType.FP16, memref_b_l0b), span)
    tile_c = ir.Var("tile_c", ir.TileType([dim64, dim64], DataType.FP32, memref_c_l0c), span)

    load_a = block.load(input_a, offsets=[0, 0], shapes=[64, 64])
    load_b = block.load(input_b, offsets=[0, 0], shapes=[64, 64])
    move_a = block.move(tile_a, target_memory=ir.MemorySpace.L0A)
    move_b = block.move(tile_b, target_memory=ir.MemorySpace.L0B)
    matmul_op = block.matmul(tile_a_cube, tile_b_cube)
    store_op = block.store(tile_c, offsets=[0, 0], shapes=[64, 64], output_tensor=output)

    store_result = ir.Var("store_result", ir.TensorType([64, 64], DataType.FP32), span)

    # Build Before IR
    body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_a, load_a, span),
                    ir.AssignStmt(tile_b, load_b, span),
                    ir.AssignStmt(tile_a_cube, move_a, span),
                    ir.AssignStmt(tile_b_cube, move_b, span),
                    ir.AssignStmt(tile_c, matmul_op, span),
                    ir.AssignStmt(store_result, store_op, span),
                ],
                span,
            ),
            ir.ReturnStmt(span),
        ],
        span,
    )
    func = ir.Function("test_cube_sync", [input_a, input_b, output], [], body, span, ir.FunctionType.InCore)
    Before = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    After = passes.insert_sync()(Before)

    # Build Expected IR
    expected_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_a, block.load(input_a, offsets=[0, 0], shapes=[64, 64]), span),
                    make_sync_src(MTE2, MTE1, 0),
                    ir.AssignStmt(tile_b, block.load(input_b, offsets=[0, 0], shapes=[64, 64]), span),
                    make_sync_src(MTE2, MTE1, 1),
                    make_sync_dst(MTE2, MTE1, 0),
                    ir.AssignStmt(tile_a_cube, block.move(tile_a, target_memory=ir.MemorySpace.L0A), span),
                    make_sync_dst(MTE2, MTE1, 1),
                    ir.AssignStmt(tile_b_cube, block.move(tile_b, target_memory=ir.MemorySpace.L0B), span),
                    make_sync_src(MTE1, M, 0),
                    make_sync_dst(MTE1, M, 0),
                    ir.AssignStmt(tile_c, block.matmul(tile_a_cube, tile_b_cube), span),
                    make_sync_src(M, MTE3, 0),
                    make_sync_dst(M, MTE3, 0),
                    ir.AssignStmt(
                        store_result,
                        block.store(tile_c, offsets=[0, 0], shapes=[64, 64], output_tensor=output),
                        span,
                    ),
                ],
                span,
            ),
            ir.ReturnStmt(span),
        ],
        span,
    )
    expected_func = ir.Function(
        "test_cube_sync", [input_a, input_b, output], [], expected_body, span, ir.FunctionType.InCore
    )
    Expected = ir.Program([expected_func], "test_program", span)

    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_if_both_branches():
    """Test InsertSyncPass when both if branches depend on a statement before if.

    Expected IR after pass:
        tile_a = load(input)            # MTE2
        sync_src(MTE2 -> V, event=0)
        sync_dst(MTE2 -> V, event=0)
        if (cond):
          then:
            tile_b = add(tile_a, tile_a)
            yield [tile_b]
          else:
            tile_c = mul(tile_a, tile_a)
            yield [tile_c]
        return
    """
    span = _span
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 100)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 101)
    memref_c = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(32768, DataType.INT64, span), 16384, 102)

    input_tensor = ir.Var("input", ir.TensorType([64, 64], DataType.FP32), span)
    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    tile_c = ir.Var("tile_c", ir.TileType([dim64, dim64], DataType.FP32, memref_c), span)
    condition = ir.ConstBool(True, span)
    if_return_var = ir.Var("result", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)

    # Build Before IR
    then_body = ir.SeqStmts(
        [
            ir.OpStmts([ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span)], span),
            ir.YieldStmt([tile_b], span),
        ],
        span,
    )
    else_body = ir.SeqStmts(
        [
            ir.OpStmts([ir.AssignStmt(tile_c, block.mul(tile_a, tile_a), span)], span),
            ir.YieldStmt([tile_c], span),
        ],
        span,
    )
    body = ir.SeqStmts(
        [
            ir.OpStmts(
                [ir.AssignStmt(tile_a, block.load(input_tensor, offsets=[0, 0], shapes=[64, 64]), span)],
                span,
            ),
            ir.IfStmt(condition, then_body, else_body, [if_return_var], span),
            ir.ReturnStmt(span),
        ],
        span,
    )
    func = ir.Function("test_if_both_branches", [input_tensor], [], body, span, ir.FunctionType.InCore)
    Before = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    After = passes.insert_sync()(Before)

    # Build Expected IR: sync_dst merged into same OpStmts as load + sync_src
    expected_then = ir.SeqStmts(
        [
            ir.OpStmts([ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span)], span),
            ir.YieldStmt([tile_b], span),
        ],
        span,
    )
    expected_else = ir.SeqStmts(
        [
            ir.OpStmts([ir.AssignStmt(tile_c, block.mul(tile_a, tile_a), span)], span),
            ir.YieldStmt([tile_c], span),
        ],
        span,
    )
    expected_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_a, block.load(input_tensor, offsets=[0, 0], shapes=[64, 64]), span),
                    make_sync_src(MTE2, V, 0),
                    make_sync_dst(MTE2, V, 0),
                ],
                span,
            ),
            ir.IfStmt(condition, expected_then, expected_else, [if_return_var], span),
            ir.ReturnStmt(span),
        ],
        span,
    )
    expected_func = ir.Function(
        "test_if_both_branches", [input_tensor], [], expected_body, span, ir.FunctionType.InCore
    )
    Expected = ir.Program([expected_func], "test_program", span)

    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_if_one_branch():
    """Test InsertSyncPass when only one if branch depends on a statement before if.

    Expected IR after pass:
        tile_a = load(input)            # MTE2
        sync_src(MTE2 -> V, event=0)
        sync_dst(MTE2 -> V, event=0)
        if (cond):
          then:
            tile_b = add(tile_a, tile_a)
            yield [tile_b]
          else:
            yield [tile_a]
        return
    """
    span = _span
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 100)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 101)

    input_tensor = ir.Var("input", ir.TensorType([64, 64], DataType.FP32), span)
    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    condition = ir.ConstBool(True, span)

    # Build Before IR
    then_body = ir.SeqStmts(
        [
            ir.OpStmts([ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span)], span),
            ir.YieldStmt([tile_b], span),
        ],
        span,
    )
    else_body = ir.SeqStmts([ir.YieldStmt([tile_a], span)], span)
    body = ir.SeqStmts(
        [
            ir.OpStmts(
                [ir.AssignStmt(tile_a, block.load(input_tensor, offsets=[0, 0], shapes=[64, 64]), span)],
                span,
            ),
            ir.IfStmt(condition, then_body, else_body, [], span),
            ir.ReturnStmt(span),
        ],
        span,
    )
    func = ir.Function("test_one_branch", [input_tensor], [], body, span, ir.FunctionType.InCore)
    Before = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    After = passes.insert_sync()(Before)

    # Build Expected IR
    expected_then = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span),
                ],
                span,
            ),
            ir.YieldStmt([tile_b], span),
        ],
        span,
    )
    expected_else = ir.SeqStmts(
        [ir.YieldStmt([tile_a], span)],
        span,
    )
    expected_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_a, block.load(input_tensor, offsets=[0, 0], shapes=[64, 64]), span),
                    make_sync_src(MTE2, V, 0),
                    make_sync_dst(MTE2, V, 0),
                ],
                span,
            ),
            ir.IfStmt(condition, expected_then, expected_else, [], span),
            ir.ReturnStmt(span),
        ],
        span,
    )
    expected_func = ir.Function(
        "test_one_branch", [input_tensor], [], expected_body, span, ir.FunctionType.InCore
    )
    Expected = ir.Program([expected_func], "test_program", span)

    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_branch_merge():
    """Test InsertSyncPass when if branches merge and result is used after if.

    Expected IR after pass:
        tile_a = load(input)                # MTE2
        sync_src(MTE2 -> V, event=0)
        sync_dst(MTE2 -> V, event=0)
        if (cond):
          then:
            tile_b = add(tile_a, tile_a)    # V
            yield [tile_b]
          else:
            tile_b = mul(tile_a, tile_a)    # V
            yield [tile_b]
        sync_src(V -> MTE3, event=0)
        sync_dst(V -> MTE3, event=0)
        store(tile_b, output)               # MTE3
        return
    """
    span = _span
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 200)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 201)

    input_tensor = ir.Var("input", ir.TensorType([64, 64], DataType.FP32), span)
    output_tensor = ir.Var("output", ir.TensorType([64, 64], DataType.FP32), span)
    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    condition = ir.ConstBool(True, span)
    store_result = ir.Var("store_result", ir.TensorType([64, 64], DataType.FP32), span)

    # Build Before IR (test_branch_merge)
    then_body = ir.SeqStmts(
        [
            ir.OpStmts([ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span)], span),
            ir.YieldStmt([tile_b], span),
        ],
        span,
    )
    else_body = ir.SeqStmts(
        [
            ir.OpStmts([ir.AssignStmt(tile_b, block.mul(tile_a, tile_a), span)], span),
            ir.YieldStmt([tile_b], span),
        ],
        span,
    )
    body = ir.SeqStmts(
        [
            ir.OpStmts(
                [ir.AssignStmt(tile_a, block.load(input_tensor, offsets=[0, 0], shapes=[64, 64]), span)],
                span,
            ),
            ir.IfStmt(condition, then_body, else_body, [], span),
            ir.OpStmts(
                [
                    ir.AssignStmt(
                        store_result,
                        block.store(tile_b, offsets=[0, 0], shapes=[64, 64], output_tensor=output_tensor),
                        span,
                    ),
                ],
                span,
            ),
            ir.ReturnStmt(span),
        ],
        span,
    )
    func = ir.Function(
        "test_branch_merge", [input_tensor, output_tensor], [], body, span, ir.FunctionType.InCore
    )
    Before = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    After = passes.insert_sync()(Before)

    # Build Expected IR
    expected_then = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span),
                ],
                span,
            ),
            ir.YieldStmt([tile_b], span),
        ],
        span,
    )
    expected_else = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_b, block.mul(tile_a, tile_a), span),
                ],
                span,
            ),
            ir.YieldStmt([tile_b], span),
        ],
        span,
    )
    expected_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_a, block.load(input_tensor, offsets=[0, 0], shapes=[64, 64]), span),
                    make_sync_src(MTE2, V, 0),
                    make_sync_dst(MTE2, V, 0),
                ],
                span,
            ),
            ir.IfStmt(condition, expected_then, expected_else, [], span),
            ir.OpStmts(
                [
                    make_sync_src(V, MTE3, 0),
                    make_sync_dst(V, MTE3, 0),
                    ir.AssignStmt(
                        store_result,
                        block.store(tile_b, offsets=[0, 0], shapes=[64, 64], output_tensor=output_tensor),
                        span,
                    ),
                ],
                span,
            ),
            ir.ReturnStmt(span),
        ],
        span,
    )
    expected_func = ir.Function(
        "test_branch_merge", [input_tensor, output_tensor], [], expected_body, span, ir.FunctionType.InCore
    )
    Expected = ir.Program([expected_func], "test_program", span)

    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_for_loop():
    """Test InsertSyncPass for simple for loop: load before for, add inside for, store after for.

    Expected IR after pass:
        tile_a = load(input)                    # MTE2
        sync_src(MTE2 -> V, event=0)
        sync_dst(MTE2 -> V, event=0)
        for i in range(0, 4, 1):
            tile_b = add(tile_a, tile_a)        # V
            bar_v                                # cross-iteration at end
            yield []
        sync_src(V -> MTE3, event=0)
        sync_dst(V -> MTE3, event=0)
        store(tile_b, output)                   # MTE3
        return
    """
    span = _span
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 50)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 51)

    input_tensor = ir.Var("input", ir.TensorType([64, 64], DataType.FP32), span)
    output_tensor = ir.Var("output", ir.TensorType([64, 64], DataType.FP32), span)
    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    store_result = ir.Var("store_result", ir.TensorType([64, 64], DataType.FP32), span)

    loop_var = ir.Var("i", ir.ScalarType(DataType.INT32), span)
    start = ir.ConstInt(0, DataType.INT32, span)
    stop = ir.ConstInt(4, DataType.INT32, span)
    step = ir.ConstInt(1, DataType.INT32, span)

    # Build Before IR
    for_body = ir.SeqStmts(
        [
            ir.OpStmts([ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span)], span),
            ir.YieldStmt([], span),
        ],
        span,
    )
    body = ir.SeqStmts(
        [
            ir.OpStmts(
                [ir.AssignStmt(tile_a, block.load(input_tensor, offsets=[0, 0], shapes=[64, 64]), span)],
                span,
            ),
            ir.ForStmt(loop_var, start, stop, step, [], for_body, [], span),
            ir.OpStmts(
                [
                    ir.AssignStmt(
                        store_result,
                        block.store(tile_b, offsets=[0, 0], shapes=[64, 64], output_tensor=output_tensor),
                        span,
                    ),
                ],
                span,
            ),
            ir.ReturnStmt(span),
        ],
        span,
    )
    func = ir.Function(
        "test_for_simple", [input_tensor, output_tensor], [], body, span, ir.FunctionType.InCore
    )
    Before = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    After = passes.insert_sync()(Before)

    # Build Expected IR
    expected_for_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span),
                    make_bar_v(),
                ],
                span,
            ),
            ir.YieldStmt([], span),
        ],
        span,
    )
    expected_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_a, block.load(input_tensor, offsets=[0, 0], shapes=[64, 64]), span),
                    make_sync_src(MTE2, V, 0),
                    make_sync_dst(MTE2, V, 0),
                ],
                span,
            ),
            ir.ForStmt(loop_var, start, stop, step, [], expected_for_body, [], span),
            ir.OpStmts(
                [
                    make_sync_src(V, MTE3, 0),
                    make_sync_dst(V, MTE3, 0),
                    ir.AssignStmt(
                        store_result,
                        block.store(tile_b, offsets=[0, 0], shapes=[64, 64], output_tensor=output_tensor),
                        span,
                    ),
                ],
                span,
            ),
            ir.ReturnStmt(span),
        ],
        span,
    )
    expected_func = ir.Function(
        "test_for_simple", [input_tensor, output_tensor], [], expected_body, span, ir.FunctionType.InCore
    )
    Expected = ir.Program([expected_func], "test_program", span)

    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_for_cross_iteration():
    """Test InsertSyncPass for cross-iteration dependencies within a for loop.

    Expected IR after pass:
        for i in range(0, 4, 1):
            tile_a = load(input)                # MTE2
            sync_src(MTE2 -> V, event=0)
            sync_dst(MTE2 -> V, event=0)
            tile_b = add(tile_a, tile_a)        # V
            bar_v
            tile_a = mul(tile_b, tile_b)        # V (WAW with next iter's load)
            sync_src(V -> MTE2, event=0)
            sync_dst(V -> MTE2, event=0)
            yield []
        return
    """
    span = _span
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 300)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 301)

    input_tensor = ir.Var("input", ir.TensorType([64, 64], DataType.FP32), span)
    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)

    loop_var = ir.Var("i", ir.ScalarType(DataType.INT32), span)
    start = ir.ConstInt(0, DataType.INT32, span)
    stop = ir.ConstInt(4, DataType.INT32, span)
    step = ir.ConstInt(1, DataType.INT32, span)

    # Build Before IR
    for_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_a, block.load(input_tensor, offsets=[0, 0], shapes=[64, 64]), span),
                    ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span),
                    ir.AssignStmt(tile_a, block.mul(tile_b, tile_b), span),
                ],
                span,
            ),
            ir.YieldStmt([], span),
        ],
        span,
    )
    body = ir.SeqStmts(
        [ir.ForStmt(loop_var, start, stop, step, [], for_body, [], span), ir.ReturnStmt(span)], span
    )
    func = ir.Function("test_cross_iteration", [input_tensor], [], body, span, ir.FunctionType.InCore)
    Before = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    After = passes.insert_sync()(Before)

    # Build Expected IR
    expected_for_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_a, block.load(input_tensor, offsets=[0, 0], shapes=[64, 64]), span),
                    make_sync_src(MTE2, V, 0),
                    make_sync_dst(MTE2, V, 0),
                    ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span),
                    make_bar_v(),
                    ir.AssignStmt(tile_a, block.mul(tile_b, tile_b), span),
                    make_sync_src(V, MTE2, 0),
                    make_sync_dst(V, MTE2, 0),
                ],
                span,
            ),
            ir.YieldStmt([], span),
        ],
        span,
    )
    expected_body = ir.SeqStmts(
        [
            ir.ForStmt(loop_var, start, stop, step, [], expected_for_body, [], span),
            ir.ReturnStmt(span),
        ],
        span,
    )
    expected_func = ir.Function(
        "test_cross_iteration", [input_tensor], [], expected_body, span, ir.FunctionType.InCore
    )
    Expected = ir.Program([expected_func], "test_program", span)

    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_for_cross_iteration_mte3_to_mte2():
    """Test InsertSyncPass for cross-iteration dependencies with load-add-store pattern.

    Expected IR after pass:
        for i in range(0, 4, 1):
            tile_a = load(data)                 # MTE2
            sync_src(MTE2 -> V, event=0)
            sync_dst(MTE2 -> V, event=0)
            tile_b = add(tile_a, tile_a)        # V
            sync_src(V -> MTE3, event=0)
            sync_dst(V -> MTE3, event=0)
            store(tile_b, data)                 # MTE3
            sync_src(MTE3 -> MTE2, event=0)
            sync_dst(MTE3 -> MTE2, event=0)
            yield []
        return
    """
    span = _span
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 600)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 601)
    memref_data = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 16384, 602)

    data_tensor = ir.Var("data", ir.TensorType([64, 64], DataType.FP32, memref_data), span)
    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    store_result = ir.Var("store_result", ir.TensorType([64, 64], DataType.FP32, memref_data), span)

    loop_var = ir.Var("i", ir.ScalarType(DataType.INT32), span)
    start = ir.ConstInt(0, DataType.INT32, span)
    stop = ir.ConstInt(4, DataType.INT32, span)
    step = ir.ConstInt(1, DataType.INT32, span)

    # Build Before IR
    for_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_a, block.load(data_tensor, offsets=[0, 0], shapes=[64, 64]), span),
                    ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span),
                    ir.AssignStmt(
                        store_result,
                        block.store(tile_b, offsets=[0, 0], shapes=[64, 64], output_tensor=data_tensor),
                        span,
                    ),
                ],
                span,
            ),
            ir.YieldStmt([], span),
        ],
        span,
    )
    body = ir.SeqStmts(
        [ir.ForStmt(loop_var, start, stop, step, [], for_body, [], span), ir.ReturnStmt(span)], span
    )
    func = ir.Function("test_for_mte3_to_mte2", [data_tensor], [], body, span, ir.FunctionType.InCore)
    Before = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    After = passes.insert_sync()(Before)

    # Build Expected IR
    expected_for_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_a, block.load(data_tensor, offsets=[0, 0], shapes=[64, 64]), span),
                    make_sync_src(MTE2, V, 0),
                    make_sync_dst(MTE2, V, 0),
                    ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span),
                    make_sync_src(V, MTE3, 0),
                    make_sync_dst(V, MTE3, 0),
                    ir.AssignStmt(
                        store_result,
                        block.store(tile_b, offsets=[0, 0], shapes=[64, 64], output_tensor=data_tensor),
                        span,
                    ),
                    make_sync_src(MTE3, MTE2, 0),
                    make_sync_dst(MTE3, MTE2, 0),
                ],
                span,
            ),
            ir.YieldStmt([], span),
        ],
        span,
    )
    expected_body = ir.SeqStmts(
        [
            ir.ForStmt(loop_var, start, stop, step, [], expected_for_body, [], span),
            ir.ReturnStmt(span),
        ],
        span,
    )
    expected_func = ir.Function(
        "test_for_mte3_to_mte2", [data_tensor], [], expected_body, span, ir.FunctionType.InCore
    )
    Expected = ir.Program([expected_func], "test_program", span)

    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_for_with_if_branches():
    """Test InsertSyncPass for for loop with if branches (load->if->store pattern).

    Expected IR after pass:
        for i in range(0, 4, 1):
            tile_a = load(data)                     # MTE2
            sync_src(MTE2 -> V, event=0)
            sync_dst(MTE2 -> V, event=0)
            if (cond):
              then:
                tile_b = add(tile_a, tile_a)        # V
                yield [tile_b]
              else:
                tile_b = mul(tile_a, tile_a)        # V
                yield [tile_b]
            sync_src(V -> MTE3, event=0)
            sync_dst(V -> MTE3, event=0)
            store(tile_b, data)                     # MTE3
            sync_src(MTE3 -> MTE2, event=0)
            sync_dst(MTE3 -> MTE2, event=0)
            yield []
        return
    """
    span = _span
    dim64 = ir.ConstInt(64, DataType.INT64, span)

    memref_a = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16384, 500)
    memref_b = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(16384, DataType.INT64, span), 16384, 501)
    memref_data = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 16384, 502)

    data_tensor = ir.Var("data", ir.TensorType([64, 64], DataType.FP32, memref_data), span)
    tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_a), span)
    tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_b), span)
    store_result = ir.Var("store_result", ir.TensorType([64, 64], DataType.FP32, memref_data), span)
    condition = ir.ConstBool(True, span)

    loop_var = ir.Var("i", ir.ScalarType(DataType.INT32), span)
    start = ir.ConstInt(0, DataType.INT32, span)
    stop = ir.ConstInt(4, DataType.INT32, span)
    step = ir.ConstInt(1, DataType.INT32, span)

    # Build Before IR
    then_body = ir.SeqStmts(
        [
            ir.OpStmts([ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span)], span),
            ir.YieldStmt([tile_b], span),
        ],
        span,
    )
    else_body = ir.SeqStmts(
        [
            ir.OpStmts([ir.AssignStmt(tile_b, block.mul(tile_a, tile_a), span)], span),
            ir.YieldStmt([tile_b], span),
        ],
        span,
    )
    for_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [ir.AssignStmt(tile_a, block.load(data_tensor, offsets=[0, 0], shapes=[64, 64]), span)],
                span,
            ),
            ir.IfStmt(condition, then_body, else_body, [], span),
            ir.OpStmts(
                [
                    ir.AssignStmt(
                        store_result,
                        block.store(tile_b, offsets=[0, 0], shapes=[64, 64], output_tensor=data_tensor),
                        span,
                    ),
                ],
                span,
            ),
            ir.YieldStmt([], span),
        ],
        span,
    )
    body = ir.SeqStmts(
        [ir.ForStmt(loop_var, start, stop, step, [], for_body, [], span), ir.ReturnStmt(span)], span
    )
    func = ir.Function("test_for_if_branches", [data_tensor], [], body, span, ir.FunctionType.InCore)
    Before = ir.Program([func], "test_program", span)

    # Run InsertSyncPass
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    After = passes.insert_sync()(Before)

    # Build Expected IR
    expected_then = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_b, block.add(tile_a, tile_a), span),
                ],
                span,
            ),
            ir.YieldStmt([tile_b], span),
        ],
        span,
    )
    expected_else = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_b, block.mul(tile_a, tile_a), span),
                ],
                span,
            ),
            ir.YieldStmt([tile_b], span),
        ],
        span,
    )
    expected_for_body = ir.SeqStmts(
        [
            ir.OpStmts(
                [
                    ir.AssignStmt(tile_a, block.load(data_tensor, offsets=[0, 0], shapes=[64, 64]), span),
                    make_sync_src(MTE2, V, 0),
                    make_sync_dst(MTE2, V, 0),
                ],
                span,
            ),
            ir.IfStmt(condition, expected_then, expected_else, [], span),
            ir.OpStmts(
                [
                    make_sync_src(V, MTE3, 0),
                    make_sync_dst(V, MTE3, 0),
                    ir.AssignStmt(
                        store_result,
                        block.store(tile_b, offsets=[0, 0], shapes=[64, 64], output_tensor=data_tensor),
                        span,
                    ),
                    make_sync_src(MTE3, MTE2, 0),
                    make_sync_dst(MTE3, MTE2, 0),
                ],
                span,
            ),
            ir.YieldStmt([], span),
        ],
        span,
    )
    expected_body = ir.SeqStmts(
        [
            ir.ForStmt(loop_var, start, stop, step, [], expected_for_body, [], span),
            ir.ReturnStmt(span),
        ],
        span,
    )
    expected_func = ir.Function(
        "test_for_if_branches", [data_tensor], [], expected_body, span, ir.FunctionType.InCore
    )
    Expected = ir.Program([expected_func], "test_program", span)

    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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


def test_insert_sync_cross_pipe():
    """Test InsertSyncPass for cross-pipe dependencies (MTE2 -> V -> MTE3)."""
    ib = builder.IRBuilder()

    with ib.function("test_sync") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))

        # MTE2
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 64, 64))

        # VECTOR (Depends on MTE2)
        tile_c = ib.let("tile_c", block.add(tile_a, tile_b))

        # MTE3 (Depends on VECTOR)
        res = ib.let("res", block.store(tile_c, 0, 0, 64, 64, output))
        ib.return_stmt(res)

    func = f.get_result()

    # Run passes
    # 1. InitMemRefPass (required for InsertSyncPass to see memrefs)
    init_memref = passes.InitMemRefPass()
    func = init_memref.run(func)

    # 2. InsertSyncPass
    insert_sync = passes.InsertSyncPass()
    synced_func = insert_sync.run(func)

    # Verify sync ops are inserted
    assert isinstance(synced_func.body, ir.SeqStmts)
    stmts = synced_func.body.stmts

    # Expected sequence roughly:
    # load a
    # load b
    # sync_src (MTE2 -> V)
    # sync_dst (MTE2 -> V)
    # add
    # sync_src (V -> MTE3)
    # sync_dst (V -> MTE3)
    # store

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

    # Two sync pairs for MTE2->V and one for V->MTE3 are expected.
    assert sync_src_count == 3
    assert sync_dst_count == 3


def test_insert_sync_intra_pipe():
    """Test InsertSyncPass for intra-pipe dependencies (V -> V)."""
    ib = builder.IRBuilder()

    with ib.function("test_sync_intra") as f:
        t_a = f.param("t_a", ir.TileType([64, 64], DataType.FP32))
        t_b = f.param("t_b", ir.TileType([64, 64], DataType.FP32))

        # V
        t_c = ib.let("t_c", block.add(t_a, t_b))
        # V (Depends on previous V)
        t_d = ib.let("t_d", block.add(t_c, t_a))

        ib.return_stmt(t_d)

    func = f.get_result()

    # Run InitMemRefPass
    init_memref = passes.InitMemRefPass()
    func = init_memref.run(func)

    # Run InsertSyncPass
    insert_sync = passes.InsertSyncPass()
    synced_func = insert_sync.run(func)

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


if __name__ == "__main__":
    pytest.main([__file__])
